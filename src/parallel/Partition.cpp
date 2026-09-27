// SPDX-License-Identifier: LGPL-2.1-or-later
#include "dualmesh/parallel/Partition.h"

#include "dualmesh/core/InputParameters.h"
#include "dualmesh/fe/ReferenceElement.h"

#include <algorithm>
#include <cmath>
#include <deque>
#include <iostream>
#include <map>
#include <numeric>
#include <set>

#ifdef DUALMESH_HAVE_METIS
#include <metis.h>
#endif

namespace dualmesh
{

bool
haveMetis()
{
#ifdef DUALMESH_HAVE_METIS
  return true;
#else
  return false;
#endif
}

std::vector<std::vector<Index>>
elementAdjacency(const Mesh & mesh)
{
  // Map every face, identified by its sorted node list, to the elements that
  // own it.  A face shared by two elements makes them neighbours.
  std::map<std::vector<Index>, std::vector<Index>> faces;
  for (Index e = 0; e < mesh.numElements(); ++e)
  {
    const auto & ref = ReferenceElement::get(mesh.element(e).type);
    for (int s = 0; s < ref.numSides(); ++s)
    {
      std::vector<Index> key = mesh.sideNodes({e, s});
      std::sort(key.begin(), key.end());
      faces[key].push_back(e);
    }
  }
  std::vector<std::vector<Index>> adjacency(mesh.numElements());
  for (const auto & [key, owners] : faces)
  {
    (void) key;
    for (std::size_t i = 0; i < owners.size(); ++i)
      for (std::size_t j = i + 1; j < owners.size(); ++j)
      {
        adjacency[owners[i]].push_back(owners[j]);
        adjacency[owners[j]].push_back(owners[i]);
      }
  }
  for (auto & row : adjacency)
  {
    std::sort(row.begin(), row.end());
    row.erase(std::unique(row.begin(), row.end()), row.end());
  }
  return adjacency;
}

namespace
{
/// Recursive coordinate bisection: split the elements listed in @p indices
/// into @p parts parts, writing the part of each element into @p part.
void
bisect(const Mesh & mesh,
       std::vector<Index> & indices,
       int first_part,
       int parts,
       std::vector<int> & part)
{
  if (parts <= 1)
  {
    for (Index e : indices)
      part[e] = first_part;
    return;
  }
  // Longest axis of the bounding box of the centroids.
  Point lo{1e300, 1e300, 1e300}, hi{-1e300, -1e300, -1e300};
  std::vector<Point> centroid(indices.size());
  for (std::size_t i = 0; i < indices.size(); ++i)
  {
    centroid[i] = mesh.elementCentroid(indices[i]);
    for (int d = 0; d < 3; ++d)
    {
      lo[d] = std::min(lo[d], centroid[i][d]);
      hi[d] = std::max(hi[d], centroid[i][d]);
    }
  }
  int axis = 0;
  for (int d = 1; d < mesh.dimension(); ++d)
    if (hi[d] - lo[d] > hi[axis] - lo[axis])
      axis = d;
  // Split so that the two halves get shares proportional to their part counts.
  const int left_parts = parts / 2;
  const std::size_t left_size = static_cast<std::size_t>(
      std::llround(static_cast<double>(indices.size()) * left_parts / parts));
  std::vector<std::size_t> order(indices.size());
  std::iota(order.begin(), order.end(), 0);
  std::nth_element(order.begin(),
                   order.begin() + static_cast<long>(left_size),
                   order.end(),
                   [&](std::size_t a, std::size_t b)
                   { return centroid[a][axis] < centroid[b][axis]; });
  std::vector<Index> left, right;
  left.reserve(left_size);
  right.reserve(indices.size() - left_size);
  for (std::size_t k = 0; k < order.size(); ++k)
    (k < left_size ? left : right).push_back(indices[order[k]]);
  bisect(mesh, left, first_part, left_parts, part);
  bisect(mesh, right, first_part + left_parts, parts - left_parts, part);
}

/// Grow each part outward from a seed element through the face connectivity,
/// which keeps the parts connected and cuts fewer faces than a geometric
/// split on an unstructured mesh.
std::vector<int>
graphGrow(const Mesh & mesh, int num_parts)
{
  const Index ne = mesh.numElements();
  const auto adjacency = elementAdjacency(mesh);
  std::vector<int> part(ne, -1);
  const Index target = (ne + num_parts - 1) / num_parts;
  Index assigned = 0;
  Index seed = 0;
  for (int p = 0; p < num_parts && assigned < ne; ++p)
  {
    // The seed of a new part is the unassigned element farthest from the
    // already assigned ones, approximated by the first unassigned element
    // with the fewest unassigned neighbours (a corner of what is left).
    Index best = -1;
    int best_degree = 1 << 30;
    for (Index e = seed; e < ne; ++e)
    {
      if (part[e] >= 0)
        continue;
      int degree = 0;
      for (Index n : adjacency[e])
        if (part[n] < 0)
          ++degree;
      if (degree < best_degree)
      {
        best_degree = degree;
        best = e;
        if (degree <= 1)
          break;
      }
    }
    if (best < 0)
      break;
    seed = best;
    const Index quota = (p == num_parts - 1) ? ne : target;
    std::deque<Index> queue{best};
    Index count = 0;
    while (!queue.empty() && count < quota)
    {
      const Index e = queue.front();
      queue.pop_front();
      if (part[e] >= 0)
        continue;
      part[e] = p;
      ++count;
      ++assigned;
      for (Index n : adjacency[e])
        if (part[n] < 0)
          queue.push_back(n);
    }
  }
  // Anything left (a disconnected piece) goes to the smallest part.
  std::vector<Index> sizes(num_parts, 0);
  for (Index e = 0; e < ne; ++e)
    if (part[e] >= 0)
      ++sizes[part[e]];
  for (Index e = 0; e < ne; ++e)
    if (part[e] < 0)
    {
      const int p = static_cast<int>(std::min_element(sizes.begin(), sizes.end()) - sizes.begin());
      part[e] = p;
      ++sizes[p];
    }
  return part;
}

#ifdef DUALMESH_HAVE_METIS
std::vector<int>
metisPartition(const Mesh & mesh, int num_parts)
{
  const auto adjacency = elementAdjacency(mesh);
  const idx_t nvtxs = static_cast<idx_t>(mesh.numElements());
  std::vector<idx_t> xadj(nvtxs + 1, 0), adjncy;
  for (idx_t i = 0; i < nvtxs; ++i)
  {
    xadj[i + 1] = xadj[i] + static_cast<idx_t>(adjacency[i].size());
    for (Index n : adjacency[i])
      adjncy.push_back(static_cast<idx_t>(n));
  }
  idx_t ncon = 1, nparts = num_parts, objval = 0;
  std::vector<idx_t> part(nvtxs, 0);
  idx_t options[METIS_NOPTIONS];
  METIS_SetDefaultOptions(options);
  options[METIS_OPTION_NUMBERING] = 0;
  const int status = METIS_PartGraphKway(const_cast<idx_t *>(&nvtxs),
                                         &ncon,
                                         xadj.data(),
                                         adjncy.data(),
                                         nullptr,
                                         nullptr,
                                         nullptr,
                                         &nparts,
                                         nullptr,
                                         nullptr,
                                         options,
                                         &objval,
                                         part.data());
  if (status != METIS_OK)
    throw InputError("METIS failed to partition the mesh.");
  return std::vector<int>(part.begin(), part.end());
}
#endif
} // namespace

std::vector<Index>
MeshPartition::elementsOf(int part) const
{
  std::vector<Index> out;
  for (std::size_t e = 0; e < element_part.size(); ++e)
    if (element_part[e] == part)
      out.push_back(static_cast<Index>(e));
  return out;
}

Index
MeshPartition::edgeCut(const Mesh & mesh) const
{
  const auto adjacency = elementAdjacency(mesh);
  Index cut = 0;
  for (Index e = 0; e < mesh.numElements(); ++e)
    for (Index n : adjacency[e])
      if (n > e && element_part[e] != element_part[n])
        ++cut;
  return cut;
}

std::pair<Index, Index>
MeshPartition::partSizes() const
{
  std::vector<Index> sizes(num_parts, 0);
  for (int p : element_part)
    ++sizes[p];
  return {*std::max_element(sizes.begin(), sizes.end()),
          *std::min_element(sizes.begin(), sizes.end())};
}

MeshPartition
partitionMesh(const Mesh & mesh, int num_parts, const std::string & method)
{
  if (num_parts < 1)
    throw InputError("The number of parts must be at least one.");
  if (num_parts > mesh.numElements())
    throw InputError("The mesh has fewer elements (" + std::to_string(mesh.numElements()) +
                     ") than the requested number of parts (" + std::to_string(num_parts) + ").");
  MeshPartition out;
  out.num_parts = num_parts;
  if (num_parts == 1)
    out.element_part.assign(mesh.numElements(), 0);
  else if (method == "recursive_coordinate_bisection" || method == "rcb")
  {
    out.element_part.assign(mesh.numElements(), 0);
    std::vector<Index> all(mesh.numElements());
    std::iota(all.begin(), all.end(), 0);
    bisect(mesh, all, 0, num_parts, out.element_part);
  }
  else if (method == "graph")
    out.element_part = graphGrow(mesh, num_parts);
  else if (method == "metis")
  {
#ifdef DUALMESH_HAVE_METIS
    out.element_part = metisPartition(mesh, num_parts);
#else
    std::cerr << "dualmesh: this build has no METIS; using the built-in graph partitioner.\n";
    out.element_part = graphGrow(mesh, num_parts);
#endif
  }
  else
    throw InputError("Unknown partitioner '" + method +
                     "' (use recursive_coordinate_bisection, graph, or metis).");

  // Which parts touch each node, and which of them owns it.
  out.node_parts.assign(mesh.numNodes(), {});
  for (Index e = 0; e < mesh.numElements(); ++e)
  {
    const auto & el = mesh.element(e);
    for (int k = 0; k < el.numNodes(); ++k)
      out.node_parts[el.nodes[k]].push_back(out.element_part[e]);
  }
  out.node_owner.assign(mesh.numNodes(), 0);
  for (Index n = 0; n < mesh.numNodes(); ++n)
  {
    auto & parts = out.node_parts[n];
    std::sort(parts.begin(), parts.end());
    parts.erase(std::unique(parts.begin(), parts.end()), parts.end());
    out.node_owner[n] = parts.empty() ? 0 : parts.front();
  }
  return out;
}

Mesh
subMesh(const Mesh & mesh,
        const std::vector<Index> & elements,
        std::vector<Index> & local_to_global)
{
  Mesh out(mesh.dimension());
  local_to_global.clear();
  std::vector<Index> global_to_local(mesh.numNodes(), -1);
  std::map<Index, Index> element_map; // global element -> local element
  for (Index e : elements)
  {
    const auto & el = mesh.element(e);
    std::vector<Index> nodes(el.numNodes());
    for (int k = 0; k < el.numNodes(); ++k)
    {
      const Index g = el.nodes[k];
      if (global_to_local[g] < 0)
      {
        global_to_local[g] = out.addNode(mesh.node(g));
        local_to_global.push_back(g);
      }
      nodes[k] = global_to_local[g];
    }
    element_map[e] = out.addElement(el.type, nodes, el.block);
  }
  for (const auto & [block, name] : mesh.blockNames())
    out.setBlockName(block, name);
  for (const auto & [name, sides] : mesh.sidesets())
  {
    std::vector<Side> kept;
    for (const Side & s : sides)
    {
      auto it = element_map.find(s.first);
      if (it != element_map.end())
        kept.push_back({it->second, s.second});
    }
    out.addSideset(name, kept);
    // A node of the boundary may belong to this sub-mesh while the boundary
    // side that carries it belongs to another one.  Essential boundary
    // conditions are prescribed node by node, so the complete node list of the
    // global side set is restricted to the local nodes and stored as a node
    // set of the same name; Mesh::boundaryNodes prefers the node set, so every
    // rank prescribes the value at every boundary node it holds, whether it
    // owns the side or not.  Integrated boundary conditions still use the side
    // list, so each side is integrated exactly once, on one rank.
    std::vector<Index> boundary_nodes;
    for (Index g : mesh.boundaryNodes(name))
      if (global_to_local[g] >= 0)
        boundary_nodes.push_back(global_to_local[g]);
    out.addNodeset(name, boundary_nodes);
  }
  for (const auto & [name, nodes] : mesh.nodesets())
  {
    std::vector<Index> kept;
    for (Index n : nodes)
      if (global_to_local[n] >= 0)
        kept.push_back(global_to_local[n]);
    out.addNodeset(name, kept);
  }
  return out;
}

} // namespace dualmesh
