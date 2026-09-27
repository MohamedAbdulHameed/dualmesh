// SPDX-License-Identifier: LGPL-2.1-or-later
#include "dualmesh/mesh/Adapt.h"
#include "dualmesh/core/InputParameters.h"
#include "dualmesh/fe/ReferenceElement.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <set>

namespace dualmesh
{

namespace
{
using Edge = std::pair<Index, Index>;

Edge
edgeKey(Index a, Index b)
{
  return a < b ? Edge{a, b} : Edge{b, a};
}

/// One triangle of the working mesh.  Triangles are never erased, only
/// retired, so that an index stays valid while the algorithm runs.
struct Triangle
{
  std::array<Index, 3> nodes;
  int block = 0;
  Index parent = 0; ///< index in the original mesh
  bool live = true;
};
} // namespace

Mesh
refineMarked(const Mesh & mesh, const std::vector<char> & marked, std::vector<Index> * parents)
{
  for (const auto & el : mesh.elements())
    if (el.type != ElementType::Tri3)
      throw InputError(
          "Local refinement is implemented for triangular (Tri3) meshes only, because the "
          "longest-edge bisection that keeps the mesh conforming is a triangle algorithm. A "
          "mesh of another element type can be refined uniformly with refined(), and a "
          "quadrilateral mesh can be converted to triangles before refining.");
  if (marked.size() != static_cast<std::size_t>(mesh.numElements()))
    throw InputError("The refinement marker must have one entry per element.");

  // ---- working data -------------------------------------------------------
  std::vector<Point> nodes = mesh.nodes();
  std::vector<Triangle> tris;
  tris.reserve(mesh.numElements() * 4);
  for (Index e = 0; e < mesh.numElements(); ++e)
  {
    Triangle t;
    const auto & el = mesh.element(e);
    t.nodes = {el.nodes[0], el.nodes[1], el.nodes[2]};
    t.block = el.block;
    t.parent = e;
    tris.push_back(t);
  }

  // Every edge remembers the live triangles that use it (one or two), and
  // every bisected edge remembers the node that was put on it, so that the two
  // triangles sharing it get the same node.
  std::map<Edge, std::vector<std::size_t>> edge_triangles;
  std::map<Edge, Index> midpoint;
  const auto attach = [&](std::size_t t)
  {
    for (int i = 0; i < 3; ++i)
      edge_triangles[edgeKey(tris[t].nodes[i], tris[t].nodes[(i + 1) % 3])].push_back(t);
  };
  const auto detach = [&](std::size_t t)
  {
    for (int i = 0; i < 3; ++i)
    {
      auto & list = edge_triangles[edgeKey(tris[t].nodes[i], tris[t].nodes[(i + 1) % 3])];
      list.erase(std::remove(list.begin(), list.end(), t), list.end());
    }
  };
  for (std::size_t t = 0; t < tris.size(); ++t)
    attach(t);

  const auto length = [&](Index a, Index b) { return norm(nodes[a] - nodes[b]); };
  /// The local index of the longest edge (i, i+1) of a triangle.  Ties are
  /// broken by the node numbers so that two triangles sharing an edge always
  /// agree on whether it is the longest one; without that the propagation can
  /// bounce between two triangles for ever on a mesh with equal edges.
  const auto longestEdge = [&](std::size_t t)
  {
    int best = 0;
    Edge best_key = edgeKey(tris[t].nodes[0], tris[t].nodes[1]);
    double best_length = length(best_key.first, best_key.second);
    for (int i = 1; i < 3; ++i)
    {
      const Edge key = edgeKey(tris[t].nodes[i], tris[t].nodes[(i + 1) % 3]);
      const double l = length(key.first, key.second);
      const double scale = std::max(l, best_length);
      const bool longer = l > best_length + 1e-12 * scale;
      const bool equal = std::abs(l - best_length) <= 1e-12 * scale;
      if (longer || (equal && key > best_key))
      {
        best_length = l;
        best = i;
        best_key = key;
      }
    }
    return best;
  };
  const auto nodeOnEdge = [&](const Edge & e)
  {
    auto it = midpoint.find(e);
    if (it != midpoint.end())
      return it->second;
    nodes.push_back(0.5 * (nodes[e.first] + nodes[e.second]));
    const Index id = static_cast<Index>(nodes.size() - 1);
    midpoint[e] = id;
    return id;
  };
  /// Split one triangle across the midpoint @p m of its local edge @p i.
  const auto bisect = [&](std::size_t t, int i, Index m)
  {
    const Index a = tris[t].nodes[i];
    const Index b = tris[t].nodes[(i + 1) % 3];
    const Index c = tris[t].nodes[(i + 2) % 3];
    detach(t);
    tris[t].live = false;
    Triangle left = tris[t], right = tris[t];
    left.live = right.live = true;
    left.nodes = {a, m, c};
    right.nodes = {m, b, c};
    tris.push_back(left);
    attach(tris.size() - 1);
    tris.push_back(right);
    attach(tris.size() - 1);
    return std::pair<std::size_t, std::size_t>{tris.size() - 2, tris.size() - 1};
  };

  // ---- the refinement loop ------------------------------------------------
  // A triangle on the stack is one that still has to be bisected.  Bisecting
  // it requires that its longest edge is also the longest edge of the
  // neighbour across it; if it is not, the neighbour goes on the stack first
  // and is dealt with before this triangle is looked at again.  Every such
  // detour bisects a strictly longer edge than the one that caused it, so the
  // chain cannot cycle and the loop ends.
  std::vector<std::size_t> stack;
  for (std::size_t t = 0; t < tris.size(); ++t)
    if (marked[tris[t].parent])
      stack.push_back(t);

  // Rivara's theorem says the loop ends, but a wrong geometry (a degenerate
  // triangle, say) could still make it spin; the cap turns that into an error
  // instead of a program that never returns.
  const std::size_t step_limit = 1000 * (mesh.numElements() + 1);
  std::size_t steps = 0;
  while (!stack.empty())
  {
    if (++steps > step_limit)
      throw InputError("Local refinement did not terminate. This should not happen for a valid "
                       "triangulation; check the mesh for degenerate or duplicated elements.");
    const std::size_t t = stack.back();
    if (!tris[t].live)
    {
      stack.pop_back();
      continue;
    }
    const int i = longestEdge(t);
    const Edge key = edgeKey(tris[t].nodes[i], tris[t].nodes[(i + 1) % 3]);
    // The neighbour across the longest edge, if there is one.  The sentinel
    // has to be a value that can never become a valid index: the triangle
    // vector grows during the bisection below, so "one past the end" would
    // turn into one of the new children.
    constexpr std::size_t none = static_cast<std::size_t>(-1);
    std::size_t neighbor = none;
    for (std::size_t other : edge_triangles[key])
      if (other != t && tris[other].live)
        neighbor = other;

    int neighbor_edge = -1;
    if (neighbor != none)
    {
      neighbor_edge = longestEdge(neighbor);
      const Edge neighbor_key = edgeKey(tris[neighbor].nodes[neighbor_edge],
                                        tris[neighbor].nodes[(neighbor_edge + 1) % 3]);
      if (neighbor_key != key)
      {
        // Refine the neighbour first; this triangle stays on the stack.
        stack.push_back(neighbor);
        continue;
      }
    }

    stack.pop_back();
    const Index m = nodeOnEdge(key);
    bisect(t, i, m);
    if (neighbor != none)
      bisect(neighbor, neighbor_edge, m);
  }

  // ---- assemble the refined mesh ------------------------------------------
  Mesh out(mesh.dimension());
  for (const auto & [block, name] : mesh.blockNames())
    out.setBlockName(block, name);
  for (const auto & p : nodes)
    out.addNode(p);
  std::vector<Index> parent_of_child;
  for (const auto & t : tris)
  {
    if (!t.live)
      continue;
    out.addElement(ElementType::Tri3, {t.nodes[0], t.nodes[1], t.nodes[2]}, t.block);
    parent_of_child.push_back(t.parent);
  }
  if (parents)
    *parents = parent_of_child;

  // ---- side sets and node sets --------------------------------------------
  // A child side belongs to a parent's side set when it is on the exterior of
  // the refined mesh and all of its nodes lie on that side set: either they
  // are nodes of the set already, or they are midpoints of edges whose ends
  // both lie on it.  The midpoints are added one round at a time, because a
  // midpoint can itself be the end of a later edge.
  std::map<std::vector<Index>, Side> child_sides;
  for (const auto & s : out.exteriorSides())
  {
    auto key = out.sideNodes(s);
    std::sort(key.begin(), key.end());
    child_sides[key] = s;
  }
  const auto closure = [&](std::set<Index> members)
  {
    bool grew = true;
    while (grew)
    {
      grew = false;
      for (const auto & [edge, id] : midpoint)
        if (!members.count(id) && members.count(edge.first) && members.count(edge.second))
        {
          members.insert(id);
          grew = true;
        }
    }
    return members;
  };
  for (const auto & [name, sides] : mesh.sidesets())
  {
    std::set<Index> members;
    for (const auto & s : sides)
      for (Index n : mesh.sideNodes(s))
        members.insert(n);
    members = closure(members);
    std::vector<Side> list;
    for (const auto & [key, side] : child_sides)
    {
      bool inside = true;
      for (Index n : key)
        inside = inside && members.count(n);
      if (inside)
        list.push_back(side);
    }
    out.addSideset(name, list);
  }
  for (const auto & [name, ids] : mesh.nodesets())
  {
    const auto members = closure(std::set<Index>(ids.begin(), ids.end()));
    out.addNodeset(name, {members.begin(), members.end()});
  }
  return out;
}

} // namespace dualmesh
