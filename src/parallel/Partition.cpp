// SPDX-License-Identifier: LGPL-2.1-or-later
#include "dualmesh/parallel/Partition.h"

#include "dualmesh/core/InputParameters.h"
#include "dualmesh/fe/ReferenceElement.h"

#include <algorithm>
#include <map>
#include <set>

#ifdef DUALMESH_HAVE_PETSC
#include "Plex.h"
#include "dualmesh/linalg/PetscSolver.h"
#define DM_PETSC(call) petsc::check((call), #call)
#endif

namespace dualmesh
{

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

std::vector<std::string>
availablePartitioners()
{
  std::vector<std::string> out;
#ifdef DUALMESH_HAVE_PETSC
#ifdef PETSC_HAVE_PTSCOTCH
  out.push_back("ptscotch");
#endif
#ifdef PETSC_HAVE_PARMETIS
  out.push_back("parmetis");
#endif
  out.push_back("simple");
#endif
  return out;
}

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
  out.element_part.assign(mesh.numElements(), 0);
  if (num_parts > 1)
  {
#ifdef DUALMESH_HAVE_PETSC
    // PETSc's partitioner divides the graph of the elements that share a face, built by DMPlex from the corner topology.
    const std::string type = plex::partitionerType(method);
    DM dm = plex::cornerTopology(mesh, PETSC_COMM_SELF, true);
    PetscInt num_vertices = 0;
    PetscInt * offsets = nullptr;
    PetscInt * adjacency = nullptr;
    IS numbering = nullptr;
    DM_PETSC(DMPlexCreatePartitionerGraph(dm, 0, &num_vertices, &offsets, &adjacency, &numbering));
    PetscPartitioner partitioner;
    DM_PETSC(PetscPartitionerCreate(PETSC_COMM_SELF, &partitioner));
    DM_PETSC(PetscPartitionerSetType(partitioner, type.c_str()));
    PetscSection parts;
    DM_PETSC(PetscSectionCreate(PETSC_COMM_SELF, &parts));
    IS partition = nullptr;
    // PETSc 3.21 added the section of the edge weights.
#if PETSC_VERSION_GE(3, 21, 0)
    DM_PETSC(PetscPartitionerPartition(partitioner,
                                       num_parts,
                                       num_vertices,
                                       offsets,
                                       adjacency,
                                       nullptr,
                                       nullptr,
                                       nullptr,
                                       parts,
                                       &partition));
#else
    DM_PETSC(PetscPartitionerPartition(partitioner,
                                       num_parts,
                                       num_vertices,
                                       offsets,
                                       adjacency,
                                       nullptr,
                                       nullptr,
                                       parts,
                                       &partition));
#endif
    // The partition lists the elements part after part, and the section gives the number of elements of each part.
    const PetscInt * elements = nullptr;
    DM_PETSC(ISGetIndices(partition, &elements));
    for (int p = 0; p < num_parts; ++p)
    {
      PetscInt count = 0, offset = 0;
      DM_PETSC(PetscSectionGetDof(parts, p, &count));
      DM_PETSC(PetscSectionGetOffset(parts, p, &offset));
      for (PetscInt k = offset; k < offset + count; ++k)
        out.element_part[elements[k]] = p;
    }
    DM_PETSC(ISRestoreIndices(partition, &elements));
    DM_PETSC(ISDestroy(&partition));
    DM_PETSC(PetscSectionDestroy(&parts));
    DM_PETSC(PetscPartitionerDestroy(&partitioner));
    DM_PETSC(ISDestroy(&numbering));
    DM_PETSC(PetscFree(offsets));
    DM_PETSC(PetscFree(adjacency));
    DM_PETSC(DMDestroy(&dm));
#else
    (void) method;
    throw InputError(
        "Partitioning a mesh needs a build with PETSc (MPI, or -DDUALMESH_ENABLE_PETSC=ON).");
#endif
  }

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
