// SPDX-License-Identifier: LGPL-2.1-or-later
//
// Mesh partitioning for distributed-memory runs.
//
// A distributed run splits the elements of the mesh into one group per MPI
// rank.  Every element belongs to exactly one rank, so the element integrals
// are partitioned exactly; a node that several ranks touch receives a partial
// contribution from each of them, and the partial contributions are summed
// during assembly.  One of the ranks that touch a node is declared its owner,
// which is the rank that stores the node's degrees of freedom in the global
// vectors.
//
// Three partitioners are available.  "recursive_coordinate_bisection" is
// geometric: it repeatedly splits the set of element centroids in half along
// its longest axis.  It is fast, deterministic, and needs no connectivity, but
// it cuts more faces than necessary on unstructured meshes.  "graph" grows
// each part outward from a seed element through the face connectivity, which
// respects the actual mesh topology.  "metis" calls the METIS library on the
// dual graph of the mesh, which is what libMesh (and therefore MOOSE) uses;
// it is available only when the library was built with METIS.
#pragma once

#include "dualmesh/mesh/Mesh.h"

#include <string>
#include <vector>

namespace dualmesh
{

/// Which rank each element and each node belongs to, and the node numbering
/// each rank needs.
struct MeshPartition
{
  int num_parts = 1;
  /// Part of every element.
  std::vector<int> element_part;
  /// Owner rank of every node: the smallest part index among the parts that
  /// touch it, which every rank computes identically without communication.
  std::vector<int> node_owner;
  /// Parts that touch each node, sorted and without duplicates.
  std::vector<std::vector<int>> node_parts;

  /// Elements of one part.
  std::vector<Index> elementsOf(int part) const;
  /// Number of faces of the mesh whose two elements are in different parts,
  /// which is the quantity a good partitioner minimizes.
  Index edgeCut(const Mesh & mesh) const;
  /// Largest and smallest part size, as a measure of load balance.
  std::pair<Index, Index> partSizes() const;
};

/// Partition @p mesh into @p num_parts parts.
/// @param method "recursive_coordinate_bisection" (the default), "graph", or
///        "metis"; "metis" falls back to "graph" with a warning when the
///        library was built without METIS.
MeshPartition partitionMesh(const Mesh & mesh,
                            int num_parts,
                            const std::string & method = "recursive_coordinate_bisection");

/// Whether the library was built against METIS.
bool haveMetis();

/// The element adjacency (dual) graph: for every element, the elements that
/// share a whole face with it.
std::vector<std::vector<Index>> elementAdjacency(const Mesh & mesh);

/// Extract the sub-mesh made of the given elements.
///
/// Nodes are renumbered consecutively in the order in which they are first
/// met; @p local_to_global receives the global node index of every local node.
/// Blocks are preserved, side sets and node sets are restricted to the
/// sides and nodes that survive, and a side set that ends up empty is still
/// created, so that boundary conditions referring to it remain valid on every
/// rank.
Mesh subMesh(const Mesh & mesh,
             const std::vector<Index> & elements,
             std::vector<Index> & local_to_global);

} // namespace dualmesh
