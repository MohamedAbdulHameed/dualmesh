// SPDX-License-Identifier: LGPL-2.1-or-later
//
// The parts of a distributed mesh, and how each rank gets its part.
#pragma once

#include "dualmesh/mesh/Mesh.h"
#include "dualmesh/parallel/Communicator.h"

#include <memory>
#include <string>
#include <vector>

namespace dualmesh
{

/// The part of a distributed mesh that one rank holds: its elements, every node they touch, and the global index of each of those nodes.
/// Every element belongs to exactly one rank.
/// Every rank lists the same side set names, also when it holds none of the sides of one; the side list holds only the rank's own sides.
/// A part may also hold ghost elements, copies of elements of other ranks next to its own, which follow its own elements; the side sets and node sets do not list them.
struct LocalMesh
{
  std::shared_ptr<Mesh> mesh;
  /// The global index of every local node.
  std::vector<Index> global_nodes;
  /// The global index of every local element; empty when the part was built without it, which a checkpoint of the history of the material and the ghost elements need.
  std::vector<Index> global_elements;
  /// The number of ghost elements at the end of the mesh.
  Index num_ghost_elements = 0;
  /// The layers of ghost elements around the rank's own elements: each layer holds every element of another rank that shares a node with an element of the layers inside it.
  int ghost_layers = 0;
};

/// Partition @p mesh on the root rank with @p partitioner, and send every rank its part with @p ghost_layers layers of ghost elements.
/// Only the root reads @p mesh, so the other ranks may pass an empty mesh and need not hold the whole mesh in memory.
/// A part keeps the global node numbering of @p mesh, its block names, its own sides of every side set, and the nodes it holds of every node set and of every side set (so that a value prescribed on a boundary reaches every rank that holds a node of it).
LocalMesh distributeMesh(const Mesh & mesh,
                         const Communicator & comm,
                         const std::string & partitioner,
                         int ghost_layers = 0);

} // namespace dualmesh
