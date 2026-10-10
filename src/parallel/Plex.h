// SPDX-License-Identifier: LGPL-2.1-or-later
//
// The bridge between a dualmesh Mesh and PETSc's unstructured mesh, DMPlex, which partitions and distributes the elements.
// Only the corner topology goes into DMPlex: every element is a cell whose cone is its corner nodes, so that every element type, the quadratic ones included, is partitioned by its corners, and the element itself travels as data attached to its cell.
#pragma once

#ifdef DUALMESH_HAVE_PETSC

#include "dualmesh/mesh/Mesh.h"

#include <petscdmplex.h>

#include <string>

namespace dualmesh
{
namespace plex
{

/// The corner topology of every element of @p mesh, as an uninterpolated DMPlex on @p comm in which cell e is element e.
/// A rank with @p holds false contributes no cell.
DM cornerTopology(const Mesh & mesh, MPI_Comm comm, bool holds);

/// The PETSc partitioner type of a partitioner name: "ptscotch", "parmetis", "simple", or "automatic" (PT-Scotch, else ParMETIS, else simple).
/// An unknown name, or one that this PETSc lacks, is an InputError that lists the available ones.
std::string partitionerType(const std::string & name);

} // namespace plex
} // namespace dualmesh

#endif
