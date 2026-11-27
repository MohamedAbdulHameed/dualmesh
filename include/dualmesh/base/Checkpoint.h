// SPDX-License-Identifier: LGPL-2.1-or-later
//
// Checkpoints: the state of a problem written to one file, from which a run restarts on any number of processes.
// The file is PETSc's binary format, written and read in parallel by PETSc's viewers.
// Every field is stored in its natural order, by global node and variable, and every history by global element, so that the file does not depend on the partition of the run that wrote it.
// The file holds, in order: a header (format version, time, step size, step number, numbers of nodes, variables and elements, number of histories, fingerprints of the nodes and of the elements), the solution, and the histories: that of the material, then that of every other object that keeps one (CheckpointHistory), each as the number of its points per element and of its values per point, then its records.
#pragma once

#include "dualmesh/base/Problem.h"

#include <string>
#include <vector>

namespace dualmesh
{

/// Where the local entities of a problem lie in the whole problem.
struct CheckpointLayout
{
  /// Whether the problem spans every process (a distributed problem) or only this one.
  bool distributed = false;
  /// The global index of every local node, and the number of nodes of the whole problem.
  std::vector<Index> global_nodes;
  Index num_global_nodes = 0;
  /// The global index of every local element, and the number of elements of the whole problem; empty when unknown, in which case a problem with a material history cannot be checkpointed.
  std::vector<Index> global_elements;
  Index num_global_elements = 0;
};

/// The point of a run that a checkpoint records.
struct CheckpointTime
{
  double time = 0.0;
  /// The step size the run would take next.
  double dt = 0.0;
  /// The number of accepted steps.
  int step = 0;
};

/// An object other than a property object that keeps a history at points of the elements, such as
/// the damage of a cohesive zone.  A checkpoint writes and reads its committed history with the
/// history of the material, in the same form: for every local element, the keys of its points and
/// their values, the same number of values per point.  The keys identify a point independently of
/// the numbering of the elements and of the partition, so that a run restarts on any number of
/// processes.
class CheckpointHistory
{
public:
  virtual ~CheckpointHistory() = default;
  /// The committed history, one entry per local element of @p problem.
  virtual Problem::StateSnapshot committedHistory(const Problem & problem) const = 0;
  /// Make @p history, in the form of committedHistory, the committed history and the start of the
  /// next step.  Returns the reason when @p history does not fit the object, and an empty string
  /// otherwise, so that every process stops together.
  virtual std::string restoreHistory(const Problem & problem,
                                     const Problem::StateSnapshot & history) = 0;
};

/// The point of the run that @p file records, read from its header on every process of the problem (all of them for a distributed problem).
CheckpointTime checkpointTime(const std::string & file, bool distributed);

/// Write the solution and the histories of @p problem at @p time to @p file.
/// The file is written to a temporary name and renamed when complete, so that a run stopped while writing leaves the previous checkpoint intact.
void writeCheckpoint(const std::string & file,
                     const Problem & problem,
                     const CheckpointLayout & layout,
                     const CheckpointTime & time);

/// Read @p file into @p problem, which must have been initialized, and return the point of the run it records.
/// The problem must have the same variables, nodes and elements as the one that wrote the file, on any number of processes.
/// The fingerprints of the header check that the mesh is the same: the coordinates and the global numbers of its nodes, to the round-off of a sum in another order, and the global numbers, types, blocks and nodes of its elements, exactly.
CheckpointTime
readCheckpoint(const std::string & file, Problem & problem, const CheckpointLayout & layout);

} // namespace dualmesh
