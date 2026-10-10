// SPDX-License-Identifier: LGPL-2.1-or-later
//
// The gather-scatter of shared entities: the one communication primitive of every distributed operation on the nodes (or other entities) that several ranks hold.
// The caller gives the global index of every local entity, and the operations then combine the values of each index across the ranks that hold it, as gs_setup and gs_op of gslib do (Nek5000).
// Two libraries can do the communication: PETSc's star forests (PetscSF, the default), in which the owner of an entity is a root and the copies on the other ranks are its leaves, and gslib itself, which chooses between pairwise exchanges, a crystal router and an all-reduce by timing them at setup.
// Without MPI, or on one rank, every operation leaves the values unchanged.
#pragma once

#include "dualmesh/core/Types.h"
#include "dualmesh/parallel/Communicator.h"

#include <memory>
#include <string>
#include <vector>

namespace dualmesh
{

class GatherScatter
{
public:
  /// The library that does the communication.
  enum class Library
  {
    Petsc,
    Gslib
  };
  /// Whether this build has gslib (a build with MPI, unless configured without it).
  static bool haveGslib();
  /// The library named "petsc" or "gslib".
  static Library library(const std::string & name);

  GatherScatter();
  ~GatherScatter();
  GatherScatter(const GatherScatter &) = delete;
  GatherScatter & operator=(const GatherScatter &) = delete;

  /// Find the ranks that hold each of @p global_ids, the global index of every local entity, and the owner of each, the smallest of those ranks.
  /// Local entities that share an index are copies of one entity: their values are summed with those of the other ranks, and only the first copy on the owner counts as owned.
  /// A negative index marks an entity that only this rank has: no operation changes its values, and no rank owns it.
  /// @p claims, when given, holds one flag per local entity: the owner is then the smallest of the ranks whose copies claim the entity, and the smallest of all only when none claims it.
  void setup(const Communicator & comm,
             const std::vector<Index> & global_ids,
             Library library = Library::Petsc,
             const std::vector<char> & claims = {});

  /// Number of local entities.
  Index size() const { return static_cast<Index>(_owner.size()); }
  /// The owner of a local entity (-1 for an entity that only this rank has).
  int owner(Index local) const { return _owner[local]; }
  bool owns(Index local) const { return _owns[local] != 0; }
  /// The number of local entities this rank owns.
  Index numOwned() const;

  /// Add, at every shared entity, the values that the other ranks hold, so that every rank ends with the sum.
  /// @p values holds @p width consecutive values per entity.
  void sum(double * values, int width) const;
  /// Replace the values of every shared entity that this rank does not own by the values of its owner.
  void copyFromOwners(double * values, int width) const;

private:
  void setupCompact(const Communicator & comm,
                    const std::vector<Index> & global_ids,
                    Library library,
                    const std::vector<char> & claims);
  void sumCompact(double * values, int width) const;
  void copyCompact(double * values, int width) const;

  struct Star;
  std::unique_ptr<Star> _star;
  /// The gslib handle, when gslib does the communication.
  [[maybe_unused]] void * _gslib = nullptr;
  int _rank = 0;
  /// Per local entity: its owner rank, whether this copy is the owned one, and its index among the distinct indices (-1 for an entity that only this rank has).
  std::vector<int> _owner;
  std::vector<char> _owns;
  std::vector<Index> _compact_of;
  /// Per distinct index: its first local copy and its owner rank.
  std::vector<Index> _representative;
  std::vector<int> _owner_compact;
  bool _duplicates = false;
};

} // namespace dualmesh
