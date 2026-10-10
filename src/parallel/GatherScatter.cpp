// SPDX-License-Identifier: LGPL-2.1-or-later
#include "dualmesh/parallel/GatherScatter.h"
#include "dualmesh/linalg/PetscSolver.h"

#include <algorithm>
#include <climits>
#include <map>
#include <stdexcept>
#include <string>
#include <unordered_map>

#ifdef DUALMESH_HAVE_MPI
#include <mpi.h>
#endif
#ifdef DUALMESH_HAVE_PETSC
#include <petscsf.h>
#endif

#ifdef DUALMESH_HAVE_GSLIB
// src/parallel/GslibShim.c, the C side of gslib.
extern "C"
{
  void * dualmesh_gslib_setup(MPI_Comm communicator, const long long * ids, unsigned n);
  void dualmesh_gslib_sum(void * self, double * values, unsigned width);
  void dualmesh_gslib_min(void * self, int * values);
  void dualmesh_gslib_free(void * self);
}
#endif

namespace dualmesh
{

bool
GatherScatter::haveGslib()
{
#ifdef DUALMESH_HAVE_GSLIB
  return true;
#else
  return false;
#endif
}

GatherScatter::Library
GatherScatter::library(const std::string & name)
{
  if (name == "petsc")
    return Library::Petsc;
  if (name == "gslib")
  {
    if (!haveGslib())
      throw std::invalid_argument("gather_scatter='gslib' needs a build with MPI and gslib "
                                  "(DUALMESH_ENABLE_GSLIB, on by default with MPI).");
    return Library::Gslib;
  }
  throw std::invalid_argument("gather_scatter is 'petsc' or 'gslib', not '" + name + "'.");
}

#ifdef DUALMESH_HAVE_PETSC

#define DM_PETSC(call) petsc::check((call), #call)

/// The star forest of the shared entities, and its expansion to every width of the values that has been used.
/// Each value is its own leaf, because MPI reduces only the values of its built-in types.
struct GatherScatter::Star
{
  Index size = 0;
  std::vector<PetscInt> leaves;
  std::vector<PetscSFNode> roots;
  mutable std::map<int, PetscSF> forests;

  ~Star()
  {
    // A problem may outlive PETSc at the exit of a program; there is then nothing left to free.
    if (!PetscInitializeCalled || PetscFinalizeCalled)
      return;
    for (auto & [width, sf] : forests)
      PetscSFDestroy(&sf);
  }

  PetscSF forest(int width) const
  {
    auto it = forests.find(width);
    if (it != forests.end())
      return it->second;
    std::vector<PetscInt> wide_leaves;
    std::vector<PetscSFNode> wide_roots;
    wide_leaves.reserve(leaves.size() * static_cast<std::size_t>(width));
    wide_roots.reserve(roots.size() * static_cast<std::size_t>(width));
    for (std::size_t k = 0; k < leaves.size(); ++k)
      for (int c = 0; c < width; ++c)
      {
        wide_leaves.push_back(leaves[k] * width + c);
        wide_roots.push_back({roots[k].rank, roots[k].index * width + c});
      }
    PetscSF sf;
    DM_PETSC(PetscSFCreate(PETSC_COMM_WORLD, &sf));
    DM_PETSC(PetscSFSetGraph(sf,
                             static_cast<PetscInt>(size * width),
                             static_cast<PetscInt>(wide_leaves.size()),
                             wide_leaves.data(),
                             PETSC_COPY_VALUES,
                             wide_roots.data(),
                             PETSC_COPY_VALUES));
    DM_PETSC(PetscSFSetUp(sf));
    forests[width] = sf;
    return sf;
  }
};

#else

struct GatherScatter::Star
{
};

#endif

GatherScatter::GatherScatter() = default;

GatherScatter::~GatherScatter()
{
#ifdef DUALMESH_HAVE_GSLIB
  if (_gslib)
    dualmesh_gslib_free(_gslib);
#endif
}

void
GatherScatter::setup(const Communicator & comm,
                     const std::vector<Index> & global_ids,
                     Library library,
                     const std::vector<char> & claims)
{
  _rank = comm.rank();
  const std::size_t n = global_ids.size();
  if (!claims.empty() && claims.size() != n)
    throw std::invalid_argument("GatherScatter: one claim flag per local entity is needed.");
  // Copies of one index on this rank are one entity: the first copy represents it, and the libraries see each index once.
  // An entity that only this rank has (a negative index) is left out.
  std::unordered_map<Index, Index> compact_of;
  compact_of.reserve(n);
  _compact_of.assign(n, -1);
  _representative.clear();
  std::vector<Index> compact_ids;
  std::vector<char> compact_claims;
  for (std::size_t i = 0; i < n; ++i)
  {
    if (global_ids[i] < 0)
      continue;
    auto [it, added] =
        compact_of.try_emplace(global_ids[i], static_cast<Index>(_representative.size()));
    if (added)
    {
      _representative.push_back(static_cast<Index>(i));
      compact_ids.push_back(global_ids[i]);
      compact_claims.push_back(0);
    }
    _compact_of[i] = it->second;
    if (!claims.empty() && claims[i])
      compact_claims[it->second] = 1;
  }
  _duplicates = _representative.size() < n;
  setupCompact(comm, compact_ids, library, claims.empty() ? claims : compact_claims);
  _owner.assign(n, -1);
  _owns.assign(n, 0);
  for (std::size_t i = 0; i < n; ++i)
  {
    const Index k = _compact_of[i];
    if (k < 0)
      continue;
    _owner[i] = _owner_compact[k];
    _owns[i] = _owner[i] == _rank && _representative[k] == static_cast<Index>(i);
  }
}

void
GatherScatter::setupCompact(const Communicator & comm,
                            const std::vector<Index> & global_ids,
                            Library library,
                            const std::vector<char> & claims)
{
  const std::size_t n = global_ids.size();
  _owner_compact.assign(n, _rank);
  _star.reset();
  // The owner is the smallest rank of the copies that claim the entity, and of all copies when none does: a copy that does not claim enters the minimum as its rank plus the number of ranks.
  const int ranks = comm.size();
  const auto key = [&](std::size_t i)
  { return claims.empty() || claims[i] ? _rank : _rank + ranks; };
  (void) key;
#ifdef DUALMESH_HAVE_GSLIB
  if (_gslib)
    dualmesh_gslib_free(_gslib);
  _gslib = nullptr;
#endif
  if (comm.size() == 1)
    return;
#ifdef DUALMESH_HAVE_GSLIB
  if (library == Library::Gslib)
  {
    // gslib ignores the index 0, so the indices are shifted by one.
    std::vector<long long> ids(n);
    for (std::size_t i = 0; i < n; ++i)
      ids[i] = static_cast<long long>(global_ids[i]) + 1;
    _gslib = dualmesh_gslib_setup(MPI_COMM_WORLD, ids.data(), static_cast<unsigned>(n));
    for (std::size_t i = 0; i < n; ++i)
      _owner_compact[i] = key(i);
    dualmesh_gslib_min(_gslib, _owner_compact.data());
    for (auto & owner : _owner_compact)
      owner %= ranks;
    return;
  }
#endif
  if (library == Library::Gslib)
    throw std::invalid_argument("GatherScatter: this build has no gslib.");
#ifdef DUALMESH_HAVE_PETSC
  petsc::initialize();
  Index largest = -1;
  for (Index g : global_ids)
    largest = std::max(largest, g);
  const Index global_size = comm.max(largest) + 1;
  if (global_size > static_cast<Index>(PETSC_MAX_INT) || n > static_cast<std::size_t>(INT_MAX))
    throw std::runtime_error("GatherScatter: the global indices exceed the integers of this PETSc "
                             "build; configure PETSc with --with-64-bit-indices.");

  // Each global index has a home rank, its root in an even layout of the indices.
  // The home learns, by a minimum reduction of the pairs (rank, local index) of every copy, which copy belongs to the smallest rank, and tells every copy.
  PetscLayout layout;
  DM_PETSC(PetscLayoutCreate(PETSC_COMM_WORLD, &layout));
  DM_PETSC(PetscLayoutSetSize(layout, static_cast<PetscInt>(global_size)));
  DM_PETSC(PetscLayoutSetUp(layout));
  PetscInt home_count = 0;
  DM_PETSC(PetscLayoutGetLocalSize(layout, &home_count));
  PetscSF home;
  DM_PETSC(PetscSFCreate(PETSC_COMM_WORLD, &home));
  std::vector<PetscInt> ids(global_ids.begin(), global_ids.end());
  DM_PETSC(PetscSFSetGraphLayout(
      home, layout, static_cast<PetscInt>(n), nullptr, PETSC_COPY_VALUES, ids.data()));
  DM_PETSC(PetscLayoutDestroy(&layout));
  std::vector<int> copies(2 * n), first(2 * static_cast<std::size_t>(home_count), INT_MAX);
  for (std::size_t i = 0; i < n; ++i)
  {
    copies[2 * i] = key(i);
    copies[2 * i + 1] = static_cast<int>(i);
  }
  DM_PETSC(PetscSFReduceBegin(home, MPI_2INT, copies.data(), first.data(), MPI_MINLOC));
  DM_PETSC(PetscSFReduceEnd(home, MPI_2INT, copies.data(), first.data(), MPI_MINLOC));
  std::vector<int> owner_of(2 * n);
  DM_PETSC(PetscSFBcastBegin(home, MPI_2INT, first.data(), owner_of.data(), MPI_REPLACE));
  DM_PETSC(PetscSFBcastEnd(home, MPI_2INT, first.data(), owner_of.data(), MPI_REPLACE));
  DM_PETSC(PetscSFDestroy(&home));

  // The star forest of the entities: every copy that this rank does not own is a leaf of the owner's copy.
  _star = std::make_unique<Star>();
  _star->size = static_cast<Index>(n);
  for (std::size_t i = 0; i < n; ++i)
  {
    _owner_compact[i] = owner_of[2 * i] % ranks;
    if (_owner_compact[i] != _rank)
    {
      _star->leaves.push_back(static_cast<PetscInt>(i));
      _star->roots.push_back({_owner_compact[i], owner_of[2 * i + 1]});
    }
  }
#else
  throw std::runtime_error("GatherScatter: more than one rank needs a build with MPI and PETSc.");
#endif
}

Index
GatherScatter::numOwned() const
{
  return static_cast<Index>(std::count(_owns.begin(), _owns.end(), 1));
}

void
GatherScatter::sum(double * values, int width) const
{
  if (!_duplicates)
  {
    sumCompact(values, width);
    return;
  }
  const auto w = static_cast<std::size_t>(width);
  std::vector<double> compact(_representative.size() * w, 0.0);
  for (std::size_t i = 0; i < _compact_of.size(); ++i)
    if (_compact_of[i] >= 0)
      for (std::size_t c = 0; c < w; ++c)
        compact[static_cast<std::size_t>(_compact_of[i]) * w + c] += values[i * w + c];
  sumCompact(compact.data(), width);
  for (std::size_t i = 0; i < _compact_of.size(); ++i)
    if (_compact_of[i] >= 0)
      for (std::size_t c = 0; c < w; ++c)
        values[i * w + c] = compact[static_cast<std::size_t>(_compact_of[i]) * w + c];
}

void
GatherScatter::copyFromOwners(double * values, int width) const
{
  if (!_duplicates)
  {
    copyCompact(values, width);
    return;
  }
  const auto w = static_cast<std::size_t>(width);
  std::vector<double> compact(_representative.size() * w);
  for (std::size_t k = 0; k < _representative.size(); ++k)
    for (std::size_t c = 0; c < w; ++c)
      compact[k * w + c] = values[static_cast<std::size_t>(_representative[k]) * w + c];
  copyCompact(compact.data(), width);
  for (std::size_t i = 0; i < _compact_of.size(); ++i)
    if (_compact_of[i] >= 0)
      for (std::size_t c = 0; c < w; ++c)
        values[i * w + c] = compact[static_cast<std::size_t>(_compact_of[i]) * w + c];
}

void
GatherScatter::sumCompact(double * values, int width) const
{
#ifdef DUALMESH_HAVE_GSLIB
  if (_gslib)
  {
    dualmesh_gslib_sum(_gslib, values, static_cast<unsigned>(width));
    return;
  }
#endif
#ifdef DUALMESH_HAVE_PETSC
  if (!_star)
    return;
  const PetscSF sf = _star->forest(width);
  // The owners collect the sums of all copies, and then send them to the copies.
  std::vector<double> owned(values, values + _star->size * width);
  DM_PETSC(PetscSFReduceBegin(sf, MPI_DOUBLE, values, owned.data(), MPI_SUM));
  DM_PETSC(PetscSFReduceEnd(sf, MPI_DOUBLE, values, owned.data(), MPI_SUM));
  std::copy(owned.begin(), owned.end(), values);
  DM_PETSC(PetscSFBcastBegin(sf, MPI_DOUBLE, owned.data(), values, MPI_REPLACE));
  DM_PETSC(PetscSFBcastEnd(sf, MPI_DOUBLE, owned.data(), values, MPI_REPLACE));
#else
  (void) values;
  (void) width;
#endif
}

void
GatherScatter::copyCompact(double * values, int width) const
{
#ifdef DUALMESH_HAVE_GSLIB
  if (_gslib)
  {
    // Only the owner contributes to the sum, which is therefore the owner's value.
    const std::size_t n = _owner_compact.size();
    std::vector<double> owned(values, values + n * static_cast<std::size_t>(width));
    for (std::size_t i = 0; i < n; ++i)
      if (_owner_compact[i] != _rank)
        std::fill_n(owned.begin() + static_cast<std::ptrdiff_t>(i * width), width, 0.0);
    dualmesh_gslib_sum(_gslib, owned.data(), static_cast<unsigned>(width));
    for (std::size_t i = 0; i < n; ++i)
      if (_owner_compact[i] != _rank)
        std::copy_n(owned.begin() + static_cast<std::ptrdiff_t>(i * width),
                    width,
                    values + i * static_cast<std::size_t>(width));
    return;
  }
#endif
#ifdef DUALMESH_HAVE_PETSC
  if (!_star)
    return;
  const PetscSF sf = _star->forest(width);
  const std::vector<double> owned(values, values + _star->size * width);
  DM_PETSC(PetscSFBcastBegin(sf, MPI_DOUBLE, owned.data(), values, MPI_REPLACE));
  DM_PETSC(PetscSFBcastEnd(sf, MPI_DOUBLE, owned.data(), values, MPI_REPLACE));
#else
  (void) values;
  (void) width;
#endif
}

} // namespace dualmesh
