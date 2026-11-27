// SPDX-License-Identifier: LGPL-2.1-or-later
#include "dualmesh/linalg/Communicator.h"
#include "dualmesh/linalg/PetscSolver.h"

#include <cstdlib>
#include <numeric>

#include <mpi.h>

namespace dualmesh
{

namespace
{
bool g_we_initialized = false;

void
finalizeIfWeStarted()
{
  // PETSc runs on top of MPI and must be finalized first.
  petsc::finalize();
  int finalized = 0;
  MPI_Finalized(&finalized);
  if (g_we_initialized && !finalized)
    MPI_Finalize();
}
} // namespace

void
Communicator::finalize()
{
  finalizeIfWeStarted();
}

Communicator::Communicator()
{
  int initialized = 0;
  MPI_Initialized(&initialized);
  if (!initialized)
  {
    // The library may be loaded from a Python interpreter that never calls
    // MPI_Init, so initialize it here and clean up at exit.  A thread-safe
    // level is requested because the assembly is threaded.
    int provided = 0;
    MPI_Init_thread(nullptr, nullptr, MPI_THREAD_FUNNELED, &provided);
    g_we_initialized = true;
    std::atexit(finalizeIfWeStarted);
  }
  MPI_Comm_rank(MPI_COMM_WORLD, &_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &_size);
}

Communicator &
Communicator::world()
{
  static Communicator instance;
  return instance;
}

double
Communicator::sum(double value) const
{
  if (_size > 1)
  {
    double out = 0;
    MPI_Allreduce(&value, &out, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return out;
  }
  return value;
}

double
Communicator::max(double value) const
{
  if (_size > 1)
  {
    double out = 0;
    MPI_Allreduce(&value, &out, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    return out;
  }
  return value;
}

double
Communicator::min(double value) const
{
  if (_size > 1)
  {
    double out = 0;
    MPI_Allreduce(&value, &out, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    return out;
  }
  return value;
}

Index
Communicator::sum(Index value) const
{
  if (_size > 1)
  {
    long in = value, out = 0;
    MPI_Allreduce(&in, &out, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
    return static_cast<Index>(out);
  }
  return value;
}

Index
Communicator::max(Index value) const
{
  if (_size > 1)
  {
    long in = value, out = 0;
    MPI_Allreduce(&in, &out, 1, MPI_LONG, MPI_MAX, MPI_COMM_WORLD);
    return static_cast<Index>(out);
  }
  return value;
}

Index
Communicator::min(Index value) const
{
  if (_size > 1)
  {
    long in = value, out = 0;
    MPI_Allreduce(&in, &out, 1, MPI_LONG, MPI_MIN, MPI_COMM_WORLD);
    return static_cast<Index>(out);
  }
  return value;
}

void
Communicator::sumInPlace(std::vector<double> & values) const
{
  if (_size > 1 && !values.empty())
    MPI_Allreduce(MPI_IN_PLACE,
                  values.data(),
                  static_cast<int>(values.size()),
                  MPI_DOUBLE,
                  MPI_SUM,
                  MPI_COMM_WORLD);
}

bool
Communicator::any(bool value) const
{
  if (_size > 1)
  {
    int in = value ? 1 : 0, out = 0;
    MPI_Allreduce(&in, &out, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    return out != 0;
  }
  return value;
}

void
Communicator::barrier() const
{
  if (_size > 1)
    MPI_Barrier(MPI_COMM_WORLD);
}

void
Communicator::broadcast(std::vector<Index> & values) const
{
  if (_size > 1)
  {
    long size = static_cast<long>(values.size());
    MPI_Bcast(&size, 1, MPI_LONG, 0, MPI_COMM_WORLD);
    values.resize(static_cast<std::size_t>(size));
    if (size > 0)
      MPI_Bcast(values.data(), static_cast<int>(size), MPI_LONG, 0, MPI_COMM_WORLD);
  }
}

namespace
{
template <typename T>
std::vector<T>
allGatherVector(int size, const std::vector<T> & values, MPI_Datatype type)
{
  std::vector<int> counts(size), offsets(size, 0);
  const int mine = static_cast<int>(values.size());
  MPI_Allgather(&mine, 1, MPI_INT, counts.data(), 1, MPI_INT, MPI_COMM_WORLD);
  for (int r = 1; r < size; ++r)
    offsets[r] = offsets[r - 1] + counts[r - 1];
  std::vector<T> out(static_cast<std::size_t>(offsets.back() + counts.back()));
  MPI_Allgatherv(
      values.data(), mine, type, out.data(), counts.data(), offsets.data(), type, MPI_COMM_WORLD);
  return out;
}
} // namespace

std::vector<Index>
Communicator::allGather(const std::vector<Index> & values) const
{
  if (_size > 1)
    return allGatherVector(_size, values, MPI_LONG);
  return values;
}

std::vector<double>
Communicator::allGather(const std::vector<double> & values) const
{
  if (_size > 1)
    return allGatherVector(_size, values, MPI_DOUBLE);
  return values;
}

std::vector<Index>
Communicator::allGather(Index value) const
{
  std::vector<Index> out(_size, value);
  if (_size > 1)
  {
    std::vector<long> buffer(_size, 0);
    long in = value;
    MPI_Allgather(&in, 1, MPI_LONG, buffer.data(), 1, MPI_LONG, MPI_COMM_WORLD);
    for (int r = 0; r < _size; ++r)
      out[r] = static_cast<Index>(buffer[r]);
  }
  return out;
}

} // namespace dualmesh
