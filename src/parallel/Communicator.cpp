// SPDX-License-Identifier: LGPL-2.1-or-later
#include "dualmesh/parallel/Communicator.h"

#include <cstdlib>
#include <numeric>
#include <type_traits>

#ifdef DUALMESH_HAVE_MPI
#include <mpi.h>
#endif

namespace dualmesh
{

namespace
{
#ifdef DUALMESH_HAVE_MPI
bool g_we_initialized = false;

void
finalizeIfWeStarted()
{
  int finalized = 0;
  MPI_Finalized(&finalized);
  if (g_we_initialized && !finalized)
    MPI_Finalize();
}
#endif
} // namespace

void
Communicator::finalize()
{
#ifdef DUALMESH_HAVE_MPI
  finalizeIfWeStarted();
#endif
}

bool
Communicator::haveMpi()
{
#ifdef DUALMESH_HAVE_MPI
  return true;
#else
  return false;
#endif
}

Communicator::Communicator()
{
#ifdef DUALMESH_HAVE_MPI
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
#endif
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
#ifdef DUALMESH_HAVE_MPI
  if (_size > 1)
  {
    double out = 0;
    MPI_Allreduce(&value, &out, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return out;
  }
#endif
  return value;
}

double
Communicator::max(double value) const
{
#ifdef DUALMESH_HAVE_MPI
  if (_size > 1)
  {
    double out = 0;
    MPI_Allreduce(&value, &out, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    return out;
  }
#endif
  return value;
}

Index
Communicator::sum(Index value) const
{
#ifdef DUALMESH_HAVE_MPI
  if (_size > 1)
  {
    long in = value, out = 0;
    MPI_Allreduce(&in, &out, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
    return static_cast<Index>(out);
  }
#endif
  return value;
}

void
Communicator::sumInPlace(std::vector<double> & values) const
{
#ifdef DUALMESH_HAVE_MPI
  if (_size > 1 && !values.empty())
    MPI_Allreduce(MPI_IN_PLACE,
                  values.data(),
                  static_cast<int>(values.size()),
                  MPI_DOUBLE,
                  MPI_SUM,
                  MPI_COMM_WORLD);
#else
  (void) values;
#endif
}

void
Communicator::minInPlace(std::vector<double> & values) const
{
#ifdef DUALMESH_HAVE_MPI
  if (_size > 1 && !values.empty())
    MPI_Allreduce(MPI_IN_PLACE,
                  values.data(),
                  static_cast<int>(values.size()),
                  MPI_DOUBLE,
                  MPI_MIN,
                  MPI_COMM_WORLD);
#else
  (void) values;
#endif
}

bool
Communicator::any(bool value) const
{
#ifdef DUALMESH_HAVE_MPI
  if (_size > 1)
  {
    int in = value ? 1 : 0, out = 0;
    MPI_Allreduce(&in, &out, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    return out != 0;
  }
#endif
  return value;
}

void
Communicator::barrier() const
{
#ifdef DUALMESH_HAVE_MPI
  if (_size > 1)
    MPI_Barrier(MPI_COMM_WORLD);
#endif
}

std::vector<Index>
Communicator::allGather(Index value) const
{
  std::vector<Index> out(_size, value);
#ifdef DUALMESH_HAVE_MPI
  if (_size > 1)
  {
    std::vector<long> buffer(_size, 0);
    long in = value;
    MPI_Allgather(&in, 1, MPI_LONG, buffer.data(), 1, MPI_LONG, MPI_COMM_WORLD);
    for (int r = 0; r < _size; ++r)
      out[r] = static_cast<Index>(buffer[r]);
  }
#endif
  return out;
}

namespace
{
#ifdef DUALMESH_HAVE_MPI
template <typename T>
void
exchangeImpl(int size,
             const std::vector<std::vector<T>> & send,
             std::vector<std::vector<T>> & recv,
             MPI_Datatype type)
{
  std::vector<int> send_sizes(size, 0), recv_sizes(size, 0);
  for (int r = 0; r < size; ++r)
    send_sizes[r] = static_cast<int>(send[r].size());
  MPI_Alltoall(send_sizes.data(), 1, MPI_INT, recv_sizes.data(), 1, MPI_INT, MPI_COMM_WORLD);
  recv.assign(size, {});
  std::vector<MPI_Request> requests;
  for (int r = 0; r < size; ++r)
    if (recv_sizes[r] > 0)
    {
      recv[r].resize(recv_sizes[r]);
      requests.emplace_back();
      MPI_Irecv(recv[r].data(), recv_sizes[r], type, r, 17, MPI_COMM_WORLD, &requests.back());
    }
  for (int r = 0; r < size; ++r)
    if (send_sizes[r] > 0)
    {
      requests.emplace_back();
      MPI_Isend(send[r].data(), send_sizes[r], type, r, 17, MPI_COMM_WORLD, &requests.back());
    }
  if (!requests.empty())
    MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE);
}
#endif
} // namespace

void
Communicator::exchange(const std::vector<std::vector<double>> & send,
                       std::vector<std::vector<double>> & recv) const
{
#ifdef DUALMESH_HAVE_MPI
  if (_size > 1)
  {
    exchangeImpl(_size, send, recv, MPI_DOUBLE);
    return;
  }
#endif
  recv = send;
}

void
Communicator::exchange(const std::vector<std::vector<Index>> & send,
                       std::vector<std::vector<Index>> & recv) const
{
#ifdef DUALMESH_HAVE_MPI
  if (_size > 1)
  {
    static_assert(std::is_same<Index, long>::value, "Index must be long to match MPI_LONG");
    exchangeImpl(_size, send, recv, MPI_LONG);
    return;
  }
#endif
  recv = send;
}

} // namespace dualmesh
