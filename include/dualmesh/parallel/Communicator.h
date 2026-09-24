// SPDX-License-Identifier: LGPL-2.1-or-later
//
// A very small wrapper around the handful of MPI calls the solver needs.
//
// The wrapper exists so that every line of the distributed solver compiles and
// runs whether or not the library was built with MPI.  Without MPI, or with
// MPI but only one rank, all the collectives are the identity and all the
// point-to-point exchanges are empty, so the distributed code path can be
// tested in a serial build and gives exactly the serial answer.
#pragma once

#include "dualmesh/core/Types.h"

#include <string>
#include <vector>

namespace dualmesh
{

class Communicator
{
public:
  /// The communicator of all ranks (MPI_COMM_WORLD), or a single-rank
  /// communicator in a build without MPI.  MPI is initialized on first use if
  /// the program has not initialized it already, and finalized at exit.
  static Communicator & world();

  int rank() const { return _rank; }
  int size() const { return _size; }
  bool isRoot() const { return _rank == 0; }
  /// Whether this build can actually talk to other processes.
  static bool haveMpi();
  /// Shut MPI down, if this library started it.  Calling it twice is
  /// harmless.  A C++ program does not need to call it, because it is
  /// registered with atexit; the Python bindings do call it, from an interpreter
  /// exit hook, because a shared library's atexit handlers are removed when
  /// the module is unloaded and one rank would then leave MPI_Finalize
  /// unmatched, hanging the others.
  static void finalize();

  double sum(double value) const;
  double max(double value) const;
  Index sum(Index value) const;
  /// Sum @p values elementwise across all ranks, in place.
  void sumInPlace(std::vector<double> & values) const;
  /// Elementwise minimum across all ranks, in place.
  void minInPlace(std::vector<double> & values) const;
  /// True on every rank when it is true on any rank.
  bool any(bool value) const;
  void barrier() const;

  /// Send @p send[r] to rank r and receive rank r's message into @p recv[r].
  /// The sizes are exchanged first, so the caller does not need to know them.
  void exchange(const std::vector<std::vector<double>> & send,
                std::vector<std::vector<double>> & recv) const;
  void exchange(const std::vector<std::vector<Index>> & send,
                std::vector<std::vector<Index>> & recv) const;

  /// Gather one value from every rank onto every rank.
  std::vector<Index> allGather(Index value) const;

private:
  Communicator();
  int _rank = 0;
  int _size = 1;
};

} // namespace dualmesh
