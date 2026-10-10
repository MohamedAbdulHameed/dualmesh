// SPDX-License-Identifier: LGPL-2.1-or-later
//
// Linear solves with PETSc (Balay et al., PETSc/TAO Users Manual).
//
// PETSc is optional.  A build configured with DUALMESH_ENABLE_PETSC finds it
// with pkg-config and compiles this interface; without it, `available()` is
// false and the solve functions throw an error that says how to enable it.
//
// Two entry points are provided.  `solve` takes a matrix held entirely by one
// process, which is how the serial Problem stores its Jacobian.  `solveDistributed`
// takes, from every process of MPI_COMM_WORLD, the rows it owns in a global
// numbering in which each process owns one contiguous range; the entries of a
// row may be given by several processes and are added together, which is how a
// matrix assembled element by element on a partitioned mesh arrives.
//
// The Krylov method and the preconditioner are chosen with PETSc's own option
// strings, for example "-ksp_type gmres -pc_type hypre -pc_hypre_type
// boomeramg" or "-ksp_type preonly -pc_type lu -pc_factor_mat_solver_type
// mumps".  The options of one solve are kept in a private options database, so
// they do not leak into the next solve or into other PETSc users in the same
// process.  The default, when no options are given, is GMRES with a direct
// factorisation (LU in serial, MUMPS in parallel when PETSc has it), because
// that is the robust choice for the non-symmetric and saddle-point matrices
// the dual mesh, finite volume and pressure-velocity discretisations produce.
#pragma once

#include "dualmesh/core/Types.h"
#include "dualmesh/linalg/SaddlePointSolver.h"

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include <memory>
#include <string>
#include <vector>

namespace dualmesh
{
namespace petsc
{

using Vector = Eigen::VectorXd;
using SparseMatrix = Eigen::SparseMatrix<double>;

/// Whether this build has PETSc.
bool available();

/// PETSc's version, "major.minor.subminor", or "" without PETSc.
std::string version();

/// Initialize PETSc if it is not yet initialized.  MPI must already be running
/// (the Communicator starts it).  Called by the solve functions.
void initialize();

/// Throw an error naming @p what when @p code, the result of a PETSc call, is not zero.
void check(int code, const char * what);
/// Finalize PETSc if this library initialized it.  Must run before MPI is
/// finalized; Communicator::finalize() calls it.
void finalize();

struct Result
{
  int iterations = 0;
  bool converged = false;
  /// PETSc's converged reason, as text (for example "CONVERGED_RTOL").
  std::string reason;
};

struct Settings
{
  /// PETSc option string; empty selects the default described above.
  std::string options;
  double relative_tolerance = 1e-12;
  double absolute_tolerance = 1e-50;
  int max_iterations = 5000;
  /// Degrees of freedom per node (the number of variables), which the
  /// dualmesh numbering interlaces.  It is given to PETSc as the matrix block
  /// size, which the field-split preconditioner (-pc_fieldsplit_0_fields 0,1
  /// -pc_fieldsplit_1_fields 2 for the velocities and the pressure of a 2D
  /// flow) and the block-aware parts of algebraic multigrid use.
  int block_size = 1;
  /// When set (serial solves only), the system is a saddle point problem and
  /// is solved with PETSc's Schur complement field split: the pressure
  /// unknowns form the second split, and the Schur complement is
  /// preconditioned by `saddle_point->schur_approximation` (the signed,
  /// viscosity-scaled pressure mass matrix).  The defaults are flexible
  /// GMRES outside, the upper block-triangular factorisation, an LU
  /// factorisation (MUMPS when available) of the momentum block and of the
  /// Schur approximation; `options` may override any of them, for instance
  /// "-fieldsplit_0_pc_type hypre" for algebraic multigrid on the momentum
  /// block.
  const SaddlePointBlocks * saddle_point = nullptr;
  /// LinearSolver only: the number of earlier solutions kept to start each solve from their best combination (the projection of Fischer 1998).
  /// The solutions are kept orthonormal in the energy norm of the matrix, so that the start is the combination closest to the solution in that norm, and the Krylov method computes only the remainder.
  /// Zero solves from the given guess instead.
  int projection_vectors = 0;
  /// Whether the matrix is symmetric positive definite: the projection then uses its energy norm, and otherwise the norm of the residual.
  bool symmetric = true;
};

/// Solve A x = b on one process.
Vector solve(const SparseMatrix & A, const Vector & b, const Settings & settings, Result & result);

/// One entry of a distributed matrix, in the global numbering.
struct GlobalEntry
{
  Index row;
  Index col;
  double value;
};

/// A linear system whose matrix is set up once and solved for many right-hand sides, as the substeps of a time step are: the preconditioner (the hierarchy of algebraic multigrid) is built at the first solve and kept.
/// The matrix is given as solveDistributed takes it, and this process owns the global rows [first_row, first_row + num_owned).
/// With @p distributed false, the system belongs to this process alone (first_row is zero and num_owned the global size).
/// Without PETSc, a system of one process is factored once with Eigen's sparse LU.
class LinearSolver
{
public:
  LinearSolver(bool distributed,
               Index global_size,
               Index first_row,
               Index num_owned,
               const std::vector<GlobalEntry> & entries,
               const Settings & settings);
  ~LinearSolver();
  LinearSolver(const LinearSolver &) = delete;
  LinearSolver & operator=(const LinearSolver &) = delete;
  /// Replace the values of the matrix, given in the order and at the positions of the entries it was made from; the structure, the Krylov method and its options are kept, and the preconditioner is rebuilt at the next solve.
  void setValues(const std::vector<double> & values);
  /// Solve for the owned part of x, starting from @p guess (the owned part of a first approximation) when it is given and no earlier solutions are projected.
  Vector solve(const Vector & b_owned, const Vector * guess, Result & result) const;

private:
  struct Impl;
  std::unique_ptr<Impl> _impl;
};

/// Solve a distributed system.  This process owns the global rows
/// [first_row, first_row + b_owned.size()); the ranges of all processes tile
/// [0, global_size).  `entries` may contain rows owned by other processes and
/// repeated positions; they are summed.  Returns the owned part of x.
Vector solveDistributed(Index global_size,
                        Index first_row,
                        const std::vector<GlobalEntry> & entries,
                        const Vector & b_owned,
                        const Settings & settings,
                        Result & result);

} // namespace petsc
} // namespace dualmesh
