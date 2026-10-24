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
#include <Eigen/Dense>
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

/// One entry of a distributed matrix, in the global numbering.
struct GlobalEntry
{
  Index row;
  Index col;
  double value;
};

/// The block preconditioner of a flow (SaddlePointSolver.h) in the global
/// numbering of a system: every process gives the rows it owns of the two
/// blocks and the entries of the operators that it computed, which are summed
/// over the processes like those of the matrix.
struct SchurBlocks
{
  SchurApproximation approximation = SchurApproximation::PressureMass;
  int sign = 1;
  /// The owned rows of the first block (the velocities and every other
  /// unknown) and of the pressure, in increasing order.
  std::vector<Index> other_rows;
  std::vector<Index> pressure_rows;
  /// The owned pressure rows whose equations are prescribed.
  std::vector<Index> constrained_pressure_rows;
  std::vector<GlobalEntry> mass;
  std::vector<GlobalEntry> stabilization;
  std::vector<GlobalEntry> commutator_stabilization;
  std::vector<GlobalEntry> convection_diffusion;
  /// The lumped masses, as entries (row, row, value).
  std::vector<GlobalEntry> lumped_mass;
};

/// The blocks of a problem in the global numbering of its system: unknown i of
/// the problem is the global row @p global_dof[i], and this process owns it
/// when @p owned[i] is set.
SchurBlocks globalSchurBlocks(const SaddlePointBlocks & blocks,
                              const std::vector<Index> & global_dof,
                              const std::vector<char> & owned);

struct Settings
{
  /// PETSc option string; empty selects the default described above.
  std::string options;
  double relative_tolerance = 1e-12;
  double absolute_tolerance = 1e-50;
  int max_iterations = 5000;
  /// Degrees of freedom per node (the number of variables), which the
  /// dualmesh numbering interlaces.  It is given to PETSc as the matrix block
  /// size, which the block-aware parts of algebraic multigrid use.
  int block_size = 1;
  /// The block preconditioner of a flow: PETSc's Schur complement field split
  /// with the velocities and the other unknowns as the first split ("0") and
  /// the pressure as the second ("1").  The defaults are flexible GMRES
  /// outside, the upper block-triangular factorisation, and hypre BoomerAMG
  /// (one V-cycle) on the first split; the Schur complement is preconditioned
  /// by the pressure mass matrix (with BoomerAMG), or by pcd or lsc, whose
  /// solves with A_p take one BoomerAMG V-cycle (options prefix
  /// "-fieldsplit_1_laplacian_").  `options` may override any of them, for
  /// instance "-fieldsplit_0_pc_type lu" for an exact momentum solve.
  const SchurBlocks * schur = nullptr;
  /// The owned rows of every block of a field split by physics (one block per
  /// group of coupled variables), solved one after the other, each with one
  /// algebraic multigrid cycle (GAMG, with the near null space below
  /// restricted to the block).  Not with `schur`.
  std::vector<std::vector<Index>> physics_blocks;
  /// The near null space of the matrix (the rigid body modes of an elastic
  /// body and the constant of every other variable), one column per vector
  /// and one row per owned row, which PETSc's algebraic multigrid (GAMG)
  /// takes for its coarse spaces.  Empty: none.
  Eigen::MatrixXd near_nullspace;
  /// LinearSolver only: the number of earlier solutions kept to start each solve from their best combination (the projection of Fischer 1998).
  /// The solutions are kept orthonormal in the energy norm of the matrix, so that the start is the combination closest to the solution in that norm, and the Krylov method computes only the remainder.
  /// Zero solves from the given guess instead.
  int projection_vectors = 0;
  /// Whether the matrix is symmetric positive definite: the projection then uses its energy norm, and otherwise the norm of the residual.
  bool symmetric = true;
};

/// Solve A x = b on one process.
Vector solve(const SparseMatrix & A, const Vector & b, const Settings & settings, Result & result);

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
  /// Not with a Schur complement split, whose operators belong to the first values.
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
