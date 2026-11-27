// SPDX-License-Identifier: LGPL-2.1-or-later
//
// Linear and nonlinear solves with PETSc (Balay et al., PETSc/TAO Users Manual).
//
// A LinearSolver takes, from every process of its communicator, the rows it owns in a global
// numbering in which each process owns one contiguous range.  The entries of a row may be given by
// several processes and are added together, which is how a matrix assembled element by element on
// a partitioned mesh arrives.  A serial problem is the case of one process.
//
// The Krylov method and the preconditioner are chosen with PETSc's own option strings, for example
// "-ksp_type gmres -pc_type hypre -pc_hypre_type boomeramg" or "-ksp_type preonly -pc_type lu
// -pc_factor_mat_solver_type mumps".  The options of one solver are kept in a private options
// database, so they do not leak into the next solver or into other PETSc users in the same
// process.  The default, when no options are given, is GMRES with a direct factorisation (PETSc
// LU on one process, MUMPS on several, which needs PETSc built with MUMPS), the
// robust choice for the non-symmetric and saddle-point matrices that the
// dual mesh and pressure-velocity discretisations produce.
//
// solveNonlinear runs Newton's method with PETSc's SNES on the matrix and the Krylov solver of a
// LinearSolver, so that they persist from one Newton iteration to the next.
#pragma once

#include "dualmesh/core/Types.h"
#include "dualmesh/linalg/SaddlePointBlocks.h"

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/SparseCore>

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace dualmesh
{
namespace petsc
{

using Vector = Eigen::VectorXd;
using SparseMatrix = Eigen::SparseMatrix<double>;

/// The orthogonalization of the GMRES methods that the code chooses: classical Gram-Schmidt with a
/// second pass wherever the first loses orthogonality.  A single pass loses it as the residual
/// nears the tolerance 1e-12, and the iteration then stalls on rounding errors.
inline constexpr const char * gmresOrthogonalization =
    "-ksp_gmres_cgs_refinement_type refine_ifneeded";

/// The options of GMRES with one V-cycle of algebraic multigrid, preconditioned from the right so
/// that it measures the true residual (the equations of a coupled problem differ in scale by many
/// orders of magnitude, and a preconditioner that fails without an error would otherwise report
/// convergence): smoothed aggregation (GAMG), which takes the rigid body modes of a solid, when
/// @p smoothed_aggregation, and otherwise hypre's BoomerAMG with the options of the field splits.
std::string multigridOptions(bool smoothed_aggregation);

/// PETSc's version, "major.minor.subminor".
std::string version();

/// Whether PETSc has MUMPS, the direct solver of a matrix distributed over several processes.
/// PETSc's own LU factorises a matrix on one process only.
bool haveMumps();

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

/// The block preconditioner of a flow (SaddlePointBlocks.h) in the global
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
  /// The factor that each owned row of the matrix is multiplied by, in the order of the rows: the
  /// matrix holds the equations scaled row by row, and the operators above belong to the unscaled
  /// equations.  The solver scales them to the matrix.
  Vector row_scale;
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

struct NonlinearProblem;
struct NonlinearSettings;
struct NonlinearResult;
class LinearSolver;
NonlinearResult solveNonlinear(const NonlinearProblem & problem,
                               LinearSolver & solver,
                               Vector & x,
                               const NonlinearSettings & settings);

/// A linear system whose matrix is set up once and solved for many right-hand sides, as the substeps of a time step are: the preconditioner (the hierarchy of algebraic multigrid) is built at the first solve and kept.
/// This process owns the global rows [first_row, first_row + num_owned); the ranges of all processes tile [0, global_size).
/// `entries` may contain rows owned by other processes and repeated positions, which are summed.
/// With @p distributed false, the system belongs to this process alone (first_row is zero and num_owned the global size).
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
  /// Replace the matrix by @p entries, with a new structure when their positions differ from the
  /// last ones.  The Krylov method, its options and the splits are kept; refresh() sets the
  /// preconditioner up on the new matrix.
  void setEntries(const std::vector<GlobalEntry> & entries);
  /// Set the preconditioner up on the current matrix, with the operators of the Schur complement
  /// split @p schur built from it (required when the solver was made with a Schur split).
  void refresh(const SchurBlocks * schur = nullptr);
  /// Solve for the owned part of x, starting from @p guess (the owned part of a first approximation) when it is given and no earlier solutions are projected.
  Vector solve(const Vector & b_owned, const Vector * guess, Result & result) const;

private:
  struct Impl;
  std::unique_ptr<Impl> _impl;
  friend NonlinearResult solveNonlinear(const NonlinearProblem & problem,
                                        LinearSolver & solver,
                                        Vector & x,
                                        const NonlinearSettings & settings);
};

/// What the test of an iteration of a nonlinear solve decides.
enum class IterationStatus
{
  iterate,
  converged,
  diverged,
};

/// The problem of a nonlinear solve.  Its state is the owned rows of the global numbering of the
/// LinearSolver that solves its linear systems.
struct NonlinearProblem
{
  /// The residual at the state x into r and its scaled norm, the norm that the line search lowers
  /// (each equation over the size of its terms).  False when the equations cannot be evaluated at
  /// x on some process: the line search then shortens the step, and without it the solve stops.
  std::function<bool(const Vector & x, Vector & r, double & norm)> residual;
  /// Give the Jacobian at x, the state of the last residual, to the solver (setEntries, then
  /// refresh).  False when it cannot be evaluated there (an entry that is not finite).
  std::function<bool(const Vector & x, LinearSolver & solver)> jacobian;
  /// The test after the residual at the state x of iteration `iteration` (0 at the start), with
  /// `linear_iterations` Krylov iterations so far.  It also sets the norm that the line search
  /// starts from.
  std::function<IterationStatus(
      int iteration, const Vector & x, int linear_iterations, double & norm)>
      test;
};

struct NonlinearSettings
{
  int max_iterations = 50;
  /// Backtracking on the scaled norm of the residual: the first of the step lengths 1, 1/2, 1/4,
  /// ... (at most max_line_search_steps halvings) at which it falls to (1 - 1e-4 a) of its value.
  /// Otherwise the full step.
  bool line_search = false;
  int max_line_search_steps = 10;
  /// Without the line search: the weight gamma of the last iterate in each new one,
  /// U^{r+1} = U^r + (1 - gamma) dU, from the second step on (the relaxation of direct iteration).
  /// With line_search, 1 - gamma is the first trial of the backtracking.
  double relaxation = 0.0;
  /// The tolerance of each linear solve of Eisenstat and Walker (SIAM J. Sci. Comput. 17 (1996)
  /// 16-32), which follows the decrease of the residual; otherwise that of the LinearSolver.
  bool eisenstat_walker = false;
  /// The user's options of the nonlinear solver (those that start with -snes_).
  std::string options;
};

struct NonlinearResult
{
  bool converged = false;
  /// Newton steps and Krylov iterations.
  int iterations = 0;
  int linear_iterations = 0;
  /// PETSc's reason, as text, and the causes that stop an iteration.
  std::string reason;
  bool linear_solve_failed = false;
  bool line_search_failed = false;
  bool domain_error = false;
};

/// Run Newton's method on @p problem from the state @p x, with the linear systems solved by
/// @p solver.  @p x holds the last state on return.  The test of the problem decides convergence.
NonlinearResult solveNonlinear(const NonlinearProblem & problem,
                               LinearSolver & solver,
                               Vector & x,
                               const NonlinearSettings & settings);

} // namespace petsc
} // namespace dualmesh
