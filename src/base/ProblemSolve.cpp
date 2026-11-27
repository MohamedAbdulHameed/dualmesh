// SPDX-License-Identifier: LGPL-2.1-or-later
//
// The linear and the nonlinear solves of a problem with PETSc, the same for a serial problem and a
// distributed one: both number their unknowns in the global numbering of Parallel::globalRows.
#include "dualmesh/base/Console.h"
#include "dualmesh/base/Problem.h"
#include "dualmesh/linalg/Communicator.h"
#include "dualmesh/linalg/LinearSolverChoice.h"
#include "dualmesh/linalg/PetscSolver.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>
#include <optional>
#include <sstream>

namespace dualmesh
{

GlobalRows
Parallel::globalRows(const Problem & problem) const
{
  // One process holds the whole problem, numbered as it is.
  GlobalRows rows;
  rows.global_size = rows.num_owned = problem.numDofs();
  rows.global_dof.resize(static_cast<std::size_t>(problem.numDofs()));
  std::iota(rows.global_dof.begin(), rows.global_dof.end(), Index(0));
  rows.owned_dof.assign(static_cast<std::size_t>(problem.numDofs()), 1);
  return rows;
}

namespace
{
/// The entries of @p A, a matrix of the local unknowns of @p problem, in the global numbering,
/// with every row multiplied by its factor in @p row_factor.  The matrix holds the contributions of
/// this process's elements, and the global matrix is their sum, in which joined nodes (periodic
/// boundaries) share one number.  The row of a prescribed unknown is a row of the diagonal on
/// every process that holds it, with the same entry, so it is entered once, by the owner.
std::vector<petsc::GlobalEntry>
globalEntries(const Problem & problem,
              const SparseMatrix & A,
              const GlobalRows & rows,
              const Vector & row_factor)
{
  const std::vector<char> fixed = problem.constrainedDofs();
  std::vector<petsc::GlobalEntry> entries;
  entries.reserve(static_cast<std::size_t>(A.nonZeros()));
  for (Index col = 0; col < A.outerSize(); ++col)
    for (SparseMatrix::InnerIterator it(A, col); it; ++it)
    {
      const Index row = it.row();
      const Index global_row = rows.global_dof[row], global_col = rows.global_dof[it.col()];
      if (global_row < 0 || global_col < 0)
        continue;
      if (fixed[row])
      {
        if (row == it.col() && rows.owned_dof[row])
          entries.push_back({global_row, global_row, row_factor[row] * it.value()});
        continue;
      }
      entries.push_back({global_row, global_col, row_factor[row] * it.value()});
    }
  return entries;
}

/// The owned entries of @p v, a vector of the local unknowns, in the order of the global numbering.
Vector
ownedPart(const Vector & v, const GlobalRows & rows)
{
  Vector owned = Vector::Zero(rows.num_owned);
  for (Index i = 0; i < v.size(); ++i)
    if (rows.owned_dof[i] && rows.global_dof[i] >= 0)
      owned[rows.global_dof[i] - rows.first_row] = v[i];
  return owned;
}

/// The rows of @p m, one per local unknown, that this process owns, in the order of the global
/// numbering.
Eigen::MatrixXd
ownedRows(const Eigen::MatrixXd & m, const GlobalRows & rows)
{
  Eigen::MatrixXd owned = Eigen::MatrixXd::Zero(rows.num_owned, m.cols());
  for (Index i = 0; i < m.rows(); ++i)
    if (rows.owned_dof[i] && rows.global_dof[i] >= 0)
      owned.row(rows.global_dof[i] - rows.first_row) = m.row(i);
  return owned;
}

/// The owned global rows of every block of the split by physics (Problem::physicsBlocks).
std::vector<std::vector<Index>>
physicsBlockRows(const Problem & problem, const GlobalRows & rows)
{
  const std::vector<std::vector<Index>> blocks = problem.physicsBlockDofs();
  std::vector<std::vector<Index>> out(blocks.size());
  for (std::size_t k = 0; k < blocks.size(); ++k)
    for (Index i : blocks[k])
      if (rows.owned_dof[i] && rows.global_dof[i] >= 0)
        out[k].push_back(rows.global_dof[i]);
  for (auto & block : out)
  {
    std::sort(block.begin(), block.end());
    block.erase(std::unique(block.begin(), block.end()), block.end());
  }
  return out;
}

/// The blocks of the Schur complement split of the flow of @p problem, whose Jacobian is @p A, in
/// the global numbering, for the matrix whose rows are multiplied by @p row_factor.
petsc::SchurBlocks
globalSchurBlocks(const Problem & problem,
                  const SparseMatrix & A,
                  SchurApproximation approximation,
                  const GlobalRows & rows,
                  const Vector & row_factor)
{
  const Parallel & par = problem.parallel();
  SaddlePointBlocks blocks = problem.flowSystem().saddlePointBlocks(problem, A, approximation);
  if (par.any(!blocks.refusal.empty()))
    throw InputError(!blocks.refusal.empty()
                         ? blocks.refusal
                         : "The Schur complement splits of a flow (lsc, pcd, pressure_mass) scale "
                           "their operators by the dynamic_viscosity, which must be positive "
                           "everywhere.");
  // The sign of the Schur complement, from the sum over the processes of their products of the
  // coupling blocks.  It is not the product of the summed blocks, but it has its sign: each part
  // couples the pressure to the velocity with the same sign.
  const double coupling = par.sum(blocks.coupling);
  if (coupling == 0.0)
    throw InputError("The pressure '" + blocks.pressure_variable +
                     "' is not coupled to the other variables in this system, so the Schur "
                     "complement split of a flow does not apply.");
  blocks.sign = coupling < 0.0 ? 1 : -1;
  petsc::SchurBlocks global = petsc::globalSchurBlocks(blocks, rows.global_dof, rows.owned_dof);
  global.row_scale = ownedPart(row_factor, rows);
  return global;
}

linear_system_traits
systemTraits(const Problem & problem, const GlobalRows & rows, const std::string & user_linear)
{
  linear_system_traits traits;
  traits.dimension = problem.mesh().dimension();
  traits.global_unknowns = rows.global_size;
  traits.flow = problem.flowConstraint() != nullptr;
  traits.mixed_order = problem.hasMixedOrder();
  traits.variables = problem.numVariables();
  for (const auto & group : problem.solidSystem().displacementGroups(problem))
    traits.displacement_variables += static_cast<int>(group.size());
  traits.user_options = user_linear;
  return traits;
}

/// The settings of @p attempt for the matrix @p A, whose rows are multiplied by @p row_factor.
/// @p schur keeps the Schur blocks that the settings point to.
petsc::Settings
attemptSettings(const Problem & problem,
                const linear_solve_attempt & attempt,
                const SparseMatrix & A,
                const GlobalRows & rows,
                const Vector & row_factor,
                const std::string & user_linear,
                std::optional<petsc::SchurBlocks> & schur)
{
  petsc::Settings settings;
  settings.block_size = problem.numVariables();
  switch (attempt.kind)
  {
  case linear_solve_kind::user_options:
    settings.options = user_linear;
    // The user's options may choose GAMG, which takes the rigid body modes of a solid.
    if (!problem.solidSystem().displacementGroups(problem).empty())
      settings.near_nullspace = ownedRows(problem.solidSystem().nearNullspace(problem), rows);
    break;
  case linear_solve_kind::direct:
    // The direct attempt factorises with MUMPS on several processes, because PETSc's own LU
    // factorises a matrix on one process only.
    if (rows.distributed && Communicator::world().size() > 1 && !petsc::haveMumps())
      throw InputError("A direct solve on several processes needs PETSc built with MUMPS, which "
                       "this PETSc lacks. Configure PETSc with --download-mumps, or give the "
                       "linear solver options of an iterative method.");
    break;
  case linear_solve_kind::Schur_split:
    schur = globalSchurBlocks(problem, A, *attempt.Schur_approximation, rows, row_factor);
    settings.schur = &*schur;
    settings.options = user_linear;
    break;
  case linear_solve_kind::physics_blocks:
    settings.options = user_linear;
    settings.physics_blocks = physicsBlockRows(problem, rows);
    settings.near_nullspace = ownedRows(problem.solidSystem().nearNullspace(problem), rows);
    break;
  case linear_solve_kind::GAMG:
    settings.options = petsc::multigridOptions(true);
    settings.near_nullspace = ownedRows(problem.solidSystem().nearNullspace(problem), rows);
    break;
  case linear_solve_kind::BoomerAMG:
    settings.options = petsc::multigridOptions(false);
    break;
  }
  return settings;
}

/// The end of the message of a matrix that is not finite.
const char * const notFinite =
    "has NaN or infinite entries: a property or a term is not finite at the current state (for "
    "example the square root of zero has an infinite slope).";

/// Solve with the linear solvers that linear_solve_attempts chooses for @p problem, in order.
/// For each, @p prepare gives the matrix and the factors of its rows (it may assemble the matrix
/// again), and @p solve solves with the settings of the attempt and returns the reason of a
/// failure, empty after a solve that converged.  An attempt that may fail stops early, so that
/// the next one is not delayed, and its failure moves on to the next one.  The failure of the last
/// throws.
template <typename Prepare, typename Solve>
void
forEachLinearAttempt(const Problem & problem,
                     const GlobalRows & rows,
                     const SolverOptions & o,
                     const user_options_parts & user,
                     Prepare && prepare,
                     Solve && solve)
{
  const Parallel & par = problem.parallel();
  const std::vector<linear_solve_attempt> attempts =
      linear_solve_attempts(systemTraits(problem, rows, user.linear), user.split);
  for (std::size_t k = 0; k < attempts.size(); ++k)
  {
    const linear_solve_attempt & attempt = attempts[k];
    const bool must_converge = attempt.must_converge || k + 1 == attempts.size();
    Vector row_factor;
    const SparseMatrix & A = prepare(row_factor);
    // A matrix with NaN or infinite entries makes every solver fail, so its cause is named first.
    if (par.any(!A.coeffs().allFinite()))
      throw std::runtime_error("dualmesh: the linear system " + std::string(notFinite) +
                               " Check the expressions of the properties and terms there.");
    std::optional<petsc::SchurBlocks> schur;
    petsc::Settings settings =
        attemptSettings(problem, attempt, A, rows, row_factor, user.linear, schur);
    settings.relative_tolerance = o.linear_tolerance;
    settings.max_iterations = must_converge ? 5000 : 500;
    const std::string failure = solve(attempt, settings, must_converge);
    if (failure.empty())
      return;
    if (must_converge)
      throw std::runtime_error("dualmesh: the linear solve did not converge (" + failure +
                               "). Choose another solver with PETSc_options, for example a "
                               "direct one: '-ksp_type preonly -pc_type lu'.");
    if (o.verbose && par.isRoot())
      std::cout << "  the linear solve (" << settings.options << ") did not converge (" << failure
                << "); trying the next solver\n";
  }
}

} // namespace

std::vector<int>
Problem::vectorEquations() const
{
  std::vector<int> equation(static_cast<std::size_t>(numVariables()));
  std::iota(equation.begin(), equation.end(), 0);
  const auto root = [&](int v)
  {
    while (equation[static_cast<std::size_t>(v)] != v)
      v = equation[static_cast<std::size_t>(v)];
    return v;
  };
  for (const auto & k : _kernels)
    for (const char * list : {"velocities", "velocity", "displacements"})
    {
      if (!k->parameters().has(list) ||
          k->parameters().all().at(list).kind != ParameterKind::StringList)
        continue;
      std::vector<int> components;
      for (const auto & name : k->parameters().getStringList(list))
        if (hasVariable(name))
          components.push_back(variableIndex(name));
      if (std::find(components.begin(), components.end(), k->variable()) == components.end())
        continue;
      for (int c : components)
      {
        const int a = root(c), b = root(components.front());
        equation[static_cast<std::size_t>(std::max(a, b))] = std::min(a, b);
      }
    }
  for (int v = 0; v < numVariables(); ++v)
    equation[static_cast<std::size_t>(v)] = root(v);
  return equation;
}

void
Problem::setOwnedPart(Vector & v, const Vector & owned, const GlobalRows & rows) const
{
  for (Index i = 0; i < v.size(); ++i)
    if (rows.owned_dof[i] && rows.global_dof[i] >= 0)
      v[i] = owned[rows.global_dof[i] - rows.first_row];
  parallel().copyFromOwners(v.data(), numVariables());
}

Vector
Problem::linearSolve(const SparseMatrix & A,
                     const Vector & b,
                     const SolverOptions & o,
                     int * iterations) const
{
  const Parallel & par = parallel();
  if (par.any(!b.allFinite()))
    throw std::runtime_error("dualmesh: the right-hand side of the linear system " +
                             std::string(notFinite));
  const GlobalRows rows = par.globalRows(*this);
  // The rows keep their sizes.  The rows of the transposed Jacobian of an adjoint belong to the
  // unknowns, and the mass matrix of the starting acceleration of a dynamic run has no reference
  // residual that would size them.
  const Vector unscaled = Vector::Ones(numDofs());
  const std::vector<petsc::GlobalEntry> entries = globalEntries(*this, A, rows, unscaled);
  const Vector b_owned = ownedPart(b, rows);
  Vector x = Vector::Zero(numDofs());
  forEachLinearAttempt(
      *this,
      rows,
      o,
      user_choice(o.petsc_options),
      [&](Vector & row_factor) -> const SparseMatrix &
      {
        row_factor = unscaled;
        return A;
      },
      [&](const linear_solve_attempt &, const petsc::Settings & settings, bool) -> std::string
      {
        const petsc::LinearSolver solver(
            rows.distributed, rows.global_size, rows.first_row, rows.num_owned, entries, settings);
        petsc::Result result;
        const Vector x_owned = solver.solve(b_owned, nullptr, result);
        if (iterations)
          *iterations += result.iterations;
        if (par.any(!result.converged))
          return result.reason + ", " + std::to_string(result.iterations) + " iterations";
        setOwnedPart(x, x_owned, rows);
        return {};
      });
  return x;
}

SolveResult
Problem::solveSteady(const SolverOptions & options)
{
  initialize();
  AssemblyOptions base;
  base.include_time_kernels = false;
  base.theta = 1.0;
  // A flow at rest starts from the Stokes solution, the same problem without the inertia.  The
  // linearisation about the state at rest, whose prescribed velocities next to zero values make
  // steep gradients, can send Newton's method far from the solution; the Stokes solution is the
  // usual first guess of the Navier-Stokes equations.  The equations are unchanged.
  SolveResult start;
  Vector initial = _U;
  if (options.nonlinear_solver != "linear" && _flow->atRest(*this))
  {
    if (options.verbose && parallel().isRoot())
      std::cout << "  The flow is at rest: Newton's method starts from the Stokes solution, the "
                   "same problem without the inertia.\n";
    const Vector at_rest = _U;
    AssemblyOptions stokes = base;
    stokes.inertia = 0.0;
    SolverOptions quiet = options;
    quiet.error_on_divergence = false;
    // The Schur complement of a Stokes problem is spectrally equivalent to the pressure mass
    // matrix, so pressure_mass is its Schur approximation.  pcd and lsc approximate the
    // convection that the Stokes problem does not have.
    const user_options_parts user = user_choice(options.petsc_options);
    if (user.split.Schur && *user.split.Schur != SchurApproximation::PressureMass)
      quiet.petsc_options = user.linear + " " + user.nonlinear + " -dualmesh_schur pressure_mass";
    start = nonlinearSolve(quiet, stokes, nullptr);
    if (!start.converged)
      _U = at_rest;
    initial = at_rest;
  }
  // A continuation ramps the prescribed values from the state the solve was given (a flow from
  // rest, which takes it through the lower Reynolds numbers).
  SolveResult result = nonlinearSolve(options, base, nullptr, &initial);
  result.total_iterations += start.total_iterations;
  result.linear_iterations += start.linear_iterations;
  result.history.insert(result.history.begin(), start.history.begin(), start.history.end());
  // A converged steady state is a state the history of stateful property objects
  // has reached: commit it, so that a following solve or time step starts
  // from it.  Without this, a transient run after a steady solve would
  // restart the history from the undeformed state and take the whole
  // steady deformation as its first increment.
  if (result.converged)
    commitState();
  return result;
}

// ---------------------------------------------------------------------------
// The nonlinear solve, serial or distributed
// ---------------------------------------------------------------------------

namespace
{
/// Refuse an unknown nonlinear solver or line search.
void
checkNonlinearOptions(const SolverOptions & o)
{
  if (o.nonlinear_solver != "newton" && o.nonlinear_solver != "picard" &&
      o.nonlinear_solver != "linear")
    throw InputError("Unknown nonlinear solver '" + o.nonlinear_solver +
                     "' (use newton, picard, or linear).");
  if (o.line_search != "automatic" && o.line_search != "none" && o.line_search != "backtracking")
    throw InputError("Unknown line_search '" + o.line_search +
                     "' (use automatic, none or backtracking).");
}
} // namespace

/// The steps of one nonlinear solve (Problem::nonlinearSolve) and the state they share: the
/// residual with the sizes of its terms, the scales of the equations, and the state of the attempt
/// under way, which the callbacks of PETSc read and write.
class Problem::NonlinearSolve
{
public:
  /// The outcome of one attempt.
  struct Outcome
  {
    bool converged = false, diverged = false, grew = false;
    /// The line search found no step length that lowers the residual.
    bool line_search_failed = false;
    /// One linear solve with nonlinear_solver "linear" left a residual above the tolerance.
    bool not_linear = false;
    double r0 = -1;
  };

  /// The solve of @p problem with the options @p o, whose iterations and history go to
  /// @p result.  Every residual assembles with @p base and adds the steady terms of the old time
  /// @p steady_old when it is given.
  NonlinearSolve(Problem & problem,
                 const SolverOptions & o,
                 const AssemblyOptions & base,
                 const OldResidual * steady_old,
                 SolveResult & result);
  /// One solve with the driving data at the fraction @p lf of their values (1: the problem
  /// itself), the step @p step_index of a continuation.
  Outcome attempt(double lf, int step_index);
  /// The continuation that follows a failed solve of the problem itself: the data go from the
  /// state @p ramp_from to their values in steps.  Returns the outcome of the last step, and sets
  /// @p done to the fraction of the data that the last converged step reached.
  Outcome continuation(const Vector & ramp_from, double & done);
  /// The message of the solve that failed with @p last, after a continuation to @p done when
  /// @p continuation.
  std::string failureMessage(const Outcome & last, bool continuation, double done) const;

private:
  /// A state where a property cannot be evaluated (an element turned inside out) fails on every
  /// process together: the line search shortens the step, or continuation takes a smaller one.
  struct InadmissibleState
  {
  };
  /// The state of one attempt, which the callbacks of PETSc share.
  struct AttemptState
  {
    double lf = 1.0;
    int step_index = 1;
    bool print = false;
    Outcome out;
    GlobalRows rows;
    /// The factors of the rows of the residual last handed to PETSc, which its Jacobian takes.
    Vector row_factor;
    /// Whether J was assembled at the state of the last residual, and whether the linear solver
    /// already holds J (it is made from the Jacobian at the start of each solve).
    bool jacobian_current = true;
    bool solver_current = false;
    std::optional<SchurApproximation> schur_approximation;
    /// The last iterate, which a failed solve returns to, and the record of its row of the
    /// history.
    Vector iterate;
    std::size_t record = 0;
    bool have_record = false;
    std::vector<double> first_sizes;
    double first_growth = 0.0;
    /// Newton steps of the solves with earlier linear solvers, and the Krylov iterations so far.
    int steps_before = 0;
    int linear_before = 0;
  };

  /// The automatic choice of the line search: backtracking where a kernel needs it
  /// (AdvectiveKernel::needsLineSearch), for Newton's method.
  static bool lineSearch(const Problem & problem, const SolverOptions & o);
  /// The residual at a state, summed over the processes, with the rows of the prescribed values,
  /// and its reference: the size of the terms of each equation (AssemblyOptions::reference), zero
  /// at the prescribed values.  Throws InadmissibleState.
  void residualAt(const Vector & U, Vector & R, SparseMatrix * J);
  /// The norm of the residuals @p norms of the equations, each over its scale.
  double scaledNorm(const std::vector<double> & norms) const;
  /// The factors of the rows of the linear solves.
  Vector rowFactors() const;
  /// The largest residual of an equation over its size at the first iterate.
  double relativeToFirst(const std::vector<double> & norms) const;
  /// The residual and the Jacobian at the start.  False at an inadmissible state.
  bool startingSystem();
  /// The check of the singularity of the Jacobian, when its pattern is new.
  void checkNewPattern();
  /// The callbacks of PETSc (petsc::NonlinearProblem).
  bool iterateResidual(const Vector & x, Vector & r, double & norm);
  bool refreshJacobian(petsc::LinearSolver & solver);
  petsc::IterationStatus iterationTest(int it, int linear_total, double & norm);
  /// Whether every equation has converged, with the residuals @p norms and the sizes @p sizes of
  /// the variables.
  bool equationsConverged(const std::vector<double> & norms,
                          const std::vector<double> & sizes) const;
  /// Newton's method with each linear solver in turn (forEachLinearAttempt).
  void solveWithLinearSolvers();
  /// The Jacobian that an attempt of a linear solver starts from, and the factors of its rows.
  const SparseMatrix & attemptJacobian(Vector & factors);
  /// Newton's method from @p x with the linear solver @p choice and its settings @p settings.
  /// Returns the reason of a failed linear solve, empty otherwise.
  std::string newtonSolve(const petsc::NonlinearProblem & problem,
                          const linear_solve_attempt & choice,
                          const petsc::Settings & settings,
                          bool must_converge,
                          Vector & x);
  /// An enclosed flow reports its pressure with a mean of zero, and its reactions at the shifted
  /// state (FlowSystem::settlePressureLevels).
  void settlePressureLevels();

  Problem & _p;
  const SolverOptions & _o;
  AssemblyOptions _base;
  const OldResidual * const _steady_old;
  const Parallel & _par;
  const bool _picard, _linear, _line_search;
  const user_options_parts _user;
  /// The reference of the last residual (residualAt).
  Vector _reference;
  /// The size of the terms of each entry, inside its fluxes and sources (Kernel::fluxTermSize),
  /// whose round-off is the floor of the test where the terms cancel.
  Vector _term_size;
  /// Why the last state was inadmissible, empty when it was not.
  std::string _inadmissible;
  /// Each equation is measured against the size of its own terms, so the tests have no unit: the
  /// scaled norm adds the residual of every equation over the largest reference it has had in
  /// this attempt (the line search compares states with it, and the divergence test the
  /// iterations).
  std::vector<double> _scales;
  const std::vector<int> _equation_of;
  SolveResult & _result;
  Vector _R;
  SparseMatrix _J;
  AttemptState _a;
};

Problem::NonlinearSolve::NonlinearSolve(Problem & problem,
                                        const SolverOptions & o,
                                        const AssemblyOptions & base,
                                        const OldResidual * steady_old,
                                        SolveResult & result)
    : _p(problem), _o(o), _base(base), _steady_old(steady_old), _par(problem.parallel()),
      _picard(o.nonlinear_solver == "picard"), _linear(o.nonlinear_solver == "linear"),
      _line_search(lineSearch(problem, o)), _user(user_choice(o.petsc_options)),
      _scales(static_cast<std::size_t>(problem.numVariables()), 0.0),
      _equation_of(problem.vectorEquations()), _result(result)
{
  _base.mode = _picard ? LinearizationMode::Picard : LinearizationMode::Newton;
}

bool
Problem::NonlinearSolve::lineSearch(const Problem & problem, const SolverOptions & o)
{
  bool needed = false;
  for (const auto & k : problem._kernels)
    if (const auto * advection = dynamic_cast<const AdvectiveKernel *>(k.get()))
      needed = needed || advection->needsLineSearch();
  return o.nonlinear_solver == "newton" &&
         (o.line_search == "backtracking" || (o.line_search == "automatic" && needed));
}

SolveResult
Problem::nonlinearSolve(const SolverOptions & o,
                        AssemblyOptions base,
                        const OldResidual * steady_old,
                        const Vector * ramp_from)
{
  checkNonlinearOptions(o);
  SolveResult result;
  NonlinearSolve solve(*this, o, base, steady_old, result);
  // The problem itself first.  When it fails, a steady solve takes the data from the starting
  // state to their values in steps (continuation, as COMSOL's stationary solver does): the
  // sources, body forces and prescribed fluxes and loads from zero (Object::isLoad), and the
  // prescribed values from those of the starting state.
  const bool continuation = continuationHasData(ramp_from) && o.nonlinear_solver != "linear";
  NonlinearSolve::Outcome last = solve.attempt(1.0, 1);
  double done = 0.0;
  if (!last.converged && continuation)
    last = solve.continuation(*ramp_from, done);
  result.converged = last.converged;
  if (!last.converged && o.error_on_divergence)
    throw std::runtime_error(solve.failureMessage(last, continuation, done));
  return result;
}

bool
Problem::continuationHasData(const Vector * ramp_from) const
{
  // A continuation needs data to ramp: an object that is data, or a prescribed value that
  // differs from the starting state.
  if (!ramp_from)
    return false;
  bool rampable = false;
  for (const auto & k : _kernels)
    rampable = rampable || k->isLoad();
  for (const auto & bc : _ibcs)
    rampable = rampable || bc->isLoad();
  rampable = rampable || !_loads.empty();
  Vector prescribed = *ramp_from;
  applyDirichlet(prescribed, 1.0);
  rampable = rampable || prescribed != *ramp_from;
  return parallel().any(rampable);
}

Problem::NonlinearSolve::Outcome
Problem::NonlinearSolve::attempt(double lf, int step_index)
{
  _base.data_fraction = lf;
  _inadmissible.clear();
  _p.applyDirichlet(_p._U, lf);
  _p._linear_dt = _base.include_time_kernels ? _base.dt : 0.0;
  _p._linear_theta = _base.include_steady_terms ? _base.theta : 0.0;
  _a = AttemptState{};
  _a.lf = lf;
  _a.step_index = step_index;
  _a.print = _o.verbose && _par.isRoot();
  if (_a.print)
    console::nonlinearHeader(_o.nonlinear_solver == "newton"
                                 ? "Newton"
                                 : (_picard ? "Picard iteration" : "linear solve"),
                             step_index,
                             lf);
  std::fill(_scales.begin(), _scales.end(), 0.0);
  if (!startingSystem())
  {
    _a.out.diverged = true;
    return _a.out;
  }
  checkNewPattern();
  _a.rows = _par.globalRows(_p);
  _a.iterate = _p._U;
  if (!(_o.relaxation >= 0.0 && _o.relaxation < 1.0))
    throw InputError("relaxation is the weight gamma of the last iterate, from 0 (none) to "
                     "below 1: the update takes the fraction 1 - gamma of each step.");
  solveWithLinearSolvers();
  // A failed solve returns to its last iterate, which a trial of the line search may have left.
  if (!_a.out.converged)
  {
    _p._U = _a.iterate;
    return _a.out;
  }
  settlePressureLevels();
  return _a.out;
}

Problem::NonlinearSolve::Outcome
Problem::NonlinearSolve::continuation(const Vector & ramp_from, double & done)
{
  // A step that converges doubles the next one, a step that fails is halved and taken again from
  // the last converged state.  The continuation starts from the state its values are ramped from
  // (a flow at rest, not the Stokes solution of the full data that Newton's method was started
  // from).
  _p._ramp_from = ramp_from;
  Vector at_done = ramp_from;
  double step = 0.5;
  const double smallest = 1.0 / 1024.0;
  int step_index = 1;
  Outcome last;
  while (done < 1.0 && step >= smallest)
  {
    const double lf = std::min(1.0, done + step);
    _p._U = at_done;
    last = attempt(lf, ++step_index);
    if (last.converged)
    {
      done = lf;
      at_done = _p._U;
      step *= 2.0;
    }
    else
      step *= 0.5;
  }
  _p._ramp_from.resize(0);
  if (!last.converged)
  {
    // The state at the last data reached, and the prescribed values of the problem itself.
    _p._U = at_done;
    _p.applyDirichlet(_p._U, 1.0);
  }
  return last;
}

std::string
Problem::NonlinearSolve::failureMessage(const Outcome & last, bool continuation, double done) const
{
  std::ostringstream os;
  const std::string ramped =
      continuation ? (done > 0.0 ? " Continuation ramped the data up to " + std::to_string(done) +
                                       " of their values, and no smaller step went further."
                                 : " Continuation from the starting state did not converge "
                                   "at any fraction of the data.")
                   : "";
  if (!_inadmissible.empty())
    os << "dualmesh: the nonlinear solve stopped: " << _inadmissible << ramped;
  else if (last.grew)
    os << "dualmesh: Newton's method diverged: the residual grew from " << last.r0
       << " to more than 1e6 times that." << ramped
       << " Either the start is too far from the solution, or the problem has no solution "
          "(check the data). Start from a closer state, refine the mesh, or use "
          "nonlinear_solver 'Picard'.";
  else if (last.line_search_failed)
    os << "dualmesh: the nonlinear solve stopped: the line search found no step length, "
          "down to 2^-"
       << _o.max_line_search_steps << " of the Newton step, that lowers the residual." << ramped
       << " A property with a kink, a singular Jacobian or a problem without a solution "
          "causes this. Check the data, or use nonlinear_solver 'Picard'.";
  else if (last.not_linear)
    os << "dualmesh: nonlinear_solver 'linear' takes one linear solve, and its residual is "
          "above relative_tolerance: either the problem is nonlinear (use nonlinear_solver "
          "'Newton', the default) or the linear solver did not converge.";
  else if (last.diverged)
    os << "dualmesh: the nonlinear solve stopped: the residual or the Jacobian has NaN or "
          "infinite entries."
       << ramped
       << " Either Newton's method diverged (start from a closer state, or refine the "
          "mesh), or a property or its derivative is not finite at the current solution "
          "(for example log(0), 1/0, or x^p with p < 1 at x = 0).";
  else
    os << "dualmesh: nonlinear solve did not converge within " << _o.max_iterations
       << " iterations." << ramped;
  return os.str();
}

void
Problem::NonlinearSolve::residualAt(const Vector & U, Vector & R, SparseMatrix * J)
{
  Vector lagged = U;
  AssemblyOptions ao = _base;
  ao.lagged = &lagged;
  ao.reference = &_reference;
  ao.term_size = &_term_size;
  bool failed = false;
  try
  {
    _p.assemble(U, ao, R, J);
  }
  catch (const InadmissibleStateError & e)
  {
    failed = true;
    _inadmissible = e.what();
  }
  if (_par.any(failed))
    throw InadmissibleState{};
  if (_steady_old)
  {
    R += _steady_old->residual;
    _reference += _steady_old->reference;
    if (_steady_old->term_size.size() == _term_size.size())
      _term_size += _steady_old->term_size;
  }
  // The equation of a hanging node or a periodic copy goes to its primaries on each process
  // before the processes add their parts, as the Jacobian's does: the solver reads a row from
  // the process that owns it, which need not hold the constraint that folds into the row.
  // The last residual is kept both ways: unfolded for the pressure-level check, which reads
  // the equations of the dependent nodes, and folded for the reactions.
  Vector folded = R;
  _p.foldDependents(folded, nullptr);
  for (Vector * v : {&R, &folded, &_reference, &_term_size})
    _par.addAtNodes(v->data(), _p.numVariables());
  _p._last_residual = R;
  _p._last_folded_residual = folded;
  R = std::move(folded);
  _p.dirichletRows(R, J);
  const std::vector<char> fixed = _p.constrainedDofs();
  for (Index d = 0; d < _reference.size(); ++d)
    if (fixed[static_cast<std::size_t>(d)])
      _reference[d] = _term_size[d] = 0.0;
}

double
Problem::NonlinearSolve::scaledNorm(const std::vector<double> & norms) const
{
  double sum = 0.0;
  for (std::size_t v = 0; v < norms.size(); ++v)
    if (_scales[v] > 0.0)
      sum += (norms[v] / _scales[v]) * (norms[v] / _scales[v]);
  return std::sqrt(sum);
}

Vector
Problem::NonlinearSolve::rowFactors() const
{
  // The linear solves take every equation over its size, so that their test, the residual over
  // the right-hand side, has no unit: the rows of each equation, those of its prescribed values
  // included, are multiplied by one over the size of its terms at the state of the last residual
  // or the largest it has had in this attempt, whichever is larger, with the components of a
  // vector unknown together as in the convergence test.  A positive factor per equation leaves
  // the solution and the block structure of the matrix as they are.  The factors change only
  // while the size of an equation grows, so the tolerance of Eisenstat and Walker compares the
  // residuals of successive iterates in one norm once the sizes have settled.  An equation whose
  // terms have all been zero so far keeps its rows unscaled.
  const std::vector<double> sizes = _p.equationNorms(_reference);
  std::vector<double> equation_size(sizes.size(), 0.0);
  for (std::size_t v = 0; v < sizes.size(); ++v)
  {
    const double size = std::max(_scales[v], sizes[v]);
    equation_size[static_cast<std::size_t>(_equation_of[v])] += size * size;
  }
  const int nv = _p.numVariables();
  Vector factor(_p.numDofs());
  for (Index d = 0; d < _p.numDofs(); ++d)
  {
    const double size = std::sqrt(
        equation_size[static_cast<std::size_t>(_equation_of[static_cast<std::size_t>(d % nv)])]);
    factor[d] = size > 0.0 ? 1.0 / size : 1.0;
  }
  return factor;
}

double
Problem::NonlinearSolve::relativeToFirst(const std::vector<double> & norms) const
{
  double largest = 0.0;
  for (std::size_t v = 0; v < norms.size(); ++v)
    if (_a.first_sizes[v] > 0.0)
      largest = std::max(largest, norms[v] / _a.first_sizes[v]);
  return largest;
}

bool
Problem::NonlinearSolve::startingSystem()
{
  // An enclosed flow fixes the level of its pressure at one unknown, which changes the rows of
  // the prescribed values, so they are taken again.
  try
  {
    residualAt(_p._U, _R, &_J);
    if (_p._flow->holdPressureLevels(_p, _J))
      residualAt(_p._U, _R, &_J);
  }
  catch (const InadmissibleState &)
  {
    return false;
  }
  return true;
}

void
Problem::NonlinearSolve::checkNewPattern()
{
  // The singularity check runs when the pattern of the Jacobian changes, which the number of rows
  // or entries is a cheap signature of: a constraint that appears folds rows into others, an
  // interface condition starts to couple two bodies, and a re-initialization after a mesh change
  // resets the check outright.  It needs the whole matrix.
  if (_par.checksSingularity() &&
      (!_p._singularity_checked || _p._singularity_pattern_rows != _J.rows() ||
       _p._singularity_pattern_nonzeros != _J.nonZeros()))
  {
    _p.checkSingularity(_J);
    _p._singularity_checked = true;
    _p._singularity_pattern_rows = _J.rows();
    _p._singularity_pattern_nonzeros = _J.nonZeros();
  }
}

bool
Problem::NonlinearSolve::iterateResidual(const Vector & x, Vector & r, double & norm)
{
  _p.setOwnedPart(_p._U, x, _a.rows);
  _p.updateDependentDofs(_p._U);
  // Without a line search every residual is that of an iterate, whose Jacobian follows.
  _a.jacobian_current = !_line_search;
  try
  {
    residualAt(_p._U, _R, _line_search ? nullptr : &_J);
  }
  catch (const InadmissibleState &)
  {
    _a.jacobian_current = false;
    return false;
  }
  _a.row_factor = rowFactors();
  r = ownedPart(_a.row_factor.cwiseProduct(_R), _a.rows);
  norm = scaledNorm(_p.equationNorms(_R));
  return true;
}

bool
Problem::NonlinearSolve::refreshJacobian(petsc::LinearSolver & solver)
{
  if (_a.solver_current)
  {
    _a.solver_current = false;
    return true;
  }
  if (!_a.jacobian_current)
  {
    try
    {
      residualAt(_p._U, _R, &_J);
    }
    catch (const InadmissibleState &)
    {
      return false;
    }
    _a.jacobian_current = true;
  }
  if (_par.any(!_J.coeffs().allFinite()))
  {
    _inadmissible = "the Jacobian " + std::string(notFinite);
    return false;
  }
  solver.setEntries(globalEntries(_p, _J, _a.rows, _a.row_factor));
  if (_a.schur_approximation)
  {
    const petsc::SchurBlocks blocks =
        globalSchurBlocks(_p, _J, *_a.schur_approximation, _a.rows, _a.row_factor);
    solver.refresh(&blocks);
  }
  else
    solver.refresh();
  return true;
}

petsc::IterationStatus
Problem::NonlinearSolve::iterationTest(int it, int linear_total, double & norm)
{
  const int k = _a.steps_before + it;
  const std::vector<double> norms = _p.equationNorms(_R), sizes = _p.equationNorms(_reference);
  for (std::size_t v = 0; v < _scales.size(); ++v)
    _scales[v] = std::max(_scales[v], sizes[v]);
  const double rn = scaledNorm(norms);
  norm = rn;
  // The step to this iterate completes the row of the last one.
  const int linear_step = linear_total - _a.linear_before;
  _a.linear_before = linear_total;
  if (_a.have_record && it > 0)
  {
    IterationRecord & last = _result.history[_a.record];
    last.step_norm = _p.relativeStep(_p._U, _a.iterate);
    if (_a.print)
      console::nonlinearIteration(
          last.iteration, last.residual_norm, _a.out.r0, false, last.step_norm, linear_step);
  }
  // An iterate that is no longer finite has diverged.
  if (!std::isfinite(rn))
  {
    _a.out.diverged = true;
    return petsc::IterationStatus::diverged;
  }
  _a.iterate = _p._U;
  // An iterate is admissible: a trial of the line search that was not leaves no message.
  _inadmissible.clear();
  if (_a.out.r0 < 0)
  {
    _a.out.r0 = rn;
    _a.first_sizes = sizes;
    _a.first_growth = relativeToFirst(norms);
    _p._flow->notePressureLevelFlux(_p);
  }
  // A residual a million times the first is a diverged iteration: stop before the iterates
  // grow into a matrix that the linear solver cannot factorise.  It is measured against the
  // sizes of the equations at the first iterate, which do not grow with a diverging one.
  if (relativeToFirst(norms) > 1e6 * _a.first_growth)
  {
    _a.out.diverged = _a.out.grew = true;
    return petsc::IterationStatus::diverged;
  }
  // A solve with the next linear solver starts at the iterate where the last one stopped,
  // which has its row already.
  if (!(it == 0 && _a.steps_before > 0))
  {
    _result.history.push_back(IterationRecord{_a.step_index, _a.lf, k + 1, rn, 0.0});
    _a.record = _result.history.size() - 1;
    _a.have_record = true;
  }
  if (equationsConverged(norms, sizes))
  {
    _a.out.converged = true;
    if (_a.print)
      console::nonlinearIteration(k + 1, rn, _a.out.r0, true);
    return petsc::IterationStatus::converged;
  }
  // The linear solver takes one step: a residual above the tolerance after it means that
  // the problem is not linear or that the linear solve failed.
  if (_linear && it >= 1)
  {
    _a.out.not_linear = true;
    return petsc::IterationStatus::diverged;
  }
  if (k >= _o.max_iterations)
    return petsc::IterationStatus::diverged;
  return petsc::IterationStatus::iterate;
}

bool
Problem::NonlinearSolve::equationsConverged(const std::vector<double> & norms,
                                            const std::vector<double> & sizes) const
{
  // Every equation has converged when its residual is at most relative_tolerance of the size
  // of its terms, or at the round-off level of its rows (in its own units), which no further
  // step removes: an equation whose terms all vanish (the momentum of a uniform flow) has no
  // size of its own.  The round-off is 64 eps times the sum of |J| |U|, which a badly scaled
  // system carries (a thin beam, a stiff solid), and of the size of the terms inside the
  // fluxes and the sources, which may cancel (the stress of a solid at a small strain, whose
  // terms are of the size of its modulus).  J is that of this iterate or of the last one.
  // The components of a vector unknown are tested as one equation, whose residual, size
  // and round-off are those of all its components: the share of one component depends on
  // the orientation of the axes, and a component whose terms all vanish (the transverse
  // momentum of a plug flow) holds only the round-off that the solve of the others leaves.
  Vector levels_rows = _J.cwiseAbs() * _p._U.cwiseAbs();
  _par.addAtNodes(levels_rows.data(), _p.numVariables());
  levels_rows += _term_size;
  const std::vector<double> levels = _p.equationNorms(levels_rows);
  std::vector<double> equation_norm(norms.size(), 0.0), equation_size(norms.size(), 0.0),
      equation_level(norms.size(), 0.0);
  for (std::size_t v = 0; v < norms.size(); ++v)
  {
    const auto e = static_cast<std::size_t>(_equation_of[v]);
    equation_norm[e] += norms[v] * norms[v];
    equation_size[e] += sizes[v] * sizes[v];
    equation_level[e] += levels[v] * levels[v];
  }
  bool equations_converged = true;
  for (std::size_t e = 0; e < norms.size(); ++e)
    equations_converged =
        equations_converged &&
        std::sqrt(equation_norm[e]) <=
            std::max(_o.relative_tolerance * std::sqrt(equation_size[e]),
                     64.0 * std::numeric_limits<double>::epsilon() * std::sqrt(equation_level[e]));
  return equations_converged;
}

void
Problem::NonlinearSolve::solveWithLinearSolvers()
{
  petsc::NonlinearProblem problem;
  problem.residual = [this](const Vector & x, Vector & r, double & norm)
  { return iterateResidual(x, r, norm); };
  problem.jacobian = [this](const Vector &, petsc::LinearSolver & solver)
  { return refreshJacobian(solver); };
  problem.test = [this](int it, const Vector &, int linear_total, double & norm)
  { return iterationTest(it, linear_total, norm); };
  // The linear solvers in order: an iterative one that fails hands its iterate to the next.
  Vector x = ownedPart(_p._U, _a.rows);
  forEachLinearAttempt(
      _p,
      _a.rows,
      _o,
      _user,
      [this](Vector & factors) -> const SparseMatrix & { return attemptJacobian(factors); },
      [&](const linear_solve_attempt & choice, const petsc::Settings & settings, bool must_converge)
          -> std::string { return newtonSolve(problem, choice, settings, must_converge, x); });
}

const SparseMatrix &
Problem::NonlinearSolve::attemptJacobian(Vector & factors)
{
  if (!_a.jacobian_current)
  {
    residualAt(_p._U, _R, &_J);
    _a.jacobian_current = true;
  }
  // The residual that PETSc takes first is that of this state, with the same factors.
  _a.row_factor = rowFactors();
  factors = _a.row_factor;
  return _J;
}

std::string
Problem::NonlinearSolve::newtonSolve(const petsc::NonlinearProblem & problem,
                                     const linear_solve_attempt & choice,
                                     const petsc::Settings & settings,
                                     bool must_converge,
                                     Vector & x)
{
  petsc::LinearSolver solver(_a.rows.distributed,
                             _a.rows.global_size,
                             _a.rows.first_row,
                             _a.rows.num_owned,
                             globalEntries(_p, _J, _a.rows, _a.row_factor),
                             settings);
  _a.solver_current = true;
  _a.schur_approximation = choice.Schur_approximation;
  petsc::NonlinearSettings nonlinear;
  nonlinear.max_iterations = _linear ? 1 : std::max(0, _o.max_iterations - _a.steps_before);
  nonlinear.line_search = _line_search;
  nonlinear.max_line_search_steps = _o.max_line_search_steps;
  nonlinear.relaxation = _o.relaxation;
  // The automatic iterative solvers take the tolerance of Eisenstat and Walker, which
  // follows the decrease of the residual.  A direct solve, a solver of the user and a
  // linear problem take linear_tolerance.
  nonlinear.eisenstat_walker = !must_converge && !_linear;
  nonlinear.options = _user.nonlinear;
  _a.linear_before = 0;
  const petsc::NonlinearResult solve = petsc::solveNonlinear(problem, solver, x, nonlinear);
  _result.linear_iterations += solve.linear_iterations;
  _result.total_iterations += solve.iterations;
  _a.steps_before += solve.iterations;
  if (solve.linear_solve_failed)
  {
    _p._U = _a.iterate;
    x = ownedPart(_p._U, _a.rows);
    _a.jacobian_current = false;
    return solve.reason;
  }
  _a.out.diverged = _a.out.diverged || solve.domain_error;
  _a.out.line_search_failed = solve.line_search_failed;
  return {};
}

void
Problem::NonlinearSolve::settlePressureLevels()
{
  _p._flow->settlePressureLevels(_p,
                                 _R,
                                 [this](const Vector & U, Vector & at, bool keep_last)
                                 {
                                   if (!keep_last)
                                   {
                                     residualAt(U, at, nullptr);
                                     return;
                                   }
                                   const Vector kept = _p._last_residual,
                                                kept_folded = _p._last_folded_residual;
                                   residualAt(U, at, nullptr);
                                   _p._last_residual = kept;
                                   _p._last_folded_residual = kept_folded;
                                 });
}

} // namespace dualmesh
