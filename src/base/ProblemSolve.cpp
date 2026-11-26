// SPDX-License-Identifier: LGPL-2.1-or-later
//
// The linear and the nonlinear solves of a problem with PETSc, the same for a serial problem and a
// distributed one: both number their unknowns in the global numbering of Parallel::globalRows.
#include "dualmesh/base/Console.h"
#include "dualmesh/base/Problem.h"
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
  SaddlePointBlocks blocks = problem.saddlePointBlocks(A, approximation);
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
  for (const auto & group : problem.displacementGroups())
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
    if (!problem.displacementGroups().empty())
      settings.near_nullspace = ownedRows(problem.nearNullspace(), rows);
    break;
  case linear_solve_kind::direct:
    break;
  case linear_solve_kind::Schur_split:
    schur = globalSchurBlocks(problem, A, *attempt.Schur_approximation, rows, row_factor);
    settings.schur = &*schur;
    settings.options = user_linear;
    break;
  case linear_solve_kind::physics_blocks:
    settings.options = user_linear;
    settings.physics_blocks = physicsBlockRows(problem, rows);
    settings.near_nullspace = ownedRows(problem.nearNullspace(), rows);
    break;
  case linear_solve_kind::GAMG:
    settings.options = petsc::multigridOptions(true);
    settings.near_nullspace = ownedRows(problem.nearNullspace(), rows);
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

// ---------------------------------------------------------------------------
// The nonlinear solve, serial or distributed
// ---------------------------------------------------------------------------
SolveResult
Problem::nonlinearSolve(const SolverOptions & o,
                        AssemblyOptions base,
                        const OldResidual * steady_old,
                        const Vector * ramp_from)
{
  if (o.nonlinear_solver != "newton" && o.nonlinear_solver != "picard" &&
      o.nonlinear_solver != "linear")
    throw InputError("Unknown nonlinear solver '" + o.nonlinear_solver +
                     "' (use newton, picard, or linear).");
  if (o.line_search != "automatic" && o.line_search != "none" && o.line_search != "backtracking")
    throw InputError("Unknown line_search '" + o.line_search +
                     "' (use automatic, none or backtracking).");
  const bool picard = o.nonlinear_solver == "picard";
  const bool linear = o.nonlinear_solver == "linear";
  // The automatic choice: backtracking where a kernel needs it (Kernel::needsLineSearch), for
  // Newton's method.
  bool needed = false;
  for (const auto & k : _kernels)
    needed = needed || k->needsLineSearch();
  const bool line_search =
      o.nonlinear_solver == "newton" &&
      (o.line_search == "backtracking" || (o.line_search == "automatic" && needed));
  const Parallel & par = parallel();
  base.mode = picard ? LinearizationMode::Picard : LinearizationMode::Newton;
  const user_options_parts user = user_choice(o.petsc_options);

  // The residual at a state, summed over the processes, with the rows of the prescribed values,
  // and its reference: the size of the terms of each equation (AssemblyOptions::reference), zero
  // at the prescribed values.
  Vector reference;
  // The size of the terms of each entry, inside its fluxes and sources (Kernel::fluxTermSize),
  // whose round-off is the floor of the test where the terms cancel.
  Vector term_size;
  // A state where a property cannot be evaluated (an element turned inside out) fails on every
  // process together: the line search shortens the step, or continuation takes a smaller one.
  struct InadmissibleState
  {
  };
  std::string inadmissible;
  const auto residualAt = [&](const Vector & U, double lf, Vector & R, SparseMatrix * J)
  {
    Vector lagged = U;
    AssemblyOptions ao = base;
    ao.lagged = &lagged;
    ao.reference = &reference;
    ao.term_size = &term_size;
    bool failed = false;
    try
    {
      assemble(U, ao, R, J);
    }
    catch (const InadmissibleStateError & e)
    {
      failed = true;
      inadmissible = e.what();
    }
    if (par.any(failed))
      throw InadmissibleState{};
    if (steady_old)
    {
      R += steady_old->residual;
      reference += steady_old->reference;
      if (steady_old->term_size.size() == term_size.size())
        term_size += steady_old->term_size;
    }
    // The equation of a hanging node or a periodic copy goes to its primaries on each process
    // before the processes add their parts, as the Jacobian's does: the solver reads a row from
    // the process that owns it, which need not hold the constraint that folds into the row.
    // The last residual is kept both ways: unfolded for the pressure-level check, which reads
    // the equations of the dependent nodes, and folded for the reactions.
    Vector folded = R;
    foldDependents(folded, nullptr);
    for (Vector * v : {&R, &folded, &reference, &term_size})
      par.addAtNodes(v->data(), numVariables());
    _last_residual = R;
    _last_folded_residual = folded;
    R = std::move(folded);
    dirichletRows(R, J);
    const std::vector<char> fixed = constrainedDofs();
    for (Index d = 0; d < reference.size(); ++d)
      if (fixed[static_cast<std::size_t>(d)])
        reference[d] = term_size[d] = 0.0;
  };
  // Each equation is measured against the size of its own terms, so the tests have no unit: the
  // scaled norm adds the residual of every equation over the largest reference it has had in this
  // attempt (the line search compares states with it, and the divergence test the iterations).
  std::vector<double> scales(static_cast<std::size_t>(numVariables()), 0.0);
  const std::vector<int> equation_of = vectorEquations();
  const auto scaledNorm = [&](const std::vector<double> & norms)
  {
    double sum = 0.0;
    for (std::size_t v = 0; v < norms.size(); ++v)
      if (scales[v] > 0.0)
        sum += (norms[v] / scales[v]) * (norms[v] / scales[v]);
    return std::sqrt(sum);
  };
  // The linear solves take every equation over its size, so that their test, the residual over
  // the right-hand side, has no unit: the rows of each equation, those of its prescribed values
  // included, are multiplied by one over the size of its terms at the state of the last residual
  // or the largest it has had in this attempt, whichever is larger, with the components of a
  // vector unknown together as in the convergence test.  A positive factor per equation leaves
  // the solution and the block structure of the matrix as they are.  The factors change only
  // while the size of an equation grows, so the tolerance of Eisenstat and Walker compares the
  // residuals of successive iterates in one norm once the sizes have settled.  An equation whose
  // terms have all been zero so far keeps its rows unscaled.
  const auto rowFactors = [&]()
  {
    const std::vector<double> sizes = equationNorms(reference);
    std::vector<double> equation_size(sizes.size(), 0.0);
    for (std::size_t v = 0; v < sizes.size(); ++v)
    {
      const double size = std::max(scales[v], sizes[v]);
      equation_size[static_cast<std::size_t>(equation_of[v])] += size * size;
    }
    const int nv = numVariables();
    Vector factor(numDofs());
    for (Index d = 0; d < numDofs(); ++d)
    {
      const double size = std::sqrt(
          equation_size[static_cast<std::size_t>(equation_of[static_cast<std::size_t>(d % nv)])]);
      factor[d] = size > 0.0 ? 1.0 / size : 1.0;
    }
    return factor;
  };

  SolveResult result;
  Vector R;
  SparseMatrix J;
  // One solve with the driving data at the fraction lf of their values (1: the problem itself).
  struct Attempt
  {
    bool converged = false, diverged = false, grew = false;
    // The line search found no step length that lowers the residual.
    bool line_search_failed = false;
    // One linear solve with nonlinear_solver "linear" left a residual above the tolerance.
    bool not_linear = false;
    double r0 = -1;
  };
  const auto attempt = [&](double lf, int step_index) -> Attempt
  {
    base.data_fraction = lf;
    inadmissible.clear();
    applyDirichlet(_U, lf);
    _linear_dt = base.include_time_kernels ? base.dt : 0.0;
    _linear_theta = base.include_steady_terms ? base.theta : 0.0;
    const bool print = o.verbose && par.isRoot();
    if (print)
      console::nonlinearHeader(o.nonlinear_solver == "newton"
                                   ? "Newton"
                                   : (picard ? "Picard iteration" : "linear solve"),
                               step_index,
                               lf);
    std::fill(scales.begin(), scales.end(), 0.0);
    Attempt out;
    // The residual and the Jacobian at the start.  An enclosed flow fixes the level of its
    // pressure at one unknown, which changes the rows of the prescribed values, so they are taken
    // again.  The singularity check runs when the pattern of the Jacobian changes, which the
    // number of rows or entries is a cheap signature of: a constraint that appears folds rows into
    // others, an interface condition starts to couple two bodies, and a re-initialization after a
    // mesh change resets the check outright.  It needs the whole matrix.
    try
    {
      residualAt(_U, lf, R, &J);
      if (!_levels_checked && holdPressureLevels(J))
        residualAt(_U, lf, R, &J);
    }
    catch (const InadmissibleState &)
    {
      out.diverged = true;
      return out;
    }
    if (par.checksSingularity() &&
        (!_singularity_checked || _singularity_pattern_rows != J.rows() ||
         _singularity_pattern_nonzeros != J.nonZeros()))
    {
      checkSingularity(J);
      _singularity_checked = true;
      _singularity_pattern_rows = J.rows();
      _singularity_pattern_nonzeros = J.nonZeros();
    }
    const GlobalRows rows = par.globalRows(*this);
    // The factors of the rows of the residual last handed to PETSc, which its Jacobian takes.
    Vector row_factor;

    // The state of the last residual, whether J was assembled there, and whether the linear
    // solver already holds J (it is made from the Jacobian at the start of each solve).
    bool jacobian_current = true;
    bool solver_current = false;
    std::optional<SchurApproximation> schur_approximation;
    // The last iterate, which a failed solve returns to, and the record of its row of the history.
    Vector iterate = _U;
    std::size_t record = 0;
    bool have_record = false;
    std::vector<double> first_sizes;
    double first_growth = 0.0;
    // Newton steps of the solves with earlier linear solvers, and the Krylov iterations so far.
    int steps_before = 0;
    int linear_before = 0;
    const auto relativeToFirst = [&](const std::vector<double> & norms)
    {
      double largest = 0.0;
      for (std::size_t v = 0; v < norms.size(); ++v)
        if (first_sizes[v] > 0.0)
          largest = std::max(largest, norms[v] / first_sizes[v]);
      return largest;
    };

    petsc::NonlinearProblem problem;
    problem.residual = [&](const Vector & x, Vector & r, double & norm)
    {
      setOwnedPart(_U, x, rows);
      updateDependentDofs(_U);
      // Without a line search every residual is that of an iterate, whose Jacobian follows.
      jacobian_current = !line_search;
      try
      {
        residualAt(_U, lf, R, line_search ? nullptr : &J);
      }
      catch (const InadmissibleState &)
      {
        jacobian_current = false;
        return false;
      }
      row_factor = rowFactors();
      r = ownedPart(row_factor.cwiseProduct(R), rows);
      norm = scaledNorm(equationNorms(R));
      return true;
    };
    problem.jacobian = [&](const Vector &, petsc::LinearSolver & solver)
    {
      if (solver_current)
      {
        solver_current = false;
        return true;
      }
      if (!jacobian_current)
      {
        try
        {
          residualAt(_U, lf, R, &J);
        }
        catch (const InadmissibleState &)
        {
          return false;
        }
        jacobian_current = true;
      }
      if (par.any(!J.coeffs().allFinite()))
      {
        inadmissible = "the Jacobian " + std::string(notFinite);
        return false;
      }
      solver.setEntries(globalEntries(*this, J, rows, row_factor));
      if (schur_approximation)
      {
        const petsc::SchurBlocks blocks =
            globalSchurBlocks(*this, J, *schur_approximation, rows, row_factor);
        solver.refresh(&blocks);
      }
      else
        solver.refresh();
      return true;
    };
    problem.test = [&](int it, const Vector &, int linear_total, double & norm)
    {
      const int k = steps_before + it;
      const std::vector<double> norms = equationNorms(R), sizes = equationNorms(reference);
      for (std::size_t v = 0; v < scales.size(); ++v)
        scales[v] = std::max(scales[v], sizes[v]);
      const double rn = scaledNorm(norms);
      norm = rn;
      // The step to this iterate completes the row of the last one.
      const int linear_step = linear_total - linear_before;
      linear_before = linear_total;
      if (have_record && it > 0)
      {
        IterationRecord & last = result.history[record];
        last.step_norm = relativeStep(_U, iterate);
        if (print)
          console::nonlinearIteration(
              last.iteration, last.residual_norm, out.r0, false, last.step_norm, linear_step);
      }
      // An iterate that is no longer finite has diverged.
      if (!std::isfinite(rn))
      {
        out.diverged = true;
        return petsc::IterationStatus::diverged;
      }
      iterate = _U;
      // An iterate is admissible: a trial of the line search that was not leaves no message.
      inadmissible.clear();
      if (out.r0 < 0)
      {
        out.r0 = rn;
        first_sizes = sizes;
        first_growth = relativeToFirst(norms);
        notePressureLevelFlux();
      }
      // A residual a million times the first is a diverged iteration: stop before the iterates
      // grow into a matrix that the linear solver cannot factorise.  It is measured against the
      // sizes of the equations at the first iterate, which do not grow with a diverging one.
      if (relativeToFirst(norms) > 1e6 * first_growth)
      {
        out.diverged = out.grew = true;
        return petsc::IterationStatus::diverged;
      }
      // A solve with the next linear solver starts at the iterate where the last one stopped,
      // which has its row already.
      if (!(it == 0 && steps_before > 0))
      {
        result.history.push_back(IterationRecord{step_index, lf, k + 1, rn, 0.0});
        record = result.history.size() - 1;
        have_record = true;
      }
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
      Vector levels_rows = J.cwiseAbs() * _U.cwiseAbs();
      par.addAtNodes(levels_rows.data(), numVariables());
      levels_rows += term_size;
      const std::vector<double> levels = equationNorms(levels_rows);
      std::vector<double> equation_norm(norms.size(), 0.0), equation_size(norms.size(), 0.0),
          equation_level(norms.size(), 0.0);
      for (std::size_t v = 0; v < norms.size(); ++v)
      {
        const auto e = static_cast<std::size_t>(equation_of[v]);
        equation_norm[e] += norms[v] * norms[v];
        equation_size[e] += sizes[v] * sizes[v];
        equation_level[e] += levels[v] * levels[v];
      }
      bool equations_converged = true;
      for (std::size_t e = 0; e < norms.size(); ++e)
        equations_converged =
            equations_converged && std::sqrt(equation_norm[e]) <=
                                       std::max(o.relative_tolerance * std::sqrt(equation_size[e]),
                                                64.0 * std::numeric_limits<double>::epsilon() *
                                                    std::sqrt(equation_level[e]));
      if (equations_converged)
      {
        out.converged = true;
        if (print)
          console::nonlinearIteration(k + 1, rn, out.r0, true);
        return petsc::IterationStatus::converged;
      }
      // The linear solver takes one step: a residual above the tolerance after it means that
      // the problem is not linear or that the linear solve failed.
      if (linear && it >= 1)
      {
        out.not_linear = true;
        return petsc::IterationStatus::diverged;
      }
      if (k >= o.max_iterations)
        return petsc::IterationStatus::diverged;
      return petsc::IterationStatus::iterate;
    };

    // The linear solvers in order: an iterative one that fails hands its iterate to the next.
    if (!(o.relaxation >= 0.0 && o.relaxation < 1.0))
      throw InputError("relaxation is the weight gamma of the last iterate, from 0 (none) to "
                       "below 1: the update takes the fraction 1 - gamma of each step.");
    Vector x = ownedPart(_U, rows);
    forEachLinearAttempt(
        *this,
        rows,
        o,
        user,
        [&](Vector & factors) -> const SparseMatrix &
        {
          if (!jacobian_current)
          {
            residualAt(_U, lf, R, &J);
            jacobian_current = true;
          }
          // The residual that PETSc takes first is that of this state, with the same factors.
          row_factor = rowFactors();
          factors = row_factor;
          return J;
        },
        [&](const linear_solve_attempt & choice,
            const petsc::Settings & settings,
            bool must_converge) -> std::string
        {
          petsc::LinearSolver solver(rows.distributed,
                                     rows.global_size,
                                     rows.first_row,
                                     rows.num_owned,
                                     globalEntries(*this, J, rows, row_factor),
                                     settings);
          solver_current = true;
          schur_approximation = choice.Schur_approximation;
          petsc::NonlinearSettings nonlinear;
          nonlinear.max_iterations = linear ? 1 : std::max(0, o.max_iterations - steps_before);
          nonlinear.line_search = line_search;
          nonlinear.max_line_search_steps = o.max_line_search_steps;
          nonlinear.relaxation = o.relaxation;
          // The automatic iterative solvers take the tolerance of Eisenstat and Walker, which
          // follows the decrease of the residual; a direct solve, a solver of the user and a
          // linear problem take linear_tolerance.
          nonlinear.eisenstat_walker = !must_converge && !linear;
          nonlinear.options = user.nonlinear;
          linear_before = 0;
          const petsc::NonlinearResult solve = petsc::solveNonlinear(problem, solver, x, nonlinear);
          result.linear_iterations += solve.linear_iterations;
          result.total_iterations += solve.iterations;
          steps_before += solve.iterations;
          if (solve.linear_solve_failed)
          {
            _U = iterate;
            x = ownedPart(_U, rows);
            jacobian_current = false;
            return solve.reason;
          }
          out.diverged = out.diverged || solve.domain_error;
          out.line_search_failed = solve.line_search_failed;
          return {};
        });
    // A failed solve returns to its last iterate, which a trial of the line search may have left.
    if (!out.converged)
    {
      _U = iterate;
      return out;
    }

    // An enclosed flow reports its pressure with a mean of zero, and its reactions at the shifted
    // state.  A second shift, of the size of the pressure in the equations, checks that the
    // level is still free at the solution: a property that depends on the pressure makes it
    // matter, which the start at rest does not show.  The mass equations of an enclosed flow sum
    // to its boundary flux whatever the pressure, so every level then gives a solution of its
    // own, and the problem needs the level as data.  The last residual is that of the solution.
    if (!_levels.empty())
    {
      checkPressureLevelBalance(_last_residual);
      const std::vector<char> fixed = constrainedDofs();
      Vector reactions = Vector::Zero(numDofs());
      for (Index d = 0; d < numDofs(); ++d)
        if (fixed[d])
          reactions[d] = _last_residual[d];
      const std::vector<double> forces = equationNorms(reactions);
      // Each equation must stay as it is, to 1e-6 of the size that the shift would change it by
      // if the level mattered.  A shift too small to show that is replaced by a test shift.
      const Vector before = R;
      std::vector<double> shifts = shiftPressureLevels(_U);
      const bool shifted =
          std::any_of(shifts.begin(), shifts.end(), [](double s) { return s != 0.0; });
      if (shifted)
        residualAt(_U, lf, R, nullptr);
      Vector change = R - before;
      Vector test = _U;
      const std::vector<double> test_shifts = testShiftPressureLevels(test, forces);
      bool small = true;
      for (std::size_t l = 0; l < shifts.size(); ++l)
        small = small && std::abs(shifts[l]) < 1e-3 * test_shifts[l];
      if (small)
      {
        const Vector kept = _last_residual, kept_folded = _last_folded_residual;
        Vector at_test;
        residualAt(test, lf, at_test, nullptr);
        _last_residual = kept;
        _last_folded_residual = kept_folded;
        change = at_test - R;
        shifts = test_shifts;
      }
      const std::vector<double> changes = equationNorms(change),
                                allowed = pressureLevelChange(shifts);
      for (std::size_t v = 0; v < changes.size(); ++v)
        if (changes[v] > 1e-6 * allowed[v])
          throw InputError(enclosedLevelMessage());
    }
    return out;
  };

  // The problem itself first.  When it fails, a steady solve takes the data from the starting
  // state to their values in steps (continuation, as COMSOL's stationary solver does): the
  // sources, body forces and prescribed fluxes and loads from zero (Object::isLoad), and the
  // prescribed values from those of the starting state.  A step that converges doubles the next
  // one, a step that fails is halved and taken again from the last converged state.
  // A continuation needs data to ramp: an object that is data, or a prescribed value that
  // differs from the starting state.
  bool rampable = false;
  if (ramp_from)
  {
    for (const auto & k : _kernels)
      rampable = rampable || k->isLoad();
    for (const auto & bc : _ibcs)
      rampable = rampable || bc->isLoad();
    rampable = rampable || !_loads.empty();
    Vector prescribed = *ramp_from;
    applyDirichlet(prescribed, 1.0);
    rampable = rampable || prescribed != *ramp_from;
    rampable = par.any(rampable);
  }
  const bool continuation = rampable && !linear;
  int step_index = 1;
  Attempt last = attempt(1.0, step_index);
  double done = 0.0;
  if (!last.converged && continuation)
  {
    // The continuation starts from the state its values are ramped from (a flow at rest, not the
    // Stokes solution of the full data that Newton's method was started from).
    _ramp_from = *ramp_from;
    Vector at_done = *ramp_from;
    double step = 0.5;
    const double smallest = 1.0 / 1024.0;
    while (done < 1.0 && step >= smallest)
    {
      const double lf = std::min(1.0, done + step);
      _U = at_done;
      last = attempt(lf, ++step_index);
      if (last.converged)
      {
        done = lf;
        at_done = _U;
        step *= 2.0;
      }
      else
        step *= 0.5;
    }
    _ramp_from.resize(0);
    if (!last.converged)
    {
      // The state at the last data reached, and the prescribed values of the problem itself.
      _U = at_done;
      applyDirichlet(_U, 1.0);
    }
  }
  if (!last.converged)
  {
    result.converged = false;
    if (o.error_on_divergence)
    {
      std::ostringstream os;
      const std::string ramped =
          continuation
              ? (done > 0.0 ? " Continuation ramped the data up to " + std::to_string(done) +
                                  " of their values, and no smaller step went further."
                            : " Continuation from the starting state did not converge "
                              "at any fraction of the data.")
              : "";
      if (!inadmissible.empty())
        os << "dualmesh: the nonlinear solve stopped: " << inadmissible << ramped;
      else if (last.grew)
        os << "dualmesh: Newton's method diverged: the residual grew from " << last.r0
           << " to more than 1e6 times that." << ramped
           << " Either the start is too far from the solution, or the problem has no solution "
              "(check the data). Start from a closer state, refine the mesh, or use "
              "nonlinear_solver 'Picard'.";
      else if (last.line_search_failed)
        os << "dualmesh: the nonlinear solve stopped: the line search found no step length, "
              "down to 2^-"
           << o.max_line_search_steps << " of the Newton step, that lowers the residual." << ramped
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
        os << "dualmesh: nonlinear solve did not converge within " << o.max_iterations
           << " iterations." << ramped;
      throw std::runtime_error(os.str());
    }
    return result;
  }
  result.converged = true;
  return result;
}

} // namespace dualmesh
