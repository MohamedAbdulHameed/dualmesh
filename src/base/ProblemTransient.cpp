// SPDX-License-Identifier: LGPL-2.1-or-later
#include "dualmesh/base/Checkpoint.h"
#include "dualmesh/base/Problem.h"

#include <algorithm>
#include <cmath>
#include <deque>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>

namespace dualmesh
{

namespace
{
/// Whether any entry of the mask @p mask is set.
bool
anySet(const std::vector<char> & mask)
{
  return std::any_of(mask.begin(), mask.end(), [](char c) { return c != 0; });
}
} // namespace

bool
Problem::hasSecondOrderVariables() const
{
  return anySet(_second_order_variables);
}

/// The time integration of the problem's own assembly.  The theta family weights the steady part
/// of the residual between the old and the new state:
///   R_time(U^{n+1}) + theta R_ss(U^{n+1}) + (1 - theta) R_ss(U^n) = 0.
/// The old contribution is evaluated once, before the nonlinear iteration, and at the old time
/// t^n: a source, a coefficient or a boundary flux that depends on time explicitly is taken at
/// t^n in this term and at t^{n+1} in the other, which keeps the Crank-Nicolson method second
/// order.  An equation with a second time derivative takes the generalized-alpha method, or
/// Newmark's average acceleration (alpha_m = alpha_f = 0): its steady terms at the level
/// 1 - alpha_f between the two states, as the theta method weights them, and its time derivatives
/// by the Newmark relations from the velocity and the acceleration of the old state, at the
/// levels 1 - alpha_f (velocity) and 1 - alpha_m (acceleration).  The first-order equations of
/// the same problem take theta = 1 - alpha_f.
class Problem::ThetaIntegrator : public TimeIntegrator
{
public:
  explicit ThetaIntegrator(Problem & problem) : _problem(problem) {}

  /// Take the scheme of @p transient and the solver options @p options of the next run.
  void configure(const TransientOptions & transient, const SolverOptions & options);
  void start(const Vector & solution, double /*time*/) override;
  SolveResult
  step(Vector & solution, const Vector & old, double time, double dt, bool continues) override;
  void accept(const Vector & solution, double /*time*/) override;
  /// The trapezoidal rule (theta = 1/2, Crank-Nicolson) is second order and every other member of
  /// the theta family first.  The generalized-alpha and Newmark methods are second order.
  int order() const override
  {
    return _dynamic || std::abs(_transient.theta - 0.5) < 1e-12 ? 2 : 1;
  }
  bool predict(double dt, Vector & predicted, double & factor) const override;
  /// The velocity of the last accepted state of a second-order equation, empty before a step.
  Vector velocity() const { return _committed ? _committed->v : Vector(); }

private:
  /// A state that the integration reached: its solution, velocity and acceleration (for a
  /// second-order equation), and the step that reached it: the state it started from, its size
  /// and its theta (a size of zero at the start of a run).  The old residual of the next step
  /// takes that step, whose residual gives the velocity that the mass equation of the step
  /// conserved.
  struct Level
  {
    Vector u, v, a, start;
    double dt = 0.0;
    double theta = 1.0;
  };

  Problem & _problem;
  TransientOptions _transient;
  SolverOptions _options;
  bool _dynamic = false;
  double _theta = 1.0, _alpha_m = 0.0, _alpha_f = 0.0, _beta = 0.25, _gamma = 0.5;
  /// The level of the last accepted state, and that of the last step taken.
  std::optional<Level> _committed, _trial;
  /// The accepted states, newest first, each with the span of the steps that reached it: the
  /// nodes of the predictor.  The span of the steps taken since the last accepted state.
  std::deque<std::pair<Vector, double>> _accepted;
  double _span = 0.0;
};

void
Problem::ThetaIntegrator::configure(const TransientOptions & transient,
                                    const SolverOptions & options)
{
  _transient = transient;
  _options = options;
  _dynamic = _problem.hasSecondOrderVariables();
  _theta = transient.theta;
  _alpha_m = _alpha_f = 0.0;
  _beta = 0.25;
  _gamma = 0.5;
  if (_dynamic)
  {
    if (transient.second_order_scheme == "generalized_alpha")
    {
      const double rho = transient.spectral_radius;
      if (!(rho >= 0.0 && rho <= 1.0))
        throw InputError("The spectral radius of the generalized-alpha method is between 0 and 1.");
      _alpha_m = (2.0 * rho - 1.0) / (rho + 1.0);
      _alpha_f = rho / (rho + 1.0);
      _gamma = 0.5 - _alpha_m + _alpha_f;
      _beta = 0.25 * (1.0 - _alpha_m + _alpha_f) * (1.0 - _alpha_m + _alpha_f);
    }
    else if (transient.second_order_scheme != "Newmark")
      throw InputError("Unknown scheme '" + transient.second_order_scheme +
                       "' of a second-order equation (generalized_alpha or Newmark).");
    _theta = 1.0 - _alpha_f;
  }
  if (_theta <= 0.0)
    for (const auto & k : _problem._kernels)
      if (k->residualHasTimeDerivative())
        throw InputError("'" + k->name() +
                         "' is a stabilisation whose residual holds the time derivative, which "
                         "the forward_Euler time scheme cannot weight. Use backward_Euler or "
                         "Crank-Nicolson.");
}

void
Problem::ThetaIntegrator::start(const Vector & solution, double /*time*/)
{
  _accepted = {{solution, 0.0}};
  _trial.reset();
}

SolveResult
Problem::ThetaIntegrator::step(
    Vector & /*solution*/, const Vector & old, double time, double dt, bool continues)
{
  // The nonlinear solve advances the problem's own solution at the problem's time.
  Problem & p = _problem;
  // Runs @p f with the problem's clock at the start of the step, where the old state and its
  // prescribed values and loads belong, and puts the clock back at the end, @p time.
  const auto atStart = [&](auto && f)
  {
    p._time = time - dt;
    try
    {
      f();
    }
    catch (...)
    {
      p._time = time;
      throw;
    }
    p._time = time;
  };
  _span = continues ? _span + dt : dt;
  // The history of stateful property objects follows the solution: a step starts from the
  // committed history when it starts from the last accepted solution, and from the state its
  // predecessor reached when it continues that predecessor.
  if (continues)
    p.startStateFromTrial();
  else
    p.startStateFromCommitted();
  // The start of a run with inertia: the initial velocity, and the acceleration that it and the
  // prescribed values give.
  if (_dynamic && !_committed)
  {
    Level begin;
    begin.u = old;
    if (p._initial_velocity.size() != 0 && p._initial_velocity.size() != old.size())
      throw InputError("Transient: the initial velocity has " +
                       std::to_string(p._initial_velocity.size()) + " values, and the problem " +
                       std::to_string(old.size()) + " unknowns.");
    begin.v = p._initial_velocity.size() == old.size() ? p._initial_velocity
                                                       : Vector(Vector::Zero(old.size()));
    atStart([&] { begin.a = p.initialAcceleration(old, begin.v, dt, _options); });
    _committed = std::move(begin);
  }
  // The level of the old state: the step just taken when this step continues it, and otherwise
  // the last accepted state.
  const std::optional<Level> & before = continues ? _trial : _committed;
  Vector rate_shift, accel_shift, predicted, predicted_rate;
  if (_dynamic)
  {
    // The predictors: u~ = u_n + dt v_n + dt^2 (1/2 - beta) a_n, v~ = v_n + dt (1 - gamma) a_n,
    // with a_{n+1} = (u - u~) / (beta dt^2) and v_{n+1} = v~ + gamma dt a_{n+1}.
    predicted = before->u + dt * before->v + dt * dt * (0.5 - _beta) * before->a;
    predicted_rate = before->v + dt * (1.0 - _gamma) * before->a;
    const double accel_scale = (1.0 - _alpha_m) / (_beta * dt * dt);
    const double rate_scale = (1.0 - _alpha_f) * _gamma / (_beta * dt);
    accel_shift = accel_scale * predicted - _alpha_m * before->a;
    rate_shift = rate_scale * predicted - (1.0 - _alpha_f) * predicted_rate - _alpha_f * before->v;
  }
  OldResidual old_residual;
  if (_theta < 1.0)
  {
    AssemblyOptions ao;
    ao.include_time_kernels = false;
    ao.theta = 1.0 - _theta;
    ao.old_level = true;
    ao.dt = dt;
    if (before && before->dt > 0.0)
    {
      ao.old = &before->start;
      ao.dt = before->dt;
      ao.previous_step = true;
      ao.previous_theta = before->theta;
    }
    ao.reference = &old_residual.reference;
    ao.term_size = &old_residual.term_size;
    atStart([&] { p.assemble(old, ao, old_residual.residual, nullptr); });
  }
  AssemblyOptions base;
  base.old = &old;
  base.dt = dt;
  base.theta = _theta;
  base.include_time_kernels = true;
  if (_dynamic)
  {
    base.second_order = true;
    base.rate_scale = (1.0 - _alpha_f) * _gamma / (_beta * dt);
    base.accel_scale = (1.0 - _alpha_m) / (_beta * dt * dt);
    base.rate_shift = &rate_shift;
    base.accel_shift = &accel_shift;
  }
  // An explicit step still takes the equations without a time derivative at the new time.
  base.include_steady_terms = _theta > 0 || anySet(p._new_time_only);
  SolveResult result = p.nonlinearSolve(_options, base, _theta < 1.0 ? &old_residual : nullptr);
  if (!result.converged)
    return result;
  Level reached;
  reached.start = old;
  reached.dt = dt;
  reached.theta = _theta;
  if (_dynamic)
  {
    reached.u = p._U;
    reached.a = (p._U - predicted) / (_beta * dt * dt);
    reached.v = predicted_rate + _gamma * dt * reached.a;
  }
  _trial = std::move(reached);
  return result;
}

void
Problem::ThetaIntegrator::accept(const Vector & solution, double /*time*/)
{
  _committed = _trial;
  _accepted.emplace_front(solution, _span);
  if (_accepted.size() > 3)
    _accepted.pop_back();
}

bool
Problem::ThetaIntegrator::predict(double h, Vector & predicted, double & factor) const
{
  // A second-order equation takes step doubling, which needs no error constant.
  if (_dynamic)
    return false;
  // The predictor of a step of the theta method is the polynomial through the last accepted
  // states extrapolated to the end of the step, and its difference from the step is in
  // proportion to the local error of the step (Milne's device).  With h the step, h1 and h2 the
  // steps before it, u2 and u3 the second and third time derivatives:
  //   theta method of order one: error (theta - 1/2) h^2 u2, linear predictor of error
  //   -u2 h (h + h1) / 2, so e = |U - U_p| |2 c h / (2 c h + h + h1)| with c = theta - 1/2;
  //   Crank-Nicolson: error h^3 u3 / 12, quadratic predictor of error -u3 P / 6 with
  //   P = h (h + h1) (h + h1 + h2), so e = |U - U_p| h^3 / (2 P + h^3).
  // Each error is that of the step or the predictor minus the exact solution.
  const std::size_t nodes = order() >= 2 ? 2 : 1;
  if (_accepted.size() < nodes + 1)
    return false;
  const Vector & old = _accepted[0].first;
  const double h1 = _accepted[0].second;
  if (nodes == 2)
  {
    const double h2 = _accepted[1].second;
    const Vector & u1 = _accepted[1].first;
    const Vector & u2 = _accepted[2].first;
    // The Lagrange polynomial through (-h1 - h2, u2), (-h1, u1) and (0, old), at h.
    predicted = old * ((h + h1) * (h + h1 + h2) / (h1 * (h1 + h2))) -
                u1 * (h * (h + h1 + h2) / (h1 * h2)) + u2 * (h * (h + h1) / ((h1 + h2) * h2));
    const double product = h * (h + h1) * (h + h1 + h2);
    factor = h * h * h / (2.0 * product + h * h * h);
  }
  else
  {
    predicted = old + (h / h1) * (old - _accepted[1].first);
    const double c = _transient.theta - 0.5;
    factor = std::abs(2.0 * c * h / (2.0 * c * h + h + h1));
  }
  return true;
}

Vector
Problem::initialAcceleration(const Vector & u, Vector & v, double dt, const SolverOptions & options)
{
  const Parallel & par = parallel();
  const Index n = numDofs();
  const int nv = numVariables();
  const std::vector<char> fixed = constrainedDofs();
  // The prescribed values set the velocity and the acceleration of their unknowns, by central
  // differences in time over a thousandth of the step.
  const double h = 1e-3 * dt;
  const double now = _time;
  Vector before = u, after = u;
  _time = now - h;
  applyDirichlet(before, 1.0);
  _time = now + h;
  applyDirichlet(after, 1.0);
  _time = now;
  Vector a = Vector::Zero(n);
  for (Index d = 0; d < n; ++d)
    if (fixed[d] && _second_order_variables[static_cast<std::size_t>(d % nv)])
    {
      v[d] = (after[d] - before[d]) / (2.0 * h);
      a[d] = (after[d] - 2.0 * u[d] + before[d]) / (h * h);
    }
  // A periodic copy and a hanging node move with their primaries.  The differences above hold
  // them still, which would drop the velocity carried from the last mesh.
  updateDependentDofs(v);
  // M a = -R(u, v): the residual with the velocity v and no acceleration, and the Jacobian of
  // the second time derivatives alone, which is M.
  AssemblyOptions ao;
  ao.old = &u;
  ao.dt = dt;
  ao.second_order = true;
  ao.rate_scale = 0.0;
  ao.accel_scale = 1.0;
  const Vector rate_shift = -v;
  ao.rate_shift = &rate_shift;
  ao.accel_shift = &u;
  Vector R, unused;
  assemble(u, ao, R, nullptr);
  // The equations of periodic copies and hanging nodes belong to their primaries.  Each process
  // folds its own part before the processes add them, as it folds its part of M: the solver reads
  // a row from the process that owns it, which need not hold the constraint that folds into it.
  foldDependents(R, nullptr);
  par.addAtNodes(R.data(), nv);
  AssemblyOptions mass = ao;
  mass.include_steady_terms = false;
  SparseMatrix M;
  assemble(u, mass, unused, &M);
  foldDependents(R, &M);
  // The other unknowns keep their acceleration: identity rows and columns.
  std::vector<char> keep(static_cast<std::size_t>(n), 0);
  for (Index d = 0; d < n; ++d)
    keep[d] = !fixed[d] && _second_order_variables[static_cast<std::size_t>(d % nv)];
  // The prescribed accelerations move to the right-hand side.  M holds this process's elements
  // only, so the processes add their parts of M a before the owner of a row reads it.
  Vector lift = Vector::Zero(n);
  std::vector<Eigen::Triplet<double>> entries;
  for (int k = 0; k < M.outerSize(); ++k)
    for (SparseMatrix::InnerIterator it(M, k); it; ++it)
    {
      if (keep[it.row()] && keep[it.col()])
        entries.emplace_back(it.row(), it.col(), it.value());
      else if (keep[it.row()])
        lift[it.row()] += it.value() * a[it.col()];
    }
  par.addAtNodes(lift.data(), nv);
  Vector rhs = -R - lift;
  for (Index d = 0; d < n; ++d)
    if (!keep[d])
    {
      entries.emplace_back(d, d, 1.0);
      rhs[d] = a[d];
    }
  SparseMatrix system(n, n);
  system.setFromTriplets(entries.begin(), entries.end());
  int iterations = 0;
  Vector acceleration = linearSolve(system, rhs, options, &iterations);
  // A periodic copy and a hanging node take the acceleration of their primaries.
  updateDependentDofs(acceleration);
  return acceleration;
}

Vector
Problem::velocity() const
{
  return _theta ? _theta->velocity() : Vector();
}

namespace
{
/// Times closer to the end of a run than this fraction of its span count as the end.
constexpr double kEndTolerance = 1e-12;
/// An output time closer to the time of the run than this fraction of its span is reached.
constexpr double kOutputTolerance = 1e-10;
/// The span that the tolerance of the output times takes for a run of zero span.
constexpr double kSmallestSpan = 1e-300;
/// A step that would leave less than this fraction of the span of the run before its target
/// reaches the target.
constexpr double kSliverFraction = 1e-8;
/// Step sizes that differ by less than this fraction of the step count as equal.
constexpr double kStepRounding = 1e-8;
/// The smallest step, as a fraction of the given step, when the options set none.
constexpr double kSmallestStepFraction = 1e-6;
/// The safety factor of the step that the error controller asks for, and the smallest fraction
/// of a rejected step that it retries with.
constexpr double kSafetyFactor = 0.9;
constexpr double kSmallestRetryFraction = 0.1;
/// The step after a step shortened to land on a target is at most this multiple of it.
constexpr double kShortenedGrowth = 2.0;

/// The step controllers of TransientOptions::time_stepper.
enum class Stepper
{
  fixed,
  error,
  iteration,
  courant
};

/// The step controller of @p tr, after its options are checked for @p integrator.
Stepper
checkTransientOptions(const TransientOptions & tr, const TimeIntegrator & integrator)
{
  if (tr.dt <= 0)
    throw InputError("Transient: the time step must be positive.");
  if (tr.theta < 0 || tr.theta > 1)
    throw InputError("Transient: theta must be in [0, 1].");
  Stepper stepper = Stepper::fixed;
  if (tr.time_stepper == "error")
    stepper = Stepper::error;
  else if (tr.time_stepper == "iteration")
    stepper = Stepper::iteration;
  else if (tr.time_stepper == "cfl")
    stepper = Stepper::courant;
  else if (tr.time_stepper != "fixed")
    throw InputError("Transient: unknown time stepper '" + tr.time_stepper +
                     "' (use fixed, error, iteration or cfl).");
  if (stepper == Stepper::courant && !integrator.measuresCourantNumber())
    throw InputError("Transient: the cfl stepper needs a time integration that measures the "
                     "Courant number, the projection integration of a flow.");
  if (stepper == Stepper::courant && !(tr.courant_number > 0))
    throw InputError("Transient: the Courant number of the cfl stepper must be positive.");
  if (tr.growth_factor < 1.0)
    throw InputError("Transient: growth_factor must be at least one.");
  if (tr.cutback_factor <= 0.0 || tr.cutback_factor >= 1.0)
    throw InputError("Transient: cutback_factor must lie strictly between zero and one.");
  for (std::size_t k = 1; k < tr.output_times.size(); ++k)
    if (!(tr.output_times[k] > tr.output_times[k - 1]))
      throw InputError("Transient: the output times must increase.");
  return stepper;
}

/// The error of a step, which the error controller takes.  The scale of each equation is the
/// largest of the range of its variables, their change from the start of the run and the square
/// root of the machine epsilon times their magnitude, the largest that the accepted states and
/// the step have reached, with the components of a vector unknown together as in the convergence
/// test of the nonlinear solver.  The change is offset-free, so a temperature in kelvin is not
/// measured against its 300 K, and a field that is uniform, or zero to round-off, has the scale
/// of its motion (a rigid motion, a uniform flow).  The floor keeps the round-off of a field at
/// rest from counting as an error.
///
/// The error of a step is that of the variables with a time derivative, of the first or the
/// second order.  A variable without one (the pressure of an incompressible flow) is algebraic:
/// it has no truncation error of its own, and the theta method leaves its value at the new time
/// with oscillations that are no error of the step, so it is left out of the test (as the error
/// test of IDA can leave out the algebraic variables).  A problem without a time derivative
/// measures every variable.  Every process takes the same numbers, so they all take the same
/// steps and accept or reject together.
class StepError
{
public:
  /// The error of the steps of @p problem from its current solution, the start of the run, whose
  /// variables have no time derivative where @p new_time_only is set unless @p second_order is.
  StepError(const Problem & problem,
            const std::vector<char> & new_time_only,
            const std::vector<char> & second_order)
      : _problem(problem), _differential(new_time_only.size(), 0),
        _equation_of(problem.vectorEquations()), _initial(problem.solution()),
        _reached(problem.variableSizes(_initial, _initial))
  {
    for (std::size_t v = 0; v < _differential.size(); ++v)
      _differential[v] = !new_time_only[v] || second_order[v];
    if (!anySet(_differential))
      _differential.clear();
  }

  /// The error of the step from @p before to @p after.
  double estimate(const Vector & after, const Vector & before) const
  {
    Problem::VariableSizes sizes = _problem.variableSizes(after, _initial);
    takeLargest(sizes, _reached);
    const double floor = std::sqrt(std::numeric_limits<double>::epsilon());
    std::vector<double> scale(sizes.range.size(), 0.0);
    for (std::size_t v = 0; v < scale.size(); ++v)
    {
      const std::size_t e = static_cast<std::size_t>(_equation_of[v]);
      scale[e] = std::max({scale[e], sizes.range[v], sizes.change[v], floor * sizes.magnitude[v]});
    }
    const std::vector<double> step = _problem.variableSizes(after, before).change;
    double estimate = 0.0;
    for (std::size_t v = 0; v < step.size(); ++v)
    {
      if ((!_differential.empty() && !_differential[v]) || !(step[v] > 0.0))
        continue;
      const double s = scale[static_cast<std::size_t>(_equation_of[v])];
      estimate =
          std::max(estimate, s > 0.0 ? step[v] / s : std::numeric_limits<double>::infinity());
    }
    return estimate;
  }

  /// Take the sizes of the accepted state @p state into the scales.
  void accept(const Vector & state)
  {
    takeLargest(_reached, _problem.variableSizes(state, _initial));
  }

private:
  /// Raise every size of @p to to that of @p from where it is larger.
  static void takeLargest(Problem::VariableSizes & to, const Problem::VariableSizes & from)
  {
    for (std::size_t v = 0; v < to.range.size(); ++v)
    {
      to.range[v] = std::max(to.range[v], from.range[v]);
      to.change[v] = std::max(to.change[v], from.change[v]);
      to.magnitude[v] = std::max(to.magnitude[v], from.magnitude[v]);
    }
  }

  const Problem & _problem;
  std::vector<char> _differential;
  const std::vector<int> _equation_of;
  const Vector _initial;
  Problem::VariableSizes _reached;
};

/// The output times of a run: the solution is written at each of them, and the steps land on
/// them.  Times closer than @p tolerance count as equal.
class OutputSchedule
{
public:
  OutputSchedule(const std::vector<double> & times, double tolerance)
      : _times(times), _tolerance(tolerance)
  {
  }

  /// The time that the next step aims at: the next output time, or @p end_time.
  double target(double end_time) const
  {
    return _next < _times.size() ? std::min(_times[_next], end_time) : end_time;
  }

  /// Write with @p write the output times at the start time @p time.  An output time before the
  /// start belongs to an earlier run (a restart continues one), so it is passed unwritten.
  void writeAtStart(double time, const std::function<void(int)> & write)
  {
    writeUpTo(time, time - _tolerance, write);
  }

  /// Write with @p write every output time up to @p time that is not yet written.
  void writeUpTo(double time, const std::function<void(int)> & write)
  {
    writeUpTo(time, -std::numeric_limits<double>::infinity(), write);
  }

private:
  /// Pass every output time up to @p time, and write those from @p earliest on.
  void writeUpTo(double time, double earliest, const std::function<void(int)> & write)
  {
    while (_next < _times.size() && _times[_next] <= time + _tolerance)
    {
      if (write && _times[_next] >= earliest)
        write(static_cast<int>(_next));
      ++_next;
    }
  }

  const std::vector<double> & _times;
  const double _tolerance;
  std::size_t _next = 0;
};

/// The size of a step from @p time towards @p target, and whether it is shorter than the step
/// @p dt that the controller chose.
struct Landing
{
  double size;
  bool shortened;
};

/// Land exactly on the next output time or on the end time, @p target, from @p time with the
/// chosen step @p dt, in a run of span @p span, and do not leave a sliver behind: if one step
/// would nearly reach the target, reach it, and if the target is less than two steps away, reach
/// it in two equal steps.  A step is then shorter than half the size the controller chose only
/// where two targets are closer than that, and the step after a shortened one is at most twice
/// it, which keeps the step ratio below the bound 1 + sqrt 2 of the zero-stability of
/// variable-step BDF2 (Grigorieff, Numerische Mathematik 42, 1983).  Sizes that differ by less
/// than kStepRounding dt count as equal, so the rounding of the times neither halves a step
/// whose target is two steps away nor marks a step of the chosen size as shortened.
Landing
landOnTarget(double time, double target, double dt, double span)
{
  const double rounding = kStepRounding * dt;
  const double remaining = target - time;
  double size = std::min(dt, remaining);
  if (remaining - size < kSliverFraction * std::abs(span))
    size = remaining;
  else if (remaining < 2.0 * dt - rounding)
    size = 0.5 * remaining;
  return {size, size < dt - rounding};
}
} // namespace

/// The time-stepping loop of one transient run (Problem::runTransient) and the state its steps
/// share: the step size, the counts of the steps, the output times passed and the error scales.
class Problem::TransientRun
{
public:
  TransientRun(Problem & problem,
               TimeIntegrator & integrator,
               const TransientOptions & tr,
               const SolverOptions & options,
               const CheckpointLayout & layout,
               int first_step);
  /// Take the steps from the start time to the end time.
  SolveResult run();

private:
  /// The decision on a step: whether it is accepted, the size of the step after it (or of the
  /// retry of a rejected one), the largest next step that the controller allows without its
  /// growth cap (the step that the error estimate or the Courant number asks for, the given step
  /// of the fixed stepper, or the choice of the iteration stepper), and the nonlinear iterations
  /// of the step.
  struct Decision
  {
    bool accepted = true;
    double next_dt = 0.0;
    double uncapped = 0.0;
    int iterations = 0;
  };

  /// A step is kept within [dt_min, dt_max].  An adaptive stepper also keeps it at most the
  /// largest whose discretization does not depend on it: a larger step would change the answer
  /// by more than its error estimate measures.  A fixed step is the one given.
  double bounded(double h) const;
  /// A trial step from @p from to @p end of size @p h: the last accepted state, or, when it
  /// continues, the state of the step just taken.
  SolveResult trial(const Vector & from, double end, double h, bool continues);
  /// The step of size @p size from the accepted state @p old at @p start, and its decision.
  Decision attempt(const Vector & old, double start, double size);
  /// A step with a predictor takes one solve, and its difference from the predictor
  /// @p predicted, times the factor @p factor of the predictor, estimates its local error
  /// (Milne's device).
  Decision predictedStep(
      const Vector & old, double start, double size, const Vector & predicted, double factor);
  /// One coarse step, and two half steps from the same starting state, whose difference
  /// estimates the local error (Richardson).
  Decision doubledStep(const Vector & old, double start, double size);
  /// One step, sized by the Courant number, the iterations or the given step.
  Decision singleStep(const Vector & old, double start, double size);
  /// The decision of the error controller on the estimate @p estimate of the local error of a
  /// step of size @p size.  A rejected step is retried at the size that the estimate asks for, at
  /// least a tenth of the step (Hairer, Norsett and Wanner 1993, Sect. II.4).
  Decision control(double estimate, double size, int iterations) const;
  /// The rejection of a step of size @p size whose solve failed: a retry at a cut-back size.
  Decision cutBack(double size, int iterations) const;
  /// Return to the state @p old at @p start after the rejected step @p d, and take the size of
  /// the retry.  False when the run stops because the size fell below dt_min or the steps were
  /// rejected more than max_rejected_steps times in a row, which throws when the solver options
  /// ask for it.
  bool reject(const Decision & d, const Vector & old, double start);
  /// Count, check and commit the accepted step of size @p size, and write the output times it
  /// passed.
  void accept(double size, int iterations);
  /// The size of the step after the accepted step @p d of @p landing.
  double nextStep(const Decision & d, const Landing & landing);
  /// Write a checkpoint after the accepted step when one is due.
  void checkpoint() const;

  Problem & _p;
  TimeIntegrator & _integrator;
  const TransientOptions & _tr;
  const SolverOptions & _options;
  const CheckpointLayout & _layout;
  const int _first_step;
  const Parallel & _par;
  const Stepper _stepper;
  StepError _error;
  const double _span, _end, _dt_min, _dt_max, _order;
  OutputSchedule _outputs;
  SolveResult _total;
  int _step = 0;
  double _dt = 0.0;
  /// The step that the controller chose before an output time shortened a step, which the steps
  /// after it return to (zero when there is none).
  double _resume = 0.0;
  int _rejected_in_a_row = 0;
};

Problem::TransientRun::TransientRun(Problem & problem,
                                    TimeIntegrator & integrator,
                                    const TransientOptions & tr,
                                    const SolverOptions & options,
                                    const CheckpointLayout & layout,
                                    int first_step)
    : _p(problem), _integrator(integrator), _tr(tr), _options(options), _layout(layout),
      _first_step(first_step), _par(problem.parallel()),
      _stepper(checkTransientOptions(tr, integrator)),
      _error(problem, problem._new_time_only, problem._second_order_variables),
      _span(tr.end_time - tr.start_time), _end(tr.end_time - kEndTolerance * std::abs(_span)),
      _dt_min(tr.dt_min > 0 ? tr.dt_min : tr.dt * kSmallestStepFraction),
      _dt_max(tr.dt_max > 0 ? tr.dt_max : std::abs(_span)), _order(integrator.order()),
      _outputs(tr.output_times, kOutputTolerance * std::max(std::abs(_span), kSmallestSpan))
{
}

SolveResult
Problem::TransientRun::run()
{
  _p._time = _tr.start_time;
  _dt = bounded(_tr.dt);
  _outputs.writeAtStart(_p._time, _p._output_callback);
  while (_p._time < _end)
  {
    const Landing landing = landOnTarget(_p._time, _outputs.target(_tr.end_time), _dt, _span);
    const Vector old = _p._U;
    const double start = _p._time;
    const Decision d = attempt(old, start, landing.size);
    if (!d.accepted)
    {
      if (!reject(d, old, start))
        return _total;
      continue;
    }
    accept(landing.size, d.iterations);
    _dt = nextStep(d, landing);
    checkpoint();
    _total.next_time_step = _dt;
    if (_tr.max_steps > 0 && _step >= _tr.max_steps)
      break;
  }
  _total.converged = true;
  return _total;
}

double
Problem::TransientRun::bounded(double h) const
{
  h = std::min(std::max(h, _dt_min), _dt_max);
  return _stepper != Stepper::fixed ? std::min(h, _integrator.largestStep(_p._U)) : h;
}

SolveResult
Problem::TransientRun::trial(const Vector & from, double end, double h, bool continues)
{
  _p._time = end;
  SolveResult r = _integrator.step(_p._U, from, _p._time, h, continues);
  _total.total_iterations += r.total_iterations;
  _total.linear_iterations += r.linear_iterations;
  _total.history.insert(_total.history.end(), r.history.begin(), r.history.end());
  return r;
}

Problem::TransientRun::Decision
Problem::TransientRun::attempt(const Vector & old, double start, double size)
{
  Vector predicted;
  double factor = 0.0;
  if (_stepper == Stepper::error && _integrator.predict(size, predicted, factor))
    return predictedStep(old, start, size, predicted, factor);
  if (_stepper == Stepper::error)
    return doubledStep(old, start, size);
  return singleStep(old, start, size);
}

Problem::TransientRun::Decision
Problem::TransientRun::predictedStep(
    const Vector & old, double start, double size, const Vector & predicted, double factor)
{
  const SolveResult r = trial(old, start + size, size, false);
  if (!r.converged)
    return cutBack(size, r.total_iterations);
  return control(factor * _error.estimate(_p._U, predicted), size, r.total_iterations);
}

Problem::TransientRun::Decision
Problem::TransientRun::doubledStep(const Vector & old, double start, double size)
{
  const SolveResult coarse = trial(old, start + size, size, false);
  const Vector coarse_solution = _p._U;
  SolveResult fine_a, fine_b;
  if (coarse.converged)
  {
    _p._U = old;
    fine_a = trial(old, start + 0.5 * size, 0.5 * size, false);
    if (fine_a.converged)
    {
      const Vector half = _p._U;
      fine_b = trial(half, start + size, 0.5 * size, true);
    }
  }
  const int iterations =
      std::max({coarse.total_iterations, fine_a.total_iterations, fine_b.total_iterations});
  if (!coarse.converged || !fine_a.converged || !fine_b.converged)
    return cutBack(size, iterations);
  // Richardson estimate of the error of the two half steps, which are the more accurate answer
  // and the one kept.
  return control(
      _error.estimate(_p._U, coarse_solution) / (std::pow(2.0, _order) - 1.0), size, iterations);
}

Problem::TransientRun::Decision
Problem::TransientRun::singleStep(const Vector & old, double start, double size)
{
  const SolveResult r = trial(old, start + size, size, false);
  const int iterations = r.total_iterations;
  if (!r.converged)
    return cutBack(size, iterations);
  Decision d;
  d.iterations = iterations;
  if (_stepper == Stepper::courant)
  {
    // The step that gives the Courant number aimed at, at the velocity just reached.
    const double courant = _integrator.courantNumber(_p._U, size);
    d.uncapped =
        courant > 0 ? size * _tr.courant_number / courant : std::numeric_limits<double>::infinity();
    d.next_dt = std::min(d.uncapped, size * _tr.growth_factor);
  }
  else if (_stepper == Stepper::iteration)
  {
    if (iterations < _tr.optimal_iterations - _tr.iteration_window)
      d.next_dt = size * _tr.growth_factor;
    else if (iterations > _tr.optimal_iterations + _tr.iteration_window)
      d.next_dt = size * _tr.cutback_factor;
    else
      d.next_dt = size;
    d.uncapped = d.next_dt;
  }
  else
  {
    // The fixed stepper grows back to the given step after a cutback.
    d.uncapped = _tr.dt;
    d.next_dt = std::min(size * _tr.growth_factor, _tr.dt);
  }
  return d;
}

Problem::TransientRun::Decision
Problem::TransientRun::control(double estimate, double size, int iterations) const
{
  const double ratio =
      estimate > 0 ? kSafetyFactor * std::pow(_tr.error_tolerance / estimate, 1.0 / (_order + 1.0))
                   : _tr.growth_factor;
  Decision d;
  d.iterations = iterations;
  if (estimate > _tr.error_tolerance)
  {
    d.accepted = false;
    d.next_dt = size * std::max(kSmallestRetryFraction, ratio);
  }
  else
  {
    d.uncapped =
        estimate > 0 ? size * std::max(1.0, ratio) : std::numeric_limits<double>::infinity();
    d.next_dt = std::min(d.uncapped, size * _tr.growth_factor);
  }
  return d;
}

Problem::TransientRun::Decision
Problem::TransientRun::cutBack(double size, int iterations) const
{
  Decision d;
  d.accepted = false;
  d.next_dt = size * _tr.cutback_factor;
  d.iterations = iterations;
  return d;
}

bool
Problem::TransientRun::reject(const Decision & d, const Vector & old, double start)
{
  ++_total.rejected_steps;
  ++_rejected_in_a_row;
  _p._U = old;
  _p._time = start;
  // A rejected step asks for a smaller one, so the step chosen before a shortening is dropped.
  _resume = 0.0;
  _dt = std::max(d.next_dt, _dt_min);
  if (_options.verbose && _par.isRoot())
    std::cout << "  step rejected at t = " << start << "; retrying with dt = " << _dt << "\n";
  const bool stops = d.next_dt < _dt_min || _rejected_in_a_row > _tr.max_rejected_steps;
  if (!stops)
    return true;
  if (_options.error_on_divergence)
  {
    std::ostringstream os;
    os << "dualmesh: the transient solve failed at t = " << start << ". The step was rejected "
       << _rejected_in_a_row << " times in a row";
    if (d.next_dt < _dt_min)
      os << " and the step size reached " << d.next_dt << ", below the minimum " << _dt_min << ".";
    else
      os << ", more than max_rejected_steps = " << _tr.max_rejected_steps << ".";
    throw std::runtime_error(os.str());
  }
  return false;
}

void
Problem::TransientRun::accept(double size, int iterations)
{
  _rejected_in_a_row = 0;
  ++_step;
  ++_total.time_steps;
  _total.step_history.emplace_back(_p._time, size);
  if (_options.verbose && _par.isRoot())
    std::cout << "  step " << _step << ": t = " << _p._time << ", dt = " << size << ", iterations "
              << iterations << "\n";
  // A step whose solver reports success can still hold NaN or infinite values (an unstable
  // explicit term): the run stops there, before the values reach the output.
  if (_par.any(!_p._U.allFinite()))
  {
    std::ostringstream os;
    os << "dualmesh: the solution has NaN or infinite values after the step to t = " << _p._time
       << ". Either the time step is too large for the explicit terms (lower time_step or "
          "courant_number), or a property is not finite at the solution.";
    throw std::runtime_error(os.str());
  }
  _integrator.accept(_p._U, _p._time);
  _p.commitState();
  _p.notifyStepAccepted();
  _error.accept(_p._U);
  _outputs.writeUpTo(_p._time, _p._output_callback);
}

double
Problem::TransientRun::nextStep(const Decision & d, const Landing & landing)
{
  // A shortened step is not the choice of the controller, so its growth cap, a multiple of that
  // step, is no limit of the controller.  Where the cap binds, the steps after it return to the
  // step chosen before the shortening by at most a doubling each, and never beyond the step that
  // the controller allows without its cap (the step of the Courant number aimed at, or that the
  // error estimate allows).  Where the cap does not bind, the step the controller chose stands.
  // The step after a shortened one is at most twice it, so that two output times close together
  // do not leave a step ratio beyond the stability of the backward differences.
  double next_dt = d.next_dt;
  if (landing.shortened)
    _resume = std::max(_resume, _dt);
  if (_resume > 0.0)
  {
    next_dt = std::max(next_dt, std::min({_resume, d.uncapped, kShortenedGrowth * landing.size}));
    if (next_dt >= _resume || d.uncapped < _resume)
      _resume = 0.0;
  }
  if (landing.shortened)
    next_dt = std::min(next_dt, kShortenedGrowth * landing.size);
  return bounded(next_dt);
}

void
Problem::TransientRun::checkpoint() const
{
  if (_tr.checkpoint_file.empty())
    return;
  const bool last = _p._time >= _end;
  if ((_tr.checkpoint_interval > 0 && _step % _tr.checkpoint_interval == 0) || last)
  {
    writeCheckpoint(_tr.checkpoint_file, _p, _layout, {_p._time, _dt, _first_step + _step});
    if (_p._checkpoint_callback)
      _p._checkpoint_callback();
  }
}

SolveResult
Problem::runTransient(TimeIntegrator & integrator,
                      const TransientOptions & tr,
                      const SolverOptions & options,
                      const CheckpointLayout & layout,
                      int first_step)
{
  return TransientRun(*this, integrator, tr, options, layout, first_step).run();
}

SolveResult
Problem::solveTransient(const TransientOptions & transient, const SolverOptions & options)
{
  initialize();
  const Parallel & par = parallel();
  // The step is retried on failure, so the nonlinear solver must report a
  // failure rather than throw.
  SolverOptions attempt = options;
  attempt.error_on_divergence = false;
  const CheckpointLayout layout = par.checkpointLayout(*this);
  // A restart continues from the time, the step size and the state of its checkpoint.
  TransientOptions tr = transient;
  int first_step = 0;
  if (!tr.restart_file.empty())
  {
    const CheckpointTime point = readCheckpoint(tr.restart_file, *this, layout);
    tr.start_time = point.time;
    tr.dt = point.dt > 0 ? point.dt : tr.dt;
    first_step = point.step;
  }
  _time = tr.start_time;
  applyDirichlet(_U, 1.0);
  if (!tr.restart_file.empty() && hasSecondOrderVariables())
    throw InputError("A run with inertia does not restart from a checkpoint: the checkpoint "
                     "holds no velocity and acceleration.");
  // The run continues the last transient run of the problem when it starts from the state that
  // run reached, on every process.  It then keeps the history of the theta method (the step that
  // reached the state, and the velocity and acceleration there).  Otherwise the initial velocity
  // of a flow is projected onto a field without divergence, unless the run restarts from a
  // checkpoint.
  const bool continues = !par.any(!(_transient_end.size() == _U.size() && _transient_end == _U));
  std::optional<double> start_correction;
  if (tr.restart_file.empty() && !continues)
    start_correction = _flow->projectInitialVelocity(*this);
  TimeIntegrator * integrator = _time_integrator.get();
  if (!integrator)
  {
    if (!_theta || !continues)
      _theta = std::make_shared<ThetaIntegrator>(*this);
    _theta->configure(tr, attempt);
    integrator = _theta.get();
  }
  integrator->start(_U, _time);
  SolveResult result = runTransient(*integrator, tr, options, layout, first_step);
  result.start_correction = start_correction;
  _transient_end = _U;
  return result;
}

} // namespace dualmesh
