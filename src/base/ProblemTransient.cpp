// SPDX-License-Identifier: LGPL-2.1-or-later
#include "dualmesh/base/Problem.h"
#include "dualmesh/parallel/Checkpoint.h"

#include <algorithm>
#include <cmath>
#include <deque>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>

namespace dualmesh
{

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
  void start(const Vector & solution, double time) override;
  SolveResult
  step(Vector & solution, const Vector & old, double time, double dt, bool continues) override;
  void accept(const Vector & solution, double time) override;
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
  _dynamic = std::any_of(_problem._second_order_variables.begin(),
                         _problem._second_order_variables.end(),
                         [](char c) { return c != 0; });
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
Problem::ThetaIntegrator::start(const Vector & solution, double time)
{
  (void) time;
  _accepted = {{solution, 0.0}};
  _trial.reset();
}

SolveResult
Problem::ThetaIntegrator::step(
    Vector & solution, const Vector & old, double time, double dt, bool continues)
{
  // The nonlinear solve advances the problem's own solution at the problem's time.
  (void) solution;
  (void) time;
  Problem & p = _problem;
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
    begin.v = p._initial_velocity.size() == old.size() ? p._initial_velocity
                                                       : Vector(Vector::Zero(old.size()));
    // The problem's clock is at the end of the step; the start state belongs to its start, where
    // the prescribed values give their velocity and acceleration and the loads their size.
    const double new_time = p._time;
    p._time = new_time - dt;
    try
    {
      begin.a = p.initialAcceleration(old, begin.v, dt, _options);
    }
    catch (...)
    {
      p._time = new_time;
      throw;
    }
    p._time = new_time;
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
    const double new_time = p._time;
    p._time = new_time - dt;
    try
    {
      ao.reference = &old_residual.reference;
      ao.term_size = &old_residual.term_size;
      p.assemble(old, ao, old_residual.residual, nullptr);
    }
    catch (...)
    {
      p._time = new_time;
      throw;
    }
    p._time = new_time;
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
  base.include_steady_terms =
      _theta > 0 ||
      std::any_of(p._new_time_only.begin(), p._new_time_only.end(), [](char c) { return c != 0; });
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
Problem::ThetaIntegrator::accept(const Vector & solution, double time)
{
  (void) time;
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

SolveResult
Problem::runTransient(TimeIntegrator & integrator,
                      const TransientOptions & tr,
                      const SolverOptions & options,
                      const CheckpointLayout & layout,
                      int first_step)
{
  if (tr.dt <= 0)
    throw InputError("Transient: the time step must be positive.");
  if (tr.theta < 0 || tr.theta > 1)
    throw InputError("Transient: theta must be in [0, 1].");
  const bool error_control = tr.time_stepper == "error";
  const bool iteration_control = tr.time_stepper == "iteration";
  const bool courant_control = tr.time_stepper == "cfl";
  if (!error_control && !iteration_control && !courant_control && tr.time_stepper != "fixed")
    throw InputError("Transient: unknown time stepper '" + tr.time_stepper +
                     "' (use fixed, error, iteration or cfl).");
  if (courant_control && !integrator.measuresCourantNumber())
    throw InputError("Transient: the cfl stepper needs a time integration that measures the "
                     "Courant number, the projection integration of a flow.");
  if (courant_control && !(tr.courant_number > 0))
    throw InputError("Transient: the Courant number of the cfl stepper must be positive.");
  if (tr.growth_factor < 1.0)
    throw InputError("Transient: growth_factor must be at least one.");
  if (tr.cutback_factor <= 0.0 || tr.cutback_factor >= 1.0)
    throw InputError("Transient: cutback_factor must lie strictly between zero and one.");
  const Parallel & par = parallel();

  // The scale of each equation in the error of a step: the largest of the range of its variables,
  // their change from the start of the run and the square root of the machine epsilon times their
  // magnitude, the largest that the accepted states and the step have reached, with the
  // components of a vector unknown together as in the convergence test of the nonlinear solver.
  // The change is offset-free, so a temperature in kelvin is not measured against its 300 K, and
  // a field that is uniform, or zero to round-off, has the scale of its motion (a rigid motion, a
  // uniform flow).  The floor keeps the round-off of a field at rest from counting as an error.
  // The error of a step is that of the variables with a time derivative, of the first or the
  // second order.  A variable without one (the pressure of an incompressible flow) is algebraic:
  // it has no truncation error of its own, and the theta method leaves its value at the new time
  // with oscillations that are no error of the step, so it is left out of the test (as the error
  // test of IDA can leave out the algebraic variables).  A problem without a time derivative
  // measures every variable.  Every process takes the same numbers, so they all take the same
  // steps and accept or reject together.
  std::vector<char> differential(_new_time_only.size(), 0);
  for (std::size_t v = 0; v < differential.size(); ++v)
    differential[v] = !_new_time_only[v] || _second_order_variables[v];
  if (std::none_of(differential.begin(), differential.end(), [](char c) { return c != 0; }))
    differential.clear();
  const std::vector<int> equation_of = vectorEquations();
  const Vector initial = _U;
  VariableSizes reached = variableSizes(_U, initial);
  const auto largest = [](VariableSizes & to, const VariableSizes & from)
  {
    for (std::size_t v = 0; v < to.range.size(); ++v)
    {
      to.range[v] = std::max(to.range[v], from.range[v]);
      to.change[v] = std::max(to.change[v], from.change[v]);
      to.magnitude[v] = std::max(to.magnitude[v], from.magnitude[v]);
    }
  };
  const auto error = [&](const Vector & after, const Vector & before)
  {
    VariableSizes sizes = variableSizes(after, initial);
    largest(sizes, reached);
    const double floor = std::sqrt(std::numeric_limits<double>::epsilon());
    std::vector<double> scale(sizes.range.size(), 0.0);
    for (std::size_t v = 0; v < scale.size(); ++v)
    {
      const std::size_t e = static_cast<std::size_t>(equation_of[v]);
      scale[e] = std::max({scale[e], sizes.range[v], sizes.change[v], floor * sizes.magnitude[v]});
    }
    const std::vector<double> step = variableSizes(after, before).change;
    double estimate = 0.0;
    for (std::size_t v = 0; v < step.size(); ++v)
    {
      if ((!differential.empty() && !differential[v]) || !(step[v] > 0.0))
        continue;
      const double s = scale[static_cast<std::size_t>(equation_of[v])];
      estimate =
          std::max(estimate, s > 0.0 ? step[v] / s : std::numeric_limits<double>::infinity());
    }
    return estimate;
  };

  const double span = tr.end_time - tr.start_time;
  const double dt_min = tr.dt_min > 0 ? tr.dt_min : tr.dt * 1.0e-6;
  const double dt_max = tr.dt_max > 0 ? tr.dt_max : std::abs(span);
  const double order = integrator.order();

  SolveResult total;
  _time = tr.start_time;
  int step = 0;
  // A trial step from @p from to @p end of size @p h: the last accepted state, or, when it
  // continues, the state of the step just taken.
  const auto trial = [&](const Vector & from, double end, double h, bool continues)
  {
    _time = end;
    SolveResult r = integrator.step(_U, from, _time, h, continues);
    total.total_iterations += r.total_iterations;
    total.linear_iterations += r.linear_iterations;
    total.history.insert(total.history.end(), r.history.begin(), r.history.end());
    return r;
  };

  double dt = std::min(std::max(tr.dt, dt_min), dt_max);
  int rejected_in_a_row = 0;

  // The output times: the solution is written at each of them, and the steps
  // land on them.  The times at the start are written before the first step.
  const double time_tolerance = 1e-10 * std::max(std::abs(span), 1e-300);
  for (std::size_t k = 1; k < tr.output_times.size(); ++k)
    if (!(tr.output_times[k] > tr.output_times[k - 1]))
      throw InputError("Transient: the output times must increase.");
  std::size_t next_output = 0;
  // An output time before the start belongs to an earlier run (a restart continues one), so only the times at the start are written.
  while (next_output < tr.output_times.size() &&
         tr.output_times[next_output] <= _time + time_tolerance)
  {
    if (_output_callback && tr.output_times[next_output] >= _time - time_tolerance)
      _output_callback(static_cast<int>(next_output));
    ++next_output;
  }

  while (_time < tr.end_time - 1e-12 * std::abs(span))
  {
    // Land exactly on the next output time or on the end time, and do not
    // leave a sliver behind: if one step would nearly reach the target, reach
    // it.
    const double target = next_output < tr.output_times.size()
                              ? std::min(tr.output_times[next_output], tr.end_time)
                              : tr.end_time;
    const double remaining = target - _time;
    double step_size = std::min(dt, remaining);
    if (remaining - step_size < 1e-8 * std::abs(span))
      step_size = remaining;
    // A step that is shortened to land on an output time is not the choice of
    // the step controller, so the step after it starts from the size that the
    // controller chose.
    const bool shortened = step_size < dt;

    const Vector old = _U;
    const double start = _time;
    bool accepted = true;
    double next_dt = dt;
    int iterations = 0;
    const auto cutBack = [&]
    {
      accepted = false;
      next_dt = step_size * tr.cutback_factor;
    };
    // The decision of the error controller on the estimate of the local error of the step.  A
    // rejected step is retried at the size that the estimate asks for, at least a tenth of the
    // step (Hairer, Norsett and Wanner 1993, Sect. II.4).
    const auto control = [&](double estimate)
    {
      const double ratio = estimate > 0
                               ? 0.9 * std::pow(tr.error_tolerance / estimate, 1.0 / (order + 1.0))
                               : tr.growth_factor;
      if (estimate > tr.error_tolerance)
      {
        accepted = false;
        next_dt = step_size * std::max(0.1, ratio);
      }
      else
        next_dt = step_size * std::min(tr.growth_factor, std::max(1.0, ratio));
    };
    // A step with a predictor takes one solve, and its difference from the predictor, times the
    // factor of the predictor, estimates its local error (Milne's device).
    Vector predicted;
    double factor = 0.0;
    if (error_control && integrator.predict(step_size, predicted, factor))
    {
      const SolveResult r = trial(old, start + step_size, step_size, false);
      iterations = r.total_iterations;
      if (!r.converged)
        cutBack();
      else
        control(factor * error(_U, predicted));
    }
    else if (error_control)
    {
      // One coarse step, and two half steps from the same starting state.
      const SolveResult coarse = trial(old, start + step_size, step_size, false);
      const Vector coarse_solution = _U;
      SolveResult fine_a, fine_b;
      if (coarse.converged)
      {
        _U = old;
        fine_a = trial(old, start + 0.5 * step_size, 0.5 * step_size, false);
        if (fine_a.converged)
        {
          const Vector half = _U;
          fine_b = trial(half, start + step_size, 0.5 * step_size, true);
        }
      }
      iterations =
          std::max({coarse.total_iterations, fine_a.total_iterations, fine_b.total_iterations});
      if (!coarse.converged || !fine_a.converged || !fine_b.converged)
        cutBack();
      else
        // Richardson estimate of the error of the two half steps, which are the more accurate
        // answer and the one kept.
        control(error(_U, coarse_solution) / (std::pow(2.0, order) - 1.0));
    }
    else
    {
      const SolveResult r = trial(old, start + step_size, step_size, false);
      iterations = r.total_iterations;
      if (!r.converged)
        cutBack();
      else if (courant_control)
      {
        // The step that gives the Courant number aimed at, at the velocity just reached.
        const double courant = integrator.courantNumber(_U, step_size);
        next_dt =
            courant > 0 ? step_size * tr.courant_number / courant : step_size * tr.growth_factor;
        next_dt = std::min(next_dt, step_size * tr.growth_factor);
      }
      else if (iteration_control)
      {
        if (iterations < tr.optimal_iterations - tr.iteration_window)
          next_dt = step_size * tr.growth_factor;
        else if (iterations > tr.optimal_iterations + tr.iteration_window)
          next_dt = step_size * tr.cutback_factor;
        else
          next_dt = step_size;
      }
      else
        // The fixed stepper grows back to the given step after a cutback.
        next_dt = std::min(std::max(dt, step_size * tr.growth_factor), tr.dt);
    }

    if (!accepted)
    {
      ++total.rejected_steps;
      ++rejected_in_a_row;
      _U = old;
      _time = start;
      dt = std::max(next_dt, dt_min);
      if (options.verbose && par.isRoot())
        std::cout << "  step rejected at t = " << start << "; retrying with dt = " << dt << "\n";
      if (next_dt < dt_min || rejected_in_a_row > tr.max_rejected_steps)
      {
        if (options.error_on_divergence)
        {
          std::ostringstream os;
          os << "dualmesh: the transient solve failed at t = " << start
             << ". The step was rejected " << rejected_in_a_row << " times in a row";
          if (next_dt < dt_min)
            os << " and the step size reached " << next_dt << ", below the minimum " << dt_min
               << ".";
          else
            os << ", more than max_rejected_steps = " << tr.max_rejected_steps << ".";
          throw std::runtime_error(os.str());
        }
        return total;
      }
      continue;
    }

    rejected_in_a_row = 0;
    ++step;
    ++total.time_steps;
    total.step_history.emplace_back(_time, step_size);
    if (options.verbose && par.isRoot())
      std::cout << "  step " << step << ": t = " << _time << ", dt = " << step_size
                << ", iterations " << iterations << "\n";
    // A step whose solver reports success can still hold NaN or infinite values (an unstable
    // explicit term): the run stops there, before the values reach the output.
    if (par.any(!_U.allFinite()))
    {
      std::ostringstream os;
      os << "dualmesh: the solution has NaN or infinite values after the step to t = " << _time
         << ". Either the time step is too large for the explicit terms (lower time_step or "
            "courant_number), or a property is not finite at the solution.";
      throw std::runtime_error(os.str());
    }
    integrator.accept(_U, _time);
    commitState();
    notifyStepAccepted();
    largest(reached, variableSizes(_U, initial));
    while (next_output < tr.output_times.size() &&
           tr.output_times[next_output] <= _time + time_tolerance)
    {
      if (_output_callback)
        _output_callback(static_cast<int>(next_output));
      ++next_output;
    }
    if (shortened)
      next_dt = std::max(next_dt, dt);
    dt = std::min(std::max(next_dt, dt_min), dt_max);
    if (!tr.checkpoint_file.empty())
    {
      const bool last = _time >= tr.end_time - 1e-12 * std::abs(span);
      if ((tr.checkpoint_interval > 0 && step % tr.checkpoint_interval == 0) || last)
        writeCheckpoint(tr.checkpoint_file, *this, layout, {_time, dt, first_step + step});
    }
    total.next_time_step = dt;
    if (tr.max_steps > 0 && step >= tr.max_steps)
      break;
  }
  total.converged = true;
  return total;
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
  if (!tr.restart_file.empty() && std::any_of(_second_order_variables.begin(),
                                              _second_order_variables.end(),
                                              [](char c) { return c != 0; }))
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
    start_correction = projectInitialVelocity();
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

std::optional<double>
Problem::projectInitialVelocity()
{
  initialize();
  if (_start_projections.empty())
    return std::nullopt;
  applyDirichlet(_U, 1.0);
  double correction = 0.0;
  for (const auto & project : _start_projections)
    correction = std::max(correction, project(_U));
  return correction;
}

} // namespace dualmesh
