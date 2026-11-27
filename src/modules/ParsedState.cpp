// SPDX-License-Identifier: LGPL-2.1-or-later
//
// The internal states of a balance law dm/dt + div F = s, onto which the input lowers a model of
// your own: parsed_state, a property object that integrates the rate laws dq/dt = g of the states
// at each integration point.  The storage m of the law, which may read the states, is the kernel
// time_derivative.  The expressions are compiled once (FieldExpressions) and evaluated with
// automatic differentiation, so the Jacobian is exact.
#include "dualmesh/base/Factory.h"
#include "dualmesh/base/FieldExpression.h"
#include "dualmesh/base/Problem.h"
#include "dualmesh/base/Property.h"

#include <Eigen/Dense>

#include <array>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

namespace dualmesh
{

namespace
{

const char * kStateExpressionHelp =
    "The expressions use the fields by their names, the components of their gradients as "
    "grad_x(u), grad_y(u) and grad_z(u), the coordinates x, y and z, the time t, the registered "
    "functions, the element fields, the scalar properties of the property objects added before, "
    "and the named constants.";

} // namespace

// ---------------------------------------------------------------------------
/// Internal states q_i, each with a rate law dq_i/dt = g_i(q, u, grad u, x, t), integrated at each
/// integration point by the backward Euler method, and each declared as a scalar property.
class ParsedState : public Property
{
public:
  /// The most states of one object, the most iterations of the local Newton method and the most
  /// halvings of one of its steps.
  static constexpr int kMaxStates = 8;
  static constexpr int kMaxIterations = 50;
  static constexpr int kMaxHalvings = 30;
  using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, kMaxStates, kMaxStates>;
  using Column = Eigen::Matrix<double, Eigen::Dynamic, 1, 0, kMaxStates, 1>;

  static InputParameters validParams()
  {
    InputParameters p = Property::validParams();
    p.setClassDescription(
        std::string(
            "Internal states q_i that each integration point keeps from step to step, such as the "
            "occupancy of the traps of a diffusing species, each with a rate law dq_i/dt = g_i "
            "of the states and the fields. A step of a transient solve integrates the states by "
            "the backward Euler method, q_i - q_i_old - dt g_i(q) = 0, with q_i_old the state at "
            "the start of the step and dt the time step. A steady solve holds each state at the "
            "root of its rate, g_i(q) = 0, the equilibrium of the point. At each point the code "
            "solves these equations by Newton's method in plain numbers from q_old, with the "
            "derivatives dg_i/dq_j of 'rate_derivatives', and halves a step that does not lower "
            "the residual. The derivatives of the states with respect to the fields follow from "
            "the implicit function theorem, so the Jacobian is exact. Each state is the scalar "
            "property of its name, which kernels and later property objects read. The history "
            "of each point starts at 'initial_values'. A point at which the equations have no "
            "finite root, or at which " +
            std::to_string(kMaxIterations) +
            " iterations do not reach it, is inadmissible: Newton's method of the problem "
            "shortens its step there, or the step is cut. ") +
        kStateExpressionHelp);
    p.addRequired("states",
                  ParameterKind::StringList,
                  "The names of the states, at most " + std::to_string(kMaxStates) +
                      ", which the expressions and the declared properties take.");
    p.addRequired("rates", ParameterKind::StringList, "The rate g_i of each state, an expression.");
    p.addRequired("rate_derivatives",
                  ParameterKind::StringList,
                  "The derivatives dg_i/dq_j of the rates by the states at fixed fields, row by "
                  "row: the derivatives of the first rate by every state, then of the second.");
    p.addRequired("initial_values",
                  ParameterKind::RealList,
                  "The value of each state at the start of the run, at every point.");
    p.addOptional("constants",
                  ParameterKind::RealMap,
                  RealMap{},
                  "Named constants that the expressions use, as name -> value pairs. Default: "
                  "none, for expressions that need no constant of their own.");
    return p;
  }

  explicit ParsedState(const InputParameters & p)
      : Property(p), _states(p.getStringList("states")), _rates(p.getStringList("rates")),
        _derivatives(p.getStringList("rate_derivatives")), _initial(p.getRealList("initial_values"))
  {
    const std::size_t n = _states.size();
    if (n == 0 || n > static_cast<std::size_t>(kMaxStates))
      throw InputError("'" + name() + "': give from 1 to " + std::to_string(kMaxStates) +
                       " states.");
    if (_rates.size() != n || _initial.size() != n)
      throw InputError("'" + name() + "': give one rate and one initial value per state.");
    if (_derivatives.size() != n * n)
      throw InputError("'" + name() + "': give " + std::to_string(n * n) +
                       " rate derivatives, the derivative of each rate by each state.");
  }

  void initialSetup(Problem & problem) override
  {
    Property::initialSetup(problem);
    // The states are constants of the expressions while they are compiled, since they are
    // declared after this object is set up.  A state that the object of another block declared
    // is a property of the problem already, which the expressions find by its name.  The
    // evaluation overwrites the argument of a state either way.
    FieldExpressions::Names names;
    for (const auto & state : _states)
      if (!problem.propertyRegistry().has(state))
        names.constants.emplace_back(state, 0.0);
    for (const auto & [n, value] : _params.getRealMap("constants"))
      names.constants.emplace_back(n, value);
    std::vector<std::string> texts = _rates;
    texts.insert(texts.end(), _derivatives.begin(), _derivatives.end());
    _expressions.compile(problem, texts, names, name());
    _slots.clear();
    for (const auto & state : _states)
      _slots.push_back(_expressions.argumentIndex(state));
  }

  void declareProperties(PropertyRegistry & r) override
  {
    _props.clear();
    for (const auto & state : _states)
      _props.push_back(r.declare(state, 1));
  }

  /// The history record holds q - q_initial, so that a record that starts at zero starts each
  /// state at its initial value.
  int stateSize() const override { return static_cast<int>(_states.size()); }

  /// State @p i at the start of the step at the point of @p ctx: the history of the point, or
  /// the initial value at a point without one.
  double startValue(const QpContext & ctx, std::size_t i) const
  {
    const double held = ctx.state_old ? ctx.state_old[stateOffset() + static_cast<int>(i)] : 0.0;
    return _initial[i] + held;
  }
  bool startValue(const QpContext & ctx, int property, double & value) const override
  {
    for (std::size_t i = 0; i < _props.size(); ++i)
      if (_props[i] == property)
      {
        value = startValue(ctx, i);
        return true;
      }
    return false;
  }

  void computeProperties(QpContext & ctx) const override
  {
    const int n = static_cast<int>(_states.size());
    Column start(n);
    for (int i = 0; i < n; ++i)
      start[i] = startValue(ctx, static_cast<std::size_t>(i));
    // A point without its own history (an output at a centroid, a point outside the assembly),
    // the old time of a step of the theta method and an evaluation outside a solve keep the
    // state at the start of the step.
    const bool own = ctx.state_old && ctx.state_new && ctx.state_key != Problem::kCentroidStateKey;
    const bool step = ctx.dt > 0.0 && !ctx.old_level && !ctx.steady_solve;
    if (!own || !(step || ctx.steady_solve))
    {
      for (int i = 0; i < n; ++i)
        ctx.property(_props[i]) = ADReal(start[i]);
      return;
    }
    // The residual r_i = c (q_i - q_i_old) - g_i(q): c = 1/dt in a step, 0 in a steady solve.
    const double c = step ? 1.0 / ctx.dt : 0.0;
    std::array<ADReal, FieldExpressions::kMaxArguments> args;
    _expressions.arguments(ctx, args.data());
    std::array<double, FieldExpressions::kMaxArguments> values{};
    for (std::size_t k = 0; k < _expressions.numArguments(); ++k)
      values[k] = args[k].value();
    const Column q = solve(ctx, start, c, values.data());
    setStates(ctx, q, start, c, values.data(), args.data());
  }

private:
  /// Put the states @p q into the arguments @p values of the expressions.
  void place(const Column & q, double * values) const
  {
    for (int i = 0; i < q.size(); ++i)
      if (_slots[i] >= 0)
        values[_slots[i]] = q[i];
  }

  /// The residual @p r at the states @p q, and whether every component is at the round-off of
  /// its terms.
  bool residual(const Column & q, const Column & start, double c, double * values, Column & r) const
  {
    place(q, values);
    const double tolerance = 64.0 * std::numeric_limits<double>::epsilon();
    bool small = true;
    for (int i = 0; i < q.size(); ++i)
    {
      r[i] = c * (q[i] - start[i]) - _expressions.value(static_cast<std::size_t>(i), values);
      const double size = c * (std::abs(q[i]) + std::abs(start[i])) +
                          _expressions.termSize(static_cast<std::size_t>(i), values);
      small = small && std::abs(r[i]) <= tolerance * size;
    }
    return small;
  }

  /// The derivative @p J of the residual by the states, at the states @p q.
  void jacobian(const Column & q, double c, double * values, Matrix & J) const
  {
    place(q, values);
    const int n = static_cast<int>(q.size());
    for (int i = 0; i < n; ++i)
      for (int j = 0; j < n; ++j)
        J(i, j) = (i == j ? c : 0.0) -
                  _expressions.value(static_cast<std::size_t>(n + i * n + j), values);
  }

  /// The root of the residual by Newton's method in plain numbers from the states @p start at
  /// the start of the step, with each step halved until the norm of the residual falls.
  Column solve(const QpContext & ctx, const Column & start, double c, double * values) const
  {
    const int n = static_cast<int>(start.size());
    Column q = start, r(n), trial(n), trial_r(n), dq(n);
    Matrix J(n, n);
    for (int iteration = 0;; ++iteration)
    {
      const bool converged = residual(q, start, c, values, r);
      if (!r.allFinite())
        fail(ctx, "the rates are not finite");
      if (converged)
        break;
      if (iteration == kMaxIterations)
        fail(ctx, std::to_string(kMaxIterations) + " iterations did not reach the root");
      jacobian(q, c, values, J);
      const Eigen::FullPivLU<Matrix> lu(J);
      if (!lu.isInvertible())
        fail(ctx, "the derivative of the residual by the states is singular");
      dq = -lu.solve(r);
      if (!dq.allFinite())
        fail(ctx, "the Newton step is not finite");
      // Halve the step until the norm of the residual falls (the Armijo condition).
      const double norm = r.norm();
      double fraction = 1.0;
      for (int halving = 0;; ++halving)
      {
        trial = q + fraction * dq;
        residual(trial, start, c, values, trial_r);
        if (trial_r.allFinite() && trial_r.norm() <= (1.0 - 1e-4 * fraction) * norm)
          break;
        if (halving == kMaxHalvings)
          fail(ctx, "no step along the Newton direction lowers the residual");
        fraction *= 0.5;
      }
      // A step at the round-off of the states ends the iteration.
      bool tiny = true;
      for (int i = 0; i < n; ++i)
        tiny = tiny && std::abs(trial[i] - q[i]) <= 64.0 * std::numeric_limits<double>::epsilon() *
                                                        (std::abs(trial[i]) + std::abs(start[i]));
      q = trial;
      if (tiny)
        break;
    }
    return q;
  }

  /// The states as properties and the history at the root @p q, by the implicit function theorem
  /// in one step of automatic differentiation: the root as a constant, less J^-1 r with r at the
  /// root carrying the derivatives of the fields in @p args.  Its derivative is -J^-1 dr/du, and
  /// its value is one more Newton step from the root.
  void setStates(QpContext & ctx,
                 const Column & q,
                 const Column & start,
                 double c,
                 double * values,
                 ADReal * args) const
  {
    const int n = static_cast<int>(q.size());
    Matrix J(n, n);
    jacobian(q, c, values, J);
    const Eigen::FullPivLU<Matrix> lu(J);
    if (!lu.isInvertible())
      fail(ctx, "the derivative of the residual by the states is singular");
    const Matrix inverse = lu.inverse();
    for (int i = 0; i < n; ++i)
      if (_slots[i] >= 0)
        args[_slots[i]] = ADReal(q[i]);
    std::array<ADReal, kMaxStates> r_ad;
    for (int i = 0; i < n; ++i)
      r_ad[i] = c * (q[i] - start[i]) - _expressions.value(static_cast<std::size_t>(i), args);
    for (int i = 0; i < n; ++i)
    {
      ADReal value(q[i]);
      for (int j = 0; j < n; ++j)
        value -= inverse(i, j) * r_ad[j];
      ctx.property(_props[i]) = value;
      ctx.state_new[stateOffset() + i] = q[i] - _initial[i];
    }
  }

  [[noreturn]] void fail(const QpContext & ctx, const std::string & why) const
  {
    throw InadmissibleStateError("'" + name() + "': no value of the states solves their rate " +
                                 "laws at the point (" + std::to_string(ctx.x[0]) + ", " +
                                 std::to_string(ctx.x[1]) + ", " + std::to_string(ctx.x[2]) +
                                 "): " + why + ".");
  }

  std::vector<std::string> _states;
  std::vector<std::string> _rates;
  std::vector<std::string> _derivatives;
  std::vector<double> _initial;
  FieldExpressions _expressions;
  std::vector<int> _slots;
  std::vector<int> _props;
};

void
registerParsedStateObjects(Factory & f)
{
  f.add<ParsedState>("parsed_state", ObjectCategory::Property, "framework");
}

} // namespace dualmesh
