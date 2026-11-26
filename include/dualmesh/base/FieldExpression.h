// SPDX-License-Identifier: LGPL-2.1-or-later
//
// Expressions of the fields of a problem, compiled once and evaluated with automatic differentiation at an integration point.
// The names in scope are every field of the problem, the components of its gradient (grad_x(u), grad_y(u) and grad_z(u), up to the dimension of the mesh), the coordinates x, y and z, the time t, the eigenvalue of an eigenvalue study (eigenvalue, 0 in every other study), every registered function, every element field, every scalar property declared so far, and the constants that the owner names.
// A name is found where it is defined, so an expression needs no list of the names it uses; a name defined twice is refused only when an expression uses it.
// The fields and their gradients take their values at the current iterate, with their derivatives, under Newton's method and Picard iteration alike. An expression may be a term of an equation (the flux of parsed_kernel or of a Neumann condition) as well as a coefficient, and a term evaluated at the previous iterate would carry no derivative of its own unknown into the matrix of Picard iteration. Both iterations converge to the same solution, since the residual takes the current iterate in both.
// The expressions of one owner share one table of arguments: each expression is compiled against every name in scope, and then again against the names that the expressions use, so that an evaluation fills only those.
#pragma once

#include "dualmesh/base/QpContext.h"
#include "dualmesh/core/Function.h"
#include "dualmesh/core/InputParameters.h"
#include "dualmesh/core/ParsedFunction.h"

#include <array>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace dualmesh
{

class Problem;

class FieldExpressions
{
public:
  static constexpr int kMaxArguments = 64;

  /// The names of the owner's constants, which the expressions may use besides the names of the problem.
  struct Names
  {
    std::vector<std::pair<std::string, double>> constants;
  };

  /// Compile @p texts for @p problem. @p owner names the object in the messages.
  void compile(const Problem & problem,
               const std::vector<std::string> & texts,
               const Names & names,
               const std::string & owner);

  std::size_t size() const { return _expressions.size(); }

  /// Fill the arguments at an integration point; @p out holds at least kMaxArguments numbers.
  void arguments(const QpContext & ctx, ADReal * out) const;
  /// The same at the position @p x in place of the position of the integration point: the fields
  /// are those of the integration point (a section point of a plate or a beam, off its mid-surface).
  void arguments(const QpContext & ctx, const Point & x, ADReal * out) const;
  /// The value of expression @p i, with the arguments filled by arguments().
  ADReal value(std::size_t i, const ADReal * arguments) const
  {
    return _expressions[i]->value(arguments);
  }
  /// The value of expression @p i in plain numbers, with the values of the arguments that
  /// arguments() fills, in the same order.
  double value(std::size_t i, const double * values) const
  {
    return _expressions[i]->value(values);
  }
  /// The size of the terms of expression @p i (ParsedExpression::termSize), with the arguments
  /// filled by arguments().
  double termSize(std::size_t i, const ADReal * arguments) const
  {
    std::array<double, kMaxArguments> values{};
    for (std::size_t k = 0; k < _arguments.size(); ++k)
      values[k] = arguments[k].value();
    return termSize(i, values.data());
  }
  /// The same with the values of the arguments in plain numbers.
  double termSize(std::size_t i, const double * values) const
  {
    return _expressions[i]->termSize(values);
  }
  /// The number of arguments that arguments() fills.
  std::size_t numArguments() const { return _arguments.size(); }
  /// The place of the name @p name in the arguments that arguments() fills, or -1 when no
  /// expression uses it.
  int argumentIndex(const std::string & name) const;
  /// Whether an expression reads a property.
  bool readsProperties() const
  {
    for (const auto & a : _arguments)
      if (a.kind == Argument::Property)
        return true;
    return false;
  }
  /// Fill the arguments with the state at the start of a step: each field and the components of
  /// its gradient at their old values (QpContext::u_old, grad_u_old), the time t - dt, and each
  /// property by @p start, called with the place of the property in the registry, since only the
  /// property object that owns a property knows its value at the start of the step.  The
  /// arguments carry no derivatives.
  void startArguments(const QpContext & ctx,
                      const std::function<double(int)> & start,
                      ADReal * out) const;
  /// The value of expression @p i at an integration point.
  ADReal value(std::size_t i, const QpContext & ctx) const;
  /// The value of expression @p i with the fields of an integration point at the position @p x.
  ADReal valueAt(std::size_t i, const QpContext & ctx, const Point & x) const;

private:
  struct Argument
  {
    enum Kind
    {
      Value,
      Gradient,
      Coordinate,
      Time,
      Eigenvalue,
      Function,
      ElementField,
      Property,
      Constant
    } kind;
    int index = 0;
    int component = 0;
    double constant = 0.0;
    FunctionPtr function;
    const std::vector<double> * field = nullptr;
  };

  std::vector<Argument> _arguments;
  /// The names of the arguments, in their order.
  std::vector<std::string> _names;
  std::vector<std::shared_ptr<ParsedExpression>> _expressions;
};

/// A coefficient of an object: a number, a registered function of position and time, or an expression of the fields, their gradients, x, y, z and t.
/// A text that is not the name of a registered function is compiled as an expression.
class FieldCoefficient
{
public:
  /// Set up from the parameter @p param of @p params, for @p problem. @p owner names the object in the messages.
  void setup(const Problem & problem,
             const InputParameters & params,
             const std::string & param,
             const std::string & owner);
  /// Set up from a text: a number, the name of a registered function, or an expression (which may be the name of a property).
  void setup(const Problem & problem, const std::string & text, const std::string & owner);
  ADReal value(const QpContext & ctx) const
  {
    return _function ? ADReal(_function->value(ctx.x, ctx.time)) : _expression.value(0, ctx);
  }
  /// The value with the fields of the integration point @p ctx at the position @p x.
  ADReal valueAt(const QpContext & ctx, const Point & x) const
  {
    return _function ? ADReal(_function->value(x, ctx.time)) : _expression.valueAt(0, ctx, x);
  }
  /// Whether the coefficient is an expression that reads a property.
  bool readsProperties() const { return !_function && _expression.readsProperties(); }

private:
  FunctionPtr _function;
  FieldExpressions _expression;
};

} // namespace dualmesh
