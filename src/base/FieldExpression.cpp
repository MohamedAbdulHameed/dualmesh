// SPDX-License-Identifier: LGPL-2.1-or-later
#include "dualmesh/base/FieldExpression.h"

#include "dualmesh/base/Problem.h"
#include "dualmesh/base/Property.h"

#include <algorithm>
#include <array>
#include <memory>
#include <set>

namespace dualmesh
{

void
FieldExpressions::compile(const Problem & problem,
                          const std::vector<std::string> & texts,
                          const Names & extra,
                          const std::string & owner)
{
  const int dim = problem.mesh().spaceDimension();

  // Every name in scope, with every place it is defined.
  std::vector<std::string> names;
  std::vector<std::vector<Argument>> candidates;
  std::vector<std::vector<std::string>> kinds;
  const auto add = [&](const std::string & n, Argument a, const std::string & kind)
  {
    const auto it = std::find(names.begin(), names.end(), n);
    if (it == names.end())
    {
      names.push_back(n);
      candidates.push_back({std::move(a)});
      kinds.push_back({kind});
      return;
    }
    const auto k = static_cast<std::size_t>(it - names.begin());
    candidates[k].push_back(std::move(a));
    kinds[k].push_back(kind);
  };
  const char * components[3] = {"grad_x", "grad_y", "grad_z"};
  for (int v = 0; v < problem.numVariables(); ++v)
  {
    const std::string & field = problem.variable(v).name;
    add(field, {Argument::Value, v, 0, 0.0, nullptr}, "a field");
    for (int c = 0; c < dim; ++c)
      add(std::string(components[c]) + "(" + field + ")",
          {Argument::Gradient, v, c, 0.0, nullptr},
          "a gradient");
  }
  const char * coordinates[3] = {"x", "y", "z"};
  for (int c = 0; c < 3; ++c)
    add(coordinates[c], {Argument::Coordinate, 0, c, 0.0, nullptr}, "a coordinate");
  add("t", {Argument::Time, 0, 0, 0.0, nullptr}, "the time");
  add("eigenvalue", {Argument::Eigenvalue, 0, 0, 0.0, nullptr}, "the eigenvalue");
  for (const auto & [n, value] : extra.constants)
    add(n, {Argument::Constant, 0, 0, value, nullptr}, "a constant");
  for (const auto & f : problem.functionNames())
    add(f, {Argument::Function, 0, 0, 0.0, problem.function(f)}, "a function");
  for (const auto & f : problem.elementFieldNames())
  {
    Argument a{Argument::ElementField, 0, 0, 0.0, nullptr};
    a.field = problem.findElementField(f);
    add(f, a, "an element field");
  }
  for (const auto & [n, slot] : problem.propertyRegistry().all())
    if (slot.second == 1)
      add(n, {Argument::Property, slot.first, 0, 0.0, nullptr}, "a property");

  // Compile against every name to find the names in use, then again against those only.
  std::set<int> used;
  for (const auto & text : texts)
  {
    std::unique_ptr<ParsedExpression> all;
    try
    {
      all = std::make_unique<ParsedExpression>(text, names);
    }
    catch (const InputError & error)
    {
      throw InputError("'" + owner + "': " + error.what() +
                       " An expression may use the fields, their gradients (grad_x(u)), x, y, z, "
                       "t, the registered functions, the element fields, the properties of the "
                       "property objects added before it, and the named constants.");
    }
    for (std::size_t k = 0; k < names.size(); ++k)
      if (all->uses(static_cast<int>(k)))
        used.insert(static_cast<int>(k));
  }
  if (static_cast<int>(used.size()) > kMaxArguments)
    throw InputError("'" + owner + "': the expressions use " + std::to_string(used.size()) +
                     " names, and at most " + std::to_string(kMaxArguments) + " are allowed.");
  std::vector<std::string> used_names;
  _arguments.clear();
  for (int k : used)
  {
    if (candidates[k].size() > 1)
    {
      std::string what;
      for (std::size_t i = 0; i < kinds[k].size(); ++i)
        what += (i ? (i + 1 == kinds[k].size() ? " and " : ", ") : "") + kinds[k][i];
      throw InputError("'" + owner + "': the name '" + names[k] + "' is " + what +
                       ", so the expression is ambiguous. Rename one of them.");
    }
    used_names.push_back(names[k]);
    _arguments.push_back(candidates[k].front());
    if (candidates[k].front().kind == Argument::Property)
      problem.propertyRegistry().markRead(names[k]);
  }
  _expressions.clear();
  for (const auto & text : texts)
    _expressions.push_back(std::make_shared<ParsedExpression>(text, used_names));
}

void
FieldExpressions::arguments(const QpContext & ctx, ADReal * out) const
{
  arguments(ctx, ctx.x, out);
}

void
FieldExpressions::arguments(const QpContext & ctx, const Point & x, ADReal * out) const
{
  for (std::size_t k = 0; k < _arguments.size(); ++k)
  {
    const Argument & a = _arguments[k];
    switch (a.kind)
    {
    case Argument::Value:
      out[k] = ctx.value(a.index);
      break;
    case Argument::Gradient:
      out[k] = ctx.gradient(a.index)[a.component];
      break;
    case Argument::Coordinate:
      out[k] = ADReal(x[a.component]);
      break;
    case Argument::Time:
      out[k] = ADReal(ctx.time);
      break;
    case Argument::Eigenvalue:
      out[k] = ADReal(ctx.eigenvalue);
      break;
    case Argument::Function:
      out[k] = ADReal(a.function->value(x, ctx.time));
      break;
    case Argument::ElementField:
      out[k] = ADReal(ctx.element >= 0 ? (*a.field)[ctx.element] : 0.0);
      break;
    case Argument::Property:
      out[k] = ctx.property(a.index);
      break;
    case Argument::Constant:
      out[k] = ADReal(a.constant);
      break;
    }
  }
}

ADReal
FieldExpressions::value(std::size_t i, const QpContext & ctx) const
{
  std::array<ADReal, kMaxArguments> args;
  arguments(ctx, args.data());
  return _expressions[i]->value(args.data());
}

ADReal
FieldExpressions::valueAt(std::size_t i, const QpContext & ctx, const Point & x) const
{
  std::array<ADReal, kMaxArguments> args;
  arguments(ctx, x, args.data());
  return _expressions[i]->value(args.data());
}

void
FieldCoefficient::setup(const Problem & problem,
                        const InputParameters & params,
                        const std::string & param,
                        const std::string & owner)
{
  _function = nullptr;
  const auto & raw = params.getRaw(param);
  if (std::holds_alternative<double>(raw))
    _function = std::make_shared<ConstantFunction>(std::get<double>(raw));
  else if (std::holds_alternative<long>(raw))
    _function = std::make_shared<ConstantFunction>(double(std::get<long>(raw)));
  else if (std::holds_alternative<std::shared_ptr<Function>>(raw))
    _function = std::get<std::shared_ptr<Function>>(raw);
  else if (std::holds_alternative<std::string>(raw))
    setup(problem, std::get<std::string>(raw), owner + "', parameter '" + param);
  else
    throw InputError("Parameter '" + param + "' of '" + owner + "' has no value.");
}

void
FieldCoefficient::setup(const Problem & problem,
                        const std::string & text,
                        const std::string & owner)
{
  _function = nullptr;
  if (problem.hasFunction(text))
  {
    _function = problem.function(text);
    return;
  }
  try
  {
    std::size_t used = 0;
    const double value = std::stod(text, &used);
    if (used == text.size())
    {
      _function = std::make_shared<ConstantFunction>(value);
      return;
    }
  }
  catch (const std::exception &)
  {
  }
  _expression.compile(problem, {text}, {}, owner);
}

} // namespace dualmesh
