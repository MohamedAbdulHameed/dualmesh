// SPDX-License-Identifier: LGPL-2.1-or-later
#include "dualmesh/base/FieldExpression.h"

#include "dualmesh/base/Problem.h"
#include "dualmesh/base/Property.h"

#include <algorithm>
#include <array>
#include <set>

namespace dualmesh
{

void
FieldExpressions::compile(const Problem & problem,
                          const std::vector<std::string> & texts,
                          const Names & extra,
                          const std::string & owner)
{
  const int dim = problem.mesh().dimension();

  // Every name in scope, with where its value comes from.
  std::vector<std::string> names;
  std::vector<Argument> candidates;
  const auto add = [&](const std::string & n, Argument a)
  {
    if (std::find(names.begin(), names.end(), n) != names.end())
      throw InputError(
          "'" + owner + "': the name '" + n +
          "' is used twice (a field, a function, a property or a constant). Rename one of them.");
    names.push_back(n);
    candidates.push_back(std::move(a));
  };
  const char * components[3] = {"grad_x", "grad_y", "grad_z"};
  for (int v = 0; v < problem.numVariables(); ++v)
  {
    const std::string & field = problem.variable(v).name;
    add(field, {Argument::Value, v, 0, 0.0, nullptr});
    for (int c = 0; c < dim; ++c)
      add(std::string(components[c]) + "(" + field + ")", {Argument::Gradient, v, c, 0.0, nullptr});
  }
  const char * coordinates[3] = {"x", "y", "z"};
  for (int c = 0; c < 3; ++c)
    add(coordinates[c], {Argument::Coordinate, 0, c, 0.0, nullptr});
  add("t", {Argument::Time, 0, 0, 0.0, nullptr});
  add("eigenvalue", {Argument::Eigenvalue, 0, 0, 0.0, nullptr});
  for (const auto & f : extra.functions)
    add(f, {Argument::Function, 0, 0, 0.0, problem.function(f)});
  const auto & registry = problem.propertyRegistry();
  for (const auto & n : extra.properties)
  {
    if (!registry.has(n))
      throw InputError("'" + owner + "': no property object declares the property '" + n + "'.");
    if (registry.components(n) != 1)
      throw InputError("'" + owner + "': the property '" + n +
                       "' has several components; only scalar properties can be used.");
    add(n, {Argument::Property, registry.id(n), 0, 0.0, nullptr});
  }
  for (const auto & [n, value] : extra.constants)
    add(n, {Argument::Constant, 0, 0, value, nullptr});

  // Compile against every name to find the names in use, then again against those only.
  std::set<int> used;
  for (const auto & text : texts)
  {
    const ParsedExpression all(text, names);
    for (std::size_t k = 0; k < names.size(); ++k)
      if (all.uses(static_cast<int>(k)))
        used.insert(static_cast<int>(k));
  }
  if (static_cast<int>(used.size()) > kMaxArguments)
    throw InputError("'" + owner + "': the expressions use " + std::to_string(used.size()) +
                     " names, and at most " + std::to_string(kMaxArguments) + " are allowed.");
  std::vector<std::string> used_names;
  _arguments.clear();
  _uses_fields = false;
  for (int k : used)
  {
    used_names.push_back(names[k]);
    _arguments.push_back(candidates[k]);
    if (candidates[k].kind == Argument::Value || candidates[k].kind == Argument::Gradient)
      _uses_fields = true;
  }
  _expressions.clear();
  for (const auto & text : texts)
    _expressions.push_back(std::make_shared<ParsedExpression>(text, used_names));
}

void
FieldExpressions::arguments(const QpContext & ctx, ADReal * out) const
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
      out[k] = ADReal(ctx.x[a.component]);
      break;
    case Argument::Time:
      out[k] = ADReal(ctx.time);
      break;
    case Argument::Eigenvalue:
      out[k] = ADReal(ctx.eigenvalue);
      break;
    case Argument::Function:
      out[k] = ADReal(a.function->value(ctx.x, ctx.time));
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
  {
    const auto & text = std::get<std::string>(raw);
    if (problem.hasFunction(text))
      _function = problem.function(text);
    else
      _expression.compile(problem, {text}, {}, owner + "', parameter '" + param);
  }
  else
    throw InputError("Parameter '" + param + "' of '" + owner + "' has no value.");
}

} // namespace dualmesh
