// SPDX-License-Identifier: LGPL-2.1-or-later
#include "dualmesh/core/InputParameters.h"
#include "dualmesh/core/Function.h"

#include <sstream>

namespace dualmesh
{

std::string
parameterKindName(ParameterKind kind)
{
  switch (kind)
  {
  case ParameterKind::Boolean:
    return "boolean";
  case ParameterKind::Integer:
    return "integer";
  case ParameterKind::Real:
    return "real";
  case ParameterKind::String:
    return "string";
  case ParameterKind::RealList:
    return "list of reals";
  case ParameterKind::IntegerList:
    return "list of integers";
  case ParameterKind::StringList:
    return "list of strings";
  case ParameterKind::Function:
    return "real or function";
  }
  return "unknown";
}

void
InputParameters::addRequired(const std::string & name,
                             ParameterKind kind,
                             const std::string & description)
{
  ParameterInfo p;
  p.kind = kind;
  p.description = description;
  p.required = true;
  _params[name] = p;
}

void
InputParameters::addOptional(const std::string & name,
                             ParameterKind kind,
                             ParameterValue default_value,
                             const std::string & description)
{
  ParameterInfo p;
  p.kind = kind;
  p.description = description;
  p.required = false;
  p.value = std::move(default_value);
  _params[name] = p;
}

namespace
{
std::string
valueTypeName(const ParameterValue & v)
{
  static const char * names[] = {"nothing",
                                 "boolean",
                                 "integer",
                                 "real",
                                 "string",
                                 "list of reals",
                                 "list of integers",
                                 "list of strings",
                                 "function"};
  return names[v.index()];
}
} // namespace

void
InputParameters::set(const std::string & name, ParameterValue v)
{
  auto it = _params.find(name);
  if (it == _params.end())
  {
    std::ostringstream os;
    os << "Unknown parameter '" << name << "'. Accepted parameters are:";
    for (const auto & [n, _] : _params)
      if (n.rfind('_', 0) != 0)
        os << " " << n;
    throw InputError(os.str());
  }
  auto & p = it->second;
  const auto bad = [&]()
  {
    throw InputError("Parameter '" + name + "' expects a " + parameterKindName(p.kind) +
                     " but received a " + valueTypeName(v) + ".");
  };
  switch (p.kind)
  {
  case ParameterKind::Boolean:
    if (std::holds_alternative<long>(v))
      v = bool(std::get<long>(v));
    if (!std::holds_alternative<bool>(v))
      bad();
    break;
  case ParameterKind::Integer:
    if (std::holds_alternative<bool>(v))
      v = long(std::get<bool>(v));
    if (std::holds_alternative<double>(v))
    {
      double d = std::get<double>(v);
      if (d != static_cast<double>(static_cast<long>(d)))
        bad();
      v = static_cast<long>(d);
    }
    if (!std::holds_alternative<long>(v))
      bad();
    break;
  case ParameterKind::Real:
    if (std::holds_alternative<long>(v))
      v = static_cast<double>(std::get<long>(v));
    if (!std::holds_alternative<double>(v))
      bad();
    break;
  case ParameterKind::String:
    if (!std::holds_alternative<std::string>(v))
      bad();
    break;
  case ParameterKind::RealList:
    if (std::holds_alternative<std::vector<std::string>>(v) &&
        std::get<std::vector<std::string>>(v).empty())
      v = std::vector<double>{};
    if (std::holds_alternative<double>(v))
      v = std::vector<double>{std::get<double>(v)};
    else if (std::holds_alternative<long>(v))
      v = std::vector<double>{double(std::get<long>(v))};
    else if (std::holds_alternative<std::vector<long>>(v))
    {
      const auto & l = std::get<std::vector<long>>(v);
      v = std::vector<double>(l.begin(), l.end());
    }
    if (!std::holds_alternative<std::vector<double>>(v))
      bad();
    break;
  case ParameterKind::IntegerList:
    if (std::holds_alternative<std::vector<std::string>>(v) &&
        std::get<std::vector<std::string>>(v).empty())
      v = std::vector<long>{};
    if (std::holds_alternative<long>(v))
      v = std::vector<long>{std::get<long>(v)};
    if (!std::holds_alternative<std::vector<long>>(v))
      bad();
    break;
  case ParameterKind::StringList:
    if ((std::holds_alternative<std::vector<double>>(v) &&
         std::get<std::vector<double>>(v).empty()) ||
        (std::holds_alternative<std::vector<long>>(v) && std::get<std::vector<long>>(v).empty()))
      v = std::vector<std::string>{};
    if (std::holds_alternative<std::string>(v))
      v = std::vector<std::string>{std::get<std::string>(v)};
    if (!std::holds_alternative<std::vector<std::string>>(v))
      bad();
    break;
  case ParameterKind::Function:
    if (std::holds_alternative<long>(v))
      v = static_cast<double>(std::get<long>(v));
    if (!(std::holds_alternative<double>(v) || std::holds_alternative<std::string>(v) ||
          std::holds_alternative<std::shared_ptr<Function>>(v)))
      bad();
    break;
  }
  p.value = std::move(v);
  p.set_by_user = true;
}

void
InputParameters::setPrivate(const std::string & name, ParameterValue value)
{
  auto & p = _params[name];
  p.value = std::move(value);
  p.set_by_user = true;
}

void
InputParameters::validate(const std::string & object_description) const
{
  for (const auto & [name, p] : _params)
    if (p.required && !p.set_by_user)
      throw InputError("Missing required parameter '" + name + "' (" + p.description + ") for " +
                       object_description + ".");
}

bool
InputParameters::isSet(const std::string & name) const
{
  auto it = _params.find(name);
  return it != _params.end() &&
         (it->second.set_by_user || !std::holds_alternative<std::monostate>(it->second.value));
}

const ParameterInfo &
InputParameters::info(const std::string & name) const
{
  auto it = _params.find(name);
  if (it == _params.end())
    throw InputError("Parameter '" + name + "' was not declared.");
  if (std::holds_alternative<std::monostate>(it->second.value))
    throw InputError("Parameter '" + name + "' has no value.");
  return it->second;
}

bool
InputParameters::getBool(const std::string & name) const
{
  return std::get<bool>(info(name).value);
}
long
InputParameters::getInt(const std::string & name) const
{
  return std::get<long>(info(name).value);
}
double
InputParameters::getReal(const std::string & name) const
{
  const auto & v = info(name).value;
  if (std::holds_alternative<long>(v))
    return double(std::get<long>(v));
  if (!std::holds_alternative<double>(v))
    throw InputError("Parameter '" + name + "' must be a real number here.");
  return std::get<double>(v);
}
std::string
InputParameters::getString(const std::string & name) const
{
  return std::get<std::string>(info(name).value);
}
std::vector<double>
InputParameters::getRealList(const std::string & name) const
{
  return std::get<std::vector<double>>(info(name).value);
}
std::vector<long>
InputParameters::getIntList(const std::string & name) const
{
  return std::get<std::vector<long>>(info(name).value);
}
std::vector<std::string>
InputParameters::getStringList(const std::string & name) const
{
  return std::get<std::vector<std::string>>(info(name).value);
}
const ParameterValue &
InputParameters::getRaw(const std::string & name) const
{
  auto it = _params.find(name);
  if (it == _params.end())
    throw InputError("Parameter '" + name + "' was not declared.");
  return it->second.value;
}

std::string
InputParameters::describe() const
{
  std::ostringstream os;
  if (!_class_description.empty())
    os << _class_description << "\n\n";
  for (const auto & [name, p] : _params)
  {
    if (name.rfind('_', 0) == 0)
      continue;
    os << "  " << name << " (" << parameterKindName(p.kind) << (p.required ? ", required" : "")
       << "): " << p.description << "\n";
  }
  return os.str();
}

} // namespace dualmesh
