// SPDX-License-Identifier: LGPL-2.1-or-later
//
// The conversions of Python values that the bindings of several subsystems share.
#include "bindings.h"

#include <algorithm>
#include <cctype>

namespace dualmesh::bindings
{

namespace
{

/// A Function backed by a Python callable f(x, y, z, t).
class PythonFunction : public Function
{
public:
  // Evaluating this object calls back into the interpreter, which needs
  // the global interpreter lock, so the problem must assemble serially.
  bool threadSafe() const override { return false; }

  explicit PythonFunction(py::function f) : _f(std::move(f)) {}
  double value(const Point & x, double t) const override
  {
    py::gil_scoped_acquire gil;
    return py::cast<double>(_f(x[0], x[1], x[2], t));
  }
  std::string description() const override { return "python function"; }

private:
  py::function _f;
};

} // namespace

FunctionPtr
toFunction(const py::object & o)
{
  // A compiled function is checked first: pybind11's isinstance<function> is
  // Python's callable() test, which a ParsedFunction (it has __call__) also
  // passes, and wrapping it as a Python callback would evaluate it through the
  // interpreter and force the assembly onto one thread.
  if (py::isinstance<Function>(o))
    return py::cast<std::shared_ptr<Function>>(o);
  if (py::isinstance<py::function>(o))
    return std::make_shared<PythonFunction>(py::cast<py::function>(o));
  return std::make_shared<ConstantFunction>(py::cast<double>(o));
}

/// Convert a Python value into a ParameterValue.
ParameterValue
toParameter(const py::handle & v)
{
  if (py::isinstance<py::bool_>(v))
    return py::cast<bool>(v);
  if (py::isinstance<py::int_>(v))
    return static_cast<long>(py::cast<long long>(v));
  if (py::isinstance<py::float_>(v))
    return py::cast<double>(v);
  if (py::isinstance<py::str>(v))
    return py::cast<std::string>(v);
  if (py::isinstance<Function>(v)) // before the callable test; see toFunction
    return py::cast<std::shared_ptr<Function>>(v);
  if (py::isinstance<py::function>(v))
    return std::static_pointer_cast<Function>(
        std::make_shared<PythonFunction>(py::cast<py::function>(v)));
  if (py::isinstance<py::dict>(v))
  {
    // Names with a number each: a RealMap.
    RealMap map;
    for (const auto & item : py::cast<py::dict>(v))
    {
      if (py::isinstance<py::bool_>(item.second) ||
          !(py::isinstance<py::int_>(item.second) || py::isinstance<py::float_>(item.second)))
        throw InputError("A mapping of names takes a number for each name, not '" +
                         py::cast<std::string>(py::str(item.second)) + "'.");
      map[py::cast<std::string>(item.first)] = py::cast<double>(item.second);
    }
    return map;
  }
  if (py::isinstance<py::sequence>(v) && !py::isinstance<py::str>(v))
  {
    auto seq = py::cast<py::sequence>(v);
    bool all_int = true, all_num = true, all_str = true;
    for (const auto & item : seq)
    {
      all_int = all_int && py::isinstance<py::int_>(item) && !py::isinstance<py::bool_>(item);
      all_num = all_num && (py::isinstance<py::int_>(item) || py::isinstance<py::float_>(item));
      all_str = all_str && py::isinstance<py::str>(item);
    }
    if (py::len(seq) > 0 && all_str)
      return py::cast<std::vector<std::string>>(seq);
    // Numbers and texts together (a vector of expressions, some of them constants): all as texts.
    bool all_text_or_number = py::len(seq) > 0;
    for (const auto & item : seq)
      all_text_or_number = all_text_or_number &&
                           (py::isinstance<py::str>(item) || py::isinstance<py::int_>(item) ||
                            py::isinstance<py::float_>(item)) &&
                           !py::isinstance<py::bool_>(item);
    if (all_text_or_number && !all_num)
    {
      std::vector<std::string> texts;
      for (const auto & item : seq)
        texts.push_back(py::cast<std::string>(py::str(item)));
      return texts;
    }
    if (all_int && py::len(seq) > 0)
      return py::cast<std::vector<long>>(seq);
    if (all_num)
      return py::cast<std::vector<double>>(seq);
  }
  throw InputError("Unsupported parameter value of type " +
                   py::cast<std::string>(py::str(py::type::of(v))));
}

InputParameters
paramsFromKwargs(InputParameters p, const py::kwargs & kwargs)
{
  for (const auto & item : kwargs)
    p.set(py::cast<std::string>(item.first), toParameter(item.second));
  return p;
}

Method
methodFromString(const std::string & given)
{
  // The abbreviations keep their capitals, and the lower case is accepted too.
  std::string s = given;
  std::transform(s.begin(),
                 s.end(),
                 s.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  if (s == "dmcdm")
    return Method::DualMesh;
  if (s == "fem")
    return Method::FiniteElement;
  throw InputError("Unknown method '" + given +
                   "' (use 'FEM', the finite element method, for solids, or 'DMCDM', the dual "
                   "mesh control domain method, for fluids).");
}

CoordinateSystem
coordFromString(const std::string & s)
{
  if (s == "cartesian")
    return CoordinateSystem::Cartesian;
  if (s == "axisymmetric")
    return CoordinateSystem::Axisymmetric;
  if (s == "spherical")
    return CoordinateSystem::SphericalRadial;
  throw InputError("Unknown coordinate system '" + s +
                   "' (use 'cartesian', 'axisymmetric', or 'spherical').");
}

} // namespace dualmesh::bindings
