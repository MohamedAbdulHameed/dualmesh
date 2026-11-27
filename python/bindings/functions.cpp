// SPDX-License-Identifier: LGPL-2.1-or-later
//
// Bindings of the automatic differentiation numbers, the compiled expressions and the functions.
#include "bindings.h"
#include "dualmesh/core/ADReal.h"
#include "dualmesh/core/Function.h"
#include "dualmesh/core/ParsedFunction.h"

#include <pybind11/operators.h>

#include <string>
#include <vector>

namespace dualmesh::bindings
{

void
bindFunctions(py::module_ & m)
{
  // ---- automatic differentiation -----------------------------------------
  py::class_<ADReal>(m, "ADReal", "Forward-mode automatic differentiation number.")
      .def(py::init<double>())
      .def_property_readonly("value", [](const ADReal & a) { return a.value(); })
      .def("derivative", &ADReal::derivative)
      .def("__float__", [](const ADReal & a) { return a.value(); })
      .def("__repr__", [](const ADReal & a) { return "ADReal(" + std::to_string(a.value()) + ")"; })
      .def(py::self + py::self)
      .def(py::self - py::self)
      .def(py::self * py::self)
      .def(py::self / py::self)
      .def(py::self + double())
      .def(py::self - double())
      .def(py::self * double())
      .def(py::self / double())
      .def(double() + py::self)
      .def(double() - py::self)
      .def(double() * py::self)
      .def(double() / py::self)
      .def(-py::self)
      .def("__pow__", [](const ADReal & a, double p) { return pow(a, p); })
      .def("__pow__", [](const ADReal & a, const ADReal & b) { return pow(a, b); })
      .def("__rpow__", [](const ADReal & b, double a) { return pow(ADReal(a), b); });

  m.def("sqrt", [](const ADReal & a) { return sqrt(a); });
  m.def("exp", [](const ADReal & a) { return exp(a); });
  m.def("log", [](const ADReal & a) { return log(a); });
  m.def("sin", [](const ADReal & a) { return sin(a); });
  m.def("cos", [](const ADReal & a) { return cos(a); });
  m.def("tanh", [](const ADReal & a) { return tanh(a); });
  m.def("abs", [](const ADReal & a) { return abs(a); });

  // ---- expressions -------------------------------------------------------
  py::class_<ParsedExpression, std::shared_ptr<ParsedExpression>>(
      m,
      "Expression",
      "An expression of named variables, compiled once to C++ and evaluated with plain "
      "numbers or arrays. It is the expression language of the parsed objects "
      "(parsed_property, parsed_eigenstrain and the parsed creep of small_strain_stress), "
      "so a correlation can be checked in Python before it is used in a problem.")
      .def(py::init<std::string, std::vector<std::string>>(),
           py::arg("expression"),
           py::arg("variables"))
      .def_property_readonly("expression", &ParsedExpression::text)
      .def_property_readonly("variables", &ParsedExpression::variables)
      .def_property_readonly(
          "used_variables",
          [](const ParsedExpression & e)
          {
            std::vector<std::string> used;
            for (std::size_t k = 0; k < e.variables().size(); ++k)
              if (e.uses(static_cast<int>(k)))
                used.push_back(e.variables()[k]);
            return used;
          },
          "The variables the expression actually uses, in the order they were named.")
      .def(
          "__call__",
          [](const ParsedExpression & e, py::kwargs kwargs) -> py::object
          {
            const auto & names = e.variables();
            std::vector<py::array_t<double>> arrays;
            std::vector<std::string> missing;
            for (const auto & n : names)
            {
              if (!kwargs.contains(n.c_str()))
              {
                missing.push_back(n);
                continue;
              }
              arrays.push_back(py::array_t<double>::ensure(kwargs[n.c_str()]));
            }
            if (!missing.empty())
            {
              std::string text;
              for (const auto & n : missing)
                text += (text.empty() ? "" : ", ") + n;
              throw InputError("Expression \"" + e.text() + "\": no value given for " + text + ".");
            }
            for (auto item : kwargs)
            {
              const auto key = py::cast<std::string>(item.first);
              if (e.index(key) < 0)
                throw InputError("Expression \"" + e.text() + "\" has no variable '" + key + "'.");
            }
            // An expression of no variables is a number.
            if (names.empty())
              return py::float_(e.value(static_cast<const double *>(nullptr)));
            // Broadcast every argument to one shape.
            py::module_ np = py::module_::import("numpy");
            py::list items;
            for (auto & a : arrays)
              items.append(a);
            py::tuple broadcast = np.attr("broadcast_arrays")(*items);
            const std::size_t n = names.size();
            py::array_t<double> first = py::array_t<double>::ensure(broadcast[0]);
            const py::ssize_t count = first.size();
            std::vector<py::array_t<double>> flat;
            for (std::size_t k = 0; k < n; ++k)
              flat.push_back(py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(
                  np.attr("ascontiguousarray")(broadcast[k])));
            py::array_t<double> out(count);
            auto o = out.mutable_unchecked<1>();
            std::vector<double> values(n);
            for (py::ssize_t i = 0; i < count; ++i)
            {
              for (std::size_t k = 0; k < n; ++k)
                values[k] = flat[k].data()[i];
              o(i) = e.value(values.data());
            }
            if (first.ndim() == 0)
              return py::float_(o(0));
            return out.attr("reshape")(first.attr("shape"));
          },
          "Evaluate with every variable given by name, as a number or an array (the arrays "
          "broadcast together).");

  m.def("expression_functions",
        &ParsedExpression::functions,
        "The functions of the expression language as (name, number of arguments) pairs, from "
        "its one table in C++.");

  // ---- functions ---------------------------------------------------------
  py::class_<Function, std::shared_ptr<Function>>(m, "Function").def("value", &Function::value);
  py::class_<ParsedFunction, Function, std::shared_ptr<ParsedFunction>>(m, "ParsedFunction")
      .def(py::init<std::string>(), py::arg("expression"))
      .def_property_readonly("expression", &ParsedFunction::expression)
      .def(
          "__call__",
          [](const ParsedFunction & f, double x, double y, double z, double t)
          { return f.value(Point{x, y, z}, t); },
          py::arg("x"),
          py::arg("y") = 0.0,
          py::arg("z") = 0.0,
          py::arg("t") = 0.0)
      .def("__repr__",
           [](const ParsedFunction & f)
           { return "<dualmesh.ParsedFunction '" + f.expression() + "'>"; });
  py::class_<ConstantFunction, Function, std::shared_ptr<ConstantFunction>>(m, "ConstantFunction")
      .def(py::init<double>());
  py::class_<PiecewiseLinearFunction, Function, std::shared_ptr<PiecewiseLinearFunction>>(
      m, "PiecewiseLinearFunction")
      .def(py::init<std::vector<double>, std::vector<double>, int>(),
           py::arg("abscissa"),
           py::arg("ordinate"),
           py::arg("axis") = 3);
  py::class_<SettableFunction, Function, std::shared_ptr<SettableFunction>>(
      m, "SettableFunction", "A constant whose value can be changed between solves with set().")
      .def(py::init<double>(), py::arg("value") = 0.0)
      .def("set", &SettableFunction::set, py::arg("value"))
      .def("get", &SettableFunction::get);
}

} // namespace dualmesh::bindings
