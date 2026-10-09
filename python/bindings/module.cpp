// SPDX-License-Identifier: LGPL-2.1-or-later
//
// pybind11 bindings: the whole framework is usable from Python, including
// user-defined kernels, boundary conditions, and property objects written in Python.
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "dualmesh/base/Factory.h"
#include "dualmesh/base/Problem.h"
#include "dualmesh/core/ParsedFunction.h"
#include "dualmesh/fe/Assembly.h"
#include "dualmesh/linalg/PetscSolver.h"
#include "dualmesh/materials/Gases.h"
#include "dualmesh/mesh/Adapt.h"
#include "dualmesh/modules/Framework.h"
#include "dualmesh/parallel/DistributedProblem.h"

#include <array>
#include <cstdio>
#include <iostream>
#include <map>
#include <optional>
#include <sstream>
#include <type_traits>

namespace py = pybind11;
using namespace dualmesh;

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

// ---- trampolines ---------------------------------------------------------
class PyKernel : public Kernel
{
public:
  // Evaluating this object calls back into the interpreter, which needs
  // the global interpreter lock, so the problem must assemble serially.
  bool threadSafe() const override { return false; }

  using Kernel::Kernel;
  void initialSetup(Problem & problem) override
  {
    PYBIND11_OVERRIDE_NAME(void, Kernel, "initial_setup", initialSetup, std::ref(problem));
  }
  bool hasFlux() const override { PYBIND11_OVERRIDE_NAME(bool, Kernel, "has_flux", hasFlux); }
  bool hasSource() const override { PYBIND11_OVERRIDE_NAME(bool, Kernel, "has_source", hasSource); }
  bool isTimeKernel() const override
  {
    PYBIND11_OVERRIDE_NAME(bool, Kernel, "is_time_kernel", isTimeKernel);
  }
  std::vector<std::string> requiredProperties() const override
  {
    PYBIND11_OVERRIDE_NAME(
        std::vector<std::string>, Kernel, "required_properties", requiredProperties);
  }
  void computeFlux(const QpContext & ctx, ADVector3 & F) const override
  {
    py::gil_scoped_acquire gil;
    py::function f = py::get_override(static_cast<const Kernel *>(this), "compute_flux");
    if (!f)
      return;
    py::object r = f(py::cast(&ctx, py::return_value_policy::reference));
    auto seq = py::cast<py::sequence>(r);
    for (std::size_t i = 0; i < py::len(seq) && i < 3; ++i)
      F[i] = py::isinstance<ADReal>(seq[i]) ? py::cast<ADReal>(seq[i])
                                            : ADReal(py::cast<double>(seq[i]));
  }
  ADReal computeSource(const QpContext & ctx) const override
  {
    py::gil_scoped_acquire gil;
    py::function f = py::get_override(static_cast<const Kernel *>(this), "compute_source");
    if (!f)
      return ADReal(0.0);
    py::object r = f(py::cast(&ctx, py::return_value_policy::reference));
    return py::isinstance<ADReal>(r) ? py::cast<ADReal>(r) : ADReal(py::cast<double>(r));
  }
};

class PyIntegratedBC : public IntegratedBC
{
public:
  // Evaluating this object calls back into the interpreter, which needs
  // the global interpreter lock, so the problem must assemble serially.
  bool threadSafe() const override { return false; }

  using IntegratedBC::IntegratedBC;
  void initialSetup(Problem & problem) override
  {
    PYBIND11_OVERRIDE_NAME(void, IntegratedBC, "initial_setup", initialSetup, std::ref(problem));
  }
  ADReal computeBoundaryFlux(const QpContext & ctx) const override
  {
    py::gil_scoped_acquire gil;
    py::function f =
        py::get_override(static_cast<const IntegratedBC *>(this), "compute_boundary_flux");
    if (!f)
      return ADReal(0.0);
    py::object r = f(py::cast(&ctx, py::return_value_policy::reference));
    return py::isinstance<ADReal>(r) ? py::cast<ADReal>(r) : ADReal(py::cast<double>(r));
  }
};

class PyNodalBC : public NodalBC
{
public:
  // Evaluating this object calls back into the interpreter, which needs
  // the global interpreter lock, so the problem must assemble serially.
  bool threadSafe() const override { return false; }

  using NodalBC::NodalBC;
  void initialSetup(Problem & problem) override
  {
    PYBIND11_OVERRIDE_NAME(void, NodalBC, "initial_setup", initialSetup, std::ref(problem));
  }
  double computeValue(const Point & x, double t) const override
  {
    PYBIND11_OVERRIDE_PURE_NAME(double, NodalBC, "compute_value", computeValue, x, t);
  }
};

class PyProperty : public Property
{
public:
  // Evaluating this object calls back into the interpreter, which needs
  // the global interpreter lock, so the problem must assemble serially.
  bool threadSafe() const override { return false; }

  using Property::Property;
  void initialSetup(Problem & problem) override
  {
    PYBIND11_OVERRIDE_NAME(void, Property, "initial_setup", initialSetup, std::ref(problem));
  }
  void declareProperties(PropertyRegistry & r) override
  {
    PYBIND11_OVERRIDE_PURE_NAME(
        void, Property, "declare_properties", declareProperties, std::ref(r));
  }
  void computeProperties(QpContext & ctx) const override
  {
    py::gil_scoped_acquire gil;
    py::function f = py::get_override(static_cast<const Property *>(this), "compute_properties");
    if (f)
      f(py::cast(&ctx, py::return_value_policy::reference));
  }
};

Method
methodFromString(const std::string & s)
{
  if (s == "dmcdm" || s == "dual_mesh" || s == "dual mesh")
    return Method::DualMesh;
  if (s == "fem" || s == "finite_element")
    return Method::FiniteElement;
  if (s == "hfvm" || s == "fvm" || s == "finite_volume" || s == "vertex_finite_volume")
    return Method::FiniteVolumeVertex;
  if (s == "zfvm" || s == "cell_finite_volume" || s == "cell_centered_finite_volume")
    return Method::FiniteVolumeCell;
  throw InputError("Unknown method '" + s +
                   "' (use 'dmcdm', 'fem', 'hfvm' for the vertex-centred finite volume "
                   "method, or 'zfvm' for the cell-centred finite volume method).");
}

BoundaryGradient
boundaryGradientFromString(const std::string & s)
{
  if (s == "first_order")
    return BoundaryGradient::FirstOrder;
  if (s == "second_order")
    return BoundaryGradient::SecondOrder;
  throw InputError("Unknown boundary gradient '" + s + "' (use 'first_order' or 'second_order').");
}

CoordinateSystem
coordFromString(const std::string & s)
{
  if (s == "cartesian" || s == "xyz")
    return CoordinateSystem::Cartesian;
  if (s == "axisymmetric" || s == "rz" || s == "cylindrical")
    return CoordinateSystem::Axisymmetric;
  if (s == "spherical")
    return CoordinateSystem::SphericalRadial;
  throw InputError("Unknown coordinate system '" + s +
                   "' (use 'cartesian', 'axisymmetric', or 'spherical').");
}

} // namespace

namespace
{
/// A parameter value as the text a user would type.
std::string
parameterText(const ParameterValue & value)
{
  std::string text;
  std::visit(
      [&text](const auto & v)
      {
        using T = std::decay_t<decltype(v)>;
        if constexpr (std::is_same_v<T, std::monostate>)
          text = "";
        else if constexpr (std::is_same_v<T, bool>)
          text = v ? "true" : "false";
        else if constexpr (std::is_same_v<T, long>)
          text = std::to_string(v);
        else if constexpr (std::is_same_v<T, double>)
        {
          std::ostringstream os;
          os << v;
          text = os.str();
        }
        else if constexpr (std::is_same_v<T, std::string>)
          text = v;
        else if constexpr (std::is_same_v<T, std::vector<double>> ||
                           std::is_same_v<T, std::vector<long>>)
        {
          std::ostringstream os;
          for (std::size_t i = 0; i < v.size(); ++i)
            os << (i ? ", " : "") << v[i];
          text = os.str();
        }
        else if constexpr (std::is_same_v<T, std::vector<std::string>>)
        {
          for (std::size_t i = 0; i < v.size(); ++i)
            text += (i ? ", " : "") + v[i];
        }
        else
          text = "function";
      },
      value);
  return text;
}
} // namespace

PYBIND11_MODULE(_core, m)
{
  m.doc() = "dualmesh: dual mesh control domain method for computational mechanics (core)";

  py::register_exception<InputError>(m, "InputError", PyExc_ValueError);

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
      .def("__pow__", [](const ADReal & a, const ADReal & b) { return pow(a, b); });
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
            // Broadcast every argument to one shape.
            py::module_ np = py::module_::import("numpy");
            py::list items;
            for (auto & a : arrays)
              items.append(a);
            py::tuple broadcast = np.attr("broadcast_arrays")(*items);
            const std::size_t n = names.size();
            py::array_t<double> first = py::array_t<double>::ensure(broadcast[0]);
            const py::ssize_t count = n == 0 ? 1 : first.size();
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
            if (n == 0 || first.ndim() == 0)
              return py::float_(o(0));
            return out.attr("reshape")(first.attr("shape"));
          },
          "Evaluate with every variable given by name, as a number or an array (the arrays "
          "broadcast together).");

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
  m.def("make_function", &toFunction);

  // ---- material properties ---------------------------------------------
  {
    auto gm = m.def_submodule("gas",
                              "Thermal conductivity of helium, argon, krypton, xenon, hydrogen and "
                              "nitrogen at low density, and of their mixtures.");
    auto mole_fractions = [](const std::map<std::string, double> & composition)
    {
      std::array<double, materials::kNumGases> x{};
      double total = 0;
      for (const auto & [g, v] : composition)
      {
        if (v < 0)
          throw InputError("the mole fraction of " + g + " must not be negative");
        x[static_cast<int>(materials::gasFromName(g))] += v;
        total += v;
      }
      if (total <= 0)
        throw InputError("the gas composition needs a positive mole fraction");
      for (double & xi : x)
        xi /= total;
      return x;
    };
    gm.def(
        "thermal_conductivity",
        [mole_fractions](const std::map<std::string, double> & composition,
                         py::array_t<double> temperature)
        {
          const auto x = mole_fractions(composition);
          return py::vectorize(
              [&x](double T)
              { return materials::gasMixtureConductivity(x.data(), T); })(temperature);
        },
        py::arg("composition"),
        py::arg("temperature"),
        "Thermal conductivity of a gas mixture at low density, W/(m K). composition maps the "
        "gas names (helium, argon, krypton, xenon, hydrogen, nitrogen) to mole fractions, which "
        "are normalized to a sum of one. temperature is in K.");
    gm.def(
        "temperature_jump_distance",
        [mole_fractions](const std::map<std::string, double> & composition,
                         double temperature,
                         double pressure,
                         std::optional<double> accommodation_coefficient)
        {
          const auto x = mole_fractions(composition);
          const double k = materials::gasMixtureConductivity(x.data(), temperature);
          return materials::gasJumpDistance(
              x.data(), temperature, pressure, k, accommodation_coefficient.value_or(0.0));
        },
        py::arg("composition"),
        py::arg("temperature"),
        py::arg("pressure"),
        py::arg("accommodation_coefficient") = py::none(),
        "Sum of the temperature jump distances of a gas mixture at the two walls of a gap, m "
        "(Kennard, as Lanning and Hann, BNWL-1894, 1975, write it). Without an "
        "accommodation_coefficient, the fits of Ullman et al. for helium and xenon are "
        "interpolated by the mixture molar mass.");
  }

  // ---- mesh --------------------------------------------------------------
  py::class_<Mesh, std::shared_ptr<Mesh>>(
      m,
      "Mesh",
      "A mesh of finite elements: nodes, elements of any of the fourteen supported types "
      "(which may be mixed), element blocks, side sets and node sets. Meshes are usually "
      "made by the generators (generate_rectangle_mesh and the others) or read from a file "
      "(read_mesh), but can be built node by node with add_node and add_element.")
      .def(py::init<int>(),
           py::arg("dimension") = 2,
           "An empty mesh of the given spatial dimension (1, 2 or 3).")
      .def_property_readonly("dimension", &Mesh::dimension, "The spatial dimension, 1, 2 or 3.")
      .def_property_readonly("num_nodes", &Mesh::numNodes, "The number of nodes.")
      .def_property_readonly("num_elements", &Mesh::numElements, "The number of elements.")
      .def("add_node",
           &Mesh::addNode,
           py::arg("point"),
           "Add a node at (x, y, z) and return its index.")
      .def(
          "add_element",
          [](Mesh & mesh, const std::string & type, const std::vector<Index> & nodes, int block)
          { return mesh.addElement(elementTypeFromName(type), nodes, block); },
          py::arg("element_type"),
          py::arg("nodes"),
          py::arg("block") = 0,
          "Add an element of the named type (for instance 'Quad4') on the given nodes, "
          "numbered as VTK numbers them, in element block 'block'. Returns its index.")
      .def("node", &Mesh::node, py::arg("index"), "The coordinates (x, y, z) of a node.")
      .def(
          "points",
          [](const Mesh & mesh)
          {
            py::array_t<double> a(
                {static_cast<py::ssize_t>(mesh.numNodes()), static_cast<py::ssize_t>(3)});
            auto r = a.mutable_unchecked<2>();
            for (Index i = 0; i < mesh.numNodes(); ++i)
              for (int d = 0; d < 3; ++d)
                r(i, d) = mesh.node(i)[d];
            return a;
          },
          "The coordinates of every node, as an array of shape (num_nodes, 3).")
      .def(
          "cells",
          [](const Mesh & mesh)
          {
            py::list out;
            for (const auto & el : mesh.elements())
            {
              py::list conn;
              for (int k = 0; k < el.numNodes(); ++k)
                conn.append(el.nodes[k]);
              out.append(py::make_tuple(elementTypeName(el.type), conn, el.block));
            }
            return out;
          },
          "Every element as a tuple (type, node indices, block).")
      .def(
          "element_nodes",
          [](const Mesh & mesh, Index e)
          {
            std::vector<Index> out;
            for (int k = 0; k < mesh.element(e).numNodes(); ++k)
              out.push_back(mesh.element(e).nodes[k]);
            return out;
          },
          py::arg("element"),
          "The node indices of an element.")
      .def(
          "element_block",
          [](const Mesh & mesh, Index e) { return mesh.element(e).block; },
          py::arg("element"),
          "The block id of an element.")
      .def(
          "element_type",
          [](const Mesh & mesh, Index e) { return elementTypeName(mesh.element(e).type); },
          py::arg("element"),
          "The type of an element, for instance 'Hex8'.")
      .def("set_block_name",
           &Mesh::setBlockName,
           py::arg("block"),
           py::arg("name"),
           "Give an element block a name, by which objects can then be restricted to it.")
      .def("block_ids", &Mesh::blockIds, "The ids of the element blocks present.")
      .def("block_names", &Mesh::blockNames, "The names of the named element blocks, by block id.")
      .def("add_sideset_from_faces",
           &Mesh::addSidesetFromFaces,
           py::arg("name"),
           py::arg("faces"),
           "Add a side set from faces given by their node indices (a list of node lists); "
           "each face is matched to the element side with those nodes.")
      .def("add_nodeset",
           &Mesh::addNodeset,
           py::arg("name"),
           py::arg("nodes"),
           "Add a node set from a list of node indices.")
      .def("alias_sideset",
           &Mesh::aliasSideset,
           py::arg("name"),
           py::arg("alias"),
           "Make 'alias' a second name for the existing side set 'name'.")
      .def(
          "add_sideset_by_predicate",
          [](Mesh & mesh, const std::string & name, const py::function & f)
          {
            mesh.addSidesetByPredicate(name,
                                       [f](const Point & p)
                                       {
                                         py::gil_scoped_acquire gil;
                                         return py::cast<bool>(f(p[0], p[1], p[2]));
                                       });
          },
          py::arg("name"),
          py::arg("predicate"),
          "Add a side set of the boundary sides whose centroid satisfies "
          "predicate(x, y, z).")
      .def(
          "add_nodeset_by_predicate",
          [](Mesh & mesh, const std::string & name, const py::function & f)
          {
            mesh.addNodesetByPredicate(name,
                                       [f](const Point & p)
                                       {
                                         py::gil_scoped_acquire gil;
                                         return py::cast<bool>(f(p[0], p[1], p[2]));
                                       });
          },
          py::arg("name"),
          py::arg("predicate"),
          "Add a node set of the nodes that satisfy predicate(x, y, z).")
      .def(
          "transform_nodes",
          [](Mesh & mesh, const py::function & f)
          {
            mesh.transformNodes(
                [f](const Point & p)
                {
                  py::gil_scoped_acquire gil;
                  auto r = py::cast<std::vector<double>>(f(p[0], p[1], p[2]));
                  Point out{0, 0, 0};
                  for (std::size_t i = 0; i < r.size() && i < 3; ++i)
                    out[i] = r[i];
                  return out;
                });
          },
          py::arg("map"),
          "Move every node: map(x, y, z) returns the new coordinates. Promote a mesh to "
          "second order before mapping it onto a curved domain, so that the mid-side nodes "
          "land on the curve.")
      .def("fix_orientation",
           &Mesh::fixOrientation,
           "Renumber the nodes of every element whose Jacobian determinant is negative, so "
           "that all elements are wound consistently. The readers call it.")
      .def("refined",
           &Mesh::refined,
           "A uniformly refined copy: every element is split into 2^dim children, and the "
           "side sets and node sets are carried over. Only linear meshes can be refined; "
           "refine first and promote to second order afterwards.")
      .def("second_order",
           &Mesh::secondOrder,
           py::arg("serendipity") = false,
           "A copy of this mesh with quadratic elements. Edge2 becomes Edge3, Tri3 becomes "
           "Tri6, Quad4 becomes Quad9, Tet4 becomes Tet10 and Hex8 becomes Hex27; with "
           "serendipity=True, Quad4 becomes Quad8 and Hex8 becomes Hex20 instead, which work "
           "with the finite element and cell-centred finite volume methods only. The corner "
           "nodes keep their numbers and positions, so the geometry of the domain does not "
           "change; the added nodes sit at the midpoints of the edges, faces and cells they "
           "belong to. Side sets carry over unchanged and node sets gain the added nodes that "
           "lie between members.")
      .def_static(
          "vtk_node_order",
          [](const std::string & element_type)
          { return vtkNodeOrder(elementTypeFromName(element_type)); },
          py::arg("element_type"),
          "The permutation from VTK local node numbering to dualmesh local node numbering. "
          "dualmesh numbers every element as VTK does, so it is the identity.")
      .def("add_bounding_box_sidesets",
           &Mesh::addBoundingBoxSidesets,
           py::arg("tolerance") = 1e-10,
           "Add side sets named left, right, bottom, top (and front, back in three "
           "dimensions) from the boundary sides that lie on the faces of the bounding box.")
      .def("boundary_nodes",
           &Mesh::boundaryNodes,
           py::arg("name"),
           "The nodes of a side set or a node set.")
      .def(
          "sideset",
          [](const Mesh & mesh, const std::string & name)
          {
            std::vector<std::pair<Index, int>> out;
            for (const Side & side : mesh.sideset(name))
              out.emplace_back(side.first, side.second);
            return out;
          },
          py::arg("name"),
          "The sides of a side set, as (element, local side) pairs.")
      .def(
          "exterior_sides",
          [](const Mesh & mesh)
          {
            std::vector<std::pair<Index, int>> out;
            for (const Side & side : mesh.exteriorSides())
              out.emplace_back(side.first, side.second);
            return out;
          },
          "All the sides on the exterior boundary of the domain, as (element, local side) pairs.")
      .def(
          "side_nodes",
          [](const Mesh & mesh, const std::vector<std::pair<Index, int>> & sides)
          {
            std::vector<std::vector<Index>> out;
            out.reserve(sides.size());
            for (const auto & side : sides)
              out.push_back(mesh.sideNodes({side.first, side.second}));
            return out;
          },
          py::arg("sides"),
          "The node indices of each side in a list of (element, local side) pairs. The corner "
          "nodes come first, in the order of the side's own element type.")
      .def(
          "sideset_measure",
          [](const Mesh & mesh, const std::vector<std::string> & names)
          { return sidesetMeasure(mesh, names); },
          py::arg("names"),
          "The total length (in two dimensions) or area (in three) of the named side sets, "
          "integrated with the isoparametric map of the elements. A side in several of the sets "
          "is counted once.")
      .def(
          "sideset_names",
          [](const Mesh & mesh)
          {
            std::vector<std::string> out;
            for (const auto & [n, s] : mesh.sidesets())
              out.push_back(n);
            return out;
          },
          "The names of the side sets.")
      .def(
          "nodeset_names",
          [](const Mesh & mesh)
          {
            std::vector<std::string> out;
            for (const auto & [n, s] : mesh.nodesets())
              out.push_back(n);
            return out;
          },
          "The names of the node sets.")
      .def("bounding_box",
           &Mesh::boundingBox,
           "The smallest and the largest coordinates, as two points.")
      .def(
          "element_measures",
          [](const Mesh & mesh)
          {
            std::vector<double> out(mesh.numElements());
            for (Index e = 0; e < mesh.numElements(); ++e)
              out[e] = elementMeasure(mesh, e);
            return out;
          },
          "The length, area or volume of every element.")
      .def("element_centroid",
           &Mesh::elementCentroid,
           py::arg("element"),
           "The mean of the coordinates of an element's nodes.")
      .def("summary", &Mesh::summary, "A one-line description of the mesh.")
      .def("__repr__", [](const Mesh & mesh) { return "<dualmesh.Mesh " + mesh.summary() + ">"; });

  m.def("generate_line_mesh", &generateLineMesh, py::arg("x"));
  m.def("generate_rectangle_mesh",
        &generateRectangleMesh,
        py::arg("x"),
        py::arg("y"),
        py::arg("element_type") = "Quad4",
        py::arg("diagonal") = "right");
  m.def(
      "refine_marked",
      [](const Mesh & mesh, const std::vector<int> & marked)
      {
        std::vector<Index> parents;
        // pybind11 maps std::vector<char> to a sequence of one-character
        // strings, so the flags cross the boundary as integers.
        std::vector<char> flags(marked.begin(), marked.end());
        Mesh out = refineMarked(mesh, flags, &parents);
        return py::make_tuple(out, parents);
      },
      py::arg("mesh"),
      py::arg("marked"),
      "Refine the marked triangles by longest-edge bisection, keeping the mesh conforming. "
      "Returns the refined mesh and, for every element of it, the index of the element it "
      "came from.");

  m.def("generate_box_mesh",
        &generateBoxMesh,
        py::arg("x"),
        py::arg("y"),
        py::arg("z"),
        py::arg("element_type") = "Hex8");

  // ---- parameters and objects ---------------------------------------------
  py::class_<InputParameters>(m, "InputParameters")
      .def(py::init<>())
      .def("set",
           [](InputParameters & p, const std::string & n, const py::object & v)
           { p.set(n, toParameter(v)); })
      .def("describe", &InputParameters::describe);

  py::class_<QpContext>(m, "QpContext", "State at one integration point.")
      .def_readonly("dim", &QpContext::dim)
      .def_readonly("x", &QpContext::x)
      .def_readonly("time", &QpContext::time)
      .def_readonly("dt", &QpContext::dt)
      .def_readonly("element", &QpContext::element)
      .def_readonly("block", &QpContext::block)
      .def_readonly("normal", &QpContext::normal)
      .def_readonly("on_boundary", &QpContext::on_boundary)
      .def_readonly("load_factor", &QpContext::load_factor)
      .def("value", &QpContext::value, py::return_value_policy::copy)
      .def("gradient",
           [](const QpContext & c, int v)
           {
             const auto & g = c.gradient(v);
             return std::vector<ADReal>{g[0], g[1], g[2]};
           })
      .def("coefficient_value", &QpContext::coefficientValue, py::return_value_policy::copy)
      .def("coefficient_gradient",
           [](const QpContext & c, int v)
           {
             const auto & g = c.coefficientGradient(v);
             return std::vector<ADReal>{g[0], g[1], g[2]};
           })
      .def("old_value", &QpContext::oldValue)
      .def("old_gradient", [](const QpContext & c, int v) { return c.oldGradient(v); })
      .def(
          "property",
          [](const QpContext & c, int id, int component) { return c.property(id, component); },
          py::arg("property_id"),
          py::arg("component") = 0)
      .def(
          "set_property",
          [](QpContext & c, int id, const py::object & v, int component)
          {
            c.property(id, component) =
                py::isinstance<ADReal>(v) ? py::cast<ADReal>(v) : ADReal(py::cast<double>(v));
          },
          py::arg("property_id"),
          py::arg("value"),
          py::arg("component") = 0)
      .def_property_readonly(
          "is_picard", [](const QpContext & c) { return c.mode == LinearizationMode::Picard; });

  py::class_<PropertyRegistry>(m, "PropertyRegistry")
      .def("declare", &PropertyRegistry::declare, py::arg("name"), py::arg("components") = 1)
      .def("id", &PropertyRegistry::id);

  py::class_<Object, std::shared_ptr<Object>>(m, "Object")
      .def_property_readonly("name", &Object::name)
      .def_property_readonly("type", &Object::type)
      .def(
          "parameter_rows",
          [](const Object & object)
          {
            py::list out;
            for (const auto & [name, info] : object.parameters().all())
            {
              if (!name.empty() && name[0] == '_')
                continue;
              py::dict row;
              row["name"] = name;
              row["value"] = parameterText(info.value);
              row["is_default"] = !info.set_by_user;
              row["description"] = info.description;
              out.append(row);
            }
            return out;
          },
          "The parameters of this object with their values, and whether each value is the "
          "default (the user did not give it).")
      .def("initial_setup", &Object::initialSetup);

  py::class_<ResidualObject, Object, std::shared_ptr<ResidualObject>>(m, "ResidualObject")
      .def_property_readonly("variable", &ResidualObject::variable)
      .def_property_readonly("variable_name", &ResidualObject::variableName);

  py::class_<Kernel, ResidualObject, std::shared_ptr<Kernel>, PyKernel>(m, "Kernel")
      .def(py::init<const InputParameters &>())
      .def_static("valid_params", &Kernel::validParams)
      .def("has_flux", &Kernel::hasFlux)
      .def("has_source", &Kernel::hasSource)
      .def("is_time_kernel", &Kernel::isTimeKernel);

  py::class_<IntegratedBC, ResidualObject, std::shared_ptr<IntegratedBC>, PyIntegratedBC>(
      m, "IntegratedBC")
      .def(py::init<const InputParameters &>())
      .def_static("valid_params", &IntegratedBC::validParams);

  py::class_<NodalBC, ResidualObject, std::shared_ptr<NodalBC>, PyNodalBC>(m, "NodalBC")
      .def(py::init<const InputParameters &>())
      .def_static("valid_params", &NodalBC::validParams);

  py::class_<NodalLoad, ResidualObject, std::shared_ptr<NodalLoad>>(m, "NodalLoad")
      .def_static("valid_params", &NodalLoad::validParams);

  py::class_<Property, Object, std::shared_ptr<Property>, PyProperty>(m, "Property")
      .def(py::init<const InputParameters &>())
      .def_static("valid_params", &Property::validParams);

  // ---- factory introspection ---------------------------------------------
  m.def("registered_types", [] { return Factory::instance().registeredTypes(); });
  m.def("object_module", [](const std::string & t) { return Factory::instance().module(t); });
  m.def("object_category",
        [](const std::string & t) { return categoryName(Factory::instance().category(t)); });
  m.def("describe_object",
        [](const std::string & t) { return Factory::instance().validParams(t).describe(); });

  // ---- solver options ----------------------------------------------------
  py::class_<SolverOptions>(m, "SolverOptions")
      .def(py::init<>())
      .def_readwrite("nonlinear_solver", &SolverOptions::nonlinear_solver)
      .def_readwrite("max_iterations", &SolverOptions::max_iterations)
      .def_readwrite("relative_tolerance", &SolverOptions::relative_tolerance)
      .def_readwrite("absolute_tolerance", &SolverOptions::absolute_tolerance)
      .def_readwrite("step_tolerance", &SolverOptions::step_tolerance)
      .def_readwrite("relaxation", &SolverOptions::relaxation)
      .def_readwrite("line_search", &SolverOptions::line_search)
      .def_readwrite("max_line_search_steps", &SolverOptions::max_line_search_steps)
      .def_readwrite("load_factors", &SolverOptions::load_factors)
      .def_readwrite("linear_solver", &SolverOptions::linear_solver)
      .def_readwrite("petsc_options", &SolverOptions::petsc_options)
      .def_readwrite("preconditioner", &SolverOptions::preconditioner)
      .def_readwrite("gmres_restart", &SolverOptions::gmres_restart)
      .def_readwrite("linear_tolerance", &SolverOptions::linear_tolerance)
      .def_readwrite("amg_strength_threshold", &SolverOptions::amg_strength_threshold)
      .def_readwrite("linear_max_iterations", &SolverOptions::linear_max_iterations)
      .def_readwrite("verbose", &SolverOptions::verbose)
      .def_readwrite("error_on_divergence", &SolverOptions::error_on_divergence);

  py::class_<TransientOptions>(m, "TransientOptions")
      .def(py::init<>())
      .def_readwrite("start_time", &TransientOptions::start_time)
      .def_readwrite("end_time", &TransientOptions::end_time)
      .def_readwrite("dt", &TransientOptions::dt)
      .def_readwrite("theta", &TransientOptions::theta)
      .def_readwrite("output_times", &TransientOptions::output_times)
      .def_readwrite("output_file_base", &TransientOptions::output_file_base)
      .def_readwrite("output_fields", &TransientOptions::output_fields)
      .def_readwrite("time_stepper", &TransientOptions::time_stepper)
      .def_readwrite("dt_min", &TransientOptions::dt_min)
      .def_readwrite("dt_max", &TransientOptions::dt_max)
      .def_readwrite("growth_factor", &TransientOptions::growth_factor)
      .def_readwrite("cutback_factor", &TransientOptions::cutback_factor)
      .def_readwrite("error_tolerance", &TransientOptions::error_tolerance)
      .def_readwrite("optimal_iterations", &TransientOptions::optimal_iterations)
      .def_readwrite("iteration_window", &TransientOptions::iteration_window)
      .def_readwrite("max_rejected_steps", &TransientOptions::max_rejected_steps);

  py::class_<IterationRecord>(m, "IterationRecord")
      .def_readonly("load_step", &IterationRecord::load_step)
      .def_readonly("load_factor", &IterationRecord::load_factor)
      .def_readonly("iteration", &IterationRecord::iteration)
      .def_readonly("residual_norm", &IterationRecord::residual_norm)
      .def_readonly("step_norm", &IterationRecord::step_norm);

  py::class_<SolveResult>(m, "SolveResult")
      .def_readonly("converged", &SolveResult::converged)
      .def_readonly("total_iterations", &SolveResult::total_iterations)
      .def_readonly("linear_iterations", &SolveResult::linear_iterations)
      .def_readonly("time_steps", &SolveResult::time_steps)
      .def_readonly("rejected_steps", &SolveResult::rejected_steps)
      .def_readonly("step_history", &SolveResult::step_history)
      .def_readonly("history", &SolveResult::history);

  m.def(
      "flush_output",
      []()
      {
        std::fflush(stdout);
        std::cout.flush();
      },
      "Write the progress lines that the library holds in its buffers, so that "
      "they come before the next lines of Python.");

  // ---- structured registry introspection (used to generate the docs) -----
  m.def(
      "build_information",
      []()
      {
        py::dict info;
        info["max_ad_derivatives"] = kMaxDerivatives;
#ifdef _OPENMP
        info["openmp"] = true;
#else
        info["openmp"] = false;
#endif
#ifdef DUALMESH_HAVE_MPI
        info["mpi"] = true;
#else
        info["mpi"] = false;
#endif
        info["petsc"] = petsc::available();
        info["petsc_version"] = petsc::version();
        return info;
      },
      "How this extension was built: the automatic differentiation budget and whether "
      "OpenMP, MPI and PETSc are available.");
  m.def(
      "object_parameters",
      [](const std::string & type)
      {
        const auto params = Factory::instance().validParams(type);
        py::list out;
        for (const auto & [name, info] : params.all())
        {
          if (!name.empty() && name[0] == '_')
            continue; // framework-internal
          py::dict row;
          row["name"] = name;
          row["type"] = parameterKindName(info.kind);
          row["required"] = info.required;
          row["description"] = info.description;
          const std::string text = parameterText(info.value);
          row["default"] = text;
          out.append(row);
        }
        return out;
      },
      py::arg("type"),
      "The declared parameters of a registered type, as a list of dictionaries with "
      "'name', 'type', 'required', 'default' and 'description'.");
  m.def(
      "object_description",
      [](const std::string & type)
      { return Factory::instance().validParams(type).classDescription(); },
      py::arg("type"),
      "The one-paragraph description of a registered type.");

  // ---- parallel ----------------------------------------------------------
  m.def("have_mpi", &Communicator::haveMpi, "Whether the extension was built with MPI support.");
  m.def("have_metis", &haveMetis, "Whether the extension was built against METIS.");
  m.def("have_petsc", &petsc::available, "Whether the extension was built with PETSc.");
  m.def("petsc_version", &petsc::version, "PETSc's version, or an empty string without PETSc.");
  m.def("finalize_mpi",
        &Communicator::finalize,
        "Shut MPI down if this library started it; registered as an interpreter exit hook by "
        "dualmesh.parallel, and harmless to call twice.");
  m.def(
      "mpi_rank",
      []() { return Communicator::world().rank(); },
      "This process's rank, or 0 in a serial run.");
  m.def(
      "mpi_size",
      []() { return Communicator::world().size(); },
      "The number of processes, or 1 in a serial run.");
  m.def(
      "partition_mesh",
      [](std::shared_ptr<Mesh> mesh, int num_parts, const std::string & method)
      {
        const auto p = partitionMesh(*mesh, num_parts, method);
        py::dict out;
        out["element_part"] = p.element_part;
        out["node_owner"] = p.node_owner;
        out["edge_cut"] = p.edgeCut(*mesh);
        const auto [largest, smallest] = p.partSizes();
        out["largest_part"] = largest;
        out["smallest_part"] = smallest;
        return out;
      },
      py::arg("mesh"),
      py::arg("num_parts"),
      py::arg("method") = "recursive_coordinate_bisection");
  m.def(
      "sub_mesh",
      [](std::shared_ptr<Mesh> mesh, const std::vector<Index> & elements)
      {
        std::vector<Index> l2g;
        auto out = std::make_shared<Mesh>(subMesh(*mesh, elements, l2g));
        return py::make_tuple(out, l2g);
      },
      py::arg("mesh"),
      py::arg("elements"));

  py::class_<DistributedOptions>(m, "DistributedOptions")
      .def(py::init<>())
      .def_readwrite("partitioner", &DistributedOptions::partitioner)
      .def_readwrite("linear_solver", &DistributedOptions::linear_solver)
      .def_readwrite("petsc_options", &DistributedOptions::petsc_options)
      .def_readwrite("preconditioner", &DistributedOptions::preconditioner)
      .def_readwrite("overlap", &DistributedOptions::overlap)
      .def_readwrite("subdomain_solver", &DistributedOptions::subdomain_solver)
      .def_readwrite("linear_tolerance", &DistributedOptions::linear_tolerance)
      .def_readwrite("linear_max_iterations", &DistributedOptions::linear_max_iterations)
      .def_readwrite("verbose", &DistributedOptions::verbose);

  py::class_<DistributedProblem>(m, "DistributedProblem")
      .def(py::init(
               [](std::shared_ptr<Mesh> mesh,
                  const std::string & method,
                  const std::string & coordinates,
                  const DistributedOptions & options)
               {
                 return std::make_unique<DistributedProblem>(
                     *mesh, methodFromString(method), coordFromString(coordinates), options);
               }),
           py::arg("mesh"),
           py::arg("method") = "dmcdm",
           py::arg("coordinates") = "cartesian",
           py::arg("options") = DistributedOptions{})
      .def(
          "local",
          [](DistributedProblem & d) -> Problem & { return d.local(); },
          py::return_value_policy::reference_internal)
      .def("rank", &DistributedProblem::rank)
      .def("num_ranks", &DistributedProblem::numRanks)
      .def("num_owned_dofs", &DistributedProblem::numOwnedDofs)
      .def("num_global_dofs", &DistributedProblem::numGlobalDofs)
      .def("global_node", &DistributedProblem::globalNode, py::arg("local_node"))
      .def("owns_node", &DistributedProblem::ownsNode, py::arg("local_node"))
      .def("set_linear_solver", &DistributedProblem::setLinearSolver, py::arg("options"))
      .def("solve_steady", &DistributedProblem::solveSteady, py::arg("options") = SolverOptions{})
      .def("solve_transient",
           &DistributedProblem::solveTransient,
           py::arg("transient"),
           py::arg("options") = SolverOptions{})
      .def("gathered_values", &DistributedProblem::gatheredValues, py::arg("variable"))
      .def("write_vtu",
           &DistributedProblem::writeVTU,
           py::arg("base"),
           py::arg("cell_properties") = std::vector<std::string>{},
           py::arg("fields") = std::vector<std::string>{})
      .def("summary", &DistributedProblem::summary);

  // ---- problem -----------------------------------------------------------
  py::class_<Problem::StateSnapshot>(m, "StateSnapshot");
  py::class_<Problem>(m, "Problem")
      .def(py::init(
               [](std::shared_ptr<Mesh> mesh,
                  const std::string & method,
                  const std::string & coordinates)
               {
                 return std::make_unique<Problem>(
                     std::move(mesh), methodFromString(method), coordFromString(coordinates));
               }),
           py::arg("mesh"),
           py::arg("method") = "dmcdm",
           py::arg("coordinates") = "cartesian")
      .def(
          "set_boundary_gradient",
          [](Problem & p, const std::string & s)
          { p.setBoundaryGradient(boundaryGradientFromString(s)); },
          py::arg("order"))
      .def("_linear_system",
           [](Problem & p)
           {
             // The steady residual and its Jacobian at the current solution,
             // with the prescribed values written in and the rows of the
             // prescribed degrees of freedom replaced, exactly as one Newton
             // step of solve_steady sees them.
             Vector R;
             SparseMatrix J;
             {
               py::gil_scoped_release release;
               p.initialize();
               p.applyDirichlet(p.solution(), 1.0);
               Problem::AssemblyOptions options;
               Vector lagged = p.solution();
               options.lagged = &lagged;
               p.assemble(p.solution(), options, R, &J);
               p.dirichletRows(p.solution(), 1.0, R, &J);
               J.makeCompressed();
             }
             const auto n = J.rows();
             py::array_t<double> residual(n, R.data());
             py::array_t<int> indptr(n + 1, J.outerIndexPtr());
             py::array_t<int> indices(J.nonZeros(), J.innerIndexPtr());
             py::array_t<double> values(J.nonZeros(), J.valuePtr());
             return py::make_tuple(residual, values, indices, indptr);
           })
      .def("is_cell_centered", &Problem::isCellCentered)
      .def("set_num_threads", &Problem::setNumThreads, py::arg("num_threads"))
      .def("num_threads", &Problem::numThreads)
      .def("effective_threads", &Problem::effectiveThreads)
      .def("thread_safe", &Problem::threadSafe)
      .def("num_entities", &Problem::numEntities)
      .def("entity_point",
           [](const Problem & p, Index i)
           {
             const Point & x = p.entityPoint(i);
             return std::vector<double>{x[0], x[1], x[2]};
           })
      .def("boundary_entities", &Problem::boundaryEntities, py::arg("boundary"))
      .def(
          "add_variable",
          [](Problem & p,
             const std::string & name,
             const std::vector<std::string> & blocks,
             const py::object & initial_condition,
             const std::string & order)
          {
            FunctionPtr ic = initial_condition.is_none() ? nullptr : toFunction(initial_condition);
            VariableOrder o;
            if (order == "mesh")
              o = VariableOrder::Mesh;
            else if (order == "first")
              o = VariableOrder::First;
            else
              throw InputError("Unknown variable order '" + order +
                               "'. Use 'mesh' (the order of the elements) or 'first' (linear "
                               "on every element, the pressure of a Taylor-Hood element).");
            return p.addVariable(name, blocks, ic, 1.0, o);
          },
          py::arg("name"),
          py::arg("blocks") = std::vector<std::string>{},
          py::arg("initial_condition") = py::none(),
          py::arg("order") = "mesh")
      .def(
          "variable_order",
          [](const Problem & p, const std::string & name)
          {
            return p.variable(p.variableIndex(name)).order == VariableOrder::First
                       ? std::string("first")
                       : std::string("mesh");
          },
          py::arg("name"))
      .def("set_variable_unit", &Problem::setVariableUnit, py::arg("name"), py::arg("unit"))
      .def(
          "variable_unit",
          [](const Problem & p, const std::string & name)
          { return p.variable(p.variableIndex(name)).unit; },
          py::arg("name"))
      .def("has_mixed_order", &Problem::hasMixedOrder)
      .def("add_function",
           [](Problem & p, const std::string & name, const py::object & f)
           { p.addFunction(name, toFunction(f)); })
      .def("set_element_field",
           &Problem::setElementField,
           py::arg("name"),
           py::arg("values"),
           "Set a named field with one value per element, read by the property objects.")
      .def(
          "element_field",
          [](Problem & p, const std::string & name) { return p.elementField(name); },
          py::arg("name"))
      .def("element_field_names", &Problem::elementFieldNames)
      .def("has_state", &Problem::hasState, "Whether a property object keeps a history.")
      .def("save_state",
           &Problem::snapshotState,
           "A copy of the committed history of the stateful property objects.")
      .def("restore_state",
           &Problem::restoreState,
           py::arg("snapshot"),
           "Restore a history saved with save_state().")
      .def("state_at_elements",
           &Problem::stateAtElements,
           "The committed history averaged over each element's volume points.")
      .def(
          "add_object",
          [](Problem & p,
             const std::string & type,
             const std::string & name,
             const py::kwargs & kwargs)
          {
            return p.addObject(
                type, name, paramsFromKwargs(Factory::instance().validParams(type), kwargs));
          },
          py::arg("type"),
          py::arg("name") = "")
      .def("add_kernel", &Problem::addKernel)
      .def("add_integrated_bc", &Problem::addIntegratedBC)
      .def("add_nodal_bc", &Problem::addNodalBC)
      .def("add_nodal_load", &Problem::addNodalLoad)
      .def("add_property", &Problem::addProperty)
      .def("initialize", &Problem::initialize)
      .def(
          "near_nullspace",
          [](Problem & p)
          {
            p.initialize();
            const Eigen::MatrixXd B = p.nearNullspace();
            std::vector<std::vector<double>> rows(
                static_cast<std::size_t>(B.rows()),
                std::vector<double>(static_cast<std::size_t>(B.cols())));
            for (Eigen::Index i = 0; i < B.rows(); ++i)
              for (Eigen::Index j = 0; j < B.cols(); ++j)
                rows[i][j] = B(i, j);
            return rows;
          },
          "The near null space of the algebraic multigrid preconditioner, one row per degree of "
          "freedom.")
      .def("boundary_measure",
           &Problem::boundaryMeasure,
           py::arg("boundaries"),
           "The measure of the named side sets, with the coordinate factor of axisymmetric and "
           "spherical problems.")
      .def(
          "num_active_dofs",
          [](Problem & p)
          {
            p.initialize();
            Index n = 0;
            for (char a : p.activeDofs())
              n += a ? 1 : 0;
            return n;
          },
          "The number of degrees of freedom that carry an equation: every variable at every node "
          "where a kernel acts on it, and for a first-order variable on a quadratic mesh only the "
          "corner nodes.")
      .def_property_readonly("mesh", &Problem::meshPointer)
      .def_property_readonly("num_variables", &Problem::numVariables)
      .def("variable_index", &Problem::variableIndex)
      .def("variable_name", [](const Problem & p, int i) { return p.variable(i).name; })
      .def(
          "property_registry",
          [](Problem & p) -> PropertyRegistry & { return p.propertyRegistry(); },
          py::return_value_policy::reference_internal)
      .def("property_id",
           [](Problem & p, const std::string & n) { return p.propertyRegistry().id(n); })
      .def_property("time", &Problem::time, &Problem::setTime)
      .def("values",
           [](const Problem & p, const std::string & v)
           {
             auto vals = p.values(v);
             return py::array_t<double>(vals.size(), vals.data());
           })
      .def("set_values",
           [](Problem & p, const std::string & v, const std::vector<double> & x)
           { p.setValues(v, x); })
      .def("solution",
           [](const Problem & p)
           { return py::array_t<double>(p.solution().size(), p.solution().data()); })
      .def("apply_initial_conditions", &Problem::applyInitialConditions)
      .def("solve_steady", &Problem::solveSteady, py::arg("options") = SolverOptions())
      .def("solve_transient",
           &Problem::solveTransient,
           py::arg("transient"),
           py::arg("options") = SolverOptions())
      .def("set_time_step_callback",
           [](Problem & p, const py::function & f)
           {
             p.setTimeStepCallback(
                 [f](double t, Problem & pb)
                 {
                   py::gil_scoped_acquire gil;
                   f(t, py::cast(&pb, py::return_value_policy::reference));
                 });
           })
      .def("set_output_callback",
           [](Problem & p, const py::object & f)
           {
             if (f.is_none())
             {
               p.setOutputCallback(nullptr);
               return;
             }
             p.setOutputCallback(
                 [f](int index)
                 {
                   py::gil_scoped_acquire gil;
                   f(index);
                 });
           })
      .def("reactions", &Problem::reactions)
      .def("total_reaction", &Problem::totalReaction)
      .def("sample", &Problem::sample)
      .def("gradient_at_centroids", &Problem::gradientAtCentroids)
      .def("error_indicator",
           &Problem::errorIndicator,
           py::arg("variable"),
           "One error indicator per element, from gradient recovery.")
      .def("property_at_centroids", &Problem::propertyAtCentroids)
      .def("kernel_flux_at_centroids", &Problem::kernelFluxAtCentroids)
      .def("integrate", &Problem::integrate)
      .def("element_integrals", &Problem::elementIntegrals)
      .def(
          "error_norms",
          [](const Problem & p,
             const std::string & variable,
             const py::object & exact,
             const py::object & gradient,
             int quadrature_points)
          {
            const FunctionPtr u = toFunction(exact);
            std::array<FunctionPtr, 3> g;
            std::array<const Function *, 3> gp = {nullptr, nullptr, nullptr};
            if (!gradient.is_none())
            {
              const auto list = py::cast<py::sequence>(gradient);
              for (std::size_t d = 0; d < 3; ++d)
              {
                g[d] =
                    d < list.size() ? toFunction(list[d]) : std::make_shared<ConstantFunction>(0.0);
                gp[d] = g[d].get();
              }
            }
            Problem::ErrorNorms norms;
            {
              // Python callables take the lock back themselves; everything
              // else runs without it, on several threads.
              py::gil_scoped_release release;
              norms = p.errorNorms(variable, *u, gp, quadrature_points);
            }
            return py::make_tuple(norms.l2, norms.h1_seminorm);
          },
          py::arg("variable"),
          py::arg("exact"),
          py::arg("exact_gradient") = py::none(),
          py::arg("quadrature_points") = 0,
          "The L2 norm and the H1 seminorm of the error against an exact solution.")
      .def("boundary_flux_integral", &Problem::boundaryFluxIntegral)
      .def("write_vtu",
           &Problem::writeVTU,
           py::arg("filename"),
           py::arg("cell_properties") = std::vector<std::string>{},
           py::arg("fields") = std::vector<std::string>{})
      .def("summary", &Problem::summary)
      .def("object_names", &Problem::objectNames)
      .def("object", &Problem::object, py::arg("name"))
      .def("variable_names",
           [](const Problem & p)
           {
             std::vector<std::string> names;
             for (int v = 0; v < p.numVariables(); ++v)
               names.push_back(p.variable(v).name);
             return names;
           });
}
