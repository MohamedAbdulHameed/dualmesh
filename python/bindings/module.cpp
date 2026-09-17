// SPDX-License-Identifier: LGPL-2.1-or-later
//
// pybind11 bindings: the whole framework is usable from Python, including
// user-defined kernels, boundary conditions, and materials written in Python.
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "dualmesh/base/Factory.h"
#include "dualmesh/base/Problem.h"
#include "dualmesh/modules/Framework.h"

namespace py = pybind11;
using namespace dualmesh;

namespace
{

/// A Function backed by a Python callable f(x, y, z, t).
class PythonFunction : public Function
{
public:
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
  if (py::isinstance<py::function>(o))
    return std::make_shared<PythonFunction>(py::cast<py::function>(o));
  if (py::isinstance<Function>(o))
    return py::cast<std::shared_ptr<Function>>(o);
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
  if (py::isinstance<py::function>(v))
    return std::static_pointer_cast<Function>(
        std::make_shared<PythonFunction>(py::cast<py::function>(v)));
  if (py::isinstance<Function>(v))
    return py::cast<std::shared_ptr<Function>>(v);
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

class PyMaterial : public Material
{
public:
  using Material::Material;
  void initialSetup(Problem & problem) override
  {
    PYBIND11_OVERRIDE_NAME(void, Material, "initial_setup", initialSetup, std::ref(problem));
  }
  void declareProperties(MaterialPropertyRegistry & r) override
  {
    PYBIND11_OVERRIDE_PURE_NAME(
        void, Material, "declare_properties", declareProperties, std::ref(r));
  }
  void computeProperties(QpContext & ctx) const override
  {
    py::gil_scoped_acquire gil;
    py::function f = py::get_override(static_cast<const Material *>(this), "compute_properties");
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
  throw InputError("Unknown method '" + s + "' (use 'dmcdm' or 'fem').");
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

  // ---- functions ---------------------------------------------------------
  py::class_<Function, std::shared_ptr<Function>>(m, "Function").def("value", &Function::value);
  py::class_<ConstantFunction, Function, std::shared_ptr<ConstantFunction>>(m, "ConstantFunction")
      .def(py::init<double>());
  py::class_<PiecewiseLinearFunction, Function, std::shared_ptr<PiecewiseLinearFunction>>(
      m, "PiecewiseLinearFunction")
      .def(py::init<std::vector<double>, std::vector<double>, int>(),
           py::arg("abscissa"),
           py::arg("ordinate"),
           py::arg("axis") = 3);
  m.def("make_function", &toFunction);

  // ---- mesh --------------------------------------------------------------
  py::class_<Mesh, std::shared_ptr<Mesh>>(m, "Mesh")
      .def(py::init<int>(), py::arg("dimension") = 2)
      .def_property_readonly("dimension", &Mesh::dimension)
      .def_property_readonly("num_nodes", &Mesh::numNodes)
      .def_property_readonly("num_elements", &Mesh::numElements)
      .def("add_node", &Mesh::addNode, py::arg("point"))
      .def(
          "add_element",
          [](Mesh & mesh, const std::string & type, const std::vector<Index> & nodes, int block)
          { return mesh.addElement(elementTypeFromName(type), nodes, block); },
          py::arg("element_type"),
          py::arg("nodes"),
          py::arg("block") = 0)
      .def("node", &Mesh::node)
      .def("points",
           [](const Mesh & mesh)
           {
             py::array_t<double> a(
                 {static_cast<py::ssize_t>(mesh.numNodes()), static_cast<py::ssize_t>(3)});
             auto r = a.mutable_unchecked<2>();
             for (Index i = 0; i < mesh.numNodes(); ++i)
               for (int d = 0; d < 3; ++d)
                 r(i, d) = mesh.node(i)[d];
             return a;
           })
      .def("cells",
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
           })
      .def("element_nodes",
           [](const Mesh & mesh, Index e)
           {
             std::vector<Index> out;
             for (int k = 0; k < mesh.element(e).numNodes(); ++k)
               out.push_back(mesh.element(e).nodes[k]);
             return out;
           })
      .def("element_type",
           [](const Mesh & mesh, Index e) { return elementTypeName(mesh.element(e).type); })
      .def("set_block_name", &Mesh::setBlockName)
      .def("block_ids", &Mesh::blockIds)
      .def("add_sideset_from_faces", &Mesh::addSidesetFromFaces)
      .def("add_nodeset", &Mesh::addNodeset)
      .def("alias_sideset", &Mesh::aliasSideset)
      .def("add_sideset_by_predicate",
           [](Mesh & mesh, const std::string & name, const py::function & f)
           {
             mesh.addSidesetByPredicate(name,
                                        [f](const Point & p)
                                        {
                                          py::gil_scoped_acquire gil;
                                          return py::cast<bool>(f(p[0], p[1], p[2]));
                                        });
           })
      .def("add_nodeset_by_predicate",
           [](Mesh & mesh, const std::string & name, const py::function & f)
           {
             mesh.addNodesetByPredicate(name,
                                        [f](const Point & p)
                                        {
                                          py::gil_scoped_acquire gil;
                                          return py::cast<bool>(f(p[0], p[1], p[2]));
                                        });
           })
      .def("transform_nodes",
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
           })
      .def("fix_orientation", &Mesh::fixOrientation)
      .def("refined", &Mesh::refined)
      .def("add_bounding_box_sidesets", &Mesh::addBoundingBoxSidesets, py::arg("tolerance") = 1e-10)
      .def("boundary_nodes", &Mesh::boundaryNodes)
      .def("sideset_names",
           [](const Mesh & mesh)
           {
             std::vector<std::string> out;
             for (const auto & [n, s] : mesh.sidesets())
               out.push_back(n);
             return out;
           })
      .def("nodeset_names",
           [](const Mesh & mesh)
           {
             std::vector<std::string> out;
             for (const auto & [n, s] : mesh.nodesets())
               out.push_back(n);
             return out;
           })
      .def("bounding_box", &Mesh::boundingBox)
      .def("element_centroid", &Mesh::elementCentroid)
      .def("summary", &Mesh::summary)
      .def("__repr__", [](const Mesh & mesh) { return "<dualmesh.Mesh " + mesh.summary() + ">"; });

  m.def("generate_line_mesh", &generateLineMesh, py::arg("x"));
  m.def("generate_rectangle_mesh",
        &generateRectangleMesh,
        py::arg("x"),
        py::arg("y"),
        py::arg("element_type") = "Quad4",
        py::arg("diagonal") = "right");
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

  py::class_<MaterialPropertyRegistry>(m, "MaterialPropertyRegistry")
      .def(
          "declare", &MaterialPropertyRegistry::declare, py::arg("name"), py::arg("components") = 1)
      .def("id", &MaterialPropertyRegistry::id);

  py::class_<Object, std::shared_ptr<Object>>(m, "Object")
      .def_property_readonly("name", &Object::name)
      .def_property_readonly("type", &Object::type)
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

  py::class_<Material, Object, std::shared_ptr<Material>, PyMaterial>(m, "Material")
      .def(py::init<const InputParameters &>())
      .def_static("valid_params", &Material::validParams);

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
      .def_readwrite("load_factors", &SolverOptions::load_factors)
      .def_readwrite("linear_solver", &SolverOptions::linear_solver)
      .def_readwrite("linear_tolerance", &SolverOptions::linear_tolerance)
      .def_readwrite("linear_max_iterations", &SolverOptions::linear_max_iterations)
      .def_readwrite("verbose", &SolverOptions::verbose)
      .def_readwrite("error_on_divergence", &SolverOptions::error_on_divergence);

  py::class_<TransientOptions>(m, "TransientOptions")
      .def(py::init<>())
      .def_readwrite("start_time", &TransientOptions::start_time)
      .def_readwrite("end_time", &TransientOptions::end_time)
      .def_readwrite("dt", &TransientOptions::dt)
      .def_readwrite("theta", &TransientOptions::theta)
      .def_readwrite("output_interval", &TransientOptions::output_interval)
      .def_readwrite("output_file_base", &TransientOptions::output_file_base);

  py::class_<IterationRecord>(m, "IterationRecord")
      .def_readonly("load_step", &IterationRecord::load_step)
      .def_readonly("load_factor", &IterationRecord::load_factor)
      .def_readonly("iteration", &IterationRecord::iteration)
      .def_readonly("residual_norm", &IterationRecord::residual_norm)
      .def_readonly("step_norm", &IterationRecord::step_norm);

  py::class_<SolveResult>(m, "SolveResult")
      .def_readonly("converged", &SolveResult::converged)
      .def_readonly("total_iterations", &SolveResult::total_iterations)
      .def_readonly("history", &SolveResult::history);

  // ---- problem -----------------------------------------------------------
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
          "add_variable",
          [](Problem & p,
             const std::string & name,
             const std::vector<std::string> & blocks,
             const py::object & initial_condition)
          {
            FunctionPtr ic = initial_condition.is_none() ? nullptr : toFunction(initial_condition);
            return p.addVariable(name, blocks, ic);
          },
          py::arg("name"),
          py::arg("blocks") = std::vector<std::string>{},
          py::arg("initial_condition") = py::none())
      .def("add_function",
           [](Problem & p, const std::string & name, const py::object & f)
           { p.addFunction(name, toFunction(f)); })
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
      .def("add_material", &Problem::addMaterial)
      .def("initialize", &Problem::initialize)
      .def_property_readonly("mesh", &Problem::meshPointer)
      .def_property_readonly("num_variables", &Problem::numVariables)
      .def("variable_index", &Problem::variableIndex)
      .def("variable_name", [](const Problem & p, int i) { return p.variable(i).name; })
      .def(
          "property_registry",
          [](Problem & p) -> MaterialPropertyRegistry & { return p.propertyRegistry(); },
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
      .def("reactions", &Problem::reactions)
      .def("total_reaction", &Problem::totalReaction)
      .def("sample", &Problem::sample)
      .def("gradient_at_centroids", &Problem::gradientAtCentroids)
      .def("property_at_centroids", &Problem::propertyAtCentroids)
      .def("kernel_flux_at_centroids", &Problem::kernelFluxAtCentroids)
      .def("integrate", &Problem::integrate)
      .def("boundary_flux_integral", &Problem::boundaryFluxIntegral)
      .def("write_vtu",
           &Problem::writeVTU,
           py::arg("filename"),
           py::arg("cell_properties") = std::vector<std::string>{})
      .def("summary", &Problem::summary);
}
