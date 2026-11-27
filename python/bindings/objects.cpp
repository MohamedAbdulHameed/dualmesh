// SPDX-License-Identifier: LGPL-2.1-or-later
//
// Bindings of the input parameters, the state at an integration point, the objects of a problem
// with their trampolines for objects written in Python, and the registry of object types.
#include "bindings.h"
#include "dualmesh/base/Factory.h"
#include "dualmesh/core/InputParameters.h"

#include <sstream>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>

namespace dualmesh::bindings
{

namespace
{

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
        else if constexpr (std::is_same_v<T, RealMap>)
        {
          std::ostringstream os;
          bool first = true;
          for (const auto & [key, item] : v)
          {
            os << (first ? "" : ", ") << key << ": " << item;
            first = false;
          }
          text = os.str();
        }
        else
          text = "function";
      },
      value);
  return text;
}

} // namespace

void
bindObjects(py::module_ & m)
{
  // ---- parameters and objects ---------------------------------------------
  py::class_<InputParameters>(m, "InputParameters")
      .def(py::init<>())
      .def("set",
           [](InputParameters & p, const std::string & n, const py::object & v)
           { p.set(n, toParameter(v)); })
      .def("has", &InputParameters::has, py::arg("name"))
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
      .def_readonly("data_fraction", &QpContext::data_fraction)
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

  // ---- structured registry introspection (used to generate the docs) -----
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
}

} // namespace dualmesh::bindings
