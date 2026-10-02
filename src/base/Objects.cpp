// SPDX-License-Identifier: LGPL-2.1-or-later
//
// Base-class implementations: Object, ResidualObject, Kernel, boundary
// conditions, nodal loads, materials, and the Factory.
#include "dualmesh/base/Factory.h"
#include "dualmesh/base/Kernel.h"
#include "dualmesh/base/Material.h"
#include "dualmesh/base/Problem.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <limits>
#include <sstream>

namespace dualmesh
{

// ---- Object -------------------------------------------------------------------
InputParameters
Object::validParams()
{
  InputParameters p;
  p.addOptional("_name", ParameterKind::String, std::string(""), "object name");
  p.addOptional("_type", ParameterKind::String, std::string(""), "registered type");
  return p;
}

Object::Object(const InputParameters & params)
    : _params(params), _name(params.getString("_name")), _type(params.getString("_type"))
{
}

void
Object::initialSetup(Problem &)
{
}

FunctionPtr
Object::getFunction(Problem & problem, const std::string & param) const
{
  const auto & raw = _params.getRaw(param);
  if (std::holds_alternative<double>(raw))
    return std::make_shared<ConstantFunction>(std::get<double>(raw));
  if (std::holds_alternative<long>(raw))
    return std::make_shared<ConstantFunction>(double(std::get<long>(raw)));
  if (std::holds_alternative<std::string>(raw))
    return problem.function(std::get<std::string>(raw));
  if (std::holds_alternative<std::shared_ptr<Function>>(raw))
    return std::get<std::shared_ptr<Function>>(raw);
  throw InputError("Parameter '" + param + "' of '" + _name + "' has no value.");
}

// ---- ResidualObject ----------------------------------------------------------
InputParameters
ResidualObject::validParams()
{
  InputParameters p = Object::validParams();
  p.addRequired("variable", ParameterKind::String, "The variable (equation) this object acts on.");
  p.addOptional("block",
                ParameterKind::StringList,
                std::vector<std::string>{},
                "Blocks (subdomains) this object acts on. An empty list selects every block.");
  p.addOptional("quadrature",
                ParameterKind::String,
                std::string("automatic"),
                "Quadrature rule for the integration of this term. The default, automatic, uses "
                "Gauss-Legendre with as many points per direction as the polynomial order of the "
                "mesh plus one, which is two points on a linear mesh and three on a quadratic "
                "one. The choices are gauss1 to gauss10 (Gauss-Legendre with that many points per "
                "direction), midpoint, trapezoid, simpson, nodal (a single point at the owning "
                "node, which lumps the term), interface (a single point at the control domain "
                "interface), and control_domain_trapezoid (the trapezoidal rule over the whole "
                "control domain, which is the source rule of the classical finite volume method). "
                "The aliases trapezoidal, lumped, centroid and cd_trapezoid are accepted, a bare "
                "gauss means gauss2, and the name is matched without regard to case. The "
                "automatic default is adequate for a smooth coefficient on a mildly distorted "
                "element. Raise it for strongly curved elements or a rapidly varying coefficient, "
                "and use nodal to lump a capacity or mass term.");
  p.addOptional("reduced_integration",
                ParameterKind::Boolean,
                false,
                "Evaluate the solution and its gradients at the centroid of the element while "
                "still integrating with the rule named by 'quadrature'. The geometry is "
                "unaffected. This is selective reduced integration. Turn it on only for the "
                "transverse shear terms of a thin beam or plate and for the penalty term of an "
                "incompressible flow, where it removes locking. Using it on a bending or "
                "diffusion term degrades the accuracy.");
  p.addOptional("scale_with_load",
                ParameterKind::Boolean,
                false,
                "Multiply this contribution by the load factor during load stepping.");
  return p;
}

bool
Object::parametersAreThreadSafe() const
{
  for (const auto & [name, info] : _params.all())
  {
    (void) name;
    if (const auto * f = std::get_if<std::shared_ptr<Function>>(&info.value))
      if (*f && !(*f)->threadSafe())
        return false;
  }
  return true;
}

ResidualObject::ResidualObject(const InputParameters & params) : Object(params)
{
  _var_name = params.getString("variable");
  _quad =
      QuadratureSpec::parse(params.getString("quadrature"), params.getBool("reduced_integration"));
  _scale_with_load = params.getBool("scale_with_load");
}

void
ResidualObject::initialSetup(Problem & problem)
{
  _var = problem.variableIndex(_var_name);
  if (_quad.automatic)
  {
    // The rule must integrate a product of shape functions exactly, so it
    // needs one more point per direction than the polynomial order.
    int order = 1;
    for (const auto & el : problem.mesh().elements())
      if (elementIsQuadratic(el.type))
        order = 2;
    _quad.points = order + 1;
    _quad.automatic = false;
  }
  _blocks.clear();
  for (const auto & b : _params.getStringList("block"))
    _blocks.insert(problem.mesh().blockId(b));
}

int
ResidualObject::coupledVariable(Problem & problem, const std::string & param) const
{
  return problem.variableIndex(_params.getString(param));
}

// ---- Kernel ---------------------------------------------------------------------
InputParameters
Kernel::validParams()
{
  return ResidualObject::validParams();
}

Kernel::Kernel(const InputParameters & params) : ResidualObject(params) {}

void
Kernel::computeFlux(const QpContext &, ADVector3 & flux) const
{
  flux = {ADReal(0.0), ADReal(0.0), ADReal(0.0)};
}

ADReal
Kernel::computeSource(const QpContext &) const
{
  return ADReal(0.0);
}

// ---- IntegratedBC ---------------------------------------------------------------
InputParameters
IntegratedBC::validParams()
{
  InputParameters p = ResidualObject::validParams();
  p.addRequired("boundary", ParameterKind::StringList, "Side sets where the condition applies.");
  return p;
}

IntegratedBC::IntegratedBC(const InputParameters & params) : ResidualObject(params)
{
  _boundaries = params.getStringList("boundary");
}

void
IntegratedBC::initialSetup(Problem & problem)
{
  ResidualObject::initialSetup(problem);
  for (const auto & b : _boundaries)
    problem.mesh().sideset(b); // throws if unknown
}

// ---- InterfaceBC --------------------------------------------------------------------
InputParameters
InterfaceBC::validParams()
{
  InputParameters p = IntegratedBC::validParams();
  p.addRequired("secondary_boundary",
                ParameterKind::StringList,
                "Side sets on the other side of the gap. Every integration point of 'boundary' "
                "is paired with the closest point of these side sets.");
  p.addOptional("secondary_variable",
                ParameterKind::String,
                std::string(),
                "The equation on the secondary side that receives the opposite flux. An empty "
                "value selects 'variable'.");
  return p;
}

InterfaceBC::InterfaceBC(const InputParameters & params) : IntegratedBC(params)
{
  _secondary = params.getStringList("secondary_boundary");
}

void
InterfaceBC::initialSetup(Problem & problem)
{
  IntegratedBC::initialSetup(problem);
  for (const auto & b : _secondary)
    problem.mesh().sideset(b);
  const auto name = _params.getString("secondary_variable");
  _secondary_var = name.empty() ? _var : problem.variableIndex(name);
  _coupled.clear();
  addCoupled(_var);
  addCoupled(_secondary_var);
}

void
InterfaceBC::addCoupled(int v)
{
  if (v >= 0 && std::find(_coupled.begin(), _coupled.end(), v) == _coupled.end())
    _coupled.push_back(v);
}

// ---- NodalBC ------------------------------------------------------------------------
InputParameters
NodalBC::validParams()
{
  InputParameters p = ResidualObject::validParams();
  p.addRequired("boundary",
                ParameterKind::StringList,
                "Node sets or side sets where the value is prescribed.");
  return p;
}

NodalBC::NodalBC(const InputParameters & params) : ResidualObject(params)
{
  _boundaries = params.getStringList("boundary");
}

void
NodalBC::initialSetup(Problem & problem)
{
  ResidualObject::initialSetup(problem);
  std::set<Index> ids;
  for (const auto & b : _boundaries)
    for (Index n : problem.boundaryEntities(b))
      ids.insert(n);
  _nodes.assign(ids.begin(), ids.end());
}

// ---- NodalLoad ------------------------------------------------------------------------
InputParameters
NodalLoad::validParams()
{
  InputParameters p = ResidualObject::validParams();
  p.setClassDescription(
      "Concentrated source at nodes (point force or point heat source). The residual of the "
      "node's equation receives -value.");
  p.addRequired("value",
                ParameterKind::Function,
                "Magnitude of the concentrated source, in the units conjugate to the "
                "variable: a force for a displacement, a heat rate in watts for a "
                "temperature. The residual of the node's equation receives minus this value, "
                "so a positive number acts along the positive direction of the variable or "
                "adds heat to the body.");
  p.addOptional("boundary",
                ParameterKind::StringList,
                std::vector<std::string>{},
                "Node sets or side sets whose nodes receive the load.");
  p.addOptional("points",
                ParameterKind::RealList,
                std::vector<double>{},
                "Coordinates of the loaded points: three numbers per point in two and three "
                "dimensions, one number per point in one dimension. Each point is snapped to the "
                "nearest node, however far away that node is, without a warning, so check that "
                "the mesh has a node where the load belongs. Points that snap to the same node "
                "are merged and the load is applied once. Give 'points', 'boundary', or both. "
                "Giving neither is an error.");
  p.addOptional("scale_with_load",
                ParameterKind::Boolean,
                true,
                "Multiply this load by the load factor during load stepping. This parameter "
                "defaults to true, unlike the framework default, because an applied load is "
                "normally the quantity that is ramped. Set it to false for a preload that must "
                "stay fixed while the rest of the loading is increased.");
  return p;
}

NodalLoad::NodalLoad(const InputParameters & params) : ResidualObject(params) {}

void
NodalLoad::initialSetup(Problem & problem)
{
  ResidualObject::initialSetup(problem);
  _value = getFunction(problem, "value");
  std::set<Index> ids;
  for (const auto & b : _params.getStringList("boundary"))
    for (Index n : problem.boundaryEntities(b))
      ids.insert(n);
  _boundary_nodes.assign(ids.begin(), ids.end());
  _point_nodes.clear();
  const auto pts = _params.getRealList("points");
  const int dim = problem.mesh().dimension();
  const int stride = dim == 1 ? 1 : 3;
  if (pts.size() % stride)
    throw InputError("NodalLoad '" + _name + "': 'points' must hold " + std::to_string(stride) +
                     " numbers per point.");
  for (std::size_t i = 0; i < pts.size(); i += stride)
  {
    Point p{pts[i], stride == 3 ? pts[i + 1] : 0.0, stride == 3 ? pts[i + 2] : 0.0};
    Index best = -1;
    double bd = std::numeric_limits<double>::infinity();
    for (Index n = 0; n < problem.numEntities(); ++n)
    {
      const double d = norm(problem.entityPoint(n) - p);
      if (d < bd)
      {
        bd = d;
        best = n;
      }
    }
    ids.insert(best);
    _point_nodes.emplace_back(best, bd);
  }
  if (ids.empty())
    throw InputError("NodalLoad '" + _name + "' has no nodes: give 'boundary' or 'points'.");
  _nodes.assign(ids.begin(), ids.end());
}

void
NodalLoad::keepPoints(const std::vector<char> & keep)
{
  if (keep.size() != _point_nodes.size())
    throw InputError("NodalLoad '" + _name + "': keepPoints needs one flag per requested point.");
  std::set<Index> ids(_boundary_nodes.begin(), _boundary_nodes.end());
  std::vector<std::pair<Index, double>> kept;
  for (std::size_t i = 0; i < keep.size(); ++i)
    if (keep[i])
    {
      ids.insert(_point_nodes[i].first);
      kept.push_back(_point_nodes[i]);
    }
  _point_nodes = kept;
  _nodes.assign(ids.begin(), ids.end());
}

double
NodalLoad::computeValue(const Point & x, double t) const
{
  return _value->value(x, t);
}

// ---- Materials -------------------------------------------------------------------------
int
MaterialPropertyRegistry::declare(const std::string & name, int components)
{
  auto it = _props.find(name);
  if (it != _props.end())
  {
    if (it->second.second != components)
      throw InputError("Material property '" + name + "' declared with different sizes.");
    return it->second.first;
  }
  const int id = _size;
  _props[name] = {id, components};
  _size += components;
  return id;
}

int
MaterialPropertyRegistry::id(const std::string & name) const
{
  auto it = _props.find(name);
  if (it == _props.end())
  {
    std::ostringstream os;
    os << "Material property '" << name << "' is not provided by any material. Available:";
    for (const auto & [n, _] : _props)
      os << " " << n;
    throw InputError(os.str());
  }
  return it->second.first;
}

int
MaterialPropertyRegistry::components(const std::string & name) const
{
  id(name);
  return _props.at(name).second;
}

InputParameters
Material::validParams()
{
  InputParameters p = Object::validParams();
  p.addOptional("block",
                ParameterKind::StringList,
                std::vector<std::string>{},
                "Blocks (subdomains) where this material applies. An empty list selects every "
                "block.");
  p.addOptional("scaled_properties",
                ParameterKind::StringList,
                std::vector<std::string>{},
                "Properties computed by this material to multiply by the matching entries of "
                "'property_factors', for sensitivity and uncertainty studies (for example "
                "thermal_conductivity). Default none.");
  p.addOptional("property_factors",
                ParameterKind::RealList,
                std::vector<double>{},
                "Factors on the 'scaled_properties', one each. Default none.");
  return p;
}

Material::Material(const InputParameters & params) : Object(params)
{
  if (_params.getStringList("scaled_properties").size() !=
      _params.getRealList("property_factors").size())
    throw InputError("'" + name() +
                     "': 'scaled_properties' and 'property_factors' must have the same length.");
}

void
Material::setupPropertyFactors(const MaterialPropertyRegistry & registry)
{
  _property_factors.clear();
  const auto & names = _params.getStringList("scaled_properties");
  const auto & factors = _params.getRealList("property_factors");
  for (std::size_t i = 0; i < names.size(); ++i)
  {
    if (!registry.has(names[i]))
      throw InputError("'" + name() + "': cannot scale '" + names[i] +
                       "', which no material declares.");
    _property_factors.emplace_back(
        registry.id(names[i]), registry.components(names[i]), factors[i]);
  }
}

void
Material::applyPropertyFactors(QpContext & ctx) const
{
  for (const auto & [id, n, f] : _property_factors)
    for (int c = 0; c < n; ++c)
      ctx.property(id, c) *= f;
}

void
Material::initialSetup(Problem & problem)
{
  _blocks.clear();
  for (const auto & b : _params.getStringList("block"))
    _blocks.insert(problem.mesh().blockId(b));
}

int
Material::coupledVariable(Problem & problem, const std::string & param) const
{
  return problem.variableIndex(_params.getString(param));
}

// ---- Factory -------------------------------------------------------------------------------
std::string
categoryName(ObjectCategory c)
{
  switch (c)
  {
  case ObjectCategory::Kernel:
    return "kernel";
  case ObjectCategory::BoundaryCondition:
    return "boundary_condition";
  case ObjectCategory::NodalBC:
    return "nodal_boundary_condition";
  case ObjectCategory::NodalLoad:
    return "nodal_load";
  case ObjectCategory::Material:
    return "material";
  }
  return "Unknown";
}

Factory &
Factory::instance()
{
  static Factory f;
  return f;
}

Factory::Factory()
{
  registerFrameworkObjects(*this);
  registerHeatTransferObjects(*this);
  registerSolidMechanicsObjects(*this);
  registerFluidObjects(*this);
  registerFuelObjects(*this);
  registerParsedObjects(*this);
}

const Factory::Entry &
Factory::entry(const std::string & type) const
{
  auto it = _entries.find(type);
  if (it == _entries.end())
  {
    std::ostringstream os;
    std::vector<std::string> names;
    for (const auto & [n, _] : _entries)
      names.push_back(n);
    // The names of 0.1 were CamelCase (DirichletBC); since 0.2 they are lower
    // case with underscores, keeping the capitals of material and proper
    // names (Dirichlet_boundary_condition, UO2_thermal).  A 0.1 name is
    // recognised by comparing the names without case and underscores.
    const auto key = [](const std::string & n)
    {
      std::string k;
      for (char c : n)
        if (c != '_')
          k += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
      if (k.size() > 2 && k.compare(k.size() - 2, 2, "bc") == 0)
        k = k.substr(0, k.size() - 2) + "boundarycondition";
      return k;
    };
    std::string renamed;
    for (const auto & n : names)
      if (key(n) == key(type))
        renamed = n;
    os << "Unknown object type '" << type << "'.";
    if (!renamed.empty())
      os << " Object types are written in lower case with underscores since dualmesh 0.2, "
            "keeping the capitals of material and proper names: use '"
         << renamed << "'.";
    else
      os << didYouMean(type, names);
    os << " Registered types:";
    for (const auto & n : names)
      os << " " << n;
    throw InputError(os.str());
  }
  return it->second;
}

InputParameters
Factory::validParams(const std::string & type) const
{
  return entry(type).params();
}

ObjectCategory
Factory::category(const std::string & type) const
{
  return entry(type).category;
}

std::string
Factory::module(const std::string & type) const
{
  return entry(type).module;
}

std::shared_ptr<Object>
Factory::create(const std::string & type, const std::string & name, InputParameters params) const
{
  const auto & e = entry(type);
  params.setPrivate("_name", name);
  params.setPrivate("_type", type);
  params.validate(type + " '" + name + "'");
  try
  {
    return e.build(params);
  }
  catch (const InputError & err)
  {
    throw InputError(type + " '" + name + "': " + err.what());
  }
}

std::vector<std::string>
Factory::registeredTypes() const
{
  std::vector<std::string> out;
  for (const auto & [n, _] : _entries)
    out.push_back(n);
  return out;
}

} // namespace dualmesh
