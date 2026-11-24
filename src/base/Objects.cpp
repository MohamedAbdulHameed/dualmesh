// SPDX-License-Identifier: LGPL-2.1-or-later
//
// Base-class implementations: Object, ResidualObject, Kernel, boundary
// conditions, nodal loads, property objects, and the Factory.
#include "dualmesh/base/Factory.h"
#include "dualmesh/base/Kernel.h"
#include "dualmesh/base/Problem.h"
#include "dualmesh/base/Property.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <limits>
#include <map>
#include <sstream>

namespace dualmesh
{

// ---- Object -------------------------------------------------------------------
InputParameters
Object::validParams()
{
  InputParameters p;
  p.addOptional("_name", ParameterKind::String, std::string(""), "The name of the object.");
  p.addOptional(
      "_type", ParameterKind::String, std::string(""), "The registered type of the object.");
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
functionParameter(const Problem & problem,
                  const InputParameters & params,
                  const std::string & param,
                  const std::string & owner)
{
  const auto & raw = params.getRaw(param);
  if (const auto * d = std::get_if<double>(&raw))
    return std::make_shared<ConstantFunction>(*d);
  if (const auto * l = std::get_if<long>(&raw))
    return std::make_shared<ConstantFunction>(double(*l));
  if (const auto * s = std::get_if<std::string>(&raw))
    return problem.function(*s);
  if (const auto * f = std::get_if<std::shared_ptr<Function>>(&raw))
    return *f;
  throw InputError("Parameter '" + param + "' of '" + owner + "' has no value.");
}

FunctionPtr
Object::getFunction(Problem & problem, const std::string & param) const
{
  return functionParameter(problem, _params, param, _name);
}

// ---- ResidualObject ----------------------------------------------------------
InputParameters
ResidualObject::validParams()
{
  InputParameters p = Object::validParams();
  p.addRequired("variable",
                ParameterKind::Names,
                "The variable (equation) this object acts on. A list of variables creates the "
                "object once for each of them, with all its other parameters equal.");
  return p;
}

void
ResidualObject::addIntegrationParams(InputParameters & p)
{
  p.addOptional("block",
                ParameterKind::StringList,
                std::vector<std::string>{},
                "Blocks (subdomains) this object acts on. Default: an empty list, which selects "
                "every block.");
  p.addOptional("quadrature",
                ParameterKind::String,
                std::string("automatic"),
                "Quadrature rule for the integration of this term. The choices are automatic, "
                "gauss1 to gauss20 (Gauss-Legendre with that many points per direction), "
                "midpoint, trapezoid, simpson, nodal (a single point at the owning node, which "
                "lumps the term), interface (a single point at the control domain interface), and "
                "control_domain_trapezoid (the trapezoidal rule over the whole control domain, "
                "which is the source rule of the classical finite volume method). The code also "
                "accepts the aliases auto, trapezoidal, lumped, centroid and cd_trapezoid, reads "
                "a bare gauss as gauss2, and matches the name without regard to case. Raise the "
                "rule for strongly curved elements or a rapidly varying coefficient, and use "
                "nodal to lump a capacity or mass term. Default automatic: Gauss-Legendre with "
                "one point per direction more than the polynomial order of the mesh, two on a "
                "linear mesh and three on a quadratic one, which integrates the product of two "
                "shape functions exactly on an undistorted element.");
  p.addOptional("reduced_integration",
                ParameterKind::Boolean,
                false,
                "Selective reduced integration: evaluate the solution and its gradients at the "
                "centroid of the element, while the rule named by 'quadrature' integrates the "
                "term and the geometry. Turn it on only for the transverse shear terms of a thin "
                "beam or plate and for the penalty term of an incompressible flow, where it "
                "removes locking. On a bending or diffusion term it lowers the accuracy. Default "
                "false, full integration, which keeps the accuracy of every other term.");
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
  // A nodal condition or load has no integral, and so no quadrature rule.
  if (params.has("quadrature"))
    _quad = QuadratureSpec::parse(params.getString("quadrature"),
                                  params.getBool("reduced_integration"));
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
  if (_params.has("block"))
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
  InputParameters p = ResidualObject::validParams();
  addIntegrationParams(p);
  return p;
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
  addIntegrationParams(p);
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
                "The equation on the secondary side that receives the opposite flux. Default: "
                "empty, which selects 'variable', the same equation on both sides of the gap.");
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
  if (params.has("boundary"))
    _boundaries = params.getStringList("boundary");
}

void
NodalBC::initialSetup(Problem & problem)
{
  ResidualObject::initialSetup(problem);
  std::set<Index> ids;
  for (const auto & b : _boundaries)
    for (Index n : problem.mesh().boundaryNodes(b))
      ids.insert(n);
  _nodes.assign(ids.begin(), ids.end());
}

// ---- NodalLoad ------------------------------------------------------------------------
InputParameters
NodalLoad::validParams()
{
  InputParameters p = ResidualObject::validParams();
  p.setClassDescription(
      "Concentrated source at nodes, such as a point force or a point heat source. Every node "
      "it acts on receives the full 'value' as a source of its equation.");
  p.addRequired("value",
                ParameterKind::Function,
                "Magnitude of the concentrated source, in the units conjugate to the variable: "
                "a force in N for a displacement, a heat rate in W for a temperature. It is a "
                "constant, a registered function, or an expression of x, y, z and t. A positive "
                "value acts along the positive direction of the variable, or adds heat to the "
                "body.");
  p.addOptional("boundary",
                ParameterKind::StringList,
                std::vector<std::string>{},
                "Node sets or side sets whose nodes receive the load. Default: none, for a load "
                "given by 'points' only.");
  p.addOptional("points",
                ParameterKind::RealList,
                std::vector<double>{},
                "Coordinates of the loaded points: one number per dimension of the space of the "
                "mesh for each point (two in two dimensions). Each point moves to the nearest "
                "node, however far away that node is, without a warning, so check that the mesh "
                "has a node where the load belongs. Points that move to the same node merge, and "
                "that node receives the load once. Give 'points', 'boundary', or both. Giving "
                "neither is an error. Default: none, for a load given by 'boundary' only.");
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
    for (Index n : problem.mesh().boundaryNodes(b))
      ids.insert(n);
  _boundary_nodes.assign(ids.begin(), ids.end());
  _point_nodes.clear();
  const auto pts = _params.getRealList("points");
  const int dim = problem.mesh().spaceDimension();
  const std::size_t stride = static_cast<std::size_t>(dim);
  if (pts.size() % stride)
    throw InputError("NodalLoad '" + _name + "': 'points' must hold " + std::to_string(stride) +
                     " numbers per point, one per dimension of the mesh.");
  for (std::size_t i = 0; i < pts.size(); i += stride)
  {
    Point p{0.0, 0.0, 0.0};
    for (std::size_t c = 0; c < stride; ++c)
      p[c] = pts[i + c];
    Index best = -1;
    double bd = std::numeric_limits<double>::infinity();
    for (Index n = 0; n < problem.mesh().numNodes(); ++n)
    {
      const double d = norm(problem.mesh().node(n) - p);
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

// ---- Property objects -------------------------------------------------------------------------
int
PropertyRegistry::declare(const std::string & name, int components)
{
  auto it = _props.find(name);
  if (it != _props.end())
  {
    if (it->second.second != components)
      throw InputError("Property '" + name + "' declared with different sizes.");
    return it->second.first;
  }
  const int id = _size;
  _props[name] = {id, components};
  _size += components;
  return id;
}

int
PropertyRegistry::id(const std::string & name) const
{
  auto it = _props.find(name);
  if (it == _props.end())
  {
    std::ostringstream os;
    os << "Property '" << name << "' is not provided by any property object. Available:";
    for (const auto & [n, _] : _props)
      os << " " << n;
    throw InputError(os.str());
  }
  _read.insert(it->second.first);
  return it->second.first;
}

void
PropertyRegistry::markRead(const std::string & name) const
{
  id(name);
}

int
PropertyRegistry::components(const std::string & name) const
{
  id(name);
  return _props.at(name).second;
}

InputParameters
Property::validParams()
{
  InputParameters p = Object::validParams();
  p.addOptional(
      "block",
      ParameterKind::StringList,
      std::vector<std::string>{},
      "Blocks (subdomains) where this property object applies. Default: an empty list, which "
      "selects every block.");
  p.addOptional("property_factors",
                ParameterKind::RealMap,
                RealMap{},
                "Factors on properties that this property object computes, as name -> factor "
                "pairs, for sensitivity and uncertainty studies (for example "
                "{thermal_conductivity: 1.1}). Default: none, which leaves every property "
                "unscaled.");
  return p;
}

Property::Property(const InputParameters & params) : Object(params) {}

void
Property::setupPropertyFactors(const PropertyRegistry & registry)
{
  _property_factors.clear();
  for (const auto & [property, factor] : _params.getRealMap("property_factors"))
  {
    if (!registry.has(property))
      throw InputError("'" + name() + "': cannot scale '" + property +
                       "', which no property object declares.");
    _property_factors.emplace_back(registry.id(property), registry.components(property), factor);
  }
}

void
Property::applyPropertyFactors(QpContext & ctx) const
{
  for (const auto & [id, n, f] : _property_factors)
    for (int c = 0; c < n; ++c)
      ctx.property(id, c) *= f;
}

void
Property::initialSetup(Problem & problem)
{
  _blocks.clear();
  for (const auto & b : _params.getStringList("block"))
    _blocks.insert(problem.mesh().blockId(b));
}

int
Property::coupledVariable(Problem & problem, const std::string & param) const
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
  case ObjectCategory::Property:
    return "property";
  case ObjectCategory::Constraint:
    return "constraint";
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
    os << "Unknown object type '" << type << "'." << didYouMean(type, names);
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
    // The message is prefixed with the type and the name of the object, once: a message that
    // begins with the name alone gets the type before it.
    const std::string prefix = type + " '" + name + "': ";
    const std::string own = "'" + name + "': ";
    const std::string what = err.what();
    if (what.rfind(prefix, 0) == 0)
      throw;
    throw InputError(prefix + (what.rfind(own, 0) == 0 ? what.substr(own.size()) : what));
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

void
addAdvectionSchemeParam(InputParameters & params)
{
  params.addOptional(
      "advection_scheme",
      ParameterKind::String,
      std::string("central"),
      "How the advective term is discretized. 'central' takes the value at the point, with "
      "either method. The other three are face fluxes of the dual mesh control domain method "
      "(DMCDM): 'upwind' takes the upstream value (first order), 'linear_upwind' corrects it by "
      "the upstream gradient (second order), and 'limited' blends it towards the central value "
      "by the van Albada limiter (second order and bounded). The finite element method (FEM) "
      "takes central only. For a streamline stabilization with FEM, add the kernel "
      "heat_streamline_stabilization for a temperature or momentum_stabilization for a flow. "
      "Default central, the one scheme that both methods take.");
}

const Kernel *
massConservationKernel(const Problem & problem)
{
  for (const auto & k : problem.kernels())
    if (k->isMassConservation())
      return k.get();
  return nullptr;
}

const Kernel *
conservingMassEquation(const Problem & problem, int variable)
{
  // By name: the mass equation may come after the kernel being set up, with its variable not yet
  // resolved.
  // Without mixed orders every variable is interpolated alike, whatever order it declares (a
  // pressure declared first on a linear mesh).
  const Kernel * mass = massConservationKernel(problem);
  if (mass && (!problem.hasMixedOrder() ||
               problem.variable(problem.variableIndex(mass->variableName())).order ==
                   problem.variable(variable).order))
    return mass;
  return nullptr;
}

ADVector3
advectingVelocity(const QpContext & ctx, const Kernel * mass, const std::vector<int> & velocities)
{
  ADVector3 w{ADReal(0.0), ADReal(0.0), ADReal(0.0)};
  // The old time of a first step has no step whose mass equation it holds, and carries v.
  const bool first_old_level = ctx.old_level && !ctx.previous_step;
  if (mass && !ctx.on_boundary && !first_old_level && mass->conservedVelocity(ctx, w))
  {
    // The velocity part as a coefficient, which Picard iteration takes from the previous iterate.
    for (std::size_t d = 0; d < velocities.size(); ++d)
      w[d] += ctx.coefficientValue(velocities[d]) - ctx.value(velocities[d]);
    return w;
  }
  for (std::size_t d = 0; d < velocities.size(); ++d)
    w[d] = ctx.coefficientValue(velocities[d]);
  return w;
}

AdvectionScheme
advectionScheme(const Problem & problem, const std::string & text, const std::string & owner)
{
  if (text == "central")
    return AdvectionScheme::Central;
  const bool face = text == "upwind" || text == "linear_upwind" || text == "limited";
  if (!face && text != "SUPG" && text != "supg")
    throw InputError("'" + owner + "': unknown advection scheme '" + text +
                     "'. The schemes are central, upwind, linear_upwind and limited.");
  if (!face || !problem.isDual())
    throw InputError(
        "'" + owner + "': the advection scheme '" + text + "' is not available with this method, " +
        (problem.isDual()
             ? std::string("which takes central, upwind, linear_upwind or limited.")
             : std::string("which takes central. The streamline stabilisation is the kernel "
                           "heat_streamline_stabilization, which heat_transfer adds with "
                           "advection_scheme 'SUPG', and momentum_stabilization for a flow. The "
                           "face fluxes upwind, linear_upwind and limited are those of "
                           "method='DMCDM'.")));
  return text == "upwind"    ? AdvectionScheme::Upwind
         : text == "limited" ? AdvectionScheme::Limited
                             : AdvectionScheme::LinearUpwind;
}

} // namespace dualmesh
