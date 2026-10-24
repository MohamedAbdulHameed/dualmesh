// SPDX-License-Identifier: LGPL-2.1-or-later
//
// Generic framework objects: diffusion,
// reaction, body force, advection, time derivative, standard boundary
// conditions, point sources, and constant and function properties.
#include "dualmesh/modules/Framework.h"
#include "dualmesh/base/Deformation.h"
#include "dualmesh/base/Factory.h"
#include "dualmesh/base/Kernel.h"
#include "dualmesh/base/Problem.h"
#include "dualmesh/base/Property.h"

#include <array>
#include <cmath>
#include <limits>
#include <map>

namespace dualmesh
{

// ---------------------------------------------------------------------------
// Diffusion: F = k(x) p(u) grad u, p(u) = c0 + c1 u + c2 u^2 + ...
// ---------------------------------------------------------------------------
InputParameters
Diffusion::validParams()
{
  InputParameters p = Kernel::validParams();
  p.setClassDescription(
      "Diffusion term -div(k grad u), with k = k(x, t) * (c0 + c1 u + c2 u^2 + ...). The flux is "
      "F = k grad u, and the natural boundary quantity is n . (k grad u).");
  p.addOptional("diffusivity",
                ParameterKind::Function,
                1.0,
                "Diffusivity k: a constant, a registered function of (x, y, z, t), the name of a "
                "property, or an expression of the fields, x, y, z, t, the functions and the "
                "properties. It must be positive. Default 1.");
  p.setRange("diffusivity", ParameterRange::Positive);
  p.addOptional("solution_polynomial",
                ParameterKind::RealList,
                std::vector<double>{1.0},
                "Coefficients c0, c1, ... of a polynomial p(u) = c0 + c1 u + c2 u^2 + ... that "
                "multiplies the diffusivity, giving an effective k(x, t) p(u). The default {1} "
                "leaves the diffusivity unchanged, so a list given here must include its own "
                "constant term. The polynomial is evaluated at the current iterate under Newton's "
                "method and at the previous iterate under direct iteration.");
  return p;
}

Diffusion::Diffusion(const InputParameters & p) : Kernel(p)
{
  _poly = p.getRealList("solution_polynomial");
}

void
Diffusion::initialSetup(Problem & problem)
{
  Kernel::initialSetup(problem);
  _k.setup(problem, _params, "diffusivity", name());
}

ADReal
Diffusion::coefficient(const QpContext & ctx) const
{
  ADReal k = _k.value(ctx);
  if (_poly.size() == 1 && _poly[0] == 1.0)
    return k;
  const ADReal & u = ctx.coefficientValue(_var);
  ADReal poly(0.0);
  for (std::size_t i = _poly.size(); i-- > 0;)
    poly = poly * u + _poly[i];
  return k * poly;
}

void
Diffusion::computeFlux(const QpContext & ctx, ADVector3 & F) const
{
  const ADReal k = coefficient(ctx);
  const auto & g = ctx.gradient(_var);
  for (int d = 0; d < ctx.dim; ++d)
    F[d] = k * g[d];
}

// ---------------------------------------------------------------------------
InputParameters
AnisotropicDiffusion::validParams()
{
  InputParameters p = Kernel::validParams();
  p.setClassDescription(
      "Anisotropic diffusion with a constant tensor K: the flux is F_i = sum_j K_ij du/dx_j "
      "and the source is zero, so the equation is -div(K grad u) = 0 and the natural "
      "boundary quantity is n . (K grad u).");
  p.addRequired("diffusivity_tensor",
                ParameterKind::RealList,
                "Entries of the constant diffusivity tensor K. Give either one value per "
                "dimension, which is read as the diagonal (K_xx, K_yy, K_zz), or dimension "
                "squared values, which are read row by row (K_xx, K_xy, K_yx, K_yy in two "
                "dimensions). Any other count is an error. The tensor is constant in space and "
                "time and is not checked for symmetry or positive definiteness. A non-symmetric K "
                "gives a non-symmetric Jacobian.");
  return p;
}

AnisotropicDiffusion::AnisotropicDiffusion(const InputParameters & p) : Kernel(p)
{
  _entries = p.getRealList("diffusivity_tensor");
}

void
AnisotropicDiffusion::initialSetup(Problem & problem)
{
  Kernel::initialSetup(problem);
  const int dim = problem.mesh().spaceDimension();
  for (auto & row : _K)
    row.fill(0.0);
  if (static_cast<int>(_entries.size()) == dim)
    for (int i = 0; i < dim; ++i)
      _K[i][i] = _entries[i];
  else if (static_cast<int>(_entries.size()) == dim * dim)
    for (int i = 0; i < dim; ++i)
      for (int j = 0; j < dim; ++j)
        _K[i][j] = _entries[i * dim + j];
  else
    throw InputError("'diffusivity_tensor' needs " + std::to_string(dim) + " or " +
                     std::to_string(dim * dim) + " values.");
}

void
AnisotropicDiffusion::computeFlux(const QpContext & ctx, ADVector3 & F) const
{
  const auto & g = ctx.gradient(_var);
  for (int i = 0; i < ctx.dim; ++i)
  {
    ADReal s(0.0);
    for (int j = 0; j < ctx.dim; ++j)
      if (_K[i][j] != 0.0)
        s += g[j] * _K[i][j];
    F[i] = s;
  }
}

// ---------------------------------------------------------------------------
InputParameters
Reaction::validParams()
{
  InputParameters p = Kernel::validParams();
  p.setClassDescription(
      "A reaction term, which in structural problems is the restoring action of an "
      "elastic foundation and in transport problems a decay or growth. It contributes no "
      "flux and the source S = c u^p. With the default exponent 1 the term is linear.");
  p.addOptional("coeff",
                ParameterKind::Function,
                1.0,
                "Coefficient c in the source S = c u^p: a constant, a registered function, the "
                "name of a property, or an expression of the fields, x, y, z, t, the functions "
                "and the properties. The source enters the residual as written, so a positive c "
                "removes u (a decay, or the restoring action of an elastic foundation) and a "
                "negative c produces it. A coefficient that depends on the solution makes the "
                "term nonlinear, and its derivative enters the Jacobian through the automatic "
                "differentiation. Default 1.");
  p.addOptional("exponent",
                ParameterKind::Real,
                1.0,
                "Exponent p in the source S = c u^p. The default 1 gives a linear reaction. Any "
                "other value makes the term nonlinear. Direct iteration then linearizes it as c "
                "u_previous^(p-1) u, and a non-integer p requires u to stay positive.");
  return p;
}

Reaction::Reaction(const InputParameters & p) : Kernel(p), _p(p.getReal("exponent")) {}

void
Reaction::initialSetup(Problem & problem)
{
  Kernel::initialSetup(problem);
  _c.setup(problem, _params, "coeff", name());
}

ADReal
Reaction::computeSource(const QpContext & ctx) const
{
  const ADReal c = _c.value(ctx);
  const ADReal & u = ctx.value(_var);
  if (_p == 1.0)
    return c * u;
  if (ctx.mode == LinearizationMode::Picard)
    return c * pow(ctx.coefficientValue(_var), _p - 1.0) * u;
  return c * pow(u, _p);
}

// ---------------------------------------------------------------------------
InputParameters
BodyForce::validParams()
{
  InputParameters p = Kernel::validParams();
  p.setClassDescription(
      "The source term of any equation, per unit volume: the heat source (volumetric heat "
      "generation, W/m^3) of an equation for a temperature, the body force (N/m^3) of an "
      "equation for a displacement or a velocity, and the production rate of an equation for a "
      "concentration. It contributes no flux and the source S = -f, so a positive value drives "
      "the variable up (adds heat, pushes along the positive axis). The heat_transfer physics "
      "adds it for its heat_source.");
  p.addOptional("value",
                ParameterKind::Function,
                1.0,
                "The source f per unit volume, as a constant, a registered function of (x, y, z, "
                "t), or an expression of x, y, z and t: for a temperature the heat source in "
                "W/m^3. A positive value drives the variable up.");
  p.addOptional("scale_with_load",
                ParameterKind::Boolean,
                true,
                "Multiply this body force by the load factor during load stepping. This parameter "
                "defaults to true, unlike the framework default, because the applied loading is "
                "normally the quantity that is ramped. Set it to false for a part of the loading "
                "that must stay fixed while the rest is increased.");
  return p;
}

BodyForce::BodyForce(const InputParameters & p) : Kernel(p) {}

void
BodyForce::initialSetup(Problem & problem)
{
  Kernel::initialSetup(problem);
  _f = getFunction(problem, "value");
}

ADReal
BodyForce::computeSource(const QpContext & ctx) const
{
  return ADReal(-_f->value(ctx.x, ctx.time));
}

// ---------------------------------------------------------------------------
InputParameters
TimeDerivative::validParams()
{
  InputParameters p = Kernel::validParams();
  p.setClassDescription(
      "The transient term of the equation. It contributes no flux and the source "
      "S = c (u - u_old) / dt. It is a time kernel: it is skipped entirely in a steady "
      "solve, and it is not multiplied by the time-integration weight theta, which applies "
      "to the steady terms only.");
  p.addOptional("coeff",
                ParameterKind::Function,
                1.0,
                "Capacity coefficient c multiplying the time derivative: a constant, a registered "
                "function of (x, y, z, t), the name of a property, or an expression of the fields. "
                "For heat conduction it is the volumetric heat capacity rho c_p, for a mass "
                "balance the porosity. Default 1.");
  return p;
}

TimeDerivative::TimeDerivative(const InputParameters & p) : Kernel(p) {}

void
TimeDerivative::initialSetup(Problem & problem)
{
  Kernel::initialSetup(problem);
  _c.setup(problem, _params, "coeff", name());
}

ADReal
TimeDerivative::computeSource(const QpContext & ctx) const
{
  return _c.value(ctx) * (ctx.value(_var) - ctx.oldValue(_var)) / ctx.dt;
}

// ---------------------------------------------------------------------------
InputParameters
Advection::validParams()
{
  InputParameters p = Kernel::validParams();
  p.setClassDescription(
      "Advection by a prescribed velocity field v. The 'conservative' form uses the flux F = -v u "
      "(so that -div F = div(v u)), and the 'non_conservative' form uses the source S = v . grad "
      "u.");
  p.addOptional("velocity",
                ParameterKind::StringList,
                std::vector<std::string>{"0", "0", "0"},
                "The velocity components (vx, vy, vz), each a number, a registered function or an "
                "expression of x, y, z, t and the fields. Missing trailing components are zero. "
                "Default zero.");
  p.addOptional("advection_form",
                ParameterKind::String,
                std::string("non_conservative"),
                "Which of the two equivalent statements of the advection term to assemble. "
                "'conservative' contributes the flux F = -v u, so the discrete term conserves u "
                "exactly over the control domains. Use it when the velocity is divergence free or "
                "when a conservation check matters. 'non_conservative', the default, contributes "
                "the source S = v . grad u, which requires less computation and is the correct "
                "reading when the equation is written in non-conservative form. With an "
                "advection scheme other than central both are face fluxes: m u_f through each "
                "face, or m (u_f - u_C) for the control volume C on each side, with m the "
                "normal velocity times the face area. Any other string is an error.");
  addAdvectionSchemeParam(p);
  return p;
}

Advection::Advection(const InputParameters & p) : Kernel(p)
{
  const auto form = p.getString("advection_form");
  const auto scheme = p.getString("advection_scheme");
  _face_flux = scheme == "upwind" || scheme == "linear_upwind" || scheme == "limited";
  if (form == "conservative")
    _conservative = true;
  else if (form != "non_conservative")
    throw InputError("'" + name() +
                     "': advection_form must be 'conservative' or 'non_conservative'.");
}

void
Advection::initialSetup(Problem & problem)
{
  Kernel::initialSetup(problem);
  auto v = _params.getStringList("velocity");
  if (v.size() > 3)
    throw InputError("'" + name() + "': the velocity has at most three components.");
  v.resize(3, "0");
  for (int d = 0; d < 3; ++d)
    _v[d].setup(problem, v[d], name() + "', velocity component " + std::to_string(d) + " '");
  _scheme = advectionScheme(problem, _params.getString("advection_scheme"), name());
}

bool
Advection::advectingNormalVelocity(const QpContext & ctx, double & velocity) const
{
  if (!_face_flux)
    return false;
  velocity = 0.0;
  for (int d = 0; d < ctx.dim; ++d)
    velocity += _v[d].value(ctx).value() * ctx.normal[d];
  return true;
}

void
Advection::computeFlux(const QpContext & ctx, ADVector3 & F) const
{
  double velocity = 0.0;
  advectingNormalVelocity(ctx, velocity);
  const ADReal u = ctx.advectedValue(_var, velocity, _scheme) - ctx.referenceValue(_var);
  for (int d = 0; d < ctx.dim; ++d)
    F[d] = -_v[d].value(ctx) * u;
}

ADReal
Advection::computeSource(const QpContext & ctx) const
{
  const auto & g = ctx.gradient(_var);
  ADReal s(0.0);
  for (int d = 0; d < ctx.dim; ++d)
    s += _v[d].value(ctx) * g[d];
  return s;
}

// ---------------------------------------------------------------------------
InputParameters
CoupledForce::validParams()
{
  InputParameters p = Kernel::validParams();
  p.setClassDescription(
      "A source proportional to another variable, which is how two equations of a "
      "multiphysics problem are coupled. It contributes no flux and the source S = -c v. "
      "The coupling is exact in Newton's method, because the derivative with respect to "
      "the coupled variable is carried through the automatic differentiation.");
  p.addRequired("coupled_variable",
                ParameterKind::String,
                "Name of the other variable that drives this equation. It must already "
                "exist on the problem.");
  p.addOptional("coeff",
                ParameterKind::Function,
                1.0,
                "Coefficient c in the source S = -c v: a constant, a registered function, the "
                "name of a property (for a coefficient that differs between regions), or an "
                "expression of the fields, x, y, z, t, the functions and the properties. A "
                "positive c makes the coupled variable v a source for this equation, and a "
                "negative c makes it a sink. Default 1.");
  return p;
}

CoupledForce::CoupledForce(const InputParameters & p) : Kernel(p) {}

void
CoupledForce::initialSetup(Problem & problem)
{
  Kernel::initialSetup(problem);
  _v = coupledVariable(problem, "coupled_variable");
  _c.setup(problem, _params, "coeff", name());
}

ADReal
CoupledForce::computeSource(const QpContext & ctx) const
{
  return -_c.value(ctx) * ctx.value(_v);
}

// ---------------------------------------------------------------------------
// Boundary conditions
// ---------------------------------------------------------------------------
InputParameters
DirichletBC::validParams()
{
  InputParameters p = NodalBC::validParams();
  p.setClassDescription("Prescribed value u = g(x, t) (essential boundary condition).");
  p.addOptional("value",
                ParameterKind::Function,
                0.0,
                "Prescribed value g(x, t), written directly into the solution at every node of "
                "the boundary. When the inherited 'scale_with_load' is true the value is "
                "multiplied by the current load factor, so a prescribed displacement is ramped "
                "along with the loads. The parameter is false by default, so the value is held "
                "fixed across the load steps.");
  return p;
}

DirichletBC::DirichletBC(const InputParameters & p) : NodalBC(p) {}

void
DirichletBC::initialSetup(Problem & problem)
{
  NodalBC::initialSetup(problem);
  _g = getFunction(problem, "value");
}

double
DirichletBC::computeValue(const Point & x, double t) const
{
  return _g->value(x, t);
}

InputParameters
PointDirichletBC::validParams()
{
  InputParameters p = NodalBC::validParams();
  p.setClassDescription(
      "Prescribed value u = g(x, t) at the single entity (node, or for the cell-centred method "
      "cell or boundary face) nearest to a point. For a first-order variable on a quadratic mesh "
      "the entity is the nearest corner node. Its main use is to fix the level of the pressure of "
      "an enclosed incompressible flow, which the equations determine only up to a constant.");
  p.remove("boundary");
  p.addRequired("point",
                ParameterKind::RealList,
                "The coordinates (x, y, z) of the point. Missing trailing coordinates are zero.");
  p.addOptional("value", ParameterKind::Function, 0.0, "Prescribed value g(x, t).");
  return p;
}

PointDirichletBC::PointDirichletBC(const InputParameters & p) : NodalBC(p) {}

void
PointDirichletBC::initialSetup(Problem & problem)
{
  NodalBC::initialSetup(problem);
  _g = getFunction(problem, "value");
  auto c = _params.getRealList("point");
  c.resize(3, 0.0);
  const Point x{c[0], c[1], c[2]};
  // A first-order variable on a quadratic mesh lives on the corner nodes only,
  // so the nearest corner node is the one to fix.
  std::vector<char> carries;
  if (problem.hasMixedOrder() && !problem.isCellCentered())
  {
    carries.assign(problem.numEntities(), 0);
    for (const auto & el : problem.mesh().elements())
      for (int a = 0; a < problem.numLocalNodes(el, variable()); ++a)
        carries[el.nodes[a]] = 1;
  }
  Index best = -1;
  double best_distance = std::numeric_limits<double>::infinity();
  for (Index i = 0; i < problem.numEntities(); ++i)
  {
    if ((!carries.empty() && !carries[i]) || !problem.integratedEntity(i))
      continue;
    const double d = norm(problem.entityPoint(i) - x);
    if (d < best_distance)
    {
      best_distance = d;
      best = i;
    }
  }
  _nodes.clear();
  if (best >= 0)
    _nodes.push_back(best);
  _distance = best_distance;
}

double
PointDirichletBC::computeValue(const Point & x, double t) const
{
  return _g->value(x, t);
}

InputParameters
NeumannBC::validParams()
{
  InputParameters p = IntegratedBC::validParams();
  p.setClassDescription(
      "Prescribed natural boundary quantity q = n . F, where F is the flux assembled by "
      "the kernels of this variable. This is the secondary variable of the duality pair, "
      "and it is the condition that applies by default on a boundary that carries no "
      "condition at all, with q = 0.");
  p.addOptional(
      "flux",
      ParameterKind::Function,
      0.0,
      "Prescribed value of n . F. For a diffusion-type kernel F = k grad u, so a positive value "
      "is an inflow of u into the body: for heat transfer, the heat flux entering the body. The "
      "default 0 is the do-nothing condition, insulated for heat transfer and traction free for "
      "elasticity. "
      "It is a constant, a registered function, or an expression of the fields, their gradients (grad_x(u), grad_y(u), grad_z(u)), x, y, z and t, which is differentiated automatically.");
  p.addOptional("scale_with_load",
                ParameterKind::Boolean,
                true,
                "Multiply this contribution by the load factor during load stepping. This "
                "parameter defaults to true, unlike the framework default, because the applied "
                "loading is normally the quantity that is ramped. Set it to false for a part of "
                "the loading that must stay fixed while the rest is increased.");
  deformation::addParameter(p,
                            "the flux is per unit deformed area and is multiplied by the area "
                            "ratio da/dA = |J F^-T N| to give the flux per unit undeformed area.");
  return p;
}

NeumannBC::NeumannBC(const InputParameters & p) : IntegratedBC(p) {}

void
NeumannBC::initialSetup(Problem & problem)
{
  IntegratedBC::initialSetup(problem);
  _q.setup(problem, _params, "flux", name());
  _F = deformation::propertyId(problem, _params, name());
}

ADReal
NeumannBC::computeBoundaryFlux(const QpContext & ctx) const
{
  const ADReal q = _q.value(ctx);
  return _F >= 0 ? q * deformation::areaRatio(ctx, _F) : q;
}

InputParameters
RobinBC::validParams()
{
  InputParameters p = IntegratedBC::validParams();
  p.setClassDescription(
      "Mixed (Robin, convection) condition n . F = q0 - h (u - u_ambient), e.g. Newton's law "
      "of cooling.");
  p.addOptional(
      "transfer_coeff",
      ParameterKind::Function,
      0.0,
      "Transfer, or film, coefficient h in n . F = q0 - h (u - u_ambient). It "
      "should be non-negative: a positive h drives u towards 'ambient_value', "
      "whereas a negative one drives it away and makes the problem unstable. The "
      "default 0 switches the transfer term off and leaves a pure Neumann "
      "condition of strength 'flux'. "
      "Like 'flux' and 'ambient_value', it is a constant, a registered function, or an expression of the fields, their gradients, x, y, z and t.");
  p.setRange("transfer_coeff", ParameterRange::NonNegative);
  p.addOptional("ambient_value",
                ParameterKind::Function,
                0.0,
                "Far-field value u_ambient that the transfer term drives u towards. It is used "
                "only when 'transfer_coeff' is non-zero. The default is 0, independent of "
                "the initial condition, so leaving it unset cools the surface towards zero.");
  p.addOptional(
      "flux",
      ParameterKind::Function,
      0.0,
      "Prescribed part q0 of the natural quantity n . F, added to the transfer term. It carries "
      "the same sign convention as Neumann: a positive value drives u into the body.");
  deformation::addParameter(p,
                            "the flux is per unit deformed area and is multiplied by the area "
                            "ratio da/dA = |J F^-T N| to give the flux per unit undeformed area.");
  return p;
}

RobinBC::RobinBC(const InputParameters & p) : IntegratedBC(p) {}

void
RobinBC::initialSetup(Problem & problem)
{
  IntegratedBC::initialSetup(problem);
  _h.setup(problem, _params, "transfer_coeff", name());
  _uinf.setup(problem, _params, "ambient_value", name());
  _q.setup(problem, _params, "flux", name());
  _F = deformation::propertyId(problem, _params, name());
}

ADReal
RobinBC::computeBoundaryFlux(const QpContext & ctx) const
{
  const ADReal q = _q.value(ctx) - _h.value(ctx) * (ctx.value(_var) - _uinf.value(ctx));
  return _F >= 0 ? q * deformation::areaRatio(ctx, _F) : q;
}

// ---------------------------------------------------------------------------
// Property objects
// ---------------------------------------------------------------------------
InputParameters
ConstantProperty::validParams()
{
  InputParameters p = Property::validParams();
  p.setClassDescription(
      "Declares scalar properties that are constant in space and time. The "
      "properties are then available, by their names, to every kernel, boundary condition, "
      "property object and expression on the same blocks: a coefficient given as the name of "
      "a property, such as diffusivity = 'thermal_conductivity', reads it.");
  p.addRequired("properties",
                ParameterKind::RealMap,
                "The declared properties: name -> value, for example "
                "{thermal_conductivity: 400.0}. The units are those that the objects reading them "
                "expect, and they are not checked.");
  return p;
}

ConstantProperty::ConstantProperty(const InputParameters & p) : Property(p)
{
  _values = p.getRealMap("properties");
  if (_values.empty())
    throw InputError("'" + name() + "': give at least one property in 'properties'.");
}

void
ConstantProperty::declareProperties(PropertyRegistry & r)
{
  _ids.clear();
  for (const auto & [n, value] : _values)
  {
    (void) value;
    _ids.push_back(r.declare(n, 1));
  }
}

void
ConstantProperty::computeProperties(QpContext & ctx) const
{
  std::size_t i = 0;
  for (const auto & [n, value] : _values)
  {
    (void) n;
    ctx.property(_ids[i++]) = ADReal(value);
  }
}

namespace
{
/// A property object that computes nothing of its own and only scales properties
/// that the property objects added before it computed.
class PropertyScaling : public Property
{
public:
  static InputParameters validParams()
  {
    InputParameters p = Property::validParams();
    p.setClassDescription(
        "Multiplies properties computed by the property objects added before it, on its blocks, "
        "by constant factors, given by 'property_factors'. It is intended for sensitivity and "
        "uncertainty studies, in which a correlation is scaled without a change to the property "
        "object that computes it.");
    return p;
  }
  explicit PropertyScaling(const InputParameters & p) : Property(p)
  {
    if (p.getRealMap("property_factors").empty())
      throw InputError("'" + name() + "': give at least one factor in 'property_factors'.");
  }
  void declareProperties(PropertyRegistry &) override {}
  void computeProperties(QpContext &) const override {}
};
} // namespace

InputParameters
FunctionProperty::validParams()
{
  InputParameters p = Property::validParams();
  p.setClassDescription(
      "Declares scalar properties whose values are functions of position and time. Use this "
      "for a property that varies through the body, such as a graded conductivity. A property "
      "that depends on the solution itself requires a parsed property, or a property object "
      "written in Python or C++.");
  p.addRequired("properties",
                ParameterKind::StringMap,
                "The declared properties: name -> a registered function or an expression of x, "
                "y, z and t, for example {thermal_conductivity: '20 + 5*x'}. Each is evaluated at "
                "every integration point, so the property may vary in space and time, "
                "independently of the solution.");
  return p;
}

FunctionProperty::FunctionProperty(const InputParameters & p) : Property(p)
{
  _functions = p.getStringMap("properties");
  if (_functions.empty())
    throw InputError("'" + name() + "': give at least one property in 'properties'.");
}

void
FunctionProperty::initialSetup(Problem & problem)
{
  Property::initialSetup(problem);
  _f.clear();
  for (const auto & [n, text] : _functions)
  {
    (void) n;
    _f.push_back(problem.function(text));
  }
}

void
FunctionProperty::declareProperties(PropertyRegistry & r)
{
  _ids.clear();
  for (const auto & [n, text] : _functions)
  {
    (void) text;
    _ids.push_back(r.declare(n, 1));
  }
}

void
FunctionProperty::computeProperties(QpContext & ctx) const
{
  for (std::size_t i = 0; i < _ids.size(); ++i)
    ctx.property(_ids[i]) = ADReal(_f[i]->value(ctx.x, ctx.time));
}

// ---------------------------------------------------------------------------
// ---- periodic boundary condition --------------------------------------------------------
InputParameters
PeriodicBC::validParams()
{
  InputParameters p = Constraint::validParams();
  p.setClassDescription(
      "Periodic boundary condition: the field on the secondary boundary is the field on the "
      "primary boundary, carried over by a translation. Every node of the secondary boundary "
      "is joined to the node of the primary boundary at the translated position, and the two "
      "act as one unknown whose equation collects both, so that its control domain wraps around "
      "the domain. A node that is periodic in two directions (a corner) is joined to one primary. "
      "The meshes of the two boundaries must match node by node after the translation. Works "
      "with fem, hfvm and dmcdm.");
  p.addRequired("primary", ParameterKind::String, "The side set or node set whose nodes are kept.");
  p.addRequired("secondary",
                ParameterKind::String,
                "The side set or node set whose nodes become copies of the primary nodes.");
  p.addOptional("variable",
                ParameterKind::Names,
                std::vector<std::string>{},
                "The variable or variables that are periodic. Default: every variable of the "
                "problem.");
  p.addOptional("translation",
                ParameterKind::RealList,
                std::vector<double>{},
                "The vector from a primary node to its secondary node (x, y and z components). "
                "Default: the vector between the centroids of the two boundaries.");
  return p;
}

PeriodicBC::PeriodicBC(const InputParameters & p) : Constraint(p) {}

std::vector<int>
PeriodicBC::periodicVariables(const Problem & problem) const
{
  std::vector<int> vars;
  const auto names = _params.getStringList("variable");
  if (names.empty())
    for (int v = 0; v < problem.numVariables(); ++v)
      vars.push_back(v);
  else
    for (const auto & v : names)
      vars.push_back(problem.variableIndex(v));
  return vars;
}

std::vector<Index>
PeriodicBC::match(const std::vector<Point> & primary, const std::vector<Point> & secondary) const
{
  if (primary.size() != secondary.size())
    throw InputError("'" + name() + "': the primary boundary has " +
                     std::to_string(primary.size()) + " nodes and the secondary " +
                     std::to_string(secondary.size()) +
                     "; periodic boundaries must match node by node.");
  std::vector<Index> out;
  if (primary.empty())
    return out;
  const auto centroid = [](const std::vector<Point> & points)
  {
    Point c{0, 0, 0};
    for (const Point & x : points)
      c = c + x;
    return (1.0 / static_cast<double>(points.size())) * c;
  };
  Point shift = centroid(secondary) - centroid(primary);
  const auto given = _params.getRealList("translation");
  if (!given.empty())
  {
    if (given.size() > 3)
      throw InputError("'" + name() + "': translation has at most three components.");
    shift = {0, 0, 0};
    for (std::size_t k = 0; k < given.size(); ++k)
      shift[k] = given[k];
  }
  // A node matches when it lies within a millionth of the size of the boundary of the translated position.
  Point lo = primary.front(), hi = lo;
  for (const Point & x : primary)
    for (int k = 0; k < 3; ++k)
    {
      lo[k] = std::min(lo[k], x[k]);
      hi[k] = std::max(hi[k], x[k]);
    }
  const double size = std::max(std::sqrt(dot(hi - lo, hi - lo)), std::sqrt(dot(shift, shift)));
  const double tolerance = 1e-6 * (size > 0 ? size : 1.0);
  // The primary points in a grid of cells of the tolerance; a match lies in the cell of the translated position or in a neighboring one.
  using Key = std::array<long long, 3>;
  std::map<Key, std::vector<Index>> grid;
  const auto key = [&](const Point & x)
  {
    return Key{std::llround(x[0] / tolerance),
               std::llround(x[1] / tolerance),
               std::llround(x[2] / tolerance)};
  };
  for (std::size_t i = 0; i < primary.size(); ++i)
    grid[key(primary[i])].push_back(static_cast<Index>(i));
  out.reserve(secondary.size());
  for (const Point & x : secondary)
  {
    const Point target = x - shift;
    const Key k = key(target);
    Index found = -1;
    for (long long i = -1; i <= 1 && found < 0; ++i)
      for (long long j = -1; j <= 1 && found < 0; ++j)
        for (long long l = -1; l <= 1 && found < 0; ++l)
        {
          const auto it = grid.find({k[0] + i, k[1] + j, k[2] + l});
          if (it == grid.end())
            continue;
          for (Index n : it->second)
          {
            const Point d = primary[n] - target;
            if (std::sqrt(dot(d, d)) <= tolerance)
              found = n;
          }
        }
    if (found < 0)
      throw InputError("'" + name() + "': the secondary node at (" + std::to_string(x[0]) + ", " +
                       std::to_string(x[1]) + ", " + std::to_string(x[2]) +
                       ") has no primary node at the translated position; the meshes of the two "
                       "boundaries must match node by node.");
    out.push_back(found);
  }
  return out;
}

void
PeriodicBC::initialSetup(Problem & problem)
{
  Constraint::initialSetup(problem);
  _pairs.clear();
  const auto primary = problem.boundaryEntities(primaryBoundary());
  const auto secondary = problem.boundaryEntities(secondaryBoundary());
  std::vector<Point> primary_points, secondary_points;
  for (Index n : primary)
    primary_points.push_back(problem.entityPoint(n));
  for (Index n : secondary)
    secondary_points.push_back(problem.entityPoint(n));
  const auto matched = match(primary_points, secondary_points);
  const auto vars = periodicVariables(problem);
  for (std::size_t k = 0; k < secondary.size(); ++k)
    for (int v : vars)
      _pairs.push_back({problem.dof(secondary[k], v), problem.dof(primary[matched[k]], v)});
}

void
registerFrameworkObjects(Factory & f)
{
  const std::string m = "framework";
  f.add<Diffusion>("diffusion", ObjectCategory::Kernel, m);
  f.add<AnisotropicDiffusion>("anisotropic_diffusion", ObjectCategory::Kernel, m);
  f.add<Reaction>("reaction", ObjectCategory::Kernel, m);
  f.add<BodyForce>("body_force", ObjectCategory::Kernel, m);
  f.add<TimeDerivative>("time_derivative", ObjectCategory::Kernel, m);
  f.add<Advection>("advection", ObjectCategory::Kernel, m);
  f.add<CoupledForce>("coupled_force", ObjectCategory::Kernel, m);
  f.add<PointDirichletBC>("point_Dirichlet", ObjectCategory::NodalBC, m);
  f.add<DirichletBC>("Dirichlet", ObjectCategory::NodalBC, m);
  f.add<NeumannBC>("Neumann", ObjectCategory::BoundaryCondition, m);
  f.add<RobinBC>("Robin", ObjectCategory::BoundaryCondition, m);
  f.add<NodalLoad>("point_source", ObjectCategory::NodalLoad, m);
  f.add<PeriodicBC>("periodic", ObjectCategory::Constraint, m);
  f.add<ConstantProperty>("constant", ObjectCategory::Property, m);
  f.add<PropertyScaling>("property_scaling", ObjectCategory::Property, m);
  f.add<FunctionProperty>("function", ObjectCategory::Property, m);
}

} // namespace dualmesh
