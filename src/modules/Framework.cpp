// SPDX-License-Identifier: LGPL-2.1-or-later
//
// Generic framework objects (MOOSE "framework" equivalents): diffusion,
// reaction, body force, advection, time derivative, standard boundary
// conditions, point sources, and generic materials.
#include "dualmesh/modules/Framework.h"
#include "dualmesh/base/Factory.h"
#include "dualmesh/base/Kernel.h"
#include "dualmesh/base/Material.h"
#include "dualmesh/base/Problem.h"

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
      "Diffusion term -div(k grad u), with k = k(x, t) * (c0 + c1 u + c2 u^2 + ...). "
      "Flux F = k grad u; the natural boundary quantity is n . (k grad u).");
  p.addOptional("diffusivity",
                ParameterKind::Function,
                1.0,
                "Diffusivity k, given as a constant or as the name of a registered function "
                "of (x, y, z, t). It must be positive. It is ignored when "
                "'diffusivity_property' names a material property.");
  p.addOptional("diffusivity_property",
                ParameterKind::String,
                std::string(""),
                "Name of a material property to use as the diffusivity (overrides "
                "'diffusivity').");
  p.addOptional("solution_polynomial",
                ParameterKind::RealList,
                std::vector<double>{1.0},
                "Coefficients c0, c1, ... of a polynomial p(u) = c0 + c1 u + c2 u^2 + ... "
                "that multiplies the diffusivity, giving an effective k(x, t) p(u). It "
                "multiplies, it does not replace: the default {1} leaves the diffusivity "
                "unchanged, so a list given here must include its own constant term. The "
                "polynomial is evaluated at the current iterate under Newton's method and at "
                "the previous iterate under direct iteration.");
  return p;
}

Diffusion::Diffusion(const InputParameters & p) : Kernel(p)
{
  _poly = p.getRealList("solution_polynomial");
  _prop_name = p.getString("diffusivity_property");
}

void
Diffusion::initialSetup(Problem & problem)
{
  Kernel::initialSetup(problem);
  _k = getFunction(problem, "diffusivity");
  _prop = _prop_name.empty() ? -1 : problem.propertyRegistry().id(_prop_name);
}

ADReal
Diffusion::coefficient(const QpContext & ctx) const
{
  ADReal k = _prop >= 0 ? ctx.property(_prop) : ADReal(_k->value(ctx.x, ctx.time));
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
                "dimensions). Any other count is an error. The tensor is constant in space "
                "and time and is not checked for symmetry or positive definiteness; a "
                "non-symmetric K gives a non-symmetric Jacobian.");
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
  const int dim = problem.mesh().dimension();
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
  p.addOptional("coefficient",
                ParameterKind::Function,
                1.0,
                "Coefficient c in the source S = c u^p, as a constant or the name of a "
                "function. The source enters the residual as written, so a positive c "
                "removes u (a decay, or the restoring action of an elastic foundation) and "
                "a negative c feeds it.");
  p.addOptional("exponent",
                ParameterKind::Real,
                1.0,
                "Exponent p in the source S = c u^p. The default 1 gives a linear reaction. "
                "Any other value makes the term nonlinear; direct iteration then linearizes "
                "it as c u_previous^(p-1) u, and a non-integer p requires u to stay "
                "positive.");
  return p;
}

Reaction::Reaction(const InputParameters & p) : Kernel(p), _p(p.getReal("exponent")) {}

void
Reaction::initialSetup(Problem & problem)
{
  Kernel::initialSetup(problem);
  _c = getFunction(problem, "coefficient");
}

ADReal
Reaction::computeSource(const QpContext & ctx) const
{
  const double c = _c->value(ctx.x, ctx.time);
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
      "A distributed source on the right-hand side of the equation. It contributes no "
      "flux and the source S = -f, so a positive intensity drives the variable up. Use it "
      "for a body force, an internal generation rate, or any forcing that depends on "
      "position and time but not on the solution.");
  p.addOptional("value",
                ParameterKind::Function,
                1.0,
                "Intensity f of the source, per unit volume, as a constant or the name of "
                "a registered function of (x, y, z, t). A positive value drives the "
                "variable up.");
  p.addOptional("scale_with_load",
                ParameterKind::Boolean,
                true,
                "Multiply this body force by the load factor during load stepping. Unlike the "
                "framework default this is true, because applied loading is normally what "
                "is ramped; set it to false for a part of the loading that must stay fixed "
                "while the rest is increased.");
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
  p.addOptional("coefficient",
                ParameterKind::Function,
                1.0,
                "Capacity coefficient c multiplying the time derivative, as a constant or the name "
                "of a function. For heat conduction it is the volumetric heat capacity rho c_p, "
                "for a mass balance the porosity, and for a wave problem the density.");
  return p;
}

TimeDerivative::TimeDerivative(const InputParameters & p) : Kernel(p) {}

void
TimeDerivative::initialSetup(Problem & problem)
{
  Kernel::initialSetup(problem);
  _c = getFunction(problem, "coefficient");
}

ADReal
TimeDerivative::computeSource(const QpContext & ctx) const
{
  const double c = _c->value(ctx.x, ctx.time);
  return c * (ctx.value(_var) - ctx.oldValue(_var)) / ctx.dt;
}

// ---------------------------------------------------------------------------
InputParameters
Advection::validParams()
{
  InputParameters p = Kernel::validParams();
  p.setClassDescription(
      "Advection by a prescribed velocity field v. The 'conservative' form uses the flux "
      "F = -v u (so that -div F = div(v u)); the 'non_conservative' form uses the source "
      "S = v . grad u.");
  p.addOptional("velocity",
                ParameterKind::RealList,
                std::vector<double>{0.0, 0.0, 0.0},
                "Constant velocity components (vx, vy, vz).");
  p.addOptional("velocity_functions",
                ParameterKind::StringList,
                std::vector<std::string>{},
                "Names of registered functions giving the velocity components. The list is "
                "matched by position and may be shorter than the dimension: a component uses "
                "its function when one is listed at that position and otherwise falls back "
                "to the corresponding entry of 'velocity', so a list with one name overrides "
                "only the x component.");
  p.addOptional("form",
                ParameterKind::String,
                std::string("non_conservative"),
                "Which of the two equivalent statements of the advection term to "
                "assemble. 'conservative' contributes the flux F = -v u, so the discrete "
                "term conserves u exactly over the control domains; use it when the velocity "
                "is divergence free or when a conservation check matters. "
                "'non_conservative', the default, contributes the source S = v . grad u, "
                "which is cheaper and is the correct reading when the equation is written in "
                "non-conservative form. Any other string is an error.");
  return p;
}

Advection::Advection(const InputParameters & p) : Kernel(p)
{
  const auto form = p.getString("form");
  if (form == "conservative")
    _conservative = true;
  else if (form == "non_conservative")
    _conservative = false;
  else
    throw InputError("Advection form must be 'conservative' or 'non_conservative'.");
}

void
Advection::initialSetup(Problem & problem)
{
  Kernel::initialSetup(problem);
  auto v = _params.getRealList("velocity");
  v.resize(3, 0.0);
  const auto names = _params.getStringList("velocity_functions");
  for (int d = 0; d < 3; ++d)
  {
    if (d < static_cast<int>(names.size()))
      _v[d] = problem.function(names[d]);
    else
      _v[d] = std::make_shared<ConstantFunction>(v[d]);
  }
}

void
Advection::computeFlux(const QpContext & ctx, ADVector3 & F) const
{
  const ADReal & u = ctx.value(_var);
  for (int d = 0; d < ctx.dim; ++d)
    F[d] = -_v[d]->value(ctx.x, ctx.time) * u;
}

ADReal
Advection::computeSource(const QpContext & ctx) const
{
  const auto & g = ctx.gradient(_var);
  ADReal s(0.0);
  for (int d = 0; d < ctx.dim; ++d)
    s += _v[d]->value(ctx.x, ctx.time) * g[d];
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
  p.addOptional("coefficient",
                ParameterKind::Function,
                1.0,
                "Coefficient c in the source S = -c v, as a constant or the name of a "
                "function. A positive c makes the coupled variable v a source for this "
                "equation; reverse the sign to make it a sink.");
  return p;
}

CoupledForce::CoupledForce(const InputParameters & p) : Kernel(p) {}

void
CoupledForce::initialSetup(Problem & problem)
{
  Kernel::initialSetup(problem);
  _v = coupledVariable(problem, "coupled_variable");
  _c = getFunction(problem, "coefficient");
}

ADReal
CoupledForce::computeSource(const QpContext & ctx) const
{
  return -_c->value(ctx.x, ctx.time) * ctx.value(_v);
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
                "Prescribed value g(x, t), written directly into the solution at every node "
                "of the boundary. When the inherited 'scale_with_load' is true the value is "
                "multiplied by the current load factor, so a prescribed displacement is "
                "ramped along with the loads; it is false by default, so the value is held "
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
NeumannBC::validParams()
{
  InputParameters p = IntegratedBC::validParams();
  p.setClassDescription(
      "Prescribed natural boundary quantity q = n . F, where F is the flux assembled by "
      "the kernels of this variable. This is the secondary variable of the duality pair, "
      "and it is the condition that applies by default on a boundary that carries no "
      "condition at all, with q = 0.");
  p.addOptional("flux",
                ParameterKind::Function,
                0.0,
                "Prescribed value of n . F. For a diffusion-type kernel F = k grad u, so a "
                "positive value drives u into the body: it is an inflow, not an outflow. "
                "This is the same convention as HeatFluxBC, whose 'heat_flux' is the heat "
                "entering the body. The default 0 is the do-nothing condition, insulated "
                "for heat transfer and traction free for elasticity.");
  p.addOptional("scale_with_load",
                ParameterKind::Boolean,
                true,
                "Multiply this contribution by the load factor during load stepping. Unlike "
                "the framework default this is true, because applied loading is normally "
                "what is ramped; set it to false for a part of the loading that must stay "
                "fixed while the rest is increased.");
  return p;
}

NeumannBC::NeumannBC(const InputParameters & p) : IntegratedBC(p) {}

void
NeumannBC::initialSetup(Problem & problem)
{
  IntegratedBC::initialSetup(problem);
  _q = getFunction(problem, "flux");
}

ADReal
NeumannBC::computeBoundaryFlux(const QpContext & ctx) const
{
  return ADReal(_q->value(ctx.x, ctx.time));
}

InputParameters
RobinBC::validParams()
{
  InputParameters p = IntegratedBC::validParams();
  p.setClassDescription(
      "Mixed (Robin, convection) condition n . F = q0 - h (u - u_ambient), e.g. Newton's law "
      "of cooling.");
  p.addOptional("transfer_coefficient",
                ParameterKind::Function,
                0.0,
                "Transfer, or film, coefficient h in n . F = q0 - h (u - u_ambient). It "
                "should be non-negative: a positive h drives u towards 'ambient_value', "
                "whereas a negative one drives it away and makes the problem unstable. The "
                "default 0 switches the transfer term off and leaves a pure Neumann "
                "condition of strength 'flux'.");
  p.addOptional("ambient_value",
                ParameterKind::Function,
                0.0,
                "Far-field value u_ambient that the transfer term drives u towards. It is "
                "used only when 'transfer_coefficient' is non-zero. The default is 0 rather "
                "than the initial condition, so leaving it unset cools the surface towards "
                "zero.");
  p.addOptional("flux",
                ParameterKind::Function,
                0.0,
                "Prescribed part q0 of the natural quantity n . F, added on top of the "
                "transfer term. It carries the same sign convention as NeumannBC: a "
                "positive value drives u into the body.");
  return p;
}

RobinBC::RobinBC(const InputParameters & p) : IntegratedBC(p) {}

void
RobinBC::initialSetup(Problem & problem)
{
  IntegratedBC::initialSetup(problem);
  _h = getFunction(problem, "transfer_coefficient");
  _uinf = getFunction(problem, "ambient_value");
  _q = getFunction(problem, "flux");
}

ADReal
RobinBC::computeBoundaryFlux(const QpContext & ctx) const
{
  return _q->value(ctx.x, ctx.time) -
         _h->value(ctx.x, ctx.time) * (ctx.value(_var) - _uinf->value(ctx.x, ctx.time));
}

// ---------------------------------------------------------------------------
// Materials
// ---------------------------------------------------------------------------
InputParameters
GenericConstantMaterial::validParams()
{
  InputParameters p = Material::validParams();
  p.setClassDescription(
      "Declares scalar material properties that are constant in space and time. The "
      "properties are then available to any kernel or boundary condition on the same "
      "blocks: a kernel consumes one by being given its name in the matching *_property "
      "parameter, for example Diffusion's 'diffusivity_property'.");
  p.addRequired("property_names",
                ParameterKind::StringList,
                "Names under which the values are declared, in the same order as "
                "'property_values'; the two lists must be the same length. A name given "
                "here is what a kernel's *_property parameter must be set to in order to "
                "consume the value.");
  p.addRequired("property_values",
                ParameterKind::RealList,
                "One constant per name in 'property_names', matched by position. The units "
                "are whatever the consuming kernel expects; nothing is checked.");
  return p;
}

GenericConstantMaterial::GenericConstantMaterial(const InputParameters & p) : Material(p)
{
  _names = p.getStringList("property_names");
  _values = p.getRealList("property_values");
  if (_names.size() != _values.size())
    throw InputError("'property_names' and 'property_values' must have the same length.");
}

void
GenericConstantMaterial::declareProperties(MaterialPropertyRegistry & r)
{
  _ids.clear();
  for (const auto & n : _names)
    _ids.push_back(r.declare(n, 1));
}

void
GenericConstantMaterial::computeProperties(QpContext & ctx) const
{
  for (std::size_t i = 0; i < _ids.size(); ++i)
    ctx.property(_ids[i]) = ADReal(_values[i]);
}

InputParameters
GenericFunctionMaterial::validParams()
{
  InputParameters p = Material::validParams();
  p.setClassDescription(
      "Declares scalar material properties whose values come from functions of position and "
      "time. The functions must already be registered on the problem. Use this for a "
      "property that varies through the body, such as a graded conductivity; a property "
      "that depends on the solution itself belongs in a material written in Python or "
      "C++.");
  p.addRequired("property_names",
                ParameterKind::StringList,
                "Names under which the values are declared, in the same order as "
                "'functions'; the two lists must be the same length.");
  p.addRequired("functions",
                ParameterKind::StringList,
                "Names of functions already registered on the problem, one per entry of "
                "'property_names' and matched by position. An unknown name is an error at "
                "set-up. Each function is evaluated at (x, y, z, t) at every integration "
                "point, so the property may vary in space and time but not with the "
                "solution.");
  return p;
}

GenericFunctionMaterial::GenericFunctionMaterial(const InputParameters & p) : Material(p)
{
  _names = p.getStringList("property_names");
  _fnames = p.getStringList("functions");
  if (_names.size() != _fnames.size())
    throw InputError("'property_names' and 'functions' must have the same length.");
}

void
GenericFunctionMaterial::initialSetup(Problem & problem)
{
  Material::initialSetup(problem);
  _f.clear();
  for (const auto & n : _fnames)
    _f.push_back(problem.function(n));
}

void
GenericFunctionMaterial::declareProperties(MaterialPropertyRegistry & r)
{
  _ids.clear();
  for (const auto & n : _names)
    _ids.push_back(r.declare(n, 1));
}

void
GenericFunctionMaterial::computeProperties(QpContext & ctx) const
{
  for (std::size_t i = 0; i < _ids.size(); ++i)
    ctx.property(_ids[i]) = ADReal(_f[i]->value(ctx.x, ctx.time));
}

// ---------------------------------------------------------------------------
void
registerFrameworkObjects(Factory & f)
{
  const std::string m = "framework";
  f.add<Diffusion>("Diffusion", ObjectCategory::Kernel, m);
  f.add<AnisotropicDiffusion>("AnisotropicDiffusion", ObjectCategory::Kernel, m);
  f.add<Reaction>("Reaction", ObjectCategory::Kernel, m);
  f.add<BodyForce>("BodyForce", ObjectCategory::Kernel, m);
  f.add<TimeDerivative>("TimeDerivative", ObjectCategory::Kernel, m);
  f.add<Advection>("Advection", ObjectCategory::Kernel, m);
  f.add<CoupledForce>("CoupledForce", ObjectCategory::Kernel, m);
  f.add<DirichletBC>("DirichletBC", ObjectCategory::NodalBC, m);
  f.add<NeumannBC>("NeumannBC", ObjectCategory::BoundaryCondition, m);
  f.add<RobinBC>("RobinBC", ObjectCategory::BoundaryCondition, m);
  f.add<NodalLoad>("PointSource", ObjectCategory::NodalLoad, m);
  f.add<GenericConstantMaterial>("GenericConstantMaterial", ObjectCategory::Material, m);
  f.add<GenericFunctionMaterial>("GenericFunctionMaterial", ObjectCategory::Material, m);
}

} // namespace dualmesh
