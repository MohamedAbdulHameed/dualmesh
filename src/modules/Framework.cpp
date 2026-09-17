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
  p.addOptional("diffusivity", ParameterKind::Function, 1.0, "Diffusivity k(x, t).");
  p.addOptional("diffusivity_property",
                ParameterKind::String,
                std::string(""),
                "Name of a material property to use as the diffusivity (overrides "
                "'diffusivity').");
  p.addOptional("solution_polynomial",
                ParameterKind::RealList,
                std::vector<double>{1.0},
                "Coefficients c0, c1, ... of the polynomial dependence of k on u.");
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
      "Anisotropic diffusion -div(K grad u) with a constant tensor K. Give 'diffusivity_tensor' "
      "as dim values (diagonal) or dim*dim values (row by row).");
  p.addRequired("diffusivity_tensor", ParameterKind::RealList, "Diffusivity tensor entries.");
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
  p.setClassDescription("Reaction (or elastic foundation) term S = c(x, t) u^p.");
  p.addOptional("coefficient", ParameterKind::Function, 1.0, "Coefficient c(x, t).");
  p.addOptional("exponent", ParameterKind::Real, 1.0, "Exponent p.");
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
  p.setClassDescription("Volumetric source f(x, t) on the right-hand side: S = -f.");
  p.addOptional("value", ParameterKind::Function, 1.0, "Source intensity f(x, t).");
  p.addOptional("scale_with_load", ParameterKind::Boolean, true, "Scale with the load factor.");
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
  p.setClassDescription("Time derivative c du/dt (backward difference within the theta scheme).");
  p.addOptional("coefficient", ParameterKind::Function, 1.0, "Capacity coefficient c(x, t).");
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
                "Names of functions for the velocity components (override 'velocity').");
  p.addOptional("form",
                ParameterKind::String,
                std::string("non_conservative"),
                "'conservative' or 'non_conservative'.");
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
  p.setClassDescription("Coupling source S = -c v, where v is another variable.");
  p.addRequired("coupled_variable", ParameterKind::String, "The coupled variable v.");
  p.addOptional("coefficient", ParameterKind::Function, 1.0, "Coefficient c(x, t).");
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
  p.addOptional("value", ParameterKind::Function, 0.0, "Prescribed value g(x, t).");
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
      "Prescribed outward normal flux q = n . F (natural boundary condition; the specified "
      "secondary variable).");
  p.addOptional("flux", ParameterKind::Function, 0.0, "Outward normal flux q(x, t).");
  p.addOptional("scale_with_load", ParameterKind::Boolean, true, "Scale with the load factor.");
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
  p.addOptional("transfer_coefficient", ParameterKind::Function, 0.0, "Coefficient h(x, t).");
  p.addOptional("ambient_value", ParameterKind::Function, 0.0, "Ambient value u_ambient(x, t).");
  p.addOptional("flux", ParameterKind::Function, 0.0, "Additional outward flux q0(x, t).");
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
  p.setClassDescription("Declares constant scalar material properties.");
  p.addRequired("property_names", ParameterKind::StringList, "Property names.");
  p.addRequired("property_values", ParameterKind::RealList, "Property values.");
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
  p.setClassDescription("Declares scalar material properties given by functions of (x, t).");
  p.addRequired("property_names", ParameterKind::StringList, "Property names.");
  p.addRequired("functions", ParameterKind::StringList, "Function names (one per property).");
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
