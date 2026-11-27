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

#include <algorithm>
#include <array>
#include <cmath>
#include <map>

namespace dualmesh
{

// ---------------------------------------------------------------------------
// Diffusion: the kernel flux k(x) p(u) grad u, p(u) = c0 + c1 u + c2 u^2 + ...
// ---------------------------------------------------------------------------
InputParameters
Diffusion::validParams()
{
  InputParameters p = Kernel::validParams();
  p.setClassDescription(
      "The diffusion term div F of the balance law dm/dt + div F = s, with the flux "
      "F = -k p(u) grad u. Here u is the variable, k the diffusivity and p(u) = c0 + c1 u + "
      "c2 u^2 + ... a polynomial of u. For heat conduction u is the temperature T in K, k the "
      "thermal conductivity in W/(m K) and F the heat flux q in W/m^2. A Neumann condition "
      "prescribes the inflow -n . F, with n the outward unit normal, and a boundary without a "
      "condition has no inflow: an insulated boundary for heat conduction.");
  p.addOptional("diffusivity",
                ParameterKind::Function,
                1.0,
                "Diffusivity k, the thermal conductivity in W/(m K) for heat conduction: a "
                "constant, a registered function of x, y, z and t, the name of a property, or an "
                "expression of the fields, x, y, z, t, the functions and the properties. It must "
                "be positive. Default 1, for a problem written in dimensionless form.");
  p.setRange("diffusivity", ParameterRange::Positive);
  p.addOptional("solution_polynomial",
                ParameterKind::RealList,
                std::vector<double>{1.0},
                "Coefficients c0, c1, c2, ... of the polynomial p(u) = c0 + c1 u + c2 u^2 + ... "
                "that multiplies the diffusivity. A list given here must include its constant "
                "term c0. The argument is the variable in its own units, with no reference value "
                "subtracted. Newton's method evaluates p at the current iterate, and direct "
                "iteration at the previous one. Default {1}, which leaves the diffusivity "
                "unchanged.");
  deformation::addParameter(
      p,
      "the variable diffuses through the deformed body: the flux per unit undeformed area is "
      "-J k F_d^-1 F_d^-T Grad u, with F_d the deformation gradient, J its determinant and Grad "
      "the gradient on the undeformed mesh. A source and a storage need no such change when "
      "they are given per unit undeformed volume.");
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
  _F = deformation::propertyId(problem, _params, name());
}

ADReal
Diffusion::diffusivity(const QpContext & ctx) const
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
  const ADReal k = diffusivity(ctx);
  const auto & g = ctx.gradient(_var);
  if (_F >= 0)
  {
    deformation::pullBackFlux(ctx, _F, k, g, F);
    return;
  }
  for (int d = 0; d < ctx.dim; ++d)
    F[d] = k * g[d];
}

ADReal
Diffusion::strongResidual(const QpContext & ctx) const
{
  if (ctx.laplacian.empty())
    return ADReal(0.0);
  return -diffusivity(ctx) * ctx.laplacian[_var];
}

// ---------------------------------------------------------------------------
InputParameters
AnisotropicDiffusion::validParams()
{
  InputParameters p = Kernel::validParams();
  p.setClassDescription(
      "The anisotropic diffusion term div F of the balance law dm/dt + div F = s, with a "
      "constant diffusivity tensor K and the flux F = -K grad u, whose component i is "
      "F_i = -sum_j K_ij du/dx_j. A Neumann condition prescribes the inflow -n . F, with n the "
      "outward unit normal.");
  p.addRequired("diffusivity_tensor",
                ParameterKind::RealList,
                "Entries of the diffusivity tensor K, constant in space and time. One value per "
                "dimension gives the diagonal (K_xx, K_yy, K_zz). The square of the dimension "
                "gives the whole tensor, row by row (K_xx, K_xy, K_yx, K_yy in two dimensions). "
                "Any other count is an error. The code does not check K for symmetry or positive "
                "definiteness, and a K that is not symmetric gives a Jacobian that is not "
                "symmetric.");
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
      "The reaction source s = -c u^p of the balance law dm/dt + div F = s, with u the "
      "variable. A positive c removes u, as a decay does in transport and an elastic foundation "
      "does under a structure, and a negative c produces it.");
  p.addOptional("coefficient",
                ParameterKind::Function,
                1.0,
                "Coefficient c of the source s = -c u^p: a constant, a registered function, the "
                "name of a property, or an expression of the fields, x, y, z, t, the functions "
                "and the properties. A coefficient that depends on the solution makes the term "
                "nonlinear, and automatic differentiation carries its derivative into the "
                "Jacobian. Default 1, for a problem written in dimensionless form.");
  p.addOptional("exponent",
                ParameterKind::Real,
                1.0,
                "Exponent p of the source s = -c u^p. Any value other than 1 makes the term "
                "nonlinear. Direct iteration then linearizes it as c u_previous^(p-1) u, with "
                "u_previous the previous iterate, and a p that is not an integer requires u to "
                "stay positive. Default 1, a linear reaction.");
  return p;
}

Reaction::Reaction(const InputParameters & p) : Kernel(p), _p(p.getReal("exponent")) {}

void
Reaction::initialSetup(Problem & problem)
{
  Kernel::initialSetup(problem);
  _c.setup(problem, _params, "coefficient", name());
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
      "The source s of the balance law dm/dt + div F = s, per unit volume: the heat source in "
      "W/m^3 of an equation for a temperature, the body force in N/m^3 of an equation for a "
      "displacement or a velocity component, or the production rate of an equation for a "
      "concentration. A positive value drives the variable up: it adds heat, or pushes along the "
      "positive axis. The heat_conduction physics adds this kernel for its heat_source.");
  p.addOptional("value",
                ParameterKind::Function,
                1.0,
                "The source s per unit volume: a constant, a registered function of x, y, z and "
                "t, the name of a property, or an expression of the fields, x, y, z, t, the "
                "functions and the properties. For a temperature it is the heat source in W/m^3. "
                "A source that depends on the solution makes the term nonlinear, and automatic "
                "differentiation carries its derivative into the Jacobian. Default 1, a unit "
                "source for a problem written in dimensionless form.");
  return p;
}

BodyForce::BodyForce(const InputParameters & p) : Kernel(p) {}

void
BodyForce::initialSetup(Problem & problem)
{
  Kernel::initialSetup(problem);
  _f.setup(problem, _params, "value", name());
}

ADReal
BodyForce::computeSource(const QpContext & ctx) const
{
  return -_f.value(ctx);
}

// ---------------------------------------------------------------------------
InputParameters
TimeDerivative::validParams()
{
  InputParameters p = Kernel::validParams();
  p.setClassDescription(
      "The storage term of the balance law dm/dt + div F = s, as c (m - m_old) / dt over a step "
      "of the theta method. Here m is the storage, an expression of the fields, their "
      "gradients, the position, the time and the states of a parsed_state object, m_old the "
      "same expression at the start of the step (the fields at their old values, the time "
      "t - dt and the states at the start of the step), c a coefficient and dt the time step. "
      "The difference of the storage itself makes the change of the stored amount over a step "
      "equal to what the fluxes and the sources bring, for any storage m. The default storage "
      "is the variable u, which gives the rate form c du/dt, with c the capacity: the "
      "volumetric heat capacity rho c_p of heat conduction, with rho the density and c_p the "
      "specific heat capacity, or the density rho of the momentum of a flow. A capacity that "
      "depends on the solution takes its value at the end of the step. In an equation that also "
      "has a second_time_derivative, du/dt follows the Newmark relations of the "
      "generalized-alpha method, and the term c du/dt is the damping of a structure. A steady "
      "solve skips the term, and the weight theta of the theta method, which multiplies the "
      "steady terms, does not multiply it. With the automatic quadrature, the term is lumped "
      "(the nodal rule) on a mesh of linear elements when no kernel of the variable carries it "
      "with a flow and the equation is of first order in time: the system of each step then "
      "keeps the maximum principle of diffusion on a mesh without obtuse angles, for every time "
      "step, and the amount stored is unchanged. A quadratic element, whose lumped weights are "
      "not all positive, a variable carried by a flow, whose transport loses phase accuracy "
      "with a lumped storage, and the damping of a structure, which is proportional to its "
      "consistent mass, take the Gauss rule. An eigenvalue study takes the Gauss rule wherever "
      "time stepping lumps the term, which under FEM makes each eigenvalue of a symmetric "
      "problem an upper bound of the exact one (the Rayleigh-Ritz method). The storage uses "
      "the fields by their names, the components of their gradients as grad_x(u), grad_y(u) "
      "and grad_z(u), the coordinates x, y and z, the time t, the registered functions, the "
      "element fields, the states of its state_object and the named constants.");
  p.addOptional("coefficient",
                ParameterKind::Function,
                1.0,
                "Coefficient c of the rate of the storage: a constant, a registered function of "
                "x, y, z and t, the name of a property, or an expression of the fields. For heat "
                "conduction it is the volumetric heat capacity rho c_p in J/(m^3 K), and for a "
                "flow the density rho in kg/m^3. Default 1, for a storage given in 'storage' or "
                "a problem written in dimensionless form.");
  p.addOptional("storage",
                ParameterKind::String,
                std::string(""),
                "The storage m, an expression. The code refuses it in an equation of second "
                "order in time, whose rate follows the Newmark relations of the variable alone. "
                "Default: the variable u, whose rate is du/dt.");
  p.addOptional("state_object",
                ParameterKind::String,
                std::string(""),
                "The name of the parsed_state object whose states the storage reads, which "
                "gives their values at the start of the step. Default: none, for a storage "
                "without states.");
  p.addOptional("constants",
                ParameterKind::RealMap,
                RealMap{},
                "Named constants that the storage uses, as name -> value pairs. Default: none, "
                "for a storage that needs no constant of its own.");
  return p;
}

TimeDerivative::TimeDerivative(const InputParameters & p)
    : StorageKernel(p), _storage(!p.getString("storage").empty())
{
  _automatic = _quad.automatic;
}

bool
TimeDerivative::lumped(const Problem & problem) const
{
  // The consistent storage has positive entries off the diagonal, so a short step undershoots
  // ahead of a front on any mesh (Thomee and Wahlbin 2008).  The lumped storage keeps the maximum
  // principle on a mesh without obtuse angles (Fujii 1973).
  for (const auto & el : problem.mesh().elements())
    if (elementIsQuadratic(el.type))
      return false;
  for (const auto & k : problem.kernels())
  {
    const auto * storage = dynamic_cast<const StorageKernel *>(k.get());
    if (k->variableName() == variableName() &&
        (k->transportsVariable() || (storage && storage->isSecondOrder())))
      return false;
  }
  return true;
}

void
TimeDerivative::initialSetup(Problem & problem)
{
  // The automatic rule is resolved again at each setup, for the mesh of that setup.
  const bool reduced = _params.getBool("reduced_integration");
  if (_automatic)
    _quad = QuadratureSpec::parse("automatic", reduced);
  Kernel::initialSetup(problem);
  _consistent = _quad;
  if (_automatic && lumped(problem))
    _quad = QuadratureSpec::parse("nodal", reduced);
  _c.setup(problem, _params, "coefficient", name());
  _state = nullptr;
  const std::string state_object = _params.getString("state_object");
  if (!_storage)
  {
    if (!state_object.empty())
      throw InputError("'" + name() + "': a state_object needs a 'storage' that reads its states.");
    return;
  }
  for (const auto & k : problem.kernels())
    if (const auto * storage = dynamic_cast<const StorageKernel *>(k.get());
        storage && storage->variableName() == variableName() && storage->isSecondOrder())
      throw InputError("'" + name() + "': the equation of '" + variableName() +
                       "' is of second order in time, whose rate follows the Newmark relations "
                       "of the variable alone, so its storage is the variable itself. Leave "
                       "'storage' out.");
  FieldExpressions::Names names;
  for (const auto & [n, value] : _params.getRealMap("constants"))
    names.constants.emplace_back(n, value);
  _expressions.compile(problem, {_params.getString("storage")}, names, name());
  if (!state_object.empty())
  {
    _state = std::dynamic_pointer_cast<const Property>(problem.object(state_object));
    if (!_state)
      throw InputError("'" + name() + "': the state_object '" + state_object +
                       "' is not a property object.");
  }
}

void
TimeDerivative::startArguments(const QpContext & ctx, ADReal * out) const
{
  _expressions.startArguments(
      ctx,
      [&](int property)
      {
        double value = 0.0;
        if (_state && _state->startValue(ctx, property, value))
          return value;
        throw InputError("'" + name() +
                         "': the storage reads a property other than the states of its "
                         "state_object, and only those keep their value at the start of the "
                         "step.");
      },
      out);
}

ADReal
TimeDerivative::computeSource(const QpContext & ctx) const
{
  if (!_storage)
    return _c.value(ctx) * ctx.rate(_var);
  std::array<ADReal, FieldExpressions::kMaxArguments> before;
  startArguments(ctx, before.data());
  return _c.value(ctx) *
         ((_expressions.value(0, ctx) - _expressions.value(0, before.data()).value()) / ctx.dt);
}

double
TimeDerivative::sourceTermSize(const QpContext & ctx, const ADReal & source) const
{
  if (!_storage)
    return std::abs(source.value());
  std::array<ADReal, FieldExpressions::kMaxArguments> now, before;
  _expressions.arguments(ctx, now.data());
  startArguments(ctx, before.data());
  return std::abs(_c.value(ctx).value()) *
         (_expressions.termSize(0, now.data()) + _expressions.termSize(0, before.data())) / ctx.dt;
}

// ---------------------------------------------------------------------------
InputParameters
SecondTimeDerivative::validParams()
{
  InputParameters p = Kernel::validParams();
  p.setClassDescription(
      "The inertia term c d2u/dt2 on the left side of the balance law dm/dt + div F = s, with "
      "u the variable. A transient run integrates an equation with this term by the "
      "generalized-alpha method, or by the average acceleration method of Newmark. An "
      "eigenvalue study takes the term as the mass of the natural frequencies. A steady solve "
      "skips it.");
  p.addOptional("coefficient",
                ParameterKind::Function,
                1.0,
                "Coefficient c of d2u/dt2: a constant, a registered function of x, y, z and t, "
                "the name of a property, or an expression of the fields. For a solid it is the "
                "density rho in kg/m^3. Default 1, for a problem written in dimensionless form.");
  return p;
}

SecondTimeDerivative::SecondTimeDerivative(const InputParameters & p) : StorageKernel(p) {}

void
SecondTimeDerivative::initialSetup(Problem & problem)
{
  Kernel::initialSetup(problem);
  _c.setup(problem, _params, "coefficient", name());
}

ADReal
SecondTimeDerivative::computeSource(const QpContext & ctx) const
{
  return _c.value(ctx) * ctx.acceleration(_var);
}

// ---------------------------------------------------------------------------
InputParameters
Advection::validParams()
{
  InputParameters p = Kernel::validParams();
  p.setClassDescription(
      "The transport of the variable u by a velocity v, with a coefficient c: the heat "
      "rho c_p v . grad T that a flow carries, with rho the density, c_p the specific heat "
      "capacity and T the temperature, the inertia rho (v . grad) v_i of the momentum equation "
      "of the velocity component v_i, or any other advection. The non_conservative form adds "
      "c v . grad u to the left side of the balance law dm/dt + div F = s, and the "
      "conservative form adds the flux F = c v u. The two forms agree where div v = 0. The "
      "velocity is prescribed, or it is the velocity of a flow that the same problem solves "
      "for, when every component names a variable. In an equal-order flow, with the finite "
      "element method (FEM) or the dual mesh control domain method (DMCDM) and the pressure "
      "formulation, the term then carries the velocity w = v - tau r_m that the mass equation "
      "conserves, with tau r_m the pressure stabilization of mass_conservation. FEM then adds "
      "c w . grad u, and DMCDM the face flux c (w . n)(u_f - u_C) for the control volume C on "
      "each side of a face, with u_f the face value of the advection scheme and n the unit "
      "normal of the face. The quantity c u is then conserved exactly against the discrete mass "
      "flux through the boundary (Hughes and Wells 2005), and the solution does not change when "
      "a constant is added to it. Without such a mass equation, in the penalty and Taylor-Hood "
      "formulations, the term carries v, which conserves c u only as far as the discrete "
      "velocity is free of divergence. Picard iteration takes the velocity of a flow from the "
      "previous iterate. A transport of a component of its own velocity is the inertia of the "
      "flow, which the Stokes start of a steady flow at rest turns off.");
  p.addOptional("velocity",
                ParameterKind::StringList,
                std::vector<std::string>{"0", "0", "0"},
                "The velocity components (v_x, v_y, v_z) in m/s, each a number, a registered "
                "function, the name of a property or an expression of x, y, z, t and the fields. "
                "For the velocity of a flow, give the names of its velocity variables, one per "
                "dimension of the mesh and in coordinate order. Missing trailing components of a "
                "prescribed velocity are zero. Default zero in every component, which advects "
                "nothing.");
  p.addOptional("coefficient",
                ParameterKind::Function,
                1.0,
                "Coefficient c of the transport: the volumetric heat capacity rho c_p in "
                "J/(m^3 K) for the heat that a flow carries, and the density rho in kg/m^3 for "
                "the inertia of a flow. It is a constant, a registered function, the name of a "
                "property, or an expression of the fields, x, y, z, t, the functions and the "
                "properties. Default 1, for a problem written in dimensionless form.");
  p.addOptional("advection_form",
                ParameterKind::String,
                std::string("non_conservative"),
                "The form of the advection term. 'conservative' adds the flux F = c v u, which "
                "conserves c u over the control domains, for an equation written as div(c v u). "
                "'non_conservative' adds c v . grad u, for an equation written in that form. "
                "With an advection scheme other than central both forms are face fluxes, with m "
                "the normal velocity times the face area: c m u_f through each face for "
                "'conservative', and c m (u_f - u_C) for the control volume C on each side for "
                "'non_conservative'. Any other string is an error. Default non_conservative, "
                "which does not see the discrete divergence of the velocity: the solution then "
                "does not change when a constant is added to it, and upwind stays bounded.");
  addAdvectionSchemeParam(p);
  return p;
}

Advection::Advection(const InputParameters & p)
    : AdvectiveKernel(p), _face_scheme(isFaceAdvectionScheme(p.getString("advection_scheme")))
{
  const auto form = p.getString("advection_form");
  if (form == "conservative")
    _conservative = true;
  else if (form != "non_conservative")
    throw InputError("'" + name() +
                     "': advection_form must be 'conservative' or 'non_conservative'.");
  _flux = _conservative || _face_scheme;
}

void
Advection::initialSetup(Problem & problem)
{
  Kernel::initialSetup(problem);
  auto v = _params.getStringList("velocity");
  if (v.size() > 3)
    throw InputError("'" + name() + "': the velocity has at most three components.");
  _velocities.clear();
  _inertia = false;
  if (std::all_of(v.begin(), v.end(), [&](const auto & n) { return problem.hasVariable(n); }))
  {
    const int dim = problem.mesh().spaceDimension();
    if (static_cast<int>(v.size()) != dim)
      throw InputError("'" + name() + "': the velocity of a flow needs one variable per " +
                       "dimension (" + std::to_string(dim) + "), got " + std::to_string(v.size()) +
                       ".");
    for (const auto & n : v)
    {
      _velocities.push_back(problem.variableIndex(n));
      _inertia = _inertia || n == variableName();
    }
  }
  else
  {
    v.resize(3, "0");
    for (int d = 0; d < 3; ++d)
      _v[d].setup(problem, v[d], name() + "', velocity component " + std::to_string(d) + " '");
  }
  _c.setup(problem, _params, "coefficient", name());
  _scheme = advectionScheme(problem, _params.getString("advection_scheme"), name());
  // The velocity of an equal-order flow, whose mass equation tests with the functions of u: the
  // term carries the velocity w that it conserves, as a face flux in DMCDM.
  _mass = _velocities.empty() ? nullptr : conservingMassEquation(problem, variable());
  _flux = _conservative || _face_scheme || (_mass && problem.isDual());
}

bool
Advection::oldStateFromFields() const
{
  if (_c.readsProperties())
    return false;
  return std::none_of(_v.begin(), _v.end(), [](const auto & v) { return v.readsProperties(); });
}

ADReal
Advection::coefficient(const QpContext & ctx) const
{
  return _inertia ? ctx.inertia * _c.value(ctx) : _c.value(ctx);
}

ADVector3
Advection::velocity(const QpContext & ctx) const
{
  ADVector3 v{ADReal(0.0), ADReal(0.0), ADReal(0.0)};
  if (_velocities.empty())
    for (int d = 0; d < ctx.dim; ++d)
      v[d] = _v[d].value(ctx);
  else
    for (std::size_t d = 0; d < _velocities.size(); ++d)
      v[d] = ctx.coefficientValue(_velocities[d]);
  return v;
}

ADVector3
Advection::carryingVelocity(const QpContext & ctx) const
{
  return _velocities.empty() ? velocity(ctx) : advectingVelocity(ctx, _mass, _velocities);
}

void
Advection::computeFlux(const QpContext & ctx, ADVector3 & F) const
{
  // The flux c (w . n) u_f through a face, less c (w . n) u_C for the control volume C of the
  // equation in the non-conservative form (QpContext::referenceValue).
  const ADVector3 w = carryingVelocity(ctx);
  double normal = 0.0;
  for (int d = 0; d < ctx.dim; ++d)
    normal += w[d].value() * ctx.normal[d];
  const ADReal u = ctx.advectedValue(_var, normal, _scheme) - ctx.referenceValue(_var);
  const ADReal c = coefficient(ctx);
  for (int d = 0; d < ctx.dim; ++d)
    F[d] = -c * w[d] * u;
}

ADReal
Advection::computeSource(const QpContext & ctx) const
{
  const auto & g = ctx.gradient(_var);
  const ADVector3 w = carryingVelocity(ctx);
  ADReal s(0.0);
  for (int d = 0; d < ctx.dim; ++d)
    s += w[d] * g[d];
  return coefficient(ctx) * s;
}

ADReal
Advection::strongResidual(const QpContext & ctx) const
{
  const auto & g = ctx.gradient(_var);
  const ADVector3 v = velocity(ctx);
  ADReal s(0.0);
  for (int d = 0; d < ctx.dim; ++d)
    s += v[d] * g[d];
  return coefficient(ctx) * s;
}

// ---------------------------------------------------------------------------
InputParameters
CoupledForce::validParams()
{
  InputParameters p = Kernel::validParams();
  p.setClassDescription(
      "A source proportional to another variable u_c, which couples two equations of a "
      "multiphysics problem: the source of the balance law dm/dt + div F = s is s = c u_c. "
      "Newton's method takes the coupling exactly, because automatic differentiation carries "
      "the derivative with respect to u_c.");
  p.addRequired("coupled_variable",
                ParameterKind::String,
                "Name of the variable u_c that drives this equation. It must already exist on "
                "the problem.");
  p.addOptional("coefficient",
                ParameterKind::Function,
                1.0,
                "Coefficient c of the source s = c u_c: a constant, a registered function, the "
                "name of a property (for a coefficient that differs between regions), or an "
                "expression of the fields, x, y, z, t, the functions and the properties. A "
                "positive c makes u_c a source for this equation, and a negative c makes it a "
                "sink. Default 1, which adds u_c itself.");
  return p;
}

CoupledForce::CoupledForce(const InputParameters & p) : Kernel(p) {}

void
CoupledForce::initialSetup(Problem & problem)
{
  Kernel::initialSetup(problem);
  _v = coupledVariable(problem, "coupled_variable");
  _c.setup(problem, _params, "coefficient", name());
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
  p.setClassDescription(
      "Prescribes the value u = g(x, t) of the variable u at the nodes of a boundary, an "
      "essential boundary condition.");
  p.addOptional("value",
                ParameterKind::Function,
                0.0,
                "Prescribed value g(x, t): a constant, a registered function, or an expression of "
                "x, y, z and t. The code writes it into the solution at every node of the "
                "boundary. When a steady solve ramps its data to converge, the value moves from "
                "the starting state to g with the other data. Default 0, the homogeneous "
                "condition.");
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
      "Prescribes the value u = g(x, t) of the variable u at the one node nearest to a point, "
      "for example to fix the level of a pressure that only its gradient enters. For a "
      "first-order variable on a quadratic mesh the node is the nearest corner node.");
  p.remove("boundary");
  p.addRequired("point",
                ParameterKind::RealList,
                "The coordinates (x, y, z) of the point, in m. Missing trailing coordinates are "
                "zero.");
  p.addOptional("value",
                ParameterKind::Function,
                0.0,
                "Prescribed value g(x, t): a constant, a registered function, or an expression of "
                "x, y, z and t. Default 0, the homogeneous condition.");
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
  if (problem.hasMixedOrder())
  {
    carries.assign(problem.mesh().numNodes(), 0);
    for (const auto & el : problem.mesh().elements())
      for (int a = 0; a < problem.numLocalNodes(el, variable()); ++a)
        carries[el.nodes[a]] = 1;
  }
  // Every process agrees on one node, so that exactly one is held: a flow with its pressure given
  // at a point has one pressure level.
  const Index nearest = problem.nearestNode(
      x, [&](Index n) { return carries.empty() || carries[static_cast<std::size_t>(n)]; });
  _nodes.clear();
  if (nearest >= 0)
    _nodes.push_back(nearest);
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
      "Prescribes the inflow q = -n . F through a boundary, where F is the flux of the balance "
      "law dm/dt + div F = s that the kernels of this variable assemble and n is the outward "
      "unit normal. The inflow is the secondary variable of the duality pair. A boundary without "
      "any condition has this condition with q = 0.");
  p.addOptional(
      "flux",
      ParameterKind::Function,
      0.0,
      "Prescribed inflow q = -n . F. A positive value enters the body: for heat transfer it is "
      "the heat flux into the body in W/m^2, and for the equation of a displacement component "
      "it is the component of the traction in Pa. It is a constant, a registered function, or "
      "an expression of the fields, their gradients (grad_x(u), grad_y(u), grad_z(u)), x, y, z "
      "and t, which automatic differentiation differentiates. Default 0, the condition of a "
      "boundary without one: insulated for heat transfer and free of traction for elasticity.");
  deformation::addParameter(p,
                            "the flux is per unit deformed area, and the code multiplies it by "
                            "the area ratio da/dA = |J F_d^-T N| to give the flux per unit "
                            "undeformed area, with F_d the deformation gradient, J its "
                            "determinant and N the outward unit normal of the undeformed "
                            "boundary.");
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
      "Mixed condition of Robin type, such as Newton's law of cooling. The inflow through the "
      "boundary is -n . F = q_0 - h_t (u - u_a), where F is the flux of the balance law "
      "dm/dt + div F = s, n the outward unit normal, u the variable, h_t the transfer "
      "coefficient, u_a the ambient value and q_0 a prescribed inflow.");
  p.addOptional(
      "transfer_coefficient",
      ParameterKind::Function,
      0.0,
      "Transfer, or film, coefficient h_t, in W/(m^2 K) for heat transfer: a constant, a "
      "registered function, or an expression of the fields, their gradients, x, y, z and t. It "
      "must not be negative, and a positive h_t drives u towards 'ambient_value'. Default 0, "
      "which leaves the prescribed inflow 'flux' alone, as in a Neumann condition.");
  p.setRange("transfer_coefficient", ParameterRange::NonNegative);
  p.addOptional("ambient_value",
                ParameterKind::Function,
                0.0,
                "Ambient value u_a that the transfer term drives u towards: a constant, a "
                "registered function, or an expression of the fields, their gradients, x, y, z "
                "and t. It acts only when 'transfer_coefficient' is not zero. Default 0, which "
                "does not follow the initial condition, so give it with every "
                "'transfer_coefficient'.");
  p.addOptional(
      "flux",
      ParameterKind::Function,
      0.0,
      "Prescribed inflow q_0, added to the transfer term, with the sign of the Neumann "
      "condition: a positive value enters the body. It is a constant, a registered function, or "
      "an expression of the fields, their gradients, x, y, z and t. Default 0, no prescribed "
      "inflow.");
  deformation::addParameter(p,
                            "the flux is per unit deformed area, and the code multiplies it by "
                            "the area ratio da/dA = |J F_d^-T N| to give the flux per unit "
                            "undeformed area, with F_d the deformation gradient, J its "
                            "determinant and N the outward unit normal of the undeformed "
                            "boundary.");
  return p;
}

RobinBC::RobinBC(const InputParameters & p) : IntegratedBC(p) {}

void
RobinBC::initialSetup(Problem & problem)
{
  IntegratedBC::initialSetup(problem);
  _h.setup(problem, _params, "transfer_coefficient", name());
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
      "Declares scalar properties that are constant in space and time. Every kernel, boundary "
      "condition, property object and expression on the same blocks reads them by name, as a "
      "coefficient given as diffusivity = 'thermal_conductivity' does.");
  p.addRequired("properties",
                ParameterKind::RealMap,
                "The declared properties, as name -> value pairs, for example "
                "{thermal_conductivity: 400.0}. Each value is in the units that the objects "
                "reading it expect, and the code does not check them.");
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
        "Multiplies properties that the property objects added before it compute on its blocks "
        "by the constant factors of 'property_factors'. It serves sensitivity and uncertainty "
        "studies, which scale a correlation without a change to the property object that "
        "computes it.");
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

// ---------------------------------------------------------------------------
// ---- periodic boundary condition --------------------------------------------------------
InputParameters
PeriodicBC::validParams()
{
  InputParameters p = Constraint::validParams();
  p.setClassDescription(
      "Periodic boundary condition: the field on the secondary boundary equals the field on the "
      "primary boundary, carried over by a translation. Every node of the secondary boundary "
      "is joined to the node of the primary boundary at the translated position. The two nodes "
      "act as one unknown whose equation collects both, so that its control domain wraps around "
      "the domain. A node that is periodic in two directions, at a corner, is joined to one "
      "primary node. The meshes of the two boundaries must match node by node after the "
      "translation, to within 10^-6 times the larger of the size of the primary boundary and "
      "the length of the translation.");
  p.addRequired("primary", ParameterKind::String, "The side set or node set whose nodes are kept.");
  p.addRequired("secondary",
                ParameterKind::String,
                "The side set or node set whose nodes become copies of the primary nodes.");
  p.addOptional("variable",
                ParameterKind::Names,
                std::vector<std::string>{},
                "The variable or variables that are periodic. Default: every variable of the "
                "problem, as in a periodic cell.");
  p.addOptional("translation",
                ParameterKind::RealList,
                std::vector<double>{},
                "The vector from a primary node to its secondary node, in m (x, y and z "
                "components). Default: the vector from the centroid of the primary nodes to the "
                "centroid of the secondary nodes, which is the translation between two boundaries "
                "that match.");
  return p;
}

PeriodicBC::PeriodicBC(const InputParameters & p) : PeriodicConstraint(p) {}

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
  const auto primary = problem.mesh().boundaryNodes(primaryBoundary());
  const auto secondary = problem.mesh().boundaryNodes(secondaryBoundary());
  std::vector<Point> primary_points, secondary_points;
  for (Index n : primary)
    primary_points.push_back(problem.mesh().node(n));
  for (Index n : secondary)
    secondary_points.push_back(problem.mesh().node(n));
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
  f.add<SecondTimeDerivative>("second_time_derivative", ObjectCategory::Kernel, m);
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
}

} // namespace dualmesh
