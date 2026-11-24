// SPDX-License-Identifier: LGPL-2.1-or-later
//
// Heat transfer module: conduction with temperature-dependent conductivity,
// volumetric heating, heat capacity, and convective/radiative/flux boundary
// conditions.  Governing equation:
//     rho c_p dT/dt - div(k(T) grad T) = q'''.
#include "dualmesh/base/Deformation.h"
#include "dualmesh/base/Factory.h"
#include "dualmesh/base/FieldExpression.h"
#include "dualmesh/base/Kernel.h"
#include "dualmesh/base/Problem.h"
#include "dualmesh/fe/Assembly.h"

#include <algorithm>
#include <cmath>

namespace dualmesh
{

namespace
{

class HeatConduction : public Kernel
{
public:
  static InputParameters validParams()
  {
    InputParameters p = Kernel::validParams();
    p.setClassDescription(
        "Heat conduction: the term div q of the energy balance, with the heat flux "
        "q = -k p(T) grad T in W/m^2. Here T is the temperature in K, k the thermal "
        "conductivity and p(T) = c0 + c1 T + c2 T^2 + ... a polynomial of T. A Neumann "
        "condition prescribes the heat flux -n . q entering the body, with n the outward unit "
        "normal, and a boundary without a condition is insulated.");
    p.addOptional("thermal_conductivity",
                  ParameterKind::Function,
                  1.0,
                  "Thermal conductivity k in W/(m K): a constant, a registered function of x, y, "
                  "z and t, the name of a property, or an expression of the fields, x, y, z, t, "
                  "the functions and the properties. It must be positive, and "
                  "'temperature_polynomial' multiplies it. Default 1, for a problem written in "
                  "dimensionless form.");
    p.setRange("thermal_conductivity", ParameterRange::Positive);
    p.addOptional("temperature_polynomial",
                  ParameterKind::RealList,
                  std::vector<double>{1.0},
                  "Coefficients c0, c1, c2, ... of the polynomial p(T) = c0 + c1 T + c2 T^2 + ... "
                  "that multiplies the conductivity. A list given here must include its constant "
                  "term c0. The argument is the temperature in the units of the variable, with no "
                  "reference temperature subtracted. Newton's method evaluates p at the current "
                  "iterate, and direct iteration at the previous one. Default {1}, which leaves "
                  "the conductivity unchanged.");
    deformation::addParameter(
        p,
        "the heat conducts through the deformed body: the heat flux per unit undeformed area "
        "is -J k F_d^-1 F_d^-T Grad T, with F_d the deformation gradient, J its determinant "
        "and Grad the gradient on the undeformed mesh. The heat source and the heat capacity "
        "need no such change when they are given per unit undeformed volume.");
    return p;
  }
  explicit HeatConduction(const InputParameters & p) : Kernel(p)
  {
    _poly = p.getRealList("temperature_polynomial");
  }
  void initialSetup(Problem & problem) override
  {
    Kernel::initialSetup(problem);
    _k.setup(problem, _params, "thermal_conductivity", name());
    _F = deformation::propertyId(problem, _params, name());
  }
  bool hasFlux() const override { return true; }
  void computeFlux(const QpContext & ctx, ADVector3 & F) const override
  {
    ADReal k = _k.value(ctx);
    if (!(_poly.size() == 1 && _poly[0] == 1.0))
    {
      const ADReal & T = ctx.coefficientValue(_var);
      ADReal poly(0.0);
      for (std::size_t i = _poly.size(); i-- > 0;)
        poly = poly * T + _poly[i];
      k = k * poly;
    }
    const auto & g = ctx.gradient(_var);
    if (_F >= 0)
    {
      deformation::pullBackFlux(ctx, _F, k, g, F);
      return;
    }
    for (int d = 0; d < ctx.dim; ++d)
      F[d] = k * g[d];
  }

private:
  FieldCoefficient _k;
  std::vector<double> _poly;
  int _F = -1;
};

/// Transport of heat by a flow that is itself an unknown of the problem.
class HeatConvection : public Kernel
{
public:
  static InputParameters validParams()
  {
    InputParameters p = Kernel::validParams();
    p.setClassDescription(
        "Convective transport of heat by a velocity v that the same problem solves for: the "
        "term rho c_p v . grad T on the left side of the energy balance of an incompressible "
        "flow, with rho the density, c_p the specific heat capacity and T the temperature. In "
        "an equal-order flow, with the finite element method (FEM) or the dual mesh control "
        "domain method (DMCDM) and the pressure formulation, the term carries the velocity "
        "w = v - tau r_m that the mass equation conserves, with tau r_m the pressure "
        "stabilization of mass_conservation. FEM then adds rho c_p w . grad T, and DMCDM the "
        "face flux rho c_p (w . n)(T_f - T_C) for the control volume C on each side of a face, "
        "with T_f the face value of the advection scheme and n the unit normal of the face. The "
        "heat is then conserved exactly against the discrete mass flux through the boundary "
        "(Hughes and Wells 2005), and the temperature does not depend on its scale. Without "
        "such a mass equation, in the penalty and Taylor-Hood formulations, the term carries v. "
        "With Boussinesq_buoyancy in the momentum equations, the term couples heat transfer and "
        "fluid flow both ways. Newton's method takes the coupling exactly, because automatic "
        "differentiation carries the derivatives with respect to the velocities.");
    p.addRequired("velocities",
                  ParameterKind::StringList,
                  "Names of the velocity variables, exactly one per mesh dimension and in "
                  "coordinate order. They must already exist on the problem.");
    p.addOptional("density",
                  ParameterKind::Function,
                  1.0,
                  "Density rho in kg/m^3: a constant, a registered function, or an expression of "
                  "x, y, z and t. Default 1, for a problem written in dimensionless form.");
    p.setRange("density", ParameterRange::Positive);
    p.addOptional("specific_heat",
                  ParameterKind::Function,
                  1.0,
                  "Specific heat capacity c_p at constant pressure in J/(kg K): a constant, a "
                  "registered function, or an expression of x, y, z and t. Default 1, for a "
                  "problem written in dimensionless form.");
    p.setRange("specific_heat", ParameterRange::Positive);
    addAdvectionSchemeParam(p);
    return p;
  }
  explicit HeatConvection(const InputParameters & p)
      : Kernel(p), _scheme_text(p.getString("advection_scheme")),
        _face_scheme(_scheme_text == "upwind" || _scheme_text == "linear_upwind" ||
                     _scheme_text == "limited"),
        _flux(_face_scheme)
  {
  }
  void initialSetup(Problem & problem) override
  {
    Kernel::initialSetup(problem);
    const auto names = _params.getStringList("velocities");
    const int dim = problem.mesh().spaceDimension();
    if (static_cast<int>(names.size()) != dim)
      throw InputError("heat_convection '" + name() +
                       "': 'velocities' needs one variable per "
                       "dimension (" +
                       std::to_string(dim) + "), got " + std::to_string(names.size()) + ".");
    _v.clear();
    for (const auto & n : names)
      _v.push_back(problem.variableIndex(n));
    _rho = getFunction(problem, "density");
    _cp = getFunction(problem, "specific_heat");
    _scheme = advectionScheme(problem, _scheme_text, name());
    _mass = conservingMassEquation(problem, variable());
    _flux = _face_scheme || (_mass && problem.isDual());
  }
  // The term is rho c_p w . grad T with the velocity w that the mass equation conserves, where a
  // mass equation tests with the functions of T (an equal-order flow), and with v otherwise: the
  // source for FEM, and for DMCDM the face flux rho c_p (w . n)(T_f - T_C) for the control
  // volume C on each side, with the face value T_f of the advection scheme (Hughes and Wells
  // 2005).  As the mass equation holds for every test function, the heat
  // is conserved exactly: what the term takes from the nodes is the heat that the discrete mass
  // flux carries through the boundary, at the temperature of the node it leaves.  And it has no
  // part T div(w), so the temperature does not depend on its scale (kelvin or celsius) at any
  // iterate.  Without such a mass equation (the penalty and Taylor-Hood formulations) it
  // conserves heat only as far as v is divergence free.
  bool hasSource() const override { return !_flux; }
  bool hasFlux() const override { return _flux; }
  bool usesFaceSides() const override { return _flux; }
  bool nonConservativeFaceFlux() const override { return _flux; }
  bool inStrongResidual() const override { return true; }
  bool needsLineSearch() const override { return _scheme == AdvectionScheme::Limited; }
  bool needsSideGradients() const override
  {
    return _scheme == AdvectionScheme::LinearUpwind || _scheme == AdvectionScheme::Limited;
  }
  bool advectingNormalVelocity(const QpContext & ctx, double & velocity) const override
  {
    if (!_flux)
      return false;
    const ADVector3 w = advectingVelocity(ctx, _mass, _v);
    velocity = 0.0;
    for (std::size_t d = 0; d < _v.size(); ++d)
      velocity += w[d].value() * ctx.normal[d];
    return true;
  }
  void computeFlux(const QpContext & ctx, ADVector3 & F) const override
  {
    const ADVector3 w = advectingVelocity(ctx, _mass, _v);
    double velocity = 0.0;
    for (std::size_t d = 0; d < _v.size(); ++d)
      velocity += w[d].value() * ctx.normal[d];
    const ADReal T =
        ctx.advectedValue(variable(), velocity, _scheme) - ctx.referenceValue(variable());
    const double rc = _rho->value(ctx.x, ctx.time) * _cp->value(ctx.x, ctx.time);
    for (std::size_t d = 0; d < _v.size(); ++d)
      F[d] = -rc * w[d] * T;
  }
  ADReal computeSource(const QpContext & ctx) const override
  {
    const auto & g = ctx.gradient(variable());
    const ADVector3 w = advectingVelocity(ctx, _mass, _v);
    ADReal s(0.0);
    for (std::size_t d = 0; d < _v.size(); ++d)
      s += w[d] * g[d];
    return _rho->value(ctx.x, ctx.time) * _cp->value(ctx.x, ctx.time) * s;
  }

private:
  std::vector<int> _v;
  FunctionPtr _rho, _cp;
  std::string _scheme_text;
  /// Whether the scheme is a face flux of DMCDM (upwind, linear_upwind, limited).
  bool _face_scheme;
  /// Whether the term is a flux (conservative with _mass, else non-conservative) or a source.
  bool _flux;
  AdvectionScheme _scheme = AdvectionScheme::Central;
  /// The mass equation whose conserved velocity the flux carries (conservingMassEquation).
  const Kernel * _mass = nullptr;
};

/// The streamline upwind Petrov-Galerkin stabilisation of the heat that a flow carries (Brooks
/// and Hughes 1982), for the finite element method.
class HeatStreamlineStabilization : public Kernel
{
public:
  static InputParameters validParams()
  {
    InputParameters p = Kernel::validParams();
    p.setClassDescription(
        "Streamline upwind Petrov-Galerkin (SUPG) stabilization of the convection of heat by a "
        "flow of velocity v (Brooks and Hughes 1982). It adds the flux F = -tau v r to the "
        "energy balance dm/dt + div F = s, which in the finite element method (FEM) adds the "
        "integral of tau (v . grad N_i) r to the equation of node i, with N_i its shape "
        "function. The dual mesh control domain method (DMCDM) takes the same flux through the "
        "faces of the control domains. The residual r is the strong residual of the energy "
        "balance, rho c_p (dT/dt + v . grad T) - div(k grad T) - s, with rho the density, c_p "
        "the specific heat capacity, k the thermal conductivity, T the temperature and s the "
        "heat source. The code builds r from the other kernels of the temperature, each "
        "weighted as the assembly weights it. Quadratic elements give the conduction as "
        "-k Laplacian(T), without the gradient of k, and linear elements leave it out, since it "
        "vanishes inside them. The parameter is tau = [(2 |v| / h)^2 + 9 (4 kappa / h^2)^2 + "
        "(2 / dt)^2]^(-1/2), with h the element size, halved on a quadratic element, "
        "kappa = k / (rho c_p) the thermal diffusivity and dt the time step, whose term enters "
        "a transient solve only. The factor 9 gives tau = h^2 / (12 kappa) where diffusion "
        "dominates. On linear elements r of the exact solution equals k Laplacian(T), so the "
        "stabilization adds an error of the order of tau k |Laplacian(T)|, which is small where "
        "convection dominates and zero for a harmonic temperature. The code refuses an "
        "advection kernel of the temperature that is written as a flux, which is the case for "
        "the conservative form and for a face flux advection scheme.");
    p.addRequired("velocities",
                  ParameterKind::StringList,
                  "Names of the velocity variables, one per mesh dimension, in coordinate order.");
    p.addOptional("density",
                  ParameterKind::Function,
                  1.0,
                  "Density rho of the flow in kg/m^3, for the diffusivity kappa of tau: a "
                  "constant, a registered function, or an expression of x, y, z and t. Default "
                  "1, for a problem written in dimensionless form.");
    p.setRange("density", ParameterRange::Positive);
    p.addOptional("specific_heat",
                  ParameterKind::Function,
                  1.0,
                  "Specific heat capacity c_p in J/(kg K), for the diffusivity kappa of tau: a "
                  "constant, a registered function, or an expression of x, y, z and t. Default "
                  "1, for a problem written in dimensionless form.");
    p.setRange("specific_heat", ParameterRange::Positive);
    p.addRequired("thermal_conductivity",
                  ParameterKind::String,
                  "Thermal conductivity k in W/(m K), for the diffusivity kappa of tau: a number, "
                  "a registered function, an expression or the name of a property.");
    return p;
  }
  explicit HeatStreamlineStabilization(const InputParameters & p) : Kernel(p) {}
  void initialSetup(Problem & problem) override
  {
    Kernel::initialSetup(problem);
    _v.clear();
    for (const auto & n : _params.getStringList("velocities"))
      _v.push_back(problem.variableIndex(n));
    _rho = getFunction(problem, "density");
    _cp = getFunction(problem, "specific_heat");
    _k.setup(problem, _params.getString("thermal_conductivity"), name());
    // The terms of the residual: the sources of the other kernels of the temperature, matched by
    // name, because a kernel after this one in the list has not resolved its variable yet.
    _terms.clear();
    for (const auto & k : problem.kernels())
    {
      if (k.get() == this || k->variableName() != variableName() ||
          k->type() == "heat_streamline_stabilization")
        continue;
      // An advection written as a flux is a term the residual of sources cannot hold.
      if (k->type() == "advection" && k->hasFlux())
        throw InputError("heat_streamline_stabilization '" + name() + "': the advection '" +
                         k->name() +
                         "' of the temperature is written as a flux, which the "
                         "residual of the stabilisation does not hold. Give it advection_form "
                         "'non_conservative'.");
      if (k->inStrongResidual())
        _terms.push_back(k.get());
    }
    const auto & mesh = problem.mesh();
    _h.resize(mesh.numElements());
    for (Index e = 0; e < mesh.numElements(); ++e)
      _h[e] = elementSize(mesh, e);
  }
  bool hasFlux() const override { return true; }
  bool residualHasTimeDerivative() const override { return true; }
  bool needsLaplacian() const override { return true; }
  void computeFlux(const QpContext & ctx, ADVector3 & F) const override
  {
    // r: the sources of the steady terms, each with the data fraction its kernel is assembled with,
    // and those of the time terms with the weight that makes the time derivative enter a step
    // of the theta method with the weight 1.
    ADReal r(0.0);
    bool transient = false;
    for (const Kernel * k : _terms)
    {
      if (!k->activeOnBlock(ctx.block))
        continue;
      double weight = k->isLoad() ? ctx.data_fraction : 1.0;
      if (k->isTimeKernel())
      {
        if (ctx.dt <= 0.0)
          continue;
        transient = true;
        weight *= ctx.time_weight;
      }
      if (weight != 0.0)
        r += weight * k->computeSource(ctx);
    }
    // The conduction - k Laplacian(T), which quadratic elements give (zero inside a linear one).
    // The gradient of k is left out: the residual is exact for a conductivity constant in space
    // and temperature.
    if (!ctx.laplacian.empty())
      r -= _k.value(ctx) * ctx.laplacian[variable()];
    const double rc = _rho->value(ctx.x, ctx.time) * _cp->value(ctx.x, ctx.time);
    const double h = ctx.element >= 0 ? _h[ctx.element] : 1.0;
    ADReal speed2(1e-300);
    for (int v : _v)
      speed2 += ctx.coefficientValue(v) * ctx.coefficientValue(v);
    const ADReal kappa = _k.value(ctx) / rc;
    ADReal inverse2 = (4.0 / (h * h)) * speed2;
    // The factor 3 (9 on the square) gives the diffusive limit h^2 / (12 kappa) of the optimal
    // parameter of linear elements.
    const ADReal diffusive = 3.0 * 4.0 * kappa / (h * h);
    inverse2 += diffusive * diffusive;
    if (transient)
      inverse2 += (2.0 / ctx.dt) * (2.0 / ctx.dt);
    const ADReal tau = 1.0 / sqrt(inverse2);
    for (std::size_t d = 0; d < _v.size(); ++d)
      F[d] = tau * ctx.coefficientValue(_v[d]) * r;
  }

private:
  std::vector<int> _v;
  FunctionPtr _rho, _cp;
  FieldCoefficient _k;
  std::vector<const Kernel *> _terms;
  std::vector<double> _h;
};

class HeatConductionTimeDerivative : public Kernel
{
public:
  static InputParameters validParams()
  {
    InputParameters p = Kernel::validParams();
    p.setClassDescription(
        "The heat capacity term of the transient energy balance: the storage term dm/dt, with "
        "m = rho c_p T, taken as rho c_p (T - T_old) / dt. Here rho is the density, c_p the "
        "specific heat capacity, T the temperature, T_old the temperature at the start of the "
        "step and dt the time step. A steady solve skips the term, and the weight theta of the "
        "theta method, which multiplies the steady terms, does not multiply it.");
    p.addOptional("density",
                  ParameterKind::Function,
                  1.0,
                  "Density rho in kg/m^3: a constant, a registered function, the name of a "
                  "property, or an expression. Only the product rho c_p enters, so leave one of "
                  "the two at 1 when the volumetric heat capacity is known. Default 1, for a "
                  "problem written in dimensionless form or with the volumetric heat capacity in "
                  "'specific_heat'.");
    p.setRange("density", ParameterRange::Positive);
    p.addOptional("specific_heat",
                  ParameterKind::Function,
                  1.0,
                  "Specific heat capacity c_p in J/(kg K): a constant, a registered function, the "
                  "name of a property, or an expression. A specific heat that depends on the "
                  "temperature takes the new temperature, so the term is "
                  "rho c_p(T) (T - T_old) / dt. Default 1, for a problem written in "
                  "dimensionless form or with the volumetric heat capacity in 'density'.");
    p.setRange("specific_heat", ParameterRange::Positive);
    return p;
  }
  explicit HeatConductionTimeDerivative(const InputParameters & p) : Kernel(p) {}
  void initialSetup(Problem & problem) override
  {
    Kernel::initialSetup(problem);
    _rho.setup(problem, _params, "density", name());
    _cp.setup(problem, _params, "specific_heat", name());
  }
  bool hasSource() const override { return true; }
  bool isTimeKernel() const override { return true; }
  ADReal computeSource(const QpContext & ctx) const override
  {
    return _rho.value(ctx) * _cp.value(ctx) * (ctx.value(_var) - ctx.oldValue(_var)) / ctx.dt;
  }

private:
  FieldCoefficient _rho, _cp;
};

class RadiativeHeatFluxBC : public IntegratedBC
{
public:
  static InputParameters validParams()
  {
    InputParameters p = IntegratedBC::validParams();
    p.setClassDescription(
        "Grey-body radiation from a surface: the heat flux leaving the body is "
        "n . q = e sigma_SB (T^4 - T_a^4), with n the outward unit normal, q the heat flux, e "
        "the emissivity, sigma_SB the Stefan-Boltzmann constant, T the temperature of the "
        "surface and T_a the ambient temperature, both in K.");
    p.addOptional("emissivity",
                  ParameterKind::Real,
                  1.0,
                  "Total hemispherical emissivity e of the surface, dimensionless, between 0 and "
                  "1. The code refuses a negative value and does not check the upper bound. The "
                  "object multiplies it by the Stefan-Boltzmann constant once when it is built, "
                  "so a later change to either has no effect. Default 1, a black body.");
    p.setRange("emissivity", ParameterRange::NonNegative);
    p.addOptional("stefan_boltzmann_constant",
                  ParameterKind::Real,
                  5.670374419e-8,
                  "Stefan-Boltzmann constant sigma_SB in W/(m^2 K^4). Change it only to work in "
                  "another system of units. Default 5.670374419e-8, its value in SI units.");
    p.addOptional("ambient_temperature",
                  ParameterKind::Function,
                  0.0,
                  "Temperature T_a of the surroundings in K: a constant, a registered function, "
                  "or an expression of x, y, z and t. It must be on an absolute scale, because "
                  "it enters as its fourth power, and a value in degrees Celsius gives a wrong "
                  "result without a warning. Default 0, radiation into deep space.");
    deformation::addParameter(
        p,
        "the flux is per unit deformed area, and the code multiplies it by the area ratio "
        "da/dA = |J F_d^-T N| to give the flux per unit undeformed area, with F_d the "
        "deformation gradient, J its determinant and N the outward unit normal of the "
        "undeformed boundary.");
    return p;
  }
  explicit RadiativeHeatFluxBC(const InputParameters & p)
      : IntegratedBC(p), _coef(p.getReal("emissivity") * p.getReal("stefan_boltzmann_constant"))
  {
  }
  void initialSetup(Problem & problem) override
  {
    IntegratedBC::initialSetup(problem);
    _Tinf = getFunction(problem, "ambient_temperature");
    _F = deformation::propertyId(problem, _params, name());
  }
  ADReal computeBoundaryFlux(const QpContext & ctx) const override
  {
    const double Ta = _Tinf->value(ctx.x, ctx.time);
    const ADReal & T = ctx.value(_var);
    ADReal q;
    if (ctx.mode == LinearizationMode::Picard)
    {
      // (T^4 - Ta^4) = (T^2 + Ta^2)(T + Ta)(T - Ta), lag the first factors.
      const ADReal & Tl = ctx.coefficientValue(_var);
      q = -_coef * (Tl * Tl + Ta * Ta) * (Tl + Ta) * (T - Ta);
    }
    else
      q = -_coef * (pow(T, 4.0) - Ta * Ta * Ta * Ta);
    return _F >= 0 ? q * deformation::areaRatio(ctx, _F) : q;
  }

private:
  double _coef;
  FunctionPtr _Tinf;
  int _F = -1;
};

/// Heat transfer across a gap between two bodies that do not touch.
class GapHeatTransfer : public InterfaceBC
{
public:
  static InputParameters validParams()
  {
    InputParameters p = InterfaceBC::validParams();
    p.setClassDescription(
        "Heat transfer across a gap between two separate bodies, from the primary side set "
        "'boundary' to the closest points of 'secondary_boundary'. The heat flux entering the "
        "primary body, per unit primary area, is q_g = h_g (T_s - T_p) + sigma_SB (T_s^4 - "
        "T_p^4) / (1/e_p + 1/e_s - 1), and the secondary body loses the same heat. Here T_p and "
        "T_s are the temperatures of the primary and secondary surfaces, h_g the gap "
        "conductance, sigma_SB the Stefan-Boltzmann constant, and e_p and e_s the emissivities "
        "of the two surfaces, whose radiation term is that of two parallel grey surfaces. The "
        "temperature is 'variable' on both sides unless 'secondary_variable' names another.");
    p.addOptional("gap_conductance",
                  ParameterKind::Function,
                  0.0,
                  "Gap conductance h_g in W/(m^2 K), per unit area of the primary surface: a "
                  "constant, a registered function, or an expression of x, y, z and t. Default "
                  "0, no conduction across the gap.");
    p.addOptional("primary_emissivity",
                  ParameterKind::Real,
                  0.0,
                  "Emissivity e_p of the primary surface. The radiation term acts only when both "
                  "emissivities are positive, and the temperatures must then be in K. Default 0, "
                  "no radiation across the gap.");
    p.addOptional("secondary_emissivity",
                  ParameterKind::Real,
                  0.0,
                  "Emissivity e_s of the secondary surface. Default 0, no radiation across the "
                  "gap.");
    p.addOptional("stefan_boltzmann_constant",
                  ParameterKind::Real,
                  5.670374419e-8,
                  "Stefan-Boltzmann constant sigma_SB in W/(m^2 K^4). Change it only to work in "
                  "another system of units. Default 5.670374419e-8, its value in SI units.");
    deformation::addParameter(p,
                              "the conductance is per unit deformed primary area, and the code "
                              "multiplies the flux by the area ratio da/dA = |J F_d^-T N| of the "
                              "primary surface, with F_d the deformation gradient, J its "
                              "determinant and N the outward unit normal of the undeformed "
                              "surface.");
    return p;
  }
  explicit GapHeatTransfer(const InputParameters & p) : InterfaceBC(p)
  {
    const double ep = p.getReal("primary_emissivity");
    const double es = p.getReal("secondary_emissivity");
    _radiation = ep > 0 && es > 0
                     ? p.getReal("stefan_boltzmann_constant") / (1.0 / ep + 1.0 / es - 1.0)
                     : 0.0;
  }
  void initialSetup(Problem & problem) override
  {
    InterfaceBC::initialSetup(problem);
    _h = getFunction(problem, "gap_conductance");
    _F = deformation::propertyId(problem, _params, name());
  }
  ADReal computeInterfaceFlux(const QpContext & ctx) const override
  {
    const ADReal & Tp = ctx.value(_var);
    const ADReal & Ts = ctx.u_other[_secondary_var];
    ADReal q = _h->value(ctx.x, ctx.time) * (Ts - Tp);
    if (_radiation > 0)
      q += _radiation * (Ts * Ts + Tp * Tp) * (Ts + Tp) * (Ts - Tp);
    return _F >= 0 ? q * deformation::areaRatio(ctx, _F) : q;
  }

private:
  FunctionPtr _h;
  double _radiation = 0.0;
  int _F = -1;
};

} // namespace

void
registerHeatTransferObjects(Factory & f)
{
  const std::string m = "heat_transfer";
  f.add<HeatConduction>("heat_conduction", ObjectCategory::Kernel, m);
  f.add<HeatConvection>("heat_convection", ObjectCategory::Kernel, m);
  f.add<HeatStreamlineStabilization>("heat_streamline_stabilization", ObjectCategory::Kernel, m);
  f.add<HeatConductionTimeDerivative>("heat_conduction_time_derivative", ObjectCategory::Kernel, m);
  f.add<RadiativeHeatFluxBC>("radiative_heat_flux", ObjectCategory::BoundaryCondition, m);
  f.add<GapHeatTransfer>("gap_heat_transfer", ObjectCategory::BoundaryCondition, m);
}

} // namespace dualmesh
