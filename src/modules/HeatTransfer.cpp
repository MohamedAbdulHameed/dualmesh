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
#include "dualmesh/materials/Gases.h"

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
        "Heat conduction -div(k grad T) with k = k(x, t) * (c0 + c1 T + c2 T^2 + ...). "
        "The natural boundary quantity n . (k grad T) is the heat flux entering the body.");
    p.addOptional("thermal_conductivity",
                  ParameterKind::Function,
                  1.0,
                  "Thermal conductivity k in watts per metre per kelvin: a constant, a registered "
                  "function of (x, y, z, t), the name of a property, or an expression of the "
                  "fields, x, y, z, t, the functions and the properties. It must be positive. "
                  "'temperature_polynomial' multiplies it. Default 1.");
    p.setRange("thermal_conductivity", ParameterRange::Positive);
    p.addOptional("temperature_polynomial",
                  ParameterKind::RealList,
                  std::vector<double>{1.0},
                  "Coefficients c0, c1, ... of a polynomial that multiplies the base "
                  "conductivity, giving k(x, t) (c0 + c1 T + c2 T^2 + ...). The default {1} "
                  "leaves the conductivity unchanged, so a list given here must include its own "
                  "constant term. The temperature is the current iterate under Newton's method "
                  "and the previous one under direct iteration. The argument of the polynomial is "
                  "the temperature itself, in the units of the variable, without subtraction of a "
                  "reference temperature.");
    deformation::addParameter(
        p,
        "the heat conducts through the deformed body: the flux per unit undeformed area is "
        "J k F^-1 F^-T Grad T, with Grad the gradient on the undeformed mesh. The heat "
        "source and the heat capacity need no such change when they are given per unit "
        "undeformed volume (the mass and the fissile atoms of an element do not change "
        "as it deforms).");
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
        "Convective transport of heat by a velocity field that is solved for in the same "
        "problem: the source S = rho c_p (v . grad T), the advective term of the energy "
        "equation of an incompressible flow. Together with a buoyancy force in the momentum "
        "equations (Boussinesq_buoyancy) it couples heat transfer and fluid flow in both "
        "directions, and in Newton's method the coupling is exact: the derivatives with "
        "respect to the velocities are carried through the automatic differentiation.");
    p.addRequired("velocities",
                  ParameterKind::StringList,
                  "Names of the velocity variables, exactly one per mesh dimension and in "
                  "coordinate order. They must already exist on the problem.");
    p.addOptional("density",
                  ParameterKind::Function,
                  1.0,
                  "Mass density rho, as a constant or the name of a function.");
    p.setRange("density", ParameterRange::Positive);
    p.addOptional("specific_heat",
                  ParameterKind::Function,
                  1.0,
                  "Specific heat capacity c_p at constant pressure, as a constant or the name "
                  "of a function.");
    p.setRange("specific_heat", ParameterRange::Positive);
    addAdvectionSchemeParam(p);
    return p;
  }
  explicit HeatConvection(const InputParameters & p)
      : Kernel(p), _scheme_text(p.getString("advection_scheme")),
        _face_flux(_scheme_text == "upwind" || _scheme_text == "linear_upwind" ||
                   _scheme_text == "limited")
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
    _mass = _face_flux ? massConservationKernel(problem) : nullptr;
  }
  // The term is the source rho c_p v . grad T, or with an advection scheme of a control volume
  // method a face flux with the face value T_f that the scheme takes from the two sides: the
  // conservative rho c_p (w . n) T_f with the velocity w that the mass equation conserves, whose
  // flux sums to zero over every control volume, so that the heat is conserved exactly; or,
  // without a mass equation, the non-conservative rho c_p (v . n)(T_f - T_C) for the control
  // volume C on each side of the face, which sees no discrete divergence of v.
  bool hasSource() const override { return !_face_flux; }
  bool hasFlux() const override { return _face_flux; }
  bool usesFaceSides() const override { return _face_flux; }
  bool nonConservativeFaceFlux() const override { return _face_flux && !_mass; }
  bool hasKinks() const override { return _scheme == AdvectionScheme::Limited; }
  bool needsSideGradients() const override
  {
    return _scheme == AdvectionScheme::LinearUpwind || _scheme == AdvectionScheme::Limited;
  }
  bool advectingNormalVelocity(const QpContext & ctx, double & velocity) const override
  {
    if (!_face_flux)
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
    ADReal s(0.0);
    for (std::size_t d = 0; d < _v.size(); ++d)
      s += ctx.coefficientValue(_v[d]) * g[d];
    return _rho->value(ctx.x, ctx.time) * _cp->value(ctx.x, ctx.time) * s;
  }

private:
  std::vector<int> _v;
  FunctionPtr _rho, _cp;
  std::string _scheme_text;
  bool _face_flux;
  AdvectionScheme _scheme = AdvectionScheme::Central;
  /// The mass equation whose conserved velocity a face flux carries, or nullptr.
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
        "Streamline upwind Petrov-Galerkin stabilisation of the convection of heat by a flow: the "
        "flux F = tau v r, which adds the integral of tau (v . grad w) r to the equation of the "
        "test function w. The residual r is the sum of the sources of every other kernel of the "
        "temperature (the convection rho c_p v . grad T, the heat sources, the heat capacity "
        "term), each weighted as the assembly weights it, that is the residual of the energy "
        "equation without its conduction term, whose second derivatives vanish on linear "
        "elements. An advection of the temperature written as a flux is refused. tau = [(2 |v| / h)^2 + 9 (4 kappa / h^2)^2 + "
        "(2 / dt)^2]^(-1/2), kappa = k / (rho c_p) (Brooks and Hughes 1982, Tezduyar 1991, with "
        "the factor 9 of MOOSE), the last term in a transient solve. Without the conduction term the residual of the exact "
        "solution is k Laplacian(T), not zero, so the stabilisation adds an error of the order of "
        "tau k |Laplacian(T)|: small where convection dominates, where tau is about h / (2 |v|), "
        "and zero for a harmonic temperature. For FEM and DMCDM, at the dual faces for DMCDM, as "
        "momentum_stabilization is.");
    p.addRequired("velocities",
                  ParameterKind::StringList,
                  "Names of the velocity variables, one per mesh dimension, in coordinate order.");
    p.addOptional("density",
                  ParameterKind::Function,
                  1.0,
                  "Mass density rho of the flow, for the diffusivity kappa of tau.");
    p.setRange("density", ParameterRange::Positive);
    p.addOptional(
        "specific_heat", ParameterKind::Function, 1.0, "Specific heat c_p, for kappa of tau.");
    p.setRange("specific_heat", ParameterRange::Positive);
    p.addRequired("thermal_conductivity",
                  ParameterKind::String,
                  "Thermal conductivity k, for kappa of tau: a number, a function, an expression "
                  "or a property.");
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
      if (k->hasSource())
        _terms.push_back(k.get());
    }
    const auto & mesh = problem.mesh();
    _h.resize(mesh.numElements());
    for (Index e = 0; e < mesh.numElements(); ++e)
      _h[e] = elementSize(mesh, e);
  }
  bool hasFlux() const override { return true; }
  bool residualHasTimeDerivative() const override { return true; }
  void computeFlux(const QpContext & ctx, ADVector3 & F) const override
  {
    // r: the sources of the steady terms, each with the load factor its kernel is assembled with,
    // and those of the time terms with the weight that makes the time derivative enter a step
    // of the theta method with the weight 1.
    ADReal r(0.0);
    bool transient = false;
    for (const Kernel * k : _terms)
    {
      if (!k->activeOnBlock(ctx.block))
        continue;
      double weight = k->scalesWithLoad() ? ctx.load_factor : 1.0;
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
    const double rc = _rho->value(ctx.x, ctx.time) * _cp->value(ctx.x, ctx.time);
    const double h = ctx.element >= 0 ? _h[ctx.element] : 1.0;
    ADReal speed2(1e-300);
    for (int v : _v)
      speed2 += ctx.coefficientValue(v) * ctx.coefficientValue(v);
    const ADReal kappa = _k.value(ctx) / rc;
    ADReal inverse2 = (4.0 / (h * h)) * speed2;
    // The factor 3 (9 on the square) gives the diffusive limit h^2 / (12 kappa) of the optimal
    // parameter of linear elements, as MOOSE's navier_stokes module takes it (INSADTauMaterial).
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
        "The heat capacity term of the transient energy equation. It contributes no flux and "
        "the source S = rho c_p (T - T_old) / dt. It is a time kernel: it is skipped entirely "
        "in a steady solve, and it is not multiplied by the time-integration weight theta, "
        "which applies to the steady terms only.");
    p.addOptional("density",
                  ParameterKind::Function,
                  1.0,
                  "Mass density rho in kilograms per cubic metre: a constant, a registered "
                  "function, the name of a property, or an expression. Only the product of the "
                  "density and the specific heat enters, so leave one of them at 1 if the "
                  "volumetric heat capacity is known directly. Default 1.");
    p.setRange("density", ParameterRange::Positive);
    p.addOptional("specific_heat",
                  ParameterKind::Function,
                  1.0,
                  "Specific heat capacity c_p in joules per kilogram per kelvin: a constant, a "
                  "registered function, the name of a property, or an expression. A "
                  "temperature-dependent specific heat is evaluated at the new temperature, so "
                  "the term is rho c_p(T) (T - T_old) / dt. Default 1.");
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
        "Grey-body radiation: n . (k grad T) = -emissivity sigma (T^4 - T_ambient^4) "
        "(temperatures in kelvin).");
    p.addOptional("emissivity",
                  ParameterKind::Real,
                  1.0,
                  "Total hemispherical emissivity of the surface, dimensionless and between "
                  "0 and 1, with 1 for a black body. The range is not checked. It is "
                  "multiplied by the Stefan-Boltzmann constant once when the object is "
                  "built, so neither may be changed afterwards.");
    p.setRange("emissivity", ParameterRange::NonNegative);
    p.addOptional("stefan_boltzmann_constant",
                  ParameterKind::Real,
                  5.670374419e-8,
                  "Stefan-Boltzmann constant, 5.670374419e-8 W/m^2/K^4 by default. Change it only "
                  "to work in a different system of units. It is a physical constant and must not "
                  "be used as a fitting parameter.");
    p.addOptional("ambient_temperature",
                  ParameterKind::Function,
                  0.0,
                  "Temperature of the surroundings, which must be on an absolute scale because it "
                  "enters as its fourth power: a value in degrees celsius gives a wrong result "
                  "without any warning. The default 0 models radiation into deep space.");
    deformation::addParameter(
        p,
        "the flux is per unit deformed area and is multiplied by the area ratio da/dA = |J F^-T N| "
        "to give the flux per unit undeformed area.");
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
        "'boundary' to the closest points of 'secondary_boundary': q = h (T_s - T_p) + sigma "
        "F (T_s^4 - T_p^4), where q is the heat flux entering the primary body per unit "
        "primary area and the secondary body receives the same heat. h is the gap "
        "conductance and F the radiation exchange factor of two parallel grey surfaces, "
        "1 / (1/eps_p + 1/eps_s - 1). The temperature variable is 'variable' on both "
        "sides. For a conductance that follows the gap width and the gas, see "
        "gas_gap_heat_transfer.");
    p.addOptional("gap_conductance",
                  ParameterKind::Function,
                  0.0,
                  "Gap conductance h in W/m^2/K, per unit area of the primary surface, as a "
                  "constant or a function of position and time.");
    p.addOptional("primary_emissivity",
                  ParameterKind::Real,
                  0.0,
                  "Emissivity of the primary surface. Radiation is included when both "
                  "emissivities are positive, and the temperatures must then be absolute.");
    p.addOptional(
        "secondary_emissivity", ParameterKind::Real, 0.0, "Emissivity of the secondary surface.");
    p.addOptional("stefan_boltzmann_constant",
                  ParameterKind::Real,
                  5.670374419e-8,
                  "Stefan-Boltzmann constant in W/m^2/K^4.");
    deformation::addParameter(p,
                              "the conductance is per unit deformed primary area, and the flux "
                              "is multiplied by the area ratio da/dA = |J F^-T N| of the "
                              "primary surface.");
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

/// Heat transfer across a gas-filled gap between two bodies, with the gap
/// width taken from the displaced surfaces and solid contact when it closes.
class GasGapHeatTransfer : public InterfaceBC
{
public:
  static InputParameters validParams()
  {
    InputParameters p = InterfaceBC::validParams();
    p.setClassDescription(
        "Heat transfer across a gas-filled gap between two bodies: gas conduction, radiation and "
        "solid contact, h_gap = h_gas + h_rad + h_solid, the model of Ross and Stoute (AECL-1552, "
        "1962, Eqs. A.9 and A.15). The gas conductance is k_gas / (g + C_r (R_p + R_s) + j), "
        "with the width g of the gap measured along the primary normal between the displaced "
        "surfaces, the roughnesses R, the roughness coefficient C_r and the temperature jump "
        "distance j (the equation of Kennard as Lanning and Hann, BNWL-1894, 1975, write it). "
        "k_gas is the mixture conductivity of the gas by Brokaw's Eqs. (12) and (13). Radiation "
        "is between two parallel grey surfaces. The solid contact conductance, used when a "
        "gap_contact condition on the same surfaces closes the gap, is C_s k_m P_c / (sqrt(R) H) "
        "with the "
        "harmonic mean conductivity k_m, the contact pressure P_c, the Meyer hardness H and R = "
        "sqrt((R_p^2 + R_s^2)/2). Temperatures must be in kelvin.");
    p.addOptional("displacements",
                  ParameterKind::StringList,
                  std::vector<std::string>{},
                  "Displacement variables. Default none: the gap keeps its as-meshed width.");
    deformation::addParameter(p,
                              "the conductance is per unit deformed primary area, and the flux "
                              "is multiplied by the area ratio da/dA = |J F^-T N| of the "
                              "primary surface.");
    for (const char * gas : {"helium", "argon", "krypton", "xenon", "hydrogen", "nitrogen"})
      p.addOptional(std::string(gas) + "_fraction",
                    ParameterKind::Function,
                    0.0,
                    std::string("Mole fraction of ") + gas +
                        " in the gap gas. Default 0. The fractions are normalized to a sum of "
                        "one, and at least one must be positive.");
    p.addRequired("gas_pressure", ParameterKind::Function, "Pressure of the gap gas, Pa.");
    p.addRequired("primary_roughness",
                  ParameterKind::Real,
                  "Arithmetic mean roughness of the primary surface, m.");
    p.addRequired("secondary_roughness",
                  ParameterKind::Real,
                  "Arithmetic mean roughness of the secondary surface, m.");
    p.addOptional("roughness_coeff",
                  ParameterKind::Real,
                  1.5,
                  "C_r, the factor on the summed roughnesses in the gas conductance. Default "
                  "1.5, the value of Ross and Stoute at 500 kgf/cm^2 of contact pressure (2.5 at "
                  "100 kgf/cm^2, Sect. 6.2.1).");
    p.addRequired("primary_emissivity",
                  ParameterKind::Real,
                  "Emissivity of the primary surface. Zero switches radiation off.");
    p.addRequired("secondary_emissivity",
                  ParameterKind::Real,
                  "Emissivity of the secondary surface. Zero switches radiation off.");
    p.addOptional("accommodation_coeff",
                  ParameterKind::Real,
                  std::monostate{},
                  "Thermal accommodation coefficient of the gas on the surfaces, between 0 and 1. "
                  "Default: the fits of Ullman et al. for helium (0.425 - 2.3e-4 T) and xenon "
                  "(0.749 - 2.5e-4 T), interpolated by the mixture molar mass (Lanning and Hann, "
                  "BNWL-1894, 1975, Appendix B, Sect. 1).");
    p.addOptional(
        "meyer_hardness",
        ParameterKind::Real,
        std::monostate{},
        "Meyer hardness H of the softer surface, Pa. Required when a gap_contact on the same surfaces closes the gap.");
    p.addOptional("solid_contact_coeff",
                  ParameterKind::Real,
                  20.0,
                  "C_s = 1/a_0 of the solid contact conductance, m^(-1/2). Default 20: a_0 = 1/2 "
                  "cm^(1/2), the value Ross and Stoute found for most of their pairs (Sect. 6.1, "
                  "range 1/2 to 1 cm^(1/2)).");
    p.addOptional("gas_conductance_factor",
                  ParameterKind::Real,
                  1.0,
                  "Factor on the gas conduction term, for sensitivity and uncertainty studies. "
                  "Default 1.");
    p.addOptional("contact_conductance_factor",
                  ParameterKind::Real,
                  1.0,
                  "Factor on the solid contact term, for sensitivity and uncertainty studies. "
                  "Default 1.");
    p.addOptional(
        "primary_conductivity",
        ParameterKind::Function,
        std::monostate{},
        "Thermal conductivity of the primary body at its surface (W/m/K), for the solid "
        "contact term. Required when a gap_contact closes the gap and no 'thermal_conductivity' "
        "property exists.");
    p.addOptional(
        "secondary_conductivity",
        ParameterKind::Function,
        std::monostate{},
        "Thermal conductivity of the secondary body at its surface (W/m/K), for the "
        "solid contact term. Required when a gap_contact on the same surfaces closes the gap.");
    return p;
  }
  explicit GasGapHeatTransfer(const InputParameters & p) : InterfaceBC(p)
  {
    _Rp = p.getReal("primary_roughness");
    _Rs = p.getReal("secondary_roughness");
    if (!(_Rp >= 0.0) || !(_Rs >= 0.0))
      throw InputError("'" + name() + "': the roughnesses must not be negative.");
    _Cr = p.getReal("roughness_coeff");
    const double ep = p.getReal("primary_emissivity"), es = p.getReal("secondary_emissivity");
    if (!(ep >= 0.0 && ep <= 1.0) || !(es >= 0.0 && es <= 1.0))
      throw InputError("'" + name() + "': the emissivities must lie between 0 and 1.");
    _F = ep > 0 && es > 0 ? 1.0 / (1.0 / ep + 1.0 / es - 1.0) : 0.0;
    if (p.isSet("accommodation_coeff"))
    {
      _accommodation = p.getReal("accommodation_coeff");
      if (!(_accommodation > 0.0 && _accommodation <= 1.0))
        throw InputError("'" + name() + "': accommodation_coeff must lie in (0, 1].");
    }
    _Cs = p.getReal("solid_contact_coeff");
    if (_Cs < 0.0)
      throw InputError("'" + name() + "': solid_contact_coeff must not be negative.");
    _gas_factor = p.getReal("gas_conductance_factor");
    _contact_factor = p.getReal("contact_conductance_factor");
    if (!(_gas_factor >= 0.0) || !(_contact_factor >= 0.0))
      throw InputError("'" + name() + "': the conductance factors must not be negative.");
  }
  void initialSetup(Problem & problem) override
  {
    InterfaceBC::initialSetup(problem);
    _disp.clear();
    for (const auto & d : _params.getStringList("displacements"))
    {
      _disp.push_back(problem.variableIndex(d));
      addCoupled(_disp.back());
    }
    int i = 0;
    for (const char * gas : {"helium", "argon", "krypton", "xenon", "hydrogen", "nitrogen"})
      _x[i++] = getFunction(problem, std::string(gas) + "_fraction");
    _P = getFunction(problem, "gas_pressure");
    // The solid contact conductance needs the contact pressure, from the penalty of the
    // gap_contact conditions on the same pair of surfaces.
    _penalty = 0.0;
    const auto same = [](std::vector<std::string> a, std::vector<std::string> b)
    {
      std::sort(a.begin(), a.end());
      std::sort(b.begin(), b.end());
      return a == b;
    };
    for (const auto & bc : problem.integratedBCs())
      if (bc->type() == "gap_contact" &&
          same(bc->parameters().getStringList("boundary"), _params.getStringList("boundary")) &&
          same(bc->parameters().getStringList("secondary_boundary"),
               _params.getStringList("secondary_boundary")))
        _penalty = bc->parameters().getReal("penalty");
    if (_penalty > 0.0)
    {
      if (!_params.isSet("meyer_hardness"))
        throw InputError("'" + name() +
                         "': the gap closes by contact (gap_contact on the same surfaces), so "
                         "the solid contact conductance needs 'meyer_hardness' (Pa), the Meyer "
                         "hardness of the softer surface.");
      if (!_params.isSet("secondary_conductivity"))
        throw InputError("'" + name() +
                         "': the gap closes by contact (gap_contact on the same surfaces), so "
                         "the solid contact conductance needs 'secondary_conductivity' (W/m/K).");
      _H = _params.getReal("meyer_hardness");
      if (!(_H > 0.0))
        throw InputError("'" + name() + "': meyer_hardness must be positive.");
    }
    _kprop = problem.propertyRegistry().has("thermal_conductivity")
                 ? problem.propertyRegistry().id("thermal_conductivity")
                 : -1;
    if (_penalty > 0.0)
    {
      if (_kprop < 0 && !_params.isSet("primary_conductivity"))
        throw InputError(
            "'" + name() +
            "': the solid contact conductance needs 'primary_conductivity' (W/m/K) when no "
            "'thermal_conductivity' property exists.");
      if (_kprop < 0)
        _kp = getFunction(problem, "primary_conductivity");
      _ks = getFunction(problem, "secondary_conductivity");
    }
    _Fprop = deformation::propertyId(problem, _params, name());
  }
  /// Gap width along the primary normal between the displaced surfaces.
  ADReal gapWidth(const QpContext & ctx) const
  {
    ADReal g(0.0);
    for (int d = 0; d < ctx.dim; ++d)
    {
      ADReal diff(ctx.x_other[d] - ctx.x[d]);
      if (d < static_cast<int>(_disp.size()))
        diff = diff + ctx.u_other[_disp[d]] - ctx.u[_disp[d]];
      g += diff * ctx.normal[d];
    }
    return g;
  }
  ADReal computeInterfaceFlux(const QpContext & ctx) const override
  {
    using namespace materials;
    const ADReal & Tp = ctx.value(_var);
    const ADReal & Ts = ctx.u_other[_secondary_var];
    double x[kNumGases];
    double sum = 0;
    for (int i = 0; i < kNumGases; ++i)
    {
      x[i] = std::max(0.0, _x[i]->value(ctx.x, ctx.time));
      sum += x[i];
    }
    if (sum <= 0)
      throw InputError("'" + name() + "': the gap gas mole fractions add up to zero.");
    for (double & xi : x)
      xi /= sum;
    // The gas correlations are evaluated at 200 K or above, the lower end of
    // the fits: Newton's method can visit a zero temperature from a cold
    // initial guess. The radiation term uses the true temperatures.
    const ADReal mean = 0.5 * (Tp + Ts);
    const ADReal Tg = mean.value() < kMinimumGasTemperature ? ADReal(kMinimumGasTemperature) : mean;
    const double P = _P->value(ctx.x, ctx.time);
    const ADReal kg = gasMixtureConductivity(x, Tg);
    const ADReal jump = gasJumpDistance(x, Tg, P, kg, _accommodation);
    const ADReal g = gapWidth(ctx);
    const ADReal open = g.value() > 0 ? g : ADReal(0.0);
    ADReal h = _gas_factor * kg / (open + _Cr * (_Rp + _Rs) + jump);
    if (_F > 0)
      h += kStefanBoltzmann * _F * (Tp * Tp + Ts * Ts) * (Tp + Ts);
    if (_penalty > 0 && g.value() < 0)
    {
      const ADReal kp = _kprop >= 0 ? ctx.property(_kprop) : ADReal(_kp->value(ctx.x, ctx.time));
      const double ks = _ks->value(ctx.x, ctx.time);
      const ADReal km = 2.0 * kp * ks / (kp + ks);
      const ADReal Pc = -_penalty * g;
      const double R = std::sqrt(0.5 * (_Rp * _Rp + _Rs * _Rs));
      h += _contact_factor * _Cs * km * Pc / (std::sqrt(R) * _H);
    }
    const ADReal q = h * (Ts - Tp);
    return _Fprop >= 0 ? q * deformation::areaRatio(ctx, _Fprop) : q;
  }

private:
  static constexpr double kMinimumGasTemperature = 200.0;
  static constexpr double kStefanBoltzmann = 5.670374419e-8;
  int _Fprop = -1;
  std::vector<int> _disp;
  FunctionPtr _x[materials::kNumGases];
  FunctionPtr _P, _kp, _ks;
  int _kprop = -1;
  double _Rp = 0.0, _Rs = 0.0, _Cr = 1.5, _F = 0.0, _penalty = 0.0, _H = 0.0, _Cs = 20.0;
  double _accommodation = 0.0;
  double _gas_factor = 1.0, _contact_factor = 1.0;
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
  f.add<GasGapHeatTransfer>("gas_gap_heat_transfer", ObjectCategory::BoundaryCondition, m);
}

} // namespace dualmesh
