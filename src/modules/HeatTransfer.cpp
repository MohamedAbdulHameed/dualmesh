// SPDX-License-Identifier: LGPL-2.1-or-later
//
// Heat transfer module: conduction with temperature-dependent conductivity,
// volumetric heating, heat capacity, and convective/radiative/flux boundary
// conditions.  Governing equation:
//     rho c_p dT/dt - div(k(T) grad T) = q'''.
#include "dualmesh/base/Deformation.h"
#include "dualmesh/base/Factory.h"
#include "dualmesh/base/Kernel.h"
#include "dualmesh/base/Problem.h"
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
                  "Thermal conductivity k in watts per metre per kelvin, as a constant or the "
                  "name of a function of (x, y, z, t). It must be positive, and it is ignored "
                  "when 'thermal_conductivity_property' names a property.");
    p.addOptional("thermal_conductivity_property",
                  ParameterKind::String,
                  std::string(""),
                  "Name of a property to use as the conductivity in place of "
                  "'thermal_conductivity'. It replaces only that base value: "
                  "'temperature_polynomial' still multiplies it. Leave it empty to use the "
                  "value.");
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
    _prop_name = p.getString("thermal_conductivity_property");
  }
  void initialSetup(Problem & problem) override
  {
    Kernel::initialSetup(problem);
    _k = getFunction(problem, "thermal_conductivity");
    _prop = _prop_name.empty() ? -1 : problem.propertyRegistry().id(_prop_name);
    _F = deformation::propertyId(problem, _params, name());
  }
  bool hasFlux() const override { return true; }
  void computeFlux(const QpContext & ctx, ADVector3 & F) const override
  {
    ADReal k = _prop >= 0 ? ctx.property(_prop) : ADReal(_k->value(ctx.x, ctx.time));
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
  FunctionPtr _k;
  std::vector<double> _poly;
  std::string _prop_name;
  int _prop = -1, _F = -1;
};

class HeatSource : public Kernel
{
public:
  static InputParameters validParams()
  {
    InputParameters p = Kernel::validParams();
    p.setClassDescription(
        "Volumetric heat generation. In the canonical form -div F + S = 0 this kernel "
        "contributes no flux and the source S = -q, so a positive generation rate adds heat "
        "to the body.");
    p.addOptional("heat_source",
                  ParameterKind::Function,
                  0.0,
                  "Volumetric heat generation rate in watts per cubic metre, as a constant or the "
                  "name of a function of (x, y, z, t). A positive value adds heat to the body. It "
                  "is a function of position and time only. A generation rate that depends on the "
                  "temperature requires a reaction kernel or a user-written kernel.");
    p.addOptional("scale_with_load",
                  ParameterKind::Boolean,
                  true,
                  "Multiply this contribution by the load factor during load stepping. This "
                  "parameter defaults to true, unlike the framework default, because the applied "
                  "loading is normally the quantity that is ramped. Set it to false for a part of "
                  "the loading that must stay fixed while the rest is increased.");
    return p;
  }
  explicit HeatSource(const InputParameters & p) : Kernel(p) {}
  void initialSetup(Problem & problem) override
  {
    Kernel::initialSetup(problem);
    _q = getFunction(problem, "heat_source");
  }
  bool hasSource() const override { return true; }
  ADReal computeSource(const QpContext & ctx) const override
  {
    return ADReal(-_q->value(ctx.x, ctx.time));
  }

private:
  FunctionPtr _q;
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
    p.addOptional("specific_heat",
                  ParameterKind::Function,
                  1.0,
                  "Specific heat capacity c_p at constant pressure, as a constant or the name "
                  "of a function.");
    return p;
  }
  explicit HeatConvection(const InputParameters & p) : Kernel(p) {}
  void initialSetup(Problem & problem) override
  {
    Kernel::initialSetup(problem);
    const auto names = _params.getStringList("velocities");
    const int dim = problem.mesh().dimension();
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
  }
  bool hasSource() const override { return true; }
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
                  "Mass density rho in kilograms per cubic metre, as a constant or the name of "
                  "a function. Only the product of the density and the specific heat enters, "
                  "so leave one of them at 1 if the volumetric heat capacity is known "
                  "directly.");
    p.addOptional("specific_heat",
                  ParameterKind::Function,
                  1.0,
                  "Specific heat capacity c_p in joules per kilogram per kelvin, as a constant "
                  "or the name of a function. See 'density': only the product enters.");
    p.addOptional("density_property",
                  ParameterKind::String,
                  std::string(""),
                  "Name of a property to use as the density in place of 'density'.");
    p.addOptional("specific_heat_property",
                  ParameterKind::String,
                  std::string(""),
                  "Name of a property to use as the specific heat in place of "
                  "'specific_heat'. A temperature-dependent specific heat is evaluated at the "
                  "new temperature, so the term is rho c_p(T) (T - T_old) / dt.");
    return p;
  }
  explicit HeatConductionTimeDerivative(const InputParameters & p) : Kernel(p) {}
  void initialSetup(Problem & problem) override
  {
    Kernel::initialSetup(problem);
    _rho = getFunction(problem, "density");
    _cp = getFunction(problem, "specific_heat");
    const auto rp = _params.getString("density_property");
    const auto cp = _params.getString("specific_heat_property");
    _rho_prop = rp.empty() ? -1 : problem.propertyRegistry().id(rp);
    _cp_prop = cp.empty() ? -1 : problem.propertyRegistry().id(cp);
  }
  bool hasSource() const override { return true; }
  bool isTimeKernel() const override { return true; }
  ADReal computeSource(const QpContext & ctx) const override
  {
    const ADReal rho =
        _rho_prop >= 0 ? ctx.property(_rho_prop) : ADReal(_rho->value(ctx.x, ctx.time));
    const ADReal cp = _cp_prop >= 0 ? ctx.property(_cp_prop) : ADReal(_cp->value(ctx.x, ctx.time));
    return rho * cp * (ctx.value(_var) - ctx.oldValue(_var)) / ctx.dt;
  }

private:
  FunctionPtr _rho, _cp;
  int _rho_prop = -1, _cp_prop = -1;
};

class ConvectiveHeatFluxBC : public IntegratedBC
{
public:
  static InputParameters validParams()
  {
    InputParameters p = IntegratedBC::validParams();
    p.setClassDescription(
        "Convection into a surrounding fluid, Newton's law of cooling: the natural boundary "
        "quantity is set to n . (k grad T) = -h (T - T_ambient), so heat leaves the body wherever "
        "the surface is hotter than the fluid. The condition is linear in the temperature and is "
        "differentiated exactly, so it converges in one Newton step for a linear conduction "
        "problem.");
    p.addRequired("heat_transfer_coefficient",
                  ParameterKind::Function,
                  "Film, or convection, coefficient h in watts per square metre per kelvin, "
                  "as a constant or the name of a function. It must be non-negative, so "
                  "that heat leaves the body wherever the surface is hotter than the "
                  "surroundings. Typical values run from a few W/m^2/K for natural "
                  "convection in air to several thousand for forced convection in water.");
    p.addOptional("ambient_temperature",
                  ParameterKind::Function,
                  0.0,
                  "Temperature of the surrounding fluid. Only the difference from the surface "
                  "temperature enters, so any consistent temperature scale may be used, whereas "
                  "radiative_heat_flux_boundary_condition requires an absolute scale. The default "
                  "0 cools the surface towards zero, which is rarely intended when the variable "
                  "is in degrees celsius.");
    deformation::addParameter(
        p,
        "the flux is per unit deformed area and is multiplied by the area ratio da/dA = |J F^-T N| "
        "to give the flux per unit undeformed area.");
    return p;
  }
  explicit ConvectiveHeatFluxBC(const InputParameters & p) : IntegratedBC(p) {}
  void initialSetup(Problem & problem) override
  {
    IntegratedBC::initialSetup(problem);
    _h = getFunction(problem, "heat_transfer_coefficient");
    _Tinf = getFunction(problem, "ambient_temperature");
    _F = deformation::propertyId(problem, _params, name());
  }
  ADReal computeBoundaryFlux(const QpContext & ctx) const override
  {
    const ADReal q =
        -_h->value(ctx.x, ctx.time) * (ctx.value(_var) - _Tinf->value(ctx.x, ctx.time));
    return _F >= 0 ? q * deformation::areaRatio(ctx, _F) : q;
  }

private:
  FunctionPtr _h, _Tinf;
  int _F = -1;
};

class HeatFluxBC : public IntegratedBC
{
public:
  static InputParameters validParams()
  {
    InputParameters p = IntegratedBC::validParams();
    p.setClassDescription(
        "A prescribed heat flux on a surface: the natural boundary quantity is set to "
        "n . (k grad T) = q, which is the heat entering the body per unit area. This is "
        "the heat-transfer spelling of Neumann_boundary_condition and carries the same sign "
        "convention.");
    p.addOptional("heat_flux",
                  ParameterKind::Function,
                  0.0,
                  "Prescribed heat flux entering the body, q = n . (k grad T), in watts per "
                  "square metre, as a constant or the name of a function. A positive value "
                  "adds heat. The default 0 is the insulated condition, which is also what a "
                  "boundary with no condition at all receives.");
    p.addOptional("scale_with_load",
                  ParameterKind::Boolean,
                  true,
                  "Multiply this contribution by the load factor during load stepping. This "
                  "parameter defaults to true, unlike the framework default, because the applied "
                  "loading is normally the quantity that is ramped. Set it to false for a part of "
                  "the loading that must stay fixed while the rest is increased.");
    deformation::addParameter(
        p,
        "the flux is per unit deformed area and is multiplied by the area ratio da/dA = |J F^-T N| "
        "to give the flux per unit undeformed area.");
    return p;
  }
  explicit HeatFluxBC(const InputParameters & p) : IntegratedBC(p) {}
  void initialSetup(Problem & problem) override
  {
    IntegratedBC::initialSetup(problem);
    _q = getFunction(problem, "heat_flux");
    _F = deformation::propertyId(problem, _params, name());
  }
  ADReal computeBoundaryFlux(const QpContext & ctx) const override
  {
    const ADReal q(_q->value(ctx.x, ctx.time));
    return _F >= 0 ? q * deformation::areaRatio(ctx, _F) : q;
  }

private:
  FunctionPtr _q;
  int _F = -1;
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
        "contact penalty is given and the gap is closed, is C_s k_m P_c / (sqrt(R) H) with the "
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
    p.addOptional("roughness_coefficient",
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
    p.addOptional("accommodation_coefficient",
                  ParameterKind::Real,
                  std::monostate{},
                  "Thermal accommodation coefficient of the gas on the surfaces, between 0 and 1. "
                  "Default: the fits of Ullman et al. for helium (0.425 - 2.3e-4 T) and xenon "
                  "(0.749 - 2.5e-4 T), interpolated by the mixture molar mass (Lanning and Hann, "
                  "BNWL-1894, 1975, Appendix B, Sect. 1).");
    p.addOptional("contact_penalty",
                  ParameterKind::Real,
                  0.0,
                  "Penalty stiffness of the mechanical contact (Pa/m), the same value as the "
                  "'penalty' of the gap_contact objects. Default 0, which disables the solid "
                  "contact conductance (for a problem without mechanics).");
    p.addOptional("meyer_hardness",
                  ParameterKind::Real,
                  std::monostate{},
                  "Meyer hardness H of the softer surface, Pa. Required with a contact_penalty.");
    p.addOptional("solid_contact_coefficient",
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
    p.addOptional("primary_conductivity",
                  ParameterKind::Function,
                  std::monostate{},
                  "Thermal conductivity of the primary body at its surface (W/m/K), for the solid "
                  "contact term. Required with a contact_penalty when no 'thermal_conductivity' "
                  "property exists.");
    p.addOptional("secondary_conductivity",
                  ParameterKind::Function,
                  std::monostate{},
                  "Thermal conductivity of the secondary body at its surface (W/m/K), for the "
                  "solid contact term. Required with a contact_penalty.");
    return p;
  }
  explicit GasGapHeatTransfer(const InputParameters & p) : InterfaceBC(p)
  {
    _Rp = p.getReal("primary_roughness");
    _Rs = p.getReal("secondary_roughness");
    if (!(_Rp >= 0.0) || !(_Rs >= 0.0))
      throw InputError("'" + name() + "': the roughnesses must not be negative.");
    _Cr = p.getReal("roughness_coefficient");
    const double ep = p.getReal("primary_emissivity"), es = p.getReal("secondary_emissivity");
    if (!(ep >= 0.0 && ep <= 1.0) || !(es >= 0.0 && es <= 1.0))
      throw InputError("'" + name() + "': the emissivities must lie between 0 and 1.");
    _F = ep > 0 && es > 0 ? 1.0 / (1.0 / ep + 1.0 / es - 1.0) : 0.0;
    if (p.isSet("accommodation_coefficient"))
    {
      _accommodation = p.getReal("accommodation_coefficient");
      if (!(_accommodation > 0.0 && _accommodation <= 1.0))
        throw InputError("'" + name() + "': accommodation_coefficient must lie in (0, 1].");
    }
    _penalty = p.getReal("contact_penalty");
    if (_penalty > 0.0)
    {
      if (!p.isSet("meyer_hardness"))
        throw InputError("'" + name() +
                         "': a contact_penalty needs 'meyer_hardness' (Pa), the Meyer hardness of "
                         "the softer surface.");
      if (!p.isSet("secondary_conductivity"))
        throw InputError("'" + name() +
                         "': a contact_penalty needs 'secondary_conductivity' (W/m/K).");
      _H = p.getReal("meyer_hardness");
      if (!(_H > 0.0))
        throw InputError("'" + name() + "': meyer_hardness must be positive.");
    }
    _Cs = p.getReal("solid_contact_coefficient");
    if (_Cs < 0.0)
      throw InputError("'" + name() + "': solid_contact_coefficient must not be negative.");
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
    _kprop = problem.propertyRegistry().has("thermal_conductivity")
                 ? problem.propertyRegistry().id("thermal_conductivity")
                 : -1;
    if (_penalty > 0.0)
    {
      if (_kprop < 0 && !_params.isSet("primary_conductivity"))
        throw InputError("'" + name() +
                         "': a contact_penalty needs 'primary_conductivity' (W/m/K) when no "
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
  f.add<HeatSource>("heat_source", ObjectCategory::Kernel, m);
  f.add<HeatConvection>("heat_convection", ObjectCategory::Kernel, m);
  f.add<HeatConductionTimeDerivative>("heat_conduction_time_derivative", ObjectCategory::Kernel, m);
  f.add<ConvectiveHeatFluxBC>(
      "convective_heat_flux_boundary_condition", ObjectCategory::BoundaryCondition, m);
  f.add<HeatFluxBC>("heat_flux_boundary_condition", ObjectCategory::BoundaryCondition, m);
  f.add<RadiativeHeatFluxBC>(
      "radiative_heat_flux_boundary_condition", ObjectCategory::BoundaryCondition, m);
  f.add<GapHeatTransfer>("gap_heat_transfer", ObjectCategory::BoundaryCondition, m);
  f.add<GasGapHeatTransfer>("gas_gap_heat_transfer", ObjectCategory::BoundaryCondition, m);
}

} // namespace dualmesh
