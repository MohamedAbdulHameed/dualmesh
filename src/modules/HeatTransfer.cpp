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
                  "when 'thermal_conductivity_property' names a material property.");
    p.addOptional("thermal_conductivity_property",
                  ParameterKind::String,
                  std::string(""),
                  "Name of a material property to use as the conductivity in place of "
                  "'thermal_conductivity'. It replaces only that base value: "
                  "'temperature_polynomial' still multiplies it. Leave it empty to use the "
                  "value.");
    p.addOptional("temperature_polynomial",
                  ParameterKind::RealList,
                  std::vector<double>{1.0},
                  "Coefficients c0, c1, ... of a polynomial that multiplies the base "
                  "conductivity, giving k(x, t) (c0 + c1 T + c2 T^2 + ...). It multiplies, it "
                  "does not replace: the default {1} leaves the conductivity unchanged, so a "
                  "list given here must include its own constant term. The temperature is the "
                  "current iterate under Newton's method and the previous one under direct "
                  "iteration, and the polynomial is written in the same temperature units as "
                  "the variable rather than relative to a reference temperature.");
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
                  "Volumetric heat generation rate in watts per cubic metre, as a constant "
                  "or the name of a function of (x, y, z, t). A positive value adds heat to "
                  "the body. It is a function of position and time only; generation that "
                  "depends on the temperature belongs in a reaction kernel or in a kernel "
                  "of your own.");
    p.addOptional("scale_with_load",
                  ParameterKind::Boolean,
                  true,
                  "Multiply this contribution by the load factor during load stepping. Unlike "
                  "the framework default this is true, because applied loading is normally "
                  "what is ramped; set it to false for a part of the loading that must stay "
                  "fixed while the rest is increased.");
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
                  "Name of a material property to use as the density in place of 'density'.");
    p.addOptional("specific_heat_property",
                  ParameterKind::String,
                  std::string(""),
                  "Name of a material property to use as the specific heat in place of "
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
        "Convection into a surrounding fluid, Newton's law of cooling: the natural "
        "boundary quantity is set to n . (k grad T) = -h (T - T_ambient), so heat leaves "
        "the body wherever the surface is hotter than the fluid. The condition is "
        "nonlinear in nothing but the temperature itself and is differentiated exactly, "
        "so it converges in one Newton step for a linear conduction problem.");
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
                  "Temperature of the surrounding fluid. Only the difference from the "
                  "surface temperature enters, so any consistent scale works; this is "
                  "unlike radiative_heat_flux_boundary_condition, which needs an absolute scale. "
                  "The default 0 "
                  "cools the surface towards zero, which is rarely intended when the "
                  "variable is in degrees celsius.");
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
                  "Multiply this contribution by the load factor during load stepping. Unlike "
                  "the framework default this is true, because applied loading is normally "
                  "what is ramped; set it to false for a part of the loading that must stay "
                  "fixed while the rest is increased.");
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
                  "Stefan-Boltzmann constant, 5.670374419e-8 W/m^2/K^4 by default. Change "
                  "it only to work in a different system of units; it is not a fitting "
                  "parameter.");
    p.addOptional("ambient_temperature",
                  ParameterKind::Function,
                  0.0,
                  "Temperature of the surroundings, which must be on an absolute scale "
                  "because it enters as its fourth power: a value in degrees celsius gives "
                  "a silently wrong answer. The default 0 models radiation into deep "
                  "space.");
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
        "sides. For a fuel rod see FuelGapHeatTransfer, whose conductance follows the gap "
        "width and the gas.");
    p.addOptional("gap_conductance",
                  ParameterKind::Function,
                  0.0,
                  "Gap conductance h in W/m^2/K, per unit area of the primary surface, as a "
                  "constant or a function of position and time.");
    p.addOptional("primary_emissivity",
                  ParameterKind::Real,
                  0.0,
                  "Emissivity of the primary surface; radiation is included when both "
                  "emissivities are positive. Temperatures must then be absolute.");
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
}

} // namespace dualmesh
