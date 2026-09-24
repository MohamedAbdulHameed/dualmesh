// SPDX-License-Identifier: LGPL-2.1-or-later
//
// Heat transfer module: conduction with temperature-dependent conductivity,
// volumetric heating, heat capacity, and convective/radiative/flux boundary
// conditions.  Governing equation:
//     rho c_p dT/dt - div(k(T) grad T) = q'''.
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
    for (int d = 0; d < ctx.dim; ++d)
      F[d] = k * g[d];
  }

private:
  FunctionPtr _k;
  std::vector<double> _poly;
  std::string _prop_name;
  int _prop = -1;
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
                  "depends on the temperature belongs in a Reaction kernel or in a kernel "
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
        "equations (BoussinesqBuoyancy) it couples heat transfer and fluid flow in both "
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
      throw InputError("HeatConvection '" + name() +
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
    return p;
  }
  explicit HeatConductionTimeDerivative(const InputParameters & p) : Kernel(p) {}
  void initialSetup(Problem & problem) override
  {
    Kernel::initialSetup(problem);
    _rho = getFunction(problem, "density");
    _cp = getFunction(problem, "specific_heat");
  }
  bool hasSource() const override { return true; }
  bool isTimeKernel() const override { return true; }
  ADReal computeSource(const QpContext & ctx) const override
  {
    const double c = _rho->value(ctx.x, ctx.time) * _cp->value(ctx.x, ctx.time);
    return c * (ctx.value(_var) - ctx.oldValue(_var)) / ctx.dt;
  }

private:
  FunctionPtr _rho, _cp;
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
                  "unlike RadiativeHeatFluxBC, which needs an absolute scale. The default 0 "
                  "cools the surface towards zero, which is rarely intended when the "
                  "variable is in degrees celsius.");
    return p;
  }
  explicit ConvectiveHeatFluxBC(const InputParameters & p) : IntegratedBC(p) {}
  void initialSetup(Problem & problem) override
  {
    IntegratedBC::initialSetup(problem);
    _h = getFunction(problem, "heat_transfer_coefficient");
    _Tinf = getFunction(problem, "ambient_temperature");
  }
  ADReal computeBoundaryFlux(const QpContext & ctx) const override
  {
    return -_h->value(ctx.x, ctx.time) * (ctx.value(_var) - _Tinf->value(ctx.x, ctx.time));
  }

private:
  FunctionPtr _h, _Tinf;
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
        "the heat-transfer spelling of NeumannBC and carries the same sign convention.");
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
    return p;
  }
  explicit HeatFluxBC(const InputParameters & p) : IntegratedBC(p) {}
  void initialSetup(Problem & problem) override
  {
    IntegratedBC::initialSetup(problem);
    _q = getFunction(problem, "heat_flux");
  }
  ADReal computeBoundaryFlux(const QpContext & ctx) const override
  {
    return ADReal(_q->value(ctx.x, ctx.time));
  }

private:
  FunctionPtr _q;
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
  }
  ADReal computeBoundaryFlux(const QpContext & ctx) const override
  {
    const double Ta = _Tinf->value(ctx.x, ctx.time);
    const ADReal & T = ctx.value(_var);
    if (ctx.mode == LinearizationMode::Picard)
    {
      // (T^4 - Ta^4) = (T^2 + Ta^2)(T + Ta)(T - Ta), lag the first factors.
      const ADReal & Tl = ctx.coefficientValue(_var);
      return -_coef * (Tl * Tl + Ta * Ta) * (Tl + Ta) * (T - Ta);
    }
    return -_coef * (pow(T, 4.0) - Ta * Ta * Ta * Ta);
  }

private:
  double _coef;
  FunctionPtr _Tinf;
};

} // namespace

void
registerHeatTransferObjects(Factory & f)
{
  const std::string m = "heat_transfer";
  f.add<HeatConduction>("HeatConduction", ObjectCategory::Kernel, m);
  f.add<HeatSource>("HeatSource", ObjectCategory::Kernel, m);
  f.add<HeatConvection>("HeatConvection", ObjectCategory::Kernel, m);
  f.add<HeatConductionTimeDerivative>("HeatConductionTimeDerivative", ObjectCategory::Kernel, m);
  f.add<ConvectiveHeatFluxBC>("ConvectiveHeatFluxBC", ObjectCategory::BoundaryCondition, m);
  f.add<HeatFluxBC>("HeatFluxBC", ObjectCategory::BoundaryCondition, m);
  f.add<RadiativeHeatFluxBC>("RadiativeHeatFluxBC", ObjectCategory::BoundaryCondition, m);
}

} // namespace dualmesh
