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
    p.addOptional("thermal_conductivity", ParameterKind::Function, 1.0, "Conductivity k(x, t).");
    p.addOptional("thermal_conductivity_property",
                  ParameterKind::String,
                  std::string(""),
                  "Material property to use as the conductivity (overrides the value).");
    p.addOptional("temperature_polynomial",
                  ParameterKind::RealList,
                  std::vector<double>{1.0},
                  "Coefficients c0, c1, ... of the polynomial dependence of k on T.");
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
    p.setClassDescription("Volumetric heat generation q'''(x, t).");
    p.addOptional("heat_source", ParameterKind::Function, 0.0, "Heat generation rate q'''.");
    p.addOptional("scale_with_load", ParameterKind::Boolean, true, "Scale with the load factor.");
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

class HeatConductionTimeDerivative : public Kernel
{
public:
  static InputParameters validParams()
  {
    InputParameters p = Kernel::validParams();
    p.setClassDescription("Heat capacity term rho c_p dT/dt.");
    p.addOptional("density", ParameterKind::Function, 1.0, "Density rho.");
    p.addOptional("specific_heat", ParameterKind::Function, 1.0, "Specific heat c_p.");
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
    p.setClassDescription("Newton cooling: n . (k grad T) = -h (T - T_ambient).");
    p.addRequired("heat_transfer_coefficient", ParameterKind::Function, "Film coefficient h.");
    p.addOptional("ambient_temperature", ParameterKind::Function, 0.0, "Ambient temperature.");
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
    p.setClassDescription("Prescribed heat flux entering the body: n . (k grad T) = q.");
    p.addOptional("heat_flux", ParameterKind::Function, 0.0, "Heat flux entering the body.");
    p.addOptional("scale_with_load", ParameterKind::Boolean, true, "Scale with the load factor.");
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
    p.addOptional("emissivity", ParameterKind::Real, 1.0, "Surface emissivity.");
    p.addOptional("stefan_boltzmann_constant",
                  ParameterKind::Real,
                  5.670374419e-8,
                  "Stefan-Boltzmann constant (W m^-2 K^-4).");
    p.addOptional("ambient_temperature", ParameterKind::Function, 0.0, "Ambient temperature.");
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
  f.add<HeatConductionTimeDerivative>("HeatConductionTimeDerivative", ObjectCategory::Kernel, m);
  f.add<ConvectiveHeatFluxBC>("ConvectiveHeatFluxBC", ObjectCategory::BoundaryCondition, m);
  f.add<HeatFluxBC>("HeatFluxBC", ObjectCategory::BoundaryCondition, m);
  f.add<RadiativeHeatFluxBC>("RadiativeHeatFluxBC", ObjectCategory::BoundaryCondition, m);
}

} // namespace dualmesh
