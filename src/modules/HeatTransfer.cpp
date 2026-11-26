// SPDX-License-Identifier: LGPL-2.1-or-later
//
// Heat conduction module: the objects of the energy balance rho c_p dT/dt - div(k grad T) = q'''
// that the generic kernels do not hold.  The conduction is the kernel diffusion, the heat
// capacity time_derivative, the heat that a flow carries advection and the heat source
// body_force.  This module adds the streamline stabilisation of the heat that a flow carries and
// the heat transfer across a gap.
#include "dualmesh/base/Deformation.h"
#include "dualmesh/base/Factory.h"
#include "dualmesh/base/FieldExpression.h"
#include "dualmesh/base/Kernel.h"
#include "dualmesh/base/Problem.h"
#include "dualmesh/modules/Framework.h"

#include <cmath>

namespace dualmesh
{

namespace
{

/// The Stefan-Boltzmann constant in W/(m^2 K^4), rounded to ten digits: the value of the physical
/// constant Stefan_Boltzmann_constant of the input.
constexpr double kStefanBoltzmann = 5.670374419e-8;

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
        "weighted as the assembly weights it, with the velocity v of the advection kernels. "
        "Quadratic elements give the conduction as -k Laplacian(T), without the gradient of k, "
        "and linear elements leave it out, since it vanishes inside them. The code refuses a "
        "term of the temperature that is written as a flux other than the conduction, which is "
        "the case for the conservative form of an advection. The parameter is "
        "tau = [(2 |v| / h)^2 + 9 (4 kappa / h^2)^2 + (2 / dt)^2]^(-1/2), the same function of "
        "the element size h, the speed |v|, the diffusivity and the time step dt as the "
        "parameter of momentum_stabilization, with h halved on a quadratic element, "
        "kappa = k / (rho c_p) the thermal diffusivity and dt the time step, whose term enters "
        "a transient solve only. The conductivity k is that of the diffusion kernels of the "
        "temperature, with its solution_polynomial, and 0 without one. The heat capacity "
        "rho c_p is the coefficient of the advection kernels of the temperature, which the "
        "stabilization needs. The factor 9 gives tau = h^2 / (12 kappa) where diffusion "
        "dominates. On linear elements r of the exact solution equals k Laplacian(T), so the "
        "stabilization adds an error of the order of tau k |Laplacian(T)|, which is small where "
        "convection dominates and zero for a harmonic temperature.");
    p.addRequired("velocities",
                  ParameterKind::StringList,
                  "Names of the velocity variables, one per mesh dimension, in coordinate order.");
    return p;
  }
  explicit HeatStreamlineStabilization(const InputParameters & p) : Kernel(p) {}
  void initialSetup(Problem & problem) override
  {
    Kernel::initialSetup(problem);
    _v.clear();
    for (const auto & n : _params.getStringList("velocities"))
      _v.push_back(problem.variableIndex(n));
    _r.setup(problem, variableName(), name());
    // The conductivity and the heat capacity: the diffusion and the advection kernels of the
    // temperature, matched by name, since a kernel after this one has not resolved its variable.
    _diffusion.clear();
    _advection.clear();
    for (const auto & k : problem.kernels())
    {
      if (k->variableName() != variableName())
        continue;
      if (const auto * d = dynamic_cast<const Diffusion *>(k.get()))
        _diffusion.push_back(d);
      if (const auto * a = dynamic_cast<const Advection *>(k.get()))
        _advection.push_back(a);
    }
    if (_advection.empty())
      throw InputError("heat_streamline_stabilization '" + name() + "': the temperature '" +
                       variableName() +
                       "' has no advection kernel, whose heat capacity rho c_p the "
                       "stabilization takes.");
  }
  bool threadSafe() const override { return Kernel::threadSafe() && _r.threadSafe(); }
  bool hasFlux() const override { return true; }
  bool isStabilization() const override { return true; }
  bool transportsVariable() const override { return true; }
  bool residualHasTimeDerivative() const override { return true; }
  bool needsLaplacian() const override { return true; }
  void computeFlux(const QpContext & ctx, ADVector3 & F) const override
  {
    // The assembly weights this kernel as a steady term, so r holds the steady terms at the
    // state of ctx, and the time terms with the weight that makes the time derivative enter a
    // step of the theta method with the weight 1.
    const ADReal r = _r.value(ctx, ctx.time_weight, false);
    bool transient = false;
    for (const Kernel * k : _r.terms())
      transient = transient || (ctx.dt > 0.0 && k->isTimeKernel() && k->activeOnBlock(ctx.block));
    ADReal conductivity(0.0);
    for (const Diffusion * d : _diffusion)
      if (d->activeOnBlock(ctx.block))
        conductivity += d->diffusivity(ctx);
    double capacity = 0.0;
    for (const Advection * a : _advection)
      if (a->activeOnBlock(ctx.block))
        capacity += a->coefficient(ctx).value();
    ADReal speed2(1e-300);
    for (int v : _v)
      speed2 += ctx.coefficientValue(v) * ctx.coefficientValue(v);
    const ADReal tau = stabilizationTau(_r.elementSize(ctx),
                                        speed2,
                                        ADReal(conductivity / capacity),
                                        transient ? 2.0 / ctx.dt : 0.0);
    for (std::size_t d = 0; d < _v.size(); ++d)
      F[d] = tau * ctx.coefficientValue(_v[d]) * r;
  }

private:
  std::vector<int> _v;
  StrongResidual _r;
  std::vector<const Diffusion *> _diffusion;
  std::vector<const Advection *> _advection;
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
        "conductance, sigma_SB = 5.670374419e-8 W/(m^2 K^4) the Stefan-Boltzmann constant, and "
        "e_p and e_s the emissivities of the two surfaces, whose radiation term is that of two "
        "parallel grey surfaces. The temperature is 'variable' on both sides unless "
        "'secondary_variable' names another.");
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
    _radiation = ep > 0 && es > 0 ? kStefanBoltzmann / (1.0 / ep + 1.0 / es - 1.0) : 0.0;
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
  const std::string m = "heat_conduction";
  f.add<HeatStreamlineStabilization>("heat_streamline_stabilization", ObjectCategory::Kernel, m);
  f.add<GapHeatTransfer>("gap_heat_transfer", ObjectCategory::BoundaryCondition, m);
}

} // namespace dualmesh
