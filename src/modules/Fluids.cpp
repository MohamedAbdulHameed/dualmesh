// SPDX-License-Identifier: LGPL-2.1-or-later
//
// Fluids module: flows of viscous incompressible fluids.
//
// Two formulations are offered.
//
// The penalty (reduced-integration) formulation of Chapter 9 of Reddy's book
// eliminates the pressure with the penalty relation P = -gamma div v, so that
// the momentum equations become, for the velocity component u_i,
//
//     rho (v . grad) u_i - div [ mu (grad u_i + (grad u)_i) + gamma (div v) e_i ]
//         - f_i = 0 .
//
// The viscous part and the penalty part are separate kernels so that the
// penalty term can be integrated with a reduced rule.
//
// The pressure-velocity formulation keeps the pressure p as an unknown:
//
//     rho (dv/dt + (v . grad) v) - div sigma - f = 0 ,
//     sigma = -p I + mu (grad v + grad v^T) ,          div v = 0 .
//
// With the same interpolation for velocity and pressure (equal order) the
// discrete problem violates the inf-sup condition and the pressure
// oscillates, so it is stabilised by the residual-based pressure term of
// Hughes, Franca and Balestra (1986) and Tezduyar (1991): the mass balance is
// written as
//
//     div [ v - tau r_m ] = 0 ,   r_m = rho (dv/dt + (v . grad) v) + grad p - f ,
//
// where r_m is the momentum residual without its viscous part.  In the
// canonical form -div F + S = 0 the mass equation has the flux
// F = -(v - tau r_m).  For the finite element method this is the Galerkin
// mass equation plus the PSPG term sum_e (tau r_m, grad q)_e; for the control
// volume methods it is the mass flux through every control volume face,
// corrected by the momentum residual, which is the idea of the momentum
// interpolation of Rhie and Chow (1983).  Because r_m vanishes for the exact
// solution, apart from the viscous term whose omission costs O(h^2), the
// stabilisation is consistent.  Taylor-Hood elements (quadratic velocity,
// linear pressure) satisfy the inf-sup condition and need no stabilisation.
#include "dualmesh/base/Factory.h"
#include "dualmesh/base/Kernel.h"
#include "dualmesh/base/Problem.h"
#include "dualmesh/base/Property.h"
#include "dualmesh/fe/Assembly.h"

#include <cmath>
#include <functional>

namespace dualmesh
{

namespace
{

std::vector<int>
resolveVelocities(Problem & problem, const std::vector<std::string> & names, int dim)
{
  if (static_cast<int>(names.size()) != dim)
    throw InputError("'velocities' needs one variable name per dimension (" + std::to_string(dim) +
                     ").");
  std::vector<int> out;
  for (const auto & n : names)
    out.push_back(problem.variableIndex(n));
  return out;
}

/// Viscous stress of a Newtonian fluid: mu (grad v + grad v^T), or its Laplacian form mu grad v.
class ViscousStress : public Kernel
{
public:
  static InputParameters validParams()
  {
    InputParameters p = Kernel::validParams();
    p.setClassDescription(
        "Viscous part of the momentum equation of one velocity component: "
        "F_i = mu (grad u_i + (grad u)_i), i.e. 2 mu du/dx and mu (du/dy + dv/dx) in two "
        "dimensions.");
    p.addRequired("velocities",
                  ParameterKind::StringList,
                  "Names of the velocity variables, exactly one per mesh dimension and in "
                  "coordinate order: (u, v) in two dimensions, (u, v, w) in three. Any other "
                  "count is an error. Every object of the fluids module must be given the same "
                  "list in the same order, because 'component' indexes into it.");
    p.addRequired("component",
                  ParameterKind::Integer,
                  "Zero-based index into 'velocities' naming which momentum equation this "
                  "instance assembles: 0 for x, 1 for y, 2 for z. It must select the same "
                  "variable as 'variable'. The range is checked but the agreement is not, so a "
                  "mismatch assembles one momentum equation into another without a warning. Add "
                  "one instance per velocity variable.");
    p.addOptional("dynamic_viscosity",
                  ParameterKind::Function,
                  1.0,
                  "Dynamic viscosity mu in pascal seconds, as a constant or the name of a "
                  "function of (x, y, z, t). Because it is evaluated from position and time only, "
                  "this kernel describes a Newtonian fluid. A shear-rate dependent viscosity "
                  "requires a user-written property object and kernel.");
    p.addOptional("form",
                  ParameterKind::String,
                  std::string("stress"),
                  "stress (the default: F_i = mu (grad u_i + (grad u)_i), whose natural boundary "
                  "condition is a zero traction, sigma n = 0) or laplacian (F_i = mu grad u_i, the "
                  "same equations for a constant viscosity and an incompressible flow, whose "
                  "natural boundary condition is the do-nothing outflow mu du/dn - p n = 0 of the "
                  "DFG benchmarks, Nek5000 and the projection time integration).");
    return p;
  }
  explicit ViscousStress(const InputParameters & p)
      : Kernel(p), _component(static_cast<int>(p.getInt("component")))
  {
    const std::string form = p.getString("form");
    if (form != "stress" && form != "laplacian")
      throw InputError("'" + name() + "': form is stress or laplacian, not '" + form + "'.");
    _laplacian = form == "laplacian";
  }
  void initialSetup(Problem & problem) override
  {
    Kernel::initialSetup(problem);
    _v =
        resolveVelocities(problem, _params.getStringList("velocities"), problem.mesh().dimension());
    _mu = getFunction(problem, "dynamic_viscosity");
    if (_component < 0 || _component >= static_cast<int>(_v.size()))
      throw InputError("'component' is outside the range of the velocity variables.");
    _hoop = problem.coordinateSystem() == CoordinateSystem::Axisymmetric && _component == 0;
  }
  bool hasFlux() const override { return true; }
  bool hasSource() const override { return _hoop; }
  void computeFlux(const QpContext & ctx, ADVector3 & F) const override
  {
    const double mu = _mu->value(ctx.x, ctx.time);
    const auto & gi = ctx.gradient(_v[_component]);
    for (int d = 0; d < ctx.dim; ++d)
      F[d] = _laplacian ? mu * gi[d] : mu * (gi[d] + ctx.gradient(_v[d])[_component]);
  }
  // In axisymmetric coordinates (r, z) without swirl, the radial equation
  // carries the hoop stress: -(div sigma)_r = -div_rz(sigma_r.) + sigma_tt / r,
  // and the viscous part of sigma_tt is 2 mu u_r / r.  The radial component of
  // the vector Laplacian carries - u_r / r^2 instead, so the Laplacian form has
  // the source mu u_r / r^2.
  ADReal computeSource(const QpContext & ctx) const override
  {
    const double r = ctx.x[0];
    if (r == 0.0)
      return ADReal(0.0);
    return ((_laplacian ? 1.0 : 2.0) * _mu->value(ctx.x, ctx.time) / (r * r)) * ctx.value(_v[0]);
  }

private:
  int _component;
  bool _hoop = false;
  bool _laplacian = false;
  std::vector<int> _v;
  FunctionPtr _mu;
};

/// Penalty term that enforces incompressibility.
class PenaltyIncompressibility : public Kernel
{
public:
  static InputParameters validParams()
  {
    InputParameters p = Kernel::validParams();
    p.setClassDescription(
        "The penalty term that replaces the pressure in the momentum equation. In the canonical "
        "form -div F + S = 0 it contributes the flux F = gamma (div v) e_i, that is, gamma (du/dx "
        "+ dv/dy + dw/dz) in the direction of this instance's component and zero in the others, "
        "with no source. It must be integrated with a reduced rule, which is the default here. A "
        "rule with more points locks the velocity field.");
    p.addRequired("velocities",
                  ParameterKind::StringList,
                  "Names of the velocity variables, exactly one per mesh dimension and in "
                  "coordinate order. Every object of the fluids module must be given the "
                  "same list in the same order, because 'component' indexes into it.");
    p.addRequired("component", ParameterKind::Integer, "Component index of this equation.");
    p.addOptional("penalty_parameter",
                  ParameterKind::Real,
                  1.0e8,
                  "Penalty parameter gamma, which enforces incompressibility through the "
                  "constitutive relation P = -gamma div v. Choose it roughly 10^4 to 10^7 times "
                  "the dynamic viscosity. A value that is too small leaves the flow measurably "
                  "compressible, and a value that is too large makes the linear system so "
                  "ill-conditioned that the direct solver loses accuracy. The default suits a "
                  "viscosity of order one. The objects penalty_incompressibility and "
                  "penalty_pressure each take their own value of gamma, and the recovered "
                  "pressure is meaningful only when the two values agree.");
    p.addOptional("quadrature",
                  ParameterKind::String,
                  std::string("midpoint"),
                  "Quadrature rule for the penalty term, which accepts the same values as the "
                  "'quadrature' parameter of the other objects. The default midpoint rule is the "
                  "one-point rule that the penalty formulation requires. A rule with more points "
                  "locks the velocity field and should be used only to demonstrate that locking.");
    p.addOptional("reduced_integration",
                  ParameterKind::Boolean,
                  true,
                  "Evaluate the velocity gradients at the centroid of the element while "
                  "integrating with the rule named by 'quadrature'. This is selective reduced "
                  "integration. It defaults to true here, unlike the framework default. Keep it "
                  "on, because turning it off locks the incompressibility constraint.");
    return p;
  }
  explicit PenaltyIncompressibility(const InputParameters & p)
      : Kernel(p), _component(static_cast<int>(p.getInt("component"))),
        _gamma(p.getReal("penalty_parameter"))
  {
  }
  void initialSetup(Problem & problem) override
  {
    Kernel::initialSetup(problem);
    // The penalty formulation works only because the constraint div v = 0 is
    // imposed once per element, at its centroid, by a reduced rule; imposed
    // more often it leaves the velocity no freedom and the flow locks.  The
    // cell-centred finite volume method has no element interior to integrate
    // over reducedly: its fluxes are evaluated at faces from reconstructed
    // cell gradients, which imposes the constraint at every face, and the
    // measured flow in a lid-driven cavity is five orders of magnitude too
    // small.  Incompressible flow on that method needs a pressure-velocity
    // coupling instead, which is not implemented, so the combination is
    // refused rather than allowed to return a locked answer.
    if (problem.method() == Method::FiniteVolumeCell)
      throw InputError(
          "penalty_incompressibility '" + name() +
          "': the penalty formulation of incompressible flow cannot be used with the "
          "cell-centred finite volume method (zfvm). It needs the incompressibility "
          "constraint to be imposed once per element by a reduced integration rule, and the "
          "cell-centred method imposes it at every face, which locks the velocity field. Use "
          "the dual mesh (dmcdm), finite element (fem) or vertex-centred finite volume (hfvm) "
          "method for incompressible flow.");
    _v =
        resolveVelocities(problem, _params.getStringList("velocities"), problem.mesh().dimension());
    _axisymmetric = problem.coordinateSystem() == CoordinateSystem::Axisymmetric;
  }
  bool hasFlux() const override { return true; }
  bool hasSource() const override { return _axisymmetric && _component == 0; }
  void computeFlux(const QpContext & ctx, ADVector3 & F) const override
  {
    const ADReal div = divergence(ctx);
    for (int d = 0; d < ctx.dim; ++d)
      F[d] = d == _component ? _gamma * div : ADReal(0.0);
  }
  // The penalty pressure enters the hoop stress of the radial equation as
  // well: S = sigma_tt / r, whose pressure part is -p = gamma div v.
  ADReal computeSource(const QpContext & ctx) const override
  {
    const double r = ctx.x[0];
    if (r == 0.0)
      return ADReal(0.0);
    return (_gamma / r) * divergence(ctx);
  }

private:
  /// div v, with the u_r / r term in axisymmetric coordinates.
  ADReal divergence(const QpContext & ctx) const
  {
    ADReal div(0.0);
    for (std::size_t d = 0; d < _v.size(); ++d)
      div += ctx.gradient(_v[d])[d];
    if (_axisymmetric && ctx.x[0] != 0.0)
      div += ctx.value(_v[0]) / ctx.x[0];
    return div;
  }
  int _component;
  double _gamma;
  bool _axisymmetric = false;
  std::vector<int> _v;
};

/// Convective (inertia) term of the Navier-Stokes equations.
class ConvectiveInertia : public Kernel
{
public:
  static InputParameters validParams()
  {
    InputParameters p = Kernel::validParams();
    p.setClassDescription(
        "Convective term rho (v . grad) u_i of the Navier-Stokes equations, added as a source. "
        "Under Picard iteration the transporting velocity is taken from the previous iterate, "
        "which is the linearization used in Reddy's book.");
    p.addRequired("velocities",
                  ParameterKind::StringList,
                  "Names of the velocity variables, exactly one per mesh dimension and in "
                  "coordinate order. Every object of the fluids module must be given the same list "
                  "in the same order, because 'component' indexes into it.");
    p.addRequired("component", ParameterKind::Integer, "Component index of this equation.");
    p.addOptional("density",
                  ParameterKind::Function,
                  1.0,
                  "Mass density rho in kilograms per cubic metre, as a constant or the name of a "
                  "function. Together with the dynamic viscosity it fixes the Reynolds number. A "
                  "Stokes flow is described by omitting this kernel altogether.");
    return p;
  }
  explicit ConvectiveInertia(const InputParameters & p)
      : Kernel(p), _component(static_cast<int>(p.getInt("component")))
  {
  }
  void initialSetup(Problem & problem) override
  {
    Kernel::initialSetup(problem);
    _v =
        resolveVelocities(problem, _params.getStringList("velocities"), problem.mesh().dimension());
    _rho = getFunction(problem, "density");
  }
  bool hasSource() const override { return true; }
  ADReal computeSource(const QpContext & ctx) const override
  {
    const double rho = _rho->value(ctx.x, ctx.time);
    const auto & gi = ctx.gradient(_v[_component]);
    ADReal s(0.0);
    for (std::size_t d = 0; d < _v.size(); ++d)
      s += ctx.coefficientValue(_v[d]) * gi[d];
    return rho * s;
  }

private:
  int _component;
  std::vector<int> _v;
  FunctionPtr _rho;
};

/// The buoyancy force of the Boussinesq approximation.
class BoussinesqBuoyancy : public Kernel
{
public:
  static InputParameters validParams()
  {
    InputParameters p = Kernel::validParams();
    p.setClassDescription(
        "Buoyancy in the Boussinesq approximation, for the momentum equation of one velocity "
        "component. The density is taken as rho0 everywhere except in the gravity term, where rho "
        "= rho0 (1 - beta (T - T0)). The constant part rho0 g is balanced by the hydrostatic "
        "pressure, which leaves the body force f_i = -rho0 beta (T - T0) g_i, so that fluid "
        "warmer than T0 rises. In the canonical form the source is S = -f_i. The temperature is "
        "an unknown of the same problem, and the coupling is exact in Newton's method.");
    p.addRequired("temperature",
                  ParameterKind::String,
                  "Name of the temperature variable. It must already exist on the problem.");
    p.addRequired("component",
                  ParameterKind::Integer,
                  "Component index of this momentum equation: 0 for x, 1 for y, 2 for z.");
    p.addRequired("gravity",
                  ParameterKind::RealList,
                  "The gravitational acceleration vector g, for instance (0, -9.81) in two "
                  "dimensions with y upward, in metres per second squared.");
    p.addOptional("density",
                  ParameterKind::Function,
                  1.0,
                  "Reference density rho0, kg/m^3. Default 1, for problems written in "
                  "dimensionless form.");
    p.addRequired("thermal_expansion_coefficient",
                  ParameterKind::Function,
                  "Volumetric thermal expansion coefficient beta, 1/K. For an ideal gas it "
                  "is 1/T0.");
    p.addOptional("reference_temperature",
                  ParameterKind::Real,
                  0.0,
                  "Temperature T0 at which the density is rho0.");
    return p;
  }
  explicit BoussinesqBuoyancy(const InputParameters & p)
      : Kernel(p), _component(static_cast<int>(p.getInt("component"))),
        _t0(p.getReal("reference_temperature"))
  {
  }
  void initialSetup(Problem & problem) override
  {
    Kernel::initialSetup(problem);
    _t = coupledVariable(problem, "temperature");
    const auto g = _params.getRealList("gravity");
    if (_component < 0 || _component >= problem.mesh().dimension())
      throw InputError("Boussinesq_buoyancy '" + name() + "': 'component' must be between 0 and " +
                       std::to_string(problem.mesh().dimension() - 1) + ".");
    _g = _component < static_cast<int>(g.size()) ? g[_component] : 0.0;
    _rho = getFunction(problem, "density");
    _beta = getFunction(problem, "thermal_expansion_coefficient");
  }
  bool hasSource() const override { return true; }
  ADReal computeSource(const QpContext & ctx) const override
  {
    // S = -f_i = rho0 beta (T - T0) g_i
    const double c = _rho->value(ctx.x, ctx.time) * _beta->value(ctx.x, ctx.time) * _g;
    return c * (ctx.value(_t) - _t0);
  }

private:
  int _component;
  double _t0;
  int _t = -1;
  double _g = 0.0;
  FunctionPtr _rho, _beta;
};

// ---------------------------------------------------------------------------
// The pressure-velocity formulation
// ---------------------------------------------------------------------------

/// The parameters shared by the objects that need the momentum residual.
void
addMomentumResidualParams(InputParameters & p)
{
  p.addRequired("velocities",
                ParameterKind::StringList,
                "Names of the velocity variables, exactly one per mesh dimension and in "
                "coordinate order.");
  p.addOptional("density",
                ParameterKind::Function,
                1.0,
                "Mass density rho, as a constant or a function. It must be the value given to the "
                "inertia kernels of the momentum equations, and zero describes a Stokes flow.");
  p.addOptional("dynamic_viscosity",
                ParameterKind::Function,
                1.0,
                "Dynamic viscosity mu, the value given to viscous_stress. It enters the "
                "stabilisation parameter only.");
  p.addOptional("body_force",
                ParameterKind::StringList,
                std::vector<std::string>{},
                "The body force f per unit volume that the momentum equations carry, one entry "
                "per component, each a number, an expression in x, y, z and t, or the name of a "
                "function. The stabilisation needs the whole momentum residual, so every force "
                "added to the momentum equations must be repeated here. A force omitted here "
                "produces a spurious stabilisation term that opposes that force.");
  p.addOptional("temperature",
                ParameterKind::String,
                std::string(),
                "For a Boussinesq buoyancy force in the momentum equations: the temperature "
                "variable. The force -rho beta (T - T0) g is then part of the momentum "
                "residual, exactly as Boussinesq_buoyancy adds it.");
  p.addOptional("gravity",
                ParameterKind::RealList,
                std::vector<double>{},
                "The gravitational acceleration vector g of the Boussinesq force.");
  p.addOptional("thermal_expansion_coefficient",
                ParameterKind::Function,
                0.0,
                "The volumetric expansion coefficient beta of the Boussinesq force, 1/K. "
                "Default 0 (no buoyancy in the stabilisation).");
  p.addOptional("reference_temperature",
                ParameterKind::Real,
                0.0,
                "The reference temperature T0 of the Boussinesq force.");
  p.addOptional("buoyancy_density",
                ParameterKind::Real,
                -1.0,
                "The density in the Boussinesq force, when it differs from 'density': a Stokes "
                "flow has no inertia (density 0) but may still be driven by buoyancy. A negative "
                "value, the default, means 'density'.");
  p.addOptional("body_force_scale_with_load",
                ParameterKind::Boolean,
                false,
                "Multiply 'body_force' by the load factor, as the body_force kernels carrying it "
                "are when their own 'scale_with_load' is true. The two must agree, or the "
                "stabilisation sees a different force from the momentum equations at every "
                "load step but the last.");
  p.addOptional("buoyancy_scale_with_load",
                ParameterKind::Boolean,
                false,
                "Multiply the Boussinesq force by the load factor, as Boussinesq_buoyancy does "
                "when its own 'scale_with_load' is true. The two settings must agree.");
}

/// The strong momentum residual without its viscous part,
///
///   r_m = rho (du/dt + (u . grad) u) + grad p - f ,
///
/// and the stabilisation parameter of Tezduyar (1991), written per unit density
/// so that it also covers a Stokes flow (rho = 0):
///
///   tau = [ (2 rho / dt)^2 + (2 rho |u| / h)^2 + (4 mu / h^2)^2 ]^(-1/2) ,
///
/// whose three terms are the reciprocal time scales of the unsteadiness, the
/// convection and the diffusion across an element of size h.  tau r_m has the
/// units of a velocity.  For a quadratic element h is half the element size,
/// because the interpolation resolves half the element.  The viscous term of
/// r_m needs second derivatives, which the kernels do not have.  For linear
/// elements it vanishes (simplices, and bilinear elements on parallelograms)
/// or nearly so, and omitting it keeps the optimal first order in the energy
/// norm.  For quadratic elements the omission is an inconsistency that limits
/// the velocity to second order in L2; Taylor-Hood elements, which need no
/// stabilisation, are the quadratic alternative.
class MomentumResidual
{
public:
  /// @param function resolves a Function-valued parameter of the owning object.
  void setup(Problem & problem,
             const std::string & owner,
             const InputParameters & p,
             int pressure,
             const std::function<FunctionPtr(const std::string &)> & function)
  {
    const int dim = problem.mesh().dimension();
    _v = resolveVelocities(problem, p.getStringList("velocities"), dim);
    _p = pressure;
    _rho = function("density");
    _mu = function("dynamic_viscosity");
    _f.clear();
    const auto forces = p.getStringList("body_force");
    if (!forces.empty() && static_cast<int>(forces.size()) != dim)
      throw InputError("'" + owner + "': 'body_force' needs one entry per dimension (" +
                       std::to_string(dim) + ").");
    for (const auto & f : forces)
      _f.push_back(problem.function(f));
    const auto t = p.getString("temperature");
    _t = t.empty() ? -1 : problem.variableIndex(t);
    _g = p.getRealList("gravity");
    _g.resize(3, 0.0);
    _beta = function("thermal_expansion_coefficient");
    _t0 = p.getReal("reference_temperature");
    _force_load = p.getBool("body_force_scale_with_load");
    _buoyancy_load = p.getBool("buoyancy_scale_with_load");
    _buoyancy_rho = p.getReal("buoyancy_density");
    if (_t >= 0 && _g[0] == 0.0 && _g[1] == 0.0 && _g[2] == 0.0)
      throw InputError("'" + owner +
                       "': a 'temperature' was given for the buoyancy "
                       "force but no 'gravity'.");

    const auto & mesh = problem.mesh();
    _h.resize(mesh.numElements());
    for (Index e = 0; e < mesh.numElements(); ++e)
      _h[e] = elementSize(mesh, e);
  }

  int dimension() const { return static_cast<int>(_v.size()); }
  int velocity(int i) const { return _v[i]; }
  /// Whether every body force may be evaluated from several threads.
  bool threadSafe() const
  {
    for (const auto & f : _f)
      if (!f->threadSafe())
        return false;
    return true;
  }

  ADReal tau(const QpContext & ctx) const
  {
    const double rho = _rho->value(ctx.x, ctx.time);
    const double mu = _mu->value(ctx.x, ctx.time);
    const double h = ctx.element >= 0 ? _h[ctx.element] : 1.0;
    ADReal speed2(1e-300);
    for (int v : _v)
      speed2 += ctx.coefficientValue(v) * ctx.coefficientValue(v);
    ADReal inverse2 = (4.0 * rho * rho / (h * h)) * speed2;
    const double viscous = 4.0 * mu / (h * h);
    inverse2 += viscous * viscous;
    if (ctx.dt > 0)
    {
      const double unsteady = 2.0 * rho / ctx.dt;
      inverse2 += unsteady * unsteady;
    }
    return 1.0 / sqrt(inverse2);
  }

  /// Component i of r_m.
  ADReal residual(const QpContext & ctx, int i) const
  {
    const double rho = _rho->value(ctx.x, ctx.time);
    ADReal r = ctx.gradient(_p)[i];
    if (rho != 0.0)
    {
      const auto & g = ctx.gradient(_v[i]);
      ADReal convection(0.0);
      for (std::size_t d = 0; d < _v.size(); ++d)
        convection += ctx.coefficientValue(_v[d]) * g[d];
      r += rho * convection;
      if (ctx.dt > 0)
        r += (rho / ctx.dt) * (ctx.value(_v[i]) - ctx.oldValue(_v[i]));
    }
    if (!_f.empty())
      r -= (_force_load ? ctx.load_factor : 1.0) * _f[i]->value(ctx.x, ctx.time);
    if (_t >= 0 && _g[i] != 0.0)
    {
      // f_i = -rho beta (T - T0) g_i, so r_m gains + rho beta (T - T0) g_i.
      const double rho_b = _buoyancy_rho >= 0.0 ? _buoyancy_rho : _rho->value(ctx.x, ctx.time);
      const double c = rho_b * _beta->value(ctx.x, ctx.time) * _g[i];
      r += (_buoyancy_load ? ctx.load_factor : 1.0) * c * (ctx.value(_t) - _t0);
    }
    return r;
  }

  /// The dynamic viscosity at the point.
  double viscosity(const QpContext & ctx) const { return _mu->value(ctx.x, ctx.time); }

private:
  std::vector<int> _v;
  int _p = -1;
  FunctionPtr _rho, _mu, _beta;
  std::vector<FunctionPtr> _f;
  int _t = -1;
  std::vector<double> _g;
  double _t0 = 0.0;
  bool _force_load = false, _buoyancy_load = false;
  double _buoyancy_rho = -1.0;
  std::vector<double> _h;
};

/// The pressure term of a momentum equation, in flux form.
class PressureGradient : public Kernel
{
public:
  static InputParameters validParams()
  {
    InputParameters p = Kernel::validParams();
    p.setClassDescription(
        "The pressure term of the momentum equation of one velocity component, as the flux "
        "F = -p e_i, which is the pressure part of the stress row sigma_i. Integrating it by "
        "parts makes the natural boundary quantity of the momentum equation the traction "
        "sigma n, so a boundary with no condition is traction free (an open outlet). In "
        "axisymmetric coordinates the radial equation also receives the hoop term -p / r.");
    p.addRequired("pressure", ParameterKind::String, "Name of the pressure variable.");
    p.addRequired("component",
                  ParameterKind::Integer,
                  "Component index of this momentum equation: 0 for x (or r), 1 for y (or z), "
                  "2 for z.");
    return p;
  }
  explicit PressureGradient(const InputParameters & p)
      : Kernel(p), _component(static_cast<int>(p.getInt("component")))
  {
  }
  void initialSetup(Problem & problem) override
  {
    Kernel::initialSetup(problem);
    _p = coupledVariable(problem, "pressure");
    if (_component < 0 || _component >= problem.mesh().dimension())
      throw InputError("pressure_gradient '" + name() + "': 'component' must be between 0 and " +
                       std::to_string(problem.mesh().dimension() - 1) + ".");
    _hoop = problem.coordinateSystem() == CoordinateSystem::Axisymmetric && _component == 0;
  }
  bool hasFlux() const override { return true; }
  bool hasSource() const override { return _hoop; }
  void computeFlux(const QpContext & ctx, ADVector3 & F) const override
  {
    F[_component] = -ctx.value(_p);
  }
  ADReal computeSource(const QpContext & ctx) const override
  {
    const double r = ctx.x[0];
    if (r == 0.0)
      return ADReal(0.0);
    return (-1.0 / r) * ctx.value(_p);
  }

private:
  int _component;
  int _p = -1;
  bool _hoop = false;
};

/// Conservation of mass for an incompressible flow, on the pressure variable.
class MassConservation : public Kernel
{
public:
  static InputParameters validParams()
  {
    InputParameters p = Kernel::validParams();
    p.setClassDescription(
        "Conservation of mass of an incompressible flow, div v = 0, assembled on the pressure "
        "variable as the flux F = -(v - tau r_m). The term tau r_m is the residual-based pressure "
        "stabilisation (PSPG for the finite element method, momentum interpolation for the "
        "control volume methods) that makes equal-order velocity and pressure stable. Here r_m is "
        "the momentum residual without its viscous part and tau is the parameter of Tezduyar. Set "
        "'stabilization' to false for Taylor-Hood elements, which are stable without it. The "
        "boundary condition mass_flux_boundary_condition must be applied on every boundary where "
        "the pressure is not prescribed, and the incompressible_flow physics applies it there.");
    addMomentumResidualParams(p);
    p.addOptional("stabilization",
                  ParameterKind::Boolean,
                  true,
                  "Add the residual-based pressure stabilisation. It is required for equal-order "
                  "interpolation and should be set to false for Taylor-Hood elements.");
    return p;
  }
  explicit MassConservation(const InputParameters & p)
      : Kernel(p), _stabilize(p.getBool("stabilization"))
  {
  }
  void initialSetup(Problem & problem) override
  {
    Kernel::initialSetup(problem);
    _r.setup(problem,
             name(),
             _params,
             variable(),
             [&](const std::string & n) { return getFunction(problem, n); });
    // Without the stabilisation only an inf-sup stable pair gives a unique
    // pressure; equal-order interpolation then has spurious pressure modes
    // (the checkerboard of Q1-Q1), and the solve fails or returns them.
    const bool taylor_hood =
        problem.hasMixedOrder() && problem.variable(variable()).order == VariableOrder::First;
    if (!_stabilize && !taylor_hood)
      throw InputError(
          "mass_conservation '" + name() + "': 'stabilization' is false, but the pressure '" +
          problem.variable(variable()).name +
          "' is interpolated like the velocity. Equal-order velocity and pressure need the "
          "stabilisation; without it the pressure has spurious modes. Either keep "
          "stabilization = true, or use the Taylor-Hood element: a quadratic mesh, method "
          "'fem', and the pressure declared with order 'first'.");
  }
  bool threadSafe() const override { return Kernel::threadSafe() && _r.threadSafe(); }
  bool hasFlux() const override { return true; }
  bool schurMassCoefficient(const QpContext & ctx, double & coefficient) const override
  {
    const double mu = _r.viscosity(ctx);
    if (!(mu > 0.0))
      throw InputError("mass_conservation '" + name() +
                       "': the pressure mass matrix preconditioner scales the mass matrix by "
                       "1 / dynamic_viscosity, which needs a positive viscosity.");
    coefficient = 1.0 / mu;
    return true;
  }
  void computeFlux(const QpContext & ctx, ADVector3 & F) const override
  {
    const int dim = _r.dimension();
    if (!_stabilize)
    {
      for (int d = 0; d < dim; ++d)
        F[d] = -ctx.value(_r.velocity(d));
      return;
    }
    const ADReal tau = _r.tau(ctx);
    for (int d = 0; d < dim; ++d)
      F[d] = tau * _r.residual(ctx, d) - ctx.value(_r.velocity(d));
  }

private:
  bool _stabilize;
  MomentumResidual _r;
};

/// Streamline-upwind/Petrov-Galerkin stabilisation of a momentum equation.
class MomentumStabilization : public Kernel
{
public:
  static InputParameters validParams()
  {
    InputParameters p = Kernel::validParams();
    p.setClassDescription(
        "Streamline-upwind stabilisation (SUPG, Brooks and Hughes 1982) of the momentum equation "
        "of one velocity component, as the flux F = rho tau v (r_m)_i. For the finite element "
        "method this is the SUPG term sum_e (rho tau (v . grad) w, r_m)_e, and for the control "
        "volume methods it is a streamline diffusion across the control volume faces. It is "
        "consistent, because r_m vanishes for the exact solution, and it suppresses the spurious "
        "oscillations of a convection-dominated flow on a coarse mesh.");
    addMomentumResidualParams(p);
    p.addRequired("pressure", ParameterKind::String, "Name of the pressure variable.");
    p.addRequired("component", ParameterKind::Integer, "Component index of this equation.");
    return p;
  }
  explicit MomentumStabilization(const InputParameters & p)
      : Kernel(p), _component(static_cast<int>(p.getInt("component")))
  {
  }
  void initialSetup(Problem & problem) override
  {
    Kernel::initialSetup(problem);
    _r.setup(problem,
             name(),
             _params,
             coupledVariable(problem, "pressure"),
             [&](const std::string & n) { return getFunction(problem, n); });
    if (_component < 0 || _component >= _r.dimension())
      throw InputError("momentum_stabilization '" + name() + "': 'component' is out of range.");
    _rho = getFunction(problem, "density");
  }
  bool threadSafe() const override { return Kernel::threadSafe() && _r.threadSafe(); }
  bool hasFlux() const override { return true; }
  void computeFlux(const QpContext & ctx, ADVector3 & F) const override
  {
    const double rho = _rho->value(ctx.x, ctx.time);
    if (rho == 0.0)
      return;
    const ADReal c = rho * _r.tau(ctx) * _r.residual(ctx, _component);
    for (int d = 0; d < _r.dimension(); ++d)
      F[d] = c * ctx.coefficientValue(_r.velocity(d));
  }

private:
  int _component;
  MomentumResidual _r;
  FunctionPtr _rho;
};

/// The mass flux through the boundary, for the mass conservation equation.
class MassFluxBC : public IntegratedBC
{
public:
  static InputParameters validParams()
  {
    InputParameters p = IntegratedBC::validParams();
    p.setClassDescription(
        "The natural boundary quantity of the mass conservation equation: n . F = -v . n, the "
        "volume flux leaving through the boundary, evaluated with the current velocity. It "
        "closes the mass balance of the control volumes and elements that touch the "
        "boundary, and it states that the stabilising part of the mass flux has no normal "
        "component there. Apply it on every boundary where the pressure is not prescribed: "
        "walls, inlets and outlets alike.");
    p.addRequired("velocities",
                  ParameterKind::StringList,
                  "Names of the velocity variables, exactly one per mesh dimension and in "
                  "coordinate order.");
    return p;
  }
  explicit MassFluxBC(const InputParameters & p) : IntegratedBC(p) {}
  void initialSetup(Problem & problem) override
  {
    IntegratedBC::initialSetup(problem);
    _v =
        resolveVelocities(problem, _params.getStringList("velocities"), problem.mesh().dimension());
  }
  ADReal computeBoundaryFlux(const QpContext & ctx) const override
  {
    ADReal q(0.0);
    for (std::size_t d = 0; d < _v.size(); ++d)
      q -= ctx.value(_v[d]) * ctx.normal[d];
    return q;
  }

private:
  std::vector<int> _v;
};

/// Pressure recovered from the penalty relation (for post-processing).
class PenaltyPressure : public Property
{
public:
  static InputParameters validParams()
  {
    InputParameters p = Property::validParams();
    p.setClassDescription(
        "Declares the property 'pressure' = -gamma div v, the pressure recovered "
        "from the penalty relation. Evaluate it at element centroids (the reduced-order "
        "points), where it is most accurate.");
    p.addRequired("velocities",
                  ParameterKind::StringList,
                  "Names of the velocity variables, exactly one per mesh dimension and in "
                  "coordinate order. Every object of the fluids module must be given the same list "
                  "in the same order, because 'component' indexes into it.");
    p.addOptional(
        "penalty_parameter",
        ParameterKind::Real,
        1.0e8,
        "Penalty parameter gamma of the relation P = -gamma div v. It must be exactly the value "
        "given to penalty_incompressibility, or the recovered pressure is meaningless.");
    return p;
  }
  explicit PenaltyPressure(const InputParameters & p)
      : Property(p), _gamma(p.getReal("penalty_parameter"))
  {
  }
  void initialSetup(Problem & problem) override
  {
    Property::initialSetup(problem);
    _v =
        resolveVelocities(problem, _params.getStringList("velocities"), problem.mesh().dimension());
    _axisymmetric = problem.coordinateSystem() == CoordinateSystem::Axisymmetric;
  }
  void declareProperties(PropertyRegistry & r) override { _pressure = r.declare("pressure", 1); }
  void computeProperties(QpContext & ctx) const override
  {
    ADReal divergence(0.0);
    for (std::size_t d = 0; d < _v.size(); ++d)
      divergence += ctx.gradient(_v[d])[d];
    if (_axisymmetric && ctx.x[0] != 0.0)
      divergence += ctx.value(_v[0]) / ctx.x[0];
    ctx.property(_pressure) = -_gamma * divergence;
  }

private:
  double _gamma;
  bool _axisymmetric = false;
  std::vector<int> _v;
  int _pressure = -1;
};

} // namespace

void
registerFluidObjects(Factory & f)
{
  const std::string m = "fluids";
  f.add<ViscousStress>("viscous_stress", ObjectCategory::Kernel, m);
  f.add<PenaltyIncompressibility>("penalty_incompressibility", ObjectCategory::Kernel, m);
  f.add<ConvectiveInertia>("convective_inertia", ObjectCategory::Kernel, m);
  f.add<BoussinesqBuoyancy>("Boussinesq_buoyancy", ObjectCategory::Kernel, m);
  f.add<PressureGradient>("pressure_gradient", ObjectCategory::Kernel, m);
  f.add<MassConservation>("mass_conservation", ObjectCategory::Kernel, m);
  f.add<MomentumStabilization>("momentum_stabilization", ObjectCategory::Kernel, m);
  f.add<MassFluxBC>("mass_flux_boundary_condition", ObjectCategory::BoundaryCondition, m);
  f.add<PenaltyPressure>("penalty_pressure", ObjectCategory::Property, m);
}

} // namespace dualmesh
