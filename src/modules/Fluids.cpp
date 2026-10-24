// SPDX-License-Identifier: LGPL-2.1-or-later
//
// Fluids module: flows of viscous incompressible fluids.
//
// Two formulations are offered.
//
// The penalty (reduced-integration) formulation eliminates the pressure with the penalty relation P = -gamma div v, so that
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

#include <algorithm>
#include <cmath>
#include <functional>

namespace dualmesh
{

namespace
{

std::vector<int>
resolveVelocities(Problem & problem, const std::vector<std::string> & names, int dim)
{
  if (problem.mesh().spaceDimension() > problem.mesh().dimension())
    throw InputError("Flow on a surface or a curve in space (a mesh of lower dimension than its "
                     "space) is not available.");
  if (static_cast<int>(names.size()) != dim)
    throw InputError("'velocities' needs one variable name per dimension (" + std::to_string(dim) +
                     ").");
  std::vector<int> out;
  for (const auto & n : names)
    out.push_back(problem.variableIndex(n));
  return out;
}

/// The component of the momentum equation of @p variable: its place in 'velocities'.
int
componentOf(const std::string & owner,
            const std::string & variable,
            const std::vector<std::string> & velocities)
{
  const auto it = std::find(velocities.begin(), velocities.end(), variable);
  if (it == velocities.end())
  {
    std::string list;
    for (const auto & v : velocities)
      list += (list.empty() ? "" : ", ") + v;
    throw InputError("'" + owner + "': the variable '" + variable +
                     "' is not one of the velocities (" + list +
                     "), so this term has no momentum equation to act on.");
  }
  return static_cast<int>(it - velocities.begin());
}

/// A Function-valued parameter of another object: a constant, a function name or a function.
FunctionPtr
functionParameter(const Problem & problem, const InputParameters & p, const std::string & name)
{
  const auto & raw = p.getRaw(name);
  if (const auto * d = std::get_if<double>(&raw))
    return std::make_shared<ConstantFunction>(*d);
  if (const auto * l = std::get_if<long>(&raw))
    return std::make_shared<ConstantFunction>(double(*l));
  if (const auto * s = std::get_if<std::string>(&raw))
    return problem.function(*s);
  if (const auto * f = std::get_if<std::shared_ptr<Function>>(&raw))
    return *f;
  return std::make_shared<ConstantFunction>(0.0);
}

/// The parameter that lists the velocities, shared by every object of the module.
void
addVelocitiesParam(InputParameters & p)
{
  p.addRequired("velocities",
                ParameterKind::StringList,
                "Names of the velocity variables, exactly one per mesh dimension and in "
                "coordinate order: (u, v) in two dimensions, (u, v, w) in three. The momentum "
                "equation of 'variable' is the one of its place in this list.");
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
    addVelocitiesParam(p);
    p.addOptional("dynamic_viscosity",
                  ParameterKind::Function,
                  1.0,
                  "Dynamic viscosity mu in pascal seconds, as a constant or the name of a "
                  "function of (x, y, z, t). Because it is evaluated from position and time only, "
                  "this kernel describes a Newtonian fluid. A shear-rate dependent viscosity "
                  "requires a user-written property object and kernel.");
    p.setRange("dynamic_viscosity", ParameterRange::Positive);
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
      : Kernel(p), _component(componentOf(name(), variableName(), p.getStringList("velocities")))
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
    addVelocitiesParam(p);
    p.addOptional("penalty_parameter",
                  ParameterKind::Real,
                  1.0e8,
                  "Penalty parameter gamma, which enforces incompressibility through the "
                  "constitutive relation P = -gamma div v. Choose it roughly 10^4 to 10^7 times "
                  "the dynamic viscosity. A value that is too small leaves the flow measurably "
                  "compressible, and a value that is too large makes the linear system so "
                  "ill-conditioned that the direct solver loses accuracy. The default suits a "
                  "viscosity of order one. penalty_pressure reads it from here.");
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
      : Kernel(p), _component(componentOf(name(), variableName(), p.getStringList("velocities"))),
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
        "Under Picard iteration the transporting velocity is taken from the previous iterate.");
    addVelocitiesParam(p);
    p.addOptional("density",
                  ParameterKind::Function,
                  1.0,
                  "Mass density rho in kilograms per cubic metre, as a constant or the name of a "
                  "function. Together with the dynamic viscosity it fixes the Reynolds number. A "
                  "Stokes flow is described by omitting this kernel altogether.");
    p.setRange("density", ParameterRange::NonNegative);
    addAdvectionSchemeParam(p);
    return p;
  }
  explicit ConvectiveInertia(const InputParameters & p)
      : Kernel(p), _component(componentOf(name(), variableName(), p.getStringList("velocities"))),
        _scheme_text(p.getString("advection_scheme")),
        _face_flux(_scheme_text == "upwind" || _scheme_text == "linear_upwind" ||
                   _scheme_text == "limited")
  {
  }
  void initialSetup(Problem & problem) override
  {
    Kernel::initialSetup(problem);
    _v =
        resolveVelocities(problem, _params.getStringList("velocities"), problem.mesh().dimension());
    _rho = getFunction(problem, "density");
    _scheme = advectionScheme(problem, _scheme_text, name());
    _mass = _face_flux ? massConservationKernel(problem) : nullptr;
    _cells = problem.isCellCentered();
  }
  // The term is the source rho (v . grad) u_i, or with an advection scheme of a control volume
  // method a face flux with the face value u_i,f that the scheme takes from the two sides: the
  // conservative rho (w . n) u_i,f with the velocity w that the mass equation conserves, so that
  // the momentum is conserved exactly; or, without a mass equation (the penalty formulation),
  // the non-conservative rho (v . n)(u_i,f - u_i,C) for the control volume C on each side.
  bool hasSource() const override { return !_face_flux; }
  bool hasFlux() const override { return _face_flux; }
  bool usesFaceSides() const override { return _face_flux; }
  bool nonConservativeFaceFlux() const override { return _face_flux && !_mass; }
  bool isInertia() const override { return true; }
  bool hasKinks() const override { return _scheme == AdvectionScheme::Limited; }
  bool needsSideGradients() const override
  {
    return _scheme == AdvectionScheme::LinearUpwind || _scheme == AdvectionScheme::Limited;
  }
  bool advectingNormalVelocity(const QpContext & ctx, double & velocity) const override
  {
    if (!_face_flux)
      return false;
    const ADVector3 w = advectingVelocity(ctx, _mass, _cells, _v);
    velocity = 0.0;
    for (std::size_t d = 0; d < _v.size(); ++d)
      velocity += w[d].value() * ctx.normal[d];
    return true;
  }
  void computeFlux(const QpContext & ctx, ADVector3 & F) const override
  {
    const ADVector3 w = advectingVelocity(ctx, _mass, _cells, _v);
    double velocity = 0.0;
    for (std::size_t d = 0; d < _v.size(); ++d)
      velocity += w[d].value() * ctx.normal[d];
    const ADReal ui =
        ctx.advectedValue(_v[_component], velocity, _scheme) - ctx.referenceValue(_v[_component]);
    const double rho = ctx.inertia * _rho->value(ctx.x, ctx.time);
    for (std::size_t d = 0; d < _v.size(); ++d)
      F[d] = -rho * w[d] * ui;
  }
  ADReal computeSource(const QpContext & ctx) const override
  {
    const double rho = ctx.inertia * _rho->value(ctx.x, ctx.time);
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
  std::string _scheme_text;
  bool _face_flux;
  AdvectionScheme _scheme = AdvectionScheme::Central;
  /// The mass equation whose conserved velocity a face flux carries, or nullptr.
  const Kernel * _mass = nullptr;
  bool _cells = false;
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
    addVelocitiesParam(p);
    p.addRequired("gravity",
                  ParameterKind::RealList,
                  "The gravitational acceleration vector g, for instance (0, -9.81) in two "
                  "dimensions with y upward, in metres per second squared.");
    p.addOptional("density",
                  ParameterKind::Function,
                  1.0,
                  "Reference density rho0, kg/m^3. Default 1, for problems written in "
                  "dimensionless form.");
    p.setRange("density", ParameterRange::NonNegative);
    p.addRequired("vol_expansion_coeff",
                  ParameterKind::Function,
                  "Volumetric thermal expansion coefficient beta = -(1/rho) d rho/dT, 1/K, the "
                  "relative change of volume per kelvin. For an ideal gas it is 1/T0.");
    p.addOptional("reference_temperature",
                  ParameterKind::Real,
                  0.0,
                  "Temperature T0 at which the density is rho0.");
    return p;
  }
  explicit BoussinesqBuoyancy(const InputParameters & p)
      : Kernel(p), _component(componentOf(name(), variableName(), p.getStringList("velocities"))),
        _t0(p.getReal("reference_temperature"))
  {
  }
  void initialSetup(Problem & problem) override
  {
    Kernel::initialSetup(problem);
    _t = coupledVariable(problem, "temperature");
    const auto g = _params.getRealList("gravity");
    _g = _component < static_cast<int>(g.size()) ? g[_component] : 0.0;
    _rho = getFunction(problem, "density");
    _beta = getFunction(problem, "vol_expansion_coeff");
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
  /// Read the density, the viscosity, the body forces and the buoyancy from the momentum kernels
  /// of @p velocities that the problem holds: convective_inertia, time_derivative,
  /// viscous_stress, body_force and Boussinesq_buoyancy.
  void setup(Problem & problem,
             const std::string & owner,
             const std::vector<std::string> & velocities,
             int pressure)
  {
    const int dim = problem.mesh().dimension();
    _v = resolveVelocities(problem, velocities, dim);
    _p = pressure;
    _rho = _rho_time = _mu = nullptr;
    _forces.assign(static_cast<std::size_t>(dim), {});
    _buoyancy.assign(static_cast<std::size_t>(dim), {});
    for (const auto & k : problem.kernels())
    {
      const auto & kp = k->parameters();
      const auto it = std::find(velocities.begin(), velocities.end(), k->variableName());
      if (it == velocities.end())
        continue;
      const auto i = static_cast<std::size_t>(it - velocities.begin());
      const auto & type = k->type();
      if (type == "convective_inertia" && !_rho)
        _rho = functionParameter(problem, kp, "density");
      else if (type == "time_derivative" && !_rho_time)
        _rho_time = functionParameter(problem, kp, "coeff");
      else if (type == "viscous_stress" && !_mu)
        _mu = functionParameter(problem, kp, "dynamic_viscosity");
      else if (type == "body_force")
        _forces[i].push_back(
            {functionParameter(problem, kp, "value"), kp.getBool("scale_with_load")});
      else if (type == "Boussinesq_buoyancy")
      {
        Buoyancy b;
        b.temperature = problem.variableIndex(kp.getString("temperature"));
        const auto g = kp.getRealList("gravity");
        b.gravity = i < g.size() ? g[i] : 0.0;
        b.density = functionParameter(problem, kp, "density");
        b.expansion = functionParameter(problem, kp, "vol_expansion_coeff");
        b.reference = kp.getReal("reference_temperature");
        b.scale_with_load = kp.getBool("scale_with_load");
        _buoyancy[i].push_back(b);
      }
    }
    // The residual holds a time derivative when the flow has inertia (a Stokes flow has none).
    _time_term = _rho || _rho_time;
    const auto zero = std::make_shared<ConstantFunction>(0.0);
    if (!_rho)
      _rho = zero;
    if (!_rho_time)
      _rho_time = _rho;
    if (!_mu)
      _mu = zero;
    (void) owner;

    const auto & mesh = problem.mesh();
    _h.resize(mesh.numElements());
    for (Index e = 0; e < mesh.numElements(); ++e)
      _h[e] = elementSize(mesh, e);
  }

  int dimension() const { return static_cast<int>(_v.size()); }
  int velocity(int i) const { return _v[i]; }
  /// Whether every function read from the momentum kernels may be evaluated from several threads.
  bool threadSafe() const
  {
    for (const auto & f : {_rho, _rho_time, _mu})
      if (f && !f->threadSafe())
        return false;
    for (const auto & list : _forces)
      for (const auto & force : list)
        if (!force.value->threadSafe())
          return false;
    for (const auto & list : _buoyancy)
      for (const auto & b : list)
        if (!b.density->threadSafe() || !b.expansion->threadSafe())
          return false;
    return true;
  }

  /// The density of the convective term, zero for a Stokes flow and for the Stokes start of a
  /// steady flow (QpContext::inertia).
  double density(const QpContext & ctx) const { return ctx.inertia * _rho->value(ctx.x, ctx.time); }

  ADReal tau(const QpContext & ctx) const
  {
    const double rho = density(ctx);
    const double rho_time = _rho_time->value(ctx.x, ctx.time);
    const double mu = _mu->value(ctx.x, ctx.time);
    const double h = ctx.element >= 0 ? _h[ctx.element] : 1.0;
    ADReal speed2(1e-300);
    for (int v : _v)
      speed2 += ctx.coefficientValue(v) * ctx.coefficientValue(v);
    ADReal inverse2 = (4.0 * rho * rho / (h * h)) * speed2;
    // The factor 3 (9 on the square) gives the viscous limit h^2 / (12 nu) of the optimal
    // parameter of linear elements, as MOOSE's navier_stokes module takes it (INSADTauMaterial).
    const double viscous = 3.0 * 4.0 * mu / (h * h);
    inverse2 += viscous * viscous;
    if (ctx.dt > 0)
    {
      const double unsteady = 2.0 * rho_time / ctx.dt;
      inverse2 += unsteady * unsteady;
    }
    return 1.0 / sqrt(inverse2);
  }

  /// Component i of r_m, the residual of the momentum equation without its viscous term.  In a
  /// step of the theta method it is the residual of the discrete equation: the time difference,
  /// the pressure gradient at the new time (Problem::steadyWeight), and the other terms weighted
  /// by theta at the new time and by 1 - theta at the old one.  It then vanishes for the
  /// solution of the discrete equations, and the stabilisation is consistent in time.
  ADReal residual(const QpContext & ctx, int i) const { return residual(ctx, i, ctx.time_weight); }
  /// The same with the weight @p time_weight of the time derivative (QpContext::time_weight).
  ADReal residual(const QpContext & ctx, int i, double time_weight) const
  {
    ADReal r = ctx.gradient(_p)[i];
    const double rho_time = _rho_time->value(ctx.x, ctx.time);
    if (rho_time != 0.0 && ctx.dt > 0)
      r += (time_weight * rho_time / ctx.dt) * (ctx.value(_v[i]) - ctx.oldValue(_v[i]));
    const bool average = ctx.dt > 0.0 && ctx.theta < 1.0;
    r += (average ? ctx.theta : 1.0) * steadyResidual(ctx, i);
    if (average)
      r += (1.0 - ctx.theta) * oldSteadyResidual(ctx, i);
    return r;
  }

  /// The dynamic viscosity at the point.
  double viscosity(const QpContext & ctx) const { return _mu->value(ctx.x, ctx.time); }
  /// The density of the time derivative at the point.
  double timeDensity(const QpContext & ctx) const { return _rho_time->value(ctx.x, ctx.time); }
  /// Whether the residual holds the time derivative rho dv/dt (a flow with inertia).
  bool hasTimeTerm() const { return _time_term; }

private:
  /// The terms of r_m besides the time difference and the pressure gradient, at the new state:
  /// the inertia, minus the body forces, plus the buoyancy.
  ADReal steadyResidual(const QpContext & ctx, int i) const
  {
    ADReal r(0.0);
    const double rho = density(ctx);
    if (rho != 0.0)
    {
      const auto & g = ctx.gradient(_v[i]);
      ADReal convection(0.0);
      for (std::size_t d = 0; d < _v.size(); ++d)
        convection += ctx.coefficientValue(_v[d]) * g[d];
      r += rho * convection;
    }
    for (const auto & force : _forces[i])
      r -= (force.scale_with_load ? ctx.load_factor : 1.0) * force.value->value(ctx.x, ctx.time);
    // f_i = -rho beta (T - T0) g_i, so r_m gains + rho beta (T - T0) g_i.
    for (const auto & b : _buoyancy[i])
      if (b.gravity != 0.0)
      {
        const double c =
            b.density->value(ctx.x, ctx.time) * b.expansion->value(ctx.x, ctx.time) * b.gravity;
        r += (b.scale_with_load ? ctx.load_factor : 1.0) * c *
             (ctx.value(b.temperature) - b.reference);
      }
    return r;
  }
  /// The same terms at the old state and the old time of a step.
  double oldSteadyResidual(const QpContext & ctx, int i) const
  {
    const double t = ctx.time - ctx.dt;
    double r = 0.0;
    const double rho = ctx.inertia * _rho->value(ctx.x, t);
    if (rho != 0.0)
    {
      const Point & g = ctx.oldGradient(_v[i]);
      double convection = 0.0;
      for (std::size_t d = 0; d < _v.size(); ++d)
        convection += ctx.oldValue(_v[d]) * g[d];
      r += rho * convection;
    }
    for (const auto & force : _forces[i])
      r -= (force.scale_with_load ? ctx.load_factor : 1.0) * force.value->value(ctx.x, t);
    for (const auto & b : _buoyancy[i])
      if (b.gravity != 0.0)
        r += (b.scale_with_load ? ctx.load_factor : 1.0) * b.density->value(ctx.x, t) *
             b.expansion->value(ctx.x, t) * b.gravity * (ctx.oldValue(b.temperature) - b.reference);
    return r;
  }

  struct Force
  {
    FunctionPtr value;
    bool scale_with_load = false;
  };
  struct Buoyancy
  {
    int temperature = -1;
    double gravity = 0.0, reference = 0.0;
    FunctionPtr density, expansion;
    bool scale_with_load = false;
  };
  std::vector<int> _v;
  int _p = -1;
  FunctionPtr _rho, _rho_time, _mu;
  bool _time_term = false;
  std::vector<std::vector<Force>> _forces;
  std::vector<std::vector<Buoyancy>> _buoyancy;
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
    addVelocitiesParam(p);
    return p;
  }
  explicit PressureGradient(const InputParameters & p)
      : Kernel(p), _component(componentOf(name(), variableName(), p.getStringList("velocities")))
  {
  }
  void initialSetup(Problem & problem) override
  {
    Kernel::initialSetup(problem);
    _p = coupledVariable(problem, "pressure");
    _hoop = problem.coordinateSystem() == CoordinateSystem::Axisymmetric && _component == 0;
  }
  bool hasFlux() const override { return true; }
  bool hasSource() const override { return _hoop; }
  bool atNewTime() const override { return true; }
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
        "density, the viscosity, the body forces and the buoyancy of r_m are read from the "
        "momentum kernels of the velocities. The boundary condition mass_flux must be applied "
        "on every boundary where the pressure is not prescribed, and the fluid_dynamics "
        "physics applies it there.");
    addVelocitiesParam(p);
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
    _r.setup(problem, name(), _params.getStringList("velocities"), variable());
    // Without the stabilisation only an inf-sup stable pair gives a unique
    // pressure; equal-order interpolation then has spurious pressure modes
    // (the checkerboard of Q1-Q1), and the solve fails or returns them.
    const bool taylor_hood = problem.hasMixedOrder() &&
                             problem.variable(variable()).order == VariableOrder::First &&
                             problem.variable(_r.velocity(0)).order != VariableOrder::First;
    if (!_stabilize && !taylor_hood)
      throw InputError(
          "mass_conservation '" + name() + "': 'stabilization' is false, but the pressure '" +
          problem.variable(variable()).name +
          "' is interpolated like the velocity. Equal-order velocity and pressure need the "
          "stabilisation; without it the pressure has spurious modes. Either keep "
          "stabilization = true, or use the Taylor-Hood element: a quadratic mesh, method "
          "'FEM', and the pressure declared with order 'first'.");
  }
  bool threadSafe() const override { return Kernel::threadSafe() && _r.threadSafe(); }
  bool hasFlux() const override { return true; }
  bool isMassConservation() const override { return true; }
  bool residualHasTimeDerivative() const override { return _stabilize && _r.hasTimeTerm(); }
  bool flowCoefficients(const QpContext & ctx, FlowCoefficients & c) const override
  {
    c.viscosity = _r.viscosity(ctx);
    c.stabilized = _stabilize;
    c.density = _r.density(ctx);
    c.time_density = _r.timeDensity(ctx);
    c.num_velocities = _r.dimension();
    for (int d = 0; d < c.num_velocities; ++d)
      c.velocities[d] = _r.velocity(d);
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
  bool conservedVelocity(const QpContext & ctx, ADVector3 & w) const override
  {
    // The flux of this kernel is -w.  The step takes it at the new time with the weight 1, so the
    // time derivative in r_m has the weight 1 at the new time and none at the old one, whatever
    // kernel asks for w.
    const int dim = _r.dimension();
    w = {ADReal(0.0), ADReal(0.0), ADReal(0.0)};
    const double time_weight = ctx.time_weight > 0.0 ? 1.0 : 0.0;
    const ADReal tau = _stabilize ? _r.tau(ctx) : ADReal(0.0);
    for (int d = 0; d < dim; ++d)
    {
      w[d] = ctx.value(_r.velocity(d));
      if (_stabilize)
        w[d] -= tau * _r.residual(ctx, d, time_weight);
    }
    return true;
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
        "oscillations of a convection-dominated flow on a coarse mesh. The density, the "
        "viscosity, the body forces and the buoyancy of r_m are read from the momentum kernels "
        "of the velocities.");
    addVelocitiesParam(p);
    p.addRequired("pressure", ParameterKind::String, "Name of the pressure variable.");
    return p;
  }
  explicit MomentumStabilization(const InputParameters & p)
      : Kernel(p), _component(componentOf(name(), variableName(), p.getStringList("velocities")))
  {
  }
  void initialSetup(Problem & problem) override
  {
    Kernel::initialSetup(problem);
    _r.setup(
        problem, name(), _params.getStringList("velocities"), coupledVariable(problem, "pressure"));
  }
  bool threadSafe() const override { return Kernel::threadSafe() && _r.threadSafe(); }
  bool hasFlux() const override { return true; }
  bool residualHasTimeDerivative() const override { return _r.hasTimeTerm(); }
  bool atNewTime() const override { return true; }
  void computeFlux(const QpContext & ctx, ADVector3 & F) const override
  {
    const double rho = _r.density(ctx);
    if (rho == 0.0)
      return;
    const ADReal c = rho * _r.tau(ctx) * _r.residual(ctx, _component);
    for (int d = 0; d < _r.dimension(); ++d)
      F[d] = c * ctx.coefficientValue(_r.velocity(d));
  }

private:
  int _component;
  MomentumResidual _r;
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
    addVelocitiesParam(p);
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
        "from the penalty relation, with the penalty parameter gamma of the "
        "penalty_incompressibility kernels of the velocities. Evaluate it at element centroids "
        "(the reduced-order points), where it is most accurate.");
    addVelocitiesParam(p);
    return p;
  }
  explicit PenaltyPressure(const InputParameters & p) : Property(p) {}
  void initialSetup(Problem & problem) override
  {
    Property::initialSetup(problem);
    const auto velocities = _params.getStringList("velocities");
    _v = resolveVelocities(problem, velocities, problem.mesh().dimension());
    // The penalty parameter is that of the penalty_incompressibility kernels of the velocities.
    _gamma = 0.0;
    for (const auto & k : problem.kernels())
      if (k->type() == "penalty_incompressibility" &&
          std::find(velocities.begin(), velocities.end(), k->variableName()) != velocities.end())
        _gamma = k->parameters().getReal("penalty_parameter");
    if (_gamma == 0.0)
      throw InputError("'" + name() +
                       "': no penalty_incompressibility kernel acts on the velocities, so the "
                       "penalty pressure has no penalty parameter.");
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
  double _gamma = 0.0;
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
  f.add<MassFluxBC>("mass_flux", ObjectCategory::BoundaryCondition, m);
  f.add<PenaltyPressure>("penalty_pressure", ObjectCategory::Property, m);
}

} // namespace dualmesh
