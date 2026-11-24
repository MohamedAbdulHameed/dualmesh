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
//     div [ v - tau r_m ] = 0 ,   r_m = rho (dv/dt + (v . grad) v) + grad p - mu Laplacian(v) - f ,
//
// where r_m is the momentum residual, whose viscous term quadratic elements give and linear ones,
// inside which it vanishes, leave out.  In the
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

/// The parameter that lists the velocities, shared by every object of the module.
void
addVelocitiesParam(InputParameters & p)
{
  p.addRequired("velocities",
                ParameterKind::StringList,
                "Names of the velocity variables, exactly one per mesh dimension and in "
                "coordinate order (x, y, then z). The place of 'variable' in this list fixes the "
                "component i of the momentum equation that the object acts on.");
}

/// Viscous stress of a Newtonian fluid: mu (grad v + grad v^T), or its Laplacian form mu grad v.
class ViscousStress : public Kernel
{
public:
  static InputParameters validParams()
  {
    InputParameters p = Kernel::validParams();
    p.setClassDescription(
        "The viscous part of the momentum flux of a Newtonian fluid, for the momentum equation "
        "of the velocity component v_i: the flux F = -mu (grad v_i + dv/dx_i) in the balance law "
        "dm/dt + div F = s, which is minus row i of the viscous stress. Here mu is the dynamic "
        "viscosity and v the velocity. In two dimensions the flux of the x component is "
        "-(2 mu dv_x/dx, mu (dv_x/dy + dv_y/dx)). In axisymmetric coordinates the radial "
        "equation also has the hoop term 2 mu v_r / r^2 on its left side, with r the radius "
        "and v_r the radial velocity, or mu v_r / r^2 in the laplacian form.");
    addVelocitiesParam(p);
    p.addOptional("dynamic_viscosity",
                  ParameterKind::Function,
                  1.0,
                  "Dynamic viscosity mu in Pa s: a constant, a registered function, or an "
                  "expression of x, y, z and t. The viscosity depends on position and time only, "
                  "so the kernel describes a Newtonian fluid and cannot take a viscosity that "
                  "depends on the shear rate. Default 1, for a problem written in dimensionless "
                  "form.");
    p.setRange("dynamic_viscosity", ParameterRange::Positive);
    p.addOptional("form",
                  ParameterKind::String,
                  std::string("stress"),
                  "The form of the viscous flux. 'stress' takes F = -mu (grad v_i + dv/dx_i), "
                  "whose natural boundary condition is zero traction, sigma n = 0, with sigma the "
                  "Cauchy stress and n the outward unit normal. 'laplacian' takes "
                  "F = -mu grad v_i, which gives the same equations for a viscosity uniform in "
                  "space and an incompressible flow. Its natural boundary condition is the "
                  "do-nothing outflow mu dv_i/dn - p n_i = 0, with p the pressure, which the "
                  "projection time integration also takes. Default stress, which holds for a "
                  "viscosity that varies in space, and whose natural boundary quantity is the "
                  "traction.");
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
        "The penalty term that replaces the pressure in the momentum equation of the velocity "
        "component v_i. It adds the pressure part p e_i of the momentum flux F to the balance "
        "law dm/dt + div F = s, with the penalty pressure p = -gamma div v, gamma the penalty "
        "parameter, v the velocity and e_i the unit vector of component i. In axisymmetric "
        "coordinates the divergence includes v_r / r, with r the radius and v_r the radial "
        "velocity, and the radial equation also has the hoop term gamma (div v) / r on its left "
        "side. The term must be integrated with a reduced rule, which is the default here, "
        "because a rule with more points locks the velocity field.");
    addVelocitiesParam(p);
    p.addOptional("penalty_parameter",
                  ParameterKind::Real,
                  1.0e8,
                  "Penalty parameter gamma in Pa s, which enforces incompressibility through the "
                  "constitutive relation p = -gamma div v. Reddy, Anand and Roy (2023) give 10^4 "
                  "to 10^12 times the dynamic viscosity. A value that is too small leaves the flow "
                  "compressible, and a value that is too large makes the linear system so "
                  "ill-conditioned that the direct solver loses accuracy. penalty_pressure reads "
                  "it from here. Default 10^8, which lies in that range for a dynamic viscosity "
                  "from 10^-4 to 10^4 Pa s.");
    p.addOptional("quadrature",
                  ParameterKind::String,
                  std::string("midpoint"),
                  "Quadrature rule for the penalty term, with the same values as the 'quadrature' "
                  "parameter of the other objects. A rule with more points locks the velocity "
                  "field. Default midpoint, the one-point rule that the penalty formulation "
                  "requires.");
    p.addOptional("reduced_integration",
                  ParameterKind::Boolean,
                  true,
                  "Selective reduced integration: evaluate the velocity gradients at the centroid "
                  "of the element, while the rule named by 'quadrature' integrates the term. "
                  "Default true, unlike the other objects, because turning it off locks the "
                  "incompressibility constraint.");
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
        "The convective term rho (v . grad) v_i on the left side of the momentum equation of the "
        "velocity component v_i, with rho the density and v the velocity. In an equal-order "
        "flow, with the finite element method (FEM) or the dual mesh control domain method "
        "(DMCDM) and the pressure formulation, the term carries the velocity w = v - tau r_m "
        "that the mass equation conserves, with tau r_m the pressure stabilization of "
        "mass_conservation. FEM then adds rho (w . grad) v_i, and DMCDM the face flux "
        "rho (w . n)(v_i,f - v_i,C) for the control volume C on each side of a face, with v_i,f "
        "the face value of the advection scheme and n the unit normal of the face. The momentum "
        "is then conserved exactly against the discrete mass flux through the boundary (Hughes "
        "and Wells 2005), and a uniform velocity stays unchanged. Without such a mass equation, "
        "in the penalty and Taylor-Hood formulations, the term carries v, which conserves the "
        "momentum only as far as the discrete velocity is free of divergence. Picard iteration "
        "takes the transporting velocity from the previous iterate.");
    addVelocitiesParam(p);
    p.addOptional("density",
                  ParameterKind::Function,
                  1.0,
                  "Density rho in kg/m^3: a constant, a registered function, or an expression of "
                  "x, y, z and t. With the dynamic viscosity it sets the Reynolds number. For a "
                  "Stokes flow, leave this kernel out. Default 1, for a problem written in "
                  "dimensionless form.");
    p.setRange("density", ParameterRange::NonNegative);
    addAdvectionSchemeParam(p);
    return p;
  }
  explicit ConvectiveInertia(const InputParameters & p)
      : Kernel(p), _component(componentOf(name(), variableName(), p.getStringList("velocities"))),
        _scheme_text(p.getString("advection_scheme")),
        _face_scheme(_scheme_text == "upwind" || _scheme_text == "linear_upwind" ||
                     _scheme_text == "limited"),
        _flux(_face_scheme)
  {
  }
  void initialSetup(Problem & problem) override
  {
    Kernel::initialSetup(problem);
    _v =
        resolveVelocities(problem, _params.getStringList("velocities"), problem.mesh().dimension());
    _rho = getFunction(problem, "density");
    _scheme = advectionScheme(problem, _scheme_text, name());
    _mass = conservingMassEquation(problem, variable());
    _flux = _face_scheme || (_mass && problem.isDual());
  }
  // The term is rho (w . grad) u_i with the velocity w that the mass equation conserves, where a
  // mass equation tests with the functions of u_i (an equal-order flow), and with v otherwise: the
  // source for FEM, and for DMCDM the face flux rho (w . n)(u_i,f - u_i,C) for the control volume
  // C on each side, as heat_convection takes it.  The momentum is then conserved exactly, against
  // the discrete mass flux through the boundary.
  bool hasSource() const override { return !_flux; }
  bool hasFlux() const override { return _flux; }
  bool usesFaceSides() const override { return _flux; }
  bool nonConservativeFaceFlux() const override { return _flux; }
  bool inStrongResidual() const override { return true; }
  bool isInertia() const override { return true; }
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
    const ADVector3 w = advectingVelocity(ctx, _mass, _v);
    ADReal s(0.0);
    for (std::size_t d = 0; d < _v.size(); ++d)
      s += w[d] * gi[d];
    return rho * s;
  }

private:
  int _component;
  std::vector<int> _v;
  FunctionPtr _rho;
  std::string _scheme_text;
  /// Whether the scheme is a face flux of DMCDM (upwind, linear_upwind, limited).
  bool _face_scheme;
  /// Whether the term is a flux (conservative with _mass, else non-conservative) or a source.
  bool _flux;
  AdvectionScheme _scheme = AdvectionScheme::Central;
  /// The mass equation whose conserved velocity the flux carries (conservingMassEquation).
  const Kernel * _mass = nullptr;
};

/// The buoyancy force of the Boussinesq approximation.
class BoussinesqBuoyancy : public Kernel
{
public:
  bool isLoad() const override { return true; }
  static InputParameters validParams()
  {
    InputParameters p = Kernel::validParams();
    p.setClassDescription(
        "Buoyancy in the Boussinesq approximation, for the momentum equation of the velocity "
        "component v_i. The density is the reference density rho_0 everywhere except in the "
        "gravity term, where it is rho_0 (1 - beta (T - T_0)), with beta the volumetric "
        "expansion coefficient, T the temperature and T_0 the reference temperature. The "
        "hydrostatic pressure balances the constant part rho_0 g, with g the gravitational "
        "acceleration, and leaves the source s = -rho_0 beta (T - T_0) g_i of the balance law "
        "dm/dt + div F = s, so that fluid warmer than T_0 rises. The temperature is an unknown "
        "of the same problem, and Newton's method takes the coupling exactly.");
    p.addRequired("temperature",
                  ParameterKind::String,
                  "Name of the temperature variable. It must already exist on the problem.");
    addVelocitiesParam(p);
    p.addRequired("gravity",
                  ParameterKind::RealList,
                  "The gravitational acceleration g in m/s^2, for example (0, -9.81) in two "
                  "dimensions with y upward. Missing trailing components are zero.");
    p.addOptional("density",
                  ParameterKind::Function,
                  1.0,
                  "Reference density rho_0 in kg/m^3: a constant, a registered function, or an "
                  "expression of x, y, z and t. Default 1, for a problem written in "
                  "dimensionless form.");
    p.setRange("density", ParameterRange::NonNegative);
    p.addRequired("volumetric_expansion_coefficient",
                  ParameterKind::Function,
                  "Volumetric expansion coefficient beta = -(1/rho) d rho/dT in 1/K, the relative "
                  "change of volume per kelvin: a constant, a registered function, or an "
                  "expression of x, y, z and t. For an ideal gas it is 1/T_0.");
    p.addOptional("reference_temperature",
                  ParameterKind::Real,
                  0.0,
                  "Reference temperature T_0 at which the density is rho_0, in the units of the "
                  "temperature variable. Default 0, for a problem written in dimensionless form, "
                  "whose temperature is the difference from T_0.");
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
    _beta = getFunction(problem, "volumetric_expansion_coefficient");
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

/// The strong momentum residual,
///
///   r_m = rho (du/dt + (u . grad) u) + grad p - mu Laplacian(u) - f ,
///
/// with the viscous term on quadratic elements only (it vanishes inside a linear one),
///
/// and the stabilisation parameter after Tezduyar (1991), written per unit density
/// so that it also covers a Stokes flow (rho = 0):
///
///   tau = [ (2 rho / dt)^2 + (2 rho |u| / h)^2 + 9 (4 mu / h^2)^2 ]^(-1/2) ,
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
  /// @p stabilized: the residual stabilizes the flow, so every term of the momentum equation must
  /// enter it.
  void setup(Problem & problem,
             const std::string & owner,
             const std::vector<std::string> & velocities,
             int pressure,
             bool stabilized)
  {
    const int dim = problem.mesh().dimension();
    _v = resolveVelocities(problem, velocities, dim);
    _p = pressure;
    _axisymmetric = problem.coordinateSystem() == CoordinateSystem::Axisymmetric;
    _rho = _rho_time = _mu = nullptr;
    _forces.assign(static_cast<std::size_t>(dim), {});
    _sources.assign(static_cast<std::size_t>(dim), {});
    _buoyancy.assign(static_cast<std::size_t>(dim), {});
    for (const auto & k : problem.kernels())
    {
      const auto & kp = k->parameters();
      const auto it = std::find(velocities.begin(), velocities.end(), k->variableName());
      if (it == velocities.end())
        continue;
      const auto i = static_cast<std::size_t>(it - velocities.begin());
      const auto & type = k->type();
      if (type == "convective_inertia")
      {
        if (!_rho)
          _rho = functionParameter(problem, kp, "density", k->name());
      }
      else if (type == "time_derivative")
      {
        if (!_rho_time)
          _rho_time = functionParameter(problem, kp, "coefficient", k->name());
      }
      else if (type == "viscous_stress")
      {
        if (!_mu)
          _mu = functionParameter(problem, kp, "dynamic_viscosity", k->name());
      }
      else if (type == "body_force")
        _forces[i].push_back({functionParameter(problem, kp, "value", k->name())});
      else if (type == "Boussinesq_buoyancy")
      {
        Buoyancy b;
        b.temperature = problem.variableIndex(kp.getString("temperature"));
        const auto g = kp.getRealList("gravity");
        b.gravity = i < g.size() ? g[i] : 0.0;
        b.density = functionParameter(problem, kp, "density", k->name());
        b.expansion = functionParameter(problem, kp, "volumetric_expansion_coefficient", k->name());
        b.reference = kp.getReal("reference_temperature");
        _buoyancy[i].push_back(b);
      }
      else if (type == "pressure_gradient" || type == "momentum_stabilization" ||
               type == "penalty_incompressibility")
        continue;
      // Any other term of the momentum equation enters the residual by its source.  A flux has
      // no divergence inside a linear element, so a term with one cannot.
      else if (stabilized && k->hasFlux())
        throw InputError("'" + owner +
                         "': the stabilization of the flow holds the residual of "
                         "the momentum equation, which takes the sources of its terms, so '" +
                         k->name() + "' (" + type + ") on '" + k->variableName() +
                         "', a term with a flux, would be missing from it. Give the term as a "
                         "source, or turn the stabilization off.");
      else if (k->hasSource())
        _sources[i].push_back(k.get());
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
    for (const auto & list : _sources)
      for (const Kernel * source : list)
        if (!source->threadSafe())
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
    // The factor 3 (9 on the square) gives the viscous limit h^2 / (12 mu) of the optimal
    // parameter of linear elements, per unit density.
    const double viscous = 3.0 * 4.0 * mu / (h * h);
    inverse2 += viscous * viscous;
    if (ctx.dt > 0)
    {
      const double unsteady = 2.0 * rho_time / ctx.dt;
      inverse2 += unsteady * unsteady;
    }
    return 1.0 / sqrt(inverse2);
  }

  /// Component i of r_m, the residual of the momentum equation (the viscous term on quadratic
  /// elements).  In a
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
    if (average && !_sources[static_cast<std::size_t>(i)].empty())
      throw InputError("A term of its own on a velocity enters the stabilized residual of the "
                       "flow at the new time only, so the flow takes the backward_Euler scheme, "
                       "whose residual holds no old state.");
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
  /// The radial component of the vector Laplacian of an axisymmetric flow without swirl is the
  /// scalar Laplacian of v_r less v_r / r^2: the factor 1 / r^2 for component @p i, zero otherwise.
  double hoop(const QpContext & ctx, int i) const
  {
    return _axisymmetric && i == 0 && ctx.x[0] > 0.0 ? 1.0 / (ctx.x[0] * ctx.x[0]) : 0.0;
  }
  /// The terms of r_m besides the time difference and the pressure gradient, at the new state:
  /// the inertia, minus the body forces, plus the buoyancy.
  ADReal steadyResidual(const QpContext & ctx, int i) const
  {
    ADReal r(0.0);
    // The viscous term - mu Laplacian(v_i), which quadratic elements give.  It vanishes inside
    // a linear element, whose residual therefore leaves it out (see Problem::fillContext).  It is
    // div(2 mu eps(v)) for a viscosity that is uniform in space, and for one that varies it lacks
    // the term grad mu . (grad v + grad v^T).
    if (!ctx.laplacian.empty())
      r -= _mu->value(ctx.x, ctx.time) * (ctx.laplacian[_v[i]] - hoop(ctx, i) * ctx.value(_v[i]));
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
      r -= ctx.data_fraction * force.value->value(ctx.x, ctx.time);
    for (const Kernel * source : _sources[static_cast<std::size_t>(i)])
      // A kernel source S stands on the side of the residual, as S = -f of a body force does.
      r += source->computeSource(ctx);
    // f_i = -rho beta (T - T0) g_i, so r_m gains + rho beta (T - T0) g_i.
    for (const auto & b : _buoyancy[i])
      if (b.gravity != 0.0)
      {
        const double c =
            b.density->value(ctx.x, ctx.time) * b.expansion->value(ctx.x, ctx.time) * b.gravity;
        r += ctx.data_fraction * c * (ctx.value(b.temperature) - b.reference);
      }
    return r;
  }
  /// The same terms at the old state and the old time of a step.
  double oldSteadyResidual(const QpContext & ctx, int i) const
  {
    const double t = ctx.time - ctx.dt;
    double r = 0.0;
    if (!ctx.laplacian_old.empty())
      r -= _mu->value(ctx.x, t) * (ctx.laplacian_old[_v[i]] - hoop(ctx, i) * ctx.oldValue(_v[i]));
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
      r -= ctx.data_fraction * force.value->value(ctx.x, t);
    for (const auto & b : _buoyancy[i])
      if (b.gravity != 0.0)
        r += ctx.data_fraction * b.density->value(ctx.x, t) * b.expansion->value(ctx.x, t) *
             b.gravity * (ctx.oldValue(b.temperature) - b.reference);
    return r;
  }

  struct Force
  {
    FunctionPtr value;
  };
  struct Buoyancy
  {
    int temperature = -1;
    double gravity = 0.0, reference = 0.0;
    FunctionPtr density, expansion;
  };
  std::vector<int> _v;
  int _p = -1;
  bool _axisymmetric = false;
  FunctionPtr _rho, _rho_time, _mu;
  bool _time_term = false;
  std::vector<std::vector<Force>> _forces;
  /// The other terms of each component, which enter by their sources.
  std::vector<std::vector<const Kernel *>> _sources;
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
        "The pressure term of the momentum equation of the velocity component v_i: the pressure "
        "part F = p e_i of the momentum flux in the balance law dm/dt + div F = s, with p the "
        "pressure and e_i the unit vector of component i. In this flux form the natural "
        "boundary quantity of the momentum equation is the traction component -n . F = "
        "(sigma n)_i, with sigma the Cauchy stress and n the outward unit normal, so a boundary "
        "without a condition is free of traction, as an open outlet is. A step of the theta "
        "method takes the term at the new time. In axisymmetric coordinates the radial equation "
        "also has the hoop term -p / r on its left side, with r the radius.");
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
        "Conservation of mass of an incompressible flow of velocity v, assembled on the "
        "pressure variable as the balance law div F = 0 with the flux F = v - tau r_m. The term "
        "tau r_m is the residual-based pressure stabilization that makes equal-order velocity "
        "and pressure stable: the pressure-stabilizing Petrov-Galerkin (PSPG) term of the finite "
        "element method (FEM), and the momentum interpolation of Rhie and Chow (1983) in the dual "
        "mesh control domain method (DMCDM). Here r_m is the strong residual of the momentum "
        "equations, rho_t dv/dt + rho (v . grad) v + grad p - mu Laplacian(v) - f, with rho_t the "
        "coefficient of their time_derivative, rho the density of their convective_inertia, p "
        "the pressure, mu the dynamic viscosity and f the body force. Quadratic elements give "
        "the viscous term, and linear elements leave it out, since it vanishes inside them. The "
        "code reads rho_t, rho, mu, the body forces, the buoyancy and the other source terms of "
        "r_m from the momentum kernels of the velocities, and refuses a momentum kernel with a "
        "flux of its own. Those other source terms require the backward_Euler time scheme. The "
        "parameter is tau = [(2 rho_t / dt)^2 + (2 rho |v| / h)^2 + "
        "9 (4 mu / h^2)^2]^(-1/2), after Tezduyar (1991), with dt the time step, whose term "
        "enters a transient solve only, and h the element size, halved on a quadratic element. "
        "The boundary condition mass_flux closes the balance on the boundary, and the "
        "fluid_dynamics physics applies it on every side set.");
    addVelocitiesParam(p);
    p.addOptional("stabilization",
                  ParameterKind::Boolean,
                  true,
                  "Add the residual-based pressure stabilization tau r_m. Equal-order velocity and "
                  "pressure need it, and the code refuses false unless the pressure is a "
                  "first-order variable with quadratic velocities, the Taylor-Hood element, which "
                  "is stable without it. Default true, for the equal-order interpolation.");
    return p;
  }
  explicit MassConservation(const InputParameters & p)
      : Kernel(p), _stabilize(p.getBool("stabilization"))
  {
  }
  void initialSetup(Problem & problem) override
  {
    Kernel::initialSetup(problem);
    _r.setup(problem, name(), _params.getStringList("velocities"), variable(), _stabilize);
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
  bool needsLaplacian() const override { return _stabilize; }
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
    const double time_weight = ctx.time_weight > 0.0 || ctx.previous_step ? 1.0 : 0.0;
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
        "Streamline upwind Petrov-Galerkin (SUPG) stabilization (Brooks and Hughes 1982) of the "
        "momentum equation of the velocity component v_i. It adds the flux "
        "F = -rho tau (r_m)_i v to the balance law dm/dt + div F = s, with rho the density of "
        "convective_inertia and v the velocity. In the finite element method (FEM) it adds the "
        "integral of rho tau (v . grad N_j) (r_m)_i to the equation of node j, with N_j its "
        "shape function. The dual mesh control domain method (DMCDM) takes the same flux "
        "through the faces of the control domains. Here r_m is the strong residual of the "
        "momentum equations and tau the stabilization parameter, both as in mass_conservation, "
        "and the code reads their coefficients and terms from the momentum kernels of the "
        "velocities. The term is consistent, because r_m vanishes for the exact solution, and "
        "it suppresses the spurious oscillations of a flow dominated by convection on a coarse "
        "mesh. It vanishes for a Stokes flow, which has no convective_inertia.");
    addVelocitiesParam(p);
    p.addRequired("pressure",
                  ParameterKind::String,
                  "Name of the pressure variable, whose gradient enters r_m.");
    return p;
  }
  explicit MomentumStabilization(const InputParameters & p)
      : Kernel(p), _component(componentOf(name(), variableName(), p.getStringList("velocities")))
  {
  }
  void initialSetup(Problem & problem) override
  {
    Kernel::initialSetup(problem);
    _r.setup(problem,
             name(),
             _params.getStringList("velocities"),
             coupledVariable(problem, "pressure"),
             true);
  }
  bool threadSafe() const override { return Kernel::threadSafe() && _r.threadSafe(); }
  bool hasFlux() const override { return true; }
  bool residualHasTimeDerivative() const override { return _r.hasTimeTerm(); }
  bool needsLaplacian() const override { return true; }
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
        "The boundary condition of mass_conservation, whose flux is F = v - tau r_m with tau r_m "
        "its pressure stabilization. It prescribes the outward flux n . F = v . n, the volume "
        "flux of the current velocity v leaving through the boundary, with n the outward unit "
        "normal. It closes the mass balance of the control volumes and elements that touch the "
        "boundary, and states that tau r_m has no normal component there. Apply it on every "
        "boundary where the pressure is not prescribed: walls, inlets and outlets alike.");
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
        "Declares the property 'pressure' = -gamma div v, the pressure that the penalty "
        "relation recovers from the velocity v, with the penalty parameter gamma of the "
        "penalty_incompressibility kernels of the velocities. In axisymmetric coordinates the "
        "divergence includes v_r / r, with r the radius and v_r the radial velocity. Evaluate "
        "it at element centroids, the points of the reduced rule, where it is most accurate.");
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
