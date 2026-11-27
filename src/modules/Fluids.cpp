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
#include "dualmesh/core/Coordinates.h"
#include "dualmesh/modules/FluidSystem.h"
#include "dualmesh/modules/Framework.h"

#include <algorithm>

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

/// The default penalty parameter gamma of the penalty formulation, in Pa s.
constexpr double kDefaultPenalty = 1.0e8;

/// The parameter penalty_parameter of the penalty formulation, for @p p.
void
addPenaltyParam(InputParameters & p)
{
  p.addOptional("penalty_parameter",
                ParameterKind::Real,
                kDefaultPenalty,
                "Penalty parameter gamma in Pa s, which enforces incompressibility through the "
                "constitutive relation p = -gamma div v. Reddy, Anand and Roy (2023) give 10^4 "
                "to 10^12 times the dynamic viscosity. A value that is too small leaves the flow "
                "compressible, and a value that is too large makes the linear system so "
                "ill-conditioned that the direct solver loses accuracy. Default 10^8, which lies "
                "in that range for a dynamic viscosity from 10^-4 to 10^4 Pa s.");
}

/// div v of the velocity @p v in the coordinate system @p coord, with its hoop term, at the point
/// where the velocity is evaluated.  The hoop term n v_r / r takes the radius r of that point
/// (QpContext::x_field) along with v_r.  Under the reduced rule of the penalty term that point is
/// the centroid of the element, so every point of the element, a face or a volume point of DMCDM
/// as well as the single point of FEM, has the same divergence: the element carries one penalty
/// pressure, which the face fluxes and the hoop stresses share and penalty_pressure reports.
ADReal
velocityDivergence(const QpContext & ctx, const std::vector<int> & v, CoordinateSystem coord)
{
  ADReal div(0.0);
  for (std::size_t d = 0; d < v.size(); ++d)
    div += ctx.gradient(v[d])[d];
  div += hoopDivergence(coord, ctx.value(v[0]), ctx.x_field[0]);
  return div;
}

/// The hoop factor of the momentum equation of the velocity component @p component: that of the
/// coordinate system for the radial component 0, and 0 for the others.
int
momentumHoopFactor(const Problem & problem, int component)
{
  return component == 0 ? hoopFactor(problem.coordinateSystem()) : 0;
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
        "and v_r the radial velocity, or mu v_r / r^2 in the laplacian form. In spherical "
        "coordinates, with their two hoop directions, the hoop term is twice as large.");
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
    _hoop = momentumHoopFactor(problem, _component);
  }
  bool hasFlux() const override { return true; }
  bool hasSource() const override { return _hoop > 0; }
  /// The dynamic viscosity mu at the point of @p ctx.
  double viscosity(const QpContext & ctx) const { return _mu->value(ctx.x, ctx.time); }
  void computeFlux(const QpContext & ctx, ADVector3 & F) const override
  {
    const double mu = viscosity(ctx);
    const auto & gi = ctx.gradient(_v[_component]);
    for (int d = 0; d < ctx.dim; ++d)
      F[d] = _laplacian ? mu * gi[d] : mu * (gi[d] + ctx.gradient(_v[d])[_component]);
  }
  // The term - mu (Laplacian(v) - n v_r / r^2 e_r)_i of the strong residual, with n the hoop
  // factor, which quadratic elements give.  It vanishes inside a linear element, whose residual therefore leaves it out
  // (see Problem::fillContext).  It is div(2 mu eps(v)) for an incompressible flow with a
  // viscosity uniform in space, and for one that varies it lacks the term
  // grad mu . (grad v + grad v^T).  The radial component of the vector Laplacian of an
  // axisymmetric flow without swirl is the scalar Laplacian of v_r less v_r / r^2, and that of a
  // spherically symmetric flow the scalar Laplacian of v_r less 2 v_r / r^2.
  bool inStrongResidual() const override { return true; }
  bool oldStateFromFields() const override { return true; }
  ADReal strongResidual(const QpContext & ctx) const override
  {
    if (ctx.laplacian.empty())
      return ADReal(0.0);
    const double hoop = _hoop > 0 && ctx.x[0] > 0.0 ? _hoop / (ctx.x[0] * ctx.x[0]) : 0.0;
    return -viscosity(ctx) * (ctx.laplacian[_v[_component]] - hoop * ctx.value(_v[_component]));
  }
  // In axisymmetric coordinates (r, z) without swirl, the radial equation
  // carries the hoop stress: -(div sigma)_r = -div_rz(sigma_r.) + sigma_tt / r,
  // and the viscous part of sigma_tt is 2 mu u_r / r.  The radial component of
  // the vector Laplacian carries - u_r / r^2 instead, so the Laplacian form has
  // the source mu u_r / r^2.  In spherical coordinates the two hoop stresses
  // sigma_tt = sigma_pp double each source.
  ADReal computeSource(const QpContext & ctx) const override
  {
    const double r = ctx.x[0];
    if (r == 0.0)
      return ADReal(0.0);
    return ((_laplacian ? 1.0 : 2.0) * _hoop * viscosity(ctx) / (r * r)) * ctx.value(_v[0]);
  }

private:
  int _component;
  /// The hoop factor of the equation (see momentumHoopFactor).
  int _hoop = 0;
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
        "side. In spherical coordinates both terms are twice as large. The term must be integrated with a reduced rule, which is the default here, "
        "because a rule with more points locks the velocity field.");
    addVelocitiesParam(p);
    addPenaltyParam(p);
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
                  "Selective reduced integration: evaluate the velocity, its gradients and so the "
                  "divergence, with its hoop term in axisymmetric and spherical coordinates, at "
                  "the centroid of the element, while the rule named by 'quadrature' integrates "
                  "the term. "
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
    _coord = problem.coordinateSystem();
    _hoop = momentumHoopFactor(problem, _component);
  }
  bool hasFlux() const override { return true; }
  bool hasSource() const override { return _hoop > 0; }
  void computeFlux(const QpContext & ctx, ADVector3 & F) const override
  {
    const ADReal div = velocityDivergence(ctx, _v, _coord);
    for (int d = 0; d < ctx.dim; ++d)
      F[d] = d == _component ? _gamma * div : ADReal(0.0);
  }
  // The penalty pressure enters the hoop stress of the radial equation as
  // well: S = n sigma_tt / r, with n the hoop factor, whose pressure part is
  // -p = gamma div v.  The factor n / r belongs to the geometry of the balance law and takes the
  // radius of the integration point.  Its product with the coordinate factor, 2 pi in
  // axisymmetric and 8 pi r in spherical coordinates, is linear in r, so the midpoint rule
  // integrates it exactly.  The pressure takes the divergence of the field point (see
  // velocityDivergence), so a pressure uniform in the element balances its face fluxes exactly in
  // each control domain.
  ADReal computeSource(const QpContext & ctx) const override
  {
    const double r = ctx.x[0];
    if (r == 0.0)
      return ADReal(0.0);
    return (_hoop * _gamma / r) * velocityDivergence(ctx, _v, _coord);
  }

private:
  int _component;
  double _gamma;
  CoordinateSystem _coord = CoordinateSystem::Cartesian;
  /// The hoop factor of the equation (see momentumHoopFactor).
  int _hoop = 0;
  std::vector<int> _v;
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
  bool oldStateFromFields() const override { return true; }
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
///   r_m = rho_t dv/dt + rho (v . grad) v + grad p - mu Laplacian(v) - f ,
///
/// the strong residual (StrongResidual) of the momentum equation of each component, with the
/// viscous term on quadratic elements only (it vanishes inside a linear one), and the
/// stabilisation parameter tau of Tezduyar (1991) (stabilizationTau), written per unit density so
/// that it also covers a Stokes flow (rho = 0):
///
///   tau = [ (2 rho_t / dt)^2 + (2 rho |v| / h)^2 + 9 (4 mu / h^2)^2 ]^(-1/2) .
///
/// tau r_m has the units of a velocity.  For a quadratic element h is half the element size,
/// because the interpolation resolves half the element.  The viscous term of r_m needs the
/// second derivatives of the shape functions, which quadratic elements provide.  Inside a linear
/// element it vanishes (simplices, and bilinear elements on parallelograms) or nearly so, and
/// leaving it out keeps the optimal first order in the energy norm.  The densities rho of the
/// convection and rho_t of the time derivative and the viscosity mu are the coefficients of the
/// advection, time_derivative and viscous_stress kernels of the first velocity that act on the
/// block of the point, and zero without one there.  So each block of a layered flow takes its
/// own fluid in tau.
class MomentumResidual
{
public:
  /// Read the coefficients of the momentum equations of @p velocities that the problem holds,
  /// and with @p stabilized collect their terms, for the stabilisation named @p owner.
  void setup(Problem & problem,
             const std::string & owner,
             const std::vector<std::string> & velocities,
             bool stabilized)
  {
    _v = resolveVelocities(problem, velocities, problem.mesh().dimension());
    _equations.assign(stabilized ? velocities.size() : 0, {});
    for (std::size_t i = 0; i < _equations.size(); ++i)
      _equations[i].setup(problem, velocities[i], owner);
    _advection.clear();
    _storage.clear();
    _viscous.clear();
    for (const auto & k : problem.kernels())
    {
      if (k->variableName() != velocities[0])
        continue;
      if (const auto * a = dynamic_cast<const Advection *>(k.get()))
        _advection.push_back(a);
      if (const auto * t = dynamic_cast<const TimeDerivative *>(k.get()))
        _storage.push_back(t);
      if (const auto * v = dynamic_cast<const ViscousStress *>(k.get()))
        _viscous.push_back(v);
    }
  }

  int dimension() const { return static_cast<int>(_v.size()); }
  int velocity(int i) const { return _v[i]; }
  bool threadSafe() const
  {
    return std::all_of(
        _equations.begin(), _equations.end(), [](const auto & e) { return e.threadSafe(); });
  }

  /// The density of the convective term, zero for a Stokes flow and for the Stokes start of a
  /// steady flow (QpContext::inertia).
  double density(const QpContext & ctx) const
  {
    const Advection * a = onBlock(_advection, ctx);
    return a ? a->coefficient(ctx).value() : 0.0;
  }
  /// The density of the time derivative at the point.
  double timeDensity(const QpContext & ctx) const
  {
    const TimeDerivative * t = onBlock(_storage, ctx);
    return t ? t->coefficient(ctx).value() : 0.0;
  }
  /// The dynamic viscosity at the point.
  double viscosity(const QpContext & ctx) const
  {
    const ViscousStress * v = onBlock(_viscous, ctx);
    return v ? v->viscosity(ctx) : 0.0;
  }

  ADReal tau(const QpContext & ctx) const
  {
    const double rho = density(ctx);
    ADReal speed2(1e-300);
    for (int v : _v)
      speed2 += ctx.coefficientValue(v) * ctx.coefficientValue(v);
    return stabilizationTau(_equations[0].elementSize(ctx),
                            ADReal(rho * rho * speed2),
                            ADReal(viscosity(ctx)),
                            ctx.dt > 0.0 ? 2.0 * timeDensity(ctx) / ctx.dt : 0.0);
  }

  /// Component i of r_m.  In a step of the theta method it is the residual of the discrete
  /// equation: the time difference and the pressure gradient at the new time
  /// (Problem::steadyWeight), and the other terms weighted by theta at the new time and by
  /// 1 - theta at the old one.  It then vanishes for the solution of the discrete equations, and
  /// the stabilisation is consistent in time.
  ADReal residual(const QpContext & ctx, int i) const { return residual(ctx, i, ctx.time_weight); }
  /// The same with the weight @p time_weight of the time derivative (QpContext::time_weight).
  ADReal residual(const QpContext & ctx, int i, double time_weight) const
  {
    return _equations[static_cast<std::size_t>(i)].value(ctx, time_weight, true);
  }

  /// Whether the residual holds the time derivative rho_t dv/dt (a transient flow).
  bool hasTimeTerm() const { return !_storage.empty(); }

private:
  /// The first of @p kernels that acts on the block of the point, or null.  A probe of the whole
  /// problem, a context without an element (the coefficients that the projection time
  /// integration reads), has no block and takes the first kernel.
  template <typename K>
  static const K * onBlock(const std::vector<const K *> & kernels, const QpContext & ctx)
  {
    if (ctx.element < 0)
      return kernels.empty() ? nullptr : kernels.front();
    for (const K * k : kernels)
      if (k->activeOnBlock(ctx.block))
        return k;
    return nullptr;
  }

  std::vector<int> _v;
  std::vector<StrongResidual> _equations;
  std::vector<const Advection *> _advection;
  std::vector<const TimeDerivative *> _storage;
  std::vector<const ViscousStress *> _viscous;
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
        "also has the hoop term -p / r on its left side, with r the radius, and in spherical "
        "coordinates the hoop term -2 p / r.");
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
    _hoop = momentumHoopFactor(problem, _component);
  }
  bool hasFlux() const override { return true; }
  bool hasSource() const override { return _hoop > 0; }
  bool atNewTime() const override { return true; }
  void computeFlux(const QpContext & ctx, ADVector3 & F) const override
  {
    F[_component] = -ctx.value(_p);
  }
  /// The term dp/dx_i of the strong residual, the divergence of the flux with the hoop source.
  bool inStrongResidual() const override { return true; }
  ADReal strongResidual(const QpContext & ctx) const override
  {
    return ctx.gradient(_p)[_component];
  }
  ADReal computeSource(const QpContext & ctx) const override
  {
    const double r = ctx.x[0];
    if (r == 0.0)
      return ADReal(0.0);
    return (-_hoop / r) * ctx.value(_p);
  }

private:
  int _component;
  int _p = -1;
  /// The hoop factor of the equation (see momentumHoopFactor).
  int _hoop = 0;
};

/// Conservation of mass for an incompressible flow, on the pressure variable.
class MassConservation : public MassEquation
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
        "coefficient of their time_derivative, rho the coefficient of their advection, p the "
        "pressure, mu the dynamic viscosity and f the body force. Quadratic elements give the "
        "viscous term, and linear elements leave it out, since it vanishes inside them. The "
        "code builds r_m from the momentum kernels of the velocities, and refuses a momentum "
        "kernel with a flux whose divergence r_m does not hold. A term other than the "
        "viscous stress, the pressure, the advection, the body force and the buoyancy requires "
        "the backward_Euler time scheme. The parameter is tau = [(2 rho_t / dt)^2 + "
        "(2 rho |v| / h)^2 + 9 (4 mu / h^2)^2]^(-1/2), after Tezduyar (1991), with dt the time "
        "step, whose term enters a transient solve only, and h the element size, halved on a "
        "quadratic element. The Taylor-Hood element, a first-order pressure with quadratic "
        "velocities, is stable without the stabilization and takes the flux F = v. The boundary "
        "condition mass_flux closes the balance on the boundary, and the fluid_dynamics physics "
        "applies it on every side set.");
    addVelocitiesParam(p);
    return p;
  }
  explicit MassConservation(const InputParameters & p) : MassEquation(p) {}
  void initialSetup(Problem & problem) override
  {
    Kernel::initialSetup(problem);
    // Only the Taylor-Hood element, an inf-sup stable pair, gives a unique pressure without the
    // stabilisation.  Equal-order interpolation has spurious pressure modes without it (the
    // checkerboard of Q1-Q1).  A balance tested with the functions of a velocity component holds
    // no pressure to stabilize: the divergence that the start of a flow without a pressure
    // removes (projectInitialVelocity).
    const auto velocities = _params.getStringList("velocities");
    const bool taylor_hood =
        problem.hasMixedOrder() && problem.variable(variable()).order == VariableOrder::First &&
        problem.variable(problem.variableIndex(velocities[0])).order != VariableOrder::First;
    _stabilize = !taylor_hood && std::find(velocities.begin(), velocities.end(), variableName()) ==
                                     velocities.end();
    _r.setup(problem, name(), velocities, _stabilize);
  }
  bool threadSafe() const override { return Kernel::threadSafe() && _r.threadSafe(); }
  bool hasFlux() const override { return true; }
  bool needsLaplacian() const override { return _stabilize; }
  bool residualHasTimeDerivative() const override { return _stabilize && _r.hasTimeTerm(); }
  void flowCoefficients(const QpContext & ctx, FlowCoefficients & c) const override
  {
    c.viscosity = _r.viscosity(ctx);
    c.stabilized = _stabilize;
    c.density = _r.density(ctx);
    c.time_density = _r.timeDensity(ctx);
    c.num_velocities = _r.dimension();
    for (int d = 0; d < c.num_velocities; ++d)
      c.velocities[d] = _r.velocity(d);
  }
  void computeFlux(const QpContext & ctx, ADVector3 & F) const override
  {
    const int dim = _r.dimension();
    if (!_stabilize || !ctx.stabilization)
    {
      for (int d = 0; d < dim; ++d)
        F[d] = -ctx.value(_r.velocity(d));
      return;
    }
    const ADReal tau = _r.tau(ctx);
    for (int d = 0; d < dim; ++d)
      F[d] = tau * _r.residual(ctx, d) - ctx.value(_r.velocity(d));
  }
  void conservedVelocity(const QpContext & ctx, ADVector3 & w) const override
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
  }

private:
  /// Whether the flux holds the pressure stabilisation (equal-order velocity and pressure).
  bool _stabilize = true;
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
        "F = -rho tau (r_m)_i v to the balance law dm/dt + div F = s, with rho the coefficient "
        "of the advection of the velocity and v the velocity. In the finite element method "
        "(FEM) it adds the integral of rho tau (v . grad N_j) (r_m)_i to the equation of node j, "
        "with N_j its shape function. The dual mesh control domain method (DMCDM) takes the "
        "same flux through the faces of the control domains. Here r_m is the strong residual of "
        "the momentum equations and tau the stabilization parameter, both as in "
        "mass_conservation, and the code builds them from the momentum kernels of the "
        "velocities, with the pressure gradient of their pressure_gradient kernels. With "
        "rho c_p = rho and k = mu, tau is the parameter of heat_streamline_stabilization divided "
        "by rho, so the two stabilizations add the same flux tau_heat r v. The term is "
        "consistent, because r_m vanishes for the exact solution, and it suppresses the "
        "spurious oscillations of a flow dominated by convection on a coarse mesh. It vanishes "
        "for a Stokes flow, which has no advection of its velocity.");
    addVelocitiesParam(p);
    return p;
  }
  explicit MomentumStabilization(const InputParameters & p)
      : Kernel(p), _component(componentOf(name(), variableName(), p.getStringList("velocities")))
  {
  }
  void initialSetup(Problem & problem) override
  {
    Kernel::initialSetup(problem);
    _r.setup(problem, name(), _params.getStringList("velocities"), true);
  }
  bool threadSafe() const override { return Kernel::threadSafe() && _r.threadSafe(); }
  bool hasFlux() const override { return true; }
  bool isStabilization() const override { return true; }
  bool transportsVariable() const override { return true; }
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
        "relation recovers from the velocity v, with gamma the penalty parameter of the "
        "penalty_incompressibility kernels of the velocities. In axisymmetric coordinates the "
        "divergence includes v_r / r, with r the radius and v_r the radial velocity, and in "
        "spherical coordinates 2 v_r / r. Evaluate it at element centroids, where the reduced "
        "rule of penalty_incompressibility evaluates the divergence: there it is the pressure "
        "of the discrete equations.");
    addVelocitiesParam(p);
    addPenaltyParam(p);
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
    _coord = problem.coordinateSystem();
  }
  void declareProperties(PropertyRegistry & r) override { _pressure = r.declare("pressure", 1); }
  void computeProperties(QpContext & ctx) const override
  {
    ctx.property(_pressure) = -_gamma * velocityDivergence(ctx, _v, _coord);
  }

private:
  double _gamma;
  CoordinateSystem _coord = CoordinateSystem::Cartesian;
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
  f.add<BoussinesqBuoyancy>("Boussinesq_buoyancy", ObjectCategory::Kernel, m);
  f.add<PressureGradient>("pressure_gradient", ObjectCategory::Kernel, m);
  f.add<MassConservation>("mass_conservation", ObjectCategory::Kernel, m);
  f.add<MomentumStabilization>("momentum_stabilization", ObjectCategory::Kernel, m);
  f.add<MassFluxBC>("mass_flux", ObjectCategory::BoundaryCondition, m);
  f.add<PenaltyPressure>("penalty_pressure", ObjectCategory::Property, m);
  f.setFlowSystem(makeFluidSystem);
}

} // namespace dualmesh
