// SPDX-License-Identifier: LGPL-2.1-or-later
//
// Fluids module: flows of viscous incompressible fluids by the penalty
// (reduced-integration) formulation of Chapter 9 of Reddy's book.
//
// The pressure is eliminated with the penalty relation
//
//     P = -gamma (du/dx + dv/dy),
//
// so that the momentum equations become, for the velocity component u_i,
//
//     rho (v . grad) u_i - div [ mu (grad u_i + (grad u)_i) + gamma (div v) e_i ]
//         - f_i = 0 .
//
// The viscous part and the penalty part are separate kernels so that the
// penalty term can be integrated with a reduced rule (evaluated at the
// element centroid), which is what makes the penalty formulation work.
#include "dualmesh/base/Factory.h"
#include "dualmesh/base/Kernel.h"
#include "dualmesh/base/Material.h"
#include "dualmesh/base/Problem.h"

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

/// Viscous stress of a Newtonian fluid: mu (grad v + grad v^T).
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
                  "mismatch silently assembles one momentum equation into another. Add one "
                  "instance per velocity variable.");
    p.addOptional("dynamic_viscosity",
                  ParameterKind::Function,
                  1.0,
                  "Dynamic viscosity mu in pascal seconds, as a constant or the name of a "
                  "function of (x, y, z, t). Because it is evaluated from position and time "
                  "only, this kernel is Newtonian; a shear-rate dependent viscosity needs a "
                  "material and a kernel of your own.");
    return p;
  }
  explicit ViscousStress(const InputParameters & p)
      : Kernel(p), _component(static_cast<int>(p.getInt("component")))
  {
  }
  void initialSetup(Problem & problem) override
  {
    Kernel::initialSetup(problem);
    _v =
        resolveVelocities(problem, _params.getStringList("velocities"), problem.mesh().dimension());
    _mu = getFunction(problem, "dynamic_viscosity");
    if (_component < 0 || _component >= static_cast<int>(_v.size()))
      throw InputError("'component' is outside the range of the velocity variables.");
  }
  bool hasFlux() const override { return true; }
  void computeFlux(const QpContext & ctx, ADVector3 & F) const override
  {
    const double mu = _mu->value(ctx.x, ctx.time);
    const auto & gi = ctx.gradient(_v[_component]);
    for (int d = 0; d < ctx.dim; ++d)
      F[d] = mu * (gi[d] + ctx.gradient(_v[d])[_component]);
  }

private:
  int _component;
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
        "The penalty term that replaces the pressure in the momentum equation. In the "
        "canonical form -div F + S = 0 it contributes the flux F = gamma (div v) e_i, that "
        "is, gamma (du/dx + dv/dy + dw/dz) in the direction of this instance's component "
        "and zero in the others, with no source. It must be integrated with a reduced rule, "
        "which is the default here; a fuller rule locks the velocity field.");
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
                  "constitutive relation P = -gamma div v. Choose it roughly 10^4 to 10^7 "
                  "times the dynamic viscosity: too small and the flow is measurably "
                  "compressible, too large and the linear system becomes so ill-conditioned "
                  "that the direct solver loses accuracy. The default suits a viscosity of "
                  "order one. PenaltyIncompressibility and PenaltyPressure each hold their own "
                  "copy, and the recovered pressure is meaningless unless the two agree.");
    p.addOptional("quadrature",
                  ParameterKind::String,
                  std::string("midpoint"),
                  "Quadrature rule for the penalty term, taking the same values as "
                  "elsewhere. The default midpoint rule is the one-point rule that the "
                  "penalty formulation requires; a fuller rule locks the velocity field and "
                  "should be used only to demonstrate that locking.");
    p.addOptional("reduced_integration",
                  ParameterKind::Boolean,
                  true,
                  "Evaluate the velocity gradients at the centroid of the element while "
                  "integrating with the geometric rule above; this is selective reduced "
                  "integration. It defaults to true here, unlike the framework default. Leave "
                  "it on: turning it off locks the incompressibility constraint.");
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
          "PenaltyIncompressibility '" + name() +
          "': the penalty formulation of incompressible flow cannot be used with the "
          "cell-centred finite volume method (zfvm). It needs the incompressibility "
          "constraint to be imposed once per element by a reduced integration rule, and the "
          "cell-centred method imposes it at every face, which locks the velocity field. Use "
          "the dual mesh (dmcdm), finite element (fem) or vertex-centred finite volume (hfvm) "
          "method for incompressible flow.");
    _v =
        resolveVelocities(problem, _params.getStringList("velocities"), problem.mesh().dimension());
  }
  bool hasFlux() const override { return true; }
  void computeFlux(const QpContext & ctx, ADVector3 & F) const override
  {
    ADReal divergence(0.0);
    for (std::size_t d = 0; d < _v.size(); ++d)
      divergence += ctx.gradient(_v[d])[d];
    for (int d = 0; d < ctx.dim; ++d)
      F[d] = d == _component ? _gamma * divergence : ADReal(0.0);
  }

private:
  int _component;
  double _gamma;
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
        "which is the linearization used in the book.");
    p.addRequired("velocities",
                  ParameterKind::StringList,
                  "Names of the velocity variables, exactly one per mesh dimension and in "
                  "coordinate order. Every object of the fluids module must be given the same list "
                  "in the same order, because 'component' indexes into it.");
    p.addRequired("component", ParameterKind::Integer, "Component index of this equation.");
    p.addOptional("density",
                  ParameterKind::Function,
                  1.0,
                  "Mass density rho in kilograms per cubic metre, as a constant or the name of "
                  "a function. Together with the dynamic viscosity it fixes the Reynolds "
                  "number. For a Stokes flow leave this kernel out altogether rather than "
                  "setting the density to zero.");
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
        "component. The density is taken as rho0 everywhere except in the gravity term, where "
        "rho = rho0 (1 - beta (T - T0)); the constant part rho0 g is balanced by the "
        "hydrostatic pressure, which leaves the body force f_i = -rho0 beta (T - T0) g_i, so "
        "that fluid warmer than T0 rises. In the canonical form the source is S = -f_i. The "
        "temperature is an unknown of the same problem, and the coupling is exact in "
        "Newton's method.");
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
    p.addOptional("density", ParameterKind::Function, 1.0, "Reference density rho0.");
    p.addOptional("thermal_expansion",
                  ParameterKind::Function,
                  1.0,
                  "Volumetric thermal expansion coefficient beta, in 1/K. For an ideal gas it "
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
      throw InputError("BoussinesqBuoyancy '" + name() + "': 'component' must be between 0 and " +
                       std::to_string(problem.mesh().dimension() - 1) + ".");
    _g = _component < static_cast<int>(g.size()) ? g[_component] : 0.0;
    _rho = getFunction(problem, "density");
    _beta = getFunction(problem, "thermal_expansion");
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

/// Pressure recovered from the penalty relation (for post-processing).
class PenaltyPressure : public Material
{
public:
  static InputParameters validParams()
  {
    InputParameters p = Material::validParams();
    p.setClassDescription(
        "Declares the material property 'pressure' = -gamma div v, the pressure recovered "
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
        "given to PenaltyIncompressibility, or the recovered pressure is meaningless.");
    return p;
  }
  explicit PenaltyPressure(const InputParameters & p)
      : Material(p), _gamma(p.getReal("penalty_parameter"))
  {
  }
  void initialSetup(Problem & problem) override
  {
    Material::initialSetup(problem);
    _v =
        resolveVelocities(problem, _params.getStringList("velocities"), problem.mesh().dimension());
  }
  void declareProperties(MaterialPropertyRegistry & r) override
  {
    _pressure = r.declare("pressure", 1);
  }
  void computeProperties(QpContext & ctx) const override
  {
    ADReal divergence(0.0);
    for (std::size_t d = 0; d < _v.size(); ++d)
      divergence += ctx.gradient(_v[d])[d];
    ctx.property(_pressure) = -_gamma * divergence;
  }

private:
  double _gamma;
  std::vector<int> _v;
  int _pressure = -1;
};

} // namespace

void
registerFluidObjects(Factory & f)
{
  const std::string m = "fluids";
  f.add<ViscousStress>("ViscousStress", ObjectCategory::Kernel, m);
  f.add<PenaltyIncompressibility>("PenaltyIncompressibility", ObjectCategory::Kernel, m);
  f.add<ConvectiveInertia>("ConvectiveInertia", ObjectCategory::Kernel, m);
  f.add<BoussinesqBuoyancy>("BoussinesqBuoyancy", ObjectCategory::Kernel, m);
  f.add<PenaltyPressure>("PenaltyPressure", ObjectCategory::Material, m);
}

} // namespace dualmesh
