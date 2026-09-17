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
    p.addRequired("velocities", ParameterKind::StringList, "Velocity variables.");
    p.addRequired("component", ParameterKind::Integer, "Component index of this equation.");
    p.addOptional("dynamic_viscosity", ParameterKind::Function, 1.0, "Dynamic viscosity mu.");
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
        "Penalty term gamma (div v) in the momentum equation, which replaces the pressure. "
        "It must be integrated with a reduced rule (the default here), otherwise the "
        "velocity field locks.");
    p.addRequired("velocities", ParameterKind::StringList, "Velocity variables.");
    p.addRequired("component", ParameterKind::Integer, "Component index of this equation.");
    p.addOptional("penalty_parameter", ParameterKind::Real, 1.0e8, "Penalty parameter gamma.");
    p.addOptional("quadrature",
                  ParameterKind::String,
                  std::string("midpoint"),
                  "Quadrature (reduced by default).");
    p.addOptional("reduced_integration",
                  ParameterKind::Boolean,
                  true,
                  "Evaluate the velocity gradients at the element centroid.");
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
    p.addRequired("velocities", ParameterKind::StringList, "Velocity variables.");
    p.addRequired("component", ParameterKind::Integer, "Component index of this equation.");
    p.addOptional("density", ParameterKind::Function, 1.0, "Mass density rho.");
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
    p.addRequired("velocities", ParameterKind::StringList, "Velocity variables.");
    p.addOptional("penalty_parameter", ParameterKind::Real, 1.0e8, "Penalty parameter gamma.");
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
  f.add<PenaltyPressure>("PenaltyPressure", ObjectCategory::Material, m);
}

} // namespace dualmesh
