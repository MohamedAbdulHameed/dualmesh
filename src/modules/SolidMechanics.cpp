// SPDX-License-Identifier: LGPL-2.1-or-later
//
// Solid mechanics module, continuum: linear elasticity in plane stress, plane
// strain, axisymmetric, and three-dimensional settings, with viscoelasticity
// and phase-field fracture, the kernels and the boundary conditions, and
// registerSolidMechanicsObjects, the entry point of the module.  SolidMechanics.h
// lists the other files of the module.
//
// Equilibrium is written in the canonical conservation form used by the
// framework.  For the displacement component u_i,
//
//     -div F_i + S_i = 0,   F_i = h (sigma_i0, sigma_i1, sigma_i2),  S_i = -h f_i,
//
// so that the natural boundary quantity n . F_i is the traction component
// t_i = n_j sigma_ij (times the thickness h).  In the axisymmetric case the
// integrals carry the factor 2 pi r and the hoop stress enters the radial
// equation as the source sigma_tt / r.
#include "dualmesh/modules/SolidMechanics.h"

#include "dualmesh/base/Deformation.h"
#include "dualmesh/base/Factory.h"
#include "dualmesh/base/FieldExpression.h"
#include "dualmesh/base/Kernel.h"
#include "dualmesh/base/Problem.h"
#include "dualmesh/base/Property.h"
#include "dualmesh/core/Coordinates.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>

namespace dualmesh
{

namespace solid_mechanics
{

namespace
{

Formulation
formulationFromName(const std::string & owner, const std::string & name)
{
  if (name == "plane_stress")
    return Formulation::PlaneStress;
  if (name == "plane_strain")
    return Formulation::PlaneStrain;
  if (name == "generalized_plane_strain")
    return Formulation::GeneralizedPlaneStrain;
  if (name == "axisymmetric")
    return Formulation::Axisymmetric;
  if (name == "axisymmetric_1d")
    return Formulation::AxisymmetricOneD;
  if (name == "three_dimensional")
    return Formulation::ThreeDimensional;
  throw InputError("'" + owner + "': unknown formulation '" + name +
                   "'. Use plane_stress, plane_strain, generalized_plane_strain, axisymmetric, "
                   "axisymmetric_1d or three_dimensional.");
}

/// The number of displacements of a formulation, which is also the dimension of its mesh.
int
formulationDimension(Formulation f)
{
  if (f == Formulation::AxisymmetricOneD)
    return 1;
  return f == Formulation::ThreeDimensional ? 3 : 2;
}

bool
isAxisymmetric(Formulation f)
{
  return f == Formulation::Axisymmetric || f == Formulation::AxisymmetricOneD;
}

/// The small strain of the displacements @p disp into @p e (six values, engineering shears), with
/// @p gradient(v) and @p value(v) the gradient and the value of the variable v: those of the
/// displacements for the strain, or those of their rates for the strain rate.  The strains that
/// are not displacement gradients stay zero: the axial strain e[1] of axisymmetric_1d, the
/// out-of-plane strain e[2] of generalized plane strain and that of plane stress.
template <typename Gradient, typename Value>
void
smallStrain(Formulation f,
            const std::vector<int> & disp,
            const Point & x,
            const Gradient & gradient,
            const Value & value,
            ADReal * e)
{
  for (int i = 0; i < 6; ++i)
    e[i] = 0.0;
  const auto & gu = gradient(disp[0]);
  e[0] = gu[0];
  if (f == Formulation::AxisymmetricOneD)
  {
    e[2] = hoopStrain(value(disp[0]), gu[0], x[0]);
    return;
  }
  const auto & gv = gradient(disp[1]);
  e[1] = gv[1];
  e[5] = gu[1];
  e[5] += gv[0];
  if (f == Formulation::Axisymmetric)
    e[2] = hoopStrain(value(disp[0]), gu[0], x[0]);
  else if (f == Formulation::ThreeDimensional)
  {
    const auto & gw = gradient(disp[2]);
    e[2] = gw[2];
    e[3] = gv[2];
    e[3] += gw[1];
    e[4] = gu[2];
    e[4] += gw[0];
  }
}

/// The component of the equilibrium equation of @p variable: its place in 'displacements'.
int
displacementComponent(const std::string & owner,
                      const std::string & variable,
                      const std::vector<std::string> & displacements)
{
  const auto it = std::find(displacements.begin(), displacements.end(), variable);
  if (it == displacements.end())
  {
    std::string list;
    for (const auto & d : displacements)
      list += (list.empty() ? "" : ", ") + d;
    throw InputError("'" + owner + "': the variable '" + variable +
                     "' is not one of the displacements (" + list + ").");
  }
  return static_cast<int>(it - displacements.begin());
}

/// The thickness of a plane problem for the condition @p owner on @p variable: that of the
/// stress_divergence kernels of the variable, else 1.  A condition reads one thickness, so
/// kernels of the variable with different thicknesses (blocks of different thickness) are
/// refused, where the first one found would silently scale the condition on the others.
double
planeThickness(const Problem & problem, const std::string & variable, const std::string & owner)
{
  double thickness = 0.0;
  for (const auto & k : problem.kernels())
    if (k->type() == "stress_divergence" && k->variableName() == variable)
    {
      const double h = k->parameters().getReal("thickness");
      if (thickness > 0.0 && h != thickness)
        throw InputError("'" + owner + "': the stress_divergence kernels of '" + variable +
                         "' have the thicknesses " + std::to_string(thickness) + " and " +
                         std::to_string(h) +
                         ", and this condition takes one. Blocks of different thickness under "
                         "one condition are not available.");
      thickness = h;
    }
  return thickness > 0.0 ? thickness : 1.0;
}

} // namespace

void
anisotropicStress(Formulation f, ADReal * ee, const std::array<double, 36> & C, ADReal * sigma)
{
  if (f == Formulation::PlaneStress)
  {
    // A e_o = -B e_p, with e_o the out-of-plane strains, e_p the in-plane ones (xx, yy, xy), A the
    // block of C on e_o and B its block from e_p to the out-of-plane stresses.
    static constexpr int out[3] = {2, 3, 4}, in[3] = {0, 1, 5};
    double A[3][3];
    std::array<ADReal, 3> b;
    for (int i = 0; i < 3; ++i)
    {
      b[i] = 0.0;
      for (int j = 0; j < 3; ++j)
      {
        A[i][j] = C[6 * out[i] + out[j]];
        b[i] -= C[6 * out[i] + in[j]] * ee[in[j]];
      }
    }
    const double det = A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1]) -
                       A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0]) +
                       A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0]);
    if (!(std::abs(det) > 0.0))
      throw InputError("A plane stress state needs a stiffness whose block on the out-of-plane "
                       "strains zz, yz and xz is invertible.");
    // The inverse of A by its cofactors.
    const double inv[3][3] = {{(A[1][1] * A[2][2] - A[1][2] * A[2][1]) / det,
                               (A[0][2] * A[2][1] - A[0][1] * A[2][2]) / det,
                               (A[0][1] * A[1][2] - A[0][2] * A[1][1]) / det},
                              {(A[1][2] * A[2][0] - A[1][0] * A[2][2]) / det,
                               (A[0][0] * A[2][2] - A[0][2] * A[2][0]) / det,
                               (A[0][2] * A[1][0] - A[0][0] * A[1][2]) / det},
                              {(A[1][0] * A[2][1] - A[1][1] * A[2][0]) / det,
                               (A[0][1] * A[2][0] - A[0][0] * A[2][1]) / det,
                               (A[0][0] * A[1][1] - A[0][1] * A[1][0]) / det}};
    for (int i = 0; i < 3; ++i)
    {
      ee[out[i]] = 0.0;
      for (int j = 0; j < 3; ++j)
        ee[out[i]] += inv[i][j] * b[j];
    }
  }
  for (int i = 0; i < 6; ++i)
  {
    sigma[i] = 0.0;
    for (int j = 0; j < 6; ++j)
      sigma[i] += C[6 * i + j] * ee[j];
  }
}

void
addDisplacementsParam(InputParameters & p, const std::string & what)
{
  p.addRequired("displacements",
                ParameterKind::StringList,
                "The displacement variables (m) in the order of the coordinate axes, for example "
                "[u, v, w] in three dimensions. " +
                    what);
  p.setModelVariables({"displacements"});
  p.addOptional("variable",
                ParameterKind::Names,
                std::vector<std::string>{},
                "The variable whose equation this object assembles. Default: every one of "
                "'displacements', each with an object of its own, so that one call covers every "
                "component. Give it for an object whose variable is a Lagrange multiplier "
                "(mortar_contact, normal_displacement) and for cohesive_zone, which acts on "
                "every displacement from one object.");
}

namespace
{

/// Divergence of the stress: the equilibrium equation of one displacement
/// component.
class StressDivergence : public Kernel
{
public:
  static InputParameters validParams()
  {
    InputParameters p = Kernel::validParams();
    p.setClassDescription(
        "The stress term of the equilibrium div(sigma) + b = 0 of one displacement component "
        "u_i, with sigma the stress (Pa) and b the body force per unit volume (N/m^3), which "
        "body_force adds. In the balance law dm/dt + div F = s of u_i, the flux is F = -H "
        "sigma_i, with sigma_i the row i of the stress and H the thickness (m), so F . n is minus "
        "the traction component sigma_ij n_j times H. The source is zero, except in axisymmetric "
        "coordinates, where the radial equation (component 0) has the source s = -sigma_tt/r, "
        "with sigma_tt the hoop component of 'stress_property' and r the radius (m), taken as "
        "zero on the axis.");
    addDisplacementsParam(p,
                          "The position of 'variable' in this list is the component i of its "
                          "equation.");
    p.addOptional("thickness",
                  ParameterKind::Real,
                  1.0,
                  "Out-of-plane thickness H (m) that multiplies the whole equilibrium equation, "
                  "for plane stress and plane strain only. A three-dimensional or axisymmetric "
                  "problem refuses any value other than 1, and the integrals of an axisymmetric "
                  "problem carry the factor 2 pi r. The traction and pressure conditions of the "
                  "same displacement read it from here, and mortar_contact and cohesive_zone take "
                  "it as a parameter of their own. It does not scale a point_source, so a "
                  "concentrated load is the total force through the thickness. Default 1: the "
                  "equation per unit thickness.");
    p.setRange("thickness", ParameterRange::Positive);
    p.addOptional("stress_property",
                  ParameterKind::String,
                  std::string("stress"),
                  "Property holding the stress: six components in Voigt order (the Cauchy stress "
                  "sigma of small_strain_stress), or nine components row by row (the first "
                  "Piola-Kirchhoff stress P of finite_strain_stress, for equations on the "
                  "undeformed mesh). Default stress, the small-strain stress.");
    p.addOptional("axisymmetric_hoop_term",
                  ParameterKind::Boolean,
                  true,
                  "Include the hoop source -sigma_tt/r in the radial equation of an axisymmetric "
                  "problem. Default true, because the radial equilibrium of a body of revolution "
                  "has this term.");
    return p;
  }

  explicit StressDivergence(const InputParameters & p)
      : Kernel(p),
        _component(displacementComponent(name(), variableName(), p.getStringList("displacements"))),
        _thickness(p.getReal("thickness")), _hoop(p.getBool("axisymmetric_hoop_term"))
  {
  }

  void initialSetup(Problem & problem) override
  {
    Kernel::initialSetup(problem);
    if (_thickness != 1.0 && (problem.mesh().dimension() == 3 ||
                              problem.coordinateSystem() == CoordinateSystem::Axisymmetric))
      throw InputError("'" + name() +
                       "': thickness is the out-of-plane thickness of a plane problem. A "
                       "three-dimensional or axisymmetric problem has none, so leave it at 1.");
    const auto & registry = problem.propertyRegistry();
    _stress = registry.id(_params.getString("stress_property"));
    _full = registry.components(_params.getString("stress_property")) == 9;
    const std::string size_name = _params.getString("stress_property") + "_term_size";
    if (registry.has(size_name))
      _stress_size = registry.id(size_name);
    _axisymmetric = problem.coordinateSystem() == CoordinateSystem::Axisymmetric;
  }

  std::vector<std::string> requiredProperties() const override
  {
    return {_params.getString("stress_property")};
  }

  bool hasFlux() const override { return true; }
  bool hasSource() const override { return _axisymmetric && _hoop && _component == 0; }

  void computeFlux(const QpContext & ctx, ADVector3 & F) const override
  {
    for (int d = 0; d < ctx.dim; ++d)
    {
      // A nine-component stress (the first Piola-Kirchhoff stress of the
      // finite-strain property objects) is stored row by row and is not symmetric.
      F[d] = _full ? ctx.property(_stress, 3 * _component + d)
                   : ctx.property(_stress, kVoigt[_component][d]);
      if (_thickness != 1.0)
        F[d] *= _thickness;
    }
  }

  ADReal computeSource(const QpContext & ctx) const override
  {
    if (!(_axisymmetric && _hoop && _component == 0))
      return ADReal(0.0);
    if (ctx.x[0] == 0.0)
      return ADReal(0.0);
    return ctx.property(_stress, _full ? 8 : 2) / ctx.x[0];
  }

  /// The size of the terms of the stress, when its object declares them as the property
  /// '<stress_property>_term_size' (a hyperelastic stress at a small strain is the sum of terms of
  /// the size of the modulus), and otherwise the magnitude of the flux.
  void fluxTermSize(const QpContext & ctx, const ADVector3 & F, double size[3]) const override
  {
    if (_stress_size < 0)
    {
      Kernel::fluxTermSize(ctx, F, size);
      return;
    }
    for (int d = 0; d < 3; ++d)
      size[d] = d < ctx.dim
                    ? std::abs(_thickness) * ctx.property(_stress_size, 3 * _component + d).value()
                    : 0.0;
  }

  double sourceTermSize(const QpContext & ctx, const ADReal & S) const override
  {
    if (_stress_size < 0 || !hasSource() || ctx.x[0] == 0.0)
      return Kernel::sourceTermSize(ctx, S);
    return ctx.property(_stress_size, 8).value() / std::abs(ctx.x[0]);
  }

private:
  int _component;
  double _thickness;
  bool _hoop;
  bool _full = false;
  bool _axisymmetric = false;
  int _stress = -1;
  int _stress_size = -1;
};

/// The damage equation of phase-field fracture (AT2): -Gc l div grad d + (Gc / l + 2 H) d = 2 H,
/// with H the history of the tensile elastic energy (fracture_driving_energy).
class PhaseFieldDamage : public Kernel
{
public:
  static InputParameters validParams()
  {
    InputParameters p = Kernel::validParams();
    p.setClassDescription(
        "The damage equation of phase-field fracture with the AT2 functional of Ambrosio and "
        "Tortorelli, -div(G_c l grad d) + (G_c/l) d = 2 (1 - d) psi_max, for the damage d "
        "(dimensionless, 0 intact and 1 broken). Here G_c is the toughness, l the length of the "
        "diffuse crack and psi_max (J/m^3) the property fracture_driving_energy of the stress "
        "object, the largest tensile elastic energy per unit volume the point has had. In the balance law dm/dt + div F = s of d, the flux is F = -G_c l grad d and the source "
        "is s = 2 (1 - d) psi_max - (G_c/l) d. A crack costs the energy G_c per unit area.");
    p.addRequired(
        "toughness", ParameterKind::Real, "The critical energy release rate G_c (J/m^2).");
    p.setRange("toughness", ParameterRange::Positive);
    p.addRequired("length", ParameterKind::Real, "The length l (m) of the diffuse crack.");
    p.setRange("length", ParameterRange::Positive);
    return p;
  }
  explicit PhaseFieldDamage(const InputParameters & p)
      : Kernel(p), _Gc(p.getReal("toughness")), _l(p.getReal("length"))
  {
  }
  void initialSetup(Problem & problem) override
  {
    Kernel::initialSetup(problem);
    _H = problem.propertyRegistry().id("fracture_driving_energy");
  }
  std::vector<std::string> requiredProperties() const override
  {
    return {"fracture_driving_energy"};
  }
  bool hasFlux() const override { return true; }
  bool hasSource() const override { return true; }
  void computeFlux(const QpContext & ctx, ADVector3 & F) const override
  {
    for (int d = 0; d < ctx.dim; ++d)
      F[d] = _Gc * _l * ctx.gradient(_var)[d];
  }
  ADReal computeSource(const QpContext & ctx) const override
  {
    const ADReal & H = ctx.property(_H);
    return (_Gc / _l + 2.0 * H) * ctx.value(_var) - 2.0 * H;
  }

private:
  double _Gc, _l;
  int _H = -1;
};

/// The axial equation of generalized plane strain: on the uniform out-of-plane strain, the
/// source sigma_zz, whose integral over the section is the axial force it carries.  A body_force
/// of F/A on the same variable makes that force F.
class OutOfPlaneStress : public Kernel
{
public:
  static InputParameters validParams()
  {
    InputParameters p = Kernel::validParams();
    p.setClassDescription(
        "The axial equation of generalized plane strain for the uniform axial strain epsilon_zz: "
        "the integral of the axial stress sigma_zz over the section balances the axial force F "
        "(N). A body_force of F/A on the same variable, with A the area of the section (m^2), "
        "applies that force. Without a body_force the axial force is zero, and the body is free "
        "to stretch along its axis.");
    return p;
  }
  explicit OutOfPlaneStress(const InputParameters & p) : Kernel(p) {}
  void initialSetup(Problem & problem) override
  {
    Kernel::initialSetup(problem);
    _stress = problem.propertyRegistry().id("stress");
  }
  std::vector<std::string> requiredProperties() const override { return {"stress"}; }
  bool hasSource() const override { return true; }
  ADReal computeSource(const QpContext & ctx) const override { return ctx.property(_stress, 2); }

private:
  int _stress = -1;
};

/// One unknown for a variable on the whole mesh: every node is a copy of the first, as the
/// periodic condition joins two boundaries (the uniform strain of generalized plane strain).
class UniformField : public Constraint
{
public:
  static InputParameters validParams()
  {
    InputParameters p = Constraint::validParams();
    p.setClassDescription("One unknown of 'variable' for the whole mesh: every node is a copy of "
                          "the first, and one equation collects the equations of all the nodes. "
                          "It makes the uniform axial strain of generalized plane strain.");
    p.addRequired("variable", ParameterKind::String, "The variable that has one value.");
    return p;
  }
  explicit UniformField(const InputParameters & p) : Constraint(p) {}
  void initialSetup(Problem & problem) override
  {
    const int v = problem.variableIndex(_params.getString("variable"));
    _pairs.clear();
    for (Index n = 1; n < problem.mesh().numNodes(); ++n)
      _pairs.push_back({problem.dof(n, v), problem.dof(0, v)});
  }
};

/// The stiffness-proportional part b K v of Rayleigh damping, for one displacement component:
/// the stress rate of the elastic stiffness, F = b h C : d(eps)/dt, from the velocity of the
/// generalized-alpha method (QpContext::rate).  K is the elastic stiffness of small strain, as
/// Rayleigh damping takes it, whatever the material law of the stress.
class StiffnessDamping : public Kernel
{
public:
  static InputParameters validParams()
  {
    InputParameters p = Kernel::validParams();
    p.setClassDescription(
        "The stiffness-proportional part b K v of Rayleigh damping, for one displacement "
        "component u_i, with K the stiffness matrix and v the velocity. In the balance law dm/dt "
        "+ div F = s of u_i, the flux is F = -b H (C : d(epsilon)/dt)_i, the row i of the stress "
        "rate times the coefficient b and the thickness H. C is the elastic stiffness of small "
        "strain, isotropic or given by 'stiffness', and d(epsilon)/dt the strain rate of the "
        "velocity that the time scheme gives. In axisymmetric problems the radial equation also "
        "has the hoop source, as in stress_divergence. With second_time_derivative (the inertia "
        "rho d2u/dt2, rho the density) and time_derivative with the coefficient a rho, it gives "
        "the Rayleigh damping a M + b K, with M the mass matrix and a (1/s) the mass coefficient.");
    addDisplacementsParam(p,
                          "The position of 'variable' in this list is the component i of its "
                          "equation.");
    p.addRequired("formulation",
                  ParameterKind::String,
                  "The formulation of the solid, as small_strain_stress takes it: plane_stress, "
                  "plane_strain, axisymmetric or three_dimensional.");
    p.addRequired("coefficient", ParameterKind::Real, "The damping coefficient b (s).");
    p.setRange("coefficient", ParameterRange::NonNegative);
    p.addOptional("youngs_modulus",
                  ParameterKind::Function,
                  std::string("youngs_modulus"),
                  "Young's modulus E (Pa), in any form that small_strain_stress takes. Default: "
                  "the property youngs_modulus, the one small_strain_stress reads by default.");
    p.addOptional("poissons_ratio",
                  ParameterKind::Function,
                  std::string("poissons_ratio"),
                  "Poisson's ratio nu (dimensionless), in any form that small_strain_stress takes. "
                  "Default: the property poissons_ratio, the one small_strain_stress reads by "
                  "default.");
    p.addOptional("thickness",
                  ParameterKind::Real,
                  1.0,
                  "The out-of-plane thickness H (m) of a plane problem, as stress_divergence "
                  "takes it. Default 1, the default of stress_divergence.");
    p.setRange("thickness", ParameterRange::Positive);
    p.addOptional("stiffness",
                  ParameterKind::RealList,
                  std::vector<double>{},
                  "An anisotropic stiffness C (Pa), 36 values as small_strain_stress takes them, "
                  "in place of youngs_modulus and poissons_ratio. Default none: the isotropic "
                  "stiffness of youngs_modulus and poissons_ratio.");
    return p;
  }

  explicit StiffnessDamping(const InputParameters & p)
      : Kernel(p),
        _component(displacementComponent(name(), variableName(), p.getStringList("displacements"))),
        _formulation(formulationFromName(name(), p.getString("formulation"))),
        _b(p.getReal("coefficient")), _thickness(p.getReal("thickness"))
  {
    if (_formulation == Formulation::AxisymmetricOneD)
      throw InputError("'" + name() +
                       "': the axisymmetric_1d formulation takes no stiffness "
                       "damping.");
    if (_formulation == Formulation::GeneralizedPlaneStrain)
      throw InputError("'" + name() +
                       "': stiffness damping is available in plane stress, plane strain, "
                       "axisymmetric and three-dimensional problems, not in generalized plane "
                       "strain.");
    if (_thickness != 1.0 && _formulation != Formulation::PlaneStress &&
        _formulation != Formulation::PlaneStrain)
      throw InputError("'" + name() +
                       "': thickness is the out-of-plane thickness of a plane problem. A "
                       "three-dimensional or axisymmetric problem has none, so leave it at 1.");
    const auto stiffness = p.getRealList("stiffness");
    if (!stiffness.empty())
    {
      if (stiffness.size() != 36)
        throw InputError("'" + name() + "': stiffness takes 36 values.");
      std::copy(stiffness.begin(), stiffness.end(), _stiffness.begin());
      _anisotropic = true;
    }
  }

  void initialSetup(Problem & problem) override
  {
    Kernel::initialSetup(problem);
    for (const auto & d : _params.getStringList("displacements"))
      _disp.push_back(problem.variableIndex(d));
    if (!_anisotropic)
    {
      _E.setup(problem, _params, "youngs_modulus", name());
      _nu.setup(problem, _params, "poissons_ratio", name());
    }
  }

  bool isTimeKernel() const override { return true; }
  bool hasFlux() const override { return true; }
  bool hasSource() const override
  {
    return _formulation == Formulation::Axisymmetric && _component == 0;
  }

  void computeFlux(const QpContext & ctx, ADVector3 & F) const override
  {
    std::array<ADReal, 6> sigma;
    stressRate(ctx, sigma);
    for (int d = 0; d < ctx.dim; ++d)
      F[d] = _b * _thickness * sigma[kVoigt[_component][d]];
  }

  ADReal computeSource(const QpContext & ctx) const override
  {
    if (ctx.x[0] == 0.0)
      return ADReal(0.0);
    std::array<ADReal, 6> sigma;
    stressRate(ctx, sigma);
    return _b * sigma[2] / ctx.x[0];
  }

private:
  /// C : d(eps)/dt, from the rates of the displacements.
  void stressRate(const QpContext & ctx, std::array<ADReal, 6> & sigma) const
  {
    std::array<ADReal, 6> e;
    smallStrain(
        _formulation,
        _disp,
        ctx.x,
        [&](int v) { return ctx.rateGradient(v); },
        [&](int v) { return ctx.rate(v); },
        e.data());
    if (_anisotropic)
    {
      anisotropicStress(_formulation, e.data(), _stiffness, sigma.data());
      return;
    }
    const ADReal E = _E.value(ctx), nu = _nu.value(ctx);
    const auto [lambda, mu] = lameConstants(E, nu);
    isotropicStress(_formulation, e.data(), lambda, mu, sigma.data());
  }

  bool _anisotropic = false;
  std::array<double, 36> _stiffness{};
  int _component;
  Formulation _formulation;
  double _b, _thickness;
  std::vector<int> _disp;
  FieldCoefficient _E, _nu;
};

} // namespace

// ---- SmallStrainStress: the elastic law, viscoelasticity and phase-field fracture ----

InputParameters
SmallStrainStress::validParams()
{
  InputParameters p = Property::validParams();
  p.setClassDescription(
      "Small-strain stress of an elastic solid, isotropic or anisotropic, with optional "
      "creep, plasticity, viscoelasticity and phase-field fracture. For an isotropic material "
      "sigma = lambda tr(epsilon_e) I + 2 G epsilon_e, with lambda the first Lame parameter, G "
      "the shear modulus and epsilon_e the elastic strain: the strain of the displacements "
      "less the eigenstrains and the creep and plastic strains. Young's modulus and Poisson's "
      "ratio are constants, functions, properties or expressions. The formulations are "
      "plane_stress (a thin body free to contract through its thickness, sigma_zz = 0), "
      "plane_strain (a long body restrained along its axis), generalized_plane_strain (a long "
      "body whose uniform axial strain is an unknown, 'out_of_plane_strain'), axisymmetric (a "
      "body of revolution on an (r, z) mesh), axisymmetric_1d (a radial slice of a long body "
      "of revolution, whose uniform axial strain is 'axial_strain') and three_dimensional. A "
      "thermal strain is an eigenstrain, for example of thermal_expansion_eigenstrain. "
      "Declares 'stress', 'strain' and 'elastic_strain' in the Voigt order xx, yy, zz, yz, xz, "
      "xy, with engineering shear strains and tensor shear stresses. In an axisymmetric "
      "formulation, x, y and z stand for r, z and the hoop direction theta. Also declares "
      "'volumetric_strain', 'von_mises_stress' and 'deformation_gradient', with creep "
      "'creep_strain' and 'equivalent_creep_strain', with plasticity 'plastic_strain', "
      "'equivalent_plastic_strain' and 'back_stress', and with damage "
      "'fracture_driving_energy'.");
  p.addRequired("displacements",
                ParameterKind::StringList,
                "The displacement variables (m): two in a plane or axisymmetric problem (the "
                "radial and the axial one in the axisymmetric case), three in three "
                "dimensions and the radial one in axisymmetric_1d.");
  p.addRequired("formulation",
                ParameterKind::String,
                "The idealization of the body: plane_stress, plane_strain, "
                "generalized_plane_strain, axisymmetric, axisymmetric_1d or "
                "three_dimensional.");
  p.addOptional("youngs_modulus",
                ParameterKind::Function,
                std::string("youngs_modulus"),
                "Young's modulus E (Pa): a constant, a registered function, the name of a "
                "property, or an expression of the fields, x, y, z, t, the functions and the "
                "properties. Default: the property youngs_modulus, which a property object such "
                "as constant or parsed declares, because no number suits every material.");
  p.addOptional("poissons_ratio",
                ParameterKind::Function,
                std::string("poissons_ratio"),
                "Poisson's ratio nu (dimensionless), with -1 < nu < 1/2: a constant, a "
                "registered function, the name of a property, or an expression. Default: the "
                "property poissons_ratio, which a property object such as constant or parsed "
                "declares, because no number suits every material.");
  p.addOptional("eigenstrain_names",
                ParameterKind::StringList,
                std::vector<std::string>{},
                "Names of the six-component eigenstrain properties to subtract from the strain. "
                "Default none: no eigenstrain.");
  p.addOptional("axial_strain",
                ParameterKind::Function,
                0.0,
                "The uniform axial strain epsilon_zz (dimensionless) of the axisymmetric_1d "
                "formulation, a constant or a function. Default 0: the body is held at its "
                "length.");
  p.addOptional("creep_model",
                ParameterKind::String,
                std::string("none"),
                "Creep of the material: none, or parsed for the rate given by 'creep_rate'. The "
                "creep strain is integrated by the backward Euler method with a radial return "
                "on the von Mises stress, and kept as a history at every integration point. "
                "Default none: no creep.");
  p.addOptional("creep_rate",
                ParameterKind::String,
                std::string(""),
                "With creep_model = parsed, the equivalent creep strain rate (1/s), as an "
                "expression of von_mises_stress (Pa), temperature (K), grain_size (m), the time "
                "t (s), for primary creep by time hardening, equivalent_creep_strain, for strain "
                "hardening, taken at the end of the step, and the constants of "
                "'creep_constants'. It is differentiated automatically, so the Jacobian of the "
                "return mapping is exact. It must not be negative. Default empty, because "
                "creep_model = none needs no rate.");
  p.addOptional("creep_constants",
                ParameterKind::RealMap,
                RealMap{},
                "Named constants that 'creep_rate' uses, as name -> value. Default none.");
  p.addOptional("temperature",
                ParameterKind::String,
                std::string(""),
                "The temperature variable (K). Required when the creep rate or the yield "
                "stress uses temperature. Default none, for a material law that does not read "
                "the temperature.");
  p.addOptional("grain_size",
                ParameterKind::Real,
                std::monostate{},
                "Grain size (m) of the material, the grain diameter. A mean linear intercept L "
                "corresponds to the grain size 1.56 L. Required when the parsed creep rate "
                "uses grain_size.");
  p.addOptional("creep_rate_factor",
                ParameterKind::Real,
                1.0,
                "Factor (dimensionless, not negative) on the creep rate, for sensitivity and "
                "uncertainty studies. Default 1: the creep rate as given.");
  p.addOptional("out_of_plane_strain",
                ParameterKind::String,
                std::string(""),
                "With formulation = generalized_plane_strain, the variable of the uniform axial "
                "strain epsilon_zz, one unknown for the whole body (a uniform_field), whose "
                "equation (out_of_plane_stress) sets the axial force the section carries. "
                "Required with generalized_plane_strain, and unused by the other formulations.");
  p.addOptional("viscous_shear_moduli",
                ParameterKind::RealList,
                std::vector<double>{},
                "Viscoelasticity (generalized Maxwell model): the shear moduli G_i (Pa) of the "
                "deviatoric branches, each relaxing as exp(-t/tau_i), beside the long-term "
                "moduli E and nu. Viscoelasticity is not available with creep, plasticity or "
                "plane_stress. Default none: no viscous branch.");
  p.addOptional("viscous_shear_times",
                ParameterKind::RealList,
                std::vector<double>{},
                "The relaxation times tau_i (s) of the shear branches, one per shear modulus. "
                "Default none: no viscous branch.");
  p.addOptional("viscous_bulk_moduli",
                ParameterKind::RealList,
                std::vector<double>{},
                "The bulk moduli K_i (Pa) of the volumetric branches of the generalized Maxwell "
                "model. Default none: no viscous branch.");
  p.addOptional("viscous_bulk_times",
                ParameterKind::RealList,
                std::vector<double>{},
                "The relaxation times (s) of the bulk branches, one per bulk modulus. Default "
                "none: no viscous branch.");
  p.addOptional("damage",
                ParameterKind::String,
                std::string(""),
                "Phase-field fracture with the AT2 functional: the damage variable d, which "
                "degrades the tensile part of the elastic energy by (1 - d)^2, with the split of "
                "Miehe by the principal strains. Declares fracture_driving_energy (J/m^3), the "
                "largest tensile energy the point has had, so the crack does not heal. "
                "Available for an isotropic elastic material in plane_strain, "
                "generalized_plane_strain and axisymmetric. Default none: no fracture.");
  p.addOptional("yield_stress",
                ParameterKind::String,
                std::string(""),
                "Von Mises (J2) plasticity: the yield stress (Pa), as an expression of "
                "equivalent_plastic_strain, temperature (K) and the constants of "
                "'plasticity_constants', which gives the isotropic hardening. A number gives "
                "perfect plasticity. The plastic strain is integrated by the backward Euler "
                "method with a radial return, with creep in the same return. Default empty: no "
                "plasticity.");
  p.addOptional("plasticity_constants",
                ParameterKind::RealMap,
                RealMap{},
                "Named constants that 'yield_stress' uses, as name -> value. Default none.");
  p.addOptional("kinematic_hardening_moduli",
                ParameterKind::RealList,
                std::vector<double>{},
                "The moduli C_k (Pa) of the back stresses X_k of kinematic hardening, by the "
                "law of Armstrong and Frederick, or of Chaboche with several: dX_k = (2/3) C_k "
                "d(epsilon_p) - gamma_k X_k dp, with epsilon_p the plastic strain and p the "
                "equivalent plastic strain. Not available with creep, an anisotropic stiffness "
                "or hill_coefficients. Default none: isotropic hardening only.");
  p.addOptional("dynamic_recovery",
                ParameterKind::RealList,
                std::vector<double>{},
                "The dynamic recovery gamma_k (dimensionless) of each back stress. Default 0 "
                "for each: the linear kinematic hardening of Prager.");
  p.addOptional("stiffness",
                ParameterKind::RealList,
                std::vector<double>{},
                "An anisotropic material: the 6 x 6 stiffness C (Pa) in the global axes and the "
                "Voigt order xx, yy, zz, yz, xz, xy, with engineering shear strains, row by row "
                "(36 values), in place of youngs_modulus and poissons_ratio. It must be "
                "symmetric and positive definite. Default none: the isotropic stiffness of "
                "youngs_modulus and poissons_ratio.");
  p.addOptional(
      "hill_coefficients",
      ParameterKind::RealList,
      std::vector<double>{},
      "Hill's anisotropic equivalent stress sqrt(sigma . L_H sigma) for creep and "
      "plasticity, with L_H the 6 x 6 Hill matrix (dimensionless) in the global axes and the "
      "Voigt order, with tensor shear stresses, row by row (36 values). With it the "
      "creep rate and the yield stress read this equivalent stress as "
      "von_mises_stress, and the inelastic flow is normal to it. It needs creep or "
      "plasticity. Default none: the von Mises stress.");
  return p;
}

SmallStrainStress::SmallStrainStress(const InputParameters & p)
    : Property(p), _formulation(formulationFromName(name(), p.getString("formulation")))
{
  checkModuli(p);
  parseCreep(p);
  parsePlasticity(p);
  parseViscous(p);
  parseDamage(p);
  parseStiffness(p);
  parseHill(p);
  checkHardening();
}

void
SmallStrainStress::checkModuli(const InputParameters & p) const
{
  // A constant is checked now, and a property or an expression where it is used.
  if (const auto * E = std::get_if<double>(&p.getRaw("youngs_modulus")); E && !(*E > 0.0))
    throw InputError("'" + name() + "': youngs_modulus must be positive.");
  if (const auto * nu = std::get_if<double>(&p.getRaw("poissons_ratio"));
      nu && !(*nu > -1.0 && *nu < 0.5))
    throw InputError("'" + name() + "': poissons_ratio must lie in (-1, 0.5).");
}

void
SmallStrainStress::parseCreep(const InputParameters & p)
{
  const auto c = p.getString("creep_model");
  if (c != "none" && c != "parsed")
    throw InputError("'" + name() + "': unknown creep_model '" + c + "' (none, parsed).");
  if (c == "parsed")
  {
    const auto text = p.getString("creep_rate");
    if (text.empty())
      throw InputError("'" + name() + "': creep_model = parsed needs 'creep_rate'.");
    std::vector<std::string> names = {
        "von_mises_stress", "temperature", "grain_size", "t", "equivalent_creep_strain"};
    for (const auto & [constant, value] : p.getRealMap("creep_constants"))
    {
      names.push_back(constant);
      _creep_constants.push_back(value);
    }
    if (names.size() > 32)
      throw InputError("'" + name() + "': too many creep constants (at most 27).");
    _creep_expression = std::make_shared<ParsedExpression>(text, names);
  }
  else if (p.isSetByUser("creep_rate"))
    throw InputError("'" + name() + "': 'creep_rate' is used only with creep_model = parsed.");
  // The grain size is required only when the creep rate reads it (the argument 2 of the parsed
  // creep rate).
  if (creeps() && _creep_expression->uses(2) && !p.isSet("grain_size"))
    throw InputError("'" + name() +
                     "': a creep_rate that uses grain_size needs 'grain_size' (m), the grain "
                     "size of the material.");
  if (p.isSet("grain_size") && !(p.getReal("grain_size") > 0.0))
    throw InputError("'" + name() + "': grain_size must be positive.");
  _grain = p.isSet("grain_size") ? p.getReal("grain_size") : 0.0;
  _creep_factor = p.getReal("creep_rate_factor");
  if (!(_creep_factor >= 0.0))
    throw InputError("'" + name() + "': creep_rate_factor must not be negative.");
}

void
SmallStrainStress::parsePlasticity(const InputParameters & p)
{
  const std::string yield = p.getString("yield_stress");
  _plastic = !yield.empty();
  if (_plastic)
  {
    std::vector<std::string> names = {"equivalent_plastic_strain", "temperature"};
    for (const auto & [constant, value] : p.getRealMap("plasticity_constants"))
    {
      names.push_back(constant);
      _plasticity_constants.push_back(value);
    }
    if (names.size() > 32)
      throw InputError("'" + name() + "': too many plasticity constants (at most 30).");
    _yield_expression = std::make_shared<ParsedExpression>(yield, names);
  }
  const auto moduli = p.getRealList("kinematic_hardening_moduli");
  auto recovery = p.getRealList("dynamic_recovery");
  if (recovery.empty())
    recovery.assign(moduli.size(), 0.0);
  if (recovery.size() != moduli.size())
    throw InputError("'" + name() +
                     "': dynamic_recovery needs one value per kinematic "
                     "hardening modulus.");
  if (!moduli.empty() && !_plastic)
    throw InputError("'" + name() +
                     "': kinematic hardening is part of plasticity. Give the "
                     "yield_stress.");
  for (std::size_t k = 0; k < moduli.size(); ++k)
  {
    if (!(moduli[k] > 0.0 && recovery[k] >= 0.0))
      throw InputError("'" + name() +
                       "': the kinematic hardening moduli are positive and the "
                       "dynamic recovery not negative.");
    _hardening.push_back({moduli[k], recovery[k]});
  }
}

void
SmallStrainStress::parseViscous(const InputParameters & p)
{
  for (const char * kind : {"shear", "bulk"})
  {
    const auto moduli = p.getRealList(std::string("viscous_") + kind + "_moduli");
    const auto times = p.getRealList(std::string("viscous_") + kind + "_times");
    if (moduli.size() != times.size())
      throw InputError("'" + name() + "': each viscous " + kind +
                       " branch takes a modulus and a relaxation time.");
    for (std::size_t i = 0; i < moduli.size(); ++i)
    {
      if (!(moduli[i] > 0.0 && times[i] > 0.0))
        throw InputError("'" + name() +
                         "': the moduli and the relaxation times of the "
                         "viscous branches are positive.");
      _viscous.push_back({moduli[i], times[i]});
    }
    if (std::string(kind) == "shear")
      _shear_branches = _viscous.size();
  }
  if (!_viscous.empty() && (inelastic() || _formulation == Formulation::PlaneStress))
    throw InputError("'" + name() +
                     "': viscoelasticity with creep or plasticity, or in plane "
                     "stress, is not available.");
}

void
SmallStrainStress::parseDamage(const InputParameters & p)
{
  _cracking = !p.getString("damage").empty();
  if (_cracking &&
      (_formulation == Formulation::PlaneStress || _formulation == Formulation::ThreeDimensional ||
       _formulation == Formulation::AxisymmetricOneD || inelastic() || !_viscous.empty() ||
       !p.getRealList("stiffness").empty()))
    throw InputError("'" + name() +
                     "': phase-field fracture is available for an elastic "
                     "isotropic material in plane strain, generalized plane "
                     "strain and axisymmetric problems.");
}

void
SmallStrainStress::parseStiffness(const InputParameters & p)
{
  const auto stiffness = p.getRealList("stiffness");
  if (stiffness.empty())
    return;
  if (stiffness.size() != 36)
    throw InputError("'" + name() + "': stiffness takes 36 values, the 6 x 6 matrix row by row.");
  if (p.isSetByUser("youngs_modulus") || p.isSetByUser("poissons_ratio"))
    throw InputError("'" + name() +
                     "': give the stiffness or youngs_modulus and "
                     "poissons_ratio, not both.");
  std::copy(stiffness.begin(), stiffness.end(), _stiffness.begin());
  checkSymmetricPositive(_stiffness, "stiffness");
  _anisotropic = true;
}

void
SmallStrainStress::parseHill(const InputParameters & p)
{
  // Von Mises as the default equivalent stress: q^2 = sigma . P sigma.
  for (int i = 0; i < 3; ++i)
  {
    for (int j = 0; j < 3; ++j)
      _hill[6 * i + j] = i == j ? 1.0 : -0.5;
    _hill[6 * (i + 3) + i + 3] = 3.0;
  }
  const auto hill = p.getRealList("hill_coefficients");
  if (hill.empty())
    return;
  if (hill.size() != 36)
    throw InputError("'" + name() +
                     "': hill_coefficients takes 36 values, the 6 x 6 matrix "
                     "row by row.");
  if (!inelastic())
    throw InputError("'" + name() +
                     "': hill_coefficients is the equivalent stress of creep "
                     "and plasticity. Give one of them.");
  if (!_hardening.empty())
    throw InputError("'" + name() +
                     "': Hill plasticity with kinematic hardening is not "
                     "available.");
  std::copy(hill.begin(), hill.end(), _hill.begin());
  _hill_given = true;
}

void
SmallStrainStress::checkHardening() const
{
  if (_anisotropic && !_hardening.empty())
    throw InputError("'" + name() +
                     "': kinematic hardening with an anisotropic stiffness is "
                     "not available.");
  if (creeps() && !_hardening.empty())
    throw InputError("'" + name() +
                     "': creep with kinematic hardening is not available "
                     "(creep acts on the stress, plasticity on the stress "
                     "less the back stress). Use isotropic hardening with "
                     "creep.");
}

void
SmallStrainStress::initialSetup(Problem & problem)
{
  Property::initialSetup(problem);
  const int dim = problem.mesh().dimension();
  if (problem.mesh().spaceDimension() > dim)
    throw InputError("'" + name() +
                     "': the mesh is a surface (or a curve) in space, and the continuum "
                     "formulations need a mesh that fills its space.  Surfaces are shells.");
  // The temperature variable, which a creep rate or a yield stress that reads it needs.
  const bool creep_reads = creeps() && _creep_expression->uses(1);
  const bool yield_reads = _plastic && _yield_expression->uses(1);
  if (!_params.getString("temperature").empty())
    _T = coupledVariable(problem, "temperature");
  else if (creep_reads || yield_reads)
    throw InputError("'" + name() + "': a " + (creep_reads ? "creep_rate" : "yield_stress") +
                     " that uses temperature needs 'temperature'.");
  _disp.clear();
  for (const auto & n : _params.getStringList("displacements"))
    _disp.push_back(problem.variableIndex(n));
  const int expected = formulationDimension(_formulation);
  const std::string formulation = _params.getString("formulation");
  if (static_cast<int>(_disp.size()) != expected || dim != expected)
    throw InputError("'" + name() + "': the " + formulation + " formulation needs " +
                     std::to_string(expected) + " displacement variable(s) on a " +
                     std::to_string(expected) + "-dimensional mesh, and has " +
                     std::to_string(_disp.size()) + " on a " + std::to_string(dim) +
                     "-dimensional mesh.");
  const bool axisymmetric_coordinates =
      problem.coordinateSystem() == CoordinateSystem::Axisymmetric;
  if (isAxisymmetric(_formulation) != axisymmetric_coordinates)
    throw InputError("'" + name() + "': the " + formulation + " formulation " +
                     (axisymmetric_coordinates
                          ? "is Cartesian, and the problem has axisymmetric coordinates. Use "
                            "the axisymmetric formulation."
                          : "needs coordinates = 'axisymmetric'."));
  _axial = getFunction(problem, "axial_strain");
  if (_cracking)
    _damage = coupledVariable(problem, "damage");
  if (_formulation == Formulation::GeneralizedPlaneStrain)
  {
    if (_params.getString("out_of_plane_strain").empty())
      throw InputError("'" + name() +
                       "': generalized_plane_strain needs the variable "
                       "'out_of_plane_strain'.");
    _out_of_plane = coupledVariable(problem, "out_of_plane_strain");
  }
  auto & reg = problem.propertyRegistry();
  _registry = &reg;
  if (!_anisotropic && !_params.isSetByUser("youngs_modulus") && !reg.has("youngs_modulus"))
    throw InputError("'" + name() +
                     "': Young's modulus is not given, and no property object declares the "
                     "property youngs_modulus. Give youngs_modulus (Pa), or add a property "
                     "object that declares it, such as constant or parsed.");
  if (!_anisotropic && !_params.isSetByUser("poissons_ratio") && !reg.has("poissons_ratio"))
    throw InputError("'" + name() +
                     "': Poisson's ratio is not given, and no property object declares the "
                     "property poissons_ratio. Give poissons_ratio, or add a property object "
                     "that declares it, such as constant or parsed.");
  if (!_anisotropic)
  {
    _E.setup(problem, _params, "youngs_modulus", name());
    _nu.setup(problem, _params, "poissons_ratio", name());
  }
  // Constant moduli are turned into the Lame constants once.
  const auto * E = std::get_if<double>(&_params.getRaw("youngs_modulus"));
  const auto * nu = std::get_if<double>(&_params.getRaw("poissons_ratio"));
  _constant_moduli = E && nu;
  if (_constant_moduli)
  {
    const auto lame = lameConstants(*E, *nu);
    _mu0 = lame.mu;
    _lambda0 = lame.lambda;
  }
  _eigen.clear();
  for (const auto & n : _params.getStringList("eigenstrain_names"))
    _eigen.push_back(reg.id(n));
}

void
SmallStrainStress::declareProperties(PropertyRegistry & r)
{
  _stress = r.declare("stress", 6);
  _strain = r.declare("strain", 6);
  _elastic = r.declare("elastic_strain", 6);
  _volumetric = r.declare("volumetric_strain", 1);
  _vm = r.declare("von_mises_stress", 1);
  _F_prop = r.declare("deformation_gradient", 9);
  if (creeps())
  {
    _creep_prop = r.declare("creep_strain", 6);
    _creep_eq = r.declare("equivalent_creep_strain", 1);
  }
  if (_cracking)
    _driving = r.declare("fracture_driving_energy", 1);
  if (_plastic)
  {
    _plastic_prop = r.declare("plastic_strain", 6);
    _plastic_eq = r.declare("equivalent_plastic_strain", 1);
    _back_prop = r.declare("back_stress", 6);
  }
}

void
SmallStrainStress::checkSymmetricPositive(const std::array<double, 36> & A,
                                          const std::string & what) const
{
  double scale = 0.0;
  for (double a : A)
    scale = std::max(scale, std::abs(a));
  for (int i = 0; i < 6; ++i)
    for (int j = 0; j < i; ++j)
      if (std::abs(A[6 * i + j] - A[6 * j + i]) > 1e-12 * scale)
        throw InputError("'" + name() + "': the " + what + " is not symmetric.");
  std::array<double, 36> L{};
  for (int j = 0; j < 6; ++j)
  {
    double d = A[6 * j + j];
    for (int k = 0; k < j; ++k)
      d -= L[6 * j + k] * L[6 * j + k];
    if (!(d > 1e-12 * scale))
      throw InputError("'" + name() + "': the " + what +
                       " is not positive definite (an unstable material).");
    L[6 * j + j] = std::sqrt(d);
    for (int i = j + 1; i < 6; ++i)
    {
      double v = A[6 * i + j];
      for (int k = 0; k < j; ++k)
        v -= L[6 * i + k] * L[6 * j + k];
      L[6 * i + j] = v / L[6 * j + j];
    }
  }
}

void
SmallStrainStress::displacementStrain(const QpContext & ctx, ADReal * e) const
{
  smallStrain(
      _formulation,
      _disp,
      ctx.x,
      [&](int v) -> const ADVector3 & { return ctx.gradient(v); },
      [&](int v) -> const ADReal & { return ctx.value(v); },
      e);
  if (_formulation == Formulation::AxisymmetricOneD)
    e[1] = ADReal(_axial->value(ctx.x, ctx.time));
  else if (_formulation == Formulation::GeneralizedPlaneStrain)
    e[2] = ctx.value(_out_of_plane);
}

void
SmallStrainStress::computeProperties(QpContext & ctx) const
{
  // The strain, the elastic strain and the stress are formed in their property storage.
  ADReal * e = &ctx.property(_strain, 0);
  ADReal * ee = &ctx.property(_elastic, 0);
  ADReal * sigma = &ctx.property(_stress, 0);
  displacementStrain(ctx, e);
  for (int i = 0; i < 6; ++i)
    ee[i] = e[i];
  for (int id : _eigen)
    for (int i = 0; i < 6; ++i)
      ee[i] -= ctx.property(id, i);
  // The creep and plastic strains at the start of the step (the history of this point).
  const double * old = nullptr;
  double * next = nullptr;
  if ((inelastic() || !_viscous.empty() || _cracking) && ctx.state_old)
  {
    old = ctx.state_old + stateOffset();
    next = ctx.state_new + stateOffset();
    if (inelastic())
      for (int i = 0; i < 6; ++i)
        ee[i] -= old[kCreepStrain + i] + old[kPlasticStrain + i];
  }
  ADReal mu, lambda;
  if (_anisotropic)
    anisotropicStress(_formulation, ee, _stiffness, sigma);
  else if (_constant_moduli)
  {
    isotropicStress(_formulation, ee, _lambda0, _mu0, sigma);
    mu = _mu0;
    lambda = _lambda0;
  }
  else
  {
    const ADReal E = _E.value(ctx);
    const ADReal nu = _nu.value(ctx);
    const auto lame = lameConstants(E, nu);
    mu = lame.mu;
    lambda = lame.lambda;
    isotropicStress(_formulation, ee, lambda, mu, sigma);
  }
  if (inelastic() && _formulation == Formulation::PlaneStress && old)
    planeStressInelasticStrain(ctx, lambda, mu, old, ee, sigma);
  if (_formulation == Formulation::PlaneStress)
  {
    // The total zz strain: the elastic one that keeps sigma_zz zero, plus the eigenstrains
    // and the inelastic strains of the start of the step.
    e[2] = ee[2];
    for (int id : _eigen)
      e[2] += ctx.property(id, 2);
    if (old)
      e[2] += old[kCreepStrain + 2] + old[kPlasticStrain + 2];
  }
  if (inelastic())
    updateInelastic(ctx, mu, old, next, ee, sigma);
  if (!_viscous.empty())
    addViscousStress(ctx, ee, old, next, sigma);
  if (_cracking)
    crackedStress(ctx, ee, lambda, mu, old, next, sigma);
  ctx.property(_volumetric) = e[0] + e[1] + e[2];
  // Two outputs that the equilibrium itself does not need are formed only when an object
  // reads them.
  if (_registry->isRead(_F_prop))
    deformationGradient(ctx, e);
  if (_registry->isRead(_vm))
    ctx.property(_vm) = vonMisesStress(sigma);
}

void
SmallStrainStress::addViscousStress(
    QpContext & ctx, const ADReal * ee, const double * old, double * next, ADReal * sigma) const
{
  const int offset = inelasticSize();
  // The volumetric strain and the deviatoric strain as a tensor, with half the engineering
  // shears, now and at the start of the step.
  const ADReal theta = ee[0] + ee[1] + ee[2];
  std::array<ADReal, 6> strain_deviator = deviator(ee);
  for (int i = 3; i < 6; ++i)
    strain_deviator[i] = 0.5 * ee[i];
  std::array<double, 6> deviator_old{0, 0, 0, 0, 0, 0};
  double theta_old = 0.0;
  if (old)
  {
    const double * e0 = old + offset;
    theta_old = e0[0] + e0[1] + e0[2];
    deviator_old = deviator(e0);
    for (int i = 3; i < 6; ++i)
      deviator_old[i] = 0.5 * e0[i];
  }
  // A steady solve is relaxed, and a step of no length (an output after a step) is the
  // instantaneous limit, decay 1 and ramp 1.
  const bool step = old && !ctx.steady_solve;
  int slot = offset + 6;
  for (std::size_t b = 0; b < _viscous.size(); ++b)
  {
    const bool shear = b < _shear_branches;
    const double x = ctx.dt / _viscous[b].time;
    const double decay = step ? std::exp(-x) : 0.0;
    const double ramp = step ? (x > 0.0 ? -std::expm1(-x) / x : 1.0) : 0.0;
    const int count = shear ? 6 : 1;
    for (int i = 0; i < count; ++i)
    {
      const double previous = step ? old[slot + i] : 0.0;
      ADReal branch;
      if (shear)
        branch = decay * previous +
                 2.0 * _viscous[b].modulus * ramp * (strain_deviator[i] - deviator_old[i]);
      else
        branch = decay * previous + _viscous[b].modulus * ramp * (theta - theta_old);
      if (shear)
        sigma[i] += branch;
      else
        for (int j = 0; j < 3; ++j)
          sigma[j] += branch;
      if (next)
        next[slot + i] = branch.value();
    }
    slot += count;
  }
  if (next)
    for (int i = 0; i < 6; ++i)
      next[offset + i] = ee[i].value();
}

void
SmallStrainStress::crackedStress(QpContext & ctx,
                                 const ADReal * ee,
                                 const ADReal & lambda,
                                 const ADReal & mu,
                                 const double * old,
                                 double * next,
                                 ADReal * sigma) const
{
  const ADReal mean = 0.5 * (ee[0] + ee[1]);
  const ADReal half = 0.5 * (ee[0] - ee[1]), shear = 0.5 * ee[5];
  const ADReal d = ctx.value(_damage);
  const ADReal g = (1.0 - d) * (1.0 - d);
  // A zero strain belongs to the compressive part, so that the undeformed state has the
  // stiffness of the intact material.
  const auto positive = [](const ADReal & x) { return x.value() > 0.0 ? x : ADReal(0.0); };
  const auto negative = [](const ADReal & x) { return x.value() <= 0.0 ? x : ADReal(0.0); };
  for (int i = 0; i < 6; ++i)
    sigma[i] = 0.0;
  ADReal energy(0.0);
  // The in-plane principal strains mean +- radius, along the directions whose projectors
  // n_k n_k are (1 +- c, 1 -+ c, +- s) / 2.  Where the two are equal (radius zero, the
  // undeformed state among others), the in-plane strain is on one side as a whole, by the
  // sign of the mean, and the split is linear there.
  const double spread = std::hypot(half.value(), shear.value());
  const double size = std::abs(ee[0].value()) + std::abs(ee[1].value()) + std::abs(ee[5].value());
  if (spread > 1e-12 * size)
  {
    const ADReal radius = sqrt(half * half + shear * shear);
    const ADReal c = half / radius, s = shear / radius;
    const std::array<ADReal, 2> principal = {mean + radius, mean - radius};
    const std::array<std::array<ADReal, 3>, 2> P = {
        std::array<ADReal, 3>{0.5 * (1.0 + c), 0.5 * (1.0 - c), 0.5 * s},
        std::array<ADReal, 3>{0.5 * (1.0 - c), 0.5 * (1.0 + c), -0.5 * s}};
    for (int k = 0; k < 2; ++k)
    {
      energy += mu * positive(principal[k]) * positive(principal[k]);
      const ADReal part = 2.0 * mu * (g * positive(principal[k]) + negative(principal[k]));
      sigma[0] += part * P[k][0];
      sigma[1] += part * P[k][1];
      sigma[5] += part * P[k][2];
    }
  }
  else
  {
    const bool tensile = mean.value() > 0.0;
    const ADReal factor = tensile ? g : ADReal(1.0);
    // e : e of the in-plane strain, with the engineering shear.
    if (tensile)
      energy += mu * (ee[0] * ee[0] + ee[1] * ee[1] + 0.5 * ee[5] * ee[5]);
    sigma[0] += 2.0 * mu * factor * ee[0];
    sigma[1] += 2.0 * mu * factor * ee[1];
    sigma[5] += mu * factor * ee[5];
  }
  // The out-of-plane principal strain, and the volumetric part.
  energy += mu * positive(ee[2]) * positive(ee[2]);
  sigma[2] += 2.0 * mu * (g * positive(ee[2]) + negative(ee[2]));
  const ADReal trace = ee[0] + ee[1] + ee[2];
  energy += 0.5 * lambda * positive(trace) * positive(trace);
  const ADReal volumetric = lambda * (g * positive(trace) + negative(trace));
  for (int i = 0; i < 3; ++i)
    sigma[i] += volumetric;
  const int slot = inelasticSize() + viscousSize();
  const double history = old ? old[slot] : 0.0;
  const ADReal H = energy.value() > history ? energy : ADReal(history);
  ctx.property(_driving) = H;
  if (next)
    next[slot] = H.value();
}

void
SmallStrainStress::deformationGradient(QpContext & ctx, const ADReal * e) const
{
  for (int i = 0; i < 9; ++i)
    ctx.property(_F_prop, i) = ADReal(i % 4 == 0 ? 1.0 : 0.0);
  const auto & gu = ctx.gradient(_disp[0]);
  ctx.property(_F_prop, 0) = 1.0 + gu[0];
  ctx.property(_F_prop, 8) = 1.0 + e[2];
  if (_formulation == Formulation::AxisymmetricOneD)
    ctx.property(_F_prop, 4) = 1.0 + e[1];
  else
  {
    const auto & gv = ctx.gradient(_disp[1]);
    ctx.property(_F_prop, 1) = gu[1];
    ctx.property(_F_prop, 3) = gv[0];
    ctx.property(_F_prop, 4) = 1.0 + gv[1];
    if (_formulation == Formulation::ThreeDimensional)
    {
      const auto & gw = ctx.gradient(_disp[2]);
      ctx.property(_F_prop, 2) = gu[2];
      ctx.property(_F_prop, 5) = gv[2];
      ctx.property(_F_prop, 6) = gw[0];
      ctx.property(_F_prop, 7) = gw[1];
    }
  }
}

namespace
{

/// Isotropic thermal expansion as an eigenstrain, with the secant coefficient alpha, which may
/// depend on the temperature, the other fields and the position: eps = alpha(T) (T - T_0).
class ThermalExpansionEigenstrain : public Property
{
public:
  static InputParameters validParams()
  {
    InputParameters p = Property::validParams();
    p.setClassDescription(
        "Isotropic thermal strain alpha (T - T_0) on the three normal components, with alpha "
        "the linear expansion coefficient, T the temperature and T_0 the stress-free "
        "temperature, as a six-component eigenstrain property for small_strain_stress and "
        "finite_strain_stress. alpha is the secant (mean) coefficient between T_0 and T, so a "
        "coefficient that depends on the temperature gives the strain alpha(T) (T - T_0).");
    p.addRequired("temperature", ParameterKind::String, "The temperature variable T (K).");
    p.addRequired("linear_expansion_coefficient",
                  ParameterKind::Function,
                  "Linear expansion coefficient alpha (1/K): the strain per kelvin along each "
                  "direction, a third of the volumetric expansion coefficient beta of an "
                  "isotropic solid. It is the secant (mean) coefficient between T_0 and T, and "
                  "may be a constant, a function, or an expression of the temperature, the other "
                  "fields and the coordinates.");
    p.addRequired("stress_free_temperature",
                  ParameterKind::Real,
                  "Temperature T_0 (K) at which the thermal strain is zero.");
    p.addRequired("eigenstrain_name",
                  ParameterKind::String,
                  "Name of the eigenstrain property this object declares. List it in the "
                  "eigenstrain_names of the stress object.");
    return p;
  }
  explicit ThermalExpansionEigenstrain(const InputParameters & p)
      : Property(p), _Tref(p.getReal("stress_free_temperature"))
  {
  }
  void initialSetup(Problem & problem) override
  {
    Property::initialSetup(problem);
    _T = coupledVariable(problem, "temperature");
    _alpha.setup(problem, _params, "linear_expansion_coefficient", name());
  }
  void declareProperties(PropertyRegistry & r) override
  {
    _prop = r.declare(_params.getString("eigenstrain_name"), 6);
  }
  void computeProperties(QpContext & ctx) const override
  {
    const ADReal e = _alpha.value(ctx) * (ctx.value(_T) - _Tref);
    for (int i = 0; i < 6; ++i)
      ctx.property(_prop, i) = i < 3 ? e : ADReal(0.0);
  }

private:
  FieldCoefficient _alpha;
  double _Tref;
  int _T = -1, _prop = -1;
};

/// Prescribed traction component on a side set.
class TractionBC : public IntegratedBC
{
public:
  bool isLoad() const override { return true; }
  static InputParameters validParams()
  {
    InputParameters p = IntegratedBC::validParams();
    p.setClassDescription(
        "Prescribed component of the surface traction (Pa) on a boundary. The component is that "
        "of the displacement named by 'variable', and a list of variables adds one object per "
        "variable. Give the load either as the traction itself or as the total force on the "
        "boundary, which is then spread uniformly over its area. In a plane problem the traction "
        "is multiplied by the thickness of the stress_divergence kernel of the same variable.");
    p.addOptional("traction",
                  ParameterKind::Function,
                  0.0,
                  "The traction component t_i (Pa) along the axis of 'variable': a constant, a "
                  "registered function, or an expression of the fields, x, y, z and t. A positive "
                  "value acts along the positive direction of that axis, whichever way the "
                  "outward normal points. In a plane problem it is multiplied by the thickness. "
                  "Default 0: no load.");
    p.addOptional("total_force",
                  ParameterKind::Function,
                  0.0,
                  "The total force component F (N) along the axis of 'variable' on the whole "
                  "boundary, as a constant or the name of a function of time. It is applied as "
                  "the uniform traction F/A, with A (m^2) the area of the boundary, which carries "
                  "the factor 2 pi r in an axisymmetric problem. In a plane problem A is the "
                  "length of the boundary times the thickness, so that F is the force on the whole "
                  "thickness. Give either 'traction' or 'total_force'. Default 0: no load.");
    return p;
  }
  explicit TractionBC(const InputParameters & p) : IntegratedBC(p) {}
  void initialSetup(Problem & problem) override
  {
    IntegratedBC::initialSetup(problem);
    _thickness = planeThickness(problem, variableName(), name());
    const bool by_force = _params.isSetByUser("total_force");
    if (by_force && _params.isSetByUser("traction"))
      throw InputError("traction '" + _name + "': give only one of 'traction' and 'total_force'.");
    if (!by_force)
    {
      _t.setup(problem, _params, "traction", name());
      _factor = _thickness;
      return;
    }
    // The assembled load is the integral of factor * F over the boundary, in
    // which the integration weight includes the coordinate factor.  With
    // factor = 1 / A, A being that same integral of 1, the load equals F, and
    // the thickness of a plane problem cancels.
    _t.setup(problem, _params, "total_force", name());
    _area = problem.boundaryArea(_boundaries);
    if (!(_area > 0.0))
      throw InputError("traction '" + _name +
                       "': the boundary has zero area, so a total force cannot be spread over it.");
    _factor = 1.0 / _area;
  }
  ADReal computeBoundaryFlux(const QpContext & ctx) const override
  {
    return _factor * _t.value(ctx);
  }

private:
  double _thickness = 1.0;
  double _factor = 1.0;
  double _area = 0.0;
  FieldCoefficient _t;
};

/// A plane of symmetry: the displacement normal to the plane is zero.
class SymmetryBC : public NodalBC
{
public:
  static InputParameters validParams()
  {
    InputParameters p = NodalBC::validParams();
    p.setClassDescription(
        "A plane of symmetry of the body and its loads: the displacement normal to the plane "
        "is zero and the tangential displacements are free, so that only one side of the "
        "plane is modelled. The side set must lie in a plane normal to a coordinate axis, "
        "which is found from the coordinates of its nodes, and the condition holds the "
        "displacement along that axis at zero. 'variable' lists the displacement variables in "
        "the order of the coordinate axes, for example [u, v, w] in three dimensions.");
    p.setNormalComponent();
    return p;
  }
  explicit SymmetryBC(const InputParameters & p) : NodalBC(p) {}
  double computeValue(const Point &, double) const override { return 0.0; }
};

/// Prescribed normal pressure: t_i = -p n_i.
class PressureBC : public IntegratedBC
{
public:
  bool isLoad() const override { return true; }
  static InputParameters validParams()
  {
    InputParameters p = IntegratedBC::validParams();
    p.setClassDescription(
        "A pressure p (Pa) acting normal to a surface, which gives the traction t_i = -p n_i "
        "with n the outward normal, so a positive pressure pushes inward. Added without "
        "'variable', it acts on every displacement. In a plane problem the pressure is "
        "multiplied by the thickness of the stress_divergence kernel of the same variable.");
    addDisplacementsParam(p, "The pressure acts on the component of 'variable'.");
    p.addOptional("pressure",
                  ParameterKind::Function,
                  0.0,
                  "The pressure p (Pa): a constant, a registered function, or an expression of the "
                  "fields, x, y, z and t. A positive value presses inward, against the outward "
                  "normal. Default 0: no load.");
    deformation::addParameter(
        p,
        "the pressure acts on the deformed surface as a follower load, for equations on the "
        "undeformed mesh: the traction per unit undeformed area is -p J F_d^-T N (Nanson's "
        "relation), with F_d the deformation gradient, J = det(F_d) and N the undeformed "
        "outward normal. Without it the pressure acts along the undeformed normal on the "
        "undeformed area, which is accurate at small strain.");
    return p;
  }
  explicit PressureBC(const InputParameters & p)
      : IntegratedBC(p),
        _component(displacementComponent(name(), variableName(), p.getStringList("displacements")))
  {
  }
  void initialSetup(Problem & problem) override
  {
    IntegratedBC::initialSetup(problem);
    _thickness = planeThickness(problem, variableName(), name());
    _p.setup(problem, _params, "pressure", name());
    _F = deformation::propertyId(problem, _params, name());
  }
  ADReal computeBoundaryFlux(const QpContext & ctx) const override
  {
    const ADReal p = _thickness * _p.value(ctx);
    if (_F < 0)
      return -p * ctx.normal[_component];
    // Row 'component' of the cofactor J F^-T, times N.
    ADReal cof[3][3];
    deformation::cofactor(ctx, _F, cof);
    ADReal t(0.0);
    for (int K = 0; K < 3; ++K)
      if (ctx.normal[K] != 0.0)
        t = t + cof[_component][K] * ctx.normal[K];
    return -p * t;
  }

private:
  int _component;
  double _thickness = 1.0;
  int _F = -1;
  FieldCoefficient _p;
};

} // namespace

} // namespace solid_mechanics

void
registerSolidMechanicsObjects(Factory & f)
{
  using namespace solid_mechanics;
  const std::string m = "solid_mechanics";
  f.add<StressDivergence>("stress_divergence", ObjectCategory::Kernel, m);
  f.add<StiffnessDamping>("stiffness_damping", ObjectCategory::Kernel, m);
  f.add<OutOfPlaneStress>("out_of_plane_stress", ObjectCategory::Kernel, m);
  f.add<PhaseFieldDamage>("phase_field_damage", ObjectCategory::Kernel, m);
  f.add<UniformField>("uniform_field", ObjectCategory::Constraint, m);
  f.add<TractionBC>("traction", ObjectCategory::BoundaryCondition, m);
  f.add<PressureBC>("pressure", ObjectCategory::BoundaryCondition, m);
  f.add<SymmetryBC>("symmetry", ObjectCategory::NodalBC, m);
  f.add<SmallStrainStress>("small_strain_stress", ObjectCategory::Property, m);
  f.add<ThermalExpansionEigenstrain>("thermal_expansion_eigenstrain", ObjectCategory::Property, m);
  registerFiniteStrainObjects(f);
  registerInterfaceObjects(f);
  registerStructuralMemberObjects(f);
  f.setSolidSystem(makeSolidMechanicsSystem());
}

} // namespace dualmesh
