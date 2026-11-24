// SPDX-License-Identifier: LGPL-2.1-or-later
//
// Solid mechanics module, continuum: linear elasticity in plane stress, plane
// strain, axisymmetric, and three-dimensional settings.  The structural
// members (beams and plates), the reduced theories of the same module, are in
// SolidMechanicsStructures.cpp.
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
#include "dualmesh/base/Deformation.h"
#include "dualmesh/base/Factory.h"
#include "dualmesh/base/FieldExpression.h"
#include "dualmesh/base/Kernel.h"
#include "dualmesh/base/Problem.h"
#include "dualmesh/base/Property.h"
#include "dualmesh/core/ParsedFunction.h"

#include <algorithm>
#include <array>
#include <map>
#include <memory>
#include <set>
#include <type_traits>

namespace dualmesh
{

namespace
{

/// The continuum formulations: the idealisation of the body and the strains it has.
enum class Formulation
{
  PlaneStress,
  PlaneStrain,
  GeneralizedPlaneStrain,
  Axisymmetric,
  AxisymmetricOneD,
  ThreeDimensional
};

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

/// The isotropic small-strain stress @p sigma of the elastic strain @p ee (Voigt order,
/// engineering shears, six values each) for the Lame constants @p lambda and @p mu.  In plane
/// stress sigma_zz = 0, and ee[2] is set to the elastic strain that makes it so; in the other
/// formulations ee[2] is a strain of the displacements and stays as it is.  The stress is
/// accumulated in place: this runs at every integration point, and a temporary value with its
/// derivatives costs as much as the arithmetic.
template <typename Real>
void
isotropicStress(Formulation f, ADReal * ee, const Real & lambda, const Real & mu, ADReal * sigma)
{
  const auto scale = [](ADReal & s, const ADReal & x, const Real & c)
  {
    if constexpr (std::is_same_v<Real, double>)
      s.addScaledBy(x, c);
    else
      s += c * x;
  };
  ADReal trace = ee[0];
  trace += ee[1];
  if (f == Formulation::PlaneStress)
  {
    // lambda* = 2 mu lambda / (lambda + 2 mu) is the plane-stress Lame constant, so that
    // sigma_xx = E / (1 - nu^2) (eps_xx + nu eps_yy), and eps_zz = -lambda / (lambda + 2 mu)
    // (eps_xx + eps_yy).
    const Real plane = 2.0 * mu * lambda / (lambda + 2.0 * mu);
    for (int i = 0; i < 2; ++i)
    {
      sigma[i] = 0.0;
      scale(sigma[i], trace, plane);
      scale(sigma[i], ee[i], 2.0 * mu);
    }
    sigma[2] = 0.0;
    ee[2] = 0.0;
    scale(ee[2], trace, -lambda / (lambda + 2.0 * mu));
  }
  else
  {
    trace += ee[2];
    for (int i = 0; i < 3; ++i)
    {
      sigma[i] = 0.0;
      scale(sigma[i], trace, lambda);
      scale(sigma[i], ee[i], 2.0 * mu);
    }
  }
  for (int i = 3; i < 6; ++i)
  {
    sigma[i] = 0.0;
    scale(sigma[i], ee[i], mu);
  }
}

/// The stress sigma = C eps_e of an anisotropic material, with C the 6 x 6 stiffness in Voigt
/// order (engineering shear strains, tensor shear stresses), row by row.  In plane stress the
/// out-of-plane strains eps_zz, gamma_yz and gamma_xz (ee[2], ee[3], ee[4]) are set to those that
/// make sigma_zz, sigma_yz and sigma_xz zero.
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

/// Solve the dense system A x = b in place by Gaussian elimination with partial pivoting
/// (A is n x n, row by row).  Returns false for a singular matrix.
bool
solveDense(int n, std::vector<double> & A, std::vector<double> & b)
{
  for (int k = 0; k < n; ++k)
  {
    int pivot = k;
    for (int i = k + 1; i < n; ++i)
      if (std::abs(A[i * n + k]) > std::abs(A[pivot * n + k]))
        pivot = i;
    if (A[pivot * n + k] == 0.0)
      return false;
    if (pivot != k)
    {
      for (int j = 0; j < n; ++j)
        std::swap(A[k * n + j], A[pivot * n + j]);
      std::swap(b[k], b[pivot]);
    }
    for (int i = k + 1; i < n; ++i)
    {
      const double factor = A[i * n + k] / A[k * n + k];
      for (int j = k; j < n; ++j)
        A[i * n + j] -= factor * A[k * n + j];
      b[i] -= factor * b[k];
    }
  }
  for (int k = n - 1; k >= 0; --k)
  {
    for (int j = k + 1; j < n; ++j)
      b[k] -= A[k * n + j] * b[j];
    b[k] /= A[k * n + k];
  }
  return true;
}

/// The inverse of an n x n matrix (row by row), by solveDense on each column.
std::vector<double>
inverseDense(int n, const std::vector<double> & A)
{
  std::vector<double> inverse(static_cast<std::size_t>(n * n));
  for (int c = 0; c < n; ++c)
  {
    std::vector<double> M = A, e(static_cast<std::size_t>(n), 0.0);
    e[c] = 1.0;
    if (!solveDense(n, M, e))
      throw InputError("dualmesh: a singular matrix in the return mapping.");
    for (int r = 0; r < n; ++r)
      inverse[r * n + c] = e[r];
  }
  return inverse;
}

/// Voigt ordering used for the "stress" and "strain" properties:
/// 0: xx, 1: yy, 2: zz (hoop in the axisymmetric case), 3: yz, 4: xz, 5: xy.
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

/// The parameter that lists the displacements of an object that acts on each of them.
void
addDisplacementsParam(InputParameters & p, const std::string & what)
{
  p.addRequired("displacements",
                ParameterKind::StringList,
                "The displacement variables in the order of the coordinate axes, e.g., [u, v, w] "
                "in three dimensions. " +
                    what);
  p.setModelVariables({"displacements"});
  p.addOptional("variable",
                ParameterKind::Names,
                std::vector<std::string>{},
                "The displacement this object acts on. Default: every one of 'displacements', "
                "each with an object of its own.");
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
                         "one condition are not available yet.");
      thickness = h;
    }
  return thickness > 0.0 ? thickness : 1.0;
}

/// Divergence of the stress: the equilibrium equation of one displacement
/// component.
class StressDivergence : public Kernel
{
public:
  static InputParameters validParams()
  {
    InputParameters p = Kernel::validParams();
    p.setClassDescription(
        "Divergence of the stress for one displacement component: "
        "F = h (sigma_i0, sigma_i1, sigma_i2), where h is the thickness. The source is zero "
        "except in the axisymmetric case, where the radial equation (component 0) carries "
        "S = sigma_tt / r, taken as zero on the axis. The natural boundary quantity is the "
        "traction component times the thickness.");
    addDisplacementsParam(p,
                          "The equilibrium equation of 'variable' is the one of its place in "
                          "this list.");
    p.addOptional("thickness",
                  ParameterKind::Real,
                  1.0,
                  "Out-of-plane thickness h multiplying the whole equilibrium equation, for plane "
                  "stress and plane strain only. A three-dimensional or axisymmetric problem, "
                  "whose integration measure already carries the factor 2 pi r, refuses any "
                  "other value than 1. The traction and pressure conditions of the same "
                  "displacement read it from here, and mortar_contact takes it as its own "
                  "parameter (the physics passes the same). A point_source is not "
                  "scaled by it, so a concentrated load must already be the total force through "
                  "the thickness.");
    p.setRange("thickness", ParameterRange::Positive);
    p.addOptional("stress_property",
                  ParameterKind::String,
                  std::string("stress"),
                  "Property holding the stress: six components in Voigt order "
                  "(the Cauchy stress of small_strain_stress), or nine components row by row "
                  "(the first Piola-Kirchhoff stress of finite_strain_stress, for equations "
                  "on the undeformed mesh). Default stress.");
    p.addOptional("axisymmetric_hoop_term",
                  ParameterKind::Boolean,
                  true,
                  "Include the sigma_tt / r source in the radial equation (axisymmetric only).");
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
                       "': thickness is the out-of-plane thickness of a plane problem; a "
                       "three-dimensional or axisymmetric problem has none, so leave it at 1.");
    const auto & registry = problem.propertyRegistry();
    _stress = registry.id(_params.getString("stress_property"));
    _full = registry.components(_params.getString("stress_property")) == 9;
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
    // Voigt index of sigma_{component, direction}
    static const int voigt[3][3] = {{0, 5, 4}, {5, 1, 3}, {4, 3, 2}};
    for (int d = 0; d < ctx.dim; ++d)
    {
      // A nine-component stress (the first Piola-Kirchhoff stress of the
      // finite-strain property objects) is stored row by row and is not symmetric.
      F[d] = _full ? ctx.property(_stress, 3 * _component + d)
                   : ctx.property(_stress, voigt[_component][d]);
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

private:
  int _component;
  double _thickness;
  bool _hoop;
  bool _full = false;
  bool _axisymmetric = false;
  int _stress = -1;
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
        "The damage equation of phase-field fracture (AT2): the flux Gc l grad d and the "
        "source (Gc / l + 2 H) d - 2 H, with H the property fracture_driving_energy of the "
        "stress object, so that a crack costs the energy Gc per unit area.");
    p.addRequired("toughness", ParameterKind::Real, "The critical energy release rate Gc, J/m^2.");
    p.setRange("toughness", ParameterRange::Positive);
    p.addRequired("length", ParameterKind::Real, "The length l of the diffuse crack, m.");
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
/// source sigma_zz, whose integral over the section is the axial force it carries; a body_force
/// of F/A on the same variable makes that force F.
class OutOfPlaneStress : public Kernel
{
public:
  static InputParameters validParams()
  {
    InputParameters p = Kernel::validParams();
    p.setClassDescription(
        "The axial equation of generalized plane strain on the uniform strain eps_zz: the "
        "source sigma_zz, integrated over the section, against the axial force (a body_force "
        "of F/A on the same variable; none: a free axial strain, zero force).");
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
    p.setClassDescription("One unknown of 'variable' on the whole mesh: every node a copy of the "
                          "first, with one equation that collects the equations of all.");
    p.addRequired("variable", ParameterKind::String, "The variable.");
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
        "The stiffness-proportional Rayleigh damping b K v of one displacement component: the "
        "flux F = b h (C : d(eps)/dt)_i, with C the isotropic elastic stiffness and the strain "
        "rate of the velocity of the time step. With the second_time_derivative of the "
        "density and a time_derivative of a rho (mass-proportional damping), it makes the "
        "damping C = a M + b K.");
    addDisplacementsParam(p, "The equation of 'variable' is the one of its place in this list.");
    p.addRequired("formulation",
                  ParameterKind::String,
                  "plane_stress, plane_strain, axisymmetric or three_dimensional.");
    p.addRequired("coefficient", ParameterKind::Real, "The damping coefficient b, s.");
    p.setRange("coefficient", ParameterRange::NonNegative);
    p.addOptional("youngs_modulus",
                  ParameterKind::Function,
                  std::string("youngs_modulus"),
                  "Young's modulus, Pa, as small_strain_stress takes it. Default: the property "
                  "youngs_modulus.");
    p.addOptional("poissons_ratio",
                  ParameterKind::Function,
                  std::string("poissons_ratio"),
                  "Poisson's ratio, as small_strain_stress takes it. Default: the property "
                  "poissons_ratio.");
    p.addOptional("thickness",
                  ParameterKind::Real,
                  1.0,
                  "The out-of-plane thickness of a plane problem, as stress_divergence takes it. "
                  "Default 1.");
    p.setRange("thickness", ParameterRange::Positive);
    p.addOptional("stiffness",
                  ParameterKind::RealList,
                  std::vector<double>{},
                  "An anisotropic stiffness, as small_strain_stress takes it (36 values), in place "
                  "of the moduli. Default none: isotropic.");
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
                       "damping yet.");
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
    static const int voigt[3][3] = {{0, 5, 4}, {5, 1, 3}, {4, 3, 2}};
    for (int d = 0; d < ctx.dim; ++d)
      F[d] = _b * _thickness * sigma[voigt[_component][d]];
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
    for (auto & x : e)
      x = 0.0;
    const ADVector3 gu = ctx.rateGradient(_disp[0]);
    const ADVector3 gv = ctx.rateGradient(_disp[1]);
    e[0] = gu[0];
    e[1] = gv[1];
    e[5] = gu[1] + gv[0];
    if (_formulation == Formulation::Axisymmetric)
      e[2] = ctx.x[0] != 0.0 ? ctx.rate(_disp[0]) / ctx.x[0] : gu[0];
    else if (_formulation == Formulation::ThreeDimensional)
    {
      const ADVector3 gw = ctx.rateGradient(_disp[2]);
      e[2] = gw[2];
      e[3] = gv[2] + gw[1];
      e[4] = gu[2] + gw[0];
    }
    if (_anisotropic)
    {
      anisotropicStress(_formulation, e.data(), _stiffness, sigma.data());
      return;
    }
    const ADReal E = _E.value(ctx), nu = _nu.value(ctx);
    const ADReal mu = E / (2.0 * (1.0 + nu));
    const ADReal lambda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
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

/// The isotropic small-strain stress of a solid in every continuum formulation: the elastic
/// strain is the strain of the displacements less the eigenstrains (a thermal strain, for
/// example) and the creep strain, and Young's modulus and Poisson's ratio may be constants,
/// functions, properties or expressions of the fields.
class SmallStrainStress : public Property
{
public:
  static InputParameters validParams()
  {
    InputParameters p = Property::validParams();
    p.setClassDescription(
        "Small-strain stress of an isotropic material: sigma = lambda tr(eps_e) I + 2 mu eps_e of "
        "the elastic strain eps_e = eps - sum of the eigenstrains - creep strain, with Young's "
        "modulus and Poisson's ratio given as constants, functions, properties or expressions, "
        "and optional creep. Formulations: plane_stress (a thin body free to contract through "
        "its thickness, sigma_zz = 0), plane_strain (a long body restrained along its axis), "
        "axisymmetric (a body of revolution on an (r, z) mesh), axisymmetric_1d (a radial slice "
        "of a long body in generalized plane strain, whose uniform axial strain is given by "
        "'axial_strain') and three_dimensional. A thermal strain is an eigenstrain, for example "
        "of thermal_expansion_eigenstrain. Declares 'stress', 'strain', 'elastic_strain' (in "
        "the Voigt order xx, yy, zz, yz, xz, xy with engineering shears, or r, z, theta in the "
        "axisymmetric case), 'volumetric_strain' and 'von_mises_stress', and with creep "
        "'creep_strain' and 'equivalent_creep_strain'.");
    p.addRequired("displacements",
                  ParameterKind::StringList,
                  "The displacement variables: two in a plane or axisymmetric problem (the "
                  "radial and the axial one in the axisymmetric case), three in three "
                  "dimensions and the radial one in axisymmetric_1d.");
    p.addRequired("formulation",
                  ParameterKind::String,
                  "plane_stress, plane_strain, axisymmetric, axisymmetric_1d or "
                  "three_dimensional.");
    p.addOptional("youngs_modulus",
                  ParameterKind::Function,
                  std::string("youngs_modulus"),
                  "Young's modulus, Pa: a constant, a registered function, the name of a property, "
                  "or an expression of the fields, x, y, z, t, the functions and the properties. "
                  "It has no numerical default. Default: the property youngs_modulus, which a "
                  "property object such as constant or parsed declares.");
    p.addOptional(
        "poissons_ratio",
        ParameterKind::Function,
        std::string("poissons_ratio"),
        "Poisson's ratio, with -1 < nu < 1/2: a constant, a registered function, the name "
        "of a property, or an expression. It has no numerical default. Default: the property "
        "poissons_ratio, which a property object such as constant or parsed declares.");
    p.addOptional("eigenstrain_names",
                  ParameterKind::StringList,
                  std::vector<std::string>{},
                  "Names of the six-component eigenstrain properties to subtract. "
                  "Default none.");
    p.addOptional("axial_strain",
                  ParameterKind::Function,
                  0.0,
                  "The uniform axial strain of the axisymmetric_1d formulation. Default 0.");
    p.addOptional("creep_model",
                  ParameterKind::String,
                  std::string("none"),
                  "Creep of the material: none or parsed (the rate given by 'creep_rate'). "
                  "The creep strain is integrated by the backward Euler method with a radial "
                  "return on the von Mises stress, and kept as a history at every integration "
                  "point. Default none.");
    p.addOptional("creep_rate",
                  ParameterKind::String,
                  std::string(""),
                  "With creep_model = parsed: the equivalent creep strain rate, 1/s, as an "
                  "expression of von_mises_stress (Pa), temperature (K), grain_size (m), the "
                  "time t (s, for primary creep by time hardening), equivalent_creep_strain "
                  "(for strain hardening, taken at the end of the step) and the constants of "
                  "'creep_constants'. It is "
                  "differentiated automatically, so the return mapping and the Jacobian are "
                  "exact. It must not be negative. Default empty.");
    p.addOptional("creep_constants",
                  ParameterKind::RealMap,
                  RealMap{},
                  "Named constants that 'creep_rate' uses: name -> value. Default none.");
    p.addOptional("temperature",
                  ParameterKind::String,
                  std::string(""),
                  "The temperature variable (kelvin), required with creep.");
    p.addOptional("grain_size",
                  ParameterKind::Real,
                  std::monostate{},
                  "Grain size (the grain diameter) of the material, m. A mean linear intercept L "
                  "corresponds to the grain size 1.56 L. Required when the parsed creep rate "
                  "uses grain_size.");
    p.addOptional("creep_rate_factor",
                  ParameterKind::Real,
                  1.0,
                  "Factor on the creep rate, for sensitivity and uncertainty studies. Default 1.");
    p.addOptional("out_of_plane_strain",
                  ParameterKind::String,
                  std::string(""),
                  "generalized_plane_strain: the variable of the uniform strain eps_zz along the "
                  "axis, one unknown of the whole body (a uniform_field), whose equation "
                  "(out_of_plane_stress) is the axial force it carries. Required there.");
    p.addOptional("viscous_shear_moduli",
                  ParameterKind::RealList,
                  std::vector<double>{},
                  "Viscoelasticity (generalized Maxwell): the shear moduli G_i, Pa, of the "
                  "deviatoric branches, each relaxing as exp(-t/tau_i), beside the long-term "
                  "moduli. Default none.");
    p.addOptional("viscous_shear_times",
                  ParameterKind::RealList,
                  std::vector<double>{},
                  "The relaxation times tau_i, s, of the shear branches.");
    p.addOptional("viscous_bulk_moduli",
                  ParameterKind::RealList,
                  std::vector<double>{},
                  "The bulk moduli K_i, Pa, of the volumetric branches. Default none.");
    p.addOptional("viscous_bulk_times",
                  ParameterKind::RealList,
                  std::vector<double>{},
                  "The relaxation times, s, of the bulk branches.");
    p.addOptional("damage",
                  ParameterKind::String,
                  std::string(""),
                  "Phase-field fracture (AT2): the damage variable d, which degrades the "
                  "tensile part of the elastic energy by (1 - d)^2 (the split of Miehe by the "
                  "principal strains). Declares fracture_driving_energy, the largest tensile "
                  "energy the point has had (the history of the crack, which does not heal). "
                  "Plane strain, generalized plane strain and axisymmetric. Default none.");
    p.addOptional("yield_stress",
                  ParameterKind::String,
                  std::string(""),
                  "J2 plasticity: the yield stress, Pa, as an expression of "
                  "equivalent_plastic_strain, temperature (K) and the constants of "
                  "'plasticity_constants', which gives the isotropic hardening (a number is "
                  "perfect plasticity). The plastic strain is integrated by the backward Euler "
                  "method with a radial return, with creep in the same return. Default empty: "
                  "no plasticity.");
    p.addOptional("plasticity_constants",
                  ParameterKind::RealMap,
                  RealMap{},
                  "Named constants that 'yield_stress' uses: name -> value. Default none.");
    p.addOptional("kinematic_hardening_moduli",
                  ParameterKind::RealList,
                  std::vector<double>{},
                  "The moduli C_k, Pa, of the back stresses of kinematic hardening "
                  "(Armstrong-Frederick, or Chaboche with several): d(alpha_k) = (2/3) C_k "
                  "d(eps_p) - gamma_k alpha_k dp. Default none.");
    p.addOptional("dynamic_recovery",
                  ParameterKind::RealList,
                  std::vector<double>{},
                  "The dynamic recovery gamma_k of each back stress (0: Prager's linear "
                  "kinematic hardening). Default 0 for each.");
    p.addOptional("stiffness",
                  ParameterKind::RealList,
                  std::vector<double>{},
                  "An anisotropic material: the 6 x 6 stiffness C, Pa, in the global axes and "
                  "the Voigt order xx, yy, zz, yz, xz, xy (engineering shear strains), row by "
                  "row (36 values), in place of youngs_modulus and poissons_ratio. It must be "
                  "symmetric and positive definite. Default none: isotropic.");
    p.addOptional("hill_coefficients",
                  ParameterKind::RealList,
                  std::vector<double>{},
                  "Hill's anisotropic equivalent stress sqrt(sigma . P sigma) for creep and "
                  "plasticity: the 6 x 6 matrix P in the global axes and the Voigt order "
                  "(tensor shear stresses), row by row (36 values). With it the creep rate and "
                  "the yield stress read this equivalent stress as von_mises_stress, and the "
                  "flow is normal to it. Default none: von Mises.");
    return p;
  }
  /// The history of a point: the creep strain and its equivalent, the plastic strain and its
  /// equivalent, and the back stresses (InelasticIncrements).
  int stateSize() const override { return inelasticSize() + viscousSize() + (_cracking ? 1 : 0); }
  int viscousSize() const
  {
    return _viscous.empty() ? 0
                            : 6 + 6 * static_cast<int>(_shear_branches) +
                                  static_cast<int>(_viscous.size() - _shear_branches);
  }
  /// The size of the history of creep and plasticity, the first part of the state.
  int inelasticSize() const
  {
    return _creep == Creep::None && !_plastic ? 0 : 14 + 6 * static_cast<int>(_hardening.size());
  }
  explicit SmallStrainStress(const InputParameters & p)
      : Property(p), _formulation(formulationFromName(name(), p.getString("formulation")))
  {
    // A constant is checked now; a property or an expression is evaluated where it is used.
    if (const auto * E = std::get_if<double>(&p.getRaw("youngs_modulus")); E && !(*E > 0.0))
      throw InputError("'" + name() + "': youngs_modulus must be positive.");
    if (const auto * nu = std::get_if<double>(&p.getRaw("poissons_ratio"));
        nu && !(*nu > -1.0 && *nu < 0.5))
      throw InputError("'" + name() + "': poissons_ratio must lie in (-1, 0.5).");
    const auto c = p.getString("creep_model");
    if (c == "none")
      _creep = Creep::None;
    else if (c == "parsed")
      _creep = Creep::Parsed;
    else
      throw InputError("'" + name() + "': unknown creep_model '" + c + "' (none, parsed).");
    if (_creep == Creep::Parsed)
    {
      const auto text = p.getString("creep_rate");
      if (text.empty())
        throw InputError("'" + name() + "': creep_model = parsed needs 'creep_rate'.");
      std::vector<std::string> names = {
          "von_mises_stress", "temperature", "grain_size", "t", "equivalent_creep_strain"};
      _creep_constants.clear();
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
    // The grain size is required only when the creep rate reads it (the
    // argument 2 of the parsed creep rate).
    if (_creep == Creep::Parsed && _creep_expression->uses(2) && !p.isSet("grain_size"))
      throw InputError("'" + name() +
                       "': a creep_rate that uses grain_size needs 'grain_size' (m), the grain "
                       "size of the material.");
    if (p.isSet("grain_size") && !(p.getReal("grain_size") > 0.0))
      throw InputError("'" + name() + "': grain_size must be positive.");
    _grain = p.isSet("grain_size") ? p.getReal("grain_size") : 0.0;
    _creep_factor = p.getReal("creep_rate_factor");
    if (!(_creep_factor >= 0.0))
      throw InputError("'" + name() + "': creep_rate_factor must not be negative.");
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
                       "': kinematic hardening is part of plasticity; give the "
                       "yield_stress.");
    for (std::size_t k = 0; k < moduli.size(); ++k)
    {
      if (!(moduli[k] > 0.0 && recovery[k] >= 0.0))
        throw InputError("'" + name() +
                         "': the kinematic hardening moduli are positive and the "
                         "dynamic recovery not negative.");
      _hardening.push_back({moduli[k], recovery[k]});
    }
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
    if (!_viscous.empty() &&
        (_creep != Creep::None || _plastic || _formulation == Formulation::PlaneStress))
      throw InputError("'" + name() +
                       "': viscoelasticity with creep or plasticity, or in plane "
                       "stress, is not available yet.");
    _cracking = !p.getString("damage").empty();
    if (_cracking && (_formulation == Formulation::PlaneStress ||
                      _formulation == Formulation::ThreeDimensional ||
                      _formulation == Formulation::AxisymmetricOneD || _creep != Creep::None ||
                      _plastic || !_viscous.empty() || !p.getRealList("stiffness").empty()))
      throw InputError("'" + name() +
                       "': phase-field fracture is available for an elastic "
                       "isotropic material in plane strain, generalized plane "
                       "strain and axisymmetric problems.");
    const auto stiffness = p.getRealList("stiffness");
    if (!stiffness.empty())
    {
      if (stiffness.size() != 36)
        throw InputError("'" + name() +
                         "': stiffness takes 36 values, the 6 x 6 matrix row by row.");
      if (p.isSetByUser("youngs_modulus") || p.isSetByUser("poissons_ratio"))
        throw InputError("'" + name() +
                         "': give the stiffness or youngs_modulus and "
                         "poissons_ratio, not both.");
      std::copy(stiffness.begin(), stiffness.end(), _stiffness.begin());
      checkSymmetricPositive(_stiffness, "stiffness");
      _anisotropic = true;
    }
    // Von Mises as the default equivalent stress: q^2 = sigma . P sigma.
    for (int i = 0; i < 3; ++i)
    {
      for (int j = 0; j < 3; ++j)
        _hill[6 * i + j] = i == j ? 1.0 : -0.5;
      _hill[6 * (i + 3) + i + 3] = 3.0;
    }
    const auto hill = p.getRealList("hill_coefficients");
    if (!hill.empty())
    {
      if (hill.size() != 36)
        throw InputError("'" + name() +
                         "': hill_coefficients takes 36 values, the 6 x 6 matrix "
                         "row by row.");
      if (_creep == Creep::None && !_plastic)
        throw InputError("'" + name() +
                         "': hill_coefficients is the equivalent stress of creep "
                         "and plasticity; give one of them.");
      if (!_hardening.empty())
        throw InputError("'" + name() +
                         "': Hill plasticity with kinematic hardening is not "
                         "available yet.");
      std::copy(hill.begin(), hill.end(), _hill.begin());
      _hill_given = true;
    }
    if (_anisotropic && !_hardening.empty())
      throw InputError("'" + name() +
                       "': kinematic hardening with an anisotropic stiffness is "
                       "not available yet.");
    if (_creep != Creep::None && !_hardening.empty())
      throw InputError("'" + name() +
                       "': creep with kinematic hardening is not available yet "
                       "(creep acts on the stress, plasticity on the stress "
                       "less the back stress); use isotropic hardening with "
                       "creep.");
  }
  void initialSetup(Problem & problem) override
  {
    Property::initialSetup(problem);
    const int dim = problem.mesh().dimension();
    if (problem.mesh().spaceDimension() > dim)
      throw InputError("'" + name() +
                       "': the mesh is a surface (or a curve) in space, and the continuum "
                       "formulations need a mesh that fills its space.  Surfaces are shells.");
    // The temperature variable, which a creep rate or a yield stress that reads it needs.
    const bool creep_reads = _creep != Creep::None && _creep_expression->uses(1);
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
      _mu0 = *E / (2.0 * (1.0 + *nu));
      _lambda0 = *E * *nu / ((1.0 + *nu) * (1.0 - 2.0 * *nu));
    }
    _eigen.clear();
    for (const auto & n : _params.getStringList("eigenstrain_names"))
      _eigen.push_back(reg.id(n));
  }
  void declareProperties(PropertyRegistry & r) override
  {
    _stress = r.declare("stress", 6);
    _strain = r.declare("strain", 6);
    _elastic = r.declare("elastic_strain", 6);
    _volumetric = r.declare("volumetric_strain", 1);
    _vm = r.declare("von_mises_stress", 1);
    _F_prop = r.declare("deformation_gradient", 9);
    if (_creep != Creep::None)
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

  /// The yield stress at the equivalent plastic strain p and the temperature T.
  template <typename T> T yieldStress(const T & p, const T & temperature) const
  {
    std::array<T, 32> args;
    args[0] = p;
    args[1] = temperature;
    for (std::size_t k = 0; k < _plasticity_constants.size(); ++k)
      args[2 + k] = T(_plasticity_constants[k]);
    return _yield_expression->value(args.data());
  }

  /// The von Mises value sqrt(3/2 s:s) of a deviatoric tensor in Voigt order (tensor shears).
  template <typename T> static T vonMisesOf(const std::array<T, 6> & s)
  {
    return sqrt(1.5 * (s[0] * s[0] + s[1] * s[1] + s[2] * s[2]) +
                3.0 * (s[3] * s[3] + s[4] * s[4] + s[5] * s[5]));
  }

  /// The increments of one step at a point.
  struct InelasticIncrements
  {
    std::array<double, 6> creep{0, 0, 0, 0, 0, 0}, plastic{0, 0, 0, 0, 0, 0};
    double creep_equivalent = 0.0, plastic_equivalent = 0.0;
    /// The back stresses at the end of the step, six values each.
    std::vector<double> back;
  };

  /// The inelastic return of one step from the trial stress @p sigma and the history @p old
  /// (the layout of stateSize): creep alone, plasticity alone, or both, in one return.  The
  /// creep acts in a step of time only; the plasticity in a steady solve as well.
  void inelasticReturn(QpContext & ctx,
                       const ADReal & mu,
                       const ADReal & T,
                       double creep_old,
                       double plastic_old,
                       const std::vector<ADReal> & back_old,
                       std::array<ADReal, 6> & sigma,
                       InelasticIncrements & inc) const
  {
    inc.back.resize(back_old.size());
    for (std::size_t j = 0; j < back_old.size(); ++j)
      inc.back[j] = back_old[j].value();
    const bool creep = _creep != Creep::None && ctx.dt > 0;
    if (_anisotropic || _hill_given)
    {
      generalReturn(ctx, mu, T, creep_old, plastic_old, sigma, inc);
      return;
    }
    if (!_plastic)
    {
      if (creep)
        returnMap(ctx, mu, T, creep_old, sigma, inc.creep, inc.creep_equivalent);
      return;
    }
    if (!creep)
    {
      plasticReturn(mu, T, plastic_old, back_old, sigma, inc);
      return;
    }
    creepPlasticReturn(ctx, mu, T, creep_old, plastic_old, sigma, inc);
  }
  /// The same, with the history of the small-strain layout (stateSize).
  void inelasticReturn(QpContext & ctx,
                       const ADReal & mu,
                       const ADReal & T,
                       const double * old,
                       std::array<ADReal, 6> & sigma,
                       InelasticIncrements & inc) const
  {
    const std::vector<ADReal> back(old + 14, old + 14 + 6 * _hardening.size());
    inelasticReturn(ctx, mu, T, old[6], old[13], back, sigma, inc);
  }

  /// The equivalent stress sqrt(sigma . P sigma) of Hill (von Mises for the isotropic P).
  template <typename T>
  T hillStress(const std::array<T, 6> & sigma, std::array<T, 6> & normal) const
  {
    std::array<T, 6> Ps;
    T square(0.0);
    for (int i = 0; i < 6; ++i)
    {
      Ps[i] = T(0.0);
      for (int j = 0; j < 6; ++j)
        Ps[i] += _hill[6 * i + j] * sigma[j];
      square += sigma[i] * Ps[i];
    }
    using std::sqrt;
    const T q = sqrt(square);
    for (int i = 0; i < 6; ++i)
      normal[i] = valueOf(q) > 0.0 ? Ps[i] / q : T(0.0);
    return q;
  }

  /// The closest-point return of creep and plasticity for an anisotropic stiffness or Hill's
  /// equivalent stress q(sigma): the stress, the creep increment dc and the plastic increment dp
  /// solve S sigma - eps_trial + (dc + dp) dq/dsigma = 0, dc = dt rate(q, T, c + dc) and,
  /// where the point yields, q = sigma_y(p + dp), by Newton's method on the eight unknowns at
  /// the point; their derivatives follow from the implicit function theorem.
  void generalReturn(QpContext & ctx,
                     const ADReal & mu,
                     const ADReal & T,
                     double c_old,
                     double p_old,
                     std::array<ADReal, 6> & sigma,
                     InelasticIncrements & inc) const
  {
    std::array<double, 36> C = _stiffness;
    if (!_anisotropic)
    {
      if (!_constant_moduli)
        throw InputError("'" + name() + "': Hill creep and plasticity take constant moduli.");
      C.fill(0.0);
      for (int i = 0; i < 3; ++i)
      {
        for (int j = 0; j < 3; ++j)
          C[6 * i + j] = _lambda0;
        C[6 * i + i] += 2.0 * _mu0;
        C[6 * (i + 3) + i + 3] = _mu0;
      }
    }
    const std::vector<double> S = inverseDense(6, std::vector<double>(C.begin(), C.end()));
    const bool creep = _creep != Creep::None && ctx.dt > 0;
    const double Tv = T.value();
    std::array<double, 6> trial;
    for (int i = 0; i < 6; ++i)
      trial[i] = sigma[i].value();
    std::array<double, 6> unused;
    const double q0 = hillStress(trial, unused);
    bool yielding = _plastic && q0 > yieldStress(p_old, Tv);
    if (!creep && !yielding)
      return;
    // The residuals at x = (sigma, dc, dp) and the elastic trial strain S sigma_trial.
    const auto residuals = [&](const std::array<ADReal, 8> & x,
                               const std::array<ADReal, 6> & strain,
                               const ADReal & temperature,
                               bool plastic,
                               std::array<ADReal, 8> & r)
    {
      std::array<ADReal, 6> s, normal;
      for (int i = 0; i < 6; ++i)
        s[i] = x[i];
      const ADReal q = hillStress(s, normal);
      for (int i = 0; i < 6; ++i)
      {
        r[i] = -strain[i] + (x[6] + x[7]) * normal[i];
        for (int j = 0; j < 6; ++j)
          r[i] += S[6 * i + j] * x[j];
      }
      r[6] = creep ? x[6] - ctx.dt * creepRate(q, temperature, ADReal(c_old) + x[6], ctx) : x[6];
      r[7] = plastic ? q - yieldStress(ADReal(p_old) + x[7], temperature) : x[7];
    };
    std::array<ADReal, 6> trial_strain;
    for (int i = 0; i < 6; ++i)
    {
      double e = 0.0;
      for (int j = 0; j < 6; ++j)
        e += S[6 * i + j] * trial[j];
      trial_strain[i] = ADReal(e);
    }
    std::array<double, 8> x{};
    for (int i = 0; i < 6; ++i)
      x[i] = trial[i];
    std::vector<double> J(64);
    const double size = std::abs(q0) + 1e-300;
    // Newton's method: first with the plastic increment held at zero (creep alone), then, if
    // the stress it leaves is beyond the yield stress, with both.
    const auto solve = [&](bool plastic)
    {
      for (int it = 0; it < 100; ++it)
      {
        std::array<ADReal, 8> X, r;
        for (int i = 0; i < 8; ++i)
          X[i] = ADReal::independent(x[i], i, 8);
        residuals(X, trial_strain, ADReal(Tv), plastic, r);
        std::vector<double> b(8);
        for (int i = 0; i < 8; ++i)
        {
          b[i] = r[i].value();
          for (int j = 0; j < 8; ++j)
            J[8 * i + j] = r[i].derivative(j);
        }
        std::vector<double> A = J;
        if (!solveDense(8, A, b))
          throw InadmissibleStateError("dualmesh: the return mapping met a singular Jacobian.");
        // A step that would make an increment negative is shortened.
        double t = 1.0;
        while ((x[6] - t * b[6] < 0.0 || x[7] - t * b[7] < 0.0) && t > 1e-8)
          t *= 0.5;
        double change = 0.0;
        for (int i = 0; i < 8; ++i)
        {
          x[i] -= t * b[i];
          change = std::max(change, std::abs(t * b[i]) * (i < 6 ? 1.0 / size : 1.0));
        }
        if (change <= 1e-14)
          break;
      }
    };
    if (creep)
      solve(false);
    if (_plastic)
    {
      std::array<double, 6> s;
      for (int i = 0; i < 6; ++i)
        s[i] = x[i];
      yielding = hillStress(s, unused) > yieldStress(p_old, Tv);
      if (yielding)
        solve(true);
    }
    // The derivatives of the unknowns: x(U) = x - J^-1 r(x, U), with the residual of the trial
    // strain of the unknowns at the solved x.
    {
      std::array<ADReal, 8> X, r;
      for (int i = 0; i < 8; ++i)
        X[i] = ADReal::independent(x[i], i, 8);
      residuals(X, trial_strain, ADReal(Tv), yielding, r);
      for (int i = 0; i < 8; ++i)
        for (int j = 0; j < 8; ++j)
          J[8 * i + j] = r[i].derivative(j);
    }
    const std::vector<double> Jinv = inverseDense(8, J);
    std::array<ADReal, 6> strain;
    for (int i = 0; i < 6; ++i)
    {
      strain[i] = ADReal(0.0);
      for (int j = 0; j < 6; ++j)
        strain[i] += S[6 * i + j] * sigma[j];
    }
    std::array<ADReal, 8> X, r;
    for (int i = 0; i < 8; ++i)
      X[i] = ADReal(x[i]);
    residuals(X, strain, T, yielding, r);
    std::array<ADReal, 8> result;
    for (int i = 0; i < 8; ++i)
    {
      result[i] = ADReal(x[i]);
      for (int j = 0; j < 8; ++j)
        result[i] -= Jinv[8 * i + j] * (r[j] - ADReal(r[j].value()));
    }
    std::array<double, 6> s, normal;
    for (int i = 0; i < 6; ++i)
    {
      s[i] = x[i];
      sigma[i] = result[i];
    }
    hillStress(s, normal);
    for (int i = 0; i < 6; ++i)
    {
      inc.creep[i] = x[6] * normal[i];
      inc.plastic[i] = x[7] * normal[i];
    }
    inc.creep_equivalent = x[6];
    inc.plastic_equivalent = x[7];
    (void) mu;
  }

  /// J2 plasticity with isotropic hardening and the back stresses alpha_k of Armstrong and
  /// Frederick (Chaboche): with a_k = 1 / (1 + gamma_k dp), the relative stress at the end of
  /// the step is parallel to xi~ = s_trial - sum a_k alpha_k^n, and the increment dp of the
  /// equivalent plastic strain solves q(xi~) - (3G + sum a_k C_k) dp - sigma_y(p_n + dp) = 0.
  void plasticReturn(const ADReal & mu,
                     const ADReal & T,
                     double p_old,
                     const std::vector<ADReal> & back_old,
                     std::array<ADReal, 6> & sigma,
                     InelasticIncrements & inc) const
  {
    const std::size_t K = _hardening.size();
    const ADReal mean = (sigma[0] + sigma[1] + sigma[2]) / 3.0;
    std::array<ADReal, 6> dev = sigma;
    for (int i = 0; i < 3; ++i)
      dev[i] = sigma[i] - mean;
    const double G = mu.value(), Tv = T.value();
    // The relative stress of the trial, with derivatives in dp only.
    const auto relative = [&](const ADReal & dp, std::array<ADReal, 6> & xi, ADReal & H)
    {
      H = 3.0 * G;
      for (int i = 0; i < 6; ++i)
        xi[i] = ADReal(dev[i].value());
      for (std::size_t k = 0; k < K; ++k)
      {
        const ADReal a = 1.0 / (1.0 + _hardening[k].recovery * dp);
        H += _hardening[k].modulus * a;
        for (int i = 0; i < 6; ++i)
          xi[i] -= a * back_old[6 * k + i].value();
      }
    };
    const auto f = [&](double dp, double & slope)
    {
      const ADReal x = ADReal::independent(dp, 0, 1);
      std::array<ADReal, 6> xi;
      ADReal H;
      relative(x, xi, H);
      const ADReal r = vonMisesOf(xi) - H * x - yieldStress(ADReal(p_old) + x, ADReal(Tv));
      slope = r.derivative(0);
      return r.value();
    };
    double slope = 1.0;
    const double f0 = f(0.0, slope);
    if (!(f0 > 0.0))
      return;
    // The root: f(0) > 0, and f falls without bound as dp grows.
    double lo = 0.0, hi = f0 / (3.0 * G);
    while (f(hi, slope) > 0.0)
    {
      lo = hi;
      hi *= 2.0;
    }
    double dp = 0.5 * (lo + hi);
    for (int it = 0; it < 100; ++it)
    {
      const double fv = f(dp, slope);
      if (fv > 0)
        lo = dp;
      else
        hi = dp;
      double next = dp - fv / slope;
      if (!(next > lo && next < hi))
        next = 0.5 * (lo + hi);
      const bool done = std::abs(next - dp) <= 1e-14 * std::abs(next) || hi - lo <= 1e-16 * hi;
      dp = next;
      if (done)
        break;
    }
    // The derivatives of dp by the implicit function theorem, from the residual with the
    // derivatives of the unknowns at the solved dp.
    f(dp, slope);
    std::array<ADReal, 6> xi = dev;
    ADReal H = 3.0 * mu;
    for (std::size_t k = 0; k < K; ++k)
    {
      const double a = 1.0 / (1.0 + _hardening[k].recovery * dp);
      H += _hardening[k].modulus * a;
      for (int i = 0; i < 6; ++i)
        xi[i] -= a * back_old[6 * k + i];
    }
    const ADReal residual = vonMisesOf(xi) - H * dp - yieldStress(ADReal(p_old + dp), T);
    const ADReal DP = dp - residual / slope;
    // The relative stress and its direction with the derivatives of DP.
    xi = dev;
    std::vector<ADReal> a(K);
    for (std::size_t k = 0; k < K; ++k)
    {
      a[k] = 1.0 / (1.0 + _hardening[k].recovery * DP);
      for (int i = 0; i < 6; ++i)
        xi[i] -= a[k] * back_old[6 * k + i];
    }
    const ADReal q = vonMisesOf(xi);
    for (int i = 0; i < 6; ++i)
    {
      // Normal components: (3/2) dp xi_i / q; shear (engineering): 3 dp xi_i / q.
      const ADReal increment = (i < 3 ? 1.5 : 3.0) * DP * xi[i] / q;
      sigma[i] -= (i < 3 ? 2.0 : 1.0) * mu * increment;
      inc.plastic[i] = increment.value();
    }
    inc.plastic_equivalent = DP.value();
    for (std::size_t k = 0; k < K; ++k)
      for (int i = 0; i < 6; ++i)
        inc.back[6 * k + i] =
            a[k].value() *
            (back_old[6 * k + i].value() + _hardening[k].modulus * dp * xi[i].value() / q.value());
  }

  /// Creep and isotropic plasticity in one return: both flow along the deviator of the trial
  /// stress, whose von Mises value falls to q - 3G (dc + dp), with dc = dt rate(q_e, T, c + dc)
  /// and, where the point yields, q_e = sigma_y(p + dp).
  void creepPlasticReturn(QpContext & ctx,
                          const ADReal & mu,
                          const ADReal & T,
                          double c_old,
                          double p_old,
                          std::array<ADReal, 6> & sigma,
                          InelasticIncrements & inc) const
  {
    // Creep alone first: where the stress it leaves is within the yield surface, that is all.
    std::array<ADReal, 6> creep_only = sigma;
    InelasticIncrements alone;
    returnMap(ctx, mu, T, c_old, creep_only, alone.creep, alone.creep_equivalent);
    const ADReal mean = (sigma[0] + sigma[1] + sigma[2]) / 3.0;
    std::array<ADReal, 6> dev = sigma;
    for (int i = 0; i < 3; ++i)
      dev[i] = sigma[i] - mean;
    const ADReal q = vonMisesOf(dev);
    const double qv = q.value(), G = mu.value(), Tv = T.value();
    if (qv - 3.0 * G * alone.creep_equivalent <= yieldStress(p_old, Tv))
    {
      sigma = creep_only;
      inc.creep = alone.creep;
      inc.creep_equivalent = alone.creep_equivalent;
      return;
    }
    // The residuals of (dc, dp) with derivatives in both, or in the unknowns at a solution.
    const auto residuals = [&](const ADReal & dc,
                               const ADReal & dp,
                               const ADReal & qq,
                               const ADReal & GG,
                               const ADReal & TT,
                               ADReal & r1,
                               ADReal & r2)
    {
      const ADReal effective = qq - 3.0 * GG * (dc + dp);
      r1 = dc - ctx.dt * creepRate(effective, TT, ADReal(c_old) + dc, ctx);
      r2 = effective - yieldStress(ADReal(p_old) + dp, TT);
    };
    double dc = alone.creep_equivalent, dp = 0.0;
    double J[2][2] = {{1, 0}, {0, 1}};
    for (int it = 0; it < 100; ++it)
    {
      ADReal r1, r2;
      residuals(ADReal::independent(dc, 0, 2),
                ADReal::independent(dp, 1, 2),
                ADReal(qv),
                ADReal(G),
                ADReal(Tv),
                r1,
                r2);
      J[0][0] = r1.derivative(0);
      J[0][1] = r1.derivative(1);
      J[1][0] = r2.derivative(0);
      J[1][1] = r2.derivative(1);
      const double det = J[0][0] * J[1][1] - J[0][1] * J[1][0];
      const double sc = (J[1][1] * r1.value() - J[0][1] * r2.value()) / det;
      const double sp = (J[0][0] * r2.value() - J[1][0] * r1.value()) / det;
      // A step that would make an increment negative or reverse the stress is shortened.
      double t = 1.0;
      while (
          (dc - t * sc < 0.0 || dp - t * sp < 0.0 || 3.0 * G * (dc - t * sc + dp - t * sp) > qv) &&
          t > 1e-6)
        t *= 0.5;
      dc -= t * sc;
      dp -= t * sp;
      if (std::abs(sc) + std::abs(sp) <= 1e-14 * (dc + dp))
        break;
    }
    // The derivatives of (dc, dp) by the implicit function theorem.
    ADReal r1, r2;
    residuals(ADReal(dc), ADReal(dp), q, mu, T, r1, r2);
    const double det = J[0][0] * J[1][1] - J[0][1] * J[1][0];
    const ADReal DC = dc - (J[1][1] * r1 - J[0][1] * r2) / det;
    const ADReal DP = dp - (J[0][0] * r2 - J[1][0] * r1) / det;
    for (int i = 0; i < 6; ++i)
    {
      const double factor = i < 3 ? 1.5 : 3.0;
      const ADReal creep = factor * DC * dev[i] / q;
      const ADReal plastic = factor * DP * dev[i] / q;
      sigma[i] -= (i < 3 ? 2.0 : 1.0) * mu * (creep + plastic);
      inc.creep[i] = creep.value();
      inc.plastic[i] = plastic.value();
    }
    inc.creep_equivalent = DC.value();
    inc.plastic_equivalent = DP.value();
  }

  /// Creep rate at the von Mises stress sigma (Pa), the temperature T and the equivalent creep
  /// strain p, at the time of the context (the end of the step).
  template <typename T>
  T creepRate(const T & sigma, const T & temperature, const T & p, const QpContext & ctx) const
  {
    // The von Mises stress is not negative; round-off in the return mapping
    // must not hand a negative one to a fractional power.
    if (valueOf(sigma) <= 0.0)
      return T(0.0);
    return _creep_factor * baseCreepRate(sigma, temperature, p, ctx);
  }
  /// Creep rate of the chosen model, before creep_rate_factor.
  template <typename T>
  T baseCreepRate(const T & sigma, const T & temperature, const T & p, const QpContext & ctx) const
  {
    if (_creep != Creep::Parsed)
      return T(0.0);
    std::array<T, 32> args;
    args[0] = sigma;
    args[1] = temperature;
    args[2] = T(_grain);
    args[3] = T(ctx.time);
    args[4] = p;
    for (std::size_t k = 0; k < _creep_constants.size(); ++k)
      args[5 + k] = T(_creep_constants[k]);
    return _creep_expression->value(args.data());
  }
  static double valueOf(double x) { return x; }
  static double valueOf(const ADReal & x) { return x.value(); }
  /// The backward Euler creep update by a radial return: from the trial
  /// stress @p sigma (Voigt, engineering shear order), the creep strain
  /// increment dp (3/2) s / q along the deviator s, with dp = dt rate(q - 3 G
  /// dp, T, p_old + dp).  Updates @p sigma and fills the increments.
  void returnMap(QpContext & ctx,
                 const ADReal & mu,
                 const ADReal & T,
                 double p_old,
                 std::array<ADReal, 6> & sigma,
                 std::array<double, 6> & creep_increment,
                 double & equivalent_increment) const
  {
    // Radial return: the creep strain increment is dp (3/2) s / q along the
    // deviator s of the trial stress, whose von Mises value q falls to
    // q - 3 G dp, and dp = dt * rate(q - 3 G dp, T) (backward Euler).
    const ADReal mean = (sigma[0] + sigma[1] + sigma[2]) / 3.0;
    std::array<ADReal, 6> dev = sigma;
    for (int i = 0; i < 3; ++i)
      dev[i] = sigma[i] - mean;
    const ADReal q = sqrt(1.5 * (dev[0] * dev[0] + dev[1] * dev[1] + dev[2] * dev[2]) +
                          3.0 * (dev[3] * dev[3] + dev[4] * dev[4] + dev[5] * dev[5]));
    const double qv = q.value(), G = mu.value();
    const double Tv = T.value();
    // The creep increment is at most q / 3G (the stress cannot reverse), so below 1e-12 of that
    // there is no creep to compute, in any units.
    if (qv > 1e-12 * 3.0 * G)
    {
      // g(dp) = dp - dt rate(q - 3 G dp, T, p_old + dp) and its derivative in dp, through a pass
      // whose only derivative is dp.
      const auto g = [&](double dp, double & slope)
      {
        const ADReal x = ADReal::independent(dp, 0, 1);
        const ADReal r = x - ctx.dt * creepRate(qv - 3.0 * G * x, ADReal(Tv), p_old + x, ctx);
        slope = r.derivative(0);
        return r.value();
      };
      // The root lies in [0, q / 3G]: g(0) <= 0 and g(q / 3G) >= 0.
      double lo = 0.0, hi = qv / (3.0 * G), slope = 1.0;
      double dp = std::min(ctx.dt * creepRate(qv, Tv, p_old, ctx), hi);
      for (int it = 0; it < 60; ++it)
      {
        const double gv = g(dp, slope);
        if (gv > 0)
          hi = dp;
        else
          lo = dp;
        double next_dp = dp - gv / slope;
        if (!(next_dp > lo && next_dp < hi))
          next_dp = 0.5 * (lo + hi);
        if (std::abs(next_dp - dp) <= 1e-14 * std::max(1e-30, std::abs(next_dp)) ||
            hi - lo < 1e-16 * std::max(1.0, hi))
        {
          dp = next_dp;
          break;
        }
        dp = next_dp;
      }
      // The derivatives of dp follow from the implicit function theorem:
      // g(dp, q, G, T) = 0, so d(dp) = -(dg/dq dq + ...) / (dg/d(dp)).
      g(dp, slope);
      const ADReal residual =
          dp - ctx.dt * creepRate(q - 3.0 * mu * dp, T, ADReal(p_old + dp), ctx);
      const ADReal DP = dp - residual / slope;
      for (int i = 0; i < 6; ++i)
      {
        // Normal components: (3/2) dp s_i / q; shear (engineering): 3 dp s_i / q.
        const ADReal inc = (i < 3 ? 1.5 : 3.0) * DP * dev[i] / q;
        sigma[i] -= (i < 3 ? 2.0 : 1.0) * mu * inc;
        creep_increment[i] = inc.value();
      }
      equivalent_increment = DP.value();
    }
  }

  /// Plane stress with creep or plasticity: the elastic zz strain z at which the stress after the return
  /// mapping has sigma_zz = 0.  The return mapping works on the full stress, so z is solved
  /// at the point by Newton's method, with dsigma_zz/dz from a pass whose only derivative is
  /// z; the derivatives of z with respect to the unknowns follow from the implicit function
  /// theorem, so the tangent stays exact.  On return @p ee holds z and @p sigma the trial
  /// stress of the full elastic strain, which updateCreep then returns.
  void planeStressInelasticStrain(QpContext & ctx,
                                  const ADReal & lambda,
                                  const ADReal & mu,
                                  const double * old,
                                  ADReal * ee,
                                  ADReal * sigma) const
  {
    const double l = lambda.value(), G = mu.value(), Tv = temperatureAt(ctx).value();
    InelasticIncrements unused;
    // sigma_zz after the return mapping at z, from the values only, with its derivative in z.
    const auto zz = [&](double z)
    {
      std::array<ADReal, 6> strain;
      for (int i = 0; i < 6; ++i)
        strain[i] = ADReal(ee[i].value());
      strain[2] = ADReal::independent(z, 0, 1);
      std::array<ADReal, 6> stress;
      elasticStress(
          Formulation::ThreeDimensional, strain.data(), ADReal(l), ADReal(G), stress.data());
      inelasticReturn(ctx, ADReal(G), ADReal(Tv), old, stress, unused);
      return stress[2];
    };
    double z = ee[2].value();
    const double scale = std::abs(sigma[0].value()) + std::abs(sigma[1].value()) +
                         std::abs(sigma[5].value()) + 1e-300;
    double slope = l + 2.0 * G;
    for (int it = 0; it < 50; ++it)
    {
      const ADReal r = zz(z);
      slope = r.derivative(0);
      const double step = -r.value() / slope;
      z += step;
      if (std::abs(r.value()) <= 1e-14 * scale)
        break;
    }
    // sigma_zz as a function of the unknowns at the solved z, and z(U) = z - sigma_zz(U) / slope.
    std::array<ADReal, 6> strain;
    for (int i = 0; i < 6; ++i)
      strain[i] = ee[i];
    strain[2] = ADReal(z);
    std::array<ADReal, 6> stress;
    elasticStress(Formulation::ThreeDimensional, strain.data(), lambda, mu, stress.data());
    inelasticReturn(ctx, mu, temperatureAt(ctx), old, stress, unused);
    ee[2] = ADReal(z) - (stress[2] - ADReal(stress[2].value())) / slope;
    elasticStress(Formulation::ThreeDimensional, ee, lambda, mu, sigma);
  }

  /// The inelastic strains of one step from the trial stress @p sigma: the stress and the
  /// elastic strain @p ee lose their increments, and the history moves from @p old to @p next.
  void updateInelastic(QpContext & ctx,
                       const ADReal & mu,
                       const double * old,
                       double * next,
                       ADReal * ee,
                       ADReal * sigma) const
  {
    InelasticIncrements inc;
    const std::size_t K = _hardening.size();
    if (old)
    {
      std::array<ADReal, 6> stress;
      for (int i = 0; i < 6; ++i)
        stress[i] = sigma[i];
      inelasticReturn(ctx, mu, temperatureAt(ctx), old, stress, inc);
      for (int i = 0; i < 6; ++i)
      {
        sigma[i] = stress[i];
        ee[i] -= inc.creep[i] + inc.plastic[i];
      }
    }
    else
      inc.back.assign(6 * K, 0.0);
    const auto history = [&](int i) { return old ? old[i] : 0.0; };
    for (int i = 0; i < 6; ++i)
    {
      const double creep = history(i) + inc.creep[i];
      const double plastic = history(7 + i) + inc.plastic[i];
      if (next)
      {
        next[i] = creep;
        next[7 + i] = plastic;
      }
      if (_creep != Creep::None)
        ctx.property(_creep_prop, i) = ADReal(creep);
      if (_plastic)
        ctx.property(_plastic_prop, i) = ADReal(plastic);
    }
    const double creep_equivalent = history(6) + inc.creep_equivalent;
    const double plastic_equivalent = history(13) + inc.plastic_equivalent;
    if (next)
    {
      next[6] = creep_equivalent;
      next[13] = plastic_equivalent;
      for (std::size_t j = 0; j < 6 * K; ++j)
        next[14 + j] = inc.back[j];
    }
    if (_creep != Creep::None)
      ctx.property(_creep_eq) = ADReal(creep_equivalent);
    if (_plastic)
    {
      ctx.property(_plastic_eq) = ADReal(plastic_equivalent);
      for (int i = 0; i < 6; ++i)
      {
        double total = 0.0;
        for (std::size_t k = 0; k < K; ++k)
          total += inc.back[6 * k + i];
        ctx.property(_back_prop, i) = ADReal(total);
      }
    }
  }

  /// The stress of the viscous branches of generalized Maxwell, added to @p sigma: with x =
  /// dt / tau, a deviatoric branch s = exp(-x) s_n + 2 G (1 - exp(-x)) / x (e - e_n) and a bulk
  /// branch p = exp(-x) p_n + K (1 - exp(-x)) / x (theta - theta_n), the exact update for a strain
  /// that varies linearly over the step (Herrmann and Taylor).  A steady state is relaxed: the
  /// branches carry no stress.  The history is the elastic strain and the branch stresses.
  void addViscousStress(
      QpContext & ctx, const ADReal * ee, const double * old, double * next, ADReal * sigma) const
  {
    const int offset = inelasticSize();
    const ADReal theta = ee[0] + ee[1] + ee[2];
    std::array<ADReal, 6> deviator;
    for (int i = 0; i < 6; ++i)
      deviator[i] = i < 3 ? ee[i] - theta / 3.0 : 0.5 * ee[i];
    std::array<double, 6> deviator_old{0, 0, 0, 0, 0, 0};
    double theta_old = 0.0;
    if (old)
    {
      const double * e0 = old + offset;
      theta_old = e0[0] + e0[1] + e0[2];
      for (int i = 0; i < 6; ++i)
        deviator_old[i] = i < 3 ? e0[i] - theta_old / 3.0 : 0.5 * e0[i];
    }
    // A steady solve is relaxed; a step of no length (an output after a step) is the
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
          branch =
              decay * previous + 2.0 * _viscous[b].modulus * ramp * (deviator[i] - deviator_old[i]);
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

  /// Phase-field fracture: the stress (1 - d)^2 sigma+ + sigma- of the split of Miehe, with the
  /// principal strains of the in-plane pair in closed form and the out-of-plane strain the
  /// third, and the history H = max(H_n, psi+), which drives the damage.
  void crackedStress(QpContext & ctx,
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

  /// The strain of the displacements into @p e (Voigt order, engineering shears).  The zz
  /// strain of plane stress follows from the stress and is set later.
  void displacementStrain(const QpContext & ctx, ADReal * e) const
  {
    for (int i = 0; i < 6; ++i)
      e[i] = 0.0;
    const auto & gu = ctx.gradient(_disp[0]);
    e[0] = gu[0];
    const auto hoop = [&](ADReal & out)
    {
      if (ctx.x[0] != 0.0)
        out = ctx.value(_disp[0]) / ctx.x[0];
      else
        out = gu[0];
    };
    if (_formulation == Formulation::AxisymmetricOneD)
    {
      e[1] = ADReal(_axial->value(ctx.x, ctx.time));
      hoop(e[2]);
      return;
    }
    const auto & gv = ctx.gradient(_disp[1]);
    e[1] = gv[1];
    e[5] = gu[1];
    e[5] += gv[0];
    if (_formulation == Formulation::Axisymmetric)
      hoop(e[2]);
    else if (_formulation == Formulation::GeneralizedPlaneStrain)
      e[2] = ctx.value(_out_of_plane);
    else if (_formulation == Formulation::ThreeDimensional)
    {
      const auto & gw = ctx.gradient(_disp[2]);
      e[2] = gw[2];
      e[3] = gv[2];
      e[3] += gw[1];
      e[4] = gu[2];
      e[4] += gw[0];
    }
  }

  void computeProperties(QpContext & ctx) const override
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
          ee[i] -= old[i] + old[7 + i];
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
      mu = E / (2.0 * (1.0 + nu));
      lambda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
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
        e[2] += old[2] + old[9];
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
      vonMises(ctx);
  }

  /// F = I + grad u, row by row, for equations that account for the change of shape (heat
  /// conduction, follower pressure) while the stress stays that of small strain.
  void deformationGradient(QpContext & ctx, const ADReal * e) const
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

  /// The von Mises stress sqrt(3 J2) of the stress.
  void vonMises(QpContext & ctx) const
  {
    const auto s = [&](int i) -> const ADReal & { return ctx.property(_stress, i); };
    const ADReal j2 = ((s(0) - s(1)) * (s(0) - s(1)) + (s(1) - s(2)) * (s(1) - s(2)) +
                       (s(2) - s(0)) * (s(2) - s(0))) /
                          6.0 +
                      s(3) * s(3) + s(4) * s(4) + s(5) * s(5);
    ctx.property(_vm) = sqrt(3.0 * j2);
  }

protected:
  enum class Creep
  {
    None,
    Parsed
  };
  std::shared_ptr<ParsedExpression> _creep_expression;
  std::vector<double> _creep_constants;
  Formulation _formulation;
  FieldCoefficient _E, _nu;
  bool _constant_moduli = false;
  double _lambda0 = 0.0, _mu0 = 0.0;
  Creep _creep = Creep::None;
  double _grain = 0.0, _creep_factor = 1.0;
  int _T = -1;
  std::vector<int> _disp, _eigen;
  FunctionPtr _axial;
  int _stress = -1, _strain = -1, _elastic = -1, _volumetric = -1, _vm = -1, _F_prop = -1;
  int _creep_prop = -1, _creep_eq = -1;
  bool _plastic = false;
  std::shared_ptr<ParsedExpression> _yield_expression;
  std::vector<double> _plasticity_constants;
  struct BackStress
  {
    double modulus, recovery;
  };
  std::vector<BackStress> _hardening;
  int _plastic_prop = -1, _plastic_eq = -1, _back_prop = -1;
  bool _anisotropic = false, _hill_given = false;
  bool _cracking = false;
  int _damage = -1, _driving = -1;
  struct ViscousBranch
  {
    double modulus, time;
  };
  /// The viscous branches: the shear ones first, then the bulk ones.
  std::vector<ViscousBranch> _viscous;
  std::size_t _shear_branches = 0;
  int _out_of_plane = -1;
  std::array<double, 36> _stiffness{}, _hill{};

  /// Refuse a matrix that is not symmetric and positive definite (by its Cholesky factor).
  void checkSymmetricPositive(const std::array<double, 36> & A, const std::string & what) const
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

  /// The elastic stress of the elastic strain @p ee: anisotropic or isotropic.
  template <typename Real>
  void elasticStress(
      Formulation f, ADReal * ee, const Real & lambda, const Real & mu, ADReal * sigma) const
  {
    if (_anisotropic)
      anisotropicStress(f, ee, _stiffness, sigma);
    else
      isotropicStress(f, ee, lambda, mu, sigma);
  }
  /// The registry of the problem, which knows the outputs that some object reads.
  const PropertyRegistry * _registry = nullptr;

  /// Whether the material has a history (creep or plasticity).
  bool inelastic() const { return _creep != Creep::None || _plastic; }
  /// The temperature at the point, or 0 for a material that reads none.
  ADReal temperatureAt(const QpContext & ctx) const
  {
    return _T >= 0 ? ctx.value(_T) : ADReal(0.0);
  }
};

// ---------------------------------------------------------------------------
// Finite strain
// ---------------------------------------------------------------------------
namespace
{
/// A 3 x 3 matrix of numbers of type T.
template <typename T> using Mat3 = std::array<std::array<T, 3>, 3>;

template <typename T>
Mat3<T>
identity3()
{
  Mat3<T> m;
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      m[i][j] = T(i == j ? 1.0 : 0.0);
  return m;
}

template <typename A, typename B>
Mat3<ADReal>
multiply3(const Mat3<A> & a, const Mat3<B> & b)
{
  Mat3<ADReal> c;
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
    {
      ADReal sum(0.0);
      for (int k = 0; k < 3; ++k)
        sum = sum + a[i][k] * b[k][j];
      c[i][j] = sum;
    }
  return c;
}

Mat3<ADReal>
transpose3(const Mat3<ADReal> & a)
{
  Mat3<ADReal> t;
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      t[i][j] = a[j][i];
  return t;
}

ADReal
determinant3(const Mat3<ADReal> & a)
{
  return a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1]) -
         a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0]) +
         a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
}

/// The inverse, by the adjugate.
Mat3<ADReal>
inverse3(const Mat3<ADReal> & a, const ADReal & det)
{
  Mat3<ADReal> inv;
  inv[0][0] = (a[1][1] * a[2][2] - a[1][2] * a[2][1]) / det;
  inv[0][1] = (a[0][2] * a[2][1] - a[0][1] * a[2][2]) / det;
  inv[0][2] = (a[0][1] * a[1][2] - a[0][2] * a[1][1]) / det;
  inv[1][0] = (a[1][2] * a[2][0] - a[1][0] * a[2][2]) / det;
  inv[1][1] = (a[0][0] * a[2][2] - a[0][2] * a[2][0]) / det;
  inv[1][2] = (a[0][2] * a[1][0] - a[0][0] * a[1][2]) / det;
  inv[2][0] = (a[1][0] * a[2][1] - a[1][1] * a[2][0]) / det;
  inv[2][1] = (a[0][1] * a[2][0] - a[0][0] * a[2][1]) / det;
  inv[2][2] = (a[0][0] * a[1][1] - a[0][1] * a[1][0]) / det;
  return inv;
}

/// Voigt index (xx, yy, zz, yz, xz, xy) of the tensor component (i, j).
constexpr int kVoigt[3][3] = {{0, 5, 4}, {5, 1, 3}, {4, 3, 2}};
} // namespace

/// Stress at finite strain, on the undeformed mesh (total Lagrangian).
class FiniteStrainStress : public SmallStrainStress
{
public:
  static InputParameters validParams()
  {
    InputParameters p = SmallStrainStress::validParams();
    p.setClassDescription(
        "Stress of an isotropic material at finite (large) strain, for equations written on the "
        "undeformed mesh (the total Lagrangian form). The deformation gradient F = I + grad u is "
        "taken with respect to the undeformed coordinates, and the stress that the equilibrium "
        "equation needs is the first Piola-Kirchhoff stress P = J sigma F^-T, declared as "
        "'first_piola_kirchhoff_stress' (nine components, row by row) for stress_divergence. "
        "'stress' is the Cauchy (true) stress. stress_update = rotated_small_strain (the default) "
        "applies the small-strain law of small_strain_stress, with the same eigenstrains and "
        "creep, to strain increments measured in a frame that rotates with the material (the "
        "Hughes-Winget midpoint rule), so that rotations produce no stress. This update suits "
        "metals and ceramics, whose elastic strains stay small while their total strains may be "
        "large. stress_update = neo-Hookean is the compressible neo-Hookean hyperelastic law P = "
        "mu (F - F^-T) + lambda ln(J) F^-T, exact for any deformation and without creep, applied "
        "to the part of F left after the eigenstrains (F = F_e F_eig). 'volumetric_strain' is J - 1, "
        "the change of volume per unit volume. Also declares "
        "'deformation_gradient' (nine components) and 'green_lagrange_strain' (Voigt, engineering "
        "shears).");
    p.addOptional("stress_update",
                  ParameterKind::String,
                  std::string("rotated_small_strain"),
                  "rotated_small_strain or neo-Hookean. Default rotated_small_strain: the "
                  "material laws of the library, including creep, carried to finite "
                  "rotation and strain.");
    return p;
  }
  explicit FiniteStrainStress(const InputParameters & p) : SmallStrainStress(p)
  {
    const auto u = p.getString("stress_update");
    if (u != "rotated_small_strain" && u != "neo-Hookean")
      throw InputError("'" + name() +
                       "': stress_update must be rotated_small_strain or "
                       "neo-Hookean, not '" +
                       u + "'.");
    _neo_hookean = u == "neo-Hookean";
    if (_formulation == Formulation::PlaneStress ||
        _formulation == Formulation::GeneralizedPlaneStrain)
      throw InputError("'" + name() +
                       "': finite strain is available in plane_strain, axisymmetric, "
                       "axisymmetric_1d and three_dimensional problems.");
    if (!_viscous.empty())
      throw InputError("'" + name() + "': viscoelasticity at finite strain is not available yet.");
    if (_cracking)
      throw InputError("'" + name() + "': phase-field fracture is available at small strain only.");
    if (_neo_hookean && _plastic)
      throw InputError("'" + name() +
                       "': plasticity is available with stress_update = rotated_small_strain "
                       "only.");
    if (_neo_hookean && _creep != Creep::None)
      throw InputError("'" + name() +
                       "': creep is available with stress_update = "
                       "rotated_small_strain only.");
  }
  /// Rotated elastic strain (6), F - I (9), eigenstrain (6), creep strain (6)
  /// and equivalent creep strain (1) of the last committed state, and with plasticity the
  /// plastic strain (6), its equivalent (1) and the rotated back stresses (6 each).
  int stateSize() const override
  {
    return 28 + (_plastic ? 7 + 6 * static_cast<int>(_hardening.size()) : 0);
  }
  void declareProperties(PropertyRegistry & r) override
  {
    SmallStrainStress::declareProperties(r);
    _pk1 = r.declare("first_piola_kirchhoff_stress", 9);
    _green = r.declare("green_lagrange_strain", 6);
  }

  void computeProperties(QpContext & ctx) const override
  {
    // ---- the deformation gradient on the undeformed configuration ----
    Mat3<ADReal> F = identity3<ADReal>();
    const auto & gu = ctx.gradient(_disp[0]);
    const double R = ctx.x[0];
    if (_formulation == Formulation::AxisymmetricOneD)
    {
      F[0][0] = 1.0 + gu[0];
      F[1][1] = ADReal(1.0 + _axial->value(ctx.x, ctx.time));
      F[2][2] = 1.0 + (R != 0.0 ? ctx.value(_disp[0]) / R : gu[0]);
    }
    else
    {
      const auto & gv = ctx.gradient(_disp[1]);
      F[0][0] = 1.0 + gu[0];
      F[0][1] = gu[1];
      F[1][0] = gv[0];
      F[1][1] = 1.0 + gv[1];
      if (_formulation == Formulation::Axisymmetric)
        F[2][2] = 1.0 + (R != 0.0 ? ctx.value(_disp[0]) / R : gu[0]);
      else if (_formulation == Formulation::ThreeDimensional)
      {
        const auto & gw = ctx.gradient(_disp[2]);
        F[0][2] = gu[2];
        F[1][2] = gv[2];
        F[2][0] = gw[0];
        F[2][1] = gw[1];
        F[2][2] = 1.0 + gw[2];
      }
    }
    const ADReal J = determinant3(F);
    if (!(J.value() > 0.0))
      throw InadmissibleStateError("'" + name() +
                                   "': the deformation turned an element inside out (det F <= 0).");
    const Mat3<ADReal> Finv = inverse3(F, J);

    // ---- the eigenstrains, in Voigt order ----
    std::array<ADReal, 6> eigen;
    eigen.fill(ADReal(0.0));
    for (int id : _eigen)
      for (int i = 0; i < 6; ++i)
        eigen[i] = eigen[i] + ctx.property(id, i);

    const ADReal E = _E.value(ctx);
    const ADReal nu = _nu.value(ctx);
    const ADReal mu = E / (2.0 * (1.0 + nu));
    const ADReal lambda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));

    // The slice of the point's history record that belongs to this object.
    const double * old = ctx.state_old ? ctx.state_old + stateOffset() : nullptr;
    double * next = ctx.state_new ? ctx.state_new + stateOffset() : nullptr;
    std::array<ADReal, 6> sigma;
    std::array<ADReal, 6> elastic;
    std::array<double, 6> creep_increment{0, 0, 0, 0, 0, 0};
    InelasticIncrements inelastic_step;
    inelastic_step.back.assign(6 * _hardening.size(), 0.0);
    double equivalent_increment = 0.0;
    Mat3<ADReal> P;

    if (_neo_hookean)
    {
      // F = F_e F_eig with F_eig = diag(1 + eps_eig) (the eigenstrains here
      // have no shear): psi_0 = J_eig psi(F_e), so P = J_eig P_e F_eig^-T.
      Mat3<ADReal> Fe = F;
      ADReal Jeig(1.0);
      std::array<ADReal, 3> stretch;
      for (int i = 0; i < 3; ++i)
      {
        stretch[i] = 1.0 + eigen[i];
        Jeig = Jeig * stretch[i];
      }
      for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
          Fe[i][j] = F[i][j] / stretch[j];
      const ADReal Je = determinant3(Fe);
      const Mat3<ADReal> FeinvT = transpose3(inverse3(Fe, Je));
      for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
        {
          const ADReal Pe = mu * (Fe[i][j] - FeinvT[i][j]) + lambda * log(Je) * FeinvT[i][j];
          P[i][j] = Jeig * Pe / stretch[j];
        }
      // Cauchy stress sigma = P F^T / J.
      const Mat3<ADReal> S = multiply3(P, transpose3(F));
      for (int i = 0; i < 3; ++i)
        for (int j = i; j < 3; ++j)
          sigma[kVoigt[i][j]] = S[i][j] / J;
      // The elastic strain: the Green-Lagrange strain of F_e, with engineering shears.
      const Mat3<ADReal> Ce = multiply3(transpose3(Fe), Fe);
      for (int i = 0; i < 3; ++i)
        for (int j = i; j < 3; ++j)
          elastic[kVoigt[i][j]] = i == j ? 0.5 * (Ce[i][j] - 1.0) : Ce[i][j];
    }
    else
    {
      // ---- Hughes-Winget: the strain increment on the midpoint configuration
      // and the rotation increment ----
      Mat3<double> F0 = identity3<double>();
      std::array<double, 6> elastic_old{0, 0, 0, 0, 0, 0}, eigen_old{0, 0, 0, 0, 0, 0};
      if (old)
      {
        for (int i = 0; i < 3; ++i)
          for (int j = 0; j < 3; ++j)
            F0[i][j] += old[6 + 3 * i + j];
        for (int i = 0; i < 6; ++i)
        {
          elastic_old[i] = old[i];
          eigen_old[i] = old[15 + i];
        }
      }
      Mat3<ADReal> mid, dF;
      for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
        {
          mid[i][j] = 0.5 * (F[i][j] + F0[i][j]);
          dF[i][j] = F[i][j] - F0[i][j];
        }
      const Mat3<ADReal> G = multiply3(dF, inverse3(mid, determinant3(mid)));
      // Q = (I - W/2)^-1 (I + W/2), W = skew(G).
      Mat3<ADReal> minus = identity3<ADReal>(), plus = identity3<ADReal>();
      for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
        {
          const ADReal w = 0.5 * (G[i][j] - G[j][i]);
          minus[i][j] = minus[i][j] - 0.5 * w;
          plus[i][j] = plus[i][j] + 0.5 * w;
        }
      const Mat3<ADReal> Q = multiply3(inverse3(minus, determinant3(minus)), plus);
      // Rotate the elastic strain of the last state (as a tensor, half shears).
      Mat3<double> e0;
      for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
          e0[i][j] = elastic_old[kVoigt[i][j]] * (i == j ? 1.0 : 0.5);
      const Mat3<ADReal> rotated = multiply3(multiply3(Q, e0), transpose3(Q));
      for (int i = 0; i < 3; ++i)
        for (int j = i; j < 3; ++j)
        {
          const int k = kVoigt[i][j];
          const double factor = i == j ? 1.0 : 2.0; // back to engineering shear
          const ADReal increment = factor * 0.5 * (G[i][j] + G[j][i]);
          elastic[k] = factor * rotated[i][j] + increment - (eigen[k] - eigen_old[k]);
        }
      const ADReal trace = elastic[0] + elastic[1] + elastic[2];
      for (int i = 0; i < 6; ++i)
        sigma[i] = i < 3 ? lambda * trace + 2.0 * mu * elastic[i] : mu * elastic[i];
      if (inelastic() && old)
      {
        // The back stresses of the last state, rotated as the elastic strain is.
        const std::size_t K = _hardening.size();
        std::vector<ADReal> back(6 * K);
        for (std::size_t k = 0; k < K; ++k)
        {
          Mat3<double> b0;
          for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
              b0[i][j] = old[35 + 6 * k + kVoigt[i][j]];
          const Mat3<ADReal> rb = multiply3(multiply3(Q, b0), transpose3(Q));
          for (int i = 0; i < 3; ++i)
            for (int j = i; j < 3; ++j)
              back[6 * k + kVoigt[i][j]] = rb[i][j];
        }
        std::array<ADReal, 6> stress;
        for (int i = 0; i < 6; ++i)
          stress[i] = sigma[i];
        inelasticReturn(ctx,
                        mu,
                        temperatureAt(ctx),
                        old[27],
                        _plastic ? old[34] : 0.0,
                        back,
                        stress,
                        inelastic_step);
        for (int i = 0; i < 6; ++i)
        {
          sigma[i] = stress[i];
          elastic[i] = elastic[i] - inelastic_step.creep[i] - inelastic_step.plastic[i];
        }
        creep_increment = inelastic_step.creep;
        equivalent_increment = inelastic_step.creep_equivalent;
      }
      // P = J sigma F^-T.
      Mat3<ADReal> S;
      for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
          S[i][j] = sigma[kVoigt[i][j]];
      P = multiply3(S, transpose3(Finv));
      for (auto & row : P)
        for (auto & v : row)
          v = J * v;
    }

    // ---- the new state and the declared properties ----
    if (next)
    {
      for (int i = 0; i < 6; ++i)
        next[i] = elastic[i].value();
      for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
          next[6 + 3 * i + j] = F[i][j].value() - (i == j ? 1.0 : 0.0);
      for (int i = 0; i < 6; ++i)
      {
        next[15 + i] = eigen[i].value();
        next[21 + i] = (old ? old[21 + i] : 0.0) + creep_increment[i];
      }
      next[27] = (old ? old[27] : 0.0) + equivalent_increment;
      if (_plastic)
      {
        for (int i = 0; i < 6; ++i)
          next[28 + i] = (old ? old[28 + i] : 0.0) + inelastic_step.plastic[i];
        next[34] = (old ? old[34] : 0.0) + inelastic_step.plastic_equivalent;
        for (std::size_t j = 0; j < inelastic_step.back.size(); ++j)
          next[35 + j] = inelastic_step.back[j];
      }
    }
    if (_plastic)
    {
      for (int i = 0; i < 6; ++i)
        ctx.property(_plastic_prop, i) =
            ADReal((old ? old[28 + i] : 0.0) + inelastic_step.plastic[i]);
      ctx.property(_plastic_eq) = ADReal((old ? old[34] : 0.0) + inelastic_step.plastic_equivalent);
      for (int i = 0; i < 6; ++i)
      {
        double total = 0.0;
        for (std::size_t k = 0; k < _hardening.size(); ++k)
          total += inelastic_step.back[6 * k + i];
        ctx.property(_back_prop, i) = ADReal(total);
      }
    }
    if (_creep != Creep::None)
    {
      for (int i = 0; i < 6; ++i)
        ctx.property(_creep_prop, i) = ADReal((old ? old[21 + i] : 0.0) + creep_increment[i]);
      ctx.property(_creep_eq) = ADReal((old ? old[27] : 0.0) + equivalent_increment);
    }
    // Green-Lagrange strain E = (F^T F - I) / 2, engineering shears.
    const Mat3<ADReal> C = multiply3(transpose3(F), F);
    for (int i = 0; i < 3; ++i)
      for (int j = i; j < 3; ++j)
      {
        const int k = kVoigt[i][j];
        ctx.property(_green, k) = (i == j ? 0.5 * (C[i][j] - 1.0) : C[i][j]);
        ctx.property(_strain, k) = ctx.property(_green, k);
      }
    for (int i = 0; i < 6; ++i)
    {
      ctx.property(_elastic, i) = elastic[i];
      ctx.property(_stress, i) = sigma[i];
    }
    ctx.property(_volumetric) = J - 1.0;
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j)
      {
        ctx.property(_pk1, 3 * i + j) = P[i][j];
        ctx.property(_F_prop, 3 * i + j) = F[i][j];
      }
    const auto s = [&](int i) -> const ADReal & { return ctx.property(_stress, i); };
    const ADReal j2 = ((s(0) - s(1)) * (s(0) - s(1)) + (s(1) - s(2)) * (s(1) - s(2)) +
                       (s(2) - s(0)) * (s(2) - s(0))) /
                          6.0 +
                      s(3) * s(3) + s(4) * s(4) + s(5) * s(5);
    ctx.property(_vm) = sqrt(3.0 * j2);
  }

private:
  bool _neo_hookean = false;
  int _pk1 = -1, _green = -1;
};

/// Isotropic thermal expansion with a constant coefficient, as an eigenstrain.
class ThermalExpansionEigenstrain : public Property
{
public:
  static InputParameters validParams()
  {
    InputParameters p = Property::validParams();
    p.setClassDescription(
        "Isotropic thermal strain alpha (T - T_ref) on the three normal components, as a "
        "six-component eigenstrain property for small_strain_stress.");
    p.addRequired("temperature", ParameterKind::String, "The temperature variable.");
    p.addRequired("linear_expansion_coefficient",
                  ParameterKind::Real,
                  "Linear thermal expansion coefficient alpha, 1/K: the strain per kelvin along "
                  "each direction, a third of the volumetric coefficient of an isotropic solid.");
    p.addRequired("stress_free_temperature",
                  ParameterKind::Real,
                  "Temperature T_ref at which the thermal strain is zero.");
    p.addRequired("eigenstrain_name", ParameterKind::String, "Name of the property.");
    return p;
  }
  explicit ThermalExpansionEigenstrain(const InputParameters & p)
      : Property(p), _alpha(p.getReal("linear_expansion_coefficient")),
        _Tref(p.getReal("stress_free_temperature"))
  {
  }
  void initialSetup(Problem & problem) override
  {
    Property::initialSetup(problem);
    _T = coupledVariable(problem, "temperature");
  }
  void declareProperties(PropertyRegistry & r) override
  {
    _prop = r.declare(_params.getString("eigenstrain_name"), 6);
  }
  void computeProperties(QpContext & ctx) const override
  {
    const ADReal e = _alpha * (ctx.value(_T) - _Tref);
    for (int i = 0; i < 6; ++i)
      ctx.property(_prop, i) = i < 3 ? e : ADReal(0.0);
  }

private:
  double _alpha, _Tref;
  int _T = -1, _prop = -1;
};

/// The sides of the surfaces 'names', each as its two nodes, for the mortar constraints. With
/// 'orient', the nodes of a side are ordered so that (t_y, -t_x) is its outward normal.
std::vector<std::array<Index, 2>>
mortarEdges(const Mesh & mesh,
            const std::vector<std::string> & names,
            bool orient,
            const std::string & who)
{
  std::vector<std::array<Index, 2>> out;
  for (const auto & boundary : names)
    for (const Side & side : mesh.sideset(boundary))
    {
      const auto nodes = mesh.sideNodes(side);
      if (nodes.size() != 2)
        throw InputError("'" + who +
                         "': a mortar surface takes linear elements (two nodes per side).");
      std::array<Index, 2> e{nodes[0], nodes[1]};
      if (orient)
      {
        const Point a = mesh.node(e[0]), b = mesh.node(e[1]);
        const Point c = mesh.elementCentroid(side.first);
        const double nx = b[1] - a[1], ny = -(b[0] - a[0]);
        if (nx * (0.5 * (a[0] + b[0]) - c[0]) + ny * (0.5 * (a[1] + b[1]) - c[1]) < 0.0)
          std::swap(e[0], e[1]);
      }
      out.push_back(e);
    }
  return out;
}

/// Contact between two surfaces of a two-dimensional problem by the mortar method, with
/// Lagrange multipliers: the contact pressure (and, with friction, the tangential traction) at
/// the nodes of the secondary surface.  The weighted gap of node j, g_j = int N_j (x_p - x_s) . n
/// over the secondary surface with the primary point x_p projected along the secondary normal n,
/// and the pressure satisfy g >= 0, lambda >= 0, lambda g = 0, written as the complementarity
/// function lambda - max(0, lambda - c g_j / D_j) = 0 (D_j = int N_j), solved by Newton's
/// method with the active set; Coulomb friction is the complementarity function of Hueber,
/// Stadler and Wohlmuth on the weighted slip.  The scale c changes the convergence only, not
/// the solution.  The pressure acts on both surfaces, -lambda n on the secondary and +lambda n
/// on the primary, so the forces balance exactly.
class MortarContact : public IntegratedBC
{
public:
  static InputParameters validParams()
  {
    InputParameters p = IntegratedBC::validParams();
    p.setClassDescription(
        "Mortar contact of two surfaces in two dimensions: 'variable' is the contact pressure "
        "(Pa) on the nodes of the secondary surface 'boundary', a Lagrange multiplier, and "
        "'primary_boundary' the other surface. The contact is exact (no penetration, no "
        "penalty), by the complementarity of the weighted gap and the pressure, and with a "
        "friction_coefficient it has Coulomb friction through 'tangential_traction'.");
    p.addRequired("primary_boundary", ParameterKind::String, "The primary surface.");
    addDisplacementsParam(p, "The displacements of both bodies.");
    p.addOptional("tangential_traction",
                  ParameterKind::String,
                  std::string(""),
                  "The variable of the tangential traction (Pa) of friction, a second Lagrange "
                  "multiplier on the secondary nodes. Required with a friction_coefficient.");
    p.addOptional("friction_coefficient",
                  ParameterKind::Real,
                  0.0,
                  "The Coulomb friction coefficient mu. Default 0: frictionless.");
    p.setRange("friction_coefficient", ParameterRange::NonNegative);
    p.addRequired("stiffness_scale",
                  ParameterKind::Real,
                  "The scale c (Pa/m) of the complementarity functions, of the size of the "
                  "Young's modulus over the element size. It changes the convergence of Newton's "
                  "method, not the solution.");
    p.setRange("stiffness_scale", ParameterRange::Positive);
    p.addOptional("thickness",
                  ParameterKind::Real,
                  1.0,
                  "The out-of-plane thickness of a plane problem, as stress_divergence takes it. "
                  "Default 1.");
    p.setRange("thickness", ParameterRange::Positive);
    p.addOptional("large_deformation",
                  ParameterKind::Boolean,
                  false,
                  "Measure and orient the contact on the deformed surfaces, for bodies at finite "
                  "strain. Default false: the measure and the normal of the undeformed surface, "
                  "as a small-strain solid measures its forces, with the points projected at "
                  "their current positions, so that the surfaces may slide.");
    return p;
  }
  explicit MortarContact(const InputParameters & p)
      : IntegratedBC(p), _mu(p.getReal("friction_coefficient")), _c(p.getReal("stiffness_scale")),
        _thickness(p.getReal("thickness")), _large(p.getBool("large_deformation"))
  {
    if (_mu > 0.0 && p.getString("tangential_traction").empty())
      throw InputError("'" + name() + "': friction needs the variable 'tangential_traction'.");
  }
  bool isConstraint() const override { return true; }
  ADReal computeBoundaryFlux(const QpContext &) const override { return ADReal(0.0); }
  std::vector<std::string> constraintBoundaries() const override
  {
    std::vector<std::string> names = _boundaries;
    names.push_back(_params.getString("primary_boundary"));
    return names;
  }

  void initialSetup(Problem & problem) override
  {
    IntegratedBC::initialSetup(problem);
    const Mesh & mesh = problem.mesh();
    if (mesh.spaceDimension() != 2)
      throw InputError("'" + name() +
                       "': mortar contact is available in two dimensions; in "
                       "three it is not available yet.");
    _axisymmetric = problem.coordinateSystem() == CoordinateSystem::Axisymmetric;
    _disp.clear();
    for (const auto & d : _params.getStringList("displacements"))
      _disp.push_back(problem.variableIndex(d));
    _tau = _params.getString("tangential_traction").empty()
               ? -1
               : problem.variableIndex(_params.getString("tangential_traction"));
    _secondary = mortarEdges(mesh, _boundaries, true, name());
    _primary = mortarEdges(mesh, {_params.getString("primary_boundary")}, false, name());
    std::set<Index> on;
    for (const auto & e : _secondary)
      on.insert(e.begin(), e.end());
    _secondary_nodes.assign(on.begin(), on.end());
    _multipliers.clear();
    for (Index n : _secondary_nodes)
    {
      _multipliers.push_back(problem.dof(n, _var));
      if (_tau >= 0)
        _multipliers.push_back(problem.dof(n, _tau));
    }
  }
  std::vector<Index> multiplierDofs(const Problem &) const override { return _multipliers; }

  void assembleConstraint(const Problem & problem,
                          const Eigen::VectorXd & U,
                          const Eigen::VectorXd * old,
                          Eigen::VectorXd & R,
                          Eigen::VectorXd * reference,
                          std::vector<Eigen::Triplet<double>> * triplets) const override
  {
    const Mesh & mesh = problem.mesh();
    const bool friction = _tau >= 0;
    // The weighted gap, the measure and the weighted slip of each secondary node, with their
    // derivatives by global unknown.
    struct Sums
    {
      double g = 0, D = 0, s = 0;
      std::map<Index, double> dg, dD, ds;
    };
    std::map<Index, Sums> sums;
    const auto position = [&](Index n, int d)
    { return mesh.node(n)[d] + U[problem.dof(n, _disp[d])]; };
    const auto bounds = [&](const std::array<Index, 2> & e, double grow, std::array<double, 4> & b)
    {
      for (int d = 0; d < 2; ++d)
      {
        const double p0 = position(e[0], d), p1 = position(e[1], d);
        b[2 * d] = std::min(p0, p1) - grow;
        b[2 * d + 1] = std::max(p0, p1) + grow;
      }
    };
    static const double gauss[2] = {-1.0 / std::sqrt(3.0), 1.0 / std::sqrt(3.0)};
    for (const auto & se : _secondary)
    {
      const double length = std::hypot(position(se[1], 0) - position(se[0], 0),
                                       position(se[1], 1) - position(se[0], 1));
      std::array<double, 4> sb;
      bounds(se, 0.5 * length, sb);
      for (const auto & pe : _primary)
      {
        std::array<double, 4> pb;
        bounds(pe, 0.0, pb);
        if (pb[1] < sb[0] || pb[0] > sb[1] || pb[3] < sb[2] || pb[2] > sb[3])
          continue;
        // The local unknowns: the multipliers of the two secondary nodes, then the
        // displacements of A, B (secondary) and C, D (primary).
        std::vector<Index> dofs;
        const std::array<Index, 4> nodes{se[0], se[1], pe[0], pe[1]};
        for (int k = 0; k < 2; ++k)
          dofs.push_back(problem.dof(se[k], _var));
        if (friction)
          for (int k = 0; k < 2; ++k)
            dofs.push_back(problem.dof(se[k], _tau));
        const int first_u = static_cast<int>(dofs.size());
        for (Index n : nodes)
          for (int d = 0; d < 2; ++d)
            dofs.push_back(problem.dof(n, _disp[d]));
        const int nd = static_cast<int>(dofs.size());
        const auto unknown = [&](int slot) { return ADReal::independent(U[dofs[slot]], slot, nd); };
        std::array<std::array<ADReal, 2>, 4> x, u;
        for (int k = 0; k < 4; ++k)
          for (int d = 0; d < 2; ++d)
          {
            u[k][d] = unknown(first_u + 2 * k + d);
            x[k][d] = mesh.node(nodes[k])[d] + u[k][d];
          }
        const ADReal lA = unknown(0), lB = unknown(1);
        const ADReal tA = friction ? unknown(2) : ADReal(0.0),
                     tB = friction ? unknown(3) : ADReal(0.0);
        const ADReal t0 = x[1][0] - x[0][0], t1 = x[1][1] - x[0][1];
        const ADReal L2 = t0 * t0 + t1 * t1;
        // The measure and the normal: of the deformed surface, or of the undeformed one.
        const Point a = mesh.node(se[0]), bnode = mesh.node(se[1]);
        const ADReal L = _large ? sqrt(L2) : ADReal(std::hypot(bnode[0] - a[0], bnode[1] - a[1]));
        const ADReal n0 = _large ? t1 / L : ADReal((bnode[1] - a[1]) / L.value());
        const ADReal n1 = _large ? -t0 / L : ADReal(-(bnode[0] - a[0]) / L.value());
        const auto along = [&](int k)
        { return ((x[k][0] - x[0][0]) * t0 + (x[k][1] - x[0][1]) * t1) / L2; };
        const ADReal xiC = along(2), xiD = along(3);
        ADReal lo = xiC.value() < xiD.value() ? xiC : xiD,
               hi = xiC.value() < xiD.value() ? xiD : xiC;
        if (lo.value() < 0.0)
          lo = ADReal(0.0);
        if (hi.value() > 1.0)
          hi = ADReal(1.0);
        if (hi.value() - lo.value() <= 1e-12)
          continue;
        std::vector<ADReal> Rloc(static_cast<std::size_t>(nd), ADReal(0.0));
        std::array<ADReal, 2> g_node, D_node, s_node;
        for (int q = 0; q < 2; ++q)
        {
          const ADReal xi = lo + (hi - lo) * (1.0 + gauss[q]) / 2.0;
          const ADReal xs0 = x[0][0] + xi * t0, xs1 = x[0][1] + xi * t1;
          const ADReal pm0 = x[3][0] - x[2][0], pm1 = x[3][1] - x[2][1];
          const ADReal eta = ((xs0 - x[2][0]) * t0 + (xs1 - x[2][1]) * t1) / (pm0 * t0 + pm1 * t1);
          const ADReal xm0 = x[2][0] + eta * pm0, xm1 = x[2][1] + eta * pm1;
          const ADReal gap = (xm0 - xs0) * n0 + (xm1 - xs1) * n1;
          ADReal w = (hi - lo) / 2.0 * L * _thickness;
          if (_axisymmetric)
            w = w * (2.0 * M_PI * (_large ? xs0 : ADReal(a[0] + xi.value() * (bnode[0] - a[0]))));
          const std::array<ADReal, 2> Ns{1.0 - xi, xi}, Nm{1.0 - eta, eta};
          const ADReal pressure = lA * Ns[0] + lB * Ns[1];
          const ADReal traction = tA * Ns[0] + tB * Ns[1];
          // The force on the secondary surface is -(lambda n + tau t), on the primary its
          // opposite; the residual takes minus the force.
          // The tangent of the same surface as the normal: (-n1, n0).
          const ADReal f0 = (pressure * n0 - traction * n1) * w;
          const ADReal f1 = (pressure * n1 + traction * n0) * w;
          for (int k = 0; k < 2; ++k)
          {
            Rloc[first_u + 2 * k] += Ns[k] * f0;
            Rloc[first_u + 2 * k + 1] += Ns[k] * f1;
            Rloc[first_u + 2 * (k + 2)] -= Nm[k] * f0;
            Rloc[first_u + 2 * (k + 2) + 1] -= Nm[k] * f1;
          }
          // The slip of the secondary point relative to the primary one, along the tangent,
          // since the start of the step.
          ADReal slip(0.0);
          if (friction)
            for (int d = 0; d < 2; ++d)
            {
              const auto start = [&](int k)
              { return old ? (*old)[dofs[first_u + 2 * k + d]] : 0.0; };
              const ADReal us = Ns[0] * (u[0][d] - start(0)) + Ns[1] * (u[1][d] - start(1));
              const ADReal um = Nm[0] * (u[2][d] - start(2)) + Nm[1] * (u[3][d] - start(3));
              slip += (us - um) * (d == 0 ? -n1 : n0);
            }
          for (int k = 0; k < 2; ++k)
          {
            g_node[k] += Ns[k] * gap * w;
            D_node[k] += Ns[k] * w;
            s_node[k] += Ns[k] * slip * w;
          }
        }
        for (int i = first_u; i < nd; ++i)
        {
          R[dofs[i]] += Rloc[i].value();
          if (reference)
            (*reference)[dofs[i]] += std::abs(Rloc[i].value());
          if (triplets)
            for (int j = 0; j < nd; ++j)
              if (Rloc[i].derivative(j) != 0.0)
                triplets->emplace_back(dofs[i], dofs[j], Rloc[i].derivative(j));
        }
        for (int k = 0; k < 2; ++k)
        {
          Sums & sum = sums[se[k]];
          sum.g += g_node[k].value();
          sum.D += D_node[k].value();
          sum.s += s_node[k].value();
          for (int j = 0; j < nd; ++j)
          {
            if (g_node[k].derivative(j) != 0.0)
              sum.dg[dofs[j]] += g_node[k].derivative(j);
            if (D_node[k].derivative(j) != 0.0)
              sum.dD[dofs[j]] += D_node[k].derivative(j);
            if (s_node[k].derivative(j) != 0.0)
              sum.ds[dofs[j]] += s_node[k].derivative(j);
          }
        }
      }
    }
    // The complementarity functions at each secondary node.
    const auto add = [&](Index row, const std::map<Index, double> & derivatives, double factor)
    {
      if (triplets && factor != 0.0)
        for (const auto & [column, value] : derivatives)
          triplets->emplace_back(row, column, factor * value);
    };
    for (Index n : _secondary_nodes)
    {
      const Index rl = problem.dof(n, _var);
      const double lambda = U[rl];
      const auto it = sums.find(n);
      if (it == sums.end() || it->second.D <= 0.0)
      {
        // No primary surface faces the node: no pressure, no traction.
        R[rl] += lambda;
        if (reference)
          (*reference)[rl] += std::abs(lambda);
        if (triplets)
          triplets->emplace_back(rl, rl, 1.0);
        if (friction)
        {
          const Index rt = problem.dof(n, _tau);
          R[rt] += U[rt];
          if (reference)
            (*reference)[rt] += std::abs(U[rt]);
          if (triplets)
            triplets->emplace_back(rt, rt, 1.0);
        }
        continue;
      }
      const Sums & sum = it->second;
      // The gap per length, gt = g / D, and its derivatives d(gt) = dg / D - g dD / D^2.
      const double gt = sum.g / sum.D;
      std::map<Index, double> dgt;
      for (const auto & [k, v] : sum.dg)
        dgt[k] += v / sum.D;
      for (const auto & [k, v] : sum.dD)
        dgt[k] -= sum.g * v / (sum.D * sum.D);
      // A node that touches is active, so that two faces in contact at the start are held
      // together from the first iterate: a gap of the round-off of the positions (1e-12 of the
      // element size, D) counts as closed.
      const double trial = lambda - _c * gt;
      const bool active = trial >= -_c * 1e-12 * sum.D;
      if (active)
      {
        R[rl] += _c * gt;
        add(rl, dgt, _c);
      }
      else
      {
        R[rl] += lambda;
        if (triplets)
          triplets->emplace_back(rl, rl, 1.0);
      }
      if (reference)
        (*reference)[rl] += std::abs(lambda) + std::abs(_c * gt);
      if (!friction)
        continue;
      // Coulomb friction: C = max(b, |z|) tau - b z, with b = mu max(0, lambda - c gt) and
      // z = tau + c st, st the slip per length.
      const Index rt = problem.dof(n, _tau);
      const double tau = U[rt];
      const double st = sum.s / sum.D;
      std::map<Index, double> dst;
      for (const auto & [k, v] : sum.ds)
        dst[k] += v / sum.D;
      for (const auto & [k, v] : sum.dD)
        dst[k] -= sum.s * v / (sum.D * sum.D);
      const double b = active ? _mu * trial : 0.0;
      // db = mu (d lambda - c d gt) where the node is active.
      std::map<Index, double> db;
      if (active)
      {
        db[rl] += _mu;
        for (const auto & [k, v] : dgt)
          db[k] -= _mu * _c * v;
      }
      const double z = tau + _c * st;
      std::map<Index, double> dz = {{rt, 1.0}};
      for (const auto & [k, v] : dst)
        dz[k] += _c * v;
      double C;
      std::map<Index, double> dC;
      if (b <= 0.0)
      {
        C = tau;
        dC[rt] = 1.0;
      }
      else if (std::abs(z) <= b)
      {
        // Stick: C = b tau - b z = -b c st.
        C = -b * _c * st;
        for (const auto & [k, v] : db)
          dC[k] -= _c * st * v;
        for (const auto & [k, v] : dst)
          dC[k] -= b * _c * v;
      }
      else
      {
        // Slip: C = |z| tau - b z.
        const double sign = z > 0.0 ? 1.0 : -1.0;
        C = std::abs(z) * tau - b * z;
        dC[rt] += std::abs(z);
        for (const auto & [k, v] : dz)
          dC[k] += sign * tau * v - b * v;
        for (const auto & [k, v] : db)
          dC[k] -= z * v;
      }
      R[rt] += C;
      if (reference)
        (*reference)[rt] += std::abs(std::max(b, std::abs(z)) * tau) + std::abs(b * z);
      add(rt, dC, 1.0);
    }
  }

private:
  double _mu, _c, _thickness;
  bool _large;
  bool _axisymmetric = false;
  int _tau = -1;
  std::vector<int> _disp;
  std::vector<std::array<Index, 2>> _secondary, _primary;
  std::vector<Index> _secondary_nodes, _multipliers;
};

/// A cohesive zone between two surfaces of a two-dimensional problem that coincide at the start:
/// a bilinear traction-separation law with damage that does not heal.  The points of the two
/// surfaces are paired once, on the undeformed surfaces, as a bonded interface pairs its material
/// points; the separation of a pair is delta = x_p - x_s, with the normal part g_n = delta . n
/// and the tangential part g_t = delta . t.  The traction on the secondary surface is
/// K (1 - D) delta, except that a closing normal part (g_n < 0) keeps the full stiffness K, so
/// that the surfaces do not pass through each other; the primary surface takes the opposite.  The
/// damage D grows with the largest effective opening kappa = max sqrt(<g_n>^2 + g_t^2) reached:
/// D = 0 up to delta_0 = strength / K, D = delta_f (kappa - delta_0) / (kappa (delta_f - delta_0))
/// up to delta_f = 2 toughness / strength, and D = 1 beyond, so that the traction peaks at the
/// strength and the work of separation per area is the toughness.
class CohesiveZone : public IntegratedBC
{
public:
  static InputParameters validParams()
  {
    InputParameters p = IntegratedBC::validParams();
    p.setClassDescription(
        "A cohesive zone between the surface 'boundary' (secondary) and 'primary_boundary' of a "
        "two-dimensional problem, which coincide at the start: a bilinear traction-separation "
        "law, normal and tangential, with damage that does not heal. The traction peaks at the "
        "'strength' and the work of separation per area is the 'toughness'. 'variable' is one "
        "of the displacements.");
    p.addRequired("primary_boundary", ParameterKind::String, "The primary surface.");
    addDisplacementsParam(p, "The displacements of both bodies.");
    p.addRequired("strength", ParameterKind::Real, "The largest traction (Pa).");
    p.setRange("strength", ParameterRange::Positive);
    p.addRequired("toughness", ParameterKind::Real, "The work of separation per area (J/m^2).");
    p.setRange("toughness", ParameterRange::Positive);
    p.addRequired("stiffness",
                  ParameterKind::Real,
                  "The stiffness K (Pa/m) of the undamaged zone, large against the bodies' (of "
                  "the size of 50 times the Young's modulus over the element size), so that the "
                  "zone opens by strength / K, a small fraction of an element, before it softens.");
    p.setRange("stiffness", ParameterRange::Positive);
    p.addOptional("thickness",
                  ParameterKind::Real,
                  1.0,
                  "The out-of-plane thickness of a plane problem, as stress_divergence takes it. "
                  "Default 1.");
    p.setRange("thickness", ParameterRange::Positive);
    p.addOptional("large_deformation",
                  ParameterKind::Boolean,
                  false,
                  "Take the normal of the deformed secondary surface, for bodies at finite strain. "
                  "Default false: the normal of the undeformed surface.");
    return p;
  }
  explicit CohesiveZone(const InputParameters & p)
      : IntegratedBC(p), _K(p.getReal("stiffness")), _thickness(p.getReal("thickness")),
        _large(p.getBool("large_deformation"))
  {
    const double strength = p.getReal("strength");
    _d0 = strength / _K;
    _df = 2.0 * p.getReal("toughness") / strength;
    if (_df <= _d0)
      throw InputError("'" + name() +
                       "': the zone would fail before it softens: the opening at "
                       "failure, 2 toughness / strength, must exceed the elastic "
                       "opening, strength / stiffness.");
  }
  bool isConstraint() const override { return true; }
  ADReal computeBoundaryFlux(const QpContext &) const override { return ADReal(0.0); }
  std::vector<std::string> constraintBoundaries() const override
  {
    std::vector<std::string> names = _boundaries;
    names.push_back(_params.getString("primary_boundary"));
    return names;
  }

  void initialSetup(Problem & problem) override
  {
    IntegratedBC::initialSetup(problem);
    const Mesh & mesh = problem.mesh();
    if (mesh.spaceDimension() != 2)
      throw InputError("'" + name() +
                       "': cohesive zones are available in two dimensions; in "
                       "three they are not available yet.");
    const bool axisymmetric = problem.coordinateSystem() == CoordinateSystem::Axisymmetric;
    _disp.clear();
    for (const auto & d : _params.getStringList("displacements"))
      _disp.push_back(problem.variableIndex(d));
    // The pairs: the two Gauss points of each overlap of a secondary and a primary side, on the
    // undeformed surfaces.
    static const double gauss[2] = {-1.0 / std::sqrt(3.0), 1.0 / std::sqrt(3.0)};
    _pairs.clear();
    for (const auto & se : mortarEdges(mesh, _boundaries, true, name()))
    {
      const Point a = mesh.node(se[0]), b = mesh.node(se[1]);
      const double t0 = b[0] - a[0], t1 = b[1] - a[1], L = std::hypot(t0, t1);
      const double n0 = t1 / L, n1 = -t0 / L;
      const auto along = [&](const Point & x)
      { return ((x[0] - a[0]) * t0 + (x[1] - a[1]) * t1) / (L * L); };
      const auto off = [&](const Point & x) { return (x[0] - a[0]) * n0 + (x[1] - a[1]) * n1; };
      double covered = 0.0;
      for (const auto & pe :
           mortarEdges(mesh, {_params.getString("primary_boundary")}, false, name()))
      {
        const Point c = mesh.node(pe[0]), d = mesh.node(pe[1]);
        if (std::abs(off(c)) > 1e-8 * L || std::abs(off(d)) > 1e-8 * L)
          continue;
        const double xiC = along(c), xiD = along(d);
        const double lo = std::max(0.0, std::min(xiC, xiD)), hi = std::min(1.0, std::max(xiC, xiD));
        if (hi - lo <= 1e-12)
          continue;
        covered += hi - lo;
        for (int q = 0; q < 2; ++q)
        {
          const double xi = lo + (hi - lo) * (1.0 + gauss[q]) / 2.0;
          const double eta = (xi - xiC) / (xiD - xiC);
          double w = (hi - lo) / 2.0 * L * _thickness;
          if (axisymmetric)
            w *= 2.0 * M_PI * (a[0] + xi * t0);
          std::array<double, 2> offset;
          for (int k = 0; k < 2; ++k)
            offset[k] = (1.0 - eta) * c[k] + eta * d[k] - (1.0 - xi) * a[k] - xi * b[k];
          _pairs.push_back({{se[0], se[1], pe[0], pe[1]},
                            {1.0 - xi, xi},
                            {1.0 - eta, eta},
                            offset,
                            w,
                            {n0, n1}});
        }
      }
      if (std::abs(covered - 1.0) > 1e-8)
        throw InputError("'" + name() +
                         "': the surfaces of a cohesive zone must coincide at the "
                         "start; part of the secondary side at (" +
                         std::to_string(a[0]) + ", " + std::to_string(a[1]) +
                         ") faces no primary side.");
    }
    _start.assign(_pairs.size(), 0.0);
    _trial = _start;
    _committed = _start;
  }

  void assembleConstraint(const Problem & problem,
                          const Eigen::VectorXd & U,
                          const Eigen::VectorXd *,
                          Eigen::VectorXd & R,
                          Eigen::VectorXd * reference,
                          std::vector<Eigen::Triplet<double>> * triplets) const override
  {
    const Mesh & mesh = problem.mesh();
    for (std::size_t i = 0; i < _pairs.size(); ++i)
    {
      const Pair & pair = _pairs[i];
      // The local unknowns: the displacements of A, B (secondary) and C, D (primary).
      std::array<Index, 8> dofs;
      for (int k = 0; k < 4; ++k)
        for (int d = 0; d < 2; ++d)
          dofs[2 * k + d] = problem.dof(pair.nodes[k], _disp[d]);
      std::array<std::array<ADReal, 2>, 4> u;
      for (int k = 0; k < 4; ++k)
        for (int d = 0; d < 2; ++d)
          u[k][d] = ADReal::independent(U[dofs[2 * k + d]], 2 * k + d, 8);
      // The separation from the displacements and the offset at the start, not from the
      // positions, whose round-off the stiffness would magnify.
      std::array<ADReal, 2> delta, n{ADReal(pair.normal[0]), ADReal(pair.normal[1])};
      for (int d = 0; d < 2; ++d)
        delta[d] = pair.offset[d] + pair.Nm[0] * u[2][d] + pair.Nm[1] * u[3][d] -
                   pair.Ns[0] * u[0][d] - pair.Ns[1] * u[1][d];
      if (_large)
      {
        const auto x = [&](int k, int d) { return mesh.node(pair.nodes[k])[d] + u[k][d]; };
        const ADReal t0 = x(1, 0) - x(0, 0), t1 = x(1, 1) - x(0, 1);
        const ADReal L = sqrt(t0 * t0 + t1 * t1);
        n = {t1 / L, -t0 / L};
      }
      const ADReal gn = delta[0] * n[0] + delta[1] * n[1];
      const ADReal gt = -delta[0] * n[1] + delta[1] * n[0];
      // The largest effective opening, which a closing normal part does not add to.
      const double open = std::max(gn.value(), 0.0);
      const double reach = std::sqrt(open * open + gt.value() * gt.value());
      ADReal kappa(_start[i]);
      if (reach > _start[i])
        kappa = sqrt((gn.value() > 0.0 ? gn * gn : ADReal(0.0)) + gt * gt);
      _trial[i] = kappa.value();
      ADReal D(0.0);
      if (kappa.value() >= _df)
        D = ADReal(1.0);
      else if (kappa.value() > _d0)
        D = _df * (kappa - _d0) / (kappa * (_df - _d0));
      const ADReal kn = gn.value() < 0.0 ? ADReal(_K) : _K * (1.0 - D), kt = _K * (1.0 - D);
      // The force per area on the secondary surface; the residual takes minus it, and the
      // primary surface the opposite.
      const std::array<ADReal, 2> f{kn * gn * n[0] - kt * gt * n[1],
                                    kn * gn * n[1] + kt * gt * n[0]};
      for (int k = 0; k < 4; ++k)
        for (int d = 0; d < 2; ++d)
        {
          const ADReal r = (k < 2 ? -pair.Ns[k] : pair.Nm[k - 2]) * f[d] * pair.weight;
          const Index row = dofs[2 * k + d];
          R[row] += r.value();
          if (reference)
            (*reference)[row] += std::abs(r.value());
          if (triplets)
            for (int j = 0; j < 8; ++j)
              if (r.derivative(j) != 0.0)
                triplets->emplace_back(row, dofs[j], r.derivative(j));
        }
    }
  }
  void commitHistory() override
  {
    _committed = _trial;
    _start = _trial;
  }
  void startHistory(bool from_committed) override { _start = from_committed ? _committed : _trial; }

private:
  struct Pair
  {
    std::array<Index, 4> nodes;
    std::array<double, 2> Ns, Nm, offset;
    double weight;
    std::array<double, 2> normal;
  };
  double _K, _thickness, _d0, _df;
  bool _large;
  std::vector<int> _disp;
  std::vector<Pair> _pairs;
  // The largest effective opening of each pair: at the start of the step, reached by the last
  // assembly, and accepted.
  std::vector<double> _start;
  mutable std::vector<double> _trial;
  std::vector<double> _committed;
};

/// A prescribed normal displacement n . u = g of a boundary, the tangential displacements free:
/// a roller on an inclined or curved face.  At each node of the boundary the normal force lambda
/// (N), a Lagrange multiplier, acts on the body along n, and the node's equation is
/// c (n . u - g) = 0, so the condition is exact.  The normal of a node is the mean of the normals
/// of its sides, weighted by their areas.  A node whose free displacements cannot move along n
/// (a corner that other conditions hold) has no multiplier: lambda = 0 there.
class NormalDisplacementBC : public IntegratedBC
{
public:
  static InputParameters validParams()
  {
    InputParameters p = IntegratedBC::validParams();
    p.setClassDescription(
        "A prescribed normal displacement n . u = 'value' on the boundary, with the tangential "
        "displacements free: a roller on an inclined or curved face, exact by a Lagrange "
        "multiplier, 'variable', the normal force (N) at each node of the boundary.");
    addDisplacementsParam(p, "The displacements of the body.");
    p.addOptional("value",
                  ParameterKind::Function,
                  0.0,
                  "The normal displacement g (m), along the outward normal: a constant or an "
                  "expression of x, y, z and t. Default 0.");
    p.addRequired("stiffness_scale",
                  ParameterKind::Real,
                  "The scale c (N/m) of the equation of a node, of the size of the stiffness of "
                  "a node. It changes the conditioning, not the solution.");
    p.setRange("stiffness_scale", ParameterRange::Positive);
    return p;
  }
  explicit NormalDisplacementBC(const InputParameters & p)
      : IntegratedBC(p), _c(p.getReal("stiffness_scale"))
  {
  }
  bool isConstraint() const override { return true; }
  ADReal computeBoundaryFlux(const QpContext &) const override { return ADReal(0.0); }
  std::vector<std::string> constraintBoundaries() const override { return _boundaries; }
  std::vector<Index> multiplierDofs(const Problem &) const override { return _multipliers; }

  void initialSetup(Problem & problem) override
  {
    IntegratedBC::initialSetup(problem);
    const Mesh & mesh = problem.mesh();
    _disp.clear();
    for (const auto & d : _params.getStringList("displacements"))
      _disp.push_back(problem.variableIndex(d));
    if (static_cast<int>(_disp.size()) != mesh.spaceDimension())
      throw InputError("'" + name() + "': give one displacement per dimension of space.");
    _g.setup(problem, _params, "value", name());
    std::map<Index, Point> normals;
    QuadratureSpec spec;
    spec.points = 3;
    std::vector<IntegrationPoint> points;
    MappedPoint mp;
    for (const auto & boundary : _boundaries)
      for (const Side & side : mesh.sideset(boundary))
      {
        const Element & el = mesh.element(side.first);
        buildSidePoints(mesh, side, false, spec, points);
        for (const auto & ip : points)
        {
          mapPoint(mesh, el, ip.xi, mp);
          for (Index n : mesh.sideNodes(side))
          {
            const int a =
                static_cast<int>(std::find(el.nodes.begin(), el.nodes.begin() + el.numNodes(), n) -
                                 el.nodes.begin());
            for (int d = 0; d < 3; ++d)
              normals[n][d] += mp.N[a] * ip.area[d];
          }
        }
      }
    _nodes.clear();
    _multipliers.clear();
    for (auto & [n, v] : normals)
    {
      const double length = std::sqrt(dot(v, v));
      for (int d = 0; d < 3; ++d)
        v[d] /= length;
      _nodes.push_back({n, v});
      _multipliers.push_back(problem.dof(n, _var));
    }
  }

  void assembleConstraint(const Problem & problem,
                          const Eigen::VectorXd & U,
                          const Eigen::VectorXd *,
                          Eigen::VectorXd & R,
                          Eigen::VectorXd * reference,
                          std::vector<Eigen::Triplet<double>> * triplets) const override
  {
    const std::vector<char> held = problem.constrainedDofs();
    const Mesh & mesh = problem.mesh();
    QpContext ctx;
    ctx.time = problem.time();
    for (const auto & [n, normal] : _nodes)
    {
      const Index row = problem.dof(n, _var);
      const double lambda = U[row];
      // Whether a free displacement of the node can move along n.
      double free = 0.0;
      for (std::size_t d = 0; d < _disp.size(); ++d)
        if (!held[static_cast<std::size_t>(problem.dof(n, _disp[d]))])
          free += normal[d] * normal[d];
      if (free < 1e-12)
      {
        R[row] += lambda;
        if (reference)
          (*reference)[row] += std::abs(lambda);
        if (triplets)
          triplets->emplace_back(row, row, 1.0);
        continue;
      }
      ctx.x = mesh.node(n);
      double gap = -_g.value(ctx).value();
      for (std::size_t d = 0; d < _disp.size(); ++d)
      {
        const Index column = problem.dof(n, _disp[d]);
        gap += normal[d] * U[column];
        // The force lambda n on the body; the residual takes minus it.
        R[column] -= lambda * normal[d];
        if (reference)
          (*reference)[column] += std::abs(lambda * normal[d]);
        if (triplets)
        {
          triplets->emplace_back(column, row, -normal[d]);
          triplets->emplace_back(row, column, _c * normal[d]);
        }
      }
      R[row] += _c * gap;
      if (reference)
        (*reference)[row] += std::abs(lambda) + std::abs(_c * gap);
    }
  }

private:
  double _c;
  std::vector<int> _disp;
  FieldCoefficient _g;
  std::vector<std::pair<Index, Point>> _nodes;
  std::vector<Index> _multipliers;
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
        "Prescribed component of the surface traction, in force per unit area. The component is "
        "the one belonging to the equation named by 'variable' (a list of variables adds one "
        "object per variable). The load is given either as the traction itself or as the total "
        "force on the boundary, which is then spread uniformly over its area. In a plane "
        "problem the traction is multiplied by the thickness of the stress_divergence kernel of "
        "the same variable.");
    p.addOptional(
        "traction",
        ParameterKind::Function,
        0.0,
        "The traction component conjugate to 'variable': a constant, a registered "
        "function, or an expression of the fields, x, y, z and t. A positive value acts along the positive direction "
        "of that variable's coordinate axis, whichever way the outward normal "
        "points. In a plane problem it is multiplied by the thickness.");
    p.addOptional("total_force",
                  ParameterKind::Function,
                  0.0,
                  "The total force component conjugate to 'variable' on the whole boundary, "
                  "in N, as a constant or the name of a function of time. It is applied as the "
                  "uniform traction t = F / A, where A is the area of the boundary (with the "
                  "factor 2 pi r in axisymmetric problems). In a plane problem A is the length "
                  "of the boundary times the thickness, so that F is the force on the whole "
                  "thickness. Give either 'traction' or 'total_force'.");
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
        "the order of the coordinate axes, e.g., [u, v, w] in three dimensions.");
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
        "A pressure acting normal to a surface, giving the traction t_i = -p n_i with n "
        "the outward normal, so a positive pressure pushes inward. Added without 'variable', it "
        "acts on every displacement. In a plane problem the pressure is multiplied by the "
        "thickness of the stress_divergence kernel of the same variable.");
    addDisplacementsParam(p, "The pressure acts on the component of 'variable'.");
    p.addOptional("pressure",
                  ParameterKind::Function,
                  0.0,
                  "Pressure in force per unit area: a constant, a registered function, or an "
                  "expression of the fields, x, y, z and t. A positive value presses inward, "
                  "against the outward normal.");
    deformation::addParameter(
        p,
        "the pressure acts on the deformed surface (a follower load), for equations on the "
        "undeformed mesh with finite_strain_stress: the traction per unit undeformed area is "
        "-p J F^-T N (Nanson's relation), with N the undeformed outward normal. Without it "
        "the pressure acts along the undeformed normal on the undeformed area, which is "
        "exact at small strain.");
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

void
registerSolidMechanicsObjects(Factory & f)
{
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
  f.add<FiniteStrainStress>("finite_strain_stress", ObjectCategory::Property, m);
  f.add<MortarContact>("mortar_contact", ObjectCategory::BoundaryCondition, m);
  f.add<CohesiveZone>("cohesive_zone", ObjectCategory::BoundaryCondition, m);
  f.add<NormalDisplacementBC>("normal_displacement", ObjectCategory::BoundaryCondition, m);
  f.add<ThermalExpansionEigenstrain>("thermal_expansion_eigenstrain", ObjectCategory::Property, m);
  registerStructuralMemberObjects(f);
}

} // namespace dualmesh
