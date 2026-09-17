// SPDX-License-Identifier: LGPL-2.1-or-later
//
// Solid mechanics module: linear elasticity in plane stress, plane strain,
// axisymmetric, and three-dimensional settings.
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
#include "dualmesh/base/Factory.h"
#include "dualmesh/base/Kernel.h"
#include "dualmesh/base/Material.h"
#include "dualmesh/base/Problem.h"

#include <array>

namespace dualmesh
{

namespace
{

enum class Formulation
{
  PlaneStress,
  PlaneStrain,
  Axisymmetric,
  ThreeDimensional
};

Formulation
formulationFromName(const std::string & name)
{
  if (name == "plane_stress")
    return Formulation::PlaneStress;
  if (name == "plane_strain")
    return Formulation::PlaneStrain;
  if (name == "axisymmetric")
    return Formulation::Axisymmetric;
  if (name == "three_dimensional")
    return Formulation::ThreeDimensional;
  throw InputError("Unknown formulation '" + name +
                   "' (use plane_stress, plane_strain, axisymmetric, or three_dimensional).");
}

/// Voigt ordering used for the "stress" and "strain" properties:
/// 0: xx, 1: yy, 2: zz (hoop in the axisymmetric case), 3: yz, 4: xz, 5: xy.
class LinearElasticStress : public Material
{
public:
  static InputParameters validParams()
  {
    InputParameters p = Material::validParams();
    p.setClassDescription(
        "Linear elastic stress from the displacement gradients. Declares the material "
        "properties 'stress' and 'strain' (Voigt order xx, yy, zz, yz, xz, xy) and "
        "'volumetric_strain'. Isotropic by default; for an orthotropic plane problem give "
        "the reduced stiffnesses c11, c12, c22, c66 instead.");
    p.addRequired("displacements",
                  ParameterKind::StringList,
                  "Displacement variables (2 names in 2D, 3 in 3D; in the axisymmetric case "
                  "the radial and axial displacements).");
    p.addOptional("formulation",
                  ParameterKind::String,
                  std::string("plane_stress"),
                  "plane_stress, plane_strain, axisymmetric, or three_dimensional.");
    p.addOptional("youngs_modulus", ParameterKind::Real, 1.0, "Young's modulus E.");
    p.addOptional("poissons_ratio", ParameterKind::Real, 0.0, "Poisson's ratio nu.");
    p.addOptional("stiffness_c11", ParameterKind::Real, 0.0, "Orthotropic plane stiffness c11.");
    p.addOptional("stiffness_c12", ParameterKind::Real, 0.0, "Orthotropic plane stiffness c12.");
    p.addOptional("stiffness_c22", ParameterKind::Real, 0.0, "Orthotropic plane stiffness c22.");
    p.addOptional("stiffness_c66", ParameterKind::Real, 0.0, "Orthotropic shear stiffness c66.");
    p.addOptional("thermal_expansion_coefficient",
                  ParameterKind::Real,
                  0.0,
                  "Coefficient of thermal expansion alpha (isotropic).");
    p.addOptional("temperature",
                  ParameterKind::String,
                  std::string(""),
                  "Temperature variable driving thermal strains (optional).");
    p.addOptional("reference_temperature", ParameterKind::Real, 0.0, "Stress-free temperature.");
    return p;
  }

  explicit LinearElasticStress(const InputParameters & p)
      : Material(p), _formulation(formulationFromName(p.getString("formulation")))
  {
    const double E = p.getReal("youngs_modulus");
    const double nu = p.getReal("poissons_ratio");
    _alpha = p.getReal("thermal_expansion_coefficient");
    _reference_temperature = p.getReal("reference_temperature");
    const bool orthotropic = p.isSet("stiffness_c11") && p.getReal("stiffness_c11") != 0.0;
    for (auto & row : _C)
      row.fill(0.0);
    if (_formulation == Formulation::ThreeDimensional)
    {
      if (orthotropic)
        throw InputError("Give an isotropic E and nu for the three-dimensional formulation.");
      const double lambda = E * nu / ((1 + nu) * (1 - 2 * nu));
      const double mu = E / (2 * (1 + nu));
      for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
          _C[i][j] = lambda + (i == j ? 2 * mu : 0.0);
      for (int i = 3; i < 6; ++i)
        _C[i][i] = mu;
    }
    else if (_formulation == Formulation::Axisymmetric)
    {
      const double lambda = E * nu / ((1 + nu) * (1 - 2 * nu));
      const double mu = E / (2 * (1 + nu));
      // (rr, zz, tt) behave like a three-dimensional material with no shear
      // in the hoop directions; the only shear is rz (stored in slot 5).
      for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
          _C[i][j] = lambda + (i == j ? 2 * mu : 0.0);
      _C[5][5] = mu;
    }
    else if (orthotropic)
    {
      _C[0][0] = p.getReal("stiffness_c11");
      _C[0][1] = _C[1][0] = p.getReal("stiffness_c12");
      _C[1][1] = p.getReal("stiffness_c22");
      _C[5][5] = p.getReal("stiffness_c66");
    }
    else if (_formulation == Formulation::PlaneStress)
    {
      const double f = E / (1 - nu * nu);
      _C[0][0] = _C[1][1] = f;
      _C[0][1] = _C[1][0] = nu * f;
      _C[5][5] = E / (2 * (1 + nu));
    }
    else // plane strain
    {
      const double f = E / ((1 + nu) * (1 - 2 * nu));
      _C[0][0] = _C[1][1] = f * (1 - nu);
      _C[0][1] = _C[1][0] = f * nu;
      _C[2][0] = _C[2][1] = f * nu; // out-of-plane stress, post-computed
      _C[2][2] = f * (1 - nu);
      _C[5][5] = E / (2 * (1 + nu));
    }
  }

  void initialSetup(Problem & problem) override
  {
    Material::initialSetup(problem);
    const auto names = _params.getStringList("displacements");
    const int dim = problem.mesh().dimension();
    const int expected = _formulation == Formulation::ThreeDimensional ? 3 : 2;
    if (static_cast<int>(names.size()) != expected)
      throw InputError("'displacements' needs " + std::to_string(expected) + " variable names.");
    if (dim != (_formulation == Formulation::ThreeDimensional ? 3 : 2))
      throw InputError("The chosen formulation does not match the mesh dimension.");
    _disp.clear();
    for (const auto & n : names)
      _disp.push_back(problem.variableIndex(n));
    const auto temperature = _params.getString("temperature");
    _temperature = temperature.empty() ? -1 : problem.variableIndex(temperature);
  }

  void declareProperties(MaterialPropertyRegistry & r) override
  {
    _stress = r.declare("stress", 6);
    _strain = r.declare("strain", 6);
    _volumetric = r.declare("volumetric_strain", 1);
  }

  void computeProperties(QpContext & ctx) const override
  {
    std::array<ADReal, 6> strain;
    strain.fill(ADReal(0.0));
    const auto & gu = ctx.gradient(_disp[0]);
    const auto & gv = ctx.gradient(_disp[1]);
    strain[0] = gu[0];
    strain[1] = gv[1];
    strain[5] = gu[1] + gv[0];
    if (_formulation == Formulation::Axisymmetric)
    {
      // eps_rr = du/dr, eps_zz = dv/dz, eps_tt = u/r, gamma_rz = du/dz + dv/dr
      strain[1] = gv[1];
      strain[2] = ctx.x[0] != 0.0 ? ctx.value(_disp[0]) / ctx.x[0] : gu[0];
      strain[5] = gu[1] + gv[0];
    }
    else if (_formulation == Formulation::ThreeDimensional)
    {
      const auto & gw = ctx.gradient(_disp[2]);
      strain[2] = gw[2];
      strain[3] = gv[2] + gw[1];
      strain[4] = gu[2] + gw[0];
    }
    ADReal thermal(0.0);
    if (_temperature >= 0 && _alpha != 0.0)
      thermal = _alpha * (ctx.value(_temperature) - _reference_temperature);
    for (int i = 0; i < 6; ++i)
      ctx.property(_strain, i) = strain[i];
    ctx.property(_volumetric) = strain[0] + strain[1] + strain[2];
    for (int i = 0; i < 6; ++i)
    {
      ADReal s(0.0);
      for (int j = 0; j < 6; ++j)
        if (_C[i][j] != 0.0)
          s += (j < 3 ? (strain[j] - thermal) : strain[j]) * _C[i][j];
      ctx.property(_stress, i) = s;
    }
  }

private:
  Formulation _formulation;
  std::array<std::array<double, 6>, 6> _C;
  std::vector<int> _disp;
  int _temperature = -1;
  double _alpha = 0.0, _reference_temperature = 0.0;
  int _stress = -1, _strain = -1, _volumetric = -1;
};

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
        "F = h (sigma_i0, sigma_i1, sigma_i2). The natural boundary quantity is the traction "
        "component times the thickness. In the axisymmetric case the hoop stress enters the "
        "radial equation as a source.");
    p.addRequired(
        "component", ParameterKind::Integer, "Component index: 0 (x or r), 1 (y or z), or 2 (z).");
    p.addOptional("thickness", ParameterKind::Real, 1.0, "Thickness h of a plane body.");
    p.addOptional("stress_property",
                  ParameterKind::String,
                  std::string("stress"),
                  "Material property holding the stress in Voigt order.");
    p.addOptional("axisymmetric_hoop_term",
                  ParameterKind::Boolean,
                  true,
                  "Include the sigma_tt / r source in the radial equation (axisymmetric only).");
    return p;
  }

  explicit StressDivergence(const InputParameters & p)
      : Kernel(p), _component(static_cast<int>(p.getInt("component"))),
        _thickness(p.getReal("thickness")), _hoop(p.getBool("axisymmetric_hoop_term"))
  {
    if (_component < 0 || _component > 2)
      throw InputError("'component' must be 0, 1, or 2.");
  }

  void initialSetup(Problem & problem) override
  {
    Kernel::initialSetup(problem);
    _stress = problem.propertyRegistry().id(_params.getString("stress_property"));
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
      F[d] = _thickness * ctx.property(_stress, voigt[_component][d]);
  }

  ADReal computeSource(const QpContext & ctx) const override
  {
    if (!(_axisymmetric && _hoop && _component == 0))
      return ADReal(0.0);
    if (ctx.x[0] == 0.0)
      return ADReal(0.0);
    return ctx.property(_stress, 2) / ctx.x[0];
  }

private:
  int _component;
  double _thickness;
  bool _hoop;
  bool _axisymmetric = false;
  int _stress = -1;
};

/// Prescribed traction component on a side set.
class TractionBC : public IntegratedBC
{
public:
  static InputParameters validParams()
  {
    InputParameters p = IntegratedBC::validParams();
    p.setClassDescription("Prescribed traction component t_i (force per unit area).");
    p.addOptional("traction", ParameterKind::Function, 0.0, "Traction component t_i(x, t).");
    p.addOptional("thickness", ParameterKind::Real, 1.0, "Thickness h of a plane body.");
    p.addOptional("scale_with_load", ParameterKind::Boolean, true, "Scale with the load factor.");
    return p;
  }
  explicit TractionBC(const InputParameters & p)
      : IntegratedBC(p), _thickness(p.getReal("thickness"))
  {
  }
  void initialSetup(Problem & problem) override
  {
    IntegratedBC::initialSetup(problem);
    _t = getFunction(problem, "traction");
  }
  ADReal computeBoundaryFlux(const QpContext & ctx) const override
  {
    return ADReal(_thickness * _t->value(ctx.x, ctx.time));
  }

private:
  double _thickness;
  FunctionPtr _t;
};

/// Prescribed normal pressure: t_i = -p n_i.
class PressureBC : public IntegratedBC
{
public:
  static InputParameters validParams()
  {
    InputParameters p = IntegratedBC::validParams();
    p.setClassDescription("Normal pressure p acting on a surface: t_i = -p n_i.");
    p.addRequired("component", ParameterKind::Integer, "Component index of this equation.");
    p.addOptional("pressure", ParameterKind::Function, 0.0, "Pressure p(x, t) (positive inward).");
    p.addOptional("thickness", ParameterKind::Real, 1.0, "Thickness h of a plane body.");
    p.addOptional("scale_with_load", ParameterKind::Boolean, true, "Scale with the load factor.");
    return p;
  }
  explicit PressureBC(const InputParameters & p)
      : IntegratedBC(p), _component(static_cast<int>(p.getInt("component"))),
        _thickness(p.getReal("thickness"))
  {
  }
  void initialSetup(Problem & problem) override
  {
    IntegratedBC::initialSetup(problem);
    _p = getFunction(problem, "pressure");
  }
  ADReal computeBoundaryFlux(const QpContext & ctx) const override
  {
    return ADReal(-_thickness * _p->value(ctx.x, ctx.time) * ctx.normal[_component]);
  }

private:
  int _component;
  double _thickness;
  FunctionPtr _p;
};

} // namespace

void
registerSolidMechanicsObjects(Factory & f)
{
  const std::string m = "solid_mechanics";
  f.add<LinearElasticStress>("LinearElasticStress", ObjectCategory::Material, m);
  f.add<StressDivergence>("StressDivergence", ObjectCategory::Kernel, m);
  f.add<TractionBC>("TractionBC", ObjectCategory::BoundaryCondition, m);
  f.add<PressureBC>("PressureBC", ObjectCategory::BoundaryCondition, m);
}

} // namespace dualmesh
