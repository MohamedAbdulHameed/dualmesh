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
                  "Which two- or three-dimensional idealisation to use. 'plane_stress' is "
                  "a thin body free to contract through its thickness, 'plane_strain' a "
                  "long body restrained along its axis, 'axisymmetric' a body of revolution "
                  "on an (r, z) mesh, which requires coordinates = 'axisymmetric' on the "
                  "problem, and 'three_dimensional' a full solid. Any other string is an "
                  "error.");
    p.addOptional("youngs_modulus",
                  ParameterKind::Real,
                  1.0,
                  "Young's modulus E, in a force-per-area unit consistent with the mesh and "
                  "the loads. It is ignored in the orthotropic branch; see "
                  "'stiffness_c11'.");
    p.addOptional("poissons_ratio",
                  ParameterKind::Real,
                  0.0,
                  "Poisson's ratio nu, which must satisfy -1 < nu < 0.5. The plane-strain, "
                  "axisymmetric and three-dimensional stiffnesses become singular as nu "
                  "approaches 0.5, so a nearly incompressible material needs care. The "
                  "default 0 gives no transverse coupling at all and is almost never the "
                  "material intended, so set it explicitly.");
    p.addOptional("stiffness_c11",
                  ParameterKind::Real,
                  0.0,
                  "Reduced plane stiffness c11 of an orthotropic material. Setting it to a "
                  "non-zero value is what selects the orthotropic branch: c11, c12, c22 and "
                  "c66 are then used and 'youngs_modulus' and 'poissons_ratio' are ignored. "
                  "It applies to 'plane_stress' and 'plane_strain' only. It is an error in "
                  "'three_dimensional', and it is ignored without warning in "
                  "'axisymmetric', which always uses E and nu.");
    p.addOptional("stiffness_c12",
                  ParameterKind::Real,
                  0.0,
                  "Reduced plane stiffness c12, the coupling term, placed symmetrically in "
                  "the stiffness matrix. Used only when 'stiffness_c11' is non-zero and the "
                  "formulation is plane stress or plane strain.");
    p.addOptional("stiffness_c22",
                  ParameterKind::Real,
                  0.0,
                  "Reduced plane stiffness c22. Used only when 'stiffness_c11' is non-zero "
                  "and the formulation is plane stress or plane strain.");
    p.addOptional("stiffness_c66",
                  ParameterKind::Real,
                  0.0,
                  "Reduced in-plane shear stiffness c66, relating the shear stress to the "
                  "engineering shear strain gamma_xy rather than to eps_xy. Used only when "
                  "'stiffness_c11' is non-zero. In the orthotropic branch the out-of-plane "
                  "row of the stiffness is left at zero, so the plane-strain sigma_zz is "
                  "not recovered.");
    p.addOptional("thermal_expansion_coefficient",
                  ParameterKind::Real,
                  0.0,
                  "Isotropic coefficient of thermal expansion alpha, per degree of the "
                  "temperature variable. It is subtracted from the three normal strains "
                  "only, leaving the shear strains untouched, so the thermal strain stays "
                  "isotropic even when the stiffness is orthotropic. It has an effect only "
                  "when 'temperature' also names a variable.");
    p.addOptional("temperature",
                  ParameterKind::String,
                  std::string(""),
                  "Name of an existing variable to use as the temperature in the thermal "
                  "strain alpha (T - T_reference). Thermal strain is applied only when this "
                  "and 'thermal_expansion_coefficient' are both set; setting one without "
                  "the other is accepted and does nothing. Leave it empty for an isothermal "
                  "analysis.");
    p.addOptional("reference_temperature",
                  ParameterKind::Real,
                  0.0,
                  "Temperature at which the body is free of stress, in the same units as "
                  "the temperature variable. It is ignored when 'temperature' is empty. The "
                  "default 0 turns the whole temperature field into a thermal load, which "
                  "is seldom intended.");
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
      // Plane strain is the three-dimensional law with eps_zz held at zero, so
      // the matrix here is the full isotropic one, lambda + 2 mu on the
      // diagonal and lambda off it, written with f = E / [(1 + nu)(1 - 2 nu)].
      // The zz column is kept even though eps_zz is always zero, because the
      // thermal strain alpha (T - T_ref) is subtracted from all three normal
      // strains before the multiplication, and the zz entry of that subtraction
      // is what carries the out-of-plane part of the thermal stress.  Dropping
      // the column makes the in-plane thermal stress of a fully restrained body
      // come out as E alpha dT / [(1 + nu)(1 - 2 nu)] instead of the correct
      // E alpha dT / (1 - 2 nu).
      const double f = E / ((1 + nu) * (1 - 2 * nu));
      _C[0][0] = _C[1][1] = _C[2][2] = f * (1 - nu);
      _C[0][1] = _C[1][0] = f * nu;
      _C[0][2] = _C[2][0] = f * nu;
      _C[1][2] = _C[2][1] = f * nu;
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
        "F = h (sigma_i0, sigma_i1, sigma_i2), where h is the thickness. The source is zero "
        "except in the axisymmetric case, where the radial equation (component 0) carries "
        "S = sigma_tt / r, taken as zero on the axis. The natural boundary quantity is the "
        "traction component times the thickness.");
    p.addRequired(
        "component",
        ParameterKind::Integer,
        "Index of the displacement component whose equilibrium equation this instance "
        "assembles: 0 for x or r, 1 for y or z, 2 for z. It must agree with the component "
        "that 'variable' represents; the agreement is not checked, and a mismatch silently "
        "assembles the wrong row. Add one instance per displacement variable.");
    p.addOptional("thickness",
                  ParameterKind::Real,
                  1.0,
                  "Out-of-plane thickness h multiplying the whole equilibrium equation, for "
                  "plane stress and plane strain only. Leave it at 1 in the "
                  "three-dimensional case and in the axisymmetric case, where the "
                  "integration measure already carries the factor 2 pi r; any other value "
                  "there is accepted and gives a wrong answer. It must match the thickness "
                  "given to every TractionBC and PressureBC of the same model. A PointSource "
                  "is not scaled by it, so a concentrated load must already be the total "
                  "force through the thickness.");
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
    {
      F[d] = ctx.property(_stress, voigt[_component][d]);
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
    p.setClassDescription(
        "Prescribed component of the surface traction, in force per unit area. The "
        "component is the one belonging to the equation named by 'variable', so the object "
        "is added once per displacement variable; unlike PressureBC it has no 'component' "
        "parameter.");
    p.addOptional("traction",
                  ParameterKind::Function,
                  0.0,
                  "The traction component conjugate to 'variable', as a constant or the "
                  "name of a function. A positive value acts along the positive direction "
                  "of that variable's coordinate axis, whichever way the outward normal "
                  "points. The value is multiplied by 'thickness'.");
    p.addOptional("thickness",
                  ParameterKind::Real,
                  1.0,
                  "Out-of-plane thickness h multiplying the traction, for plane problems. It "
                  "must equal the thickness given to the StressDivergence kernels of the "
                  "same model, or the load and the stiffness are scaled inconsistently. "
                  "Leave it at 1 in axisymmetric and three-dimensional problems.");
    p.addOptional("scale_with_load",
                  ParameterKind::Boolean,
                  true,
                  "Multiply this contribution by the load factor during load stepping. Unlike "
                  "the framework default this is true, because applied loading is normally "
                  "what is ramped; set it to false for a part of the loading that must stay "
                  "fixed while the rest is increased.");
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
    p.setClassDescription(
        "A pressure acting normal to a surface, giving the traction t_i = -p n_i with n "
        "the outward normal, so a positive pressure pushes inward. Add one instance per "
        "displacement variable, each with its own 'component'.");
    p.addRequired("component",
                  ParameterKind::Integer,
                  "Index of the displacement component this instance contributes to: 0 for "
                  "x or r, 1 for y or z, 2 for z. It selects which component of the outward "
                  "normal multiplies the pressure, so it must agree with the component that "
                  "'variable' represents; a mismatch is not detected and applies the "
                  "pressure along the wrong axis. Add one instance per displacement "
                  "variable.");
    p.addOptional("pressure",
                  ParameterKind::Function,
                  0.0,
                  "Pressure in force per unit area, as a constant or the name of a "
                  "function. A positive value presses inward, against the outward normal. "
                  "It is multiplied by 'thickness' in plane problems.");
    p.addOptional("thickness",
                  ParameterKind::Real,
                  1.0,
                  "Out-of-plane thickness h multiplying the pressure, for plane problems. It "
                  "must equal the thickness given to the StressDivergence kernels of the "
                  "same model. Leave it at 1 in axisymmetric and three-dimensional "
                  "problems.");
    p.addOptional("scale_with_load",
                  ParameterKind::Boolean,
                  true,
                  "Multiply this contribution by the load factor during load stepping. Unlike "
                  "the framework default this is true, because applied loading is normally "
                  "what is ramped; set it to false for a part of the loading that must stay "
                  "fixed while the rest is increased.");
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
  registerStructuralMemberObjects(f);
}

} // namespace dualmesh
