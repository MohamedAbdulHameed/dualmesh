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
#include "dualmesh/base/Kernel.h"
#include "dualmesh/base/Material.h"
#include "dualmesh/base/Problem.h"
#include "dualmesh/core/ParsedFunction.h"
#include "dualmesh/modules/FuelProperties.h"

#include <array>
#include <memory>

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
        "Linear elastic stress from the displacement gradients. Declares the material properties "
        "'stress' and 'strain' (Voigt order xx, yy, zz, yz, xz, xy) and 'volumetric_strain'. The "
        "material is isotropic by default. For an orthotropic plane problem, give the reduced "
        "stiffnesses c11, c12, c22 and c66.");
    p.addRequired("displacements",
                  ParameterKind::StringList,
                  "Displacement variables (2 names in 2D and 3 in 3D, which in the axisymmetric "
                  "case are the radial and axial displacements).");
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
                  "Young's modulus E, in a force-per-area unit consistent with the mesh and the "
                  "loads. It is ignored in the orthotropic branch (see 'stiffness_c11').");
    p.addOptional("poissons_ratio",
                  ParameterKind::Real,
                  0.0,
                  "Poisson's ratio nu, which must satisfy -1 < nu < 0.5. The plane-strain, "
                  "axisymmetric and three-dimensional stiffnesses become singular as nu "
                  "approaches 0.5, so a nearly incompressible material requires care. The default "
                  "0 gives no transverse coupling at all and rarely represents the intended "
                  "material, so the value should be set explicitly.");
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
                  "engineering shear strain gamma_xy = 2 eps_xy. Used only when 'stiffness_c11' "
                  "is non-zero. In the orthotropic branch the out-of-plane row of the stiffness "
                  "is left at zero, so the plane-strain sigma_zz is not recovered.");
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
                  "Name of an existing variable to use as the temperature in the thermal strain "
                  "alpha (T - T_reference). Thermal strain is applied only when this and "
                  "'thermal_expansion_coefficient' are both set. Setting one without the other is "
                  "accepted and has no effect. Leave it empty for an isothermal analysis.");
    p.addOptional("stress_free_temperature",
                  ParameterKind::Real,
                  0.0,
                  "Temperature at which the body is free of stress, in the same units as "
                  "the temperature variable. Required when 'temperature' and "
                  "'thermal_expansion_coefficient' are given, because no default is right "
                  "for every problem.");
    return p;
  }

  explicit LinearElasticStress(const InputParameters & p)
      : Material(p), _formulation(formulationFromName(p.getString("formulation")))
  {
    const double E = p.getReal("youngs_modulus");
    const double nu = p.getReal("poissons_ratio");
    _alpha = p.getReal("thermal_expansion_coefficient");
    _reference_temperature = p.getReal("stress_free_temperature");
    if (!p.getString("temperature").empty() && _alpha != 0.0 &&
        !p.isSetByUser("stress_free_temperature"))
      throw InputError("'" + name() +
                       "': give 'stress_free_temperature', the temperature at which the body "
                       "is free of stress, for the thermal strain.");
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
    // sigma_i = C_ij eps_j - thermal (C_i0 + C_i1 + C_i2), accumulated in
    // place: the scaled additions create no temporaries, which matters
    // because this runs at every integration point of every assembly.
    const bool has_thermal = thermal.value() != 0.0 || thermal.size() > 0;
    for (int i = 0; i < 6; ++i)
    {
      ADReal & s = ctx.property(_stress, i);
      s = 0.0;
      for (int j = 0; j < 6; ++j)
        if (_C[i][j] != 0.0)
          s.addScaledBy(strain[j], _C[i][j]);
      if (has_thermal)
        s.addScaledBy(thermal, -(_C[i][0] + _C[i][1] + _C[i][2]));
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
        "Index of the displacement component whose equilibrium equation this instance assembles: "
        "0 for x or r, 1 for y or z, 2 for z. It must agree with the component that 'variable' "
        "represents. The agreement is not checked, and a mismatch assembles the wrong row without "
        "a warning. Add one instance per displacement variable.");
    p.addOptional("thickness",
                  ParameterKind::Real,
                  1.0,
                  "Out-of-plane thickness h multiplying the whole equilibrium equation, for plane "
                  "stress and plane strain only. Leave it at 1 in the three-dimensional case and "
                  "in the axisymmetric case, where the integration measure already carries the "
                  "factor 2 pi r. Any other value is accepted in those cases and gives a wrong "
                  "result. It must match the thickness given to every traction_boundary_condition "
                  "and pressure_boundary_condition of the same model. A point_source is not "
                  "scaled by it, so a concentrated load must already be the total force through "
                  "the thickness.");
    p.addOptional("stress_property",
                  ParameterKind::String,
                  std::string("stress"),
                  "Material property holding the stress: six components in Voigt order "
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
      : Kernel(p), _component(static_cast<int>(p.getInt("component"))),
        _thickness(p.getReal("thickness")), _hoop(p.getBool("axisymmetric_hoop_term"))
  {
    if (_component < 0 || _component > 2)
      throw InputError("'component' must be 0, 1, or 2.");
  }

  void initialSetup(Problem & problem) override
  {
    Kernel::initialSetup(problem);
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
      // finite-strain materials) is stored row by row and is not symmetric.
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

/// Linear elastic stress with temperature-dependent elastic constants and a
/// sum of eigenstrains, for bodies whose moduli come from other materials
/// (fuel, cladding) and whose stress-free strain changes with temperature and
/// irradiation.
class SmallStrainStress : public Material
{
public:
  static InputParameters validParams()
  {
    InputParameters p = Material::validParams();
    p.setClassDescription(
        "Small-strain stress of an isotropic material: sigma = lambda tr(eps_e) I + 2 mu eps_e of "
        "the elastic strain eps_e = eps - sum of the eigenstrains - creep strain, with Young's "
        "modulus and Poisson's ratio taken from material properties (for instance those of "
        "UO2_elasticity) or given as constants, and optional creep. Formulations: axisymmetric "
        "(r, z mesh), axisymmetric_1d (a radial slice of a long body in generalized plane strain, "
        "whose uniform axial strain is given by 'axial_strain'), plane_strain and "
        "three_dimensional. Declares 'stress', 'strain', 'elastic_strain' (in the Voigt order xx, "
        "yy, zz, yz, xz, xy with engineering shears, or r, z, theta in the axisymmetric case) and "
        "'von_mises_stress', and with creep 'creep_strain' and 'equivalent_creep_strain'.");
    p.addRequired("displacements", ParameterKind::StringList, "Displacement variables.");
    p.addRequired("formulation",
                  ParameterKind::String,
                  "axisymmetric, axisymmetric_1d, plane_strain or three_dimensional.");
    p.addOptional("youngs_modulus",
                  ParameterKind::Real,
                  0.0,
                  "Constant Young's modulus, Pa. Default: not given, and the material "
                  "property named by 'youngs_modulus_property' is used.");
    p.addOptional("poissons_ratio",
                  ParameterKind::Real,
                  0.0,
                  "Constant Poisson's ratio. Default: not given, and the material property "
                  "named by 'poissons_ratio_property' is used.");
    p.addOptional("youngs_modulus_property",
                  ParameterKind::String,
                  std::string("youngs_modulus"),
                  "The material property that holds Young's modulus when 'youngs_modulus' is "
                  "not given. Default youngs_modulus, the name the elasticity materials "
                  "declare.");
    p.addOptional("poissons_ratio_property",
                  ParameterKind::String,
                  std::string("poissons_ratio"),
                  "The material property that holds Poisson's ratio when 'poissons_ratio' is "
                  "not given. Default poissons_ratio, the name the elasticity materials "
                  "declare.");
    p.addOptional("eigenstrain_names",
                  ParameterKind::StringList,
                  std::vector<std::string>{},
                  "Names of the six-component eigenstrain material properties to subtract. "
                  "Default none.");
    p.addOptional("axial_strain",
                  ParameterKind::Function,
                  0.0,
                  "The uniform axial strain of the axisymmetric_1d formulation. Default 0.");
    p.addOptional("creep_model",
                  ParameterKind::String,
                  std::string("none"),
                  "Creep of the material: none, uo2 (MATPRO FCREEP), un (dislocation creep of "
                  "Hayes et al. 1990, grain-boundary creep of AbdulHameed et al. 2025 and "
                  "irradiation creep of Konovalov et al. 2016), zircaloy (secondary thermal and "
                  "irradiation creep) or "
                  "parsed (the rate given by 'creep_rate'). "
                  "The creep strain is integrated by the backward Euler method with a radial "
                  "return on the von Mises stress, and kept as a history at every integration "
                  "point. Default none.");
    p.addOptional("creep_rate",
                  ParameterKind::String,
                  std::string(""),
                  "With creep_model = parsed: the equivalent creep strain rate, 1/s, as an "
                  "expression of von_mises_stress (Pa), temperature (K), fission_rate, "
                  "fast_neutron_flux, fast_neutron_fluence, theoretical_density_fraction, "
                  "grain_radius (m) and the constants of 'creep_constant_names'. It is "
                  "differentiated automatically, so the return mapping and the Jacobian are "
                  "exact. It must not be negative. Default empty.");
    p.addOptional("creep_constant_names",
                  ParameterKind::StringList,
                  std::vector<std::string>{},
                  "Names of constants used in 'creep_rate', matched by position with "
                  "'creep_constant_values'. Default none.");
    p.addOptional("creep_constant_values",
                  ParameterKind::RealList,
                  std::vector<double>{},
                  "Values of the constants of 'creep_constant_names'. Default none.");
    p.addOptional("temperature",
                  ParameterKind::String,
                  std::string(""),
                  "The temperature variable (kelvin), required with creep.");
    p.addOptional("fission_rate",
                  ParameterKind::Function,
                  0.0,
                  "Fission rate density, fissions/(m^3 s), for fuel creep. Default 0.");
    p.addOptional("fast_neutron_flux",
                  ParameterKind::Function,
                  0.0,
                  "Fast neutron flux (E > 1 MeV), n/(m^2 s), for cladding creep. Default 0.");
    p.addOptional("fast_neutron_fluence",
                  ParameterKind::Function,
                  0.0,
                  "Fast neutron fluence (E > 1 MeV), n/m^2, for cladding creep. Default 0.");
    p.addOptional("theoretical_density_fraction",
                  ParameterKind::Real,
                  0.95,
                  "Fuel density as a fraction of the theoretical density, for fuel creep. "
                  "Default 0.95, typical of LWR fuel.");
    p.addOptional("grain_radius",
                  ParameterKind::Real,
                  5.0e-6,
                  "Fuel grain radius, m, for UO2 and UN creep: the radius of the sphere of the "
                  "grain's volume (1.56 times half the mean linear intercept). Default 5 um (a "
                  "10 um grain), typical of LWR fuel.");
    p.addOptional("un_grain_boundary_creep",
                  ParameterKind::Boolean,
                  true,
                  "With creep_model = un: include the grain-boundary (Coble) creep term. "
                  "Default true.");
    p.addOptional("creep_rate_factor",
                  ParameterKind::Real,
                  1.0,
                  "Factor on the creep rate of every creep model, for example for a doped fuel "
                  "whose creep is measured to be faster, or for sensitivity and uncertainty "
                  "studies. Default 1.");
    return p;
  }
  int stateSize() const override { return _creep == Creep::None ? 0 : 7; }
  explicit SmallStrainStress(const InputParameters & p) : Material(p)
  {
    _formulation = p.getString("formulation");
    if (_formulation != "axisymmetric" && _formulation != "axisymmetric_1d" &&
        _formulation != "plane_strain" && _formulation != "three_dimensional")
      throw InputError("'" + name() + "': unknown formulation '" + _formulation + "'.");
    // A constant is used only when the user gives one; otherwise the property.
    _E = p.isSetByUser("youngs_modulus") ? p.getReal("youngs_modulus") : 0.0;
    _nu = p.isSetByUser("poissons_ratio") ? p.getReal("poissons_ratio") : -2.0;
    if (p.isSetByUser("youngs_modulus") && !(_E > 0.0))
      throw InputError("'" + name() + "': youngs_modulus must be positive.");
    if (p.isSetByUser("poissons_ratio") && !(_nu > -1.0 && _nu < 0.5))
      throw InputError("'" + name() + "': poissons_ratio must lie in (-1, 0.5).");
    const auto c = p.getString("creep_model");
    if (c == "none")
      _creep = Creep::None;
    else if (c == "uo2")
      _creep = Creep::UO2;
    else if (c == "un")
      _creep = Creep::UN;
    else if (c == "zircaloy")
      _creep = Creep::Zircaloy;
    else if (c == "parsed")
      _creep = Creep::Parsed;
    else
      throw InputError("'" + name() + "': unknown creep_model '" + c +
                       "' (none, uo2, un, zircaloy, parsed).");
    if (_creep == Creep::Parsed)
    {
      const auto text = p.getString("creep_rate");
      if (text.empty())
        throw InputError("'" + name() + "': creep_model = parsed needs 'creep_rate'.");
      std::vector<std::string> names = {"von_mises_stress",
                                        "temperature",
                                        "fission_rate",
                                        "fast_neutron_flux",
                                        "fast_neutron_fluence",
                                        "theoretical_density_fraction",
                                        "grain_radius"};
      const auto constants = p.getStringList("creep_constant_names");
      _creep_constants = p.getRealList("creep_constant_values");
      if (constants.size() != _creep_constants.size())
        throw InputError("'" + name() +
                         "': 'creep_constant_names' and 'creep_constant_values' must have the "
                         "same length.");
      names.insert(names.end(), constants.begin(), constants.end());
      if (names.size() > 32)
        throw InputError("'" + name() + "': too many creep constants (at most 25).");
      _creep_expression = std::make_shared<ParsedExpression>(text, names);
    }
    else if (p.isSetByUser("creep_rate"))
      throw InputError("'" + name() + "': 'creep_rate' is used only with creep_model = parsed.");
    _density = p.getReal("theoretical_density_fraction");
    // The MATPRO UO2 creep correlation takes the grain size in micrometres.
    _grain = 2.0e6 * p.getReal("grain_radius");
    _un_coble = p.getBool("un_grain_boundary_creep");
    _creep_factor = p.getReal("creep_rate_factor");
    if (!(_creep_factor >= 0.0))
      throw InputError("'" + name() + "': creep_rate_factor must not be negative.");
  }
  void initialSetup(Problem & problem) override
  {
    Material::initialSetup(problem);
    if (_creep != Creep::None)
    {
      if (_params.getString("temperature").empty())
        throw InputError("'" + name() + "': creep needs 'temperature'.");
      _T = coupledVariable(problem, "temperature");
    }
    _fission = getFunction(problem, "fission_rate");
    _flux = getFunction(problem, "fast_neutron_flux");
    _fluence = getFunction(problem, "fast_neutron_fluence");
    _disp.clear();
    for (const auto & n : _params.getStringList("displacements"))
      _disp.push_back(problem.variableIndex(n));
    const std::size_t expected =
        _formulation == "axisymmetric_1d" ? 1 : (_formulation == "three_dimensional" ? 3 : 2);
    if (_disp.size() != expected)
      throw InputError("'" + name() + "': the " + _formulation + " formulation needs " +
                       std::to_string(expected) + " displacement variable(s).");
    if ((_formulation == "axisymmetric" || _formulation == "axisymmetric_1d") &&
        problem.coordinateSystem() != CoordinateSystem::Axisymmetric)
      throw InputError("'" + name() + "': the " + _formulation +
                       " formulation needs coordinates = 'axisymmetric'.");
    _axial = getFunction(problem, "axial_strain");
    auto & reg = problem.propertyRegistry();
    _Eprop = _E > 0 ? -1 : reg.id(_params.getString("youngs_modulus_property"));
    _nuprop = _nu > -1.0 ? -1 : reg.id(_params.getString("poissons_ratio_property"));
    _eigen.clear();
    for (const auto & n : _params.getStringList("eigenstrain_names"))
      _eigen.push_back(reg.id(n));
  }
  void declareProperties(MaterialPropertyRegistry & r) override
  {
    _stress = r.declare("stress", 6);
    _strain = r.declare("strain", 6);
    _elastic = r.declare("elastic_strain", 6);
    _vm = r.declare("von_mises_stress", 1);
    _F_prop = r.declare("deformation_gradient", 9);
    if (_creep != Creep::None)
    {
      _creep_prop = r.declare("creep_strain", 6);
      _creep_eq = r.declare("equivalent_creep_strain", 1);
    }
  }

  /// Creep rate at the von Mises stress sigma (Pa) and temperature T.
  template <typename T>
  T creepRate(const T & sigma, const T & temperature_in, const QpContext & ctx) const
  {
    // The von Mises stress is not negative; round-off in the return mapping
    // must not hand a negative one to a fractional power.
    if (fuel::valueOf(sigma) <= 0.0)
      return T(0.0);
    // The correlations are evaluated at 200 K or above (see
    // fuel::correlationTemperature): Newton's method can visit a zero
    // temperature from a cold initial guess, where exp(-Q/RT) has no
    // derivative.
    const T temperature = fuel::correlationTemperature(temperature_in);
    return _creep_factor * baseCreepRate(sigma, temperature, ctx);
  }
  /// Creep rate of the chosen model, before creep_rate_factor.
  template <typename T>
  T baseCreepRate(const T & sigma, const T & temperature, const QpContext & ctx) const
  {
    switch (_creep)
    {
    case Creep::UO2:
      return fuel::uo2CreepRate(
          sigma, temperature, _fission->value(ctx.x, ctx.time), 100.0 * _density, _grain);
    case Creep::UN:
      return fuel::unCreepRate(sigma,
                               temperature,
                               _fission->value(ctx.x, ctx.time),
                               1.0 - _density,
                               _grain,
                               _un_coble,
                               false);
    case Creep::Zircaloy:
      return fuel::zryCreepRate(
          sigma, temperature, _flux->value(ctx.x, ctx.time), _fluence->value(ctx.x, ctx.time));
    case Creep::Parsed:
    {
      std::array<T, 32> args;
      args[0] = sigma;
      args[1] = temperature;
      args[2] = T(_fission->value(ctx.x, ctx.time));
      args[3] = T(_flux->value(ctx.x, ctx.time));
      args[4] = T(_fluence->value(ctx.x, ctx.time));
      args[5] = T(_density);
      args[6] = T(_grain * 0.5e-6);
      for (std::size_t k = 0; k < _creep_constants.size(); ++k)
        args[7 + k] = T(_creep_constants[k]);
      return _creep_expression->value(args.data());
    }
    default:
      return T(0.0);
    }
  }
  /// The backward Euler creep update by a radial return: from the trial
  /// stress @p sigma (Voigt, engineering shear order), the creep strain
  /// increment dp (3/2) s / q along the deviator s, with dp = dt rate(q - 3 G
  /// dp, T).  Updates @p sigma and fills the increments.
  void returnMap(QpContext & ctx,
                 const ADReal & mu,
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
    const ADReal & T = ctx.value(_T);
    const double Tv = T.value();
    if (qv > 1e-3)
    {
      const auto g = [&](double dp) { return dp - ctx.dt * creepRate(qv - 3.0 * G * dp, Tv, ctx); };
      // The root lies in [0, q / 3G]: g(0) <= 0 and g(q / 3G) >= 0.
      double lo = 0.0, hi = qv / (3.0 * G);
      double dp = std::min(ctx.dt * creepRate(qv, Tv, ctx), hi);
      for (int it = 0; it < 60; ++it)
      {
        const double gv = g(dp);
        if (gv > 0)
          hi = dp;
        else
          lo = dp;
        ADReal s1 = ADReal::withZeroDerivatives(qv - 3.0 * G * dp, 1);
        s1.setDerivative(0, 1.0);
        const double slope =
            1.0 + ctx.dt * 3.0 * G * creepRate(s1, ADReal(Tv), ctx).derivatives()[0];
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
      ADReal s1 = ADReal::withZeroDerivatives(qv - 3.0 * G * dp, 1);
      s1.setDerivative(0, 1.0);
      const double slope = 1.0 + ctx.dt * 3.0 * G * creepRate(s1, ADReal(Tv), ctx).derivatives()[0];
      const ADReal residual = dp - ctx.dt * creepRate(q - 3.0 * mu * dp, T, ctx);
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

  void computeProperties(QpContext & ctx) const override
  {
    std::array<ADReal, 6> e;
    e.fill(ADReal(0.0));
    const auto & gu = ctx.gradient(_disp[0]);
    if (_formulation == "axisymmetric_1d")
    {
      e[0] = gu[0];
      e[1] = ADReal(_axial->value(ctx.x, ctx.time));
      e[2] = ctx.x[0] != 0.0 ? ctx.value(_disp[0]) / ctx.x[0] : gu[0];
    }
    else
    {
      const auto & gv = ctx.gradient(_disp[1]);
      e[0] = gu[0];
      e[1] = gv[1];
      e[5] = gu[1] + gv[0];
      if (_formulation == "axisymmetric")
        e[2] = ctx.x[0] != 0.0 ? ctx.value(_disp[0]) / ctx.x[0] : gu[0];
      else if (_formulation == "three_dimensional")
      {
        const auto & gw = ctx.gradient(_disp[2]);
        e[2] = gw[2];
        e[3] = gv[2] + gw[1];
        e[4] = gu[2] + gw[0];
      }
    }
    std::array<ADReal, 6> ee = e;
    for (int id : _eigen)
      for (int i = 0; i < 6; ++i)
        ee[i] -= ctx.property(id, i);
    // The creep strain at the start of the step (the history of this point).
    const double * old = nullptr;
    double * next = nullptr;
    if (_creep != Creep::None && ctx.state_old)
    {
      old = ctx.state_old + stateOffset();
      next = ctx.state_new + stateOffset();
      for (int i = 0; i < 6; ++i)
        ee[i] -= old[i];
    }
    const ADReal E = _Eprop >= 0 ? ctx.property(_Eprop) : ADReal(_E);
    const ADReal nu = _nuprop >= 0 ? ctx.property(_nuprop) : ADReal(_nu);
    const ADReal mu = E / (2.0 * (1.0 + nu));
    const ADReal lambda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
    const ADReal trace = ee[0] + ee[1] + ee[2];
    std::array<ADReal, 6> sigma;
    for (int i = 0; i < 6; ++i)
      sigma[i] = i < 3 ? lambda * trace + 2.0 * mu * ee[i] : mu * ee[i];

    std::array<double, 6> creep_increment{0, 0, 0, 0, 0, 0};
    double equivalent_increment = 0.0;
    if (_creep != Creep::None && ctx.dt > 0 && old)
      returnMap(ctx, mu, sigma, creep_increment, equivalent_increment);
    if (next)
    {
      for (int i = 0; i < 6; ++i)
        next[i] = old[i] + creep_increment[i];
      next[6] = old[6] + equivalent_increment;
    }
    if (_creep != Creep::None)
    {
      for (int i = 0; i < 6; ++i)
        ctx.property(_creep_prop, i) = ADReal(old ? old[i] + creep_increment[i] : 0.0);
      ctx.property(_creep_eq) = ADReal(old ? old[6] + equivalent_increment : 0.0);
    }
    for (int i = 0; i < 6; ++i)
    {
      ctx.property(_strain, i) = e[i];
      ctx.property(_elastic, i) = ee[i] - creep_increment[i];
      ctx.property(_stress, i) = sigma[i];
    }
    // F = I + grad u, row by row, for equations that account for the change
    // of shape (heat conduction, follower pressure) while the stress stays
    // that of small strain.
    for (int i = 0; i < 9; ++i)
      ctx.property(_F_prop, i) = ADReal(i % 4 == 0 ? 1.0 : 0.0);
    ctx.property(_F_prop, 0) = 1.0 + gu[0];
    if (_formulation == "axisymmetric_1d")
    {
      ctx.property(_F_prop, 4) = 1.0 + e[1];
      ctx.property(_F_prop, 8) = 1.0 + e[2];
    }
    else
    {
      const auto & gv = ctx.gradient(_disp[1]);
      ctx.property(_F_prop, 1) = gu[1];
      ctx.property(_F_prop, 3) = gv[0];
      ctx.property(_F_prop, 4) = 1.0 + gv[1];
      if (_formulation == "axisymmetric")
        ctx.property(_F_prop, 8) = 1.0 + e[2];
      else if (_formulation == "three_dimensional")
      {
        const auto & gw = ctx.gradient(_disp[2]);
        ctx.property(_F_prop, 2) = gu[2];
        ctx.property(_F_prop, 5) = gv[2];
        ctx.property(_F_prop, 6) = gw[0];
        ctx.property(_F_prop, 7) = gw[1];
        ctx.property(_F_prop, 8) = 1.0 + gw[2];
      }
    }
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
    UO2,
    UN,
    Zircaloy,
    Parsed
  };
  std::shared_ptr<ParsedExpression> _creep_expression;
  std::vector<double> _creep_constants;
  std::string _formulation;
  double _E = 0.0, _nu = -2.0;
  Creep _creep = Creep::None;
  double _density = 0.95, _grain = 10.0, _creep_factor = 1.0;
  bool _un_coble = true;
  int _T = -1;
  std::vector<int> _disp, _eigen;
  FunctionPtr _axial, _fission, _flux, _fluence;
  int _Eprop = -1, _nuprop = -1, _stress = -1, _strain = -1, _elastic = -1, _vm = -1, _F_prop = -1;
  int _creep_prop = -1, _creep_eq = -1;
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
        "large. stress_update = neo_Hookean is the compressible neo-Hookean hyperelastic law P = "
        "mu (F - F^-T) + lambda ln(J) F^-T, exact for any deformation and without creep, applied "
        "to the part of F left after the eigenstrains (F = F_e F_eig). Also declares "
        "'deformation_gradient' (nine components) and 'green_lagrange_strain' (Voigt, engineering "
        "shears).");
    p.addOptional("stress_update",
                  ParameterKind::String,
                  std::string("rotated_small_strain"),
                  "rotated_small_strain or neo_Hookean. Default rotated_small_strain: the "
                  "material laws of the library, including creep, carried to finite "
                  "rotation and strain.");
    return p;
  }
  explicit FiniteStrainStress(const InputParameters & p) : SmallStrainStress(p)
  {
    const auto u = p.getString("stress_update");
    if (u != "rotated_small_strain" && u != "neo_Hookean")
      throw InputError("'" + name() +
                       "': stress_update must be rotated_small_strain or "
                       "neo_Hookean, not '" +
                       u + "'.");
    _neo_hookean = u == "neo_Hookean";
    if (_neo_hookean && _creep != Creep::None)
      throw InputError("'" + name() +
                       "': creep is available with stress_update = "
                       "rotated_small_strain only.");
  }
  /// Rotated elastic strain (6), F - I (9), eigenstrain (6), creep strain (6)
  /// and equivalent creep strain (1) of the last committed state.
  int stateSize() const override { return 28; }
  void declareProperties(MaterialPropertyRegistry & r) override
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
    if (_formulation == "axisymmetric_1d")
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
      if (_formulation == "axisymmetric")
        F[2][2] = 1.0 + (R != 0.0 ? ctx.value(_disp[0]) / R : gu[0]);
      else if (_formulation == "three_dimensional")
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
      throw std::runtime_error("'" + name() +
                               "': the deformation turned an element inside out (det F <= 0); "
                               "take smaller load or time steps.");
    const Mat3<ADReal> Finv = inverse3(F, J);

    // ---- the eigenstrains, in Voigt order ----
    std::array<ADReal, 6> eigen;
    eigen.fill(ADReal(0.0));
    for (int id : _eigen)
      for (int i = 0; i < 6; ++i)
        eigen[i] = eigen[i] + ctx.property(id, i);

    const ADReal E = _Eprop >= 0 ? ctx.property(_Eprop) : ADReal(_E);
    const ADReal nu = _nuprop >= 0 ? ctx.property(_nuprop) : ADReal(_nu);
    const ADReal mu = E / (2.0 * (1.0 + nu));
    const ADReal lambda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));

    const double * old = ctx.state_old;
    double * next = ctx.state_new;
    std::array<ADReal, 6> sigma;
    std::array<ADReal, 6> elastic;
    std::array<double, 6> creep_increment{0, 0, 0, 0, 0, 0};
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
      for (auto & v : elastic)
        v = ADReal(0.0);
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
      if (_creep != Creep::None && ctx.dt > 0 && old)
      {
        returnMap(ctx, mu, sigma, creep_increment, equivalent_increment);
        for (int i = 0; i < 6; ++i)
          elastic[i] = elastic[i] - creep_increment[i];
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
class ThermalExpansionEigenstrain : public Material
{
public:
  static InputParameters validParams()
  {
    InputParameters p = Material::validParams();
    p.setClassDescription(
        "Isotropic thermal strain alpha (T - T_ref) on the three normal components, as a "
        "six-component eigenstrain property for small_strain_stress.");
    p.addRequired("temperature", ParameterKind::String, "The temperature variable.");
    p.addRequired("thermal_expansion_coefficient", ParameterKind::Real, "alpha, 1/K.");
    p.addRequired("stress_free_temperature",
                  ParameterKind::Real,
                  "Temperature T_ref at which the thermal strain is zero.");
    p.addRequired("eigenstrain_name", ParameterKind::String, "Name of the property.");
    return p;
  }
  explicit ThermalExpansionEigenstrain(const InputParameters & p)
      : Material(p), _alpha(p.getReal("thermal_expansion_coefficient")),
        _Tref(p.getReal("stress_free_temperature"))
  {
  }
  void initialSetup(Problem & problem) override
  {
    Material::initialSetup(problem);
    _T = coupledVariable(problem, "temperature");
  }
  void declareProperties(MaterialPropertyRegistry & r) override
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

/// Frictionless normal contact across a gap, by a penalty.
class GapContact : public InterfaceBC
{
public:
  static InputParameters validParams()
  {
    InputParameters p = InterfaceBC::validParams();
    p.setClassDescription(
        "Frictionless contact between two bodies across a gap, enforced by a penalty: where "
        "the gap g = (x_s + u_s - x_p - u_p) . n_p between the displaced surfaces closes, "
        "the contact pressure is P_c = k max(0, -g), the primary surface receives the "
        "traction -P_c n_p and the secondary surface the opposite force. One instance per "
        "displacement component, with 'variable' that component's displacement. The "
        "penetration at equilibrium is P_c / k, so the penalty k (Pa/m) should be large "
        "compared with the stiffness of the bodies divided by their size, E / L.");
    p.addRequired("displacements", ParameterKind::StringList, "All displacement variables.");
    p.addRequired("component", ParameterKind::Integer, "The component of 'variable'.");
    p.addRequired("penalty", ParameterKind::Real, "Penalty stiffness k, Pa/m.");
    deformation::addParameter(
        p,
        "the contact pressure acts on the deformed primary surface: the traction per unit "
        "undeformed area is -P_c J F^-T N, as for pressure_boundary_condition.");
    return p;
  }
  explicit GapContact(const InputParameters & p) : InterfaceBC(p)
  {
    _component = static_cast<int>(p.getInt("component"));
    _penalty = p.getReal("penalty");
  }
  void initialSetup(Problem & problem) override
  {
    InterfaceBC::initialSetup(problem);
    _disp.clear();
    for (const auto & n : _params.getStringList("displacements"))
    {
      _disp.push_back(problem.variableIndex(n));
      addCoupled(_disp.back());
    }
    _F = deformation::propertyId(problem, _params, name());
  }
  ADReal computeInterfaceFlux(const QpContext & ctx) const override
  {
    ADReal g(0.0);
    for (int d = 0; d < ctx.dim; ++d)
    {
      ADReal diff(ctx.x_other[d] - ctx.x[d]);
      if (d < static_cast<int>(_disp.size()))
        diff = diff + ctx.u_other[_disp[d]] - ctx.u[_disp[d]];
      g += diff * ctx.normal[d];
    }
    if (g.value() >= 0)
      return ADReal(0.0);
    if (_F < 0)
      return (_penalty * g) * ctx.normal[_component];
    ADReal cof[3][3];
    deformation::cofactor(ctx, _F, cof);
    ADReal n(0.0);
    for (int K = 0; K < 3; ++K)
      if (ctx.normal[K] != 0.0)
        n = n + cof[_component][K] * ctx.normal[K];
    return (_penalty * g) * n;
  }

private:
  int _component = 0;
  int _F = -1;
  double _penalty = 0.0;
  std::vector<int> _disp;
};

/// Prescribed traction component on a side set.
class TractionBC : public IntegratedBC
{
public:
  static InputParameters validParams()
  {
    InputParameters p = IntegratedBC::validParams();
    p.setClassDescription(
        "Prescribed component of the surface traction, in force per unit area. The component is "
        "the one belonging to the equation named by 'variable', so the object is added once per "
        "displacement variable. Unlike pressure_boundary_condition, it has no 'component' "
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
                  "must equal the thickness given to the stress_divergence kernels of the "
                  "same model, or the load and the stiffness are scaled inconsistently. "
                  "Leave it at 1 in axisymmetric and three-dimensional problems.");
    p.addOptional("scale_with_load",
                  ParameterKind::Boolean,
                  true,
                  "Multiply this contribution by the load factor during load stepping. This "
                  "parameter defaults to true, unlike the framework default, because the applied "
                  "loading is normally the quantity that is ramped. Set it to false for a part of "
                  "the loading that must stay fixed while the rest is increased.");
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
                  "Index of the displacement component this instance contributes to: 0 for x or "
                  "r, 1 for y or z, 2 for z. It selects which component of the outward normal "
                  "multiplies the pressure, so it must agree with the component that 'variable' "
                  "represents. A mismatch is not detected and applies the pressure along the "
                  "wrong axis. Add one instance per displacement variable.");
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
                  "must equal the thickness given to the stress_divergence kernels of the "
                  "same model. Leave it at 1 in axisymmetric and three-dimensional "
                  "problems.");
    deformation::addParameter(
        p,
        "the pressure acts on the deformed surface (a follower load), for equations on the "
        "undeformed mesh with finite_strain_stress: the traction per unit undeformed area is "
        "-p J F^-T N (Nanson's relation), with N the undeformed outward normal. Without it "
        "the pressure acts along the undeformed normal on the undeformed area, which is "
        "exact at small strain.");
    p.addOptional("scale_with_load",
                  ParameterKind::Boolean,
                  true,
                  "Multiply this contribution by the load factor during load stepping. This "
                  "parameter defaults to true, unlike the framework default, because the applied "
                  "loading is normally the quantity that is ramped. Set it to false for a part of "
                  "the loading that must stay fixed while the rest is increased.");
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
    _F = deformation::propertyId(problem, _params, name());
  }
  ADReal computeBoundaryFlux(const QpContext & ctx) const override
  {
    const double p = _thickness * _p->value(ctx.x, ctx.time);
    if (_F < 0)
      return ADReal(-p * ctx.normal[_component]);
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
  double _thickness;
  int _F = -1;
  FunctionPtr _p;
};

} // namespace

void
registerSolidMechanicsObjects(Factory & f)
{
  const std::string m = "solid_mechanics";
  f.add<LinearElasticStress>("linear_elastic_stress", ObjectCategory::Material, m);
  f.add<StressDivergence>("stress_divergence", ObjectCategory::Kernel, m);
  f.add<TractionBC>("traction_boundary_condition", ObjectCategory::BoundaryCondition, m);
  f.add<PressureBC>("pressure_boundary_condition", ObjectCategory::BoundaryCondition, m);
  f.add<SmallStrainStress>("small_strain_stress", ObjectCategory::Material, m);
  f.add<FiniteStrainStress>("finite_strain_stress", ObjectCategory::Material, m);
  f.add<GapContact>("gap_contact", ObjectCategory::BoundaryCondition, m);
  f.add<ThermalExpansionEigenstrain>("thermal_expansion_eigenstrain", ObjectCategory::Material, m);
  registerStructuralMemberObjects(f);
}

} // namespace dualmesh
