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
#include <memory>
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
  if (name == "axisymmetric")
    return Formulation::Axisymmetric;
  if (name == "axisymmetric_1d")
    return Formulation::AxisymmetricOneD;
  if (name == "three_dimensional")
    return Formulation::ThreeDimensional;
  throw InputError("'" + owner + "': unknown formulation '" + name +
                   "'. Use plane_stress, plane_strain, axisymmetric, axisymmetric_1d or "
                   "three_dimensional.");
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
                  "other value than 1. The traction, pressure and gap_contact conditions of the "
                  "same displacement read it from here. A point_source is not "
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
        "of a property, or an expression. Default: the property poissons_ratio when a "
        "property object declares it, and otherwise 1/3, the default of the library.");
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
                  "expression of von_mises_stress (Pa), temperature (K), grain_size (m) and the "
                  "constants of 'creep_constants'. It is "
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
    return p;
  }
  int stateSize() const override { return _creep == Creep::None ? 0 : 7; }
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
    if (_creep != Creep::None && _formulation == Formulation::PlaneStress)
      throw InputError("'" + name() +
                       "': creep is available in plane_strain, axisymmetric, axisymmetric_1d "
                       "and three_dimensional problems; its return mapping keeps the full "
                       "stress, and plane stress would need sigma_zz = 0 within it.");
    if (_creep == Creep::Parsed)
    {
      const auto text = p.getString("creep_rate");
      if (text.empty())
        throw InputError("'" + name() + "': creep_model = parsed needs 'creep_rate'.");
      std::vector<std::string> names = {"von_mises_stress", "temperature", "grain_size"};
      _creep_constants.clear();
      for (const auto & [constant, value] : p.getRealMap("creep_constants"))
      {
        names.push_back(constant);
        _creep_constants.push_back(value);
      }
      if (names.size() > 32)
        throw InputError("'" + name() + "': too many creep constants (at most 29).");
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
  }
  void initialSetup(Problem & problem) override
  {
    Property::initialSetup(problem);
    const int dim = problem.mesh().dimension();
    if (problem.mesh().spaceDimension() > dim)
      throw InputError("'" + name() +
                       "': the mesh is a surface (or a curve) in space, and the continuum "
                       "formulations need a mesh that fills its space.  Surfaces are shells.");
    if (_creep != Creep::None)
    {
      if (_params.getString("temperature").empty())
        throw InputError("'" + name() + "': creep needs 'temperature'.");
      _T = coupledVariable(problem, "temperature");
    }
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
    auto & reg = problem.propertyRegistry();
    _registry = &reg;
    if (!_params.isSetByUser("youngs_modulus") && !reg.has("youngs_modulus"))
      throw InputError("'" + name() +
                       "': Young's modulus is not given, and no property object declares the "
                       "property youngs_modulus. Give youngs_modulus (Pa), or add a property "
                       "object that declares it, such as constant or parsed.");
    if (!_params.isSetByUser("poissons_ratio") && !reg.has("poissons_ratio"))
      _params.replaceDefault("poissons_ratio", 1.0 / 3.0);
    _E.setup(problem, _params, "youngs_modulus", name());
    _nu.setup(problem, _params, "poissons_ratio", name());
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
  }

  /// Creep rate at the von Mises stress sigma (Pa) and temperature T.
  template <typename T>
  T creepRate(const T & sigma, const T & temperature_in, const QpContext & ctx) const
  {
    // The von Mises stress is not negative; round-off in the return mapping
    // must not hand a negative one to a fractional power.
    if (valueOf(sigma) <= 0.0)
      return T(0.0);
    // The rate is evaluated at 200 K or above: Newton's method can visit a
    // zero temperature from a cold initial guess, where exp(-Q/RT) has no
    // derivative.
    const T temperature = valueOf(temperature_in) < kMinimumCreepTemperature
                              ? T(kMinimumCreepTemperature)
                              : temperature_in;
    return _creep_factor * baseCreepRate(sigma, temperature, ctx);
  }
  /// Creep rate of the chosen model, before creep_rate_factor.
  template <typename T>
  T baseCreepRate(const T & sigma, const T & temperature, const QpContext & ctx) const
  {
    if (_creep != Creep::Parsed)
      return T(0.0);
    std::array<T, 32> args;
    args[0] = sigma;
    args[1] = temperature;
    args[2] = T(_grain);
    for (std::size_t k = 0; k < _creep_constants.size(); ++k)
      args[3 + k] = T(_creep_constants[k]);
    return _creep_expression->value(args.data());
  }
  static double valueOf(double x) { return x; }
  static double valueOf(const ADReal & x) { return x.value(); }
  static constexpr double kMinimumCreepTemperature = 200.0;
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

  /// The creep of one step from the trial stress @p sigma: the stress and the elastic strain
  /// @p ee lose the creep increment, and the history moves from @p old to @p next.
  void updateCreep(QpContext & ctx,
                   const ADReal & mu,
                   const double * old,
                   double * next,
                   ADReal * ee,
                   ADReal * sigma) const
  {
    std::array<double, 6> increment{0, 0, 0, 0, 0, 0};
    double equivalent_increment = 0.0;
    if (ctx.dt > 0 && old)
    {
      std::array<ADReal, 6> stress;
      for (int i = 0; i < 6; ++i)
        stress[i] = sigma[i];
      returnMap(ctx, mu, stress, increment, equivalent_increment);
      for (int i = 0; i < 6; ++i)
      {
        sigma[i] = stress[i];
        ee[i] -= increment[i];
      }
    }
    for (int i = 0; i < 6; ++i)
    {
      const double total = (old ? old[i] : 0.0) + increment[i];
      if (next)
        next[i] = total;
      ctx.property(_creep_prop, i) = ADReal(total);
    }
    const double equivalent = (old ? old[6] : 0.0) + equivalent_increment;
    if (next)
      next[6] = equivalent;
    ctx.property(_creep_eq) = ADReal(equivalent);
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
    ADReal mu;
    if (_constant_moduli)
    {
      isotropicStress(_formulation, ee, _lambda0, _mu0, sigma);
      mu = _mu0;
    }
    else
    {
      const ADReal E = _E.value(ctx);
      const ADReal nu = _nu.value(ctx);
      mu = E / (2.0 * (1.0 + nu));
      const ADReal lambda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
      isotropicStress(_formulation, ee, lambda, mu, sigma);
    }
    if (_formulation == Formulation::PlaneStress)
    {
      // The total zz strain: the elastic one that keeps sigma_zz zero, plus the eigenstrains.
      e[2] = ee[2];
      for (int id : _eigen)
        e[2] += ctx.property(id, 2);
    }
    if (_creep != Creep::None)
      updateCreep(ctx, mu, old, next, ee, sigma);
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
  /// The registry of the problem, which knows the outputs that some object reads.
  const PropertyRegistry * _registry = nullptr;
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
    if (_formulation == Formulation::PlaneStress)
      throw InputError("'" + name() +
                       "': finite strain is available in plane_strain, axisymmetric, "
                       "axisymmetric_1d and three_dimensional problems.");
    if (_neo_hookean && _creep != Creep::None)
      throw InputError("'" + name() +
                       "': creep is available with stress_update = "
                       "rotated_small_strain only.");
  }
  /// Rotated elastic strain (6), F - I (9), eigenstrain (6), creep strain (6)
  /// and equivalent creep strain (1) of the last committed state.
  int stateSize() const override { return 28; }
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

    const ADReal E = _E.value(ctx);
    const ADReal nu = _nu.value(ctx);
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
    p.addRequired("linear_expansion_coeff",
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
      : Property(p), _alpha(p.getReal("linear_expansion_coeff")),
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
        "traction -P_c n_p and the secondary surface the opposite force. Added without "
        "'variable', it acts on every displacement. The "
        "penetration at equilibrium is P_c / k, so the penalty k (Pa/m) should be large "
        "compared with the stiffness of the bodies divided by their size, E / L. In a plane "
        "problem the traction is multiplied by the thickness of the stress_divergence kernel "
        "of the same displacement, as the traction and pressure conditions are.");
    addDisplacementsParam(p, "The contact traction acts on the component of 'variable'.");
    p.addRequired("penalty", ParameterKind::Real, "Penalty stiffness k, Pa/m.");
    p.setRange("penalty", ParameterRange::Positive);
    deformation::addParameter(
        p,
        "the contact pressure acts on the deformed primary surface: the traction per unit "
        "undeformed area is -P_c J F^-T N, as for the pressure condition.");
    return p;
  }
  explicit GapContact(const InputParameters & p) : InterfaceBC(p)
  {
    _component = displacementComponent(name(), variableName(), p.getStringList("displacements"));
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
    _thickness = planeThickness(problem, variableName(), name());
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
      return (_thickness * _penalty * g) * ctx.normal[_component];
    ADReal cof[3][3];
    deformation::cofactor(ctx, _F, cof);
    ADReal n(0.0);
    for (int K = 0; K < 3; ++K)
      if (ctx.normal[K] != 0.0)
        n = n + cof[_component][K] * ctx.normal[K];
    return (_thickness * _penalty * g) * n;
  }

private:
  int _component = 0;
  int _F = -1;
  double _penalty = 0.0;
  double _thickness = 1.0;
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
    p.addOptional("scale_with_load",
                  ParameterKind::Boolean,
                  true,
                  "Multiply this contribution by the load factor during load stepping. This "
                  "parameter defaults to true, unlike the framework default, because the applied "
                  "loading is normally the quantity that is ramped. Set it to false for a part of "
                  "the loading that must stay fixed while the rest is increased.");
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
    _area = problem.boundaryMeasure(_boundaries);
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
  f.add<TractionBC>("traction", ObjectCategory::BoundaryCondition, m);
  f.add<PressureBC>("pressure", ObjectCategory::BoundaryCondition, m);
  f.add<SymmetryBC>("symmetry", ObjectCategory::NodalBC, m);
  f.add<SmallStrainStress>("small_strain_stress", ObjectCategory::Property, m);
  f.add<FiniteStrainStress>("finite_strain_stress", ObjectCategory::Property, m);
  f.add<GapContact>("gap_contact", ObjectCategory::BoundaryCondition, m);
  f.add<ThermalExpansionEigenstrain>("thermal_expansion_eigenstrain", ObjectCategory::Property, m);
  registerStructuralMemberObjects(f);
}

} // namespace dualmesh
