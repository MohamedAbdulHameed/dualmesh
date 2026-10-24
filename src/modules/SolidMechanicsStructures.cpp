// SPDX-License-Identifier: LGPL-2.1-or-later
//
// Solid mechanics module, structural members: beams, axisymmetric circular
// plates, and rectangular plates, including functionally graded
// (through-thickness graded) sections and the von Karman geometric
// nonlinearity.
//
// These are the reduced theories of solid mechanics: a beam or a plate is a
// three-dimensional elastic body whose kinematics have been restricted by an
// assumption about how the displacement varies through the thickness, and
// whose equilibrium equations have been integrated through it.  They belong
// to the same module as the continuum elasticity of SolidMechanics.cpp, and
// are kept in a separate file only because they are long.
//
// All models are written as systems of second-order equations, which is what
// the dual mesh control domain method requires, so every equation takes the
// canonical form -div F + S = 0 and is discretized by the same machinery as
// the scalar problems.  A kernel added without 'variable' is added once per
// variable of the model, and each instance works out which equation it
// represents from the variable it is attached to.
//
// Shear locking is avoided as in the book: the transverse shear terms
// (phi + dw/dx) are evaluated at the centre of the primal element (selective
// reduced integration), which `quadrature = midpoint` with
// `reduced_integration = true` gives.  The physics adds the kernels so.
#include "dualmesh/base/Factory.h"
#include "dualmesh/base/FieldExpression.h"
#include "dualmesh/base/Kernel.h"
#include "dualmesh/base/Problem.h"
#include "dualmesh/fe/ReferenceElement.h"

#include <cmath>

namespace dualmesh
{

namespace
{

/// Shared parameters and variable bookkeeping of the structural models.
class StructuralKernel : public Kernel
{
public:
  /// The variables of the model are named by @p keys; added without 'variable', the kernel is added once per variable, so that it assembles every equation of the model.
  static void modelVariables(InputParameters & p, std::vector<std::string> keys)
  {
    p.setModelVariables(std::move(keys));
    p.addOptional("variable",
                  ParameterKind::Names,
                  std::vector<std::string>{},
                  "The variable (equation) this instance assembles. Default: every variable of "
                  "the model, each with an instance of its own.");
  }

  static InputParameters validParams()
  {
    InputParameters p = Kernel::validParams();
    p.addOptional(
        "youngs_modulus",
        ParameterKind::Function,
        0.0,
        "Young's modulus E of the material: a constant, a function, or an expression of the "
        "fields, the properties and the coordinates, in which z is the position through the "
        "thickness, measured from the mid-surface (a functionally graded section). Given, the "
        "section stiffnesses are the integrals of the material law through the thickness, at "
        "'through_thickness_points' Gauss points, at every integration point. Default: the section "
        "is given by its stiffnesses.");
    p.addOptional("poissons_ratio",
                  ParameterKind::Real,
                  1.0 / 3.0,
                  "Poisson's ratio nu, with -1 < nu < 1/2: in a plate the hoop and cross-coupling "
                  "terms of the stress resultants and, with youngs_modulus, the plate stiffness "
                  "E/(1 - nu^2) and the shear modulus E/(2 (1 + nu)). Default 1/3, the default of "
                  "the library.");
    p.addOptional("thickness",
                  ParameterKind::Real,
                  0.0,
                  "Thickness h of a plate given by its material. Default none.");
    p.setRange("thickness", ParameterRange::Positive);
    p.addOptional(
        "cross_section",
        ParameterKind::RealMap,
        RealMap{},
        "The cross-section of a beam given by its material: {width, height} for a rectangle, "
        "through whose height the material law is integrated, or {area, second_moment} for a "
        "homogeneous section. Default none.");
    p.addOptional("shear_correction",
                  ParameterKind::Real,
                  5.0 / 6.0,
                  "Shear correction factor K_s of a section given by its material. Default 5/6, a "
                  "homogeneous rectangle, for a plate and a rectangular beam section. A beam "
                  "section given by its area and second_moment has no known shape, so a model "
                  "with shear needs it given.");
    p.addOptional("through_thickness_points",
                  ParameterKind::Integer,
                  8L,
                  "Gauss points through the thickness of a section given by its material, 1 to 20. "
                  "Default 8, exact for a modulus that is a polynomial of degree up to 13 in z.");
    p.addOptional(
        "temperature",
        ParameterKind::String,
        std::string(""),
        "The temperature variable of a thermal strain alpha (T - T_0), uniform through "
        "the thickness, which the section integrates into the thermal force N_T and moment "
        "M_T. It needs the section given by its material. Default none.");
    p.addOptional(
        "linear_expansion_coeff",
        ParameterKind::Function,
        0.0,
        "Linear thermal expansion coefficient alpha: a constant, a function or an "
        "expression of the fields and of z, the position through the thickness. Default 0.");
    p.addOptional("stress_free_temperature",
                  ParameterKind::Real,
                  std::monostate{},
                  "Temperature T_0 at which the section is free of thermal strain. Required with "
                  "'temperature', because no value is right for every problem.");
    p.addOptional("extensional_stiffness",
                  ParameterKind::Real,
                  std::monostate{},
                  "Extensional stiffness A = integral of E dA over the cross-section, per unit "
                  "width for plates. For every plate model give the plate stiffness, which "
                  "already contains the factor 1/(1 - nu^2), because the kernel does not add this "
                  "factor. In the mixed models A enters only through the combination Abar = (D A "
                  "- B^2)/D.");
    p.addOptional("coupling_stiffness",
                  ParameterKind::Real,
                  0.0,
                  "Bending-extension coupling stiffness B = integral of E z dA. It is zero "
                  "for a section that is symmetric about its reference surface and non-zero "
                  "for a functionally graded or laminated one, where it couples the axial "
                  "and bending equations. Because z is measured from the reference surface, "
                  "B depends on where that surface was placed when A, B and D were "
                  "computed.");
    p.addOptional("bending_stiffness",
                  ParameterKind::Real,
                  std::monostate{},
                  "Bending stiffness D = integral of E z^2 dA. It must not be zero, because "
                  "the mixed models divide by it: Abar = (D A - B^2)/D, Bbar = B/D, and the "
                  "moment equation carries the term -M/D. For plates give the plate "
                  "stiffness, which already contains 1/(1 - nu^2).");
    p.addOptional("shear_stiffness",
                  ParameterKind::Real,
                  std::monostate{},
                  "Transverse shear stiffness S = K_s integral of G dA, with K_s the shear "
                  "correction factor, 5/6 for a homogeneous rectangular section. It is used "
                  "by beam_Timoshenko_displacement, beam_Timoshenko_mixed, "
                  "circular_plate_first_order and plate_first_order, and ignored by "
                  "beam_Euler-Bernoulli_mixed and circular_plate_classical_mixed, which are "
                  "shear-rigid theories.");
    p.addOptional(
        "foundation_modulus",
        ParameterKind::Real,
        0.0,
        "Winkler elastic foundation modulus, which contributes the source c_f w to the transverse "
        "equation. Its units are force per unit deflection per unit length for a beam and per "
        "unit area for a plate. It must be non-negative, and the default 0 means no foundation.");
    p.setRange("foundation_modulus", ParameterRange::NonNegative);
    p.addOptional(
        "transverse_load",
        ParameterKind::Function,
        0.0,
        "Distributed load acting along the deflection w, in force per unit length for a beam "
        "and force per unit area for a plate, as a constant or the name of a function. It "
        "enters the transverse equation as the source -q, so a positive value pushes the "
        "deflection positive.");
    p.addOptional("axial_load",
                  ParameterKind::Function,
                  0.0,
                  "Distributed load along the axial or radial displacement u, entering that "
                  "equation as the source -f. It is used by the three beam models and by "
                  "both circular plate models, and ignored by plate_first_order, which takes "
                  "its in-plane loads from 'in_plane_load' "
                  "instead.");
    p.addOptional("von_karman",
                  ParameterKind::Boolean,
                  false,
                  "Include the von Karman moderate-rotation nonlinearity: the membrane "
                  "strain gains the term (1/2)(dw/dx)^2 and the transverse flux gains the "
                  "follower term N dw/dx. Turn it on when the deflection is of the order of "
                  "the thickness or larger. It makes the problem nonlinear, so the solve "
                  "needs Newton's method or direct iteration and usually load stepping, and "
                  "it must be set the same way on every instance of the model.");
    p.addOptional("scale_with_load",
                  ParameterKind::Boolean,
                  true,
                  "Multiply this kernel's contribution by the load factor during load stepping, "
                  "which is how the distributed loads are ramped. It defaults to true here, "
                  "unlike the framework default. The weight applies to the whole contribution, "
                  "stiffness terms included, so set it to false on any instance whose internal "
                  "forces must not be ramped.");
    return p;
  }

  /// @p shear: whether the model has transverse shear deformation, and so a shear stiffness.
  StructuralKernel(const InputParameters & p, bool shear)
      : Kernel(p), _B(p.getReal("coupling_stiffness")), _cf(p.getReal("foundation_modulus")),
        _nu(p.getReal("poissons_ratio")), _von_karman(p.getBool("von_karman")),
        _material(p.isSetByUser("youngs_modulus")), _has_shear(shear)
  {
    if (!(_nu > -1.0 && _nu < 0.5))
      throw InputError("'" + name() + "': poissons_ratio must lie in (-1, 0.5).");
    if (_material)
    {
      for (const char * k :
           {"extensional_stiffness", "coupling_stiffness", "bending_stiffness", "shear_stiffness"})
        if (p.isSetByUser(k))
          throw InputError("'" + name() +
                           "': give the section by its material (youngs_modulus) or "
                           "by its stiffnesses, not both.");
      return;
    }
    // A section given by its stiffnesses: every stiffness the model uses is required.
    std::vector<std::string> needed = {"extensional_stiffness", "bending_stiffness"};
    if (_has_shear)
      needed.push_back("shear_stiffness");
    for (const auto & k : needed)
      if (!p.isSet(k))
        throw InputError("'" + name() + "': give '" + k +
                         "', which this model needs when the section is given by its "
                         "stiffnesses, or give the section by its material (youngs_modulus).");
    if (!_has_shear && p.isSetByUser("shear_stiffness"))
      throw InputError("'" + name() +
                       "': this model has no transverse shear deformation, so it "
                       "takes no 'shear_stiffness'.");
    _A = p.getReal("extensional_stiffness");
    _D = p.getReal("bending_stiffness");
    _S = _has_shear ? p.getReal("shear_stiffness") : 0.0;
    if (_D == 0.0)
      throw InputError("'bending_stiffness' must be non-zero.");
    _Abar = (_D * _A - _B * _B) / _D;
    _Bbar = _B / _D;
  }

  void initialSetup(Problem & problem) override
  {
    Kernel::initialSetup(problem);
    _q = getFunction(problem, "transverse_load");
    _f = getFunction(problem, "axial_load");
    const std::string temperature = _params.getString("temperature");
    if (!temperature.empty())
    {
      if (!_material)
        throw InputError("'" + name() +
                         "': a thermal strain needs the section given by its material "
                         "(youngs_modulus), whose law it integrates through the thickness.");
      _T = problem.variableIndex(temperature);
      _alpha.setup(problem, _params, "linear_expansion_coeff", name());
      if (!_params.isSet("stress_free_temperature"))
        throw InputError("'" + name() +
                         "': give 'stress_free_temperature', the temperature at which the "
                         "section is free of thermal strain, with 'temperature'.");
      _T0 = _params.getReal("stress_free_temperature");
    }
    if (!_material)
      return;
    _E.setup(problem, _params, "youngs_modulus", name());
    _kappa = _params.getReal("shear_correction");
    const long n = _params.getInt("through_thickness_points");
    if (n < 1 || n > 20)
      throw InputError("'" + name() + "': through_thickness_points is 1 to 20.");
    const RealMap section = _params.getRealMap("cross_section");
    const auto has = [&](const char * k) { return section.count(k) > 0; };
    if (_plate)
    {
      if (!section.empty())
        throw InputError("'" + name() + "': a plate takes its thickness, not a cross_section.");
      _height = _params.getReal("thickness");
      _width = 1.0;
      if (!(_height > 0.0))
        throw InputError("'" + name() + "': give the thickness of the plate.");
    }
    else if (has("width") && has("height") && section.size() == 2)
    {
      _width = section.at("width");
      _height = section.at("height");
    }
    else if (has("area") && has("second_moment") && section.size() == 2)
    {
      _area = section.at("area");
      _inertia = section.at("second_moment");
      if (_has_shear && !_params.isSetByUser("shear_correction"))
        throw InputError("'" + name() +
                         "': a section given by its area and second_moment has no known shape, so "
                         "give its 'shear_correction' (5/6 for a rectangle, 0.9 for a solid "
                         "circle).");
    }
    else
      throw InputError("'" + name() +
                       "': give the cross_section of the beam as {width, height} "
                       "or {area, second_moment}.");
    QuadratureSpec spec;
    spec.kind = QuadratureSpec::Kind::Gauss;
    spec.points = static_cast<int>(n);
    rule1D(spec, _z, _w);
    for (std::size_t k = 0; k < _z.size(); ++k)
    {
      _z[k] *= 0.5 * _height;
      _w[k] *= 0.5 * _height * _width;
    }
  }

protected:
  /// Which terms an instance contributes (used to integrate the transverse
  /// shear terms with a reduced rule).
  enum class ShearTreatment
  {
    Include,
    Exclude,
    Only
  };
  static ShearTreatment shearTreatment(const std::string & name)
  {
    if (name == "include")
      return ShearTreatment::Include;
    if (name == "exclude")
      return ShearTreatment::Exclude;
    if (name == "only")
      return ShearTreatment::Only;
    throw InputError("'shear_treatment' must be 'include', 'exclude', or 'only'.");
  }

  /// The membrane strain of an axis or a radius: du/dx, with the von Karman term (dw/dx)^2 / 2.
  ADReal axialStrain(const QpContext & ctx, int u, int w) const
  {
    ADReal e = ctx.gradient(u)[0];
    if (_von_karman)
    {
      const ADReal & slope = ctx.gradient(w)[0];
      e += 0.5 * slope * slope;
    }
    return e;
  }

  /// Identify which equation of the model this instance represents.
  int equationFor(Problem & problem, const std::vector<std::string> & parameters)
  {
    for (std::size_t i = 0; i < parameters.size(); ++i)
      if (_params.getString(parameters[i]) == variableName())
        return static_cast<int>(i);
    std::string names;
    for (const auto & p : parameters)
      names += " " + p + "=" + _params.getString(p);
    throw InputError("The variable '" + variableName() + "' of kernel '" + name() +
                     "' is none of the model variables:" + names +
                     ". Add this kernel once for each variable of the model.");
    (void) problem;
  }

  /// The section stiffnesses at an integration point, with Abar = (D A - B^2)/D and Bbar = B/D
  /// of the mixed models.  Given by the material, they are the integrals of its law through the
  /// thickness (per unit width for a plate, whose stiffness carries 1/(1 - nu^2)), so a modulus
  /// that depends on z, on the temperature or on any field enters them exactly.
  struct Section
  {
    ADReal A, B, D, S, Abar, Bbar;
    /// The thermal force and moment, the integrals of the thermal stress through the thickness.
    ADReal NT = 0.0, MT = 0.0;
  };

  Section section(const QpContext & ctx) const
  {
    Section s;
    if (!_material)
    {
      s.A = _A;
      s.B = _B;
      s.D = _D;
      s.S = _S;
      s.Abar = _Abar;
      s.Bbar = _Bbar;
      return s;
    }
    const double plate = _plate ? 1.0 / (1.0 - _nu * _nu) : 1.0;
    const double shear = _kappa / (2.0 * (1.0 + _nu));
    // The thermal stress of a plate is E alpha dT / (1 - nu), equal in both directions; of a beam,
    // E alpha dT.
    const double thermal = _plate ? 1.0 / (1.0 - _nu) : 1.0;
    const ADReal dT = _T >= 0 ? ctx.value(_T) - _T0 : ADReal(0.0);
    if (_area > 0.0)
    {
      const ADReal E = _E.value(ctx);
      s.A = E * _area;
      s.B = 0.0;
      s.D = E * _inertia;
      s.S = shear * E * _area;
      if (_T >= 0)
        s.NT = E * _alpha.value(ctx) * dT * _area;
    }
    else
    {
      s.A = s.B = s.D = s.S = 0.0;
      for (std::size_t k = 0; k < _z.size(); ++k)
      {
        Point x = ctx.x;
        x[2] = _z[k];
        const ADReal E = _E.valueAt(ctx, x) * _w[k];
        s.A += plate * E;
        s.B += plate * E * _z[k];
        s.D += plate * E * _z[k] * _z[k];
        s.S += shear * E;
        if (_T >= 0)
        {
          const ADReal stress = thermal * E * _alpha.valueAt(ctx, x) * dT;
          s.NT += stress;
          s.MT += stress * _z[k];
        }
      }
    }
    s.Abar = (s.D * s.A - s.B * s.B) / s.D;
    s.Bbar = s.B / s.D;
    return s;
  }

  /// The material law at the height @p z above the mid-surface: Young's modulus and the free
  /// thermal strain alpha (T - T_0) there.
  std::pair<double, double> materialAt(const QpContext & ctx, double z) const
  {
    if (!_material)
      throw InputError("'" + name() +
                       "': the stress needs the section given by its material "
                       "(youngs_modulus); a section given by its stiffnesses "
                       "has no material law to evaluate.");
    Point x = ctx.x;
    x[2] = z;
    const double E = _E.valueAt(ctx, x).value();
    const double strain =
        _T >= 0 ? _alpha.valueAt(ctx, x).value() * (ctx.value(_T).value() - _T0) : 0.0;
    return {E, strain};
  }

  /// The stresses (sigma_11, sigma_22, sigma_12) of a plate at a point where the strains are
  /// e = e0 + z k, in plane stress.
  std::vector<double>
  plateStress(const QpContext & ctx, double z, double e11, double e22, double g12) const
  {
    const auto [E, free] = materialAt(ctx, z);
    const double f = E / (1.0 - _nu * _nu);
    const double thermal = E * free / (1.0 - _nu);
    return {f * (e11 + _nu * e22) - thermal,
            f * (e22 + _nu * e11) - thermal,
            E / (2.0 * (1.0 + _nu)) * g12};
  }

  /// The axial stress of a beam whose strain is e0 + z k at the height @p z.
  std::vector<double> beamStress(const QpContext & ctx, double z, double e0, double k) const
  {
    const auto [E, free] = materialAt(ctx, z);
    return {E * (e0 + z * k - free)};
  }

  double _A = 0.0, _B, _D = 0.0, _S = 0.0, _cf;
  double _Abar = 0.0, _Bbar = 0.0;
  double _nu;
  bool _von_karman;
  FunctionPtr _q, _f;
  /// The section is given by its material, and it is a plate section (per unit width).
  bool _material = false, _plate = false;
  /// The model has transverse shear deformation.
  bool _has_shear = false;
  FieldCoefficient _E, _alpha;
  int _T = -1;
  double _T0 = 0.0;
  double _kappa = 5.0 / 6.0, _width = 1.0, _height = 0.0, _area = 0.0, _inertia = 0.0;
  std::vector<double> _z, _w;
};

// ---------------------------------------------------------------------------
// Beam, mixed model in terms of (u, w, M_xx): Euler-Bernoulli (no shear deformation) and
// Timoshenko (the shear flexibility 1/S in the moment equation)
//   -d/dx [ Abar (du/dx + (1/2)(dw/dx)^2) + Bbar M ] = f
//   -d/dx [ dM/dx + N dw/dx ] + c_f w = q
//   -d/dx [ dw/dx - (1/S) dM/dx ] + Bbar (du/dx + (1/2)(dw/dx)^2) - M / D = 0
// ---------------------------------------------------------------------------
class BeamMixed : public StructuralKernel
{
public:
  static InputParameters validParams()
  {
    InputParameters p = StructuralKernel::validParams();
    p.addRequired(
        "axial_displacement",
        ParameterKind::String,
        "Name of the variable holding the axial displacement u of the reference surface.");
    p.addRequired("transverse_displacement",
                  ParameterKind::String,
                  "Name of the variable holding the transverse deflection w, measured positive in "
                  "the direction in which a positive transverse load acts.");
    p.addRequired(
        "bending_moment",
        ParameterKind::String,
        "Name of the variable carrying the bending moment. In a mixed formulation the moment is "
        "an independent unknown of the problem, and it needs its own boundary conditions: leave "
        "it free at a clamped end, where the natural condition is the vanishing slope, and "
        "prescribe it to be zero at a simply supported or free end.");
    modelVariables(p, {"axial_displacement", "transverse_displacement", "bending_moment"});
    return p;
  }
  /// @p shear: the Timoshenko beam, whose moment equation carries the shear flexibility.
  BeamMixed(const InputParameters & p, bool shear) : StructuralKernel(p, shear) {}

  void initialSetup(Problem & problem) override
  {
    StructuralKernel::initialSetup(problem);
    _u = problem.variableIndex(_params.getString("axial_displacement"));
    _w = problem.variableIndex(_params.getString("transverse_displacement"));
    _m = problem.variableIndex(_params.getString("bending_moment"));
    _eq = equationFor(problem, {"axial_displacement", "transverse_displacement", "bending_moment"});
  }

  bool hasFlux() const override { return true; }
  bool hasSource() const override { return true; }

  ADReal axialForce(const QpContext & ctx, const Section & s) const
  {
    return s.Abar * axialStrain(ctx, _u, _w) + s.Bbar * (ctx.value(_m) + s.MT) - s.NT;
  }

  void computeFlux(const QpContext & ctx, ADVector3 & F) const override
  {
    const Section s = section(ctx);
    if (_eq == 0)
      F[0] = axialForce(ctx, s);
    else if (_eq == 1)
    {
      F[0] = ctx.gradient(_m)[0];
      if (_von_karman)
        F[0] += axialForce(ctx, s) * ctx.gradient(_w)[0];
    }
    else
    {
      F[0] = ctx.gradient(_w)[0];
      if (_has_shear)
        F[0] -= ctx.gradient(_m)[0] / s.S;
    }
  }

  ADReal computeSource(const QpContext & ctx) const override
  {
    const Section s = section(ctx);
    if (_eq == 0)
      return ADReal(-_f->value(ctx.x, ctx.time));
    if (_eq == 1)
      return _cf * ctx.value(_w) - _q->value(ctx.x, ctx.time);
    return s.Bbar * axialStrain(ctx, _u, _w) - (ctx.value(_m) + s.MT) / s.D;
  }

  std::vector<double> sectionStress(const QpContext & ctx, double z) const override
  {
    // The curvature follows from the moment: M + M_T = B e0 + D k.
    const Section s = section(ctx);
    const ADReal e0 = axialStrain(ctx, _u, _w);
    const ADReal k = (ctx.value(_m) + s.MT - s.B * e0) / s.D;
    return beamStress(ctx, z, e0.value(), k.value());
  }

private:
  int _u = -1, _w = -1, _m = -1, _eq = 0;
};

class BeamEulerBernoulliMixed : public BeamMixed
{
public:
  static InputParameters validParams()
  {
    InputParameters p = BeamMixed::validParams();
    p.setClassDescription(
        "Mixed Euler-Bernoulli beam model in terms of the axial displacement, the transverse "
        "deflection, and the bending moment. The natural boundary "
        "quantities are the axial force N, the shear force V = dM/dx + N dw/dx, and the "
        "rotation dw/dx.");
    return p;
  }
  explicit BeamEulerBernoulliMixed(const InputParameters & p) : BeamMixed(p, false) {}
};

class BeamTimoshenkoMixed : public BeamMixed
{
public:
  static InputParameters validParams()
  {
    InputParameters p = BeamMixed::validParams();
    p.setClassDescription(
        "Mixed Timoshenko beam model in terms of the axial displacement, the transverse "
        "deflection, and the bending moment. The model is free of "
        "shear locking, and the rotation is recovered as phi = -dw/dx + (1/S) dM/dx.");
    return p;
  }
  explicit BeamTimoshenkoMixed(const InputParameters & p) : BeamMixed(p, true) {}
};

// ---------------------------------------------------------------------------
// Timoshenko beam, displacement model in terms of (u, w, phi_x)
// ---------------------------------------------------------------------------
class BeamTimoshenkoDisplacement : public StructuralKernel
{
public:
  static InputParameters validParams()
  {
    InputParameters p = StructuralKernel::validParams();
    p.setClassDescription(
        "Displacement Timoshenko beam model in terms of the axial displacement, the "
        "transverse deflection, and the rotation of the cross-section. The shear term must be "
        "integrated at the centre of the element to avoid shear locking: add the kernel of the "
        "rotation equation with quadrature = midpoint and reduced_integration = true, as "
        "solid_mechanics with formulation = beam does. Integrated in full, a thin beam locks "
        "(0.07 of the deflection at L/h = 100).");
    p.addRequired(
        "axial_displacement",
        ParameterKind::String,
        "Name of the variable holding the axial displacement u of the reference surface.");
    p.addRequired("transverse_displacement",
                  ParameterKind::String,
                  "Name of the variable holding the transverse deflection w, measured positive in "
                  "the direction in which a positive transverse load acts.");
    p.addRequired("rotation", ParameterKind::String, "Rotation phi_x of the cross-section.");
    modelVariables(p, {"axial_displacement", "transverse_displacement", "rotation"});
    return p;
  }
  explicit BeamTimoshenkoDisplacement(const InputParameters & p) : StructuralKernel(p, true) {}

  void initialSetup(Problem & problem) override
  {
    StructuralKernel::initialSetup(problem);
    _u = problem.variableIndex(_params.getString("axial_displacement"));
    _w = problem.variableIndex(_params.getString("transverse_displacement"));
    _phi = problem.variableIndex(_params.getString("rotation"));
    _eq = equationFor(problem, {"axial_displacement", "transverse_displacement", "rotation"});
  }

  bool hasFlux() const override { return true; }
  bool hasSource() const override { return true; }

  ADReal axialForce(const QpContext & ctx, const Section & s) const
  {
    return s.A * axialStrain(ctx, _u, _w) + s.B * ctx.gradient(_phi)[0] - s.NT;
  }
  ADReal shearForce(const QpContext & ctx, const Section & s) const
  {
    return s.S * (ctx.value(_phi) + ctx.gradient(_w)[0]);
  }

  void computeFlux(const QpContext & ctx, ADVector3 & F) const override
  {
    const Section s = section(ctx);
    if (_eq == 0)
      F[0] = axialForce(ctx, s);
    else if (_eq == 1)
    {
      F[0] = shearForce(ctx, s);
      if (_von_karman)
        F[0] += axialForce(ctx, s) * ctx.gradient(_w)[0];
    }
    else
      F[0] = s.B * axialStrain(ctx, _u, _w) + s.D * ctx.gradient(_phi)[0] - s.MT;
  }

  ADReal computeSource(const QpContext & ctx) const override
  {
    const Section s = section(ctx);
    if (_eq == 0)
      return ADReal(-_f->value(ctx.x, ctx.time));
    if (_eq == 1)
      return _cf * ctx.value(_w) - _q->value(ctx.x, ctx.time);
    return shearForce(ctx, s);
  }

  std::vector<double> sectionStress(const QpContext & ctx, double z) const override
  {
    return beamStress(ctx, z, axialStrain(ctx, _u, _w).value(), ctx.gradient(_phi)[0].value());
  }

private:
  int _u = -1, _w = -1, _phi = -1, _eq = 0;
};

// ---------------------------------------------------------------------------
// Axisymmetric circular plate, first-order shear deformation theory,
// displacement model in terms of (u, w, phi_r).  The integrals carry the
// factor r of the axisymmetric coordinate system.
// ---------------------------------------------------------------------------
class CircularPlateFirstOrder : public StructuralKernel
{
public:
  static InputParameters validParams()
  {
    InputParameters p = StructuralKernel::validParams();
    p.setClassDescription(
        "First-order shear deformation (Mindlin) model of an axisymmetric circular plate in terms "
        "of the radial displacement, the deflection, and the rotation. Use it on a "
        "one-dimensional radial mesh with coordinates = 'axisymmetric'. "
        "The stiffnesses are the plate stiffnesses (they contain 1 / (1 - nu^2)).");
    p.addRequired("radial_displacement",
                  ParameterKind::String,
                  "Name of the variable holding the radial displacement u of the mid-plane.");
    p.addRequired("transverse_displacement",
                  ParameterKind::String,
                  "Name of the variable holding the transverse deflection w, measured positive in "
                  "the direction in which a positive transverse load acts.");
    p.addRequired("rotation",
                  ParameterKind::String,
                  "Name of the variable holding the rotation of the normal in the (r, z) plane. "
                  "The sign convention follows the shear resultant Q_r = S (phi_r + dw/dr), so "
                  "phi_r is positive when it adds to the slope.");
    p.addOptional("shear_treatment",
                  ParameterKind::String,
                  std::string("include"),
                  "Which terms this instance contributes: 'include' (all of them), "
                  "'exclude' (everything but the transverse shear force) or 'only' (the "
                  "transverse shear force alone). Splitting the kernel in two lets the shear "
                  "terms be integrated with a reduced rule, which removes shear locking in "
                  "thin plates.");
    modelVariables(p, {"radial_displacement", "transverse_displacement", "rotation"});
    return p;
  }
  explicit CircularPlateFirstOrder(const InputParameters & p)
      : StructuralKernel(p, true), _shear(shearTreatment(p.getString("shear_treatment")))
  {
    _plate = true;
  }

  void initialSetup(Problem & problem) override
  {
    StructuralKernel::initialSetup(problem);
    if (problem.coordinateSystem() != CoordinateSystem::Axisymmetric)
      throw InputError("circular_plate_first_order needs coordinates = 'axisymmetric'.");
    _u = problem.variableIndex(_params.getString("radial_displacement"));
    _w = problem.variableIndex(_params.getString("transverse_displacement"));
    _phi = problem.variableIndex(_params.getString("rotation"));
    _eq = equationFor(problem, {"radial_displacement", "transverse_displacement", "rotation"});
  }

  bool hasFlux() const override { return true; }
  bool hasSource() const override { return true; }

private:
  struct Resultants
  {
    ADReal n_rr, n_tt, m_rr, m_tt, q_r;
  };

  Resultants resultants(const QpContext & ctx, const Section & s) const
  {
    const double r = ctx.x[0];
    const ADReal & du = ctx.gradient(_u)[0];
    const ADReal & dw = ctx.gradient(_w)[0];
    const ADReal & dphi = ctx.gradient(_phi)[0];
    ADReal e_rr = du;
    if (_von_karman)
      e_rr += 0.5 * dw * dw;
    // At r = 0 the hoop strain u/r is replaced by its limit du/dr.
    const ADReal e_tt = r != 0.0 ? ctx.value(_u) / r : du;
    const ADReal k_rr = dphi;
    const ADReal k_tt = r != 0.0 ? ctx.value(_phi) / r : dphi;
    Resultants out;
    out.n_rr = s.A * (e_rr + _nu * e_tt) + s.B * (k_rr + _nu * k_tt) - s.NT;
    out.n_tt = s.A * (e_tt + _nu * e_rr) + s.B * (k_tt + _nu * k_rr) - s.NT;
    out.m_rr = s.B * (e_rr + _nu * e_tt) + s.D * (k_rr + _nu * k_tt) - s.MT;
    out.m_tt = s.B * (e_tt + _nu * e_rr) + s.D * (k_tt + _nu * k_rr) - s.MT;
    out.q_r = s.S * (ctx.value(_phi) + dw);
    return out;
  }

public:
  void computeFlux(const QpContext & ctx, ADVector3 & F) const override
  {
    const Section s = section(ctx);
    const bool shear_only = _shear == ShearTreatment::Only;
    const bool no_shear = _shear == ShearTreatment::Exclude;
    const Resultants r = resultants(ctx, s);
    if (_eq == 0)
      F[0] = shear_only ? ADReal(0.0) : r.n_rr;
    else if (_eq == 1)
    {
      F[0] = no_shear ? ADReal(0.0) : r.q_r;
      if (_von_karman && !shear_only)
        F[0] += r.n_rr * ctx.gradient(_w)[0];
    }
    else
      F[0] = shear_only ? ADReal(0.0) : r.m_rr;
  }

  ADReal computeSource(const QpContext & ctx) const override
  {
    const Section s = section(ctx);
    const bool shear_only = _shear == ShearTreatment::Only;
    const bool no_shear = _shear == ShearTreatment::Exclude;
    const double r = ctx.x[0];
    const Resultants res = resultants(ctx, s);
    if (_eq == 0)
    {
      if (shear_only)
        return ADReal(0.0);
      return (r != 0.0 ? res.n_tt / r : ADReal(0.0)) - _f->value(ctx.x, ctx.time);
    }
    if (_eq == 1)
    {
      if (shear_only)
        return ADReal(0.0);
      return _cf * ctx.value(_w) - _q->value(ctx.x, ctx.time);
    }
    ADReal out(0.0);
    if (!shear_only)
      out += (r != 0.0 ? res.m_tt / r : ADReal(0.0));
    if (!no_shear)
      out += res.q_r;
    return out;
  }

  std::vector<double> sectionStress(const QpContext & ctx, double z) const override
  {
    const double r = ctx.x[0];
    const double du = ctx.gradient(_u)[0].value();
    const double dw = ctx.gradient(_w)[0].value();
    const double dphi = ctx.gradient(_phi)[0].value();
    double e_rr = du + (_von_karman ? 0.5 * dw * dw : 0.0);
    const double e_tt = r != 0.0 ? ctx.value(_u).value() / r : du;
    const double k_tt = r != 0.0 ? ctx.value(_phi).value() / r : dphi;
    const auto out = plateStress(ctx, z, e_rr + z * dphi, e_tt + z * k_tt, 0.0);
    return {out[0], out[1]};
  }

private:
  ShearTreatment _shear;
  int _u = -1, _w = -1, _phi = -1, _eq = 0;
};

// ---------------------------------------------------------------------------
// Axisymmetric circular plate, classical (Kirchhoff) plate theory, mixed model
// in terms of (u, w, M_rr).  Reddy's DM-CP(M) model, Eqs. (8.2.23)-(8.2.25).
//
// The classical theory gives a fourth-order equation in w, which the dual mesh
// control domain method cannot use, so the bending moment M_rr is carried as a
// third unknown and the system is reduced to three second-order equations.
// With
//
//   ehat0 = du/dr + (1/2)(dw/dr)^2 + nu u/r          (Eq. 8.2.15a)
//   Abar  = (D A - B^2)/D ,  Bbar = B/D              (Eq. 8.2.19)
//   N_rr  = Abar ehat0 + Bbar M_rr                   (Eq. 8.2.18)
//   N_tt  = Abar [nu du/dr + (nu/2)(dw/dr)^2 + u/r]
//           + (1 - nu^2)(B/r)(Bbar u - dw/dr) + nu Bbar M_rr    (Eq. 8.2.22)
//   M_tt  = nu M_rr + (1 - nu^2)(B u - D dw/dr)/r    (Eq. 8.2.21)
//   Q_r   = dM_rr/dr + (M_rr - M_tt)/r               (Eq. 8.2.10)
//   V_r   = Q_r + N_rr dw/dr
//
// the three equations, each in the canonical form -div F + S = 0 with the
// axisymmetric measure 2 pi r dr, are
//
//   u  :  F = N_rr ,  S = N_tt / r - f              (Eq. 8.2.8,  8.2.23)
//   w  :  F = V_r ,   S = c_f w - q                 (Eq. 8.2.9,  8.2.24)
//   M  :  F = dw/dr ,
//         S = (1 - nu)(dw/dr)/r + Bbar ehat0 - M_rr / D         (Eq. 8.2.25)
//
// The third equation is Eq. (8.2.20) rewritten as a divergence: note that
// -(1/r) d/dr (r dw/dr) + (1 - nu)(dw/dr)/r = -(d^2 w/dr^2 + (nu/r) dw/dr),
// which is exactly the effective curvature ehat1_rr of Eq. (8.2.15b).
//
// Note on the printed form of Eq. (8.2.24): the book writes the term
// (1 - nu^2) D_rr Bbar_rr (u/r - (1/r) dw/dr), which would put the factor
// B_rr = D_rr Bbar_rr on both u and dw/dr.  Inverting Eqs. (8.2.14) gives
// M_tt - nu M_rr = (1 - nu^2)(B_rr u - D_rr dw/dr)/r, so the transverse term
// carries D_rr, not B_rr.  The form used here is the one that follows from
// Eq. (8.2.21), which the book states correctly, and it reproduces the
// published results of Tables 8.6.1 and 8.6.2.
// ---------------------------------------------------------------------------
class CircularPlateClassicalMixed : public StructuralKernel
{
public:
  static InputParameters validParams()
  {
    InputParameters p = StructuralKernel::validParams();
    p.setClassDescription(
        "Mixed classical (Kirchhoff) plate model of an axisymmetric circular plate in terms of "
        "the radial displacement, the deflection, and the radial bending moment (Reddy's DM-CP(M) "
        "model). Use it on a one-dimensional radial mesh with coordinates = 'axisymmetric'. "
        "The stiffnesses are the plate stiffnesses (they contain 1 / (1 - "
        "nu^2)). The natural boundary quantities are r N_rr, r V_r, and the slope dw/dr, so a "
        "clamped edge needs no condition on the moment equation and a simply supported edge "
        "prescribes M_rr = 0.");
    p.addRequired("radial_displacement",
                  ParameterKind::String,
                  "Name of the variable holding the radial displacement u of the mid-plane.");
    p.addRequired("transverse_displacement",
                  ParameterKind::String,
                  "Name of the variable holding the transverse deflection w, measured positive in "
                  "the direction in which a positive transverse load acts.");
    p.addRequired(
        "bending_moment",
        ParameterKind::String,
        "Name of the variable carrying the radial bending moment M_rr, the third unknown of the "
        "mixed classical plate model. Prescribe it to be zero at a simply supported edge, and "
        "leave it free at a clamped edge, where the vanishing slope is the natural condition.");
    modelVariables(p, {"radial_displacement", "transverse_displacement", "bending_moment"});
    return p;
  }
  explicit CircularPlateClassicalMixed(const InputParameters & p) : StructuralKernel(p, false)
  {
    _plate = true;
  }

  void initialSetup(Problem & problem) override
  {
    StructuralKernel::initialSetup(problem);
    if (problem.coordinateSystem() != CoordinateSystem::Axisymmetric)
      throw InputError("circular_plate_classical_mixed needs coordinates = 'axisymmetric'.");
    _u = problem.variableIndex(_params.getString("radial_displacement"));
    _w = problem.variableIndex(_params.getString("transverse_displacement"));
    _m = problem.variableIndex(_params.getString("bending_moment"));
    _eq =
        equationFor(problem, {"radial_displacement", "transverse_displacement", "bending_moment"});
  }

  bool hasFlux() const override { return true; }
  bool hasSource() const override { return true; }

private:
  /// The effective radial membrane strain of Eq. (8.2.15a).
  ADReal effectiveStrain(const QpContext & ctx) const
  {
    const double r = ctx.x[0];
    ADReal e = axialStrain(ctx, _u, _w);
    // At r = 0 the hoop strain u/r is replaced by its limit du/dr.
    e += _nu * (r != 0.0 ? ctx.value(_u) / r : ctx.gradient(_u)[0]);
    return e;
  }

  ADReal radialForce(const QpContext & ctx, const Section & s) const
  {
    return s.Abar * effectiveStrain(ctx) + s.Bbar * (ctx.value(_m) + s.MT) - s.NT;
  }

  /// Eq. (8.2.22).
  ADReal hoopForce(const QpContext & ctx, const Section & s) const
  {
    const double r = ctx.x[0];
    const ADReal & du = ctx.gradient(_u)[0];
    const ADReal & dw = ctx.gradient(_w)[0];
    ADReal e = _nu * du;
    if (_von_karman)
      e += 0.5 * _nu * dw * dw;
    e += (r != 0.0 ? ctx.value(_u) / r : du);
    ADReal out = s.Abar * e + _nu * s.Bbar * (ctx.value(_m) + s.MT) - s.NT;
    if (r != 0.0)
      out += (1.0 - _nu * _nu) * s.B / r * (s.Bbar * ctx.value(_u) - dw);
    return out;
  }

  /// Eq. (8.2.10) with M_tt from Eq. (8.2.21).
  ADReal shearForce(const QpContext & ctx, const Section & s) const
  {
    const double r = ctx.x[0];
    ADReal out = ctx.gradient(_m)[0];
    if (r != 0.0)
      out += (1.0 - _nu) * (ctx.value(_m) + s.MT) / r -
             (1.0 - _nu * _nu) / (r * r) * (s.B * ctx.value(_u) - s.D * ctx.gradient(_w)[0]);
    return out;
  }

public:
  void computeFlux(const QpContext & ctx, ADVector3 & F) const override
  {
    const Section s = section(ctx);
    if (_eq == 0)
      F[0] = radialForce(ctx, s);
    else if (_eq == 1)
    {
      F[0] = shearForce(ctx, s);
      if (_von_karman)
        F[0] += radialForce(ctx, s) * ctx.gradient(_w)[0];
    }
    else
      F[0] = ctx.gradient(_w)[0];
  }

  ADReal computeSource(const QpContext & ctx) const override
  {
    const Section s = section(ctx);
    const double r = ctx.x[0];
    if (_eq == 0)
      return (r != 0.0 ? hoopForce(ctx, s) / r : ADReal(0.0)) - _f->value(ctx.x, ctx.time);
    if (_eq == 1)
      return _cf * ctx.value(_w) - _q->value(ctx.x, ctx.time);
    ADReal out = s.Bbar * effectiveStrain(ctx) - (ctx.value(_m) + s.MT) / s.D;
    if (r != 0.0)
      out += (1.0 - _nu) * ctx.gradient(_w)[0] / r;
    return out;
  }

  std::vector<double> sectionStress(const QpContext & ctx, double z) const override
  {
    // The curvatures follow from the moment: M + M_T = B ehat0 + D (k_rr + nu k_tt), with
    // k_tt = -(dw/dr)/r, and k_rr = k_tt on the axis.
    const Section s = section(ctx);
    const double r = ctx.x[0];
    const double du = ctx.gradient(_u)[0].value();
    const double dw = ctx.gradient(_w)[0].value();
    const double combined = ((ctx.value(_m) + s.MT - s.B * effectiveStrain(ctx)) / s.D).value();
    const double k_tt = r != 0.0 ? -dw / r : combined / (1.0 + _nu);
    const double k_rr = combined - _nu * k_tt;
    const double e_rr = du + (_von_karman ? 0.5 * dw * dw : 0.0);
    const double e_tt = r != 0.0 ? ctx.value(_u).value() / r : du;
    const auto out = plateStress(ctx, z, e_rr + z * k_rr, e_tt + z * k_tt, 0.0);
    return {out[0], out[1]};
  }

private:
  int _u = -1, _w = -1, _m = -1, _eq = 0;
};

// ---------------------------------------------------------------------------
// Rectangular plate, first-order shear deformation theory, displacement model
// in terms of (u, v, w, phi_x, phi_y) with the von Karman nonlinearity.
// ---------------------------------------------------------------------------
class PlateFirstOrder : public StructuralKernel
{
public:
  static InputParameters validParams()
  {
    InputParameters p = StructuralKernel::validParams();
    p.setClassDescription(
        "First-order shear deformation (Mindlin) plate model with five variables "
        "(u, v, w, phi_x, phi_y) and the von Karman nonlinearity. "
        "The stiffnesses are the plate stiffnesses A, B, D (they contain 1 / (1 - nu^2)) and "
        "the shear stiffness S. Evaluate the shear terms at the element centre "
        "(reduced_integration = true) for thin plates.");
    p.addRequired("in_plane_displacements",
                  ParameterKind::StringList,
                  "Exactly two variable names, in the order (u along x, v along y). Any other "
                  "count is an error.");
    p.addRequired("transverse_displacement",
                  ParameterKind::String,
                  "Name of the variable holding the transverse deflection w, measured positive in "
                  "the direction in which a positive transverse load acts.");
    p.addRequired(
        "rotations",
        ParameterKind::StringList,
        "Exactly two variable names, in the order (phi_x, phi_y), the rotations of the normal in "
        "the x-z and y-z planes. The sign convention follows the shear resultants Q = S (phi + "
        "grad w), so phi_x is positive when it adds to dw/dx. Any other count is an error.");
    p.addOptional("shear_treatment",
                  ParameterKind::String,
                  std::string("include"),
                  "Which terms this instance contributes: 'include' (all of them), "
                  "'exclude' (everything but the transverse shear forces) or 'only' (the "
                  "transverse shear forces alone). Splitting the kernel in two lets the shear "
                  "terms be integrated with a reduced rule, which removes shear locking in "
                  "thin plates.");
    p.addOptional(
        "in_plane_load",
        ParameterKind::StringList,
        std::vector<std::string>{"0", "0"},
        "Distributed in-plane load (f_x, f_y), in force per unit area, each a number, a registered "
        "function or an expression of x, y, z, t and the fields, entering the equations of the two "
        "in-plane displacements as the sources -f_x and -f_y. It replaces the inherited "
        "'axial_load', which this model ignores. Default zero.");
    modelVariables(p, {"in_plane_displacements", "transverse_displacement", "rotations"});
    return p;
  }
  explicit PlateFirstOrder(const InputParameters & p)
      : StructuralKernel(p, true), _shear(shearTreatment(p.getString("shear_treatment")))
  {
    _plate = true;
  }

  void initialSetup(Problem & problem) override
  {
    StructuralKernel::initialSetup(problem);
    const auto in_plane = _params.getStringList("in_plane_displacements");
    const auto rotations = _params.getStringList("rotations");
    if (in_plane.size() != 2 || rotations.size() != 2)
      throw InputError("'in_plane_displacements' and 'rotations' need two names each.");
    _u = problem.variableIndex(in_plane[0]);
    _v = problem.variableIndex(in_plane[1]);
    _w = problem.variableIndex(_params.getString("transverse_displacement"));
    _px = problem.variableIndex(rotations[0]);
    _py = problem.variableIndex(rotations[1]);
    const auto load = _params.getStringList("in_plane_load");
    if (load.size() != 2)
      throw InputError("'" + this->name() + "': 'in_plane_load' needs two values, (f_x, f_y).");
    _fx.setup(problem, load[0], this->name() + "', in-plane load f_x '");
    _fy.setup(problem, load[1], this->name() + "', in-plane load f_y '");
    const std::string & name = variableName();
    if (name == in_plane[0])
      _eq = 0;
    else if (name == in_plane[1])
      _eq = 1;
    else if (name == _params.getString("transverse_displacement"))
      _eq = 2;
    else if (name == rotations[0])
      _eq = 3;
    else if (name == rotations[1])
      _eq = 4;
    else
      throw InputError("The variable '" + name + "' of kernel '" + this->name() +
                       "' is none of the five plate variables.");
  }

  bool hasFlux() const override { return true; }
  bool hasSource() const override { return _eq >= 2 || _shear != ShearTreatment::Only; }

private:
  struct Resultants
  {
    ADReal n_xx, n_yy, n_xy, m_xx, m_yy, m_xy, q_x, q_y;
  };

  Resultants resultants(const QpContext & ctx, const Section & s) const
  {
    const auto & gu = ctx.gradient(_u);
    const auto & gv = ctx.gradient(_v);
    const auto & gw = ctx.gradient(_w);
    const auto & gpx = ctx.gradient(_px);
    const auto & gpy = ctx.gradient(_py);
    ADReal e_xx = gu[0], e_yy = gv[1], e_xy = gu[1] + gv[0];
    if (_von_karman)
    {
      e_xx += 0.5 * gw[0] * gw[0];
      e_yy += 0.5 * gw[1] * gw[1];
      e_xy += gw[0] * gw[1];
    }
    const ADReal k_xx = gpx[0], k_yy = gpy[1], k_xy = gpx[1] + gpy[0];
    const double shear_factor = 0.5 * (1.0 - _nu);
    Resultants out;
    out.n_xx = s.A * (e_xx + _nu * e_yy) + s.B * (k_xx + _nu * k_yy) - s.NT;
    out.n_yy = s.A * (e_yy + _nu * e_xx) + s.B * (k_yy + _nu * k_xx) - s.NT;
    out.n_xy = shear_factor * (s.A * e_xy + s.B * k_xy);
    out.m_xx = s.B * (e_xx + _nu * e_yy) + s.D * (k_xx + _nu * k_yy) - s.MT;
    out.m_yy = s.B * (e_yy + _nu * e_xx) + s.D * (k_yy + _nu * k_xx) - s.MT;
    out.m_xy = shear_factor * (s.B * e_xy + s.D * k_xy);
    out.q_x = s.S * (ctx.value(_px) + gw[0]);
    out.q_y = s.S * (ctx.value(_py) + gw[1]);
    return out;
  }

public:
  void computeFlux(const QpContext & ctx, ADVector3 & F) const override
  {
    const Section s = section(ctx);
    const Resultants r = resultants(ctx, s);
    if (_shear == ShearTreatment::Only && _eq != 2)
      return; // the shear forces appear only in the transverse equation
    switch (_eq)
    {
    case 0:
      F[0] = r.n_xx;
      F[1] = r.n_xy;
      break;
    case 1:
      F[0] = r.n_xy;
      F[1] = r.n_yy;
      break;
    case 2:
    {
      if (_shear != ShearTreatment::Exclude)
      {
        F[0] = r.q_x;
        F[1] = r.q_y;
      }
      if (_von_karman && _shear != ShearTreatment::Only)
      {
        const auto & gw = ctx.gradient(_w);
        F[0] += r.n_xx * gw[0] + r.n_xy * gw[1];
        F[1] += r.n_xy * gw[0] + r.n_yy * gw[1];
      }
      break;
    }
    case 3:
      F[0] = r.m_xx;
      F[1] = r.m_xy;
      break;
    default:
      F[0] = r.m_xy;
      F[1] = r.m_yy;
      break;
    }
  }

  ADReal computeSource(const QpContext & ctx) const override
  {
    const Section s = section(ctx);
    const bool shear_only = _shear == ShearTreatment::Only;
    switch (_eq)
    {
    case 0:
      return shear_only ? ADReal(0.0) : -_fx.value(ctx);
    case 1:
      return shear_only ? ADReal(0.0) : -_fy.value(ctx);
    case 2:
      return shear_only ? ADReal(0.0) : _cf * ctx.value(_w) - _q->value(ctx.x, ctx.time);
    case 3:
      return _shear == ShearTreatment::Exclude ? ADReal(0.0) : resultants(ctx, s).q_x;
    default:
      return _shear == ShearTreatment::Exclude ? ADReal(0.0) : resultants(ctx, s).q_y;
    }
  }

  std::vector<double> sectionStress(const QpContext & ctx, double z) const override
  {
    const auto & gu = ctx.gradient(_u);
    const auto & gv = ctx.gradient(_v);
    const auto & gw = ctx.gradient(_w);
    const auto & gpx = ctx.gradient(_px);
    const auto & gpy = ctx.gradient(_py);
    double e_xx = gu[0].value(), e_yy = gv[1].value(), e_xy = gu[1].value() + gv[0].value();
    if (_von_karman)
    {
      e_xx += 0.5 * gw[0].value() * gw[0].value();
      e_yy += 0.5 * gw[1].value() * gw[1].value();
      e_xy += gw[0].value() * gw[1].value();
    }
    return plateStress(ctx,
                       z,
                       e_xx + z * gpx[0].value(),
                       e_yy + z * gpy[1].value(),
                       e_xy + z * (gpx[1].value() + gpy[0].value()));
  }

private:
  ShearTreatment _shear;
  int _u = -1, _v = -1, _w = -1, _px = -1, _py = -1, _eq = 0;
  FieldCoefficient _fx, _fy;
};

} // namespace

void
registerStructuralMemberObjects(Factory & f)
{
  const std::string m = "solid_mechanics";
  f.add<BeamEulerBernoulliMixed>("beam_Euler-Bernoulli_mixed", ObjectCategory::Kernel, m);
  f.add<BeamTimoshenkoDisplacement>("beam_Timoshenko_displacement", ObjectCategory::Kernel, m);
  f.add<BeamTimoshenkoMixed>("beam_Timoshenko_mixed", ObjectCategory::Kernel, m);
  f.add<CircularPlateFirstOrder>("circular_plate_first_order", ObjectCategory::Kernel, m);
  f.add<CircularPlateClassicalMixed>("circular_plate_classical_mixed", ObjectCategory::Kernel, m);
  f.add<PlateFirstOrder>("plate_first_order", ObjectCategory::Kernel, m);
}

} // namespace dualmesh
