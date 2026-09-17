// SPDX-License-Identifier: LGPL-2.1-or-later
//
// Structural module: beams, axisymmetric circular plates, and rectangular
// plates, including functionally graded (through-thickness graded) sections
// and the von Karman geometric nonlinearity.
//
// All models are written as systems of second-order equations, which is what
// the dual mesh control domain method requires, so every equation takes the
// canonical form -div F + S = 0 and is discretized by the same machinery as
// the scalar problems.  Each kernel is added once per variable of the model
// and works out which equation it represents from the variable it is attached
// to.
//
// Shear locking is avoided exactly as in the book: the transverse shear terms
// (phi + dw/dx) are evaluated at the centre of the primal element (selective
// reduced integration), which is what `reduced_integration = true` does.
#include "dualmesh/base/Factory.h"
#include "dualmesh/base/Kernel.h"
#include "dualmesh/base/Problem.h"

namespace dualmesh
{

namespace
{

/// Shared parameters and variable bookkeeping of the structural models.
class StructuralKernel : public Kernel
{
public:
  static InputParameters validParams()
  {
    InputParameters p = Kernel::validParams();
    p.addOptional("extensional_stiffness",
                  ParameterKind::Real,
                  1.0,
                  "A_xx = integral of E over the cross-section (per unit width for plates).");
    p.addOptional("coupling_stiffness",
                  ParameterKind::Real,
                  0.0,
                  "B_xx = integral of E z; non-zero for functionally graded sections.");
    p.addOptional("bending_stiffness", ParameterKind::Real, 1.0, "D_xx = integral of E z^2.");
    p.addOptional("shear_stiffness",
                  ParameterKind::Real,
                  1.0,
                  "S_xz = K_s integral of G (shear-deformable models only).");
    p.addOptional(
        "foundation_modulus", ParameterKind::Real, 0.0, "Elastic foundation modulus c_f.");
    p.addOptional(
        "transverse_load", ParameterKind::Function, 0.0, "Distributed transverse load q(x, t).");
    p.addOptional("axial_load", ParameterKind::Function, 0.0, "Distributed axial load f(x, t).");
    p.addOptional("von_karman",
                  ParameterKind::Boolean,
                  false,
                  "Include the von Karman nonlinear strain (1/2)(dw/dx)^2.");
    p.addOptional("scale_with_load",
                  ParameterKind::Boolean,
                  true,
                  "Scale the distributed loads with the load factor.");
    return p;
  }

  explicit StructuralKernel(const InputParameters & p)
      : Kernel(p), _A(p.getReal("extensional_stiffness")), _B(p.getReal("coupling_stiffness")),
        _D(p.getReal("bending_stiffness")), _S(p.getReal("shear_stiffness")),
        _cf(p.getReal("foundation_modulus")), _von_karman(p.getBool("von_karman"))
  {
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

  double _A, _B, _D, _S, _cf;
  double _Abar, _Bbar;
  bool _von_karman;
  FunctionPtr _q, _f;
};

// ---------------------------------------------------------------------------
// Euler-Bernoulli beam, mixed model in terms of (u, w, M_xx)
//   -d/dx [ Abar (du/dx + (1/2)(dw/dx)^2) + Bbar M ] = f
//   -d/dx [ dM/dx + N dw/dx ] + c_f w = q
//   -d^2 w/dx^2 + Bbar (du/dx + (1/2)(dw/dx)^2) - M / D = 0
// ---------------------------------------------------------------------------
class BeamEulerBernoulliMixed : public StructuralKernel
{
public:
  static InputParameters validParams()
  {
    InputParameters p = StructuralKernel::validParams();
    p.setClassDescription(
        "Mixed Euler-Bernoulli beam model in terms of the axial displacement, the transverse "
        "deflection, and the bending moment. Add it once per variable. The natural boundary "
        "quantities are the axial force N, the shear force V = dM/dx + N dw/dx, and the "
        "rotation dw/dx.");
    p.addRequired("axial_displacement", ParameterKind::String, "Axial displacement u.");
    p.addRequired("transverse_displacement", ParameterKind::String, "Deflection w.");
    p.addRequired("bending_moment", ParameterKind::String, "Bending moment M_xx.");
    return p;
  }
  explicit BeamEulerBernoulliMixed(const InputParameters & p) : StructuralKernel(p) {}

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

  /// The membrane strain, including the von Karman term.
  ADReal membraneStrain(const QpContext & ctx) const
  {
    ADReal e = ctx.gradient(_u)[0];
    if (_von_karman)
    {
      const ADReal & slope = ctx.gradient(_w)[0];
      e += 0.5 * slope * slope;
    }
    return e;
  }
  ADReal axialForce(const QpContext & ctx) const
  {
    return _Abar * membraneStrain(ctx) + _Bbar * ctx.value(_m);
  }

  void computeFlux(const QpContext & ctx, ADVector3 & F) const override
  {
    if (_eq == 0)
      F[0] = axialForce(ctx);
    else if (_eq == 1)
    {
      F[0] = ctx.gradient(_m)[0];
      if (_von_karman)
        F[0] += axialForce(ctx) * ctx.gradient(_w)[0];
    }
    else
      F[0] = ctx.gradient(_w)[0];
  }

  ADReal computeSource(const QpContext & ctx) const override
  {
    if (_eq == 0)
      return ADReal(-_f->value(ctx.x, ctx.time));
    if (_eq == 1)
      return _cf * ctx.value(_w) - _q->value(ctx.x, ctx.time);
    return _Bbar * membraneStrain(ctx) - ctx.value(_m) / _D;
  }

private:
  int _u = -1, _w = -1, _m = -1, _eq = 0;
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
        "transverse deflection, and the rotation of the cross-section. Add it once per "
        "variable. The shear terms are evaluated at the centre of the primal element to "
        "avoid shear locking.");
    p.addRequired("axial_displacement", ParameterKind::String, "Axial displacement u.");
    p.addRequired("transverse_displacement", ParameterKind::String, "Deflection w.");
    p.addRequired("rotation", ParameterKind::String, "Rotation phi_x of the cross-section.");
    return p;
  }
  explicit BeamTimoshenkoDisplacement(const InputParameters & p) : StructuralKernel(p) {}

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

  ADReal membraneStrain(const QpContext & ctx) const
  {
    ADReal e = ctx.gradient(_u)[0];
    if (_von_karman)
    {
      const ADReal & slope = ctx.gradient(_w)[0];
      e += 0.5 * slope * slope;
    }
    return e;
  }
  ADReal axialForce(const QpContext & ctx) const
  {
    return _A * membraneStrain(ctx) + _B * ctx.gradient(_phi)[0];
  }
  ADReal shearForce(const QpContext & ctx) const
  {
    return _S * (ctx.value(_phi) + ctx.gradient(_w)[0]);
  }

  void computeFlux(const QpContext & ctx, ADVector3 & F) const override
  {
    if (_eq == 0)
      F[0] = axialForce(ctx);
    else if (_eq == 1)
    {
      F[0] = shearForce(ctx);
      if (_von_karman)
        F[0] += axialForce(ctx) * ctx.gradient(_w)[0];
    }
    else
      F[0] = _B * membraneStrain(ctx) + _D * ctx.gradient(_phi)[0];
  }

  ADReal computeSource(const QpContext & ctx) const override
  {
    if (_eq == 0)
      return ADReal(-_f->value(ctx.x, ctx.time));
    if (_eq == 1)
      return _cf * ctx.value(_w) - _q->value(ctx.x, ctx.time);
    return shearForce(ctx);
  }

private:
  int _u = -1, _w = -1, _phi = -1, _eq = 0;
};

// ---------------------------------------------------------------------------
// Timoshenko beam, mixed model in terms of (u, w, M_xx)
// ---------------------------------------------------------------------------
class BeamTimoshenkoMixed : public StructuralKernel
{
public:
  static InputParameters validParams()
  {
    InputParameters p = StructuralKernel::validParams();
    p.setClassDescription(
        "Mixed Timoshenko beam model in terms of the axial displacement, the transverse "
        "deflection, and the bending moment. Add it once per variable. The model is free of "
        "shear locking, and the rotation is recovered as phi = -dw/dx + (1/S) dM/dx.");
    p.addRequired("axial_displacement", ParameterKind::String, "Axial displacement u.");
    p.addRequired("transverse_displacement", ParameterKind::String, "Deflection w.");
    p.addRequired("bending_moment", ParameterKind::String, "Bending moment M_xx.");
    return p;
  }
  explicit BeamTimoshenkoMixed(const InputParameters & p) : StructuralKernel(p) {}

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

  ADReal membraneStrain(const QpContext & ctx) const
  {
    ADReal e = ctx.gradient(_u)[0];
    if (_von_karman)
    {
      const ADReal & slope = ctx.gradient(_w)[0];
      e += 0.5 * slope * slope;
    }
    return e;
  }
  ADReal axialForce(const QpContext & ctx) const
  {
    return _Abar * membraneStrain(ctx) + _Bbar * ctx.value(_m);
  }

  void computeFlux(const QpContext & ctx, ADVector3 & F) const override
  {
    if (_eq == 0)
      F[0] = axialForce(ctx);
    else if (_eq == 1)
    {
      F[0] = ctx.gradient(_m)[0];
      if (_von_karman)
        F[0] += axialForce(ctx) * ctx.gradient(_w)[0];
    }
    else
      F[0] = ctx.gradient(_w)[0] - ctx.gradient(_m)[0] / _S;
  }

  ADReal computeSource(const QpContext & ctx) const override
  {
    if (_eq == 0)
      return ADReal(-_f->value(ctx.x, ctx.time));
    if (_eq == 1)
      return _cf * ctx.value(_w) - _q->value(ctx.x, ctx.time);
    return _Bbar * membraneStrain(ctx) - ctx.value(_m) / _D;
  }

private:
  int _u = -1, _w = -1, _m = -1, _eq = 0;
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
        "First-order shear deformation (Mindlin) model of an axisymmetric circular plate in "
        "terms of the radial displacement, the deflection, and the rotation. Use it on a "
        "one-dimensional radial mesh with coordinates = 'axisymmetric'. Add it once per "
        "variable; the stiffnesses are the plate stiffnesses (they contain 1 / (1 - nu^2)).");
    p.addRequired("radial_displacement", ParameterKind::String, "Radial displacement u.");
    p.addRequired("transverse_displacement", ParameterKind::String, "Deflection w.");
    p.addRequired("rotation", ParameterKind::String, "Rotation phi_r.");
    p.addOptional("poissons_ratio", ParameterKind::Real, 0.3, "Poisson's ratio nu.");
    p.addOptional("shear_treatment",
                  ParameterKind::String,
                  std::string("include"),
                  "Which terms this instance contributes: 'include' (all of them), "
                  "'exclude' (everything but the transverse shear force) or 'only' (the "
                  "transverse shear force alone). Splitting the kernel in two lets the shear "
                  "terms be integrated with a reduced rule, which removes shear locking in "
                  "thin plates.");
    return p;
  }
  explicit CircularPlateFirstOrder(const InputParameters & p)
      : StructuralKernel(p), _nu(p.getReal("poissons_ratio")),
        _shear(shearTreatment(p.getString("shear_treatment")))
  {
  }

  void initialSetup(Problem & problem) override
  {
    StructuralKernel::initialSetup(problem);
    if (problem.coordinateSystem() != CoordinateSystem::Axisymmetric)
      throw InputError("CircularPlateFirstOrder needs coordinates = 'axisymmetric'.");
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

  Resultants resultants(const QpContext & ctx) const
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
    out.n_rr = _A * (e_rr + _nu * e_tt) + _B * (k_rr + _nu * k_tt);
    out.n_tt = _A * (e_tt + _nu * e_rr) + _B * (k_tt + _nu * k_rr);
    out.m_rr = _B * (e_rr + _nu * e_tt) + _D * (k_rr + _nu * k_tt);
    out.m_tt = _B * (e_tt + _nu * e_rr) + _D * (k_tt + _nu * k_rr);
    out.q_r = _S * (ctx.value(_phi) + dw);
    return out;
  }

public:
  void computeFlux(const QpContext & ctx, ADVector3 & F) const override
  {
    const bool shear_only = _shear == ShearTreatment::Only;
    const bool no_shear = _shear == ShearTreatment::Exclude;
    const Resultants r = resultants(ctx);
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
    const bool shear_only = _shear == ShearTreatment::Only;
    const bool no_shear = _shear == ShearTreatment::Exclude;
    const double r = ctx.x[0];
    const Resultants res = resultants(ctx);
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

private:
  double _nu;
  ShearTreatment _shear;
  int _u = -1, _w = -1, _phi = -1, _eq = 0;
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
        "(u, v, w, phi_x, phi_y) and the von Karman nonlinearity. Add it once per variable. "
        "The stiffnesses are the plate stiffnesses A, B, D (they contain 1 / (1 - nu^2)) and "
        "the shear stiffness S. Evaluate the shear terms at the element centre "
        "(reduced_integration = true) for thin plates.");
    p.addRequired("in_plane_displacements",
                  ParameterKind::StringList,
                  "The two in-plane displacements (u, v).");
    p.addRequired("transverse_displacement", ParameterKind::String, "Deflection w.");
    p.addRequired("rotations", ParameterKind::StringList, "The two rotations (phi_x, phi_y).");
    p.addOptional("poissons_ratio", ParameterKind::Real, 0.3, "Poisson's ratio nu.");
    p.addOptional("shear_treatment",
                  ParameterKind::String,
                  std::string("include"),
                  "Which terms this instance contributes: 'include' (all of them), "
                  "'exclude' (everything but the transverse shear forces) or 'only' (the "
                  "transverse shear forces alone). Splitting the kernel in two lets the shear "
                  "terms be integrated with a reduced rule, which removes shear locking in "
                  "thin plates.");
    p.addOptional("in_plane_load_x", ParameterKind::Function, 0.0, "In-plane load f_x.");
    p.addOptional("in_plane_load_y", ParameterKind::Function, 0.0, "In-plane load f_y.");
    return p;
  }
  explicit PlateFirstOrder(const InputParameters & p)
      : StructuralKernel(p), _nu(p.getReal("poissons_ratio")),
        _shear(shearTreatment(p.getString("shear_treatment")))
  {
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
    _fx = getFunction(problem, "in_plane_load_x");
    _fy = getFunction(problem, "in_plane_load_y");
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

  Resultants resultants(const QpContext & ctx) const
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
    out.n_xx = _A * (e_xx + _nu * e_yy) + _B * (k_xx + _nu * k_yy);
    out.n_yy = _A * (e_yy + _nu * e_xx) + _B * (k_yy + _nu * k_xx);
    out.n_xy = shear_factor * (_A * e_xy + _B * k_xy);
    out.m_xx = _B * (e_xx + _nu * e_yy) + _D * (k_xx + _nu * k_yy);
    out.m_yy = _B * (e_yy + _nu * e_xx) + _D * (k_yy + _nu * k_xx);
    out.m_xy = shear_factor * (_B * e_xy + _D * k_xy);
    out.q_x = _S * (ctx.value(_px) + gw[0]);
    out.q_y = _S * (ctx.value(_py) + gw[1]);
    return out;
  }

public:
  void computeFlux(const QpContext & ctx, ADVector3 & F) const override
  {
    const Resultants r = resultants(ctx);
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
    const bool shear_only = _shear == ShearTreatment::Only;
    switch (_eq)
    {
    case 0:
      return shear_only ? ADReal(0.0) : ADReal(-_fx->value(ctx.x, ctx.time));
    case 1:
      return shear_only ? ADReal(0.0) : ADReal(-_fy->value(ctx.x, ctx.time));
    case 2:
      return shear_only ? ADReal(0.0) : _cf * ctx.value(_w) - _q->value(ctx.x, ctx.time);
    case 3:
      return _shear == ShearTreatment::Exclude ? ADReal(0.0) : resultants(ctx).q_x;
    default:
      return _shear == ShearTreatment::Exclude ? ADReal(0.0) : resultants(ctx).q_y;
    }
  }

private:
  double _nu;
  ShearTreatment _shear;
  int _u = -1, _v = -1, _w = -1, _px = -1, _py = -1, _eq = 0;
  FunctionPtr _fx, _fy;
};

} // namespace

void
registerStructuralObjects(Factory & f)
{
  const std::string m = "structural";
  f.add<BeamEulerBernoulliMixed>("BeamEulerBernoulliMixed", ObjectCategory::Kernel, m);
  f.add<BeamTimoshenkoDisplacement>("BeamTimoshenkoDisplacement", ObjectCategory::Kernel, m);
  f.add<BeamTimoshenkoMixed>("BeamTimoshenkoMixed", ObjectCategory::Kernel, m);
  f.add<CircularPlateFirstOrder>("CircularPlateFirstOrder", ObjectCategory::Kernel, m);
  f.add<PlateFirstOrder>("PlateFirstOrder", ObjectCategory::Kernel, m);
}

} // namespace dualmesh
