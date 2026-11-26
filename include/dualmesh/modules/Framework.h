// SPDX-License-Identifier: LGPL-2.1-or-later
#pragma once

#include "dualmesh/base/FieldExpression.h"
#include "dualmesh/base/Kernel.h"
#include "dualmesh/base/Property.h"

namespace dualmesh
{

/// The diffusion flux -k p(u) grad u, through the deformed body when a deformation gradient is
/// given: heat conduction, with k the thermal conductivity, and any other diffusion.
class Diffusion : public Kernel
{
public:
  explicit Diffusion(const InputParameters & p);
  static InputParameters validParams();
  void initialSetup(Problem & problem) override;
  bool hasFlux() const override { return true; }
  void computeFlux(const QpContext & ctx, ADVector3 & F) const override;
  /// The diffusivity k p(u), with u as a coefficient.
  ADReal diffusivity(const QpContext & ctx) const;
  /// The term -k p(u) Laplacian(u) of the strong residual, which quadratic elements give: the
  /// gradient of the diffusivity is left out, and a linear element, inside which the Laplacian
  /// vanishes, gives zero.
  bool inStrongResidual() const override { return true; }
  ADReal strongResidual(const QpContext & ctx) const override;

private:
  FieldCoefficient _k;
  std::vector<double> _poly;
  int _F = -1;
};

class AnisotropicDiffusion : public Kernel
{
public:
  explicit AnisotropicDiffusion(const InputParameters & p);
  static InputParameters validParams();
  void initialSetup(Problem & problem) override;
  bool hasFlux() const override { return true; }
  void computeFlux(const QpContext & ctx, ADVector3 & F) const override;

private:
  std::vector<double> _entries;
  std::array<std::array<double, 3>, 3> _K;
};

class Reaction : public Kernel
{
public:
  explicit Reaction(const InputParameters & p);
  static InputParameters validParams();
  void initialSetup(Problem & problem) override;
  bool hasSource() const override { return true; }
  ADReal computeSource(const QpContext & ctx) const override;

private:
  FieldCoefficient _c;
  double _p;
};

class BodyForce : public Kernel
{
public:
  bool isLoad() const override { return true; }
  explicit BodyForce(const InputParameters & p);
  static InputParameters validParams();
  void initialSetup(Problem & problem) override;
  bool hasSource() const override { return true; }
  ADReal computeSource(const QpContext & ctx) const override;
  bool oldStateFromFields() const override { return !_f.readsProperties(); }

private:
  FieldCoefficient _f;
};

/// The storage term c dm/dt of a balance law, as c (m - m_old) / dt over a step, with the
/// storage m an expression (the variable u by default) and c a coefficient (1 by default).
class TimeDerivative : public Kernel
{
public:
  explicit TimeDerivative(const InputParameters & p);
  static InputParameters validParams();
  void initialSetup(Problem & problem) override;
  bool hasSource() const override { return true; }
  bool isTimeKernel() const override { return true; }
  ADReal computeSource(const QpContext & ctx) const override;
  double sourceTermSize(const QpContext & ctx, const ADReal & source) const override;
  /// The coefficient c at the point of @p ctx.
  ADReal coefficient(const QpContext & ctx) const { return _c.value(ctx); }
  /// The consistent rule, which an eigenvalue study takes where time stepping lumps the term.
  const QuadratureSpec & eigenvalueStudyQuadrature() const override { return _consistent; }

private:
  /// The arguments of the storage at the start of the step.
  void startArguments(const QpContext & ctx, ADReal * out) const;
  /// Whether the automatic rule of time stepping lumps the term (TimeDerivative::validParams).
  bool lumped(const Problem & problem) const;

  /// Whether the quadrature is the automatic one, which each setup resolves for the mesh.
  bool _automatic = false;
  /// The rule without lumping: the Gauss rule of the automatic quadrature, or the rule given.
  QuadratureSpec _consistent;
  FieldCoefficient _c;
  /// The storage m, or none for m = u.
  bool _storage = false;
  FieldExpressions _expressions;
  std::shared_ptr<const Property> _state;
};

/// The second time derivative c d2u/dt2 (inertia), which a transient run integrates by the
/// generalized-alpha method.
class SecondTimeDerivative : public Kernel
{
public:
  explicit SecondTimeDerivative(const InputParameters & p);
  static InputParameters validParams();
  void initialSetup(Problem & problem) override;
  bool hasSource() const override { return true; }
  bool isTimeKernel() const override { return true; }
  bool isSecondOrder() const override { return true; }
  ADReal computeSource(const QpContext & ctx) const override;

private:
  FieldCoefficient _c;
};

/// The transport c v . grad u of a variable u carried by a velocity v, with a coefficient c: the
/// heat rho c_p v . grad T that a flow carries, the inertia rho (v . grad) v_i of a flow, and any
/// other advection.  The velocity is prescribed, or the unknown velocity of a flow, whose mass
/// equation the transport then conserves (Hughes and Wells 2005).
class Advection : public Kernel
{
public:
  explicit Advection(const InputParameters & p);
  static InputParameters validParams();
  void initialSetup(Problem & problem) override;
  bool hasFlux() const override { return _flux; }
  bool hasSource() const override { return !_flux; }
  bool usesFaceSides() const override { return _flux && (_face_scheme || !_conservative); }
  bool nonConservativeFaceFlux() const override { return usesFaceSides() && !_conservative; }
  bool needsLineSearch() const override { return _scheme == AdvectionScheme::Limited; }
  bool needsSideGradients() const override
  {
    return _scheme == AdvectionScheme::LinearUpwind || _scheme == AdvectionScheme::Limited;
  }
  bool transportsVariable() const override { return true; }
  /// The inertia of a flow: a transport of a component of its own velocity.
  bool isInertia() const override { return _inertia; }
  /// The non-conservative term c v . grad u, with the velocity v as a coefficient, and not the
  /// velocity that the mass equation conserves, whose residual holds this term.
  bool inStrongResidual() const override { return !_conservative; }
  ADReal strongResidual(const QpContext & ctx) const override;
  bool oldStateFromFields() const override;
  bool advectingNormalVelocity(const QpContext & ctx, double & velocity) const override;
  void computeFlux(const QpContext & ctx, ADVector3 & F) const override;
  ADReal computeSource(const QpContext & ctx) const override;
  /// The coefficient c at the point of @p ctx, with the factor QpContext::inertia of the inertia
  /// of a flow.
  ADReal coefficient(const QpContext & ctx) const;
  /// The velocity v at the point of @p ctx, as a coefficient.
  ADVector3 velocity(const QpContext & ctx) const;

private:
  /// The velocity that the term carries: v, or the velocity w that the mass equation conserves.
  ADVector3 carryingVelocity(const QpContext & ctx) const;

  bool _conservative = false;
  /// Whether the scheme is a face flux of DMCDM (upwind, linear_upwind, limited).
  bool _face_scheme = false;
  /// Whether the term is a flux, or the source c v . grad u.
  bool _flux = false;
  bool _inertia = false;
  AdvectionScheme _scheme = AdvectionScheme::Central;
  FieldCoefficient _c;
  /// A prescribed velocity, or the variables of the velocity of a flow.
  std::array<FieldCoefficient, 3> _v;
  std::vector<int> _velocities;
  /// The mass equation whose conserved velocity the term carries (conservingMassEquation).
  const Kernel * _mass = nullptr;
};

class CoupledForce : public Kernel
{
public:
  explicit CoupledForce(const InputParameters & p);
  static InputParameters validParams();
  void initialSetup(Problem & problem) override;
  bool hasSource() const override { return true; }
  ADReal computeSource(const QpContext & ctx) const override;

private:
  int _v = -1;
  FieldCoefficient _c;
};

class DirichletBC : public NodalBC
{
public:
  explicit DirichletBC(const InputParameters & p);
  static InputParameters validParams();
  void initialSetup(Problem & problem) override;
  double computeValue(const Point & x, double t) const override;

private:
  FunctionPtr _g;
};

/// Prescribed value at the one entity nearest to a point: the usual way to fix
/// the level of a pressure that is otherwise determined only up to a constant.
class PointDirichletBC : public NodalBC
{
public:
  explicit PointDirichletBC(const InputParameters & p);
  static InputParameters validParams();
  void initialSetup(Problem & problem) override;
  double computeValue(const Point & x, double t) const override;

private:
  FunctionPtr _g;
};

class NeumannBC : public IntegratedBC
{
public:
  bool isLoad() const override { return true; }
  explicit NeumannBC(const InputParameters & p);
  static InputParameters validParams();
  void initialSetup(Problem & problem) override;
  ADReal computeBoundaryFlux(const QpContext & ctx) const override;

private:
  FieldCoefficient _q;
  int _F = -1;
};

class RobinBC : public IntegratedBC
{
public:
  explicit RobinBC(const InputParameters & p);
  static InputParameters validParams();
  void initialSetup(Problem & problem) override;
  ADReal computeBoundaryFlux(const QpContext & ctx) const override;

protected:
  FieldCoefficient _h, _uinf, _q;
  int _F = -1;
};

class ConstantProperty : public Property
{
public:
  explicit ConstantProperty(const InputParameters & p);
  static InputParameters validParams();
  void declareProperties(PropertyRegistry & r) override;
  void computeProperties(QpContext & ctx) const override;

private:
  RealMap _values;
  std::vector<int> _ids;
};

/// Periodic boundary condition: every node of the secondary boundary is the node of the primary boundary that a translation maps it onto.
class PeriodicBC : public Constraint
{
public:
  explicit PeriodicBC(const InputParameters & p);
  static InputParameters validParams();
  void initialSetup(Problem & problem) override;

  std::string primaryBoundary() const { return _params.getString("primary"); }
  std::string secondaryBoundary() const { return _params.getString("secondary"); }
  /// The periodic variables: every variable of @p problem unless the parameter 'variable' names some.
  std::vector<int> periodicVariables(const Problem & problem) const;
  /// For every point of @p secondary, the index of the point of @p primary that the translation maps it onto.
  /// The translation is the parameter, or else the vector between the centroids of the two point sets; a point without a match is an InputError.
  std::vector<Index> match(const std::vector<Point> & primary,
                           const std::vector<Point> & secondary) const;
};

} // namespace dualmesh
