// SPDX-License-Identifier: LGPL-2.1-or-later
//
// Kernels define the governing equation of one variable in the canonical
// conservation form
//
//     R(u) = -div F(u, grad u, x, t) + S(u, grad u, x, t) = 0 .
//
// A kernel supplies the flux F, the source S, or both.  The discretization
// turns the same kernel into
//
//   dual mesh control domain method (for the control domain of node I):
//     R_I = - \oint_{\partial CD_I} F . n ds + \int_{CD_I} S dA
//
//   finite element method (Galerkin, test function psi_I):
//     R_I = \int_\Omega grad psi_I . F dA + \int_\Omega psi_I S dA
//
// Boundary terms (the secondary variables on the domain boundary) are
// supplied by integrated boundary conditions or recovered as reactions.
#pragma once

#include "dualmesh/base/Object.h"
#include "dualmesh/base/QpContext.h"

namespace dualmesh
{

/// The coefficients of an incompressible flow at a point, which the mass
/// conservation kernel gives to the block preconditioners of the saddle point
/// system: rho (dv/dt + v . grad v) - mu laplace v + grad p = f.
struct FlowCoefficients
{
  double viscosity = 0.0;
  /// The density of the convective term, zero for a Stokes flow.
  double density = 0.0;
  /// The density of the time derivative.
  double time_density = 0.0;
  int num_velocities = 0;
  int velocities[3] = {-1, -1, -1};
  /// Whether the mass balance carries the residual-based pressure
  /// stabilisation (equal-order velocity and pressure).
  bool stabilized = false;
};

class Kernel : public ResidualObject
{
public:
  explicit Kernel(const InputParameters & params);
  static InputParameters validParams();

  virtual bool hasFlux() const { return false; }
  virtual bool hasSource() const { return false; }

  /// Flux vector F at the integration point.
  virtual void computeFlux(const QpContext & ctx, ADVector3 & flux) const;
  /// Source term S at the integration point.
  virtual ADReal computeSource(const QpContext & ctx) const;

  /// Time-derivative kernels are not weighted by the time-integration theta.
  virtual bool isTimeKernel() const { return false; }

  /// Properties this kernel needs (names), resolved at setup.
  virtual std::vector<std::string> requiredProperties() const { return {}; }

  /// For the constraint equation of a saddle point problem (the mass equation
  /// of an incompressible flow, attached to the pressure): the coefficients of
  /// the flow at a point, from which the block preconditioners of the system
  /// (SaddlePointSolver.h) assemble their operators on the pressure.  Every
  /// other kernel returns false.
  virtual bool flowCoefficients(const QpContext & /*context*/,
                                FlowCoefficients & /*coefficients*/) const
  {
    return false;
  }
  /// Whether this kernel is the mass conservation equation of a flow, the
  /// one that provides flowCoefficients.
  virtual bool isMassConservation() const { return false; }

  /// For an advective flux of a control volume method: whether it needs the two sides of a face
  /// (QpContext::advectedValue), and the normal velocity at a face point, positive from side 0 to
  /// side 1.  At a face between two blocks an advective flux takes the properties of the upstream
  /// side, where every other flux takes the harmonic mean of the two.
  virtual bool usesFaceSides() const { return false; }
  virtual bool advectingNormalVelocity(const QpContext & /*context*/, double & /*velocity*/) const
  {
    return false;
  }
  /// Whether the face flux is the non-conservative v . grad u, m (u_f - u_C) for the control
  /// volume C on each side, rather than the conservative m u_f.  The assembly then evaluates the
  /// flux once for each side, with QpContext::reference_side set, and the flux subtracts
  /// QpContext::referenceValue.  The non-conservative form does not see the discrete divergence
  /// of the velocity, so it keeps the solution independent of a constant added to it and keeps
  /// upwind bounded.
  virtual bool nonConservativeFaceFlux() const { return false; }
  /// Whether the kernel evaluates the residual of its equation with the time derivative in it (a
  /// residual-based stabilisation, weighted by QpContext::time_weight).  The theta method gives
  /// that derivative no weight when theta is 0, so an explicit step refuses such a kernel.
  virtual bool residualHasTimeDerivative() const { return false; }
  /// Whether the kernel is the inertia of a flow, rho (v . grad) v_i, which the Stokes start of a
  /// steady flow at rest turns off (Problem::solveSteady).
  virtual bool isInertia() const { return false; }
  /// Whether the residual of the kernel has kinks, where its derivative jumps (the limiter of the
  /// limited advection scheme).  Newton's method can cycle between the two sides of a kink, so
  /// the automatic line search turns backtracking on for such a problem.
  virtual bool hasKinks() const { return false; }
  /// Whether the face flux needs the gradient at the two sides of a face (the linear upwind and
  /// limited schemes).  DMCDM then reconstructs it at the nodes from the nodes
  /// around them (Problem::buildNodeGradients), and the flux depends on those nodes too.
  virtual bool needsSideGradients() const { return false; }
  /// Whether the kernel reads the Laplacian of the variables at its points (QpContext::laplacian),
  /// which quadratic elements give.
  virtual bool needsLaplacian() const { return false; }
  /// For the mass equation of a flow: the velocity whose flux it conserves through the faces of
  /// the control volumes, w = v - tau r_m with its stabilisation.  An advective flux of the heat
  /// or the momentum with w conserves them exactly and sees no source from a discrete divergence.
  virtual bool conservedVelocity(const QpContext & /*context*/, ADVector3 & /*w*/) const
  {
    return false;
  }

  /// For a structural member (a plate or a beam): the stress at the height @p z above its
  /// mid-surface, from the material law of its section; empty for every other kernel.
  virtual std::vector<double> sectionStress(const QpContext & /*context*/, double /*z*/) const
  {
    return {};
  }
};

/// Integrated boundary condition: prescribes the outward normal flux
/// q = n . F on a side set.  Its residual contribution is -\int q ds over the
/// part of the boundary that belongs to each control domain.
class IntegratedBC : public ResidualObject
{
public:
  explicit IntegratedBC(const InputParameters & params);
  static InputParameters validParams();

  void initialSetup(Problem & problem) override;
  virtual ADReal computeBoundaryFlux(const QpContext & ctx) const = 0;
  const std::vector<std::string> & boundaries() const { return _boundaries; }

protected:
  std::vector<std::string> _boundaries;
};

/// Interface condition across a gap between two boundaries that do not share
/// nodes, such as the outer surface of a shaft and the inner surface of a
/// sleeve.  It is an integrated boundary condition on the *primary* side set
/// ('boundary'), each of whose integration points is paired, once, with the
/// closest point of the *secondary* side set ('secondary_boundary').  At every
/// paired point the object returns the flux q entering the primary body per
/// unit primary area, computed from the values of the variables on both sides;
/// the secondary body receives -q per unit primary area, at the paired point.
/// The two contributions are integrated with the same weights, so whatever
/// leaves one body enters the other exactly (energy, or momentum, is
/// conserved across the gap).  The primary side is integrated with the rule of
/// the discretisation; the secondary side receives its share through its own
/// test functions at the paired point (finite elements) or the control domain
/// that contains the paired point (the dual mesh).  At an interface point the context holds the primary values in
/// u and grad_u, the secondary values in u_other, the paired point in x_other,
/// and the primary outward normal in normal.  Gradients are not available on
/// the secondary side.
class InterfaceBC : public IntegratedBC
{
public:
  explicit InterfaceBC(const InputParameters & params);
  static InputParameters validParams();

  void initialSetup(Problem & problem) override;
  /// Flux entering the primary body per unit primary area.
  virtual ADReal computeInterfaceFlux(const QpContext & ctx) const = 0;
  ADReal computeBoundaryFlux(const QpContext & ctx) const override
  {
    return computeInterfaceFlux(ctx);
  }
  const std::vector<std::string> & secondaryBoundaries() const { return _secondary; }
  /// The equation on the secondary side that receives -q (by default the
  /// same variable as on the primary side).
  int secondaryVariable() const { return _secondary_var; }
  /// The variables whose values this condition reads; derivatives are carried
  /// for these only, which keeps the automatic differentiation budget small.
  const std::vector<int> & coupledVariables() const { return _coupled; }

protected:
  /// Add a variable to the coupled set (call from initialSetup).
  void addCoupled(int v);

  std::vector<std::string> _secondary;
  int _secondary_var = -1;
  std::vector<int> _coupled;
};

/// Nodal (essential) boundary condition: u = g(x, t) on a node set/side set.
class NodalBC : public ResidualObject
{
public:
  explicit NodalBC(const InputParameters & params);
  static InputParameters validParams();

  void initialSetup(Problem & problem) override;
  virtual double computeValue(const Point & x, double t) const = 0;
  const std::vector<Index> & nodes() const { return _nodes; }

protected:
  std::vector<std::string> _boundaries;
  std::vector<Index> _nodes;
};

/// Concentrated nodal source (point force, point heat source).  Its residual
/// contribution at the node is -P.
class NodalLoad : public ResidualObject
{
public:
  explicit NodalLoad(const InputParameters & params);
  static InputParameters validParams();

  void initialSetup(Problem & problem) override;
  virtual double computeValue(const Point & x, double t) const;
  const std::vector<Index> & nodes() const { return _nodes; }

  /// For a load given by coordinates rather than by a boundary: the entity each
  /// requested point was resolved to, and how far that entity is from the
  /// point.  One entry per requested point, in the order they were given.
  const std::vector<std::pair<Index, double>> & pointNodes() const { return _point_nodes; }
  /// Keep only the requested points whose flag is set, and rebuild nodes().
  /// A distributed run uses this to drop the points that another process
  /// resolved more accurately, because each process only sees its own part of
  /// the mesh and would otherwise apply the load at its own nearest node.
  void keepPoints(const std::vector<char> & keep);

protected:
  std::vector<Index> _nodes;
  std::vector<Index> _boundary_nodes;
  std::vector<std::pair<Index, double>> _point_nodes;
  FunctionPtr _value;
};

/// A condition that makes degrees of freedom copies of others.
/// A dependent degree of freedom takes the value of its primary, and its equation is added to the primary's, so the two act as one unknown whose control domain is the union of theirs.
class Constraint : public Object
{
public:
  explicit Constraint(const InputParameters & params) : Object(params) {}
  static InputParameters validParams() { return Object::validParams(); }
  /// The pairs (dependent degree of freedom, primary degree of freedom), found by initialSetup.
  const std::vector<std::pair<Index, Index>> & pairs() const { return _pairs; }

protected:
  std::vector<std::pair<Index, Index>> _pairs;
};

/// Add the parameter advection_scheme of an advective kernel to @p params.
void addAdvectionSchemeParam(InputParameters & params);

/// The mass equation of the flow of the problem @p problem (the kernel whose
/// conservedVelocity() is defined), or nullptr for a problem without one (the penalty
/// formulation, a velocity prescribed by conditions).
const Kernel * massConservationKernel(const Problem & problem);

/// The velocity that an advective face flux of a flow carries at the point of @p ctx: the
/// velocity w whose flux the mass equation @p mass conserves, so that the heat and the momentum
/// are conserved exactly and see no source from a discrete divergence of the velocity.  On the
/// boundary the mass equation takes v . n (mass_flux), and so does the flux there.  Without a mass
/// equation it is the velocity @p velocities.
ADVector3
advectingVelocity(const QpContext & ctx, const Kernel * mass, const std::vector<int> & velocities);

/// The advection scheme named by @p text, checked against the method of @p problem: "central"
/// for both methods (the term as the volume source v . grad u), and the face fluxes "upwind",
/// "linear_upwind" and "limited" for DMCDM.  "SUPG" is refused: the streamline stabilisation of the finite element
/// method is a separate kernel (heat_streamline_stabilization, momentum_stabilization).  @p owner
/// names the kernel in messages.
AdvectionScheme
advectionScheme(const Problem & problem, const std::string & text, const std::string & owner);

} // namespace dualmesh
