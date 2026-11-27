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

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include <cmath>

namespace dualmesh
{

/// The coefficients of an incompressible flow at a point, which the mass
/// equation (MassEquation) gives to the block preconditioners of the saddle point
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
  /// The size of the terms whose sum is each component of the flux @p flux at @p ctx, and of
  /// those of the source @p source.  Their round-off is about the machine epsilon times it, which
  /// is the floor of Newton's test (Problem::nonlinearSolve): terms that cancel, as those of a
  /// stress at a small strain do, leave a flux far smaller than its round-off.  The default is the
  /// magnitude of the flux and of the source, for a kernel whose terms do not cancel.
  virtual void fluxTermSize(const QpContext & /*ctx*/, const ADVector3 & flux, double size[3]) const
  {
    for (int c = 0; c < 3; ++c)
      size[c] = std::abs(flux[c].value());
  }
  virtual double sourceTermSize(const QpContext & /*ctx*/, const ADReal & source) const
  {
    return std::abs(source.value());
  }

  /// Time-derivative kernels are not weighted by the time-integration theta.
  virtual bool isTimeKernel() const { return false; }

  /// Properties this kernel needs (names), resolved at setup.
  virtual std::vector<std::string> requiredProperties() const { return {}; }

  /// Whether the term of the kernel belongs to the strong residual of its equation, which a
  /// residual-based stabilisation reads (StrongResidual): a source, and a flux whose divergence
  /// strongResidual gives.  An advection written as a face flux keeps v . grad u as its term.
  virtual bool inStrongResidual() const { return hasSource(); }
  /// The term of the kernel in the strong residual of its equation at the point of @p ctx: its
  /// source by default.
  virtual ADReal strongResidual(const QpContext & ctx) const { return computeSource(ctx); }
  /// Whether strongResidual reads the fields, the position and the time alone, so that a context
  /// of the old fields and the old time gives its term at the old state of a step.  A term that
  /// reads properties does not, since that context keeps the properties of the new state.
  virtual bool oldStateFromFields() const { return false; }
  /// Whether the kernel is a residual-based stabilisation (SUPG, PSPG), whose term vanishes with
  /// the strong residual and so stays out of it.
  virtual bool isStabilization() const { return false; }
  /// Whether the kernel carries its variable with a flow (an advection or its stabilisation).
  /// The automatic quadrature of a storage term then takes the Gauss rule, since a lumped
  /// storage loses phase accuracy in a transport.
  virtual bool transportsVariable() const { return false; }
  /// Whether the kernel evaluates the residual of its equation with the time derivative in it (a
  /// residual-based stabilisation, weighted by QpContext::time_weight).  The theta method gives
  /// that derivative no weight when theta is 0, so an explicit step refuses such a kernel.
  virtual bool residualHasTimeDerivative() const { return false; }
  /// Whether the kernel reads the Laplacian of the variables at its points (QpContext::laplacian),
  /// which quadratic elements give.
  virtual bool needsLaplacian() const { return false; }
};

/// A storage term of a balance law: a time derivative of its variable (TimeDerivative,
/// SecondTimeDerivative).  The setup finds such a kernel by a dynamic_cast.
class StorageKernel : public Kernel
{
public:
  using Kernel::Kernel;
  bool isTimeKernel() const override { return true; }
  /// Whether the term is a second time derivative (inertia), which makes its variable's equation
  /// second order in time: a transient run integrates it by the generalized-alpha method.
  virtual bool isSecondOrder() const { return false; }
  /// The quadrature rule of the term in an eigenvalue study (Problem::eigenvalueStudy): its rule
  /// of every study by default.  A storage term that time stepping lumps takes its consistent
  /// rule there (TimeDerivative).
  virtual const QuadratureSpec & eigenvalueStudyQuadrature() const { return quadrature(); }
};

/// The mass equation of an incompressible flow, attached to the pressure: the constraint equation
/// of the saddle point system.  The block preconditioners of the system and the advective fluxes
/// of the flow find it by a dynamic_cast at setup (Problem::flowConstraint,
/// massConservationKernel).
class MassEquation : public Kernel
{
public:
  using Kernel::Kernel;
  /// The coefficients of the flow at the point of @p context, from which the block
  /// preconditioners of the system (SaddlePointBlocks.h) assemble their operators on the
  /// pressure.
  virtual void flowCoefficients(const QpContext & context,
                                FlowCoefficients & coefficients) const = 0;
  /// The velocity @p w whose flux the equation conserves through the faces of the control
  /// volumes, w = v - tau r_m with its stabilisation.  An advective flux of the heat or the
  /// momentum with w conserves them exactly and sees no source from a discrete divergence.
  virtual void conservedVelocity(const QpContext & context, ADVector3 & w) const = 0;
};

/// A kernel that carries its variable with a velocity, as a volume source or as a face flux of a
/// control volume method (Advection).  The assembly finds such a kernel by a dynamic_cast when it
/// groups the kernels (Problem::Group), so that no point of an element casts.
class AdvectiveKernel : public Kernel
{
public:
  using Kernel::Kernel;
  /// Whether the flux needs the two sides of a face (QpContext::advectedValue).
  virtual bool usesFaceSides() const = 0;
  /// Whether the face flux is the non-conservative v . grad u, m (u_f - u_C) for the control
  /// volume C on each side, rather than the conservative m u_f.  The assembly then evaluates the
  /// flux once for each side, with QpContext::reference_side set, and the flux subtracts
  /// QpContext::referenceValue.  The non-conservative form does not see the discrete divergence
  /// of the velocity, so it keeps the solution independent of a constant added to it and keeps
  /// upwind bounded.
  virtual bool nonConservativeFaceFlux() const = 0;
  /// Whether Newton's method needs a line search for the kernel, which the automatic line search
  /// then turns on: a residual with kinks, where its derivative jumps (the limiter of the limited
  /// advection scheme), and the full step can cycle between the two sides.
  virtual bool needsLineSearch() const = 0;
  /// Whether the face flux needs the gradient at the two sides of a face (the linear upwind and
  /// limited schemes).  DMCDM then reconstructs it at the nodes from the nodes around them
  /// (Problem::buildNodeGradients), and the flux depends on those nodes too.
  virtual bool needsSideGradients() const = 0;
  /// Whether the kernel is the inertia of a flow, rho (v . grad) v_i, which the Stokes start of a
  /// steady flow at rest turns off (Problem::solveSteady).
  virtual bool isInertia() const = 0;
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

  /// A condition that assembles itself as a constraint between two surfaces (mortar contact):
  /// the assembly calls assembleConstraint in place of integrating computeBoundaryFlux, at the
  /// new time only.
  virtual bool isConstraint() const { return false; }
  /// Add the residual at @p U (and its Jacobian to @p triplets, when given), with the state
  /// @p old of the start of the step (null in a steady solve: the undeformed state), and the
  /// size of the terms of each entry to @p reference (AssemblyOptions::reference).
  virtual void assembleConstraint(const Problem & problem,
                                  const Eigen::VectorXd & U,
                                  const Eigen::VectorXd * old,
                                  Eigen::VectorXd & R,
                                  Eigen::VectorXd * reference,
                                  std::vector<Eigen::Triplet<double>> * triplets) const
  {
  }
  /// The unknowns of the constraint's own multipliers (on its surface), which it makes active:
  /// the rest of a multiplier variable, which no kernel acts on, stays inactive and at zero.
  virtual std::vector<Index> multiplierDofs(const Problem & problem) const { return {}; }
  /// The boundaries whose flow (Problem::boundaryOutflow) takes this constraint's forces.
  virtual std::vector<std::string> constraintBoundaries() const { return {}; }
  /// A constraint with a history (the damage of a cohesive zone) keeps it as stateful property
  /// objects keep theirs: an assembly writes the trial history from the start of the step,
  /// commitHistory accepts the trial, and startHistory starts the next attempt from the
  /// committed history or from the trial (Problem::commitState).
  virtual void commitHistory() {}
  virtual void startHistory(bool /*from_committed*/) {}

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
/// u and grad_u, the secondary values in u_other, and the primary outward normal
/// in normal.  Gradients are not available on
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
  const std::vector<std::string> & boundaries() const { return _boundaries; }

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
  bool isLoad() const override { return true; }

protected:
  std::vector<Index> _nodes;
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

/// A constraint that makes the nodes of one boundary copies of those of another (a periodic
/// boundary).  A distributed problem joins the copies in its gather-scatter, from the boundaries
/// and the pairing that this interface gives (DistributedProblem::joinConstraints).
class PeriodicConstraint : public Constraint
{
public:
  using Constraint::Constraint;
  /// The side set of the primary nodes and that of their copies.
  virtual std::string primaryBoundary() const = 0;
  virtual std::string secondaryBoundary() const = 0;
  /// The periodic variables of @p problem.
  virtual std::vector<int> periodicVariables(const Problem & problem) const = 0;
  /// For every point of @p secondary, the index of the point of @p primary that it is a copy of.
  virtual std::vector<Index> match(const std::vector<Point> & primary,
                                   const std::vector<Point> & secondary) const = 0;
};

/// Add the parameter advection_scheme of an advective kernel to @p params.
void addAdvectionSchemeParam(InputParameters & params);
/// Whether the advection scheme named by @p text is a face flux of DMCDM (upwind,
/// linear_upwind, limited).
bool isFaceAdvectionScheme(const std::string & text);

/// The parameter of a residual-based stabilisation (SUPG, PSPG) after Tezduyar (1991),
///
///   tau = [ 4 a^2 / h^2 + 9 (4 d / h^2)^2 + s^2 ]^(-1/2) ,
///
/// with h the element size, a^2 the square @p speed2 of the advective speed, d the diffusivity
/// @p diffusivity and s the rate @p unsteady of the time derivative, 2 / dt with dt the time step,
/// or 0 in a steady solve.  The three terms are the reciprocal time scales of the convection, the
/// diffusion across an element and the unsteadiness.  The factor 3 (9 on the square) gives the
/// diffusive limit h^2 / (12 d) of the optimal parameter of linear elements.  The momentum
/// equations take it per unit density: a = rho |v|, d = mu and s = 2 rho_t / dt.
template <typename Real>
Real
stabilizationTau(double h, const Real & speed2, const Real & diffusivity, double unsteady)
{
  using std::sqrt;
  Real inverse2 = (4.0 / (h * h)) * speed2;
  const Real diffusive = 12.0 * diffusivity / (h * h);
  inverse2 += diffusive * diffusive;
  inverse2 += unsteady * unsteady;
  return 1.0 / sqrt(inverse2);
}

/// The strong residual r of the equation of one variable, which a residual-based stabilisation
/// (SUPG, PSPG) reads: the sum of the terms Kernel::strongResidual of the kernels of the variable
/// that are inStrongResidual().  A term of a time kernel or of a kernel atNewTime() is taken at
/// the new state, and the steady terms as the assembly weights them.  The helper also holds the
/// size h of every element, halved on a quadratic element (elementSize).
class StrongResidual
{
public:
  /// Collect the terms of the equation of the variable named @p variable, for the stabilisation
  /// named @p owner.  It matches the kernels by the name of their variable, since a kernel after
  /// the owner has not resolved its variable yet.  It refuses a kernel with a flux that is
  /// neither in the residual nor a stabilisation, whose term the residual would miss.
  void setup(const Problem & problem, const std::string & variable, const std::string & owner);
  /// r at the point of @p ctx, with the weight @p time_weight of the time terms
  /// (QpContext::time_weight).  @p whole_step: the owner takes r at the new time with the weight
  /// 1, so r is the residual of the discrete step, with the steady terms weighted by theta at the
  /// new state and by 1 - theta at the old state, which a context of the old fields and the old
  /// time gives.  Otherwise the assembly weights the owner as it weights the steady terms, and r
  /// holds them at the state of @p ctx.
  ADReal value(const QpContext & ctx, double time_weight, bool whole_step) const;
  /// The kernels of the residual.
  const std::vector<const Kernel *> & terms() const { return _terms; }
  /// Whether the residual holds a time derivative.
  bool hasTimeTerm() const;
  bool threadSafe() const;
  /// The size h of the element of @p ctx, 1 away from an element.
  double elementSize(const QpContext & ctx) const
  {
    return ctx.element >= 0 ? _h[ctx.element] : 1.0;
  }

private:
  std::vector<const Kernel *> _terms;
  std::string _owner;
  std::vector<double> _h;
};

/// The mass equation of the flow of the problem @p problem, or nullptr for a problem without one
/// (the penalty formulation, a velocity prescribed by conditions).
const MassEquation * massConservationKernel(const Problem & problem);
/// The mass equation of the problem when it tests with the functions of @p variable (an
/// equal-order flow, whose pressure is interpolated like the variable), or nullptr.  Then the
/// mass equation holds for every test function of the variable, and an advection by the velocity
/// it conserves both conserves the variable and carries a constant unchanged.
const MassEquation * conservingMassEquation(const Problem & problem, int variable);

/// The velocity that an advective face flux of a flow carries at the point of @p ctx: the
/// velocity w whose flux the mass equation @p mass conserves, so that the heat and the momentum
/// are conserved exactly and see no source from a discrete divergence of the velocity.  On the
/// boundary the mass equation takes v . n (mass_flux), and so does the flux there.  Without a mass
/// equation it is the velocity @p velocities.
ADVector3 advectingVelocity(const QpContext & ctx,
                            const MassEquation * mass,
                            const std::vector<int> & velocities);

/// The advection scheme named by @p text, checked against the method of @p problem: "central"
/// for both methods (the term as the volume source v . grad u), and the face fluxes "upwind",
/// "linear_upwind" and "limited" for DMCDM.  "SUPG" is refused: the streamline stabilisation of the finite element
/// method is a separate kernel (heat_streamline_stabilization, momentum_stabilization).  @p owner
/// names the kernel in messages.
AdvectionScheme
advectionScheme(const Problem & problem, const std::string & text, const std::string & owner);

} // namespace dualmesh
