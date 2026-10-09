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
  /// of an incompressible flow, attached to the pressure): the coefficient c
  /// of the pressure mass matrix (c q, psi) that, with the measured sign,
  /// approximates the Schur complement; 1/mu for the Stokes equations.  The
  /// preconditioner "pressure_mass_schur" uses it (see SaddlePointSolver.h).
  /// Every other kernel returns false.
  virtual bool schurMassCoefficient(const QpContext & /*context*/, double & /*coefficient*/) const
  {
    return false;
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
/// test functions at the paired point (finite elements), the control domain
/// that contains the paired point (dual mesh and vertex-centred finite
/// volumes), or the boundary face that contains it (cell-centred finite
/// volumes).  At an interface point the context holds the primary values in
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

} // namespace dualmesh
