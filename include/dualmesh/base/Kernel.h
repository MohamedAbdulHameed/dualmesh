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

  /// Material properties this kernel needs (names), resolved at setup.
  virtual std::vector<std::string> requiredProperties() const { return {}; }
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

protected:
  std::vector<Index> _nodes;
  FunctionPtr _value;
};

} // namespace dualmesh
