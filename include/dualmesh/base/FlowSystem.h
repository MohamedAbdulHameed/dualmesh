// SPDX-License-Identifier: LGPL-2.1-or-later
//
// The flow system of a problem: what the solvers of a Problem take from the flow it holds.  The
// fluid module implements it and registers it (Factory::setFlowSystem), and every Problem makes
// its own (Problem::flowSystem), whether it holds a flow or not: without one, it does nothing.
#pragma once

#include "dualmesh/linalg/SaddlePointBlocks.h"

#include <Eigen/Sparse>

#include <functional>
#include <optional>
#include <vector>

namespace dualmesh
{

class Problem;

class FlowSystem
{
public:
  virtual ~FlowSystem() = default;

  /// Whether @p problem has the inertia of a flow and its velocity is at rest: zero at every
  /// degree of freedom that no condition prescribes, on every process.  A steady solve then starts
  /// from the Stokes solution (Problem::solveSteady).
  virtual bool atRest(const Problem & problem) const = 0;

  /// The blocks of the preconditioner of a flow for the linear system with matrix @p jacobian,
  /// with the Schur complement approximated by @p approximation (see SaddlePointBlocks.h): the
  /// pressure is the variable of the one kernel that provides the flow coefficients
  /// (Problem::flowConstraint), the operators on the pressure are assembled from them at the
  /// current solution, and the sign of the Schur complement is measured from the coupling blocks
  /// of @p jacobian.  The caller has set @p problem up (Problem::initialize), as the assembly of
  /// @p jacobian does.
  virtual SaddlePointBlocks saddlePointBlocks(const Problem & problem,
                                              const Eigen::SparseMatrix<double> & jacobian,
                                              SchurApproximation approximation) const = 0;

  // ---- the level of the pressure of an enclosed flow ----------------------------------------
  //
  // The pressure of an incompressible flow whose boundary prescribes no pressure is determined
  // only up to a constant.  The flow system holds one unknown of each such pressure at its value
  // during a solve, and shifts the pressure to a mean of zero after it.

  /// Forget the held unknowns, for a problem set up again (Problem::initialize).
  virtual void resetPressureLevels() = 0;
  /// The first time after a reset: find each pressure whose level the Jacobian @p jacobian leaves
  /// free, and hold one of its unknowns, the same on every process.  Returns whether one was held.
  virtual bool holdPressureLevels(const Problem & problem,
                                  const Eigen::SparseMatrix<double> & jacobian) = 0;
  /// The unknowns held (Problem::constrainedDofs).
  virtual std::vector<Index> heldPressureDofs() const = 0;
  /// Take the size of the mass residual at the start of a solve (Problem::lastResidual) as the
  /// scale of the boundary flux.
  virtual void notePressureLevelFlux(const Problem & problem) = 0;
  /// The residual at a state: @p residual(U, R, keep_last) writes into R the residual at U, and
  /// with keep_last leaves the last residual of the problem (Problem::lastResidual) as it was.
  using ResidualAt =
      std::function<void(const Eigen::VectorXd &, Eigen::VectorXd &, bool keep_last)>;
  /// After a converged solve, whose residual is @p R: shift each held pressure of the solution of
  /// @p problem to a mean of zero, take @p R again at the shifted state, and throw an InputError
  /// when the level of the pressure changes the equations, so that the problem needs it as data.
  virtual void
  settlePressureLevels(Problem & problem, Eigen::VectorXd & R, const ResidualAt & residual) = 0;

  // ---- the start of a transient run ---------------------------------------------------------

  /// Add the projection of the initial velocity of a flow onto a field without divergence
  /// (projectInitialVelocity of Projection.h), which returns the size of its correction.
  virtual void addStartProjection(std::function<double(Eigen::VectorXd &)> project) = 0;
  /// Project the initial velocity of every flow of @p problem (addStartProjection) onto a field
  /// without divergence that keeps the values prescribed at the current time, and return the
  /// largest correction, or nothing for a problem without a flow.  A transient run does it before
  /// its first step, unless it restarts from a checkpoint or starts from the state that the last
  /// transient run of the problem reached.
  virtual std::optional<double> projectInitialVelocity(Problem & problem) = 0;
};

} // namespace dualmesh
