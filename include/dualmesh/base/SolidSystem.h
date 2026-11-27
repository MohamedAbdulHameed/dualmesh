// SPDX-License-Identifier: LGPL-2.1-or-later
//
// The solid system of a problem: what the linear solvers and the singularity check of a Problem
// take from the solids it holds.  The solid mechanics module implements it and registers it
// (Factory::setSolidSystem), and every Problem holds it (Problem::solidSystem), whether it has a
// solid or not.
#pragma once

#include <Eigen/Core>

#include <vector>

namespace dualmesh
{

class Problem;

class SolidSystem
{
public:
  virtual ~SolidSystem() = default;
  /// The displacement variables of every elastic property object of @p problem, in component
  /// order.
  virtual std::vector<std::vector<int>> displacementGroups(const Problem & problem) const = 0;
  /// The near null space of the system of @p problem for the algebraic multigrid preconditioner:
  /// for every group of displacementGroups, the rigid body translations and (in Cartesian
  /// coordinates) rotations, and the constant vector of every other variable.  One row per
  /// degree of freedom.
  virtual Eigen::MatrixXd nearNullspace(const Problem & problem) const = 0;
};

} // namespace dualmesh
