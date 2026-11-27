// SPDX-License-Identifier: LGPL-2.1-or-later
//
// The pieces that the projection time integration (Projection.cpp) and the projection of the
// initial velocity (ProjectionStart.cpp) share: the blocks of an operator, the kernels and
// conditions that they hold for themselves, and the rows of a linear system over some variables.
#pragma once

#include "dualmesh/base/Factory.h"
#include "dualmesh/core/InputParameters.h"
#include "dualmesh/modules/Projection.h"

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace dualmesh
{
namespace projection
{

/// The part of @p A whose rows and columns are of the variables that @p rows and @p columns mark.
inline SparseMatrix
block(const Problem & problem,
      const SparseMatrix & A,
      const std::vector<char> & rows,
      const std::vector<char> & columns)
{
  const int nv = problem.numVariables();
  std::vector<Eigen::Triplet<double>> entries;
  for (Index col = 0; col < A.outerSize(); ++col)
    for (SparseMatrix::InnerIterator it(A, col); it; ++it)
      if (rows[static_cast<std::size_t>(it.row() % nv)] &&
          columns[static_cast<std::size_t>(col % nv)])
        entries.emplace_back(it.row(), col, it.value());
  SparseMatrix out(A.rows(), A.cols());
  out.setFromTriplets(entries.begin(), entries.end());
  return out;
}

/// A kernel or a boundary condition that only the projection holds, set up with @p problem.
template <typename T>
std::shared_ptr<T>
privateObject(Problem & problem,
              const std::string & type,
              const std::string & name,
              const std::vector<std::pair<std::string, ParameterValue>> & values)
{
  const auto & factory = Factory::instance();
  InputParameters parameters = factory.validParams(type);
  for (const auto & [key, value] : values)
    parameters.set(key, value);
  auto object = std::dynamic_pointer_cast<T>(factory.create(type, name, parameters));
  object->initialSetup(problem);
  return object;
}

} // namespace projection

/// The rows of a linear system over some variables of a problem, in the global numbering of its nodes (Parallel::globalRows): the row of the k-th variable at a node is width G + k, with G the global number of the node.
/// The equation of a fixed unknown (a prescribed value, a dependent or an inactive unknown) is a row of the identity, and its value moves to the right-hand side of the others.
struct ProjectionSolver::Rows
{
  Rows(const Problem & problem_, const std::vector<int> & variables_, std::vector<char> fixed_)
      : problem(problem_), variables(variables_), fixed(std::move(fixed_))
  {
    nv = problem.numVariables();
    width = static_cast<int>(variables.size());
    slot.assign(static_cast<std::size_t>(nv), -1);
    for (int k = 0; k < width; ++k)
      slot[static_cast<std::size_t>(variables[k])] = k;
    const GlobalRows rows = problem.parallel().globalRows(problem);
    across = rows.distributed;
    const Index nodes = problem.mesh().numNodes();
    node.assign(static_cast<std::size_t>(nodes), -1);
    owned.assign(static_cast<std::size_t>(nodes), 0);
    for (Index i = 0; i < nodes; ++i)
    {
      const Index g = rows.global_dof[problem.dof(i, 0)];
      node[i] = g < 0 ? -1 : g / nv;
      owned[i] = rows.owned_dof[problem.dof(i, 0)];
    }
    first = rows.first_row / nv * width;
    num_owned = rows.num_owned / nv * width;
    num_global = rows.global_size / nv * width;
  }

  /// The row of the unknown @p d of the problem, or -1 when it is not in the system.
  Index row(Index d) const
  {
    const int k = slot[static_cast<std::size_t>(d % nv)];
    const Index g = node[static_cast<std::size_t>(d / nv)];
    return k < 0 || g < 0 ? -1 : width * g + k;
  }
  /// Call @p visit with every unknown of the system that this process owns and its row.
  template <typename Visit> void forEachOwned(Visit && visit) const
  {
    for (Index i = 0; i < static_cast<Index>(node.size()); ++i)
      if (owned[i] && node[i] >= 0)
        for (int k = 0; k < width; ++k)
          visit(problem.dof(i, variables[k]), width * node[i] + k);
  }
  /// The entries of the matrix @p A of the unknowns of the problem, in the order of its entries, with the rows and columns of the fixed unknowns replaced by the identity, which keeps a symmetric matrix symmetric.
  std::vector<petsc::GlobalEntry> entries(const SparseMatrix & A) const
  {
    std::vector<petsc::GlobalEntry> out;
    out.reserve(static_cast<std::size_t>(A.nonZeros() + num_owned));
    for (Index col = 0; col < A.outerSize(); ++col)
      for (SparseMatrix::InnerIterator it(A, col); it; ++it)
      {
        const Index r = row(it.row()), c = row(col);
        if (r < 0 || c < 0 || fixed[it.row()] || fixed[col])
          continue;
        out.push_back({r, c, it.value()});
      }
    forEachOwned(
        [&](Index d, Index r)
        {
          if (fixed[d])
            out.push_back({r, r, 1.0});
        });
    return out;
  }
  /// A solver of the matrix @p A in these rows, with the PETSc @p options, the relative
  /// @p tolerance, whether @p A is @p symmetric, and the number of earlier solutions
  /// @p projection_vectors whose best combination starts each solve.
  std::unique_ptr<petsc::LinearSolver> makeSolver(const SparseMatrix & A,
                                                  const std::string & options,
                                                  double tolerance,
                                                  bool symmetric,
                                                  int projection_vectors = 0) const
  {
    petsc::Settings settings;
    settings.options = options;
    settings.relative_tolerance = tolerance;
    settings.projection_vectors = projection_vectors;
    settings.symmetric = symmetric;
    return std::make_unique<petsc::LinearSolver>(
        across, num_global, first, num_owned, entries(A), settings);
  }
  /// Solve A x = b for the unknowns of the system: @p rhs is this process's part of b (the
  /// contributions of its elements), @p values holds the fixed values and a first guess
  /// elsewhere, and receives the solution, with the values of the owners and of the dependent
  /// unknowns.  With @p remove_mean the mean of b over the unknowns of the problem is removed
  /// first, which makes a Neumann problem solvable when its discretization leaves a remainder.
  int solve(const petsc::LinearSolver & solver,
            const SparseMatrix & A,
            Vector rhs,
            Vector & values,
            bool remove_mean) const
  {
    Vector prescribed = Vector::Zero(values.size());
    forEachFixed([&](Index d) { prescribed[d] = values[d]; });
    rhs -= A * prescribed;
    problem.foldDependents(rhs, nullptr);
    problem.parallel().addAtNodes(rhs.data(), nv);
    if (remove_mean)
    {
      const std::vector<char> constrained = problem.constrainedDofs();
      double total = 0.0, count = 0.0;
      forEachOwned(
          [&](Index d, Index)
          {
            if (!constrained[d])
            {
              total += rhs[d];
              count += 1.0;
            }
          });
      const double mean = problem.parallel().sum(total) / problem.parallel().sum(count);
      for (Index d = 0; d < rhs.size(); ++d)
        if (row(d) >= 0)
          rhs[d] -= mean;
    }
    // A dependent or an inactive unknown is no unknown of the system: its row of the identity
    // holds zero, so that it does not enter the norm of b, which the tolerance is relative to, and
    // its value follows the solve.
    const auto & active = problem.activeDofs();
    Vector b(num_owned), guess(num_owned);
    forEachOwned(
        [&](Index d, Index r)
        {
          const bool none = problem.isDependent(d) || (!active.empty() && !active[d]);
          b[r - first] = none ? 0.0 : fixed[d] ? values[d] : rhs[d];
          guess[r - first] = none ? 0.0 : values[d];
        });
    petsc::Result result;
    const Vector x = solver.solve(b, &guess, result);
    if (problem.parallel().any(!result.converged))
      throw std::runtime_error(
          "dualmesh: a linear solve of the projection time integration did not "
          "converge (" +
          result.reason + ", " + std::to_string(result.iterations) + " iterations).");
    forEachOwned([&](Index d, Index r) { values[d] = x[r - first]; });
    problem.parallel().copyFromOwners(values.data(), nv);
    problem.updateDependentDofs(values);
    return result.iterations;
  }
  /// Call @p visit with every fixed unknown of the system.
  template <typename Visit> void forEachFixed(Visit && visit) const
  {
    for (Index d = 0; d < static_cast<Index>(fixed.size()); ++d)
      if (fixed[d] && row(d) >= 0)
        visit(d);
  }

  const Problem & problem;
  std::vector<int> variables;
  int nv = 0, width = 0;
  /// The place of every variable of the problem in the system, or -1.
  std::vector<int> slot;
  /// The global number of every local node (-1: one that no process numbers), and whether this process owns it.
  std::vector<Index> node;
  std::vector<char> owned;
  /// One flag per unknown of the problem.
  std::vector<char> fixed;
  Index first = 0, num_owned = 0, num_global = 0;
  bool across = false;
};

} // namespace dualmesh
