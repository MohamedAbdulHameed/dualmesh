// SPDX-License-Identifier: LGPL-2.1-or-later
#include "dualmesh/linalg/SaddlePointSolver.h"

#include <Eigen/SparseCholesky>
#include <Eigen/SparseLU>

#include <cmath>
#include <stdexcept>

namespace dualmesh
{

namespace
{
using SparseMatrix = Eigen::SparseMatrix<double>;
using Triplet = Eigen::Triplet<double>;

/// The submatrix of @p A with the given rows and columns (global indices),
/// numbered in the order of the two lists.
SparseMatrix
submatrix(const SparseMatrix & A,
          const std::vector<Index> & rows,
          const std::vector<Index> & columns)
{
  std::vector<Index> row_position(A.rows(), -1);
  for (std::size_t i = 0; i < rows.size(); ++i)
    row_position[rows[i]] = static_cast<Index>(i);
  std::vector<Triplet> entries;
  for (std::size_t j = 0; j < columns.size(); ++j)
    for (SparseMatrix::InnerIterator it(A, columns[j]); it; ++it)
    {
      const Index i = row_position[it.row()];
      if (i >= 0 && it.value() != 0.0)
        entries.emplace_back(i, static_cast<Index>(j), it.value());
    }
  SparseMatrix S(static_cast<Index>(rows.size()), static_cast<Index>(columns.size()));
  S.setFromTriplets(entries.begin(), entries.end());
  S.makeCompressed();
  return S;
}
} // namespace

struct SaddlePointSolver::Factors
{
  Eigen::SparseLU<SparseMatrix, Eigen::COLAMDOrdering<int>> momentum;
  Eigen::SimplicialLDLT<SparseMatrix> schur;
  SparseMatrix gradient; // the coupling block B^T_u: rows other_dofs, columns pressure_dofs
};

SaddlePointSolver::SaddlePointSolver(const SparseMatrix & matrix, const SaddlePointBlocks & blocks)
    : _matrix(matrix), _blocks(blocks), _factors(std::make_unique<Factors>())
{
  const SparseMatrix A = submatrix(matrix, blocks.other_dofs, blocks.other_dofs);
  _factors->gradient = submatrix(matrix, blocks.other_dofs, blocks.pressure_dofs);
  _factors->momentum.analyzePattern(A);
  _factors->momentum.factorize(A);
  if (_factors->momentum.info() != Eigen::Success)
    throw std::runtime_error(
        "dualmesh: the momentum block of the saddle point system could not be factorised (" +
        _factors->momentum.lastErrorMessage() +
        "). Check that the velocity has enough boundary conditions.");
  _factors->schur.compute(blocks.schur_approximation);
  if (_factors->schur.info() != Eigen::Success)
    throw std::runtime_error("dualmesh: the pressure mass matrix of '" + blocks.pressure_variable +
                             "' could not be factorised; is the viscosity positive everywhere?");
}

SaddlePointSolver::~SaddlePointSolver() = default;

Eigen::VectorXd
SaddlePointSolver::apply(const Eigen::VectorXd & r) const
{
  const auto & other = _blocks.other_dofs;
  const auto & pressure = _blocks.pressure_dofs;
  Eigen::VectorXd r_p(pressure.size()), r_u(other.size());
  for (std::size_t i = 0; i < pressure.size(); ++i)
    r_p[i] = r[pressure[i]];
  for (std::size_t i = 0; i < other.size(); ++i)
    r_u[i] = r[other[i]];
  // Back substitution with the upper block triangle: first the pressure,
  // then the momentum equations with the pressure moved to the right.
  const Eigen::VectorXd z_p = _factors->schur.solve(r_p);
  const Eigen::VectorXd z_u = _factors->momentum.solve(r_u - _factors->gradient * z_p);
  Eigen::VectorXd z(r.size());
  for (std::size_t i = 0; i < pressure.size(); ++i)
    z[pressure[i]] = z_p[i];
  for (std::size_t i = 0; i < other.size(); ++i)
    z[other[i]] = z_u[i];
  return z;
}

SaddlePointSolver::Result
SaddlePointSolver::solve(const Eigen::VectorXd & b,
                         Eigen::VectorXd & x,
                         double tolerance,
                         int max_iterations,
                         int restart)
{
  // Flexible GMRES (Saad, SIAM J. Sci. Comput. 14 (1993) 461-469) with right
  // preconditioning: the Krylov basis V is built from A z_j, where
  // z_j = P^{-1} v_j is kept, and the update is x += Z y.
  Result result;
  const Index n = b.size();
  if (x.size() != n)
    x = Eigen::VectorXd::Zero(n);
  const double b_norm = b.norm();
  if (b_norm == 0.0)
  {
    x.setZero();
    result.converged = true;
    return result;
  }
  restart = std::max(1, restart);
  std::vector<Eigen::VectorXd> V, Z;
  Eigen::MatrixXd H;
  Eigen::VectorXd cs, sn, g;
  while (result.iterations < max_iterations)
  {
    Eigen::VectorXd r = b - _matrix * x;
    double beta = r.norm();
    result.relative_residual = beta / b_norm;
    if (result.relative_residual <= tolerance)
    {
      result.converged = true;
      return result;
    }
    V.assign(1, r / beta);
    Z.clear();
    H = Eigen::MatrixXd::Zero(restart + 1, restart);
    cs = Eigen::VectorXd::Zero(restart);
    sn = Eigen::VectorXd::Zero(restart);
    g = Eigen::VectorXd::Zero(restart + 1);
    g[0] = beta;
    int k = 0;
    for (; k < restart && result.iterations < max_iterations; ++k)
    {
      Z.push_back(apply(V[k]));
      Eigen::VectorXd w = _matrix * Z[k];
      for (int i = 0; i <= k; ++i)
      {
        H(i, k) = w.dot(V[i]);
        w -= H(i, k) * V[i];
      }
      H(k + 1, k) = w.norm();
      if (H(k + 1, k) > 0.0)
        V.push_back(w / H(k + 1, k));
      else
        V.push_back(Eigen::VectorXd::Zero(n));
      // Apply the earlier Givens rotations, then make a new one.
      for (int i = 0; i < k; ++i)
      {
        const double t = cs[i] * H(i, k) + sn[i] * H(i + 1, k);
        H(i + 1, k) = -sn[i] * H(i, k) + cs[i] * H(i + 1, k);
        H(i, k) = t;
      }
      const double denominator = std::hypot(H(k, k), H(k + 1, k));
      cs[k] = denominator > 0.0 ? H(k, k) / denominator : 1.0;
      sn[k] = denominator > 0.0 ? H(k + 1, k) / denominator : 0.0;
      H(k, k) = denominator;
      H(k + 1, k) = 0.0;
      g[k + 1] = -sn[k] * g[k];
      g[k] = cs[k] * g[k];
      ++result.iterations;
      result.relative_residual = std::abs(g[k + 1]) / b_norm;
      if (result.relative_residual <= tolerance)
      {
        ++k;
        break;
      }
    }
    // x += Z y with H(0:k, 0:k) y = g(0:k).
    Eigen::VectorXd y = H.topLeftCorner(k, k).triangularView<Eigen::Upper>().solve(g.head(k));
    for (int i = 0; i < k; ++i)
      x += y[i] * Z[i];
    if (result.relative_residual <= tolerance)
    {
      // The recurrence residual can drift from the true one; confirm.
      result.relative_residual = (b - _matrix * x).norm() / b_norm;
      if (result.relative_residual <= 10.0 * tolerance)
      {
        result.converged = true;
        return result;
      }
    }
  }
  return result;
}

} // namespace dualmesh
