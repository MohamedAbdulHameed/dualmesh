// SPDX-License-Identifier: LGPL-2.1-or-later
#include "dualmesh/linalg/SaddlePointSolver.h"

#include "dualmesh/linalg/Amg.h"

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

/// The pressure entries of @p entries, numbered by @p position, as a matrix of size n.
SparseMatrix
pressureMatrix(const std::vector<Triplet> & entries, const std::vector<Index> & position, Index n)
{
  std::vector<Triplet> local;
  local.reserve(entries.size());
  for (const auto & e : entries)
    local.emplace_back(position[e.row()], position[e.col()], e.value());
  SparseMatrix M(n, n);
  M.setFromTriplets(local.begin(), local.end());
  M.makeCompressed();
  return M;
}
} // namespace

SparseMatrix
pressureMassApproximation(const SparseMatrix & matrix, const SaddlePointBlocks & blocks)
{
  // sign (1/mu) M_p plus the symmetric part of the pressure block C of the
  // system itself, which belongs to the Schur complement C - D F^{-1} G.  The
  // rows of the prescribed pressures are rows of the identity in C already,
  // and the mass matrix has no entries there.
  const Index np = static_cast<Index>(blocks.pressure_dofs.size());
  std::vector<Index> position(matrix.rows(), -1);
  for (Index i = 0; i < np; ++i)
    position[blocks.pressure_dofs[i]] = i;
  const SparseMatrix C = submatrix(matrix, blocks.pressure_dofs, blocks.pressure_dofs);
  const SparseMatrix C_transpose = C.transpose();
  SparseMatrix S =
      double(blocks.sign) * pressureMatrix(blocks.mass, position, np) + 0.5 * (C + C_transpose);
  S.makeCompressed();
  return S;
}

struct SaddlePointSolver::Factors
{
  Eigen::SparseLU<SparseMatrix, Eigen::COLAMDOrdering<int>> momentum;
  SparseMatrix momentum_block; // F: rows and columns other_dofs
  SparseMatrix gradient;       // G: rows other_dofs, columns pressure_dofs
  SparseMatrix gradient_transpose;
  // The symmetric matrix factorised for the Schur approximation: (1/mu) M_p
  // (with the symmetric part of C) for pressure_mass, A_p = G^T Q_v^{-1} G + C_1
  // for pcd and lsc.  The prescribed pressure unknowns are rows of the identity.
  Eigen::SimplicialLDLT<SparseMatrix> schur;
  SparseMatrix convection_diffusion;      // F_p (pcd)
  SparseMatrix commutator_stabilization;  // C_2 (lsc)
  Eigen::VectorXd inverse_velocity_mass;  // Q_v^{-1}, on other_dofs
  Eigen::VectorXd inverse_pressure_mass;  // Q_p^{-1}, on pressure_dofs
  std::vector<char> pressure_constrained; // on pressure_dofs
};

SaddlePointSolver::SaddlePointSolver(const SparseMatrix & matrix, const SaddlePointBlocks & blocks)
    : _matrix(matrix), _blocks(blocks), _factors(std::make_unique<Factors>())
{
  Factors & f = *_factors;
  const Index np = static_cast<Index>(blocks.pressure_dofs.size());
  const Index nu = static_cast<Index>(blocks.other_dofs.size());
  f.momentum_block = submatrix(matrix, blocks.other_dofs, blocks.other_dofs);
  f.gradient = submatrix(matrix, blocks.other_dofs, blocks.pressure_dofs);
  f.gradient_transpose = f.gradient.transpose();
  f.momentum.analyzePattern(f.momentum_block);
  f.momentum.factorize(f.momentum_block);
  if (f.momentum.info() != Eigen::Success)
    throw std::runtime_error(
        "dualmesh: the momentum block of the saddle point system could not be factorised (" +
        f.momentum.lastErrorMessage() +
        "). Check that the velocity has enough boundary conditions.");

  std::vector<Index> position(matrix.rows(), -1);
  for (Index i = 0; i < np; ++i)
    position[blocks.pressure_dofs[i]] = i;
  f.pressure_constrained.assign(static_cast<std::size_t>(np), 0);
  std::vector<Triplet> identity;
  for (Index i = 0; i < np; ++i)
    if (blocks.constrained[blocks.pressure_dofs[i]])
    {
      f.pressure_constrained[i] = 1;
      identity.emplace_back(i, i, 1.0);
    }
  SparseMatrix I(np, np);
  I.setFromTriplets(identity.begin(), identity.end());

  SparseMatrix schur;
  if (blocks.approximation == SchurApproximation::PressureMass)
    schur = pressureMassApproximation(matrix, blocks);
  else
  {
    f.inverse_velocity_mass.resize(nu);
    for (Index i = 0; i < nu; ++i)
    {
      const double q = blocks.lumped_mass[blocks.other_dofs[i]];
      f.inverse_velocity_mass[i] = q > 0.0 ? 1.0 / q : 0.0;
    }
    const SparseMatrix scaled = f.inverse_velocity_mass.asDiagonal() * f.gradient;
    schur = SparseMatrix(f.gradient_transpose * scaled) +
            pressureMatrix(blocks.stabilization, position, np) + I;
    if (blocks.approximation == SchurApproximation::ConvectionDiffusion)
    {
      f.convection_diffusion = pressureMatrix(blocks.convection_diffusion, position, np);
      f.inverse_pressure_mass.resize(np);
      for (Index i = 0; i < np; ++i)
      {
        const double q = blocks.lumped_mass[blocks.pressure_dofs[i]];
        f.inverse_pressure_mass[i] = q > 0.0 ? 1.0 / q : 1.0;
      }
    }
    else
      f.commutator_stabilization = pressureMatrix(blocks.commutator_stabilization, position, np);
  }
  f.schur.compute(schur);
  if (f.schur.info() != Eigen::Success)
    throw std::runtime_error(
        "dualmesh: the Schur complement approximation of '" + blocks.pressure_variable +
        "' could not be factorised. Is the viscosity positive everywhere, and is the pressure of "
        "an enclosed flow fixed at a point?");
}

SaddlePointSolver::~SaddlePointSolver() = default;

Eigen::VectorXd
SaddlePointSolver::apply(const Eigen::VectorXd & r) const
{
  const Factors & f = *_factors;
  const auto & other = _blocks.other_dofs;
  const auto & pressure = _blocks.pressure_dofs;
  Eigen::VectorXd r_p(pressure.size()), r_u(other.size());
  for (std::size_t i = 0; i < pressure.size(); ++i)
    r_p[i] = r[pressure[i]];
  for (std::size_t i = 0; i < other.size(); ++i)
    r_u[i] = r[other[i]];
  // Back substitution with the upper block triangle: first the pressure,
  // then the momentum equations with the pressure moved to the right.
  Eigen::VectorXd z_p;
  switch (_blocks.approximation)
  {
  case SchurApproximation::PressureMass:
    z_p = f.schur.solve(r_p);
    break;
  case SchurApproximation::ConvectionDiffusion:
    z_p = double(_blocks.sign) *
          f.schur.solve(f.convection_diffusion * f.inverse_pressure_mass.cwiseProduct(r_p));
    break;
  case SchurApproximation::LeastSquaresCommutator:
  {
    const Eigen::VectorXd y = f.schur.solve(r_p);
    const Eigen::VectorXd t = f.inverse_velocity_mass.cwiseProduct(
        f.momentum_block * f.inverse_velocity_mass.cwiseProduct(f.gradient * y));
    z_p = double(_blocks.sign) *
          f.schur.solve(f.gradient_transpose * t + f.commutator_stabilization * y);
    break;
  }
  }
  // A prescribed pressure is a row of the identity, whose Schur complement is 1.
  for (std::size_t i = 0; i < pressure.size(); ++i)
    if (f.pressure_constrained[i])
      z_p[i] = r_p[i];
  const Eigen::VectorXd z_u = f.momentum.solve(r_u - f.gradient * z_p);
  Eigen::VectorXd z(r.size());
  for (std::size_t i = 0; i < pressure.size(); ++i)
    z[pressure[i]] = z_p[i];
  for (std::size_t i = 0; i < other.size(); ++i)
    z[other[i]] = z_u[i];
  return z;
}

FlexibleGmresResult
flexibleGmres(const SparseMatrix & matrix,
              const Eigen::VectorXd & b,
              Eigen::VectorXd & x,
              const std::function<Eigen::VectorXd(const Eigen::VectorXd &)> & precondition,
              double tolerance,
              int max_iterations,
              int restart)
{
  // Right preconditioning: the Krylov basis V is built from A z_j, where
  // z_j = P^{-1} v_j is kept, and the update is x += Z y.
  FlexibleGmresResult result;
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
    Eigen::VectorXd r = b - matrix * x;
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
      Z.push_back(precondition(V[k]));
      Eigen::VectorXd w = matrix * Z[k];
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
      result.relative_residual = (b - matrix * x).norm() / b_norm;
      if (result.relative_residual <= 10.0 * tolerance)
      {
        result.converged = true;
        return result;
      }
    }
  }
  return result;
}

SaddlePointSolver::Result
SaddlePointSolver::solve(const Eigen::VectorXd & b,
                         Eigen::VectorXd & x,
                         double tolerance,
                         int max_iterations,
                         int restart)
{
  x = Eigen::VectorXd::Zero(b.size());
  return flexibleGmres(
      _matrix,
      b,
      x,
      [this](const Eigen::VectorXd & r) { return apply(r); },
      tolerance,
      max_iterations,
      restart);
}

std::optional<SchurApproximation>
schurApproximation(const std::string & preconditioner)
{
  if (preconditioner == "pressure_mass")
    return SchurApproximation::PressureMass;
  if (preconditioner == "pcd")
    return SchurApproximation::ConvectionDiffusion;
  if (preconditioner == "lsc")
    return SchurApproximation::LeastSquaresCommutator;
  return std::nullopt;
}

struct PhysicsBlockSolver::Block
{
  std::vector<Index> unknowns;
  SmoothedAggregationAmg amg;
};

PhysicsBlockSolver::PhysicsBlockSolver(const SparseMatrix & matrix,
                                       const std::vector<std::vector<Index>> & blocks,
                                       const Eigen::MatrixXd & near_nullspace,
                                       const std::vector<int> & block_sizes)
    : _matrix(matrix)
{
  for (std::size_t k = 0; k < blocks.size(); ++k)
  {
    auto block = std::make_unique<Block>();
    block->unknowns = blocks[k];
    const SparseMatrix A = submatrix(matrix, blocks[k], blocks[k]);
    // The columns of the near null space that do not vanish on the block.
    std::vector<Index> columns;
    for (Index c = 0; c < near_nullspace.cols(); ++c)
      for (Index i : blocks[k])
        if (near_nullspace(i, c) != 0.0)
        {
          columns.push_back(c);
          break;
        }
    Eigen::MatrixXd modes(static_cast<Index>(blocks[k].size()), static_cast<Index>(columns.size()));
    for (std::size_t i = 0; i < blocks[k].size(); ++i)
      for (std::size_t c = 0; c < columns.size(); ++c)
        modes(static_cast<Index>(i), static_cast<Index>(c)) =
            near_nullspace(blocks[k][i], columns[c]);
    AmgOptions options;
    options.block_size = block_sizes[k];
    options.symmetric = false;
    block->amg.setup(A, modes, options);
    _blocks.push_back(std::move(block));
  }
}

PhysicsBlockSolver::~PhysicsBlockSolver() = default;

Eigen::VectorXd
PhysicsBlockSolver::apply(const Eigen::VectorXd & r) const
{
  // Block Gauss-Seidel: the residual of each block with the corrections of the
  // earlier blocks moved to the right, then one V-cycle on its diagonal block.
  Eigen::VectorXd z = Eigen::VectorXd::Zero(r.size());
  for (const auto & block : _blocks)
  {
    const Eigen::VectorXd residual = r - _matrix * z;
    Eigen::VectorXd r_b(static_cast<Index>(block->unknowns.size())), z_b;
    for (std::size_t i = 0; i < block->unknowns.size(); ++i)
      r_b[static_cast<Index>(i)] = residual[block->unknowns[i]];
    block->amg.apply(r_b, z_b);
    for (std::size_t i = 0; i < block->unknowns.size(); ++i)
      z[block->unknowns[i]] = z_b[static_cast<Index>(i)];
  }
  return z;
}

FlexibleGmresResult
PhysicsBlockSolver::solve(const Eigen::VectorXd & b,
                          Eigen::VectorXd & x,
                          double tolerance,
                          int max_iterations,
                          int restart)
{
  x = Eigen::VectorXd::Zero(b.size());
  return flexibleGmres(
      _matrix,
      b,
      x,
      [this](const Eigen::VectorXd & r) { return apply(r); },
      tolerance,
      max_iterations,
      restart);
}

} // namespace dualmesh
