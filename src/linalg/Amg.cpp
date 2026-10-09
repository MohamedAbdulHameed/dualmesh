// SPDX-License-Identifier: LGPL-2.1-or-later
#include "dualmesh/linalg/Amg.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>

namespace dualmesh
{

namespace
{
using RowMatrix = SmoothedAggregationAmg::RowMatrix;
using Eigen::MatrixXd;
using Eigen::VectorXd;

/// The largest eigenvalue of D^-1 A, estimated by the Lanczos process that
/// the conjugate gradient method with the Jacobi preconditioner carries out
/// (Saad, Iterative Methods for Sparse Linear Systems, 2nd ed., 2003, section
/// 6.7.3): the coefficients of 12 iterations form a tridiagonal matrix whose
/// largest eigenvalue converges to that of D^-1 A from below, much faster
/// than the power method.  D^-1 A is similar to the symmetric D^-1/2 A D^-1/2
/// when A is symmetric, so its eigenvalues are real.  For a nonsymmetric A the
/// power method is used.
double
largestEigenvalue(const RowMatrix & A, const VectorXd & inverse_diagonal, bool symmetric)
{
  const Eigen::Index n = A.rows();
  std::mt19937 generator(2718281);
  std::uniform_real_distribution<double> uniform(-1.0, 1.0);
  VectorXd b(n);
  for (Eigen::Index i = 0; i < n; ++i)
    b[i] = uniform(generator);
  if (!symmetric)
  {
    VectorXd x = b / b.norm();
    double lambda = 1.0;
    for (int k = 0; k < 30; ++k)
    {
      VectorXd y = inverse_diagonal.cwiseProduct(A * x);
      const double norm = y.norm();
      if (!(norm > 0.0))
        return 1.0;
      lambda = norm;
      x = y / norm;
    }
    return lambda;
  }
  const int steps = static_cast<int>(std::min<Eigen::Index>(12, n));
  std::vector<double> alphas, betas;
  VectorXd x = VectorXd::Zero(n);
  VectorXd r = b;
  VectorXd z = inverse_diagonal.cwiseProduct(r);
  VectorXd p = z;
  double rz = r.dot(z);
  for (int k = 0; k < steps; ++k)
  {
    const VectorXd q = A * p;
    const double pq = p.dot(q);
    if (!(pq > 0.0) || !(rz > 0.0))
      break;
    const double alpha = rz / pq;
    r.noalias() -= alpha * q;
    z = inverse_diagonal.cwiseProduct(r);
    const double rz_next = r.dot(z);
    const double beta = rz_next / rz;
    alphas.push_back(alpha);
    betas.push_back(beta);
    p = z + beta * p;
    rz = rz_next;
    if (!(rz > 0.0))
      break;
  }
  const int m = static_cast<int>(alphas.size());
  if (m == 0)
    return 1.0;
  // The Lanczos tridiagonal matrix from the CG coefficients.
  MatrixXd T = MatrixXd::Zero(m, m);
  for (int k = 0; k < m; ++k)
  {
    T(k, k) = 1.0 / alphas[k] + (k > 0 ? betas[k - 1] / alphas[k - 1] : 0.0);
    if (k + 1 < m)
    {
      T(k, k + 1) = std::sqrt(betas[k]) / alphas[k];
      T(k + 1, k) = T(k, k + 1);
    }
  }
  Eigen::SelfAdjointEigenSolver<MatrixXd> eigen(T, Eigen::EigenvaluesOnly);
  return eigen.eigenvalues().maxCoeff();
}

/// The product C = A B of two row-major sparse matrices, row by row with a
/// dense accumulator (Gustavson, ACM Trans. Math. Softw. 4 (1978) 250-269),
/// the rows shared among the OpenMP threads.  Entries that cancel exactly are
/// dropped.
RowMatrix
multiply(const RowMatrix & A, const RowMatrix & B)
{
  const int rows = static_cast<int>(A.rows());
  const int cols = static_cast<int>(B.cols());
  std::vector<std::vector<int>> row_columns(rows);
  std::vector<std::vector<double>> row_values(rows);
#pragma omp parallel
  {
    std::vector<double> accumulator(cols, 0.0);
    std::vector<int> marker(cols, -1);
    std::vector<int> pattern;
#pragma omp for schedule(dynamic, 256)
    for (int i = 0; i < rows; ++i)
    {
      pattern.clear();
      for (RowMatrix::InnerIterator a(A, i); a; ++a)
        for (RowMatrix::InnerIterator b(B, a.col()); b; ++b)
        {
          const int j = static_cast<int>(b.col());
          if (marker[j] != i)
          {
            marker[j] = i;
            accumulator[j] = 0.0;
            pattern.push_back(j);
          }
          accumulator[j] += a.value() * b.value();
        }
      std::sort(pattern.begin(), pattern.end());
      auto & c = row_columns[i];
      auto & v = row_values[i];
      c.reserve(pattern.size());
      v.reserve(pattern.size());
      for (int j : pattern)
        if (accumulator[j] != 0.0)
        {
          c.push_back(j);
          v.push_back(accumulator[j]);
        }
    }
  }
  RowMatrix C(rows, cols);
  std::size_t total = 0;
  for (const auto & c : row_columns)
    total += c.size();
  C.resizeNonZeros(static_cast<Eigen::Index>(total));
  int * outer = C.outerIndexPtr();
  outer[0] = 0;
  for (int i = 0; i < rows; ++i)
    outer[i + 1] = outer[i] + static_cast<int>(row_columns[i].size());
#pragma omp parallel for schedule(static)
  for (int i = 0; i < rows; ++i)
  {
    std::copy(row_columns[i].begin(), row_columns[i].end(), C.innerIndexPtr() + outer[i]);
    std::copy(row_values[i].begin(), row_values[i].end(), C.valuePtr() + outer[i]);
  }
  return C;
}

/// Group the nodes of a level into aggregates (Vanek, Mandel and Brezina
/// 1996, section 4).  Returns the aggregate of every node, -1 for a node that
/// has no strong neighbour (e.g., a node whose unknowns are all prescribed),
/// and the number of aggregates.
std::vector<int>
aggregate(const RowMatrix & A, int block_size, double threshold, int & num_aggregates)
{
  const int n = static_cast<int>(A.rows());
  const int nodes = n / block_size;
  // The Frobenius norm of every node block, row by row.
  std::vector<double> diagonal(nodes, 0.0);
  std::vector<std::vector<std::pair<int, double>>> neighbours(nodes);
  std::vector<double> accumulated(nodes, 0.0);
  std::vector<int> touched;
  for (int node = 0; node < nodes; ++node)
  {
    touched.clear();
    for (int r = node * block_size; r < (node + 1) * block_size; ++r)
      for (RowMatrix::InnerIterator it(A, r); it; ++it)
      {
        const int other = it.col() / block_size;
        if (accumulated[other] == 0.0)
          touched.push_back(other);
        accumulated[other] += it.value() * it.value();
      }
    for (int other : touched)
    {
      if (other == node)
        diagonal[node] = accumulated[other];
      else
        neighbours[node].emplace_back(other, accumulated[other]);
      accumulated[other] = 0.0;
    }
  }
  // Keep the strong connections (squared norms are compared).
  const double t2 = threshold * threshold;
  std::vector<std::vector<std::pair<int, double>>> strong(nodes);
  for (int node = 0; node < nodes; ++node)
    for (const auto & [other, value] : neighbours[node])
      if (value > 0.0 && value > t2 * std::sqrt(diagonal[node] * diagonal[other]))
        strong[node].emplace_back(other, value);

  std::vector<int> owner(nodes, -1);
  int count = 0;
  // Pass 1: a node whose strong neighbours are all free starts an aggregate
  // made of itself and those neighbours.
  for (int node = 0; node < nodes; ++node)
  {
    if (owner[node] != -1 || strong[node].empty())
      continue;
    bool free = true;
    for (const auto & connection : strong[node])
      if (owner[connection.first] != -1)
      {
        free = false;
        break;
      }
    if (!free)
      continue;
    owner[node] = count;
    for (const auto & connection : strong[node])
      owner[connection.first] = count;
    ++count;
  }
  // Pass 2: a remaining node joins the aggregate of pass 1 to which it is most
  // strongly connected.
  std::vector<int> joined = owner;
  for (int node = 0; node < nodes; ++node)
  {
    if (owner[node] != -1)
      continue;
    double best = 0.0;
    for (const auto & [other, value] : strong[node])
      if (owner[other] >= 0 && value > best)
      {
        best = value;
        joined[node] = owner[other];
      }
  }
  owner = joined;
  // Pass 3: the nodes still free form aggregates with their free neighbours.
  for (int node = 0; node < nodes; ++node)
  {
    if (owner[node] != -1 || strong[node].empty())
      continue;
    owner[node] = count;
    for (const auto & connection : strong[node])
      if (owner[connection.first] == -1)
        owner[connection.first] = count;
    ++count;
  }
  num_aggregates = count;
  return owner;
}
} // namespace

void
SmoothedAggregationAmg::setup(const Eigen::SparseMatrix<double> & A,
                              const MatrixXd & near_nullspace,
                              const AmgOptions & options)
{
  if (A.rows() != A.cols())
    throw std::invalid_argument("SmoothedAggregationAmg: the matrix must be square.");
  if (near_nullspace.rows() != A.rows() || near_nullspace.cols() < 1)
    throw std::invalid_argument("SmoothedAggregationAmg: the near null space needs one row per "
                                "unknown and at least one column.");
  if (options.block_size < 1 || A.rows() % options.block_size != 0)
    throw std::invalid_argument(
        "SmoothedAggregationAmg: the number of unknowns must be a multiple of the block size.");
  _options = options;
  _levels.clear();
  Level fine;
  fine.A = RowMatrix(A);
  fine.A.makeCompressed();
  fine.block_size = options.block_size;
  _levels.push_back(std::move(fine));
  MatrixXd B = near_nullspace;
  const int k = static_cast<int>(B.cols());

  while (true)
  {
    Level & level = _levels.back();
    const int n = static_cast<int>(level.A.rows());
    level.inverse_diagonal.resize(n);
    const VectorXd diagonal = level.A.diagonal();
    for (int i = 0; i < n; ++i)
      level.inverse_diagonal[i] = diagonal[i] != 0.0 ? 1.0 / diagonal[i] : 0.0;
    level.lambda_max = largestEigenvalue(level.A, level.inverse_diagonal, options.symmetric);
    if (n <= options.coarse_size || static_cast<int>(_levels.size()) >= options.max_levels)
      break;

    const int bs = level.block_size;
    int num_aggregates = 0;
    const std::vector<int> owner =
        aggregate(level.A, bs, options.strength_threshold, num_aggregates);
    if (num_aggregates == 0 || static_cast<long>(num_aggregates) * k >= static_cast<long>(0.8 * n))
      break; // the level would not coarsen

    // The tentative prolongator: on every aggregate, an orthonormal basis of
    // the near null space restricted to its unknowns (a thin QR
    // factorisation), whose triangular factor is the coarse near null space.
    std::vector<std::vector<int>> members(num_aggregates);
    for (int node = 0; node < n / bs; ++node)
      if (owner[node] >= 0)
        members[owner[node]].push_back(node);
    std::vector<Eigen::Triplet<double>> entries;
    entries.reserve(static_cast<std::size_t>(n) * k);
    MatrixXd coarse_B = MatrixXd::Zero(static_cast<Eigen::Index>(num_aggregates) * k, k);
    for (int a = 0; a < num_aggregates; ++a)
    {
      const int m = static_cast<int>(members[a].size()) * bs;
      MatrixXd local(m, k);
      std::vector<int> rows(m);
      for (std::size_t j = 0; j < members[a].size(); ++j)
        for (int c = 0; c < bs; ++c)
        {
          const int row = members[a][j] * bs + c;
          rows[j * bs + c] = row;
          local.row(static_cast<Eigen::Index>(j) * bs + c) = B.row(row);
        }
      Eigen::HouseholderQR<MatrixXd> qr(local);
      const int rank = std::min(m, k);
      const MatrixXd Q = qr.householderQ() * MatrixXd::Identity(m, rank);
      const MatrixXd R = qr.matrixQR().topRows(rank).triangularView<Eigen::Upper>();
      for (int r = 0; r < m; ++r)
        for (int c = 0; c < rank; ++c)
          if (Q(r, c) != 0.0)
            entries.emplace_back(rows[r], a * k + c, Q(r, c));
      coarse_B.block(static_cast<Eigen::Index>(a) * k, 0, rank, k) = R;
    }
    RowMatrix T(n, static_cast<Eigen::Index>(num_aggregates) * k);
    T.setFromTriplets(entries.begin(), entries.end());

    // The smoothed prolongator P = (I - omega D^-1 A) T, and the Galerkin
    // coarse operator P^T A P.
    const double omega = options.prolongator_damping / level.lambda_max;
    RowMatrix AT = multiply(level.A, T);
    RowMatrix P = T - omega * RowMatrix(level.inverse_diagonal.asDiagonal() * AT);
    P.prune(0.0);
    RowMatrix R = P.transpose();
    RowMatrix coarse = multiply(R, multiply(level.A, P));
    // A coarse unknown with no basis function (an aggregate too small to hold
    // every near null space vector) has an empty row and column; a unit
    // diagonal decouples it.
    std::vector<Eigen::Triplet<double>> unit;
    const VectorXd coarse_diagonal = coarse.diagonal();
    for (Eigen::Index i = 0; i < coarse.rows(); ++i)
      if (coarse_diagonal[i] == 0.0)
        unit.emplace_back(static_cast<int>(i), static_cast<int>(i), 1.0);
    if (!unit.empty())
    {
      RowMatrix D(coarse.rows(), coarse.cols());
      D.setFromTriplets(unit.begin(), unit.end());
      coarse += D;
    }
    coarse.makeCompressed();
    level.P = std::move(P);
    level.R = std::move(R);
    Level next;
    next.A = std::move(coarse);
    next.block_size = k;
    B = std::move(coarse_B);
    _levels.push_back(std::move(next)); // invalidates the reference 'level'
  }

  // The coarsest level is factorised exactly.
  const Eigen::SparseMatrix<double> coarsest(_levels.back().A);
  _coarse_cholesky.compute(coarsest);
  _coarse_uses_lu = _coarse_cholesky.info() != Eigen::Success ||
                    (_coarse_cholesky.vectorD().array() <= 0.0).any();
  if (_coarse_uses_lu)
  {
    _coarse_lu.compute(coarsest);
    if (_coarse_lu.info() != Eigen::Success)
      throw std::runtime_error(
          "SmoothedAggregationAmg: the coarsest operator could not be factorised.");
  }
}

void
SmoothedAggregationAmg::smooth(const Level & level, const VectorXd & b, VectorXd & x) const
{
  // Chebyshev iteration on D^-1 A over [lambda_min, lambda_max] (Saad, Iterative
  // Methods for Sparse Linear Systems, 2nd ed., 2003, algorithm 12.1).
  const double upper = 1.1 * level.lambda_max;
  const double lower = level.lambda_max / _options.smoother_ratio;
  const double theta = 0.5 * (upper + lower);
  const double delta = 0.5 * (upper - lower);
  const double sigma = theta / delta;
  double rho = 1.0 / sigma;
  VectorXd r = b - level.A * x;
  VectorXd d = level.inverse_diagonal.cwiseProduct(r) / theta;
  x += d;
  for (int k = 1; k < _options.smoother_degree; ++k)
  {
    r.noalias() -= level.A * d;
    const double rho_next = 1.0 / (2.0 * sigma - rho);
    d = (rho_next * rho) * d + (2.0 * rho_next / delta) * level.inverse_diagonal.cwiseProduct(r);
    x += d;
    rho = rho_next;
  }
}

void
SmoothedAggregationAmg::cycle(std::size_t index, const VectorXd & b, VectorXd & x) const
{
  if (index + 1 == _levels.size())
  {
    x = _coarse_uses_lu ? VectorXd(_coarse_lu.solve(b)) : VectorXd(_coarse_cholesky.solve(b));
    return;
  }
  const Level & level = _levels[index];
  x = VectorXd::Zero(b.size());
  smooth(level, b, x);
  const VectorXd residual = b - level.A * x;
  const VectorXd coarse_residual = level.R * residual;
  VectorXd coarse_correction;
  cycle(index + 1, coarse_residual, coarse_correction);
  x.noalias() += level.P * coarse_correction;
  smooth(level, b, x);
}

void
SmoothedAggregationAmg::apply(const VectorXd & r, VectorXd & z) const
{
  cycle(0, r, z);
}

double
SmoothedAggregationAmg::operatorComplexity() const
{
  if (_levels.empty())
    return 0.0;
  double total = 0.0;
  for (const auto & level : _levels)
    total += static_cast<double>(level.A.nonZeros());
  return total / static_cast<double>(_levels.front().A.nonZeros());
}

std::string
SmoothedAggregationAmg::summary() const
{
  std::ostringstream out;
  for (std::size_t i = 0; i < _levels.size(); ++i)
    out << "    level " << i << ": " << _levels[i].A.rows() << " unknowns, "
        << _levels[i].A.nonZeros() << " nonzeros, block size " << _levels[i].block_size << "\n";
  out << "    operator complexity " << std::fixed << std::setprecision(2) << operatorComplexity()
      << "\n";
  return out.str();
}

IterativeResult
preconditionedConjugateGradient(const RowMatrix & A,
                                const VectorXd & b,
                                const SmoothedAggregationAmg & amg,
                                double tolerance,
                                int max_iterations,
                                VectorXd & x)
{
  IterativeResult result;
  const double b_norm = b.norm();
  if (x.size() != b.size())
    x = VectorXd::Zero(b.size());
  if (b_norm == 0.0)
  {
    x.setZero();
    result.converged = true;
    return result;
  }
  VectorXd r = b - A * x;
  VectorXd z;
  amg.apply(r, z);
  VectorXd p = z;
  double rz = r.dot(z);
  double best = r.norm();
  int since_best = 0;
  double last_true = std::numeric_limits<double>::infinity();
  for (int it = 1; it <= max_iterations; ++it)
  {
    const VectorXd q = A * p;
    const double pq = p.dot(q);
    if (!(pq > 0.0))
    {
      // The curvature p.Ap has lost its sign, which happens when the residual
      // has reached the level of the round-off in A and b.
      result.stagnated = true;
      break;
    }
    const double alpha = rz / pq;
    x.noalias() += alpha * p;
    r.noalias() -= alpha * q;
    result.iterations = it;
    const double r_norm = r.norm();
    result.relative_residual = r_norm / b_norm;
    if (result.relative_residual <= tolerance)
      break;
    if (r_norm < 0.5 * best)
    {
      best = r_norm;
      since_best = 0;
    }
    else if (++since_best >= 25)
    {
      result.stagnated = true; // no halving of the residual in 25 iterations
      break;
    }
    // The updated residual keeps falling after the true residual b - A x has
    // reached the round-off of the system.  Once it is small, the true residual
    // is computed every 5 iterations, and the iteration stops when it no
    // longer halves between two checks.
    if (result.relative_residual < 1e-6 && it % 5 == 0)
    {
      const double true_residual = (b - A * x).norm() / b_norm;
      if (true_residual > 0.5 * last_true)
      {
        result.stagnated = true;
        break;
      }
      last_true = true_residual;
    }
    amg.apply(r, z);
    const double rz_next = r.dot(z);
    p = z + (rz_next / rz) * p;
    rz = rz_next;
  }
  // The true residual, which round-off can separate from the recursively
  // updated one.  When the updated residual has met the tolerance and the
  // true one has not, the difference is the round-off of evaluating b - A x.
  const bool updated_met = result.relative_residual <= tolerance;
  result.relative_residual = (b - A * x).norm() / b_norm;
  result.converged = result.relative_residual <= tolerance;
  if (updated_met && !result.converged)
    result.stagnated = true;
  return result;
}

IterativeResult
preconditionedBiCGSTAB(const RowMatrix & A,
                       const VectorXd & b,
                       const SmoothedAggregationAmg & amg,
                       double tolerance,
                       int max_iterations,
                       VectorXd & x)
{
  IterativeResult result;
  const double b_norm = b.norm();
  if (x.size() != b.size())
    x = VectorXd::Zero(b.size());
  if (b_norm == 0.0)
  {
    x.setZero();
    result.converged = true;
    return result;
  }
  VectorXd r = b - A * x;
  const VectorXd r_hat = r;
  double rho = 1.0, alpha = 1.0, omega = 1.0;
  VectorXd v = VectorXd::Zero(b.size()), p = VectorXd::Zero(b.size());
  VectorXd p_hat, s_hat;
  for (int it = 1; it <= max_iterations; ++it)
  {
    const double rho_next = r_hat.dot(r);
    if (rho_next == 0.0 || omega == 0.0)
      break; // breakdown
    const double beta = (rho_next / rho) * (alpha / omega);
    p = r + beta * (p - omega * v);
    amg.apply(p, p_hat);
    v = A * p_hat;
    const double r_hat_v = r_hat.dot(v);
    if (r_hat_v == 0.0)
      break;
    alpha = rho_next / r_hat_v;
    const VectorXd s = r - alpha * v;
    result.iterations = it;
    if (s.norm() <= tolerance * b_norm)
    {
      x.noalias() += alpha * p_hat;
      r = s;
      break;
    }
    amg.apply(s, s_hat);
    const VectorXd t = A * s_hat;
    const double tt = t.dot(t);
    omega = tt > 0.0 ? t.dot(s) / tt : 0.0;
    x.noalias() += alpha * p_hat + omega * s_hat;
    r = s - omega * t;
    rho = rho_next;
    if (r.norm() <= tolerance * b_norm)
      break;
  }
  const double updated = r.norm() / b_norm;
  result.relative_residual = (b - A * x).norm() / b_norm;
  result.converged = result.relative_residual <= tolerance;
  if (!result.converged && updated <= tolerance)
    result.stagnated = true;
  return result;
}

} // namespace dualmesh
