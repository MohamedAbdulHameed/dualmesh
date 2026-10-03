// SPDX-License-Identifier: LGPL-2.1-or-later
//
// Algebraic multigrid by smoothed aggregation, as a preconditioner for the
// conjugate gradient and BiCGSTAB methods.
//
// The method (Vanek, Mandel and Brezina, Computing 56 (1996) 179-196) builds
// a hierarchy of coarser problems from the matrix alone.  The unknowns of each
// level are grouped into small aggregates of neighbouring nodes.  On every
// aggregate the near null space of the operator (the rigid body modes of an
// elastic body, the constant of a diffusion problem) is represented exactly
// by a tentative prolongator, which is then smoothed by one damped Jacobi
// step so that the coarse basis functions overlap.  The coarse operator is
// the Galerkin product P^T A P.  A V-cycle with a Chebyshev polynomial
// smoother (Adams, Brezina, Hu and Tuminaro, J. Comput. Phys. 188 (2003)
// 593-610) on every level and a sparse Cholesky factorisation on the coarsest
// one is the preconditioner.  For a three-dimensional elastic body the number
// of conjugate gradient iterations then grows only slowly with the number of
// unknowns, while a direct factorisation costs O(n^2) operations.
//
// The unknowns are numbered node by node, with the block_size unknowns of a
// node next to each other, which is the numbering of Problem::dof.
#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include <Eigen/SparseLU>

#include <memory>
#include <string>
#include <vector>

namespace dualmesh
{

struct AmgOptions
{
  /// Number of unknowns per node on the finest level.
  int block_size = 1;
  /// Two nodes are strongly connected when ||A_ij|| > threshold sqrt(||A_ii|| ||A_jj||),
  /// with ||.|| the Frobenius norm of the node blocks.  Only strong
  /// connections join nodes into aggregates.
  double strength_threshold = 0.03;
  /// The hierarchy stops when a level has at most this many unknowns.
  int coarse_size = 2000;
  int max_levels = 12;
  /// Degree of the Chebyshev smoother.
  int smoother_degree = 2;
  /// The smoother acts on the eigenvalues of D^-1 A in [lambda_max / ratio, 1.1 lambda_max].
  double smoother_ratio = 20.0;
  /// The damping of the prolongator smoothing step, omega = factor / lambda_max(D^-1 A).
  double prolongator_damping = 4.0 / 3.0;
  /// Whether the matrix is symmetric, which selects the Lanczos estimate of
  /// the largest eigenvalue of D^-1 A (the power method otherwise).
  bool symmetric = true;
};

/// The result of an iterative solve.
struct IterativeResult
{
  bool converged = false;
  int iterations = 0;
  double relative_residual = 0.0;
  /// The iteration stopped before the tolerance because the residual no
  /// longer decreased, i.e., it reached the round-off level of the system.
  bool stagnated = false;
};

class SmoothedAggregationAmg
{
public:
  using RowMatrix = Eigen::SparseMatrix<double, Eigen::RowMajor, int>;

  /// Build the hierarchy of the square matrix @p A.  The columns of
  /// @p near_nullspace (one row per unknown) are the vectors the coarse levels
  /// must represent exactly.
  void setup(const Eigen::SparseMatrix<double> & A,
             const Eigen::MatrixXd & near_nullspace,
             const AmgOptions & options);

  /// z = M^-1 r: one V-cycle from a zero initial guess.
  void apply(const Eigen::VectorXd & r, Eigen::VectorXd & z) const;

  int numLevels() const { return static_cast<int>(_levels.size()); }
  /// The sum of the nonzeros of all level operators divided by those of the finest.
  double operatorComplexity() const;
  /// One line per level: unknowns, nonzeros and block size.
  std::string summary() const;
  const RowMatrix & fineOperator() const { return _levels.front().A; }

private:
  struct Level
  {
    RowMatrix A;
    RowMatrix P; ///< prolongation to this level from the next coarser one
    RowMatrix R; ///< restriction, P^T
    Eigen::VectorXd inverse_diagonal;
    double lambda_max = 1.0;
    int block_size = 1;
  };
  std::vector<Level> _levels;
  AmgOptions _options;
  Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> _coarse_cholesky;
  Eigen::SparseLU<Eigen::SparseMatrix<double>> _coarse_lu;
  bool _coarse_uses_lu = false;

  void smooth(const Level & level, const Eigen::VectorXd & b, Eigen::VectorXd & x) const;
  void cycle(std::size_t index, const Eigen::VectorXd & b, Eigen::VectorXd & x) const;
};

/// The conjugate gradient method preconditioned by @p amg, for a symmetric
/// positive definite @p A.  @p x holds the initial guess and receives the
/// solution.  The iteration stops when ||b - A x|| <= tolerance ||b||.
IterativeResult preconditionedConjugateGradient(const SmoothedAggregationAmg::RowMatrix & A,
                                                const Eigen::VectorXd & b,
                                                const SmoothedAggregationAmg & amg,
                                                double tolerance,
                                                int max_iterations,
                                                Eigen::VectorXd & x);

/// BiCGSTAB (van der Vorst, SIAM J. Sci. Stat. Comput. 13 (1992) 631-644)
/// right-preconditioned by @p amg, for a general square @p A.
IterativeResult preconditionedBiCGSTAB(const SmoothedAggregationAmg::RowMatrix & A,
                                       const Eigen::VectorXd & b,
                                       const SmoothedAggregationAmg & amg,
                                       double tolerance,
                                       int max_iterations,
                                       Eigen::VectorXd & x);

} // namespace dualmesh
