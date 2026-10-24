// SPDX-License-Identifier: LGPL-2.1-or-later
//
// Block preconditioning of saddle point systems.
//
// The Jacobian of an incompressible flow with the pressure as an unknown has
// the block form
//
//     [ F   G ] [ du ]   [ r_u ]
//     [ D   C ] [ dp ] = [ r_p ]
//
// with F the momentum block, D the discrete divergence, G the pressure
// gradient and C the pressure block (zero for the Taylor-Hood element, a small
// pressure Laplacian for a stabilised equal-order element).  The block upper
// triangular preconditioner
//
//     P = [ F   G     ]
//         [ 0   S_hat ]
//
// makes the number of GMRES iterations depend only on how well S_hat
// approximates the Schur complement S = C - D F^{-1} G, which is dense.
// Up to its sign (measured from the coupling blocks), S ~ G^T F^{-1} G.  Three
// approximations are provided:
//
// * pressure_mass: S_hat = (1/mu) M_p.  For the Stokes equations S and M_p/mu
//   are spectrally equivalent with bounds independent of the mesh size; the
//   approximation ignores convection and the time derivative.
// * pcd, the pressure convection-diffusion preconditioner (Kay, Loghin and
//   Wathen, SIAM J. Sci. Comput. 24 (2002) 237-256) in the form of Elman and
//   Tuminaro (ETNA 35 (2009) 257-280):
//       S_hat^{-1} = A_p^{-1} F_p Q_p^{-1},   A_p = G^T Q_v^{-1} G,
//   with Q_v and Q_p lumped velocity and pressure mass matrices and F_p the
//   operator of the momentum equation on the pressure space,
//       F_p = (rho_t/dt) M_p + theta (mu K_p + rho N_p(w)),
//   where w is the current velocity.  F_p carries the Robin condition
//   mu dp/dn = rho (w . n) p on the boundaries with a prescribed velocity
//   (which is zero on a wall, so it acts at an inflow only) and the natural
//   condition elsewhere, which makes the commutator vanish at the boundary.
// * lsc, the least squares commutator preconditioner (Elman, Howle, Shadid,
//   Shuttleworth and Tuminaro, SIAM J. Sci. Comput. 27 (2006) 1651-1668):
//       S_hat^{-1} = A_p^{-1} (G^T Q_v^{-1} F Q_v^{-1} G) A_p^{-1},
//   which needs no operator besides the blocks of the matrix.
//
// For a stabilised equal-order element, G^T Q_v^{-1} G has the spurious
// pressure modes of the element in its null space, and A_p and the middle
// factor of lsc are stabilised by the element terms of Elman, Howle, Shadid,
// Silvester and Tuminaro (SIAM J. Sci. Comput. 30 (2007) 290-311, Sect. 4.1),
//       C_1 = sum_k h_k^{-2} (M_k - m_k m_k^T / |k|),
//       C_2 = sum_k (theta mu / h_k^4 + rho_t / (dt h_k^2)) (M_k - m_k m_k^T / |k|),
// with M_k the element pressure mass matrix and m_k the integrals of its shape
// functions, so that M_k - m_k m_k^T/|k| is the L2 distance from the element
// mean; h_k = |k|^{1/d}.  (Their C_2 is the steady form; the time term is the
// same scaling applied to the mass part of F.)  The paper derives C_1 for lsc.
// pcd is not covered there; it takes 100 C_1, because it applies A_p^{-1} once
// to F_p Q_p^{-1} r, which keeps the spurious modes at full size.  Measured
// on the cavity and a channel with Q1-Q1: with C_1 the iterations are three
// to four times those with 100 C_1, and from 100 C_1 to 10^5 C_1 they change
// by less than 15 per cent.  An assembled pressure Laplacian in place of A_p
// needs a Robin condition at an outflow whose weight changes the iterations
// fivefold, which is why A_p is taken from the matrix.
//
// The outer method is the flexible GMRES of Saad (1993), which accepts a
// preconditioner that changes from one iteration to the next.
//
// The same outer method serves the split by physics (PhysicsBlockSolver): the
// unknowns of each physics form a block, and the block lower-triangular
// preconditioner solves the blocks one after the other, each with one V-cycle
// of algebraic multigrid on its diagonal block, the coupling to the earlier
// blocks moved to the right-hand side (block Gauss-Seidel).
#pragma once

#include "dualmesh/core/Types.h"

#include <Eigen/Sparse>

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace dualmesh
{

/// The approximation of the Schur complement of a saddle point system.
enum class SchurApproximation
{
  PressureMass,
  ConvectionDiffusion,
  LeastSquaresCommutator
};

/// The Schur complement approximation that a preconditioner name selects
/// ("pressure_mass", "pcd" or "lsc"), if it names one.
std::optional<SchurApproximation> schurApproximation(const std::string & preconditioner);

/// The pieces that define the block preconditioner of a saddle point system,
/// in the numbering of the problem that assembled it.
struct SaddlePointBlocks
{
  SchurApproximation approximation = SchurApproximation::PressureMass;
  /// Why the blocks cannot be used (a viscosity that is not positive), or
  /// empty.  The caller refuses, on every process together.
  std::string refusal;
  /// Name of the pressure variable, for messages.
  std::string pressure_variable;
  /// The pressure unknowns (the second block) and all the others (the first
  /// block); together they cover every unknown once.
  std::vector<Index> pressure_dofs;
  std::vector<Index> other_dofs;
  /// The Frobenius product <G, D^T> of the coupling blocks, which is zero
  /// when the pressure is not coupled to the other unknowns, and the sign of
  /// S relative to G^T F^{-1} G that it gives: +1 when it is negative (G = -D^T).
  double coupling = 0.0;
  int sign = 1;
  /// Per unknown: whether its equation is prescribed (a row of the identity,
  /// whose Schur complement is exactly 1).
  std::vector<char> constrained;
  /// Operators on the pressure as entries (row, column, value), summed where
  /// repeated, without the prescribed pressure unknowns: (1/mu) M_p for
  /// pressure_mass; C_1 (pcd, lsc) and C_2 (lsc) for a stabilised element;
  /// F_p for pcd.
  std::vector<Eigen::Triplet<double>> mass;
  std::vector<Eigen::Triplet<double>> stabilization;
  std::vector<Eigen::Triplet<double>> commutator_stabilization;
  std::vector<Eigen::Triplet<double>> convection_diffusion;
  /// The lumped mass of every unknown: Q_p on the pressure, Q_v on the
  /// velocities, zero elsewhere (pcd, lsc).
  Eigen::VectorXd lumped_mass;
};

/// The Schur complement approximation of pressure_mass, in the order of
/// blocks.pressure_dofs: sign (1/mu) M_p plus the symmetric part of the
/// pressure block of @p matrix, whose prescribed pressures are rows of the
/// identity.
Eigen::SparseMatrix<double> pressureMassApproximation(const Eigen::SparseMatrix<double> & matrix,
                                                      const SaddlePointBlocks & blocks);

/// The result of a flexible GMRES solve.
struct FlexibleGmresResult
{
  int iterations = 0;
  bool converged = false;
  double relative_residual = 0.0;
};

/// Right-preconditioned flexible GMRES (Saad, SIAM J. Sci. Comput. 14 (1993)
/// 461-469): solve @p matrix x = @p b to the relative tolerance from the
/// initial guess in @p x, restarting after @p restart Krylov vectors, within
/// @p max_iterations iterations in total; @p precondition applies the
/// preconditioner, z = P^{-1} r.
FlexibleGmresResult
flexibleGmres(const Eigen::SparseMatrix<double> & matrix,
              const Eigen::VectorXd & b,
              Eigen::VectorXd & x,
              const std::function<Eigen::VectorXd(const Eigen::VectorXd &)> & precondition,
              double tolerance,
              int max_iterations,
              int restart);

/// Flexible GMRES with the block upper-triangular preconditioner described
/// above.  The momentum block is factorised with a sparse LU, and the
/// symmetric matrices of the Schur approximation (the pressure mass matrix, or
/// A_p) with a sparse Cholesky factorisation.
class SaddlePointSolver
{
public:
  using Result = FlexibleGmresResult;

  /// @p matrix and @p blocks are kept by reference and must outlive the solver.
  SaddlePointSolver(const Eigen::SparseMatrix<double> & matrix, const SaddlePointBlocks & blocks);
  ~SaddlePointSolver();

  /// Solve matrix * x = b with flexibleGmres from x = 0.
  Result solve(const Eigen::VectorXd & b,
               Eigen::VectorXd & x,
               double tolerance,
               int max_iterations,
               int restart);

  /// Apply the preconditioner: z = P^{-1} r.
  Eigen::VectorXd apply(const Eigen::VectorXd & r) const;

private:
  struct Factors;
  const Eigen::SparseMatrix<double> & _matrix;
  const SaddlePointBlocks & _blocks;
  std::unique_ptr<Factors> _factors;
};

/// Flexible GMRES with the block lower-triangular preconditioner of a split by
/// physics, one V-cycle of smoothed aggregation multigrid (Amg.h) on each
/// diagonal block.
class PhysicsBlockSolver
{
public:
  /// @param blocks the unknowns of every block, which together cover every
  ///        unknown once.
  /// @param near_nullspace the near null space of the whole matrix (one row
  ///        per unknown); each block takes the columns that do not vanish on it.
  /// @param block_sizes the number of unknowns per node of every block.
  /// @p matrix is kept by reference and must outlive the solver.
  PhysicsBlockSolver(const Eigen::SparseMatrix<double> & matrix,
                     const std::vector<std::vector<Index>> & blocks,
                     const Eigen::MatrixXd & near_nullspace,
                     const std::vector<int> & block_sizes);
  ~PhysicsBlockSolver();

  FlexibleGmresResult solve(const Eigen::VectorXd & b,
                            Eigen::VectorXd & x,
                            double tolerance,
                            int max_iterations,
                            int restart);
  Eigen::VectorXd apply(const Eigen::VectorXd & r) const;

private:
  struct Block;
  const Eigen::SparseMatrix<double> & _matrix;
  std::vector<std::unique_ptr<Block>> _blocks;
};

} // namespace dualmesh
