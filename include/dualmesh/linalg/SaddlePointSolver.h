// SPDX-License-Identifier: LGPL-2.1-or-later
//
// Block preconditioning of saddle point systems.
//
// The Jacobian of an incompressible flow with the pressure as an unknown has
// the block form
//
//     [ A   B^T_u ] [ du ]   [ r_u ]
//     [ B_p   C   ] [ dp ] = [ r_p ]
//
// with A the momentum block, B_p the discrete divergence, B^T_u the pressure
// gradient and C the pressure block (zero for the Taylor-Hood element, a small
// pressure Laplacian for a stabilised equal-order element).  Its Schur
// complement S = C - B_p A^{-1} B^T_u is dense, but for the Stokes equations it
// is spectrally equivalent to the pressure mass matrix scaled by the inverse
// viscosity, S ~ (1/mu) M_p, with bounds that do not depend on the mesh size
// (the inf-sup constant gives the lower one).  The block upper-triangular
// preconditioner
//
//     P = [ A   B^T_u ]
//         [ 0   S_hat ]      with S_hat = sign * (1/mu) M_p ,
//
// applied exactly in A, then makes the number of GMRES iterations independent
// of the mesh.  The sign is that of S itself, which depends on how the
// divergence and the gradient are signed in the equations; it is measured from
// the two coupling blocks rather than assumed.
//
// The preconditioner changes from left to right from one iteration to the
// next only through its exact solves, so the outer method is the flexible
// GMRES of Saad (1993), which also accepts an inexact velocity solve.
#pragma once

#include "dualmesh/core/Types.h"

#include <Eigen/Sparse>

#include <memory>
#include <string>
#include <vector>

namespace dualmesh
{

/// The pieces that define the block preconditioner of a saddle point system.
struct SaddlePointBlocks
{
  /// Global indices of the pressure unknowns (the second block) and of all the
  /// others (the first block); together they cover every unknown once.
  std::vector<Index> pressure_dofs;
  std::vector<Index> other_dofs;
  /// The Schur complement approximation, in the order of pressure_dofs,
  /// including its sign.  Rows of prescribed or inactive pressure unknowns
  /// hold 1 on the diagonal, which is their exact Schur complement because
  /// their equations are rows of the identity.
  Eigen::SparseMatrix<double> schur_approximation;
  /// +1 or -1: the sign of the Schur complement that was measured.
  int sign = 1;
  /// Name of the pressure variable, for messages.
  std::string pressure_variable;
};

/// Right-preconditioned flexible GMRES with the block upper-triangular
/// preconditioner described above.  The momentum block is factorised with a
/// sparse LU and the Schur approximation with a sparse Cholesky
/// factorisation (it is a scaled mass matrix, hence symmetric and definite).
class SaddlePointSolver
{
public:
  struct Result
  {
    int iterations = 0;
    bool converged = false;
    double relative_residual = 0.0;
  };

  SaddlePointSolver(const Eigen::SparseMatrix<double> & matrix, const SaddlePointBlocks & blocks);
  ~SaddlePointSolver();

  /// Solve matrix * x = b to the relative tolerance, restarting after
  /// `restart` Krylov vectors, within `max_iterations` iterations in total.
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

} // namespace dualmesh
