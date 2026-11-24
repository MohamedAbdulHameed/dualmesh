// SPDX-License-Identifier: LGPL-2.1-or-later
//
// The blocks of the preconditioners of saddle point systems, which PETSc's Schur complement field
// split applies (PetscSolver.h).
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
#pragma once

#include "dualmesh/core/Types.h"

#include <Eigen/Sparse>

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

/// The Schur complement approximation that a name selects ("pressure_mass", "pcd" or "lsc"), if it
/// names one.
std::optional<SchurApproximation> schurApproximation(const std::string & name);

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
  /// Per unknown: whether its equation is prescribed (a row of the diagonal,
  /// whose Schur complement is exactly its entry).
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

} // namespace dualmesh
