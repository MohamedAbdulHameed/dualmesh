// SPDX-License-Identifier: LGPL-2.1-or-later
//
// The choice of the linear solver of a system, the same for a serial problem and a distributed
// one.  The choice is a list of attempts, tried in order until one converges: an iterative attempt
// has a bounded budget, and the last attempt is a direct factorisation, so that the automatic
// choice is never less robust than the direct solver.  The problem fills in the data of each kind
// (its blocks and its near null space, in its own numbering) and runs the attempts with PETSc.
#pragma once

#include "dualmesh/linalg/SaddlePointBlocks.h"

#include <optional>
#include <string>
#include <vector>

namespace dualmesh
{

/// What the choice of the linear solver needs to know about a system.
struct linear_system_traits
{
  /// The space dimension of the mesh and the number of unknowns over all processes.
  int dimension = 2;
  long long global_unknowns = 0;
  /// A pressure-velocity flow, and a system whose variables have elements of different orders.
  bool flow = false;
  bool mixed_order = false;
  /// The number of variables, and of those that are displacements of a solid.
  int variables = 1;
  int displacement_variables = 0;
  /// The PETSc options of the user, without the options of dualmesh that `user_choice` reads.
  std::string user_options;
};

/// The kinds of linear solves.
enum class linear_solve_kind
{
  /// The options of the user, as they are.
  user_options,
  /// A direct factorisation: LU in serial, MUMPS in parallel.
  direct,
  /// Flexible GMRES with the Schur complement split of a flow.
  Schur_split,
  /// GMRES with the split by physics, one algebraic multigrid cycle per block.
  physics_blocks,
  /// GMRES with smoothed aggregation (GAMG) and the rigid body modes of a solid.
  GAMG,
  /// GMRES with hypre's BoomerAMG.
  BoomerAMG,
};

/// One attempt of a linear solve.
struct linear_solve_attempt
{
  linear_solve_kind kind = linear_solve_kind::direct;
  /// The Schur approximation of a Schur split, and none for any other kind.
  std::optional<SchurApproximation> Schur_approximation = std::nullopt;
  /// Whether a failure stops the solve; otherwise the next attempt follows.
  bool must_converge = true;
};

/// The split that the user chose with the options of dualmesh in the PETSc options:
/// "-dualmesh_schur lsc|pcd|pressure_mass" splits a flow with that Schur approximation, and
/// "-dualmesh_split physics" splits the system by physics.
struct user_split
{
  std::optional<SchurApproximation> Schur;
  bool physics = false;
};

/// The PETSc options of the user, sorted: the split chosen with the options of dualmesh, the
/// options of the nonlinear solver (those that start with -snes_), and the others, which belong to
/// the linear solver.
struct user_options_parts
{
  user_split split;
  std::string linear;
  std::string nonlinear;
};
user_options_parts user_choice(const std::string & options);

/// The attempts of a system, in order.
std::vector<linear_solve_attempt> linear_solve_attempts(const linear_system_traits & system,
                                                        const user_split & split);

/// Whether a direct factorisation is the cheaper solve of a system of @p unknowns in @p dimension.
bool prefer_direct_solver(int dimension, long long unknowns);

} // namespace dualmesh
