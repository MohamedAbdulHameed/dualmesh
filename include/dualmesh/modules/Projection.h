// SPDX-License-Identifier: LGPL-2.1-or-later
//
// The projection (fractional step) time integration of incompressible flow, the splitting of Nek5000 and nekRS (Fischer et al. 2021, Eqs. 4 to 14).
//
// With the kinematic pressure P = p / rho and the kinematic viscosity nu = mu / rho, one step to the time t^n takes
//
//     u* = - sum_j beta_j u^{n-j} - dt sum_j alpha_j (u^{n-j} . grad) u^{n-j} + dt f^n / rho ,
//
// the backward difference (BDFk) of the time derivative and the extrapolation (EXTk) of the advection, with j = 1 to k.
// The pressure follows from the divergence of the momentum equation, tested with the gradient of a pressure test function q:
//
//     (grad q, grad P) = (grad q, u*) / dt - (beta_0 / dt) int_G q n . u_b - nu int_G n . (omega x grad q) ,
//
// where u_b is the velocity on the boundary and omega the vorticity extrapolated from the steps before (the curl-curl form of the viscous term, which keeps the divergence error at the boundary small).
// The last term is the boundary form of nu (grad q, curl omega), so it needs only first derivatives of the velocity and applies to elements of any order.
// Each velocity component then solves a Helmholtz equation,
//
//     nu (grad v, grad u^n) + (beta_0 / dt) (v, u^n) = (v, u*) / dt - (v, dP/dx) .
//
// The equations are written with test functions: the shape functions of the finite element method, or the indicator of the control domain of each node for the methods of the dual mesh (dmcdm and hfvm), for which (v, f) is an integral over the control domain and (grad q, F) the flux of F out of it.
// The pressure equation is a Poisson equation, solved by the conjugate gradient method with algebraic multigrid (hypre BoomerAMG), and each Helmholtz equation is symmetric and diagonally dominant, solved by the conjugate gradient method with a Jacobi preconditioner.
// The matrices are assembled once and kept, with their preconditioners, as long as the step size is unchanged.
// The coefficients beta_j and alpha_j are those of a variable step, so a step shortened to land on an output time keeps the order.
#pragma once

#include "dualmesh/base/Problem.h"
#include "dualmesh/linalg/PetscSolver.h"

#include <deque>
#include <memory>
#include <string>
#include <vector>

namespace dualmesh
{

class DistributedProblem;

struct ProjectionSettings
{
  /// The velocity variables, one per coordinate, and the pressure variable.
  std::vector<std::string> velocities;
  std::string pressure = "pressure";
  double density = 1.0;
  double dynamic_viscosity = 1.0;
  /// The body force per unit volume, one function per component; an empty list or a null function is no force.
  std::vector<FunctionPtr> body_force;
  /// The order k of the backward difference and of the extrapolation, 1 to 3.
  int order = 2;
  /// PETSc's options of the pressure and of the velocity solves, and their relative tolerances.
  /// Empty chooses algebraic multigrid (hypre BoomerAMG) for the pressure and a Jacobi preconditioner for the velocity, inside the conjugate gradient method for the finite element method, whose matrices are symmetric, and inside GMRES for the methods of the dual mesh, whose matrices are not in general.
  std::string pressure_solver;
  std::string velocity_solver;
  double pressure_tolerance = 1e-10;
  double velocity_tolerance = 1e-10;
  /// The objects of the flow itself: its kernels and boundary conditions.
  /// The projection time integration assembles the flow's equations itself, so any other kernel or integrated boundary condition on the velocity or the pressure (a body force added on its own, a traction, a buoyancy) is refused rather than ignored.
  std::vector<std::string> own_objects;
  /// The number of earlier pressures from whose best combination each pressure solve starts (the projection of Fischer 1998); zero starts from the last pressure.
  int pressure_projection_vectors = 8;
};

class ProjectionSolver : public TimeIntegrator
{
public:
  /// @param distributed the distributed problem whose local problem is @p problem, or nullptr for a problem on one process.
  ProjectionSolver(Problem & problem,
                   DistributedProblem * distributed,
                   ProjectionSettings settings);
  ~ProjectionSolver() override;

  void start(const Vector & solution, double time) override;
  SolveResult step(Vector & solution, const Vector & old, double time, double dt) override;
  void accept(const Vector & solution, double time) override;
  /// dt times the largest speed over the size of its element, over the nodes of every element of every process.
  double courantNumber(const Vector & solution, double dt) const override;

  /// The coefficients beta_0 to beta_k of the backward difference and alpha_1 to alpha_k of the extrapolation at the time levels @p times (the new time first, then the earlier ones), scaled as in the equations above.
  static void coefficients(const std::vector<double> & times,
                           std::vector<double> & beta,
                           std::vector<double> & alpha);

  /// The iterations of the last step's pressure solve and of its velocity solves.
  int lastPressureIterations() const { return _pressure_iterations; }
  int lastVelocityIterations() const { return _velocity_iterations; }

private:
  struct Layout;
  struct Geometry;
  void setUp();
  /// The entries of a scalar system in the global numbering: those of @p local (one row per local entity), with the rows and columns of @p fixed replaced by the identity.
  std::vector<petsc::GlobalEntry> globalEntries(const std::vector<Eigen::Triplet<double>> & local,
                                                const std::vector<char> & fixed) const;
  /// One scalar system: the matrix @p local (one row per local entity), with the rows and columns of @p fixed replaced by the identity.
  std::unique_ptr<petsc::LinearSolver> makeSolver(const std::vector<Eigen::Triplet<double>> & local,
                                                  const std::vector<char> & fixed,
                                                  const std::string & options,
                                                  double tolerance,
                                                  int projection_vectors = 0) const;
  /// Solve the system of @p solver for one value per local entity: @p rhs is this process's part of the right-hand side, @p values holds the prescribed values on the rows of @p fixed and a first guess elsewhere, and receives the solution.
  int solve(const petsc::LinearSolver & solver,
            const SparseMatrix & matrix,
            const std::vector<char> & fixed,
            Vector rhs,
            Vector & values,
            bool remove_mean) const;

  Problem & _problem;
  DistributedProblem * _distributed;
  ProjectionSettings _settings;
  std::unique_ptr<Layout> _layout;
  std::unique_ptr<Geometry> _geometry;
  bool _ready = false;
  /// Whether the method tests with the control volumes of the dual mesh (dmcdm, hfvm) rather than with the shape functions (fem).
  bool _dual = false;
  std::string _pressure_options, _velocity_options;
  int _dim = 0;
  std::vector<int> _velocity;
  int _pressure = -1;
  /// The mass and stiffness matrices of one value per local entity, and the gradient matrices (v, d phi / dx_d).
  SparseMatrix _mass, _stiffness;
  std::vector<SparseMatrix> _gradient;
  /// The size of every integrated element, for the stabilization parameter.
  std::vector<double> _element_size;
  std::vector<Eigen::Triplet<double>> _mass_entries, _stiffness_entries;
  /// The prescribed rows of each velocity component and of the pressure.
  std::vector<std::vector<char>> _velocity_fixed;
  std::vector<char> _pressure_fixed;
  bool _pressure_pinned = false;
  std::unique_ptr<petsc::LinearSolver> _pressure_solver;
  /// The Helmholtz matrix nu K + c M of each velocity component, and the c it was built for.
  std::vector<std::unique_ptr<petsc::LinearSolver>> _velocity_solvers;
  /// The local Helmholtz matrix nu K + c M, and K and M on its sparsity pattern, so that a new c costs one sum of their values.
  SparseMatrix _helmholtz, _stiffness_on_pattern, _mass_on_pattern;
  /// The values of the entries of each velocity component's system that come from K and from M, in the order of the system's entries, and the number of identity rows that follow them.
  std::vector<std::vector<double>> _helmholtz_stiffness_values, _helmholtz_mass_values;
  std::vector<std::size_t> _helmholtz_identity_rows;
  double _helmholtz_c = -1.0;
  /// The sides of the boundary, on which the pressure equation takes its boundary terms.
  std::vector<Side> _boundary_sides;
  /// The accepted solutions and their times, the latest first.
  std::deque<std::pair<double, Vector>> _history;
  Vector _kinematic_pressure;
  int _pressure_iterations = 0, _velocity_iterations = 0;
};

} // namespace dualmesh
