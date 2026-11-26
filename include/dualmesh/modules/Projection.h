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
// The equations are written with test functions: the shape functions of the finite element method, or the indicator of the control domain of each node for the methods of the dual mesh (DMCDM), for which (v, f) is an integral over the control domain and (grad q, F) the flux of F out of it.
// On a mesh with hanging nodes each method treats them as its monolithic assembly does.
// FEM makes a hanging value the weighted sum of independent ones: every matrix A becomes C^T A C and every right-hand side b becomes C^T b, with C the map from the independent values to all of them, and the hanging values follow each solve.
// DMCDM keeps a hanging node as an unknown with a control domain of its own: a cell with hanging nodes on its sides is a polygon (PolygonCell), whose control domains and interfaces carry the integrals and the fluxes, and whose fields come from its corners.
// The pressure equation is a Poisson equation, solved with algebraic multigrid (hypre BoomerAMG), and each Helmholtz equation is diagonally dominant, solved with a Jacobi preconditioner: inside the conjugate gradient method for the finite element method, whose matrices are symmetric, and inside GMRES for the methods of the dual mesh, whose matrices are not in general.
// The matrices are assembled once and kept, with their preconditioners, as long as the step size is unchanged.
// The coefficients beta_j and alpha_j are those of a variable step, so a step shortened to land on an output time keeps the order.
#pragma once

#include "dualmesh/base/Problem.h"
#include "dualmesh/linalg/PetscSolver.h"

#include <algorithm>
#include <deque>
#include <memory>
#include <string>
#include <vector>

namespace dualmesh
{

class DistributedProblem;

/// The operators of a flow on a scalar field of one value per node, built with the test functions of the method (the equations above): the mass matrix (v, phi), the stiffness matrix (grad v, grad phi) and the gradient matrices (v, d phi / dx_d), each with the coordinate factor of the problem (2 pi r in axisymmetric coordinates).
/// They are assembled over the elements on which the first velocity lives.
/// The layout places the field in the global systems across hanging nodes, periodic boundaries and processes.
struct FlowOperators
{
  struct Layout;
  struct Geometry;
  FlowOperators(Problem & problem,
                DistributedProblem * distributed,
                const std::vector<int> & velocities);
  ~FlowOperators();

  /// The prescribed rows of @p variable: those of its nodal boundary conditions and, with @p inactive, of its inactive degrees of freedom.
  /// A copy of an entity is prescribed when any copy is, and a hanging node has no row.
  std::vector<char> prescribed(int variable, bool inactive = true) const;
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
  /// The sum of @p value over the processes.
  double sum(double value) const;
  /// The largest of @p value over the processes.
  double maximum(double value) const;

  Problem & problem;
  DistributedProblem * distributed;
  std::unique_ptr<Layout> layout;
  std::unique_ptr<Geometry> geometry;
  /// Whether the method tests with the control volumes of the dual mesh (DMCDM) rather than with the shape functions (FEM).
  bool dual = false;
  int dim = 0;
  std::vector<int> velocities;
  /// Whether each integrated element carries the flow.
  std::vector<char> in_flow;
  std::vector<Eigen::Triplet<double>> mass_entries, stiffness_entries;
  SparseMatrix mass, stiffness;
  std::vector<SparseMatrix> gradient;
  /// The size of every integrated element, for the stabilization parameter.
  std::vector<double> element_size;
  /// The prescribed rows of each velocity component.
  std::vector<std::vector<char>> velocity_fixed;
  /// The sides of the side sets on elements of the flow.
  std::vector<Side> boundary_sides;
};

/// Project the velocity of an incompressible flow in @p solution onto a field without divergence that keeps the prescribed values: the field u = u_0 + du nearest to u_0 in the norm of the mass matrix M with D u = 0, for the discrete divergence D of the mass balance of the flow without its stabilization.
/// The correction is du = - M^-1 D^T lambda, and the multiplier lambda solves the Poisson problem D M^-1 D^T lambda = D u_0, which is solved in its saddle point form with the mass matrix.
/// @p pressure is the pressure variable, whose prescribed values replace the mass balance of their nodes, or -1 for a flow without one (the penalty formulation).
/// Returns the largest change of the velocity over the largest speed after the projection, on every process alike.
double projectInitialVelocity(Problem & problem,
                              DistributedProblem * distributed,
                              const std::vector<int> & velocities,
                              int pressure,
                              Vector & solution);

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
  SolveResult
  step(Vector & solution, const Vector & old, double time, double dt, bool continues) override;
  void accept(const Vector & solution, double time) override;
  /// The velocity converges at the order k of the backward difference up to two, the order of
  /// the splitting of the pressure (docs/verification.rst, the stagnation flow).
  int order() const override { return std::min(_settings.order, 2); }
  /// The polynomial through the last order() + 1 accepted states, extrapolated to the end of the
  /// step, and the factor of Milne's device for the backward difference of order order().
  bool predict(double dt, Vector & predicted, double & factor) const override;
  bool measuresCourantNumber() const override { return true; }
  /// dt times the largest speed over the size of its element, over the nodes of every element of every process.
  double courantNumber(const Vector & solution, double dt) const override;

  /// The coefficients beta_0 to beta_k of the backward difference and alpha_1 to alpha_k of the extrapolation at the time levels @p times (the new time first, then the earlier ones), scaled as in the equations above.
  static void coefficients(const std::vector<double> & times,
                           std::vector<double> & beta,
                           std::vector<double> & alpha);

  /// The iterations of the last step's pressure solve and of its velocity solves (the benchmark
  /// program reports them).
  int lastPressureIterations() const { return _pressure_iterations; }
  int lastVelocityIterations() const { return _velocity_iterations; }

private:
  void setUp();

  Problem & _problem;
  DistributedProblem * _distributed;
  ProjectionSettings _settings;
  std::unique_ptr<FlowOperators> _operators;
  bool _ready = false;
  std::string _pressure_options, _velocity_options;
  std::vector<int> _velocity;
  int _pressure = -1;
  /// The prescribed rows of the pressure.
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
  /// The accepted solutions and their times, the latest first.
  std::deque<std::pair<double, Vector>> _history;
  /// The time and the solution of the last step taken since the last accepted one, from which
  /// the second half of a doubled step continues.
  std::pair<double, Vector> _trial;
  Vector _kinematic_pressure;
  /// The volume of every entity, the integral of its shape function, for the mean of the pressure.
  Vector _volume;
  double _total_volume = 0.0;
  int _pressure_iterations = 0, _velocity_iterations = 0;
};

} // namespace dualmesh
