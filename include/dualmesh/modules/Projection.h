// SPDX-License-Identifier: LGPL-2.1-or-later
//
// The projection (fractional step) time integration of incompressible flow, the splitting of Nek5000 and nekRS (Fischer et al. 2021, Eqs. 4 to 14).
//
// With the density rho, the kinematic pressure P = p / rho and the kinematic viscosity nu = mu / rho, one step to the time t^n takes
//
//     u* = - sum_j beta_j u^{n-j} - dt sum_j alpha_j (u^{n-j} . grad) u^{n-j} + dt s^n ,
//
// the backward difference (BDFk) of the time derivative and the extrapolation (EXTk) of the advection, with j = 1 to k, and the sources s of the momentum equations per unit mass at the new time (a body force over rho).
// The pressure follows from the divergence of the momentum equation, tested with the gradient of a pressure test function q:
//
//     (grad q, grad P) = (grad q, u*) / dt - (beta_0 / dt) int_G q n . u_b - nu int_G n . (omega x grad q) ,
//
// where u_b is the velocity on the boundary and omega the vorticity extrapolated from the steps before.
// The last term is the boundary form of nu (grad q, curl omega): the viscous term in its rotational form - nu curl omega, which differs from the Laplacian of the velocity by the gradient of its divergence.
// It gives the pressure the boundary condition of the momentum equation without a numerical boundary layer (the velocity-correction scheme in rotational form of Guermond and Shen 2003, the scheme of Karniadakis, Israeli and Orszag 1991), and needs only first derivatives of the velocity.
// The velocity then solves the momentum equations at the new time with that pressure,
//
//     rho (beta_0 / dt) M u^n + A u^n = - rho M sum_j beta_j u^{n-j} / dt - rho (v, sum_j alpha_j (u^{n-j} . grad) u^{n-j}) + (v, rho s^n) - G p^n ,
//
// with the mass matrix M, the viscous operator A in the form of the physics (laplacian or stress, which couples the components) and the pressure gradient G.
// These operators are those of the problem's own assembly of its kernels, in its coordinates (axisymmetric too), its formulation (equal-order or Taylor-Hood) and its numbering, across hanging nodes, periodic boundaries and processes, so a step and a monolithic solve of the same physics share their discretization and their boundary conditions.
// The equations are written with test functions: the shape functions of the finite element method, or the indicator of the control domain of each node for the dual mesh control domain method (DMCDM), for which (v, f) is an integral over the control domain and (grad q, F) the flux of F out of it.
// The pressure equation is a Poisson equation, solved with algebraic multigrid (hypre BoomerAMG), and the velocity equations are diagonally dominant, solved with a Jacobi preconditioner: inside the conjugate gradient method for the finite element method, whose matrices are symmetric, and inside GMRES for DMCDM, whose matrices are not in general.
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

/// Project the velocity of an incompressible flow in @p solution onto a field without divergence that keeps the prescribed values: the field u = u_0 + du nearest to u_0 in the norm of the mass matrix M of its time derivative, with D u = 0 for the discrete divergence D of the mass balance of the flow without its stabilization.
/// The correction is du = - M^-1 D^T lambda, and the multiplier lambda solves the Poisson problem D M^-1 D^T lambda = D u_0, which is solved in its saddle point form with the mass matrix.
/// @p pressure is the pressure variable, whose prescribed values replace the mass balance of their nodes, or -1 for a flow without one (the penalty formulation), whose balance is tested with the functions of the first velocity.
/// A flow without a time derivative of its velocity (a Stokes flow) has no initial velocity of its own, and is left as it is.
/// Returns the largest change of the velocity over the largest speed after the projection, on every process alike.
double projectInitialVelocity(Problem & problem,
                              const std::vector<int> & velocities,
                              int pressure,
                              Vector & solution);

struct ProjectionSettings
{
  /// The order k of the backward difference and of the extrapolation, 1 to 3.
  int order = 2;
  /// PETSc's options of the pressure and of the velocity solves, and their relative tolerances.
  /// Empty chooses algebraic multigrid (hypre BoomerAMG) for the pressure and a Jacobi preconditioner for the velocity, inside the conjugate gradient method for the finite element method, whose matrices are symmetric, and inside GMRES for DMCDM, whose matrices are not in general.
  std::string pressure_solver;
  std::string velocity_solver;
  double pressure_tolerance = 1e-10;
  double velocity_tolerance = 1e-10;
  /// The number of earlier pressures from whose best combination each pressure solve starts (the projection of Fischer 1998), and zero starts from the last pressure.
  int pressure_projection_vectors = 8;
};

/// The projection time integration of the flow of a problem: its mass_conservation equation, its velocity and its pressure.
/// The terms of the momentum equations take their roles from their kernels: the time derivative gives M, the terms with a flux (the viscous stress and the pressure) are taken at the new time through their Jacobian at rest (A and G), which holds for terms linear in the flow, the inertia (the advection of the velocity) is extrapolated, the other sources are taken at the new time with the extrapolated velocity, and the momentum stabilization of the monolithic solve is left out.
class ProjectionSolver : public TimeIntegrator
{
public:
  ProjectionSolver(Problem & problem, ProjectionSettings settings);
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

  struct Rows;
  struct Points;

private:
  void setUp();
  /// The source s of the momentum equations per unit mass at @p time with the velocity @p U, as one value per unknown of the velocity: the L2 projection of the residual of the sources onto the velocity field.
  void sources(const Vector & U, double time);

  Problem & _problem;
  ProjectionSettings _settings;
  bool _ready = false;
  std::vector<int> _velocity;
  int _pressure = -1;
  /// The density, and the kinematic viscosity of every element.
  double _density = 0.0;
  std::vector<double> _viscosity;
  /// Whether the mass equation is stabilized (equal-order velocity and pressure), and whether the
  /// momentum equations hold the advection of the velocity.
  bool _stabilized = false, _advection = false;
  /// Whether each element carries the flow, and its size.
  std::vector<char> _in_flow;
  std::vector<double> _element_size;
  /// The sides of the side sets on elements of the flow.
  std::vector<Side> _boundary_sides;
  std::unique_ptr<Points> _points;
  /// The systems of the velocity: one per component when the viscous operator does not couple
  /// them (the laplacian form), and one of every component otherwise (the stress form).
  std::vector<std::unique_ptr<Rows>> _velocity_rows;
  std::vector<std::unique_ptr<petsc::LinearSolver>> _velocity_solvers;
  std::unique_ptr<Rows> _pressure_rows, _source_rows;
  /// The operators in the unknowns of the problem: rho M, A and G, and the Laplacian of the pressure.
  SparseMatrix _mass, _viscous, _gradient, _laplacian;
  /// The sources of the momentum equations, their last residual and its projection s.
  std::vector<const Kernel *> _sources;
  Vector _source_residual, _source;
  /// The Helmholtz matrix rho c M + A of the velocity, and the c it was built for.
  SparseMatrix _helmholtz;
  double _helmholtz_c = -1.0;
  std::unique_ptr<petsc::LinearSolver> _pressure_solver, _source_solver;
  /// Whether the pressure is determined up to a constant only (an enclosed flow), and the weights
  /// of its mean.
  bool _pressure_pinned = false;
  Vector _pressure_weight;
  double _total_weight = 0.0;
  /// The accepted solutions and their times, the latest first.
  std::deque<std::pair<double, Vector>> _history;
  /// The time and the solution of the last step taken since the last accepted one, from which
  /// the second half of a doubled step continues.
  std::pair<double, Vector> _trial;
  /// The kinematic pressure of the last step, the first guess of the next pressure solve.
  Vector _kinematic_pressure;
  int _pressure_iterations = 0, _velocity_iterations = 0;
};

} // namespace dualmesh
