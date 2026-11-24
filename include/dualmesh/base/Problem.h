// SPDX-License-Identifier: LGPL-2.1-or-later
//
// Problem: owns the mesh, the variables (fields), and all objects, assembles
// residuals and Jacobians with either the dual mesh control domain method or
// the finite element method, and runs steady or transient executioners.
#pragma once

#include "dualmesh/base/Factory.h"
#include "dualmesh/base/Kernel.h"
#include "dualmesh/base/Property.h"
#include "dualmesh/fe/Assembly.h"
#include "dualmesh/linalg/SaddlePointBlocks.h"
#include "dualmesh/mesh/Mesh.h"

#include <Eigen/Sparse>

#include <array>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace dualmesh
{

using Vector = Eigen::VectorXd;
using SparseMatrix = Eigen::SparseMatrix<double>;

/// The polynomial order a variable is interpolated with on each element.
enum class VariableOrder
{
  /// The order of the element: linear on a linear element, quadratic on a
  /// quadratic one.  Every node carries the variable.
  Mesh,
  /// Linear on every element, including quadratic ones, where only the corner
  /// nodes carry the variable.  This is the pressure of the Taylor-Hood
  /// element (quadratic velocity, linear pressure), which is stable for
  /// incompressible flow without any stabilisation.  Finite element method
  /// only.
  First
};

struct Variable
{
  std::string name;
  int index;
  std::set<int> blocks; ///< empty = everywhere
  FunctionPtr initial_condition;
  double scaling = 1.0;
  VariableOrder order = VariableOrder::Mesh;
  /// The SI unit of the variable, written with its values (empty: none given).
  std::string unit;
};

struct SolverOptions
{
  /// "newton" (exact AD Jacobian), "picard" (direct iteration with lagged
  /// coefficients), or "linear" (one linear solve; for linear problems).
  std::string nonlinear_solver = "newton";
  int max_iterations = 50;
  /// Converged when the residual of every equation (the unknowns of one variable) is at most
  /// relative_tolerance times its reference residual, the size of the terms of that equation
  /// (AssemblyOptions::reference), or at the round-off level of its rows: a test with no unit,
  /// whatever the start.
  double relative_tolerance = 1e-10;
  /// Book relaxation (Eq. 6.2.15): U = (1 - gamma) U^{r+1} + gamma U^r, from the second step on,
  /// without a line search.
  double relaxation = 0.0;
  /// Newton only: "none" takes the full Newton step; "backtracking" halves
  /// the step until the residual norm falls, ||R(U + a dU)|| <= (1 - 1e-4 a)
  /// ||R(U)||, at most max_line_search_steps times (the Armijo condition on
  /// the residual norm).  It costs one residual assembly per trial.
  /// "automatic", the default, is backtracking when a kernel needs it
  /// (Kernel::needsLineSearch, the limiter of an advection scheme, on whose
  /// kinks the full step can cycle), and none otherwise.
  std::string line_search = "automatic";
  int max_line_search_steps = 10;
  /// Relative tolerance of the Krylov solvers, ||b - A x|| <= tol ||b||, for a linear solver that
  /// the user chose, for nonlinear_solver "linear" and for the linear solves outside Newton's
  /// method.  The automatic iterative solvers of Newton's method take the tolerance of Eisenstat
  /// and Walker instead, which follows the decrease of the residual.
  double linear_tolerance = 1e-12;
  /// PETSc's option string, for example "-ksp_type gmres -pc_type hypre" or
  /// "-snes_linesearch_damping 0.6" (those that start with -snes_ go to the nonlinear solver, the
  /// others to the linear solver).  Without options of the linear solver, the solver takes the
  /// automatic choice of LinearSolverChoice.h: a direct factorisation where it is cheap, and
  /// otherwise the iteration that suits the system (the Schur complement split of a flow,
  /// smoothed aggregation with the rigid body modes of a solid, the split by physics of coupled
  /// physics, BoomerAMG for the rest), with the direct factorisation as the fallback.  The options
  /// "-dualmesh_schur lsc|pcd|pressure_mass" and "-dualmesh_split physics" choose a split.
  std::string petsc_options;
  bool verbose = false;
  /// Throw if the nonlinear iteration does not converge.
  bool error_on_divergence = true;
};

struct TransientOptions
{
  double start_time = 0.0;
  double end_time = 1.0;
  /// The time step, or the first time step when the step size adapts.
  double dt = 0.1;
  /// 0 = forward (explicit) Euler, 0.5 = Crank-Nicolson, 1 = backward Euler.
  double theta = 1.0;
  /// The scheme of an equation with a second time derivative (Kernel::isSecondOrder):
  /// "generalized_alpha" (Chung and Hulbert 1993) with the spectral radius rho_infinity at
  /// infinite frequency, or "Newmark" (the average acceleration rule, without numerical
  /// damping).  The first-order equations of the same problem take theta = 1 - alpha_f.
  std::string second_order_scheme = "generalized_alpha";
  double spectral_radius = 0.75;
  /// The times at which the solution is written, in increasing order.  The
  /// steps land on these times.  Empty: the run writes no file.
  std::vector<double> output_times;
  /// The path and the start of the name of the files.  Empty: no file.
  std::string output_file_base;
  /// The variables that the files contain.  Empty: all the variables.
  std::vector<std::string> output_fields;
  /// The file of the checkpoint, written every checkpoint_interval accepted steps and after the last step.
  /// Empty: no checkpoint.
  std::string checkpoint_file;
  int checkpoint_interval = 0;
  /// A checkpoint to continue from: the run starts at its time, with its step size, its solution and the history of its material.
  std::string restart_file;

  // ---- adaptive time stepping --------------------------------------------
  /// How the step size is chosen.
  ///
  /// "fixed" keeps `dt` throughout, except that the last step is shortened to
  /// land exactly on `end_time`.
  ///
  /// "error" controls the local truncation error.  The theta method estimates it from one step
  /// and the polynomial through the last accepted states extrapolated to the end of the step
  /// (linear for an order-one member, quadratic for Crank-Nicolson): their difference is in
  /// proportion to the error.  The first steps, an equation with a second time derivative and a
  /// time integration of its own estimate it by step doubling instead: the same interval is
  /// advanced once with one step and once with two half steps, and the solution of the half steps
  /// is kept, at three nonlinear solves per step.  A step whose estimated error exceeds
  /// `error_tolerance` is thrown away and retried with a smaller step; an accepted step is
  /// followed by the largest step the estimate says will still meet the tolerance.
  ///
  /// "iteration" watches the nonlinear solver instead: a step that converged
  /// in fewer than `optimal_iterations - iteration_window` iterations is
  /// followed by a larger one, a step that needed more than
  /// `optimal_iterations + iteration_window` by a smaller one, and a step that
  /// did not converge at all is thrown away and retried.  It costs nothing
  /// beyond the solve itself and is the right choice when the difficulty is
  /// the nonlinearity rather than the accuracy.
  std::string time_stepper = "fixed";
  /// Smallest step allowed; a step that would have to go below it is an error.
  /// Zero means `dt` divided by one million.
  double dt_min = 0.0;
  /// Largest step allowed.  Zero means the whole time interval.
  double dt_max = 0.0;
  /// The most the step may grow from one step to the next.
  double growth_factor = 2.0;
  /// The factor applied to the step after it has been rejected.
  double cutback_factor = 0.5;
  /// Target for the relative local error of one step ("error" stepper).
  double error_tolerance = 1.0e-3;
  /// Nonlinear iteration count the "iteration" stepper aims for.
  int optimal_iterations = 4;
  /// Half-width of the band around `optimal_iterations` in which the step is
  /// left alone.
  int iteration_window = 2;
  /// How many times in a row a step may be rejected before giving up.
  int max_rejected_steps = 10;
  /// Stop after this many accepted steps, at the time they reach (an adaptive run adapts the mesh
  /// there and continues).  Zero: no limit.
  int max_steps = 0;
  /// The Courant number the "cfl" stepper aims for: each step is the one that gives this Courant number at the solution of the step before, grown by at most growth_factor.
  /// The stepper needs a time integration that measures the Courant number (the projection integration of a flow).
  double courant_number = 0.5;
};

struct IterationRecord
{
  /// The solve of the problem itself is step 1; the steps of a continuation follow it.
  int continuation_step;
  /// The fraction of the driving data the step solves with (1: the problem itself).
  double data_fraction;
  int iteration;
  double residual_norm;
  double step_norm;
};

struct SolveResult
{
  bool converged = false;
  /// Nonlinear (Newton or Picard) iterations, summed over the continuation steps and,
  /// in a transient run, over the time steps.
  int total_iterations = 0;
  /// Iterations taken by the iterative linear solver, summed over all the
  /// linear systems that were solved.  Zero for the direct solver.  This is
  /// the number to watch when judging a preconditioner, and in a distributed
  /// run it is how the cost of adding ranks shows up.
  int linear_iterations = 0;
  /// Time steps accepted (transient runs).
  int time_steps = 0;
  /// Time steps that were computed and then thrown away because the estimated
  /// error was too large or the nonlinear solve failed (adaptive runs).
  int rejected_steps = 0;
  /// The time reached and the step taken, for every accepted step.
  std::vector<std::pair<double, double>> step_history;
  /// The step that the step controller chose to take next (transient runs).
  double next_time_step = 0.0;
  std::vector<IterationRecord> history;
};

/// The time-stepping loop, shared by the serial and distributed executioners.
///
/// @param take_step advances @p solution by one step of the given size,
///        starting from the state it is given, and reports whether the
///        nonlinear solve converged and how many iterations it took.  It must
///        not throw on non-convergence.  Its third argument says whether the
///        step starts from the last accepted state (so a material history
///        starts from its committed values) or continues the step before it
///        (the second half of a doubled step).
/// @param on_accept is called after every accepted step with the step number
///        and the step size, for user callbacks.
/// @param write_output is called with the index of an output time (of
///        TransientOptions::output_times) when the solution is at that time:
///        at the start, and after the accepted step that lands on it.  The
///        steps land on the output times.  It can be empty.
/// @param verbose_root suppresses the progress messages on every process but
///        one in a distributed run.
/// @param relative_difference the difference of two solution vectors relative to their size,
///        without a unit (Problem::relativeStep: each variable against its range), global in a
///        distributed run so that every process estimates the same error and takes the same
///        decision.  Empty: the norm of the difference over the norm of the first, at least 1.
/// @param error_by_extrapolation the "error" stepper estimates the local error of the theta method
///        from one step and the extrapolation of the last accepted states.  Otherwise, and before
///        enough states are accepted, it estimates it by step doubling.
SolveResult
runTransient(const TransientOptions & transient,
             const SolverOptions & options,
             Vector & solution,
             double & current_time,
             const std::function<SolveResult(const Vector &, double, bool)> & take_step,
             const std::function<void(int, double)> & on_accept,
             const std::function<void(int)> & write_output,
             bool verbose_root = true,
             const std::function<void(int, double)> & after_step = {},
             const std::function<double(double)> & courant = {},
             const std::function<double(const Vector &, const Vector &)> & relative_difference = {},
             bool error_by_extrapolation = false);

/// A time integration that takes the place of the theta method in a transient solve.
/// The transient loop stays (the output times, the checkpoints, the callbacks), and the integrator advances the solution by one step.
class TimeIntegrator
{
public:
  virtual ~TimeIntegrator() = default;
  /// Called once before the first step, with the solution at the start time.
  virtual void start(const Vector & solution, double time) = 0;
  /// Advance @p solution from @p old, the last accepted solution, at time - dt, to @p time.
  virtual SolveResult step(Vector & solution, const Vector & old, double time, double dt) = 0;
  /// The step to @p time was accepted with @p solution.
  virtual void accept(const Vector & solution, double time) = 0;
  /// The Courant number of a step @p dt at @p solution, on every process alike; zero for an integration that does not measure it.
  virtual double courantNumber(const Vector & solution, double dt) const
  {
    (void) solution;
    (void) dt;
    return 0.0;
  }
};

class Problem;
struct CheckpointLayout;

/// The numbering of the unknowns of a problem in the system that its processes solve together.
struct GlobalRows
{
  /// Whether the system spans the processes of MPI_COMM_WORLD (otherwise this process alone).
  bool distributed = false;
  Index global_size = 0;
  /// This process owns the rows [first_row, first_row + num_owned).
  Index first_row = 0;
  Index num_owned = 0;
  /// The global row of every local unknown (-1: one that no process solves for), and whether this
  /// process owns it.
  std::vector<Index> global_dof;
  std::vector<char> owned_dof;
};

/// What the solvers of a Problem need from the processes that share it.  The
/// default is the serial identity: one process holds the whole problem.  A
/// DistributedProblem implements it for the part of the problem its process
/// holds, so that one nonlinear driver and one transient driver serve both.
class Parallel
{
public:
  virtual ~Parallel() = default;
  /// Whether this process prints the progress of a solve.
  virtual bool isRoot() const { return true; }
  /// The sum of a number over the processes.
  virtual double sum(double value) const { return value; }
  /// Whether a condition holds on some process.
  virtual bool any(bool value) const { return value; }
  /// The smallest and the largest of a number over the processes.
  virtual double minimum(double value) const { return value; }
  virtual double maximum(double value) const { return value; }
  /// The number of local unknown @p dof in the numbering of all processes; -1 for one that no
  /// process solves for.
  virtual Index globalDof(Index dof) const { return dof; }
  /// The norm of a vector of unknowns over the processes, each unknown counted once.
  virtual double norm(const Vector & v) const { return v.norm(); }
  /// Add the contributions of the other processes to the entries of an assembled residual
  /// that they share.
  virtual void addShared(Vector &) const {}
  /// After the prescribed values are written into @p solution: make them, and every value of
  /// a node another process owns, those of the owner.
  virtual void shareDirichlet(Vector &) const {}
  /// The global numbering of the unknowns of @p problem.
  virtual GlobalRows globalRows(const Problem & problem) const;
  /// Make every value of @p v at a node another process owns that of the owner.
  virtual void copyFromOwners(Vector &) const {}
  /// Add to @p values, @p width per local node, the values of the other processes at the nodes
  /// they share.
  virtual void addAtNodes(std::vector<double> & values, int width) const
  {
    (void) values;
    (void) width;
  }
  /// Whether the singularity check of the Jacobian runs, which needs the whole matrix.
  virtual bool checksSingularity() const { return true; }
  /// The global numbers of the nodes and elements of @p problem, for a checkpoint.
  virtual CheckpointLayout checkpointLayout(const Problem & problem) const;
  /// Write the fields of @p problem to the files of @p base (one file, or one per process).
  virtual void writeFields(Problem & problem,
                           const std::string & base,
                           const std::vector<std::string> & fields) const;
};

class Problem
{
public:
  /// The processes that share this problem (the serial identity unless a DistributedProblem
  /// holds it).
  const Parallel & parallel() const;
  void setParallel(const Parallel * parallel) { _parallel = parallel; }
  /// The sides of the side sets @p names, each once (a side may lie in several of them), in the
  /// order of the names and then of each side set.
  std::vector<Side> boundarySides(const std::vector<std::string> & names) const;
  /// The norm of @p v over the unknowns of each variable, on every process: one value per
  /// equation, in the units of that equation.
  std::vector<double> equationNorms(const Vector & v) const;
  /// The largest step of @p after from @p before over the unknowns of each variable, relative to
  /// the range of that variable (its largest value less its smallest, or its largest magnitude
  /// when they are equal): a number with no unit, which a temperature offset does not shrink.
  /// With @p variables, over the variables it marks only.
  double relativeStep(const Vector & after,
                      const Vector & before,
                      const std::vector<char> * variables = nullptr) const;
  Problem(std::shared_ptr<Mesh> mesh,
          Method method = Method::DualMesh,
          CoordinateSystem coord = CoordinateSystem::Cartesian);

  // ---- setup ------------------------------------------------------------------
  int addVariable(const std::string & name,
                  const std::vector<std::string> & blocks = {},
                  FunctionPtr initial_condition = nullptr,
                  double scaling = 1.0,
                  VariableOrder order = VariableOrder::Mesh);
  /// Let a variable also live on @p blocks (empty: everywhere), for a second physics that shares
  /// the field on other blocks.
  void extendVariableBlocks(const std::string & name, const std::vector<std::string> & blocks);
  void addFunction(const std::string & name, FunctionPtr f);
  std::shared_ptr<Object>
  addObject(const std::string & type, const std::string & name, InputParameters params);
  /// Add an already constructed object (used for Python-defined objects).
  void addKernel(std::shared_ptr<Kernel> k);
  void addIntegratedBC(std::shared_ptr<IntegratedBC> bc);
  void addNodalBC(std::shared_ptr<NodalBC> bc);
  void addNodalLoad(std::shared_ptr<NodalLoad> load);
  void addConstraint(std::shared_ptr<Constraint> constraint);
  /// The distributed solver joins the constraints across processes, so the local problem of one process leaves them to it.
  void setConstraintsHandledOutside(bool outside)
  {
    _constraints_outside = outside;
    _initialized = false;
  }
  /// Degrees of freedom to treat as prescribed besides the problem's own: the distributed solver sets the copies of prescribed nodes that only another process knows as prescribed.
  void setExtraConstrainedDofs(std::vector<char> mask) { _extra_constrained = std::move(mask); }
  /// The residual from which reactions are read: the distributed solver sets the residual summed over the processes.
  void setLastResidual(const Vector & R) { _last_residual = R; }
  /// The residual of the last solve before its prescribed rows were replaced (its reactions).
  const Vector & lastResidual() const { return _last_residual; }
  void addProperty(std::shared_ptr<Property> m);

  /// Resolve all objects (idempotent; called automatically by solve()).
  void initialize();

  // ---- queries --------------------------------------------------------------------
  const Mesh & mesh() const { return *_mesh; }
  std::shared_ptr<Mesh> meshPointer() const { return _mesh; }
  Method method() const { return _method; }
  /// Whether the method is the dual mesh control domain method, which tests with control volumes.
  bool isDual() const { return _method == Method::DualMesh; }
  /// Change the method before the problem is initialised: the method chosen from the physics.
  void setMethod(Method method);
  CoordinateSystem coordinateSystem() const { return _coord; }
  /// The value of the symbol eigenvalue that the expressions read; an eigenvalue study sets it.
  void setEigenvalue(double value) { _eigenvalue = value; }
  double eigenvalue() const { return _eigenvalue; }
  int numVariables() const { return static_cast<int>(_vars.size()); }
  const Variable & variable(int i) const { return _vars[i]; }
  int variableIndex(const std::string & name) const;
  bool hasVariable(const std::string & name) const;
  FunctionPtr function(const std::string & name) const;
  /// Whether a function of this name is registered.
  bool hasFunction(const std::string & name) const { return _functions.count(name) > 0; }
  /// The kernels and the integrated boundary conditions of the problem, in the order they were added.
  const std::vector<std::shared_ptr<Kernel>> & kernels() const { return _kernels; }
  const std::vector<std::shared_ptr<IntegratedBC>> & integratedBCs() const { return _ibcs; }
  /// The names of the registered functions.
  std::vector<std::string> functionNames() const
  {
    std::vector<std::string> out;
    for (const auto & item : _functions)
      out.push_back(item.first);
    return out;
  }
  Index numDofs() const { return _mesh->numNodes() * numVariables(); }
  Index dof(Index node, int var) const { return node * numVariables() + var; }

  // ---- variables of different orders --------------------------------------------
  //
  // Every mesh node has a slot for every variable in the global numbering
  // above, which keeps the numbering, the parallel exchange and the block
  // structure of the matrix (one block per node) the same for every problem.
  // A first-order variable on a quadratic element leaves the slots of the
  // mid-edge, mid-face and interior nodes unused: they are inactive (their
  // equations are the identity), and after every change of the solution they
  // hold the value of the linear interpolant, so that the element's quadratic
  // interpolation of the nodal values reproduces the linear field exactly and
  // every post-processing routine sees the right field.
  //
  // Inside one element the local degrees of freedom are numbered compactly:
  // first every variable at every corner node, node by node, then the
  // mesh-order variables at the remaining nodes.  Automatic differentiation
  // seeds one derivative per local degree of freedom, so the unused slots cost
  // nothing.

  /// Whether some variable is first order and the mesh has quadratic elements.
  bool hasMixedOrder() const { return _mixed_order; }
  /// Whether variable @p var is interpolated linearly on element @p el while
  /// the element itself is quadratic.
  bool usesCornerShape(const Element & el, int var) const
  {
    return _mixed_order && _full_order_rank[var] < 0 && el.numCorners() < el.numNodes();
  }
  /// The number of nodes of @p el that carry variable @p var: the corners for
  /// a first-order variable, every node otherwise.
  int numLocalNodes(const Element & el, int var) const
  {
    return usesCornerShape(el, var) ? el.numCorners() : el.numNodes();
  }
  /// The number of local degrees of freedom of element @p el.
  int numLocalDofs(const Element & el) const
  {
    const int corners = _mixed_order ? el.numCorners() : el.numNodes();
    return corners * numVariables() + (el.numNodes() - corners) * _num_full_order;
  }
  /// The local index of variable @p var at local node @p node of @p el, or -1
  /// when the node does not carry the variable.
  int localDofIndex(const Element & el, int node, int var) const
  {
    const int corners = _mixed_order ? el.numCorners() : el.numNodes();
    if (node < corners)
      return node * numVariables() + var;
    const int rank = _full_order_rank[var];
    return rank < 0 ? -1 : corners * numVariables() + (node - corners) * _num_full_order + rank;
  }
  /// Set every degree of freedom that is not an unknown of its own: the unused slots of every first-order variable (the non-corner nodes of quadratic elements) to the linear interpolant of the corner values, and every degree of freedom that a constraint makes dependent to the value of its primary.
  void updateDependentDofs(Vector & U) const;

  /// The blocks of the preconditioner of a flow for the linear system with
  /// matrix @p jacobian, with the Schur complement approximated by
  /// @p approximation (see SaddlePointBlocks.h): the pressure is the variable
  /// of the one kernel that provides the flow coefficients (MassConservation),
  /// the operators on the pressure are assembled from them at the current
  /// solution, and the sign of the Schur complement is measured from the
  /// coupling blocks of @p jacobian.
  SaddlePointBlocks saddlePointBlocks(const SparseMatrix & jacobian,
                                      SchurApproximation approximation) const;
  /// The mass conservation equation of a flow: the one kernel that provides
  /// the flow coefficients, or null.
  const Kernel * flowConstraint() const;
  /// Whether "automatic" should factorise a system of @p n unknowns directly.
  bool preferDirectSolver(Index n) const;
  /// The time step and the weight of the steady terms of the last linear solve of a Newton step
  /// (zero and one for a steady solve), which the flow preconditioners take.
  double linearDt() const { return _linear_dt; }
  double linearTheta() const { return _linear_theta; }

  /// Number of threads requested for the assembly loops; 0 means the OpenMP
  /// default (the value of OMP_NUM_THREADS, or the number of cores).
  void setNumThreads(int n) { _num_threads = n; }
  int numThreads() const { return _num_threads; }
  /// The number of threads that will actually be used.  This is 1 when the
  /// library was built without OpenMP, and 1 when any object of the problem is
  /// defined in Python, because calling back into the interpreter requires the
  /// global interpreter lock.
  int effectiveThreads() const;
  /// Whether every object of the problem can be called from several threads.
  /// The answer depends on the objects that have been added, so the problem is
  /// resolved first if that has not happened yet.
  bool threadSafe() const
  {
    const_cast<Problem *>(this)->initialize();
    return _thread_safe;
  }
  /// Fewer elements than this per thread and the assembly runs serially,
  /// because starting a team of threads would cost more than it saves.
  static constexpr Index kMinElementsPerThread = 400;
  PropertyRegistry & propertyRegistry() { return _props; }
  const PropertyRegistry & propertyRegistry() const { return _props; }
  std::shared_ptr<Object> object(const std::string & name) const;
  /// The concentrated loads of the problem, needed by the distributed solver
  /// to decide which process applies a load given by coordinates.
  const std::vector<std::shared_ptr<NodalLoad>> & nodalLoads() const { return _loads; }
  const std::vector<std::shared_ptr<Constraint>> & constraints() const { return _constraints; }
  /// The primary of every degree of freedom that a constraint makes dependent, and -1 for every other one; empty when the problem has no constraint.
  const std::vector<Index> & primaryDofs() const { return _primary_of; }
  const std::vector<std::shared_ptr<NodalBC>> & nodalBCs() const { return _nbcs; }
  std::vector<std::string> objectNames() const;
  double time() const { return _time; }
  void setTime(double t) { _time = t; }

  // ---- history of stateful property objects ------------------------------------------------
  /// Make the state reached at the last solve the committed one (the start of
  /// the next step).  The transient solver calls it for every accepted step.
  void commitState();
  /// Start the next step from the committed history, or from the state reached by the last solve (the second half of a doubled step).
  void startStateFromCommitted();
  void startStateFromTrial();
  /// A copy of the committed history, and its restoration: for a caller that
  /// solves the same step more than once (an outer iteration).
  struct StateSnapshot
  {
    std::vector<std::vector<std::uint64_t>> keys;
    std::vector<std::vector<double>> values;
  };
  StateSnapshot snapshotState() const;
  void restoreState(const StateSnapshot & snapshot);
  /// Keys of the points whose history is kept; ps is the kind of point set.
  static std::uint64_t stateKey(unsigned ps, unsigned a, unsigned b, unsigned q)
  {
    return (static_cast<std::uint64_t>(ps & 0xff) << 56) |
           (static_cast<std::uint64_t>(a & 0xffff) << 40) |
           (static_cast<std::uint64_t>(b & 0xffff) << 24) | (q & 0xffffff);
  }
  static constexpr std::uint64_t kCentroidStateKey = ~static_cast<std::uint64_t>(0);

  // ---- element fields ----------------------------------------------------------------
  /// Named fields with one value per element, set from outside the solve (for
  /// instance a strain accumulated between time steps) and read
  /// by property objects at any integration point of the element.  A property object asks
  /// for its field during setup with elementField(), which creates a field of
  /// zeros if none has been set, and keeps a reference: a later
  /// setElementField() replaces the values in place, so the property objects see
  /// them at the next assembly.
  void setElementField(const std::string & name, const std::vector<double> & values);
  const std::vector<double> & elementField(const std::string & name);
  std::vector<std::string> elementFieldNames() const;
  /// The values of an element field, or nullptr when the problem has none of that name.
  const std::vector<double> * findElementField(const std::string & name) const
  {
    const auto it = _element_fields.find(name);
    return it == _element_fields.end() ? nullptr : &it->second;
  }

  // ---- solution ---------------------------------------------------------------------
  Vector & solution() { return _U; }
  const Vector & solution() const { return _U; }
  std::vector<double> values(const std::string & var) const;
  void setValues(const std::string & var, const std::vector<double> & v);
  void applyInitialConditions();

  // ---- assembly -----------------------------------------------------------------------
  struct AssemblyOptions
  {
    LinearizationMode mode = LinearizationMode::Newton;
    const Vector * lagged = nullptr; ///< previous iterate (Picard)
    const Vector * old = nullptr;    ///< previous time step (transient)
    double dt = 0.0;
    double data_fraction = 1.0;
    double inertia = 1.0; ///< the factor of the inertia of a flow (QpContext::inertia)
    double theta = 1.0;   ///< weight of steady terms
    /// The old time of a step of the theta method, whose steady terms are weighted by 1 - theta.
    bool old_level = false;
    /// At the old time: @p old holds the state that the previous step started from, and dt and
    /// previous_theta are those of that step (QpContext::previous_step).
    bool previous_step = false;
    double previous_theta = 1.0;
    bool include_time_kernels = true;
    bool include_steady_terms = true;
    /// The time derivatives of the second-order variables (QpContext::rate, acceleration):
    /// du/dt = rate_scale u - rate_shift and d2u/dt2 = accel_scale u - accel_shift.  Without
    /// the shifts they are rate_scale u and accel_scale u (the operators of the modes).
    bool second_order = false;
    double rate_scale = 0.0, accel_scale = 0.0;
    const Vector * rate_shift = nullptr;
    const Vector * accel_shift = nullptr;
    /// When set, the assembly also sums the size of every contribution to each entry of the
    /// residual: the reference residual, whose norm over the unknowns of one variable is the
    /// size of the terms of its equation, in its units (the conducted and carried heat, the
    /// forces).  The convergence of each equation is measured against it (nonlinearSolve).
    Vector * reference = nullptr;
    /// When given: the size of the terms that each entry of the residual is computed from, inside
    /// the fluxes and the sources (Kernel::fluxTermSize).  Its round-off is about the machine
    /// epsilon times it, which is the floor of Newton's test where terms cancel.
    Vector * term_size = nullptr;
  };
  /// The initial velocity of the second-order variables of a transient run (zero: at rest).
  void setInitialVelocity(const Vector & v) { _initial_velocity = v; }
  /// The velocity of the solution, after a run of the generalized-alpha method (empty
  /// otherwise).
  Vector velocity() const;
  /// The variables of second-order equations (Kernel::isSecondOrder), one flag per variable.
  const std::vector<char> & secondOrderVariables() const { return _second_order_variables; }
  /// Assemble R(U) (and optionally dR/dU) without applying Dirichlet rows.
  void assemble(const Vector & U, const AssemblyOptions & opts, Vector & R, SparseMatrix * J);
  /// Write the prescribed values into @p U at the nodes of every essential
  /// boundary condition.
  void applyDirichlet(Vector & U, double data_fraction) const;
  /// Replace the equations of the prescribed degrees of freedom: the residual
  /// entry becomes zero, and the row and the column of the Jacobian are
  /// dropped but for the diagonal, which takes the mean size of the diagonal
  /// of the equation of that variable (so the system has the same form in any
  /// units).  Dropping the column as
  /// well as the row keeps a symmetric problem symmetric, which is valid
  /// because the increment at a prescribed degree of freedom is zero.
  void dirichletRows(const Vector & U, double data_fraction, Vector & R, SparseMatrix * J) const;
  /// The degrees of freedom whose equations are replaced by a prescribed
  /// value: those of the nodal boundary conditions and the inactive ones.
  std::vector<char> constrainedDofs() const;
  /// The near null space of the system for the algebraic multigrid
  /// preconditioner: for every set of displacement variables named by a
  /// property object, the rigid body translations and (in Cartesian coordinates)
  /// rotations, and the constant vector of every other variable.  One row per
  /// degree of freedom.
  Eigen::MatrixXd nearNullspace() const;
  /// The displacement variables of every elastic property object, in component order.
  std::vector<std::vector<int>> displacementGroups() const;
  /// The blocks of the preconditioner "physics_blocks": every group of
  /// displacements, and every other variable on its own, in the order of
  /// their variables (the order in which the physics were added).  A flow is
  /// refused: its velocity and pressure need a Schur complement split.
  std::vector<std::vector<int>> physicsBlocks() const;
  /// The unknowns of every block of physicsBlocks, in increasing order.
  std::vector<std::vector<Index>> physicsBlockDofs() const;
  /// Degrees of freedom that no kernel of this problem touches, and whose
  /// equations are therefore replaced by the identity.
  const std::vector<char> & activeDofs() const { return _active_dof; }
  /// Replace the active degree of freedom mask.  A distributed run needs this,
  /// because a degree of freedom that no local element touches may still be
  /// touched by an element on another rank.
  void overrideActiveDofs(const std::vector<char> & active);
  /// Mark which entities this process owns.  Contributions that are attached
  /// to an entity rather than to an element or a side (concentrated loads)
  /// are applied only on the owning process, so that a shared node does not
  /// receive the same point load once per process.  An empty mask, the
  /// default, means that every entity is owned.
  void setOwnedNodes(std::vector<char> owned) { _owned_node = std::move(owned); }

  // ---- executioners ---------------------------------------------------------------------
  SolveResult solveSteady(const SolverOptions & options = {});
  /// Solve A x = b with PETSc, A and b in the local unknowns, by the solver chosen by @p o
  /// (LinearSolverChoice.h); the result holds the owners' values on every process.
  Vector linearSolve(const SparseMatrix & A,
                     const Vector & b,
                     const SolverOptions & o,
                     int * iterations = nullptr) const;
  SolveResult solveTransient(const TransientOptions & transient,
                             const SolverOptions & options = {});
  /// Called after every converged time step (time, problem).
  /// The elements that this problem integrates: all of them, except the ghost elements at the end of the mesh that a distributed run adds to read the other side of an interface.
  Index numIntegratedElements() const
  {
    return _num_integrated_elements >= 0 ? _num_integrated_elements : _mesh->numElements();
  }
  /// Make the elements from @p first on ghosts, which are read but not integrated, and size the solution for the nodes added to the mesh with them.
  void setGhostElements(Index first);
  /// Sides of ghost elements on side set @p sideset, which an interface condition with that secondary side set may pair with.
  void addInterfaceCandidates(const std::string & sideset, const std::vector<Side> & sides);
  /// Call the callback of setTimeStepCallback, if any, for a step accepted at the current time.
  void notifyStepAccepted()
  {
    if (_step_callback)
      _step_callback(_time, *this);
  }
  /// Advance the transient solves with @p integrator instead of the theta method; nullptr restores the theta method.
  void setTimeIntegrator(std::shared_ptr<TimeIntegrator> integrator)
  {
    _time_integrator = std::move(integrator);
  }
  void setTimeStepCallback(std::function<void(double, Problem &)> cb)
  {
    _step_callback = std::move(cb);
  }
  /// Called with the index of an output time of a transient run
  /// (TransientOptions::output_times) when the solution is at that time,
  /// after the VTU file of that time is written.  An empty function removes
  /// it.
  void setOutputCallback(std::function<void(int)> cb) { _output_callback = std::move(cb); }

  // ---- post-processing --------------------------------------------------------------------
  /// The flow of the flux of the equation of @p var out of the body through @p boundary, node by
  /// node: at a prescribed value the reaction (what the condition supplies, taken out of the
  /// body), and elsewhere the share of the node in the integral of the boundary conditions.  A
  /// node on two boundaries of prescribed values carries the reaction of both; a node that only
  /// another boundary prescribes carries none here.  With @p owned,
  /// the reaction counts only at the nodes it marks (the part of a distributed problem that owns
  /// them).  Available after a solve.
  std::vector<std::pair<Index, double>>
  boundaryOutflow(const std::string & var,
                  const std::string & boundary,
                  const std::vector<char> * owned = nullptr) const;
  /// The sum of boundaryOutflow() over the nodes of @p boundary.
  double totalOutflow(const std::string & var, const std::string & boundary) const;
  /// Interpolate a variable at arbitrary points (NaN outside the mesh).
  std::vector<double> sample(const std::string & var, const std::vector<Point> & points) const;
  /// The gradient of a variable at arbitrary points, in the element that holds each point: NaN outside
  /// the mesh, and on a side between elements the gradient of one of them.
  std::vector<Point> sampleGradient(const std::string & var,
                                    const std::vector<Point> & points) const;
  /// Gradient of a variable at element centroids (rows: elements).
  std::vector<Point> gradientAtCentroids(const std::string & var) const;
  /// An error indicator, one value per element, from gradient recovery.
  ///
  /// The gradient of the computed solution is discontinuous between elements.
  /// A smoother gradient is recovered by averaging the element gradients onto
  /// the nodes, weighted by the part of each element that belongs to the
  /// node's control domain, and interpolating that nodal field back over the
  /// element.  The indicator of an element is the square root of the integral
  /// over it of the squared difference between the two gradients,
  ///
  ///     eta_e^2 = \int_e |grad u_recovered - grad u_h|^2 dV .
  ///
  /// The idea is due to Zienkiewicz and Zhu, "A simple error estimator and
  /// adaptive procedure for practical engineering analysis", International
  /// Journal for Numerical Methods in Engineering 24 (1987) 337-357, where it
  /// is shown that the recovered gradient is more accurate than the computed
  /// one, so that their difference estimates the error in the computed one.
  /// It is an indicator, not a bound: it says which elements carry most of the
  /// error, which is what an adaptive loop needs, and it does not certify the
  /// size of the error.
  std::vector<double> errorIndicator(const std::string & var) const;

  /// The error of a computed field against a known exact solution, in the two
  /// norms that convergence theory is stated in:
  ///
  ///     L2          ( \int |u_h - u|^2 dV )^(1/2)
  ///     H1 seminorm ( \int |grad u_h - grad u|^2 dV )^(1/2)
  ///
  /// u_h is the finite-dimensional field the method actually represents: the
  /// element interpolation of the nodal values.  The integrals are evaluated with a
  /// Gauss rule of @p quadrature_points per direction on every element patch
  /// (zero chooses the polynomial order plus four), with the coordinate factor
  /// of the problem, so an axisymmetric error is the error of the body of
  /// revolution.  The H1 seminorm is computed only when all three gradient
  /// components are supplied; unused components of a lower-dimensional
  /// problem may be given as zero functions.
  struct ErrorNorms
  {
    double l2 = 0.0;
    double h1_seminorm = 0.0;
  };
  ErrorNorms errorNorms(const std::string & var,
                        const Function & exact,
                        const std::array<const Function *, 3> & exact_gradient,
                        int quadrature_points = 0) const;
  /// A property evaluated at element centroids (rows: elements).
  std::vector<std::vector<double>> propertyAtCentroids(const std::string & property) const;
  /// Flux of a kernel evaluated at element centroids.
  std::vector<Point> kernelFluxAtCentroids(const std::string & kernel) const;
  /// The stress at the height @p z above the mid-surface, at the element centroids, of the
  /// structural kernel @p kernel (Kernel::sectionStress), one row per element.
  std::vector<std::vector<double>> sectionStressAtCentroids(const std::string & kernel,
                                                            double z) const;
  /// Call @p visit with the context of every element centroid (the solution and the properties
  /// evaluated there) and the block of the element.
  void centroidContexts(const std::function<void(QpContext &, int)> & visit) const;
  /// The integral over the blocks of variable @p v of the shape function of each node (partial
  /// at the nodes that processes share).
  Vector nodeWeights(int v) const;
  /// Integral of a variable over the domain (with the coordinate factor).
  double integrate(const std::string & var) const;
  /// The integral of a variable over every element, with the coordinate factor.  An empty name gives
  /// the volume of every element.
  std::vector<double> elementIntegrals(const std::string & var) const;
  /// Measure of the named side sets: their total length in two dimensions and
  /// area in three, with the coordinate factor of axisymmetric and spherical
  /// problems.  In a distributed run it is the measure on the whole mesh.
  double boundaryArea(const std::vector<std::string> & boundaries) const;
  /// Measures of side sets of the whole mesh, given by a distributed problem
  /// to its rank-local problem, whose mesh holds only part of each side set.
  void setGlobalBoundaryAreas(std::map<std::string, double> measures)
  {
    _global_boundary_areas = std::move(measures);
  }
  /// The J-integral and the stress intensity factors K_I and K_II of a crack tip in two
  /// dimensions, by the domain integral and the interaction integral over the ring between the
  /// two radii; the crack lies along @p angle (radians from x), its faces behind the tip.
  std::array<double, 3> crackTipIntegrals(const std::vector<std::string> & displacements,
                                          const Point & tip,
                                          double angle,
                                          double inner_radius,
                                          double outer_radius,
                                          double youngs_modulus,
                                          double poissons_ratio,
                                          bool plane_strain) const;
  /// The integral of u . n over a side set, for the vector u of the variables @p components.
  double normalIntegral(const std::vector<std::string> & components,
                        const std::string & boundary) const;
  /// Integral of the normal flux of a kernel over a side set.
  double boundaryFluxIntegral(const std::string & kernel, const std::string & boundary) const;
  /// Write a VTK unstructured-grid file with nodal fields and cell data.
  /// @p fields selects the variables (empty: all the variables).
  void writeVTU(const std::string & filename,
                const std::vector<std::string> & cell_properties = {},
                const std::vector<std::string> & fields = {}) const;
  /// Set the SI unit of a variable.
  void setVariableUnit(const std::string & name, const std::string & unit);

  // ---- adjoint sensitivities -------------------------------------------------
  /// An objective functional of the solution: the integral of a variable over
  /// the whole domain (an empty boundary) or over one side set, with the
  /// coordinate factor of the problem.
  struct Objective
  {
    std::string variable;
    std::string boundary; ///< empty: the whole domain
  };
  /// The gradient d g / d U of an objective, one entry per degree of freedom.
  /// For the integral of a variable it is the vector of the integrals of its
  /// shape functions, which the adjoint solve turns into one sensitivity per
  /// parameter.
  Vector objectiveGradient(const Objective & objective) const;
  /// The value of an objective at the current solution.
  double objectiveValue(const Objective & objective) const;
  /// d g / d p of every named parameter, by one adjoint solve.  The
  /// parameters are settable constants registered by name (addParameter);
  /// the problem must hold a converged solution, which solve() leaves.
  ///
  /// With the solution U(p) defined by R(U, p) = 0, the chain rule gives
  /// dg/dp = (dg/dU)(dU/dp), and eliminating dU/dp with the adjoint lambda,
  ///
  ///     J^T lambda = (dg/dU)^T ,   dg/dp = - lambda . dR/dp ,
  ///
  /// in which J is the exact Jacobian of the converged solution (one adjoint
  /// solve, by the same linear solvers as the primal) and dR/dp is a central
  /// difference of the residual at the converged solution: one assembly per
  /// parameter, no solve.  The prescribed degrees of freedom carry their own
  /// dU/dp, which enters through both terms above.  The cost is independent
  /// of the number of unknowns beyond one adjoint solve, and linear in the
  /// number of parameters.
  ///
  /// The state (history of stateful properties) and periodic constraints are
  /// not differentiated yet: a problem with periodic boundary conditions is
  /// refused.
  std::map<std::string, double> sensitivities(const Objective & objective,
                                              const std::vector<std::string> & parameters,
                                              const SolverOptions & options = {});
  /// Register (or replace) a named settable constant: a parameter that
  /// sensitivities() differentiates with respect to.  Give its name wherever
  /// a coefficient, a source or a boundary value accepts a function.
  void addParameter(const std::string & name, double value);
  /// Assign a registered parameter (its value is read by the objects that use
  /// it at the next assembly; call solve() for the new solution).
  void setParameter(const std::string & name, double value);
  /// The value of a registered parameter.
  double parameterValue(const std::string & name) const;

  std::string summary() const;

private:
  /// Create a symmetry_boundary_condition on the displacement normal to its plane.
  std::shared_ptr<Object> addSymmetryCondition(const std::string & name, InputParameters params);
  std::map<std::string, double> _global_boundary_areas;
  /// Everything one thread needs to assemble one element or one face, so that
  /// nothing is shared between threads except the read-only problem
  /// definition, the solution vectors, and the global residual, which is
  /// updated atomically.
  struct ThreadScratch
  {
    std::vector<double> cur, lag, old, rate_shift, accel_shift;
    std::vector<Index> dofs;
    /// Node-based methods: the columns of a face flux that depends on the gradients at the face's
    /// nodes, the element's degrees of freedom and then those of the nodes around the two.
    std::vector<Index> columns;
    std::vector<ADReal> Rloc;
    std::vector<char> touched;
    /// The size of the terms of each entry of Rloc (AssemblyOptions::term_size).
    std::vector<double> sizes;
    std::vector<IntegrationPoint> pts;
    /// The points of `pts` mapped to the element, when the rule provides them.
    std::vector<MappedPoint> mapped;
    std::vector<const Kernel *> kernels;
    MappedPoint geo, fld;
    QpContext ctx;
    std::vector<Eigen::Triplet<double>> triplets;
  };
  struct Group
  {
    QuadratureSpec spec;
    std::vector<const Kernel *> kernels;
  };
  void buildGroups();
  void fillContext(QpContext & ctx,
                   const Element & el,
                   const MappedPoint & field_point,
                   const std::vector<double> & cur_local,
                   const std::vector<double> * lag_local,
                   const std::vector<double> * old_local,
                   int num_derivatives) const;
  /// The shifts of the Newmark time derivatives at a point (AssemblyOptions::rate_shift).
  void fillShifts(QpContext & ctx,
                  const Element & el,
                  const MappedPoint & field_point,
                  const std::vector<double> & rate_local,
                  const std::vector<double> & accel_local) const;
  void evaluateProperties(QpContext & ctx) const;
  /// One integration point of an interface condition, paired with the
  /// closest point on the other side of the gap.
  struct InterfacePoint
  {
    Index element = -1; ///< secondary element
    int side = -1;      ///< its local side
    Point xi{0, 0, 0};  ///< secondary parent reference coordinates
    Point x{0, 0, 0};   ///< the paired point
    int owner = -1;     ///< dual mesh: secondary local node owning the point
  };
  /// Pair every integration point of every interface condition (called by
  /// initialize()).  Indexed [condition][primary side or face][point].
  void buildInterfacePairs();
  void assembleInterface(const InterfaceBC & bc,
                         const Vector & U,
                         const AssemblyOptions & opts,
                         double wfac,
                         Vector & R,
                         std::vector<Eigen::Triplet<double>> * triplets,
                         ThreadScratch & scratch);
  double coordFactor(const Point & x) const;
  void markActiveDofs();
  /// Refuse an element whose Jacobian determinant is negative at its centroid
  /// or at any of its nodes: an inverted element would be integrated with a
  /// positive measure and give a wrong answer silently.  Surfaces and curves
  /// in space are exempt (their measure is a square root); their orientation
  /// is a convention that fixOrientation sets.
  void checkMeshInversions();
  /// Refuse, in a serial run, a boundary side that no side set holds, through which an advective
  /// flux of a control volume method would not leave the domain.
  void checkAdvectiveBoundary() const;
  /// Whether the problem has the inertia of a flow and its velocity is at rest: zero at every
  /// degree of freedom that no condition prescribes, on every process.
  bool flowAtRest() const;
  /// The weight of the steady term @p object in the assembly @p options: theta at the new time
  /// of a step of the theta method and 1 - theta at the old one (options.theta), but 1 and 0 for
  /// a term of an equation without a time derivative or a term atNewTime(), which a step takes
  /// at the new time only; times the data fraction of a continuation if the term is data
  /// (Object::isLoad).
  double steadyWeight(const ResidualObject & object, const AssemblyOptions & options) const;
  /// QpContext::time_weight for the kernel @p kernel: the inverse of its steady weight (without
  /// the data fraction) in a transient step, so that a time derivative inside its residual enters
  /// the step with the weight 1, and 0 where there is no time derivative to weight.
  double timeWeight(const Kernel & kernel, const AssemblyOptions & options) const;
  /// Whether the equation of each variable has no time kernel (Problem::steadyWeight).
  std::vector<char> _new_time_only;
  /// Node-based methods with a second-order advection scheme: the least-squares gradient at
  /// every node, grad u_a = sum_b w_ab (u_b - u_a) over the nodes b of the elements around a,
  /// with the weights w_ab = G_a^-1 (x_b - x_a), G_a = sum_b (x_b - x_a)(x_b - x_a)^T, which is
  /// exact for a linear field.  CSR layout by node.
  void buildNodeGradients();
  std::vector<Index> _gradient_offsets, _gradient_nodes;
  std::vector<Point> _gradient_weights;
  /// Add to the F_p of @p blocks the Robin condition at the inflow of the flow whose mass balance
  /// is @p constraint (see SaddlePointBlocks.h).
  void addInflowRobin(const Kernel & constraint, double theta, SaddlePointBlocks & blocks) const;
  /// Throw an InputError when the Jacobian is singular because a part of the
  /// mesh is not held: a group of coupled unknowns, cut off from every
  /// prescribed value, on which a vector of the near null space (a constant,
  /// a rigid body motion) is a solution of the homogeneous problem, e.g., a
  /// body without supports or a region without any temperature condition.
  void checkSingularity(const SparseMatrix & J) const;
  /// The pressure of an incompressible flow whose boundary prescribes no pressure (an enclosed
  /// flow) is determined only up to a constant, which the Jacobian @p J shows as a constant
  /// pressure that it maps to zero.  Hold one unknown of each such pressure at its value, the
  /// same unknown on every process, and return whether one was added.
  bool holdPressureLevels(const SparseMatrix & J);
  /// Shift each held pressure of @p U to a mean of zero over its fluid, and return the shifts.
  std::vector<double> shiftPressureLevels(Vector & U) const;
  /// Add to each held pressure of @p U a test shift of the size of the pressure in its equations:
  /// the larger of its largest value and, for each equation, the reactions of that equation
  /// @p forces over the size of the constant pressure in it.  Return the shifts.
  std::vector<double> testShiftPressureLevels(Vector & U, const std::vector<double> & forces) const;
  /// The size, in each equation, of the change that the pressure shifts @p shifts make: the sum
  /// over the held pressures of |shift| times the size of the constant pressure in that equation.
  std::vector<double> pressureLevelChange(const std::vector<double> & shifts) const;
  /// Take the size of the mass residual at the start of a solve as the scale of the boundary flux.
  void notePressureLevelFlux();
  /// Warn once when the mass equation of a held unknown, at the converged @p R, carries a net
  /// flux of the boundary: data that an incompressible flow cannot satisfy.
  void checkPressureLevelBalance(const Vector & R);
  /// The steady terms of a theta step at its old time, which every residual of the step adds:
  /// their sum, and the sum of their sizes for the reference residual.
  struct OldResidual
  {
    Vector residual, reference, term_size;
  };
  /// Write the owned rows @p owned, in the order of the global numbering @p rows, into @p v, a
  /// vector of the local unknowns, and give the other unknowns the values of their owners.
  void setOwnedPart(Vector & v, const Vector & owned, const GlobalRows & rows) const;
  /// The refusal of an enclosed flow whose pressure level changes the flow.
  static const char * enclosedLevelMessage();
  /// With @p ramp_from, a solve that fails is taken again by continuation: the data are ramped
  /// up from zero, and the prescribed values from those of @p ramp_from.
  SolveResult nonlinearSolve(const SolverOptions & options,
                             AssemblyOptions base,
                             const OldResidual * steady_old,
                             const Vector * ramp_from = nullptr);
  /// Advance the solution from @p old by one step of size @p dt, leaving the
  /// result in solution().  Used by every time stepper.
  SolveResult takeTimeStep(const Vector & old,
                           double dt,
                           const TransientOptions & transient,
                           const SolverOptions & options);
  /// A step taken: the state it started from and the state it reached, its size and its theta.
  /// The old time of the next step finds there the step that reached its old state, whose residual
  /// gives the velocity that the mass equation of that step conserved.  A step may be retried or
  /// split, so the last few are kept.
  struct StepRecord
  {
    Vector start, reached;
    double dt = 0.0;
    double theta = 1.0;
  };
  std::vector<StepRecord> _step_records;
  /// The velocity and the acceleration of the states that steps of the generalized-alpha method
  /// reached, by their solution (the last few), and the initial velocity of a run.
  struct DynamicState
  {
    Vector u, v, a;
  };
  std::vector<DynamicState> _dynamic_states;
  Vector _initial_velocity;
  /// The acceleration at the state @p u with the velocity @p v: M a = -R(u, v), and the second
  /// derivative of the prescribed values in time.  The prescribed values also set @p v.
  Vector
  initialAcceleration(const Vector & u, Vector & v, double dt, const SolverOptions & options);

  std::shared_ptr<Mesh> _mesh;
  Method _method;
  CoordinateSystem _coord;
  int _num_threads = 0;
  bool _thread_safe = true;
  std::vector<char> _owned_node;
  std::vector<Variable> _vars;
  /// See hasMixedOrder() and localDofIndex(): whether the layout is mixed,
  /// the position of every mesh-order variable among the mesh-order variables
  /// (-1 for a first-order variable), and their number.
  bool _mixed_order = false;
  std::vector<int> _full_order_rank;
  int _num_full_order = 0;
  /// Recompute the three members above from the variables and the mesh.
  void buildOrderLayout();
  std::map<std::string, FunctionPtr> _functions;
  std::vector<std::shared_ptr<Kernel>> _kernels;
  std::vector<std::shared_ptr<IntegratedBC>> _ibcs;
  std::vector<std::shared_ptr<NodalBC>> _nbcs;
  std::vector<std::shared_ptr<NodalLoad>> _loads;
  std::vector<std::shared_ptr<Constraint>> _constraints;
  std::vector<Index> _primary_of;
  /// The hanging degrees of freedom of a mesh with hanging nodes (Mesh::nodeConstraints): each one
  /// a weighted sum of independent degrees of freedom, empty for the others.
  std::vector<std::vector<std::pair<Index, double>>> _hanging_of;
  Index _num_integrated_elements = -1;
  std::map<std::string, std::vector<Side>> _interface_candidates;
  bool _constraints_outside = false;
  std::vector<char> _extra_constrained;
  /// The two parts of buildConstraints: the copies of periodic boundaries, and the hanging nodes.
  void buildPeriodicConstraints();
  void buildHangingConstraints();
  /// The dual mesh method keeps a hanging node as an unknown of its own and splits a cell with
  /// hanging nodes on its sides as a polygon (PolygonCell): the polygon of each such element, by
  /// element (-1 for the others).
  void buildPolygonCells();
  std::vector<int> _polygon_of;
  std::vector<PolygonCell> _polygon_cells;
  /// Resolve the pairs of the constraints into one primary per dependent degree of freedom, following chains (a corner node periodic in two directions).
  void buildConstraints();
  void interpolateFirstOrderVariables(Vector & U) const;
  std::vector<std::shared_ptr<Property>> _property_objects;
  std::map<std::string, std::shared_ptr<Object>> _by_name;
  PropertyRegistry _props;
  std::vector<Group> _groups;
  std::map<const IntegratedBC *, std::vector<std::vector<InterfacePoint>>> _interface_pairs;
  std::vector<char> _active_dof;
  bool _initialized = false;
  /// "variable on boundary..." of every Dirichlet condition, to refuse a second one.
  std::set<std::string> _dirichlet_keys;
  /// The time step and the weight of the steady terms of the equations whose
  /// Jacobian the next linear solve takes (zero and one for a steady solve),
  /// which the operators of the flow preconditioners need.
  mutable bool _advective_boundary_checked = false;
  double _linear_dt = 0.0;
  double _linear_theta = 1.0;
  /// The processes that share the problem; null for the serial identity.
  const Parallel * _parallel = nullptr;
  mutable bool _singularity_checked = false;
  /// The matrix whose singularity was checked last, by its shape: the check
  /// runs again when the pattern of the Jacobian changes (constraints that
  /// appear, an interface that starts to couple two bodies).
  Index _singularity_pattern_rows = -1;
  Index _singularity_pattern_nonzeros = -1;
  /// A pressure whose level is free: the variable, the unknown held for it (-1 on a process
  /// that does not have it), the norm of |J| times its constant mode, the largest norm of its
  /// mass residual at the start of a solve (the scale of the boundary flux), the weights of its
  /// mean (nodeWeights), and whether the net flux was reported.
  struct PressureLevel
  {
    int variable = -1;
    Index dof = -1;
    /// The size of the constant pressure in each equation, the norm of |J| 1_p over its unknowns.
    std::vector<double> scales;
    double flux = 0.0;
    Vector weight;
    bool warned = false;
  };
  bool _levels_checked = false;
  std::vector<PressureLevel> _levels;
  /// Whether a kernel reads the Laplacian of the variables (Kernel::needsLaplacian).
  bool _needs_laplacian = false;
  Vector _U;
  Vector _last_residual;
  std::vector<char> _second_order_variables;
  /// The state a continuation of a steady solve started from, from which the prescribed values
  /// are ramped (applyDirichlet); empty outside a continuation.
  Vector _ramp_from;
  double _time = 0.0;
  double _eigenvalue = 0.0;
  std::function<void(double, Problem &)> _step_callback;
  std::shared_ptr<TimeIntegrator> _time_integrator;
  std::function<void(int)> _output_callback;
  std::map<std::string, std::vector<double>> _element_fields;

  /// The history of one owner (an element): the keys
  /// of its points and, for each, a record in three buffers.  Only the
  /// thread that assembles an owner touches its entry.
  struct OwnerState
  {
    std::vector<std::uint64_t> keys;
    std::vector<double> start, trial, committed;
  };
  int _state_record = 0;
  mutable std::vector<OwnerState> _state;
  /// Point the context at the history of its point (inserting a zero record
  /// the first time), or clear the pointers if the point has none.
  void bindState(QpContext & ctx) const;
};

} // namespace dualmesh
