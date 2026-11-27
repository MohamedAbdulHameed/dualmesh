// SPDX-License-Identifier: LGPL-2.1-or-later
//
// Problem: owns the mesh, the variables (fields), and all objects, assembles
// residuals and Jacobians with either the dual mesh control domain method or
// the finite element method, and runs steady or transient executioners.
#pragma once

#include "dualmesh/base/Factory.h"
#include "dualmesh/base/FlowSystem.h"
#include "dualmesh/base/Kernel.h"
#include "dualmesh/base/Property.h"
#include "dualmesh/base/SolidSystem.h"
#include "dualmesh/core/Coordinates.h"
#include "dualmesh/fe/Assembly.h"
#include "dualmesh/mesh/Mesh.h"

#include <Eigen/Sparse>

#include <algorithm>
#include <array>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
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
  /// The relaxation gamma, from 0 to below 1, of an iteration without a line search:
  /// U = (1 - gamma) U^{r+1} + gamma U^r, from the second step on.
  double relaxation = 0.0;
  /// Newton only: "none" takes the full Newton step; "backtracking" halves
  /// the step until the residual norm falls, ||R(U + a dU)|| <= (1 - 1e-4 a)
  /// ||R(U)||, at most max_line_search_steps times (the Armijo condition on
  /// the residual norm).  It costs one residual assembly per trial.
  /// "automatic", the default, is backtracking when a kernel needs it
  /// (AdvectiveKernel::needsLineSearch, the limiter of an advection scheme, on whose
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
  /// The scheme of an equation with a second time derivative (StorageKernel::isSecondOrder):
  /// "generalized_alpha" (Chung and Hulbert 1993) with the spectral radius rho_infinity at
  /// infinite frequency, or "Newmark" (the average acceleration rule, without numerical
  /// damping).  The first-order equations of the same problem take theta = 1 - alpha_f.
  std::string second_order_scheme = "generalized_alpha";
  double spectral_radius = 0.75;
  /// The times at which the output callback (Problem::setOutputCallback) is called, in
  /// increasing order.  The steps land on these times.  Empty: no call.
  std::vector<double> output_times;
  /// The file of the checkpoint, written every checkpoint_interval accepted steps of the run and
  /// after its last step, each time followed by the callback of Problem::setCheckpointCallback.
  /// Empty: no checkpoint.
  std::string checkpoint_file;
  int checkpoint_interval = 0;
  /// A checkpoint to continue from: the run starts at its time, with its step size, its solution and the history of its material.
  std::string restart_file;

  // ---- adaptive time stepping --------------------------------------------
  /// How the step size is chosen.
  ///
  /// "error", the default, controls the local truncation error.  The theta method estimates it
  /// from one step and the polynomial through the last accepted states extrapolated to the end
  /// of the step (linear for an order-one member, quadratic for Crank-Nicolson): their
  /// difference is in proportion to the error.  The projection integration of a flow compares
  /// its step with the extrapolation of its own order (TimeIntegrator::predict).  The first
  /// steps and an equation with a second time derivative estimate it by step doubling instead:
  /// the same interval is advanced once with one step and once with two half steps, and the
  /// solution of the half steps is kept, at three nonlinear solves per step.  Each variable is
  /// measured against the largest of its range, its change from the start of the run and the
  /// square root of the machine epsilon times its magnitude, the largest each has reached in the
  /// run, with the components of a vector unknown together.  A step whose estimated error
  /// exceeds `error_tolerance` is thrown away and retried with a smaller step.  An accepted step
  /// is followed by the largest step the estimate says will still meet the tolerance.
  ///
  /// "fixed" keeps `dt` throughout, except that the last step is shortened to
  /// land exactly on `end_time`.
  ///
  /// "iteration" watches the nonlinear solver instead: a step that converged
  /// in fewer than `optimal_iterations - iteration_window` iterations is
  /// followed by a larger one, a step that needed more than
  /// `optimal_iterations + iteration_window` by a smaller one, and a step that
  /// did not converge at all is thrown away and retried.  It costs nothing
  /// beyond the solve itself and is the right choice when the difficulty is
  /// the nonlinearity rather than the accuracy.
  std::string time_stepper = "error";
  /// Smallest step allowed; a step that would have to go below it is an error.
  /// Zero means `dt` divided by one million.
  double dt_min = 0.0;
  /// Largest step allowed.  Zero means the whole time interval.
  double dt_max = 0.0;
  /// The most the step may grow from one step to the next.
  double growth_factor = 2.0;
  /// The factor applied to a step whose nonlinear solve failed, or that the "iteration" stepper
  /// shrinks.  A step that the "error" stepper rejects takes the step that its estimate allows.
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
  /// The largest change of the velocity of a flow over its largest speed, when a transient run
  /// projected the initial velocity onto a field without divergence
  /// (FlowSystem::projectInitialVelocity), and empty when it did not.
  std::optional<double> start_correction;
  std::vector<IterationRecord> history;
};

/// A time integration that takes the place of the theta method in a transient solve.
/// The transient loop stays (the output times, the checkpoints, the callbacks), and the integrator advances the solution by one step.
class TimeIntegrator
{
public:
  virtual ~TimeIntegrator() = default;
  /// Called once before the first step, with the solution at the start time.
  virtual void start(const Vector & solution, double time) = 0;
  /// Advance @p solution from @p old, at time - dt, to @p time.  @p old is the last accepted
  /// solution, or, when @p continues, the solution of the step just taken (the second half of a
  /// doubled step).
  virtual SolveResult
  step(Vector & solution, const Vector & old, double time, double dt, bool continues) = 0;
  /// The step to @p time was accepted with @p solution.
  virtual void accept(const Vector & solution, double time) = 0;
  /// The order of accuracy in time of a step, which the error estimate and the choice of the
  /// next step take.
  virtual int order() const = 0;
  /// The predictor of the step @p dt from the last accepted solution, and the factor that turns
  /// the difference of the step from it into the estimate of its local error, as runTransient
  /// takes them.  False when too few states are accepted for it, or for an integration without
  /// one: the step is then doubled.
  virtual bool predict(double /*dt*/, Vector & /*predicted*/, double & /*factor*/) const
  {
    return false;
  }
  /// Whether the integration measures the Courant number of a step, which the cfl stepper takes.
  virtual bool measuresCourantNumber() const { return false; }
  /// The Courant number of a step @p dt at @p solution, on every process alike.
  virtual double courantNumber(const Vector & /*solution*/, double /*dt*/) const { return 0.0; }
  /// The largest step at @p solution whose discretization does not depend on the step, on every
  /// process alike, which the adaptive steppers keep to. Infinite for an integration whose
  /// discretization never depends on its step.
  virtual double largestStep(const Vector & /*solution*/) const
  {
    return std::numeric_limits<double>::infinity();
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

/// Disjoint sets of the indices 0, 1, 2, ... (a union-find with path halving).  The root of a set
/// is its smallest member.
class DisjointSet
{
public:
  /// The sets {0}, ..., {n - 1}.
  explicit DisjointSet(Index n = 0) : _parent(static_cast<std::size_t>(n))
  {
    for (Index i = 0; i < n; ++i)
      _parent[i] = i;
  }
  /// Add the set {i} of the next index i, and return i.
  Index add()
  {
    _parent.push_back(static_cast<Index>(_parent.size()));
    return _parent.back();
  }
  /// The root of the set of @p i.
  Index root(Index i)
  {
    while (_parent[i] != i)
    {
      _parent[i] = _parent[_parent[i]];
      i = _parent[i];
    }
    return i;
  }
  /// Join the sets of @p a and @p b.
  void join(Index a, Index b)
  {
    a = root(a);
    b = root(b);
    if (a != b)
      _parent[std::max(a, b)] = std::min(a, b);
  }

private:
  std::vector<Index> _parent;
};

/// Join the ids of every pair (secondary, primary) into groups (a union-find), so that an id
/// joined in several directions, such as a corner of a doubly periodic domain, ends in one group.
/// Returns, for every joined id other than its group's representative, the representative: the
/// smallest member that is a primary only.  A group whose every member is a secondary was joined
/// in a loop, which is an InputError.
std::unordered_map<Index, Index>
joinedRepresentatives(const std::vector<std::pair<Index, Index>> & pairs);

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
  /// The number of local node @p node in the numbering of the nodes of all processes.
  virtual Index globalNode(Index node) const { return node; }
  /// Add to @p values, @p width per local node (the unknowns of a node, or a field), the values
  /// of the other processes at the nodes they share.
  virtual void addAtNodes(double * /*values*/, int /*width*/) const {}
  /// Make every value of @p values, @p width per local node, at a node another process owns
  /// that of the owner.
  virtual void copyFromOwners(double * /*values*/, int /*width*/) const {}
  /// After the prescribed values are written into @p solution: make them, and every value of
  /// a node another process owns, those of the owner.
  virtual void shareDirichlet(Vector &) const {}
  /// The global numbering of the unknowns of @p problem.
  virtual GlobalRows globalRows(const Problem & problem) const;
  /// Whether the singularity check of the Jacobian runs, which needs the whole matrix.
  virtual bool checksSingularity() const { return true; }
  /// The global numbers of the nodes and elements of @p problem, for a checkpoint.
  virtual CheckpointLayout checkpointLayout(const Problem & problem) const;
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
  double relativeStep(const Vector & after, const Vector & before) const;
  /// The sizes of each variable of @p state over its active unknowns, on every process.
  struct VariableSizes
  {
    /// The largest value less the smallest.
    std::vector<double> range;
    /// The largest difference from the reference state.
    std::vector<double> change;
    /// The largest magnitude.
    std::vector<double> magnitude;
  };
  VariableSizes variableSizes(const Vector & state, const Vector & reference) const;
  /// The equation of each variable: the components of a vector unknown, a velocity or a
  /// displacement, which a kernel of one of them names in its list 'velocities', 'velocity' or
  /// 'displacements', share the number of the first of them, and any other variable has its own.
  std::vector<int> vectorEquations() const;
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
  /// Degrees of freedom to treat as prescribed besides the problem's own: the distributed solver sets the copies of prescribed nodes that only another process knows as prescribed.
  void setExtraConstrainedDofs(std::vector<char> mask) { _extra_constrained = std::move(mask); }
  /// The residual from which reactions are read, with the equations of the dependent unknowns folded into their primaries before the processes added their parts (the projection's momentum residual).
  void setLastResidual(const Vector & R) { _last_residual = _last_folded_residual = R; }
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
  /// The factor of the volume element of the coordinate system of the problem at @p x (see
  /// coordinateFactor).
  double coordFactor(const Point & x) const { return coordinateFactor(_coord, x); }
  /// The value of the symbol eigenvalue that the expressions read; an eigenvalue study sets it.
  void setEigenvalue(double value) { _eigenvalue = value; }
  double eigenvalue() const { return _eigenvalue; }
  /// Whether the assembly is that of an eigenvalue study, whose kernels take their rule of that
  /// study (StorageKernel::eigenvalueStudyQuadrature): a storage term takes its consistent rule
  /// there.
  void setEigenvalueStudy(bool on);
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
  /// Set every dependent degree of freedom (_dependent_of) to the weighted sum of the independent ones it depends on.
  void updateDependentDofs(Vector & U) const;

  /// The mass conservation equation of a flow: the one kernel that provides
  /// the flow coefficients, or null.
  const MassEquation * flowConstraint() const;
  /// The flow system of the problem (FlowSystem.h), which the fluid module provides.
  FlowSystem & flowSystem() { return *_flow; }
  const FlowSystem & flowSystem() const { return *_flow; }
  /// The solid system of the problem (SolidSystem.h), which the solid mechanics module provides.
  const SolidSystem & solidSystem() const { return *_solid; }
  /// The time step and the weight of the steady terms of the equations whose Jacobian the next
  /// linear solve takes (zero and one for a steady solve), which the operators of the flow
  /// preconditioners need.
  double linearTimeStep() const { return _linear_dt; }
  double linearSteadyWeight() const { return _linear_theta; }

  /// Number of threads requested for the assembly loops; 0 means the OpenMP
  /// default (the value of OMP_NUM_THREADS, or the number of cores).
  void setNumThreads(int n) { _num_threads = n; }
  int numThreads() const { return _num_threads; }
  /// The number of threads that will actually be used.  This is 1 when the
  /// library was built without OpenMP, and 1 when any object of the problem is
  /// defined in Python, because calling back into the interpreter requires the
  /// global interpreter lock.  It sets the problem up first (threadSafe).
  int effectiveThreads();
  /// Whether every object of the problem can be called from several threads.
  /// The answer depends on the objects that have been added, so the problem is
  /// set up first (initialize) if that has not happened yet.
  bool threadSafe()
  {
    initialize();
    return _thread_safe;
  }
  /// Fewer elements than this per thread and the assembly runs serially,
  /// because starting a team of threads would cost more than it saves.
  static constexpr Index kMinElementsPerThread = 400;
  PropertyRegistry & propertyRegistry() { return _props; }
  const PropertyRegistry & propertyRegistry() const { return _props; }
  std::shared_ptr<Object> object(const std::string & name) const;
  const std::vector<std::shared_ptr<Constraint>> & constraints() const { return _constraints; }
  /// The property objects, in the order they were added.
  const std::vector<std::shared_ptr<Property>> & properties() const { return _property_objects; }
  /// Whether degree of freedom @p d depends on others (_dependent_of).
  bool isDependent(Index d) const
  {
    return !_dependent_of.empty() && !_dependent_of[static_cast<std::size_t>(d)].empty();
  }
  /// The degrees of freedom that a nodal boundary condition prescribes.
  std::vector<char> prescribedDofs() const;
  /// Under DMCDM, the polygon (polyhedron) of element @p e when nodes hang on its sides, and null
  /// for every other element.
  const PolygonCell * polygonCell(Index e) const
  {
    return _polygon_of.empty() || _polygon_of[static_cast<std::size_t>(e)] < 0
               ? nullptr
               : &_polygon_cells[static_cast<std::size_t>(
                     _polygon_of[static_cast<std::size_t>(e)])];
  }
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
  /// What one assembly (assemble) takes, in five groups.  The states it reads (the solution,
  /// lagged, old and the two shifts) have one entry per degree of freedom, and the vectors it
  /// writes (the residual, reference and term_size) are distinct from each other and from the
  /// states.  assemble checks these rules, and those of the previous step and of the second time
  /// derivatives, before it starts.
  struct AssemblyOptions
  {
    // ---- the terms to assemble ----
    bool include_steady_terms = true;
    bool include_time_kernels = true;
    /// When given, the assembly takes these kernels and integrated boundary conditions in place
    /// of the problem's, and leaves out its advective boundary fluxes and its nodal loads: the
    /// operators of some terms (the time integration of a flow), with objects that the problem
    /// does not hold once they are set up with it.
    const std::vector<const Kernel *> * kernels = nullptr;
    const std::vector<const IntegratedBC *> * conditions = nullptr;
    /// The fraction of the driving data, which multiplies every term that is data
    /// (Object::isLoad): 1 for the problem itself, less in a step of a continuation.
    double data_fraction = 1.0;
    double inertia = 1.0; ///< the factor of the inertia of a flow (QpContext::inertia)
    /// Whether the mass equation of a flow holds its pressure stabilization
    /// (QpContext::stabilization).
    bool stabilization = true;

    // ---- the linearization ----
    LinearizationMode mode = LinearizationMode::Newton;
    /// The previous iterate, whose values Picard iteration takes in the coefficients.  Without
    /// it, Picard iteration takes the current state.
    const Vector * lagged = nullptr;

    // ---- the time step ----
    double dt = 0.0;
    double theta = 1.0;           ///< weight of steady terms
    const Vector * old = nullptr; ///< previous time step (transient)
    /// The old time of a step of the theta method, whose steady terms are weighted by 1 - theta.
    bool old_level = false;
    /// At the old time: @p old holds the state that the previous step started from, and dt and
    /// previous_theta are those of that step (QpContext::previous_step).  It needs old_level and
    /// old.
    bool previous_step = false;
    double previous_theta = 1.0;

    // ---- the second time derivatives ----
    /// The time derivatives of the second-order variables (QpContext::rate, acceleration):
    /// du/dt = rate_scale u - rate_shift and d2u/dt2 = accel_scale u - accel_shift.  Without
    /// the shifts they are rate_scale u and accel_scale u (the operators of the modes).  The two
    /// shifts come together, and only with second_order.
    bool second_order = false;
    double rate_scale = 0.0, accel_scale = 0.0;
    const Vector * rate_shift = nullptr;
    const Vector * accel_shift = nullptr;

    // ---- the sizes of the terms, which the nonlinear solve asks for ----
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
  /// The terms and the state of an operator (operatorOf).
  struct OperatorOptions
  {
    /// The kernels and the integrated boundary conditions that the operator holds in place of
    /// the problem's (AssemblyOptions::kernels).  Without kernels it holds those of the problem.
    std::optional<std::vector<const Kernel *>> kernels;
    std::vector<const IntegratedBC *> conditions;
    /// Whether the operator holds the steady terms and the time kernels.
    bool steady_terms = true;
    bool time_kernels = true;
    /// Whether the mass equation of a flow holds its pressure stabilization.
    bool stabilization = true;
    /// Whether the Jacobian is taken at the current solution.  Otherwise it is taken at rest,
    /// with every unknown zero.
    bool at_solution = false;
    /// Whether the equation of every dependent unknown is folded into those of the unknowns it
    /// depends on (foldDependents).
    bool fold_dependents = true;
    /// When given, the assembly is that of an eigenvalue study (setEigenvalueStudy), with the
    /// symbol eigenvalue at this value.
    std::optional<double> eigenvalue;
    /// In a problem with second-order variables, the time derivative whose coefficient the time
    /// kernels give: 1 for c du/dt and 2 for c d2u/dt2.  Order 0 takes the time kernels as one
    /// step of size one assembles them.  Order 2 of a problem without second-order variables is
    /// the zero operator.
    int order = 0;
  };
  /// The Jacobian of the terms that @p options chooses, with the time kernels at a time step of
  /// one, so that a time kernel c (u - u_old) / dt contributes c and the operator of the terms
  /// linear in the unknowns is their matrix.  The rows of the prescribed degrees of freedom are
  /// not replaced (constrainedDofs marks them).  The problem must be initialised.
  SparseMatrix operatorOf(const OperatorOptions & options);
  /// The steady residual @p R and its Jacobian @p J at the current solution, exactly as one
  /// Newton step of solveSteady sees them: the prescribed values are written into the solution,
  /// and the rows of the prescribed degrees of freedom are replaced (dirichletRows).
  void linearSystem(Vector & R, SparseMatrix & J);
  /// The initial velocity of the second-order variables of a transient run (zero: at rest).
  void setInitialVelocity(const Vector & v) { _initial_velocity = v; }
  /// The velocity that the last transient run of the generalized-alpha or Newmark method
  /// reached (empty before one).
  Vector velocity() const;
  /// The variables of second-order equations (StorageKernel::isSecondOrder), one flag per
  /// variable.
  const std::vector<char> & secondOrderVariables() const { return _second_order_variables; }
  /// Whether any variable has a second-order equation.
  bool hasSecondOrderVariables() const;
  /// Assemble R(U) (and optionally dR/dU) without applying Dirichlet rows.
  void assemble(const Vector & U, const AssemblyOptions & opts, Vector & R, SparseMatrix * J);
  /// Write the prescribed values into @p U at the nodes of every essential boundary condition,
  /// set the dependent degrees of freedom, and give every process the values of the owners
  /// (Parallel::shareDirichlet).
  void applyDirichlet(Vector & U, double data_fraction) const;
  /// Replace the equations of the prescribed degrees of freedom: the residual
  /// entry becomes zero, and the row and the column of the Jacobian are
  /// dropped but for the diagonal, which takes the mean size of the diagonal
  /// of the equation of that variable (so the system has the same form in any
  /// units).  Dropping the column as
  /// well as the row keeps a symmetric problem symmetric, which is valid
  /// because the increment at a prescribed degree of freedom is zero.
  void dirichletRows(Vector & R, SparseMatrix * J) const;
  /// Move the equation of every dependent degree of freedom into those of the independent ones
  /// it depends on, with its weights: R becomes C^T R and J becomes C^T J C, with C the map from
  /// the independent degrees of freedom to all of them.  The rows and columns of the dependent
  /// ones are left empty, for the caller to give them the identity.
  void foldDependents(Vector & R, SparseMatrix * J) const;
  /// The degrees of freedom whose equations are replaced by a prescribed value: those of the
  /// nodal boundary conditions, the inactive ones and the dependent ones.
  std::vector<char> constrainedDofs() const;
  /// The blocks of the preconditioner "physics_blocks": every group of displacements
  /// (SolidSystem::displacementGroups), and every other variable on its own, in the order of
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
  /// (TransientOptions::output_times) when the solution is at that time, to write the files of
  /// that time.  An empty function removes it.
  void setOutputCallback(std::function<void(int)> cb) { _output_callback = std::move(cb); }
  /// Called right after a transient run writes its checkpoint (TransientOptions::checkpoint_file),
  /// so that the caller writes beside it what the checkpoint does not hold.  An empty function
  /// removes it.
  void setCheckpointCallback(std::function<void()> cb) { _checkpoint_callback = std::move(cb); }

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
  /// A property evaluated at element centroids (rows: elements).  This and the other evaluations
  /// of the properties and the fluxes below set the problem up first (initialize) if that has
  /// not happened yet.
  std::vector<std::vector<double>> propertyAtCentroids(const std::string & property);
  /// A property evaluated at arbitrary points, in the element that holds each (rows: points, NaN
  /// outside the mesh).  A property with a history takes the average history of that element.
  std::vector<std::vector<double>> propertyAtPoints(const std::string & property,
                                                    const std::vector<Point> & points);
  /// Flux of a kernel evaluated at element centroids.
  std::vector<Point> kernelFluxAtCentroids(const std::string & kernel);
  /// Call @p visit with the context of every element centroid (the solution and the properties
  /// evaluated there) and the block of the element.
  void centroidContexts(const std::function<void(QpContext &, int)> & visit);
  /// The values of the solution at the local degrees of freedom of element @p el, in the layout
  /// of localDofIndex.
  std::vector<double> localValues(const Element & el) const;
  /// Fill @p ctx, for post-processing, at the point @p mp of element @p e that mapPoint mapped:
  /// the solution there without derivatives, from the local values @p local of the element
  /// (localValues), and the properties, with the history that ctx.state_owner and ctx.state_key
  /// name (none for a negative state_owner).
  void
  evaluateAt(QpContext & ctx, Index e, MappedPoint & mp, const std::vector<double> & local) const;
  /// The weight of each node in the nodal (lumped) rule over the blocks of variable @p v, every
  /// block when @p v is negative: under DMCDM the measure of its control domain, the pieces in a
  /// cell with hanging nodes included, and under FEM the integral of its shape function, each
  /// with the coordinate factor of the problem inside the integral (2 pi r in axisymmetric
  /// coordinates).  Partial at the nodes that processes share.  For a first-order variable on
  /// quadratic elements the weights of the nodes that are not corners hold its interpolant, so
  /// the weighted sum of its values is still its integral.
  Vector nodeWeights(int v) const;
  /// The node nearest @p x among those that @p accept, over every process, with a tie to the
  /// smaller global number of the node so that every process agrees: its local number, or -1
  /// where this process does not hold it.  @p distance receives its distance (infinite when no
  /// node is accepted).  Collective in a distributed problem.
  Index nearestNode(const Point & x,
                    const std::function<bool(Index)> & accept,
                    double * distance = nullptr) const;
  /// The measure of each node, nodeWeights over every block, with the measure of the whole mesh
  /// at a node that processes share.  The problem must be set up (initialize).
  std::vector<double> nodeMeasures() const;
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
  /// The integral of u . n over a side set, for the vector u of the variables @p components.
  double normalIntegral(const std::vector<std::string> & components,
                        const std::string & boundary) const;
  /// Integral of the normal flux of a kernel over a side set.
  double boundaryFluxIntegral(const std::string & kernel, const std::string & boundary);
  /// Write a VTK unstructured-grid file with nodal fields and cell data.
  /// @p fields selects the variables (empty: all the variables).  The cell data are properties
  /// at the centroids (propertyAtCentroids).
  void writeVTU(const std::string & filename,
                const std::vector<std::string> & cell_properties = {},
                const std::vector<std::string> & fields = {});
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
  /// parameter.  It sets the problem up first (initialize) if that has not happened yet.
  Vector objectiveGradient(const Objective & objective);
  /// The value of an objective at the current solution.
  double objectiveValue(const Objective & objective);
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
  /// Register a new named settable constant: a parameter that
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
  /// Add the nodal condition @p type, which holds the component normal to its boundary
  /// (InputParameters::normalComponent, a plane of symmetry), for the component of 'variable' along
  /// the axis normal to the boundary.
  std::shared_ptr<Object> addNormalComponentCondition(const std::string & type,
                                                      const std::string & name,
                                                      InputParameters params);
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
    /// Each kernel of `kernels` as an advective kernel, or null (Group::advective).
    std::vector<const AdvectiveKernel *> advective;
    MappedPoint geo, fld;
    QpContext ctx;
    std::vector<Eigen::Triplet<double>> triplets;
  };
  struct Group
  {
    QuadratureSpec spec;
    std::vector<const Kernel *> kernels;
    /// Each kernel of `kernels` as an advective kernel, or null, cast once when the group is
    /// built so that the points of an element do not cast.
    std::vector<const AdvectiveKernel *> advective;
    void add(const Kernel * kernel)
    {
      kernels.push_back(kernel);
      advective.push_back(dynamic_cast<const AdvectiveKernel *>(kernel));
    }
  };
  /// What the parts of one assembly share: the state @p U, the options, the residual they add to,
  /// whether they assemble the Jacobian, and whether the elements add to the residual atomically
  /// (several threads assemble them).
  struct AssemblyPass
  {
    const Vector & U;
    const AssemblyOptions & opts;
    Vector & R;
    bool want_jac;
    bool atomic;
  };
  // ---- the stages of initialize (ProblemSetup.cpp), in the order it calls them ----
  /// Refuse a mixed-order layout that the method or the element types cannot take.
  void checkMixedOrder() const;
  /// Mark the variables of second-order equations, those of equations without a time
  /// derivative, and whether a kernel reads the Laplacian.
  void classifyEquations();
  /// Build the gradients at the nodes when an advective kernel of DMCDM needs them.
  void setupNodeGradients();
  /// Refuse an interface condition that couples a first-order variable.
  void checkInterfaceOrders() const;
  /// Size the history records of the stateful property objects.
  void allocateState();
  /// Build the caches that the mesh and the reference elements fill on first use, so that no two
  /// threads of an assembly fill them at the same time.
  void warmCaches() const;
  /// Whether every object, function and initial condition can be called from several threads at
  /// once.  Anything defined in Python cannot, because the interpreter is protected by the
  /// global interpreter lock.
  bool everythingThreadSafe() const;
  void buildGroups();
  void fillContext(QpContext & ctx,
                   const Element & el,
                   const MappedPoint & field_point,
                   const std::vector<double> & cur_local,
                   const std::vector<double> * lag_local,
                   const std::vector<double> * old_local,
                   int num_derivatives) const;
  /// The Laplacian of the variables at a point of a quadratic element, and of their old values
  /// (QpContext::laplacian), for fillContext.
  void fillLaplacian(QpContext & ctx,
                     const Element & el,
                     const MappedPoint & field_point,
                     const std::vector<double> & cur_local,
                     const std::vector<double> * old_local,
                     int num_derivatives) const;
  /// The shape functions of a variable at a point, and their gradients, at @p count nodes.
  struct Shape
  {
    const double * N;
    const Point * dN;
    int count;
  };
  /// The shape functions that variable @p v of element @p el is interpolated with at the point
  /// @p fp: the element's own, or those of its corners for a first-order variable on a quadratic
  /// element.
  Shape variableShape(const Element & el, const MappedPoint & fp, int v) const;
  /// The index of variable @p v at local node @p k of @p el in the local vectors: localDofIndex,
  /// computed inline without mixed orders.
  int localIndex(const Element & el, int k, int v) const
  {
    return _mixed_order ? localDofIndex(el, k, v) : k * numVariables() + v;
  }
  /// The value at a point of the field whose value at local node k is @p nodal(k), with the shape
  /// functions @p N of @p count nodes there.
  template <class Nodal> static double interpolate(const double * N, int count, const Nodal & nodal)
  {
    double value = 0;
    for (int k = 0; k < count; ++k)
      value += N[k] * nodal(k);
    return value;
  }
  /// The same value, and the gradient @p g of the field in @p dim dimensions, with the gradients
  /// @p dN of the shape functions.
  template <class Nodal>
  static double interpolate(
      const double * N, const Point * dN, int count, int dim, const Nodal & nodal, Point & g)
  {
    double value = 0;
    g = {0, 0, 0};
    for (int k = 0; k < count; ++k)
    {
      const double u = nodal(k);
      value += N[k] * u;
      for (int d = 0; d < dim; ++d)
        g[d] += dN[k][d] * u;
    }
    return value;
  }
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
  /// The paired points of an interface condition: its index among the integrated conditions of
  /// the problem, which keys the history of its points, and its points, indexed [primary side or
  /// face][point].
  struct InterfacePairs
  {
    unsigned index = 0;
    std::vector<std::vector<InterfacePoint>> points;
  };
  /// Pair every integration point of every interface condition (called by
  /// initialize()).
  void buildInterfacePairs();
  /// The local fields of element @p el from @p U into @p s, in the layout of localDofIndex and
  /// then at the hanging nodes of @p polygon: the current values, and the lagged and old values
  /// and the shifts that @p opts gives, with zero local residuals.  Returns the number of local
  /// degrees of freedom.
  int gatherLocal(ThreadScratch & s,
                  const Element & el,
                  const Vector & U,
                  const AssemblyOptions & opts,
                  const PolygonCell * polygon = nullptr) const;
  /// The geometry of integration point @p ip of element @p el, @p mapped when the points were
  /// mapped as they were built, and otherwise mapped into s.geo.  It fills s.ctx with the local
  /// fields of @p s there (at the field point of a @p reduced rule), with @p nd derivatives, and
  /// evaluates the properties for the history state @p key of element @p owner (-1 for none).
  const MappedPoint & evaluatePoint(ThreadScratch & s,
                                    const Element & el,
                                    const IntegrationPoint & ip,
                                    MappedPoint * mapped,
                                    bool reduced,
                                    const AssemblyOptions & opts,
                                    int nd,
                                    Index owner,
                                    std::uint64_t key) const;
  /// The residual of integrated condition @p bc, the condition @p index of the problem, on
  /// @p side, times @p wfac, into the local residual of @p s, whose fields gatherLocal set.
  void conditionResidual(const IntegratedBC & bc,
                         unsigned index,
                         const Side & side,
                         double wfac,
                         const AssemblyOptions & opts,
                         int nd,
                         ThreadScratch & s) const;
  /// The values at a point of an interface condition (assembleInterface): of the variables and
  /// their gradients at the point @p geo of the primary element @p el, of the old state @p old
  /// there (the current state without it), and of the variables at the paired point @p sgeo of
  /// the secondary element @p sel, from the state @p U with @p nd zero derivatives.
  void setInterfaceValues(QpContext & ctx,
                          const Element & el,
                          const MappedPoint & geo,
                          const Element & sel,
                          const MappedPoint & sgeo,
                          const Vector & U,
                          const Vector * old,
                          int nd) const;
  void assembleInterface(const InterfaceBC & bc,
                         const Vector & U,
                         const AssemblyOptions & opts,
                         double wfac,
                         Vector & R,
                         std::vector<Eigen::Triplet<double>> * triplets,
                         ThreadScratch & scratch);
  // ---- the parts of assemble (ProblemAssembly.cpp), in the order it calls them ----
  /// Throw a std::logic_error when @p opts breaks a rule of AssemblyOptions for the state @p U
  /// and the residual @p R.
  void checkAssemblyOptions(const Vector & U, const AssemblyOptions & opts, const Vector & R) const;
  /// The scratch space of @p nthreads threads, with the context of the options @p opts.
  std::vector<ThreadScratch>
  assemblyScratch(const AssemblyOptions & opts, int nthreads, bool want_jac) const;
  /// The kernels @p kernels in groups of one quadrature rule, in the order of their first kernel.
  std::vector<Group> kernelGroups(const std::vector<const Kernel *> & kernels) const;
  /// The quadrature rule of @p kernel in this study: its rule of an eigenvalue study
  /// (StorageKernel::eigenvalueStudyQuadrature) when the problem is in one.
  const QuadratureSpec & studyQuadrature(const Kernel & kernel) const;
  /// The element loop: every integrated element in the threads of @p scratch, one scratch space
  /// each.  The first error of a thread is rethrown after the loop.
  void assembleElements(const AssemblyPass & pass,
                        const std::vector<Group> & groups,
                        std::vector<ThreadScratch> & scratch) const;
  /// The kernels of @p groups on element @p e into the residual and the matrix entries of @p s.
  void assembleElement(ThreadScratch & s,
                       Index e,
                       const AssemblyPass & pass,
                       const std::vector<Group> & groups) const;
  /// The volume integrals of the kernels s.kernels of @p group, the group @p group_index, on
  /// element @p e (or its cell @p polygon) into the local residual of @p s: the control volumes of
  /// DMCDM, or the interior of a finite element.  @p nd is the number of derivatives.
  void elementVolume(ThreadScratch & s,
                     Index e,
                     const PolygonCell * polygon,
                     const Group & group,
                     unsigned group_index,
                     int nd,
                     const AssemblyPass & pass) const;
  /// The fluxes of the kernels s.kernels of @p group through the interfaces between the control
  /// domains of element @p e (DMCDM), as elementVolume takes its arguments.
  void controlDomainFaces(ThreadScratch & s,
                          Index e,
                          const PolygonCell * polygon,
                          const Group & group,
                          unsigned group_index,
                          int nd,
                          const AssemblyPass & pass) const;
  /// The face flux of @p kernel through interface @p ip, times @p scale, whose scheme takes the
  /// gradients at the nodes @p owner_node and @p neighbor_node of the face from the nodes around
  /// them (AdvectiveKernel::needsSideGradients).  Its two rows go straight to the residual and
  /// the matrix entries of @p s.
  void sideGradientFaceFlux(ThreadScratch & s,
                            const AdvectiveKernel & kernel,
                            const IntegrationPoint & ip,
                            double scale,
                            Index owner_node,
                            Index neighbor_node,
                            const AssemblyPass & pass) const;
  /// Set the two sides of a face of the control domains in @p ctx for an advective flux
  /// (QpContext::on_face): the values of every variable at the local nodes @p owner and
  /// @p neighbor in @p values, with a derivative in their slots when @p nd is positive, and the
  /// positions @p x_owner and @p x_neighbor.  A node has no gradient of its own, so the sides
  /// carry none.
  void setFaceSides(QpContext & ctx,
                    const std::vector<double> & values,
                    int nd,
                    int owner,
                    int neighbor,
                    const Point & x_owner,
                    const Point & x_neighbor) const;
  /// The integrated boundary conditions of the options, or of the problem, in the first
  /// thread's scratch space @p s.
  void assembleConditions(const AssemblyPass & pass, ThreadScratch & s);
  /// The advective flux of DMCDM through the boundary faces of the control volumes, in the first
  /// thread's scratch space @p s.
  void assembleAdvectiveBoundary(const AssemblyPass & pass, ThreadScratch & s) const;
  /// The advective flux of @p kernel through the boundary side @p side (assembleAdvectiveBoundary).
  void advectiveBoundaryFlux(const AdvectiveKernel & kernel,
                             const Side & side,
                             const AssemblyPass & pass,
                             ThreadScratch & s) const;
  /// The concentrated nodal loads.
  void assembleNodalLoads(const AssemblyPass & pass) const;
  /// The matrix entries of every thread's scratch space, in the order of the threads, into @p J.
  void mergeJacobian(const std::vector<ThreadScratch> & scratch, SparseMatrix & J) const;
  /// The weight of kernel @p kernel in the assembly @p options: its steady weight, or for a time
  /// kernel the data fraction if it is data and 1 otherwise.
  double kernelWeight(const Kernel & kernel, const AssemblyOptions & options) const;
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
  /// Throw an InputError when the Jacobian is singular because a part of the
  /// mesh is not held: a group of coupled unknowns, cut off from every
  /// prescribed value, on which a vector of the near null space (a constant,
  /// a rigid body motion) is a solution of the homogeneous problem, e.g., a
  /// body without supports or a region without any temperature condition.
  void checkSingularity(const SparseMatrix & J) const;
  /// The message of checkSingularity for the near null space vector @p b, a solution of the
  /// homogeneous problem on the group @p c of coupled unknowns, with @p component the group of
  /// every unknown: the variables and the part of the mesh involved.
  std::string
  singularPartMessage(const std::vector<Index> & component, Index c, const Vector & b) const;
  /// The steady terms of a theta step at its old time, which every residual of the step adds:
  /// their sum, and the sum of their sizes for the reference residual.
  struct OldResidual
  {
    Vector residual, reference, term_size;
  };
  /// Write the owned rows @p owned, in the order of the global numbering @p rows, into @p v, a
  /// vector of the local unknowns, and give the other unknowns the values of their owners.
  void setOwnedPart(Vector & v, const Vector & owned, const GlobalRows & rows) const;
  /// With @p ramp_from, a solve that fails is taken again by continuation: the data are ramped
  /// up from zero, and the prescribed values from those of @p ramp_from.
  SolveResult nonlinearSolve(const SolverOptions & options,
                             AssemblyOptions base,
                             const OldResidual * steady_old,
                             const Vector * ramp_from = nullptr);
  /// The state that the steps of nonlinearSolve share (ProblemSolve.cpp).
  class NonlinearSolve;
  /// Whether a continuation from @p ramp_from has data to ramp: an object that is data, or a
  /// prescribed value that differs from @p ramp_from, on any process.  False without
  /// @p ramp_from.
  bool continuationHasData(const Vector * ramp_from) const;
  /// The theta, generalized-alpha and Newmark methods (ProblemTransient.cpp), kept for the next
  /// run when it continues the last one.
  class ThetaIntegrator;
  std::shared_ptr<ThetaIntegrator> _theta;
  Vector _initial_velocity;
  /// The steps of runTransient and the state they share (ProblemTransient.cpp).
  class TransientRun;
  /// The time-stepping loop of solveTransient: the steps of @p integrator, chosen by the
  /// stepper of @p transient, the outputs at the output times, and the checkpoints of
  /// @p layout numbered from @p first_step.
  SolveResult runTransient(TimeIntegrator & integrator,
                           const TransientOptions & transient,
                           const SolverOptions & options,
                           const CheckpointLayout & layout,
                           int first_step);
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
  /// The map of the dependent degrees of freedom: for each, the independent degrees of freedom
  /// and the weights of u_d = sum_j w_j u_j, with chains resolved.  A periodic copy has weight 1
  /// on its group's representative (on one process; a distributed run joins the copies in its
  /// gather-scatter), a hanging node under FEM the weights of the side it lies on (DMCDM keeps it
  /// as an unknown), and the slot of a first-order variable at a non-corner node of a quadratic
  /// element the linear interpolant of the corners.  An empty list for every independent degree
  /// of freedom, and empty when none depends.
  std::vector<std::vector<std::pair<Index, double>>> _dependent_of;
  /// Whether a dependent degree of freedom carries an equation to fold (a periodic copy or a
  /// hanging node; the slots of first-order variables carry none).
  bool _fold_dependents = false;
  Index _num_integrated_elements = -1;
  std::map<std::string, std::vector<Side>> _interface_candidates;
  std::vector<char> _extra_constrained;
  /// The dual mesh method keeps a hanging node as an unknown of its own and splits a cell with
  /// hanging nodes on its sides as a polygon (PolygonCell): the polygon of each such element, by
  /// element (-1 for the others).
  void buildPolygonCells();
  std::vector<int> _polygon_of;
  std::vector<PolygonCell> _polygon_cells;
  /// Build the map _dependent_of, and the polygon cells of DMCDM.
  void buildDependentDofs();
  std::vector<std::shared_ptr<Property>> _property_objects;
  std::map<std::string, std::shared_ptr<Object>> _by_name;
  PropertyRegistry _props;
  std::vector<Group> _groups;
  std::map<const IntegratedBC *, InterfacePairs> _interface_pairs;
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
  /// The flow system and the solid system, which the modules registered (Factory).
  std::unique_ptr<FlowSystem> _flow;
  std::shared_ptr<const SolidSystem> _solid;
  /// Whether a kernel reads the Laplacian of the variables (Kernel::needsLaplacian).
  bool _needs_laplacian = false;
  Vector _U;
  Vector _last_residual;
  /// The last residual with the equation of each dependent unknown folded into its primaries on each process before the processes added their parts: the reactions read it, since the owner of a primary need not hold the constraint that folds into it.
  Vector _last_folded_residual;
  std::vector<char> _second_order_variables;
  /// The state a continuation of a steady solve started from, from which the prescribed values
  /// are ramped (applyDirichlet); empty outside a continuation.
  Vector _ramp_from;
  double _time = 0.0;
  double _eigenvalue = 0.0;
  bool _eigenvalue_study = false;
  std::function<void(double, Problem &)> _step_callback;
  std::shared_ptr<TimeIntegrator> _time_integrator;
  /// The solution that the last transient run reached, from which a run that continues it starts.
  Vector _transient_end;
  std::function<void(int)> _output_callback;
  std::function<void()> _checkpoint_callback;
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
