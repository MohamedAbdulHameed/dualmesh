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
#include "dualmesh/fv/CellMesh.h"
#include "dualmesh/linalg/SaddlePointSolver.h"
#include "dualmesh/mesh/Mesh.h"

#include <Eigen/Sparse>

#include <array>
#include <functional>
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
  /// Converged when ||R|| <= relative_tolerance * ||R_0|| ...
  double relative_tolerance = 1e-10;
  /// ... or when ||R|| <= absolute_tolerance ...
  double absolute_tolerance = 1e-12;
  /// ... or when ||U^{r+1} - U^r|| / ||U^{r+1}|| <= step_tolerance (book criterion).
  double step_tolerance = 1e-12;
  /// Book relaxation (Eq. 6.2.15): U = (1 - gamma) U^{r+1} + gamma U^r.
  double relaxation = 0.0;
  /// Newton only: "none" takes the full Newton step; "backtracking" halves
  /// the step until the residual norm falls, ||R(U + a dU)|| <= (1 - 1e-4 a)
  /// ||R(U)||, at most max_line_search_steps times (the Armijo condition on
  /// the residual norm).  It costs one residual assembly per trial.
  std::string line_search = "none";
  int max_line_search_steps = 10;
  /// Load stepping: the load factors applied in sequence (empty = {1}).
  std::vector<double> load_factors;
  /// "automatic" (the default: sparse LU where a direct factorisation is
  /// cheap, otherwise BiCGSTAB with the preconditioner below, falling back to
  /// LU if the iteration fails; for a large Taylor-Hood system, flexible
  /// GMRES with the "pressure_mass_schur" preconditioner), "lu" (sparse LU), "bicgstab", "gmres"
  /// (restarted), "cg" (symmetric positive definite systems only), or
  /// "petsc" (PETSc's solvers, configured by petsc_options; needs a build
  /// with PETSc).
  std::string linear_solver = "automatic";
  /// For bicgstab and gmres: "ilu" (incomplete LU with no fill, ILU(0)),
  /// "ilut" (incomplete LU with a threshold), "jacobi", or "none".  For cg
  /// and bicgstab: "amg", algebraic multigrid by smoothed aggregation
  /// (Amg.h), whose near null space is built from the displacement variables
  /// of the elastic property objects (the rigid body modes) and the constant of
  /// every other variable.
  ///
  /// "pressure_mass_schur" is for pressure-velocity flow (the saddle point
  /// systems of the Taylor-Hood and the stabilised elements): the block
  /// upper-triangular preconditioner whose Schur complement approximation is
  /// the pressure mass matrix scaled by 1/viscosity (SaddlePointSolver.h).
  /// With linear_solver "gmres" or "automatic" it runs in the built-in
  /// flexible GMRES with exact factorisations of both blocks; with "petsc" it
  /// sets up PETSc's Schur complement field split with that matrix, and
  /// petsc_options can change the block solvers (for instance
  /// "-fieldsplit_0_pc_type hypre").
  std::string preconditioner = "ilu";
  /// Relative tolerance of the Krylov solvers, ||b - A x|| <= tol ||b||.
  double linear_tolerance = 1e-12;
  /// Strength threshold of the aggregation of the "amg" preconditioner.
  double amg_strength_threshold = 0.03;
  int linear_max_iterations = 5000;
  /// Krylov vectors kept by gmres before it restarts.
  int gmres_restart = 60;
  /// With linear_solver = "petsc": PETSc's option string, for example
  /// "-ksp_type gmres -pc_type hypre".  Empty selects GMRES with a direct
  /// factorisation, by MUMPS when PETSc has it (MUMPS pivots, so it also
  /// factorises the saddle point systems of the Taylor-Hood element, which
  /// have zeros on the diagonal).  The tolerance and iteration limit above are passed on and
  /// may be overridden here (-ksp_rtol, -ksp_max_it).
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
  /// "error" controls the local truncation error by step doubling: the same
  /// interval is advanced once with one step and once with two half steps, and
  /// the difference between the two answers estimates the error of the coarse
  /// one.  A step whose estimated error exceeds `error_tolerance` is thrown
  /// away and retried with a smaller step; an accepted step is followed by the
  /// largest step the estimate says will still meet the tolerance.  The
  /// solution that is kept is the one from the two half steps, which is the
  /// more accurate of the two.  Three nonlinear solves per accepted step is
  /// the price of the estimate.
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
  /// The Courant number the "cfl" stepper aims for: each step is the one that gives this Courant number at the solution of the step before, grown by at most growth_factor.
  /// The stepper needs a time integration that measures the Courant number (the projection integration of a flow).
  double courant_number = 0.5;
};

struct IterationRecord
{
  int load_step;
  double load_factor;
  int iteration;
  double residual_norm;
  double step_norm;
};

struct SolveResult
{
  bool converged = false;
  /// Nonlinear (Newton or Picard) iterations, summed over the load steps and,
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
  std::vector<IterationRecord> history;
};

/// The time-stepping loop, shared by the serial and distributed executioners.
///
/// @param take_step advances @p solution by one step of the given size,
///        starting from the state it is given, and reports whether the
///        nonlinear solve converged and how many iterations it took.  It must
///        not throw on non-convergence.
/// @param on_accept is called after every accepted step with the step number
///        and the step size, for user callbacks.
/// @param write_output is called with the index of an output time (of
///        TransientOptions::output_times) when the solution is at that time:
///        at the start, and after the accepted step that lands on it.  The
///        steps land on the output times.  It can be empty.
/// @param verbose_root suppresses the progress messages on every process but
///        one in a distributed run.
SolveResult runTransient(const TransientOptions & transient,
                         const SolverOptions & options,
                         Vector & solution,
                         double & current_time,
                         const std::function<SolveResult(const Vector &, double)> & take_step,
                         const std::function<void(int, double)> & on_accept,
                         const std::function<void(int)> & write_output,
                         bool verbose_root = true,
                         const std::function<void(int, double)> & after_step = {},
                         const std::function<double(double)> & courant = {});

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

class Problem
{
public:
  Problem(std::shared_ptr<Mesh> mesh,
          Method method = Method::DualMesh,
          CoordinateSystem coord = CoordinateSystem::Cartesian);

  // ---- setup ------------------------------------------------------------------
  int addVariable(const std::string & name,
                  const std::vector<std::string> & blocks = {},
                  FunctionPtr initial_condition = nullptr,
                  double scaling = 1.0,
                  VariableOrder order = VariableOrder::Mesh);
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
  void addProperty(std::shared_ptr<Property> m);

  /// Resolve all objects (idempotent; called automatically by solve()).
  void initialize();

  // ---- queries --------------------------------------------------------------------
  const Mesh & mesh() const { return *_mesh; }
  std::shared_ptr<Mesh> meshPointer() const { return _mesh; }
  Method method() const { return _method; }
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
  /// True when the unknowns sit at cell centroids instead of mesh nodes.
  bool isCellCentered() const { return _method == Method::FiniteVolumeCell; }
  /// The cell-centred finite volume mesh (cell-centred methods only).
  const CellMesh & cellMesh() const;
  /// Number of entities carrying degrees of freedom: mesh nodes for the dual
  /// mesh, finite element and vertex-centred finite volume methods; cells plus
  /// boundary faces for the cell-centred finite volume method.
  Index numEntities() const;
  /// Position of the degree of freedom entity @p i.
  const Point & entityPoint(Index i) const;
  /// Entities of a boundary, used by nodal (essential) boundary conditions.
  std::vector<Index> boundaryEntities(const std::string & name) const;
  Index numDofs() const { return numEntities() * numVariables(); }
  Index dof(Index entity, int var) const { return entity * numVariables() + var; }

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

  /// The blocks of the pressure mass matrix Schur preconditioner for the
  /// linear system with matrix @p jacobian (see SaddlePointSolver.h): the
  /// pressure is the variable of the one kernel that provides a Schur mass
  /// coefficient (MassConservation), the mass matrix is assembled with that
  /// coefficient (1/mu) and the pressure's own test functions, and its sign
  /// is measured from the coupling blocks of @p jacobian.
  SaddlePointBlocks saddlePointBlocks(const SparseMatrix & jacobian) const;
  BoundaryGradient boundaryGradient() const { return _boundary_gradient; }
  void setBoundaryGradient(BoundaryGradient g) { _boundary_gradient = g; }

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
  /// Whether any property object keeps a history at its integration points.
  bool hasState() const { return _state_record > 0; }
  /// Size of one point's history record.
  int stateRecordSize() const { return _state_record; }
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
    std::vector<std::vector<std::uint64_t>> keys[2];
    std::vector<std::vector<double>> values[2];
  };
  StateSnapshot snapshotState() const;
  void restoreState(const StateSnapshot & snapshot);
  /// The committed history averaged over the volume points of each element
  /// (one record per element), for output.
  std::vector<std::vector<double>> stateAtElements() const;
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
  bool hasElementField(const std::string & name) const { return _element_fields.count(name) > 0; }
  std::vector<std::string> elementFieldNames() const;

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
    double load_factor = 1.0;
    double theta = 1.0; ///< weight of steady terms
    bool include_time_kernels = true;
    bool include_steady_terms = true;
  };
  /// Assemble R(U) (and optionally dR/dU) without applying Dirichlet rows.
  void assemble(const Vector & U, const AssemblyOptions & opts, Vector & R, SparseMatrix * J);
  /// Write the prescribed values into @p U at the nodes of every essential
  /// boundary condition.
  void applyDirichlet(Vector & U, double load_factor) const;
  /// Replace the equations of the prescribed degrees of freedom: the residual
  /// entry becomes zero and the row and the column of the Jacobian become the
  /// corresponding row and column of the identity.  Dropping the column as
  /// well as the row keeps a symmetric problem symmetric, which is valid
  /// because the increment at a prescribed degree of freedom is zero.
  void dirichletRows(const Vector & U, double load_factor, Vector & R, SparseMatrix * J) const;
  /// The degrees of freedom whose equations are replaced by a prescribed
  /// value: those of the nodal boundary conditions and the inactive ones.
  std::vector<char> constrainedDofs() const;
  /// The near null space of the system for the algebraic multigrid
  /// preconditioner: for every set of displacement variables named by a
  /// property object, the rigid body translations and (in Cartesian coordinates)
  /// rotations, and the constant vector of every other variable.  One row per
  /// degree of freedom.
  Eigen::MatrixXd nearNullspace() const;
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
  void setOwnedEntities(std::vector<char> owned) { _owned_entity = std::move(owned); }

  // ---- executioners ---------------------------------------------------------------------
  SolveResult solveSteady(const SolverOptions & options = {});
  /// Solve A x = b with the serial linear solvers chosen by @p o (the solve of each Newton iteration).
  Vector linearSolve(const SparseMatrix & A,
                     const Vector & b,
                     const SolverOptions & o,
                     int * iterations = nullptr) const;
  SolveResult solveTransient(const TransientOptions & transient,
                             const SolverOptions & options = {});
  /// Called after every converged time step (time, problem).
  /// The elements that this problem integrates: all of them, except the ghost elements at the end of the mesh that a distributed run adds to read the other side of an interface or, for the cell-centered method, the cells next to its part.
  Index numIntegratedElements() const
  {
    return _num_integrated_elements >= 0 ? _num_integrated_elements : _mesh->numElements();
  }
  /// Make the elements from @p first on ghosts, which are read but not integrated, and size the solution for the nodes added to the mesh with them.
  void setGhostElements(Index first);
  /// The global number of every local element.
  /// The processes of a distributed problem of the cell-centered method agree by it which of them integrates a face between two of their parts: the one that integrates the cell with the smaller number.
  void setGlobalElementNumbers(std::vector<Index> numbers)
  {
    _global_element_numbers = std::move(numbers);
  }
  /// Whether this problem integrates face @p f of the cell-centered method: a face of two integrated cells, a boundary face of an integrated cell, and a face between an integrated cell and a ghost cell when the integrated cell has the smaller global number.
  /// Every face of a distributed mesh is thus integrated by exactly one process.
  bool integratesFace(const CellFace & f) const;
  /// Whether entity @p i belongs to the part that this problem integrates: for the cell-centered method, an integrated cell or a boundary face of one; every node otherwise.
  /// A value or a load given at a point goes to the nearest of these entities, so that a ghost cell's face, which is not a face of the mesh, is never chosen.
  bool integratedEntity(Index i) const;
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
  const std::shared_ptr<TimeIntegrator> & timeIntegrator() const { return _time_integrator; }
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
  /// Secondary variables (reactions) at the nodes of a boundary: for every
  /// node, the integral of the normal flux over the boundary part of its
  /// control domain.  Available after solveSteady().
  std::vector<std::pair<Index, double>> reactions(const std::string & var,
                                                  const std::string & boundary) const;
  double totalReaction(const std::string & var, const std::string & boundary) const;
  /// Interpolate a variable at arbitrary points (NaN outside the mesh).
  std::vector<double> sample(const std::string & var, const std::vector<Point> & points) const;
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
  /// element interpolation of the nodal values for the three node-based
  /// methods, and for the cell-centred finite volume method the linear
  /// reconstruction U_c + G_c . (x - x_c) in every cell, whose gradient is the
  /// reconstructed cell gradient G_c.  The integrals are evaluated with a
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
  /// Integral of a variable over the domain (with the coordinate factor).
  double integrate(const std::string & var) const;
  /// The integral of a variable over every element (over every cell of the
  /// cell-centred method), with the coordinate factor.  An empty name gives
  /// the volume of every element.
  std::vector<double> elementIntegrals(const std::string & var) const;
  /// Measure of the named side sets: their total length in two dimensions and
  /// area in three, with the coordinate factor of axisymmetric and spherical
  /// problems.  In a distributed run it is the measure on the whole mesh.
  double boundaryMeasure(const std::vector<std::string> & boundaries) const;
  /// Measures of side sets of the whole mesh, given by a distributed problem
  /// to its rank-local problem, whose mesh holds only part of each side set.
  void setGlobalBoundaryMeasures(std::map<std::string, double> measures)
  {
    _global_boundary_measures = std::move(measures);
  }
  /// Integral of the normal flux of a kernel over a side set.
  double boundaryFluxIntegral(const std::string & kernel, const std::string & boundary) const;
  /// Write a VTK unstructured-grid file with nodal fields and cell data.
  /// @p fields selects the variables (empty: all the variables).
  void writeVTU(const std::string & filename,
                const std::vector<std::string> & cell_properties = {},
                const std::vector<std::string> & fields = {}) const;
  /// Set the SI unit of a variable.
  void setVariableUnit(const std::string & name, const std::string & unit);

  /// Nodal residual of the last converged solution (before Dirichlet rows).
  const Vector & lastResidual() const { return _last_residual; }

  std::string summary() const;

private:
  /// Create a symmetry_boundary_condition on the displacement normal to its plane.
  std::shared_ptr<Object> addSymmetryCondition(const std::string & name, InputParameters params);
  std::map<std::string, double> _global_boundary_measures;
  /// Everything one thread needs to assemble one element or one face, so that
  /// nothing is shared between threads except the read-only problem
  /// definition, the solution vectors, and the global residual, which is
  /// updated atomically.
  struct ThreadScratch
  {
    std::vector<double> cur, lag, old;
    std::vector<Index> dofs;
    /// Cell-centred finite volume only: the entities whose values a face flux
    /// depends on, in the order of the local degrees of freedom.
    std::vector<Index> entities;
    std::vector<ADReal> Rloc;
    std::vector<char> touched;
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
  /// Assembly for the cell-centred finite volume method (ProblemFiniteVolume.cpp).
  void assembleCellFiniteVolume(const Vector & U,
                                const AssemblyOptions & opts,
                                Vector & R,
                                SparseMatrix * J);
  /// Green-Gauss gradient of every variable at every cell centroid.
  std::vector<std::vector<Point>> cellGradients(const Vector & U) const;
  /// Fill a context with the cell value and the Green-Gauss cell gradient.
  void
  fillContextCell(QpContext & ctx, Index cell, const std::vector<std::vector<Point>> & grads) const;
  void fillContext(QpContext & ctx,
                   const Element & el,
                   const MappedPoint & field_point,
                   const std::vector<double> & cur_local,
                   const std::vector<double> * lag_local,
                   const std::vector<double> * old_local,
                   int num_derivatives) const;
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
    Index face = -1;    ///< cell-centred method: secondary boundary face
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
  /// The weight of the equation of a cell-centred boundary entity: the
  /// coordinate factor, or 1 where that vanishes (on an axis of symmetry).
  double boundaryEntityFactor(const Point & x) const;
  void markActiveDofs();
  /// Whether "automatic" should factorise a system of @p n unknowns directly.
  bool preferDirectSolver(Index n) const;
  /// The direct solve, which reuses the factorisation of the previous call
  /// when the matrix has not changed (a linear problem in its second Newton
  /// iteration, or a transient with a constant matrix).
  Vector cachedDirectSolve(const SparseMatrix & A, const Vector & b) const;
  /// Conjugate gradients (symmetric matrix) or BiCGSTAB preconditioned by
  /// algebraic multigrid, whose hierarchy is reused while the matrix is
  /// unchanged.  False when the iteration did not converge.
  bool amgSolve(const SparseMatrix & A,
                const Vector & b,
                const SolverOptions & o,
                const std::string & method,
                int max_iterations,
                int * iterations,
                Vector & x) const;
  /// Throw an InputError when the Jacobian is singular because a part of the
  /// mesh is not held: a group of coupled unknowns, cut off from every
  /// prescribed value, on which a vector of the near null space (a constant,
  /// a rigid body motion) is a solution of the homogeneous problem, e.g., a
  /// body without supports or a region without any temperature condition.
  void checkSingularity(const SparseMatrix & J) const;
  struct LinearSolverCache;
  mutable std::shared_ptr<LinearSolverCache> _solver_cache;
  SolveResult nonlinearSolve(const SolverOptions & options,
                             AssemblyOptions base,
                             const Vector * steady_old_residual);
  /// Advance the solution from @p old by one step of size @p dt, leaving the
  /// result in solution().  Used by every time stepper.
  SolveResult takeTimeStep(const Vector & old,
                           double dt,
                           const TransientOptions & transient,
                           const SolverOptions & options);

  std::shared_ptr<Mesh> _mesh;
  Method _method;
  CoordinateSystem _coord;
  std::shared_ptr<CellMesh> _cells;
  BoundaryGradient _boundary_gradient = BoundaryGradient::FirstOrder;
  int _num_threads = 0;
  bool _thread_safe = true;
  std::vector<char> _owned_entity;
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
  Index _num_integrated_elements = -1;
  std::vector<Index> _global_element_numbers;
  std::map<std::string, std::vector<Side>> _interface_candidates;
  bool _constraints_outside = false;
  std::vector<char> _extra_constrained;
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
  mutable bool _singularity_checked = false;
  Vector _U;
  Vector _last_residual;
  double _time = 0.0;
  double _eigenvalue = 0.0;
  std::function<void(double, Problem &)> _step_callback;
  std::shared_ptr<TimeIntegrator> _time_integrator;
  std::function<void(int)> _output_callback;
  std::map<std::string, std::vector<double>> _element_fields;

  /// The history of one owner (an element, or a cell-centred face): the keys
  /// of its points and, for each, a record in three buffers.  Only the
  /// thread that assembles an owner touches its entry.
  struct OwnerState
  {
    std::vector<std::uint64_t> keys;
    std::vector<double> start, trial, committed;
  };
  int _state_record = 0;
  mutable std::vector<OwnerState> _state[2];
  Vector _state_committed_U;
  /// Point the context at the history of its point (inserting a zero record
  /// the first time), or clear the pointers if the point has none.
  void bindState(QpContext & ctx) const;
};

} // namespace dualmesh
