// SPDX-License-Identifier: LGPL-2.1-or-later
//
// Problem: owns the mesh, the variables (fields), and all objects, assembles
// residuals and Jacobians with either the dual mesh control domain method or
// the finite element method, and runs steady or transient executioners.
#pragma once

#include "dualmesh/base/Factory.h"
#include "dualmesh/base/Kernel.h"
#include "dualmesh/base/Material.h"
#include "dualmesh/fe/Assembly.h"
#include "dualmesh/fv/CellMesh.h"
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

struct Variable
{
  std::string name;
  int index;
  std::set<int> blocks; ///< empty = everywhere
  FunctionPtr initial_condition;
  double scaling = 1.0;
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
  /// Load stepping: the load factors applied in sequence (empty = {1}).
  std::vector<double> load_factors;
  /// "automatic" (the default: sparse LU where a direct factorisation is
  /// cheap, otherwise BiCGSTAB with the preconditioner below, falling back to
  /// LU if the iteration fails), "lu" (sparse LU), "bicgstab", "gmres"
  /// (restarted), or "cg" (symmetric positive definite systems only).
  std::string linear_solver = "automatic";
  /// For bicgstab and gmres: "ilu" (incomplete LU with no fill, ILU(0)),
  /// "ilut" (incomplete LU with a threshold), "jacobi", or "none".
  std::string preconditioner = "ilu";
  /// Relative tolerance of the Krylov solvers, ||b - A x|| <= tol ||b||.
  double linear_tolerance = 1e-12;
  int linear_max_iterations = 5000;
  /// Krylov vectors kept by gmres before it restarts.
  int gmres_restart = 60;
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
  /// Write the solution every `output_interval` steps (0: never).
  int output_interval = 0;
  std::string output_file_base;

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
///        and the step size, for output and for user callbacks.
/// @param verbose_root suppresses the progress messages on every process but
///        one in a distributed run.
SolveResult runTransient(const TransientOptions & transient,
                         const SolverOptions & options,
                         Vector & solution,
                         double & current_time,
                         const std::function<SolveResult(const Vector &, double)> & take_step,
                         const std::function<void(int, double)> & on_accept,
                         bool verbose_root = true);

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
                  double scaling = 1.0);
  void addFunction(const std::string & name, FunctionPtr f);
  std::shared_ptr<Object>
  addObject(const std::string & type, const std::string & name, InputParameters params);
  /// Add an already constructed object (used for Python-defined objects).
  void addKernel(std::shared_ptr<Kernel> k);
  void addIntegratedBC(std::shared_ptr<IntegratedBC> bc);
  void addNodalBC(std::shared_ptr<NodalBC> bc);
  void addNodalLoad(std::shared_ptr<NodalLoad> load);
  void addMaterial(std::shared_ptr<Material> m);

  /// Resolve all objects (idempotent; called automatically by solve()).
  void initialize();

  // ---- queries --------------------------------------------------------------------
  const Mesh & mesh() const { return *_mesh; }
  std::shared_ptr<Mesh> meshPointer() const { return _mesh; }
  Method method() const { return _method; }
  CoordinateSystem coordinateSystem() const { return _coord; }
  int numVariables() const { return static_cast<int>(_vars.size()); }
  const Variable & variable(int i) const { return _vars[i]; }
  int variableIndex(const std::string & name) const;
  bool hasVariable(const std::string & name) const;
  FunctionPtr function(const std::string & name) const;
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
  MaterialPropertyRegistry & propertyRegistry() { return _props; }
  const MaterialPropertyRegistry & propertyRegistry() const { return _props; }
  std::shared_ptr<Object> object(const std::string & name) const;
  /// The concentrated loads of the problem, needed by the distributed solver
  /// to decide which process applies a load given by coordinates.
  const std::vector<std::shared_ptr<NodalLoad>> & nodalLoads() const { return _loads; }
  std::vector<std::string> objectNames() const;
  double time() const { return _time; }
  void setTime(double t) { _time = t; }

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
  SolveResult solveTransient(const TransientOptions & transient,
                             const SolverOptions & options = {});
  /// Called after every converged time step (time, problem).
  void setTimeStepCallback(std::function<void(double, Problem &)> cb)
  {
    _step_callback = std::move(cb);
  }

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
  /// A material property evaluated at element centroids (rows: elements).
  std::vector<std::vector<double>> propertyAtCentroids(const std::string & property) const;
  /// Flux of a kernel evaluated at element centroids.
  std::vector<Point> kernelFluxAtCentroids(const std::string & kernel) const;
  /// Integral of a variable over the domain (with the coordinate factor).
  double integrate(const std::string & var) const;
  /// Integral of the normal flux of a kernel over a side set.
  double boundaryFluxIntegral(const std::string & kernel, const std::string & boundary) const;
  /// Write a VTK unstructured-grid file with nodal fields and cell data.
  void writeVTU(const std::string & filename,
                const std::vector<std::string> & cell_properties = {}) const;

  /// Nodal residual of the last converged solution (before Dirichlet rows).
  const Vector & lastResidual() const { return _last_residual; }

  std::string summary() const;

private:
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
  void computeMaterials(QpContext & ctx) const;
  double coordFactor(const Point & x) const;
  void markActiveDofs();
  /// Whether "automatic" should factorise a system of @p n unknowns directly.
  bool preferDirectSolver(Index n) const;
  Vector linearSolve(const SparseMatrix & A,
                     const Vector & b,
                     const SolverOptions & o,
                     int * iterations = nullptr) const;
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
  std::map<std::string, FunctionPtr> _functions;
  std::vector<std::shared_ptr<Kernel>> _kernels;
  std::vector<std::shared_ptr<IntegratedBC>> _ibcs;
  std::vector<std::shared_ptr<NodalBC>> _nbcs;
  std::vector<std::shared_ptr<NodalLoad>> _loads;
  std::vector<std::shared_ptr<Material>> _materials;
  std::map<std::string, std::shared_ptr<Object>> _by_name;
  MaterialPropertyRegistry _props;
  std::vector<Group> _groups;
  std::vector<char> _active_dof;
  bool _initialized = false;
  Vector _U;
  Vector _last_residual;
  double _time = 0.0;
  std::function<void(double, Problem &)> _step_callback;
};

} // namespace dualmesh
