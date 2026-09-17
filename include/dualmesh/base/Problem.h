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
#include "dualmesh/mesh/Mesh.h"

#include <Eigen/Sparse>

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
  /// "lu" (sparse LU), "bicgstab" (ILUT preconditioned), or "cg".
  std::string linear_solver = "lu";
  double linear_tolerance = 1e-12;
  int linear_max_iterations = 5000;
  bool verbose = false;
  /// Throw if the nonlinear iteration does not converge.
  bool error_on_divergence = true;
};

struct TransientOptions
{
  double start_time = 0.0;
  double end_time = 1.0;
  double dt = 0.1;
  /// 0 = forward (explicit) Euler, 0.5 = Crank-Nicolson, 1 = backward Euler.
  double theta = 1.0;
  /// Write the solution every `output_interval` steps (0: never).
  int output_interval = 0;
  std::string output_file_base;
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
  int total_iterations = 0;
  std::vector<IterationRecord> history;
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
  Index numDofs() const { return _mesh->numNodes() * numVariables(); }
  Index dof(Index node, int var) const { return node * numVariables() + var; }
  MaterialPropertyRegistry & propertyRegistry() { return _props; }
  const MaterialPropertyRegistry & propertyRegistry() const { return _props; }
  std::shared_ptr<Object> object(const std::string & name) const;
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
  void computeMaterials(QpContext & ctx) const;
  double coordFactor(const Point & x) const;
  void applyDirichlet(Vector & U, double load_factor) const;
  void dirichletRows(const Vector & U, double load_factor, Vector & R, SparseMatrix * J) const;
  void markActiveDofs();
  Vector linearSolve(const SparseMatrix & A, const Vector & b, const SolverOptions & o) const;
  SolveResult nonlinearSolve(const SolverOptions & options,
                             AssemblyOptions base,
                             const Vector * steady_old_residual);

  std::shared_ptr<Mesh> _mesh;
  Method _method;
  CoordinateSystem _coord;
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
