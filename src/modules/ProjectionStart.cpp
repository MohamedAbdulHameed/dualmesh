// SPDX-License-Identifier: LGPL-2.1-or-later
//
// The projection of the initial velocity of a flow onto a field without divergence
// (projectInitialVelocity of Projection.h).
#include "ProjectionShared.h"
#include "dualmesh/linalg/Communicator.h"
#include "dualmesh/modules/FluidSystem.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace dualmesh
{

namespace
{

/// The weight eps of the regularization of the saddle point system (solveCorrection).
constexpr double kRegularization = 1e-4;
/// The relative tolerance of each solve of the saddle point system.
constexpr double kSolveTolerance = 1e-12;
/// The largest number of solves of the iterated regularization, and the change of the velocity,
/// relative to the largest speed, at which it stops.
constexpr int kMaxSolves = 20;
constexpr double kSettledChange = 1e-13;
/// The largest sum of the columns of D at a free velocity, relative to the largest sum of their
/// sizes, of an enclosed flow (isEnclosed).
constexpr double kEnclosedTolerance = 1e-10;

/// The operators of the projection: the divergence D, the Laplacian K_l of the multiplier and the
/// mass matrix M of the velocity over the density.
struct StartOperators
{
  SparseMatrix D, laplacian, M;
};

/// The StartOperators of the flow of the velocity variables @p velocities, marked by
/// @p is_velocity, with the multiplier tested with the functions of the variable @p m and the
/// time derivatives @p storage of the momentum equations.
/// D is the mass balance of the flow without its stabilization: (q, div u) with the coordinate
/// factor under FEM, and the flux of u out of the control domain of each node under DMCDM.
StartOperators
assembleOperators(Problem & problem,
                  const std::vector<int> & velocities,
                  const std::vector<char> & is_velocity,
                  int m,
                  const std::vector<const Kernel *> & storage)
{
  const int nv = problem.numVariables();
  std::vector<std::string> names;
  for (int v : velocities)
    names.push_back(problem.variable(v).name);
  const std::string multiplier = problem.variable(m).name;
  std::vector<std::string> boundaries;
  for (const auto & [name, list] : problem.mesh().sidesets())
    boundaries.push_back(name);
  const auto balance =
      projection::privateObject<MassEquation>(problem,
                                              "mass_conservation",
                                              "start_projection_divergence",
                                              {{"variable", multiplier}, {"velocities", names}});
  std::vector<std::shared_ptr<IntegratedBC>> outflow;
  if (!boundaries.empty())
    outflow.push_back(projection::privateObject<IntegratedBC>(
        problem,
        "mass_flux",
        "start_projection_outflow",
        {{"variable", multiplier}, {"velocities", names}, {"boundary", boundaries}}));
  const auto diffusion = projection::privateObject<Kernel>(
      problem, "diffusion", "start_projection_laplacian", {{"variable", multiplier}});
  std::vector<const IntegratedBC *> conditions;
  for (const auto & bc : outflow)
    conditions.push_back(bc.get());
  std::vector<char> is_multiplier(static_cast<std::size_t>(nv), 0);
  is_multiplier[static_cast<std::size_t>(m)] = 1;
  StartOperators operators;
  Problem::OperatorOptions divergence;
  divergence.kernels = std::vector<const Kernel *>{balance.get()};
  divergence.conditions = conditions;
  divergence.stabilization = false;
  operators.D =
      projection::block(problem, problem.operatorOf(divergence), is_multiplier, is_velocity);
  Problem::OperatorOptions at_rest;
  at_rest.kernels = std::vector<const Kernel *>{diffusion.get()};
  operators.laplacian = problem.operatorOf(at_rest);
  const double density = flowCoefficientsAt(problem, *balance, problem.mesh().node(0)).time_density;
  at_rest.kernels = storage;
  operators.M = (1.0 / (density > 0.0 ? density : 1.0)) *
                projection::block(problem, problem.operatorOf(at_rest), is_velocity, is_velocity);
  return operators;
}

/// The saddle point system of the projection over the velocity components and the multiplier at
/// every node, the multiplier last, and the unknowns that it fixes.
struct StartSystem
{
  StartSystem(const Problem & problem_,
              const std::vector<int> & velocities_,
              int m_,
              const StartOperators & operators_)
      : problem(problem_), velocities(velocities_), m(m_), operators(operators_),
        nv(problem_.numVariables()), dim(static_cast<int>(velocities_.size())),
        numbering(problem_, fieldsOf(velocities_, m_), {})
  {
    field.assign(static_cast<std::size_t>(nv), -1);
    for (int d = 0; d < dim; ++d)
      field[static_cast<std::size_t>(velocities[d])] = d;
    constrained = problem.constrainedDofs();
  }

  /// The global row of the field @p f at node @p i, and its place among the rows of this process.
  Index row(Index i, int f) const { return numbering.width * numbering.node[i] + f; }
  Index at(Index i, int f) const { return row(i, f) - numbering.first; }
  /// Whether the field @p f at node @p i is an unknown of the system.
  bool free(Index i, int f) const
  {
    if (numbering.node[i] < 0)
      return false;
    return f < dim ? !velocity_fixed[static_cast<std::size_t>(f)][i] : !fixed[i];
  }
  /// The field of the unknown @p d of the problem: a velocity component, or -1.
  int fieldOf(Index d) const { return field[static_cast<std::size_t>(d % nv)]; }
  /// D u, summed over the processes.
  Vector divergence(const Vector & u) const
  {
    Vector r = operators.D * u;
    problem.parallel().addAtNodes(r.data(), nv);
    return r;
  }

  const Problem & problem;
  const std::vector<int> & velocities;
  /// The variable whose functions test the multiplier.
  int m;
  const StartOperators & operators;
  int nv, dim;
  const ProjectionSolver::Rows numbering;
  /// The velocity component of every variable of the problem, or -1.
  std::vector<int> field;
  std::vector<char> constrained;
  /// Whether the multiplier at each node is fixed at zero, and whether each velocity component at
  /// each node is prescribed.
  std::vector<char> fixed;
  std::vector<std::vector<char>> velocity_fixed;

private:
  static std::vector<int> fieldsOf(std::vector<int> velocities, int m)
  {
    velocities.push_back(m);
    return velocities;
  }
};

/// Whether the multiplier at each node is fixed at zero: where the pressure @p pressure is
/// prescribed, whose mass balance is not an equation of the flow, at a dependent unknown, and at
/// a node that tests no divergence (a node outside the flow).
std::vector<char>
fixedMultipliers(const StartSystem & system, int pressure)
{
  const Problem & problem = system.problem;
  const SparseMatrix & D = system.operators.D;
  Vector tested = Vector::Zero(problem.numDofs());
  for (Index col = 0; col < D.outerSize(); ++col)
    for (SparseMatrix::InnerIterator it(D, col); it; ++it)
      tested[it.row()] += std::abs(it.value());
  problem.parallel().addAtNodes(tested.data(), system.nv);
  const Index nodes = problem.mesh().numNodes();
  std::vector<char> fixed(static_cast<std::size_t>(nodes), 0);
  for (Index i = 0; i < nodes; ++i)
  {
    const Index d = problem.dof(i, system.m);
    const bool given = pressure >= 0 ? system.constrained[d] != 0 : problem.isDependent(d);
    fixed[i] = given || !(tested[d] > 0.0) ? 1 : 0;
  }
  return fixed;
}

/// Whether each velocity component at each node is prescribed.
std::vector<std::vector<char>>
fixedVelocities(const StartSystem & system)
{
  const Index nodes = system.problem.mesh().numNodes();
  std::vector<std::vector<char>> fixed(static_cast<std::size_t>(system.dim));
  for (int d = 0; d < system.dim; ++d)
  {
    fixed[d].resize(static_cast<std::size_t>(nodes));
    for (Index i = 0; i < nodes; ++i)
      fixed[d][i] = system.constrained[system.problem.dof(i, system.velocities[d])];
  }
  return fixed;
}

/// Whether the flow prescribes its normal velocity on the whole boundary (an enclosed flow, or
/// one with a prescribed inflow and outflow) and prescribes no pressure @p pressure.  Such a flow
/// leaves the multiplier free up to a constant: D^T 1 is zero at every free velocity.
bool
isEnclosed(const StartSystem & system, int pressure)
{
  const Problem & problem = system.problem;
  const Parallel & parallel = problem.parallel();
  const Index nodes = problem.mesh().numNodes();
  if (pressure >= 0)
  {
    bool given = false;
    for (Index i = 0; i < nodes; ++i)
    {
      const Index d = problem.dof(i, pressure);
      given =
          given || (system.constrained[d] && problem.activeDofs()[d] && !problem.isDependent(d));
    }
    if (parallel.any(given))
      return false;
  }
  const SparseMatrix & D = system.operators.D;
  Vector column = Vector::Zero(problem.numDofs()), size = Vector::Zero(problem.numDofs());
  for (Index col = 0; col < D.outerSize(); ++col)
    for (SparseMatrix::InnerIterator it(D, col); it; ++it)
    {
      column[col] += it.value();
      size[col] += std::abs(it.value());
    }
  parallel.addAtNodes(column.data(), system.nv);
  parallel.addAtNodes(size.data(), system.nv);
  double open = 0.0, scale = 0.0;
  for (Index i = 0; i < nodes; ++i)
    for (int d = 0; d < system.dim; ++d)
    {
      const Index dof = problem.dof(i, system.velocities[d]);
      scale = std::max(scale, size[dof]);
      if (!system.velocity_fixed[d][i])
        open = std::max(open, std::abs(column[dof]));
    }
  return parallel.maximum(open) <= kEnclosedTolerance * parallel.maximum(scale);
}

/// For an enclosed flow: spread the divergence that the data give the whole flow, the mean of
/// @p rhs, over its volume, and fix the multiplier at the free node of the smallest global
/// number.
void
spreadDivergence(StartSystem & system, Vector & rhs)
{
  const Problem & problem = system.problem;
  const Parallel & parallel = problem.parallel();
  const Index nodes = problem.mesh().numNodes();
  const std::vector<Index> & global = system.numbering.node;
  const std::vector<char> & owned = system.numbering.owned;
  // The measure of the test function of every node, over which the divergence is spread.
  Vector volume = problem.nodeWeights(system.m);
  parallel.addAtNodes(volume.data(), 1);
  double total = 0.0, measure = 0.0;
  for (Index i = 0; i < nodes; ++i)
    if (owned[i] && !system.fixed[i])
    {
      total += rhs[problem.dof(i, system.m)];
      measure += volume[i];
    }
  const double mean = parallel.sum(total) / parallel.sum(measure);
  double smallest = std::numeric_limits<double>::infinity();
  for (Index i = 0; i < nodes; ++i)
    if (!system.fixed[i])
    {
      rhs[problem.dof(i, system.m)] -= mean * volume[i];
      if (owned[i])
        smallest = std::min(smallest, static_cast<double>(global[i]));
    }
  smallest = parallel.minimum(smallest);
  for (Index i = 0; i < nodes; ++i)
    if (static_cast<double>(global[i]) == smallest)
      system.fixed[i] = 1;
}

/// The entries of the saddle point system
///
///     [ M      D^T    ] [ du     ]   [ 0       ]
///     [ D   -eps K_l  ] [ lambda ] = [ - D u_0 ]
///
/// with M the mass matrix of the velocity and K_l the Laplacian of the multiplier, which has the
/// spectrum of D M^-1 D^T on the smooth fields, and a row of the identity at every fixed unknown.
/// The small regularization eps K_l fixes the multiplier where D^T has a null space (the spurious
/// pressure modes of equal-order elements), whose divergence no velocity changes.
std::vector<petsc::GlobalEntry>
saddleEntries(const StartSystem & system)
{
  const int nv = system.nv, dim = system.dim, m = system.m;
  const StartOperators & operators = system.operators;
  std::vector<petsc::GlobalEntry> entries;
  const auto add = [&](Index i, int fi, Index j, int fj, double value)
  {
    if (system.free(i, fi) && system.free(j, fj))
      entries.push_back({system.row(i, fi), system.row(j, fj), value});
  };
  for (Index col = 0; col < operators.M.outerSize(); ++col)
    for (SparseMatrix::InnerIterator it(operators.M, col); it; ++it)
      add(it.row() / nv, system.fieldOf(it.row()), col / nv, system.fieldOf(col), it.value());
  for (Index col = 0; col < operators.D.outerSize(); ++col)
    for (SparseMatrix::InnerIterator it(operators.D, col); it; ++it)
    {
      add(it.row() / nv, dim, col / nv, system.fieldOf(col), it.value());
      add(col / nv, system.fieldOf(col), it.row() / nv, dim, it.value());
    }
  for (Index col = 0; col < operators.laplacian.outerSize(); ++col)
    for (SparseMatrix::InnerIterator it(operators.laplacian, col); it; ++it)
      if (it.row() % nv == m && col % nv == m)
        add(it.row() / nv, dim, col / nv, dim, -kRegularization * it.value());
  const Index nodes = system.problem.mesh().numNodes();
  for (Index i = 0; i < nodes; ++i)
    if (system.numbering.owned[i] && system.numbering.node[i] >= 0)
      for (int f = 0; f <= dim; ++f)
        if (!system.free(i, f))
          entries.push_back({system.row(i, f), system.row(i, f), 1.0});
  return entries;
}

/// The largest size of a velocity component of @p u over every process.
double
largestComponent(const StartSystem & system, const Vector & u)
{
  double largest = 0.0;
  for (Index i = 0; i < system.problem.mesh().numNodes(); ++i)
    for (int v : system.velocities)
      largest = std::max(largest, std::abs(u[system.problem.dof(i, v)]));
  return system.problem.parallel().maximum(largest);
}

/// The update of the velocity @p velocity_update at its unknowns, with the values of the owners
/// and of the dependent unknowns, and of the multiplier @p multiplier_update at those of m, from
/// the solution @p x of the saddle point system.
void
unpack(const StartSystem & system,
       const Vector & x,
       Vector & velocity_update,
       Vector & multiplier_update)
{
  const Problem & problem = system.problem;
  const int dim = system.dim;
  velocity_update = Vector::Zero(problem.numDofs());
  multiplier_update = Vector::Zero(problem.numDofs());
  for (Index i = 0; i < problem.mesh().numNodes(); ++i)
    if (system.numbering.owned[i])
      for (int f = 0; f <= dim; ++f)
        if (system.free(i, f))
          (f < dim ? velocity_update[problem.dof(i, system.velocities[f])]
                   : multiplier_update[problem.dof(i, system.m)]) = x[system.at(i, f)];
  problem.parallel().copyFromOwners(velocity_update.data(), system.nv);
  problem.parallel().copyFromOwners(multiplier_update.data(), system.nv);
  problem.updateDependentDofs(velocity_update);
}

/// Set the free rows of @p b to the residual of the saddle point system without its
/// regularization at the correction @p du and the multiplier @p lambda: the momentum rows
/// - (M du + D^T lambda) and the divergence left, - D u_0 - D du, with - D u_0 the multiplier rows
/// @p rhs.
void
setResidual(const StartSystem & system,
            const Vector & rhs,
            const Vector & du,
            const Vector & lambda,
            Vector & b)
{
  const Problem & problem = system.problem;
  const int dim = system.dim;
  Vector momentum = system.operators.M * du + system.operators.D.transpose() * lambda;
  problem.parallel().addAtNodes(momentum.data(), system.nv);
  const Vector divergence_left = rhs - system.divergence(du);
  for (Index i = 0; i < problem.mesh().numNodes(); ++i)
    if (system.numbering.owned[i])
      for (int f = 0; f <= dim; ++f)
        if (system.free(i, f))
          b[system.at(i, f)] = f < dim ? -momentum[problem.dof(i, system.velocities[f])]
                                       : divergence_left[problem.dof(i, system.m)];
}

/// The velocity correction du, in the unknowns of the problem, of the saddle point system with
/// the multiplier rows @p rhs = - D u_0 at the initial velocity @p solution.
/// The regularization moves the smooth part of the solution by about eps relative to its size,
/// which corrections with the residual of the system without it take back (iterated
/// regularization), until a correction changes the velocity by less than kSettledChange of the
/// speed.  A spurious mode that D^T nearly annihilates keeps a part of its divergence, which a
/// velocity of bounded size cannot remove.
Vector
solveCorrection(const StartSystem & system, const Vector & rhs, const Vector & solution)
{
  const Problem & problem = system.problem;
  const Parallel & parallel = problem.parallel();
  const ProjectionSolver::Rows & numbering = system.numbering;
  const int dim = system.dim, m = system.m;
  const Index nodes = problem.mesh().numNodes();
  const std::vector<petsc::GlobalEntry> entries = saddleEntries(system);
  // The default options factorise the system directly, which on several processes needs MUMPS.
  if (numbering.across && Communicator::world().size() > 1 && !petsc::haveMumps())
    throw InputError("The divergence-free projection of the initial velocity on several "
                     "processes needs PETSc built with MUMPS, which this PETSc lacks. Configure "
                     "PETSc with --download-mumps.");
  petsc::Settings settings;
  settings.relative_tolerance = kSolveTolerance;
  settings.symmetric = false;
  const petsc::LinearSolver solver(numbering.across,
                                   numbering.num_global,
                                   numbering.first,
                                   numbering.num_owned,
                                   entries,
                                   settings);
  // The velocity correction and the multiplier, one per node.
  Vector du = Vector::Zero(problem.numDofs());
  Vector lambda = Vector::Zero(problem.numDofs());
  Vector b = Vector::Zero(numbering.num_owned);
  for (Index i = 0; i < nodes; ++i)
    if (numbering.owned[i] && system.free(i, dim))
      b[system.at(i, dim)] = rhs[problem.dof(i, m)];
  const double speed0 = largestComponent(system, solution);
  for (int pass = 0; pass < kMaxSolves; ++pass)
  {
    petsc::Result result;
    const Vector x = solver.solve(b, nullptr, result);
    if (parallel.any(!result.converged))
      throw std::runtime_error("dualmesh: the projection of the initial velocity onto a field "
                               "without divergence did not converge (" +
                               result.reason + ", " + std::to_string(result.iterations) +
                               " iterations).");
    Vector velocity_update, multiplier_update;
    unpack(system, x, velocity_update, multiplier_update);
    du += velocity_update;
    lambda += multiplier_update;
    double moved = 0.0, reached = speed0;
    for (Index i = 0; i < nodes; ++i)
      for (int v : system.velocities)
      {
        moved = std::max(moved, std::abs(velocity_update[problem.dof(i, v)]));
        reached = std::max(reached, std::abs(du[problem.dof(i, v)]));
      }
    if (parallel.maximum(moved) <= kSettledChange * parallel.maximum(reached))
      break;
    setResidual(system, rhs, du, lambda, b);
  }
  return du;
}

/// Add the correction @p du to the free velocity of @p solution, and return the largest change
/// of the velocity over the largest speed after it, on every process alike.
double
applyCorrection(const StartSystem & system, const Vector & du, Vector & solution)
{
  const Problem & problem = system.problem;
  double change = 0.0, speed = 0.0;
  for (Index i = 0; i < problem.mesh().numNodes(); ++i)
  {
    double change2 = 0.0, speed2 = 0.0;
    for (int d = 0; d < system.dim; ++d)
    {
      const Index dof = problem.dof(i, system.velocities[d]);
      double & u = solution[dof];
      // A dependent unknown follows those it depends on.
      if (!system.velocity_fixed[d][i] || problem.isDependent(dof))
      {
        u += du[dof];
        change2 += du[dof] * du[dof];
      }
      speed2 += u * u;
    }
    change = std::max(change, std::sqrt(change2));
    speed = std::max(speed, std::sqrt(speed2));
  }
  change = problem.parallel().maximum(change);
  speed = problem.parallel().maximum(speed);
  return speed > 0.0 ? change / speed : 0.0;
}

} // namespace

double
projectInitialVelocity(Problem & problem,
                       const std::vector<int> & velocities,
                       int pressure,
                       Vector & solution)
{
  problem.initialize();
  const int nv = problem.numVariables();
  std::vector<char> is_velocity(static_cast<std::size_t>(nv), 0);
  for (int v : velocities)
    is_velocity[static_cast<std::size_t>(v)] = 1;
  // The norm of the kinetic energy: the storage of the momentum equations.
  std::vector<const Kernel *> storage;
  for (const auto & k : problem.kernels())
    if (k->isTimeKernel() && is_velocity[static_cast<std::size_t>(k->variable())])
      storage.push_back(k.get());
  if (storage.empty())
    return 0.0;
  // The multiplier is tested with the functions of the pressure, or of the first velocity for a
  // flow without a pressure.
  const int m = pressure >= 0 ? pressure : velocities[0];
  const StartOperators operators = assembleOperators(problem, velocities, is_velocity, m, storage);
  StartSystem system(problem, velocities, m, operators);
  system.fixed = fixedMultipliers(system, pressure);
  system.velocity_fixed = fixedVelocities(system);
  const bool enclosed = isEnclosed(system, pressure);
  // The rows of the multiplier: its equation is D u = - D u_0.
  Vector rhs = -system.divergence(solution);
  if (enclosed)
    spreadDivergence(system, rhs);
  const Vector du = solveCorrection(system, rhs, solution);
  return applyCorrection(system, du, solution);
}

} // namespace dualmesh
