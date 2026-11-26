// SPDX-License-Identifier: LGPL-2.1-or-later
#include "dualmesh/modules/Projection.h"
#include "dualmesh/base/Factory.h"
#include "dualmesh/core/InputParameters.h"
#include "dualmesh/fe/Assembly.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>

namespace dualmesh
{

namespace
{

/// The Jacobian at rest of @p kernels and @p conditions, the operator of terms linear in the unknowns, with every dependent unknown folded into those it depends on (C^T J C, Problem::foldDependents).
/// A time kernel takes the step one, so that its operator is the coefficient of the rate.
SparseMatrix
operatorOf(Problem & problem,
           const std::vector<const Kernel *> & kernels,
           const std::vector<const IntegratedBC *> & conditions = {},
           bool stabilization = true)
{
  Problem::AssemblyOptions options;
  options.kernels = &kernels;
  options.conditions = &conditions;
  options.dt = 1.0;
  options.stabilization = stabilization;
  const Vector rest = Vector::Zero(problem.numDofs());
  options.old = &rest;
  Vector R;
  SparseMatrix J;
  problem.assemble(rest, options, R, &J);
  problem.foldDependents(R, &J);
  return J;
}

/// The part of @p A whose rows and columns are of the variables that @p rows and @p columns mark.
SparseMatrix
block(const Problem & problem,
      const SparseMatrix & A,
      const std::vector<char> & rows,
      const std::vector<char> & columns)
{
  const int nv = problem.numVariables();
  std::vector<Eigen::Triplet<double>> entries;
  for (Index col = 0; col < A.outerSize(); ++col)
    for (SparseMatrix::InnerIterator it(A, col); it; ++it)
      if (rows[static_cast<std::size_t>(it.row() % nv)] &&
          columns[static_cast<std::size_t>(col % nv)])
        entries.emplace_back(it.row(), col, it.value());
  SparseMatrix out(A.rows(), A.cols());
  out.setFromTriplets(entries.begin(), entries.end());
  return out;
}

/// A kernel or a boundary condition that only this file holds, set up with @p problem.
template <typename T>
std::shared_ptr<T>
privateObject(Problem & problem,
              const std::string & type,
              const std::string & name,
              const std::vector<std::pair<std::string, ParameterValue>> & values)
{
  const auto & factory = Factory::instance();
  InputParameters parameters = factory.validParams(type);
  for (const auto & [key, value] : values)
    parameters.set(key, value);
  auto object = std::dynamic_pointer_cast<T>(factory.create(type, name, parameters));
  object->initialSetup(problem);
  return object;
}

/// The flow coefficients of the mass equation @p constraint at the point @p x.
FlowCoefficients
coefficientsAt(const Problem & problem, const Kernel & constraint, const Point & x)
{
  QpContext probe;
  probe.dim = problem.mesh().spaceDimension();
  probe.time = problem.time();
  probe.x = x;
  FlowCoefficients c;
  constraint.flowCoefficients(probe, c);
  return c;
}

/// The term u_r / r of the divergence in axisymmetric coordinates (2 u_r / r in spherical ones), as its factor at @p x.
double
hoopFactor(const Problem & problem, const Point & x)
{
  if (problem.coordinateSystem() == CoordinateSystem::Cartesian || !(x[0] > 0.0))
    return 0.0;
  return (problem.coordinateSystem() == CoordinateSystem::Axisymmetric ? 1.0 : 2.0) / x[0];
}

} // namespace

/// The rows of a linear system over some variables of a problem, in the global numbering of its nodes (Parallel::globalRows): the row of the k-th variable at a node is width G + k, with G the global number of the node.
/// The equation of a fixed unknown (a prescribed value, a dependent or an inactive unknown) is a row of the identity, and its value moves to the right-hand side of the others.
struct ProjectionSolver::Rows
{
  Rows(const Problem & problem_, const std::vector<int> & variables_, std::vector<char> fixed_)
      : problem(problem_), variables(variables_), fixed(std::move(fixed_))
  {
    nv = problem.numVariables();
    width = static_cast<int>(variables.size());
    slot.assign(static_cast<std::size_t>(nv), -1);
    for (int k = 0; k < width; ++k)
      slot[static_cast<std::size_t>(variables[k])] = k;
    const GlobalRows rows = problem.parallel().globalRows(problem);
    across = rows.distributed;
    const Index nodes = problem.mesh().numNodes();
    node.assign(static_cast<std::size_t>(nodes), -1);
    owned.assign(static_cast<std::size_t>(nodes), 0);
    for (Index i = 0; i < nodes; ++i)
    {
      const Index g = rows.global_dof[problem.dof(i, 0)];
      node[i] = g < 0 ? -1 : g / nv;
      owned[i] = rows.owned_dof[problem.dof(i, 0)];
    }
    first = rows.first_row / nv * width;
    num_owned = rows.num_owned / nv * width;
    num_global = rows.global_size / nv * width;
  }

  /// The row of the unknown @p d of the problem, or -1 when it is not in the system.
  Index row(Index d) const
  {
    const int k = slot[static_cast<std::size_t>(d % nv)];
    const Index g = node[static_cast<std::size_t>(d / nv)];
    return k < 0 || g < 0 ? -1 : width * g + k;
  }
  /// Call @p visit with every unknown of the system that this process owns and its row.
  template <typename Visit> void forEachOwned(Visit && visit) const
  {
    for (Index i = 0; i < static_cast<Index>(node.size()); ++i)
      if (owned[i] && node[i] >= 0)
        for (int k = 0; k < width; ++k)
          visit(problem.dof(i, variables[k]), width * node[i] + k);
  }
  /// The entries of the matrix @p A of the unknowns of the problem, in the order of its entries, with the rows and columns of the fixed unknowns replaced by the identity, which keeps a symmetric matrix symmetric.
  std::vector<petsc::GlobalEntry> entries(const SparseMatrix & A) const
  {
    std::vector<petsc::GlobalEntry> out;
    out.reserve(static_cast<std::size_t>(A.nonZeros() + num_owned));
    for (Index col = 0; col < A.outerSize(); ++col)
      for (SparseMatrix::InnerIterator it(A, col); it; ++it)
      {
        const Index r = row(it.row()), c = row(col);
        if (r < 0 || c < 0 || fixed[it.row()] || fixed[col])
          continue;
        out.push_back({r, c, it.value()});
      }
    forEachOwned(
        [&](Index d, Index r)
        {
          if (fixed[d])
            out.push_back({r, r, 1.0});
        });
    return out;
  }
  std::unique_ptr<petsc::LinearSolver> makeSolver(const SparseMatrix & A,
                                                  const std::string & options,
                                                  double tolerance,
                                                  bool symmetric,
                                                  int projection_vectors = 0) const
  {
    petsc::Settings settings;
    settings.options = options;
    settings.relative_tolerance = tolerance;
    settings.projection_vectors = projection_vectors;
    settings.symmetric = symmetric;
    return std::make_unique<petsc::LinearSolver>(
        across, num_global, first, num_owned, entries(A), settings);
  }
  /// Solve A x = b for the unknowns of the system: @p rhs is this process's part of b (the
  /// contributions of its elements), @p values holds the fixed values and a first guess
  /// elsewhere, and receives the solution, with the values of the owners and of the dependent
  /// unknowns.  With @p remove_mean the mean of b over the unknowns of the problem is removed
  /// first, which makes a Neumann problem solvable when its discretization leaves a remainder.
  int solve(const petsc::LinearSolver & solver,
            const SparseMatrix & A,
            Vector rhs,
            Vector & values,
            bool remove_mean) const
  {
    Vector prescribed = Vector::Zero(values.size());
    forEachFixed([&](Index d) { prescribed[d] = values[d]; });
    rhs -= A * prescribed;
    problem.foldDependents(rhs, nullptr);
    problem.parallel().addAtNodes(rhs.data(), nv);
    if (remove_mean)
    {
      const std::vector<char> constrained = problem.constrainedDofs();
      double total = 0.0, count = 0.0;
      forEachOwned(
          [&](Index d, Index)
          {
            if (!constrained[d])
            {
              total += rhs[d];
              count += 1.0;
            }
          });
      const double mean = problem.parallel().sum(total) / problem.parallel().sum(count);
      for (Index d = 0; d < rhs.size(); ++d)
        if (row(d) >= 0)
          rhs[d] -= mean;
    }
    // A dependent or an inactive unknown is no unknown of the system: its row of the identity
    // holds zero, so that it does not enter the norm of b, which the tolerance is relative to, and
    // its value follows the solve.
    const auto & active = problem.activeDofs();
    Vector b(num_owned), guess(num_owned);
    forEachOwned(
        [&](Index d, Index r)
        {
          const bool none = problem.isDependent(d) || (!active.empty() && !active[d]);
          b[r - first] = none ? 0.0 : fixed[d] ? values[d] : rhs[d];
          guess[r - first] = none ? 0.0 : values[d];
        });
    petsc::Result result;
    const Vector x = solver.solve(b, &guess, result);
    if (problem.parallel().any(!result.converged))
      throw std::runtime_error(
          "dualmesh: a linear solve of the projection time integration did not "
          "converge (" +
          result.reason + ", " + std::to_string(result.iterations) + " iterations).");
    forEachOwned([&](Index d, Index r) { values[d] = x[r - first]; });
    problem.parallel().copyFromOwners(values.data(), nv);
    problem.updateDependentDofs(values);
    return result.iterations;
  }
  template <typename Visit> void forEachFixed(Visit && visit) const
  {
    for (Index d = 0; d < static_cast<Index>(fixed.size()); ++d)
      if (fixed[d] && row(d) >= 0)
        visit(d);
  }

  const Problem & problem;
  std::vector<int> variables;
  int nv = 0, width = 0;
  /// The place of every variable of the problem in the system, or -1.
  std::vector<int> slot;
  /// The global number of every local node (-1: one that no process numbers), and whether this process owns it.
  std::vector<Index> node;
  std::vector<char> owned;
  /// One flag per unknown of the problem.
  std::vector<char> fixed;
  Index first = 0, num_owned = 0, num_global = 0;
  bool across = false;
};

/// The geometry of the quadrature points of every element of the flow, computed once for the right-hand sides of the steps: the weight (with the Jacobian and the coordinate factor), the position and the inverse transposed Jacobian at each point, and the shape functions and their reference derivatives once per element type, with those of its corners for a pressure of first order (the Taylor-Hood element).
/// There are two sets of points: those of the element interiors (FEM) or of the control domains, which carry the node that owns them, and, for DMCDM, those of the interfaces between control domains, which carry their two nodes and their area vector.
/// FEM takes a Gauss rule of order + 1 points per direction, which integrates the advection exactly on affine elements.
/// DMCDM takes order points per direction of every patch of a control domain and of every interface, the midpoint rule for linear elements, which is second order like the method itself: the errors of the Taylor-Green vortex are those of two points per direction, and a step costs 2.4 times less.
/// Under DMCDM a cell with nodes hanging on its sides is a polygon (PolygonCell), whose points are those of the monolithic assembly (buildElementPoints): its control domains and interfaces belong to its corners and to its hanging nodes, and the fields inside are those of its corners.
struct ProjectionSolver::Points
{
  struct Table
  {
    int nodes = 0, corners = 0;
    std::vector<double> N, Nc;  // [point][node]
    std::vector<Point> dN, dNc; // [point][node], reference derivatives
  };
  struct Set
  {
    std::map<ElementType, Table> tables;
    /// The shape functions of the points of every polygon cell, whose points are its own, and the table of each element (-1 for an element that takes the table of its type).
    std::vector<Table> polygon_tables;
    std::vector<int> polygon_table;
    std::vector<Index> first; // the first point of each element
    std::vector<double> weight;
    std::vector<Point> area, x;
    std::vector<std::array<double, 9>> G; // J^{-T}, row major
    /// The nodes of the mesh that own a point and that lie across its interface, or -1 for FEM.
    std::vector<Index> owner, neighbor;

    int count(Index e) const { return static_cast<int>(first[e + 1] - first[e]); }

    /// Fill @p mp with the shape functions and physical gradients at point @p q of element @p e, with those of its corners when the table holds them.
    void at(const Element & el, Index e, int q, MappedPoint & mp) const
    {
      const int polygon = polygon_table[static_cast<std::size_t>(e)];
      const Table & t =
          polygon >= 0 ? polygon_tables[static_cast<std::size_t>(polygon)] : tables.at(el.type);
      const Index point = first[e] + q;
      const auto & g = G[point];
      const auto physical = [&](const Point & r) -> Point
      {
        return {g[0] * r[0] + g[1] * r[1] + g[2] * r[2],
                g[3] * r[0] + g[4] * r[1] + g[5] * r[2],
                g[6] * r[0] + g[7] * r[1] + g[8] * r[2]};
      };
      for (int a = 0; a < t.nodes; ++a)
      {
        mp.N[a] = t.N[q * t.nodes + a];
        mp.dN[a] = physical(t.dN[q * t.nodes + a]);
      }
      mp.num_corners = t.corners;
      for (int a = 0; a < t.corners; ++a)
      {
        mp.corner_shape[a] = t.Nc[q * t.corners + a];
        mp.corner_gradient[a] = physical(t.dNc[q * t.corners + a]);
      }
      mp.x = x[point];
    }

    /// The points of @p set of every element of the flow of @p problem, with the corner functions when @p corners holds for the element.
    void build(const Problem & problem,
               const std::vector<char> & in_flow,
               PointSet set,
               const std::function<bool(const Element &)> & corners)
    {
      const Mesh & mesh = problem.mesh();
      const bool dual = problem.isDual();
      const Index elements = static_cast<Index>(in_flow.size());
      std::vector<IntegrationPoint> points;
      MappedPoint mp;
      first.assign(static_cast<std::size_t>(elements) + 1, 0);
      polygon_table.assign(static_cast<std::size_t>(elements), -1);
      const auto fill = [&](Table & table, const Element & el, const ReferenceElement & ref)
      {
        table.nodes = el.numNodes();
        const bool linear = corners(el);
        table.corners = linear ? el.numCorners() : 0;
        for (const auto & p : points)
        {
          double N[kMaxElementNodes];
          Point dN[kMaxElementNodes];
          ref.shape(p.xi, N, dN);
          table.N.insert(table.N.end(), N, N + el.numNodes());
          table.dN.insert(table.dN.end(), dN, dN + el.numNodes());
          if (!linear)
            continue;
          ReferenceElement::get(elementCornerType(el.type)).shape(p.xi, N, dN);
          table.Nc.insert(table.Nc.end(), N, N + el.numCorners());
          table.dNc.insert(table.dNc.end(), dN, dN + el.numCorners());
        }
      };
      for (Index e = 0; e < elements; ++e)
      {
        first[e + 1] = first[e];
        if (!in_flow[e])
          continue;
        const Element & el = mesh.element(e);
        const auto & ref = ReferenceElement::get(el.type);
        QuadratureSpec spec;
        spec.points = dual ? ref.order() : ref.order() + 1;
        const PolygonCell * polygon = dual ? problem.polygonCell(e) : nullptr;
        buildElementPoints(mesh, e, dual, set, spec, points, nullptr, polygon);
        if (polygon)
        {
          polygon_table[static_cast<std::size_t>(e)] = static_cast<int>(polygon_tables.size());
          polygon_tables.emplace_back();
          fill(polygon_tables.back(), el, ref);
        }
        else
        {
          auto [table, added] = tables.try_emplace(el.type);
          if (added)
            fill(table->second, el, ref);
        }
        // The node of a slot: a corner of the element, or a node that hangs on its sides.
        const auto nodeOf = [&](int slot) -> Index
        {
          if (slot < 0)
            return -1;
          return polygon ? polygon->nodes[static_cast<std::size_t>(slot)] : el.nodes[slot];
        };
        first[e + 1] = first[e] + static_cast<Index>(points.size());
        for (const auto & p : points)
        {
          mapPoint(mesh, el, p.xi, mp);
          // J^{-T}: the physical gradient is G times the reference gradient.
          const auto & inv = mp.Jinv;
          std::array<double, 9> g{};
          for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
              g[3 * i + j] = inv[j][i];
          G.push_back(g);
          const double factor = problem.coordFactor(mp.x);
          weight.push_back(p.weight * factor);
          area.push_back(factor * p.area);
          x.push_back(mp.x);
          owner.push_back(nodeOf(p.owner));
          neighbor.push_back(nodeOf(p.neighbor));
        }
      }
    }
  };
  Set volume, faces;
};

ProjectionSolver::ProjectionSolver(Problem & problem, ProjectionSettings settings)
    : _problem(problem), _settings(std::move(settings))
{
  if (_settings.order < 1 || _settings.order > 3)
    throw InputError("Projection: the order of the time integration is 1, 2 or 3.");
}

ProjectionSolver::~ProjectionSolver() = default;

void
ProjectionSolver::coefficients(const std::vector<double> & times,
                               std::vector<double> & beta,
                               std::vector<double> & alpha)
{
  // beta_j / dt is the derivative at t^n of the Lagrange basis polynomial of t^{n-j} on the levels t^n, ..., t^{n-k}, and alpha_j the value at t^n of the Lagrange basis polynomial of t^{n-j} on the levels t^{n-1}, ..., t^{n-k}.
  const int k = static_cast<int>(times.size()) - 1;
  const double dt = times[0] - times[1];
  beta.assign(static_cast<std::size_t>(k + 1), 0.0);
  alpha.assign(static_cast<std::size_t>(k + 1), 0.0);
  for (int j = 0; j <= k; ++j)
  {
    double derivative = 0;
    if (j == 0)
      for (int m = 1; m <= k; ++m)
        derivative += 1.0 / (times[0] - times[m]);
    else
    {
      double numerator = 1, denominator = 1;
      for (int m = 0; m <= k; ++m)
        if (m != j)
        {
          denominator *= times[j] - times[m];
          if (m != 0)
            numerator *= times[0] - times[m];
        }
      derivative = numerator / denominator;
    }
    beta[j] = dt * derivative;
  }
  for (int j = 1; j <= k; ++j)
  {
    double value = 1;
    for (int m = 1; m <= k; ++m)
      if (m != j)
        value *= (times[0] - times[m]) / (times[j] - times[m]);
    alpha[j] = value;
  }
}

void
ProjectionSolver::setUp()
{
  _problem.initialize();
  const Mesh & mesh = _problem.mesh();
  const int nv = _problem.numVariables();
  const Kernel * constraint = _problem.flowConstraint();
  if (!constraint)
    throw InputError("Projection: the projection time integration advances a flow with a "
                     "pressure, the variable of its mass_conservation equation, and this problem "
                     "has none (the penalty formulation eliminates the pressure).");
  _pressure = constraint->variable();
  const FlowCoefficients flow = coefficientsAt(_problem, *constraint, mesh.node(0));
  _velocity.assign(flow.velocities, flow.velocities + flow.num_velocities);
  _stabilized = flow.stabilized;
  if (nv != flow.num_velocities + 1)
    throw InputError("Projection: the projection time integration advances the velocity and the "
                     "pressure of a flow alone, and this problem has other variables as well.");
  std::vector<char> is_velocity(static_cast<std::size_t>(nv), 0), is_pressure(is_velocity);
  for (int v : _velocity)
    is_velocity[static_cast<std::size_t>(v)] = 1;
  is_pressure[static_cast<std::size_t>(_pressure)] = 1;

  // ---- the terms of the momentum equations, by their roles ----------------------------------
  const auto refuse = [](const ResidualObject & object)
  {
    throw InputError("Projection: '" + object.name() +
                     "' acts on the flow in a way that the projection time integration does not "
                     "split: a term of the mass equation other than mass_conservation and "
                     "mass_flux, or a boundary condition of the momentum equations other than a "
                     "prescribed velocity. Prescribe the velocity or the pressure on the "
                     "boundary, or use time_integration='monolithic'.");
  };
  std::vector<const Kernel *> storage, implicit;
  _sources.clear();
  _advection = false;
  for (const auto & k : _problem.kernels())
  {
    if (k->variable() == _pressure)
    {
      if (k.get() != constraint)
        refuse(*k);
      continue;
    }
    if (k->isTimeKernel())
      storage.push_back(k.get());
    else if (k->isStabilization())
      continue;
    else if (k->isInertia())
      _advection = true;
    else if (k->hasFlux())
      implicit.push_back(k.get());
    else
      _sources.push_back(k.get());
  }
  for (const auto & bc : _problem.integratedBCs())
    if (is_velocity[static_cast<std::size_t>(bc->variable())] ||
        (bc->variable() == _pressure && bc->type() != "mass_flux"))
      refuse(*bc);

  // ---- the density and the viscosity ----------------------------------------------------------
  // The pressure equation divides the momentum equations by the density, so the fluid is one of
  // uniform density.
  _density = flow.time_density;
  for (Index i = 0; i < mesh.numNodes(); ++i)
  {
    const FlowCoefficients c = coefficientsAt(_problem, *constraint, mesh.node(i));
    const double scale = std::abs(_density);
    if (std::abs(c.time_density - _density) > 1e-12 * scale ||
        (_advection && std::abs(c.density - _density) > 1e-12 * scale))
      throw InputError("Projection: the projection time integration takes a fluid of one "
                       "uniform density, the coefficient of the time derivative and of the "
                       "advection of its velocity.");
  }
  if (storage.empty() || !(_density > 0.0))
    throw InputError("Projection: the projection time integration needs a positive density. A "
                     "flow without inertia (Stokes flow) has no time derivative to integrate.");
  const Index elements = _problem.numIntegratedElements();
  const auto & blocks = _problem.variable(_velocity[0]).blocks;
  _in_flow.assign(static_cast<std::size_t>(elements), 0);
  _element_size.assign(static_cast<std::size_t>(elements), 0.0);
  _viscosity.assign(static_cast<std::size_t>(elements), 0.0);
  for (Index e = 0; e < elements; ++e)
  {
    const Element & el = mesh.element(e);
    _in_flow[e] = blocks.empty() || blocks.count(el.block) ? 1 : 0;
    _element_size[e] = elementSize(mesh, e);
    Point centre{0, 0, 0};
    for (int a = 0; a < el.numNodes(); ++a)
      centre = centre + (1.0 / el.numNodes()) * mesh.node(el.nodes[a]);
    _viscosity[e] = coefficientsAt(_problem, *constraint, centre).viscosity / _density;
  }
  std::set<std::pair<Index, int>> sides;
  for (const auto & [name, list] : mesh.sidesets())
    for (const Side & side : list)
      if (side.first < elements && _in_flow[side.first])
        sides.insert({side.first, side.second});
  _boundary_sides.assign(sides.begin(), sides.end());

  // ---- the operators, from the problem's assembly -----------------------------------------
  _mass = block(_problem, operatorOf(_problem, storage), is_velocity, is_velocity);
  const SparseMatrix terms = operatorOf(_problem, implicit);
  _viscous = block(_problem, terms, is_velocity, is_velocity);
  _gradient = block(_problem, terms, is_velocity, is_pressure);
  const auto laplacian = privateObject<Kernel>(_problem,
                                               "diffusion",
                                               "projection_pressure_laplacian",
                                               {{"variable", _problem.variable(_pressure).name}});
  _laplacian = operatorOf(_problem, {laplacian.get()});
  _helmholtz = _viscous + _mass;
  _helmholtz.makeCompressed();
  _helmholtz_c = -1.0;

  // ---- the points of the right-hand sides --------------------------------------------------------
  _points = std::make_unique<Points>();
  const auto corners = [&](const Element & el) { return _problem.usesCornerShape(el, _pressure); };
  _points->volume.build(_problem, _in_flow, PointSet::Volume, corners);
  if (_problem.isDual())
    _points->faces.build(_problem, _in_flow, PointSet::Faces, corners);

  // ---- the systems ------------------------------------------------------------------------------
  const std::vector<char> constrained = _problem.constrainedDofs();
  const auto & active = _problem.activeDofs();
  const bool dual = _problem.isDual();
  // A boundary without a prescribed velocity, open to the outside, takes the pressure from the
  // traction there, which the pressure equation needs as its condition on that boundary.
  std::set<std::string> periodic;
  for (const auto & constraint_object : _problem.constraints())
    if (constraint_object->type() == "periodic")
      for (const char * key : {"primary", "secondary"})
        periodic.insert(constraint_object->parameters().getString(key));
  for (const auto & [name, list] : mesh.sidesets())
  {
    if (periodic.count(name))
      continue;
    for (const Side & side : list)
    {
      if (side.first >= elements || !_in_flow[side.first])
        continue;
      for (Index node : mesh.sideNodes(side))
      {
        bool open = !constrained[_problem.dof(node, _pressure)];
        for (int v : _velocity)
          open = open && !constrained[_problem.dof(node, v)];
        if (open)
          throw InputError("Projection: the side set '" + name +
                           "' is open, with no velocity prescribed, and the projection time "
                           "integration takes the pressure on an open boundary from a condition. "
                           "Prescribe the pressure there, whose value completes the outflow "
                           "condition of the viscous form, or use time_integration='monolithic'.");
      }
    }
  }
  bool coupled = false;
  for (Index col = 0; col < _viscous.outerSize() && !coupled; ++col)
    for (SparseMatrix::InnerIterator it(_viscous, col); it; ++it)
      coupled = coupled || (it.row() % nv != col % nv && it.value() != 0.0);
  coupled = _problem.parallel().any(coupled);
  _velocity_rows.clear();
  if (coupled)
    _velocity_rows.push_back(std::make_unique<Rows>(_problem, _velocity, constrained));
  else
    for (int v : _velocity)
      _velocity_rows.push_back(std::make_unique<Rows>(_problem, std::vector<int>{v}, constrained));
  // An enclosed flow, whose pressure no condition gives, determines it up to a constant: the
  // constant is fixed at the free unknown of the smallest row, after the right-hand side is made
  // compatible, and the pressure is then shifted to a mean of zero, as the monolithic solve
  // reports it.
  std::vector<char> pressure_fixed = constrained;
  _pressure_rows = std::make_unique<Rows>(_problem, std::vector<int>{_pressure}, constrained);
  bool given = false;
  _pressure_rows->forEachOwned(
      [&](Index d, Index)
      { given = given || (constrained[d] && active[d] && !_problem.isDependent(d)); });
  _pressure_pinned = !_problem.parallel().any(given);
  if (_pressure_pinned)
  {
    double smallest = std::numeric_limits<double>::infinity();
    _pressure_rows->forEachOwned(
        [&](Index d, Index r)
        {
          if (!constrained[d])
            smallest = std::min(smallest, static_cast<double>(r));
        });
    smallest = _problem.parallel().minimum(smallest);
    for (Index d = 0; d < _problem.numDofs(); ++d)
      if (static_cast<double>(_pressure_rows->row(d)) == smallest)
        pressure_fixed[d] = 1;
    _pressure_rows->fixed = pressure_fixed;
    // The weights of the mean of the pressure, those of the monolithic solve.
    _pressure_weight = _problem.nodeWeights(_pressure);
    _problem.parallel().addAtNodes(_pressure_weight.data(), 1);
    double total = 0.0;
    for (Index i = 0; i < mesh.numNodes(); ++i)
      if (_pressure_rows->owned[i])
        total += _pressure_weight[i];
    _total_weight = _problem.parallel().sum(total);
  }
  const std::string krylov = dual ? "-ksp_type gmres" : "-ksp_type cg";
  _pressure_solver = _pressure_rows->makeSolver(
      _laplacian,
      _settings.pressure_solver.empty() ? krylov + " -pc_type hypre -pc_hypre_type boomeramg"
                                        : _settings.pressure_solver,
      _settings.pressure_tolerance,
      !dual,
      _settings.pressure_projection_vectors);
  _velocity_solvers.clear();
  // The sources per unit mass are the L2 projection of their residual onto the velocity, a
  // solve with the mass matrix whose rows only the dependent and inactive unknowns fix.  A source
  // that does not change is projected once, so the solve goes to round-off, below the
  // tolerances of the steps.
  _source_solver.reset();
  _source_residual.resize(0);
  if (!_sources.empty())
  {
    std::vector<char> free_fixed(constrained.size(), 0);
    for (Index d = 0; d < _problem.numDofs(); ++d)
      free_fixed[d] = _problem.isDependent(d) || (!active.empty() && !active[d]) ? 1 : 0;
    _source_rows = std::make_unique<Rows>(_problem, _velocity, free_fixed);
    _source_solver = _source_rows->makeSolver(_mass, krylov + " -pc_type jacobi", 1e-14, !dual);
  }
  _ready = true;
}

void
ProjectionSolver::sources(const Vector & U, double time)
{
  if (_sources.empty())
  {
    _source.resize(0);
    return;
  }
  Problem::AssemblyOptions options;
  options.kernels = &_sources;
  const std::vector<const IntegratedBC *> none;
  options.conditions = &none;
  _problem.setTime(time);
  Vector R;
  _problem.assemble(U, options, R, nullptr);
  // A source that has not changed (a constant body force) keeps its projection.
  if (_source_residual.size() == R.size() && R == _source_residual)
    return;
  _source_residual = R;
  if (_source.size() != R.size())
    _source = Vector::Zero(R.size());
  // M s = - R, with the density in M: s is the source per unit mass.
  _source_rows->solve(*_source_solver, _mass, -R, _source, false);
}

void
ProjectionSolver::start(const Vector & solution, double time)
{
  if (!_ready)
    setUp();
  _history.clear();
  _history.push_front({time, solution});
  _trial = {};
  _kinematic_pressure = solution / _density;
}

double
ProjectionSolver::courantNumber(const Vector & solution, double dt) const
{
  const Mesh & mesh = _problem.mesh();
  double largest = 0;
  for (Index e = 0; e < static_cast<Index>(_in_flow.size()); ++e)
  {
    if (!_in_flow[e])
      continue;
    const Element & el = mesh.element(e);
    double speed2 = 0;
    for (int a = 0; a < el.numNodes(); ++a)
    {
      double s2 = 0;
      for (int v : _velocity)
      {
        const double u = solution[_problem.dof(el.nodes[a], v)];
        s2 += u * u;
      }
      speed2 = std::max(speed2, s2);
    }
    largest = std::max(largest, std::sqrt(speed2) / _element_size[e]);
  }
  return dt * _problem.parallel().maximum(largest);
}

void
ProjectionSolver::accept(const Vector & solution, double time)
{
  _history.push_front({time, solution});
  // The step takes the last k states, and the predictor the last order() + 1.
  while (static_cast<int>(_history.size()) > std::max(_settings.order, order() + 1))
    _history.pop_back();
  _trial = {};
}

bool
ProjectionSolver::predict(double dt, Vector & predicted, double & factor) const
{
  // The predictor of a step of order q = order() is the polynomial of degree q through the last
  // q + 1 accepted states, extrapolated to the end of the step, and its corrector the backward
  // difference of order q.  With the levels t_n (the end of the step) to t_{n-q-1} and A the
  // derivative of order q + 1 of the velocity over (q + 1)!, the corrector has the local error
  // (dt / beta_0) A prod_{m=1..q} (t_n - t_{n-m}) and the predictor
  // - A prod_{m=1..q+1} (t_n - t_{n-m}), so the error of the step is its difference from the
  // predictor times dt / (dt + beta_0 (t_n - t_{n-q-1})) (Milne's device).  For time_order 3,
  // whose velocity converges at the second order of the splitting, the pair is that of order 2.
  // The splitting error of the pressure is not in the estimate.
  const int q = order();
  if (static_cast<int>(_history.size()) < q + 1)
    return false;
  const double time = _history[0].first + dt;
  std::vector<double> times{time};
  for (int j = 0; j <= q; ++j)
    times.push_back(_history[static_cast<std::size_t>(j)].first);
  predicted = Vector::Zero(_history[0].second.size());
  for (int j = 0; j <= q; ++j)
  {
    double basis = 1.0;
    for (int m = 0; m <= q; ++m)
      if (m != j)
        basis *= (time - times[static_cast<std::size_t>(m + 1)]) /
                 (times[static_cast<std::size_t>(j + 1)] - times[static_cast<std::size_t>(m + 1)]);
    predicted += basis * _history[static_cast<std::size_t>(j)].second;
  }
  std::vector<double> beta, alpha;
  coefficients(std::vector<double>(times.begin(), times.end() - 1), beta, alpha);
  factor = dt / (dt + beta[0] * (time - times.back()));
  return true;
}

SolveResult
ProjectionSolver::step(
    Vector & solution, const Vector & old, double time, double dt, bool continues)
{
  const Mesh & mesh = _problem.mesh();
  const bool dual = _problem.isDual();
  const int dim = static_cast<int>(_velocity.size());
  const double rho = _density;
  // The levels of the step: the accepted states, after the state of the step just taken when
  // this step continues it (the second half of a doubled step).  The newest is @p old.
  (void) old;
  std::vector<const std::pair<double, Vector> *> levels;
  if (continues)
    levels.push_back(&_trial);
  for (const auto & level : _history)
    levels.push_back(&level);
  const int k = std::min<int>(_settings.order, static_cast<int>(levels.size()));
  std::vector<double> times{time};
  for (int j = 0; j < k; ++j)
    times.push_back(levels[static_cast<std::size_t>(j)]->first);
  if (std::abs((times[0] - times[1]) - dt) > 1e-9 * dt)
    throw std::logic_error(
        "Projection: a step starts from the last accepted solution or from the step just taken.");
  std::vector<double> beta, alpha;
  coefficients(times, beta, alpha);

  // The Helmholtz matrix changes with beta_0 / dt: in the first steps, at a step shortened to land on an output time, and at every step of the cfl stepper.
  // Its structure does not, so the solver is made once and later given the new values.
  const double c = beta[0] / dt;
  // A step of the same size gives the same coefficient up to the rounding of the times, which must not touch the solver.
  if (std::abs(c - _helmholtz_c) > 1e-9 * std::abs(c))
  {
    _helmholtz = c * _mass + _viscous;
    _helmholtz.makeCompressed();
    for (std::size_t i = 0; i < _velocity_rows.size(); ++i)
      if (_velocity_solvers.size() <= i)
        _velocity_solvers.push_back(_velocity_rows[i]->makeSolver(
            _helmholtz,
            _settings.velocity_solver.empty()
                ? std::string(dual ? "-ksp_type gmres" : "-ksp_type cg") + " -pc_type jacobi"
                : _settings.velocity_solver,
            _settings.velocity_tolerance,
            !dual));
      else
      {
        std::vector<double> values;
        for (const auto & entry : _velocity_rows[i]->entries(_helmholtz))
          values.push_back(entry.value);
        _velocity_solvers[i]->setValues(values);
      }
    _helmholtz_c = c;
  }

  // The extrapolated solution, with the values prescribed at the new time: the boundary velocity of the pressure equation, the velocity of the sources and the first guess of the solves.
  Vector extrapolated = Vector::Zero(solution.size());
  for (int j = 1; j <= k; ++j)
    extrapolated += alpha[j] * levels[static_cast<std::size_t>(j - 1)]->second;
  _problem.setTime(time);
  _problem.applyDirichlet(extrapolated, 1.0);
  sources(extrapolated, time);
  const bool with_source = _source.size() > 0;

  // ---- the right-hand sides: (grad q, u*) / dt for the pressure and the advection for the velocity ----
  QuadratureSpec spec;
  spec.points =
      mesh.numElements() > 0 && ReferenceElement::get(mesh.element(0).type).order() > 1 ? 4 : 3;
  // The velocity of the steps before, and its gradient, at one point: [level][component] and [level][component][direction].
  struct Scratch
  {
    std::vector<IntegrationPoint> points;
    MappedPoint mp;
    double u[3][3], grad[3][3][3];
    Vector pressure_rhs, advection_rhs;
  };
  const auto evaluate = [&](const Element & el, Scratch & sc)
  {
    for (int j = 0; j < k; ++j)
    {
      const Vector & U = levels[static_cast<std::size_t>(j)]->second;
      for (int d = 0; d < 3; ++d)
      {
        sc.u[j][d] = 0;
        for (int m = 0; m < 3; ++m)
          sc.grad[j][d][m] = 0;
      }
      for (int a = 0; a < el.numNodes(); ++a)
        for (int d = 0; d < dim; ++d)
        {
          const double value = U[_problem.dof(el.nodes[a], _velocity[d])];
          sc.u[j][d] += sc.mp.N[a] * value;
          for (int m = 0; m < dim; ++m)
            sc.grad[j][d][m] += sc.mp.dN[a][m] * value;
        }
    }
  };
  // The extrapolated advection (sum_j alpha_j (u^{n-j} . grad) u^{n-j}) and u* at the point of sc.mp.
  const auto computeStar =
      [&](const Element & el, const Scratch & sc, double advection[3], double star[3])
  {
    for (int d = 0; d < 3; ++d)
      advection[d] = star[d] = 0;
    for (int d = 0; d < dim; ++d)
    {
      for (int j = 1; j <= k; ++j)
      {
        double carried = 0;
        if (_advection)
          for (int m = 0; m < dim; ++m)
            carried += sc.u[j - 1][m] * sc.grad[j - 1][d][m];
        advection[d] += alpha[j] * carried;
        star[d] -= beta[j] * sc.u[j - 1][d];
      }
      star[d] -= dt * advection[d];
      if (with_source)
        for (int a = 0; a < el.numNodes(); ++a)
          star[d] += dt * sc.mp.N[a] * _source[_problem.dof(el.nodes[a], _velocity[d])];
    }
  };
  const Points & geo = *_points;
  const auto element = [&](Index e, Scratch & sc)
  {
    if (!_in_flow[e])
      return;
    const Element & el = mesh.element(e);
    const double h = _element_size[e], nu = _viscosity[e];
    const bool corner = _problem.usesCornerShape(el, _pressure);
    MappedPoint & mp = sc.mp;
    for (int q = 0; q < geo.volume.count(e); ++q)
    {
      geo.volume.at(el, e, q, mp);
      const Index point = geo.volume.first[e] + q;
      const double w = geo.volume.weight[point];
      evaluate(el, sc);
      // The splitting weighs the divergence of the velocity in the pressure equation by beta_0 / dt, as a pressure stabilization of parameter dt / beta_0 would.
      // That parameter vanishes with the step, and equal-order elements then lose their pressure stability, so the weight is capped at 1 / tau, with the parameter tau of Tezduyar (1991) of the monolithic solver without its time term (stabilizationTau): the scheme is unchanged where dt / beta_0 exceeds tau, and tends to the stabilized monolithic discretization as dt goes to zero.
      // The Taylor-Hood element is stable without it and keeps beta_0 / dt.
      // The weighted divergence is that of the combination - sum_j beta_j u^{n-j} / beta_0 of the steps before, which enters the pressure equation through u*.
      const double hoop = hoopFactor(_problem, mp.x);
      double speed2 = 0, divergence = 0;
      for (int d = 0; d < dim; ++d)
      {
        double value = 0;
        for (int a = 0; a < el.numNodes(); ++a)
          value += mp.N[a] * extrapolated[_problem.dof(el.nodes[a], _velocity[d])];
        speed2 += value * value;
        for (int j = 1; j <= k; ++j)
          divergence -=
              beta[j] / beta[0] * (sc.grad[j - 1][d][d] + (d == 0 ? hoop * sc.u[j - 1][0] : 0.0));
      }
      const double weight =
          _stabilized ? std::min(c, 1.0 / stabilizationTau(h, speed2, nu, 0.0)) : c;
      double advection[3], star[3];
      computeStar(el, sc, advection, star);
      if (dual)
      {
        // The integral over the control domain of the node that owns the point.
        // The part c of the weighted divergence is the flux out of the control domain (below), as the term (grad q, u*) / dt is, so that the two cancel where the velocity of the steps before has no divergence of its own, also where the velocity is not continuous across the sides of a cell with hanging nodes.
        const Index o = geo.volume.owner[point];
        for (int d = 0; d < dim; ++d)
          sc.advection_rhs[_problem.dof(o, _velocity[d])] += w * rho * advection[d];
        sc.pressure_rhs[_problem.dof(o, _pressure)] -= w * weight * divergence;
        continue;
      }
      for (int a = 0; a < el.numNodes(); ++a)
        for (int d = 0; d < dim; ++d)
          sc.advection_rhs[_problem.dof(el.nodes[a], _velocity[d])] +=
              w * rho * mp.N[a] * advection[d];
      const int tests = corner ? el.numCorners() : el.numNodes();
      for (int a = 0; a < tests; ++a)
      {
        const double test = corner ? mp.corner_shape[a] : mp.N[a];
        const Point & gradient = corner ? mp.corner_gradient[a] : mp.dN[a];
        double flux = 0;
        for (int d = 0; d < dim; ++d)
          flux += gradient[d] * star[d];
        sc.pressure_rhs[_problem.dof(el.nodes[a], _pressure)] +=
            w * (flux / dt + (c - weight) * test * divergence);
      }
    }
    if (!dual)
      return;
    // (grad q, u*) / dt of a control domain is - the flux of u* / dt out of it.
    for (int q = 0; q < geo.faces.count(e); ++q)
    {
      geo.faces.at(el, e, q, mp);
      const Index point = geo.faces.first[e] + q;
      evaluate(el, sc);
      double advection[3], star[3];
      computeStar(el, sc, advection, star);
      const Point & a = geo.faces.area[point];
      double flux = 0;
      for (int d = 0; d < dim; ++d)
      {
        flux += star[d] * a[d];
        for (int j = 1; j <= k; ++j)
          flux += dt * c * beta[j] / beta[0] * sc.u[j - 1][d] * a[d];
      }
      sc.pressure_rhs[_problem.dof(geo.faces.owner[point], _pressure)] -= flux / dt;
      sc.pressure_rhs[_problem.dof(geo.faces.neighbor[point], _pressure)] += flux / dt;
    }
  };
  // Every thread adds into vectors of its own.
  const int threads = _problem.effectiveThreads();
  std::vector<Scratch> scratch(static_cast<std::size_t>(threads));
  for (auto & sc : scratch)
  {
    sc.pressure_rhs = Vector::Zero(solution.size());
    sc.advection_rhs = Vector::Zero(solution.size());
  }
  const Index elements = static_cast<Index>(_in_flow.size());
  std::string failure;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(threads) if (threads > 1)
#endif
  for (Index e = 0; e < elements; ++e)
  {
    try
    {
#ifdef _OPENMP
      element(e, scratch[static_cast<std::size_t>(omp_get_thread_num())]);
#else
      element(e, scratch[0]);
#endif
    }
    catch (const std::exception & error)
    {
#ifdef _OPENMP
#pragma omp critical(dualmesh_projection_error)
#endif
      if (failure.empty())
        failure = error.what();
    }
  }
  if (!failure.empty())
    throw std::runtime_error(failure);
  Vector pressure_rhs = std::move(scratch[0].pressure_rhs);
  Vector advection_rhs = std::move(scratch[0].advection_rhs);
  for (std::size_t t = 1; t < scratch.size(); ++t)
  {
    pressure_rhs += scratch[t].pressure_rhs;
    advection_rhs += scratch[t].advection_rhs;
  }

  // ---- the boundary terms of the pressure equation -----------------------------------------
  // - (beta_0 / dt) int q n . u_b - nu int n . (omega x grad q), with omega the extrapolated vorticity.
  Scratch boundary;
  MappedPoint & mp = boundary.mp;
  for (const Side & side : _boundary_sides)
  {
    const Element & el = mesh.element(side.first);
    const bool corner = _problem.usesCornerShape(el, _pressure);
    const double nu = _viscosity[side.first];
    buildSidePoints(mesh, side, dual, spec, boundary.points);
    for (const auto & p : boundary.points)
    {
      mapPoint(mesh, el, p.xi, mp);
      if (corner)
        mapCornerShape(el, mp);
      evaluate(el, boundary);
      double ub[3] = {0, 0, 0};
      for (int a = 0; a < el.numNodes(); ++a)
        for (int d = 0; d < dim; ++d)
          ub[d] += mp.N[a] * extrapolated[_problem.dof(el.nodes[a], _velocity[d])];
      double omega[3] = {0, 0, 0};
      for (int j = 1; j <= k; ++j)
      {
        const auto & g = boundary.grad[j - 1];
        omega[0] += alpha[j] * (g[2][1] - g[1][2]);
        omega[1] += alpha[j] * (g[0][2] - g[2][0]);
        omega[2] += alpha[j] * (g[1][0] - g[0][1]);
      }
      const Point area = _problem.coordFactor(mp.x) * p.area;
      double normal_velocity = 0;
      for (int d = 0; d < dim; ++d)
      {
        normal_velocity += area[d] * ub[d];
        // The part c of the weighted divergence of DMCDM, through the boundary.
        if (dual)
          for (int j = 1; j <= k; ++j)
            normal_velocity += beta[j] / beta[0] * boundary.u[j - 1][d] * area[d];
      }
      const int tests = corner ? el.numCorners() : el.numNodes();
      for (int a = 0; a < tests; ++a)
      {
        const Point & gq = corner ? mp.corner_gradient[a] : mp.dN[a];
        // n . (omega x grad q)
        const double curl = area[0] * (omega[1] * gq[2] - omega[2] * gq[1]) +
                            area[1] * (omega[2] * gq[0] - omega[0] * gq[2]) +
                            area[2] * (omega[0] * gq[1] - omega[1] * gq[0]);
        // FEM tests the boundary velocity with the shape functions, DMCDM with the part of the boundary of each control domain.
        const double test = dual     ? (a == p.owner ? 1.0 : 0.0)
                            : corner ? mp.corner_shape[a]
                                     : mp.N[a];
        pressure_rhs[_problem.dof(el.nodes[a], _pressure)] -=
            c * test * normal_velocity + nu * curl;
      }
    }
  }

  // ---- the pressure -----------------------------------------------------------------------------
  const Rows & pressure_rows = *_pressure_rows;
  Vector P = _kinematic_pressure;
  pressure_rows.forEachFixed([&](Index d)
                             { P[d] = _pressure_pinned ? 0.0 : extrapolated[d] / rho; });
  _pressure_iterations =
      pressure_rows.solve(*_pressure_solver, _laplacian, pressure_rhs, P, _pressure_pinned);
  // The pressure of an enclosed flow with a mean of zero, as the monolithic solve reports it.
  if (_pressure_pinned)
  {
    double total = 0.0;
    for (Index i = 0; i < mesh.numNodes(); ++i)
      if (pressure_rows.owned[i])
        total += _pressure_weight[i] * P[_problem.dof(i, _pressure)];
    const double mean = _problem.parallel().sum(total) / _total_weight;
    for (Index i = 0; i < mesh.numNodes(); ++i)
      P[_problem.dof(i, _pressure)] -= mean;
  }
  _kinematic_pressure = P;

  // ---- the velocity -------------------------------------------------------------------------------
  // The explicit terms of the momentum equations: the backward difference of the steps before, the advection, the sources and the pressure.
  Vector history = Vector::Zero(solution.size());
  for (int j = 1; j <= k; ++j)
    history += beta[j] * levels[static_cast<std::size_t>(j - 1)]->second;
  Vector rhs = -(1.0 / dt) * (_mass * history) - advection_rhs - rho * (_gradient * P);
  if (with_source)
    rhs -= _source_residual;
  solution = extrapolated;
  _velocity_iterations = 0;
  for (std::size_t i = 0; i < _velocity_rows.size(); ++i)
    _velocity_iterations +=
        _velocity_rows[i]->solve(*_velocity_solvers[i], _helmholtz, rhs, solution, false);
  // The residual of the momentum equations, whose rows of prescribed velocities are the reactions (the force on the boundary is minus their sum), as the monolithic solve leaves them.
  Vector residual = _helmholtz * solution - rhs;
  _problem.foldDependents(residual, nullptr);
  _problem.parallel().addAtNodes(residual.data(), _problem.numVariables());
  for (Index i = 0; i < mesh.numNodes(); ++i)
  {
    const Index d = _problem.dof(i, _pressure);
    residual[d] = 0.0;
    solution[d] = rho * P[d];
  }
  _problem.setLastResidual(residual);
  _trial = {time, solution};

  SolveResult result;
  result.converged = true;
  result.total_iterations = 1;
  result.linear_iterations = _pressure_iterations + _velocity_iterations;
  return result;
}

double
projectInitialVelocity(Problem & problem,
                       const std::vector<int> & velocities,
                       int pressure,
                       Vector & solution)
{
  problem.initialize();
  const Mesh & mesh = problem.mesh();
  const int nv = problem.numVariables();
  const int dim = static_cast<int>(velocities.size());
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

  // ---- the divergence D and the Laplacian of the multiplier --------------------------------
  // D is the mass balance of the flow without its stabilization, tested with the functions of the
  // pressure, or of the first velocity for a flow without a pressure: (q, div u) with the
  // coordinate factor under FEM, and the flux of u out of the control domain of each node under
  // DMCDM.
  const int m = pressure >= 0 ? pressure : velocities[0];
  std::vector<std::string> names;
  for (int v : velocities)
    names.push_back(problem.variable(v).name);
  const std::string multiplier = problem.variable(m).name;
  std::vector<std::string> boundaries;
  for (const auto & [name, list] : mesh.sidesets())
    boundaries.push_back(name);
  const auto balance = privateObject<Kernel>(problem,
                                             "mass_conservation",
                                             "start_projection_divergence",
                                             {{"variable", multiplier}, {"velocities", names}});
  std::vector<std::shared_ptr<IntegratedBC>> outflow;
  if (!boundaries.empty())
    outflow.push_back(privateObject<IntegratedBC>(
        problem,
        "mass_flux",
        "start_projection_outflow",
        {{"variable", multiplier}, {"velocities", names}, {"boundary", boundaries}}));
  const auto diffusion = privateObject<Kernel>(
      problem, "diffusion", "start_projection_laplacian", {{"variable", multiplier}});
  std::vector<const IntegratedBC *> conditions;
  for (const auto & bc : outflow)
    conditions.push_back(bc.get());
  std::vector<char> is_multiplier(static_cast<std::size_t>(nv), 0);
  is_multiplier[static_cast<std::size_t>(m)] = 1;
  const SparseMatrix D = block(
      problem, operatorOf(problem, {balance.get()}, conditions, false), is_multiplier, is_velocity);
  const SparseMatrix laplacian = operatorOf(problem, {diffusion.get()});
  const double density = coefficientsAt(problem, *balance, mesh.node(0)).time_density;
  const SparseMatrix M = (1.0 / (density > 0.0 ? density : 1.0)) *
                         block(problem, operatorOf(problem, storage), is_velocity, is_velocity);

  // ---- the numbering: the velocity components and the multiplier at every node --------------
  const Parallel & parallel = problem.parallel();
  std::vector<int> fields = velocities;
  fields.push_back(m);
  const ProjectionSolver::Rows numbering(problem, fields, {});
  const int width = dim + 1;
  std::vector<int> field(static_cast<std::size_t>(nv), -1);
  for (int d = 0; d < dim; ++d)
    field[static_cast<std::size_t>(velocities[d])] = d;
  const Index nodes = mesh.numNodes();
  const std::vector<Index> & global = numbering.node;
  const std::vector<char> & owned = numbering.owned;
  const Index first = numbering.first, num_owned = numbering.num_owned;

  // ---- the rows of the multiplier ----------------------------------------------------------
  // The multiplier is zero where the pressure is prescribed, whose mass balance is not an equation of the flow, at a dependent unknown, and at a node that tests no divergence (a node outside the flow).
  const std::vector<char> constrained = problem.constrainedDofs();
  Vector tested = Vector::Zero(problem.numDofs());
  for (Index col = 0; col < D.outerSize(); ++col)
    for (SparseMatrix::InnerIterator it(D, col); it; ++it)
      tested[it.row()] += std::abs(it.value());
  parallel.addAtNodes(tested.data(), nv);
  std::vector<char> fixed(static_cast<std::size_t>(nodes), 0);
  for (Index i = 0; i < nodes; ++i)
  {
    const Index d = problem.dof(i, m);
    const bool given = pressure >= 0 ? constrained[d] != 0 : problem.isDependent(d);
    fixed[i] = given || !(tested[d] > 0.0) ? 1 : 0;
  }
  std::vector<std::vector<char>> velocity_fixed(static_cast<std::size_t>(dim));
  for (int d = 0; d < dim; ++d)
  {
    velocity_fixed[d].resize(static_cast<std::size_t>(nodes));
    for (Index i = 0; i < nodes; ++i)
      velocity_fixed[d][i] = constrained[problem.dof(i, velocities[d])];
  }
  // A flow whose normal velocity is prescribed on the whole boundary (an enclosed flow, or one
  // with a prescribed inflow and outflow) leaves the multiplier free up to a constant: D^T 1 is
  // zero at every free velocity.  The constant is fixed at the node of the smallest global
  // number, and the divergence that the data give the whole flow is spread over its volume.
  bool enclosed = true;
  if (pressure >= 0)
  {
    bool given = false;
    for (Index i = 0; i < nodes; ++i)
    {
      const Index d = problem.dof(i, pressure);
      given = given || (constrained[d] && problem.activeDofs()[d] && !problem.isDependent(d));
    }
    enclosed = !parallel.any(given);
  }
  if (enclosed)
  {
    Vector column = Vector::Zero(problem.numDofs()), size = Vector::Zero(problem.numDofs());
    for (Index col = 0; col < D.outerSize(); ++col)
      for (SparseMatrix::InnerIterator it(D, col); it; ++it)
      {
        column[col] += it.value();
        size[col] += std::abs(it.value());
      }
    parallel.addAtNodes(column.data(), nv);
    parallel.addAtNodes(size.data(), nv);
    double open = 0.0, scale = 0.0;
    for (Index i = 0; i < nodes; ++i)
      for (int d = 0; d < dim; ++d)
      {
        const Index dof = problem.dof(i, velocities[d]);
        scale = std::max(scale, size[dof]);
        if (!velocity_fixed[d][i])
          open = std::max(open, std::abs(column[dof]));
      }
    enclosed = parallel.maximum(open) <= 1e-10 * parallel.maximum(scale);
  }
  // The rows of the multiplier: its equation is D u = - D u_0.
  const auto multiplierOf = [&](const Vector & u)
  {
    Vector r = D * u;
    parallel.addAtNodes(r.data(), nv);
    return r;
  };
  Vector rhs = -multiplierOf(solution);
  // The measure of the test function of every node, over which the divergence is spread.
  Vector volume = problem.nodeWeights(m);
  parallel.addAtNodes(volume.data(), 1);
  if (enclosed)
  {
    double total = 0.0, measure = 0.0;
    for (Index i = 0; i < nodes; ++i)
      if (owned[i] && !fixed[i])
      {
        total += rhs[problem.dof(i, m)];
        measure += volume[i];
      }
    const double mean = parallel.sum(total) / parallel.sum(measure);
    double smallest = std::numeric_limits<double>::infinity();
    for (Index i = 0; i < nodes; ++i)
      if (!fixed[i])
      {
        rhs[problem.dof(i, m)] -= mean * volume[i];
        if (owned[i])
          smallest = std::min(smallest, static_cast<double>(global[i]));
      }
    smallest = parallel.minimum(smallest);
    for (Index i = 0; i < nodes; ++i)
      if (static_cast<double>(global[i]) == smallest)
        fixed[i] = 1;
  }

  // ---- the saddle point system -------------------------------------------------------------
  //     [ M      D^T    ] [ du     ]   [ 0       ]
  //     [ D   -eps K_l  ] [ lambda ] = [ - D u_0 ]
  // with M the mass matrix of the velocity and K_l the Laplacian of the multiplier, which has the spectrum of D M^-1 D^T on the smooth fields.
  // The small regularization eps K_l fixes the multiplier where D^T has a null space (the spurious pressure modes of equal-order elements), whose divergence no velocity changes.
  const double eps = 1e-4;
  const auto row = [&](Index i, int f) { return width * global[i] + f; };
  const auto free = [&](Index i, int f)
  {
    if (global[i] < 0)
      return false;
    return f < dim ? !velocity_fixed[static_cast<std::size_t>(f)][i] : !fixed[i];
  };
  std::vector<petsc::GlobalEntry> entries;
  const auto add = [&](Index i, int fi, Index j, int fj, double value)
  {
    if (free(i, fi) && free(j, fj))
      entries.push_back({row(i, fi), row(j, fj), value});
  };
  const auto fieldOf = [&](Index d) { return field[static_cast<std::size_t>(d % nv)]; };
  for (Index col = 0; col < M.outerSize(); ++col)
    for (SparseMatrix::InnerIterator it(M, col); it; ++it)
      add(it.row() / nv, fieldOf(it.row()), col / nv, fieldOf(col), it.value());
  for (Index col = 0; col < D.outerSize(); ++col)
    for (SparseMatrix::InnerIterator it(D, col); it; ++it)
    {
      add(it.row() / nv, dim, col / nv, fieldOf(col), it.value());
      add(col / nv, fieldOf(col), it.row() / nv, dim, it.value());
    }
  for (Index col = 0; col < laplacian.outerSize(); ++col)
    for (SparseMatrix::InnerIterator it(laplacian, col); it; ++it)
      if (it.row() % nv == m && col % nv == m)
        add(it.row() / nv, dim, col / nv, dim, -eps * it.value());
  for (Index i = 0; i < nodes; ++i)
    if (owned[i] && global[i] >= 0)
      for (int f = 0; f <= dim; ++f)
        if (!free(i, f))
          entries.push_back({row(i, f), row(i, f), 1.0});
  petsc::Settings settings;
  settings.relative_tolerance = 1e-12;
  settings.symmetric = false;
  const petsc::LinearSolver system(
      numbering.across, numbering.num_global, first, num_owned, entries, settings);
  // The regularization moves the smooth part of the solution by about eps relative to its size, which corrections with the residual of the system without it take back (iterated regularization), until a correction changes the velocity by less than 1e-13 of the speed.
  // A spurious mode that D^T nearly annihilates keeps a part of its divergence, which a velocity of bounded size cannot remove.
  const auto at = [&](Index i, int f) { return row(i, f) - first; };
  // The velocity correction, in the unknowns of the problem, and the multiplier, one per node.
  Vector du = Vector::Zero(problem.numDofs());
  Vector lambda = Vector::Zero(problem.numDofs());
  Vector b = Vector::Zero(num_owned);
  for (Index i = 0; i < nodes; ++i)
    if (owned[i] && free(i, dim))
      b[at(i, dim)] = rhs[problem.dof(i, m)];
  double speed0 = 0.0;
  for (Index i = 0; i < nodes; ++i)
    for (int v : velocities)
      speed0 = std::max(speed0, std::abs(solution[problem.dof(i, v)]));
  speed0 = parallel.maximum(speed0);
  for (int pass = 0; pass < 20; ++pass)
  {
    petsc::Result result;
    const Vector x = system.solve(b, nullptr, result);
    if (parallel.any(!result.converged))
      throw std::runtime_error("dualmesh: the projection of the initial velocity onto a field "
                               "without divergence did not converge (" +
                               result.reason + ", " + std::to_string(result.iterations) +
                               " iterations).");
    // The update of the velocity at its unknowns, and of the multiplier at those of m.
    Vector velocity_update = Vector::Zero(problem.numDofs());
    Vector multiplier_update = Vector::Zero(problem.numDofs());
    for (Index i = 0; i < nodes; ++i)
      if (owned[i])
        for (int f = 0; f <= dim; ++f)
          if (free(i, f))
            (f < dim ? velocity_update[problem.dof(i, velocities[f])]
                     : multiplier_update[problem.dof(i, m)]) = x[at(i, f)];
    parallel.copyFromOwners(velocity_update.data(), nv);
    parallel.copyFromOwners(multiplier_update.data(), nv);
    problem.updateDependentDofs(velocity_update);
    du += velocity_update;
    lambda += multiplier_update;
    double moved = 0.0, reached = speed0;
    for (Index i = 0; i < nodes; ++i)
      for (int v : velocities)
      {
        moved = std::max(moved, std::abs(velocity_update[problem.dof(i, v)]));
        reached = std::max(reached, std::abs(du[problem.dof(i, v)]));
      }
    if (parallel.maximum(moved) <= 1e-13 * parallel.maximum(reached))
      break;
    // The residual of the system without the regularization: the momentum rows M du + D^T lambda
    // and the divergence left, - D u_0 - D du.
    Vector momentum = M * du + D.transpose() * lambda;
    parallel.addAtNodes(momentum.data(), nv);
    const Vector divergence_left = rhs - multiplierOf(du);
    for (Index i = 0; i < nodes; ++i)
      if (owned[i])
        for (int f = 0; f <= dim; ++f)
          if (free(i, f))
            b[at(i, f)] = f < dim ? -momentum[problem.dof(i, velocities[f])]
                                  : divergence_left[problem.dof(i, m)];
  }

  // ---- the velocity ------------------------------------------------------------------------
  double change = 0.0, speed = 0.0;
  for (Index i = 0; i < nodes; ++i)
  {
    double change2 = 0.0, speed2 = 0.0;
    for (int d = 0; d < dim; ++d)
    {
      const Index dof = problem.dof(i, velocities[d]);
      double & u = solution[dof];
      // A dependent unknown follows those it depends on.
      if (!velocity_fixed[d][i] || problem.isDependent(dof))
      {
        u += du[dof];
        change2 += du[dof] * du[dof];
      }
      speed2 += u * u;
    }
    change = std::max(change, std::sqrt(change2));
    speed = std::max(speed, std::sqrt(speed2));
  }
  change = parallel.maximum(change);
  speed = parallel.maximum(speed);
  return speed > 0.0 ? change / speed : 0.0;
}

} // namespace dualmesh
