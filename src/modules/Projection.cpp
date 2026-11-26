// SPDX-License-Identifier: LGPL-2.1-or-later
#include "dualmesh/modules/Projection.h"
#include "dualmesh/core/InputParameters.h"
#include "dualmesh/fe/Assembly.h"
#include "dualmesh/modules/Framework.h"
#include "dualmesh/parallel/DistributedProblem.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>

namespace dualmesh
{

/// Where the values of one scalar field (one value per local entity) lie in the global systems, and the exchanges between the processes that hold copies of an entity.
/// A dependent entity (Problem::dependentDofs: a hanging node under FEM, and on one process a periodic copy) is the weighted sum of independent ones: it has no row of its own, its equation goes to theirs with the weights, and its value follows from theirs.
struct FlowOperators::Layout
{
  DistributedProblem * distributed = nullptr;
  Index n = 0;
  /// The global number of every local entity, and -1 for a dependent one.
  std::vector<Index> global;
  std::vector<char> owned;
  /// The independent entities and the weights of every dependent entity, and an empty list for every other one. Empty when no entity depends.
  std::vector<std::vector<std::pair<Index, double>>> dependent;
  Index first = 0, num_owned = 0, num_global = 0;

  bool depends(Index i) const
  {
    return !dependent.empty() && !dependent[static_cast<std::size_t>(i)].empty();
  }
  /// The entities that variable @p v of @p problem depends on, refused unless they are those of every other field of the flow.
  void requireAlike(const Problem & problem, int v) const
  {
    const auto & map = problem.dependentDofs();
    const int nv = problem.numVariables();
    for (Index i = 0; i < n && !map.empty(); ++i)
    {
      const auto & on = map[static_cast<std::size_t>(problem.dof(i, v))];
      bool alike = on.size() == (depends(i) ? dependent[static_cast<std::size_t>(i)].size() : 0);
      for (std::size_t k = 0; alike && k < on.size(); ++k)
        alike = on[k].first / nv == dependent[static_cast<std::size_t>(i)][k].first &&
                on[k].second == dependent[static_cast<std::size_t>(i)][k].second;
      if (!alike)
        throw InputError("A periodic boundary of a flow joins every component of its velocity "
                         "and its pressure alike. Leave 'variables' out.");
    }
  }
  /// Call @p add with every independent entity of entity @p i and its weight: the entity itself with one, or those it depends on.
  template <typename Add> void forEachIndependent(Index i, Add && add) const
  {
    if (!depends(i))
    {
      add(i, 1.0);
      return;
    }
    for (const auto & [p, w] : dependent[static_cast<std::size_t>(i)])
      add(p, w);
  }
  /// Move the value of every dependent entity into its independent ones with the weights (R_j += w_j R_d), which leaves zero at the dependent entity.
  void fold(Vector & v) const
  {
    for (Index i = 0; i < n && !dependent.empty(); ++i)
      if (depends(i))
      {
        for (const auto & [p, w] : dependent[static_cast<std::size_t>(i)])
          v[p] += w * v[i];
        v[i] = 0.0;
      }
  }
  /// The matrix C^T A C of the matrix A given by @p entries, with C the map from the independent entities to all of them.
  void fold(std::vector<Eigen::Triplet<double>> & entries) const
  {
    if (dependent.empty())
      return;
    std::vector<Eigen::Triplet<double>> folded;
    folded.reserve(entries.size());
    for (const auto & t : entries)
      forEachIndependent(t.row(),
                         [&](Index r, double wr)
                         {
                           forEachIndependent(t.col(),
                                              [&](Index c, double wc)
                                              { folded.emplace_back(r, c, wr * wc * t.value()); });
                         });
    entries = std::move(folded);
  }
  /// Fold the dependent entities into their independent ones, then add the values of every copy of an entity on the other processes, so that every copy holds the sum.
  void sum(Vector & v) const
  {
    fold(v);
    if (distributed)
      distributed->addAtNodes(v.data(), 1);
  }
  /// Give every copy of an entity the value of the copy that owns it, and every dependent entity the weighted sum of its independent ones.
  void copy(Vector & v) const
  {
    if (distributed)
      distributed->copyFromOwners(v.data(), 1);
    // The distribution gives every process that holds a hanging node the nodes it hangs on.
    for (Index i = 0; i < n && !dependent.empty(); ++i)
      if (depends(i))
      {
        double value = 0.0;
        for (const auto & [p, w] : dependent[static_cast<std::size_t>(i)])
          value += w * v[p];
        v[i] = value;
      }
  }
  /// The sum of the values over the owned entities of every process.
  double sumOwned(const Vector & v) const
  {
    double local = 0;
    for (Index i = 0; i < n; ++i)
      if (owned[i])
        local += v[i];
    return distributed ? distributed->communicator().sum(local) : local;
  }
  bool any(bool local) const
  {
    return distributed ? distributed->communicator().any(local) : local;
  }
};

/// The geometry of the quadrature points of every integrated element, computed once: the weight (with the Jacobian), the position and the inverse transposed Jacobian at each point, and the shape functions and their reference derivatives once per element type.
/// There are two sets of points: those of the element interiors (the finite element method) or of the control volumes, which carry the node that owns them, and, for the methods of the dual mesh, those of the interfaces between control volumes, which carry their two nodes and their area vector.
/// The finite element method takes a Gauss rule of order + 1 points per direction, which integrates the mass matrix and the advection exactly on affine elements.
/// The methods of the dual mesh take order points per direction of every patch of a control volume and of every interface, the midpoint rule for linear elements, which is second order like the methods themselves: the errors of the Taylor-Green vortex are those of two points per direction, and a step costs 2.4 times less.
/// Under DMCDM a cell with nodes hanging on its sides is a polygon (PolygonCell), whose points are those of the monolithic assembly (buildElementPoints): its control domains and interfaces belong to its corners and to its hanging nodes, and the fields inside are those of its corners.
struct FlowOperators::Geometry
{
  struct Table
  {
    int nodes = 0;
    std::vector<double> N; // [point][node]
    std::vector<Point> dN; // [point][node], reference derivatives
  };
  struct Points
  {
    std::map<ElementType, Table> tables;
    /// The shape functions of the points of every polygon cell, whose points are its own, and the table of each element (-1 for an element that takes the table of its type).
    std::vector<Table> polygon_tables;
    std::vector<int> polygon_table;
    std::vector<Index> first; // the first point of each element
    std::vector<double> weight;
    std::vector<Point> area, x;
    std::vector<std::array<double, 9>> G; // J^{-T}, row major
    /// The nodes of the mesh that own a point and that lie across its interface, or -1 for the finite element method.
    std::vector<Index> owner, neighbor;

    int count(Index e) const { return static_cast<int>(first[e + 1] - first[e]); }

    /// Fill @p mp with the shape functions and physical gradients at point @p q of element @p e.
    void at(const Element & el, Index e, int q, MappedPoint & mp) const
    {
      const int polygon = polygon_table[static_cast<std::size_t>(e)];
      const Table & t =
          polygon >= 0 ? polygon_tables[static_cast<std::size_t>(polygon)] : tables.at(el.type);
      const Index point = first[e] + q;
      const auto & g = G[point];
      for (int a = 0; a < t.nodes; ++a)
      {
        mp.N[a] = t.N[q * t.nodes + a];
        const Point & r = t.dN[q * t.nodes + a];
        mp.dN[a] = {g[0] * r[0] + g[1] * r[1] + g[2] * r[2],
                    g[3] * r[0] + g[4] * r[1] + g[5] * r[2],
                    g[6] * r[0] + g[7] * r[1] + g[8] * r[2]};
      }
      mp.x = x[point];
    }

    /// The points of every element of the mesh of @p problem up to @p elements, of @p set.
    void build(const Problem & problem, Index elements, bool dual, PointSet set)
    {
      const Mesh & mesh = problem.mesh();
      std::vector<IntegrationPoint> points;
      MappedPoint mp;
      first.assign(static_cast<std::size_t>(elements) + 1, 0);
      polygon_table.assign(static_cast<std::size_t>(elements), -1);
      const auto fill = [&](Table & table, const Element & el, const ReferenceElement & ref)
      {
        table.nodes = el.numNodes();
        for (const auto & p : points)
        {
          double N[kMaxElementNodes];
          Point dN[kMaxElementNodes];
          ref.shape(p.xi, N, dN);
          table.N.insert(table.N.end(), N, N + el.numNodes());
          table.dN.insert(table.dN.end(), dN, dN + el.numNodes());
        }
      };
      for (Index e = 0; e < elements; ++e)
      {
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
          weight.push_back(p.weight);
          area.push_back(p.area);
          x.push_back(mp.x);
          owner.push_back(nodeOf(p.owner));
          neighbor.push_back(nodeOf(p.neighbor));
        }
      }
    }
  };
  Points volume, faces;
};

ProjectionSolver::ProjectionSolver(Problem & problem,
                                   DistributedProblem * distributed,
                                   ProjectionSettings settings)
    : _problem(problem), _distributed(distributed), _settings(std::move(settings))
{
  if (_settings.order < 1 || _settings.order > 3)
    throw InputError("Projection: the order of the time integration is 1, 2 or 3.");
  if (!(_settings.density > 0))
    throw InputError("Projection: the density must be positive; a flow without inertia (Stokes "
                     "flow) has no time derivative to integrate.");
  if (_settings.dynamic_viscosity < 0)
    throw InputError("Projection: the viscosity cannot be negative.");
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

FlowOperators::FlowOperators(Problem & problem_,
                             DistributedProblem * distributed_,
                             const std::vector<int> & velocities_)
    : problem(problem_), distributed(distributed_), velocities(velocities_)
{
  const Mesh & mesh = problem.mesh();
  dim = mesh.dimension();
  dual = problem.isDual();
  const int nv = problem.numVariables();

  // ---- the layout of a scalar field --------------------------------------------------------
  layout = std::make_unique<Layout>();
  Layout & L = *layout;
  L.distributed = distributed;
  L.n = mesh.numNodes();
  // The dependent entities of the first velocity are those of every component.
  const auto & dependent = problem.dependentDofs();
  if (!dependent.empty())
  {
    L.dependent.assign(static_cast<std::size_t>(L.n), {});
    for (Index i = 0; i < L.n; ++i)
      for (const auto & [p, w] : dependent[static_cast<std::size_t>(problem.dof(i, velocities[0]))])
        L.dependent[static_cast<std::size_t>(i)].push_back({p / nv, w});
    for (int v : velocities)
      L.requireAlike(problem, v);
  }
  L.owned.assign(static_cast<std::size_t>(L.n), 0);
  for (Index i = 0; i < L.n; ++i)
    L.owned[i] = (!distributed || distributed->ownsNode(i)) && !L.depends(i) ? 1 : 0;
  L.num_owned = static_cast<Index>(std::count(L.owned.begin(), L.owned.end(), 1));
  // The owned entities of each process are numbered contiguously, those of the processes before it first, and a copy takes the number of its owner. A dependent entity has none.
  if (distributed)
    L.global = distributed->globalNumbers(L.owned, 1, L.first, L.num_global);
  else
  {
    L.global.assign(static_cast<std::size_t>(L.n), -1);
    for (Index i = 0; i < L.n; ++i)
      if (L.owned[i])
        L.global[i] = L.num_global++;
  }

  // ---- the elements of the flow and the geometry of their quadrature points ----------------
  const Index elements = problem.numIntegratedElements();
  const auto & blocks = problem.variable(velocities[0]).blocks;
  in_flow.assign(static_cast<std::size_t>(elements), 1);
  for (Index e = 0; e < elements; ++e)
    in_flow[e] = blocks.empty() || blocks.count(mesh.element(e).block) ? 1 : 0;
  geometry = std::make_unique<Geometry>();
  Geometry & geo = *geometry;
  geo.volume.build(problem, elements, dual, PointSet::Volume);
  if (dual)
    geo.faces.build(problem, elements, true, PointSet::Faces);

  // ---- the mass, stiffness and gradient matrices --------------------------------------------
  // The finite element method tests with the shape functions.
  // The methods of the dual mesh test with the control volumes: the mass and the gradient are integrals over the control volume of each node, and the Laplacian is - the flux of the gradient out of it, through the interfaces between control volumes.
  // A node that hangs on the side of a polygon cell owns control volume and interfaces in it, and the fields there come from the corners of the cell.
  // Every weight carries the coordinate factor (2 pi r in axisymmetric coordinates).
  const auto factor = [&](const Point & x) { return problem.coordFactor(x); };
  std::vector<Eigen::Triplet<double>> gradient_entries[3];
  MappedPoint mp;
  for (Index e = 0; e < elements; ++e)
  {
    if (!in_flow[e])
      continue;
    const Element & el = mesh.element(e);
    const int nn = el.numNodes();
    for (int q = 0; q < geo.volume.count(e); ++q)
    {
      geo.volume.at(el, e, q, mp);
      const Index point = geo.volume.first[e] + q;
      const double w = geo.volume.weight[point] * factor(mp.x);
      if (dual)
      {
        const Index o = geo.volume.owner[point];
        for (int b = 0; b < nn; ++b)
        {
          mass_entries.emplace_back(o, el.nodes[b], w * mp.N[b]);
          for (int d = 0; d < dim; ++d)
            gradient_entries[d].emplace_back(o, el.nodes[b], w * mp.dN[b][d]);
        }
        continue;
      }
      for (int a = 0; a < nn; ++a)
        for (int b = 0; b < nn; ++b)
        {
          mass_entries.emplace_back(el.nodes[a], el.nodes[b], w * mp.N[a] * mp.N[b]);
          stiffness_entries.emplace_back(el.nodes[a], el.nodes[b], w * dot(mp.dN[a], mp.dN[b]));
          for (int d = 0; d < dim; ++d)
            gradient_entries[d].emplace_back(el.nodes[a], el.nodes[b], w * mp.N[a] * mp.dN[b][d]);
        }
    }
    if (!dual)
      continue;
    for (int q = 0; q < geo.faces.count(e); ++q)
    {
      geo.faces.at(el, e, q, mp);
      const Index point = geo.faces.first[e] + q;
      const Index o = geo.faces.owner[point], n = geo.faces.neighbor[point];
      const Point & a = geo.faces.area[point];
      const double f = factor(mp.x);
      for (int b = 0; b < nn; ++b)
      {
        const double flux = f * dot(mp.dN[b], a);
        stiffness_entries.emplace_back(o, el.nodes[b], -flux);
        stiffness_entries.emplace_back(n, el.nodes[b], flux);
      }
    }
  }
  // A dependent node takes the equations of the nodes it depends on: every matrix becomes C^T A C.
  L.fold(mass_entries);
  L.fold(stiffness_entries);
  for (int d = 0; d < dim; ++d)
    L.fold(gradient_entries[d]);
  mass.resize(L.n, L.n);
  mass.setFromTriplets(mass_entries.begin(), mass_entries.end());
  stiffness.resize(L.n, L.n);
  stiffness.setFromTriplets(stiffness_entries.begin(), stiffness_entries.end());
  gradient.assign(static_cast<std::size_t>(dim), SparseMatrix(L.n, L.n));
  for (int d = 0; d < dim; ++d)
    gradient[d].setFromTriplets(gradient_entries[d].begin(), gradient_entries[d].end());
  element_size.resize(static_cast<std::size_t>(elements));
  for (Index e = 0; e < elements; ++e)
    element_size[e] = elementSize(mesh, e);

  // ---- the prescribed velocities and the boundary ------------------------------------------
  velocity_fixed.clear();
  for (int v : velocities)
    velocity_fixed.push_back(prescribed(v));
  std::set<std::pair<Index, int>> sides;
  for (const auto & [name, list] : mesh.sidesets())
    for (const Side & side : list)
      if (side.first < elements && in_flow[side.first])
        sides.insert({side.first, side.second});
  boundary_sides.clear();
  for (const auto & [element, side] : sides)
    boundary_sides.push_back({element, side});
}

FlowOperators::~FlowOperators() = default;

std::vector<char>
FlowOperators::prescribed(int variable, bool inactive) const
{
  const Layout & L = *layout;
  const std::vector<char> constrained = problem.constrainedDofs();
  const auto & active = problem.activeDofs();
  Vector mark(L.n);
  for (Index i = 0; i < L.n; ++i)
  {
    const Index d = problem.dof(i, variable);
    const bool given = constrained[d] && (inactive || active.empty() || active[d]);
    mark[i] = given && !L.depends(i) ? 1.0 : 0.0;
  }
  L.sum(mark);
  std::vector<char> fixed(static_cast<std::size_t>(L.n));
  for (Index i = 0; i < L.n; ++i)
    fixed[i] = mark[i] > 0.5 ? 1 : 0;
  return fixed;
}

double
FlowOperators::sum(double value) const
{
  return distributed ? distributed->communicator().sum(value) : value;
}

double
FlowOperators::maximum(double value) const
{
  return distributed ? distributed->communicator().max(value) : value;
}

void
ProjectionSolver::setUp()
{
  const Mesh & mesh = _problem.mesh();
  const int dim = mesh.dimension();
  if (mesh.spaceDimension() > dim)
    throw InputError("The projection solver of incompressible flow is not available on a surface "
                     "or a curve in space.");
  if (_problem.coordinateSystem() != CoordinateSystem::Cartesian)
    throw InputError("Projection: the projection time integration runs in Cartesian coordinates "
                     "only.");
  if (_problem.hasMixedOrder())
    throw InputError("Projection: the velocity and the pressure must have the order of the mesh "
                     "(the projection time integration does not take the Taylor-Hood "
                     "element).");
  if (static_cast<int>(_settings.velocities.size()) != dim)
    throw InputError("Projection: one velocity variable per coordinate is needed.");
  _velocity.clear();
  for (const auto & name : _settings.velocities)
    _velocity.push_back(_problem.variableIndex(name));
  _pressure = _problem.variableIndex(_settings.pressure);
  if (_problem.numVariables() != dim + 1)
    throw InputError("Projection: the projection time integration advances the velocity and the "
                     "pressure of a flow alone, and this problem has other variables as well.");
  // The equations are assembled here, from the density, the viscosity and the body force; another term on the flow's variables would be ignored, so it is refused.
  if (!_settings.own_objects.empty())
  {
    const std::set<std::string> own(_settings.own_objects.begin(), _settings.own_objects.end());
    for (const auto & name : _problem.objectNames())
    {
      if (own.count(name))
        continue;
      const auto object = _problem.object(name);
      const auto * term = dynamic_cast<const ResidualObject *>(object.get());
      const bool assembled = dynamic_cast<const Kernel *>(object.get()) ||
                             dynamic_cast<const IntegratedBC *>(object.get());
      if (term && assembled &&
          (term->variable() == _pressure ||
           std::find(_velocity.begin(), _velocity.end(), term->variable()) != _velocity.end()))
        throw InputError("Projection: '" + name +
                         "' acts on the flow, which the projection time integration assembles "
                         "from its density, viscosity and body force alone. Give a body force "
                         "with the body_force of fluid_dynamics, prescribe the velocity or the "
                         "pressure on the boundary, or use time_integration='monolithic'.");
    }
  }

  _operators = std::make_unique<FlowOperators>(_problem, _distributed, _velocity);
  const FlowOperators & ops = *_operators;
  const FlowOperators::Layout & L = *ops.layout;
  // The pressure shares the layout of the velocity, so a periodic boundary joins it as it joins the velocity.
  L.requireAlike(_problem, _pressure);

  // ---- the pressure ------------------------------------------------------------------------------
  // A copy of an entity is prescribed when any copy is.
  // A dependent node has no row: its value follows from the nodes it depends on, which are prescribed wherever it is.
  _pressure_fixed = ops.prescribed(_pressure);
  // An enclosed flow determines the pressure up to a constant: the constant is fixed at the entity numbered zero, after the right-hand side is made compatible.
  _pressure_pinned =
      !L.any(std::find(_pressure_fixed.begin(), _pressure_fixed.end(), 1) != _pressure_fixed.end());
  if (_pressure_pinned)
  {
    for (Index i = 0; i < L.n; ++i)
      _pressure_fixed[i] = L.global[i] == 0 ? 1 : 0;
    // The weights of the mean of the pressure, those of the monolithic solve, with those of a dependent node in the nodes it depends on.
    _volume = _problem.nodeWeights(_pressure);
    L.sum(_volume);
    _total_volume = L.sumOwned(_volume);
  }
  // The Laplacian of the finite element method is symmetric, solved by the conjugate gradient method; that of the dual mesh is not in general, solved by GMRES.
  _pressure_options = _settings.pressure_solver.empty()
                          ? std::string(ops.dual ? "-ksp_type gmres" : "-ksp_type cg") +
                                " -pc_type hypre -pc_hypre_type boomeramg"
                          : _settings.pressure_solver;
  _velocity_options =
      _settings.velocity_solver.empty()
          ? std::string(ops.dual ? "-ksp_type gmres" : "-ksp_type cg") + " -pc_type jacobi"
          : _settings.velocity_solver;
  _pressure_solver = ops.makeSolver(ops.stiffness_entries,
                                    _pressure_fixed,
                                    _pressure_options,
                                    _settings.pressure_tolerance,
                                    _settings.pressure_projection_vectors);
  _helmholtz_c = -1.0;
  _ready = true;
}

std::vector<petsc::GlobalEntry>
FlowOperators::globalEntries(const std::vector<Eigen::Triplet<double>> & local,
                             const std::vector<char> & fixed) const
{
  const Layout & L = *layout;
  // The rows and the columns of the prescribed entities are those of the identity, so that the matrix stays symmetric; the prescribed values enter the right-hand side (solve).
  std::vector<petsc::GlobalEntry> entries;
  entries.reserve(local.size() + static_cast<std::size_t>(L.num_owned));
  for (const auto & t : local)
  {
    const Index i = t.row(), j = t.col();
    if (fixed[i] || fixed[j] || L.global[i] < 0 || L.global[j] < 0)
      continue;
    entries.push_back({L.global[i], L.global[j], t.value()});
  }
  for (Index i = 0; i < L.n; ++i)
    if (fixed[i] && L.owned[i])
      entries.push_back({L.global[i], L.global[i], 1.0});
  return entries;
}

std::unique_ptr<petsc::LinearSolver>
FlowOperators::makeSolver(const std::vector<Eigen::Triplet<double>> & local,
                          const std::vector<char> & fixed,
                          const std::string & options,
                          double tolerance,
                          int projection_vectors) const
{
  const Layout & L = *layout;
  const std::vector<petsc::GlobalEntry> entries = globalEntries(local, fixed);
  petsc::Settings settings;
  settings.options = options;
  settings.relative_tolerance = tolerance;
  settings.projection_vectors = projection_vectors;
  settings.symmetric = !dual;
  const bool across = distributed != nullptr && distributed->communicator().size() > 1;
  return std::make_unique<petsc::LinearSolver>(
      across, L.num_global, L.first, L.num_owned, entries, settings);
}

int
FlowOperators::solve(const petsc::LinearSolver & solver,
                     const SparseMatrix & matrix,
                     const std::vector<char> & fixed,
                     Vector rhs,
                     Vector & values,
                     bool remove_mean) const
{
  const Layout & L = *layout;
  // The prescribed values move to the right-hand side of the other rows.
  Vector prescribed = Vector::Zero(L.n);
  for (Index i = 0; i < L.n; ++i)
    if (fixed[i])
      prescribed[i] = values[i];
  rhs -= matrix * prescribed;
  L.sum(rhs);
  // A Neumann problem is solvable when its right-hand side sums to zero, which the discretization satisfies only up to its error.
  if (remove_mean)
    rhs.array() -= L.sumOwned(rhs) / static_cast<double>(L.num_global);
  Vector b_owned(L.num_owned), guess(L.num_owned);
  for (Index i = 0; i < L.n; ++i)
    if (L.owned[i])
    {
      b_owned[L.global[i] - L.first] = fixed[i] ? values[i] : rhs[i];
      guess[L.global[i] - L.first] = values[i];
    }
  petsc::Result result;
  const Vector x = solver.solve(b_owned, &guess, result);
  if (L.any(!result.converged))
    throw std::runtime_error("dualmesh: a linear solve of the projection time integration did not "
                             "converge (" +
                             result.reason + ", " + std::to_string(result.iterations) +
                             " iterations).");
  values.setZero(L.n);
  for (Index i = 0; i < L.n; ++i)
    if (L.owned[i])
      values[i] = x[L.global[i] - L.first];
  L.copy(values);
  return result.iterations;
}

void
ProjectionSolver::start(const Vector & solution, double time)
{
  if (!_ready)
    setUp();
  _history.clear();
  _history.push_front({time, solution});
  _trial = {};
  _kinematic_pressure.resize(_operators->layout->n);
  for (Index i = 0; i < _operators->layout->n; ++i)
    _kinematic_pressure[i] = solution[_problem.dof(i, _pressure)] / _settings.density;
}

double
ProjectionSolver::courantNumber(const Vector & solution, double dt) const
{
  const FlowOperators & ops = *_operators;
  const Mesh & mesh = _problem.mesh();
  double largest = 0;
  for (Index e = 0; e < _problem.numIntegratedElements(); ++e)
  {
    if (!ops.in_flow[e])
      continue;
    const Element & el = mesh.element(e);
    double speed2 = 0;
    for (int a = 0; a < el.numNodes(); ++a)
    {
      double s2 = 0;
      for (int d = 0; d < ops.dim; ++d)
      {
        const double v = solution[_problem.dof(el.nodes[a], _velocity[d])];
        s2 += v * v;
      }
      speed2 = std::max(speed2, s2);
    }
    largest = std::max(largest, std::sqrt(speed2) / ops.element_size[e]);
  }
  if (_distributed)
    largest = _distributed->communicator().max(largest);
  return dt * largest;
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
ProjectionSolver::step(Vector & solution, const Vector & old, double time, double dt, bool continues)
{
  const FlowOperators & ops = *_operators;
  const FlowOperators::Layout & L = *ops.layout;
  const Mesh & mesh = _problem.mesh();
  const double rho = _settings.density;
  const double nu = _settings.dynamic_viscosity / rho;
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

  // The Helmholtz matrices change with beta_0 / dt: in the first steps, at a step shortened to land on an output time, and at every step of the cfl stepper.
  // Their structure does not, so the solvers are made once and later given the new values.
  const double c = beta[0] / dt;
  // A step of the same size gives the same coefficient up to the rounding of the times, which must not touch the solvers.
  if (std::abs(c - _helmholtz_c) > 1e-9 * std::abs(c))
  {
    if (_velocity_solvers.empty())
    {
      // The first time: the solvers, and the values of K and M in the order of their entries.
      std::vector<Eigen::Triplet<double>> entries;
      entries.reserve(ops.mass_entries.size() + ops.stiffness_entries.size());
      for (const auto & t : ops.stiffness_entries)
        entries.emplace_back(t.row(), t.col(), nu * t.value());
      for (const auto & t : ops.mass_entries)
        entries.emplace_back(t.row(), t.col(), c * t.value());
      _stiffness_on_pattern = ops.stiffness + 0.0 * ops.mass;
      _mass_on_pattern = ops.mass + 0.0 * ops.stiffness;
      _stiffness_on_pattern.makeCompressed();
      _mass_on_pattern.makeCompressed();
      if (_stiffness_on_pattern.nonZeros() != _mass_on_pattern.nonZeros())
        throw std::logic_error("Projection: the two Helmholtz terms do not share a pattern.");
      _helmholtz = _stiffness_on_pattern;
      const auto split = [&](const std::vector<Eigen::Triplet<double>> & local, int d)
      {
        std::vector<double> values;
        for (const auto & entry : ops.globalEntries(local, ops.velocity_fixed[d]))
          values.push_back(entry.value);
        return values;
      };
      _helmholtz_stiffness_values.clear();
      _helmholtz_mass_values.clear();
      _helmholtz_identity_rows.clear();
      for (int d = 0; d < ops.dim; ++d)
      {
        _velocity_solvers.push_back(ops.makeSolver(
            entries, ops.velocity_fixed[d], _velocity_options, _settings.velocity_tolerance));
        // globalEntries keeps the order of the triplets and appends the identity rows, so the values of the system are those of K, then of M, then ones.
        const std::size_t identity = ops.globalEntries({}, ops.velocity_fixed[d]).size();
        auto k = split(ops.stiffness_entries, d), m = split(ops.mass_entries, d);
        k.resize(k.size() - identity);
        m.resize(m.size() - identity);
        _helmholtz_stiffness_values.push_back(std::move(k));
        _helmholtz_mass_values.push_back(std::move(m));
        _helmholtz_identity_rows.push_back(identity);
      }
    }
    else
      for (int d = 0; d < ops.dim; ++d)
      {
        const auto & k = _helmholtz_stiffness_values[d];
        const auto & m = _helmholtz_mass_values[d];
        std::vector<double> values;
        values.reserve(k.size() + m.size() + _helmholtz_identity_rows[d]);
        for (double v : k)
          values.push_back(nu * v);
        for (double v : m)
          values.push_back(c * v);
        values.resize(values.size() + _helmholtz_identity_rows[d], 1.0);
        _velocity_solvers[d]->setValues(values);
      }
    // The local operator on the same pattern, for the prescribed values and the reactions.
    const Index nonzeros = _helmholtz.nonZeros();
    for (Index k = 0; k < nonzeros; ++k)
      _helmholtz.valuePtr()[k] =
          nu * _stiffness_on_pattern.valuePtr()[k] + c * _mass_on_pattern.valuePtr()[k];
    _helmholtz_c = c;
  }

  // The extrapolated solution, with the values prescribed at the new time: the boundary velocity of the pressure equation and the first guess of the solves.
  Vector extrapolated = Vector::Zero(solution.size());
  for (int j = 1; j <= k; ++j)
    extrapolated += alpha[j] * levels[static_cast<std::size_t>(j - 1)]->second;
  _problem.setTime(time);
  _problem.applyDirichlet(extrapolated, 1.0);

  // ---- the right-hand sides: (grad q, u*) / dt for the pressure and (v, u*) / dt for the velocity ----
  QuadratureSpec spec;
  spec.points =
      mesh.numElements() > 0 && ReferenceElement::get(mesh.element(0).type).order() > 1 ? 4 : 3;
  // The velocity of the steps before, and its gradient, at one point: [level][component] and [level][component][direction].
  struct Scratch
  {
    std::vector<IntegrationPoint> points;
    MappedPoint mp;
    double u[3][3], grad[3][3][3];
    Vector pressure_rhs;
    std::vector<Vector> velocity_rhs;
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
        for (int d = 0; d < ops.dim; ++d)
        {
          const double value = U[_problem.dof(el.nodes[a], _velocity[d])];
          sc.u[j][d] += sc.mp.N[a] * value;
          for (int m = 0; m < ops.dim; ++m)
            sc.grad[j][d][m] += sc.mp.dN[a][m] * value;
        }
    }
  };
  const FlowOperators::Geometry & geo = *ops.geometry;
  // u* at the point of sc.mp, from the velocity of the steps before.
  const auto computeStar = [&](const Scratch & sc, double star[3])
  {
    for (int d = 0; d < 3; ++d)
      star[d] = 0;
    for (int d = 0; d < ops.dim; ++d)
    {
      for (int j = 1; j <= k; ++j)
      {
        double advection = 0;
        for (int m = 0; m < ops.dim; ++m)
          advection += sc.u[j - 1][m] * sc.grad[j - 1][d][m];
        star[d] += -beta[j] * sc.u[j - 1][d] - dt * alpha[j] * advection;
      }
      if (d < static_cast<int>(_settings.body_force.size()) && _settings.body_force[d])
        star[d] += dt * _settings.body_force[d]->value(sc.mp.x, time) / rho;
    }
  };
  const auto element = [&](Index e, Scratch & sc)
  {
    if (!ops.in_flow[e])
      return;
    const Element & el = mesh.element(e);
    const double h = ops.element_size[e];
    MappedPoint & mp = sc.mp;
    for (int q = 0; q < geo.volume.count(e); ++q)
    {
      geo.volume.at(el, e, q, mp);
      const Index point = geo.volume.first[e] + q;
      const double w = geo.volume.weight[point];
      evaluate(el, sc);
      // The splitting weighs the divergence of the velocity in the pressure equation by beta_0 / dt, as a pressure stabilization of parameter dt / beta_0 would.
      // That parameter vanishes with the step, and equal-order elements then lose their pressure stability, so the weight is capped at 1 / tau, with the parameter tau of Tezduyar (1991) of the monolithic solver without its time term (stabilizationTau): the scheme is unchanged where dt / beta_0 exceeds tau, and tends to the stabilized monolithic discretization as dt goes to zero.
      // The weighted divergence is that of the combination - sum_j beta_j u^{n-j} / beta_0 of the steps before, which enters the pressure equation through u*.
      double speed2 = 0, divergence = 0;
      for (int d = 0; d < ops.dim; ++d)
      {
        double value = 0;
        for (int a = 0; a < el.numNodes(); ++a)
          value += mp.N[a] * extrapolated[_problem.dof(el.nodes[a], _velocity[d])];
        speed2 += value * value;
        for (int j = 1; j <= k; ++j)
          divergence -= beta[j] / beta[0] * sc.grad[j - 1][d][d];
      }
      const double weight = std::min(c, 1.0 / stabilizationTau(h, speed2, nu, 0.0));
      double star[3];
      computeStar(sc, star);
      if (ops.dual)
      {
        // The integral over the control volume of the node that owns the point.
        // The part c of the weighted divergence is the flux out of the control volume (below), as the term (grad q, u*) / dt is, so that the two cancel where the velocity of the steps before has no divergence of its own, also where the velocity is not continuous across the sides of a cell with hanging nodes.
        const Index o = geo.volume.owner[point];
        for (int d = 0; d < ops.dim; ++d)
          sc.velocity_rhs[d][o] += w * star[d] / dt;
        sc.pressure_rhs[o] -= w * weight * divergence;
        continue;
      }
      for (int a = 0; a < el.numNodes(); ++a)
      {
        const Index n = el.nodes[a];
        double flux = 0;
        for (int d = 0; d < ops.dim; ++d)
        {
          flux += mp.dN[a][d] * star[d];
          sc.velocity_rhs[d][n] += w * mp.N[a] * star[d] / dt;
        }
        sc.pressure_rhs[n] += w * (flux / dt + (c - weight) * mp.N[a] * divergence);
      }
    }
    if (!ops.dual)
      return;
    // (grad q, u*) / dt of a control volume is - the flux of u* / dt out of it.
    for (int q = 0; q < geo.faces.count(e); ++q)
    {
      geo.faces.at(el, e, q, mp);
      const Index point = geo.faces.first[e] + q;
      evaluate(el, sc);
      double star[3];
      computeStar(sc, star);
      const Point & a = geo.faces.area[point];
      double flux = 0;
      for (int d = 0; d < ops.dim; ++d)
      {
        flux += star[d] * a[d];
        for (int j = 1; j <= k; ++j)
          flux += dt * c * beta[j] / beta[0] * sc.u[j - 1][d] * a[d];
      }
      sc.pressure_rhs[geo.faces.owner[point]] -= flux / dt;
      sc.pressure_rhs[geo.faces.neighbor[point]] += flux / dt;
    }
  };
  // Every thread adds into vectors of its own; a body force written in Python keeps the loop on one thread.
  int threads = _problem.effectiveThreads();
  for (const auto & f : _settings.body_force)
    if (f && !f->threadSafe())
      threads = 1;
  std::vector<Scratch> scratch(static_cast<std::size_t>(threads));
  for (auto & sc : scratch)
  {
    sc.pressure_rhs = Vector::Zero(L.n);
    sc.velocity_rhs.assign(static_cast<std::size_t>(ops.dim), Vector::Zero(L.n));
  }
  const Index elements = _problem.numIntegratedElements();
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
  std::vector<Vector> velocity_rhs = std::move(scratch[0].velocity_rhs);
  for (std::size_t t = 1; t < scratch.size(); ++t)
  {
    pressure_rhs += scratch[t].pressure_rhs;
    for (int d = 0; d < ops.dim; ++d)
      velocity_rhs[d] += scratch[t].velocity_rhs[d];
  }
  Scratch boundary;
  MappedPoint & mp = boundary.mp;
  auto & points = boundary.points;
  auto & grad = boundary.grad;
  // ---- the boundary terms of the pressure equation -----------------------------------------
  // - (beta_0 / dt) int q n . u_b - nu int n . (omega x grad q), with omega the extrapolated vorticity.
  for (const Side & side : ops.boundary_sides)
  {
    const Element & el = mesh.element(side.first);
    buildSidePoints(mesh, side, ops.dual, spec, points);
    for (const auto & p : points)
    {
      mapPoint(mesh, el, p.xi, mp);
      evaluate(el, boundary);
      double ub[3] = {0, 0, 0};
      for (int a = 0; a < el.numNodes(); ++a)
        for (int d = 0; d < ops.dim; ++d)
          ub[d] += mp.N[a] * extrapolated[_problem.dof(el.nodes[a], _velocity[d])];
      double omega[3] = {0, 0, 0};
      for (int j = 1; j <= k; ++j)
      {
        const auto & g = grad[j - 1];
        omega[0] += alpha[j] * (g[2][1] - g[1][2]);
        omega[1] += alpha[j] * (g[0][2] - g[2][0]);
        omega[2] += alpha[j] * (g[1][0] - g[0][1]);
      }
      const Point & area = p.area;
      double normal_velocity = 0;
      for (int d = 0; d < ops.dim; ++d)
      {
        normal_velocity += area[d] * ub[d];
        // The part c of the weighted divergence of DMCDM, through the boundary.
        if (ops.dual)
          for (int j = 1; j <= k; ++j)
            normal_velocity += beta[j] / beta[0] * boundary.u[j - 1][d] * area[d];
      }
      for (int a = 0; a < el.numNodes(); ++a)
      {
        const Point & gq = mp.dN[a];
        // n . (omega x grad q)
        const double curl = area[0] * (omega[1] * gq[2] - omega[2] * gq[1]) +
                            area[1] * (omega[2] * gq[0] - omega[0] * gq[2]) +
                            area[2] * (omega[0] * gq[1] - omega[1] * gq[0]);
        // The finite element method tests the boundary velocity with the shape functions, the methods of the dual mesh with the part of the boundary of each control volume.
        const double test = ops.dual ? (a == p.owner ? 1.0 : 0.0) : mp.N[a];
        pressure_rhs[el.nodes[a]] -= c * test * normal_velocity + nu * curl;
      }
    }
  }

  // ---- the pressure -----------------------------------------------------------------------------
  Vector P = _kinematic_pressure;
  for (Index i = 0; i < L.n; ++i)
    if (_pressure_fixed[i])
      P[i] = _pressure_pinned ? 0.0 : extrapolated[_problem.dof(i, _pressure)] / rho;
  _pressure_iterations = ops.solve(
      *_pressure_solver, ops.stiffness, _pressure_fixed, pressure_rhs, P, _pressure_pinned);
  // The pressure of an enclosed flow with a mean of zero, as the monolithic solve reports it.
  if (_pressure_pinned)
    P.array() -= L.sumOwned(_volume.cwiseProduct(P)) / _total_volume;
  _kinematic_pressure = P;

  // ---- the velocity -------------------------------------------------------------------------------
  // The integral of P n over the boundary of the support of every test function (of every control volume), which turns the pressure term (w, dP/dx) of the velocity equation into its integrated form - (dw/dx, P) + int w P n_x, for the reactions below.
  std::vector<Vector> boundary_pressure(static_cast<std::size_t>(ops.dim), Vector::Zero(L.n));
  for (const Side & side : ops.boundary_sides)
  {
    const Element & el = mesh.element(side.first);
    buildSidePoints(mesh, side, ops.dual, spec, points);
    for (const auto & p : points)
    {
      mapPoint(mesh, el, p.xi, mp);
      double pressure = 0;
      for (int a = 0; a < el.numNodes(); ++a)
        pressure += mp.N[a] * P[el.nodes[a]];
      for (int a = 0; a < el.numNodes(); ++a)
      {
        const double test = ops.dual ? (a == p.owner ? 1.0 : 0.0) : mp.N[a];
        for (int d = 0; d < ops.dim; ++d)
          boundary_pressure[d][el.nodes[a]] += test * pressure * p.area[d];
      }
    }
  }
  _velocity_iterations = 0;
  solution = extrapolated;
  // The residual of the momentum equations, whose rows of prescribed velocities are the reactions (the force on the boundary is minus their sum), as the monolithic solve leaves them.
  Vector residual = Vector::Zero(solution.size());
  for (int d = 0; d < ops.dim; ++d)
  {
    Vector rhs = velocity_rhs[d] - ops.gradient[d] * P;
    Vector values(L.n);
    for (Index i = 0; i < L.n; ++i)
      values[i] = extrapolated[_problem.dof(i, _velocity[d])];
    _velocity_iterations +=
        ops.solve(*_velocity_solvers[d], _helmholtz, ops.velocity_fixed[d], rhs, values, false);
    for (Index i = 0; i < L.n; ++i)
      solution[_problem.dof(i, _velocity[d])] = values[i];
    Vector r = _helmholtz * values - rhs - boundary_pressure[d];
    L.sum(r);
    // A dependent node (a periodic copy on one process) carries no reaction of its own, so that a sum over a boundary counts the node once.
    for (Index i = 0; i < L.n; ++i)
      residual[_problem.dof(i, _velocity[d])] = rho * r[i];
  }
  for (Index i = 0; i < L.n; ++i)
    solution[_problem.dof(i, _pressure)] = rho * P[i];
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
                       DistributedProblem * distributed,
                       const std::vector<int> & velocities,
                       int pressure,
                       Vector & solution)
{
  const FlowOperators ops(problem, distributed, velocities);
  const FlowOperators::Layout & L = *ops.layout;
  const FlowOperators::Geometry & geo = *ops.geometry;
  const Mesh & mesh = problem.mesh();
  const int dim = ops.dim;
  const Index elements = problem.numIntegratedElements();

  // ---- the divergence D_d and the Laplacian of the multiplier --------------------------------
  // D tests the divergence as the mass balance of the flow does without its stabilization: with the shape functions of the pressure (FEM), (q, div u) with the coordinate factor and the hoop term u_r / r, or as the flux of u out of the control volume of each node (DMCDM).
  // The Taylor-Hood element tests with the linear shape functions of the corners, which are the quadratic ones weighted by their values at the nodes.
  const bool corners = pressure >= 0 && problem.hasMixedOrder() &&
                       problem.variable(pressure).order == VariableOrder::First;
  const bool curved = problem.coordinateSystem() != CoordinateSystem::Cartesian;
  std::vector<Eigen::Triplet<double>> divergence[3], laplacian;
  // The measure of the test function of every node.
  Vector volume = Vector::Zero(L.n);
  MappedPoint mp;
  std::vector<IntegrationPoint> points;
  for (Index e = 0; e < elements; ++e)
  {
    if (!ops.in_flow[e])
      continue;
    const Element & el = mesh.element(e);
    const int nn = el.numNodes();
    if (ops.dual)
    {
      for (int q = 0; q < geo.volume.count(e); ++q)
      {
        geo.volume.at(el, e, q, mp);
        const Index point = geo.volume.first[e] + q;
        volume[geo.volume.owner[point]] += geo.volume.weight[point] * problem.coordFactor(mp.x);
      }
      for (int q = 0; q < geo.faces.count(e); ++q)
      {
        geo.faces.at(el, e, q, mp);
        const Index point = geo.faces.first[e] + q;
        const Index o = geo.faces.owner[point], n = geo.faces.neighbor[point];
        const double f = problem.coordFactor(mp.x);
        for (int b = 0; b < nn; ++b)
          for (int d = 0; d < dim; ++d)
          {
            const double flux = f * mp.N[b] * geo.faces.area[point][d];
            divergence[d].emplace_back(o, el.nodes[b], flux);
            divergence[d].emplace_back(n, el.nodes[b], -flux);
          }
      }
      continue;
    }
    // The weights of the corner shape functions in the shape functions of the element.
    const int nq = corners ? el.numCorners() : nn;
    double weights[kMaxElementNodes][kMaxElementNodes];
    for (int a = 0; a < nq; ++a)
      for (int b = 0; b < nn; ++b)
        weights[a][b] = a == b ? 1.0 : 0.0;
    if (corners && nq < nn)
    {
      const auto & quadratic = ReferenceElement::get(el.type);
      const auto & linear = ReferenceElement::get(elementCornerType(el.type));
      double N[kMaxElementNodes];
      Point dN[kMaxElementNodes];
      for (int b = 0; b < nn; ++b)
      {
        linear.shape(quadratic.node(b), N, dN);
        for (int a = 0; a < nq; ++a)
          weights[a][b] = N[a];
      }
    }
    for (int q = 0; q < geo.volume.count(e); ++q)
    {
      geo.volume.at(el, e, q, mp);
      const Index point = geo.volume.first[e] + q;
      const double w = geo.volume.weight[point] * problem.coordFactor(mp.x);
      // The term u_r / r of the divergence in axisymmetric coordinates, and 2 u_r / r in spherical ones.
      const double hoop =
          curved && mp.x[0] > 0.0
              ? (problem.coordinateSystem() == CoordinateSystem::Axisymmetric ? 1.0 : 2.0) / mp.x[0]
              : 0.0;
      double test[kMaxElementNodes];
      Point grad[kMaxElementNodes];
      for (int a = 0; a < nq; ++a)
      {
        test[a] = 0.0;
        grad[a] = {0.0, 0.0, 0.0};
        for (int b = 0; b < nn; ++b)
        {
          test[a] += weights[a][b] * mp.N[b];
          for (int d = 0; d < 3; ++d)
            grad[a][d] += weights[a][b] * mp.dN[b][d];
        }
      }
      for (int a = 0; a < nq; ++a)
      {
        volume[el.nodes[a]] += w * test[a];
        for (int b = 0; b < nn; ++b)
          for (int d = 0; d < dim; ++d)
            divergence[d].emplace_back(el.nodes[a],
                                       el.nodes[b],
                                       w * test[a] *
                                           (mp.dN[b][d] + (d == 0 ? hoop * mp.N[b] : 0.0)));
        for (int b = 0; b < nq; ++b)
          laplacian.emplace_back(el.nodes[a], el.nodes[b], w * dot(grad[a], grad[b]));
      }
    }
  }
  if (ops.dual)
  {
    // The flux through the boundary of the domain, of the prescribed velocity where it is prescribed.
    for (const Side & side : ops.boundary_sides)
    {
      const Element & el = mesh.element(side.first);
      QuadratureSpec spec;
      spec.points = ReferenceElement::get(el.type).order();
      buildSidePoints(mesh, side, true, spec, points);
      for (const auto & p : points)
      {
        mapPoint(mesh, el, p.xi, mp);
        const double f = problem.coordFactor(mp.x);
        for (int b = 0; b < el.numNodes(); ++b)
          for (int d = 0; d < dim; ++d)
            divergence[d].emplace_back(el.nodes[p.owner], el.nodes[b], f * mp.N[b] * p.area[d]);
      }
    }
    laplacian = ops.stiffness_entries;
  }
  for (int d = 0; d < dim; ++d)
    L.fold(divergence[d]);
  L.fold(laplacian);
  L.sum(volume);
  // ---- the rows of the multiplier ----------------------------------------------------------
  // The multiplier is zero where the pressure is prescribed, whose mass balance is not an equation of the flow, and at a node that tests no divergence (a node outside the flow, or a node of the Taylor-Hood element that is not a corner).
  std::vector<SparseMatrix> D(static_cast<std::size_t>(dim), SparseMatrix(L.n, L.n));
  for (int d = 0; d < dim; ++d)
    D[d].setFromTriplets(divergence[d].begin(), divergence[d].end());
  Vector tested = Vector::Zero(L.n);
  for (int d = 0; d < dim; ++d)
    for (const auto & t : divergence[d])
      tested[t.row()] += std::abs(t.value());
  L.sum(tested);
  const std::vector<char> given =
      pressure >= 0 ? ops.prescribed(pressure, false) : std::vector<char>(L.n, 0);
  std::vector<char> fixed(static_cast<std::size_t>(L.n), 0);
  for (Index i = 0; i < L.n; ++i)
    fixed[i] = !L.depends(i) && (given[i] || !(tested[i] > 0.0)) ? 1 : 0;
  // A flow whose normal velocity is prescribed on the whole boundary (an enclosed flow, or one
  // with a prescribed inflow and outflow) leaves the multiplier free up to a constant: D^T 1 is
  // zero at every free velocity.  The constant is fixed at the entity of the smallest global
  // number, and the divergence that the data give the whole flow is spread over its volume.
  bool enclosed = !L.any(std::any_of(given.begin(), given.end(), [](char c) { return c != 0; }));
  for (int d = 0; d < dim && enclosed; ++d)
  {
    Vector column = Vector::Zero(L.n), size = Vector::Zero(L.n);
    for (const auto & t : divergence[d])
    {
      column[t.col()] += t.value();
      size[t.col()] += std::abs(t.value());
    }
    L.sum(column);
    L.sum(size);
    double open = 0.0, scale = 0.0;
    for (Index i = 0; i < L.n; ++i)
    {
      scale = std::max(scale, size[i]);
      if (!ops.velocity_fixed[d][i])
        open = std::max(open, std::abs(column[i]));
    }
    enclosed = ops.maximum(open) <= 1e-10 * ops.maximum(scale);
  }
  Vector rhs = Vector::Zero(L.n);
  for (int d = 0; d < dim; ++d)
  {
    Vector u(L.n);
    for (Index i = 0; i < L.n; ++i)
      u[i] = solution[problem.dof(i, velocities[d])];
    rhs -= D[d] * u;
  }
  L.sum(rhs);
  if (enclosed)
  {
    double total = 0.0, measure = 0.0;
    for (Index i = 0; i < L.n; ++i)
      if (L.owned[i] && !fixed[i])
      {
        total += rhs[i];
        measure += volume[i];
      }
    const double mean = ops.sum(total) / ops.sum(measure);
    double first = std::numeric_limits<double>::infinity();
    for (Index i = 0; i < L.n; ++i)
      if (!fixed[i])
      {
        rhs[i] -= mean * volume[i];
        if (L.owned[i])
          first = std::min(first, static_cast<double>(L.global[i]));
      }
    first = distributed ? distributed->communicator().min(first) : first;
    for (Index i = 0; i < L.n; ++i)
      if (!L.depends(i) && static_cast<double>(L.global[i]) == first)
        fixed[i] = 1;
  }

  // ---- the saddle point system -------------------------------------------------------------
  //     [ M      D^T    ] [ du     ]   [ 0       ]
  //     [ D   -eps K_l  ] [ lambda ] = [ - D u_0 ]
  // with M the mass matrix of each velocity component and K_l the Laplacian of the multiplier, which has the spectrum of D M^-1 D^T on the smooth fields.
  // The small regularization eps K_l fixes the multiplier where D^T has a null space (the spurious pressure modes of equal-order elements), whose divergence no velocity changes.
  const double eps = 1e-4;
  const int width = dim + 1;
  const auto row = [&](Index i, int field) { return width * L.global[i] + field; };
  const auto free = [&](Index i, int field)
  {
    if (L.global[i] < 0)
      return false;
    return field < dim ? !ops.velocity_fixed[field][i] : !fixed[i];
  };
  std::vector<petsc::GlobalEntry> entries;
  const auto add = [&](Index i, int fi, Index j, int fj, double value)
  {
    if (free(i, fi) && free(j, fj))
      entries.push_back({row(i, fi), row(j, fj), value});
  };
  for (int d = 0; d < dim; ++d)
    for (const auto & t : ops.mass_entries)
      add(t.row(), d, t.col(), d, t.value());
  for (int d = 0; d < dim; ++d)
    for (const auto & t : divergence[d])
    {
      add(t.row(), dim, t.col(), d, t.value());
      add(t.col(), d, t.row(), dim, t.value());
    }
  for (const auto & t : laplacian)
    add(t.row(), dim, t.col(), dim, -eps * t.value());
  for (Index i = 0; i < L.n; ++i)
    if (L.owned[i])
      for (int f = 0; f <= dim; ++f)
        if (!free(i, f))
          entries.push_back({row(i, f), row(i, f), 1.0});
  petsc::Settings settings;
  settings.relative_tolerance = 1e-12;
  settings.symmetric = false;
  const bool across = distributed != nullptr && distributed->communicator().size() > 1;
  const petsc::LinearSolver system(
      across, width * L.num_global, width * L.first, width * L.num_owned, entries, settings);
  // The regularization moves the smooth part of the solution by about eps relative to its size, which corrections with the residual of the system without it take back (iterated regularization), until a correction changes the velocity by less than 1e-13 of the speed.
  // A spurious mode that D^T nearly annihilates keeps a part of its divergence, which a velocity of bounded size cannot remove.
  const auto at = [&](Index i, int field) { return width * (L.global[i] - L.first) + field; };
  std::vector<Vector> du(static_cast<std::size_t>(dim), Vector::Zero(L.n));
  Vector lambda = Vector::Zero(L.n);
  Vector b = Vector::Zero(width * L.num_owned);
  for (Index i = 0; i < L.n; ++i)
    if (L.owned[i] && free(i, dim))
      b[at(i, dim)] = rhs[i];
  double speed0 = 0.0;
  for (Index i = 0; i < L.n; ++i)
    for (int d = 0; d < dim; ++d)
      speed0 = std::max(speed0, std::abs(solution[problem.dof(i, velocities[d])]));
  speed0 = ops.maximum(speed0);
  for (int pass = 0; pass < 20; ++pass)
  {
    petsc::Result result;
    const Vector x = system.solve(b, nullptr, result);
    if (L.any(!result.converged))
      throw std::runtime_error("dualmesh: the projection of the initial velocity onto a field "
                               "without divergence did not converge (" +
                               result.reason + ", " + std::to_string(result.iterations) +
                               " iterations).");
    double moved = 0.0, reached = speed0;
    for (int f = 0; f <= dim; ++f)
    {
      Vector update = Vector::Zero(L.n);
      for (Index i = 0; i < L.n; ++i)
        if (L.owned[i] && free(i, f))
          update[i] = x[at(i, f)];
      L.copy(update);
      Vector & total = f < dim ? du[static_cast<std::size_t>(f)] : lambda;
      total += update;
      if (f < dim)
      {
        moved = std::max(moved, update.lpNorm<Eigen::Infinity>());
        reached = std::max(reached, total.lpNorm<Eigen::Infinity>());
      }
    }
    if (ops.maximum(moved) <= 1e-13 * ops.maximum(reached))
      break;
    Vector divergence_left = rhs;
    for (int d = 0; d < dim; ++d)
    {
      Vector momentum = ops.mass * du[d] + D[d].transpose() * lambda;
      L.sum(momentum);
      Vector change = D[d] * du[d];
      L.sum(change);
      divergence_left -= change;
      for (Index i = 0; i < L.n; ++i)
        if (L.owned[i] && free(i, d))
          b[at(i, d)] = -momentum[i];
    }
    for (Index i = 0; i < L.n; ++i)
      if (L.owned[i] && free(i, dim))
        b[at(i, dim)] = divergence_left[i];
  }

  // ---- the velocity ------------------------------------------------------------------------
  double change = 0.0, speed = 0.0;
  for (Index i = 0; i < L.n; ++i)
  {
    double change2 = 0.0, speed2 = 0.0;
    for (int d = 0; d < dim; ++d)
    {
      double & u = solution[problem.dof(i, velocities[d])];
      if (!ops.velocity_fixed[d][i])
      {
        u += du[d][i];
        change2 += du[d][i] * du[d][i];
      }
      speed2 += u * u;
    }
    change = std::max(change, std::sqrt(change2));
    speed = std::max(speed, std::sqrt(speed2));
  }
  change = ops.maximum(change);
  speed = ops.maximum(speed);
  return speed > 0.0 ? change / speed : 0.0;
}

} // namespace dualmesh
