// SPDX-License-Identifier: LGPL-2.1-or-later
#include "dualmesh/modules/Projection.h"
#include "dualmesh/core/InputParameters.h"
#include "dualmesh/fe/Assembly.h"
#include "dualmesh/parallel/DistributedProblem.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <set>
#include <stdexcept>

namespace dualmesh
{

/// Where the values of one scalar field (one value per local entity) lie in the global systems, and the exchanges between the processes that hold copies of an entity.
/// A problem on one process numbers its entities as they are, except that a periodic boundary makes a node and its primary one entity.
struct ProjectionSolver::Layout
{
  DistributedProblem * distributed = nullptr;
  Index n = 0;
  std::vector<Index> global;
  std::vector<char> owned;
  /// The entity that represents each local entity (itself, or the primary of a periodic copy); used on one process only.
  std::vector<Index> representative;
  Index first = 0, num_owned = 0, num_global = 0;

  /// Add the values of every copy of an entity, so that every copy holds the sum.
  void sum(Vector & v) const
  {
    if (distributed)
    {
      distributed->addAcrossRanks(v.data(), 1);
      return;
    }
    for (Index i = 0; i < n; ++i)
      if (representative[i] != i)
        v[representative[i]] += v[i];
    copy(v);
  }
  /// Give every copy of an entity the value of the copy that owns it.
  void copy(Vector & v) const
  {
    if (distributed)
    {
      distributed->copyFromOwners(v.data(), 1);
      return;
    }
    for (Index i = 0; i < n; ++i)
      v[i] = v[representative[i]];
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

/// The geometry of the quadrature points of every integrated element, computed once, as Nek5000 keeps its geometric factors: the weight (with the Jacobian), the position and the inverse transposed Jacobian at each point, and the shape functions and their reference derivatives once per element type.
/// There are two sets of points: those of the element interiors (the finite element method) or of the control volumes, which carry the node that owns them, and, for the methods of the dual mesh, those of the interfaces between control volumes, which carry their two nodes and their area vector.
/// The finite element method takes a Gauss rule of order + 1 points per direction, which integrates the mass matrix and the advection exactly on affine elements.
/// The methods of the dual mesh take order points per direction of every patch of a control volume and of every interface, the midpoint rule for linear elements, which is second order like the methods themselves: the errors of the Taylor-Green vortex are those of two points per direction, and a step costs 2.4 times less.
struct ProjectionSolver::Geometry
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
    std::vector<Index> first; // the first point of each element
    std::vector<double> weight;
    std::vector<Point> area, x;
    std::vector<std::array<double, 9>> G; // J^{-T}, row major
    std::vector<int> owner, neighbor;     // local nodes; -1 for the finite element method

    int count(Index e) const { return static_cast<int>(first[e + 1] - first[e]); }

    /// Fill @p mp with the shape functions and physical gradients at point @p q of element @p e.
    void at(const Element & el, Index e, int q, MappedPoint & mp) const
    {
      const Table & t = tables.at(el.type);
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

    /// The points of every element of @p mesh up to @p elements, of @p set.
    void build(const Mesh & mesh, Index elements, bool dual, PointSet set)
    {
      std::vector<IntegrationPoint> points;
      MappedPoint mp;
      first.assign(static_cast<std::size_t>(elements) + 1, 0);
      for (Index e = 0; e < elements; ++e)
      {
        const Element & el = mesh.element(e);
        const auto & ref = ReferenceElement::get(el.type);
        QuadratureSpec spec;
        spec.points = dual ? ref.order() : ref.order() + 1;
        buildElementPoints(mesh, e, dual, set, spec, points);
        auto [table, added] = tables.try_emplace(el.type);
        if (added)
        {
          table->second.nodes = el.numNodes();
          for (const auto & p : points)
          {
            double N[kMaxElementNodes];
            Point dN[kMaxElementNodes];
            ref.shape(p.xi, N, dN);
            table->second.N.insert(table->second.N.end(), N, N + el.numNodes());
            table->second.dN.insert(table->second.dN.end(), dN, dN + el.numNodes());
          }
        }
        first[e + 1] = first[e] + static_cast<Index>(points.size());
        const int dim = ref.dimension();
        for (const auto & p : points)
        {
          mapPoint(mesh, el, p.xi, mp);
          // J^{-T}: the physical gradient is G times the reference gradient.
          const auto & J = mp.J;
          const double id = 1.0 / mp.detJ;
          double inv[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
          if (dim == 1)
            inv[0][0] = id;
          else if (dim == 2)
          {
            inv[0][0] = J[1][1] * id;
            inv[0][1] = -J[0][1] * id;
            inv[1][0] = -J[1][0] * id;
            inv[1][1] = J[0][0] * id;
          }
          else
          {
            inv[0][0] = (J[1][1] * J[2][2] - J[1][2] * J[2][1]) * id;
            inv[0][1] = (J[0][2] * J[2][1] - J[0][1] * J[2][2]) * id;
            inv[0][2] = (J[0][1] * J[1][2] - J[0][2] * J[1][1]) * id;
            inv[1][0] = (J[1][2] * J[2][0] - J[1][0] * J[2][2]) * id;
            inv[1][1] = (J[0][0] * J[2][2] - J[0][2] * J[2][0]) * id;
            inv[1][2] = (J[0][2] * J[1][0] - J[0][0] * J[1][2]) * id;
            inv[2][0] = (J[1][0] * J[2][1] - J[1][1] * J[2][0]) * id;
            inv[2][1] = (J[0][1] * J[2][0] - J[0][0] * J[2][1]) * id;
            inv[2][2] = (J[0][0] * J[1][1] - J[0][1] * J[1][0]) * id;
          }
          std::array<double, 9> g{};
          for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
              g[3 * i + j] = inv[j][i];
          G.push_back(g);
          weight.push_back(p.weight);
          area.push_back(p.area);
          x.push_back(mp.x);
          owner.push_back(p.owner);
          neighbor.push_back(p.neighbor);
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

void
ProjectionSolver::setUp()
{
  const Mesh & mesh = _problem.mesh();
  _dim = mesh.dimension();
  if (_problem.method() == Method::FiniteVolumeCell)
    throw InputError("Projection: the projection time integration runs with the finite element "
                     "method and the methods of the dual mesh (fem, dmcdm and hfvm) so far.");
  _dual = _problem.method() != Method::FiniteElement;
  if (_problem.coordinateSystem() != CoordinateSystem::Cartesian)
    throw InputError("Projection: the projection time integration runs in Cartesian coordinates "
                     "so far.");
  if (_problem.hasMixedOrder())
    throw InputError("Projection: the velocity and the pressure must have the order of the mesh "
                     "(the Taylor-Hood element is not supported by the projection time "
                     "integration so far).");
  if (static_cast<int>(_settings.velocities.size()) != _dim)
    throw InputError("Projection: one velocity variable per coordinate is needed.");
  _velocity.clear();
  for (const auto & name : _settings.velocities)
    _velocity.push_back(_problem.variableIndex(name));
  _pressure = _problem.variableIndex(_settings.pressure);
  if (_problem.numVariables() != _dim + 1)
    throw InputError("Projection: the projection time integration advances the velocity and the "
                     "pressure of a flow alone; this problem has other variables as well.");
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
        throw InputError(
            "Projection: '" + name +
            "' acts on the flow, which the projection time integration assembles "
            "from its density, viscosity and body force alone. Give a body force "
            "with the body_force of incompressible_flow, prescribe the velocity or the "
            "pressure on the boundary, or use time_integration='monolithic'.");
    }
  }

  // ---- the layout of a scalar field --------------------------------------------------------
  const int nv = _problem.numVariables();
  _layout = std::make_unique<Layout>();
  Layout & L = *_layout;
  L.distributed = _distributed;
  L.n = _problem.numEntities();
  if (_distributed)
  {
    L.global = _distributed->globalEntityNumbers();
    L.owned.assign(static_cast<std::size_t>(L.n), 0);
    for (Index i = 0; i < L.n; ++i)
      L.owned[i] = _distributed->ownsNode(i) ? 1 : 0;
    L.first = _distributed->firstOwnedEntity();
    L.num_owned = _distributed->numOwnedEntities();
    L.num_global = _distributed->numGlobalEntities();
  }
  else
  {
    // A periodic boundary joins every variable of a flow, so the joins of the first velocity are those of every field.
    const auto & primary = _problem.primaryDofs();
    L.representative.resize(static_cast<std::size_t>(L.n));
    for (Index i = 0; i < L.n; ++i)
    {
      L.representative[i] = i;
      if (!primary.empty())
      {
        const Index p = primary[_problem.dof(i, _velocity[0])];
        for (int v = 0; v < nv; ++v)
        {
          const Index q = primary[_problem.dof(i, v)];
          if ((q < 0) != (p < 0) || (q >= 0 && q / nv != p / nv))
            throw InputError("Projection: a periodic boundary of a flow must join the velocity "
                             "and the pressure alike; leave 'variables' out.");
        }
        if (p >= 0)
          L.representative[i] = p / nv;
      }
    }
    L.global.assign(static_cast<std::size_t>(L.n), -1);
    L.owned.assign(static_cast<std::size_t>(L.n), 0);
    Index next = 0;
    for (Index i = 0; i < L.n; ++i)
      if (L.representative[i] == i)
      {
        L.global[i] = next++;
        L.owned[i] = 1;
      }
    for (Index i = 0; i < L.n; ++i)
      L.global[i] = L.global[L.representative[i]];
    L.first = 0;
    L.num_owned = L.num_global = next;
  }

  // ---- the geometry of the quadrature points ---------------------------------------------------
  _geometry = std::make_unique<Geometry>();
  Geometry & geo = *_geometry;
  const Index elements = _problem.numIntegratedElements();
  geo.volume.build(mesh, elements, _dual, PointSet::Volume);
  if (_dual)
    geo.faces.build(mesh, elements, true, PointSet::Faces);

  // ---- the mass, stiffness and gradient matrices ---------------------------------------------------
  // The finite element method tests with the shape functions.
  // The methods of the dual mesh test with the control volumes: the mass and the gradient are integrals over the control volume of each node, and the Laplacian is - the flux of the gradient out of it, through the interfaces between control volumes.
  // The vertex-centered finite volume method takes the gradient on an interface from the difference along the edge between its two nodes, with the interpolated part transverse to the edge (as its assembly does).
  std::vector<Eigen::Triplet<double>> gradient_entries[3];
  MappedPoint mp;
  const bool edge_gradient = _problem.method() == Method::FiniteVolumeVertex;
  for (Index e = 0; e < elements; ++e)
  {
    const Element & el = mesh.element(e);
    const int nn = el.numNodes();
    for (int q = 0; q < geo.volume.count(e); ++q)
    {
      geo.volume.at(el, e, q, mp);
      const Index point = geo.volume.first[e] + q;
      const double w = geo.volume.weight[point];
      if (_dual)
      {
        const Index o = el.nodes[geo.volume.owner[point]];
        for (int b = 0; b < nn; ++b)
        {
          _mass_entries.emplace_back(o, el.nodes[b], w * mp.N[b]);
          for (int d = 0; d < _dim; ++d)
            gradient_entries[d].emplace_back(o, el.nodes[b], w * mp.dN[b][d]);
        }
        continue;
      }
      for (int a = 0; a < nn; ++a)
        for (int b = 0; b < nn; ++b)
        {
          _mass_entries.emplace_back(el.nodes[a], el.nodes[b], w * mp.N[a] * mp.N[b]);
          _stiffness_entries.emplace_back(el.nodes[a], el.nodes[b], w * dot(mp.dN[a], mp.dN[b]));
          for (int d = 0; d < _dim; ++d)
            gradient_entries[d].emplace_back(el.nodes[a], el.nodes[b], w * mp.N[a] * mp.dN[b][d]);
        }
    }
    if (!_dual)
      continue;
    for (int q = 0; q < geo.faces.count(e); ++q)
    {
      geo.faces.at(el, e, q, mp);
      const Index point = geo.faces.first[e] + q;
      const int lo = geo.faces.owner[point], ln = geo.faces.neighbor[point];
      const Index o = el.nodes[lo], n = el.nodes[ln];
      const Point & a = geo.faces.area[point];
      const Point d = mesh.node(n) - mesh.node(o);
      const double dd = dot(d, d);
      for (int b = 0; b < nn; ++b)
      {
        Point g = mp.dN[b];
        if (edge_gradient && dd > 0)
        {
          const double jump = (b == ln ? 1.0 : 0.0) - (b == lo ? 1.0 : 0.0);
          g = g + ((jump - dot(g, d)) / dd) * d;
        }
        const double flux = dot(g, a);
        _stiffness_entries.emplace_back(o, el.nodes[b], -flux);
        _stiffness_entries.emplace_back(n, el.nodes[b], flux);
      }
    }
  }
  _mass.resize(L.n, L.n);
  _mass.setFromTriplets(_mass_entries.begin(), _mass_entries.end());
  _stiffness.resize(L.n, L.n);
  _stiffness.setFromTriplets(_stiffness_entries.begin(), _stiffness_entries.end());
  _gradient.assign(static_cast<std::size_t>(_dim), SparseMatrix(L.n, L.n));
  _element_size.resize(static_cast<std::size_t>(_problem.numIntegratedElements()));
  for (Index e = 0; e < _problem.numIntegratedElements(); ++e)
    _element_size[e] = elementSize(mesh, e);
  for (int d = 0; d < _dim; ++d)
    _gradient[d].setFromTriplets(gradient_entries[d].begin(), gradient_entries[d].end());

  // ---- the prescribed rows -----------------------------------------------------------------------
  // A copy of an entity is prescribed when any copy is.
  const std::vector<char> constrained = _problem.constrainedDofs();
  const auto & primary = _problem.primaryDofs();
  const auto fixedOf = [&](int variable)
  {
    Vector mark(L.n);
    for (Index i = 0; i < L.n; ++i)
    {
      const Index d = _problem.dof(i, variable);
      mark[i] = constrained[d] && (primary.empty() || primary[d] < 0) ? 1.0 : 0.0;
    }
    L.sum(mark);
    std::vector<char> fixed(static_cast<std::size_t>(L.n));
    for (Index i = 0; i < L.n; ++i)
      fixed[i] = mark[i] > 0.5 ? 1 : 0;
    return fixed;
  };
  _velocity_fixed.clear();
  for (int d = 0; d < _dim; ++d)
    _velocity_fixed.push_back(fixedOf(_velocity[d]));
  _pressure_fixed = fixedOf(_pressure);
  // An enclosed flow determines the pressure up to a constant: the constant is fixed at the entity numbered zero, after the right-hand side is made compatible.
  _pressure_pinned =
      !L.any(std::find(_pressure_fixed.begin(), _pressure_fixed.end(), 1) != _pressure_fixed.end());
  if (_pressure_pinned)
    for (Index i = 0; i < L.n; ++i)
      _pressure_fixed[i] = L.global[i] == 0 ? 1 : 0;
  // The Laplacian of the finite element method is symmetric, solved by the conjugate gradient method; that of the dual mesh is not in general, solved by GMRES.
  _pressure_options = _settings.pressure_solver.empty()
                          ? std::string(_dual ? "-ksp_type gmres" : "-ksp_type cg") +
                                " -pc_type hypre -pc_hypre_type boomeramg"
                          : _settings.pressure_solver;
  _velocity_options =
      _settings.velocity_solver.empty()
          ? std::string(_dual ? "-ksp_type gmres" : "-ksp_type cg") + " -pc_type jacobi"
          : _settings.velocity_solver;
  _pressure_solver = makeSolver(_stiffness_entries,
                                _pressure_fixed,
                                _pressure_options,
                                _settings.pressure_tolerance,
                                _settings.pressure_projection_vectors);
  _helmholtz_c = -1.0;

  // ---- the boundary ----------------------------------------------------------------------------
  std::set<std::pair<Index, int>> sides;
  for (const auto & [name, list] : mesh.sidesets())
    for (const Side & side : list)
      if (side.first < _problem.numIntegratedElements())
        sides.insert({side.first, side.second});
  _boundary_sides.clear();
  for (const auto & [element, side] : sides)
    _boundary_sides.push_back({element, side});
  _ready = true;
}

std::vector<petsc::GlobalEntry>
ProjectionSolver::globalEntries(const std::vector<Eigen::Triplet<double>> & local,
                                const std::vector<char> & fixed) const
{
  const Layout & L = *_layout;
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
ProjectionSolver::makeSolver(const std::vector<Eigen::Triplet<double>> & local,
                             const std::vector<char> & fixed,
                             const std::string & options,
                             double tolerance,
                             int projection_vectors) const
{
  const Layout & L = *_layout;
  const std::vector<petsc::GlobalEntry> entries = globalEntries(local, fixed);
  petsc::Settings settings;
  settings.options = options;
  settings.relative_tolerance = tolerance;
  settings.projection_vectors = projection_vectors;
  settings.symmetric = !_dual;
  const bool distributed = _distributed != nullptr && _distributed->communicator().size() > 1;
  return std::make_unique<petsc::LinearSolver>(
      distributed, L.num_global, L.first, L.num_owned, entries, settings);
}

int
ProjectionSolver::solve(const petsc::LinearSolver & solver,
                        const SparseMatrix & matrix,
                        const std::vector<char> & fixed,
                        Vector rhs,
                        Vector & values,
                        bool remove_mean) const
{
  const Layout & L = *_layout;
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
  _kinematic_pressure.resize(_layout->n);
  for (Index i = 0; i < _layout->n; ++i)
    _kinematic_pressure[i] = solution[_problem.dof(i, _pressure)] / _settings.density;
}

double
ProjectionSolver::courantNumber(const Vector & solution, double dt) const
{
  const Mesh & mesh = _problem.mesh();
  double largest = 0;
  for (Index e = 0; e < _problem.numIntegratedElements(); ++e)
  {
    const Element & el = mesh.element(e);
    double speed2 = 0;
    for (int a = 0; a < el.numNodes(); ++a)
    {
      double s2 = 0;
      for (int d = 0; d < _dim; ++d)
      {
        const double v = solution[_problem.dof(el.nodes[a], _velocity[d])];
        s2 += v * v;
      }
      speed2 = std::max(speed2, s2);
    }
    largest = std::max(largest, std::sqrt(speed2) / _element_size[e]);
  }
  if (_distributed)
    largest = _distributed->communicator().max(largest);
  return dt * largest;
}

void
ProjectionSolver::accept(const Vector & solution, double time)
{
  _history.push_front({time, solution});
  while (static_cast<int>(_history.size()) > _settings.order)
    _history.pop_back();
}

SolveResult
ProjectionSolver::step(Vector & solution, const Vector & old, double time, double dt)
{
  (void) old;
  const Layout & L = *_layout;
  const Mesh & mesh = _problem.mesh();
  const double rho = _settings.density;
  const double nu = _settings.dynamic_viscosity / rho;
  const int k = std::min<int>(_settings.order, static_cast<int>(_history.size()));
  std::vector<double> times{time};
  for (int j = 0; j < k; ++j)
    times.push_back(_history[j].first);
  if (std::abs((times[0] - times[1]) - dt) > 1e-9 * dt)
    throw std::logic_error("Projection: a step does not start from the last accepted solution.");
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
      entries.reserve(_mass_entries.size() + _stiffness_entries.size());
      for (const auto & t : _stiffness_entries)
        entries.emplace_back(t.row(), t.col(), nu * t.value());
      for (const auto & t : _mass_entries)
        entries.emplace_back(t.row(), t.col(), c * t.value());
      _stiffness_on_pattern = _stiffness + 0.0 * _mass;
      _mass_on_pattern = _mass + 0.0 * _stiffness;
      _stiffness_on_pattern.makeCompressed();
      _mass_on_pattern.makeCompressed();
      if (_stiffness_on_pattern.nonZeros() != _mass_on_pattern.nonZeros())
        throw std::logic_error("Projection: the two Helmholtz terms do not share a pattern.");
      _helmholtz = _stiffness_on_pattern;
      const auto split = [&](const std::vector<Eigen::Triplet<double>> & local, int d)
      {
        std::vector<double> values;
        for (const auto & entry : globalEntries(local, _velocity_fixed[d]))
          values.push_back(entry.value);
        return values;
      };
      _helmholtz_stiffness_values.clear();
      _helmholtz_mass_values.clear();
      _helmholtz_identity_rows.clear();
      for (int d = 0; d < _dim; ++d)
      {
        _velocity_solvers.push_back(makeSolver(
            entries, _velocity_fixed[d], _velocity_options, _settings.velocity_tolerance));
        // globalEntries keeps the order of the triplets and appends the identity rows, so the values of the system are those of K, then of M, then ones.
        const std::size_t identity = globalEntries({}, _velocity_fixed[d]).size();
        auto k = split(_stiffness_entries, d), m = split(_mass_entries, d);
        k.resize(k.size() - identity);
        m.resize(m.size() - identity);
        _helmholtz_stiffness_values.push_back(std::move(k));
        _helmholtz_mass_values.push_back(std::move(m));
        _helmholtz_identity_rows.push_back(identity);
      }
    }
    else
      for (int d = 0; d < _dim; ++d)
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
    extrapolated += alpha[j] * _history[j - 1].second;
  _problem.setTime(time);
  if (_distributed)
    _distributed->applyDirichlet(extrapolated);
  else
  {
    _problem.applyDirichlet(extrapolated, 1.0);
    _problem.updateDependentDofs(extrapolated);
  }

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
      const Vector & U = _history[j].second;
      for (int d = 0; d < 3; ++d)
      {
        sc.u[j][d] = 0;
        for (int m = 0; m < 3; ++m)
          sc.grad[j][d][m] = 0;
      }
      for (int a = 0; a < el.numNodes(); ++a)
        for (int d = 0; d < _dim; ++d)
        {
          const double value = U[_problem.dof(el.nodes[a], _velocity[d])];
          sc.u[j][d] += sc.mp.N[a] * value;
          for (int m = 0; m < _dim; ++m)
            sc.grad[j][d][m] += sc.mp.dN[a][m] * value;
        }
    }
  };
  const Geometry & geo = *_geometry;
  // u* at the point of sc.mp, from the velocity of the steps before.
  const auto computeStar = [&](const Scratch & sc, double star[3])
  {
    for (int d = 0; d < 3; ++d)
      star[d] = 0;
    for (int d = 0; d < _dim; ++d)
    {
      for (int j = 1; j <= k; ++j)
      {
        double advection = 0;
        for (int m = 0; m < _dim; ++m)
          advection += sc.u[j - 1][m] * sc.grad[j - 1][d][m];
        star[d] += -beta[j] * sc.u[j - 1][d] - dt * alpha[j] * advection;
      }
      if (d < static_cast<int>(_settings.body_force.size()) && _settings.body_force[d])
        star[d] += dt * _settings.body_force[d]->value(sc.mp.x, time) / rho;
    }
  };
  const auto element = [&](Index e, Scratch & sc)
  {
    const Element & el = mesh.element(e);
    const double h = _element_size[e];
    MappedPoint & mp = sc.mp;
    for (int q = 0; q < geo.volume.count(e); ++q)
    {
      geo.volume.at(el, e, q, mp);
      const Index point = geo.volume.first[e] + q;
      const double w = geo.volume.weight[point];
      evaluate(el, sc);
      // The splitting weighs the divergence of the velocity in the pressure equation by beta_0 / dt, as a pressure stabilization of parameter dt / beta_0 would.
      // That parameter vanishes with the step, and equal-order elements then lose their pressure stability, so the weight is capped at 1 / tau, with the parameter tau of Tezduyar (1991) of the monolithic solver: the scheme is unchanged where dt / beta_0 exceeds tau, and tends to the stabilized monolithic discretization as dt goes to zero.
      // The weighted divergence is that of the combination - sum_j beta_j u^{n-j} / beta_0 of the steps before, which enters the pressure equation through u*.
      double speed2 = 0, divergence = 0;
      for (int d = 0; d < _dim; ++d)
      {
        double value = 0;
        for (int a = 0; a < el.numNodes(); ++a)
          value += mp.N[a] * extrapolated[_problem.dof(el.nodes[a], _velocity[d])];
        speed2 += value * value;
        for (int j = 1; j <= k; ++j)
          divergence -= beta[j] / beta[0] * sc.grad[j - 1][d][d];
      }
      const double advective = 2.0 * std::sqrt(speed2) / h, viscous = 4.0 * nu / (h * h);
      const double weight = std::min(c, std::sqrt(advective * advective + viscous * viscous));
      double star[3];
      computeStar(sc, star);
      if (_dual)
      {
        // The integral over the control volume of the node that owns the point.
        const Index o = el.nodes[geo.volume.owner[point]];
        for (int d = 0; d < _dim; ++d)
          sc.velocity_rhs[d][o] += w * star[d] / dt;
        sc.pressure_rhs[o] += w * (c - weight) * divergence;
        continue;
      }
      for (int a = 0; a < el.numNodes(); ++a)
      {
        const Index n = el.nodes[a];
        double flux = 0;
        for (int d = 0; d < _dim; ++d)
        {
          flux += mp.dN[a][d] * star[d];
          sc.velocity_rhs[d][n] += w * mp.N[a] * star[d] / dt;
        }
        sc.pressure_rhs[n] += w * (flux / dt + (c - weight) * mp.N[a] * divergence);
      }
    }
    if (!_dual)
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
      for (int d = 0; d < _dim; ++d)
        flux += star[d] * a[d];
      sc.pressure_rhs[el.nodes[geo.faces.owner[point]]] -= flux / dt;
      sc.pressure_rhs[el.nodes[geo.faces.neighbor[point]]] += flux / dt;
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
    sc.velocity_rhs.assign(static_cast<std::size_t>(_dim), Vector::Zero(L.n));
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
    for (int d = 0; d < _dim; ++d)
      velocity_rhs[d] += scratch[t].velocity_rhs[d];
  }
  Scratch boundary;
  MappedPoint & mp = boundary.mp;
  auto & points = boundary.points;
  auto & grad = boundary.grad;
  // ---- the boundary terms of the pressure equation -----------------------------------------
  // - (beta_0 / dt) int q n . u_b - nu int n . (omega x grad q), with omega the extrapolated vorticity.
  for (const Side & side : _boundary_sides)
  {
    const Element & el = mesh.element(side.first);
    buildSidePoints(mesh, side, _dual, spec, points);
    for (const auto & p : points)
    {
      mapPoint(mesh, el, p.xi, mp);
      evaluate(el, boundary);
      double ub[3] = {0, 0, 0};
      for (int a = 0; a < el.numNodes(); ++a)
        for (int d = 0; d < _dim; ++d)
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
      for (int d = 0; d < _dim; ++d)
        normal_velocity += area[d] * ub[d];
      for (int a = 0; a < el.numNodes(); ++a)
      {
        const Point & gq = mp.dN[a];
        // n . (omega x grad q)
        const double curl = area[0] * (omega[1] * gq[2] - omega[2] * gq[1]) +
                            area[1] * (omega[2] * gq[0] - omega[0] * gq[2]) +
                            area[2] * (omega[0] * gq[1] - omega[1] * gq[0]);
        // The finite element method tests the boundary velocity with the shape functions, the methods of the dual mesh with the part of the boundary of each control volume.
        const double test = _dual ? (a == p.owner ? 1.0 : 0.0) : mp.N[a];
        pressure_rhs[el.nodes[a]] -= c * test * normal_velocity + nu * curl;
      }
    }
  }

  // ---- the pressure -----------------------------------------------------------------------------
  Vector P = _kinematic_pressure;
  for (Index i = 0; i < L.n; ++i)
    if (_pressure_fixed[i])
      P[i] = _pressure_pinned ? 0.0 : extrapolated[_problem.dof(i, _pressure)] / rho;
  _pressure_iterations =
      solve(*_pressure_solver, _stiffness, _pressure_fixed, pressure_rhs, P, _pressure_pinned);
  _kinematic_pressure = P;

  // ---- the velocity -------------------------------------------------------------------------------
  // The integral of P n over the boundary of the support of every test function (of every control volume), which turns the pressure term (w, dP/dx) of the velocity equation into its integrated form - (dw/dx, P) + int w P n_x, for the reactions below.
  std::vector<Vector> boundary_pressure(static_cast<std::size_t>(_dim), Vector::Zero(L.n));
  for (const Side & side : _boundary_sides)
  {
    const Element & el = mesh.element(side.first);
    buildSidePoints(mesh, side, _dual, spec, points);
    for (const auto & p : points)
    {
      mapPoint(mesh, el, p.xi, mp);
      double pressure = 0;
      for (int a = 0; a < el.numNodes(); ++a)
        pressure += mp.N[a] * P[el.nodes[a]];
      for (int a = 0; a < el.numNodes(); ++a)
      {
        const double test = _dual ? (a == p.owner ? 1.0 : 0.0) : mp.N[a];
        for (int d = 0; d < _dim; ++d)
          boundary_pressure[d][el.nodes[a]] += test * pressure * p.area[d];
      }
    }
  }
  _velocity_iterations = 0;
  solution = extrapolated;
  // The residual of the momentum equations, whose rows of prescribed velocities are the reactions (the force on the boundary is minus their sum), as the monolithic solve leaves them.
  Vector residual = Vector::Zero(solution.size());
  for (int d = 0; d < _dim; ++d)
  {
    Vector rhs = velocity_rhs[d] - _gradient[d] * P;
    Vector values(L.n);
    for (Index i = 0; i < L.n; ++i)
      values[i] = extrapolated[_problem.dof(i, _velocity[d])];
    _velocity_iterations +=
        solve(*_velocity_solvers[d], _helmholtz, _velocity_fixed[d], rhs, values, false);
    for (Index i = 0; i < L.n; ++i)
      solution[_problem.dof(i, _velocity[d])] = values[i];
    Vector r = _helmholtz * values - rhs - boundary_pressure[d];
    L.sum(r);
    // A periodic copy of a node on one process carries no reaction of its own, so that a sum over a boundary counts the node once.
    for (Index i = 0; i < L.n; ++i)
      residual[_problem.dof(i, _velocity[d])] = _distributed || L.owned[i] ? rho * r[i] : 0.0;
  }
  for (Index i = 0; i < L.n; ++i)
    solution[_problem.dof(i, _pressure)] = rho * P[i];
  _problem.setLastResidual(residual);

  SolveResult result;
  result.converged = true;
  result.total_iterations = 1;
  result.linear_iterations = _pressure_iterations + _velocity_iterations;
  return result;
}

} // namespace dualmesh
