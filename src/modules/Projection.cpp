// SPDX-License-Identifier: LGPL-2.1-or-later
#include "dualmesh/modules/Projection.h"
#include "ProjectionShared.h"
#include "dualmesh/core/Coordinates.h"
#include "dualmesh/fe/Assembly.h"
#include "dualmesh/modules/FluidSystem.h"

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
#include <string>

namespace dualmesh
{

namespace
{

/// The relative difference of the density over the fluid below which it is uniform.
constexpr double kUniformDensity = 1e-12;
/// The relative difference below which two step sizes, or the coefficients beta_0 / dt of two
/// steps, are the same up to the rounding of the times.
constexpr double kSameStep = 1e-9;
/// The relative tolerance of the projection of the sources, which goes to round-off.
constexpr double kSourceTolerance = 1e-14;
/// The points per direction of the integrals over the boundary, for elements of first order and
/// of higher order.
constexpr int kBoundaryPointsLinear = 3;
constexpr int kBoundaryPointsQuadratic = 4;

} // namespace

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

struct ProjectionSolver::Terms
{
  /// The time derivatives (M), and the terms with a flux (A and G), taken at the new time.
  std::vector<const Kernel *> storage, implicit;
};

void
ProjectionSolver::setUp()
{
  _problem.initialize();
  const int nv = _problem.numVariables();
  const MassEquation * constraint = _problem.flowConstraint();
  if (!constraint)
    throw InputError("Projection: the projection time integration advances a flow with a "
                     "pressure, the variable of its mass_conservation equation, and this problem "
                     "has none (the penalty formulation eliminates the pressure).");
  _pressure = constraint->variable();
  const FlowCoefficients flow = flowCoefficientsAt(_problem, *constraint, _problem.mesh().node(0));
  _velocity.assign(flow.velocities, flow.velocities + flow.num_velocities);
  _stabilized = flow.stabilized;
  if (nv != flow.num_velocities + 1)
    throw InputError("Projection: the projection time integration advances the velocity and the "
                     "pressure of a flow alone, and this problem has other variables as well.");
  std::vector<char> is_velocity(static_cast<std::size_t>(nv), 0), is_pressure(is_velocity);
  for (int v : _velocity)
    is_velocity[static_cast<std::size_t>(v)] = 1;
  is_pressure[static_cast<std::size_t>(_pressure)] = 1;

  const Terms terms = splitTerms(*constraint, is_velocity);
  setDensity(*constraint, flow.time_density, terms);
  setElementData(*constraint);
  setBoundarySides();
  assembleOperators(terms, is_velocity, is_pressure);
  buildPoints();
  const std::vector<char> constrained = _problem.constrainedDofs();
  checkOpenBoundaries(constrained);
  buildVelocityRows(constrained);
  buildPressureSystem(constrained);
  buildSourceSystem(constrained);
  _ready = true;
}

ProjectionSolver::Terms
ProjectionSolver::splitTerms(const MassEquation & constraint, const std::vector<char> & is_velocity)
{
  const auto refuse = [](const ResidualObject & object)
  {
    throw InputError("Projection: '" + object.name() +
                     "' acts on the flow in a way that the projection time integration does not "
                     "split: a term of the mass equation other than mass_conservation and "
                     "mass_flux, or a boundary condition of the momentum equations other than a "
                     "prescribed velocity. Prescribe the velocity or the pressure on the "
                     "boundary, or use time_integration='monolithic'.");
  };
  Terms terms;
  _sources.clear();
  _advection = false;
  for (const auto & k : _problem.kernels())
  {
    if (k->variable() == _pressure)
    {
      if (k.get() != &constraint)
        refuse(*k);
      continue;
    }
    if (k->isTimeKernel())
      terms.storage.push_back(k.get());
    else if (k->isStabilization())
      continue;
    else if (const auto * advection = dynamic_cast<const AdvectiveKernel *>(k.get());
             advection && advection->isInertia())
      _advection = true;
    else if (k->hasFlux())
      terms.implicit.push_back(k.get());
    else
      _sources.push_back(k.get());
  }
  for (const auto & bc : _problem.integratedBCs())
    if (is_velocity[static_cast<std::size_t>(bc->variable())] ||
        (bc->variable() == _pressure && bc->type() != "mass_flux"))
      refuse(*bc);
  return terms;
}

void
ProjectionSolver::setDensity(const MassEquation & constraint, double density, const Terms & terms)
{
  // The pressure equation divides the momentum equations by the density, so the fluid is one of
  // uniform density.
  const Mesh & mesh = _problem.mesh();
  _density = density;
  for (Index i = 0; i < mesh.numNodes(); ++i)
  {
    const FlowCoefficients c = flowCoefficientsAt(_problem, constraint, mesh.node(i));
    const double scale = std::abs(_density);
    if (std::abs(c.time_density - _density) > kUniformDensity * scale ||
        (_advection && std::abs(c.density - _density) > kUniformDensity * scale))
      throw InputError("Projection: the projection time integration takes a fluid of one "
                       "uniform density, the coefficient of the time derivative and of the "
                       "advection of its velocity.");
  }
  if (terms.storage.empty() || !(_density > 0.0))
    throw InputError("Projection: the projection time integration needs a positive density. A "
                     "flow without inertia (Stokes flow) has no time derivative to integrate.");
}

void
ProjectionSolver::setElementData(const MassEquation & constraint)
{
  const Mesh & mesh = _problem.mesh();
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
    _viscosity[e] = flowCoefficientsAt(_problem, constraint, centre, e).viscosity / _density;
  }
}

void
ProjectionSolver::setBoundarySides()
{
  const Index elements = static_cast<Index>(_in_flow.size());
  std::set<std::pair<Index, int>> sides;
  for (const auto & [name, list] : _problem.mesh().sidesets())
    for (const Side & side : list)
      if (side.first < elements && _in_flow[side.first])
        sides.insert({side.first, side.second});
  _boundary_sides.assign(sides.begin(), sides.end());
}

void
ProjectionSolver::assembleOperators(const Terms & terms,
                                    const std::vector<char> & is_velocity,
                                    const std::vector<char> & is_pressure)
{
  Problem::OperatorOptions at_rest;
  at_rest.kernels = terms.storage;
  _mass = projection::block(_problem, _problem.operatorOf(at_rest), is_velocity, is_velocity);
  at_rest.kernels = terms.implicit;
  const SparseMatrix implicit = _problem.operatorOf(at_rest);
  _viscous = projection::block(_problem, implicit, is_velocity, is_velocity);
  _gradient = projection::block(_problem, implicit, is_velocity, is_pressure);
  const auto laplacian =
      projection::privateObject<Kernel>(_problem,
                                        "diffusion",
                                        "projection_pressure_laplacian",
                                        {{"variable", _problem.variable(_pressure).name}});
  at_rest.kernels = std::vector<const Kernel *>{laplacian.get()};
  _laplacian = _problem.operatorOf(at_rest);
  _helmholtz = _viscous + _mass;
  _helmholtz.makeCompressed();
  _helmholtz_c = -1.0;
}

void
ProjectionSolver::buildPoints()
{
  _points = std::make_unique<Points>();
  const auto corners = [&](const Element & el) { return _problem.usesCornerShape(el, _pressure); };
  _points->volume.build(_problem, _in_flow, PointSet::Volume, corners);
  if (_problem.isDual())
    _points->faces.build(_problem, _in_flow, PointSet::Faces, corners);
}

void
ProjectionSolver::checkOpenBoundaries(const std::vector<char> & constrained) const
{
  // A boundary without a prescribed velocity, open to the outside, takes the pressure from the
  // traction there, which the pressure equation needs as its condition on that boundary.
  const Mesh & mesh = _problem.mesh();
  const Index elements = static_cast<Index>(_in_flow.size());
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
}

void
ProjectionSolver::buildVelocityRows(const std::vector<char> & constrained)
{
  const int nv = _problem.numVariables();
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
  // The solvers follow at the first step, with the Helmholtz matrix of its step size.
  _velocity_solvers.clear();
}

void
ProjectionSolver::buildPressureSystem(const std::vector<char> & constrained)
{
  // An enclosed flow, whose pressure no condition gives, determines it up to a constant: the
  // constant is fixed at the free unknown of the smallest row, after the right-hand side is made
  // compatible, and the pressure is then shifted to a mean of zero, as the monolithic solve
  // reports it.
  const auto & active = _problem.activeDofs();
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
    for (Index i = 0; i < _problem.mesh().numNodes(); ++i)
      if (_pressure_rows->owned[i])
        total += _pressure_weight[i];
    _total_weight = _problem.parallel().sum(total);
  }
  _pressure_solver =
      _pressure_rows->makeSolver(_laplacian,
                                 krylovOption() + " -pc_type hypre -pc_hypre_type boomeramg",
                                 _settings.pressure_tolerance,
                                 !_problem.isDual(),
                                 _settings.pressure_projection_vectors);
}

void
ProjectionSolver::buildSourceSystem(const std::vector<char> & constrained)
{
  // The sources per unit mass are the L2 projection of their residual onto the velocity, a
  // solve with the mass matrix whose rows only the dependent and inactive unknowns fix.  A source
  // that does not change is projected once, so the solve goes to round-off, below the
  // tolerances of the steps.
  const auto & active = _problem.activeDofs();
  _source_solver.reset();
  _source_residual.resize(0);
  if (_sources.empty())
    return;
  std::vector<char> free_fixed(constrained.size(), 0);
  for (Index d = 0; d < _problem.numDofs(); ++d)
    free_fixed[d] = _problem.isDependent(d) || (!active.empty() && !active[d]) ? 1 : 0;
  _source_rows = std::make_unique<Rows>(_problem, _velocity, free_fixed);
  _source_solver = _source_rows->makeSolver(
      _mass, krylovOption() + " -pc_type jacobi", kSourceTolerance, !_problem.isDual());
}

std::string
ProjectionSolver::krylovOption() const
{
  return _problem.isDual() ? "-ksp_type gmres" : "-ksp_type cg";
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
ProjectionSolver::elementTau(Index e, double speed2) const
{
  return stabilizationTau(_element_size[e], speed2, _viscosity[e], 0.0);
}

double
ProjectionSolver::largestNodalSpeed2(Index e, const Vector & solution) const
{
  const Element & el = _problem.mesh().element(e);
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
  return speed2;
}

double
ProjectionSolver::courantNumber(const Vector & solution, double dt) const
{
  double largest = 0;
  for (Index e = 0; e < static_cast<Index>(_in_flow.size()); ++e)
    if (_in_flow[e])
      largest = std::max(largest, std::sqrt(largestNodalSpeed2(e, solution)) / _element_size[e]);
  return dt * _problem.parallel().maximum(largest);
}

double
ProjectionSolver::largestStep(const Vector & solution) const
{
  if (!_stabilized)
    return std::numeric_limits<double>::infinity();
  // The pressure equation weighs the divergence by min(beta_0 / dt, 1 / tau), with tau at the
  // speed of each point.  The largest nodal speed of an element gives its smallest tau.
  double smallest = std::numeric_limits<double>::infinity();
  for (Index e = 0; e < static_cast<Index>(_in_flow.size()); ++e)
    if (_in_flow[e])
      smallest = std::min(smallest, elementTau(e, largestNodalSpeed2(e, solution)));
  // The next step starts from the accepted states and takes the order k of step(): 1 in the
  // first step, then one more per accepted state up to the order of the settings.  Its beta_0 is
  // that of a constant step, 1 + 1/2 + ... + 1/k.
  const int k = stepOrder(_history.size());
  std::vector<double> times, beta, alpha;
  for (int j = 0; j <= k; ++j)
    times.push_back(-static_cast<double>(j));
  coefficients(times, beta, alpha);
  return beta[0] * _problem.parallel().minimum(smallest);
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

std::vector<double>
ProjectionSolver::controlDomainWeights(const Vector & extrapolated, double c) const
{
  // DMCDM weighs the divergence of each control domain as the flux through its sides, so its
  // weight is one value per control domain: that is what leaves a velocity without divergence
  // without weighted divergence.  It is min(c, 1 / tau) with 1 / tau averaged over the control
  // domain at the velocity of its node.  The two integrals of the average add over the processes,
  // so every process gives a node the same weight.
  std::vector<double> node_weight(static_cast<std::size_t>(_problem.mesh().numNodes()), c);
  if (!_stabilized)
    return node_weight;
  const Points & geo = *_points;
  std::vector<double> inverse_tau(node_weight.size(), 0.0), measure(node_weight.size(), 0.0);
  for (Index e = 0; e < static_cast<Index>(_in_flow.size()); ++e)
  {
    if (!_in_flow[e])
      continue;
    for (int q = 0; q < geo.volume.count(e); ++q)
    {
      const Index point = geo.volume.first[e] + q;
      const auto n = static_cast<std::size_t>(geo.volume.owner[point]);
      double speed2 = 0;
      for (int v : _velocity)
      {
        const double u = extrapolated[_problem.dof(static_cast<Index>(n), v)];
        speed2 += u * u;
      }
      const double w = geo.volume.weight[point];
      inverse_tau[n] += w / elementTau(e, speed2);
      measure[n] += w;
    }
  }
  _problem.parallel().addAtNodes(inverse_tau.data(), 1);
  _problem.parallel().addAtNodes(measure.data(), 1);
  for (std::size_t n = 0; n < node_weight.size(); ++n)
    if (measure[n] > 0)
      node_weight[n] = std::min(c, inverse_tau[n] / measure[n]);
  return node_weight;
}
struct ProjectionSolver::StepData
{
  /// The levels of the step, the latest first, and its order k.
  std::vector<const std::pair<double, Vector> *> levels;
  int k = 0;
  /// The step size, the coefficients beta_j and alpha_j, and c = beta_0 / dt.
  double dt = 0.0, c = 0.0;
  std::vector<double> beta, alpha;
  /// The extrapolated solution, with the values prescribed at the new time: the boundary velocity
  /// of the pressure equation, the velocity of the sources and the first guess of the solves.
  Vector extrapolated;
  /// Whether the momentum equations have sources (_source).
  bool with_source = false;
  /// DMCDM: the weight of the divergence of each control domain (controlDomainWeights).
  std::vector<double> node_weight;
};

struct ProjectionSolver::Scratch
{
  std::vector<IntegrationPoint> points;
  MappedPoint mp;
  /// The velocity of the steps before, and its gradient, at one point: [level][component] and
  /// [level][component][direction].
  double u[3][3], grad[3][3][3];
  /// The right-hand sides that this thread adds into.
  Vector pressure_rhs, advection_rhs;
};

struct ProjectionSolver::RightHandSides
{
  Vector pressure, advection;
};

SolveResult
ProjectionSolver::step(
    Vector & solution, const Vector & /* old */, double time, double dt, bool continues)
{
  StepData s = stepData(time, dt, continues);
  updateHelmholtz(s.c);
  s.extrapolated = extrapolate(s, time, solution.size());
  sources(s.extrapolated, time);
  s.with_source = _source.size() > 0;
  if (_problem.isDual())
    s.node_weight = controlDomainWeights(s.extrapolated, s.c);
  RightHandSides rhs = assembleRightHandSides(s, solution.size());
  addBoundaryTerms(s, rhs.pressure);
  const Vector P = solvePressure(s, rhs.pressure);
  solveVelocity(s, rhs.advection, P, solution);
  _trial = {time, solution};

  SolveResult result;
  result.converged = true;
  result.total_iterations = 1;
  result.linear_iterations = _pressure_iterations + _velocity_iterations;
  return result;
}

ProjectionSolver::StepData
ProjectionSolver::stepData(double time, double dt, bool continues) const
{
  // The levels of the step: the accepted states, after the state of the step just taken when
  // this step continues it (the second half of a doubled step).  The solver keeps these states
  // with their times, which the coefficients of a variable step need, so its newest level holds
  // the state that the interface passes as old, and that argument is not read.
  StepData s;
  if (continues)
    s.levels.push_back(&_trial);
  for (const auto & level : _history)
    s.levels.push_back(&level);
  s.k = stepOrder(s.levels.size());
  std::vector<double> times{time};
  for (int j = 0; j < s.k; ++j)
    times.push_back(s.levels[static_cast<std::size_t>(j)]->first);
  if (std::abs((times[0] - times[1]) - dt) > kSameStep * dt)
    throw std::logic_error(
        "Projection: a step starts from the last accepted solution or from the step just taken.");
  coefficients(times, s.beta, s.alpha);
  s.dt = dt;
  s.c = s.beta[0] / dt;
  return s;
}

void
ProjectionSolver::updateHelmholtz(double c)
{
  // The Helmholtz matrix changes with beta_0 / dt: in the first steps, at a step shortened to land on an output time, and at every step of the cfl stepper.
  // Its structure does not, so the solver is made once and later given the new values.
  // A step of the same size gives the same coefficient up to the rounding of the times, which must not touch the solver.
  if (!(std::abs(c - _helmholtz_c) > kSameStep * std::abs(c)))
    return;
  _helmholtz = c * _mass + _viscous;
  _helmholtz.makeCompressed();
  for (std::size_t i = 0; i < _velocity_rows.size(); ++i)
    if (_velocity_solvers.size() <= i)
      _velocity_solvers.push_back(_velocity_rows[i]->makeSolver(_helmholtz,
                                                                krylovOption() + " -pc_type jacobi",
                                                                _settings.velocity_tolerance,
                                                                !_problem.isDual()));
    else
    {
      std::vector<double> values;
      for (const auto & entry : _velocity_rows[i]->entries(_helmholtz))
        values.push_back(entry.value);
      _velocity_solvers[i]->setValues(values);
    }
  _helmholtz_c = c;
}

Vector
ProjectionSolver::extrapolate(const StepData & s, double time, Index size)
{
  Vector extrapolated = Vector::Zero(size);
  for (int j = 1; j <= s.k; ++j)
    extrapolated += s.alpha[j] * s.levels[static_cast<std::size_t>(j - 1)]->second;
  _problem.setTime(time);
  _problem.applyDirichlet(extrapolated, 1.0);
  return extrapolated;
}

void
ProjectionSolver::evaluateLevels(const Element & el, const StepData & s, Scratch & sc) const
{
  const int dim = static_cast<int>(_velocity.size());
  for (int j = 0; j < s.k; ++j)
  {
    const Vector & U = s.levels[static_cast<std::size_t>(j)]->second;
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
}

void
ProjectionSolver::explicitTerms(const Element & el,
                                const StepData & s,
                                const Scratch & sc,
                                double advection[3],
                                double star[3]) const
{
  const int dim = static_cast<int>(_velocity.size());
  for (int d = 0; d < 3; ++d)
    advection[d] = star[d] = 0;
  for (int d = 0; d < dim; ++d)
  {
    for (int j = 1; j <= s.k; ++j)
    {
      double carried = 0;
      if (_advection)
        for (int m = 0; m < dim; ++m)
          carried += sc.u[j - 1][m] * sc.grad[j - 1][d][m];
      advection[d] += s.alpha[j] * carried;
      star[d] -= s.beta[j] * sc.u[j - 1][d];
    }
    star[d] -= s.dt * advection[d];
    if (s.with_source)
      for (int a = 0; a < el.numNodes(); ++a)
        star[d] += s.dt * sc.mp.N[a] * _source[_problem.dof(el.nodes[a], _velocity[d])];
  }
}

double
ProjectionSolver::divergenceWeight(Index e,
                                   const Element & el,
                                   const MappedPoint & mp,
                                   const StepData & s) const
{
  // The splitting weighs the divergence of the velocity in the pressure equation by beta_0 / dt, as a pressure stabilization of parameter dt / beta_0 would.
  // That parameter vanishes with the step, and equal-order elements then lose their pressure stability, so the weight is capped at 1 / tau, with the parameter tau of Tezduyar (1991) of the monolithic solver without its time term (stabilizationTau): the scheme is unchanged where dt / beta_0 exceeds tau, and tends to the stabilized monolithic discretization as dt goes to zero.
  // The Taylor-Hood element is stable without it and keeps beta_0 / dt.  The weight at a point takes the extrapolated velocity there.
  if (!_stabilized)
    return s.c;
  double u[3];
  extrapolatedVelocity(el, mp, s, u);
  double speed2 = 0;
  for (int d = 0; d < static_cast<int>(_velocity.size()); ++d)
    speed2 += u[d] * u[d];
  return std::min(s.c, 1.0 / elementTau(e, speed2));
}

void
ProjectionSolver::extrapolatedVelocity(const Element & el,
                                       const MappedPoint & mp,
                                       const StepData & s,
                                       double u[3]) const
{
  const int dim = static_cast<int>(_velocity.size());
  for (int d = 0; d < 3; ++d)
    u[d] = 0;
  for (int d = 0; d < dim; ++d)
    for (int a = 0; a < el.numNodes(); ++a)
      u[d] += mp.N[a] * s.extrapolated[_problem.dof(el.nodes[a], _velocity[d])];
}

void
ProjectionSolver::extrapolatedVorticity(const StepData & s,
                                        const Scratch & sc,
                                        double omega[3]) const
{
  for (int d = 0; d < 3; ++d)
    omega[d] = 0;
  for (int j = 1; j <= s.k; ++j)
  {
    const auto & g = sc.grad[j - 1];
    omega[0] += s.alpha[j] * (g[2][1] - g[1][2]);
    omega[1] += s.alpha[j] * (g[0][2] - g[2][0]);
    omega[2] += s.alpha[j] * (g[1][0] - g[0][1]);
  }
}

ProjectionSolver::RightHandSides
ProjectionSolver::assembleRightHandSides(const StepData & s, Index size) const
{
  // (grad q, u*) / dt for the pressure and the advection for the velocity.  Every thread adds
  // into vectors of its own.
  const Mesh & mesh = _problem.mesh();
  const bool dual = _problem.isDual();
  const int threads = _problem.effectiveThreads();
  std::vector<Scratch> scratch(static_cast<std::size_t>(threads));
  for (auto & sc : scratch)
  {
    sc.pressure_rhs = Vector::Zero(size);
    sc.advection_rhs = Vector::Zero(size);
  }
  const auto element = [&](Index e, Scratch & sc)
  {
    if (!_in_flow[e])
      return;
    const Element & el = mesh.element(e);
    addVolumeTerms(e, el, s, sc);
    if (dual)
      addInterfaceTerms(e, el, s, sc);
  };
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
  RightHandSides rhs;
  rhs.pressure = std::move(scratch[0].pressure_rhs);
  rhs.advection = std::move(scratch[0].advection_rhs);
  for (std::size_t t = 1; t < scratch.size(); ++t)
  {
    rhs.pressure += scratch[t].pressure_rhs;
    rhs.advection += scratch[t].advection_rhs;
  }
  return rhs;
}

void
ProjectionSolver::addVolumeTerms(Index e,
                                 const Element & el,
                                 const StepData & s,
                                 Scratch & sc) const
{
  const bool dual = _problem.isDual();
  const int dim = static_cast<int>(_velocity.size());
  const double rho = _density;
  const std::vector<double> & beta = s.beta;
  const double c = s.c;
  const Points & geo = *_points;
  const bool corner = _problem.usesCornerShape(el, _pressure);
  MappedPoint & mp = sc.mp;
  for (int q = 0; q < geo.volume.count(e); ++q)
  {
    geo.volume.at(el, e, q, mp);
    const Index point = geo.volume.first[e] + q;
    const double w = geo.volume.weight[point];
    evaluateLevels(el, s, sc);
    double advection[3], star[3];
    explicitTerms(el, s, sc, advection, star);
    if (dual)
    {
      // The integral over the control domain of the node that owns the point.  Its weighted
      // divergence is a flux through the sides of the control domain (addInterfaceTerms).
      const Index o = geo.volume.owner[point];
      for (int d = 0; d < dim; ++d)
        sc.advection_rhs[_problem.dof(o, _velocity[d])] += w * rho * advection[d];
      continue;
    }
    // The weighted divergence is that of the combination - sum_j beta_j u^{n-j} / beta_0 of the
    // steps before, which enters the pressure equation through u* with the weight c: the part
    // c - weight comes off.
    // The hoop term n u_r / r of the divergence, with n the hoop factor, as its factor n / r.
    const double hoop = hoopOverRadius(_problem.coordinateSystem(), mp.x[0]);
    double divergence = 0;
    for (int d = 0; d < dim; ++d)
      for (int j = 1; j <= s.k; ++j)
        divergence -=
            beta[j] / beta[0] * (sc.grad[j - 1][d][d] + (d == 0 ? hoop * sc.u[j - 1][0] : 0.0));
    const double weight = divergenceWeight(e, el, mp, s);
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
          w * (flux / s.dt + (c - weight) * test * divergence);
    }
  }
}

void
ProjectionSolver::addInterfaceTerms(Index e,
                                    const Element & el,
                                    const StepData & s,
                                    Scratch & sc) const
{
  // (grad q, u*) / dt of a control domain is - the flux of u* / dt out of it.  The velocity of
  // the steps before enters u* with the weight c of its divergence, of which the part c - weight
  // of the control domain comes off: the weighted divergence is the flux through the same sides
  // as that of u*.
  const int dim = static_cast<int>(_velocity.size());
  const std::vector<double> & beta = s.beta;
  const double c = s.c;
  const Points & geo = *_points;
  for (int q = 0; q < geo.faces.count(e); ++q)
  {
    geo.faces.at(el, e, q, sc.mp);
    const Index point = geo.faces.first[e] + q;
    evaluateLevels(el, s, sc);
    double advection[3], star[3];
    explicitTerms(el, s, sc, advection, star);
    const Point & a = geo.faces.area[point];
    double flux = 0, old = 0;
    for (int d = 0; d < dim; ++d)
    {
      flux += star[d] * a[d];
      for (int j = 1; j <= s.k; ++j)
        old += beta[j] / beta[0] * sc.u[j - 1][d] * a[d];
    }
    const Index owner = geo.faces.owner[point], neighbor = geo.faces.neighbor[point];
    const auto part = [&](Index n) { return c - s.node_weight[static_cast<std::size_t>(n)]; };
    sc.pressure_rhs[_problem.dof(owner, _pressure)] -= flux / s.dt + part(owner) * old;
    sc.pressure_rhs[_problem.dof(neighbor, _pressure)] += flux / s.dt + part(neighbor) * old;
  }
}

void
ProjectionSolver::addBoundaryTerms(const StepData & s, Vector & pressure_rhs) const
{
  // - (beta_0 / dt) int q n . u_b - nu int n . (omega x grad q), with omega the extrapolated vorticity.
  const Mesh & mesh = _problem.mesh();
  const bool dual = _problem.isDual();
  const int dim = static_cast<int>(_velocity.size());
  const std::vector<double> & beta = s.beta;
  const double c = s.c;
  QuadratureSpec spec;
  spec.points = mesh.numElements() > 0 && ReferenceElement::get(mesh.element(0).type).order() > 1
                    ? kBoundaryPointsQuadratic
                    : kBoundaryPointsLinear;
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
      evaluateLevels(el, s, boundary);
      double ub[3], omega[3];
      extrapolatedVelocity(el, mp, s, ub);
      extrapolatedVorticity(s, boundary, omega);
      const Point area = _problem.coordFactor(mp.x) * p.area;
      // The weighted divergence of DMCDM through the boundary: the velocity of the steps before
      // with the part c - weight of its weight c (the sides inside, addInterfaceTerms).
      const double old_part =
          dual ? (c - s.node_weight[static_cast<std::size_t>(el.nodes[p.owner])]) / c : 0.0;
      double normal_velocity = 0;
      for (int d = 0; d < dim; ++d)
      {
        normal_velocity += area[d] * ub[d];
        for (int j = 1; j <= s.k; ++j)
          normal_velocity += old_part * beta[j] / beta[0] * boundary.u[j - 1][d] * area[d];
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
}

Vector
ProjectionSolver::solvePressure(const StepData & s, const Vector & pressure_rhs)
{
  const Mesh & mesh = _problem.mesh();
  const Rows & pressure_rows = *_pressure_rows;
  Vector P = _kinematic_pressure;
  pressure_rows.forEachFixed([&](Index d)
                             { P[d] = _pressure_pinned ? 0.0 : s.extrapolated[d] / _density; });
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
  return P;
}

void
ProjectionSolver::solveVelocity(const StepData & s,
                                const Vector & advection_rhs,
                                const Vector & P,
                                Vector & solution)
{
  // The explicit terms of the momentum equations: the backward difference of the steps before, the advection, the sources and the pressure.
  const double rho = _density;
  Vector history = Vector::Zero(solution.size());
  for (int j = 1; j <= s.k; ++j)
    history += s.beta[j] * s.levels[static_cast<std::size_t>(j - 1)]->second;
  Vector rhs = -(1.0 / s.dt) * (_mass * history) - advection_rhs - rho * (_gradient * P);
  if (s.with_source)
    rhs -= _source_residual;
  solution = s.extrapolated;
  _velocity_iterations = 0;
  for (std::size_t i = 0; i < _velocity_rows.size(); ++i)
    _velocity_iterations +=
        _velocity_rows[i]->solve(*_velocity_solvers[i], _helmholtz, rhs, solution, false);
  // The residual of the momentum equations, whose rows of prescribed velocities are the reactions (the force on the boundary is minus their sum), as the monolithic solve leaves them.
  Vector residual = _helmholtz * solution - rhs;
  _problem.foldDependents(residual, nullptr);
  _problem.parallel().addAtNodes(residual.data(), _problem.numVariables());
  for (Index i = 0; i < _problem.mesh().numNodes(); ++i)
  {
    const Index d = _problem.dof(i, _pressure);
    residual[d] = 0.0;
    solution[d] = rho * P[d];
  }
  _problem.setLastResidual(residual);
}

} // namespace dualmesh
