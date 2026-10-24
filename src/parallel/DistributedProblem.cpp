// SPDX-License-Identifier: LGPL-2.1-or-later
#include "dualmesh/parallel/DistributedProblem.h"

#include "dualmesh/base/Console.h"
#include "dualmesh/linalg/PetscSolver.h"
#include "dualmesh/linalg/SaddlePointSolver.h"
#include "dualmesh/modules/Framework.h"
#include "dualmesh/parallel/Checkpoint.h"
#include "dualmesh/parallel/Partition.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <unordered_map>

#ifdef DUALMESH_HAVE_PETSC
#include <petscsf.h>
#define DM_PETSC(call) petsc::check((call), #call)
#endif

namespace dualmesh
{

DistributedProblem::DistributedProblem(const Mesh & global_mesh,
                                       Method method,
                                       CoordinateSystem coord,
                                       const DistributedOptions & options)
    : DistributedProblem(distributeMesh(global_mesh, Communicator::world(), options.partitioner),
                         method,
                         coord,
                         options)
{
}

DistributedProblem::DistributedProblem(LocalMesh part,
                                       Method method,
                                       CoordinateSystem coord,
                                       const DistributedOptions & options)
    : _comm(Communicator::world()), _options(options)
{
  if (!part.mesh || part.global_nodes.size() != static_cast<std::size_t>(part.mesh->numNodes()))
    throw InputError("The part of the mesh given to the distributed solver needs one global node "
                     "index per local node.");
  _mesh = std::move(part.mesh);
  _local_to_global = std::move(part.global_nodes);
  _global_elements = std::move(part.global_elements);
  _num_owned_elements = _mesh->numElements();
  Index largest = -1;
  for (Index g : _local_to_global)
    largest = std::max(largest, g);
  _num_global_nodes = _comm.max(largest) + 1;
  _fewest_elements = _comm.min(_num_owned_elements);
  _most_elements = _comm.max(_num_owned_elements);

  // The gather-scatter finds, from the global node numbers alone, which ranks share each node and which rank owns it.
  shareNodes();
  completeBoundaryNodes();

  _local = std::make_unique<Problem>(_mesh, method, coord);
  // The local problem's drivers solve the whole problem through this object.
  _local->setParallel(this);
  _local->setConstraintsHandledOutside(true);
  // The local mesh holds only part of each side set, so a quantity defined by the measure of a whole side set (the total_force of a traction) takes the sum over the ranks.
  std::map<std::string, double> measures;
  for (const auto & [name, sides] : _mesh->sidesets())
    measures[name] = _comm.sum(sidesetMeasure(*_mesh, {name}, coord));
  _local->setGlobalBoundaryMeasures(std::move(measures));
}

void
DistributedProblem::completeBoundaryNodes()
{
  const auto & sidesets = _mesh->sidesets();
  // The loop below is collective, so every rank must visit the same side sets in the same order.
  std::size_t signature = sidesets.size();
  for (const auto & [name, sides] : sidesets)
    signature = signature * 1000003u + std::hash<std::string>{}(name);
  const auto folded = static_cast<Index>(signature >> 2);
  if (_comm.min(folded) != _comm.max(folded))
    throw InputError("Every rank's part of a distributed mesh must list the same side set names, "
                     "also those of which it holds no side.");
  // Essential boundary conditions are prescribed node by node, so every rank that holds a node of a side set must know it, whichever rank holds the side.
  // Mesh::boundaryNodes prefers the node set of the same name, which therefore receives every such node.
  for (const auto & [name, sides] : sidesets)
  {
    std::vector<double> on_boundary(_local_to_global.size(), 0.0);
    for (Index n : _mesh->boundaryNodes(name))
      on_boundary[n] = 1.0;
    _gs.sum(on_boundary.data(), 1);
    std::vector<Index> nodes;
    for (std::size_t n = 0; n < on_boundary.size(); ++n)
      if (on_boundary[n] > 0.0)
        nodes.push_back(static_cast<Index>(n));
    _mesh->addNodeset(name, nodes);
  }
}

void
DistributedProblem::setLinearSolver(const DistributedOptions & options)
{
  _options.linear_tolerance = options.linear_tolerance;
  _options.linear_max_iterations = options.linear_max_iterations;
  _options.petsc_options = options.petsc_options;
  _options.verbose = options.verbose;
}

void
DistributedProblem::fetchInterfaceGhosts()
{
  if (_has_ghosts)
    return;
  std::vector<const InterfaceBC *> interfaces;
  for (const auto & name : _local->objectNames())
    if (const auto * ib = dynamic_cast<const InterfaceBC *>(_local->object(name).get()))
      interfaces.push_back(ib);
  if (interfaces.empty() || _comm.size() == 1)
    return;
  if (_global_elements.size() != static_cast<std::size_t>(_mesh->numElements()))
    throw InputError("An interface condition in a distributed problem needs the global numbers "
                     "of the elements of every part of the mesh.");
  // The elements fetched here follow those of the part.
  const Index first_ghost = _mesh->numElements();
  std::unordered_map<Index, Index> local_element, local_node;
  for (Index e = 0; e < first_ghost; ++e)
    local_element[_global_elements[e]] = e;
  for (std::size_t n = 0; n < _local_to_global.size(); ++n)
    local_node[_local_to_global[n]] = static_cast<Index>(n);
  // Every rank sees the same side set names in the same order, because the objects are the same on every rank.
  std::set<std::string> secondary;
  for (const auto * ib : interfaces)
    secondary.insert(ib->secondaryBoundaries().begin(), ib->secondaryBoundaries().end());
  for (const auto & name : secondary)
  {
    // Does this rank hold a primary side of an interface whose secondary side is this side set?
    bool needs = false;
    for (const auto * ib : interfaces)
      if (std::count(ib->secondaryBoundaries().begin(), ib->secondaryBoundaries().end(), name))
        for (const auto & primary : ib->boundaries())
          needs = needs || !_mesh->sideset(primary).empty();
    // Every rank shares the elements of its sides of the side set: global element, side, type, block, global nodes, and the coordinates of the nodes.
    std::vector<Index> integers;
    std::vector<double> coordinates;
    for (const Side & side : _mesh->sideset(name))
    {
      const Element & el = _mesh->element(side.first);
      integers.insert(integers.end(),
                      {_global_elements[side.first],
                       side.second,
                       static_cast<Index>(el.type),
                       el.block,
                       el.numNodes()});
      for (int k = 0; k < el.numNodes(); ++k)
      {
        integers.push_back(_local_to_global[el.nodes[k]]);
        const Point & x = _mesh->node(el.nodes[k]);
        coordinates.insert(coordinates.end(), x.begin(), x.end());
      }
    }
    const auto all_integers = _comm.allGather(integers);
    const auto all_coordinates = _comm.allGather(coordinates);
    if (!needs)
      continue;
    std::vector<Side> candidates;
    std::size_t at = 0, xyz = 0;
    while (at < all_integers.size())
    {
      const Index global_element = all_integers[at++];
      const int side = static_cast<int>(all_integers[at++]);
      const auto type = static_cast<ElementType>(all_integers[at++]);
      const int block = static_cast<int>(all_integers[at++]);
      const auto count = static_cast<std::size_t>(all_integers[at++]);
      const Index * nodes = all_integers.data() + at;
      const double * x = all_coordinates.data() + xyz;
      at += count;
      xyz += 3 * count;
      auto found = local_element.find(global_element);
      if (found != local_element.end() && found->second < _num_owned_elements)
        continue; // this rank holds the element and its side already
      if (found == local_element.end())
      {
        std::vector<Index> local(count);
        for (std::size_t k = 0; k < count; ++k)
        {
          auto [it, added] = local_node.try_emplace(nodes[k], _mesh->numNodes());
          if (added)
          {
            _mesh->addNode({x[3 * k], x[3 * k + 1], x[3 * k + 2]});
            _local_to_global.push_back(nodes[k]);
          }
          local[k] = it->second;
        }
        found = local_element.emplace(global_element, _mesh->addElement(type, local, block)).first;
        _global_elements.push_back(global_element);
      }
      candidates.push_back({found->second, side});
    }
    if (!candidates.empty())
      _local->addInterfaceCandidates(name, candidates);
  }
  _has_ghosts = _comm.any(_mesh->numElements() > first_ghost);
  if (!_has_ghosts)
    return;
  // The ghost elements are read, not integrated, and their nodes are copies of nodes of other ranks.
  _local->setGhostElements(_num_owned_elements);
  shareNodes();
}

void
DistributedProblem::shareNodes()
{
  _gs.setup(_comm, _local_to_global, GatherScatter::library(_options.gather_scatter));
  _owned_node.assign(_local_to_global.size(), 0);
  for (std::size_t i = 0; i < _local_to_global.size(); ++i)
    _owned_node[i] = _gs.owns(static_cast<Index>(i)) ? 1 : 0;
  _owns_node = _owned_node;
}

void
DistributedProblem::joinConstraints()
{
  _prescribed_elsewhere.clear();
  const auto & constraints = _local->constraints();
  if (constraints.empty())
    return;
  // Every rank gathers the nodes of the periodic boundaries of all ranks, by global index and position, so that every rank pairs them in the same way.
  // The memory this takes grows with the periodic boundaries, not with the mesh.
  const int nv = _local->numVariables();
  std::vector<std::pair<Index, Index>> pairs; // (secondary, primary) global node indices
  for (const auto & c : constraints)
  {
    const auto * periodic = dynamic_cast<const PeriodicBC *>(c.get());
    if (!periodic)
      throw InputError("'" + c->name() +
                       "': the distributed solver joins periodic constraints only.");
    if (static_cast<int>(periodic->periodicVariables(*_local).size()) != nv)
      throw InputError("'" + c->name() +
                       "': in a distributed problem a periodic boundary "
                       "condition joins every variable; leave 'variables' out.");
    const auto gather = [&](const std::string & boundary)
    {
      std::vector<Index> ids;
      std::vector<double> xyz;
      for (Index n : _local->mesh().boundaryNodes(boundary))
      {
        ids.push_back(_local_to_global[n]);
        const Point & x = _local->mesh().node(n);
        xyz.insert(xyz.end(), x.begin(), x.end());
      }
      const auto all_ids = _comm.allGather(ids);
      const auto all_xyz = _comm.allGather(xyz);
      // A node on the boundary of several parts arrives once from each; keep it once, in increasing global index.
      std::map<Index, Point> unique;
      for (std::size_t k = 0; k < all_ids.size(); ++k)
        unique.emplace(all_ids[k], Point{all_xyz[3 * k], all_xyz[3 * k + 1], all_xyz[3 * k + 2]});
      std::vector<Index> out_ids;
      std::vector<Point> out_points;
      for (const auto & [g, x] : unique)
      {
        out_ids.push_back(g);
        out_points.push_back(x);
      }
      return std::make_pair(out_ids, out_points);
    };
    const auto [primary_ids, primary_points] = gather(periodic->primaryBoundary());
    const auto [secondary_ids, secondary_points] = gather(periodic->secondaryBoundary());
    const auto matched = periodic->match(primary_points, secondary_points);
    for (std::size_t k = 0; k < secondary_ids.size(); ++k)
      if (secondary_ids[k] != primary_ids[matched[k]])
        pairs.push_back({secondary_ids[k], primary_ids[matched[k]]});
  }
  // One unknown per group of joined nodes (a union-find over global indices), represented by its smallest member that is a primary only, as in a serial problem.
  std::map<Index, Index> parent;
  std::set<Index> dependent;
  const auto root = [&](Index g)
  {
    auto it = parent.find(g);
    if (it == parent.end())
      return g;
    Index r = g;
    while (parent.count(r) && parent[r] != r)
      r = parent[r];
    parent[g] = r;
    return r;
  };
  for (const auto & [secondary, primary] : pairs)
  {
    dependent.insert(secondary);
    for (Index g : {secondary, primary})
      parent.emplace(g, g);
    const Index a = root(secondary), b = root(primary);
    if (a != b)
      parent[std::max(a, b)] = std::min(a, b);
  }
  std::map<Index, Index> representative;
  for (const auto & [g, p] : parent)
    if (!dependent.count(g) && !representative.count(root(g)))
      representative[root(g)] = g;
  std::unordered_map<Index, Index> joined; // global node -> the global node it is joined to
  for (const auto & [g, p] : parent)
  {
    const auto it = representative.find(root(g));
    if (it == representative.end())
      throw InputError("The constraints make a node a copy of itself through a loop; one node of "
                       "each loop must stay a primary.");
    if (it->second != g)
      joined[g] = it->second;
  }
  // The gather-scatter treats a joined node and its representative as one entity: the residuals of both are added, and the global numbering gives both one number.
  std::vector<Index> ids(_local_to_global);
  for (auto & g : ids)
  {
    const auto it = joined.find(g);
    if (it != joined.end())
      g = it->second;
  }
  _gs.setup(_comm, ids, GatherScatter::library(_options.gather_scatter));
  for (std::size_t i = 0; i < _local_to_global.size(); ++i)
    _owned_node[i] = _gs.owns(static_cast<Index>(i)) ? 1 : 0;
}

std::vector<char>
DistributedProblem::localPrescribed() const
{
  std::vector<char> mask(static_cast<std::size_t>(_local->numDofs()), 0);
  for (const auto & bc : _local->nodalBCs())
    for (Index n : bc->nodes())
      mask[_local->dof(n, bc->variable())] = 1;
  return mask;
}

void
DistributedProblem::agreePrescribedValues()
{
  _prescribed_elsewhere.clear();
  if (_local->constraints().empty() && !_has_ghosts)
    return;
  // A prescribed value on one node of a group of joined nodes binds every copy, on every rank.
  const auto fixed = localPrescribed();
  Vector prescribed(_local->numDofs());
  for (Index d = 0; d < _local->numDofs(); ++d)
    prescribed[d] = fixed[d] ? 1.0 : 0.0;
  addAcrossRanks(prescribed);
  _prescribed_elsewhere.assign(static_cast<std::size_t>(_local->numDofs()), 0);
  for (Index d = 0; d < _local->numDofs(); ++d)
    _prescribed_elsewhere[d] = prescribed[d] > 0.5 && !fixed[d] ? 1 : 0;
  _local->setExtraConstrainedDofs(_prescribed_elsewhere);
}

void
DistributedProblem::shareDirichletValues(Vector & U) const
{
  if (_prescribed_elsewhere.empty())
    return;
  // Each rank that prescribes a value contributes it, and the copies that do not know it take the average of the contributions (all equal when the conditions agree).
  const auto fixed = localPrescribed();
  Vector value = Vector::Zero(U.size()), count = Vector::Zero(U.size());
  for (Index d = 0; d < U.size(); ++d)
    if (fixed[d] && !_prescribed_elsewhere[d])
    {
      value[d] = U[d];
      count[d] = 1.0;
    }
  addAcrossRanks(value);
  addAcrossRanks(count);
  for (Index d = 0; d < U.size(); ++d)
    if (_prescribed_elsewhere[d] && count[d] > 0)
      U[d] = value[d] / count[d];
}

void
DistributedProblem::prepare()
{
  if (_prepared)
    return;
  fetchInterfaceGhosts();
  _local->initialize();
  joinConstraints();
  // A degree of freedom that no local element touches may still be touched by
  // an element on another rank, so the mask of active degrees of freedom has
  // to be agreed on globally before it is used to replace equations.
  Vector active(_local->numDofs());
  for (Index i = 0; i < _local->numDofs(); ++i)
    active[i] = _local->activeDofs()[i] ? 1.0 : 0.0;
  addAcrossRanks(active);
  std::vector<char> global_active(_local->numDofs(), 0);
  for (Index i = 0; i < _local->numDofs(); ++i)
    global_active[i] = active[i] > 0.5 ? 1 : 0;
  _local->overrideActiveDofs(global_active);
  // Concentrated loads are attached to a node, not to an element, so only the
  // owning rank may apply them.
  _local->setOwnedNodes(_owns_node);
  // A concentrated load given by coordinates is resolved to the nearest node,
  // and each process only sees its own part of the mesh, so every one of them
  // would find a nearest node of its own.  The process whose node is globally
  // nearest keeps the load; a tie (the point sits on a shared node) is broken
  // by the ownership mask above.
  for (const auto & load : _local->nodalLoads())
  {
    const auto & points = load->pointNodes();
    if (points.empty())
      continue;
    std::vector<double> distance;
    distance.reserve(points.size());
    for (const auto & [node, d] : points)
    {
      (void) node;
      distance.push_back(d);
    }
    std::vector<double> best = distance;
    _comm.minInPlace(best);
    std::vector<char> keep(points.size(), 0);
    for (std::size_t i = 0; i < points.size(); ++i)
      keep[i] = distance[i] <= best[i] * (1.0 + 1e-12) + 1e-12 ? 1 : 0;
    load->keepPoints(keep);
  }
  // The same holds for a value prescribed at the entity nearest to a point,
  // such as the pressure level of an enclosed flow: every process would pin a
  // node of its own, and the flow would get several pressure levels.  The
  // globally nearest node is kept, with a tie between two different nodes at
  // the same distance broken by the smaller global number, so that exactly one
  // global node is constrained; a node shared by several processes is kept by
  // all of them, which is the same degree of freedom.
  for (const auto & bc : _local->nodalBCs())
  {
    auto * point = dynamic_cast<PointDirichletBC *>(bc.get());
    if (!point)
      continue;
    const bool have = !point->nodes().empty();
    const double distance = have ? point->distance() : std::numeric_limits<double>::infinity();
    std::vector<double> best(1, distance);
    _comm.minInPlace(best);
    const bool candidate = have && distance <= best[0] * (1.0 + 1e-12) + 1e-14;
    const double id = candidate ? static_cast<double>(_local_to_global[point->nodes()[0]])
                                : std::numeric_limits<double>::infinity();
    std::vector<double> winner(1, id);
    _comm.minInPlace(winner);
    if (!candidate || id != winner[0])
      point->clear();
  }
  agreePrescribedValues();
  _num_global_dofs = _comm.sum(numOwnedDofs());
  // A global numbering of the degrees of freedom, for assembling one
  // distributed matrix: the owned ones of rank r follow those of ranks 0 to
  // r - 1, and a shared one takes its owner's number.
  {
    const std::vector<Index> owned = _comm.allGather(numOwnedDofs());
    _first_owned_dof = 0;
    for (int r = 0; r < _comm.rank(); ++r)
      _first_owned_dof += owned[r];
    Vector number = Vector::Zero(_local->numDofs());
    Index next = _first_owned_dof;
    for (Index i = 0; i < _local->mesh().numNodes(); ++i)
      if (_owned_node[i])
        for (int k = 0; k < _local->numVariables(); ++k)
          number[_local->dof(i, k)] = static_cast<double>(next++);
    _gs.copyFromOwners(number.data(), _local->numVariables());
    _global_dof.resize(_local->numDofs());
    for (Index i = 0; i < _local->numDofs(); ++i)
      _global_dof[i] = static_cast<Index>(std::llround(number[i]));
  }
  // The same numbering of the entities, for systems with one unknown per entity.
  {
    const std::vector<Index> owned = _comm.allGather(numOwnedEntities());
    _first_owned_entity = 0;
    _num_global_entities = 0;
    for (int r = 0; r < _comm.size(); ++r)
    {
      if (r < _comm.rank())
        _first_owned_entity += owned[r];
      _num_global_entities += owned[r];
    }
    std::vector<double> number(static_cast<std::size_t>(_local->mesh().numNodes()), 0.0);
    Index next = _first_owned_entity;
    for (Index i = 0; i < _local->mesh().numNodes(); ++i)
      if (_owned_node[i])
        number[i] = static_cast<double>(next++);
    _gs.copyFromOwners(number.data(), 1);
    _global_entity.resize(number.size());
    for (std::size_t i = 0; i < number.size(); ++i)
      _global_entity[i] = static_cast<Index>(std::llround(number[i]));
  }
  _prepared = true;
}

Index
DistributedProblem::numOwnedEntities() const
{
  return static_cast<Index>(std::count(_owned_node.begin(), _owned_node.end(), 1));
}

void
DistributedProblem::applyDirichlet(Vector & U, double load_factor) const
{
  _local->applyDirichlet(U, load_factor);
  shareDirichletValues(U);
  copyFromOwners(U);
}

Index
DistributedProblem::numOwnedDofs() const
{
  Index count = 0;
  const int nv = _local->numVariables();
  for (std::size_t i = 0; i < _owned_node.size(); ++i)
    if (_owned_node[i])
      count += nv;
  return count;
}

void
DistributedProblem::addAcrossRanks(Vector & v) const
{
  _gs.sum(v.data(), _local->numVariables());
}

void
DistributedProblem::copyFromOwners(Vector & v) const
{
  const int nv = _local->numVariables();
  _gs.copyFromOwners(v.data(), nv);
}

double
DistributedProblem::dot(const Vector & a, const Vector & b) const
{
  const int nv = _local->numVariables();
  double local = 0;
  for (std::size_t i = 0; i < _owned_node.size(); ++i)
  {
    if (!_owned_node[i])
      continue;
    for (int k = 0; k < nv; ++k)
    {
      const Index d = _local->dof(static_cast<Index>(i), k);
      local += a[d] * b[d];
    }
  }
  return _comm.sum(local);
}

DistributedProblem::GlobalSystem
DistributedProblem::globalSystem(const SparseMatrix & A, const Vector & b) const
{
  // The local matrix holds the contributions of this rank's elements; the global matrix is their sum, formed from the entries in the global numbering, in which joined nodes (periodic boundaries) share one number.
  // The rows of prescribed degrees of freedom have been replaced by rows of the identity on every rank that holds them, so they are entered once, by the owner.
  GlobalSystem out;
  const std::vector<char> fixed = _local->constrainedDofs();
  const int nv = _local->numVariables();
  out.owned_dof.assign(static_cast<std::size_t>(_local->numDofs()), 0);
  for (Index i = 0; i < _local->mesh().numNodes(); ++i)
    if (_owned_node[i])
      for (int k = 0; k < nv; ++k)
        out.owned_dof[_local->dof(i, k)] = 1;
  out.entries.reserve(static_cast<std::size_t>(A.nonZeros()));
  for (int col = 0; col < A.outerSize(); ++col)
    for (SparseMatrix::InnerIterator it(A, col); it; ++it)
    {
      const Index row = it.row();
      if (_global_dof[row] < 0 || _global_dof[it.col()] < 0)
        continue;
      if (fixed[row])
      {
        if (row == it.col() && out.owned_dof[row])
          out.entries.push_back({_global_dof[row], _global_dof[row], 1.0});
        continue;
      }
      out.entries.push_back({_global_dof[row], _global_dof[it.col()], it.value()});
    }
  out.b_owned = Vector::Zero(numOwnedDofs());
  for (Index i = 0; i < _local->numDofs(); ++i)
    if (out.owned_dof[i])
      out.b_owned[_global_dof[i] - _first_owned_dof] = b[i];
  return out;
}

Vector
DistributedProblem::solveWithPetsc(const GlobalSystem & system,
                                   petsc::Settings settings,
                                   int * iterations,
                                   bool must_converge) const
{
  settings.relative_tolerance = _options.linear_tolerance;
  // An attempt that may fail stops early, so that the fallback is not delayed.
  settings.max_iterations = must_converge ? _options.linear_max_iterations
                                          : std::min(_options.linear_max_iterations, 500);
  settings.block_size = _local->numVariables();
  petsc::Result result;
  const Vector x_owned = petsc::solveDistributed(
      _num_global_dofs, _first_owned_dof, system.entries, system.b_owned, settings, result);
  if (iterations)
    *iterations += result.iterations;
  if (!must_converge && _comm.any(!result.converged))
  {
    if (_options.verbose && _comm.isRoot())
      std::cout << "  PETSc (" << settings.options << "): " << result.reason
                << ", falling back to the direct solver\n";
    return Vector();
  }
  if (_comm.any(!result.converged))
    throw std::runtime_error("dualmesh: the PETSc solve did not converge (" + result.reason + ", " +
                             std::to_string(result.iterations) +
                             " iterations). Adjust petsc_options.");
  Vector x = Vector::Zero(_local->numDofs());
  for (Index i = 0; i < _local->numDofs(); ++i)
    if (system.owned_dof[i])
      x[i] = x_owned[_global_dof[i] - _first_owned_dof];
  copyFromOwners(x);
  if (_options.verbose && _comm.isRoot())
    std::cout << "  PETSc: " << result.iterations << " iterations, " << result.reason << "\n";
  return x;
}

Vector
DistributedProblem::solveSchurSplit(const SparseMatrix & A,
                                    const GlobalSystem & system,
                                    SchurApproximation approximation,
                                    int * iterations,
                                    bool must_converge) const
{
  SaddlePointBlocks blocks = _local->saddlePointBlocks(A, approximation);
  if (_comm.any(!blocks.refusal.empty()))
    throw InputError("The block preconditioners of the flow (pressure_mass, pcd, lsc) scale their "
                     "operators by the dynamic_viscosity, which must be positive everywhere.");
  // The sign of the Schur complement, from the sum over the parts of their products of the
  // coupling blocks.  It is not the product of the summed blocks, but it has its sign: each part
  // couples the pressure to the velocity with the same sign.
  const double coupling = _comm.sum(blocks.coupling);
  if (coupling == 0.0)
    throw InputError("The pressure '" + blocks.pressure_variable +
                     "' is not coupled to the other variables in this system; the block "
                     "preconditioner of the flow does not apply.");
  blocks.sign = coupling < 0.0 ? 1 : -1;
  const petsc::SchurBlocks global = petsc::globalSchurBlocks(blocks, _global_dof, system.owned_dof);
  petsc::Settings settings;
  settings.options = _options.petsc_options;
  settings.schur = &global;
  return solveWithPetsc(system, settings, iterations, must_converge);
}

std::vector<std::vector<Index>>
DistributedProblem::physicsBlockRows(const GlobalSystem & system) const
{
  const std::vector<std::vector<Index>> blocks = _local->physicsBlockDofs();
  std::vector<std::vector<Index>> rows(blocks.size());
  for (std::size_t k = 0; k < blocks.size(); ++k)
    for (Index i : blocks[k])
      if (system.owned_dof[i])
        rows[k].push_back(_global_dof[i]);
  for (auto & block : rows)
  {
    std::sort(block.begin(), block.end());
    block.erase(std::unique(block.begin(), block.end()), block.end());
  }
  return rows;
}

Eigen::MatrixXd
DistributedProblem::ownedNearNullspace(const GlobalSystem & system) const
{
  const Eigen::MatrixXd local = _local->nearNullspace();
  Eigen::MatrixXd owned = Eigen::MatrixXd::Zero(numOwnedDofs(), local.cols());
  for (Index i = 0; i < _local->numDofs(); ++i)
    if (system.owned_dof[i])
      owned.row(_global_dof[i] - _first_owned_dof) = local.row(i);
  return owned;
}

Vector
DistributedProblem::solveLinearSystem(const SparseMatrix & A,
                                      const Vector & b,
                                      const SolverOptions & o,
                                      int * iterations) const
{
  // A build without PETSc has no MPI and so one rank, whose local matrix is the whole matrix: the serial solvers apply.
  // With joined nodes (periodic boundaries) the matrix is first formed in the global numbering, as PETSc would form it.
  if (!petsc::available())
  {
    if (_local->constraints().empty())
      return _local->linearSolve(A, b, o, iterations);
    const GlobalSystem system = globalSystem(A, b);
    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(system.entries.size());
    for (const auto & e : system.entries)
      triplets.emplace_back(e.row, e.col, e.value);
    SparseMatrix global(_num_global_dofs, _num_global_dofs);
    global.setFromTriplets(triplets.begin(), triplets.end());
    const Vector x_global = _local->linearSolve(global, system.b_owned, o, iterations);
    Vector x = Vector::Zero(_local->numDofs());
    for (Index i = 0; i < _local->numDofs(); ++i)
      if (system.owned_dof[i])
        x[i] = x_global[_global_dof[i]];
    copyFromOwners(x);
    return x;
  }
  const GlobalSystem system = globalSystem(A, b);
  petsc::Settings settings;
  settings.options = _options.petsc_options;
  // The block preconditioners: the Schur complement split of a flow, and the split by physics.
  if (const auto approximation = schurApproximation(o.preconditioner))
    return solveSchurSplit(A, system, *approximation, iterations);
  if (o.preconditioner == "physics_blocks")
  {
    settings.near_nullspace = ownedNearNullspace(system);
    settings.physics_blocks = physicsBlockRows(system);
    return solveWithPetsc(system, settings, iterations);
  }
  // PETSc's smoothed aggregation, which takes the rigid body modes of a solid.  GMRES is
  // preconditioned from the right, so that it measures the true residual: the equations of a
  // coupled problem differ in scale by many orders of magnitude, and a preconditioned residual
  // can then be small while the true one is not.
  const std::string gamg =
      "-ksp_type gmres -ksp_gmres_restart 100 -ksp_pc_side right -pc_type gamg";
  if (o.preconditioner == "amg")
  {
    settings.near_nullspace = ownedNearNullspace(system);
    settings.options = gamg + " " + _options.petsc_options;
    return solveWithPetsc(system, settings, iterations);
  }
  // The user's options may choose GAMG, which takes the rigid body modes of a solid.
  if (!_options.petsc_options.empty())
  {
    if (!_local->displacementGroups().empty())
      settings.near_nullspace = ownedNearNullspace(system);
    return solveWithPetsc(system, settings, iterations);
  }
  // The automatic choice.  A flow (the methods of the nodes) is factorised by MUMPS where that is
  // cheap, as the serial automatic solver decides it, and split by the least squares commutator
  // otherwise: on a three-dimensional cavity it is about twice as fast as MUMPS from 37 000
  // Taylor-Hood unknowns on, and three to five times as fast as BoomerAMG on the whole
  // equal-order system at Re = 100 and 400.  A solid takes PETSc's smoothed aggregation with its
  // rigid body modes, a solid coupled to other physics the split by physics, and any other
  // system algebraic multigrid (hypre BoomerAMG) inside GMRES, which scales with the number of
  // processes; MUMPS when the iteration fails.
  const bool flow = _local->flowConstraint();
  if (flow || _local->hasMixedOrder())
  {
    if (flow && !_local->preferDirectSolver(_num_global_dofs))
    {
      const Vector x =
          solveSchurSplit(A, system, SchurApproximation::LeastSquaresCommutator, iterations, false);
      if (x.size() == b.size())
        return x;
    }
    settings.options = "";
    return solveWithPetsc(system, settings, iterations);
  }
  const auto groups = _local->displacementGroups();
  std::size_t displacements = 0;
  for (const auto & group : groups)
    displacements += group.size();
  petsc::Settings attempt = settings;
  if (groups.empty())
    attempt.options =
        "-ksp_type gmres -ksp_gmres_restart 100 -pc_type hypre -pc_hypre_type boomeramg";
  else
  {
    attempt.near_nullspace = ownedNearNullspace(system);
    if (displacements == static_cast<std::size_t>(_local->numVariables()))
      attempt.options = gamg;
    else
      attempt.physics_blocks = physicsBlockRows(system);
  }
  const Vector x = solveWithPetsc(system, attempt, iterations, false);
  if (x.size() == b.size())
    return x;
  settings.options = "";
  return solveWithPetsc(system, settings, iterations);
}

// ---- the Parallel of the local problem -----------------------------------------------------
void
DistributedProblem::shareDirichlet(Vector & U) const
{
  shareDirichletValues(U);
  copyFromOwners(U);
}

Vector
DistributedProblem::linearSolve(Problem &,
                                const SparseMatrix & A,
                                const Vector & b,
                                const SolverOptions & options,
                                int * iterations) const
{
  Vector x = solveLinearSystem(A, b, options, iterations);
  copyFromOwners(x);
  return x;
}

CheckpointLayout
DistributedProblem::checkpointLayout(const Problem &) const
{
  CheckpointLayout layout;
  layout.distributed = true;
  layout.global_nodes = _local_to_global;
  layout.num_global_nodes = _num_global_nodes;
  // The ghost elements belong to the parts of other processes.
  layout.global_elements.assign(
      _global_elements.begin(),
      _global_elements.begin() +
          std::min<std::ptrdiff_t>(static_cast<std::ptrdiff_t>(_global_elements.size()),
                                   static_cast<std::ptrdiff_t>(_num_owned_elements)));
  Index largest_element = -1;
  for (Index e : _global_elements)
    largest_element = std::max(largest_element, e);
  layout.num_global_elements = _comm.max(largest_element) + 1;
  return layout;
}

void
DistributedProblem::writeFields(Problem &,
                                const std::string & base,
                                const std::vector<std::string> & fields) const
{
  writeVTU(base, {}, fields);
}

// ---- the solves: the drivers of the local problem, through this object --------------------
SolveResult
DistributedProblem::solveSteady(const SolverOptions & options)
{
  prepare();
  return _local->solveSteady(options);
}

SolveResult
DistributedProblem::solveTransient(const TransientOptions & transient,
                                   const SolverOptions & options)
{
  prepare();
  return _local->solveTransient(transient, options);
}

std::vector<double>
DistributedProblem::gatheredValues(const std::string & variable) const
{
  const auto local_values = _local->values(variable);
  std::vector<double> global(_num_global_nodes, 0.0);
  for (std::size_t i = 0; i < _local_to_global.size(); ++i)
    if (_owns_node[i])
      global[_local_to_global[i]] = local_values[i];
  _comm.sumInPlace(global);
  return global;
}

void
DistributedProblem::writeVTU(const std::string & base,
                             const std::vector<std::string> & cell_properties,
                             const std::vector<std::string> & fields) const
{
  if (_comm.size() == 1)
  {
    _local->writeVTU(base + ".vtu", cell_properties, fields);
    return;
  }
  std::ostringstream piece;
  piece << base << "_" << std::setw(4) << std::setfill('0') << _comm.rank() << ".vtu";
  _local->writeVTU(piece.str(), cell_properties, fields);
  _comm.barrier();
  if (!_comm.isRoot())
    return;
  // The ".pvtu" index lists the pieces; ParaView and VisIt open it as one
  // data set.
  std::ofstream f(base + ".pvtu");
  if (!f)
    throw InputError("Cannot open '" + base + ".pvtu' for writing.");
  f << "<?xml version=\"1.0\"?>\n<VTKFile type=\"PUnstructuredGrid\" version=\"0.1\" "
       "byte_order=\"LittleEndian\">\n<PUnstructuredGrid GhostLevel=\"0\">\n";
  f << "<PPoints><PDataArray type=\"Float64\" NumberOfComponents=\"3\"/></PPoints>\n";
  f << "<PPointData>\n";
  for (int v = 0; v < _local->numVariables(); ++v)
    if (fields.empty() ||
        std::find(fields.begin(), fields.end(), _local->variable(v).name) != fields.end())
      f << "<PDataArray type=\"Float64\" Name=\"" << _local->variable(v).name << "\"/>\n";
  f << "</PPointData>\n<PCellData>\n<PDataArray type=\"Int32\" Name=\"block\"/>\n";
  for (const auto & p : cell_properties)
    f << "<PDataArray type=\"Float64\" Name=\"" << p << "\"/>\n";
  f << "</PCellData>\n";
  for (int r = 0; r < _comm.size(); ++r)
  {
    std::ostringstream name;
    name << base << "_" << std::setw(4) << std::setfill('0') << r << ".vtu";
    // Strip any directory, because the index and the pieces sit side by side.
    std::string file = name.str();
    const auto slash = file.find_last_of('/');
    if (slash != std::string::npos)
      file = file.substr(slash + 1);
    f << "<Piece Source=\"" << file << "\"/>\n";
  }
  f << "</PUnstructuredGrid>\n</VTKFile>\n";
}

double
DistributedProblem::totalReaction(const std::string & variable, const std::string & boundary) const
{
  double local = 0;
  for (const auto & [node, reaction] : _local->reactions(variable, boundary))
    if (_owns_node[node])
      local += reaction;
  return _comm.sum(local);
}

std::string
DistributedProblem::summary() const
{
  std::ostringstream os;
  os << "distributed problem: " << _comm.size() << " rank" << (_comm.size() == 1 ? "" : "s")
     << ", partitioner " << _options.partitioner << ", gather-scatter " << _options.gather_scatter
     << ", elements per rank " << _fewest_elements << " to " << _most_elements << ", rank "
     << _comm.rank() << " holds " << _num_owned_elements << " elements";
  if (_mesh->numElements() > _num_owned_elements)
    os << " and " << _mesh->numElements() - _num_owned_elements << " ghost elements";
  os << ", and " << _owned_node.size() << " nodes"
     << " of which " << std::count(_owned_node.begin(), _owned_node.end(), 1) << " are owned";
  return os.str();
}

} // namespace dualmesh
