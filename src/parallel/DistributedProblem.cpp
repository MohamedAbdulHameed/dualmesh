// SPDX-License-Identifier: LGPL-2.1-or-later
#include "dualmesh/parallel/DistributedProblem.h"

#include "dualmesh/base/Checkpoint.h"
#include "dualmesh/base/Console.h"
#include "dualmesh/linalg/PetscSolver.h"
#include "dualmesh/parallel/Partition.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <map>
#include <set>
#include <sstream>
#include <unordered_map>

#include <petscsf.h>
#define DM_PETSC(call) petsc::check((call), #call)

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
  // The local mesh holds only part of each side set, so a quantity defined by the area of a whole side set (the total_force of a traction) takes the sum over the ranks.
  std::map<std::string, double> measures;
  for (const auto & [name, sides] : _mesh->sidesets())
    measures[name] = _comm.sum(sidesetMeasure(*_mesh, {name}, coord));
  _local->setGlobalBoundaryAreas(std::move(measures));
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

namespace
{
/// The elements of the sides of a side set as a rank shares them with the others.  For each side,
/// integers holds the global element, the side, the type, the block, the number of nodes and the
/// global nodes, and coordinates holds the three coordinates of each node.
struct SharedSides
{
  std::vector<Index> integers;
  std::vector<double> coordinates;
};

/// One side read back from SharedSides, with its nodes and their coordinates in the buffers.
struct SharedSide
{
  Index global_element;
  int side;
  ElementType type;
  int block;
  /// The number of nodes of the element.
  std::size_t count;
  /// The global nodes of the element.
  const Index * nodes;
  /// The coordinates of the nodes, three per node.
  const double * x;
};

/// Pack the elements of the given sides of mesh, whose global element and node numbers are
/// global_elements and local_to_global.
SharedSides
packSides(const Mesh & mesh,
          const std::vector<Side> & sides,
          const std::vector<Index> & global_elements,
          const std::vector<Index> & local_to_global)
{
  SharedSides out;
  for (const Side & side : sides)
  {
    const Element & el = mesh.element(side.first);
    out.integers.insert(out.integers.end(),
                        {global_elements[side.first],
                         side.second,
                         static_cast<Index>(el.type),
                         el.block,
                         el.numNodes()});
    for (int k = 0; k < el.numNodes(); ++k)
    {
      out.integers.push_back(local_to_global[el.nodes[k]]);
      const Point & x = mesh.node(el.nodes[k]);
      out.coordinates.insert(out.coordinates.end(), x.begin(), x.end());
    }
  }
  return out;
}

/// Read the side that starts at position at of all.integers and xyz of all.coordinates, and move
/// both positions past it.
SharedSide
unpackSide(const SharedSides & all, std::size_t & at, std::size_t & xyz)
{
  SharedSide out;
  out.global_element = all.integers[at++];
  out.side = static_cast<int>(all.integers[at++]);
  out.type = static_cast<ElementType>(all.integers[at++]);
  out.block = static_cast<int>(all.integers[at++]);
  out.count = static_cast<std::size_t>(all.integers[at++]);
  out.nodes = all.integers.data() + at;
  out.x = all.coordinates.data() + xyz;
  at += out.count;
  xyz += 3 * out.count;
  return out;
}

/// Add the element of a shared side to mesh, with those of its nodes that mesh lacks, and return
/// its local number.  local_node maps the global node numbers to the local ones, and
/// local_to_global maps them back.  Both receive the added nodes.
Index
addGhostElement(const SharedSide & shared,
                Mesh & mesh,
                std::unordered_map<Index, Index> & local_node,
                std::vector<Index> & local_to_global)
{
  std::vector<Index> local(shared.count);
  for (std::size_t k = 0; k < shared.count; ++k)
  {
    auto [it, added] = local_node.try_emplace(shared.nodes[k], mesh.numNodes());
    if (added)
    {
      mesh.addNode({shared.x[3 * k], shared.x[3 * k + 1], shared.x[3 * k + 2]});
      local_to_global.push_back(shared.nodes[k]);
    }
    local[k] = it->second;
  }
  return mesh.addElement(shared.type, local, shared.block);
}
} // namespace

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
    // Every rank shares the elements of its sides of the side set.
    const SharedSides own =
        packSides(*_mesh, _mesh->sideset(name), _global_elements, _local_to_global);
    const SharedSides all{_comm.allGather(own.integers), _comm.allGather(own.coordinates)};
    if (!needs)
      continue;
    std::vector<Side> candidates;
    std::size_t at = 0, xyz = 0;
    while (at < all.integers.size())
    {
      const SharedSide shared = unpackSide(all, at, xyz);
      auto found = local_element.find(shared.global_element);
      if (found != local_element.end() && found->second < _num_owned_elements)
        continue; // This rank holds the element and its side already
      if (found == local_element.end())
      {
        found = local_element
                    .emplace(shared.global_element,
                             addGhostElement(shared, *_mesh, local_node, _local_to_global))
                    .first;
        _global_elements.push_back(shared.global_element);
      }
      candidates.push_back({found->second, shared.side});
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
    const auto * periodic = dynamic_cast<const PeriodicConstraint *>(c.get());
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
  // One unknown per group of joined nodes, represented as in a serial problem.
  const std::unordered_map<Index, Index> joined = joinedRepresentatives(pairs);
  // Under FEM a hanging node takes its value from the side it lies on, which this process may not
  // hold for every copy of the node, so the copies of a periodic group cannot hang.
  std::set<Index> members;
  for (const auto & [g, r] : joined)
    members.insert({g, r});
  bool hangs_and_joins = false;
  if (!_local->isDual())
    for (const auto & c : _local->mesh().nodeConstraints())
      hangs_and_joins =
          hangs_and_joins || members.count(_local_to_global[static_cast<std::size_t>(c.node)]);
  if (_comm.any(hangs_and_joins))
    throw InputError("A distributed run under FEM takes no node that hangs on a periodic "
                     "boundary: the gather-scatter joins the copies of a periodic node, and the "
                     "process of one copy does not hold the side another copy hangs on. Run it "
                     "on one process, or refine the mesh alike at both periodic boundaries "
                     "without leaving a node hanging there.");
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

void
DistributedProblem::agreePrescribedValues()
{
  _prescribed_elsewhere.clear();
  if (_local->constraints().empty() && !_has_ghosts)
    return;
  // A prescribed value on one node of a group of joined nodes binds every copy, on every rank.
  const auto fixed = _local->prescribedDofs();
  Vector prescribed(_local->numDofs());
  for (Index d = 0; d < _local->numDofs(); ++d)
    prescribed[d] = fixed[d] ? 1.0 : 0.0;
  addAtNodes(prescribed.data(), _local->numVariables());
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
  const auto fixed = _local->prescribedDofs();
  Vector value = Vector::Zero(U.size()), count = Vector::Zero(U.size());
  for (Index d = 0; d < U.size(); ++d)
    if (fixed[d] && !_prescribed_elsewhere[d])
    {
      value[d] = U[d];
      count[d] = 1.0;
    }
  addAtNodes(value.data(), _local->numVariables());
  addAtNodes(count.data(), _local->numVariables());
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
  addAtNodes(active.data(), _local->numVariables());
  std::vector<char> global_active(_local->numDofs(), 0);
  for (Index i = 0; i < _local->numDofs(); ++i)
    global_active[i] = active[i] > 0.5 ? 1 : 0;
  _local->overrideActiveDofs(global_active);
  // Concentrated loads are attached to a node, not to an element, so only the
  // owning rank may apply them.
  _local->setOwnedNodes(_owns_node);
  // A concentrated load and a value given at a point resolve to the node nearest the point over
  // every process (Problem::nearestNode).
  agreePrescribedValues();
  // A global numbering of the degrees of freedom, for assembling one distributed matrix.
  _global_dof =
      globalNumbers(_owned_node, _local->numVariables(), _first_owned_dof, _num_global_dofs);
  _prepared = true;
}

std::vector<Index>
DistributedProblem::globalNumbers(const std::vector<char> & owned,
                                  int width,
                                  Index & first,
                                  Index & total) const
{
  const Index count = static_cast<Index>(std::count(owned.begin(), owned.end(), 1)) * width;
  const std::vector<Index> counts = _comm.allGather(count);
  first = 0;
  total = 0;
  for (int r = 0; r < _comm.size(); ++r)
  {
    if (r < _comm.rank())
      first += counts[static_cast<std::size_t>(r)];
    total += counts[static_cast<std::size_t>(r)];
  }
  std::vector<double> number(owned.size() * static_cast<std::size_t>(width), -1.0);
  Index next = first;
  for (std::size_t i = 0; i < owned.size(); ++i)
    if (owned[i])
      for (int k = 0; k < width; ++k)
        number[i * static_cast<std::size_t>(width) + static_cast<std::size_t>(k)] =
            static_cast<double>(next++);
  _gs.copyFromOwners(number.data(), width);
  std::vector<Index> out(number.size());
  for (std::size_t i = 0; i < number.size(); ++i)
    out[i] = static_cast<Index>(std::llround(number[i]));
  return out;
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

GlobalRows
DistributedProblem::globalRows(const Problem &) const
{
  // Each rank numbers the unknowns of the nodes it owns contiguously from _first_owned_dof, and
  // joined nodes (periodic boundaries) share one number.
  GlobalRows rows;
  rows.distributed = true;
  rows.global_size = _num_global_dofs;
  rows.first_row = _first_owned_dof;
  rows.num_owned = numOwnedDofs();
  rows.global_dof = _global_dof;
  const int nv = _local->numVariables();
  rows.owned_dof.assign(static_cast<std::size_t>(_local->numDofs()), 0);
  for (Index i = 0; i < _local->mesh().numNodes(); ++i)
    if (_owned_node[i])
      for (int k = 0; k < nv; ++k)
        rows.owned_dof[_local->dof(i, k)] = 1;
  return rows;
}

// ---- the Parallel of the local problem -----------------------------------------------------
void
DistributedProblem::shareDirichlet(Vector & U) const
{
  shareDirichletValues(U);
  copyFromOwners(U.data(), _local->numVariables());
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
  return gatheredNodeValues(_local->values(variable), 1);
}

std::vector<double>
DistributedProblem::gatheredNodeValues(const std::vector<double> & local, int width) const
{
  const std::size_t w = static_cast<std::size_t>(width);
  if (local.size() != _local_to_global.size() * w)
    throw std::logic_error("dualmesh: the values are not of the local nodes.");
  std::vector<double> global(static_cast<std::size_t>(_num_global_nodes) * w, 0.0);
  for (std::size_t i = 0; i < _local_to_global.size(); ++i)
    if (_owns_node[i])
      for (std::size_t c = 0; c < w; ++c)
        global[static_cast<std::size_t>(_local_to_global[i]) * w + c] = local[i * w + c];
  _comm.sumInPlace(global);
  return global;
}

std::vector<double>
DistributedProblem::localNodeValues(const std::vector<double> & global, int width) const
{
  const std::size_t w = static_cast<std::size_t>(width);
  std::vector<double> local(_local_to_global.size() * w);
  for (std::size_t i = 0; i < _local_to_global.size(); ++i)
    for (std::size_t c = 0; c < w; ++c)
      local[i * w + c] = global.at(static_cast<std::size_t>(_local_to_global[i]) * w + c);
  return local;
}

std::vector<double>
DistributedProblem::gatheredElementValues(const std::vector<double> & local) const
{
  if (_global_elements.size() < static_cast<std::size_t>(_num_owned_elements) ||
      local.size() < static_cast<std::size_t>(_num_owned_elements))
    throw std::logic_error("dualmesh: the part has no global numbers of its elements.");
  Index largest = -1;
  for (Index e = 0; e < _num_owned_elements; ++e)
    largest = std::max(largest, _global_elements[static_cast<std::size_t>(e)]);
  std::vector<double> global(static_cast<std::size_t>(_comm.max(largest) + 1), 0.0);
  for (Index e = 0; e < _num_owned_elements; ++e)
    global[static_cast<std::size_t>(_global_elements[static_cast<std::size_t>(e)])] =
        local[static_cast<std::size_t>(e)];
  _comm.sumInPlace(global);
  return global;
}

void
DistributedProblem::setGatheredValues(const std::string & variable,
                                      const std::vector<double> & global)
{
  _local->setValues(variable, localNodeValues(global, 1));
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
DistributedProblem::totalOutflow(const std::string & variable, const std::string & boundary) const
{
  // A node on the boundary of two parts has the reaction of its whole equation on both, and the
  // integral of the conditions over the sides of each part: the reaction counts on its owner,
  // the integral on every part.
  double local = 0;
  for (const auto & [node, flow] : _local->boundaryOutflow(variable, boundary, &_owns_node))
    local += flow;
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
