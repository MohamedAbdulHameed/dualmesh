// SPDX-License-Identifier: LGPL-2.1-or-later
#include "dualmesh/parallel/DistributedProblem.h"
#include "dualmesh/base/Console.h"
#include "dualmesh/linalg/PetscSolver.h"
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

namespace
{
/// For every local cell of a distributed mesh, given by its global element number @p ids: the sides that are faces of the boundary of the domain (bit s for side s), and the number of such faces of all elements with smaller numbers.
/// Only the process that integrates a cell (@p integrated) sees all its neighbors and so knows its boundary faces (@p mask); every copy of the cell receives that process's answer.
/// The offsets follow the global element numbers, so they do not depend on the partition.
/// Returns the total number of boundary faces.
Index
boundaryFaceOffsets(const Communicator & comm,
                    const std::vector<Index> & ids,
                    Index integrated,
                    std::vector<Index> & mask,
                    std::vector<Index> & offset)
{
  const std::size_t n = ids.size();
  const auto faces = [](Index m)
  {
    Index count = 0;
    for (; m; m &= m - 1)
      ++count;
    return count;
  };
  offset.assign(n, 0);
#ifdef DUALMESH_HAVE_PETSC
  // Each element number has a home process in an even layout of the numbers; the integrating process tells the home the sides, the homes number the faces in the order of the element numbers, and every copy of a cell learns the answer from its home.
  petsc::initialize();
  Index largest = -1;
  for (Index g : ids)
    largest = std::max(largest, g);
  PetscLayout layout;
  DM_PETSC(PetscLayoutCreate(PETSC_COMM_WORLD, &layout));
  DM_PETSC(PetscLayoutSetSize(layout, static_cast<PetscInt>(comm.max(largest) + 1)));
  DM_PETSC(PetscLayoutSetUp(layout));
  PetscInt home_count = 0;
  DM_PETSC(PetscLayoutGetLocalSize(layout, &home_count));
  PetscSF home;
  DM_PETSC(PetscSFCreate(PETSC_COMM_WORLD, &home));
  std::vector<PetscInt> remote(ids.begin(), ids.end());
  DM_PETSC(PetscSFSetGraphLayout(
      home, layout, static_cast<PetscInt>(n), nullptr, PETSC_COPY_VALUES, remote.data()));
  DM_PETSC(PetscLayoutDestroy(&layout));
  std::vector<PetscInt> leaf(n, 0), root(static_cast<std::size_t>(home_count), 0);
  for (std::size_t c = 0; c < n; ++c)
    leaf[c] = static_cast<Index>(c) < integrated ? static_cast<PetscInt>(mask[c]) : 0;
  DM_PETSC(PetscSFReduceBegin(home, MPIU_INT, leaf.data(), root.data(), MPI_SUM));
  DM_PETSC(PetscSFReduceEnd(home, MPIU_INT, leaf.data(), root.data(), MPI_SUM));
  DM_PETSC(PetscSFBcastBegin(home, MPIU_INT, root.data(), leaf.data(), MPI_REPLACE));
  DM_PETSC(PetscSFBcastEnd(home, MPIU_INT, root.data(), leaf.data(), MPI_REPLACE));
  for (std::size_t c = 0; c < n; ++c)
    mask[c] = leaf[c];
  Index running = 0;
  for (auto & r : root)
  {
    const Index count = faces(r);
    r = static_cast<PetscInt>(running);
    running += count;
  }
  const std::vector<Index> totals = comm.allGather(running);
  Index before = 0, total = 0;
  for (int r = 0; r < comm.size(); ++r)
  {
    if (r < comm.rank())
      before += totals[r];
    total += totals[r];
  }
  for (auto & r : root)
    r += static_cast<PetscInt>(before);
  DM_PETSC(PetscSFBcastBegin(home, MPIU_INT, root.data(), leaf.data(), MPI_REPLACE));
  DM_PETSC(PetscSFBcastEnd(home, MPIU_INT, root.data(), leaf.data(), MPI_REPLACE));
  DM_PETSC(PetscSFDestroy(&home));
  for (std::size_t c = 0; c < n; ++c)
    offset[c] = leaf[c];
  return total;
#else
  // Without MPI one process integrates every cell.
  (void) comm;
  (void) integrated;
  std::map<Index, std::size_t> order;
  for (std::size_t c = 0; c < n; ++c)
    order[ids[c]] = c;
  Index running = 0;
  for (const auto & [id, c] : order)
  {
    offset[c] = running;
    running += faces(mask[c]);
  }
  return running;
#endif
}
} // namespace

DistributedProblem::DistributedProblem(const Mesh & global_mesh,
                                       Method method,
                                       CoordinateSystem coord,
                                       const DistributedOptions & options)
    : DistributedProblem(distributeMesh(global_mesh,
                                        Communicator::world(),
                                        options.partitioner,
                                        method == Method::FiniteVolumeCell ? 2 : 0),
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
  // The cell-centered method integrates the faces of its cells, and the flux through a face between two parts needs the gradients of the cells on both sides, which are reconstructed from their neighbors: so every part needs the cells of other parts up to two layers away.
  _cell_centered = method == Method::FiniteVolumeCell;
  if (_cell_centered && _comm.any(part.ghost_layers < 2) && _comm.size() > 1)
    throw InputError("The cell-centered finite volume method (zfvm) needs two layers of ghost "
                     "cells around every part of a distributed mesh: build the parts with "
                     "distributeMesh(mesh, comm, partitioner, 2).");
  if (part.num_ghost_elements > 0 &&
      part.global_elements.size() != static_cast<std::size_t>(part.mesh->numElements()))
    throw InputError("A part of a mesh with ghost elements needs the global number of every "
                     "element.");
  _mesh = std::move(part.mesh);
  _local_to_global = std::move(part.global_nodes);
  _global_elements = std::move(part.global_elements);
  _num_owned_elements = _mesh->numElements() - part.num_ghost_elements;
  Index largest = -1;
  for (Index g : _local_to_global)
    largest = std::max(largest, g);
  _num_global_nodes = _comm.max(largest) + 1;
  _fewest_elements = _comm.min(_num_owned_elements);
  _most_elements = _comm.max(_num_owned_elements);

  // The gather-scatter finds, from the global node numbers alone, which ranks share each node and which rank owns it.
  // The unknowns of the cell-centered method sit at cells and faces instead, which are numbered once the cells are known (numberCellEntities).
  if (!_cell_centered)
  {
    _gs.setup(_comm, _local_to_global, GatherScatter::library(_options.gather_scatter));
    _owned_node.assign(_local_to_global.size(), 0);
    for (std::size_t i = 0; i < _local_to_global.size(); ++i)
      _owned_node[i] = _gs.owns(static_cast<Index>(i)) ? 1 : 0;
    completeBoundaryNodes();
    _owns_node = _owned_node;
  }

  _local = std::make_unique<Problem>(_mesh, method, coord);
  _local->setConstraintsHandledOutside(true);
  if (part.num_ghost_elements > 0)
    _local->setGhostElements(_num_owned_elements);
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
  // The elements fetched here follow those of the part and its ghost layers.
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
  if (_cell_centered)
    return;
  _gs.setup(_comm, _local_to_global, GatherScatter::library(_options.gather_scatter));
  _owned_node.assign(_local_to_global.size(), 0);
  for (std::size_t i = 0; i < _local_to_global.size(); ++i)
    _owned_node[i] = _gs.owns(static_cast<Index>(i)) ? 1 : 0;
  _owns_node = _owned_node;
}

void
DistributedProblem::numberCellEntities()
{
  // The unknowns of the cell-centered method sit at the cells and at the faces of the boundary of the domain.
  // A cell is numbered by its element; the faces follow all cells, in the order of the elements they belong to and of their sides, as in a serial problem, so that the numbering does not depend on the partition.
  // The outer faces of the outermost ghost cells are no faces of the mesh: only this process has them, and they get no number.
  const CellMesh & cm = _local->cellMesh();
  const Index cells = cm.numCells();
  std::vector<Index> element_ids = _global_elements;
  if (element_ids.empty())
    for (Index c = 0; c < cells; ++c)
      element_ids.push_back(c);
  std::vector<Index> mask(static_cast<std::size_t>(cells), 0), offset;
  for (Index c = 0; c < _num_owned_elements; ++c)
    for (int fi : cm.cellFaces(c))
      if (cm.faces()[fi].neighbor < 0)
        mask[c] |= Index(1) << cm.faces()[fi].side.second;
  const Index boundary_faces =
      boundaryFaceOffsets(_comm, element_ids, _num_owned_elements, mask, offset);
  Index largest = -1;
  for (Index g : element_ids)
    largest = std::max(largest, g);
  const Index num_elements = _comm.max(largest) + 1;
  _local_to_global.assign(static_cast<std::size_t>(cm.numEntities()), -1);
  std::vector<char> claims(static_cast<std::size_t>(cm.numEntities()), 0);
  for (Index c = 0; c < cells; ++c)
  {
    _local_to_global[c] = element_ids[c];
    claims[c] = c < _num_owned_elements ? 1 : 0;
  }
  for (Index b = cells; b < cm.numEntities(); ++b)
  {
    const CellFace & f = cm.faces()[cm.faceOfBoundaryEntity(b)];
    const int side = f.side.second;
    const Index sides = mask[f.owner];
    if (!((sides >> side) & 1))
      continue;
    Index before = 0;
    for (Index m = sides & ((Index(1) << side) - 1); m; m &= m - 1)
      ++before;
    _local_to_global[b] = num_elements + offset[f.owner] + before;
    claims[b] = f.owner < _num_owned_elements ? 1 : 0;
  }
  _num_global_nodes = num_elements + boundary_faces;
  // The process that integrates a cell owns it and its boundary faces.
  _gs.setup(_comm, _local_to_global, GatherScatter::library(_options.gather_scatter), claims);
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
      for (Index n : _local->boundaryEntities(boundary))
      {
        ids.push_back(_local_to_global[n]);
        const Point & x = _local->entityPoint(n);
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
  if (_local->constraints().empty() && !_has_ghosts && !_cell_centered)
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
  if (_cell_centered && !_global_elements.empty())
    _local->setGlobalElementNumbers(_global_elements);
  _local->initialize();
  if (_cell_centered)
    numberCellEntities();
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
  const int nv = _local->numVariables();
  std::vector<char> owned_entities(_local->numEntities(), 1);
  for (Index i = 0; i < _local->numEntities(); ++i)
    owned_entities[i] = _owns_node[i];
  _local->setOwnedEntities(owned_entities);
  (void) nv;
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
    for (Index i = 0; i < _local->numEntities(); ++i)
      if (_owned_node[i])
        for (int k = 0; k < _local->numVariables(); ++k)
          number[_local->dof(i, k)] = static_cast<double>(next++);
    _gs.copyFromOwners(number.data(), _local->numVariables());
    _global_dof.resize(_local->numDofs());
    for (Index i = 0; i < _local->numDofs(); ++i)
      _global_dof[i] = static_cast<Index>(std::llround(number[i]));
    // An entity that only this process has is in no system.
    for (Index i = 0; i < _local->numEntities(); ++i)
      if (_local_to_global[i] < 0)
        for (int k = 0; k < _local->numVariables(); ++k)
          _global_dof[_local->dof(i, k)] = -1;
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
    std::vector<double> number(static_cast<std::size_t>(_local->numEntities()), 0.0);
    Index next = _first_owned_entity;
    for (Index i = 0; i < _local->numEntities(); ++i)
      if (_owned_node[i])
        number[i] = static_cast<double>(next++);
    _gs.copyFromOwners(number.data(), 1);
    _global_entity.resize(number.size());
    for (std::size_t i = 0; i < number.size(); ++i)
      _global_entity[i] =
          _local_to_global[i] < 0 ? -1 : static_cast<Index>(std::llround(number[i]));
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
  if (!_cell_centered)
    return;
  // The outer faces of the outermost ghost cells take the values of their cells, so that every local value lies among the values of the cells.
  const CellMesh & cm = _local->cellMesh();
  for (Index b = cm.numCells(); b < cm.numEntities(); ++b)
    if (_local_to_global[b] < 0)
    {
      const Index c = cm.faces()[cm.faceOfBoundaryEntity(b)].owner;
      for (int k = 0; k < nv; ++k)
        v[_local->dof(b, k)] = v[_local->dof(c, k)];
    }
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
  for (Index i = 0; i < _local->numEntities(); ++i)
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
DistributedProblem::solveWithPetsc(const SparseMatrix & A,
                                   const Vector & b,
                                   const std::string & options,
                                   int * iterations,
                                   bool must_converge) const
{
  const GlobalSystem system = globalSystem(A, b);
  const auto & entries = system.entries;
  const auto & b_owned = system.b_owned;
  const auto & owned_dof = system.owned_dof;
  const int nv = _local->numVariables();

  petsc::Settings settings;
  settings.options = options;
  settings.relative_tolerance = _options.linear_tolerance;
  // An attempt that may fail stops early, so that the fallback is not delayed.
  settings.max_iterations = must_converge ? _options.linear_max_iterations
                                          : std::min(_options.linear_max_iterations, 500);
  settings.block_size = nv;
  petsc::Result result;
  const Vector x_owned = petsc::solveDistributed(
      _num_global_dofs, _first_owned_dof, entries, b_owned, settings, result);
  if (iterations)
    *iterations += result.iterations;
  if (!must_converge && _comm.any(!result.converged))
  {
    if (_options.verbose && _comm.isRoot())
      std::cout << "  PETSc (" << options << "): " << result.reason
                << ", falling back to the direct solver\n";
    return Vector();
  }
  if (_comm.any(!result.converged))
    throw std::runtime_error("dualmesh: the PETSc solve did not converge (" + result.reason + ", " +
                             std::to_string(result.iterations) +
                             " iterations). Adjust petsc_options.");
  Vector x = Vector::Zero(_local->numDofs());
  for (Index i = 0; i < _local->numDofs(); ++i)
    if (owned_dof[i])
      x[i] = x_owned[_global_dof[i] - _first_owned_dof];
  copyFromOwners(x);
  if (_options.verbose && _comm.isRoot())
    std::cout << "  PETSc: " << result.iterations << " iterations, " << result.reason << "\n";
  return x;
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
  if (!_options.petsc_options.empty())
    return solveWithPetsc(A, b, _options.petsc_options, iterations);
  // The automatic choice, as the serial "automatic" solver makes it: algebraic multigrid (hypre BoomerAMG) inside GMRES, which scales with the number of processes, and a parallel direct factorisation (the PETSc default, MUMPS) when multigrid fails or cannot apply.
  // A Taylor-Hood system has a zero pressure block, on which multigrid cannot apply.
  if (!_local->hasMixedOrder())
  {
    const Vector x = solveWithPetsc(A,
                                    b,
                                    "-ksp_type gmres -ksp_gmres_restart 100 -pc_type hypre "
                                    "-pc_hypre_type boomeramg",
                                    iterations,
                                    false);
    if (x.size() == b.size())
      return x;
  }
  return solveWithPetsc(A, b, "", iterations);
}

SolveResult
DistributedProblem::nonlinearSolve(const SolverOptions & o,
                                   Problem::AssemblyOptions base,
                                   const Vector * steady_old)
{
  prepare();
  const bool picard = o.nonlinear_solver == "picard";
  base.mode = picard ? LinearizationMode::Picard : LinearizationMode::Newton;
  std::vector<double> factors = o.load_factors;
  if (factors.empty())
    factors = {1.0};

  SolveResult result;
  Vector R;
  SparseMatrix J;
  Vector & U = _local->solution();
  int step_index = 0;
  for (double lf : factors)
  {
    ++step_index;
    base.load_factor = lf;
    _local->applyDirichlet(U, lf);
    shareDirichletValues(U);
    copyFromOwners(U);
    const bool print = o.verbose && _comm.isRoot();
    if (print)
      console::nonlinearHeader(
          picard ? "Picard iteration" : "Newton", step_index, static_cast<int>(factors.size()), lf);
    int linear_before = result.linear_iterations;
    double r0 = -1;
    bool converged = false;
    bool diverged = false;
    for (int it = 1; it <= o.max_iterations + 1; ++it)
    {
      Vector lagU = U;
      base.lagged = &lagU;
      _local->assemble(U, base, R, &J);
      if (steady_old)
        R += *steady_old;
      addAcrossRanks(R);
      _local->setLastResidual(R);
      _local->dirichletRows(U, lf, R, &J);
      const double rn = norm(R);
      // An iterate that is no longer finite has diverged; going on would hand the linear solver a matrix of infinities.
      if (!std::isfinite(rn) || _comm.any(!J.coeffs().allFinite()))
      {
        diverged = true;
        break;
      }
      if (r0 < 0)
        r0 = rn;
      IterationRecord record{step_index, lf, it, rn, 0.0};
      if (rn <= o.absolute_tolerance || (it > 1 && rn <= o.relative_tolerance * r0) ||
          (o.nonlinear_solver == "linear" && it > 1))
      {
        converged = true;
        result.history.push_back(record);
        if (print)
          console::nonlinearIteration(it, rn, r0, true);
        break;
      }
      if (it > o.max_iterations)
      {
        result.history.push_back(record);
        break;
      }
      Vector delta = solveLinearSystem(J, -R, o, &result.linear_iterations);
      copyFromOwners(delta);
      Vector Unew = U + delta;
      if (o.relaxation > 0 && it > 1)
        Unew = (1.0 - o.relaxation) * Unew + o.relaxation * U;
      const double un = norm(Unew);
      record.step_norm = norm(Vector(Unew - U)) / (un > 0 ? un : 1.0);
      U = Unew;
      _local->updateDependentDofs(U);
      result.history.push_back(record);
      ++result.total_iterations;
      if (print)
        console::nonlinearIteration(
            it, rn, r0, false, record.step_norm, result.linear_iterations - linear_before);
      linear_before = result.linear_iterations;
      if (o.nonlinear_solver == "linear" || (it > 1 && record.step_norm <= o.step_tolerance))
      {
        converged = true;
        break;
      }
    }
    if (!converged)
    {
      result.converged = false;
      if (o.error_on_divergence)
      {
        std::ostringstream os;
        if (diverged)
          os << "dualmesh: the distributed nonlinear solve stopped at load factor " << lf
             << ": the residual or the Jacobian has NaN or infinite entries. Either Newton's "
                "method diverged (take smaller load steps, start from a closer state, or refine "
                "the mesh), or a property or its derivative is not finite at the current "
                "solution (for example log(0), 1/0, or x^p with p < 1 at x = 0).";
        else
          os << "dualmesh: the distributed nonlinear solve did not converge at load factor " << lf
             << " within " << o.max_iterations << " iterations.";
        throw std::runtime_error(os.str());
      }
      return result;
    }
  }
  result.converged = true;
  return result;
}

SolveResult
DistributedProblem::solveSteady(const SolverOptions & options)
{
  prepare();
  Problem::AssemblyOptions base;
  base.include_time_kernels = false;
  base.theta = 1.0;
  return nonlinearSolve(options, base, nullptr);
}

SolveResult
DistributedProblem::solveTransient(const TransientOptions & transient,
                                   const SolverOptions & options)
{
  prepare();
  if (transient.dt <= 0)
    throw InputError("Transient: the time step must be positive.");
  SolverOptions attempt = options;
  attempt.error_on_divergence = false;
  Vector & U = _local->solution();
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
  // A restart continues from the time, the step size and the state of its checkpoint.
  TransientOptions tr = transient;
  int first_step = 0;
  if (!tr.restart_file.empty())
  {
    const CheckpointTime point = readCheckpoint(tr.restart_file, *_local, layout);
    tr.start_time = point.time;
    tr.dt = point.dt > 0 ? point.dt : tr.dt;
    first_step = point.step;
  }
  _local->setTime(tr.start_time);
  _local->applyDirichlet(U, 1.0);
  shareDirichletValues(U);
  copyFromOwners(U);
  double time = tr.start_time;
  Vector committed = U;
  const auto & integrator = _local->timeIntegrator();
  if (integrator)
    integrator->start(U, time);
  const auto write = [&](int index)
  {
    if (tr.output_file_base.empty())
      return;
    std::ostringstream os;
    os << tr.output_file_base << "_" << std::setw(5) << std::setfill('0') << index;
    writeVTU(os.str(), {}, tr.output_fields);
  };
  // Every rank runs the same controller on the same numbers, so they all take
  // the same steps and accept or reject together: the residual norms that the
  // decision rests on are global reductions, and the error estimate is formed
  // from the global norm as well.
  return runTransient(
      tr,
      options,
      U,
      time,
      [&](const Vector & old, double dt)
      {
        _local->setTime(time);
        if (integrator)
          return integrator->step(U, old, time, dt);
        // The history of the material follows the solution, as in a serial run.
        if (_local->hasState())
        {
          const bool from_committed =
              old.size() == committed.size() && (old.array() == committed.array()).all();
          if (from_committed)
            _local->startStateFromCommitted();
          else
            _local->startStateFromTrial();
        }
        Vector old_residual;
        if (tr.theta < 1.0)
        {
          Problem::AssemblyOptions ao;
          ao.include_time_kernels = false;
          ao.theta = 1.0 - tr.theta;
          ao.dt = dt;
          // The old part of the theta method at the old time; see
          // Problem::takeTimeStep.
          _local->setTime(time - dt);
          _local->assemble(old, ao, old_residual, nullptr);
          _local->setTime(time);
        }
        Problem::AssemblyOptions base;
        base.old = &old;
        base.dt = dt;
        base.theta = tr.theta;
        base.include_time_kernels = true;
        base.include_steady_terms = tr.theta > 0;
        return nonlinearSolve(attempt, base, tr.theta < 1.0 ? &old_residual : nullptr);
      },
      [&](int, double)
      {
        if (integrator)
          integrator->accept(U, time);
        if (_local->hasState())
        {
          _local->commitState();
          committed = U;
        }
        _local->setTime(time);
        _local->notifyStepAccepted();
      },
      write,
      _comm.isRoot(),
      [&](int step, double next_dt)
      {
        if (tr.checkpoint_file.empty())
          return;
        const bool last = time >= tr.end_time - 1e-12 * std::abs(tr.end_time - tr.start_time);
        if ((tr.checkpoint_interval > 0 && step % tr.checkpoint_interval == 0) || last)
          writeCheckpoint(tr.checkpoint_file, *_local, layout, {time, next_dt, first_step + step});
      },
      integrator ? std::function<double(double)>([&](double dt)
                                                 { return integrator->courantNumber(U, dt); })
                 : std::function<double(double)>());
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
  os << ", and " << _owned_node.size() << (_cell_centered ? " cells and faces" : " nodes")
     << " of which " << std::count(_owned_node.begin(), _owned_node.end(), 1) << " are owned";
  return os.str();
}

} // namespace dualmesh
