// SPDX-License-Identifier: LGPL-2.1-or-later
#include "dualmesh/parallel/MeshDistribution.h"
#include "dualmesh/core/InputParameters.h"

#include <map>
#include <numeric>
#include <stdexcept>
#include <unordered_map>

#ifdef DUALMESH_HAVE_PETSC
#include "Plex.h"
#include "dualmesh/linalg/PetscSolver.h"

#include <petscsf.h>
#define DM_PETSC(call) petsc::check((call), #call)
#endif

namespace dualmesh
{

namespace
{
void
putName(std::vector<Index> & out, const std::string & name)
{
  out.push_back(static_cast<Index>(name.size()));
  for (char c : name)
    out.push_back(static_cast<Index>(static_cast<unsigned char>(c)));
}

std::string
takeName(const std::vector<Index> & in, std::size_t & at)
{
  const auto length = static_cast<std::size_t>(in[at++]);
  std::string name(length, ' ');
  for (std::size_t k = 0; k < length; ++k)
    name[k] = static_cast<char>(in[at++]);
  return name;
}

/// What every rank needs to know of the whole mesh: its dimension, its block names, and the names of its side sets and node sets.
struct Header
{
  int dimension = 0;
  std::map<int, std::string> block_names;
  std::vector<std::string> sidesets, nodesets;

  std::vector<Index> encode() const
  {
    std::vector<Index> out{dimension, static_cast<Index>(block_names.size())};
    for (const auto & [block, name] : block_names)
    {
      out.push_back(block);
      putName(out, name);
    }
    for (const auto * names : {&sidesets, &nodesets})
    {
      out.push_back(static_cast<Index>(names->size()));
      for (const auto & name : *names)
        putName(out, name);
    }
    return out;
  }

  static Header decode(const std::vector<Index> & in)
  {
    Header h;
    std::size_t at = 0;
    h.dimension = static_cast<int>(in[at++]);
    for (Index k = 0, n = in [at++]; k < n; ++k)
    {
      const int block = static_cast<int>(in[at++]);
      h.block_names[block] = takeName(in, at);
    }
    for (auto * names : {&h.sidesets, &h.nodesets})
      for (Index k = 0, n = in [at++]; k < n; ++k)
        names->push_back(takeName(in, at));
    return h;
  }
};

/// The part of a mesh that one rank receives, built element by element.
class PartBuilder
{
public:
  explicit PartBuilder(const Header & header) : _header(header)
  {
    _part.mesh = std::make_shared<Mesh>(header.dimension);
    _sides.resize(header.sidesets.size());
    _nodes.resize(header.nodesets.size());
  }

  /// Add one element from its record and the three coordinates of each of its nodes.
  /// The record holds the global index of the element, the type, the block, the number of nodes, their global indices, the number of sides and their (side set, side) pairs, and the number of memberships and their (node of the element, node set) pairs.
  /// The sides and memberships of a ghost element belong to its own rank and are left out.
  void add(const Index * record, const double * coordinates, bool ghost = false)
  {
    std::size_t at = 0;
    _part.global_elements.push_back(record[at++]);
    const auto type = static_cast<ElementType>(record[at++]);
    const int block = static_cast<int>(record[at++]);
    const int count = static_cast<int>(record[at++]);
    std::vector<Index> nodes(static_cast<std::size_t>(count));
    for (int k = 0; k < count; ++k)
    {
      const Index g = record[at++];
      auto [it, added] = _local_of.try_emplace(g, static_cast<Index>(_part.global_nodes.size()));
      if (added)
      {
        _part.mesh->addNode({coordinates[3 * k], coordinates[3 * k + 1], coordinates[3 * k + 2]});
        _part.global_nodes.push_back(g);
      }
      nodes[k] = it->second;
    }
    const Index element = _part.mesh->addElement(type, nodes, block);
    if (ghost)
    {
      ++_part.num_ghost_elements;
      return;
    }
    for (Index k = 0, n = record[at++]; k < n; ++k)
    {
      const auto set = static_cast<std::size_t>(record[at++]);
      _sides[set].push_back({element, static_cast<int>(record[at++])});
    }
    for (Index k = 0, n = record[at++]; k < n; ++k)
    {
      const auto node = static_cast<std::size_t>(record[at++]);
      _nodes[static_cast<std::size_t>(record[at++])].push_back(nodes[node]);
    }
  }

  LocalMesh finish()
  {
    for (const auto & [block, name] : _header.block_names)
      _part.mesh->setBlockName(block, name);
    // Every name is created on every rank, also where it is empty, so that a boundary condition naming it is valid everywhere.
    for (std::size_t s = 0; s < _sides.size(); ++s)
      _part.mesh->addSideset(_header.sidesets[s], _sides[s]);
    for (std::size_t s = 0; s < _nodes.size(); ++s)
      _part.mesh->addNodeset(_header.nodesets[s], _nodes[s]);
    return std::move(_part);
  }

private:
  const Header & _header;
  LocalMesh _part;
  std::unordered_map<Index, Index> _local_of;
  std::vector<std::vector<Side>> _sides;
  std::vector<std::vector<Index>> _nodes;
};

#ifdef DUALMESH_HAVE_PETSC
/// The records of every element of @p mesh, in sections over the cells of @p dm, which numbers cell e as element e.
/// Every side set is also a node set of the same name, holding its boundary nodes, so that a value prescribed on a boundary reaches every rank that holds one of its nodes, whichever rank holds the side.
void
records(const Mesh & mesh,
        const Header & header,
        DM dm,
        PetscSection & integer_section,
        std::vector<PetscInt> & integers,
        PetscSection & real_section,
        std::vector<PetscReal> & reals)
{
  const Index num_elements = mesh.numElements();
  std::vector<std::vector<std::pair<int, int>>> sides_of(static_cast<std::size_t>(num_elements));
  std::vector<std::vector<int>> sets_of(static_cast<std::size_t>(mesh.numNodes()));
  if (num_elements > 0)
  {
    for (std::size_t s = 0; s < header.sidesets.size(); ++s)
      for (const Side & side : mesh.sideset(header.sidesets[s]))
        sides_of[side.first].push_back({static_cast<int>(s), side.second});
    for (std::size_t s = 0; s < header.nodesets.size(); ++s)
      for (Index n : mesh.boundaryNodes(header.nodesets[s]))
        sets_of[n].push_back(static_cast<int>(s));
  }

  PetscInt chart_start = 0, chart_end = 0;
  DM_PETSC(DMPlexGetChart(dm, &chart_start, &chart_end));
  DM_PETSC(PetscSectionCreate(PETSC_COMM_WORLD, &integer_section));
  DM_PETSC(PetscSectionCreate(PETSC_COMM_WORLD, &real_section));
  DM_PETSC(PetscSectionSetChart(integer_section, chart_start, chart_end));
  DM_PETSC(PetscSectionSetChart(real_section, chart_start, chart_end));
  std::vector<std::vector<PetscInt>> record(static_cast<std::size_t>(num_elements));
  for (Index e = 0; e < num_elements; ++e)
  {
    const Element & el = mesh.element(e);
    auto & r = record[e];
    r = {static_cast<PetscInt>(e), static_cast<PetscInt>(el.type), el.block, el.numNodes()};
    for (int k = 0; k < el.numNodes(); ++k)
      r.push_back(static_cast<PetscInt>(el.nodes[k]));
    r.push_back(static_cast<PetscInt>(sides_of[e].size()));
    for (const auto & [set, side] : sides_of[e])
      r.insert(r.end(), {set, side});
    const std::size_t count_at = r.size();
    r.push_back(0);
    for (int k = 0; k < el.numNodes(); ++k)
      for (int set : sets_of[el.nodes[k]])
      {
        r.insert(r.end(), {k, set});
        ++r[count_at];
      }
    DM_PETSC(PetscSectionSetDof(integer_section, e, static_cast<PetscInt>(r.size())));
    DM_PETSC(PetscSectionSetDof(real_section, e, 3 * el.numNodes()));
  }
  DM_PETSC(PetscSectionSetUp(integer_section));
  DM_PETSC(PetscSectionSetUp(real_section));
  for (Index e = 0; e < num_elements; ++e)
  {
    integers.insert(integers.end(), record[e].begin(), record[e].end());
    const Element & el = mesh.element(e);
    for (int k = 0; k < el.numNodes(); ++k)
      for (int c = 0; c < 3; ++c)
        reals.push_back(mesh.node(el.nodes[k])[c]);
  }
}

/// Rebuild the local part from the records of the cells of @p dm.
LocalMesh
rebuild(DM dm,
        const Header & header,
        PetscSection integer_section,
        const PetscInt * integers,
        PetscSection real_section,
        const PetscReal * reals)
{
  PartBuilder part(header);
  PetscInt cell_start = 0, cell_end = 0;
  DM_PETSC(DMPlexGetHeightStratum(dm, 0, &cell_start, &cell_end));
  // The cells that another rank owns, the leaves of the point star forest, are the ghost elements, and follow the rank's own.
  std::vector<char> ghost(static_cast<std::size_t>(cell_end - cell_start), 0);
  PetscSF points;
  DM_PETSC(DMGetPointSF(dm, &points));
  PetscInt num_leaves = 0;
  const PetscInt * leaves = nullptr;
  DM_PETSC(PetscSFGetGraph(points, nullptr, &num_leaves, &leaves, nullptr));
  for (PetscInt k = 0; k < num_leaves; ++k)
  {
    const PetscInt point = leaves ? leaves[k] : k;
    if (point >= cell_start && point < cell_end)
      ghost[point - cell_start] = 1;
  }
  std::vector<Index> record;
  for (const bool ghosts : {false, true})
    for (PetscInt c = cell_start; c < cell_end; ++c)
    {
      if (static_cast<bool>(ghost[c - cell_start]) != ghosts)
        continue;
      PetscInt size = 0, offset = 0, real_offset = 0;
      DM_PETSC(PetscSectionGetDof(integer_section, c, &size));
      DM_PETSC(PetscSectionGetOffset(integer_section, c, &offset));
      DM_PETSC(PetscSectionGetOffset(real_section, c, &real_offset));
      record.assign(integers + offset, integers + offset + size);
      part.add(record.data(), reals + real_offset, ghosts);
    }
  return part.finish();
}
#endif
} // namespace

LocalMesh
distributeMesh(const Mesh & mesh,
               const Communicator & comm,
               const std::string & partitioner,
               int ghost_layers)
{
  if (ghost_layers < 0)
    throw InputError("The number of ghost layers cannot be negative.");
  // One rank holds the whole mesh, numbered as it is, and has no other rank's elements next to its own.
  if (comm.size() == 1)
  {
    LocalMesh part;
    part.ghost_layers = ghost_layers;
    part.mesh = std::make_shared<Mesh>(mesh);
    part.global_nodes.resize(static_cast<std::size_t>(mesh.numNodes()));
    std::iota(part.global_nodes.begin(), part.global_nodes.end(), 0);
    part.global_elements.resize(static_cast<std::size_t>(mesh.numElements()));
    std::iota(part.global_elements.begin(), part.global_elements.end(), 0);
    return part;
  }
#ifdef DUALMESH_HAVE_PETSC
  // Every rank waits for its part, so an error on the root must stop every rank.
  std::string error;
  Header header;
  std::string type;
  try
  {
    type = plex::partitionerType(partitioner);
    if (comm.isRoot())
    {
      if (mesh.numElements() < comm.size())
        throw InputError("The mesh has fewer elements (" + std::to_string(mesh.numElements()) +
                         ") than there are processes (" + std::to_string(comm.size()) + ").");
      header.dimension = mesh.dimension();
      header.block_names = mesh.blockNames();
      for (const auto & [name, sides] : mesh.sidesets())
        header.sidesets.push_back(name);
      header.nodesets = header.sidesets;
      for (const auto & [name, nodes] : mesh.nodesets())
        if (!mesh.sidesets().count(name))
          header.nodesets.push_back(name);
    }
  }
  catch (const std::exception & e)
  {
    error = e.what();
  }
  if (comm.any(!error.empty()))
    throw InputError(error.empty() ? "The root rank could not distribute the mesh; its message "
                                     "gives the reason."
                                   : error);
  std::vector<Index> encoded = comm.isRoot() ? header.encode() : std::vector<Index>();
  comm.broadcast(encoded);
  header = Header::decode(encoded);

  // The root's DMPlex holds every cell; PETSc's partitioner divides them, and DMPlexDistribute and DMPlexDistributeData move each cell and its record to its rank, together with the cells of the ghost layers (PETSc's overlap: in a mesh of cells and vertices, a layer holds the cells that share a vertex with the layers inside it).
  const Mesh empty(header.dimension);
  const Mesh & held = comm.isRoot() ? mesh : empty;
  // PETSC_COMM_WORLD is valid only once PETSc runs, which may be first here.
  petsc::initialize();
  DM dm = plex::cornerTopology(held, PETSC_COMM_WORLD, comm.isRoot());
  PetscPartitioner petsc_partitioner;
  DM_PETSC(DMPlexGetPartitioner(dm, &petsc_partitioner));
  DM_PETSC(PetscPartitionerSetType(petsc_partitioner, type.c_str()));
  PetscSection integer_section, real_section;
  std::vector<PetscInt> integers;
  std::vector<PetscReal> reals;
  records(held, header, dm, integer_section, integers, real_section, reals);
  PetscSF migration = nullptr;
  DM distributed = nullptr;
  DM_PETSC(DMPlexDistribute(dm, static_cast<PetscInt>(ghost_layers), &migration, &distributed));
  PetscSection new_integer_section, new_real_section;
  DM_PETSC(PetscSectionCreate(PETSC_COMM_WORLD, &new_integer_section));
  DM_PETSC(PetscSectionCreate(PETSC_COMM_WORLD, &new_real_section));
  PetscInt * new_integers = nullptr;
  PetscReal * new_reals = nullptr;
  DM_PETSC(DMPlexDistributeData(dm,
                                migration,
                                integer_section,
                                MPIU_INT,
                                integers.data(),
                                new_integer_section,
                                reinterpret_cast<void **>(&new_integers)));
  DM_PETSC(DMPlexDistributeData(dm,
                                migration,
                                real_section,
                                MPIU_REAL,
                                reals.data(),
                                new_real_section,
                                reinterpret_cast<void **>(&new_reals)));
  LocalMesh part =
      rebuild(distributed, header, new_integer_section, new_integers, new_real_section, new_reals);
  part.ghost_layers = ghost_layers;
  DM_PETSC(PetscFree(new_integers));
  DM_PETSC(PetscFree(new_reals));
  DM_PETSC(PetscSectionDestroy(&new_integer_section));
  DM_PETSC(PetscSectionDestroy(&new_real_section));
  DM_PETSC(PetscSectionDestroy(&integer_section));
  DM_PETSC(PetscSectionDestroy(&real_section));
  DM_PETSC(PetscSFDestroy(&migration));
  DM_PETSC(DMDestroy(&distributed));
  DM_PETSC(DMDestroy(&dm));
  return part;
#else
  (void) partitioner;
  throw std::runtime_error("distributeMesh: more than one rank needs a build with MPI and PETSc.");
#endif
}

} // namespace dualmesh
