// SPDX-License-Identifier: LGPL-2.1-or-later
#include "dualmesh/base/Checkpoint.h"
#include "dualmesh/core/InputParameters.h"
#include "dualmesh/linalg/PetscSolver.h"
#include "dualmesh/mesh/Mesh.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>

#include <petscsf.h>
#include <petscvec.h>
#include <petscviewer.h>
#define DM_PETSC(call) petsc::check((call), #call)

namespace dualmesh
{

namespace
{
constexpr double kFormatVersion = 2.0;
constexpr int kHeaderSize = 11;
/// The largest relative difference of two fingerprints of the nodes of one mesh.  The sum of the
/// fingerprint runs in another order on another number of processes, which changes it by about
/// the machine precision times the square root of the number of nodes, relative to the sum of
/// the sizes of its terms.  A node moved by a fraction d of the size of the mesh changes it by
/// about d over the number of nodes, so the check finds a moved node down to d of about the number
/// of nodes times this tolerance.
constexpr double kNodeFingerprintTolerance = 1e-10;

MPI_Comm
communicator(const CheckpointLayout & layout)
{
  return layout.distributed ? PETSC_COMM_WORLD : PETSC_COMM_SELF;
}

/// The star forest from @p width values per local entity to their natural positions, global index times width plus component, in the layout of @p natural.
/// An entity with a negative global index, which only this process has, is left out.
PetscSF
naturalForest(MPI_Comm comm, Vec natural, const std::vector<Index> & global, int width)
{
  PetscLayout layout;
  DM_PETSC(VecGetLayout(natural, &layout));
  std::vector<PetscInt> local, remote;
  local.reserve(global.size() * static_cast<std::size_t>(width));
  remote.reserve(global.size() * static_cast<std::size_t>(width));
  for (std::size_t i = 0; i < global.size(); ++i)
    if (global[i] >= 0)
      for (int c = 0; c < width; ++c)
      {
        local.push_back(static_cast<PetscInt>(i * static_cast<std::size_t>(width)) + c);
        remote.push_back(static_cast<PetscInt>(global[i] * width + c));
      }
  PetscSF sf;
  DM_PETSC(PetscSFCreate(comm, &sf));
  DM_PETSC(PetscSFSetGraphLayout(sf,
                                 layout,
                                 static_cast<PetscInt>(remote.size()),
                                 local.data(),
                                 PETSC_COPY_VALUES,
                                 remote.data()));
  return sf;
}

/// Move the local values into the natural vector (every copy of a value is equal, so any one may land).
void
gatherNatural(MPI_Comm comm,
              Vec natural,
              const std::vector<Index> & global,
              int width,
              const std::vector<double> & local)
{
  PetscSF sf = naturalForest(comm, natural, global, width);
  PetscScalar * array = nullptr;
  DM_PETSC(VecGetArray(natural, &array));
  DM_PETSC(PetscSFReduceBegin(sf, MPIU_SCALAR, local.data(), array, MPI_REPLACE));
  DM_PETSC(PetscSFReduceEnd(sf, MPIU_SCALAR, local.data(), array, MPI_REPLACE));
  DM_PETSC(VecRestoreArray(natural, &array));
  DM_PETSC(PetscSFDestroy(&sf));
}

/// Give every local entity its values from the natural vector.
std::vector<double>
scatterNatural(MPI_Comm comm, Vec natural, const std::vector<Index> & global, int width)
{
  PetscSF sf = naturalForest(comm, natural, global, width);
  std::vector<double> local(global.size() * static_cast<std::size_t>(width), 0.0);
  const PetscScalar * array = nullptr;
  DM_PETSC(VecGetArrayRead(natural, &array));
  DM_PETSC(PetscSFBcastBegin(sf, MPIU_SCALAR, array, local.data(), MPI_REPLACE));
  DM_PETSC(PetscSFBcastEnd(sf, MPIU_SCALAR, array, local.data(), MPI_REPLACE));
  DM_PETSC(VecRestoreArrayRead(natural, &array));
  DM_PETSC(PetscSFDestroy(&sf));
  return local;
}

/// The length @p count times @p width of a vector of the checkpoint, as PETSc's integer.  A length
/// beyond the range of PetscInt is refused, because the cast would wrap around.  @p count is a
/// number of the whole problem, the same on every process, so every process refuses together.
PetscInt
vectorLength(Index count, int width, const char * what)
{
  if (count < 0 || width < 0 ||
      (width > 0 && count > static_cast<Index>(PETSC_MAX_INT) / static_cast<Index>(width)))
    throw std::runtime_error("dualmesh: the checkpoint of " + std::string(what) + " has " +
                             std::to_string(count) + " times " + std::to_string(width) +
                             " values, beyond the largest index of this PETSc "
                             "build (" +
                             std::to_string(static_cast<long long>(PETSC_MAX_INT)) +
                             "). Configure PETSc with --with-64-bit-indices.");
  return static_cast<PetscInt>(count * static_cast<Index>(width));
}

PetscViewer
openViewer(MPI_Comm comm, const std::string & file, PetscFileMode mode)
{
  PetscViewer viewer;
  DM_PETSC(PetscViewerCreate(comm, &viewer));
  DM_PETSC(PetscViewerSetType(viewer, PETSCVIEWERBINARY));
  DM_PETSC(PetscViewerFileSetMode(viewer, mode));
  // PETSc's options file beside the data is of no use here.
  DM_PETSC(PetscViewerBinarySkipInfo(viewer));
  DM_PETSC(PetscViewerFileSetName(viewer, file.c_str()));
  return viewer;
}

/// The size of the record of one element: the number of points, then for each point its key (in two halves of 32 bits, which a double holds exactly) and its history.
struct RecordShape
{
  int points = 0;
  int history = 0;
  int size() const { return points > 0 ? 1 + points * (2 + history) : 0; }
};

RecordShape
recordShape(MPI_Comm comm, const Problem::StateSnapshot & state)
{
  int points = 0, history = 0;
  for (std::size_t e = 0; e < state.keys.size(); ++e)
  {
    const auto & keys = state.keys[e];
    points = std::max(points, static_cast<int>(keys.size()));
    if (!keys.empty())
      history = std::max(history, static_cast<int>(state.values[e].size() / keys.size()));
  }
  RecordShape shape;
  MPI_Allreduce(&points, &shape.points, 1, MPI_INT, MPI_MAX, comm);
  MPI_Allreduce(&history, &shape.history, 1, MPI_INT, MPI_MAX, comm);
  return shape;
}

/// Stop every process of @p comm together when one of them found an error: a check that fails on
/// one process alone, before a collective call, would leave the others waiting in it.  The
/// process that found the error throws its message, and the others name the process.
void
stopTogether(MPI_Comm comm, const std::string & error)
{
  int failed = error.empty() ? -1 : 0;
  int rank = 0;
  MPI_Comm_rank(comm, &rank);
  if (!error.empty())
    failed = rank;
  int first = -1;
  MPI_Allreduce(&failed, &first, 1, MPI_INT, MPI_MAX, comm);
  if (first < 0)
    return;
  if (!error.empty())
    throw InputError(error);
  throw InputError("The checkpoint stopped on process " + std::to_string(first) +
                   ", whose message gives the reason.");
}

/// The objects other than the property objects that keep a history, in the order of the problem,
/// which is the same on every process.
std::vector<CheckpointHistory *>
otherHistories(const Problem & problem)
{
  std::vector<CheckpointHistory *> out;
  for (const auto & bc : problem.integratedBCs())
    if (auto * history = dynamic_cast<CheckpointHistory *>(bc.get()))
      out.push_back(history);
  return out;
}

/// Write @p values, which rank 0 sets, as one vector.
void
writeValues(MPI_Comm comm, PetscViewer viewer, const std::vector<double> & values)
{
  int rank = 0;
  MPI_Comm_rank(comm, &rank);
  const auto n = static_cast<PetscInt>(values.size());
  Vec v;
  DM_PETSC(VecCreateMPI(comm, PETSC_DECIDE, n, &v));
  if (rank == 0)
  {
    std::vector<PetscInt> indices(values.size());
    std::iota(indices.begin(), indices.end(), PetscInt(0));
    DM_PETSC(VecSetValues(v, n, indices.data(), values.data(), INSERT_VALUES));
  }
  DM_PETSC(VecAssemblyBegin(v));
  DM_PETSC(VecAssemblyEnd(v));
  DM_PETSC(VecView(v, viewer));
  DM_PETSC(VecDestroy(&v));
}

/// The next vector of the file, whatever its size, on every process of @p comm.
std::vector<double>
readValues(PetscViewer viewer, MPI_Comm comm)
{
  Vec v;
  DM_PETSC(VecCreate(comm, &v));
  DM_PETSC(VecSetType(v, VECSTANDARD));
  DM_PETSC(VecLoad(v, viewer));
  PetscInt n = 0;
  DM_PETSC(VecGetSize(v, &n));
  VecScatter to_all;
  Vec everywhere;
  DM_PETSC(VecScatterCreateToAll(v, &to_all, &everywhere));
  DM_PETSC(VecScatterBegin(to_all, v, everywhere, INSERT_VALUES, SCATTER_FORWARD));
  DM_PETSC(VecScatterEnd(to_all, v, everywhere, INSERT_VALUES, SCATTER_FORWARD));
  const PetscScalar * a = nullptr;
  DM_PETSC(VecGetArrayRead(everywhere, &a));
  const std::vector<double> values(a, a + n);
  DM_PETSC(VecRestoreArrayRead(everywhere, &a));
  DM_PETSC(VecScatterDestroy(&to_all));
  DM_PETSC(VecDestroy(&everywhere));
  DM_PETSC(VecDestroy(&v));
  return values;
}

/// The header of a checkpoint, on every process of @p comm.
std::vector<double>
readHeader(PetscViewer viewer, MPI_Comm comm, const std::string & file)
{
  const std::vector<double> head = readValues(viewer, comm);
  if (head.empty())
    throw InputError("'" + file + "' is not a dualmesh checkpoint.");
  if (head[0] != kFormatVersion)
    throw InputError("'" + file + "' was written by another version of the checkpoint format.");
  if (head.size() != static_cast<std::size_t>(kHeaderSize))
    throw InputError("'" + file + "' is not a dualmesh checkpoint.");
  return head;
}

/// Write one history by global element: its shape (points per element and values per point), and
/// then, when it has points, its records in a vector of @p length values.
void
writeHistory(MPI_Comm comm,
             PetscViewer viewer,
             const CheckpointLayout & layout,
             const Problem::StateSnapshot & state,
             const RecordShape & shape,
             PetscInt length)
{
  writeValues(
      comm, viewer, {static_cast<double>(shape.points), static_cast<double>(shape.history)});
  if (shape.size() == 0)
    return;
  const int size = shape.size();
  const std::size_t elements = layout.global_elements.size();
  std::vector<double> records(elements * static_cast<std::size_t>(size), 0.0);
  for (std::size_t e = 0; e < elements; ++e)
  {
    double * r = records.data() + e * static_cast<std::size_t>(size);
    const auto & keys = state.keys[e];
    const auto & values = state.values[e];
    r[0] = static_cast<double>(keys.size());
    for (std::size_t k = 0; k < keys.size(); ++k)
    {
      double * point = r + 1 + k * static_cast<std::size_t>(2 + shape.history);
      point[0] = static_cast<double>(keys[k] >> 32);
      point[1] = static_cast<double>(keys[k] & 0xffffffffu);
      for (int h = 0; h < shape.history; ++h)
        point[2 + h] = values[k * static_cast<std::size_t>(shape.history) + h];
    }
  }
  Vec history;
  DM_PETSC(VecCreateMPI(comm, PETSC_DECIDE, length, &history));
  gatherNatural(comm, history, layout.global_elements, size, records);
  DM_PETSC(VecView(history, viewer));
  DM_PETSC(VecDestroy(&history));
}

/// Read one history that writeHistory wrote into the records of the local elements of @p state.
/// False when the file holds no points of it, and @p state is then unchanged.
bool
readHistory(PetscViewer viewer,
            MPI_Comm comm,
            const std::string & file,
            const CheckpointLayout & layout,
            Index file_elements,
            Problem::StateSnapshot & state)
{
  const std::vector<double> shape_values = readValues(viewer, comm);
  if (shape_values.size() != 2)
    throw InputError("'" + file + "' is not a dualmesh checkpoint.");
  RecordShape shape;
  shape.points = static_cast<int>(shape_values[0]);
  shape.history = static_cast<int>(shape_values[1]);
  if (shape.size() == 0)
    return false;
  std::string error;
  if (file_elements != layout.num_global_elements ||
      layout.global_elements.size() > state.keys.size())
    error = "'" + file + "' holds a history of " + std::to_string(file_elements) +
            " elements, which does not fit this problem; a checkpoint restarts the problem "
            "that wrote it.";
  stopTogether(comm, error);
  const int size = shape.size();
  Vec history;
  DM_PETSC(VecCreateMPI(
      comm, PETSC_DECIDE, vectorLength(layout.num_global_elements, size, "a history"), &history));
  DM_PETSC(VecLoad(history, viewer));
  const auto records = scatterNatural(comm, history, layout.global_elements, size);
  DM_PETSC(VecDestroy(&history));
  for (std::size_t e = 0; e < layout.global_elements.size(); ++e)
  {
    const double * r = records.data() + e * static_cast<std::size_t>(size);
    const auto points = static_cast<std::size_t>(r[0]);
    auto & keys = state.keys[e];
    auto & values = state.values[e];
    keys.assign(points, 0);
    values.assign(points * static_cast<std::size_t>(shape.history), 0.0);
    for (std::size_t k = 0; k < points; ++k)
    {
      const double * point = r + 1 + k * static_cast<std::size_t>(2 + shape.history);
      keys[k] = (static_cast<std::uint64_t>(point[0]) << 32) | static_cast<std::uint64_t>(point[1]);
      for (int q = 0; q < shape.history; ++q)
        values[k * static_cast<std::size_t>(shape.history) + q] = point[2 + q];
    }
  }
  return true;
}

/// The finaliser of the splitmix64 generator: 64 well-mixed bits from @p z.
std::uint64_t
scramble(std::uint64_t z)
{
  z += 0x9e3779b97f4a7c15ull;
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
  return z ^ (z >> 31);
}

/// The hash @p hash followed by @p value, which depends on the order of the values.
std::uint64_t
combine(std::uint64_t hash, std::uint64_t value)
{
  return scramble(hash ^ scramble(value));
}

/// A weight in [1, 2) that depends on @p key alone.
double
weight(std::uint64_t key)
{
  return 1.0 + static_cast<double>(scramble(key) >> 11) * 0x1.0p-53;
}

/// The fingerprint of a mesh in the global numbering of a checkpoint, the same on every process
/// and for every partition.
struct MeshFingerprint
{
  /// The sum over global nodes n of w(3n + k) x_k, the coordinates x_k of the node weighted by
  /// weights w that depend on n and k, and the sum of the sizes of the terms.
  double nodes = 0.0;
  double node_scale = 0.0;
  /// The sum, modulo 2^64, of the hashes of the global elements, each of its global number, its
  /// type, its block and the global numbers of its nodes in order.  An integer sum is exact in
  /// any order.
  std::uint64_t elements = 0;
};

MeshFingerprint
meshFingerprint(MPI_Comm comm, const Problem & problem, const CheckpointLayout & layout)
{
  const Mesh & mesh = problem.mesh();
  MeshFingerprint fingerprint;
  // The nodes shared by several processes land once in the natural vector, which holds the term
  // of each global node and its size.
  std::vector<double> terms(layout.global_nodes.size() * 2, 0.0);
  for (std::size_t i = 0; i < layout.global_nodes.size(); ++i)
  {
    const auto n = static_cast<std::uint64_t>(layout.global_nodes[i]);
    const Point & x = mesh.node(static_cast<Index>(i));
    for (std::uint64_t k = 0; k < 3; ++k)
    {
      const double term = weight(3 * n + k) * x[k];
      terms[2 * i] += term;
      terms[2 * i + 1] += std::abs(term);
    }
  }
  Vec natural;
  DM_PETSC(VecCreateMPI(comm,
                        PETSC_DECIDE,
                        vectorLength(layout.num_global_nodes, 2, "the fingerprint of the mesh"),
                        &natural));
  gatherNatural(comm, natural, layout.global_nodes, 2, terms);
  PetscInt begin = 0, end = 0;
  DM_PETSC(VecGetOwnershipRange(natural, &begin, &end));
  const PetscScalar * array = nullptr;
  DM_PETSC(VecGetArrayRead(natural, &array));
  double local[2] = {0.0, 0.0};
  for (PetscInt j = begin; j < end; ++j)
    local[j % 2] += array[j - begin];
  DM_PETSC(VecRestoreArrayRead(natural, &array));
  DM_PETSC(VecDestroy(&natural));
  double sums[2] = {0.0, 0.0};
  MPI_Allreduce(local, sums, 2, MPI_DOUBLE, MPI_SUM, comm);
  fingerprint.nodes = sums[0];
  fingerprint.node_scale = sums[1];

  // Each global element is integrated by one process, which adds its hash.
  std::uint64_t hashes = 0;
  for (std::size_t e = 0; e < layout.global_elements.size(); ++e)
  {
    const Element & element = mesh.element(static_cast<Index>(e));
    std::uint64_t hash = scramble(static_cast<std::uint64_t>(layout.global_elements[e]));
    hash = combine(hash, static_cast<std::uint64_t>(element.type));
    hash = combine(hash, static_cast<std::uint64_t>(element.block));
    for (int k = 0; k < element.numNodes(); ++k)
      hash = combine(hash,
                     static_cast<std::uint64_t>(
                         layout.global_nodes[static_cast<std::size_t>(element.nodes[k])]));
    hashes += hash;
  }
  MPI_Allreduce(&hashes, &fingerprint.elements, 1, MPI_UINT64_T, MPI_SUM, comm);
  return fingerprint;
}
} // namespace

// ---------------------------------------------------------------------------
// The checkpoint layout of the serial identity of the parallel interface
// ---------------------------------------------------------------------------
CheckpointLayout
Parallel::checkpointLayout(const Problem & problem) const
{
  // One process holds the whole problem, numbered as it is.
  CheckpointLayout layout;
  layout.num_global_nodes = problem.mesh().numNodes();
  layout.global_nodes.resize(static_cast<std::size_t>(problem.mesh().numNodes()));
  std::iota(layout.global_nodes.begin(), layout.global_nodes.end(), 0);
  layout.num_global_elements = problem.mesh().numElements();
  layout.global_elements.resize(static_cast<std::size_t>(problem.mesh().numElements()));
  std::iota(layout.global_elements.begin(), layout.global_elements.end(), 0);
  return layout;
}

void
writeCheckpoint(const std::string & file,
                const Problem & problem,
                const CheckpointLayout & layout,
                const CheckpointTime & time)
{
  petsc::initialize();
  const MPI_Comm comm = communicator(layout);
  int rank = 0;
  MPI_Comm_rank(comm, &rank);
  const int nv = problem.numVariables();
  // The histories: that of the material, then those of the other objects that keep one.
  std::vector<Problem::StateSnapshot> histories{problem.snapshotState()};
  for (const CheckpointHistory * other : otherHistories(problem))
    histories.push_back(other->committedHistory(problem));
  std::vector<RecordShape> shapes;
  std::string error;
  // The ghost elements of a distributed run follow the integrated ones and belong to another rank's part of the file.
  const std::size_t elements = layout.global_elements.size();
  for (const auto & state : histories)
  {
    shapes.push_back(recordShape(comm, state));
    if (shapes.back().size() > 0 && elements > state.keys.size())
      error = "This problem has a history (of the material or of a cohesive zone), and its part "
              "of the mesh was built without the global numbers of its elements, which a "
              "checkpoint needs.";
  }
  stopTogether(comm, error);
  // The lengths of the vectors, checked before the file is opened.
  const PetscInt solution_length = vectorLength(layout.num_global_nodes, nv, "the solution");
  std::vector<PetscInt> history_lengths;
  for (const auto & shape : shapes)
    history_lengths.push_back(vectorLength(layout.num_global_elements, shape.size(), "a history"));
  const MeshFingerprint fingerprint = meshFingerprint(comm, problem, layout);

  const std::string temporary = file + ".writing";
  PetscViewer viewer = openViewer(comm, temporary, FILE_MODE_WRITE);
  // The hash of the elements is written in two halves of 32 bits, which a double holds exactly.
  writeValues(comm,
              viewer,
              {kFormatVersion,
               time.time,
               time.dt,
               static_cast<double>(time.step),
               static_cast<double>(layout.num_global_nodes),
               static_cast<double>(nv),
               static_cast<double>(layout.num_global_elements),
               static_cast<double>(histories.size()),
               fingerprint.nodes,
               static_cast<double>(fingerprint.elements >> 32),
               static_cast<double>(fingerprint.elements & 0xffffffffu)});

  // The solution, by global node and variable.
  Vec solution;
  DM_PETSC(VecCreateMPI(comm, PETSC_DECIDE, solution_length, &solution));
  const Vector & U = problem.solution();
  gatherNatural(comm,
                solution,
                layout.global_nodes,
                nv,
                std::vector<double>(U.data(), U.data() + layout.global_nodes.size() * nv));
  DM_PETSC(VecView(solution, viewer));
  DM_PETSC(VecDestroy(&solution));

  // The histories, by global element.
  for (std::size_t h = 0; h < histories.size(); ++h)
    writeHistory(comm, viewer, layout, histories[h], shapes[h], history_lengths[h]);
  DM_PETSC(PetscViewerDestroy(&viewer));
  // The complete file replaces the previous checkpoint.
  if (rank == 0 && std::rename(temporary.c_str(), file.c_str()) != 0)
    error = "dualmesh: could not rename the checkpoint '" + temporary + "' to '" + file + "'.";
  stopTogether(comm, error);
}

CheckpointTime
checkpointTime(const std::string & file, bool distributed)
{
  petsc::initialize();
  const MPI_Comm comm = distributed ? PETSC_COMM_WORLD : PETSC_COMM_SELF;
  PetscViewer viewer = openViewer(comm, file, FILE_MODE_READ);
  const auto head = readHeader(viewer, comm, file);
  DM_PETSC(PetscViewerDestroy(&viewer));
  return {head[1], head[2], static_cast<int>(head[3])};
}

CheckpointTime
readCheckpoint(const std::string & file, Problem & problem, const CheckpointLayout & layout)
{
  petsc::initialize();
  const MPI_Comm comm = communicator(layout);
  const int nv = problem.numVariables();
  PetscViewer viewer = openViewer(comm, file, FILE_MODE_READ);
  const std::vector<double> head = readHeader(viewer, comm, file);
  CheckpointTime time;
  time.time = head[1];
  time.dt = head[2];
  time.step = static_cast<int>(head[3]);
  const auto file_nodes = static_cast<Index>(head[4]);
  const auto file_variables = static_cast<int>(head[5]);
  const auto file_elements = static_cast<Index>(head[6]);
  const auto file_histories = static_cast<std::size_t>(head[7]);
  const double file_node_fingerprint = head[8];
  const std::uint64_t file_element_fingerprint =
      (static_cast<std::uint64_t>(head[9]) << 32) | static_cast<std::uint64_t>(head[10]);
  // Every process reads the same header and reduces the same fingerprint, so every process finds
  // the same error.
  std::string error;
  const std::vector<CheckpointHistory *> others = otherHistories(problem);
  if (file_nodes != layout.num_global_nodes || file_variables != nv)
    error = "'" + file + "' holds " + std::to_string(file_nodes) + " nodes and " +
            std::to_string(file_variables) + " variables, and the problem has " +
            std::to_string(layout.num_global_nodes) + " nodes and " + std::to_string(nv) +
            " variables; a checkpoint restarts the problem that wrote it.";
  else if (file_histories != 1 + others.size())
    error = "'" + file + "' holds " + std::to_string(file_histories) +
            " histories (the material and the cohesive zones), and the problem has " +
            std::to_string(1 + others.size()) +
            "; a checkpoint restarts the problem that wrote it.";
  else
  {
    const MeshFingerprint fingerprint = meshFingerprint(comm, problem, layout);
    const double difference = std::abs(file_node_fingerprint - fingerprint.nodes);
    if (!(difference <= kNodeFingerprintTolerance * fingerprint.node_scale))
    {
      std::ostringstream relative;
      relative << difference / fingerprint.node_scale;
      error = "'" + file + "' was written on another mesh: the nodes of the mesh have other " +
              "coordinates or another numbering (relative difference " + relative.str() +
              " of the fingerprints of the nodes). A checkpoint restarts the problem that "
              "wrote it, on the same mesh.";
    }
    else if (file_element_fingerprint != fingerprint.elements)
      error = "'" + file + "' was written on another mesh: the elements of the mesh have " +
              "another numbering, other nodes, other types or other blocks. A checkpoint " +
              "restarts the problem that wrote it, on the same mesh.";
  }
  if (!error.empty())
  {
    DM_PETSC(PetscViewerDestroy(&viewer));
    throw InputError(error);
  }

  Vec solution;
  DM_PETSC(VecCreateMPI(
      comm, PETSC_DECIDE, vectorLength(layout.num_global_nodes, nv, "the solution"), &solution));
  DM_PETSC(VecLoad(solution, viewer));
  const auto U = scatterNatural(comm, solution, layout.global_nodes, nv);
  DM_PETSC(VecDestroy(&solution));
  Vector & target = problem.solution();
  for (std::size_t i = 0; i < U.size(); ++i)
    target[static_cast<Index>(i)] = U[i];
  problem.updateDependentDofs(target);

  auto material = problem.snapshotState();
  if (readHistory(viewer, comm, file, layout, file_elements, material))
    problem.restoreState(material);
  for (CheckpointHistory * other : others)
  {
    auto state = other->committedHistory(problem);
    if (readHistory(viewer, comm, file, layout, file_elements, state))
      stopTogether(comm, other->restoreHistory(problem, state));
  }
  DM_PETSC(PetscViewerDestroy(&viewer));
  return time;
}

} // namespace dualmesh
