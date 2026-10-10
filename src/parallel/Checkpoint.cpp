// SPDX-License-Identifier: LGPL-2.1-or-later
#include "dualmesh/parallel/Checkpoint.h"
#include "dualmesh/core/InputParameters.h"
#include "dualmesh/linalg/PetscSolver.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <stdexcept>

#ifdef DUALMESH_HAVE_PETSC
#include <petscsf.h>
#include <petscvec.h>
#include <petscviewer.h>
#define DM_PETSC(call) petsc::check((call), #call)
#endif

namespace dualmesh
{

bool
checkpointsAvailable()
{
#ifdef DUALMESH_HAVE_PETSC
  return true;
#else
  return false;
#endif
}

#ifdef DUALMESH_HAVE_PETSC

namespace
{
constexpr double kFormatVersion = 1.0;
constexpr int kHeaderSize = 10;

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
  for (std::size_t e = 0; e < state.keys[0].size(); ++e)
  {
    const auto & keys = state.keys[0][e];
    points = std::max(points, static_cast<int>(keys.size()));
    if (!keys.empty())
      history = std::max(history, static_cast<int>(state.values[0][e].size() / keys.size()));
  }
  RecordShape shape;
  MPI_Allreduce(&points, &shape.points, 1, MPI_INT, MPI_MAX, comm);
  MPI_Allreduce(&history, &shape.history, 1, MPI_INT, MPI_MAX, comm);
  return shape;
}

/// The header of a checkpoint, on every process of @p comm.
std::vector<double>
readHeader(PetscViewer viewer, MPI_Comm comm, const std::string & file)
{
  Vec header;
  DM_PETSC(VecCreate(comm, &header));
  DM_PETSC(VecSetType(header, VECSTANDARD));
  DM_PETSC(VecLoad(header, viewer));
  PetscInt header_size = 0;
  DM_PETSC(VecGetSize(header, &header_size));
  if (header_size != kHeaderSize)
    throw InputError("'" + file + "' is not a dualmesh checkpoint.");
  VecScatter to_all;
  Vec everywhere;
  DM_PETSC(VecScatterCreateToAll(header, &to_all, &everywhere));
  DM_PETSC(VecScatterBegin(to_all, header, everywhere, INSERT_VALUES, SCATTER_FORWARD));
  DM_PETSC(VecScatterEnd(to_all, header, everywhere, INSERT_VALUES, SCATTER_FORWARD));
  const PetscScalar * h = nullptr;
  DM_PETSC(VecGetArrayRead(everywhere, &h));
  const std::vector<double> head(h, h + kHeaderSize);
  DM_PETSC(VecRestoreArrayRead(everywhere, &h));
  DM_PETSC(VecScatterDestroy(&to_all));
  DM_PETSC(VecDestroy(&everywhere));
  DM_PETSC(VecDestroy(&header));
  if (head[0] != kFormatVersion)
    throw InputError("'" + file + "' was written by another version of the checkpoint format.");
  return head;
}
} // namespace

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
  const auto state = problem.snapshotState();
  for (const auto & keys : state.keys[1])
    if (!keys.empty())
      throw InputError("A checkpoint holds the history of the material at the elements; the "
                       "history at the faces of the cell-centered method cannot be written yet.");
  const RecordShape shape = recordShape(comm, state);
  // The ghost elements of a distributed run follow the integrated ones and belong to another rank's part of the file.
  const std::size_t elements = layout.global_elements.size();
  if (shape.size() > 0 && elements > state.keys[0].size())
    throw InputError("This problem has a history of the material, and its part of the mesh was "
                     "built without the global numbers of its elements, which a checkpoint needs.");

  const std::string temporary = file + ".writing";
  PetscViewer viewer = openViewer(comm, temporary, FILE_MODE_WRITE);
  // The header: rank 0 sets every value.
  Vec header;
  DM_PETSC(VecCreateMPI(comm, PETSC_DECIDE, kHeaderSize, &header));
  if (rank == 0)
  {
    const PetscScalar values[kHeaderSize] = {kFormatVersion,
                                             time.time,
                                             time.dt,
                                             static_cast<double>(time.step),
                                             static_cast<double>(layout.num_global_nodes),
                                             static_cast<double>(nv),
                                             static_cast<double>(layout.num_global_elements),
                                             static_cast<double>(shape.size()),
                                             static_cast<double>(shape.points),
                                             static_cast<double>(shape.history)};
    PetscInt indices[kHeaderSize];
    for (int k = 0; k < kHeaderSize; ++k)
      indices[k] = k;
    DM_PETSC(VecSetValues(header, kHeaderSize, indices, values, INSERT_VALUES));
  }
  DM_PETSC(VecAssemblyBegin(header));
  DM_PETSC(VecAssemblyEnd(header));
  DM_PETSC(VecView(header, viewer));
  DM_PETSC(VecDestroy(&header));

  // The solution, by global node and variable.
  Vec solution;
  DM_PETSC(VecCreateMPI(
      comm, PETSC_DECIDE, static_cast<PetscInt>(layout.num_global_nodes * nv), &solution));
  const Vector & U = problem.solution();
  gatherNatural(comm,
                solution,
                layout.global_nodes,
                nv,
                std::vector<double>(U.data(), U.data() + layout.global_nodes.size() * nv));
  DM_PETSC(VecView(solution, viewer));
  DM_PETSC(VecDestroy(&solution));

  // The history of the material, by global element.
  if (shape.size() > 0)
  {
    const int size = shape.size();
    std::vector<double> records(elements * static_cast<std::size_t>(size), 0.0);
    for (std::size_t e = 0; e < elements; ++e)
    {
      double * r = records.data() + e * static_cast<std::size_t>(size);
      const auto & keys = state.keys[0][e];
      const auto & values = state.values[0][e];
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
    DM_PETSC(VecCreateMPI(
        comm, PETSC_DECIDE, static_cast<PetscInt>(layout.num_global_elements * size), &history));
    gatherNatural(comm, history, layout.global_elements, size, records);
    DM_PETSC(VecView(history, viewer));
    DM_PETSC(VecDestroy(&history));
  }
  DM_PETSC(PetscViewerDestroy(&viewer));
  // The complete file replaces the previous checkpoint.
  if (rank == 0 && std::rename(temporary.c_str(), file.c_str()) != 0)
    throw std::runtime_error("dualmesh: could not rename the checkpoint '" + temporary + "' to '" +
                             file + "'.");
  MPI_Barrier(comm);
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
  RecordShape shape;
  shape.points = static_cast<int>(head[8]);
  shape.history = static_cast<int>(head[9]);
  if (file_nodes != layout.num_global_nodes || file_variables != nv)
    throw InputError("'" + file + "' holds " + std::to_string(file_nodes) + " nodes and " +
                     std::to_string(file_variables) + " variables, and the problem has " +
                     std::to_string(layout.num_global_nodes) + " nodes and " + std::to_string(nv) +
                     " variables; a checkpoint restarts the problem that wrote it.");

  Vec solution;
  DM_PETSC(VecCreateMPI(
      comm, PETSC_DECIDE, static_cast<PetscInt>(layout.num_global_nodes * nv), &solution));
  DM_PETSC(VecLoad(solution, viewer));
  const auto U = scatterNatural(comm, solution, layout.global_nodes, nv);
  DM_PETSC(VecDestroy(&solution));
  Vector & target = problem.solution();
  for (std::size_t i = 0; i < U.size(); ++i)
    target[static_cast<Index>(i)] = U[i];
  problem.updateDependentDofs(target);

  if (shape.size() > 0)
  {
    auto state = problem.snapshotState();
    if (file_elements != layout.num_global_elements ||
        layout.global_elements.size() > state.keys[0].size())
      throw InputError("'" + file + "' holds the history of the material of " +
                       std::to_string(file_elements) +
                       " elements, which does not fit this problem; a checkpoint restarts the "
                       "problem that wrote it.");
    const int size = shape.size();
    Vec history;
    DM_PETSC(VecCreateMPI(
        comm, PETSC_DECIDE, static_cast<PetscInt>(layout.num_global_elements * size), &history));
    DM_PETSC(VecLoad(history, viewer));
    const auto records = scatterNatural(comm, history, layout.global_elements, size);
    DM_PETSC(VecDestroy(&history));
    for (std::size_t e = 0; e < layout.global_elements.size(); ++e)
    {
      const double * r = records.data() + e * static_cast<std::size_t>(size);
      const auto points = static_cast<std::size_t>(r[0]);
      auto & keys = state.keys[0][e];
      auto & values = state.values[0][e];
      keys.assign(points, 0);
      values.assign(points * static_cast<std::size_t>(shape.history), 0.0);
      for (std::size_t k = 0; k < points; ++k)
      {
        const double * point = r + 1 + k * static_cast<std::size_t>(2 + shape.history);
        keys[k] =
            (static_cast<std::uint64_t>(point[0]) << 32) | static_cast<std::uint64_t>(point[1]);
        for (int q = 0; q < shape.history; ++q)
          values[k * static_cast<std::size_t>(shape.history) + q] = point[2 + q];
      }
    }
    problem.restoreState(state);
  }
  DM_PETSC(PetscViewerDestroy(&viewer));
  return time;
}

#else

void
writeCheckpoint(const std::string &,
                const Problem &,
                const CheckpointLayout &,
                const CheckpointTime &)
{
  throw std::runtime_error("dualmesh: checkpoints need a build with PETSc (MPI, or "
                           "-DDUALMESH_ENABLE_PETSC=ON).");
}

CheckpointTime
checkpointTime(const std::string &, bool)
{
  throw std::runtime_error("dualmesh: checkpoints need a build with PETSc (MPI, or "
                           "-DDUALMESH_ENABLE_PETSC=ON).");
}

CheckpointTime
readCheckpoint(const std::string &, Problem &, const CheckpointLayout &)
{
  throw std::runtime_error("dualmesh: checkpoints need a build with PETSc (MPI, or "
                           "-DDUALMESH_ENABLE_PETSC=ON).");
}

#endif

} // namespace dualmesh
