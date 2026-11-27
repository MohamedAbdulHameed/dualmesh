// SPDX-License-Identifier: LGPL-2.1-or-later
//
// Bindings of the build information, MPI, the mesh partition and the distributed problem.
#include "bindings.h"
#include "dualmesh/core/ADReal.h"
#include "dualmesh/linalg/Communicator.h"
#include "dualmesh/linalg/PetscSolver.h"
#include "dualmesh/parallel/DistributedProblem.h"
#include "dualmesh/parallel/GatherScatter.h"
#include "dualmesh/parallel/Partition.h"

#include <string>
#include <vector>

namespace dualmesh::bindings
{

void
bindParallel(py::module_ & m)
{
  // ---- build information -------------------------------------------------
  m.def(
      "build_information",
      []()
      {
        py::dict info;
        info["max_ad_derivatives"] = kMaxDerivatives;
#ifdef _OPENMP
        info["openmp"] = true;
#else
        info["openmp"] = false;
#endif
        info["petsc_version"] = petsc::version();
        return info;
      },
      "How this extension was built: the automatic differentiation budget, whether OpenMP is "
      "on, and the version of PETSc.");

  // ---- parallel ----------------------------------------------------------
  m.def("have_gslib",
        &GatherScatter::haveGslib,
        "Whether the extension was built with gslib, the gather-scatter of Nek5000.");
  m.def("petsc_version", &petsc::version, "PETSc's version, or an empty string without PETSc.");
  m.def("finalize_mpi",
        &Communicator::finalize,
        "Shut MPI down if this library started it; registered as an interpreter exit hook by "
        "dualmesh.parallel, and harmless to call twice.");
  m.def(
      "mpi_rank",
      []() { return Communicator::world().rank(); },
      "This process's rank, or 0 in a serial run.");
  m.def(
      "mpi_sum",
      [](std::vector<double> values)
      {
        Communicator::world().sumInPlace(values);
        return values;
      },
      "The elementwise sum over all processes of a list of numbers.");
  m.def(
      "mpi_max",
      [](double value) { return Communicator::world().max(value); },
      "The largest value over all processes.");
  m.def(
      "mpi_min",
      [](double value) { return Communicator::world().min(value); },
      "The smallest value over all processes.");
  m.def(
      "mpi_size",
      []() { return Communicator::world().size(); },
      "The number of processes, or 1 in a serial run.");
  m.def(
      "partition_mesh",
      [](std::shared_ptr<Mesh> mesh, int num_parts, const std::string & method)
      {
        const auto p = partitionMesh(*mesh, num_parts, method);
        py::dict out;
        out["element_part"] = p.element_part;
        out["node_owner"] = p.node_owner;
        out["edge_cut"] = p.edgeCut(*mesh);
        const auto [largest, smallest] = p.partSizes();
        out["largest_part"] = largest;
        out["smallest_part"] = smallest;
        return out;
      },
      py::arg("mesh"),
      py::arg("num_parts"),
      py::arg("method") = "automatic");
  m.def(
      "sub_mesh",
      [](std::shared_ptr<Mesh> mesh, const std::vector<Index> & elements)
      {
        std::vector<Index> l2g;
        auto out = std::make_shared<Mesh>(subMesh(*mesh, elements, l2g));
        return py::make_tuple(out, l2g);
      },
      py::arg("mesh"),
      py::arg("elements"));

  py::class_<DistributedOptions>(m, "DistributedOptions")
      .def(py::init<>())
      .def_readwrite("partitioner", &DistributedOptions::partitioner)
      .def_readwrite("gather_scatter", &DistributedOptions::gather_scatter);

  py::class_<DistributedProblem>(m, "DistributedProblem")
      .def(py::init(
               [](std::shared_ptr<Mesh> mesh,
                  const std::string & method,
                  const std::string & coordinates,
                  const DistributedOptions & options)
               {
                 return std::make_unique<DistributedProblem>(
                     *mesh, methodFromString(method), coordFromString(coordinates), options);
               }),
           py::arg("mesh"),
           py::arg("method"),
           py::arg("coordinates") = "cartesian",
           py::arg("options") = DistributedOptions{})
      .def(
          "local",
          [](DistributedProblem & d) -> Problem & { return d.local(); },
          py::return_value_policy::reference_internal)
      .def("rank", &DistributedProblem::rank)
      .def("num_ranks", &DistributedProblem::numRanks)
      .def("num_owned_dofs", &DistributedProblem::numOwnedDofs)
      .def("num_global_dofs", &DistributedProblem::numGlobalDofs)
      .def("solve_steady", &DistributedProblem::solveSteady, py::arg("options") = SolverOptions{})
      .def("solve_transient",
           &DistributedProblem::solveTransient,
           py::arg("transient"),
           py::arg("options") = SolverOptions{})
      .def("gathered_values", &DistributedProblem::gatheredValues, py::arg("variable"))
      .def("gathered_element_values",
           &DistributedProblem::gatheredElementValues,
           py::arg("local"),
           "The values of every element of the global mesh, on every rank, from one value per "
           "local element. Collective.")
      .def("set_gathered_values",
           &DistributedProblem::setGatheredValues,
           py::arg("variable"),
           py::arg("values"),
           "Set a variable from its values at every node of the global mesh.")
      .def("gathered_node_values",
           &DistributedProblem::gatheredNodeValues,
           py::arg("local"),
           py::arg("width"),
           "The values of every node of the global mesh, width per node, on every rank, from "
           "width values per local node. Collective.")
      .def("local_node_values",
           &DistributedProblem::localNodeValues,
           py::arg("global_values"),
           py::arg("width"),
           "The values of the local nodes, width per node, from width values per node of the "
           "global mesh.")
      .def("total_outflow",
           &DistributedProblem::totalOutflow,
           py::arg("variable"),
           py::arg("boundary"))
      .def(
          "write_vtu",
          [](const DistributedProblem & d,
             std::string filename,
             const std::vector<std::string> & cell_properties,
             const std::vector<std::string> & fields)
          {
            // The pieces and their index take the name without its extension.
            for (const std::string extension : {".pvtu", ".vtu"})
              if (filename.ends_with(extension))
              {
                filename.resize(filename.size() - extension.size());
                break;
              }
            d.writeVTU(filename, cell_properties, fields);
          },
          py::arg("filename"),
          py::arg("cell_properties") = std::vector<std::string>{},
          py::arg("fields") = std::vector<std::string>{})
      .def(
          "sum",
          [](const DistributedProblem & d, std::vector<double> values)
          {
            d.communicator().sumInPlace(values);
            return values;
          },
          py::arg("values"),
          "The elementwise sum over the ranks of a list of numbers. Collective.")
      .def(
          "maximum",
          [](const DistributedProblem & d, double value) { return d.maximum(value); },
          py::arg("value"),
          "The largest value over the ranks. Collective.")
      .def(
          "minimum",
          [](const DistributedProblem & d, double value) { return d.minimum(value); },
          py::arg("value"),
          "The smallest value over the ranks. Collective.")
      .def("summary", &DistributedProblem::summary);
}

} // namespace dualmesh::bindings
