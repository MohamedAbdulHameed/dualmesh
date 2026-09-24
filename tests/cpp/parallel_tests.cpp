// SPDX-License-Identifier: LGPL-2.1-or-later
//
// Tests of the mesh partitioners and of the distributed solver.
//
// Build without MPI, this program checks the partitioners and runs the
// distributed solver on a single rank, where it must reproduce the serial
// answer exactly.  Built with MPI (-DDUALMESH_ENABLE_MPI=ON) and launched with
//
//     mpirun -n 4 dualmesh_parallel_tests
//
// it additionally checks that the answer does not depend on the number of
// ranks: every rank computes the reference serial solution on the whole mesh
// and compares it, node by node, with the distributed one.
#include "dualmesh/base/Problem.h"
#include "dualmesh/parallel/DistributedProblem.h"

#include <cmath>
#include <functional>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace dualmesh;

namespace
{
int g_failures = 0;

void
check(bool condition, const std::string & what)
{
  if (!Communicator::world().isRoot())
    return;
  if (condition)
    std::cout << "[  OK  ] " << what << "\n";
  else
  {
    std::cout << "[ FAIL ] " << what << "\n";
    ++g_failures;
  }
}

std::vector<double>
grid(int n, double a = 0.0, double b = 1.0)
{
  std::vector<double> x(n + 1);
  for (int i = 0; i <= n; ++i)
    x[i] = a + (b - a) * i / n;
  return x;
}

InputParameters
params(const std::string & type, std::initializer_list<std::pair<std::string, ParameterValue>> kv)
{
  auto p = Factory::instance().validParams(type);
  for (const auto & [k, v] : kv)
    p.set(k, v);
  return p;
}

/// Steady heat conduction with a source on the unit square, with a mixture of
/// prescribed temperature and convection boundary conditions, so that both
/// kinds of boundary term are exercised.
void
defineConduction(Problem & problem)
{
  problem.addVariable("temperature");
  problem.addObject(
      "HeatConduction",
      "conduction",
      params("HeatConduction",
             {{"variable", std::string("temperature")}, {"thermal_conductivity", 2.5}}));
  problem.addObject(
      "HeatSource",
      "source",
      params("HeatSource", {{"variable", std::string("temperature")}, {"heat_source", 40.0}}));
  problem.addObject("DirichletBC",
                    "cold",
                    params("DirichletBC",
                           {{"variable", std::string("temperature")},
                            {"boundary", std::vector<std::string>{"left", "bottom"}},
                            {"value", 20.0}}));
  problem.addObject("ConvectiveHeatFluxBC",
                    "convection",
                    params("ConvectiveHeatFluxBC",
                           {{"variable", std::string("temperature")},
                            {"boundary", std::vector<std::string>{"right", "top"}},
                            {"heat_transfer_coefficient", 15.0},
                            {"ambient_temperature", 5.0}}));
}
} // namespace

int
main()
{
  auto & comm = Communicator::world();
  if (comm.isRoot())
    std::cout << "dualmesh parallel tests: " << comm.size() << " rank"
              << (comm.size() == 1 ? "" : "s") << ", MPI "
              << (Communicator::haveMpi() ? "enabled" : "not built in") << ", METIS "
              << (haveMetis() ? "available" : "not available") << "\n\n";

  // ---- partitioners ------------------------------------------------------
  {
    Mesh mesh = generateRectangleMesh(grid(20), grid(20), "Quad4");
    std::vector<std::string> methods = {"recursive_coordinate_bisection", "graph"};
    if (haveMetis())
      methods.push_back("metis");
    for (const auto & method : methods)
      for (int parts : {2, 4, 7})
      {
        const auto p = partitionMesh(mesh, parts, method);
        const auto [largest, smallest] = p.partSizes();
        bool all_assigned = true;
        for (int e : p.element_part)
          all_assigned = all_assigned && e >= 0 && e < parts;
        // No part may be more than 25% larger than the average.
        const double average = static_cast<double>(mesh.numElements()) / parts;
        const bool balanced = largest <= 1.25 * average + 1;
        std::ostringstream name;
        name << "partitioner " << method << " into " << parts << " parts (cut " << p.edgeCut(mesh)
             << " faces, sizes " << smallest << " to " << largest << ")";
        check(all_assigned && balanced && smallest > 0, name.str());
      }
  }

  // ---- sub-meshes preserve the boundary ----------------------------------
  {
    Mesh mesh = generateRectangleMesh(grid(8), grid(8), "Quad4");
    const auto p = partitionMesh(mesh, 4, "graph");
    Index total_sides = 0;
    std::vector<char> seen(mesh.numNodes(), 0);
    for (int r = 0; r < 4; ++r)
    {
      std::vector<Index> l2g;
      Mesh part = subMesh(mesh, p.elementsOf(r), l2g);
      for (const auto & name : {"left", "right", "bottom", "top"})
        total_sides += static_cast<Index>(part.sideset(name).size());
      for (Index g : l2g)
        seen[g] = 1;
    }
    Index reference = 0;
    for (const auto & name : {"left", "right", "bottom", "top"})
      reference += static_cast<Index>(mesh.sideset(name).size());
    check(total_sides == reference,
          "the sub-meshes of a partition hold every boundary side exactly once");
    check(std::count(seen.begin(), seen.end(), 1) == mesh.numNodes(),
          "the sub-meshes of a partition cover every node");
  }

  // ---- the distributed solution does not depend on the partition ---------
  for (auto [name, method] :
       std::vector<std::pair<std::string, Method>>{{"dmcdm", Method::DualMesh},
                                                   {"fem", Method::FiniteElement},
                                                   {"hfvm", Method::FiniteVolumeVertex}})
  {
    Mesh mesh = generateRectangleMesh(grid(12), grid(12), "Quad4");

    // Reference: the whole problem on one process.
    auto reference_mesh = std::make_shared<Mesh>(mesh);
    Problem reference(reference_mesh, method);
    defineConduction(reference);
    reference.solveSteady();
    const auto expected = reference.values("temperature");

    struct Variant
    {
      const char * preconditioner;
      int overlap;
      const char * subdomain_solver;
    };
    const Variant variants[] = {{"jacobi", 0, "ilu"},
                                {"additive_schwarz", 0, "ilu"},
                                {"additive_schwarz", 1, "ilu"},
                                {"additive_schwarz", 2, "lu"},
                                {"two_level_schwarz", 1, "ilu"},
                                {"two_level_schwarz", 2, "lu"}};
    for (const auto & partitioner : {"recursive_coordinate_bisection", "graph"})
      for (const auto & variant : variants)
      {
        const std::string preconditioner = std::string(variant.preconditioner) + " overlap " +
                                           std::to_string(variant.overlap) + " " +
                                           variant.subdomain_solver;
        DistributedOptions options;
        options.partitioner = partitioner;
        options.preconditioner = variant.preconditioner;
        options.overlap = variant.overlap;
        options.subdomain_solver = variant.subdomain_solver;
        options.linear_solver = "bicgstab";
        options.linear_tolerance = 1e-13;
        DistributedProblem distributed(mesh, method, CoordinateSystem::Cartesian, options);
        defineConduction(distributed.local());
        distributed.solveSteady();
        const auto got = distributed.gatheredValues("temperature");
        double worst = 0;
        for (Index n = 0; n < mesh.numNodes(); ++n)
          worst = std::max(worst, std::abs(got[n] - expected[n]));
        std::ostringstream label;
        label << name << ": distributed solution matches the serial one (" << partitioner << ", "
              << preconditioner << ", worst difference " << worst << ")";
        check(worst < 1e-8, label.str());
      }
  }

  // ---- a concentrated load is applied exactly once -----------------------
  {
    Mesh mesh = generateRectangleMesh(grid(10), grid(10), "Quad4");
    const auto define = [](Problem & problem)
    {
      problem.addVariable("u");
      problem.addObject(
          "Diffusion", "diffusion", params("Diffusion", {{"variable", std::string("u")}}));
      problem.addObject("PointSource",
                        "load",
                        params("PointSource",
                               {{"variable", std::string("u")},
                                {"value", 3.0},
                                {"points", std::vector<double>{0.5, 0.5, 0.0}}}));
      problem.addObject(
          "DirichletBC",
          "edges",
          params("DirichletBC",
                 {{"variable", std::string("u")},
                  {"boundary", std::vector<std::string>{"left", "right", "bottom", "top"}},
                  {"value", 0.0}}));
    };
    auto reference_mesh = std::make_shared<Mesh>(mesh);
    Problem reference(reference_mesh, Method::DualMesh);
    define(reference);
    reference.solveSteady();
    const auto expected = reference.values("u");

    DistributedOptions options;
    options.linear_tolerance = 1e-13;
    DistributedProblem distributed(mesh, Method::DualMesh, CoordinateSystem::Cartesian, options);
    define(distributed.local());
    distributed.solveSteady();
    const auto got = distributed.gatheredValues("u");
    double worst = 0;
    for (Index n = 0; n < mesh.numNodes(); ++n)
      worst = std::max(worst, std::abs(got[n] - expected[n]));
    std::ostringstream label;
    label << "a concentrated load is applied once, not once per rank (worst difference " << worst
          << ")";
    check(worst < 1e-8, label.str());
  }

  // ---- a vector problem on overlapping subdomains -------------------------
  // Plane elasticity: two unknowns per node, so the subdomain matrix, the
  // ghost exchange and the coarse space all carry more than one variable.
  {
    Mesh mesh = generateRectangleMesh(grid(16, 0.0, 2.0), grid(8), "Quad4");
    const auto define = [](Problem & problem)
    {
      problem.addVariable("disp_x");
      problem.addVariable("disp_y");
      problem.addObject("LinearElasticStress",
                        "stress",
                        params("LinearElasticStress",
                               {{"displacements", std::vector<std::string>{"disp_x", "disp_y"}},
                                {"youngs_modulus", 200.0},
                                {"poissons_ratio", 0.3}}));
      for (int c = 0; c < 2; ++c)
      {
        const std::string v = c == 0 ? "disp_x" : "disp_y";
        problem.addObject(
            "StressDivergence",
            "equilibrium_" + v,
            params("StressDivergence", {{"variable", v}, {"component", static_cast<long>(c)}}));
        problem.addObject(
            "DirichletBC",
            "clamp_" + v,
            params(
                "DirichletBC",
                {{"variable", v}, {"boundary", std::vector<std::string>{"left"}}, {"value", 0.0}}));
      }
      problem.addObject("TractionBC",
                        "load",
                        params("TractionBC",
                               {{"variable", std::string("disp_y")},
                                {"boundary", std::vector<std::string>{"right"}},
                                {"traction", -1.0}}));
    };
    auto reference_mesh = std::make_shared<Mesh>(mesh);
    Problem reference(reference_mesh, Method::DualMesh);
    define(reference);
    reference.solveSteady();
    for (const char * preconditioner : {"additive_schwarz", "two_level_schwarz"})
    {
      DistributedOptions options;
      options.preconditioner = preconditioner;
      options.overlap = 1;
      options.linear_tolerance = 1e-13;
      DistributedProblem distributed(mesh, Method::DualMesh, CoordinateSystem::Cartesian, options);
      define(distributed.local());
      distributed.solveSteady();
      double worst = 0, scale = 0;
      for (const char * v : {"disp_x", "disp_y"})
      {
        const auto expected = reference.values(v);
        const auto got = distributed.gatheredValues(v);
        for (Index n = 0; n < mesh.numNodes(); ++n)
        {
          worst = std::max(worst, std::abs(got[n] - expected[n]));
          scale = std::max(scale, std::abs(expected[n]));
        }
      }
      std::ostringstream label;
      label << "plane elasticity on overlapping subdomains matches the serial solution ("
            << preconditioner << ", worst relative difference " << worst / scale << ")";
      check(worst <= 1e-8 * scale, label.str());
    }
  }

  // ---- the conjugate gradient method with every preconditioner ----------
  // It needs a symmetric preconditioner, so the Schwarz methods are applied
  // in their classical (unrestricted) form; the Galerkin matrix of the
  // finite element method is symmetric positive definite.
  {
    Mesh mesh = generateRectangleMesh(grid(16), grid(16), "Quad4");
    auto reference_mesh = std::make_shared<Mesh>(mesh);
    Problem reference(reference_mesh, Method::FiniteElement);
    defineConduction(reference);
    reference.solveSteady();
    const auto expected = reference.values("temperature");
    for (const char * preconditioner : {"jacobi", "additive_schwarz", "two_level_schwarz"})
    {
      DistributedOptions options;
      options.linear_solver = "cg";
      options.preconditioner = preconditioner;
      options.linear_tolerance = 1e-13;
      DistributedProblem distributed(
          mesh, Method::FiniteElement, CoordinateSystem::Cartesian, options);
      defineConduction(distributed.local());
      const auto result = distributed.solveSteady();
      const auto got = distributed.gatheredValues("temperature");
      double worst = 0;
      for (Index n = 0; n < mesh.numNodes(); ++n)
        worst = std::max(worst, std::abs(got[n] - expected[n]));
      std::ostringstream label;
      label << "fem with cg and " << preconditioner << " matches the serial solution ("
            << result.linear_iterations << " iterations, worst difference " << worst << ")";
      check(worst < 1e-8, label.str());
    }
  }

  // ---- the two-level method does not degrade as ranks are added ----------
  // On the Poisson problem of a 48 by 48 mesh the measured BiCGSTAB counts
  // are 23 to 25 on every rank count from one to sixteen; without the coarse
  // level they grow to 33.  The bound leaves room for other partitioners.
  {
    Mesh mesh = generateRectangleMesh(grid(48), grid(48), "Quad4");
    DistributedOptions options;
    options.preconditioner = "two_level_schwarz";
    options.linear_tolerance = 1e-10;
    DistributedProblem distributed(mesh, Method::DualMesh, CoordinateSystem::Cartesian, options);
    Problem & problem = distributed.local();
    problem.addVariable("u");
    problem.addObject(
        "Diffusion", "diffusion", params("Diffusion", {{"variable", std::string("u")}}));
    problem.addObject("BodyForce",
                      "source",
                      params("BodyForce", {{"variable", std::string("u")}, {"value", 1.0}}));
    problem.addObject(
        "DirichletBC",
        "walls",
        params("DirichletBC",
               {{"variable", std::string("u")},
                {"boundary", std::vector<std::string>{"left", "right", "bottom", "top"}},
                {"value", 0.0}}));
    const auto result = distributed.solveSteady();
    std::ostringstream label;
    label << "two-level Schwarz with overlap 1 takes " << result.linear_iterations
          << " iterations on " << comm.size() << " ranks (bound 45)";
    check(result.linear_iterations <= 45, label.str());
  }

  // ---- transient -----------------------------------------------------------
  {
    Mesh mesh = generateRectangleMesh(grid(10), grid(10), "Quad4");
    struct Hill : Function
    {
      double value(const Point & x, double) const override
      {
        return std::sin(M_PI * x[0]) * std::sin(M_PI * x[1]);
      }
    };
    const auto define = [](Problem & problem)
    {
      problem.addVariable("temperature", {}, std::make_shared<Hill>());
      problem.addObject("Diffusion",
                        "diffusion",
                        params("Diffusion", {{"variable", std::string("temperature")}}));
      problem.addObject("TimeDerivative",
                        "time",
                        params("TimeDerivative", {{"variable", std::string("temperature")}}));
      problem.addObject(
          "DirichletBC",
          "edges",
          params("DirichletBC",
                 {{"variable", std::string("temperature")},
                  {"boundary", std::vector<std::string>{"left", "right", "bottom", "top"}},
                  {"value", 0.0}}));
    };
    TransientOptions transient;
    transient.end_time = 0.02;
    transient.dt = 0.002;
    transient.theta = 0.5;

    auto reference_mesh = std::make_shared<Mesh>(mesh);
    Problem reference(reference_mesh, Method::DualMesh);
    define(reference);
    reference.solveTransient(transient);
    const auto expected = reference.values("temperature");

    DistributedOptions options;
    options.linear_tolerance = 1e-13;
    DistributedProblem distributed(mesh, Method::DualMesh, CoordinateSystem::Cartesian, options);
    define(distributed.local());
    distributed.solveTransient(transient);
    const auto got = distributed.gatheredValues("temperature");
    double worst = 0;
    for (Index n = 0; n < mesh.numNodes(); ++n)
      worst = std::max(worst, std::abs(got[n] - expected[n]));
    std::ostringstream label;
    label << "a transient distributed solve matches the serial one (worst difference " << worst
          << ")";
    check(worst < 1e-8, label.str());
  }

  if (comm.isRoot())
    std::cout << "\n" << (g_failures ? "FAILED" : "all parallel tests passed") << "\n";
  return g_failures ? 1 : 0;
}
