// SPDX-License-Identifier: LGPL-2.1-or-later
//
// Timing harness.
//
//     dualmesh_benchmark [elements_per_side] [max_threads]
//
// builds a three-dimensional linear elasticity problem on a structured mesh of Hex8 elements, assembles the residual and the Jacobian a few times with a given number of threads, and reports the throughput.
// The residual norm is printed for every thread count, so a change in the number of threads that changes the answer shows up immediately.
//
//     mpirun -n P dualmesh_benchmark projection [elements_per_side] [steps] [order] [pressure_projection_vectors] [method]
//
// advances the decaying Taylor-Green vortex in a periodic square with the projection time integration on P processes, and reports the time per step and the iterations of the pressure and velocity solves per step, the measures of the scaling studies of Nek5000 and nekRS.
#include "dualmesh/base/Problem.h"
#include "dualmesh/modules/Framework.h"
#include "dualmesh/modules/Projection.h"
#include "dualmesh/parallel/DistributedProblem.h"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <tuple>
#include <vector>

using namespace dualmesh;

namespace
{
std::vector<double>
grid(int n)
{
  std::vector<double> x(n + 1);
  for (int i = 0; i <= n; ++i)
    x[i] = static_cast<double>(i) / n;
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

std::unique_ptr<Problem>
elasticity(int n, Method method)
{
  auto mesh = std::make_shared<Mesh>(generateBoxMesh(grid(n), grid(n), grid(n), "Hex8"));
  auto problem = std::make_unique<Problem>(mesh, method);
  const std::vector<std::string> displacements = {"disp_x", "disp_y", "disp_z"};
  for (const auto & d : displacements)
    problem->addVariable(d);
  problem->addObject("linear_elastic_stress",
                     "stress",
                     params("linear_elastic_stress",
                            {{"displacements", displacements},
                             {"formulation", std::string("three_dimensional")},
                             {"youngs_modulus", 200.0e9},
                             {"poissons_ratio", 0.3}}));
  for (int c = 0; c < 3; ++c)
    problem->addObject(
        "stress_divergence",
        "sd_" + std::to_string(c),
        params("stress_divergence",
               {{"variable", displacements[c]}, {"component", static_cast<long>(c)}}));
  for (int c = 0; c < 3; ++c)
    problem->addObject("Dirichlet_boundary_condition",
                       "fix_" + std::to_string(c),
                       params("Dirichlet_boundary_condition",
                              {{"variable", displacements[c]},
                               {"boundary", std::vector<std::string>{"left"}},
                               {"value", 0.0}}));
  problem->addObject("Dirichlet_boundary_condition",
                     "pull",
                     params("Dirichlet_boundary_condition",
                            {{"variable", std::string("disp_x")},
                             {"boundary", std::vector<std::string>{"right"}},
                             {"value", 1.0e-3}}));
  problem->initialize();
  return problem;
}

/// The velocity of the Taylor-Green vortex, decaying at the rate 2 nu.
struct TaylorGreen : Function
{
  int component = 0;
  double nu = 0.01;
  double value(const Point & x, double t) const override
  {
    const double decay = std::exp(-2.0 * nu * t);
    return component == 0 ? -std::cos(x[0]) * std::sin(x[1]) * decay
                          : std::sin(x[0]) * std::cos(x[1]) * decay;
  }
};

int
projection(int n, int steps, int order, int vectors, const std::string & method_name)
{
  const Method method = method_name == "dmcdm"  ? Method::DualMesh
                        : method_name == "hfvm" ? Method::FiniteVolumeVertex
                                                : Method::FiniteElement;
  auto & comm = Communicator::world();
  const double two_pi = 2.0 * std::acos(-1.0);
  std::vector<double> x = grid(n);
  for (auto & v : x)
    v *= two_pi;
  const Mesh mesh = generateRectangleMesh(x, x, "Quad4");
  DistributedProblem distributed(mesh, method);
  Problem & problem = distributed.local();
  for (int c = 0; c < 2; ++c)
  {
    auto exact = std::make_shared<TaylorGreen>();
    exact->component = c;
    const std::string name = c == 0 ? "u" : "v";
    problem.addVariable(name, {}, exact);
    problem.addObject("diffusion", name, params("diffusion", {{"variable", name}}));
  }
  problem.addVariable("pressure");
  problem.addObject(
      "diffusion", "pressure", params("diffusion", {{"variable", std::string("pressure")}}));
  for (const auto & [name, primary, secondary] :
       std::vector<std::tuple<std::string, std::string, std::string>>{
           {"periodic_x", "left", "right"}, {"periodic_y", "bottom", "top"}})
    problem.addObject(
        "periodic_boundary_condition",
        name,
        params("periodic_boundary_condition", {{"primary", primary}, {"secondary", secondary}}));
  ProjectionSettings settings;
  settings.velocities = {"u", "v"};
  settings.density = 1.0;
  settings.dynamic_viscosity = 0.01;
  settings.order = order;
  settings.pressure_projection_vectors = vectors;
  settings.pressure_tolerance = settings.velocity_tolerance = 1e-8;
  auto solver = std::make_shared<ProjectionSolver>(problem, &distributed, settings);
  problem.setTimeIntegrator(solver);
  // The step of a Courant number of one half at the largest speed, one.
  const double h = two_pi / n;
  TransientOptions transient;
  transient.dt = 0.5 * h;
  transient.end_time = steps * transient.dt;
  // The wall time after every step: the first ten steps (the setup of the solvers, the steps of lower order) are left out of the time per step.
  const int warmup = 10;
  std::vector<double> stamps;
  std::vector<int> pressure_iterations, velocity_iterations;
  const auto start = std::chrono::steady_clock::now();
  problem.setTimeStepCallback(
      [&](double, Problem &)
      {
        stamps.push_back(
            std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count());
        pressure_iterations.push_back(solver->lastPressureIterations());
        velocity_iterations.push_back(solver->lastVelocityIterations());
      });
  const SolveResult result = distributed.solveTransient(transient);
  const int recorded = static_cast<int>(stamps.size());
  if (recorded <= warmup + 1)
    throw std::runtime_error("projection benchmark: take more than eleven steps.");
  // The slowest process sets the pace.
  const double step_time =
      comm.max((stamps.back() - stamps[warmup]) / static_cast<double>(recorded - 1 - warmup));
  const double setup = comm.max(stamps[warmup] - warmup * step_time);
  double pressure = 0, velocity = 0;
  for (int k = warmup + 1; k < recorded; ++k)
  {
    pressure += pressure_iterations[k];
    velocity += velocity_iterations[k];
  }
  pressure /= recorded - 1 - warmup;
  velocity /= recorded - 1 - warmup;
  const Index unknowns = distributed.numGlobalDofs();
  if (comm.isRoot())
  {
    std::cout << "projection (" << method_name << "): Taylor-Green vortex, " << n << " x " << n
              << " Quad4 elements, " << comm.size() << " process" << (comm.size() == 1 ? "" : "es")
              << ", BDF" << order << "/EXT" << order << ", " << vectors << " projected pressures\n"
              << "  " << result.time_steps << " steps, " << std::fixed << std::setprecision(4)
              << step_time << " s per step after the first " << warmup << ", setup "
              << std::setprecision(2) << setup << " s, " << std::setprecision(1) << pressure
              << " pressure and " << velocity << " velocity iterations per step\n";
    std::cout << "RESULT method=" << method_name << " processes=" << comm.size() << " n=" << n
              << " unknowns=" << unknowns << " steps=" << result.time_steps << std::setprecision(6)
              << " step_time=" << step_time << " setup=" << setup
              << " pressure_iterations=" << pressure << " velocity_iterations=" << velocity << "\n";
  }
  return 0;
}
} // namespace

int
main(int argc, char ** argv)
{
  if (argc > 1 && std::string(argv[1]) == "projection")
    return projection(argc > 2 ? std::atoi(argv[2]) : 128,
                      argc > 3 ? std::atoi(argv[3]) : 20,
                      argc > 4 ? std::atoi(argv[4]) : 2,
                      argc > 5 ? std::atoi(argv[5]) : 8,
                      argc > 6 ? std::string(argv[6]) : std::string("fem"));
  const int n = argc > 1 ? std::atoi(argv[1]) : 24;
  const int max_threads = argc > 2 ? std::atoi(argv[2]) : 4;
  const int repeats = 3;

  std::cout << "dualmesh assembly benchmark\n"
            << "mesh: " << n << " x " << n << " x " << n << " Hex8 elements ("
            << static_cast<long>(n) * n * n << " elements, " << 3L * (n + 1) * (n + 1) * (n + 1)
            << " degrees of freedom)\n\n";

  for (auto [name, method] :
       std::vector<std::pair<std::string, Method>>{{"dmcdm", Method::DualMesh},
                                                   {"fem", Method::FiniteElement},
                                                   {"hfvm", Method::FiniteVolumeVertex}})
  {
    std::cout << name << "\n";
    double serial = 0;
    for (int threads = 1; threads <= max_threads; threads *= 2)
    {
      auto problem = elasticity(n, method);
      problem->setNumThreads(threads);
      const int used = problem->effectiveThreads();
      // A non-zero state, so that the residual is a meaningful fingerprint of
      // the assembly rather than identically zero.
      Vector & U = problem->solution();
      for (Index i = 0; i < U.size(); ++i)
        U[i] = 1.0e-4 * std::sin(0.37 * static_cast<double>(i) + 1.0);
      Problem::AssemblyOptions opts;
      opts.include_time_kernels = false;
      Vector R;
      SparseMatrix J;
      // One untimed pass, so that the triplet buffers are already allocated.
      problem->assemble(problem->solution(), opts, R, &J);
      const auto start = std::chrono::steady_clock::now();
      for (int k = 0; k < repeats; ++k)
        problem->assemble(problem->solution(), opts, R, &J);
      const auto stop = std::chrono::steady_clock::now();
      const double seconds = std::chrono::duration<double>(stop - start).count() / repeats;
      if (threads == 1)
        serial = seconds;
      std::cout << "  threads requested " << std::setw(2) << threads << ", used " << std::setw(2)
                << used << ":  " << std::fixed << std::setprecision(3) << seconds << " s"
                << "   speedup " << std::setprecision(2) << serial / seconds << "x"
                << "   |R| = " << std::scientific << std::setprecision(12) << R.norm() << "\n";
    }
    std::cout << "\n";
  }
  return 0;
}
