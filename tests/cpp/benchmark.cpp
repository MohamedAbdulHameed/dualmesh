// SPDX-License-Identifier: LGPL-2.1-or-later
//
// Timing harness for the parallel assembly.  It builds a three-dimensional
// linear elasticity problem on a structured mesh of Hex8 elements, assembles
// the residual and the Jacobian a few times with a given number of threads,
// and reports the throughput.  Run it as
//
//     dualmesh_benchmark [elements_per_side] [max_threads]
//
// The residual norm is printed for every thread count, so a change in the
// number of threads that changes the answer shows up immediately.
#include "dualmesh/base/Problem.h"
#include "dualmesh/modules/Framework.h"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
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
  problem->addObject("LinearElasticStress",
                     "stress",
                     params("LinearElasticStress",
                            {{"displacements", displacements},
                             {"formulation", std::string("three_dimensional")},
                             {"youngs_modulus", 200.0e9},
                             {"poissons_ratio", 0.3}}));
  for (int c = 0; c < 3; ++c)
    problem->addObject(
        "StressDivergence",
        "sd_" + std::to_string(c),
        params("StressDivergence",
               {{"variable", displacements[c]}, {"component", static_cast<long>(c)}}));
  for (int c = 0; c < 3; ++c)
    problem->addObject("DirichletBC",
                       "fix_" + std::to_string(c),
                       params("DirichletBC",
                              {{"variable", displacements[c]},
                               {"boundary", std::vector<std::string>{"left"}},
                               {"value", 0.0}}));
  problem->addObject("DirichletBC",
                     "pull",
                     params("DirichletBC",
                            {{"variable", std::string("disp_x")},
                             {"boundary", std::vector<std::string>{"right"}},
                             {"value", 1.0e-3}}));
  problem->initialize();
  return problem;
}
} // namespace

int
main(int argc, char ** argv)
{
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
