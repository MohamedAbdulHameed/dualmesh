// SPDX-License-Identifier: LGPL-2.1-or-later
//
// The console output of the nonlinear solvers, shared by the serial and the
// distributed executioners, so that both print the same aligned table:
//
//   Newton, load step 1 of 2, load factor 0.5
//     iteration   residual |R|   |R| / |R_0|   step |dU|/|U|   linear iterations
//             1      3.742e+00     1.000e+00       8.944e-01                   0
//             2      2.079e-15     5.556e-16       converged
#pragma once

#include <cstdio>
#include <string>

namespace dualmesh
{
namespace console
{

/// The heading of one load step of a nonlinear solve.
inline void
nonlinearHeader(const std::string & method, int step, int num_steps, double load_factor)
{
  std::printf(
      "  %s, load step %d of %d, load factor %.6g\n", method.c_str(), step, num_steps, load_factor);
  std::printf("    iteration   residual |R|   |R| / |R_0|   step |dU|/|U|   linear "
              "iterations\n");
}

/// One row of the table: the residual before the step, and the step taken
/// (or "converged" when none was needed).
inline void
nonlinearIteration(int iteration,
                   double residual,
                   double initial_residual,
                   bool converged,
                   double step = 0.0,
                   int linear_iterations = 0)
{
  const double relative = initial_residual > 0 ? residual / initial_residual : 0.0;
  if (converged)
    std::printf("    %9d   %12.3e   %11.3e   %13s\n", iteration, residual, relative, "converged");
  else
    std::printf("    %9d   %12.3e   %11.3e   %13.3e   %17d\n",
                iteration,
                residual,
                relative,
                step,
                linear_iterations);
  std::fflush(stdout);
}

} // namespace console
} // namespace dualmesh
