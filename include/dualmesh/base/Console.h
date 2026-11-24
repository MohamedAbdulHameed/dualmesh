// SPDX-License-Identifier: LGPL-2.1-or-later
//
// The console output of the nonlinear solvers, shared by the serial and the
// distributed executioners, so that both print the same aligned table:
//
//   Newton, continuation step 2: the data at 0.5 of their values
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

/// The heading of a nonlinear solve, or of one step of its continuation (step 2 on).
inline void
nonlinearHeader(const std::string & method, int step, double data_fraction)
{
  if (step == 1)
    std::printf("  %s\n", method.c_str());
  else
    std::printf("  %s, continuation step %d: the data at %.6g of their values\n",
                method.c_str(),
                step,
                data_fraction);
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
