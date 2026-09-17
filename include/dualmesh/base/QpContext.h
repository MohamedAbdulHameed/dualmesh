// SPDX-License-Identifier: LGPL-2.1-or-later
//
// Everything a kernel, boundary condition, or material may need at an
// integration point: position, time, the solution and its gradient
// (as AD numbers), lagged values for Picard iteration, old values for
// transient problems, and the material properties computed at that point.
#pragma once

#include "dualmesh/core/ADReal.h"
#include "dualmesh/core/Types.h"

#include <string>
#include <vector>

namespace dualmesh
{

enum class LinearizationMode
{
  Newton, ///< coefficients see the current iterate (exact Jacobian)
  Picard  ///< coefficients see the previous iterate (direct iteration)
};

class QpContext
{
public:
  // ---- geometry and time ---------------------------------------------------
  int dim = 1;
  Point x{0, 0, 0};
  double time = 0.0;
  double dt = 0.0;
  Index element = -1;
  int block = 0;
  /// Outward unit normal (boundary integration points only).
  Point normal{0, 0, 0};
  bool on_boundary = false;
  double load_factor = 1.0;
  LinearizationMode mode = LinearizationMode::Newton;

  // ---- fields (indexed by variable) -----------------------------------------
  std::vector<ADReal> u;
  std::vector<ADVector3> grad_u;
  std::vector<ADReal> u_lag;
  std::vector<ADVector3> grad_u_lag;
  std::vector<double> u_old;
  std::vector<Point> grad_u_old;

  /// Current value of variable v.
  const ADReal & value(int v) const { return u[v]; }
  const ADVector3 & gradient(int v) const { return grad_u[v]; }
  /// Value to be used inside nonlinear coefficients: equals value() for
  /// Newton's method and the previous iterate (no derivatives) for Picard.
  const ADReal & coefficientValue(int v) const
  {
    return mode == LinearizationMode::Newton ? u[v] : u_lag[v];
  }
  const ADVector3 & coefficientGradient(int v) const
  {
    return mode == LinearizationMode::Newton ? grad_u[v] : grad_u_lag[v];
  }
  double oldValue(int v) const { return u_old[v]; }
  const Point & oldGradient(int v) const { return grad_u_old[v]; }

  // ---- material properties ------------------------------------------------------
  /// Storage for material properties (see MaterialPropertyRegistry).
  std::vector<ADReal> properties;
  const ADReal & property(int id, int component = 0) const { return properties[id + component]; }
  ADReal & property(int id, int component = 0) { return properties[id + component]; }
};

} // namespace dualmesh
