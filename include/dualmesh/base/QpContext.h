// SPDX-License-Identifier: LGPL-2.1-or-later
//
// Everything a kernel, boundary condition, or property object may need at an
// integration point: position, time, the solution and its gradient
// (as AD numbers), lagged values for Picard iteration, old values for
// transient problems, and the properties computed at that point.
#pragma once

#include "dualmesh/core/ADReal.h"
#include "dualmesh/core/Types.h"

#include <cstdint>
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

  // ---- interface (gap) points ---------------------------------------------------
  /// At an integration point of an InterfaceBC: the values of the variables on
  /// the other side of the gap at the paired point, and that point.
  std::vector<ADReal> u_other;
  Point x_other{0, 0, 0};

  // ---- history (state) of stateful property objects ------------------------------------
  /// Where this point's history is kept: the owner (an element, or with
  /// state_domain 1 a cell-centred face) and a key unique within the owner.
  /// An owner of -1 means that the point has no history (a stateful property object
  /// then sees a zero old state and its new state is discarded).
  int state_domain = 0;
  Index state_owner = -1;
  std::uint64_t state_key = 0;
  /// Filled by the problem before the property objects run: the state at the start
  /// of the step and the place to write the new state, each the whole record
  /// of the point (a property object adds its own offset).
  const double * state_old = nullptr;
  double * state_new = nullptr;

  // ---- properties ------------------------------------------------------
  /// Storage for properties (see PropertyRegistry).
  std::vector<ADReal> properties;
  const ADReal & property(int id, int component = 0) const { return properties[id + component]; }
  ADReal & property(int id, int component = 0) { return properties[id + component]; }
};

} // namespace dualmesh
