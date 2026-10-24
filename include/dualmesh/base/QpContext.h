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

/// How an advective flux takes the value it carries across a face of a control volume method
/// (Jasak 1996, Sect. 3.3.1; the limited form of OpenFOAM's limitedScheme):
/// Central interpolates between the two sides, Upwind takes the upstream side, LinearUpwind
/// corrects it by the upstream gradient, phi_C + (x_f - x_C) . grad phi_C, and Limited blends the
/// upstream and the central values by the van Albada limiter of r = 2 d . grad phi_C / (phi_D -
/// phi_C) - 1.
enum class AdvectionScheme
{
  Central,
  Upwind,
  LinearUpwind,
  Limited
};

class QpContext
{
public:
  // ---- geometry and time ---------------------------------------------------
  int dim = 1;
  Point x{0, 0, 0};
  double time = 0.0;
  double dt = 0.0;
  /// The weight that a steady kernel gives the time derivative inside a residual it evaluates (the
  /// residual of a streamline stabilisation), so that the derivative enters a step of the theta
  /// method with weight 1: 1 / theta at the new time, where the kernel is weighted by theta, and 0
  /// at the old time, where it is weighted by 1 - theta and has no old value.
  double time_weight = 1.0;
  /// The weight theta of the steady terms at the new time of a step of the theta method, and 1
  /// otherwise.  A residual-based stabilisation of the new time weights the steady part of its
  /// residual by it, and the old state's by 1 - theta.
  double theta = 1.0;
  /// The value of the symbol eigenvalue of the expressions, set by an eigenvalue study (0 otherwise).
  double eigenvalue = 0.0;
  Index element = -1;
  int block = 0;
  /// Outward unit normal (boundary integration points only).
  Point normal{0, 0, 0};
  bool on_boundary = false;
  double load_factor = 1.0;
  /// The factor of the inertia of a flow: 1, or 0 for the Stokes solve with which a steady flow
  /// at rest starts (Problem::solveSteady).
  double inertia = 1.0;
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

  // ---- the two sides of a face (control volume methods) ------------------------
  /// At a point of a face between two control volumes, when an advective kernel asks for them:
  /// the values of the variables on side 0 (the owner) and side 1 (the neighbour), their
  /// reconstructed gradients where the method has them (empty for a side that has none), the
  /// positions of the two sides, the weight of side 0 in the central value, and in `normal` the
  /// unit normal from side 0 to side 1.
  bool on_face = false;
  std::vector<ADReal> side_u[2];
  std::vector<ADVector3> side_grad[2];
  Point side_x[2]{{0, 0, 0}, {0, 0, 0}};
  double side_weight = 0.5;
  /// The side (0 the owner, 1 the neighbour) for whose equation a non-conservative face flux is
  /// evaluated, or -1.  The flux of v . grad u through a face is m (u_f - u_C) for the control
  /// volume C on either side, so it differs between the two sides (Kernel::nonConservativeFaceFlux).
  int reference_side = -1;

  /// The value of variable @p v that an advective flux carries, whose normal velocity
  /// @p normal_velocity is positive from side 0 to side 1.  Away from a face (a volume point, the
  /// finite element method) it is the value at the point.
  /// The value of the side for whose equation a non-conservative face flux is evaluated
  /// (reference_side), or zero for a conservative one.
  ADReal referenceValue(int v) const
  {
    return on_face && reference_side >= 0 ? side_u[reference_side][v] : ADReal(0.0);
  }
  ADReal advectedValue(int v, double normal_velocity, AdvectionScheme scheme) const
  {
    if (!on_face || scheme == AdvectionScheme::Central)
      return u[v];
    const int c = normal_velocity >= 0.0 ? 0 : 1;
    const ADReal & upstream = side_u[c][v];
    if (scheme == AdvectionScheme::Upwind || side_grad[c].empty())
      return upstream;
    const ADVector3 & g = side_grad[c][v];
    if (scheme == AdvectionScheme::LinearUpwind)
    {
      ADReal value = upstream;
      for (int k = 0; k < 3; ++k)
        value += g[k] * (x[k] - side_x[c][k]);
      return value;
    }
    // r = 2a / b - 1 with a = d . grad phi_C, b = phi_1 - phi_0 and d = x_1 - x_0, and the van
    // Albada limiter psi(r) = r (r + 1) / (r^2 + 1) for r > 0, zero otherwise, written without the
    // division by b: psi = 2a (2a - b) / ((2a - b)^2 + b^2).
    ADReal a(0.0);
    for (int k = 0; k < 3; ++k)
      a += g[k] * (side_x[1][k] - side_x[0][k]);
    const ADReal b = side_u[1][v] - side_u[0][v];
    const ADReal two_a_minus_b = 2.0 * a - b;
    const ADReal central = side_weight * side_u[0][v] + (1.0 - side_weight) * side_u[1][v];
    if (two_a_minus_b.value() * b.value() <= 0.0)
      return upstream;
    const ADReal psi = 2.0 * a * two_a_minus_b / (two_a_minus_b * two_a_minus_b + b * b);
    return upstream + psi * (central - upstream);
  }

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
