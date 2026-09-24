// SPDX-License-Identifier: LGPL-2.1-or-later
#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <vector>

namespace dualmesh
{

using Real = double;
using Index = long;

/// A point (or vector) in physical space; unused components are zero.
using Point = std::array<double, 3>;

inline Point
operator+(const Point & a, const Point & b)
{
  return {a[0] + b[0], a[1] + b[1], a[2] + b[2]};
}
inline Point
operator-(const Point & a, const Point & b)
{
  return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}
inline Point
operator*(double s, const Point & a)
{
  return {s * a[0], s * a[1], s * a[2]};
}
inline double
dot(const Point & a, const Point & b)
{
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
inline Point
cross(const Point & a, const Point & b)
{
  return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}
inline double
norm(const Point & a)
{
  return std::sqrt(dot(a, a));
}

/// Coordinate system used for integration.
enum class CoordinateSystem
{
  Cartesian,      ///< dV = dx dy dz
  Axisymmetric,   ///< dV = 2 pi r dr dz, with r = x (first coordinate)
  SphericalRadial ///< dV = 4 pi r^2 dr (one-dimensional meshes only)
};

/// Discretization method.
enum class Method
{
  DualMesh,      ///< dual mesh control domain method (DMCDM)
  FiniteElement, ///< standard Galerkin weak form (for comparison)
  /// Vertex-centred finite volume method: the control domains of the dual mesh
  /// with two-point (edge) gradients at their interfaces.  This is the
  /// half-control volume formulation, HFVM, of Reddy, Chapter 3.
  FiniteVolumeVertex,
  /// Cell-centred finite volume method: one unknown per element plus one per
  /// boundary face.  This is the zero-thickness control volume formulation,
  /// ZFVM, of Reddy, Chapter 3, and the layout used by OpenFOAM.
  FiniteVolumeCell
};

/// Approximation of the normal gradient on a boundary face of the cell-centred
/// finite volume method.
enum class BoundaryGradient
{
  FirstOrder, ///< two-point difference between the cell and the boundary node
  SecondOrder ///< one-sided quadratic through the boundary node and two cells
};

} // namespace dualmesh
