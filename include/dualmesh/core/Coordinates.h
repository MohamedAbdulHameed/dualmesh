// SPDX-License-Identifier: LGPL-2.1-or-later
//
// The terms that the coordinate system adds to an equation: the factor of the volume element and
// the hoop terms of a divergence and of a strain.  The radius r is the first coordinate x[0].  An
// axisymmetric body has one hoop direction (theta), and a spherically symmetric body two (theta
// and phi), which carry the same strain u_r / r and the same stress.
#pragma once

#include "dualmesh/core/Types.h"

namespace dualmesh
{

/// The factor of the volume element of the coordinate system at the radius @p r: 1 in Cartesian
/// coordinates, 2 pi r in axisymmetric and 4 pi r^2 in spherical coordinates.
template <typename Scalar>
Scalar
coordinateFactor(CoordinateSystem coord, const Scalar & r)
{
  switch (coord)
  {
  case CoordinateSystem::Cartesian:
    return Scalar(1.0);
  case CoordinateSystem::Axisymmetric:
    return 2.0 * M_PI * r;
  case CoordinateSystem::SphericalRadial:
    return 4.0 * M_PI * r * r;
  }
  return Scalar(1.0);
}

/// The factor of the volume element of the coordinate system at the point @p x, with r = x[0].
inline double
coordinateFactor(CoordinateSystem coord, const Point & x)
{
  return coordinateFactor(coord, x[0]);
}

/// The number of hoop directions of the coordinate system: 0 in Cartesian, 1 in axisymmetric and
/// 2 in spherical coordinates.  It multiplies every hoop term: the term n v_r / r of the
/// divergence, with n this number, and the hoop stresses n sigma_tt / r of the radial equation.
constexpr int
hoopFactor(CoordinateSystem coord)
{
  switch (coord)
  {
  case CoordinateSystem::Cartesian:
    return 0;
  case CoordinateSystem::Axisymmetric:
    return 1;
  case CoordinateSystem::SphericalRadial:
    return 2;
  }
  return 0;
}

/// The factor n / r of the hoop terms at the radius @p r, with n the hoop factor, and zero on the
/// axis (r = 0), where a hoop term takes its limit elsewhere.
inline double
hoopOverRadius(CoordinateSystem coord, double r)
{
  return r > 0.0 ? hoopFactor(coord) / r : 0.0;
}

/// The hoop term n v_r / r of the divergence of a vector of radial component @p v_r at the radius
/// @p r, with n the hoop factor: div v = dv_r/dr + n v_r / r in one dimension.  It is zero on the
/// axis r = 0 and in Cartesian coordinates.
template <typename Scalar>
Scalar
hoopDivergence(CoordinateSystem coord, const Scalar & v_r, double r)
{
  const int n = hoopFactor(coord);
  if (n == 0 || r == 0.0)
    return Scalar(0.0);
  return static_cast<double>(n) * v_r / r;
}

/// The hoop strain u_r / r of the radial displacement @p u_r at the radius @p r, the same in each
/// hoop direction.  On the axis r = 0 it is its limit du_r/dr, given as @p du_r_dr.
template <typename Scalar>
Scalar
hoopStrain(const Scalar & u_r, const Scalar & du_r_dr, double r)
{
  return r != 0.0 ? u_r / r : du_r_dr;
}

} // namespace dualmesh
