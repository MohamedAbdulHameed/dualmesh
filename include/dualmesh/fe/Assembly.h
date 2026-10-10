// SPDX-License-Identifier: LGPL-2.1-or-later
//
// Integration points for the dual mesh control domain method and for the
// finite element method, and evaluation of the element map.
#pragma once

#include "dualmesh/fe/ReferenceElement.h"
#include "dualmesh/mesh/Mesh.h"

#include <vector>

namespace dualmesh
{

/// Geometry of the isoparametric map at one reference point.
struct MappedPoint
{
  Point xi{0, 0, 0};
  Point x{0, 0, 0};
  double N[kMaxElementNodes];
  Point dN[kMaxElementNodes]; ///< physical gradients of the shape functions
  double detJ = 0;
  double J[3][3];
  /// The linear shape functions of the element's corners and their physical
  /// gradients, for a variable interpolated at first order on a quadratic
  /// element (the pressure of a Taylor-Hood element).  They are filled by
  /// mapCornerShape; mapPoint resets `num_corners` to zero, so a stale set is never
  /// used by mistake.
  static constexpr int kMaxCorners = 8;
  int num_corners = 0;
  double corner_shape[kMaxCorners];
  Point corner_gradient[kMaxCorners];
};

/// Evaluate the element map, shape functions and physical gradients at xi.
void mapPoint(const Mesh & mesh, const Element & el, const Point & xi, MappedPoint & mp);

/// Add to a mapped point the shape functions of the linear element with the
/// same corners (Tri3 for Tri6, Quad4 for Quad9 and Quad8, Tet4 for Tet10, Hex8
/// for Hex27 and Hex20), evaluated at the same reference point.  Their physical
/// gradients use the Jacobian of the element's own (quadratic) map,
/// grad N^c = J^{-T} dN^c/dxi, so a first-order field on a curved quadratic
/// element is linear in the reference coordinates, which is the standard
/// Taylor-Hood construction.  The corners are the first nodes of every
/// quadratic element, in the order of the linear element, so corner a of the
/// linear element is local node a.
void mapCornerShape(const Element & el, MappedPoint & mp);

/// One integration point.
struct IntegrationPoint
{
  Point xi;       ///< reference point for the geometry and test functions
  Point xi_field; ///< reference point where the solution is evaluated
  double weight;  ///< |J| times quadrature weight (volume points)
  Point area;     ///< n dA, oriented from `owner` to `neighbor` / outward
  int owner;      ///< local node owning the point (-1: all nodes, FEM)
  int neighbor;   ///< the other node of a dual face (-1 otherwise)
};

enum class PointSet
{
  Volume, ///< DMCDM: control volumes; FEM: element interior
  Faces,  ///< DMCDM only: interfaces between control volumes
};

/// Build the integration points of element @p e.
///
/// Building a point maps it to the physical element, and a caller that needs
/// the mapped point (the shape functions and their physical gradients) can
/// ask for it through @p mapped instead of mapping the point a second time.
/// It is filled, one entry per point of @p out, for the Gauss-type rules on
/// the volume and on the dual faces; for the other rules it is left empty,
/// and the caller maps the points itself.
void buildElementPoints(const Mesh & mesh,
                        Index e,
                        bool dual_mesh,
                        PointSet set,
                        const QuadratureSpec & spec,
                        std::vector<IntegrationPoint> & out,
                        std::vector<MappedPoint> * mapped = nullptr);

/// Build the integration points of side @p side (boundary integrals).
void buildSidePoints(const Mesh & mesh,
                     const Side & side,
                     bool dual_mesh,
                     const QuadratureSpec & spec,
                     std::vector<IntegrationPoint> & out);

/// A symmetric quadrature rule on the reference triangle (dim 2) or
/// tetrahedron (dim 3) exact to at least @p degree, with points in reference
/// coordinates and weights that sum to the reference measure.  Returns false
/// when no tabulated rule reaches the degree.
/// The point of a side closest to @p x.
struct SideProjection
{
  Point xi{0, 0, 0}; ///< parent reference coordinates of the closest point
  Point x{0, 0, 0};  ///< its physical position
  double distance = 0.0;
};

/// Project @p x onto the side (element, local side) of the mesh: the closest
/// point of the side, found by a Gauss-Newton iteration in the side's own
/// coordinates, restricted to the side.
SideProjection projectOntoSide(const Mesh & mesh, const Side & side, const Point & x);

/// The local node of the element whose control domain contains the side point
/// with parent reference coordinates @p xi (dual mesh and vertex-centred
/// methods): the owner of the side control patch that contains it, or the
/// nearest one.
int sideControlOwner(const Mesh & mesh, const Side & side, const Point & xi);

bool
simplexQuadrature(int dim, int degree, std::vector<Point> & points, std::vector<double> & weights);

/// Measure (length/area/volume) of element e.
double elementMeasure(const Mesh & mesh, Index e);

/// The size h of element e for the stabilization parameters: the length of the side of a cube of the element's measure, corrected for simplices, prisms and pyramids so that the generators' elements of a grid of spacing h all get h, and halved for a quadratic element, whose interpolation resolves half the element.
double elementSize(const Mesh & mesh, Index e);

/// Measure (length in two dimensions, area in three) of a boundary side,
/// integrated with the isoparametric map, so that a curved quadratic side is
/// measured exactly.  In axisymmetric and spherical coordinates the measure
/// includes the coordinate factor (2 pi r and 4 pi r^2), i.e., it is the area
/// of the surface of revolution that the side represents.
double sideMeasure(const Mesh & mesh,
                   const Side & side,
                   CoordinateSystem coord = CoordinateSystem::Cartesian);

/// Total measure of the sides of the named side sets.  A side that belongs to
/// several of them is counted once.
double sidesetMeasure(const Mesh & mesh,
                      const std::vector<std::string> & names,
                      CoordinateSystem coord = CoordinateSystem::Cartesian);

/// Find the element containing x and its reference coordinates.
/// Returns false if no element contains the point.
class PointLocator
{
public:
  explicit PointLocator(const Mesh & mesh);
  bool locate(const Point & x, Index & element, Point & xi, double tolerance = 1e-8) const;

private:
  const Mesh & _mesh;
  Point _lo, _hi;
  int _n[3];
  std::vector<std::vector<Index>> _bins;
  int binIndex(int i, int j, int k) const { return (k * _n[1] + j) * _n[0] + i; }
};

} // namespace dualmesh
