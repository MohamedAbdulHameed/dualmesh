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
};

/// Evaluate the element map, shape functions and physical gradients at xi.
void mapPoint(const Mesh & mesh, const Element & el, const Point & xi, MappedPoint & mp);

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
bool
simplexQuadrature(int dim, int degree, std::vector<Point> & points, std::vector<double> & weights);

/// Measure (length/area/volume) of element e.
double elementMeasure(const Mesh & mesh, Index e);

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
