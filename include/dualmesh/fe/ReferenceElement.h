// SPDX-License-Identifier: LGPL-2.1-or-later
//
// Reference element definitions and the geometry of the dual mesh.
//
// The dual mesh is the "median dual" (box-method) construction.  Inside every
// primal element, the control domain (CD) of a node is the sub-cell bounded
// by the node, the midpoints of its edges, the centroids of its faces, and the
// element centroid.  The interface between the CDs of the two end nodes of an
// element edge (a, b) is the surface through the edge midpoint, the adjacent
// face centroids, and the element centroid.  For rectangles this reproduces
// the bisecting control domains of Reddy's book; for triangles and general
// quadrilaterals it reproduces the construction of Section 9.9 of the book.
//
// Every sub-cell and every interface is described as a tensor-product
// "patch" whose corners are given in reference coordinates of the parent
// element; physical integration uses the parent isoparametric map.
#pragma once

#include "dualmesh/mesh/Mesh.h"

#include <string>
#include <tuple>
#include <vector>

namespace dualmesh
{

/// A tensor-product parametric cell of dimension 0..3 with 2^dim corners,
/// expressed in reference coordinates of a parent element.  Corner ordering
/// follows the Edge2/Quad4/Hex8 reference node ordering.
struct Patch
{
  int dim = 0;
  std::vector<Point> corners;
  /// Map patch coordinates s in [-1,1]^dim to parent reference coordinates;
  /// also returns the tangent vectors dxi/ds_k (k < dim).
  Point map(const Point & s, std::array<Point, 3> * tangents = nullptr) const;
};

/// Interface between the control domains of nodes a and b inside an element.
struct DualFace
{
  int a, b;
  Patch patch;
};

class ReferenceElement
{
public:
  static const ReferenceElement & get(ElementType type);

  ElementType type() const { return _type; }
  int dimension() const { return _dim; }
  int numNodes() const { return static_cast<int>(_nodes.size()); }
  const Point & node(int i) const { return _nodes[i]; }
  const Point & centroid() const { return _centroid; }

  /// Shape functions and their reference gradients at xi.
  void shape(const Point & xi, double * N, Point * dN) const;

  /// Sides: local node lists (in outward-consistent order) and their type.
  int numSides() const { return static_cast<int>(_sides.size()); }
  const std::vector<int> & sideNodes(int s) const { return _sides[s]; }
  /// Type of a side; for Edge2 the sides are points (returned as Edge2 with
  /// the flag sideIsPoint() set).
  bool sideIsPoint() const { return _dim == 1; }
  ElementType sideType() const { return _side_type; }

  /// Dual mesh geometry.
  const Patch & controlVolume(int node) const { return _scv[node]; }
  const std::vector<DualFace> & dualFaces() const { return _faces; }
  /// Sub-patch of side @p s that belongs to the control domain of the
  /// side's local node @p j (in parent reference coordinates).
  const Patch & sideControlPatch(int s, int j) const { return _side_patches[s][j]; }
  /// The whole side as a patch (FEM boundary integration).  For simplicial
  /// sides this is a list of sub-patches (the union covers the side).
  const std::vector<Patch> & sidePatches(int s) const { return _side_full[s]; }
  /// The whole element as a list of patches (FEM volume integration).
  const std::vector<Patch> & elementPatches() const { return _elem_full; }
  bool isSimplex() const { return _type == ElementType::Tri3 || _type == ElementType::Tet4; }

private:
  explicit ReferenceElement(ElementType type);
  void buildDualGeometry(const std::vector<std::pair<int, int>> & edges);

  ElementType _type;
  int _dim;
  std::vector<Point> _nodes;
  Point _centroid;
  std::vector<std::vector<int>> _sides;
  ElementType _side_type;
  std::vector<Patch> _scv;
  std::vector<DualFace> _faces;
  std::vector<std::vector<Patch>> _side_patches;
  std::vector<std::vector<Patch>> _side_full;
  std::vector<Patch> _elem_full;
};

/// One-dimensional quadrature rules on [-1, 1] and tensor products.
struct QuadratureSpec
{
  enum class Kind
  {
    Gauss,                  ///< Gauss-Legendre with `points` points per direction
    Trapezoid,              ///< corners of the cell
    Simpson,                ///< Simpson's one-third rule per direction
    Nodal,                  ///< one point at the owning node (lumped); volume terms only
    ControlDomainTrapezoid, ///< trapezoidal rule over the whole control
                            ///< domain: in one dimension this is the classical
                            ///< finite-volume source rule
                            ///< F_I = (dx/2) [f(x_A) + f(x_B)], with the
                            ///< half-domains at boundary nodes handled
                            ///< consistently. Volume terms only.
    Interface               ///< one point at the far corner of the control volume (the
                            ///< control-domain interface in 1D, the element centroid in 2D
                            ///< and 3D); reproduces the trapezoidal source rule of the
                            ///< finite volume literature. Volume terms only.
  };
  Kind kind = Kind::Gauss;
  int points = 2;
  /// Evaluate the solution at the element centroid (selective reduced
  /// integration, used to avoid shear/membrane locking and for penalty terms).
  bool reduced = false;

  static QuadratureSpec parse(const std::string & text, bool reduced = false);
  std::string str() const;
  bool operator<(const QuadratureSpec & o) const
  {
    return std::tie(kind, points, reduced) < std::tie(o.kind, o.points, o.reduced);
  }
  bool operator==(const QuadratureSpec & o) const
  {
    return kind == o.kind && points == o.points && reduced == o.reduced;
  }
};

/// Points and weights on [-1,1].
void rule1D(const QuadratureSpec & spec, std::vector<double> & x, std::vector<double> & w);
/// Tensor-product rule on [-1,1]^dim (dim = 0 gives a single point, weight 1).
void
ruleTensor(const QuadratureSpec & spec, int dim, std::vector<Point> & x, std::vector<double> & w);

} // namespace dualmesh
