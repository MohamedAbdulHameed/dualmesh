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

#include <array>
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
  /// Extra Gauss points per direction that integrating over this patch needs.
  /// A triangle, a tetrahedron, a prism or a pyramid is integrated as one
  /// square or cube with some corners collapsed onto each other (the
  /// collapsed-coordinate rule of Duffy, 1982).  The collapse puts a factor of
  /// degree dim - 1 into the Jacobian, so a Gauss rule exact to a given
  /// degree on the element needs one point more per direction on the
  /// collapsed patch; this field records that.
  int extra_points = 0;
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
  ///
  /// The control domain of a node inside one element is a list of patches
  /// rather than a single one.  For the linear and the tensor-product
  /// quadratic elements the list always holds exactly one patch.  For a
  /// quadratic simplex it holds several: the element is first split into
  /// linear sub-simplices, and the control domain of a node is the union of
  /// its median sub-cells in every sub-simplex that touches it.
  const std::vector<Patch> & controlVolume(int node) const { return _scv[node]; }
  const std::vector<DualFace> & dualFaces() const { return _faces; }
  /// Sub-patches of side @p s that belong to the control domain of the
  /// side's local node @p j (in parent reference coordinates).
  const std::vector<Patch> & sideControlPatch(int s, int j) const { return _side_patches[s][j]; }
  /// The whole side as a patch (FEM boundary integration).  For simplicial
  /// sides this is a list of sub-patches (the union covers the side).
  const std::vector<Patch> & sidePatches(int s) const { return _side_full[s]; }
  /// The whole element as a list of patches (FEM volume integration).
  const std::vector<Patch> & elementPatches() const { return _elem_full; }
  /// True for the triangles and tetrahedra, whose reference coordinates are
  /// barycentric.
  bool isSimplex() const
  {
    return _type == ElementType::Tri3 || _type == ElementType::Tet4 || _type == ElementType::Tri6 ||
           _type == ElementType::Tet10;
  }
  /// True when the nodes sit on a tensor-product grid of one-dimensional
  /// coordinates, which is the case for Edge2, Edge3, Quad4, Quad9, Hex8 and
  /// Hex27.  For these elements the shape functions are products of
  /// one-dimensional Lagrange polynomials and the dual mesh is a grid of
  /// boxes.
  bool isTensor() const { return !_grid_coordinates.empty(); }
  /// The polynomial order of the shape functions: 1 or 2.
  int order() const { return elementIsQuadratic(_type) ? 2 : 1; }
  /// Whether this element carries a dual mesh, which the dual mesh control
  /// domain method and the vertex-centred finite volume method need.  Every
  /// element supports the finite element method and the cell-centred finite
  /// volume method; Quad8, Hex20 and Pyramid5 support only those two.
  bool supportsDualMesh() const { return !_scv.empty(); }
  /// Why the element has no dual mesh, in a sentence that completes "this
  /// element cannot be used with the dual mesh methods because ...".  Empty
  /// when it has one.
  std::string dualMeshLimitation() const;
  /// Whether the reference point @p xi lies in the reference element, with a
  /// tolerance @p tol on every bounding face.
  bool contains(const Point & xi, double tol = 1e-10) const;

private:
  explicit ReferenceElement(ElementType type);
  /// The dual of an element whose nodes sit on a tensor grid: every node owns
  /// the box bounded by the planes half way to its neighbours in each
  /// direction.  @p coordinates are the one-dimensional grid coordinates,
  /// {-1, 1} for the linear elements and {-1, 0, 1} for the quadratic ones.
  void buildTensorDual(const std::vector<double> & coordinates);
  /// The dual of a simplex or of a quadratic simplex.  The element is first
  /// written as a set of linear sub-simplices (one for Tri3 and Tet4, four for
  /// Tri6 and eight for Tet10) and the median dual of every sub-simplex is
  /// accumulated onto its vertices.
  void buildSubdividedSimplexDual(const std::vector<std::vector<int>> & sub_simplices);
  /// The median dual of the triangular prism, built as the median dual of its
  /// triangle times the halves of its height.
  void buildPrismDual();
  /// Only the whole-element and whole-side patches, for the elements that have
  /// no dual mesh (Quad8, Hex20, Pyramid5).
  void buildElementGeometryOnly();
  /// Append to @p scv and @p faces the median sub-cells and the interfaces of
  /// one linear sub-simplex whose vertices are the local nodes @p verts.
  /// @p dim is the dimension of the sub-simplex, which is the element
  /// dimension for the volume dual and one less for a side.
  /// @p slots gives, for every vertex, the index in @p scv that its sub-cell
  /// is appended to; it is the vertex itself for the volume dual and the
  /// side-local position for a side.
  void addSimplexSubCell(const std::vector<int> & verts,
                         const std::vector<int> & slots,
                         int dim,
                         std::vector<std::vector<Patch>> & scv,
                         std::vector<DualFace> * faces) const;

  ElementType _type;
  int _dim;
  std::vector<Point> _nodes;
  Point _centroid;
  std::vector<std::vector<int>> _sides;
  ElementType _side_type;
  std::vector<std::vector<Patch>> _scv;
  std::vector<DualFace> _faces;
  std::vector<std::vector<std::vector<Patch>>> _side_patches;
  std::vector<std::vector<Patch>> _side_full;
  std::vector<Patch> _elem_full;
  /// Tensor elements only: the one-dimensional grid coordinates and, for every
  /// node, its index into that list in each direction.
  std::vector<double> _grid_coordinates;
  std::vector<std::array<int, 3>> _grid;
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
  /// Set by the rule name "automatic": the number of Gauss points is then
  /// chosen from the polynomial order of the elements in the mesh, two per
  /// direction for a linear mesh and three for a quadratic one.  The flag is
  /// cleared as soon as the choice has been made, which happens once, when the
  /// object is set up and the mesh is known.
  bool automatic = false;
  /// Evaluate the solution at the element centroid (selective reduced
  /// integration, used to avoid shear/membrane locking and for penalty terms).
  bool reduced = false;

  static QuadratureSpec parse(const std::string & text, bool reduced = false);
  std::string str() const;
  bool operator<(const QuadratureSpec & o) const
  {
    return std::tie(kind, points, reduced, automatic) <
           std::tie(o.kind, o.points, o.reduced, o.automatic);
  }
  bool operator==(const QuadratureSpec & o) const
  {
    return kind == o.kind && points == o.points && reduced == o.reduced && automatic == o.automatic;
  }
};

/// Points and weights on [-1,1].
void rule1D(const QuadratureSpec & spec, std::vector<double> & x, std::vector<double> & w);
/// Tensor-product rule on [-1,1]^dim (dim = 0 gives a single point, weight 1).
void
ruleTensor(const QuadratureSpec & spec, int dim, std::vector<Point> & x, std::vector<double> & w);

} // namespace dualmesh
