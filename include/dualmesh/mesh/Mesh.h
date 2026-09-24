// SPDX-License-Identifier: LGPL-2.1-or-later
//
// Unstructured primal mesh: nodes, elements, blocks (subdomains), side sets
// (boundary faces), and node sets.  The dual mesh of control domains is never
// stored explicitly; it is generated element by element from the reference
// element definitions (see ReferenceElement.h).
#pragma once

#include "dualmesh/core/Types.h"

#include <array>
#include <functional>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace dualmesh
{

/// The element types the library knows.
///
/// The linear family (Edge2, Tri3, Quad4, Tet4, Hex8, Wedge6) and the full
/// quadratic family (Edge3, Tri6, Quad9, Tet10, Hex27) carry a dual mesh and so
/// work with every discretisation.  The serendipity elements Quad8 and Hex20
/// and the pyramid Pyramid5 have no dual mesh and work with the finite element
/// method and the cell-centred finite volume method only; a Problem refuses
/// them for the other two methods and says why.  Node numbering follows VTK
/// for every type, so files written and read through VTK need no reordering.
enum class ElementType
{
  Edge2,
  Tri3,
  Quad4,
  Tet4,
  Hex8,
  Edge3,
  Tri6,
  Quad9,
  Tet10,
  Hex27,
  Quad8,
  Hex20,
  Wedge6,
  Pyramid5
};

/// The largest number of nodes any supported element has (Hex27).
inline constexpr int kMaxElementNodes = 27;

std::string elementTypeName(ElementType t);
ElementType elementTypeFromName(const std::string & name);
int elementNumNodes(ElementType t);
int elementDimension(ElementType t);

struct Element
{
  ElementType type;
  std::array<Index, kMaxElementNodes> nodes;
  int block = 0;
  int numNodes() const { return elementNumNodes(type); }
  /// The linear element with the same corners, used for geometric queries
  /// such as point location and for output formats that cannot store the
  /// higher-order nodes.
  ElementType cornerType() const;
  int numCorners() const { return elementNumNodes(cornerType()); }
};

/// Whether a type carries mid-edge, mid-face or interior nodes.
bool elementIsQuadratic(ElementType t);
/// The linear element with the same corners.
ElementType elementCornerType(ElementType t);
/// The quadratic element with the same corners, for mesh promotion.
ElementType elementQuadraticType(ElementType t);

/// The VTK cell type of an element type (VTK_LINE = 3, VTK_TRIANGLE = 5, and
/// so on), used by the VTU writer and by the meshio bridge.
int vtkCellType(ElementType t);
/// The permutation from VTK local node numbering to dualmesh local node
/// numbering: entry i is the dualmesh node that goes into VTK position i.
/// dualmesh numbers every element as VTK does, so this is the identity; it is
/// kept as the one place to change should a type ever need reordering.
const std::vector<int> & vtkNodeOrder(ElementType t);

/// A boundary side: (element index, local side index).
using Side = std::pair<Index, int>;

class Mesh
{
public:
  explicit Mesh(int dimension = 2) : _dim(dimension) {}

  // ---- construction ------------------------------------------------------
  Index addNode(const Point & p);
  Index addElement(ElementType type, const std::vector<Index> & nodes, int block = 0);
  void setBlockName(int block, const std::string & name) { _block_names[block] = name; }
  void setDimension(int d) { _dim = d; }

  /// Add (or extend) a side set from a list of boundary faces given by node ids.
  /// Faces not found on the boundary are reported as an error.
  void addSidesetFromFaces(const std::string & name, const std::vector<std::vector<Index>> & faces);
  void addSideset(const std::string & name, const std::vector<Side> & sides);
  void addNodeset(const std::string & name, const std::vector<Index> & nodes);
  /// Give an existing side set (and its nodes) an additional name.
  void aliasSideset(const std::string & existing, const std::string & new_name);
  /// All exterior sides whose centroid satisfies @p predicate become a side set.
  void addSidesetByPredicate(const std::string & name,
                             const std::function<bool(const Point &)> & predicate);
  /// All nodes that satisfy @p predicate become a node set.
  void addNodesetByPredicate(const std::string & name,
                             const std::function<bool(const Point &)> & predicate);
  /// Apply x -> f(x) to every node (e.g. to bend a rectangle into an annulus).
  void transformNodes(const std::function<Point(const Point &)> & f);
  /// Fix element orientation so that every Jacobian determinant is positive.
  void fixOrientation();
  /// Uniformly refine (every element is split into 2^dim children).
  Mesh refined() const;
  /// Return the same mesh with quadratic elements: Edge2 becomes Edge3, Tri3
  /// becomes Tri6, Quad4 becomes Quad9, Tet4 becomes Tet10 and Hex8 becomes
  /// Hex27.  With @p serendipity, Quad4 becomes Quad8 and Hex8 becomes Hex20
  /// instead, which work with the finite element and cell-centred finite
  /// volume methods only.  The corner nodes keep their numbers, the geometry is unchanged
  /// (the added nodes are placed at the midpoints of the edges, faces and
  /// cells they belong to), and side sets and node sets are carried over.
  Mesh secondOrder(bool serendipity = false) const;

  // ---- queries -----------------------------------------------------------
  int dimension() const { return _dim; }
  Index numNodes() const { return static_cast<Index>(_nodes.size()); }
  Index numElements() const { return static_cast<Index>(_elements.size()); }
  const Point & node(Index i) const { return _nodes[i]; }
  const std::vector<Point> & nodes() const { return _nodes; }
  const Element & element(Index e) const { return _elements[e]; }
  const std::vector<Element> & elements() const { return _elements; }
  const std::map<std::string, std::vector<Side>> & sidesets() const { return _sidesets; }
  const std::map<std::string, std::vector<Index>> & nodesets() const { return _nodesets; }
  const std::map<int, std::string> & blockNames() const { return _block_names; }
  std::vector<int> blockIds() const;
  /// Resolve a block given by name or by its integer id written as text.
  int blockId(const std::string & name) const;
  bool hasBoundary(const std::string & name) const;
  const std::vector<Side> & sideset(const std::string & name) const;
  /// Nodes of a node set, or of the sides of a side set with that name.
  std::vector<Index> boundaryNodes(const std::string & name) const;
  /// All exterior sides (computed on demand and cached).
  const std::vector<Side> & exteriorSides() const;
  /// One flag per node: is the node on the exterior boundary of the domain?
  const std::vector<char> & boundaryNodeMarkers() const;
  /// Global node ids of a side.
  std::vector<Index> sideNodes(const Side & s) const;
  Point elementCentroid(Index e) const;
  Point sideCentroid(const Side & s) const;
  /// Bounding box: {min, max}.
  std::pair<Point, Point> boundingBox() const;
  /// Build the default side sets (left/right/bottom/top/back/front) from the
  /// bounding box for meshes that are aligned with the coordinate axes.
  void addBoundingBoxSidesets(double tolerance = 1e-10);
  /// Build the node sets that are implied by every side set.
  std::string summary() const;

private:
  int _dim;
  std::vector<Point> _nodes;
  std::vector<Element> _elements;
  std::map<int, std::string> _block_names;
  std::map<std::string, std::vector<Side>> _sidesets;
  std::map<std::string, std::vector<Index>> _nodesets;
  mutable std::vector<Side> _exterior_cache;
  mutable bool _exterior_valid = false;
  mutable std::vector<char> _boundary_nodes;
};

// ---- mesh generators (MOOSE "GeneratedMesh" equivalents) -------------------

/// 1D mesh through the given, increasing node coordinates.
/// Side sets: "left", "right".
Mesh generateLineMesh(const std::vector<double> & x);

/// 2D structured mesh on the tensor grid x (columns) by y (rows).
/// @param element "Quad4" or "Tri3".
/// @param diagonal for triangles: "right" (/), "left" (\), "alternate", or "union"-free crossed not
/// supported. Side sets: "left", "right", "bottom", "top".
Mesh generateRectangleMesh(const std::vector<double> & x,
                           const std::vector<double> & y,
                           const std::string & element = "Quad4",
                           const std::string & diagonal = "right");

/// 3D structured mesh on the tensor grid x by y by z.
/// @param element "Hex8" or "Tet4" (each hexahedron split into 6 tetrahedra).
/// Side sets: "left", "right", "bottom", "top", "back", "front".
Mesh generateBoxMesh(const std::vector<double> & x,
                     const std::vector<double> & y,
                     const std::vector<double> & z,
                     const std::string & element = "Hex8");

} // namespace dualmesh
