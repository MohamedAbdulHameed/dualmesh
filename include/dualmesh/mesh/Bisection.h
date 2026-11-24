// SPDX-License-Identifier: LGPL-2.1-or-later
//
// Local refinement and coarsening of a mesh of triangles by newest-vertex bisection, and of a mesh
// of tetrahedra by longest-edge bisection.
//
// Bisection keeps the mesh conforming, so no node hangs.  Newest-vertex bisection keeps the
// triangles in finitely many similarity classes however often they are bisected, so they cannot
// degenerate, and it is the refinement of the proofs that adaptive methods converge at the optimal
// rate (Binev, Dahmen and DeVore, Numer. Math. 97 (2004) 219-268, and Stevenson, Found. Comput.
// Math. 7 (2007) 245-269).
//
// Each triangle lists its refinement edge first and its newest vertex last, [a, b, c].  Its
// bisection puts the node m at the middle of the refinement edge a-b and gives the children
// [c, a, m] and [b, c, m], whose refinement edges are c-a and b-c and whose newest vertex is m.
// The refinement follows Funken, Praetorius and Wissgott (Comput. Methods Appl. Math. 11 (2011)
// 460-490): the marked triangles mark their refinement edges, every triangle with a marked edge
// marks its refinement edge too, and each triangle splits by the pattern of its marked edges, so
// that every marked edge has one middle node on both of its sides.  The coarsening undoes one
// bisection: a node made by bisection goes when it is the newest vertex of every triangle around
// it, there are two of them (on the boundary) or four, and all are marked, and the pairs merge
// into their parents.
//
// A tetrahedral mesh is refined by longest-edge bisection (Rivara and Levin, Commun. Appl. Numer.
// Methods 8 (1992) 281-290): a marked tetrahedron bisects its longest edge, and so does every
// tetrahedron around that edge, each through the same middle node.  A tetrahedron around the edge
// whose longest edge is another is bisected first, across its own longest edge.  Each such detour
// bisects a strictly longer edge, so the recursion ends for any initial mesh.  Ties of length go
// to the larger node numbers, so that all agree.  The coarsening undoes one bisection: a node made
// by bisection goes when every tetrahedron around it was made by that bisection and all are
// marked, and the pairs that share a face through the node merge into their parents.
//
// Quadratic triangles and tetrahedra (Tri6, Tet10) are refined by their corners.  The middle node
// of an edge is the node that its bisection makes, and every node is placed by the quadratic map of
// the element of the initial mesh it lies in, so that new nodes follow a curved boundary.
//
// The hierarchy keeps its nodes under fixed numbers.  A node of the new mesh that the old one had
// keeps its values, and a new node takes the value of the interpolant of the old element it lies
// in, which carries linear and quadratic fields exactly.
#pragma once

#include "dualmesh/mesh/Mesh.h"

#include <Eigen/Core>

#include <algorithm>
#include <array>
#include <functional>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace dualmesh
{

class Bisection
{
public:
  /// The hierarchy of the mesh @p initial of triangles (Tri3 or Tri6), which take their longest
  /// edge as their first refinement edge, or of tetrahedra (Tet4 or Tet10).
  explicit Bisection(const Mesh & initial);

  /// The current mesh.
  const Mesh & mesh() const { return _mesh; }

  /// Refine the triangles of the current mesh marked in @p marked (one entry per element), those
  /// of fewer than @p max_generation bisections, and the triangles that the conformity needs.
  /// Return whether the mesh changed.
  bool refine(const std::vector<char> & marked, int max_generation);

  /// Undo one bisection wherever every triangle of its patch is marked in @p marked.  Return
  /// whether the mesh changed.
  bool coarsen(const std::vector<char> & marked);

  /// Coarsen where @p coarsen marks every triangle of a patch that @p refine does not mark, then
  /// refine as refine() does, in one change of the mesh: transfer() carries values across both.
  /// Return whether the mesh changed.
  bool
  adapt(const std::vector<char> & refine, const std::vector<char> & coarsen, int max_generation);

  /// Move the nodes of side set @p name onto a curved boundary: each takes the point that
  /// @p projection gives for it.  The moved positions stay, so the nodes that later bisections put
  /// on the side start from them, and moving again after each refinement keeps the side on the
  /// curve.  It changes no numbering, so transfer() still carries values from the mesh before.
  void project(const std::string & name, const std::function<Point(const Point &)> & projection);

  /// The number of bisections of each element of the current mesh from its triangle of the
  /// initial mesh.
  std::vector<int> generations() const;
  /// The element of the initial mesh that each element of the current mesh comes from.
  std::vector<Index> origins() const;

  /// Values at the nodes of the mesh before the last change, @p width per node, carried to the
  /// nodes of the current mesh: a node that was there keeps its values, a new node takes the value
  /// of the interpolant of the old element it lies in (exact for the fields of the elements), and
  /// a removed node is dropped.
  Eigen::VectorXd transfer(const Eigen::VectorXd & before, int width) const;

private:
  using Edge = std::pair<Index, Index>;
  static Edge key(Index a, Index b) { return a < b ? Edge{a, b} : Edge{b, a}; }

  struct Triangle
  {
    /// The refinement edge n[0]-n[1] and the newest vertex n[2], in the fixed node numbers.
    std::array<Index, 3> n;
    int block = 0;
    int generation = 0;
    /// The element of the initial mesh that the triangle comes from.
    Index origin = 0;
  };

  struct Tetrahedron
  {
    /// The nodes, in the fixed node numbers and the orientation of the initial mesh.
    std::array<Index, 4> n;
    int block = 0;
    int generation = 0;
    Index origin = 0;
    /// The middle nodes of the bisections that made it, the last one the newest.
    std::vector<Index> births;
  };
  using Face = std::array<Index, 3>;
  static Face faceKey(Index a, Index b, Index c)
  {
    Face f{a, b, c};
    std::sort(f.begin(), f.end());
    return f;
  }
  /// The number of elements of the current mesh.
  std::size_t numElements() const
  {
    return _dimension == 3 ? _tetrahedra.size() : _triangles.size();
  }
  /// The steps of refine() and coarsen() on the tetrahedra.
  bool refineTetrahedra(const std::vector<char> & marked, int max_generation);
  bool coarsenTetrahedra(const std::vector<char> & marked, std::vector<char> * carried = nullptr);
  /// The steps of refine() and coarsen() on the triangles, without rebuilding the mesh.  The
  /// coarsening gives the marks of @p carried to the triangles it keeps, and none to the merged.
  bool refineTriangles(const std::vector<char> & marked, int max_generation);
  bool coarsenTriangles(const std::vector<char> & marked, std::vector<char> * carried = nullptr);
  /// The node at the middle of @p edge of an element of initial element @p origin, made when it
  /// is not there yet.
  Index middle(const Edge & edge, Index origin);
  /// The coordinates of node @p n in the reference element of initial element @p origin, which
  /// holds it on its closure.
  Point reference(Index n, Index origin) const;
  /// The nodes of an element of the current mesh: its corners, then the middles of its edges in
  /// the order of the quadratic element.
  std::vector<Index> elementNodes(const Index * corners, Index origin);
  /// Rebuild the mesh from the triangles, keeping the numbers of its nodes for transfer().
  void build();
  /// The end of a call that changed nothing: transfer() then gives the values back.
  bool unchanged();
  /// Add the node sets to @p out, whose elements and constraints are set.
  void addNodeSets(Mesh & out) const;

  /// An element of a mesh: its initial element and its nodes in the fixed numbers.
  struct Placed
  {
    Index origin = 0;
    std::vector<Index> nodes;
  };

  int _dimension = 2;
  ElementType _type = ElementType::Tri3;
  /// The nodes of every element of the initial mesh, and the reference coordinates of its nodes.
  std::vector<std::vector<Index>> _origin_nodes;
  std::vector<std::map<Index, Point>> _origin_reference;
  /// The elements of the current mesh and of the mesh before the last change.
  std::vector<Placed> _placed, _placed_before;
  int _space_dimension = 2;
  std::map<int, std::string> _block_names;
  /// Every node ever made, in its fixed number, the edge it is the middle of ({-1, -1} for a node
  /// of the initial mesh), and whether it is a node of the current mesh.
  std::vector<Point> _points;
  std::vector<Edge> _parent_edge;
  std::vector<char> _live;
  std::map<Edge, Index> _middles;
  std::vector<Triangle> _triangles;
  std::vector<Tetrahedron> _tetrahedra;
  /// The side sets of every face of a tetrahedral mesh that is a side of one, as _edge_sets.
  std::map<Face, std::vector<std::pair<std::string, int>>> _face_sets;
  /// The side sets of every edge that is a side of one, each with the block of the element whose
  /// side it is (an edge between two blocks is a side of either), and the node sets of every node.
  std::map<Edge, std::vector<std::pair<std::string, int>>> _edge_sets;
  std::vector<std::string> _node_set_names;
  std::vector<std::vector<char>> _in_node_set;
  /// Whether every member of each node set of the initial mesh lies on its boundary, and the
  /// number of nodes of the initial mesh.
  std::vector<char> _boundary_node_set;
  Index _initial_nodes = 0;
  /// The fixed number of each node of the current mesh and of the mesh before the last change.
  std::vector<Index> _numbers, _numbers_before;
  Mesh _mesh;
};

} // namespace dualmesh
