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
#pragma once

#include "dualmesh/mesh/MeshHierarchy.h"

#include <array>
#include <map>
#include <utility>
#include <vector>

namespace dualmesh
{

class Bisection : public MeshHierarchy
{
public:
  /// The hierarchy of the mesh @p initial of triangles (Tri3 or Tri6), which take their longest
  /// edge as their first refinement edge, or of tetrahedra (Tet4 or Tet10).  The level of an
  /// element is its number of bisections.
  explicit Bisection(const Mesh & initial);

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

  std::vector<Placed> currentElements() override;
  bool coarsenElements(const std::vector<char> & coarsen, std::vector<char> & refine) override;
  bool refineElements(const std::vector<char> & refine, int max_level) override;
  bool locate(const Placed & element, Index node, Point & xi) const override;

  /// The steps of refineElements() and coarsenElements() on the tetrahedra and on the triangles.
  /// The coarsening gives the marks of @p carried to the elements it keeps, and none to the merged.
  bool refineTetrahedra(const std::vector<char> & marked, int max_generation);
  bool coarsenTetrahedra(const std::vector<char> & marked, std::vector<char> & carried);
  bool refineTriangles(const std::vector<char> & marked, int max_generation);
  bool coarsenTriangles(const std::vector<char> & marked, std::vector<char> & carried);
  /// The node at the middle of @p edge of an element of initial element @p origin, made when it
  /// is not there yet.
  Index middle(const Edge & edge, Index origin);
  /// The coordinates of node @p n in the reference element of initial element @p origin, which
  /// holds it on its closure.
  Point reference(Index n, Index origin) const;
  /// The nodes of an element of the current mesh: its corners, then the middles of its edges in
  /// the order of the quadratic element.
  std::vector<Index> elementNodes(const Index * corners, Index origin);
  /// Record in _removed the middle node @p m of its edge, which the coarsening removed.
  void recordRemoval(Index m);

  /// The nodes of every element of the initial mesh, and the reference coordinates of its nodes.
  std::vector<std::vector<Index>> _origin_nodes;
  std::vector<std::map<Index, Point>> _origin_reference;
  /// The edge that each node is the middle of ({-1, -1} for a node of the initial mesh).
  std::vector<Edge> _parent_edge;
  std::map<Edge, Index> _middles;
  std::vector<Triangle> _triangles;
  std::vector<Tetrahedron> _tetrahedra;
};

} // namespace dualmesh
