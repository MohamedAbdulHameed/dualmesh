// SPDX-License-Identifier: LGPL-2.1-or-later
//
// Local refinement and coarsening of a mesh of quadrilaterals (Quad4) or hexahedra (Hex8) by
// subdivision, with hanging nodes.
//
// A cell splits into 2^d children of half its size, through the middles of its edges, the centres
// of its faces and its centre.  A neighbour that is not split keeps its side whole, and the node in
// the middle of that side hangs: it is a corner of the small cells and lies inside the side of the
// large one.  The mesh is kept 2:1 balanced across edges and faces, as deal.II, MFEM and p4est
// keep theirs: a cell is split when a neighbour across one of its edges or faces is more than one
// level finer, so a node hangs on one level only.  A hanging node takes the value of the large
// side, which is that of its bilinear (trilinear) interpolant there: the mean of the two ends of a
// hanging edge, and of the four corners of a hanging face.  hanging() gives those constraints.
//
// The coarsening merges a family of 2^d leaves back into their parent when all are marked and the
// parent keeps the balance.  A new node is placed by the bilinear (trilinear) map of the corners of
// its parent as they are, so the nodes that project() moves onto a curved boundary carry over to
// the cells that later splits make.
#pragma once

#include "dualmesh/mesh/MeshHierarchy.h"

#include <array>
#include <map>
#include <utility>
#include <vector>

namespace dualmesh
{

class Subdivision : public MeshHierarchy
{
public:
  /// The hierarchy of the mesh @p initial of quadrilaterals (Quad4) or hexahedra (Hex8).  The
  /// level of an element is its number of splits.
  explicit Subdivision(const Mesh & initial);

  /// A hanging node of the current mesh and the nodes whose values give its value, with their
  /// weights, in the numbers of the nodes of the current mesh.
  struct Hanging
  {
    Index node = 0;
    std::vector<std::pair<Index, double>> on;
  };
  const std::vector<Hanging> & hanging() const { return _hanging; }

private:
  using Edge = std::pair<Index, Index>;
  static Edge key(Index a, Index b) { return a < b ? Edge{a, b} : Edge{b, a}; }
  /// A side of a cell, by its sorted corners: two in 2D, four in 3D.
  using Face = std::vector<Index>;

  struct Cell
  {
    /// The corners, in the order of the nodes of Quad4 or Hex8, in the fixed node numbers.
    std::vector<Index> n;
    int block = 0;
    int level = 0;
    Index origin = 0;
    /// The box of the cell in the reference coordinates of its initial cell.
    Point low{-1, -1, -1}, high{1, 1, 1};
    Index parent = -1;
    std::vector<Index> children;
    bool alive = true;
    bool leaf = true;
  };

  std::vector<Placed> currentElements() override;
  bool coarsenElements(const std::vector<char> & coarsen, std::vector<char> & refine) override;
  bool refineElements(const std::vector<char> & refine, int max_level) override;
  bool locate(const Placed & element, Index node, Point & xi) const override;
  std::vector<Mesh::NodeConstraint> hangingConstraints(const std::vector<Index> & index) override;

  /// The live leaves, in the order of the elements of the current mesh.
  std::vector<Index> leaves() const;
  /// The node at the middle of @p edge, and at the centre of a face with the corners @p face.
  Index middle(const Edge & edge, const Point & position);
  Index centre(const Face & face, const Point & position);
  /// The point of @p cell at its reference coordinates @p eta, by its corners as they are now.
  Point place(const Cell & cell, const Point & eta) const;
  /// Split leaf @p c into its children.
  void split(Index c);
  /// The leaves whose split the 2:1 balance needs.
  std::vector<Index> unbalanced() const;
  /// Whether a node is a corner of a live leaf.
  std::vector<char> usedNodes() const;
  /// The steps of refineElements() and coarsenElements(), with one mark per cell.
  bool refineCells(const std::vector<char> & marked_cells, int max_level);
  bool coarsenCells(const std::vector<char> & marked_cells);

  /// The corner pairs of the edges and the corner sets of the faces of a cell, by local number.
  std::vector<std::array<int, 2>> _edges;
  std::vector<std::vector<int>> _faces;
  /// The reference coordinates of the corners of the reference cell.
  std::vector<Point> _corner_xi;
  std::map<Edge, Index> _middles;
  std::map<Face, Index> _centres;
  /// The reference coordinates of every node in the initial cells whose closure holds it.
  std::vector<std::map<Index, Point>> _origin_reference;
  std::vector<Cell> _cells;
  std::vector<Hanging> _hanging;
};

} // namespace dualmesh
