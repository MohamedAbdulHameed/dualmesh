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
// parent keeps the balance.  The hierarchy keeps its nodes under fixed numbers, and a new node
// takes the value of the interpolant of the old cell it lies in, which carries the fields of the
// elements exactly.  A new node is placed by the bilinear (trilinear) map of the corners of its
// parent as they are, so the nodes that project() moves onto a curved boundary carry over to the
// cells that later splits make.
#pragma once

#include "dualmesh/mesh/Mesh.h"

#include <Eigen/Core>

#include <array>
#include <functional>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace dualmesh
{

class Subdivision
{
public:
  /// The hierarchy of the mesh @p initial of quadrilaterals (Quad4) or hexahedra (Hex8).
  explicit Subdivision(const Mesh & initial);

  /// The current mesh: the leaves of the hierarchy.
  const Mesh & mesh() const { return _mesh; }

  /// Split the cells marked in @p marked (one entry per element) below @p max_level, and those that
  /// the balance needs.  Return whether the mesh changed.
  bool refine(const std::vector<char> & marked, int max_level);
  /// Merge every family whose leaves are all marked, where the parent keeps the balance.
  bool coarsen(const std::vector<char> & marked);
  /// Coarsen where @p coarsen marks a whole family that @p refine does not mark, then refine, in
  /// one change of the mesh: transfer() carries values across both.
  bool adapt(const std::vector<char> & refine, const std::vector<char> & coarsen, int max_level);

  /// Move the nodes of side set @p name onto a curved boundary: each takes the point that
  /// @p projection gives for it, except a hanging node, which stays on the straight side of its
  /// large neighbour until that side is split.  It changes no numbering, so transfer() still
  /// carries values from the mesh before.
  void project(const std::string & name, const std::function<Point(const Point &)> & projection);

  /// The level of each element of the current mesh (0 for a cell of the initial mesh), and the
  /// element of the initial mesh it comes from.
  std::vector<int> levels() const;
  std::vector<Index> origins() const;

  /// A hanging node of the current mesh and the nodes whose values give its value, with their
  /// weights, in the numbers of the nodes of the current mesh.
  struct Hanging
  {
    Index node = 0;
    std::vector<std::pair<Index, double>> on;
  };
  const std::vector<Hanging> & hanging() const { return _hanging; }

  /// Values at the nodes of the mesh before the last change, @p width per node, carried to the
  /// nodes of the current mesh: a node that was there keeps its values, and a new node takes the
  /// value of the interpolant of the old cell it lies in.
  Eigen::VectorXd transfer(const Eigen::VectorXd & before, int width) const;

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
  /// The corner pairs of the edges and the corner sets of the faces of a cell, by local number.
  const std::vector<std::array<int, 2>> & edges() const { return _edges; }
  const std::vector<std::vector<int>> & faces() const { return _faces; }
  void build();
  /// The end of a call that changed nothing: transfer() then gives the values back.
  bool unchanged();
  /// Add the node sets to @p out, whose elements and constraints are set.
  void addNodeSets(Mesh & out) const;
  bool refineCells(const std::vector<char> & marked_cells, int max_level);
  bool coarsenCells(const std::vector<char> & marked_cells);

  int _dimension = 2;
  int _space_dimension = 2;
  ElementType _type = ElementType::Quad4;
  std::map<int, std::string> _block_names;
  std::vector<std::array<int, 2>> _edges;
  std::vector<std::vector<int>> _faces;
  /// The reference coordinates of the corners of the reference cell.
  std::vector<Point> _corner_xi;
  std::vector<Point> _points;
  std::map<Edge, Index> _middles;
  std::map<Face, Index> _centres;
  /// The reference coordinates of every node in the initial cells whose closure holds it.
  std::vector<std::map<Index, Point>> _origin_reference;
  std::vector<Cell> _cells;
  /// The cells of the elements of the current mesh, and of the mesh before the last change.
  std::vector<Index> _leaves, _leaves_before;
  std::map<Face, std::vector<std::pair<std::string, int>>> _face_sets;
  std::vector<std::string> _node_set_names;
  std::vector<std::vector<char>> _in_node_set;
  /// Whether every member of each node set of the initial mesh lies on its boundary, and the
  /// number of nodes of the initial mesh.
  std::vector<char> _boundary_node_set;
  Index _initial_nodes = 0;
  std::vector<Index> _numbers, _numbers_before;
  std::vector<Hanging> _hanging;
  Mesh _mesh;
};

} // namespace dualmesh
