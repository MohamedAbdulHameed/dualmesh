// SPDX-License-Identifier: LGPL-2.1-or-later
//
// What the hierarchies of local refinement and coarsening share: the nodes under fixed numbers,
// the side sets and node sets, the mesh built from the current elements, the projection of a side
// set onto a curved boundary, and the transfer of nodal values to the mesh after a change.
// Bisection (triangles and tetrahedra) and Subdivision (quadrilaterals and hexahedra, with hanging
// nodes) give the elements and how they split and merge.
//
// The hierarchy keeps its nodes under fixed numbers.  A node of the new mesh that the old one had
// keeps its values, and a new node takes the value of the interpolant of the old element it lies
// in, which carries the fields of the elements exactly.
#pragma once

#include "dualmesh/mesh/Mesh.h"

#include <Eigen/Core>

#include <functional>
#include <initializer_list>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace dualmesh
{

class MeshHierarchy
{
public:
  virtual ~MeshHierarchy() = default;

  /// The current mesh.
  const Mesh & mesh() const { return _mesh; }

  /// Refine the elements marked in @p marked (one entry per element) that have fewer than
  /// @p max_level refinements, and the elements that the conformity or the balance needs.
  /// Return whether the mesh changed.
  bool refine(const std::vector<char> & marked, int max_level);
  /// Coarsen every family whose elements @p marked all marks (the patch of a bisection, the
  /// children of a split), where the hierarchy allows it.  Return whether the mesh changed.
  bool coarsen(const std::vector<char> & marked);
  /// Coarsen where @p coarsen marks a whole family that @p refine does not mark, then refine as
  /// refine() does, in one change of the mesh: transfer() carries values across both.  Return
  /// whether the mesh changed.
  bool adapt(const std::vector<char> & refine, const std::vector<char> & coarsen, int max_level);

  /// Move the nodes of side set @p name onto a curved boundary: each takes the point that
  /// @p projection gives for it, except a hanging node, which stays on the straight side of its
  /// large neighbour until that side is split.  The moved positions stay, so the nodes that later
  /// refinements put on the side start from them.  It changes no numbering, so transfer() still
  /// carries values from the mesh before.
  void project(const std::string & name, const std::function<Point(const Point &)> & projection);

  /// The number of refinements of each element of the current mesh from its element of the
  /// initial mesh (bisections or splits), and that element.
  std::vector<int> levels() const;
  std::vector<Index> origins() const;

  /// Values at the nodes of the mesh before the last change, @p width per node, carried to the
  /// nodes of the current mesh: a node that was there keeps its values, a new node takes the value
  /// of the interpolant of the old element it lies in, and a removed node is dropped.
  Eigen::VectorXd transfer(const Eigen::VectorXd & before, int width) const;

  /// Values at the nodes of the mesh before the last change, @p width per node, carried to the
  /// nodes of the current mesh so that each keeps its content, the sum of its values times the
  /// measures @p measures of the nodes before the change (one per node).  A node that the
  /// coarsening removes lies in the merged element, whose shape functions take the weights P_mj at
  /// it: 1/2 at each end of an edge, 1/4 at each corner of a face and 1/2^d at each corner of a
  /// cell.  Node j of the coarsened mesh therefore takes the measure-weighted mean
  /// u_j = sum_i P_ij M_i u_i / sum_i P_ij M_i over the nodes i before, with P_ij = 1 for i = j.
  /// The refinement of the same change then interpolates the coarsened field, as transfer() does.
  /// The total sum_j M_j u_j stays where the measures of the coarsened mesh are the sums
  /// sum_i P_ij M_i, as the integrals of the shape functions are, and so the measures of the
  /// control domains of linear simplices, parallelograms and parallelepipeds where no node hangs.
  /// Linear elements only.
  Eigen::VectorXd
  restrictValues(const Eigen::VectorXd & before, int width, const Eigen::VectorXd & measures) const;

protected:
  /// An element of a mesh of the hierarchy: its element of the initial mesh, its block, its
  /// number of refinements, the record of the derived hierarchy it comes from (-1 when there is
  /// none), and its nodes in the fixed numbers.
  struct Placed
  {
    Index origin = 0;
    int block = 0;
    int level = 0;
    Index record = -1;
    std::vector<Index> nodes;
  };

  /// Capture the nodes, the blocks, the side sets and the node sets of @p initial, whose elements
  /// must all be of type @p type, one of @p types.  The derived constructor sets up its elements
  /// and then calls start().
  MeshHierarchy(const Mesh & initial, ElementType type, std::initializer_list<ElementType> types);
  /// Build the first mesh, which is also the mesh before.
  void start();
  /// A new node at @p position between the nodes @p parents.  It joins every node set of which all
  /// of them are members.
  Index addNode(const Point & position, const std::vector<Index> & parents);
  /// The key of a side: its corner nodes, sorted.
  static std::vector<Index> sideKey(std::vector<Index> corners);

  /// The elements of the current mesh.  It may add the middle nodes of quadratic elements.
  virtual std::vector<Placed> currentElements() = 0;
  /// Coarsen where @p coarsen marks (one mark per element of currentElements()), and give the
  /// marks of @p refine to the elements kept, none to the merged ones.  Record each removed node
  /// in _removed.  Return whether anything changed.
  virtual bool coarsenElements(const std::vector<char> & coarsen, std::vector<char> & refine) = 0;
  /// Refine where @p refine marks (one mark per element of currentElements()), below
  /// @p max_level.  Return whether anything changed.
  virtual bool refineElements(const std::vector<char> & refine, int max_level) = 0;
  /// The reference coordinates @p xi of node @p node in element @p element of a mesh of the
  /// hierarchy, of the same initial element, when the element holds the node.
  virtual bool locate(const Placed & element, Index node, Point & xi) const = 0;
  /// The constraints of the nodes of the current mesh that hang, in the numbers of its nodes,
  /// @p index of each fixed number (-1 for an unused one).
  virtual std::vector<Mesh::NodeConstraint> hangingConstraints(const std::vector<Index> & index)
  {
    (void) index;
    return {};
  }

  int _dimension = 2;
  int _space_dimension = 2;
  ElementType _type = ElementType::Tri3;
  /// Every node ever made, in its fixed number.
  std::vector<Point> _points;
  /// The side sets of every side of one, by its key, each with the block of the element whose
  /// side it is (a side between two blocks is a side of either).
  std::map<std::vector<Index>, std::vector<std::pair<std::string, int>>> _side_sets;
  /// The nodes that the coarsening of the last change removed, each with the nodes of the merged
  /// element and the values of their shape functions at it, for restrictValues().  The refinement
  /// of the same change may make some of them again.
  std::vector<std::pair<Index, std::vector<std::pair<Index, double>>>> _removed;

private:
  /// Rebuild the mesh from currentElements(), keeping the numbers of its nodes for transfer().
  void build();
  /// The end of a call that changed nothing: transfer() and restrictValues() then give the values
  /// back.
  bool unchanged();
  /// The values of the nodes of the current mesh, @p width per node, from @p values by the fixed
  /// node numbers: a node that @p known marks keeps its value, and any other takes the value of
  /// the interpolant of the element of @p elements, of its initial element, that holds it.
  Eigen::VectorXd carry(const Eigen::VectorXd & values,
                        const std::vector<char> & known,
                        const std::vector<Placed> & elements,
                        int width) const;

  std::map<int, std::string> _block_names;
  /// The number of corners of a side.
  int _side_corners = 2;
  /// The node sets: their names, whether each node is a member, whether every member of each
  /// node set of the initial mesh lies on its boundary, and the number of nodes of the initial
  /// mesh.
  std::vector<std::string> _node_set_names;
  std::vector<std::vector<char>> _in_node_set;
  std::vector<char> _boundary_node_set;
  Index _initial_nodes = 0;
  /// The elements of the current mesh, of the mesh before the last change, and of the mesh after
  /// the coarsening of the last change, which restrictValues() interpolates.
  std::vector<Placed> _placed, _placed_before, _coarsened;
  /// The fixed number of each node of the current mesh and of the mesh before the last change.
  std::vector<Index> _numbers, _numbers_before;
  Mesh _mesh;
};

} // namespace dualmesh
