// SPDX-License-Identifier: LGPL-2.1-or-later
#include "dualmesh/mesh/MeshHierarchy.h"

#include "dualmesh/core/InputParameters.h"
#include "dualmesh/fe/ReferenceElement.h"

#include <algorithm>
#include <array>

namespace dualmesh
{

MeshHierarchy::MeshHierarchy(const Mesh & initial,
                             ElementType type,
                             std::initializer_list<ElementType> types)
    : _dimension(initial.dimension()), _space_dimension(initial.spaceDimension()), _type(type),
      _block_names(initial.blockNames()), _mesh(initial.dimension())
{
  bool fits = std::find(types.begin(), types.end(), type) != types.end() &&
              ReferenceElement::get(type).dimension() == _dimension;
  for (const auto & el : initial.elements())
    fits = fits && el.type == type;
  if (!fits)
    throw InputError(
        "Local refinement and coarsening need a mesh of one element type: Tri3, Tri6, Tet4 or "
        "Tet10, refined by bisection, or Quad4 or Hex8, refined by subdivision.");
  _side_corners = _dimension == 2 ? 2 : (elementNumNodes(elementCornerType(type)) == 4 ? 3 : 4);
  _points = initial.nodes();
  for (const auto & [name, sides] : initial.sidesets())
    for (const Side & side : sides)
    {
      std::vector<Index> corners = initial.sideNodes(side);
      corners.resize(static_cast<std::size_t>(_side_corners));
      _side_sets[sideKey(corners)].push_back({name, initial.element(side.first).block});
    }
  _initial_nodes = initial.numNodes();
  const std::vector<char> & on_boundary = initial.boundaryNodeMarkers();
  for (const auto & [name, members] : initial.nodesets())
  {
    _node_set_names.push_back(name);
    std::vector<char> in(_points.size(), 0);
    bool boundary = !members.empty();
    for (Index n : members)
    {
      in[static_cast<std::size_t>(n)] = 1;
      boundary = boundary && on_boundary[static_cast<std::size_t>(n)];
    }
    _in_node_set.push_back(std::move(in));
    _boundary_node_set.push_back(boundary);
  }
}

void
MeshHierarchy::start()
{
  build();
  unchanged();
}

std::vector<Index>
MeshHierarchy::sideKey(std::vector<Index> corners)
{
  std::sort(corners.begin(), corners.end());
  return corners;
}

Index
MeshHierarchy::addNode(const Point & position, const std::vector<Index> & parents)
{
  const Index n = static_cast<Index>(_points.size());
  _points.push_back(position);
  for (auto & in : _in_node_set)
  {
    bool all = true;
    for (Index p : parents)
      all = all && in[static_cast<std::size_t>(p)];
    in.push_back(all);
  }
  return n;
}

bool
MeshHierarchy::unchanged()
{
  _numbers_before = _numbers;
  _placed_before = _placed;
  _coarsened = _placed;
  _removed.clear();
  return false;
}

bool
MeshHierarchy::refine(const std::vector<char> & marked, int max_level)
{
  return adapt(marked, std::vector<char>(marked.size(), 0), max_level);
}

bool
MeshHierarchy::coarsen(const std::vector<char> & marked)
{
  return adapt(std::vector<char>(marked.size(), 0), marked, 0);
}

bool
MeshHierarchy::adapt(const std::vector<char> & refine,
                     const std::vector<char> & coarsen,
                     int max_level)
{
  if (refine.size() != _placed.size() || coarsen.size() != _placed.size())
    throw InputError("The refinement and coarsening markers must have one entry per element.");
  _removed.clear();
  // An element marked for both is refined.
  std::vector<char> only_coarsen(coarsen.size());
  for (std::size_t e = 0; e < coarsen.size(); ++e)
    only_coarsen[e] = coarsen[e] && !refine[e];
  std::vector<char> marks = refine;
  const bool coarsened = coarsenElements(only_coarsen, marks);
  _coarsened = currentElements();
  const bool refined = refineElements(marks, max_level);
  if (!coarsened && !refined)
    return unchanged();
  build();
  return true;
}

void
MeshHierarchy::project(const std::string & name,
                       const std::function<Point(const Point &)> & projection)
{
  const auto found = _mesh.sidesets().find(name);
  if (found == _mesh.sidesets().end())
    throw InputError("There is no side set '" + name + "' to move onto its boundary.");
  // A hanging node stays on the straight side of its large neighbour, where the constraint of its
  // value puts it.  It moves onto the curve at the projection after that side is split.
  std::vector<char> done(static_cast<std::size_t>(_mesh.numNodes()), 0);
  for (const auto & c : _mesh.nodeConstraints())
    done[static_cast<std::size_t>(c.node)] = 1;
  for (const Side & side : found->second)
    for (Index i : _mesh.sideNodes(side))
    {
      if (done[static_cast<std::size_t>(i)])
        continue;
      done[static_cast<std::size_t>(i)] = 1;
      Point & x = _points[static_cast<std::size_t>(_numbers[static_cast<std::size_t>(i)])];
      x = projection(x);
    }
  // The same mesh with the moved positions, and the same mesh before the last change.
  const std::vector<Index> numbers_before = _numbers_before;
  const std::vector<Placed> placed_before = _placed_before;
  build();
  _numbers_before = numbers_before;
  _placed_before = placed_before;
}

std::vector<int>
MeshHierarchy::levels() const
{
  std::vector<int> out;
  for (const Placed & element : _placed)
    out.push_back(element.level);
  return out;
}

std::vector<Index>
MeshHierarchy::origins() const
{
  std::vector<Index> out;
  for (const Placed & element : _placed)
    out.push_back(element.origin);
  return out;
}

void
MeshHierarchy::build()
{
  _numbers_before = _numbers;
  _placed_before = std::move(_placed);
  _placed = currentElements();
  std::vector<char> used(_points.size(), 0);
  for (const Placed & element : _placed)
    for (Index n : element.nodes)
      used[static_cast<std::size_t>(n)] = 1;
  _numbers.clear();
  std::vector<Index> index(_points.size(), -1);
  Mesh out(_dimension);
  out.setSpaceDimension(_space_dimension);
  for (const auto & [block, name] : _block_names)
    out.setBlockName(block, name);
  for (Index n = 0; n < static_cast<Index>(_points.size()); ++n)
    if (used[static_cast<std::size_t>(n)])
    {
      index[static_cast<std::size_t>(n)] = static_cast<Index>(_numbers.size());
      _numbers.push_back(n);
      out.addNode(_points[static_cast<std::size_t>(n)]);
    }
  for (const Placed & element : _placed)
  {
    std::vector<Index> nodes;
    for (Index n : element.nodes)
      nodes.push_back(index[static_cast<std::size_t>(n)]);
    out.addElement(_type, nodes, element.block);
  }
  // The sides of every side key: one on the boundary, two inside.  A side of a set between two
  // blocks is the side of the element of its block.
  std::map<std::vector<Index>, std::vector<Side>> sides;
  for (Index e = 0; e < out.numElements(); ++e)
    for (int k = 0; k < ReferenceElement::get(_type).numSides(); ++k)
    {
      std::vector<Index> corners = out.sideNodes({e, k});
      corners.resize(static_cast<std::size_t>(_side_corners));
      for (Index & n : corners)
        n = _numbers[static_cast<std::size_t>(n)];
      sides[sideKey(corners)].push_back({e, k});
    }
  std::map<std::string, std::vector<Side>> sets;
  for (const auto & [key, names] : _side_sets)
  {
    const auto found = sides.find(key);
    if (found == sides.end())
      continue;
    for (const auto & [name, block] : names)
    {
      Side chosen = found->second.front();
      for (const Side & side : found->second)
        if (out.element(side.first).block == block)
          chosen = side;
      sets[name].push_back(chosen);
    }
  }
  for (const auto & [name, list] : sets)
    out.addSideset(name, list);
  out.setNodeConstraints(hangingConstraints(index));
  // After the constraints, which decide which sides are on the boundary.  A new node joins a node
  // set on the boundary only where it lies on the boundary too, as Mesh::refined grows its sets:
  // the middle of an edge between two members may lie inside, on a chord across a corner.
  const std::vector<char> & boundary = out.boundaryNodeMarkers();
  for (std::size_t k = 0; k < _node_set_names.size(); ++k)
  {
    std::vector<Index> members;
    for (std::size_t i = 0; i < _numbers.size(); ++i)
    {
      const Index n = _numbers[i];
      if (!_in_node_set[k][static_cast<std::size_t>(n)])
        continue;
      if (_boundary_node_set[k] && n >= _initial_nodes && !boundary[i])
        continue;
      members.push_back(static_cast<Index>(i));
    }
    out.addNodeset(_node_set_names[k], members);
  }
  _mesh = std::move(out);
}

Eigen::VectorXd
MeshHierarchy::transfer(const Eigen::VectorXd & before, int width) const
{
  if (before.size() != static_cast<Index>(_numbers_before.size()) * width)
    throw InputError("The values to transfer have " + std::to_string(before.size()) +
                     " entries, and the mesh before the change has " +
                     std::to_string(_numbers_before.size()) + " nodes of " + std::to_string(width) +
                     " values.");
  Eigen::VectorXd values = Eigen::VectorXd::Zero(static_cast<Index>(_points.size()) * width);
  std::vector<char> known(_points.size(), 0);
  for (std::size_t i = 0; i < _numbers_before.size(); ++i)
  {
    values.segment(_numbers_before[i] * width, width) =
        before.segment(static_cast<Index>(i) * width, width);
    known[static_cast<std::size_t>(_numbers_before[i])] = 1;
  }
  return carry(values, known, _placed_before, width);
}

Eigen::VectorXd
MeshHierarchy::restrictValues(const Eigen::VectorXd & before,
                              int width,
                              const Eigen::VectorXd & measures) const
{
  if (_type != ElementType::Tri3 && _type != ElementType::Tet4 && _type != ElementType::Quad4 &&
      _type != ElementType::Hex8)
    throw InputError("DMCDM carries the fields to a new mesh by a conservative mean over the "
                     "control domains, which takes linear elements: Tri3, Tet4, Quad4 or Hex8. The "
                     "shape functions of a quadratic element are negative at some of the nodes "
                     "that a coarsening removes, which would give the mean negative weights.");
  const Index count = static_cast<Index>(_numbers_before.size());
  if (before.size() != count * width || measures.size() != count)
    throw InputError("The values to restrict have " + std::to_string(before.size()) +
                     " entries and the measures " + std::to_string(measures.size()) +
                     ", and the mesh before the change has " + std::to_string(count) +
                     " nodes of " + std::to_string(width) + " values.");
  std::vector<Index> position_before(_points.size(), -1);
  for (Index i = 0; i < count; ++i)
    position_before[static_cast<std::size_t>(_numbers_before[static_cast<std::size_t>(i)])] = i;
  std::vector<char> removed(_points.size(), 0);
  for (const auto & entry : _removed)
    removed[static_cast<std::size_t>(entry.first)] = 1;
  // The coarsening: a removed node gives its measure and its content to the nodes of its merged
  // element, each with the weight of its shape function there.  They stay: the ends of a
  // bisected edge, and the corners of a merged cell, which the 2:1 balance keeps from hanging.
  Eigen::VectorXd gained = Eigen::VectorXd::Zero(count);
  Eigen::VectorXd content = Eigen::VectorXd::Zero(count * width);
  for (const auto & [m, weights] : _removed)
  {
    const Index i = position_before[static_cast<std::size_t>(m)];
    for (const auto & [node, w] : weights)
    {
      const Index j = position_before[static_cast<std::size_t>(node)];
      if (i < 0 || j < 0 || removed[static_cast<std::size_t>(node)])
        throw std::logic_error("dualmesh: a node that the coarsening removed lies in a merged "
                               "element whose node is not a node of the coarsened mesh.");
      gained[j] += w * measures[i];
      content.segment(j * width, width) += w * measures[i] * before.segment(i * width, width);
    }
  }
  // The values of the coarsened mesh, by the fixed node numbers: the measure-weighted mean at a
  // node that gained content, and the value before at every other node.
  Eigen::VectorXd coarse = Eigen::VectorXd::Zero(static_cast<Index>(_points.size()) * width);
  std::vector<char> known(_points.size(), 0);
  for (Index i = 0; i < count; ++i)
  {
    const Index n = _numbers_before[static_cast<std::size_t>(i)];
    if (removed[static_cast<std::size_t>(n)])
      continue;
    const double total = measures[i] + gained[i];
    if (gained[i] > 0.0 && total > 0.0)
      coarse.segment(n * width, width) =
          (measures[i] * before.segment(i * width, width) + content.segment(i * width, width)) /
          total;
    else
      coarse.segment(n * width, width) = before.segment(i * width, width);
    known[static_cast<std::size_t>(n)] = 1;
  }
  // The refinement of the same change interpolates the coarsened field in the elements of the
  // coarsened mesh.
  return carry(coarse, known, _coarsened, width);
}

Eigen::VectorXd
MeshHierarchy::carry(const Eigen::VectorXd & values,
                     const std::vector<char> & known,
                     const std::vector<Placed> & elements,
                     int width) const
{
  std::map<Index, std::vector<const Placed *>> of_origin;
  for (const Placed & element : elements)
    of_origin[element.origin].push_back(&element);
  // An initial element of each node: that of an element of the current mesh that holds it.
  std::vector<Index> origin_of(_points.size(), -1);
  for (const Placed & element : _placed)
    for (Index n : element.nodes)
      origin_of[static_cast<std::size_t>(n)] = element.origin;
  const ReferenceElement & reference = ReferenceElement::get(_type);
  std::array<double, kMaxElementNodes> N{};
  std::array<Point, kMaxElementNodes> dN{};
  Eigen::VectorXd after(static_cast<Index>(_numbers.size()) * width);
  for (std::size_t i = 0; i < _numbers.size(); ++i)
  {
    const Index n = _numbers[i];
    if (known[static_cast<std::size_t>(n)])
    {
      after.segment(static_cast<Index>(i) * width, width) = values.segment(n * width, width);
      continue;
    }
    // The element of the same initial element that holds the node, and the value of its
    // interpolant there.
    bool found = false;
    for (const Placed * old : of_origin[origin_of[static_cast<std::size_t>(n)]])
    {
      Point xi{0, 0, 0};
      if (!locate(*old, n, xi))
        continue;
      reference.shape(xi, N.data(), dN.data());
      Eigen::VectorXd value = Eigen::VectorXd::Zero(width);
      for (std::size_t k = 0; k < old->nodes.size(); ++k)
        value += N[k] * values.segment(old->nodes[k] * width, width);
      after.segment(static_cast<Index>(i) * width, width) = value;
      found = true;
      break;
    }
    if (!found)
      throw std::logic_error("dualmesh: a new node lies in no element of the mesh before.");
  }
  return after;
}

} // namespace dualmesh
