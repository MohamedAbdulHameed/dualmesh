// SPDX-License-Identifier: LGPL-2.1-or-later
#include "dualmesh/mesh/Subdivision.h"

#include "dualmesh/core/InputParameters.h"
#include "dualmesh/fe/ReferenceElement.h"

#include <algorithm>
#include <cmath>

namespace dualmesh
{

namespace
{
/// The face of @p corners, sorted, as the key of a side.
std::vector<Index>
sortedFace(std::vector<Index> corners)
{
  std::sort(corners.begin(), corners.end());
  return corners;
}
} // namespace

Subdivision::Subdivision(const Mesh & initial)
    : _dimension(initial.dimension()), _space_dimension(initial.spaceDimension()),
      _type(initial.dimension() == 3 ? ElementType::Hex8 : ElementType::Quad4),
      _block_names(initial.blockNames()), _mesh(initial.dimension())
{
  for (const auto & el : initial.elements())
    if (el.type != _type || (_dimension != 2 && _dimension != 3))
      throw InputError("Local refinement and coarsening by subdivision need a mesh of "
                       "quadrilaterals (Quad4) or of hexahedra (Hex8).");
  const ReferenceElement & reference = ReferenceElement::get(_type);
  const int corners = 1 << _dimension;
  for (int k = 0; k < corners; ++k)
    _corner_xi.push_back(reference.node(k));
  // The edges join corners that differ in one reference coordinate; a face holds the corners
  // that share one.
  for (int a = 0; a < corners; ++a)
    for (int b = a + 1; b < corners; ++b)
    {
      int differ = 0;
      for (int d = 0; d < _dimension; ++d)
        differ += _corner_xi[a][d] != _corner_xi[b][d];
      if (differ == 1)
        _edges.push_back({a, b});
    }
  if (_dimension == 2)
    for (const auto & e : _edges)
      _faces.push_back({e[0], e[1]});
  else
    for (int d = 0; d < 3; ++d)
      for (const double value : {-1.0, 1.0})
      {
        std::vector<int> face;
        for (int k = 0; k < corners; ++k)
          if (_corner_xi[k][d] == value)
            face.push_back(k);
        _faces.push_back(face);
      }

  _points = initial.nodes();
  for (const auto & el : initial.elements())
  {
    Cell cell;
    cell.n.assign(el.nodes.begin(), el.nodes.end());
    cell.block = el.block;
    cell.origin = static_cast<Index>(_cells.size());
    if (_dimension == 2)
      cell.low[2] = cell.high[2] = 0.0;
    _origin_nodes.push_back(cell.n);
    std::map<Index, Point> xi;
    for (int k = 0; k < corners; ++k)
      xi[cell.n[static_cast<std::size_t>(k)]] = _corner_xi[k];
    _origin_reference.push_back(std::move(xi));
    _cells.push_back(cell);
  }
  for (const auto & [name, sides] : initial.sidesets())
    for (const Side & side : sides)
    {
      std::vector<Index> nodes = initial.sideNodes(side);
      nodes.resize(static_cast<std::size_t>(1 << (_dimension - 1)));
      _face_sets[sortedFace(nodes)].push_back({name, initial.element(side.first).block});
    }
  for (const auto & [name, members] : initial.nodesets())
  {
    _node_set_names.push_back(name);
    std::vector<char> in(_points.size(), 0);
    for (Index n : members)
      in[static_cast<std::size_t>(n)] = 1;
    _in_node_set.push_back(std::move(in));
  }
  build();
  _numbers_before = _numbers;
  _leaves_before = _leaves;
}

Point
Subdivision::place(Index origin, const Point & xi) const
{
  std::array<double, 8> N{};
  std::array<Point, 8> dN{};
  ReferenceElement::get(_type).shape(xi, N.data(), dN.data());
  Point x{0, 0, 0};
  const auto & corners = _origin_nodes[static_cast<std::size_t>(origin)];
  for (std::size_t k = 0; k < corners.size(); ++k)
    x = x + N[k] * _points[static_cast<std::size_t>(corners[k])];
  return x;
}

Index
Subdivision::middle(const Edge & edge, const Point & position)
{
  const auto found = _middles.find(edge);
  if (found != _middles.end())
    return found->second;
  const Index m = static_cast<Index>(_points.size());
  _points.push_back(position);
  for (auto & in : _in_node_set)
    in.push_back(in[static_cast<std::size_t>(edge.first)] &&
                 in[static_cast<std::size_t>(edge.second)]);
  _middles[edge] = m;
  return m;
}

Index
Subdivision::centre(const Face & face, const Point & position)
{
  const auto found = _centres.find(face);
  if (found != _centres.end())
    return found->second;
  const Index c = static_cast<Index>(_points.size());
  _points.push_back(position);
  for (auto & in : _in_node_set)
  {
    bool all = true;
    for (Index n : face)
      all = all && in[static_cast<std::size_t>(n)];
    in.push_back(all);
  }
  _centres[face] = c;
  return c;
}

void
Subdivision::split(Index c)
{
  const int d = _dimension;
  const int corners = 1 << d;
  const int size = d == 3 ? 27 : 9;
  // The nodes of the lattice of 3^d points: the corners, the middles of the edges, the centres of
  // the faces and the centre.  Lattice point t (each coordinate 0, 1 or 2) spans the corners whose
  // lattice coordinates agree with t wherever t is 0 or 2.
  std::vector<Index> lattice(static_cast<std::size_t>(size), -1);
  const Cell parent = _cells[static_cast<std::size_t>(c)];
  for (int p = 0; p < size; ++p)
  {
    const int t[3] = {p % 3, (p / 3) % 3, p / 9};
    Point xi{0, 0, 0};
    for (int a = 0; a < d; ++a)
      xi[a] = parent.low[a] + 0.5 * t[a] * (parent.high[a] - parent.low[a]);
    std::vector<Index> spanned;
    for (int k = 0; k < corners; ++k)
    {
      bool agrees = true;
      for (int a = 0; a < d; ++a)
        if (t[a] != 1)
          agrees = agrees && static_cast<int>(_corner_xi[k][a] + 1.0) == t[a];
      if (agrees)
        spanned.push_back(parent.n[static_cast<std::size_t>(k)]);
    }
    Index node;
    if (spanned.size() == 1)
      node = spanned[0];
    else if (spanned.size() == 2)
      node = middle(key(spanned[0], spanned[1]), place(parent.origin, xi));
    else if (spanned.size() == 4 && d == 3)
      node = centre(sortedFace(spanned), place(parent.origin, xi));
    else
    {
      node = static_cast<Index>(_points.size());
      _points.push_back(place(parent.origin, xi));
      for (auto & in : _in_node_set)
      {
        bool all = true;
        for (Index n : spanned)
          all = all && in[static_cast<std::size_t>(n)];
        in.push_back(all);
      }
    }
    lattice[static_cast<std::size_t>(p)] = node;
    _origin_reference[static_cast<std::size_t>(parent.origin)].emplace(node, xi);
  }
  // Child (c0, c1, c2) takes, for its corner k, the lattice point c + (xi_k + 1) / 2.
  std::vector<Index> children;
  for (int child = 0; child < corners; ++child)
  {
    const int offset[3] = {child & 1, (child >> 1) & 1, (child >> 2) & 1};
    Cell cell;
    cell.block = parent.block;
    cell.level = parent.level + 1;
    cell.origin = parent.origin;
    cell.parent = c;
    for (int a = 0; a < 3; ++a)
    {
      const double half = 0.5 * (parent.high[a] - parent.low[a]);
      cell.low[a] = a < d ? parent.low[a] + offset[a] * half : parent.low[a];
      cell.high[a] = a < d ? cell.low[a] + half : parent.high[a];
    }
    for (int k = 0; k < corners; ++k)
    {
      int p = 0, stride = 1;
      for (int a = 0; a < d; ++a)
      {
        p += (offset[a] + static_cast<int>((_corner_xi[k][a] + 1.0) / 2.0)) * stride;
        stride *= 3;
      }
      cell.n.push_back(lattice[static_cast<std::size_t>(p)]);
    }
    children.push_back(static_cast<Index>(_cells.size()));
    _cells.push_back(cell);
  }
  // A side of a set splits into the sides of the children on it: those of the same local face
  // of the children on that side of the parent.
  for (std::size_t f = 0; f < _faces.size(); ++f)
  {
    std::vector<Index> nodes;
    for (int k : _faces[f])
      nodes.push_back(parent.n[static_cast<std::size_t>(k)]);
    const auto found = _face_sets.find(sortedFace(nodes));
    if (found == _face_sets.end())
      continue;
    const auto sets = found->second;
    _face_sets.erase(found);
    // The axis and the side of the face.
    int axis = 0;
    for (int a = 0; a < d; ++a)
    {
      bool same = true;
      for (int k : _faces[f])
        same = same && _corner_xi[k][a] == _corner_xi[_faces[f][0]][a];
      if (same)
        axis = a;
    }
    const int side = _corner_xi[_faces[f][0]][axis] > 0 ? 1 : 0;
    for (int child = 0; child < corners; ++child)
      if (((child >> axis) & 1) == side)
      {
        std::vector<Index> child_nodes;
        for (int k : _faces[f])
          child_nodes.push_back(
              _cells[static_cast<std::size_t>(children[static_cast<std::size_t>(child)])]
                  .n[static_cast<std::size_t>(k)]);
        _face_sets[sortedFace(child_nodes)] = sets;
      }
  }
  Cell & stored = _cells[static_cast<std::size_t>(c)];
  stored.children = children;
  stored.leaf = false;
}

std::vector<char>
Subdivision::usedNodes() const
{
  std::vector<char> used(_points.size(), 0);
  for (const Cell & cell : _cells)
    if (cell.alive && cell.leaf)
      for (Index n : cell.n)
        used[static_cast<std::size_t>(n)] = 1;
  return used;
}

std::vector<Index>
Subdivision::unbalanced() const
{
  // A leaf is more than one level coarser than a neighbour when the neighbour uses a node at a
  // quarter of one of its edges, or, in 3D, at the middle of a line from an edge middle to the
  // centre of one of its faces.
  const std::vector<char> used = usedNodes();
  const auto usedMiddle = [&](Index a, Index b)
  {
    const auto found = _middles.find(key(a, b));
    return found != _middles.end() && used[static_cast<std::size_t>(found->second)];
  };
  const auto middleOf = [&](Index a, Index b)
  {
    const auto found = _middles.find(key(a, b));
    return found == _middles.end() ? Index(-1) : found->second;
  };
  std::vector<Index> out;
  for (Index c = 0; c < static_cast<Index>(_cells.size()); ++c)
  {
    const Cell & cell = _cells[static_cast<std::size_t>(c)];
    if (!cell.alive || !cell.leaf)
      continue;
    bool split = false;
    for (const auto & e : _edges)
    {
      const Index a = cell.n[static_cast<std::size_t>(e[0])],
                  b = cell.n[static_cast<std::size_t>(e[1])];
      const Index m = middleOf(a, b);
      if (m >= 0 && (usedMiddle(a, m) || usedMiddle(m, b)))
        split = true;
    }
    if (_dimension == 3)
      for (const auto & face : _faces)
      {
        std::vector<Index> nodes;
        for (int k : face)
          nodes.push_back(cell.n[static_cast<std::size_t>(k)]);
        const auto found = _centres.find(sortedFace(nodes));
        if (found == _centres.end())
          continue;
        for (const auto & e : _edges)
        {
          const bool on_face = std::find(face.begin(), face.end(), e[0]) != face.end() &&
                               std::find(face.begin(), face.end(), e[1]) != face.end();
          if (!on_face)
            continue;
          const Index m = middleOf(cell.n[static_cast<std::size_t>(e[0])],
                                   cell.n[static_cast<std::size_t>(e[1])]);
          if (m >= 0 && usedMiddle(m, found->second))
            split = true;
        }
      }
    if (split)
      out.push_back(c);
  }
  return out;
}

bool
Subdivision::refineCells(const std::vector<char> & marked_cells, int max_level)
{
  std::vector<Index> todo;
  for (Index c = 0; c < static_cast<Index>(_cells.size()); ++c)
  {
    const Cell & cell = _cells[static_cast<std::size_t>(c)];
    if (cell.alive && cell.leaf && marked_cells[static_cast<std::size_t>(c)] &&
        cell.level < max_level)
      todo.push_back(c);
  }
  if (todo.empty())
    return false;
  // Split, then split the leaves that the balance needs, until it holds.  Each round splits leaves
  // coarser than a neighbour, so the rounds end.
  while (!todo.empty())
  {
    for (Index c : todo)
      if (_cells[static_cast<std::size_t>(c)].leaf)
        split(c);
    todo = unbalanced();
  }
  return true;
}

bool
Subdivision::coarsenCells(const std::vector<char> & marked_cells)
{
  const std::vector<char> used = usedNodes();
  const auto usedMiddle = [&](Index a, Index b)
  {
    const auto found = _middles.find(key(a, b));
    return found != _middles.end() && used[static_cast<std::size_t>(found->second)];
  };
  bool changed = false;
  const Index count = static_cast<Index>(_cells.size());
  for (Index p = 0; p < count; ++p)
  {
    Cell & parent = _cells[static_cast<std::size_t>(p)];
    if (!parent.alive || parent.leaf || parent.children.empty())
      continue;
    bool merge = true;
    for (Index child : parent.children)
    {
      const Cell & cell = _cells[static_cast<std::size_t>(child)];
      merge = merge && cell.alive && cell.leaf && marked_cells[static_cast<std::size_t>(child)];
    }
    if (!merge)
      continue;
    // The parent keeps the balance when no neighbour uses a node at a quarter of one of its edges,
    // nor, in 3D, at the middle of a line from an edge middle to a face centre.
    for (const auto & e : _edges)
    {
      const Index a = parent.n[static_cast<std::size_t>(e[0])],
                  b = parent.n[static_cast<std::size_t>(e[1])];
      const Index m = _middles.at(key(a, b));
      if (usedMiddle(a, m) || usedMiddle(m, b))
        merge = false;
    }
    if (_dimension == 3)
      for (const auto & face : _faces)
      {
        std::vector<Index> nodes;
        for (int k : face)
          nodes.push_back(parent.n[static_cast<std::size_t>(k)]);
        const Index centre_node = _centres.at(sortedFace(nodes));
        for (const auto & e : _edges)
          if (std::find(face.begin(), face.end(), e[0]) != face.end() &&
              std::find(face.begin(), face.end(), e[1]) != face.end() &&
              usedMiddle(_middles.at(key(parent.n[static_cast<std::size_t>(e[0])],
                                         parent.n[static_cast<std::size_t>(e[1])])),
                         centre_node))
            merge = false;
      }
    if (!merge)
      continue;
    // The sides of a set on a side of the parent merge back into it.
    for (const auto & face : _faces)
    {
      std::vector<Index> nodes;
      for (int k : face)
        nodes.push_back(parent.n[static_cast<std::size_t>(k)]);
      std::vector<std::pair<std::string, int>> sets;
      for (Index child : parent.children)
      {
        std::vector<Index> child_nodes;
        for (int k : face)
          child_nodes.push_back(
              _cells[static_cast<std::size_t>(child)].n[static_cast<std::size_t>(k)]);
        const auto found = _face_sets.find(sortedFace(child_nodes));
        if (found == _face_sets.end())
          continue;
        sets = found->second;
        _face_sets.erase(found);
      }
      if (!sets.empty())
        _face_sets[sortedFace(nodes)] = sets;
    }
    for (Index child : parent.children)
      _cells[static_cast<std::size_t>(child)].alive = false;
    parent.leaf = true;
    changed = true;
  }
  return changed;
}

bool
Subdivision::refine(const std::vector<char> & marked, int max_level)
{
  if (marked.size() != _leaves.size())
    throw InputError("The refinement marker must have one entry per element.");
  std::vector<char> cells(_cells.size(), 0);
  for (std::size_t e = 0; e < _leaves.size(); ++e)
    cells[static_cast<std::size_t>(_leaves[e])] = marked[e];
  if (!refineCells(cells, max_level))
    return false;
  build();
  return true;
}

bool
Subdivision::coarsen(const std::vector<char> & marked)
{
  if (marked.size() != _leaves.size())
    throw InputError("The coarsening marker must have one entry per element.");
  std::vector<char> cells(_cells.size(), 0);
  for (std::size_t e = 0; e < _leaves.size(); ++e)
    cells[static_cast<std::size_t>(_leaves[e])] = marked[e];
  if (!coarsenCells(cells))
    return false;
  build();
  return true;
}

bool
Subdivision::adapt(const std::vector<char> & refine,
                   const std::vector<char> & coarsen,
                   int max_level)
{
  if (refine.size() != _leaves.size() || coarsen.size() != _leaves.size())
    throw InputError("The refinement and coarsening markers must have one entry per element.");
  // A leaf marked for both is refined.
  std::vector<char> to_refine(_cells.size(), 0), to_coarsen(_cells.size(), 0);
  for (std::size_t e = 0; e < _leaves.size(); ++e)
  {
    to_refine[static_cast<std::size_t>(_leaves[e])] = refine[e];
    to_coarsen[static_cast<std::size_t>(_leaves[e])] = coarsen[e] && !refine[e];
  }
  const bool coarsened = coarsenCells(to_coarsen);
  to_refine.resize(_cells.size(), 0);
  const bool refined = refineCells(to_refine, max_level);
  if (!coarsened && !refined)
    return false;
  build();
  return true;
}

std::vector<int>
Subdivision::levels() const
{
  std::vector<int> out;
  for (Index c : _leaves)
    out.push_back(_cells[static_cast<std::size_t>(c)].level);
  return out;
}

std::vector<Index>
Subdivision::origins() const
{
  std::vector<Index> out;
  for (Index c : _leaves)
    out.push_back(_cells[static_cast<std::size_t>(c)].origin);
  return out;
}

void
Subdivision::build()
{
  _numbers_before = _numbers;
  _leaves_before = _leaves;
  _leaves.clear();
  for (Index c = 0; c < static_cast<Index>(_cells.size()); ++c)
    if (_cells[static_cast<std::size_t>(c)].alive && _cells[static_cast<std::size_t>(c)].leaf)
      _leaves.push_back(c);
  const std::vector<char> used = usedNodes();
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
  for (Index c : _leaves)
  {
    const Cell & cell = _cells[static_cast<std::size_t>(c)];
    std::vector<Index> nodes;
    for (Index n : cell.n)
      nodes.push_back(index[static_cast<std::size_t>(n)]);
    out.addElement(_type, nodes, cell.block);
  }
  // The sides of the sets, the side of the element of its block where a side is between two.
  std::map<Face, std::vector<Side>> sides;
  for (Index e = 0; e < out.numElements(); ++e)
    for (std::size_t f = 0; f < _faces.size(); ++f)
    {
      std::vector<Index> nodes = out.sideNodes({e, static_cast<int>(f)});
      nodes.resize(static_cast<std::size_t>(1 << (_dimension - 1)));
      for (Index & n : nodes)
        n = _numbers[static_cast<std::size_t>(n)];
      sides[sortedFace(nodes)].push_back({e, static_cast<int>(f)});
    }
  std::map<std::string, std::vector<Side>> sets;
  for (const auto & [face, names] : _face_sets)
  {
    const auto found = sides.find(face);
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
  for (std::size_t k = 0; k < _node_set_names.size(); ++k)
  {
    std::vector<Index> members;
    for (std::size_t i = 0; i < _numbers.size(); ++i)
      if (_in_node_set[k][static_cast<std::size_t>(_numbers[i])])
        members.push_back(static_cast<Index>(i));
    out.addNodeset(_node_set_names[k], members);
  }
  // The hanging nodes: a used middle of an edge of a leaf, and in 3D a used centre of a face of a
  // leaf, take the mean of the ends of the edge or the corners of the face.
  std::map<Index, Hanging> hanging;
  for (Index c : _leaves)
  {
    const Cell & cell = _cells[static_cast<std::size_t>(c)];
    for (const auto & e : _edges)
    {
      const Index a = cell.n[static_cast<std::size_t>(e[0])],
                  b = cell.n[static_cast<std::size_t>(e[1])];
      const auto found = _middles.find(key(a, b));
      if (found == _middles.end() || !used[static_cast<std::size_t>(found->second)])
        continue;
      Hanging h;
      h.node = index[static_cast<std::size_t>(found->second)];
      h.on = {{index[static_cast<std::size_t>(a)], 0.5}, {index[static_cast<std::size_t>(b)], 0.5}};
      hanging[h.node] = h;
    }
    if (_dimension == 3)
      for (const auto & face : _faces)
      {
        std::vector<Index> nodes;
        for (int k : face)
          nodes.push_back(cell.n[static_cast<std::size_t>(k)]);
        const auto found = _centres.find(sortedFace(nodes));
        if (found == _centres.end() || !used[static_cast<std::size_t>(found->second)])
          continue;
        Hanging h;
        h.node = index[static_cast<std::size_t>(found->second)];
        for (Index n : nodes)
          h.on.push_back({index[static_cast<std::size_t>(n)], 0.25});
        hanging[h.node] = h;
      }
  }
  _hanging.clear();
  std::vector<Mesh::NodeConstraint> constraints;
  for (const auto & [node, h] : hanging)
  {
    _hanging.push_back(h);
    constraints.push_back({h.node, h.on});
  }
  out.setNodeConstraints(std::move(constraints));
  _mesh = std::move(out);
}

Eigen::VectorXd
Subdivision::transfer(const Eigen::VectorXd & before, int width) const
{
  if (before.size() != static_cast<Index>(_numbers_before.size()) * width)
    throw InputError("The values to transfer have " + std::to_string(before.size()) +
                     " entries, and the mesh before the change has " +
                     std::to_string(_numbers_before.size()) + " nodes of " + std::to_string(width) +
                     " values.");
  std::vector<Index> position_before(_points.size(), -1);
  for (std::size_t i = 0; i < _numbers_before.size(); ++i)
    position_before[static_cast<std::size_t>(_numbers_before[i])] = static_cast<Index>(i);
  std::map<Index, std::vector<Index>> old_of_origin;
  for (Index c : _leaves_before)
    old_of_origin[_cells[static_cast<std::size_t>(c)].origin].push_back(c);
  std::vector<Index> origin_of(_points.size(), -1);
  for (Index c : _leaves)
    for (Index n : _cells[static_cast<std::size_t>(c)].n)
      origin_of[static_cast<std::size_t>(n)] = _cells[static_cast<std::size_t>(c)].origin;
  const ReferenceElement & reference = ReferenceElement::get(_type);
  Eigen::VectorXd after(static_cast<Index>(_numbers.size()) * width);
  for (std::size_t i = 0; i < _numbers.size(); ++i)
  {
    const Index n = _numbers[i];
    if (position_before[static_cast<std::size_t>(n)] >= 0)
    {
      after.segment(static_cast<Index>(i) * width, width) =
          before.segment(position_before[static_cast<std::size_t>(n)] * width, width);
      continue;
    }
    // The old leaf of the same initial cell whose box holds the node, in the reference
    // coordinates of the initial cell, and the value of its interpolant there.
    const Index origin = origin_of[static_cast<std::size_t>(n)];
    const Point & x = _origin_reference[static_cast<std::size_t>(origin)].at(n);
    bool found = false;
    for (Index c : old_of_origin[origin])
    {
      const Cell & old = _cells[static_cast<std::size_t>(c)];
      Point xi{0, 0, 0};
      bool inside = true;
      for (int a = 0; a < _dimension; ++a)
      {
        const double width_a = old.high[a] - old.low[a];
        xi[a] = 2.0 * (x[a] - old.low[a]) / width_a - 1.0;
        inside = inside && std::abs(xi[a]) <= 1.0 + 1e-10;
      }
      if (!inside)
        continue;
      std::array<double, 8> N{};
      std::array<Point, 8> dN{};
      reference.shape(xi, N.data(), dN.data());
      Eigen::VectorXd value = Eigen::VectorXd::Zero(width);
      for (std::size_t k = 0; k < old.n.size(); ++k)
        value += N[k] *
                 before.segment(position_before[static_cast<std::size_t>(old.n[k])] * width, width);
      after.segment(static_cast<Index>(i) * width, width) = value;
      found = true;
      break;
    }
    if (!found)
      throw std::logic_error("dualmesh: a new node lies in no cell of the mesh before.");
  }
  return after;
}

} // namespace dualmesh
