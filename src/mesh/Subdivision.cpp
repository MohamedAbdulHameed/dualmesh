// SPDX-License-Identifier: LGPL-2.1-or-later
#include "dualmesh/mesh/Subdivision.h"

#include "dualmesh/core/InputParameters.h"
#include "dualmesh/mesh/ReferenceElement.h"

#include <algorithm>
#include <cmath>

namespace dualmesh
{

Subdivision::Subdivision(const Mesh & initial)
    : MeshHierarchy(initial,
                    initial.dimension() == 3 ? ElementType::Hex8 : ElementType::Quad4,
                    {ElementType::Quad4, ElementType::Hex8})
{
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
  for (const auto & el : initial.elements())
  {
    Cell cell;
    cell.n.assign(el.nodes.begin(), el.nodes.begin() + el.numNodes());
    cell.block = el.block;
    cell.origin = static_cast<Index>(_cells.size());
    if (_dimension == 2)
      cell.low[2] = cell.high[2] = 0.0;
    std::map<Index, Point> xi;
    for (int k = 0; k < corners; ++k)
      xi[cell.n[static_cast<std::size_t>(k)]] = _corner_xi[k];
    _origin_reference.push_back(std::move(xi));
    _cells.push_back(cell);
  }
  start();
}

Point
Subdivision::place(const Cell & cell, const Point & eta) const
{
  std::array<double, 8> N{};
  std::array<Point, 8> dN{};
  ReferenceElement::get(_type).shape(eta, N.data(), dN.data());
  Point x{0, 0, 0};
  for (std::size_t k = 0; k < cell.n.size(); ++k)
    x = x + N[k] * _points[static_cast<std::size_t>(cell.n[k])];
  return x;
}

Index
Subdivision::middle(const Edge & edge, const Point & position)
{
  const auto found = _middles.find(edge);
  if (found != _middles.end())
    return found->second;
  const Index m = addNode(position, {edge.first, edge.second});
  _middles[edge] = m;
  return m;
}

Index
Subdivision::centre(const Face & face, const Point & position)
{
  const auto found = _centres.find(face);
  if (found != _centres.end())
    return found->second;
  const Index c = addNode(position, face);
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
    // The point in the reference coordinates of the initial cell, which the transfer locates,
    // and in those of the parent, where its corners as they are now place it.
    Point xi{0, 0, 0}, eta{0, 0, 0};
    for (int a = 0; a < d; ++a)
    {
      xi[a] = parent.low[a] + 0.5 * t[a] * (parent.high[a] - parent.low[a]);
      eta[a] = t[a] - 1.0;
    }
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
      node = middle(key(spanned[0], spanned[1]), place(parent, eta));
    else if (spanned.size() == 4 && d == 3)
      node = centre(sideKey(spanned), place(parent, eta));
    else
      node = addNode(place(parent, eta), spanned);
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
    const auto found = _side_sets.find(sideKey(nodes));
    if (found == _side_sets.end())
      continue;
    const auto sets = found->second;
    _side_sets.erase(found);
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
        _side_sets[sideKey(child_nodes)] = sets;
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
        const auto found = _centres.find(sideKey(nodes));
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
  std::vector<Index> merged;
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
        const Index centre_node = _centres.at(sideKey(nodes));
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
        const auto found = _side_sets.find(sideKey(child_nodes));
        if (found == _side_sets.end())
          continue;
        sets = found->second;
        _side_sets.erase(found);
      }
      if (!sets.empty())
        _side_sets[sideKey(nodes)] = sets;
    }
    for (Index child : parent.children)
      _cells[static_cast<std::size_t>(child)].alive = false;
    parent.leaf = true;
    merged.push_back(p);
  }
  // The corners of the children that no leaf uses any more are removed.  Each is recorded with
  // the shape functions at it of one merged parent that holds it: parents that share it share
  // the edge or face it lies on, where their shape functions agree.
  const std::vector<char> kept = usedNodes();
  std::vector<char> seen(_points.size(), 0);
  std::array<double, 8> N{};
  std::array<Point, 8> dN{};
  for (Index p : merged)
  {
    const Cell & parent = _cells[static_cast<std::size_t>(p)];
    for (Index child : parent.children)
      for (Index n : _cells[static_cast<std::size_t>(child)].n)
      {
        if (kept[static_cast<std::size_t>(n)] || seen[static_cast<std::size_t>(n)])
          continue;
        seen[static_cast<std::size_t>(n)] = 1;
        const Point & x = _origin_reference[static_cast<std::size_t>(parent.origin)].at(n);
        Point eta{0, 0, 0};
        for (int a = 0; a < _dimension; ++a)
          eta[a] = 2.0 * (x[a] - parent.low[a]) / (parent.high[a] - parent.low[a]) - 1.0;
        ReferenceElement::get(_type).shape(eta, N.data(), dN.data());
        std::vector<std::pair<Index, double>> weights;
        for (std::size_t k = 0; k < parent.n.size(); ++k)
          if (N[k] != 0.0)
            weights.push_back({parent.n[k], N[k]});
        _removed.push_back({n, std::move(weights)});
      }
  }
  return !merged.empty();
}

std::vector<Index>
Subdivision::leaves() const
{
  std::vector<Index> out;
  for (Index c = 0; c < static_cast<Index>(_cells.size()); ++c)
    if (_cells[static_cast<std::size_t>(c)].alive && _cells[static_cast<std::size_t>(c)].leaf)
      out.push_back(c);
  return out;
}

std::vector<MeshHierarchy::Placed>
Subdivision::currentElements()
{
  std::vector<Placed> out;
  for (Index c : leaves())
  {
    const Cell & cell = _cells[static_cast<std::size_t>(c)];
    out.push_back({cell.origin, cell.block, cell.level, c, cell.n});
  }
  return out;
}

bool
Subdivision::coarsenElements(const std::vector<char> & coarsen, std::vector<char> & refine)
{
  std::vector<char> merge(_cells.size(), 0), split(_cells.size(), 0);
  const std::vector<Index> before = leaves();
  for (std::size_t e = 0; e < before.size(); ++e)
  {
    merge[static_cast<std::size_t>(before[e])] = coarsen[e];
    split[static_cast<std::size_t>(before[e])] = refine[e];
  }
  const bool merged = coarsenCells(merge);
  refine.clear();
  for (Index c : leaves())
    refine.push_back(split[static_cast<std::size_t>(c)]);
  return merged;
}

bool
Subdivision::refineElements(const std::vector<char> & refine, int max_level)
{
  std::vector<char> split(_cells.size(), 0);
  const std::vector<Index> current = leaves();
  for (std::size_t e = 0; e < current.size(); ++e)
    split[static_cast<std::size_t>(current[e])] = refine[e];
  return refineCells(split, max_level);
}

bool
Subdivision::locate(const Placed & element, Index node, Point & xi) const
{
  // The box of the cell, in the reference coordinates of its initial cell.
  const Cell & cell = _cells[static_cast<std::size_t>(element.record)];
  const Point & x = _origin_reference[static_cast<std::size_t>(element.origin)].at(node);
  xi = Point{0, 0, 0};
  for (int a = 0; a < _dimension; ++a)
  {
    xi[a] = 2.0 * (x[a] - cell.low[a]) / (cell.high[a] - cell.low[a]) - 1.0;
    if (std::abs(xi[a]) > 1.0 + 1e-10)
      return false;
  }
  return true;
}

std::vector<Mesh::NodeConstraint>
Subdivision::hangingConstraints(const std::vector<Index> & index)
{
  // A used middle of an edge of a leaf, and in 3D a used centre of a face of a leaf, hangs: it
  // takes the mean of the ends of the edge or of the corners of the face.
  const auto used = [&](Index n) { return index[static_cast<std::size_t>(n)] >= 0; };
  std::map<Index, Hanging> hanging;
  for (Index c : leaves())
  {
    const Cell & cell = _cells[static_cast<std::size_t>(c)];
    for (const auto & e : _edges)
    {
      const Index a = cell.n[static_cast<std::size_t>(e[0])],
                  b = cell.n[static_cast<std::size_t>(e[1])];
      const auto found = _middles.find(key(a, b));
      if (found == _middles.end() || !used(found->second))
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
        const auto found = _centres.find(sideKey(nodes));
        if (found == _centres.end() || !used(found->second))
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
  return constraints;
}

} // namespace dualmesh
