// SPDX-License-Identifier: LGPL-2.1-or-later
#include "dualmesh/mesh/Bisection.h"

#include "dualmesh/core/InputParameters.h"
#include "dualmesh/fe/ReferenceElement.h"

#include <Eigen/Dense>

#include <algorithm>
#include <set>

namespace dualmesh
{

Bisection::Bisection(const Mesh & initial)
    : _dimension(initial.dimension()), _space_dimension(initial.spaceDimension()),
      _block_names(initial.blockNames()), _mesh(initial.dimension())
{
  _type = initial.numElements() > 0 ? initial.element(0).type : ElementType::Tri3;
  const bool simplices = _dimension == 2
                             ? (_type == ElementType::Tri3 || _type == ElementType::Tri6)
                             : (_type == ElementType::Tet4 || _type == ElementType::Tet10);
  for (const auto & el : initial.elements())
    if (!simplices || el.type != _type || (_dimension != 2 && _dimension != 3))
      throw InputError(
          "Local refinement and coarsening need a mesh of one element type: Tri3, Tri6, Tet4 or "
          "Tet10, refined by bisection, or Quad4 or Hex8, refined by subdivision.");
  const ReferenceElement & reference_element = ReferenceElement::get(_type);
  for (const auto & el : initial.elements())
  {
    _origin_nodes.emplace_back(el.nodes.begin(), el.nodes.end());
    std::map<Index, Point> coordinates;
    for (std::size_t k = 0; k < el.nodes.size(); ++k)
      coordinates[el.nodes[k]] = reference_element.node(static_cast<int>(k));
    _origin_reference.push_back(std::move(coordinates));
  }
  _points = initial.nodes();
  _parent_edge.assign(_points.size(), Edge{-1, -1});
  _live.assign(_points.size(), 1);
  // The middle nodes of a quadratic mesh are the middles of their edges, which bisection makes
  // corners, in the order of the nodes of Tri6 and Tet10.
  static const int triangle_edges[3][2] = {{0, 1}, {1, 2}, {2, 0}};
  static const int tetrahedron_edges[6][2] = {{0, 1}, {1, 2}, {0, 2}, {0, 3}, {1, 3}, {2, 3}};
  for (const auto & el : initial.elements())
  {
    const int corners = _dimension + 1;
    for (int k = corners; k < static_cast<int>(el.nodes.size()); ++k)
    {
      const int * e =
          _dimension == 3 ? tetrahedron_edges[k - corners] : triangle_edges[k - corners];
      const Edge edge = key(el.nodes[e[0]], el.nodes[e[1]]);
      _middles[edge] = el.nodes[k];
      _parent_edge[static_cast<std::size_t>(el.nodes[k])] = edge;
    }
  }
  const auto length = [&](Index a, Index b) { return norm(_points[a] - _points[b]); };
  for (const auto & el : initial.elements())
  {
    if (_dimension == 3)
    {
      Tetrahedron t;
      t.n = {el.nodes[0], el.nodes[1], el.nodes[2], el.nodes[3]};
      t.block = el.block;
      t.origin = static_cast<Index>(_tetrahedra.size());
      _tetrahedra.push_back(t);
      continue;
    }
    // The longest edge first, by a rotation of the nodes, which keeps the orientation.  Ties go
    // to the larger node numbers, so that the choice does not depend on rounding.
    int first = 0;
    for (int i = 1; i < 3; ++i)
    {
      const double l = length(el.nodes[i], el.nodes[(i + 1) % 3]);
      const double best = length(el.nodes[first], el.nodes[(first + 1) % 3]);
      const double scale = std::max(l, best);
      if (l > best + 1e-12 * scale || (std::abs(l - best) <= 1e-12 * scale &&
                                       key(el.nodes[i], el.nodes[(i + 1) % 3]) >
                                           key(el.nodes[first], el.nodes[(first + 1) % 3])))
        first = i;
    }
    Triangle t;
    t.n = {el.nodes[first], el.nodes[(first + 1) % 3], el.nodes[(first + 2) % 3]};
    t.block = el.block;
    t.origin = static_cast<Index>(_triangles.size());
    _triangles.push_back(t);
  }
  for (const auto & [name, sides] : initial.sidesets())
    for (const Side & side : sides)
    {
      const std::vector<Index> ends = initial.sideNodes(side);
      const std::pair<std::string, int> set{name, initial.element(side.first).block};
      if (_dimension == 3)
        _face_sets[faceKey(ends[0], ends[1], ends[2])].push_back(set);
      else
        _edge_sets[key(ends[0], ends[1])].push_back(set);
    }
  for (const auto & [name, members] : initial.nodesets())
  {
    _node_set_names.push_back(name);
    std::vector<char> in(_points.size(), 0);
    for (Index n : members)
      in[n] = 1;
    _in_node_set.push_back(std::move(in));
  }
  build();
  _numbers_before = _numbers;
  _placed_before = _placed;
}

Point
Bisection::reference(Index n, Index origin) const
{
  const auto & known = _origin_reference[static_cast<std::size_t>(origin)];
  const auto found = known.find(n);
  if (found != known.end())
    return found->second;
  // A node made by bisection is the middle of its edge, in the reference element of every initial
  // element whose closure holds the edge: bisection is affine there.
  const Edge & edge = _parent_edge[static_cast<std::size_t>(n)];
  if (edge.first < 0)
    throw std::logic_error("dualmesh: a node of the bisection is outside its initial element.");
  return 0.5 * (reference(edge.first, origin) + reference(edge.second, origin));
}

Index
Bisection::middle(const Edge & edge, Index origin)
{
  const auto found = _middles.find(edge);
  if (found != _middles.end())
    return found->second;
  const Index m = static_cast<Index>(_points.size());
  // The point of the quadratic map of the initial element at the middle of the edge in its
  // reference element; for a linear element, the middle of the straight edge.
  Point position = 0.5 * (_points[edge.first] + _points[edge.second]);
  if (_type == ElementType::Tri6 || _type == ElementType::Tet10)
  {
    const Point xi = 0.5 * (reference(edge.first, origin) + reference(edge.second, origin));
    const auto & nodes = _origin_nodes[static_cast<std::size_t>(origin)];
    std::array<double, 10> N{};
    std::array<Point, 10> dN{};
    ReferenceElement::get(_type).shape(xi, N.data(), dN.data());
    position = Point{0, 0, 0};
    for (std::size_t k = 0; k < nodes.size(); ++k)
      position = position + N[k] * _points[static_cast<std::size_t>(nodes[k])];
  }
  _points.push_back(position);
  _parent_edge.push_back(edge);
  _live.push_back(1);
  for (auto & in : _in_node_set)
    in.push_back(in[edge.first] && in[edge.second]);
  _middles[edge] = m;
  return m;
}

bool
Bisection::refine(const std::vector<char> & marked, int max_generation)
{
  if (marked.size() != numElements())
    throw InputError("The refinement marker must have one entry per element.");
  if (!(_dimension == 3 ? refineTetrahedra(marked, max_generation)
                        : refineTriangles(marked, max_generation)))
    return false;
  build();
  return true;
}

bool
Bisection::coarsen(const std::vector<char> & marked)
{
  if (marked.size() != numElements())
    throw InputError("The coarsening marker must have one entry per element.");
  if (!(_dimension == 3 ? coarsenTetrahedra(marked) : coarsenTriangles(marked)))
    return false;
  build();
  return true;
}

bool
Bisection::adapt(const std::vector<char> & refine,
                 const std::vector<char> & coarsen,
                 int max_generation)
{
  if (refine.size() != numElements() || coarsen.size() != numElements())
    throw InputError("The refinement and coarsening markers must have one entry per element.");
  // A triangle marked for both is refined.
  std::vector<char> only_coarsen(coarsen.size());
  for (std::size_t t = 0; t < coarsen.size(); ++t)
    only_coarsen[t] = coarsen[t] && !refine[t];
  std::vector<char> carried = refine;
  const bool coarsened = _dimension == 3 ? coarsenTetrahedra(only_coarsen, &carried)
                                         : coarsenTriangles(only_coarsen, &carried);
  const bool refined = _dimension == 3 ? refineTetrahedra(carried, max_generation)
                                       : refineTriangles(carried, max_generation);
  if (!coarsened && !refined)
    return false;
  build();
  return true;
}

bool
Bisection::refineTriangles(const std::vector<char> & marked, int max_generation)
{
  // The marked triangles mark their refinement edges, and every triangle with a marked edge marks
  // its refinement edge, so that each triangle splits by one of the patterns below.
  std::set<Edge> edges;
  for (std::size_t t = 0; t < _triangles.size(); ++t)
    if (marked[t] && _triangles[t].generation < max_generation)
      edges.insert(key(_triangles[t].n[0], _triangles[t].n[1]));
  if (edges.empty())
    return false;
  for (bool grew = true; grew;)
  {
    grew = false;
    for (const Triangle & t : _triangles)
      if (!edges.count(key(t.n[0], t.n[1])) &&
          (edges.count(key(t.n[1], t.n[2])) || edges.count(key(t.n[2], t.n[0]))))
      {
        edges.insert(key(t.n[0], t.n[1]));
        grew = true;
      }
  }
  // [a, b, c] splits into [c, a, m] and [b, c, m] at the middle m of a-b, and each child again
  // when its refinement edge, c-a or b-c, is marked.
  std::vector<Triangle> refined;
  refined.reserve(_triangles.size() + 3 * edges.size());
  for (const Triangle & t : _triangles)
  {
    const Index a = t.n[0], b = t.n[1], c = t.n[2];
    if (!edges.count(key(a, b)))
    {
      refined.push_back(t);
      continue;
    }
    const Index m = middle(key(a, b), t.origin);
    const auto child = [&](Index x, Index y, Index z, int generation)
    {
      Triangle u = t;
      u.n = {x, y, z};
      u.generation = generation;
      refined.push_back(u);
    };
    if (edges.count(key(c, a)))
    {
      const Index m1 = middle(key(c, a), t.origin);
      child(m, c, m1, t.generation + 2);
      child(a, m, m1, t.generation + 2);
    }
    else
      child(c, a, m, t.generation + 1);
    if (edges.count(key(b, c)))
    {
      const Index m2 = middle(key(b, c), t.origin);
      child(m, b, m2, t.generation + 2);
      child(c, m, m2, t.generation + 2);
    }
    else
      child(b, c, m, t.generation + 1);
  }
  _triangles = std::move(refined);
  // The halves of an edge of a side set are in that set.
  for (const Edge & edge : edges)
  {
    const auto found = _edge_sets.find(edge);
    if (found == _edge_sets.end())
      continue;
    const auto sets = found->second;
    _edge_sets.erase(found);
    const Index m = _middles.at(edge);
    _edge_sets[key(edge.first, m)] = sets;
    _edge_sets[key(m, edge.second)] = sets;
  }
  return true;
}

bool
Bisection::coarsenTriangles(const std::vector<char> & marked, std::vector<char> * carried)
{
  std::vector<std::vector<std::size_t>> around(_points.size());
  for (std::size_t t = 0; t < _triangles.size(); ++t)
    for (Index n : _triangles[t].n)
      around[n].push_back(t);
  std::vector<char> removed(_triangles.size(), 0);
  std::vector<Triangle> parents;
  std::vector<Index> gone;
  for (Index m = 0; m < static_cast<Index>(_points.size()); ++m)
  {
    const Edge edge = _parent_edge[m];
    const auto & patch = around[m];
    if (edge.first < 0 || (patch.size() != 2 && patch.size() != 4))
      continue;
    bool undo = true;
    for (std::size_t t : patch)
      undo = undo && marked[t] && _triangles[t].n[2] == m;
    if (!undo)
      continue;
    // The children [c, a, m] and [b, c, m] of [a, b, c] share their node c.
    std::map<Index, std::vector<std::size_t>> by_c;
    for (std::size_t t : patch)
    {
      const Triangle & u = _triangles[t];
      const Index c = (u.n[0] == edge.first || u.n[0] == edge.second) ? u.n[1] : u.n[0];
      by_c[c].push_back(t);
    }
    std::vector<Triangle> merged;
    for (const auto & [c, pair] : by_c)
    {
      if (pair.size() != 2)
        break;
      const Triangle & p = _triangles[pair[0]];
      const Triangle & q = _triangles[pair[1]];
      const Triangle & first = p.n[0] == c ? p : q;
      const Triangle & second = p.n[0] == c ? q : p;
      if (first.n[0] != c || second.n[1] != c || first.block != second.block)
        break;
      Triangle parent = first;
      parent.n = {first.n[1], second.n[0], c};
      parent.generation = first.generation - 1;
      merged.push_back(parent);
    }
    if (2 * merged.size() != patch.size())
      continue;
    for (std::size_t t : patch)
      removed[t] = 1;
    parents.insert(parents.end(), merged.begin(), merged.end());
    gone.push_back(m);
  }
  if (gone.empty())
    return false;
  std::vector<Triangle> kept;
  std::vector<char> marks;
  for (std::size_t t = 0; t < _triangles.size(); ++t)
    if (!removed[t])
    {
      kept.push_back(_triangles[t]);
      if (carried)
        marks.push_back((*carried)[t]);
    }
  kept.insert(kept.end(), parents.begin(), parents.end());
  _triangles = std::move(kept);
  if (carried)
  {
    marks.resize(_triangles.size(), 0);
    *carried = std::move(marks);
  }
  for (Index m : gone)
  {
    const Edge edge = _parent_edge[m];
    const auto half = _edge_sets.find(key(edge.first, m));
    if (half != _edge_sets.end())
    {
      _edge_sets[edge] = half->second;
      _edge_sets.erase(half);
      _edge_sets.erase(key(m, edge.second));
    }
    _live[m] = 0;
  }
  return true;
}

bool
Bisection::refineTetrahedra(const std::vector<char> & marked, int max_generation)
{
  // The working set: tetrahedra are retired, not erased, so that indices stay valid.
  std::vector<Tetrahedron> tets = _tetrahedra;
  std::vector<char> live(tets.size(), 1);
  std::map<Edge, std::vector<std::size_t>> around;
  const auto edgesOf = [](const Tetrahedron & t)
  {
    std::array<Edge, 6> out;
    int k = 0;
    for (int i = 0; i < 4; ++i)
      for (int j = i + 1; j < 4; ++j)
        out[k++] = key(t.n[i], t.n[j]);
    return out;
  };
  const auto attach = [&](std::size_t t)
  {
    for (const Edge & e : edgesOf(tets[t]))
      around[e].push_back(t);
  };
  for (std::size_t t = 0; t < tets.size(); ++t)
    attach(t);
  const auto length = [&](const Edge & e) { return norm(_points[e.first] - _points[e.second]); };
  // The longest edge, ties to the larger node numbers, so that every tetrahedron around an edge
  // agrees on whether it is the longest of each.
  const auto longest = [&](std::size_t t)
  {
    const std::array<Edge, 6> edges = edgesOf(tets[t]);
    Edge best = edges[0];
    for (int k = 1; k < 6; ++k)
    {
      const double l = length(edges[k]), b = length(best);
      const double scale = std::max(l, b);
      if (l > b + 1e-12 * scale || (std::abs(l - b) <= 1e-12 * scale && edges[k] > best))
        best = edges[k];
    }
    return best;
  };

  std::vector<std::size_t> stack;
  for (std::size_t t = 0; t < tets.size(); ++t)
    if (marked[t] && tets[t].generation < max_generation)
      stack.push_back(t);
  if (stack.empty())
    return false;
  // The recursion ends (each detour bisects a strictly longer edge); the cap turns a wrong
  // geometry, a degenerate or duplicated tetrahedron, into an error.
  const std::size_t step_limit = 1000 * (tets.size() + 1);
  std::size_t steps = 0;
  while (!stack.empty())
  {
    if (++steps > step_limit)
      throw InputError("The refinement of the tetrahedra did not terminate. Check the mesh for "
                       "degenerate or duplicated elements.");
    const std::size_t t = stack.back();
    if (!live[t])
    {
      stack.pop_back();
      continue;
    }
    const Edge edge = longest(t);
    std::vector<std::size_t> ring;
    bool blocked = false;
    for (std::size_t u : around[edge])
      if (live[u])
      {
        ring.push_back(u);
        if (longest(u) != edge)
        {
          stack.push_back(u);
          blocked = true;
          break;
        }
      }
    if (blocked)
      continue;
    stack.pop_back();
    const Index m = middle(edge, tets[ring.front()].origin);
    // A face of a side set through the edge is split into its two halves.
    for (std::size_t u : ring)
      for (Index c : tets[u].n)
        if (c != edge.first && c != edge.second)
        {
          const auto found = _face_sets.find(faceKey(edge.first, edge.second, c));
          if (found == _face_sets.end())
            continue;
          const auto sets = found->second;
          _face_sets.erase(found);
          _face_sets[faceKey(edge.first, m, c)] = sets;
          _face_sets[faceKey(m, edge.second, c)] = sets;
        }
    // Each tetrahedron of the ring splits into the one with the second end of the edge replaced
    // by the middle node and the one with the first end replaced, which keep its orientation.
    for (std::size_t u : ring)
    {
      live[u] = 0;
      for (const Index replaced : {edge.second, edge.first})
      {
        Tetrahedron child = tets[u];
        for (Index & n : child.n)
          if (n == replaced)
            n = m;
        child.generation += 1;
        child.births.push_back(m);
        tets.push_back(child);
        live.push_back(1);
        attach(tets.size() - 1);
      }
    }
  }
  _tetrahedra.clear();
  for (std::size_t t = 0; t < tets.size(); ++t)
    if (live[t])
      _tetrahedra.push_back(tets[t]);
  return true;
}

bool
Bisection::coarsenTetrahedra(const std::vector<char> & marked, std::vector<char> * carried)
{
  std::vector<std::vector<std::size_t>> around(_points.size());
  for (std::size_t t = 0; t < _tetrahedra.size(); ++t)
    for (Index n : _tetrahedra[t].n)
      around[n].push_back(t);
  std::vector<char> removed(_tetrahedra.size(), 0);
  std::vector<Tetrahedron> parents;
  std::vector<Index> gone;
  for (Index m = 0; m < static_cast<Index>(_points.size()); ++m)
  {
    const Edge edge = _parent_edge[m];
    const auto & ring = around[m];
    if (edge.first < 0 || ring.empty() || ring.size() % 2 != 0)
      continue;
    bool undo = true;
    for (std::size_t t : ring)
      undo =
          undo && marked[t] && !_tetrahedra[t].births.empty() && _tetrahedra[t].births.back() == m;
    if (!undo)
      continue;
    // The two children of a parent share the face of m and the two other nodes of the parent;
    // the one with the first end of the edge has m where the parent has the second end.
    std::map<Edge, std::vector<std::size_t>> by_others;
    for (std::size_t t : ring)
    {
      std::vector<Index> others;
      for (Index n : _tetrahedra[t].n)
        if (n != m && n != edge.first && n != edge.second)
          others.push_back(n);
      if (others.size() != 2)
      {
        undo = false;
        break;
      }
      by_others[key(others[0], others[1])].push_back(t);
    }
    if (!undo)
      continue;
    std::vector<Tetrahedron> merged;
    for (const auto & [others, pair] : by_others)
    {
      if (pair.size() != 2)
        break;
      const auto holds = [&](std::size_t t, Index n)
      {
        return std::find(_tetrahedra[t].n.begin(), _tetrahedra[t].n.end(), n) !=
               _tetrahedra[t].n.end();
      };
      const std::size_t first = holds(pair[0], edge.first) ? pair[0] : pair[1];
      const std::size_t second = first == pair[0] ? pair[1] : pair[0];
      if (!holds(first, edge.first) || !holds(second, edge.second) ||
          _tetrahedra[first].block != _tetrahedra[second].block)
        break;
      Tetrahedron parent = _tetrahedra[first];
      for (Index & n : parent.n)
        if (n == m)
          n = edge.second;
      parent.generation -= 1;
      parent.births.pop_back();
      merged.push_back(parent);
    }
    if (2 * merged.size() != ring.size())
      continue;
    for (std::size_t t : ring)
      removed[t] = 1;
    parents.insert(parents.end(), merged.begin(), merged.end());
    gone.push_back(m);
    // The halves of a face of a side set through the edge merge back.
    for (const auto & [others, pair] : by_others)
      for (const Index c : {others.first, others.second})
      {
        const auto half = _face_sets.find(faceKey(edge.first, m, c));
        if (half == _face_sets.end())
          continue;
        _face_sets[faceKey(edge.first, edge.second, c)] = half->second;
        _face_sets.erase(half);
        _face_sets.erase(faceKey(m, edge.second, c));
      }
  }
  if (gone.empty())
    return false;
  std::vector<Tetrahedron> kept;
  std::vector<char> marks;
  for (std::size_t t = 0; t < _tetrahedra.size(); ++t)
    if (!removed[t])
    {
      kept.push_back(_tetrahedra[t]);
      if (carried)
        marks.push_back((*carried)[t]);
    }
  kept.insert(kept.end(), parents.begin(), parents.end());
  _tetrahedra = std::move(kept);
  if (carried)
  {
    marks.resize(_tetrahedra.size(), 0);
    *carried = std::move(marks);
  }
  for (Index m : gone)
    _live[m] = 0;
  return true;
}

void
Bisection::project(const std::string & name, const std::function<Point(const Point &)> & projection)
{
  const auto found = _mesh.sidesets().find(name);
  if (found == _mesh.sidesets().end())
    throw InputError("There is no side set '" + name + "' to move onto its boundary.");
  std::vector<char> moved(_points.size(), 0);
  for (const Side & side : found->second)
    for (Index i : _mesh.sideNodes(side))
    {
      const Index n = _numbers[static_cast<std::size_t>(i)];
      if (moved[static_cast<std::size_t>(n)])
        continue;
      _points[static_cast<std::size_t>(n)] = projection(_points[static_cast<std::size_t>(n)]);
      moved[static_cast<std::size_t>(n)] = 1;
    }
  // The same mesh with the moved positions, and the same mesh before the last change.
  const std::vector<Index> numbers_before = _numbers_before;
  const std::vector<Placed> placed_before = _placed_before;
  build();
  _numbers_before = numbers_before;
  _placed_before = placed_before;
}

std::vector<int>
Bisection::generations() const
{
  std::vector<int> out;
  for (const Triangle & t : _triangles)
    out.push_back(t.generation);
  for (const Tetrahedron & t : _tetrahedra)
    out.push_back(t.generation);
  return out;
}

std::vector<Index>
Bisection::origins() const
{
  std::vector<Index> out;
  for (const Triangle & t : _triangles)
    out.push_back(t.origin);
  for (const Tetrahedron & t : _tetrahedra)
    out.push_back(t.origin);
  return out;
}

void
Bisection::build()
{
  _numbers_before = _numbers;
  _placed_before = std::move(_placed);
  _placed.clear();
  // The nodes of every element; the middles of the edges of a quadratic element may be new.
  std::vector<int> blocks;
  for (const Triangle & t : _triangles)
  {
    _placed.push_back({t.origin, elementNodes(t.n.data(), t.origin)});
    blocks.push_back(t.block);
  }
  for (const Tetrahedron & t : _tetrahedra)
  {
    _placed.push_back({t.origin, elementNodes(t.n.data(), t.origin)});
    blocks.push_back(t.block);
  }
  std::vector<char> used(_points.size(), 0);
  for (const Placed & element : _placed)
    for (Index n : element.nodes)
      used[n] = 1;
  _numbers.clear();
  std::vector<Index> index(_points.size(), -1);
  Mesh out(_dimension);
  out.setSpaceDimension(_space_dimension);
  for (const auto & [block, name] : _block_names)
    out.setBlockName(block, name);
  for (Index n = 0; n < static_cast<Index>(_points.size()); ++n)
    if (used[n])
    {
      index[n] = static_cast<Index>(_numbers.size());
      _numbers.push_back(n);
      out.addNode(_points[n]);
    }
  for (std::size_t e = 0; e < _placed.size(); ++e)
  {
    std::vector<Index> nodes;
    for (Index n : _placed[e].nodes)
      nodes.push_back(index[n]);
    out.addElement(_type, nodes, blocks[e]);
  }
  // The sides of every edge (face): one on the boundary, two inside.  A side of a set between two
  // blocks is the side of the element of its block.
  std::map<std::string, std::vector<Side>> sets;
  const auto addSets = [&](const auto & sides_of, const auto & sets_of)
  {
    for (const auto & [face, names] : sets_of)
    {
      const auto found = sides_of.find(face);
      if (found == sides_of.end())
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
  };
  if (_dimension == 3)
  {
    std::map<Face, std::vector<Side>> sides;
    for (Index e = 0; e < out.numElements(); ++e)
      for (int k = 0; k < 4; ++k)
      {
        const std::vector<Index> c = out.sideNodes({e, k});
        sides[faceKey(_numbers[c[0]], _numbers[c[1]], _numbers[c[2]])].push_back({e, k});
      }
    addSets(sides, _face_sets);
  }
  else
  {
    std::map<Edge, std::vector<Side>> sides;
    for (Index e = 0; e < out.numElements(); ++e)
      for (int k = 0; k < 3; ++k)
      {
        const std::vector<Index> ends = out.sideNodes({e, k});
        sides[key(_numbers[ends[0]], _numbers[ends[1]])].push_back({e, k});
      }
    addSets(sides, _edge_sets);
  }
  for (const auto & [name, list] : sets)
    out.addSideset(name, list);
  for (std::size_t k = 0; k < _node_set_names.size(); ++k)
  {
    std::vector<Index> members;
    for (std::size_t i = 0; i < _numbers.size(); ++i)
      if (_in_node_set[k][_numbers[i]])
        members.push_back(static_cast<Index>(i));
    out.addNodeset(_node_set_names[k], members);
  }
  _mesh = std::move(out);
}

std::vector<Index>
Bisection::elementNodes(const Index * corners, Index origin)
{
  std::vector<Index> nodes(corners, corners + (_dimension == 3 ? 4 : 3));
  // The middles of the edges, in the order of the nodes of Tri6 and Tet10.
  static const int triangle_edges[3][2] = {{0, 1}, {1, 2}, {2, 0}};
  static const int tetrahedron_edges[6][2] = {{0, 1}, {1, 2}, {0, 2}, {0, 3}, {1, 3}, {2, 3}};
  if (_type == ElementType::Tri6)
    for (const auto & e : triangle_edges)
      nodes.push_back(middle(key(corners[e[0]], corners[e[1]]), origin));
  if (_type == ElementType::Tet10)
    for (const auto & e : tetrahedron_edges)
      nodes.push_back(middle(key(corners[e[0]], corners[e[1]]), origin));
  return nodes;
}

Eigen::VectorXd
Bisection::transfer(const Eigen::VectorXd & before, int width) const
{
  if (before.size() != static_cast<Index>(_numbers_before.size()) * width)
    throw InputError("The values to transfer have " + std::to_string(before.size()) +
                     " entries, and the mesh before the change has " +
                     std::to_string(_numbers_before.size()) + " nodes of " + std::to_string(width) +
                     " values.");
  std::vector<Index> position_before(_points.size(), -1);
  for (std::size_t i = 0; i < _numbers_before.size(); ++i)
    position_before[static_cast<std::size_t>(_numbers_before[i])] = static_cast<Index>(i);
  // The old elements of each initial element, where a new node is looked for.
  std::map<Index, std::vector<std::size_t>> old_of_origin;
  for (std::size_t e = 0; e < _placed_before.size(); ++e)
    old_of_origin[_placed_before[e].origin].push_back(e);
  // An initial element of each new node: that of an element of the current mesh that holds it.
  std::vector<Index> origin_of(_points.size(), -1);
  for (const Placed & element : _placed)
    for (Index n : element.nodes)
      origin_of[static_cast<std::size_t>(n)] = element.origin;
  const ReferenceElement & reference_element = ReferenceElement::get(_type);
  const int corners = _dimension + 1;
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
    // The old element of the same initial element whose corners hold the node, in the reference
    // coordinates of the initial element, where every element of the hierarchy is affine.
    const Index origin = origin_of[static_cast<std::size_t>(n)];
    const Point x = reference(n, origin);
    bool found = false;
    for (std::size_t e : old_of_origin[origin])
    {
      const Placed & old = _placed_before[e];
      // The barycentric coordinates of x in the old element: x = c0 + sum_k xi_k (c_k - c0).
      Eigen::Matrix3d A = Eigen::Matrix3d::Identity();
      Eigen::Vector3d rhs = Eigen::Vector3d::Zero();
      const Point c0 = reference(old.nodes[0], origin);
      for (int k = 1; k < corners; ++k)
      {
        const Point ck = reference(old.nodes[static_cast<std::size_t>(k)], origin);
        for (int d = 0; d < 3; ++d)
          A(d, k - 1) = ck[d] - c0[d];
      }
      for (int d = 0; d < 3; ++d)
        rhs[d] = x[d] - c0[d];
      const Eigen::Vector3d xi = A.colPivHouseholderQr().solve(rhs);
      double sum = 0.0;
      bool inside = true;
      for (int k = 0; k < _dimension; ++k)
      {
        inside = inside && xi[k] >= -1e-10;
        sum += xi[k];
      }
      if (!inside || sum > 1.0 + 1e-10)
        continue;
      std::array<double, 10> N{};
      std::array<Point, 10> dN{};
      reference_element.shape(Point{xi[0], xi[1], xi[2]}, N.data(), dN.data());
      Eigen::VectorXd value = Eigen::VectorXd::Zero(width);
      for (std::size_t k = 0; k < old.nodes.size(); ++k)
        value += N[k] * before.segment(
                            position_before[static_cast<std::size_t>(old.nodes[k])] * width, width);
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
