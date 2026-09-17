// SPDX-License-Identifier: LGPL-2.1-or-later
#include "dualmesh/mesh/Mesh.h"
#include "dualmesh/core/InputParameters.h"
#include "dualmesh/fe/ReferenceElement.h"

#include <algorithm>
#include <limits>
#include <map>
#include <set>
#include <sstream>

namespace dualmesh
{

std::string
elementTypeName(ElementType t)
{
  switch (t)
  {
  case ElementType::Edge2:
    return "Edge2";
  case ElementType::Tri3:
    return "Tri3";
  case ElementType::Quad4:
    return "Quad4";
  case ElementType::Tet4:
    return "Tet4";
  case ElementType::Hex8:
    return "Hex8";
  }
  return "Unknown";
}

ElementType
elementTypeFromName(const std::string & n)
{
  if (n == "Edge2" || n == "EDGE2" || n == "line" || n == "Line2")
    return ElementType::Edge2;
  if (n == "Tri3" || n == "TRI3" || n == "triangle")
    return ElementType::Tri3;
  if (n == "Quad4" || n == "QUAD4" || n == "quad")
    return ElementType::Quad4;
  if (n == "Tet4" || n == "TET4" || n == "tetra")
    return ElementType::Tet4;
  if (n == "Hex8" || n == "HEX8" || n == "hexahedron")
    return ElementType::Hex8;
  throw InputError("Unsupported element type '" + n +
                   "'. Supported: Edge2, Tri3, Quad4, Tet4, Hex8.");
}

int
elementNumNodes(ElementType t)
{
  switch (t)
  {
  case ElementType::Edge2:
    return 2;
  case ElementType::Tri3:
    return 3;
  case ElementType::Quad4:
    return 4;
  case ElementType::Tet4:
    return 4;
  case ElementType::Hex8:
    return 8;
  }
  return 0;
}

int
elementDimension(ElementType t)
{
  switch (t)
  {
  case ElementType::Edge2:
    return 1;
  case ElementType::Tri3:
  case ElementType::Quad4:
    return 2;
  case ElementType::Tet4:
  case ElementType::Hex8:
    return 3;
  }
  return 0;
}

Index
Mesh::addNode(const Point & p)
{
  _nodes.push_back(p);
  return static_cast<Index>(_nodes.size()) - 1;
}

Index
Mesh::addElement(ElementType type, const std::vector<Index> & nodes, int block)
{
  if (static_cast<int>(nodes.size()) != elementNumNodes(type))
    throw InputError("Element of type " + elementTypeName(type) + " needs " +
                     std::to_string(elementNumNodes(type)) + " nodes.");
  if (elementDimension(type) != _dim)
    throw InputError("Element of type " + elementTypeName(type) +
                     " does not match the mesh dimension " + std::to_string(_dim) + ".");
  Element e;
  e.type = type;
  e.nodes.fill(-1);
  for (std::size_t i = 0; i < nodes.size(); ++i)
  {
    if (nodes[i] < 0 || nodes[i] >= numNodes())
      throw InputError("Element node index out of range.");
    e.nodes[i] = nodes[i];
  }
  e.block = block;
  _elements.push_back(e);
  _exterior_valid = false;
  return static_cast<Index>(_elements.size()) - 1;
}

std::vector<Index>
Mesh::sideNodes(const Side & s) const
{
  const auto & el = _elements[s.first];
  const auto & ref = ReferenceElement::get(el.type);
  std::vector<Index> out;
  for (int i : ref.sideNodes(s.second))
    out.push_back(el.nodes[i]);
  return out;
}

const std::vector<Side> &
Mesh::exteriorSides() const
{
  if (_exterior_valid)
    return _exterior_cache;
  std::map<std::vector<Index>, std::pair<Side, int>> count;
  for (Index e = 0; e < numElements(); ++e)
  {
    const auto & ref = ReferenceElement::get(_elements[e].type);
    for (int s = 0; s < ref.numSides(); ++s)
    {
      auto key = sideNodes({e, s});
      std::sort(key.begin(), key.end());
      auto it = count.find(key);
      if (it == count.end())
        count.emplace(key, std::make_pair(Side{e, s}, 1));
      else
        it->second.second++;
    }
  }
  _exterior_cache.clear();
  for (const auto & [key, v] : count)
    if (v.second == 1)
      _exterior_cache.push_back(v.first);
  std::sort(_exterior_cache.begin(), _exterior_cache.end());
  _exterior_valid = true;
  return _exterior_cache;
}

const std::vector<char> &
Mesh::boundaryNodeMarkers() const
{
  if (_exterior_valid && _boundary_nodes.size() == static_cast<std::size_t>(numNodes()))
    return _boundary_nodes;
  const auto & sides = exteriorSides();
  _boundary_nodes.assign(numNodes(), 0);
  for (const auto & s : sides)
    for (Index n : sideNodes(s))
      _boundary_nodes[n] = 1;
  return _boundary_nodes;
}

void
Mesh::addSidesetFromFaces(const std::string & name, const std::vector<std::vector<Index>> & faces)
{
  std::map<std::vector<Index>, Side> lookup;
  for (const auto & s : exteriorSides())
  {
    auto key = sideNodes(s);
    std::sort(key.begin(), key.end());
    lookup[key] = s;
  }
  auto & list = _sidesets[name];
  std::size_t missing = 0;
  for (auto f : faces)
  {
    std::sort(f.begin(), f.end());
    auto it = lookup.find(f);
    if (it == lookup.end())
      ++missing;
    else
      list.push_back(it->second);
  }
  if (missing)
    throw InputError("Side set '" + name + "': " + std::to_string(missing) +
                     " faces are not on the exterior boundary of the mesh.");
  std::sort(list.begin(), list.end());
  list.erase(std::unique(list.begin(), list.end()), list.end());
}

void
Mesh::addSideset(const std::string & name, const std::vector<Side> & sides)
{
  auto & list = _sidesets[name];
  list.insert(list.end(), sides.begin(), sides.end());
  std::sort(list.begin(), list.end());
  list.erase(std::unique(list.begin(), list.end()), list.end());
}

void
Mesh::addNodeset(const std::string & name, const std::vector<Index> & nodes)
{
  auto & list = _nodesets[name];
  list.insert(list.end(), nodes.begin(), nodes.end());
  std::sort(list.begin(), list.end());
  list.erase(std::unique(list.begin(), list.end()), list.end());
}

void
Mesh::aliasSideset(const std::string & existing, const std::string & new_name)
{
  addSideset(new_name, sideset(existing));
  addNodeset(new_name, boundaryNodes(existing));
}

void
Mesh::addSidesetByPredicate(const std::string & name,
                            const std::function<bool(const Point &)> & predicate)
{
  std::vector<Side> sides;
  for (const auto & s : exteriorSides())
    if (predicate(sideCentroid(s)))
      sides.push_back(s);
  addSideset(name, sides);
}

void
Mesh::addNodesetByPredicate(const std::string & name,
                            const std::function<bool(const Point &)> & predicate)
{
  std::vector<Index> ids;
  for (Index i = 0; i < numNodes(); ++i)
    if (predicate(_nodes[i]))
      ids.push_back(i);
  addNodeset(name, ids);
}

void
Mesh::transformNodes(const std::function<Point(const Point &)> & f)
{
  for (auto & p : _nodes)
    p = f(p);
}

namespace
{
double
detAtCentroid(const Mesh & mesh, const Element & el)
{
  const auto & ref = ReferenceElement::get(el.type);
  double N[8];
  Point dN[8];
  ref.shape(ref.centroid(), N, dN);
  const int dim = ref.dimension();
  double J[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
  for (int k = 0; k < ref.numNodes(); ++k)
    for (int i = 0; i < dim; ++i)
      for (int j = 0; j < dim; ++j)
        J[i][j] += mesh.node(el.nodes[k])[i] * dN[k][j];
  if (dim == 1)
    return J[0][0];
  if (dim == 2)
    return J[0][0] * J[1][1] - J[0][1] * J[1][0];
  return J[0][0] * (J[1][1] * J[2][2] - J[1][2] * J[2][1]) -
         J[0][1] * (J[1][0] * J[2][2] - J[1][2] * J[2][0]) +
         J[0][2] * (J[1][0] * J[2][1] - J[1][1] * J[2][0]);
}
} // namespace

void
Mesh::fixOrientation()
{
  // Remember side sets as node lists (independent of local numbering).
  std::map<std::string, std::vector<std::vector<Index>>> faces;
  for (const auto & [name, sides] : _sidesets)
    for (const auto & s : sides)
      faces[name].push_back(sideNodes(s));

  bool changed = false;
  for (auto & el : _elements)
  {
    if (detAtCentroid(*this, el) >= 0)
      continue;
    changed = true;
    switch (el.type)
    {
    case ElementType::Edge2:
    case ElementType::Tri3:
      std::swap(el.nodes[0], el.nodes[1]);
      break;
    case ElementType::Quad4:
      std::swap(el.nodes[1], el.nodes[3]);
      break;
    case ElementType::Tet4:
      std::swap(el.nodes[1], el.nodes[2]);
      break;
    case ElementType::Hex8:
      std::swap(el.nodes[1], el.nodes[3]);
      std::swap(el.nodes[5], el.nodes[7]);
      break;
    }
  }
  if (!changed)
    return;
  _exterior_valid = false;
  _sidesets.clear();
  for (const auto & [name, list] : faces)
    addSidesetFromFaces(name, list);
}

std::vector<int>
Mesh::blockIds() const
{
  std::set<int> ids;
  for (const auto & e : _elements)
    ids.insert(e.block);
  return {ids.begin(), ids.end()};
}

int
Mesh::blockId(const std::string & name) const
{
  for (const auto & [id, n] : _block_names)
    if (n == name)
      return id;
  try
  {
    std::size_t pos = 0;
    int id = std::stoi(name, &pos);
    if (pos == name.size())
      return id;
  }
  catch (...)
  {
  }
  throw InputError("Unknown block '" + name + "'.");
}

bool
Mesh::hasBoundary(const std::string & name) const
{
  return _sidesets.count(name) || _nodesets.count(name);
}

const std::vector<Side> &
Mesh::sideset(const std::string & name) const
{
  auto it = _sidesets.find(name);
  if (it == _sidesets.end())
  {
    std::ostringstream os;
    os << "Unknown side set '" << name << "'. Available side sets:";
    for (const auto & [n, _] : _sidesets)
      os << " " << n;
    throw InputError(os.str());
  }
  return it->second;
}

std::vector<Index>
Mesh::boundaryNodes(const std::string & name) const
{
  auto nit = _nodesets.find(name);
  if (nit != _nodesets.end())
    return nit->second;
  auto sit = _sidesets.find(name);
  if (sit == _sidesets.end())
  {
    std::ostringstream os;
    os << "Unknown boundary '" << name << "'. Available side sets:";
    for (const auto & [n, _] : _sidesets)
      os << " " << n;
    os << "; node sets:";
    for (const auto & [n, _] : _nodesets)
      os << " " << n;
    throw InputError(os.str());
  }
  std::set<Index> ids;
  for (const auto & s : sit->second)
    for (Index n : sideNodes(s))
      ids.insert(n);
  return {ids.begin(), ids.end()};
}

Point
Mesh::elementCentroid(Index e) const
{
  const auto & el = _elements[e];
  Point c{0, 0, 0};
  for (int i = 0; i < el.numNodes(); ++i)
    c = c + _nodes[el.nodes[i]];
  return (1.0 / el.numNodes()) * c;
}

Point
Mesh::sideCentroid(const Side & s) const
{
  Point c{0, 0, 0};
  const auto ids = sideNodes(s);
  for (Index n : ids)
    c = c + _nodes[n];
  return (1.0 / ids.size()) * c;
}

std::pair<Point, Point>
Mesh::boundingBox() const
{
  const double inf = std::numeric_limits<double>::infinity();
  Point lo{inf, inf, inf}, hi{-inf, -inf, -inf};
  for (const auto & p : _nodes)
    for (int d = 0; d < 3; ++d)
    {
      lo[d] = std::min(lo[d], p[d]);
      hi[d] = std::max(hi[d], p[d]);
    }
  return {lo, hi};
}

void
Mesh::addBoundingBoxSidesets(double tolerance)
{
  const auto [lo, hi] = boundingBox();
  static const char * names[3][2] = {{"left", "right"}, {"bottom", "top"}, {"back", "front"}};
  for (int d = 0; d < _dim; ++d)
  {
    const double tol = tolerance * std::max(1.0, hi[d] - lo[d]);
    for (int side = 0; side < 2; ++side)
    {
      const double target = side == 0 ? lo[d] : hi[d];
      std::vector<Side> sides;
      for (const auto & s : exteriorSides())
      {
        bool all = true;
        for (Index n : sideNodes(s))
          all = all && std::abs(_nodes[n][d] - target) <= tol;
        if (all)
          sides.push_back(s);
      }
      addSideset(names[d][side], sides);
    }
  }
}

std::string
Mesh::summary() const
{
  std::ostringstream os;
  os << "Mesh: dimension " << _dim << ", " << numNodes() << " nodes, " << numElements()
     << " elements\n";
  std::map<std::string, Index> types;
  for (const auto & e : _elements)
    types[elementTypeName(e.type)]++;
  for (const auto & [t, n] : types)
    os << "  " << n << " x " << t << "\n";
  os << "  blocks:";
  for (int b : blockIds())
  {
    os << " " << b;
    auto it = _block_names.find(b);
    if (it != _block_names.end())
      os << " (" << it->second << ")";
  }
  os << "\n  side sets:";
  for (const auto & [n, s] : _sidesets)
    os << " " << n << "[" << s.size() << "]";
  os << "\n  node sets:";
  for (const auto & [n, s] : _nodesets)
    os << " " << n << "[" << s.size() << "]";
  os << "\n";
  return os.str();
}

// ----------------------------------------------------------------------------
// Uniform refinement
// ----------------------------------------------------------------------------
Mesh
Mesh::refined() const
{
  Mesh out(_dim);
  out._block_names = _block_names;
  for (const auto & p : _nodes)
    out.addNode(p);
  // New nodes are identified by the (sorted) set of parent nodes.
  std::map<std::vector<Index>, Index> created;
  const auto nodeFor = [&](std::vector<Index> parents)
  {
    std::sort(parents.begin(), parents.end());
    auto it = created.find(parents);
    if (it != created.end())
      return it->second;
    Point c{0, 0, 0};
    for (Index p : parents)
      c = c + _nodes[p];
    Index id = out.addNode((1.0 / parents.size()) * c);
    created[parents] = id;
    return id;
  };

  std::vector<std::vector<Index>> children(_elements.size());
  for (Index e = 0; e < numElements(); ++e)
  {
    const auto & el = _elements[e];
    const auto n = [&](int i) { return el.nodes[i]; };
    const auto add = [&](ElementType t, const std::vector<Index> & c)
    { children[e].push_back(out.addElement(t, c, el.block)); };
    switch (el.type)
    {
    case ElementType::Edge2:
    {
      Index m = nodeFor({n(0), n(1)});
      add(el.type, {n(0), m});
      add(el.type, {m, n(1)});
      break;
    }
    case ElementType::Tri3:
    {
      Index m01 = nodeFor({n(0), n(1)}), m12 = nodeFor({n(1), n(2)}), m20 = nodeFor({n(2), n(0)});
      add(el.type, {n(0), m01, m20});
      add(el.type, {m01, n(1), m12});
      add(el.type, {m20, m12, n(2)});
      add(el.type, {m01, m12, m20});
      break;
    }
    case ElementType::Quad4:
    {
      Index m01 = nodeFor({n(0), n(1)}), m12 = nodeFor({n(1), n(2)}), m23 = nodeFor({n(2), n(3)}),
            m30 = nodeFor({n(3), n(0)}), c = nodeFor({n(0), n(1), n(2), n(3)});
      add(el.type, {n(0), m01, c, m30});
      add(el.type, {m01, n(1), m12, c});
      add(el.type, {c, m12, n(2), m23});
      add(el.type, {m30, c, m23, n(3)});
      break;
    }
    case ElementType::Tet4:
    {
      Index m01 = nodeFor({n(0), n(1)}), m02 = nodeFor({n(0), n(2)}), m03 = nodeFor({n(0), n(3)}),
            m12 = nodeFor({n(1), n(2)}), m13 = nodeFor({n(1), n(3)}), m23 = nodeFor({n(2), n(3)});
      add(el.type, {n(0), m01, m02, m03});
      add(el.type, {m01, n(1), m12, m13});
      add(el.type, {m02, m12, n(2), m23});
      add(el.type, {m03, m13, m23, n(3)});
      add(el.type, {m01, m02, m03, m13});
      add(el.type, {m01, m02, m12, m13});
      add(el.type, {m02, m03, m13, m23});
      add(el.type, {m02, m12, m13, m23});
      break;
    }
    case ElementType::Hex8:
    {
      // Local grid of 27 points indexed by (i,j,k) in {0,1,2}.
      static const int corner[8][3] = {
          {0, 0, 0}, {2, 0, 0}, {2, 2, 0}, {0, 2, 0}, {0, 0, 2}, {2, 0, 2}, {2, 2, 2}, {0, 2, 2}};
      const auto cornerId = [&](int i, int j, int k)
      {
        for (int c = 0; c < 8; ++c)
          if (corner[c][0] == i && corner[c][1] == j && corner[c][2] == k)
            return c;
        return -1;
      };
      const auto gridNode = [&](int i, int j, int k) -> Index
      {
        // parents = corners spanned by the (i,j,k) position
        std::vector<Index> parents;
        for (int a : (i == 1 ? std::vector<int>{0, 2} : std::vector<int>{i}))
          for (int b : (j == 1 ? std::vector<int>{0, 2} : std::vector<int>{j}))
            for (int c : (k == 1 ? std::vector<int>{0, 2} : std::vector<int>{k}))
              parents.push_back(n(cornerId(a, b, c)));
        if (parents.size() == 1)
          return parents[0];
        return nodeFor(parents);
      };
      for (int k = 0; k < 2; ++k)
        for (int j = 0; j < 2; ++j)
          for (int i = 0; i < 2; ++i)
          {
            std::vector<Index> c;
            for (int q = 0; q < 8; ++q)
              c.push_back(
                  gridNode(i + corner[q][0] / 2, j + corner[q][1] / 2, k + corner[q][2] / 2));
            add(el.type, c);
          }
      break;
    }
    }
  }

  // Side sets: child sides that are exterior and lie on a parent side.
  std::map<std::vector<Index>, Side> child_lookup;
  for (const auto & s : out.exteriorSides())
  {
    auto key = out.sideNodes(s);
    std::sort(key.begin(), key.end());
    child_lookup[key] = s;
  }
  for (const auto & [name, sides] : _sidesets)
  {
    std::vector<Side> list;
    for (const auto & ps : sides)
    {
      const auto pn = sideNodes(ps);
      std::set<Index> allowed(pn.begin(), pn.end());
      for (const auto & [parents, id] : created)
      {
        bool inside = true;
        for (Index p : parents)
          inside = inside && allowed.count(p);
        if (inside)
          allowed.insert(id);
      }
      for (Index c : children[ps.first])
      {
        const auto & ref = ReferenceElement::get(out._elements[c].type);
        for (int s = 0; s < ref.numSides(); ++s)
        {
          auto key = out.sideNodes({c, s});
          bool inside = true;
          for (Index k : key)
            inside = inside && allowed.count(k);
          if (!inside)
            continue;
          std::sort(key.begin(), key.end());
          if (child_lookup.count(key))
            list.push_back({c, s});
        }
      }
    }
    out.addSideset(name, list);
  }
  for (const auto & [name, ids] : _nodesets)
  {
    std::set<Index> set(ids.begin(), ids.end());
    std::vector<Index> list(ids.begin(), ids.end());
    for (const auto & [parents, id] : created)
    {
      bool inside = true;
      for (Index p : parents)
        inside = inside && set.count(p);
      if (inside)
        list.push_back(id);
    }
    out.addNodeset(name, list);
  }
  return out;
}

// ----------------------------------------------------------------------------
// Generators
// ----------------------------------------------------------------------------
namespace
{
void
checkIncreasing(const std::vector<double> & x, const char * name)
{
  if (x.size() < 2)
    throw InputError(std::string("Coordinate list '") + name + "' needs at least two values.");
  for (std::size_t i = 1; i < x.size(); ++i)
    if (!(x[i] > x[i - 1]))
      throw InputError(std::string("Coordinate list '") + name + "' must be strictly increasing.");
}
} // namespace

Mesh
generateLineMesh(const std::vector<double> & x)
{
  checkIncreasing(x, "x");
  Mesh m(1);
  for (double v : x)
    m.addNode({v, 0, 0});
  for (std::size_t i = 0; i + 1 < x.size(); ++i)
    m.addElement(ElementType::Edge2, {Index(i), Index(i + 1)});
  m.addBoundingBoxSidesets();
  m.addNodeset("left", {0});
  m.addNodeset("right", {Index(x.size() - 1)});
  return m;
}

Mesh
generateRectangleMesh(const std::vector<double> & x,
                      const std::vector<double> & y,
                      const std::string & element,
                      const std::string & diagonal)
{
  checkIncreasing(x, "x");
  checkIncreasing(y, "y");
  const ElementType type = elementTypeFromName(element);
  if (type != ElementType::Quad4 && type != ElementType::Tri3)
    throw InputError("Rectangle meshes support Quad4 or Tri3 elements.");
  Mesh m(2);
  const Index nx = x.size(), ny = y.size();
  for (Index j = 0; j < ny; ++j)
    for (Index i = 0; i < nx; ++i)
      m.addNode({x[i], y[j], 0});
  const auto id = [&](Index i, Index j) { return j * nx + i; };
  for (Index j = 0; j + 1 < ny; ++j)
    for (Index i = 0; i + 1 < nx; ++i)
    {
      Index n00 = id(i, j), n10 = id(i + 1, j), n11 = id(i + 1, j + 1), n01 = id(i, j + 1);
      if (type == ElementType::Quad4)
        m.addElement(type, {n00, n10, n11, n01});
      else
      {
        bool right = diagonal == "right" || (diagonal == "alternate" && (i + j) % 2 == 0);
        if (diagonal != "right" && diagonal != "left" && diagonal != "alternate")
          throw InputError("Triangle diagonal must be 'right', 'left', or 'alternate'.");
        if (right)
        {
          m.addElement(type, {n00, n10, n11});
          m.addElement(type, {n00, n11, n01});
        }
        else
        {
          m.addElement(type, {n00, n10, n01});
          m.addElement(type, {n10, n11, n01});
        }
      }
    }
  m.addBoundingBoxSidesets();
  return m;
}

Mesh
generateBoxMesh(const std::vector<double> & x,
                const std::vector<double> & y,
                const std::vector<double> & z,
                const std::string & element)
{
  checkIncreasing(x, "x");
  checkIncreasing(y, "y");
  checkIncreasing(z, "z");
  const ElementType type = elementTypeFromName(element);
  if (type != ElementType::Hex8 && type != ElementType::Tet4)
    throw InputError("Box meshes support Hex8 or Tet4 elements.");
  Mesh m(3);
  const Index nx = x.size(), ny = y.size(), nz = z.size();
  for (Index k = 0; k < nz; ++k)
    for (Index j = 0; j < ny; ++j)
      for (Index i = 0; i < nx; ++i)
        m.addNode({x[i], y[j], z[k]});
  const auto id = [&](Index i, Index j, Index k) { return (k * ny + j) * nx + i; };
  for (Index k = 0; k + 1 < nz; ++k)
    for (Index j = 0; j + 1 < ny; ++j)
      for (Index i = 0; i + 1 < nx; ++i)
      {
        std::vector<Index> h = {id(i, j, k),
                                id(i + 1, j, k),
                                id(i + 1, j + 1, k),
                                id(i, j + 1, k),
                                id(i, j, k + 1),
                                id(i + 1, j, k + 1),
                                id(i + 1, j + 1, k + 1),
                                id(i, j + 1, k + 1)};
        if (type == ElementType::Hex8)
          m.addElement(type, h);
        else
        {
          // Kuhn subdivision about the main diagonal 0-6 (conforming).
          static const int tets[6][4] = {
              {0, 1, 2, 6}, {0, 2, 3, 6}, {0, 3, 7, 6}, {0, 7, 4, 6}, {0, 4, 5, 6}, {0, 5, 1, 6}};
          for (const auto & t : tets)
            m.addElement(type, {h[t[0]], h[t[1]], h[t[2]], h[t[3]]});
        }
      }
  m.fixOrientation();
  m.addBoundingBoxSidesets();
  return m;
}

} // namespace dualmesh
