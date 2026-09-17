// SPDX-License-Identifier: LGPL-2.1-or-later
#include "dualmesh/fe/ReferenceElement.h"
#include "dualmesh/core/InputParameters.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <map>
#include <memory>
#include <mutex>
#include <set>

namespace dualmesh
{

namespace
{
Point
mid(const Point & a, const Point & b)
{
  return 0.5 * (a + b);
}

Point
average(const std::vector<Point> & pts)
{
  Point c{0, 0, 0};
  for (const auto & p : pts)
    c = c + p;
  return (1.0 / pts.size()) * c;
}

/// Tensor-product linear shape functions on [-1,1]^dim in Edge2/Quad4/Hex8
/// corner ordering.
void
tensorShape(int dim, const Point & s, double * N, Point * dN)
{
  if (dim == 0)
  {
    N[0] = 1.0;
    if (dN)
      dN[0] = {0, 0, 0};
    return;
  }
  if (dim == 1)
  {
    N[0] = 0.5 * (1 - s[0]);
    N[1] = 0.5 * (1 + s[0]);
    if (dN)
    {
      dN[0] = {-0.5, 0, 0};
      dN[1] = {0.5, 0, 0};
    }
    return;
  }
  static const int sgn2[4][2] = {{-1, -1}, {1, -1}, {1, 1}, {-1, 1}};
  if (dim == 2)
  {
    for (int k = 0; k < 4; ++k)
    {
      const double a = sgn2[k][0], b = sgn2[k][1];
      N[k] = 0.25 * (1 + a * s[0]) * (1 + b * s[1]);
      if (dN)
        dN[k] = {0.25 * a * (1 + b * s[1]), 0.25 * b * (1 + a * s[0]), 0};
    }
    return;
  }
  for (int k = 0; k < 8; ++k)
  {
    const double a = sgn2[k % 4][0], b = sgn2[k % 4][1], c = k < 4 ? -1 : 1;
    N[k] = 0.125 * (1 + a * s[0]) * (1 + b * s[1]) * (1 + c * s[2]);
    if (dN)
      dN[k] = {0.125 * a * (1 + b * s[1]) * (1 + c * s[2]),
               0.125 * b * (1 + a * s[0]) * (1 + c * s[2]),
               0.125 * c * (1 + a * s[0]) * (1 + b * s[1])};
  }
}
} // namespace

Point
Patch::map(const Point & s, std::array<Point, 3> * tangents) const
{
  double N[8];
  Point dN[8];
  tensorShape(dim, s, N, dN);
  Point x{0, 0, 0};
  const int n = 1 << dim;
  for (int k = 0; k < n; ++k)
    x = x + N[k] * corners[k];
  if (tangents)
    for (int d = 0; d < dim; ++d)
    {
      Point t{0, 0, 0};
      for (int k = 0; k < n; ++k)
        t = t + dN[k][d] * corners[k];
      (*tangents)[d] = t;
    }
  return x;
}

const ReferenceElement &
ReferenceElement::get(ElementType type)
{
  static std::map<ElementType, std::unique_ptr<ReferenceElement>> cache;
  static std::mutex m;
  std::lock_guard<std::mutex> lock(m);
  auto & p = cache[type];
  if (!p)
    p.reset(new ReferenceElement(type));
  return *p;
}

ReferenceElement::ReferenceElement(ElementType type) : _type(type), _dim(elementDimension(type))
{
  std::vector<std::pair<int, int>> edges;
  switch (type)
  {
  case ElementType::Edge2:
    _nodes = {{-1, 0, 0}, {1, 0, 0}};
    _sides = {{0}, {1}};
    _side_type = ElementType::Edge2; // sides are points
    edges = {{0, 1}};
    break;
  case ElementType::Tri3:
    _nodes = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}};
    _sides = {{0, 1}, {1, 2}, {2, 0}};
    _side_type = ElementType::Edge2;
    edges = {{0, 1}, {1, 2}, {2, 0}};
    break;
  case ElementType::Quad4:
    _nodes = {{-1, -1, 0}, {1, -1, 0}, {1, 1, 0}, {-1, 1, 0}};
    _sides = {{0, 1}, {1, 2}, {2, 3}, {3, 0}};
    _side_type = ElementType::Edge2;
    edges = {{0, 1}, {1, 2}, {2, 3}, {3, 0}};
    break;
  case ElementType::Tet4:
    _nodes = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    _sides = {{0, 1, 3}, {1, 2, 3}, {0, 3, 2}, {0, 2, 1}};
    _side_type = ElementType::Tri3;
    edges = {{0, 1}, {1, 2}, {2, 0}, {0, 3}, {1, 3}, {2, 3}};
    break;
  case ElementType::Hex8:
    _nodes = {{-1, -1, -1},
              {1, -1, -1},
              {1, 1, -1},
              {-1, 1, -1},
              {-1, -1, 1},
              {1, -1, 1},
              {1, 1, 1},
              {-1, 1, 1}};
    _sides = {{0, 1, 5, 4}, {1, 2, 6, 5}, {2, 3, 7, 6}, {0, 4, 7, 3}, {0, 3, 2, 1}, {4, 5, 6, 7}};
    _side_type = ElementType::Quad4;
    edges = {{0, 1},
             {1, 2},
             {2, 3},
             {3, 0},
             {4, 5},
             {5, 6},
             {6, 7},
             {7, 4},
             {0, 4},
             {1, 5},
             {2, 6},
             {3, 7}};
    break;
  }
  _centroid = average(_nodes);
  buildDualGeometry(edges);
}

void
ReferenceElement::shape(const Point & xi, double * N, Point * dN) const
{
  switch (_type)
  {
  case ElementType::Edge2:
    tensorShape(1, xi, N, dN);
    return;
  case ElementType::Quad4:
    tensorShape(2, xi, N, dN);
    return;
  case ElementType::Hex8:
    tensorShape(3, xi, N, dN);
    return;
  case ElementType::Tri3:
    N[0] = 1 - xi[0] - xi[1];
    N[1] = xi[0];
    N[2] = xi[1];
    if (dN)
    {
      dN[0] = {-1, -1, 0};
      dN[1] = {1, 0, 0};
      dN[2] = {0, 1, 0};
    }
    return;
  case ElementType::Tet4:
    N[0] = 1 - xi[0] - xi[1] - xi[2];
    N[1] = xi[0];
    N[2] = xi[1];
    N[3] = xi[2];
    if (dN)
    {
      dN[0] = {-1, -1, -1};
      dN[1] = {1, 0, 0};
      dN[2] = {0, 1, 0};
      dN[3] = {0, 0, 1};
    }
    return;
  }
}

void
ReferenceElement::buildDualGeometry(const std::vector<std::pair<int, int>> & edges)
{
  const int nn = numNodes();

  // Faces (as node lists) of the element, used to find face centroids.
  // In 2D the "face" of the element is the element itself.
  std::vector<std::vector<int>> faces;
  if (_dim == 3)
    faces = _sides;

  const auto faceCentroid = [&](const std::vector<int> & f)
  {
    std::vector<Point> pts;
    for (int i : f)
      pts.push_back(_nodes[i]);
    return average(pts);
  };
  const auto facesWith = [&](const std::vector<int> & wanted)
  {
    std::vector<int> ids;
    for (std::size_t f = 0; f < faces.size(); ++f)
    {
      bool all = true;
      for (int w : wanted)
        all = all && std::find(faces[f].begin(), faces[f].end(), w) != faces[f].end();
      if (all)
        ids.push_back(static_cast<int>(f));
    }
    return ids;
  };
  // Neighbours of node a along element edges.
  const auto neighbours = [&](int a)
  {
    std::vector<int> nb;
    for (auto [p, q] : edges)
    {
      if (p == a)
        nb.push_back(q);
      if (q == a)
        nb.push_back(p);
    }
    return nb;
  };

  // ---- sub-control volumes -------------------------------------------------
  _scv.resize(nn);
  for (int a = 0; a < nn; ++a)
  {
    Patch & P = _scv[a];
    P.dim = _dim;
    const Point & A = _nodes[a];
    if (_dim == 1)
      P.corners = {A, _centroid};
    else if (_type == ElementType::Quad4)
    {
      P.corners = {A, {0, A[1], 0}, {0, 0, 0}, {A[0], 0, 0}};
    }
    else if (_type == ElementType::Tri3)
    {
      const auto nb = neighbours(a);
      P.corners = {A, mid(A, _nodes[nb[0]]), _centroid, mid(A, _nodes[nb[1]])};
    }
    else if (_type == ElementType::Hex8)
    {
      P.corners.resize(8);
      static const int sgn[8][3] = {{-1, -1, -1},
                                    {1, -1, -1},
                                    {1, 1, -1},
                                    {-1, 1, -1},
                                    {-1, -1, 1},
                                    {1, -1, 1},
                                    {1, 1, 1},
                                    {-1, 1, 1}};
      for (int k = 0; k < 8; ++k)
        for (int d = 0; d < 3; ++d)
          P.corners[k][d] = sgn[k][d] < 0 ? A[d] : 0.0;
    }
    else // Tet4
    {
      int b = -1, c = -1, d = -1;
      for (int i = 0; i < 4; ++i)
        if (i != a)
        {
          if (b < 0)
            b = i;
          else if (c < 0)
            c = i;
          else
            d = i;
        }
      const auto fc = [&](int i, int j, int k)
      { return average({_nodes[i], _nodes[j], _nodes[k]}); };
      P.corners = {A,
                   mid(A, _nodes[b]),
                   fc(a, b, c),
                   mid(A, _nodes[c]),
                   mid(A, _nodes[d]),
                   fc(a, b, d),
                   _centroid,
                   fc(a, c, d)};
    }
  }

  // ---- interfaces between control volumes ------------------------------------
  for (auto [a, b] : edges)
  {
    DualFace F;
    F.a = a;
    F.b = b;
    F.patch.dim = _dim - 1;
    const Point m = mid(_nodes[a], _nodes[b]);
    if (_dim == 1)
      F.patch.corners = {m};
    else if (_dim == 2)
      F.patch.corners = {m, _centroid};
    else
    {
      const auto fs = facesWith({a, b});
      if (fs.size() != 2)
        throw std::logic_error("dualmesh: edge is not shared by exactly two faces");
      F.patch.corners = {m, faceCentroid(faces[fs[0]]), _centroid, faceCentroid(faces[fs[1]])};
    }
    _faces.push_back(F);
  }

  // ---- side patches -------------------------------------------------------------
  const int ns = numSides();
  _side_patches.resize(ns);
  _side_full.resize(ns);
  for (int s = 0; s < ns; ++s)
  {
    const auto & sn = _sides[s];
    const int m = static_cast<int>(sn.size());
    std::vector<Point> P;
    for (int i : sn)
      P.push_back(_nodes[i]);
    const Point sc = average(P);
    for (int j = 0; j < m; ++j)
    {
      Patch q;
      q.dim = _dim - 1;
      if (_dim == 1)
        q.corners = {P[j]};
      else if (_dim == 2)
        q.corners = {P[j], sc};
      else
        q.corners = {P[j], mid(P[j], P[(j + 1) % m]), sc, mid(P[j], P[(j + m - 1) % m])};
      _side_patches[s].push_back(q);
    }
    if (m == 3) // triangular side: union of the node sub-patches
      _side_full[s].insert(_side_full[s].end(), _side_patches[s].begin(), _side_patches[s].end());
    else
    {
      Patch q;
      q.dim = _dim - 1;
      q.corners = P;
      _side_full[s] = {q};
    }
  }

  // ---- whole-element patches ------------------------------------------------------
  if (isSimplex())
    _elem_full.insert(_elem_full.end(), _scv.begin(), _scv.end());
  else
  {
    Patch q;
    q.dim = _dim;
    q.corners = _nodes;
    _elem_full = {q};
  }
}

// ----------------------------------------------------------------------------
// Quadrature
// ----------------------------------------------------------------------------

QuadratureSpec
QuadratureSpec::parse(const std::string & text_in, bool reduced)
{
  std::string text;
  for (char c : text_in)
    text.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
  QuadratureSpec q;
  q.reduced = reduced;
  if (text == "trapezoid" || text == "trapezoidal")
    q.kind = Kind::Trapezoid;
  else if (text == "simpson")
    q.kind = Kind::Simpson;
  else if (text == "nodal" || text == "lumped")
    q.kind = Kind::Nodal;
  else if (text == "interface")
    q.kind = Kind::Interface;
  else if (text == "control_domain_trapezoid" || text == "cd_trapezoid")
    q.kind = Kind::ControlDomainTrapezoid;
  else if (text == "midpoint" || text == "centroid")
  {
    q.kind = Kind::Gauss;
    q.points = 1;
  }
  else if (text.rfind("gauss", 0) == 0)
  {
    q.kind = Kind::Gauss;
    const std::string rest = text.substr(5);
    q.points = rest.empty() ? 2 : std::stoi(rest);
    if (q.points < 1 || q.points > 10)
      throw InputError("Gauss quadrature supports 1 to 10 points per direction.");
  }
  else
    throw InputError("Unknown quadrature '" + text_in +
                     "'. Use gauss1..gauss10, midpoint, trapezoid, simpson, nodal, interface, or "
                     "control_domain_trapezoid.");
  return q;
}

std::string
QuadratureSpec::str() const
{
  std::string s;
  switch (kind)
  {
  case Kind::Gauss:
    s = "gauss" + std::to_string(points);
    break;
  case Kind::Trapezoid:
    s = "trapezoid";
    break;
  case Kind::Simpson:
    s = "simpson";
    break;
  case Kind::Nodal:
    s = "nodal";
    break;
  case Kind::Interface:
    s = "interface";
    break;
  case Kind::ControlDomainTrapezoid:
    s = "control_domain_trapezoid";
    break;
  }
  return reduced ? s + "(reduced)" : s;
}

void
rule1D(const QuadratureSpec & spec, std::vector<double> & x, std::vector<double> & w)
{
  x.clear();
  w.clear();
  switch (spec.kind)
  {
  case QuadratureSpec::Kind::Trapezoid:
    x = {-1, 1};
    w = {1, 1};
    return;
  case QuadratureSpec::Kind::Simpson:
    x = {-1, 0, 1};
    w = {1.0 / 3, 4.0 / 3, 1.0 / 3};
    return;
  case QuadratureSpec::Kind::Nodal:
    x = {-1};
    w = {2};
    return;
  case QuadratureSpec::Kind::Interface:
  case QuadratureSpec::Kind::ControlDomainTrapezoid:
    x = {1};
    w = {2};
    return;
  case QuadratureSpec::Kind::Gauss:
    break;
  }
  // Gauss-Legendre by Newton iteration on the Legendre polynomial.
  const int n = spec.points;
  x.resize(n);
  w.resize(n);
  for (int i = 0; i < n; ++i)
  {
    double z = std::cos(M_PI * (i + 0.75) / (n + 0.5));
    double dp = 0;
    for (int it = 0; it < 100; ++it)
    {
      double p0 = 1, p1 = 0;
      for (int k = 1; k <= n; ++k)
      {
        const double p2 = p1;
        p1 = p0;
        p0 = ((2.0 * k - 1) * z * p1 - (k - 1.0) * p2) / k;
      }
      dp = n * (z * p0 - p1) / (z * z - 1);
      const double dz = p0 / dp;
      z -= dz;
      if (std::abs(dz) < 1e-15)
        break;
    }
    x[n - 1 - i] = z;
    w[n - 1 - i] = 2.0 / ((1 - z * z) * dp * dp);
  }
  if (n == 1)
  {
    x[0] = 0;
    w[0] = 2;
  }
}

void
ruleTensor(const QuadratureSpec & spec, int dim, std::vector<Point> & x, std::vector<double> & w)
{
  x.clear();
  w.clear();
  if (dim == 0)
  {
    x.push_back({0, 0, 0});
    w.push_back(1.0);
    return;
  }
  std::vector<double> x1, w1;
  rule1D(spec, x1, w1);
  const int n = static_cast<int>(x1.size());
  if (dim == 1)
    for (int i = 0; i < n; ++i)
    {
      x.push_back({x1[i], 0, 0});
      w.push_back(w1[i]);
    }
  else if (dim == 2)
    for (int j = 0; j < n; ++j)
      for (int i = 0; i < n; ++i)
      {
        x.push_back({x1[i], x1[j], 0});
        w.push_back(w1[i] * w1[j]);
      }
  else
    for (int k = 0; k < n; ++k)
      for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i)
        {
          x.push_back({x1[i], x1[j], x1[k]});
          w.push_back(w1[i] * w1[j] * w1[k]);
        }
}

} // namespace dualmesh
