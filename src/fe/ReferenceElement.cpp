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
#include <stdexcept>

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

/// A triangle or tetrahedron as one square or cube with corners collapsed
/// onto each other: (a, b, c, c) for a triangle, (a, b, c, c, d, d, d, d) for
/// a tetrahedron, and (a, b, c, c, a', b', c', c') for a prism given its six
/// vertices.  A single collapsed patch integrates the whole element with far
/// fewer points than a union of sub-cells; see Patch::extra_points.
Patch
collapsedPatch(const std::vector<Point> & v)
{
  Patch q;
  q.extra_points = 1;
  if (v.size() == 3)
  {
    q.dim = 2;
    q.corners = {v[0], v[1], v[2], v[2]};
  }
  else if (v.size() == 4)
  {
    q.dim = 3;
    q.corners = {v[0], v[1], v[2], v[2], v[3], v[3], v[3], v[3]};
  }
  else
  {
    q.dim = 3;
    q.corners = {v[0], v[1], v[2], v[2], v[3], v[4], v[5], v[5]};
  }
  return q;
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
  // Exactly one of the two lists below is filled.  `grid` holds the
  // one-dimensional coordinates of a tensor-product element, and
  // `sub_simplices` holds the linear simplices that tile a simplicial
  // element: the element itself when it is linear, and its four (Tri6) or
  // eight (Tet10) "red" sub-simplices when it is quadratic.
  std::vector<double> grid;
  std::vector<std::vector<int>> sub_simplices;

  switch (type)
  {
  case ElementType::Edge2:
    _nodes = {{-1, 0, 0}, {1, 0, 0}};
    _sides = {{0}, {1}};
    _side_type = ElementType::Edge2; // sides are points
    grid = {-1, 1};
    break;
  case ElementType::Edge3:
    _nodes = {{-1, 0, 0}, {1, 0, 0}, {0, 0, 0}};
    _sides = {{0}, {1}};
    _side_type = ElementType::Edge2; // sides are points
    grid = {-1, 0, 1};
    break;
  case ElementType::Tri3:
    _nodes = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}};
    _sides = {{0, 1}, {1, 2}, {2, 0}};
    _side_type = ElementType::Edge2;
    sub_simplices = {{0, 1, 2}};
    break;
  case ElementType::Tri6:
    _nodes = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0.5, 0, 0}, {0.5, 0.5, 0}, {0, 0.5, 0}};
    _sides = {{0, 1, 3}, {1, 2, 4}, {2, 0, 5}};
    _side_type = ElementType::Edge3;
    // Red refinement: three corner triangles and the central one.
    sub_simplices = {{0, 3, 5}, {3, 1, 4}, {5, 4, 2}, {3, 4, 5}};
    break;
  case ElementType::Quad4:
    _nodes = {{-1, -1, 0}, {1, -1, 0}, {1, 1, 0}, {-1, 1, 0}};
    _sides = {{0, 1}, {1, 2}, {2, 3}, {3, 0}};
    _side_type = ElementType::Edge2;
    grid = {-1, 1};
    break;
  case ElementType::Quad9:
    _nodes = {{-1, -1, 0},
              {1, -1, 0},
              {1, 1, 0},
              {-1, 1, 0},
              {0, -1, 0},
              {1, 0, 0},
              {0, 1, 0},
              {-1, 0, 0},
              {0, 0, 0}};
    _sides = {{0, 1, 4}, {1, 2, 5}, {2, 3, 6}, {3, 0, 7}};
    _side_type = ElementType::Edge3;
    grid = {-1, 0, 1};
    break;
  case ElementType::Tet4:
    _nodes = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    _sides = {{0, 1, 3}, {1, 2, 3}, {0, 3, 2}, {0, 2, 1}};
    _side_type = ElementType::Tri3;
    sub_simplices = {{0, 1, 2, 3}};
    break;
  case ElementType::Tet10:
    _nodes = {{0, 0, 0},
              {1, 0, 0},
              {0, 1, 0},
              {0, 0, 1},
              {0.5, 0, 0},
              {0.5, 0.5, 0},
              {0, 0.5, 0},
              {0, 0, 0.5},
              {0.5, 0, 0.5},
              {0, 0.5, 0.5}};
    _sides = {{0, 1, 3, 4, 8, 7}, {1, 2, 3, 5, 9, 8}, {0, 3, 2, 7, 9, 6}, {0, 2, 1, 6, 5, 4}};
    _side_type = ElementType::Tri6;
    // Red refinement: four corner tetrahedra and the four tetrahedra that
    // tile the inner octahedron, split along its diagonal 4-9.
    sub_simplices = {{0, 4, 6, 7},
                     {4, 1, 5, 8},
                     {6, 5, 2, 9},
                     {7, 8, 9, 3},
                     {4, 9, 5, 6},
                     {4, 9, 6, 7},
                     {4, 9, 7, 8},
                     {4, 9, 8, 5}};
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
    grid = {-1, 1};
    break;
  case ElementType::Hex27:
    _nodes = {{-1, -1, -1}, {1, -1, -1}, {1, 1, -1},  {-1, 1, -1}, {-1, -1, 1}, {1, -1, 1},
              {1, 1, 1},    {-1, 1, 1},  {0, -1, -1}, {1, 0, -1},  {0, 1, -1},  {-1, 0, -1},
              {0, -1, 1},   {1, 0, 1},   {0, 1, 1},   {-1, 0, 1},  {-1, -1, 0}, {1, -1, 0},
              {1, 1, 0},    {-1, 1, 0},  {-1, 0, 0},  {1, 0, 0},   {0, -1, 0},  {0, 1, 0},
              {0, 0, -1},   {0, 0, 1},   {0, 0, 0}};
    // VTK order throughout: mid-edge nodes 8-11 on the bottom face, 12-15 on
    // the top face, 16-19 on the vertical edges; face centres 20-25 on the
    // faces x = -1, x = +1, y = -1, y = +1, z = -1, z = +1; and the centre 26.
    _sides = {{0, 1, 5, 4, 8, 17, 12, 16, 22},
              {1, 2, 6, 5, 9, 18, 13, 17, 21},
              {2, 3, 7, 6, 10, 19, 14, 18, 23},
              {0, 4, 7, 3, 16, 15, 19, 11, 20},
              {0, 3, 2, 1, 11, 10, 9, 8, 24},
              {4, 5, 6, 7, 12, 13, 14, 15, 25}};
    _side_type = ElementType::Quad9;
    grid = {-1, 0, 1};
    break;
  case ElementType::Quad8:
    _nodes = {{-1, -1, 0},
              {1, -1, 0},
              {1, 1, 0},
              {-1, 1, 0},
              {0, -1, 0},
              {1, 0, 0},
              {0, 1, 0},
              {-1, 0, 0}};
    _sides = {{0, 1, 4}, {1, 2, 5}, {2, 3, 6}, {3, 0, 7}};
    _side_type = ElementType::Edge3;
    break;
  case ElementType::Hex20:
    // VTK order: mid-edge nodes 8-11 on the bottom face, 12-15 on the top
    // face, 16-19 on the vertical edges.
    _nodes = {{-1, -1, -1}, {1, -1, -1}, {1, 1, -1}, {-1, 1, -1}, {-1, -1, 1},
              {1, -1, 1},   {1, 1, 1},   {-1, 1, 1}, {0, -1, -1}, {1, 0, -1},
              {0, 1, -1},   {-1, 0, -1}, {0, -1, 1}, {1, 0, 1},   {0, 1, 1},
              {-1, 0, 1},   {-1, -1, 0}, {1, -1, 0}, {1, 1, 0},   {-1, 1, 0}};
    _sides = {{0, 1, 5, 4, 8, 17, 12, 16},
              {1, 2, 6, 5, 9, 18, 13, 17},
              {2, 3, 7, 6, 10, 19, 14, 18},
              {0, 4, 7, 3, 16, 15, 19, 11},
              {0, 3, 2, 1, 11, 10, 9, 8},
              {4, 5, 6, 7, 12, 13, 14, 15}};
    _side_type = ElementType::Quad8;
    break;
  case ElementType::Wedge6:
    // VTK order: the base triangle 0, 1, 2 at zeta = -1, numbered so that its
    // right-hand normal points away from the top triangle 3, 4, 5.
    _nodes = {{0, 0, -1}, {0, 1, -1}, {1, 0, -1}, {0, 0, 1}, {0, 1, 1}, {1, 0, 1}};
    _sides = {{0, 1, 4, 3}, {1, 2, 5, 4}, {2, 0, 3, 5}, {0, 1, 2}, {3, 4, 5}};
    _side_type = ElementType::Quad4;
    break;
  case ElementType::Pyramid5:
    // VTK order: the base 0, 1, 2, 3 anticlockwise seen from the apex 4.
    _nodes = {{-1, -1, 0}, {1, -1, 0}, {1, 1, 0}, {-1, 1, 0}, {0, 0, 1}};
    _sides = {{0, 3, 2, 1}, {0, 1, 4}, {1, 2, 4}, {2, 3, 4}, {3, 0, 4}};
    _side_type = ElementType::Tri3;
    break;
  }
  _centroid = average(_nodes);
  switch (type)
  {
  case ElementType::Quad8:
  case ElementType::Hex20:
  case ElementType::Pyramid5:
    buildElementGeometryOnly();
    break;
  case ElementType::Wedge6:
    buildPrismDual();
    break;
  default:
    if (!grid.empty())
      buildTensorDual(grid);
    else
      buildSubdividedSimplexDual(sub_simplices);
  }
}

namespace
{
/// The Lagrange basis of the one-dimensional node set @p c and its derivative,
/// evaluated at @p t.  Writing the basis this way means the same code serves
/// the linear node set {-1, 1} and the quadratic one {-1, 0, 1}.
void
lagrange1D(const std::vector<double> & c, double t, double * L, double * dL)
{
  const int n = static_cast<int>(c.size());
  for (int k = 0; k < n; ++k)
  {
    double value = 1.0, derivative = 0.0;
    for (int m = 0; m < n; ++m)
    {
      if (m == k)
        continue;
      const double d = c[k] - c[m];
      // Product rule, accumulated one factor at a time.
      derivative = derivative * ((t - c[m]) / d) + value / d;
      value *= (t - c[m]) / d;
    }
    L[k] = value;
    dL[k] = derivative;
  }
}
} // namespace

void
ReferenceElement::shape(const Point & xi, double * N, Point * dN) const
{
  switch (_type)
  {
  case ElementType::Quad8:
  {
    // Serendipity: N = (1 + x xa)(1 + y ya)(x xa + y ya - 1) / 4 at a corner,
    // (1 - x^2)(1 + y ya) / 2 or (1 + x xa)(1 - y^2) / 2 at a mid-side node.
    const double x = xi[0], y = xi[1];
    for (int a = 0; a < 8; ++a)
    {
      const double xa = _nodes[a][0], ya = _nodes[a][1];
      if (a < 4)
      {
        N[a] = 0.25 * (1 + x * xa) * (1 + y * ya) * (x * xa + y * ya - 1);
        if (dN)
          dN[a] = {0.25 * xa * (1 + y * ya) * (2 * x * xa + y * ya),
                   0.25 * ya * (1 + x * xa) * (x * xa + 2 * y * ya),
                   0};
      }
      else if (xa == 0)
      {
        N[a] = 0.5 * (1 - x * x) * (1 + y * ya);
        if (dN)
          dN[a] = {-x * (1 + y * ya), 0.5 * ya * (1 - x * x), 0};
      }
      else
      {
        N[a] = 0.5 * (1 + x * xa) * (1 - y * y);
        if (dN)
          dN[a] = {0.5 * xa * (1 - y * y), -y * (1 + x * xa), 0};
      }
    }
    return;
  }
  case ElementType::Hex20:
  {
    // Serendipity: N = (1 + x xa)(1 + y ya)(1 + z za)(x xa + y ya + z za - 2) / 8
    // at a corner and (1 - s^2)(1 + t ta)(1 + u ua) / 4 at the midpoint of an
    // edge along which s varies.
    const double c[3] = {xi[0], xi[1], xi[2]};
    for (int a = 0; a < 20; ++a)
    {
      const double ca[3] = {_nodes[a][0], _nodes[a][1], _nodes[a][2]};
      if (a < 8)
      {
        const double f[3] = {1 + c[0] * ca[0], 1 + c[1] * ca[1], 1 + c[2] * ca[2]};
        const double g = c[0] * ca[0] + c[1] * ca[1] + c[2] * ca[2] - 2;
        N[a] = 0.125 * f[0] * f[1] * f[2] * g;
        if (dN)
          dN[a] = {0.125 * ca[0] * f[1] * f[2] * (g + f[0]),
                   0.125 * ca[1] * f[0] * f[2] * (g + f[1]),
                   0.125 * ca[2] * f[0] * f[1] * (g + f[2])};
        continue;
      }
      // The edge runs along the direction in which the node's coordinate is 0.
      const int d = ca[0] == 0 ? 0 : (ca[1] == 0 ? 1 : 2);
      double f[3], df[3];
      for (int k = 0; k < 3; ++k)
      {
        f[k] = k == d ? 1 - c[k] * c[k] : 1 + c[k] * ca[k];
        df[k] = k == d ? -2 * c[k] : ca[k];
      }
      N[a] = 0.25 * f[0] * f[1] * f[2];
      if (dN)
        dN[a] = {
            0.25 * df[0] * f[1] * f[2], 0.25 * f[0] * df[1] * f[2], 0.25 * f[0] * f[1] * df[2]};
    }
    return;
  }
  case ElementType::Wedge6:
  {
    // Triangle times segment.  With the VTK numbering the base nodes 0, 1, 2
    // sit at (xi, eta) = (0, 0), (0, 1), (1, 0).
    const double x = xi[0], y = xi[1], z = xi[2];
    const double tri[3] = {1 - x - y, y, x};
    const Point dtri[3] = {{-1, -1, 0}, {0, 1, 0}, {1, 0, 0}};
    for (int a = 0; a < 6; ++a)
    {
      const int t = a % 3;
      const double h = a < 3 ? 0.5 * (1 - z) : 0.5 * (1 + z);
      const double dh = a < 3 ? -0.5 : 0.5;
      N[a] = tri[t] * h;
      if (dN)
        dN[a] = {dtri[t][0] * h, dtri[t][1] * h, tri[t] * dh};
    }
    return;
  }
  case ElementType::Pyramid5:
  {
    // The rational basis of Bedrosian (1992), which libMesh and Gmsh also use:
    // N_a = (zeta - sx_a xi - 1)(zeta - sy_a eta - 1) / [4 (1 - zeta)] at a
    // base corner with signs (sx_a, sy_a), and N_4 = zeta at the apex.  It is
    // linear on every edge and on the triangular faces and bilinear on the
    // base, so it conforms to Tet4 and Hex8 neighbours.  The denominator
    // vanishes only at the apex, which no quadrature point reaches.
    const double x = xi[0], y = xi[1], z = xi[2];
    const double den = std::max(1 - z, 1e-14);
    const double sx[4] = {-1, 1, 1, -1}, sy[4] = {-1, -1, 1, 1};
    for (int a = 0; a < 4; ++a)
    {
      const double p = z - sx[a] * x - 1, q = z - sy[a] * y - 1;
      N[a] = 0.25 * p * q / den;
      if (dN)
        dN[a] = {-0.25 * sx[a] * q / den,
                 -0.25 * sy[a] * p / den,
                 0.25 * (p + q) / den + 0.25 * p * q / (den * den)};
    }
    N[4] = z;
    if (dN)
      dN[4] = {0, 0, 1};
    return;
  }
  default:
    break;
  }
  if (isTensor())
  {
    // The shape function of a node is the product, over the directions, of the
    // one-dimensional Lagrange polynomial that is one at that node's grid
    // coordinate and zero at the others.
    double L[3][3] = {{1, 0, 0}, {1, 0, 0}, {1, 0, 0}};
    double dL[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
    for (int d = 0; d < _dim; ++d)
      lagrange1D(_grid_coordinates, xi[d], L[d], dL[d]);
    for (int a = 0; a < numNodes(); ++a)
    {
      double value = 1.0;
      for (int d = 0; d < _dim; ++d)
        value *= L[d][_grid[a][d]];
      N[a] = value;
      if (!dN)
        continue;
      Point g{0, 0, 0};
      for (int d = 0; d < _dim; ++d)
      {
        double t = dL[d][_grid[a][d]];
        for (int e = 0; e < _dim; ++e)
          if (e != d)
            t *= L[e][_grid[a][e]];
        g[d] = t;
      }
      dN[a] = g;
    }
    return;
  }

  // Simplices.  The natural variables are the barycentric coordinates
  // lambda_0 = 1 - sum(xi), lambda_i = xi_i, whose gradients are constant.
  const int nl = _dim + 1;
  double lambda[4];
  Point glambda[4];
  lambda[0] = 1.0;
  glambda[0] = {0, 0, 0};
  for (int d = 0; d < _dim; ++d)
  {
    lambda[d + 1] = xi[d];
    lambda[0] -= xi[d];
    Point g{0, 0, 0};
    g[d] = 1.0;
    glambda[d + 1] = g;
    glambda[0][d] = -1.0;
  }
  if (order() == 1)
  {
    for (int i = 0; i < nl; ++i)
    {
      N[i] = lambda[i];
      if (dN)
        dN[i] = glambda[i];
    }
    return;
  }
  // Quadratic simplex: N_i = lambda_i (2 lambda_i - 1) at a corner and
  // N_ij = 4 lambda_i lambda_j at the midpoint of edge (i, j).
  for (int i = 0; i < nl; ++i)
  {
    N[i] = lambda[i] * (2.0 * lambda[i] - 1.0);
    if (dN)
      dN[i] = (4.0 * lambda[i] - 1.0) * glambda[i];
  }
  static const int tri_edges[3][2] = {{0, 1}, {1, 2}, {2, 0}};
  static const int tet_edges[6][2] = {{0, 1}, {1, 2}, {0, 2}, {0, 3}, {1, 3}, {2, 3}};
  const int ne = _dim == 2 ? 3 : 6;
  for (int e = 0; e < ne; ++e)
  {
    const int p = _dim == 2 ? tri_edges[e][0] : tet_edges[e][0];
    const int q = _dim == 2 ? tri_edges[e][1] : tet_edges[e][1];
    N[nl + e] = 4.0 * lambda[p] * lambda[q];
    if (dN)
      dN[nl + e] = (4.0 * lambda[q]) * glambda[p] + (4.0 * lambda[p]) * glambda[q];
  }
}

void
ReferenceElement::buildTensorDual(const std::vector<double> & c)
{
  _grid_coordinates = c;
  const int nn = numNodes();
  const int n = static_cast<int>(c.size());

  // Grid index of every node in every direction.
  _grid.assign(nn, {0, 0, 0});
  for (int a = 0; a < nn; ++a)
    for (int d = 0; d < _dim; ++d)
    {
      int best = 0;
      for (int k = 1; k < n; ++k)
        if (std::abs(_nodes[a][d] - c[k]) < std::abs(_nodes[a][d] - c[best]))
          best = k;
      _grid[a][d] = best;
    }

  // The control domain of a node is the box bounded by the planes half way to
  // its neighbours; at the edge of the element the box stops at the element
  // boundary.
  const auto lower = [&](int a, int d)
  {
    const int k = _grid[a][d];
    return k == 0 ? c.front() : 0.5 * (c[k - 1] + c[k]);
  };
  const auto upper = [&](int a, int d)
  {
    const int k = _grid[a][d];
    return k == n - 1 ? c.back() : 0.5 * (c[k] + c[k + 1]);
  };

  // An axis-aligned box patch: the directions in `dirs` run from lo to hi, and
  // every other direction is held at the value it has in `base`.  The corners
  // come out in the Edge2 / Quad4 / Hex8 ordering that Patch::map expects.
  const auto boxPatch = [](const std::vector<int> & dirs,
                           const std::vector<double> & lo,
                           const std::vector<double> & hi,
                           const Point & base)
  {
    static const int sgn[8][3] = {{-1, -1, -1},
                                  {1, -1, -1},
                                  {1, 1, -1},
                                  {-1, 1, -1},
                                  {-1, -1, 1},
                                  {1, -1, 1},
                                  {1, 1, 1},
                                  {-1, 1, 1}};
    Patch p;
    p.dim = static_cast<int>(dirs.size());
    const int nc = 1 << p.dim;
    p.corners.assign(nc, base);
    for (int k = 0; k < nc; ++k)
      for (int i = 0; i < p.dim; ++i)
        p.corners[k][dirs[i]] = sgn[k][i] < 0 ? lo[i] : hi[i];
    return p;
  };

  std::vector<int> all_dirs;
  for (int d = 0; d < _dim; ++d)
    all_dirs.push_back(d);

  // ---- control domains -----------------------------------------------------
  _scv.assign(nn, {});
  for (int a = 0; a < nn; ++a)
  {
    std::vector<double> lo, hi;
    for (int d : all_dirs)
    {
      lo.push_back(lower(a, d));
      hi.push_back(upper(a, d));
    }
    _scv[a] = {boxPatch(all_dirs, lo, hi, {0, 0, 0})};
  }

  // ---- interfaces ----------------------------------------------------------
  // Two nodes share an interface when their grid indices differ by one in
  // exactly one direction.  The interface is the face of their boxes on the
  // plane half way between them.
  for (int a = 0; a < nn; ++a)
    for (int b = a + 1; b < nn; ++b)
    {
      int direction = -1, differences = 0;
      for (int d = 0; d < _dim; ++d)
        if (_grid[a][d] != _grid[b][d])
        {
          ++differences;
          direction = d;
        }
      if (differences != 1 || std::abs(_grid[a][direction] - _grid[b][direction]) != 1)
        continue;
      DualFace F;
      F.a = a;
      F.b = b;
      Point base{0, 0, 0};
      base[direction] = 0.5 * (c[_grid[a][direction]] + c[_grid[b][direction]]);
      std::vector<int> dirs;
      std::vector<double> lo, hi;
      for (int d = 0; d < _dim; ++d)
        if (d != direction)
        {
          dirs.push_back(d);
          lo.push_back(lower(a, d));
          hi.push_back(upper(a, d));
        }
      F.patch = boxPatch(dirs, lo, hi, base);
      _faces.push_back(F);
    }

  // ---- side patches --------------------------------------------------------
  const int ns = numSides();
  _side_patches.assign(ns, {});
  _side_full.assign(ns, {});
  for (int s = 0; s < ns; ++s)
  {
    const auto & sn = _sides[s];
    // A side of a tensor cell is the set of nodes whose grid index in one
    // direction is pinned to the first or the last coordinate.
    int fixed = -1;
    for (int d = 0; d < _dim; ++d)
    {
      const int k = _grid[sn.front()][d];
      if (k != 0 && k != n - 1)
        continue;
      bool same = true;
      for (int i : sn)
        same = same && _grid[i][d] == k;
      if (same)
      {
        fixed = d;
        break;
      }
    }
    if (fixed < 0)
      throw std::logic_error("dualmesh: side of a tensor element is not on a coordinate plane");
    Point base{0, 0, 0};
    base[fixed] = c[_grid[sn.front()][fixed]];
    std::vector<int> dirs;
    for (int d = 0; d < _dim; ++d)
      if (d != fixed)
        dirs.push_back(d);

    _side_patches[s].resize(sn.size());
    for (std::size_t j = 0; j < sn.size(); ++j)
    {
      std::vector<double> lo, hi;
      for (int d : dirs)
      {
        lo.push_back(lower(sn[j], d));
        hi.push_back(upper(sn[j], d));
      }
      _side_patches[s][j] = {boxPatch(dirs, lo, hi, base)};
    }
    std::vector<double> lo(dirs.size(), c.front()), hi(dirs.size(), c.back());
    _side_full[s] = {boxPatch(dirs, lo, hi, base)};
  }

  // ---- whole element -------------------------------------------------------
  {
    std::vector<double> lo(all_dirs.size(), c.front()), hi(all_dirs.size(), c.back());
    _elem_full = {boxPatch(all_dirs, lo, hi, {0, 0, 0})};
  }
}

void
ReferenceElement::buildElementGeometryOnly()
{
  // Elements that the finite element and the cell-centred finite volume
  // methods can use but the dual mesh methods cannot.  Only the whole-element
  // and whole-side patches are needed for their integrals; the control domain
  // lists stay empty, which is what supportsDualMesh() tests.
  const int ns = numSides();
  _scv.clear();
  _faces.clear();
  _side_patches.assign(ns, {});
  _side_full.assign(ns, {});
  _elem_full.clear();

  if (_type == ElementType::Pyramid5)
  {
    // The pyramid as a hexahedron whose top face is collapsed onto the apex.
    // A tensor-product Gauss rule on it is the collapsed-coordinate
    // ("conical product") rule of Duffy (1982): the Jacobian of the collapse
    // vanishes like (1 - zeta)^2, which cancels the 1 / (1 - zeta) of the
    // rational shape function gradients, so the integrand of a stiffness term
    // is a polynomial and the rule integrates it exactly.
    Patch q;
    q.dim = 3;
    q.corners = {
        _nodes[0], _nodes[1], _nodes[2], _nodes[3], _nodes[4], _nodes[4], _nodes[4], _nodes[4]};
    q.extra_points = 1;
    _elem_full = {q};
  }
  else
  {
    // Quad8 and Hex20: the corner nodes span the reference square or cube.
    Patch q;
    q.dim = _dim;
    q.corners.assign(_nodes.begin(), _nodes.begin() + (1 << _dim));
    _elem_full = {q};
  }

  for (int s = 0; s < ns; ++s)
  {
    const auto & sn = _sides[s];
    if (_dim == 2)
    {
      Patch q;
      q.dim = 1;
      q.corners = {_nodes[sn[0]], _nodes[sn[1]]};
      _side_full[s] = {q};
    }
    else if (sn.size() == 3)
      _side_full[s] = {collapsedPatch({_nodes[sn[0]], _nodes[sn[1]], _nodes[sn[2]]})};
    else
    {
      // A quadrilateral side: its first four nodes are its corners, in order.
      Patch q;
      q.dim = 2;
      q.corners = {_nodes[sn[0]], _nodes[sn[1]], _nodes[sn[2]], _nodes[sn[3]]};
      _side_full[s] = {q};
    }
  }
}

void
ReferenceElement::buildPrismDual()
{
  // The prism is a triangle times a segment, and so is its median dual: the
  // control domain of a node is the median sub-cell of its triangle vertex
  // times the half of the segment on its side.  That product is exactly the
  // general median construction (bounded by edge midpoints, face centroids and
  // the element centroid), so on a triangular face it matches the dual of a
  // neighbouring tetrahedron and on a quadrilateral face that of a
  // neighbouring hexahedron.
  const auto at = [](const Point & p, double z) { return Point{p[0], p[1], z}; };
  Point p[3];
  for (int i = 0; i < 3; ++i)
    p[i] = {_nodes[i][0], _nodes[i][1], 0};
  const Point c = average({p[0], p[1], p[2]});
  const auto other = [](int i, int k) { return (i + 1 + k) % 3; };
  // The median quadrilateral of triangle vertex i, in the plane.
  const auto median = [&](int i)
  { return std::array<Point, 4>{p[i], mid(p[i], p[other(i, 0)]), c, mid(p[i], p[other(i, 1)])}; };
  const auto level = [](int node) { return node < 3 ? -1.0 : 1.0; };

  // ---- control domains -----------------------------------------------------
  _scv.assign(6, {});
  for (int a = 0; a < 6; ++a)
  {
    const auto q = median(a % 3);
    const double z0 = level(a);
    Patch h;
    h.dim = 3;
    h.corners = {at(q[0], z0),
                 at(q[1], z0),
                 at(q[2], z0),
                 at(q[3], z0),
                 at(q[0], 0),
                 at(q[1], 0),
                 at(q[2], 0),
                 at(q[3], 0)};
    _scv[a] = {h};
  }

  // ---- interfaces ----------------------------------------------------------
  _faces.clear();
  for (int off : {0, 3})
    for (int i = 0; i < 3; ++i)
      for (int j = i + 1; j < 3; ++j)
      {
        // Between two nodes of the same triangle: the triangle's median
        // segment, extruded over the node's half of the segment.
        const Point m = mid(p[i], p[j]);
        const double z0 = level(off);
        DualFace f;
        f.a = i + off;
        f.b = j + off;
        f.patch.dim = 2;
        f.patch.corners = {at(m, z0), at(c, z0), at(c, 0), at(m, 0)};
        _faces.push_back(f);
      }
  for (int i = 0; i < 3; ++i)
  {
    // Between a node and the one above it: the median quadrilateral at the
    // mid-height of the prism.
    const auto q = median(i);
    DualFace f;
    f.a = i;
    f.b = i + 3;
    f.patch.dim = 2;
    f.patch.corners = {at(q[0], 0), at(q[1], 0), at(q[2], 0), at(q[3], 0)};
    _faces.push_back(f);
  }

  // ---- sides ---------------------------------------------------------------
  const int ns = numSides();
  _side_patches.assign(ns, {});
  _side_full.assign(ns, {});
  for (int s = 0; s < ns; ++s)
  {
    const auto & sn = _sides[s];
    _side_patches[s].assign(sn.size(), {});
    if (sn.size() == 3)
    {
      // A triangular end face: each node keeps its median quadrilateral.
      for (std::size_t j = 0; j < sn.size(); ++j)
      {
        const auto q = median(sn[j] % 3);
        const double z0 = level(sn[j]);
        Patch r;
        r.dim = 2;
        r.corners = {at(q[0], z0), at(q[1], z0), at(q[2], z0), at(q[3], z0)};
        _side_patches[s][j] = {r};
      }
      _side_full[s] = {collapsedPatch({_nodes[sn[0]], _nodes[sn[1]], _nodes[sn[2]]})};
      continue;
    }
    // A quadrilateral side over the triangle edge (t, t'): each node keeps the
    // half of the edge on its side, over the half of the height on its side.
    for (std::size_t j = 0; j < sn.size(); ++j)
    {
      const int t = sn[j] % 3;
      int partner = -1;
      for (int k : sn)
        if (k % 3 != t)
          partner = k % 3;
      const Point m = mid(p[t], p[partner]);
      const double z0 = level(sn[j]);
      Patch r;
      r.dim = 2;
      r.corners = {at(p[t], z0), at(m, z0), at(m, 0), at(p[t], 0)};
      _side_patches[s][j] = {r};
    }
    Patch whole;
    whole.dim = 2;
    whole.corners = {_nodes[sn[0]], _nodes[sn[1]], _nodes[sn[2]], _nodes[sn[3]]};
    _side_full[s] = {whole};
  }

  // ---- whole element -------------------------------------------------------
  _elem_full = {collapsedPatch(_nodes)};
}

bool
ReferenceElement::contains(const Point & xi, double tol) const
{
  switch (_type)
  {
  case ElementType::Tri3:
  case ElementType::Tri6:
  case ElementType::Tet4:
  case ElementType::Tet10:
  {
    double sum = 0;
    for (int d = 0; d < _dim; ++d)
    {
      if (xi[d] < -tol)
        return false;
      sum += xi[d];
    }
    return sum <= 1 + tol;
  }
  case ElementType::Wedge6:
    return xi[0] >= -tol && xi[1] >= -tol && xi[0] + xi[1] <= 1 + tol && std::abs(xi[2]) <= 1 + tol;
  case ElementType::Pyramid5:
    return xi[2] >= -tol && xi[2] <= 1 + tol && std::abs(xi[0]) <= 1 - xi[2] + tol &&
           std::abs(xi[1]) <= 1 - xi[2] + tol;
  default:
    for (int d = 0; d < _dim; ++d)
      if (std::abs(xi[d]) > 1 + tol)
        return false;
    return true;
  }
}

std::string
ReferenceElement::dualMeshLimitation() const
{
  switch (_type)
  {
  case ElementType::Quad8:
  case ElementType::Hex20:
    return "it is a serendipity element: it has mid-edge nodes but no interior node, so "
           "when the element is divided into node-centred control domains there is no "
           "sub-cell to give the element centre, and the median dual is not defined";
  case ElementType::Pyramid5:
    return "its apex is shared by four edges rather than three, so the median sub-cell of "
           "the apex is not a hexahedral patch, and the non-tensor decomposition it would "
           "need to match the control domain interfaces exactly is not implemented";
  default:
    return "";
  }
}

void
ReferenceElement::addSimplexSubCell(const std::vector<int> & verts,
                                    const std::vector<int> & slots,
                                    int dim,
                                    std::vector<std::vector<Patch>> & scv,
                                    std::vector<DualFace> * faces) const
{
  const int nv = static_cast<int>(verts.size());
  std::vector<Point> X;
  for (int v : verts)
    X.push_back(_nodes[v]);
  Point cc{0, 0, 0};
  for (const auto & p : X)
    cc = cc + p;
  cc = (1.0 / nv) * cc;
  const auto face = [&](int p, int q, int r) { return average({X[p], X[q], X[r]}); };

  // The median sub-cell of each vertex: the region of the sub-simplex that is
  // closer to that vertex than to any other, in the median (not the
  // perpendicular-bisector) sense.
  for (int i = 0; i < nv; ++i)
  {
    std::vector<int> other;
    for (int k = 0; k < nv; ++k)
      if (k != i)
        other.push_back(k);
    Patch P;
    P.dim = dim;
    if (dim == 1)
      P.corners = {X[i], cc};
    else if (dim == 2)
      P.corners = {X[i], mid(X[i], X[other[0]]), cc, mid(X[i], X[other[1]])};
    else
    {
      const int b = other[0], d = other[1], e = other[2];
      P.corners = {X[i],
                   mid(X[i], X[b]),
                   face(i, b, d),
                   mid(X[i], X[d]),
                   mid(X[i], X[e]),
                   face(i, b, e),
                   cc,
                   face(i, d, e)};
    }
    scv[slots[i]].push_back(P);
  }

  if (!faces)
    return;
  // The interface between two vertices passes through the midpoint of their
  // edge, the centroids of the faces that share that edge, and the centroid of
  // the sub-simplex.
  for (int i = 0; i < nv; ++i)
    for (int j = i + 1; j < nv; ++j)
    {
      DualFace F;
      F.a = verts[i];
      F.b = verts[j];
      F.patch.dim = dim - 1;
      const Point m = mid(X[i], X[j]);
      if (dim == 1)
        F.patch.corners = {m};
      else if (dim == 2)
        F.patch.corners = {m, cc};
      else
      {
        std::vector<int> other;
        for (int k = 0; k < nv; ++k)
          if (k != i && k != j)
            other.push_back(k);
        F.patch.corners = {m, face(i, j, other[0]), cc, face(i, j, other[1])};
      }
      faces->push_back(F);
    }
}

void
ReferenceElement::buildSubdividedSimplexDual(const std::vector<std::vector<int>> & sub_simplices)
{
  const int nn = numNodes();
  _scv.assign(nn, {});
  for (const auto & verts : sub_simplices)
    addSimplexSubCell(verts, verts, _dim, _scv, &_faces);

  // ---- side patches --------------------------------------------------------
  // The side is refined in exactly the same way, so that the traces of the
  // control domains on the boundary agree with the control domains themselves.
  static const std::vector<std::vector<int>> linear_edge = {{0, 1}};
  static const std::vector<std::vector<int>> quadratic_edge = {{0, 2}, {2, 1}};
  static const std::vector<std::vector<int>> linear_triangle = {{0, 1, 2}};
  static const std::vector<std::vector<int>> quadratic_triangle = {
      {0, 3, 5}, {3, 1, 4}, {5, 4, 2}, {3, 4, 5}};

  const int ns = numSides();
  _side_patches.assign(ns, {});
  _side_full.assign(ns, {});
  for (int s = 0; s < ns; ++s)
  {
    const auto & sn = _sides[s];
    const std::vector<std::vector<int>> * side_sub = nullptr;
    if (_dim == 2)
      side_sub = sn.size() == 2 ? &linear_edge : &quadratic_edge;
    else
      side_sub = sn.size() == 3 ? &linear_triangle : &quadratic_triangle;

    std::vector<std::vector<Patch>> patches(sn.size());
    for (const auto & ss : *side_sub)
    {
      std::vector<int> verts;
      for (int k : ss)
        verts.push_back(sn[k]);
      addSimplexSubCell(verts, ss, _dim - 1, patches, nullptr);
    }
    _side_patches[s] = patches;

    // For finite element boundary integration the side is used whole.  In
    // reference coordinates the side is straight even when the element is
    // quadratic, so its corner nodes describe it exactly.
    if (_dim == 2)
    {
      Patch q;
      q.dim = 1;
      q.corners = {_nodes[sn[0]], _nodes[sn[1]]};
      _side_full[s] = {q};
    }
    else
      _side_full[s] = {collapsedPatch({_nodes[sn[0]], _nodes[sn[1]], _nodes[sn[2]]})};
  }

  // ---- whole element -------------------------------------------------------
  // The whole simplex as one collapsed square or cube over its corner nodes.
  std::vector<Point> corners(_nodes.begin(), _nodes.begin() + _dim + 1);
  _elem_full = {collapsedPatch(corners)};
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
  if (text == "automatic" || text == "auto")
  {
    q.kind = Kind::Gauss;
    q.points = 2;
    q.automatic = true;
  }
  else if (text == "trapezoid" || text == "trapezoidal")
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
                     "'. Use automatic, gauss1..gauss10, midpoint, trapezoid, simpson, nodal, "
                     "interface, or control_domain_trapezoid.");
  return q;
}

std::string
QuadratureSpec::str() const
{
  std::string s;
  switch (kind)
  {
  case Kind::Gauss:
    s = automatic ? "automatic" : "gauss" + std::to_string(points);
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

namespace
{
constexpr int kMaxGaussPoints = 20;
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
  // Gauss-Legendre rules are computed once, by Newton iteration on the
  // Legendre polynomial, and then looked up: they are asked for on every
  // element of every assembly.  The initialisation of a function-local static
  // is thread-safe.
  static const auto table = []
  {
    std::vector<std::pair<std::vector<double>, std::vector<double>>> t(kMaxGaussPoints + 1);
    for (int n = 1; n <= kMaxGaussPoints; ++n)
    {
      auto & [xs, ws] = t[n];
      xs.resize(n);
      ws.resize(n);
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
        xs[n - 1 - i] = z;
        ws[n - 1 - i] = 2.0 / ((1 - z * z) * dp * dp);
      }
      if (n == 1)
      {
        xs[0] = 0;
        ws[0] = 2;
      }
    }
    return t;
  }();
  const int n = spec.points;
  if (n < 1 || n > kMaxGaussPoints)
    throw InputError("A Gauss rule needs between 1 and " + std::to_string(kMaxGaussPoints) +
                     " points per direction; " + std::to_string(n) + " were asked for.");
  x = table[n].first;
  w = table[n].second;
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
