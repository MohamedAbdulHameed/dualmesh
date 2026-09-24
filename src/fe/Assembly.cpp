// SPDX-License-Identifier: LGPL-2.1-or-later
#include "dualmesh/fe/Assembly.h"
#include "dualmesh/core/InputParameters.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <stdexcept>
#include <tuple>

namespace dualmesh
{

namespace
{
/// The part of mapPoint that depends on the element's nodes: the position,
/// the Jacobian of the map, its determinant and inverse, and the physical
/// gradients of the shape functions, from the shape functions @p N and their
/// reference gradients @p dNr at @p xi.  It runs for every integration point
/// of every element in every assembly, so the dimension is a template
/// parameter and the small loops are unrolled.
template <int DIM>
void
mapWithShapeDim(const Mesh & mesh,
                const Element & el,
                int nn,
                const Point & xi,
                const double * N,
                const Point * dNr,
                MappedPoint & mp)
{
  if (N != mp.N)
    std::copy(N, N + nn, mp.N);
  mp.xi = xi;
  double x[3] = {0, 0, 0};
  double J[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
  for (int k = 0; k < nn; ++k)
  {
    const Point & X = mesh.node(el.nodes[k]);
    const double n = N[k];
    for (int i = 0; i < 3; ++i)
      x[i] += n * X[i];
    for (int i = 0; i < DIM; ++i)
      for (int j = 0; j < DIM; ++j)
        J[i][j] += X[i] * dNr[k][j];
  }
  mp.x = {x[0], x[1], x[2]};
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      mp.J[i][j] = J[i][j];
  double inv[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
  if constexpr (DIM == 1)
  {
    mp.detJ = J[0][0];
    inv[0][0] = 1.0 / J[0][0];
  }
  else if constexpr (DIM == 2)
  {
    mp.detJ = J[0][0] * J[1][1] - J[0][1] * J[1][0];
    const double id = 1.0 / mp.detJ;
    inv[0][0] = J[1][1] * id;
    inv[0][1] = -J[0][1] * id;
    inv[1][0] = -J[1][0] * id;
    inv[1][1] = J[0][0] * id;
  }
  else
  {
    mp.detJ = J[0][0] * (J[1][1] * J[2][2] - J[1][2] * J[2][1]) -
              J[0][1] * (J[1][0] * J[2][2] - J[1][2] * J[2][0]) +
              J[0][2] * (J[1][0] * J[2][1] - J[1][1] * J[2][0]);
    const double id = 1.0 / mp.detJ;
    inv[0][0] = (J[1][1] * J[2][2] - J[1][2] * J[2][1]) * id;
    inv[0][1] = (J[0][2] * J[2][1] - J[0][1] * J[2][2]) * id;
    inv[0][2] = (J[0][1] * J[1][2] - J[0][2] * J[1][1]) * id;
    inv[1][0] = (J[1][2] * J[2][0] - J[1][0] * J[2][2]) * id;
    inv[1][1] = (J[0][0] * J[2][2] - J[0][2] * J[2][0]) * id;
    inv[1][2] = (J[0][2] * J[1][0] - J[0][0] * J[1][2]) * id;
    inv[2][0] = (J[1][0] * J[2][1] - J[1][1] * J[2][0]) * id;
    inv[2][1] = (J[0][1] * J[2][0] - J[0][0] * J[2][1]) * id;
    inv[2][2] = (J[0][0] * J[1][1] - J[0][1] * J[1][0]) * id;
  }
  if (mp.detJ == 0.0 || !std::isfinite(mp.detJ))
    throw std::runtime_error("dualmesh: degenerate element (zero Jacobian determinant)");
  // grad N = J^{-T} dN/dxi
  for (int k = 0; k < nn; ++k)
  {
    Point g{0, 0, 0};
    for (int i = 0; i < DIM; ++i)
      for (int j = 0; j < DIM; ++j)
        g[i] += inv[j][i] * dNr[k][j];
    mp.dN[k] = g;
  }
}

void
mapWithShape(const Mesh & mesh,
             const Element & el,
             int dim,
             int nn,
             const Point & xi,
             const double * N,
             const Point * dNr,
             MappedPoint & mp)
{
  if (dim == 3)
    mapWithShapeDim<3>(mesh, el, nn, xi, N, dNr, mp);
  else if (dim == 2)
    mapWithShapeDim<2>(mesh, el, nn, xi, N, dNr, mp);
  else
    mapWithShapeDim<1>(mesh, el, nn, xi, N, dNr, mp);
}
} // namespace

void
mapPoint(const Mesh & mesh, const Element & el, const Point & xi, MappedPoint & mp)
{
  const auto & ref = ReferenceElement::get(el.type);
  Point dNr[kMaxElementNodes];
  ref.shape(xi, mp.N, dNr);
  mapWithShape(mesh, el, ref.dimension(), ref.numNodes(), xi, mp.N, dNr, mp);
}

namespace
{
/// Physical tangents of a patch: t_k = J * (dxi/ds_k).
Point
applyJ(const MappedPoint & mp, const Point & T, int dim)
{
  Point t{0, 0, 0};
  for (int i = 0; i < dim; ++i)
    for (int j = 0; j < dim; ++j)
      t[i] += mp.J[i][j] * T[j];
  return t;
}

double
patchDet(const std::array<Point, 3> & T, int dim)
{
  if (dim == 0)
    return 1.0;
  if (dim == 1)
    return T[0][0];
  if (dim == 2)
    return T[0][0] * T[1][1] - T[0][1] * T[1][0];
  return dot(T[0], cross(T[1], T[2]));
}

/// Oriented area vector (n dA per unit parametric measure) of a patch of
/// dimension dim-1 in a dim-dimensional element.
Point
areaVector(const MappedPoint & mp, const std::array<Point, 3> & T, int dim)
{
  if (dim == 1)
    return {1.0, 0, 0};
  if (dim == 2)
  {
    const Point t = applyJ(mp, T[0], 2);
    return {t[1], -t[0], 0};
  }
  return cross(applyJ(mp, T[0], 3), applyJ(mp, T[1], 3));
}

/// The tensor-product rule for one patch, with the extra Gauss points a
/// collapsed (simplex, prism or pyramid) patch needs.
void
patchRule(const QuadratureSpec & spec,
          const Patch & patch,
          std::vector<Point> & s,
          std::vector<double> & w)
{
  if (patch.extra_points > 0 && spec.kind == QuadratureSpec::Kind::Gauss)
  {
    QuadratureSpec more = spec;
    more.points = std::min(20, spec.points + patch.extra_points);
    ruleTensor(more, patch.dim, s, w);
    return;
  }
  ruleTensor(spec, patch.dim, s, w);
}

double
patchMeasure(const Mesh & mesh, const Element & el, const Patch & patch)
{
  QuadratureSpec g;
  g.kind = QuadratureSpec::Kind::Gauss;
  g.points = 3;
  std::vector<Point> s;
  std::vector<double> w;
  patchRule(g, patch, s, w);
  const int dim = elementDimension(el.type);
  MappedPoint mp;
  double m = 0;
  for (std::size_t q = 0; q < s.size(); ++q)
  {
    std::array<Point, 3> T;
    const Point xi = patch.map(s[q], &T);
    mapPoint(mesh, el, xi, mp);
    if (patch.dim == dim)
      m += w[q] * std::abs(mp.detJ * patchDet(T, dim));
    else
      m += w[q] * norm(areaVector(mp, T, dim));
  }
  return m;
}

/// The measure of a control domain that is described by several patches: the
/// patches of one control domain never overlap, so the measure of their union
/// is the sum of their measures.
double
patchMeasure(const Mesh & mesh, const Element & el, const std::vector<Patch> & patches)
{
  double m = 0;
  for (const auto & p : patches)
    m += patchMeasure(mesh, el, p);
  return m;
}

/// One representative point for a control domain, in reference coordinates.
///
/// The point taken from a patch is the corner of that patch that lies deepest
/// inside the element, that is, the corner closest to the element centroid.
/// In one dimension this is the control-domain interface, which is what the
/// finite volume source rule of the book evaluates; in two and three
/// dimensions, and for a linear element, it is the element centroid.  When the
/// control domain is the union of several sub-cells, as it is for a quadratic
/// simplex, the points of the sub-cells are averaged with their measures as
/// weights.  The weight of the integration point is always the measure of the
/// whole control domain, so the rule is exact for a constant source whichever
/// point is chosen.
Point
controlVolumePoint(const Mesh & mesh, const Element & el, const std::vector<Patch> & patches)
{
  const Point & centroid = ReferenceElement::get(el.type).centroid();
  const auto deepest = [&](const Patch & p)
  {
    const Point * best = &p.corners.front();
    double best_distance = norm(*best - centroid);
    for (const auto & corner : p.corners)
    {
      const double d = norm(corner - centroid);
      if (d < best_distance)
      {
        best_distance = d;
        best = &corner;
      }
    }
    return *best;
  };
  if (patches.size() == 1)
    return deepest(patches.front());
  Point x{0, 0, 0};
  double total = 0;
  for (const auto & p : patches)
  {
    const double m = patchMeasure(mesh, el, p);
    x = x + m * deepest(p);
    total += m;
  }
  return (1.0 / total) * x;
}
} // namespace

double
elementMeasure(const Mesh & mesh, Index e)
{
  const auto & el = mesh.element(e);
  const auto & ref = ReferenceElement::get(el.type);
  double m = 0;
  for (const auto & p : ref.elementPatches())
    m += patchMeasure(mesh, el, p);
  return m;
}

namespace
{
/// An integration point as it is on the reference element: everything about
/// it that does not depend on where the element's nodes are.
struct ReferencePoint
{
  Point xi;
  /// The rule's weight, times |det(dxi/ds)| of the patch for a volume point.
  double weight;
  std::array<Point, 3> tangents; ///< dxi/ds_k, for the area of a face point
  int owner, neighbor;
  double N[kMaxElementNodes];
  Point dNr[kMaxElementNodes];
};

/// Symmetric quadrature rules for the triangle and the tetrahedron.
///
/// On a simplex the collapsed-coordinate rule of patchRule needs n + 1 Gauss
/// points in every direction to be exact to degree 2n - 1, which is 27 points
/// for the default rule on a tetrahedron.  A rule whose points form orbits of
/// the symmetry group of the simplex needs far fewer: 8 for degree 3.  The
/// rules below have positive weights and interior points.  Each orbit is
/// given by its weight (the integral over the reference simplex, of area 1/2
/// or volume 1/6, carried by each of its points) and one barycentric
/// coordinate a: a 3-orbit of the triangle holds the permutations of
/// (a, a, 1 - 2a); a 4-orbit of the tetrahedron those of (a, a, a, 1 - 3a);
/// a 6-orbit those of (a, a, 1/2 - a, 1/2 - a).  The triangle rules of degree
/// 4 and 5 are those of Dunavant (Int. J. Numer. Methods Eng. 21 (1985)
/// 1129-1148) and the tetrahedron rule of degree 2 is the classical four-point
/// rule; the others were computed for this library by solving the moment
/// equations.  All of them were recomputed to 40 digits, and the unit test
/// simplex_quadrature_is_exact checks every monomial up to the stated degree.
struct Orbit
{
  int size; ///< 1 (the centroid), 3, 4 or 6
  double weight;
  double a;
};
struct SimplexRule
{
  int degree;
  std::vector<Orbit> orbits;
};

const std::vector<SimplexRule> &
simplexRules(int dim)
{
  static const std::vector<SimplexRule> triangle = {
      {1, {{1, 0.5, 0.0}}},
      {4,
       {{3, 0.11169079483900573285, 0.44594849091596488632},
        {3, 0.054975871827660933819, 0.09157621350977074346}}},
      {5,
       {{1, 0.1125, 0.0},
        {3, 0.066197076394253090369, 0.47014206410511508977},
        {3, 0.062969590272413576298, 0.1012865073234563388}}},
  };
  static const std::vector<SimplexRule> tetrahedron = {
      {1, {{1, 1.0 / 6.0, 0.0}}},
      {2, {{4, 1.0 / 24.0, 0.13819660112501051518}}},
      {3,
       {{4, 0.021197385985612709744, 0.11384180112626613797},
        {4, 0.020469280681053956923, 0.32903247342811021732}}},
      {5,
       {{1, 0.016115857201819612382, 0.0},
        {4, 0.012053394621802986443, 0.092162071944342712782},
        {4, 0.012981443355599784009, 0.31750572624599253245},
        {6, 0.0084019095925393287465, 0.053994675478648660432}}},
  };
  return dim == 2 ? triangle : tetrahedron;
}

} // namespace

bool
simplexQuadrature(int dim, int degree, std::vector<Point> & points, std::vector<double> & weights)
{
  points.clear();
  weights.clear();
  if (dim != 2 && dim != 3)
    return false;
  for (const auto & rule : simplexRules(dim))
  {
    if (rule.degree < degree)
      continue;
    // The reference simplex has its right-angle vertex at the origin, so the
    // reference coordinates of a point are its barycentric coordinates
    // lambda_1 .. lambda_dim.
    for (const auto & o : rule.orbits)
    {
      std::vector<std::array<double, 4>> lambdas;
      if (o.size == 1)
        lambdas.push_back({1.0 / (dim + 1), 1.0 / (dim + 1), 1.0 / (dim + 1), 1.0 / (dim + 1)});
      else if (dim == 2)
      {
        const double b = 1.0 - 2.0 * o.a;
        lambdas = {{o.a, o.a, b, 0}, {o.a, b, o.a, 0}, {b, o.a, o.a, 0}};
      }
      else if (o.size == 4)
      {
        const double b = 1.0 - 3.0 * o.a;
        lambdas = {{o.a, o.a, o.a, b}, {o.a, o.a, b, o.a}, {o.a, b, o.a, o.a}, {b, o.a, o.a, o.a}};
      }
      else
      {
        const double b = 0.5 - o.a;
        lambdas = {{o.a, o.a, b, b},
                   {o.a, b, o.a, b},
                   {o.a, b, b, o.a},
                   {b, o.a, o.a, b},
                   {b, o.a, b, o.a},
                   {b, b, o.a, o.a}};
      }
      for (const auto & l : lambdas)
      {
        points.push_back({l[1], l[2], dim == 3 ? l[3] : 0.0});
        weights.push_back(o.weight);
      }
    }
    return true;
  }
  return false;
}

namespace
{
/// The reference points of one element type, rule and point set.  They are
/// the same for every element of the type, so they are built once, on first
/// use, instead of once per element per assembly; each thread keeps its own
/// table, so no lock is needed.
const std::vector<ReferencePoint> &
referencePoints(ElementType type, bool dual_mesh, PointSet set, const QuadratureSpec & spec)
{
  using Key = std::tuple<int, bool, int, int, int, bool>;
  thread_local std::map<Key, std::vector<ReferencePoint>> table;
  const Key key{static_cast<int>(type),
                dual_mesh,
                static_cast<int>(set),
                static_cast<int>(spec.kind),
                spec.points,
                spec.reduced};
  auto found = table.find(key);
  if (found != table.end())
    return found->second;

  const auto & ref = ReferenceElement::get(type);
  const int dim = ref.dimension();
  std::vector<ReferencePoint> points;
  std::vector<Point> s;
  std::vector<double> w;
  const auto add =
      [&](const Point & xi, double weight, const std::array<Point, 3> & T, int owner, int neighbor)
  {
    ReferencePoint p;
    p.xi = xi;
    p.weight = weight;
    p.tangents = T;
    p.owner = owner;
    p.neighbor = neighbor;
    ref.shape(xi, p.N, p.dNr);
    points.push_back(p);
  };
  // A symmetric simplex rule, where there is one, for the whole element; the
  // triangular prism takes the product of a triangle rule and a Gauss rule.
  const bool simplex = type == ElementType::Tri3 || type == ElementType::Tri6 ||
                       type == ElementType::Tet4 || type == ElementType::Tet10;
  const bool prism = type == ElementType::Wedge6;
  if (set == PointSet::Volume && !dual_mesh && (simplex || prism) &&
      spec.kind == QuadratureSpec::Kind::Gauss &&
      simplexQuadrature(simplex ? dim : 2, 2 * spec.points - 1, s, w))
  {
    const std::array<Point, 3> none{};
    if (simplex)
      for (std::size_t q = 0; q < s.size(); ++q)
        add(s[q], w[q], none, -1, -1);
    else
    {
      QuadratureSpec line = spec;
      std::vector<Point> z;
      std::vector<double> wz;
      ruleTensor(line, 1, z, wz);
      for (std::size_t k = 0; k < z.size(); ++k)
        for (std::size_t q = 0; q < s.size(); ++q)
          add({s[q][0], s[q][1], z[k][0]}, w[q] * wz[k], none, -1, -1);
    }
    return table.emplace(key, std::move(points)).first->second;
  }
  if (set == PointSet::Volume)
  {
    const auto addPatches = [&](const std::vector<Patch> & patches, int owner)
    {
      for (const auto & patch : patches)
      {
        patchRule(spec, patch, s, w);
        for (std::size_t q = 0; q < s.size(); ++q)
        {
          std::array<Point, 3> T;
          const Point xi = patch.map(s[q], &T);
          add(xi, w[q] * std::abs(patchDet(T, dim)), T, owner, -1);
        }
      }
    };
    if (dual_mesh)
      for (int a = 0; a < ref.numNodes(); ++a)
        addPatches(ref.controlVolume(a), a);
    else
      addPatches(ref.elementPatches(), -1);
  }
  else
  {
    QuadratureSpec fs = spec;
    if (fs.kind == QuadratureSpec::Kind::Nodal || fs.kind == QuadratureSpec::Kind::Interface ||
        fs.kind == QuadratureSpec::Kind::ControlDomainTrapezoid)
    {
      fs.kind = QuadratureSpec::Kind::Gauss;
      fs.points = 2;
    }
    for (const auto & face : ref.dualFaces())
    {
      ruleTensor(fs, face.patch.dim, s, w);
      for (std::size_t q = 0; q < s.size(); ++q)
      {
        std::array<Point, 3> T;
        const Point xi = face.patch.map(s[q], &T);
        add(xi, w[q], T, face.a, face.b);
      }
    }
  }
  return table.emplace(key, std::move(points)).first->second;
}
} // namespace

void
buildElementPoints(const Mesh & mesh,
                   Index e,
                   bool dual_mesh,
                   PointSet set,
                   const QuadratureSpec & spec,
                   std::vector<IntegrationPoint> & out,
                   std::vector<MappedPoint> * mapped)
{
  out.clear();
  if (mapped)
    mapped->clear();
  const auto & el = mesh.element(e);
  const auto & ref = ReferenceElement::get(el.type);
  const int dim = ref.dimension();
  const Point centroid = ref.centroid();
  std::vector<Point> s;
  std::vector<double> w;
  MappedPoint mp;

  if (!dual_mesh && set == PointSet::Faces)
    return;

  if (set == PointSet::Volume)
  {
    if (spec.kind == QuadratureSpec::Kind::ControlDomainTrapezoid)
    {
      // Trapezoidal rule over the whole control domain.  For a node inside the
      // domain the control domain is bounded by interfaces only, so the rule
      // uses the interface points of each sub-cell; the (half) control domain
      // of a boundary node is bounded by the node itself as well, which
      // contributes with half the weight.
      const auto & on_boundary = mesh.boundaryNodeMarkers();
      for (int a = 0; a < ref.numNodes(); ++a)
      {
        const double measure = patchMeasure(mesh, el, ref.controlVolume(a));
        const bool boundary_node = on_boundary[el.nodes[a]] != 0;
        IntegrationPoint ip;
        ip.area = {0, 0, 0};
        ip.owner = a;
        ip.neighbor = -1;
        ip.xi = controlVolumePoint(mesh, el, ref.controlVolume(a));
        ip.xi_field = spec.reduced ? centroid : ip.xi;
        ip.weight = boundary_node ? 0.5 * measure : measure;
        out.push_back(ip);
        if (boundary_node)
        {
          ip.xi = ref.node(a);
          ip.xi_field = spec.reduced ? centroid : ip.xi;
          ip.weight = 0.5 * measure;
          out.push_back(ip);
        }
      }
      return;
    }
    if (spec.kind == QuadratureSpec::Kind::Nodal || spec.kind == QuadratureSpec::Kind::Interface)
    {
      const bool at_node = spec.kind == QuadratureSpec::Kind::Nodal;
      // Lumped: one point at each node.  DMCDM weight = |CD_a within e|;
      // FEM weight = \int psi_a (row-sum lumping).
      for (int a = 0; a < ref.numNodes(); ++a)
      {
        IntegrationPoint ip;
        ip.xi = at_node ? ref.node(a) : controlVolumePoint(mesh, el, ref.controlVolume(a));
        ip.xi_field = spec.reduced ? centroid : ip.xi;
        ip.area = {0, 0, 0};
        ip.owner = a;
        ip.neighbor = -1;
        if (dual_mesh)
          ip.weight = patchMeasure(mesh, el, ref.controlVolume(a));
        else
        {
          QuadratureSpec g;
          g.points = 3;
          double m = 0;
          for (const auto & patch : ref.elementPatches())
          {
            patchRule(g, patch, s, w);
            for (std::size_t q = 0; q < s.size(); ++q)
            {
              std::array<Point, 3> T;
              const Point xi = patch.map(s[q], &T);
              mapPoint(mesh, el, xi, mp);
              m += w[q] * std::abs(mp.detJ * patchDet(T, dim)) * (at_node ? mp.N[a] : 1.0);
            }
          }
          ip.weight = m;
        }
        out.push_back(ip);
      }
      return;
    }
    if (!dual_mesh && spec.reduced)
    {
      IntegrationPoint ip;
      ip.xi = centroid;
      ip.xi_field = centroid;
      ip.weight = elementMeasure(mesh, e);
      ip.area = {0, 0, 0};
      ip.owner = -1;
      ip.neighbor = -1;
      out.push_back(ip);
      return;
    }
    const int nn = ref.numNodes();
    for (const auto & rp : referencePoints(el.type, dual_mesh, PointSet::Volume, spec))
    {
      mapWithShape(mesh, el, dim, nn, rp.xi, rp.N, rp.dNr, mp);
      IntegrationPoint ip;
      ip.xi = rp.xi;
      ip.xi_field = spec.reduced ? centroid : ip.xi;
      ip.weight = rp.weight * std::abs(mp.detJ);
      ip.area = {0, 0, 0};
      ip.owner = rp.owner;
      ip.neighbor = -1;
      out.push_back(ip);
      if (mapped)
        mapped->push_back(mp);
    }
    return;
  }

  // Interfaces between control volumes.
  const int nn = ref.numNodes();
  for (const auto & rp : referencePoints(el.type, dual_mesh, PointSet::Faces, spec))
  {
    mapWithShape(mesh, el, dim, nn, rp.xi, rp.N, rp.dNr, mp);
    const Point xab = mesh.node(el.nodes[rp.neighbor]) - mesh.node(el.nodes[rp.owner]);
    Point n = areaVector(mp, rp.tangents, dim);
    if (dot(n, xab) < 0)
      n = -1.0 * n;
    IntegrationPoint ip;
    ip.xi = rp.xi;
    // The faces are integrated with a Gauss rule whatever the kind of `spec`,
    // but a reduced rule still evaluates the field at the centroid.
    ip.xi_field = spec.reduced ? centroid : ip.xi;
    ip.area = rp.weight * n;
    ip.weight = norm(ip.area);
    ip.owner = rp.owner;
    ip.neighbor = rp.neighbor;
    out.push_back(ip);
    if (mapped)
      mapped->push_back(mp);
  }
}

void
buildSidePoints(const Mesh & mesh,
                const Side & side,
                bool dual_mesh,
                const QuadratureSpec & spec,
                std::vector<IntegrationPoint> & out)
{
  out.clear();
  const auto & el = mesh.element(side.first);
  const auto & ref = ReferenceElement::get(el.type);
  const int dim = ref.dimension();
  const Point xc = mesh.elementCentroid(side.first);
  const auto & sn = ref.sideNodes(side.second);
  std::vector<Point> s;
  std::vector<double> w;
  MappedPoint mp;
  QuadratureSpec qs = spec;
  const bool nodal = qs.kind == QuadratureSpec::Kind::Nodal ||
                     qs.kind == QuadratureSpec::Kind::Interface ||
                     qs.kind == QuadratureSpec::Kind::ControlDomainTrapezoid;
  if (nodal)
  {
    qs.kind = QuadratureSpec::Kind::Gauss;
    qs.points = 3;
  }

  const auto addPatches = [&](const std::vector<Patch> & patches, int owner)
  {
    std::vector<IntegrationPoint> pts;
    for (const auto & patch : patches)
    {
      patchRule(qs, patch, s, w);
      for (std::size_t q = 0; q < s.size(); ++q)
      {
        std::array<Point, 3> T;
        IntegrationPoint ip;
        ip.xi = patch.map(s[q], &T);
        ip.xi_field = qs.reduced ? ref.centroid() : ip.xi;
        mapPoint(mesh, el, ip.xi, mp);
        Point n = areaVector(mp, T, dim);
        if (dot(n, mp.x - xc) < 0)
          n = -1.0 * n;
        ip.area = w[q] * n;
        ip.weight = norm(ip.area);
        ip.owner = owner;
        ip.neighbor = -1;
        pts.push_back(ip);
      }
    }
    if (pts.empty())
      return;
    if (nodal && owner >= 0)
    {
      // Collapse onto the node, keeping the total (vector) area.
      IntegrationPoint ip = pts.front();
      ip.xi = ref.node(owner);
      ip.xi_field = qs.reduced ? ref.centroid() : ip.xi;
      ip.area = {0, 0, 0};
      for (const auto & p : pts)
        ip.area = ip.area + p.area;
      ip.weight = 0;
      for (const auto & p : pts)
        ip.weight += p.weight;
      out.push_back(ip);
    }
    else
      out.insert(out.end(), pts.begin(), pts.end());
  };

  if (dual_mesh)
  {
    for (std::size_t j = 0; j < sn.size(); ++j)
      addPatches(ref.sideControlPatch(side.second, static_cast<int>(j)), sn[j]);
  }
  else
    addPatches(ref.sidePatches(side.second), -1);
}

// ----------------------------------------------------------------------------
// Point location
// ----------------------------------------------------------------------------
PointLocator::PointLocator(const Mesh & mesh) : _mesh(mesh)
{
  auto bb = mesh.boundingBox();
  _lo = bb.first;
  _hi = bb.second;
  const int dim = mesh.dimension();
  const double ne = std::max<Index>(1, mesh.numElements());
  const int per = std::max(1, static_cast<int>(std::ceil(std::pow(ne, 1.0 / dim))));
  for (int d = 0; d < 3; ++d)
  {
    _n[d] = d < dim ? per : 1;
    const double pad = 1e-9 * std::max(1.0, _hi[d] - _lo[d]);
    _lo[d] -= pad;
    _hi[d] += pad;
  }
  _bins.resize(_n[0] * _n[1] * _n[2]);
  const auto clampBin = [&](double v, int d)
  {
    int i = static_cast<int>(std::floor((v - _lo[d]) / (_hi[d] - _lo[d]) * _n[d]));
    return std::min(std::max(i, 0), _n[d] - 1);
  };
  for (Index e = 0; e < mesh.numElements(); ++e)
  {
    const auto & el = mesh.element(e);
    Point a = mesh.node(el.nodes[0]), b = a;
    for (int k = 1; k < el.numNodes(); ++k)
      for (int d = 0; d < 3; ++d)
      {
        a[d] = std::min(a[d], mesh.node(el.nodes[k])[d]);
        b[d] = std::max(b[d], mesh.node(el.nodes[k])[d]);
      }
    int i0 = clampBin(a[0], 0), i1 = clampBin(b[0], 0);
    int j0 = clampBin(a[1], 1), j1 = clampBin(b[1], 1);
    int k0 = clampBin(a[2], 2), k1 = clampBin(b[2], 2);
    for (int k = k0; k <= k1; ++k)
      for (int j = j0; j <= j1; ++j)
        for (int i = i0; i <= i1; ++i)
          _bins[binIndex(i, j, k)].push_back(e);
  }
}

bool
PointLocator::locate(const Point & x, Index & element, Point & xi, double tol) const
{
  int idx[3];
  for (int d = 0; d < 3; ++d)
  {
    if (x[d] < _lo[d] || x[d] > _hi[d])
      return false;
    idx[d] = std::min(static_cast<int>((x[d] - _lo[d]) / (_hi[d] - _lo[d]) * _n[d]), _n[d] - 1);
  }
  const int dim = _mesh.dimension();
  MappedPoint mp;
  for (Index e : _bins[binIndex(idx[0], idx[1], idx[2])])
  {
    const auto & el = _mesh.element(e);
    const auto & ref = ReferenceElement::get(el.type);
    Point r = ref.centroid();
    bool ok = false;
    for (int it = 0; it < 30; ++it)
    {
      mapPoint(_mesh, el, r, mp);
      Point res = x - mp.x;
      // Solve J dr = res.
      Point dr{0, 0, 0};
      if (dim == 1)
        dr[0] = res[0] / mp.J[0][0];
      else if (dim == 2)
      {
        const double det = mp.detJ;
        dr[0] = (mp.J[1][1] * res[0] - mp.J[0][1] * res[1]) / det;
        dr[1] = (-mp.J[1][0] * res[0] + mp.J[0][0] * res[1]) / det;
      }
      else
      {
        const auto & J = mp.J;
        const double det = mp.detJ;
        for (int c = 0; c < 3; ++c)
        {
          double M[3][3];
          for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
              M[i][j] = (j == c) ? res[i] : J[i][j];
          dr[c] = (M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1]) -
                   M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0]) +
                   M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0])) /
                  det;
        }
      }
      r = r + dr;
      if (norm(dr) < 1e-13)
      {
        ok = true;
        break;
      }
    }
    if (!ok)
      continue;
    const bool inside = ref.contains(r, tol);
    if (inside)
    {
      element = e;
      xi = r;
      return true;
    }
  }
  return false;
}

} // namespace dualmesh
