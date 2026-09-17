// SPDX-License-Identifier: LGPL-2.1-or-later
#include "dualmesh/fe/Assembly.h"
#include "dualmesh/core/InputParameters.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace dualmesh
{

void
mapPoint(const Mesh & mesh, const Element & el, const Point & xi, MappedPoint & mp)
{
  const auto & ref = ReferenceElement::get(el.type);
  const int dim = ref.dimension();
  const int nn = ref.numNodes();
  Point dNr[8];
  ref.shape(xi, mp.N, dNr);
  mp.xi = xi;
  mp.x = {0, 0, 0};
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      mp.J[i][j] = 0.0;
  for (int k = 0; k < nn; ++k)
  {
    const Point & X = mesh.node(el.nodes[k]);
    mp.x = mp.x + mp.N[k] * X;
    for (int i = 0; i < dim; ++i)
      for (int j = 0; j < dim; ++j)
        mp.J[i][j] += X[i] * dNr[k][j];
  }
  double inv[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
  const auto & J = mp.J;
  if (dim == 1)
  {
    mp.detJ = J[0][0];
    inv[0][0] = 1.0 / J[0][0];
  }
  else if (dim == 2)
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
    for (int i = 0; i < dim; ++i)
      for (int j = 0; j < dim; ++j)
        g[i] += inv[j][i] * dNr[k][j];
    mp.dN[k] = g;
  }
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

double
patchMeasure(const Mesh & mesh, const Element & el, const Patch & patch)
{
  QuadratureSpec g;
  g.kind = QuadratureSpec::Kind::Gauss;
  g.points = 3;
  std::vector<Point> s;
  std::vector<double> w;
  ruleTensor(g, patch.dim, s, w);
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

void
buildElementPoints(const Mesh & mesh,
                   Index e,
                   bool dual_mesh,
                   PointSet set,
                   const QuadratureSpec & spec,
                   std::vector<IntegrationPoint> & out)
{
  out.clear();
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
        ip.xi = ref.controlVolume(a).map({1, 1, 1});
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
        ip.xi = at_node ? ref.node(a) : ref.controlVolume(a).map({1, 1, 1});
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
            ruleTensor(g, patch.dim, s, w);
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
    const auto addPatch = [&](const Patch & patch, int owner)
    {
      ruleTensor(spec, patch.dim, s, w);
      for (std::size_t q = 0; q < s.size(); ++q)
      {
        std::array<Point, 3> T;
        IntegrationPoint ip;
        ip.xi = patch.map(s[q], &T);
        ip.xi_field = spec.reduced ? centroid : ip.xi;
        mapPoint(mesh, el, ip.xi, mp);
        ip.weight = w[q] * std::abs(mp.detJ * patchDet(T, dim));
        ip.area = {0, 0, 0};
        ip.owner = owner;
        ip.neighbor = -1;
        out.push_back(ip);
      }
    };
    if (dual_mesh)
      for (int a = 0; a < ref.numNodes(); ++a)
        addPatch(ref.controlVolume(a), a);
    else
      for (const auto & patch : ref.elementPatches())
        addPatch(patch, -1);
    return;
  }

  // Interfaces between control volumes.
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
    const Point xab = mesh.node(el.nodes[face.b]) - mesh.node(el.nodes[face.a]);
    for (std::size_t q = 0; q < s.size(); ++q)
    {
      std::array<Point, 3> T;
      IntegrationPoint ip;
      ip.xi = face.patch.map(s[q], &T);
      ip.xi_field = fs.reduced ? centroid : ip.xi;
      mapPoint(mesh, el, ip.xi, mp);
      Point n = areaVector(mp, T, dim);
      if (dot(n, xab) < 0)
        n = -1.0 * n;
      ip.area = w[q] * n;
      ip.weight = norm(ip.area);
      ip.owner = face.a;
      ip.neighbor = face.b;
      out.push_back(ip);
    }
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

  const auto addPatch = [&](const Patch & patch, int owner)
  {
    ruleTensor(qs, patch.dim, s, w);
    std::vector<IntegrationPoint> pts;
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
      addPatch(ref.sideControlPatch(side.second, static_cast<int>(j)), sn[j]);
  }
  else
  {
    for (const auto & patch : ref.sidePatches(side.second))
      addPatch(patch, -1);
  }
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
    bool inside = true;
    if (ref.isSimplex())
    {
      double sum = 0;
      for (int d = 0; d < dim; ++d)
      {
        inside = inside && r[d] >= -tol;
        sum += r[d];
      }
      inside = inside && sum <= 1 + tol;
    }
    else
      for (int d = 0; d < dim; ++d)
        inside = inside && std::abs(r[d]) <= 1 + tol;
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
