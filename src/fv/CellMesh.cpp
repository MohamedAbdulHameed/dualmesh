// SPDX-License-Identifier: LGPL-2.1-or-later
#include "dualmesh/fv/CellMesh.h"

#include "dualmesh/core/InputParameters.h"
#include "dualmesh/fe/Assembly.h"

#include <algorithm>
#include <cmath>
#include <map>

namespace dualmesh
{

namespace
{
/// Key that identifies a face independently of the element it is seen from.
std::vector<Index>
faceKey(const Mesh & mesh, const Side & side)
{
  std::vector<Index> key = mesh.sideNodes(side);
  std::sort(key.begin(), key.end());
  return key;
}
} // namespace

CellMesh::CellMesh(const Mesh & mesh)
{
  _num_cells = mesh.numElements();
  if (_num_cells == 0)
    throw InputError("The cell-centred finite volume method needs a mesh with elements.");

  // ---- cell centroids and volumes ------------------------------------------
  QuadratureSpec spec;
  spec.points = 3;
  std::vector<IntegrationPoint> pts;
  _cell_volume.assign(_num_cells, 0.0);
  _cell_block.assign(_num_cells, 0);
  std::vector<Point> centroid(_num_cells, Point{0, 0, 0});
  MappedPoint mp;
  for (Index c = 0; c < _num_cells; ++c)
  {
    buildElementPoints(mesh, c, false, PointSet::Volume, spec, pts);
    double vol = 0;
    Point xc{0, 0, 0};
    for (const auto & ip : pts)
    {
      mapPoint(mesh, mesh.element(c), ip.xi, mp);
      vol += ip.weight;
      xc = xc + ip.weight * mp.x;
    }
    if (vol <= 0)
      throw InputError("Element " + std::to_string(c) + " has a non-positive measure.");
    _cell_volume[c] = vol;
    centroid[c] = (1.0 / vol) * xc;
    _cell_block[c] = mesh.element(c).block;
  }

  // ---- faces ---------------------------------------------------------------
  _cell_faces.assign(_num_cells, {});
  std::map<std::vector<Index>, int> seen; // face key -> index into _faces
  QuadratureSpec side_spec;
  side_spec.points = 3;
  for (Index c = 0; c < _num_cells; ++c)
  {
    const auto & ref = ReferenceElement::get(mesh.element(c).type);
    for (int s = 0; s < ref.numSides(); ++s)
    {
      const Side side{c, s};
      const auto key = faceKey(mesh, side);
      auto it = seen.find(key);
      if (it != seen.end())
      {
        // Interior face: this element is the neighbour.
        CellFace & f = _faces[it->second];
        f.neighbor = c;
        _cell_faces[c].push_back(it->second);
        _side_to_face[{c, s}] = it->second;
        continue;
      }
      CellFace f;
      f.owner = c;
      f.side = side;
      buildSidePoints(mesh, side, false, side_spec, pts);
      Point area{0, 0, 0};
      Point xf{0, 0, 0};
      double measure = 0;
      for (const auto & ip : pts)
      {
        mapPoint(mesh, mesh.element(c), ip.xi, mp);
        area = area + ip.area;
        measure += ip.weight;
        xf = xf + ip.weight * mp.x;
      }
      f.area = area;
      f.measure = measure;
      f.centroid = (1.0 / measure) * xf;
      const int index = static_cast<int>(_faces.size());
      _faces.push_back(f);
      seen[key] = index;
      _cell_faces[c].push_back(index);
      _side_to_face[{c, s}] = index;
    }
  }

  // ---- boundary entities ---------------------------------------------------
  _entity_point = centroid;
  for (int i = 0; i < static_cast<int>(_faces.size()); ++i)
    if (_faces[i].neighbor < 0)
    {
      _faces[i].boundary_entity = _num_cells + static_cast<Index>(_boundary_faces.size());
      _boundary_faces.push_back(i);
      _entity_point.push_back(_faces[i].centroid);
    }

  // ---- the second cell used by the second-order boundary gradient ----------
  for (int bi : _boundary_faces)
  {
    CellFace & f = _faces[bi];
    const Point n = (1.0 / f.measure) * f.area;
    double best = -0.5; // require a face at least a little opposed to n
    Index pick = -1;
    for (int fi : _cell_faces[f.owner])
    {
      if (fi == bi)
        continue;
      const CellFace & g = _faces[fi];
      if (g.neighbor < 0)
        continue;
      const Index other = g.owner == f.owner ? g.neighbor : g.owner;
      Point m = (1.0 / g.measure) * g.area;
      if (g.owner != f.owner)
        m = -1.0 * m;
      const double align = -dot(m, n);
      if (align > best)
      {
        best = align;
        pick = other;
      }
    }
    f.second_cell = pick;
  }

  // ---- least-squares gradient stencils ------------------------------------
  const int dim = mesh.dimension();
  _lsq.assign(_num_cells, {});
  for (Index c = 0; c < _num_cells; ++c)
  {
    std::vector<std::pair<Index, Point>> nbrs; // entity, d = x_i - x_c
    for (int fi : _cell_faces[c])
    {
      const CellFace & f = _faces[fi];
      const Index other =
          f.neighbor < 0 ? f.boundary_entity : (f.owner == c ? f.neighbor : f.owner);
      nbrs.emplace_back(other, _entity_point[other] - _entity_point[c]);
    }
    // Normal equations A g = sum_i w_i (U_i - U_c) d_i, A = sum_i w_i d_i d_i^T.
    double A[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
    std::vector<double> weight(nbrs.size());
    for (std::size_t i = 0; i < nbrs.size(); ++i)
    {
      const Point & d = nbrs[i].second;
      const double dd = dot(d, d);
      weight[i] = dd > 0 ? 1.0 / dd : 0.0;
      for (int a = 0; a < dim; ++a)
        for (int b = 0; b < dim; ++b)
          A[a][b] += weight[i] * d[a] * d[b];
    }
    // Invert the dim x dim matrix.
    double inv[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
    if (dim == 1)
      inv[0][0] = A[0][0] != 0 ? 1.0 / A[0][0] : 0.0;
    else if (dim == 2)
    {
      const double det = A[0][0] * A[1][1] - A[0][1] * A[1][0];
      const double id = det != 0 ? 1.0 / det : 0.0;
      inv[0][0] = A[1][1] * id;
      inv[0][1] = -A[0][1] * id;
      inv[1][0] = -A[1][0] * id;
      inv[1][1] = A[0][0] * id;
    }
    else
    {
      const double det = A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1]) -
                         A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0]) +
                         A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0]);
      const double id = det != 0 ? 1.0 / det : 0.0;
      inv[0][0] = (A[1][1] * A[2][2] - A[1][2] * A[2][1]) * id;
      inv[0][1] = (A[0][2] * A[2][1] - A[0][1] * A[2][2]) * id;
      inv[0][2] = (A[0][1] * A[1][2] - A[0][2] * A[1][1]) * id;
      inv[1][0] = (A[1][2] * A[2][0] - A[1][0] * A[2][2]) * id;
      inv[1][1] = (A[0][0] * A[2][2] - A[0][2] * A[2][0]) * id;
      inv[1][2] = (A[0][2] * A[1][0] - A[0][0] * A[1][2]) * id;
      inv[2][0] = (A[1][0] * A[2][1] - A[1][1] * A[2][0]) * id;
      inv[2][1] = (A[0][1] * A[2][0] - A[0][0] * A[2][1]) * id;
      inv[2][2] = (A[0][0] * A[1][1] - A[0][1] * A[1][0]) * id;
    }
    for (std::size_t i = 0; i < nbrs.size(); ++i)
    {
      Point coefficient{0, 0, 0};
      for (int a = 0; a < dim; ++a)
        for (int b = 0; b < dim; ++b)
          coefficient[a] += inv[a][b] * weight[i] * nbrs[i].second[b];
      _lsq[c].emplace_back(nbrs[i].first, coefficient);
    }
  }
}

std::vector<int>
CellMesh::boundaryFaceIndices(const Mesh & mesh, const std::string & name) const
{
  std::vector<int> out;
  for (const Side & s : mesh.sideset(name))
  {
    auto it = _side_to_face.find({s.first, s.second});
    if (it == _side_to_face.end())
      throw InputError("Side set '" + name + "' refers to a side that is not in the mesh.");
    if (_faces[it->second].neighbor >= 0)
      throw InputError("Side set '" + name +
                       "' contains an interior face, which the "
                       "cell-centred finite volume method cannot use.");
    out.push_back(it->second);
  }
  return out;
}

std::vector<Index>
CellMesh::boundaryEntities(const Mesh & mesh, const std::string & name) const
{
  std::vector<Index> out;
  for (int fi : boundaryFaceIndices(mesh, name))
    out.push_back(_faces[fi].boundary_entity);
  std::sort(out.begin(), out.end());
  out.erase(std::unique(out.begin(), out.end()), out.end());
  return out;
}

} // namespace dualmesh
