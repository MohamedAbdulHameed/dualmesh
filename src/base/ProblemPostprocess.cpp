// SPDX-License-Identifier: LGPL-2.1-or-later
//
// Post-processing: reactions (secondary variables), sampling, centroid
// quantities, integrals, and VTK output.
#include "dualmesh/base/Problem.h"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>

namespace dualmesh
{

std::vector<std::pair<Index, double>>
Problem::reactions(const std::string & var, const std::string & boundary) const
{
  const int v = variableIndex(var);
  if (_last_residual.size() != numDofs())
    throw InputError("Reactions are available after a solve.");
  std::vector<std::pair<Index, double>> out;
  for (Index n : _mesh->boundaryNodes(boundary))
    out.emplace_back(n, _last_residual[dof(n, v)]);
  return out;
}

double
Problem::totalReaction(const std::string & var, const std::string & boundary) const
{
  double s = 0;
  for (const auto & [n, r] : reactions(var, boundary))
    s += r;
  return s;
}

std::vector<double>
Problem::sample(const std::string & var, const std::vector<Point> & points) const
{
  const int v = variableIndex(var);
  PointLocator loc(*_mesh);
  std::vector<double> out;
  MappedPoint mp;
  for (const auto & p : points)
  {
    Index e;
    Point xi;
    if (!loc.locate(p, e, xi, 1e-8))
    {
      out.push_back(std::numeric_limits<double>::quiet_NaN());
      continue;
    }
    const auto & el = _mesh->element(e);
    mapPoint(*_mesh, el, xi, mp);
    double s = 0;
    for (int k = 0; k < el.numNodes(); ++k)
      s += mp.N[k] * _U[dof(el.nodes[k], v)];
    out.push_back(s);
  }
  return out;
}

std::vector<Point>
Problem::gradientAtCentroids(const std::string & var) const
{
  const int v = variableIndex(var);
  std::vector<Point> out;
  MappedPoint mp;
  for (Index e = 0; e < _mesh->numElements(); ++e)
  {
    const auto & el = _mesh->element(e);
    mapPoint(*_mesh, el, ReferenceElement::get(el.type).centroid(), mp);
    Point g{0, 0, 0};
    for (int k = 0; k < el.numNodes(); ++k)
      g = g + _U[dof(el.nodes[k], v)] * mp.dN[k];
    out.push_back(g);
  }
  return out;
}

namespace
{
std::vector<double>
localValues(const Problem & pb, const Element & el)
{
  const int nv = pb.numVariables();
  std::vector<double> loc(el.numNodes() * nv);
  for (int k = 0; k < el.numNodes(); ++k)
    for (int v = 0; v < nv; ++v)
      loc[k * nv + v] = pb.solution()[pb.dof(el.nodes[k], v)];
  return loc;
}
} // namespace

std::vector<std::vector<double>>
Problem::propertyAtCentroids(const std::string & property) const
{
  auto * self = const_cast<Problem *>(this);
  self->initialize();
  const int id = _props.id(property);
  const int nc = _props.components(property);
  std::vector<std::vector<double>> out;
  MappedPoint mp;
  QpContext ctx;
  ctx.dim = _mesh->dimension();
  ctx.time = _time;
  for (Index e = 0; e < _mesh->numElements(); ++e)
  {
    const auto & el = _mesh->element(e);
    mapPoint(*_mesh, el, ReferenceElement::get(el.type).centroid(), mp);
    const auto loc = localValues(*this, el);
    fillContext(ctx, el, mp, loc, nullptr, nullptr, 0);
    ctx.x = mp.x;
    ctx.element = e;
    ctx.block = el.block;
    computeMaterials(ctx);
    std::vector<double> row(nc);
    for (int c = 0; c < nc; ++c)
      row[c] = ctx.property(id, c).value();
    out.push_back(row);
  }
  return out;
}

std::vector<Point>
Problem::kernelFluxAtCentroids(const std::string & kernel) const
{
  auto * self = const_cast<Problem *>(this);
  self->initialize();
  auto k = std::dynamic_pointer_cast<Kernel>(object(kernel));
  if (!k)
    throw InputError("'" + kernel + "' is not a kernel.");
  std::vector<Point> out;
  MappedPoint mp;
  QpContext ctx;
  ctx.dim = _mesh->dimension();
  ctx.time = _time;
  for (Index e = 0; e < _mesh->numElements(); ++e)
  {
    const auto & el = _mesh->element(e);
    mapPoint(*_mesh, el, ReferenceElement::get(el.type).centroid(), mp);
    const auto loc = localValues(*this, el);
    fillContext(ctx, el, mp, loc, nullptr, nullptr, 0);
    ctx.x = mp.x;
    ctx.element = e;
    ctx.block = el.block;
    computeMaterials(ctx);
    ADVector3 F{ADReal(0.0), ADReal(0.0), ADReal(0.0)};
    if (k->hasFlux() && k->activeOnBlock(el.block))
      k->computeFlux(ctx, F);
    out.push_back({F[0].value(), F[1].value(), F[2].value()});
  }
  return out;
}

double
Problem::integrate(const std::string & var) const
{
  const int v = variableIndex(var);
  QuadratureSpec g;
  g.points = 3;
  double total = 0;
  MappedPoint mp;
  std::vector<IntegrationPoint> pts;
  for (Index e = 0; e < _mesh->numElements(); ++e)
  {
    const auto & el = _mesh->element(e);
    buildElementPoints(*_mesh, e, false, PointSet::Volume, g, pts);
    for (const auto & ip : pts)
    {
      mapPoint(*_mesh, el, ip.xi, mp);
      double s = 0;
      for (int k = 0; k < el.numNodes(); ++k)
        s += mp.N[k] * _U[dof(el.nodes[k], v)];
      total += s * ip.weight * coordFactor(mp.x);
    }
  }
  return total;
}

double
Problem::boundaryFluxIntegral(const std::string & kernel, const std::string & boundary) const
{
  auto * self = const_cast<Problem *>(this);
  self->initialize();
  auto k = std::dynamic_pointer_cast<Kernel>(object(kernel));
  if (!k)
    throw InputError("'" + kernel + "' is not a kernel.");
  QuadratureSpec g;
  g.points = 3;
  double total = 0;
  MappedPoint mp;
  QpContext ctx;
  ctx.dim = _mesh->dimension();
  ctx.time = _time;
  std::vector<IntegrationPoint> pts;
  for (const auto & side : _mesh->sideset(boundary))
  {
    const auto & el = _mesh->element(side.first);
    if (!k->activeOnBlock(el.block))
      continue;
    buildSidePoints(*_mesh, side, false, g, pts);
    const auto loc = localValues(*this, el);
    for (const auto & ip : pts)
    {
      mapPoint(*_mesh, el, ip.xi, mp);
      fillContext(ctx, el, mp, loc, nullptr, nullptr, 0);
      ctx.x = mp.x;
      ctx.element = side.first;
      ctx.block = el.block;
      ctx.normal = (1.0 / ip.weight) * ip.area;
      ctx.on_boundary = true;
      computeMaterials(ctx);
      ADVector3 F{ADReal(0.0), ADReal(0.0), ADReal(0.0)};
      k->computeFlux(ctx, F);
      total += (F[0].value() * ip.area[0] + F[1].value() * ip.area[1] + F[2].value() * ip.area[2]) *
               coordFactor(mp.x);
    }
  }
  return total;
}

void
Problem::writeVTU(const std::string & filename,
                  const std::vector<std::string> & cell_properties) const
{
  std::ofstream f(filename);
  if (!f)
    throw InputError("Cannot open '" + filename + "' for writing.");
  const auto & m = *_mesh;
  f << std::setprecision(12);
  f << "<?xml version=\"1.0\"?>\n<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" "
       "byte_order=\"LittleEndian\">\n<UnstructuredGrid>\n";
  f << "<Piece NumberOfPoints=\"" << m.numNodes() << "\" NumberOfCells=\"" << m.numElements()
    << "\">\n";
  f << "<PointData>\n";
  for (const auto & v : _vars)
  {
    f << "<DataArray type=\"Float64\" Name=\"" << v.name << "\" format=\"ascii\">\n";
    for (Index n = 0; n < m.numNodes(); ++n)
      f << _U[dof(n, v.index)] << "\n";
    f << "</DataArray>\n";
  }
  f << "</PointData>\n<CellData>\n";
  f << "<DataArray type=\"Int32\" Name=\"block\" format=\"ascii\">\n";
  for (const auto & el : m.elements())
    f << el.block << "\n";
  f << "</DataArray>\n";
  for (const auto & prop : cell_properties)
  {
    const auto vals = propertyAtCentroids(prop);
    const int nc = vals.empty() ? 1 : static_cast<int>(vals[0].size());
    f << "<DataArray type=\"Float64\" Name=\"" << prop << "\" NumberOfComponents=\"" << nc
      << "\" format=\"ascii\">\n";
    for (const auto & row : vals)
    {
      for (double x : row)
        f << x << " ";
      f << "\n";
    }
    f << "</DataArray>\n";
  }
  f << "</CellData>\n<Points>\n<DataArray type=\"Float64\" NumberOfComponents=\"3\" "
       "format=\"ascii\">\n";
  for (const auto & p : m.nodes())
    f << p[0] << " " << p[1] << " " << p[2] << "\n";
  f << "</DataArray>\n</Points>\n<Cells>\n";
  f << "<DataArray type=\"Int64\" Name=\"connectivity\" format=\"ascii\">\n";
  for (const auto & el : m.elements())
  {
    for (int k = 0; k < el.numNodes(); ++k)
      f << el.nodes[k] << " ";
    f << "\n";
  }
  f << "</DataArray>\n<DataArray type=\"Int64\" Name=\"offsets\" format=\"ascii\">\n";
  Index off = 0;
  for (const auto & el : m.elements())
  {
    off += el.numNodes();
    f << off << "\n";
  }
  f << "</DataArray>\n<DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n";
  for (const auto & el : m.elements())
  {
    int t = 0;
    switch (el.type)
    {
    case ElementType::Edge2:
      t = 3;
      break;
    case ElementType::Tri3:
      t = 5;
      break;
    case ElementType::Quad4:
      t = 9;
      break;
    case ElementType::Tet4:
      t = 10;
      break;
    case ElementType::Hex8:
      t = 12;
      break;
    }
    f << t << "\n";
  }
  f << "</DataArray>\n</Cells>\n</Piece>\n</UnstructuredGrid>\n</VTKFile>\n";
}

std::string
Problem::summary() const
{
  std::ostringstream os;
  os << "Problem: method " << (_method == Method::DualMesh ? "dual mesh (DMCDM)" : "finite element")
     << ", coordinates "
     << (_coord == CoordinateSystem::Cartesian
             ? "Cartesian"
             : (_coord == CoordinateSystem::Axisymmetric ? "axisymmetric" : "spherical"))
     << "\n";
  os << _mesh->summary();
  os << "  variables:";
  for (const auto & v : _vars)
    os << " " << v.name;
  os << "\n  objects:\n";
  for (const auto & [n, o] : _by_name)
    os << "    " << n << " (" << o->type() << ")\n";
  os << "  degrees of freedom: " << numDofs() << "\n";
  return os.str();
}

} // namespace dualmesh
