// SPDX-License-Identifier: LGPL-2.1-or-later
//
// Post-processing: reactions (secondary variables), sampling, centroid
// quantities, integrals, and VTK output.
#include "dualmesh/base/Problem.h"

#ifdef _OPENMP
#include <omp.h>
#endif

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
  for (Index n : boundaryEntities(boundary))
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
  // The cell-centred method stores one value per cell; a point value is
  // recovered by the linear reconstruction u(x) = U_c + grad u|_c . (x - x_c),
  // which is second-order accurate, as in every finite volume code.
  std::vector<std::vector<Point>> cg;
  if (_cells)
    cg = cellGradients(_U);
  for (const auto & p : points)
  {
    Index e;
    Point xi;
    if (!loc.locate(p, e, xi, 1e-8))
    {
      out.push_back(std::numeric_limits<double>::quiet_NaN());
      continue;
    }
    if (_cells)
    {
      out.push_back(_U[dof(e, v)] + dot(cg[v][e], p - _cells->entityPoint(e)));
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
  if (_cells)
    return cellGradients(_U)[v];
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

std::vector<double>
Problem::errorIndicator(const std::string & var) const
{
  const int v = variableIndex(var);
  if (_cells)
    throw InputError("The error indicator is not available for the cell-centred finite volume "
                     "method, whose unknowns are cell averages rather than a nodal field; "
                     "recovering a nodal gradient from them needs a different construction.");
  const Index nn = _mesh->numNodes();
  const int dim = _mesh->dimension();
  std::vector<Point> recovered(nn, Point{0, 0, 0});
  std::vector<double> weight(nn, 0.0);

  QuadratureSpec rule;
  rule.points = 3;
  std::vector<IntegrationPoint> points;
  MappedPoint mp;

  // Pass one: the weighted average of the element gradients at every node.
  // The weight of an element at a node is the measure of the part of the
  // element that belongs to that node's control domain, which is the natural
  // weight here because it is exactly the volume the node is responsible for.
  for (Index e = 0; e < _mesh->numElements(); ++e)
  {
    const auto & el = _mesh->element(e);
    buildElementPoints(*_mesh, e, true, PointSet::Volume, rule, points);
    std::vector<double> share(el.numNodes(), 0.0);
    std::vector<Point> gradient(el.numNodes(), Point{0, 0, 0});
    for (const auto & p : points)
    {
      mapPoint(*_mesh, el, p.xi, mp);
      Point g{0, 0, 0};
      for (int k = 0; k < el.numNodes(); ++k)
        g = g + _U[dof(el.nodes[k], v)] * mp.dN[k];
      share[p.owner] += p.weight;
      gradient[p.owner] = gradient[p.owner] + p.weight * g;
    }
    for (int k = 0; k < el.numNodes(); ++k)
    {
      if (share[k] <= 0)
        continue;
      recovered[el.nodes[k]] = recovered[el.nodes[k]] + gradient[k];
      weight[el.nodes[k]] += share[k];
    }
  }
  for (Index n = 0; n < nn; ++n)
    if (weight[n] > 0)
      recovered[n] = (1.0 / weight[n]) * recovered[n];

  // Pass two: the norm of the difference, element by element.
  std::vector<double> out(_mesh->numElements(), 0.0);
  for (Index e = 0; e < _mesh->numElements(); ++e)
  {
    const auto & el = _mesh->element(e);
    buildElementPoints(*_mesh, e, false, PointSet::Volume, rule, points);
    double sum = 0;
    for (const auto & p : points)
    {
      mapPoint(*_mesh, el, p.xi, mp);
      Point computed{0, 0, 0}, smooth{0, 0, 0};
      for (int k = 0; k < el.numNodes(); ++k)
      {
        computed = computed + _U[dof(el.nodes[k], v)] * mp.dN[k];
        smooth = smooth + mp.N[k] * recovered[el.nodes[k]];
      }
      double square = 0;
      for (int i = 0; i < dim; ++i)
        square += (smooth[i] - computed[i]) * (smooth[i] - computed[i]);
      sum += p.weight * square * coordFactor(mp.x);
    }
    out[e] = std::sqrt(std::max(sum, 0.0));
  }
  return out;
}

Problem::ErrorNorms
Problem::errorNorms(const std::string & var,
                    const Function & exact,
                    const std::array<const Function *, 3> & exact_gradient,
                    int quadrature_points) const
{
  const int v = variableIndex(var);
  const bool want_h1 = exact_gradient[0] && exact_gradient[1] && exact_gradient[2];
  int order = 1;
  for (const auto & el : _mesh->elements())
    if (elementIsQuadratic(el.type))
      order = 2;
  // The squared error of a degree-p interpolant against a smooth function is,
  // to leading order, a polynomial of degree 2p + 2 on each element.  A Gauss
  // rule with n points per direction integrates degree 2n - 1 exactly, so
  // n = p + 2 already over-integrates the leading term; the collapsed rules
  // of simplices, prisms and pyramids carry their own extra point for the
  // Jacobian of the collapse.  More points change the norms only in digits
  // far below the discretisation error.
  QuadratureSpec rule;
  rule.points = quadrature_points > 0 ? quadrature_points : order + 2;
  std::vector<std::vector<Point>> cell_gradients;
  if (_cells)
    cell_gradients = cellGradients(_U);

  // The sum is split over threads when every function may be called
  // concurrently; a Python callable keeps it on one thread.
  bool concurrent = exact.threadSafe();
  for (const auto * g : exact_gradient)
    concurrent = concurrent && (!g || g->threadSafe());
  int nthreads = 1;
#ifdef _OPENMP
  if (concurrent)
    nthreads = std::max(1, _num_threads > 0 ? _num_threads : omp_get_max_threads());
#endif
  const Index ne = _mesh->numElements();
  if (ne < 64)
    nthreads = 1;

  double l2 = 0, h1 = 0;
  std::string failure;
#pragma omp parallel num_threads(nthreads) if (nthreads > 1) reduction(+ : l2, h1)
  {
    std::vector<IntegrationPoint> points;
    MappedPoint mp;
#pragma omp for schedule(static)
    for (Index e = 0; e < ne; ++e)
    {
      try
      {
        const auto & el = _mesh->element(e);
        buildElementPoints(*_mesh, e, false, PointSet::Volume, rule, points);
        for (const auto & p : points)
        {
          mapPoint(*_mesh, el, p.xi, mp);
          double uh = 0;
          Point gh{0, 0, 0};
          if (_cells)
          {
            gh = cell_gradients[v][e];
            uh = _U[dof(e, v)] + dot(gh, mp.x - _cells->entityPoint(e));
          }
          else
            for (int k = 0; k < el.numNodes(); ++k)
            {
              const double U = _U[dof(el.nodes[k], v)];
              uh += mp.N[k] * U;
              gh = gh + U * mp.dN[k];
            }
          const double w = p.weight * coordFactor(mp.x);
          const double diff = uh - exact.value(mp.x, _time);
          l2 += w * diff * diff;
          if (want_h1)
            for (int d = 0; d < 3; ++d)
            {
              const double gd = gh[d] - exact_gradient[d]->value(mp.x, _time);
              h1 += w * gd * gd;
            }
        }
      }
      catch (const std::exception & ex)
      {
#pragma omp critical(dualmesh_error_norms)
        if (failure.empty())
          failure = ex.what();
      }
    }
  }
  if (!failure.empty())
    throw InputError(failure);
  ErrorNorms out;
  out.l2 = std::sqrt(l2);
  out.h1_seminorm = want_h1 ? std::sqrt(h1) : std::numeric_limits<double>::quiet_NaN();
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
  std::vector<std::vector<Point>> cg;
  if (_cells)
    cg = cellGradients(_U);
  for (Index e = 0; e < _mesh->numElements(); ++e)
  {
    const auto & el = _mesh->element(e);
    if (_cells)
      fillContextCell(ctx, e, cg);
    else
    {
      mapPoint(*_mesh, el, ReferenceElement::get(el.type).centroid(), mp);
      const auto loc = localValues(*this, el);
      fillContext(ctx, el, mp, loc, nullptr, nullptr, 0);
      ctx.x = mp.x;
      ctx.element = e;
      ctx.block = el.block;
    }
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
  std::vector<std::vector<Point>> cg;
  if (_cells)
    cg = cellGradients(_U);
  for (Index e = 0; e < _mesh->numElements(); ++e)
  {
    const auto & el = _mesh->element(e);
    if (_cells)
      fillContextCell(ctx, e, cg);
    else
    {
      mapPoint(*_mesh, el, ReferenceElement::get(el.type).centroid(), mp);
      const auto loc = localValues(*this, el);
      fillContext(ctx, el, mp, loc, nullptr, nullptr, 0);
      ctx.x = mp.x;
      ctx.element = e;
      ctx.block = el.block;
    }
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
  if (_cells)
  {
    for (Index c = 0; c < _cells->numCells(); ++c)
      total += _U[dof(c, v)] * _cells->cellVolume(c) * coordFactor(_cells->entityPoint(c));
    return total;
  }
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
  if (_cells)
  {
    const auto cg = cellGradients(_U);
    for (int fi : _cells->boundaryFaceIndices(*_mesh, boundary))
    {
      const CellFace & f = _cells->faces()[fi];
      if (!k->activeOnBlock(_cells->cellBlock(f.owner)))
        continue;
      fillContextCell(ctx, f.owner, cg);
      ctx.x = f.centroid;
      ctx.normal = (1.0 / f.measure) * f.area;
      ctx.on_boundary = true;
      computeMaterials(ctx);
      ADVector3 F{ADReal(0.0), ADReal(0.0), ADReal(0.0)};
      k->computeFlux(ctx, F);
      total += (F[0].value() * f.area[0] + F[1].value() * f.area[1] + F[2].value() * f.area[2]) *
               coordFactor(f.centroid);
    }
    return total;
  }
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
  if (!_cells)
    for (const auto & v : _vars)
    {
      f << "<DataArray type=\"Float64\" Name=\"" << v.name << "\" format=\"ascii\">\n";
      for (Index n = 0; n < m.numNodes(); ++n)
        f << _U[dof(n, v.index)] << "\n";
      f << "</DataArray>\n";
    }
  f << "</PointData>\n<CellData>\n";
  // The cell-centred method stores its unknowns on the cells.
  if (_cells)
    for (const auto & v : _vars)
    {
      f << "<DataArray type=\"Float64\" Name=\"" << v.name << "\" format=\"ascii\">\n";
      for (Index c = 0; c < m.numElements(); ++c)
        f << _U[dof(c, v.index)] << "\n";
      f << "</DataArray>\n";
    }
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
    for (int k : vtkNodeOrder(el.type))
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
    f << vtkCellType(el.type) << "\n";
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
