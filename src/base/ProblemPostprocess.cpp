// SPDX-License-Identifier: LGPL-2.1-or-later
//
// Post-processing: reactions (secondary variables), sampling, centroid
// quantities, integrals, and VTK output.
#include "dualmesh/base/Problem.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <sstream>

namespace dualmesh
{

namespace
{
std::vector<double>
localValues(const Problem & pb, const Element & el)
{
  const int nv = pb.numVariables();
  std::vector<double> loc(pb.numLocalDofs(el));
  for (int k = 0; k < el.numNodes(); ++k)
    for (int v = 0; v < nv; ++v)
    {
      const int i = pb.localDofIndex(el, k, v);
      if (i >= 0)
        loc[i] = pb.solution()[pb.dof(el.nodes[k], v)];
    }
  return loc;
}
} // namespace

std::vector<std::pair<Index, double>>
Problem::boundaryOutflow(const std::string & var,
                         const std::string & boundary,
                         const std::vector<char> * owned) const
{
  const int v = variableIndex(var);
  if (_last_residual.size() != numDofs())
    throw InputError("The flow through a boundary is available after a solve.");
  // Where the boundary prescribes the value the flow is the reaction: the residual of the
  // equation there, which is what the condition supplies, less what the integrated conditions
  // add there.  Everywhere it has what the integrated conditions of the boundary add.  Both are
  // taken out of the body.  A node that the condition of another boundary holds (a corner) has
  // the reaction of that boundary only.
  const auto on = [&](const std::vector<std::string> & names)
  { return std::find(names.begin(), names.end(), boundary) != names.end(); };
  bool held = false;
  for (const auto & bc : _nbcs)
    held = held || (bc->variable() == v && on(bc->boundaries()));
  std::map<Index, double> out;
  const std::vector<char> fixed = constrainedDofs();
  // The equations of periodic copies and hanging nodes belong to their primaries, as in the solve,
  // so a held primary carries their share of the reaction.
  Vector residual = _last_residual;
  foldDependents(residual, nullptr);
  for (Index n : _mesh->boundaryNodes(boundary))
  {
    const Index d = dof(n, v);
    out[n] = held && fixed[d] && !isDependent(d) && (!owned || (*owned)[n]) ? -residual[d] : 0.0;
  }
  // The flux of each integrated condition of the boundary, as the assembly adds it to the
  // residual.
  const int nv = numVariables();
  const AssemblyOptions steady;
  ThreadScratch scratch;
  scratch.ctx.dim = _mesh->spaceDimension();
  scratch.ctx.time = _time;
  // A node set (a support at a point) has no sides to integrate over.
  static const std::vector<Side> no_sides;
  const bool sides =
      _mesh->sidesets().count(boundary) > 0 || _mesh->nodesets().count(boundary) == 0;
  for (std::size_t index = 0; index < _ibcs.size(); ++index)
  {
    const IntegratedBC & bc = *_ibcs[index];
    if (bc.isConstraint() || bc.variable() != v || !on(bc.boundaries()))
      continue;
    if (dynamic_cast<const InterfaceBC *>(&bc))
      throw InputError("The flow through '" + boundary +
                       "' is not available: an interface condition acts on it, and the flow "
                       "across an interface is not computed.");
    for (const Side & side : sides ? _mesh->sideset(boundary) : no_sides)
    {
      const Element & el = _mesh->element(side.first);
      if (!bc.activeOnBlock(el.block))
        continue;
      gatherLocal(scratch, el, _U, steady);
      conditionResidual(bc, static_cast<unsigned>(index), side, 1.0, steady, 0, scratch);
      for (std::size_t i = 0; i < scratch.dofs.size(); ++i)
        if (scratch.touched[i])
          out[scratch.dofs[i] / nv] += scratch.Rloc[i].value();
    }
  }
  // The forces of a constraint (contact) on the boundary: what it adds to the residual of the
  // equations there.
  for (const auto & bc : _ibcs)
  {
    if (!bc->isConstraint() || !on(bc->constraintBoundaries()))
      continue;
    Vector forces = Vector::Zero(numDofs());
    bc->assembleConstraint(*this, _U, nullptr, forces, nullptr, nullptr);
    for (auto & [n, q] : out)
      q += forces[dof(n, v)];
  }
  return {out.begin(), out.end()};
}

double
Problem::totalOutflow(const std::string & var, const std::string & boundary) const
{
  double s = 0;
  for (const auto & [n, q] : boundaryOutflow(var, boundary))
    s += q;
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
Problem::sampleGradient(const std::string & var, const std::vector<Point> & points) const
{
  const int v = variableIndex(var);
  PointLocator loc(*_mesh);
  const double nan = std::numeric_limits<double>::quiet_NaN();
  std::vector<Point> out;
  MappedPoint mp;
  for (const auto & p : points)
  {
    Index e;
    Point xi;
    if (!loc.locate(p, e, xi, 1e-8))
    {
      out.push_back(Point{nan, nan, nan});
      continue;
    }
    const auto & el = _mesh->element(e);
    mapPoint(*_mesh, el, xi, mp);
    Point g{0, 0, 0};
    for (int k = 0; k < el.numNodes(); ++k)
      g = g + _U[dof(el.nodes[k], v)] * mp.dN[k];
    out.push_back(g);
  }
  return out;
}

std::vector<Point>
Problem::gradientAtCentroids(const std::string & var) const
{
  const int v = variableIndex(var);
  std::vector<Point> out;
  MappedPoint mp;
  for (Index e = 0; e < numIntegratedElements(); ++e)
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
  const Index nn = _mesh->numNodes();
  const int dim = _mesh->spaceDimension();
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
  for (Index e = 0; e < numIntegratedElements(); ++e)
  {
    const auto & el = _mesh->element(e);
    // An element without control domains shares each point equally among its nodes, which keeps
    // the weights positive (the corner shape functions of Quad8 integrate to negative values).
    const bool dual = ReferenceElement::get(el.type).hasControlDomains();
    buildElementPoints(*_mesh, e, dual, PointSet::Volume, rule, points);
    std::vector<double> share(el.numNodes(), 0.0);
    std::vector<Point> gradient(el.numNodes(), Point{0, 0, 0});
    for (const auto & p : points)
    {
      mapPoint(*_mesh, el, p.xi, mp);
      Point g{0, 0, 0};
      for (int k = 0; k < el.numNodes(); ++k)
        g = g + _U[dof(el.nodes[k], v)] * mp.dN[k];
      if (dual)
      {
        share[p.owner] += p.weight;
        gradient[p.owner] = gradient[p.owner] + p.weight * g;
        continue;
      }
      const double w = p.weight / el.numNodes();
      for (int k = 0; k < el.numNodes(); ++k)
      {
        share[k] += w;
        gradient[k] = gradient[k] + w * g;
      }
    }
    for (int k = 0; k < el.numNodes(); ++k)
    {
      if (share[k] <= 0)
        continue;
      recovered[el.nodes[k]] = recovered[el.nodes[k]] + gradient[k];
      weight[el.nodes[k]] += share[k];
    }
  }
  // A node shared by several processes takes the elements of all of them.
  std::vector<double> sums(static_cast<std::size_t>(4 * nn));
  for (Index n = 0; n < nn; ++n)
  {
    for (int i = 0; i < 3; ++i)
      sums[static_cast<std::size_t>(4 * n + i)] = recovered[n][i];
    sums[static_cast<std::size_t>(4 * n + 3)] = weight[n];
  }
  parallel().addAtNodes(sums.data(), 4);
  for (Index n = 0; n < nn; ++n)
  {
    for (int i = 0; i < 3; ++i)
      recovered[n][i] = sums[static_cast<std::size_t>(4 * n + i)];
    weight[n] = sums[static_cast<std::size_t>(4 * n + 3)];
  }
  for (Index n = 0; n < nn; ++n)
    if (weight[n] > 0)
      recovered[n] = (1.0 / weight[n]) * recovered[n];

  // Pass two: the norm of the difference, element by element.
  std::vector<double> out(_mesh->numElements(), 0.0);
  for (Index e = 0; e < numIntegratedElements(); ++e)
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
  const Index ne = numIntegratedElements();
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
  ctx.dim = _mesh->spaceDimension();
  ctx.time = _time;
  for (Index e = 0; e < numIntegratedElements(); ++e)
  {
    const auto & el = _mesh->element(e);
    mapPoint(*_mesh, el, ReferenceElement::get(el.type).centroid(), mp);
    const auto loc = localValues(*this, el);
    if (_mixed_order)
      mapCornerShape(el, mp);
    fillContext(ctx, el, mp, loc, nullptr, nullptr, 0);
    ctx.x = mp.x;
    ctx.element = e;
    ctx.block = el.block;
    ctx.state_owner = e;
    ctx.state_key = kCentroidStateKey;
    evaluateProperties(ctx);
    std::vector<double> row(nc);
    for (int c = 0; c < nc; ++c)
      row[c] = ctx.property(id, c).value();
    out.push_back(row);
  }
  return out;
}

std::vector<std::vector<double>>
Problem::propertyAtPoints(const std::string & property, const std::vector<Point> & points) const
{
  auto * self = const_cast<Problem *>(this);
  self->initialize();
  const int id = _props.id(property);
  const int nc = _props.components(property);
  PointLocator locator(*_mesh);
  std::vector<std::vector<double>> out;
  MappedPoint mp;
  QpContext ctx;
  ctx.dim = _mesh->spaceDimension();
  ctx.time = _time;
  for (const auto & p : points)
  {
    Index e;
    Point xi;
    if (!locator.locate(p, e, xi, 1e-8))
    {
      out.emplace_back(nc, std::numeric_limits<double>::quiet_NaN());
      continue;
    }
    const auto & el = _mesh->element(e);
    mapPoint(*_mesh, el, xi, mp);
    const auto loc = localValues(*this, el);
    if (_mixed_order)
      mapCornerShape(el, mp);
    fillContext(ctx, el, mp, loc, nullptr, nullptr, 0);
    ctx.x = mp.x;
    ctx.element = e;
    ctx.block = el.block;
    // A property with a history takes the average committed history of the element.
    ctx.state_owner = e;
    ctx.state_key = kCentroidStateKey;
    evaluateProperties(ctx);
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
  auto k = std::dynamic_pointer_cast<Kernel>(object(kernel));
  if (!k)
    throw InputError("'" + kernel + "' is not a kernel.");
  std::vector<Point> out;
  centroidContexts(
      [&](QpContext & ctx, int block)
      {
        ADVector3 F{ADReal(0.0), ADReal(0.0), ADReal(0.0)};
        if (k->hasFlux() && k->activeOnBlock(block))
          k->computeFlux(ctx, F);
        out.push_back({F[0].value(), F[1].value(), F[2].value()});
      });
  return out;
}

std::vector<std::vector<double>>
Problem::sectionStressAtCentroids(const std::string & kernel, double z) const
{
  auto k = std::dynamic_pointer_cast<Kernel>(object(kernel));
  if (!k)
    throw InputError("'" + kernel + "' is not a kernel.");
  std::vector<std::vector<double>> out;
  centroidContexts(
      [&](QpContext & ctx, int block)
      {
        auto row = k->activeOnBlock(block) ? k->sectionStress(ctx, z) : std::vector<double>{};
        if (row.empty() && k->activeOnBlock(block))
          throw InputError("'" + kernel + "' is not a structural member with a section.");
        out.push_back(std::move(row));
      });
  return out;
}

void
Problem::centroidContexts(const std::function<void(QpContext &, int)> & visit) const
{
  auto * self = const_cast<Problem *>(this);
  self->initialize();
  MappedPoint mp;
  QpContext ctx;
  ctx.dim = _mesh->spaceDimension();
  ctx.time = _time;
  for (Index e = 0; e < numIntegratedElements(); ++e)
  {
    const auto & el = _mesh->element(e);
    mapPoint(*_mesh, el, ReferenceElement::get(el.type).centroid(), mp);
    const auto loc = localValues(*this, el);
    if (_mixed_order)
      mapCornerShape(el, mp);
    fillContext(ctx, el, mp, loc, nullptr, nullptr, 0);
    ctx.x = mp.x;
    ctx.element = e;
    ctx.block = el.block;
    ctx.state_owner = e;
    ctx.state_key = kCentroidStateKey;
    evaluateProperties(ctx);
    visit(ctx, el.block);
  }
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
  for (Index e = 0; e < numIntegratedElements(); ++e)
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

std::vector<double>
Problem::elementIntegrals(const std::string & var) const
{
  const int v = var.empty() ? -1 : variableIndex(var);
  std::vector<double> out(_mesh->numElements(), 0.0);
  QuadratureSpec g;
  g.points = 3;
  MappedPoint mp;
  std::vector<IntegrationPoint> pts;
  for (Index e = 0; e < numIntegratedElements(); ++e)
  {
    const auto & el = _mesh->element(e);
    buildElementPoints(*_mesh, e, false, PointSet::Volume, g, pts);
    double total = 0;
    for (const auto & ip : pts)
    {
      mapPoint(*_mesh, el, ip.xi, mp);
      double s = v < 0 ? 1.0 : 0.0;
      for (int k = 0; v >= 0 && k < el.numNodes(); ++k)
        s += mp.N[k] * _U[dof(el.nodes[k], v)];
      total += s * ip.weight * coordFactor(mp.x);
    }
    out[e] = total;
  }
  return out;
}

double
Problem::boundaryArea(const std::vector<std::string> & boundaries) const
{
  if (!_global_boundary_areas.empty())
  {
    // Side sets do not overlap in the inputs this is used for, so the measures
    // of several side sets add.
    double total = 0.0;
    for (const auto & name : boundaries)
    {
      const auto it = _global_boundary_areas.find(name);
      if (it == _global_boundary_areas.end())
        throw InputError("Unknown side set '" + name + "'.");
      total += it->second;
    }
    return total;
  }
  for (const auto & name : boundaries)
    if (!_mesh->sidesets().count(name))
      throw InputError("Unknown side set '" + name + "'." +
                       didYouMean(name,
                                  [this]
                                  {
                                    std::vector<std::string> n;
                                    for (const auto & [k, v] : _mesh->sidesets())
                                      n.push_back(k);
                                    return n;
                                  }()));
  return sidesetMeasure(*_mesh, boundaries, _coord);
}

std::array<double, 3>
Problem::crackTipIntegrals(const std::vector<std::string> & displacements,
                           const Point & tip,
                           double angle,
                           double inner_radius,
                           double outer_radius,
                           double youngs_modulus,
                           double poissons_ratio,
                           bool plane_strain) const
{
  // The domain integrals of a crack tip in two dimensions, in the axes of the crack (x1 along
  // it, ahead of the tip): with the weight q = 1 inside the inner radius, 0 outside the outer
  // one and linear between,
  //   J = int (sigma_ij u_j,1 - W delta_1i) q,i dA,
  //   I = int (sigma_ij v_j,1 + s_ij u_j,1 - sigma_kl e_kl delta_1i) q,i dA,
  // with (v, s, e) the Williams field of mode I or II at K = 1, and K = E' I / 2.
  auto * self = const_cast<Problem *>(this);
  self->initialize();
  if (_mesh->spaceDimension() != 2)
    throw InputError("The crack tip integrals are available in two dimensions.");
  if (!(outer_radius > inner_radius && inner_radius >= 0.0))
    throw InputError("The radii of the crack tip integrals are 0 <= inner < outer.");
  const int ux = variableIndex(displacements.at(0)), uy = variableIndex(displacements.at(1));
  const int stress = _props.id("stress"), strain = _props.id("strain");
  const double mu = youngs_modulus / (2.0 * (1.0 + poissons_ratio));
  const double kappa =
      plane_strain ? 3.0 - 4.0 * poissons_ratio : (3.0 - poissons_ratio) / (1.0 + poissons_ratio);
  const double plane =
      plane_strain ? youngs_modulus / (1.0 - poissons_ratio * poissons_ratio) : youngs_modulus;
  const double c = std::cos(angle), sn = std::sin(angle);
  // The Williams displacements of mode I (m = 0) or II (m = 1) at K = 1, in the crack axes.
  const auto williams = [&](int m, double x1, double x2, double & v1, double & v2)
  {
    const double r = std::hypot(x1, x2), t = std::atan2(x2, x1);
    const double f = std::sqrt(r / (2.0 * M_PI)) / (2.0 * mu);
    const double ch = std::cos(t / 2), sh = std::sin(t / 2);
    if (m == 0)
    {
      v1 = f * ch * (kappa - 1.0 + 2.0 * sh * sh);
      v2 = f * sh * (kappa + 1.0 - 2.0 * ch * ch);
    }
    else
    {
      v1 = f * sh * (kappa + 1.0 + 2.0 * ch * ch);
      v2 = -f * ch * (kappa - 1.0 - 2.0 * sh * sh);
    }
  };
  // Its stresses (s11, s22, s12).
  const auto williamsStress = [&](int m, double x1, double x2, std::array<double, 3> & out)
  {
    const double r = std::hypot(x1, x2), t = std::atan2(x2, x1);
    const double a = 1.0 / std::sqrt(2.0 * M_PI * r);
    const double ch = std::cos(t / 2), sh = std::sin(t / 2), c3 = std::cos(1.5 * t),
                 s3 = std::sin(1.5 * t);
    if (m == 0)
      out = {a * ch * (1 - sh * s3), a * ch * (1 + sh * s3), a * sh * ch * c3};
    else
      out = {-a * sh * (2 + ch * c3), a * sh * ch * c3, a * ch * (1 - sh * s3)};
  };
  QuadratureSpec rule;
  rule.points = 4;
  std::vector<IntegrationPoint> pts;
  MappedPoint mp;
  QpContext ctx;
  ctx.dim = 2;
  ctx.time = _time;
  double J = 0.0, I[2] = {0.0, 0.0};
  for (Index e = 0; e < numIntegratedElements(); ++e)
  {
    const auto & el = _mesh->element(e);
    buildElementPoints(*_mesh, e, false, PointSet::Volume, rule, pts);
    const auto loc = localValues(*this, el);
    for (const auto & ip : pts)
    {
      mapPoint(*_mesh, el, ip.xi, mp);
      // The point in the crack axes, and the gradient of q there.
      const double dx = mp.x[0] - tip[0], dy = mp.x[1] - tip[1];
      const double x1 = c * dx + sn * dy, x2 = -sn * dx + c * dy;
      const double r = std::hypot(x1, x2);
      if (!(r > inner_radius && r < outer_radius))
        continue;
      const double dq1 = -x1 / r / (outer_radius - inner_radius),
                   dq2 = -x2 / r / (outer_radius - inner_radius);
      if (_mixed_order)
        mapCornerShape(el, mp);
      fillContext(ctx, el, mp, loc, nullptr, nullptr, 0);
      ctx.x = mp.x;
      ctx.element = e;
      ctx.block = el.block;
      evaluateProperties(ctx);
      // The stress, the strain and the displacement gradient, rotated to the crack axes.
      const double sxx = ctx.property(stress, 0).value(), syy = ctx.property(stress, 1).value(),
                   sxy = ctx.property(stress, 5).value();
      const double s11 = c * c * sxx + 2 * c * sn * sxy + sn * sn * syy;
      const double s22 = sn * sn * sxx - 2 * c * sn * sxy + c * c * syy;
      const double s12 = (c * c - sn * sn) * sxy + c * sn * (syy - sxx);
      const double W = 0.5 * (ctx.property(stress, 0).value() * ctx.property(strain, 0).value() +
                              ctx.property(stress, 1).value() * ctx.property(strain, 1).value() +
                              ctx.property(stress, 2).value() * ctx.property(strain, 2).value() +
                              ctx.property(stress, 5).value() * ctx.property(strain, 5).value());
      // grad u in global axes, then G' = R G R^T.
      const Point gx = {ctx.gradient(ux)[0].value(), ctx.gradient(ux)[1].value(), 0.0};
      const Point gy = {ctx.gradient(uy)[0].value(), ctx.gradient(uy)[1].value(), 0.0};
      const double R[2][2] = {{c, sn}, {-sn, c}};
      const double Gg[2][2] = {{gx[0], gx[1]}, {gy[0], gy[1]}};
      double G[2][2] = {{0, 0}, {0, 0}};
      for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
          for (int k = 0; k < 2; ++k)
            for (int l = 0; l < 2; ++l)
              G[i][j] += R[i][k] * Gg[k][l] * R[j][l];
      const double weight = ip.weight * coordFactor(mp.x);
      // J: (sigma_1j u_j,1 - W) q,1 + sigma_2j u_j,1 q,2.
      J += ((s11 * G[0][0] + s12 * G[1][0] - W) * dq1 + (s12 * G[0][0] + s22 * G[1][0]) * dq2) *
           weight;
      for (int m = 0; m < 2; ++m)
      {
        // The auxiliary gradients by central differences in the crack axes.
        const double h = 1e-6 * r;
        double a1, a2, b1, b2, d1, d2, e1, e2;
        williams(m, x1 + h, x2, a1, a2);
        williams(m, x1 - h, x2, b1, b2);
        williams(m, x1, x2 + h, d1, d2);
        williams(m, x1, x2 - h, e1, e2);
        const double v11 = (a1 - b1) / (2 * h), v21 = (a2 - b2) / (2 * h);
        const double v12 = (d1 - e1) / (2 * h), v22 = (d2 - e2) / (2 * h);
        std::array<double, 3> aux;
        williamsStress(m, x1, x2, aux);
        // sigma : e_aux, with e_aux the symmetric part of grad v.
        const double interaction = s11 * v11 + s22 * v22 + s12 * (v12 + v21);
        const double first = s11 * v11 + s12 * v21 + aux[0] * G[0][0] + aux[2] * G[1][0];
        const double second = s12 * v11 + s22 * v21 + aux[2] * G[0][0] + aux[1] * G[1][0];
        I[m] += ((first - interaction) * dq1 + second * dq2) * weight;
      }
    }
  }
  return {J, plane * I[0] / 2.0, plane * I[1] / 2.0};
}

double
Problem::normalIntegral(const std::vector<std::string> & components,
                        const std::string & boundary) const
{
  std::vector<int> vars;
  for (const auto & c : components)
    vars.push_back(variableIndex(c));
  QuadratureSpec g;
  g.points = 3;
  double total = 0;
  MappedPoint mp;
  std::vector<IntegrationPoint> pts;
  for (const auto & side : _mesh->sideset(boundary))
  {
    const auto & el = _mesh->element(side.first);
    buildSidePoints(*_mesh, side, false, g, pts);
    for (const auto & ip : pts)
    {
      mapPoint(*_mesh, el, ip.xi, mp);
      for (std::size_t d = 0; d < vars.size(); ++d)
      {
        double u = 0;
        for (int k = 0; k < el.numNodes(); ++k)
          u += mp.N[k] * _U[dof(el.nodes[k], vars[d])];
        total += u * ip.area[d] * coordFactor(mp.x);
      }
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
  // An advective face flux of DMCDM is taken at the points of the boundary faces of the control
  // volumes, where both sides are the node whose control volume the face bounds: the flux m u_f
  // that leaves, with the face value its scheme takes there (the assembly adds m (u_f - u_C), and
  // the mass flux carries the rest out at the node values).
  const bool dual_advective = isDual() && k->usesFaceSides();
  if (dual_advective)
    g = k->quadrature();
  const int nv = numVariables();
  double total = 0;
  MappedPoint mp;
  QpContext ctx;
  ctx.dim = _mesh->spaceDimension();
  ctx.time = _time;
  std::vector<IntegrationPoint> pts;
  for (const auto & side : _mesh->sideset(boundary))
  {
    const auto & el = _mesh->element(side.first);
    if (!k->activeOnBlock(el.block))
      continue;
    buildSidePoints(*_mesh, side, dual_advective, g, pts);
    const auto loc = localValues(*this, el);
    for (const auto & ip : pts)
    {
      mapPoint(*_mesh, el, ip.xi, mp);
      if (_mixed_order)
        mapCornerShape(el, mp);
      fillContext(ctx, el, mp, loc, nullptr, nullptr, 0);
      ctx.x = mp.x;
      ctx.element = side.first;
      ctx.block = el.block;
      ctx.normal = (1.0 / ip.weight) * ip.area;
      ctx.on_boundary = true;
      if (dual_advective)
      {
        ctx.on_face = true;
        ctx.side_u[0].resize(nv);
        for (int u = 0; u < nv; ++u)
          ctx.side_u[0][u] = ADReal(loc[ip.owner * nv + u]);
        ctx.side_u[1] = ctx.side_u[0];
        ctx.side_grad[0].clear();
        ctx.side_grad[1].clear();
        ctx.side_x[0] = ctx.side_x[1] = _mesh->node(el.nodes[ip.owner]);
      }
      evaluateProperties(ctx);
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
                  const std::vector<std::string> & cell_properties,
                  const std::vector<std::string> & fields) const
{
  for (const auto & name : fields)
    if (!hasVariable(name))
    {
      std::string names;
      for (const auto & v : _vars)
        names += " " + v.name;
      throw InputError("The output field '" + name +
                       "' is not a variable. The variables are:" + names + ".");
    }
  const auto written = [&](const Variable & v)
  { return fields.empty() || std::find(fields.begin(), fields.end(), v.name) != fields.end(); };
  // The unit is an attribute of the data array: ParaView and VisIt read the
  // array and do not use the attribute.
  const auto unit = [](const Variable & v)
  { return v.unit.empty() ? std::string() : " units=\"" + v.unit + "\""; };
  std::ofstream f(filename);
  if (!f)
    throw InputError("Cannot open '" + filename + "' for writing.");
  const auto & m = *_mesh;
  // The ghost elements of a distributed run belong to another process's file.
  const Index ne = numIntegratedElements();
  f << std::setprecision(12);
  f << "<?xml version=\"1.0\"?>\n<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" "
       "byte_order=\"LittleEndian\">\n<UnstructuredGrid>\n";
  f << "<Piece NumberOfPoints=\"" << m.numNodes() << "\" NumberOfCells=\"" << ne << "\">\n";
  f << "<PointData>\n";
  for (const auto & v : _vars)
  {
    if (!written(v))
      continue;
    f << "<DataArray type=\"Float64\" Name=\"" << v.name << "\"" << unit(v)
      << " format=\"ascii\">\n";
    for (Index n = 0; n < m.numNodes(); ++n)
      f << _U[dof(n, v.index)] << "\n";
    f << "</DataArray>\n";
  }
  f << "</PointData>\n<CellData>\n";
  f << "<DataArray type=\"Int32\" Name=\"block\" format=\"ascii\">\n";
  for (Index e = 0; e < ne; ++e)
    f << m.element(e).block << "\n";
  f << "</DataArray>\n";
  for (const auto & prop : cell_properties)
  {
    const auto vals = propertyAtCentroids(prop);
    const int nc = vals.empty() ? 1 : static_cast<int>(vals[0].size());
    f << "<DataArray type=\"Float64\" Name=\"" << prop << "\" NumberOfComponents=\"" << nc
      << "\" format=\"ascii\">\n";
    for (Index e = 0; e < ne && e < static_cast<Index>(vals.size()); ++e)
    {
      for (double x : vals[e])
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
  for (Index e = 0; e < ne; ++e)
  {
    const auto & el = m.element(e);
    for (int k : vtkNodeOrder(el.type))
      f << el.nodes[k] << " ";
    f << "\n";
  }
  f << "</DataArray>\n<DataArray type=\"Int64\" Name=\"offsets\" format=\"ascii\">\n";
  Index off = 0;
  for (Index e = 0; e < ne; ++e)
  {
    off += m.element(e).numNodes();
    f << off << "\n";
  }
  f << "</DataArray>\n<DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n";
  for (Index e = 0; e < ne; ++e)
    f << vtkCellType(m.element(e).type) << "\n";
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
