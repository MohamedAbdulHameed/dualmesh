// SPDX-License-Identifier: LGPL-2.1-or-later
//
// Assembly for the cell-centred finite volume method (the zero-thickness
// control volume formulation, ZFVM, of Reddy, Chapter 3).
//
// The unknowns sit at the cell centroids and at the centroids of the boundary
// faces.  For every face the normal flux is evaluated once, at the face
// centroid, and added to the balance of the cell on one side and subtracted
// from the balance of the cell (or boundary node) on the other, so the method
// is conservative by construction.
//
// The gradient at a face is the standard "corrected" (over-relaxed)
// approximation of the finite volume literature,
//
//     grad u |_f  =  gbar + [ (U_N - U_O) - gbar . d ] d / |d|^2 ,
//                                              d = x_N - x_O ,
//
// where gbar is the distance-weighted average of the reconstructed cell
// gradients of the two cells.  Those gradients come from the weighted
// least-squares fit of CellMesh::gradientStencil, which is exact for a linear
// field on any mesh; an earlier version used a Green-Gauss average of face
// values, which is not, and failed the patch test on a skewed mesh.  The first term on the right is
// lagged (deferred correction) and the second carries the derivatives, so the Jacobian stencil is
// the compact owner-neighbour pair.  On a mesh whose cell-centre line is parallel to the face
// normal, gbar contributes nothing to the normal flux and the scheme is exactly the two-point
// formula of Eqs. (3.2.16) and (3.4.26) of the book, with no lagging at all.
#include "dualmesh/base/Problem.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#include <algorithm>
#include <cmath>

namespace dualmesh
{

namespace
{
using Triplet = Eigen::Triplet<double>;

/// Quadrature that makes sense for an integral over a whole cell.
QuadratureSpec
cellQuadrature(const QuadratureSpec & spec)
{
  QuadratureSpec out = spec;
  switch (spec.kind)
  {
  case QuadratureSpec::Kind::Nodal:
  case QuadratureSpec::Kind::Interface:
  case QuadratureSpec::Kind::ControlDomainTrapezoid:
    // "Lumping at the node" in a cell-centred method means evaluating the
    // integrand once, at the cell centroid.
    out.kind = QuadratureSpec::Kind::Gauss;
    out.points = 1;
    break;
  default:
    break;
  }
  return out;
}
} // namespace

std::vector<std::vector<Point>>
Problem::cellGradients(const Vector & U) const
{
  const CellMesh & cm = *_cells;
  const int nv = numVariables();
  std::vector<std::vector<Point>> g(nv, std::vector<Point>(cm.numCells(), Point{0, 0, 0}));
  const int nthreads = effectiveThreads();
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(nthreads) if (nthreads > 1)
#endif
  for (Index c = 0; c < cm.numCells(); ++c)
    for (int v = 0; v < nv; ++v)
    {
      const double uc = U[dof(c, v)];
      Point gc{0, 0, 0};
      for (const auto & [entity, coefficient] : cm.gradientStencil(c))
        gc = gc + (U[dof(entity, v)] - uc) * coefficient;
      g[v][c] = gc;
    }
  (void) nthreads;
  return g;
}

void
Problem::fillContextCell(QpContext & ctx,
                         Index cell,
                         const std::vector<std::vector<Point>> & grads) const
{
  const int nv = numVariables();
  ctx.u.resize(nv);
  ctx.grad_u.resize(nv);
  ctx.u_lag.resize(nv);
  ctx.grad_u_lag.resize(nv);
  ctx.u_old.assign(nv, 0.0);
  ctx.grad_u_old.assign(nv, Point{0, 0, 0});
  for (int v = 0; v < nv; ++v)
  {
    ctx.u[v] = ADReal(_U[dof(cell, v)]);
    const Point & g = grads[v][cell];
    ctx.grad_u[v] = {ADReal(g[0]), ADReal(g[1]), ADReal(g[2])};
    ctx.u_lag[v] = ctx.u[v];
    ctx.grad_u_lag[v] = ctx.grad_u[v];
    ctx.u_old[v] = ctx.u[v].value();
    ctx.grad_u_old[v] = g;
  }
  ctx.x = _cells->entityPoint(cell);
  ctx.element = cell;
  ctx.block = _cells->cellBlock(cell);
}

void
Problem::assembleCellFiniteVolume(const Vector & U,
                                  const AssemblyOptions & opts,
                                  Vector & R,
                                  SparseMatrix * J)
{
  const CellMesh & cm = *_cells;
  const int nv = numVariables();
  const int dim = _mesh->dimension();
  const bool want_jac = J != nullptr;
  R.setZero(numDofs());

  const bool use_lag = opts.mode == LinearizationMode::Picard && opts.lagged;
  // Gradients used for the (lagged) non-orthogonal correction and by kernels
  // that need a gradient inside a cell.
  const auto grad_cur = cellGradients(U);
  const auto grad_lag = use_lag ? cellGradients(*opts.lagged) : grad_cur;
  const auto grad_old = opts.old ? cellGradients(*opts.old) : grad_cur;

  const auto weightOf = [&](const ResidualObject * k, bool time_kernel)
  {
    double w = time_kernel ? 1.0 : opts.theta;
    if (k->scalesWithLoad())
      w *= opts.load_factor;
    return w;
  };

  const auto scatter = [&](const std::vector<ADReal> & Rloc,
                           const std::vector<Index> & dofs,
                           const std::vector<char> & touched,
                           std::vector<Triplet> & triplets,
                           bool atomic_update)
  {
    const int ld = static_cast<int>(dofs.size());
    for (int i = 0; i < ld; ++i)
    {
      if (!touched[i])
        continue;
      const double value = Rloc[i].value();
      double & target = R[dofs[i]];
      if (atomic_update)
      {
#pragma omp atomic
        target += value;
      }
      else
        target += value;
      if (want_jac)
      {
        const int n = std::min(Rloc[i].size(), ld);
        const double * d = Rloc[i].derivatives();
        for (int j = 0; j < n; ++j)
          if (d[j] != 0.0)
            triplets.emplace_back(dofs[i], dofs[j], d[j]);
      }
    }
  };

  // ---- face fluxes ---------------------------------------------------------
  // Every face is independent: the two balances it contributes to are updated
  // atomically and each thread keeps its own list of matrix entries.
  const int nthreads = effectiveThreads();
  const bool atomic = nthreads > 1;
  std::vector<ThreadScratch> scratch(nthreads);
  for (auto & s : scratch)
  {
    s.ctx.dim = dim;
    s.ctx.time = _time;
    s.ctx.dt = opts.dt;
    s.ctx.eigenvalue = _eigenvalue;
    s.ctx.load_factor = opts.load_factor;
    s.ctx.mode = opts.mode;
  }
  const auto assembleFace = [&](ThreadScratch & sc, Index face_index)
  {
    auto & dofs = sc.dofs;
    auto & Rloc = sc.Rloc;
    auto & touched = sc.touched;
    auto & ctx = sc.ctx;
    const CellFace & f = cm.faces()[face_index];
    // In a distributed run another process integrates this face.
    if (!integratesFace(f))
      return;
    const bool interior = f.neighbor >= 0;
    const Index other = interior ? f.neighbor : f.boundary_entity;
    const int block = cm.cellBlock(f.owner);

    std::vector<const Kernel *> & ks = sc.kernels;
    ks.clear();
    for (const auto & k : _kernels)
    {
      if (!k->hasFlux() || !k->activeOnBlock(block))
        continue;
      const auto & vb = _vars[k->variable()].blocks;
      if (!vb.empty() && !vb.count(block))
        continue;
      if (k->isTimeKernel() ? !opts.include_time_kernels : !opts.include_steady_terms)
        continue;
      ks.push_back(k.get());
    }
    if (ks.empty())
      return;

    const Point xO = cm.entityPoint(f.owner);
    const Point xN = cm.entityPoint(other);
    const Point d = xN - xO;
    const double dd = dot(d, d);
    const Point nhat = (1.0 / f.measure) * f.area;
    double w = 0.0;
    if (interior)
    {
      const double dO = norm(f.centroid - xO);
      const double dN = norm(f.centroid - xN);
      w = (dO + dN) > 0 ? dN / (dO + dN) : 0.5;
    }

    // Coefficients of the one-sided quadratic normal derivative at a boundary
    // face: du/dn = cb U_b + cO U_O + c2 U_2 (Eq. (3.2.14) of the book).
    bool second_order = false;
    double cb = 0, cO = 0, c2 = 0;
    Index cell2 = -1;
    Point tO{0, 0, 0}, t2{0, 0, 0}; // tangential offsets of the two centroids
    if (!interior && _boundary_gradient == BoundaryGradient::SecondOrder && f.second_cell >= 0)
    {
      cell2 = f.second_cell;
      const Point rO = xO - f.centroid;
      const Point r2 = cm.entityPoint(cell2) - f.centroid;
      const double s1 = -dot(rO, nhat);
      const double s2 = -dot(r2, nhat);
      tO = rO + s1 * nhat;
      t2 = r2 + s2 * nhat;
      if (s1 > 0 && s2 > s1)
      {
        // Derivative of the Lagrange interpolant at s = 0, along -n, negated.
        cb = (s1 + s2) / (s1 * s2);
        cO = s2 / (s1 * (s1 - s2));
        c2 = s1 / (s2 * (s2 - s1));
        second_order = true;
      }
    }

    // The entities the face flux depends on.  The first two are the owner and
    // the entity across the face, and the third, for the second-order boundary
    // gradient, the next cell in.  The reconstructed cell gradients that enter
    // the non-orthogonal correction depend on every entity of the owner's and
    // the neighbour's least-squares stencils as well, and those follow, so
    // that the derivatives of the correction are carried like any other and
    // the Jacobian is exact.  A lagged correction instead converges only
    // linearly, at a rate that on a strongly skewed mesh (a pyramid's centroid
    // lies far off the normal through its faces) can be as slow as 0.99 per
    // iteration.  If the full stencil would overflow the automatic
    // differentiation budget, the correction falls back to being lagged.
    auto & ents = sc.entities;
    ents.assign({f.owner, other});
    if (second_order)
      ents.push_back(cell2);
    const std::size_t direct = ents.size();
    const auto slotOf = [&](Index e)
    {
      for (std::size_t i = 0; i < ents.size(); ++i)
        if (ents[i] == e)
          return static_cast<int>(i);
      ents.push_back(e);
      return static_cast<int>(ents.size() - 1);
    };
    std::vector<Index> gradient_cells = {f.owner};
    if (interior)
      gradient_cells.push_back(f.neighbor);
    if (second_order)
      gradient_cells.push_back(cell2);
    bool exact_correction = want_jac;
    if (exact_correction)
    {
      for (Index c : gradient_cells)
        for (const auto & entry : cm.gradientStencil(c))
          slotOf(entry.first);
      if (static_cast<int>(ents.size()) * nv > kMaxDerivatives)
      {
        ents.resize(direct);
        exact_correction = false;
      }
    }

    const int ldx = static_cast<int>(ents.size()) * nv;
    dofs.resize(ldx);
    for (std::size_t i = 0; i < ents.size(); ++i)
      for (int v = 0; v < nv; ++v)
        dofs[i * nv + v] = dof(ents[i], v);
    Rloc.assign(ldx, ADReal(0.0));
    touched.assign(ldx, 0);
    const int ndx = want_jac ? ldx : 0;

    // The reconstructed gradient of a cell as an automatic differentiation
    // vector: G_c = sum_e (U_e - U_c) a_e over the stencil, so its derivative
    // is a_e with respect to U_e and minus their sum with respect to U_c.
    const auto cellGradient = [&](Index c, int v)
    {
      const Point & value = grad_cur[v][c];
      ADVector3 g;
      for (int k = 0; k < 3; ++k)
        g[k] = ADReal::withZeroDerivatives(value[k], ndx);
      if (!exact_correction)
        return g;
      Point sum{0, 0, 0};
      for (const auto & [e, a] : cm.gradientStencil(c))
      {
        const int slot = slotOf(e) * nv + v;
        for (int k = 0; k < 3; ++k)
          g[k].setDerivative(slot, g[k].derivatives()[slot] + a[k]);
        sum = sum + a;
      }
      const int self = slotOf(c) * nv + v;
      for (int k = 0; k < 3; ++k)
        g[k].setDerivative(self, g[k].derivatives()[self] - sum[k]);
      return g;
    };

    ctx.element = f.owner;
    ctx.block = block;
    ctx.on_boundary = !interior;
    ctx.normal = nhat;
    ctx.x = f.centroid;
    ctx.u.resize(nv);
    ctx.grad_u.resize(nv);
    ctx.u_lag.resize(nv);
    ctx.grad_u_lag.resize(nv);
    ctx.u_old.assign(nv, 0.0);
    ctx.grad_u_old.assign(nv, Point{0, 0, 0});

    for (int v = 0; v < nv; ++v)
    {
      const double UO = U[dofs[v]];
      const double UN = U[dofs[nv + v]];
      // Face value.
      ADReal uf = ADReal::withZeroDerivatives(w * UO + (1.0 - w) * UN, ndx);
      if (ndx > 0)
      {
        uf.setDerivative(v, w);
        uf.setDerivative(nv + v, 1.0 - w);
      }
      ctx.u[v] = uf;

      // Reconstructed gradient at the face, the distance-weighted average of
      // the two cell gradients.
      const ADVector3 gradO = cellGradient(f.owner, v);
      ADVector3 gbar = gradO;
      if (interior)
      {
        const ADVector3 gradN = cellGradient(f.neighbor, v);
        for (int k = 0; k < 3; ++k)
          gbar[k] = w * gradO[k] + (1.0 - w) * gradN[k];
      }
      Point gbar_value{gbar[0].value(), gbar[1].value(), gbar[2].value()};

      ADVector3 g;
      if (second_order)
      {
        // The two cell centroids are not on the normal through the face
        // centroid, so their values are first moved onto it with the cell
        // gradients.  Without this the one-sided quadratic would not even
        // reproduce a linear field on a triangular mesh.
        const double U2 = U[dofs[2 * nv + v]];
        const ADVector3 grad2 = cellGradient(cell2, v);
        ADReal uO = ADReal::withZeroDerivatives(UO, ndx);
        ADReal uN = ADReal::withZeroDerivatives(UN, ndx);
        ADReal u2 = ADReal::withZeroDerivatives(U2, ndx);
        if (ndx > 0)
        {
          uO.setDerivative(v, 1.0);
          uN.setDerivative(nv + v, 1.0);
          u2.setDerivative(2 * nv + v, 1.0);
        }
        ADReal movedO = uO, moved2 = u2;
        for (int k = 0; k < 3; ++k)
        {
          movedO -= gradO[k] * tO[k];
          moved2 -= grad2[k] * t2[k];
        }
        const ADReal dn = cb * uN + cO * movedO + c2 * moved2;
        ADReal gn(0.0);
        for (int k = 0; k < 3; ++k)
          gn += gbar[k] * nhat[k];
        for (int k = 0; k < 3; ++k)
          g[k] = dn * nhat[k] + (gbar[k] - gn * nhat[k]);
      }
      else
      {
        ADReal du = ADReal::withZeroDerivatives(UN - UO, ndx);
        if (ndx > 0)
        {
          du.setDerivative(nv + v, 1.0);
          du.setDerivative(v, -1.0);
        }
        ADReal proj(0.0);
        for (int k = 0; k < 3; ++k)
          proj += gbar[k] * d[k];
        const ADReal s = (du - proj) * (dd > 0 ? 1.0 / dd : 0.0);
        for (int k = 0; k < 3; ++k)
          g[k] = gbar[k] + s * d[k];
      }

      ctx.grad_u[v] = g;

      // Lagged and old values, for Picard iteration and transient terms.
      const double ul = use_lag
                            ? w * (*opts.lagged)[dofs[v]] + (1 - w) * (*opts.lagged)[dofs[nv + v]]
                            : ctx.u[v].value();
      ctx.u_lag[v] = ADReal(ul);
      Point gl = use_lag ? grad_lag[v][f.owner] : gbar_value;
      if (use_lag && interior)
        gl = (w * gl) + ((1.0 - w) * grad_lag[v][f.neighbor]);
      ctx.grad_u_lag[v] = {ADReal(gl[0]), ADReal(gl[1]), ADReal(gl[2])};
      if (opts.old)
      {
        ctx.u_old[v] = w * (*opts.old)[dofs[v]] + (1 - w) * (*opts.old)[dofs[nv + v]];
        Point go = grad_old[v][f.owner];
        if (interior)
          go = (w * go) + ((1.0 - w) * grad_old[v][f.neighbor]);
        ctx.grad_u_old[v] = go;
      }
      else
      {
        ctx.u_old[v] = ctx.u[v].value();
        for (int k = 0; k < 3; ++k)
          ctx.grad_u_old[v][k] = ctx.grad_u[v][k].value();
      }
    }
    ctx.state_domain = 1;
    ctx.state_owner = face_index;
    ctx.state_key = 0;
    evaluateProperties(ctx);

    const double cf = coordFactor(f.centroid);
    // The equation of a boundary entity is the condition on its face's flux,
    // and its scale does not matter; it is weighted like the face so that
    // the integrated boundary conditions, weighted the same way, enter it
    // consistently.  On the axis of an axisymmetric or the centre of a
    // spherical problem the weight r vanishes, which would leave the entity
    // with an empty equation, so the weight there is 1 instead.
    const double ef = interior ? cf : boundaryEntityFactor(f.centroid);
    const auto normalFlux = [&](const Kernel * k)
    {
      ADVector3 F{ADReal(0.0), ADReal(0.0), ADReal(0.0)};
      k->computeFlux(ctx, F);
      ADReal fn(0.0);
      for (int kk = 0; kk < dim; ++kk)
        if (f.area[kk] != 0.0)
          fn += F[kk] * f.area[kk];
      return fn;
    };
    std::vector<ADReal> fluxes;
    fluxes.reserve(ks.size());
    for (const Kernel * k : ks)
      fluxes.push_back(normalFlux(k));
    // A face between two blocks: the properties of the two sides differ, and
    // the flux that is continuous across the face is the one of the
    // distance-weighted harmonic mean of the two coefficients,
    // F = 1 / ((1 - w) / F_O + w / F_N), where F_O and F_N are the fluxes
    // with the properties of the owner and of the neighbour and w = d_N / d.
    // For F = k grad u with a two-point gradient this is the harmonic
    // average of k that makes the flux continuous when u is linear on each
    // side of the face, which keeps the method second order across a jump
    // of the coefficient.
    const int neighbor_block = interior ? _cells->cellBlock(f.neighbor) : block;
    if (interior && neighbor_block != block)
    {
      ctx.element = f.neighbor;
      ctx.block = neighbor_block;
      evaluateProperties(ctx);
      for (std::size_t i = 0; i < ks.size(); ++i)
      {
        if (!ks[i]->activeOnBlock(neighbor_block))
          continue;
        const ADReal fN = normalFlux(ks[i]);
        const ADReal fO = fluxes[i];
        // The conductances of the two sides, the derivatives of their fluxes
        // with respect to the value across the face, set the harmonic mean
        // also where the flux vanishes (e.g., the zero state at which the
        // eigenvalue study assembles its operators).
        const int slot = nv + ks[i]->variable();
        const double cO = ndx > 0 ? fO.derivatives()[slot] : 0.0;
        const double cN = ndx > 0 ? fN.derivatives()[slot] : 0.0;
        // The harmonic mean of the two fluxes themselves carries every
        // derivative (also that of a coefficient that depends on the
        // solution), so it is preferred.  Where the fluxes vanish, the
        // conductances set it.
        const double scale = std::max(std::abs(fO.value()), std::abs(fN.value()));
        if (fO.value() * fN.value() > 0.0 &&
            std::min(std::abs(fO.value()), std::abs(fN.value())) > 1e-12 * scale)
          fluxes[i] = fO * fN / ((1.0 - w) * fN + w * fO);
        else if (cO * cN > 0.0)
          fluxes[i] = fO * (cN / ((1.0 - w) * cN + w * cO));
        else
          fluxes[i] = w * fO + (1.0 - w) * fN;
      }
      ctx.element = f.owner;
      ctx.block = block;
    }
    for (std::size_t i = 0; i < ks.size(); ++i)
    {
      const Kernel * k = ks[i];
      const int v = k->variable();
      ADReal fn = fluxes[i];
      fn *= weightOf(k, k->isTimeKernel());
      Rloc[v] -= cf * fn;
      Rloc[nv + v] += ef * fn;
      touched[v] = 1;
      touched[nv + v] = 1;
    }
    scatter(Rloc, dofs, touched, sc.triplets, atomic);
  };

  std::string thread_error;
#ifdef _OPENMP
#pragma omp parallel num_threads(nthreads) if (nthreads > 1)
  {
    ThreadScratch & sc = scratch[omp_get_thread_num()];
#pragma omp for schedule(static)
    for (Index i = 0; i < static_cast<Index>(cm.faces().size()); ++i)
    {
      try
      {
        assembleFace(sc, i);
      }
      catch (const std::exception & ex)
      {
#pragma omp critical(dualmesh_thread_error)
        if (thread_error.empty())
          thread_error = ex.what();
      }
    }
  }
#else
  for (Index i = 0; i < static_cast<Index>(cm.faces().size()); ++i)
    assembleFace(scratch[0], i);
#endif
  if (!thread_error.empty())
    throw std::runtime_error(thread_error);

  // ---- cell (volume) integrals: sources and time terms ---------------------
  auto & dofs = scratch[0].dofs;
  auto & Rloc = scratch[0].Rloc;
  auto & touched = scratch[0].touched;
  auto & ctx = scratch[0].ctx;
  std::vector<IntegrationPoint> pts;
  MappedPoint geo;
  // A source may depend on the gradient, as the convective inertia
  // rho u . grad u does.  The cell gradient is the least-squares
  // reconstruction G_c = sum_e (U_e - U_c) a_e, so its derivatives are
  // carried over the cell's stencil, which makes the Jacobian of such a source
  // exact; a lagged gradient would reduce Newton's method to linear
  // convergence.  A stencil that would overflow the automatic
  // differentiation budget keeps the gradient lagged.
  std::vector<Index> cell_entities;
  for (Index c = 0; c < numIntegratedElements(); ++c)
  {
    const int block = cm.cellBlock(c);
    cell_entities.assign(1, c);
    if (want_jac)
    {
      for (const auto & entry : cm.gradientStencil(c))
        if (std::find(cell_entities.begin(), cell_entities.end(), entry.first) ==
            cell_entities.end())
          cell_entities.push_back(entry.first);
      if (static_cast<int>(cell_entities.size()) * nv > kMaxDerivatives)
        cell_entities.resize(1);
    }
    const bool exact_gradient = cell_entities.size() > 1;
    const int ld = static_cast<int>(cell_entities.size()) * nv;
    const int nd = want_jac ? ld : 0;
    dofs.resize(ld);
    for (std::size_t i = 0; i < cell_entities.size(); ++i)
      for (int v = 0; v < nv; ++v)
        dofs[i * nv + v] = dof(cell_entities[i], v);
    Rloc.assign(ld, ADReal(0.0));
    touched.assign(ld, 0);
    bool any = false;

    for (const auto & group : _groups)
    {
      std::vector<const Kernel *> ks;
      for (const Kernel * k : group.kernels)
      {
        if (!k->hasSource() || !k->activeOnBlock(block))
          continue;
        const auto & vb = _vars[k->variable()].blocks;
        if (!vb.empty() && !vb.count(block))
          continue;
        if (k->isTimeKernel() ? !opts.include_time_kernels : !opts.include_steady_terms)
          continue;
        if (k->isTimeKernel() && opts.dt <= 0)
          continue;
        ks.push_back(k);
      }
      if (ks.empty())
        continue;
      any = true;
      buildElementPoints(*_mesh, c, false, PointSet::Volume, cellQuadrature(group.spec), pts);
      for (const auto & ip : pts)
      {
        mapPoint(*_mesh, _mesh->element(c), ip.xi, geo);
        ctx.element = c;
        ctx.block = block;
        ctx.on_boundary = false;
        ctx.normal = {0, 0, 0};
        ctx.x = geo.x;
        ctx.u.resize(nv);
        ctx.grad_u.resize(nv);
        ctx.u_lag.resize(nv);
        ctx.grad_u_lag.resize(nv);
        ctx.u_old.assign(nv, 0.0);
        ctx.grad_u_old.assign(nv, Point{0, 0, 0});
        for (int v = 0; v < nv; ++v)
        {
          ADReal uv = ADReal::withZeroDerivatives(U[dofs[v]], nd);
          if (nd > 0)
            uv.setDerivative(v, 1.0);
          ctx.u[v] = uv;
          const Point & g = grad_cur[v][c];
          ctx.grad_u[v] = {ADReal::withZeroDerivatives(g[0], nd),
                           ADReal::withZeroDerivatives(g[1], nd),
                           ADReal::withZeroDerivatives(g[2], nd)};
          if (exact_gradient)
          {
            Point sum{0, 0, 0};
            for (const auto & [e, a] : cm.gradientStencil(c))
            {
              const auto it = std::find(cell_entities.begin(), cell_entities.end(), e);
              const int slot = static_cast<int>(it - cell_entities.begin()) * nv + v;
              for (int k = 0; k < 3; ++k)
                ctx.grad_u[v][k].setDerivative(slot, ctx.grad_u[v][k].derivatives()[slot] + a[k]);
              sum = sum + a;
            }
            for (int k = 0; k < 3; ++k)
              ctx.grad_u[v][k].setDerivative(v, ctx.grad_u[v][k].derivatives()[v] - sum[k]);
          }
          // The value at the quadrature point is the linear reconstruction
          // u_c + G_c . (x - x_c), which is what makes a multi-point rule
          // second-order accurate; with the bare cell value a source such as
          // the hoop stress sigma_tt / r of an axisymmetric body would be
          // integrated with an O(1) error in the cells next to the axis.
          const Point dx = geo.x - cm.entityPoint(c);
          for (int k = 0; k < dim; ++k)
            if (dx[k] != 0.0)
              ctx.u[v] += ctx.grad_u[v][k] * dx[k];
          const Point & gl = grad_lag[v][c];
          double lag_value = use_lag ? (*opts.lagged)[dofs[v]] : U[dofs[v]];
          double old_value = opts.old ? (*opts.old)[dofs[v]] : U[dofs[v]];
          const Point & go = grad_old[v][c];
          for (int k = 0; k < dim; ++k)
          {
            lag_value += (use_lag ? gl[k] : g[k]) * dx[k];
            old_value += (opts.old ? go[k] : g[k]) * dx[k];
          }
          ctx.u_lag[v] = ADReal(lag_value);
          ctx.grad_u_lag[v] = {ADReal(gl[0]), ADReal(gl[1]), ADReal(gl[2])};
          ctx.u_old[v] = old_value;
          ctx.grad_u_old[v] = go;
        }
        ctx.state_domain = 0;
        ctx.state_owner = c;
        ctx.state_key = stateKey(4,
                                 static_cast<unsigned>(&group - _groups.data()),
                                 0,
                                 static_cast<unsigned>(&ip - pts.data()));
        evaluateProperties(ctx);
        const double cf = coordFactor(geo.x) * ip.weight;
        for (const Kernel * k : ks)
        {
          const ADReal S = k->computeSource(ctx) * cf * weightOf(k, k->isTimeKernel());
          Rloc[k->variable()] += S;
          touched[k->variable()] = 1;
        }
      }
    }
    if (any)
      scatter(Rloc, dofs, touched, scratch[0].triplets, false);
  }

  // ---- integrated boundary conditions --------------------------------------
  if (opts.include_steady_terms)
    for (const auto & bc : _ibcs)
    {
      const int v = bc->variable();
      const double wfac = opts.theta * (bc->scalesWithLoad() ? opts.load_factor : 1.0);
      if (const auto * ib = dynamic_cast<const InterfaceBC *>(bc.get()))
      {
        // A gap between two boundaries: the boundary unknown of every primary
        // face is paired with the boundary unknown of the secondary face that
        // contains the closest point.  The primary face's equation receives
        // the flux q, the secondary face's equation -q, both multiplied by the
        // primary face's measure, so that the heat (or force) is conserved.
        const auto & pairs = _interface_pairs.at(bc.get());
        const int vs = ib->secondaryVariable();
        std::size_t face_index = 0;
        for (const auto & bname : bc->boundaries())
          for (int fi : cm.boundaryFaceIndices(*_mesh, bname))
          {
            const auto & pair = pairs[face_index++].front();
            const CellFace & f = cm.faces()[fi];
            const CellFace & g = cm.faces()[pair.face];
            const int block = cm.cellBlock(f.owner);
            if (!bc->activeOnBlock(block))
              continue;
            const int ld = 2 * nv;
            const int nd = want_jac ? ld : 0;
            dofs.resize(ld);
            for (int k = 0; k < nv; ++k)
            {
              dofs[k] = dof(f.boundary_entity, k);
              dofs[nv + k] = dof(g.boundary_entity, k);
            }
            Rloc.assign(ld, ADReal(0.0));
            touched.assign(ld, 0);
            ctx.element = f.owner;
            ctx.block = block;
            ctx.on_boundary = true;
            ctx.normal = (1.0 / f.measure) * f.area;
            ctx.x = f.centroid;
            ctx.x_other = pair.x;
            ctx.u.resize(nv);
            ctx.grad_u.resize(nv);
            ctx.u_other.resize(nv);
            ctx.u_lag.resize(nv);
            ctx.grad_u_lag.resize(nv);
            ctx.u_old.assign(nv, 0.0);
            ctx.grad_u_old.assign(nv, Point{0, 0, 0});
            for (int k = 0; k < nv; ++k)
            {
              ADReal up = ADReal::withZeroDerivatives(U[dofs[k]], nd);
              ADReal us = ADReal::withZeroDerivatives(U[dofs[nv + k]], nd);
              if (nd > 0)
              {
                up.setDerivative(k, 1.0);
                us.setDerivative(nv + k, 1.0);
              }
              ctx.u[k] = up;
              ctx.u_other[k] = us;
              const Point & gr = grad_cur[k][f.owner];
              ctx.grad_u[k] = {ADReal(gr[0]), ADReal(gr[1]), ADReal(gr[2])};
              ctx.u_lag[k] = ADReal(U[dofs[k]]);
              ctx.grad_u_lag[k] = ctx.grad_u[k];
              ctx.u_old[k] = opts.old ? (*opts.old)[dofs[k]] : U[dofs[k]];
              ctx.grad_u_old[k] = gr;
            }
            ctx.state_domain = 0;
            ctx.state_owner = f.owner;
            ctx.state_key = stateKey(6,
                                     static_cast<unsigned>(&bc - _ibcs.data()),
                                     static_cast<unsigned>(f.side.second),
                                     0);
            evaluateProperties(ctx);
            const double wt = f.measure * boundaryEntityFactor(f.centroid) * wfac;
            const ADReal q = ib->computeInterfaceFlux(ctx) * wt;
            Rloc[v] -= q;
            Rloc[nv + vs] += q;
            touched[v] = 1;
            touched[nv + vs] = 1;
            scatter(Rloc, dofs, touched, scratch[0].triplets, false);
          }
        continue;
      }
      for (const auto & bname : bc->boundaries())
        for (int fi : cm.boundaryFaceIndices(*_mesh, bname))
        {
          const CellFace & f = cm.faces()[fi];
          const int block = cm.cellBlock(f.owner);
          if (!bc->activeOnBlock(block))
            continue;
          const int ld = 2 * nv;
          const int nd = want_jac ? ld : 0;
          dofs.resize(ld);
          for (int k = 0; k < nv; ++k)
          {
            dofs[k] = dof(f.owner, k);
            dofs[nv + k] = dof(f.boundary_entity, k);
          }
          Rloc.assign(ld, ADReal(0.0));
          touched.assign(ld, 0);
          ctx.element = f.owner;
          ctx.block = block;
          ctx.on_boundary = true;
          ctx.normal = (1.0 / f.measure) * f.area;
          ctx.x = f.centroid;
          ctx.u.resize(nv);
          ctx.grad_u.resize(nv);
          ctx.u_lag.resize(nv);
          ctx.grad_u_lag.resize(nv);
          ctx.u_old.assign(nv, 0.0);
          ctx.grad_u_old.assign(nv, Point{0, 0, 0});
          for (int k = 0; k < nv; ++k)
          {
            ADReal ub = ADReal::withZeroDerivatives(U[dofs[nv + k]], nd);
            if (nd > 0)
              ub.setDerivative(nv + k, 1.0);
            ctx.u[k] = ub;
            const Point & g = grad_cur[k][f.owner];
            ctx.grad_u[k] = {ADReal(g[0]), ADReal(g[1]), ADReal(g[2])};
            ctx.u_lag[k] = ADReal(use_lag ? (*opts.lagged)[dofs[nv + k]] : U[dofs[nv + k]]);
            ctx.grad_u_lag[k] = ctx.grad_u[k];
            ctx.u_old[k] = opts.old ? (*opts.old)[dofs[nv + k]] : U[dofs[nv + k]];
            ctx.grad_u_old[k] = g;
          }
          ctx.state_domain = 0;
          ctx.state_owner = f.owner;
          ctx.state_key = stateKey(5,
                                   static_cast<unsigned>(&bc - _ibcs.data()),
                                   static_cast<unsigned>(f.side.second),
                                   0);
          evaluateProperties(ctx);
          const double wt = f.measure * boundaryEntityFactor(f.centroid) * wfac;
          const ADReal q = bc->computeBoundaryFlux(ctx) * wt;
          Rloc[nv + v] -= q;
          touched[nv + v] = 1;
          scatter(Rloc, dofs, touched, scratch[0].triplets, false);
        }
    }

  // ---- concentrated sources ------------------------------------------------
  if (opts.include_steady_terms)
    for (const auto & load : _loads)
    {
      const int v = load->variable();
      const double wfac = opts.theta * (load->scalesWithLoad() ? opts.load_factor : 1.0);
      for (Index n : load->nodes())
      {
        if (!_owned_entity.empty() && !_owned_entity[n])
          continue;
        R[dof(n, v)] -= wfac * load->computeValue(entityPoint(n), _time);
      }
    }

  if (want_jac)
  {
    std::vector<Triplet> trip;
    std::size_t total = 0;
    for (const auto & s : scratch)
      total += s.triplets.size();
    trip.reserve(total);
    for (const auto & s : scratch)
      trip.insert(trip.end(), s.triplets.begin(), s.triplets.end());
    J->resize(numDofs(), numDofs());
    J->setFromTriplets(trip.begin(), trip.end());
  }
}

} // namespace dualmesh
