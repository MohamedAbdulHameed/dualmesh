// SPDX-License-Identifier: LGPL-2.1-or-later
//
// Residual/Jacobian assembly and the steady and transient executioners.
#include "dualmesh/base/Console.h"
#include "dualmesh/base/Problem.h"

#include "dualmesh/linalg/IncompleteLU.h"
#include "dualmesh/linalg/PetscSolver.h"
#include "dualmesh/linalg/SaddlePointSolver.h"

#include <Eigen/IterativeLinearSolvers>
#include <Eigen/SparseLU>
#include <unsupported/Eigen/IterativeSolvers>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>

namespace dualmesh
{

namespace
{
using Triplet = Eigen::Triplet<double>;

/// Scatter an element-local AD residual into the global residual/Jacobian.
/// With @p atomic the residual is updated atomically, which is what lets
/// several threads assemble different elements into the same vector.
void
scatter(const std::vector<ADReal> & Rloc,
        const std::vector<Index> & dofs,
        const std::vector<char> & touched,
        Vector & R,
        std::vector<Triplet> * trip,
        bool atomic)
{
  const int ld = static_cast<int>(dofs.size());
  for (int i = 0; i < ld; ++i)
  {
    if (!touched[i])
      continue;
    const double value = Rloc[i].value();
    double & target = R[dofs[i]];
    if (atomic)
    {
#pragma omp atomic
      target += value;
    }
    else
      target += value;
    if (trip)
    {
      const int n = std::min(Rloc[i].size(), ld);
      const double * d = Rloc[i].derivatives();
      for (int j = 0; j < n; ++j)
        if (d[j] != 0.0)
          trip->emplace_back(dofs[i], dofs[j], d[j]);
    }
  }
}
} // namespace

int
Problem::effectiveThreads() const
{
#ifdef _OPENMP
  if (!threadSafe())
    return 1;
  const int requested = _num_threads > 0 ? _num_threads : omp_get_max_threads();
  // Starting a team of threads costs a few microseconds, which is more than a
  // small mesh takes to assemble, so small problems are assembled serially.
  const Index elements = _mesh->numElements();
  const int affordable = static_cast<int>(elements / kMinElementsPerThread);
  return std::max(1, std::min(requested, affordable));
#else
  return 1;
#endif
}

void
Problem::assemble(const Vector & U, const AssemblyOptions & opts, Vector & R, SparseMatrix * J)
{
  initialize();
  if (_method == Method::FiniteVolumeCell)
  {
    assembleCellFiniteVolume(U, opts, R, J);
    return;
  }
  const Index ndof = numDofs();
  const int nv = numVariables();
  const int dim = _mesh->dimension();
  const bool dual = _method != Method::FiniteElement;
  // The vertex-centred finite volume method replaces the interpolated gradient
  // at a control-domain interface by the two-point difference along the edge
  // that joins the two nodes, keeping the interpolated part transverse to the
  // edge as a non-orthogonal correction:
  //
  //   grad u  <-  grad u + [ (U_N - U_O) - grad u . d ] d / |d|^2 ,
  //                                                       d = x_N - x_O .
  //
  // The correction vanishes for a linear field, so the patch test still holds;
  // on a mesh whose edges are parallel to the interface normals it reproduces
  // the half-control volume formulation of the finite volume method exactly.
  const bool edge_gradient = _method == Method::FiniteVolumeVertex;
  const bool want_jac = J != nullptr;
  R.setZero(ndof);
  const int nthreads = effectiveThreads();
  const bool atomic = nthreads > 1;
  std::vector<ThreadScratch> scratch(nthreads);
  for (auto & s : scratch)
  {
    s.ctx.dim = dim;
    s.ctx.time = _time;
    s.ctx.dt = opts.dt;
    s.ctx.load_factor = opts.load_factor;
    s.ctx.mode = opts.mode;
    if (want_jac)
      s.triplets.reserve(static_cast<std::size_t>(_mesh->numElements()) * 16 * nv * nv * 4 /
                         nthreads);
  }
  const bool use_lag = opts.mode == LinearizationMode::Picard && opts.lagged;

  // The local degrees of freedom of an element, in the compact layout of
  // localDofIndex: every variable at every node, except that a first-order
  // variable on a quadratic element has only its corner nodes.
  const bool mixed = _mixed_order;
  const auto gatherLocal = [&](ThreadScratch & s, const Element & el)
  {
    const int nn = el.numNodes();
    const int ld = numLocalDofs(el);
    s.cur.resize(ld);
    s.dofs.resize(ld);
    for (int k = 0; k < nn; ++k)
      for (int v = 0; v < nv; ++v)
      {
        const int i = localDofIndex(el, k, v);
        if (i < 0)
          continue;
        const Index d = dof(el.nodes[k], v);
        s.dofs[i] = d;
        s.cur[i] = U[d];
      }
    if (use_lag)
    {
      s.lag.resize(ld);
      for (int i = 0; i < ld; ++i)
        s.lag[i] = (*opts.lagged)[s.dofs[i]];
    }
    if (opts.old)
    {
      s.old.resize(ld);
      for (int i = 0; i < ld; ++i)
        s.old[i] = (*opts.old)[s.dofs[i]];
    }
    s.Rloc.assign(ld, ADReal(0.0));
    s.touched.assign(ld, 0);
    return ld;
  };

  const auto assembleElement = [&](ThreadScratch & s, Index e)
  {
    auto & cur = s.cur;
    auto & lag = s.lag;
    auto & old = s.old;
    auto & dofs = s.dofs;
    auto & Rloc = s.Rloc;
    auto & touched = s.touched;
    auto & pts = s.pts;
    // The mapped point of the current integration point: the one built with
    // the points when there is one (no copy), otherwise s.geo.
    MappedPoint * gp = &s.geo;
    auto & fld = s.fld;
    auto & ctx = s.ctx;
    const Element & el = _mesh->element(e);
    const int nn = el.numNodes();
    const int ld = gatherLocal(s, el);
    const int nd = want_jac ? ld : 0;
    ctx.element = e;
    ctx.block = el.block;
    ctx.on_boundary = false;
    bool any = false;

    for (const auto & group : _groups)
    {
      const unsigned group_index = static_cast<unsigned>(&group - _groups.data());
      unsigned point_set = 0;
      std::vector<const Kernel *> & ks = s.kernels;
      ks.clear();
      bool has_flux = false, has_source = false;
      for (const Kernel * k : group.kernels)
      {
        if (!k->activeOnBlock(el.block))
          continue;
        const auto & vb = _vars[k->variable()].blocks;
        if (!vb.empty() && !vb.count(el.block))
          continue;
        if (k->isTimeKernel() ? !opts.include_time_kernels : !opts.include_steady_terms)
          continue;
        if (k->isTimeKernel() && opts.dt <= 0)
          continue;
        ks.push_back(k);
        has_flux = has_flux || k->hasFlux();
        has_source = has_source || k->hasSource();
      }
      if (ks.empty())
        continue;
      any = true;

      const auto weightOf = [&](const Kernel * k)
      {
        double w = k->isTimeKernel() ? 1.0 : opts.theta;
        if (k->scalesWithLoad())
          w *= opts.load_factor;
        return w;
      };
      // The points were mapped when they were built; reuse those maps.
      const auto evaluateAt = [&](const IntegrationPoint & ip, std::size_t q)
      {
        if (q < s.mapped.size())
          gp = &s.mapped[q];
        else
        {
          mapPoint(*_mesh, el, ip.xi, s.geo);
          gp = &s.geo;
        }
        MappedPoint & geo = *gp;
        if (mixed)
          mapCornerShape(el, geo);
        const MappedPoint * fp = &geo;
        if (group.spec.reduced)
        {
          mapPoint(*_mesh, el, ip.xi_field, fld);
          if (mixed)
            mapCornerShape(el, fld);
          fp = &fld;
        }
        fillContext(ctx, el, *fp, cur, use_lag ? &lag : nullptr, opts.old ? &old : nullptr, nd);
        // A reduced rule evaluates the field at the element centroid, where the
        // interpolated gradient is the element's mean gradient; that is the
        // whole point of selective reduced integration (the penalty term of
        // incompressible flow locks without it), so the two-point edge
        // correction, which would replace the mean by an edge difference, is
        // not applied there.
        if (edge_gradient && !group.spec.reduced && ip.owner >= 0 && ip.neighbor >= 0)
        {
          const Point d = _mesh->node(el.nodes[ip.neighbor]) - _mesh->node(el.nodes[ip.owner]);
          const double dd = dot(d, d);
          if (dd > 0)
            for (int v = 0; v < nv; ++v)
            {
              const int io = ip.owner * nv + v;
              const int in = ip.neighbor * nv + v;
              ADReal du = ADReal::withZeroDerivatives(cur[in] - cur[io], nd);
              if (nd > 0)
              {
                du.setDerivative(in, 1.0);
                du.setDerivative(io, -1.0);
              }
              ADReal proj(0.0);
              for (int k = 0; k < dim; ++k)
                proj += ctx.grad_u[v][k] * d[k];
              const ADReal s = (du - proj) * (1.0 / dd);
              for (int k = 0; k < dim; ++k)
                ctx.grad_u[v][k] += s * d[k];
              if (use_lag)
              {
                double projl = 0;
                for (int k = 0; k < dim; ++k)
                  projl += ctx.grad_u_lag[v][k].value() * d[k];
                const double sl = (lag[in] - lag[io] - projl) / dd;
                for (int k = 0; k < dim; ++k)
                  ctx.grad_u_lag[v][k] += ADReal(sl * d[k]);
              }
              else
                ctx.grad_u_lag[v] = ctx.grad_u[v];
              if (opts.old)
              {
                double projo = 0;
                for (int k = 0; k < dim; ++k)
                  projo += ctx.grad_u_old[v][k] * d[k];
                const double so = (old[in] - old[io] - projo) / dd;
                for (int k = 0; k < dim; ++k)
                  ctx.grad_u_old[v][k] += so * d[k];
              }
              else
                for (int k = 0; k < 3; ++k)
                  ctx.grad_u_old[v][k] = ctx.grad_u[v][k].value();
            }
        }
        ctx.x = geo.x;
        ctx.state_domain = 0;
        ctx.state_owner = e;
        ctx.state_key = stateKey(point_set, group_index, 0, static_cast<unsigned>(q));
        computeMaterials(ctx);
        return coordFactor(geo.x);
      };

      // Volume integrals (DMCDM: control volumes; FEM: element interior).
      if (has_source || (!dual && has_flux))
      {
        buildElementPoints(*_mesh, e, dual, PointSet::Volume, group.spec, pts, &s.mapped);
        point_set = 0;
        for (std::size_t q = 0; q < pts.size(); ++q)
        {
          const IntegrationPoint & ip = pts[q];
          const double cf = evaluateAt(ip, q) * ip.weight;
          for (const Kernel * k : ks)
          {
            const int v = k->variable();
            const double w = cf * weightOf(k);
            // The test functions of the equation of variable v: the element's
            // shape functions, or its corners' for a first-order variable.
            const bool corner = mixed && usesCornerShape(el, v);
            const MappedPoint & geo = *gp;
            const double * testN = corner ? geo.corner_shape : geo.N;
            const Point * testdN = corner ? geo.corner_gradient : geo.dN;
            const int ntest = corner ? geo.num_corners : nn;
            const auto slot = [&](int a) { return mixed ? localDofIndex(el, a, v) : a * nv + v; };
            if (k->hasSource())
            {
              const ADReal S = k->computeSource(ctx);
              if (ip.owner >= 0)
              {
                Rloc[ip.owner * nv + v].addScaledBy(S, w);
                touched[ip.owner * nv + v] = 1;
              }
              else
                for (int a = 0; a < ntest; ++a)
                {
                  const int i = slot(a);
                  Rloc[i].addScaledBy(S, w * testN[a]);
                  touched[i] = 1;
                }
            }
            if (!dual && k->hasFlux() && ip.owner < 0)
            {
              ADVector3 F{ADReal(0.0), ADReal(0.0), ADReal(0.0)};
              k->computeFlux(ctx, F);
              for (int a = 0; a < ntest; ++a)
              {
                const int i = slot(a);
                ADReal & r = Rloc[i];
                for (int d = 0; d < dim; ++d)
                  if (testdN[a][d] != 0.0)
                    r.addScaledBy(F[d], w * testdN[a][d]);
                touched[i] = 1;
              }
            }
          }
        }
      }

      // Interfaces between control volumes (DMCDM only).
      if (dual && has_flux)
      {
        buildElementPoints(*_mesh, e, dual, PointSet::Faces, group.spec, pts, &s.mapped);
        point_set = 1;
        for (std::size_t q = 0; q < pts.size(); ++q)
        {
          const IntegrationPoint & ip = pts[q];
          const double cf = evaluateAt(ip, q);
          for (const Kernel * k : ks)
          {
            if (!k->hasFlux())
              continue;
            const int v = k->variable();
            ADVector3 F{ADReal(0.0), ADReal(0.0), ADReal(0.0)};
            k->computeFlux(ctx, F);
            // -\oint F.n over the boundary of CD_a; the same interface
            // enters CD_b with the opposite normal.
            const double scale = cf * weightOf(k);
            ADReal & ro = Rloc[ip.owner * nv + v];
            ADReal & rn = Rloc[ip.neighbor * nv + v];
            for (int d = 0; d < dim; ++d)
              if (ip.area[d] != 0.0)
              {
                ro.addScaledBy(F[d], -scale * ip.area[d]);
                rn.addScaledBy(F[d], scale * ip.area[d]);
              }
            touched[ip.owner * nv + v] = 1;
            touched[ip.neighbor * nv + v] = 1;
          }
        }
      }
    }
    if (any)
      scatter(Rloc, dofs, touched, R, want_jac ? &s.triplets : nullptr, atomic);
  };

  // The element loop is the expensive part and it is embarrassingly parallel:
  // every thread keeps its own scratch space and its own list of matrix
  // entries, and the only shared write is the atomic update of the residual.
  // Threading is disabled automatically when any object is defined in Python,
  // because calling back into the interpreter needs the global interpreter
  // lock (see Problem::initialize).
  std::string thread_error;
#ifdef _OPENMP
#pragma omp parallel num_threads(nthreads) if (nthreads > 1)
  {
    ThreadScratch & s = scratch[omp_get_thread_num()];
#pragma omp for schedule(static)
    for (Index e = 0; e < _mesh->numElements(); ++e)
    {
      try
      {
        assembleElement(s, e);
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
  for (Index e = 0; e < _mesh->numElements(); ++e)
    assembleElement(scratch[0], e);
#endif
  if (!thread_error.empty())
    throw std::runtime_error(thread_error);

  // The boundary terms are a small fraction of the work and are assembled
  // serially, in the first thread's scratch space.
  ThreadScratch & s0 = scratch[0];
  auto & cur = s0.cur;
  auto & lag = s0.lag;
  auto & old = s0.old;
  auto & dofs = s0.dofs;
  auto & Rloc = s0.Rloc;
  auto & touched = s0.touched;
  auto & pts = s0.pts;
  auto & geo = s0.geo;
  auto & fld = s0.fld;
  auto & ctx = s0.ctx;

  // ---- integrated boundary conditions ------------------------------------
  if (opts.include_steady_terms)
    for (const auto & bc : _ibcs)
    {
      const int v = bc->variable();
      double wfac = opts.theta * (bc->scalesWithLoad() ? opts.load_factor : 1.0);
      if (const auto * ib = dynamic_cast<const InterfaceBC *>(bc.get()))
      {
        assembleInterface(*ib, U, opts, wfac, R, want_jac ? &s0.triplets : nullptr, s0);
        continue;
      }
      for (const auto & bname : bc->boundaries())
        for (const Side & side : _mesh->sideset(bname))
        {
          const Element & el = _mesh->element(side.first);
          if (!bc->activeOnBlock(el.block))
            continue;
          const int nn = el.numNodes();
          const int ld = gatherLocal(s0, el);
          const int nd = want_jac ? ld : 0;
          ctx.element = side.first;
          ctx.block = el.block;
          ctx.on_boundary = true;
          buildSidePoints(*_mesh, side, dual, bc->quadrature(), pts);
          const bool corner = usesCornerShape(el, v);
          for (const auto & ip : pts)
          {
            mapPoint(*_mesh, el, ip.xi, geo);
            if (mixed)
              mapCornerShape(el, geo);
            const MappedPoint * fp = &geo;
            if (bc->quadrature().reduced)
            {
              mapPoint(*_mesh, el, ip.xi_field, fld);
              if (mixed)
                mapCornerShape(el, fld);
              fp = &fld;
            }
            fillContext(ctx, el, *fp, cur, use_lag ? &lag : nullptr, opts.old ? &old : nullptr, nd);
            ctx.x = geo.x;
            ctx.normal = (1.0 / ip.weight) * ip.area;
            ctx.state_domain = 0;
            ctx.state_owner = side.first;
            ctx.state_key = stateKey(2,
                                     static_cast<unsigned>(&bc - _ibcs.data()),
                                     static_cast<unsigned>(side.second),
                                     static_cast<unsigned>(&ip - pts.data()));
            computeMaterials(ctx);
            const double w = ip.weight * coordFactor(geo.x) * wfac;
            const ADReal q = bc->computeBoundaryFlux(ctx) * w;
            if (ip.owner >= 0)
            {
              Rloc[ip.owner * nv + v] -= q;
              touched[ip.owner * nv + v] = 1;
            }
            else
            {
              const double * testN = corner ? geo.corner_shape : geo.N;
              const int ntest = corner ? geo.num_corners : nn;
              for (int a = 0; a < ntest; ++a)
              {
                const int i = localDofIndex(el, a, v);
                Rloc[i] -= q * testN[a];
                touched[i] = 1;
              }
            }
          }
          scatter(Rloc, dofs, touched, R, want_jac ? &s0.triplets : nullptr, false);
        }
    }

  // ---- concentrated nodal loads ----------------------------------------------
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
    // Merge the per-thread lists of matrix entries.  setFromTriplets sums
    // duplicates, so the order of the merge does not change the result beyond
    // floating-point round-off.
    std::vector<Triplet> trip;
    std::size_t total = 0;
    for (const auto & s : scratch)
      total += s.triplets.size();
    trip.reserve(total);
    for (const auto & s : scratch)
      trip.insert(trip.end(), s.triplets.begin(), s.triplets.end());
    J->resize(ndof, ndof);
    J->setFromTriplets(trip.begin(), trip.end());
  }
}

void
Problem::assembleInterface(const InterfaceBC & bc,
                           const Vector & U,
                           const AssemblyOptions & opts,
                           double wfac,
                           Vector & R,
                           std::vector<Eigen::Triplet<double>> * triplets,
                           ThreadScratch & s)
{
  // Local degrees of freedom: the coupled variables at the nodes of the
  // primary side, then at the nodes of the paired secondary side.  Only the
  // side nodes are included, because the shape functions of every other node
  // vanish on a side, and at an interface point only values are needed.
  const int nv = numVariables();
  const int dim = _mesh->dimension();
  const bool dual = _method != Method::FiniteElement;
  const bool want_jac = triplets != nullptr;
  const auto & coupled = bc.coupledVariables();
  const int nc = static_cast<int>(coupled.size());
  const auto slot = [&](int v)
  { return static_cast<int>(std::find(coupled.begin(), coupled.end(), v) - coupled.begin()); };
  const int cp = slot(bc.variable());
  const int cs = slot(bc.secondaryVariable());
  const auto & pairs = _interface_pairs.at(&bc);
  unsigned bc_index = 0;
  for (std::size_t i = 0; i < _ibcs.size(); ++i)
    if (_ibcs[i].get() == &bc)
      bc_index = static_cast<unsigned>(i);
  auto & ctx = s.ctx;
  auto & pts = s.pts;
  auto & geo = s.geo;
  MappedPoint sgeo;
  std::size_t side_index = 0;
  for (const auto & bname : bc.boundaries())
    for (const Side & side : _mesh->sideset(bname))
    {
      const auto & side_pairs = pairs[side_index++];
      const Element & el = _mesh->element(side.first);
      if (!bc.activeOnBlock(el.block))
        continue;
      const auto & sn = ReferenceElement::get(el.type).sideNodes(side.second);
      const int np = static_cast<int>(sn.size());
      buildSidePoints(*_mesh, side, dual, bc.quadrature(), pts);
      for (std::size_t q = 0; q < pts.size(); ++q)
      {
        const IntegrationPoint & ip = pts[q];
        const auto & pair = side_pairs[q];
        const Element & sel = _mesh->element(pair.element);
        const auto & ssn = ReferenceElement::get(sel.type).sideNodes(pair.side);
        const int ns = static_cast<int>(ssn.size());
        const int ld = (np + ns) * nc;
        if (ld > kMaxDerivatives)
          throw InputError("'" + bc.name() + "': an interface point couples " + std::to_string(ld) +
                           " degrees of freedom, more than the " + std::to_string(kMaxDerivatives) +
                           " the automatic differentiation provides. Rebuild with a larger "
                           "DUALMESH_MAX_AD_DERIVATIVES.");
        const int nd = want_jac ? ld : 0;
        mapPoint(*_mesh, el, ip.xi, geo);
        mapPoint(*_mesh, sel, pair.xi, sgeo);
        ctx.u.resize(nv);
        ctx.grad_u.resize(nv);
        ctx.u_other.resize(nv);
        ctx.u_old.assign(nv, 0.0);
        ctx.grad_u_old.assign(nv, Point{0, 0, 0});
        for (int v = 0; v < nv; ++v)
        {
          double val = 0, other = 0, old_val = 0;
          Point g{0, 0, 0};
          for (int a = 0; a < el.numNodes(); ++a)
          {
            const Index d = dof(el.nodes[a], v);
            val += geo.N[a] * U[d];
            if (opts.old)
              old_val += geo.N[a] * (*opts.old)[d];
            for (int k = 0; k < dim; ++k)
              g[k] += geo.dN[a][k] * U[d];
          }
          for (int b = 0; b < sel.numNodes(); ++b)
            other += sgeo.N[b] * U[dof(sel.nodes[b], v)];
          ctx.u[v] = ADReal::withZeroDerivatives(val, nd);
          ctx.u_other[v] = ADReal::withZeroDerivatives(other, nd);
          ctx.grad_u[v] = {ADReal(g[0]), ADReal(g[1]), ADReal(g[2])};
          ctx.u_old[v] = opts.old ? old_val : val;
          ctx.grad_u_old[v] = g;
        }
        if (nd > 0)
          for (int c = 0; c < nc; ++c)
          {
            const int v = coupled[c];
            for (int j = 0; j < np; ++j)
              ctx.u[v].setDerivative(j * nc + c, geo.N[sn[j]]);
            for (int j = 0; j < ns; ++j)
              ctx.u_other[v].setDerivative((np + j) * nc + c, sgeo.N[ssn[j]]);
          }
        ctx.u_lag = ctx.u;
        ctx.grad_u_lag = ctx.grad_u;
        ctx.element = side.first;
        ctx.block = el.block;
        ctx.on_boundary = true;
        ctx.x = geo.x;
        ctx.x_other = sgeo.x;
        ctx.normal = (1.0 / ip.weight) * ip.area;
        ctx.state_domain = 0;
        ctx.state_owner = side.first;
        ctx.state_key =
            stateKey(3, bc_index, static_cast<unsigned>(side.second), static_cast<unsigned>(q));
        computeMaterials(ctx);
        const double w = ip.weight * coordFactor(geo.x) * wfac;
        const ADReal flux = bc.computeInterfaceFlux(ctx) * w;

        s.dofs.resize(ld);
        for (int c = 0; c < nc; ++c)
        {
          for (int j = 0; j < np; ++j)
            s.dofs[j * nc + c] = dof(el.nodes[sn[j]], coupled[c]);
          for (int j = 0; j < ns; ++j)
            s.dofs[(np + j) * nc + c] = dof(sel.nodes[ssn[j]], coupled[c]);
        }
        s.Rloc.assign(ld, ADReal(0.0));
        s.touched.assign(ld, 0);
        // The primary body receives +flux through its test functions (or
        // the control domain that owns the point), the secondary body -flux
        // at the paired point.
        if (ip.owner >= 0)
        {
          const int j = static_cast<int>(std::find(sn.begin(), sn.end(), ip.owner) - sn.begin());
          s.Rloc[j * nc + cp] -= flux;
          s.touched[j * nc + cp] = 1;
        }
        else
          for (int j = 0; j < np; ++j)
          {
            s.Rloc[j * nc + cp].addScaledBy(flux, -geo.N[sn[j]]);
            s.touched[j * nc + cp] = 1;
          }
        if (dual)
        {
          const int j =
              static_cast<int>(std::find(ssn.begin(), ssn.end(), pair.owner) - ssn.begin());
          s.Rloc[(np + j) * nc + cs] += flux;
          s.touched[(np + j) * nc + cs] = 1;
        }
        else
          for (int j = 0; j < ns; ++j)
          {
            s.Rloc[(np + j) * nc + cs].addScaledBy(flux, sgeo.N[ssn[j]]);
            s.touched[(np + j) * nc + cs] = 1;
          }
        scatter(s.Rloc, s.dofs, s.touched, R, triplets, false);
      }
    }
}

void
Problem::buildInterfacePairs()
{
  _interface_pairs.clear();
  const bool cell = _method == Method::FiniteVolumeCell;
  const bool dual = _method != Method::FiniteElement;
  std::vector<IntegrationPoint> pts;
  MappedPoint mp;
  for (const auto & bc : _ibcs)
  {
    const auto * ib = dynamic_cast<const InterfaceBC *>(bc.get());
    if (!ib)
      continue;
    // The candidate sides of the other body, with their centroids for a
    // quick first selection.
    std::vector<Side> sides;
    std::vector<Index> faces;
    for (const auto & name : ib->secondaryBoundaries())
    {
      const auto & set = _mesh->sideset(name);
      sides.insert(sides.end(), set.begin(), set.end());
      if (cell)
        for (int fi : cellMesh().boundaryFaceIndices(*_mesh, name))
          faces.push_back(fi);
    }
    if (sides.empty())
      throw InputError("'" + ib->name() + "': the secondary side sets are empty.");
    std::vector<Point> centroids;
    for (const Side & sd : sides)
      centroids.push_back(_mesh->sideCentroid(sd));
    const auto pair = [&](const Point & x)
    {
      const std::size_t candidates = std::min<std::size_t>(8, sides.size());
      std::vector<std::pair<double, std::size_t>> order(sides.size());
      for (std::size_t i = 0; i < sides.size(); ++i)
        order[i] = {norm(centroids[i] - x), i};
      std::partial_sort(order.begin(), order.begin() + candidates, order.end());
      InterfacePoint best;
      double best_distance = std::numeric_limits<double>::infinity();
      for (std::size_t k = 0; k < candidates; ++k)
      {
        const std::size_t i = order[k].second;
        const SideProjection p = projectOntoSide(*_mesh, sides[i], x);
        if (p.distance < best_distance - 1e-14)
        {
          best_distance = p.distance;
          best.element = sides[i].first;
          best.side = sides[i].second;
          best.xi = p.xi;
          best.x = p.x;
          best.face = cell ? faces[i] : -1;
        }
      }
      if (dual && !cell)
        best.owner = sideControlOwner(*_mesh, {best.element, best.side}, best.xi);
      return best;
    };
    auto & out = _interface_pairs[bc.get()];
    for (const auto & bname : ib->boundaries())
    {
      if (cell)
      {
        for (int fi : cellMesh().boundaryFaceIndices(*_mesh, bname))
          out.push_back({pair(cellMesh().faces()[fi].centroid)});
        continue;
      }
      for (const Side & side : _mesh->sideset(bname))
      {
        const Element & el = _mesh->element(side.first);
        buildSidePoints(*_mesh, side, dual, ib->quadrature(), pts);
        std::vector<InterfacePoint> row;
        for (const auto & ip : pts)
        {
          mapPoint(*_mesh, el, ip.xi, mp);
          row.push_back(pair(mp.x));
        }
        out.push_back(std::move(row));
      }
    }
  }
}

void
Problem::applyDirichlet(Vector & U, double load_factor) const
{
  for (const auto & bc : _nbcs)
  {
    const int v = bc->variable();
    const double f = bc->scalesWithLoad() ? load_factor : 1.0;
    for (Index n : bc->nodes())
      U[dof(n, v)] = f * bc->computeValue(entityPoint(n), _time);
  }
  // A condition on a first-order variable also wrote the unused slots of the
  // non-corner boundary nodes; restore the linear interpolant there.
  interpolateFirstOrderVariables(U);
}

std::vector<char>
Problem::constrainedDofs() const
{
  std::vector<char> fixed(numDofs(), 0);
  for (const auto & bc : _nbcs)
    for (Index n : bc->nodes())
      fixed[dof(n, bc->variable())] = 1;
  for (Index i = 0; i < numDofs(); ++i)
    if (!_active_dof[i])
      fixed[i] = 1;
  return fixed;
}

void
Problem::dirichletRows(const Vector & /*U*/, double /*lf*/, Vector & R, SparseMatrix * J) const
{
  const std::vector<char> fixed = constrainedDofs();
  for (Index i = 0; i < numDofs(); ++i)
    if (fixed[i])
      R[i] = 0.0;
  if (J)
  {
    // Both the rows and the columns of the prescribed degrees of freedom are
    // removed.  The right-hand side does not change, because the solution
    // already satisfies the prescribed values and the increment there is zero;
    // dropping the columns keeps a symmetric problem symmetric, which the
    // conjugate gradient solver needs.
    J->prune([&](const Index & row, const Index & col, const double &)
             { return !fixed[row] && !fixed[col]; });
    std::vector<Triplet> diag;
    for (Index i = 0; i < numDofs(); ++i)
      if (fixed[i])
        diag.emplace_back(i, i, 1.0);
    SparseMatrix D(numDofs(), numDofs());
    D.setFromTriplets(diag.begin(), diag.end());
    *J += D;
  }
}

namespace
{
/// Solve with a Krylov method; false when it did not converge.
template <typename Solver>
bool
krylov(Solver & solver,
       const SparseMatrix & A,
       const Vector & b,
       const SolverOptions & o,
       int max_iterations,
       int * iterations,
       Vector & x)
{
  solver.setTolerance(o.linear_tolerance);
  solver.setMaxIterations(max_iterations);
  solver.compute(A);
  if (solver.info() != Eigen::Success)
    return false;
  x = solver.solve(b);
  if (iterations)
    *iterations += static_cast<int>(solver.iterations());
  return solver.info() == Eigen::Success && x.allFinite();
}

template <typename Preconditioner>
bool
krylovWith(const std::string & method,
           const SparseMatrix & A,
           const Vector & b,
           const SolverOptions & o,
           int max_iterations,
           int * iterations,
           Vector & x)
{
  if (method == "gmres")
  {
    Eigen::GMRES<SparseMatrix, Preconditioner> solver;
    solver.set_restart(o.gmres_restart);
    return krylov(solver, A, b, o, max_iterations, iterations, x);
  }
  Eigen::BiCGSTAB<SparseMatrix, Preconditioner> solver;
  return krylov(solver, A, b, o, max_iterations, iterations, x);
}

bool
preconditionedKrylov(const std::string & method,
                     const SparseMatrix & A,
                     const Vector & b,
                     const SolverOptions & o,
                     int max_iterations,
                     int * iterations,
                     Vector & x)
{
  if (o.preconditioner == "ilu")
  {
    // A replaced pivot means a zero on the diagonal, as in the pressure block
    // of a mixed formulation; ILU(0) is then a poor preconditioner and the
    // caller is better served by the direct solver, so this is a failure.
    if (method == "gmres")
    {
      Eigen::GMRES<SparseMatrix, IncompleteLU0> solver;
      solver.set_restart(o.gmres_restart);
      solver.preconditioner().compute(A);
      if (solver.preconditioner().info() != Eigen::Success ||
          solver.preconditioner().replacedPivots() > 0)
        return false;
      return krylov(solver, A, b, o, max_iterations, iterations, x);
    }
    Eigen::BiCGSTAB<SparseMatrix, IncompleteLU0> solver;
    solver.preconditioner().compute(A);
    if (solver.preconditioner().info() != Eigen::Success ||
        solver.preconditioner().replacedPivots() > 0)
      return false;
    return krylov(solver, A, b, o, max_iterations, iterations, x);
  }
  if (o.preconditioner == "ilut")
  {
    if (method == "gmres")
    {
      Eigen::GMRES<SparseMatrix, Eigen::IncompleteLUT<double>> solver;
      solver.set_restart(o.gmres_restart);
      solver.preconditioner().setDroptol(1e-4);
      solver.preconditioner().setFillfactor(10);
      return krylov(solver, A, b, o, max_iterations, iterations, x);
    }
    Eigen::BiCGSTAB<SparseMatrix, Eigen::IncompleteLUT<double>> solver;
    solver.preconditioner().setDroptol(1e-4);
    solver.preconditioner().setFillfactor(10);
    return krylov(solver, A, b, o, max_iterations, iterations, x);
  }
  if (o.preconditioner == "jacobi")
    return krylovWith<Eigen::DiagonalPreconditioner<double>>(
        method, A, b, o, max_iterations, iterations, x);
  if (o.preconditioner == "none")
    return krylovWith<Eigen::IdentityPreconditioner>(
        method, A, b, o, max_iterations, iterations, x);
  throw InputError("Unknown preconditioner '" + o.preconditioner +
                   "' (use ilu, ilut, jacobi, or none).");
}

Vector
directSolve(const SparseMatrix & A, const Vector & b)
{
  Eigen::SparseLU<SparseMatrix, Eigen::COLAMDOrdering<int>> lu;
  lu.analyzePattern(A);
  lu.factorize(A);
  if (lu.info() != Eigen::Success)
  {
    // A NaN or infinite entry also stops the factorization, and Eigen then
    // reports a "structurally singular" matrix.  Say which it is.
    bool finite = true;
    for (int k = 0; k < A.outerSize() && finite; ++k)
      for (SparseMatrix::InnerIterator it(A, k); it; ++it)
        if (!std::isfinite(it.value()))
        {
          finite = false;
          break;
        }
    if (!finite)
      throw std::runtime_error(
          "dualmesh: the Jacobian contains NaN or infinite entries, so it cannot be factorized. "
          "A material property or its derivative is not finite at the current solution "
          "(for example log(0), 1/0, or x^p with p < 1 at x = 0).");
    throw std::runtime_error(
        "dualmesh: sparse LU factorization failed (singular system?): " + lu.lastErrorMessage() +
        ". Check that every variable has enough boundary conditions.");
  }
  Vector x = lu.solve(b);
  if (lu.info() != Eigen::Success || !x.allFinite())
    throw std::runtime_error("dualmesh: sparse LU solve failed.");
  return x;
}
} // namespace

SaddlePointBlocks
Problem::saddlePointBlocks(const SparseMatrix & J) const
{
  const_cast<Problem *>(this)->initialize();
  if (_cells)
    throw InputError("The pressure mass matrix preconditioner is not available for the "
                     "cell-centred finite volume method yet; use it with fem, dmcdm or hfvm.");
  // The constraint equation: the one kernel that provides a Schur mass
  // coefficient.  A probe context at the first node asks each kernel.
  QpContext probe;
  probe.dim = _mesh->dimension();
  probe.time = _time;
  probe.x = _mesh->node(0);
  const Kernel * constraint = nullptr;
  for (const auto & k : _kernels)
  {
    double c = 0.0;
    if (k->schurMassCoefficient(probe, c))
    {
      if (constraint && constraint->variable() != k->variable())
        throw InputError("The pressure mass matrix preconditioner found two constraint "
                         "equations ('" +
                         constraint->name() + "' and '" + k->name() +
                         "') on different variables; it handles one pressure.");
      constraint = k.get();
    }
  }
  if (!constraint)
    throw InputError("The preconditioner 'pressure_mass_schur' is for pressure-velocity flow, "
                     "and this problem has no mass conservation equation (mass_conservation). "
                     "Use another preconditioner.");
  const int p = constraint->variable();
  const int nv = numVariables();
  SaddlePointBlocks blocks;
  blocks.pressure_variable = _vars[p].name;
  std::vector<Index> position(numDofs(), -1);
  for (Index d = 0; d < numDofs(); ++d)
    if (d % nv == p)
    {
      position[d] = static_cast<Index>(blocks.pressure_dofs.size());
      blocks.pressure_dofs.push_back(d);
    }
    else
      blocks.other_dofs.push_back(d);

  // The sign of the Schur complement C - B A^{-1} B^T: with the gradient block
  // G = J(u, p) and the divergence block D = J(p, u), the Frobenius product
  // <G, D^T> is negative when G = -D^T, and then -D A^{-1} G = D A^{-1} D^T is
  // positive definite.
  double product = 0.0;
  for (Index d : blocks.pressure_dofs)
    for (SparseMatrix::InnerIterator it(J, d); it; ++it)
      if (position[it.row()] < 0)
        product += it.value() * J.coeff(d, it.row());
  if (product == 0.0)
    throw InputError("The pressure '" + blocks.pressure_variable +
                     "' is not coupled to the other variables in this system; the pressure mass "
                     "matrix preconditioner does not apply.");
  blocks.sign = product < 0.0 ? 1 : -1;

  // The mass matrix (c q, psi) with the pressure's own shape functions and a
  // Gauss rule one order above the element's.
  const std::vector<char> constrained = constrainedDofs();
  std::vector<Triplet> entries;
  std::vector<IntegrationPoint> points;
  std::vector<MappedPoint> mapped;
  MappedPoint geo;
  QpContext context = probe;
  for (Index e = 0; e < _mesh->numElements(); ++e)
  {
    const Element & el = _mesh->element(e);
    if (!constraint->activeOnBlock(el.block))
      continue;
    const auto & vb = _vars[p].blocks;
    if (!vb.empty() && !vb.count(el.block))
      continue;
    QuadratureSpec spec;
    spec.kind = QuadratureSpec::Kind::Gauss;
    spec.points = ReferenceElement::get(el.type).order() + 2;
    buildElementPoints(*_mesh, e, false, PointSet::Volume, spec, points, &mapped);
    const bool corner = usesCornerShape(el, p);
    for (std::size_t q = 0; q < points.size(); ++q)
    {
      if (q < mapped.size())
        geo = mapped[q];
      else
        mapPoint(*_mesh, el, points[q].xi, geo);
      if (corner)
        mapCornerShape(el, geo);
      const double * N = corner ? geo.corner_shape : geo.N;
      const int count = corner ? geo.num_corners : el.numNodes();
      context.x = geo.x;
      context.element = e;
      context.block = el.block;
      double c = 0.0;
      constraint->schurMassCoefficient(context, c);
      const double w = points[q].weight * coordFactor(geo.x) * c * blocks.sign;
      for (int a = 0; a < count; ++a)
      {
        const Index da = dof(el.nodes[a], p);
        if (constrained[da])
          continue;
        for (int b = 0; b < count; ++b)
        {
          const Index db = dof(el.nodes[b], p);
          if (!constrained[db])
            entries.emplace_back(position[da], position[db], w * N[a] * N[b]);
        }
      }
    }
  }
  // The pressure block C of the system itself belongs to the Schur complement
  // C - B A^{-1} B^T: zero for the Taylor-Hood element, the pressure Laplacian
  // of the stabilisation for an equal-order element.  Its symmetric part is
  // added, which keeps the approximation symmetric.
  for (Index d : blocks.pressure_dofs)
  {
    if (constrained[d])
      continue;
    for (SparseMatrix::InnerIterator it(J, d); it; ++it)
    {
      const Index r = it.row();
      if (position[r] >= 0 && !constrained[r] && it.value() != 0.0)
      {
        entries.emplace_back(position[r], position[d], 0.5 * it.value());
        entries.emplace_back(position[d], position[r], 0.5 * it.value());
      }
    }
  }
  // Prescribed and inactive pressure unknowns: their equations are rows of the
  // identity, whose Schur complement is exactly 1.
  for (Index d : blocks.pressure_dofs)
    if (constrained[d])
      entries.emplace_back(position[d], position[d], 1.0);
  const Index np = static_cast<Index>(blocks.pressure_dofs.size());
  blocks.schur_approximation.resize(np, np);
  blocks.schur_approximation.setFromTriplets(entries.begin(), entries.end());
  blocks.schur_approximation.makeCompressed();
  return blocks;
}

bool
Problem::preferDirectSolver(Index n) const
{
  // A direct factorisation of a sparse matrix from a mesh fills in: with a
  // good ordering the factors of a two-dimensional problem hold O(n log n)
  // entries, but those of a three-dimensional problem hold O(n^{4/3}) and
  // cost O(n^2) operations (George, SIAM J. Numer. Anal. 10 (1973) 345-363;
  // Lipton, Rose and Tarjan, SIAM J. Numer. Anal. 16 (1979) 346-358).  A
  // preconditioned Krylov iteration costs a few matrix-vector products per
  // iteration.  So the direct solver is kept where it is cheap, which is
  // everywhere in one dimension, up to a large size in two and only for small
  // problems in three; the limits were measured on this library's own
  // systems.
  const int d = _mesh->dimension();
  if (d <= 1)
    return true;
  if (d == 2)
    return n <= 100000;
  return n <= 4000;
}

Vector
Problem::linearSolve(const SparseMatrix & A_in,
                     const Vector & b,
                     const SolverOptions & o,
                     int * iterations) const
{
  SparseMatrix A = A_in;
  A.makeCompressed();
  const std::string & method = o.linear_solver;
  if (o.preconditioner == "pressure_mass_schur")
  {
    const SaddlePointBlocks blocks = saddlePointBlocks(A);
    if (method == "petsc")
    {
      petsc::Settings settings;
      settings.options = o.petsc_options;
      settings.relative_tolerance = o.linear_tolerance;
      settings.max_iterations = o.linear_max_iterations;
      settings.saddle_point = &blocks;
      petsc::Result result;
      Vector x = petsc::solve(A, b, settings, result);
      if (iterations)
        *iterations += result.iterations;
      if (!result.converged)
        throw std::runtime_error("dualmesh: the PETSc Schur complement solve did not converge (" +
                                 result.reason + ", " + std::to_string(result.iterations) +
                                 " iterations).");
      return x;
    }
    if (method != "gmres" && method != "automatic")
      throw InputError("The preconditioner 'pressure_mass_schur' works with linear_solver "
                       "'gmres' or 'automatic' (the built-in flexible GMRES) and 'petsc'; it "
                       "was given with '" +
                       method + "'.");
    SaddlePointSolver solver(A, blocks);
    Vector x = Vector::Zero(b.size());
    const auto result =
        solver.solve(b, x, o.linear_tolerance, o.linear_max_iterations, o.gmres_restart);
    if (iterations)
      *iterations += result.iterations;
    if (!result.converged)
      throw std::runtime_error(
          "dualmesh: flexible GMRES with the pressure mass matrix preconditioner did not "
          "converge in " +
          std::to_string(result.iterations) + " iterations (relative residual " +
          std::to_string(result.relative_residual) +
          "). A flow with strong inertia needs a better Schur complement approximation than the "
          "mass matrix; try linear_solver='lu'.");
    return x;
  }
  if (method == "lu")
    return directSolve(A, b);
  if (method == "automatic")
  {
    // A mixed-order (Taylor-Hood) flow problem is a saddle point problem whose
    // pressure block is exactly zero, so an incomplete factorisation meets a
    // zero pivot at once; factorise it directly.  Large ones are better
    // served by PETSc with a Schur complement field split.
    if (preferDirectSolver(A.rows()))
      return directSolve(A, b);
    if (_mixed_order)
    {
      // A large Taylor-Hood system: flexible GMRES with the pressure mass
      // matrix Schur preconditioner, whose iteration count does not grow with
      // the mesh (measured: 13 per Newton step for Stokes flow on 8 x 8 to
      // 32 x 32 Quad9 meshes), and which costs one factorisation of the
      // momentum block instead of one of the whole system.
      SolverOptions schur = o;
      schur.preconditioner = "pressure_mass_schur";
      schur.linear_solver = "gmres";
      if (o.verbose)
        std::cout << "  automatic linear solver: flexible GMRES with the pressure mass matrix "
                     "Schur preconditioner ("
                  << A.rows() << " unknowns)\n";
      return linearSolve(A, b, schur, iterations);
    }
    // The iteration is given a bounded budget and the direct solver is the
    // fallback, so that "automatic" is never less robust than "lu": a system
    // on which ILU(0) is a poor preconditioner (a saddle point problem, for
    // instance) costs a failed attempt, not a failed solve.
    Vector x;
    const int budget = std::min(o.linear_max_iterations, 1000);
    if (preconditionedKrylov("bicgstab", A, b, o, budget, iterations, x))
      return x;
    if (o.verbose)
      std::cout << "  the preconditioned iteration did not converge; using the direct solver\n";
    return directSolve(A, b);
  }
  if (method == "bicgstab" || method == "gmres")
  {
    Vector x;
    if (!preconditionedKrylov(method, A, b, o, o.linear_max_iterations, iterations, x))
      throw std::runtime_error("dualmesh: " + method + " with the " + o.preconditioner +
                               " preconditioner did not converge in " +
                               std::to_string(o.linear_max_iterations) +
                               " iterations. Try linear_solver='lu', or another preconditioner.");
    return x;
  }
  if (method == "petsc")
  {
    petsc::Settings settings;
    settings.options = o.petsc_options;
    settings.relative_tolerance = o.linear_tolerance;
    settings.max_iterations = o.linear_max_iterations;
    settings.block_size = numVariables();
    petsc::Result result;
    Vector x = petsc::solve(A, b, settings, result);
    if (iterations)
      *iterations += result.iterations;
    if (!result.converged)
      throw std::runtime_error("dualmesh: the PETSc solve did not converge (" + result.reason +
                               ", " + std::to_string(result.iterations) +
                               " iterations). Adjust petsc_options, for example a direct "
                               "solver: '-ksp_type preonly -pc_type lu'.");
    return x;
  }
  if (method == "cg")
  {
    Eigen::ConjugateGradient<SparseMatrix, Eigen::Lower | Eigen::Upper> solver;
    Vector x;
    if (!krylov(solver, A, b, o, o.linear_max_iterations, iterations, x))
      throw std::runtime_error(
          "dualmesh: the conjugate gradient method did not converge. It requires a symmetric "
          "positive definite matrix, which the Galerkin finite element method gives but the "
          "dual mesh and finite volume methods do not; use linear_solver = 'bicgstab' or 'lu' "
          "with those.");
    return x;
  }
  throw InputError("Unknown linear solver '" + o.linear_solver +
                   "' (use automatic, lu, bicgstab, gmres, cg, or petsc).");
}

SolveResult
Problem::nonlinearSolve(const SolverOptions & o, AssemblyOptions base, const Vector * steady_old)
{
  if (o.nonlinear_solver != "newton" && o.nonlinear_solver != "picard" &&
      o.nonlinear_solver != "linear")
    throw InputError("Unknown nonlinear solver '" + o.nonlinear_solver +
                     "' (use newton, picard, or linear).");
  const bool picard = o.nonlinear_solver == "picard";
  base.mode = picard ? LinearizationMode::Picard : LinearizationMode::Newton;
  std::vector<double> factors = o.load_factors;
  if (factors.empty())
    factors = {1.0};

  SolveResult result;
  Vector R;
  SparseMatrix J;
  int step_index = 0;
  for (double lf : factors)
  {
    ++step_index;
    base.load_factor = lf;
    applyDirichlet(_U, lf);
    if (o.verbose)
      console::nonlinearHeader(o.nonlinear_solver == "newton"
                                   ? "Newton"
                                   : (picard ? "Picard iteration" : "linear solve"),
                               step_index,
                               static_cast<int>(factors.size()),
                               lf);
    int linear_before = result.linear_iterations;
    double r0 = -1;
    bool converged = false;
    bool have_residual_at_U = false;
    for (int it = 1; it <= o.max_iterations + 1; ++it)
    {
      Vector lagU = _U;
      base.lagged = &lagU;
      assemble(_U, base, R, &J);
      if (steady_old)
        R += *steady_old;
      _last_residual = R;
      dirichletRows(_U, lf, R, &J);
      const double rn = R.norm();
      have_residual_at_U = true;
      if (r0 < 0)
        r0 = rn;
      IterationRecord rec{step_index, lf, it, rn, 0.0};
      if (rn <= o.absolute_tolerance || (it > 1 && rn <= o.relative_tolerance * r0) ||
          (o.nonlinear_solver == "linear" && it > 1))
      {
        converged = true;
        result.history.push_back(rec);
        if (o.verbose)
          console::nonlinearIteration(it, rn, r0, true);
        break;
      }
      if (it > o.max_iterations)
      {
        result.history.push_back(rec);
        break;
      }
      Vector delta = linearSolve(J, -R, o, &result.linear_iterations);
      if (o.line_search != "none" && o.line_search != "backtracking")
        throw InputError("Unknown line_search '" + o.line_search + "' (use none or backtracking).");
      if (o.line_search == "backtracking" && o.nonlinear_solver == "newton")
      {
        // Backtracking on the residual norm: accept the first step length
        // with ||R(U + a dU)|| <= (1 - 1e-4 a) ||R(U)||, or the best tried.
        double alpha = 1.0, best_alpha = 1.0, best = std::numeric_limits<double>::infinity();
        Vector Rt;
        for (int ls = 0; ls <= o.max_line_search_steps; ++ls)
        {
          Vector trial = _U + alpha * delta;
          interpolateFirstOrderVariables(trial);
          double rt = std::numeric_limits<double>::infinity();
          try
          {
            Vector lagT = trial;
            AssemblyOptions ao = base;
            ao.lagged = &lagT;
            assemble(trial, ao, Rt, nullptr);
            if (steady_old)
              Rt += *steady_old;
            dirichletRows(trial, lf, Rt, nullptr);
            rt = Rt.norm();
          }
          catch (const InputError &)
          {
            throw;
          }
          catch (const std::exception &)
          {
            // A trial state where a material cannot be evaluated is rejected.
          }
          if (std::isfinite(rt) && rt < best)
          {
            best = rt;
            best_alpha = alpha;
          }
          if (std::isfinite(rt) && rt <= (1.0 - 1e-4 * alpha) * rn)
            break;
          alpha *= 0.5;
        }
        if (std::isfinite(best))
          delta *= best_alpha;
      }
      Vector Unew = _U + delta;
      if (o.relaxation > 0 && it > 1)
        Unew = (1.0 - o.relaxation) * Unew + o.relaxation * _U;
      const double un = Unew.norm();
      rec.step_norm = (Unew - _U).norm() / (un > 0 ? un : 1.0);
      _U = Unew;
      interpolateFirstOrderVariables(_U);
      have_residual_at_U = false;
      result.history.push_back(rec);
      ++result.total_iterations;
      if (o.verbose)
        console::nonlinearIteration(
            it, rn, r0, false, rec.step_norm, result.linear_iterations - linear_before);
      linear_before = result.linear_iterations;
      if (o.nonlinear_solver == "linear" || (it > 1 && rec.step_norm <= o.step_tolerance))
      {
        converged = true;
        break;
      }
    }
    if (!have_residual_at_U)
    {
      Vector lagU = _U;
      base.lagged = &lagU;
      assemble(_U, base, R, nullptr);
      if (steady_old)
        R += *steady_old;
      _last_residual = R;
    }
    if (!converged)
    {
      result.converged = false;
      if (o.error_on_divergence)
      {
        std::ostringstream os;
        os << "dualmesh: nonlinear solve did not converge at load factor " << lf << " within "
           << o.max_iterations << " iterations.";
        throw std::runtime_error(os.str());
      }
      return result;
    }
  }
  result.converged = true;
  return result;
}

SolveResult
Problem::solveSteady(const SolverOptions & options)
{
  initialize();
  AssemblyOptions base;
  base.include_time_kernels = false;
  base.theta = 1.0;
  SolveResult result = nonlinearSolve(options, base, nullptr);
  // A converged steady state is a state the history of stateful materials
  // has reached: commit it, so that a following solve or time step starts
  // from it.  Without this, a transient run after a steady solve would
  // restart the history from the undeformed state and take the whole
  // steady deformation as its first increment.
  if (result.converged && hasState())
  {
    commitState();
    _state_committed_U = _U;
  }
  return result;
}

SolveResult
Problem::takeTimeStep(const Vector & old,
                      double dt,
                      const TransientOptions & tr,
                      const SolverOptions & options)
{
  // The theta family weights the steady part of the residual between the old
  // and the new state:  R_time(U^{n+1}) + theta R_ss(U^{n+1})
  //                                     + (1 - theta) R_ss(U^n) = 0.
  // The old contribution is evaluated once, before the nonlinear iteration,
  // and it is evaluated at the old time t^n: a source, a coefficient or a
  // boundary flux that depends on time explicitly has to be taken at t^n in
  // this term and at t^{n+1} in the other.  Evaluating both at t^{n+1}, as an
  // earlier version did, is consistent but only first-order accurate, which
  // the manufactured-solution tests exposed as a Crank-Nicolson run
  // converging at order one.
  Vector old_residual;
  if (tr.theta < 1.0)
  {
    AssemblyOptions ao;
    ao.include_time_kernels = false;
    ao.theta = 1.0 - tr.theta;
    ao.dt = dt;
    const double new_time = _time;
    _time = new_time - dt;
    try
    {
      assemble(old, ao, old_residual, nullptr);
    }
    catch (...)
    {
      _time = new_time;
      throw;
    }
    _time = new_time;
  }
  AssemblyOptions base;
  base.old = &old;
  base.dt = dt;
  base.theta = tr.theta;
  base.include_time_kernels = true;
  base.include_steady_terms = tr.theta > 0;
  return nonlinearSolve(options, base, tr.theta < 1.0 ? &old_residual : nullptr);
}

SolveResult
runTransient(const TransientOptions & tr,
             const SolverOptions & options,
             Vector & solution,
             double & current_time,
             const std::function<SolveResult(const Vector &, double)> & take_step,
             const std::function<void(int, double)> & on_accept,
             bool verbose_root)
{
  if (tr.dt <= 0)
    throw InputError("Transient: the time step must be positive.");
  if (tr.theta < 0 || tr.theta > 1)
    throw InputError("Transient: theta must be in [0, 1].");
  const bool error_control = tr.time_stepper == "error";
  const bool iteration_control = tr.time_stepper == "iteration";
  if (!error_control && !iteration_control && tr.time_stepper != "fixed")
    throw InputError("Transient: unknown time stepper '" + tr.time_stepper +
                     "' (use fixed, error, or iteration).");
  if (tr.growth_factor < 1.0)
    throw InputError("Transient: growth_factor must be at least one.");
  if (tr.cutback_factor <= 0.0 || tr.cutback_factor >= 1.0)
    throw InputError("Transient: cutback_factor must lie strictly between zero and one.");

  const double span = tr.end_time - tr.start_time;
  const double dt_min = tr.dt_min > 0 ? tr.dt_min : tr.dt * 1.0e-6;
  const double dt_max = tr.dt_max > 0 ? tr.dt_max : std::abs(span);
  // Order of the local truncation error of one step of the theta method: the
  // midpoint rule (theta = 1/2) is second order, every other member is first.
  const double order = std::abs(tr.theta - 0.5) < 1e-12 ? 2.0 : 1.0;

  SolveResult total;
  current_time = tr.start_time;
  int step = 0;
  const auto record = [&](const SolveResult & r)
  {
    total.total_iterations += r.total_iterations;
    total.linear_iterations += r.linear_iterations;
    total.history.insert(total.history.end(), r.history.begin(), r.history.end());
  };

  double dt = std::min(std::max(tr.dt, dt_min), dt_max);
  int rejected_in_a_row = 0;

  while (current_time < tr.end_time - 1e-12 * std::abs(span))
  {
    const double remaining = tr.end_time - current_time;
    // Land exactly on the end time, and do not leave a sliver behind: if one
    // step would nearly finish the interval, finish it.
    double step_size = std::min(dt, remaining);
    if (remaining - step_size < 1e-8 * std::abs(span))
      step_size = remaining;

    const Vector old = solution;
    const double oldcurrent_time = current_time;
    bool accepted = true;
    double next_dt = dt;
    int iterations = 0;

    if (error_control)
    {
      // One coarse step.
      current_time = oldcurrent_time + step_size;
      SolveResult coarse = take_step(old, step_size);
      record(coarse);
      const Vector coarse_solution = solution;
      SolveResult fine_a, fine_b;
      if (coarse.converged)
      {
        // Two half steps from the same starting state.
        solution = old;
        current_time = oldcurrent_time + 0.5 * step_size;
        fine_a = take_step(old, 0.5 * step_size);
        record(fine_a);
        if (fine_a.converged)
        {
          const Vector half = solution;
          current_time = oldcurrent_time + step_size;
          fine_b = take_step(half, 0.5 * step_size);
          record(fine_b);
        }
      }
      iterations =
          std::max({coarse.total_iterations, fine_a.total_iterations, fine_b.total_iterations});
      if (!coarse.converged || !fine_a.converged || !fine_b.converged)
      {
        accepted = false;
        next_dt = step_size * tr.cutback_factor;
      }
      else
      {
        // Richardson estimate of the error of the coarse step.  The two half
        // steps are the more accurate answer and are the one kept.
        const double scale = std::max(solution.norm(), 1.0);
        const double estimate =
            (solution - coarse_solution).norm() / ((std::pow(2.0, order) - 1.0) * scale);
        if (estimate > tr.error_tolerance)
        {
          accepted = false;
          next_dt = step_size *
                    std::max(tr.cutback_factor,
                             0.9 * std::pow(tr.error_tolerance / estimate, 1.0 / (order + 1.0)));
        }
        else
        {
          const double growth =
              estimate > 0 ? 0.9 * std::pow(tr.error_tolerance / estimate, 1.0 / (order + 1.0))
                           : tr.growth_factor;
          next_dt = step_size * std::min(tr.growth_factor, std::max(1.0, growth));
        }
      }
    }
    else
    {
      current_time = oldcurrent_time + step_size;
      SolveResult r = take_step(old, step_size);
      record(r);
      iterations = r.total_iterations;
      accepted = r.converged;
      if (!accepted)
        next_dt = step_size * tr.cutback_factor;
      else if (iteration_control)
      {
        if (iterations < tr.optimal_iterations - tr.iteration_window)
          next_dt = step_size * tr.growth_factor;
        else if (iterations > tr.optimal_iterations + tr.iteration_window)
          next_dt = step_size * tr.cutback_factor;
        else
          next_dt = step_size;
      }
      else
        next_dt = dt;
    }

    if (!accepted)
    {
      ++total.rejected_steps;
      ++rejected_in_a_row;
      solution = old;
      current_time = oldcurrent_time;
      dt = std::max(next_dt, dt_min);
      if (options.verbose && verbose_root)
        std::cout << "  step rejected at t = " << oldcurrent_time << "; retrying with dt = " << dt
                  << "\n";
      if (next_dt < dt_min || rejected_in_a_row > tr.max_rejected_steps)
      {
        if (options.error_on_divergence)
        {
          std::ostringstream os;
          os << "dualmesh: the transient solve failed at t = " << oldcurrent_time
             << ". The step was rejected " << rejected_in_a_row << " times and the step size "
             << "reached " << next_dt << ", below the minimum " << dt_min << ".";
          throw std::runtime_error(os.str());
        }
        return total;
      }
      continue;
    }

    rejected_in_a_row = 0;
    ++step;
    ++total.time_steps;
    total.step_history.emplace_back(current_time, step_size);
    if (options.verbose && verbose_root)
      std::cout << "  step " << step << ": t = " << current_time << ", dt = " << step_size
                << ", iterations " << iterations << "\n";
    on_accept(step, step_size);
    dt = std::min(std::max(next_dt, dt_min), dt_max);
  }
  total.converged = true;
  return total;
}

SolveResult
Problem::solveTransient(const TransientOptions & tr, const SolverOptions & options)
{
  initialize();
  // The step is retried on failure, so the nonlinear solver must report a
  // failure rather than throw.
  SolverOptions attempt = options;
  attempt.error_on_divergence = false;
  applyDirichlet(_U, 1.0);
  const auto write = [&](int step)
  {
    if (tr.output_interval > 0 && step % tr.output_interval == 0 && !tr.output_file_base.empty())
    {
      std::ostringstream os;
      os << tr.output_file_base << "_" << std::setw(5) << std::setfill('0') << step << ".vtu";
      writeVTU(os.str());
    }
  };
  write(0);
  // The history of stateful materials follows the solution: a step starts
  // from the committed history when it starts from the last accepted
  // solution, and from the state its predecessor reached when it continues
  // that predecessor (the second half of a doubled step).
  _state_committed_U = _U;
  return runTransient(
      tr,
      options,
      _U,
      _time,
      [&](const Vector & old, double dt)
      {
        if (hasState())
        {
          const bool from_committed = old.size() == _state_committed_U.size() &&
                                      (old.array() == _state_committed_U.array()).all();
          if (from_committed)
            startStateFromCommitted();
          else
            startStateFromTrial();
        }
        return takeTimeStep(old, dt, tr, attempt);
      },
      [&](int step, double)
      {
        if (hasState())
        {
          commitState();
          _state_committed_U = _U;
        }
        if (_step_callback)
          _step_callback(_time, *this);
        write(step);
      },
      true);
}

} // namespace dualmesh
