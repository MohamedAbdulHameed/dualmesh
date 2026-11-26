// SPDX-License-Identifier: LGPL-2.1-or-later
//
// Residual/Jacobian assembly and the steady and transient executioners.
#include "dualmesh/base/Console.h"
#include "dualmesh/base/Problem.h"
#include "dualmesh/parallel/Checkpoint.h"

#include "dualmesh/linalg/LinearSolverChoice.h"
#include "dualmesh/linalg/PetscSolver.h"
#include "dualmesh/linalg/SaddlePointBlocks.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <deque>
#include <exception>
#include <iostream>
#include <limits>
#include <numeric>
#include <optional>
#include <set>
#include <sstream>
#include <tuple>

namespace dualmesh
{

namespace
{
using Triplet = Eigen::Triplet<double>;

/// Add atomically when @p atomic, which is what lets several threads assemble different elements
/// into the same vector.
void
addTo(double & target, double value, bool atomic)
{
  if (atomic)
  {
#pragma omp atomic
    target += value;
  }
  else
    target += value;
}

/// Add @p value to entry @p row of the residual, its magnitude to the reference residual when
/// there is one (AssemblyOptions::reference), and @p size, the size of the terms it is computed
/// from, to the term sizes when they are asked for (AssemblyOptions::term_size).
void
addToResidual(Vector & R,
              Vector * reference,
              Vector * term_size,
              Index row,
              double value,
              double size,
              bool atomic)
{
  addTo(R[row], value, atomic);
  if (reference)
    addTo((*reference)[row], std::abs(value), atomic);
  if (term_size)
    addTo((*term_size)[row], size, atomic);
}

/// Scatter an element-local AD residual into the global residual/Jacobian.
/// The term sizes of the entries are @p sizes when they are given, and otherwise the magnitudes
/// of the entries.
void
scatter(const std::vector<ADReal> & Rloc,
        const std::vector<Index> & dofs,
        const std::vector<char> & touched,
        Vector & R,
        Vector * reference,
        Vector * term_size,
        const std::vector<double> * sizes,
        std::vector<Triplet> * trip,
        bool atomic)
{
  const int ld = static_cast<int>(dofs.size());
  for (int i = 0; i < ld; ++i)
  {
    if (!touched[i])
      continue;
    const double value = Rloc[i].value();
    addToResidual(R,
                  reference,
                  term_size,
                  dofs[i],
                  value,
                  sizes ? (*sizes)[static_cast<std::size_t>(i)] : std::abs(value),
                  atomic);
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
  const Index ndof = numDofs();
  const int nv = numVariables();
  const int dim = _mesh->spaceDimension();
  const bool dual = isDual();
  const bool want_jac = J != nullptr;
  R.setZero(ndof);
  if (opts.reference)
    opts.reference->setZero(ndof);
  if (opts.term_size)
    opts.term_size->setZero(ndof);
  Vector * const term_size = opts.term_size;
  const int nthreads = effectiveThreads();
  const bool atomic = nthreads > 1;
  std::vector<ThreadScratch> scratch(nthreads);
  for (auto & s : scratch)
  {
    s.ctx.dim = dim;
    s.ctx.time = _time;
    s.ctx.dt = opts.dt;
    s.ctx.eigenvalue = _eigenvalue;
    s.ctx.data_fraction = opts.data_fraction;
    s.ctx.inertia = opts.inertia;
    s.ctx.stabilization = opts.stabilization;
    s.ctx.steady_solve = !opts.include_time_kernels && opts.dt == 0.0;
    s.ctx.theta =
        opts.previous_step ? opts.previous_theta : (opts.include_time_kernels ? opts.theta : 1.0);
    s.ctx.old_level = opts.old_level;
    s.ctx.previous_step = opts.previous_step;
    s.ctx.mode = opts.mode;
    s.ctx.second_order.clear();
    if (opts.second_order)
    {
      s.ctx.second_order = secondOrderVariables();
      s.ctx.rate_scale = opts.rate_scale;
      s.ctx.accel_scale = opts.accel_scale;
      s.ctx.rate_shift.assign(nv, 0.0);
      s.ctx.accel_shift.assign(nv, 0.0);
      s.ctx.grad_rate_shift.assign(nv, Point{0, 0, 0});
    }
    if (want_jac)
      s.triplets.reserve(static_cast<std::size_t>(_mesh->numElements()) * 16 * nv * nv * 4 /
                         nthreads);
  }
  const bool mixed = _mixed_order;
  // The kernels of the assembly in groups of one rule: the problem's, or those of the options.
  std::vector<Group> chosen;
  if (opts.kernels)
    for (const Kernel * k : *opts.kernels)
    {
      const QuadratureSpec spec =
          _eigenvalue_study ? k->eigenvalueStudyQuadrature() : k->quadrature();
      auto it = std::find_if(
          chosen.begin(), chosen.end(), [&](const Group & g) { return g.spec == spec; });
      if (it == chosen.end())
        it = chosen.insert(chosen.end(), Group{spec, {}});
      it->kernels.push_back(k);
    }
  const std::vector<Group> & groups = opts.kernels ? chosen : _groups;

  const auto assembleElement = [&](ThreadScratch & s, Index e)
  {
    auto & cur = s.cur;
    auto & dofs = s.dofs;
    auto & Rloc = s.Rloc;
    auto & touched = s.touched;
    auto & sizes = s.sizes;
    auto & pts = s.pts;
    // The mapped point of the current integration point: the one built with
    // the points when there is one (no copy), otherwise s.geo.
    const MappedPoint * gp = &s.geo;
    auto & ctx = s.ctx;
    const Element & el = _mesh->element(e);
    const int nn = el.numNodes();
    const PolygonCell * polygon = polygonCell(e);
    const int all = gatherLocal(s, el, U, opts, polygon);
    const int nd = want_jac ? all : 0;
    const auto nodeOf = [&](int slot)
    { return slot < nn ? el.nodes[slot] : polygon->nodes[static_cast<std::size_t>(slot)]; };
    ctx.element = e;
    ctx.block = el.block;
    ctx.on_boundary = false;
    bool any = false;

    for (const auto & group : groups)
    {
      const unsigned group_index = static_cast<unsigned>(&group - groups.data());
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
        if (!k->isTimeKernel())
          return steadyWeight(*k, opts);
        return k->isLoad() ? opts.data_fraction : 1.0;
      };
      // The points were mapped when they were built; reuse those maps.
      const auto evaluateAt = [&](const IntegrationPoint & ip, std::size_t q)
      {
        gp = &evaluatePoint(s,
                            el,
                            ip,
                            q < s.mapped.size() ? &s.mapped[q] : nullptr,
                            group.spec.reduced,
                            opts,
                            nd,
                            e,
                            stateKey(point_set, group_index, 0, static_cast<unsigned>(q)));
        return coordFactor(gp->x);
      };

      // Volume integrals (DMCDM: control volumes; FEM: element interior).
      if (has_source || (!dual && has_flux))
      {
        buildElementPoints(*_mesh, e, dual, PointSet::Volume, group.spec, pts, &s.mapped, polygon);
        point_set = 0;
        for (std::size_t q = 0; q < pts.size(); ++q)
        {
          const IntegrationPoint & ip = pts[q];
          const double cf = evaluateAt(ip, q) * ip.weight;
          for (const Kernel * k : ks)
          {
            const int v = k->variable();
            const double w = cf * weightOf(k);
            ctx.time_weight = timeWeight(*k, opts);
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
              const double S_size = term_size ? k->sourceTermSize(ctx, S) : 0.0;
              if (ip.owner >= 0)
              {
                Rloc[ip.owner * nv + v].addScaledBy(S, w);
                touched[ip.owner * nv + v] = 1;
                if (term_size)
                  sizes[ip.owner * nv + v] += std::abs(w) * S_size;
              }
              else
                for (int a = 0; a < ntest; ++a)
                {
                  const int i = slot(a);
                  Rloc[i].addScaledBy(S, w * testN[a]);
                  touched[i] = 1;
                  if (term_size)
                    sizes[i] += std::abs(w * testN[a]) * S_size;
                }
            }
            if (!dual && k->hasFlux() && ip.owner < 0)
            {
              ADVector3 F{ADReal(0.0), ADReal(0.0), ADReal(0.0)};
              k->computeFlux(ctx, F);
              double F_size[3] = {0.0, 0.0, 0.0};
              if (term_size)
                k->fluxTermSize(ctx, F, F_size);
              for (int a = 0; a < ntest; ++a)
              {
                const int i = slot(a);
                ADReal & r = Rloc[i];
                for (int d = 0; d < dim; ++d)
                  if (testdN[a][d] != 0.0)
                  {
                    r.addScaledBy(F[d], w * testdN[a][d]);
                    if (term_size)
                      sizes[i] += std::abs(w * testdN[a][d]) * F_size[d];
                  }
                touched[i] = 1;
              }
            }
          }
        }
      }

      // Interfaces between control volumes (DMCDM only).
      if (dual && has_flux)
      {
        buildElementPoints(*_mesh, e, dual, PointSet::Faces, group.spec, pts, &s.mapped, polygon);
        point_set = 1;
        // The flux of kernel k through interface ip, times scale: out of the control domain of the
        // owner (-F.n) and into that of the neighbour, with the sizes of the two terms.  A
        // non-conservative face flux of an advective kernel differs between the two sides
        // (Kernel::nonConservativeFaceFlux): FN is that of the neighbour's equation.
        const auto faceFlux = [&](const Kernel * k,
                                  const IntegrationPoint & ip,
                                  double scale,
                                  bool advective,
                                  ADReal & ro,
                                  ADReal & rn,
                                  double & ro_size,
                                  double & rn_size)
        {
          const bool per_side = advective && k->nonConservativeFaceFlux();
          ADVector3 F{ADReal(0.0), ADReal(0.0), ADReal(0.0)};
          if (per_side)
            ctx.reference_side = 0;
          k->computeFlux(ctx, F);
          ADVector3 FN = F;
          if (per_side)
          {
            ctx.reference_side = 1;
            FN = {ADReal(0.0), ADReal(0.0), ADReal(0.0)};
            k->computeFlux(ctx, FN);
            ctx.reference_side = -1;
          }
          double F_size[3] = {0.0, 0.0, 0.0}, FN_size[3] = {0.0, 0.0, 0.0};
          if (term_size)
          {
            k->fluxTermSize(ctx, F, F_size);
            if (per_side)
              k->fluxTermSize(ctx, FN, FN_size);
            else
              std::copy(F_size, F_size + 3, FN_size);
          }
          for (int d = 0; d < dim; ++d)
            if (ip.area[d] != 0.0)
            {
              ro.addScaledBy(F[d], -scale * ip.area[d]);
              rn.addScaledBy(FN[d], scale * ip.area[d]);
              ro_size += std::abs(scale * ip.area[d]) * F_size[d];
              rn_size += std::abs(scale * ip.area[d]) * FN_size[d];
            }
        };
        const bool sides =
            std::any_of(ks.begin(), ks.end(), [](const Kernel * k) { return k->usesFaceSides(); });
        for (std::size_t q = 0; q < pts.size(); ++q)
        {
          const IntegrationPoint & ip = pts[q];
          const double cf = evaluateAt(ip, q);
          if (sides)
          {
            // The two nodes of the face, for an advective flux: their values (with their
            // derivatives), their positions, and the normal from the owner to the neighbour.
            // A node has no gradient of its own, so the sides carry none.
            ctx.on_face = true;
            const int ends[2] = {ip.owner, ip.neighbor};
            for (int side = 0; side < 2; ++side)
            {
              ctx.side_u[side].resize(nv);
              ctx.side_grad[side].clear();
              ctx.side_x[side] = _mesh->node(nodeOf(ends[side]));
              for (int v = 0; v < nv; ++v)
              {
                const int slot = ends[side] * nv + v;
                ADReal value = ADReal::withZeroDerivatives(cur[slot], nd);
                if (nd > 0)
                  value.setDerivative(slot, 1.0);
                ctx.side_u[side][v] = value;
              }
            }
            ctx.side_weight = 0.5;
            const double area = std::sqrt(dot(ip.area, ip.area));
            ctx.normal = area > 0.0 ? (1.0 / area) * ip.area : Point{0, 0, 0};
          }
          for (const Kernel * k : ks)
          {
            if (!k->hasFlux())
              continue;
            const int v = k->variable();
            ctx.time_weight = timeWeight(*k, opts);
            if (sides && k->needsSideGradients() && !_gradient_offsets.empty())
            {
              // The linear upwind and limited schemes take the gradient at the upstream node,
              // reconstructed from the nodes around it, so the flux depends on those nodes too.
              // They take the slots after the element's own; the two rows of the face go out
              // here with those columns.
              auto & columns = s.columns;
              columns.assign(dofs.begin(), dofs.end());
              // A slot beyond the derivatives of this build is refused before it is written.
              const auto slotOf = [&](Index node)
              {
                const Index d = dof(node, v);
                for (std::size_t i = 0; i < columns.size(); ++i)
                  if (columns[i] == d)
                    return static_cast<int>(i);
                if (static_cast<int>(columns.size()) >= kMaxDerivatives)
                  throw InputError("The face flux of '" + k->name() + "' depends on more than " +
                                   std::to_string(kMaxDerivatives) +
                                   " unknowns, the derivatives of this build. Use the upwind "
                                   "scheme here.");
                columns.push_back(d);
                return static_cast<int>(columns.size()) - 1;
              };
              for (int side = 0; side < 2; ++side)
              {
                const Index node = nodeOf(side == 0 ? ip.owner : ip.neighbor);
                ctx.side_grad[side].assign(nv, {ADReal(0.0), ADReal(0.0), ADReal(0.0)});
                ADVector3 & g = ctx.side_grad[side][v];
                const ADReal & ua = ctx.side_u[side][v];
                const auto a = static_cast<std::size_t>(node);
                for (Index i = _gradient_offsets[a]; i < _gradient_offsets[a + 1]; ++i)
                {
                  const Index b = _gradient_nodes[static_cast<std::size_t>(i)];
                  ADReal ub(U[dof(b, v)]);
                  if (want_jac)
                  {
                    const int slot = slotOf(b);
                    ub = ADReal::withZeroDerivatives(U[dof(b, v)], slot + 1);
                    ub.setDerivative(slot, 1.0);
                  }
                  const ADReal difference = ub - ua;
                  const Point & w = _gradient_weights[static_cast<std::size_t>(i)];
                  for (int c = 0; c < dim; ++c)
                    g[c].addScaledBy(difference, w[c]);
                }
              }
              ADReal ro(0.0), rn(0.0);
              double ro_size = 0.0, rn_size = 0.0;
              faceFlux(k, ip, cf * weightOf(k), true, ro, rn, ro_size, rn_size);
              ctx.side_grad[0].clear();
              ctx.side_grad[1].clear();
              for (const auto & [row, r, size] : {std::tuple<Index, const ADReal *, double>{
                                                      dofs[ip.owner * nv + v], &ro, ro_size},
                                                  std::tuple<Index, const ADReal *, double>{
                                                      dofs[ip.neighbor * nv + v], &rn, rn_size}})
              {
                addToResidual(R, opts.reference, term_size, row, r->value(), size, atomic);
                if (want_jac)
                {
                  const double * dr = r->derivatives();
                  for (int j = 0; j < r->size(); ++j)
                    if (dr[j] != 0.0)
                      s.triplets.emplace_back(row, columns[static_cast<std::size_t>(j)], dr[j]);
                }
              }
              continue;
            }
            double ro_size = 0.0, rn_size = 0.0;
            faceFlux(k,
                     ip,
                     cf * weightOf(k),
                     sides,
                     Rloc[ip.owner * nv + v],
                     Rloc[ip.neighbor * nv + v],
                     ro_size,
                     rn_size);
            if (term_size)
            {
              sizes[ip.owner * nv + v] += ro_size;
              sizes[ip.neighbor * nv + v] += rn_size;
            }
            touched[ip.owner * nv + v] = 1;
            touched[ip.neighbor * nv + v] = 1;
          }
        }
        ctx.on_face = false;
      }
    }
    if (any)
      scatter(Rloc,
              dofs,
              touched,
              R,
              opts.reference,
              term_size,
              term_size ? &sizes : nullptr,
              want_jac ? &s.triplets : nullptr,
              atomic);
  };

  // The element loop is the expensive part and it is embarrassingly parallel:
  // every thread keeps its own scratch space and its own list of matrix
  // entries, and the only shared write is the atomic update of the residual.
  // Threading is disabled automatically when any object is defined in Python,
  // because calling back into the interpreter needs the global interpreter
  // lock (see Problem::initialize).
  // The first error of a thread, rethrown as it is, so that its type (InputError) is kept.
  std::exception_ptr thread_error;
#ifdef _OPENMP
#pragma omp parallel num_threads(nthreads) if (nthreads > 1)
  {
    ThreadScratch & s = scratch[omp_get_thread_num()];
#pragma omp for schedule(static)
    for (Index e = 0; e < numIntegratedElements(); ++e)
    {
      try
      {
        assembleElement(s, e);
      }
      catch (...)
      {
#pragma omp critical(dualmesh_thread_error)
        if (!thread_error)
          thread_error = std::current_exception();
      }
    }
  }
#else
  for (Index e = 0; e < numIntegratedElements(); ++e)
    assembleElement(scratch[0], e);
#endif
  if (thread_error)
    std::rethrow_exception(thread_error);

  // The boundary terms are a small fraction of the work and are assembled
  // serially, in the first thread's scratch space.
  ThreadScratch & s0 = scratch[0];
  auto & cur = s0.cur;
  auto & dofs = s0.dofs;
  auto & Rloc = s0.Rloc;
  auto & touched = s0.touched;
  auto & pts = s0.pts;
  auto & ctx = s0.ctx;

  // ---- integrated boundary conditions ------------------------------------
  std::vector<const IntegratedBC *> conditions;
  if (opts.conditions)
    conditions = *opts.conditions;
  else if (!opts.kernels)
    for (const auto & bc : _ibcs)
      conditions.push_back(bc.get());
  if (opts.include_steady_terms)
    for (std::size_t index = 0; index < conditions.size(); ++index)
    {
      const IntegratedBC * bc = conditions[index];
      if (bc->isConstraint())
      {
        // A constraint holds at the new time only.
        if (!opts.old_level)
          bc->assembleConstraint(
              *this, U, opts.old, R, opts.reference, want_jac ? &s0.triplets : nullptr);
        continue;
      }
      const double wfac = steadyWeight(*bc, opts);
      if (const auto * ib = dynamic_cast<const InterfaceBC *>(bc))
      {
        assembleInterface(*ib, U, opts, wfac, R, want_jac ? &s0.triplets : nullptr, s0);
        continue;
      }
      for (const Side & side : boundarySides(bc->boundaries()))
      {
        const Element & el = _mesh->element(side.first);
        if (!bc->activeOnBlock(el.block))
          continue;
        const int ld = gatherLocal(s0, el, U, opts);
        conditionResidual(
            *bc, static_cast<unsigned>(index), side, wfac, opts, want_jac ? ld : 0, s0);
        scatter(s0.Rloc,
                s0.dofs,
                s0.touched,
                R,
                opts.reference,
                term_size,
                nullptr,
                want_jac ? &s0.triplets : nullptr,
                false);
      }
    }

  // ---- advective flux through the boundary (DMCDM) ----------------------------
  // An advective face flux passes through the boundary faces of the control volumes too.  Both
  // sides of such a face are the node whose control volume the face bounds: at an outflow that
  // node is upstream, and at an inflow without a condition it holds the only value there is.
  // The conservative form m u_f carries the face value out of the domain, so a boundary without a
  // condition has zero conductive flux.  The non-conservative form m (u_f - u_C) takes the face
  // value of its scheme less the node's: zero for upwind, and for central the difference between
  // the value at the face and at the node, which keeps it consistent at an outlet (Poiseuille
  // flow is exact on quadratic elements).  The boundary is that of the side sets, which hold the
  // boundary of the whole mesh (the part of a distributed run has none on its interfaces with
  // the other parts).
  if (dual && opts.include_steady_terms && !opts.kernels)
  {
    std::vector<const Kernel *> advective;
    for (const auto & k : _kernels)
      if (k->usesFaceSides())
        advective.push_back(k.get());
    if (!advective.empty())
    {
      checkAdvectiveBoundary();
      std::set<Side> sides;
      for (const auto & [name, list] : _mesh->sidesets())
        for (const Side & side : list)
          if (side.first < numIntegratedElements())
            sides.insert(side);
      for (const Side & side : sides)
      {
        const Element & el = _mesh->element(side.first);
        for (const Kernel * k : advective)
        {
          if (!k->activeOnBlock(el.block))
            continue;
          const int v = k->variable();
          const auto & vb = _vars[v].blocks;
          if (!vb.empty() && !vb.count(el.block))
            continue;
          const double wfac = steadyWeight(*k, opts);
          const int ld = gatherLocal(s0, el, U, opts);
          const int nd = want_jac ? ld : 0;
          ctx.element = side.first;
          ctx.block = el.block;
          ctx.on_boundary = true;
          buildSidePoints(*_mesh, side, dual, k->quadrature(), pts);
          for (const auto & ip : pts)
          {
            ctx.normal = (1.0 / ip.weight) * ip.area;
            const MappedPoint & geo =
                evaluatePoint(s0, el, ip, nullptr, k->quadrature().reduced, opts, nd, -1, 0);
            // Both sides are the owner node, so that every scheme takes its value.
            ctx.on_face = true;
            ctx.side_u[0].resize(nv);
            for (int u = 0; u < nv; ++u)
            {
              const int slot = ip.owner * nv + u;
              ctx.side_u[0][u] = ADReal::withZeroDerivatives(cur[slot], nd);
              if (nd > 0)
                ctx.side_u[0][u].setDerivative(slot, 1.0);
            }
            ctx.side_u[1] = ctx.side_u[0];
            ctx.side_grad[0].clear();
            ctx.side_grad[1].clear();
            ctx.side_x[0] = ctx.side_x[1] = _mesh->node(el.nodes[ip.owner]);
            ctx.side_weight = 0.5;
            ctx.reference_side = k->nonConservativeFaceFlux() ? 0 : -1;
            ADVector3 F{ADReal(0.0), ADReal(0.0), ADReal(0.0)};
            k->computeFlux(ctx, F);
            ctx.reference_side = -1;
            ADReal Fn(0.0);
            for (int d = 0; d < _mesh->spaceDimension(); ++d)
              Fn += F[d] * ctx.normal[d];
            const double w = ip.weight * coordFactor(geo.x) * wfac;
            Rloc[ip.owner * nv + v].addScaledBy(Fn, -w);
            touched[ip.owner * nv + v] = 1;
          }
          ctx.on_face = false;
          scatter(Rloc,
                  dofs,
                  touched,
                  R,
                  opts.reference,
                  term_size,
                  nullptr,
                  want_jac ? &s0.triplets : nullptr,
                  false);
        }
      }
    }
  }

  // ---- concentrated nodal loads ----------------------------------------------
  if (opts.include_steady_terms && !opts.kernels)
    for (const auto & load : _loads)
    {
      const int v = load->variable();
      const double wfac = steadyWeight(*load, opts);
      for (Index n : load->nodes())
      {
        if (!_owned_node.empty() && !_owned_node[n])
          continue;
        const double value = -wfac * load->computeValue(_mesh->node(n), _time);
        addToResidual(R, opts.reference, term_size, dof(n, v), value, std::abs(value), false);
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

int
Problem::gatherLocal(ThreadScratch & s,
                     const Element & el,
                     const Vector & U,
                     const AssemblyOptions & opts,
                     const PolygonCell * polygon) const
{
  // Every variable at every node, except that a first-order variable on a quadratic element has
  // only its corner nodes.  A cell of the dual mesh method with nodes hanging on its sides
  // (PolygonCell) takes them as further slots after its own: they own control domain and
  // interfaces in the cell, and the fields of the cell, which come from its own nodes, do not
  // depend on them.
  const int nv = numVariables();
  s.dofs.resize(static_cast<std::size_t>(numLocalDofs(el)));
  for (int k = 0; k < el.numNodes(); ++k)
    for (int v = 0; v < nv; ++v)
    {
      const int i = localDofIndex(el, k, v);
      if (i >= 0)
        s.dofs[i] = dof(el.nodes[k], v);
    }
  if (polygon)
    for (std::size_t k = static_cast<std::size_t>(el.numNodes()); k < polygon->nodes.size(); ++k)
      for (int v = 0; v < nv; ++v)
        s.dofs.push_back(dof(polygon->nodes[k], v));
  const int ld = static_cast<int>(s.dofs.size());
  const auto take = [&](const Vector * from, std::vector<double> & to)
  {
    if (!from)
      return;
    to.resize(ld);
    for (int i = 0; i < ld; ++i)
      to[i] = (*from)[s.dofs[i]];
  };
  take(&U, s.cur);
  take(opts.mode == LinearizationMode::Picard ? opts.lagged : nullptr, s.lag);
  take(opts.old, s.old);
  take(opts.rate_shift, s.rate_shift);
  take(opts.rate_shift ? opts.accel_shift : nullptr, s.accel_shift);
  s.Rloc.assign(ld, ADReal(0.0));
  s.touched.assign(ld, 0);
  if (opts.term_size)
    s.sizes.assign(static_cast<std::size_t>(ld), 0.0);
  return ld;
}

const MappedPoint &
Problem::evaluatePoint(ThreadScratch & s,
                       const Element & el,
                       const IntegrationPoint & ip,
                       MappedPoint * mapped,
                       bool reduced,
                       const AssemblyOptions & opts,
                       int nd,
                       Index owner,
                       std::uint64_t key) const
{
  MappedPoint & geo = mapped ? *mapped : s.geo;
  if (!mapped)
    mapPoint(*_mesh, el, ip.xi, geo);
  if (_mixed_order)
    mapCornerShape(el, geo);
  const MappedPoint * fp = &geo;
  if (reduced)
  {
    mapPoint(*_mesh, el, ip.xi_field, s.fld);
    if (_mixed_order)
      mapCornerShape(el, s.fld);
    fp = &s.fld;
  }
  const bool lagged = opts.mode == LinearizationMode::Picard && opts.lagged;
  fillContext(
      s.ctx, el, *fp, s.cur, lagged ? &s.lag : nullptr, opts.old ? &s.old : nullptr, nd);
  if (opts.rate_shift)
    fillShifts(s.ctx, el, *fp, s.rate_shift, s.accel_shift);
  s.ctx.x = geo.x;
  s.ctx.state_owner = owner;
  s.ctx.state_key = key;
  evaluateProperties(s.ctx);
  return geo;
}

void
Problem::conditionResidual(const IntegratedBC & bc,
                           unsigned index,
                           const Side & side,
                           double wfac,
                           const AssemblyOptions & opts,
                           int nd,
                           ThreadScratch & s) const
{
  const Element & el = _mesh->element(side.first);
  const int v = bc.variable();
  const int nv = numVariables();
  s.ctx.element = side.first;
  s.ctx.block = el.block;
  s.ctx.on_boundary = true;
  buildSidePoints(*_mesh, side, isDual(), bc.quadrature(), s.pts);
  const bool corner = usesCornerShape(el, v);
  for (std::size_t q = 0; q < s.pts.size(); ++q)
  {
    const IntegrationPoint & ip = s.pts[q];
    s.ctx.normal = (1.0 / ip.weight) * ip.area;
    const MappedPoint & geo = evaluatePoint(
        s,
        el,
        ip,
        nullptr,
        bc.quadrature().reduced,
        opts,
        nd,
        side.first,
        stateKey(2, index, static_cast<unsigned>(side.second), static_cast<unsigned>(q)));
    // The flux of the condition enters the body.
    const ADReal flux = bc.computeBoundaryFlux(s.ctx) * (ip.weight * coordFactor(geo.x) * wfac);
    if (ip.owner >= 0)
    {
      s.Rloc[ip.owner * nv + v] -= flux;
      s.touched[ip.owner * nv + v] = 1;
      continue;
    }
    const double * testN = corner ? geo.corner_shape : geo.N;
    const int ntest = corner ? geo.num_corners : el.numNodes();
    for (int a = 0; a < ntest; ++a)
    {
      const int i = localDofIndex(el, a, v);
      s.Rloc[i] -= flux * testN[a];
      s.touched[i] = 1;
    }
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
  const int dim = _mesh->spaceDimension();
  const bool dual = isDual();
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
  for (const Side & side : boundarySides(bc.boundaries()))
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
      if (ctx.mode == LinearizationMode::Picard)
      {
        ctx.u_lag = ctx.u;
        ctx.grad_u_lag = ctx.grad_u;
      }
      ctx.element = side.first;
      ctx.block = el.block;
      ctx.on_boundary = true;
      ctx.x = geo.x;
      ctx.x_other = sgeo.x;
      ctx.normal = (1.0 / ip.weight) * ip.area;
      ctx.state_owner = side.first;
      ctx.state_key =
          stateKey(3, bc_index, static_cast<unsigned>(side.second), static_cast<unsigned>(q));
      evaluateProperties(ctx);
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
        const int j = static_cast<int>(std::find(ssn.begin(), ssn.end(), pair.owner) - ssn.begin());
        s.Rloc[(np + j) * nc + cs] += flux;
        s.touched[(np + j) * nc + cs] = 1;
      }
      else
        for (int j = 0; j < ns; ++j)
        {
          s.Rloc[(np + j) * nc + cs].addScaledBy(flux, sgeo.N[ssn[j]]);
          s.touched[(np + j) * nc + cs] = 1;
        }
      scatter(
          s.Rloc, s.dofs, s.touched, R, opts.reference, opts.term_size, nullptr, triplets, false);
    }
  }
}

void
Problem::buildInterfacePairs()
{
  _interface_pairs.clear();
  const bool dual = isDual();
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
    for (const auto & name : ib->secondaryBoundaries())
    {
      const auto & set = _mesh->sideset(name);
      sides.insert(sides.end(), set.begin(), set.end());
      // The sides of the ghost elements that a distributed run added for this interface.
      const auto ghosts = _interface_candidates.find(name);
      if (ghosts != _interface_candidates.end())
        sides.insert(sides.end(), ghosts->second.begin(), ghosts->second.end());
    }
    // Only primary points need a partner: a part of a distributed mesh that holds no primary side has nothing to pair.
    bool has_primary = false;
    for (const auto & bname : ib->boundaries())
      has_primary = has_primary || !_mesh->sideset(bname).empty();
    if (sides.empty() && has_primary)
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
        }
      }
      if (dual)
        best.owner = sideControlOwner(*_mesh, {best.element, best.side}, best.xi);
      return best;
    };
    auto & out = _interface_pairs[bc.get()];
    for (const Side & side : boundarySides(ib->boundaries()))
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

void
Problem::applyDirichlet(Vector & U, double data_fraction) const
{
  // A ramped value moves from the state the continuation started from (_ramp_from) to its own.
  const bool ramp = data_fraction != 1.0 && _ramp_from.size() == U.size();
  for (const auto & bc : _nbcs)
  {
    const int v = bc->variable();
    for (Index n : bc->nodes())
    {
      const Index d = dof(n, v);
      const double g = bc->computeValue(_mesh->node(n), _time);
      U[d] = ramp ? _ramp_from[d] + data_fraction * (g - _ramp_from[d]) : g;
    }
  }
  // A condition on a first-order variable also wrote the unused slots of the
  // non-corner boundary nodes; restore the linear interpolant there.
  updateDependentDofs(U);
  parallel().shareDirichlet(U);
}

std::vector<char>
Problem::prescribedDofs() const
{
  std::vector<char> prescribed(static_cast<std::size_t>(numDofs()), 0);
  for (const auto & bc : _nbcs)
    for (Index n : bc->nodes())
      prescribed[static_cast<std::size_t>(dof(n, bc->variable()))] = 1;
  return prescribed;
}

std::vector<char>
Problem::constrainedDofs() const
{
  std::vector<char> fixed = prescribedDofs();
  for (Index i = 0; i < numDofs(); ++i)
    if (!_active_dof[i] || isDependent(i))
      fixed[i] = 1;
  for (std::size_t i = 0; i < _extra_constrained.size() && i < fixed.size(); ++i)
    if (_extra_constrained[i])
      fixed[i] = 1;
  for (const auto & level : _levels)
    if (level.dof >= 0)
      fixed[level.dof] = 1;
  return fixed;
}

void
Problem::foldDependents(Vector & R, SparseMatrix * J) const
{
  if (!_fold_dependents)
    return;
  // u_d = sum_j w_j u_j: the equation of d goes to those of the j with the weights.
  for (std::size_t d = 0; d < _dependent_of.size(); ++d)
    if (!_dependent_of[d].empty())
    {
      for (const auto & [p, w] : _dependent_of[d])
        R[p] += w * R[static_cast<Index>(d)];
      R[static_cast<Index>(d)] = 0.0;
    }
  if (!J)
    return;
  const auto expand = [&](Index i, auto && add)
  {
    const auto & on = _dependent_of[static_cast<std::size_t>(i)];
    if (on.empty())
      add(i, 1.0);
    else
      for (const auto & [p, w] : on)
        add(p, w);
  };
  std::vector<Triplet> folded;
  folded.reserve(static_cast<std::size_t>(J->nonZeros()));
  for (Index col = 0; col < J->outerSize(); ++col)
    for (SparseMatrix::InnerIterator it(*J, col); it; ++it)
      expand(it.row(),
             [&](Index r, double wr)
             {
               expand(it.col(),
                      [&](Index c, double wc) { folded.emplace_back(r, c, wr * wc * it.value()); });
             });
  J->setFromTriplets(folded.begin(), folded.end());
}

void
Problem::dirichletRows(Vector & R, SparseMatrix * J) const
{
  foldDependents(R, J);
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
    // conjugate gradient solver needs.  The row of a prescribed value takes the
    // mean size of the diagonal of its equation, the same on every process: a
    // row of the identity would keep its size when a change of units scales
    // the rest of its equation, and the iterative solvers would see another
    // system in other units.  An equation without a diagonal (the pressure of
    // Taylor-Hood elements) takes the mean size of its rows instead.
    const int nv = numVariables();
    const Vector diagonal = J->diagonal();
    const Vector rows = J->cwiseAbs() * Vector::Ones(J->cols());
    std::vector<double> size(static_cast<std::size_t>(nv), 0.0), row_size(size), count(size);
    for (Index i = 0; i < numDofs(); ++i)
      if (!fixed[i])
      {
        const auto v = static_cast<std::size_t>(i % nv);
        size[v] += std::abs(diagonal[i]);
        row_size[v] += rows[i];
        count[v] += 1.0;
      }
    const Parallel & par = parallel();
    for (std::size_t v = 0; v < size.size(); ++v)
    {
      const double total = par.sum(size[v]), row_total = par.sum(row_size[v]);
      const double number = par.sum(count[v]);
      size[v] = total > 0.0 ? total / number : (row_total > 0.0 ? row_total / number : 1.0);
    }
    J->prune([&](const Index & row, const Index & col, const double &)
             { return !fixed[row] && !fixed[col]; });
    std::vector<Triplet> diag;
    for (Index i = 0; i < numDofs(); ++i)
      if (fixed[i])
        diag.emplace_back(i, i, size[static_cast<std::size_t>(i % nv)]);
    SparseMatrix D(numDofs(), numDofs());
    D.setFromTriplets(diag.begin(), diag.end());
    *J += D;
  }
}

double
Problem::steadyWeight(const ResidualObject & object, const AssemblyOptions & options) const
{
  // A step of the theta method weights the steady terms by theta at the new time and by 1 - theta
  // at the old one.  An equation without a time derivative (the mass equation of a flow, the
  // equilibrium of a quasi-static solid) is a constraint of the new time: weighting it between
  // the two times would make its unknown (the pressure) an average of two steps, whose error
  // comes back with the opposite sign at every step, and would evaluate a material history at
  // the old time without it.  Such a term, and the pressure gradient, is taken at the new time.
  const auto v = static_cast<std::size_t>(object.variable());
  const bool new_time = object.atNewTime() || (v < _new_time_only.size() && _new_time_only[v]);
  const double w = new_time ? (options.old_level ? 0.0 : 1.0) : options.theta;
  return object.isLoad() ? w * options.data_fraction : w;
}

double
Problem::timeWeight(const Kernel & kernel, const AssemblyOptions & options) const
{
  if (!options.include_time_kernels || options.dt <= 0.0)
    return 0.0;
  const auto v = static_cast<std::size_t>(kernel.variable());
  const bool new_time = kernel.atNewTime() || (v < _new_time_only.size() && _new_time_only[v]);
  const double w = new_time ? 1.0 : options.theta;
  return w > 0.0 ? 1.0 / w : 0.0;
}

bool
Problem::flowAtRest() const
{
  const int nv = numVariables();
  std::vector<char> inertial(static_cast<std::size_t>(nv), 0);
  bool any = false;
  for (const auto & k : _kernels)
    if (k->isInertia())
    {
      inertial[static_cast<std::size_t>(k->variable())] = 1;
      any = true;
    }
  if (!any)
    return false;
  const std::vector<char> fixed = constrainedDofs();
  bool moving = false;
  for (Index i = 0; i < numDofs() && !moving; ++i)
    moving = inertial[static_cast<std::size_t>(i % nv)] && !fixed[i] && _U[i] != 0.0;
  return !parallel().any(moving);
}

void
Problem::checkAdvectiveBoundary() const
{
  // In a serial run every exterior side must be in a side set, or the advective flux would not
  // leave through it (a part of a distributed run has exterior sides on its interfaces, which
  // are rightly in no side set).
  if (_parallel || _advective_boundary_checked)
    return;
  std::set<Side> named;
  for (const auto & [name, list] : _mesh->sidesets())
    named.insert(list.begin(), list.end());
  for (const Side & side : _mesh->exteriorSides())
    if (!named.count(side))
      throw InputError("The conservative face flux of an advection scheme leaves the domain "
                       "through its boundary, which is that of the side sets. The side " +
                       std::to_string(side.second) + " of element " + std::to_string(side.first) +
                       " is in none. Give the whole boundary side sets (generated meshes have "
                       "them, and mesh.add_bounding_box_sidesets() adds them to others).");
  _advective_boundary_checked = true;
}

const Kernel *
Problem::flowConstraint() const
{
  const Kernel * constraint = nullptr;
  for (const auto & k : _kernels)
    if (k->isMassConservation())
    {
      if (constraint && constraint->variable() != k->variable())
        throw InputError("The block preconditioner of the flow found two mass conservation "
                         "equations ('" +
                         constraint->name() + "' and '" + k->name() +
                         "') on different variables; it handles one pressure.");
      constraint = k.get();
    }
  return constraint;
}

SaddlePointBlocks
Problem::saddlePointBlocks(const SparseMatrix & J, SchurApproximation approximation) const
{
  const_cast<Problem *>(this)->initialize();
  const Kernel * constraint = flowConstraint();
  QpContext probe;
  probe.dim = _mesh->spaceDimension();
  probe.time = _time;
  probe.x = _mesh->node(0);
  if (!constraint)
    throw InputError("The preconditioners pressure_mass, pcd and lsc are for pressure-velocity "
                     "flow, and this problem has no mass conservation equation "
                     "(mass_conservation). Use another preconditioner.");
  const int p = constraint->variable();
  const int nv = numVariables();
  SaddlePointBlocks blocks;
  blocks.approximation = approximation;
  blocks.pressure_variable = _vars[p].name;
  blocks.constrained = constrainedDofs();
  std::vector<char> is_pressure(numDofs(), 0);
  for (Index d = 0; d < numDofs(); ++d)
    if (d % nv == p)
    {
      is_pressure[d] = 1;
      blocks.pressure_dofs.push_back(d);
    }
    else
      blocks.other_dofs.push_back(d);

  // The sign of S = C - D F^{-1} G relative to G^T F^{-1} G: the Frobenius
  // product <G, D^T> is negative when G = -D^T, and then -D F^{-1} G is
  // G^T F^{-1} G.  It is taken over the velocities only: the other unknowns
  // that the flow couples to the pressure (the temperature) have other units,
  // and would make the sign depend on them.
  FlowCoefficients velocities;
  constraint->flowCoefficients(probe, velocities);
  std::vector<char> is_velocity(static_cast<std::size_t>(nv), 0);
  for (int d = 0; d < velocities.num_velocities; ++d)
    is_velocity[static_cast<std::size_t>(velocities.velocities[d])] = 1;
  double product = 0.0;
  for (Index d : blocks.pressure_dofs)
    for (SparseMatrix::InnerIterator it(J, d); it; ++it)
      if (is_velocity[static_cast<std::size_t>(it.row() % nv)])
        product += it.value() * J.coeff(d, it.row());
  blocks.coupling = product;
  blocks.sign = product < 0.0 ? 1 : -1;

  const bool pcd = approximation == SchurApproximation::ConvectionDiffusion;
  const bool lsc = approximation == SchurApproximation::LeastSquaresCommutator;
  // The time derivative and the weight theta of the steady terms of the
  // equations whose Jacobian is solved (zero and one for a steady solve).
  const double dt = _linear_dt;
  const double theta = _linear_theta;
  const int dim = _mesh->dimension();
  blocks.lumped_mass = Eigen::VectorXd::Zero(numDofs());
  const auto keep = [&](Index a, Index b)
  { return !blocks.constrained[a] && !blocks.constrained[b]; };

  std::vector<IntegrationPoint> points;
  std::vector<MappedPoint> mapped;
  MappedPoint geo;
  QpContext context = probe;
  FlowCoefficients c;
  constexpr int kMax = kMaxElementNodes;
  for (Index e = 0; e < numIntegratedElements(); ++e)
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
    const int np = corner ? el.numCorners() : el.numNodes();
    const int nn = el.numNodes();
    // Element matrices on the pressure nodes: the mass M_k, the integrals m_k
    // of the shape functions, (1/mu) M_k, and F_p; and the diagonal of the
    // velocity mass matrix.
    double M[kMax][kMax] = {}, Mmu[kMax][kMax] = {}, F[kMax][kMax] = {};
    double m[kMax] = {}, velocity_diagonal[kMax] = {};
    double volume = 0.0, area = 0.0, viscosity = 0.0, time_density = 0.0;
    for (std::size_t q = 0; q < points.size(); ++q)
    {
      if (q < mapped.size())
        geo = mapped[q];
      else
        mapPoint(*_mesh, el, points[q].xi, geo);
      if (corner)
        mapCornerShape(el, geo);
      const double * N = corner ? geo.corner_shape : geo.N;
      const Point * dN = corner ? geo.corner_gradient : geo.dN;
      context.x = geo.x;
      context.element = e;
      context.block = el.block;
      constraint->flowCoefficients(context, c);
      if (!(c.viscosity > 0.0))
        blocks.refusal = "The block preconditioners of the flow (pressure_mass, pcd, lsc) scale "
                         "their operators by the dynamic_viscosity, which must be positive "
                         "everywhere.";
      const double w = points[q].weight * coordFactor(geo.x);
      // The measure of the element itself, without the factor of the
      // coordinate system, gives its size.
      area += points[q].weight;
      volume += w;
      viscosity += w * c.viscosity;
      time_density += w * c.time_density;
      // The current velocity, the wind of the convective term.
      Point wind{0.0, 0.0, 0.0};
      for (int d = 0; d < c.num_velocities; ++d)
        for (int a = 0; a < nn; ++a)
          wind[d] += geo.N[a] * _U[dof(el.nodes[a], c.velocities[d])];
      for (int a = 0; a < nn; ++a)
        velocity_diagonal[a] += w * geo.N[a] * geo.N[a];
      const double mass_rate = dt > 0.0 ? c.time_density / dt : 0.0;
      for (int a = 0; a < np; ++a)
      {
        m[a] += w * N[a];
        for (int b = 0; b < np; ++b)
        {
          M[a][b] += w * N[a] * N[b];
          Mmu[a][b] += w * N[a] * N[b] / c.viscosity;
          if (pcd)
          {
            double diffusion = 0.0, convection = 0.0;
            for (int d = 0; d < 3; ++d)
            {
              diffusion += dN[a][d] * dN[b][d];
              convection += wind[d] * dN[b][d];
            }
            F[a][b] += w * (mass_rate * N[a] * N[b] +
                            theta * (c.viscosity * diffusion + c.density * convection * N[a]));
          }
        }
      }
    }
    // Lumped masses, by the diagonal scaled to the element measure (Hinton,
    // Rock and Zienkiewicz), which is positive for quadratic elements too.
    double pressure_trace = 0.0, velocity_trace = 0.0;
    for (int a = 0; a < np; ++a)
      pressure_trace += M[a][a];
    for (int a = 0; a < nn; ++a)
      velocity_trace += velocity_diagonal[a];
    for (int a = 0; a < np; ++a)
      blocks.lumped_mass[dof(el.nodes[a], p)] += M[a][a] * volume / pressure_trace;
    for (int d = 0; d < c.num_velocities; ++d)
      for (int a = 0; a < nn; ++a)
        blocks.lumped_mass[dof(el.nodes[a], c.velocities[d])] +=
            velocity_diagonal[a] * volume / velocity_trace;
    // The stabilisation of A_p and of the middle factor of lsc for an
    // equal-order element: the L2 distance from the element mean, scaled.
    const bool stabilized = c.stabilized;
    const double h2 = std::pow(area, 2.0 / dim);
    viscosity /= volume;
    time_density /= volume;
    // pcd applies A_p^{-1} to F_p Q_p^{-1} r, which holds the spurious modes at
    // full size, and needs C_1 a hundred times larger than lsc does (see
    // SaddlePointBlocks.h).
    const double c1 = (pcd ? 100.0 : 1.0) / h2;
    const double c2 = theta * viscosity / (h2 * h2) + (dt > 0.0 ? time_density / (dt * h2) : 0.0);
    for (int a = 0; a < np; ++a)
      for (int b = 0; b < np; ++b)
      {
        const Index da = dof(el.nodes[a], p), db = dof(el.nodes[b], p);
        if (!keep(da, db))
          continue;
        if (approximation == SchurApproximation::PressureMass)
          blocks.mass.emplace_back(da, db, Mmu[a][b]);
        if (pcd)
          blocks.convection_diffusion.emplace_back(da, db, F[a][b]);
        const double distance = M[a][b] - m[a] * m[b] / volume;
        if ((pcd || lsc) && stabilized)
          blocks.stabilization.emplace_back(da, db, c1 * distance);
        if (lsc && stabilized)
          blocks.commutator_stabilization.emplace_back(da, db, c2 * distance);
      }
  }

  if (pcd)
    addInflowRobin(*constraint, theta, blocks);
  return blocks;
}

void
Problem::addInflowRobin(const Kernel & constraint, double theta, SaddlePointBlocks & blocks) const
{
  // The Robin condition of F_p, mu dp/dn = rho (w . n) p, on the sides where
  // every velocity is prescribed: it adds theta rho max(0, -w . n) (p, q) on
  // the side, which is zero on a wall and acts at an inflow only.
  const int p = constraint.variable();
  QpContext context;
  context.dim = _mesh->spaceDimension();
  context.time = _time;
  FlowCoefficients c;
  std::vector<IntegrationPoint> points;
  MappedPoint side_geo;
  for (const Side & side : _mesh->exteriorSides())
  {
    if (side.first >= numIntegratedElements())
      continue;
    const Element & el = _mesh->element(side.first);
    if (!constraint.activeOnBlock(el.block))
      continue;
    const auto & vb = _vars[p].blocks;
    if (!vb.empty() && !vb.count(el.block))
      continue;
    context.element = side.first;
    context.block = el.block;
    context.x = _mesh->sideCentroid(side);
    constraint.flowCoefficients(context, c);
    bool prescribed = true;
    for (Index node : _mesh->sideNodes(side))
      for (int d = 0; d < c.num_velocities; ++d)
        prescribed = prescribed && blocks.constrained[dof(node, c.velocities[d])];
    if (!prescribed)
      continue;
    QuadratureSpec spec;
    spec.kind = QuadratureSpec::Kind::Gauss;
    spec.points = ReferenceElement::get(el.type).order() + 2;
    buildSidePoints(*_mesh, side, false, spec, points);
    const bool corner = usesCornerShape(el, p);
    const int np = corner ? el.numCorners() : el.numNodes();
    for (const auto & ip : points)
    {
      mapPoint(*_mesh, el, ip.xi, side_geo);
      if (corner)
        mapCornerShape(el, side_geo);
      context.x = side_geo.x;
      constraint.flowCoefficients(context, c);
      double inflow = 0.0;
      for (int d = 0; d < c.num_velocities; ++d)
      {
        double w_d = 0.0;
        for (int a = 0; a < el.numNodes(); ++a)
          w_d += side_geo.N[a] * _U[dof(el.nodes[a], c.velocities[d])];
        inflow -= w_d * ip.area[d];
      }
      if (!(inflow > 0.0))
        continue;
      const double * N = corner ? side_geo.corner_shape : side_geo.N;
      const double w = theta * c.density * inflow * coordFactor(side_geo.x);
      for (int a = 0; a < np; ++a)
        for (int b = 0; b < np; ++b)
        {
          const Index da = dof(el.nodes[a], p), db = dof(el.nodes[b], p);
          if (!blocks.constrained[da] && !blocks.constrained[db] && N[a] * N[b] != 0.0)
            blocks.convection_diffusion.emplace_back(da, db, w * N[a] * N[b]);
        }
    }
  }
}

std::vector<std::vector<int>>
Problem::displacementGroups() const
{
  const int nv = numVariables();
  std::vector<std::vector<int>> groups;
  std::vector<char> grouped(nv, 0);
  for (const auto & material : _property_objects)
  {
    const auto & p = material->parameters();
    if (!p.has("displacements") || !p.isSet("displacements"))
      continue;
    std::vector<int> group;
    for (const auto & name : p.getStringList("displacements"))
      for (int v = 0; v < nv; ++v)
        if (_vars[v].name == name)
          group.push_back(v);
    if (group.empty() || std::any_of(group.begin(), group.end(), [&](int v) { return grouped[v]; }))
      continue;
    for (int v : group)
      grouped[v] = 1;
    groups.push_back(group);
  }
  return groups;
}

std::vector<std::vector<int>>
Problem::physicsBlocks() const
{
  if (flowConstraint())
    throw InputError("The split by physics (-dualmesh_split physics) solves each block with "
                     "algebraic multigrid, which does not apply to the velocity and pressure of "
                     "a flow. Use the Schur complement split of the flow (-dualmesh_schur lsc or "
                     "pcd), which takes the other unknowns (a temperature, for instance) into its "
                     "first block.");
  std::vector<std::vector<int>> blocks = displacementGroups();
  std::vector<char> grouped(numVariables(), 0);
  for (const auto & group : blocks)
    for (int v : group)
      grouped[v] = 1;
  for (int v = 0; v < numVariables(); ++v)
    if (!grouped[v])
      blocks.push_back({v});
  // In the order of the variables, which is the order in which the physics were added: a
  // coupling that one physics drives (a temperature that expands a solid) is then solved first.
  std::sort(blocks.begin(),
            blocks.end(),
            [](const std::vector<int> & a, const std::vector<int> & b) {
              return *std::min_element(a.begin(), a.end()) < *std::min_element(b.begin(), b.end());
            });
  return blocks;
}

std::vector<std::vector<Index>>
Problem::physicsBlockDofs() const
{
  const std::vector<std::vector<int>> groups = physicsBlocks();
  std::vector<int> block_of(numVariables(), 0);
  for (std::size_t k = 0; k < groups.size(); ++k)
    for (int v : groups[k])
      block_of[v] = static_cast<int>(k);
  std::vector<std::vector<Index>> blocks(groups.size());
  for (Index d = 0; d < numDofs(); ++d)
    blocks[block_of[d % numVariables()]].push_back(d);
  return blocks;
}

Eigen::MatrixXd
Problem::nearNullspace() const
{
  const int nv = numVariables();
  const Index entities = _mesh->numNodes();
  const int dim = _mesh->spaceDimension();
  // The displacement variables of the elastic property objects, in component order.
  const std::vector<std::vector<int>> groups = displacementGroups();
  std::vector<char> grouped(nv, 0);
  for (const auto & group : groups)
    for (int v : group)
      grouped[v] = 1;
  const bool cartesian = _coord == CoordinateSystem::Cartesian;
  int columns = 0;
  for (const auto & group : groups)
  {
    const int d = static_cast<int>(group.size());
    columns += d + (cartesian && d == 2 && dim >= 2 ? 1 : 0) + (cartesian && d == 3 ? 3 : 0);
  }
  for (int v = 0; v < nv; ++v)
    if (!grouped[v])
      ++columns;
  Eigen::MatrixXd B = Eigen::MatrixXd::Zero(numDofs(), columns);
  // Rotations about the centroid of the entities, for a well-conditioned basis.
  Point centre{0.0, 0.0, 0.0};
  for (Index e = 0; e < entities; ++e)
    centre = centre + _mesh->node(e);
  centre = (1.0 / std::max<Index>(entities, 1)) * centre;
  int column = 0;
  for (const auto & group : groups)
  {
    const int d = static_cast<int>(group.size());
    for (int c = 0; c < d; ++c, ++column)
      for (Index e = 0; e < entities; ++e)
        B(dof(e, group[c]), column) = 1.0;
    if (!cartesian)
      continue;
    if (d == 2 && dim >= 2)
    {
      for (Index e = 0; e < entities; ++e)
      {
        const Point x = _mesh->node(e) - centre;
        B(dof(e, group[0]), column) = -x[1];
        B(dof(e, group[1]), column) = x[0];
      }
      ++column;
    }
    if (d == 3)
    {
      for (Index e = 0; e < entities; ++e)
      {
        const Point x = _mesh->node(e) - centre;
        B(dof(e, group[0]), column) = -x[1]; // about z
        B(dof(e, group[1]), column) = x[0];
        B(dof(e, group[1]), column + 1) = -x[2]; // about x
        B(dof(e, group[2]), column + 1) = x[1];
        B(dof(e, group[2]), column + 2) = -x[0]; // about y
        B(dof(e, group[0]), column + 2) = x[2];
      }
      column += 3;
    }
  }
  for (int v = 0; v < nv; ++v)
    if (!grouped[v])
    {
      for (Index e = 0; e < entities; ++e)
        B(dof(e, v), column) = 1.0;
      ++column;
    }
  return B;
}

void
Problem::checkSingularity(const SparseMatrix & J) const
{
  const Index n = J.rows();
  const int nv = numVariables();
  const std::vector<char> fixed = constrainedDofs();
  // The groups of coupled unknowns: the connected components of the graph of
  // the Jacobian without the prescribed unknowns (union-find with path halving).
  std::vector<Index> parent(n);
  for (Index i = 0; i < n; ++i)
    parent[i] = i;
  const auto root = [&parent](Index i)
  {
    while (parent[i] != i)
    {
      parent[i] = parent[parent[i]];
      i = parent[i];
    }
    return i;
  };
  for (Index col = 0; col < J.outerSize(); ++col)
  {
    if (fixed[col])
      continue;
    for (SparseMatrix::InnerIterator it(J, col); it; ++it)
      if (!fixed[it.row()] && it.row() != col && it.value() != 0.0)
      {
        const Index a = root(it.row()), b = root(col);
        if (a != b)
          parent[a] = b;
      }
  }
  Eigen::MatrixXd B = nearNullspace();
  for (Index i = 0; i < n; ++i)
    if (fixed[i])
      B.row(i).setZero();
  const Eigen::MatrixXd JB = J * B;
  // For every component and every near null space vector: the norms of the
  // vector and of its image on the component, and the largest diagonal entry.
  std::vector<Index> component(n, -1);
  std::vector<Index> roots;
  for (Index i = 0; i < n; ++i)
    if (!fixed[i])
    {
      const Index r = root(i);
      if (component[r] < 0)
      {
        component[r] = static_cast<Index>(roots.size());
        roots.push_back(r);
      }
      component[i] = component[r];
    }
  const Index nc = static_cast<Index>(roots.size());
  const Eigen::Index k = B.cols();
  // For every component and every near null space vector b: the norms of J b
  // and of D b, D being the diagonal of J, so that the comparison is made row
  // by row in the units of each equation (a temperature equation and an
  // equilibrium equation differ by many orders of magnitude).
  Eigen::MatrixXd scaled_norm = Eigen::MatrixXd::Zero(nc, k),
                  image_norm = Eigen::MatrixXd::Zero(nc, k);
  const Vector diagonal = J.diagonal();
  for (Index i = 0; i < n; ++i)
  {
    if (fixed[i])
      continue;
    const Index c = component[i];
    for (Eigen::Index j = 0; j < k; ++j)
    {
      const double db = diagonal[i] * B(i, j);
      scaled_norm(c, j) += db * db;
      image_norm(c, j) += JB(i, j) * JB(i, j);
    }
  }
  for (Index c = 0; c < nc; ++c)
    for (Eigen::Index j = 0; j < k; ++j)
    {
      const double reference = std::sqrt(scaled_norm(c, j));
      if (!(reference > 0.0) || std::sqrt(image_norm(c, j)) > 1e-9 * reference)
        continue;
      // Describe the part of the mesh and the variables involved.
      Point lo{1e300, 1e300, 1e300}, hi{-1e300, -1e300, -1e300};
      std::set<int> variables;
      std::set<Index> entities;
      for (Index i = 0; i < n; ++i)
        if (!fixed[i] && component[i] == c && B(i, j) != 0.0)
        {
          variables.insert(static_cast<int>(i % nv));
          entities.insert(i / nv);
          const Point & x = _mesh->node(i / nv);
          for (int d = 0; d < 3; ++d)
          {
            lo[d] = std::min(lo[d], x[d]);
            hi[d] = std::max(hi[d], x[d]);
          }
        }
      std::ostringstream os;
      os << "The system is singular: ";
      std::string names;
      for (int v : variables)
        names += (names.empty() ? "'" : ", '") + _vars[v].name + "'";
      os << (variables.size() > 1 ? "the variables " : "the variable ") << names
         << " on a part of the mesh with " << entities.size()
         << (entities.size() == 1 ? " node" : " nodes") << " (";
      const char * axes = "xyz";
      for (int d = 0; d < _mesh->spaceDimension(); ++d)
        os << (d ? ", " : "") << axes[d] << " from " << lo[d] << " to " << hi[d];
      os << ") " << (variables.size() > 1 ? "are" : "is")
         << " not held by any boundary condition. A "
         << (variables.size() > 1 ? "rigid body motion" : "constant")
         << " can be added there without changing the equations, so the solution is not unique. "
            "Give this part a Dirichlet condition (for a solid, the word fixed), or connect it to the rest "
            "of the mesh, e.g., by checking that the parts of an imported mesh share their nodes.";
      throw InputError(os.str());
    }
}

const char *
Problem::enclosedLevelMessage()
{
  return "The pressure of this enclosed flow enters its equations (through a viscosity, a density "
         "or a source that depends on it), so its level changes the flow, and no boundary fixes "
         "it. Give the pressure at a point, flow.add_boundary_condition(\"pressure = 0\", "
         "point=[...]), or a boundary with a pressure or traction condition.";
}

bool
Problem::holdPressureLevels(const SparseMatrix & J)
{
  const Parallel & par = parallel();
  const std::vector<char> fixed = constrainedDofs();
  for (const auto & k : _kernels)
  {
    if (!k->isMassConservation())
      continue;
    PressureLevel level;
    level.variable = k->variable();
    const int p = level.variable;
    std::vector<Index> free;
    for (Index n = 0; n < _mesh->numNodes(); ++n)
      if (!fixed[dof(n, p)])
        free.push_back(n);
    // The constant pressure, one at every free unknown, its image under J against that under |J|
    // for the scale, and the sum of the mass equations (J^T 1).
    Vector e = Vector::Zero(numDofs());
    for (Index n : free)
      e[dof(n, p)] = 1.0;
    const SparseMatrix size = J.cwiseAbs();
    Vector image = J * e, scale = size * e;
    Vector rows = J.transpose() * e, row_scale = size.transpose() * e;
    for (Vector * v : {&image, &scale, &rows, &row_scale})
      par.addAtNodes(v->data(), numVariables());
    // Each equation is compared with itself, in its own units: the heat that a flow carries
    // depends on the pressure through the conserved velocity, in numbers that would hide the
    // momentum equations.
    level.scales = equationNorms(scale);
    if (std::none_of(level.scales.begin(), level.scales.end(), [](double s) { return s > 0.0; }))
      continue;
    const std::vector<double> images = equationNorms(image);
    bool changes = false;
    for (std::size_t v = 0; v < images.size(); ++v)
      changes = changes || images[v] > 1e-9 * level.scales[v];
    if (changes)
    {
      // The level changes the equations.  The mass equations of an enclosed flow sum to its
      // boundary flux whatever the solution (J^T 1 = 0, in the columns of every variable), so
      // they do not fix the level either.
      const std::vector<double> sums = equationNorms(rows), sum_scales = equationNorms(row_scale);
      bool enclosed = true;
      for (std::size_t v = 0; v < sums.size(); ++v)
        enclosed = enclosed && sums[v] <= 1e-9 * sum_scales[v];
      if (enclosed)
      {
        _levels.clear();
        throw InputError(enclosedLevelMessage());
      }
      continue;
    }
    // Hold the unknown nearest the centre of the free unknowns: one in a corner slows the Krylov
    // solvers by about 40 % (the cavity with the pressure mass and pcd preconditioners).
    Point centre{0, 0, 0};
    for (Index n : free)
      centre = centre + _mesh->node(n);
    const double count = par.sum(static_cast<double>(free.size()));
    for (int c = 0; c < 3; ++c)
      centre[c] = par.sum(centre[c]) / std::max(count, 1.0);
    std::vector<char> candidate(static_cast<std::size_t>(_mesh->numNodes()), 0);
    for (Index n : free)
      candidate[static_cast<std::size_t>(n)] = par.globalDof(dof(n, p)) >= 0 ? 1 : 0;
    double distance = 0.0;
    const Index held = nearestNode(
        centre, [&](Index n) { return candidate[static_cast<std::size_t>(n)] != 0; }, &distance);
    if (!std::isfinite(distance))
      continue;
    if (held >= 0)
      level.dof = dof(held, p);
    level.weight = nodeWeights(p);
    _levels.push_back(std::move(level));
  }
  _levels_checked = true;
  return !_levels.empty();
}

Index
Problem::nearestNode(const Point & x,
                     const std::function<bool(Index)> & accept,
                     double * distance) const
{
  const Parallel & par = parallel();
  double nearest = std::numeric_limits<double>::infinity();
  for (Index n = 0; n < _mesh->numNodes(); ++n)
    if (accept(n))
      nearest = std::min(nearest, norm(_mesh->node(n) - x));
  nearest = par.minimum(nearest);
  if (distance)
    *distance = nearest;
  double first = std::numeric_limits<double>::infinity();
  for (Index n = 0; n < _mesh->numNodes(); ++n)
    if (accept(n) && norm(_mesh->node(n) - x) <= nearest * (1.0 + 1e-12) + 1e-14)
      first = std::min(first, static_cast<double>(par.globalNode(n)));
  first = par.minimum(first);
  for (Index n = 0; n < _mesh->numNodes(); ++n)
    if (accept(n) && static_cast<double>(par.globalNode(n)) == first)
      return n;
  return -1;
}

Vector
Problem::nodeWeights(int v) const
{
  // The points of the nodal rule: one per node of each element, weighted by the measure of the
  // control domain of the node in the element (DMCDM) or the integral of its shape function (FEM).
  QuadratureSpec nodal;
  nodal.kind = QuadratureSpec::Kind::Nodal;
  const bool dual = isDual();
  const auto * blocks = v >= 0 ? &_vars[static_cast<std::size_t>(v)].blocks : nullptr;
  Vector weight = Vector::Zero(_mesh->numNodes());
  std::vector<IntegrationPoint> points;
  MappedPoint mp;
  for (Index e = 0; e < numIntegratedElements(); ++e)
  {
    const Element & el = _mesh->element(e);
    if (blocks && !blocks->empty() && !blocks->count(el.block))
      continue;
    const PolygonCell * polygon = polygonCell(e);
    buildElementPoints(*_mesh, e, dual, PointSet::Volume, nodal, points, nullptr, polygon);
    for (const auto & ip : points)
    {
      mapPoint(*_mesh, el, ip.xi, mp);
      const Index node = polygon ? polygon->nodes[static_cast<std::size_t>(ip.owner)]
                                 : el.nodes[static_cast<std::size_t>(ip.owner)];
      weight[node] += ip.weight * coordFactor(mp.x);
    }
  }
  return weight;
}

std::vector<double>
Problem::nodeMeasures() const
{
  if (!_initialized)
    throw InputError(
        "The node measures need the problem set up. Solve it, or initialize it, first.");
  Vector measure = nodeWeights(-1);
  // A node shared by several processes takes the elements of all of them.
  parallel().addAtNodes(measure.data(), 1);
  return {measure.data(), measure.data() + measure.size()};
}

std::vector<double>
Problem::shiftPressureLevels(Vector & U) const
{
  const Parallel & par = parallel();
  std::vector<double> shifts;
  for (const auto & level : _levels)
  {
    double integral = 0.0;
    for (Index n = 0; n < _mesh->numNodes(); ++n)
      integral += level.weight[n] * U[dof(n, level.variable)];
    const double shift = -par.sum(integral) / par.sum(level.weight.sum());
    for (Index n = 0; n < _mesh->numNodes(); ++n)
      if (_active_dof[dof(n, level.variable)])
        U[dof(n, level.variable)] += shift;
    shifts.push_back(shift);
  }
  updateDependentDofs(U);
  return shifts;
}

std::vector<double>
Problem::testShiftPressureLevels(Vector & U, const std::vector<double> & forces) const
{
  const Parallel & par = parallel();
  std::vector<double> shifts;
  for (const auto & level : _levels)
  {
    double largest = 0.0;
    for (Index n = 0; n < _mesh->numNodes(); ++n)
      if (_active_dof[dof(n, level.variable)])
        largest = std::max(largest, std::abs(U[dof(n, level.variable)]));
    double shift = par.maximum(largest);
    for (std::size_t v = 0; v < forces.size(); ++v)
      if (level.scales[v] > 0.0)
        shift = std::max(shift, forces[v] / level.scales[v]);
    if (!(shift > 0.0))
      shift = 1.0;
    for (Index n = 0; n < _mesh->numNodes(); ++n)
      if (_active_dof[dof(n, level.variable)])
        U[dof(n, level.variable)] += shift;
    shifts.push_back(shift);
  }
  updateDependentDofs(U);
  return shifts;
}

std::vector<double>
Problem::pressureLevelChange(const std::vector<double> & shifts) const
{
  std::vector<double> change(static_cast<std::size_t>(numVariables()), 0.0);
  for (std::size_t l = 0; l < _levels.size(); ++l)
    for (std::size_t v = 0; v < change.size(); ++v)
      change[v] += std::abs(shifts[l]) * _levels[l].scales[v];
  return change;
}

std::vector<Side>
Problem::boundarySides(const std::vector<std::string> & names) const
{
  std::vector<Side> sides;
  std::set<Side> seen;
  for (const auto & name : names)
    for (const Side & side : _mesh->sideset(name))
      if (seen.insert(side).second)
        sides.push_back(side);
  return sides;
}

std::vector<double>
Problem::equationNorms(const Vector & v) const
{
  const int nv = numVariables();
  std::vector<double> norms(static_cast<std::size_t>(nv));
  Vector part = Vector::Zero(v.size());
  for (int var = 0; var < nv; ++var)
  {
    part.setZero();
    for (Index n = 0; n < _mesh->numNodes(); ++n)
      part[dof(n, var)] = v[dof(n, var)];
    norms[static_cast<std::size_t>(var)] = parallel().norm(part);
  }
  return norms;
}

Problem::VariableSizes
Problem::variableSizes(const Vector & state, const Vector & reference) const
{
  const Parallel & par = parallel();
  const std::size_t nv = static_cast<std::size_t>(numVariables());
  VariableSizes sizes;
  sizes.range.assign(nv, 0.0);
  sizes.change.assign(nv, 0.0);
  sizes.magnitude.assign(nv, 0.0);
  for (std::size_t var = 0; var < nv; ++var)
  {
    double low = std::numeric_limits<double>::infinity(), high = -low;
    for (Index n = 0; n < _mesh->numNodes(); ++n)
    {
      const Index d = dof(n, static_cast<int>(var));
      if (!_active_dof[d])
        continue;
      low = std::min(low, state[d]);
      high = std::max(high, state[d]);
      sizes.magnitude[var] = std::max(sizes.magnitude[var], std::abs(state[d]));
      sizes.change[var] = std::max(sizes.change[var], std::abs(state[d] - reference[d]));
    }
    high = par.maximum(high);
    low = par.minimum(low);
    sizes.range[var] = high > low ? high - low : 0.0;
    sizes.magnitude[var] = par.maximum(sizes.magnitude[var]);
    sizes.change[var] = par.maximum(sizes.change[var]);
  }
  return sizes;
}

double
Problem::relativeStep(const Vector & after, const Vector & before) const
{
  const VariableSizes sizes = variableSizes(after, before);
  double largest = 0.0;
  for (std::size_t var = 0; var < sizes.change.size(); ++var)
  {
    if (!(sizes.change[var] > 0.0))
      continue;
    const double range = sizes.range[var] > 0.0 ? sizes.range[var] : sizes.magnitude[var];
    largest = std::max(largest,
                       range > 0.0 ? sizes.change[var] / range
                                   : std::numeric_limits<double>::infinity());
  }
  return largest;
}

void
Problem::notePressureLevelFlux()
{
  const Parallel & par = parallel();
  const std::vector<char> fixed = constrainedDofs();
  for (auto & level : _levels)
  {
    Vector flux = Vector::Zero(numDofs());
    for (Index n = 0; n < _mesh->numNodes(); ++n)
      if (!fixed[dof(n, level.variable)])
        flux[dof(n, level.variable)] = _last_residual[dof(n, level.variable)];
    level.flux = std::max(level.flux, par.norm(flux));
  }
}

void
Problem::checkPressureLevelBalance(const Vector & R)
{
  const Parallel & par = parallel();
  for (auto & level : _levels)
  {
    const double imbalance = par.maximum(level.dof >= 0 ? std::abs(R[level.dof]) : 0.0);
    if (level.warned || !(level.flux > 0.0) || imbalance <= 1e-6 * level.flux)
      continue;
    level.warned = true;
    if (par.isRoot())
      std::fprintf(stderr,
                   "dualmesh: warning: the velocity prescribed on the boundary of the enclosed "
                   "flow of '%s' carries a net flux of %.3g (%.2g of its boundary flux). An "
                   "incompressible flow cannot take it in, so it goes to the mass equation of "
                   "the one pressure unknown that fixes the pressure level, and the pressure has "
                   "a spike there. Prescribe a velocity whose net flux is zero, or give the flow "
                   "an outlet.\n",
                   _vars[level.variable].name.c_str(),
                   imbalance,
                   imbalance / level.flux);
  }
}

// ---------------------------------------------------------------------------
// The serial identity of the parallel interface
// ---------------------------------------------------------------------------
CheckpointLayout
Parallel::checkpointLayout(const Problem & problem) const
{
  // One process holds the whole problem, numbered as it is.
  CheckpointLayout layout;
  layout.num_global_nodes = problem.mesh().numNodes();
  layout.global_nodes.resize(static_cast<std::size_t>(problem.mesh().numNodes()));
  std::iota(layout.global_nodes.begin(), layout.global_nodes.end(), 0);
  layout.num_global_elements = problem.mesh().numElements();
  layout.global_elements.resize(static_cast<std::size_t>(problem.mesh().numElements()));
  std::iota(layout.global_elements.begin(), layout.global_elements.end(), 0);
  return layout;
}

const Parallel &
Problem::parallel() const
{
  static const Parallel serial;
  return _parallel ? *_parallel : serial;
}

SolveResult
Problem::solveSteady(const SolverOptions & options)
{
  initialize();
  AssemblyOptions base;
  base.include_time_kernels = false;
  base.theta = 1.0;
  // A flow at rest starts from the Stokes solution, the same problem without the inertia.  The
  // linearisation about the state at rest, whose prescribed velocities next to zero values make
  // steep gradients, can send Newton's method far from the solution; the Stokes solution is the
  // usual first guess of the Navier-Stokes equations.  The equations are unchanged.
  SolveResult start;
  Vector initial = _U;
  if (options.nonlinear_solver != "linear" && flowAtRest())
  {
    if (options.verbose && parallel().isRoot())
      std::cout << "  The flow is at rest: Newton's method starts from the Stokes solution, the "
                   "same problem without the inertia.\n";
    const Vector at_rest = _U;
    AssemblyOptions stokes = base;
    stokes.inertia = 0.0;
    SolverOptions quiet = options;
    quiet.error_on_divergence = false;
    // The Schur complement of a Stokes problem is spectrally equivalent to the pressure mass
    // matrix, so pressure_mass is its Schur approximation.  pcd and lsc approximate the
    // convection that the Stokes problem does not have.
    const user_options_parts user = user_choice(options.petsc_options);
    if (user.split.Schur && *user.split.Schur != SchurApproximation::PressureMass)
      quiet.petsc_options = user.linear + " " + user.nonlinear + " -dualmesh_schur pressure_mass";
    start = nonlinearSolve(quiet, stokes, nullptr);
    if (!start.converged)
      _U = at_rest;
    initial = at_rest;
  }
  // A continuation ramps the prescribed values from the state the solve was given (a flow from
  // rest, which takes it through the lower Reynolds numbers).
  SolveResult result = nonlinearSolve(options, base, nullptr, &initial);
  result.total_iterations += start.total_iterations;
  result.linear_iterations += start.linear_iterations;
  result.history.insert(result.history.begin(), start.history.begin(), start.history.end());
  // A converged steady state is a state the history of stateful property objects
  // has reached: commit it, so that a following solve or time step starts
  // from it.  Without this, a transient run after a steady solve would
  // restart the history from the undeformed state and take the whole
  // steady deformation as its first increment.
  if (result.converged)
    commitState();
  return result;
}

} // namespace dualmesh
