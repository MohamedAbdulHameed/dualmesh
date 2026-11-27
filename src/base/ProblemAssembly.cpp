// SPDX-License-Identifier: LGPL-2.1-or-later
//
// Residual and Jacobian assembly, the prescribed values, and the checks of the assembled system.
#include "dualmesh/base/Problem.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#include <algorithm>
#include <cmath>
#include <exception>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
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
/// The flux of kernel @p k through the interface @p ip of two control domains, times @p scale,
/// into @p ro and @p rn: out of the control domain of the owner (-F.n) and into that of the
/// neighbour, with the sizes of the two terms added to @p ro_size and @p rn_size (from the term
/// sizes when @p sizes).  @p per_side: the flux differs between the two sides, as a
/// non-conservative face flux of an advective kernel does
/// (AdvectiveKernel::nonConservativeFaceFlux), and rn takes that of the neighbour's equation.
void
faceFlux(QpContext & ctx,
         const Kernel & k,
         const IntegrationPoint & ip,
         int dim,
         double scale,
         bool per_side,
         bool sizes,
         ADReal & ro,
         ADReal & rn,
         double & ro_size,
         double & rn_size)
{
  ADVector3 F{ADReal(0.0), ADReal(0.0), ADReal(0.0)};
  if (per_side)
    ctx.reference_side = 0;
  k.computeFlux(ctx, F);
  ADVector3 FN = F;
  if (per_side)
  {
    ctx.reference_side = 1;
    FN = {ADReal(0.0), ADReal(0.0), ADReal(0.0)};
    k.computeFlux(ctx, FN);
    ctx.reference_side = -1;
  }
  double F_size[3] = {0.0, 0.0, 0.0}, FN_size[3] = {0.0, 0.0, 0.0};
  if (sizes)
  {
    k.fluxTermSize(ctx, F, F_size);
    if (per_side)
      k.fluxTermSize(ctx, FN, FN_size);
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
}
} // namespace

int
Problem::effectiveThreads()
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
  checkAssemblyOptions(U, opts, R);
  const Index ndof = numDofs();
  const bool want_jac = J != nullptr;
  R.setZero(ndof);
  if (opts.reference)
    opts.reference->setZero(ndof);
  if (opts.term_size)
    opts.term_size->setZero(ndof);
  const int nthreads = effectiveThreads();
  std::vector<ThreadScratch> scratch = assemblyScratch(opts, nthreads, want_jac);
  // The kernels of the assembly in groups of one rule: the problem's, or those of the options.
  const std::vector<Group> chosen =
      opts.kernels ? kernelGroups(*opts.kernels) : std::vector<Group>{};
  const std::vector<Group> & groups = opts.kernels ? chosen : _groups;
  const AssemblyPass pass{U, opts, R, want_jac, nthreads > 1};

  assembleElements(pass, groups, scratch);
  // The boundary terms are a small fraction of the work and are assembled serially, in the first
  // thread's scratch space.
  assembleConditions(pass, scratch[0]);
  if (isDual() && opts.include_steady_terms && !opts.kernels)
    assembleAdvectiveBoundary(pass, scratch[0]);
  if (opts.include_steady_terms && !opts.kernels)
    assembleNodalLoads(pass);
  if (want_jac)
    mergeJacobian(scratch, *J);
}

void
Problem::checkAssemblyOptions(const Vector & U,
                              const AssemblyOptions & opts,
                              const Vector & R) const
{
  const auto fail = [](const std::string & rule)
  { throw std::logic_error("dualmesh: internal error: the options of an assembly " + rule + "."); };
  const Index ndof = numDofs();
  const std::pair<const Vector *, const char *> states[] = {{&U, "the solution"},
                                                            {opts.lagged, "lagged"},
                                                            {opts.old, "old"},
                                                            {opts.rate_shift, "rate_shift"},
                                                            {opts.accel_shift, "accel_shift"}};
  for (const auto & [state, name] : states)
    if (state && state->size() != ndof)
      fail("give " + std::string(name) + " with " + std::to_string(state->size()) +
           " entries for " + std::to_string(ndof) + " degrees of freedom");
  // A vector the assembly writes is zeroed first, so it must be no other vector of the assembly.
  const std::pair<const Vector *, const char *> outputs[] = {
      {&R, "the residual"}, {opts.reference, "reference"}, {opts.term_size, "term_size"}};
  for (std::size_t i = 0; i < std::size(outputs); ++i)
  {
    const auto & [output, name] = outputs[i];
    if (!output)
      continue;
    for (const auto & [state, state_name] : states)
      if (output == state)
        fail("write " + std::string(name) + " into " + state_name + ", which they read");
    for (std::size_t j = i + 1; j < std::size(outputs); ++j)
      if (output == outputs[j].first)
        fail("write " + std::string(name) + " and " + outputs[j].second + " into one vector");
  }
  if (opts.previous_step && !(opts.old_level && opts.old))
    fail("take the previous step without old_level and old");
  if ((opts.rate_shift != nullptr) != (opts.accel_shift != nullptr))
    fail("give one of rate_shift and accel_shift without the other");
  if (opts.rate_shift && !opts.second_order)
    fail("give rate_shift and accel_shift without second_order");
}

std::vector<Problem::ThreadScratch>
Problem::assemblyScratch(const AssemblyOptions & opts, int nthreads, bool want_jac) const
{
  const int nv = numVariables();
  std::vector<ThreadScratch> scratch(nthreads);
  for (auto & s : scratch)
  {
    s.ctx.dim = _mesh->spaceDimension();
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
  return scratch;
}

std::vector<Problem::Group>
Problem::kernelGroups(const std::vector<const Kernel *> & kernels) const
{
  std::vector<Group> groups;
  for (const Kernel * k : kernels)
  {
    const QuadratureSpec spec = studyQuadrature(*k);
    auto it =
        std::find_if(groups.begin(), groups.end(), [&](const Group & g) { return g.spec == spec; });
    if (it == groups.end())
      it = groups.insert(groups.end(), Group{spec, {}, {}});
    it->add(k);
  }
  return groups;
}

const QuadratureSpec &
Problem::studyQuadrature(const Kernel & kernel) const
{
  if (_eigenvalue_study)
    if (const auto * storage = dynamic_cast<const StorageKernel *>(&kernel))
      return storage->eigenvalueStudyQuadrature();
  return kernel.quadrature();
}

void
Problem::assembleElements(const AssemblyPass & pass,
                          const std::vector<Group> & groups,
                          std::vector<ThreadScratch> & scratch) const
{
  // The element loop is the expensive part and it is embarrassingly parallel:
  // every thread keeps its own scratch space and its own list of matrix
  // entries, and the only shared write is the atomic update of the residual.
  // Threading is disabled automatically when any object is defined in Python,
  // because calling back into the interpreter needs the global interpreter
  // lock (see Problem::initialize).
  // The first error of a thread, rethrown as it is, so that its type (InputError) is kept.
  std::exception_ptr thread_error;
#ifdef _OPENMP
  const int nthreads = static_cast<int>(scratch.size());
#pragma omp parallel num_threads(nthreads) if (nthreads > 1)
  {
    ThreadScratch & s = scratch[omp_get_thread_num()];
#pragma omp for schedule(static)
    for (Index e = 0; e < numIntegratedElements(); ++e)
    {
      try
      {
        assembleElement(s, e, pass, groups);
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
    assembleElement(scratch[0], e, pass, groups);
#endif
  if (thread_error)
    std::rethrow_exception(thread_error);
}

void
Problem::assembleElement(ThreadScratch & s,
                         Index e,
                         const AssemblyPass & pass,
                         const std::vector<Group> & groups) const
{
  const AssemblyOptions & opts = pass.opts;
  const bool dual = isDual();
  const Element & el = _mesh->element(e);
  const PolygonCell * polygon = polygonCell(e);
  const int all = gatherLocal(s, el, pass.U, opts, polygon);
  const int nd = pass.want_jac ? all : 0;
  s.ctx.element = e;
  s.ctx.block = el.block;
  s.ctx.on_boundary = false;
  bool any = false;
  for (const auto & group : groups)
  {
    // The kernels of the group that act on this element in this assembly.
    std::vector<const Kernel *> & ks = s.kernels;
    ks.clear();
    s.advective.clear();
    bool has_flux = false, has_source = false;
    for (std::size_t i = 0; i < group.kernels.size(); ++i)
    {
      const Kernel * k = group.kernels[i];
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
      s.advective.push_back(group.advective[i]);
      has_flux = has_flux || k->hasFlux();
      has_source = has_source || k->hasSource();
    }
    if (ks.empty())
      continue;
    any = true;
    const unsigned group_index = static_cast<unsigned>(&group - groups.data());
    // Volume integrals: the control volumes of DMCDM, or the interior of a finite element.
    if (has_source || (!dual && has_flux))
      elementVolume(s, e, polygon, group, group_index, nd, pass);
    // Interfaces between control volumes (DMCDM only).
    if (dual && has_flux)
      controlDomainFaces(s, e, polygon, group, group_index, nd, pass);
  }
  if (any)
    scatter(s.Rloc,
            s.dofs,
            s.touched,
            pass.R,
            opts.reference,
            opts.term_size,
            opts.term_size ? &s.sizes : nullptr,
            pass.want_jac ? &s.triplets : nullptr,
            pass.atomic);
}

void
Problem::elementVolume(ThreadScratch & s,
                       Index e,
                       const PolygonCell * polygon,
                       const Group & group,
                       unsigned group_index,
                       int nd,
                       const AssemblyPass & pass) const
{
  const AssemblyOptions & opts = pass.opts;
  const bool dual = isDual();
  const bool mixed = _mixed_order;
  const bool sizes = opts.term_size != nullptr;
  const int nv = numVariables();
  const int dim = _mesh->spaceDimension();
  const Element & el = _mesh->element(e);
  QpContext & ctx = s.ctx;
  buildElementPoints(
      *_mesh, e, dual, PointSet::Volume, group.spec, s.pts, &s.mapped, polygon, _coord);
  for (std::size_t q = 0; q < s.pts.size(); ++q)
  {
    const IntegrationPoint & ip = s.pts[q];
    // The points were mapped when they were built, and evaluatePoint reuses those maps.
    const MappedPoint & geo = evaluatePoint(s,
                                            el,
                                            ip,
                                            q < s.mapped.size() ? &s.mapped[q] : nullptr,
                                            group.spec.reduced,
                                            opts,
                                            nd,
                                            e,
                                            stateKey(0, group_index, 0, static_cast<unsigned>(q)));
    const double cf = pointMeasure(ip, coordFactor(geo.x));
    for (const Kernel * k : s.kernels)
    {
      const int v = k->variable();
      const double w = cf * kernelWeight(*k, opts);
      ctx.time_weight = timeWeight(*k, opts);
      // The test functions of the equation of variable v: the element's
      // shape functions, or its corners' for a first-order variable.
      const bool corner = mixed && usesCornerShape(el, v);
      const double * testN = corner ? geo.corner_shape : geo.N;
      const Point * testdN = corner ? geo.corner_gradient : geo.dN;
      const int ntest = corner ? geo.num_corners : el.numNodes();
      const auto slot = [&](int a) { return mixed ? localDofIndex(el, a, v) : a * nv + v; };
      if (k->hasSource())
      {
        const ADReal S = k->computeSource(ctx);
        const double S_size = sizes ? k->sourceTermSize(ctx, S) : 0.0;
        if (ip.owner >= 0)
        {
          s.Rloc[ip.owner * nv + v].addScaledBy(S, w);
          s.touched[ip.owner * nv + v] = 1;
          if (sizes)
            s.sizes[ip.owner * nv + v] += std::abs(w) * S_size;
        }
        else
          for (int a = 0; a < ntest; ++a)
          {
            const int i = slot(a);
            s.Rloc[i].addScaledBy(S, w * testN[a]);
            s.touched[i] = 1;
            if (sizes)
              s.sizes[i] += std::abs(w * testN[a]) * S_size;
          }
      }
      if (!dual && k->hasFlux() && ip.owner < 0)
      {
        ADVector3 F{ADReal(0.0), ADReal(0.0), ADReal(0.0)};
        k->computeFlux(ctx, F);
        double F_size[3] = {0.0, 0.0, 0.0};
        if (sizes)
          k->fluxTermSize(ctx, F, F_size);
        for (int a = 0; a < ntest; ++a)
        {
          const int i = slot(a);
          ADReal & r = s.Rloc[i];
          for (int d = 0; d < dim; ++d)
            if (testdN[a][d] != 0.0)
            {
              r.addScaledBy(F[d], w * testdN[a][d]);
              if (sizes)
                s.sizes[i] += std::abs(w * testdN[a][d]) * F_size[d];
            }
          s.touched[i] = 1;
        }
      }
    }
  }
}

void
Problem::controlDomainFaces(ThreadScratch & s,
                            Index e,
                            const PolygonCell * polygon,
                            const Group & group,
                            unsigned group_index,
                            int nd,
                            const AssemblyPass & pass) const
{
  const AssemblyOptions & opts = pass.opts;
  const bool sizes = opts.term_size != nullptr;
  const int nv = numVariables();
  const int dim = _mesh->spaceDimension();
  const Element & el = _mesh->element(e);
  const int nn = el.numNodes();
  const auto nodeOf = [&](int slot)
  { return slot < nn ? el.nodes[slot] : polygon->nodes[static_cast<std::size_t>(slot)]; };
  QpContext & ctx = s.ctx;
  buildElementPoints(*_mesh, e, true, PointSet::Faces, group.spec, s.pts, &s.mapped, polygon);
  const bool sides = std::any_of(s.advective.begin(),
                                 s.advective.end(),
                                 [](const AdvectiveKernel * a) { return a && a->usesFaceSides(); });
  for (std::size_t q = 0; q < s.pts.size(); ++q)
  {
    const IntegrationPoint & ip = s.pts[q];
    // The points were mapped when they were built, and evaluatePoint reuses those maps.
    const MappedPoint & geo = evaluatePoint(s,
                                            el,
                                            ip,
                                            q < s.mapped.size() ? &s.mapped[q] : nullptr,
                                            group.spec.reduced,
                                            opts,
                                            nd,
                                            e,
                                            stateKey(1, group_index, 0, static_cast<unsigned>(q)));
    const double cf = coordFactor(geo.x);
    if (sides)
    {
      // The two nodes of the face, for an advective flux, and the normal from the owner to the
      // neighbour.
      setFaceSides(ctx,
                   s.cur,
                   nd,
                   ip.owner,
                   ip.neighbor,
                   _mesh->node(nodeOf(ip.owner)),
                   _mesh->node(nodeOf(ip.neighbor)));
      const double area = std::sqrt(dot(ip.area, ip.area));
      ctx.normal = area > 0.0 ? (1.0 / area) * ip.area : Point{0, 0, 0};
    }
    for (std::size_t i = 0; i < s.kernels.size(); ++i)
    {
      const Kernel * k = s.kernels[i];
      if (!k->hasFlux())
        continue;
      const AdvectiveKernel * a = sides ? s.advective[i] : nullptr;
      const int v = k->variable();
      ctx.time_weight = timeWeight(*k, opts);
      const double scale = cf * kernelWeight(*k, opts);
      if (a && a->needsSideGradients() && !_gradient_offsets.empty())
      {
        sideGradientFaceFlux(s, *a, ip, scale, nodeOf(ip.owner), nodeOf(ip.neighbor), pass);
        continue;
      }
      double ro_size = 0.0, rn_size = 0.0;
      faceFlux(ctx,
               *k,
               ip,
               dim,
               scale,
               a && a->nonConservativeFaceFlux(),
               sizes,
               s.Rloc[ip.owner * nv + v],
               s.Rloc[ip.neighbor * nv + v],
               ro_size,
               rn_size);
      if (sizes)
      {
        s.sizes[ip.owner * nv + v] += ro_size;
        s.sizes[ip.neighbor * nv + v] += rn_size;
      }
      s.touched[ip.owner * nv + v] = 1;
      s.touched[ip.neighbor * nv + v] = 1;
    }
  }
  ctx.on_face = false;
}

void
Problem::sideGradientFaceFlux(ThreadScratch & s,
                              const AdvectiveKernel & kernel,
                              const IntegrationPoint & ip,
                              double scale,
                              Index owner_node,
                              Index neighbor_node,
                              const AssemblyPass & pass) const
{
  // The linear upwind and limited schemes take the gradient at the upstream node, reconstructed
  // from the nodes around it, so the flux depends on those nodes too.  They take the slots after
  // the element's own, and the two rows of the face go out here with those columns.
  const int v = kernel.variable();
  const int dim = _mesh->spaceDimension();
  QpContext & ctx = s.ctx;
  auto & columns = s.columns;
  columns.assign(s.dofs.begin(), s.dofs.end());
  // A slot beyond the derivatives of this build is refused before it is written.
  const auto slotOf = [&](Index node)
  {
    const Index d = dof(node, v);
    for (std::size_t i = 0; i < columns.size(); ++i)
      if (columns[i] == d)
        return static_cast<int>(i);
    if (static_cast<int>(columns.size()) >= kMaxDerivatives)
      throw InputError("The face flux of '" + kernel.name() + "' depends on more than " +
                       std::to_string(kMaxDerivatives) +
                       " unknowns, the derivatives of this build. Use the upwind scheme here.");
    columns.push_back(d);
    return static_cast<int>(columns.size()) - 1;
  };
  for (int side = 0; side < 2; ++side)
  {
    const Index node = side == 0 ? owner_node : neighbor_node;
    ctx.side_grad[side].assign(numVariables(), {ADReal(0.0), ADReal(0.0), ADReal(0.0)});
    ADVector3 & g = ctx.side_grad[side][v];
    const ADReal & ua = ctx.side_u[side][v];
    const auto a = static_cast<std::size_t>(node);
    for (Index i = _gradient_offsets[a]; i < _gradient_offsets[a + 1]; ++i)
    {
      const Index b = _gradient_nodes[static_cast<std::size_t>(i)];
      ADReal ub(pass.U[dof(b, v)]);
      if (pass.want_jac)
      {
        const int slot = slotOf(b);
        ub = ADReal::withZeroDerivatives(pass.U[dof(b, v)], slot + 1);
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
  faceFlux(ctx,
           kernel,
           ip,
           dim,
           scale,
           kernel.nonConservativeFaceFlux(),
           pass.opts.term_size != nullptr,
           ro,
           rn,
           ro_size,
           rn_size);
  ctx.side_grad[0].clear();
  ctx.side_grad[1].clear();
  const int nv = numVariables();
  for (const auto & [row, r, size] :
       {std::tuple<Index, const ADReal *, double>{s.dofs[ip.owner * nv + v], &ro, ro_size},
        std::tuple<Index, const ADReal *, double>{s.dofs[ip.neighbor * nv + v], &rn, rn_size}})
  {
    addToResidual(
        pass.R, pass.opts.reference, pass.opts.term_size, row, r->value(), size, pass.atomic);
    if (pass.want_jac)
    {
      const double * dr = r->derivatives();
      for (int j = 0; j < r->size(); ++j)
        if (dr[j] != 0.0)
          s.triplets.emplace_back(row, columns[static_cast<std::size_t>(j)], dr[j]);
    }
  }
}

void
Problem::setFaceSides(QpContext & ctx,
                      const std::vector<double> & values,
                      int nd,
                      int owner,
                      int neighbor,
                      const Point & x_owner,
                      const Point & x_neighbor) const
{
  const int nv = numVariables();
  const int ends[2] = {owner, neighbor};
  ctx.on_face = true;
  for (int side = 0; side < 2; ++side)
  {
    ctx.side_u[side].resize(nv);
    ctx.side_grad[side].clear();
    ctx.side_x[side] = side == 0 ? x_owner : x_neighbor;
    for (int v = 0; v < nv; ++v)
    {
      const int slot = ends[side] * nv + v;
      ADReal value = ADReal::withZeroDerivatives(values[slot], nd);
      if (nd > 0)
        value.setDerivative(slot, 1.0);
      ctx.side_u[side][v] = value;
    }
  }
  ctx.side_weight = 0.5;
}

void
Problem::assembleConditions(const AssemblyPass & pass, ThreadScratch & s)
{
  const AssemblyOptions & opts = pass.opts;
  if (!opts.include_steady_terms)
    return;
  std::vector<const IntegratedBC *> conditions;
  if (opts.conditions)
    conditions = *opts.conditions;
  else if (!opts.kernels)
    for (const auto & bc : _ibcs)
      conditions.push_back(bc.get());
  std::vector<Eigen::Triplet<double>> * const triplets = pass.want_jac ? &s.triplets : nullptr;
  for (std::size_t index = 0; index < conditions.size(); ++index)
  {
    const IntegratedBC * bc = conditions[index];
    if (bc->isConstraint())
    {
      // A constraint holds at the new time only.
      if (!opts.old_level)
        bc->assembleConstraint(*this, pass.U, opts.old, pass.R, opts.reference, triplets);
      continue;
    }
    const double wfac = steadyWeight(*bc, opts);
    if (const auto * ib = dynamic_cast<const InterfaceBC *>(bc))
    {
      assembleInterface(*ib, pass.U, opts, wfac, pass.R, triplets, s);
      continue;
    }
    for (const Side & side : boundarySides(bc->boundaries()))
    {
      const Element & el = _mesh->element(side.first);
      if (!bc->activeOnBlock(el.block))
        continue;
      const int ld = gatherLocal(s, el, pass.U, opts);
      conditionResidual(
          *bc, static_cast<unsigned>(index), side, wfac, opts, pass.want_jac ? ld : 0, s);
      scatter(s.Rloc,
              s.dofs,
              s.touched,
              pass.R,
              opts.reference,
              opts.term_size,
              nullptr,
              triplets,
              false);
    }
  }
}

void
Problem::assembleAdvectiveBoundary(const AssemblyPass & pass, ThreadScratch & s) const
{
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
  std::vector<const AdvectiveKernel *> advective;
  for (const auto & k : _kernels)
    if (const auto * a = dynamic_cast<const AdvectiveKernel *>(k.get()); a && a->usesFaceSides())
      advective.push_back(a);
  if (advective.empty())
    return;
  checkAdvectiveBoundary();
  std::set<Side> sides;
  for (const auto & [name, list] : _mesh->sidesets())
    for (const Side & side : list)
      if (side.first < numIntegratedElements())
        sides.insert(side);
  for (const Side & side : sides)
  {
    const Element & el = _mesh->element(side.first);
    for (const AdvectiveKernel * k : advective)
    {
      if (!k->activeOnBlock(el.block))
        continue;
      const auto & vb = _vars[k->variable()].blocks;
      if (!vb.empty() && !vb.count(el.block))
        continue;
      advectiveBoundaryFlux(*k, side, pass, s);
    }
  }
}

void
Problem::advectiveBoundaryFlux(const AdvectiveKernel & kernel,
                               const Side & side,
                               const AssemblyPass & pass,
                               ThreadScratch & s) const
{
  const AssemblyOptions & opts = pass.opts;
  const int nv = numVariables();
  const int v = kernel.variable();
  const Element & el = _mesh->element(side.first);
  QpContext & ctx = s.ctx;
  const double wfac = steadyWeight(kernel, opts);
  const int ld = gatherLocal(s, el, pass.U, opts);
  const int nd = pass.want_jac ? ld : 0;
  ctx.element = side.first;
  ctx.block = el.block;
  ctx.on_boundary = true;
  buildSidePoints(*_mesh, side, true, kernel.quadrature(), s.pts, _coord);
  for (const auto & ip : s.pts)
  {
    ctx.normal = (1.0 / ip.weight) * ip.area;
    const MappedPoint & geo =
        evaluatePoint(s, el, ip, nullptr, kernel.quadrature().reduced, opts, nd, -1, 0);
    // Both sides are the owner node, so that every scheme takes its value.
    const Point & x = _mesh->node(el.nodes[ip.owner]);
    setFaceSides(ctx, s.cur, nd, ip.owner, ip.owner, x, x);
    ctx.reference_side = kernel.nonConservativeFaceFlux() ? 0 : -1;
    ADVector3 F{ADReal(0.0), ADReal(0.0), ADReal(0.0)};
    kernel.computeFlux(ctx, F);
    ctx.reference_side = -1;
    ADReal Fn(0.0);
    for (int d = 0; d < _mesh->spaceDimension(); ++d)
      Fn += F[d] * ctx.normal[d];
    const double w = pointMeasure(ip, coordFactor(geo.x)) * wfac;
    s.Rloc[ip.owner * nv + v].addScaledBy(Fn, -w);
    s.touched[ip.owner * nv + v] = 1;
  }
  ctx.on_face = false;
  scatter(s.Rloc,
          s.dofs,
          s.touched,
          pass.R,
          opts.reference,
          opts.term_size,
          nullptr,
          pass.want_jac ? &s.triplets : nullptr,
          false);
}

void
Problem::assembleNodalLoads(const AssemblyPass & pass) const
{
  for (const auto & load : _loads)
  {
    const int v = load->variable();
    const double wfac = steadyWeight(*load, pass.opts);
    for (Index n : load->nodes())
    {
      if (!_owned_node.empty() && !_owned_node[n])
        continue;
      const double value = -wfac * load->computeValue(_mesh->node(n), _time);
      addToResidual(pass.R,
                    pass.opts.reference,
                    pass.opts.term_size,
                    dof(n, v),
                    value,
                    std::abs(value),
                    false);
    }
  }
}

void
Problem::mergeJacobian(const std::vector<ThreadScratch> & scratch, SparseMatrix & J) const
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
  J.resize(numDofs(), numDofs());
  J.setFromTriplets(trip.begin(), trip.end());
}

double
Problem::kernelWeight(const Kernel & kernel, const AssemblyOptions & options) const
{
  if (!kernel.isTimeKernel())
    return steadyWeight(kernel, options);
  return kernel.isLoad() ? options.data_fraction : 1.0;
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
  fillContext(s.ctx, el, *fp, s.cur, lagged ? &s.lag : nullptr, opts.old ? &s.old : nullptr, nd);
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
  buildSidePoints(*_mesh, side, isDual(), bc.quadrature(), s.pts, _coord);
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
    const ADReal flux =
        bc.computeBoundaryFlux(s.ctx) * (pointMeasure(ip, coordFactor(geo.x)) * wfac);
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
Problem::setInterfaceValues(QpContext & ctx,
                            const Element & el,
                            const MappedPoint & geo,
                            const Element & sel,
                            const MappedPoint & sgeo,
                            const Vector & U,
                            const Vector * old,
                            int nd) const
{
  const int nv = numVariables();
  const int dim = _mesh->spaceDimension();
  ctx.u.resize(nv);
  ctx.grad_u.resize(nv);
  ctx.u_other.resize(nv);
  ctx.u_old.assign(nv, 0.0);
  ctx.grad_u_old.assign(nv, Point{0, 0, 0});
  for (int v = 0; v < nv; ++v)
  {
    const auto current = [&](int a) { return U[dof(el.nodes[a], v)]; };
    const auto previous = [&](int a) { return (*old)[dof(el.nodes[a], v)]; };
    const auto paired = [&](int b) { return U[dof(sel.nodes[b], v)]; };
    Point g;
    const double val = interpolate(geo.N, geo.dN, el.numNodes(), dim, current, g);
    ctx.u[v] = ADReal::withZeroDerivatives(val, nd);
    ctx.u_other[v] = ADReal::withZeroDerivatives(interpolate(sgeo.N, sel.numNodes(), paired), nd);
    ctx.grad_u[v] = {ADReal(g[0]), ADReal(g[1]), ADReal(g[2])};
    ctx.u_old[v] = old ? interpolate(geo.N, el.numNodes(), previous) : val;
    ctx.grad_u_old[v] = g;
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
  const bool dual = isDual();
  const bool want_jac = triplets != nullptr;
  const auto & coupled = bc.coupledVariables();
  const int nc = static_cast<int>(coupled.size());
  const auto slot = [&](int v)
  { return static_cast<int>(std::find(coupled.begin(), coupled.end(), v) - coupled.begin()); };
  const int cp = slot(bc.variable());
  const int cs = slot(bc.secondaryVariable());
  const InterfacePairs & interface = _interface_pairs.at(&bc);
  auto & ctx = s.ctx;
  auto & pts = s.pts;
  auto & geo = s.geo;
  MappedPoint sgeo;
  // The flux @p flux times @p sign into the equation of the coupled variable @p c at the side
  // nodes @p nodes, numbered locally from @p first: at the node @p owner alone when @p at_owner
  // (the control domain that owns the point), and otherwise through the shape functions @p N.
  const auto addFlux = [&](const ADReal & flux,
                           double sign,
                           const std::vector<int> & nodes,
                           int first,
                           int c,
                           bool at_owner,
                           int owner,
                           const double * N)
  {
    if (at_owner)
    {
      const int j = static_cast<int>(std::find(nodes.begin(), nodes.end(), owner) - nodes.begin());
      s.Rloc[(first + j) * nc + c].addScaledBy(flux, sign);
      s.touched[(first + j) * nc + c] = 1;
      return;
    }
    for (int j = 0; j < static_cast<int>(nodes.size()); ++j)
    {
      s.Rloc[(first + j) * nc + c].addScaledBy(flux, sign * N[nodes[j]]);
      s.touched[(first + j) * nc + c] = 1;
    }
  };
  std::size_t side_index = 0;
  for (const Side & side : boundarySides(bc.boundaries()))
  {
    const auto & side_pairs = interface.points[side_index++];
    const Element & el = _mesh->element(side.first);
    if (!bc.activeOnBlock(el.block))
      continue;
    const auto & sn = ReferenceElement::get(el.type).sideNodes(side.second);
    const int np = static_cast<int>(sn.size());
    buildSidePoints(*_mesh, side, dual, bc.quadrature(), pts, _coord);
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
      setInterfaceValues(ctx, el, geo, sel, sgeo, U, opts.old, nd);
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
      ctx.x_field = geo.x;
      ctx.normal = (1.0 / ip.weight) * ip.area;
      ctx.state_owner = side.first;
      ctx.state_key = stateKey(
          3, interface.index, static_cast<unsigned>(side.second), static_cast<unsigned>(q));
      evaluateProperties(ctx);
      const double w = pointMeasure(ip, coordFactor(geo.x)) * wfac;
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
      addFlux(flux, -1.0, sn, 0, cp, ip.owner >= 0, ip.owner, geo.N);
      addFlux(flux, 1.0, ssn, np, cs, dual, pair.owner, sgeo.N);
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
  for (std::size_t index = 0; index < _ibcs.size(); ++index)
  {
    const auto * ib = dynamic_cast<const InterfaceBC *>(_ibcs[index].get());
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
    InterfacePairs & out = _interface_pairs[_ibcs[index].get()];
    out.index = static_cast<unsigned>(index);
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
      out.points.push_back(std::move(row));
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
  for (Index d : _flow->heldPressureDofs())
    fixed[d] = 1;
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

const MassEquation *
Problem::flowConstraint() const
{
  const MassEquation * constraint = nullptr;
  for (const auto & k : _kernels)
    if (const auto * mass = dynamic_cast<const MassEquation *>(k.get()))
    {
      if (constraint && constraint->variable() != mass->variable())
        throw InputError("The block preconditioner of the flow found two mass conservation "
                         "equations ('" +
                         constraint->name() + "' and '" + mass->name() +
                         "') on different variables; it handles one pressure.");
      constraint = mass;
    }
  return constraint;
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
  std::vector<std::vector<int>> blocks = _solid->displacementGroups(*this);
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

namespace
{
/// A near null space vector whose image under the Jacobian on a component is at most this
/// fraction of its image under the diagonal of the Jacobian solves the homogeneous problem there.
constexpr double kSingularImage = 1e-9;

/// The groups of coupled unknowns: the connected components of the graph of @p J without the
/// unknowns that @p fixed marks.  Returns the component of every unknown (-1 for a fixed one),
/// numbered in the order of their first unknown, and sets @p count to their number.
std::vector<Index>
coupledComponents(const SparseMatrix & J, const std::vector<char> & fixed, Index & count)
{
  const Index n = J.rows();
  DisjointSet sets(n);
  for (Index col = 0; col < J.outerSize(); ++col)
  {
    if (fixed[col])
      continue;
    for (SparseMatrix::InnerIterator it(J, col); it; ++it)
      if (!fixed[it.row()] && it.row() != col && it.value() != 0.0)
        sets.join(it.row(), col);
  }
  std::vector<Index> component(n, -1);
  count = 0;
  for (Index i = 0; i < n; ++i)
    if (!fixed[i])
    {
      const Index r = sets.root(i);
      if (component[r] < 0)
        component[r] = count++;
      component[i] = component[r];
    }
  return component;
}
} // namespace

void
Problem::checkSingularity(const SparseMatrix & J) const
{
  const Index n = J.rows();
  const std::vector<char> fixed = constrainedDofs();
  Index nc = 0;
  const std::vector<Index> component = coupledComponents(J, fixed, nc);
  Eigen::MatrixXd B = _solid->nearNullspace(*this);
  for (Index i = 0; i < n; ++i)
    if (fixed[i])
      B.row(i).setZero();
  const Eigen::MatrixXd JB = J * B;
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
  // A vector that J takes to zero on a component, relative to D, is a solution of the
  // homogeneous problem there.
  for (Index c = 0; c < nc; ++c)
    for (Eigen::Index j = 0; j < k; ++j)
    {
      const double reference = std::sqrt(scaled_norm(c, j));
      if (reference > 0.0 && !(std::sqrt(image_norm(c, j)) > kSingularImage * reference))
        throw InputError(singularPartMessage(component, c, B.col(j)));
    }
}

std::string
Problem::singularPartMessage(const std::vector<Index> & component, Index c, const Vector & b) const
{
  // The part of the mesh and the variables involved.
  const int nv = numVariables();
  const Index n = static_cast<Index>(component.size());
  constexpr double huge = 1e300;
  Point lo{huge, huge, huge}, hi{-huge, -huge, -huge};
  std::set<int> variables;
  std::set<Index> entities;
  for (Index i = 0; i < n; ++i)
    if (component[i] == c && b[i] != 0.0)
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
  os << ") " << (variables.size() > 1 ? "are" : "is") << " not held by any boundary condition. A "
     << (variables.size() > 1 ? "rigid body motion" : "constant")
     << " can be added there without changing the equations, so the solution is not unique. "
        "Give this part a Dirichlet condition (for a solid, the word fixed), or connect it to the rest "
        "of the mesh, e.g., by checking that the parts of an imported mesh share their nodes.";
  return os.str();
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
  // control domain of the node in the element (DMCDM) or the integral of its shape function (FEM),
  // with the coordinate factor integrated over it (IntegrationPoint::coordinate_measure).
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
    buildElementPoints(*_mesh, e, dual, PointSet::Volume, nodal, points, nullptr, polygon, _coord);
    for (const auto & ip : points)
    {
      mapPoint(*_mesh, el, ip.xi, mp);
      const Index node = polygon ? polygon->nodes[static_cast<std::size_t>(ip.owner)]
                                 : el.nodes[static_cast<std::size_t>(ip.owner)];
      weight[node] += pointMeasure(ip, coordFactor(mp.x));
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
    largest = std::max(
        largest, range > 0.0 ? sizes.change[var] / range : std::numeric_limits<double>::infinity());
  }
  return largest;
}

const Parallel &
Problem::parallel() const
{
  static const Parallel serial;
  return _parallel ? *_parallel : serial;
}

} // namespace dualmesh
