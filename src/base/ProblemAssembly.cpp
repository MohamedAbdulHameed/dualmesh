// SPDX-License-Identifier: LGPL-2.1-or-later
//
// Residual/Jacobian assembly and the steady and transient executioners.
#include "dualmesh/base/Problem.h"

#include "dualmesh/linalg/IncompleteLU.h"

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

  const auto gatherLocal = [&](ThreadScratch & s, const Element & el)
  {
    const int nn = el.numNodes();
    const int ld = nn * nv;
    s.cur.resize(ld);
    s.dofs.resize(ld);
    for (int k = 0; k < nn; ++k)
      for (int v = 0; v < nv; ++v)
      {
        const Index d = dof(el.nodes[k], v);
        s.dofs[k * nv + v] = d;
        s.cur[k * nv + v] = U[d];
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
    auto & geo = s.geo;
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
          geo = s.mapped[q];
        else
          mapPoint(*_mesh, el, ip.xi, geo);
        const MappedPoint * fp = &geo;
        if (group.spec.reduced)
        {
          mapPoint(*_mesh, el, ip.xi_field, fld);
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
        computeMaterials(ctx);
        return coordFactor(geo.x);
      };

      // Volume integrals (DMCDM: control volumes; FEM: element interior).
      if (has_source || (!dual && has_flux))
      {
        buildElementPoints(*_mesh, e, dual, PointSet::Volume, group.spec, pts, &s.mapped);
        for (std::size_t q = 0; q < pts.size(); ++q)
        {
          const IntegrationPoint & ip = pts[q];
          const double cf = evaluateAt(ip, q) * ip.weight;
          for (const Kernel * k : ks)
          {
            const int v = k->variable();
            const double w = cf * weightOf(k);
            if (k->hasSource())
            {
              const ADReal S = k->computeSource(ctx);
              if (ip.owner >= 0)
              {
                Rloc[ip.owner * nv + v].addScaledBy(S, w);
                touched[ip.owner * nv + v] = 1;
              }
              else
                for (int a = 0; a < nn; ++a)
                {
                  Rloc[a * nv + v].addScaledBy(S, w * geo.N[a]);
                  touched[a * nv + v] = 1;
                }
            }
            if (!dual && k->hasFlux() && ip.owner < 0)
            {
              ADVector3 F{ADReal(0.0), ADReal(0.0), ADReal(0.0)};
              k->computeFlux(ctx, F);
              for (int a = 0; a < nn; ++a)
              {
                ADReal & r = Rloc[a * nv + v];
                for (int d = 0; d < dim; ++d)
                  if (geo.dN[a][d] != 0.0)
                    r.addScaledBy(F[d], w * geo.dN[a][d]);
                touched[a * nv + v] = 1;
              }
            }
          }
        }
      }

      // Interfaces between control volumes (DMCDM only).
      if (dual && has_flux)
      {
        buildElementPoints(*_mesh, e, dual, PointSet::Faces, group.spec, pts, &s.mapped);
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
          for (const auto & ip : pts)
          {
            mapPoint(*_mesh, el, ip.xi, geo);
            const MappedPoint * fp = &geo;
            if (bc->quadrature().reduced)
            {
              mapPoint(*_mesh, el, ip.xi_field, fld);
              fp = &fld;
            }
            fillContext(ctx, el, *fp, cur, use_lag ? &lag : nullptr, opts.old ? &old : nullptr, nd);
            ctx.x = geo.x;
            ctx.normal = (1.0 / ip.weight) * ip.area;
            computeMaterials(ctx);
            const double w = ip.weight * coordFactor(geo.x) * wfac;
            const ADReal q = bc->computeBoundaryFlux(ctx) * w;
            if (ip.owner >= 0)
            {
              Rloc[ip.owner * nv + v] -= q;
              touched[ip.owner * nv + v] = 1;
            }
            else
              for (int a = 0; a < nn; ++a)
              {
                Rloc[a * nv + v] -= q * geo.N[a];
                touched[a * nv + v] = 1;
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
Problem::applyDirichlet(Vector & U, double load_factor) const
{
  for (const auto & bc : _nbcs)
  {
    const int v = bc->variable();
    const double f = bc->scalesWithLoad() ? load_factor : 1.0;
    for (Index n : bc->nodes())
      U[dof(n, v)] = f * bc->computeValue(entityPoint(n), _time);
  }
}

void
Problem::dirichletRows(const Vector & /*U*/, double /*lf*/, Vector & R, SparseMatrix * J) const
{
  std::vector<char> fixed(numDofs(), 0);
  for (const auto & bc : _nbcs)
    for (Index n : bc->nodes())
      fixed[dof(n, bc->variable())] = 1;
  for (Index i = 0; i < numDofs(); ++i)
    if (!_active_dof[i])
      fixed[i] = 1;
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
    throw std::runtime_error(
        "dualmesh: sparse LU factorization failed (singular system?): " + lu.lastErrorMessage() +
        ". Check that every variable has enough boundary conditions.");
  Vector x = lu.solve(b);
  if (lu.info() != Eigen::Success || !x.allFinite())
    throw std::runtime_error("dualmesh: sparse LU solve failed.");
  return x;
}
} // namespace

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
  if (method == "lu")
    return directSolve(A, b);
  if (method == "automatic")
  {
    if (preferDirectSolver(A.rows()))
      return directSolve(A, b);
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
                   "' (use automatic, lu, bicgstab, gmres, or cg).");
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
          std::cout << "  load " << lf << " iteration " << it << " |R| = " << rn
                    << " (converged)\n";
        break;
      }
      if (it > o.max_iterations)
      {
        result.history.push_back(rec);
        break;
      }
      Vector delta = linearSolve(J, -R, o, &result.linear_iterations);
      Vector Unew = _U + delta;
      if (o.relaxation > 0 && it > 1)
        Unew = (1.0 - o.relaxation) * Unew + o.relaxation * _U;
      const double un = Unew.norm();
      rec.step_norm = (Unew - _U).norm() / (un > 0 ? un : 1.0);
      _U = Unew;
      have_residual_at_U = false;
      result.history.push_back(rec);
      ++result.total_iterations;
      if (o.verbose)
        std::cout << "  load " << lf << " iteration " << it << " |R| = " << std::scientific
                  << std::setprecision(3) << rn << " |dU|/|U| = " << rec.step_norm << "\n";
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
  return nonlinearSolve(options, base, nullptr);
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
  return runTransient(
      tr,
      options,
      _U,
      _time,
      [&](const Vector & old, double dt) { return takeTimeStep(old, dt, tr, attempt); },
      [&](int step, double)
      {
        if (_step_callback)
          _step_callback(_time, *this);
        write(step);
      },
      true);
}

} // namespace dualmesh
