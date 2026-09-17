// SPDX-License-Identifier: LGPL-2.1-or-later
//
// Residual/Jacobian assembly and the steady and transient executioners.
#include "dualmesh/base/Problem.h"

#include <Eigen/IterativeLinearSolvers>
#include <Eigen/SparseLU>

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
void
scatter(const std::vector<ADReal> & Rloc,
        const std::vector<Index> & dofs,
        const std::vector<char> & touched,
        Vector & R,
        std::vector<Triplet> * trip)
{
  const int ld = static_cast<int>(dofs.size());
  for (int i = 0; i < ld; ++i)
  {
    if (!touched[i])
      continue;
    R[dofs[i]] += Rloc[i].value();
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

void
Problem::assemble(const Vector & U, const AssemblyOptions & opts, Vector & R, SparseMatrix * J)
{
  initialize();
  const Index ndof = numDofs();
  const int nv = numVariables();
  const int dim = _mesh->dimension();
  const bool dual = _method == Method::DualMesh;
  const bool want_jac = J != nullptr;
  R.setZero(ndof);
  std::vector<Triplet> trip;
  if (want_jac)
    trip.reserve(static_cast<std::size_t>(_mesh->numElements()) * 16 * nv * nv * 4);

  std::vector<double> cur, lag, old;
  std::vector<Index> dofs;
  std::vector<ADReal> Rloc;
  std::vector<char> touched;
  std::vector<IntegrationPoint> pts;
  MappedPoint geo, fld, cen;
  QpContext ctx;
  ctx.dim = dim;
  ctx.time = _time;
  ctx.dt = opts.dt;
  ctx.load_factor = opts.load_factor;
  ctx.mode = opts.mode;
  const bool use_lag = opts.mode == LinearizationMode::Picard && opts.lagged;

  const auto gatherLocal = [&](const Element & el)
  {
    const int nn = el.numNodes();
    const int ld = nn * nv;
    cur.resize(ld);
    dofs.resize(ld);
    for (int k = 0; k < nn; ++k)
      for (int v = 0; v < nv; ++v)
      {
        const Index d = dof(el.nodes[k], v);
        dofs[k * nv + v] = d;
        cur[k * nv + v] = U[d];
      }
    if (use_lag)
    {
      lag.resize(ld);
      for (int i = 0; i < ld; ++i)
        lag[i] = (*opts.lagged)[dofs[i]];
    }
    if (opts.old)
    {
      old.resize(ld);
      for (int i = 0; i < ld; ++i)
        old[i] = (*opts.old)[dofs[i]];
    }
    Rloc.assign(ld, ADReal(0.0));
    touched.assign(ld, 0);
    return ld;
  };

  for (Index e = 0; e < _mesh->numElements(); ++e)
  {
    const Element & el = _mesh->element(e);
    const auto & ref = ReferenceElement::get(el.type);
    const int nn = el.numNodes();
    const int ld = gatherLocal(el);
    const int nd = want_jac ? ld : 0;
    ctx.element = e;
    ctx.block = el.block;
    ctx.on_boundary = false;
    bool any = false;

    for (const auto & group : _groups)
    {
      std::vector<const Kernel *> ks;
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
      const auto evaluateAt = [&](const IntegrationPoint & ip)
      {
        mapPoint(*_mesh, el, ip.xi, geo);
        const MappedPoint * fp = &geo;
        if (group.spec.reduced)
        {
          mapPoint(*_mesh, el, ip.xi_field, fld);
          fp = &fld;
        }
        fillContext(ctx, el, *fp, cur, use_lag ? &lag : nullptr, opts.old ? &old : nullptr, nd);
        ctx.x = geo.x;
        computeMaterials(ctx);
        return coordFactor(geo.x);
      };

      // Volume integrals (DMCDM: control volumes; FEM: element interior).
      if (has_source || (!dual && has_flux))
      {
        buildElementPoints(*_mesh, e, dual, PointSet::Volume, group.spec, pts);
        for (const auto & ip : pts)
        {
          const double cf = evaluateAt(ip) * ip.weight;
          for (const Kernel * k : ks)
          {
            const int v = k->variable();
            const double w = cf * weightOf(k);
            if (k->hasSource())
            {
              const ADReal S = k->computeSource(ctx) * w;
              if (ip.owner >= 0)
              {
                Rloc[ip.owner * nv + v] += S;
                touched[ip.owner * nv + v] = 1;
              }
              else
                for (int a = 0; a < nn; ++a)
                {
                  Rloc[a * nv + v] += S * geo.N[a];
                  touched[a * nv + v] = 1;
                }
            }
            if (!dual && k->hasFlux() && ip.owner < 0)
            {
              ADVector3 F{ADReal(0.0), ADReal(0.0), ADReal(0.0)};
              k->computeFlux(ctx, F);
              for (int a = 0; a < nn; ++a)
              {
                ADReal c(0.0);
                for (int d = 0; d < dim; ++d)
                  if (geo.dN[a][d] != 0.0)
                    c += F[d] * geo.dN[a][d];
                c *= w;
                Rloc[a * nv + v] += c;
                touched[a * nv + v] = 1;
              }
            }
          }
        }
      }

      // Interfaces between control volumes (DMCDM only).
      if (dual && has_flux)
      {
        buildElementPoints(*_mesh, e, dual, PointSet::Faces, group.spec, pts);
        for (const auto & ip : pts)
        {
          const double cf = evaluateAt(ip);
          for (const Kernel * k : ks)
          {
            if (!k->hasFlux())
              continue;
            const int v = k->variable();
            ADVector3 F{ADReal(0.0), ADReal(0.0), ADReal(0.0)};
            k->computeFlux(ctx, F);
            ADReal fn(0.0);
            for (int d = 0; d < dim; ++d)
              if (ip.area[d] != 0.0)
                fn += F[d] * ip.area[d];
            fn *= cf * weightOf(k);
            // -\oint F.n over the boundary of CD_a; the same interface
            // enters CD_b with the opposite normal.
            Rloc[ip.owner * nv + v] -= fn;
            Rloc[ip.neighbor * nv + v] += fn;
            touched[ip.owner * nv + v] = 1;
            touched[ip.neighbor * nv + v] = 1;
          }
        }
      }
    }
    (void) ref;
    if (any)
      scatter(Rloc, dofs, touched, R, want_jac ? &trip : nullptr);
  }

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
          const int ld = gatherLocal(el);
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
          scatter(Rloc, dofs, touched, R, want_jac ? &trip : nullptr);
        }
    }

  // ---- concentrated nodal loads ----------------------------------------------
  if (opts.include_steady_terms)
    for (const auto & load : _loads)
    {
      const int v = load->variable();
      const double wfac = opts.theta * (load->scalesWithLoad() ? opts.load_factor : 1.0);
      for (Index n : load->nodes())
        R[dof(n, v)] -= wfac * load->computeValue(_mesh->node(n), _time);
    }

  if (want_jac)
  {
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
      U[dof(n, v)] = f * bc->computeValue(_mesh->node(n), _time);
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

Vector
Problem::linearSolve(const SparseMatrix & A_in, const Vector & b, const SolverOptions & o) const
{
  SparseMatrix A = A_in;
  A.makeCompressed();
  if (o.linear_solver == "lu")
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
  if (o.linear_solver == "bicgstab")
  {
    Eigen::BiCGSTAB<SparseMatrix, Eigen::IncompleteLUT<double>> solver;
    solver.setTolerance(o.linear_tolerance);
    solver.setMaxIterations(o.linear_max_iterations);
    solver.preconditioner().setDroptol(1e-5);
    solver.preconditioner().setFillfactor(20);
    solver.compute(A);
    Vector x = solver.solve(b);
    if (solver.info() != Eigen::Success)
      throw std::runtime_error("dualmesh: BiCGSTAB did not converge.");
    return x;
  }
  if (o.linear_solver == "cg")
  {
    Eigen::ConjugateGradient<SparseMatrix, Eigen::Lower | Eigen::Upper> solver;
    solver.setTolerance(o.linear_tolerance);
    solver.setMaxIterations(o.linear_max_iterations);
    solver.compute(A);
    Vector x = solver.solve(b);
    if (solver.info() != Eigen::Success)
      throw std::runtime_error("dualmesh: conjugate gradient did not converge.");
    return x;
  }
  throw InputError("Unknown linear solver '" + o.linear_solver + "' (use lu, bicgstab, or cg).");
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
      Vector delta = linearSolve(J, -R, o);
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
Problem::solveTransient(const TransientOptions & tr, const SolverOptions & options)
{
  initialize();
  if (tr.dt <= 0)
    throw InputError("Transient: the time step must be positive.");
  if (tr.theta < 0 || tr.theta > 1)
    throw InputError("Transient: theta must be in [0, 1].");
  SolveResult total;
  _time = tr.start_time;
  int step = 0;
  const auto output = [&]()
  {
    if (tr.output_interval > 0 && step % tr.output_interval == 0 && !tr.output_file_base.empty())
    {
      std::ostringstream os;
      os << tr.output_file_base << "_" << std::setw(5) << std::setfill('0') << step << ".vtu";
      writeVTU(os.str());
    }
  };
  applyDirichlet(_U, 1.0);
  output();
  while (_time < tr.end_time - 1e-12 * tr.dt)
  {
    const double dt = std::min(tr.dt, tr.end_time - _time);
    Vector Uold = _U;
    Vector Rold;
    if (tr.theta < 1.0)
    {
      AssemblyOptions ao;
      ao.include_time_kernels = false;
      ao.theta = 1.0 - tr.theta;
      ao.dt = dt;
      assemble(Uold, ao, Rold, nullptr);
    }
    _time += dt;
    ++step;
    AssemblyOptions base;
    base.old = &Uold;
    base.dt = dt;
    base.theta = tr.theta;
    base.include_time_kernels = true;
    base.include_steady_terms = tr.theta > 0;
    auto r = nonlinearSolve(options, base, tr.theta < 1.0 ? &Rold : nullptr);
    total.total_iterations += r.total_iterations;
    total.history.insert(total.history.end(), r.history.begin(), r.history.end());
    if (!r.converged)
      return total;
    if (_step_callback)
      _step_callback(_time, *this);
    output();
  }
  total.converged = true;
  return total;
}

} // namespace dualmesh
