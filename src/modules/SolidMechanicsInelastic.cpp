// SPDX-License-Identifier: LGPL-2.1-or-later
//
// Solid mechanics module, continuum: creep and plasticity of small_strain_stress, which
// finite_strain_stress applies in its rotated frame.  The strains of one step are integrated by
// the backward Euler method, by a return mapping at each integration point:
//
//   returnMap           creep alone, a radial return on the von Mises stress
//   plasticReturn       J2 plasticity with isotropic and kinematic hardening, a radial return
//   creepPlasticReturn  creep and isotropic plasticity in one radial return
//   generalReturn       an anisotropic stiffness or Hill's equivalent stress, a closest-point
//                       return by Newton's method on the stress and the two increments
//
// The derivatives of each return by the unknowns follow from the implicit function theorem at
// the solved point, so the tangent is exact.  In plane stress planeStressInelasticStrain finds
// the out-of-plane strain at which the stress after the return has sigma_zz = 0.
#include "dualmesh/modules/SolidMechanics.h"

#include <Eigen/Dense>

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

namespace dualmesh::solid_mechanics
{

namespace
{

double
valueOf(double x)
{
  return x;
}

double
valueOf(const ADReal & x)
{
  return x.value();
}

/// The values of six numbers with derivatives.
std::array<double, 6>
valuesOf(const std::array<ADReal, 6> & a)
{
  std::array<double, 6> v;
  for (int i = 0; i < 6; ++i)
    v[i] = a[i].value();
  return v;
}

/// The equivalent stress sqrt(3/2 s : s) of the deviatoric tensor @p s.
ADReal
equivalentStress(const std::array<ADReal, 6> & s)
{
  return sqrt(1.5 * (s[0] * s[0] + s[1] * s[1] + s[2] * s[2]) +
              3.0 * (s[3] * s[3] + s[4] * s[4] + s[5] * s[5]));
}

/// The inelastic strain increment dp (3/2) s / q of a radial return of the equivalent strain
/// increment @p dp along the deviatoric tensor @p s of equivalent stress @p q: (3/2) dp s_i / q
/// for the normal components and 3 dp s_i / q for the engineering shears.
std::array<ADReal, 6>
radialIncrement(const ADReal & dp, const std::array<ADReal, 6> & s, const ADReal & q)
{
  std::array<ADReal, 6> increment;
  for (int i = 0; i < 6; ++i)
    increment[i] = (i < 3 ? 1.5 : 3.0) * dp * s[i] / q;
  return increment;
}

/// Subtract from @p sigma the stress 2 G e of the deviatoric strain increment @p e
/// (engineering shears), with G the shear modulus @p mu.
void
subtractDeviatoricStress(std::array<ADReal, 6> & sigma,
                         const ADReal & mu,
                         const std::array<ADReal, 6> & e)
{
  for (int i = 0; i < 6; ++i)
    sigma[i] -= (i < 3 ? 2.0 : 1.0) * mu * e[i];
}

/// The root of the monotone function @p f on the bracket [@p lo, @p hi], by Newton's method from
/// @p x, safeguarded by bisection: a Newton step that leaves the bracket is replaced by its
/// midpoint.  f(x, slope) returns f at x and sets @p slope to its derivative there.  @p increasing
/// tells whether f rises or falls through the root.  The iteration stops after the step on which
/// converged(next, x, lo, hi) holds, with next the new iterate, or after @p max_iterations steps.
template <typename Function, typename Converged>
double
safeguardedNewton(const Function & f,
                  bool increasing,
                  double x,
                  double lo,
                  double hi,
                  int max_iterations,
                  const Converged & converged)
{
  for (int it = 0; it < max_iterations; ++it)
  {
    double slope = 0.0;
    const double fx = f(x, slope);
    if ((fx > 0.0) == increasing)
      hi = x;
    else
      lo = x;
    double next = x - fx / slope;
    if (!(next > lo && next < hi))
      next = 0.5 * (lo + hi);
    const bool done = converged(next, x, lo, hi);
    x = next;
    if (done)
      break;
  }
  return x;
}

/// The most iterations of the root finders of the creep return and of the plastic return.
constexpr int kCreepIterations = 60, kPlasticIterations = 100;

/// Whether the partial-pivot LU factors @p lu have a zero pivot, that is, whether their matrix
/// is singular.
template <typename LU>
bool
hasZeroPivot(const LU & lu)
{
  return (lu.matrixLU().diagonal().array() == 0.0).any();
}

} // namespace

template <typename T>
T
SmallStrainStress::yieldStress(const T & p, const T & temperature) const
{
  std::array<T, 32> args;
  args[0] = p;
  args[1] = temperature;
  for (std::size_t k = 0; k < _plasticity_constants.size(); ++k)
    args[2 + k] = T(_plasticity_constants[k]);
  return _yield_expression->value(args.data());
}

template <typename T>
T
SmallStrainStress::hillStress(const std::array<T, 6> & sigma, std::array<T, 6> & normal) const
{
  std::array<T, 6> Ps;
  T square(0.0);
  for (int i = 0; i < 6; ++i)
  {
    Ps[i] = T(0.0);
    for (int j = 0; j < 6; ++j)
      Ps[i] += _hill[6 * i + j] * sigma[j];
    square += sigma[i] * Ps[i];
  }
  using std::sqrt;
  const T q = sqrt(square);
  for (int i = 0; i < 6; ++i)
    normal[i] = valueOf(q) > 0.0 ? Ps[i] / q : T(0.0);
  return q;
}

template <typename T>
T
SmallStrainStress::creepRate(const T & sigma,
                             const T & temperature,
                             const T & p,
                             const QpContext & ctx) const
{
  // The von Mises stress is not negative, and round-off in the return mapping
  // must not hand a negative one to a fractional power.
  if (valueOf(sigma) <= 0.0)
    return T(0.0);
  std::array<T, 32> args;
  args[0] = sigma;
  args[1] = temperature;
  args[2] = T(_grain);
  args[3] = T(ctx.time);
  args[4] = p;
  for (std::size_t k = 0; k < _creep_constants.size(); ++k)
    args[5 + k] = T(_creep_constants[k]);
  return _creep_factor * _creep_expression->value(args.data());
}

void
SmallStrainStress::inelasticReturn(QpContext & ctx,
                                   const ADReal & mu,
                                   const ADReal & T,
                                   double creep_old,
                                   double plastic_old,
                                   const std::vector<ADReal> & back_old,
                                   std::array<ADReal, 6> & sigma,
                                   InelasticIncrements & inc) const
{
  inc.back.resize(back_old.size());
  for (std::size_t j = 0; j < back_old.size(); ++j)
    inc.back[j] = back_old[j].value();
  const bool creep = creeps() && ctx.dt > 0;
  if (_anisotropic || _hill_given)
  {
    generalReturn(ctx, T, creep_old, plastic_old, sigma, inc);
    return;
  }
  if (!_plastic)
  {
    if (creep)
      returnMap(ctx, mu, T, creep_old, sigma, inc.creep, inc.creep_equivalent);
    return;
  }
  if (!creep)
  {
    plasticReturn(mu, T, plastic_old, back_old, sigma, inc);
    return;
  }
  creepPlasticReturn(ctx, mu, T, creep_old, plastic_old, sigma, inc);
}

void
SmallStrainStress::inelasticReturn(QpContext & ctx,
                                   const ADReal & mu,
                                   const ADReal & T,
                                   const double * old,
                                   std::array<ADReal, 6> & sigma,
                                   InelasticIncrements & inc) const
{
  const double * back_old = old + kBackStress;
  const std::vector<ADReal> back(back_old, back_old + 6 * _hardening.size());
  inelasticReturn(ctx, mu, T, old[kCreepEquivalent], old[kPlasticEquivalent], back, sigma, inc);
}

void
SmallStrainStress::generalReturn(QpContext & ctx,
                                 const ADReal & T,
                                 double c_old,
                                 double p_old,
                                 std::array<ADReal, 6> & sigma,
                                 InelasticIncrements & inc) const
{
  std::array<double, 36> C = _stiffness;
  if (!_anisotropic)
  {
    if (!_constant_moduli)
      throw InputError("'" + name() + "': Hill creep and plasticity take constant moduli.");
    C.fill(0.0);
    for (int i = 0; i < 3; ++i)
    {
      for (int j = 0; j < 3; ++j)
        C[6 * i + j] = _lambda0;
      C[6 * i + i] += 2.0 * _mu0;
      C[6 * (i + 3) + i + 3] = _mu0;
    }
  }
  // The compliance S = C^-1.
  const Eigen::PartialPivLU<Eigen::Matrix<double, 6, 6>> C_lu(
      Eigen::Map<const Eigen::Matrix<double, 6, 6, Eigen::RowMajor>>(C.data()));
  if (hasZeroPivot(C_lu))
    throw InputError("dualmesh: a singular matrix in the return mapping.");
  const Eigen::Matrix<double, 6, 6> S = C_lu.inverse();
  const bool creep = creeps() && ctx.dt > 0;
  const double Tv = T.value();
  std::array<double, 6> trial;
  for (int i = 0; i < 6; ++i)
    trial[i] = sigma[i].value();
  std::array<double, 6> unused;
  const double q0 = hillStress(trial, unused);
  bool yielding = _plastic && q0 > yieldStress(p_old, Tv);
  if (!creep && !yielding)
    return;
  // The residuals at x = (sigma, dc, dp) and the elastic trial strain S sigma_trial.
  const auto residuals = [&](const std::array<ADReal, 8> & x,
                             const std::array<ADReal, 6> & strain,
                             const ADReal & temperature,
                             bool plastic,
                             std::array<ADReal, 8> & r)
  {
    std::array<ADReal, 6> s, normal;
    for (int i = 0; i < 6; ++i)
      s[i] = x[i];
    const ADReal q = hillStress(s, normal);
    for (int i = 0; i < 6; ++i)
    {
      r[i] = -strain[i] + (x[6] + x[7]) * normal[i];
      for (int j = 0; j < 6; ++j)
        r[i] += S(i, j) * x[j];
    }
    r[6] = creep ? x[6] - ctx.dt * creepRate(q, temperature, ADReal(c_old) + x[6], ctx) : x[6];
    r[7] = plastic ? q - yieldStress(ADReal(p_old) + x[7], temperature) : x[7];
  };
  std::array<ADReal, 6> trial_strain;
  for (int i = 0; i < 6; ++i)
  {
    double e = 0.0;
    for (int j = 0; j < 6; ++j)
      e += S(i, j) * trial[j];
    trial_strain[i] = ADReal(e);
  }
  std::array<double, 8> x{};
  for (int i = 0; i < 6; ++i)
    x[i] = trial[i];
  Eigen::Matrix<double, 8, 8> J;
  const double size = std::abs(q0) + 1e-300;
  // Newton's method: first with the plastic increment held at zero (creep alone), then, if
  // the stress it leaves is beyond the yield stress, with both.
  const auto solve = [&](bool plastic)
  {
    double change = 1.0;
    for (int it = 0; it < 100 && change > 1e-14; ++it)
    {
      std::array<ADReal, 8> X, r;
      for (int i = 0; i < 8; ++i)
        X[i] = ADReal::independent(x[i], i, 8);
      residuals(X, trial_strain, ADReal(Tv), plastic, r);
      Eigen::Matrix<double, 8, 1> residual;
      for (int i = 0; i < 8; ++i)
      {
        residual(i) = r[i].value();
        for (int j = 0; j < 8; ++j)
          J(i, j) = r[i].derivative(j);
      }
      const Eigen::PartialPivLU<Eigen::Matrix<double, 8, 8>> lu(J);
      if (hasZeroPivot(lu))
        throw InadmissibleStateError("dualmesh: the return mapping met a singular Jacobian.");
      const Eigen::Matrix<double, 8, 1> b = lu.solve(residual);
      // A step that would make an increment negative is shortened.  Convergence is judged
      // on the full Newton step, so a step shortened to nothing does not pass for a solution.
      double t = 1.0;
      while ((x[6] - t * b(6) < 0.0 || x[7] - t * b(7) < 0.0) && t > 1e-8)
        t *= 0.5;
      change = 0.0;
      for (int i = 0; i < 8; ++i)
      {
        x[i] -= t * b(i);
        change = std::max(change, std::abs(b(i)) * (i < 6 ? 1.0 / size : 1.0));
      }
    }
    // A return that has not settled to round-off leaves a stress off the yield and creep
    // laws, so the state is rejected and the solve takes a smaller step.
    if (!(change <= 1e-8))
      throw InadmissibleStateError("dualmesh: the return mapping did not converge.");
  };
  if (creep)
    solve(false);
  if (_plastic)
  {
    std::array<double, 6> s;
    for (int i = 0; i < 6; ++i)
      s[i] = x[i];
    yielding = hillStress(s, unused) > yieldStress(p_old, Tv);
    if (yielding)
      solve(true);
  }
  // The derivatives of the unknowns: x(U) = x - J^-1 r(x, U), with the residual of the trial
  // strain of the unknowns at the solved x.
  {
    std::array<ADReal, 8> X, r;
    for (int i = 0; i < 8; ++i)
      X[i] = ADReal::independent(x[i], i, 8);
    residuals(X, trial_strain, ADReal(Tv), yielding, r);
    for (int i = 0; i < 8; ++i)
      for (int j = 0; j < 8; ++j)
        J(i, j) = r[i].derivative(j);
  }
  const Eigen::PartialPivLU<Eigen::Matrix<double, 8, 8>> J_lu(J);
  if (hasZeroPivot(J_lu))
    throw InputError("dualmesh: a singular matrix in the return mapping.");
  const Eigen::Matrix<double, 8, 8> Jinv = J_lu.inverse();
  std::array<ADReal, 6> strain;
  for (int i = 0; i < 6; ++i)
  {
    strain[i] = ADReal(0.0);
    for (int j = 0; j < 6; ++j)
      strain[i] += S(i, j) * sigma[j];
  }
  std::array<ADReal, 8> X, r;
  for (int i = 0; i < 8; ++i)
    X[i] = ADReal(x[i]);
  residuals(X, strain, T, yielding, r);
  std::array<ADReal, 8> result;
  for (int i = 0; i < 8; ++i)
  {
    result[i] = ADReal(x[i]);
    for (int j = 0; j < 8; ++j)
      result[i] -= Jinv(i, j) * (r[j] - ADReal(r[j].value()));
  }
  std::array<double, 6> s, normal;
  for (int i = 0; i < 6; ++i)
  {
    s[i] = x[i];
    sigma[i] = result[i];
  }
  hillStress(s, normal);
  for (int i = 0; i < 6; ++i)
  {
    inc.creep[i] = x[6] * normal[i];
    inc.plastic[i] = x[7] * normal[i];
  }
  inc.creep_equivalent = x[6];
  inc.plastic_equivalent = x[7];
}

void
SmallStrainStress::plasticReturn(const ADReal & mu,
                                 const ADReal & T,
                                 double p_old,
                                 const std::vector<ADReal> & back_old,
                                 std::array<ADReal, 6> & sigma,
                                 InelasticIncrements & inc) const
{
  const std::size_t K = _hardening.size();
  const std::array<ADReal, 6> dev = deviator(sigma.data());
  const double G = mu.value(), Tv = T.value();
  // The relative stress xi = s - sum a_k alpha_k^n of the deviator s and the back stresses
  // alpha_k^n, with a_k = 1 / (1 + gamma_k dp), and, when @p H is given, the slope 3G + sum a_k
  // C_k of the plastic increment: from plain numbers, or with the derivatives of the unknowns.
  const auto relative = [&](const auto & dp,
                            const auto & s,
                            const auto & back,
                            const auto & shear_modulus,
                            std::array<ADReal, 6> & xi,
                            ADReal * H)
  {
    if (H)
      *H = 3.0 * shear_modulus;
    for (int i = 0; i < 6; ++i)
      xi[i] = s[i];
    for (std::size_t k = 0; k < K; ++k)
    {
      const auto a = 1.0 / (1.0 + _hardening[k].recovery * dp);
      if (H)
        *H += _hardening[k].modulus * a;
      for (int i = 0; i < 6; ++i)
        xi[i] -= a * back[6 * k + i];
    }
  };
  // The residual f(dp) = q(xi) - H dp - sigma_y(p_n + dp) from the values of the trial, with
  // its derivative in dp.
  const std::array<double, 6> dev_values = valuesOf(dev);
  std::vector<double> back_values(back_old.size());
  for (std::size_t j = 0; j < back_old.size(); ++j)
    back_values[j] = back_old[j].value();
  const auto f = [&](double dp, double & slope)
  {
    const ADReal x = ADReal::independent(dp, 0, 1);
    std::array<ADReal, 6> xi;
    ADReal H;
    relative(x, dev_values, back_values, G, xi, &H);
    const ADReal r = equivalentStress(xi) - H * x - yieldStress(ADReal(p_old) + x, ADReal(Tv));
    slope = r.derivative(0);
    return r.value();
  };
  double slope = 1.0;
  const double f0 = f(0.0, slope);
  if (!(f0 > 0.0))
    return;
  // The root: f(0) > 0, and f falls without bound as dp grows.
  double lo = 0.0, hi = f0 / (3.0 * G);
  while (f(hi, slope) > 0.0)
  {
    lo = hi;
    hi *= 2.0;
  }
  // Converged at a step or a bracket at the round-off of dp.
  const auto converged = [](double next, double x, double low, double high)
  { return std::abs(next - x) <= 1e-14 * std::abs(next) || high - low <= 1e-16 * high; };
  const double dp =
      safeguardedNewton(f, false, 0.5 * (lo + hi), lo, hi, kPlasticIterations, converged);
  // The derivatives of dp by the implicit function theorem, from the residual with the
  // derivatives of the unknowns at the solved dp.
  f(dp, slope);
  std::array<ADReal, 6> xi;
  ADReal H;
  relative(dp, dev, back_old, mu, xi, &H);
  const ADReal residual = equivalentStress(xi) - H * dp - yieldStress(ADReal(p_old + dp), T);
  const ADReal DP = dp - residual / slope;
  // The relative stress and its direction with the derivatives of DP.
  relative(DP, dev, back_old, mu, xi, nullptr);
  const ADReal q = equivalentStress(xi);
  const std::array<ADReal, 6> increment = radialIncrement(DP, xi, q);
  subtractDeviatoricStress(sigma, mu, increment);
  inc.plastic = valuesOf(increment);
  inc.plastic_equivalent = DP.value();
  for (std::size_t k = 0; k < K; ++k)
  {
    const double a = (1.0 / (1.0 + _hardening[k].recovery * DP)).value();
    for (int i = 0; i < 6; ++i)
      inc.back[6 * k + i] = a * (back_old[6 * k + i].value() +
                                 _hardening[k].modulus * dp * xi[i].value() / q.value());
  }
}

void
SmallStrainStress::creepPlasticReturn(QpContext & ctx,
                                      const ADReal & mu,
                                      const ADReal & T,
                                      double c_old,
                                      double p_old,
                                      std::array<ADReal, 6> & sigma,
                                      InelasticIncrements & inc) const
{
  // Creep alone first: where the stress it leaves is within the yield surface, that is all.
  std::array<ADReal, 6> creep_only = sigma;
  InelasticIncrements alone;
  returnMap(ctx, mu, T, c_old, creep_only, alone.creep, alone.creep_equivalent);
  const std::array<ADReal, 6> dev = deviator(sigma.data());
  const ADReal q = equivalentStress(dev);
  const double qv = q.value(), G = mu.value(), Tv = T.value();
  if (qv - 3.0 * G * alone.creep_equivalent <= yieldStress(p_old, Tv))
  {
    sigma = creep_only;
    inc.creep = alone.creep;
    inc.creep_equivalent = alone.creep_equivalent;
    return;
  }
  // The residuals of (dc, dp) with derivatives in both, or in the unknowns at a solution.
  const auto residuals = [&](const ADReal & dc,
                             const ADReal & dp,
                             const ADReal & qq,
                             const ADReal & GG,
                             const ADReal & TT,
                             ADReal & r1,
                             ADReal & r2)
  {
    const ADReal effective = qq - 3.0 * GG * (dc + dp);
    r1 = dc - ctx.dt * creepRate(effective, TT, ADReal(c_old) + dc, ctx);
    r2 = effective - yieldStress(ADReal(p_old) + dp, TT);
  };
  double dc = alone.creep_equivalent, dp = 0.0, change = 0.0;
  double J[2][2] = {{1, 0}, {0, 1}};
  for (int it = 0; it < 100; ++it)
  {
    ADReal r1, r2;
    residuals(ADReal::independent(dc, 0, 2),
              ADReal::independent(dp, 1, 2),
              ADReal(qv),
              ADReal(G),
              ADReal(Tv),
              r1,
              r2);
    J[0][0] = r1.derivative(0);
    J[0][1] = r1.derivative(1);
    J[1][0] = r2.derivative(0);
    J[1][1] = r2.derivative(1);
    const double det = J[0][0] * J[1][1] - J[0][1] * J[1][0];
    const double sc = (J[1][1] * r1.value() - J[0][1] * r2.value()) / det;
    const double sp = (J[0][0] * r2.value() - J[1][0] * r1.value()) / det;
    // A step that would make an increment negative or reverse the stress is shortened.
    double t = 1.0;
    while ((dc - t * sc < 0.0 || dp - t * sp < 0.0 || 3.0 * G * (dc - t * sc + dp - t * sp) > qv) &&
           t > 1e-6)
      t *= 0.5;
    dc -= t * sc;
    dp -= t * sp;
    // The change is judged on the stress scale: 3G (dc + dp) is the fall of the equivalent
    // stress from the trial value qv, whose round-off bounds how well the increments settle.
    change = 3.0 * G * (std::abs(sc) + std::abs(sp));
    if (change <= 1e-14 * qv)
      break;
  }
  if (!(change <= 1e-8 * qv))
    throw InadmissibleStateError("dualmesh: the return mapping of creep and plasticity did not "
                                 "converge.");
  // The derivatives of (dc, dp) by the implicit function theorem.
  ADReal r1, r2;
  residuals(ADReal(dc), ADReal(dp), q, mu, T, r1, r2);
  const double det = J[0][0] * J[1][1] - J[0][1] * J[1][0];
  const ADReal DC = dc - (J[1][1] * r1 - J[0][1] * r2) / det;
  const ADReal DP = dp - (J[0][0] * r2 - J[1][0] * r1) / det;
  const std::array<ADReal, 6> creep = radialIncrement(DC, dev, q);
  const std::array<ADReal, 6> plastic = radialIncrement(DP, dev, q);
  std::array<ADReal, 6> both;
  for (int i = 0; i < 6; ++i)
    both[i] = creep[i] + plastic[i];
  subtractDeviatoricStress(sigma, mu, both);
  inc.creep = valuesOf(creep);
  inc.plastic = valuesOf(plastic);
  inc.creep_equivalent = DC.value();
  inc.plastic_equivalent = DP.value();
}

void
SmallStrainStress::returnMap(QpContext & ctx,
                             const ADReal & mu,
                             const ADReal & T,
                             double p_old,
                             std::array<ADReal, 6> & sigma,
                             std::array<double, 6> & creep_increment,
                             double & equivalent_increment) const
{
  // Radial return: the creep strain increment is dp (3/2) s / q along the
  // deviator s of the trial stress, whose von Mises value q falls to
  // q - 3 G dp, and dp = dt * rate(q - 3 G dp, T) (backward Euler).
  const std::array<ADReal, 6> dev = deviator(sigma.data());
  const ADReal q = equivalentStress(dev);
  const double qv = q.value(), G = mu.value();
  const double Tv = T.value();
  // The creep increment is at most q / 3G (the stress cannot reverse), so below 1e-12 of that
  // there is no creep to compute, in any units.
  if (qv > 1e-12 * 3.0 * G)
  {
    // g(dp) = dp - dt rate(q - 3 G dp, T, p_old + dp) and its derivative in dp, through a pass
    // whose only derivative is dp.
    const auto g = [&](double dp, double & slope)
    {
      const ADReal x = ADReal::independent(dp, 0, 1);
      const ADReal r = x - ctx.dt * creepRate(qv - 3.0 * G * x, ADReal(Tv), p_old + x, ctx);
      slope = r.derivative(0);
      return r.value();
    };
    // The root lies in [0, q / 3G]: g(0) <= 0 and g(q / 3G) >= 0.
    const double hi = qv / (3.0 * G);
    // Converged at a step at the round-off of dp, or a bracket below 1e-16 of max(1, hi).
    const auto converged = [](double next, double x, double low, double high)
    {
      return std::abs(next - x) <= 1e-14 * std::max(1e-30, std::abs(next)) ||
             high - low < 1e-16 * std::max(1.0, high);
    };
    const double dp = safeguardedNewton(g,
                                        true,
                                        std::min(ctx.dt * creepRate(qv, Tv, p_old, ctx), hi),
                                        0.0,
                                        hi,
                                        kCreepIterations,
                                        converged);
    double slope = 0.0;
    // The derivatives of dp follow from the implicit function theorem:
    // g(dp, q, G, T) = 0, so d(dp) = -(dg/dq dq + ...) / (dg/d(dp)).
    g(dp, slope);
    const ADReal residual = dp - ctx.dt * creepRate(q - 3.0 * mu * dp, T, ADReal(p_old + dp), ctx);
    const ADReal DP = dp - residual / slope;
    const std::array<ADReal, 6> increment = radialIncrement(DP, dev, q);
    subtractDeviatoricStress(sigma, mu, increment);
    creep_increment = valuesOf(increment);
    equivalent_increment = DP.value();
  }
}

void
SmallStrainStress::planeStressInelasticStrain(QpContext & ctx,
                                              const ADReal & lambda,
                                              const ADReal & mu,
                                              const double * old,
                                              ADReal * ee,
                                              ADReal * sigma) const
{
  const double l = lambda.value(), G = mu.value(), Tv = temperatureAt(ctx).value();
  InelasticIncrements unused;
  // sigma_zz after the return mapping at z, from the values only, with its derivative in z.
  const auto zz = [&](double z)
  {
    std::array<ADReal, 6> strain;
    for (int i = 0; i < 6; ++i)
      strain[i] = ADReal(ee[i].value());
    strain[2] = ADReal::independent(z, 0, 1);
    std::array<ADReal, 6> stress;
    elasticStress(
        Formulation::ThreeDimensional, strain.data(), ADReal(l), ADReal(G), stress.data());
    inelasticReturn(ctx, ADReal(G), ADReal(Tv), old, stress, unused);
    return stress[2];
  };
  double z = ee[2].value();
  const double scale =
      std::abs(sigma[0].value()) + std::abs(sigma[1].value()) + std::abs(sigma[5].value()) + 1e-300;
  double slope = l + 2.0 * G;
  double residual = 0.0;
  for (int it = 0; it < 50; ++it)
  {
    const ADReal r = zz(z);
    slope = r.derivative(0);
    residual = std::abs(r.value());
    if (residual <= 1e-14 * scale)
      break;
    z += -r.value() / slope;
  }
  if (!(residual <= 1e-8 * scale))
    throw InadmissibleStateError("dualmesh: no out-of-plane strain gives a zero out-of-plane "
                                 "stress after the return mapping.");
  // sigma_zz as a function of the unknowns at the solved z, and z(U) = z - sigma_zz(U) / slope.
  std::array<ADReal, 6> strain;
  for (int i = 0; i < 6; ++i)
    strain[i] = ee[i];
  strain[2] = ADReal(z);
  std::array<ADReal, 6> stress;
  elasticStress(Formulation::ThreeDimensional, strain.data(), lambda, mu, stress.data());
  inelasticReturn(ctx, mu, temperatureAt(ctx), old, stress, unused);
  ee[2] = ADReal(z) - (stress[2] - ADReal(stress[2].value())) / slope;
  elasticStress(Formulation::ThreeDimensional, ee, lambda, mu, sigma);
}

void
SmallStrainStress::updateInelastic(QpContext & ctx,
                                   const ADReal & mu,
                                   const double * old,
                                   double * next,
                                   ADReal * ee,
                                   ADReal * sigma) const
{
  InelasticIncrements inc;
  const std::size_t K = _hardening.size();
  if (old)
  {
    std::array<ADReal, 6> stress;
    for (int i = 0; i < 6; ++i)
      stress[i] = sigma[i];
    inelasticReturn(ctx, mu, temperatureAt(ctx), old, stress, inc);
    for (int i = 0; i < 6; ++i)
    {
      sigma[i] = stress[i];
      ee[i] -= inc.creep[i] + inc.plastic[i];
    }
  }
  else
    inc.back.assign(6 * K, 0.0);
  const auto history = [&](int i) { return old ? old[i] : 0.0; };
  for (int i = 0; i < 6; ++i)
  {
    const double creep = history(kCreepStrain + i) + inc.creep[i];
    const double plastic = history(kPlasticStrain + i) + inc.plastic[i];
    if (next)
    {
      next[kCreepStrain + i] = creep;
      next[kPlasticStrain + i] = plastic;
    }
    if (creeps())
      ctx.property(_creep_prop, i) = ADReal(creep);
    if (_plastic)
      ctx.property(_plastic_prop, i) = ADReal(plastic);
  }
  const double creep_equivalent = history(kCreepEquivalent) + inc.creep_equivalent;
  const double plastic_equivalent = history(kPlasticEquivalent) + inc.plastic_equivalent;
  if (next)
  {
    next[kCreepEquivalent] = creep_equivalent;
    next[kPlasticEquivalent] = plastic_equivalent;
    for (std::size_t j = 0; j < 6 * K; ++j)
      next[kBackStress + j] = inc.back[j];
  }
  if (creeps())
    ctx.property(_creep_eq) = ADReal(creep_equivalent);
  if (_plastic)
  {
    ctx.property(_plastic_eq) = ADReal(plastic_equivalent);
    for (int i = 0; i < 6; ++i)
    {
      double total = 0.0;
      for (std::size_t k = 0; k < K; ++k)
        total += inc.back[6 * k + i];
      ctx.property(_back_prop, i) = ADReal(total);
    }
  }
}

} // namespace dualmesh::solid_mechanics
