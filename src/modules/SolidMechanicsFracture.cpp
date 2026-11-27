// SPDX-License-Identifier: LGPL-2.1-or-later
//
// Solid mechanics module: the J-integral and the stress intensity factors of a crack tip in two
// dimensions, by the domain integral and the interaction integral with the Williams field.
#include "dualmesh/base/Problem.h"
#include "dualmesh/fe/Assembly.h"
#include "dualmesh/modules/SolidMechanics.h"

#include <array>
#include <cmath>

namespace dualmesh
{

namespace
{

/// The Williams field of mode I or II at K = 1 at a point (x1, x2) in the axes of the crack: the
/// gradient of its displacement v and its stress s.
struct AuxiliaryField
{
  /// v_i,j, the derivative of v_i by x_j.
  double v11, v21, v12, v22;
  /// s11, s22 and s12.
  std::array<double, 3> s;
};

/// The Williams field of an isotropic material at K = 1, in the axes of the crack, for the shear
/// modulus mu (Pa) and the Kolosov constant kappa.
struct WilliamsField
{
  double mu, kappa;

  /// The displacements (v1, v2) of mode I (m = 0) or II (m = 1).
  void displacement(int m, double x1, double x2, double & v1, double & v2) const
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
  }

  /// The stresses (s11, s22, s12) of mode I (m = 0) or II (m = 1).
  static std::array<double, 3> stress(int m, double x1, double x2)
  {
    const double r = std::hypot(x1, x2), t = std::atan2(x2, x1);
    const double a = 1.0 / std::sqrt(2.0 * M_PI * r);
    const double ch = std::cos(t / 2), sh = std::sin(t / 2), c3 = std::cos(1.5 * t),
                 s3 = std::sin(1.5 * t);
    if (m == 0)
      return {a * ch * (1 - sh * s3), a * ch * (1 + sh * s3), a * sh * ch * c3};
    return {-a * sh * (2 + ch * c3), a * sh * ch * c3, a * ch * (1 - sh * s3)};
  }

  /// The field of mode @p m at the point (x1, x2) at the distance @p r from the tip, with the
  /// displacement gradient by central differences of the step 1e-6 r.
  AuxiliaryField at(int m, double x1, double x2, double r) const
  {
    const double h = 1e-6 * r;
    double a1, a2, b1, b2, d1, d2, e1, e2;
    displacement(m, x1 + h, x2, a1, a2);
    displacement(m, x1 - h, x2, b1, b2);
    displacement(m, x1, x2 + h, d1, d2);
    displacement(m, x1, x2 - h, e1, e2);
    AuxiliaryField out;
    out.v11 = (a1 - b1) / (2 * h);
    out.v21 = (a2 - b2) / (2 * h);
    out.v12 = (d1 - e1) / (2 * h);
    out.v22 = (d2 - e2) / (2 * h);
    out.s = stress(m, x1, x2);
    return out;
  }
};

/// The in-plane stress (s11, s22, s12) of the point of @p ctx in the axes of the crack, whose
/// direction has the cosine @p c and the sine @p sn.
std::array<double, 3>
stressInCrackAxes(const QpContext & ctx, int stress, double c, double sn)
{
  const double sxx = ctx.property(stress, 0).value(), syy = ctx.property(stress, 1).value(),
               sxy = ctx.property(stress, 5).value();
  const double s11 = c * c * sxx + 2 * c * sn * sxy + sn * sn * syy;
  const double s22 = sn * sn * sxx - 2 * c * sn * sxy + c * c * syy;
  const double s12 = (c * c - sn * sn) * sxy + c * sn * (syy - sxx);
  return {s11, s22, s12};
}

/// The displacement gradient G = R G_g R^T of the displacements @p ux and @p uy at the point of
/// @p ctx in the axes of the crack, with G_g the gradient in the global axes and R the rotation
/// to the crack axes, of the cosine @p c and the sine @p sn.
std::array<std::array<double, 2>, 2>
gradientInCrackAxes(const QpContext & ctx, int ux, int uy, double c, double sn)
{
  const double R[2][2] = {{c, sn}, {-sn, c}};
  const double Gg[2][2] = {{ctx.gradient(ux)[0].value(), ctx.gradient(ux)[1].value()},
                           {ctx.gradient(uy)[0].value(), ctx.gradient(uy)[1].value()}};
  std::array<std::array<double, 2>, 2> G{};
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 2; ++j)
      for (int k = 0; k < 2; ++k)
        for (int l = 0; l < 2; ++l)
          G[i][j] += R[i][k] * Gg[k][l] * R[j][l];
  return G;
}

} // namespace

std::array<double, 3>
crackTipIntegrals(Problem & problem,
                  const std::vector<std::string> & displacements,
                  const Point & tip,
                  double angle,
                  double inner_radius,
                  double outer_radius,
                  double youngs_modulus,
                  double poissons_ratio,
                  bool plane_strain)
{
  // The domain integrals of a crack tip in two dimensions, in the axes of the crack (x1 along
  // it, ahead of the tip): with the weight q = 1 inside the inner radius, 0 outside the outer
  // one and linear between,
  //   J = int (sigma_ij u_j,1 - W delta_1i) q,i dA,
  //   I = int (sigma_ij v_j,1 + s_ij u_j,1 - sigma_kl e_kl delta_1i) q,i dA,
  // with (v, s, e) the Williams field of mode I or II at K = 1, and K = E' I / 2.
  problem.initialize();
  const Mesh & mesh = problem.mesh();
  if (mesh.spaceDimension() != 2)
    throw InputError("The crack tip integrals are available in two dimensions.");
  if (!(outer_radius > inner_radius && inner_radius >= 0.0))
    throw InputError("The radii of the crack tip integrals are 0 <= inner < outer.");
  const int ux = problem.variableIndex(displacements.at(0)),
            uy = problem.variableIndex(displacements.at(1));
  const int stress = problem.propertyRegistry().id("stress"),
            strain = problem.propertyRegistry().id("strain");
  const double mu = youngs_modulus / (2.0 * (1.0 + poissons_ratio));
  const double kappa =
      plane_strain ? 3.0 - 4.0 * poissons_ratio : (3.0 - poissons_ratio) / (1.0 + poissons_ratio);
  const double plane =
      plane_strain ? youngs_modulus / (1.0 - poissons_ratio * poissons_ratio) : youngs_modulus;
  const WilliamsField williams{mu, kappa};
  const double c = std::cos(angle), sn = std::sin(angle);
  QuadratureSpec rule;
  rule.points = 4;
  std::vector<IntegrationPoint> pts;
  MappedPoint mp;
  QpContext ctx;
  ctx.dim = 2;
  ctx.time = problem.time();
  double J = 0.0, I[2] = {0.0, 0.0};
  for (Index e = 0; e < problem.numIntegratedElements(); ++e)
  {
    const auto & el = mesh.element(e);
    buildElementPoints(mesh, e, false, PointSet::Volume, rule, pts);
    const auto loc = problem.localValues(el);
    for (const auto & ip : pts)
    {
      mapPoint(mesh, el, ip.xi, mp);
      // The point in the crack axes, and the gradient of q there.
      const double dx = mp.x[0] - tip[0], dy = mp.x[1] - tip[1];
      const double x1 = c * dx + sn * dy, x2 = -sn * dx + c * dy;
      const double r = std::hypot(x1, x2);
      if (!(r > inner_radius && r < outer_radius))
        continue;
      const double dq1 = -x1 / r / (outer_radius - inner_radius),
                   dq2 = -x2 / r / (outer_radius - inner_radius);
      problem.evaluateAt(ctx, e, mp, loc);
      const auto [s11, s22, s12] = stressInCrackAxes(ctx, stress, c, sn);
      const double W = 0.5 * (ctx.property(stress, 0).value() * ctx.property(strain, 0).value() +
                              ctx.property(stress, 1).value() * ctx.property(strain, 1).value() +
                              ctx.property(stress, 2).value() * ctx.property(strain, 2).value() +
                              ctx.property(stress, 5).value() * ctx.property(strain, 5).value());
      const auto G = gradientInCrackAxes(ctx, ux, uy, c, sn);
      const double weight = ip.weight * problem.coordFactor(mp.x);
      // J: (sigma_1j u_j,1 - W) q,1 + sigma_2j u_j,1 q,2.
      J += ((s11 * G[0][0] + s12 * G[1][0] - W) * dq1 + (s12 * G[0][0] + s22 * G[1][0]) * dq2) *
           weight;
      for (int m = 0; m < 2; ++m)
      {
        const AuxiliaryField aux = williams.at(m, x1, x2, r);
        // sigma : e_aux, with e_aux the symmetric part of grad v.
        const double interaction = s11 * aux.v11 + s22 * aux.v22 + s12 * (aux.v12 + aux.v21);
        const double first =
            s11 * aux.v11 + s12 * aux.v21 + aux.s[0] * G[0][0] + aux.s[2] * G[1][0];
        const double second =
            s12 * aux.v11 + s22 * aux.v21 + aux.s[2] * G[0][0] + aux.s[1] * G[1][0];
        I[m] += ((first - interaction) * dq1 + second * dq2) * weight;
      }
    }
  }
  return {J, plane * I[0] / 2.0, plane * I[1] / 2.0};
}

} // namespace dualmesh
