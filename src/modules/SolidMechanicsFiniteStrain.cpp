// SPDX-License-Identifier: LGPL-2.1-or-later
//
// Solid mechanics module, continuum: the stress at finite strain on the undeformed mesh (total
// Lagrangian), by the small-strain law in a frame that rotates with the material (Hughes and
// Winget) or by a hyperelastic law.  It extends small_strain_stress, whose creep and plasticity
// it applies in the rotated frame.
#include "dualmesh/modules/SolidMechanics.h"

#include "dualmesh/base/Problem.h"
#include "dualmesh/core/Coordinates.h"

#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace dualmesh::solid_mechanics
{

namespace
{

/// A 3 x 3 matrix of numbers of type T.
template <typename T> using Mat3 = std::array<std::array<T, 3>, 3>;

template <typename T>
Mat3<T>
identity3()
{
  Mat3<T> m;
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      m[i][j] = T(i == j ? 1.0 : 0.0);
  return m;
}

template <typename A, typename B>
Mat3<ADReal>
multiply3(const Mat3<A> & a, const Mat3<B> & b)
{
  Mat3<ADReal> c;
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
    {
      ADReal sum(0.0);
      for (int k = 0; k < 3; ++k)
        sum = sum + a[i][k] * b[k][j];
      c[i][j] = sum;
    }
  return c;
}

Mat3<ADReal>
transpose3(const Mat3<ADReal> & a)
{
  Mat3<ADReal> t;
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      t[i][j] = a[j][i];
  return t;
}

ADReal
determinant3(const Mat3<ADReal> & a)
{
  return a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1]) -
         a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0]) +
         a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
}

/// The inverse, by the adjugate.
Mat3<ADReal>
inverse3(const Mat3<ADReal> & a, const ADReal & det)
{
  Mat3<ADReal> inv;
  inv[0][0] = (a[1][1] * a[2][2] - a[1][2] * a[2][1]) / det;
  inv[0][1] = (a[0][2] * a[2][1] - a[0][1] * a[2][2]) / det;
  inv[0][2] = (a[0][1] * a[1][2] - a[0][2] * a[1][1]) / det;
  inv[1][0] = (a[1][2] * a[2][0] - a[1][0] * a[2][2]) / det;
  inv[1][1] = (a[0][0] * a[2][2] - a[0][2] * a[2][0]) / det;
  inv[1][2] = (a[0][2] * a[1][0] - a[0][0] * a[1][2]) / det;
  inv[2][0] = (a[1][0] * a[2][1] - a[1][1] * a[2][0]) / det;
  inv[2][1] = (a[0][1] * a[2][0] - a[0][0] * a[2][1]) / det;
  inv[2][2] = (a[0][0] * a[1][1] - a[0][1] * a[1][0]) / det;
  return inv;
}

/// Stress at finite strain, on the undeformed mesh (total Lagrangian).
class FiniteStrainStress : public SmallStrainStress
{
public:
  static InputParameters validParams()
  {
    InputParameters p = SmallStrainStress::validParams();
    p.setClassDescription(
        "Stress of an isotropic material at finite strain, for equations written on the "
        "undeformed mesh (the total Lagrangian form). The deformation gradient F_d = I + grad u "
        "is taken with respect to the undeformed coordinates. The equilibrium equation needs the "
        "first Piola-Kirchhoff stress P = J sigma F_d^-T, with J = det(F_d) and sigma the Cauchy "
        "stress, declared as 'first_piola_kirchhoff_stress' (nine components, row by row), which "
        "stress_divergence reads with stress_property = first_piola_kirchhoff_stress. 'stress' "
        "is sigma. stress_update = rotated_small_strain applies the small-strain law of "
        "small_strain_stress, with its eigenstrains, creep and plasticity, to strain increments "
        "measured in a frame that rotates with the material (the midpoint rule of Hughes and "
        "Winget), so that rigid rotations produce no stress. This update suits materials whose "
        "elastic strains stay small while their total strains are large, such as metals and "
        "ceramics. The other stress updates are hyperelastic: P derives from a strain energy psi "
        "per unit volume of the stress-free state (J/m^3), a function of the invariants I_C = "
        "tr(C_e) and trace_C_squared = tr(C_e^2) of C_e = F_e^T F_e and of J_e = det(F_e), with "
        "F_e the elastic part of F_d = F_e F_eig left after the eigenstrains, F_eig = diag(1 + "
        "eps_eig). Then P_e = 2 psi_1 F_e + 4 psi_2 F_e C_e + psi_J J_e F_e^-T, with psi_1, psi_2 "
        "and psi_J the derivatives of psi by I_C, trace_C_squared and J_e, and P = det(F_eig) P_e "
        "F_eig^-T. neo-Hookean is the compressible neo-Hookean law psi = G/2 (I_C - 3) - G ln(J_e) "
        "+ lambda/2 ln(J_e)^2, with G the shear modulus and lambda the first Lame parameter (Pa). "
        "Saint_Venant_Kirchhoff is psi = lambda/2 tr(E_e)^2 + G tr(E_e^2), with E_e = (C_e - I)/2. "
        "strain_energy takes psi as an expression, whose first and second derivatives the code "
        "takes by forward automatic differentiation, so the Jacobian is exact. In plane stress the "
        "stretch of the thickness F_33 is the root of P_33 = 0, which the code finds at each point "
        "by Newton's method from the last state, and whose derivatives by the unknowns follow "
        "from the implicit function theorem. 'volumetric_strain' is J - 1, the change of volume "
        "per unit volume, and 'strain' is the Green-Lagrange strain. Also declares "
        "'deformation_gradient' (nine components), 'green_lagrange_strain' (Voigt order, "
        "engineering shears) and 'first_piola_kirchhoff_stress_term_size', the size of the terms "
        "whose sum is each component of P, which sets the round-off floor of Newton's test. "
        "Available in plane_strain, axisymmetric, axisymmetric_1d and "
        "three_dimensional, and in plane_stress with a hyperelastic stress update, without "
        "viscoelasticity or phase-field fracture.");
    p.addOptional("stress_update",
                  ParameterKind::String,
                  std::string("rotated_small_strain"),
                  "The finite-strain law: rotated_small_strain, or the hyperelastic neo-Hookean, "
                  "Saint_Venant_Kirchhoff or strain_energy. A hyperelastic law takes no creep or "
                  "plasticity. Default rotated_small_strain, because it carries every law of "
                  "small_strain_stress, creep and plasticity included, to finite rotation and "
                  "strain.");
    p.addOptional("strain_energy",
                  ParameterKind::String,
                  std::string(""),
                  "The strain energy psi (J/m^3) of stress_update = strain_energy, as an "
                  "expression of I_C, trace_C_squared, J (that is J_e), shear_modulus and "
                  "first_Lame_parameter (Pa), the last two from youngs_modulus and "
                  "poissons_ratio. For example, the neo-Hookean energy is "
                  "'shear_modulus/2*(I_C - 3) - shear_modulus*log(J) + "
                  "first_Lame_parameter/2*log(J)^2'. Default: none, for the other stress updates.");
    return p;
  }
  explicit FiniteStrainStress(const InputParameters & p) : SmallStrainStress(p)
  {
    const auto u = p.getString("stress_update");
    if (u == "rotated_small_strain")
      _law = Law::RotatedSmallStrain;
    else if (u == "neo-Hookean")
      _law = Law::NeoHookean;
    else if (u == "Saint_Venant_Kirchhoff")
      _law = Law::SaintVenantKirchhoff;
    else if (u == "strain_energy")
      _law = Law::StrainEnergy;
    else
      throw InputError("'" + name() +
                       "': stress_update must be rotated_small_strain, neo-Hookean, "
                       "Saint_Venant_Kirchhoff or strain_energy, not '" +
                       u + "'.");
    const std::string energy = p.getString("strain_energy");
    if ((_law == Law::StrainEnergy) == energy.empty())
      throw InputError("'" + name() +
                       "': stress_update = strain_energy takes the expression 'strain_energy', "
                       "and the other stress updates take none.");
    if (_law == Law::StrainEnergy)
      _energy = std::make_shared<ParsedExpression>(
          energy,
          std::vector<std::string>{
              "I_C", "trace_C_squared", "J", "shear_modulus", "first_Lame_parameter"});
    const bool hyperelastic = _law != Law::RotatedSmallStrain;
    if (_formulation == Formulation::GeneralizedPlaneStrain ||
        (_formulation == Formulation::PlaneStress && !hyperelastic))
      throw InputError("'" + name() +
                       "': finite strain is available in plane_strain, axisymmetric, "
                       "axisymmetric_1d and three_dimensional problems, and in plane_stress with "
                       "a hyperelastic stress_update (neo-Hookean, Saint_Venant_Kirchhoff or "
                       "strain_energy).");
    if (!_viscous.empty())
      throw InputError("'" + name() + "': viscoelasticity at finite strain is not available.");
    if (_cracking)
      throw InputError("'" + name() + "': phase-field fracture is available at small strain only.");
    // The stress laws at finite strain read the isotropic moduli.
    if (_anisotropic)
      throw InputError("'" + name() +
                       "': an anisotropic stiffness is available at small strain only.");
    if (hyperelastic && _plastic)
      throw InputError("'" + name() +
                       "': plasticity is available with stress_update = rotated_small_strain "
                       "only.");
    if (hyperelastic && creeps())
      throw InputError("'" + name() +
                       "': creep is available with stress_update = "
                       "rotated_small_strain only.");
  }
  /// The history of a point: the rotated elastic strain, F - I and the eigenstrain of the last
  /// committed state, then the history of creep and plasticity (kInelastic), which holds the
  /// creep strain and its equivalent, and with plasticity the plastic strain, its equivalent and
  /// the rotated back stresses.
  int stateSize() const override
  {
    return kInelastic +
           (_plastic ? kBackStress + 6 * static_cast<int>(_hardening.size()) : kPlasticStrain);
  }
  void declareProperties(PropertyRegistry & r) override
  {
    SmallStrainStress::declareProperties(r);
    _pk1 = r.declare("first_piola_kirchhoff_stress", 9);
    _pk1_size = r.declare("first_piola_kirchhoff_stress_term_size", 9);
    _green = r.declare("green_lagrange_strain", 6);
  }

  void computeProperties(QpContext & ctx) const override
  {
    Mat3<ADReal> F = deformationGradientAt(ctx);
    const std::array<ADReal, 6> eigen = eigenstrain(ctx);
    // F = F_e F_eig with F_eig = diag(1 + eps_eig): the eigenstrains here have no shear.
    std::array<ADReal, 3> stretch;
    for (int i = 0; i < 3; ++i)
      stretch[i] = 1.0 + eigen[i];
    const ADReal E = _E.value(ctx);
    const ADReal nu = _nu.value(ctx);
    const auto [lambda, mu] = lameConstants(E, nu);

    // The slice of the point's history record that belongs to this object.
    const double * old = ctx.state_old ? ctx.state_old + stateOffset() : nullptr;
    double * next = ctx.state_new ? ctx.state_new + stateOffset() : nullptr;
    if (_formulation == Formulation::PlaneStress)
      F[2][2] = thicknessStretch(ctx, F, stretch, mu, lambda, old);

    const ADReal J = determinant3(F);
    if (!(J.value() > 0.0))
      throw InadmissibleStateError("'" + name() +
                                   "': the deformation turned an element inside out (det F <= 0).");
    Response response;
    response.step.back.assign(6 * _hardening.size(), 0.0);
    if (_law != Law::RotatedSmallStrain)
      hyperelasticResponse(F, J, stretch, mu, lambda, response);
    else
      rotatedSmallStrainResponse(ctx, F, J, eigen, mu, lambda, old, response);
    if (next)
      writeHistory(F, eigen, response, old, next);
    writeProperties(ctx, F, J, response, old);
  }

private:
  /// The layout of the history of a point: the rotated elastic strain (six values), F - I (nine
  /// values, row by row), the eigenstrain (six values), then the history of creep and plasticity
  /// in the layout of small_strain_stress (kCreepStrain and the others, from kInelastic).
  static constexpr int kElasticStrain = 0, kDeformation = 6, kEigenstrain = 15, kInelastic = 21;

  /// The stress and the strain of a point in the step.
  struct Response
  {
    /// The Cauchy stress and the elastic strain, in Voigt order.
    std::array<ADReal, 6> sigma, elastic;
    /// The first Piola-Kirchhoff stress, and the size of the terms of each of its components,
    /// the scale of its round-off.
    Mat3<ADReal> P;
    Mat3<double> P_size;
    /// The increments of creep and plasticity.
    InelasticIncrements step;
  };

  /// The slot @p slot of the history of creep and plasticity at the start of the step, or 0 at a
  /// point without a history.
  static double inelasticOld(const double * old, int slot)
  {
    return old ? old[kInelastic + slot] : 0.0;
  }

  /// The deformation gradient F = I + grad u on the undeformed configuration.
  Mat3<ADReal> deformationGradientAt(const QpContext & ctx) const
  {
    Mat3<ADReal> F = identity3<ADReal>();
    const auto & gu = ctx.gradient(_disp[0]);
    const double R = ctx.x[0];
    if (_formulation == Formulation::AxisymmetricOneD)
    {
      F[0][0] = 1.0 + gu[0];
      F[1][1] = ADReal(1.0 + _axial->value(ctx.x, ctx.time));
      F[2][2] = 1.0 + hoopStrain(ctx.value(_disp[0]), gu[0], R);
      return F;
    }
    const auto & gv = ctx.gradient(_disp[1]);
    F[0][0] = 1.0 + gu[0];
    F[0][1] = gu[1];
    F[1][0] = gv[0];
    F[1][1] = 1.0 + gv[1];
    if (_formulation == Formulation::Axisymmetric)
      F[2][2] = 1.0 + hoopStrain(ctx.value(_disp[0]), gu[0], R);
    else if (_formulation == Formulation::ThreeDimensional)
    {
      const auto & gw = ctx.gradient(_disp[2]);
      F[0][2] = gu[2];
      F[1][2] = gv[2];
      F[2][0] = gw[0];
      F[2][1] = gw[1];
      F[2][2] = 1.0 + gw[2];
    }
    return F;
  }

  /// The sum of the eigenstrains, in Voigt order.
  std::array<ADReal, 6> eigenstrain(const QpContext & ctx) const
  {
    std::array<ADReal, 6> eigen;
    eigen.fill(ADReal(0.0));
    for (int id : _eigen)
      for (int i = 0; i < 6; ++i)
        eigen[i] = eigen[i] + ctx.property(id, i);
    return eigen;
  }

  /// The response of a hyperelastic law: P of the strain energy, the Cauchy stress sigma = P F^T
  /// / J and the elastic strain, the Green-Lagrange strain of F_e with engineering shears.
  void hyperelasticResponse(const Mat3<ADReal> & F,
                            const ADReal & J,
                            const std::array<ADReal, 3> & stretch,
                            const ADReal & mu,
                            const ADReal & lambda,
                            Response & out) const
  {
    out.P = hyperelasticStress(F, stretch, mu, lambda, &out.P_size);
    const Mat3<ADReal> S = multiply3(out.P, transpose3(F));
    for (int i = 0; i < 3; ++i)
      for (int j = i; j < 3; ++j)
        out.sigma[kVoigt[i][j]] = S[i][j] / J;
    const Mat3<ADReal> Fe = elasticPart(F, stretch);
    const Mat3<ADReal> Ce = multiply3(transpose3(Fe), Fe);
    for (int i = 0; i < 3; ++i)
      for (int j = i; j < 3; ++j)
        out.elastic[kVoigt[i][j]] = i == j ? 0.5 * (Ce[i][j] - 1.0) : Ce[i][j];
  }

  /// The increment of the displacement gradient on the midpoint configuration, G = (F - F_0)
  /// ((F + F_0) / 2)^-1, and the rotation increment Q = (I - W/2)^-1 (I + W/2), with W the skew
  /// part of G (Hughes and Winget).
  static void midpointIncrement(const Mat3<ADReal> & F,
                                const Mat3<double> & F0,
                                Mat3<ADReal> & G,
                                Mat3<ADReal> & Q)
  {
    Mat3<ADReal> mid, dF;
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j)
      {
        mid[i][j] = 0.5 * (F[i][j] + F0[i][j]);
        dF[i][j] = F[i][j] - F0[i][j];
      }
    G = multiply3(dF, inverse3(mid, determinant3(mid)));
    Mat3<ADReal> minus = identity3<ADReal>(), plus = identity3<ADReal>();
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j)
      {
        const ADReal w = 0.5 * (G[i][j] - G[j][i]);
        minus[i][j] = minus[i][j] - 0.5 * w;
        plus[i][j] = plus[i][j] + 0.5 * w;
      }
    Q = multiply3(inverse3(minus, determinant3(minus)), plus);
  }

  /// The tensor Q a Q^T of the symmetric tensor @p a of the last state, in Voigt order, with
  /// @p shear the factor of its shear components (1/2 for engineering shears).
  static Mat3<ADReal> rotated(const Mat3<ADReal> & Q, const double * a, double shear)
  {
    Mat3<double> t;
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j)
        t[i][j] = a[kVoigt[i][j]] * (i == j ? 1.0 : shear);
    return multiply3(multiply3(Q, t), transpose3(Q));
  }

  /// The response of rotated_small_strain: the small-strain law, with creep and plasticity, on
  /// the strain increment of the midpoint configuration and the elastic strain and back stresses
  /// of the last state, rotated by the rotation increment.  P = J sigma F^-T.
  void rotatedSmallStrainResponse(QpContext & ctx,
                                  const Mat3<ADReal> & F,
                                  const ADReal & J,
                                  const std::array<ADReal, 6> & eigen,
                                  const ADReal & mu,
                                  const ADReal & lambda,
                                  const double * old,
                                  Response & out) const
  {
    Mat3<double> F0 = identity3<double>();
    std::array<double, 6> elastic_old{0, 0, 0, 0, 0, 0}, eigen_old{0, 0, 0, 0, 0, 0};
    if (old)
    {
      for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
          F0[i][j] += old[kDeformation + 3 * i + j];
      for (int i = 0; i < 6; ++i)
      {
        elastic_old[i] = old[kElasticStrain + i];
        eigen_old[i] = old[kEigenstrain + i];
      }
    }
    Mat3<ADReal> G, Q;
    midpointIncrement(F, F0, G, Q);
    // The rotated elastic strain of the last state (as a tensor, half shears), plus the strain
    // increment, less the increment of the eigenstrains.
    const Mat3<ADReal> rotated_elastic = rotated(Q, elastic_old.data(), 0.5);
    for (int i = 0; i < 3; ++i)
      for (int j = i; j < 3; ++j)
      {
        const int k = kVoigt[i][j];
        const double factor = i == j ? 1.0 : 2.0; // back to engineering shear
        const ADReal increment = factor * 0.5 * (G[i][j] + G[j][i]);
        out.elastic[k] = factor * rotated_elastic[i][j] + increment - (eigen[k] - eigen_old[k]);
      }
    const ADReal trace = out.elastic[0] + out.elastic[1] + out.elastic[2];
    for (int i = 0; i < 6; ++i)
      out.sigma[i] = i < 3 ? lambda * trace + 2.0 * mu * out.elastic[i] : mu * out.elastic[i];
    if (inelastic() && old)
    {
      // The back stresses of the last state, rotated as the elastic strain is.
      const std::size_t K = _hardening.size();
      std::vector<ADReal> back(6 * K);
      for (std::size_t k = 0; k < K; ++k)
      {
        const Mat3<ADReal> rb = rotated(Q, old + kInelastic + kBackStress + 6 * k, 1.0);
        for (int i = 0; i < 3; ++i)
          for (int j = i; j < 3; ++j)
            back[6 * k + kVoigt[i][j]] = rb[i][j];
      }
      std::array<ADReal, 6> stress;
      for (int i = 0; i < 6; ++i)
        stress[i] = out.sigma[i];
      inelasticReturn(ctx,
                      mu,
                      temperatureAt(ctx),
                      old[kInelastic + kCreepEquivalent],
                      _plastic ? old[kInelastic + kPlasticEquivalent] : 0.0,
                      back,
                      stress,
                      out.step);
      for (int i = 0; i < 6; ++i)
      {
        out.sigma[i] = stress[i];
        out.elastic[i] = out.elastic[i] - out.step.creep[i] - out.step.plastic[i];
      }
    }
    Mat3<ADReal> S;
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j)
        S[i][j] = out.sigma[kVoigt[i][j]];
    out.P = multiply3(S, transpose3(inverse3(F, J)));
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j)
      {
        out.P[i][j] = J * out.P[i][j];
        out.P_size[i][j] = std::abs(out.P[i][j].value());
      }
  }

  /// The history of the point at the end of the step.
  void writeHistory(const Mat3<ADReal> & F,
                    const std::array<ADReal, 6> & eigen,
                    const Response & response,
                    const double * old,
                    double * next) const
  {
    const InelasticIncrements & step = response.step;
    for (int i = 0; i < 6; ++i)
      next[kElasticStrain + i] = response.elastic[i].value();
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j)
        next[kDeformation + 3 * i + j] = F[i][j].value() - (i == j ? 1.0 : 0.0);
    double * inelastic_next = next + kInelastic;
    for (int i = 0; i < 6; ++i)
    {
      next[kEigenstrain + i] = eigen[i].value();
      inelastic_next[kCreepStrain + i] = inelasticOld(old, kCreepStrain + i) + step.creep[i];
    }
    inelastic_next[kCreepEquivalent] = inelasticOld(old, kCreepEquivalent) + step.creep_equivalent;
    if (!_plastic)
      return;
    for (int i = 0; i < 6; ++i)
      inelastic_next[kPlasticStrain + i] = inelasticOld(old, kPlasticStrain + i) + step.plastic[i];
    inelastic_next[kPlasticEquivalent] =
        inelasticOld(old, kPlasticEquivalent) + step.plastic_equivalent;
    for (std::size_t j = 0; j < step.back.size(); ++j)
      inelastic_next[kBackStress + j] = step.back[j];
  }

  /// The declared properties of the point.
  void writeProperties(QpContext & ctx,
                       const Mat3<ADReal> & F,
                       const ADReal & J,
                       const Response & response,
                       const double * old) const
  {
    const InelasticIncrements & step = response.step;
    if (_plastic)
    {
      for (int i = 0; i < 6; ++i)
        ctx.property(_plastic_prop, i) =
            ADReal(inelasticOld(old, kPlasticStrain + i) + step.plastic[i]);
      ctx.property(_plastic_eq) =
          ADReal(inelasticOld(old, kPlasticEquivalent) + step.plastic_equivalent);
      for (int i = 0; i < 6; ++i)
      {
        double total = 0.0;
        for (std::size_t k = 0; k < _hardening.size(); ++k)
          total += step.back[6 * k + i];
        ctx.property(_back_prop, i) = ADReal(total);
      }
    }
    if (creeps())
    {
      for (int i = 0; i < 6; ++i)
        ctx.property(_creep_prop, i) = ADReal(inelasticOld(old, kCreepStrain + i) + step.creep[i]);
      ctx.property(_creep_eq) = ADReal(inelasticOld(old, kCreepEquivalent) + step.creep_equivalent);
    }
    // Green-Lagrange strain E = (F^T F - I) / 2, engineering shears.
    const Mat3<ADReal> C = multiply3(transpose3(F), F);
    for (int i = 0; i < 3; ++i)
      for (int j = i; j < 3; ++j)
      {
        const int k = kVoigt[i][j];
        ctx.property(_green, k) = (i == j ? 0.5 * (C[i][j] - 1.0) : C[i][j]);
        ctx.property(_strain, k) = ctx.property(_green, k);
      }
    for (int i = 0; i < 6; ++i)
    {
      ctx.property(_elastic, i) = response.elastic[i];
      ctx.property(_stress, i) = response.sigma[i];
    }
    ctx.property(_volumetric) = J - 1.0;
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j)
      {
        ctx.property(_pk1, 3 * i + j) = response.P[i][j];
        ctx.property(_pk1_size, 3 * i + j) = ADReal(response.P_size[i][j]);
        ctx.property(_F_prop, 3 * i + j) = F[i][j];
      }
    ctx.property(_vm) = vonMisesStress(response.sigma.data());
  }

  /// The most iterations of the Newton method for the stretch of the thickness.
  static constexpr int kMaxIterations = 50;

  /// The elastic part F_e = F F_eig^-1 of @p F, with F_eig = diag(@p stretch).
  static Mat3<ADReal> elasticPart(const Mat3<ADReal> & F, const std::array<ADReal, 3> & stretch)
  {
    Mat3<ADReal> Fe;
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j)
        Fe[i][j] = F[i][j] / stretch[j];
    return Fe;
  }

  /// The derivatives psi_1, psi_2 and psi_J of the strain energy by I_C, trace_C_squared and J_e.
  std::array<ADReal, 3> energyDerivatives(const ADReal & I1,
                                          const ADReal & I2,
                                          const ADReal & Je,
                                          const ADReal & mu,
                                          const ADReal & lambda) const
  {
    if (_law == Law::NeoHookean)
      return {0.5 * mu, ADReal(0.0), (lambda * log(Je) - mu) / Je};
    if (_law == Law::SaintVenantKirchhoff)
      return {0.25 * lambda * (I1 - 3.0) - 0.5 * mu, 0.25 * mu, ADReal(0.0)};
    // The expression: its gradient g and Hessian H at the values a_0 of its arguments a, and the
    // first derivatives as functions of the unknowns, g_k + sum_m H_km (a_m - a_0m), which are
    // exact to first order, as the Jacobian needs.
    const std::array<const ADReal *, 5> a{&I1, &I2, &Je, &mu, &lambda};
    double values[5], value = 0.0, g[5], H[25];
    for (int m = 0; m < 5; ++m)
      values[m] = a[m]->value();
    _energy->secondOrder(values, value, g, H);
    std::array<ADReal, 3> d;
    for (int k = 0; k < 3; ++k)
    {
      d[k] = ADReal(g[k]);
      for (int m = 0; m < 5; ++m)
        if (H[5 * k + m] != 0.0)
          d[k] = d[k] + H[5 * k + m] * (*a[m] - values[m]);
    }
    return d;
  }

  /// The first Piola-Kirchhoff stress P = det(F_eig) P_e F_eig^-T of a hyperelastic law, with
  /// F_eig = diag(@p stretch).  @p sizes, when given, receives the size of the terms of each
  /// component of P, the scale of its round-off: at a small strain the terms are of the size of
  /// the modulus and cancel to a far smaller stress.
  Mat3<ADReal> hyperelasticStress(const Mat3<ADReal> & F,
                                  const std::array<ADReal, 3> & stretch,
                                  const ADReal & mu,
                                  const ADReal & lambda,
                                  Mat3<double> * sizes = nullptr) const
  {
    const Mat3<ADReal> Fe = elasticPart(F, stretch);
    const Mat3<ADReal> Ce = multiply3(transpose3(Fe), Fe);
    const Mat3<ADReal> FeCe = multiply3(Fe, Ce);
    ADReal I1(0.0), I2(0.0);
    for (int i = 0; i < 3; ++i)
    {
      I1 = I1 + Ce[i][i];
      for (int j = 0; j < 3; ++j)
        I2 = I2 + Ce[i][j] * Ce[i][j];
    }
    const ADReal Je = determinant3(Fe);
    const Mat3<ADReal> FeinvT = transpose3(inverse3(Fe, Je));
    const auto [psi1, psi2, psiJ] = energyDerivatives(I1, I2, Je, mu, lambda);
    const ADReal Jeig = stretch[0] * stretch[1] * stretch[2];
    Mat3<ADReal> P;
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j)
        P[i][j] = Jeig *
                  (2.0 * psi1 * Fe[i][j] + 4.0 * psi2 * FeCe[i][j] + psiJ * Je * FeinvT[i][j]) /
                  stretch[j];
    if (sizes)
      for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
          (*sizes)[i][j] = std::abs(Jeig.value() / stretch[j].value()) *
                           (std::abs(2.0 * psi1.value() * Fe[i][j].value()) +
                            std::abs(4.0 * psi2.value() * FeCe[i][j].value()) +
                            std::abs(psiJ.value() * Je.value() * FeinvT[i][j].value()));
    return P;
  }

  /// The stretch of the thickness F_33 of plane stress, the root of P_33 = 0.  Newton's method
  /// finds it from the values of the other inputs, starting from the last state, and the root
  /// as a function of the unknowns follows from the implicit function theorem: one more Newton
  /// step from the root, t - P_33(U, t) / (dP_33/dF_33), whose derivative by the unknowns U is
  /// -(dP_33/dU) / (dP_33/dF_33).
  ADReal thicknessStretch(const QpContext & ctx,
                          const Mat3<ADReal> & F,
                          const std::array<ADReal, 3> & stretch,
                          const ADReal & mu,
                          const ADReal & lambda,
                          const double * old) const
  {
    Mat3<ADReal> plain;
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j)
        plain[i][j] = ADReal(F[i][j].value());
    const std::array<ADReal, 3> plain_stretch{
        ADReal(stretch[0].value()), ADReal(stretch[1].value()), ADReal(stretch[2].value())};
    const ADReal plain_mu(mu.value()), plain_lambda(lambda.value());
    const double tolerance = 64.0 * std::numeric_limits<double>::epsilon();
    // The F_33 of the last state, stored as F - I.
    double t = old ? 1.0 + old[kDeformation + 8] : 1.0;
    double slope = 0.0;
    for (int iteration = 0;; ++iteration)
    {
      plain[2][2] = ADReal::independent(t, 0, 1);
      Mat3<double> sizes;
      const ADReal r =
          hyperelasticStress(plain, plain_stretch, plain_mu, plain_lambda, &sizes)[2][2];
      const double size = sizes[2][2];
      slope = r.derivative(0);
      if (!std::isfinite(r.value()) || !std::isfinite(slope) || !(slope > 0.0))
        thicknessFailure(ctx,
                         "P_33 or its derivative is not finite, or the derivative is not positive");
      if (std::abs(r.value()) <= tolerance * size)
        break;
      if (iteration == kMaxIterations)
        thicknessFailure(ctx, std::to_string(kMaxIterations) + " iterations did not reach it");
      double step = -r.value() / slope;
      if (!std::isfinite(step))
        thicknessFailure(ctx, "the Newton step is not finite");
      while (!(t + step > 0.0))
        step *= 0.5;
      t += step;
    }
    Mat3<ADReal> at = F;
    at[2][2] = ADReal(t);
    return t - hyperelasticStress(at, stretch, mu, lambda)[2][2] / slope;
  }

  [[noreturn]] void thicknessFailure(const QpContext & ctx, const std::string & why) const
  {
    throw InadmissibleStateError("'" + name() +
                                 "': no stretch of the thickness gives a zero out-of-plane "
                                 "stress at the point (" +
                                 std::to_string(ctx.x[0]) + ", " + std::to_string(ctx.x[1]) + ", " +
                                 std::to_string(ctx.x[2]) + "): " + why + ".");
  }

  enum class Law
  {
    RotatedSmallStrain,
    NeoHookean,
    SaintVenantKirchhoff,
    StrainEnergy
  };
  Law _law = Law::RotatedSmallStrain;
  std::shared_ptr<ParsedExpression> _energy;
  int _pk1 = -1, _pk1_size = -1, _green = -1;
};

} // namespace

void
registerFiniteStrainObjects(Factory & f)
{
  f.add<FiniteStrainStress>("finite_strain_stress", ObjectCategory::Property, "solid_mechanics");
}

} // namespace dualmesh::solid_mechanics
