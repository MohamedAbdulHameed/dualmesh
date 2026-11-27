// SPDX-License-Identifier: LGPL-2.1-or-later
//
// Solid mechanics module, continuum: the declarations that its source files share.
//
//   SolidMechanics.cpp              the elastic laws, viscoelasticity, phase-field fracture, the
//                                   kernels, the boundary conditions, the thermal eigenstrain and
//                                   registerSolidMechanicsObjects, the entry point of the module
//   SolidMechanicsInelastic.cpp     creep and plasticity: the laws and their return maps
//   SolidMechanicsFiniteStrain.cpp  the stress at finite strain
//   SolidMechanicsInterfaces.cpp    mortar contact, the cohesive zone and the normal displacement
//   SolidMechanicsStructures.cpp    the beams, plates, trusses and shells, the reduced theories of
//                                   the module, and the stresses of their sections
//   SolidMechanicsFracture.cpp      the crack tip integrals
//   SolidMechanicsSystem.cpp        the rigid body modes for the linear solvers (SolidSystem.h)
//
// Stresses and strains of six components are in the Voigt order 0: xx, 1: yy, 2: zz (the hoop
// direction in the axisymmetric case), 3: yz, 4: xz, 5: xy, with engineering shear strains and
// tensor shear stresses.
#pragma once

#include "dualmesh/base/Factory.h"
#include "dualmesh/base/FieldExpression.h"
#include "dualmesh/base/Kernel.h"
#include "dualmesh/base/Property.h"
#include "dualmesh/core/ParsedFunction.h"

#include <array>
#include <cmath>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

namespace dualmesh::solid_mechanics
{

/// The continuum formulations: the idealisation of the body and the strains it has.
enum class Formulation
{
  PlaneStress,
  PlaneStrain,
  GeneralizedPlaneStrain,
  Axisymmetric,
  AxisymmetricOneD,
  ThreeDimensional
};

/// The Voigt index of the tensor component (i, j).
inline constexpr int kVoigt[3][3] = {{0, 5, 4}, {5, 1, 3}, {4, 3, 2}};

/// The Lame constants of an isotropic material.
template <typename T> struct LameConstants
{
  T lambda, mu;
};

/// The Lame constants lambda = E nu / ((1 + nu) (1 - 2 nu)) and mu = E / (2 (1 + nu)), the shear
/// modulus, of Young's modulus @p E and Poisson's ratio @p nu.
template <typename T>
LameConstants<T>
lameConstants(const T & E, const T & nu)
{
  return {E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu)), E / (2.0 * (1.0 + nu))};
}

/// The isotropic small-strain stress @p sigma of the elastic strain @p ee (six values each) for
/// the Lame constants @p lambda and @p mu.  In plane stress sigma_zz = 0, and ee[2] is set to the
/// elastic strain that makes it so.  In the other formulations ee[2] is a strain of the
/// displacements and stays as it is.  The stress is accumulated in place: this runs at every
/// integration point, and a temporary value with its derivatives costs as much as the arithmetic.
template <typename Real>
void
isotropicStress(Formulation f, ADReal * ee, const Real & lambda, const Real & mu, ADReal * sigma)
{
  const auto scale = [](ADReal & s, const ADReal & x, const Real & c)
  {
    if constexpr (std::is_same_v<Real, double>)
      s.addScaledBy(x, c);
    else
      s += c * x;
  };
  ADReal trace = ee[0];
  trace += ee[1];
  if (f == Formulation::PlaneStress)
  {
    // lambda* = 2 mu lambda / (lambda + 2 mu) is the plane-stress Lame constant, so that
    // sigma_xx = E / (1 - nu^2) (eps_xx + nu eps_yy), and eps_zz = -lambda / (lambda + 2 mu)
    // (eps_xx + eps_yy).
    const Real plane = 2.0 * mu * lambda / (lambda + 2.0 * mu);
    for (int i = 0; i < 2; ++i)
    {
      sigma[i] = 0.0;
      scale(sigma[i], trace, plane);
      scale(sigma[i], ee[i], 2.0 * mu);
    }
    sigma[2] = 0.0;
    ee[2] = 0.0;
    scale(ee[2], trace, -lambda / (lambda + 2.0 * mu));
  }
  else
  {
    trace += ee[2];
    for (int i = 0; i < 3; ++i)
    {
      sigma[i] = 0.0;
      scale(sigma[i], trace, lambda);
      scale(sigma[i], ee[i], 2.0 * mu);
    }
  }
  for (int i = 3; i < 6; ++i)
  {
    sigma[i] = 0.0;
    scale(sigma[i], ee[i], mu);
  }
}

/// The stress sigma = C eps_e of an anisotropic material, with C the 6 x 6 stiffness, row by row.
/// In plane stress the out-of-plane strains eps_zz, gamma_yz and gamma_xz (ee[2], ee[3], ee[4])
/// are set to those that make sigma_zz, sigma_yz and sigma_xz zero.
void
anisotropicStress(Formulation f, ADReal * ee, const std::array<double, 36> & C, ADReal * sigma);

/// The parameter that lists the displacements of an object that acts on each of them.
void addDisplacementsParam(InputParameters & p, const std::string & what);

/// The deviator of the symmetric tensor @p s: the normal components less their mean, and the
/// shear components as they are.
template <typename T>
std::array<T, 6>
deviator(const T * s)
{
  const T mean = (s[0] + s[1] + s[2]) / 3.0;
  std::array<T, 6> dev;
  for (int i = 0; i < 6; ++i)
    dev[i] = i < 3 ? s[i] - mean : s[i];
  return dev;
}

/// The von Mises stress sqrt(3 J_2) of the stress @p s.  J_2 is formed from the differences of
/// the normal components, so that it is exactly zero for a hydrostatic stress.
template <typename T>
T
vonMisesStress(const T * s)
{
  const T j2 = ((s[0] - s[1]) * (s[0] - s[1]) + (s[1] - s[2]) * (s[1] - s[2]) +
                (s[2] - s[0]) * (s[2] - s[0])) /
                   6.0 +
               s[3] * s[3] + s[4] * s[4] + s[5] * s[5];
  using std::sqrt;
  return sqrt(3.0 * j2);
}

/// The isotropic small-strain stress of a solid in every continuum formulation: the elastic
/// strain is the strain of the displacements less the eigenstrains (a thermal strain, for
/// example) and the creep strain, and Young's modulus and Poisson's ratio may be constants,
/// functions, properties or expressions of the fields.  The elastic law, viscoelasticity and
/// phase-field fracture are in SolidMechanics.cpp, creep and plasticity in
/// SolidMechanicsInelastic.cpp.
class SmallStrainStress : public Property
{
public:
  static InputParameters validParams();
  explicit SmallStrainStress(const InputParameters & p);
  /// The history of a point: the creep strain and its equivalent, the plastic strain and its
  /// equivalent, and the back stresses (InelasticIncrements), then the viscous branches and the
  /// fracture driving energy.
  int stateSize() const override { return inelasticSize() + viscousSize() + (_cracking ? 1 : 0); }
  void initialSetup(Problem & problem) override;
  void declareProperties(PropertyRegistry & r) override;
  void computeProperties(QpContext & ctx) const override;

protected:
  /// The layout of the history of creep and plasticity, from its start: the creep strain (six
  /// values), the equivalent creep strain, the plastic strain (six values), the equivalent plastic
  /// strain and the back stresses (six values each).
  static constexpr int kCreepStrain = 0, kCreepEquivalent = 6, kPlasticStrain = 7,
                       kPlasticEquivalent = 13, kBackStress = 14;

  /// The increments of one step at a point.
  struct InelasticIncrements
  {
    std::array<double, 6> creep{0, 0, 0, 0, 0, 0}, plastic{0, 0, 0, 0, 0, 0};
    double creep_equivalent = 0.0, plastic_equivalent = 0.0;
    /// The back stresses at the end of the step, six values each.
    std::vector<double> back;
  };

  /// The size of the history of creep and plasticity, the first part of the state.
  int inelasticSize() const
  {
    return inelastic() ? kBackStress + 6 * static_cast<int>(_hardening.size()) : 0;
  }
  /// The size of the history of the viscous branches, which follows that of creep and plasticity.
  int viscousSize() const
  {
    return _viscous.empty() ? 0
                            : 6 + 6 * static_cast<int>(_shear_branches) +
                                  static_cast<int>(_viscous.size() - _shear_branches);
  }
  /// Whether the material creeps.
  bool creeps() const { return _creep_expression != nullptr; }
  /// Whether the material has a history (creep or plasticity).
  bool inelastic() const { return creeps() || _plastic; }
  /// The temperature at the point, or 0 for a material that reads none.
  ADReal temperatureAt(const QpContext & ctx) const
  {
    return _T >= 0 ? ctx.value(_T) : ADReal(0.0);
  }

  /// The inelastic return of one step from the trial stress @p sigma and the history @p old
  /// (the layout of stateSize): creep alone, plasticity alone, or both, in one return.  The
  /// creep acts in a step of time only, the plasticity in a steady solve as well.
  void inelasticReturn(QpContext & ctx,
                       const ADReal & mu,
                       const ADReal & T,
                       double creep_old,
                       double plastic_old,
                       const std::vector<ADReal> & back_old,
                       std::array<ADReal, 6> & sigma,
                       InelasticIncrements & inc) const;

  /// The creep rate of creep_model = parsed, or null for a material that does not creep.
  std::shared_ptr<ParsedExpression> _creep_expression;
  std::vector<double> _creep_constants;
  Formulation _formulation;
  FieldCoefficient _E, _nu;
  bool _constant_moduli = false;
  double _lambda0 = 0.0, _mu0 = 0.0;
  double _grain = 0.0, _creep_factor = 1.0;
  int _T = -1;
  std::vector<int> _disp, _eigen;
  FunctionPtr _axial;
  int _stress = -1, _strain = -1, _elastic = -1, _volumetric = -1, _vm = -1, _F_prop = -1;
  int _creep_prop = -1, _creep_eq = -1;
  bool _plastic = false;
  std::shared_ptr<ParsedExpression> _yield_expression;
  std::vector<double> _plasticity_constants;
  struct BackStress
  {
    double modulus, recovery;
  };
  std::vector<BackStress> _hardening;
  int _plastic_prop = -1, _plastic_eq = -1, _back_prop = -1;
  bool _anisotropic = false, _hill_given = false;
  bool _cracking = false;
  int _damage = -1, _driving = -1;
  struct ViscousBranch
  {
    double modulus, time;
  };
  /// The viscous branches: the shear ones first, then the bulk ones.
  std::vector<ViscousBranch> _viscous;
  std::size_t _shear_branches = 0;
  int _out_of_plane = -1;
  std::array<double, 36> _stiffness{}, _hill{};
  /// The registry of the problem, which knows the outputs that some object reads.
  const PropertyRegistry * _registry = nullptr;

private:
  // The elastic law, viscoelasticity and phase-field fracture (SolidMechanics.cpp).

  // The parts of the constructor, in the order of its checks.

  /// Refuse a constant Young's modulus or Poisson's ratio out of its range.
  void checkModuli(const InputParameters & p) const;
  /// The creep rate, the grain size and the creep rate factor.
  void parseCreep(const InputParameters & p);
  /// The yield stress and the back stresses of kinematic hardening.
  void parsePlasticity(const InputParameters & p);
  /// The viscous branches of generalized Maxwell.
  void parseViscous(const InputParameters & p);
  /// Phase-field fracture, and the materials it is available for.
  void parseDamage(const InputParameters & p);
  /// An anisotropic stiffness in place of Young's modulus and Poisson's ratio.
  void parseStiffness(const InputParameters & p);
  /// Hill's equivalent stress, von Mises by default.
  void parseHill(const InputParameters & p);
  /// Refuse kinematic hardening with an anisotropic stiffness or with creep.
  void checkHardening() const;

  /// Refuse a matrix that is not symmetric and positive definite (by its Cholesky factor).
  void checkSymmetricPositive(const std::array<double, 36> & A, const std::string & what) const;
  /// The strain of the displacements into @p e (engineering shears).  The zz strain of plane
  /// stress follows from the stress and is set later.
  void displacementStrain(const QpContext & ctx, ADReal * e) const;
  /// The stress of the viscous branches of generalized Maxwell, added to @p sigma: with x =
  /// dt / tau, a deviatoric branch s = exp(-x) s_n + 2 G (1 - exp(-x)) / x (e - e_n) and a bulk
  /// branch p = exp(-x) p_n + K (1 - exp(-x)) / x (theta - theta_n), the exact update for a strain
  /// that varies linearly over the step (Herrmann and Taylor).  A steady state is relaxed: the
  /// branches carry no stress.  The history is the elastic strain and the branch stresses.
  void addViscousStress(
      QpContext & ctx, const ADReal * ee, const double * old, double * next, ADReal * sigma) const;
  /// Phase-field fracture: the stress (1 - d)^2 sigma+ + sigma- of the split of Miehe, with the
  /// principal strains of the in-plane pair in closed form and the out-of-plane strain the
  /// third, and the history H = max(H_n, psi+), which drives the damage.
  void crackedStress(QpContext & ctx,
                     const ADReal * ee,
                     const ADReal & lambda,
                     const ADReal & mu,
                     const double * old,
                     double * next,
                     ADReal * sigma) const;
  /// F = I + grad u, row by row, for equations that account for the change of shape (heat
  /// conduction, follower pressure) while the stress stays that of small strain.
  void deformationGradient(QpContext & ctx, const ADReal * e) const;
  /// The elastic stress of the elastic strain @p ee: anisotropic or isotropic.
  template <typename Real>
  void elasticStress(
      Formulation f, ADReal * ee, const Real & lambda, const Real & mu, ADReal * sigma) const
  {
    if (_anisotropic)
      anisotropicStress(f, ee, _stiffness, sigma);
    else
      isotropicStress(f, ee, lambda, mu, sigma);
  }

  // Creep and plasticity (SolidMechanicsInelastic.cpp).

  /// inelasticReturn with the history of the small-strain layout (stateSize).
  void inelasticReturn(QpContext & ctx,
                       const ADReal & mu,
                       const ADReal & T,
                       const double * old,
                       std::array<ADReal, 6> & sigma,
                       InelasticIncrements & inc) const;
  /// The yield stress at the equivalent plastic strain p and the temperature T.
  template <typename T> T yieldStress(const T & p, const T & temperature) const;
  /// The equivalent stress sqrt(sigma . P sigma) of Hill (von Mises for the isotropic P), and
  /// its gradient @p normal.
  template <typename T>
  T hillStress(const std::array<T, 6> & sigma, std::array<T, 6> & normal) const;
  /// The closest-point return of creep and plasticity for an anisotropic stiffness or Hill's
  /// equivalent stress q(sigma): the stress, the creep increment dc and the plastic increment dp
  /// solve S sigma - eps_trial + (dc + dp) dq/dsigma = 0, dc = dt rate(q, T, c + dc) and,
  /// where the point yields, q = sigma_y(p + dp), by Newton's method on the eight unknowns at
  /// the point.  Their derivatives follow from the implicit function theorem.
  void generalReturn(QpContext & ctx,
                     const ADReal & T,
                     double c_old,
                     double p_old,
                     std::array<ADReal, 6> & sigma,
                     InelasticIncrements & inc) const;
  /// J2 plasticity with isotropic hardening and the back stresses alpha_k of Armstrong and
  /// Frederick (Chaboche): with a_k = 1 / (1 + gamma_k dp), the relative stress at the end of
  /// the step is parallel to xi~ = s_trial - sum a_k alpha_k^n, and the increment dp of the
  /// equivalent plastic strain solves q(xi~) - (3G + sum a_k C_k) dp - sigma_y(p_n + dp) = 0.
  void plasticReturn(const ADReal & mu,
                     const ADReal & T,
                     double p_old,
                     const std::vector<ADReal> & back_old,
                     std::array<ADReal, 6> & sigma,
                     InelasticIncrements & inc) const;
  /// Creep and isotropic plasticity in one return: both flow along the deviator of the trial
  /// stress, whose von Mises value falls to q - 3G (dc + dp), with dc = dt rate(q_e, T, c + dc)
  /// and, where the point yields, q_e = sigma_y(p + dp).
  void creepPlasticReturn(QpContext & ctx,
                          const ADReal & mu,
                          const ADReal & T,
                          double c_old,
                          double p_old,
                          std::array<ADReal, 6> & sigma,
                          InelasticIncrements & inc) const;
  /// Creep rate at the von Mises stress sigma (Pa), the temperature T and the equivalent creep
  /// strain p, at the time of the context (the end of the step), times creep_rate_factor.
  template <typename T>
  T creepRate(const T & sigma, const T & temperature, const T & p, const QpContext & ctx) const;
  /// The backward Euler creep update by a radial return: from the trial stress @p sigma, the
  /// creep strain increment dp (3/2) s / q along the deviator s, with dp = dt rate(q - 3 G dp, T,
  /// p_old + dp).  Updates @p sigma and fills the increments.
  void returnMap(QpContext & ctx,
                 const ADReal & mu,
                 const ADReal & T,
                 double p_old,
                 std::array<ADReal, 6> & sigma,
                 std::array<double, 6> & creep_increment,
                 double & equivalent_increment) const;
  /// Plane stress with creep or plasticity: the elastic zz strain z at which the stress after
  /// the return mapping has sigma_zz = 0.  The return mapping works on the full stress, so z is
  /// solved at the point by Newton's method, with dsigma_zz/dz from a pass whose only derivative
  /// is z.  The derivatives of z with respect to the unknowns follow from the implicit function
  /// theorem, so the tangent stays exact.  On return @p ee holds z and @p sigma the trial stress
  /// of the full elastic strain, which updateInelastic then returns.
  void planeStressInelasticStrain(QpContext & ctx,
                                  const ADReal & lambda,
                                  const ADReal & mu,
                                  const double * old,
                                  ADReal * ee,
                                  ADReal * sigma) const;
  /// The inelastic strains of one step from the trial stress @p sigma: the stress and the
  /// elastic strain @p ee lose their increments, and the history moves from @p old to @p next.
  void updateInelastic(QpContext & ctx,
                       const ADReal & mu,
                       const double * old,
                       double * next,
                       ADReal * ee,
                       ADReal * sigma) const;
};

/// Register finite_strain_stress (SolidMechanicsFiniteStrain.cpp).
void registerFiniteStrainObjects(Factory & f);
/// Register mortar_contact, cohesive_zone and normal_displacement (SolidMechanicsInterfaces.cpp).
void registerInterfaceObjects(Factory & f);

} // namespace dualmesh::solid_mechanics

namespace dualmesh
{

/// A structural member with a section (a plate or a beam), whose stress varies over the height of
/// the section (SolidMechanicsStructures.cpp).
class SectionKernel : public Kernel
{
public:
  using Kernel::Kernel;
  /// The stress at the height @p z above the mid-surface, from the material law of the section.
  virtual std::vector<double> sectionStress(const QpContext & context, double z) const = 0;
};

/// Register the beams, plates, trusses and shells (SolidMechanicsStructures.cpp).
void registerStructuralMemberObjects(Factory & f);

/// The solid system of the module (SolidSystem.h), which registerSolidMechanicsObjects registers
/// (SolidMechanicsSystem.cpp).
std::shared_ptr<const SolidSystem> makeSolidMechanicsSystem();
/// The stress at the height @p z above the mid-surface, at the element centroids, of the
/// structural kernel @p kernel of @p problem (SectionKernel::sectionStress), one row per element
/// (SolidMechanicsStructures.cpp).
std::vector<std::vector<double>>
sectionStressAtCentroids(Problem & problem, const std::string & kernel, double z);
/// The J-integral and the stress intensity factors K_I and K_II of a crack tip of @p problem in
/// two dimensions, by the domain integral and the interaction integral over the ring between the
/// two radii (SolidMechanicsFracture.cpp).  The crack lies along @p angle (radians from x), its
/// faces behind the tip.
std::array<double, 3> crackTipIntegrals(Problem & problem,
                                        const std::vector<std::string> & displacements,
                                        const Point & tip,
                                        double angle,
                                        double inner_radius,
                                        double outer_radius,
                                        double youngs_modulus,
                                        double poissons_ratio,
                                        bool plane_strain);

} // namespace dualmesh
