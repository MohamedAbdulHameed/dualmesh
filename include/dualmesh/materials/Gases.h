// SPDX-License-Identifier: LGPL-2.1-or-later
//
// Thermal conductivity of helium, argon, krypton, xenon, hydrogen and
// nitrogen at low density, and of their mixtures.
//
// Units are SI, with temperatures in kelvin.  The functions are templates so
// that they can be evaluated both on plain numbers and on automatic
// differentiation numbers (ADReal).
#pragma once

#include "dualmesh/core/ADReal.h"
#include "dualmesh/core/Types.h"

#include <cmath>
#include <string>

namespace dualmesh
{
namespace materials
{

using std::exp;
using std::log;
using std::pow;
using std::sqrt;

enum class Gas
{
  Helium = 0,
  Argon,
  Krypton,
  Xenon,
  Hydrogen,
  Nitrogen,
  Count
};
constexpr int kNumGases = static_cast<int>(Gas::Count);
/// Molar masses, kg/mol.
constexpr double kGasMolarMass[kNumGases] = {
    4.0026e-3, 39.948e-3, 83.798e-3, 131.293e-3, 2.016e-3, 28.014e-3};
/// k = A T^B, W/(m K): the MATPRO GASCON fits, NUREG/CR-6150 Vol. 4 (1995)
/// Table 13-2, used for hydrogen and nitrogen.
constexpr double kGasA[kNumGases] = {2.639e-3, 2.986e-4, 8.247e-5, 4.351e-5, 1.097e-3, 5.314e-4};
constexpr double kGasB[kNumGases] = {0.7085, 0.7224, 0.8363, 0.8616, 0.8785, 0.6898};
/// ln k = c0 + c1 ln T + c2 (ln T)^2 + c3 (ln T)^3, k in W/(m K): fits to the
/// low-density conductivities of Tables 1, 3, 4 and 5 of J. Kestin, K.
/// Knierim, E. A. Mason, B. Najafi, S. T. Ro and M. Waldman, J. Phys. Chem.
/// Ref. Data 13 (1984) 229-303, over 200-2273 K (maximum deviation 0.005 %
/// He, 0.052 % Ar, 0.046 % Kr and 1.05 % Xe).  The fit is reproduced by
/// verification/correlations/kestin1984_gas_fit.py.
constexpr double kGasKestin[4][4] = {
    {-5.2532347534e+00, 5.1483336706e-01, 1.4181490440e-02, 9.3426153211e-06},
    {-1.6548530604e+01, 4.2657846105e+00, -4.8906679795e-01, 2.2102852922e-02},
    {-1.8704655012e+01, 4.7549670873e+00, -5.3442160336e-01, 2.3257258353e-02},
    {-1.6433076351e+01, 3.2558216504e+00, -2.7966068851e-01, 9.4971467825e-03}};

/// The gas of a name: helium, argon, krypton, xenon, hydrogen or nitrogen.
inline Gas
gasFromName(const std::string & name)
{
  static const char * names[kNumGases] = {
      "helium", "argon", "krypton", "xenon", "hydrogen", "nitrogen"};
  for (int i = 0; i < kNumGases; ++i)
    if (name == names[i])
      return static_cast<Gas>(i);
  throw InputError("Unknown gas '" + name +
                   "' (use helium, argon, krypton, xenon, hydrogen or nitrogen).");
}

/// Thermal conductivity of one gas, W/(m K).
template <typename T>
T
gasConductivity(Gas g, const T & temperature)
{
  const int i = static_cast<int>(g);
  if (i < 4)
  {
    const T x = log(temperature);
    const double * c = kGasKestin[i];
    return exp(c[0] + x * (c[1] + x * (c[2] + x * c[3])));
  }
  return kGasA[i] * pow(temperature, kGasB[i]);
}

/// Conductivity of a gas mixture with mole fractions x, W/(m K): Eqs. (12)
/// and (13) of R. S. Brokaw, J. Chem. Phys. 29 (1958) 391-397, with the
/// collision integral ratios A* = B* = 1.1, which give the factors 2.41 and
/// 0.142 of MATPRO GTHCON.
template <typename T>
T
gasMixtureConductivity(const double * x, const T & temperature)
{
  T k[kNumGases];
  for (int i = 0; i < kNumGases; ++i)
    k[i] = gasConductivity(static_cast<Gas>(i), temperature);
  T total(0.0);
  for (int i = 0; i < kNumGases; ++i)
  {
    if (x[i] <= 0)
      continue;
    T denominator(x[i]);
    for (int j = 0; j < kNumGases; ++j)
    {
      if (j == i || x[j] <= 0)
        continue;
      const double Mi = kGasMolarMass[i], Mj = kGasMolarMass[j];
      const T r = sqrt(k[i] / k[j]) * std::pow(Mi / Mj, 0.25) + 1.0;
      const T phi = r * r / (std::pow(2.0, 1.5) * std::sqrt(1.0 + Mi / Mj));
      const T psi = phi * (1.0 + 2.41 * (Mi - Mj) * (Mi - 0.142 * Mj) / ((Mi + Mj) * (Mi + Mj)));
      denominator = denominator + psi * x[j];
    }
    total = total + k[i] * x[i] / denominator;
  }
  return total;
}

inline double
valueOf(double x)
{
  return x;
}
inline double
valueOf(const ADReal & x)
{
  return x.value();
}

/// Sum of the temperature jump distances at the two surfaces of a gap, m:
/// the equation of E. H. Kennard for a mixture of monatomic gases as D. D.
/// Lanning and C. R. Hann (BNWL-1894, 1975), Appendix B, Sect. 2, write it,
///   g = 2878 (2 - a)/a k sqrt(T) / P (sum x_i / M_i)^(-1/2)
/// per surface, with k in cal/(s cm K), P in dyn/cm^2 and g in cm.  In SI
/// units (k in W/(m K), P in Pa, g in m, M in g/mol) the constant of the sum
/// g1 + g2 is 2 x 2878 / 41868 = 0.013748.  For the diatomic H2 and N2 the
/// monatomic heat capacities overstate the jump distance by half.  A positive
/// @p accommodation is the accommodation coefficient a.  Otherwise a is
/// interpolated between helium (0.425 - 2.3e-4 T) and xenon (0.749 - 2.5e-4
/// T), the fits of Ullman et al., by the mixture molar mass sum x_i M_i
/// (Lanning and Hann, Appendix B, Sect. 1).  The helium fit turns negative at
/// 1848 K, so both fits are bounded below by 0.07, the MATPRO estimate for
/// helium (NUREG/CR-6150 Vol. 4 Table 13-4).
template <typename T>
T
gasJumpDistance(const double * x,
                const T & temperature,
                double pressure,
                const T & k_mix,
                double accommodation = 0.0)
{
  double mass = 0.0, inverse = 0.0;
  for (int i = 0; i < kNumGases; ++i)
  {
    mass += x[i] * kGasMolarMass[i] * 1000.0;
    if (x[i] > 0)
      inverse += x[i] / (kGasMolarMass[i] * 1000.0);
  }
  T a = temperature * 0.0 + accommodation;
  if (!(accommodation > 0.0))
  {
    const double mhe = kGasMolarMass[0] * 1000.0, mxe = kGasMolarMass[3] * 1000.0;
    T ahe = 0.425 - 2.3e-4 * temperature;
    T axe = 0.749 - 2.5e-4 * temperature;
    if (valueOf(ahe) < 0.07)
      ahe = temperature * 0.0 + 0.07;
    if (valueOf(axe) < 0.07)
      axe = temperature * 0.0 + 0.07;
    a = ahe + (axe - ahe) * ((mass - mhe) / (mxe - mhe));
  }
  return 0.013748 * (2.0 - a) / a * k_mix * sqrt(temperature) / pressure / std::sqrt(inverse);
}

} // namespace materials
} // namespace dualmesh
