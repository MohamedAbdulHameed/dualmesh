// SPDX-License-Identifier: LGPL-2.1-or-later
//
// Material correlations for nuclear fuel performance: uranium dioxide (UO2),
// uranium mononitride (UN), Zircaloy-4 cladding and the fill and fission
// gases of the fuel-cladding gap.
//
// The source of every correlation is named next to it, and the documentation
// chapter on the fuel correlations lists every correlation with its source.
//
// Units are SI throughout, with temperatures in kelvin, unless a comment says
// otherwise.  The functions are templates so that they can be evaluated both
// on plain numbers and on automatic differentiation numbers (ADReal), which
// is how the property objects obtain the exact Jacobian of a temperature-dependent
// property.
#pragma once

#include "dualmesh/core/ADReal.h"

#include <algorithm>
#include <cmath>
#include <string>

namespace dualmesh
{
namespace fuel
{

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

/// The lowest temperature (K) at which the property objects evaluate a correlation.
/// The correlations are fits to data above room temperature, and several are
/// singular at absolute zero (Fink's T^(-5/2) exp(-16.35/t), the square root
/// in Lucuta's factor, the gas conductivity A T^B with B < 1).  Newton's
/// method can visit such temperatures when it starts from a cold guess (a
/// temperature left at its default initial value of zero), so the property objects
/// evaluate the correlations at max(T, 200 K), with zero derivative below.
/// Converged reactor temperatures are far above this floor.
constexpr double kMinimumCorrelationTemperature = 200.0;

template <typename T>
T
correlationTemperature(const T & temperature)
{
  return valueOf(temperature) < kMinimumCorrelationTemperature ? T(kMinimumCorrelationTemperature)
                                                               : temperature;
}

using std::atan;
using std::exp;
using std::log;
using std::pow;
using std::sqrt;
using std::tanh;

/// Hyperbolic sine for plain and automatic differentiation numbers.
template <typename T>
T
sinh_(const T & x)
{
  return 0.5 * (exp(x) - exp(-x));
}

constexpr double kBoltzmann = 1.380649e-23;         ///< J/K
constexpr double kAvogadro = 6.02214076e23;         ///< 1/mol
constexpr double kGasConstant = 8.314462618;        ///< J/(mol K)
constexpr double kStefanBoltzmann = 5.670374419e-8; ///< W/(m^2 K^4)
constexpr double kJoulePerMeV = 1.602176634e-13;
/// Molar masses (kg/mol) of the heavy-metal compounds, natural isotopics.
constexpr double kMolarMassUO2 = 0.2700277; ///< 238.02891 + 2 x 15.9994 g/mol
constexpr double kMolarMassUN = 0.25204;    ///< 238.03 + 14.007 g/mol
/// Theoretical densities (kg/m^3).
constexpr double kDensityUO2 = 10963.0; ///< at 273 K, Fink (2000)
constexpr double kDensityUN = 14326.0;  ///< at 298 K, Hayes et al. (1990, part I), Eq. (3)

// =============================================================================
// Uranium dioxide
// =============================================================================

/// Thermal conductivity of unirradiated UO2 at 95 % of theoretical density,
/// W/(m K), 298-3120 K: J. K. Fink, J. Nucl. Mater. 279 (2000) 1-18.  The
/// first term is the lattice (phonon) conductivity, the second the ambipolar
/// (small polaron) contribution.
template <typename T>
T
uo2ConductivityFink95(const T & temperature)
{
  const T t = temperature / 1000.0;
  return 100.0 / (7.5408 + 17.692 * t + 3.6142 * t * t) + 6400.0 / pow(t, 2.5) * exp(-16.35 / t);
}

/// Fink's conductivity converted to fully dense UO2 with the porosity
/// correction he recommends (Brandt and Neuer 1976), k_100 = k_95 / (1 -
/// (2.6 - 0.5 t) 0.05).
template <typename T>
T
uo2ConductivityFink100(const T & temperature)
{
  const T t = temperature / 1000.0;
  return uo2ConductivityFink95(temperature) / (1.0 - (2.6 - 0.5 * t) * 0.05);
}

/// The four multiplicative factors of P. G. Lucuta, Hj. Matzke and I. J.
/// Hastings, J. Nucl. Mater. 232 (1996) 166-180, for dissolved fission
/// products (kappa_1d), precipitated fission products (kappa_1p), porosity
/// with non-conducting spherical pores (kappa_2p, Maxwell-Eucken) and
/// radiation damage (kappa_4r), Eqs. (14b), (14c), (14d) and (14f).  beta is
/// the burnup in atom per cent (FIMA times 100) and p the porosity fraction.
/// The porosity factor (1 - p)/(1 + (s - 1) p) with the shape factor s = 1.5
/// of spherical pores is (1 - p)/(1 + 0.5 p).  It is used only for the
/// departure from 95 % density (see uo2ConductivityFinkLucuta).
template <typename T>
T
lucutaDissolved(const T & temperature, double beta)
{
  if (beta <= 1e-9)
    return T(1.0);
  const T w = 1.09 / std::pow(beta, 3.265) + 0.0643 * sqrt(temperature / beta);
  return w * atan(1.0 / w);
}
template <typename T>
T
lucutaPrecipitated(const T & temperature, double beta)
{
  const double c = 0.019 * beta / (3.0 - 0.019 * beta);
  return 1.0 + c / (1.0 + exp(-(temperature - 1200.0) / 100.0));
}
inline double
lucutaPorosity(double porosity)
{
  return (1.0 - porosity) / (1.0 + 0.5 * porosity);
}
template <typename T>
T
lucutaRadiation(const T & temperature)
{
  return 1.0 - 0.2 / (1.0 + exp((temperature - 900.0) / 80.0));
}

/// Irradiated UO2 conductivity, W/(m K): Fink's recommended conductivity of
/// 95 % dense fuel, corrected to the actual porosity with the ratio of
/// Maxwell-Eucken factors kappa_2p(p) / kappa_2p(0.05), times Lucuta's burnup
/// and radiation factors.  At zero burnup and 95 % density it returns Fink's
/// value exactly.  The radiation-damage factor applies to irradiated fuel
/// (beta > 0) only.
template <typename T>
T
uo2ConductivityFinkLucuta(const T & temperature, double beta, double porosity)
{
  T k = uo2ConductivityFink95(temperature) * lucutaDissolved(temperature, beta) *
        lucutaPrecipitated(temperature, beta) * (lucutaPorosity(porosity) / lucutaPorosity(0.05));
  if (beta > 0)
    k = k * lucutaRadiation(temperature);
  return k;
}

/// The modified NFI conductivity of UO2 (and UO2-Gd2O3), W/(m K), as given
/// in the FRAPCON-4.0 code description (Geelhood et al., PNNL-19418 Vol. 1
/// Rev. 2, Eqs. 2.52 to 2.56) after
/// K. Ohira and N. Itagaki (1997) with the PNNL modifications (Lanning et al.,
/// NUREG/CR-6534 Vol. 4, 2005).  burnup is in MWd/kgU (GWd/tU), gadolinia
/// the weight fraction of Gd2O3, density_fraction the fraction of theoretical
/// density; the result includes Lucuta's spherical-pore density correction
/// from 95 %.  Valid 300-3000 K, 0-62 GWd/tU, 92-97 % TD.
template <typename T>
T
uo2ConductivityNFI(const T & temperature, double burnup, double gadolinia, double density_fraction)
{
  const double f = 0.00187 * burnup;
  const double g = burnup > 0 ? 0.038 * std::pow(burnup, 0.28) : 0.0;
  const T h = 1.0 / (1.0 + 396.0 * exp(-6380.0 / temperature));
  const T k95 = 1.0 / (0.0452 + 1.1599 * gadolinia + 2.46e-4 * temperature + f +
                       (1.0 - 0.9 * std::exp(-0.04 * burnup)) * g * h) +
                3.5e9 / (temperature * temperature) * exp(-16361.0 / temperature);
  const double d = density_fraction;
  return 1.0789 * k95 * d / (1.0 + 0.5 * (1.0 - d));
}

/// The Halden conductivity of irradiated UO2, W/(m K), of W. Wiesenack,
/// "Assessment of UO2 conductivity degradation based on in-pile temperature
/// data" (Sect. V), recommended for irradiated fuel by IAEA-TECDOC-1496
/// (2006), Sect. 6.1.2:
///   k95 = 1 / (0.1148 + 0.0035 B + 2.475e-4 (1 - 0.00333 B) min(T_C, 1650))
///         + 0.0132 exp(0.00188 T_C),
/// for 95 % dense fuel, with T_C in degrees Celsius and the burnup B in
/// MWd/kgUO2, valid to 75 MWd/kgUO2 (uncertainty within 20 % up to 2000 K).
/// The temperature of the phonon term is limited to 1650 C, as Wiesenack
/// states (Sect. III.A) and CASL-U-2019-1870 Eq. (1) writes.  It is corrected
/// to the actual density with the ratio of Maxwell-Eucken factors,
/// 1.0789 d / (1 + 0.5 (1 - d)), as CASL-U-2019-1870 Eq. (2).
template <typename T>
T
uo2ConductivityHalden(const T & temperature, double burnup_uo2, double density_fraction)
{
  const T tc = temperature - 273.15;
  const T phonon_tc = valueOf(tc) > 1650.0 ? tc * 0.0 + 1650.0 : tc;
  const double B = std::max(0.0, burnup_uo2);
  const T k95 = 1.0 / (0.1148 + 0.0035 * B + 2.475e-4 * (1.0 - 0.00333 * B) * phonon_tc) +
                0.0132 * exp(0.00188 * tc);
  const double d = density_fraction;
  return 1.0789 * k95 * d / (1.0 + 0.5 * (1.0 - d));
}

/// Relocation of UO2 fragments as a transverse strain (the diametral strain
/// dD/D), the ESCORE model of FALCON MOD01 Vol. 1 (EPRI 1011307, 2004),
/// Eqs. 5-30 and 5-31: dD/D = 0.80 Q (G0/D0)(0.005 Bu^0.3 - 0.20 D0 + 0.3), with the
/// linear heat rate q' in kW/ft (Q = 0 below 6, (q' - 6)^(1/3) up to 14 and
/// (q' - 10)/2 above), the cold diametral gap G0 and pellet diameter D0 (the
/// bracket takes D0 in inches) and the burnup in MWd/tU, held at
/// burnup_limit (MWd/kgHM) above it.  linear_heat_rate in W/m, the lengths
/// in m, burnup in MWd/kgHM.  Never negative.
inline double
uo2RelocationStrain(double linear_heat_rate,
                    double burnup,
                    double pellet_diameter,
                    double diametral_gap,
                    double burnup_limit)
{
  const double kw_ft = linear_heat_rate * 0.3048 / 1000.0;
  double Q = 0.0;
  if (kw_ft > 14.0)
    Q = (kw_ft - 10.0) / 2.0;
  else if (kw_ft > 6.0)
    Q = std::cbrt(kw_ft - 6.0);
  const double mwd_per_t = 1000.0 * std::min(burnup_limit, std::max(0.0, burnup));
  const double d0 = pellet_diameter / 0.0254;
  const double strain = 0.80 * Q * (diametral_gap / pellet_diameter) *
                        (0.005 * std::pow(mwd_per_t, 0.3) - 0.20 * d0 + 0.3);
  return std::max(0.0, strain);
}

/// Specific heat of UO2, J/(kg K), 298-3120 K: Fink (2000), Einstein term,
/// linear term and Frenkel-defect term, divided by the molar mass of UO2.
template <typename T>
T
uo2SpecificHeatFink(const T & temperature)
{
  const double C1 = 81.613, theta = 548.68, C2 = 2.285e-3, C3 = 2.360e7, Ea = 18531.7;
  const T e = exp(theta / temperature);
  const T cp = C1 * theta * theta * e / (temperature * temperature * (e - 1.0) * (e - 1.0)) +
               2.0 * C2 * temperature +
               C3 * Ea * exp(-Ea / temperature) / (temperature * temperature);
  return cp / kMolarMassUO2;
}

/// Specific heat of UO2, J/(kg K), in the MATPRO form with the constants of
/// J. F. Kerrisk and D. G. Clifton, Nucl. Technol. 16 (1972) 531-535, as
/// listed in the FRAPCON-4.0 material property report (PNNL-19417); oxygen to
/// metal ratio 2.  It differs from Fink's by less than 1 % below 2000 K.
template <typename T>
T
uo2SpecificHeatMATPRO(const T & temperature, double oxygen_to_metal = 2.0)
{
  const double K1 = 296.7, K2 = 2.43e-2, K3 = 8.745e7, theta = 535.285, ED = 1.577e5, R = 8.3143;
  const T e = exp(theta / temperature);
  return K1 * theta * theta * e / (temperature * temperature * (e - 1.0) * (e - 1.0)) +
         K2 * temperature +
         oxygen_to_metal * K3 * ED / (2.0 * R * temperature * temperature) *
             exp(-ED / (R * temperature));
}

/// Linear dimension of UO2 relative to its value at 273 K, L(T)/L(273):
/// D. G. Martin, J. Nucl. Mater. 152 (1988) 94-101, Eqs. (1a) and (1b).
template <typename T>
T
uo2RelativeLengthMartin(const T & temperature)
{
  const T & t = temperature;
  if (valueOf(t) <= 923.0)
    return 0.99734 + 9.802e-6 * t - 2.705e-10 * t * t + 4.391e-13 * t * t * t;
  return 0.99672 + 1.179e-5 * t - 2.429e-9 * t * t + 1.219e-12 * t * t * t;
}

/// Linear thermal strain of UO2 between a reference temperature and T.
template <typename T>
T
uo2ThermalStrainMartin(const T & temperature, double reference)
{
  return uo2RelativeLengthMartin(temperature) / uo2RelativeLengthMartin(reference) - 1.0;
}

/// Young's modulus of UO2, Pa, in the MATPRO FELMOD form E = 2.334e11 (1 -
/// 2.752 (1 - D)) (1 - 1.0915e-4 T), D the fraction of theoretical density,
/// with Poisson's ratio 0.316 (MATPRO FPOIR), T in K.  Checked against the
/// public INL/LANL report CASL-U-2019-1870-000 (Gamble, Pastore, Cooper and
/// Andersson, 2019), which quotes FELMOD as E = 2.334e11 (1 - 1.0915e-4 T)
/// (1 - 2.752 P) with P the porosity; the MATPRO O/M correction is omitted
/// (stoichiometric fuel).
template <typename T>
T
uo2YoungsModulus(const T & temperature, double density_fraction)
{
  return 2.334e11 * (1.0 - 2.752 * (1.0 - density_fraction)) * (1.0 - 1.0915e-4 * temperature);
}
constexpr double kPoissonRatioUO2 = 0.316; ///< MATPRO, independent of temperature

/// Densification of UO2 as a (negative) volumetric strain, the ESCORE model
/// of FALCON MOD01 (EPRI 1011307, 2004), Eqs. 5-22 and 5-24:
///   eps = dRho0 [exp(Bu ln(0.01) / (C Bu_D)) - 1],
///   C = 7.235 - 0.0086 (T_C - 25) below 750 C and 1 above,
/// with dRho0 the total densification (fraction of theoretical density), Bu_D
/// the burnup at which it is complete, both inputs, and Bu the pellet-average
/// burnup.  The intercept 7.235 = 1 + 0.0086 (750 - 25) makes C, and so the
/// densification, continuous at 750 C (FALCON prints it rounded to 7.2).
/// burnup and complete_burnup in the same unit.
template <typename T>
T
uo2DensificationESCORE(const T & temperature, double burnup, double total, double complete_burnup)
{
  if (total <= 0 || complete_burnup <= 0)
    return T(0.0);
  const T tc = temperature - 273.15;
  const T c = valueOf(tc) < 750.0 ? 7.235 - 0.0086 * (tc - 25.0) : T(1.0);
  return total * (exp(burnup * std::log(0.01) / (c * complete_burnup)) - 1.0);
}

/// Densification of UO2 as a (negative) linear strain, fraction: MATPRO FUDENS,
/// NUREG/CR-6150 Vol. 4 (1995) Eqs. (2-81), (2-82) and (2-85),
///   dL/L = (dL/L)_m + exp(-3 (Bu + B)) + 2 exp(-35 (Bu + B))   (per cent),
///   (dL/L)_m = -0.0015 RSNTR (T < 1000 K), -0.00285 RSNTR (T >= 1000 K),
/// with the resintering density change RSNTR (kg/m^3), the burnup Bu in
/// MWd/kgU and B chosen so that dL/L = 0 at Bu = 0.  The temperature is the
/// highest the fuel has reached, so that densification never reverses.  The
/// switch of (dL/L)_m at 1000 K is spread linearly over 950-1050 K.
inline double
uo2DensificationMATPRO(double maximum_temperature, double burnup, double resintering_change)
{
  if (resintering_change <= 0)
    return 0.0;
  // MATPRO switches the coefficient from 0.0015 to 0.00285 at 1000 K.  The
  // step is spread linearly over 950-1050 K, so that the strain of an element
  // near 1000 K does not jump with a small change of its temperature.
  const double w = std::clamp((maximum_temperature - 950.0) / 100.0, 0.0, 1.0);
  const double m = -((1.0 - w) * 0.0015 + w * 0.00285) * resintering_change; // per cent
  // exp(-3 B) + 2 exp(-35 B) = -m, solved by Newton's method from B = 0; the
  // left side falls monotonically from 3, so a root exists for -m < 3.
  const double target = std::min(-m, 2.999999);
  double B = 0.0;
  for (int it = 0; it < 100; ++it)
  {
    const double f = std::exp(-3.0 * B) + 2.0 * std::exp(-35.0 * B) - target;
    const double df = -3.0 * std::exp(-3.0 * B) - 70.0 * std::exp(-35.0 * B);
    const double step = f / df;
    B -= step;
    if (std::abs(step) < 1e-14)
      break;
  }
  const double x = std::max(0.0, burnup) + B;
  return 0.01 * (-target + std::exp(-3.0 * x) + 2.0 * std::exp(-35.0 * x));
}

/// Solid fission-product swelling of UO2 (volumetric strain per unit burnup
/// in FIMA): MATPRO's 2.5e-29 per fission per m^3, written per FIMA for a
/// fuel of density rho (kg/m^3): 5.577e-5 rho.
inline double
uo2SolidSwellingRate(double density)
{
  return 2.5e-29 * kAvogadro / kMolarMassUO2 * density;
}

/// Gaseous swelling of UO2, MATPRO FSWELL: the increment of volumetric strain
/// for a burnup increment dbu (FIMA) at temperature T and total burnup bu
/// (FIMA), for a fuel of density rho.  Zero above 2800 K.
template <typename T>
T
uo2GaseousSwellingIncrement(const T & temperature, double density, double bu, double dbu)
{
  if (valueOf(temperature) >= 2800.0)
    return T(0.0);
  const double per_fima = density * kAvogadro / kMolarMassUO2; // fissions per m^3 per FIMA
  const T d = 2800.0 - temperature;
  return 8.8e-56 * pow(d, 11.73) * exp(-0.0162 * d) * std::exp(-8.0e-27 * bu * per_fima) *
         (dbu * per_fima);
}

/// Creep rate of UO2, 1/s: MATPRO FCREEP, NUREG/CR-6150 Vol. 4 (1995)
/// Eqs. (2-60)-(2-63) and (2-69), read in the report: diffusional,
/// dislocation and fission-induced creep,
///   rate = (A1 + A2 F) s1 exp(-Q1/RT) / ((A3 + D) G^2)
///        + A4 sigma^4.5 exp(-Q2/RT) / (A6 + D) + A7 F sigma exp(-2616.8/T),
/// with s1 = min(sigma, sigma_t), the transition stress sigma_t =
/// 1.6547e7 / G^0.5714 Pa (Eq. 2-60, "for stresses greater than sigma_t the
/// transition stress is used in the first term").  Q1 = 17884.8 f + 72124.23
/// and Q2 = 19872 f + 111543.5 cal/mol with f = 1/(exp(-20/ln(x - 2) - 8) + 1)
/// (x the O/M ratio; f = 1/(exp(-8) + 1) at x = 2).  The last term follows
/// the derivation of Eq. (2-69) (Brucklacher); the constant list of Eq. (2-63)
/// prints Q3 = 2.6167e3 J/mol with a positive exponent, a misprint.
/// stress: von Mises stress (Pa); fission_rate: fissions/(m^3 s); density:
/// per cent of theoretical density; grain_size: micrometres.  The primary
/// (time-dependent) factor of Eq. (2-65) is not included.
template <typename T>
T
uo2CreepRate(const T & stress,
             const T & temperature,
             double fission_rate,
             double density_percent,
             double grain_size,
             double x = 2.0)
{
  const double A1 = 0.3919, A2 = 1.31e-19, A3 = -87.7, A4 = 2.0391e-25, A6 = -90.5,
               A7 = 3.72264e-35, R = kGasConstant, calorie = 4.184;
  const double fx = x <= 2.0 ? 1.0 / (std::exp(-8.0) + 1.0)
                             : 1.0 / (std::exp(-20.0 / std::log(x - 2.0) - 8.0) + 1.0);
  const double Q1 = (17884.8 * fx + 72124.23) * calorie;
  const double Q2 = (19872.0 * fx + 111543.5) * calorie;
  const double transition = 1.6547e7 / std::pow(grain_size, 0.5714);
  // Above the transition the first term takes the constant sigma_t; written as
  // 0 * stress + sigma_t so that it keeps the derivative layout of stress.
  const T s1 = valueOf(stress) > transition ? stress * 0.0 + transition : stress;
  const T term1 = (A1 + A2 * fission_rate) / ((A3 + density_percent) * grain_size * grain_size) *
                  s1 * exp(-Q1 / (R * temperature));
  const T term2 = A4 / (A6 + density_percent) * pow(stress, 4.5) * exp(-Q2 / (R * temperature));
  const T term3 = A7 * fission_rate * stress * exp(-2616.8 / temperature);
  return term1 + term2 + term3;
}

// =============================================================================
// Uranium mononitride
// =============================================================================
// The Hayes, Thomas and Peddicord series, J. Nucl. Mater. 171 (1990): part I
// physical properties 262-270, part II mechanical 271-288, part III transport
// 289-299, part IV thermodynamic 300-318.

/// Thermal conductivity of UN, W/(m K), 298-1923 K, porosity P up to 0.2:
/// Hayes et al. (1990, part III), k = 1.864 exp(-2.14 P) T^0.361.
template <typename T>
T
unConductivityHayes(const T & temperature, double porosity)
{
  return 1.864 * std::exp(-2.14 * porosity) * pow(temperature, 0.361);
}

/// Specific heat of UN, J/(kg K), 298-2628 K: Hayes et al. (1990, part IV),
/// with the Einstein temperature 365.7 K.
template <typename T>
T
unSpecificHeatHayes(const T & temperature)
{
  const double theta = 365.7;
  const T e = exp(theta / temperature);
  const T cp = 51.14 * (theta / temperature) * (theta / temperature) * e / ((e - 1.0) * (e - 1.0)) +
               9.491e-3 * temperature +
               2.642e11 / (temperature * temperature) * exp(-18081.0 / temperature);
  return cp / kMolarMassUN;
}

/// Theoretical density of UN, kg/m^3, 298-2523 K: Hayes et al. (1990, part I),
/// Eq. (3) in the body of the paper, the fit of the density of the lattice
/// parameter of Eq. (1).  The abstract prints 2.779e-4 for the linear
/// coefficient, which departs from that density by 0.056 g/cm^3.
template <typename T>
T
unDensityHayes(const T & temperature)
{
  return 1000.0 * (14.42 - 2.997e-4 * temperature - 4.897e-8 * temperature * temperature);
}

/// Linear thermal strain of UN between a reference temperature and T, from
/// the mean expansion coefficient of Hayes et al. (1990, part I), measured
/// from 298 K: eps(T) = (7.096e-6 + 1.409e-9 T)(T - 298).
template <typename T>
T
unThermalStrainHayes(const T & temperature, double reference)
{
  const auto strain = [](const auto & t) { return (7.096e-6 + 1.409e-9 * t) * (t - 298.0); };
  return strain(temperature) - strain(reference);
}

/// Young's modulus (Pa) and Poisson's ratio of UN, Hayes et al. (1990, part
/// II), for 70-100 % TD and 298-1473 K; D is in per cent of theoretical
/// density.
template <typename T>
T
unYoungsModulus(const T & temperature, double density_percent)
{
  return 1e6 * 0.258 * std::pow(density_percent, 3.002) * (1.0 - 2.375e-5 * temperature);
}
inline double
unPoissonRatio(double density_percent)
{
  return 1.26e-3 * std::pow(density_percent, 1.174);
}

/// Creep rate of UN, 1/s, the sum of three terms, with sigma in MPa:
///  * dislocation creep of dense UN, 2.054e-3 sigma^4.5 exp(-39369.5/T)
///    (Hayes, Thomas and Peddicord, J. Nucl. Mater. 171 (1990) 271-288);
///  * grain-boundary (Coble) creep, 582610.427 sigma / (T d^3)
///    exp(-2.28 eV / k T) with the grain size d in micrometres (M.
///    AbdulHameed, B. Beeler, C. O. T. Galvin, M. W. D. Cooper, N. Elamrawy
///    and A. Claisse, J. Nucl. Mater. 617 (2025) 156153, Eqs. (14) and
///    (15));
///  * irradiation creep, 2.9e-22 sigma G exp(0.2 P) per hour with the fission
///    rate G in fissions/(cm^3 s) and the porosity P in per cent (I. I.
///    Konovalov, B. A. Tarasov and E. M. Glagovsky, IOP Conf. Ser. Mater. Sci.
///    Eng. 130 (2016) 012030, Eq. (13), the middle of its range 2.5-3.3e-22).
/// grain_size is in micrometres and porosity a fraction.  With
/// hayes_porosity_factor the dislocation term is multiplied by 0.987 exp(-8.65
/// P) / (1 - P)^27.6, a factor attributed to Hayes et al.  It is off by
/// default.
template <typename T>
T
unCreepRate(const T & stress,
            const T & temperature,
            double fission_rate,
            double porosity,
            double grain_size,
            bool coble = true,
            bool hayes_porosity_factor = false)
{
  const T s = stress / 1e6;
  const double fp = hayes_porosity_factor
                        ? 0.987 * std::exp(-8.65 * porosity) / std::pow(1.0 - porosity, 27.6)
                        : 1.0;
  T rate = 2.054e-3 * pow(s, 4.5) * exp(-39369.5 / temperature) * fp;
  if (coble && grain_size > 0)
  {
    const double kB = 8.617333262e-5; // eV/K
    rate += 582610.427 * s / (temperature * grain_size * grain_size * grain_size) *
            exp(-2.28 / (kB * temperature));
  }
  rate += 2.9e-22 * s * (fission_rate * 1e-6) * std::exp(0.2 * 100.0 * porosity) / 3600.0;
  return rate;
}

/// The smallest measured swelling rate of nitride fuel, 1 per cent of volume
/// per atom per cent of burnup: "high swelling rates have been measured for
/// FRs nitride fuel from 1 %/at% to several 10 %/at%" (OECD/NEA, State-of-the-
/// Art Report on Light Water Reactor Accident-Tolerant Fuels, NEA No. 7317,
/// 2018, Sect. 17).  It is used as a floor under the Ross correlation, whose
/// T^3.12 dependence falls below it under about 1100 K.
constexpr double kUNMinimumSwellingPercentPerAtomPercent = 1.0;

/// Total volumetric swelling of UN in per cent: S. B. Ross, M. S. El-Genk and
/// R. B. Matthews, J. Nucl. Mater. 170 (1990) 169-177 (read in the abstract),
/// 4.7e-11 T^3.12 B^0.83 rho^0.5, with the volume-average temperature T (K),
/// the burnup B in atom per cent and the density rho in per cent of
/// theoretical density.  The abstract gives an accuracy of 25 % above 1.12
/// at. % and up to 60 % below; the data are from space-reactor pins.
inline double
unSwellingRoss(double temperature, double burnup_atom_percent, double density_percent)
{
  if (burnup_atom_percent <= 0)
    return 0.0;
  return 4.7e-11 * std::pow(temperature, 3.12) * std::pow(burnup_atom_percent, 0.83) *
         std::sqrt(density_percent);
}

/// Fraction (per cent) of the fission gas released from UN: E. K. Storms,
/// J. Nucl. Mater. 158 (1988) 119-129, R = 100 / (exp(0.0025 (90 TD^0.77 /
/// B^0.09 - T)) + 1), T the average temperature (K), B the burnup (atom per
/// cent), TD the density (per cent of theoretical).
inline double
unGasReleaseStorms(double temperature, double burnup_atom_percent, double density_percent)
{
  if (burnup_atom_percent <= 0)
    return 0.0;
  const double x =
      0.0025 *
      (90.0 * std::pow(density_percent, 0.77) / std::pow(burnup_atom_percent, 0.09) - temperature);
  return 100.0 / (std::exp(x) + 1.0);
}

// =============================================================================
// Zircaloy-4
// =============================================================================

/// Thermal conductivity of Zircaloy-2 and Zircaloy-4, W/(m K): the
/// recommendation of IAEA-TECDOC-1496 (2008) Sect. 6.2.1.4, Eq. (1),
/// k = 12.767 - 5.4348e-4 T + 8.9818e-6 T^2, a fit to 321 conductivity and
/// diffusivity points, valid 300-1800 K with a one-standard-deviation
/// uncertainty of 4 % (300 K) to 7 % (1200 K).  The source advises against
/// extrapolation, so the value is held at its 1800 K value above 1800 K.
template <typename T>
T
zryConductivity(const T & temperature)
{
  const T t = valueOf(temperature) > 1800.0 ? temperature * 0.0 + 1800.0 : temperature;
  return 12.767 - 5.4348e-4 * t + 8.9818e-6 * t * t;
}

/// The MATPRO CTHCON conductivity, NUREG/CR-6150 Vol. 4, k = 7.51 + 2.09e-2 T
/// - 1.45e-5 T^2 + 7.67e-9 T^3 below 2098 K, kept for comparison; it runs 5 %
/// above the data between 400 and 1200 K (IAEA-TECDOC-1496, Sect. 6.2.1.4).
template <typename T>
T
zryConductivityMATPRO(const T & temperature)
{
  const T & t = temperature;
  if (valueOf(t) >= 2098.0)
    return T(36.0);
  return 7.51 + 2.09e-2 * t - 1.45e-5 * t * t + 7.67e-9 * t * t * t;
}

/// Specific heat of Zircaloy, J/(kg K), 273-2000 K: the recommendation of
/// IAEA-TECDOC-1496 (2006), with the Gaussian peak of the alpha-beta
/// transition.
template <typename T>
T
zrySpecificHeat(const T & temperature)
{
  const T & t = temperature;
  const double v = valueOf(t);
  const T peak = 1058.4 * exp(-(t - 1213.8) * (t - 1213.8) / 719.61);
  if (v < 1100.0)
    return 255.66 + 0.1024 * t;
  if (v < 1214.0)
    return 255.66 + 0.1024 * t + peak;
  if (v < 1320.0)
    return 597.1 - 0.4088 * t + 1.565e-4 * t * t + peak;
  return 597.1 - 0.4088 * t + 1.565e-4 * t * t;
}

/// Density of alpha-phase Zircaloy, kg/m^3 (IAEA-TECDOC-1496).
template <typename T>
T
zryDensity(const T & temperature)
{
  return 6595.2 - 0.1477 * temperature;
}

/// Thermal strains of alpha-phase Zircaloy cladding of unknown texture, 300 K
/// to 1083 K: IAEA-TECDOC-1496 (2008) Sect. 6.2.1.5, Eqs. (4)-(6), the MATPRO
/// fits to the data of Bunnell et al.: diametral (hoop) -2.128e-3 + 7.092e-6
/// T, axial -1.623e-3 + 5.458e-6 T and, for epsilon_33(Lab), -2.998e-3 +
/// 9.999e-6 T, which MATPRO assigns to the radial direction (the c-axes of
/// the grains of a tube lie near the radial direction).  The strains are
/// measured from a reference temperature, so only the slopes enter.
constexpr double kZryAlphaHoop = 7.092e-6;
constexpr double kZryAlphaAxial = 5.458e-6;
constexpr double kZryAlphaRadial = 9.999e-6;
template <typename T>
T
zryAxialThermalStrain(const T & temperature, double reference)
{
  return kZryAlphaAxial * (temperature - reference);
}
template <typename T>
T
zryHoopThermalStrain(const T & temperature, double reference)
{
  return kZryAlphaHoop * (temperature - reference);
}
template <typename T>
T
zryRadialThermalStrain(const T & temperature, double reference)
{
  return kZryAlphaRadial * (temperature - reference);
}

/// Young's and shear moduli of alpha-phase Zircaloy, Pa: MATPRO CELMOD and
/// CSHEAR, NUREG/CR-6150 Vol. 4 Eqs. (4-71), (4-74), (4-75) and (4-76),
/// read in the report, without the oxygen term (as-received oxygen), with
/// the cold-work term K2 = -2.6e10 C (C the cold work as a ratio of areas)
/// and the fast fluence (n/m^2) factor K3 = 0.88 + 0.12 exp(-fluence / 1e25).
/// Standard error of CELMOD 6.4 GPa.
template <typename T>
T
zryYoungsModulus(const T & temperature, double fluence, double cold_work = 0.0)
{
  return (1.088e11 - 5.475e7 * temperature - 2.6e10 * cold_work) /
         (0.88 + 0.12 * std::exp(-fluence / 1e25));
}
template <typename T>
T
zryShearModulus(const T & temperature, double fluence, double cold_work = 0.0)
{
  return (4.04e10 - 2.168e7 * temperature - 2.6e10 * cold_work) /
         (0.88 + 0.12 * std::exp(-fluence / 1e25));
}

/// Secondary creep rate of stress-relieved Zircaloy-4, 1/s, in the form of
/// M. Limbäck and T. Andersson (ASTM STP 1295, 1996) with the irradiation
/// creep of N. E. Hoppe (1991), with the constants documented publicly for
/// BISON: thermal A E/T [sinh(a_i sigma/E)]^n exp(-Q/RT) per hour with
/// A = 1.08e9 K/MPa/h, n = 2, Q = 201 kJ/mol, E = 1.148e5 - 59.9 T MPa and
/// a_i = 650 [1 - 0.56 (1 - exp(-1.4e-27 phi_t^1.3))] with the fast fluence
/// phi_t in n/cm^2; irradiation 3.557e-24 phi^0.85 sigma per hour with the
/// fast flux phi in n/(m^2 s) and sigma in MPa.  The original papers were not
/// read; the FRAPCON-4.0 report gives a different irradiation constant with a
/// temperature factor.  Primary creep is not included.
template <typename T>
T
zryCreepRate(const T & stress, const T & temperature, double fast_flux, double fast_fluence)
{
  const T s = stress / 1e6; // MPa
  const T E = 1.148e5 - 59.9 * temperature;
  const double fluence_cm2 = fast_fluence * 1e-4;
  const double ai = 650.0 * (1.0 - 0.56 * (1.0 - std::exp(-1.4e-27 * std::pow(fluence_cm2, 1.3))));
  const T sh = sinh_(ai * s / E);
  const T thermal = 1.08e9 * E / temperature * sh * sh * exp(-201000.0 / (8.314 * temperature));
  const T irradiation = 3.557e-24 * std::pow(fast_flux, 0.85) * s;
  return (thermal + irradiation) / 3600.0;
}

/// Meyer hardness of Zircaloy, Pa: MATPRO CMHARD, NUREG/CR-6150 Vol. 4
/// Eq. (4-280), MH = exp(26.034 - 2.6394e-2 T + 4.3504e-5 T^2 - 2.5621e-8 T^3),
/// fitted to the data of Peggs and Godin (298-877 K), with the report's
/// minimum of 1e5 Pa.  The scanned report shows the sign of the cubic term
/// indistinctly; the minus sign reproduces its Fig. 4-59 (about 2000 MPa at
/// room temperature and 200 MPa at 875 K), the plus sign does not.
template <typename T>
T
zryMeyerHardness(const T & temperature)
{
  const T & t = temperature;
  const T mh = exp(26.034 + t * (-2.6394e-2 + t * (4.3504e-5 - 2.5621e-8 * t)));
  return valueOf(mh) < 1e5 ? t * 0.0 + 1e5 : mh;
}

/// Axial irradiation growth of stress-relieved Zircaloy-4: D. G. Franklin,
/// ASTM STP 754 (1982), eps = 2.18e-21 phi_t^0.845 with the fast fluence in
/// n/cm^2 (coefficients from the FRAPCON-4.0 material property report).
inline double
zryIrradiationGrowth(double fast_fluence)
{
  const double fluence_cm2 = fast_fluence * 1e-4;
  return fluence_cm2 > 0 ? 2.18e-21 * std::pow(fluence_cm2, 0.845) : 0.0;
}

// =============================================================================
// Gases of the fuel-cladding gap
// =============================================================================

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
/// Table 13-2.  They are used for hydrogen and nitrogen; for the noble gases
/// they are kept for comparison only (helium 3-4 % low, krypton and xenon 5-17
/// % high against Kestin et al. 1984).
constexpr double kGasA[kNumGases] = {2.639e-3, 2.986e-4, 8.247e-5, 4.351e-5, 1.097e-3, 5.314e-4};
constexpr double kGasB[kNumGases] = {0.7085, 0.7224, 0.8363, 0.8616, 0.8785, 0.6898};
/// ln k = c0 + c1 ln T + c2 (ln T)^2 + c3 (ln T)^3, k in W/(m K): fits to the
/// low-density conductivities of Tables 1, 3, 4 and 5 of J. Kestin, K.
/// Knierim, E. A. Mason, B. Najafi, S. T. Ro and M. Waldman, J. Phys. Chem.
/// Ref. Data 13 (1984) 229-303, over 200-2273 K (maximum deviation 0.005 %
/// He, 0.05 % Ar and Kr, 1.0 % Xe).  The fit is reproduced by
/// verification/correlations/kestin1984_gas_fit.py.
constexpr double kGasKestin[4][4] = {
    {-5.2532347534e+00, 5.1483336706e-01, 1.4181490440e-02, 9.3426153211e-06},
    {-1.6548530604e+01, 4.2657846105e+00, -4.8906679795e-01, 2.2102852922e-02},
    {-1.8704655012e+01, 4.7549670873e+00, -5.3442160336e-01, 2.3257258353e-02},
    {-1.6433076351e+01, 3.2558216504e+00, -2.7966068851e-01, 9.4971467825e-03}};

Gas gasFromName(const std::string & name);

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

/// Sum of the temperature jump distances at the two surfaces of the gap, m:
/// the equation of E. H. Kennard for a mixture of monatomic gases as D. D.
/// Lanning and C. R. Hann (BNWL-1894, 1975), Appendix B, Sect. 2, write it,
///   g = 2878 (2 - a)/a k sqrt(T) / P (sum x_i / M_i)^(-1/2)
/// per surface, with k in cal/(s cm K), P in dyn/cm^2 and g in cm.  In SI
/// units (k in W/(m K), P in Pa, g in m, M in g/mol) the constant of the sum
/// g1 + g2 is 2 x 2878 / 41868 = 0.013748.  For the diatomic H2 and N2 the
/// monatomic heat capacities overstate the jump distance by half.  The
/// accommodation coefficient a is interpolated between helium (0.425 -
/// 2.3e-4 T) and xenon (0.749 - 2.5e-4 T), the fits of Ullman et al., by the
/// mixture molar mass sum x_i M_i (Lanning and Hann, Appendix B, Sect. 1).
/// The helium fit turns negative at 1848 K, so both fits are bounded below by
/// 0.07, the MATPRO estimate for helium on Zircaloy (NUREG/CR-6150 Vol. 4
/// Table 13-4).  The helium bound acts above 1543 K, and the xenon fit reaches
/// it only above 2700 K.
template <typename T>
T
gasJumpDistance(const double * x, const T & temperature, double pressure, const T & k_mix)
{
  double mass = 0.0, inverse = 0.0;
  for (int i = 0; i < kNumGases; ++i)
  {
    mass += x[i] * kGasMolarMass[i] * 1000.0;
    if (x[i] > 0)
      inverse += x[i] / (kGasMolarMass[i] * 1000.0);
  }
  const double mhe = kGasMolarMass[0] * 1000.0, mxe = kGasMolarMass[3] * 1000.0;
  T ahe = 0.425 - 2.3e-4 * temperature;
  T axe = 0.749 - 2.5e-4 * temperature;
  if (valueOf(ahe) < 0.07)
    ahe = temperature * 0.0 + 0.07;
  if (valueOf(axe) < 0.07)
    axe = temperature * 0.0 + 0.07;
  const T a = ahe + (axe - ahe) * ((mass - mhe) / (mxe - mhe));
  return 0.013748 * (2.0 - a) / a * k_mix * sqrt(temperature) / pressure / std::sqrt(inverse);
}

/// Conductance of the solid contact between two rough surfaces, W/(m^2 K):
/// Eq. (A.9) of A. M. Ross and R. L. Stoute (AECL-1552, 1962),
///   h_s = k_m P_c / (a_0 R^(1/2) H),  R = ((R_1^2 + R_2^2)/2)^(1/2),
/// with the harmonic mean conductivity k_m = 2 k_1 k_2/(k_1 + k_2), the
/// contact pressure P_c, the Meyer hardness H of the softer surface and the
/// arithmetic mean roughnesses R_1 and R_2.  coefficient is 1/a_0,
/// m^(-1/2).  Ross and Stoute found a_0 between 1/2 and 1 cm^(1/2), and
/// about 1/2 cm^(1/2) (coefficient 20 m^(-1/2)) for most of their pairs
/// (Sect. 6.1).
template <typename T>
T
solidContactConductance(const T & mean_conductivity,
                        const T & contact_pressure,
                        const T & hardness,
                        double roughness_1,
                        double roughness_2,
                        double coefficient)
{
  const double R = std::sqrt(0.5 * (roughness_1 * roughness_1 + roughness_2 * roughness_2));
  return coefficient * mean_conductivity * contact_pressure / (std::sqrt(R) * hardness);
}

} // namespace fuel
} // namespace dualmesh
