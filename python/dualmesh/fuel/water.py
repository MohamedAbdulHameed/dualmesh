# SPDX-License-Identifier: LGPL-2.1-or-later
r"""Properties of compressed liquid water from the IAPWS formulations.

The coolant of a pressurised water reactor is liquid water at about 15.5 MPa
and 560 to 620 K, which is region 1 of the IAPWS Industrial Formulation 1997.
This module evaluates, as functions of pressure and temperature:

* the specific volume, enthalpy, isobaric and isochoric heat capacities from
  the dimensionless Gibbs free energy of region 1, IAPWS R7-97(2012), Eq. (7)
  and Tables 2 to 4;
* the temperature for a given enthalpy, by Newton's method on the same
  equation (instead of the backward equation, Eq. 11, so that the two are
  exactly consistent);
* the saturation temperature, Eq. (31) with Table 34;
* the viscosity, IAPWS R12-08, Eqs. (10) to (12) with the critical
  enhancement set to one, the recommendation for industrial use (Sect. 3);
* the thermal conductivity, IAPWS R15-11, Eqs. (15) to (26), with the
  industrial recommendation of Sect. 3.1 (the thermodynamic properties from
  IF97 and the reference compressibility from Eq. 25).

All the equations are taken from the IAPWS releases, and the test module
checks the values of their verification tables (IF97 Tables 5 and 35;
R12-08 Table 4; R15-11 Table 7).  Units are SI: Pa, K, J/kg, kg/m^3, Pa s and
W/(m K).  The functions take and return numpy arrays or floats.
"""

from __future__ import annotations

import numpy as np

#: Specific gas constant of IF97, J/(kg K) (R7-97 Eq. 1).
R_IF97 = 461.526
#: Specific gas constant of R15-11 (and IAPWS-95), J/(kg K).
R_IAPWS95 = 461.51805
#: Critical constants (R15-11 Eqs. 1-3).
T_CRITICAL, P_CRITICAL, RHO_CRITICAL = 647.096, 22.064e6, 322.0

# R7-97 Table 2: I, J, n of region 1.
# fmt: off
_I1 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 8, 8,
                21, 23, 29, 30, 31, 32], dtype=float)
_J1 = np.array([-2, -1, 0, 1, 2, 3, 4, 5, -9, -7, -1, 0, 1, 3, -3, 0, 1, 3, 17, -4, 0, 6, -5, -2,
                10, -8, -11, -6, -29, -31, -38, -39, -40, -41], dtype=float)
_N1 = np.array([
    0.14632971213167, -0.84548187169114, -0.37563603672040e1, 0.33855169168385e1,
    -0.95791963387872, 0.15772038513228, -0.16616417199501e-1, 0.81214629983568e-3,
    0.28319080123804e-3, -0.60706301565874e-3, -0.18990068218419e-1, -0.32529748770505e-1,
    -0.21841717175414e-1, -0.52838357969930e-4, -0.47184321073267e-3, -0.30001780793026e-3,
    0.47661393906987e-4, -0.44141845330846e-5, -0.72694996297594e-15, -0.31679644845054e-4,
    -0.28270797985312e-5, -0.85205128120103e-9, -0.22425281908000e-5, -0.65171222895601e-6,
    -0.14341729937924e-12, -0.40516996860117e-6, -0.12734301741641e-8, -0.17424871230634e-9,
    -0.68762131295531e-18, 0.14478307828521e-19, 0.26335781662795e-22, -0.11947622640071e-22,
    0.18228094581404e-23, -0.93537087292458e-25])
# R7-97 Table 34: saturation line.
_NS = np.array([0.11670521452767e4, -0.72421316703206e6, -0.17073846940092e2, 0.12020824702470e5,
                -0.32325550322333e7, 0.14915108613530e2, -0.48232657361591e4, 0.40511340542057e6,
                -0.23855557567849, 0.65017534844798e3])
# R12-08 Tables 1 and 2.
_H0 = np.array([1.67752, 2.20462, 0.6366564, -0.241605])
_H1 = np.zeros((6, 7))
for _i, _j, _h in [(0, 0, 5.20094e-1), (1, 0, 8.50895e-2), (2, 0, -1.08374), (3, 0, -2.89555e-1),
                   (0, 1, 2.22531e-1), (1, 1, 9.99115e-1), (2, 1, 1.88797), (3, 1, 1.26613),
                   (5, 1, 1.20573e-1), (0, 2, -2.81378e-1), (1, 2, -9.06851e-1),
                   (2, 2, -7.72479e-1), (3, 2, -4.89837e-1), (4, 2, -2.57040e-1),
                   (0, 3, 1.61913e-1), (1, 3, 2.57399e-1), (0, 4, -3.25372e-2),
                   (3, 4, 6.98452e-2), (4, 5, 8.72102e-3), (3, 6, -4.35673e-3),
                   (5, 6, -5.93264e-4)]:
    _H1[_i, _j] = _h
# R15-11 Tables 1, 2, 3 and 6.
_L0 = np.array([2.443221e-3, 1.323095e-2, 6.770357e-3, -3.454586e-3, 4.096266e-4])
_L1 = np.array([
    [1.60397357, -0.646013523, 0.111443906, 0.102997357, -0.0504123634, 0.00609859258],
    [2.33771842, -2.78843778, 1.53616167, -0.463045512, 0.0832827019, -0.00719201245],
    [2.19650529, -4.54580785, 3.55777244, -1.40944978, 0.275418278, -0.0205938816],
    [-1.21051378, 1.60812989, -0.621178141, 0.0716373224, 0.0, 0.0],
    [-2.7203370, 4.57586331, -3.18369245, 1.1168348, -0.19268305, 0.012913842]])
_A_ZETA = np.array([
    [6.53786807199516, 6.52717759281799, 5.35500529896124, 1.55225959906681, 1.11999926419994],
    [-5.61149954923348, -6.30816983387575, -3.96415689925446, 0.464621290821181,
     0.595748562571649],
    [3.39624167361325, 8.08379285492595, 8.91990208918795, 8.93237374861479, 9.88952565078920],
    [-2.27492629730878, -9.82240510197603, -12.0338729505790, -11.0321960061126,
     -10.3255051147040],
    [10.2631854662709, 12.1358413791395, 9.19494865194302, 6.16780999933360, 4.66861294457414],
    [1.97815050331519, -5.54349664571295, -2.16866274479712, -0.965458722086812,
     -0.503243546373828]])
_ZETA_BOUNDS = np.array([0.310559006, 0.776397516, 1.242236025, 1.863354037])
# fmt: on


def _gibbs(p, T):
    """gamma and its derivatives (g, g_pi, g_pipi, g_tau, g_tautau, g_pitau)
    of region 1 at pressure p (Pa) and temperature T (K)."""
    pi = np.asarray(p, dtype=float)[..., None] / 16.53e6
    tau = 1386.0 / np.asarray(T, dtype=float)[..., None]
    a, b = 7.1 - pi, tau - 1.222
    ei, ej, n = _I1, _J1, _N1
    g = np.sum(n * a**ei * b**ej, axis=-1)
    g_pi = np.sum(-n * ei * a ** (ei - 1) * b**ej, axis=-1)
    g_pipi = np.sum(n * ei * (ei - 1) * a ** (ei - 2) * b**ej, axis=-1)
    g_tau = np.sum(n * a**ei * ej * b ** (ej - 1), axis=-1)
    g_tautau = np.sum(n * a**ei * ej * (ej - 1) * b ** (ej - 2), axis=-1)
    g_pitau = np.sum(-n * ei * a ** (ei - 1) * ej * b ** (ej - 1), axis=-1)
    return g, g_pi, g_pipi, g_tau, g_tautau, g_pitau, pi[..., 0], tau[..., 0]


def specific_volume(p, T):
    """m^3/kg (R7-97 Table 3)."""
    _, g_pi, *_rest, pi, _tau = _gibbs(p, T)
    return pi * g_pi * R_IF97 * np.asarray(T, dtype=float) / np.asarray(p, dtype=float)


def density(p, T):
    """kg/m^3."""
    return 1.0 / specific_volume(p, T)


def enthalpy(p, T):
    """J/kg (R7-97 Table 3: h / RT = tau gamma_tau)."""
    _, _, _, g_tau, _, _, _, tau = _gibbs(p, T)
    return tau * g_tau * R_IF97 * np.asarray(T, dtype=float)


def isobaric_heat_capacity(p, T):
    """J/(kg K) (cp / R = -tau^2 gamma_tautau)."""
    _, _, _, _, g_tautau, _, _, tau = _gibbs(p, T)
    return -(tau**2) * g_tautau * R_IF97


def isochoric_heat_capacity(p, T):
    """J/(kg K) (cv / R = -tau^2 gamma_tautau + (gamma_pi - tau gamma_pitau)^2 /
    gamma_pipi)."""
    _, g_pi, g_pipi, _, g_tautau, g_pitau, _, tau = _gibbs(p, T)
    return (-(tau**2) * g_tautau + (g_pi - tau * g_pitau) ** 2 / g_pipi) * R_IF97


def density_pressure_derivative(p, T):
    """(d rho / d p) at constant T, kg/(m^3 Pa): with v = (R T / p*) gamma_pi,
    dv/dp = (R T / p*^2) gamma_pipi and d rho / dp = -rho^2 dv/dp."""
    _, g_pi, g_pipi, *_ = _gibbs(p, T)
    T = np.asarray(T, dtype=float)
    v = R_IF97 * T / 16.53e6 * g_pi
    dv = R_IF97 * T / 16.53e6**2 * g_pipi
    return -dv / v**2


def temperature(p, h):
    """K, for a pressure (Pa) and a specific enthalpy (J/kg) in region 1, by
    Newton's method on the enthalpy of Eq. (7)."""
    p = np.asarray(p, dtype=float)
    h = np.asarray(h, dtype=float)
    T = np.full(np.broadcast(p, h).shape, 500.0)
    for _ in range(50):
        step = (enthalpy(p, T) - h) / isobaric_heat_capacity(p, T)
        T = T - step
        if np.all(np.abs(step) < 1e-9):
            break
    return T


def saturation_temperature(p):
    """K, R7-97 Eq. (31), 611.213 Pa to 22.064 MPa."""
    beta = (np.asarray(p, dtype=float) / 1e6) ** 0.25
    n = _NS
    E = beta**2 + n[2] * beta + n[5]
    F = n[0] * beta**2 + n[3] * beta + n[6]
    G = n[1] * beta**2 + n[4] * beta + n[7]
    D = 2.0 * G / (-F - np.sqrt(F**2 - 4.0 * E * G))
    return 0.5 * (n[9] + D - np.sqrt((n[9] + D) ** 2 - 4.0 * (n[8] + n[9] * D)))


def viscosity_from_density(rho, T):
    """Pa s, R12-08 Eqs. (10)-(12) with mu_2 = 1."""
    Tb = np.asarray(T, dtype=float) / T_CRITICAL
    rb = np.asarray(rho, dtype=float) / RHO_CRITICAL
    mu0 = 100.0 * np.sqrt(Tb) / sum(_H0[i] / Tb**i for i in range(4))
    x, y = 1.0 / Tb - 1.0, rb - 1.0
    inner = sum(x**i * sum(_H1[i, j] * y**j for j in range(7)) for i in range(6))
    return 1e-6 * mu0 * np.exp(rb * inner)


def viscosity(p, T):
    """Pa s, with the IF97 density (the industrial recommendation)."""
    return viscosity_from_density(density(p, T), T)


def thermal_conductivity(p, T, parts: bool = False):
    """W/(m K), R15-11 Eq. (15) with the industrial recommendation.  With
    ``parts=True`` returns (lambda, lambda_0, lambda_1, lambda_2) in
    mW/(m K), for comparison with the verification tables."""
    T = np.asarray(T, dtype=float)
    rho = density(p, T)
    Tb, rb = T / T_CRITICAL, rho / RHO_CRITICAL
    l0 = np.sqrt(Tb) / sum(_L0[k] / Tb**k for k in range(5))
    x, y = 1.0 / Tb - 1.0, rb - 1.0
    l1 = np.exp(rb * sum(x**i * sum(_L1[i, j] * y**j for j in range(6)) for i in range(5)))
    # Critical enhancement, Eqs. (18)-(26).
    cp = isobaric_heat_capacity(p, T)
    cv = isochoric_heat_capacity(p, T)
    zeta = density_pressure_derivative(p, T) * P_CRITICAL / RHO_CRITICAL
    zeta = np.where((zeta < 0) | (zeta > 1e13), 1e13, zeta)
    cp = np.where((cp < 0) | (cp > 1e13 * R_IAPWS95), 1e13 * R_IAPWS95, cp)
    band = np.searchsorted(_ZETA_BOUNDS, rb, side="left")
    zeta_ref = 1.0 / sum(_A_ZETA[i][band] * rb**i for i in range(6))
    dchi = np.maximum(rb * (zeta - zeta_ref * 1.5 / Tb), 0.0)
    xi = 0.13 * (dchi / 0.06) ** (0.630 / 1.239)  # nm
    ydim = xi / 0.40
    kappa = cp / cv
    mu = viscosity_from_density(rho, T) / 1e-6
    cpb = cp / R_IAPWS95
    with np.errstate(divide="ignore", invalid="ignore"):
        Z = 2.0 / (np.pi * ydim) * ((1.0 - 1.0 / kappa) * np.arctan(ydim) + ydim / kappa - (1.0 - np.exp(-1.0 / (1.0 / ydim + ydim**2 / (3.0 * rb**2)))))
    Z = np.where(ydim < 1.2e-7, 0.0, Z)
    l2 = 177.8514 * rb * cpb * Tb / mu * Z
    total = l0 * l1 + l2
    if parts:
        return total, l0, l1, l2
    return 1e-3 * total


def properties(p, T) -> dict:
    """Density, specific heat, conductivity and viscosity at (p, T)."""
    return dict(density=density(p, T), specific_heat=isobaric_heat_capacity(p, T), thermal_conductivity=thermal_conductivity(p, T), viscosity=viscosity(p, T))
