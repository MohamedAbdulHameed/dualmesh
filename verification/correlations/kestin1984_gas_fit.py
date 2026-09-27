# SPDX-License-Identifier: LGPL-2.1-or-later
"""Fit of the low-density thermal conductivity of He, Ar, Kr and Xe.

The data are the thermal conductivities (column lambda, mW/(m K)) of Tables 1,
3, 4 and 5 of J. Kestin, K. Knierim, E. A. Mason, B. Najafi, S. T. Ro and
M. Waldman, "Equilibrium and transport properties of the noble gases and
their mixtures at low density", J. Phys. Chem. Ref. Data 13 (1984) 229-303,
transcribed from the page images of pp. 238-239.  The fit is
ln k = c0 + c1 ln T + c2 (ln T)^2 + c3 (ln T)^3 with k in W/(m K) and T in K, over 200 K to
2273.15 K.  Run this script to reproduce the coefficients that
include/dualmesh/modules/FuelProperties.h uses and the maximum relative
error of the fit.
"""

import numpy as np

KELVIN = [200.0, 250.0, 300.0]
CELSIUS = [
    0,
    20,
    40,
    60,
    80,
    100,
    150,
    200,
    250,
    300,
    350,
    400,
    450,
    500,
    600,
    700,
    800,
    900,
    1000,
    1500,
    2000,
]
# fmt: off
TABLES = {
    "helium": [119.32, 138.53, 156.66,
               147.04, 154.23, 161.29, 168.22, 175.04, 181.75, 198.11, 213.96, 229.37, 244.39,
               259.07, 273.44, 287.54, 301.39, 328.43, 354.70, 380.30, 405.33, 429.84, 546.30,
               655.33],
    "argon": [12.41, 15.23, 17.83,
              16.46, 17.49, 18.49, 19.46, 20.41, 21.33, 23.55, 25.64, 27.65, 29.56,
              31.40, 33.18, None, 36.56, 39.76, 42.80, 45.72, 48.52, 51.23, 63.73, 75.08],
    "krypton": [6.38, 8.00, 9.52,
                8.72, 9.32, 9.91, 10.48, 11.04, 11.59, 12.91, 14.16, 15.36, 16.51,
                17.61, 18.67, 19.70, 20.69, 22.60, 24.41, 26.14, 27.80, 29.40, 36.72, 43.25],
    "xenon": [3.66, 4.57, 5.50,
              5.00, 5.37, 5.75, 6.11, 6.48, 6.83, 7.69, 8.52, 9.31, 10.06,
              10.79, 11.50, 12.18, 12.84, 14.11, 15.31, 16.46, 17.56, 18.62, 23.44, 27.70],
}
# fmt: on
# The argon value at 450 C is illegible on the scan and is left out.


def data(gas):
    T = np.array(KELVIN + [c + 273.15 for c in CELSIUS])
    k = TABLES[gas]
    keep = [v is not None for v in k]
    return T[keep], 1e-3 * np.array([v for v in k if v is not None])


def fit(gas):
    T, k = data(gas)
    x = np.log(T)
    A = np.vstack([x**i for i in range(4)]).T
    c, *_ = np.linalg.lstsq(A, np.log(k), rcond=None)
    model = np.exp(A @ c)
    return c, np.max(np.abs(model / k - 1.0))


if __name__ == "__main__":
    for gas in TABLES:
        c, err = fit(gas)
        print(f"{gas:8s} " + ", ".join(f"{x:.10e}" for x in c) + f"  max error {100 * err:.2f} %")
