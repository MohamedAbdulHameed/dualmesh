# SPDX-License-Identifier: LGPL-2.1-or-later
"""Norton-law fit to the minimum creep rates of chromium.

The data are Table I of J. R. Stephens and W. D. Klopp, "High-temperature
creep of polycrystalline chromium", NASA TM X-2499 (1972), transcribed from
the page image: temperature (C), initial grain size (cm), stress (MN/m^2) and
minimum creep rate (1/s).  The fit is ln(rate) = ln A + n ln(sigma) - Q/(R T)
by least squares over all 49 points, both grain sizes.  Run this script to
reproduce the constants that dualmesh.fuel.ChromiumCoating uses and the
comparison with the law of Wagih et al. (2018), which was fitted to the
816 C data only.
"""

import numpy as np

R = 8.314462618
# fmt: off
DATA = {
    (816, 0.023): [(50.1, 0.37e-6), (61.2, 0.94e-6), (66.9, 1.7e-6), (72.5, 3.5e-6), (78.1, 5.8e-6),
                   (84.6, 8.0e-6), (92.5, 15e-6), (100.5, 31e-6)],
    (816, 0.090): [(54.9, 0.79e-6), (58.7, 1.5e-6), (63.1, 2.5e-6), (72.1, 6.6e-6), (82.7, 22e-6)],
    (982, 0.023): [(20.0, 0.34e-6), (21.2, 1.3e-6), (24.1, 1.7e-6), (27.4, 2.9e-6), (30.8, 4.8e-6),
                   (34.5, 9.8e-6), (38.3, 22e-6), (42.6, 51e-6)],
    (982, 0.090): [(25.6, 0.73e-6), (30.1, 4.0e-6), (33.1, 6.2e-6), (36.5, 11e-6), (40.7, 24e-6),
                   (45.4, 82e-6)],
    (1149, 0.023): [(7.93, 0.30e-6), (11.5, 1.7e-6), (14.5, 4.8e-6), (18.6, 14e-6)],
    (1149, 0.090): [(10.5, 2.0e-6), (11.1, 2.2e-6), (11.8, 2.4e-6), (12.9, 2.8e-6), (13.9, 3.8e-6),
                    (15.4, 9.1e-6), (18.1, 21e-6)],
    (1316, 0.023): [(3.70, 0.35e-6), (4.13, 0.52e-6), (5.01, 1.5e-6), (5.92, 3.3e-6),
                    (7.45, 7.3e-6), (9.38, 20e-6), (11.6, 65e-6)],
    (1316, 0.090): [(5.04, 2.8e-6), (5.67, 4.2e-6), (6.61, 7.8e-6), (7.72, 10e-6)],
}
# fmt: on


def points():
    rows = [(c + 273.15, s, r) for (c, _), values in DATA.items() for s, r in values]
    return np.array(rows).T


def fit():
    T, s, rate = points()
    A = np.vstack([np.ones_like(T), np.log(s), -1.0 / (R * T)]).T
    (lnA, n, Q), *_ = np.linalg.lstsq(A, np.log(rate), rcond=None)
    rms = np.sqrt(np.mean((A @ [lnA, n, Q] - np.log(rate)) ** 2))
    return np.exp(lnA), n, Q, rms


def wagih(T, s):
    return 5.1596e-3 * s**6.2 * np.exp(-306268.8 / (R * T))


if __name__ == "__main__":
    A, n, Q, rms = fit()
    T, s, rate = points()
    print(
        f"{len(T)} points: A = {A:.4g} MPa^-n/s, n = {n:.4f}, Q = {Q:.5g} J/mol, "
        f"rms ln error {rms:.3f}"
    )
    model = A * s**n * np.exp(-Q / (R * T))
    for c in (816, 982, 1149, 1316):
        k = np.isclose(T, c + 273.15)
        print(
            f"{c} C: median fit/data {np.median(model[k] / rate[k]):.2f}, "
            f"Wagih/data {np.median(wagih(T[k], s[k]) / rate[k]):.2f}"
        )
