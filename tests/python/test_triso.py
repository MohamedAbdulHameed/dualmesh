# SPDX-License-Identifier: LGPL-2.1-or-later
"""TRISO coating stresses against the closed form solutions of CRP-6.

The benchmark cases are those of IAEA-TECDOC-1674 (2012), section 9.2.
"""

import dataclasses

import numpy as np
import pytest
from dualmesh.fuel import triso


def lame_tangential(r, ri, ro, pi, po):
    """Tangential stress in a thick sphere under inner and outer pressure."""
    return (pi * ri**3 * (2 * r**3 + ro**3) - po * ro**3 * (2 * r**3 + ri**3)) / (2 * r**3 * (ro**3 - ri**3))


def lame_displacement(r, ri, ro, pi, po, e, nu):
    a = (pi * ri**3 - po * ro**3) / (ro**3 - ri**3)
    b = (pi - po) * ri**3 * ro**3 / (ro**3 - ri**3)
    return a * (1 - 2 * nu) / e * r + b * (1 + nu) / (2 * e) / r**2


def bonded_shells(swelling_strain):
    """Closed form of cases 3 and 4a: IPyC with an isotropic strain g, bonded to SiC."""
    ri, rm, ro = 350.0, 390.0, 425.0
    pi, po = 25.0, 0.1

    def mismatch(p):
        ua = lame_displacement(rm, ri, rm, pi, p, 3.96e4, 0.33) + swelling_strain * rm
        ub = lame_displacement(rm, rm, ro, p, po, 3.70e5, 0.13)
        return ua - ub

    # The mismatch is linear in the interface pressure.
    p0, p1 = mismatch(0.0), mismatch(1.0)
    p = -p0 / (p1 - p0)
    return -p, lame_tangential(ri, ri, rm, pi, p), lame_tangential(rm, rm, ro, p, po)


def mpa(values):
    return np.asarray(values) * 1e-6


def test_case1_and_case2_match_lame():
    p, h = triso.crp6_case("1")
    r = triso.TrisoParticleModel(particle=p, history=h).run()
    assert mpa(r.inner_tangential_stress["SiC"][0]) == pytest.approx(lame_tangential(350, 350, 385, 25, 0.1), abs=0.01)
    assert lame_tangential(350, 350, 385, 25, 0.1) == pytest.approx(125.19, abs=0.005)
    p, h = triso.crp6_case("2")
    r = triso.TrisoParticleModel(particle=p, history=h).run()
    assert mpa(r.inner_tangential_stress["IPyC"][0]) == pytest.approx(50.20, abs=0.01)


def test_case3_bonded_shells():
    p, h = triso.crp6_case("3")
    r = triso.TrisoParticleModel(particle=p, history=h).run()
    radial, ipyc, sic = bonded_shells(0.0)
    assert (radial, ipyc, sic) == pytest.approx((-18.76, 8.78, 104.38), abs=0.01)
    assert mpa(r.interface_radial_stress["IPyC/SiC"][0]) == pytest.approx(radial, abs=0.01)
    assert mpa(r.inner_tangential_stress["IPyC"][0]) == pytest.approx(ipyc, abs=0.01)
    assert mpa(r.inner_tangential_stress["SiC"][0]) == pytest.approx(sic, abs=0.01)


def test_case4a_constant_shrinkage_without_creep():
    p, h = triso.crp6_case("4a")
    r = triso.TrisoParticleModel(particle=p, history=h, numerics=triso.ParticleNumerics(steps=10)).run()
    radial, ipyc, sic = bonded_shells(-0.015)
    assert mpa(r.inner_tangential_stress["IPyC"][-1]) == pytest.approx(ipyc, abs=0.1)
    assert mpa(r.inner_tangential_stress["SiC"][-1]) == pytest.approx(sic, abs=0.1)
    assert mpa(r.interface_radial_stress["IPyC/SiC"][-1]) == pytest.approx(radial, abs=0.1)
    # The report's closed form values (926.80 and -845.71 MPa) are 0.2 % off.
    assert ipyc == pytest.approx(926.80, rel=3e-3)
    assert sic == pytest.approx(-845.71, rel=3e-3)


def test_case4b_creep_relaxes_ipyc_to_hydrostatic_pressure():
    p, h = triso.crp6_case("4b")
    r = triso.TrisoParticleModel(particle=p, history=h, numerics=triso.ParticleNumerics(steps=200)).run()
    assert mpa(r.inner_tangential_stress["IPyC"][-1]) == pytest.approx(-25.0, abs=0.02)
    assert mpa(r.interface_radial_stress["IPyC/SiC"][-1]) == pytest.approx(-25.0, abs=0.02)
    assert mpa(r.inner_tangential_stress["SiC"][-1]) == pytest.approx(lame_tangential(390, 390, 425, 25, 0.1), abs=0.02)


def test_case4c_steady_state_of_creep_and_shrinkage():
    p, h = triso.crp6_case("4c")
    r = triso.TrisoParticleModel(particle=p, history=h, numerics=triso.ParticleNumerics(steps=200)).run()
    m = (390 / 350) ** 3
    gdot, k = -0.005, 2.71e-4
    ipyc = -2 * m * gdot / k - 25.0  # Eq. 9.28 with rigid SiC
    radial = -4 * gdot * (m - 1) / (3 * k) - 25.0  # Eq. 9.29
    assert ipyc == pytest.approx(26.05, abs=0.01)
    assert mpa(r.inner_tangential_stress["IPyC"][-1]) == pytest.approx(ipyc, abs=0.1)
    assert mpa(r.interface_radial_stress["IPyC/SiC"][-1]) == pytest.approx(radial, abs=0.05)
    assert mpa(r.inner_tangential_stress["SiC"][-1]) == pytest.approx(86.5, abs=0.1)


def test_case4d_quasi_equilibrium():
    """Eq. 9.30 and 9.31 of the report, with the relaxation dose C = 0.17e25."""
    p, h = triso.crp6_case("4d")
    r = triso.TrisoParticleModel(particle=p, history=h, numerics=triso.ParticleNumerics(steps=300)).run()
    s = triso.CRP6_SWELLING["a"]
    m, k, c = (390 / 350) ** 3, 2.71e-4, 0.17

    def ghat(coefficients, x):
        poly = np.polynomial.Polynomial(coefficients)
        return poly(x) - c * poly.deriv()(x) + c**2 * poly.deriv(2)(x) - c**3 * poly.deriv(3)(x)

    for x in (1.5, 2.0, 2.5, 3.0):
        g1, g3 = ghat(s.tangential, x), ghat(s.radial, x)
        expected = -(4.5 * m * g1 - 1.5 * (m - 1) * (g1 - g3)) / (2.25 * k) - 25.0
        actual = mpa(np.interp(x * 1e25, r.fluence, r.inner_tangential_stress["IPyC"]))
        assert actual == pytest.approx(expected, abs=0.5)
    assert mpa(r.maximum_tangential_stress("IPyC")) == pytest.approx(149.1, abs=0.5)
    assert mpa(r.minimum_tangential_stress("SiC")) == pytest.approx(-41.8, abs=0.5)


@pytest.mark.parametrize("case, ipyc_max, sic_min, sic_end", [("5", 182.3, -307.9, -45.3), ("6", 167.8, -288.5, 32.1), ("7", 174.0, -299.6, 9.4)])
def test_cases_5_to_7_regression(case, ipyc_max, sic_min, sic_end):
    """Regression values, inside the spread of the eight CRP-6 codes."""
    p, h = triso.crp6_case(case)
    r = triso.TrisoParticleModel(particle=p, history=h, numerics=triso.ParticleNumerics(steps=300)).run()
    assert mpa(r.maximum_tangential_stress("IPyC")) == pytest.approx(ipyc_max, abs=0.3)
    assert mpa(r.minimum_tangential_stress("SiC")) == pytest.approx(sic_min, abs=0.3)
    assert mpa(r.inner_tangential_stress["SiC"][-1]) == pytest.approx(sic_end, abs=0.3)


def test_case8_temperature_cycles():
    p, h = triso.crp6_case("8")
    assert h.temperature_at(0.0) == 873.0
    assert h.temperature_at(0.29e25) == pytest.approx(1273.0)
    assert h.temperature_at(0.30e25) == pytest.approx(873.0)
    assert h.pressure_at(2.99e25) == pytest.approx(26.13e6)
    r = triso.TrisoParticleModel(particle=p, history=h, numerics=triso.ParticleNumerics(steps=600)).run()
    assert mpa(r.maximum_tangential_stress("IPyC")) == pytest.approx(229.7, abs=1.0)
    assert mpa(r.minimum_tangential_stress("SiC")) == pytest.approx(-417.7, abs=1.0)


def test_creep_correlation_and_swelling_integral():
    # Correlation (d) at 1000 C equals the isothermal constant 2.71e-4 within 0.5 %.
    k = triso.crp6_creep_coefficient(1273.15) / 1e-31
    assert k == pytest.approx(2.71e-4, rel=5e-3)
    for name, s in triso.CRP6_SWELLING.items():
        f = np.array([0.5e25, 2.0e25, 7.0e25])
        h = 1e21
        radial_p, tangential_p = s.strain(f + h)
        radial_m, tangential_m = s.strain(f - h)
        radial_rate, tangential_rate = s.rate(f)
        assert (radial_p - radial_m) / (2 * h) == pytest.approx(radial_rate, rel=1e-5), name
        assert (tangential_p - tangential_m) / (2 * h) == pytest.approx(tangential_rate, rel=1e-5), name


def test_mesh_and_step_convergence():
    p, h = triso.crp6_case("4d")
    fine = triso.TrisoParticleModel(particle=p, history=h, numerics=triso.ParticleNumerics(steps=600, elements_per_layer=32)).run()
    coarse = triso.TrisoParticleModel(particle=p, history=h, numerics=triso.ParticleNumerics(steps=150, elements_per_layer=16)).run()
    assert np.max(np.abs(mpa(fine.inner_tangential_stress["IPyC"]))) > 100
    diff = np.max(np.abs(mpa(fine.inner_tangential_stress["IPyC"][::4]) - mpa(coarse.inner_tangential_stress["IPyC"])))
    assert diff < 0.3


def test_weibull_uniform_stress():
    r = np.linspace(390e-6, 425e-6, 201)
    sigma = np.full_like(r, 300e6)
    s0, m = 9.64e6, 6.0
    volume = 4.0 / 3.0 * np.pi * (r[-1] ** 3 - r[0] ** 3)
    expected = 1 - np.exp(-volume * (300e6 / s0) ** m)
    assert triso.weibull_failure_probability(r, sigma, s0, m) == pytest.approx(expected, rel=1e-4)
    assert triso.weibull_failure_probability(r, -sigma, s0, m) == 0.0


def test_input_validation():
    with pytest.raises(ValueError, match="swelling"):
        triso.CoatingLayer(name="IPyC", thickness=40e-6, youngs_modulus=3.96e10, poissons_ratio=0.33, swelling="z")
    with pytest.raises(ValueError, match="thickness"):
        triso.CoatingLayer(name="IPyC", thickness=0.0, youngs_modulus=3.96e10, poissons_ratio=0.33)
    layer = triso.CoatingLayer(name="IPyC", thickness=40e-6, youngs_modulus=3.96e10, poissons_ratio=0.33)
    with pytest.raises(ValueError, match="unique"):
        triso.TrisoParticle(kernel_diameter=500e-6, buffer_thickness=100e-6, layers=(layer, dataclasses.replace(layer)))
    with pytest.raises(ValueError, match="case"):
        triso.crp6_case("9")


def test_monte_carlo_reduces_to_deterministic_particle():
    p, h = triso.crp6_case("1")
    h = dataclasses.replace(h, internal_pressure=200e6)
    s0, m = 4.0e8, 8.0
    mean, error = triso.monte_carlo_failure_probability(p, h, characteristic_strength=s0, modulus=m, samples=3)
    r = triso.TrisoParticleModel(particle=p, history=h).run()
    sic = r.layer_of_point == 0
    single = triso.weibull_failure_probability(r.radius[sic], r.tangential_stress[sic], s0, m)
    assert 0.0 < single < 1.0
    assert mean == pytest.approx(single, rel=1e-12)
    assert error == 0.0
    spread, _ = triso.monte_carlo_failure_probability(p, h, characteristic_strength=s0, modulus=m, samples=40, thickness_standard_deviation={"SiC": 4e-6})
    # Thinner shells carry more stress, and the failure probability is convex
    # in the thickness, so scatter raises the mean failure fraction.
    assert spread > single
    with pytest.raises(ValueError, match="unknown layers"):
        triso.monte_carlo_failure_probability(p, h, characteristic_strength=s0, modulus=m, thickness_standard_deviation={"OPyC": 1e-6})
