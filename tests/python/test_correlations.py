# SPDX-License-Identifier: LGPL-2.1-or-later
"""Check values of the correlations against their primary sources.

Every number asserted here was read in the source named next to it, or
computed by hand from an equation read there.  The chapter
docs/theory/correlations.rst lists the same checks.
"""

from __future__ import annotations

import numpy as np
import pytest
from dualmesh import fuel
from dualmesh.fuel import water

props = fuel.properties


# ---------------------------------------------------------------------------
# Water (IAPWS)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "p, T, v, h, cp",
    [
        # IAPWS R7-97(2012), Table 5.
        (3e6, 300.0, 0.100215168e-2, 0.115331273e3, 0.417301218e1),
        (80e6, 300.0, 0.971180894e-3, 0.184142828e3, 0.401008987e1),
        (3e6, 500.0, 0.120241800e-2, 0.975542239e3, 0.465580682e1),
    ],
)
def test_if97_region_1(p, T, v, h, cp):
    # The table gives nine significant digits.
    assert water.specific_volume(p, T) == pytest.approx(v, rel=5e-9)
    assert water.enthalpy(p, T) / 1e3 == pytest.approx(h, rel=5e-9)
    assert water.isobaric_heat_capacity(p, T) / 1e3 == pytest.approx(cp, rel=5e-9)
    # The inverse T(p, h) is exact.
    assert water.temperature(p, water.enthalpy(p, T)) == pytest.approx(T, abs=1e-8)


def test_if97_saturation_and_transport():
    # R7-97 Table 35 (read backwards through Eq. 31).
    ps = np.array([0.353658941e-2, 0.263889776e1, 0.123443146e2]) * 1e6
    assert water.saturation_temperature(ps) == pytest.approx([300.0, 500.0, 600.0], abs=1e-6)
    # R12-08 Table 4.
    assert water.viscosity_from_density(998.0, 298.15) * 1e6 == pytest.approx(889.735100, rel=1e-8)
    assert water.viscosity_from_density(100.0, 873.15) * 1e6 == pytest.approx(35.802262, rel=1e-7)
    # R15-11 Table 7 (industrial formulation, region 1), mW/(m K).
    for p, expected in ((20e6, (481.485195, 48.4911627, 9.66869008, 12.6391714)),
                        (50e6, (545.038940, 48.4911627, 11.1212177, 5.75816285))):  # fmt: skip
        assert water.thermal_conductivity(p, 620.0, parts=True) == pytest.approx(expected, rel=1e-8)


def test_coolant_enthalpy_balance_and_correlations():
    """A PWR channel (pitch 12.6 mm, cladding radius 4.75 mm, 3500
    kg/(m^2 s), 15.5 MPa, 565 K inlet) heated by 17.9 kW/m over 3.66 m: the
    enthalpy balance with IAPWS properties gives 601.6 K at the outlet,
    2 K below the constant-cp estimate with c_p = 5500 J/(kg K); Weisman's
    coefficient is about 40 % above Dittus-Boelter's."""
    coolant = fuel.ForcedConvection(
        inlet_temperature=565.0, pressure=15.5e6, mass_flux=3500.0, rod_pitch=12.6e-3
    )
    radius = 4.75e-3
    outlet = coolant.bulk_temperature(17.9e3 * 3.66, radius)
    assert outlet == pytest.approx(601.6, abs=0.1)
    constant = fuel.ForcedConvection(
        inlet_temperature=565.0,
        pressure=15.5e6,
        mass_flux=3500.0,
        rod_pitch=12.6e-3,
        specific_heat=5500.0,
        thermal_conductivity=0.55,
        dynamic_viscosity=8.8e-5,
    )
    assert constant.bulk_temperature(17.9e3 * 3.66, radius) - outlet == pytest.approx(2.1, abs=0.3)
    weisman = fuel.ForcedConvection(
        inlet_temperature=565.0,
        pressure=15.5e6,
        mass_flux=3500.0,
        rod_pitch=12.6e-3,
        correlation="weisman",
    )
    ratio = weisman.heat_transfer_coefficient_for(radius, 585.0) / (
        coolant.heat_transfer_coefficient_for(radius, 585.0)
    )
    assert ratio == pytest.approx(1.39, abs=0.03)
    with pytest.raises(ValueError, match="saturation temperature"):
        coolant.bulk_temperature(60e3 * 3.66, radius)


# ---------------------------------------------------------------------------
# UO2
# ---------------------------------------------------------------------------
def test_uo2_creep_matpro_transition_stress():
    """MATPRO FCREEP, NUREG/CR-6150 Vol. 4 Eqs. (2-60)-(2-63): above the
    transition stress 1.6547e7 / G^0.5714 Pa the first term takes the
    transition stress.  Hand values with Q1 = 72124.23 and Q2 = 111543.5
    cal/mol at O/M = 2 (the stoichiometric limit of f(x) = 1/(e^-8 + 1))."""
    R = 8.314462618
    fx = 1.0 / (np.exp(-8.0) + 1.0)
    Q1 = (17884.8 * fx + 72124.23) * 4.184
    Q2 = (19872.0 * fx + 111543.5) * 4.184

    def hand(s, T, F, G=10.0, D=95.0):
        st = 1.6547e7 / G**0.5714
        return (
            (0.3919 + 1.31e-19 * F) * min(s, st) * np.exp(-Q1 / (R * T)) / ((-87.7 + D) * G**2)
            + 2.0391e-25 * s**4.5 * np.exp(-Q2 / (R * T)) / (-90.5 + D)
            + 3.72264e-35 * F * s * np.exp(-2616.8 / T)
        )

    for s, T in ((20e6, 1500.0), (60e6, 1800.0), (2e6, 1500.0)):
        assert props.uo2_creep_rate(s, T, 1e19, 0.95, 5e-6) == pytest.approx(
            hand(s, T, 1e19), rel=1e-9
        )
    # The audit's check values: 2.10e-9 and 6.32e-7 1/s.
    assert props.uo2_creep_rate(20e6, 1500.0, 1e19, 0.95, 5e-6) == pytest.approx(
        2.1008e-9, rel=1e-3
    )
    assert props.uo2_creep_rate(60e6, 1800.0, 1e19, 0.95, 5e-6) == pytest.approx(
        6.3216e-7, rel=1e-3
    )


def test_uo2_densification_matpro():
    """MATPRO FUDENS, Eqs. (2-81), (2-82) and (2-85): zero at zero burnup,
    and at large burnup (dL/L)_m = -0.00285 RSNTR per cent above 1000 K
    (-0.0015 RSNTR below), three times that in volume."""
    rsntr = 0.01 * 10963.0
    assert props.uo2_densification(1200.0, 0.0) == pytest.approx(0.0, abs=1e-15)
    late = props.uo2_densification(1200.0, 0.2)
    assert late == pytest.approx(3 * -0.00285 * rsntr / 100.0, rel=1e-6)
    cold = props.uo2_densification(800.0, 0.2)
    assert cold == pytest.approx(3 * -0.0015 * rsntr / 100.0, rel=1e-6)
    # Monotone in burnup.
    fima = np.linspace(0.0, 0.02, 21)
    assert np.all(np.diff(props.uo2_densification(1200.0, fima)) <= 0)


def test_xenon_diffusivity_is_turnbull_three_term():
    """D1 and D2 as in Zullo et al. (2023) Table 1 and CASL-U-2019-1870 Eq. 8,
    D3 = 2e-40 F of Turnbull et al. (1982), Eq. 9: 3.7511e-21 m^2/s at 1000 K
    and 1e19 fissions per m^3 per second (9.7511e-21 with the 8e-40 of the
    secondary sources)."""
    model = fuel.BoothFissionGasRelease(1)
    assert model.diffusivity(np.array([1000.0]), np.array([1e19]))[0] == pytest.approx(
        3.7511e-21, rel=1e-4
    )
    assert model.diffusivity(np.array([1500.0]), np.array([1e19]))[0] == pytest.approx(
        2.2704e-19, rel=1e-4
    )
    secondary = fuel.BoothFissionGasRelease(1, athermal_coefficient=8e-40)
    assert secondary.diffusivity(np.array([1000.0]), np.array([1e19]))[0] == pytest.approx(
        9.7511e-21, rel=1e-4
    )


def test_intragranular_trapping_and_grain_face_bubbles():
    """Trapping slows the diffusion (b / (b + g) < 1), the bubble density
    tends to the nucleation/re-solution balance, and grain-face bubbles hold
    the gas at low temperature: nothing is released at 900 K after 3.3
    years at 1e19 fissions/(m^3 s), while at 1500 K about half is."""
    T = np.array([900.0, 1500.0])
    F = np.full(2, 1e19)
    model = fuel.BoothFissionGasRelease(2)
    for k in range(40):
        model.advance(30 * 86400.0, T, F, 5e6, np.full(2, 1e-3 * (k + 1)))
    D = model.diffusivity(T, F)
    effective, b = model._speight(D, F, model.intragranular())
    assert np.all(effective < D)
    assert model.bubble_density == pytest.approx(
        2 * 25.0 * 1e19 / b, rel=1e-2
    )  # near equilibrium nu / b after many re-solution times
    fraction = model.released / model.produced
    assert fraction[0] == 0.0
    assert 0.3 < fraction[1] < 0.7
    # The coverage of the faces never exceeds saturation.
    assert np.all(model.faces.coverage() <= 0.5 + 1e-12)


# ---------------------------------------------------------------------------
# Zircaloy and the gap
# ---------------------------------------------------------------------------
def test_zircaloy_thermal_expansion_is_anisotropic():
    """IAEA-TECDOC-1496 Sect. 6.2.1.5, Eqs. (4)-(6): from 293.15 to 600 K the
    hoop strain is 7.092e-6 x 306.85 = 2.176e-3."""
    radial, hoop, axial = props.zircaloy_thermal_strains(600.0, 293.15)
    assert hoop == pytest.approx(2.176e-3, rel=1e-3)
    assert axial == pytest.approx(5.458e-6 * 306.85, rel=1e-9)
    assert radial == pytest.approx(9.999e-6 * 306.85, rel=1e-9)


def test_helium_conductivity_against_kestin():
    """Kestin et al. (1984) Table 1: helium 156.66 mW/(m K) at 300 K and
    429.84 at 1000 C."""
    assert props.gas_conductivity({"helium": 1.0}, 300.0) * 1e3 == pytest.approx(156.66, rel=1e-3)
    assert props.gas_conductivity({"helium": 1.0}, 1273.15) * 1e3 == pytest.approx(429.84, rel=1e-3)


# ---------------------------------------------------------------------------
# UN
# ---------------------------------------------------------------------------
def test_un_creep_terms():
    """Hayes (dense), the Coble creep of AbdulHameed et al. (2025) Eq. (14),
    and Konovalov et al. (2016) Eq. (13): at 30 MPa, 1400 K and 10 um grains
    the thermal part is 7.740e-8 + 5.582e-9 1/s."""
    thermal = props.un_creep_rate(30e6, 1400.0, 0.0, 0.95, 5e-6)
    assert thermal == pytest.approx(7.740e-8 + 5.582e-9, rel=2e-3)
    without_coble = props.un_creep_rate(30e6, 1400.0, 0.0, 0.95, 0.0)
    assert without_coble == pytest.approx(5.582e-9, rel=2e-3)
    irradiation = props.un_creep_rate(30e6, 800.0, 1e19, 0.95, 0.0) - props.un_creep_rate(
        30e6, 800.0, 0.0, 0.95, 0.0
    )
    assert irradiation == pytest.approx(2.9e-22 * 30 * 1e13 * np.exp(1.0) / 3600, rel=1e-9)


def test_un_swelling_floor():
    """Below about 1100 K the Ross correlation falls under the smallest
    measured swelling rate of nitride fuel, 1 % per at. % (NEA No. 7317)."""
    assert props.un_volumetric_swelling(700.0, 0.02, 0.95) == pytest.approx(0.02, rel=1e-12)
    assert props.un_volumetric_swelling(1500.0, 0.02, 0.95) == pytest.approx(0.0661, abs=1e-4)


def test_zircaloy_meyer_hardness_and_cold_work():
    """MATPRO CMHARD (Eq. 4-280): about 2 GPa at room temperature and 0.2 GPa
    at 875 K (Fig. 4-59); CELMOD with cold work (Eq. 4-74)."""
    assert props.zircaloy_meyer_hardness(298.0) == pytest.approx(1.9e9, rel=0.1)
    assert props.zircaloy_meyer_hardness(875.0) == pytest.approx(2.0e8, rel=0.15)
    assert props.zircaloy_meyer_hardness(3000.0) == 1e5
    e_annealed = props.zircaloy_youngs_modulus(600.0)
    e_cold_worked = props.zircaloy_youngs_modulus(600.0, 0.0, 0.5)
    assert e_annealed - e_cold_worked == pytest.approx(1.3e10, rel=1e-12)


# ---------------------------------------------------------------------------
# Fission gas: burst release, grain-boundary re-solution, White and Tucker
# ---------------------------------------------------------------------------
def _faces(**kw):
    return fuel.GrainFaceBubbles(1, 5e-6, micro_cracking=True, **kw)


def test_micro_cracking_parameter_of_barani():
    """Barani et al. (2017), Eq. 6 with B = 10 K and Q = 33: on heating the
    derivative dm/dT peaks at T_infl with the value (1 + Q)^(-(1+Q)/Q) / B =
    2.644e-3 per K (Fig. 1, right, reads about 2.6e-3), m is 0 far below and
    rises slowly towards 1 far above; on cooling it is the mirror image.  Eq. 8: T_infl = 1773 +
    520 exp(-bu/10) K, 2293 K fresh."""
    faces = _faces()
    bu = 1e6  # T_infl = 1773 K
    T = np.linspace(1500.0, 2300.0, 80001)
    m = faces.cracking_parameter(T, bu, True)
    dm = np.gradient(m, T)
    assert T[np.argmax(dm)] == pytest.approx(1773.0, abs=0.05)
    assert dm.max() == pytest.approx(34.0 ** (-34.0 / 33.0) / 10.0, rel=1e-4)
    # Far below T_infl, m ~ exp((T - T_infl)/B); far above it m approaches 1
    # slowly, 1 - Q^(-1/Q) exp(-(T - T_infl)/(Q B)).
    assert m[0] == pytest.approx(np.exp(-273.0 / 10.0), rel=1e-3)
    assert m[-1] == pytest.approx(1 - 33.0 ** (-1 / 33) * np.exp(-527.0 / 330.0), rel=1e-6)
    cooling = faces.cracking_parameter(1773.0 - (T - 1773.0), bu, False)
    assert cooling == pytest.approx(m, abs=1e-12)
    assert faces.cracking_parameter(np.array([2293.0]), 0.0, True)[0] == pytest.approx(
        1.0 - 34.0 ** (-1.0 / 33.0), rel=1e-12
    )


def test_micro_cracking_releases_the_gas_of_cracked_faces_and_heals():
    """A shutdown from 1400 K to 500 K at high burnup cracks the fraction
    1 - exp(-(m(500) - m(1400))) of the faces (Eqs. 4-6), which vent their
    gas (dF = F df_c).  Healing restores the intact fraction as 1 - (1 -
    f0) exp(-bu/1 GWd/tU) (Eqs. 9-11)."""
    faces = _faces()
    arrived = np.array([1e24])
    faces.advance(1e6, np.array([1400.0]), 0.0, arrived, 60.0)
    stored = faces.stored()[0]
    m = faces.cracking_parameter
    expected = 1 - np.exp(-(m(500.0, 60.0, False) - m(1400.0, 60.0, False)))
    vented = faces.advance(1.0, np.array([500.0]), 0.0, np.zeros(1), 60.0)[0]
    assert 1.0 - faces.intact[0] == pytest.approx(expected, rel=1e-9)
    assert 0.15 < expected < 0.3
    assert vented == pytest.approx(expected * stored, rel=1e-9)
    assert faces.stored()[0] + vented == pytest.approx(stored, rel=1e-12)
    f0 = faces.intact[0]
    faces.advance(1.0, np.array([500.0]), 0.0, np.zeros(1), 61.0)
    assert faces.intact[0] == pytest.approx(1 - (1 - f0) * np.exp(-1.0), rel=1e-12)
    # A constant temperature cracks nothing.
    fresh = _faces()
    for bu in (1.0, 2.0, 3.0):
        fresh.advance(1e6, np.array([1400.0]), 0.0, arrived, bu)
    assert fresh.intact[0] == 1.0


def test_gas_is_conserved_with_every_option():
    """Produced = in the grains + on the boundaries + released, element by
    element, with burst release and grain-boundary re-solution, over a
    history with shutdowns."""
    for options in (
        {},
        dict(boundary_resolution=3e-32),
        dict(intragranular_bubbles="white_tucker", venting_shrink=True),
    ):
        shrink = options.pop("venting_shrink", False)
        faces = dict(venting="shrink") if shrink else None
        model = fuel.BoothFissionGasRelease(3, grain_face_parameters=faces, **options)
        T = np.array([900.0, 1300.0, 1700.0])
        F = np.full(3, 1e19)
        for k in range(120):
            on = k % 20 != 19
            model.advance(
                1e6, T if on else np.full(3, 500.0), F if on else 0 * F, 5e6, np.full(3, 1e-4 * k)
            )
        total = model.intragranular() + model.boundary + model.released
        assert total == pytest.approx(model.produced, rel=1e-9)


def test_grain_boundary_resolution_holds_the_surface_concentration():
    """With re-solution from the boundaries the grain concentration at the
    surface is psi = kappa F N_f / (2 D_eff), N_f = G 2a/3 (White and Tucker
    1983, Eq. 33), and less gas leaves the grains than without it."""
    T, F = np.array([1300.0]), np.array([1e19])
    kappa = 3e-32
    with_res = fuel.BoothFissionGasRelease(1, boundary_resolution=kappa, micro_cracking=False)
    without = fuel.BoothFissionGasRelease(1, micro_cracking=False)
    for _ in range(50):
        with_res.advance(1e6, T, F, 5e6)
        without.advance(1e6, T, F, 5e6)
    D = with_res.diffusivity(T, F)
    Deff, _ = with_res._speight(D, F, with_res.intragranular(), T)
    N_f = with_res.boundary * 2 * with_res.grain_radius / 3
    psi = kappa * F * N_f / (2 * Deff)
    # psi was set with the D_eff of the last step, which barely moves.
    assert with_res.boundary_concentration == pytest.approx(psi, rel=2e-2)
    assert (with_res.boundary + with_res.released)[0] < 0.8 * (without.boundary + without.released)[
        0
    ]


def test_white_tucker_bubbles_reproduce_their_table_1():
    """White and Tucker (1983), Eq. 27, against their Table 1 (Baker's
    bubbles, 1000-1800 degC): the radius and density within 4 % and 6 %, and
    with D from the table, g = 4 pi R N D and b = 3.03 pi l_f (R + Z0)^2 F
    (Eqs. 24-25) at F = 9.3e18 m^-3 s^-1 the attenuated D b/(b + g) of the
    table within 7 %."""
    celsius = np.arange(1000.0, 1801.0, 100.0)
    R = np.array([5.5, 6.0, 6.5, 7.0, 8.0, 8.75, 9.75, 11.0, 12.5]) * 1e-10
    N = np.array([8.7, 7.8, 7.0, 6.4, 5.7, 5.3, 4.8, 4.4, 3.8]) * 1e23
    D = np.array(
        [7.98e-21, 1.46e-20, 4.49e-20, 1.74e-19, 6.32e-19, 2.04e-18, 5.83e-18, 1.50e-17, 3.53e-17]
    )
    attenuated = np.array(
        [7.70e-21, 1.37e-20, 3.82e-20, 1.06e-19, 2.02e-19, 2.78e-19, 3.32e-19, 3.76e-19, 4.22e-19]
    )
    T = celsius + 273.15
    F = np.full(len(T), 9.3e18)
    model = fuel.BoothFissionGasRelease(
        len(T), intragranular_bubbles="white_tucker", micro_cracking=False
    )
    effective, _ = model._speight(D, F, np.zeros(len(T)), T)
    assert model.intragranular_bubble_radius == pytest.approx(R, rel=0.04)
    assert model.bubble_density == pytest.approx(N, rel=0.065)
    assert effective == pytest.approx(attenuated, rel=0.07)
