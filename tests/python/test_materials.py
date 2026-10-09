# SPDX-License-Identifier: LGPL-2.1-or-later
"""Check the material properties against their primary sources.

Every number asserted here was read in the source named next to it.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from dualmesh.materials import gas, water

KESTIN_FIT = Path(__file__).resolve().parents[2] / "verification" / "correlations" / "kestin1984_gas_fit.py"


def kestin_tables():
    spec = importlib.util.spec_from_file_location("kestin1984_gas_fit", KESTIN_FIT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


# ---------------------------------------------------------------------------
# Gases
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name, deviation", [("helium", 5e-5), ("argon", 5.2e-4), ("krypton", 4.6e-4), ("xenon", 1.05e-2)])
def test_the_noble_gases_reproduce_every_value_of_kestin_1984(name, deviation):
    """Tables 1, 3, 4 and 5 of Kestin et al. (1984), 200 K to 2,273 K, to the
    maximum deviation of the fit."""
    temperatures, conductivities = kestin_tables().data(name)
    assert gas.thermal_conductivity({name: 1.0}, temperatures) == pytest.approx(conductivities, rel=deviation)


def test_helium_against_kestin_1984_table_1():
    """156.66 mW/(m K) at 300 K and 429.84 mW/(m K) at 1000 C."""
    assert gas.thermal_conductivity({"helium": 1.0}, 300.0) * 1e3 == pytest.approx(156.66, rel=1e-4)
    assert gas.thermal_conductivity({"helium": 1.0}, 1273.15) * 1e3 == pytest.approx(429.84, rel=1e-4)


def test_hydrogen_and_nitrogen_follow_the_matpro_power_laws():
    """k = A T^B, NUREG/CR-6150 Vol. 4 Table 13-2."""
    temperatures = np.array([300.0, 800.0, 1500.0])
    assert gas.thermal_conductivity({"hydrogen": 1.0}, temperatures) == pytest.approx(1.097e-3 * temperatures**0.8785, rel=1e-14)
    assert gas.thermal_conductivity({"nitrogen": 1.0}, temperatures) == pytest.approx(5.314e-4 * temperatures**0.6898, rel=1e-14)


def test_a_mixture_is_normalized_and_lies_between_its_gases():
    """The mole fractions are normalized, a mixture of one gas is that gas,
    and the conductivity of a helium-xenon mixture falls from that of helium
    to that of xenon as xenon is added."""
    temperature = 700.0
    assert gas.thermal_conductivity({"helium": 2.0, "xenon": 2.0}, temperature) == pytest.approx(gas.thermal_conductivity({"helium": 0.5, "xenon": 0.5}, temperature), rel=1e-14)
    assert gas.thermal_conductivity({"argon": 1.0, "xenon": 0.0}, temperature) == pytest.approx(gas.thermal_conductivity({"argon": 1.0}, temperature), rel=1e-14)
    fractions = np.linspace(0.0, 1.0, 11)
    values = [gas.thermal_conductivity({"helium": 1.0 - x, "xenon": x}, temperature) for x in fractions]
    assert np.all(np.diff(values) < 0.0)
    assert values[0] == pytest.approx(gas.thermal_conductivity({"helium": 1.0}, temperature), rel=1e-14)
    assert values[-1] == pytest.approx(gas.thermal_conductivity({"xenon": 1.0}, temperature), rel=1e-14)


def test_the_temperature_jump_distance_follows_kennard():
    """For helium with a given accommodation coefficient a, the sum of the two
    jump distances is 0.013748 (2 - a)/a k sqrt(T) / (P sqrt(1/M)), M in g/mol
    (Lanning and Hann, BNWL-1894, Appendix B, Sect. 2). It falls as the
    pressure rises and as the accommodation rises."""
    temperature, pressure, accommodation = 600.0, 2.0e6, 0.3
    conductivity = gas.thermal_conductivity({"helium": 1.0}, temperature)
    expected = 0.013748 * (2.0 - accommodation) / accommodation * conductivity * np.sqrt(temperature) / pressure / np.sqrt(1.0 / 4.0026)
    assert gas.temperature_jump_distance({"helium": 1.0}, temperature, pressure, accommodation) == pytest.approx(expected, rel=1e-14)
    assert gas.temperature_jump_distance({"helium": 1.0}, temperature, 2 * pressure, accommodation) == pytest.approx(0.5 * expected, rel=1e-14)
    assert gas.temperature_jump_distance({"helium": 1.0}, temperature, pressure, 0.6) < expected
    # Without a coefficient, Ullman's helium fit 0.425 - 2.3e-4 T applies.
    fitted = 0.425 - 2.3e-4 * temperature
    assert gas.temperature_jump_distance({"helium": 1.0}, temperature, pressure) == pytest.approx(expected * (2 - fitted) / fitted * accommodation / (2 - accommodation), rel=1e-14)


@pytest.mark.parametrize("composition, message", [({"neon": 1.0}, "Unknown gas 'neon'"), ({"helium": 0.0}, "positive mole fraction"), ({"helium": 1.0, "argon": -0.1}, "must not be negative")])
def test_an_invalid_composition_is_refused(composition, message):
    with pytest.raises(Exception, match=message):
        gas.thermal_conductivity(composition, 300.0)
