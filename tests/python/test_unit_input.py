# SPDX-License-Identifier: LGPL-2.1-or-later
"""The units of the input: every unit converts to the International System of Units (SI) by
its defined factor, the grammar reads compound units as written, and a value of the wrong
dimension, or a number without the unit its quantity needs, is refused with its key."""

import math
import re
from fractions import Fraction

import pytest
from dualmesh.input.units import base, dimensionless, input_error, parse_quantity, parse_unit, to_SI

length = base("length")
mass = base("mass")
time = base("time")
electric_current = base("electric_current")
temperature = base("temperature")
amount_of_substance = base("amount_of_substance")
pressure = mass / length / time**2
energy = mass * length**2 / time**2
power = energy / time
elementary_charge = 1.602176634e-19


def close(actual, expected):
    return math.isclose(actual, expected, rel_tol=1.0e-14, abs_tol=0.0)


@pytest.mark.parametrize(
    "text, factor, expected",
    [
        ("g", 1.0e-3, mass),
        ("N", 1.0, mass * length / time**2),
        ("Pa", 1.0, pressure),
        ("J", 1.0, energy),
        ("W", 1.0, power),
        ("V", 1.0, power / electric_current),
        ("rad", 1.0, dimensionless),
        ("deg", math.pi / 180.0, dimensionless),
        ("L", 1.0e-3, length**3),
        ("min", 60.0, time),
        ("h", 3600.0, time),
        ("d", 86400.0, time),
        ("bar", 1.0e5, pressure),
        ("eV", elementary_charge, energy),
        ("GPa", 1.0e9, pressure),
        ("um", 1.0e-6, length),
        ("mmol", 1.0e-3, amount_of_substance),
        ("keV", 1.0e3 * elementary_charge, energy),
    ],
)
def test_each_unit_has_its_defined_factor(text, factor, expected):
    parsed_factor, parsed_dimension = parse_unit(text)
    assert close(parsed_factor, factor)
    assert parsed_dimension == expected


def test_a_symbol_of_the_table_wins_over_a_prefix():
    """min is the minute and cd the candela, not a milli-in or a centi-day."""
    assert parse_unit("min") == (60.0, time)
    assert parse_unit("cd") == (1.0, base("luminous_intensity"))
    factor, parsed = parse_unit("mm")
    assert close(factor, 1.0e-3) and parsed == length


def test_compound_units_read_as_written():
    """A prefix applies before the power, division is left associative, and powers may be
    fractions."""
    assert parse_unit("W/(m^2*K)") == (1.0, mass / time**3 / temperature)
    assert parse_unit("J/mol/K") == parse_unit("J/(mol*K)")
    factor, parsed = parse_unit("mm^2")
    assert close(factor, 1.0e-6) and parsed == length**2
    factor, parsed = parse_unit("km/h")
    assert close(factor, 1000.0 / 3600.0) and parsed == length / time
    assert parse_unit("s^(1/2)")[1] == time ** Fraction(1, 2)
    assert parse_unit("kg*m^-1*s^-2")[1] == pressure


@pytest.mark.parametrize(
    "text, message",
    [
        ("m2", "expected"),
        ("m^1.5", "unexpected character"),
        ("m^(1/0)", "denominator 0"),
        ("(m", '")"'),
        ("m/", "at the end"),
        ("N.m", "unexpected character"),
        ("kgg", 'unknown unit "kgg"'),
        ("kmin", "takes no prefix"),
        ("W/(m*degC)", "has an offset from the kelvin"),
    ],
)
def test_a_malformed_unit_is_refused(text, message):
    with pytest.raises(input_error, match=re.escape(message)):
        parse_unit(text)


@pytest.mark.parametrize(
    "text, value, expected",
    [
        ("7850 kg/m^3", 7850.0, mass / length**3),
        ("12e-6 1/K", 12.0e-6, temperature**-1),
        ("5 mm", 5.0e-3, length),
        ("10 min", 600.0, time),
        ("0.3", 0.3, dimensionless),
        ("40 degC", 313.15, temperature),
        ("-40 degF", 233.15, temperature),
        ("90 deg", math.pi / 2, dimensionless),
    ],
)
def test_a_quantity_converts_to_SI(text, value, expected):
    parsed_value, parsed_dimension = parse_quantity(text)
    assert math.isclose(parsed_value, value, rel_tol=1e-12)
    assert parsed_dimension == expected


@pytest.mark.parametrize(
    "text, message",
    [
        ("7850kg/m^3", "separate it from the unit with a space"),
        ("kg/m^3", "not a number followed by a unit"),
        ("nan K", "no finite number"),
    ],
)
def test_a_malformed_quantity_is_refused(text, message):
    with pytest.raises(input_error, match=message):
        parse_quantity(text)


def test_a_number_without_the_unit_of_its_quantity_is_refused_with_its_key():
    """A bare 200e9 for Young's modulus could be in Pa or in GPa: the input states it."""
    key = "properties.Young_modulus"
    with pytest.raises(input_error) as error:
        to_SI(200.0e9, pressure, key)
    assert str(error.value).startswith(key + ":")
    assert "has no unit" in str(error.value)


def test_a_value_of_the_wrong_dimension_is_refused_with_both_units():
    key = "properties.Young_modulus"
    with pytest.raises(input_error) as error:
        to_SI("200 K", pressure, key)
    message = str(error.value)
    assert message.startswith(key + ":")
    assert 'the unit "K"' in message and '"Pa"' in message


def test_a_dimensionless_value_takes_no_unit():
    with pytest.raises(input_error, match="dimensionless and takes no unit"):
        to_SI("0.3 m", dimensionless, "properties.Poisson_ratio")
    assert to_SI(0.3, dimensionless, "properties.Poisson_ratio") == 0.3


def test_a_temperature_in_degrees_Celsius_converts_to_kelvin_with_its_key():
    temperature = base("temperature")
    assert to_SI("40 degC", temperature, "boundary_conditions.left.temperature") == pytest.approx(
        313.15, rel=1e-14
    )
    with pytest.raises(input_error, match="has an offset from the kelvin"):
        to_SI(
            "20 W/(m*degC)",
            mass * length / time**3 / temperature,
            "properties.thermal_conductivity",
        )
