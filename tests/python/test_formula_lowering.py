# SPDX-License-Identifier: LGPL-2.1-or-later
"""A formula of the input reaches the engine with its quantities in SI, its fields by the
engine's names, and its time and position as the engine reads them; a formula of the wrong
dimension, or with a name the vocabulary does not hold, is refused with its key."""

import math

import pytest
from dualmesh._core import InputError, ParsedFunction
from dualmesh.model.dimensions import dimension_error
from dualmesh.model.engine_form import engine_formula


def value_at(text, unit, x=0.0, y=0.0, z=0.0, t=0.0):
    return ParsedFunction(engine_formula(text, unit, "properties.test"))(x, y, z, t)


def test_the_quantities_of_a_formula_reach_the_engine_in_SI():
    """A ramp of 300 MW/m^3 over 100 s is 1.5e8 W/m^3 at half its time."""
    formula = "[300 MW/m^3]*min(time/[100 s], 1)"
    assert value_at(formula, "W/m^3", t=50.0) == pytest.approx(1.5e8, rel=1e-14)
    assert value_at(formula, "W/m^3", t=500.0) == pytest.approx(3.0e8, rel=1e-14)


def test_the_position_and_offset_temperatures_reach_the_engine():
    formula = "[40 degC] + [2 K/mm]*x - [1 K/m]*z"
    assert value_at(formula, "K", x=0.5, z=2.0) == pytest.approx(313.15 + 1000.0 - 2.0, rel=1e-14)


def test_a_conditional_and_the_functions_keep_their_meaning():
    formula = "conditional(less(x, [1 m]), exp(-x/[1 m]), sqrt(x/[1 m]))"
    assert value_at(formula, "1", x=0.5) == pytest.approx(math.exp(-0.5), rel=1e-14)
    assert value_at(formula, "1", x=4.0) == pytest.approx(2.0, rel=1e-14)


def test_the_fields_take_the_names_of_the_engine():
    text = engine_formula(
        "[200 GPa]*(1 - [1e-4 1/m]*displacement_x)", "Pa", "properties.Young_modulus"
    )
    assert "disp_x" in text and "displacement_x" not in text
    text = engine_formula(
        "1/([0.0375 m*K/W] + [2.165e-4 m/W]*temperature)", "W/(m*K)", "properties.k"
    )
    assert "temperature" in text


def test_a_formula_of_the_wrong_dimension_is_refused_with_its_key():
    with pytest.raises(dimension_error, match=r"^properties\.thermal_conductivity: .*W/\(m\*K\)"):
        engine_formula("[2 m]*temperature", "W/(m*K)", "properties.thermal_conductivity")


def test_an_unknown_name_is_refused_with_its_key():
    with pytest.raises(InputError, match=r"^properties\.density: .*unknown name 'temprature'"):
        engine_formula("[7850 kg/m^3]*(1 - [1e-5 1/K]*temprature)", "kg/m^3", "properties.density")
