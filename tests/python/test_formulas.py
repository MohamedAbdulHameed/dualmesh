# SPDX-License-Identifier: LGPL-2.1-or-later
"""The formulas of the input: the parser keeps the precedence of the grammar, reads the names
of the vocabulary, the position and the time, and every operation checks the dimensions of its
operands."""

from fractions import Fraction

import pytest
from dualmesh._core import InputError
from dualmesh.model import parse, syntax_error
from dualmesh.model.dimensions import dimension_error
from dualmesh.model.graph import (
    conditional,
    exp,
    field,
    less,
    maximum,
    number,
    position,
    sin,
    sqrt,
    time,
)
from dualmesh.units import base, dimensionless

kelvin = base("temperature")
metre = base("length")
second = base("time")


@pytest.mark.parametrize(
    "text, expected",
    [
        ("2+3*4", 14.0),
        ("(2+3)*4", 20.0),
        ("2^3^2", 512.0),
        ("-2^2", -4.0),
        ("2^-1", 0.5),
        ("8/4/2", 1.0),
        ("2-3-4", -5.0),
        ("2*-3", -6.0),
    ],
)
def test_the_precedence_of_the_grammar(text, expected):
    """A power is right associative and binds tighter than a unary minus; a division is left
    associative."""
    assert parse(text).value == pytest.approx(expected, rel=1e-15)


def test_the_names_of_the_vocabulary_the_position_and_the_time():
    assert parse("x") is position()[0]
    assert parse("r", coordinate_system="axisymmetric") is position()[0]
    assert parse("z", coordinate_system="axisymmetric") is position()[1]
    assert parse("displacement_x") is field("displacement")[0]
    assert (
        parse("[300 K] + [2 K/s]*time")
        is number(300.0, kelvin) + number(2.0, kelvin / second) * time()
    )
    with pytest.raises(syntax_error, match="unknown name 't'"):
        parse("[2 K/s]*t")
    with pytest.raises(syntax_error, match="unknown name 'x'"):
        parse("x", coordinate_system="axisymmetric")


def test_the_dimensions_of_the_operations():
    temperature = field("temperature")
    assert sqrt(temperature).dimension == kelvin ** Fraction(1, 2)
    assert (temperature ** (1.0 / 3.0)).dimension == kelvin ** Fraction(1, 3)
    assert less(temperature, number(300.0, kelvin)).dimension == dimensionless
    assert maximum(temperature, 0).dimension == kelvin
    assert conditional(less(temperature, 0), temperature, 0).dimension == kelvin


def test_an_operation_on_operands_of_unequal_dimensions_is_refused():
    temperature = field("temperature")
    with pytest.raises(dimension_error, match="add needs operands of equal dimension"):
        temperature + number(1.0, metre)
    with pytest.raises(dimension_error, match="exp needs a dimensionless argument"):
        exp(temperature)
    with pytest.raises(dimension_error, match="sin needs a dimensionless argument"):
        sin(position()[0])
    with pytest.raises(dimension_error, match="denominator of at most 12"):
        temperature**0.37
    with pytest.raises(dimension_error, match="maximum"):
        maximum(temperature, number(1.0, metre))


@pytest.mark.parametrize(
    "text, message",
    [
        ("temprature + 1", "unknown name 'temprature'.*temperature"),
        ("2*", "column 3"),
        ("2 temperature", "Write every product with"),
        ("expo(1)", "unknown function 'expo'.*exp"),
        ("max(1)", "max takes 2 arguments, got 1"),
        ("(1 + 2", "expected '\\)'"),
        ("   ", "empty"),
    ],
)
def test_a_malformed_formula_is_refused_with_its_column_or_a_suggestion(text, message):
    with pytest.raises(InputError, match=message):
        parse(text)
