"""Units, dimensions and the vocabulary of standard names.

This module reads a quantity such as "7850 kg/m^3" or "200 GPa", converts it to
the International System of Units (SI), and checks its dimension. It also loads
`standard_names.toml`, the vocabulary of standard names, and looks a name up
with its component suffix.

A dimension is the vector of exponents of the seven SI base units. The module
uses no external library.
"""

import dataclasses
import difflib
import functools
import math
import pathlib
import re
from fractions import Fraction

import tomllib

from ._core import InputError

base_dimension_names = (
    "length",
    "mass",
    "time",
    "electric_current",
    "temperature",
    "amount_of_substance",
    "luminous_intensity",
)
# SI base units in this order: m, kg, s, A, K, mol, cd
base_unit_symbols = ("m", "kg", "s", "A", "K", "mol", "cd")


def fail(path: str, message: str):
    """Raises InputError with the path of the key in front of the message."""
    raise InputError(f"{path}: {message}" if path else message)


@dataclasses.dataclass(frozen=True)
class dimension:
    """The exponents of the seven SI base units, in `base_dimension_names` order."""

    exponents: tuple[Fraction, ...] = (0, 0, 0, 0, 0, 0, 0)

    def __post_init__(self):
        exponents = tuple(Fraction(exponent) for exponent in self.exponents)
        if len(exponents) != len(base_dimension_names):
            raise ValueError(
                f"a dimension has {len(base_dimension_names)} exponents, got {len(exponents)}"
            )
        object.__setattr__(self, "exponents", exponents)

    def __mul__(self, other: "dimension") -> "dimension":
        if not isinstance(other, dimension):
            return NotImplemented
        pairs = zip(self.exponents, other.exponents, strict=True)
        return dimension(tuple(left + right for left, right in pairs))

    def __truediv__(self, other: "dimension") -> "dimension":
        if not isinstance(other, dimension):
            return NotImplemented
        pairs = zip(self.exponents, other.exponents, strict=True)
        return dimension(tuple(left - right for left, right in pairs))

    def __pow__(self, power: int | Fraction) -> "dimension":
        if isinstance(power, bool) or not isinstance(power, (int, Fraction)):
            raise TypeError(f"a dimension takes an int or Fraction power, got {power!r}")
        return dimension(tuple(exponent * power for exponent in self.exponents))

    def is_dimensionless(self) -> bool:
        return not any(self.exponents)

    def __str__(self) -> str:
        """The dimension in SI base units, for example "kg*m^-1*s^-2".

        Positive exponents come first, then negative ones, each in base order.
        The text parses back with `parse_unit`.
        """
        pairs = list(zip(base_unit_symbols, self.exponents, strict=True))
        positive = [(symbol, exponent) for symbol, exponent in pairs if exponent > 0]
        negative = [(symbol, exponent) for symbol, exponent in pairs if exponent < 0]
        factors = [symbol + _exponent_text(exponent) for symbol, exponent in positive + negative]
        return "*".join(factors) if factors else "1"

    def __repr__(self) -> str:
        return f"dimension({str(self)!r})"


def _exponent_text(exponent: Fraction) -> str:
    if exponent == 1:
        return ""
    if exponent.denominator == 1:
        return f"^{exponent.numerator}"
    return f"^({exponent.numerator}/{exponent.denominator})"


dimensionless = dimension()


def base(name: str) -> dimension:
    """The dimension of one base quantity.

    base("length") has the exponents (1, 0, 0, 0, 0, 0, 0).
    """
    if name not in base_dimension_names:
        raise InputError(
            f'unknown base dimension "{name}". '
            f"The base dimensions are {', '.join(base_dimension_names)}."
        )
    return dimension(tuple(1 if base_name == name else 0 for base_name in base_dimension_names))


def _dimension_of(**exponents: int) -> dimension:
    return dimension(tuple(exponents.get(name, 0) for name in base_dimension_names))


# The unit table. Each entry gives the factor to SI, the dimension, and whether
# the unit takes an SI prefix. The SI brochure forbids a prefix on kg, min, h and d.
# The elementary charge is exact. Its value comes from the table of the National
# Institute of Standards and Technology (NIST) of the values that the Committee
# on Data of the International Science Council (CODATA) recommends, accessed
# 2026-11-19.
_elementary_charge_in_coulomb = 1.602176634e-19

_length = _dimension_of(length=1)
_mass = _dimension_of(mass=1)
_time = _dimension_of(time=1)
_current = _dimension_of(electric_current=1)
_newton = _dimension_of(mass=1, length=1, time=-2)
_pascal = _newton / _length**2
_joule = _newton * _length
_watt = _joule / _time
_coulomb = _current * _time
_volt = _watt / _current
_weber = _volt * _time

_unit_table: dict[str, tuple[float, dimension, bool]] = {
    # SI base units
    "m": (1.0, _length, True),
    "kg": (1.0, _mass, False),
    "g": (1.0e-3, _mass, True),
    "s": (1.0, _time, True),
    "A": (1.0, _current, True),
    "K": (1.0, _dimension_of(temperature=1), True),
    "mol": (1.0, _dimension_of(amount_of_substance=1), True),
    "cd": (1.0, _dimension_of(luminous_intensity=1), True),
    # SI derived units with special names
    "rad": (1.0, dimensionless, True),
    "deg": (math.pi / 180.0, dimensionless, False),
    "sr": (1.0, dimensionless, True),
    "Hz": (1.0, _time**-1, True),
    "N": (1.0, _newton, True),
    "Pa": (1.0, _pascal, True),
    "J": (1.0, _joule, True),
    "W": (1.0, _watt, True),
    "C": (1.0, _coulomb, True),
    "V": (1.0, _volt, True),
    "F": (1.0, _coulomb / _volt, True),
    "ohm": (1.0, _volt / _current, True),
    "S": (1.0, _current / _volt, True),
    "Wb": (1.0, _weber, True),
    "T": (1.0, _weber / _length**2, True),
    "H": (1.0, _weber / _current, True),
    "lm": (1.0, _dimension_of(luminous_intensity=1), True),
    "lx": (1.0, _dimension_of(luminous_intensity=1, length=-2), True),
    "Bq": (1.0, _time**-1, True),
    "Gy": (1.0, _joule / _mass, True),
    "Sv": (1.0, _joule / _mass, True),
    "kat": (1.0, _dimension_of(amount_of_substance=1, time=-1), True),
    # Units outside the SI that the input accepts
    "L": (1.0e-3, _length**3, True),
    "min": (60.0, _time, False),
    "h": (3600.0, _time, False),
    "d": (86400.0, _time, False),
    "bar": (1.0e5, _pascal, True),
    "eV": (_elementary_charge_in_coulomb, _joule, True),
}

# SI prefixes from pico to giga, with u for micro.
_prefixes = {
    "p": 1.0e-12,
    "n": 1.0e-9,
    "u": 1.0e-6,
    "m": 1.0e-3,
    "c": 1.0e-2,
    "d": 1.0e-1,
    "da": 1.0e1,
    "h": 1.0e2,
    "k": 1.0e3,
    "M": 1.0e6,
    "G": 1.0e9,
}

# Named units that `SI_unit_text` may lead with. On a tie, the earlier one wins.
named_units = ("W", "J", "Pa", "N", "C", "V", "F", "ohm", "S", "Wb", "T", "H")


# The temperature scales with an offset from the kelvin: the value in K is factor * value + offset.
offset_temperature_units = {"degC": (1.0, 273.15), "degF": (5.0 / 9.0, 273.15 - 32.0 * 5.0 / 9.0)}


def _symbol(symbol: str) -> tuple[float, dimension]:
    """One unit symbol, with or without a prefix.

    A symbol of the table wins over a prefixed reading, so "cd" is the candela and
    "min" the minute. Any other symbol is a prefix on a unit of the table, so "mm"
    is the millimetre.
    """
    if symbol in offset_temperature_units:
        raise InputError(
            f'"{symbol}" has an offset from the kelvin, so it gives a temperature only alone, as in '
            f'"40 {symbol}". Write a temperature difference and a compound unit with K, as in '
            '"W/(m*K)".'
        )
    if symbol in _unit_table:
        factor, unit_dimension, _ = _unit_table[symbol]
        return factor, unit_dimension
    for prefix in sorted(_prefixes, key=len, reverse=True):
        rest = symbol[len(prefix) :]
        if symbol.startswith(prefix) and rest in _unit_table:
            factor, unit_dimension, takes_prefix = _unit_table[rest]
            if not takes_prefix:
                raise InputError(f'the unit "{rest}" takes no prefix, so "{symbol}" is not a unit')
            return _prefixes[prefix] * factor, unit_dimension
    nearest = difflib.get_close_matches(symbol, list(_unit_table), n=3)
    hint = f" The nearest units are {', '.join(nearest)}." if nearest else ""
    raise InputError(f'unknown unit "{symbol}".{hint}')


_token_pattern = re.compile(
    r"\s*(?:(?P<symbol>[A-Za-z]+)|(?P<integer>[0-9]+)|(?P<operator>[*/^()\-]))"
)


def _tokens(text: str) -> list[tuple[str, str, int]]:
    """Splits a unit into (kind, text, position) tokens. Whitespace between tokens is allowed."""
    tokens = []
    position = 0
    while position < len(text):
        if text[position:].isspace():
            break
        match = _token_pattern.match(text, position)
        if match is None:
            start = len(text) - len(text[position:].lstrip())
            raise InputError(
                f'unit "{text}": unexpected character "{text[start]}" at position {start + 1}'
            )
        kind = match.lastgroup
        tokens.append((kind, match.group(kind), match.start(kind)))
        position = match.end()
    return tokens


class _unit_parser:
    """Recursive descent over the unit grammar.

    unit     := factor (("*" | "/") factor)*          left-associative, so "J/mol/K" is J/(mol*K)
    factor   := atom ("^" exponent)?
    atom     := symbol | "1" | "(" unit ")"
    exponent := ["-"] integer | "(" ["-"] integer "/" integer ")"
    """

    def __init__(self, text: str):
        self.text = text
        self.tokens = _tokens(text)
        self.index = 0

    def peek(self) -> str | None:
        return self.tokens[self.index][1] if self.index < len(self.tokens) else None

    def fail(self, expected: str):
        if self.index < len(self.tokens):
            _, found, position = self.tokens[self.index]
            raise InputError(
                f'unit "{self.text}": expected {expected} '
                f'at position {position + 1}, found "{found}"'
            )
        raise InputError(f'unit "{self.text}": expected {expected} at the end')

    def take(self, kind: str, text: str | None = None) -> str:
        if self.index < len(self.tokens):
            token_kind, token_text, _ = self.tokens[self.index]
            if token_kind == kind and (text is None or token_text == text):
                self.index += 1
                return token_text
        self.fail(f'"{text}"' if text else ("an integer" if kind == "integer" else f"a {kind}"))

    def parse(self) -> tuple[float, dimension]:
        factor, unit_dimension = self.unit()
        if self.index < len(self.tokens):
            self.fail('"*", "/" or the end')
        return factor, unit_dimension

    def unit(self) -> tuple[float, dimension]:
        factor, unit_dimension = self.factor()
        while self.peek() in ("*", "/"):
            operator = self.take("operator")
            right_factor, right_dimension = self.factor()
            if operator == "*":
                factor, unit_dimension = factor * right_factor, unit_dimension * right_dimension
            else:
                factor, unit_dimension = factor / right_factor, unit_dimension / right_dimension
        return factor, unit_dimension

    def factor(self) -> tuple[float, dimension]:
        factor, unit_dimension = self.atom()
        if self.peek() == "^":
            self.take("operator", "^")
            power = self.exponent()
            factor, unit_dimension = factor ** float(power), unit_dimension**power
        return factor, unit_dimension

    def atom(self) -> tuple[float, dimension]:
        if self.index >= len(self.tokens):
            self.fail("a unit")
        kind, text, _ = self.tokens[self.index]
        if kind == "symbol":
            self.index += 1
            return _symbol(text)
        if kind == "integer" and text == "1":
            self.index += 1
            return 1.0, dimensionless
        if text == "(":
            self.index += 1
            result = self.unit()
            self.take("operator", ")")
            return result
        self.fail('a unit symbol, "1" or "("')

    def exponent(self) -> Fraction:
        if self.peek() == "(":
            self.take("operator", "(")
            numerator = self.signed_integer()
            self.take("operator", "/")
            denominator = int(self.take("integer"))
            self.take("operator", ")")
            if denominator == 0:
                raise InputError(f'unit "{self.text}": an exponent has the denominator 0')
            return Fraction(numerator, denominator)
        return Fraction(self.signed_integer())

    def signed_integer(self) -> int:
        sign = -1 if self.peek() == "-" else 1
        if sign == -1:
            self.take("operator", "-")
        return sign * int(self.take("integer"))


def parse_unit(text: str) -> tuple[float, dimension]:
    """The factor to SI and the dimension of a unit.

    parse_unit("GPa") gives 1.0e9 and the dimension of pressure.
    """
    if not isinstance(text, str) or not text.strip():
        raise InputError(f"a unit is a non-empty string, got {text!r}")
    return _unit_parser(text).parse()


def _split_quantity(text: str) -> tuple[float, str | None]:
    """Splits "7850 kg/m^3" into the number 7850.0 and the unit "kg/m^3".

    A number alone has the unit None.
    """
    parts = text.strip().split(None, 1)
    if not parts:
        raise InputError("a quantity is a number with a unit, got an empty string")
    try:
        number = float(parts[0])
    except ValueError:
        raise InputError(
            f'"{text}" is not a number followed by a unit. '
            "Write the number first and separate it from the unit "
            'with a space, as in "7850 kg/m^3".'
        ) from None
    if not math.isfinite(number):
        raise InputError(f'"{text}" holds no finite number')
    return number, (parts[1] if len(parts) == 2 else None)


def parse_quantity(text: str) -> tuple[float, dimension]:
    """The value in SI and the dimension of a quantity.

    "7850 kg/m^3" gives 7850.0 and the dimension of density. "0.3" gives 0.3
    and dimensionless.
    """
    if not isinstance(text, str):
        raise InputError(f"a quantity is a string, got {text!r}")
    number, unit = _split_quantity(text)
    if unit is None:
        return number, dimensionless
    if unit.strip() in offset_temperature_units:
        factor, offset = offset_temperature_units[unit.strip()]
        return factor * number + offset, _dimension_of(temperature=1)
    factor, unit_dimension = parse_unit(unit)
    return number * factor, unit_dimension


def SI_unit_text(expected: dimension) -> str:
    """The coherent SI unit of a dimension, as a text that `parse_unit` reads back with the factor 1.

    It is the unit of the first standard name of this dimension, so the dimension
    of thermal conductivity gives "W/(m*K)" and a dimensionless one "1". Without
    such a name, the text leads with the named unit that leaves the fewest base
    unit factors, or with none when no named unit saves a factor.
    """
    for entry in _load().values():
        if entry.physical_dimension == expected and parse_unit(entry.unit)[0] == 1.0:
            return entry.unit

    def factors(remainder: dimension) -> Fraction:
        return sum(abs(exponent) for exponent in remainder.exponents)

    best_named, best_cost = None, factors(expected)
    for symbol in named_units:
        cost = 1 + factors(expected / _unit_table[symbol][1])
        if cost < best_cost:
            best_named, best_cost = symbol, cost
    remainder = expected / _unit_table[best_named][1] if best_named else expected
    pairs = list(zip(base_unit_symbols, remainder.exponents, strict=True))
    numerator = ([best_named] if best_named else []) + [
        symbol + _exponent_text(exponent) for symbol, exponent in pairs if exponent > 0
    ]
    denominator = [symbol + _exponent_text(-exponent) for symbol, exponent in pairs if exponent < 0]
    text = "*".join(numerator) or "1"
    if len(denominator) == 1:
        return f"{text}/{denominator[0]}"
    if denominator:
        return f"{text}/({'*'.join(denominator)})"
    return text


def to_SI(
    value: str | int | float, expected: dimension, key: str, expected_unit: str | None = None
) -> float:
    """Converts value to SI and checks its dimension against expected.

    A bare number, or a string that holds only a number, passes only when
    expected is dimensionless. Every error message starts with key, for
    example "properties.Young_modulus.cladding". The messages show
    expected_unit, for example the unit "J/(kg*K)" of a standard name. Without
    expected_unit, they show the SI unit of expected.
    """
    if expected_unit is None:
        expected_unit = SI_unit_text(expected)
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise InputError(f"{key}: expected a number or a quantity string, got {value!r}")
    if isinstance(value, str):
        try:
            number, unit = _split_quantity(value)
            if unit is not None and unit.strip() in offset_temperature_units:
                # A temperature on a scale with an offset converts to K as a whole quantity.
                number, given = parse_quantity(value)
                factor = 1.0
            else:
                factor, given = parse_unit(unit) if unit is not None else (1.0, dimensionless)
        except InputError as error:
            raise InputError(f"{key}: {error}") from None
        shown_value = value.strip()
    else:
        number, unit, factor, given = float(value), None, 1.0, dimensionless
        if not math.isfinite(number):
            raise InputError(f"{key}: {value!r} is not a finite number")
        shown_value = repr(value)
    if given == expected:
        return number * factor
    if unit is None:
        raise InputError(
            f'{key}: "{shown_value}" has no unit. '
            f"It needs a unit of dimension {expected}, "
            f'for example "{shown_value} {expected_unit}".'
        )
    if expected.is_dimensionless():
        raise InputError(
            f'{key}: "{shown_value}" has the unit "{unit}" of dimension {given}, '
            "but the value is dimensionless and takes no unit."
        )
    raise InputError(
        f'{key}: "{shown_value}" has the unit "{unit}" of dimension {given}, '
        f'but it needs the unit "{expected_unit}" '
        f"or another unit of dimension {expected}."
    )


# The vocabulary of standard names

shapes = ("scalar", "vector", "tensor", "scalar_or_tensor")
# The names of the position coordinates of each coordinate system, in the order of the mesh
# coordinates. Their letters are also the component suffixes, so displacement_z is component
# (2,) in cartesian and (1,) in axisymmetric coordinates.
position_names = {"cartesian": ("x", "y", "z"), "axisymmetric": ("r", "z")}
_component_length = {"vector": 1, "tensor": 2, "scalar_or_tensor": 2}
standard_names_file = pathlib.Path(__file__).with_name("standard_names.toml")


@dataclasses.dataclass(frozen=True)
class standard_name:
    """One entry of the vocabulary. value is in SI and set for a physical constant only."""

    name: str
    unit: str
    physical_dimension: dimension
    shape: str
    description: str
    value: float | None
    ISO_name: str | None


def _entry(name: str, table: dict) -> standard_name:
    where = f"{standard_names_file.name}, [{name}]"
    allowed_keys = {"unit", "shape", "description", "value", "ISO_name"}
    unknown_keys = set(table) - allowed_keys
    if unknown_keys:
        raise InputError(f"{where}: unknown keys {sorted(unknown_keys)}")
    for required in ("unit", "shape", "description"):
        if not isinstance(table.get(required), str) or not table[required]:
            raise InputError(f"{where}: {required} is a required non-empty string")
    if table["shape"] not in shapes:
        raise InputError(f'{where}: shape "{table["shape"]}" is not one of {", ".join(shapes)}')
    try:
        _, unit_dimension = parse_unit(table["unit"])
    except InputError as error:
        raise InputError(f"{where}: {error}") from None
    ISO_name = table.get("ISO_name")
    if ISO_name is not None and (not isinstance(ISO_name, str) or not ISO_name):
        raise InputError(f"{where}: ISO_name is a non-empty string when present")
    if ISO_name == name.replace("_", " "):
        raise InputError(
            f'{where}: ISO_name "{ISO_name}" equals the standard name. '
            "Give ISO_name only where the ISO 80000 name differs"
        )
    value = None
    if "value" in table:
        value = to_SI(table["value"], unit_dimension, where, expected_unit=table["unit"])
    return standard_name(
        name=name,
        unit=table["unit"],
        physical_dimension=unit_dimension,
        shape=table["shape"],
        description=table["description"],
        value=value,
        ISO_name=ISO_name,
    )


@functools.cache
def _load() -> dict[str, standard_name]:
    with open(standard_names_file, "rb") as file:
        tables = tomllib.load(file)
    entries = {}
    for name, table in tables.items():
        if not isinstance(table, dict):
            raise InputError(f"{standard_names_file.name}, {name}: each name is a table")
        entries[name] = _entry(name, table)
    return entries


def standard_names() -> dict[str, standard_name]:
    """Every entry of the vocabulary by name. The file is read once."""
    return dict(_load())


def _letters(coordinate_system: str) -> tuple[str, ...]:
    if coordinate_system not in position_names:
        raise InputError(
            f'unknown coordinate system "{coordinate_system}". '
            f"The coordinate systems are {', '.join(position_names)}."
        )
    return position_names[coordinate_system]


def component_name(
    name: str, component: tuple[int, ...], coordinate_system: str = "cartesian"
) -> str:
    """The name of one component.

    component_name("stress", (0, 1)) gives "stress_xy", and component_name("displacement",
    (1,), "axisymmetric") gives "displacement_z".
    """
    letters = _letters(coordinate_system)
    return name + ("_" + "".join(letters[index] for index in component) if component else "")


def split_component(
    name: str, coordinate_system: str = "cartesian"
) -> tuple[str, tuple[int, ...]] | None:
    """The base name and the component of a name with a component suffix.

    split_component("displacement_x") gives ("displacement", (0,)), split_component("stress_xy")
    gives ("stress", (0, 1)) and split_component("displacement_z", "axisymmetric") gives
    ("displacement", (1,)). A name whose last word holds a letter outside the position names
    of the coordinate system gives None.
    """
    letters = _letters(coordinate_system)
    base_name, _, suffix = name.rpartition("_")
    if base_name and suffix and all(letter in letters for letter in suffix):
        return base_name, tuple(letters.index(letter) for letter in suffix)
    return None


def lookup(
    name: str, coordinate_system: str = "cartesian"
) -> tuple[standard_name, tuple[int, ...]]:
    """The entry of a standard name and the component that its suffix selects.

    lookup("temperature") gives (entry, ()), lookup("displacement_x") gives (entry of
    displacement, (0,)) and lookup("stress_xy") gives (entry of stress, (0, 1)). The suffix
    letters are the position names of the coordinate system. An unknown name raises
    InputError that lists the three nearest names.
    """
    if not isinstance(name, str):
        raise InputError(f"a standard name is a string, got {name!r}")
    names = _load()
    if name in names:
        return names[name], ()
    base_name, component = split_component(name, coordinate_system) or (None, ())
    if base_name in names:
        entry = names[base_name]
        if _component_length.get(entry.shape) != len(component):
            letters = position_names[coordinate_system]
            if entry.shape == "vector":
                allowed = "the components " + ", ".join(f"_{letter}" for letter in letters)
            elif entry.shape == "scalar":
                allowed = "no components"
            else:
                allowed = (
                    f"components such as _{letters[0]}{letters[0]} and _{letters[0]}{letters[1]}"
                )
            raise InputError(
                f'"{name}": {base_name} has the shape {entry.shape} and takes {allowed}'
            )
        return entry, component
    nearest = difflib.get_close_matches(name, list(names), n=3, cutoff=0.0)
    raise InputError(f'unknown standard name "{name}". The nearest names are {", ".join(nearest)}.')
