"""The one schema of the input tree.

`validate` checks every key of a tree, names the nearest valid key for a
misspelled one, converts every quantity to SI with `units.to_SI`, checks the
dimension of every formula and table, reads the CSV tables and fills the
defaults. It returns a new tree in `section_order`.

Each part of the schema is a check function `check(value, path, tree_context)`
that returns the validated value. `path` names the key in error messages, for
example "properties.thermal_conductivity.fuel". `tree_context` gives access to
the whole tree, for the names that one section takes from another.
"""

import copy
import difflib
import inspect
import itertools
import os
import re
from pathlib import Path

from dualmesh.input.physics_options import option_choices, physics_functions
from dualmesh.input.read import read_table
from dualmesh.input.units import (
    SI_unit_text,
    dimensionless,
    input_error,
    lookup,
    parse_unit,
    position_names,
    standard_names,
    to_SI,
)
from dualmesh.model import field, parse

section_order = (
    "mesh",
    "method",
    "physics",
    "properties",
    "gravity",
    "models",
    "initial_conditions",
    "boundary_conditions",
    "time",
    "mesh_adaptivity",
    "postprocessors",
    "output",
    "solver",
)
# Without a method, a problem with a flow of the equal-order formulation takes DMCDM, any other FEM.
methods = ("DMCDM", "FEM")
mesh_forms = ("segment", "rectangle", "box", "file")
generated_dimensions = {"segment": 1, "rectangle": 2, "box": 3}
generated_boundaries = {
    "segment": ("left", "right"),
    "rectangle": ("left", "right", "bottom", "top"),
    "box": ("left", "right", "bottom", "top", "back", "front"),
}
generated_region = "domain"
default_cell_types = {"rectangle": "quadrilateral", "box": "hexahedron"}
# The step controllers of the engine: the error of the step, the Newton iterations it takes,
# the Courant number of a flow, or a fixed step.
step_controls = ("error", "iterations", "Courant", "fixed")
# The Zienkiewicz-Zhu indicator, from the jump between the gradient and its recovery.
indicators = ("gradient_recovery",)
reductions = ("integral", "average", "maximum", "minimum", "point_value", "boundary_flux")
region_reductions = ("integral", "average", "maximum", "minimum")
model_keys = (
    "unknown",
    "expressions",
    "states",
    "storage",
    "inertia",
    "flux",
    "source",
    "boundary_fluxes",
    "flux_name",
    "method",
)
# Names that a formula reads as the position, the time or the cell size, so no property takes them.
reserved_names = {*itertools.chain(*position_names.values()), "time", "time_step", "cell_size"}
ranks_of_shape = {"scalar": (0,), "vector": (1,), "tensor": (2,), "scalar_or_tensor": (0, 2)}
fixed_value_message = "a fixed value takes a constant. A formula or a table is not available"
declaration_example = "{unit: mol/m^3, value: 10 mol/m^3}"

required = object()  # Marks a key without a default that must be present
optional = object()  # Marks a key without a default that may be absent


# Messages


def fail(path: str, message: str):
    """Raises input_error with the path of the key in front of the message."""
    raise input_error(f"{path}: {message}" if path else message)


def suggestion(word: str, choices) -> str:
    """Names the nearest valid choice of a misspelled word, or lists all choices."""
    choices = sorted(choices)
    if not choices:
        return ""
    nearest = difflib.get_close_matches(word, choices, n=1)
    if nearest:
        return f" The nearest valid name is '{nearest[0]}'."
    return " The valid names are " + ", ".join(f"'{choice}'" for choice in choices) + "."


def join(path: str, key) -> str:
    return f"{path}.{key}" if path else str(key)


# Context: what one section needs to know about the others


def _mapping_or_empty(value) -> dict:
    return value if isinstance(value, dict) else {}


name_pattern = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


class context:
    """The raw tree under validation, the folder of its files and the names it declares.

    `declared` maps each name outside the vocabulary that the tree declares with a
    unit to its dimension, its shape and the factor of its unit to SI: a property
    in the form {unit: ..., value: ...}, and the unknown, the states and the
    expressions of a model. `symbols` maps the same names to field nodes of that
    dimension and rank, so that formulas that read them parse and check. Unreadable
    declarations are left out here and reported by the check of their own section.
    """

    def __init__(self, tree: dict, directory: str | Path = "."):
        self.tree = _mapping_or_empty(tree)
        self.directory = Path(directory)
        mesh = _mapping_or_empty(self.tree.get("mesh"))
        coordinate_system = mesh.get("coordinate_system", "cartesian")
        self.coordinate_system = (
            coordinate_system if coordinate_system in position_names else "cartesian"
        )
        forms = [form for form in mesh_forms if form in mesh]
        self.mesh_form = forms[0] if len(forms) == 1 else None
        self.declared = {}
        for name, entry in _mapping_or_empty(self.tree.get("properties")).items():
            if isinstance(entry, dict) and "unit" in entry:
                self._declare(name, entry["unit"], "scalar")
        for entry in _mapping_or_empty(self.tree.get("models")).values():
            entry = _mapping_or_empty(entry)
            for section in ("unknown", "states", "expressions"):
                for name, declaration in _mapping_or_empty(entry.get(section)).items():
                    declaration = _mapping_or_empty(declaration)
                    self._declare(name, declaration.get("unit"), declaration.get("shape", "scalar"))
        self.symbols = {
            name: field(name, unit=SI_unit_text(physical_dimension), shape=shape)
            for name, (physical_dimension, shape, _) in self.declared.items()
        }

    def _declare(self, name, unit, shape) -> None:
        if (
            not isinstance(name, str)
            or name_pattern.fullmatch(name) is None
            or name in standard_names()
            or name in reserved_names
            or not isinstance(unit, str)
            or shape not in ("scalar", "vector", "tensor")
        ):
            return
        try:
            factor, physical_dimension = parse_unit(unit)
        except input_error:
            return
        self.declared[name] = (physical_dimension, shape, factor)

    def boundary_names(self) -> tuple[str, ...] | None:
        """The boundaries of a generated mesh, or None for a mesh file."""
        return generated_boundaries.get(self.mesh_form)

    def region_names(self) -> tuple[str, ...] | None:
        """The one region of a generated mesh, or None for a mesh file."""
        return (generated_region,) if self.mesh_form in generated_dimensions else None

    def space_dimension(self) -> int | None:
        return generated_dimensions.get(self.mesh_form)

    def number_factor(self, name: str) -> float | None:
        """The factor to SI of the declared unit of name, which a bare number of name is in, or None."""
        return self.declared[name][2] if name in self.declared else None

    def flux_names(self) -> list[str]:
        """The names of the fluxes that a boundary flux postprocessor can name.

        These are the vector standard names and the flux names of the models.
        """
        names = {name for name, entry in standard_names().items() if entry.shape == "vector"}
        for entry in _mapping_or_empty(self.tree.get("models")).values():
            flux_name = _mapping_or_empty(entry).get("flux_name")
            if isinstance(flux_name, str):
                names.add(flux_name)
        return sorted(names)

    def quantity_of(self, name: str, path: str) -> tuple:
        """The dimension, the unit text and the ranks of a standard name, a component of one or a declared name."""
        if name in self.declared:
            physical_dimension, shape, _ = self.declared[name]
            return physical_dimension, SI_unit_text(physical_dimension), ranks_of_shape[shape]
        try:
            entry, component = lookup(name, self.coordinate_system)
        except input_error as error:
            fail(
                path,
                f"{error} A name outside the vocabulary declares its unit once in properties, "
                f"as in {name}: {declaration_example}.",
            )
        if entry.value is not None:
            fail(
                path,
                f"'{name}' is a physical constant with a fixed value, so the input cannot set it.",
            )
        return (
            entry.physical_dimension,
            entry.unit,
            (0,) if component else ranks_of_shape[entry.shape],
        )

    def path_of(self, file) -> str:
        """The absolute path of a file named relative to the folder of the input."""
        return os.path.abspath(os.path.join(self.directory, file))


# Check functions


def mapping(value, path: str) -> dict:
    """Checks that value is a mapping with text keys."""
    if not isinstance(value, dict):
        fail(path, f"expects a mapping of keys to values, got {value!r}.")
    for key in value:
        if not isinstance(key, str):
            fail(path, f"expects text keys, got the key {key!r}.")
    return value


def quantity(unit: str, positive: bool = False):
    """A value with the dimension of unit, converted to SI."""
    expected = parse_unit(unit)[1]

    def check(value, path, tree_context):
        result = to_SI(value, expected, path)
        if positive and not result > 0.0:
            fail(path, f"expects a positive value, got {value!r}.")
        return result

    return check


def number(minimum: float | None = None, maximum: float | None = None, positive: bool = False):
    """A dimensionless number. A numeric text passes too."""
    bounds = [
        *(["greater than 0"] if positive else []),
        *([f"at least {minimum}"] if minimum is not None else []),
        *([f"at most {maximum}"] if maximum is not None else []),
    ]

    def check(value, path, tree_context):
        result = to_SI(value, dimensionless, path)
        if (
            (positive and not result > 0.0)
            or (minimum is not None and result < minimum)
            or (maximum is not None and result > maximum)
        ):
            fail(path, f"expects a number {' and '.join(bounds)}, got {value!r}.")
        return result

    return check


def integer(minimum: int):
    def check(value, path, tree_context):
        if isinstance(value, bool) or not isinstance(value, int):
            fail(path, f"expects an integer, got {value!r}.")
        if value < minimum:
            fail(path, f"expects an integer of at least {minimum}, got {value}.")
        return value

    return check


def text(choices=None):
    """A text, one of choices when choices is not None."""

    def check(value, path, tree_context):
        if not isinstance(value, str):
            fail(path, f"expects a text, got {value!r}.")
        if choices is not None and value not in choices:
            fail(path, f"unknown value '{value}'." + suggestion(value, choices))
        return value

    return check


def sequence(item, length: int | None = None):
    def check(value, path, tree_context):
        if not isinstance(value, (list, tuple)):
            fail(path, f"expects a list, got {value!r}.")
        if length is not None and len(value) != length:
            fail(path, f"expects a list of {length} entries, got {len(value)}.")
        return [item(entry, f"{path}[{index}]", tree_context) for index, entry in enumerate(value)]

    return check


def table(keys: dict):
    """A mapping with fixed keys. keys maps each key to (check, default)."""

    def check(value, path, tree_context):
        given = mapping(value, path)
        for key in given:
            if key not in keys:
                fail(path, f"unknown key '{key}'." + suggestion(key, keys))
        result = {}
        for key, (check_key, default) in keys.items():
            if key in given:
                result[key] = check_key(given[key], join(path, key), tree_context)
            elif default is required:
                fail(path, f"the key '{key}' is missing.")
            elif default is not optional:
                result[key] = copy.deepcopy(default)
        return result

    return check


def named(item):
    """A mapping from names that the user chooses, such as postprocessor names, to entries."""

    def check(value, path, tree_context):
        return {
            name: item(entry, join(path, name), tree_context)
            for name, entry in mapping(value, path).items()
        }

    return check


def field_name(value, path, tree_context):
    """The name of a field, a component of one or another standard or declared name."""
    text()(value, path, tree_context)
    tree_context.quantity_of(value, path)
    return value


def anything(value, path, tree_context):
    return value


# Property values: a constant, a formula, an inline table or a CSV table

unit_pattern = re.compile(r"[A-Za-z1(][A-Za-z0-9*/^()\s-]*")


def is_quantity_text(value: str) -> bool:
    """True when value reads as a number with an optional unit, as "45 W/(m*K)" and "200 Gpa" do.

    The unit starts with a letter, "1" or "(", holds only the characters of a unit and
    names no standard name. So "2 temperature" and "2 * densty" read as formulas,
    and "200 Gpa" as a quantity whose unit is misspelled.
    """
    parts = value.split(None, 1)
    try:
        float(parts[0])
    except (ValueError, IndexError):
        return False
    if len(parts) == 1:
        return True
    return unit_pattern.fullmatch(parts[1]) is not None and not any(
        name in standard_names() for name in name_pattern.findall(parts[1])
    )


def is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def check_formula(
    formula: str, name: str, expected, unit: str, ranks, path: str, tree_context
) -> str:
    """Parses a formula and checks its dimension and rank against those of name. Returns the text."""
    try:
        node = parse(formula, tree_context.symbols, tree_context.coordinate_system)
    except input_error as error:
        fail(path, str(error))
    if node.dimension != expected:
        fail(
            path,
            f"the formula gives the unit {SI_unit_text(node.dimension)}, but {name} needs the unit "
            f"{unit} or another unit of dimension {expected}.",
        )
    if node.rank not in ranks:
        fail(
            path,
            f"the formula has rank {node.rank}, but {name} takes rank "
            + " or ".join(map(str, ranks))
            + ".",
        )
    return formula


def table_columns(value, path, tree_context) -> tuple[dict, str]:
    """The columns of a table, from a CSV file or inline, and the path that names them in messages."""
    given = mapping(value, path)
    if "file" in given:
        if len(given) != 1:
            fail(path, "a table from a file holds the key 'file' alone.")
        file_path = join(path, "file")
        file = text()(given["file"], file_path, tree_context)
        try:
            return read_table(tree_context.path_of(file)), file_path
        except input_error as error:
            fail(file_path, str(error))
    column = table({"unit": (text(), "1"), "values": (sequence(numeric_value), required)})
    return {
        name: column(entry, join(path, name), tree_context) for name, entry in given.items()
    }, path


def numeric_value(value, path, tree_context) -> float:
    if not is_number(value):
        fail(path, f"expects a number, got {value!r}.")
    return to_SI(value, dimensionless, path)


def table_value(value, name: str, expected, path: str, tree_context) -> dict:
    """A table of 1 to 3 inputs in the validated form, with SI values on a full grid.

    The column `name` is the output. Every other column is an input whose name parses
    as a scalar, such as temperature, time or r. Each input axis comes sorted, and
    the output values come row-major over the inputs in column order.
    """
    columns, columns_path = table_columns(value, path, tree_context)
    if name not in columns:
        fail(
            columns_path,
            f"has no column '{name}'. The output column of a table has the name of its quantity.",
        )
    inputs = [column_name for column_name in columns if column_name != name]
    if not 1 <= len(inputs) <= 3:
        fail(columns_path, f"expects 1 to 3 input columns besides '{name}', got {len(inputs)}.")
    lengths = {len(column["values"]) for column in columns.values()}
    if len(lengths) != 1:
        fail(
            columns_path,
            "expects the same number of values in every column, got "
            + ", ".join(
                f"{len(column['values'])} for '{column_name}'"
                for column_name, column in columns.items()
            )
            + ".",
        )
    converted = {}
    for column_name, column in columns.items():
        column_path = join(columns_path, column_name)
        if column_name == name:
            column_dimension = expected
        else:
            try:
                node = parse(column_name, tree_context.symbols, tree_context.coordinate_system)
            except input_error as error:
                fail(column_path, str(error))
            if node.rank != 0:
                fail(
                    column_path,
                    f"an input of a table is a scalar, but '{column_name}' has rank {node.rank}.",
                )
            column_dimension = node.dimension
        try:
            factor, unit_dimension = parse_unit(column["unit"])
        except input_error as error:
            fail(column_path, str(error))
        if unit_dimension != column_dimension:
            fail(
                column_path,
                f"the unit '{column['unit']}' has the dimension {unit_dimension}, but '{column_name}' "
                f"needs the unit {SI_unit_text(column_dimension)} or another unit of dimension "
                f"{column_dimension}.",
            )
        converted[column_name] = [entry * factor for entry in column["values"]]
    axes = [sorted(set(converted[input_name])) for input_name in inputs]
    for input_name, axis in zip(inputs, axes, strict=True):
        if len(axis) < 2:
            fail(
                join(columns_path, input_name),
                f"expects at least 2 distinct values, got {len(axis)}.",
            )
    grid = {}
    for row, point in enumerate(
        zip(*(converted[input_name] for input_name in inputs), strict=True), start=1
    ):
        if point in grid:
            fail(
                columns_path,
                f"row {row} repeats the inputs of an earlier row, "
                + point_text(inputs, point)
                + ".",
            )
        grid[point] = converted[name][row - 1]
    for point in itertools.product(*axes):
        if point not in grid:
            fail(
                columns_path,
                "the rows do not fill a full grid of the inputs. No row holds "
                + point_text(inputs, point)
                + ".",
            )
    return {
        "inputs": [
            {"name": input_name, "values": axis}
            for input_name, axis in zip(inputs, axes, strict=True)
        ],
        "values": [grid[point] for point in itertools.product(*axes)],
    }


def point_text(inputs: list, point: tuple) -> str:
    return ", ".join(
        f"{input_name} = {coordinate:g}"
        for input_name, coordinate in zip(inputs, point, strict=True)
    )


def property_value(value, name: str, path: str, tree_context, expected_quantity: tuple):
    """One value of the four forms, for name of expected_quantity, a tuple (dimension, unit text, ranks).

    A text is a quantity or a formula. A constant has rank 0, so a name of another
    rank takes a formula or a table.
    """
    expected, unit, ranks = expected_quantity
    if isinstance(value, dict):
        if "table" not in value:
            fail(
                path,
                f"expects a number, a quantity, a formula or a table, got the mapping {value!r}.",
            )
        if len(value) != 1:
            fail(
                path,
                "a mapping with the key 'table' is a table and holds no other key. "
                "The region names table, unit and value are invalid.",
            )
        return {
            "table": table_value(value["table"], name, expected, join(path, "table"), tree_context)
        }
    if isinstance(value, str) and not is_quantity_text(value):
        return check_formula(value, name, expected, unit, ranks, path, tree_context)
    if isinstance(value, (list, tuple)):
        fail(path, f"expects one value, got the list {value!r}.")
    if 0 not in ranks:
        fail(
            path,
            f"{name} takes rank "
            + " or ".join(map(str, ranks))
            + ", so it takes a formula of that rank. "
            "A constant has rank 0.",
        )
    return constant(value, name, path, tree_context, expected, unit)


def constant(value, name: str, path: str, tree_context, expected, unit: str) -> float:
    """A number or a quantity text of name, in SI.

    A bare number of a declared name is in its declared unit. Any other bare number
    is dimensionless.
    """
    factor = tree_context.number_factor(name)
    if factor is not None and is_number(value):
        return to_SI(value, dimensionless, path) * factor
    return to_SI(value, expected, path, expected_unit=unit)


def fixed_value(value, name: str, path: str, tree_context):
    """A constant for a scalar, or a list of one constant per component for a vector, in the unit of name."""
    expected, unit, ranks = tree_context.quantity_of(name, path)
    entries = value if isinstance(value, (list, tuple)) else [value]
    for entry in entries:
        if isinstance(entry, (dict, list, tuple)) or (
            isinstance(entry, str) and not is_quantity_text(entry)
        ):
            fail(path, fixed_value_message)
    if 0 in ranks:
        if isinstance(value, (list, tuple)):
            fail(path, f"{name} is a scalar, so it takes one constant, got the list {value!r}.")
        return constant(value, name, path, tree_context, expected, unit)
    if ranks != (1,):
        fail(path, f"{name} is a tensor. A fixed value of a tensor is not available.")
    count = tree_context.space_dimension()
    if not isinstance(value, (list, tuple)) or (count is not None and len(value) != count):
        fail(
            path,
            f"{name} is a vector, so it takes a list of "
            + (f"{count} constants" if count is not None else "one constant per component")
            + f", got {value!r}.",
        )
    return [
        constant(entry, name, f"{path}[{index}]", tree_context, expected, unit)
        for index, entry in enumerate(value)
    ]


def region_values(value, path: str, tree_context, read) -> dict:
    """A mapping of regions to values, each value read by read(value, path)."""
    known = tree_context.region_names()
    regions = {}
    for region, region_value in mapping(value, path).items():
        if region in ("table", "unit", "value"):
            fail(
                path,
                f"the region names table, unit and value are invalid, got the region '{region}'.",
            )
        if known is not None and region not in known:
            fail(path, f"unknown region '{region}'." + suggestion(region, known))
        regions[region] = read(region_value, join(path, region))
    if not regions:
        fail(path, "expects a value or a mapping of regions to values, got an empty mapping.")
    return regions


# Sections

property_name_message = (
    "the name of a property is a word of letters, digits and underscores, other than "
    + ", ".join(sorted(reserved_names))
    + "."
)


def properties(value, path, tree_context):
    """One entry per property: {"unit", "value", "regions"} with exactly one of value and regions set.

    A standard name takes its unit from the vocabulary. A name outside it declares its
    unit once, in the form {unit: ..., value: ...}, whose value takes every form of a
    property value, a mapping of regions included. A bare number of a declared name
    is in its declared unit.
    """
    result = {}
    for name, given in mapping(value, path).items():
        name_path = join(path, name)
        if name in reserved_names:
            fail(name_path, property_name_message)
        entry = standard_names().get(name)
        if isinstance(given, dict) and ("unit" in given or "value" in given):
            if entry is not None:
                fail(
                    name_path,
                    f"'{name}' is a standard name with the unit {entry.unit}, so it takes no unit. "
                    "Give its value alone.",
                )
            if name_pattern.fullmatch(name) is None:
                fail(name_path, property_name_message)
            declaration = table({"unit": (text(), required), "value": (anything, required)})(
                given, name_path, tree_context
            )
            try:
                parse_unit(declaration["unit"])
            except input_error as error:
                fail(join(name_path, "unit"), str(error))
            given, name_path = declaration["value"], join(name_path, "value")
        elif entry is None:
            try:
                entry, _ = lookup(name, tree_context.coordinate_system)
            except input_error as error:
                fail(
                    name_path,
                    f"{error} A name outside the vocabulary declares its unit once, "
                    f"as in {name}: {declaration_example}.",
                )
            fail(
                name_path,
                f"a property takes the whole quantity {entry.name}, not one component of it.",
            )
        expected_quantity = tree_context.quantity_of(name, name_path)

        def read(property_given, property_path, name=name, expected_quantity=expected_quantity):
            return property_value(
                property_given, name, property_path, tree_context, expected_quantity
            )

        if isinstance(given, dict) and "table" not in given:
            result[name] = {
                "unit": expected_quantity[1],
                "value": None,
                "regions": region_values(given, name_path, tree_context, read),
            }
        else:
            result[name] = {
                "unit": expected_quantity[1],
                "value": read(given, name_path),
                "regions": {},
            }
    return result


def initial_conditions(value, path, tree_context):
    """A constant per field, or a list of constants for a vector field."""
    return {
        name: fixed_value(entry, name, join(path, name), tree_context)
        for name, entry in mapping(value, path).items()
    }


def boundary_conditions(value, path, tree_context):
    """One entry per boundary. Each key sets a field value or a boundary flux.

    A key with a standard name, such as heat_flux, is a field or a boundary flux with
    one constant. It takes a boundary value or a mapping of regions to boundary values.
    Any other key, such as convection, names a boundary flux with several constants and
    takes a mapping of them. Each of its constants takes a boundary value or a mapping
    of regions to boundary values. The laws of the problem decide which keys are
    fields, so the constant-only rule of a fixed value applies there.
    """
    known = tree_context.boundary_names()
    result = {}
    for boundary, entries in mapping(value, path).items():
        boundary_path = join(path, boundary)
        if known is not None and boundary not in known:
            fail(path, f"unknown boundary '{boundary}'." + suggestion(boundary, known))
        result[boundary] = {}
        for key, entry in mapping(entries, boundary_path).items():
            key_path = join(boundary_path, key)
            if isinstance(entry, dict) and "table" not in entry and key not in standard_names():
                result[boundary][key] = {
                    name: flux_constant(given, name, join(key_path, name), tree_context)
                    for name, given in mapping(entry, key_path).items()
                }
            else:
                result[boundary][key] = flux_constant(entry, key, key_path, tree_context)
    return result


def flux_constant(value, name: str, path: str, tree_context):
    """One constant of a boundary flux: a boundary value, or a mapping of regions to boundary values."""
    if isinstance(value, dict) and "table" not in value:
        return region_values(
            value,
            path,
            tree_context,
            lambda region_value, region_path: boundary_value(
                region_value, name, region_path, tree_context
            ),
        )
    return boundary_value(value, name, path, tree_context)


def boundary_value(value, name: str, path: str, tree_context):
    """A value of the property value forms, or a list of constants for a vector field."""
    if isinstance(value, (list, tuple)):
        return fixed_value(value, name, path, tree_context)
    expected, unit, _ = tree_context.quantity_of(name, path)
    # A boundary value is scalar: a component, or the normal part of a flux.
    return property_value(value, name, path, tree_context, (expected, unit, (0,)))


def physics(value, path, tree_context):
    """One entry per built-in physics.

    The keys of an entry are the parameters of its physics function after properties, and
    method. The value of a parameter with a standard name is converted to SI.
    """
    result = {}
    for name, entry in mapping(value, path).items():
        entry_path = join(path, name)
        if name not in physics_functions:
            fail(path, f"unknown physics '{name}'." + suggestion(name, physics_functions))
        given = mapping(entry, entry_path)
        parameters = list(inspect.signature(physics_functions[name]).parameters.values())[1:]
        allowed = [parameter.name for parameter in parameters] + ["method"]
        for key in given:
            if key not in allowed:
                fail(entry_path, f"unknown key '{key}'." + suggestion(key, allowed))
        for parameter in parameters:
            if parameter.name not in given and parameter.default is inspect.Parameter.empty:
                fail(entry_path, f"the key '{parameter.name}' is missing.")
        checked = {}
        for key, option in given.items():
            key_path = join(entry_path, key)
            if key == "method":
                checked[key] = text(methods)(option, key_path, tree_context)
            elif key in option_choices.get(name, {}):
                checked[key] = text(option_choices[name][key])(option, key_path, tree_context)
            elif key in standard_names():
                checked[key] = fixed_value(option, key, key_path, tree_context)
            else:
                checked[key] = option
        result[name] = checked
    return result


def models(value, path, tree_context):
    """User models keep their strings, which model.parse reads and checks."""
    result = {}
    for name, entry in mapping(value, path).items():
        entry_path = join(path, name)
        keys = dict.fromkeys(model_keys, (anything, optional))
        keys["unknown"] = (anything, required)
        keys["method"] = (text(methods), optional)
        result[name] = table(keys)(entry, entry_path, tree_context)
    return result


def step_control(value, path, tree_context):
    """By error with relative_tolerance, by iterations with target_iterations, by Courant with
    Courant_number, or fixed."""
    by = text(step_controls)(
        mapping(value, path).get("by", "error"), join(path, "by"), tree_context
    )
    keys = {"by": (text(step_controls), "error")}
    if by == "iterations":
        keys["target_iterations"] = (integer(1), 5)
    elif by == "Courant":
        keys["Courant_number"] = (number(positive=True), 0.5)
    elif by == "error":
        keys["relative_tolerance"] = (number(maximum=1.0, positive=True), 1.0e-3)
    return table(keys)(value, path, tree_context)


def postprocessor(value, path, tree_context):
    """One reduction of a field, with the region, point or boundary it needs."""
    given = mapping(value, path)
    present = [key for key in reductions if key in given]
    if len(present) != 1:
        fail(path, "expects exactly one of " + ", ".join(f"'{key}'" for key in reductions) + ".")
    reduction = present[0]
    keys = {
        reduction: (
            text(tree_context.flux_names()) if reduction == "boundary_flux" else field_name,
            required,
        )
    }
    if reduction in region_reductions:
        keys["region"] = (text(tree_context.region_names()), optional)
    if reduction == "point_value":
        keys["point"] = (sequence(quantity("m"), tree_context.space_dimension()), required)
    if reduction == "boundary_flux":
        keys["boundary"] = (text(tree_context.boundary_names()), required)
    return table(keys)(given, path, tree_context)


def PETSc_options(value, path, tree_context):
    """PETSc option names without the leading dash, mapped to their values as text."""
    result = {}
    for key, option in mapping(value, path).items():
        if key.startswith("-"):
            fail(path, f"write the option '{key}' without the leading dash.")
        if isinstance(option, bool):
            result[key] = "true" if option else "false"
        elif isinstance(option, (int, float, str)):
            result[key] = str(option)
        else:
            fail(join(path, key), f"expects a text or a number, got {option!r}.")
    return result


def solver(value, path, tree_context):
    result = table(
        {
            "relative_tolerance": (number(positive=True), optional),
            "PETSc_options": (PETSc_options, optional),
        }
    )(value, path, tree_context)
    if "relative_tolerance" in result and "snes_rtol" in result.get("PETSc_options", {}):
        fail(
            path,
            "gives the relative tolerance twice, as relative_tolerance and as PETSc_options.snes_rtol. "
            "Keep relative_tolerance.",
        )
    return result


def generated_mesh(form: str):
    count = generated_dimensions[form]
    if form == "segment":
        return table(
            {
                "length": (quantity("m", positive=True), required),
                "cells": (integer(1), required),
            }
        )
    cell_types = (
        ("quadrilateral", "triangle") if form == "rectangle" else ("hexahedron", "tetrahedron")
    )
    return table(
        {
            "size": (sequence(quantity("m", positive=True), count), required),
            "cells": (sequence(integer(1), count), required),
            "cell_type": (text(cell_types), default_cell_types[form]),
        }
    )


def mesh_file(value, path, tree_context):
    return tree_context.path_of(text()(value, path, tree_context))


def length_unit(value, path, tree_context):
    """The unit of the coordinates of a mesh file, which the file does not state."""
    unit = text()(value, path, tree_context)
    try:
        factor, unit_dimension = parse_unit(unit)
    except input_error as error:
        fail(path, str(error))
    if unit_dimension != parse_unit("m")[1]:
        fail(path, f"expects a unit of length, such as m or mm, got '{unit}'.")
    return factor


def mesh(value, path, tree_context):
    """Exactly one of segment, rectangle, box and file, plus the coordinate system."""
    keys = {form: (generated_mesh(form), optional) for form in generated_dimensions}
    keys["file"] = (mesh_file, optional)
    keys["length_unit"] = (length_unit, optional)
    keys["coordinate_system"] = (text(tuple(position_names)), "cartesian")
    result = table(keys)(value, path, tree_context)
    forms = [form for form in mesh_forms if form in result]
    if len(forms) != 1:
        fail(path, "expects exactly one of " + ", ".join(f"'{form}'" for form in mesh_forms) + ".")
    form = forms[0]
    if form == "file" and "length_unit" not in result:
        fail(
            path,
            "a mesh file does not state the unit of its coordinates. Give it as length_unit, "
            "such as length_unit: mm.",
        )
    if form != "file" and "length_unit" in result:
        fail(
            join(path, "length_unit"),
            "a generated mesh takes its sizes with their units, so it takes no length_unit.",
        )
    if result["coordinate_system"] == "axisymmetric" and form in ("segment", "box"):
        fail(
            join(path, "coordinate_system"),
            f"axisymmetric coordinates need a mesh of space dimension 2, but a "
            f"{form} has space dimension {generated_dimensions[form]}.",
        )
    return result


def gravity(value, path, tree_context):
    """The gravitational acceleration: one quantity in m/s^2 per direction of space."""
    count = tree_context.space_dimension()
    if not isinstance(value, (list, tuple)) or (count is not None and len(value) != count):
        examples = {1: '["-9.81 m/s^2"]', 3: '["0 m/s^2", "0 m/s^2", "-9.81 m/s^2"]'}
        example = examples.get(count, '["0 m/s^2", "-9.81 m/s^2"]')
        fail(
            path,
            "gravity is a list of "
            + (f"{count} accelerations" if count is not None else "one acceleration per direction")
            + f", such as {example}, got {value!r}.",
        )
    return [
        quantity("m/s^2")(entry, f"{path}[{index}]", tree_context)
        for index, entry in enumerate(value)
    ]


sections = {
    "mesh": (mesh, required),
    "method": (text(methods), optional),
    "physics": (physics, optional),  # A problem of user models alone has no physics section
    "properties": (properties, optional),
    "gravity": (gravity, optional),
    "models": (models, optional),
    "initial_conditions": (initial_conditions, optional),
    "boundary_conditions": (boundary_conditions, optional),
    "time": (
        table(
            {
                "end_time": (quantity("s", positive=True), required),
                "initial_time_step": (quantity("s", positive=True), required),
                "step_control": (step_control, {"by": "error", "relative_tolerance": 1.0e-3}),
            }
        ),
        optional,
    ),
    "mesh_adaptivity": (
        table(
            {
                "indicator": (text(indicators), required),
                "field": (field_name, required),
                "refine_fraction": (number(minimum=0.0, maximum=1.0), required),
                "coarsen_fraction": (number(minimum=0.0, maximum=1.0), required),
                "interval": (integer(1), required),
                "maximum_refinement_level": (integer(0), required),
            }
        ),
        optional,
    ),
    "postprocessors": (named(postprocessor), optional),
    "output": (
        table(
            {
                "file": (text(), required),
                "interval": (integer(1), required),
                "fields": (sequence(field_name), required),
            }
        ),
        optional,
    ),
    "solver": (solver, optional),
}


def validate(tree: dict, directory: str | Path = ".") -> dict:
    """Validates a whole input tree and returns it with SI values and defaults, in section_order.

    directory is the folder against which relative mesh and table files resolve: the
    folder of the input file, or the working directory for the Python API.
    """
    return table(sections)(tree, "", context(mapping(tree, ""), directory))


def validate_section(tree: dict, key: str, directory: str | Path = "."):
    """Validates one top-level section of a tree that may still be incomplete, and returns it."""
    if key not in sections:
        fail("", f"unknown section '{key}'." + suggestion(key, sections))
    if key not in tree:
        fail(key, "the section is missing from the tree.")
    check_section, _ = sections[key]
    return check_section(tree[key], key, context(tree, directory))
