"""The one schema of the input tree.

`validate` checks every key of a tree, names the nearest valid key for a
misspelled one, converts every quantity to SI with `units.to_SI`, checks the
dimension of every formula and table, reads the CSV tables and fills the
defaults. It returns a new tree in `section_order`. It accepts only what the
build lowers onto the engine, and refuses the rest at its own key. An option
whose choice the engine owns, such as the formulation of a flow or the step
control, passes through unset, so the engine's default holds.

Each part of the schema is a check function `check(value, path, tree_context)`
that returns the validated value. `path` names the key in error messages, for
example "properties.thermal_conductivity.fuel". `tree_context` gives access to
the whole tree, for the names that one section takes from another.
"""

import collections
import copy
import importlib
import inspect
import itertools
import os
import re
from pathlib import Path

from dualmesh.console import suggestion
from dualmesh.input.physics_options import option_choices, physics_functions, read_only_with
from dualmesh.input.read import read_table
from dualmesh.input.units import (
    SI_unit_text,
    dimensionless,
    fail,
    input_error,
    lookup,
    parse_unit,
    position_names,
    standard_names,
    to_SI,
)
from dualmesh.model import conservation_law_from_input, field, parse
from dualmesh.model.parse import formula_text
from dualmesh.tables import is_number

# import_module returns the module itself, which dualmesh.model hides behind the class of the
# same name.
graph = importlib.import_module("dualmesh.model.expression")
graph_leaves = graph.leaves

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
    "eigenvalues",
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
reductions = ("integral", "average", "maximum", "minimum", "point_value", "boundary_flux", "line")
# The word that takes the place of the mapping of a boundary: a plane of symmetry.
symmetry = "symmetry"
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
# The symbol of the eigenvalue of an eigenvalue study, which the source of a model may read.
eigenvalue_symbol = "eigenvalue"
# Names that a formula reads as the position, the time, the cell size or the eigenvalue, so no
# property takes them.
reserved_names = {
    *itertools.chain(*position_names.values()),
    "time",
    "time_step",
    "cell_size",
    eigenvalue_symbol,
}
ranks_of_shape = {"scalar": (0,), "vector": (1,), "tensor": (2,), "scalar_or_tensor": (0, 2)}
declaration_example = "{unit: mol/m^3, value: 10 mol/m^3}"

required = object()  # Marks a key without a default that must be present
optional = object()  # Marks a key without a default that may be absent


# Messages


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
    dimension and rank, so that formulas that read them parse and check. The flux_name of a
    model outside the vocabulary is declared too, with the dimension of the flux of the model,
    so that a boundary condition gives its normal part. Unreadable declarations are left out
    here and reported by the check of their own section.

    `complete` is true for the whole tree, and false for one section of a tree that the Python
    API builds section by section, in which a model may name properties that come later.
    """

    def __init__(self, tree: dict, directory: str | Path = ".", complete: bool = False):
        self.tree = _mapping_or_empty(tree)
        self.directory = Path(directory)
        self.complete = complete
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
        for entry in _mapping_or_empty(self.tree.get("models")).values():
            self._declare_flux(_mapping_or_empty(entry))
        # The models read the eigenvalue besides the declared names, as a constant that the
        # engine sets in an eigenvalue study.
        self.model_symbols = collections.ChainMap(
            {eigenvalue_symbol: graph.constant(eigenvalue_symbol, unit=self.eigenvalue_unit())},
            self.symbols,
        )

    def eigenvalue_unit(self) -> str:
        """The unit of the eigenvalue: 1/s when the problem has a time derivative, the time terms
        of a physics or the storage of a model, and 1 when it has none. A field varies in time as
        u e^(-lambda t), so lambda is a rate when time derivatives define it, and a number when
        it only appears in the source of a model, as 1/k does in a criticality problem."""
        if _mapping_or_empty(self.tree.get("physics")):
            return "1/s"
        for entry in _mapping_or_empty(self.tree.get("models")).values():
            for key in ("storage", "inertia"):
                term = _mapping_or_empty(entry).get(key)
                if term is not None and not (is_number(term) and float(term) == 0.0):
                    return "1/s"
        return "1"

    def _declare_flux(self, entry: dict) -> None:
        """Declares the flux_name of a model outside the vocabulary with the dimension of its flux."""
        flux_name, flux = entry.get("flux_name"), entry.get("flux")
        if (
            not isinstance(flux_name, str)
            or not isinstance(flux, str)
            or name_pattern.fullmatch(flux_name) is None
            or flux_name in standard_names()
            or flux_name in reserved_names
            or flux_name in self.declared
        ):
            return
        try:
            node = parse(flux, self.symbols, self.coordinate_system)
        except input_error:
            return
        self.declared[flux_name] = (node.dimension, "vector", 1.0)
        self.symbols[flux_name] = field(
            flux_name, unit=SI_unit_text(node.dimension), shape="vector"
        )

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
        """The names of the fluxes that a boundary flux postprocessor can name: the heat flux, the
        traction of a solid and its components, and the flux names of the models."""
        letters = position_names[self.coordinate_system][: self.space_dimension()]
        names = {"heat_flux", "traction", *(f"traction_{letter}" for letter in letters)}
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


def check_formula(
    formula: str, name: str, expected, unit: str, ranks, path: str, tree_context
) -> str:
    """Parses a formula and checks its dimension and rank against those of name.

    Returns the text with its graph, which the build lowers without a second parse.
    """
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
    return formula_text(formula, node)


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
    """A table of one input in the validated form, with SI values sorted by the input.

    The column `name` is the output. The other column is the input, whose name parses as a
    scalar, such as temperature, time or x. The engine interpolates linearly between the rows.
    """
    columns, columns_path = table_columns(value, path, tree_context)
    if name not in columns:
        fail(
            columns_path,
            f"has no column '{name}'. The output column of a table has the name of its quantity.",
        )
    inputs = [column_name for column_name in columns if column_name != name]
    if len(inputs) != 1:
        fail(
            columns_path,
            f"a table interpolates in one input, so it takes one column besides '{name}', "
            f"and it has {len(inputs)}.",
        )
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
    rows = sorted(zip(converted[inputs[0]], converted[name], strict=True))
    points = [point for point, _ in rows]
    input_path = join(columns_path, inputs[0])
    if len(points) < 2:
        fail(input_path, f"expects at least 2 values, got {len(points)}.")
    for earlier, later in zip(points, points[1:]):
        if later == earlier:
            fail(input_path, f"the value {later:g} appears twice.")
    return {
        "inputs": [{"name": inputs[0], "values": points}],
        "values": [output for _, output in rows],
    }


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
    """A constant or a formula of the coordinates per field, or a list of them for a vector field."""
    if "eigenvalues" in tree_context.tree:
        fail(
            path,
            "an eigenvalue study finds the modes of the problem from no starting state, so it "
            "takes no initial_conditions.",
        )
    return {
        name: initial_value(entry, name, join(path, name), tree_context)
        for name, entry in mapping(value, path).items()
    }


# The leaves of a formula of the coordinates: numbers, constants with a value and the position.
coordinate_leaves = {"number", "constant", "position", "position_component", "identity"}


def initial_value(value, name: str, path: str, tree_context):
    """The value of a field at the start: a constant or a formula of the coordinates for a scalar,
    and a list of them, one per component, for a vector."""
    expected, unit, ranks = tree_context.quantity_of(name, path)
    if 0 in ranks:
        if isinstance(value, (list, tuple)):
            fail(path, f"{name} is a scalar, so it takes one value, got the list {value!r}.")
        return initial_component(value, name, path, tree_context, expected, unit)
    if ranks != (1,):
        fail(path, f"{name} is a tensor. An initial value of a tensor is not available.")
    count = tree_context.space_dimension()
    if not isinstance(value, (list, tuple)) or (count is not None and len(value) != count):
        fail(
            path,
            f"{name} is a vector, so it takes a list of "
            + (f"{count} values" if count is not None else "one value per component")
            + f", got {value!r}.",
        )
    return [
        initial_component(entry, name, f"{path}[{index}]", tree_context, expected, unit)
        for index, entry in enumerate(value)
    ]


def initial_component(value, name: str, path: str, tree_context, expected, unit: str):
    """One constant or one formula of the coordinates, in the unit of name."""
    if isinstance(value, (dict, list, tuple)):
        fail(path, "an initial value is a constant or a formula of the coordinates.")
    if isinstance(value, str) and not is_quantity_text(value):
        formula = check_formula(value, name, expected, unit, (0,), path, tree_context)
        others = sorted(
            {
                leaf.name if leaf.name else leaf.kind
                for leaf in graph_leaves(formula.node)
                if leaf.kind not in coordinate_leaves
                or (leaf.kind == "constant" and leaf.value is None)
            }
        )
        if others:
            fail(
                path,
                "an initial value is a formula of the coordinates, and this one reads "
                + ", ".join(others)
                + ".",
            )
        return formula
    return constant(value, name, path, tree_context, expected, unit)


def boundary_conditions(value, path, tree_context):
    """One entry per boundary: the word symmetry, or a mapping in which each key sets a field
    value or a boundary flux.

    A key with a standard name, such as heat_flux, is a field or a boundary flux with
    one constant, which takes a boundary value. Any other key, such as convection, names a
    boundary flux with several constants and takes a mapping of them, each a boundary value.
    The laws of the problem decide which keys are fields, so the constant-only rule of a fixed
    value applies there. The word symmetry makes the boundary a plane of symmetry, where each
    physics applies its own condition.
    """
    known = tree_context.boundary_names()
    result = {}
    for boundary, entries in mapping(value, path).items():
        boundary_path = join(path, boundary)
        if known is not None and boundary not in known:
            fail(path, f"unknown boundary '{boundary}'." + suggestion(boundary, known))
        if isinstance(entries, str):
            if entries != symmetry:
                fail(
                    boundary_path,
                    f"a boundary takes a mapping of its conditions or the word {symmetry}, got "
                    f"'{entries}'." + suggestion(entries, [symmetry]),
                )
            result[boundary] = symmetry
            continue
        result[boundary] = {}
        for key, entry in mapping(entries, boundary_path).items():
            key_path = join(boundary_path, key)
            if isinstance(entry, dict) and "table" not in entry and key not in standard_names():
                result[boundary][key] = {
                    name: boundary_value(given, name, join(key_path, name), tree_context)
                    for name, given in mapping(entry, key_path).items()
                }
            else:
                result[boundary][key] = boundary_value(entry, key, key_path, tree_context)
    return result


def boundary_value(value, name: str, path: str, tree_context):
    """A constant or a formula, or a list of them for a vector, one per component: a condition
    holds one value along its boundary, which the engine takes as an expression."""
    if isinstance(value, dict):
        fail(
            path,
            "a boundary condition takes a constant or a formula, so it takes no table and no "
            "value per region.",
        )
    expected, unit, ranks = tree_context.quantity_of(name, path)
    if not isinstance(value, (list, tuple)):
        # A boundary value is scalar: a component, or the normal part of a flux.
        return property_value(value, name, path, tree_context, (expected, unit, (0,)))
    if 0 in ranks:
        fail(path, f"{name} is a scalar, so it takes one value, got the list {value!r}.")
    if ranks != (1,):
        fail(path, f"{name} is a tensor. A boundary value of a tensor is not available.")
    count = tree_context.space_dimension()
    if count is not None and len(value) != count:
        fail(path, f"{name} is a vector, so it takes a list of {count} values, got {value!r}.")
    return [
        property_value(entry, name, f"{path}[{index}]", tree_context, (expected, unit, (0,)))
        for index, entry in enumerate(value)
    ]


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
        checked = {}
        for key, option in given.items():
            key_path = join(entry_path, key)
            if key == "method":
                checked[key] = one_method(option, key_path, tree_context)
            elif key in option_choices.get(name, {}):
                checked[key] = text(option_choices[name][key])(option, key_path, tree_context)
            else:
                checked[key] = option
        # The defaults of the options, from the signature of the physics function. An option
        # whose default is None is the engine's choice and stays unset.
        for parameter in parameters:
            if parameter.name in checked or parameter.default is None:
                continue
            if parameter.default is inspect.Parameter.empty:
                fail(entry_path, f"the key '{parameter.name}' is missing.")
            checked[parameter.name] = parameter.default
        for key, (other, value, reason) in read_only_with.get(name, {}).items():
            if key in given and checked[other] != value:
                fail(
                    join(entry_path, key),
                    f"{reason}, so {name} with {other} {checked[other]} takes no {key}.",
                )
        result[name] = checked
    return result


def one_method(value, path, tree_context) -> str:
    """The method of a physics or a model, which is the one method of the whole problem."""
    method = text(methods)(value, path, tree_context)
    tree = tree_context.tree
    others = {
        _mapping_or_empty(entry).get("method")
        for section in ("physics", "models")
        for entry in _mapping_or_empty(tree.get(section)).values()
    }
    others = (others | {tree.get("method")}) - {None, method}
    if others:
        fail(
            path,
            f"one problem takes one method, and the input gives {method} and "
            + " and ".join(sorted(map(str, others)))
            + ".",
        )
    return method


def models(value, path, tree_context):
    """User models keep their strings, which model.parse reads and checks.

    The whole tree parses each model into its conservation law and checks the dimensions of its
    terms, of its state rates and of its flux_name. One section alone checks the keys only,
    since the properties that a model reads may come in a later section.
    """
    result = {}
    for name, entry in mapping(value, path).items():
        entry_path = join(path, name)
        keys = dict.fromkeys(model_keys, (anything, optional))
        keys["unknown"] = (anything, required)
        keys["method"] = (one_method, optional)
        result[name] = table(keys)(entry, entry_path, tree_context)
        if tree_context.complete:
            check_model(name, result[name], entry_path, tree_context)
    return result


def check_model(name: str, entry: dict, path: str, tree_context) -> None:
    """Parses a model with the names of the tree and checks the dimensions of its law."""
    law = conservation_law_from_input(
        name, entry, tree_context.model_symbols, tree_context.coordinate_system
    )
    try:
        law.check()
    except input_error as error:
        raise type(error)(f"{path}: {error}") from None
    check_eigenvalue(law, path, tree_context)
    flux_name = entry.get("flux_name")
    if flux_name is None:
        return
    flux_path = join(path, "flux_name")
    if not isinstance(flux_name, str) or name_pattern.fullmatch(flux_name) is None:
        fail(flux_path, f"expects a word of letters, digits and underscores, got {flux_name!r}.")
    if flux_name == law.unknown.name or flux_name in reserved_names:
        fail(flux_path, f"'{flux_name}' names the unknown or a reserved name, not a flux.")
    standard = standard_names().get(flux_name)
    if standard is not None and (
        standard.shape != "vector" or standard.physical_dimension != law.flux.dimension
    ):
        fail(
            flux_path,
            f"'{flux_name}' is a standard name of the unit {standard.unit}, and the flux of the "
            f"model has the unit {SI_unit_text(law.flux.dimension)}.",
        )


def reads_eigenvalue(node) -> bool:
    return node is not None and any(
        leaf.kind == "constant" and leaf.name == eigenvalue_symbol for leaf in graph_leaves(node)
    )


def check_eigenvalue(law, path: str, tree_context) -> None:
    """The eigenvalue of an eigenvalue study appears in the source of a model only, and only in
    a problem with the eigenvalues section, since every other run has no eigenvalue."""
    terms = {"storage": law.storage, "inertia": law.inertia, "flux": law.flux}
    terms |= {f"states.{node.name}.rate": rate for node, rate in law.state_rates.items()}
    for key, node in terms.items():
        if reads_eigenvalue(node):
            fail(
                join(path, key),
                f"the {eigenvalue_symbol} of an eigenvalue study appears in the source of a model "
                "only.",
            )
    if reads_eigenvalue(law.source) and "eigenvalues" not in tree_context.tree:
        fail(
            join(path, "source"),
            f"the source reads the {eigenvalue_symbol}, which exists in an eigenvalue study only, "
            "so the problem needs the eigenvalues section.",
        )


# The step controllers of the engine that the input selects, each with the key of its one
# setting: the error of the step, the Newton iterations it takes, or a fixed step. A setting left
# out takes the engine's default.
step_controls = {
    "error": ("relative_tolerance", number(maximum=1.0, positive=True)),
    "iterations": ("target_iterations", integer(1)),
    "fixed": None,
}


def step_control(value, path, tree_context):
    """By error with relative_tolerance, by iterations with target_iterations, or fixed."""
    given = mapping(value, path)
    if "by" not in given:
        fail(path, "the key 'by' is missing.")
    by = text(step_controls)(given["by"], join(path, "by"), tree_context)
    keys = {"by": (text(step_controls), required)}
    if step_controls[by] is not None:
        setting, check = step_controls[by]
        keys[setting] = (check, optional)
    return table(keys)(value, path, tree_context)


def postprocessor(value, path, tree_context):
    """One reduction of a field over the whole domain, with the point or boundary it needs, or the
    values of a field along a line."""
    given = mapping(value, path)
    present = [key for key in reductions if key in given]
    if len(present) != 1:
        fail(path, "expects exactly one of " + ", ".join(f"'{key}'" for key in reductions) + ".")
    reduction = present[0]
    if "region" in given:
        fail(
            join(path, "region"),
            "a postprocessor reduces over the whole domain, so it takes no region.",
        )
    if "eigenvalues" in tree_context.tree:
        fail(
            path,
            "the modes of an eigenvalue study have no scale of their own, so an eigenvalue study "
            "takes no postprocessors.",
        )
    keys = {
        reduction: (
            text(tree_context.flux_names()) if reduction == "boundary_flux" else field_name,
            required,
        )
    }
    point = (sequence(quantity("m"), tree_context.space_dimension()), required)
    if reduction == "point_value":
        keys["point"] = point
    if reduction == "boundary_flux":
        keys["boundary"] = (text(tree_context.boundary_names()), required)
    if reduction == "line":
        keys.update({"from": point, "to": point, "points": (integer(2), required)})
    result = table(keys)(given, path, tree_context)
    if reduction == "line":
        if result["from"] == result["to"]:
            fail(join(path, "to"), "a line runs between two different points.")
        if tree_context.complete and "output" not in tree_context.tree:
            fail(
                path,
                "a line writes its values to <output file>_<name>.csv, so it needs the output "
                "section.",
            )
    return result


def PETSc_options(value, path, tree_context):
    """PETSc option names without the leading dash, mapped to their values as text, or to true
    for a flag without a value and false, as the engine's petsc_options take them."""
    result = {}
    for key, option in mapping(value, path).items():
        if key.startswith("-"):
            fail(path, f"write the option '{key}' without the leading dash.")
        if isinstance(option, bool):
            result[key] = option
        elif isinstance(option, (int, float, str)):
            result[key] = str(option)
        else:
            fail(join(path, key), f"expects a text or a number, got {option!r}.")
    return result


def solver(value, path, tree_context):
    """The options of the nonlinear solve, with the engine's keywords of them."""
    result = table(
        {
            "relative_tolerance": (number(positive=True), optional),
            "PETSc_options": (PETSc_options, optional),
        }
    )(value, path, tree_context)
    if "PETSc_options" in result:
        result["petsc_options"] = result.pop("PETSc_options")
    return result


def generated_mesh(form: str):
    """A generated mesh: its size, its cells, the corner with the smallest coordinates (origin,
    default 0 m) and the ratio of the length of each cell to the one before it along each axis
    (grading, default 1, a uniform mesh)."""
    count = generated_dimensions[form]
    if form == "segment":
        return table(
            {
                "length": (quantity("m", positive=True), required),
                "cells": (integer(1), required),
                "origin": (quantity("m"), 0.0),
                "grading": (number(positive=True), 1.0),
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
            "origin": (sequence(quantity("m"), count), [0.0] * count),
            "grading": (sequence(number(positive=True), count), [1.0] * count),
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
    axisymmetric = result["coordinate_system"] == "axisymmetric"
    if axisymmetric and form == "rectangle" and result[form]["origin"][0] < 0.0:
        fail(
            join(join(path, form), "origin"),
            "in axisymmetric coordinates the first coordinate is the radius r, so the mesh "
            "starts at r = 0 m or beyond.",
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


def time(value, path, tree_context):
    """The end of a transient run, its first step and the step control."""
    if "eigenvalues" in tree_context.tree:
        fail(
            path,
            "an eigenvalue study takes the place of a steady or transient run, so a problem with "
            "eigenvalues has no time section.",
        )
    return table(
        {
            "end_time": (quantity("s", positive=True), required),
            "initial_time_step": (quantity("s", positive=True), required),
            "step_control": (step_control, optional),
        }
    )(value, path, tree_context)


def eigenvalues(value, path, tree_context):
    """The eigenvalue study: the count eigenvalues closest to near. Every field varies in time as
    u e^(-lambda t), so the time terms of the problem define the eigenvalue lambda in 1/s. A
    problem without a time derivative defines it by the source of a model, where it is a
    number."""
    tree = tree_context.tree
    for section in ("time", "mesh_adaptivity"):
        if section in tree:
            fail(
                path,
                f"an eigenvalue study takes the place of a steady or transient run, so a problem "
                f"with eigenvalues has no {section} section.",
            )
    near = quantity(tree_context.eigenvalue_unit())
    if not tree_context.complete and not (tree.get("physics") or tree.get("models")):
        # The physics and the models, which set the unit of near, may come in a later call.
        near = anything
    return table({"count": (integer(1), 1), "near": (near, 0.0)})(value, path, tree_context)


def mesh_adaptivity(value, path, tree_context):
    """The loop that adapts the mesh to the error indicator of a field: after each solve of a
    steady run, and every interval accepted steps of a transient one."""
    transient = "time" in tree_context.tree
    keys = {
        "field": (field_name, required),
        "refine_fraction": (number(maximum=1.0, positive=True), required),
        "coarsen_fraction": (number(minimum=0.0, maximum=1.0), required),
        "maximum_refinement_level": (integer(0), required),
    }
    keys["interval"] = (integer(1), required if transient else optional)
    if not transient and tree_context.complete and "interval" in mapping(value, path):
        fail(
            join(path, "interval"),
            "a steady run adapts the mesh after each solve, so it takes no interval.",
        )
    return table(keys)(value, path, tree_context)


def output(value, path, tree_context):
    """The files of the fields: once at the end of a steady run, and at the start, every interval
    accepted steps and the end of a transient one.  The interval counts accepted steps, so a
    steady run takes none."""
    transient = "time" in tree_context.tree
    keys = {
        "file": (text(), required),
        "interval": (integer(1), 1 if transient else optional),
        "fields": (sequence(field_name), required),
    }
    if not transient and tree_context.complete and "interval" in mapping(value, path):
        fail(
            join(path, "interval"),
            "a steady run writes the fields once, at the end, so it takes no interval.",
        )
    return table(keys)(value, path, tree_context)


sections = {
    "mesh": (mesh, required),
    "method": (text(methods), optional),
    "physics": (physics, optional),  # A problem of user models alone has no physics section
    "properties": (properties, optional),
    "gravity": (gravity, optional),
    "models": (models, optional),
    "initial_conditions": (initial_conditions, optional),
    "boundary_conditions": (boundary_conditions, optional),
    "time": (time, optional),
    "eigenvalues": (eigenvalues, optional),
    "mesh_adaptivity": (mesh_adaptivity, optional),
    "postprocessors": (named(postprocessor), optional),
    "output": (output, optional),
    "solver": (solver, optional),
}


def validate(tree: dict, directory: str | Path = ".") -> dict:
    """Validates a whole input tree and returns it with SI values and defaults, in section_order.

    directory is the folder against which relative mesh and table files resolve: the
    folder of the input file, or the working directory for the Python API.
    """
    return table(sections)(tree, "", context(mapping(tree, ""), directory, complete=True))


def validate_section(tree: dict, key: str, directory: str | Path = "."):
    """Validates one top-level section of a tree that may still be incomplete, and returns it."""
    if key not in sections:
        fail("", f"unknown section '{key}'." + suggestion(key, sections))
    if key not in tree:
        fail(key, "the section is missing from the tree.")
    check_section, _ = sections[key]
    return check_section(tree[key], key, context(tree, directory))
