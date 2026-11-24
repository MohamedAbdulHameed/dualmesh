"""Turns the formulas of YAML and TOML into expression graphs.

The grammar, from the lowest to the highest precedence:

```text
sum      := product (("+" | "-") product)*        left-associative
product  := unary (("*" | "/") unary)*            left-associative
unary    := ("-" | "+") unary | power
power    := primary ("^" unary)?                  right-associative, so 2^3^2 is 2^9
primary  := number | "[" quantity "]" | name | name "(" arguments ")" | "(" sum ")"
```

A unary minus binds less tightly than `^`, so `-a^2` is $-(a^2)$. A
quantity in square brackets, such as `[2.77 eV]`, becomes a `number` in SI
with its dimension. A bare number is dimensionless.

A name resolves, in this order, to:

1. the entry of `symbols`, or one component of a vector or tensor entry, for
   a name such as `hydrogen_flow_x` or `back_stress_xy`,
2. a position name of the coordinate system, a component of the position in
   m: `x`, `y` and `z` in Cartesian coordinates, the radius `r` and the axial
   coordinate `z` in axisymmetric ones,
3. `time` and `time_step` in s, or `cell_size`, the diameter of the cell in m,
   which is the largest distance between two of its vertices,
4. a physical constant of the registry, such as `molar_gas_constant`,
5. `field(name)` for a standard name, or one component of it for a name such
   as `displacement_x` or `stress_xy`.

`symbols` is any mapping. The parser reads it through `in` and `[]` only, so a
mapping that builds its entries on demand builds only the names a formula
reads.

Steps 1 and 5 read the component suffix with `units.split_component`. A
suffix of the wrong length stops the parse. At step 1 the message names the
rank of the symbol. At step 5, for a name such as `displacement_xy`, the
registry gives the message.

A call names a function of the model language, such as `exp`, `grad` or
`max`. `time`, `time_step` and `cell_size` are also calls without arguments,
so `time()` equals `time`.
"""

import collections
import difflib
import importlib
import re
from collections.abc import Mapping

from dualmesh.input import units
from dualmesh.input.units import input_error
from dualmesh.model.conservation_law import conservation_law
from dualmesh.model.dimensions import dimension_error

# import_module returns the module itself even when dualmesh.model binds the
# name expression to the class of the same name.
graph = importlib.import_module("dualmesh.model.expression")


class syntax_error(input_error):
    """A formula that does not follow the grammar."""


calls = {
    "grad": (graph.grad, 1),
    "exp": (graph.exp, 1),
    "log": (graph.log, 1),
    "sqrt": (graph.sqrt, 1),
    "sin": (graph.sin, 1),
    "cos": (graph.cos, 1),
    "tanh": (graph.tanh, 1),
    "abs": (abs, 1),
    "max": (graph.maximum, 2),
    "min": (graph.minimum, 2),
    "less": (graph.less, 2),
    "greater": (graph.greater, 2),
    "conditional": (graph.conditional, 3),
    "dot": (graph.dot, 2),
    "inner": (graph.inner, 2),
    "outer": (graph.outer, 2),
    "transpose": (graph.transpose, 1),
    "trace": (graph.trace, 1),
    "symmetric": (graph.symmetric, 1),
    "deviatoric": (graph.deviatoric, 1),
    "determinant": (graph.determinant, 1),
    "inverse": (graph.inverse, 1),
    "norm": (graph.norm, 1),
    "position": (graph.position, 0),
    "time": (graph.time, 0),
    "time_step": (graph.time_step, 0),
    "cell_size": (graph.cell_size, 0),
    "normal": (graph.normal, 0),
    "identity": (graph.identity, 0),
}

point_names = {"time": graph.time, "time_step": graph.time_step, "cell_size": graph.cell_size}


def check_coordinate_system(coordinate_system: str) -> str:
    if coordinate_system not in units.position_names:
        raise input_error(
            f"unknown coordinate system '{coordinate_system}'. "
            f"The coordinate systems are {', '.join(units.position_names)}"
        )
    return coordinate_system


token_pattern = re.compile(
    r"""
    (?P<space>\s+)
  | (?P<number>(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)
  | (?P<name>[A-Za-z_][A-Za-z0-9_]*)
  | (?P<quantity>\[[^\[\]]*\])
  | (?P<operator>[-+*/^(),])
""",
    re.VERBOSE,
)


def tokenize(text: str) -> list:
    """Return the tokens of text as (kind, text, column), with column counted from 1."""
    tokens = []
    position = 0
    while position < len(text):
        match = token_pattern.match(text, position)
        if match is None:
            raise syntax_error(
                f"formula '{text}', column {position + 1}: unexpected character '{text[position]}'"
            )
        if match.lastgroup != "space":
            tokens.append((match.lastgroup, match.group(), position + 1))
        position = match.end()
    tokens.append(("end", "", len(text) + 1))
    return tokens


class parser:
    """Recursive-descent parser of one formula."""

    def __init__(self, text: str, symbols: Mapping, coordinate_system: str):
        self.text = text
        self.symbols = symbols
        self.coordinate_system = coordinate_system
        self.position_names = units.position_names[coordinate_system]
        self.tokens = tokenize(text)
        self.index = 0

    def fail(self, message: str, column: int | None = None):
        column = self.tokens[self.index][2] if column is None else column
        raise syntax_error(f"formula '{self.text}', column {column}: {message}")

    def peek(self) -> tuple:
        return self.tokens[self.index]

    def take(self) -> tuple:
        token = self.tokens[self.index]
        self.index += 1
        return token

    def accept(self, operator: str) -> bool:
        kind, token_text, _ = self.peek()
        if kind == "operator" and token_text == operator:
            self.index += 1
            return True
        return False

    def expect(self, operator: str):
        if not self.accept(operator):
            found = self.peek()[1] or "the end"
            self.fail(f"expected '{operator}', found '{found}'")

    def parse_all(self):
        if self.peek()[0] == "end":
            self.fail("the formula is empty")
        result = self.parse_sum()
        if self.peek()[0] != "end":
            self.fail(f"unexpected '{self.peek()[1]}'. Write every product with '*'")
        return result

    def parse_sum(self):
        result = self.parse_product()
        while True:
            if self.accept("+"):
                result = result + self.parse_product()
            elif self.accept("-"):
                result = result - self.parse_product()
            else:
                return result

    def parse_product(self):
        result = self.parse_unary()
        while True:
            if self.accept("*"):
                result = result * self.parse_unary()
            elif self.accept("/"):
                result = result / self.parse_unary()
            else:
                return result

    def parse_unary(self):
        if self.accept("-"):
            return -self.parse_unary()
        if self.accept("+"):
            return self.parse_unary()
        return self.parse_power()

    def parse_power(self):
        base_operand = self.parse_primary()
        if self.accept("^"):
            return base_operand ** self.parse_unary()
        return base_operand

    def parse_primary(self):
        kind, token_text, column = self.take()
        if kind == "number":
            return graph.number(float(token_text))
        if kind == "quantity":
            value, quantity_dimension = units.parse_quantity(token_text[1:-1])
            return graph.number(value, quantity_dimension)
        if kind == "name":
            if self.accept("("):
                return self.parse_call(token_text, column)
            return self.resolve(token_text, column)
        if kind == "operator" and token_text == "(":
            result = self.parse_sum()
            self.expect(")")
            return result
        self.index -= 1
        self.fail(
            f"expected a number, a quantity, a name or '(', found '{token_text or 'the end'}'"
        )

    def parse_call(self, name: str, column: int):
        if name not in calls:
            nearest = difflib.get_close_matches(name, list(calls))
            hint = f". Did you mean {', '.join(nearest)}?" if nearest else ""
            self.fail(f"unknown function '{name}'{hint}", column)
        function, argument_count = calls[name]
        arguments = []
        if not self.accept(")"):
            arguments.append(self.parse_sum())
            while self.accept(","):
                arguments.append(self.parse_sum())
            self.expect(")")
        if len(arguments) != argument_count:
            self.fail(f"{name} takes {argument_count} arguments, got {len(arguments)}", column)
        return function(*arguments)

    def resolve(self, name: str, column: int):
        if name in self.symbols:
            return self.symbols[name]
        base_name, component = units.split_component(name, self.coordinate_system) or (None, ())
        if base_name is not None and base_name in self.symbols:
            node = self.symbols[base_name]
            if node.rank != len(component):
                self.fail(
                    f"'{base_name}' has rank {node.rank}, so '{name}' is no component of it. "
                    f"A vector takes one suffix letter and a tensor two, from "
                    f"{', '.join(self.position_names)}",
                    column,
                )
            return node[component]
        if name in self.position_names:
            return graph.position()[self.position_names.index(name)]
        if name in point_names:
            return point_names[name]()
        registry = units.standard_names()
        if name in registry or base_name in registry:
            try:
                entry, component = units.lookup(name, self.coordinate_system)
            except input_error as error:
                self.fail(str(error), column)
            if entry.value is not None:
                return graph.physical_constant(entry.name)
            # A component of a scalar_or_tensor name selects the tensor form.
            node = graph.field(entry.name, shape="tensor" if len(component) == 2 else "scalar")
            return node[component] if component else node
        nearest = difflib.get_close_matches(name, [*self.position_names, *point_names, *registry])
        hint = f". The nearest names are {', '.join(nearest)}" if nearest else ""
        self.fail(f"unknown name '{name}'{hint}", column)


def parse(
    text: str, symbols: Mapping | None = None, coordinate_system: str = "cartesian"
) -> graph.expression:
    """Return the expression graph of a formula. Names resolve first through symbols."""
    if not isinstance(text, str):
        raise input_error(f"a formula must be a string, got {text!r}")
    check_coordinate_system(coordinate_system)
    try:
        return parser(text, {} if symbols is None else symbols, coordinate_system).parse_all()
    except syntax_error:
        raise
    except input_error as error:
        raise type(error)(f"formula '{text}': {error}") from None


# Models of the input


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
term_keys = ("storage", "inertia", "flux", "source")


def check_keys(mapping, allowed, path: str):
    if not isinstance(mapping, dict):
        raise input_error(f"{path} must be a mapping, got {mapping!r}")
    for key in mapping:
        if key not in allowed:
            nearest = difflib.get_close_matches(str(key), list(allowed), n=1)
            hint = (
                f". Did you mean '{nearest[0]}'?"
                if nearest
                else f". The keys are {', '.join(allowed)}"
            )
            raise input_error(f"{path}: unknown key '{key}'{hint}")


def section(entry: dict, key: str, path: str) -> dict:
    """Return the mapping under key of a model entry, or an empty mapping when key is absent."""
    value = entry.get(key)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise input_error(f"{path}.{key} must be a mapping, got {value!r}")
    return value


def at(path: str, function, *arguments, **keywords):
    """Return function(*arguments, **keywords), with path at the start of every input_error."""
    try:
        return function(*arguments, **keywords)
    except input_error as error:
        raise type(error)(f"{path}: {error}") from None


def parse_value(value, symbols: Mapping, coordinate_system: str, path: str):
    """Return the expression of an input value: a formula, a number or None."""
    if value is None or isinstance(value, graph.expression):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return at(path, graph.number, value)
    return at(path, parse, value, symbols, coordinate_system)


def check_expression_dimension(node, expression_name: str, unit, path: str):
    """Check an expression against the registry for a standard name, else against its declared unit."""
    standard = units.standard_names().get(expression_name)
    if standard is not None:
        if standard.value is not None:
            raise input_error(
                f"{path}: '{expression_name}' is a physical constant and cannot name an expression"
            )
        if unit is not None:
            raise input_error(
                f"{path}: '{expression_name}' is a standard name with the unit {standard.unit}. "
                f"Leave out unit"
            )
        if standard.shape == "scalar_or_tensor":
            expected_ranks = (0, 2)
        else:
            expected_ranks = (graph.rank_of_shape[standard.shape],)
        if node.rank not in expected_ranks:
            raise graph.rank_error(
                f"{path}: the expression has rank {node.rank}, but '{expression_name}' "
                f"has the shape {standard.shape}"
            )
        expected, shown_unit = standard.physical_dimension, standard.unit
    else:
        if unit is None:
            raise input_error(
                f"{path}: '{expression_name}' is not a standard name, so it needs a unit, "
                f"for example {{unit: 1/s, value: ...}}"
            )
        expected, shown_unit = graph.SI_dimension(unit, path), unit
    if node.dimension != expected:
        raise dimension_error(
            f"{path}: the expression has the dimension {node.dimension}, "
            f"which differs from the unit '{shown_unit}'"
        )


def conservation_law_from_input(
    name: str, entry: dict, symbols: Mapping, coordinate_system: str
) -> conservation_law:
    """Return the conservation law of one entry of the models section.

    The entry holds the keys of `model_keys`, with its strings not yet
    converted. The key `method` belongs to the problem and is not read here.
    Every term, state rate and boundary flux resolves a name first through the
    unknown, the states and the expressions of the entry, then through
    symbols, which hold the properties of one region. Any term or boundary
    flux may read the field of another physics by its standard name, such as
    `temperature`. A constant that a boundary flux reads takes its value from
    the boundary condition entry of that flux.
    """
    path = f"models.{name}"
    check_keys(entry, model_keys, path)
    check_coordinate_system(coordinate_system)
    if "unknown" not in entry:
        raise input_error(f"{path}: the key 'unknown' is missing")

    unknown_entry = entry["unknown"]
    if isinstance(unknown_entry, str):
        unknown_entry = {unknown_entry: None}
    if not isinstance(unknown_entry, dict) or len(unknown_entry) != 1:
        raise input_error(f"{path}.unknown must name one field, got {unknown_entry!r}")
    ((unknown_name, declaration),) = unknown_entry.items()
    unknown_path = f"{path}.unknown.{unknown_name}"
    declaration = declaration or {}
    check_keys(declaration, ("unit", "shape"), unknown_path)
    unknown = at(
        unknown_path,
        graph.field,
        unknown_name,
        unit=declaration.get("unit"),
        shape=declaration.get("shape", "scalar"),
    )
    local_symbols = {unknown_name: unknown}
    # The names of the entry come first, then the properties of the region.
    model_symbols = collections.ChainMap(local_symbols, symbols)

    def declare(symbol: str, node, symbol_path: str):
        if symbol in local_symbols:
            raise input_error(f"{symbol_path}: the name '{symbol}' is declared twice")
        local_symbols[symbol] = node

    def formula(value, value_path: str):
        return parse_value(value, model_symbols, coordinate_system, value_path)

    states = {}
    for state_name, state_entry in section(entry, "states", path).items():
        state_path = f"{path}.states.{state_name}"
        check_keys(state_entry, ("unit", "initial", "shape", "rate"), state_path)
        if "rate" not in state_entry:
            raise input_error(f"{state_path}: the key 'rate' is missing")
        node = at(
            state_path,
            graph.state,
            state_name,
            unit=state_entry.get("unit"),
            initial=state_entry.get("initial", 0.0),
            shape=state_entry.get("shape", "scalar"),
        )
        declare(state_name, node, state_path)
        states[node] = state_entry["rate"]

    for expression_name, value in section(entry, "expressions", path).items():
        expression_path = f"{path}.expressions.{expression_name}"
        unit = None
        if isinstance(value, dict):
            check_keys(value, ("unit", "value"), expression_path)
            unit = value.get("unit")
            value = value.get("value")
        node = formula(value, expression_path)
        if node is None:
            raise input_error(f"{expression_path}: the expression has no value")
        check_expression_dimension(node, expression_name, unit, expression_path)
        declare(expression_name, node, expression_path)

    terms = {key: formula(entry.get(key), f"{path}.{key}") for key in term_keys}
    state_rates = {
        node: formula(rate, f"{path}.states.{node.name}.rate") for node, rate in states.items()
    }
    boundary_fluxes = {
        flux_name: formula(value, f"{path}.boundary_fluxes.{flux_name}")
        for flux_name, value in section(entry, "boundary_fluxes", path).items()
    }
    return at(
        path,
        conservation_law,
        name=name,
        unknown=unknown,
        state_rates=state_rates,
        boundary_fluxes=boundary_fluxes,
        flux_name=entry.get("flux_name"),
        **terms,
    )
