"""Turns the formulas of YAML and TOML and the equations of general_form_PDE into expression graphs.

One language serves the formulas of the input, the models of your own and the equations of
`general_form_PDE`. The grammar, from the lowest to the highest precedence:

```text
equation    := expression "=" expression
expression  := conjunction ("or" conjunction)*
conjunction := comparison ("and" comparison)*
comparison  := sum (("<" | ">" | "<=" | ">=" | "==" | "!=") sum)?
sum         := product (("+" | "-") product)*        left-associative
product     := unary (("*" | "/") unary)*            left-associative
unary       := ("-" | "+") unary | power
power       := primary (("^" | "**") unary)?         right-associative, so 2^3^2 is 2^9
primary     := number | "[" quantity "]" | "[" expression ("," expression)* "]"
             | derivative | name | name "(" arguments ")" | "(" expression ")"
derivative  := "d" field "/d" coordinate | "d2" field "/dt2"
```

A unary minus binds less tightly than `^`, so `-a^2` is $-(a^2)$. Square brackets around a
number and its unit, such as `[2.77 eV]`, hold a quantity, which becomes a `number` in SI with
its dimension. Square brackets around anything else, such as `[0, -g]`, hold the items of a
vector, or the rows of a tensor. A bare number is dimensionless. A comparison is 1 where it
holds and 0 elsewhere. `and` and `or` join conditions, each of which holds where it is not 0.

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

A call names a function of `calls`, such as `exp`, `grad` or `max`. `aliases` holds the short
spellings of the equation language, such as `sym` for `symmetric` and `if` for `conditional`,
which mean the same. `time`, `time_step` and `cell_size` are also calls without arguments, so
`time()` equals `time`.

The equations of general_form_PDE parse in an `equation_scope`, which knows the dimension of
space and the fields. It adds the calls of `space_calls`, which differentiate along the
coordinates (`div`, `laplacian`, `curl`, `grad_x`, and `grad` of any expression), `cross`, and
the derivative notation `du/dt`, `d2u/dt2`, `du/dx` and `du/dn`. `read_equations` sorts the
terms of each equation into a conservation law, and `read_condition` reads a boundary
condition.
"""

import collections
import difflib
import importlib
import math
import operator
import re
import zlib
from collections.abc import Mapping

from dualmesh.console import suggestion
from dualmesh.input import units
from dualmesh.input.units import dimensionless, input_error
from dualmesh.model.conservation_law import conservation_law
from dualmesh.model.dimensions import comparisons, dimension_error, is_zero

# import_module returns the module itself even when dualmesh.model binds the
# names expression and differentiate to the class and the function of the same names.
graph = importlib.import_module("dualmesh.model.expression")
calculus = importlib.import_module("dualmesh.model.differentiate")


class syntax_error(input_error):
    """A formula that does not follow the grammar."""


calls = {
    "grad": (graph.grad, 1),
    "exp": (graph.exp, 1),
    "log": (graph.log, 1),
    "log10": (graph.log10, 1),
    "sqrt": (graph.sqrt, 1),
    "sin": (graph.sin, 1),
    "cos": (graph.cos, 1),
    "tan": (graph.tan, 1),
    "asin": (graph.asin, 1),
    "acos": (graph.acos, 1),
    "atan": (graph.atan, 1),
    "sinh": (graph.sinh, 1),
    "cosh": (graph.cosh, 1),
    "tanh": (graph.tanh, 1),
    "erf": (graph.erf, 1),
    "floor": (graph.floor, 1),
    "ceil": (graph.ceil, 1),
    "sign": (graph.sign, 1),
    "abs": (abs, 1),
    "max": (graph.maximum, 2),
    "min": (graph.minimum, 2),
    "atan2": (graph.atan2, 2),
    "hypot": (graph.hypot, 2),
    "pow": (operator.pow, 2),
    "less": (graph.less, 2),
    "greater": (graph.greater, 2),
    "conditional": (graph.conditional, 3),
    "dot": (graph.dot, 2),
    "inner": (graph.inner, 2),
    "outer": (graph.outer, 2),
    "transpose": (graph.transpose, 1),
    "trace": (graph.trace, 1),
    "symmetric": (graph.symmetric, 1),
    "skew": (lambda tensor: 0.5 * (tensor - graph.transpose(tensor)), 1),
    "deviatoric": (graph.deviatoric, 1),
    "determinant": (graph.determinant, 1),
    "inverse": (graph.inverse, 1),
    "norm": (lambda argument: abs(argument) if argument.rank == 0 else graph.norm(argument), 1),
    "position": (graph.position, 0),
    "time": (graph.time, 0),
    "time_step": (graph.time_step, 0),
    "cell_size": (graph.cell_size, 0),
    "normal": (graph.normal, 0),
    "identity": (graph.identity, 0),
}

# The short spellings, each with the name of its call.
aliases = {
    "sym": "symmetric",
    "dev": "deviatoric",
    "det": "determinant",
    "if": "conditional",
    "ddot": "inner",
}

# The calls that differentiate along the coordinates or need the dimension of space, with their
# number of arguments. They take the space of an equation, so only the equations of
# general_form_PDE have them. A formula takes grad of a field from `calls`.
space_calls = {
    "grad": 1,
    "grad_x": 1,
    "grad_y": 1,
    "grad_z": 1,
    "div": 1,
    "laplacian": 1,
    "curl": 1,
    "cross": 2,
}

# The kind of each comparison operator.
comparison_kinds = {operator_text: kind for kind, operator_text in comparisons.items()}

point_names = {"time": graph.time, "time_step": graph.time_step, "cell_size": graph.cell_size}


def check_coordinate_system(coordinate_system: str) -> str:
    if coordinate_system not in units.position_names:
        raise input_error(
            f"unknown coordinate system '{coordinate_system}'. "
            f"The coordinate systems are {', '.join(units.position_names)}"
        )
    return coordinate_system


number_pattern = r"(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?"
# A quantity is a number and a unit in square brackets. The unit starts with a letter, '%' or
# '(', or, after a space, with the digit 1 of a unit such as 1/K. So [1 - x], [2*x, 0] and [11]
# are lists.
token_pattern = re.compile(
    rf"""
    (?P<space>\s+)
  | (?P<number>{number_pattern})
  | (?P<name>[A-Za-z_][A-Za-z0-9_]*)
  | (?P<quantity>\[\s*[-+]?{number_pattern}(?:\s*[A-Za-z%(]|\s+1(?=[\s/*^)\]]))[^\[\],]*\])
  | (?P<operator>\*\*|<=|>=|==|!=|[-+*/^(),<>=\[\]])
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
    """Recursive-descent parser of one formula or one equation.

    scope is the equation_scope of an equation of general_form_PDE, which is then also the
    symbols, or None for a formula.
    """

    def __init__(self, text: str, symbols: Mapping, coordinate_system: str, scope=None):
        self.text = text
        self.symbols = symbols
        self.scope = scope
        self.what = "formula" if scope is None else "equation"
        self.coordinate_system = coordinate_system
        self.position_names = units.position_names[coordinate_system]
        self.tokens = tokenize(text)
        self.index = 0

    def fail(self, message: str, column: int | None = None):
        column = self.tokens[self.index][2] if column is None else column
        raise syntax_error(f"{self.what} '{self.text}', column {column}: {message}")

    def peek(self) -> tuple:
        return self.tokens[self.index]

    def take(self) -> tuple:
        token = self.tokens[self.index]
        self.index += 1
        return token

    def accept(self, operator_text: str) -> bool:
        kind, token_text, _ = self.peek()
        if kind == "operator" and token_text == operator_text:
            self.index += 1
            return True
        return False

    def accept_word(self, word: str) -> bool:
        kind, token_text, _ = self.peek()
        if kind == "name" and token_text == word:
            self.index += 1
            return True
        return False

    def expect(self, operator_text: str):
        if not self.accept(operator_text):
            found = self.peek()[1] or "the end"
            self.fail(f"expected '{operator_text}', found '{found}'")

    def expect_end(self):
        if self.peek()[0] != "end":
            self.fail(f"unexpected '{self.peek()[1]}'. Write every product with '*'")

    def parse_all(self):
        if self.peek()[0] == "end":
            self.fail(f"the {self.what} is empty")
        result = self.parse_expression()
        self.expect_end()
        return result

    def parse_equation(self) -> tuple:
        if self.peek()[0] == "end":
            self.fail("the equation is empty")
        left = self.parse_expression()
        if not self.accept("="):
            self.fail("an equation needs '=' between its two sides")
        right = self.parse_expression()
        self.expect_end()
        return left, right

    def parse_expression(self):
        result = self.parse_conjunction()
        while self.accept_word("or"):
            result = graph.logical_or(result, self.parse_conjunction())
        return result

    def parse_conjunction(self):
        result = self.parse_comparison()
        while self.accept_word("and"):
            result = graph.logical_and(result, self.parse_comparison())
        return result

    def parse_comparison(self):
        result = self.parse_sum()
        kind, token_text, _ = self.peek()
        if kind == "operator" and token_text in comparison_kinds:
            self.index += 1
            return graph.build(comparison_kinds[token_text], (result, self.parse_sum()))
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
        if self.accept("^") or self.accept("**"):
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
            derivative = self.parse_derivative(token_text, column)
            if derivative is not None:
                return derivative
            if self.accept("("):
                return self.parse_call(token_text, column)
            return self.resolve(token_text, column)
        if kind == "operator" and token_text == "(":
            result = self.parse_expression()
            self.expect(")")
            return result
        if kind == "operator" and token_text == "[":
            return self.parse_list(column)
        self.index -= 1
        self.fail(
            f"expected a number, a quantity, a name, '(' or '[', found '{token_text or 'the end'}'"
        )

    def parse_list(self, column: int):
        items = [self.parse_expression()]
        while self.accept(","):
            items.append(self.parse_expression())
        self.expect("]")
        if self.scope is not None and len(items) != self.scope.dimension:
            self.fail(
                f"a vector here has {self.scope.dimension} components, one per dimension, "
                f"and this list has {len(items)}",
                column,
            )
        return graph.list_of(items)

    def parse_derivative(self, name: str, column: int):
        """Return the derivative d<field>/d<coordinate> or d2<field>/dt2 that starts with the
        name, or None when the name starts no derivative of a field of the scope."""
        if self.scope is None or len(name) < 2 or name[0] != "d":
            return None
        if self.index + 1 >= len(self.tokens):
            return None
        slash, denominator = self.tokens[self.index], self.tokens[self.index + 1]
        if slash[:2] != ("operator", "/") or denominator[0] != "name":
            return None
        if len(denominator[1]) < 2 or denominator[1][0] != "d":
            return None
        second = name[1] == "2" and self.scope.is_field(name[2:])
        field_name = name[2:] if second else name[1:]
        if not self.scope.is_field(field_name):
            return None
        coordinate = denominator[1][1:]
        if second:
            if not coordinate.endswith("2"):
                self.fail(f"write the second derivative as d2{field_name}/d{coordinate}2", column)
            coordinate = coordinate[:-1]
        self.index += 2
        return self.scope.derivative_of(field_name, coordinate, 2 if second else 1)

    def parse_call(self, written: str, column: int):
        name = aliases.get(written, written)
        if name not in calls and name not in space_calls:
            nearest = difflib.get_close_matches(written, [*calls, *space_calls, *aliases])
            hint = f". Did you mean {', '.join(nearest)}?" if nearest else ""
            self.fail(f"unknown function '{written}'{hint}", column)
        arguments = []
        if not self.accept(")"):
            arguments.append(self.parse_expression())
            while self.accept(","):
                arguments.append(self.parse_expression())
            self.expect(")")
        if self.scope is not None and name in space_calls:
            argument_count = space_calls[name]
        elif name in calls:
            function, argument_count = calls[name]
        else:
            self.fail(
                f"{written} takes the space of an equation, its dimension and coordinates, "
                f"which only the equations of general_form_PDE have",
                column,
            )
        if len(arguments) != argument_count:
            self.fail(f"{written} takes {argument_count} arguments, got {len(arguments)}", column)
        if self.scope is not None and name in space_calls:
            return self.scope.call(name, arguments)
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


class formula_text(str):
    """The text of a formula with its graph `node`, as the validated input tree holds it."""

    def __new__(cls, text: str, node):
        formula = super().__new__(cls, text)
        formula.node = node
        return formula

    def __reduce__(self):
        return (formula_text, (str(self), self.node))


def parse_in_scope(text: str, scope, equation: bool):
    """Return the graph of an expression, or the two sides of an equation, in an equation_scope."""
    what = "equation" if equation else "expression"
    if not isinstance(text, str):
        raise input_error(f"an {what} must be a string, got {text!r}")
    reader = parser(text, scope, scope.coordinate_system, scope)
    try:
        return reader.parse_equation() if equation else reader.parse_all()
    except syntax_error:
        raise
    except input_error as error:
        raise type(error)(f"{what} '{text}': {error}") from None


# The equations of general_form_PDE


def rank_name(rank: int) -> str:
    return ("a scalar", "a vector", "a tensor")[rank]


class equation_scope(Mapping):
    """The names and the space of the equations of general_form_PDE.

    The equations carry no units, so the fields, the coordinates, the time and the definitions
    are dimensionless. A name resolves, in this order, to:

    1. a field of `fields`, which maps each name to its rank, 0 or 1,
    2. a definition of `definitions`, in their order: a number is a named constant, and a text a
       named expression, which reads the fields and the definitions before it,
    3. a coordinate of the problem: x, y and z, up to the dimension, or r and z in axisymmetric
       coordinates. Any other of the letters x, y, z and r is refused,
    4. `t` for the time, `pi`, `e`, `I` for the identity tensor and `n` for the outward unit
       normal of a boundary,
    5. any other name, such as `eigenvalue`, a property, a registered function or an element
       field, which the engine resolves. A derivative of such a name is refused, except for
       `eigenvalue`, which is constant in space.

    A component of a vector field, such as `U_x`, resolves through its field. The engine names
    the components of a vector field U by `variables`: U_x, U_y and U_z, or U_r and U_z in
    axisymmetric coordinates. In spherical coordinates the one coordinate is x. `reads` collects
    every name that a parsed text reads, and `externals` the names of step 5.
    """

    def __init__(
        self,
        fields: dict,
        dimension: int,
        coordinate_system: str = "cartesian",
        definitions: Mapping | None = None,
        flux_names=("normal_flux",),
    ):
        self.fields = dict(fields)
        self.dimension = dimension
        self.axisymmetric = coordinate_system == "axisymmetric"
        self.coordinate_system = "axisymmetric" if self.axisymmetric else "cartesian"
        self.coordinates = units.position_names[self.coordinate_system][:dimension]
        self.flux_names = tuple(flux_names)
        self.variables = {
            name: [name]
            if rank == 0
            else [
                units.component_name(name, (index,), self.coordinate_system)
                for index in range(dimension)
            ]
            for name, rank in self.fields.items()
        }
        self.reads = set()
        self.externals = set()
        self.names = {}
        self.pending = set(definitions or {})
        for name, value in (definitions or {}).items():
            self.names[name] = self.definition(name, value)
            self.pending.discard(name)

    def definition(self, name: str, value):
        """Return the node of one definition: a named constant or a named expression."""
        if isinstance(value, str):
            try:
                return parse_in_scope(value, self, equation=False)
            except input_error as error:
                raise type(error)(f"the definition '{name}': {error}") from None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise input_error(
                f"the definition {name} = {value!r} is not a number or an expression (a text)"
            )
        if not math.isfinite(value):
            raise input_error(f"the definition {name} = {value!r} is not a finite number")
        return graph.build(
            "constant", name=name, value=float(value), rank=0, physical_dimension=dimensionless
        )

    def with_definitions(self, definitions: Mapping) -> "equation_scope":
        """Return a scope with further definitions, which read the names of this one."""
        scope = object.__new__(equation_scope)
        scope.__dict__.update(self.__dict__)
        scope.names = dict(self.names)
        scope.reads = set()
        scope.externals = self.externals
        scope.pending = set(definitions)
        for name, value in definitions.items():
            scope.names[name] = scope.definition(name, value)
            scope.pending.discard(name)
        return scope

    def __contains__(self, name) -> bool:
        if name in self.fields or name in self.names:
            return True
        return self.component_of(name) is None

    def __getitem__(self, name):
        if name in self.pending:
            raise input_error(
                f"'{name}' is defined after this expression, or is this expression itself. "
                f"Define a name before the expressions that read it"
            )
        self.reads.add(name)
        if name in self.fields:
            return self.field(name)
        if name in self.names:
            return self.names[name]
        if name in self.coordinates:
            return self.coordinate(self.coordinates.index(name))
        # The engine reads x, y and z as its coordinates whatever the problem calls them, so such
        # a name is refused here rather than passed on as a constant.
        if name in ("x", "y", "z", "r"):
            raise input_error(
                f"'{name}' is not a coordinate of this problem, whose coordinates are "
                f"{', '.join(self.coordinates)}"
            )
        builtin = {
            "t": lambda: graph.build("time", rank=0, physical_dimension=dimensionless),
            "pi": lambda: graph.number(math.pi),
            "e": lambda: graph.number(math.e),
            "I": graph.identity,
            "n": graph.normal,
        }
        if name in builtin:
            return builtin[name]()
        if name != "eigenvalue":
            self.externals.add(name)
        return graph.build("constant", name=name, rank=0, physical_dimension=dimensionless)

    def __iter__(self):
        return iter([*self.fields, *self.names])

    def __len__(self) -> int:
        return len(self.fields) + len(self.names)

    def component_of(self, name: str):
        """Return the vector field and the component of a name such as U_x, or None."""
        split = units.split_component(name, self.coordinate_system)
        if split is None or self.fields.get(split[0]) != 1 or len(split[1]) != 1:
            return None
        if split[1][0] >= self.dimension:
            return None
        return split

    def is_field(self, name: str) -> bool:
        """Whether name is a field or a component of a vector field."""
        return name in self.fields or self.component_of(name) is not None

    def field(self, name: str):
        return graph.build(
            "field", name=name, rank=self.fields[name], physical_dimension=dimensionless
        )

    def coordinate(self, direction: int):
        return graph.build(
            "position_component",
            direction=direction,
            rank=0,
            physical_dimension=dimensionless,
        )

    def variable(self, name: str, component: tuple) -> str:
        """The engine's name of one scalar component of a field."""
        return self.variables[name][component[0] if component else 0]

    # The derivatives along the coordinates

    def rate_along(self, leaf, direction: int):
        """Return the derivative of a scalar leaf along a coordinate, or None where it is zero."""
        kind = leaf.kind
        if kind == "field_component":
            return graph.build(
                "gradient_component",
                name=leaf.name,
                component=leaf.component,
                direction=direction,
                rank=0,
                physical_dimension=leaf.dimension,
            )
        if kind == "position_component":
            return graph.number(1.0) if leaf.direction == direction else None
        if kind == "gradient_component":
            raise input_error(
                f"this needs a second derivative of {self.variable(leaf.name, leaf.component)} "
                f"outside a divergence. Second derivatives stand only in div(... grad(...)) or "
                f"laplacian(...), which are integrated by parts, and a fourth-order equation is "
                f"written as two second-order ones"
            )
        if kind == "constant" and leaf.name in self.externals:
            raise input_error(
                f"'{leaf.name}' would be differentiated, and it is not a field, a coordinate or "
                f"a definition. Write it as a definition of the coordinates and the fields, or "
                f"keep it outside derivatives"
            )
        return None

    def partial(self, node, direction: int):
        """Return the derivative of node along a coordinate, of the rank of node."""

        def walk(components):
            if isinstance(components, list):
                return graph.list_of([walk(entry) for entry in components])
            return calculus.derivative_along(
                components, lambda leaf: self.rate_along(leaf, direction)
            )

        return walk(graph.expand(node, self.dimension))

    def refuse_hoop_terms(self, what: str):
        if self.axisymmetric:
            raise input_error(
                f"{what} in axisymmetric coordinates has hoop terms that the equation language "
                f"does not write. Use the axisymmetric physics"
            )

    def gradient(self, node):
        directions = range(self.dimension)
        if node.rank == 0:
            return graph.list_of([self.partial(node, column) for column in directions])
        if node.rank == 1:
            self.refuse_hoop_terms("the gradient of a vector")
            return graph.list_of(
                [
                    graph.list_of([self.partial(node[row], column) for column in directions])
                    for row in directions
                ]
            )
        raise input_error("grad takes a scalar or a vector, and here it has a tensor")

    def divergence(self, node):
        """Return the divergence of node: a divergence node when node holds a derivative of a
        field, the flux of a term of the equation, and else the divergence itself."""
        if node.rank not in (1, 2):
            raise input_error(
                f"div takes a vector or a tensor, and here it has {rank_name(node.rank)}. "
                f"The gradient is grad"
            )
        if any(leaf.kind == "gradient_component" for leaf in graph.leaves(node)):
            return graph.divergence(node, dimensionless)
        directions = range(self.dimension)
        if node.rank == 1:
            result = sum(
                (self.partial(node[index], index) for index in directions), graph.number(0)
            )
            if self.axisymmetric:
                result = result + node[0] / self.coordinate(0)
            return result
        self.refuse_hoop_terms("the divergence of a tensor")
        return graph.list_of(
            [
                sum(
                    (self.partial(node[row, column], column) for column in directions),
                    graph.number(0),
                )
                for row in directions
            ]
        )

    def curl(self, node):
        along = self.partial
        if node.rank == 1 and self.dimension == 3:
            return graph.list_of(
                [
                    along(node[2], 1) - along(node[1], 2),
                    along(node[0], 2) - along(node[2], 0),
                    along(node[1], 0) - along(node[0], 1),
                ]
            )
        if node.rank == 1 and self.dimension == 2:
            return along(node[1], 0) - along(node[0], 1)
        if node.rank == 0 and self.dimension == 2:
            return graph.list_of([along(node, 1), -along(node, 0)])
        raise input_error("curl takes a vector in two or three dimensions, or a scalar in two")

    def cross(self, first, second):
        if first.rank != 1 or second.rank != 1:
            raise input_error("cross takes two vectors")
        if self.dimension == 3:
            return graph.list_of(
                [
                    first[1] * second[2] - first[2] * second[1],
                    first[2] * second[0] - first[0] * second[2],
                    first[0] * second[1] - first[1] * second[0],
                ]
            )
        if self.dimension == 2:
            return first[0] * second[1] - first[1] * second[0]
        raise input_error("cross is defined in two and three dimensions")

    def call(self, name: str, arguments: list):
        """Return the call of `space_calls` of name on its arguments."""
        if name == "grad":
            return self.gradient(arguments[0])
        if name in ("grad_x", "grad_y", "grad_z"):
            direction = "xyz".index(name[-1])
            if direction >= self.dimension:
                raise input_error(f"{name} is not defined in {self.dimension} dimension(s)")
            return self.partial(arguments[0], direction)
        if name == "div":
            return self.divergence(arguments[0])
        if name == "laplacian":
            return self.divergence(self.gradient(arguments[0]))
        if name == "curl":
            return self.curl(arguments[0])
        return self.cross(*arguments)

    def derivative_of(self, name: str, coordinate: str, order: int):
        """Return the derivative written d<name>/d<coordinate>, or d2<name>/dt2 for order 2."""
        component = self.component_of(name)
        node = self[name] if component is None else self[component[0]][component[1]]
        if coordinate == "t":
            if component is not None:
                raise input_error(
                    f"write the time derivative of the vector field, d{component[0]}/dt, not "
                    f"of one of its components"
                )
            return graph.rate(node, order, dimensionless)
        if order == 2:
            raise input_error(
                f"second derivatives in space are written with div, grad and laplacian, for "
                f"example laplacian({name})"
            )
        if coordinate == "n":
            if node.rank != 0:
                raise input_error(f"d{name}/dn is defined for a scalar field")
            return sum(
                (
                    graph.normal_component(direction) * self.partial(node, direction)
                    for direction in range(self.dimension)
                ),
                graph.number(0),
            )
        if coordinate in self.coordinates:
            return self.partial(node, self.coordinates.index(coordinate))
        if coordinate in ("x", "y", "z", "r"):
            raise input_error(
                f"d{name}/d{coordinate}: {coordinate} is not a coordinate of this problem"
            )
        listed = ", ".join(f"d{name}/d{letter}" for letter in ("t", *self.coordinates))
        raise input_error(
            f"'d{name}/d{coordinate}' is not a derivative: the derivatives are {listed} "
            f"and d{name}/dn"
        )

    def varies_in_space(self, node) -> bool:
        """Whether node reads a field, a coordinate, the normal or an external name."""
        varying = {
            "field",
            "field_component",
            "gradient_component",
            "position",
            "position_component",
            "normal",
            "normal_component",
        }
        return any(
            leaf.kind in varying or (leaf.kind == "constant" and leaf.name in self.externals)
            for leaf in graph.leaves(node)
        )

    def antiderivative(self, coefficient, unknown):
        """Return the storage whose derivative by each component of unknown is coefficient, so
        that its rate is coefficient*du/dt, or None for a zero coefficient."""
        if is_zero(coefficient):
            return None
        if unknown.rank == 0:
            return graph.antiderivative(coefficient, unknown)
        return graph.list_of(
            [
                graph.antiderivative(
                    coefficient, graph.field_component(unknown.name, (index,), unknown.dimension)
                )
                for index in range(self.dimension)
            ]
        )


class sorted_equation:
    """One equation d2 d2u/dt2 + d1 du/dt + div(F) + r = 0, read from its text.

    `capacity` is d1 and `mass` d2, the factors of the time derivatives of `time_field`, `flux`
    is F and `rest` is r, the terms that are neither time derivatives nor divergences of a
    flux. `flux_fields` names the fields whose gradients the flux reads.
    """

    def __init__(self, text: str):
        self.text = text
        self.rank = None
        self.time_field = None
        self.capacity = graph.number(0.0)
        self.mass = graph.number(0.0)
        self.flux = None
        self.rest = None
        self.flux_fields = set()

    def take_rank(self, rank: int):
        if self.rank is None:
            self.rank = rank
        elif self.rank != rank:
            raise input_error(
                f"the terms of an equation must be of one kind, and here "
                f"{rank_name(self.rank)} meets {rank_name(rank)}"
            )


def holds_term(node, memo: dict) -> bool:
    """Whether node holds a time derivative or the divergence of a flux."""
    if node not in memo:
        memo[node] = node.kind in ("rate", "divergence") or any(
            holds_term(operand, memo) for operand in node.operands
        )
    return memo[node]


def collect_terms(node, factor, terms: list, memo: dict):
    """Append each term of node as (kind, factor, part): a "rate", a "flux" or a "rest"."""
    if not holds_term(node, memo):
        terms.append(("rest", factor, node))
        return
    kind, operands = node.kind, node.operands
    if kind == "add":
        for operand in operands:
            collect_terms(operand, factor, terms, memo)
    elif kind == "negate":
        collect_terms(operands[0], -factor, terms, memo)
    elif kind == "multiply":
        first, second = (holds_term(operand, memo) for operand in operands)
        if first and second:
            raise input_error(
                "a product of two terms with time derivatives or divergences cannot be sorted. "
                "Make each of them a term of its own"
            )
        term, coefficient = operands if first else operands[::-1]
        collect_terms(term, factor * coefficient, terms, memo)
    elif kind == "divide":
        if holds_term(operands[1], memo):
            raise input_error("a time derivative or a divergence does not divide")
        collect_terms(operands[0], factor / operands[1], terms, memo)
    elif kind == "rate":
        terms.append(("rate", factor, node))
    elif kind == "divergence":
        terms.append(("flux", factor, operands[0]))
    else:
        raise input_error(
            "a time derivative, div or laplacian stands only as a term of the equation, "
            "multiplied or divided by coefficients, not inside a function or another operator"
        )


def read_equation(text: str, scope: equation_scope) -> sorted_equation:
    """Return the terms of one equation, sorted by the rules of the equation language."""
    left, right = parse_in_scope(text, scope, equation=True)
    equation = sorted_equation(text)
    try:
        terms, memo = [], {}
        collect_terms(left, graph.number(1.0), terms, memo)
        collect_terms(right, graph.number(-1.0), terms, memo)
        for kind, factor, part in terms:
            if kind != "rest" and factor.rank != 0:
                raise input_error(
                    f"a time derivative or a divergence is multiplied by a scalar only, and here "
                    f"the factor is {rank_name(factor.rank)}"
                )
            if kind == "rate":
                field_name = part.operands[0].name
                if equation.time_field not in (None, field_name):
                    raise input_error(
                        f"an equation has the time derivative of one field only, and this one "
                        f"has d{equation.time_field}/dt and d{field_name}/dt"
                    )
                equation.take_rank(part.rank)
                equation.time_field = field_name
                if part.value == 1:
                    equation.capacity = equation.capacity + factor
                else:
                    equation.mass = equation.mass + factor
            elif kind == "flux":
                if scope.varies_in_space(factor):
                    raise input_error(
                        f"a div(...) or laplacian(...) term is multiplied only by a factor that "
                        f"is constant in space, here '{factor}'. Move the factor inside, as in "
                        f"div(k*grad(u)), which is the conservative form"
                    )
                equation.take_rank(part.rank - 1)
                equation.flux_fields |= {
                    leaf.name for leaf in graph.leaves(part) if leaf.kind == "gradient_component"
                }
                term = factor * part
                equation.flux = term if equation.flux is None else equation.flux + term
            else:
                term = factor * part
                if term.rank == 0 and is_zero(term):
                    # "... = 0" holds for an equation of any kind.
                    continue
                equation.take_rank(term.rank)
                equation.rest = term if equation.rest is None else equation.rest + term
        if equation.rank is None:
            raise input_error("the equation has no terms")
    except input_error as error:
        raise type(error)(f"the equation '{text}': {error}") from None
    return equation


def match_fields(equations: list, keys: list, fields: list) -> dict:
    """Return the index of the equation of each field.

    A key names the field of its equation. Otherwise an equation with a time derivative governs
    the field of that derivative, an equation whose flux reads the gradient of one field without
    an equation governs that field, a rule that repeats while it assigns fields, and a single
    equation left over governs the single field left over.
    """
    equation_of = {}
    field_of = [None] * len(equations)

    def assign(index: int, field_name: str):
        if field_name in equation_of:
            raise input_error(
                f"the equations '{equations[equation_of[field_name]].text}' and "
                f"'{equations[index].text}' both govern {field_name}. Give the field of each "
                f'equation as a key: equations={{"field": "equation", ...}}'
            )
        field_of[index] = field_name
        equation_of[field_name] = index

    for index, key in enumerate(keys):
        if key:
            assign(index, key)
    for index, equation in enumerate(equations):
        if field_of[index] is None and equation.time_field is not None:
            assign(index, equation.time_field)
    changed = True
    while changed:
        changed = False
        for index, equation in enumerate(equations):
            if field_of[index] is not None:
                continue
            candidates = [name for name in sorted(equation.flux_fields) if name not in equation_of]
            if len(candidates) == 1:
                assign(index, candidates[0])
                changed = True
    open_equations = [index for index in range(len(equations)) if field_of[index] is None]
    open_fields = [name for name in fields if name not in equation_of]
    if len(open_equations) == 1:
        assign(open_equations[0], open_fields[0])
    elif open_equations:
        listed = ", ".join(f"'{equations[index].text}'" for index in open_equations)
        raise input_error(
            f"the equations {listed} could govern any of the fields {', '.join(open_fields)}. "
            f'Give the field of each equation as a key: equations={{"field": "equation", ...}}'
        )
    return equation_of


def read_equations(texts: list, keys: list, scope: equation_scope) -> dict:
    """Return the conservation law of each field of scope, in the order of the fields.

    keys holds the field of each equation, or an empty text for an equation that
    `match_fields` matches. The law of a field u holds the storage whose derivative by u is
    d1, an antiderivative, so that its rate is d1 du/dt, the inertia whose derivative by u is
    d2, the flux F and the source s = -r.
    """
    fields = list(scope.fields)
    if len(texts) != len(fields):
        raise input_error(
            f"there are {len(fields)} field(s) and {len(texts)} equation(s). Give one equation "
            f"for each field"
        )
    equations = [read_equation(text, scope) for text in texts]
    equation_of = match_fields(equations, keys or [""] * len(texts), fields)
    laws = {}
    for name, rank in scope.fields.items():
        equation = equations[equation_of[name]]
        if equation.rank != rank:
            raise input_error(
                f"the equation '{equation.text}' is {rank_name(equation.rank)} equation, and it "
                f"governs {name}, which is {rank_name(rank)} field"
            )
        if equation.time_field not in (None, name):
            raise input_error(
                f"the equation '{equation.text}' governs {name} and has the time derivative of "
                f"{equation.time_field}. The time derivative of a field belongs to its own "
                f"equation"
            )
        unknown = scope.field(name)
        laws[name] = conservation_law(
            name,
            unknown,
            storage=scope.antiderivative(equation.capacity, unknown),
            inertia=scope.antiderivative(equation.mass, unknown),
            flux=equation.flux,
            source=None if equation.rest is None else -equation.rest,
        )
    return laws


def sample_values(nodes, trial: int) -> dict:
    """Return a number from 0.5 to 1.5 for each leaf of nodes that is not a number, the same in
    every run for one leaf and one trial. A constant with a value takes its value."""
    values = {}
    for node in nodes:
        for leaf in graph.leaves(node):
            if leaf.kind == "number":
                continue
            if leaf.kind == "constant" and leaf.value is not None:
                values[leaf] = graph.number(leaf.value, leaf.dimension)
                continue
            seed = zlib.crc32(f"{leaf}|{trial}".encode())
            values[leaf] = graph.number(0.5 + (seed % 1000) / 1000.0, leaf.dimension)
    return values


def agree(first, second) -> bool:
    """Whether two scalar expressions agree at three sets of sample values of their leaves."""
    for trial in range(3):
        values = sample_values((first, second), trial)
        try:
            first_value = graph.substitute(first, values)
            second_value = graph.substitute(second, values)
        except input_error:
            return False
        if first_value.kind != "number" or second_value.kind != "number":
            return first == second
        size = max(1.0, abs(first_value.value), abs(second_value.value))
        if abs(first_value.value - second_value.value) > 1.0e-9 * size:
            return False
    return True


def read_condition(text: str, scope: equation_scope, laws: dict) -> list:
    """Return the scalar conditions of a boundary condition text, each as (field, component,
    flux, value).

    A field alone on the left prescribes its value, and flux is False. A flux on the left
    prescribes the normal flux n . F of the equation of a field along the outward unit normal
    n, and flux is True. The flux is written out with d<field>/dn, which must equal n . F, or
    named by a name of scope.flux_names, whose field is told by the rank of the value among the
    fields whose equations have a flux. value is a scalar expression.
    """
    left, right = parse_in_scope(text, scope, equation=True)
    try:
        return condition_terms(left, right, scope, laws)
    except input_error as error:
        raise type(error)(f"the boundary condition '{text}': {error}") from None


def condition_terms(left, right, scope: equation_scope, laws: dict) -> list:
    if left.kind == "field" or (left.kind == "component" and left.operands[0].kind == "field"):
        name = left.name if left.kind == "field" else left.operands[0].name
        if left.rank != right.rank:
            raise input_error(
                f"the field is {rank_name(left.rank)} and the value {rank_name(right.rank)}"
            )
        if left.kind == "component":
            components = [left.component]
        else:
            components = [()] if left.rank == 0 else [(index,) for index in range(scope.dimension)]
        values = graph.expand(right, scope.dimension)
        values = values if isinstance(values, list) else [values]
        for value in values:
            if any(leaf.kind == "gradient_component" for leaf in graph.leaves(value)):
                raise input_error(
                    "the value of a field on a boundary may not depend on derivatives. A "
                    "condition on the flux has the flux on the left"
                )
        return [
            (name, component, False, value)
            for component, value in zip(components, values, strict=True)
        ]
    named = left.kind == "constant" and left.value is None and left.name in scope.flux_names
    written = graph.expand(left, scope.dimension) if left.rank == 0 else left
    normal = any(leaf.kind == "normal_component" for leaf in graph.leaves(written))
    if not named and not normal:
        raise input_error(
            f"the left side is a field (a value on the boundary), the normal flux written with "
            f"d<field>/dn, or one of the names of the normal flux "
            f"({', '.join(scope.flux_names)})"
        )
    has_flux = [name for name, law in laws.items() if not is_zero(law.flux)]
    if named:
        candidates = [name for name in has_flux if scope.fields[name] == right.rank]
        if len(candidates) != 1:
            told = (
                "is the flux of no field with a divergence term"
                if not candidates
                else f"could be the flux of {', '.join(candidates)}"
            )
            raise input_error(
                f"'{left.name}' with {rank_name(right.rank)} value {told}. Write the flux out "
                f"with d<field>/dn"
            )
        name = candidates[0]
    else:
        names = {leaf.name for leaf in graph.leaves(written) if leaf.kind == "gradient_component"}
        if len(names) != 1:
            raise input_error("a flux condition is the flux of one field")
        (name,) = names
        if left.rank != 0 or scope.fields[name] != 0:
            raise input_error("a flux written with d<field>/dn is the flux of a scalar field")
    law = laws[name]
    if name not in has_flux:
        raise input_error(
            f"the equation of {name} has no divergence term, so it has no flux to prescribe"
        )
    if right.rank != scope.fields[name]:
        raise input_error(
            f"the value is {rank_name(right.rank)}, and the flux of {name} is "
            f"{rank_name(scope.fields[name])}"
        )
    if not named:
        flux = graph.expand(law.flux, scope.dimension)
        normal_flux = sum(
            (graph.normal_component(index) * flux[index] for index in range(scope.dimension)),
            graph.number(0.0),
        )
        if not agree(written, normal_flux):
            raise input_error(
                f"the left side is not the normal flux of the equation of {name}, which is "
                f"n . F = {normal_flux}, with n the outward normal. Write that, or the name "
                f"{scope.flux_names[0]}"
            )
    values = graph.expand(right, scope.dimension)
    if right.rank == 0:
        return [(name, (), True, values)]
    return [(name, (index,), True, value) for index, value in enumerate(values)]


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
            raise input_error(f"{path}: unknown key '{key}'." + suggestion(key, allowed))


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
