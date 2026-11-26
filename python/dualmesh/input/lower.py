"""Lowers the expressions of the input and of general_form_PDE onto the expressions of the engine.

This module holds the one printer of the engine's expression language, which only it writes.
A formula of the input, such as `1/([0.0375 m*K/W] + [2.165e-4 m/W]*temperature)`, is parsed
into an expression graph, which checks its names and its dimension. The validated tree holds
that graph beside the text (`dualmesh.model.parse.formula_text`), so a formula parses once. The
graph is then written in the syntax of the engine's expression parser:

- every number, with every quantity in square brackets, and every constant with a value, in SI;
- every field by the engine's name of its variable, so `displacement_x` becomes `disp_x`, and
  the derivative of a field along a direction as `grad_x(...)`, `grad_y(...)` or `grad_z(...)`;
- a state of a model by its name, the name of the property that holds it;
- a constant without a value, such as a property or `eigenvalue`, by its name;
- `time` as `t`;
- the position as `x`, `y` and `z`, which the engine reads as the radius and the axial
  coordinate in axisymmetric problems, as `r` and `z` are in the input;
- `conditional` as `if(...)`, the comparisons with their operators, and every other function by
  its name in the engine.

The engine differentiates the expression it reads, so the graph need not hold derivatives.

The engine's names of the variables come from `variables`, which maps each field of the input
to the names of its variables, one per component, as build.py makes it from the physics and the
models of the problem. Without it, a component of the displacement or the velocity takes the
Cartesian names of `engine_component_names`.
"""

import importlib

from dualmesh.input import units
from dualmesh.input.units import input_error
from dualmesh.model import parse
from dualmesh.model.dimensions import comparisons, dimension_error

# import_module returns the module itself, which dualmesh.model hides behind the class of the
# same name.
graph = importlib.import_module("dualmesh.model.expression")

# The engine's names of the components of the vector fields of the input, in Cartesian
# coordinates, for a formula written before the physics name their variables.
engine_component_names = {
    "displacement": ("disp_x", "disp_y", "disp_z"),
    "velocity": ("velocity_x", "velocity_y", "velocity_z"),
}
engine_position_names = ("x", "y", "z")

# The functions of the graph with their names in the engine.
functions = {kind: kind for kind in graph.scalar_functions}
functions |= {"maximum": "max", "minimum": "min", "atan2": "atan2", "hypot": "hypot"}
functions["conditional"] = "if"


def engine_formula(
    text: str,
    unit: str,
    path: str,
    coordinate_system: str = "cartesian",
    variables: dict | None = None,
    symbols=None,
) -> str:
    """The engine's expression of a formula, whose dimension must be that of `unit`.

    `path` is the key of the formula in the input, which every message names. `symbols` maps
    the names outside the vocabulary that the formula may read to their expressions, such as the
    unknown of a model.
    """
    try:
        node = formula_node(text, symbols, coordinate_system)
    except input_error as error:
        raise input_error(f"{path}: {error}") from None
    _, expected = units.parse_unit(unit)
    if node.rank != 0:
        raise input_error(f"{path}: the formula '{text}' is not a scalar.")
    if node.dimension != expected and not (node.kind == "number" and node.value == 0.0):
        given = units.SI_unit_text(node.dimension)
        raise dimension_error(
            f"{path}: the formula '{text}' has the unit {given}, and the property takes {unit}."
        )
    return _write(node, text, path, variables)


def formula_node(text: str, symbols, coordinate_system: str):
    """The graph of a formula, with each name of `symbols` read as its entry.

    The validated tree holds the graph of each formula, parsed with the declared names of the
    tree as fields, so a declared name that `symbols` holds takes its entry by substitution.
    A text without a graph, such as one of the Python interface, is parsed here.
    """
    node = getattr(text, "node", None)
    if node is None:
        return parse(text, symbols, coordinate_system)
    replacements = {}
    for leaf in graph.leaves(node):
        if leaf.kind != "field":
            continue
        if symbols is not None and leaf.name in symbols:
            replacements[leaf] = symbols[leaf.name]
        elif leaf.name not in units.standard_names():
            raise input_error(f"formula '{text}': unknown name '{leaf.name}'")
    try:
        return graph.substitute(node, replacements) if replacements else node
    except input_error as error:
        raise type(error)(f"formula '{text}': {error}") from None


def write_expression(node, path: str, variables: dict | None = None) -> str:
    """The engine's expression of a scalar node of the model language, such as a derivative."""
    return _write(node, node, path, variables)


def _variable(name: str, component: tuple, variables: dict | None) -> str | None:
    """The engine's name of one component of a field, or None for a name it does not know."""
    if variables is not None and name in variables:
        names = variables[name]
        if not component and len(names) == 1:
            return names[0]
        if len(component) == 1 and component[0] < len(names):
            return names[component[0]]
        return None
    if len(component) == 1 and name in engine_component_names:
        return engine_component_names[name][component[0]]
    if not component:
        return name
    return None


def _write(node, source, path: str, variables: dict | None = None) -> str:
    """The text of node. source is the formula, or the node, that messages name."""
    kind, operands = node.kind, node.operands
    write = lambda operand: _write(operand, source, path, variables)  # noqa: E731
    if kind == "number" or (kind == "constant" and node.value is not None):
        return f"({node.value!r})"
    if kind == "constant":
        return node.name
    if kind == "time":
        return "t"
    if kind == "position_component":
        return engine_position_names[node.direction]
    if kind == "field" and node.rank == 0:
        return _variable(node.name, (), variables) or node.name
    if kind in ("state", "state_component") and node.rank == 0 and not node.component:
        return node.name
    if kind == "gradient_component":
        variable = _variable(node.name, node.component, variables)
        if variable is not None:
            return f"grad_{engine_position_names[node.direction]}({variable})"
    if kind == "field_component":
        variable = _variable(node.name, node.component, variables)
        if variable is not None:
            return variable
        return units.component_name(node.name, node.component)
    if kind == "component" and operands[0].kind == "position":
        return engine_position_names[node.component[0]]
    if kind == "component" and operands[0].kind == "field":
        variable = _variable(operands[0].name, node.component, variables)
        if variable is not None:
            return variable
    if kind == "add":
        return "(" + " + ".join(write(operand) for operand in operands) + ")"
    if kind == "negate":
        return f"(-{write(operands[0])})"
    if kind == "multiply":
        return "(" + "*".join(write(operand) for operand in operands) + ")"
    if kind == "divide":
        return f"({write(operands[0])}/{write(operands[1])})"
    if kind == "power":
        return f"({write(operands[0])}^{write(operands[1])})"
    if kind in comparisons:
        return f"({write(operands[0])} {comparisons[kind]} {write(operands[1])})"
    if kind in functions and node.rank == 0:
        return f"{functions[kind]}({', '.join(write(operand) for operand in operands)})"
    raise input_error(
        f"{path}: the formula '{source}' uses {kind}, which the engine does not evaluate. "
        "A formula is a function of the fields, the states, the position and the time."
    )
