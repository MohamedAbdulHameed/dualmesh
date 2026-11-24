"""Lowers a formula of the input onto the expressions of the engine.

A formula of the input, such as `1/([0.0375 m*K/W] + [2.165e-4 m/W]*temperature)`, is parsed
into an expression graph, which checks its names and its dimension. The graph is then written
in the syntax of the engine's expression parser:

- every number, with every quantity in square brackets, in SI;
- every field by the engine's name of its variable, so `displacement_x` becomes `disp_x`;
- `time` as `t`;
- the position as `x`, `y` and `z`, which the engine reads as the radius and the axial
  coordinate in axisymmetric problems, as `r` and `z` are in the input.

The engine differentiates the expression it reads, so the graph need not hold derivatives.
"""

from dualmesh.input import units
from dualmesh.input.units import input_error
from dualmesh.model import parse
from dualmesh.model.dimensions import dimension_error

# The engine's names of the components of the vector fields of the input.
engine_component_names = {
    "displacement": ("disp_x", "disp_y", "disp_z"),
    "velocity": ("velocity_x", "velocity_y", "velocity_z"),
}
engine_position_names = ("x", "y", "z")

functions = {"exp", "log", "sqrt", "sin", "cos", "tanh", "abs"}
pairs = {"maximum": "max", "minimum": "min"}
comparisons = {"less": "<", "greater": ">"}


def engine_formula(text: str, unit: str, path: str, coordinate_system: str = "cartesian") -> str:
    """The engine's expression of a formula, whose dimension must be that of `unit`.

    `path` is the key of the formula in the input, which every message names.
    """
    try:
        node = parse(text, None, coordinate_system)
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
    return _write(node, text, path)


def write_expression(node, path: str) -> str:
    """The engine's expression of a node of the model language, such as a derivative."""
    return _write(node, str(node), path)


def _write(node, text: str, path: str) -> str:
    kind, operands = node.kind, node.operands
    write = lambda operand: _write(operand, text, path)  # noqa: E731
    if kind == "number" or kind == "constant":
        return f"({node.value!r})"
    if kind == "time":
        return "t"
    if kind == "position_component":
        return engine_position_names[node.direction]
    if kind == "field" and node.rank == 0:
        return node.name
    if (
        kind == "gradient_component"
        and node.name in engine_component_names
        and len(node.component) == 1
    ):
        variable = engine_component_names[node.name][node.component[0]]
        return f"grad_{engine_position_names[node.direction]}({variable})"
    if kind == "field_component":
        if node.name in engine_component_names and len(node.component) == 1:
            return engine_component_names[node.name][node.component[0]]
        return units.component_name(node.name, node.component)
    if kind == "component" and operands[0].kind == "position":
        return engine_position_names[node.component[0]]
    if kind == "component" and operands[0].kind == "field":
        field = operands[0]
        if field.name in engine_component_names and len(node.component) == 1:
            return engine_component_names[field.name][node.component[0]]
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
    if kind in functions:
        return f"{kind}({write(operands[0])})"
    if kind in pairs:
        return f"{pairs[kind]}({write(operands[0])}, {write(operands[1])})"
    if kind in comparisons:
        return f"({write(operands[0])} {comparisons[kind]} {write(operands[1])})"
    if kind == "conditional":
        return f"if({write(operands[0])}, {write(operands[1])}, {write(operands[2])})"
    raise input_error(
        f"{path}: the formula '{text}' uses {kind}, which a property formula cannot hold. "
        "A property is a function of the fields, the position and the time."
    )
