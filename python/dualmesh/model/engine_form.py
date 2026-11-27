"""Lowers the expressions of the input and of general_form_PDE onto the expressions of the engine.

This module holds the one printer of the engine's expression language, which only it writes.
A formula of the input, such as `1/([0.0375 m*K/W] + [2.165e-4 m/W]*temperature)`, is parsed
into an expression graph, which checks its names and its dimension. The validated tree holds
that graph beside the text (`dualmesh.model.syntax.formula_text`), so a formula parses once. The
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
to the names of its variables, one per component, as `dualmesh.input.build` makes it from the
physics and the models of the problem. Without it, a component of the displacement or the
velocity takes the Cartesian names of `engine_component_names`.

`add_law` lowers a conservation law onto the kernels and the property objects of the engine,
and `add_boundary_condition` a condition on its unknown or its flux. Both serve the models of
the input and the equations of general_form_PDE.
"""

from dualmesh import units
from dualmesh._core import InputError
from dualmesh.model import graph
from dualmesh.model.calculus import differentiate
from dualmesh.model.dimensions import comparisons, dimension_error, is_zero
from dualmesh.model.syntax import parse

# The engine's names of the components of the vector fields of the input, in Cartesian
# coordinates, for a formula written before the physics name their variables.
engine_component_names = {
    "displacement": ("disp_x", "disp_y", "disp_z"),
    "velocity": ("velocity_x", "velocity_y", "velocity_z"),
}
engine_position_names = ("x", "y", "z")

# The functions of the graph with their names in the engine. `_write` writes a power with ^
# before it reads this table.
functions = {kind: name for name, kind in graph.engine_kinds.items()}
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
    except InputError as error:
        raise InputError(f"{path}: {error}") from None
    _, expected = units.parse_unit(unit)
    if node.rank != 0:
        raise InputError(f"{path}: the formula '{text}' is not a scalar.")
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
            raise InputError(f"formula '{text}': unknown name '{leaf.name}'")
    try:
        return graph.substitute(node, replacements) if replacements else node
    except InputError as error:
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
    raise InputError(
        f"{path}: the formula '{source}' uses {kind}, which the engine does not evaluate. "
        "A formula is a function of the fields, the states, the position and the time."
    )


def add_law(problem, law, variables: dict, dimension: int, transient: bool, restriction, where):
    """Adds the objects of the engine of one conservation law, on the blocks of restriction.

    This is the one lowering of a conservation law, for the models of the input and the
    equations of general_form_PDE alike. `variables` maps each field to the engine's names of
    its variables, and `where` names the law in messages. The law becomes:

    - a parsed_state property object, which integrates the states at each integration point by
      the backward Euler method and holds each as the property of its name;
    - for each scalar component of the unknown, with the terms of `conservation_law.components`:
      a second_time_derivative kernel with the coefficient of the inertia, a time_derivative
      kernel with the capacity c of a storage written as an antiderivative (the rate form
      c du/dt) or, in a transient problem, with any other storage m (the difference
      (m - m_old)/dt), and a parsed_kernel with the flux and the source.
    """

    def write(node) -> str:
        return write_expression(node, where, variables)

    states = law.states()
    state_object = None
    if states:
        components = [graph.state_component(node.name, (), node.dimension) for node in states]
        rates = [graph.expand(law.state_rates[node], dimension) for node in states]
        derivatives = [differentiate(rate, component) for rate in rates for component in components]
        state_object = problem.add_property(
            "parsed_state",
            states=[node.name for node in states],
            rates=[write(rate) for rate in rates],
            rate_derivatives=[write(entry) for entry in derivatives],
            initial_values=[float(node.initial) for node in states],
            **restriction,
        )
    names = variables[law.unknown.name]
    for variable, terms in zip(names, law.components(dimension), strict=True):
        if terms.mass is not None:
            problem.add_kernel(
                "second_time_derivative",
                variable=variable,
                coefficient=write(terms.mass),
                **restriction,
            )
        if terms.capacity is not None:
            problem.add_kernel(
                "time_derivative",
                variable=variable,
                coefficient=write(terms.capacity),
                **restriction,
            )
        if transient and terms.storage is not None:
            storage = {"variable": variable, "storage": write(terms.storage), **restriction}
            if state_object is not None:
                storage["state_object"] = state_object.name
            problem.add_kernel("time_derivative", **storage)
        kernel = {}
        if terms.flux is not None:
            kernel["flux"] = [write(component) for component in terms.flux]
        if terms.source is not None:
            kernel["source"] = write(terms.source)
        if kernel:
            advective = _advective(terms.flux or [], terms.source, terms.unknown)
            problem.add_kernel(
                "parsed_kernel", variable=variable, **kernel, advective=advective, **restriction
            )


def add_boundary_condition(
    problem, boundary, variable: str, value: str, flux: bool, **parameters
) -> None:
    """Adds the condition of one variable on a boundary: its value, or with flux the outward
    normal flux F . n of its law. Both the models of the input and general_form_PDE take it."""
    if flux:
        # Neumann takes the flux entering the body, -F . n.
        problem.add_boundary_condition(
            "Neumann", boundary, variable=variable, flux=f"-({value})", **parameters
        )
    else:
        problem.add_boundary_condition(
            "Dirichlet", boundary, variable=variable, value=value, **parameters
        )


def _advective(flux: list, source, unknown) -> bool:
    """Whether the term carries the unknown: a flux left where the gradient of the unknown is zero,
    or a source that reads that gradient."""
    if source is not None and any(
        leaf.kind == "gradient_component" and leaf.name == unknown.name
        for leaf in graph.leaves(source)
    ):
        return True
    for component in flux:
        gradients = {
            leaf: graph.number(0.0, leaf.dimension)
            for leaf in graph.leaves(component)
            if leaf.kind == "gradient_component" and leaf.name == unknown.name
        }
        if not is_zero(graph.substitute(component, gradients)):
            return True
    return False
