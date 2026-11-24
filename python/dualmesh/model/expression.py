"""The expression graph of the model language.

A model writes its storage, inertia, flux, source and state rates as
expressions at one point. Each expression is a node of a graph with the
attributes below.

- `kind` names the call that made the node, such as `field`, `add` or `exp`.
- `operands` holds the child nodes.
- `rank` is 0 for a scalar, 1 for a vector and 2 for a tensor. Each index runs
  over the space dimension.
- `dimension` is the physical dimension, checked by `dimensions.dimension_of`
  when the node is built, so an inconsistent graph never exists.
- `name`, `value`, `component`, `direction` and `initial` describe leaves.
- `table` holds the grid of an `interpolation` node, whose `name` is the
  input path of the table, such as `properties.thermal_conductivity`.

Nodes are immutable and hash-consed: building the same structure twice gives
the same object. Equal subexpressions therefore share one node, which is the
common-subexpression identity that code generation reads. The constructors
fold numbers and remove additions of zero and multiplications by zero and one.

The gradient of a vector field $\\mathbf{u}$ is the tensor
$(\\nabla\\mathbf{u})_{ij} = \\partial u_i / \\partial x_j$.

A Python number in an expression is a dimensionless `number`. Values of
`constant` and `state` given as Python numbers are in SI. Values given as text
carry their unit, for example `"10 mol/m^3"`.
"""

import dataclasses
import itertools
import math
import operator
import weakref
from collections.abc import Sequence
from functools import reduce

from dualmesh.input import units
from dualmesh.input.units import base, dimension, dimensionless, input_error
from dualmesh.model.dimensions import dimension_of, is_zero, leaf_kinds, length

__all__ = [
    "expression",
    "rank_error",
    "field",
    "state",
    "constant",
    "number",
    "physical_constant",
    "position",
    "time",
    "time_step",
    "cell_size",
    "normal",
    "identity",
    "grad",
    "exp",
    "log",
    "sqrt",
    "sin",
    "cos",
    "tanh",
    "maximum",
    "minimum",
    "less",
    "greater",
    "conditional",
    "dot",
    "inner",
    "outer",
    "transpose",
    "trace",
    "symmetric",
    "deviatoric",
    "determinant",
    "inverse",
    "norm",
    "interpolation",
    "antiderivative",
    "expand",
    "substitute",
    "leaves",
    "field_component",
    "gradient_component",
    "state_component",
    "position_component",
    "normal_component",
    "as_expression",
]


class rank_error(input_error):
    """An operation on operands of the wrong rank."""


rank_of_shape = {"scalar": 0, "vector": 1, "tensor": 2}
largest_rank = 2
largest_table_input_count = 3
time_dimension = base("time")

scalar_functions = frozenset({"exp", "log", "sqrt", "sin", "cos", "tanh", "abs"})

# Every live node, keyed by its structure. A node leaves the table when no
# reference to it remains.
interned_nodes = weakref.WeakValueDictionary()


class expression:
    """One immutable, hash-consed node of the expression graph."""

    __slots__ = (
        "kind",
        "operands",
        "rank",
        "dimension",
        "name",
        "value",
        "component",
        "direction",
        "initial",
        "table",
        "structure",
        "structure_hash",
        "__weakref__",
    )

    def __setattr__(self, attribute, value):
        raise AttributeError("an expression is immutable")

    def __eq__(self, other):
        if self is other:
            return True
        if not isinstance(other, expression):
            return NotImplemented
        return self.structure_hash == other.structure_hash and self.structure == other.structure

    def __hash__(self):
        return self.structure_hash

    def __bool__(self):
        raise TypeError("an expression has no truth value. Use less, greater and conditional")

    def __iter__(self):
        raise TypeError("an expression is not iterable. Use expand to get its components")

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self

    def __reduce__(self):
        return (rebuild, (self.kind, self.operands, self.attributes()))

    def attributes(self) -> dict:
        """Return the keyword arguments of `build` that, with the kind and the operands, rebuild this node."""
        attributes = {
            "name": self.name,
            "value": self.value,
            "component": self.component,
            "direction": self.direction,
            "initial": self.initial,
            "table": self.table,
        }
        if self.kind in leaf_kinds or self.kind == "interpolation":
            attributes["physical_dimension"] = self.dimension
        if self.kind in leaf_kinds:
            attributes["rank"] = self.rank
        return attributes

    def __add__(self, other):
        other = operand_or_none(other)
        return NotImplemented if other is None else build("add", (self, other))

    def __radd__(self, other):
        other = operand_or_none(other)
        return NotImplemented if other is None else build("add", (other, self))

    def __sub__(self, other):
        other = operand_or_none(other)
        return NotImplemented if other is None else build("add", (self, build("negate", (other,))))

    def __rsub__(self, other):
        other = operand_or_none(other)
        return NotImplemented if other is None else build("add", (other, build("negate", (self,))))

    def __mul__(self, other):
        other = operand_or_none(other)
        return NotImplemented if other is None else build("multiply", (self, other))

    def __rmul__(self, other):
        other = operand_or_none(other)
        return NotImplemented if other is None else build("multiply", (other, self))

    def __truediv__(self, other):
        other = operand_or_none(other)
        return NotImplemented if other is None else build("divide", (self, other))

    def __rtruediv__(self, other):
        other = operand_or_none(other)
        return NotImplemented if other is None else build("divide", (other, self))

    def __pow__(self, other):
        other = operand_or_none(other)
        return NotImplemented if other is None else build("power", (self, other))

    def __rpow__(self, other):
        other = operand_or_none(other)
        return NotImplemented if other is None else build("power", (other, self))

    def __neg__(self):
        return build("negate", (self,))

    def __pos__(self):
        return self

    def __abs__(self):
        return build("abs", (self,))

    def __lt__(self, other):
        return less(self, other)

    def __gt__(self, other):
        return greater(self, other)

    def __getitem__(self, index):
        index = index if isinstance(index, tuple) else (index,)
        return build("component", (self,), component=check_indices(index, "component index"))

    def __str__(self):
        return text(self)

    def __repr__(self):
        return f"expression({self.kind}: {text(self)})"


def operand_or_none(value):
    """Return value as an expression, or None when it cannot enter an expression."""
    if isinstance(value, expression):
        return value
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return number(value)


def as_expression(value) -> expression:
    """Return value as an expression. A Python number becomes a dimensionless number."""
    node = operand_or_none(value)
    if node is None:
        raise TypeError(
            f"an expression cannot hold a value of type {type(value).__name__}: {value!r}"
        )
    return node


def check_indices(index: tuple, what: str) -> tuple:
    """Return index as a tuple of ints, each from 0 to 2."""
    for entry in index:
        if isinstance(entry, bool) or not isinstance(entry, int) or not 0 <= entry < 3:
            raise rank_error(f"{what} must be an integer from 0 to 2, got {entry!r}")
    return tuple(index)


# Building nodes


def build(
    kind: str,
    operands: tuple = (),
    *,
    name=None,
    value=None,
    component=(),
    direction=None,
    initial=None,
    table=None,
    rank=None,
    physical_dimension=None,
) -> expression:
    """Return the node of kind with operands, checked, simplified and hash-consed.

    Leaves pass their rank and dimension. Operations compute both, except the
    dimension of an `interpolation`, which it passes.
    """
    operands = tuple(as_expression(operand) for operand in operands)
    if kind not in leaf_kinds:
        rank = rank_of(kind, operands, component)
    node_dimension = dimension_of(kind, operands, physical_dimension=physical_dimension)
    simplified = simplify(kind, operands, rank, node_dimension)
    if simplified is not None:
        return simplified
    structure = (
        kind,
        operands,
        rank,
        node_dimension,
        name,
        value,
        component,
        direction,
        initial,
        table,
    )
    existing = interned_nodes.get(structure)
    if existing is not None:
        return existing
    node = object.__new__(expression)
    for attribute, attribute_value in (
        ("kind", kind),
        ("operands", operands),
        ("rank", rank),
        ("dimension", node_dimension),
        ("name", name),
        ("value", value),
        ("component", component),
        ("direction", direction),
        ("initial", initial),
        ("table", table),
        ("structure", structure),
        ("structure_hash", hash(structure)),
    ):
        object.__setattr__(node, attribute, attribute_value)
    interned_nodes[structure] = node
    return node


def rebuild(kind: str, operands: tuple, attributes: dict) -> expression:
    """Return the node of kind with operands and the attributes of `expression.attributes`."""
    return build(kind, operands, **attributes)


def rank_of(kind: str, operands: tuple, component: tuple) -> int:
    """Return the rank of an operation node, or raise rank_error."""
    ranks = [operand.rank for operand in operands]

    def require(condition: bool, message: str):
        if not condition:
            raise rank_error(
                f"{kind}: {message}, got ranks {ranks} for "
                + ", ".join(f"'{operand}'" for operand in operands)
            )

    if kind == "add":
        require(ranks[0] == ranks[1], "the operands need equal ranks")
        return ranks[0]
    if kind in ("negate",):
        return ranks[0]
    if kind == "multiply":
        require(
            ranks[0] == 0 or ranks[1] == 0,
            "one operand must be a scalar. Use dot, inner or outer for two non-scalars",
        )
        return ranks[0] + ranks[1]
    if kind == "divide":
        require(ranks[1] == 0, "the denominator must be a scalar")
        return ranks[0]
    if kind in scalar_functions or kind in ("power", "maximum", "minimum", "less", "greater"):
        require(all(rank == 0 for rank in ranks), "the operands must be scalars")
        return 0
    if kind == "component":
        require(1 <= len(component) <= ranks[0], "the number of indices must be from 1 to the rank")
        return ranks[0] - len(component)
    if kind == "interpolation":
        require(all(rank == 0 for rank in ranks), "every input must be a scalar")
        return 0
    if kind == "antiderivative":
        require(
            ranks == [0, 0] and operands[1].kind in ("field", "field_component"),
            "the integrand must be a scalar and the variable a scalar field",
        )
        return 0
    if kind == "grad":
        require(operands[0].kind == "field", "the operand must be a field")
        require(ranks[0] < largest_rank, "the gradient of a tensor field is not supported")
        return ranks[0] + 1
    if kind == "conditional":
        require(
            ranks[0] == 0 and ranks[1] == ranks[2],
            "the condition must be a scalar and the branches need equal ranks",
        )
        return ranks[1]
    if kind == "dot":
        require(
            ranks[0] >= 1 and ranks[1] >= 1 and ranks[0] + ranks[1] - 2 <= largest_rank,
            "the operands must be vectors or tensors",
        )
        return ranks[0] + ranks[1] - 2
    if kind == "inner":
        require(ranks[0] == ranks[1] >= 1, "the operands must be vectors or tensors of equal rank")
        return 0
    if kind == "outer":
        require(ranks[0] + ranks[1] <= largest_rank, "the result rank must be at most 2")
        return ranks[0] + ranks[1]
    if kind in ("transpose", "symmetric", "deviatoric", "inverse"):
        require(ranks[0] == 2, "the operand must be a tensor")
        return 2
    if kind in ("trace", "determinant"):
        require(ranks[0] == 2, "the operand must be a tensor")
        return 0
    if kind == "norm":
        require(ranks[0] >= 1, "the operand must be a vector or a tensor")
        return 0
    raise rank_error(f"unknown expression kind '{kind}'")


# Simplification


def is_number(node: expression) -> bool:
    return node.kind == "number"


def is_one(node: expression) -> bool:
    """Return whether node is the dimensionless number 1."""
    return node.kind == "number" and node.value == 1.0 and node.dimension.is_dimensionless()


def values_text(values) -> str:
    return ", ".join(format(value, ".15g") for value in values)


def fold(kind: str, function, *values) -> float:
    """Return function of values, or raise input_error when it has no real value."""
    try:
        result = function(*values)
    except (ValueError, OverflowError, ZeroDivisionError) as error:
        raise input_error(f"{kind} of {values_text(values)}: {error}") from None
    if isinstance(result, complex) or not math.isfinite(result):
        raise input_error(f"{kind} of {values_text(values)} has no finite real value")
    return result


number_functions = {
    "exp": math.exp,
    "log": math.log,
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tanh": math.tanh,
    "abs": abs,
    "negate": operator.neg,
}
number_operations = {
    "add": operator.add,
    "multiply": operator.mul,
    "divide": operator.truediv,
    "power": operator.pow,
    "maximum": max,
    "minimum": min,
}


def simplify(kind: str, operands: tuple, rank: int, node_dimension: dimension):
    """Return a simpler node equal to the node of kind with operands, or None."""
    if kind in leaf_kinds:
        return None
    if all(is_number(operand) for operand in operands):
        if kind in number_functions:
            return number(fold(kind, number_functions[kind], operands[0].value), node_dimension)
        if kind in number_operations:
            values = (operands[0].value, operands[1].value)
            return number(fold(kind, number_operations[kind], *values), node_dimension)
    if kind == "add":
        first, second = operands
        if is_zero(second) and first.dimension == node_dimension:
            return first
        if is_zero(first) and second.dimension == node_dimension:
            return second
    elif kind == "negate":
        if operands[0].kind == "negate":
            return operands[0].operands[0]
    elif kind == "multiply":
        first, second = operands
        if rank == 0 and (is_zero(first) or is_zero(second)):
            return number(0.0, node_dimension)
        if is_one(first):
            return second
        if is_one(second):
            return first
    elif kind == "divide":
        numerator, denominator = operands
        if is_zero(denominator):
            raise input_error(f"division by zero in '{numerator}/{denominator}'")
        if rank == 0 and is_zero(numerator):
            return number(0.0, node_dimension)
        if is_one(denominator):
            return numerator
    elif kind == "power":
        base_operand, exponent = operands
        if is_zero(exponent):
            return number(1.0)
        if is_one(exponent):
            return base_operand
    elif kind == "conditional":
        condition, if_true, if_false = operands
        if if_true == if_false and if_true.dimension == node_dimension:
            return if_true
        # A dimensionless zero matches every dimension, so two zero branches may differ
        # in dimension and still give zero.
        if is_zero(if_true) and is_zero(if_false):
            return number(0.0, node_dimension)
        if all(is_number(operand) for operand in condition.operands):
            first, second = condition.operands
            holds = (
                first.value < second.value
                if condition.kind == "less"
                else first.value > second.value
            )
            chosen = if_true if holds else if_false
            if chosen.dimension == node_dimension:
                return chosen
    return None


# Leaves


def registry_entry(name: str):
    """Return the standard name entry of name, or None when name is not a standard name."""
    return units.standard_names().get(name)


def check_name(name, what: str) -> str:
    if not isinstance(name, str) or not name.isidentifier():
        raise input_error(
            f"the name of a {what} must be a word of letters, digits and underscores, got {name!r}"
        )
    return name


def SI_dimension(unit, what: str) -> dimension:
    """Return the dimension of unit, which must be a coherent SI unit with factor 1.

    what starts every error message, so it names the declaration or the input key.
    """
    try:
        factor, unit_dimension = units.parse_unit(str(unit))
    except input_error as error:
        raise input_error(f"{what}: {error}") from None
    if abs(factor - 1.0) > 1.0e-12:
        raise input_error(
            f"{what}: '{unit}' is not a coherent SI unit. Values inside the model are in SI, "
            f"so declare the SI unit, for example 'mol/m^3' in place of 'mmol/m^3'"
        )
    return unit_dimension


def declared_rank_and_dimension(name: str, unit, shape: str, what: str) -> tuple:
    """Return the rank and dimension of a field or state from the registry or from unit and shape."""
    entry = registry_entry(name)
    if entry is not None:
        if entry.value is not None:
            raise input_error(
                f"{what} '{name}' is a physical constant. Use physical_constant('{name}')"
            )
        if unit is not None:
            raise input_error(
                f"{what} '{name}' is a standard name with the unit {entry.unit}. Leave out unit"
            )
        if entry.shape != "scalar_or_tensor":
            shape = entry.shape
        elif shape not in ("scalar", "tensor"):
            raise input_error(f"{what} '{name}' is a scalar or a tensor, got shape '{shape}'")
        return rank_of_shape[shape], entry.physical_dimension
    if unit is None:
        raise input_error(
            f"{what} '{name}' is not a standard name, so it needs a unit, for example unit=\"mol/m^3\""
        )
    if shape not in rank_of_shape:
        raise input_error(
            f"the shape of {what} '{name}' must be scalar, vector or tensor, got '{shape}'"
        )
    return rank_of_shape[shape], SI_dimension(unit, f"{what} '{name}'")


def field(name: str, unit=None, shape: str = "scalar") -> expression:
    """Return an unknown field. A standard name takes its unit and shape from the registry."""
    check_name(name, "field")
    rank, field_dimension = declared_rank_and_dimension(name, unit, shape, "field")
    return build("field", name=name, rank=rank, physical_dimension=field_dimension)


def state(name: str, unit=None, initial=0.0, shape: str = "scalar") -> expression:
    """Return an internal state. initial is a number in SI or a quantity text with its unit."""
    check_name(name, "state")
    rank, state_dimension = declared_rank_and_dimension(name, unit, shape, "state")
    if isinstance(initial, str):
        initial_value = units.to_SI(initial, state_dimension, f"initial value of state {name}")
    elif (
        isinstance(initial, (int, float))
        and not isinstance(initial, bool)
        and math.isfinite(initial)
    ):
        initial_value = float(initial)
    else:
        raise input_error(
            f"the initial value of state '{name}' must be a finite number or a quantity, "
            f"got {initial!r}"
        )
    return build(
        "state", name=name, initial=initial_value, rank=rank, physical_dimension=state_dimension
    )


def constant(name: str, value=None, unit=None) -> expression:
    """Return a named scalar constant.

    The dimension comes from the registry for a standard name, else from the
    unit of a quantity text value, else from unit. A number value is in SI.
    A constant without a value takes it at run time from its boundary
    condition entry.
    """
    check_name(name, "constant")
    entry = registry_entry(name)
    key = f"constant {name}"
    if entry is not None:
        if entry.value is not None:
            raise input_error(
                f"constant '{name}' is a physical constant. Use physical_constant('{name}')"
            )
        if unit is not None:
            raise input_error(
                f"constant '{name}' is a standard name with the unit {entry.unit}. Leave out unit"
            )
        constant_dimension = entry.physical_dimension
        if isinstance(value, str):
            value = units.to_SI(value, constant_dimension, key)
    elif isinstance(value, str):
        value, constant_dimension = units.parse_quantity(value)
        if unit is not None and SI_dimension(unit, key) != constant_dimension:
            raise input_error(
                f"constant '{name}': the value has the dimension {constant_dimension}, "
                f"which differs from the unit '{unit}'"
            )
    elif unit is not None:
        constant_dimension = SI_dimension(unit, key)
    else:
        constant_dimension = dimensionless
    if value is not None:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise input_error(
                f"the value of constant '{name}' must be a finite number or a quantity, got {value!r}"
            )
        value = float(value)
    return build("constant", name=name, value=value, rank=0, physical_dimension=constant_dimension)


def number(value, physical_dimension: dimension = dimensionless) -> expression:
    """Return a number in SI with a dimension, dimensionless by default."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise input_error(f"a number must be a finite int or float, got {value!r}")
    # Adding 0.0 turns -0.0 into 0.0, so both zeros share one node.
    return build("number", value=float(value) + 0.0, rank=0, physical_dimension=physical_dimension)


def physical_constant(name: str) -> expression:
    """Return the number of a physical constant of the registry, such as molar_gas_constant."""
    entry = registry_entry(name)
    if entry is None or entry.value is None:
        known = sorted(
            key for key, item in units.standard_names().items() if item.value is not None
        )
        raise input_error(
            f"'{name}' is not a physical constant. The physical constants are {', '.join(known)}"
        )
    return number(entry.value, entry.physical_dimension)


def position() -> expression:
    """Return the position vector in m."""
    return build("position", rank=1, physical_dimension=length)


def time() -> expression:
    """Return the time in s."""
    return build("time", rank=0, physical_dimension=time_dimension)


def time_step() -> expression:
    """Return the time step in s."""
    return build("time_step", rank=0, physical_dimension=time_dimension)


def cell_size() -> expression:
    """Return the diameter of the cell in m."""
    return build("cell_size", rank=0, physical_dimension=length)


def normal() -> expression:
    """Return the outward unit normal of a boundary, for boundary terms only."""
    return build("normal", rank=1, physical_dimension=dimensionless)


def identity() -> expression:
    """Return the identity tensor."""
    return build("identity", rank=2, physical_dimension=dimensionless)


def component_dimension(name: str, field_dimension) -> dimension:
    """Return the dimension of the field name, from field_dimension or from the registry."""
    if field_dimension is not None:
        return field_dimension
    entry = registry_entry(name)
    if entry is None:
        raise input_error(f"'{name}' is not a standard name. Pass the dimension of the field")
    return entry.physical_dimension


def field_component(name: str, component=(), physical_dimension=None) -> expression:
    """Return one component of a field. physical_dimension is that of the field, from the registry by default."""
    return build(
        "field_component",
        name=name,
        component=check_indices(tuple(component), "component"),
        rank=0,
        physical_dimension=component_dimension(name, physical_dimension),
    )


def gradient_component(name: str, component, direction: int, physical_dimension=None) -> expression:
    """Return the derivative of one field component along one direction.

    physical_dimension is that of the field, from the registry by default.
    """
    check_indices((direction,), "direction")
    return build(
        "gradient_component",
        name=name,
        component=check_indices(tuple(component), "component"),
        direction=direction,
        rank=0,
        physical_dimension=component_dimension(name, physical_dimension) / length,
    )


def state_component(name: str, component=(), physical_dimension=None) -> expression:
    """Return one component of a state. physical_dimension is that of the state, from the registry by default."""
    return build(
        "state_component",
        name=name,
        component=check_indices(tuple(component), "component"),
        rank=0,
        physical_dimension=component_dimension(name, physical_dimension),
    )


def position_component(direction: int) -> expression:
    check_indices((direction,), "direction")
    return build("position_component", direction=direction, rank=0, physical_dimension=length)


def normal_component(direction: int) -> expression:
    check_indices((direction,), "direction")
    return build("normal_component", direction=direction, rank=0, physical_dimension=dimensionless)


# Operations


def grad(field_node: expression) -> expression:
    """Return the gradient of a field, of rank one higher and dimension over length."""
    return build("grad", (field_node,))


def exp(argument) -> expression:
    return build("exp", (argument,))


def log(argument) -> expression:
    """Return the natural logarithm."""
    return build("log", (argument,))


def sqrt(argument) -> expression:
    return build("sqrt", (argument,))


def sin(argument) -> expression:
    return build("sin", (argument,))


def cos(argument) -> expression:
    return build("cos", (argument,))


def tanh(argument) -> expression:
    return build("tanh", (argument,))


def maximum(first, second) -> expression:
    return build("maximum", (first, second))


def minimum(first, second) -> expression:
    return build("minimum", (first, second))


def less(first, second) -> expression:
    """Return the condition first < second, for conditional."""
    return build("less", (first, second))


def greater(first, second) -> expression:
    """Return the condition first > second, for conditional."""
    return build("greater", (first, second))


def conditional(condition, if_true, if_false) -> expression:
    """Return if_true where condition holds and if_false elsewhere."""
    return build("conditional", (condition, if_true, if_false))


def dot(first, second) -> expression:
    """Return the contraction of the last index of first with the first index of second."""
    return build("dot", (first, second))


def inner(first, second) -> expression:
    """Return the full contraction of two vectors or two tensors."""
    return build("inner", (first, second))


def outer(first, second) -> expression:
    return build("outer", (first, second))


def transpose(tensor) -> expression:
    return build("transpose", (tensor,))


def trace(tensor) -> expression:
    return build("trace", (tensor,))


def symmetric(tensor) -> expression:
    """Return the symmetric part $(\\mathbf{A} + \\mathbf{A}^T)/2$."""
    return build("symmetric", (tensor,))


def deviatoric(tensor) -> expression:
    """Return $\\mathbf{A} - (\\mathrm{tr}\\,\\mathbf{A}/d)\\,\\mathbf{I}$, with $d$ the space dimension.

    For $d = 2$, this is the deviator of the in-plane tensor. The von Mises stress
    and $J_2$ plasticity of a plane-stress or plane-strain solid need the deviator
    of the full stress, whose trace includes the out-of-plane stress
    $\\sigma_{zz}$ in $\\mathrm{Pa}$. Build that deviator from $\\sigma_{zz}$ in the
    constitutive law.
    """
    return build("deviatoric", (tensor,))


def determinant(tensor) -> expression:
    """Return the determinant of a dimensionless tensor."""
    return build("determinant", (tensor,))


def inverse(tensor) -> expression:
    return build("inverse", (tensor,))


def norm(argument) -> expression:
    """Return the Euclidean norm of a vector or the Frobenius norm of a tensor."""
    return build("norm", (argument,))


@dataclasses.dataclass(frozen=True)
class table_data:
    """The grid of an `interpolation` node.

    `axes[a]` holds the strictly increasing values of input `a` in SI, and
    `values` the table values in SI, row-major over the inputs.
    `derivative[a]` is 1 where the node is the partial derivative by input `a`,
    and 0 elsewhere. `input_names` name the inputs in reports.
    """

    input_names: tuple[str, ...]
    axes: tuple[tuple[float, ...], ...]
    values: tuple[float, ...]
    derivative: tuple[int, ...]


def finite_numbers(numbers, what: str) -> tuple[float, ...]:
    """Return numbers as a tuple of floats, or raise input_error naming what."""
    result = []
    for entry in numbers:
        if (
            isinstance(entry, bool)
            or not isinstance(entry, (int, float))
            or not math.isfinite(entry)
        ):
            raise input_error(f"{what} must be finite numbers, got {entry!r}")
        result.append(float(entry) + 0.0)
    return tuple(result)


def interpolation(
    source: str,
    inputs: Sequence,
    axes: Sequence[Sequence[float]],
    values: Sequence[float],
    value_dimension: dimension,
    input_names: Sequence[str],
) -> expression:
    """Return the multilinear interpolation of a table on a grid of 1 to 3 scalar inputs.

    source names the input path of the table in reports, for example
    `properties.thermal_conductivity`. Each axis holds at least 2 strictly
    increasing values in the SI unit of its input. values holds the product of
    the axis lengths, row-major over the inputs, in the SI unit of
    value_dimension. input_names name the inputs in reports as the table columns
    name them, such as temperature, time or r. Between grid points the node is
    linear in each input, so its derivative is the exact derivative of the
    interpolant. A read outside the grid stops the run with a message that
    names source.
    """
    if not isinstance(source, str) or not source:
        raise input_error(f"the source of a table must be a non-empty text, got {source!r}")
    inputs = tuple(as_expression(node) for node in inputs)
    if not 1 <= len(inputs) <= largest_table_input_count:
        raise input_error(
            f"{source}: a table takes 1 to {largest_table_input_count} inputs, got {len(inputs)}"
        )
    if len(axes) != len(inputs):
        raise input_error(f"{source}: the table has {len(inputs)} inputs but {len(axes)} axes")
    checked_axes = []
    for index, axis in enumerate(axes):
        axis = finite_numbers(axis, f"{source}: the values of axis {index}")
        if len(axis) < 2 or any(second <= first for first, second in itertools.pairwise(axis)):
            raise input_error(
                f"{source}: axis {index} needs at least 2 strictly increasing values, got {axis}"
            )
        checked_axes.append(axis)
    values = finite_numbers(values, f"{source}: the table values")
    expected_count = math.prod(len(axis) for axis in checked_axes)
    if len(values) != expected_count:
        raise input_error(
            f"{source}: the grid holds {expected_count} points, but the table has "
            f"{len(values)} values"
        )
    names = tuple(input_names)
    if len(names) != len(inputs) or not all(isinstance(name, str) and name for name in names):
        raise input_error(f"{source}: give one non-empty name per input, got {input_names!r}")
    grid = table_data(names, tuple(checked_axes), values, (0,) * len(inputs))
    return build(
        "interpolation", inputs, name=source, table=grid, physical_dimension=value_dimension
    )


def interpolation_partial(node: expression, index: int) -> expression:
    """Return the partial derivative of an interpolation node by its input index.

    The interpolant is linear in each input between grid points, so a second
    derivative by the same input is zero.
    """
    partial_dimension = node.dimension / node.operands[index].dimension
    if node.table.derivative[index] == 1:
        return number(0.0, partial_dimension)
    derivative = tuple(
        1 if input_index == index else order
        for input_index, order in enumerate(node.table.derivative)
    )
    return build(
        "interpolation",
        node.operands,
        name=node.name,
        table=dataclasses.replace(node.table, derivative=derivative),
        physical_dimension=partial_dimension,
    )


def antiderivative(integrand, variable) -> expression:
    """Return $H(u) = \\int^{u} f(u')\\,\\mathrm{d}u'$, with $f$ the integrand and $u$ the variable.

    The variable is a scalar field. The derivative of $H$ by $u$ is $f$, so a
    storage written as an antiderivative has the rate $f\\,\\dot{u}$ for any
    form of $f$, such as a table. The generated code never evaluates $H$
    itself, only its derivatives.
    """
    return build("antiderivative", (integrand, variable))


# Expansion into scalar components


def expand(node: expression, space_dimension: int):
    """Return the scalar components of node in a space of space_dimension.

    A scalar gives one expression, a vector a list of space_dimension
    expressions and a tensor a list of lists. Every result is rank 0 and its
    leaves are the scalar leaves field_component, gradient_component,
    state_component, position_component, normal_component, constant, number,
    time, time_step and cell_size.
    """
    if space_dimension not in (1, 2, 3):
        raise input_error(f"the space dimension must be 1, 2 or 3, got {space_dimension!r}")
    expanded = {}

    def entry(tensor, index):
        for position_index in index:
            tensor = tensor[position_index]
        return tensor

    def tabulate(rank, function):
        if rank == 0:
            return function(())
        if rank == 1:
            return [function((row,)) for row in range(space_dimension)]
        return [
            [function((row, column)) for column in range(space_dimension)]
            for row in range(space_dimension)
        ]

    def indices(rank):
        return itertools.product(range(space_dimension), repeat=rank)

    def total(terms):
        return reduce(operator.add, terms)

    def walk(current: expression):
        if current in expanded:
            return expanded[current]
        result = expand_node(current)
        expanded[current] = result
        return result

    def expand_node(current: expression):
        kind = current.kind
        rank = current.rank
        if kind == "field":
            return tabulate(
                rank, lambda index: field_component(current.name, index, current.dimension)
            )
        if kind == "state":
            return tabulate(
                rank, lambda index: state_component(current.name, index, current.dimension)
            )
        if kind == "grad":
            unknown = current.operands[0]
            return tabulate(
                rank,
                lambda index: gradient_component(
                    unknown.name, index[:-1], index[-1], unknown.dimension
                ),
            )
        if kind == "position":
            return tabulate(1, lambda index: position_component(index[0]))
        if kind == "normal":
            return tabulate(1, lambda index: normal_component(index[0]))
        if kind == "identity":
            return tabulate(2, lambda index: number(1.0 if index[0] == index[1] else 0.0))
        if not current.operands:
            return current
        parts = [walk(operand) for operand in current.operands]
        ranks = [operand.rank for operand in current.operands]
        if kind == "component":
            if any(index >= space_dimension for index in current.component):
                raise rank_error(
                    f"component {current.component} of '{current.operands[0]}' "
                    f"does not exist in {space_dimension} dimensions"
                )
            return entry(parts[0], current.component)
        if kind == "add":
            return tabulate(rank, lambda index: entry(parts[0], index) + entry(parts[1], index))
        if kind == "negate":
            return tabulate(rank, lambda index: -entry(parts[0], index))
        if kind in ("multiply", "outer"):
            return tabulate(
                rank,
                lambda index: (
                    entry(parts[0], index[: ranks[0]]) * entry(parts[1], index[ranks[0] :])
                ),
            )
        if kind == "divide":
            return tabulate(rank, lambda index: entry(parts[0], index) / parts[1])
        if kind == "conditional":
            return tabulate(
                rank,
                lambda index: conditional(parts[0], entry(parts[1], index), entry(parts[2], index)),
            )
        if kind == "dot":
            split = ranks[0] - 1
            return tabulate(
                rank,
                lambda index: total(
                    entry(parts[0], index[:split] + (summation_index,))
                    * entry(parts[1], (summation_index,) + index[split:])
                    for summation_index in range(space_dimension)
                ),
            )
        if kind == "inner":
            return total(
                entry(parts[0], index) * entry(parts[1], index) for index in indices(ranks[0])
            )
        if kind == "norm":
            return sqrt(total(entry(parts[0], index) ** 2 for index in indices(ranks[0])))
        tensor = parts[0]
        if kind == "transpose":
            return tabulate(2, lambda index: tensor[index[1]][index[0]])
        diagonal = range(space_dimension)
        if kind == "trace":
            return total(tensor[row][row] for row in diagonal)
        if kind == "symmetric":
            return tabulate(
                2, lambda index: 0.5 * (tensor[index[0]][index[1]] + tensor[index[1]][index[0]])
            )
        if kind == "deviatoric":
            mean = total(tensor[row][row] for row in diagonal) / space_dimension
            return tabulate(
                2,
                lambda index: (
                    tensor[index[0]][index[1]] - mean
                    if index[0] == index[1]
                    else tensor[index[0]][index[1]]
                ),
            )
        if kind == "determinant":
            return determinant_and_cofactors(tensor, space_dimension)[0]
        if kind == "inverse":
            tensor_determinant, cofactors = determinant_and_cofactors(tensor, space_dimension)
            return tabulate(2, lambda index: cofactors[index[1]][index[0]] / tensor_determinant)
        # Scalar operations: power, the functions, maximum, minimum, less, greater,
        # interpolation and antiderivative.
        return rebuild(kind, tuple(parts), current.attributes())

    return walk(node)


def determinant_and_cofactors(tensor: list, space_dimension: int) -> tuple:
    """Return the determinant and the cofactor matrix of an expanded tensor."""
    if space_dimension == 1:
        return tensor[0][0], [[number(1.0)]]
    if space_dimension == 2:
        cofactors = [[tensor[1][1], -tensor[1][0]], [-tensor[0][1], tensor[0][0]]]
    else:
        cofactors = [
            [
                tensor[(row + 1) % 3][(column + 1) % 3] * tensor[(row + 2) % 3][(column + 2) % 3]
                - tensor[(row + 1) % 3][(column + 2) % 3] * tensor[(row + 2) % 3][(column + 1) % 3]
                for column in range(3)
            ]
            for row in range(3)
        ]
    tensor_determinant = reduce(
        operator.add,
        (tensor[0][column] * cofactors[0][column] for column in range(space_dimension)),
    )
    return tensor_determinant, cofactors


# Traversal


def substitute(node: expression, replacements: dict) -> expression:
    """Return node with every key of replacements replaced by its value, checked again."""
    replacements = {key: as_expression(value) for key, value in replacements.items()}
    substituted = {}

    def walk(current: expression) -> expression:
        if current in replacements:
            return replacements[current]
        if not current.operands:
            return current
        if current in substituted:
            return substituted[current]
        operands = tuple(walk(operand) for operand in current.operands)
        if all(new is old for new, old in zip(operands, current.operands, strict=True)):
            result = current
        else:
            result = rebuild(current.kind, operands, current.attributes())
        substituted[current] = result
        return result

    return walk(as_expression(node))


def leaves(node: expression) -> set:
    """Return every node without operands in the graph of node."""
    found = set()
    seen = set()
    pending = [node]
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        if current.operands:
            pending.extend(current.operands)
        else:
            found.add(current)
    return found


# Text


precedence_of_kind = {"add": 1, "multiply": 2, "divide": 2, "negate": 3, "power": 4}
atom_precedence = 5


def precedence(node: expression) -> int:
    if node.kind == "number" and node.value < 0.0 and node.dimension.is_dimensionless():
        return precedence_of_kind["negate"]
    return precedence_of_kind.get(node.kind, atom_precedence)


def text(node: expression) -> str:
    """Return node as text for messages.

    The text follows the syntax of parse, except for components, which show
    their indices in square brackets as in `grad(displacement_x)[1]`. parse
    reads square brackets as a quantity, so such a text does not parse back.
    """
    kind = node.kind

    def wrapped(operand: expression, lowest: int) -> str:
        operand_text = text(operand)
        return f"({operand_text})" if precedence(operand) < lowest else operand_text

    if kind == "number":
        value = format(node.value, ".15g")
        return value if node.dimension.is_dimensionless() else f"[{value} {node.dimension}]"
    if kind in ("field", "state", "constant"):
        return node.name
    if kind in ("position", "normal", "identity"):
        return f"{kind}()"
    if kind in ("time", "time_step", "cell_size"):
        return kind
    if kind in ("field_component", "state_component"):
        return units.component_name(node.name, node.component)
    if kind == "gradient_component":
        return f"grad({units.component_name(node.name, node.component)})[{node.direction}]"
    if kind == "position_component":
        return f"position()[{node.direction}]"
    if kind == "normal_component":
        return f"normal()[{node.direction}]"
    operands = node.operands
    if kind == "add":
        second = operands[1]
        if second.kind == "negate":
            return f"{wrapped(operands[0], 1)} - {wrapped(second.operands[0], 2)}"
        return f"{wrapped(operands[0], 1)} + {wrapped(second, 1)}"
    if kind == "multiply":
        return f"{wrapped(operands[0], 2)}*{wrapped(operands[1], 2)}"
    if kind == "divide":
        return f"{wrapped(operands[0], 2)}/{wrapped(operands[1], 3)}"
    if kind == "negate":
        return f"-{wrapped(operands[0], 3)}"
    if kind == "power":
        return f"{wrapped(operands[0], 5)}^{wrapped(operands[1], 3)}"
    if kind == "component":
        return f"{wrapped(operands[0], 5)}[{', '.join(str(index) for index in node.component)}]"
    if kind == "interpolation":
        orders = "".join(
            f", derivative by {input_name}"
            for input_name, order in zip(node.table.input_names, node.table.derivative, strict=True)
            if order
        )
        return f"table({node.name} at {', '.join(text(operand) for operand in operands)}{orders})"
    return f"{kind}({', '.join(text(operand) for operand in operands)})"
