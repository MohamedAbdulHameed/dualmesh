"""The dimension of every expression node, checked as the graph is built.

`expression` calls `dimension_of` in its constructor, so a node with
inconsistent dimensions never exists. The function reads only the attributes
`kind`, `value` and `dimension` of the operands, so this module does not
import `expression`. It also holds the set of leaf kinds, the length dimension
and the zero test, which `expression` and `differentiate` import from here.

The rules:

- Sums, `maximum`, `minimum`, `less`, `greater` and both branches of
  `conditional` need operands of equal dimension. A dimensionless zero, such
  as the Python number 0, matches every dimension, since zero is zero in every
  unit.
- `exp`, `log`, `sin`, `cos` and `tanh` need a dimensionless operand.
- A power of an operand with a dimension needs a `number` exponent. The
  exponent becomes a `Fraction` with a denominator of at most 12.
- `determinant` needs a dimensionless operand, since the dimension of the
  result depends on the space dimension, which the graph does not know.
- An `interpolation` carries the dimension of its values, divided by the
  dimension of each input by which it is differentiated.
- An `antiderivative` has the dimension of its integrand times that of its
  variable.
"""

from fractions import Fraction

from dualmesh.input.units import base, dimension, dimensionless, input_error


class dimension_error(input_error):
    """An expression whose dimensions do not agree."""


largest_exponent_denominator = 12

length = base("length")

# Kinds whose dimension is the dimension of their first operand.
same_as_first_operand = frozenset(
    {
        "negate",
        "component",
        "abs",
        "transpose",
        "trace",
        "symmetric",
        "deviatoric",
        "norm",
    }
)

# Kinds that need a dimensionless operand and give a dimensionless result.
dimensionless_functions = frozenset({"exp", "log", "sin", "cos", "tanh"})

# Kinds whose operands need equal dimensions.
equal_dimension_kinds = frozenset({"add", "maximum", "minimum", "less", "greater"})

# Leaves carry their dimension as an attribute.
leaf_kinds = frozenset(
    {
        "field",
        "state",
        "constant",
        "number",
        "position",
        "time",
        "time_step",
        "cell_size",
        "normal",
        "identity",
        "field_component",
        "gradient_component",
        "state_component",
        "position_component",
        "normal_component",
    }
)


def is_zero(node, dimensionless_only: bool = False) -> bool:
    """Return whether node is a `number` of value 0.

    With dimensionless_only, the number must also be dimensionless.
    """
    if node.kind != "number" or node.value != 0.0:
        return False
    return not dimensionless_only or node.dimension.is_dimensionless()


def describe(node) -> str:
    """Return the text of node with its dimension, for error messages."""
    unit = "1" if node.dimension.is_dimensionless() else str(node.dimension)
    return f"'{node}' [{unit}]"


def common_dimension(kind: str, first, second) -> dimension:
    """Return the dimension that two operands share, or raise dimension_error."""
    if first.dimension == second.dimension or is_zero(second, dimensionless_only=True):
        return first.dimension
    if is_zero(first, dimensionless_only=True):
        return second.dimension
    raise dimension_error(
        f"{kind} needs operands of equal dimension, got {describe(first)} and {describe(second)}"
    )


def exponent_fraction(exponent) -> Fraction:
    """Return the exponent of a power of a quantity with a dimension as a Fraction."""
    if exponent.kind != "number":
        raise dimension_error(
            f"power of a quantity with a dimension needs a number exponent, got {describe(exponent)}"
        )
    fraction = Fraction(exponent.value).limit_denominator(largest_exponent_denominator)
    if abs(float(fraction) - exponent.value) > 1.0e-12 * max(1.0, abs(exponent.value)):
        raise dimension_error(
            f"power of a quantity with a dimension needs an exponent that is a fraction with a "
            f"denominator of at most {largest_exponent_denominator}, got {exponent.value!r}"
        )
    return fraction


def dimension_of(kind: str, operands: tuple, **attributes) -> dimension:
    """Return the dimension of a node of kind with operands, or raise dimension_error.

    A leaf or an interpolation passes its dimension in the attribute `physical_dimension`.
    """
    if kind in leaf_kinds:
        leaf_dimension = attributes.get("physical_dimension")
        return dimensionless if leaf_dimension is None else leaf_dimension
    if kind in same_as_first_operand:
        return operands[0].dimension
    if kind in equal_dimension_kinds:
        result = common_dimension(kind, operands[0], operands[1])
        return dimensionless if kind in ("less", "greater") else result
    if kind in dimensionless_functions:
        if not operands[0].dimension.is_dimensionless():
            raise dimension_error(
                f"{kind} needs a dimensionless argument, got {describe(operands[0])}"
            )
        return dimensionless
    if kind in ("multiply", "dot", "inner", "outer"):
        return operands[0].dimension * operands[1].dimension
    if kind == "divide":
        return operands[0].dimension / operands[1].dimension
    if kind == "power":
        base_operand, exponent = operands
        if not exponent.dimension.is_dimensionless():
            raise dimension_error(f"power needs a dimensionless exponent, got {describe(exponent)}")
        if base_operand.dimension.is_dimensionless():
            return dimensionless
        return base_operand.dimension ** exponent_fraction(exponent)
    if kind == "sqrt":
        return operands[0].dimension ** Fraction(1, 2)
    if kind == "grad":
        return operands[0].dimension / length
    if kind == "interpolation":
        if attributes.get("physical_dimension") is None:
            raise dimension_error("an interpolation needs the dimension of its values")
        return attributes["physical_dimension"]
    if kind == "antiderivative":
        return operands[0].dimension * operands[1].dimension
    if kind == "conditional":
        condition, if_true, if_false = operands
        if condition.kind not in ("less", "greater"):
            raise dimension_error(
                f"conditional needs a condition made by less or greater, got '{condition}'"
            )
        return common_dimension(kind, if_true, if_false)
    if kind == "inverse":
        return operands[0].dimension ** -1
    if kind == "determinant":
        if not operands[0].dimension.is_dimensionless():
            raise dimension_error(
                f"determinant needs a dimensionless operand, got {describe(operands[0])}. "
                f"Divide the operand by a scale with its dimension first"
            )
        return dimensionless
    raise dimension_error(f"unknown expression kind '{kind}'")
