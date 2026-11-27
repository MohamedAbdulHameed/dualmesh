"""Symbolic derivatives of scalar expressions.

`differentiate` takes a rank-0 expression whose leaves are scalar leaves, as
`expand` returns it, and a scalar leaf such as a `field_component`, a
`gradient_component` or a `state_component`. It applies the sum, product,
quotient and chain rules. The node constructors fold numbers, remove zeros
and ones, and fold a `conditional` whose branches are both zero. The
derivative of a term that does not read the variable is therefore a `number`
zero.

The derivative of $|a|$ is $-a'$ where $a < 0$ and $a'$ elsewhere. The
derivative of $\\max(a, b)$ is $a'$ where $a > b$ and $b'$ elsewhere, and the
derivative of $\\min(a, b)$ is $a'$ where $a < b$ and $b'$ elsewhere. A
comparison, `floor`, `ceil` and `sign` are constant between their jumps, so
their derivative is zero, as in the automatic differentiation of the engine.

The derivative of a table, an `interpolation` node, follows the chain rule
through its inputs. Each partial derivative is the `interpolation` node of the
same grid with that derivative order set, which the kernel evaluates as the
exact slope of the interpolant.

The derivative of an `antiderivative` $H(u) = \\int^{u} f(u')\\,\\mathrm{d}u'$ by
its variable $u$ is the integrand $f$. Its derivative by a leaf that $f$ does
not read is zero. Any other derivative, by time included, stops with an
`InputError`, since $H$ is never evaluated.

`derivative_along` applies the chain rule over every leaf, for a derivative
along a path on which each leaf changes at a given rate, such as the
derivative along a direction of space.
"""

import math

from dualmesh._core import InputError
from dualmesh.model.dimensions import comparisons, is_zero
from dualmesh.model.graph import (
    conditional,
    cos,
    cosh,
    exp,
    expression,
    greater,
    interpolation_partial,
    leaves,
    less,
    log,
    number,
    sin,
    sinh,
    sqrt,
    tanh,
)

# The derivative of each function of one scalar a by its argument, from a and the node f(a).
outer_derivatives = {
    "exp": lambda a, node: node,
    "log": lambda a, node: 1.0 / a,
    "log10": lambda a, node: 1.0 / (math.log(10.0) * a),
    "sqrt": lambda a, node: 1.0 / (2.0 * node),
    "sin": lambda a, node: cos(a),
    "cos": lambda a, node: -sin(a),
    "tan": lambda a, node: 1.0 / cos(a) ** 2,
    "asin": lambda a, node: 1.0 / sqrt(1.0 - a**2),
    "acos": lambda a, node: -1.0 / sqrt(1.0 - a**2),
    "atan": lambda a, node: 1.0 / (1.0 + a**2),
    "sinh": lambda a, node: cosh(a),
    "cosh": lambda a, node: sinh(a),
    "tanh": lambda a, node: 1.0 - tanh(a) ** 2,
    "erf": lambda a, node: 2.0 / math.sqrt(math.pi) * exp(-(a**2)),
}
# Kinds that are constant between their jumps.
piecewise_constant = frozenset({"floor", "ceil", "sign", *comparisons})


def differentiate(node: expression, variable: expression) -> expression:
    """Return the derivative of node with respect to the scalar leaf variable.

    The result has the dimension of node over the dimension of variable.
    """
    if not isinstance(node, expression) or node.rank != 0:
        raise InputError(f"differentiate needs a scalar expression, got '{node}'. Expand it first")
    if not isinstance(variable, expression) or variable.rank != 0 or variable.operands:
        raise InputError(f"differentiate needs a scalar leaf as the variable, got '{variable}'")
    derivatives = {}

    def zero(current: expression) -> expression:
        # A dimensionless zero matches every dimension, so its derivative stays a dimensionless
        # zero and still matches the derivative beside it, as in maximum(temperature, 0).
        if is_zero(current, dimensionless_only=True):
            return current
        return number(0.0, current.dimension / variable.dimension)

    def derivative(current: expression) -> expression:
        if current in derivatives:
            return derivatives[current]
        result = derivative_of_node(current)
        derivatives[current] = result
        return result

    def derivative_of_node(current: expression) -> expression:
        if current == variable:
            return number(1.0)
        if not current.operands:
            return zero(current)
        kind = current.kind
        operands = current.operands
        if current.rank != 0 or any(operand.rank != 0 for operand in operands):
            raise InputError(
                f"differentiate needs a scalar expression, got '{current}'. Expand it first"
            )
        if kind == "conditional":
            condition, if_true, if_false = operands
            return conditional(condition, derivative(if_true), derivative(if_false))
        if kind in piecewise_constant:
            return zero(current)
        if kind == "interpolation":
            result = zero(current)
            for index, table_input in enumerate(operands):
                result = result + interpolation_partial(current, index) * derivative(table_input)
            return result
        if kind == "antiderivative":
            integrand, antiderivative_variable = operands
            if antiderivative_variable == variable:
                return integrand
            if is_zero(derivative(integrand)) and is_zero(derivative(antiderivative_variable)):
                return zero(current)
            raise InputError(
                f"{current}: an antiderivative is differentiated only by its variable "
                f"{antiderivative_variable}, here by {variable}"
            )
        first = operands[0]
        first_derivative = derivative(first)
        if kind in outer_derivatives:
            return outer_derivatives[kind](first, current) * first_derivative
        if kind == "negate":
            return -first_derivative
        if kind == "abs":
            return conditional(less(first, 0.0), -first_derivative, first_derivative)
        second = operands[1]
        second_derivative = derivative(second)
        if kind == "add":
            return first_derivative + second_derivative
        if kind == "multiply":
            return first_derivative * second + first * second_derivative
        if kind == "divide":
            return first_derivative / second - first * second_derivative / second**2
        if kind == "power":
            if second.kind == "number":
                return second * first ** (second.value - 1.0) * first_derivative
            # An exponent that does not read the variable, such as a named constant, takes the
            # power rule, which stays finite at a zero base.
            if is_zero(second_derivative):
                return second * first ** (second - 1.0) * first_derivative
            # Here the base is dimensionless, so log(base) exists.
            return current * (second_derivative * log(first) + second * first_derivative / first)
        if kind == "maximum":
            return conditional(greater(first, second), first_derivative, second_derivative)
        if kind == "minimum":
            return conditional(less(first, second), first_derivative, second_derivative)
        if kind == "atan2":
            return (second * first_derivative - first * second_derivative) / (first**2 + second**2)
        if kind == "hypot":
            return (first * first_derivative + second * second_derivative) / current
        raise InputError(
            f"differentiate does not know the kind '{kind}' of '{current}'. Expand it first"
        )

    return derivative(node)


def derivative_along(node: expression, rate_of) -> expression:
    """Return the derivative of the scalar node along a path, by the chain rule over its leaves.

    rate_of(leaf) returns the rate of change of a scalar leaf along the path, or None for a leaf
    that does not change. The leaves are taken in the order of their text, so the result does
    not depend on the order of a set.
    """
    result = number(0.0)
    for leaf in sorted(leaves(node), key=str):
        leaf_rate = rate_of(leaf)
        if leaf_rate is not None:
            result = result + differentiate(node, leaf) * leaf_rate
    return result
