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
derivative of $\\min(a, b)$ is $a'$ where $a < b$ and $b'$ elsewhere.

The derivative of a table, an `interpolation` node, follows the chain rule
through its inputs. Each partial derivative is the `interpolation` node of the
same grid with that derivative order set, which the kernel evaluates as the
exact slope of the interpolant.

The derivative of an `antiderivative` $H(u) = \\int^{u} f(u')\\,\\mathrm{d}u'$ by
its variable $u$ is the integrand $f$. Its derivative by a leaf that $f$ does
not read is zero. Any other derivative, by time included, stops with an
`input_error`, since $H$ is never evaluated.
"""

from dualmesh.input.units import input_error
from dualmesh.model.dimensions import is_zero
from dualmesh.model.expression import (
    conditional,
    cos,
    exp,
    expression,
    greater,
    interpolation_partial,
    less,
    log,
    number,
    sin,
    sqrt,
    tanh,
)


def differentiate(node: expression, variable: expression) -> expression:
    """Return the derivative of node with respect to the scalar leaf variable.

    The result has the dimension of node over the dimension of variable.
    """
    if not isinstance(node, expression) or node.rank != 0:
        raise input_error(f"differentiate needs a scalar expression, got '{node}'. Expand it first")
    if not isinstance(variable, expression) or variable.rank != 0 or variable.operands:
        raise input_error(f"differentiate needs a scalar leaf as the variable, got '{variable}'")
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
            raise input_error(
                f"differentiate needs a scalar expression, got '{current}'. Expand it first"
            )
        if kind == "conditional":
            condition, if_true, if_false = operands
            return conditional(condition, derivative(if_true), derivative(if_false))
        if kind in ("less", "greater"):
            raise input_error(f"a condition has no derivative, got '{current}'")
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
            raise input_error(
                f"{current}: an antiderivative is differentiated only by its variable "
                f"{antiderivative_variable}, here by {variable}"
            )
        first = operands[0]
        first_derivative = derivative(first)
        if kind == "negate":
            return -first_derivative
        if kind == "add":
            return first_derivative + derivative(operands[1])
        if kind == "multiply":
            second = operands[1]
            return first_derivative * second + first * derivative(second)
        if kind == "divide":
            second = operands[1]
            return first_derivative / second - first * derivative(second) / second**2
        if kind == "power":
            exponent = operands[1]
            if exponent.kind == "number":
                return exponent * first ** (exponent.value - 1.0) * first_derivative
            # Here the base is dimensionless, so log(base) exists.
            return current * (
                derivative(exponent) * log(first) + exponent * first_derivative / first
            )
        if kind == "maximum":
            return conditional(
                greater(first, operands[1]), first_derivative, derivative(operands[1])
            )
        if kind == "minimum":
            return conditional(less(first, operands[1]), first_derivative, derivative(operands[1]))
        if kind == "exp":
            return exp(first) * first_derivative
        if kind == "log":
            return first_derivative / first
        if kind == "sqrt":
            return first_derivative / (2.0 * sqrt(first))
        if kind == "sin":
            return cos(first) * first_derivative
        if kind == "cos":
            return -sin(first) * first_derivative
        if kind == "tanh":
            return (1.0 - tanh(first) ** 2) * first_derivative
        if kind == "abs":
            return conditional(less(first, 0.0), -first_derivative, first_derivative)
        raise input_error(
            f"differentiate does not know the kind '{kind}' of '{current}'. Expand it first"
        )

    return derivative(node)
