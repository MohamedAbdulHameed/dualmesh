"""Converts expression graphs to SymPy and back.

`to_sympy` writes a scalar expression, as `expand` returns it, as a SymPy
expression. A function of the caller gives the SymPy form of each leaf, such as
the exact solution of a field and its derivatives for a manufactured solution.
A comparison becomes 1 where it holds and 0 elsewhere, and a `conditional` a
`Piecewise`.

`from_sympy` reads a SymPy expression back into a dimensionless graph, which
the printer of the engine (`dualmesh.model.engine_form.write_expression`) writes.
The symbols x, y and z become the coordinates and t the time, and any other
symbol a name that the engine resolves.

`coordinate_symbols`, `sympy_expression` and `sympy_gradient` give the SymPy
forms that a manufactured solution is written in, in the symbols x, y, z and t.

SymPy is needed only here, and it is imported when a conversion runs.
"""

import operator
from functools import reduce

from dualmesh._core import InputError
from dualmesh.model import graph
from dualmesh.model.dimensions import comparisons
from dualmesh.units import dimensionless


def sympy_module():
    try:
        import sympy
    except ImportError as error:  # pragma: no cover - depends on the environment
        raise ImportError(
            "the manufactured solutions and the conversion to SymPy need SymPy. Install it with "
            "'pip install sympy'"
        ) from error
    return sympy


def coordinate_symbols():
    """Return the real SymPy symbols x, y, z and t."""
    return sympy_module().symbols("x y z t", real=True)


def sympy_expression(value):
    """Return a number, a text or a SymPy expression as a SymPy expression in x, y, z and t."""
    sympy = sympy_module()
    if isinstance(value, str):
        x, y, z, t = coordinate_symbols()
        names = {"x": x, "y": y, "z": z, "t": t, "pi": sympy.pi, "e": sympy.E}
        return sympy.sympify(value, locals=names)
    return sympy.sympify(value)


def sympy_gradient(u):
    """Return the derivatives of the SymPy expression u along x, y and z."""
    x, y, z, _t = coordinate_symbols()
    return [u.diff(x), u.diff(y), u.diff(z)]


def number_form(value: float):
    """Return a SymPy integer for an integral value, which keeps polynomials exact, else a float."""
    sympy = sympy_module()
    return sympy.Integer(int(value)) if float(value).is_integer() else sympy.Float(value)


def to_sympy(node, leaf):
    """Return the SymPy form of the scalar expression node.

    leaf(node) returns the SymPy form of a scalar leaf other than a number or a constant with a
    value, which become their values.
    """
    sympy = sympy_module()
    functions = {
        "negate": operator.neg,
        "exp": sympy.exp,
        "log": sympy.log,
        "log10": lambda argument: sympy.log(argument, 10),
        "sqrt": sympy.sqrt,
        "sin": sympy.sin,
        "cos": sympy.cos,
        "tan": sympy.tan,
        "asin": sympy.asin,
        "acos": sympy.acos,
        "atan": sympy.atan,
        "sinh": sympy.sinh,
        "cosh": sympy.cosh,
        "tanh": sympy.tanh,
        "erf": sympy.erf,
        "floor": sympy.floor,
        "ceil": sympy.ceiling,
        "sign": sympy.sign,
        "abs": sympy.Abs,
    }
    pairs = {
        "add": operator.add,
        "multiply": operator.mul,
        "divide": operator.truediv,
        "power": operator.pow,
        "maximum": sympy.Max,
        "minimum": sympy.Min,
        "atan2": sympy.atan2,
        "hypot": lambda first, second: sympy.sqrt(first**2 + second**2),
    }
    relations = {
        "less": sympy.Lt,
        "greater": sympy.Gt,
        "less_equal": sympy.Le,
        "greater_equal": sympy.Ge,
        "equal": sympy.Eq,
        "not_equal": sympy.Ne,
    }
    converted = {}

    def relation(condition):
        if condition.kind in comparisons:
            first, second = (walk(operand) for operand in condition.operands)
            return relations[condition.kind](first, second)
        return sympy.Ne(walk(condition), 0)

    def walk(current):
        if current in converted:
            return converted[current]
        kind = current.kind
        if kind == "number" or (kind == "constant" and current.value is not None):
            result = number_form(current.value)
        elif not current.operands:
            result = leaf(current)
        elif kind in functions:
            result = functions[kind](walk(current.operands[0]))
        elif kind in pairs:
            result = pairs[kind](*(walk(operand) for operand in current.operands))
        elif kind in comparisons:
            result = sympy.Piecewise((1, relation(current)), (0, True))
        elif kind == "conditional":
            condition, if_true, if_false = current.operands
            result = sympy.Piecewise((walk(if_true), relation(condition)), (walk(if_false), True))
        else:
            raise InputError(
                f"'{current}' has no SymPy form. Expand it into scalar expressions first"
            )
        converted[current] = result
        return result

    return walk(node)


def from_sympy(expression):
    """Return the dimensionless graph of a SymPy expression in x, y, z and t."""
    sympy = sympy_module()
    expression = sympy.sympify(expression)
    functions = {
        "exp": graph.exp,
        "log": graph.log,
        "sin": graph.sin,
        "cos": graph.cos,
        "tan": graph.tan,
        "asin": graph.asin,
        "acos": graph.acos,
        "atan": graph.atan,
        "sinh": graph.sinh,
        "cosh": graph.cosh,
        "tanh": graph.tanh,
        "erf": graph.erf,
        "floor": graph.floor,
        "ceiling": graph.ceil,
        "sign": graph.sign,
        "Abs": abs,
        "Max": graph.maximum,
        "Min": graph.minimum,
        "atan2": graph.atan2,
        "Add": operator.add,
        "Mul": operator.mul,
        "Pow": operator.pow,
        "And": graph.logical_and,
        "Or": graph.logical_or,
        "StrictLessThan": graph.less,
        "StrictGreaterThan": graph.greater,
        "LessThan": graph.less_equal,
        "GreaterThan": graph.greater_equal,
        "Equality": graph.equal,
        "Unequality": graph.not_equal,
    }
    coordinates = ("x", "y", "z")
    converted = {}

    def walk(current):
        if current in converted:
            return converted[current]
        if current is sympy.true:
            result = graph.number(1.0)
        elif current.is_Number or current.is_NumberSymbol:
            result = graph.number(float(current))
        elif current.is_Symbol and current.name in coordinates:
            result = graph.build(
                "position_component",
                direction=coordinates.index(current.name),
                rank=0,
                physical_dimension=dimensionless,
            )
        elif current.is_Symbol and current.name == "t":
            result = graph.build("time", rank=0, physical_dimension=dimensionless)
        elif current.is_Symbol:
            result = graph.build(
                "constant", name=current.name, rank=0, physical_dimension=dimensionless
            )
        elif isinstance(current, sympy.Piecewise):
            # The last pair holds where no pair before it holds.
            result = None
            for value, condition in reversed(current.args):
                if result is None:
                    result = walk(value)
                else:
                    result = graph.conditional(walk(condition), walk(value), result)
        elif type(current).__name__ in functions:
            function = functions[type(current).__name__]
            arguments = [walk(argument) for argument in current.args]
            result = function(*arguments) if len(arguments) == 1 else reduce(function, arguments)
        else:
            raise InputError(
                f"the SymPy expression '{current}' holds {type(current).__name__}, which the "
                f"engine does not evaluate"
            )
        converted[current] = result
        return result

    return walk(expression)
