# SPDX-License-Identifier: LGPL-2.1-or-later
"""Expressions given as text, turned into functions of (x, y, z, t).

This is the counterpart of MOOSE's ParsedFunction: input files (and Python
code, if convenient) can describe a coefficient or a boundary value as an
expression::

    import dualmesh as dm

    top = dm.parsed_function("500*(1 - 10*x^2)")
    problem.add_boundary_condition(
        "DirichletBC", variable="temperature", boundary="top", value=top)
"""

from __future__ import annotations

import math

_MATH_NAMESPACE = {
    name: getattr(math, name)
    for name in (
        "sin",
        "cos",
        "tan",
        "asin",
        "acos",
        "atan",
        "atan2",
        "sinh",
        "cosh",
        "tanh",
        "exp",
        "log",
        "log10",
        "sqrt",
        "pi",
        "e",
        "fabs",
        "pow",
        "erf",
        "floor",
        "ceil",
        "hypot",
    )
}
_MATH_NAMESPACE["abs"] = abs
_MATH_NAMESPACE["min"] = min
_MATH_NAMESPACE["max"] = max


def parsed_function(expression: str):
    """Turn an expression in ``x``, ``y``, ``z``, ``t`` into a callable.

    The usual mathematical functions are available (``sin``, ``exp``,
    ``sqrt``, ...), ``^`` is accepted for exponentiation, and nothing else is
    in scope.  Input files are trusted input: do not feed expressions from an
    untrusted source to this function.
    """
    code = compile(expression.replace("^", "**"), "<dualmesh expression>", "eval")

    def evaluate(x=0.0, y=0.0, z=0.0, t=0.0):
        return float(eval(code, {"__builtins__": {}}, dict(_MATH_NAMESPACE, x=x, y=y, z=z, t=t)))

    evaluate.expression = expression
    return evaluate
