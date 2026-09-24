# SPDX-License-Identifier: LGPL-2.1-or-later
"""Expressions given as text, turned into functions of (x, y, z, t).

This is the counterpart of MOOSE's ParsedFunction.  A coefficient, a source or
a boundary value can be written as an expression::

    import dualmesh as dm

    top = dm.parsed_function("500*(1 - 10*x^2)")
    problem.add_boundary_condition(
        "DirichletBC", variable="temperature", boundary="top", value=top)

or, more briefly, passed as the text itself, which is compiled the same way::

    problem.add_kernel("BodyForce", variable="u", value="sin(pi*x)*exp(-t)")

The expression is compiled once, to a short program evaluated in C++.  That
matters for speed and for threading: a Python callable would be called back at
every quadrature point of every element on every iteration, and because calling
into Python needs the interpreter lock it would also force the assembly onto a
single thread.  A compiled expression does neither.

The grammar is the usual one.  ``+ - * /`` have their usual precedence, ``^``
and ``**`` both mean exponentiation and bind tighter than a unary minus, and
comparisons ``< > <= >= == !=`` give 1 or 0.  The names in scope are ``x``,
``y``, ``z``, ``t``, the constants ``pi`` and ``e``, and the functions ``sin``,
``cos``, ``tan``, ``asin``, ``acos``, ``atan``, ``sinh``, ``cosh``, ``tanh``,
``exp``, ``log``, ``log10``, ``sqrt``, ``abs``, ``floor``, ``ceil``, ``erf``,
``sign``, ``atan2``, ``pow``, ``hypot``, ``min``, ``max`` and
``if(condition, a, b)``.  SymPy's printed form (``E``, ``Abs``, ``**``) is
accepted unchanged, so a manufactured source derived symbolically can be
passed straight in.
"""

from __future__ import annotations

from . import _core

ParsedFunction = _core.ParsedFunction


def parsed_function(expression: str) -> ParsedFunction:
    """Compile an expression in ``x``, ``y``, ``z`` and ``t``.

    The result is a :class:`dualmesh.ParsedFunction`, which every parameter
    that takes a function accepts, and which can also be called from Python as
    ``f(x, y, z, t)`` to check it.  A syntax error is reported with its
    position in the text.
    """
    return _core.ParsedFunction(str(expression))
