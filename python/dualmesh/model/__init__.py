"""The model language: expressions, their dimensions and derivatives, the parser and the conservation law.

The modules:

- `graph` builds the expression graph, and `dimensions` checks the dimension of each node.
- `calculus` differentiates an expression.
- `syntax` parses the formulas and the equations into graphs.
- `conservation` holds the conservation law that a model is made of.
- `engine_form` writes a graph in the expression language of the engine and adds a law to the
  engine, and `sympy_form` converts a graph to SymPy and back.

The package gives the names that a model is written with, such as `expression`, `field`,
`differentiate`, `parse` and `conservation_law`. No name of the package is the name of one of
its modules.
"""

from dualmesh.model.calculus import differentiate
from dualmesh.model.conservation import conservation_law
from dualmesh.model.dimensions import dimension_error

# The names are those of graph.__all__.
from dualmesh.model.graph import *  # noqa: F403
from dualmesh.model.graph import __all__ as expression_names
from dualmesh.model.syntax import (
    conservation_law_from_input,
    equation_scope,
    parse,
    read_condition,
    read_equations,
    syntax_error,
)

__all__ = [
    *expression_names,
    "dimension_error",
    "differentiate",
    "parse",
    "syntax_error",
    "conservation_law_from_input",
    "conservation_law",
    "equation_scope",
    "read_equations",
    "read_condition",
]
