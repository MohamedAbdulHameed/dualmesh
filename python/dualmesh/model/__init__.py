"""The model language: expressions, their dimensions and derivatives, the parser and the conservation law.

The package binds `expression`, `differentiate`, `parse` and `conservation_law`
to the class or the function of that name. Those names hide the modules of the
same spelling, so code inside the package loads such a module with
`importlib.import_module("dualmesh.model.expression")`, which returns the
module itself.
"""

from dualmesh.model.conservation_law import conservation_law
from dualmesh.model.differentiate import differentiate
from dualmesh.model.dimensions import dimension_error

# The names are those of expression.__all__.
from dualmesh.model.expression import *  # noqa: F403
from dualmesh.model.expression import __all__ as expression_names
from dualmesh.model.parse import conservation_law_from_input, parse, syntax_error

__all__ = [
    *expression_names,
    "dimension_error",
    "differentiate",
    "parse",
    "syntax_error",
    "conservation_law_from_input",
    "conservation_law",
]
