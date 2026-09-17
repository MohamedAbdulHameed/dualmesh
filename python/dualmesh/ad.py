# SPDX-License-Identifier: LGPL-2.1-or-later
"""Elementary functions that work for both floats and AD numbers.

Kernels written in Python receive :class:`dualmesh.ADReal` values, whose
derivatives with respect to the local degrees of freedom are carried along so
that the Jacobian used by Newton's method stays exact.  Use these functions
instead of :mod:`math` inside such kernels::

    import dualmesh as dm

    class ArrheniusReaction(dm.PythonKernel):
        def compute_source(self, ctx):
            temperature = ctx.value(self.temperature_index)
            return self.pre_exponential * dm.exp(-self.activation / temperature)
"""

from __future__ import annotations

import builtins
import math

from . import _core

ADReal = _core.ADReal


def _dispatch(name):
    ad_function = getattr(_core, name)
    math_function = getattr(math, name)

    def wrapper(value):
        if isinstance(value, ADReal):
            return ad_function(value)
        return math_function(value)

    wrapper.__name__ = name
    wrapper.__doc__ = f"{name}(x) for floats and AD numbers."
    return wrapper


sqrt = _dispatch("sqrt")
exp = _dispatch("exp")
log = _dispatch("log")
sin = _dispatch("sin")
cos = _dispatch("cos")
tanh = _dispatch("tanh")


def abs(value):  # noqa: A001 - mirrors the builtin on purpose
    """Absolute value for floats and AD numbers."""
    if isinstance(value, ADReal):
        return _core.abs(value)
    return builtins.abs(value)


def pow(base, exponent):  # noqa: A001 - mirrors the builtin on purpose
    """Power for floats and AD numbers."""
    if isinstance(base, ADReal) or isinstance(exponent, ADReal):
        return base**exponent
    return base**exponent
