# SPDX-License-Identifier: LGPL-2.1-or-later
"""Every example of the user guide page "From paper to code"
(docs/user_guide/equations.rst) runs and gives a physically right result, so
that the page stays true to the code and to the physics.

Each example is checked against what its physics requires: a value of an
analytical solution, a maximum principle, or a conservation law.  An example
without a check is refused, so that a new example comes with one."""

import math
import re
from pathlib import Path

import numpy as np
import pytest

PAGE = Path(__file__).resolve().parents[2] / "docs" / "user_guide" / "equations.rst"
BLOCK = re.compile(r"\.\. code-block:: python\n\n((?:(?:   .*)?\n)+)")


def _values(env, field):
    return np.asarray(env["problem"].values(field))


def _within(values, low, high, tolerance=1e-6):
    span = max(1.0, abs(high - low))
    return values.min() >= low - tolerance * span and values.max() <= high + tolerance * span


def _poisson(env):
    # The maximum of -laplacian(u) = 1 on the unit square with u = 0 on its
    # sides, from the series solution: 0.0736713532.
    assert _values(env, "u").max() == pytest.approx(0.0736713532, rel=2e-3)


def _heat(env):
    # Heated by its source from 293 K, with the boundaries at or towards
    # 293 K: never colder than 293 K.
    assert _values(env, "T").min() >= 293.0 - 1e-6


def _advection(env):
    # A decaying species entering at c = 1: 0 <= c <= 1, and decaying
    # downstream.
    c = _values(env, "c")
    assert _within(c, 0.0, 1.0)
    inlet, outlet = env["problem"].sample("c", [[0.0, 0.5], [1.0, 0.5]])
    assert outlet < inlet


def _fisher(env):
    # Logistic growth from a value in [0, 1] stays in [0, 1].
    assert _within(_values(env, "u"), 0.0, 1.0, 1e-3)


def _positive(field):
    def check(env):
        # A positive source with zero boundary values: u >= 0.
        assert _values(env, field).min() >= -1e-10

    return check


def _allen_cahn(env):
    # The phase field stays between its two phases, -1 and 1.
    assert _within(_values(env, "phi"), -1.0, 1.0, 1e-2)


def _cahn_hilliard(env):
    # The Cahn-Hilliard equation conserves the mean composition (here 0).
    c = _values(env, "c")
    assert abs(c.mean()) < 1e-3


def _gray_scott(env):
    u, v = _values(env, "u"), _values(env, "v")
    assert _within(u, 0.0, 1.0, 1e-3)
    assert v.min() >= -1e-3


def _two_group(env):
    # A bare square of side 1 m: k from two-group theory with B^2 = 2 pi^2.
    buckling = 2 * math.pi**2
    exact = 13.5 * 2.0 / ((1.0 + 2.0 + 0.015 * buckling) * (8.0 + 0.004 * buckling))
    assert env["k"] == pytest.approx(exact, rel=2e-3)


def _axisymmetric(env):
    # Insulated ends: the radial solution T(r) = T_s + q0/k ((R^2 - r^2)/4
    # - (R^4 - r^4)/(16 R^2)), 2475 K on the axis.
    # Ten linear elements across the radius are within 1 % of the rise; the
    # error falls at second order as the mesh is refined.
    rise = 3.0e8 * 0.01**2 / 3.0 * 3.0 / 16.0
    centre = env["problem"].sample("T", [[0.0, 0.01]])[0]
    assert abs(centre - (600.0 + rise)) < 0.01 * rise


def _elasticity(env):
    # Clamped on the left and pulled down on the right: the right end moves
    # down.
    assert env["problem"].sample("U_y", [[1.0, 0.5]])[0] < 0.0


CHECKS = {
    "Poisson's equation": _poisson,
    "Transient heat conduction with a source": _heat,
    "Advection-diffusion": _advection,
    "Reaction-diffusion: the Fisher-KPP equation": _fisher,
    "Anisotropic diffusion": _positive("u"),
    "Nonlinear diffusion: the p-Laplacian": _positive("u"),
    "The Allen-Cahn equation": _allen_cahn,
    "The Cahn-Hilliard equation": _cahn_hilliard,
    "The Gray-Scott equations": _gray_scott,
    "Two-group neutron diffusion: an eigenvalue problem": _two_group,
    "Axisymmetric heat conduction": _axisymmetric,
    "Linear elasticity: a vector field": _elasticity,
}


def _code(block: str) -> str:
    return "\n".join(line[3:] for line in block.split("\n"))


def _examples():
    text = PAGE.read_text()
    setup = _code(BLOCK.search(text).group(1))
    start = text.index("\nExamples\n")
    examples = []
    for number, match in enumerate(BLOCK.finditer(text, start)):
        title = re.findall(r"\n([^\n]+)\n\^+\n", text[: match.start()])[-1]
        code = setup + "\n" + _code(match.group(1))
        examples.append(pytest.param(title, code, id=f"{number}-{title}"))
    return examples


@pytest.mark.parametrize("title, code", _examples())
def test_the_example_runs_and_is_physical(title, code):
    assert title in CHECKS, f"the example '{title}' has no physical check in CHECKS"
    env = {}
    exec(compile(code, str(PAGE), "exec"), env)
    CHECKS[title](env)
