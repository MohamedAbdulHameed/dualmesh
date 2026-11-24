# SPDX-License-Identifier: LGPL-2.1-or-later
"""Two-way coupled multiphysics: natural convection in a heated cavity.

The benchmark of de Vahl Davis (1983) couples the momentum equations of an
incompressible flow to the energy equation in both directions: the buoyancy
force makes the temperature drive the flow, and convection makes the flow carry
the heat.  The problem is solved monolithically, by Newton's method with the
exact Jacobian of the whole coupled system, so the tests check both that the
physics is right (the published Nusselt numbers and velocity maxima) and that
the coupling blocks of the Jacobian are exact (the number of Newton
iterations).

The problem definition is the one of ``examples/natural_convection.py``, which
is loaded from the file rather than copied, so that the example and its test
cannot drift apart.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_example = Path(__file__).resolve().parents[2] / "examples" / "natural_convection.py"
_spec = importlib.util.spec_from_file_location("natural_convection", _example)
natural_convection = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(natural_convection)


@pytest.mark.parametrize("method", ["dmcdm", "fem"])
@pytest.mark.parametrize("Rayleigh_number", [1.0e3, 1.0e4, 1.0e5])
def test_natural_convection_matches_de_vahl_davis(method, Rayleigh_number):
    """On a 32 by 32 mesh graded towards the walls, the average Nusselt number
    is within 0.5 % of the benchmark and the velocity maxima within 2 %.  With
    the default streamline stabilisation of the heat the measured differences
    are up to 0.35 % and 1.5 % (FEM, Ra = 10^5).  It leaves the conduction term
    out of its residual, which costs some accuracy where conduction matters as
    much as convection, as it does here.  Without it (advection_scheme
    'central', below) they are below 0.2 % and 0.7 %."""
    problem, result = natural_convection.solve(Rayleigh_number, 32, method)
    computed = natural_convection.measure(problem)
    reference = natural_convection.benchmark[Rayleigh_number]
    assert computed["nusselt"] == pytest.approx(reference["nusselt"], rel=5e-3)
    assert computed["u_max"] == pytest.approx(reference["u_max"], rel=2e-2)
    assert computed["v_max"] == pytest.approx(reference["v_max"], rel=2e-2)
    # Energy balance: the heat entering through the hot wall leaves through
    # the cold one, because the top and bottom are adiabatic.
    assert -problem.boundary_flux("temperature", "right") == pytest.approx(
        -computed["nusselt"], rel=1e-9
    )
    # Newton's method with the exact coupled Jacobian: a handful of
    # iterations per decade of Rayleigh number.
    decades = len([r for r in (1e3, 1e4, 1e5, 1e6) if r <= Rayleigh_number])
    assert result.total_iterations <= 8 * decades


def test_the_coupling_is_exact_in_newtons_method():
    """Near the solution Newton's method converges quadratically only if the
    Jacobian is exact, including the blocks that couple the temperature and
    the velocities.  The relative size of the Newton step, s_k, must then
    satisfy s_{k+1} <= C s_k^2 once it is small.  The residual itself is not
    used, because with a penalty parameter of 10^7 it reaches its round-off
    floor (about 10^-7) one step before the solution stops changing.  The
    steps of the default solve after its Stokes start are 6.6e-2, 1.8e-3 and
    4.2e-7, so C is about 0.4 and 0.13."""
    problem, result = natural_convection.solve(1.0e4, 16, "dmcdm")
    # The iterations of the last solve: the history holds the Stokes start before it.
    last = max(i for i, r in enumerate(result.history) if r.iteration == 1)
    steps = [r.step_norm for r in result.history[last:] if r.step_norm > 1e-15]
    ratios = [steps[i + 1] / steps[i] ** 2 for i in range(len(steps) - 1) if steps[i] < 0.1]
    assert len(ratios) >= 2, "the solve did not reach the quadratic regime"
    assert max(ratios) < 10.0


@pytest.mark.parametrize("method", ["FEM", "DMCDM"])
def test_natural_convection_without_streamline_stabilisation_of_the_heat(method):
    """The heat carried with advection_scheme 'central' at Ra = 10^5, where conduction matters
    as much as convection: closer than the default streamline stabilisation (measured below
    0.1 % for the Nusselt number and 0.7 % for the velocity maxima)."""
    problem, _ = natural_convection.solve(1.0e5, 32, method, advection_scheme="central")
    computed = natural_convection.measure(problem)
    reference = natural_convection.benchmark[1.0e5]
    assert computed["nusselt"] == pytest.approx(reference["nusselt"], rel=2e-3)
    assert computed["u_max"] == pytest.approx(reference["u_max"], rel=1e-2)
    assert computed["v_max"] == pytest.approx(reference["v_max"], rel=1e-2)
