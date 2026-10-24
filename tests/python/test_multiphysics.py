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

import dualmesh as dm
import pytest

_EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "natural_convection.py"
_spec = importlib.util.spec_from_file_location("natural_convection", _EXAMPLE)
natural_convection = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(natural_convection)


@pytest.mark.parametrize("method", ["dmcdm", "fem"])
@pytest.mark.parametrize("rayleigh_number", [1.0e3, 1.0e4, 1.0e5])
def test_natural_convection_matches_de_vahl_davis(method, rayleigh_number):
    """On a 32 by 32 mesh graded towards the walls, the average Nusselt number
    is within 0.5 % of the benchmark and the velocity maxima within 2 %.  With
    the default streamline stabilisation of the heat the measured differences
    are up to 0.35 % and 1.5 % (FEM, Ra = 10^5).  It leaves the conduction term
    out of its residual, which costs some accuracy where conduction matters as
    much as convection, as it does here.  Without it (advection_scheme
    'central', below) they are below 0.2 % and 0.7 %."""
    problem, result = natural_convection.solve(rayleigh_number, 32, method)
    computed = natural_convection.measure(problem)
    reference = natural_convection.BENCHMARK[rayleigh_number]
    assert computed["nusselt"] == pytest.approx(reference["nusselt"], rel=5e-3)
    assert computed["u_max"] == pytest.approx(reference["u_max"], rel=2e-2)
    assert computed["v_max"] == pytest.approx(reference["v_max"], rel=2e-2)
    # Energy balance: the heat entering through the hot wall leaves through
    # the cold one, because the top and bottom are adiabatic.
    assert problem.total_reaction("temperature", "right") == pytest.approx(
        -computed["nusselt"], rel=1e-9
    )
    # Newton's method with the exact coupled Jacobian: a handful of
    # iterations per decade of Rayleigh number.
    decades = len([r for r in (1e3, 1e4, 1e5, 1e6) if r <= rayleigh_number])
    assert result.total_iterations <= 8 * decades


def test_the_coupling_is_exact_in_newtons_method():
    """Near the solution Newton's method converges quadratically only if the
    Jacobian is exact, including the blocks that couple the temperature and
    the velocities.  The relative size of the Newton step, s_k, must then
    satisfy s_{k+1} <= C s_k^2 once it is small.  The residual itself is not
    used, because with a penalty parameter of 10^7 it reaches its round-off
    floor (about 10^-7) one step before the solution stops changing.  The
    measured steps on the last load step are 1.8e-3, 1.5e-6 and 2.1e-12, so
    C is about 0.5 and 0.9."""
    problem, result = natural_convection.solve(1.0e4, 16, "dmcdm")
    steps = [r.step_norm for r in result.history if r.load_factor == 1.0 and r.step_norm > 1e-15]
    ratios = [steps[i + 1] / steps[i] ** 2 for i in range(len(steps) - 1) if steps[i] < 1e-2]
    assert len(ratios) >= 2, "the last load step did not reach the quadratic regime"
    assert max(ratios) < 10.0


@pytest.mark.parametrize("method", ["FEM", "DMCDM"])
def test_natural_convection_without_streamline_stabilisation_of_the_heat(method):
    """The heat carried with advection_scheme 'central' at Ra = 10^5, where conduction matters
    as much as convection: closer than the default streamline stabilisation (measured below
    0.1 % for the Nusselt number and 0.7 % for the velocity maxima)."""
    problem, _ = natural_convection.solve(1.0e5, 32, method, advection_scheme="central")
    computed = natural_convection.measure(problem)
    reference = natural_convection.BENCHMARK[1.0e5]
    assert computed["nusselt"] == pytest.approx(reference["nusselt"], rel=2e-3)
    assert computed["u_max"] == pytest.approx(reference["u_max"], rel=1e-2)
    assert computed["v_max"] == pytest.approx(reference["v_max"], rel=1e-2)


def test_the_physics_and_couplings_equal_the_objects():
    """The fluid_dynamics and heat_transfer physics with the
    heat_convection and Boussinesq_buoyancy couplings of the example generate
    the same equations as the objects written one by one, and give the same
    answer to round-off."""
    reference, _ = natural_convection.solve(1.0e4, 16)
    problem = dm.Problem(natural_convection.cavity_mesh(16), method="DMCDM")
    velocities = ["velocity_x", "velocity_y"]
    for variable in velocities:
        problem.add_variable(variable)
    problem.add_variable("temperature")
    for variable in velocities:
        problem.add_kernel(
            "viscous_stress",
            variable=variable,
            velocities=velocities,
            dynamic_viscosity=0.71,
        )
        problem.add_kernel(
            "penalty_incompressibility",
            variable=variable,
            velocities=velocities,
            penalty_parameter=1e7,
        )
        problem.add_kernel(
            "convective_inertia",
            variable=variable,
            velocities=velocities,
            density=1.0,
        )
    problem.add_property("penalty_pressure", velocities=velocities)
    problem.add_kernel("heat_conduction", variable="temperature")
    problem.add_kernel("heat_convection", variable="temperature", velocities=velocities)
    # The streamline stabilisation of the heat, which the coupling adds by default.
    problem.add_kernel(
        "heat_streamline_stabilization",
        variable="temperature",
        velocities=velocities,
        thermal_conductivity="1.0",
    )
    problem.add_kernel(
        "Boussinesq_buoyancy",
        variable="velocity_y",
        velocities=["velocity_x", "velocity_y"],
        temperature="temperature",
        gravity=[0.0, -1.0],
        vol_expansion_coeff=1.0e4 * 0.71,
        reference_temperature=0.0,
        scale_with_load=True,
    )
    for variable in velocities:
        problem.add_boundary_condition(
            "Dirichlet",
            ["left", "right", "bottom", "top"],
            variable=variable,
            value=0.0,
        )
    problem.add_boundary_condition("Dirichlet", "left", variable="temperature", value=0.5)
    problem.add_boundary_condition("Dirichlet", "right", variable="temperature", value=-0.5)
    problem.solve(load_factors=[0.1, 1.0], max_iterations=40)
    assert problem.values("temperature") == pytest.approx(
        reference.values("temperature"), abs=1e-10
    )
