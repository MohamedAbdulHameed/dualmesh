# SPDX-License-Identifier: LGPL-2.1-or-later
"""How a steady nonlinear solve starts, and how it reports that it diverged.

A steady flow at rest starts Newton's method from the Stokes solution, the same problem without
the inertia, which a linearisation of the inertia about prescribed inflow values next to zero
values inside cannot give.
"""

from __future__ import annotations

import dualmesh as dm
import numpy as np
import pytest


def channel(inflow, n, density=100.0):
    """A channel at Re = density that develops from the given inflow between no-slip walls."""
    mesh = dm.generate_rectangle_mesh(x=[0, 1], y=[0, 1], num_elements=[n, n])
    problem = dm.Problem(mesh)
    flow = problem.add_physics(
        "fluid_dynamics", density=density, dynamic_viscosity=1.0, formulation="pressure"
    )
    flow.add_boundary_condition(["velocity_x = 0", "velocity_y = 0"], boundary=["bottom", "top"])
    flow.add_boundary_condition([f"velocity_x = {inflow}", "velocity_y = 0"], boundary="left")
    return problem


FIELDS = ("velocity_x", "velocity_y", "pressure")


@pytest.mark.parametrize("n", [16, 32])
@pytest.mark.parametrize("inflow", ["0.5", "y", "6*y*(1-y)"])
def test_a_flow_at_rest_converges_to_the_solution_that_continuation_reaches(inflow, n):
    """Started at rest, the channel at Re = 100 converges in a few Newton steps to the solution
    that continuation from Re = 10 reaches."""
    problem = channel(inflow, n)
    result = problem.solve(report="none")
    assert result.total_iterations <= 7
    start = channel(inflow, n, density=10.0)
    start.solve(report="none")
    continued = channel(inflow, n)
    continued.initialize()
    for name in FIELDS:
        continued.set_values(name, start.values(name))
    continued.solve(report="none")
    for name in FIELDS:
        scale = np.abs(continued.values(name)).max()
        assert np.abs(problem.values(name) - continued.values(name)).max() <= 1e-8 * scale, name


def test_the_report_states_the_stokes_start(capfd):
    channel("y", 8, density=10.0).solve(report="full")
    assert "starts from the Stokes solution" in capfd.readouterr().out


def test_a_velocity_given_as_the_start_is_kept(capfd):
    """A start that the user sets is not replaced by the Stokes solution."""
    problem = channel("y", 8, density=10.0)
    problem.initialize()
    problem.set_values("velocity_x", 0.5 * np.ones(len(problem.entity_points())))
    problem.solve(report="full")
    assert "starts from the Stokes solution" not in capfd.readouterr().out


def test_a_problem_without_inertia_does_not_start_from_stokes(capfd):
    channel("y", 8, density=0.0).solve(report="full")
    assert "Stokes solution" not in capfd.readouterr().out


def test_a_diverging_newton_iteration_is_reported_as_such():
    """-u'' = 5 exp(u), u(0) = u(1) = 0, has no solution (the Bratu problem above its critical
    value 3.51): the residual grows, and the solve says that Newton's method diverged rather than
    handing the linear solver a matrix it cannot factorise."""
    mesh = dm.generate_line_mesh(start=0.0, end=1.0, num_elements=20)
    problem = dm.Problem(mesh, method="fem")
    problem.add_variable("u")
    problem.add_kernel("diffusion", variable="u", diffusivity=1.0)
    problem.add_kernel("parsed_kernel", variable="u", source="5*exp(u)")
    problem.add_boundary_condition("Dirichlet", ["left", "right"], variable="u", value=0.0)
    with pytest.raises(RuntimeError, match="the residual grew from"):
        problem.solve(report="none", max_iterations=200)
