# SPDX-License-Identifier: LGPL-2.1-or-later
"""How a steady nonlinear solve starts, and how it reports that it diverged.

A steady flow at rest starts Newton's method from the Stokes solution, the same problem without
the inertia, which a linearisation of the inertia about prescribed inflow values next to zero
values inside cannot give.
"""

from __future__ import annotations

import dualmesh
import numpy
import pytest


def channel(inflow, n, density=100.0):
    """A channel at Re = density that develops from the given inflow between no-slip walls."""
    mesh = dualmesh.generate_rectangle_mesh(x=[0, 1], y=[0, 1], num_elements=[n, n])
    problem = dualmesh.Problem(mesh)
    flow = problem.add_physics(
        "fluid_dynamics", density=density, dynamic_viscosity=1.0, formulation="equal_order"
    )
    flow.add_boundary_condition(["velocity_x = 0", "velocity_y = 0"], boundary=["bottom", "top"])
    flow.add_boundary_condition([f"velocity_x = {inflow}", "velocity_y = 0"], boundary="left")
    return problem


fields = ("velocity_x", "velocity_y", "pressure")


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
    for name in fields:
        continued.set_values(name, start.values(name))
    continued.solve(report="none")
    for name in fields:
        scale = numpy.abs(continued.values(name)).max()
        assert numpy.abs(problem.values(name) - continued.values(name)).max() <= 1e-8 * scale, name


def test_a_diverging_newton_iteration_is_reported_as_such():
    """-u'' = 5 exp(u), u(0) = u(1) = 0, has no solution (the Bratu problem above its critical
    value 3.51): the residual grows, and the solve says that Newton's method diverged rather than
    handing the linear solver a matrix it cannot factorise."""
    mesh = dualmesh.generate_line_mesh(start=0.0, end=1.0, num_elements=20)
    problem = dualmesh.Problem(mesh, method="fem")
    problem.add_variable("u")
    problem.add_kernel("diffusion", variable="u", diffusivity=1.0)
    problem.add_kernel("parsed_kernel", variable="u", source="5*exp(u)")
    problem.add_boundary_condition("Dirichlet", ["left", "right"], variable="u", value=0.0)
    with pytest.raises(RuntimeError, match="the residual grew from"):
        problem.solve(report="none")


# ---- continuation ---------------------------------------------------------------------------
def cavity(Reynolds_number, n=32):
    """The lid-driven cavity at a Reynolds number, its pressure level held at a point."""
    mesh = dualmesh.generate_rectangle_mesh(x=[0, 1], y=[0, 1], num_elements=[n, n])
    problem = dualmesh.Problem(mesh)
    flow = problem.add_physics(
        "fluid_dynamics", density=float(Reynolds_number), dynamic_viscosity=1.0
    )
    flow.add_boundary_condition(
        ["velocity_x = 0", "velocity_y = 0"], boundary=["left", "right", "bottom"]
    )
    flow.add_boundary_condition(["velocity_x = 1", "velocity_y = 0"], boundary="top")
    flow.add_boundary_condition("pressure = 0", point=[0.5, 0.0])
    return problem


def test_a_cavity_at_a_high_Reynolds_number_solves_by_continuation():
    """At Re = 3200 Newton's method does not converge from the Stokes start, so the solve ramps the
    lid speed up from rest (measured: 5 steps).  The state it returns is the solution of the
    problem itself: a solve started from it converges at once, with the lid at its full speed."""
    problem = cavity(3200)
    result = problem.solve(report="none")
    steps = {r.continuation_step for r in result.history}
    assert result.converged and len(steps) > 1
    assert result.history[-1].data_fraction == 1.0
    lid = problem.nodes_where(lambda x, y, z: y == 1.0 and 0.0 < x < 1.0)
    assert problem.values("velocity_x")[lid] == pytest.approx(1.0, abs=1e-14)
    again = problem.solve(report="none")
    assert again.total_iterations <= 1


def test_continuation_ramps_the_prescribed_values_from_the_starting_state():
    """A prescribed temperature far from the start, with a conductivity that grows as T^6:
    Newton's method does not converge from 300 K, and the continuation takes the wall from 300 K,
    the starting state, to 3000 K (measured: 12 steps)."""
    mesh = dualmesh.generate_line_mesh(start=0.0, end=1.0, num_elements=20)
    problem = dualmesh.Problem(mesh)
    heat = problem.add_physics(
        "heat_conduction", thermal_conductivity="(temperature/300)^6", initial_condition=300.0
    )
    heat.add_boundary_condition("temperature = 300", boundary="left")
    heat.add_boundary_condition("temperature = 3000", boundary="right")
    result = problem.solve(report="none")
    assert result.converged and len({r.continuation_step for r in result.history}) > 1
    # `k` T^6 dT/dx is constant: T^7 is linear in x.
    x = problem.node_coordinates()[:, 0]
    exact = (300.0**7 + (3000.0**7 - 300.0**7) * x) ** (1 / 7)
    assert problem.values("temperature") == pytest.approx(exact, rel=0.05)


@pytest.mark.parametrize("relaxation", [-0.5, 1.0, 1.5])
def test_a_relaxation_outside_zero_to_one_is_refused(relaxation):
    """The update takes the fraction 1 - gamma of each step, so gamma = 1 would stop the
    iteration and a larger one would reverse it."""
    mesh = dualmesh.generate_rectangle_mesh(0.0, 1.0, 0.0, 1.0, 2, 2)
    problem = dualmesh.Problem(mesh)
    problem.add_variable("u")
    problem.add_kernel("diffusion", variable="u")
    problem.add_boundary_condition("Dirichlet", "left", variable="u", value=1.0)
    with pytest.raises(Exception, match="relaxation is the weight"):
        problem.solve(nonlinear_solver="picard", relaxation=relaxation, report="none")
