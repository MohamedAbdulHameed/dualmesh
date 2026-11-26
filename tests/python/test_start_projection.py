# SPDX-License-Identifier: LGPL-2.1-or-later
"""The start of a transient incompressible flow from its initial velocity projected onto a field
without divergence.

The projection changes the velocity by du = - M^-1 D^T lambda, with M the mass matrix and D the
discrete divergence of the mass balance of the flow, so that D (u_0 + du) = 0 and the prescribed
values stay.  A projected field leaves a second projection unchanged only when its discrete
divergence is zero to round-off.
"""

from __future__ import annotations

import dualmesh
import numpy
import pytest

methods = ["FEM", "DMCDM"]


def flow(mesh, method, **parameters):
    problem = dualmesh.Problem(mesh, method=method)
    physics = problem.add_physics(
        "fluid_dynamics", density=1.0, dynamic_viscosity=0.1, **parameters
    )
    return problem, physics


def velocities(problem):
    return numpy.array([problem.values("velocity_x"), problem.values("velocity_y")])


def vortex(x, y):
    return numpy.array([-numpy.cos(x) * numpy.sin(y), numpy.sin(x) * numpy.cos(y)])


def inflow(x, y):
    return numpy.array([6 * y * (1 - y), 0 * y])


def at_rest(x, y):
    return numpy.zeros((2, len(x)))


def walled_vortex(method, **parameters):
    """The Taylor-Green vortex on [0, 2 pi]^2 with its velocity prescribed on the whole boundary,
    on 8 by 8 quadrilaterals: an enclosed flow.  Returns the problem and the velocity that each
    boundary prescribes."""
    mesh = dualmesh.generate_rectangle_mesh(0.0, 2 * numpy.pi, 0.0, 2 * numpy.pi, 8, 8)
    problem, physics = flow(mesh, method, **parameters)
    walls = ["left", "right", "bottom", "top"]
    physics.add_boundary_condition(
        ["velocity_x = -cos(x)*sin(y)", "velocity_y = sin(x)*cos(y)"], boundary=walls
    )
    problem.initialize()
    x, y = problem.node_coordinates()[:, 0], problem.node_coordinates()[:, 1]
    problem.set_values("velocity_x", vortex(x, y)[0])
    problem.set_values("velocity_y", vortex(x, y)[1])
    return problem, dict.fromkeys(walls, vortex)


def channel_from_rest(method, **parameters):
    """A parabolic inflow into a channel at rest, open at its right end, on 24 by 8
    quadrilaterals.  Returns the problem and the velocity that each boundary prescribes."""
    mesh = dualmesh.generate_rectangle_mesh(0.0, 3.0, 0.0, 1.0, 24, 8)
    problem, physics = flow(mesh, method, **parameters)
    physics.add_boundary_condition(["velocity_x = 6*y*(1 - y)", "velocity_y = 0"], boundary="left")
    physics.add_boundary_condition(["velocity_x = 0", "velocity_y = 0"], boundary=["top", "bottom"])
    problem.initialize()
    return problem, {"left": inflow, "bottom": at_rest, "top": at_rest}


cases = {
    "enclosed vortex": (walled_vortex, {"formulation": "equal_order"}),
    "channel from rest": (channel_from_rest, {"formulation": "equal_order"}),
    "channel, projection integration": (
        channel_from_rest,
        {"formulation": "equal_order", "time_integration": "projection"},
    ),
    "channel, penalty": (channel_from_rest, {"formulation": "penalty"}),
}


@pytest.mark.parametrize("method", methods)
@pytest.mark.parametrize("case", cases)
def test_the_projection_removes_the_divergence_and_keeps_the_prescribed_values(method, case):
    build, parameters = cases[case]
    problem, prescribed = build(method, **parameters)
    before = velocities(problem)
    correction = problem._problem.project_initial_velocity()
    assert correction > 1e-3, case
    after = velocities(problem)
    # The velocity holds its prescribed values, and the others change.
    points = problem.node_coordinates()
    for name, values in prescribed.items():
        nodes = problem.mesh.boundary_nodes(name)
        assert numpy.abs(after[:, nodes] - values(*points[nodes, :2].T)).max() < 1e-15, name
    assert numpy.abs(after - before).max() > 1e-3 * numpy.abs(after).max()
    # The projected field has no divergence, so a second projection leaves it as it is.
    assert problem._problem.project_initial_velocity() < 1e-10
    assert numpy.abs(velocities(problem) - after).max() < 1e-10 * numpy.abs(after).max()


@pytest.mark.parametrize("method", methods)
@pytest.mark.parametrize("enclosed", [True, False])
def test_a_field_without_divergence_is_unchanged(method, enclosed):
    """The stagnation flow u = (x, -y) is linear, so every element holds it and its discrete
    divergence is zero to round-off, on an enclosed square and on one open at the top."""
    mesh = dualmesh.generate_rectangle_mesh(0.0, 1.0, 0.0, 1.0, 6, 6, element_type="Tri3")
    problem, physics = flow(mesh, method, formulation="equal_order")
    walls = ["left", "right", "bottom"] + (["top"] if enclosed else [])
    physics.add_boundary_condition(["velocity_x = x", "velocity_y = -y"], boundary=walls)
    problem.initialize()
    x, y = problem.node_coordinates()[:, 0], problem.node_coordinates()[:, 1]
    problem.set_values("velocity_x", x)
    problem.set_values("velocity_y", -y)
    assert problem._problem.project_initial_velocity() < 1e-12
    assert numpy.abs(velocities(problem) - numpy.array([x, -y])).max() < 1e-12


@pytest.mark.parametrize("formulation", ["equal_order", "Taylor-Hood"])
def test_the_projection_keeps_the_part_without_divergence(formulation):
    """u_0 = (1 - x, 0) in [0, 2] x [0, 1] with the inflow (1, 0) at the left, the walls at the
    top and bottom holding velocity_y = 0, and the right end open.  Its part without divergence
    is u = (1, 0), and the multiplier (x^2 - 4) / 2 of its gradient part (-x, 0) lies in the
    space of quadratic elements, so FEM on Quad9 recovers u to round-off."""
    mesh = dualmesh.generate_rectangle_mesh(0.0, 2.0, 0.0, 1.0, 4, 2, element_type="Quad9")
    problem, physics = flow(mesh, "FEM", formulation=formulation)
    physics.add_boundary_condition(["velocity_x = 1", "velocity_y = 0"], boundary="left")
    physics.add_boundary_condition("velocity_y = 0", boundary=["top", "bottom"])
    problem.initialize()
    x = problem.node_coordinates()[:, 0]
    problem.set_values("velocity_x", 1.0 - x)
    problem.set_values("velocity_y", 0.0 * x)
    problem._problem.project_initial_velocity()
    assert numpy.abs(problem.values("velocity_x") - 1.0).max() < 1e-11
    assert numpy.abs(problem.values("velocity_y")).max() < 1e-11


@pytest.mark.parametrize("method", methods)
def test_a_transient_run_reports_the_projection_of_its_start(method):
    """The first run projects its initial velocity and reports the correction, and a run that
    goes on from where the first stopped starts from that state as it is: two runs of backward
    Euler steps end where one run over both ends."""
    steps = {"time_step": 0.01, "time_stepper": "fixed", "report": "none"}
    whole, _ = channel_from_rest(method, formulation="equal_order")
    whole.solve_transient(end_time=0.04, **steps)
    split, _ = channel_from_rest(method, formulation="equal_order")
    first = split.solve_transient(end_time=0.02, **steps)
    assert first.start_correction > 0.1
    assert "initial velocity correction" in first.summary()
    second = split.solve_transient(start_time=0.02, end_time=0.04, **steps)
    assert second.start_correction is None
    assert numpy.abs(velocities(split) - velocities(whole)).max() < 1e-12
