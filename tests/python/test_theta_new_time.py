# SPDX-License-Identifier: LGPL-2.1-or-later
"""The theta method with equations that have no time derivative.

A step weights the steady terms by theta at the new time and by 1 - theta at the old one.  An
equation without a time derivative (the mass equation of an incompressible flow, the equilibrium
of a quasi-static solid) and the pressure gradient are taken at the new time only.  Weighted
between the two times, the pressure became an average of two steps, whose error came back with the
opposite sign at every step and grew (Crank-Nicolson: 1.9e-2 on 8 x 8 and 5.2e-2 on 16 x 16 for
the flow below, which backward Euler gives to round-off), and creep was evaluated at the old time
without the material history (a strain 185 % too large).
"""

from __future__ import annotations

import dualmesh as dm
import numpy as np
import pytest


@pytest.mark.parametrize("implicitness", [1.0, 0.5])
def test_a_uniformly_accelerating_flow_is_exact(implicitness):
    """v = (1 + t, 0), p = -x: rho dv/dt = -grad p, which linear elements hold exactly, so every
    step of every theta is exact to round-off, with the streamline stabilisation included."""
    mesh = dm.generate_rectangle_mesh(x=[0, 1], y=[0, 1], num_elements=[8, 8])
    problem = dm.Problem(mesh, method="fem")
    flow = problem.add_physics(
        "fluid_dynamics",
        density=1.0,
        dynamic_viscosity=0.01,
        formulation="pressure",
        initial_condition={"velocity": [1, 0], "pressure": "-x"},
    )
    flow.add_boundary_condition(
        ["velocity_x = 1 + t", "velocity_y = 0"], boundary=["left", "right", "bottom", "top"]
    )
    flow.add_boundary_condition("pressure = 0", boundary="left")
    problem.solve_transient(end_time=0.2, time_step=0.05, implicitness=implicitness, report="none")
    x = mesh.points()
    assert np.abs(problem.values("velocity_x") - 1.2).max() < 1e-12
    assert np.abs(problem.values("velocity_y")).max() < 1e-12
    assert np.abs(problem.values("pressure") + x[:, 0]).max() < 1e-12


def rotating_flow_error(implicitness, n):
    """v = (1 + t)(y, x) with its pressure, an exact solution of the Navier-Stokes equations
    without a body force, in a box whose walls prescribe the velocity."""
    mesh = dm.generate_rectangle_mesh(x=[0, 1], y=[0, 1], num_elements=[n, n])
    problem = dm.Problem(mesh, method="fem")
    flow = problem.add_physics(
        "fluid_dynamics",
        density=1.0,
        dynamic_viscosity=0.01,
        formulation="pressure",
        initial_condition={"velocity": ["y", "x"]},
    )
    flow.add_boundary_condition(
        ["velocity_x = (1 + t)*y", "velocity_y = (1 + t)*x"],
        boundary=["left", "right", "bottom", "top"],
    )
    problem.solve_transient(end_time=0.2, time_step=0.05, implicitness=implicitness, report="none")
    x = mesh.points()
    return max(
        np.abs(problem.values("velocity_x") - 1.2 * x[:, 1]).max(),
        np.abs(problem.values("velocity_y") - 1.2 * x[:, 0]).max(),
    )


def test_crank_nicolson_keeps_the_accuracy_of_backward_euler_on_a_rotating_flow():
    """The velocity is linear in time, so both steps are exact in time, and the error is that of
    the space discretisation, the same for both, and falls by more than 8 over two refinements.  Before, Crank-Nicolson's error
    grew from 1.4e-3 to 9.8e-3 as the mesh was refined."""
    crank_nicolson = [rotating_flow_error(0.5, n) for n in (8, 16, 32)]
    backward_euler = [rotating_flow_error(1.0, n) for n in (8, 16, 32)]
    for cn, be in zip(crank_nicolson, backward_euler):
        assert cn < 1.1 * be, (crank_nicolson, backward_euler)
    assert crank_nicolson[0] / crank_nicolson[2] > 8.0, crank_nicolson


def test_an_explicit_step_solves_an_equation_without_a_time_derivative_at_the_new_time():
    """dT/dt = T'' and u'' + T = 0 on (0, 1), both zero at the ends, T = sin(pi x) at the start.
    The equation of u has no time derivative.  After explicit steps (implicitness 0), u solves its
    equation for the T of the last step: K u = M T with the linear finite element matrices,
    assembled here independently."""
    n = 20
    mesh = dm.generate_line_mesh(start=0.0, end=1.0, num_elements=n)
    problem = dm.Problem(mesh, method="fem")
    for name in ("T", "u"):
        problem.add_variable(name)
        problem.add_kernel("diffusion", variable=name, diffusivity=1.0)
        problem.add_boundary_condition("Dirichlet", ["left", "right"], variable=name, value=0.0)
    problem.add_kernel("time_derivative", variable="T")
    problem.add_kernel("parsed_kernel", variable="u", source="T")
    problem.initialize()
    x = problem.entity_points()[:, 0]
    order = np.argsort(x)
    problem.set_values("T", np.sin(np.pi * x))
    problem.solve_transient(end_time=4e-3, time_step=1e-3, implicitness=0.0, report="none")
    T = problem.values("T")[order]
    h = 1.0 / n
    K = (np.diag(np.full(n + 1, 2.0)) - np.diag(np.ones(n), 1) - np.diag(np.ones(n), -1)) / h
    M = (np.diag(np.full(n + 1, 4.0)) + np.diag(np.ones(n), 1) + np.diag(np.ones(n), -1)) * h / 6
    inner = slice(1, n)
    u = np.zeros(n + 1)
    u[inner] = np.linalg.solve(K[inner, inner], (M @ T)[inner])
    assert np.abs(u).max() > 0.05
    np.testing.assert_allclose(problem.values("u")[order], u, atol=1e-12)
