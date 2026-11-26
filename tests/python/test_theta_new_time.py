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

import dualmesh
import numpy
import pytest


@pytest.mark.parametrize("time_scheme", ["backward_Euler", "Crank-Nicolson"])
def test_a_uniformly_accelerating_flow_is_exact(time_scheme):
    """v = (1 + t, 0), p = -x: rho dv/dt = -grad p, which linear elements hold exactly, so every
    step of every theta is exact to round-off, with the streamline stabilisation included."""
    mesh = dualmesh.generate_rectangle_mesh(x=[0, 1], y=[0, 1], num_elements=[8, 8])
    problem = dualmesh.Problem(mesh, method="fem")
    flow = problem.add_physics(
        "fluid_dynamics",
        density=1.0,
        dynamic_viscosity=0.01,
        formulation="equal_order",
        initial_condition={"velocity": [1, 0], "pressure": "-x"},
    )
    flow.add_boundary_condition(
        ["velocity_x = 1 + t", "velocity_y = 0"], boundary=["left", "right", "bottom", "top"]
    )
    flow.add_boundary_condition("pressure = 0", boundary="left")
    problem.solve_transient(end_time=0.2, time_step=0.05, time_scheme=time_scheme, report="none")
    x = mesh.points()
    assert numpy.abs(problem.values("velocity_x") - 1.2).max() < 1e-12
    assert numpy.abs(problem.values("velocity_y")).max() < 1e-12
    assert numpy.abs(problem.values("pressure") + x[:, 0]).max() < 1e-12


def rotating_flow_error(time_scheme, n):
    """v = (1 + t)(y, x) with its pressure, an exact solution of the Navier-Stokes equations
    without a body force, in a box whose walls prescribe the velocity."""
    mesh = dualmesh.generate_rectangle_mesh(x=[0, 1], y=[0, 1], num_elements=[n, n])
    problem = dualmesh.Problem(mesh, method="fem")
    flow = problem.add_physics(
        "fluid_dynamics",
        density=1.0,
        dynamic_viscosity=0.01,
        formulation="equal_order",
        initial_condition={"velocity": ["y", "x"]},
    )
    flow.add_boundary_condition(
        ["velocity_x = (1 + t)*y", "velocity_y = (1 + t)*x"],
        boundary=["left", "right", "bottom", "top"],
    )
    problem.solve_transient(end_time=0.2, time_step=0.05, time_scheme=time_scheme, report="none")
    x = mesh.points()
    return max(
        numpy.abs(problem.values("velocity_x") - 1.2 * x[:, 1]).max(),
        numpy.abs(problem.values("velocity_y") - 1.2 * x[:, 0]).max(),
    )


def test_crank_Nicolson_keeps_the_accuracy_of_backward_Euler_on_a_rotating_flow():
    """The velocity is linear in time, so both steps are exact in time, and the error is that of
    the space discretisation, the same for both, and falls by more than 8 over two refinements."""
    crank_Nicolson = [rotating_flow_error("Crank-Nicolson", n) for n in (8, 16, 32)]
    backward_Euler = [rotating_flow_error("backward_Euler", n) for n in (8, 16, 32)]
    for cn, be in zip(crank_Nicolson, backward_Euler):
        assert cn < 1.1 * be, (crank_Nicolson, backward_Euler)
    assert crank_Nicolson[0] / crank_Nicolson[2] > 8.0, crank_Nicolson


def test_an_explicit_step_solves_an_equation_without_a_time_derivative_at_the_new_time():
    """dT/dt = T'' and u'' + T = 0 on (0, 1), both zero at the ends, T = sin(pi x) at the start.
    The equation of u has no time derivative.  After explicit steps (forward_Euler), u solves its
    equation for the T of the last step: K u = M T with the linear finite element matrices,
    assembled here independently."""
    n = 20
    mesh = dualmesh.generate_line_mesh(start=0.0, end=1.0, num_elements=n)
    problem = dualmesh.Problem(mesh, method="fem")
    for name in ("T", "u"):
        problem.add_variable(name)
        problem.add_kernel("diffusion", variable=name, diffusivity=1.0)
        problem.add_boundary_condition("Dirichlet", ["left", "right"], variable=name, value=0.0)
    problem.add_kernel("time_derivative", variable="T")
    problem.add_kernel("parsed_kernel", variable="u", source="T")
    problem.initialize()
    x = problem.node_coordinates()[:, 0]
    order = numpy.argsort(x)
    problem.set_values("T", numpy.sin(numpy.pi * x))
    problem.solve_transient(
        end_time=4e-3,
        time_step=1e-3,
        time_scheme="forward_Euler",
        time_stepper="fixed",
        report="none",
    )
    T = problem.values("T")[order]
    h = 1.0 / n
    K = (
        numpy.diag(numpy.full(n + 1, 2.0))
        - numpy.diag(numpy.ones(n), 1)
        - numpy.diag(numpy.ones(n), -1)
    ) / h
    M = (
        (
            numpy.diag(numpy.full(n + 1, 4.0))
            + numpy.diag(numpy.ones(n), 1)
            + numpy.diag(numpy.ones(n), -1)
        )
        * h
        / 6
    )
    inner = slice(1, n)
    u = numpy.zeros(n + 1)
    u[inner] = numpy.linalg.solve(K[inner, inner], (M @ T)[inner])
    assert numpy.abs(u).max() > 0.05
    numpy.testing.assert_allclose(problem.values("u")[order], u, atol=1e-12)
