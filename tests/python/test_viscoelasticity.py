# SPDX-License-Identifier: LGPL-2.1-or-later
"""Generalized Maxwell viscoelasticity: relaxation and creep against the closed forms."""

from __future__ import annotations

import dualmesh
import numpy
import pytest

E, nu = 2e9, 0.3
g_long = E / (2 * (1 + nu))
branches = [[1e9, 0.1], [5e8, 2.0]]


def sheared_cube(gamma, **solid):
    """A unit cube whose every face is held at the simple shear u_x = gamma y."""
    mesh = dualmesh.generate_box_mesh(0, 1, 0, 1, 0, 1, 1, 1, 1)
    problem = dualmesh.Problem(mesh)
    body = problem.add_physics("solid_mechanics", youngs_modulus=E, poissons_ratio=nu, **solid)
    sides = list(mesh.sideset_names())
    body.add_boundary_condition(f"disp_x = {gamma}*y", boundary=sides)
    body.add_boundary_condition("disp_y = 0", boundary=sides)
    body.add_boundary_condition("disp_z = 0", boundary=sides)
    return problem, body


def test_a_held_shear_relaxes_as_the_Prony_series():
    """A shear strain reached over the first step and then held: each branch carries
    G_i gamma (tau_i / dt)(1 - exp(-dt / tau_i)) exp(-(t - dt) / tau_i), the exact response of
    the ramp, so the update is exact at any step."""
    gamma, dt = 1e-3, 0.05
    problem, _ = sheared_cube(f"{gamma}*min(t/{dt}, 1)", viscoelasticity={"shear": branches})
    shear, times = [], []
    problem.set_time_step_callback(
        lambda t, core: (
            times.append(t),
            shear.append(problem.property_at_centroids("stress")[0][5]),
        )
    )
    problem.solve_transient(end_time=1.0, time_step=dt, time_stepper="fixed", report="none")
    for t, s in zip(times, shear):
        expected = g_long * gamma
        for G, tau in branches:
            expected += (
                G * gamma * (tau / dt) * (-numpy.expm1(-dt / tau)) * numpy.exp(-(t - dt) / tau)
            )
        assert s == pytest.approx(expected, rel=1e-10)


def test_a_steady_solve_takes_the_long_term_moduli():
    problem, _ = sheared_cube(1e-3, viscoelasticity={"shear": branches, "bulk": [[1e9, 0.5]]})
    problem.solve(report="none")
    assert problem.property_at_centroids("stress")[0][5] == pytest.approx(g_long * 1e-3, rel=1e-12)


def test_a_bulk_branch_relaxes_the_pressure():
    """A uniform dilatation theta held after the first step: the mean stress is K_long theta plus
    the relaxing bulk branch, K_1 theta (tau / dt)(1 - exp(-dt/tau)) exp(-(t - dt)/tau)."""
    theta, dt, k1, tau = 3e-3, 0.1, 1e9, 0.5
    mesh = dualmesh.generate_box_mesh(0, 1, 0, 1, 0, 1, 1, 1, 1)
    problem = dualmesh.Problem(mesh)
    body = problem.add_physics(
        "solid_mechanics",
        youngs_modulus=E,
        poissons_ratio=nu,
        viscoelasticity={"bulk": [[k1, tau]]},
    )
    sides = list(mesh.sideset_names())
    ramp = f"min(t/{dt}, 1)"
    for name, c in (("disp_x", "x"), ("disp_y", "y"), ("disp_z", "z")):
        body.add_boundary_condition(f"{name} = {theta / 3}*{ramp}*{c}", boundary=sides)
    mean, times = [], []
    problem.set_time_step_callback(
        lambda t, core: (
            times.append(t),
            mean.append(numpy.mean(problem.property_at_centroids("stress")[0][:3])),
        )
    )
    problem.solve_transient(end_time=1.0, time_step=dt, time_stepper="fixed", report="none")
    K_long = E / (3 * (1 - 2 * nu))
    for t, m in zip(times, mean):
        expected = K_long * theta + k1 * theta * (tau / dt) * (-numpy.expm1(-dt / tau)) * numpy.exp(
            -(t - dt) / tau
        )
        assert m == pytest.approx(expected, rel=1e-10)
