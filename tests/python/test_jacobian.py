# SPDX-License-Identifier: LGPL-2.1-or-later
"""The Jacobian that the automatic differentiation gives agrees with a
finite-difference Jacobian of the residual, for nonlinear problems of the
physics level with every method, also across a jump of the coefficients
between two blocks."""

from __future__ import annotations

import dualmesh as dm
import numpy as np
import pytest

METHODS = ("fem", "dmcdm", "hfvm", "zfvm")


def finite_difference_jacobian(problem, step=1e-7):
    """Central differences of the residual of :meth:`Problem.linear_system`
    with respect to every unknown."""
    names = list(problem._problem.variable_names())
    count = len(names)
    base = {n: np.array(problem.values(n), dtype=float) for n in names}
    residual, _ = problem.linear_system()
    n = len(residual)
    columns = []
    for j in range(n):
        name, entity = names[j % count], j // count
        values = base[name].copy()
        h = step * max(1.0, abs(values[entity]))
        values[entity] += h
        problem.set_values(name, values)
        plus, _ = problem.linear_system()
        values[entity] -= 2 * h
        problem.set_values(name, values)
        minus, _ = problem.linear_system()
        problem.set_values(name, base[name])
        columns.append((plus - minus) / (2 * h))
    return np.array(columns).T


def compare(exact, approximate):
    """The largest difference outside the rows and columns of the prescribed
    unknowns, which linear_system writes before it assembles, so that a
    finite difference does not see them."""
    prescribed = np.array(
        [np.allclose(approximate[:, j], 0.0) and exact[j, j] == 1.0 for j in range(exact.shape[0])]
    )
    free = ~prescribed
    a, b = exact[np.ix_(free, free)], approximate[np.ix_(free, free)]
    return np.max(np.abs(a - b)) / np.max(np.abs(a))


def two_block_square(n=4):
    mesh = dm.generate_rectangle_mesh(
        x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0, num_x_elements=n, num_y_elements=n
    )
    points = np.array([mesh.element_centroid(e) for e in range(mesh.num_elements)])
    cells = [list(mesh.element_nodes(e)) for e in range(mesh.num_elements)]
    blocks = [0 if p[0] < 0.5 else 1 for p in points]
    out = dm.mesh_from_arrays(
        np.asarray(mesh.points())[:, :2], cells, element_type="Quad4", blocks=blocks
    )
    out.set_block_name(0, "left_block")
    out.set_block_name(1, "right_block")
    out.add_sideset_by_predicate("left", lambda x, y, z: x < 1e-9)
    return out


@pytest.mark.parametrize("method", METHODS)
def test_nonlinear_coefficients_and_a_jump_between_blocks(method):
    """k(u) = 1 + u^2 on one block and 3 (1 + u^2) on the other, and the
    absorption 0.5 u^3, at a state that is not a solution."""
    mesh = two_block_square()
    problem = dm.Problem(mesh, method=method)
    # k is the property of the two parsed objects below, one per block.
    pde = problem.add_physics(
        "general_form_PDE", variables="u", equations="-div(k*grad(u)) + 0.5*u^3 = 1"
    )
    problem.add_property(
        "parsed",
        block=["left_block"],
        property_name="k",
        expression="1 + u^2",
    )
    problem.add_property(
        "parsed",
        block=["right_block"],
        property_name="k",
        expression="3 * (1 + u^2)",
    )
    pde.add_boundary_condition("u = 0.5", boundary="left")
    problem.initialize()
    points = problem.entity_points()
    problem.set_values("u", 0.5 + 0.3 * np.sin(2.0 * points[:, 0]) + 0.2 * points[:, 1] ** 2)
    _, jacobian = problem.linear_system()
    exact = jacobian.toarray()
    approximate = finite_difference_jacobian(problem)
    assert compare(exact, approximate) < 1e-6


@pytest.mark.parametrize("method", ["fem", "dmcdm", "zfvm"])
def test_navier_stokes_in_the_pressure_formulation(method):
    mesh = dm.generate_rectangle_mesh(
        x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0, num_x_elements=3, num_y_elements=3
    )
    problem = dm.Problem(mesh, method=method)
    flow = problem.add_physics("fluid_dynamics", formulation="pressure", density=10.0)
    flow.add_boundary_condition(["velocity_x = 1", "velocity_y = 0"], boundary="top")
    flow.add_boundary_condition(
        ["velocity_x = 0", "velocity_y = 0"], boundary=["left", "right", "bottom"]
    )
    problem.initialize()
    points = problem.entity_points()
    problem.set_values("velocity_x", np.sin(3.0 * points[:, 1]) * points[:, 0])
    problem.set_values("velocity_y", np.cos(2.0 * points[:, 0]) * points[:, 1])
    problem.set_values("pressure", points[:, 0] * points[:, 1])
    _, jacobian = problem.linear_system()
    exact = jacobian.toarray()
    approximate = finite_difference_jacobian(problem)
    assert compare(exact, approximate) < 1e-6
