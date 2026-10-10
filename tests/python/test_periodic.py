# SPDX-License-Identifier: LGPL-2.1-or-later
"""Periodic boundary conditions.

The checks have exact answers.
The solutions do not vanish on the periodic boundaries, so a condition that did not join the two boundaries (a natural, zero-flux boundary instead) gives another answer, and the error would not fall with the mesh.
"""

from __future__ import annotations

import math

import dualmesh as dm
import numpy as np
import pytest

METHODS = ["fem", "hfvm", "dmcdm"]


def _orders(errors):
    return [math.log2(errors[k] / errors[k + 1]) for k in range(len(errors) - 1)]


def _nodal_error(problem, mesh, exact):
    points = np.asarray(mesh.points())
    return np.max(np.abs(problem.values("u") - exact(points[:, 0], points[:, 1])))


def _periodic_in_x(method, n):
    """-div grad u = 5 pi^2 u0 on the unit square, periodic in x, u = 0 at y = 0 and y = 1, with u = cos(2 pi x + 0.3) sin(pi y)."""
    mesh = dm.generate_rectangle_mesh(0.0, 1.0, 0.0, 1.0, n, n)
    problem = dm.Problem(mesh, method=method, distributed=False)
    problem.add_variable("u")
    problem.add_kernel("diffusion", "diffusion", variable="u")
    problem.add_kernel(
        "body_force", "source", variable="u", value="5*pi^2*cos(2*pi*x + 0.3)*sin(pi*y)"
    )
    problem.add_boundary_condition(
        "Dirichlet_boundary_condition", "walls", variable="u", boundary=["bottom", "top"], value=0.0
    )
    problem.add_boundary_condition(
        "periodic_boundary_condition", "periodic_x", primary="left", secondary="right"
    )
    problem.solve(report="none")
    return _nodal_error(problem, mesh, lambda x, y: np.cos(2 * np.pi * x + 0.3) * np.sin(np.pi * y))


@pytest.mark.parametrize("method", METHODS)
def test_periodic_in_one_direction_converges_at_second_order(method):
    errors = [_periodic_in_x(method, n) for n in (8, 16, 32)]
    assert errors[-1] < 5e-3
    assert min(_orders(errors)) > 1.8


def _doubly_periodic(method, n):
    """-div grad u + u = f on the unit square, periodic in x and in y, with u = cos(2 pi (x + y)) + sin(2 pi x)/2, so that the corner nodes are joined in two directions."""
    mesh = dm.generate_rectangle_mesh(0.0, 1.0, 0.0, 1.0, n, n)
    problem = dm.Problem(mesh, method=method, distributed=False)
    problem.add_variable("u")
    problem.add_kernel("diffusion", "diffusion", variable="u")
    problem.add_kernel("reaction", "reaction", variable="u")
    problem.add_kernel(
        "body_force",
        "source",
        variable="u",
        value="(8*pi^2 + 1)*cos(2*pi*(x + y)) + (4*pi^2 + 1)*sin(2*pi*x)/2",
    )
    problem.add_boundary_condition(
        "periodic_boundary_condition", "periodic_x", primary="left", secondary="right"
    )
    problem.add_boundary_condition(
        "periodic_boundary_condition", "periodic_y", primary="bottom", secondary="top"
    )
    problem.solve(report="none")
    return _nodal_error(
        problem, mesh, lambda x, y: np.cos(2 * np.pi * (x + y)) + 0.5 * np.sin(2 * np.pi * x)
    )


@pytest.mark.parametrize("method", METHODS)
def test_doubly_periodic_converges_at_second_order(method):
    errors = [_doubly_periodic(method, n) for n in (8, 16, 32)]
    assert errors[-1] < 1e-2
    assert min(_orders(errors)) > 1.8


def test_the_secondary_nodes_equal_their_primaries():
    mesh = dm.generate_rectangle_mesh(0.0, 1.0, 0.0, 1.0, 6, 6)
    problem = dm.Problem(mesh, distributed=False)
    problem.add_variable("u")
    problem.add_kernel("diffusion", "diffusion", variable="u")
    problem.add_kernel("body_force", "source", variable="u", value="x*(1 - y)")
    problem.add_boundary_condition(
        "Dirichlet_boundary_condition", "walls", variable="u", boundary=["bottom", "top"], value=0.0
    )
    problem.add_boundary_condition(
        "periodic_boundary_condition", "periodic_x", primary="left", secondary="right"
    )
    problem.solve(report="none")
    points = np.asarray(mesh.points())
    u = problem.values("u")
    left = np.flatnonzero(np.isclose(points[:, 0], 0.0))
    right = np.flatnonzero(np.isclose(points[:, 0], 1.0))
    left = left[np.argsort(points[left, 1])]
    right = right[np.argsort(points[right, 1])]
    assert np.array_equal(u[left], u[right])
    assert np.abs(u).max() > 1e-3


def test_boundaries_that_do_not_match_are_refused():
    mesh = dm.generate_rectangle_mesh(0.0, 1.0, 0.0, 1.0, 4, 4)
    problem = dm.Problem(mesh, distributed=False)
    problem.add_variable("u")
    problem.add_kernel("diffusion", "diffusion", variable="u")
    problem.add_boundary_condition(
        "periodic_boundary_condition",
        "periodic_x",
        primary="left",
        secondary="right",
        translation=[1.0, 0.1],
    )
    with pytest.raises(ValueError, match="no primary node at the translated position"):
        problem.solve(report="none")


def test_a_value_prescribed_on_the_secondary_boundary_only_is_refused():
    mesh = dm.generate_rectangle_mesh(0.0, 1.0, 0.0, 1.0, 4, 4)
    problem = dm.Problem(mesh, distributed=False)
    problem.add_variable("u")
    problem.add_kernel("diffusion", "diffusion", variable="u")
    problem.add_boundary_condition("Dirichlet_boundary_condition", "right", variable="u", value=1.0)
    problem.add_boundary_condition(
        "periodic_boundary_condition", "periodic_x", primary="left", secondary="right"
    )
    with pytest.raises(ValueError, match="prescribed value that its primary node does not"):
        problem.solve(report="none")


def test_the_cell_centred_method_is_refused():
    mesh = dm.generate_rectangle_mesh(0.0, 1.0, 0.0, 1.0, 4, 4)
    problem = dm.Problem(mesh, method="zfvm", distributed=False)
    problem.add_variable("u")
    problem.add_kernel("diffusion", "diffusion", variable="u")
    problem.add_boundary_condition(
        "periodic_boundary_condition", "periodic_x", primary="left", secondary="right"
    )
    with pytest.raises(ValueError, match="zfvm"):
        problem.solve(report="none")
