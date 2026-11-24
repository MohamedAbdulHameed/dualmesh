# SPDX-License-Identifier: LGPL-2.1-or-later
"""Periodic boundary conditions.

The checks have exact answers.
The solutions do not vanish on the periodic boundaries, so a condition that did not join the two boundaries (a natural, zero-flux boundary instead) gives another answer, and the error would not fall with the mesh.
"""

from __future__ import annotations

import math

import dualmesh
import numpy
import pytest

methods = ["fem", "dmcdm"]


def _orders(errors):
    return [math.log2(errors[k] / errors[k + 1]) for k in range(len(errors) - 1)]


def _nodal_error(problem, mesh, exact):
    points = numpy.asarray(mesh.points())
    return numpy.max(numpy.abs(problem.values("u") - exact(points[:, 0], points[:, 1])))


def _periodic_in_x(method, n):
    """-div grad u = 5 pi^2 u0 on the unit square, periodic in x, u = 0 at y = 0 and y = 1, with u = cos(2 pi x + 0.3) sin(pi y)."""
    mesh = dualmesh.generate_rectangle_mesh(0.0, 1.0, 0.0, 1.0, n, n)
    problem = dualmesh.Problem(mesh, method=method, distributed=False)
    problem.add_variable("u")
    problem.add_kernel("diffusion", variable="u")
    problem.add_kernel("body_force", variable="u", value="5*pi^2*cos(2*pi*x + 0.3)*sin(pi*y)")
    problem.add_boundary_condition("Dirichlet", ["bottom", "top"], variable="u", value=0.0)
    problem.add_boundary_condition("periodic", primary="left", secondary="right")
    problem.solve(report="none")
    return _nodal_error(
        problem, mesh, lambda x, y: numpy.cos(2 * numpy.pi * x + 0.3) * numpy.sin(numpy.pi * y)
    )


@pytest.mark.parametrize("method", methods)
def test_periodic_in_one_direction_converges_at_second_order(method):
    errors = [_periodic_in_x(method, n) for n in (8, 16, 32)]
    assert errors[-1] < 5e-3
    assert min(_orders(errors)) > 1.8


def _doubly_periodic(method, n):
    """-div grad u + u = f on the unit square, periodic in x and in y, with u = cos(2 pi (x + y)) + sin(2 pi x)/2, so that the corner nodes are joined in two directions."""
    mesh = dualmesh.generate_rectangle_mesh(0.0, 1.0, 0.0, 1.0, n, n)
    problem = dualmesh.Problem(mesh, method=method, distributed=False)
    problem.add_variable("u")
    problem.add_kernel("diffusion", variable="u")
    problem.add_kernel("reaction", variable="u")
    problem.add_kernel(
        "body_force",
        variable="u",
        value="(8*pi^2 + 1)*cos(2*pi*(x + y)) + (4*pi^2 + 1)*sin(2*pi*x)/2",
    )
    problem.add_boundary_condition("periodic", primary="left", secondary="right")
    problem.add_boundary_condition("periodic", primary="bottom", secondary="top")
    problem.solve(report="none")
    return _nodal_error(
        problem,
        mesh,
        lambda x, y: numpy.cos(2 * numpy.pi * (x + y)) + 0.5 * numpy.sin(2 * numpy.pi * x),
    )


@pytest.mark.parametrize("method", methods)
def test_doubly_periodic_converges_at_second_order(method):
    errors = [_doubly_periodic(method, n) for n in (8, 16, 32)]
    assert errors[-1] < 1e-2
    assert min(_orders(errors)) > 1.8


def test_boundaries_that_do_not_match_are_refused():
    mesh = dualmesh.generate_rectangle_mesh(0.0, 1.0, 0.0, 1.0, 4, 4)
    problem = dualmesh.Problem(mesh, distributed=False)
    problem.add_variable("u")
    problem.add_kernel("diffusion", variable="u")
    problem.add_boundary_condition(
        "periodic",
        primary="left",
        secondary="right",
        translation=[1.0, 0.1],
    )
    with pytest.raises(ValueError, match="no primary node at the translated position"):
        problem.solve(report="none")


def test_a_value_prescribed_on_the_secondary_boundary_only_is_refused():
    mesh = dualmesh.generate_rectangle_mesh(0.0, 1.0, 0.0, 1.0, 4, 4)
    problem = dualmesh.Problem(mesh, distributed=False)
    problem.add_variable("u")
    problem.add_kernel("diffusion", variable="u")
    problem.add_boundary_condition("Dirichlet", "right", variable="u", value=1.0)
    problem.add_boundary_condition("periodic", primary="left", secondary="right")
    with pytest.raises(ValueError, match="prescribed value that its primary node does not"):
        problem.solve(report="none")
