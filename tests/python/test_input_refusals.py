# SPDX-License-Identifier: LGPL-2.1-or-later
"""Inputs that describe an impossible or ambiguous problem are refused where they are given, with
a message that names them, instead of solving to a plausible wrong answer.  The cases come from a
blind test of the library: each was accepted before."""

from __future__ import annotations

import dualmesh
import pytest
from dualmesh._core import InputError


def _square():
    return dualmesh.generate_rectangle_mesh(0.0, 1.0, 0.0, 1.0, num_elements=[4, 4])


@pytest.mark.parametrize(
    "kernel, parameters, message",
    [
        ("diffusion", {"diffusivity": -1.0}, "'diffusivity' must be positive, and is -1"),
        (
            "heat_conduction",
            {"thermal_conductivity": 0.0},
            "'thermal_conductivity' must be positive",
        ),
    ],
)
def test_a_coefficient_out_of_its_range_is_refused(kernel, parameters, message):
    problem = dualmesh.Problem(_square(), method="fem")
    problem.add_variable("u")
    with pytest.raises(InputError, match=message):
        problem.add_kernel(kernel, variable="u", **parameters)


@pytest.mark.parametrize("alias", ["fvm", "finite_volume", "finite_element", "dual_mesh"])
def test_a_method_has_one_name(alias):
    with pytest.raises(InputError, match="Unknown method"):
        dualmesh.Problem(_square(), method=alias)


def test_a_mesh_without_elements_is_refused():
    mesh = dualmesh.Mesh(2)
    for node in [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]:
        mesh.add_node(list(node))
    with pytest.raises(InputError, match="has no elements"):
        dualmesh.Problem(mesh)


def test_a_second_value_on_the_same_boundary_is_refused():
    problem = dualmesh.Problem(_square(), method="fem")
    problem.add_variable("u")
    problem.add_boundary_condition("Dirichlet", "left", variable="u", value=0.0)
    with pytest.raises(InputError, match="u on left is already prescribed"):
        problem.add_boundary_condition("Dirichlet", "left", variable="u", value=100.0)
    # Side sets that only meet at a corner carry values of their own.
    problem.add_boundary_condition("Dirichlet", "bottom", variable="u", value=1.0)


@pytest.mark.parametrize("coordinates", ["axisymmetric", "spherical"])
def test_a_negative_radius_is_refused(coordinates):
    if coordinates == "spherical":
        mesh = dualmesh.generate_line_mesh(start=-1.0, end=1.0, num_elements=4)
    else:
        mesh = dualmesh.generate_rectangle_mesh(-1.0, 1.0, 0.0, 1.0, num_elements=[4, 2])
    with pytest.raises(InputError, match="the radius is x, which must not be negative"):
        dualmesh.Problem(mesh, coordinates=coordinates)
