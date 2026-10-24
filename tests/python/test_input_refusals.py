# SPDX-License-Identifier: LGPL-2.1-or-later
"""Inputs that describe an impossible or ambiguous problem are refused where they are given, with
a message that names them, instead of solving to a plausible wrong answer.  The cases come from a
blind test of the library: each was accepted before."""

from __future__ import annotations

import dualmesh as dm
import numpy as np
import pytest
from dualmesh._core import InputError


def _square():
    return dm.generate_rectangle_mesh(0.0, 1.0, 0.0, 1.0, num_elements=[4, 4])


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
    problem = dm.Problem(_square(), method="fem")
    problem.add_variable("u")
    with pytest.raises(InputError, match=message):
        problem.add_kernel(kernel, variable="u", **parameters)


def test_a_negative_transfer_coefficient_is_refused():
    problem = dm.Problem(_square(), method="fem")
    problem.add_variable("u")
    with pytest.raises(InputError, match="'transfer_coeff' must be non-negative"):
        problem.add_boundary_condition("Robin", "left", variable="u", transfer_coeff=-5.0)


@pytest.mark.parametrize("alias", ["fvm", "finite_volume", "finite_element", "dual_mesh"])
def test_a_method_has_one_name(alias):
    with pytest.raises(InputError, match="Unknown method"):
        dm.Problem(_square(), method=alias)


def test_a_mesh_without_elements_is_refused():
    mesh = dm.Mesh(2)
    for node in [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]:
        mesh.add_node(list(node))
    with pytest.raises(InputError, match="has no elements"):
        dm.Problem(mesh)


def test_a_second_value_on_the_same_boundary_is_refused():
    problem = dm.Problem(_square(), method="fem")
    problem.add_variable("u")
    problem.add_boundary_condition("Dirichlet", "left", variable="u", value=0.0)
    with pytest.raises(InputError, match="u on left is already prescribed"):
        problem.add_boundary_condition("Dirichlet", "left", variable="u", value=100.0)
    # Side sets that only meet at a corner carry values of their own.
    problem.add_boundary_condition("Dirichlet", "bottom", variable="u", value=1.0)


@pytest.mark.parametrize("coordinates", ["axisymmetric", "spherical"])
def test_a_negative_radius_is_refused(coordinates):
    if coordinates == "spherical":
        mesh = dm.generate_line_mesh(start=-1.0, end=1.0, num_elements=4)
    else:
        mesh = dm.generate_rectangle_mesh(-1.0, 1.0, 0.0, 1.0, num_elements=[4, 2])
    with pytest.raises(InputError, match="the radius is x, which must not be negative"):
        dm.Problem(mesh, coordinates=coordinates)


def test_the_cells_of_a_mesh_build_the_same_mesh():
    mesh = dm.mesh_from_arrays(
        [[0, 0], [1, 0], [1, 1], [0, 1], [2, 0], [2, 1]],
        [[0, 1, 2, 3], [1, 4, 5, 2]],
        element_type="Quad4",
        blocks=[1, 2],
    )
    copy = dm.mesh_from_arrays(np.asarray(mesh.points())[:, :2], list(mesh.cells()))
    assert copy.num_elements == 2 and copy.block_ids() == [1, 2]
    assert [copy.element_nodes(e) for e in range(2)] == [mesh.element_nodes(e) for e in range(2)]
    with pytest.raises(ValueError, match="carry the blocks"):
        dm.mesh_from_arrays(np.asarray(mesh.points())[:, :2], list(mesh.cells()), blocks=[0, 0])
