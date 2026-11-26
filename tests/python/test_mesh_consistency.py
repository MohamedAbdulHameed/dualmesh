# SPDX-License-Identifier: LGPL-2.1-or-later
"""Consistency of a mesh through promotion, refinement and reading: node sets
keep to their geometry, side sets cover the right faces, and malformed
options are refused where they are given."""

import dualmesh
import numpy
import pytest
from dualmesh._core import InputError


def wall_square():
    # Two triangles whose shared diagonal joins (0, 0) and (1, 1); the node
    # set "wall" holds three boundary corners, so the diagonal is a chord
    # between two of its members.
    mesh = dualmesh.mesh_from_arrays(
        [[0, 0], [1, 0], [1, 1], [0, 1]], [[0, 1, 2], [0, 2, 3]], element_type="Tri3"
    )
    mesh.add_nodeset("wall", [0, 2, 3])
    return mesh


def wall_points(mesh):
    P = numpy.asarray(mesh.points())[:, :2]
    return sorted(tuple(numpy.round(p, 6)) for p in P[mesh.boundary_nodes("wall")])


@pytest.mark.parametrize(
    "grow",
    [
        lambda m: m.second_order(),
        lambda m: m.refined(),
        lambda m: dualmesh.refine_marked(m, [1, 1])[0],
    ],
)
def test_a_boundary_node_set_does_not_take_the_midpoint_of_a_chord(grow):
    # The wall is the two edges x = 0 and y = 1: after any refinement or
    # promotion the set holds exactly the nodes on them, and not the
    # midpoint of the diagonal.
    grown = grow(wall_square())
    P = numpy.asarray(grown.points())[:, :2]
    on_wall = sorted(tuple(numpy.round(p, 6)) for p in P if p[0] == 0.0 or p[1] == 1.0)
    assert wall_points(grown) == on_wall


def test_a_side_set_with_a_face_off_the_boundary_is_not_added():
    mesh = dualmesh.generate_rectangle_mesh(x_min=0, x_max=1, y_min=0, y_max=1, num_elements=[2, 2])
    with pytest.raises(InputError, match="1 of 2 faces"):
        mesh.add_sideset_from_faces("wall", [[0, 1], [0, 8]])
    assert "wall" not in mesh.sideset_names()


def test_a_problem_refuses_an_inverted_element():
    """A mesh whose winding is reversed is not integrated with a positive
    measure: initialize() names the element and asks for fix_orientation().
    The generators and read_mesh fix the orientation themselves, so this is
    reached only by a mesh built node by node."""
    mesh = dualmesh.Mesh(2)
    for node in [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)]:
        mesh.add_node(list(node))
    mesh.add_element("Quad4", [0, 3, 2, 1])  # Clockwise: det J < 0
    problem = dualmesh.Problem(mesh, method="fem")
    problem.add_variable("temperature")
    problem.add_kernel("diffusion", variable="temperature", diffusivity=1.0)
    with pytest.raises(ValueError, match="Element 0 of block 0 .* is inverted"):
        problem.initialize()


def test_a_problem_names_a_degenerate_element():
    # The third node of the second triangle lies on the line through the first two.
    mesh = dualmesh.Mesh(2)
    for node in [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (2.0, 0.0, 0.0)]:
        mesh.add_node(list(node))
    mesh.add_element("Tri3", [0, 1, 2])
    mesh.add_element("Tri3", [0, 1, 3])
    problem = dualmesh.Problem(mesh, method="fem")
    problem.add_variable("temperature")
    problem.add_kernel("diffusion", variable="temperature", diffusivity=1.0)
    with pytest.raises(ValueError, match="Element 1 of block 0 is degenerate"):
        problem.initialize()


def test_the_bounds_of_an_axis_are_a_pair_or_a_minimum_and_a_maximum():
    pair = dualmesh.generate_box_mesh(x=[0, 1], y=[0, 2], z=[-1, 1], num_elements=[2, 2, 2])
    named = dualmesh.generate_box_mesh(
        x_min=0, x_max=1, y_min=0, y_max=2, z_min=-1, z_max=1, num_elements=[2, 2, 2]
    )
    assert numpy.array_equal(numpy.asarray(pair.points()), numpy.asarray(named.points()))
    with pytest.raises(ValueError, match="not both"):
        dualmesh.generate_rectangle_mesh(x=[0, 1], x_min=0, y=[0, 1], num_elements=[2, 2])
    with pytest.raises(ValueError, match="with x_max > x_min"):
        dualmesh.generate_rectangle_mesh(x=[1, 0], y=[0, 1], num_elements=[2, 2])
    with pytest.raises(ValueError, match="the coordinates hold the bounds"):
        dualmesh.generate_rectangle_mesh(
            x_coordinates=[0, 1], x_min=0, x_max=1, y=[0, 1], num_y_elements=2
        )
