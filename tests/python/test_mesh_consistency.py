# SPDX-License-Identifier: LGPL-2.1-or-later
"""Consistency of a mesh through promotion, refinement and reading: node sets
keep to their geometry, side sets cover the right faces, and malformed
options are refused where they are given."""

import dualmesh as dm
import numpy as np
import pytest
from dualmesh._core import InputError


def wall_square():
    # Two triangles whose shared diagonal joins (0, 0) and (1, 1); the node
    # set "wall" holds three boundary corners, so the diagonal is a chord
    # between two of its members.
    mesh = dm.mesh_from_arrays(
        [[0, 0], [1, 0], [1, 1], [0, 1]], [[0, 1, 2], [0, 2, 3]], element_type="Tri3"
    )
    mesh.add_nodeset("wall", [0, 2, 3])
    return mesh


def wall_points(mesh):
    P = np.asarray(mesh.points())[:, :2]
    return sorted(tuple(np.round(p, 6)) for p in P[mesh.boundary_nodes("wall")])


@pytest.mark.parametrize(
    "grow",
    [
        lambda m: m.second_order(),
        lambda m: m.refined(),
        lambda m: dm.refine_marked(m, [1, 1])[0],
    ],
)
def test_a_boundary_node_set_does_not_take_the_midpoint_of_a_chord(grow):
    # The wall is the two edges x = 0 and y = 1: after any refinement or
    # promotion the set holds exactly the nodes on them, and not the
    # midpoint of the diagonal.
    grown = grow(wall_square())
    P = np.asarray(grown.points())[:, :2]
    on_wall = sorted(tuple(np.round(p, 6)) for p in P if p[0] == 0.0 or p[1] == 1.0)
    assert wall_points(grown) == on_wall


def test_an_interior_node_set_grows_along_its_edges():
    # An interface: the line x = 0.5 through a 2 x 2 mesh, given as nodes; its
    # middle node is interior, so the set grows along the edges between its
    # members.
    mesh = dm.generate_rectangle_mesh(x_min=0, x_max=1, y_min=0, y_max=1, num_elements=[2, 2])
    P = np.asarray(mesh.points())
    mesh.add_nodeset("interface", [int(i) for i in np.flatnonzero(np.isclose(P[:, 0], 0.5))])
    quadratic = mesh.second_order()
    Q = np.asarray(quadratic.points())[:, :2]
    members = sorted(tuple(np.round(q, 6)) for q in Q[quadratic.boundary_nodes("interface")])
    assert members == [(0.5, y) for y in (0.0, 0.25, 0.5, 0.75, 1.0)]


def test_bounding_box_side_sets_of_a_surface_follow_its_space():
    # A plate in the xz plane, as a surface in space: no y faces, and x and z
    # faces with one row of edges each.
    plate = dm.mesh_from_arrays(
        [[x, 0.0, z] for z in (0.0, 0.5, 1.0) for x in (0.0, 1.0, 2.0)],
        [[0, 1, 4, 3], [1, 2, 5, 4], [3, 4, 7, 6], [4, 5, 8, 7]],
        element_type="Quad4",
        dimension=3,
    )
    plate.add_bounding_box_sidesets()
    counts = {name: len(plate.sideset(name)) for name in plate.sideset_names()}
    assert counts == {"left": 2, "right": 2, "back": 2, "front": 2}


@pytest.mark.parametrize("z", [0.0, 5.0, 1e-17])
def test_a_flat_mesh_stays_in_its_plane(z):
    mesh = dm.mesh_from_arrays(
        [[0, 0, z], [1, 0, z], [1, 1, z], [0, 1, z]], [[0, 1, 2, 3]], element_type="Quad4"
    )
    assert mesh.space_dimension == 2


def test_gmsh_names_are_found_by_tag_and_dimension(tmp_path):
    # Gmsh numbers its physical groups per dimension: the surface group 1 is
    # the plate and the curve group 1 is its rim, and each keeps its own name.
    meshio = pytest.importorskip("meshio")
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    cells = [("quad", np.array([[0, 1, 2, 3]])), ("line", np.array([[0, 1], [1, 2]]))]
    tags = [np.array([1]), np.array([1, 1])]
    file = meshio.Mesh(
        points,
        cells,
        cell_data={"gmsh:physical": tags, "gmsh:geometrical": tags},
        field_data={"plate": np.array([1, 2]), "rim": np.array([1, 1])},
    )
    path = tmp_path / "plate.msh"
    meshio.write(path, file, file_format="gmsh22", binary=False)
    mesh = dm.read_mesh(str(path))
    assert mesh.sideset_names() == ["rim"]
    assert len(mesh.sideset("rim")) == 2
    assert mesh.block_names() == {1: "plate"}


def test_a_side_set_with_a_face_off_the_boundary_is_not_added():
    mesh = dm.generate_rectangle_mesh(x_min=0, x_max=1, y_min=0, y_max=1, num_elements=[2, 2])
    with pytest.raises(InputError, match="1 of 2 faces"):
        mesh.add_sideset_from_faces("wall", [[0, 1], [0, 8]])
    assert "wall" not in mesh.sideset_names()


def test_an_expression_of_no_variables_is_a_number():
    assert dm.Expression("2*3", [])() == 6.0


def test_a_malformed_gauss_rule_is_refused():
    mesh = dm.generate_rectangle_mesh(x_min=0, x_max=1, y_min=0, y_max=1, num_elements=[2, 2])
    problem = dm.Problem(mesh)
    problem.add_variable("u")
    with pytest.raises(InputError, match="gauss1 to gauss20"):
        problem.add_kernel("diffusion", variable="u", quadrature="gauss_3")
        problem.solve()


def test_a_problem_refuses_an_inverted_element():
    """A mesh whose winding is reversed is not integrated with a positive
    measure: initialize() names the element and asks for fix_orientation().
    The generators and read_mesh fix the orientation themselves, so this is
    reached only by a mesh built node by node."""
    mesh = dm.Mesh(2)
    for node in [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)]:
        mesh.add_node(list(node))
    mesh.add_element("Quad4", [0, 3, 2, 1])  # clockwise: det J < 0
    problem = dm.Problem(mesh, method="fem")
    problem.add_variable("temperature")
    problem.add_kernel("heat_conduction", variable="temperature", thermal_conductivity=1.0)
    with pytest.raises(ValueError, match="Element 0 of block 0 .* is inverted"):
        problem.initialize()


def test_a_problem_names_a_degenerate_element():
    # The third node of the second triangle lies on the line through the first two.
    mesh = dm.Mesh(2)
    for node in [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (2.0, 0.0, 0.0)]:
        mesh.add_node(list(node))
    mesh.add_element("Tri3", [0, 1, 2])
    mesh.add_element("Tri3", [0, 1, 3])
    problem = dm.Problem(mesh, method="fem")
    problem.add_variable("temperature")
    problem.add_kernel("heat_conduction", variable="temperature", thermal_conductivity=1.0)
    with pytest.raises(ValueError, match="Element 1 of block 0 is degenerate"):
        problem.initialize()


def test_the_bounds_of_an_axis_are_a_pair_or_a_minimum_and_a_maximum():
    pair = dm.generate_box_mesh(x=[0, 1], y=[0, 2], z=[-1, 1], num_elements=[2, 2, 2])
    named = dm.generate_box_mesh(
        x_min=0, x_max=1, y_min=0, y_max=2, z_min=-1, z_max=1, num_elements=[2, 2, 2]
    )
    assert np.array_equal(np.asarray(pair.points()), np.asarray(named.points()))
    with pytest.raises(ValueError, match="not both"):
        dm.generate_rectangle_mesh(x=[0, 1], x_min=0, y=[0, 1], num_elements=[2, 2])
    with pytest.raises(ValueError, match="with x_max > x_min"):
        dm.generate_rectangle_mesh(x=[1, 0], y=[0, 1], num_elements=[2, 2])
    with pytest.raises(ValueError, match="the coordinates hold the bounds"):
        dm.generate_rectangle_mesh(
            x_coordinates=[0, 1], x_min=0, x_max=1, y=[0, 1], num_y_elements=2
        )
