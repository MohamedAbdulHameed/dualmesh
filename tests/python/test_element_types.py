# SPDX-License-Identifier: LGPL-2.1-or-later
"""Tests of the element types that not every method can use.

Every element supports the finite element method and the cell-centred finite
volume method, because both need only the element itself.  The dual mesh
control domain method and the vertex-centred finite volume method integrate
over the control domains of a dual mesh, so they need an element that carries
one.  ``Wedge6`` does.  ``Quad8`` and ``Hex20`` do not, because a serendipity
element has no interior node and the median dual is undefined for it, and
``Pyramid5`` does not, because its apex is shared by four edges and the
decomposition its control domain would need is not implemented.

The tests below check that each element is offered to exactly the methods that
can use it, that each passes the patch test with every method that accepts it,
and that the elements work together in hybrid meshes.
"""

from __future__ import annotations

import dualmesh as dm
import numpy as np
import pytest

METHODS = ("dmcdm", "fem", "hfvm", "zfvm")
SUPPORTED = {
    "Quad8": {"fem", "zfvm"},
    "Hex20": {"fem", "zfvm"},
    "Wedge6": set(METHODS),
    "Pyramid5": {"fem", "zfvm"},
}


def distorted_square(element_type, num_elements=3):
    mesh = dm.generate_rectangle_mesh(
        x_min=0.0,
        x_max=1.0,
        y_min=0.0,
        y_max=1.0,
        num_x_elements=num_elements,
        num_y_elements=num_elements,
        element_type=element_type,
    )
    mesh.transform_nodes(
        lambda x, y, z: [
            x + 0.3 * x * (1 - x) * y * (1 - y),
            y - 0.2 * x * (1 - x) * y * (1 - y),
            0.0,
        ]
    )
    return mesh


def distorted_cube(element_type, num_elements=2):
    mesh = dm.generate_box_mesh(
        x_min=0.0,
        x_max=1.0,
        y_min=0.0,
        y_max=1.0,
        z_min=0.0,
        z_max=1.0,
        num_x_elements=num_elements,
        num_y_elements=num_elements,
        num_z_elements=num_elements,
        element_type=element_type,
    )
    mesh.transform_nodes(
        lambda x, y, z: [
            x + 0.2 * x * (1 - x) * y,
            y - 0.15 * y * (1 - y) * z,
            z + 0.1 * z * (1 - z) * x,
        ]
    )
    return mesh


def mesh_for(element_type):
    return (
        distorted_square(element_type) if element_type == "Quad8" else distorted_cube(element_type)
    )


def patch_test_error(mesh, method):
    def exact(x, y, z, t):
        return 1.0 + 2.0 * x - 3.0 * y + 0.5 * z

    problem = dm.Problem(mesh, method=method)
    problem.add_variable("u")
    problem.add_kernel("Diffusion", "diffusion", variable="u")
    problem.add_boundary_condition(
        "DirichletBC", "all", variable="u", boundary=mesh.sideset_names(), value=exact
    )
    problem.solve()
    points = problem.entity_points()
    expected = 1.0 + 2.0 * points[:, 0] - 3.0 * points[:, 1] + 0.5 * points[:, 2]
    return float(np.abs(problem.values("u") - expected).max())


def hybrid_mesh(second_type):
    """A box whose cells with x < 1/2 are hexahedra and whose other cells are
    converted to ``second_type`` (``Wedge6`` or ``Pyramid5``).

    Both conversions keep the faces between the two halves whole, so the mesh
    is conforming: a prism split on its base diagonal presents a full
    quadrilateral to its neighbour in x, and a cell divided into six pyramids
    presents each of its faces as the base of one of them.
    """
    base = dm.generate_box_mesh(
        x_min=0.0,
        x_max=1.0,
        y_min=0.0,
        y_max=1.0,
        z_min=0.0,
        z_max=1.0,
        num_x_elements=4,
        num_y_elements=2,
        num_z_elements=2,
    )
    points = [list(p) for p in np.asarray(base.points())]
    cells = []
    for _type, connectivity, _block in base.cells():
        h = list(connectivity)
        centroid = np.mean([points[i] for i in h], axis=0)
        if centroid[0] < 0.5:
            cells.append(("Hex8", [h]))
        elif second_type == "Wedge6":
            cells.append(
                (
                    "Wedge6",
                    [[h[0], h[1], h[2], h[4], h[5], h[6]], [h[0], h[2], h[3], h[4], h[6], h[7]]],
                )
            )
        else:
            points.append(list(centroid))
            c = len(points) - 1
            faces = [
                (0, 1, 5, 4),
                (1, 2, 6, 5),
                (2, 3, 7, 6),
                (0, 4, 7, 3),
                (0, 3, 2, 1),
                (4, 5, 6, 7),
            ]
            cells.append(("Pyramid5", [[h[f[0]], h[f[1]], h[f[2]], h[f[3]], c] for f in faces]))
    groups = {}
    for element_type, connectivity in cells:
        groups.setdefault(element_type, []).extend(connectivity)
    mesh = dm.mesh_from_arrays(np.asarray(points), list(groups.items()))
    mesh.add_bounding_box_sidesets()
    mesh.transform_nodes(lambda x, y, z: [x + 0.1 * x * (1 - x) * y, y + 0.05 * y * (1 - y) * z, z])
    return mesh


# ---------------------------------------------------------------------------
# Which methods accept which elements
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("element_type", sorted(SUPPORTED))
@pytest.mark.parametrize("method", METHODS)
def test_each_element_is_offered_to_exactly_the_methods_that_can_use_it(element_type, method):
    mesh = mesh_for(element_type)
    if method in SUPPORTED[element_type]:
        dm.Problem(mesh, method=method)
    else:
        with pytest.raises(ValueError, match=f"{element_type}.*method='fem'.*method='zfvm'"):
            dm.Problem(mesh, method=method)


# ---------------------------------------------------------------------------
# The patch test
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "element_type, method",
    [(t, m) for t in sorted(SUPPORTED) for m in METHODS if m in SUPPORTED[t]],
)
def test_linear_field_is_exact_on_a_distorted_mesh(element_type, method):
    tolerance = 1e-12
    assert patch_test_error(mesh_for(element_type), method) < tolerance


@pytest.mark.parametrize("method", METHODS)
def test_a_hexahedron_and_prism_hybrid_mesh_passes_the_patch_test(method):
    """A hybrid mesh passes the patch test only if the elements agree on their
    shared faces.  For the two dual mesh methods it also needs their control
    domains to agree there: the quarter of a hexahedron's face that belongs to
    a node must be the quarter of the prism's face that belongs to it."""
    tolerance = 1e-12
    assert patch_test_error(hybrid_mesh("Wedge6"), method) < tolerance


@pytest.mark.parametrize("method", ["fem", "zfvm"])
def test_a_hexahedron_and_pyramid_hybrid_mesh_passes_the_patch_test(method):
    tolerance = 1e-12
    assert patch_test_error(hybrid_mesh("Pyramid5"), method) < tolerance


# ---------------------------------------------------------------------------
# Mesh operations
# ---------------------------------------------------------------------------
def total_volume(mesh):
    problem = dm.Problem(mesh, method="fem")
    problem.add_variable("u")
    problem.set_values("u", np.ones(len(problem.entity_points())))
    return problem.integrate("u")


@pytest.mark.parametrize("element_type", ["Wedge6", "Pyramid5"])
def test_uniform_refinement_of_prisms_and_pyramids(element_type):
    """A prism refines into eight prisms.  A pyramid refines into six pyramids
    and four tetrahedra, which is why a refined pyramid mesh is a mixed mesh.
    Either way the volume is unchanged, the refined mesh is conforming (it
    passes the patch test), and the boundary sets come across."""
    mesh = distorted_cube(element_type, num_elements=1)
    refined = mesh.refined()
    per_parent = 8 if element_type == "Wedge6" else 10
    assert refined.num_elements == per_parent * mesh.num_elements
    assert total_volume(refined) == pytest.approx(total_volume(mesh), rel=1e-12)
    assert sorted(refined.sideset_names()) == sorted(mesh.sideset_names())
    assert patch_test_error(refined, "fem") < 1e-12
    if element_type == "Pyramid5":
        kinds = {t for t, _c, _b in refined.cells()}
        assert kinds == {"Pyramid5", "Tet4"}


def test_serendipity_promotion():
    """``second_order(serendipity=True)`` adds the mid-edge nodes and no
    others: a 2 by 2 Quad4 mesh has 9 corners and 12 edges, so 21 nodes, where
    the full promotion to Quad9 adds the 4 centres as well and gives 25."""
    mesh = dm.generate_rectangle_mesh(
        x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0, num_x_elements=2, num_y_elements=2
    )
    assert mesh.second_order(serendipity=True).num_nodes == 21
    assert mesh.second_order().num_nodes == 25
    box = dm.generate_box_mesh(
        x_min=0.0,
        x_max=1.0,
        y_min=0.0,
        y_max=1.0,
        z_min=0.0,
        z_max=1.0,
        num_x_elements=1,
        num_y_elements=1,
        num_z_elements=1,
    )
    assert box.second_order(serendipity=True).num_nodes == 20
    assert box.second_order().num_nodes == 27


def test_prisms_and_pyramids_cannot_be_promoted_to_second_order():
    """No quadratic prism or pyramid is implemented, and leaving them linear
    next to quadratic neighbours would make the mesh non-conforming, so the
    promotion refuses rather than doing it silently."""
    for element_type in ("Wedge6", "Pyramid5"):
        with pytest.raises(ValueError, match="cannot be promoted"):
            distorted_cube(element_type, num_elements=1).second_order()


@pytest.mark.parametrize("element_type", ["Quad8", "Hex20", "Wedge6", "Pyramid5"])
def test_new_elements_survive_a_file_round_trip(tmp_path, element_type):
    """dualmesh numbers every element as VTK does, which is meshio's order, so
    a mesh written and read back is the same mesh node for node."""
    pytest.importorskip("meshio")
    mesh = mesh_for(element_type)
    path = tmp_path / "mesh.vtu"
    dm.write_mesh(mesh, str(path))
    back = dm.read_mesh(str(path), add_bounding_box_sidesets=True)
    assert back.num_elements == mesh.num_elements
    assert np.asarray(back.points()) == pytest.approx(np.asarray(mesh.points()))
    for (type_a, nodes_a, _), (type_b, nodes_b, _) in zip(mesh.cells(), back.cells()):
        assert type_a == type_b
        assert list(nodes_a) == list(nodes_b)


def test_the_generators_build_the_expected_element_counts():
    box = dict(
        x_min=0.0,
        x_max=1.0,
        y_min=0.0,
        y_max=1.0,
        z_min=0.0,
        z_max=1.0,
        num_x_elements=2,
        num_y_elements=2,
        num_z_elements=2,
    )
    assert dm.generate_box_mesh(element_type="Wedge6", **box).num_elements == 16
    pyramids = dm.generate_box_mesh(element_type="Pyramid5", **box)
    assert pyramids.num_elements == 48
    assert pyramids.num_nodes == 27 + 8  # the grid plus one centre per cell
    # 27 grid nodes and one on each of the 54 grid edges.
    assert dm.generate_box_mesh(element_type="Hex20", **box).num_nodes == 27 + 54
