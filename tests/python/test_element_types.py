# SPDX-License-Identifier: LGPL-2.1-or-later
"""Tests of the element types that not every method can use.

Every element supports the finite element method, which needs only the element
itself.  The dual mesh control domain method integrates over the control
domains of a dual mesh, so it needs an element that carries one.  ``Wedge6`` does.  ``Quad8`` and ``Hex20`` do not, because a serendipity
element has no interior node and the median dual is undefined for it, and
``Pyramid5`` does not, because its apex is shared by four edges and the
decomposition its control domain would need is not implemented.

The tests below check that each element is offered to exactly the methods that
can use it, that each passes the patch test with every method that accepts it,
and that the elements work together in hybrid meshes.
"""

from __future__ import annotations

import dualmesh
import numpy
import pytest
from dualmesh._core import InputError

methods = ("dmcdm", "fem")
supported = {
    "Quad8": {"fem"},
    "Hex20": {"fem"},
    "Wedge6": set(methods),
    "Pyramid5": {"fem"},
}


def distorted_square(element_type, num_elements=3):
    mesh = dualmesh.generate_rectangle_mesh(
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
    mesh = dualmesh.generate_box_mesh(
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

    problem = dualmesh.Problem(mesh, method=method)
    problem.add_variable("u")
    problem.add_kernel("diffusion", variable="u")
    problem.add_boundary_condition(
        "Dirichlet",
        mesh.sideset_names(),
        variable="u",
        value=exact,
    )
    problem.solve()
    points = problem.node_coordinates()
    expected = 1.0 + 2.0 * points[:, 0] - 3.0 * points[:, 1] + 0.5 * points[:, 2]
    return float(numpy.abs(problem.values("u") - expected).max())


def hybrid_mesh(second_type):
    """A box whose cells with x < 1/2 are hexahedra and whose other cells are
    converted to ``second_type`` (``Wedge6`` or ``Pyramid5``).

    Both conversions keep the faces between the two halves whole, so the mesh
    is conforming: a prism split on its base diagonal presents a full
    quadrilateral to its neighbour in x, and a cell divided into six pyramids
    presents each of its faces as the base of one of them.
    """
    base = dualmesh.generate_box_mesh(
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
    points = [list(p) for p in numpy.asarray(base.points())]
    cells = []
    for _type, connectivity, _block in base.cells():
        h = list(connectivity)
        centroid = numpy.mean([points[i] for i in h], axis=0)
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
    mesh = dualmesh.mesh_from_arrays(numpy.asarray(points), list(groups.items()))
    mesh.add_bounding_box_sidesets()
    mesh.transform_nodes(lambda x, y, z: [x + 0.1 * x * (1 - x) * y, y + 0.05 * y * (1 - y) * z, z])
    return mesh


# ---------------------------------------------------------------------------
# Which methods accept which elements
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("element_type", sorted(supported))
@pytest.mark.parametrize("method", methods)
def test_each_element_is_offered_to_exactly_the_methods_that_can_use_it(element_type, method):
    mesh = mesh_for(element_type)
    if method in supported[element_type]:
        dualmesh.Problem(mesh, method=method)
    else:
        with pytest.raises(ValueError, match=f"{element_type}.*method='FEM'"):
            dualmesh.Problem(mesh, method=method)


# ---------------------------------------------------------------------------
# The patch test
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "element_type, method",
    [(t, m) for t in sorted(supported) for m in methods if m in supported[t]],
)
def test_linear_field_is_exact_on_a_distorted_mesh(element_type, method):
    tolerance = 1e-12
    assert patch_test_error(mesh_for(element_type), method) < tolerance


@pytest.mark.parametrize("method", methods)
def test_a_hexahedron_and_prism_hybrid_mesh_passes_the_patch_test(method):
    """A hybrid mesh passes the patch test only if the elements agree on their
    shared faces.  For the two dual mesh methods it also needs their control
    domains to agree there: the quarter of a hexahedron's face that belongs to
    a node must be the quarter of the prism's face that belongs to it."""
    tolerance = 1e-12
    assert patch_test_error(hybrid_mesh("Wedge6"), method) < tolerance


@pytest.mark.parametrize("method", ["fem"])
def test_a_hexahedron_and_pyramid_hybrid_mesh_passes_the_patch_test(method):
    tolerance = 1e-12
    assert patch_test_error(hybrid_mesh("Pyramid5"), method) < tolerance


# ---------------------------------------------------------------------------
# Mesh operations
# ---------------------------------------------------------------------------
def total_volume(mesh):
    problem = dualmesh.Problem(mesh, method="fem")
    problem.add_variable("u")
    problem.set_values("u", numpy.ones(len(problem.node_coordinates())))
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


def test_prisms_and_pyramids_cannot_be_promoted_to_second_order():
    """No quadratic prism or pyramid is implemented, and leaving them linear
    next to quadratic neighbours would make the mesh non-conforming, so the
    promotion refuses rather than doing it silently."""
    for element_type in ("Wedge6", "Pyramid5"):
        with pytest.raises(ValueError, match="cannot be promoted"):
            distorted_cube(element_type, num_elements=1).second_order()


@pytest.mark.parametrize("quadrature", ["interface", "control_domain_trapezoid"])
def test_control_domain_rules_refuse_elements_without_control_domains(quadrature):
    """The interface and control_domain_trapezoid rules place their points in
    the control domains of the nodes, which a serendipity element lacks."""
    problem = dualmesh.Problem(distorted_square("Quad8", num_elements=2), method="fem")
    problem.add_variable("u")
    problem.add_kernel("diffusion", variable="u")
    problem.add_kernel("body_force", variable="u", value=1.0, quadrature=quadrature)
    problem.add_boundary_condition("Dirichlet", "left", variable="u", value=0.0)
    with pytest.raises(InputError, match="does not have"):
        problem.solve()


def test_interface_rule_under_FEM_weights_each_node_by_its_shape_function():
    """-u'' = 2 with u(0) = u(1) = 0 has the solution u = x (1 - x).  Under the
    finite element method the interface rule weights its point by the integral
    of the shape function, so linear elements keep the nodal values exact."""
    mesh = dualmesh.generate_line_mesh(start=0.0, end=1.0, num_elements=8)
    problem = dualmesh.Problem(mesh, method="fem")
    problem.add_variable("u")
    problem.add_kernel("diffusion", variable="u")
    problem.add_kernel("body_force", variable="u", value=2.0, quadrature="interface")
    problem.add_boundary_condition("Dirichlet", "left", variable="u", value=0.0)
    problem.add_boundary_condition("Dirichlet", "right", variable="u", value=0.0)
    problem.solve()
    x = numpy.linspace(0.0, 1.0, 9)
    assert problem.values("u") == pytest.approx(x * (1.0 - x), abs=1e-10)
