# SPDX-License-Identifier: LGPL-2.1-or-later
"""Tests of the quadratic element family and of the dual mesh built on it.

dualmesh supports two families of elements.  The linear family (``Edge2``,
``Tri3``, ``Quad4``, ``Tet4``, ``Hex8``) interpolates with polynomials of
degree one; the quadratic family (``Edge3``, ``Tri6``, ``Quad9``, ``Tet10``,
``Hex27``) adds a node at the midpoint of every edge, at the centre of every
quadrilateral face and, for ``Quad9`` and ``Hex27``, at the centre of the
element, and interpolates with polynomials of degree two.

The serendipity elements ``Quad8`` and ``Hex20`` are deliberately absent.  They
have the mid-edge nodes but no interior node, so the median dual has no
sub-cell to hand to a centroid node and the construction is undefined for
them; asking for one gives an error that says so.

Three properties are checked here.  First, that the dual mesh built on a
quadratic element is a genuine partition of it: the control domains tile the
element and their surfaces close.  Second, that every discretisation still
reproduces a linear field exactly on a distorted quadratic mesh, which is the
standard patch test.  Third, the accuracy the quadratic elements actually buy,
which is different for the finite element method and for the dual mesh method
and is stated honestly in the documentation.
"""

from __future__ import annotations

import dualmesh as dm
import numpy as np
import pytest

QUADRATIC_2D = ["Tri6", "Quad9"]
QUADRATIC_3D = ["Tet10", "Hex27"]
METHODS = ["dmcdm", "fem", "hfvm", "zfvm"]


def distorted_square(num_elements=3, element_type="Quad9"):
    """A unit square whose interior nodes are pushed off the structured grid.

    The distortion vanishes on the boundary, so the domain is still the unit
    square, but no element is a parallelogram and, for a quadratic mesh, no
    edge is straight.
    """
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


def distorted_cube(num_elements=2, element_type="Hex27"):
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


# ---------------------------------------------------------------------------
# Promotion of a linear mesh to second order
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "linear, num_nodes",
    [
        # 4 elements: 5 corner nodes and 4 mid-edge nodes.
        ("Edge2", 9),
        # 2 x 2 elements: 9 corner nodes, 12 mid-edge nodes and 4 element
        # centres, which is the 5 x 5 grid of a once-refined mesh.
        ("Quad4", 25),
        # The same square split into 8 triangles: 9 corner nodes and 16
        # mid-edge nodes (6 horizontal, 6 vertical and 4 diagonal edges).
        ("Tri3", 25),
    ],
)
def test_promotion_gives_the_expected_number_of_nodes(linear, num_nodes):
    if linear == "Edge2":
        mesh = dm.generate_line_mesh(start=0.0, end=1.0, num_elements=4)
        expected_elements = 4
    else:
        mesh = dm.generate_rectangle_mesh(
            x_min=0.0,
            x_max=1.0,
            y_min=0.0,
            y_max=1.0,
            num_x_elements=2,
            num_y_elements=2,
            element_type="Tri3" if linear == "Tri6" else linear,
        )
        expected_elements = mesh.num_elements
    promoted = mesh.second_order()
    assert promoted.num_elements == expected_elements
    assert promoted.num_nodes == num_nodes
    # The corner nodes keep their numbers and their positions.
    original = np.asarray(mesh.points())
    assert np.asarray(promoted.points())[: mesh.num_nodes] == pytest.approx(original)


def test_promotion_keeps_the_boundary_sets():
    mesh = dm.generate_rectangle_mesh(
        x_min=0.0, x_max=1.0, y_min=0.0, y_max=2.0, num_x_elements=3, num_y_elements=3
    )
    promoted = mesh.second_order()
    assert sorted(promoted.sideset_names()) == sorted(mesh.sideset_names())
    # A side set now covers the mid-edge nodes as well, so every node the side
    # set touches is on the geometric boundary it names.
    points = np.asarray(promoted.points())
    for name, coordinate, value in (
        ("left", 0, 0.0),
        ("right", 0, 1.0),
        ("bottom", 1, 0.0),
        ("top", 1, 2.0),
    ):
        ids = promoted.boundary_nodes(name)
        assert len(ids) == 7  # 4 corner nodes and 3 mid-edge nodes
        assert points[ids, coordinate] == pytest.approx(value)


def test_promoting_an_already_quadratic_mesh_changes_nothing():
    mesh = distorted_square(element_type="Quad9")
    again = mesh.second_order()
    assert again.num_nodes == mesh.num_nodes
    assert np.asarray(again.points()) == pytest.approx(np.asarray(mesh.points()))


def test_the_serendipity_elements_are_refused_by_the_dual_mesh_methods_only():
    """Quad8 has mid-edge nodes but no interior node, so the median dual is not
    defined for it.  It is still a perfectly good finite element, and the
    cell-centred finite volume method needs only its geometry, so the mesh is
    accepted and only the two methods that integrate over control domains
    refuse it, naming the reason and the methods that do work."""
    mesh = dm.mesh_from_arrays(
        [[0, 0], [1, 0], [1, 1], [0, 1], [0.5, 0], [1, 0.5], [0.5, 1], [0, 0.5]],
        [[0, 1, 2, 3, 4, 5, 6, 7]],
        element_type="Quad8",
    )
    for method in ("dmcdm", "hfvm"):
        with pytest.raises(ValueError, match="no interior node.*method='fem'"):
            dm.Problem(mesh, method=method)
    for method in ("fem", "zfvm"):
        dm.Problem(mesh, method=method)


def test_a_quadratic_mesh_cannot_be_refined_directly():
    mesh = distorted_square()
    with pytest.raises(Exception, match="quadratic"):
        mesh.refined()


# ---------------------------------------------------------------------------
# The patch test
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("element_type", QUADRATIC_2D)
@pytest.mark.parametrize("method", METHODS)
def test_linear_field_is_exact_in_two_dimensions(element_type, method):
    """Every discretisation must reproduce a linear field exactly, whatever the
    element and however distorted the mesh.  This is the patch test, and it is
    the minimum a discretisation has to pass to be consistent."""
    mesh = distorted_square(element_type=element_type)

    def exact(x, y, z, t):
        return 1.0 + 2.0 * x - 3.0 * y

    problem = dm.Problem(mesh, method=method)
    problem.add_variable("u")
    problem.add_kernel("Diffusion", "diffusion", variable="u")
    problem.add_boundary_condition(
        "DirichletBC", "all", variable="u", boundary=mesh.sideset_names(), value=exact
    )
    problem.solve()
    points = problem.entity_points()
    expected = 1.0 + 2.0 * points[:, 0] - 3.0 * points[:, 1]
    tolerance = 1e-12
    assert problem.values("u") == pytest.approx(expected, abs=tolerance)


@pytest.mark.parametrize("element_type", QUADRATIC_3D)
@pytest.mark.parametrize("method", ["dmcdm", "fem"])
def test_linear_field_is_exact_in_three_dimensions(element_type, method):
    mesh = distorted_cube(element_type=element_type)

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
    assert problem.values("u") == pytest.approx(expected, abs=1e-11)


# ---------------------------------------------------------------------------
# Accuracy
# ---------------------------------------------------------------------------
def poisson_error(num_elements, element_type, method, quadrature="automatic"):
    """Nodal error for -div(grad u) = 2 pi^2 sin(pi x) sin(pi y) on the unit
    square with u = 0 on the boundary, whose solution is
    u = sin(pi x) sin(pi y)."""
    mesh = dm.generate_rectangle_mesh(
        x_min=0.0,
        x_max=1.0,
        y_min=0.0,
        y_max=1.0,
        num_x_elements=num_elements,
        num_y_elements=num_elements,
        element_type=element_type,
    )
    problem = dm.Problem(mesh, method=method)
    problem.add_variable("u")
    problem.add_kernel("Diffusion", "diffusion", variable="u", quadrature=quadrature)
    problem.add_kernel(
        "BodyForce",
        "source",
        variable="u",
        value=lambda x, y, z, t: 2.0 * np.pi**2 * np.sin(np.pi * x) * np.sin(np.pi * y),
        quadrature=quadrature,
    )
    problem.add_boundary_condition(
        "DirichletBC", "walls", variable="u", boundary=mesh.sideset_names(), value=0.0
    )
    problem.solve()
    points = problem.entity_points()
    exact = np.sin(np.pi * points[:, 0]) * np.sin(np.pi * points[:, 1])
    return float(np.abs(problem.values("u") - exact).max())


def observed_rate(errors):
    return [np.log2(errors[i] / errors[i + 1]) for i in range(len(errors) - 1)]


@pytest.mark.parametrize("element_type", QUADRATIC_2D)
def test_finite_elements_superconverge_at_the_nodes(element_type):
    """With quadratic elements the finite element solution is fourth-order
    accurate at the nodes.  The rate in the energy norm is two and in the L2
    norm three; the extra order at the nodes is the classical superconvergence
    of the Galerkin method, and it is what makes quadratic elements worth their
    extra unknowns."""
    errors = [poisson_error(n, element_type, "fem") for n in (4, 8, 16)]
    rates = observed_rate(errors)
    assert min(rates) > 3.7


@pytest.mark.parametrize("element_type", QUADRATIC_2D)
def test_the_dual_mesh_method_stays_second_order_on_quadratic_elements(element_type):
    """The dual mesh method does not gain an order from quadratic elements, and
    the documentation says so.

    The reason is where the control domain interfaces sit.  The flux through an
    interface is the gradient of the interpolant evaluated there, and the
    gradient of a quadratic interpolant is third-order accurate only at the two
    Gauss points of the element, at xi = +- 1/sqrt(3).  The median dual puts its
    interfaces half way between the nodes, at xi = +- 1/2, where the gradient
    error is O(h^2); the balance of two such fluxes over a control domain of
    width O(h) is then a second-order truncation error.  The solution error
    follows.  The comparison against the linear element below shows the
    accuracy that is gained: a constant factor, not an order."""
    quadratic = [poisson_error(n, element_type, "dmcdm") for n in (4, 8, 16)]
    rates = observed_rate(quadratic)
    assert min(rates) > 1.8 and max(rates) < 2.3

    # Against the linear element on the same nodes, the change is a constant
    # factor whose size depends on the element, and it is not always an
    # improvement.  Promoting Quad4 to Quad9 divides the error by about three;
    # promoting Tri3 to Tri6 multiplies it by about eight, because the dual
    # mesh of a linear triangle is already unusually accurate for this
    # discretisation and the sub-cells of the refined triangle are not.  Both
    # ratios are steady as the mesh is refined, which is the statement that the
    # order has not changed.
    linear_type = {"Tri6": "Tri3", "Quad9": "Quad4"}[element_type]
    linear = [poisson_error(n, linear_type, "dmcdm") for n in (8, 16, 32)]
    ratios = [qe / le for qe, le in zip(quadratic, linear)]
    expected = {"Quad9": 0.34, "Tri6": 8.2}[element_type]
    assert ratios == pytest.approx([expected] * 3, rel=0.15)


def test_one_dimensional_dual_mesh_on_quadratic_elements_equals_the_refined_linear_mesh():
    """In one dimension the statement above can be made exactly.

    On an ``Edge3`` element with nodes at xi = -1, 0, +1 the interpolant is
    p(xi) = N_0 u_0 + N_1 u_1 + N_2 u_m, and its derivative is
    p'(xi) = (xi - 1/2) u_0 + (xi + 1/2) u_1 - 2 xi u_m.  At the control domain
    interface xi = -1/2 this collapses to p' = u_m - u_0, that is, to the
    two-point difference across the half element; the same happens at
    xi = +1/2.  The dual mesh discretisation of a quadratic mesh is therefore
    algebraically identical to the dual mesh discretisation of the linear mesh
    with the same nodes, and this test confirms it to machine precision.  The
    quadrature is fixed on both sides, because the default rule is chosen from
    the element order and a difference there would mask the identity.
    """

    def solve(num_elements, element_type):
        mesh = dm.generate_line_mesh(
            start=0.0, end=1.0, num_elements=num_elements, element_type=element_type
        )
        problem = dm.Problem(mesh, method="dmcdm")
        problem.add_variable("u")
        problem.add_kernel("Diffusion", "diffusion", variable="u", quadrature="gauss8")
        problem.add_kernel(
            "BodyForce",
            "source",
            variable="u",
            value=lambda x, y, z, t: np.pi**2 * np.sin(np.pi * x),
            quadrature="gauss8",
        )
        problem.add_boundary_condition(
            "DirichletBC", "ends", variable="u", boundary=["left", "right"], value=0.0
        )
        problem.solve()
        order = np.argsort(problem.entity_points()[:, 0])
        return problem.values("u")[order]

    assert solve(4, "Edge3") == pytest.approx(solve(8, "Edge2"), abs=1e-14)


@pytest.mark.parametrize("element_type", ["Quad9", "Tri6"])
def test_curved_boundaries_are_resolved_to_fourth_order(element_type):
    """What a quadratic mesh does buy the dual mesh method is the geometry.

    The mid-side node of an element on a circular boundary is placed on the
    circle, so the element edge is a parabola through three points of the arc
    instead of a chord through two.  The area of an annular sector, which a
    linear mesh gets wrong at second order, is then correct to fourth order.
    """
    exact_area = 0.25 * np.pi * (2.0**2 - 1.0**2)

    def area_error(num_elements, etype):
        mesh = dm.generate_annulus_mesh(
            inner_radius=1.0,
            outer_radius=2.0,
            num_radial_elements=num_elements,
            num_angular_elements=num_elements,
            element_type=etype,
        )
        problem = dm.Problem(mesh)
        problem.add_variable("u")
        problem.set_values("u", np.ones(len(problem.entity_points())))
        return abs(problem.integrate("u") - exact_area)

    errors = [area_error(n, element_type) for n in (2, 4, 8)]
    assert min(observed_rate(errors)) > 3.7
    linear_type = {"Tri6": "Tri3", "Quad9": "Quad4"}[element_type]
    assert errors[0] < 0.01 * area_error(2, linear_type)


# ---------------------------------------------------------------------------
# A quadratic solution is reproduced exactly
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("element_type", QUADRATIC_2D)
@pytest.mark.parametrize("method", ["dmcdm", "fem"])
def test_a_quadratic_solution_is_reproduced_exactly(element_type, method):
    """When the exact solution lies in the space the elements can represent,
    both discretisations must find it exactly: the interpolant is then the
    solution, so every flux balance and every weighted residual is satisfied
    identically.  Here u = 1 + x - 2 y + 3 x^2 - x y + y^2, whose Laplacian is
    the constant 8, so the source is constant.

    The mesh is graded but its elements are straight-sided, so the map from the
    reference element to the physical element is affine.  That matters: on a
    curved element the pull-back of x^2 is not a quadratic in the reference
    coordinates, the exact solution is then outside the space the elements can
    represent, and neither method reproduces it exactly.  The patch test above,
    which uses a linear field, is the one that must pass on a curved mesh.
    """
    mesh = dm.generate_rectangle_mesh(
        x_coordinates=[0.0, 0.3, 0.7, 1.0],
        y_coordinates=[0.0, 0.45, 0.6, 1.0],
        element_type=element_type,
    )

    def exact(x, y, z=0.0, t=0.0):
        return 1.0 + x - 2.0 * y + 3.0 * x**2 - x * y + y**2

    problem = dm.Problem(mesh, method=method)
    problem.add_variable("u")
    problem.add_kernel("Diffusion", "diffusion", variable="u")
    problem.add_kernel("BodyForce", "source", variable="u", value=-8.0)
    problem.add_boundary_condition(
        "DirichletBC", "all", variable="u", boundary=mesh.sideset_names(), value=exact
    )
    problem.solve()
    points = problem.entity_points()
    assert problem.values("u") == pytest.approx(exact(points[:, 0], points[:, 1]), abs=1e-10)


# ---------------------------------------------------------------------------
# Reading and writing
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("element_type", ["Edge3", "Tri6", "Quad9", "Tet10", "Hex27"])
def test_quadratic_meshes_survive_a_file_round_trip(tmp_path, element_type):
    """meshio stores the connectivity of a triquadratic hexahedron in the VTK
    order, in which the six face-centre nodes come x-min, x-max, y-min, y-max,
    z-min, z-max; dualmesh follows the Exodus order.  The reader and the writer
    apply the permutation, so a mesh written and read back must be the mesh
    that was written, node for node."""
    pytest.importorskip("meshio")
    if element_type == "Edge3":
        mesh = dm.generate_line_mesh(start=0.0, end=1.0, num_elements=3, element_type=element_type)
    elif element_type in ("Tri6", "Quad9"):
        mesh = distorted_square(num_elements=2, element_type=element_type)
    else:
        mesh = distorted_cube(num_elements=1, element_type=element_type)

    path = tmp_path / "quadratic.vtu"
    dm.write_mesh(mesh, str(path))
    back = dm.read_mesh(str(path), add_bounding_box_sidesets=True)
    assert back.num_elements == mesh.num_elements
    assert np.asarray(back.points()) == pytest.approx(np.asarray(mesh.points()))
    for (type_a, nodes_a, _), (type_b, nodes_b, _) in zip(mesh.cells(), back.cells()):
        assert type_a == type_b
        assert list(nodes_a) == list(nodes_b)


# ---------------------------------------------------------------------------
# Limits
# ---------------------------------------------------------------------------
def test_too_many_variables_for_the_derivative_budget_is_reported_clearly():
    """Automatic differentiation seeds one derivative slot per element node and
    variable.  A Hex27 mesh uses 27 slots per variable, so a second variable
    needs 54 and does not fit in the default budget of 48.  The error must say
    which element is responsible and how to raise the limit, not fail
    obscurely later."""
    mesh = distorted_cube(num_elements=1, element_type="Hex27")
    problem = dm.Problem(mesh)
    problem.add_variable("temperature")
    with pytest.raises(Exception, match="Hex27.*DUALMESH_MAX_AD_DERIVATIVES"):
        problem.add_variable("pressure")
    # The rejected variable was not left behind.
    with pytest.raises(ValueError, match="pressure"):
        problem.variable_index("pressure")
    assert problem.variable_index("temperature") == 0
