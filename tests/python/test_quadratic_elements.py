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

import dualmesh
import numpy
import pytest

quadratic_2d = ["Tri6", "Quad9"]
quadratic_3d = ["Tet10", "Hex27"]
methods = ["dmcdm", "fem"]


def distorted_square(num_elements=3, element_type="Quad9"):
    """A unit square whose interior nodes are pushed off the structured grid.

    The distortion vanishes on the boundary, so the domain is still the unit
    square, but no element is a parallelogram and, for a quadratic mesh, no
    edge is straight.
    """
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


def distorted_cube(num_elements=2, element_type="Hex27"):
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


def test_the_serendipity_elements_are_refused_by_the_dual_mesh_method_only():
    """Quad8 has mid-edge nodes but no interior node, so the median dual is not
    defined for it.  It is still a perfectly good finite element, so the mesh is
    accepted by FEM and refused by DMCDM, which names the reason and the method
    that does work."""
    mesh = dualmesh.mesh_from_arrays(
        [[0, 0], [1, 0], [1, 1], [0, 1], [0.5, 0], [1, 0.5], [0.5, 1], [0, 0.5]],
        [[0, 1, 2, 3, 4, 5, 6, 7]],
        element_type="Quad8",
    )
    with pytest.raises(ValueError, match="no interior node.*method='FEM'"):
        dualmesh.Problem(mesh, method="DMCDM")
    dualmesh.Problem(mesh, method="FEM")


# ---------------------------------------------------------------------------
# The patch test
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("element_type", quadratic_2d)
@pytest.mark.parametrize("method", methods)
def test_linear_field_is_exact_in_two_dimensions(element_type, method):
    """Every discretisation must reproduce a linear field exactly, whatever the
    element and however distorted the mesh.  This is the patch test, and it is
    the minimum a discretisation has to pass to be consistent."""
    mesh = distorted_square(element_type=element_type)

    def exact(x, y, z, t):
        return 1.0 + 2.0 * x - 3.0 * y

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
    expected = 1.0 + 2.0 * points[:, 0] - 3.0 * points[:, 1]
    tolerance = 1e-12
    assert problem.values("u") == pytest.approx(expected, abs=tolerance)


@pytest.mark.parametrize("element_type", quadratic_3d)
@pytest.mark.parametrize("method", ["dmcdm", "fem"])
def test_linear_field_is_exact_in_three_dimensions(element_type, method):
    mesh = distorted_cube(element_type=element_type)

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
    assert problem.values("u") == pytest.approx(expected, abs=1e-11)


# ---------------------------------------------------------------------------
# Accuracy
# ---------------------------------------------------------------------------
def Poisson_error(num_elements, element_type, method, quadrature="automatic", diagonal="alternate"):
    """Nodal error for -div(grad u) = 2 pi^2 sin(pi x) sin(pi y) on the unit
    square with u = 0 on the boundary, whose solution is
    u = sin(pi x) sin(pi y)."""
    mesh = dualmesh.generate_rectangle_mesh(
        x_min=0.0,
        x_max=1.0,
        y_min=0.0,
        y_max=1.0,
        num_x_elements=num_elements,
        num_y_elements=num_elements,
        element_type=element_type,
        diagonal=diagonal,
    )
    problem = dualmesh.Problem(mesh, method=method)
    problem.add_variable("u")
    problem.add_kernel("diffusion", variable="u", quadrature=quadrature)
    problem.add_kernel(
        "body_force",
        variable="u",
        value=lambda x, y, z, t: (
            2.0 * numpy.pi**2 * numpy.sin(numpy.pi * x) * numpy.sin(numpy.pi * y)
        ),
        quadrature=quadrature,
    )
    problem.add_boundary_condition(
        "Dirichlet",
        mesh.sideset_names(),
        variable="u",
        value=0.0,
    )
    problem.solve()
    points = problem.node_coordinates()
    exact = numpy.sin(numpy.pi * points[:, 0]) * numpy.sin(numpy.pi * points[:, 1])
    return float(numpy.abs(problem.values("u") - exact).max())


def observed_rate(errors):
    return [numpy.log2(errors[i] / errors[i + 1]) for i in range(len(errors) - 1)]


@pytest.mark.parametrize("element_type", quadratic_2d)
def test_finite_elements_superconverge_at_the_nodes(element_type):
    """With quadratic elements the finite element solution is fourth-order
    accurate at the nodes.  The rate in the energy norm is two and in the L2
    norm three; the extra order at the nodes is the classical superconvergence
    of the Galerkin method, and it is what makes quadratic elements worth their
    extra unknowns.  It needs a uniform mesh, every cell alike: the triangles take one diagonal
    direction (with the alternating one, which repeats every second cell, the rates measured
    3.4 and 2.9)."""
    errors = [Poisson_error(n, element_type, "fem", diagonal="right") for n in (4, 8, 16)]
    rates = observed_rate(errors)
    assert min(rates) > 3.7


@pytest.mark.parametrize("element_type", quadratic_2d)
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
    quadratic = [Poisson_error(n, element_type, "dmcdm") for n in (4, 8, 16)]
    # The order over the two refinements: on Tri6 the rates are 2.37 and 2.12 as the error
    # approaches its asymptote, second order and not third.
    order = numpy.log2(quadratic[0] / quadratic[-1]) / 2
    assert 1.8 < order < 2.3, observed_rate(quadratic)

    # Against the linear element on the same nodes, the change is a constant
    # factor whose size depends on the element, and it is not always an
    # improvement.  Promoting Quad4 to Quad9 divides the error by about three;
    # on a uniform triangulation (one diagonal direction) promoting Tri3 to Tri6
    # multiplies it by about eight, because the dual mesh of such a linear
    # triangulation is unusually accurate for this discretisation and the
    # sub-cells of the refined triangle are not.  Both ratios are steady as the
    # mesh is refined, which is the statement that the order has not changed.
    # On the default alternating triangulation the linear triangles lose that
    # accuracy and the ratio falls towards one (1.55, 1.14, 1.03).
    linear_type = {"Tri6": "Tri3", "Quad9": "Quad4"}[element_type]
    uniform = "right"
    if element_type == "Tri6":
        quadratic = [Poisson_error(n, element_type, "dmcdm", diagonal=uniform) for n in (4, 8, 16)]
    linear = [Poisson_error(n, linear_type, "dmcdm", diagonal=uniform) for n in (8, 16, 32)]
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
        mesh = dualmesh.generate_line_mesh(
            start=0.0, end=1.0, num_elements=num_elements, element_type=element_type
        )
        problem = dualmesh.Problem(mesh, method="dmcdm")
        problem.add_variable("u")
        problem.add_kernel("diffusion", variable="u", quadrature="gauss8")
        problem.add_kernel(
            "body_force",
            variable="u",
            value=lambda x, y, z, t: numpy.pi**2 * numpy.sin(numpy.pi * x),
            quadrature="gauss8",
        )
        problem.add_boundary_condition(
            "Dirichlet",
            ["left", "right"],
            variable="u",
            value=0.0,
        )
        problem.solve()
        order = numpy.argsort(problem.node_coordinates()[:, 0])
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
    exact_area = 0.25 * numpy.pi * (2.0**2 - 1.0**2)

    def area_error(num_elements, etype):
        mesh = dualmesh.generate_annulus_mesh(
            inner_radius=1.0,
            outer_radius=2.0,
            num_radial_elements=num_elements,
            num_angular_elements=num_elements,
            element_type=etype,
        )
        problem = dualmesh.Problem(mesh)
        problem.add_variable("u")
        problem.set_values("u", numpy.ones(len(problem.node_coordinates())))
        return abs(problem.integrate("u") - exact_area)

    errors = [area_error(n, element_type) for n in (2, 4, 8)]
    assert min(observed_rate(errors)) > 3.7
    linear_type = {"Tri6": "Tri3", "Quad9": "Quad4"}[element_type]
    assert errors[0] < 0.01 * area_error(2, linear_type)


# ---------------------------------------------------------------------------
# A quadratic solution is reproduced exactly
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("element_type", quadratic_2d)
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
    mesh = dualmesh.generate_rectangle_mesh(
        x_coordinates=[0.0, 0.3, 0.7, 1.0],
        y_coordinates=[0.0, 0.45, 0.6, 1.0],
        element_type=element_type,
    )

    def exact(x, y, z=0.0, t=0.0):
        return 1.0 + x - 2.0 * y + 3.0 * x**2 - x * y + y**2

    problem = dualmesh.Problem(mesh, method=method)
    problem.add_variable("u")
    problem.add_kernel("diffusion", variable="u")
    problem.add_kernel("body_force", variable="u", value=-8.0)
    problem.add_boundary_condition(
        "Dirichlet",
        mesh.sideset_names(),
        variable="u",
        value=exact,
    )
    problem.solve()
    points = problem.node_coordinates()
    assert problem.values("u") == pytest.approx(exact(points[:, 0], points[:, 1]), abs=1e-10)


# ---------------------------------------------------------------------------
# Limits
# ---------------------------------------------------------------------------
def test_too_many_variables_for_the_derivative_budget_is_reported_clearly():
    """Automatic differentiation seeds one derivative slot per element node and
    variable.  A Hex27 mesh uses 27 slots per variable, so a fourth variable
    needs 108 and does not fit in the default budget of 96.  The error must
    say which element is responsible and how to raise the limit, not fail
    obscurely later."""
    mesh = distorted_cube(num_elements=1, element_type="Hex27")
    problem = dualmesh.Problem(mesh)
    for name in ("disp_x", "disp_y", "temperature"):
        problem.add_variable(name)
    with pytest.raises(Exception, match="Hex27.*DUALMESH_MAX_AD_DERIVATIVES"):
        problem.add_variable("pressure")
    # The rejected variable was not left behind.
    with pytest.raises(ValueError, match="pressure"):
        problem.variable_index("pressure")
    assert problem.variable_index("temperature") == 2
    # A first-order fourth variable lives on the 8 corners only: 81 + 8 = 89 fits.
    problem.add_variable("pressure", order="first")
