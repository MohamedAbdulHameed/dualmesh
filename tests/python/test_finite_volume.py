# SPDX-License-Identifier: LGPL-2.1-or-later
"""Verification of the finite volume method against Chapter 3 of

    J. N. Reddy, *Computational Methods in Engineering: Finite Difference,
    Finite Volume, Finite Element, and Dual Mesh Control Domain Methods*,
    CRC Press, 2024.

Two finite volume formulations are covered.

``"hfvm"`` is the half-control volume formulation: the control volumes are
those of the dual mesh, and the gradient at an interface is the two-point
difference along the edge that joins the two nodes.  In one dimension, and on
rectangular meshes, this is exactly the scheme of Eqs. (3.2.16) and (3.4.6).

``"zfvm"`` is the zero-thickness control volume formulation: one unknown at
every cell centroid and one at every boundary face centroid, the layout every
production finite volume code uses.  Both the first-order boundary difference
and the second-order one-sided formula of Eq. (3.2.14) are tested.
"""

from __future__ import annotations

import dualmesh as dm
import numpy as np
import pytest


def cosine_source_problem(
    method,
    num_elements,
    case,
    boundary_gradient="first_order",
    quadrature="control_domain_trapezoid",
):
    """-u'' = 10 cos(x) on (0, 1); Example 3.3.1 of the book."""
    mesh = dm.generate_line_mesh(start=0.0, end=1.0, num_elements=num_elements)
    problem = dm.Problem(mesh, method=method, boundary_gradient=boundary_gradient)
    problem.add_variable("u")
    problem.add_kernel("Diffusion", "diffusion", variable="u")
    problem.add_function("source", lambda x, y, z, t: 10.0 * np.cos(x))
    problem.add_kernel("BodyForce", "source", variable="u", value="source", quadrature=quadrature)
    problem.add_boundary_condition("DirichletBC", "left", variable="u", boundary="left", value=0.0)
    if case == "dirichlet":
        problem.add_boundary_condition(
            "DirichletBC", "right", variable="u", boundary="right", value=0.0
        )
    problem.solve()
    return problem


# ---------------------------------------------------------------------------
# Example 3.3.1 and Table 3.3.1: one-dimensional problem, both formulations
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "case, expected",
    [
        # Half-control volume formulation, four subdivisions (Table 3.3.1).
        ("dirichlet", [0.0, 0.8362, 1.0715, 0.7626, 0.0]),
        ("mixed", [0.0, 1.7834, 2.9659, 3.6042, 3.7888]),
    ],
)
def test_half_control_volume_one_dimension(case, expected):
    problem = cosine_source_problem("hfvm", 4, case)
    assert problem.values("u") == pytest.approx(expected, abs=6e-5)


def test_half_control_volume_eight_subdivisions():
    """The HFVM(8) column of Table 3.3.1."""
    problem = cosine_source_problem("hfvm", 8, "dirichlet")
    # The nodes of the eight-subdivision mesh; the intermediate entries of
    # Table 3.3.1 are underlined there because they are interpolated.
    expected = [0.0, 0.4963, 0.8378, 1.0283, 1.0736, 0.9821, 0.7641, 0.4320, 0.0]
    assert problem.values("u") == pytest.approx(expected, abs=6e-5)


def test_half_control_volume_is_the_dual_mesh_method_in_one_dimension():
    """Reddy, Section 5.1: "The DMCDM ... is the same as the half-control FVM"."""
    hfvm = cosine_source_problem("hfvm", 6, "mixed")
    dmcdm = cosine_source_problem("dmcdm", 6, "mixed")
    assert hfvm.values("u") == pytest.approx(dmcdm.values("u"), abs=1e-12)


@pytest.mark.parametrize(
    "boundary_gradient, case, expected",
    [
        # ZFVM with the first-order boundary difference, Eq. (7) of Example 3.3.1.
        ("first_order", "dirichlet", [0.5686, 1.0906, 1.0356, 0.4777]),
        # ZFVM with the second-order boundary formula, Eq. (8).
        ("second_order", "dirichlet", [0.4951, 1.0240, 0.9757, 0.4246]),
        # Case 2, Eqs. (12) and (13).
        ("first_order", "mixed", [1.0464, 2.5238, 3.4242, 3.8217]),
        ("second_order", "mixed", [0.9694, 2.4469, 3.3473, 3.7448]),
    ],
)
def test_zero_thickness_control_volume_one_dimension(boundary_gradient, case, expected):
    problem = cosine_source_problem("zfvm", 4, case, boundary_gradient, quadrature="trapezoid")
    cells = problem.values("u")[:4]
    assert cells == pytest.approx(expected, abs=1.1e-4)


def test_zero_thickness_boundary_value_second_order():
    """Eq. (14): the boundary value recovered from the second-order formula."""
    problem = cosine_source_problem("zfvm", 4, "mixed", "second_order", quadrature="trapezoid")
    right = problem.boundary_entities("right")[0]
    assert problem.values("u")[right] == pytest.approx(3.7945, abs=1.1e-4)


@pytest.mark.parametrize(
    "method, boundary_gradient, quadrature, expected",
    [
        # Table 3.3.1, the rows -q(0) and q(1).
        ("hfvm", "first_order", "control_domain_trapezoid", (4.5898, 3.7888)),
        ("zfvm", "first_order", "trapezoid", (4.5492, 3.8217)),
        ("zfvm", "second_order", "trapezoid", (4.5764, 3.7944)),
    ],
)
def test_boundary_heats_table_3_3_1(method, boundary_gradient, quadrature, expected):
    """The secondary variables at the two ends, Table 3.3.1."""
    problem = cosine_source_problem(
        "hfvm" if method == "hfvm" else "zfvm", 4, "dirichlet", boundary_gradient, quadrature
    )
    left = -problem.total_reaction("u", "left")
    right = -problem.total_reaction("u", "right")
    assert (left, right) == pytest.approx(expected, abs=6e-5)


# ---------------------------------------------------------------------------
# Example 3.4.1: two-dimensional conduction on a 3a x 2a rectangle
# ---------------------------------------------------------------------------
def conduction_3a_by_2a(method, nx, ny):
    mesh = dm.generate_rectangle_mesh(0.0, 3.0, 0.0, 2.0, nx, ny)
    problem = dm.Problem(mesh, method=method)
    problem.add_variable("temperature")
    problem.add_kernel("Diffusion", "conduction", variable="temperature")
    problem.add_function("top", lambda x, y, z, t: np.cos(np.pi * x / 6.0))
    problem.add_boundary_condition(
        "DirichletBC", "cold", variable="temperature", boundary="right", value=0.0
    )
    problem.add_boundary_condition(
        "DirichletBC", "hot", variable="temperature", boundary="top", value="top"
    )
    problem.solve()
    return problem


def conduction_exact(x, y):
    return np.cosh(np.pi * y / 6.0) * np.cos(np.pi * x / 6.0) / np.cosh(np.pi / 3.0)


def test_example_3_4_1_half_control_volume():
    """Eq. (9): the 3 x 2 half-control volume solution."""
    problem = conduction_3a_by_2a("hfvm", 3, 2)
    points = [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1)]
    values = [problem.values("temperature")[problem.node_at(p)] for p in points]
    assert values == pytest.approx([0.6362, 0.5510, 0.3181, 0.7214, 0.6248, 0.3607], abs=1.1e-4)


def test_example_3_4_1_zero_thickness_control_volume():
    """Eq. (14): the 3 x 2 zero-thickness control volume solution."""
    problem = conduction_3a_by_2a("zfvm", 3, 2)
    centroids = problem.entity_points()[:6, :2]
    wanted = [(0.5, 0.5), (1.5, 0.5), (2.5, 0.5), (0.5, 1.5), (1.5, 1.5), (2.5, 1.5)]
    order = [int(np.argmin(np.linalg.norm(centroids - np.array(c), axis=1))) for c in wanted]
    values = problem.values("temperature")[order]
    assert values == pytest.approx([0.6145, 0.4499, 0.1647, 0.7792, 0.5704, 0.2088], abs=1.1e-4)


def test_example_3_4_1_differs_from_the_dual_mesh_method_in_two_dimensions():
    """In two dimensions the HFVM and the DMCDM are close but not identical.

    The HFVM keeps the interpolated gradient transverse to the edge and
    replaces only its component along the edge by the two-point difference
    between the two nodes.  In one dimension there is no transverse part and
    the two methods coincide exactly; in two dimensions the replaced component
    differs from the interpolated one, so the answers separate.
    """
    hfvm = conduction_3a_by_2a("hfvm", 3, 2).values("temperature")
    dmcdm = conduction_3a_by_2a("dmcdm", 3, 2).values("temperature")
    assert np.abs(hfvm - dmcdm).max() > 1e-3
    assert np.abs(hfvm - dmcdm).max() < 5e-2


# ---------------------------------------------------------------------------
# Properties that must hold for arbitrary meshes
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("method", ["hfvm", "zfvm"])
@pytest.mark.parametrize("element_type", ["Quad4", "Tri3"])
@pytest.mark.parametrize("distorted", [False, True])
def test_patch_test(method, element_type, distorted):
    """A linear field is reproduced exactly, even on a distorted mesh."""
    mesh = dm.generate_rectangle_mesh(0.0, 1.0, 0.0, 1.0, 4, 4, element_type=element_type)
    if distorted:
        mesh.transform_nodes(
            lambda x, y, z: (
                x + 0.3 * x * (1 - x) * y * (1 - y),
                y - 0.2 * x * (1 - x) * y * (1 - y),
                z,
            )
        )
    problem = dm.Problem(mesh, method=method, boundary_gradient="second_order")
    problem.add_variable("u")
    problem.add_kernel("Diffusion", "diffusion", variable="u")
    problem.add_function("linear", lambda x, y, z, t: 1.0 + 2.0 * x - 3.0 * y)
    problem.add_boundary_condition(
        "DirichletBC",
        "all",
        variable="u",
        boundary=["left", "right", "bottom", "top"],
        value="linear",
    )
    problem.solve()
    points = problem.entity_points()
    exact = 1.0 + 2.0 * points[:, 0] - 3.0 * points[:, 1]
    assert problem.values("u") == pytest.approx(exact, abs=1e-8)


@pytest.mark.parametrize("method", ["hfvm", "zfvm"])
def test_patch_test_three_dimensions(method):
    mesh = dm.generate_box_mesh(0, 1, 0, 1, 0, 1, 3, 3, 3)
    problem = dm.Problem(mesh, method=method)
    problem.add_variable("u")
    problem.add_kernel("Diffusion", "diffusion", variable="u")
    problem.add_function("linear", lambda x, y, z, t: 1.0 + 2.0 * x - 3.0 * y + 0.5 * z)
    problem.add_boundary_condition(
        "DirichletBC",
        "all",
        variable="u",
        boundary=["left", "right", "bottom", "top", "back", "front"],
        value="linear",
    )
    problem.solve()
    points = problem.entity_points()
    exact = 1.0 + 2.0 * points[:, 0] - 3.0 * points[:, 1] + 0.5 * points[:, 2]
    assert problem.values("u") == pytest.approx(exact, abs=1e-10)


@pytest.mark.parametrize("method", ["hfvm", "zfvm"])
@pytest.mark.parametrize("element_type", ["Quad4", "Tri3"])
def test_second_order_convergence(method, element_type):
    errors = []
    for n in (2, 4, 8):
        problem = conduction_3a_by_2a(method, 3 * n, 2 * n) if element_type == "Quad4" else None
        if problem is None:
            mesh = dm.generate_rectangle_mesh(0.0, 3.0, 0.0, 2.0, 3 * n, 2 * n, element_type="Tri3")
            problem = dm.Problem(mesh, method=method)
            problem.add_variable("temperature")
            problem.add_kernel("Diffusion", "conduction", variable="temperature")
            problem.add_function("top", lambda x, y, z, t: np.cos(np.pi * x / 6.0))
            problem.add_boundary_condition(
                "DirichletBC", "cold", variable="temperature", boundary="right", value=0.0
            )
            problem.add_boundary_condition(
                "DirichletBC", "hot", variable="temperature", boundary="top", value="top"
            )
            problem.solve()
        points = problem.entity_points()
        exact = conduction_exact(points[:, 0], points[:, 1])
        errors.append(np.abs(problem.values("temperature") - exact).max())
    rates = [np.log2(errors[i] / errors[i + 1]) for i in range(2)]
    assert min(rates) > 1.8
    assert errors[-1] < errors[0]


# ---------------------------------------------------------------------------
# The finite volume methods share the nonlinear, transient and boundary
# condition machinery with the dual mesh method
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("method", ["dmcdm", "hfvm", "zfvm"])
@pytest.mark.parametrize("solver", ["newton", "picard"])
def test_nonlinear_conduction(method, solver):
    """-d/dx(k(T) dT/dx) = 0 with k(T) = 1 + T/2, T(0) = 0, T(1) = 1.

    The Kirchhoff transform gives T + T^2/4 = x (1 + 1/4) in closed form.
    """
    mesh = dm.generate_line_mesh(start=0.0, end=1.0, num_elements=40)
    problem = dm.Problem(mesh, method=method)
    problem.add_variable("temperature")
    problem.add_kernel(
        "HeatConduction",
        "conduction",
        variable="temperature",
        thermal_conductivity=1.0,
        temperature_polynomial=[1.0, 0.5],
    )
    problem.add_boundary_condition(
        "DirichletBC", "left", variable="temperature", boundary="left", value=0.0
    )
    problem.add_boundary_condition(
        "DirichletBC", "right", variable="temperature", boundary="right", value=1.0
    )
    result = problem.solve(nonlinear_solver=solver)
    assert result.converged
    x = problem.entity_points()[:, 0]
    exact = (-1.0 + np.sqrt(1.0 + x * (1.0 + 0.25))) / 0.5
    assert problem.values("temperature") == pytest.approx(exact, abs=1e-4)


@pytest.mark.parametrize("method", ["hfvm", "zfvm"])
def test_transient_slab(method):
    """T_t = T_xx with T(x, 0) = sin(pi x) decays as exp(-pi^2 t) sin(pi x)."""
    mesh = dm.generate_line_mesh(start=0.0, end=1.0, num_elements=40)
    problem = dm.Problem(mesh, method=method)
    problem.add_variable("temperature", initial_condition=lambda x, y, z, t: np.sin(np.pi * x))
    problem.add_kernel("Diffusion", "diffusion", variable="temperature")
    problem.add_kernel("TimeDerivative", "time", variable="temperature")
    problem.add_boundary_condition(
        "DirichletBC", "ends", variable="temperature", boundary=["left", "right"], value=0.0
    )
    problem.solve_transient(end_time=0.1, dt=0.001, theta=0.5)
    x = problem.entity_points()[:, 0]
    exact = np.exp(-(np.pi**2) * 0.1) * np.sin(np.pi * x)
    assert problem.values("temperature") == pytest.approx(exact, abs=3e-4)


@pytest.mark.parametrize("method", ["dmcdm", "hfvm", "zfvm"])
def test_convective_boundary_condition(method):
    """A fin with a convecting end: all three methods converge to the same
    analytical solution, which checks the sign of the Robin condition."""
    length, base, transfer, m = 1.0, 100.0, 2.0, 2.0

    def exact(x):
        return (
            base
            * (np.cosh(m * (length - x)) + (transfer / m) * np.sinh(m * (length - x)))
            / (np.cosh(m * length) + (transfer / m) * np.sinh(m * length))
        )

    errors = []
    for n in (8, 16, 32):
        mesh = dm.generate_line_mesh(start=0.0, end=length, num_elements=n)
        problem = dm.Problem(mesh, method=method)
        problem.add_variable("u")
        problem.add_kernel("Diffusion", "diffusion", variable="u", diffusivity=1.0)
        problem.add_kernel("Reaction", "convection", variable="u", coefficient=m * m)
        problem.add_boundary_condition(
            "DirichletBC", "base", variable="u", boundary="left", value=base
        )
        problem.add_boundary_condition(
            "RobinBC", "tip", variable="u", boundary="right", transfer_coefficient=transfer
        )
        problem.solve()
        errors.append(np.abs(problem.values("u") - exact(problem.entity_points()[:, 0])).max())
    rates = [np.log2(errors[i] / errors[i + 1]) for i in range(2)]
    assert min(rates) > 1.8


def test_second_order_boundary_gradient_is_more_accurate():
    """Eq. (3.2.14) reduces the boundary error of the cell-centred method."""
    first = cosine_source_problem("zfvm", 8, "dirichlet", "first_order", "trapezoid")
    second = cosine_source_problem("zfvm", 8, "dirichlet", "second_order", "trapezoid")

    def error(problem):
        x = problem.entity_points()[:, 0]
        exact = 10.0 * (-(1.0 - np.cos(x)) + x * (1.0 - np.cos(1.0)))
        return np.abs(problem.values("u") - exact).max()

    assert error(second) < 0.4 * error(first)


@pytest.mark.parametrize("boundary_gradient", ["first_order", "second_order"])
def test_the_cell_centred_jacobian_is_exact_on_a_skewed_mesh(boundary_gradient):
    """The non-orthogonal correction depends on the reconstructed gradients of
    the two cells, and so on every entity of their least-squares stencils.
    Those dependencies are carried through the automatic differentiation, so
    the Jacobian is exact and a linear problem converges in one Newton step
    even on a mesh of pyramids, whose centroids lie far off the normals through
    their faces.  With the correction lagged instead, the same problem needed
    thousands of iterations, contracting by about 0.99 per iteration."""
    mesh = dm.generate_box_mesh(
        x_min=0.0,
        x_max=1.0,
        y_min=0.0,
        y_max=1.0,
        z_min=0.0,
        z_max=1.0,
        num_x_elements=2,
        num_y_elements=2,
        num_z_elements=2,
        element_type="Pyramid5",
    )
    mesh.transform_nodes(lambda x, y, z: [x + 0.2 * x * (1 - x) * y, y, z + 0.1 * z * (1 - z) * x])
    problem = dm.Problem(mesh, method="zfvm", boundary_gradient=boundary_gradient)
    problem.add_variable("u")
    problem.add_kernel("Diffusion", "diffusion", variable="u")
    problem.add_boundary_condition(
        "DirichletBC",
        "all",
        variable="u",
        boundary=mesh.sideset_names(),
        value=lambda x, y, z, t: 1.0 + 2.0 * x - 3.0 * y + 0.5 * z,
    )
    result = problem.solve()
    assert result.total_iterations == 1
    points = problem.entity_points()
    expected = 1.0 + 2.0 * points[:, 0] - 3.0 * points[:, 1] + 0.5 * points[:, 2]
    assert problem.values("u") == pytest.approx(expected, abs=1e-13)
