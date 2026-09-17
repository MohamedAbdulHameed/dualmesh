# SPDX-License-Identifier: LGPL-2.1-or-later
"""Verification against the heat-transfer examples of

    J. N. Reddy, *Computational Methods in Engineering: Finite Element,
    Finite Volume, and Dual Mesh Control Domain Methods*, CRC Press, 2024.

Every reference number in this file is taken from the book (the tables of
Chapters 3, 5, and 6), so a failure here means the software no longer
reproduces the published dual mesh control domain results.
"""

from __future__ import annotations

import math

import dualmesh as dm
import numpy as np
import pytest


def line_problem(num_elements, length=1.0, start=0.0, method="dmcdm"):
    mesh = dm.generate_line_mesh(start=start, end=start + length, num_elements=num_elements)
    return dm.Problem(mesh, method=method)


# ---------------------------------------------------------------------------
# Chapter 3: one-dimensional finite volume (identical to the DMCDM)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "case, expected",
    [
        # Example 3.3.1, Case 1: u(0) = u(1) = 0, four elements (Table 3.3.1).
        ("dirichlet", [0.8362, 1.0715, 0.7626]),
        # Case 2: u(0) = 0, u'(1) = 0, four elements.
        ("mixed", [1.7834, 2.9659, 3.6042, 3.7888]),
    ],
)
def test_example_3_3_1_source_with_trapezoidal_rule(case, expected):
    """-u'' = 10 cos(x); the source is integrated with the trapezoidal rule."""
    problem = line_problem(4)
    problem.add_variable("u")
    problem.add_kernel("Diffusion", variable="u")
    problem.add_kernel(
        "BodyForce",
        variable="u",
        value=lambda x, y, z, t: 10.0 * math.cos(x),
        quadrature="control_domain_trapezoid",  # F_I = 0.5 dx [f(x_A) + f(x_B)]
    )
    problem.add_boundary_condition("DirichletBC", variable="u", boundary="left", value=0.0)
    if case == "dirichlet":
        problem.add_boundary_condition("DirichletBC", variable="u", boundary="right", value=0.0)
    else:
        problem.add_boundary_condition("NeumannBC", variable="u", boundary="right", flux=0.0)
    problem.solve()
    computed = problem.values("u")[1 : 1 + len(expected)]
    assert computed == pytest.approx(expected, abs=5e-4)


# ---------------------------------------------------------------------------
# Chapter 5: linear problems
# ---------------------------------------------------------------------------
def cooling_fin(num_elements):
    """Example 5.3.1: -u'' + 400 u = 0, u(0) = 300, u'(L) + 2 u(L) = 0."""
    problem = line_problem(num_elements, length=0.05)
    problem.add_variable("u")
    problem.add_kernel("Diffusion", variable="u", name="conduction")
    problem.add_kernel("Reaction", variable="u", coefficient=400.0)
    problem.add_boundary_condition("DirichletBC", variable="u", boundary="left", value=300.0)
    problem.add_boundary_condition(
        "RobinBC", variable="u", boundary="right", transfer_coefficient=2.0
    )
    problem.solve()
    return problem


def test_example_5_3_1_five_elements():
    problem = cooling_fin(5)
    expected = [300.0, 257.62, 225.59, 202.64, 187.83, 180.57]
    assert problem.values("u") == pytest.approx(expected, abs=6e-3)
    # Q(0) = 4817 (the secondary variable at the fixed end)
    assert problem.total_reaction("u", "left") == pytest.approx(4817.0, rel=2e-4)


def test_example_5_3_1_table_5_3_1():
    """Table 5.3.1: DMCDM solutions on 10 and 20 elements, and Q(0)."""
    sample_points = np.arange(1, 11) * 0.005
    references = {
        10: (
            [277.44, 257.65, 240.45, 225.65, 213.11, 202.70, 194.33, 187.90, 183.35, 180.64],
            4807.0,
        ),
        20: (
            [277.44, 257.66, 240.46, 225.66, 213.12, 202.72, 194.34, 187.91, 183.37, 180.65],
            4804.5,
        ),
    }
    for num_elements, (expected, heat) in references.items():
        problem = cooling_fin(num_elements)
        computed = problem.sample("u", sample_points.reshape(-1, 1))
        assert computed == pytest.approx(expected, abs=6e-3)
        assert problem.total_reaction("u", "left") == pytest.approx(heat, rel=3e-4)


@pytest.mark.parametrize("num_elements", [4, 8])
def test_example_5_3_2_cylinder_with_heat_generation(num_elements):
    """Example 5.3.2: -(1/r) d/dr (k r dT/dr) = g0 in a solid cylinder.

    The DMCDM reproduces the exact solution at the nodes, and the heat flow at
    the outer surface is pi R0^2 g0 = 2 pi 10^4 W/m.
    """
    outer_radius, conductivity, generation, surface_temperature = 0.01, 20.0, 2.0e8, 100.0
    mesh = dm.generate_line_mesh(start=0.0, end=outer_radius, num_elements=num_elements)
    problem = dm.Problem(mesh, method="dmcdm", coordinates="axisymmetric")
    problem.add_variable("temperature")
    problem.add_kernel("HeatConduction", variable="temperature", thermal_conductivity=conductivity)
    problem.add_kernel("HeatSource", variable="temperature", heat_source=generation)
    problem.add_boundary_condition(
        "DirichletBC", variable="temperature", boundary="right", value=surface_temperature
    )
    problem.solve()

    radii = np.linspace(0.0, outer_radius, num_elements + 1)
    exact = surface_temperature + generation * (outer_radius**2 - radii**2) / (4 * conductivity)
    assert problem.values("temperature") == pytest.approx(exact, rel=1e-10)
    # The reaction is the heat leaving through r = R0 (per unit length).
    assert problem.total_reaction("temperature", "right") == pytest.approx(
        -math.pi * outer_radius**2 * generation, rel=1e-10
    )


def rectangular_conduction(nx, ny, element_type="Quad4", method="dmcdm"):
    """Example 5.4.1: 3a x 2a region, insulated at x = 0 and y = 0."""
    mesh = dm.generate_rectangle_mesh(
        x_min=0.0,
        x_max=3.0,
        y_min=0.0,
        y_max=2.0,
        num_x_elements=nx,
        num_y_elements=ny,
        element_type=element_type,
    )
    problem = dm.Problem(mesh, method=method)
    problem.add_variable("temperature")
    problem.add_kernel("Diffusion", variable="temperature")
    problem.add_boundary_condition(
        "DirichletBC", variable="temperature", boundary="right", value=0.0
    )
    problem.add_boundary_condition(
        "DirichletBC",
        variable="temperature",
        boundary="top",
        value=lambda x, y, z, t: math.cos(math.pi * x / 6.0),
    )
    problem.solve()
    return problem


def test_example_5_4_1_coarse_mesh():
    """The 3 x 2 mesh nodal temperatures quoted in Example 5.4.1."""
    problem = rectangular_conduction(3, 2)
    values = problem.values("temperature")
    expected = {0: 0.6190, 1: 0.5360, 2: 0.3095, 4: 0.7078, 5: 0.6130, 6: 0.3539}
    for node, reference in expected.items():
        assert values[node] == pytest.approx(reference, abs=5e-4)


def test_example_5_4_1_table_5_4_1():
    x = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5])
    table = {
        (6, 4): {
            0.0: [0.6234, 0.6022, 0.5399, 0.4408, 0.3117, 0.1614],
            1.0: [0.7114, 0.6871, 0.6161, 0.5030, 0.3557, 0.1841],
        },
        (12, 8): {
            0.0: [0.6245, 0.6032, 0.5409, 0.4416, 0.3123, 0.1616],
            1.0: [0.7122, 0.6880, 0.6168, 0.5036, 0.3561, 0.1843],
        },
    }
    for (nx, ny), rows in table.items():
        problem = rectangular_conduction(nx, ny)
        for y, expected in rows.items():
            points = np.column_stack([x, np.full_like(x, y)])
            assert problem.sample("temperature", points) == pytest.approx(expected, abs=6e-4)


def test_triangles_make_the_dual_mesh_method_equal_the_finite_element_method():
    """With linear triangles and constant coefficients both methods coincide."""
    dual = rectangular_conduction(12, 8, element_type="Tri3", method="dmcdm")
    finite_element = rectangular_conduction(12, 8, element_type="Tri3", method="fem")
    assert dual.values("temperature") == pytest.approx(
        finite_element.values("temperature"), rel=1e-11, abs=1e-12
    )
    # Table 5.4.1, 12 x 8 T column
    x = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5])
    points = np.column_stack([x, np.zeros_like(x)])
    assert dual.sample("temperature", points) == pytest.approx(
        [0.6256, 0.6043, 0.5418, 0.4424, 0.3128, 0.1619], abs=6e-4
    )


def bus_bar_like_problem(conductivity_polynomial=(1.0,)):
    """Example 5.4.2 / 6.3.1: 0.2 m x 0.1 m plate, insulated bottom."""
    mesh = dm.generate_rectangle_mesh(
        x_min=0.0, x_max=0.2, y_min=0.0, y_max=0.1, num_x_elements=8, num_y_elements=8
    )
    problem = dm.Problem(mesh)
    problem.add_variable("temperature", initial_condition=0.0)
    problem.add_kernel(
        "HeatConduction",
        variable="temperature",
        thermal_conductivity=0.2,
        temperature_polynomial=list(conductivity_polynomial),
    )
    problem.add_boundary_condition(
        "DirichletBC", variable="temperature", boundary="left", value=500.0
    )
    problem.add_boundary_condition(
        "DirichletBC", variable="temperature", boundary="right", value=300.0
    )
    problem.add_boundary_condition(
        "DirichletBC",
        variable="temperature",
        boundary="top",
        value=lambda x, y, z, t: 500.0 * (1.0 - 10.0 * x * x),
    )
    return problem


def test_example_5_4_2_linear_conduction():
    problem = bus_bar_like_problem()
    problem.solve()
    x = np.arange(1, 8) * 0.025
    bottom = [482.85, 464.48, 443.88, 420.42, 393.88, 364.48, 332.85]
    # The last entry is printed as 335.34 in Table 5.4.2 of the book; every
    # other entry of that table agrees with this code to the last printed
    # digit, and the finite element value there is 335.49, so 335.34 appears
    # to be a typographical error for 335.54 (see docs/verification.rst).
    middle = [485.54, 469.30, 450.01, 426.98, 400.01, 369.30, 335.54]
    assert problem.sample("temperature", np.column_stack([x, np.zeros_like(x)])) == pytest.approx(
        bottom, abs=6e-3
    )
    assert problem.sample(
        "temperature", np.column_stack([x, np.full_like(x, 0.05)])
    ) == pytest.approx(middle, abs=6e-3)


@pytest.mark.parametrize("nonlinear_solver", ["newton", "picard"])
def test_example_6_3_1_temperature_dependent_conductivity(nonlinear_solver):
    """k = k0 [1 + k1 T] with k1 = 100 K^-1 (Table 6.3.1, DMCDM column)."""
    problem = bus_bar_like_problem(conductivity_polynomial=(1.0, 100.0))
    problem.solve(nonlinear_solver=nonlinear_solver, max_iterations=100)
    x = np.arange(1, 8) * 0.025
    bottom = [485.05, 468.64, 449.85, 427.95, 402.40, 372.81, 338.84]
    middle = [487.18, 472.45, 454.60, 432.88, 406.86, 376.26, 340.80]
    # The book iterates to a relative tolerance of 1e-3, so the published
    # values carry an uncertainty of a few hundredths of a degree.
    assert problem.sample("temperature", np.column_stack([x, np.zeros_like(x)])) == pytest.approx(
        bottom, abs=3e-2
    )
    assert problem.sample(
        "temperature", np.column_stack([x, np.full_like(x, 0.05)])
    ) == pytest.approx(middle, abs=3e-2)


def bus_bar(nx, ny, conductivity_polynomial=(1.0,), conductivity=20.0):
    """Examples 5.4.3 and 6.3.2: bus bar with heat generation and convection."""
    mesh = dm.generate_rectangle_mesh(
        x_min=0.0, x_max=0.1, y_min=0.0, y_max=0.05, num_x_elements=nx, num_y_elements=ny
    )
    problem = dm.Problem(mesh)
    problem.add_variable("temperature")
    problem.add_kernel(
        "HeatConduction",
        variable="temperature",
        thermal_conductivity=conductivity,
        temperature_polynomial=list(conductivity_polynomial),
    )
    problem.add_kernel("HeatSource", variable="temperature", heat_source=1.0e6)
    problem.add_boundary_condition(
        "DirichletBC", variable="temperature", boundary="left", value=40.0
    )
    problem.add_boundary_condition(
        "DirichletBC", variable="temperature", boundary="right", value=10.0
    )
    problem.add_boundary_condition(
        "ConvectiveHeatFluxBC",
        variable="temperature",
        boundary="top",
        heat_transfer_coefficient=75.0,
        ambient_temperature=0.0,
    )
    problem.solve(max_iterations=100)
    return problem


def test_example_5_4_3_bus_bar():
    # Table 5.4.3 is captioned "20 x 10 mesh", but Fig. 5.4.15(b) shows a
    # 10 x 5 primal mesh and the values of Table 5.4.3 coincide with the ones
    # quoted for the 10 x 5 mesh in Example 6.3.2; the 10 x 5 mesh reproduces
    # the published numbers to every printed digit.
    problem = bus_bar(10, 5)
    x = np.arange(1, 10) * 0.01
    bottom = [58.120, 71.386, 79.926, 83.828, 83.142, 77.878, 68.008, 53.469, 34.171]
    top = [55.029, 66.599, 74.132, 77.558, 76.859, 72.010, 62.969, 49.666, 32.014]
    assert problem.sample("temperature", np.column_stack([x, np.zeros_like(x)])) == pytest.approx(
        bottom, abs=5e-3
    )
    assert problem.sample(
        "temperature", np.column_stack([x, np.full_like(x, 0.05)])
    ) == pytest.approx(top, abs=5e-3)


def test_example_6_3_2_nonlinear_bus_bar():
    """k(T) = 20 + 0.2 T on the 10 x 5 mesh (linear and nonlinear solutions)."""
    linear = bus_bar(10, 5)
    nonlinear = bus_bar(10, 5, conductivity_polynomial=(1.0, 0.01))  # 20 (1 + 0.01 T)
    x = np.arange(1, 10) * 0.01
    points = np.column_stack([x, np.full_like(x, 0.05)])
    assert linear.sample("temperature", points) == pytest.approx(
        [55.029, 66.599, 74.132, 77.558, 76.859, 72.010, 62.969, 49.666, 32.014], abs=5e-3
    )
    assert nonlinear.sample("temperature", points) == pytest.approx(
        [50.233, 57.599, 62.088, 63.871, 63.016, 59.464, 53.013, 43.251, 29.405], abs=5e-3
    )


@pytest.mark.parametrize(
    "peclet, num_elements, x_values, expected",
    [
        (
            75,
            50,
            [0.86, 0.88, 0.90, 0.92, 0.94, 0.96, 0.98],
            [1.00000, 0.99999, 0.99989, 0.99917, 0.99418, 0.95960, 0.73470],
        ),
        (
            75,
            100,
            [0.90, 0.92, 0.94, 0.96, 0.98, 0.99],
            [0.99926, 0.99637, 0.98245, 0.91646, 0.62947, 0.29753],
        ),
    ],
)
def test_example_5_4_4_advection_diffusion(peclet, num_elements, x_values, expected):
    """u_x + u_y = (1/Pe) (u_xx + u_yy) with the exact solution on the boundary."""

    def exact(x, y):
        return (
            (1 - math.exp((x - 1) * peclet))
            * (1 - math.exp((y - 1) * peclet))
            / (1 - math.exp(-peclet)) ** 2
        )

    mesh = dm.generate_rectangle_mesh(
        x_min=0.0,
        x_max=1.0,
        y_min=0.0,
        y_max=1.0,
        num_x_elements=num_elements,
        num_y_elements=num_elements,
    )
    problem = dm.Problem(mesh)
    problem.add_variable("u")
    problem.add_kernel("Diffusion", variable="u", diffusivity=1.0 / peclet)
    problem.add_kernel("Advection", variable="u", velocity=[1.0, 1.0], form="non_conservative")
    problem.add_boundary_condition(
        "DirichletBC",
        variable="u",
        boundary=["left", "right", "bottom", "top"],
        value=lambda x, y, z, t: exact(x, y),
    )
    problem.solve(linear_solver="lu")
    points = np.column_stack([x_values, x_values])
    assert problem.sample("u", points) == pytest.approx(expected, abs=2e-5)


# ---------------------------------------------------------------------------
# Chapter 6: nonlinear one-dimensional problems
# ---------------------------------------------------------------------------
class NonlinearFluxBoundaryCondition(dm.PythonBoundaryCondition):
    """n . F = -u^2 at x = 2 (Example 6.2.2), written in Python."""

    def compute_boundary_flux(self, ctx):
        u = ctx.value(self.variable)
        return -(u * u)


@pytest.mark.parametrize(
    "num_elements, expected",
    [
        (4, [1.0000, 0.8001, 0.6669, 0.5719, 0.5006]),
        (8, [0.8889, 0.8001, 0.7274, 0.6669, 0.6156, 0.5717, 0.5336, 0.5003]),
    ],
)
def test_example_6_2_2_cubic_reaction(num_elements, expected):
    """-u'' + 2 u^3 = 0 on (1, 2) with u(1) = 1 and u'(2) + u(2)^2 = 0."""
    problem = line_problem(num_elements, length=1.0, start=1.0)
    problem.add_variable("u", initial_condition=0.5)
    problem.add_kernel("Diffusion", variable="u")
    problem.add_kernel("Reaction", variable="u", coefficient=2.0, exponent=3.0)
    problem.add_boundary_condition("DirichletBC", variable="u", boundary="left", value=1.0)
    problem.add_boundary_condition(NonlinearFluxBoundaryCondition(variable="u", boundary="right"))
    problem.solve(max_iterations=200)
    values = problem.values("u")
    # The published lists give the values at the nodes after the first one.
    assert values[1:] == pytest.approx(expected[-(num_elements):], abs=1e-3)


@pytest.mark.parametrize(
    "num_elements, expected",
    [
        (4, [453.94, 405.54, 354.40]),
        (8, [477.24, 453.94, 430.06, 405.54, 380.35, 354.40, 327.65]),
    ],
)
def test_example_6_2_3_temperature_dependent_bar(num_elements, expected):
    """k(T) = 0.2 (1 + 0.002 T) between 500 K and 300 K (Table 6.2.2)."""
    problem = line_problem(num_elements, length=0.18)
    problem.add_variable("temperature", initial_condition=400.0)
    problem.add_kernel(
        "HeatConduction",
        variable="temperature",
        thermal_conductivity=0.2,
        temperature_polynomial=[1.0, 2.0e-3],
    )
    problem.add_boundary_condition(
        "DirichletBC", variable="temperature", boundary="left", value=500.0
    )
    problem.add_boundary_condition(
        "DirichletBC", variable="temperature", boundary="right", value=300.0
    )
    problem.solve()
    assert problem.values("temperature")[1:-1] == pytest.approx(expected, abs=2e-2)


class LargeDeformationBar(dm.PythonKernel):
    """Bar with a(u') = EA (1 + 1.5 u' + 0.5 u'^2) (Example 6.2.4)."""

    def compute_flux(self, ctx):
        strain = ctx.coefficient_gradient(self.variable)[0]
        stiffness = self.axial_stiffness * (1.0 + 1.5 * strain + 0.5 * strain * strain)
        return [stiffness * ctx.gradient(self.variable)[0]]


@pytest.mark.parametrize("load, expected_tip", [(0.2, 0.1597), (1.0, 0.5213), (5.0, 1.3087)])
def test_example_6_2_4_large_deformation_bar(load, expected_tip):
    problem = line_problem(8, length=1.0)
    problem.add_variable("displacement")
    problem.add_kernel(LargeDeformationBar(variable="displacement", axial_stiffness=1.0))
    problem.add_boundary_condition(
        "DirichletBC", variable="displacement", boundary="left", value=0.0
    )
    problem.add_boundary_condition(
        "NeumannBC", variable="displacement", boundary="right", flux=load
    )
    problem.solve(load_factors=list(np.linspace(0.2, 1.0, 5)), max_iterations=100)
    assert problem.values("displacement")[-1] == pytest.approx(expected_tip, abs=1e-3)
