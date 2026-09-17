# SPDX-License-Identifier: LGPL-2.1-or-later
"""Verification of the plane elasticity module against Chapter 9 of Reddy's book."""

from __future__ import annotations

import math

import dualmesh as dm
import numpy as np
import pytest


def plane_elasticity_problem(
    mesh,
    youngs_modulus,
    poissons_ratio,
    thickness=1.0,
    formulation="plane_stress",
    method="dmcdm",
    coordinates="cartesian",
):
    """Set up a two-dimensional elasticity problem with displacements (u, v)."""
    problem = dm.Problem(mesh, method=method, coordinates=coordinates)
    problem.add_variable("u")
    problem.add_variable("v")
    problem.add_material(
        "LinearElasticStress",
        "elasticity",
        displacements=["u", "v"],
        formulation=formulation,
        youngs_modulus=youngs_modulus,
        poissons_ratio=poissons_ratio,
    )
    for component, variable in enumerate(("u", "v")):
        problem.add_kernel(
            "StressDivergence",
            f"equilibrium_{variable}",
            variable=variable,
            component=component,
            thickness=thickness,
        )
    return problem


def test_example_9_8_1_uniform_edge_stress():
    """A quadrant of a plate under uniform edge stress: u(a) = t0 a / E exactly."""
    youngs_modulus, poissons_ratio = 207.0e9, 0.25
    edge_stress, side, thickness = 1.0e6, 1.0, 0.01
    mesh = dm.generate_rectangle_mesh(
        x_min=0.0, x_max=side, y_min=0.0, y_max=side, num_x_elements=4, num_y_elements=4
    )
    problem = plane_elasticity_problem(mesh, youngs_modulus, poissons_ratio, thickness=thickness)
    problem.add_boundary_condition("DirichletBC", variable="u", boundary="left", value=0.0)
    problem.add_boundary_condition("DirichletBC", variable="v", boundary="bottom", value=0.0)
    problem.add_boundary_condition(
        "TractionBC",
        variable="u",
        boundary="right",
        traction=edge_stress,
        thickness=thickness,
    )
    problem.solve()

    expected = edge_stress * side / youngs_modulus  # 0.4831e-5 m
    right_edge = problem.mesh.boundary_nodes("right")
    assert problem.values_at_nodes("u", right_edge) == pytest.approx(expected, rel=1e-12)
    assert expected == pytest.approx(0.4831e-5, rel=1e-3)
    # The stress field is exactly uniform: sigma_xx = t0, sigma_yy = sigma_xy = 0.
    stress = problem.property_at_centroids("stress")
    assert stress[:, 0] == pytest.approx(edge_stress, rel=1e-9)
    assert stress[:, 1] == pytest.approx(0.0, abs=1e-6 * edge_stress)
    assert stress[:, 5] == pytest.approx(0.0, abs=1e-6 * edge_stress)


@pytest.mark.parametrize(
    "num_elements, displacement_u, displacement_v, stress_xx",
    [
        (1, 10.784, 1.961, 277.78),
        (2, 11.058, 1.946, 277.78),
        (4, 11.146, 1.992, 287.27),
        (8, 11.161, 1.995, 306.57),
    ],
)
def test_example_9_8_2_plate_with_edge_load(
    num_elements, displacement_u, displacement_v, stress_xx
):
    """Table 9.8.1 (DMCDM rows): cantilevered plate under a uniform edge load."""
    width, height, thickness = 120.0, 160.0, 0.036
    youngs_modulus, poissons_ratio = 30.0e6, 0.25
    edge_load = 10.0  # lb/in, i.e. force per unit length of the edge
    mesh = dm.generate_rectangle_mesh(
        x_min=0.0,
        x_max=width,
        y_min=0.0,
        y_max=height,
        num_x_elements=num_elements,
        num_y_elements=num_elements,
    )
    problem = plane_elasticity_problem(mesh, youngs_modulus, poissons_ratio, thickness=thickness)
    for variable in ("u", "v"):
        problem.add_boundary_condition("DirichletBC", variable=variable, boundary="left", value=0.0)
    problem.add_boundary_condition(
        "TractionBC",
        variable="u",
        boundary="right",
        traction=edge_load / thickness,
        thickness=thickness,
    )
    problem.solve()

    corner = problem.node_at((width, 0.0), tolerance=1e-9)
    assert problem.values("u")[corner] * 1e4 == pytest.approx(displacement_u, abs=5e-3)
    assert problem.values("v")[corner] * 1e4 == pytest.approx(displacement_v, abs=5e-3)
    # Stress at the centre of the element closest to (x, y) = (width/(2 n), height/(2 n)).
    stress = problem.property_at_centroids("stress")
    centroids = np.array(
        [problem.mesh.element_centroid(e) for e in range(problem.mesh.num_elements)]
    )
    target = np.array([width / (2 * num_elements), height / (2 * num_elements)])
    element = int(np.argmin(np.linalg.norm(centroids[:, :2] - target, axis=1)))
    assert stress[element, 0] == pytest.approx(stress_xx, rel=2e-4)


def pressurized_cylinder_exact(radius, inner, outer, pressure, youngs_modulus, poissons_ratio):
    """Plane-strain thick cylinder under internal pressure (Reddy, Eq. 9.9.24)."""
    factor = pressure * inner**2 * (1 + poissons_ratio) / (youngs_modulus * (outer**2 - inner**2))
    return factor * ((1 - 2 * poissons_ratio) * radius + outer**2 / radius)


def pressurized_cylinder(element_type, radial_elements, angular_elements, method="dmcdm"):
    """Quarter of the thick cylinder of Section 9.9.4.1 under internal pressure."""
    inner, outer = 0.05, 0.1
    pressure, youngs_modulus, poissons_ratio = 120.0e6, 200.0e9, 0.3
    mesh = dm.generate_annulus_mesh(
        inner_radius=inner,
        outer_radius=outer,
        num_radial_elements=radial_elements,
        num_angular_elements=angular_elements,
        start_angle=0.0,
        end_angle=90.0,
        element_type=element_type,
    )
    problem = plane_elasticity_problem(
        mesh, youngs_modulus, poissons_ratio, formulation="plane_strain", method=method
    )
    # Symmetry: no normal displacement on the two straight edges.
    problem.add_boundary_condition("DirichletBC", variable="v", boundary="start", value=0.0)
    problem.add_boundary_condition("DirichletBC", variable="u", boundary="end", value=0.0)
    for component, variable in enumerate(("u", "v")):
        problem.add_boundary_condition(
            "PressureBC",
            f"internal_pressure_{variable}",
            variable=variable,
            boundary="inner",
            component=component,
            pressure=pressure,
        )
    problem.solve()
    radial_displacement = problem.values("u") * 1e3  # metres -> millimetres
    return problem, radial_displacement


@pytest.mark.parametrize(
    "radial_elements, angular_elements, inner_u, outer_u",
    [
        # Table 9.9.1, DMCDM columns for quadrilaterals (in millimetres).  The
        # book gives the number of subdivisions across the wall and the total
        # element count (40 and 152 here), from which the number of
        # circumferential subdivisions follows.
        (4, 10, 0.05611, 0.03603),
        (8, 19, 0.05692, 0.03630),
        (12, 29, 0.05708, 0.03636),
    ],
)
def test_example_9_9_4_1_pressurized_cylinder_quadrilaterals(
    radial_elements, angular_elements, inner_u, outer_u
):
    inner, outer = 0.05, 0.1
    problem, displacement = pressurized_cylinder("Quad4", radial_elements, angular_elements)
    computed_inner = displacement[problem.node_at((inner, 0.0))]
    computed_outer = displacement[problem.node_at((outer, 0.0))]
    # The book does not state how the circumference is discretized beyond the
    # total element count, so the agreement is to about half a percent.
    assert computed_inner == pytest.approx(inner_u, abs=5e-4)
    assert computed_outer == pytest.approx(outer_u, abs=5e-4)


def test_pressurized_cylinder_converges_to_the_analytical_solution():
    inner, outer = 0.05, 0.1
    pressure, youngs_modulus, poissons_ratio = 120.0e6, 200.0e9, 0.3
    exact_inner = (
        pressurized_cylinder_exact(inner, inner, outer, pressure, youngs_modulus, poissons_ratio)
        * 1e3
    )
    exact_outer = (
        pressurized_cylinder_exact(outer, inner, outer, pressure, youngs_modulus, poissons_ratio)
        * 1e3
    )
    assert exact_inner == pytest.approx(0.05720, abs=1e-5)  # Table 9.9.1
    assert exact_outer == pytest.approx(0.03640, abs=1e-5)
    errors = []
    for element_type in ("Quad4", "Tri3"):
        previous = None
        for radial_elements, angular_elements in [(4, 10), (8, 19), (12, 29), (16, 38)]:
            problem, displacement = pressurized_cylinder(
                element_type, radial_elements, angular_elements
            )
            error = abs(displacement[problem.node_at((inner, 0.0))] - exact_inner)
            if previous is not None:
                assert error < previous  # monotone convergence
            previous = error
        errors.append(previous / exact_inner)
    # On the finest mesh both element types are within one percent.
    assert max(errors) < 0.01


def test_triangles_reproduce_the_finite_element_method_in_elasticity():
    """For constant-coefficient elasticity on linear triangles DMCDM == FEM."""
    mesh = dm.generate_annulus_mesh(
        inner_radius=0.05,
        outer_radius=0.1,
        num_radial_elements=4,
        num_angular_elements=10,
        element_type="Tri3",
    )
    solutions = []
    for method in ("dmcdm", "fem"):
        problem = plane_elasticity_problem(
            mesh, 200.0e9, 0.3, formulation="plane_strain", method=method
        )
        problem.add_boundary_condition("DirichletBC", variable="v", boundary="start", value=0.0)
        problem.add_boundary_condition("DirichletBC", variable="u", boundary="end", value=0.0)
        for component, variable in enumerate(("u", "v")):
            problem.add_boundary_condition(
                "PressureBC",
                f"pressure_{variable}",
                variable=variable,
                boundary="inner",
                component=component,
                pressure=120.0e6,
            )
        problem.solve()
        solutions.append(problem.values("u"))
    scale = np.max(np.abs(solutions[0]))
    assert solutions[0] == pytest.approx(solutions[1], rel=1e-9, abs=1e-10 * scale)


def test_plate_with_a_circular_hole_stress_concentration():
    """A large plate with a hole under uniaxial tension: sigma_max / sigma = 3."""
    hole_radius, outer_radius = 1.0, 20.0
    applied_stress = 1.0
    mesh = dm.generate_annulus_mesh(
        inner_radius=hole_radius,
        outer_radius=outer_radius,
        num_radial_elements=30,
        num_angular_elements=30,
        radial_bias=1.15,
    )
    problem = plane_elasticity_problem(mesh, 1.0, 0.3)
    problem.add_boundary_condition("DirichletBC", variable="v", boundary="start", value=0.0)
    problem.add_boundary_condition("DirichletBC", variable="u", boundary="end", value=0.0)
    # Remote uniaxial tension in x: t = (sigma n_x, 0) on the outer arc.
    problem.add_boundary_condition(
        "TractionBC",
        "remote_tension",
        variable="u",
        boundary="outer",
        traction=lambda x, y, z, t: applied_stress * x / math.hypot(x, y),
    )
    problem.add_boundary_condition(
        "TractionBC",
        "remote_tension_y",
        variable="v",
        boundary="outer",
        traction=0.0,
    )
    problem.solve()

    # The hoop stress at the top of the hole (x = 0, y = a) is 3 sigma.
    stress = problem.property_at_centroids("stress")
    centroids = np.array(
        [problem.mesh.element_centroid(e) for e in range(problem.mesh.num_elements)]
    )
    radius = np.hypot(centroids[:, 0], centroids[:, 1])
    near_hole = radius < hole_radius * 1.05
    top = near_hole & (centroids[:, 1] > centroids[:, 0])
    assert np.max(stress[top, 0]) == pytest.approx(3.0 * applied_stress, rel=0.05)


def test_axisymmetric_pressurized_cylinder_matches_plane_strain():
    """The same cylinder solved as a one-dimensional axisymmetric problem."""
    inner, outer = 0.05, 0.1
    pressure, youngs_modulus, poissons_ratio = 120.0e6, 200.0e9, 0.3
    mesh = dm.generate_rectangle_mesh(
        x_min=inner,
        x_max=outer,
        y_min=0.0,
        y_max=0.01,
        num_x_elements=40,
        num_y_elements=1,
    )
    problem = dm.Problem(mesh, coordinates="axisymmetric")
    problem.add_variable("u")
    problem.add_variable("v")
    problem.add_material(
        "LinearElasticStress",
        "elasticity",
        displacements=["u", "v"],
        formulation="axisymmetric",
        youngs_modulus=youngs_modulus,
        poissons_ratio=poissons_ratio,
    )
    for component, variable in enumerate(("u", "v")):
        problem.add_kernel(
            "StressDivergence", f"equilibrium_{variable}", variable=variable, component=component
        )
    # Plane strain: no axial displacement (held between rigid walls).
    problem.add_boundary_condition(
        "DirichletBC", variable="v", boundary=["bottom", "top"], value=0.0
    )
    problem.add_boundary_condition(
        "PressureBC", "pressure_u", variable="u", boundary="left", component=0, pressure=pressure
    )
    problem.add_boundary_condition(
        "PressureBC", "pressure_v", variable="v", boundary="left", component=1, pressure=pressure
    )
    problem.solve()

    radii = np.linspace(inner, outer, 9)
    exact = [
        pressurized_cylinder_exact(r, inner, outer, pressure, youngs_modulus, poissons_ratio)
        for r in radii
    ]
    computed = problem.sample("u", np.column_stack([radii, np.full_like(radii, 0.005)]))
    assert computed == pytest.approx(exact, rel=2e-3)
