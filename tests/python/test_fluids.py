# SPDX-License-Identifier: LGPL-2.1-or-later
"""Verification of the penalty flow module against Chapter 9 of Reddy's book."""

from __future__ import annotations

import dualmesh as dm
import numpy as np
import pytest

PENALTY = 1.0e8


def flow_problem(mesh, viscosity=1.0, density=0.0, penalty=PENALTY, method="dmcdm"):
    """Penalty formulation of a two-dimensional incompressible flow."""
    problem = dm.Problem(mesh, method=method)
    problem.add_variable("u")
    problem.add_variable("v")
    velocities = ["u", "v"]
    for component, variable in enumerate(velocities):
        problem.add_kernel(
            "ViscousStress",
            f"viscous_{variable}",
            variable=variable,
            component=component,
            velocities=velocities,
            dynamic_viscosity=viscosity,
        )
        problem.add_kernel(
            "PenaltyIncompressibility",
            f"penalty_{variable}",
            variable=variable,
            component=component,
            velocities=velocities,
            penalty_parameter=penalty,
        )
        if density > 0.0:
            problem.add_kernel(
                "ConvectiveInertia",
                f"inertia_{variable}",
                variable=variable,
                component=component,
                velocities=velocities,
                density=density,
            )
    problem.add_material(
        "PenaltyPressure", "pressure", velocities=velocities, penalty_parameter=penalty
    )
    return problem


def test_example_9_8_3_squeezed_flow():
    """Table 9.8.2 (DMCDM column): creeping flow squeezed between two plates."""
    half_length, half_gap, plate_velocity = 6.0, 2.0, 1.0
    dx = [0.5] * 8 + [0.25] * 4 + [0.125] * 8
    dy = [0.125] * 16
    mesh = dm.generate_rectangle_mesh(
        x_coordinates=dm.meshing.coordinates_from_spacings(0.0, dx),
        y_coordinates=dm.meshing.coordinates_from_spacings(0.0, dy),
    )
    problem = flow_problem(mesh)
    # Symmetry planes: v = 0 on y = 0 and u = 0 on x = 0.
    problem.add_boundary_condition("DirichletBC", variable="v", boundary="bottom", value=0.0)
    problem.add_boundary_condition("DirichletBC", variable="u", boundary="left", value=0.0)
    # The plate at y = b moves down with velocity V0 and does not slip.
    problem.add_boundary_condition("DirichletBC", variable="u", boundary="top", value=0.0)
    problem.add_boundary_condition(
        "DirichletBC", variable="v", boundary="top", value=-plate_velocity
    )
    problem.solve(linear_solver="lu")

    y = np.arange(0.0, 2.0, 0.125)
    reference = {
        4.0: [
            3.0302,
            3.0180,
            2.9805,
            2.9185,
            2.8316,
            2.7208,
            2.5854,
            2.4269,
            2.2446,
            2.0401,
            1.8128,
            1.5643,
            1.2937,
            1.0024,
            0.6896,
            0.3557,
        ],
        6.0: [
            4.2150,
            4.2019,
            4.1622,
            4.0961,
            4.0030,
            3.8830,
            3.7348,
            3.5589,
            3.3531,
            3.1181,
            2.8497,
            2.5506,
            2.2088,
            1.8382,
            1.3828,
            0.9513,
        ],
    }
    for x, expected in reference.items():
        points = np.column_stack([np.full_like(y, x), y])
        assert problem.sample("u", points) == pytest.approx(expected, abs=2e-3)

    # The pressure recovered from the penalty relation follows Nadai's
    # approximate solution P = 3 mu V0 (a^2 + y^2 - x^2) / (2 b^3) in the
    # interior, away from the corner singularities.
    pressure = problem.property_at_centroids("pressure").ravel()
    centroids = np.array(
        [problem.mesh.element_centroid(e) for e in range(problem.mesh.num_elements)]
    )
    analytical = (
        3.0
        * plate_velocity
        * (half_length**2 + centroids[:, 1] ** 2 - centroids[:, 0] ** 2)
        / (2.0 * half_gap**3)
    )
    interior = (centroids[:, 0] < 4.0) & (centroids[:, 1] < 1.0)
    assert np.corrcoef(pressure[interior], analytical[interior])[0, 1] > 0.999
    # and the magnitude is right to within a few percent in the interior
    assert np.polyfit(analytical[interior], pressure[interior], 1)[0] == pytest.approx(
        1.0, rel=0.05
    )


def cavity_mesh():
    """The 16 x 20 graded mesh of Example 9.8.4 (refined towards the lid)."""
    dx = [0.0625] * 16
    dy = [0.0625] * 12 + [0.03125] * 8
    return dm.generate_rectangle_mesh(
        x_coordinates=dm.meshing.coordinates_from_spacings(0.0, dx),
        y_coordinates=dm.meshing.coordinates_from_spacings(0.0, dy),
    )


def lid_driven_cavity(reynolds_number, method="dmcdm", ramp_lid=False, **solver_options):
    problem = flow_problem(
        cavity_mesh(), viscosity=1.0, density=float(reynolds_number), method=method
    )
    for variable in ("u", "v"):
        problem.add_boundary_condition(
            "DirichletBC",
            f"walls_{variable}",
            variable=variable,
            boundary=["left", "right", "bottom"],
            value=0.0,
        )
    problem.add_boundary_condition(
        "DirichletBC",
        "lid_u",
        variable="u",
        boundary="top",
        value=1.0,
        scale_with_load=ramp_lid,
    )
    problem.add_boundary_condition("DirichletBC", "lid_v", variable="v", boundary="top", value=0.0)
    problem.solve(**solver_options)
    return problem


CAVITY_Y = [
    0.0625,
    0.1250,
    0.1875,
    0.2500,
    0.3125,
    0.3750,
    0.4375,
    0.5000,
    0.5625,
    0.6250,
    0.6875,
    0.7500,
    0.7813,
    0.8125,
    0.8438,
    0.8750,
    0.9063,
    0.9375,
    0.9688,
]
CAVITY_STOKES = [
    -0.0369,
    -0.0663,
    -0.0919,
    -0.1158,
    -0.1386,
    -0.1598,
    -0.1775,
    -0.1881,
    -0.1857,
    -0.1619,
    -0.1057,
    -0.0046,
    0.0673,
    0.1542,
    0.2584,
    0.3772,
    0.5149,
    0.6634,
    0.8279,
]
CAVITY_RE1000 = [
    -0.1430,
    -0.2177,
    -0.2502,
    -0.2301,
    -0.1776,
    -0.1255,
    -0.0803,
    -0.0365,
    0.0091,
    0.0570,
    0.1097,
    0.1612,
    0.1861,
    0.2084,
    0.2273,
    0.2383,
    0.2496,
    0.2937,
    0.5245,
]


def test_example_9_8_4_cavity_stokes():
    """Table 9.8.3, DMCDM column at Re = 0 (Stokes flow)."""
    problem = lid_driven_cavity(0)
    points = np.column_stack([np.full(len(CAVITY_Y), 0.5), CAVITY_Y])
    assert problem.sample("u", points) == pytest.approx(CAVITY_STOKES, abs=2e-3)


def test_example_9_8_4_cavity_reynolds_1000():
    """Table 9.8.3, DMCDM column at Re = 1000 (Navier-Stokes).

    Newton's method started from rest does not converge at this Reynolds
    number, so the lid velocity is ramped up in load steps (the converged
    solution of one step is the initial guess of the next).
    """
    problem = lid_driven_cavity(
        1000,
        ramp_lid=True,
        nonlinear_solver="newton",
        max_iterations=60,
        load_factors=[0.1, 0.25, 0.5, 0.75, 1.0],
    )
    points = np.column_stack([np.full(len(CAVITY_Y), 0.5), CAVITY_Y])
    assert problem.sample("u", points) == pytest.approx(CAVITY_RE1000, abs=2e-3)


def test_example_9_8_4_cavity_reynolds_1000_by_direct_iteration():
    """The same solution obtained by direct (Picard) iteration with relaxation."""
    problem = lid_driven_cavity(
        1000,
        nonlinear_solver="picard",
        relaxation=0.5,
        max_iterations=400,
        step_tolerance=1e-9,
    )
    points = np.column_stack([np.full(len(CAVITY_Y), 0.5), CAVITY_Y])
    assert problem.sample("u", points) == pytest.approx(CAVITY_RE1000, abs=2e-3)


def test_cavity_picard_matches_newton():
    """Direct (Picard) iteration and Newton's method reach the same solution."""
    newton = lid_driven_cavity(400, nonlinear_solver="newton")
    picard = lid_driven_cavity(
        400, nonlinear_solver="picard", max_iterations=200, step_tolerance=1e-10
    )
    assert picard.values("u") == pytest.approx(newton.values("u"), abs=1e-6)


def test_penalty_flow_is_almost_divergence_free():
    """The penalty formulation keeps the divergence small: |div v| ~ p / gamma."""
    problem = lid_driven_cavity(0)
    du = problem.gradient_at_centroids("u")
    dv = problem.gradient_at_centroids("v")
    divergence = du[:, 0] + dv[:, 1]
    assert np.max(np.abs(divergence)) < 1e-5
