# SPDX-License-Identifier: LGPL-2.1-or-later
"""Generalized plane strain: a long body whose uniform axial strain is one unknown of its
section, set by the axial force it carries."""

from __future__ import annotations

import dualmesh
import numpy
import pytest

E, nu = 200e9, 0.3


def square(**solid):
    mesh = dualmesh.generate_rectangle_mesh(0, 1, 0, 1, 3, 3)
    problem = dualmesh.Problem(mesh)
    body = problem.add_physics(
        "solid_mechanics",
        formulation="generalized_plane_strain",
        youngs_modulus=E,
        poissons_ratio=nu,
        **solid,
    )
    body.add_boundary_condition("disp_x = 0", boundary="left")
    body.add_boundary_condition("disp_y = 0", boundary="bottom")
    return problem, body


@pytest.mark.parametrize("force", [0.0, 5e9])
def test_the_axial_strain_carries_the_axial_force(force):
    """A traction of 100 MPa along x and the axial force F on the unit section: sigma_zz = F,
    so eps_zz = (F - nu 100 MPa)/E and eps_xx = (100 MPa - nu F)/E, and strain_zz is uniform."""
    problem, body = square(axial_force=force)
    body.add_boundary_condition("traction = [100e6, 0]", boundary="right")
    problem.solve(report="none")
    axial = problem.values("strain_zz")
    assert numpy.ptp(axial) == 0.0
    assert axial[0] == pytest.approx((force - nu * 100e6) / E, rel=1e-10)
    strain = problem.sample("disp_x", [[1 - 1e-12, 0.5]])[0]
    assert strain == pytest.approx((100e6 - nu * force) / E, rel=1e-10)


def test_a_free_thermal_expansion_is_free_of_stress():
    """Heated uniformly with no load and no axial force: eps_zz = alpha dT, and no stress."""
    mesh = dualmesh.generate_rectangle_mesh(0, 1, 0, 1, 3, 3)
    problem = dualmesh.Problem(mesh)
    heat = problem.add_physics("heat_conduction", thermal_conductivity=1.0)
    body = problem.add_physics(
        "solid_mechanics",
        formulation="generalized_plane_strain",
        youngs_modulus=E,
        poissons_ratio=nu,
    )
    problem.add_coupling(
        "thermal_expansion",
        heat_conduction=heat,
        solid_mechanics=body,
        linear_expansion_coefficient=1e-5,
        stress_free_temperature=300.0,
    )
    heat.add_boundary_condition("temperature = 400", boundary=["left", "right"])
    body.add_boundary_condition("disp_x = 0", boundary="left")
    body.add_boundary_condition("disp_y = 0", boundary="bottom")
    problem.solve(report="none")
    assert problem.values("strain_zz")[0] == pytest.approx(1e-3, rel=1e-10)
    assert numpy.abs(problem.property_at_centroids("stress")).max() < 1e-6 * E * 1e-3


def test_an_open_ended_tube_under_pressure_is_lames():
    """A tube a = 1, b = 2 under the inner pressure p with free ends (no axial force): A = p a^2
    / (b^2 - a^2), eps_zz = -2 nu A / E, and the radial displacement of the bore is
    ((1 - nu) A a + (1 + nu) B / a) / E with B = A b^2, to second order in the mesh."""
    a, b, p = 1.0, 2.0, 100e6
    A = p * a**2 / (b**2 - a**2)
    exact_bore = ((1 - nu) * A * a + (1 + nu) * A * b**2 / a) / E
    errors = []
    for n in (8, 16):
        mesh = dualmesh.generate_annulus_mesh(a, b, n, n, element_type="Quad9")
        problem = dualmesh.Problem(mesh)
        body = problem.add_physics(
            "solid_mechanics",
            formulation="generalized_plane_strain",
            youngs_modulus=E,
            poissons_ratio=nu,
        )
        body.add_boundary_condition("symmetry", boundary="start")
        body.add_boundary_condition("symmetry", boundary="end")
        body.add_boundary_condition(f"pressure = {p}", boundary="inner")
        problem.solve(report="none")
        assert problem.values("strain_zz")[0] == pytest.approx(-2 * nu * A / E, rel=1e-4)
        bore = problem.sample("disp_x", [[a, 0.0]])[0]
        errors.append(abs(bore - exact_bore))
    assert errors[1] < errors[0] and errors[1] < 1e-4 * exact_bore
