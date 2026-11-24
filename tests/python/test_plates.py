# SPDX-License-Identifier: LGPL-2.1-or-later
"""Verification of the plate models against Chapters 8 and 10 of Reddy's book.

Chapter 8: axisymmetric circular plates, radius a = 10 in, thickness
h = 0.1 in, E1 = 30e6 psi, E2 = 3e6 psi, nu = 0.3, K_s = 5/6, q0 = 0.5 lb/in.

Chapter 10: rectangular plates, a = b = 1 in, h = 0.01 in, E1 = 3e7 psi,
E2 = 3e6 psi, nu = 0.3, q = 1 lb/in^2, modelled on one quarter of the plate.
"""

from __future__ import annotations

import dualmesh
import pytest

# ---------------------------------------------------------------------------
# Chapter 8: axisymmetric circular plates
# ---------------------------------------------------------------------------
circular_radius = 10.0
circular_thickness = 0.1
circular_load = 0.5
modulus_top = 30.0e6
modulus_bottom = 3.0e6
Poisson_ratio = 0.3


def circular_plate_stiffness(power_law_index):
    return dualmesh.fgm.beam_stiffness(
        modulus_top=modulus_top,
        modulus_bottom=modulus_bottom,
        power_law_index=power_law_index,
        height=circular_thickness,
        width=1.0,
        poissons_ratio=Poisson_ratio,
        plate=True,
    )


def circular_plate(num_elements, power_law_index=0.0, support="hinged"):
    """Axisymmetric circular plate, first-order shear deformation theory."""
    s = circular_plate_stiffness(power_law_index)
    mesh = dualmesh.generate_line_mesh(start=0.0, end=circular_radius, num_elements=num_elements)
    problem = dualmesh.Problem(mesh, method="DMCDM", coordinates="axisymmetric")
    for name in ("radial_displacement", "deflection", "rotation"):
        problem.add_variable(name)
    common = dict(
        radial_displacement="radial_displacement",
        transverse_displacement="deflection",
        rotation="rotation",
        extensional_stiffness=s.extensional,
        coupling_stiffness=s.coupling,
        bending_stiffness=s.bending,
        shear_stiffness=s.shear,
        poissons_ratio=Poisson_ratio,
        transverse_load=circular_load,
    )
    for variable in ("radial_displacement", "deflection", "rotation"):
        problem.add_kernel(
            "circular_plate_first_order",
            variable=variable,
            shear_treatment="exclude",
            **common,
        )
        problem.add_kernel(
            "circular_plate_first_order",
            variable=variable,
            shear_treatment="only",
            quadrature="midpoint",
            reduced_integration=True,
            **common,
        )
    # Symmetry at r = 0
    problem.add_boundary_condition(
        "Dirichlet",
        "left",
        variable="radial_displacement",
        value=0.0,
    )
    problem.add_boundary_condition(
        "Dirichlet",
        "left",
        variable="rotation",
        value=0.0,
    )
    # Edge conditions at r = a
    problem.add_boundary_condition(
        "Dirichlet",
        "right",
        variable="radial_displacement",
        value=0.0,
    )
    problem.add_boundary_condition("Dirichlet", "right", variable="deflection", value=0.0)
    if support == "clamped":
        problem.add_boundary_condition(
            "Dirichlet",
            "right",
            variable="rotation",
            value=0.0,
        )
    problem.solve()
    return problem


def hinged_exact_deflection(power_law_index=0.0):
    """Reddy, Eq. (8.6.4b): centre deflection of a hinged FGM circular plate."""
    s = circular_plate_stiffness(power_law_index)
    reduced = s.bending * s.extensional - s.coupling**2
    a, q = circular_radius, circular_load
    bending_part = (
        s.extensional * q * a**4 / (64 * reduced) * (5 + Poisson_ratio) / (1 + Poisson_ratio)
    )
    shear_part = q * a**2 / (4 * s.shear)
    return bending_part + shear_part


def clamped_exact_deflection(power_law_index=0.0):
    """Reddy, Eq. (8.6.6b): centre deflection of a clamped FGM circular plate."""
    s = circular_plate_stiffness(power_law_index)
    reduced = s.bending * s.extensional - s.coupling**2
    a, q = circular_radius, circular_load
    return s.extensional * q * a**4 / (64 * reduced) + q * a**2 / (4 * s.shear)


def test_table_8_6_1_hinged_homogeneous_circular_plate():
    """Table 8.6.1, DM-FS(D) column: w(0) for 2, 4, 8, 16, 32 elements."""
    expected = [0.1052, 0.1134, 0.1153, 0.1158, 0.1159]
    for num_elements, reference in zip((2, 4, 8, 16, 32), expected):
        problem = circular_plate(num_elements)
        assert problem.values("deflection")[0] == pytest.approx(reference, abs=1.1e-4)
    assert hinged_exact_deflection() == pytest.approx(0.1159, abs=1.1e-4)


@pytest.mark.parametrize(
    "power_law_index, expected",
    [
        # Table 8.6.2, DM-FP(D) column (32 elements)
        (0.0, 0.0284),
        (1.0, 0.0665),
        (2.0, 0.0976),
        (3.0, 0.1152),
        (4.0, 0.1244),
        (5.0, 0.1296),
        (7.5, 0.1370),
        (10.0, 0.1425),
        (15.0, 0.1527),
        (20.0, 0.1622),
    ],
)
def test_table_8_6_2_clamped_functionally_graded_circular_plate(power_law_index, expected):
    problem = circular_plate(32, power_law_index=power_law_index, support="clamped")
    centre = problem.values("deflection")[0]
    assert centre == pytest.approx(expected, abs=1.1e-4)
    assert centre == pytest.approx(clamped_exact_deflection(power_law_index), rel=5e-3)


# ---------------------------------------------------------------------------
# Chapter 10: rectangular plates
# ---------------------------------------------------------------------------
plate_side = 1.0
plate_thickness = 0.01
plate_load = 1.0


def rectangular_plate_stiffness(power_law_index):
    return dualmesh.fgm.beam_stiffness(
        modulus_top=3.0e7,
        modulus_bottom=3.0e6,
        power_law_index=power_law_index,
        height=plate_thickness,
        width=1.0,
        poissons_ratio=Poisson_ratio,
        plate=True,
    )


def rectangular_plate(
    num_elements,
    power_law_index=0.0,
    support="clamped",
    method="dmcdm",
    von_karman=False,
    load=plate_load,
):
    """Quarter model of a rectangular plate, first-order shear deformation theory."""
    s = rectangular_plate_stiffness(power_law_index)
    mesh = dualmesh.generate_rectangle_mesh(
        x_min=0.0,
        x_max=plate_side / 2,
        y_min=0.0,
        y_max=plate_side / 2,
        num_x_elements=num_elements,
        num_y_elements=num_elements,
    )
    problem = dualmesh.Problem(mesh, method=method)
    variables = ["u", "v", "w", "rotation_x", "rotation_y"]
    for name in variables:
        problem.add_variable(name)
    common = dict(
        in_plane_displacements=["u", "v"],
        transverse_displacement="w",
        rotations=["rotation_x", "rotation_y"],
        extensional_stiffness=s.extensional,
        coupling_stiffness=s.coupling,
        bending_stiffness=s.bending,
        shear_stiffness=s.shear,
        poissons_ratio=Poisson_ratio,
        transverse_load=load,
        von_karman=von_karman,
    )
    for variable in variables:
        problem.add_kernel(
            "plate_first_order",
            variable=variable,
            shear_treatment="exclude",
            **common,
        )
        problem.add_kernel(
            "plate_first_order",
            variable=variable,
            shear_treatment="only",
            quadrature="midpoint",
            reduced_integration=True,
            **common,
        )
    # Symmetry: u = phi_x = 0 on x = 0 and v = phi_y = 0 on y = 0
    problem.add_boundary_condition("Dirichlet", "left", variable="u", value=0.0)
    problem.add_boundary_condition("Dirichlet", "left", variable="rotation_x", value=0.0)
    problem.add_boundary_condition("Dirichlet", "bottom", variable="v", value=0.0)
    problem.add_boundary_condition(
        "Dirichlet",
        "bottom",
        variable="rotation_y",
        value=0.0,
    )
    if support == "clamped":
        for name in variables:
            problem.add_boundary_condition(
                "Dirichlet",
                ["right", "top"],
                variable=name,
                value=0.0,
            )
    else:  # Simply supported (SS-1)
        problem.add_boundary_condition("Dirichlet", "right", variable="v", value=0.0)
        problem.add_boundary_condition("Dirichlet", "right", variable="w", value=0.0)
        problem.add_boundary_condition(
            "Dirichlet",
            "right",
            variable="rotation_y",
            value=0.0,
        )
        problem.add_boundary_condition("Dirichlet", "top", variable="u", value=0.0)
        problem.add_boundary_condition("Dirichlet", "top", variable="w", value=0.0)
        problem.add_boundary_condition(
            "Dirichlet",
            "top",
            variable="rotation_x",
            value=0.0,
        )
    return problem


def normalized_centre_deflection(problem):
    """w(0, 0) E1 h^3 / (q a^4)."""
    centre = problem.values("w")[problem.node_at((0.0, 0.0))]
    return centre * 3.0e7 * plate_thickness**3 / (plate_load * plate_side**4)


table_10_5_1 = {
    # Power-law index: DMCDM values for 2x2, 4x4, 8x8, 16x16, 32x32 meshes
    0: [0.01281, 0.01359, 0.01377, 0.01383, 0.01384],
    1: [0.02997, 0.03174, 0.03222, 0.03234, 0.03237],
    2: [0.04395, 0.04659, 0.04728, 0.04746, 0.04749],
    3: [0.05187, 0.05499, 0.05580, 0.05601, 0.05607],
    4: [0.05601, 0.05937, 0.06024, 0.06048, 0.06054],
    5: [0.05838, 0.06186, 0.06279, 0.06300, 0.06306],
}


@pytest.mark.parametrize("power_law_index", sorted(table_10_5_1))
def test_table_10_5_1_clamped_functionally_graded_plate(power_law_index):
    expected = table_10_5_1[power_law_index]
    for num_elements, reference in zip((2, 4, 8, 16), expected):
        problem = rectangular_plate(num_elements, power_law_index=float(power_law_index))
        problem.solve()
        # The book prints five decimals; agreement to the last printed digit.
        assert normalized_centre_deflection(problem) == pytest.approx(reference, abs=2.1e-5)


# ---------------------------------------------------------------------------
# Chapter 8, mixed classical plate theory model DM-CP(M)
# ---------------------------------------------------------------------------
def classical_circular_plate(
    num_elements, power_law_index=0.0, support="hinged", von_karman=False, load=circular_load
):
    """Axisymmetric circular plate, mixed classical (Kirchhoff) theory."""
    s = circular_plate_stiffness(power_law_index)
    mesh = dualmesh.generate_line_mesh(start=0.0, end=circular_radius, num_elements=num_elements)
    problem = dualmesh.Problem(mesh, method="DMCDM", coordinates="axisymmetric")
    plate = problem.add_physics(
        "solid_mechanics",
        formulation="circular_plate",
        theory="classical",
        extensional_stiffness=s.extensional,
        coupling_stiffness=s.coupling,
        bending_stiffness=s.bending,
        poissons_ratio=Poisson_ratio,
        transverse_load=load,
        von_karman=von_karman,
    )
    # Symmetry at r = 0: u(0) = 0.  The slope dw/dr is the natural boundary
    # quantity of the moment equation, so dw/dr(0) = 0 needs no condition.
    plate.add_boundary_condition("disp_r = 0", boundary="left")
    # A hinged edge holds the displacements, and the physics holds its moment at zero.
    if support == "hinged":
        plate.add_boundary_condition(["disp_r = 0", "disp_z = 0"], boundary="right")
    else:
        plate.add_boundary_condition("fixed", boundary="right")
    if von_karman:
        problem.solve()
    else:
        problem.solve()
    return problem


def test_table_8_6_1_mixed_classical_plate():
    """Table 8.6.1, DM-CP(M) column: w(0) for 2, 4, 8, 16, 32, and 64 elements."""
    expected = [0.1149, 0.1156, 0.1158, 0.1159, 0.1159, 0.1159]
    for num_elements, reference in zip((2, 4, 8, 16, 32, 64), expected):
        problem = classical_circular_plate(num_elements)
        centre = problem.values("disp_z")[problem.node_at((0.0,))]
        assert centre == pytest.approx(reference, abs=1.1e-4)


@pytest.mark.parametrize(
    "power_law_index, expected",
    [
        # Table 8.6.2, DM-CP(M) column (32 elements)
        (0.0, 0.0284),
        (1.0, 0.0665),
        (2.0, 0.0975),
        (3.0, 0.1151),
        (4.0, 0.1242),
        (5.0, 0.1294),
        (7.5, 0.1369),
        (10.0, 0.1424),
        (12.0, 0.1466),
        (15.0, 0.1527),
        (20.0, 0.1621),
    ],
)
def test_table_8_6_2_mixed_classical_plate(power_law_index, expected):
    problem = classical_circular_plate(32, power_law_index=power_law_index, support="clamped")
    centre = problem.values("disp_z")[problem.node_at((0.0,))]
    assert centre == pytest.approx(expected, abs=1.1e-4)


@pytest.mark.parametrize("support", ["hinged", "clamped"])
def test_mixed_classical_plate_bending_moment(support):
    """Reddy, Eqs. (8.6.4d) and (8.6.6d): the nodal moment matches the exact
    solution away from the origin.  At r = 0 the mixed models of both the FEM
    and the DMCDM are known to be inaccurate (Section 8.6.2 of the book)."""
    problem = classical_circular_plate(32, support=support)
    radius = problem.node_coordinates()[:, 0]
    moment = problem.values("bending_moment")
    xi = radius / circular_radius
    scale = circular_load * circular_radius**2 / 16.0
    if support == "hinged":
        exact = scale * (3 + Poisson_ratio) * (1 - xi**2)
    else:
        exact = scale * ((1 + Poisson_ratio) - (3 + Poisson_ratio) * xi**2)
    interior = radius > 1e-9
    assert abs(moment[interior] - exact[interior]).max() < 0.05 * scale
