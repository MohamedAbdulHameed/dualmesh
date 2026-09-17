# SPDX-License-Identifier: LGPL-2.1-or-later
"""Verification of the plate models against Chapters 8 and 10 of Reddy's book.

Chapter 8: axisymmetric circular plates, radius a = 10 in, thickness
h = 0.1 in, E1 = 30e6 psi, E2 = 3e6 psi, nu = 0.3, K_s = 5/6, q0 = 0.5 lb/in.

Chapter 10: rectangular plates, a = b = 1 in, h = 0.01 in, E1 = 3e7 psi,
E2 = 3e6 psi, nu = 0.3, q = 1 lb/in^2, modelled on one quarter of the plate.
"""

from __future__ import annotations

import dualmesh as dm
import pytest

# ---------------------------------------------------------------------------
# Chapter 8: axisymmetric circular plates
# ---------------------------------------------------------------------------
CIRCULAR_RADIUS = 10.0
CIRCULAR_THICKNESS = 0.1
CIRCULAR_LOAD = 0.5
MODULUS_TOP = 30.0e6
MODULUS_BOTTOM = 3.0e6
POISSON = 0.3


def circular_plate_stiffness(power_law_index):
    return dm.fgm.beam_stiffness(
        modulus_top=MODULUS_TOP,
        modulus_bottom=MODULUS_BOTTOM,
        power_law_index=power_law_index,
        height=CIRCULAR_THICKNESS,
        width=1.0,
        poisson_ratio=POISSON,
        plate=True,
    )


def circular_plate(num_elements, power_law_index=0.0, support="hinged"):
    """Axisymmetric circular plate, first-order shear deformation theory."""
    s = circular_plate_stiffness(power_law_index)
    mesh = dm.generate_line_mesh(start=0.0, end=CIRCULAR_RADIUS, num_elements=num_elements)
    problem = dm.Problem(mesh, coordinates="axisymmetric")
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
        poissons_ratio=POISSON,
        transverse_load=CIRCULAR_LOAD,
    )
    for variable in ("radial_displacement", "deflection", "rotation"):
        problem.add_kernel(
            "CircularPlateFirstOrder",
            f"bending_{variable}",
            variable=variable,
            shear_treatment="exclude",
            **common,
        )
        problem.add_kernel(
            "CircularPlateFirstOrder",
            f"shear_{variable}",
            variable=variable,
            shear_treatment="only",
            quadrature="midpoint",
            reduced_integration=True,
            **common,
        )
    # Symmetry at r = 0
    problem.add_boundary_condition(
        "DirichletBC", "centre_u", variable="radial_displacement", boundary="left", value=0.0
    )
    problem.add_boundary_condition(
        "DirichletBC", "centre_rotation", variable="rotation", boundary="left", value=0.0
    )
    # Edge conditions at r = a
    problem.add_boundary_condition(
        "DirichletBC", "edge_u", variable="radial_displacement", boundary="right", value=0.0
    )
    problem.add_boundary_condition(
        "DirichletBC", "edge_w", variable="deflection", boundary="right", value=0.0
    )
    if support == "clamped":
        problem.add_boundary_condition(
            "DirichletBC", "edge_rotation", variable="rotation", boundary="right", value=0.0
        )
    problem.solve()
    return problem


def hinged_exact_deflection(power_law_index=0.0):
    """Reddy, Eq. (8.6.4b): centre deflection of a hinged FGM circular plate."""
    s = circular_plate_stiffness(power_law_index)
    reduced = s.bending * s.extensional - s.coupling**2
    a, q = CIRCULAR_RADIUS, CIRCULAR_LOAD
    bending_part = s.extensional * q * a**4 / (64 * reduced) * (5 + POISSON) / (1 + POISSON)
    shear_part = q * a**2 / (4 * s.shear)
    return bending_part + shear_part


def clamped_exact_deflection(power_law_index=0.0):
    """Reddy, Eq. (8.6.6b): centre deflection of a clamped FGM circular plate."""
    s = circular_plate_stiffness(power_law_index)
    reduced = s.bending * s.extensional - s.coupling**2
    a, q = CIRCULAR_RADIUS, CIRCULAR_LOAD
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
PLATE_SIDE = 1.0
PLATE_THICKNESS = 0.01
PLATE_LOAD = 1.0


def rectangular_plate_stiffness(power_law_index):
    return dm.fgm.beam_stiffness(
        modulus_top=3.0e7,
        modulus_bottom=3.0e6,
        power_law_index=power_law_index,
        height=PLATE_THICKNESS,
        width=1.0,
        poisson_ratio=POISSON,
        plate=True,
    )


def rectangular_plate(
    num_elements,
    power_law_index=0.0,
    support="clamped",
    method="dmcdm",
    von_karman=False,
    load=PLATE_LOAD,
):
    """Quarter model of a rectangular plate, first-order shear deformation theory."""
    s = rectangular_plate_stiffness(power_law_index)
    mesh = dm.generate_rectangle_mesh(
        x_min=0.0,
        x_max=PLATE_SIDE / 2,
        y_min=0.0,
        y_max=PLATE_SIDE / 2,
        num_x_elements=num_elements,
        num_y_elements=num_elements,
    )
    problem = dm.Problem(mesh, method=method)
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
        poissons_ratio=POISSON,
        transverse_load=load,
        von_karman=von_karman,
    )
    for variable in variables:
        problem.add_kernel(
            "PlateFirstOrder",
            f"bending_{variable}",
            variable=variable,
            shear_treatment="exclude",
            **common,
        )
        problem.add_kernel(
            "PlateFirstOrder",
            f"shear_{variable}",
            variable=variable,
            shear_treatment="only",
            quadrature="midpoint",
            reduced_integration=True,
            **common,
        )
    # Symmetry: u = phi_x = 0 on x = 0 and v = phi_y = 0 on y = 0
    problem.add_boundary_condition("DirichletBC", "sym_u", variable="u", boundary="left", value=0.0)
    problem.add_boundary_condition(
        "DirichletBC", "sym_px", variable="rotation_x", boundary="left", value=0.0
    )
    problem.add_boundary_condition(
        "DirichletBC", "sym_v", variable="v", boundary="bottom", value=0.0
    )
    problem.add_boundary_condition(
        "DirichletBC", "sym_py", variable="rotation_y", boundary="bottom", value=0.0
    )
    if support == "clamped":
        for name in variables:
            problem.add_boundary_condition(
                "DirichletBC",
                f"clamped_{name}",
                variable=name,
                boundary=["right", "top"],
                value=0.0,
            )
    else:  # simply supported (SS-1)
        problem.add_boundary_condition(
            "DirichletBC", "ss_v", variable="v", boundary="right", value=0.0
        )
        problem.add_boundary_condition(
            "DirichletBC", "ss_w_right", variable="w", boundary="right", value=0.0
        )
        problem.add_boundary_condition(
            "DirichletBC", "ss_py", variable="rotation_y", boundary="right", value=0.0
        )
        problem.add_boundary_condition(
            "DirichletBC", "ss_u", variable="u", boundary="top", value=0.0
        )
        problem.add_boundary_condition(
            "DirichletBC", "ss_w_top", variable="w", boundary="top", value=0.0
        )
        problem.add_boundary_condition(
            "DirichletBC", "ss_px", variable="rotation_x", boundary="top", value=0.0
        )
    return problem


def normalized_centre_deflection(problem):
    """w(0, 0) E1 h^3 / (q a^4)."""
    centre = problem.values("w")[problem.node_at((0.0, 0.0))]
    return centre * 3.0e7 * PLATE_THICKNESS**3 / (PLATE_LOAD * PLATE_SIDE**4)


TABLE_10_5_1 = {
    # power-law index: DMCDM values for 2x2, 4x4, 8x8, 16x16, 32x32 meshes
    0: [0.01281, 0.01359, 0.01377, 0.01383, 0.01384],
    1: [0.02997, 0.03174, 0.03222, 0.03234, 0.03237],
    2: [0.04395, 0.04659, 0.04728, 0.04746, 0.04749],
    3: [0.05187, 0.05499, 0.05580, 0.05601, 0.05607],
    4: [0.05601, 0.05937, 0.06024, 0.06048, 0.06054],
    5: [0.05838, 0.06186, 0.06279, 0.06300, 0.06306],
}


@pytest.mark.parametrize("power_law_index", sorted(TABLE_10_5_1))
def test_table_10_5_1_clamped_functionally_graded_plate(power_law_index):
    expected = TABLE_10_5_1[power_law_index]
    for num_elements, reference in zip((2, 4, 8, 16), expected):
        problem = rectangular_plate(num_elements, power_law_index=float(power_law_index))
        problem.solve()
        # The book prints five decimals; agreement to the last printed digit.
        assert normalized_centre_deflection(problem) == pytest.approx(reference, abs=2.1e-5)


def test_clamped_plate_matches_the_classical_thin_plate_solution():
    """For a/h = 100 the FST result is the classical value 0.00126 q a^4 / D."""
    problem = rectangular_plate(16)
    problem.solve()
    s = rectangular_plate_stiffness(0.0)
    centre = problem.values("w")[problem.node_at((0.0, 0.0))]
    classical = 0.00126 * PLATE_LOAD * PLATE_SIDE**4 / s.bending
    assert centre == pytest.approx(classical, rel=0.01)


def test_plate_reduced_integration_is_required_for_thin_plates():
    """Without reduced integration of the shear terms the plate locks."""
    s = rectangular_plate_stiffness(0.0)
    mesh = dm.generate_rectangle_mesh(
        x_min=0.0, x_max=0.5, y_min=0.0, y_max=0.5, num_x_elements=8, num_y_elements=8
    )
    problem = dm.Problem(mesh)
    variables = ["u", "v", "w", "rotation_x", "rotation_y"]
    for name in variables:
        problem.add_variable(name)
    for variable in variables:
        problem.add_kernel(
            "PlateFirstOrder",
            f"plate_{variable}",
            variable=variable,
            in_plane_displacements=["u", "v"],
            transverse_displacement="w",
            rotations=["rotation_x", "rotation_y"],
            extensional_stiffness=s.extensional,
            coupling_stiffness=s.coupling,
            bending_stiffness=s.bending,
            shear_stiffness=s.shear,
            poissons_ratio=POISSON,
            transverse_load=PLATE_LOAD,
        )
    problem.add_boundary_condition("DirichletBC", "a", variable="u", boundary="left", value=0.0)
    problem.add_boundary_condition(
        "DirichletBC", "b", variable="rotation_x", boundary="left", value=0.0
    )
    problem.add_boundary_condition("DirichletBC", "c", variable="v", boundary="bottom", value=0.0)
    problem.add_boundary_condition(
        "DirichletBC", "d", variable="rotation_y", boundary="bottom", value=0.0
    )
    for name in variables:
        problem.add_boundary_condition(
            "DirichletBC", f"e_{name}", variable=name, boundary=["right", "top"], value=0.0
        )
    problem.solve()
    locked = problem.values("w")[problem.node_at((0.0, 0.0))]

    unlocked_problem = rectangular_plate(8)
    unlocked_problem.solve()
    unlocked = unlocked_problem.values("w")[unlocked_problem.node_at((0.0, 0.0))]
    assert locked < 0.2 * unlocked
