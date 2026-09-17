# SPDX-License-Identifier: LGPL-2.1-or-later
"""Verification of the beam models against Chapter 7 of Reddy's book.

Geometry and material of every example: length L = 100 in., width b = 1 in.,
height h = 1 in., E1 = 30e6 psi (top), E2 = 3e6 psi (bottom), nu = 0.3, shear
correction factor 5/6.  Only half of the beam is modelled, 0 <= x <= L/2, with
symmetry conditions at the centre.
"""

from __future__ import annotations

import dualmesh as dm
import numpy as np
import pytest

LENGTH = 100.0
WIDTH = 1.0
HEIGHT = 1.0
MODULUS_TOP = 30.0e6
MODULUS_BOTTOM = 3.0e6
POISSON = 0.3
LOAD = 0.5


def stiffness(power_law_index):
    return dm.fgm.beam_stiffness(
        modulus_top=MODULUS_TOP,
        modulus_bottom=MODULUS_BOTTOM,
        power_law_index=power_law_index,
        height=HEIGHT,
        width=WIDTH,
        poisson_ratio=POISSON,
    )


def test_functionally_graded_stiffness_formulas():
    """The closed-form FGM stiffnesses agree with numerical integration."""
    for n in (0.0, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0):
        s = stiffness(n)
        a, b, d = dm.fgm.stiffness_by_quadrature(MODULUS_TOP, MODULUS_BOTTOM, n, HEIGHT, WIDTH)
        assert s.extensional == pytest.approx(a, rel=1e-8)
        assert s.coupling == pytest.approx(b, rel=1e-8)
        assert s.bending == pytest.approx(d, rel=1e-8)


def beam_problem(
    model,
    num_elements,
    power_law_index=0.0,
    support="pinned",
    method="dmcdm",
    von_karman=False,
    load=LOAD,
):
    """Half-beam model; returns (problem, third_variable_name)."""
    s = stiffness(power_law_index)
    mesh = dm.generate_line_mesh(start=0.0, end=LENGTH / 2, num_elements=num_elements)
    problem = dm.Problem(mesh, method=method)
    third = "rotation" if model == "BeamTimoshenkoDisplacement" else "moment"
    for name in ("axial_displacement", "deflection", third):
        problem.add_variable(name)
    common = dict(
        extensional_stiffness=s.extensional,
        coupling_stiffness=s.coupling,
        bending_stiffness=s.bending,
        shear_stiffness=s.shear,
        transverse_load=load,
        von_karman=von_karman,
    )
    if model == "BeamTimoshenkoDisplacement":
        variables = dict(
            axial_displacement="axial_displacement",
            transverse_displacement="deflection",
            rotation="rotation",
        )
    else:
        variables = dict(
            axial_displacement="axial_displacement",
            transverse_displacement="deflection",
            bending_moment="moment",
        )
    for variable in ("axial_displacement", "deflection", third):
        options = dict(common, **variables, variable=variable)
        if model == "BeamTimoshenkoDisplacement" and variable == "rotation":
            # Reduced integration of the shear term removes shear locking.
            options.update(quadrature="midpoint", reduced_integration=True)
        problem.add_kernel(model, f"{model}_{variable}", **options)

    # x = 0: pinned (w = 0, M = 0) or clamped (w = 0, rotation = 0)
    problem.add_boundary_condition(
        "DirichletBC", "fixed_axial", variable="axial_displacement", boundary="left", value=0.0
    )
    problem.add_boundary_condition(
        "DirichletBC", "support", variable="deflection", boundary="left", value=0.0
    )
    if support == "pinned" and third == "moment":
        problem.add_boundary_condition(
            "DirichletBC", "no_moment", variable="moment", boundary="left", value=0.0
        )
    elif support == "clamped" and third == "rotation":
        problem.add_boundary_condition(
            "DirichletBC", "no_rotation", variable="rotation", boundary="left", value=0.0
        )
    # x = L/2: symmetry (u = 0, and the natural conditions V = 0, rotation = 0)
    problem.add_boundary_condition(
        "DirichletBC",
        "symmetry_axial",
        variable="axial_displacement",
        boundary="right",
        value=0.0,
    )
    if third == "rotation":
        problem.add_boundary_condition(
            "DirichletBC", "symmetry_rotation", variable="rotation", boundary="right", value=0.0
        )
    return problem, third


@pytest.mark.parametrize(
    "model, expected",
    [
        # Table 7.5.1: centre deflection w(L/2) in inches, q0 = 0.5 lb/in
        ("BeamEulerBernoulliMixed", [0.2588, 0.2600, 0.2603, 0.2604]),
        ("BeamTimoshenkoDisplacement", [0.2540, 0.2588, 0.2600, 0.2604]),
        ("BeamTimoshenkoMixed", [0.2588, 0.2600, 0.2604, 0.2604]),
    ],
)
def test_table_7_5_1_pinned_homogeneous_beam(model, expected):
    for num_elements, reference in zip((4, 8, 16, 32), expected):
        problem, _ = beam_problem(model, num_elements)
        problem.solve()
        centre = problem.values("deflection")[-1]
        # The book prints four decimals, so +-1.1e-4 is agreement to the
        # last published digit.
        assert centre == pytest.approx(reference, abs=1.1e-4)


@pytest.mark.parametrize("model", ["BeamEulerBernoulliMixed", "BeamTimoshenkoMixed"])
@pytest.mark.parametrize("num_elements", [4, 8, 16, 32])
def test_table_7_5_2_bending_moment_is_exact(model, num_elements):
    """The mixed dual mesh models give the exact centre moment q0 L^2 / 8."""
    problem, _ = beam_problem(model, num_elements)
    problem.solve()
    assert problem.values("moment")[-1] == pytest.approx(LOAD * LENGTH**2 / 8.0, rel=1e-10)


@pytest.mark.parametrize(
    "power_law_index, euler_mixed, timoshenko_displacement, timoshenko_mixed",
    [
        # Table 7.5.3: w(L/2) Dhat / (q0 L^4) x 10, 16 elements in the half beam
        (0.0, 0.1302, 0.1300, 0.1302),
        (1.0, 0.1069, 0.1068, 0.1069),
        (2.0, 0.0919, 0.0918, 0.0919),
        (3.0, 0.0879, 0.0878, 0.0879),
        (5.0, 0.0900, 0.0899, 0.0900),
        (10.0, 0.1012, 0.1011, 0.1012),
        (20.0, 0.1141, 0.1140, 0.1142),
    ],
)
def test_table_7_5_3_functionally_graded_pinned_beam(
    power_law_index, euler_mixed, timoshenko_displacement, timoshenko_mixed
):
    s = stiffness(power_law_index)
    reduced_bending = s.bending - s.coupling**2 / s.extensional
    references = {
        "BeamEulerBernoulliMixed": euler_mixed,
        "BeamTimoshenkoDisplacement": timoshenko_displacement,
        "BeamTimoshenkoMixed": timoshenko_mixed,
    }
    for model, reference in references.items():
        problem, _ = beam_problem(model, 16, power_law_index=power_law_index)
        problem.solve()
        centre = problem.values("deflection")[-1]
        normalized = centre * reduced_bending / (LOAD * LENGTH**4) * 10
        assert normalized == pytest.approx(reference, abs=1.1e-4)


@pytest.mark.parametrize(
    "model, expected",
    [
        # Table 7.5.4: w(L/2) Dhat / (q0 L^4) x 100 for clamped-clamped beams
        ("BeamEulerBernoulliMixed", [0.2685, 0.2624, 0.2609, 0.2605, 0.2604]),
        ("BeamTimoshenkoDisplacement", [0.2445, 0.2567, 0.2597, 0.2605, 0.2607]),
        ("BeamTimoshenkoMixed", [0.2688, 0.2628, 0.2612, 0.2609, 0.2608]),
    ],
)
def test_table_7_5_4_clamped_homogeneous_beam(model, expected):
    s = stiffness(0.0)
    reduced_bending = s.bending - s.coupling**2 / s.extensional
    for num_elements, reference in zip((4, 8, 16, 32, 64), expected):
        problem, _ = beam_problem(model, num_elements, support="clamped")
        problem.solve()
        centre = problem.values("deflection")[-1]
        normalized = centre * reduced_bending / (LOAD * LENGTH**4) * 100
        assert normalized == pytest.approx(reference, abs=1.1e-4)


def test_euler_bernoulli_and_timoshenko_agree_for_a_slender_beam():
    """With L/h = 100 shear deformation is negligible."""
    euler, _ = beam_problem("BeamEulerBernoulliMixed", 32)
    euler.solve()
    timoshenko, _ = beam_problem("BeamTimoshenkoMixed", 32)
    timoshenko.solve()
    assert euler.values("deflection")[-1] == pytest.approx(
        timoshenko.values("deflection")[-1], rel=1e-3
    )


def test_reduced_integration_removes_shear_locking():
    """Full integration of the shear term locks the displacement model."""
    reduced_problem, _ = beam_problem("BeamTimoshenkoDisplacement", 8)
    # the same model again, but with the shear term integrated fully
    s = stiffness(0.0)
    mesh = dm.generate_line_mesh(start=0.0, end=LENGTH / 2, num_elements=8)
    problem = dm.Problem(mesh)
    for name in ("axial_displacement", "deflection", "rotation"):
        problem.add_variable(name)
    for variable in ("axial_displacement", "deflection", "rotation"):
        problem.add_kernel(
            "BeamTimoshenkoDisplacement",
            f"beam_{variable}",
            variable=variable,
            axial_displacement="axial_displacement",
            transverse_displacement="deflection",
            rotation="rotation",
            extensional_stiffness=s.extensional,
            coupling_stiffness=s.coupling,
            bending_stiffness=s.bending,
            shear_stiffness=s.shear,
            transverse_load=LOAD,
        )
    problem.add_boundary_condition(
        "DirichletBC", "a", variable="axial_displacement", boundary="left", value=0.0
    )
    problem.add_boundary_condition(
        "DirichletBC", "b", variable="deflection", boundary="left", value=0.0
    )
    problem.add_boundary_condition(
        "DirichletBC", "c", variable="axial_displacement", boundary="right", value=0.0
    )
    problem.add_boundary_condition(
        "DirichletBC", "d", variable="rotation", boundary="right", value=0.0
    )
    problem.solve()
    reduced_problem.solve()
    full_integration = problem.values("deflection")[-1]
    reduced = reduced_problem.values("deflection")[-1]
    # The locked solution is far too stiff, the reduced-integration one is not.
    assert full_integration < 0.2 * reduced
    assert reduced == pytest.approx(0.2588, abs=1.1e-4)


# ---------------------------------------------------------------------------
# Nonlinear (von Karman) analysis, Section 7.6
# ---------------------------------------------------------------------------
NONLINEAR_REFERENCE = {
    # q0: (DM-TB(D), DM-EB(M), DM-TB(M)) of Table 7.6.1,
    # tabulated as w(L/2) Dhat / L^4 x 10
    0.5: (0.0563, 0.0564, 0.0563),
    1.0: (0.0921, 0.0921, 0.0921),
    2.0: (0.1364, 0.1364, 0.1364),
    5.0: (0.2078, 0.2078, 0.2078),
    10.0: (0.2743, 0.2743, 0.2743),
    20.0: (0.3547, 0.3547, 0.3547),
}


@pytest.mark.parametrize("load, reference", sorted(NONLINEAR_REFERENCE.items()))
def test_table_7_6_1_nonlinear_pinned_beam(load, reference):
    """von Karman bending of a pinned-pinned beam with 16 elements.

    The book uses load increments of 1 lb/in and direct (Picard) iteration with
    the acceleration parameter 0.35; Newton's method reaches the same answer.
    """
    s = stiffness(0.0)
    reduced_bending = s.bending - s.coupling**2 / s.extensional
    models = ("BeamTimoshenkoDisplacement", "BeamEulerBernoulliMixed", "BeamTimoshenkoMixed")
    steps = max(1, int(round(load)))
    factors = list(np.linspace(1.0 / steps, 1.0, steps))
    for model, expected in zip(models, reference):
        problem, _ = beam_problem(model, 16, von_karman=True, load=load)
        problem.solve(load_factors=factors, max_iterations=100)
        normalized = problem.values("deflection")[-1] * reduced_bending / LENGTH**4 * 10
        assert normalized == pytest.approx(expected, abs=1.1e-4)


def test_nonlinear_beam_direct_iteration_with_acceleration():
    """Direct iteration with the book's acceleration parameter beta = 0.35."""
    s = stiffness(0.0)
    reduced_bending = s.bending - s.coupling**2 / s.extensional
    problem, _ = beam_problem("BeamEulerBernoulliMixed", 16, von_karman=True, load=10.0)
    problem.solve(
        nonlinear_solver="picard",
        relaxation=0.35,
        load_factors=list(np.linspace(0.1, 1.0, 10)),
        max_iterations=200,
        step_tolerance=1e-8,
    )
    normalized = problem.values("deflection")[-1] * reduced_bending / LENGTH**4 * 10
    assert normalized == pytest.approx(0.2743, abs=1.1e-4)


def test_nonlinear_beam_is_stiffer_than_the_linear_one():
    linear, _ = beam_problem("BeamEulerBernoulliMixed", 16, load=10.0)
    linear.solve()
    nonlinear, _ = beam_problem("BeamEulerBernoulliMixed", 16, von_karman=True, load=10.0)
    nonlinear.solve(load_factors=list(np.linspace(0.1, 1.0, 10)), max_iterations=100)
    assert nonlinear.values("deflection")[-1] < 0.3 * linear.values("deflection")[-1]
