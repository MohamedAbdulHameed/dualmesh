# SPDX-License-Identifier: LGPL-2.1-or-later
"""Linear buckling: the Euler columns of beams and of a plane continuum."""

from __future__ import annotations

import dualmesh
import numpy
import pytest

E, B, H, L, P = 200e9, 0.02, 0.01, 1.0, 1000.0
second_moment = B * H**3 / 12


def pinned_column(theory, n):
    mesh = dualmesh.generate_line_mesh(start=0.0, end=L, num_elements=n)
    problem = dualmesh.Problem(mesh)
    beam = problem.add_physics(
        "solid_mechanics",
        formulation="beam",
        theory=theory,
        youngs_modulus=E,
        poissons_ratio=0.3,
        cross_section={"width": B, "height": H},
        von_karman=True,
    )
    beam.add_boundary_condition("disp_x = 0", boundary="left")
    beam.add_boundary_condition("disp_z = 0", boundary=["left", "right"])
    problem.add_point_source(variable="disp_x", value=-P, points=[L])
    problem.solve(report="none")
    return problem


@pytest.mark.parametrize("theory", ["Euler-Bernoulli", "Timoshenko"])
def test_a_pinned_column_buckles_at_the_Euler_load(theory):
    """P_k = k^2 pi^2 E I / L^2 (Euler-Bernoulli), and for Timoshenko P_k / (1 + P_k / (kappa G
    A)) with kappa = 5/6: the first two load factors converge at second order in the element
    size (measured 0.32 % and 0.08 % at 16 and 32 elements), and the study leaves the solution as
    it was."""
    Euler_loads = numpy.pi**2 * E * second_moment / L**2 * numpy.array([1.0, 4.0])
    if theory == "Timoshenko":
        shear = 5.0 / 6.0 * E / (2 * 1.3) * B * H
        Euler_loads = Euler_loads / (1 + Euler_loads / shear)
    exact = Euler_loads / P
    errors = []
    for n in (16, 32):
        problem = pinned_column(theory, n)
        before = numpy.asarray(problem._problem.solution()).copy()
        result = problem.solve_buckling(num_modes=2, report="none")
        assert numpy.asarray(problem._problem.solution()) == pytest.approx(before, abs=0.0)
        assert max(result.residual_norms) < 1e-10
        errors.append(abs(result.load_factors[0] - exact[0]))
        assert result.load_factors == pytest.approx(exact, rel=0.02)
    assert errors[0] / errors[1] == pytest.approx(4.0, rel=0.1)


def test_a_plane_column_buckles_at_the_Euler_load_of_a_cantilever():
    """A clamped-free plane-strain column (nu = 0, L/h = 40) at finite strain: P = pi^2 E I /
    (4 L^2) per unit thickness, to the shear correction of its slenderness (measured 8e-5)."""
    h = 0.025
    mesh = dualmesh.generate_rectangle_mesh(0, h, 0, 1.0, 2, 40, element_type="Quad9")
    problem = dualmesh.Problem(mesh)
    solid = problem.add_physics(
        "solid_mechanics",
        formulation="plane_strain",
        youngs_modulus=E,
        poissons_ratio=0.0,
        large_deformation=True,
    )
    solid.add_boundary_condition("fixed", boundary="bottom")
    solid.add_boundary_condition("traction = [0, -1e6]", boundary="top")
    problem.solve(report="none")
    result = problem.solve_buckling(report="none")
    exact = numpy.pi**2 * E * h**3 / 12 / 4 / (1e6 * h)
    assert result.load_factors[0] == pytest.approx(exact, rel=2e-4)


def test_buckling_needs_a_geometric_stiffness():
    mesh = dualmesh.generate_line_mesh(start=0.0, end=L, num_elements=8)
    problem = dualmesh.Problem(mesh)
    beam = problem.add_physics(
        "solid_mechanics",
        formulation="beam",
        youngs_modulus=E,
        poissons_ratio=0.3,
        cross_section={"width": B, "height": H},
    )
    beam.add_boundary_condition("fixed", boundary="left")
    problem.add_point_source(variable="disp_x", value=-P, points=[L])
    problem.solve(report="none")
    with pytest.raises(ValueError, match="no geometric stiffness"):
        problem.solve_buckling(report="none")
