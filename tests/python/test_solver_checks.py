# SPDX-License-Identifier: LGPL-2.1-or-later
"""A solve reports success only when its result solves the problem.

Each test gives an input on which a solve could return a wrong answer without a message: a
nonlinear problem given to the one-step linear solver, a line search that accepts a growing
residual, and a material with a missing modulus.
"""

from __future__ import annotations

import dualmesh
import numpy
import pytest
from dualmesh._core import InputError


def bratu(factor):
    """-u'' = factor exp(u) on (0, 1) with u(0) = u(1) = 0, which has a solution for factor
    below its critical value 3.51 and none above it."""
    mesh = dualmesh.generate_line_mesh(start=0.0, end=1.0, num_elements=20)
    problem = dualmesh.Problem(mesh, method="fem")
    problem.add_variable("u")
    problem.add_kernel("diffusion", variable="u", diffusivity=1.0)
    problem.add_kernel("parsed_kernel", variable="u", source=f"{factor}*exp(u)")
    problem.add_boundary_condition("Dirichlet", ["left", "right"], variable="u", value=0.0)
    return problem


def test_the_linear_solver_refuses_a_nonlinear_problem():
    """One linear solve of the Bratu problem leaves the residual of its nonlinear source."""
    with pytest.raises(RuntimeError, match="nonlinear_solver 'linear' takes one linear solve"):
        bratu(1.0).solve(nonlinear_solver="linear", report="none")


def test_the_linear_solver_solves_a_linear_problem():
    """-u'' = 1 with u(0) = u(1) = 0 has the solution u = x (1 - x) / 2, which linear
    elements reproduce at the nodes."""
    mesh = dualmesh.generate_line_mesh(start=0.0, end=1.0, num_elements=10)
    problem = dualmesh.Problem(mesh, method="fem")
    problem.add_variable("u")
    problem.add_kernel("diffusion", variable="u", diffusivity=1.0)
    problem.add_kernel("parsed_kernel", variable="u", source="1")
    problem.add_boundary_condition("Dirichlet", ["left", "right"], variable="u", value=0.0)
    result = problem.solve(nonlinear_solver="linear", report="none")
    assert result.converged
    x = problem.node_coordinates()[:, 0]
    assert numpy.abs(problem.values("u") - x * (1 - x) / 2).max() < 1e-12


def test_options_of_the_nonlinear_solver_reach_it():
    """PETSc options that start with -snes_ go to the nonlinear solver: Newton's method damped to
    half steps converges linearly, in more iterations, to the same solution."""
    full = bratu(1.0)
    full_result = full.solve(report="none")
    damped = bratu(1.0)
    damped_result = damped.solve(petsc_options="-snes_linesearch_damping 0.5", report="none")
    assert damped_result.total_iterations > full_result.total_iterations + 3
    assert numpy.abs(damped.values("u") - full.values("u")).max() < 1e-9


def test_relaxation_damps_every_step_after_the_first():
    """Direct iteration with the relaxation gamma takes U + (1 - gamma) dU from its second step on,
    and reaches the solution of the full steps."""
    full = bratu(1.0)
    full_result = full.solve(nonlinear_solver="picard", report="none")
    relaxed = bratu(1.0)
    relaxed_result = relaxed.solve(nonlinear_solver="picard", relaxation=0.5, report="none")
    assert relaxed_result.total_iterations > full_result.total_iterations
    assert numpy.abs(relaxed.values("u") - full.values("u")).max() < 1e-8


def test_the_line_search_never_accepts_a_larger_residual():
    """The Bratu problem above its critical value has no solution.  Backtracking may fail on it,
    but every iterate it accepts has a residual no larger than the one before."""
    result = bratu(5.0).solve(line_search="backtracking", error_on_divergence=False, report="none")
    assert not result.converged
    steps = {}
    for record in result.history:
        steps.setdefault(record.continuation_step, []).append(record.residual_norm)
    for norms in steps.values():
        assert all(later <= earlier for earlier, later in zip(norms, norms[1:])), norms


def test_a_continuum_without_poissons_ratio_is_refused():
    """Poisson's ratio has no value that suits every material, so a missing one is an error."""
    mesh = dualmesh.generate_rectangle_mesh(0, 1, 0, 1, 2, 2)
    problem = dualmesh.Problem(mesh, method="fem")
    with pytest.raises(InputError, match="Poisson's ratio is not given"):
        solid = problem.add_physics(
            "solid_mechanics", formulation="plane_strain", youngs_modulus=1e9
        )
        solid.add_boundary_condition("fixed", boundary="left")
        problem.solve(report="none")


def test_phase_field_fracture_at_finite_strain_is_refused():
    """The degradation of the elastic energy acts on the small strain path only."""
    mesh = dualmesh.generate_rectangle_mesh(0, 1, 0, 1, 2, 2)
    problem = dualmesh.Problem(mesh, method="fem")
    with pytest.raises(InputError, match="phase-field fracture is available at small strain only"):
        solid = problem.add_physics(
            "solid_mechanics",
            formulation="plane_strain",
            youngs_modulus=1e9,
            poissons_ratio=0.3,
            large_deformation=True,
            phase_field_fracture={"toughness": 100.0, "length": 0.01},
        )
        solid.add_boundary_condition("fixed", boundary="left")
        problem.solve(report="none")
