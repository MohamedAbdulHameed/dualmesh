# SPDX-License-Identifier: LGPL-2.1-or-later
"""The pressure mass matrix Schur complement preconditioner.

The Jacobian of a pressure-velocity flow is a saddle point matrix
[[A, G], [D, C]] (momentum, pressure gradient, divergence, pressure block).
Its Schur complement S = C - D A^{-1} G is spectrally equivalent, for the
Stokes equations, to the pressure mass matrix scaled by 1/mu, with bounds
independent of the mesh size.  The block upper-triangular preconditioner with
that approximation (``preconditioner="pressure_mass_schur"``) should therefore
converge in a number of flexible GMRES iterations that does not grow with the
mesh.  The tests check

* that the iteration count is mesh independent for the Taylor-Hood element
  (15 to 16 per Newton step measured on 8 x 8, 16 x 16 and 32 x 32 meshes at
  a relative tolerance of 1e-12) and for the stabilised equal-order element
  (16 to 19, with the stabilisation's own pressure block added to the
  approximation);
* that the answer is the one of the direct solver;
* that PETSc's Schur complement field split, given the same matrix, agrees;
* that misuse is refused with a message that says what to do.
"""

from __future__ import annotations

import dualmesh as dm
import numpy as np
import pytest


def _cavity(formulation, element_type, n, density=0.0, method="fem"):
    mesh = dm.generate_rectangle_mesh(0, 1, 0, 1, n, n, element_type=element_type)
    problem = dm.Problem(mesh, method=method)
    problem.add_physics(
        "incompressible_flow",
        "flow",
        velocities=["u", "v"],
        density=density,
        formulation=formulation,
        pressure_pin_point=(0.5, 0.0),
    )
    problem.add_boundary_condition(
        "Dirichlet_boundary_condition", "lid", variable="u", boundary="top", value=1.0
    )
    problem.add_boundary_condition(
        "Dirichlet_boundary_condition", "lid_v", variable="v", boundary="top", value=0.0
    )
    for variable in ("u", "v"):
        problem.add_boundary_condition(
            "Dirichlet_boundary_condition",
            f"walls_{variable}",
            variable=variable,
            boundary=["left", "right", "bottom"],
            value=0.0,
        )
    return problem


def _relative_difference(problem, reference):
    return max(
        np.abs(problem.values(v) - reference.values(v)).max()
        / max(np.abs(reference.values(v)).max(), 1e-300)
        for v in ("u", "v", "pressure")
    )


@pytest.mark.parametrize(
    "formulation, element_type, limit", [("Taylor_Hood", "Quad9", 20), ("pressure", "Quad4", 22)]
)
def test_iterations_do_not_grow_with_the_mesh(formulation, element_type, limit):
    counts = []
    for n in (8, 16, 32):
        reference = _cavity(formulation, element_type, n)
        reference.solve(linear_solver="lu")
        problem = _cavity(formulation, element_type, n)
        result = problem.solve(
            linear_solver="gmres", preconditioner="pressure_mass_schur", linear_tolerance=1e-12
        )
        counts.append(result.linear_iterations / result.total_iterations)
        assert _relative_difference(problem, reference) < 1e-8
    assert max(counts) <= limit, counts
    assert max(counts) - min(counts) <= 3, counts


def test_navier_stokes_at_a_moderate_reynolds_number():
    """At Re = 100 the mass matrix is a cruder approximation (the Schur
    complement then also carries convection); measured about 40 iterations
    per Newton step."""
    reference = _cavity("Taylor_Hood", "Quad9", 16, density=100.0)
    reference.solve(linear_solver="lu")
    problem = _cavity("Taylor_Hood", "Quad9", 16, density=100.0)
    result = problem.solve(linear_solver="gmres", preconditioner="pressure_mass_schur")
    assert _relative_difference(problem, reference) < 1e-8
    assert result.linear_iterations / result.total_iterations < 60


@pytest.mark.skipif(not dm.have_petsc(), reason="built without PETSc")
@pytest.mark.parametrize("options", [None, "-fieldsplit_0_pc_type gamg"])
def test_petsc_field_split_with_the_pressure_mass_matrix(options):
    reference = _cavity("Taylor_Hood", "Quad9", 12)
    reference.solve(linear_solver="lu")
    problem = _cavity("Taylor_Hood", "Quad9", 12)
    problem.solve(
        linear_solver="petsc",
        preconditioner="pressure_mass_schur",
        petsc_options=options,
        linear_tolerance=1e-12,
    )
    assert _relative_difference(problem, reference) < 1e-7


def test_it_works_with_the_dual_mesh_method():
    reference = _cavity("pressure", "Quad4", 12, method="dmcdm")
    reference.solve(linear_solver="lu")
    problem = _cavity("pressure", "Quad4", 12, method="dmcdm")
    problem.solve(linear_solver="gmres", preconditioner="pressure_mass_schur")
    assert _relative_difference(problem, reference) < 1e-8


def test_it_needs_a_krylov_solver():
    problem = _cavity("Taylor_Hood", "Quad9", 4)
    with pytest.raises(ValueError, match="'gmres' or 'automatic'"):
        problem.solve(linear_solver="bicgstab", preconditioner="pressure_mass_schur")


def test_it_needs_a_mass_conservation_equation():
    mesh = dm.generate_rectangle_mesh(0, 1, 0, 1, 4, 4)
    problem = dm.Problem(mesh, method="fem")
    problem.add_variable("temperature")
    problem.add_kernel("diffusion", "conduction", variable="temperature")
    problem.add_kernel("body_force", "heating", variable="temperature", value=1.0)
    problem.add_boundary_condition(
        "Dirichlet_boundary_condition", "cold", variable="temperature", boundary="left", value=0.0
    )
    with pytest.raises(ValueError, match="mass_conservation"):
        problem.solve(linear_solver="gmres", preconditioner="pressure_mass_schur")
