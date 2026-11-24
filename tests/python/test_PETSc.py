# SPDX-License-Identifier: LGPL-2.1-or-later
"""The linear solves of PETSc: whatever Krylov method and preconditioner the options select,
a solve gives the answer of the direct factorisation.  The distributed solves are tested in
``test_parallel.py``.
"""

from __future__ import annotations

import dualmesh
import pytest


def _cavity(problem, Reynolds_number=50.0, formulation="equal_order"):
    problem.add_physics(
        "fluid_dynamics",
        dynamic_viscosity=1.0,
        density=Reynolds_number,
        formulation=formulation,
    )
    problem.add_boundary_condition("Dirichlet", "top", variable="velocity_x", value=1.0)
    problem.add_boundary_condition("Dirichlet", "top", variable="velocity_y", value=0.0)
    for variable in ("velocity_x", "velocity_y"):
        problem.add_boundary_condition(
            "Dirichlet",
            ["left", "right", "bottom"],
            variable=variable,
            value=0.0,
        )


@pytest.mark.parametrize("method", ["fem", "dmcdm"])
@pytest.mark.parametrize(
    "options",
    [
        None,
        {"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"},
        "-ksp_type gmres -ksp_gmres_restart 300 -pc_type ilu",
        # The Schur complement field split of a saddle-point problem: the
        # interlaced velocities (fields 0 and 1) and the pressure (field 2).
        "-ksp_type fgmres -pc_type fieldsplit -pc_fieldsplit_0_fields 0,1 -pc_fieldsplit_1_fields 2 -pc_fieldsplit_type schur -pc_fieldsplit_schur_precondition selfp -fieldsplit_0_pc_type lu -fieldsplit_1_pc_type lu",
    ],
)
def test_PETSc_reproduces_the_direct_solver_on_a_flow(method, options):
    """The cavity in the pressure-velocity formulation is a non-symmetric
    saddle-point problem, the hardest matrix the library produces."""
    mesh = dualmesh.generate_rectangle_mesh(0, 1, 0, 1, 12, 12)
    reference = dualmesh.Problem(mesh, method=method)
    _cavity(reference)
    reference.solve(petsc_options="-ksp_type preonly -pc_type lu")
    problem = dualmesh.Problem(mesh, method=method)
    _cavity(problem)
    result = problem.solve(petsc_options=options)
    for variable in ("velocity_x", "velocity_y", "pressure"):
        assert problem.values(variable) == pytest.approx(reference.values(variable), abs=1e-9)
    assert result.linear_iterations >= result.total_iterations


def _cantilever_3D(mesh):
    problem = dualmesh.Problem(mesh, method="fem")
    solid = problem.add_physics(
        "solid_mechanics",
        formulation="three_dimensional",
        youngs_modulus=2.0e11,
        poissons_ratio=0.3,
    )
    solid.add_boundary_condition("fixed", boundary="left")
    solid.add_boundary_condition("traction = [0, -1e6, 0]", boundary="right")
    return problem


def test_a_large_steel_solid_takes_multigrid_and_reaches_the_direct_answer():
    """Above 4000 unknowns a three-dimensional solid takes GMRES with smoothed aggregation and its
    rigid body modes, at the linear tolerance of Eisenstat and Walker.  Steel carries round-off of
    order 1e-9 of its residual, below which no iteration goes; the inexact Newton steps stop above
    it and give the answer of the direct solver."""
    mesh = dualmesh.generate_box_mesh(0, 1, 0, 0.2, 0, 0.2, 30, 6, 6)
    direct = _cantilever_3D(mesh)
    direct.solve(petsc_options="-ksp_type preonly -pc_type lu", report="none")
    iterative = _cantilever_3D(mesh)
    result = iterative.solve(report="none")
    assert iterative.num_active_dofs() > 4000
    assert result.linear_iterations > 2 * result.total_iterations
    for variable in ("disp_x", "disp_y", "disp_z"):
        expected = direct.values(variable)
        got = iterative.values(variable)
        assert abs(got - expected).max() <= 1e-6 * abs(direct.values("disp_y")).max()
