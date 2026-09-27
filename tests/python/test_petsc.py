# SPDX-License-Identifier: LGPL-2.1-or-later
"""Linear solves with PETSc (``linear_solver="petsc"``).

PETSc is optional.  With it, every solve must give the answer of the built-in
direct solver, whatever Krylov method and preconditioner the options select;
without it, asking for it must say how to get it.  The distributed solves are
tested in ``test_parallel.py``.
"""

from __future__ import annotations

import dualmesh as dm
import numpy as np
import pytest
from dualmesh.problem import petsc_option_string

needs_petsc = pytest.mark.skipif(not dm.have_petsc(), reason="built without PETSc")


def test_petsc_option_strings():
    assert petsc_option_string(None) == ""
    assert petsc_option_string("-ksp_type cg") == "-ksp_type cg"
    assert (
        petsc_option_string({"ksp_type": "cg", "-pc_type": "gamg", "ksp_monitor": None})
        == "-ksp_type cg -pc_type gamg -ksp_monitor"
    )
    assert petsc_option_string({"ksp_view": True, "ksp_monitor": False}) == "-ksp_view"


def _cavity(problem, reynolds_number=50.0, formulation="pressure"):
    dm.physics.add_incompressible_flow(
        problem,
        velocities=["u", "v"],
        dynamic_viscosity=1.0,
        density=reynolds_number,
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


def _heat(problem):
    problem.add_variable("T")
    problem.add_kernel("heat_conduction", "k", variable="T", thermal_conductivity=2.0)
    problem.add_kernel("heat_source", "q", variable="T", heat_source=10.0)
    problem.add_boundary_condition(
        "Dirichlet_boundary_condition", "cold", variable="T", boundary=["left", "bottom"], value=0.0
    )


@needs_petsc
@pytest.mark.parametrize("method", ["fem", "dmcdm", "hfvm", "zfvm"])
@pytest.mark.parametrize(
    "options",
    [
        None,
        {"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"},
        "-ksp_type gmres -ksp_gmres_restart 300 -pc_type ilu",
        # The Schur complement field split of a saddle-point problem: the
        # interlaced velocities (fields 0 and 1) and the pressure (field 2).
        "-ksp_type fgmres -pc_type fieldsplit -pc_fieldsplit_0_fields 0,1 "
        "-pc_fieldsplit_1_fields 2 -pc_fieldsplit_type schur "
        "-pc_fieldsplit_schur_precondition selfp -fieldsplit_0_pc_type lu "
        "-fieldsplit_1_pc_type lu",
    ],
)
def test_petsc_reproduces_the_direct_solver_on_a_flow(method, options):
    """The cavity in the pressure-velocity formulation is a non-symmetric
    saddle-point problem, the hardest matrix the library produces."""
    mesh = dm.generate_rectangle_mesh(0, 1, 0, 1, 12, 12)
    reference = dm.Problem(mesh, method=method)
    _cavity(reference)
    reference.solve(linear_solver="lu")
    problem = dm.Problem(mesh, method=method)
    _cavity(problem)
    result = problem.solve(linear_solver="petsc", petsc_options=options, linear_tolerance=1e-12)
    for variable in ("u", "v", "pressure"):
        assert problem.values(variable) == pytest.approx(reference.values(variable), abs=1e-9)
    assert result.linear_iterations >= result.total_iterations


@needs_petsc
@pytest.mark.parametrize(
    "options",
    [
        None,
        "-ksp_type fgmres -ksp_rtol 1e-12 -pc_type fieldsplit -pc_fieldsplit_0_fields 0,1 "
        "-pc_fieldsplit_1_fields 2 -pc_fieldsplit_type schur -pc_fieldsplit_schur_fact_type full "
        "-pc_fieldsplit_schur_precondition selfp -fieldsplit_0_ksp_type preonly "
        "-fieldsplit_0_pc_type lu -fieldsplit_0_pc_factor_mat_solver_type mumps "
        "-fieldsplit_1_ksp_type gmres -fieldsplit_1_ksp_rtol 1e-10 -fieldsplit_1_pc_type jacobi",
    ],
)
def test_petsc_solves_the_taylor_hood_saddle_point_system(options):
    """The Taylor-Hood system has no pressure diagonal at all.  The matrix
    handed to PETSc stores an explicit zero diagonal, which PETSc's
    factorisations and field splits need in the pattern, and the default
    factorisation is MUMPS, which pivots."""
    mesh = dm.generate_rectangle_mesh(0, 1, 0, 1, 8, 8, element_type="Quad9")
    reference = dm.Problem(mesh, method="fem")
    _cavity(reference, formulation="taylor_hood")
    reference.solve(linear_solver="lu")
    problem = dm.Problem(mesh, method="fem")
    _cavity(problem, formulation="taylor_hood")
    problem.solve(linear_solver="petsc", petsc_options=options, linear_tolerance=1e-12)
    for variable in ("u", "v", "pressure"):
        assert problem.values(variable) == pytest.approx(reference.values(variable), abs=1e-9)


@needs_petsc
@pytest.mark.parametrize("preconditioner", ["gamg", "hypre"])
def test_petsc_algebraic_multigrid_on_conduction(preconditioner):
    """Conjugate gradients with algebraic multigrid on the symmetric positive
    definite matrix of the finite element method: a few tens of iterations."""
    mesh = dm.generate_rectangle_mesh(0, 1, 0, 1, 40, 40)
    reference = dm.Problem(mesh, method="fem")
    _heat(reference)
    reference.solve(linear_solver="lu")
    problem = dm.Problem(mesh, method="fem")
    _heat(problem)
    result = problem.solve(
        linear_solver="petsc", petsc_options={"ksp_type": "cg", "pc_type": preconditioner}
    )
    assert problem.values("T") == pytest.approx(reference.values("T"), abs=1e-9)
    assert result.linear_iterations < 40


@needs_petsc
def test_petsc_reports_a_solve_that_does_not_converge():
    mesh = dm.generate_rectangle_mesh(0, 1, 0, 1, 10, 10)
    problem = dm.Problem(mesh, method="fem")
    _heat(problem)
    with pytest.raises(RuntimeError, match="DIVERGED_ITS"):
        problem.solve(
            linear_solver="petsc",
            petsc_options="-ksp_type richardson -pc_type none -ksp_max_it 3",
        )


@needs_petsc
def test_petsc_options_do_not_leak_between_solves():
    """The options of one solve live in a private database, so a later solve
    without options gets the default (a direct factorisation), not the
    three-iteration limit of the earlier one."""
    mesh = dm.generate_rectangle_mesh(0, 1, 0, 1, 10, 10)
    first = dm.Problem(mesh, method="fem")
    _heat(first)
    with pytest.raises(RuntimeError):
        first.solve(linear_solver="petsc", petsc_options="-ksp_type richardson -ksp_max_it 3")
    second = dm.Problem(mesh, method="fem")
    _heat(second)
    second.solve(linear_solver="petsc")
    assert np.all(np.isfinite(second.values("T")))


@pytest.mark.skipif(dm.have_petsc(), reason="built with PETSc")
def test_petsc_without_petsc_says_how_to_get_it():
    mesh = dm.generate_rectangle_mesh(0, 1, 0, 1, 4, 4)
    problem = dm.Problem(mesh, method="fem")
    _heat(problem)
    with pytest.raises(RuntimeError, match="DUALMESH_ENABLE_PETSC"):
        problem.solve(linear_solver="petsc")
