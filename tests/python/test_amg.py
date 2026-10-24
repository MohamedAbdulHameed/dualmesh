# SPDX-License-Identifier: LGPL-2.1-or-later
"""The algebraic multigrid preconditioner (smoothed aggregation): agreement
with the direct solver, iteration counts that do not grow with the mesh, the
near null space of coupled problems, and the input checks."""

import numpy as np
import pytest

dm = pytest.importorskip("dualmesh")


def cantilever(n, method="fem", element_type="Hex8", **options):
    """A steel bar 4 x 1 x 1 m clamped at x = 0 under its own weight."""
    mesh = dm.generate_box_mesh(
        x_min=0.0,
        x_max=4.0,
        y_min=0.0,
        y_max=1.0,
        z_min=0.0,
        z_max=1.0,
        num_x_elements=4 * n,
        num_y_elements=n,
        num_z_elements=n,
        element_type=element_type,
    )
    problem = dm.Problem(mesh, method=method)
    problem.add_physics(
        "solid_mechanics",
        formulation="three_dimensional",
        youngs_modulus=200.0e9,
        poissons_ratio=0.3,
        body_force=[0.0, 0.0, -7.8e4],
    )
    problem.add_boundary_condition(
        "Dirichlet", "left", variable=["disp_x", "disp_y", "disp_z"], value=0.0
    )
    result = problem.solve(**options)
    return problem, result


def deflection(problem):
    return np.array(problem.values("disp_z"))


@pytest.mark.parametrize("method", ["fem", "dmcdm"])
def test_multigrid_agrees_with_the_direct_solver(method):
    reference, _ = cantilever(6, method, linear_solver="lu")
    solver = "cg" if method == "fem" else "bicgstab"
    problem, result = cantilever(6, method, linear_solver=solver, preconditioner="amg")
    assert result.converged
    w, w_ref = deflection(problem), deflection(reference)
    assert np.max(np.abs(w - w_ref)) < 1e-8 * np.max(np.abs(w_ref))


def test_iterations_do_not_grow_with_the_mesh():
    # The number of unknowns grows about threefold from n = 8 to n = 12, and the
    # conjugate gradient iterations of smoothed aggregation stay about constant
    # (those of ILU(0) grow by half).
    counts = {}
    for n in (8, 12):
        _, result = cantilever(
            n, linear_solver="cg", preconditioner="amg", nonlinear_solver="linear"
        )
        counts[n] = result.linear_iterations
    assert counts[12] <= 30
    assert counts[12] <= counts[8] + 4


def test_automatic_uses_multigrid_on_a_large_three_dimensional_system():
    # 8019 unknowns, above the size up to which "automatic" factorises a
    # three-dimensional system directly.
    problem, result = cantilever(8, nonlinear_solver="linear")
    assert 0 < result.linear_iterations <= 30
    reference, _ = cantilever(8, linear_solver="lu", nonlinear_solver="linear")
    assert np.max(np.abs(deflection(problem) - deflection(reference))) < 1e-8 * np.max(
        np.abs(deflection(reference))
    )


def test_thermoelastic_near_null_space_has_rigid_modes_and_a_constant():
    mesh = dm.generate_box_mesh(
        x_min=0.0,
        x_max=1.0,
        y_min=0.0,
        y_max=1.0,
        z_min=0.0,
        z_max=1.0,
        num_x_elements=2,
        num_y_elements=2,
        num_z_elements=2,
    )
    problem = dm.Problem(mesh, method="fem")
    for name in ("temperature", "disp_x", "disp_y", "disp_z"):
        problem.add_variable(name)
    problem.add_kernel("heat_conduction", variable="temperature", thermal_conductivity=1.0)
    problem.add_property(
        "small_strain_stress",
        displacements=["disp_x", "disp_y", "disp_z"],
        formulation="three_dimensional",
        youngs_modulus=1.0,
        poissons_ratio=0.3,
    )
    for name in ("disp_x", "disp_y", "disp_z"):
        problem.add_kernel(
            "stress_divergence", variable=name, displacements=["disp_x", "disp_y", "disp_z"]
        )
    problem.add_boundary_condition(
        "Dirichlet", "left", variable=["disp_x", "disp_y", "disp_z"], value=0.0
    )
    problem.add_boundary_condition("Dirichlet", "left", variable="temperature", value=0.0)
    problem.initialize()
    B = np.asarray(problem._problem.near_nullspace())
    # Three translations, three rotations and the constant temperature.
    assert B.shape == (4 * mesh.num_nodes, 7)
    assert np.linalg.matrix_rank(B) == 7


def test_multigrid_input_checks():
    with pytest.raises(
        Exception, match="'AMG' works with linear_solver 'CG', 'BiCGStab' or 'automatic'"
    ):
        cantilever(2, linear_solver="gmres", preconditioner="amg")
    with pytest.raises(Exception, match="needs a symmetric matrix"):
        # The cell-centred finite volume method gives a nonsymmetric matrix on
        # a tetrahedral mesh.
        cantilever(3, method="zfvm", element_type="Tet4", linear_solver="cg", preconditioner="amg")
