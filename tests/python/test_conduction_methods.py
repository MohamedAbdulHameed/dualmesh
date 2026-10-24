# SPDX-License-Identifier: LGPL-2.1-or-later
"""Conduction with both methods: a nonlinear conductivity, a transient slab, a convecting end
and a jump of the conductivity between blocks, each against its closed-form solution."""

from __future__ import annotations

import dualmesh as dm
import numpy as np
import pytest


@pytest.mark.parametrize("method", ["FEM", "DMCDM"])
@pytest.mark.parametrize("solver", ["newton", "picard"])
def test_nonlinear_conduction(method, solver):
    """-d/dx(k(T) dT/dx) = 0 with k(T) = 1 + T/2, T(0) = 0, T(1) = 1.

    The Kirchhoff transform gives T + T^2/4 = x (1 + 1/4) in closed form.
    """
    mesh = dm.generate_line_mesh(start=0.0, end=1.0, num_elements=40)
    problem = dm.Problem(mesh, method=method)
    problem.add_variable("temperature")
    problem.add_kernel(
        "heat_conduction",
        variable="temperature",
        thermal_conductivity=1.0,
        temperature_polynomial=[1.0, 0.5],
    )
    problem.add_boundary_condition("Dirichlet", "left", variable="temperature", value=0.0)
    problem.add_boundary_condition("Dirichlet", "right", variable="temperature", value=1.0)
    result = problem.solve(nonlinear_solver=solver)
    assert result.converged
    x = problem.entity_points()[:, 0]
    exact = (-1.0 + np.sqrt(1.0 + x * (1.0 + 0.25))) / 0.5
    assert problem.values("temperature") == pytest.approx(exact, abs=1e-4)


@pytest.mark.parametrize("method", ["FEM", "DMCDM"])
def test_transient_slab(method):
    """T_t = T_xx with T(x, 0) = sin(pi x) decays as exp(-pi^2 t) sin(pi x)."""
    mesh = dm.generate_line_mesh(start=0.0, end=1.0, num_elements=40)
    problem = dm.Problem(mesh, method=method)
    problem.add_variable("temperature", initial_condition=lambda x, y, z, t: np.sin(np.pi * x))
    problem.add_kernel("diffusion", variable="temperature")
    problem.add_kernel("time_derivative", variable="temperature")
    problem.add_boundary_condition(
        "Dirichlet",
        ["left", "right"],
        variable="temperature",
        value=0.0,
    )
    problem.solve_transient(end_time=0.1, time_step=0.001, implicitness=0.5)
    x = problem.entity_points()[:, 0]
    exact = np.exp(-(np.pi**2) * 0.1) * np.sin(np.pi * x)
    assert problem.values("temperature") == pytest.approx(exact, abs=3e-4)


@pytest.mark.parametrize("method", ["FEM", "DMCDM"])
def test_convective_boundary_condition(method):
    """A fin with a convecting end: both methods converge to the same
    analytical solution, which checks the sign of the Robin condition."""
    length, base, transfer, m = 1.0, 100.0, 2.0, 2.0

    def exact(x):
        return (
            base
            * (np.cosh(m * (length - x)) + (transfer / m) * np.sinh(m * (length - x)))
            / (np.cosh(m * length) + (transfer / m) * np.sinh(m * length))
        )

    errors = []
    for n in (8, 16, 32):
        mesh = dm.generate_line_mesh(start=0.0, end=length, num_elements=n)
        problem = dm.Problem(mesh, method=method)
        problem.add_variable("u")
        problem.add_kernel("diffusion", variable="u", diffusivity=1.0)
        problem.add_kernel("reaction", variable="u", coeff=m * m)
        problem.add_boundary_condition("Dirichlet", "left", variable="u", value=base)
        problem.add_boundary_condition(
            "Robin",
            "right",
            variable="u",
            transfer_coeff=transfer,
        )
        problem.solve()
        errors.append(np.abs(problem.values("u") - exact(problem.entity_points()[:, 0])).max())
    rates = [np.log2(errors[i] / errors[i + 1]) for i in range(2)]
    assert min(rates) > 1.8


# ---------------------------------------------------------------------------
# Example 3.3.1 and Table 3.3.1: one-dimensional problem, both formulations
# ---------------------------------------------------------------------------
def _two_block_line(n):
    """The unit interval with a block 'inner' on x < 0.6 and 'outer' beyond."""
    x = np.linspace(0.0, 1.0, n + 1)
    cells = np.array([[i, i + 1] for i in range(n)])
    blocks = [0 if (i + 0.5) / n < 0.6 else 1 for i in range(n)]
    mesh = dm.mesh_from_arrays(x[:, None], cells, element_type="Edge2", blocks=blocks, dimension=1)
    mesh.set_block_name(0, "inner")
    mesh.set_block_name(1, "outer")
    mesh.add_sideset_by_predicate("left", lambda x, y, z: x < 0.5)
    mesh.add_sideset_by_predicate("right", lambda x, y, z: x > 0.5)
    return mesh


@pytest.mark.parametrize("method", ["FEM", "DMCDM"])
def test_a_jump_of_the_conductivity_between_blocks_is_exact_at_the_nodes(method):
    """-(k u')' = 1 with k = 1 for x < 0.6 and k = 3 beyond, u = 0 at both
    ends: the exact solution is piecewise quadratic with a continuous flux,
    which both methods hold at the nodes of the face between the blocks."""
    a = 0.6
    matrix = np.array([[a, -(a - 1.0)], [1.0, -3.0]])
    rhs = np.array([-((a - 1.0) ** 2) / 6.0 + a**2 / 2.0, -(a - 1.0) + a])
    c1, c2 = np.linalg.solve(matrix, rhs)
    errors = []
    for n in (10, 20, 40):
        mesh = _two_block_line(n)
        problem = dm.Problem(mesh, method=method)
        heat = problem.add_physics("heat_transfer", heat_source=1.0)
        problem.add_property(
            "constant",
            block=["inner"],
            properties={"thermal_conductivity": 1.0},
        )
        problem.add_property(
            "constant",
            block=["outer"],
            properties={"thermal_conductivity": 3.0},
        )
        heat.add_boundary_condition("temperature = 0", boundary="left")
        heat.add_boundary_condition("temperature = 0", boundary="right")
        problem.solve()
        x = problem.entity_points()[:, 0]
        exact = np.where(x < a, -(x**2) / 2.0 + c1 * x, -((x - 1.0) ** 2) / 6.0 + c2 * (x - 1.0))
        errors.append(np.abs(problem.values("temperature") - exact).max())
    # Both methods are exact at the nodes in one dimension, where the nodes include the face
    # between the blocks.
    assert max(errors) < 1e-12, errors
