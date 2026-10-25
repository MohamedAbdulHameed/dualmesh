# SPDX-License-Identifier: LGPL-2.1-or-later
"""Poiseuille flow, the test of the consistency of the stabilised residual.

In a channel 4 x 1 with mu = 1, a parabolic inflow u = 4 y (1 - y) and a do-nothing outlet
(viscous_form="laplacian", whose natural condition mu du/dn - p n = 0 Poiseuille flow satisfies),
the exact flow is u = 4 y (1 - y), v = 0, p = 8 (4 - x).  The velocity does not depend on x and
the pressure is linear, so a scheme whose stabilisation vanishes for the exact solution is exact at
the nodes.  Quadratic elements give the viscous term - mu Laplacian(v) of the residual and are
exact.  Inside a linear element that term vanishes, so the residual of the exact solution is
- mu Laplacian(v) = 8, and the error falls at second order in the velocity (the standard
residual, as MOOSE and COMSOL take it; decision_2026-10-25_consistent_SUPG.md).
"""

from __future__ import annotations

import dualmesh as dm
import numpy as np
import pytest


def poiseuille(method, n, element_type, density=0.0):
    """The largest nodal errors of u, v and p on n cells across."""
    mesh = dm.generate_rectangle_mesh(0, 4, 0, 1, 4 * n, n, element_type=element_type)
    problem = dm.Problem(mesh, method=method)
    flow = problem.add_physics(
        "fluid_dynamics",
        density=density,
        dynamic_viscosity=1.0,
        formulation="pressure",
        viscous_form="laplacian",
    )
    flow.add_boundary_condition(["velocity_x = 4*y*(1 - y)", "velocity_y = 0"], boundary="left")
    flow.add_boundary_condition(["velocity_x = 0", "velocity_y = 0"], boundary=["top", "bottom"])
    problem.solve(report="none")
    x = problem.entity_points()
    return (
        np.abs(problem.values("velocity_x") - 4 * x[:, 1] * (1 - x[:, 1])).max(),
        np.abs(problem.values("velocity_y")).max(),
        np.abs(problem.values("pressure") - 8 * (4 - x[:, 0])).max(),
    )


@pytest.mark.parametrize("method", ["FEM", "DMCDM"])
@pytest.mark.parametrize("element_type", ["Quad9", "Tri6"])
@pytest.mark.parametrize("density", [0.0, 10.0])
def test_quadratic_elements_are_exact(method, element_type, density):
    """Measured: 4e-14 in the velocity, 5e-12 in the pressure."""
    for n in (2, 4):
        u, v, p = poiseuille(method, n, element_type, density)
        assert max(u, v) < 1e-11 and p < 1e-9, (n, u, v, p)


@pytest.mark.parametrize("method", ["FEM", "DMCDM"])
@pytest.mark.parametrize("element_type", ["Quad4", "Tri3"])
def test_linear_elements_converge_at_second_order(method, element_type):
    """The documented inconsistency of linear elements: the nodal velocity error 1.8e-2, 4.4e-3,
    1.1e-3 on 8, 16, 32 cells, falling at second order, and the pressure at first order (1.2,
    0.48, 0.22)."""
    errors = [poiseuille(method, n, element_type) for n in (8, 16, 32)]
    u = [e[0] for e in errors]
    p = [e[2] for e in errors]
    assert min(np.log2(u[i] / u[i + 1]) for i in range(2)) > 1.9, u
    assert min(np.log2(p[i] / p[i + 1]) for i in range(2)) > 1.0, p
    assert u[-1] < 2e-3, u
