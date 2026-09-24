# SPDX-License-Identifier: LGPL-2.1-or-later
"""Two-way coupled multiphysics: natural convection in a heated cavity.

The benchmark of de Vahl Davis (1983) couples the momentum equations of an
incompressible flow to the energy equation in both directions: the buoyancy
force makes the temperature drive the flow, and convection makes the flow carry
the heat.  The problem is solved monolithically, by Newton's method with the
exact Jacobian of the whole coupled system, so the tests check both that the
physics is right (the published Nusselt numbers and velocity maxima) and that
the coupling blocks of the Jacobian are exact (the number of Newton
iterations).

The problem definition is the one of ``examples/natural_convection.py``, which
is loaded from the file rather than copied, so that the example and its test
cannot drift apart.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import dualmesh as dm
import pytest

_EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "natural_convection.py"
_spec = importlib.util.spec_from_file_location("natural_convection", _EXAMPLE)
natural_convection = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(natural_convection)


@pytest.mark.parametrize("method", ["dmcdm", "fem", "hfvm"])
@pytest.mark.parametrize("rayleigh_number", [1.0e3, 1.0e4, 1.0e5])
def test_natural_convection_matches_de_vahl_davis(method, rayleigh_number):
    """On a 32 by 32 mesh graded towards the walls, the average Nusselt number
    is within 0.5 % of the benchmark and the velocity maxima within 1.5 %; the
    measured differences are below 0.1 % and 0.6 % for all three methods."""
    problem, result = natural_convection.solve(rayleigh_number, 32, method)
    computed = natural_convection.measure(problem)
    reference = natural_convection.BENCHMARK[rayleigh_number]
    assert computed["nusselt"] == pytest.approx(reference["nusselt"], rel=5e-3)
    assert computed["u_max"] == pytest.approx(reference["u_max"], rel=1.5e-2)
    assert computed["v_max"] == pytest.approx(reference["v_max"], rel=1.5e-2)
    # Energy balance: the heat entering through the hot wall leaves through
    # the cold one, because the top and bottom are adiabatic.
    assert problem.total_reaction("temperature", "right") == pytest.approx(
        -computed["nusselt"], rel=1e-9
    )
    # Newton's method with the exact coupled Jacobian: a handful of
    # iterations per decade of Rayleigh number.
    decades = len([r for r in (1e3, 1e4, 1e5, 1e6) if r <= rayleigh_number])
    assert result.total_iterations <= 8 * decades


def test_the_coupling_is_exact_in_newtons_method():
    """Near the solution Newton's method converges quadratically only if the
    Jacobian is exact, including the blocks that couple the temperature and
    the velocities.  The relative size of the Newton step, s_k, must then
    satisfy s_{k+1} <= C s_k^2 once it is small.  The residual itself is not
    used, because with a penalty parameter of 10^7 it reaches its round-off
    floor (about 10^-7) one step before the solution stops changing.  The
    measured steps on the last load step are 2.3e-3, 2.5e-6 and 5.1e-12, so
    C is about 0.5 and 0.8."""
    problem, result = natural_convection.solve(1.0e4, 16, "dmcdm")
    steps = [r.step_norm for r in result.history if r.load_factor == 1.0 and r.step_norm > 1e-15]
    ratios = [steps[i + 1] / steps[i] ** 2 for i in range(len(steps) - 1) if steps[i] < 1e-2]
    assert len(ratios) >= 2, "the last load step did not reach the quadratic regime"
    assert max(ratios) < 10.0


def test_penalty_flow_is_refused_by_the_cell_centred_method():
    """The penalty formulation needs the incompressibility constraint imposed
    once per element by a reduced rule; the cell-centred finite volume method
    imposes it at every face and locks.  It is refused with an explanation."""
    mesh = dm.generate_rectangle_mesh(0.0, 1.0, 0.0, 1.0, 4, 4)
    problem = dm.Problem(mesh, method="zfvm")
    dm.physics.add_incompressible_flow(problem, velocities=["u", "v"])
    for variable in ("u", "v"):
        problem.add_boundary_condition(
            "DirichletBC",
            f"walls_{variable}",
            variable=variable,
            boundary=["left", "right", "bottom", "top"],
            value=0.0,
        )
    with pytest.raises(ValueError, match="cell-centred finite volume method"):
        problem.solve()


def test_the_vertex_centred_method_does_not_lock_the_penalty_term():
    """The vertex-centred method replaces the interface gradient by a two-point
    difference, but not where a kernel asks for reduced integration: there the
    element's mean gradient is the point.  With the correction applied to the
    penalty term the flow in a lid-driven cavity was five orders of magnitude
    too small; without it the three node-based methods agree."""
    values = {}
    for method in ("dmcdm", "hfvm"):
        mesh = dm.generate_rectangle_mesh(0.0, 1.0, 0.0, 1.0, 16, 16)
        problem = dm.Problem(mesh, method=method)
        dm.physics.add_incompressible_flow(problem, velocities=["u", "v"])
        for variable in ("u", "v"):
            problem.add_boundary_condition(
                "DirichletBC",
                f"walls_{variable}",
                variable=variable,
                boundary=["left", "right", "bottom"],
                value=0.0,
            )
        problem.add_boundary_condition(
            "DirichletBC", "lid", variable="u", boundary="top", value=1.0
        )
        problem.add_boundary_condition(
            "DirichletBC", "lid_v", variable="v", boundary="top", value=0.0
        )
        problem.solve()
        values[method] = problem.sample("u", [[0.5, 0.25]])[0]
    assert values["hfvm"] == pytest.approx(values["dmcdm"], rel=0.05)
    assert values["dmcdm"] < -0.1


def test_the_buoyancy_helper_adds_the_same_kernels():
    """``dm.physics.add_boussinesq_buoyancy`` adds one kernel per component
    along which gravity acts, and gives the example's answer exactly."""
    reference, _ = natural_convection.solve(1.0e4, 16)
    problem = dm.Problem(natural_convection.cavity_mesh(16))
    velocities = dm.physics.add_incompressible_flow(
        problem, velocities=["u", "v"], dynamic_viscosity=0.71, density=1.0, penalty_parameter=1e7
    )
    problem.add_variable("temperature")
    problem.add_kernel("HeatConduction", "conduction", variable="temperature")
    problem.add_kernel(
        "HeatConvection", "convection", variable="temperature", velocities=velocities
    )
    added = dm.physics.add_boussinesq_buoyancy(
        problem,
        "temperature",
        velocities,
        gravity=[0.0, -1.0],
        thermal_expansion=1.0e4 * 0.71,
        scale_with_load=True,
    )
    assert added == ["v"]
    for variable in velocities:
        problem.add_boundary_condition(
            "DirichletBC",
            f"no_slip_{variable}",
            variable=variable,
            boundary=["left", "right", "bottom", "top"],
            value=0.0,
        )
    problem.add_boundary_condition(
        "DirichletBC", "hot", variable="temperature", boundary="left", value=0.5
    )
    problem.add_boundary_condition(
        "DirichletBC", "cold", variable="temperature", boundary="right", value=-0.5
    )
    problem.solve(load_factors=[0.1, 1.0], max_iterations=40)
    assert problem.values("temperature") == pytest.approx(
        reference.values("temperature"), abs=1e-10
    )
