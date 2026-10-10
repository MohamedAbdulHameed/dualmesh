# SPDX-License-Identifier: LGPL-2.1-or-later
"""The projection time integration of incompressible flow (time_integration="projection").

The decaying Taylor-Green vortex, u = (-cos x sin y, sin x cos y) exp(-2 nu t), checks the accuracy in space, with its velocity prescribed on the walls of [0, 2 pi]^2 and in a periodic box.
The stagnation flow u = g(t) (x, -y), whose velocity is linear and whose pressure is quadratic, is represented exactly by quadratic elements, so its error is that of the time integration alone.
"""

from __future__ import annotations

import dualmesh as dm
import numpy as np
import pytest

NU = 0.1


METHODS = ["fem", "dmcdm", "hfvm"]


def taylor_green(
    n,
    dt,
    order=2,
    element="Quad4",
    periodic=False,
    end=0.5,
    distributed=False,
    method="fem",
    **solve,
):
    mesh = dm.generate_rectangle_mesh(0.0, 2 * np.pi, 0.0, 2 * np.pi, n, n, element_type=element)
    problem = dm.Problem(mesh, method=method, distributed=distributed)
    problem.add_physics(
        "incompressible_flow",
        "flow",
        velocities=["u", "v"],
        density=1.0,
        dynamic_viscosity=NU,
        formulation="pressure",
        time_integration="projection",
        time_order=order,
    )
    decay = f"exp(-2*{NU}*t)"
    if periodic:
        problem.add_boundary_condition(
            "periodic_boundary_condition", "periodic_x", primary="left", secondary="right"
        )
        problem.add_boundary_condition(
            "periodic_boundary_condition", "periodic_y", primary="bottom", secondary="top"
        )
    else:
        for name, value in (("u", f"-cos(x)*sin(y)*{decay}"), ("v", f"sin(x)*cos(y)*{decay}")):
            problem.add_boundary_condition(
                "Dirichlet_boundary_condition",
                f"walls_{name}",
                variable=name,
                boundary=["left", "right", "bottom", "top"],
                value=value,
            )
    problem.initialize()
    points = np.asarray(mesh.points())
    x, y = points[:, 0], points[:, 1]
    problem.set_values("u", -np.cos(x) * np.sin(y))
    problem.set_values("v", np.sin(x) * np.cos(y))
    result = problem.solve_transient(end_time=end, time_step=dt, report="none", **solve)
    factor = np.exp(-2 * NU * end)
    exact_p = -0.25 * (np.cos(2 * x) + np.cos(2 * y)) * factor**2
    p = problem.values("pressure")
    velocity_error = np.abs(problem.values("u") + np.cos(x) * np.sin(y) * factor).max()
    pressure_error = np.abs((p - p.mean()) - (exact_p - exact_p.mean())).max()
    return velocity_error, pressure_error, result, problem


def rates(errors):
    return [np.log2(a / b) for a, b in zip(errors, errors[1:])]


def test_the_coefficients_are_exact_on_polynomials_for_variable_steps():
    # Constant steps give the textbook coefficients.
    beta, alpha = dm._core.ProjectionSolver.coefficients([3.0, 2.0, 1.0])
    assert beta == pytest.approx([1.5, -2.0, 0.5])
    assert alpha[1:] == pytest.approx([2.0, -1.0])
    beta, alpha = dm._core.ProjectionSolver.coefficients([4.0, 3.0, 2.0, 1.0])
    assert beta == pytest.approx([11 / 6, -3.0, 1.5, -1 / 3])
    assert alpha[1:] == pytest.approx([3.0, -3.0, 1.0])
    # Unequal steps: the backward difference differentiates, and the extrapolation reproduces, every polynomial of degree k and k - 1.
    times = [1.0, 0.85, 0.6, 0.5]
    beta, alpha = dm._core.ProjectionSolver.coefficients(times)
    dt = times[0] - times[1]
    for m in range(4):
        derivative = sum(b * t**m for b, t in zip(beta, times)) / dt
        assert derivative == pytest.approx(m * times[0] ** (m - 1) if m else 0.0, abs=1e-12)
    for m in range(3):
        assert sum(a * t**m for a, t in zip(alpha[1:], times[1:])) == pytest.approx(times[0] ** m)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("element", ["Quad4", "Tri3"])
def test_the_vortex_with_walls_converges_at_second_order_in_space(method, element):
    errors = [taylor_green(n, 0.8 / n, element=element, method=method)[:2] for n in (8, 16, 32)]
    velocity_rates = rates([e[0] for e in errors])
    assert min(velocity_rates[1:]) > 1.8, velocity_rates
    assert errors[-1][1] < errors[0][1] / 8


@pytest.mark.parametrize("method", METHODS)
def test_the_vortex_in_a_periodic_box_converges_at_second_order(method):
    errors = [taylor_green(n, 0.4 / n, periodic=True, method=method)[:2] for n in (8, 16, 32, 64)]
    assert min(rates([e[0] for e in errors])) > 1.9
    # The pressure error falls irregularly on these grids, at second order over the whole range.
    assert np.log2(errors[0][1] / errors[-1][1]) / 3 > 1.9


def stagnation(dt, order, method="fem"):
    """u = g(t) (x, -y), p = -(g' + g^2) x^2 / 2 + (g' - g^2) y^2 / 2 with g = 1 + sin(2 t) / 2, on quadratic elements, which represent it exactly."""
    mesh = dm.generate_rectangle_mesh(0.0, 1.0, 0.0, 1.0, 3, 3, element_type="Quad9")
    problem = dm.Problem(mesh, method=method, distributed=False)
    problem.add_physics(
        "incompressible_flow",
        "flow",
        velocities=["u", "v"],
        density=1.0,
        dynamic_viscosity=0.05,
        formulation="pressure",
        time_integration="projection",
        time_order=order,
    )
    g = "(1+0.5*sin(2*t))"
    walls = ["left", "right", "bottom", "top"]
    problem.add_boundary_condition(
        "Dirichlet_boundary_condition", "u_walls", variable="u", boundary=walls, value=f"{g}*x"
    )
    problem.add_boundary_condition(
        "Dirichlet_boundary_condition", "v_walls", variable="v", boundary=walls, value=f"-{g}*y"
    )
    problem.initialize()
    points = np.asarray(mesh.points())
    x, y = points[:, 0], points[:, 1]
    problem.set_values("u", x)
    problem.set_values("v", -y)
    problem.solve_transient(end_time=1.0, time_step=dt, report="none")
    G, Gp = 1 + 0.5 * np.sin(2.0), np.cos(2.0)
    exact = -(Gp + G * G) * x * x / 2 + (Gp - G * G) * y * y / 2
    p = problem.values("pressure")
    return np.abs((p - p.mean()) - (exact - exact.mean())).max()


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("order, expected", [(1, 0.9), (2, 1.95), (3, 2.6)])
def test_the_time_integration_has_its_order(method, order, expected):
    errors = [stagnation(dt, order, method) for dt in (0.05, 0.025, 0.0125)]
    assert rates(errors)[-1] > expected, errors


@pytest.mark.parametrize("method", METHODS)
def test_small_steps_keep_the_accuracy_of_equal_order_elements(method):
    """The splitting weighs the divergence by beta_0 / dt, which would vanish as a stabilization with the step; the cap at 1 / tau keeps the answer of a small step that of a large one."""
    large = taylor_green(16, 0.1, method=method)[:2]
    small = taylor_green(16, 0.004, method=method)[:2]
    assert small[0] < 1.1 * large[0] and small[1] < 1.2 * large[1]


def test_the_cfl_stepper_keeps_the_courant_number():
    _, _, result, _ = taylor_green(
        24,
        0.01,
        periodic=True,
        end=1.0,
        time_stepper="cfl",
        courant_number=0.5,
        growth_factor=2.0,
    )
    steps = [dt for _, dt in result.step_history]
    # The step doubles from the first one, then settles at the step of the Courant number, h / 2 at the largest speed, one.
    assert steps[1] == pytest.approx(2 * steps[0])
    assert np.median(steps) == pytest.approx(0.5 * 2 * np.pi / 24, rel=0.05)
    # The matrices follow the step: the error is that of a fixed step of the same size.
    changing = taylor_green(
        24, 0.01, periodic=True, end=1.0, time_stepper="cfl", courant_number=0.5
    )
    fixed = taylor_green(24, float(np.median(steps)), periodic=True, end=1.0)
    assert changing[0] == pytest.approx(fixed[0], rel=0.15)


@pytest.mark.parametrize("method", METHODS)
def test_the_distributed_path_gives_the_serial_answer(method):
    serial = taylor_green(8, 0.1, method=method)[3]
    distributed = taylor_green(8, 0.1, distributed=True, method=method)[3]
    for name in ("u", "v"):
        assert np.asarray(distributed.gathered_values(name)) == pytest.approx(
            serial.values(name), abs=1e-9
        )


def flow(**parameters):
    mesh = dm.generate_rectangle_mesh(0.0, 1.0, 0.0, 1.0, 4, 4)
    problem = dm.Problem(mesh, method=parameters.pop("method", "fem"), distributed=False)
    problem.add_physics(
        "incompressible_flow",
        "flow",
        velocities=["u", "v"],
        density=parameters.pop("density", 1.0),
        dynamic_viscosity=0.1,
        time_integration="projection",
        **parameters,
    )
    return problem


@pytest.mark.parametrize(
    "parameters, message",
    [
        ({"formulation": "penalty"}, "formulation='pressure'"),
        ({"formulation": "pressure", "density": 0.0}, "positive density"),
        ({"formulation": "pressure", "time_order": 4}, "time_order"),
    ],
)
def test_unsupported_settings_are_refused(parameters, message):
    with pytest.raises(ValueError, match=message):
        flow(**parameters)


def test_the_cell_centered_method_is_refused():
    problem = flow(formulation="pressure", method="zfvm")
    with pytest.raises(ValueError, match="fem, dmcdm and hfvm"):
        problem.solve_transient(end_time=0.1, time_step=0.05, report="none")


@pytest.mark.parametrize(
    "kind, parameters",
    [
        ("body_force", {"variable": "u", "value": 1.0}),
        ("Neumann_boundary_condition", {"variable": "u", "boundary": "right", "flux": 1.0}),
    ],
)
def test_a_term_the_projection_would_ignore_is_refused(kind, parameters):
    """The projection time integration assembles the flow itself, so a term added on its own would be dropped silently; it is refused instead."""
    problem = flow(formulation="pressure")
    if kind == "body_force":
        problem.add_kernel(kind, "extra", **parameters)
    else:
        problem.add_boundary_condition(kind, "extra", **parameters)
    with pytest.raises(ValueError, match="acts on the flow"):
        problem.solve_transient(end_time=0.1, time_step=0.05, report="none")


def test_the_body_force_of_the_flow_is_applied():
    """A uniform body force on a fluid at rest in a periodic box accelerates it uniformly: u = f t / rho."""
    mesh = dm.generate_rectangle_mesh(0.0, 1.0, 0.0, 1.0, 4, 4)
    problem = dm.Problem(mesh, method="fem", distributed=False)
    problem.add_physics(
        "incompressible_flow",
        "flow",
        velocities=["u", "v"],
        density=2.0,
        dynamic_viscosity=0.1,
        formulation="pressure",
        time_integration="projection",
        body_force=[3.0, 0.0],
    )
    problem.add_boundary_condition(
        "periodic_boundary_condition", "periodic_x", primary="left", secondary="right"
    )
    problem.add_boundary_condition(
        "periodic_boundary_condition", "periodic_y", primary="bottom", secondary="top"
    )
    problem.solve_transient(end_time=0.5, time_step=0.1, report="none")
    assert problem.values("u") == pytest.approx(3.0 * 0.5 / 2.0, rel=1e-9)
    assert np.abs(problem.values("v")).max() < 1e-12


@pytest.mark.parametrize("method", METHODS)
def test_the_reactions_balance_the_body_force(method):
    """A body force f drives the flow through a channel periodic in x; at the steady state the walls carry the whole force, so the force on them, minus the sum of the reactions, is f times the area of the channel."""
    mesh = dm.generate_rectangle_mesh(0.0, 2.0, 0.0, 1.0, 8, 6, element_type="Tri3")
    problem = dm.Problem(mesh, method=method, distributed=False)
    problem.add_physics(
        "incompressible_flow",
        "flow",
        velocities=["u", "v"],
        density=1.0,
        dynamic_viscosity=1.0,
        formulation="pressure",
        time_integration="projection",
        body_force=[3.0, 0.0],
    )
    problem.add_boundary_condition(
        "periodic_boundary_condition", "periodic_x", primary="left", secondary="right"
    )
    for name in ("u", "v"):
        problem.add_boundary_condition(
            "Dirichlet_boundary_condition",
            f"walls_{name}",
            variable=name,
            boundary=["bottom", "top"],
            value=0.0,
        )
    problem.solve_transient(end_time=4.0, time_step=0.05, report="none")
    force = -problem.total_reaction("u", "bottom") - problem.total_reaction("u", "top")
    assert force == pytest.approx(3.0 * 2.0 * 1.0, rel=1e-6)
    assert problem.total_reaction("v", "bottom") + problem.total_reaction(
        "v", "top"
    ) == pytest.approx(0.0, abs=1e-8)
