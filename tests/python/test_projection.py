# SPDX-License-Identifier: LGPL-2.1-or-later
"""The projection time integration of incompressible flow (time_integration="projection").

The decaying Taylor-Green vortex, u = (-cos x sin y, sin x cos y) exp(-2 nu t), checks the accuracy in space, with its velocity prescribed on the walls of [0, 2 pi]^2 and in a periodic box.
The stagnation flow u = g(t) (x, -y), whose velocity is linear and whose pressure is quadratic, is represented exactly by quadratic elements, so its error is that of the time integration alone.
"""

from __future__ import annotations

import dualmesh
import numpy
import pytest

nu = 0.1


methods = ["fem", "dmcdm"]


def Taylor_green(
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
    mesh = dualmesh.generate_rectangle_mesh(
        0.0, 2 * numpy.pi, 0.0, 2 * numpy.pi, n, n, element_type=element
    )
    problem = dualmesh.Problem(mesh, method=method, distributed=distributed)
    problem.add_physics(
        "fluid_dynamics",
        density=1.0,
        dynamic_viscosity=nu,
        formulation="equal_order",
        time_integration="projection",
        time_order=order,
    )
    decay = f"exp(-2*{nu}*t)"
    if periodic:
        problem.add_boundary_condition("periodic", primary="left", secondary="right")
        problem.add_boundary_condition("periodic", primary="bottom", secondary="top")
    else:
        for name, value in (
            ("velocity_x", f"-cos(x)*sin(y)*{decay}"),
            ("velocity_y", f"sin(x)*cos(y)*{decay}"),
        ):
            problem.add_boundary_condition(
                "Dirichlet",
                ["left", "right", "bottom", "top"],
                variable=name,
                value=value,
            )
    problem.initialize()
    points = numpy.asarray(mesh.points())
    x, y = points[:, 0], points[:, 1]
    problem.set_values("velocity_x", -numpy.cos(x) * numpy.sin(y))
    problem.set_values("velocity_y", numpy.sin(x) * numpy.cos(y))
    result = problem.solve_transient(end_time=end, time_step=dt, report="none", **solve)
    factor = numpy.exp(-2 * nu * end)
    exact_p = -0.25 * (numpy.cos(2 * x) + numpy.cos(2 * y)) * factor**2
    p = problem.values("pressure")
    velocity_error = numpy.abs(
        problem.values("velocity_x") + numpy.cos(x) * numpy.sin(y) * factor
    ).max()
    pressure_error = numpy.abs((p - p.mean()) - (exact_p - exact_p.mean())).max()
    return velocity_error, pressure_error, result, problem


def rates(errors):
    return [numpy.log2(a / b) for a, b in zip(errors, errors[1:])]


def test_the_coefficients_are_exact_on_polynomials_for_variable_steps():
    # Constant steps give the textbook coefficients.
    beta, alpha = dualmesh._core.ProjectionSolver.coefficients([3.0, 2.0, 1.0])
    assert beta == pytest.approx([1.5, -2.0, 0.5])
    assert alpha[1:] == pytest.approx([2.0, -1.0])
    beta, alpha = dualmesh._core.ProjectionSolver.coefficients([4.0, 3.0, 2.0, 1.0])
    assert beta == pytest.approx([11 / 6, -3.0, 1.5, -1 / 3])
    assert alpha[1:] == pytest.approx([3.0, -3.0, 1.0])
    # Unequal steps: the backward difference differentiates, and the extrapolation reproduces, every polynomial of degree k and k - 1.
    times = [1.0, 0.85, 0.6, 0.5]
    beta, alpha = dualmesh._core.ProjectionSolver.coefficients(times)
    dt = times[0] - times[1]
    for m in range(4):
        derivative = sum(b * t**m for b, t in zip(beta, times)) / dt
        assert derivative == pytest.approx(m * times[0] ** (m - 1) if m else 0.0, abs=1e-12)
    for m in range(3):
        assert sum(a * t**m for a, t in zip(alpha[1:], times[1:])) == pytest.approx(times[0] ** m)


@pytest.mark.parametrize("method", methods)
@pytest.mark.parametrize("element", ["Quad4", "Tri3"])
def test_the_vortex_with_walls_converges_at_second_order_in_space(method, element):
    """The velocity at second order over the two refinements (on Tri3 the rates are 2.05 and
    1.78, and 1.96 on the next mesh), the pressure by at least 7 (measured 7.3 and 7.7 on Tri3,
    whose alternating triangles converge less regularly in pressure than the quadrilaterals)."""
    errors = [Taylor_green(n, 0.8 / n, element=element, method=method)[:2] for n in (8, 16, 32)]
    velocity_order = numpy.log2(errors[0][0] / errors[-1][0]) / 2
    assert velocity_order > 1.85, rates([e[0] for e in errors])
    assert errors[-1][1] < errors[0][1] / 7


@pytest.mark.parametrize("method", methods)
def test_the_vortex_in_a_periodic_box_converges_at_second_order(method):
    errors = [Taylor_green(n, 0.4 / n, periodic=True, method=method)[:2] for n in (8, 16, 32, 64)]
    assert min(rates([e[0] for e in errors])) > 1.9
    # The pressure error falls irregularly on the coarse grids (measured 1.5 to 3.9 from 8 to 16
    # elements) and at second order on the finer ones.
    assert min(rates([e[1] for e in errors])[1:]) > 1.9


def stagnation(dt, order, method="fem"):
    """u = g(t) (x, -y), p = -(g' + g^2) x^2 / 2 + (g' - g^2) y^2 / 2 with g = 1 + sin(2 t) / 2, on quadratic elements, which represent it exactly."""
    mesh = dualmesh.generate_rectangle_mesh(0.0, 1.0, 0.0, 1.0, 3, 3, element_type="Quad9")
    problem = dualmesh.Problem(mesh, method=method, distributed=False)
    problem.add_physics(
        "fluid_dynamics",
        density=1.0,
        dynamic_viscosity=0.05,
        formulation="equal_order",
        time_integration="projection",
        time_order=order,
    )
    g = "(1+0.5*sin(2*t))"
    walls = ["left", "right", "bottom", "top"]
    problem.add_boundary_condition("Dirichlet", walls, variable="velocity_x", value=f"{g}*x")
    problem.add_boundary_condition("Dirichlet", walls, variable="velocity_y", value=f"-{g}*y")
    problem.initialize()
    points = numpy.asarray(mesh.points())
    x, y = points[:, 0], points[:, 1]
    problem.set_values("velocity_x", x)
    problem.set_values("velocity_y", -y)
    problem.solve_transient(end_time=1.0, time_step=dt, report="none")
    G, G_p = 1 + 0.5 * numpy.sin(2.0), numpy.cos(2.0)
    exact = -(G_p + G * G) * x * x / 2 + (G_p - G * G) * y * y / 2
    p = problem.values("pressure")
    return numpy.abs((p - p.mean()) - (exact - exact.mean())).max()


@pytest.mark.parametrize("method", methods)
@pytest.mark.parametrize("order, expected", [(1, 0.9), (2, 1.95), (3, 2.6)])
def test_the_time_integration_has_its_order(method, order, expected):
    errors = [stagnation(dt, order, method) for dt in (0.05, 0.025, 0.0125)]
    assert rates(errors)[-1] > expected, errors


@pytest.mark.parametrize("method", methods)
def test_small_steps_keep_the_accuracy_of_equal_order_elements(method):
    """The splitting weighs the divergence by beta_0 / dt, which would vanish as a stabilization with the step; the cap at 1 / tau keeps the answer of a small step that of a large one."""
    large = Taylor_green(16, 0.1, method=method)[:2]
    small = Taylor_green(16, 0.004, method=method)[:2]
    assert small[0] < 1.1 * large[0] and small[1] < 1.2 * large[1]


@pytest.mark.parametrize("method", methods)
def test_the_distributed_path_gives_the_serial_answer(method):
    serial = Taylor_green(8, 0.1, method=method)[3]
    distributed = Taylor_green(8, 0.1, distributed=True, method=method)[3]
    for name in ("velocity_x", "velocity_y"):
        assert numpy.asarray(distributed.gathered_values(name)) == pytest.approx(
            serial.values(name), abs=1e-9
        )


def flow(**parameters):
    mesh = dualmesh.generate_rectangle_mesh(0.0, 1.0, 0.0, 1.0, 4, 4)
    problem = dualmesh.Problem(mesh, method=parameters.pop("method", "fem"), distributed=False)
    problem.add_physics(
        "fluid_dynamics",
        density=parameters.pop("density", 1.0),
        dynamic_viscosity=0.1,
        time_integration="projection",
        **parameters,
    )
    return problem


@pytest.mark.parametrize(
    "parameters, message",
    [
        ({"formulation": "penalty"}, "formulation='equal_order'"),
        ({"formulation": "equal_order", "density": 0.0}, "positive density"),
        ({"formulation": "equal_order", "time_order": 4}, "time_order"),
    ],
)
def test_unsupported_settings_are_refused(parameters, message):
    with pytest.raises(ValueError, match=message):
        flow(**parameters)


@pytest.mark.parametrize(
    "kind, parameters",
    [
        ("body_force", {"variable": "velocity_x", "value": 1.0}),
        ("Neumann", {"variable": "velocity_x", "boundary": "right", "flux": 1.0}),
    ],
)
def test_a_term_the_projection_would_ignore_is_refused(kind, parameters):
    """The projection time integration assembles the flow itself, so a term added on its own would be dropped silently; it is refused instead."""
    problem = flow(formulation="equal_order")
    if kind == "body_force":
        problem.add_kernel(kind, **parameters)
    else:
        problem.add_boundary_condition(kind, **parameters)
    with pytest.raises(ValueError, match="acts on the flow"):
        problem.solve_transient(end_time=0.1, time_step=0.05, report="none")


def test_the_body_force_of_the_flow_is_applied():
    """A uniform body force on a fluid at rest in a periodic box accelerates it uniformly: u = f t / rho."""
    mesh = dualmesh.generate_rectangle_mesh(0.0, 1.0, 0.0, 1.0, 4, 4)
    problem = dualmesh.Problem(mesh, method="fem", distributed=False)
    problem.add_physics(
        "fluid_dynamics",
        density=2.0,
        dynamic_viscosity=0.1,
        formulation="equal_order",
        time_integration="projection",
        body_force=[3.0, 0.0],
    )
    problem.add_boundary_condition("periodic", primary="left", secondary="right")
    problem.add_boundary_condition("periodic", primary="bottom", secondary="top")
    problem.solve_transient(end_time=0.5, time_step=0.1, report="none")
    assert problem.values("velocity_x") == pytest.approx(3.0 * 0.5 / 2.0, rel=1e-9)
    assert numpy.abs(problem.values("velocity_y")).max() < 1e-12


@pytest.mark.parametrize("method", methods)
def test_the_reactions_balance_the_body_force(method):
    """A body force f drives the flow through a channel periodic in x; at the steady state the walls carry the whole force, so the force on them, minus the sum of the reactions, is f times the area of the channel."""
    mesh = dualmesh.generate_rectangle_mesh(0.0, 2.0, 0.0, 1.0, 8, 6, element_type="Tri3")
    problem = dualmesh.Problem(mesh, method=method, distributed=False)
    problem.add_physics(
        "fluid_dynamics",
        density=1.0,
        dynamic_viscosity=1.0,
        formulation="equal_order",
        time_integration="projection",
        body_force=[3.0, 0.0],
    )
    problem.add_boundary_condition("periodic", primary="left", secondary="right")
    for name in ("velocity_x", "velocity_y"):
        problem.add_boundary_condition(
            "Dirichlet",
            ["bottom", "top"],
            variable=name,
            value=0.0,
        )
    problem.solve_transient(end_time=4.0, time_step=0.05, report="none")
    force = problem.boundary_flux("velocity_x", "bottom") + problem.boundary_flux(
        "velocity_x", "top"
    )
    assert force == pytest.approx(3.0 * 2.0 * 1.0, rel=1e-6)
    assert -problem.boundary_flux("velocity_y", "bottom") - problem.boundary_flux(
        "velocity_y", "top"
    ) == pytest.approx(0.0, abs=1e-8)
