# SPDX-License-Identifier: LGPL-2.1-or-later
"""The projection time integration of incompressible flow (time_integration="projection").

The decaying Taylor-Green vortex, u = (-cos x sin y, sin x cos y) exp(-2 nu t), checks the accuracy in space, with its velocity prescribed on the walls of [0, 2 pi]^2 and in a periodic box.
The stagnation flow u = g(t) (x, -y), whose velocity is linear and whose pressure is quadratic, is represented exactly by quadratic elements, so its error is that of the time integration alone.
On meshes with hanging nodes, a flow of linear velocity and pressure stays exact, and the body force balances the momentum of the fluid and the force on the walls.
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
    mesh=None,
    **solve,
):
    if mesh is None:
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


def split(mesh, *rounds):
    """The quadrilateral or hexahedral `mesh` with, in each round, the cells whose centre the
    round accepts split into 2^d children, which leaves nodes hanging on the sides of the cells
    around them."""
    from dualmesh import _core

    hierarchy = _core.Subdivision(mesh)
    for accepts in rounds:
        points = numpy.asarray(hierarchy.mesh().points())
        marked = [
            int(accepts(points[list(cell)].mean(axis=0)))
            for _type, cell, _block in hierarchy.mesh().cells()
        ]
        assert any(marked)
        hierarchy.refine(marked, 5)
    assert hierarchy.hanging()
    return hierarchy.mesh()


def near(point):
    """Accept the cell centred at `point`."""
    return lambda centre: numpy.allclose(centre[: len(point)], point)


def square_with_hanging_nodes():
    """The unit square of 8 by 8 quadrilaterals whose cell centred at (0.4375, 0.4375) is split,
    and then its child at the lower left: nodes hang on two levels, all inside the square."""
    mesh = dualmesh.generate_rectangle_mesh(0.0, 1.0, 0.0, 1.0, 8, 8, element_type="Quad4")
    return split(mesh, near((0.4375, 0.4375)), near((0.40625, 0.40625)))


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
    errors = [
        Taylor_green(n, 0.8 / n, element=element, method=method, time_stepper="fixed")[:2]
        for n in (8, 16, 32)
    ]
    velocity_order = numpy.log2(errors[0][0] / errors[-1][0]) / 2
    assert velocity_order > 1.85, rates([e[0] for e in errors])
    assert errors[-1][1] < errors[0][1] / 7


@pytest.mark.parametrize("method", methods)
def test_the_vortex_in_a_periodic_box_converges_at_second_order(method):
    errors = [
        Taylor_green(n, 0.4 / n, periodic=True, method=method, time_stepper="fixed")[:2]
        for n in (8, 16, 32, 64)
    ]
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
    problem.solve_transient(end_time=1.0, time_step=dt, time_stepper="fixed", report="none")
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
    large = Taylor_green(16, 0.1, method=method, time_stepper="fixed")[:2]
    small = Taylor_green(16, 0.004, method=method, time_stepper="fixed")[:2]
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


def test_a_flow_of_the_projection_takes_the_laplacian_form_in_every_solve():
    """The pressure of the projection is that of the laplacian form, so a flow of the projection
    takes it by default, in its steady solves as in its steps, which share the do-nothing outflow."""
    mesh = dualmesh.generate_rectangle_mesh(0.0, 1.0, 0.0, 1.0, 2, 2)
    problem = dualmesh.Problem(mesh, method="fem")
    physics = problem.add_physics(
        "fluid_dynamics", density=1.0, dynamic_viscosity=0.1, time_integration="projection"
    )
    assert physics.viscous_form == "laplacian"


def test_a_term_the_projection_does_not_split_is_refused():
    """A traction on the boundary would need the pressure of the splitting to follow it, so it is
    refused rather than dropped."""
    problem = flow(formulation="equal_order")
    problem.add_boundary_condition("Neumann", variable="velocity_x", boundary="right", flux=1.0)
    with pytest.raises(ValueError, match="acts on the flow"):
        problem.solve_transient(end_time=0.1, time_step=0.05, report="none")


def test_an_open_boundary_needs_its_pressure():
    """The pressure equation of the splitting takes the pressure on a boundary without a
    prescribed velocity from a condition, so a flow open there without one is refused."""
    problem = flow(formulation="equal_order")
    problem.add_boundary_condition(
        "Dirichlet", ["left", "bottom", "top"], variable="velocity_x", value=0.0
    )
    problem.add_boundary_condition(
        "Dirichlet", ["left", "bottom", "top"], variable="velocity_y", value=0.0
    )
    with pytest.raises(ValueError, match="'right' is open"):
        problem.solve_transient(end_time=0.1, time_step=0.05, report="none")


def open_channel(method, viscous_form, time_integration):
    """A parabolic inflow into a channel of 16 by 8 quadrilaterals with the pressure zero at its
    open right end, in the steady state: of a steady solve, or of the projection marched to
    t = 10, ten times the viscous time of the width.  Returns the velocity at the outlet."""
    mesh = dualmesh.generate_rectangle_mesh(0.0, 2.0, 0.0, 1.0, 16, 8)
    problem = dualmesh.Problem(mesh, method=method, distributed=False)
    physics = problem.add_physics(
        "fluid_dynamics",
        density=1.0,
        dynamic_viscosity=0.1,
        formulation="equal_order",
        viscous_form=viscous_form,
        time_integration=time_integration,
    )
    physics.add_boundary_condition(["velocity_x = 4*y*(1 - y)", "velocity_y = 0"], boundary="left")
    physics.add_boundary_condition(["velocity_x = 0", "velocity_y = 0"], boundary=["top", "bottom"])
    physics.add_boundary_condition("pressure = 0", boundary="right")
    if time_integration == "projection":
        problem.solve_transient(end_time=10.0, time_step=0.02, time_stepper="fixed", report="none")
    else:
        problem.solve(report="none")
    nodes = problem.mesh.boundary_nodes("right")
    return numpy.array([problem.values("velocity_x")[nodes], problem.values("velocity_y")[nodes]])


@pytest.mark.parametrize("method", methods)
def test_the_projection_has_the_outflow_of_its_viscous_form(method):
    """The stress form, whose natural condition is zero traction, and the laplacian form, whose
    natural condition is the do-nothing outflow, give outlet velocities that differ by about 0.19
    of the largest speed.  The steady state of the projection has the outflow of its own form:
    within 1.3e-2 of the steady solve of that form (measured), the difference that the pressure
    stabilization of the splitting, of parameter dt / beta_0, makes on this mesh.  A fixed step
    keeps that parameter one value (docs/theory/fluids.rst, the steady state of the projection)."""
    for form, other in (("stress", "laplacian"), ("laplacian", "stress")):
        steady = open_channel(method, form, "monolithic")
        split = open_channel(method, form, "projection")
        apart = numpy.abs(steady - open_channel(method, other, "monolithic")).max()
        assert apart > 0.15
        assert numpy.abs(split - steady).max() < 0.1 * apart, form


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


def driven_channel(method, **steps):
    """A body force f = 3 drives the flow through a channel of 8 by 6 triangles, periodic in x, to
    its steady state at t = 4.  Returns the force on the walls, minus the sum of the reactions."""
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
        problem.add_boundary_condition("Dirichlet", ["bottom", "top"], variable=name, value=0.0)
    problem.solve_transient(end_time=4.0, report="none", **steps)
    return problem.boundary_flux("velocity_x", "bottom") + problem.boundary_flux(
        "velocity_x", "top"
    )


@pytest.mark.parametrize("method", methods)
def test_the_steady_state_is_that_of_the_stabilized_flow_below_the_cap(method):
    """The splitting weighs the divergence in the pressure equation by min(beta_0 / dt, 1 / tau),
    which is the pressure stabilization of its steady state.  Below dt = beta_0 tau, about 5e-3
    on this mesh, the weight is 1 / tau whatever the step, so the steady state does not depend on
    the step (measured: 2e-10 apart at dt = 4e-3 and 2e-3), and the walls carry the body force
    times the area of the channel within 1e-6 of it (measured 2.4e-7, the momentum that the
    advection of a velocity with a discrete divergence carries).  Above it the steady state
    depends on the step (docs/theory/fluids.rst): the steady force balance of a flow is that of
    the monolithic solve (test_conservation.py)."""
    forces = [driven_channel(method, time_step=dt, time_stepper="fixed") for dt in (0.004, 0.002)]
    assert forces[0] == pytest.approx(forces[1], rel=1e-8)
    assert forces[1] == pytest.approx(3.0 * 2.0 * 1.0, rel=1e-6)


# ---------------------------------------------------------------------------
# Meshes with hanging nodes
# ---------------------------------------------------------------------------
def accelerating_shear(mesh, dimension, method):
    """u = (g(t) + y, 0, 0) and p = -rho g'(t) x with g = 1 + t / 2 and rho = 1.  Its advection and
    its viscous term vanish, so it solves the Navier-Stokes equations with the pressure gradient
    driving the acceleration g'.  Its velocity and pressure are linear in space, which every element
    and every hanging constraint hold, and linear in time, which the backward difference and the
    extrapolation hold.  Returns the largest errors of the velocity and of the pressure."""
    velocities = ["velocity_x", "velocity_y", "velocity_z"][:dimension]
    problem = dualmesh.Problem(mesh, method=method, distributed=False)
    problem.add_physics(
        "fluid_dynamics",
        density=1.0,
        dynamic_viscosity=0.1,
        formulation="equal_order",
        time_integration="projection",
    )
    walls = mesh.sideset_names()
    problem.add_boundary_condition("Dirichlet", walls, variable="velocity_x", value="1+0.5*t+y")
    for name in velocities[1:]:
        problem.add_boundary_condition("Dirichlet", walls, variable=name, value=0.0)
    problem.initialize()
    points = numpy.asarray(mesh.points())
    x, y = points[:, 0], points[:, 1]
    problem.set_values("velocity_x", 1.0 + y)
    end = 0.5
    problem.solve_transient(end_time=end, time_step=0.1, report="none")
    velocity_error = max(
        numpy.abs(problem.values("velocity_x") - (1.0 + 0.5 * end + y)).max(),
        *(numpy.abs(problem.values(name)).max() for name in velocities[1:]),
    )
    p, exact = problem.values("pressure"), -0.5 * x
    pressure_error = numpy.abs((p - p.mean()) - (exact - exact.mean())).max()
    return velocity_error, pressure_error


@pytest.mark.parametrize("method", methods)
@pytest.mark.parametrize("dimension", [2, 3])
def test_a_linear_flow_stays_exact_across_hanging_nodes(method, dimension):
    """The accelerating shear flow on a mesh with nodes hanging inside it: the unit square of 8 by
    8 quadrilaterals split on two levels, or the unit cube of 4 by 4 by 4 hexahedra whose cell
    centred at (0.375, 0.375, 0.375) is split.  FEM constrains a hanging value to the side it lies
    on, which holds a linear field, and DMCDM splits a large cell as a polygon whose control
    domains tile the domain with those of its small neighbours, so the flux of a linear field out
    of every control domain is exact.  The scheme then reproduces the flow up to its linear solves,
    which stop at a relative residual of 1e-10."""
    if dimension == 2:
        mesh = square_with_hanging_nodes()
    else:
        mesh = split(
            dualmesh.generate_box_mesh(0, 1, 0, 1, 0, 1, 4, 4, 4, element_type="Hex8"),
            near((0.375, 0.375, 0.375)),
        )
    velocity_error, pressure_error = accelerating_shear(mesh, dimension, method)
    assert velocity_error < 1e-7
    assert pressure_error < 1e-7


@pytest.mark.parametrize("method", methods)
def test_a_body_force_accelerates_the_fluid_as_one_across_hanging_nodes(method):
    """A uniform body force on a fluid at rest in a periodic box with nodes hanging inside: every
    control domain, or every constrained test function, gains its share of the force, so the
    fluid moves as one body, u = f t / rho, at the hanging nodes too.  The pressure and velocity
    solves of the projection stop at a relative residual of 1e-10, so they leave a velocity_y of
    about 1e-10 of the speed U = f t / rho that the fluid reaches, which bounds it."""
    problem = dualmesh.Problem(square_with_hanging_nodes(), method=method, distributed=False)
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
    speed = 3.0 * 0.5 / 2.0
    assert problem.values("velocity_x") == pytest.approx(speed, rel=1e-9)
    assert numpy.abs(problem.values("velocity_y")).max() < 1e-10 * speed


@pytest.mark.parametrize(
    "method, time_integration",
    [("fem", "projection"), ("dmcdm", "projection"), ("fem", "monolithic")],
)
def test_a_body_force_accelerates_the_fluid_as_one_across_nodes_hanging_on_periodic_sides(
    method, time_integration
):
    """The uniform body force on a fluid at rest in the unit cube, periodic in every direction,
    whose cells centred at (0.125, 0.375, 0.375) and (0.875, 0.375, 0.375) are split, so that
    nodes hang on the edges of both faces x = 0 and x = 1 and their periodic copies hang alike.
    Under FEM a copy of a hanging node depends on the independent nodes once, so every
    constrained test function gains its share of the force, and the fluid moves as one body,
    u = f t / rho, in the projection integration and in the monolithic one after its start
    projection, which shares the layout of the projection.  DMCDM keeps the hanging nodes as
    unknowns, so only its projection integration is checked here."""
    mesh = split(
        dualmesh.generate_box_mesh(0, 1, 0, 1, 0, 1, 4, 4, 4, element_type="Hex8"),
        lambda centre: numpy.allclose(centre[1:], (0.375, 0.375)) and centre[0] in (0.125, 0.875),
    )
    problem = dualmesh.Problem(mesh, method=method, distributed=False)
    problem.add_physics(
        "fluid_dynamics",
        density=2.0,
        dynamic_viscosity=0.1,
        formulation="equal_order",
        time_integration=time_integration,
        body_force=[3.0, 0.0, 0.0],
    )
    for primary, secondary in (("left", "right"), ("bottom", "top"), ("back", "front")):
        problem.add_boundary_condition("periodic", primary=primary, secondary=secondary)
    problem.solve_transient(end_time=0.5, time_step=0.1, report="none")
    speed = 3.0 * 0.5 / 2.0
    assert problem.values("velocity_x") == pytest.approx(speed, rel=1e-9)
    for name in ("velocity_y", "velocity_z"):
        assert numpy.abs(problem.values(name)).max() < 1e-9 * speed


def test_the_reactions_balance_the_body_force_across_hanging_nodes():
    """The channel of test_the_reactions_balance_the_body_force, of 8 by 4 quadrilaterals whose
    second row is split, which leaves nodes hanging on the sides of the rows above and below it.
    Every constrained test function of FEM is a function of x times a function of y, so the steady
    flow is the same in every column, its advection carries no momentum, and the walls carry the
    whole force: f times the area of the channel."""
    mesh = split(
        dualmesh.generate_rectangle_mesh(0.0, 2.0, 0.0, 1.0, 8, 4, element_type="Quad4"),
        lambda centre: 0.25 < centre[1] < 0.5,
    )
    problem = dualmesh.Problem(mesh, method="fem", distributed=False)
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
        problem.add_boundary_condition("Dirichlet", ["bottom", "top"], variable=name, value=0.0)
    problem.solve_transient(end_time=4.0, time_step=0.05, report="none")
    force = problem.boundary_flux("velocity_x", "bottom") + problem.boundary_flux(
        "velocity_x", "top"
    )
    assert force == pytest.approx(3.0 * 2.0 * 1.0, rel=1e-6)
    assert -problem.boundary_flux("velocity_y", "bottom") - problem.boundary_flux(
        "velocity_y", "top"
    ) == pytest.approx(0.0, abs=1e-8)


@pytest.mark.parametrize("method", methods)
def test_the_distributed_path_gives_the_serial_answer_across_hanging_nodes(method):
    """The Taylor-Green vortex with walls on 8 by 8 quadrilaterals split on two levels inside."""

    def mesh():
        h = numpy.pi / 4
        square = dualmesh.generate_rectangle_mesh(
            0.0, 2 * numpy.pi, 0.0, 2 * numpy.pi, 8, 8, element_type="Quad4"
        )
        return split(square, near((3.5 * h, 3.5 * h)), near((3.25 * h, 3.25 * h)))

    serial = Taylor_green(8, 0.1, method=method, mesh=mesh())[3]
    distributed = Taylor_green(8, 0.1, distributed=True, method=method, mesh=mesh())[3]
    for name in ("velocity_x", "velocity_y"):
        assert numpy.asarray(distributed.gathered_values(name)) == pytest.approx(
            serial.values(name), abs=1e-9
        )


# ---------------------------------------------------------------------------
# Coordinates and formulations of the problem
# ---------------------------------------------------------------------------
def axisymmetric_stagnation(dt, method):
    """u_r = g r / 2 and u_z = -g z with g = 1 + sin(2 t) / 2, the stagnation flow of a jet on a
    wall, without divergence (u_r / r + du_r/dr + du_z/dz = 0) and without a viscous term, with the
    pressure p = -rho ((g' / 4 + g^2 / 8) r^2 + (g^2 - g') z^2 / 2).  Quadratic elements represent
    it exactly, so its error is that of the time integration alone.  Returns the largest error of
    the pressure."""
    mesh = dualmesh.generate_rectangle_mesh(0.0, 1.0, 0.0, 1.0, 3, 3, element_type="Quad9")
    problem = dualmesh.Problem(mesh, method=method, coordinates="axisymmetric", distributed=False)
    problem.add_physics(
        "fluid_dynamics",
        density=1.0,
        dynamic_viscosity=0.05,
        formulation="equal_order",
        time_integration="projection",
    )
    g = "(1+0.5*sin(2*t))"
    walls = ["left", "right", "bottom", "top"]
    problem.add_boundary_condition("Dirichlet", walls, variable="velocity_r", value=f"{g}*x/2")
    problem.add_boundary_condition("Dirichlet", walls, variable="velocity_z", value=f"-{g}*y")
    problem.initialize()
    points = numpy.asarray(mesh.points())
    r, z = points[:, 0], points[:, 1]
    problem.set_values("velocity_r", r / 2)
    problem.set_values("velocity_z", -z)
    problem.solve_transient(end_time=1.0, time_step=dt, time_stepper="fixed", report="none")
    G, G_p = 1 + 0.5 * numpy.sin(2.0), numpy.cos(2.0)
    exact = -((G_p / 4 + G * G / 8) * r * r + (G * G - G_p) * z * z / 2)
    p = problem.values("pressure")
    return numpy.abs((p - p.mean()) - (exact - exact.mean())).max()


@pytest.mark.parametrize("method", methods)
def test_an_axisymmetric_flow_has_the_order_of_the_time_integration(method):
    """The operators of the projection are those of the problem, with the coordinate factor 2 pi r
    and the hoop terms of the axisymmetric flow, so the axisymmetric stagnation flow converges at
    the second order of BDF2, as the plane one does."""
    errors = [axisymmetric_stagnation(dt, method) for dt in (0.05, 0.025, 0.0125)]
    assert rates(errors)[-1] > 1.95, errors


def test_the_Taylor_Hood_element_keeps_a_linear_flow_exact():
    """The accelerating shear flow on quadratic elements with a linear pressure (the Taylor-Hood
    element), whose pressure equation tests with the linear functions of the corners."""
    mesh = dualmesh.generate_rectangle_mesh(0.0, 1.0, 0.0, 1.0, 4, 4, element_type="Quad9")
    problem = dualmesh.Problem(mesh, method="fem", distributed=False)
    physics = problem.add_physics(
        "fluid_dynamics", density=1.0, dynamic_viscosity=0.1, time_integration="projection"
    )
    assert physics.formulation == "Taylor-Hood"
    walls = mesh.sideset_names()
    problem.add_boundary_condition("Dirichlet", walls, variable="velocity_x", value="1+0.5*t+y")
    problem.add_boundary_condition("Dirichlet", walls, variable="velocity_y", value=0.0)
    problem.initialize()
    points = numpy.asarray(mesh.points())
    x, y = points[:, 0], points[:, 1]
    problem.set_values("velocity_x", 1.0 + y)
    problem.solve_transient(end_time=0.5, time_step=0.1, report="none")
    assert numpy.abs(problem.values("velocity_x") - (1.25 + y)).max() < 1e-7
    assert numpy.abs(problem.values("velocity_y")).max() < 1e-7
    p = problem.values("pressure")
    assert numpy.abs((p - p.mean()) - (-0.5 * x - (-0.5 * x).mean())).max() < 1e-7
