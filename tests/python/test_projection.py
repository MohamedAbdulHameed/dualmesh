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
        ({"formulation": "equal_order", "viscous_form": "stress"}, "takes the laplacian form"),
    ],
)
def test_unsupported_settings_are_refused(parameters, message):
    with pytest.raises(ValueError, match=message):
        flow(**parameters)


def test_a_flow_of_the_projection_takes_the_laplacian_form_in_every_solve():
    """The projection assembles the viscous term as mu grad v, so the physics takes the laplacian
    form, and its steady solves have the same do-nothing outflow as its steps."""
    mesh = dualmesh.generate_rectangle_mesh(0.0, 1.0, 0.0, 1.0, 2, 2)
    problem = dualmesh.Problem(mesh, method="fem")
    physics = problem.add_physics(
        "fluid_dynamics", density=1.0, dynamic_viscosity=0.1, time_integration="projection"
    )
    assert physics.viscous_form == "laplacian"


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
