"""The pressure level of an enclosed flow, which the solver fixes itself: a flow whose boundary
prescribes no pressure determines it only up to a constant, and the reported pressure has a mean
of zero.  A flow with an outlet keeps the level its outlet sets."""

from __future__ import annotations

import dualmesh as dm
import numpy as np
import pytest
from dualmesh._core import InputError

METHODS = ["FEM", "DMCDM"]


def cavity(
    method,
    n=8,
    bottom="wall",
    pin=None,
    pin_value=0.0,
    formulation="pressure",
    element_type="Quad4",
):
    """The lid-driven cavity at Re = 100; the bottom a no-slip wall or a plane of symmetry."""
    mesh = dm.generate_rectangle_mesh(0, 1, 0, 1, n, n, element_type=element_type)
    problem = dm.Problem(mesh, method=method)
    flow = problem.add_physics(
        "fluid_dynamics", density=100.0, dynamic_viscosity=1.0, formulation=formulation
    )
    flow.add_boundary_condition(
        ["velocity_x = 1", "velocity_y = 0"], boundary="top", scale_with_load=True
    )
    walls = ["left", "right", "bottom"]
    if bottom == "symmetry":
        walls.remove("bottom")
        flow.add_boundary_condition("symmetry", boundary="bottom")
    flow.add_boundary_condition(["velocity_x = 0", "velocity_y = 0"], boundary=walls)
    if pin is not None:
        problem.add_boundary_condition(
            "point_Dirichlet", variable="pressure", point=list(pin), value=pin_value
        )
    problem.solve(report="none")
    return problem


def pressure_mean(problem, area=1.0):
    return problem.integrate("pressure") / area


@pytest.mark.parametrize("method", METHODS)
def test_an_enclosed_flow_reports_the_pressure_with_a_mean_of_zero(method):
    problem = cavity(method)
    p = problem.values("pressure")
    assert abs(pressure_mean(problem)) < 1e-10 * np.abs(p).max()


@pytest.mark.parametrize("method", METHODS)
def test_the_held_level_changes_the_pressure_by_a_constant_only(method):
    """Against the pressure prescribed at a point: the velocity is the same, and the pressures
    differ by a constant."""
    free, pinned = cavity(method), cavity(method, pin=(0.5, 0.0))
    for v in ("velocity_x", "velocity_y"):
        np.testing.assert_allclose(free.values(v), pinned.values(v), atol=1e-10)
    difference = free.values("pressure") - pinned.values("pressure")
    assert np.ptp(difference) < 1e-8 * np.abs(pinned.values("pressure")).max()
    # A condition that sets the level is kept: no shift applies to the pinned pressure.
    assert abs(pressure_mean(pinned)) > 1e-3 * np.abs(pinned.values("pressure")).max()


@pytest.mark.parametrize("method", METHODS)
def test_a_plane_of_symmetry_leaves_the_level_free(method):
    """A symmetry plane holds the normal velocity only; the flow is still enclosed."""
    problem = cavity(method, bottom="symmetry")
    assert abs(pressure_mean(problem)) < 1e-10 * np.abs(problem.values("pressure")).max()


def test_the_taylor_hood_pressure_has_a_mean_of_zero():
    problem = cavity("FEM", n=4, formulation="Taylor-Hood", element_type="Quad9")
    assert abs(pressure_mean(problem)) < 1e-10 * np.abs(problem.values("pressure")).max()


@pytest.mark.parametrize("method", METHODS)
def test_an_outlet_sets_the_level(method):
    """Poiseuille flow with a traction-free outlet: the pressure falls to zero at the outlet,
    so its mean is positive, which a shift to zero mean would destroy."""
    mesh = dm.generate_rectangle_mesh(0, 4, 0, 1, 16, 4)
    problem = dm.Problem(mesh, method=method)
    flow = problem.add_physics(
        "fluid_dynamics", density=0.0, dynamic_viscosity=1.0, formulation="pressure"
    )
    flow.add_boundary_condition(["velocity_x = 4*y*(1 - y)", "velocity_y = 0"], boundary="left")
    flow.add_boundary_condition(["velocity_x = 0", "velocity_y = 0"], boundary=["top", "bottom"])
    problem.solve(report="none")
    # dp/dx = -8 mu U / H^2 = -8 over a length 4: 32 at the inlet, a mean of 16, which the
    # stabilised methods reach to within 20 % on four cells across.
    assert pressure_mean(problem, area=4.0) == pytest.approx(16.0, rel=0.2)


def test_the_projection_reports_the_pressure_with_a_mean_of_zero():
    """The projection solves its own pressure equation, with the same convention."""
    n = 16
    mesh = dm.generate_rectangle_mesh(0.0, 2 * np.pi, 0.0, 2 * np.pi, n, n)
    problem = dm.Problem(mesh, method="FEM")
    problem.add_physics(
        "fluid_dynamics",
        density=1.0,
        dynamic_viscosity=0.1,
        formulation="pressure",
        time_integration="projection",
    )
    decay = "exp(-0.2*t)"
    for name, value in (
        ("velocity_x", f"-cos(x)*sin(y)*{decay}"),
        ("velocity_y", f"sin(x)*cos(y)*{decay}"),
    ):
        problem.add_boundary_condition(
            "Dirichlet", ["left", "right", "bottom", "top"], variable=name, value=value
        )
    problem.initialize()
    x, y = np.asarray(mesh.points())[:, 0], np.asarray(mesh.points())[:, 1]
    problem.set_values("velocity_x", -np.cos(x) * np.sin(y))
    problem.set_values("velocity_y", np.sin(x) * np.cos(y))
    problem.solve_transient(end_time=0.1, time_step=0.02, report="none")
    assert (
        abs(pressure_mean(problem, area=(2 * np.pi) ** 2))
        < 1e-10 * np.abs(problem.values("pressure")).max()
    )


class PressureSource(dm.PythonKernel):
    """A momentum source c p (or c p u, zero at rest, as the stress of a viscosity that depends on
    the pressure, the law of Barus, is): a term whose value changes with the pressure level."""

    def setup(self, problem):
        self.p = problem.variable_index("pressure")
        self.u = problem.variable_index("velocity_x")

    def compute_source(self, ctx):
        value = 0.1 * ctx.value(self.p)
        return value * ctx.value(self.u) if self.zero_at_rest else value


@pytest.mark.parametrize("zero_at_rest", [False, True])
def test_a_level_that_enters_the_flow_is_refused(zero_at_rest):
    """With such a term every pressure level gives a flow of its own, and the mass equations of an
    enclosed flow do not fix it: the level must be given.  The term c p shows at the start, c p u
    only in the converged flow."""
    mesh = dm.generate_rectangle_mesh(0, 1, 0, 1, 6, 6)
    problem = dm.Problem(mesh, method="DMCDM")
    flow = problem.add_physics(
        "fluid_dynamics", density=0.0, dynamic_viscosity=1.0, formulation="pressure"
    )
    flow.add_boundary_condition(["velocity_x = 1", "velocity_y = 0"], boundary="top")
    flow.add_boundary_condition(
        ["velocity_x = 0", "velocity_y = 0"], boundary=["left", "right", "bottom"]
    )
    problem.add_kernel(PressureSource(variable="velocity_x", zero_at_rest=zero_at_rest))
    with pytest.raises(InputError, match="point_Dirichlet"):
        problem.solve(report="none")


@pytest.mark.parametrize("method", METHODS)
def test_a_net_flux_through_the_walls_is_reported(method, capfd):
    """A velocity that leaves through every wall of a box cannot belong to an incompressible flow.
    The solve says so, once, instead of leaving a silent pressure spike."""
    mesh = dm.generate_rectangle_mesh(0, 1, 0, 1, 6, 6)
    problem = dm.Problem(mesh, method=method)
    flow = problem.add_physics(
        "fluid_dynamics", density=0.0, dynamic_viscosity=1.0, formulation="pressure"
    )
    flow.add_boundary_condition(
        ["velocity_x = x - 0.5", "velocity_y = y - 0.5"],
        boundary=["left", "right", "bottom", "top"],
    )
    problem.solve(report="none")
    problem.solve(report="none")
    assert capfd.readouterr().err.count("dualmesh: warning: the velocity prescribed") == 1


@pytest.mark.parametrize("method", METHODS)
def test_the_reactions_belong_to_the_reported_pressure(method):
    """The force on a wall depends on the pressure level.  It is taken with the pressure as
    reported: the reported pressure prescribed at a point gives the same reactions."""
    free = cavity(method)
    p = free.sample("pressure", np.array([[0.5, 0.5]]))[0]
    pinned = cavity(method, pin=(0.5, 0.5), pin_value=p)
    for variable in ("velocity_x", "velocity_y"):
        for wall in ("left", "bottom", "top"):
            assert free.total_reaction(variable, wall) == pytest.approx(
                pinned.total_reaction(variable, wall), rel=1e-8, abs=1e-10
            ), (variable, wall)


def channel(method, top_velocity=0.0, body_force=1.0):
    """A channel periodic in x between walls, driven by a body force or a moving top wall: the
    exact pressure is zero, which the solve must not take for a level that changes the flow."""
    mesh = dm.generate_rectangle_mesh(0, 2, 0, 1, 8, 4)
    problem = dm.Problem(mesh, method=method)
    flow = problem.add_physics(
        "fluid_dynamics",
        density=1.0,
        dynamic_viscosity=1.0,
        formulation="pressure",
        body_force=[body_force, 0.0],
    )
    problem.add_boundary_condition("periodic", primary="left", secondary="right")
    flow.add_boundary_condition(["velocity_x = 0", "velocity_y = 0"], boundary="bottom")
    flow.add_boundary_condition([f"velocity_x = {top_velocity}", "velocity_y = 0"], boundary="top")
    problem.solve(report="none")
    return problem.sample("velocity_x", np.array([[1.0, 0.5]]))[0]


@pytest.mark.parametrize("method", METHODS)
def test_a_flow_with_no_pressure_keeps_its_level_free(method):
    """Poiseuille flow by a body force (u = y (1 - y) / 2) and Couette flow (u = y)."""
    assert channel(method) == pytest.approx(0.125, rel=1e-6)
    assert channel(method, top_velocity=1.0, body_force=0.0) == pytest.approx(0.5, rel=1e-6)
