# SPDX-License-Identifier: LGPL-2.1-or-later
"""Tests of the time integrators and of the adaptive time steppers.

The reference problem throughout is the cooling of a slab,

    dT/dt = d^2 T / dx^2,   T(x, 0) = sin(pi x),   T(0, t) = T(1, t) = 0,

whose solution is T(x, t) = exp(-pi^2 t) sin(pi x).  It is linear, so every
nonlinear iteration converges in one step and any difference between the
steppers is a difference in the time integration alone.
"""

from __future__ import annotations

import dualmesh as dm
import numpy as np
import pytest


def slab(num_elements=40, method="dmcdm"):
    mesh = dm.generate_line_mesh(start=0.0, end=1.0, num_elements=num_elements)
    problem = dm.Problem(mesh, method=method)
    problem.add_variable("temperature")
    problem.add_kernel("Diffusion", "diffusion", variable="temperature")
    problem.add_kernel("TimeDerivative", "storage", variable="temperature")
    problem.add_boundary_condition(
        "DirichletBC", "ends", variable="temperature", boundary=["left", "right"], value=0.0
    )
    # The initial condition is set after the variable exists, so that it does
    # not force the assembly onto one thread.
    values = np.sin(np.pi * problem.entity_points()[:, 0])
    values[[0, -1]] = 0.0
    problem.set_values("temperature", values)
    return problem


def exact(problem, time):
    return np.exp(-(np.pi**2) * time) * np.sin(np.pi * problem.entity_points()[:, 0])


def error(problem, time):
    return float(np.abs(problem.values("temperature") - exact(problem, time)).max())


# ---------------------------------------------------------------------------
# The theta family
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("theta, order", [(1.0, 1.0), (0.5, 2.0)])
def test_theta_method_converges_at_its_order(theta, order):
    """Backward Euler is first-order in time and Crank-Nicolson second-order.

    The error measured is the error in *time* alone: the reference is the same
    spatial discretisation advanced with a step small enough that its own
    temporal error is negligible.  Comparing against the analytical solution
    instead would mix in the spatial error, which does not shrink as the step
    is refined and would flatten the measured rate.
    """
    reference = slab(num_elements=100)
    reference.solve_transient(end_time=0.02, dt=1.0e-5, theta=theta)
    exact_in_time = reference.values("temperature")

    errors = []
    for dt in (0.002, 0.001, 0.0005):
        problem = slab(num_elements=100)
        problem.solve_transient(end_time=0.02, dt=dt, theta=theta)
        errors.append(float(np.abs(problem.values("temperature") - exact_in_time).max()))
    rates = [np.log2(errors[i] / errors[i + 1]) for i in range(2)]
    assert min(rates) > order - 0.2
    assert max(rates) < order + 0.3


def test_forward_euler_is_stable_below_the_critical_step_and_unstable_above():
    """With theta = 0 the steady terms are evaluated at the old state, which
    makes the scheme explicit and therefore only conditionally stable.

    The critical step is set by the largest eigenvalue of the capacity-scaled
    conduction operator.  For this discretisation it sits between 0.2 and 0.3
    times the square of the element size, so a step a quarter of that is
    comfortably stable and one twice it diverges.
    """
    h = 1.0 / 20
    stable = slab(num_elements=20)
    stable.solve_transient(end_time=0.1, dt=0.05 * h * h, theta=0.0)
    assert np.isfinite(stable.values("temperature")).all()
    assert error(stable, 0.1) < 1e-2

    unstable = slab(num_elements=20)
    unstable.solve_transient(end_time=0.1, dt=0.5 * h * h, theta=0.0)
    assert np.abs(unstable.values("temperature")).max() > 1.0e3


# ---------------------------------------------------------------------------
# The adaptive steppers
# ---------------------------------------------------------------------------
def test_fixed_stepper_lands_exactly_on_the_end_time():
    problem = slab()
    result = problem.solve_transient(end_time=0.1, dt=0.003, theta=1.0)
    assert result.converged
    assert sum(dt for _, dt in result.step_history) == pytest.approx(0.1, abs=1e-12)
    assert result.step_history[-1][0] == pytest.approx(0.1, abs=1e-12)


def test_error_stepper_reaches_the_same_accuracy_in_fewer_steps():
    """The error-controlled stepper should need far fewer steps than a fixed
    step of the size it starts from, for the same final accuracy."""
    fixed = slab(num_elements=80)
    fixed_result = fixed.solve_transient(end_time=0.1, dt=0.001, theta=0.5)

    adaptive = slab(num_elements=80)
    adaptive_result = adaptive.solve_transient(
        end_time=0.1, dt=1e-4, theta=0.5, time_stepper="error", error_tolerance=1e-6
    )
    assert adaptive_result.converged
    assert adaptive_result.time_steps < 0.6 * fixed_result.time_steps
    assert error(adaptive, 0.1) < 2.0 * error(fixed, 0.1)
    # Starting from a step much smaller than the tolerance requires, the
    # controller must grow it.
    assert adaptive_result.step_history[-1][1] > 5.0 * adaptive_result.step_history[0][1]


def test_error_stepper_responds_to_its_tolerance():
    """Tightening the tolerance must shorten the steps and reduce the error,
    until the spatial discretisation error takes over."""
    steps, errors = [], []
    for tolerance in (1e-3, 1e-5, 1e-7):
        problem = slab(num_elements=80)
        result = problem.solve_transient(
            end_time=0.1,
            dt=1e-4,
            theta=0.5,
            time_stepper="error",
            error_tolerance=tolerance,
        )
        steps.append(result.time_steps)
        errors.append(error(problem, 0.1))
    assert steps[0] < steps[1] < steps[2]
    assert errors[0] > errors[1] > errors[2]


def test_iteration_stepper_grows_the_step_on_an_easy_problem():
    """A linear problem converges in one iteration every step, so the
    iteration controller should keep enlarging the step."""
    problem = slab()
    result = problem.solve_transient(
        end_time=0.1,
        dt=0.0005,
        theta=1.0,
        time_stepper="iteration",
        optimal_iterations=4,
        iteration_window=2,
    )
    assert result.converged
    assert result.step_history[-1][1] > 10.0 * result.step_history[0][1]
    assert result.time_steps < 30


def test_a_step_that_is_too_inaccurate_is_rejected_and_retried():
    """Given a first step far larger than the tolerance can accept, the
    error-controlled stepper must throw it away, cut back, and recover."""
    problem = slab(num_elements=80)
    result = problem.solve_transient(
        end_time=0.1,
        dt=0.05,
        theta=0.5,
        time_stepper="error",
        error_tolerance=1e-8,
    )
    assert result.converged
    assert result.rejected_steps > 0
    assert result.step_history[0][1] < 0.05
    assert error(problem, 0.1) < 1e-4


def test_a_nonlinear_step_that_will_not_converge_is_rejected_and_retried():
    """Radiation from a cold start is a hard nonlinear step.  Given a step far
    too large for the iteration limit, the stepper must cut back and recover
    rather than fail."""
    mesh = dm.generate_line_mesh(start=0.0, end=0.05, num_elements=20)
    problem = dm.Problem(mesh)
    problem.add_variable("temperature")
    problem.set_values("temperature", np.full(mesh.num_nodes, 300.0))
    problem.add_kernel(
        "HeatConduction", "conduction", variable="temperature", thermal_conductivity=20.0
    )
    problem.add_kernel(
        "HeatConductionTimeDerivative",
        "storage",
        variable="temperature",
        density=7800.0,
        specific_heat=460.0,
    )
    problem.add_boundary_condition(
        "DirichletBC", "hot", variable="temperature", boundary="left", value=3000.0
    )
    problem.add_boundary_condition(
        "RadiativeHeatFluxBC",
        "radiation",
        variable="temperature",
        boundary="right",
        emissivity=0.9,
        ambient_temperature=300.0,
    )
    result = problem.solve_transient(
        end_time=400.0,
        dt=400.0,
        theta=1.0,
        time_stepper="iteration",
        max_iterations=2,
        cutback_factor=0.25,
    )
    assert result.converged
    assert result.rejected_steps > 0
    assert np.isfinite(problem.values("temperature")).all()
    assert problem.values("temperature").max() == pytest.approx(3000.0, rel=1e-6)


def test_the_run_reports_a_failure_instead_of_grinding_to_a_halt():
    """A step that can never converge must end the run with a clear error
    rather than halving the step for ever."""
    mesh = dm.generate_line_mesh(start=0.0, end=1.0, num_elements=10)
    problem = dm.Problem(mesh)
    problem.add_variable("u")
    problem.add_kernel("Diffusion", "diffusion", variable="u")
    problem.add_kernel("TimeDerivative", "time", variable="u")
    # A reaction with a negative coefficient and a large exponent, which has no
    # bounded solution, so no step size helps.
    problem.add_kernel("Reaction", "runaway", variable="u", coefficient=-1.0e12, exponent=3.0)
    problem.add_boundary_condition("DirichletBC", "left", variable="u", boundary="left", value=1.0)
    with pytest.raises(RuntimeError, match="transient solve failed"):
        problem.solve_transient(
            end_time=1.0,
            dt=0.5,
            time_stepper="iteration",
            max_iterations=5,
            dt_min=1e-6,
            max_rejected_steps=4,
        )


@pytest.mark.parametrize("time_stepper", ["fixed", "error", "iteration"])
@pytest.mark.parametrize("method", ["dmcdm", "fem", "hfvm", "zfvm"])
def test_every_discretisation_works_with_every_stepper(method, time_stepper):
    problem = slab(num_elements=40, method=method)
    result = problem.solve_transient(
        end_time=0.05,
        dt=0.0005,
        theta=0.5,
        time_stepper=time_stepper,
        error_tolerance=1e-7,
    )
    assert result.converged
    assert error(problem, 0.05) < 5e-3


def test_the_distributed_solver_takes_the_same_steps():
    """The adaptive controller is driven by global norms, so every rank makes
    the same decision; on one rank the answer must match the serial one."""
    mesh = dm.generate_line_mesh(start=0.0, end=1.0, num_elements=40)

    def define(problem):
        problem.add_variable("temperature")
        problem.add_kernel("Diffusion", "diffusion", variable="temperature")
        problem.add_kernel("TimeDerivative", "storage", variable="temperature")
        problem.add_boundary_condition(
            "DirichletBC", "ends", variable="temperature", boundary=["left", "right"], value=0.0
        )
        values = np.sin(np.pi * problem.entity_points()[:, 0])
        values[[0, -1]] = 0.0
        problem.set_values("temperature", values)

    serial = dm.Problem(mesh)
    define(serial)
    serial_result = serial.solve_transient(
        end_time=0.1, dt=0.001, theta=0.5, time_stepper="error", error_tolerance=1e-6
    )

    distributed = dm.DistributedProblem(mesh, linear_tolerance=1e-13)
    define(distributed.local)
    distributed_result = distributed.solve_transient(
        end_time=0.1, dt=0.001, theta=0.5, time_stepper="error", error_tolerance=1e-6
    )
    assert distributed_result.time_steps == serial_result.time_steps
    got = np.asarray(distributed.gathered_values("temperature"))
    assert got == pytest.approx(serial.values("temperature"), abs=1e-8)
