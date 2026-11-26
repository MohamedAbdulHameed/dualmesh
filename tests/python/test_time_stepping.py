# SPDX-License-Identifier: LGPL-2.1-or-later
"""Tests of the time integrators and of the adaptive time steppers.

The reference problem throughout is the cooling of a slab,

    dT/dt = d^2 T / dx^2,   T(x, 0) = sin(pi x),   T(0, t) = T(1, t) = 0,

whose solution is T(x, t) = exp(-pi^2 t) sin(pi x).  It is linear, so every
nonlinear iteration converges in one step and any difference between the
steppers is a difference in the time integration alone.
"""

from __future__ import annotations

import dualmesh
import numpy
import pytest
from dualmesh.input import units


def slab(num_elements=40, method="dmcdm"):
    mesh = dualmesh.generate_line_mesh(start=0.0, end=1.0, num_elements=num_elements)
    problem = dualmesh.Problem(mesh, method=method)
    problem.add_variable("temperature")
    problem.add_kernel("diffusion", variable="temperature")
    problem.add_kernel("time_derivative", variable="temperature")
    problem.add_boundary_condition(
        "Dirichlet",
        ["left", "right"],
        variable="temperature",
        value=0.0,
    )
    # The initial condition is set after the variable exists, so that it does
    # not force the assembly onto one thread.
    values = numpy.sin(numpy.pi * problem.node_coordinates()[:, 0])
    values[[0, -1]] = 0.0
    problem.set_values("temperature", values)
    return problem


def exact(problem, time):
    return numpy.exp(-(numpy.pi**2) * time) * numpy.sin(numpy.pi * problem.node_coordinates()[:, 0])


def error(problem, time):
    return float(numpy.abs(problem.values("temperature") - exact(problem, time)).max())


# ---------------------------------------------------------------------------
# The theta family
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("time_scheme, order", [("backward_Euler", 1.0), ("Crank-Nicolson", 2.0)])
def test_time_schemes_converge_at_their_order(time_scheme, order):
    """Backward Euler is first-order in time and Crank-Nicolson second-order.

    The error measured is the error in *time* alone: the reference is the same
    spatial discretisation advanced with a step small enough that its own
    temporal error is negligible.  Comparing against the analytical solution
    instead would mix in the spatial error, which does not shrink as the step
    is refined and would flatten the measured rate.
    """
    reference = slab(num_elements=100)
    reference.solve_transient(
        end_time=0.02, time_step=1.0e-5, time_scheme=time_scheme, time_stepper="fixed"
    )
    exact_in_time = reference.values("temperature")

    errors = []
    for dt in (0.002, 0.001, 0.0005):
        problem = slab(num_elements=100)
        problem.solve_transient(
            end_time=0.02, time_step=dt, time_scheme=time_scheme, time_stepper="fixed"
        )
        errors.append(float(numpy.abs(problem.values("temperature") - exact_in_time).max()))
    rates = [numpy.log2(errors[i] / errors[i + 1]) for i in range(2)]
    assert min(rates) > order - 0.2
    assert max(rates) < order + 0.3


def test_forward_Euler_is_stable_below_the_critical_step_and_unstable_above():
    """With theta = 0 the steady terms are evaluated at the old state, which
    makes the scheme explicit and therefore only conditionally stable.

    The capacity of time_derivative is lumped on these linear elements, so each
    of the N - 1 interior nodes holds the capacity h, with h = 1/N the element
    size, and the conduction operator is (1/h) tridiag(-1, 2, -1).  The
    eigenvalues of the capacity-scaled operator are then
    lambda_k = (4/h^2) sin^2(k pi / (2N)), k = 1, ..., N - 1, and forward Euler
    is stable when dt lambda_{N-1} <= 2, below the critical step
    dt_c = h^2 / (2 sin^2((N - 1) pi / (2N))), about 0.503 h^2 for N = 20.  A
    step a tenth of it is comfortably stable, and one twice it diverges: the
    mode k = N - 1, which the initial state holds at round-off, about 1e-17,
    grows by |1 - 4| = 3 per step, past 1e3 within the 80 steps to t = 0.2.
    """
    N = 20
    h = 1.0 / N
    critical = h * h / (2.0 * numpy.sin((N - 1) * numpy.pi / (2 * N)) ** 2)
    stable = slab(num_elements=N)
    stable.solve_transient(
        end_time=0.1, time_step=0.1 * critical, time_scheme="forward_Euler", time_stepper="fixed"
    )
    assert numpy.isfinite(stable.values("temperature")).all()
    assert error(stable, 0.1) < 1e-2

    unstable = slab(num_elements=N)
    unstable.solve_transient(
        end_time=0.2, time_step=2.0 * critical, time_scheme="forward_Euler", time_stepper="fixed"
    )
    assert numpy.abs(unstable.values("temperature")).max() > 1.0e3


@pytest.mark.parametrize("time_scheme", ["backward_Euler", "Crank-Nicolson"])
def test_the_error_of_a_theta_step_is_estimated_from_one_solve(time_scheme):
    """After the first steps, the error stepper estimates the error of a step by extrapolating the
    last accepted states, at one solve per step where step doubling takes three.  The slab is
    linear, so a solve takes one Newton iteration and the iterations count the solves.  A tighter
    tolerance gives a smaller error in time, measured against the same mesh advanced with a step
    of 1e-4."""
    reference = slab(num_elements=80)
    reference.solve_transient(
        end_time=0.1,
        time_step=1e-4,
        time_scheme="Crank-Nicolson",
        time_stepper="fixed",
        report="none",
    )
    errors = []
    for tolerance in (1e-4, 1e-5):
        problem = slab(num_elements=80)
        result = problem.solve_transient(
            end_time=0.1,
            time_step=1e-3,
            time_scheme=time_scheme,
            time_stepper="error",
            error_tolerance=tolerance,
            report="none",
        )
        steps = result.time_steps + result.rejected_steps
        assert result.total_iterations <= 1.2 * steps + 10, (result.total_iterations, steps)
        errors.append(
            float(numpy.abs(problem.values("temperature") - reference.values("temperature")).max())
        )
    assert errors[1] < errors[0]


def test_the_error_of_a_step_leaves_out_the_algebraic_variables():
    """A variable without a time derivative has no truncation error of its own.  Beside the slab,
    a = sin(1000 t) holds at every node without a time derivative: it changes a thousand times
    faster than the temperature, and the steps follow the temperature alone, as without it."""
    tolerance = 1e-5
    alone = slab(num_elements=40).solve_transient(
        end_time=0.1,
        time_step=1e-3,
        time_scheme="Crank-Nicolson",
        time_stepper="error",
        error_tolerance=tolerance,
        report="none",
    )
    coupled = slab(num_elements=40)
    coupled.add_variable("a")
    coupled.add_kernel("parsed_kernel", variable="a", source="a - sin(1000*t)")
    result = coupled.solve_transient(
        end_time=0.1,
        time_step=1e-3,
        time_scheme="Crank-Nicolson",
        time_stepper="error",
        error_tolerance=tolerance,
        report="none",
    )
    # The temperatures of the two runs agree to the Newton tolerance, which may move one decision.
    assert abs(result.time_steps - alone.time_steps) <= 1
    assert coupled.values("a") == pytest.approx(numpy.sin(1000 * 0.1), abs=1e-10)


def test_a_step_that_is_too_inaccurate_is_rejected_and_retried():
    """Given a first step far larger than the tolerance can accept, the
    error-controlled stepper must throw it away, cut back, and recover."""
    problem = slab(num_elements=80)
    result = problem.solve_transient(
        end_time=0.1,
        time_step=0.05,
        time_scheme="Crank-Nicolson",
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
    mesh = dualmesh.generate_line_mesh(start=0.0, end=0.05, num_elements=20)
    problem = dualmesh.Problem(mesh)
    problem.add_variable("temperature")
    problem.set_values("temperature", numpy.full(mesh.num_nodes, 300.0))
    problem.add_kernel("diffusion", variable="temperature", diffusivity=20.0)
    problem.add_kernel(
        "time_derivative",
        variable="temperature",
        coefficient=7800.0 * 460.0,
    )
    problem.add_boundary_condition("Dirichlet", "left", variable="temperature", value=3000.0)
    # Grey-body radiation, e sigma (T^4 - T_a^4) leaving the body, with the emissivity e = 0.9.
    sigma = units.standard_names()["Stefan_Boltzmann_constant"].value
    problem.add_boundary_condition(
        "Neumann",
        "right",
        variable="temperature",
        flux=f"-{sigma!r}*0.9*(temperature^4 - 300^4)",
    )
    result = problem.solve_transient(
        end_time=400.0,
        time_step=400.0,
        time_scheme="backward_Euler",
        time_stepper="iteration",
        max_iterations=2,
        cutback_factor=0.25,
    )
    assert result.converged
    assert result.rejected_steps > 0
    assert numpy.isfinite(problem.values("temperature")).all()
    assert problem.values("temperature").max() == pytest.approx(3000.0, rel=1e-6)


def test_the_run_reports_a_failure_instead_of_grinding_to_a_halt():
    """A step that can never converge must end the run with a clear error
    rather than halving the step for ever."""
    mesh = dualmesh.generate_line_mesh(start=0.0, end=1.0, num_elements=10)
    problem = dualmesh.Problem(mesh)
    problem.add_variable("u")
    problem.add_kernel("diffusion", variable="u")
    problem.add_kernel("time_derivative", variable="u")
    # A reaction with a negative coefficient and a large exponent, which has no
    # bounded solution, so no step size helps.
    problem.add_kernel("reaction", variable="u", coefficient=-1.0e12, exponent=3.0)
    problem.add_boundary_condition("Dirichlet", "left", variable="u", value=1.0)
    with pytest.raises(RuntimeError, match="transient solve failed"):
        problem.solve_transient(
            end_time=1.0,
            time_step=0.5,
            time_stepper="iteration",
            max_iterations=5,
            min_time_step=1e-6,
            max_rejected_steps=4,
        )


def test_the_distributed_solver_takes_the_same_steps():
    """The adaptive controller is driven by global norms, so every rank makes
    the same decision; on one rank the answer must match the serial one."""
    mesh = dualmesh.generate_line_mesh(start=0.0, end=1.0, num_elements=40)

    def define(problem):
        problem.add_variable("temperature")
        problem.add_kernel("diffusion", variable="temperature")
        problem.add_kernel("time_derivative", variable="temperature")
        problem.add_boundary_condition(
            "Dirichlet",
            ["left", "right"],
            variable="temperature",
            value=0.0,
        )
        values = numpy.sin(numpy.pi * problem.node_coordinates()[:, 0])
        values[[0, -1]] = 0.0
        problem.set_values("temperature", values)

    serial = dualmesh.Problem(mesh)
    define(serial)
    serial_result = serial.solve_transient(
        end_time=0.1,
        time_step=0.001,
        time_scheme="Crank-Nicolson",
        time_stepper="error",
        error_tolerance=1e-6,
    )

    distributed = dualmesh.Problem(mesh, distributed=True)
    define(distributed)
    distributed_result = distributed.solve_transient(
        end_time=0.1,
        time_step=0.001,
        time_scheme="Crank-Nicolson",
        time_stepper="error",
        error_tolerance=1e-6,
    )
    assert distributed_result.time_steps == serial_result.time_steps
    got = numpy.asarray(distributed.gathered_values("temperature"))
    assert got == pytest.approx(serial.values("temperature"), abs=1e-8)
