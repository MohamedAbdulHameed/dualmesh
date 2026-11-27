# SPDX-License-Identifier: LGPL-2.1-or-later
"""The eigenvalue study of any problem (Problem.solve_eigenvalue).

Every field varies in time as u_hat exp(-lambda t), and the expressions may
use the symbol eigenvalue. The checks have exact answers:

* the Laplacian on the unit square with zero values on the boundary has the
  eigenvalues pi^2 (m^2 + n^2): 2 pi^2, 5 pi^2 (twice), 8 pi^2, and every
  method converges to them at second order;
* the eigenvalue written as a symbol gives the eigenvalues of the time
  derivative, and a quadratic symbol gives their square roots;
* the non-symmetric operator -u'' + c u' on (0, 1) has the eigenvalues
  (n pi)^2 + c^2 / 4;
* two diffusion equations coupled by a rotation have the complex eigenvalues
  2 pi^2 +- i omega;
* an equation that is not at most quadratic in the eigenvalue, and a problem
  that defines no eigenvalue, are refused.
"""

from __future__ import annotations

import dualmesh
import numpy
import pytest
from helpers import unit_square

methods = ("fem", "dmcdm")
laplacian = numpy.pi**2 * numpy.array([2.0, 5.0, 5.0, 8.0])


def clamp(problem, variables=("u",)):
    for variable in variables:
        problem.add_boundary_condition(
            "Dirichlet",
            list(problem.mesh.sideset_names()),
            variable=variable,
            value=0.0,
        )


def laplacian_modes(method, n, source=None, near=0.0, num_modes=4):
    problem = dualmesh.Problem(unit_square(n), method=method)
    if source is None:
        problem.add_physics(
            "general_form_PDE",
            variables="u",
            c=1.0,
            d=1.0,
        )
    else:
        problem.add_physics(
            "general_form_PDE",
            variables="u",
            equations=f"-laplacian(u) = {source}",
        )
    clamp(problem)
    return problem, problem.solve_eigenvalue(num_modes=num_modes, near=near, report="none")


@pytest.mark.parametrize("method", methods)
def test_the_laplacian_eigenvalues_converge_at_second_order(method):
    errors = []
    for n in (8, 16, 32):
        _problem, result = laplacian_modes(method, n)
        values = numpy.sort(result.eigenvalues)
        errors.append(numpy.abs(values - laplacian) / laplacian)
    orders = numpy.log2(errors[1] / errors[2])
    assert orders.min() > 1.9
    # The leading error of a second-order method grows as lambda h^2 / 12.
    assert numpy.all(errors[2] < laplacian * (1.0 / 32) ** 2 / 8)


@pytest.mark.parametrize("method", ["fem", "dmcdm"])
def test_a_repeated_eigenvalue_is_found_with_its_multiplicity(method):
    """5 pi^2 is double. Asked for exactly four modes, the study returns it
    twice."""
    for n in (8, 16):
        _problem, result = laplacian_modes(method, n)
        values = numpy.sort(result.eigenvalues)
        assert values[1] == pytest.approx(values[2], rel=1e-8)


@pytest.mark.parametrize("method", methods)
def test_the_eigenvalue_as_a_symbol_is_the_eigenvalue_of_the_time_derivative(method):
    """-div grad u = eigenvalue u gives the same eigenvalues as the time
    derivative du/dt - div grad u = 0, to round-off."""
    _p, time_form = laplacian_modes(method, 12)
    _p, symbol = laplacian_modes(method, 12, source="eigenvalue*u")
    assert numpy.sort(symbol.eigenvalues) == pytest.approx(
        numpy.sort(time_form.eigenvalues), rel=1e-10
    )
    assert not symbol.quadratic


def test_a_quadratic_symbol_gives_the_square_roots():
    """-div grad u = eigenvalue^2 u: the eigenvalues are lambda = +- pi sqrt(m^2 + n^2),
    and near is a value of lambda itself. Closest to 4 are pi sqrt(2), pi sqrt(5) twice
    and pi sqrt(8). Closest to 8.9 is the simple eigenvalue 2 sqrt(2) pi = 8.886 of the
    mode m = n = 2, the fourth from the lowest, and closest to -8.9 is -2 sqrt(2) pi. A
    near read as an angular frequency, as the square of lambda, or not read at all
    would select pi sqrt(2) = 4.443 instead."""
    _p, linear = laplacian_modes("fem", 24)
    _p, quadratic = laplacian_modes("fem", 24, source="eigenvalue^2*u", near=4.0)
    assert quadratic.quadratic
    assert not quadratic.vibration
    roots = numpy.sqrt(numpy.sort(linear.eigenvalues))
    assert numpy.sort(quadratic.eigenvalues) == pytest.approx(roots, rel=1e-8)
    # The fourth root is that of the mode m = n = 2, to the error of the mesh.
    assert roots[3] == pytest.approx(2.0 * numpy.sqrt(2.0) * numpy.pi, rel=2e-2)
    for near in (8.9, -8.9):
        _p, chosen = laplacian_modes("fem", 24, source="eigenvalue^2*u", near=near, num_modes=1)
        assert chosen.eigenvalues == pytest.approx([numpy.sign(near) * roots[3]], rel=1e-8)


@pytest.mark.parametrize("method", methods)
def test_a_non_symmetric_operator_has_its_exact_eigenvalues(method):
    """-u'' + c u' = lambda u on (0, 1), u(0) = u(1) = 0: lambda_n = (n pi)^2 +
    c^2 / 4. The convection makes the operator non-symmetric."""
    c = 4.0
    exact = (numpy.arange(1, 4) * numpy.pi) ** 2 + c**2 / 4.0
    errors = []
    for n in (64, 128):
        problem = dualmesh.Problem(dualmesh.generate_line_mesh(0.0, 1.0, n), method=method)
        problem.add_physics(
            "general_form_PDE",
            variables="u",
            c=1.0,
            beta=[c],
            d=1.0,
        )
        problem.add_boundary_condition(
            "Dirichlet",
            ["left", "right"],
            variable="u",
            value=0.0,
        )
        result = problem.solve_eigenvalue(num_modes=3, report="none")
        errors.append(numpy.abs(numpy.sort(numpy.real(result.eigenvalues)) - exact) / exact)
    assert errors[1].max() < 2e-3
    assert numpy.log2(errors[0] / errors[1]).min() > 1.8


def test_coupled_diffusion_with_a_rotation_has_complex_eigenvalues():
    """du/dt - div grad u = -w v and dv/dt - div grad v = w u: lambda = mu +- i w,
    where mu is an eigenvalue of the Laplacian."""
    omega = 30.0
    problem = dualmesh.Problem(unit_square(16), method="fem")
    problem.add_physics(
        "general_form_PDE",
        variables=["u", "v"],
        w=omega,
        equations=["du/dt - laplacian(u) = -w*v", "dv/dt - laplacian(v) = w*u"],
    )
    clamp(problem, ("u", "v"))
    result = problem.solve_eigenvalue(num_modes=2, near=20.0, report="none")
    _p, scalar = laplacian_modes("fem", 16, num_modes=1)
    mu = scalar.eigenvalues[0]
    assert numpy.sort_complex(result.eigenvalues) == pytest.approx(
        [mu - 1j * omega, mu + 1j * omega], rel=1e-8
    )
    assert set(result.modes[0]) == {"u", "v"}


@pytest.mark.parametrize(
    "source, message",
    [
        ("eigenvalue^3*u", "more than quadratically"),
        (None, "defines no eigenvalue"),
    ],
)
def test_an_equation_without_a_proper_eigenvalue_is_refused(source, message):
    problem = dualmesh.Problem(unit_square(4), method="fem")
    problem.add_physics(
        "general_form_PDE", variables="u", equations=f"-laplacian(u) = {source or 0}"
    )
    clamp(problem)
    with pytest.raises(ValueError, match=message):
        problem.solve_eigenvalue(report="none")


def test_too_many_modes_are_refused():
    problem = dualmesh.Problem(unit_square(2), method="fem")
    problem.add_physics("general_form_PDE", variables="u", c=1.0, d=1.0)
    clamp(problem)
    with pytest.raises(ValueError, match="Ask for fewer modes"):
        problem.solve_eigenvalue(num_modes=5, report="none")


def cooling_square(method, quadrature="automatic"):
    """du/dt = laplacian(u) on the unit square of linear elements, zero on the boundary, whose
    automatic storage time stepping lumps."""
    problem = dualmesh.Problem(unit_square(8), method=method)
    problem.add_variable("u")
    problem.add_kernel("diffusion", variable="u")
    problem.add_kernel("time_derivative", variable="u", quadrature=quadrature)
    clamp(problem)
    return problem


def cooled(problem):
    """The field after a transient from u = sin(pi x) sin(pi y) + x y (1 - x) (1 - y)."""
    problem.initialize()
    x, y = problem.node_coordinates()[:, 0], problem.node_coordinates()[:, 1]
    problem.set_values(
        "u", numpy.sin(numpy.pi * x) * numpy.sin(numpy.pi * y) + x * y * (1 - x) * (1 - y)
    )
    problem.solve_transient(end_time=0.05, time_step=0.01, report="none")
    return problem.values("u")


@pytest.mark.parametrize("method", methods)
def test_a_transient_after_an_eigenvalue_study_lumps_its_storage_again(method):
    """The eigenvalue study takes the consistent storage, and the transient solve after it on the
    same problem takes the lumped storage of time stepping again: it gives the field of the same
    transient on a fresh problem to round-off, and not that of the consistent storage."""
    studied = cooling_square(method)
    studied.solve_eigenvalue(num_modes=2, report="none")
    after = cooled(studied)
    fresh = cooled(cooling_square(method))
    assert numpy.abs(after - fresh).max() < 1e-12 * numpy.abs(fresh).max()
    consistent = cooled(cooling_square(method, quadrature="gauss2"))
    assert numpy.abs(consistent - fresh).max() > 1e-6 * numpy.abs(fresh).max()
