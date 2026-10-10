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

import dualmesh as dm
import numpy as np
import pytest

METHODS = ("fem", "dmcdm", "hfvm", "zfvm")
LAPLACIAN = np.pi**2 * np.array([2.0, 5.0, 5.0, 8.0])


def unit_square(n):
    return dm.generate_rectangle_mesh(
        x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0, num_x_elements=n, num_y_elements=n
    )


def clamp(problem, variables=("u",)):
    for variable in variables:
        problem.add_boundary_condition(
            "Dirichlet_boundary_condition",
            f"walls_{variable}",
            variable=variable,
            boundary=list(problem.mesh.sideset_names()),
            value=0.0,
        )


def laplacian_modes(method, n, source=None, near=0.0, num_modes=4):
    problem = dm.Problem(unit_square(n), method=method)
    if source is None:
        problem.add_physics(
            "coefficient_form_PDE",
            "pde",
            diffusion_coefficient=1.0,
            time_derivative_coefficient=1.0,
        )
    else:
        problem.add_physics(
            "general_form_PDE",
            "pde",
            variables=["u"],
            flux={"u": ["-grad_x(u)", "-grad_y(u)"]},
            source={"u": source},
        )
    clamp(problem)
    return problem, problem.solve_eigenvalue(num_modes=num_modes, near=near, report="none")


@pytest.mark.parametrize("method", METHODS)
def test_the_laplacian_eigenvalues_converge_at_second_order(method):
    errors = []
    for n in (8, 16, 32):
        _problem, result = laplacian_modes(method, n)
        values = np.sort(result.eigenvalues)
        errors.append(np.abs(values - LAPLACIAN) / LAPLACIAN)
    orders = np.log2(errors[1] / errors[2])
    assert orders.min() > 1.9
    # The leading error of a second-order method grows as lambda h^2 / 12.
    assert np.all(errors[2] < LAPLACIAN * (1.0 / 32) ** 2 / 8)


@pytest.mark.parametrize("method", ["fem", "dmcdm"])
def test_a_repeated_eigenvalue_is_found_with_its_multiplicity(method):
    """5 pi^2 is double. Asked for exactly four modes, the study returns it
    twice."""
    for n in (8, 16):
        _problem, result = laplacian_modes(method, n)
        values = np.sort(result.eigenvalues)
        assert values[1] == pytest.approx(values[2], rel=1e-8)


@pytest.mark.parametrize("method", METHODS)
def test_the_eigenvalue_as_a_symbol_is_the_eigenvalue_of_the_time_derivative(method):
    """-div grad u = eigenvalue u gives the same eigenvalues as the time
    derivative du/dt - div grad u = 0, to round-off."""
    _p, time_form = laplacian_modes(method, 12)
    _p, symbol = laplacian_modes(method, 12, source="eigenvalue*u")
    assert np.sort(symbol.eigenvalues) == pytest.approx(np.sort(time_form.eigenvalues), rel=1e-10)
    assert not symbol.quadratic


def test_a_quadratic_symbol_gives_the_square_roots():
    """-div grad u = eigenvalue^2 u: the eigenvalues closest to 4 are pi sqrt(2),
    pi sqrt(5) twice and pi sqrt(8), and the study solves the companion form."""
    _p, linear = laplacian_modes("fem", 24)
    _p, quadratic = laplacian_modes("fem", 24, source="eigenvalue^2*u", near=4.0)
    assert quadratic.quadratic
    roots = np.sqrt(np.sort(linear.eigenvalues))
    assert np.sort(quadratic.eigenvalues) == pytest.approx(roots, rel=1e-8)


@pytest.mark.parametrize("method", METHODS)
def test_a_non_symmetric_operator_has_its_exact_eigenvalues(method):
    """-u'' + c u' = lambda u on (0, 1), u(0) = u(1) = 0: lambda_n = (n pi)^2 +
    c^2 / 4. The convection makes the operator non-symmetric."""
    c = 4.0
    exact = (np.arange(1, 4) * np.pi) ** 2 + c**2 / 4.0
    errors = []
    for n in (64, 128):
        problem = dm.Problem(dm.generate_line_mesh(0.0, 1.0, n), method=method)
        problem.add_physics(
            "coefficient_form_PDE",
            "pde",
            diffusion_coefficient=1.0,
            convection_coefficient=[c],
            time_derivative_coefficient=1.0,
        )
        problem.add_boundary_condition(
            "Dirichlet_boundary_condition",
            "ends",
            variable="u",
            boundary=["left", "right"],
            value=0.0,
        )
        result = problem.solve_eigenvalue(num_modes=3, report="none")
        errors.append(np.abs(np.sort(np.real(result.eigenvalues)) - exact) / exact)
    assert errors[1].max() < 2e-3
    assert np.log2(errors[0] / errors[1]).min() > 1.8


def test_coupled_diffusion_with_a_rotation_has_complex_eigenvalues():
    """du/dt - div grad u = -w v and dv/dt - div grad v = w u: lambda = mu +- i w,
    where mu is an eigenvalue of the Laplacian."""
    omega = 30.0
    problem = dm.Problem(unit_square(16), method="fem")
    problem.add_physics(
        "general_form_PDE",
        "rotation",
        variables=["u", "v"],
        flux={"u": ["-grad_x(u)", "-grad_y(u)"], "v": ["-grad_x(v)", "-grad_y(v)"]},
        source={"u": "-w*v", "v": "w*u"},
        time_derivative_coefficient={"u": 1.0, "v": 1.0},
        constants={"w": omega},
    )
    clamp(problem, ("u", "v"))
    result = problem.solve_eigenvalue(num_modes=2, near=20.0, report="none")
    _p, scalar = laplacian_modes("fem", 16, num_modes=1)
    mu = scalar.eigenvalues[0]
    assert np.sort_complex(result.eigenvalues) == pytest.approx(
        [mu - 1j * omega, mu + 1j * omega], rel=1e-8
    )
    assert set(result.modes[0]) == {"u", "v"}


def test_the_result_and_the_first_mode():
    problem, result = laplacian_modes("fem", 10, num_modes=2)
    mode = result.modes[0]["u"]
    assert np.abs(mode).max() == pytest.approx(1.0)
    assert np.asarray(problem.values("u")) == pytest.approx(np.real(mode))
    assert result.residual_norms.max() < 1e-8
    assert "Eigenvalue study" in result.summary()
    data = result.to_dict()
    assert data["eigenvalues_real"][0] == pytest.approx(float(np.real(result.eigenvalues[0])))
    assert list(result.tables) == ["eigenvalues"]


def test_the_result_writes_its_table(tmp_path):
    _problem, result = laplacian_modes("fem", 6, num_modes=2)
    result.write_csv(tmp_path / "modes.csv")
    header = (tmp_path / "modes.csv").read_text().splitlines()[0]
    assert header.startswith("mode,eigenvalue_real,eigenvalue_imaginary")


@pytest.mark.parametrize(
    "source, message",
    [
        ("eigenvalue^3*u", "more than quadratically"),
        (None, "defines no eigenvalue"),
    ],
)
def test_an_equation_without_a_proper_eigenvalue_is_refused(source, message):
    problem = dm.Problem(unit_square(4), method="fem")
    sources = {} if source is None else {"u": source}
    problem.add_physics(
        "general_form_PDE",
        "pde",
        variables=["u"],
        flux={"u": ["-grad_x(u)", "-grad_y(u)"]},
        source=sources,
    )
    clamp(problem)
    with pytest.raises(ValueError, match=message):
        problem.solve_eigenvalue(report="none")


def test_too_many_modes_are_refused():
    problem = dm.Problem(unit_square(2), method="fem")
    problem.add_physics(
        "coefficient_form_PDE", "pde", diffusion_coefficient=1.0, time_derivative_coefficient=1.0
    )
    clamp(problem)
    with pytest.raises(ValueError, match="Ask for fewer modes"):
        problem.solve_eigenvalue(num_modes=5, report="none")
