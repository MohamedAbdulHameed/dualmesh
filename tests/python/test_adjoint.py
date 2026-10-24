# SPDX-License-Identifier: LGPL-2.1-or-later
"""Adjoint sensitivities: d(objective)/d(parameter) of a converged solution,
checked against the central difference of the same objective over two solves
with the parameter moved, which is the definition of the derivative and needs
no theory of the adjoint to be trusted."""

from __future__ import annotations

import dualmesh as dm
import pytest
from dualmesh._core import InputError

METHODS = ("fem", "dmcdm", "hfvm", "zfvm")


def heat_problem(method="fem", nonlinear=False):
    """Conduction on a square with a source q, conductivity k (linear in the
    temperature when nonlinear), and 0 and T0 on the walls.  Every number the
    residual depends on is a named settable parameter."""
    mesh = dm.generate_rectangle_mesh(0.0, 1.0, 0.0, 1.0, 12, 12)
    problem = dm.Problem(mesh, method=method)
    problem.add_parameter("k", 2.0)
    problem.add_parameter("q", 6.0)
    problem.add_parameter("T0", 0.5)
    problem.add_variable("temperature")
    problem.add_kernel(
        "heat_conduction",
        variable="temperature",
        thermal_conductivity="k",
        **({"temperature_polynomial": [1.0, 0.05]} if nonlinear else {}),
    )
    problem.add_kernel("body_force", variable="temperature", value="q")
    problem.add_boundary_condition("Dirichlet", "left", variable="temperature", value=0.0)
    problem.add_boundary_condition("Dirichlet", "right", variable="temperature", value="T0")
    problem.solve(report="none")
    return problem


def solved_gradient(problem, name, objective):
    """d(objective)/d(name) by the central difference of two solves, the
    reference the adjoint is checked against."""
    p0 = problem._problem.parameter_value(name)
    h = 1e-6 * max(1.0, abs(p0))

    def objective_at(value):
        problem.set_parameter(name, value)
        problem.solve(report="none")
        if isinstance(objective, str):
            return problem.integrate(objective)
        return problem.boundary_integral(objective["variable"], objective["boundary"])

    plus, minus = objective_at(p0 + h), objective_at(p0 - h)
    problem.set_parameter(name, p0)
    problem.solve(report="none")
    return (plus - minus) / (2.0 * h)


def agree(derivative, reference):
    scale = max(1.0, abs(reference))
    return abs(derivative - reference) <= 1e-5 * scale


def test_the_sensitivities_of_linear_conduction_match_the_solves():
    problem = heat_problem()
    result = problem.sensitivities("temperature", ["k", "q", "T0"], report="none")
    for name in ("k", "q", "T0"):
        assert agree(result.values[name], solved_gradient(problem, name, "temperature")), name


@pytest.mark.parametrize("method", METHODS)
def test_the_sensitivities_of_every_method_match(method):
    problem = heat_problem(method=method)
    result = problem.sensitivities("temperature", ["q"], report="none")
    assert agree(result.values["q"], solved_gradient(problem, name="q", objective="temperature"))


def test_the_adjoint_solve_uses_the_jacobian_of_a_nonlinear_problem():
    problem = heat_problem(nonlinear=True)
    result = problem.sensitivities("temperature", ["k", "T0"], report="none")
    for name in ("k", "T0"):
        assert agree(result.values[name], solved_gradient(problem, name, "temperature")), name


def test_the_objective_may_be_an_integral_over_a_boundary():
    problem = heat_problem()
    objective = {"variable": "temperature", "boundary": "top"}
    result = problem.sensitivities(objective, ["q", "T0"], report="none")
    for name in ("q", "T0"):
        assert agree(result.values[name], solved_gradient(problem, name, objective)), name


def test_the_result_follows_the_protocol(tmp_path):
    problem = heat_problem()
    result = problem.sensitivities("temperature", ["q"], report="none")
    assert "q" in result.summary()
    assert result.to_dict()["sensitivities"]["q"] == pytest.approx(result.values["q"])
    result.write_json(tmp_path / "s.json")
    result.write_csv(tmp_path / "s.csv")
    assert (tmp_path / "s.json").exists() and (tmp_path / "s.csv").exists()


def test_wrong_names_are_refused_with_an_explanation():
    problem = heat_problem()
    problem.add_function("spatial", lambda x, t: 1.0)
    problem.solve(report="none")
    with pytest.raises(InputError, match="not a registered function"):
        problem.sensitivities("temperature", ["nope"], report="none")
    with pytest.raises(InputError, match="not a settable constant"):
        problem.sensitivities("temperature", ["spatial"], report="none")
    with pytest.raises(InputError, match="already a registered function"):
        problem.add_parameter("q", 3.0)
    with pytest.raises(InputError, match="objective"):
        problem.sensitivities(42, ["q"], report="none")
    with pytest.raises(InputError, match="parameters"):
        problem.sensitivities("temperature", [], report="none")


def test_a_solution_that_is_not_of_the_model_as_it_stands_is_refused():
    problem = heat_problem()
    problem.set_parameter("q", 7.0)
    with pytest.raises(InputError, match="no converged steady solution"):
        problem.sensitivities("temperature", ["q"], report="none")
    problem.solve(report="none")
    problem.sensitivities("temperature", ["q"], report="none")
    problem.add_kernel("time_derivative", variable="temperature")
    problem.solve_transient(end_time=0.1, time_step=0.05, report="none")
    with pytest.raises(InputError, match="no converged steady solution"):
        problem.sensitivities("temperature", ["q"], report="none")


def test_a_problem_with_a_material_history_is_refused():
    mesh = dm.generate_rectangle_mesh(0.0, 1.0, 0.0, 1.0, 2, 2)
    problem = dm.Problem(mesh, method="fem")
    problem.add_parameter("E", 2.0e11)
    for name in ("u", "v", "T"):
        problem.add_variable(name)
    problem.add_kernel("diffusion", variable="T")
    problem.add_boundary_condition("Dirichlet", "left", variable="T", value=900.0)
    problem.add_boundary_condition("Dirichlet", "right", variable="T", value=900.0)
    problem.add_property(
        "small_strain_stress",
        displacements=["u", "v"],
        formulation="plane_strain",
        youngs_modulus="E",
        creep_model="parsed",
        creep_rate="1e-30*von_mises_stress^3",
        temperature="T",
    )
    problem.add_kernel("stress_divergence", displacements=["u", "v"])
    problem.add_boundary_condition("Dirichlet", "left", variable="u", value=0.0)
    problem.add_boundary_condition("Dirichlet", "bottom", variable="v", value=0.0)
    problem.solve(report="none")
    with pytest.raises(InputError, match="material history"):
        problem.sensitivities("u", ["E"], report="none")
