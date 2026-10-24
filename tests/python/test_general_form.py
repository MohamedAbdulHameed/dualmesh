# SPDX-License-Identifier: LGPL-2.1-or-later
"""Systems of partial differential equations in general form: parsed_kernel
and general_form_PDE.

For every field u, d du/dt + div Gamma = f, where the flux Gamma and the
source f are expressions of all the fields, their gradients, x, y, z and t.
The checks:

* the general form of an equation reproduces the dedicated kernels to
  round-off (diffusion with a source, and linear elasticity against
  solid_mechanics);
* manufactured solutions of a nonlinear coupled system with fluxes that
  depend on the gradients converge at the optimal order on every method and
  element, in two and three dimensions and in axisymmetric coordinates, and
  in time;
* the Jacobian agrees with finite differences on every method;
* every incomplete or wrong input is refused with a message that says what
  to do.
"""

from __future__ import annotations

import dualmesh as dm
import numpy as np
import pytest

sympy = pytest.importorskip("sympy")
from dualmesh import mms  # noqa: E402
from test_jacobian import compare, finite_difference_jacobian  # noqa: E402
from test_mms import DUAL_MESH_TYPES, METHODS, check, cube_family, square_family  # noqa: E402

# A nonlinear system of two fields: a p-Laplacian-like flux for u, a flux of v
# whose coefficient depends on u with a convective part, and sources that couple
# the two through the values and a gradient.
SYSTEM_EQUATIONS = [
    "div(-(1 + dot(grad(u), grad(u)))*grad(u)) = -u*v",
    "div(-(1 + u^2)*grad(v) + [0.3*v, 0]) = -k*v*u + grad_x(u)",
]
SYSTEM_FIELDS = {"u": "1 + sin(x)*cos(y)", "v": "x**2 + x*y + 0.5"}


def system_study(fields=SYSTEM_FIELDS, dimension=2, coordinates="cartesian", **parameters):
    study = mms.ManufacturedSolution(fields, dimension=dimension, coordinates=coordinates)
    options = dict(variables=["u", "v"], k=2.0, equations=SYSTEM_EQUATIONS)
    options.update(parameters)
    study.add_physics("general_form_PDE", **options)
    return study


def unit_square(n=6, element_type="Quad4"):
    return dm.generate_rectangle_mesh(
        x_min=0.0,
        x_max=1.0,
        y_min=0.0,
        y_max=1.0,
        num_x_elements=n,
        num_y_elements=n - 1,
        element_type=element_type,
    )


# ---------------------------------------------------------------------------
# The general form against the dedicated kernels
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("method", METHODS)
def test_the_general_form_of_diffusion_is_the_diffusion_kernel(method):
    """-div(2.5 grad u) = 3 + x y: the parsed flux and source give the same
    solution as the diffusion kernel, to the last bit."""

    def solve(general):
        problem = dm.Problem(unit_square(), method=method)
        problem.add_variable("u")
        if general:
            flux = ["-2.5*grad_x(u)", "-2.5*grad_y(u)"]
            problem.add_kernel("parsed_kernel", variable="u", flux=flux, source="3 + x*y")
        else:
            problem.add_kernel("diffusion", variable="u", diffusivity=2.5)
            problem.add_kernel("parsed_kernel", variable="u", source="3 + x*y")
        problem.add_boundary_condition(
            "Dirichlet",
            ["left", "bottom"],
            variable="u",
            value=0.0,
        )
        problem.solve(report="none")
        return problem.values("u")

    assert np.array_equal(solve(True), solve(False))


@pytest.mark.parametrize("method", ["fem", "dmcdm", "hfvm"])
def test_linear_elasticity_in_general_form_is_solid_mechanics(method):
    """Plane strain written as two equations of general form, with the stress
    of Hooke's law as the flux, gives the displacements of solid_mechanics to
    round-off."""
    youngs_modulus, poissons_ratio = 2.0e11, 0.3
    lame = youngs_modulus * poissons_ratio / ((1 + poissons_ratio) * (1 - 2 * poissons_ratio))
    shear = youngs_modulus / (2 * (1 + poissons_ratio))

    def solve(general):
        problem = dm.Problem(unit_square(5), method=method)
        if general:
            displacements = ["U_x", "U_y"]
            physics = problem.add_physics(
                "general_form_PDE",
                variables=[{"U": "vector"}],
                lam=lame,
                mu=shear,
                strain="sym(grad(U))",
                stress="lam*trace(strain)*I + 2*mu*strain",
                equations="div(stress) + [0, -1e9] = 0",
            )
        else:
            displacements = ["disp_x", "disp_y"]
            physics = problem.add_physics(
                "solid_mechanics",
                formulation="plane_strain",
                youngs_modulus=youngs_modulus,
                poissons_ratio=poissons_ratio,
                body_force=[0.0, -1e9],
            )
        for variable in displacements:
            problem.add_boundary_condition(
                "Dirichlet",
                "left",
                variable=variable,
                value=0.0,
            )
        del physics
        problem.solve(report="none")
        return np.concatenate([problem.values(v) for v in displacements])

    general, dedicated = solve(True), solve(False)
    assert np.abs(general - dedicated).max() < 1e-12 * np.abs(dedicated).max()


# ---------------------------------------------------------------------------
# Manufactured solutions
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("element_type", ["Tri3", "Quad4", "Quad9"])
@pytest.mark.parametrize("method", METHODS)
def test_a_nonlinear_system_converges_at_the_optimal_order(method, element_type):
    if method != "fem" and element_type not in DUAL_MESH_TYPES:
        pytest.skip(f"{element_type} has no dual mesh")
    result = system_study().convergence_study(
        square_family(element_type), [4, 8, 16, 32], method=method, report="none"
    )
    check(result, "u", method, element_type)
    check(result, "v", method, element_type)


@pytest.mark.parametrize("method", METHODS)
def test_a_nonlinear_system_converges_in_three_dimensions(method):
    fields = {"u": "1 + sin(x)*cos(y)*exp(0.5*z)", "v": "x**2 + y*z + 0.5"}
    equations = [
        "div(-(1 + du/dx^2)*grad(u)) = -u*v",
        "div([-(1 + u^2)*dv/dx, -(1 + u^2)*dv/dy, -dv/dz]) = -k*v*u + du/dx",
    ]
    study = system_study(fields, dimension=3, equations=equations)
    result = study.convergence_study(cube_family("Hex8"), [2, 4, 8], method=method, report="none")
    check(result, "u", method, "Hex8")
    check(result, "v", method, "Hex8")


@pytest.mark.parametrize("method", METHODS)
def test_a_nonlinear_system_converges_in_axisymmetric_coordinates(method):
    """The divergence carries the factor of the radius, r^-1 d(r Gamma_r)/dr."""
    result = system_study(coordinates="axisymmetric").convergence_study(
        square_family("Quad4", 0.5, 1.5),
        [4, 8, 16, 32],
        method=method,
        report="none",
    )
    check(result, "u", method, "Quad4")
    check(result, "v", method, "Quad4")


@pytest.mark.parametrize("method", METHODS)
def test_a_transient_system_converges_in_space_and_time(method):
    """Crank-Nicolson with time_step = h / 4: second order in space and time
    together."""
    fields = {"u": "exp(-t)*(1 + sin(x)*cos(y))", "v": "(1 + t)*(x**2 + 0.5)"}
    equations = [
        "du/dt + div(-(1 + dot(grad(u), grad(u)))*grad(u)) = -u*v",
        "2*dv/dt + div(-(1 + u^2)*grad(v) + [0.3*v, 0]) = -k*v*u + grad_x(u)",
    ]
    study = system_study(fields, equations=equations)
    result = study.convergence_study(
        square_family("Quad4"),
        [4, 8, 16, 32],
        method=method,
        report="none",
        transient={"end_time": 0.25, "time_step": lambda h: h / 4, "implicitness": 0.5},
    )
    check(result, "u", method, "Quad4", norms=("l2",))
    check(result, "v", method, "Quad4", norms=("l2",))


# ---------------------------------------------------------------------------
# The Jacobian
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("method", METHODS)
def test_the_jacobian_of_a_nonlinear_system_agrees_with_finite_differences(method):
    problem = dm.Problem(unit_square(4), method=method)
    problem.add_physics(
        "general_form_PDE",
        variables=["u", "v"],
        k=2.0,
        equations=[
            "div(-(1 + dot(grad(u), grad(u)))*grad(u)) = -u*v + sin(x)",
            "div(-(1 + u^2)*grad(v) + [0.3*v, 0]) = -k*v*u^3 + du/dx*dv/dy + t",
        ],
    )
    problem.add_boundary_condition("Dirichlet", "left", variable="u", value=0.0)
    problem.add_boundary_condition("Dirichlet", "right", variable="v", value=0.0)
    problem.initialize()
    points = problem.entity_points()
    x, y = points[:, 0], points[:, 1]
    problem.set_values("u", 0.3 + np.sin(2 * x) * y)
    problem.set_values("v", 0.2 * x**2 - 0.4 * y)
    _, jacobian = problem.linear_system()
    assert compare(jacobian.toarray(), finite_difference_jacobian(problem)) < 1e-7


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
def add_system(**parameters):
    problem = dm.Problem(unit_square(3))
    options = dict(variables=["u"], equations="-laplacian(u) = 1")
    options.update(parameters)
    problem.add_physics("general_form_PDE", **options)
    problem.initialize()
    return problem


@pytest.mark.parametrize(
    "parameters, message",
    [
        ({"variables": []}, "give the fields in 'variables'"),
        ({"variables": ["u", "u"]}, "every name in 'variables' must be different"),
        ({"equations": {"w": "-laplacian(u) = 1"}}, "names a field that is not in 'variables'"),
        ({"variables": ["u", "v"]}, r"2 field\(s\) \(u, v\) and 1 equation\(s\)"),
        ({"equations": "-laplacian(w) = 1"}, "'w' would be differentiated"),
        ({"equations": "-div([du/dx, du/dz]) = 1"}, "z is not a coordinate"),
        ({"equations": "-div([du/dx]) = 1"}, "has 2 components"),
        ({"equations": "-laplacian(u) = q*u"}, "unknown name 'q'"),
        ({"u": 1.0}, "cannot be defined"),
    ],
)
def test_an_incomplete_or_wrong_system_is_refused(parameters, message):
    with pytest.raises(Exception, match=message):
        add_system(**parameters)


def test_parsed_kernel_needs_a_flux_or_a_source():
    problem = dm.Problem(unit_square(3))
    problem.add_variable("u")
    with pytest.raises(Exception, match="give a 'flux', a 'source' or both"):
        problem.add_kernel("parsed_kernel", variable="u")


# ---------------------------------------------------------------------------
# Conditions that depend on the fields
# ---------------------------------------------------------------------------
def nonlinear_condition_problem(method, n):
    """The system on a square whose side 'right' carries only conditions that
    depend on the fields, each nonzero at the exact solution: for u, a Robin
    condition with the transfer coefficient u, the ambient value 0 and the
    flux q + u_exact^2, so that n . F = q + u_exact^2 - u^2; for v, a Neumann
    flux q - u_exact + u. Here q is the exact flux n . F of each equation. A
    wrong value of u, or a wrong factor, in either condition breaks the exact
    solution."""
    study = mms.ManufacturedSolution(SYSTEM_FIELDS, dimension=2)
    study.add_physics(
        "general_form_PDE",
        variables=["u", "v"],
        k=2.0,
        equations=SYSTEM_EQUATIONS,
    )
    problem = study.build(
        square_family("Quad4")(n), method=method, boundary=["left", "bottom", "top"]
    )
    exact = study.exact("u")
    normal_flux = {v: str(study.flux(v)[0]) for v in ("u", "v")}
    problem.add_boundary_condition(
        "Robin",
        "right",
        variable="u",
        flux=f"({normal_flux['u']}) + ({exact})^2",
        transfer_coeff="u",
        ambient_value=0.0,
    )
    problem.add_boundary_condition(
        "Neumann",
        "right",
        variable="v",
        flux=f"({normal_flux['v']}) - ({exact}) + u",
    )
    return study, problem


@pytest.mark.parametrize("method", METHODS)
def test_conditions_that_depend_on_the_fields_keep_the_optimal_order(method):
    errors = []
    for n in (4, 8, 16, 32):
        study, problem = nonlinear_condition_problem(method, n)
        problem.solve(report="none")
        errors.append(study.errors(problem)[("u", "l2")])
    orders = np.log2(np.array(errors[:-1]) / np.array(errors[1:]))
    assert orders[-1] > 1.9


@pytest.mark.parametrize("method", METHODS)
def test_the_jacobian_of_conditions_that_depend_on_the_fields(method):
    _study, problem = nonlinear_condition_problem(method, 4)
    problem.initialize()
    points = problem.entity_points()
    problem.set_values("u", 0.3 + np.sin(2 * points[:, 0]) * points[:, 1])
    _, jacobian = problem.linear_system()
    assert compare(jacobian.toarray(), finite_difference_jacobian(problem)) < 1e-7


@pytest.mark.parametrize("method", METHODS)
def test_a_time_coefficient_that_depends_on_the_field_has_an_exact_jacobian(method):
    problem = dm.Problem(unit_square(4), method=method)
    pde = problem.add_physics(
        "general_form_PDE", variables="u", equations="(1 + u^2)*du/dt = laplacian(u) + 1"
    )
    pde.add_boundary_condition("u = 0", boundary="left")
    problem.solve_transient(end_time=0.02, time_step=0.01, report="none")
    points = problem.entity_points()
    problem.set_values("u", 0.3 + np.sin(2 * points[:, 0]) * points[:, 1])
    _, jacobian = problem.linear_system()
    assert compare(jacobian.toarray(), finite_difference_jacobian(problem)) < 1e-7


@pytest.mark.parametrize(
    "variables",
    [[{"U": "vector"}, "p"], [{"U": "vector"}, {"p": "scalar"}]],
)
def test_a_field_without_a_kind_is_scalar(variables):
    problem = dm.Problem(unit_square(3), method="fem")
    pde = problem.add_physics(
        "general_form_PDE",
        variables=variables,
        equations=["div(-p*I + grad(U)) = 0", "div(U) = 0"],
    )
    assert pde.variable_names() == ["U_x", "U_y", "p"]


def test_fields_are_given_one_per_item():
    problem = dm.Problem(unit_square(3), method="fem")
    with pytest.raises(Exception, match="give each field as its own item"):
        problem.add_physics(
            "general_form_PDE",
            variables={"U": "vector", "p": "scalar"},
            equations=["div(-p*I + grad(U)) = 0", "div(U) = 0"],
        )


def test_normal_flux_is_told_by_the_kind_of_its_value():
    # A solid and its temperature: a vector value is the traction of U, a
    # scalar value the heat flux of T.
    def solve():
        problem = dm.Problem(unit_square(4), method="fem")
        pde = problem.add_physics(
            "general_form_PDE",
            variables=[{"U": "vector"}, "T"],
            k=2.0,
            equations=["div(100*sym(grad(U))) = 0", "-div(k*grad(T)) = 0"],
        )
        pde.add_boundary_condition(["U_x = 0", "U_y = 0", "T = 0"], boundary="left")
        pde.add_boundary_condition(["normal_flux = [1, -2]", "normal_flux = -3"], boundary="right")
        problem.solve()
        return problem

    problem = solve()
    # T: -k dT/dx = -3 on the right, T = 0 on the left, so T = 1.5 x.
    assert problem.sample("T", [[1.0, 0.5]])[0] == pytest.approx(1.5, rel=1e-10)
    assert problem.sample("U_x", [[1.0, 0.5]])[0] > 0.0
    assert problem.sample("U_y", [[1.0, 0.5]])[0] < 0.0

    two = dm.Problem(unit_square(3), method="fem")
    pde = two.add_physics(
        "general_form_PDE",
        variables=["u", "v"],
        equations=["-laplacian(u) = v", "-laplacian(v) = 1"],
    )
    with pytest.raises(Exception, match="could be the flux of u, v"):
        pde.add_boundary_condition("normal_flux = 1", boundary="right")
