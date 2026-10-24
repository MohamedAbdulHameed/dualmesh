# SPDX-License-Identifier: LGPL-2.1-or-later
"""The physics level: every physics and coupling generates the same equations
as the objects written one by one, its vector conditions split into the
right components, and its errors say what to do."""

from __future__ import annotations

import math

import dualmesh as dm
import numpy as np
import pytest
from dualmesh._core import InputError


def largest_difference(a, b, variables):
    return max(
        float(np.max(np.abs(np.asarray(a.values(v)) - np.asarray(b.values(v)))))
        / max(float(np.max(np.abs(a.values(v)))), 1e-300)
        for v in variables
    )


# ---- heat transfer ---------------------------------------------------------------
def bus_bar_mesh():
    return dm.generate_rectangle_mesh(
        x_min=0.0, x_max=0.1, y_min=0.0, y_max=0.05, num_x_elements=10, num_y_elements=5
    )


def test_heat_transfer_equals_the_objects_and_reproduces_reddy():
    objects = dm.Problem(bus_bar_mesh(), method="dmcdm")
    objects.add_variable("temperature")
    objects.add_kernel("heat_conduction", variable="temperature", thermal_conductivity=20.0)
    objects.add_kernel("body_force", variable="temperature", value=1.0e6)
    objects.add_boundary_condition("Dirichlet", "left", variable="temperature", value=40.0)
    objects.add_boundary_condition("Dirichlet", "right", variable="temperature", value=10.0)
    objects.add_boundary_condition("Robin", "top", variable="temperature", transfer_coeff=75.0)
    objects.solve()

    physics = dm.Problem(bus_bar_mesh(), method="dmcdm")
    heat = physics.add_physics("heat_transfer", thermal_conductivity=20.0, heat_source=1.0e6)
    heat.add_boundary_condition("temperature = 40", boundary="left")
    heat.add_boundary_condition("temperature = 10", boundary="right")
    heat.add_boundary_condition("heat_flux = 75*(temperature - 0)", boundary="top")
    physics.solve()
    assert largest_difference(objects, physics, ["temperature"]) == 0.0
    # Reddy, Table 5.4.3
    assert physics.sample("temperature", [[0.05, 0.0]])[0] == pytest.approx(83.142, abs=5e-3)


def test_heat_transfer_reads_the_conductivity_of_the_property_objects():
    mesh = bus_bar_mesh()
    constant = dm.Problem(mesh)
    heat = constant.add_physics("heat_transfer", thermal_conductivity=20.0, heat_source=1.0e6)
    heat.add_boundary_condition("temperature = 0", boundary=["left", "right"])
    constant.solve()
    material = dm.Problem(mesh)
    material.add_property(
        "constant",
        properties={"thermal_conductivity": 20.0},
    )
    heat = material.add_physics("heat_transfer", heat_source=1.0e6)
    heat.add_boundary_condition("temperature = 0", boundary=["left", "right"])
    material.solve()
    assert largest_difference(constant, material, ["temperature"]) < 1e-12


def test_a_transient_heat_transfer_equals_the_objects():
    mesh = dm.generate_line_mesh(start=0.0, end=1.0, num_elements=20)
    objects = dm.Problem(mesh)
    objects.add_variable("temperature", initial_condition=1.0)
    objects.add_kernel("heat_conduction", variable="temperature", thermal_conductivity=0.5)
    objects.add_kernel(
        "heat_conduction_time_derivative", variable="temperature", density=2.0, specific_heat=3.0
    )
    objects.add_boundary_condition(
        "Dirichlet",
        ["left", "right"],
        variable="temperature",
        value=0.0,
    )
    objects.solve_transient(end_time=0.5, time_step=0.05)
    physics = dm.Problem(mesh)
    heat = physics.add_physics(
        "heat_transfer",
        thermal_conductivity=0.5,
        density=2.0,
        specific_heat=3.0,
        initial_condition=1.0,
    )
    heat.add_boundary_condition("temperature = 0", boundary=["left", "right"])
    physics.solve_transient(end_time=0.5, time_step=0.05)
    assert largest_difference(objects, physics, ["temperature"]) == 0.0
    # The slowest mode decays as exp(-pi^2 k t / (rho c_p)).
    exact = 4.0 / math.pi * math.exp(-(math.pi**2) * 0.5 * 0.5 / 6.0)
    assert physics.sample("temperature", [[0.5]])[0] == pytest.approx(exact, rel=0.05)


# ---- solid mechanics -------------------------------------------------------------
def beam_mesh():
    return dm.generate_box_mesh(
        x_min=0.0,
        x_max=1.0,
        y_min=0.0,
        y_max=0.2,
        z_min=0.0,
        z_max=0.1,
        num_x_elements=10,
        num_y_elements=2,
        num_z_elements=2,
    )


def test_solid_mechanics_equals_the_objects_and_splits_the_force():
    objects = dm.Problem(beam_mesh(), method="fem")
    for name in ("disp_x", "disp_y", "disp_z"):
        objects.add_variable(name)
    objects.add_property(
        "small_strain_stress",
        displacements=["disp_x", "disp_y", "disp_z"],
        formulation="three_dimensional",
        youngs_modulus=200.0e9,
        poissons_ratio=0.3,
    )
    for name in ("disp_x", "disp_y", "disp_z"):
        objects.add_kernel(
            "stress_divergence", variable=name, displacements=["disp_x", "disp_y", "disp_z"]
        )
    objects.add_boundary_condition(
        "Dirichlet", "left", variable=["disp_x", "disp_y", "disp_z"], value=0.0
    )
    objects.add_boundary_condition("traction", "right", variable="disp_y", total_force=-50.0)
    objects.add_boundary_condition("traction", "right", variable="disp_z", total_force=20.0)
    objects.solve()

    physics = dm.Problem(beam_mesh(), method="fem")
    solid = physics.add_physics(
        "solid_mechanics",
        youngs_modulus=200.0e9,
        poissons_ratio=0.3,
    )
    solid.add_boundary_condition("fixed", boundary="left")
    solid.add_boundary_condition("total_force = [0, -50, 20]", boundary="right")
    physics.solve()
    assert largest_difference(objects, physics, ["disp_x", "disp_y", "disp_z"]) < 1e-10
    # The reaction balances the applied force.
    assert physics.total_reaction("disp_y", "left") == pytest.approx(50.0, rel=1e-8)
    assert physics.total_reaction("disp_z", "left") == pytest.approx(-20.0, rel=1e-8)


def test_solid_mechanics_default_names_and_formulations():
    problem = dm.Problem(beam_mesh(), method="fem")
    solid = problem.add_physics("solid_mechanics", youngs_modulus=1.0, poissons_ratio=0.3)
    assert solid.formulation == "three_dimensional"
    assert solid.variable_names() == ["disp_x", "disp_y", "disp_z"]
    axisymmetric = dm.Problem(
        dm.generate_rectangle_mesh(1.0, 2.0, 0.0, 1.0, 2, 2), coordinates="axisymmetric"
    )
    solid = axisymmetric.add_physics("solid_mechanics", youngs_modulus=1.0, poissons_ratio=0.3)
    assert solid.formulation == "axisymmetric"
    assert solid.variable_names() == ["disp_r", "disp_z"]
    with pytest.raises(ValueError, match="plane_stress.*plane_strain"):
        dm.Problem(dm.generate_rectangle_mesh(0, 1, 0, 1, 2, 2)).add_physics(
            "solid_mechanics", youngs_modulus=1.0, poissons_ratio=0.3
        )


# Timoshenko and Goodier, Theory of Elasticity, 3rd ed. (1970), Art. 150: a long
# hollow cylinder in plane strain with the temperature T_a ln(b/r)/ln(b/a).
INNER, OUTER, T_INNER, MODULUS, POISSON, EXPANSION = 0.1, 0.2, 100.0, 200.0e9, 0.3, 1.2e-5


def hoop_stress(r):
    L = math.log(OUTER / INNER)
    c = EXPANSION * MODULUS * T_INNER / (2.0 * (1.0 - POISSON) * L)
    return c * (
        1.0 - math.log(OUTER / r) - INNER**2 / (OUTER**2 - INNER**2) * (1.0 + OUTER**2 / r**2) * L
    )


def cylinder(symmetry: bool):
    mesh = dm.generate_annulus_mesh(
        inner_radius=INNER,
        outer_radius=OUTER,
        num_radial_elements=16,
        num_angular_elements=16,
        element_type="Quad9",
    )
    problem = dm.Problem(mesh, method="fem")
    heat = problem.add_physics("heat_transfer", thermal_conductivity=50.0)
    heat.add_boundary_condition("temperature = T_inner", boundary="inner", T_inner=T_INNER)
    heat.add_boundary_condition("temperature = 0", boundary="outer")
    solid = problem.add_physics(
        "solid_mechanics",
        formulation="plane_strain",
        youngs_modulus=MODULUS,
        poissons_ratio=POISSON,
    )
    if symmetry:
        solid.add_boundary_condition("symmetry", boundary="start")
        solid.add_boundary_condition("symmetry", boundary="end")
    else:
        solid.add_boundary_condition("disp_y = 0", boundary="start")
        solid.add_boundary_condition("disp_x = 0", boundary="end")
    problem.add_coupling(
        "thermal_expansion",
        heat_transfer=heat,
        solid_mechanics=solid,
        linear_expansion_coeff=EXPANSION,
        stress_free_temperature=0.0,
    )
    problem.solve()
    return problem


def test_thermal_expansion_reproduces_the_thermal_stress_of_a_hollow_cylinder():
    problem = cylinder(symmetry=False)
    mesh = problem.mesh
    stress = problem.property_at_centroids("stress")
    centroids = np.array([mesh.element_centroid(e) for e in range(mesh.num_elements)])
    r = np.hypot(centroids[:, 0], centroids[:, 1])
    theta = np.arctan2(centroids[:, 1], centroids[:, 0])
    hoop = (
        stress[:, 0] * np.sin(theta) ** 2
        + stress[:, 1] * np.cos(theta) ** 2
        - 2.0 * stress[:, 5] * np.sin(theta) * np.cos(theta)
    )
    exact = np.array([hoop_stress(x) for x in r])
    assert np.max(np.abs(hoop - exact)) < 5e-3 * np.max(np.abs(exact))


def test_symmetry_condition_equals_the_normal_displacement_held_at_zero():
    a, b = cylinder(symmetry=False), cylinder(symmetry=True)
    assert largest_difference(a, b, ["temperature", "disp_x", "disp_y"]) < 1e-12


def test_symmetry_condition_refuses_an_inclined_side_set():
    mesh = dm.generate_annulus_mesh(
        inner_radius=1.0,
        outer_radius=2.0,
        num_radial_elements=2,
        num_angular_elements=4,
        start_angle=10.0,
        end_angle=80.0,
    )
    problem = dm.Problem(mesh)
    solid = problem.add_physics(
        "solid_mechanics",
        formulation="plane_stress",
        youngs_modulus=1.0,
        poissons_ratio=0.3,
    )
    with pytest.raises(ValueError, match="plane normal to a coordinate axis"):
        solid.add_boundary_condition("symmetry", boundary="start")


def test_solid_mechanics_reads_the_elastic_properties_of_the_property_objects():
    def build(from_material):
        problem = dm.Problem(dm.generate_rectangle_mesh(0.0, 1.0, 0.0, 0.2, 10, 2), method="fem")
        if from_material:
            problem.add_property(
                "constant",
                properties={"youngs_modulus": 200.0e9, "poissons_ratio": 0.3},
            )
            solid = problem.add_physics("solid_mechanics", formulation="plane_strain")
        else:
            solid = problem.add_physics(
                "solid_mechanics",
                formulation="plane_strain",
                youngs_modulus=200.0e9,
                poissons_ratio=0.3,
            )
        solid.add_boundary_condition("fixed", boundary="left")
        solid.add_boundary_condition("traction = [0, -1000000]", boundary="right")
        problem.solve()
        return problem

    assert largest_difference(build(False), build(True), ["disp_x", "disp_y"]) < 1e-10


# ---- flow and couplings ----------------------------------------------------------
def test_the_lid_driven_cavity_physics_equals_the_objects():
    mesh = dm.generate_rectangle_mesh(0.0, 1.0, 0.0, 1.0, 8, 8)
    objects = dm.Problem(mesh)
    velocities = ["velocity_x", "velocity_y"]
    for variable in velocities:
        objects.add_variable(variable)
    for variable in velocities:
        objects.add_kernel(
            "viscous_stress",
            variable=variable,
            velocities=velocities,
            dynamic_viscosity=1.0,
        )
        objects.add_kernel(
            "penalty_incompressibility",
            variable=variable,
            velocities=velocities,
            penalty_parameter=1.0e8,
        )
        objects.add_kernel(
            "convective_inertia",
            variable=variable,
            velocities=velocities,
            density=10.0,
        )
    objects.add_property("penalty_pressure", velocities=velocities)
    objects.add_boundary_condition(
        "Dirichlet",
        ["left", "right", "bottom"],
        variable=velocities,
        value=0.0,
    )
    objects.add_boundary_condition("Dirichlet", "top", variable="velocity_x", value=1.0)
    objects.add_boundary_condition("Dirichlet", "top", variable="velocity_y", value=0.0)
    objects.solve()
    physics = dm.Problem(mesh)
    flow = physics.add_physics("fluid_dynamics", dynamic_viscosity=1.0, density=10.0)
    flow.add_boundary_condition(
        ["velocity_x = 0", "velocity_y = 0"], boundary=["left", "right", "bottom"]
    )
    flow.add_boundary_condition(["velocity_x = 1", "velocity_y = 0"], boundary="top")
    physics.solve()
    assert largest_difference(objects, physics, velocities) < 1e-10


# ---- the registry and the errors -------------------------------------------------
# ---- coefficient form PDE ---------------------------------------------------------
def unit_square(n=8):
    return dm.generate_rectangle_mesh(
        x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0, num_x_elements=n, num_y_elements=n
    )


@pytest.mark.parametrize("method", ["fem", "dmcdm", "zfvm"])
def test_the_general_form_equals_the_dedicated_kernels(method):
    objects = dm.Problem(unit_square(), method=method)
    objects.add_variable("c")
    objects.add_kernel("diffusion", variable="c", diffusivity="1 + 0.5*x")
    objects.add_kernel("advection", variable="c", velocity=[1.0, 0.5, 0.0])
    objects.add_kernel(
        "advection", variable="c", velocity=[0.2, -0.1, 0.0], advection_form="conservative"
    )
    objects.add_kernel("reaction", variable="c", coeff=2.0)
    objects.add_kernel("body_force", variable="c", value="sin(pi*y)", scale_with_load=False)
    objects.add_boundary_condition("Dirichlet", "left", variable="c", value=1.0)
    objects.add_boundary_condition("Neumann", "right", variable="c", flux=0.5)
    objects.solve()

    physics = dm.Problem(unit_square(), method=method)
    pde = physics.add_physics(
        "general_form_PDE",
        variables=["c"],
        c="1 + 0.5*x",
        beta=[1.0, 0.5],
        alpha=[-0.2, 0.1],
        a=2.0,
        f="sin(pi*y)",
    )
    pde.add_boundary_condition("c = 1", boundary="left")
    # The Neumann flux 0.5 enters the body: n . Gamma = -0.5.
    pde.add_boundary_condition("normal_flux = -0.5", boundary="right")
    physics.solve()
    assert largest_difference(objects, physics, ["c"]) < 1e-13


def test_the_general_form_with_coefficients_of_the_variable_equals_the_objects():
    # k(u) u-dependence through a parsed material, against the polynomial of
    # the diffusion kernel, and a(u) u = u^3 / 2 against the reaction exponent.
    objects = dm.Problem(unit_square(), method="fem")
    objects.add_variable("u")
    objects.add_kernel("diffusion", variable="u", solution_polynomial=[1.0, 0.5, 0.2])
    objects.add_kernel("reaction", variable="u", coeff=0.5, exponent=3.0)
    objects.add_kernel("body_force", variable="u", value=4.0, scale_with_load=False)
    objects.add_boundary_condition("Dirichlet", "left", variable="u", value=1.0)
    objects.solve()

    physics = dm.Problem(unit_square(), method="fem")
    pde = physics.add_physics(
        "general_form_PDE",
        variables="u",
        c="1 + 0.5*u + 0.2*u**2",
        a="0.5*u**2",
        f=4.0,
    )
    pde.add_boundary_condition("u = 1", boundary="left")
    physics.solve()
    assert largest_difference(objects, physics, ["u"]) < 1e-11


def test_the_general_form_transient_and_with_a_tensor():
    mesh = unit_square(6)
    objects = dm.Problem(mesh, method="fem")
    objects.add_variable("u")
    objects.add_kernel("anisotropic_diffusion", variable="u", diffusivity_tensor=[2.0, 0.5])
    objects.add_kernel("time_derivative", variable="u", coeff=3.0)
    objects.add_boundary_condition("Dirichlet", "left", variable="u", value=1.0)
    objects.solve_transient(end_time=0.2, time_step=0.05)
    physics = dm.Problem(mesh, method="fem")
    pde = physics.add_physics(
        "general_form_PDE",
        variables="u",
        c=[2.0, 0.5],
        d=3.0,
    )
    pde.add_boundary_condition("u = 1", boundary="left")
    physics.solve_transient(end_time=0.2, time_step=0.05)
    assert largest_difference(objects, physics, ["u"]) < 1e-13


def test_definitions_of_the_general_form_equal_their_text():
    inline = dm.Problem(unit_square(), method="fem")
    pde = inline.add_physics(
        "general_form_PDE",
        variables="u",
        equations="div(-(1 + grad_x(u)^2 + grad_y(u)^2)*grad(u)) = 2*1.5",
    )
    pde.add_boundary_condition("u = 0", boundary="left")
    inline.solve()

    named = dm.Problem(unit_square(), method="fem")
    pde = named.add_physics(
        "general_form_PDE",
        variables="u",
        k=1.5,
        grad_u_squared="dot(grad(u), grad(u))",
        conductivity="1 + grad_u_squared",
        equations="div(-conductivity*grad(u)) = 2*k",
    )
    pde.add_boundary_condition("u = 0", boundary="left")
    named.solve()
    assert largest_difference(inline, named, ["u"]) < 1e-13


@pytest.mark.parametrize(
    "definitions, message",
    [
        (dict(conductivity="1 + s", s="u^2"), "defined after this expression"),
        (dict(s="1 + s"), "defined after this expression"),
        (dict(u="1"), "cannot be defined"),
        (dict(x=1.0), "cannot be defined"),
        (dict(equation="1"), "Did you mean 'equations'"),
        (dict(k=[1.0, 2.0]), "is not a number or an expression"),
    ],
)
def test_definitions_of_the_general_form_are_checked(definitions, message):
    problem = dm.Problem(unit_square(), method="fem")
    with pytest.raises(InputError, match=message):
        problem.add_physics(
            "general_form_PDE",
            variables="u",
            equations="-laplacian(u) = conductivity"
            if "conductivity" in definitions
            else "-laplacian(u) = 1",
            **definitions,
        )


def test_an_unused_definition_is_refused():
    problem = dm.Problem(unit_square(), method="fem")
    pde = problem.add_physics(
        "general_form_PDE", variables="u", equations="-laplacian(u) = 1", h=2.0
    )
    pde.add_boundary_condition("u = 0", boundary="left")
    with pytest.raises(InputError, match="'h' is defined and not used"):
        problem.solve()


def test_boundary_conditions_of_the_general_form_as_text():
    # -k u'' = 1 on (0, 1) x (0, 1) with u = 0 on the left, -k du/dn = h (u - 1) on the
    # right: the conditions written as text equal the objects.
    objects = dm.Problem(unit_square(), method="fem")
    objects.add_variable("u")
    objects.add_kernel("diffusion", variable="u", diffusivity=2.0)
    objects.add_kernel("body_force", variable="u", value=1.0, scale_with_load=False)
    objects.add_boundary_condition("Dirichlet", "left", variable="u", value="0.5*y")
    objects.add_boundary_condition(
        "Robin", "right", variable="u", transfer_coeff=3.0, ambient_value=1.0
    )
    objects.add_boundary_condition("Neumann", "top", variable="u", flux=0.25)
    objects.solve()

    text = dm.Problem(unit_square(), method="fem")
    pde = text.add_physics(
        "general_form_PDE", variables="u", k=2.0, h=3.0, equations="-div(k*grad(u)) = 1"
    )
    pde.add_boundary_condition("u = 0.5*y", boundary="left")
    pde.add_boundary_condition("-k*du/dn = h*(u - 1)", boundary="right")
    pde.add_boundary_condition("normal_flux = -0.25", boundary="top")
    text.solve()
    assert largest_difference(objects, text, ["u"]) < 1e-12


@pytest.mark.parametrize(
    "condition, message",
    [
        ("-2*k*du/dn = 1", "is not the normal flux"),
        ("u + 1 = 0", "the left side is a field"),
        ("u = du/dx", "may not depend on derivatives"),
    ],
)
def test_boundary_conditions_as_text_are_checked(condition, message):
    problem = dm.Problem(unit_square(), method="fem")
    pde = problem.add_physics(
        "general_form_PDE", variables="u", k=2.0, equations="-div(k*grad(u)) = 1"
    )
    with pytest.raises(InputError, match=message):
        pde.add_boundary_condition(condition, boundary="left")


def test_a_coupled_system_in_general_form_converges():
    from dualmesh import mms
    from test_mms import check, square_family

    study = mms.ManufacturedSolution(
        {"u": "1 + sin(x)*cos(y)", "v": "x**2 + x*y + 0.5"}, dimension=2
    )
    study.add_physics(
        "general_form_PDE",
        variables=["u", "v"],
        c={"u": "1 + 0.5*v", "v": {"v": "1 + u^2", "u": 0.5}},
        a={"u": {"v": -1.0}},
        beta={"v": {"u": [0.3, "v"]}},
        f={"u": "sin(x)"},
    )
    for method in ("fem", "dmcdm"):
        result = study.convergence_study(
            square_family("Quad4"), [4, 8, 16, 32], method=method, report="none"
        )
        check(result, "u", method, "Quad4")
        check(result, "v", method, "Quad4")


def test_describe_and_list_include_the_physics_and_couplings():
    assert {
        "heat_transfer",
        "solid_mechanics",
        "fluid_dynamics",
        "general_form_PDE",
    } == set(dm.list_objects(category="physics"))
    assert set(dm.list_objects(category="coupling")) == {"thermal_expansion", "nonisothermal_flow"}
    text = dm.describe("solid_mechanics")
    assert text.startswith("solid_mechanics (physics, module solid_mechanics)")
    assert "youngs_modulus (real or string, Pa)" in text
    assert "symmetry" in dm.list_objects(category="nodal_boundary_condition")


def test_describe_answers_a_misspelled_type_and_the_command_line_reports_it():
    from dualmesh import cli

    assert "heat_transfer" in dm.describe("heat_transfer")
    assert "gas_pressure" in dm.describe("gas_gap_heat_transfer")
    with pytest.raises(ValueError, match="Did you mean 'heat_conduction'"):
        dm.describe("heat_conductoin")
    with pytest.raises(ValueError, match="Unknown category 'application'"):
        dm.list_objects(category="application")
    assert cli.main(["describe", "solid_mechanics"]) == 0
    assert cli.main(["describe", "heat_conductoin"]) == 1


def test_misuse_is_answered_with_what_to_do():
    problem = dm.Problem(bus_bar_mesh())
    with pytest.raises(ValueError, match="Did you mean 'heat_transfer'"):
        problem.add_physics("heat_transfr")
    with pytest.raises(ValueError, match="Did you mean 'thermal_conductivity'"):
        problem.add_physics("heat_transfer", thermal_conductivty=1.0)
    with pytest.raises(ValueError, match="is a coupling"):
        problem.add_physics(
            "thermal_expansion",
            heat_transfer="a",
            solid_mechanics="b",
            linear_expansion_coeff=1.0,
            stress_free_temperature=0.0,
        )
    with pytest.raises(ValueError, match="has 0 heat_transfer physics"):
        problem.add_coupling(
            "thermal_expansion",
            linear_expansion_coeff=1.0,
            stress_free_temperature=0.0,
        )
    heat = problem.add_physics("heat_transfer", thermal_conductivity=1.0)
    other = problem.add_physics("heat_transfer", thermal_conductivity=1.0)
    assert (heat.name, other.name) == ("heat_transfer", "heat_transfer_2")
    solid = problem.add_physics(
        "solid_mechanics",
        formulation="plane_stress",
        youngs_modulus=1.0,
        poissons_ratio=0.3,
    )
    with pytest.raises(ValueError, match="has 2 heat_transfer physics"):
        problem.add_coupling(
            "thermal_expansion", linear_expansion_coeff=1.0, stress_free_temperature=0.0
        )
    with pytest.raises(ValueError, match="the object that add_physics returned, not 'heat'"):
        problem.add_coupling(
            "thermal_expansion",
            heat_transfer="heat",
            linear_expansion_coeff=1.0,
            stress_free_temperature=0.0,
        )
    with pytest.raises(ValueError, match="is a heat_transfer physics, and needs a solid_mechanics"):
        problem.add_coupling(
            "thermal_expansion",
            heat_transfer=heat,
            solid_mechanics=other,
            linear_expansion_coeff=1.0,
            stress_free_temperature=0.0,
        )
    with pytest.raises(ValueError, match="one value per component"):
        solid.add_boundary_condition("traction = 1", boundary="right")
    with pytest.raises(ValueError, match="the vector has 3 values, and it needs 2"):
        solid.add_boundary_condition("traction = [1, 0, 0]", boundary="right")
    with pytest.raises(ValueError, match="write the condition as text"):
        solid.add_boundary_condition("traction", "right", traction=[1.0, 0.0])
    with pytest.raises(ValueError, match="between -1 and 0.5"):
        dm.Problem(bus_bar_mesh()).add_physics(
            "solid_mechanics", formulation="plane_stress", youngs_modulus=1.0, poissons_ratio=0.5
        )
    with pytest.raises(ValueError, match="Taylor-Hood element needs the finite element method"):
        dm.Problem(bus_bar_mesh()).add_physics("fluid_dynamics", formulation="Taylor_Hood")


def test_couplings_are_added_before_the_solve():
    problem = dm.Problem(bus_bar_mesh(), method="fem")
    heat = problem.add_physics("heat_transfer", thermal_conductivity=1.0)
    heat.add_boundary_condition("temperature = 1", boundary="left")
    solid = problem.add_physics(
        "solid_mechanics",
        formulation="plane_strain",
        youngs_modulus=1.0,
        poissons_ratio=0.3,
    )
    solid.add_boundary_condition("fixed", boundary="left")
    problem.solve()
    with pytest.raises(ValueError, match="before the problem is solved"):
        problem.add_coupling(
            "thermal_expansion",
            heat_transfer=heat,
            solid_mechanics=solid,
            linear_expansion_coeff=1.0,
            stress_free_temperature=0.0,
        )


def test_a_distributed_problem_takes_the_same_physics():
    def build(distributed):
        problem = dm.Problem(bus_bar_mesh(), method="fem", distributed=distributed)
        heat = problem.add_physics("heat_transfer", thermal_conductivity=20.0, heat_source=1.0e6)
        heat.add_boundary_condition("temperature = 40", boundary="left")
        heat.add_boundary_condition("heat_flux = 75*(temperature - 0)", boundary="top")
        problem.solve(linear_tolerance=1e-13)
        return problem

    serial, distributed = build(False), build(True)
    assert distributed.is_distributed and not serial.is_distributed
    assert np.asarray(distributed.gathered_values("temperature")) == pytest.approx(
        serial.values("temperature"), abs=1e-8
    )
    assert distributed.nodal_extreme("temperature", "max") == pytest.approx(
        serial.nodal_extreme("temperature", "max"), abs=1e-8
    )


def test_physics_take_boundary_conditions_as_text():
    def heat(text):
        problem = dm.Problem(unit_square(), method="fem")
        physics = problem.add_physics("heat_transfer", thermal_conductivity=20.0, heat_source=1.0e3)
        if text:
            physics.add_boundary_condition("temperature = 300 + 10*y", boundary="left")
            physics.add_boundary_condition("heat_flux = 75*(temperature - 293)", boundary="right")
            physics.add_boundary_condition("heat_flux = -500", boundary="top")
        else:
            # The objects of the reference.
            T = "temperature"
            problem.add_boundary_condition("Dirichlet", "left", variable=T, value="300 + 10*y")
            problem.add_boundary_condition(
                "Robin", "right", variable=T, transfer_coeff=75.0, ambient_value=293.0
            )
            problem.add_boundary_condition("Neumann", "top", variable=T, flux=500.0)
        problem.solve()
        return problem

    assert largest_difference(heat(True), heat(False), ["temperature"]) < 1e-10

    def solid(text):
        problem = dm.Problem(unit_square(), method="fem")
        physics = problem.add_physics(
            "solid_mechanics", formulation="plane_stress", youngs_modulus=200.0, poissons_ratio=0.3
        )
        if text:
            physics.add_boundary_condition(["disp_x = 0", "disp_y = 0"], boundary="left")
            physics.add_boundary_condition("traction = [0, -1 - x]", boundary="right")
        else:
            physics.add_boundary_condition("fixed", boundary="left")
            problem.add_boundary_condition(
                "traction", "right", variable="disp_y", traction="-1 - x"
            )
        problem.solve()
        return problem

    assert largest_difference(solid(True), solid(False), ["disp_x", "disp_y"]) < 1e-12


def test_a_condition_as_text_names_a_variable_or_the_flux():
    problem = dm.Problem(unit_square(), method="fem")
    physics = problem.add_physics("heat_transfer", thermal_conductivity=20.0)
    with pytest.raises(InputError, match="temperature or heat_flux"):
        physics.add_boundary_condition("T = 300", boundary="left")
    with pytest.raises(InputError, match="give boundary, one or more of"):
        physics.add_boundary_condition("temperature = 300")


def test_a_condition_as_text_takes_definitions():
    def solve(definitions):
        problem = dm.Problem(unit_square(), method="fem")
        heat = problem.add_physics("heat_transfer", thermal_conductivity=20.0, heat_source=1.0e3)
        if definitions:
            heat.add_boundary_condition(
                ["temperature = T0 + rise", "heat_flux = 0"],
                boundary="left",
                T0=300.0,
                rise="10*y",
            )
            heat.add_boundary_condition(
                "heat_flux = h*(temperature - T_air)", boundary="right", h=75.0, T_air=293.0
            )
        else:
            heat.add_boundary_condition("temperature = 300 + 10*y", boundary="left")
            heat.add_boundary_condition("heat_flux = 75*(temperature - 293)", boundary="right")
        problem.solve()
        return problem

    assert largest_difference(solve(True), solve(False), ["temperature"]) < 1e-12
    problem = dm.Problem(unit_square(), method="fem")
    heat = problem.add_physics("heat_transfer", thermal_conductivity=20.0)
    with pytest.raises(InputError, match="'h' is defined and not used"):
        heat.add_boundary_condition("temperature = 300", boundary="left", h=2.0)
    with pytest.raises(InputError, match="Did you mean 'scale_with_load'"):
        heat.add_boundary_condition("temperature = 300", boundary="left", scale_with_lod=True)


def test_a_traction_that_depends_on_the_displacement_has_an_exact_jacobian():
    # An elastic support of stiffness k_s on the right edge, sigma . n = -k_s U, under a
    # pull: the support carries load, and the Jacobian is exact.
    from test_jacobian import compare, finite_difference_jacobian

    problem = dm.Problem(unit_square(4), method="fem")
    solid = problem.add_physics(
        "solid_mechanics", formulation="plane_stress", youngs_modulus=200.0, poissons_ratio=0.3
    )
    solid.add_boundary_condition("fixed", boundary="left")
    solid.add_boundary_condition(
        "traction = [1 - k_s*disp_x, -k_s*disp_y]", boundary="right", k_s=50.0
    )
    problem.solve()
    free = dm.Problem(unit_square(4), method="fem")
    unsupported = free.add_physics(
        "solid_mechanics", formulation="plane_stress", youngs_modulus=200.0, poissons_ratio=0.3
    )
    unsupported.add_boundary_condition("fixed", boundary="left")
    unsupported.add_boundary_condition("traction = [1, 0]", boundary="right")
    free.solve()
    supported = problem.sample("disp_x", [[1.0, 0.5]])[0]
    assert 0.0 < supported < free.sample("disp_x", [[1.0, 0.5]])[0]
    _, jacobian = problem.linear_system()
    assert compare(jacobian.toarray(), finite_difference_jacobian(problem)) < 1e-7
