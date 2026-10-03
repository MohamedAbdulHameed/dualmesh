# SPDX-License-Identifier: LGPL-2.1-or-later
"""The physics level: every physics and coupling generates the same equations
as the objects written one by one, its vector conditions split into the
right components, and its errors say what to do."""

from __future__ import annotations

import math

import dualmesh as dm
import numpy as np
import pytest


def largest_difference(a, b, variables):
    return max(float(np.max(np.abs(np.asarray(a.values(v)) - np.asarray(b.values(v))))) / max(float(np.max(np.abs(a.values(v)))), 1e-300) for v in variables)


# ---- heat transfer ---------------------------------------------------------------
def bus_bar_mesh():
    return dm.generate_rectangle_mesh(x_min=0.0, x_max=0.1, y_min=0.0, y_max=0.05, num_x_elements=10, num_y_elements=5)


def test_heat_transfer_equals_the_objects_and_reproduces_reddy():
    objects = dm.Problem(bus_bar_mesh(), method="dmcdm")
    objects.add_variable("temperature")
    objects.add_kernel("heat_conduction", variable="temperature", thermal_conductivity=20.0)
    objects.add_kernel("heat_source", variable="temperature", heat_source=1.0e6)
    objects.add_boundary_condition("Dirichlet_boundary_condition", "left", variable="temperature", value=40.0)
    objects.add_boundary_condition("Dirichlet_boundary_condition", "right", variable="temperature", value=10.0)
    objects.add_boundary_condition("convective_heat_flux_boundary_condition", "top", variable="temperature", heat_transfer_coefficient=75.0)
    objects.solve()

    physics = dm.Problem(bus_bar_mesh(), method="dmcdm")
    heat = physics.add_physics("heat_transfer", "heat", thermal_conductivity=20.0, heat_source=1.0e6)
    heat.add_boundary_condition("Dirichlet_boundary_condition", "left", value=40.0)
    heat.add_boundary_condition("Dirichlet_boundary_condition", "right", value=10.0)
    heat.add_boundary_condition("convective_heat_flux_boundary_condition", "top", heat_transfer_coefficient=75.0)
    physics.solve()
    assert largest_difference(objects, physics, ["temperature"]) == 0.0
    # Reddy, Table 5.4.3
    assert physics.sample("temperature", [[0.05, 0.0]])[0] == pytest.approx(83.142, abs=5e-3)


def test_heat_transfer_reads_the_conductivity_of_the_property_objects():
    mesh = bus_bar_mesh()
    constant = dm.Problem(mesh)
    heat = constant.add_physics("heat_transfer", "heat", thermal_conductivity=20.0, heat_source=1.0e6)
    heat.add_boundary_condition("Dirichlet_boundary_condition", "walls", boundary=["left", "right"], value=0.0)
    constant.solve()
    material = dm.Problem(mesh)
    material.add_property("constant_property", "copper", property_names=["thermal_conductivity"], property_values=[20.0])
    heat = material.add_physics("heat_transfer", "heat", heat_source=1.0e6)
    heat.add_boundary_condition("Dirichlet_boundary_condition", "walls", boundary=["left", "right"], value=0.0)
    material.solve()
    assert largest_difference(constant, material, ["temperature"]) < 1e-12


def test_a_transient_heat_transfer_equals_the_objects():
    mesh = dm.generate_line_mesh(start=0.0, end=1.0, num_elements=20)
    objects = dm.Problem(mesh)
    objects.add_variable("temperature", initial_condition=1.0)
    objects.add_kernel("heat_conduction", variable="temperature", thermal_conductivity=0.5)
    objects.add_kernel("heat_conduction_time_derivative", variable="temperature", density=2.0, specific_heat=3.0)
    objects.add_boundary_condition("Dirichlet_boundary_condition", "ends", variable="temperature", boundary=["left", "right"], value=0.0)
    objects.solve_transient(end_time=0.5, time_step=0.05)
    physics = dm.Problem(mesh)
    heat = physics.add_physics("heat_transfer", "heat", thermal_conductivity=0.5, density=2.0, specific_heat=3.0, initial_condition=1.0)
    heat.add_boundary_condition("Dirichlet_boundary_condition", "ends", boundary=["left", "right"], value=0.0)
    physics.solve_transient(end_time=0.5, time_step=0.05)
    assert largest_difference(objects, physics, ["temperature"]) == 0.0
    # The slowest mode decays as exp(-pi^2 k t / (rho c_p)).
    exact = 4.0 / math.pi * math.exp(-(math.pi**2) * 0.5 * 0.5 / 6.0)
    assert physics.sample("temperature", [[0.5]])[0] == pytest.approx(exact, rel=0.05)


# ---- solid mechanics -------------------------------------------------------------
def beam_mesh():
    return dm.generate_box_mesh(x_min=0.0, x_max=1.0, y_min=0.0, y_max=0.2, z_min=0.0, z_max=0.1, num_x_elements=10, num_y_elements=2, num_z_elements=2)


def test_solid_mechanics_equals_the_objects_and_splits_the_force():
    objects = dm.Problem(beam_mesh(), method="fem")
    for name in ("u", "v", "w"):
        objects.add_variable(name)
    objects.add_property("linear_elastic_stress", "steel", displacements=["u", "v", "w"], formulation="three_dimensional", youngs_modulus=200.0e9, poissons_ratio=0.3)
    for component, name in enumerate(("u", "v", "w")):
        objects.add_kernel("stress_divergence", f"equilibrium_{name}", variable=name, component=component)
    objects.add_boundary_condition("fixed_constraint", "left", displacements=["u", "v", "w"])
    objects.add_boundary_condition("traction_boundary_condition", "right_v", variable="v", boundary="right", total_force=-50.0)
    objects.add_boundary_condition("traction_boundary_condition", "right_w", variable="w", boundary="right", total_force=20.0)
    objects.solve()

    physics = dm.Problem(beam_mesh(), method="fem")
    solid = physics.add_physics("solid_mechanics", "solid", displacements=["u", "v", "w"], youngs_modulus=200.0e9, poissons_ratio=0.3)
    solid.add_boundary_condition("fixed_constraint", "left")
    solid.add_boundary_condition("traction_boundary_condition", "right", total_force=[0.0, -50.0, 20.0])
    physics.solve()
    assert largest_difference(objects, physics, ["u", "v", "w"]) < 1e-10
    # The reaction balances the applied force.
    assert physics.total_reaction("v", "left") == pytest.approx(50.0, rel=1e-8)
    assert physics.total_reaction("w", "left") == pytest.approx(-20.0, rel=1e-8)


def test_solid_mechanics_default_names_and_formulations():
    problem = dm.Problem(beam_mesh(), method="fem")
    solid = problem.add_physics("solid_mechanics", "solid", youngs_modulus=1.0, poissons_ratio=0.3)
    assert solid.formulation == "three_dimensional"
    assert solid.variables() == ["displacement_x", "displacement_y", "displacement_z"]
    axisymmetric = dm.Problem(dm.generate_rectangle_mesh(1.0, 2.0, 0.0, 1.0, 2, 2), coordinates="axisymmetric")
    solid = axisymmetric.add_physics("solid_mechanics", "solid", youngs_modulus=1.0, poissons_ratio=0.3)
    assert solid.formulation == "axisymmetric"
    assert solid.variables() == ["displacement_r", "displacement_z"]
    with pytest.raises(ValueError, match="plane_stress.*plane_strain"):
        dm.Problem(dm.generate_rectangle_mesh(0, 1, 0, 1, 2, 2)).add_physics("solid_mechanics", youngs_modulus=1.0, poissons_ratio=0.3)


# Timoshenko and Goodier, Theory of Elasticity, 3rd ed. (1970), Art. 150: a long
# hollow cylinder in plane strain with the temperature T_a ln(b/r)/ln(b/a).
INNER, OUTER, T_INNER, MODULUS, POISSON, EXPANSION = 0.1, 0.2, 100.0, 200.0e9, 0.3, 1.2e-5


def hoop_stress(r):
    L = math.log(OUTER / INNER)
    c = EXPANSION * MODULUS * T_INNER / (2.0 * (1.0 - POISSON) * L)
    return c * (1.0 - math.log(OUTER / r) - INNER**2 / (OUTER**2 - INNER**2) * (1.0 + OUTER**2 / r**2) * L)


def cylinder(symmetry: bool):
    mesh = dm.generate_annulus_mesh(inner_radius=INNER, outer_radius=OUTER, num_radial_elements=16, num_angular_elements=16, element_type="Quad9")
    problem = dm.Problem(mesh, method="fem")
    heat = problem.add_physics("heat_transfer", "heat", thermal_conductivity=50.0)
    heat.add_boundary_condition("Dirichlet_boundary_condition", "inner", value=T_INNER)
    heat.add_boundary_condition("Dirichlet_boundary_condition", "outer", value=0.0)
    solid = problem.add_physics("solid_mechanics", "solid", displacements=["u", "v"], formulation="plane_strain", youngs_modulus=MODULUS, poissons_ratio=POISSON)
    if symmetry:
        solid.add_boundary_condition("symmetry_boundary_condition", "start")
        solid.add_boundary_condition("symmetry_boundary_condition", "end")
    else:
        solid.add_boundary_condition("Dirichlet_boundary_condition", "start", value=[None, 0.0])
        solid.add_boundary_condition("Dirichlet_boundary_condition", "end", value=[0.0, None])
    problem.add_coupling("thermal_expansion", "expansion", heat_transfer="heat", solid_mechanics="solid", thermal_expansion_coefficient=EXPANSION, stress_free_temperature=0.0)
    problem.solve()
    return problem


def test_thermal_expansion_reproduces_the_thermal_stress_of_a_hollow_cylinder():
    problem = cylinder(symmetry=False)
    mesh = problem.mesh
    stress = problem.property_at_centroids("stress")
    centroids = np.array([mesh.element_centroid(e) for e in range(mesh.num_elements)])
    r = np.hypot(centroids[:, 0], centroids[:, 1])
    theta = np.arctan2(centroids[:, 1], centroids[:, 0])
    hoop = stress[:, 0] * np.sin(theta) ** 2 + stress[:, 1] * np.cos(theta) ** 2 - 2.0 * stress[:, 5] * np.sin(theta) * np.cos(theta)
    exact = np.array([hoop_stress(x) for x in r])
    assert np.max(np.abs(hoop - exact)) < 5e-3 * np.max(np.abs(exact))


def test_symmetry_condition_equals_the_normal_displacement_held_at_zero():
    a, b = cylinder(symmetry=False), cylinder(symmetry=True)
    assert largest_difference(a, b, ["temperature", "u", "v"]) < 1e-12


def test_symmetry_condition_refuses_an_inclined_side_set():
    mesh = dm.generate_annulus_mesh(inner_radius=1.0, outer_radius=2.0, num_radial_elements=2, num_angular_elements=4, start_angle=10.0, end_angle=80.0)
    problem = dm.Problem(mesh)
    solid = problem.add_physics("solid_mechanics", "solid", formulation="plane_stress", youngs_modulus=1.0, poissons_ratio=0.3)
    with pytest.raises(ValueError, match="plane normal to a coordinate axis"):
        solid.add_boundary_condition("symmetry_boundary_condition", "start")


def test_solid_mechanics_reads_the_elastic_properties_of_the_property_objects():
    def build(from_material):
        problem = dm.Problem(dm.generate_rectangle_mesh(0.0, 1.0, 0.0, 0.2, 10, 2), method="fem")
        if from_material:
            problem.add_property("constant_property", "steel", property_names=["youngs_modulus", "poissons_ratio"], property_values=[200.0e9, 0.3])
            solid = problem.add_physics("solid_mechanics", "solid", formulation="plane_strain")
        else:
            solid = problem.add_physics("solid_mechanics", "solid", formulation="plane_strain", youngs_modulus=200.0e9, poissons_ratio=0.3)
        solid.add_boundary_condition("fixed_constraint", "left")
        solid.add_boundary_condition("traction_boundary_condition", "right", traction=[0.0, -1.0e6])
        problem.solve()
        return problem

    assert largest_difference(build(False), build(True), ["displacement_x", "displacement_y"]) < 1e-10


# ---- flow and couplings ----------------------------------------------------------
def test_the_lid_driven_cavity_physics_equals_the_objects():
    mesh = dm.generate_rectangle_mesh(0.0, 1.0, 0.0, 1.0, 8, 8)
    objects = dm.Problem(mesh)
    velocities = ["u", "v"]
    for variable in velocities:
        objects.add_variable(variable)
    for component, variable in enumerate(velocities):
        objects.add_kernel("viscous_stress", f"viscous_{variable}", variable=variable, component=component, velocities=velocities, dynamic_viscosity=1.0)
        objects.add_kernel("penalty_incompressibility", f"penalty_{variable}", variable=variable, component=component, velocities=velocities, penalty_parameter=1.0e8)
        objects.add_kernel("convective_inertia", f"inertia_{variable}", variable=variable, component=component, velocities=velocities, density=10.0)
    objects.add_property("penalty_pressure", "pressure", velocities=velocities, penalty_parameter=1.0e8)
    objects.add_boundary_condition("Dirichlet_boundary_condition", "walls", variables=velocities, boundary=["left", "right", "bottom"], value=0.0)
    objects.add_boundary_condition("Dirichlet_boundary_condition", "lid_u", variable="u", boundary="top", value=1.0)
    objects.add_boundary_condition("Dirichlet_boundary_condition", "lid_v", variable="v", boundary="top", value=0.0)
    objects.solve()
    physics = dm.Problem(mesh)
    flow = physics.add_physics("incompressible_flow", "flow", velocities=velocities, dynamic_viscosity=1.0, density=10.0)
    flow.add_boundary_condition("Dirichlet_boundary_condition", "walls", boundary=["left", "right", "bottom"], value=[0.0, 0.0])
    flow.add_boundary_condition("Dirichlet_boundary_condition", "top", value=[1.0, 0.0])
    physics.solve()
    assert largest_difference(objects, physics, velocities) < 1e-10


# ---- the registry and the errors -------------------------------------------------
# ---- coefficient form PDE ---------------------------------------------------------
def unit_square(n=8):
    return dm.generate_rectangle_mesh(x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0, num_x_elements=n, num_y_elements=n)


@pytest.mark.parametrize("method", ["fem", "dmcdm", "zfvm"])
def test_coefficient_form_pde_equals_the_objects(method):
    objects = dm.Problem(unit_square(), method=method)
    objects.add_variable("c")
    objects.add_kernel("diffusion", variable="c", diffusivity="1 + 0.5*x")
    objects.add_kernel("advection", "beta", variable="c", velocity=[1.0, 0.5, 0.0])
    objects.add_kernel("advection", "alpha", variable="c", velocity=[0.2, -0.1, 0.0], advection_form="conservative")
    objects.add_kernel("reaction", variable="c", coefficient=2.0)
    objects.add_kernel("body_force", variable="c", value="sin(pi*y)", scale_with_load=False)
    objects.add_boundary_condition("Dirichlet_boundary_condition", "left", variable="c", value=1.0)
    objects.add_boundary_condition("Neumann_boundary_condition", "right", variable="c", flux=0.5)
    objects.solve()

    physics = dm.Problem(unit_square(), method=method)
    pde = physics.add_physics("coefficient_form_PDE", "transport", variable="c", diffusion_coefficient="1 + 0.5*x", convection_coefficient=[1.0, 0.5], conservative_flux_convection_coefficient=[-0.2, 0.1], absorption_coefficient=2.0, source="sin(pi*y)")
    pde.add_boundary_condition("Dirichlet_boundary_condition", "left", value=1.0)
    pde.add_boundary_condition("Neumann_boundary_condition", "right", flux=0.5)
    physics.solve()
    assert largest_difference(objects, physics, ["c"]) < 1e-13


def test_coefficient_form_pde_with_coefficients_of_the_variable_equals_the_objects():
    # k(u) u-dependence through a parsed material, against the polynomial of
    # the diffusion kernel, and a(u) u = u^3 / 2 against the reaction exponent.
    objects = dm.Problem(unit_square(), method="fem")
    objects.add_variable("u")
    objects.add_kernel("diffusion", variable="u", solution_polynomial=[1.0, 0.5, 0.2])
    objects.add_kernel("reaction", variable="u", coefficient=0.5, exponent=3.0)
    objects.add_kernel("body_force", variable="u", value=4.0, scale_with_load=False)
    objects.add_boundary_condition("Dirichlet_boundary_condition", "left", variable="u", value=1.0)
    objects.solve()

    physics = dm.Problem(unit_square(), method="fem")
    pde = physics.add_physics("coefficient_form_PDE", "pde", diffusion_coefficient="1 + 0.5*u + 0.2*u**2", absorption_coefficient="0.5*u**2", source=4.0)
    pde.add_boundary_condition("Dirichlet_boundary_condition", "left", value=1.0)
    physics.solve()
    assert largest_difference(objects, physics, ["u"]) < 1e-11


def test_coefficient_form_pde_transient_and_tensor_and_material():
    mesh = unit_square(6)
    objects = dm.Problem(mesh, method="fem")
    objects.add_variable("u")
    objects.add_kernel("anisotropic_diffusion", variable="u", diffusivity_tensor=[2.0, 0.5])
    objects.add_kernel("time_derivative", variable="u", coefficient=3.0)
    objects.add_boundary_condition("Dirichlet_boundary_condition", "left", variable="u", value=1.0)
    objects.solve_transient(end_time=0.2, time_step=0.05)
    physics = dm.Problem(mesh, method="fem")
    pde = physics.add_physics("coefficient_form_PDE", "pde", diffusion_coefficient=[2.0, 0.5], time_derivative_coefficient=3.0)
    pde.add_boundary_condition("Dirichlet_boundary_condition", "left", value=1.0)
    physics.solve_transient(end_time=0.2, time_step=0.05)
    assert largest_difference(objects, physics, ["u"]) < 1e-13
    # Without a diffusion coefficient, the property of that name.
    material = dm.Problem(mesh, method="fem")
    material.add_property("constant_property", "medium", property_names=["diffusion_coefficient"], property_values=[2.0])
    pde = material.add_physics("coefficient_form_PDE", "pde", source=1.0)
    pde.add_boundary_condition("Dirichlet_boundary_condition", "left", value=0.0)
    material.solve()
    constant = dm.Problem(mesh, method="fem")
    pde = constant.add_physics("coefficient_form_PDE", "pde", diffusion_coefficient=2.0, source=1.0)
    pde.add_boundary_condition("Dirichlet_boundary_condition", "left", value=0.0)
    constant.solve()
    assert largest_difference(material, constant, ["u"]) < 1e-13


def test_coefficient_form_pde_refuses_ambiguous_coefficients():
    problem = dm.Problem(unit_square(2))
    with pytest.raises(ValueError, match="parsed_property"):
        problem.add_physics("coefficient_form_PDE", "pde", diffusion_coefficient="1 + u*x")
    with pytest.raises(ValueError, match="absorption_coefficient"):
        problem.add_physics("coefficient_form_PDE", "pde", source="u**2")
    with pytest.raises(ValueError, match="Did you mean 'diffusion_coefficient'"):
        problem.add_physics("coefficient_form_PDE", "pde", difusion_coefficient=1.0)


def test_describe_and_list_include_the_physics_and_couplings():
    assert {"heat_transfer", "solid_mechanics", "incompressible_flow", "beam", "plate", "circular_plate"} <= set(dm.list_objects(category="physics"))
    assert set(dm.list_objects(category="coupling")) == {"thermal_expansion", "nonisothermal_flow"}
    text = dm.describe("solid_mechanics")
    assert text.startswith("solid_mechanics (physics, module solid_mechanics)")
    assert "youngs_modulus (real, Pa)" in text
    assert "symmetry_boundary_condition" in dm.list_objects(category="nodal_boundary_condition")


def test_describe_and_list_include_the_applications_and_their_input_groups():
    from dualmesh import cli

    assert dm.list_objects(category="application") == ["FuelRod", "TrisoParticleModel"]
    groups = dm.list_objects(category="input_group")
    for name in ("RodGeometry", "RodNumerics", "UO2Fuel", "ZircaloyCladding", "TrisoParticle", "ParticleNumerics"):
        assert name in groups
    assert "RodContext" not in groups
    text = dm.describe("FuelRod")
    assert "application, module fuel" in text and "numerics (RodNumerics" in text
    text = dm.describe("RodNumerics")
    assert "max_burnup_step (real, MWd/kgHM)" in text
    with pytest.raises(ValueError, match="Did you mean 'FuelRod'"):
        dm.describe("FuelRd")
    assert cli.main(["describe", "TrisoParticleModel"]) == 0
    assert cli.main(["describe", "FuelRd"]) == 1


def test_misuse_is_answered_with_what_to_do():
    problem = dm.Problem(bus_bar_mesh())
    with pytest.raises(ValueError, match="Did you mean 'heat_transfer'"):
        problem.add_physics("heat_transfr")
    with pytest.raises(ValueError, match="Did you mean 'thermal_conductivity'"):
        problem.add_physics("heat_transfer", thermal_conductivty=1.0)
    with pytest.raises(ValueError, match="is a coupling"):
        problem.add_physics("thermal_expansion", heat_transfer="a", solid_mechanics="b", thermal_expansion_coefficient=1.0, stress_free_temperature=0.0)
    with pytest.raises(ValueError, match="missing required parameter 'heat_transfer'"):
        problem.add_coupling("thermal_expansion", solid_mechanics="b", thermal_expansion_coefficient=1.0, stress_free_temperature=0.0)
    problem.add_physics("heat_transfer", "heat", thermal_conductivity=1.0)
    with pytest.raises(ValueError, match="already has a physics or coupling named 'heat'"):
        problem.add_physics("heat_transfer", "heat")
    with pytest.raises(ValueError, match="no physics 'solid'"):
        problem.add_coupling("thermal_expansion", "expansion", heat_transfer="heat", solid_mechanics="solid", thermal_expansion_coefficient=1.0, stress_free_temperature=0.0)
    solid = problem.add_physics("solid_mechanics", "solid", formulation="plane_stress", youngs_modulus=1.0, poissons_ratio=0.3)
    with pytest.raises(ValueError, match="one value per component"):
        solid.add_boundary_condition("traction_boundary_condition", "right", traction=1.0)
    with pytest.raises(ValueError, match="give 2 values"):
        solid.add_boundary_condition("traction_boundary_condition", "right", traction=[1.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="between -1 and 0.5"):
        dm.Problem(bus_bar_mesh()).add_physics("solid_mechanics", formulation="plane_stress", youngs_modulus=1.0, poissons_ratio=0.5)
    with pytest.raises(ValueError, match="Taylor-Hood element needs the finite element method"):
        dm.Problem(bus_bar_mesh()).add_physics("incompressible_flow", formulation="Taylor_Hood")


def test_couplings_are_added_before_the_solve():
    problem = dm.Problem(bus_bar_mesh(), method="fem")
    heat = problem.add_physics("heat_transfer", "heat", thermal_conductivity=1.0)
    heat.add_boundary_condition("Dirichlet_boundary_condition", "left", value=1.0)
    solid = problem.add_physics("solid_mechanics", "solid", formulation="plane_strain", youngs_modulus=1.0, poissons_ratio=0.3)
    solid.add_boundary_condition("fixed_constraint", "left")
    problem.solve()
    with pytest.raises(ValueError, match="before the problem is solved"):
        problem.add_coupling("thermal_expansion", "expansion", heat_transfer="heat", solid_mechanics="solid", thermal_expansion_coefficient=1.0, stress_free_temperature=0.0)


def test_a_distributed_problem_takes_the_same_physics():
    def build(distributed):
        problem = dm.Problem(bus_bar_mesh(), method="fem", distributed=distributed)
        heat = problem.add_physics("heat_transfer", "heat", thermal_conductivity=20.0, heat_source=1.0e6)
        heat.add_boundary_condition("Dirichlet_boundary_condition", "left", value=40.0)
        heat.add_boundary_condition("convective_heat_flux_boundary_condition", "top", heat_transfer_coefficient=75.0)
        problem.solve(linear_tolerance=1e-13)
        return problem

    serial, distributed = build(False), build(True)
    assert distributed.is_distributed and not serial.is_distributed
    assert np.asarray(distributed.gathered_values("temperature")) == pytest.approx(serial.values("temperature"), abs=1e-8)
    with pytest.raises(ValueError, match="not available in a distributed problem"):
        distributed.add_postprocessor("nodal_extreme_value", "hottest", variable="temperature", value_type="max")
