# SPDX-License-Identifier: LGPL-2.1-or-later
"""The physics level: every physics and coupling generates the same equations
as the objects written one by one, its vector conditions split into the
right components, and its errors say what to do."""

from __future__ import annotations

import math

import dualmesh
import numpy
import pytest
from dualmesh._core import InputError


def largest_difference(a, b, variables):
    return max(
        float(numpy.max(numpy.abs(numpy.asarray(a.values(v)) - numpy.asarray(b.values(v)))))
        / max(float(numpy.max(numpy.abs(a.values(v)))), 1e-300)
        for v in variables
    )


# ---- heat transfer ---------------------------------------------------------------
def bus_bar_mesh():
    return dualmesh.generate_rectangle_mesh(
        x_min=0.0, x_max=0.1, y_min=0.0, y_max=0.05, num_x_elements=10, num_y_elements=5
    )


def test_heat_conduction_equals_the_objects_and_reproduces_Reddy():
    objects = dualmesh.Problem(bus_bar_mesh(), method="dmcdm")
    objects.add_variable("temperature")
    objects.add_kernel("diffusion", variable="temperature", diffusivity=20.0)
    objects.add_kernel("body_force", variable="temperature", value=1.0e6)
    objects.add_boundary_condition("Dirichlet", "left", variable="temperature", value=40.0)
    objects.add_boundary_condition("Dirichlet", "right", variable="temperature", value=10.0)
    objects.add_boundary_condition(
        "Robin", "top", variable="temperature", transfer_coefficient=75.0
    )
    objects.solve()

    physics = dualmesh.Problem(bus_bar_mesh(), method="dmcdm")
    # The objects start at 0, so the physics does too, for the same iterates.
    heat = physics.add_physics(
        "heat_conduction", thermal_conductivity=20.0, heat_source=1.0e6, initial_condition=0.0
    )
    heat.add_boundary_condition("temperature = 40", boundary="left")
    heat.add_boundary_condition("temperature = 10", boundary="right")
    heat.add_boundary_condition("heat_flux = 75*(temperature - 0)", boundary="top")
    physics.solve()
    assert largest_difference(objects, physics, ["temperature"]) == 0.0
    # Reddy, Table 5.4.3
    assert physics.sample("temperature", [[0.05, 0.0]])[0] == pytest.approx(83.142, abs=5e-3)


# ---- solid mechanics -------------------------------------------------------------
def beam_mesh():
    return dualmesh.generate_box_mesh(
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
    objects = dualmesh.Problem(beam_mesh(), method="fem")
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

    physics = dualmesh.Problem(beam_mesh(), method="fem")
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
    assert -physics.boundary_flux("disp_y", "left") == pytest.approx(50.0, rel=1e-8)
    assert -physics.boundary_flux("disp_z", "left") == pytest.approx(-20.0, rel=1e-8)


# Timoshenko and Goodier, Theory of Elasticity, 3rd ed. (1970), Art. 150: a long
# hollow cylinder in plane strain with the temperature T_a ln(b/r)/ln(b/a).
inner, outer, t_inner, modulus, Poisson_ratio, expansion = 0.1, 0.2, 100.0, 200.0e9, 0.3, 1.2e-5


def hoop_stress(r):
    L = math.log(outer / inner)
    c = expansion * modulus * t_inner / (2.0 * (1.0 - Poisson_ratio) * L)
    return c * (
        1.0 - math.log(outer / r) - inner**2 / (outer**2 - inner**2) * (1.0 + outer**2 / r**2) * L
    )


def cylinder(symmetry: bool):
    mesh = dualmesh.generate_annulus_mesh(
        inner_radius=inner,
        outer_radius=outer,
        num_radial_elements=16,
        num_angular_elements=16,
        element_type="Quad9",
    )
    problem = dualmesh.Problem(mesh, method="fem")
    heat = problem.add_physics("heat_conduction", thermal_conductivity=50.0)
    heat.add_boundary_condition("temperature = T_inner", boundary="inner", T_inner=t_inner)
    heat.add_boundary_condition("temperature = 0", boundary="outer")
    solid = problem.add_physics(
        "solid_mechanics",
        formulation="plane_strain",
        youngs_modulus=modulus,
        poissons_ratio=Poisson_ratio,
    )
    if symmetry:
        solid.add_boundary_condition("symmetry", boundary="start")
        solid.add_boundary_condition("symmetry", boundary="end")
    else:
        solid.add_boundary_condition("disp_y = 0", boundary="start")
        solid.add_boundary_condition("disp_x = 0", boundary="end")
    problem.add_coupling(
        "thermal_expansion",
        heat_conduction=heat,
        solid_mechanics=solid,
        linear_expansion_coefficient=expansion,
        stress_free_temperature=0.0,
    )
    problem.solve()
    return problem


def test_thermal_expansion_reproduces_the_thermal_stress_of_a_hollow_cylinder():
    problem = cylinder(symmetry=False)
    mesh = problem.mesh
    stress = problem.property_at_centroids("stress")
    centroids = numpy.array([mesh.element_centroid(e) for e in range(mesh.num_elements)])
    r = numpy.hypot(centroids[:, 0], centroids[:, 1])
    theta = numpy.arctan2(centroids[:, 1], centroids[:, 0])
    hoop = (
        stress[:, 0] * numpy.sin(theta) ** 2
        + stress[:, 1] * numpy.cos(theta) ** 2
        - 2.0 * stress[:, 5] * numpy.sin(theta) * numpy.cos(theta)
    )
    exact = numpy.array([hoop_stress(x) for x in r])
    assert numpy.max(numpy.abs(hoop - exact)) < 5e-3 * numpy.max(numpy.abs(exact))


def test_symmetry_condition_refuses_an_inclined_side_set():
    mesh = dualmesh.generate_annulus_mesh(
        inner_radius=1.0,
        outer_radius=2.0,
        num_radial_elements=2,
        num_angular_elements=4,
        start_angle=10.0,
        end_angle=80.0,
    )
    problem = dualmesh.Problem(mesh)
    solid = problem.add_physics(
        "solid_mechanics",
        formulation="plane_stress",
        youngs_modulus=1.0,
        poissons_ratio=0.3,
    )
    with pytest.raises(ValueError, match="plane normal to a coordinate axis"):
        solid.add_boundary_condition("symmetry", boundary="start")


# ---- the registry and the errors -------------------------------------------------
# ---- coefficient form PDE ---------------------------------------------------------
def unit_square(n=8):
    return dualmesh.generate_rectangle_mesh(
        x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0, num_x_elements=n, num_y_elements=n
    )


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
    problem = dualmesh.Problem(unit_square(), method="fem")
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
    problem = dualmesh.Problem(unit_square(), method="fem")
    PDE = problem.add_physics(
        "general_form_PDE", variables="u", equations="-laplacian(u) = 1", h=2.0
    )
    PDE.add_boundary_condition("u = 0", boundary="left")
    with pytest.raises(InputError, match="'h' is defined and not used"):
        problem.solve()


@pytest.mark.parametrize(
    "condition, message",
    [
        ("-2*k*du/dn = 1", "is not the normal flux"),
        ("u + 1 = 0", "the left side is a field"),
        ("u = du/dx", "may not depend on derivatives"),
    ],
)
def test_boundary_conditions_as_text_are_checked(condition, message):
    problem = dualmesh.Problem(unit_square(), method="fem")
    PDE = problem.add_physics(
        "general_form_PDE", variables="u", k=2.0, equations="-div(k*grad(u)) = 1"
    )
    with pytest.raises(InputError, match=message):
        PDE.add_boundary_condition(condition, boundary="left")


def test_a_coupled_system_in_general_form_converges():
    from dualmesh import mms
    from test_MMS import check, square_family

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


def test_misuse_is_answered_with_what_to_do():
    problem = dualmesh.Problem(bus_bar_mesh())
    with pytest.raises(ValueError, match="Did you mean 'heat_conduction'"):
        problem.add_physics("heat_conductoin")
    with pytest.raises(ValueError, match="Did you mean 'thermal_conductivity'"):
        problem.add_physics("heat_conduction", thermal_conductivty=1.0)
    with pytest.raises(ValueError, match="is a coupling"):
        problem.add_physics(
            "thermal_expansion",
            heat_conduction="a",
            solid_mechanics="b",
            linear_expansion_coefficient=1.0,
            stress_free_temperature=0.0,
        )
    with pytest.raises(ValueError, match="has 0 heat_conduction physics"):
        problem.add_coupling(
            "thermal_expansion",
            linear_expansion_coefficient=1.0,
            stress_free_temperature=0.0,
        )
    heat = problem.add_physics("heat_conduction", thermal_conductivity=1.0)
    other = problem.add_physics("heat_conduction", thermal_conductivity=1.0)
    assert (heat.name, other.name) == ("heat_conduction", "heat_conduction_2")
    solid = problem.add_physics(
        "solid_mechanics",
        formulation="plane_stress",
        youngs_modulus=1.0,
        poissons_ratio=0.3,
    )
    with pytest.raises(ValueError, match="has 2 heat_conduction physics"):
        problem.add_coupling(
            "thermal_expansion", linear_expansion_coefficient=1.0, stress_free_temperature=0.0
        )
    with pytest.raises(ValueError, match="the object that add_physics returned, not 'heat'"):
        problem.add_coupling(
            "thermal_expansion",
            heat_conduction="heat",
            linear_expansion_coefficient=1.0,
            stress_free_temperature=0.0,
        )
    with pytest.raises(
        ValueError, match="is a heat_conduction physics, and needs a solid_mechanics"
    ):
        problem.add_coupling(
            "thermal_expansion",
            heat_conduction=heat,
            solid_mechanics=other,
            linear_expansion_coefficient=1.0,
            stress_free_temperature=0.0,
        )
    with pytest.raises(ValueError, match="one value per component"):
        solid.add_boundary_condition("traction = 1", boundary="right")
    with pytest.raises(ValueError, match="the vector has 3 values, and it needs 2"):
        solid.add_boundary_condition("traction = [1, 0, 0]", boundary="right")
    with pytest.raises(ValueError, match="write the condition as text"):
        solid.add_boundary_condition("traction", "right", traction=[1.0, 0.0])
    with pytest.raises(ValueError, match="between -1 and 0.5"):
        dualmesh.Problem(bus_bar_mesh()).add_physics(
            "solid_mechanics", formulation="plane_stress", youngs_modulus=1.0, poissons_ratio=0.5
        )
    with pytest.raises(ValueError, match="Taylor-Hood element needs the finite element method"):
        dualmesh.Problem(bus_bar_mesh(), method="DMCDM").add_physics(
            "fluid_dynamics", formulation="Taylor-Hood"
        )


def test_couplings_are_added_before_the_solve():
    problem = dualmesh.Problem(bus_bar_mesh(), method="fem")
    heat = problem.add_physics("heat_conduction", thermal_conductivity=1.0)
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
            heat_conduction=heat,
            solid_mechanics=solid,
            linear_expansion_coefficient=1.0,
            stress_free_temperature=0.0,
        )


def test_a_traction_that_depends_on_the_displacement_has_an_exact_Jacobian():
    # An elastic support of stiffness k_s on the right edge, sigma . n = -k_s U, under a
    # pull: the support carries load, and the Jacobian is exact.
    from test_Jacobian import compare, finite_difference_Jacobian

    problem = dualmesh.Problem(unit_square(4), method="fem")
    solid = problem.add_physics(
        "solid_mechanics", formulation="plane_stress", youngs_modulus=200.0, poissons_ratio=0.3
    )
    solid.add_boundary_condition("fixed", boundary="left")
    solid.add_boundary_condition(
        "traction = [1 - k_s*disp_x, -k_s*disp_y]", boundary="right", k_s=50.0
    )
    problem.solve()
    free = dualmesh.Problem(unit_square(4), method="fem")
    unsupported = free.add_physics(
        "solid_mechanics", formulation="plane_stress", youngs_modulus=200.0, poissons_ratio=0.3
    )
    unsupported.add_boundary_condition("fixed", boundary="left")
    unsupported.add_boundary_condition("traction = [1, 0]", boundary="right")
    free.solve()
    supported = problem.sample("disp_x", [[1.0, 0.5]])[0]
    assert 0.0 < supported < free.sample("disp_x", [[1.0, 0.5]])[0]
    _, Jacobian = problem.linear_system()
    assert compare(Jacobian.toarray(), finite_difference_Jacobian(problem)) < 1e-7
