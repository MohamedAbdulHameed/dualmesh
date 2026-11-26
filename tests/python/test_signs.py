# SPDX-License-Identifier: LGPL-2.1-or-later
"""The direction of every flux, load, source and result, against hand results.

Each test pins one convention with a problem whose answer is known by hand, in every method:

* a boundary condition on a flux gives it along the outward normal, out of the body
  (``heat_flux``, ``normal_flux``), and a traction is the force per area on the body;
* a source, a body force and a point load push their variable up;
* ``heat_flow``, ``boundary_flux`` and ``flow_rate`` are what leaves the body, ``reaction_force``
  is the force of the supports on the solid and ``force`` the force of the fluid on the wall;
* a boundary without a condition carries exactly nothing, and a corner node belongs to the
  boundary whose condition holds it.
"""

from __future__ import annotations

import dualmesh
import numpy
import pytest

methods = ["FEM", "DMCDM"]
exact = 1e-10


def bar(method, n=8):
    """The unit square, which with conditions on left and right only is a bar along x."""
    return dualmesh.Problem(dualmesh.generate_rectangle_mesh(0, 1, 0, 1, n, 2), method=method)


# ---- heat transfer ---------------------------------------------------------------------------
@pytest.mark.parametrize("method", methods)
def test_a_heat_flux_condition_is_the_heat_leaving(method):
    """T(0) = 0 and 2 W/m^2 leaving at x = 1: the heat flows to the right, so T falls,
    T = -2x/k, and the same heat enters at the left."""
    problem = bar(method)
    heat = problem.add_physics("heat_conduction", thermal_conductivity=4.0)
    heat.add_boundary_condition("temperature = 0", boundary="left")
    heat.add_boundary_condition("heat_flux = 2", boundary="right")
    problem.solve(report="none")
    assert problem.minimum("temperature") == pytest.approx(-0.5, rel=1e-10)
    assert heat.heat_flow("right") == pytest.approx(2.0, rel=exact)
    assert heat.heat_flow("left") == pytest.approx(-2.0, rel=exact)


@pytest.mark.parametrize("method", methods)
def test_a_heat_source_heats_and_its_heat_leaves_through_the_held_walls(method):
    """q = 8 W/m^3 between walls at 0: T rises to q/(8k) in the middle, and each wall takes half."""
    problem = bar(method)
    heat = problem.add_physics("heat_conduction", thermal_conductivity=1.0, heat_source=8.0)
    heat.add_boundary_condition("temperature = 0", boundary=["left", "right"])
    problem.solve(report="none")
    assert problem.maximum("temperature") == pytest.approx(1.0, rel=1e-10)
    assert heat.heat_flow("left") == pytest.approx(4.0, rel=exact)
    assert heat.heat_flow("right") == pytest.approx(4.0, rel=exact)
    assert heat.heat_flow("top") == 0.0
    assert heat.heat_flow("bottom") == 0.0


@pytest.mark.parametrize("method", methods)
def test_cooling_by_convection_is_heat_leaving(method):
    """A body at 300 K on the left cooled to 290 K air on the right: heat leaves on the right."""
    problem = bar(method)
    heat = problem.add_physics("heat_conduction", thermal_conductivity=1.0)
    heat.add_boundary_condition("temperature = 300", boundary="left")
    heat.add_boundary_condition("heat_flux = 2*(temperature - 290)", boundary="right")
    problem.solve(report="none")
    # In series, 1/k + 1/h = 1.5: the heat flow is 10 / 1.5.
    assert heat.heat_flow("right") == pytest.approx(10.0 / 1.5, rel=exact)
    assert heat.heat_flow("left") == pytest.approx(-10.0 / 1.5, rel=exact)


@pytest.mark.parametrize("method", methods)
def test_a_corner_belongs_to_the_boundary_that_holds_it(method):
    """A source in a square held at 0 on the left only: the corners of the left wall belong to it,
    so the insulated top and bottom carry nothing and the left takes the whole source."""
    mesh = dualmesh.generate_rectangle_mesh(0, 1, 0, 1, 5, 4)
    problem = dualmesh.Problem(mesh, method=method)
    heat = problem.add_physics("heat_conduction", thermal_conductivity=1.0, heat_source=3.0)
    heat.add_boundary_condition("temperature = 0", boundary="left")
    problem.solve(report="none")
    assert heat.heat_flow("top") == 0.0
    assert heat.heat_flow("bottom") == 0.0
    assert heat.heat_flow("left") == pytest.approx(3.0, rel=exact)


@pytest.mark.parametrize("method", methods)
def test_a_point_heat_source_heats(method):
    problem = bar(method)
    heat = problem.add_physics("heat_conduction", thermal_conductivity=1.0)
    heat.add_boundary_condition("temperature = 0", boundary=["left", "right"])
    problem.add_point_source(variable="temperature", value=2.0, points=[[0.5, 0.5]])
    problem.solve(report="none")
    assert problem.minimum("temperature") >= -1e-12
    assert heat.heat_flow("left") + heat.heat_flow("right") == pytest.approx(2.0, rel=exact)


@pytest.mark.parametrize("method", methods)
def test_the_heat_of_an_axisymmetric_body_counts_its_whole_circumference(method):
    """A cylinder of radius 1 and height 1 with q = 1 W/m^3, held at 0 on its outer surface: the
    heat leaving is the heat generated, q pi R^2 H = pi W."""
    mesh = dualmesh.generate_rectangle_mesh(0, 1, 0, 1, 6, 3)
    problem = dualmesh.Problem(mesh, method=method, coordinates="axisymmetric")
    heat = problem.add_physics("heat_conduction", thermal_conductivity=1.0, heat_source=1.0)
    heat.add_boundary_condition("temperature = 0", boundary="right")
    problem.solve(report="none")
    assert heat.heat_flow("right") == pytest.approx(numpy.pi, rel=exact)
    assert heat.heat_flow("left") == 0.0


# ---- the general form ------------------------------------------------------------------------
@pytest.mark.parametrize("method", methods)
def test_a_normal_flux_condition_is_the_flux_leaving(method):
    """div(-grad u) = 0, u(0) = 1 and n . Gamma = 3 at x = 1: u falls as 1 - 3x."""
    problem = bar(method)
    PDE = problem.add_physics("general_form_PDE", variables="u", c=1.0)
    PDE.add_boundary_condition("u = 1", boundary="left")
    PDE.add_boundary_condition("normal_flux = 3", boundary="right")
    problem.solve(report="none")
    assert problem.minimum("u") == pytest.approx(-2.0, rel=1e-10)
    assert PDE.boundary_flux("right") == pytest.approx(3.0, rel=exact)
    assert PDE.boundary_flux("left") == pytest.approx(-3.0, rel=exact)


# ---- solids ----------------------------------------------------------------------------------
def plane_strain(method, **parameters):
    problem = dualmesh.Problem(dualmesh.generate_rectangle_mesh(0, 2, 0, 1, 6, 3), method=method)
    solid = problem.add_physics(
        "solid_mechanics",
        formulation="plane_strain",
        youngs_modulus=100.0,
        poissons_ratio=0.25,
        **parameters,
    )
    return problem, solid


@pytest.mark.parametrize("method", methods)
def test_a_traction_pulls_the_body_and_the_support_holds_it_back(method):
    """A traction [1, -3] on the right of a plate fixed on the left: the right end moves along it,
    and the support pushes the other way."""
    problem, solid = plane_strain(method)
    solid.add_boundary_condition("fixed", boundary="left")
    solid.add_boundary_condition("traction = [1, -3]", boundary="right")
    problem.solve(report="none")
    tip = problem.node_at([2.0, 1.0])
    assert problem.values("disp_x")[tip] > 0.0
    assert problem.values("disp_y")[tip] < 0.0
    support = solid.reaction_force("left")
    assert support["x"] == pytest.approx(-1.0, rel=exact)
    assert support["y"] == pytest.approx(3.0, rel=exact)
    load = solid.reaction_force("right")
    assert load["x"] == pytest.approx(1.0, rel=exact)
    assert load["y"] == pytest.approx(-3.0, rel=exact)
    assert solid.reaction_force("top") == {"x": 0.0, "y": 0.0}


@pytest.mark.parametrize("method", methods)
def test_a_pressure_pushes_into_the_body(method):
    problem, solid = plane_strain(method)
    solid.add_boundary_condition("fixed", boundary="left")
    solid.add_boundary_condition("pressure = 2", boundary="right")
    problem.solve(report="none")
    assert problem.values("disp_x")[problem.node_at([2.0, 1.0])] < 0.0
    assert solid.reaction_force("left")["x"] == pytest.approx(2.0, rel=exact)


@pytest.mark.parametrize("method", methods)
def test_the_weight_rests_on_the_support(method):
    """A body force [0, -5] on a 2 x 1 plate standing on its bottom: it sags, and the support
    pushes up with the weight."""
    problem, solid = plane_strain(method, body_force=[0.0, -5.0])
    solid.add_boundary_condition("fixed", boundary="bottom")
    problem.solve(report="none")
    assert problem.values("disp_y")[problem.node_at([1.0, 1.0])] < 0.0
    assert solid.reaction_force("bottom")["y"] == pytest.approx(10.0, rel=exact)


@pytest.mark.parametrize("method", methods)
def test_a_point_force_pushes_along_its_value(method):
    problem, solid = plane_strain(method)
    solid.add_boundary_condition("fixed", boundary="left")
    problem.add_point_source(variable="disp_x", value=-4.0, points=[[2.0, 1.0]])
    problem.solve(report="none")
    assert problem.values("disp_x")[problem.node_at([2.0, 1.0])] < 0.0
    assert solid.reaction_force("left")["x"] == pytest.approx(4.0, rel=exact)


def test_a_transverse_load_lifts_a_beam_and_the_clamp_holds_it_down():
    mesh = dualmesh.generate_line_mesh(start=0.0, end=2.0, num_elements=8)
    problem = dualmesh.Problem(mesh)
    beam = problem.add_physics(
        "solid_mechanics",
        formulation="beam",
        youngs_modulus=100.0,
        poissons_ratio=0.3,
        cross_section={"width": 0.2, "height": 0.1},
        transverse_load=3.0,
    )
    beam.add_boundary_condition("fixed", boundary="left")
    problem.solve(report="none")
    assert problem.values("disp_z")[-1] > 0.0
    assert beam.reaction_force("left")["z"] == pytest.approx(-6.0, rel=exact)


# ---- flows -----------------------------------------------------------------------------------
@pytest.mark.parametrize("method", methods)
def test_a_flow_drags_its_walls_downstream_and_leaves_through_the_outlet(method):
    """Poiseuille flow along x: the fluid pulls the walls along x and pushes them apart (its
    pressure is positive upstream of the outlet); the flow rate is negative at the inlet and
    positive at the outlet, and the walls pass none."""
    mesh = dualmesh.generate_rectangle_mesh(0, 4, 0, 1, 16, 6)
    problem = dualmesh.Problem(mesh, method=method)
    flow = problem.add_physics("fluid_dynamics", density=1.0, dynamic_viscosity=1.0)
    flow.add_boundary_condition(["velocity_x = 6*y*(1 - y)", "velocity_y = 0"], boundary="left")
    flow.add_boundary_condition(["velocity_x = 0", "velocity_y = 0"], boundary=["top", "bottom"])
    flow.add_boundary_condition("pressure = 0", boundary="right")
    problem.solve(report="none")
    top, bottom = flow.force("top"), flow.force("bottom")
    assert top["x"] > 0.0 and bottom["x"] > 0.0
    assert top["y"] > 0.0 and bottom["y"] < 0.0
    # The inflow is the integral of the linear interpolant of the profile, 1 - h^2 = 35/36.
    assert flow.flow_rate("left") == pytest.approx(-35.0 / 36.0, rel=exact)
    assert flow.flow_rate("right") == pytest.approx(35.0 / 36.0, rel=exact)
    assert flow.flow_rate("top") == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("method", methods)
def test_a_heated_fluid_rises(method):
    """Natural convection with the hot wall on the left and gravity along -y: the fluid rises
    along the hot wall and falls along the cold one."""
    mesh = dualmesh.generate_rectangle_mesh(0, 1, 0, 1, 12, 12)
    problem = dualmesh.Problem(mesh, method=method)
    flow = problem.add_physics("fluid_dynamics", density=1.0, dynamic_viscosity=1.0)
    heat = problem.add_physics(
        "heat_conduction", thermal_conductivity=1.0, density=1.0, specific_heat=1.0
    )
    problem.add_coupling(
        "nonisothermal_flow",
        heat_conduction=heat,
        fluid_dynamics=flow,
        gravity=[0.0, -1.0],
        volumetric_expansion_coefficient=1000.0,
    )
    flow.add_boundary_condition(
        ["velocity_x = 0", "velocity_y = 0"], boundary=["left", "right", "bottom", "top"]
    )
    flow.add_boundary_condition("pressure = 0", point=[0.5, 0.0])
    heat.add_boundary_condition("temperature = 1", boundary="left")
    heat.add_boundary_condition("temperature = 0", boundary="right")
    problem.solve(report="none")
    v = problem.sample("velocity_y", [[0.1, 0.5], [0.9, 0.5]])
    assert v[0] > 0.0 and v[1] < 0.0
    assert heat.heat_flow("left") < 0.0 and heat.heat_flow("right") > 0.0


# ---- couplings and time ----------------------------------------------------------------------
@pytest.mark.parametrize("method", methods)
def test_heating_expands_a_free_body(method):
    """A plate fixed only at its left edge, heated 10 K above its stress-free temperature with
    alpha = 1e-3: its right edge moves out by alpha dT L = 0.02 (plane strain adds (1 + nu))."""
    problem, solid = plane_strain(method)
    heat = problem.add_physics("heat_conduction", thermal_conductivity=1.0)
    problem.add_coupling(
        "thermal_expansion",
        heat_conduction=heat,
        solid_mechanics=solid,
        linear_expansion_coefficient=1.0e-3,
        stress_free_temperature=300.0,
    )
    heat.add_boundary_condition("temperature = 310", boundary=["left", "right"])
    solid.add_boundary_condition("disp_x = 0", boundary="left")
    solid.add_boundary_condition("disp_y = 0", boundary="bottom")
    problem.solve(report="none")
    assert problem.values("disp_x")[problem.node_at([2.0, 1.0])] == pytest.approx(
        1.25 * 0.02, rel=1e-8
    )


@pytest.mark.parametrize("method", methods)
def test_a_source_heats_a_body_in_time(method):
    """An insulated body with q = 6 W/m^3 and rho c_p = 3: dT/dt = 2 K/s everywhere."""
    problem = bar(method)
    problem.add_physics(
        "heat_conduction",
        thermal_conductivity=1.0,
        heat_source=6.0,
        density=1.0,
        specific_heat=3.0,
        initial_condition=300.0,
    )
    problem.solve_transient(end_time=0.5, time_step=0.25, report="none")
    assert problem.values("temperature") == pytest.approx(301.0, rel=1e-12)


@pytest.mark.parametrize("method", methods)
def test_a_flow_carries_heat_downstream(method):
    """Plug flow along x through a channel with a hot inlet and a cold outlet: the flow carries
    the inlet temperature downstream, where conduction alone would give a linear fall."""
    mesh = dualmesh.generate_rectangle_mesh(0, 4, 0, 1, 16, 2)
    problem = dualmesh.Problem(mesh, method=method)
    flow = problem.add_physics("fluid_dynamics", density=1.0, dynamic_viscosity=1.0)
    heat = problem.add_physics(
        "heat_conduction", thermal_conductivity=0.1, density=1.0, specific_heat=1.0
    )
    problem.add_coupling("nonisothermal_flow", heat_conduction=heat, fluid_dynamics=flow)
    flow.add_boundary_condition(["velocity_x = 1", "velocity_y = 0"], boundary="left")
    flow.add_boundary_condition("velocity_y = 0", boundary=["top", "bottom"])
    flow.add_boundary_condition("pressure = 0", boundary="right")
    heat.add_boundary_condition("temperature = 1", boundary="left")
    heat.add_boundary_condition("temperature = 0", boundary="right")
    problem.solve(report="none")
    # Pe = 40: the temperature stays near 1 until a boundary layer at the outlet.
    assert problem.sample("temperature", [[3.0, 0.5]])[0] > 0.9
