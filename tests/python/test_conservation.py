# SPDX-License-Identifier: LGPL-2.1-or-later
"""The conservation laws of every physics and coupling, checked on the discrete solution.

A conservation law says that what enters a region equals what leaves it plus what it stores.  The
reactions (the residual at the nodes of prescribed values) are the conductive heat flux or the
force that the prescribed boundaries put into the domain.  With the sources, the boundary fluxes
and the storage they close the balance exactly where the discretisation conserves: to round-off,
which the tests check, with Newton's method taken to the round-off level.  Where it does not (the source form of the advection in the penalty and
Taylor-Hood formulations of a flow) the test states the measured gap.
"""

from __future__ import annotations

import dualmesh
import numpy
import pytest
from dualmesh._core import InputError

methods = ["FEM", "DMCDM"]
exact = 1e-11
#: Newton's method to the round-off level of every equation, so that the balances close to it.
to_round_off = {"relative_tolerance": 1e-14}


def reaction(problem, variable, boundaries):
    """The flow of the equation of a variable into the body through side sets of prescribed
    values, each node once (a corner node belongs to two side sets)."""
    nodes = {}
    for boundary in boundaries:
        nodes.update(problem.boundary_flux(variable, boundary, per_node=True))
    return -sum(nodes.values())


def advective_outflow(problem, kernel, variable, coefficient):
    """What an advection kernel carries out of the domain with the discrete mass flux: through
    the boundary with the velocity there, and through the nodes whose pressure is prescribed, where
    the residual of the mass equation is the mass flux that leaves beyond it, at the value of the
    node (coefficient is rho c_p for the heat and rho for the momentum).  This is the identity of
    the discretisation, so it reads the residual itself."""
    through = sum(
        -problem.kernel_boundary_flux(kernel, side) for side in problem.mesh.sideset_names()
    )
    # The residual is zero, to round-off, at the nodes whose mass equations are solved.
    values = problem.values(variable)
    residual = problem._problem._last_residual()
    pressure = problem.variable_index("pressure")
    count = len(problem._problem.variable_names())
    nodes = {
        node
        for side in problem.mesh.sideset_names()
        for node in problem.boundary_flux("pressure", side, per_node=True)
    }
    return through - coefficient * sum(values[n] * residual[n * count + pressure] for n in nodes)


# ---- heat transfer --------------------------------------------------------------------------


@pytest.mark.parametrize("method", methods)
@pytest.mark.parametrize("element_type", ["Quad4", "Tri3", "Quad9", "Tri6"])
@pytest.mark.parametrize("coordinates", ["cartesian", "axisymmetric"])
def test_steady_conduction(method, element_type, coordinates):
    """The heat that the prescribed temperature takes away equals the heat source plus the heat
    that flows in through a given flux and a convective (Robin) condition."""
    mesh = dualmesh.generate_rectangle_mesh(0.5, 1.5, 0, 1, 6, 5, element_type=element_type)
    problem = dualmesh.Problem(mesh, method=method, coordinates=coordinates)
    heat = problem.add_physics(
        "heat_conduction", thermal_conductivity="1 + 0.5*x", heat_source="10 + 5*x*y"
    )
    heat.add_boundary_condition("temperature = 0", boundary="left")
    heat.add_boundary_condition("heat_flux = 3", boundary="right")
    heat.add_boundary_condition("heat_flux = 2*(temperature - 1)", boundary="top")
    problem.solve(report="none", **to_round_off)
    # The integral of the source 10 + 5 x y over [0.5, 1.5] x [0, 1], with 2 pi r in axisymmetry.
    if coordinates == "axisymmetric":
        source = 2 * numpy.pi * (10 * 1.0 + 5 * (1.5**3 - 0.5**3) / 3 * 0.5)
    else:
        source = 10 * 1.0 + 5 * 1.0 * 0.5
    given = 3 * problem.boundary_area("right")
    robin = 2 * (problem.boundary_integral("temperature", "top") - problem.boundary_area("top"))
    balance = reaction(problem, "temperature", ["left"]) + source - given - robin
    assert abs(balance) < exact * source, balance


@pytest.mark.parametrize("method", methods)
@pytest.mark.parametrize("element_type", ["Quad4", "Quad9"])
@pytest.mark.parametrize("time_scheme", ["backward_Euler", "Crank-Nicolson"])
def test_transient_conduction(method, element_type, time_scheme):
    """Over a step of the theta method the stored heat rho c_p dT/dt equals the source, the flux
    given at a boundary and the reaction, each at its weight of the two time levels."""
    mesh = dualmesh.generate_rectangle_mesh(0, 1, 0, 1, 6, 5, element_type=element_type)
    problem = dualmesh.Problem(mesh, method=method)
    heat = problem.add_physics(
        "heat_conduction",
        thermal_conductivity=1.0,
        density=2.0,
        specific_heat=3.0,
        heat_source=10.0,
        initial_condition="x*(1 - x)*y",
    )
    heat.add_boundary_condition("temperature = 0", boundary="left")
    heat.add_boundary_condition("heat_flux = 3", boundary="right")
    problem.initialize()
    start = problem.integrate("temperature")
    problem.solve_transient(
        end_time=0.05,
        time_step=0.05,
        time_scheme=time_scheme,
        time_stepper="fixed",
        report="none",
        **to_round_off,
    )
    storage = 6.0 * (problem.integrate("temperature") - start) / 0.05
    balance = reaction(problem, "temperature", ["left"]) + 10.0 - 3.0 - storage
    assert abs(balance) < exact * abs(storage), balance


@pytest.mark.parametrize("method", methods)
def test_a_nonlinear_storage_is_conserved_exactly(method):
    """The storage m = u + u^3 of an insulated bar, taken as (m - m_old) / dt over each step:
    nothing crosses the boundary, so the stored amount, the sum of m(u_i) w_i over the nodes with
    the weights w_i of the lumped storage of linear elements, stays the same to round-off.  The
    rate form (1 + 3 u^2) du/dt of the same storage does not conserve it."""

    def bar(**storage):
        mesh = dualmesh.generate_line_mesh(start=0.0, end=1.0, num_elements=20)
        problem = dualmesh.Problem(mesh, method=method)
        problem.add_variable(
            "u", initial_condition=lambda x, y, z, t: 1.0 + 0.8 * numpy.cos(numpy.pi * x)
        )
        problem.add_kernel("diffusion", variable="u")
        problem.add_kernel("time_derivative", variable="u", **storage)
        problem.initialize()
        return problem

    def stored(problem):
        x = problem.node_coordinates()[:, 0]
        order = numpy.argsort(x)
        x, u = x[order], problem.values("u")[order]
        weights = numpy.zeros_like(x)
        weights[1:] += 0.5 * numpy.diff(x)
        weights[:-1] += 0.5 * numpy.diff(x)
        return float(numpy.sum(weights * (u + u**3)))

    steps = {"end_time": 0.2, "time_step": 0.05, "time_scheme": "backward_Euler", "report": "none"}
    conserving = bar(storage="u + u^3")
    start = stored(conserving)
    conserving.solve_transient(**steps, **to_round_off)
    assert abs(stored(conserving) - start) < exact * start
    rate = bar(coefficient="1 + 3*u^2")
    rate.solve_transient(**steps, **to_round_off)
    assert abs(stored(rate) - start) > 1e-6 * start


@pytest.mark.parametrize("method", methods)
@pytest.mark.parametrize("element_type", ["Quad4", "Tri3"])
def test_an_insulated_body_of_revolution_heats_uniformly(method, element_type):
    """The cylinder 0 <= r <= 1, 0 <= z <= 1 in axisymmetric coordinates, insulated, at the
    temperature 1 and heated by the uniform source Q = 10 with rho c_p = 6.  It heats uniformly,
    T = 1 + Q t / (rho c_p), and its heat sum_i rho c_p T_i M_i, with M_i the node measures of the
    lumped storage, rises by Q V t, with V = int 2 pi r dr dz = pi its volume.  The storage is
    lumped on these linear elements, and a uniform field conducts no heat, so node i heats at the
    rate S_i / (rho c_p M_i), with S_i the integral of the source over the node.  The field stays
    uniform only where M_i integrates 2 pi r as S_i does, which includes the nodes on the axis
    r = 0."""
    mesh = dualmesh.generate_rectangle_mesh(0, 1, 0, 1, 4, 4, element_type=element_type)
    problem = dualmesh.Problem(mesh, method=method, coordinates="axisymmetric")
    problem.add_physics(
        "heat_conduction",
        thermal_conductivity=1.0,
        density=2.0,
        specific_heat=3.0,
        heat_source=10.0,
        initial_condition=1.0,
    )
    problem.initialize()
    measures = problem.node_measures()
    start = 6.0 * numpy.sum(problem.values("temperature") * measures)
    problem.solve_transient(
        end_time=0.2,
        time_step=0.05,
        time_scheme="backward_Euler",
        time_stepper="fixed",
        report="none",
        **to_round_off,
    )
    temperature = problem.values("temperature")
    assert temperature == pytest.approx(1.0 + 10.0 * 0.2 / 6.0, rel=1e-12)
    heat = 6.0 * numpy.sum(temperature * measures)
    assert abs(heat - start - 10.0 * numpy.pi * 0.2) < exact * heat


@pytest.mark.parametrize("method", methods)
def test_a_body_of_revolution_heated_through_its_base_heats_as_a_column(method):
    """The cylinder 0 <= r <= 1, 0 <= z <= 1 in axisymmetric coordinates, insulated but for the
    uniform inflow q = 5 through its base z = 0, with rho c_p = 6 and the storage and the inflow
    lumped at the nodes.  The temperature depends on z and t only, so the nodes of each row of the
    Quad4 grid share it.  The heat that enters, q pi t through the base of area pi, is the heat
    stored, sum_i rho c_p (T_i - 1) M_i with M_i the node measures.  On this grid every term of
    the equation of a node, the storage, the conduction along z and the inflow, carries the same
    integral of 2 pi r over the node, so the rows hold the same temperature only where the lumped
    storage and the lumped inflow integrate 2 pi r, which includes the nodes on the axis."""
    mesh = dualmesh.generate_rectangle_mesh(0, 1, 0, 1, 4, 4, element_type="Quad4")
    problem = dualmesh.Problem(mesh, method=method, coordinates="axisymmetric")
    problem.add_variable("temperature", initial_condition=lambda x, y, z, t: 1.0)
    problem.add_kernel("diffusion", variable="temperature", diffusivity=1.0)
    problem.add_kernel("time_derivative", variable="temperature", coefficient=6.0)
    problem.add_boundary_condition(
        "Neumann", "bottom", variable="temperature", flux=5.0, quadrature="nodal"
    )
    problem.initialize()
    measures = problem.node_measures()
    problem.solve_transient(
        end_time=0.2,
        time_step=0.05,
        time_scheme="backward_Euler",
        time_stepper="fixed",
        report="none",
        **to_round_off,
    )
    temperature = problem.values("temperature")
    z = problem.node_coordinates()[:, 1]
    for height in numpy.unique(numpy.round(z, 12)):
        row = temperature[numpy.isclose(z, height)]
        assert numpy.ptp(row) < 1e-12 * numpy.abs(row).max(), (height, row)
    stored = 6.0 * numpy.sum((temperature - 1.0) * measures)
    assert abs(stored - 5.0 * numpy.pi * 0.2) < exact * stored


# ---- solid mechanics ------------------------------------------------------------------------


@pytest.mark.parametrize("method", methods)
@pytest.mark.parametrize("element_type", ["Quad4", "Tri3", "Quad9"])
def test_plane_stress_equilibrium(method, element_type):
    """The reactions of the support balance the traction and the body force."""
    mesh = dualmesh.generate_rectangle_mesh(0, 2, 0, 1, 6, 4, element_type=element_type)
    problem = dualmesh.Problem(mesh, method=method)
    solid = problem.add_physics(
        "solid_mechanics",
        formulation="plane_stress",
        youngs_modulus=100.0,
        poissons_ratio=0.3,
        body_force=[0.5, -2.0],
    )
    solid.add_boundary_condition("fixed", boundary="left")
    solid.add_boundary_condition("traction = [1, -3]", boundary="right")
    problem.solve(report="none", **to_round_off)
    assert abs(reaction(problem, "disp_x", ["left"]) + 1.0 * 1 + 0.5 * 2) < exact
    assert abs(reaction(problem, "disp_y", ["left"]) - 3.0 * 1 - 2.0 * 2) < exact


@pytest.mark.parametrize("method", methods)
def test_axisymmetric_equilibrium(method):
    """A thick cylinder under its weight and an inner pressure: the reaction of the support is the
    weight, and the pressure, which is radial, has no resultant along the axis."""
    mesh = dualmesh.generate_rectangle_mesh(1, 2, 0, 1, 5, 4)
    problem = dualmesh.Problem(mesh, method=method, coordinates="axisymmetric")
    solid = problem.add_physics(
        "solid_mechanics", youngs_modulus=100.0, poissons_ratio=0.3, body_force=[0.0, -1.0]
    )
    solid.add_boundary_condition("disp_z = 0", boundary="bottom")
    solid.add_boundary_condition("pressure = 2", boundary="left")
    problem.solve(report="none", **to_round_off)
    weight = numpy.pi * (2**2 - 1**2) * 1.0
    assert abs(reaction(problem, "disp_z", ["bottom"]) - weight) < exact * weight


@pytest.mark.parametrize("method", methods)
@pytest.mark.parametrize("theory", ["Euler-Bernoulli", "Timoshenko"])
def test_beam_equilibrium(method, theory):
    """A cantilever under the load 3 + x per unit length: the shear at the clamp is the load, 8."""
    mesh = dualmesh.generate_line_mesh(start=0.0, end=2.0, num_elements=8)
    problem = dualmesh.Problem(mesh, method=method)
    beam = problem.add_physics(
        "solid_mechanics",
        formulation="beam",
        theory=theory,
        youngs_modulus=100.0,
        poissons_ratio=0.3,
        cross_section={"width": 0.2, "height": 0.1},
        transverse_load="3 + x",
    )
    beam.add_boundary_condition("fixed", boundary="left")
    problem.solve(report="none", **to_round_off)
    assert abs(beam.reaction_force("left")["z"] + 8.0) < exact * 8.0


@pytest.mark.parametrize("method", methods)
@pytest.mark.parametrize("theory", ["first_order", "classical"])
def test_circular_plate_equilibrium(method, theory):
    """A clamped circular plate of radius 1 under the pressure 2: the edge carries 2 pi."""
    mesh = dualmesh.generate_line_mesh(start=0.0, end=1.0, num_elements=8)
    problem = dualmesh.Problem(mesh, method=method, coordinates="axisymmetric")
    plate = problem.add_physics(
        "solid_mechanics",
        formulation="circular_plate",
        theory=theory,
        youngs_modulus=100.0,
        poissons_ratio=0.3,
        thickness=0.1,
        transverse_load=2.0,
    )
    plate.add_boundary_condition("fixed", boundary="right")
    problem.solve(report="none", **to_round_off)
    assert abs(plate.reaction_force("right")["z"] + 2 * numpy.pi) < exact * 2 * numpy.pi


# ---- general form ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", methods)
def test_general_form(method):
    """div(-c grad u - alpha u + gamma) + a u = f: the conservative fluxes leave only at the
    prescribed boundary (the others hold zero total flux), so the reaction balances the source f
    less the absorption a u."""
    mesh = dualmesh.generate_rectangle_mesh(0, 2, 0, 1, 8, 4)
    problem = dualmesh.Problem(mesh, method=method)
    PDE = problem.add_physics(
        "general_form_PDE",
        variables="u",
        c="1 + x",
        alpha=[1.0, 0.5],
        gamma=[0.2, "x"],
        a=2.0,
        f="3 + y",
    )
    PDE.add_boundary_condition("u = 0", boundary="left")
    problem.solve(report="none", **to_round_off)
    source = 3.0 * 2 + 0.5 * 2
    balance = reaction(problem, "u", ["left"]) + source - 2.0 * problem.integrate("u")
    assert abs(balance) < exact * source, balance


# ---- couplings of solids --------------------------------------------------------------------


@pytest.mark.parametrize("method", methods)
def test_thermal_expansion(method):
    """A heated bar held at both ends: the two supports push with equal and opposite forces, and
    they take away the heat of the source."""
    mesh = dualmesh.generate_rectangle_mesh(0, 2, 0, 1, 6, 3)
    problem = dualmesh.Problem(mesh, method=method)
    solid = problem.add_physics(
        "solid_mechanics", formulation="plane_stress", youngs_modulus=100.0, poissons_ratio=0.3
    )
    heat = problem.add_physics("heat_conduction", thermal_conductivity=1.0, heat_source=5.0)
    problem.add_coupling(
        "thermal_expansion",
        solid_mechanics=solid,
        heat_conduction=heat,
        linear_expansion_coefficient=1e-3,
        stress_free_temperature=0.0,
    )
    solid.add_boundary_condition("fixed", boundary=["left", "right"])
    heat.add_boundary_condition("temperature = 0", boundary=["left", "right"])
    problem.solve(report="none", **to_round_off)
    left = reaction(problem, "disp_x", ["left"])
    assert abs(left) > 1e-3
    assert abs(reaction(problem, "disp_x", ["left", "right"])) < exact * abs(left)
    assert abs(reaction(problem, "temperature", ["left", "right"]) + 5.0 * 2) < exact * 10


# ---- fluid dynamics -------------------------------------------------------------------------


def channel(method, element_type="Quad4", formulation="equal_order", density=10.0, outlet="free"):
    """A channel 3 x 1 with a skewed inflow and a body force: it develops along its length, so the
    momentum that enters differs from the momentum that leaves.  The outlet is traction-free, or
    has its pressure prescribed."""
    mesh = dualmesh.generate_rectangle_mesh(0, 3, 0, 1, 12, 4, element_type=element_type)
    problem = dualmesh.Problem(mesh, method=method)
    flow = problem.add_physics(
        "fluid_dynamics",
        density=density,
        dynamic_viscosity=1.0,
        formulation=formulation,
        viscous_form="laplacian",
        body_force=[0.3, 0.1],
    )
    flow.add_boundary_condition(
        ["velocity_x = 6*y*(1 - y)*(1 + y)/1.5", "velocity_y = 0"], boundary="left"
    )
    flow.add_boundary_condition(["velocity_x = 0", "velocity_y = 0"], boundary=["top", "bottom"])
    if outlet == "pressure":
        problem.add_boundary_condition("Dirichlet", "right", variable="pressure", value=0.0)
    problem.solve(report="none", **to_round_off)
    return problem


@pytest.mark.parametrize("method", methods)
@pytest.mark.parametrize("outlet", ["free", "pressure"])
def test_mass(method, outlet):
    """What flows in flows out: the flows of the mass equation through the boundaries sum to zero,
    and the inflow is that of the inlet profile.  Where the outlet pressure is prescribed, the
    mass equations of its nodes are replaced, and the flow there is their reaction."""
    problem = channel(method, outlet=outlet)
    inflow = problem.boundary_integral("velocity_x", "left")
    flows = {side: problem.boundary_flux("pressure", side) for side in problem.mesh.sideset_names()}
    assert abs(flows["left"] + inflow) < exact * inflow, flows
    assert abs(sum(flows.values())) < exact * inflow, flows


@pytest.mark.parametrize("method", methods)
@pytest.mark.parametrize("element_type", ["Quad4", "Tri3", "Quad9"])
@pytest.mark.parametrize("outlet", ["free", "pressure"])
@pytest.mark.parametrize("density", [0.0, 10.0])
def test_momentum(method, element_type, outlet, density):
    """The forces on the fluid (the reactions of the walls and the inlet, and the body force)
    equal the momentum that it carries out with the discrete mass flux.  The outlet takes no
    force: it is traction-free, or its pressure is zero."""
    problem = channel(method, element_type, density=density, outlet=outlet)
    walls = ["left", "top", "bottom"]
    for component, (velocity, force) in enumerate((("velocity_x", 0.3), ("velocity_y", 0.1))):
        kernel = f"advection_{component}"
        carried = advective_outflow(problem, kernel, velocity, density) if density else 0.0
        balance = reaction(problem, velocity, walls) + force * 3.0 - carried
        assert abs(balance) < exact * 10, (velocity, balance)


def cavity(method, n=8):
    """A closed cavity with a smooth lid."""
    mesh = dualmesh.generate_rectangle_mesh(0, 1, 0, 1, n, n)
    problem = dualmesh.Problem(mesh, method=method)
    flow = problem.add_physics(
        "fluid_dynamics", density=100.0, dynamic_viscosity=1.0, formulation="equal_order"
    )
    flow.add_boundary_condition(
        ["velocity_x = 16*x*x*(1 - x)*(1 - x)", "velocity_y = 0"], boundary="top"
    )
    flow.add_boundary_condition(
        ["velocity_x = 0", "velocity_y = 0"], boundary=["left", "right", "bottom"]
    )
    return problem


@pytest.mark.parametrize("method", methods)
def test_momentum_of_a_closed_flow(method):
    """The forces of the walls on a steady enclosed flow sum to zero: the lid drags the fluid one
    way and the other walls hold it back."""
    problem = cavity(method)
    problem.solve(report="none", **to_round_off)
    walls = ["left", "right", "bottom", "top"]
    lid = reaction(problem, "velocity_x", ["top"])
    for velocity in ("velocity_x", "velocity_y"):
        assert abs(reaction(problem, velocity, walls)) < exact * abs(lid), velocity


@pytest.mark.parametrize("method", methods)
def test_the_walls_of_a_driven_channel_carry_its_body_force(method):
    """A body force f drives the flow through a channel of 8 by 6 triangles, periodic in x, from
    rest to its steady state at t = 4: the walls then carry the whole force, so the force on them,
    minus the sum of the reactions, is f times the area of the channel.  The transient solve of
    the flow is the monolithic one, whose steady state does not depend on the step (that of the
    projection does above a step, test_projection.py)."""
    mesh = dualmesh.generate_rectangle_mesh(0.0, 2.0, 0.0, 1.0, 8, 6, element_type="Tri3")
    problem = dualmesh.Problem(mesh, method=method, distributed=False)
    problem.add_physics(
        "fluid_dynamics",
        density=1.0,
        dynamic_viscosity=1.0,
        formulation="equal_order",
        body_force=[3.0, 0.0],
    )
    problem.add_boundary_condition("periodic", primary="left", secondary="right")
    for name in ("velocity_x", "velocity_y"):
        problem.add_boundary_condition("Dirichlet", ["bottom", "top"], variable=name, value=0.0)
    problem.solve_transient(end_time=4.0, time_step=0.05, report="none")
    force = problem.boundary_flux("velocity_x", "bottom") + problem.boundary_flux(
        "velocity_x", "top"
    )
    assert force == pytest.approx(3.0 * 2.0 * 1.0, rel=1e-6)
    assert -problem.boundary_flux("velocity_y", "bottom") - problem.boundary_flux(
        "velocity_y", "top"
    ) == pytest.approx(0.0, abs=1e-8)


@pytest.mark.parametrize("method", methods)
@pytest.mark.parametrize("time_scheme", ["backward_Euler", "Crank-Nicolson"])
def test_momentum_of_a_flow_started_from_rest(method, time_scheme):
    """Over a step the forces of the walls change the momentum of the fluid, rho int dv/dt.  The
    second step is checked: its old time carries the velocity that the mass equation of the
    first step conserved."""
    problem = cavity(method)
    dt = 0.01
    problem.solve_transient(
        end_time=dt,
        time_step=dt,
        time_scheme=time_scheme,
        time_stepper="fixed",
        report="none",
        **to_round_off,
    )
    start = {v: problem.integrate(v) for v in ("velocity_x", "velocity_y")}
    problem.solve_transient(
        start_time=dt,
        end_time=2 * dt,
        time_step=dt,
        time_scheme=time_scheme,
        time_stepper="fixed",
        report="none",
    )
    walls = ["left", "right", "bottom", "top"]
    lid = abs(reaction(problem, "velocity_x", ["top"]))
    for velocity in ("velocity_x", "velocity_y"):
        storage = 100.0 * (problem.integrate(velocity) - start[velocity]) / dt
        balance = reaction(problem, velocity, walls) - storage
        assert abs(balance) < exact * lid, (velocity, balance)


# ---- heat carried by a flow -----------------------------------------------------------------

schemes = [("FEM", None), ("FEM", "central"), ("DMCDM", None), ("DMCDM", "central")]
schemes += [("DMCDM", "upwind"), ("DMCDM", "linear_upwind"), ("DMCDM", "limited")]


@pytest.mark.parametrize("method, scheme", schemes)
def test_heat_in_a_closed_cavity(method, scheme):
    """Natural convection in a square heated on the left and cooled at the top, which has no
    symmetry that would make the balance close by itself: the heat that enters equals the heat
    that leaves."""
    mesh = dualmesh.generate_rectangle_mesh(0, 1, 0, 1, 8, 8)
    problem = dualmesh.Problem(mesh, method=method)
    flow = problem.add_physics(
        "fluid_dynamics", dynamic_viscosity=0.71, density=1.0, formulation="equal_order"
    )
    heat = problem.add_physics(
        "heat_conduction",
        thermal_conductivity=1.0,
        density=1.0,
        specific_heat=1.0,
        advection_scheme=scheme,
        initial_condition=0.0,
    )
    problem.add_coupling(
        "nonisothermal_flow",
        heat_conduction=heat,
        fluid_dynamics=flow,
        gravity=[0.0, -1.0],
        volumetric_expansion_coefficient=1e4 * 0.71,
    )
    flow.add_boundary_condition(
        ["velocity_x = 0", "velocity_y = 0"], boundary=["left", "right", "bottom", "top"]
    )
    heat.add_boundary_condition("temperature = 1", boundary="left")
    heat.add_boundary_condition("temperature = 0", boundary="top")
    problem.solve(report="none", **to_round_off)
    hot = reaction(problem, "temperature", ["left"])
    assert abs(reaction(problem, "temperature", ["left", "top"])) < exact * hot


@pytest.mark.parametrize("method, scheme", schemes)
@pytest.mark.parametrize("outlet", ["free", "pressure"])
def test_heat_carried_out_of_a_channel(method, scheme, outlet):
    """A channel heated from below: the heat that the wall gives and the inlet conducts equals the
    heat that the flow carries out with the discrete mass flux."""
    mesh = dualmesh.generate_rectangle_mesh(0, 3, 0, 1, 12, 4)
    problem = dualmesh.Problem(mesh, method=method)
    flow = problem.add_physics(
        "fluid_dynamics",
        density=10.0,
        dynamic_viscosity=1.0,
        formulation="equal_order",
        viscous_form="laplacian",
    )
    heat = problem.add_physics(
        "heat_conduction",
        thermal_conductivity=0.05,
        density=10.0,
        specific_heat=1.0,
        advection_scheme=scheme,
    )
    problem.add_coupling("nonisothermal_flow", heat_conduction=heat, fluid_dynamics=flow)
    flow.add_boundary_condition(
        ["velocity_x = 6*y*(1 - y)*(1 + y)/1.5", "velocity_y = 0"], boundary="left"
    )
    flow.add_boundary_condition(["velocity_x = 0", "velocity_y = 0"], boundary=["top", "bottom"])
    if outlet == "pressure":
        problem.add_boundary_condition("Dirichlet", "right", variable="pressure", value=0.0)
    heat.add_boundary_condition("temperature = 0", boundary="left")
    heat.add_boundary_condition("temperature = 1 + x", boundary="bottom")
    problem.solve(report="none", **to_round_off)
    carried = advective_outflow(problem, "advection_2", "temperature", 10.0)
    balance = reaction(problem, "temperature", ["left", "bottom"]) - carried
    assert abs(balance) < exact * carried, balance


@pytest.mark.parametrize(
    "heat_capacity, message",
    [
        ({"density": 1.0, "specific_heat": 4182.0}, "the density 1.0"),
        ({}, "needs its density and specific_heat"),
    ],
)
def test_a_flow_that_carries_another_heat_capacity_than_it_stores_is_refused(
    heat_capacity, message
):
    """Heat is conserved only when its storage and its convection take one density and one
    specific heat: a heat_conduction physics of another density than the flow, or without a
    specific heat, is refused."""
    mesh = dualmesh.generate_rectangle_mesh(0, 1, 0, 1, 2, 2)
    problem = dualmesh.Problem(mesh)
    flow = problem.add_physics("fluid_dynamics", density=1000.0, dynamic_viscosity=1e-3)
    heat = problem.add_physics("heat_conduction", thermal_conductivity=0.6, **heat_capacity)
    problem.add_coupling("nonisothermal_flow", heat_conduction=heat, fluid_dynamics=flow)
    with pytest.raises(InputError, match=message):
        problem.solve(report="none")
