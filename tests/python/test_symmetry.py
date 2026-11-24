# SPDX-License-Identifier: LGPL-2.1-or-later
"""The symmetries of every physics and coupling, checked on the discrete solution.

A problem whose domain, mesh, data and conditions are symmetric under a mirror or a point
reflection has a solution with the same symmetry: a scalar and the velocity component along the
mirror are even, the component across it is odd.  The discretisation keeps it exactly on a
symmetric mesh, which the tests check to round-off.  The triangles take diagonal="alternate",
which a user gives to make a triangulation symmetric (the default draws every diagonal the same
way, and the mesh then has no mirror symmetry).

A self-adjoint operator (conduction, elasticity) gives a symmetric matrix with FEM, whose test
functions are its trial functions, and with it reciprocity: the response at B to a unit source at
A equals the response at A to a unit source at B.  DMCDM tests with the control volumes, so its
matrix is not symmetric where the coefficients vary, and the measured gap is stated.
"""

from __future__ import annotations

import dualmesh
import numpy
import pytest

methods = ["FEM", "DMCDM"]
elements = ["Quad4", "Tri3", "Quad9", "Tri6"]
exact = 1e-12


def mesh(x, y, n, element_type="Quad4"):
    diagonal = "alternate" if element_type.startswith("Tri") else "right"
    return dualmesh.generate_rectangle_mesh(
        x[0], x[1], y[0], y[1], n[0], n[1], element_type=element_type, diagonal=diagonal
    )


def partner(points, centre, axis=None):
    """The node at the image of every node: across the line x_axis = centre[axis], or through
    the point centre when axis is None."""
    image = points.copy()
    if axis is None:
        image = 2 * numpy.asarray(centre) - image
    else:
        image[:, axis] = 2 * centre[axis] - image[:, axis]
    index = {tuple(numpy.round(p, 9)): i for i, p in enumerate(points)}
    return numpy.array([index[tuple(numpy.round(q, 9))] for q in image])


def asymmetry(problem, variable, image, parity=1.0):
    """The largest difference between a field and its image, over its largest value; parity -1
    for an odd field."""
    values = problem.values(variable)
    return numpy.abs(values - parity * values[image]).max() / numpy.abs(values).max()


def points_of(problem):
    return numpy.asarray(problem.mesh.points())[:, : problem.mesh.space_dimension]


# ---- mirror symmetry ------------------------------------------------------------------------


@pytest.mark.parametrize("method", methods)
@pytest.mark.parametrize("element_type", elements)
def test_conduction(method, element_type):
    """A conductivity, a source and walls symmetric about x = 1."""
    problem = dualmesh.Problem(mesh((0, 2), (0, 1), (8, 4), element_type), method=method)
    heat = problem.add_physics(
        "heat_transfer",
        thermal_conductivity="1 + (x - 1)^2",
        heat_source="2 + cos(3*(x - 1)) + y",
    )
    heat.add_boundary_condition("temperature = 0", boundary=["left", "right"])
    heat.add_boundary_condition("heat_flux = 2*(temperature - 1)", boundary="top")
    problem.solve(report="none")
    image = partner(points_of(problem), (1.0, 0.0), axis=0)
    assert asymmetry(problem, "temperature", image) < exact


@pytest.mark.parametrize("method", methods)
def test_transient_conduction(method):
    """Crank-Nicolson from a symmetric start."""
    problem = dualmesh.Problem(mesh((0, 2), (0, 1), (8, 4)), method=method)
    heat = problem.add_physics(
        "heat_transfer",
        thermal_conductivity=1.0,
        density=1.0,
        specific_heat=1.0,
        heat_source="1 + (x - 1)^2",
        initial_condition="cos(x - 1)*y",
    )
    heat.add_boundary_condition("temperature = 0", boundary=["left", "right"])
    problem.solve_transient(
        end_time=0.1, time_step=0.02, time_scheme="Crank-Nicolson", report="none"
    )
    image = partner(points_of(problem), (1.0, 0.0), axis=0)
    assert asymmetry(problem, "temperature", image) < exact


@pytest.mark.parametrize("method", methods)
@pytest.mark.parametrize("element_type", elements)
def test_plane_elasticity(method, element_type):
    """A block on a fixed base, loaded on its top: u_x is odd about x = 1, u_y even."""
    problem = dualmesh.Problem(mesh((0, 2), (0, 1), (8, 4), element_type), method=method)
    solid = problem.add_physics(
        "solid_mechanics",
        formulation="plane_strain",
        youngs_modulus="100*(1 + (x - 1)^2)",
        poissons_ratio=0.3,
        body_force=[0.0, -1.0],
    )
    solid.add_boundary_condition("fixed", boundary="bottom")
    solid.add_boundary_condition("traction = [0, -1]", boundary="top")
    problem.solve(report="none")
    image = partner(points_of(problem), (1.0, 0.0), axis=0)
    assert asymmetry(problem, "disp_x", image, -1.0) < exact
    assert asymmetry(problem, "disp_y", image) < exact


@pytest.mark.parametrize("method", methods)
def test_axisymmetric_elasticity(method):
    """A thick cylinder under an inner pressure, held at both ends: u_r is even about the middle
    plane z = 1, and u_z odd (measured against u_r, as it vanishes up to round-off)."""
    problem = dualmesh.Problem(
        mesh((1, 2), (0, 2), (4, 8)), method=method, coordinates="axisymmetric"
    )
    solid = problem.add_physics("solid_mechanics", youngs_modulus=100.0, poissons_ratio=0.3)
    solid.add_boundary_condition("disp_z = 0", boundary=["bottom", "top"])
    solid.add_boundary_condition("pressure = 2", boundary="left")
    problem.solve(report="none")
    image = partner(points_of(problem), (0.0, 1.0), axis=1)
    assert asymmetry(problem, "disp_r", image) < exact
    u_z = problem.values("disp_z")
    assert numpy.abs(u_z + u_z[image]).max() < exact * numpy.abs(problem.values("disp_r")).max()


@pytest.mark.parametrize("method", methods)
@pytest.mark.parametrize("theory", ["Euler-Bernoulli", "Timoshenko"])
def test_beam(method, theory):
    """A beam pinned at both ends under a load symmetric about its middle: the deflection is
    even."""
    problem = dualmesh.Problem(
        dualmesh.generate_line_mesh(start=0.0, end=2.0, num_elements=8), method=method
    )
    beam = problem.add_physics(
        "solid_mechanics",
        formulation="beam",
        theory=theory,
        youngs_modulus=100.0,
        poissons_ratio=0.3,
        cross_section={"width": 0.2, "height": 0.1},
        transverse_load="1 + (x - 1)^2",
    )
    beam.add_boundary_condition("disp_z = 0", boundary=["left", "right"])
    beam.add_boundary_condition("disp_x = 0", boundary="left")
    problem.solve(report="none")
    image = partner(points_of(problem), (1.0,), axis=0)
    assert asymmetry(problem, "disp_z", image) < exact


@pytest.mark.parametrize("method", methods)
@pytest.mark.parametrize("element_type", elements)
@pytest.mark.parametrize("density", [0.0, 50.0])
def test_channel_flow(method, element_type, density):
    """A channel with an inflow symmetric about y = 1/2: u and p even, v odd, Stokes and
    Navier-Stokes."""
    problem = dualmesh.Problem(mesh((0, 3), (0, 1), (12, 4), element_type), method=method)
    flow = problem.add_physics(
        "fluid_dynamics", density=density, dynamic_viscosity=1.0, formulation="equal_order"
    )
    flow.add_boundary_condition(["velocity_x = 6*y*(1 - y)", "velocity_y = 0"], boundary="left")
    flow.add_boundary_condition(["velocity_x = 0", "velocity_y = 0"], boundary=["top", "bottom"])
    problem.solve(report="none")
    image = partner(points_of(problem), (0.0, 0.5), axis=1)
    speed = numpy.abs(problem.values("velocity_x")).max()
    assert asymmetry(problem, "velocity_x", image) < exact
    v = problem.values("velocity_y")
    assert numpy.abs(v + v[image]).max() < exact * speed
    assert asymmetry(problem, "pressure", image) < exact


schemes = [("FEM", None), ("FEM", "central"), ("DMCDM", None), ("DMCDM", "central")]
schemes += [("DMCDM", "upwind"), ("DMCDM", "limited")]


@pytest.mark.parametrize("method, scheme", schemes)
def test_natural_convection(method, scheme):
    """A cavity heated on the left and cooled on the right is symmetric under the point
    reflection through its centre: T and the velocity are odd, the pressure even."""
    problem = dualmesh.Problem(mesh((0, 1), (0, 1), (10, 10)), method=method)
    flow = problem.add_physics(
        "fluid_dynamics", dynamic_viscosity=0.71, density=1.0, formulation="equal_order"
    )
    heat = problem.add_physics("heat_transfer", thermal_conductivity=1.0, advection_scheme=scheme)
    problem.add_coupling(
        "nonisothermal_flow",
        heat_transfer=heat,
        fluid_dynamics=flow,
        gravity=[0.0, -1.0],
        volumetric_expansion_coefficient=1e4 * 0.71,
    )
    flow.add_boundary_condition(
        ["velocity_x = 0", "velocity_y = 0"], boundary=["left", "right", "bottom", "top"]
    )
    heat.add_boundary_condition("temperature = 0.5", boundary="left")
    heat.add_boundary_condition("temperature = -0.5", boundary="right")
    problem.solve(report="none")
    image = partner(points_of(problem), (0.5, 0.5))
    for variable in ("temperature", "velocity_x", "velocity_y"):
        assert asymmetry(problem, variable, image, -1.0) < exact, variable
    assert asymmetry(problem, "pressure", image) < exact


@pytest.mark.parametrize("method", methods)
def test_thermal_expansion(method):
    """A plate heated by a source symmetric about x = 1 and held on its base."""
    problem = dualmesh.Problem(mesh((0, 2), (0, 1), (8, 4)), method=method)
    solid = problem.add_physics(
        "solid_mechanics", formulation="plane_stress", youngs_modulus=100.0, poissons_ratio=0.3
    )
    heat = problem.add_physics(
        "heat_transfer", thermal_conductivity=1.0, heat_source="5*(1 + (x - 1)^2)"
    )
    problem.add_coupling(
        "thermal_expansion",
        solid_mechanics=solid,
        heat_transfer=heat,
        linear_expansion_coefficient=1e-3,
        stress_free_temperature=0.0,
    )
    solid.add_boundary_condition("fixed", boundary="bottom")
    heat.add_boundary_condition("temperature = 0", boundary=["left", "right", "bottom"])
    problem.solve(report="none")
    image = partner(points_of(problem), (1.0, 0.0), axis=0)
    assert asymmetry(problem, "temperature", image) < exact
    assert asymmetry(problem, "disp_x", image, -1.0) < exact
    assert asymmetry(problem, "disp_y", image) < exact


@pytest.mark.parametrize("method", methods)
def test_general_form(method):
    """div(-c grad u - alpha u) + a u = f with c, f even about x = 1 and alpha along the
    mirror."""
    problem = dualmesh.Problem(mesh((0, 2), (0, 1), (8, 4)), method=method)
    PDE = problem.add_physics(
        "general_form_PDE",
        variables="u",
        c="1 + (x - 1)^2",
        alpha=[0.0, 0.5],
        a=2.0,
        f="3 + cos(x - 1)",
    )
    PDE.add_boundary_condition("u = 0", boundary=["left", "right"])
    problem.solve(report="none")
    image = partner(points_of(problem), (1.0, 0.0), axis=0)
    assert asymmetry(problem, "u", image) < exact


# ---- symmetric matrices and reciprocity -----------------------------------------------------


def conduction(method, element_type, coordinates, source=None):
    problem = dualmesh.Problem(
        mesh((0.5, 1.5), (0, 1), (8, 8), element_type), method=method, coordinates=coordinates
    )
    heat = problem.add_physics(
        "heat_transfer", thermal_conductivity="1 + x*y", initial_condition=0.0
    )
    heat.add_boundary_condition("temperature = 0", boundary="left")
    heat.add_boundary_condition("heat_flux = 2*temperature", boundary="top")
    if source is not None:
        problem.add_point_source(variable="temperature", value=1.0, points=source)
    return problem


def elasticity(method, element_type, coordinates, load=None):
    problem = dualmesh.Problem(
        mesh((0.5, 1.5), (0, 1), (8, 8), element_type), method=method, coordinates=coordinates
    )
    plane = {} if coordinates == "axisymmetric" else {"formulation": "plane_strain"}
    solid = problem.add_physics(
        "solid_mechanics", youngs_modulus="100*(1 + x)", poissons_ratio=0.3, **plane
    )
    solid.add_boundary_condition("fixed", boundary="bottom")
    if load is not None:
        point, component = load
        problem.add_point_source(
            variable=solid.variable_names()[component],
            value=1.0,
            points=point,
        )
    return problem, solid.variable_names()


def matrix_asymmetry(problem):
    _, Jacobian = problem.linear_system()
    Jacobian = Jacobian.toarray()
    return numpy.abs(Jacobian - Jacobian.T).max() / numpy.abs(Jacobian).max()


coordinate_systems = ["cartesian", "axisymmetric"]


@pytest.mark.parametrize("element_type", ["Quad4", "Tri3", "Quad9"])
@pytest.mark.parametrize("coordinates", coordinate_systems)
def test_finite_element_matrices_are_symmetric(element_type, coordinates):
    """Conduction with a varying conductivity and a convective condition, and elasticity with a
    varying modulus."""
    assert matrix_asymmetry(conduction("FEM", element_type, coordinates)) < exact
    assert matrix_asymmetry(elasticity("FEM", element_type, coordinates)[0]) < exact


@pytest.mark.parametrize(
    "method, formulation, element_type",
    [
        ("FEM", "Taylor-Hood", "Quad9"),
        ("FEM", "equal_order", "Quad4"),
        ("DMCDM", "equal_order", "Quad4"),
    ],
)
def test_the_Stokes_blocks_are_symmetric(method, formulation, element_type):
    """The viscous block is symmetric, and the gradient block is minus the transpose of the
    divergence block, on both methods."""
    problem = dualmesh.Problem(mesh((0, 1), (0, 1), (4, 4), element_type), method=method)
    flow = problem.add_physics(
        "fluid_dynamics", density=0.0, dynamic_viscosity=1.0, formulation=formulation
    )
    flow.add_boundary_condition(["velocity_x = 1", "velocity_y = 0"], boundary="top")
    flow.add_boundary_condition(
        ["velocity_x = 0", "velocity_y = 0"], boundary=["left", "right", "bottom"]
    )
    _, Jacobian = problem.linear_system()
    Jacobian = Jacobian.toarray()
    pressure = numpy.arange(Jacobian.shape[0]) % 3 == problem.variable_index("pressure")
    viscous = Jacobian[numpy.ix_(~pressure, ~pressure)]
    gradient = Jacobian[numpy.ix_(~pressure, pressure)]
    divergence = Jacobian[numpy.ix_(pressure, ~pressure)]
    assert numpy.abs(viscous - viscous.T).max() < exact * numpy.abs(viscous).max()
    assert numpy.abs(gradient + divergence.T).max() < exact * numpy.abs(gradient).max()


A, B = (0.75, 0.25), (1.25, 0.75)


@pytest.mark.parametrize("element_type", ["Quad4", "Tri3", "Quad9"])
@pytest.mark.parametrize("coordinates", coordinate_systems)
def test_reciprocity(element_type, coordinates):
    """The temperature at B of a unit source at A is that at A of a unit source at B (the Green's
    function is symmetric), and the displacement along y at B of a unit load along x at A is
    the displacement along x at A of a unit load along y at B (Maxwell-Betti).  In
    axisymmetric coordinates the sources are rings."""
    at_b = conduction("FEM", element_type, coordinates, A)
    at_a = conduction("FEM", element_type, coordinates, B)
    at_b.solve(report="none")
    at_a.solve(report="none")
    t_b = at_b.sample("temperature", numpy.array([B]))[0]
    t_a = at_a.sample("temperature", numpy.array([A]))[0]
    assert abs(t_b - t_a) < exact * abs(t_b)
    first, names = elasticity("FEM", element_type, coordinates, (A, 0))
    second, _ = elasticity("FEM", element_type, coordinates, (B, 1))
    first.solve(report="none")
    second.solve(report="none")
    u = first.sample(names[1], numpy.array([B]))[0]
    w = second.sample(names[0], numpy.array([A]))[0]
    assert abs(u - w) < exact * abs(u)
