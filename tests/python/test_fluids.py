# SPDX-License-Identifier: LGPL-2.1-or-later
"""Verification of the penalty flow module against Chapter 9 of Reddy's book, and of the
stabilization of an equal-order flow of two fluids in layers."""

from __future__ import annotations

import dualmesh
import numpy
import pytest

default_penalty = 1.0e8


def flow_problem(mesh, viscosity=1.0, density=0.0, penalty=default_penalty, method="dmcdm"):
    """Penalty formulation of a two-dimensional incompressible flow."""
    problem = dualmesh.Problem(mesh, method=method)
    problem.add_variable("u")
    problem.add_variable("v")
    velocities = ["u", "v"]
    for variable in velocities:
        problem.add_kernel(
            "viscous_stress",
            variable=variable,
            velocities=velocities,
            dynamic_viscosity=viscosity,
        )
        problem.add_kernel(
            "penalty_incompressibility",
            variable=variable,
            velocities=velocities,
            penalty_parameter=penalty,
        )
        if density > 0.0:
            problem.add_kernel(
                "advection",
                variable=variable,
                velocity=velocities,
                coefficient=density,
            )
    problem.add_property("penalty_pressure", velocities=velocities, penalty_parameter=penalty)
    return problem


def test_example_9_8_3_squeezed_flow():
    """Table 9.8.2 (DMCDM column): creeping flow squeezed between two plates."""
    half_length, half_gap, plate_velocity = 6.0, 2.0, 1.0
    dx = [0.5] * 8 + [0.25] * 4 + [0.125] * 8
    dy = [0.125] * 16
    mesh = dualmesh.generate_rectangle_mesh(
        x_coordinates=dualmesh.meshing.coordinates_from_spacings(0.0, dx),
        y_coordinates=dualmesh.meshing.coordinates_from_spacings(0.0, dy),
    )
    problem = flow_problem(mesh)
    # Symmetry planes: v = 0 on y = 0 and u = 0 on x = 0.
    problem.add_boundary_condition("Dirichlet", "bottom", variable="v", value=0.0)
    problem.add_boundary_condition("Dirichlet", "left", variable="u", value=0.0)
    # The plate at y = b moves down with velocity V0 and does not slip.
    problem.add_boundary_condition("Dirichlet", "top", variable="u", value=0.0)
    problem.add_boundary_condition("Dirichlet", "top", variable="v", value=-plate_velocity)
    problem.solve()

    y = numpy.arange(0.0, 2.0, 0.125)
    reference = {
        4.0: [
            3.0302,
            3.0180,
            2.9805,
            2.9185,
            2.8316,
            2.7208,
            2.5854,
            2.4269,
            2.2446,
            2.0401,
            1.8128,
            1.5643,
            1.2937,
            1.0024,
            0.6896,
            0.3557,
        ],
        6.0: [
            4.2150,
            4.2019,
            4.1622,
            4.0961,
            4.0030,
            3.8830,
            3.7348,
            3.5589,
            3.3531,
            3.1181,
            2.8497,
            2.5506,
            2.2088,
            1.8382,
            1.3828,
            0.9513,
        ],
    }
    for x, expected in reference.items():
        points = numpy.column_stack([numpy.full_like(y, x), y])
        assert problem.sample("u", points) == pytest.approx(expected, abs=2e-3)

    # The pressure recovered from the penalty relation follows Nadai's
    # approximate solution P = 3 mu V0 (a^2 + y^2 - x^2) / (2 b^3) in the
    # interior, away from the corner singularities.
    pressure = problem.property_at_centroids("pressure").ravel()
    centroids = numpy.array(
        [problem.mesh.element_centroid(e) for e in range(problem.mesh.num_elements)]
    )
    analytical = (
        3.0
        * plate_velocity
        * (half_length**2 + centroids[:, 1] ** 2 - centroids[:, 0] ** 2)
        / (2.0 * half_gap**3)
    )
    interior = (centroids[:, 0] < 4.0) & (centroids[:, 1] < 1.0)
    assert numpy.corrcoef(pressure[interior], analytical[interior])[0, 1] > 0.999
    # And the magnitude is right to within a few percent in the interior
    assert numpy.polyfit(analytical[interior], pressure[interior], 1)[0] == pytest.approx(
        1.0, rel=0.05
    )


def cavity_mesh():
    """The 16 x 20 graded mesh of Example 9.8.4 (refined towards the lid)."""
    dx = [0.0625] * 16
    dy = [0.0625] * 12 + [0.03125] * 8
    return dualmesh.generate_rectangle_mesh(
        x_coordinates=dualmesh.meshing.coordinates_from_spacings(0.0, dx),
        y_coordinates=dualmesh.meshing.coordinates_from_spacings(0.0, dy),
    )


def lid_driven_cavity(Reynolds_number, method="dmcdm", **solver_options):
    problem = flow_problem(
        cavity_mesh(), viscosity=1.0, density=float(Reynolds_number), method=method
    )
    for variable in ("u", "v"):
        problem.add_boundary_condition(
            "Dirichlet",
            ["left", "right", "bottom"],
            variable=variable,
            value=0.0,
        )
    problem.add_boundary_condition(
        "Dirichlet",
        "top",
        variable="u",
        value=1.0,
    )
    problem.add_boundary_condition("Dirichlet", "top", variable="v", value=0.0)
    problem.solve(**solver_options)
    return problem


cavity_y = [
    0.0625,
    0.1250,
    0.1875,
    0.2500,
    0.3125,
    0.3750,
    0.4375,
    0.5000,
    0.5625,
    0.6250,
    0.6875,
    0.7500,
    0.7813,
    0.8125,
    0.8438,
    0.8750,
    0.9063,
    0.9375,
    0.9688,
]
cavity_Stokes = [
    -0.0369,
    -0.0663,
    -0.0919,
    -0.1158,
    -0.1386,
    -0.1598,
    -0.1775,
    -0.1881,
    -0.1857,
    -0.1619,
    -0.1057,
    -0.0046,
    0.0673,
    0.1542,
    0.2584,
    0.3772,
    0.5149,
    0.6634,
    0.8279,
]
cavity_re1000 = [
    -0.1430,
    -0.2177,
    -0.2502,
    -0.2301,
    -0.1776,
    -0.1255,
    -0.0803,
    -0.0365,
    0.0091,
    0.0570,
    0.1097,
    0.1612,
    0.1861,
    0.2084,
    0.2273,
    0.2383,
    0.2496,
    0.2937,
    0.5245,
]


def test_example_9_8_4_cavity_Stokes():
    """Table 9.8.3, DMCDM column at Re = 0 (Stokes flow)."""
    problem = lid_driven_cavity(0)
    points = numpy.column_stack([numpy.full(len(cavity_y), 0.5), cavity_y])
    assert problem.sample("u", points) == pytest.approx(cavity_Stokes, abs=2e-3)


def test_example_9_8_4_cavity_Reynolds_1000():
    """Table 9.8.3, DMCDM column at Re = 1000 (Navier-Stokes).

    The solve converges at this Reynolds number by itself: where Newton's
    method does not converge from the Stokes start, it ramps the lid speed.
    """
    problem = lid_driven_cavity(1000)
    points = numpy.column_stack([numpy.full(len(cavity_y), 0.5), cavity_y])
    assert problem.sample("u", points) == pytest.approx(cavity_re1000, abs=2e-3)


def test_example_9_8_4_cavity_Reynolds_1000_by_direct_iteration():
    """The same solution obtained by direct (Picard) iteration with relaxation."""
    problem = lid_driven_cavity(1000, nonlinear_solver="picard", relaxation=0.5)
    points = numpy.column_stack([numpy.full(len(cavity_y), 0.5), cavity_y])
    assert problem.sample("u", points) == pytest.approx(cavity_re1000, abs=2e-3)


def test_penalty_flow_is_almost_divergence_free():
    """The penalty formulation keeps the divergence small: |div v| ~ p / gamma."""
    problem = lid_driven_cavity(0)
    du = problem.gradient_at_centroids("u")
    dv = problem.gradient_at_centroids("v")
    divergence = du[:, 0] + dv[:, 1]
    assert numpy.max(numpy.abs(divergence)) < 1e-5


# ---------------------------------------------------------------------------
# Two fluids in layers
# ---------------------------------------------------------------------------
# A channel 0 < x < L, 0 < y < H holds a fluid of viscosity mu_1 below y = d and one of viscosity
# mu_2 above.  A body force g per unit volume along x (gravity along an inclined channel) drives a
# creeping flow, and both ends are open (the do-nothing condition mu du/dn - p n = 0 of the
# laplacian form of viscous_stress, which the developed flow satisfies).  For a flow along x that
# varies with y only, the laplacian form and the stress form give the same equations and the same
# interface condition, the continuity of the shear stress mu du/dy.
layer_length, layer_height, layer_interface = 6.0, 1.0, 0.5
layer_viscosities, layer_force = (1.0, 10.0), 8.0


def layered_profile(y, force):
    """The velocity u(y) of the two layers under the driving force per unit volume ``force``.

    Each layer satisfies mu_i u'' = -g, so u is quadratic in each:

        u_1 = -g y^2 / (2 mu_1) + c_1 y                   (0 < y < d),
        u_2 = -g (H - y)^2 / (2 mu_2) + c_2 (H - y)       (d < y < H),

    which vanish on the walls.  The velocity and the shear stress mu u' are continuous at the
    interface y = d, which gives

        mu_1 c_1 + mu_2 c_2 = g H ,
        d c_1 - (H - d) c_2 = g d^2 / (2 mu_1) - g (H - d)^2 / (2 mu_2) ."""
    mu_1, mu_2 = layer_viscosities
    H, d, g = layer_height, layer_interface, force
    c_1, c_2 = numpy.linalg.solve(
        [[mu_1, mu_2], [d, -(H - d)]],
        [g * H, g * d**2 / (2 * mu_1) - g * (H - d) ** 2 / (2 * mu_2)],
    )
    return numpy.where(
        y <= d, -g * y**2 / (2 * mu_1) + c_1 * y, -g * (H - y) ** 2 / (2 * mu_2) + c_2 * (H - y)
    )


def layered_channel(method, num_y):
    """The channel on square Quad4 elements, with the blocks lower and upper."""
    num_x = int(round(num_y * layer_length / layer_height))
    grid = dualmesh.generate_rectangle_mesh(
        x_min=0.0,
        x_max=layer_length,
        y_min=0.0,
        y_max=layer_height,
        num_x_elements=num_x,
        num_y_elements=num_y,
        element_type="Quad4",
    )
    points = numpy.asarray(grid.points())[:, :2]
    cells = [
        (element_type, nodes, int(points[list(nodes), 1].mean() > layer_interface))
        for element_type, nodes, _block in grid.cells()
    ]
    mesh = dualmesh.mesh_from_arrays(points, cells)
    mesh.set_block_name(0, "lower")
    mesh.set_block_name(1, "upper")
    tolerance = 1e-9 * layer_length
    for name, axis, position in (
        ("left", 0, 0.0),
        ("right", 0, layer_length),
        ("bottom", 1, 0.0),
        ("top", 1, layer_height),
    ):
        mesh.add_sideset_by_predicate(
            name,
            lambda x, y, z, axis=axis, position=position: abs((x, y)[axis] - position) < tolerance,
        )

    problem = dualmesh.Problem(mesh, method=method)
    velocities = ["velocity_x", "velocity_y"]
    for name in velocities + ["pressure"]:
        problem.add_variable(name)
    # The lower fluid comes first, so a stabilization that read the first viscous kernel would
    # take mu_1 in both layers.
    for block, viscosity in zip(("lower", "upper"), layer_viscosities):
        for variable in velocities:
            problem.add_kernel(
                "viscous_stress",
                variable=variable,
                velocities=velocities,
                dynamic_viscosity=viscosity,
                form="laplacian",
                block=[block],
            )
            problem.add_kernel(
                "pressure_gradient",
                variable=variable,
                velocities=velocities,
                pressure="pressure",
                block=[block],
            )
    problem.add_kernel("body_force", variable="velocity_x", value=layer_force)
    problem.add_kernel("mass_conservation", variable="pressure", velocities=velocities)
    problem.add_boundary_condition(
        "mass_flux", ["left", "right", "bottom", "top"], variable="pressure", velocities=velocities
    )
    for variable in velocities:
        problem.add_boundary_condition("Dirichlet", ["bottom", "top"], variable=variable, value=0.0)
    problem.solve(report="none")
    return problem, layer_length / num_x


@pytest.mark.parametrize("method", ["FEM", "DMCDM"])
def test_the_stabilization_of_two_fluids_in_layers_takes_the_viscosity_of_each(method):
    """The pressure stabilization (PSPG in FEM, momentum interpolation in DMCDM) makes the mass
    equation the balance of the flux v - tau r_m, with r_m the strong momentum residual and
    tau = h^2 / (12 mu) for a creeping flow on elements of size h.  On bilinear elements r_m has
    no viscous term, so r_m = (dp/dx - g, 0) in the developed flow away from the ends, which is
    not zero.  Inside the channel the flux tau r_m does not vary along x, so the developed flow
    solves the discrete equations of the nodes away from the ends, and they hold the two-layer
    profile.  At the open ends the mass equations close with v . n alone, so the flow Q through
    the outlet exceeds the flow of the velocity across a section in the middle by

        Q - int u dy = (g - dp/dx) h^2 / 12 (d / mu_1 + (H - d) / mu_2) ,

    the conserved flux tau (g - dp/dx) across that section.  A tau with mu_1 in both layers
    would give (g - dp/dx) h^2 H / (12 mu_1), 1.8 times as much."""
    problem, h = layered_channel(method, num_y=8)
    points = problem.node_coordinates()
    x, y = points[:, 0], points[:, 1]
    u = problem.values("velocity_x")
    pressure = problem.values("pressure")

    def column(position):
        nodes = numpy.flatnonzero(numpy.abs(x - position) < 1e-9)
        return nodes[numpy.argsort(y[nodes])]

    middle = 0.5 * layer_length
    left, right = column(middle), column(middle + h)
    assert len(left) == len(right) == int(round(layer_height / h)) + 1
    gradient = (pressure[right].mean() - pressure[left].mean()) / h
    # The ends leave the developed flow a small pressure gradient against the body force.
    assert abs(gradient) < 2e-2 * layer_force
    force = layer_force - gradient

    # The nodes in the middle hold the two-layer profile of the driving force g - dp/dx.
    exact = layered_profile(y[left], force)
    assert u[left] == pytest.approx(exact, abs=1e-4 * exact.max())
    assert u[right] == pytest.approx(exact, abs=1e-4 * exact.max())

    def section_flow(nodes):
        # The velocity is linear between the nodes of a column, so the trapezoidal rule is exact.
        return float(numpy.sum(0.5 * (u[nodes][1:] + u[nodes][:-1]) * numpy.diff(y[nodes])))

    mu_1, mu_2 = layer_viscosities
    stabilization_flow = (
        force * h**2 / 12 * (layer_interface / mu_1 + (layer_height - layer_interface) / mu_2)
    )
    outlet_flow = problem.boundary_flux("pressure", "right")
    for nodes in (left, right):
        assert outlet_flow - section_flow(nodes) == pytest.approx(stabilization_flow, rel=1e-3)


# ---------------------------------------------------------------------------
# A source flow in a spherical shell
# ---------------------------------------------------------------------------


def spherical_source_flow(method, num_elements, inflow):
    """The largest relative error of the radial velocity at the nodes of the shell 1 <= r <= 2,
    and the largest error of the penalty pressure at the element centroids against the exact
    -mu C / 2 with mu = 1, under the penalty formulation with its defaults, for v_r = ``inflow``
    at r = 1 and a free outlet at r = 2."""
    mesh = dualmesh.generate_line_mesh(start=1.0, end=2.0, num_elements=num_elements)
    problem = dualmesh.Problem(mesh, method=method, coordinates="spherical")
    problem.add_variable("radial_velocity")
    velocities = ["radial_velocity"]
    problem.add_kernel("viscous_stress", variable="radial_velocity", velocities=velocities)
    problem.add_kernel(
        "penalty_incompressibility", variable="radial_velocity", velocities=velocities
    )
    problem.add_property("penalty_pressure", velocities=velocities)
    problem.add_boundary_condition("Dirichlet", "left", variable="radial_velocity", value=inflow)
    problem.solve(report="none")
    r = problem.node_coordinates()[:, 0]
    exact = inflow / r**2
    velocity_error = numpy.max(numpy.abs(problem.values("radial_velocity") - exact) / exact)
    pressure = numpy.asarray(problem.property_at_centroids("pressure"))[:, 0]
    return float(velocity_error), float(numpy.max(numpy.abs(pressure + inflow / 2)))


@pytest.mark.parametrize("method", ["FEM", "DMCDM"])
def test_a_creeping_source_flow_in_a_spherical_shell_keeps_its_mass_flow(method):
    """A source at the centre feeds a creeping radial flow through the shell 1 <= r <= 2, with
    v_r = C at r = 1 and a traction-free outlet at r = 2.  The divergence in spherical
    coordinates is div v = dv_r/dr + 2 v_r / r = (1/r^2) d(r^2 v_r)/dr, so the flow free of
    divergence is v_r = C / r^2, whose mass flow 4 pi r^2 v_r is the same through every sphere.

    The viscous stress tau = 2 mu eps(v), with mu the dynamic viscosity, has the radial component
    tau_rr = 2 mu dv_r/dr = -4 mu C / r^3 and the hoop components tau_tt = tau_pp =
    2 mu v_r / r = 2 mu C / r^3.  The radial momentum balance of a creeping flow is
    -dp/dr + dtau_rr/dr + (2 tau_rr - tau_tt - tau_pp) / r = 0, and here
    dtau_rr/dr = 12 mu C / r^4 and (2 tau_rr - tau_tt - tau_pp) / r = -12 mu C / r^4, so the
    pressure p is uniform.  The free outlet, -p + tau_rr = 0 at r = 2, sets p = -mu C / 2.  The
    penalty formulation takes div v = -p / gamma = mu C / (2 gamma), with gamma = 10^8 Pa s the
    default penalty parameter, which moves v_r by less than 10^-8 C.

    The penalty term makes the discrete divergence vanish at the centroid r_m of each element
    of size h.  The reduced rule of both methods takes it there from the nodal values, with the
    hoop term at r_m, (v_{i+1} - v_i) / h + (v_i + v_{i+1}) / r_m = 0, so
    v_{i+1} / v_i = (r_m - h) / (r_m + h), against the exact ratio (r_i / r_{i+1})^2.  Their
    logarithms differ by -h^3 / (2 r_m^3), and the sum over the elements gives the relative error
    h^2 (1 - 1/r^2) / 4 at the node r, at most 3 h^2 / 16 at the outlet.

    The penalty pressure p_e = -gamma div v at the centroid is the one pressure of each element
    in the discrete equations, and its error is of second order as well.  In DMCDM the momentum
    balance of the control domain of node i gives p_{e+} - p_{e-} = 9 mu C h^3 / r_i^6 between
    its two elements, and that of the outlet node p = -mu C / 2 + O(h^3), so p falls to about
    -mu C (1/2 + 1.74 h^2) at the inlet.  FEM gives errors of 1.5 h^2 and 1.6 h^2 times mu C on 16
    and 32 elements.  The test bounds the error by 3 mu C h^2.

    Without the hoop term 2 v_r / r the divergence is dv_r/dr alone, which the penalty drives to
    zero: v_r = C at every node, four times the exact value at the outlet."""
    sizes = (16, 32)
    (coarse, coarse_pressure), (fine, fine_pressure) = (
        spherical_source_flow(method, n, inflow=1.0) for n in sizes
    )
    assert coarse == pytest.approx(3.0 / 16.0 / sizes[0] ** 2, rel=2e-2)
    assert fine < coarse / 3.0
    assert coarse_pressure < 3.0 / sizes[0] ** 2
    assert fine_pressure < 3.0 / sizes[1] ** 2
    assert fine_pressure < coarse_pressure / 3.0


# ---------------------------------------------------------------------------
# A source flow in a cylindrical annulus
# ---------------------------------------------------------------------------


def axisymmetric_source_flow(method, num_elements, inflow):
    """The largest relative error of the radial velocity at the nodes of the annulus 1 <= r <= 2,
    0 <= z <= 2 h, on square Quad4 elements of size h = 1 / ``num_elements``, and the largest error
    of the penalty pressure at the element centroids against the exact -mu C / 2 with mu = 1.
    The penalty formulation takes its defaults, with v_r = ``inflow`` and v_z = 0 at r = 1, free
    slip (v_z = 0) on the planes z = 0 and z = 2 h, and a free outlet at r = 2."""
    h = 1.0 / num_elements
    mesh = dualmesh.generate_rectangle_mesh(
        x_min=1.0,
        x_max=2.0,
        y_min=0.0,
        y_max=2.0 * h,
        num_x_elements=num_elements,
        num_y_elements=2,
        element_type="Quad4",
    )
    problem = dualmesh.Problem(mesh, method=method, coordinates="axisymmetric")
    velocities = ["radial_velocity", "axial_velocity"]
    for variable in velocities:
        problem.add_variable(variable)
        problem.add_kernel("viscous_stress", variable=variable, velocities=velocities)
        problem.add_kernel("penalty_incompressibility", variable=variable, velocities=velocities)
    problem.add_property("penalty_pressure", velocities=velocities)
    problem.add_boundary_condition("Dirichlet", "left", variable="radial_velocity", value=inflow)
    problem.add_boundary_condition(
        "Dirichlet", ["left", "bottom", "top"], variable="axial_velocity", value=0.0
    )
    problem.solve(report="none")
    r = problem.node_coordinates()[:, 0]
    exact = inflow / r
    velocity_error = numpy.max(numpy.abs(problem.values("radial_velocity") - exact) / exact)
    pressure = numpy.asarray(problem.property_at_centroids("pressure"))[:, 0]
    return float(velocity_error), float(numpy.max(numpy.abs(pressure + inflow / 2)))


@pytest.mark.parametrize("method", ["FEM", "DMCDM"])
def test_a_creeping_source_flow_in_a_cylindrical_annulus_has_a_uniform_pressure(method):
    """A line source on the axis feeds a creeping radial flow through the annulus 1 <= r <= 2,
    with v_r = C at r = 1, v_z = 0, and a traction-free outlet at r = 2.  The divergence in
    axisymmetric coordinates is div v = dv_r/dr + v_r / r + dv_z/dz = (1/r) d(r v_r)/dr + dv_z/dz,
    so the flow free of divergence with v_z = 0 is v_r = C / r.

    The viscous stress tau = 2 mu eps(v) has tau_rr = 2 mu dv_r/dr = -2 mu C / r^2, the hoop
    component tau_tt = 2 mu v_r / r = 2 mu C / r^2, and tau_rz = tau_zz = 0.  The radial momentum
    balance of a creeping flow is -dp/dr + (1/r) d(r tau_rr)/dr - tau_tt / r = 0, and here
    (1/r) d(r tau_rr)/dr = 2 mu C / r^3 = tau_tt / r, so the pressure p is uniform.  The free
    outlet, -p + tau_rr = 0 at r = 2, sets p = -mu C / 2.  The flow satisfies the free slip of the
    planes z = 0 and z = 2 h exactly, since v_z = 0 and the shear stress tau_rz = 0 there, and
    the stress form of viscous_stress has the zero traction of the outlet as its natural
    condition.  So the outlet condition makes the closed form exact.

    The reduced rule makes the divergence vanish at the centroid r_m of each element:
    (v_{i+1} - v_i) / h + (v_i + v_{i+1}) / (2 r_m) = 0, which is r_{i+1} v_{i+1} = r_i v_i, since
    r_m = (r_i + r_{i+1}) / 2.  The nodal velocity is therefore exact for both methods, up to
    the dilatation -p / gamma of the penalty and the rounding of the solve.  The pressure carries
    the error of the viscous term on the linear interpolant of C / r, of second order: in DMCDM
    the momentum balance of the control domains gives p of about -1/2 - 0.30 h^2 at the inlet,
    and in FEM about -1/2 - 0.18 h^2.  The test bounds the error by h^2."""
    sizes = (8, 16)
    (coarse, coarse_pressure), (fine, fine_pressure) = (
        axisymmetric_source_flow(method, n, inflow=1.0) for n in sizes
    )
    assert coarse < 1e-6
    assert fine < 1e-6
    assert coarse_pressure < 1.0 / sizes[0] ** 2
    assert fine_pressure < 1.0 / sizes[1] ** 2
    assert fine_pressure < coarse_pressure / 3.0
