# SPDX-License-Identifier: LGPL-2.1-or-later
"""The advection schemes: central, upwind, linear_upwind, limited, and SUPG of the heat.

The one-dimensional problem is u' = u'' / Pe on (0, 1) with u(0) = 0 and u(1) = 1, whose
solution expm1(Pe x) / expm1(Pe) is a boundary layer of thickness 1 / Pe at x = 1.  With 20
elements and Pe = 100 the cell Peclet number is 5: the central scheme oscillates and the upwind
and limited schemes do not.  At Pe = 5 the layer is resolved and the error shows the order.

The skew problem carries a jump of the inflow value across the unit square at 30 degrees to the
mesh, with k = 1e-6: the classic test of Brooks and Hughes (1982) for streamline upwinding.
"""

from __future__ import annotations

import math

import dualmesh as dm
import numpy as np
import pytest
from dualmesh._core import InputError

FACE_SCHEMES = ("upwind", "linear_upwind", "limited")


def boundary_layer(method, scheme, n, pe, mesh=None):
    mesh = mesh if mesh is not None else dm.generate_line_mesh(start=0.0, end=1.0, num_elements=n)
    problem = dm.Problem(mesh, method=method)
    problem.add_variable("u")
    problem.add_kernel("diffusion", variable="u", diffusivity=1.0 / pe)
    problem.add_kernel("advection", variable="u", velocity=[1.0], advection_scheme=scheme)
    problem.add_boundary_condition("Dirichlet", "left", variable="u", value=0.0)
    problem.add_boundary_condition("Dirichlet", "right", variable="u", value=1.0)
    result = problem.solve(report="none")
    x = problem.entity_points()[:, 0]
    return problem.values("u"), np.expm1(pe * x) / np.expm1(pe), result


def order(method, scheme):
    errors = []
    for n in (20, 40, 80):
        u, exact, _ = boundary_layer(method, scheme, n, 5.0)
        errors.append(np.abs(u - exact).max())
    return min(np.log2(errors[i] / errors[i + 1]) for i in range(2))


# ---------------------------------------------------------------------------
# Boundedness and order in one dimension
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("scheme", ["upwind", "limited"])
def test_the_bounded_schemes_do_not_oscillate_at_a_cell_peclet_number_of_5(scheme):
    """The solution stays in [0, 1], as the exact one does, where the central scheme of the
    same "DMCDM" undershoots (the control that the problem can oscillate)."""
    u, _, _ = boundary_layer("DMCDM", scheme, 20, 100.0)
    assert u.min() > -1e-12 and u.max() < 1.0 + 1e-12, (u.min(), u.max())
    central, _, _ = boundary_layer("DMCDM", "central", 20, 100.0)
    assert central.min() < -0.1


@pytest.mark.parametrize(
    "scheme, low, high",
    [
        ("upwind", 0.8, 1.2),
        ("central", 1.8, 2.3),
        ("linear_upwind", 1.6, 2.3),
        ("limited", 1.6, 2.3),
    ],
)
def test_each_scheme_converges_at_its_order(scheme, low, high):
    """Upwind is first order, and central, linear upwind and the limited scheme second order."""
    p = order("DMCDM", scheme)
    assert low < p < high, p


def two_block_line(n):
    """The unit interval in n cells, the block "inner" for x < 0.6 and "outer" beyond."""
    x = np.linspace(0.0, 1.0, n + 1)
    cells = np.array([[i, i + 1] for i in range(n)])
    blocks = [0 if (i + 0.5) / n < 0.6 else 1 for i in range(n)]
    mesh = dm.mesh_from_arrays(x[:, None], cells, element_type="Edge2", blocks=blocks, dimension=1)
    mesh.set_block_name(0, "inner")
    mesh.set_block_name(1, "outer")
    mesh.add_sideset_by_predicate("left", lambda x, y, z: x < 0.5)
    mesh.add_sideset_by_predicate("right", lambda x, y, z: x > 0.5)
    return mesh


def alternating_line(n):
    """The unit interval in cells of size h and 2h in turn."""
    h = np.tile([1.0, 2.0], n // 2)
    x = np.concatenate([[0.0], np.cumsum(h)]) / h.sum()
    cells = np.array([[i, i + 1] for i in range(n)])
    mesh = dm.mesh_from_arrays(x[:, None], cells, element_type="Edge2", blocks=[0] * n, dimension=1)
    mesh.add_sideset_by_predicate("left", lambda x, y, z: x < 0.5)
    mesh.add_sideset_by_predicate("right", lambda x, y, z: x > 0.5)
    return mesh


@pytest.mark.parametrize("scheme", ["linear_upwind", "limited"])
def test_the_second_order_schemes_are_exact_for_a_linear_field_on_uneven_cells(scheme):
    """u = x solves u' - 0.01 u'' = 1.  The least-squares gradient is exact for it, so linear
    upwind is exact, and r = 1, psi = 1, so the limited value is the central one, which is exact."""
    mesh = alternating_line(20)
    problem = dm.Problem(mesh, method="DMCDM")
    problem.add_variable("u")
    problem.add_kernel("diffusion", variable="u", diffusivity=0.01)
    problem.add_kernel("advection", variable="u", velocity=[1.0], advection_scheme=scheme)
    problem.add_kernel("body_force", variable="u", value=1.0)
    problem.add_boundary_condition("Dirichlet", "left", variable="u", value=0.0)
    problem.add_boundary_condition("Dirichlet", "right", variable="u", value=1.0)
    problem.solve(report="none")
    x = problem.entity_points()[:, 0]
    assert np.abs(problem.values("u") - x).max() < 1e-12


@pytest.mark.parametrize("element", ["Quad4", "Tri3"])
@pytest.mark.parametrize("scheme", ["linear_upwind", "limited"])
def test_the_second_order_schemes_are_exact_for_a_linear_field_in_two_dimensions(element, scheme):
    """u = x + 2 y solves v . grad u - 0.01 Laplacian(u) = 2 with v = (1, 0.5).  The node-based
    methods reconstruct the gradient at a node by least squares over the nodes around it, which
    is exact for a linear field."""
    mesh = dm.generate_rectangle_mesh(x=[0, 1], y=[0, 1], num_elements=[8, 8], element_type=element)
    problem = dm.Problem(mesh, method="DMCDM")
    problem.add_variable("u")
    problem.add_kernel("diffusion", variable="u", diffusivity=0.01)
    problem.add_kernel("advection", variable="u", velocity=[1.0, 0.5], advection_scheme=scheme)
    problem.add_kernel("body_force", variable="u", value=2.0)
    problem.add_boundary_condition(
        "Dirichlet", ["left", "bottom", "right", "top"], variable="u", value="x + 2*y"
    )
    problem.solve(report="none")
    x = problem.entity_points()
    assert np.abs(problem.values("u") - x[:, 0] - 2.0 * x[:, 1]).max() < 1e-9


@pytest.mark.parametrize("element", ["Quad4", "Tri3"])
def test_the_jacobian_of_the_node_gradient_is_exact(element):
    """Linear upwind is linear in u, so Newton's "DMCDM" converges in one step when the Jacobian
    holds the derivatives with respect to every node of the reconstructed gradient, including the
    nodes outside the element of the face."""
    mesh = dm.generate_rectangle_mesh(x=[0, 1], y=[0, 1], num_elements=[8, 8], element_type=element)
    problem = dm.Problem(mesh, method="DMCDM")
    problem.add_variable("u")
    problem.add_kernel("diffusion", variable="u", diffusivity=0.01)
    problem.add_kernel(
        "advection", variable="u", velocity=[1.0, 0.5], advection_scheme="linear_upwind"
    )
    problem.add_kernel("body_force", variable="u", value="sin(3*x)*cos(2*y)")
    problem.add_boundary_condition("Dirichlet", ["left", "bottom"], variable="u", value=0.0)
    result = problem.solve(report="none", relative_tolerance=1e-12)
    assert result.total_iterations == 1


def test_the_limited_scheme_converges_in_few_newton_iterations():
    """The limiter is differentiated exactly, so Newton's method converges in a few steps on the
    boundary layer (measured 4).  Of the bounded schemes the limited one has the smaller error
    there (0.11 against 0.16 for upwind); linear upwind is closer (0.048) but undershoots to -0.04,
    and central to -0.43."""
    u, exact, result = boundary_layer("DMCDM", "limited", 20, 100.0)
    assert result.total_iterations <= 6
    assert u.min() > -1e-12 and u.max() < 1.0 + 1e-12
    upwind, _, _ = boundary_layer("DMCDM", "upwind", 20, 100.0)
    assert np.abs(u - exact).max() < 0.8 * np.abs(upwind - exact).max()


@pytest.mark.parametrize("form, beyond", [("conservative", 0.5), ("non_conservative", 1.0)])
@pytest.mark.parametrize("scheme", ["upwind", "limited"])
def test_a_face_between_two_blocks_carries_the_upstream_properties(scheme, form, beyond):
    """The speed is 1 in the block x < 0.6 and 2 beyond, a property of each block, with u = 1 at
    the inflow and almost no diffusion (k = 1e-8).  In the conservative form div(v u) = 0 the flux
    v u is the same everywhere, so u = 1 and then 0.5.  The face between the blocks must take the
    speed of the upstream block: the downstream one would make the flux 2 u there.  In the
    non-conservative form v . grad u = 0, so u = 1 everywhere."""
    n = 20
    problem = dm.Problem(two_block_line(n), method="DMCDM")
    problem.add_variable("u")
    problem.add_property("constant", block=["inner"], properties={"speed": 1.0})
    problem.add_property("constant", block=["outer"], properties={"speed": 2.0})
    problem.add_kernel("diffusion", variable="u", diffusivity=1e-8)
    problem.add_kernel(
        "advection",
        variable="u",
        velocity=["speed"],
        advection_scheme=scheme,
        advection_form=form,
    )
    problem.add_boundary_condition("Dirichlet", "left", variable="u", value=1.0)
    problem.solve(report="none")
    xc = problem.entity_points()[:, 0]
    np.testing.assert_allclose(problem.values("u"), np.where(xc < 0.6, 1.0, beyond), atol=1e-6)


def test_the_non_conservative_form_takes_the_upstream_speed_between_two_blocks():
    """v u' = 1 with the speed 1 in the block x < 0.6 and 2 beyond, and u(0) = 0.  Upwind steps
    by h / v from cell to cell, and the face between the blocks takes the speed of the upstream
    block, so the step there is h (the downstream speed would make it h / 2)."""
    n = 20
    problem = dm.Problem(two_block_line(n), method="DMCDM")
    problem.add_variable("u")
    problem.add_property("constant", block=["inner"], properties={"speed": 1.0})
    problem.add_property("constant", block=["outer"], properties={"speed": 2.0})
    problem.add_kernel("diffusion", variable="u", diffusivity=1e-10)
    problem.add_kernel("advection", variable="u", velocity=["speed"], advection_scheme="upwind")
    problem.add_kernel("body_force", variable="u", value=1.0)
    problem.add_boundary_condition("Dirichlet", "left", variable="u", value=0.0)
    problem.solve(report="none")
    order = np.argsort(problem.entity_points()[:, 0])
    steps = np.diff(problem.values("u")[order])
    h = 1.0 / n
    np.testing.assert_allclose(steps[:12], h, rtol=1e-6)
    # The outflow node has half a control volume, over which upwind takes the difference across a
    # whole element: its step is h / 4, the first-order error of upwind at an outflow.
    np.testing.assert_allclose(steps[12:-1], h / 2, rtol=1e-6)
    np.testing.assert_allclose(steps[-1], h / 4, rtol=1e-6)


# ---------------------------------------------------------------------------
# The outflow boundary in two dimensions
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("form", ["conservative", "non_conservative"])
@pytest.mark.parametrize("element", ["Quad4", "Tri3"])
@pytest.mark.parametrize("scheme", FACE_SCHEMES)
def test_a_uniform_state_leaves_through_the_outflow_unchanged(element, scheme, form):
    """u = 1 on the inflow sides and no condition on the outflow sides: u = 1 everywhere, to
    round-off, so the conservative boundary flux carries the boundary value."""
    mesh = dm.generate_rectangle_mesh(x=[0, 1], y=[0, 1], num_elements=[8, 8], element_type=element)
    problem = dm.Problem(mesh, method="DMCDM")
    problem.add_variable("u")
    problem.add_kernel("diffusion", variable="u", diffusivity=0.01)
    problem.add_kernel(
        "advection",
        variable="u",
        velocity=[1.0, 0.3],
        advection_scheme=scheme,
        advection_form=form,
    )
    problem.add_boundary_condition("Dirichlet", ["left", "bottom"], variable="u", value=1.0)
    problem.solve(report="none")
    assert np.abs(problem.values("u") - 1.0).max() < 1e-12


def skew(method, scheme, n=24, outflow_value=None):
    """The jump of Brooks and Hughes: u = 1 on the left side above y = 0.25, u = 0 below it and on
    the bottom, carried at 30 degrees to the mesh with k = 1e-6.  With outflow_value the right and
    top sides are held at that value, which puts a boundary layer there."""
    c, s = math.cos(math.radians(30.0)), math.sin(math.radians(30.0))
    mesh = dm.generate_rectangle_mesh(x=[0, 1], y=[0, 1], num_elements=[n, n])
    problem = dm.Problem(mesh, method=method)
    flow = problem.add_physics(
        "fluid_dynamics", density=1.0, dynamic_viscosity=1.0, formulation="pressure"
    )
    heat = problem.add_physics("heat_transfer", thermal_conductivity=1e-6, advection_scheme=scheme)
    problem.add_coupling("nonisothermal_flow", heat_transfer=heat, fluid_dynamics=flow)
    flow.add_boundary_condition(
        [f"velocity_x = {c}", f"velocity_y = {s}"], boundary=["left", "bottom", "top"]
    )
    heat.add_boundary_condition("temperature = y > 0.25", boundary="left")
    heat.add_boundary_condition("temperature = 0", boundary="bottom")
    if outflow_value is not None:
        heat.add_boundary_condition(f"temperature = {outflow_value}", boundary=["right", "top"])
    problem.solve(report="none")
    return mesh.points(), problem.values("temperature")


def test_the_node_methods_upwind_a_skew_jump_without_oscillation():
    """The upwind temperature stays in [0, 1], to the tolerance of the nonlinear solve, up to the
    outflow sides, which have no condition."""
    _, T = skew("DMCDM", "upwind")
    assert T.min() > -1e-9 and T.max() < 1.0 + 1e-9, (T.min(), T.max())


def test_the_conservative_flux_through_the_boundary_keeps_a_skew_jump_bounded():
    """The skew jump carried by the constant velocity of advection in the conservative form: the
    flux through the outflow sides must carry the value of the node's own control volume (a value
    interpolated along the side couples the nodes of the side and undershoots)."""
    c, s = math.cos(math.radians(30.0)), math.sin(math.radians(30.0))
    mesh = dm.generate_rectangle_mesh(x=[0, 1], y=[0, 1], num_elements=[24, 24])
    problem = dm.Problem(mesh, method="DMCDM")
    problem.add_variable("u")
    problem.add_kernel("diffusion", variable="u", diffusivity=1e-6)
    problem.add_kernel(
        "advection",
        variable="u",
        velocity=[c, s],
        advection_scheme="upwind",
        advection_form="conservative",
    )
    problem.add_boundary_condition("Dirichlet", "left", variable="u", value="y > 0.25")
    problem.add_boundary_condition("Dirichlet", "bottom", variable="u", value=0.0)
    problem.solve(report="none")
    u = problem.values("u")
    assert u.min() > -1e-9 and u.max() < 1.0 + 1e-9, (u.min(), u.max())


@pytest.mark.parametrize("scheme", FACE_SCHEMES)
def test_the_inertia_of_a_constant_velocity_component_vanishes(scheme):
    """v_x = x (from a Laplace equation with these boundary values, which the methods solve
    exactly) and v_y = 1 at the inflow: (v . grad) v_y = 0, so v_y = 1 everywhere, although
    div v = 1.  The conservative face form would add the source v_y div v."""
    mesh = dm.generate_rectangle_mesh(x=[0.5, 1.5], y=[0, 1], num_elements=[8, 8])
    problem = dm.Problem(mesh, method="DMCDM")
    for name in ("vx", "vy"):
        problem.add_variable(name)
        problem.add_kernel("diffusion", variable=name, diffusivity=0.01)
    problem.add_kernel(
        "convective_inertia",
        variable="vy",
        velocities=["vx", "vy"],
        density=1.0,
        advection_scheme=scheme,
    )
    problem.add_boundary_condition(
        "Dirichlet", ["left", "right", "bottom", "top"], variable="vx", value="x"
    )
    problem.add_boundary_condition("Dirichlet", "left", variable="vy", value=1.0)
    problem.solve(report="none")
    x = problem.entity_points()
    assert np.abs(problem.values("vx") - x[:, 0]).max() < 1e-10
    assert np.abs(problem.values("vy") - 1.0).max() < 1e-10


@pytest.mark.parametrize("scheme", FACE_SCHEMES)
def test_the_heat_a_flow_carries_is_conserved_exactly(scheme):
    """A cavity driven by its lid at Re = 100, hot on the left, cold on the right, adiabatic at
    the top and the bottom.  The face flux carries the velocity that the mass equation conserves,
    whose flux through every control volume sums to zero and through the walls is zero, so the
    heat entering at the hot wall leaves at the cold one to round-off.  With the interpolated
    velocity, whose discrete divergence is not zero, the balance missed the integral of
    T div_h(v)."""
    mesh = dm.generate_rectangle_mesh(x=[0, 1], y=[0, 1], num_elements=[16, 16])
    problem = dm.Problem(mesh, method="DMCDM")
    flow = problem.add_physics(
        "fluid_dynamics",
        density=100.0,
        dynamic_viscosity=1.0,
        formulation="pressure",
    )
    heat = problem.add_physics("heat_transfer", thermal_conductivity=0.1, advection_scheme=scheme)
    problem.add_coupling("nonisothermal_flow", heat_transfer=heat, fluid_dynamics=flow)
    flow.add_boundary_condition(["velocity_x = 1", "velocity_y = 0"], boundary="top")
    flow.add_boundary_condition(
        ["velocity_x = 0", "velocity_y = 0"], boundary=["left", "right", "bottom"]
    )
    heat.add_boundary_condition("temperature = 1", boundary="left")
    heat.add_boundary_condition("temperature = 0", boundary="right")
    problem.solve(report="none")
    hot = problem.total_reaction("temperature", "left")
    cold = problem.total_reaction("temperature", "right")
    assert abs(hot) > 0.1
    assert abs(hot + cold) < 1e-9 * abs(hot), (hot, cold)


def channel(method, scheme, offset):
    """A channel flow at Re = 100 that develops from a uniform inflow between no-slip walls, on
    16 x 16 elements, carrying the temperature offset + y from the inflow with the walls at offset
    and offset + 1.  The discrete velocity is not exactly divergence free."""
    mesh = dm.generate_rectangle_mesh(x=[0, 1], y=[0, 1], num_elements=[16, 16])
    problem = dm.Problem(mesh, method=method)
    flow = problem.add_physics(
        "fluid_dynamics", density=100.0, dynamic_viscosity=1.0, formulation="pressure"
    )
    heat = problem.add_physics("heat_transfer", thermal_conductivity=1.0, advection_scheme=scheme)
    problem.add_coupling("nonisothermal_flow", heat_transfer=heat, fluid_dynamics=flow)
    flow.add_boundary_condition(["velocity_x = 0", "velocity_y = 0"], boundary=["bottom", "top"])
    flow.add_boundary_condition(["velocity_x = 1", "velocity_y = 0"], boundary="left")
    heat.add_boundary_condition(f"temperature = {offset} + y", boundary="left")
    heat.add_boundary_condition(f"temperature = {offset}", boundary="bottom")
    heat.add_boundary_condition(f"temperature = {offset + 1}", boundary="top")
    problem.solve(report="none")
    return problem.values("temperature") - offset


@pytest.mark.parametrize("scheme", FACE_SCHEMES)
def test_the_heat_a_flow_carries_does_not_depend_on_the_temperature_scale(scheme):
    """The same channel in kelvin, 300 K higher: the temperature rises by 300 K everywhere and
    stays in [0, 1] above it, as the equation v . grad T = div(k grad T) / (rho c_p) requires.
    The conservative face flux m T_f would see the discrete divergence of the velocity and give a
    source T div(v), which moved the temperature by 13 K (DMCDM) and took it to
    1.67 even at offset 0."""
    T = channel("DMCDM", scheme, 0.0)
    assert T.min() > -1e-9 and T.max() < 1.0 + 1e-9, (T.min(), T.max())
    np.testing.assert_allclose(channel("DMCDM", scheme, 300.0), T, atol=1e-6)


def test_supg_removes_the_oscillation_of_galerkin_at_an_outflow_layer():
    """With the outflow held at 0, Galerkin's temperature reaches several hundred.  SUPG keeps it
    within 0.12 of [0, 1] away from the outflow layer, and within 0.7 at the node next to the
    outflow corner, where the layer meets the jump (the overshoot that SUPG is known to keep)."""
    x, galerkin = skew("fem", "central", outflow_value=0.0)
    assert galerkin.max() > 100.0
    x, T = skew("fem", "SUPG", outflow_value=0.0)
    inner = (x[:, 0] < 0.9) & (x[:, 1] < 0.9)
    assert T[inner].min() > -0.12 and T[inner].max() < 1.09, (T[inner].min(), T[inner].max())
    assert T.min() > -0.12 and T.max() < 1.7, (T.min(), T.max())


def test_supg_is_written_in_capitals_and_accepted_in_lower_case():
    """An abbreviation keeps its capitals in the input; the lower-case spelling gives the same."""
    heat = dm.Problem(
        dm.generate_rectangle_mesh(x=[0, 1], y=[0, 1], num_elements=[2, 2]), method="fem"
    ).add_physics("heat_transfer", thermal_conductivity=1.0, advection_scheme="supg")
    assert heat.advection_scheme == "SUPG"
    _, lower = skew("fem", "supg", n=8, outflow_value=0.0)
    _, upper = skew("fem", "SUPG", n=8, outflow_value=0.0)
    np.testing.assert_array_equal(lower, upper)


def test_supg_resolves_a_boundary_layer_without_oscillation():
    """The boundary layer of Pe = 100 on a strip one element thick, which makes the problem one
    dimensional, on 40 elements (a cell Peclet number of 1.25).  Galerkin undershoots to -0.11.
    SUPG stays in [0, 1], with a maximum error of 0.0024: the parameter tau has the diffusive
    limit h^2 / (12 kappa) of the optimal one of linear elements.  With the limit h^2 / (4 kappa)
    of Tezduyar's form it overdiffused and the error was 0.143."""
    n = 40
    mesh = dm.generate_rectangle_mesh(x=[0, 1], y=[0, 1.0 / n], num_elements=[n, 1])
    problem = dm.Problem(mesh, method="fem")
    flow = problem.add_physics(
        "fluid_dynamics", density=1.0, dynamic_viscosity=1.0, formulation="pressure"
    )
    heat = problem.add_physics("heat_transfer", thermal_conductivity=0.01, advection_scheme="SUPG")
    problem.add_coupling("nonisothermal_flow", heat_transfer=heat, fluid_dynamics=flow)
    flow.add_boundary_condition(
        ["velocity_x = 1", "velocity_y = 0"], boundary=["left", "bottom", "top"]
    )
    flow.add_boundary_condition("pressure = 0", boundary="right")
    heat.add_boundary_condition("temperature = 0", boundary="left")
    heat.add_boundary_condition("temperature = 1", boundary="right")
    problem.solve(report="none")
    x = mesh.points()[:, 0]
    T = problem.values("temperature")
    assert T.min() > -1e-12 and T.max() < 1.0 + 1e-12
    assert np.abs(T - np.expm1(100.0 * x) / np.expm1(100.0)).max() < 0.005


def test_supg_is_consistent_and_keeps_second_order_on_a_smooth_solution():
    """T = exp(x) sin(y) carried by v = (1, 0.5) with k = 0.01 and the matching heat source
    q = v . grad T.  T is harmonic, so the residual of the equation that SUPG adds is zero for the
    exact solution, also without the conduction term that it leaves out, and the error falls at
    second order as Galerkin's does.  A missing source or a wrong sign would leave a first-order
    term."""
    vx, vy = 1.0, 0.5
    source = f"exp(x)*({vx}*sin(y) + {vy}*cos(y))"
    errors = []
    for n in (8, 16, 32):
        mesh = dm.generate_rectangle_mesh(x=[0, 1], y=[0, 1], num_elements=[n, n])
        problem = dm.Problem(mesh, method="fem")
        flow = problem.add_physics(
            "fluid_dynamics", density=1.0, dynamic_viscosity=1.0, formulation="pressure"
        )
        heat = problem.add_physics(
            "heat_transfer", thermal_conductivity=0.01, heat_source=source, advection_scheme="SUPG"
        )
        problem.add_coupling("nonisothermal_flow", heat_transfer=heat, fluid_dynamics=flow)
        flow.add_boundary_condition(
            [f"velocity_x = {vx}", f"velocity_y = {vy}"], boundary=["left", "bottom", "top"]
        )
        heat.add_boundary_condition(
            "temperature = exp(x)*sin(y)", boundary=["left", "right", "bottom", "top"]
        )
        problem.solve(report="none")
        x = mesh.points()
        exact = np.exp(x[:, 0]) * np.sin(x[:, 1])
        errors.append(np.abs(problem.values("temperature") - exact).max())
    orders = [np.log2(errors[i] / errors[i + 1]) for i in range(2)]
    assert min(orders) > 1.8, errors


def test_the_supg_residual_holds_a_heat_source_added_as_a_kernel():
    """The smooth case above with the heat source added as a kernel of the physics: the residual
    of SUPG is the sum of the sources of every kernel of the temperature, so the order stays two.
    With only the parameter heat_source of the physics in it, the order was 1.2."""
    vx, vy = 1.0, 0.5
    errors = []
    for n in (8, 16, 32):
        mesh = dm.generate_rectangle_mesh(x=[0, 1], y=[0, 1], num_elements=[n, n])
        problem = dm.Problem(mesh, method="fem")
        flow = problem.add_physics(
            "fluid_dynamics", density=1.0, dynamic_viscosity=1.0, formulation="pressure"
        )
        heat = problem.add_physics(
            "heat_transfer", thermal_conductivity=0.01, advection_scheme="SUPG"
        )
        heat.add_kernel("heat_source", heat_source=f"exp(x)*({vx}*sin(y) + {vy}*cos(y))")
        problem.add_coupling("nonisothermal_flow", heat_transfer=heat, fluid_dynamics=flow)
        flow.add_boundary_condition(
            [f"velocity_x = {vx}", f"velocity_y = {vy}"], boundary=["left", "bottom", "top"]
        )
        heat.add_boundary_condition(
            "temperature = exp(x)*sin(y)", boundary=["left", "right", "bottom", "top"]
        )
        problem.solve(report="none")
        x = mesh.points()
        errors.append(
            np.abs(problem.values("temperature") - np.exp(x[:, 0]) * np.sin(x[:, 1])).max()
        )
    orders = [np.log2(errors[i] / errors[i + 1]) for i in range(2)]
    assert min(orders) > 1.8, errors


def supg_without_a_flow():
    """The heat kernels with SUPG, carried by a velocity that Dirichlet conditions fix, with no
    kernel of a flow (whose own stabilisation would also hold a time derivative)."""
    mesh = dm.generate_rectangle_mesh(x=[0, 1], y=[0, 1], num_elements=[2, 2])
    problem = dm.Problem(mesh, method="fem")
    for name in ("u", "v", "temperature"):
        problem.add_variable(name)
        problem.add_kernel("diffusion", variable=name, diffusivity=0.01)
    walls = ["left", "right", "bottom", "top"]
    problem.add_boundary_condition("Dirichlet", walls, variable="u", value=1.0)
    problem.add_boundary_condition("Dirichlet", walls, variable="v", value=0.0)
    problem.add_boundary_condition("Dirichlet", "left", variable="temperature", value=0.0)
    problem.add_kernel("heat_convection", variable="temperature", velocities=["u", "v"])
    problem.add_kernel("heat_conduction_time_derivative", variable="temperature")
    problem.add_kernel(
        "heat_streamline_stabilization",
        variable="temperature",
        velocities=["u", "v"],
        thermal_conductivity="0.01",
    )
    return problem


def test_an_explicit_step_with_supg_is_refused():
    """The explicit step gives the time derivative inside the residual of SUPG no weight."""
    problem = supg_without_a_flow()
    with pytest.raises(InputError, match="heat_streamline_stabilization.*implicitness 0"):
        problem.solve_transient(end_time=0.1, time_step=0.05, implicitness=0.0, report="none")


def test_the_modes_of_a_problem_with_supg_are_refused():
    """The mass operator of the modes has the time kernels only, not the time derivative inside the
    residual of SUPG."""
    problem = supg_without_a_flow()
    with pytest.raises(InputError, match="no mass operator for 'heat_streamline_stabilization"):
        problem.solve_eigenvalue(report="none")


def test_supg_with_an_advection_written_as_a_flux_is_refused():
    """The residual of SUPG is a sum of sources, which cannot hold a flux."""
    problem = supg_without_a_flow()
    problem.add_kernel(
        "advection", variable="temperature", velocity=[1.0, 0.0], advection_form="conservative"
    )
    with pytest.raises(InputError, match="written as a flux"):
        problem.solve(report="none")


def test_the_modes_of_a_stabilised_stokes_flow_are_not_refused():
    """A Stokes flow has no time derivative in the residual of its pressure stabilisation, so its
    modes have a complete mass operator: the onset of convection at an infinite Prandtl number."""
    mesh = dm.generate_rectangle_mesh(x=[0, 1], y=[0, 1], num_elements=[4, 4])
    problem = dm.Problem(mesh, method="fem")
    flow = problem.add_physics("fluid_dynamics", dynamic_viscosity=1.0, formulation="pressure")
    flow.add_boundary_condition(
        ["velocity_x = 0", "velocity_y = 0"], boundary=["left", "right", "bottom", "top"]
    )
    flow.add_boundary_condition("pressure = 0", boundary="bottom")
    heat = problem.add_physics(
        "heat_transfer", thermal_conductivity=1.0, density=1.0, specific_heat=1.0
    )
    heat.add_boundary_condition("temperature = 0", boundary=["left", "right", "bottom", "top"])
    assert problem.solve_eigenvalue(report="none").eigenvalues[0] > 0.0


def test_the_supg_residual_holds_a_source_added_after_a_solve():
    """A heat source added to a problem that was solved before: the kernel comes after the SUPG
    kernel in the list, so it is matched by its variable's name."""
    vx, vy = 1.0, 0.5
    errors = []
    for n in (8, 16, 32):
        mesh = dm.generate_rectangle_mesh(x=[0, 1], y=[0, 1], num_elements=[n, n])
        problem = dm.Problem(mesh, method="fem")
        flow = problem.add_physics(
            "fluid_dynamics", density=1.0, dynamic_viscosity=1.0, formulation="pressure"
        )
        heat = problem.add_physics(
            "heat_transfer", thermal_conductivity=0.01, advection_scheme="SUPG"
        )
        problem.add_coupling("nonisothermal_flow", heat_transfer=heat, fluid_dynamics=flow)
        flow.add_boundary_condition(
            [f"velocity_x = {vx}", f"velocity_y = {vy}"], boundary=["left", "bottom", "top"]
        )
        heat.add_boundary_condition(
            "temperature = exp(x)*sin(y)", boundary=["left", "right", "bottom", "top"]
        )
        problem.solve(report="none")
        problem.add_kernel(
            "body_force",
            variable="temperature",
            value=f"exp(x)*({vx}*sin(y) + {vy}*cos(y))",
        )
        problem.solve(report="none")
        x = mesh.points()
        errors.append(
            np.abs(problem.values("temperature") - np.exp(x[:, 0]) * np.sin(x[:, 1])).max()
        )
    orders = [np.log2(errors[i] / errors[i + 1]) for i in range(2)]
    assert min(orders) > 1.8, errors


@pytest.mark.parametrize("implicitness", [1.0, 0.5])
def test_supg_is_consistent_in_time_with_backward_euler_and_crank_nicolson(implicitness):
    """T = exp(x) sin(y) (1 + t), whose time derivative varies in space, with the matching source.
    T is linear in time, so both steps are exact in time, and T is harmonic, so the residual that
    SUPG adds is zero for the exact solution.  The theta method weights a steady kernel by theta
    and evaluates it again at the old time, so the time derivative inside the residual must enter
    with weight 1 in total: with weight theta the error stalls at first order."""
    vx, vy = 1.0, 0.5
    errors = []
    for n in (8, 16, 32):
        mesh = dm.generate_rectangle_mesh(x=[0, 1], y=[0, 1], num_elements=[n, n])
        problem = dm.Problem(mesh, method="fem")
        flow = problem.add_physics(
            "fluid_dynamics",
            density=1.0,
            dynamic_viscosity=1.0,
            formulation="pressure",
            initial_condition={"velocity": [vx, vy]},
        )
        heat = problem.add_physics(
            "heat_transfer",
            thermal_conductivity=0.01,
            density=1.0,
            specific_heat=1.0,
            heat_source=f"exp(x)*sin(y) + (1 + t)*exp(x)*({vx}*sin(y) + {vy}*cos(y))",
            initial_condition="exp(x)*sin(y)",
            advection_scheme="SUPG",
        )
        problem.add_coupling("nonisothermal_flow", heat_transfer=heat, fluid_dynamics=flow)
        flow.add_boundary_condition(
            [f"velocity_x = {vx}", f"velocity_y = {vy}"], boundary=["left", "bottom", "top"]
        )
        heat.add_boundary_condition(
            "temperature = exp(x)*sin(y)*(1 + t)", boundary=["left", "right", "bottom", "top"]
        )
        problem.solve_transient(
            end_time=0.2, time_step=0.05, implicitness=implicitness, report="none"
        )
        x = mesh.points()
        exact = 1.2 * np.exp(x[:, 0]) * np.sin(x[:, 1])
        errors.append(np.abs(problem.values("temperature") - exact).max())
    orders = [np.log2(errors[i] / errors[i + 1]) for i in range(2)]
    assert min(orders) > 1.8 and errors[-1] < 1e-5, errors


# ---------------------------------------------------------------------------
# The momentum equations: Kovasznay flow
# ---------------------------------------------------------------------------
RE = 40.0
LAMBDA = RE / 2 - math.sqrt(RE**2 / 4 + 4 * math.pi**2)
KOVASZNAY = {
    "velocity_x": f"1 - exp({LAMBDA!r}*x)*cos(2*pi*y)",
    "velocity_y": f"{LAMBDA / (2 * math.pi)!r}*exp({LAMBDA!r}*x)*sin(2*pi*y)",
    "pressure": f"(1 - exp({2 * LAMBDA!r}*x))/2",
}


def kovasznay_errors(method):
    from dualmesh import mms

    study = mms.ManufacturedSolution(KOVASZNAY, dimension=2, flux_boundaries={"right": (1.0, 0.0)})
    study.add_physics(
        "fluid_dynamics", density=1.0, dynamic_viscosity=1.0 / RE, formulation="pressure"
    )
    return study.convergence_study(
        lambda n: dm.generate_rectangle_mesh(-0.5, 1.0, -0.5, 1.5, n, round(4 * n / 3)),
        [16, 32],
        method=method,
        report="none",
    )


@pytest.mark.parametrize("method", ["FEM", "DMCDM"])
def test_the_flow_of_every_method_converges_at_second_order_on_kovasznay_flow(method):
    """The exact Navier-Stokes solution of Kovasznay (1948) at Re = 40, with the exact velocity on
    three sides and the exact traction on the outflow side, with the streamline stabilisation."""
    result = kovasznay_errors(method)
    assert 1.8 < result.order("velocity_x", last=2) < 2.3, result.summary()


@pytest.mark.parametrize("method", ["FEM", "DMCDM"])
def test_the_heat_a_flow_carries_takes_the_streamline_stabilisation_by_default(method):
    """The default of advection_scheme is the streamline stabilisation, which measured the
    closest on the boundary layer, the skew jump and natural convection."""
    _, by_default = skew(method, None, n=8, outflow_value=0.0)
    _, given = skew(method, "SUPG", n=8, outflow_value=0.0)
    np.testing.assert_allclose(by_default, given, atol=1e-12)


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "method, scheme, words",
    [
        ("DMCDM", "SUPG", "central, upwind, linear_upwind or limited"),
        ("fem", "upwind", "heat_streamline_stabilization"),
        ("fem", "SUPG", "heat_streamline_stabilization"),
        ("fem", "supg", "heat_streamline_stabilization"),
        ("fem", "quick", "unknown advection scheme"),
    ],
)
def test_a_scheme_that_the_method_does_not_have_is_refused(method, scheme, words):
    mesh = dm.generate_line_mesh(start=0.0, end=1.0, num_elements=4)
    problem = dm.Problem(mesh, method=method)
    problem.add_variable("u")
    problem.add_kernel("diffusion", variable="u", diffusivity=1.0)
    problem.add_kernel("advection", variable="u", velocity=[1.0], advection_scheme=scheme)
    problem.add_boundary_condition("Dirichlet", ["left", "right"], variable="u", value=0.0)
    with pytest.raises(InputError, match=words):
        problem.solve(report="none")


def test_the_node_gradient_schemes_are_refused_on_a_distributed_mesh():
    """A part of a distributed mesh does not hold the nodes around its interface nodes, from
    which the gradient at a node is reconstructed."""
    mesh = dm.generate_line_mesh(start=0.0, end=1.0, num_elements=4)
    problem = dm.Problem(mesh, method="DMCDM", distributed=True)
    problem.add_variable("u")
    problem.add_kernel("diffusion", variable="u", diffusivity=1.0)
    problem.add_kernel("advection", variable="u", velocity=[1.0], advection_scheme="limited")
    problem.add_boundary_condition("Dirichlet", ["left", "right"], variable="u", value=0.0)
    with pytest.raises(InputError, match="distributed"):
        problem.solve(report="none")


def test_an_error_of_an_object_names_it_once():
    mesh = dm.generate_line_mesh(start=0.0, end=1.0, num_elements=4)
    problem = dm.Problem(mesh, method="DMCDM")
    problem.add_variable("u")
    with pytest.raises(InputError, match="advection_form") as error:
        problem.add_kernel("advection", variable="u", velocity=[1.0], advection_form="both")
    assert str(error.value).count("advection_0") == 1, str(error.value)


def test_a_heat_scheme_without_a_flow_is_refused():
    mesh = dm.generate_rectangle_mesh(x=[0, 1], y=[0, 1], num_elements=[2, 2])
    problem = dm.Problem(mesh, method="DMCDM")
    heat = problem.add_physics("heat_transfer", thermal_conductivity=1.0, advection_scheme="upwind")
    heat.add_boundary_condition("temperature = 0", boundary="left")
    with pytest.raises(InputError, match="no nonisothermal_flow coupling"):
        problem.solve(report="none")
