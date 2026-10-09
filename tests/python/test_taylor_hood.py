# SPDX-License-Identifier: LGPL-2.1-or-later
"""The Taylor-Hood element: quadratic velocity, linear pressure.

A mixed velocity-pressure pair gives a unique, convergent pressure only if it
satisfies the discrete inf-sup (Ladyzhenskaya-Babuska-Brezzi) condition,

.. math::

   \\beta_h = \\inf_{q_h} \\sup_{v_h}
   \\frac{(q_h, \\nabla \\cdot v_h)}{|v_h|_1 \\, \\|q_h\\|_0} \\ge \\beta_0 > 0,

with :math:`\\beta_0` independent of the mesh size.  Equal-order pairs fail it,
which is why the ``pressure`` formulation stabilises the mass equation.  The
Taylor-Hood pairs (:math:`P_2`-:math:`P_1` on triangles and tetrahedra,
:math:`Q_2`-:math:`Q_1` on quadrilaterals and hexahedra; Taylor and Hood 1973)
satisfy it, need no stabilisation, and on a smooth solution converge at

* third order for the velocity in :math:`L^2`,
* second order for the velocity in :math:`H^1`,
* second order for the pressure in :math:`L^2`.

The tests check

* those orders by manufactured solutions, on distorted quadrilaterals,
  triangles, serendipity quadrilaterals, axisymmetric meshes that include the
  axis, tetrahedra and hexahedra, for the Stokes and the Navier-Stokes
  equations, with an outlet and in an enclosed cavity with a pinned pressure;
* the inf-sup condition itself, by computing :math:`\\beta_h` on refined meshes
  (the numerical inf-sup test of Chapelle and Bathe 1993): it stays bounded
  for Taylor-Hood and is zero for the unstable equal-order pairs;
* Newton's quadratic convergence, the lid-driven cavity of Ghia et al. (1982),
  the pressure reported at the non-corner nodes, and that every misuse is
  refused with a message that says what to do.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import dualmesh as dm
import numpy as np
import pytest

sympy = pytest.importorskip("sympy")
scipy_linalg = pytest.importorskip("scipy.linalg")
from dualmesh import mms  # noqa: E402

_ROOT = Path(__file__).resolve().parents[2]

# A divergence-free velocity (from a stream function) and a pressure with no
# symmetry.
CARTESIAN = {
    "u": "sin(pi*x)*cos(pi*y)",
    "v": "-cos(pi*x)*sin(pi*y)",
    "pressure": "sin(2*x)*cos(y) + x*y",
}
# Regular on the axis: u_r odd and u_z even in r (Stokes stream function
# psi = r^2 (1 + r^2) cos z).
AXISYMMETRIC = {
    "u": "x*(1 + x**2)*sin(y)",
    "v": "(2 + 4*x**2)*cos(y)",
    "pressure": "cos(x)*sin(y) + x**2*y",
}
# From the stream function psi = sin^2(pi x) sin^2(pi y): the velocity and its
# normal flux vanish on the whole boundary of the unit square, so the flow is
# enclosed and compatible, and the pressure must be pinned.
ENCLOSED = {
    "u": "2*pi*sin(pi*x)**2*sin(pi*y)*cos(pi*y)",
    "v": "-2*pi*sin(pi*x)*cos(pi*x)*sin(pi*y)**2",
    "pressure": "cos(pi*x)*sin(y) + x*y",
}
THREE_DIMENSIONAL = {
    "u": "sin(y+z)",
    "v": "sin(x+z)",
    "w": "sin(x+y)",
    "pressure": "cos(x)*y*z + x",
}


def _distorted_square(n, element_type):
    """The unit square shifted off the origin, with curved interior mesh
    lines, so that no symmetry or superconvergence helps."""
    mesh = dm.generate_rectangle_mesh(0.1, 1.1, 0.2, 1.2, n, n, element_type=element_type)
    mesh.transform_nodes(
        lambda x, y, z: [
            x + 0.6 * (x - 0.1) * (1.1 - x) * (y - 0.2) * (1.2 - y) * np.sin(3 * y),
            y - 0.5 * (x - 0.1) * (1.1 - x) * (y - 0.2) * (1.2 - y),
            0.0,
        ]
    )
    return mesh


def _flow_study(fields, velocities, dimension, coordinates="cartesian", outflow=None, **options):
    """A manufactured incompressible flow with the Taylor-Hood element."""
    study = mms.ManufacturedSolution(
        fields, dimension=dimension, coordinates=coordinates, flux_boundaries=outflow
    )
    study.add_physics(
        "incompressible_flow", "flow", velocities=velocities, formulation="Taylor_Hood", **options
    )
    return study


def _check(result, velocity, velocity_l2=2.8, velocity_h1=1.8, pressure_l2=1.8):
    """Orders from the two finest meshes, against the theoretical 3, 2, 2."""
    assert result.order(velocity, "l2") > velocity_l2, result.errors
    assert result.order(velocity, "h1") > velocity_h1, result.errors
    assert result.order("pressure", "l2") > pressure_l2, result.errors


# ---------------------------------------------------------------------------
# Orders of convergence
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("element_type", ["Quad9", "Tri6", "Quad8"])
@pytest.mark.parametrize("density", [0.0, 1.0])
def test_manufactured_flow_converges_at_the_taylor_hood_orders(element_type, density):
    """Measured on the finest pair: 3.00, 2.00 and 2.00 on Quad9 and Quad8,
    and 3.00, 1.99 and 2.4 (the pressure still approaching 2 from above) on
    Tri6."""
    study = _flow_study(CARTESIAN, ["u", "v"], 2, outflow={"right": (1.0, 0.0)}, density=density)
    result = study.convergence_study(
        lambda n: _distorted_square(n, element_type), [4, 8, 16], method="fem"
    )
    _check(result, "u")


@pytest.mark.parametrize("element_type", ["Quad9", "Tri6"])
def test_axisymmetric_flow_including_the_axis(element_type):
    """The radial momentum equation carries the hoop term, and the axis r = 0
    is part of the domain."""
    study = _flow_study(
        AXISYMMETRIC,
        ["u", "v"],
        2,
        coordinates="axisymmetric",
        outflow={"top": (0.0, 1.0)},
        density=1.0,
    )
    result = study.convergence_study(
        lambda n: dm.generate_rectangle_mesh(0, 1, 0, 1, n, n, element_type=element_type),
        [4, 8, 16],
        method="fem",
    )
    _check(result, "u")


@pytest.mark.parametrize("element_type", ["Quad9", "Tri6"])
def test_enclosed_flow_with_a_pinned_pressure(element_type):
    """Every boundary is a wall, the pressure is fixed at one corner node, and
    the orders are unchanged."""
    study = _flow_study(ENCLOSED, ["u", "v"], 2, density=1.0)
    result = study.convergence_study(
        lambda n: dm.generate_rectangle_mesh(0, 1, 0, 1, n, n, element_type=element_type),
        [4, 8, 16],
        method="fem",
    )
    _check(result, "u")


def test_no_pressure_spike_at_the_pin_when_the_boundary_flux_does_not_vanish():
    """The stabilised equal-order element turns the O(h^2) net boundary flux of
    an interpolated velocity into an O(1) pressure spike at the pin (see
    ``mms.ManufacturedSolution``).  The Taylor-Hood element has no pressure
    Laplacian to do that: the pressure still converges at second order."""
    study = _flow_study(CARTESIAN, ["u", "v"], 2)
    result = study.convergence_study(
        lambda n: dm.generate_rectangle_mesh(0, 1, 0, 1, n, n, element_type="Quad9"),
        [4, 8, 16],
        method="fem",
    )
    _check(result, "u")


@pytest.mark.parametrize("element_type", ["Tet10", "Hex27"])
def test_three_dimensional_flow(element_type):
    """P2-P1 on tetrahedra and Q2-Q1 on hexahedra.  Measured on n = 2, 4:
    2.96, 1.97, 2.25 (Tet10) and 3.00, 1.99, 2.09 (Hex27).  Hex27 needs 89
    derivative slots per element (27 x 3 + 8), within the default of 96."""
    study = _flow_study(
        THREE_DIMENSIONAL, ["u", "v", "w"], 3, outflow={"right": (1.0, 0.0, 0.0)}, density=1.0
    )
    result = study.convergence_study(
        lambda n: dm.generate_box_mesh(0, 1, 0, 1, 0, 1, n, n, n, element_type=element_type),
        [2, 4],
        method="fem",
    )
    _check(result, "u", velocity_l2=2.7)


def test_newton_converges_quadratically():
    """Navier-Stokes at a Reynolds number of about 20, started from half the
    exact solution: the steps satisfy s_{k+1} <= C s_k^2, which needs the
    exact Jacobian of the mixed-order assembly."""
    study = _flow_study(CARTESIAN, ["u", "v"], 2, outflow={"right": (1.0, 0.0)}, density=20.0)
    problem = study.build(_distorted_square(8, "Quad9"), method="fem")
    for variable in ("u", "v", "pressure"):
        problem.set_values(variable, 0.5 * problem.values(variable))
    result = problem.solve(relative_tolerance=1e-13, absolute_tolerance=1e-13)
    steps = [r.step_norm for r in result.history if r.step_norm > 1e-14]
    ratios = [steps[i + 1] / steps[i] ** 2 for i in range(len(steps) - 1) if steps[i] < 0.1]
    assert ratios, steps
    assert max(ratios) < 50.0, steps
    assert result.total_iterations <= 8


# ---------------------------------------------------------------------------
# The inf-sup condition
# ---------------------------------------------------------------------------
def _inf_sup_constant(mesh, pressure_order):
    """The discrete inf-sup constant of the pair, for velocities that vanish on
    the boundary.

    With the vector Laplacian A (the H^1 seminorm of the velocity), the
    discrete divergence B and the pressure mass matrix M,
    beta_h^2 is the smallest non-zero eigenvalue of B A^{-1} B^T q = lambda M q.
    The smallest eigenvalue is zero, for the constant pressure, which an
    enclosed flow does not determine.  One problem supplies all three blocks:
    ``diffusion`` gives A, ``pressure_gradient`` gives B^T (as the coupling of
    the momentum equations to the pressure), and ``reaction`` on the pressure
    gives M.
    """
    problem = dm.Problem(mesh, method="fem")
    for velocity in ("u", "v"):
        problem.add_variable(velocity)
    problem.add_variable("pressure", order=pressure_order)
    for component, velocity in enumerate(("u", "v")):
        problem.add_kernel("diffusion", f"laplacian_{velocity}", variable=velocity)
        problem.add_kernel(
            "pressure_gradient",
            f"gradient_{velocity}",
            variable=velocity,
            component=component,
            pressure="pressure",
        )
        problem.add_boundary_condition(
            "Dirichlet_boundary_condition",
            f"wall_{velocity}",
            variable=velocity,
            boundary=["left", "right", "bottom", "top"],
            value=0.0,
        )
    problem.add_kernel("reaction", "pressure_mass", variable="pressure")
    _, jacobian = problem.linear_system()
    jacobian = jacobian.toarray()
    indices = np.arange(jacobian.shape[0])
    variable = indices % 3

    def prescribed(i):
        row = jacobian[i].copy()
        diagonal, row[i] = row[i], 0.0
        return diagonal == 1.0 and not row.any()

    velocity_dofs = [i for i in indices if variable[i] < 2 and not prescribed(i)]
    pressure_dofs = [i for i in indices if variable[i] == 2 and not prescribed(i)]
    A = jacobian[np.ix_(velocity_dofs, velocity_dofs)]
    # The pressure term of the momentum equations is -(p, div w), so the
    # coupling block is -B^T.
    B = -jacobian[np.ix_(velocity_dofs, pressure_dofs)].T
    M = jacobian[np.ix_(pressure_dofs, pressure_dofs)]
    eigenvalues = np.sort(scipy_linalg.eigh(B @ np.linalg.solve(A, B.T), M, eigvals_only=True))
    assert abs(eigenvalues[0]) < 1e-10  # the constant pressure
    return float(np.sqrt(max(eigenvalues[1], 0.0))), len(pressure_dofs)


@pytest.mark.parametrize("element_type", ["Quad4", "Tri3"])
def test_the_inf_sup_constant_is_bounded_below_for_taylor_hood(element_type):
    """Measured: 0.475, 0.463, 0.455 for Q2-Q1 and 0.368, 0.366, 0.366 for
    P2-P1 at n = 4, 8, 16.  The triangles use the structured 'right' diagonal,
    which leaves two corner triangles with every vertex on the boundary; the
    constant is bounded all the same."""
    betas = []
    for n in (4, 8, 16):
        mesh = dm.generate_rectangle_mesh(0, 1, 0, 1, n, n, element_type=element_type)
        beta, num_pressure = _inf_sup_constant(mesh.second_order(), "first")
        assert num_pressure == (n + 1) ** 2  # the corner nodes only
        betas.append(beta)
    assert min(betas) > 0.3, betas
    assert betas[-1] > 0.95 * betas[0], betas


@pytest.mark.parametrize("promote", [False, True])
def test_equal_order_pairs_fail_the_inf_sup_condition(promote):
    """The control: Q1-Q1 and Q2-Q2 without stabilisation have spurious
    pressure modes (the checkerboard), so the constant is zero."""
    mesh = dm.generate_rectangle_mesh(0, 1, 0, 1, 8, 8)
    if promote:
        mesh = mesh.second_order()
    beta, _ = _inf_sup_constant(mesh, "mesh")
    assert beta < 1e-6


# ---------------------------------------------------------------------------
# Physics and the interface
# ---------------------------------------------------------------------------
def _cavity(n):
    mesh = dm.generate_rectangle_mesh(0, 1, 0, 1, n, n, element_type="Quad9")
    problem = dm.Problem(mesh, method="fem")
    problem.add_physics(
        "incompressible_flow",
        "flow",
        velocities=["u", "v"],
        dynamic_viscosity=1.0,
        density=100.0,
        formulation="Taylor_Hood",
        pressure_pin_point=(0.5, 0.0),
    )
    problem.add_boundary_condition(
        "Dirichlet_boundary_condition",
        "lid",
        variable="u",
        boundary="top",
        value=1.0,
        scale_with_load=True,
    )
    problem.add_boundary_condition(
        "Dirichlet_boundary_condition", "lid_v", variable="v", boundary="top", value=0.0
    )
    for variable in ("u", "v"):
        problem.add_boundary_condition(
            "Dirichlet_boundary_condition",
            f"walls_{variable}",
            variable=variable,
            boundary=["left", "right", "bottom"],
            value=0.0,
        )
    problem.solve()
    return problem


def test_cavity_matches_ghia():
    """Re = 100 on 16 by 16 Quad9 elements (the nodes of a 32 by 32 linear
    mesh): the centreline velocities agree with Ghia, Ghia and Shin (1982)
    within 0.01."""
    spec = importlib.util.spec_from_file_location(
        "compare_cavity", _ROOT / "verification" / "openfoam" / "compare_cavity.py"
    )
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    problem = _cavity(16)
    u = problem.sample(
        "u", np.column_stack([np.full_like(reference.GHIA_Y, 0.5), reference.GHIA_Y])
    )
    v = problem.sample(
        "v", np.column_stack([reference.GHIA_X, np.full_like(reference.GHIA_X, 0.5)])
    )
    assert np.max(np.abs(u - reference.GHIA_U)) < 0.01
    assert np.max(np.abs(v - reference.GHIA_V)) < 0.01


def test_the_pressure_at_non_corner_nodes_is_the_linear_field():
    """Only the corner nodes carry the pressure.  The values reported at the
    other nodes are those of the linear field, so that sampling, output and
    error norms, which interpolate with the element's own (quadratic) shape
    functions, see exactly the linear pressure.  The pin lands on a corner
    node even when the point given is a mid-edge node."""
    mesh = dm.generate_rectangle_mesh(0, 1, 0, 1, 4, 4, element_type="Quad9")
    problem = dm.Problem(mesh, method="fem")
    problem.add_physics(
        "incompressible_flow",
        "flow",
        velocities=["u", "v"],
        formulation="Taylor_Hood",
        pressure_pin_point=(0.125, 0.0),
    )
    problem.add_boundary_condition(
        "Dirichlet_boundary_condition", "lid", variable="u", boundary="top", value=1.0
    )
    for variable, boundaries in (("u", ["left", "right", "bottom"]), ("v", ["top"])):
        problem.add_boundary_condition(
            "Dirichlet_boundary_condition",
            f"wall_{variable}",
            variable=variable,
            boundary=boundaries,
            value=0.0,
        )
    problem.add_boundary_condition(
        "Dirichlet_boundary_condition",
        "walls_v",
        variable="v",
        boundary=["left", "right", "bottom"],
        value=0.0,
    )
    problem.solve()
    assert problem.variable_order("pressure") == "first"
    assert problem.num_active_dofs() == 2 * 9 * 9 + 5 * 5
    pressure = problem.values("pressure")
    points = problem.entity_points()
    # Each element is a unit-size square of the 4 x 4 grid; its Q1 interpolant
    # at an edge midpoint is the mean of the two ends, and at the centre the
    # mean of the four corners.
    for e in range(mesh.num_elements):
        nodes = mesh.element_nodes(e)
        corners = [pressure[k] for k in nodes[:4]]
        for edge in range(4):
            a, b = nodes[edge], nodes[(edge + 1) % 4]
            assert pressure[nodes[4 + edge]] == pytest.approx(0.5 * (pressure[a] + pressure[b]))
        assert pressure[nodes[8]] == pytest.approx(np.mean(corners))
    # The pin: the point (0.125, 0) is a mid-edge node; the corner nodes at x
    # = 0 and 0.25 are equally near, and one of them holds exactly zero.
    pinned = [k for k, p in enumerate(points) if p[1] == 0.0 and p[0] in (0.0, 0.25)]
    assert min(abs(pressure[k]) for k in pinned) == 0.0


# ---------------------------------------------------------------------------
# Misuse is refused, with a message that says what to do
# ---------------------------------------------------------------------------
def test_an_unknown_order_lists_the_valid_ones():
    problem = dm.Problem(dm.generate_rectangle_mesh(0, 1, 0, 1, 2, 2), method="fem")
    with pytest.raises(ValueError, match="'mesh'.*'first'"):
        problem.add_variable("pressure", order="second")


@pytest.mark.parametrize("method", ["dmcdm", "hfvm", "zfvm"])
def test_a_first_order_variable_needs_the_finite_element_method(method):
    mesh = dm.generate_rectangle_mesh(0, 1, 0, 1, 2, 2, element_type="Quad9")
    problem = dm.Problem(mesh, method=method)
    problem.add_variable("temperature", order="first")
    problem.add_kernel("diffusion", "conduction", variable="temperature")
    with pytest.raises(ValueError, match="method='fem'"):
        problem.solve()


def test_taylor_hood_needs_a_quadratic_mesh_and_the_finite_element_method():
    linear = dm.generate_rectangle_mesh(0, 1, 0, 1, 2, 2)
    with pytest.raises(ValueError, match="second_order"):
        dm.Problem(linear, method="fem").add_physics(
            "incompressible_flow", "flow", velocities=["u", "v"], formulation="Taylor_Hood"
        )
    quadratic = linear.second_order()
    with pytest.raises(ValueError, match="method='fem'"):
        dm.Problem(quadratic, method="dmcdm").add_physics(
            "incompressible_flow", "flow", velocities=["u", "v"], formulation="Taylor_Hood"
        )


@pytest.mark.parametrize("option", ["stabilization", "streamline_stabilization"])
def test_taylor_hood_refuses_stabilisation(option):
    mesh = dm.generate_rectangle_mesh(0, 1, 0, 1, 2, 2, element_type="Quad9")
    with pytest.raises(ValueError, match="Taylor-Hood"):
        dm.Problem(mesh, method="fem").add_physics(
            "incompressible_flow",
            "flow",
            velocities=["u", "v"],
            formulation="Taylor_Hood",
            **{option: True},
        )


def test_an_existing_pressure_of_the_wrong_order_is_refused():
    mesh = dm.generate_rectangle_mesh(0, 1, 0, 1, 2, 2, element_type="Quad9")
    problem = dm.Problem(mesh, method="fem")
    problem.add_variable("pressure")
    with pytest.raises(ValueError, match="order='first'"):
        problem.add_physics(
            "incompressible_flow", "flow", velocities=["u", "v"], formulation="Taylor_Hood"
        )


def test_unstabilised_equal_order_is_refused():
    """Without stabilisation an equal-order pressure has spurious modes; the
    kernel says so instead of returning them."""
    mesh = dm.generate_rectangle_mesh(0, 1, 0, 1, 2, 2, element_type="Quad9")
    problem = dm.Problem(mesh, method="fem")
    for variable in ("u", "v", "pressure"):
        problem.add_variable(variable)
    problem.add_kernel(
        "mass_conservation", "mass", variable="pressure", velocities=["u", "v"], stabilization=False
    )
    with pytest.raises(ValueError, match="order 'first'"):
        problem.solve()


def test_an_unknown_formulation_lists_the_valid_ones():
    mesh = dm.generate_rectangle_mesh(0, 1, 0, 1, 2, 2)
    with pytest.raises(ValueError, match="penalty, pressure, Taylor_Hood"):
        dm.Problem(mesh, method="fem").add_physics(
            "incompressible_flow", "flow", velocities=["u", "v"], formulation="taylor-hood"
        )
