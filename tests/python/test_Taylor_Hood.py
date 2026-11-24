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

import dualmesh
import numpy
import pytest

sympy = pytest.importorskip("sympy")
scipy_linalg = pytest.importorskip("scipy.linalg")
from dualmesh import mms  # noqa: E402

_root = Path(__file__).resolve().parents[2]

# A divergence-free velocity (from a stream function) and a pressure with no
# symmetry.
cartesian = {
    "velocity_x": "sin(pi*x)*cos(pi*y)",
    "velocity_y": "-cos(pi*x)*sin(pi*y)",
    "pressure": "sin(2*x)*cos(y) + x*y",
}
# Regular on the axis: u_r odd and u_z even in r (Stokes stream function
# psi = r^2 (1 + r^2) cos z).
axisymmetric = {
    "velocity_r": "x*(1 + x**2)*sin(y)",
    "velocity_z": "(2 + 4*x**2)*cos(y)",
    "pressure": "cos(x)*sin(y) + x**2*y",
}
# From the stream function psi = sin^2(pi x) sin^2(pi y): the velocity and its
# normal flux vanish on the whole boundary of the unit square, so the flow is
# enclosed and compatible, and the pressure must be pinned.
enclosed = {
    "velocity_x": "2*pi*sin(pi*x)**2*sin(pi*y)*cos(pi*y)",
    "velocity_y": "-2*pi*sin(pi*x)*cos(pi*x)*sin(pi*y)**2",
    "pressure": "cos(pi*x)*sin(y) + x*y",
}
three_dimensional = {
    "velocity_x": "sin(y+z)",
    "velocity_y": "sin(x+z)",
    "velocity_z": "sin(x+y)",
    "pressure": "cos(x)*y*z + x",
}


def _distorted_square(n, element_type):
    """The unit square shifted off the origin, with curved interior mesh
    lines, so that no symmetry or superconvergence helps."""
    mesh = dualmesh.generate_rectangle_mesh(0.1, 1.1, 0.2, 1.2, n, n, element_type=element_type)
    mesh.transform_nodes(
        lambda x, y, z: [
            x + 0.6 * (x - 0.1) * (1.1 - x) * (y - 0.2) * (1.2 - y) * numpy.sin(3 * y),
            y - 0.5 * (x - 0.1) * (1.1 - x) * (y - 0.2) * (1.2 - y),
            0.0,
        ]
    )
    return mesh


def _flow_study(fields, dimension, coordinates="cartesian", outflow=None, **options):
    """A manufactured incompressible flow with the Taylor-Hood element."""
    study = mms.ManufacturedSolution(
        fields, dimension=dimension, coordinates=coordinates, flux_boundaries=outflow
    )
    study.add_physics("fluid_dynamics", formulation="Taylor-Hood", **options)
    return study


def _check(result, velocity, velocity_L2=2.8, velocity_H1=1.8, pressure_L2=1.8):
    """Orders from the two finest meshes, against the theoretical 3, 2, 2."""
    assert result.order(velocity, "l2") > velocity_L2, result.errors
    assert result.order(velocity, "h1") > velocity_H1, result.errors
    assert result.order("pressure", "l2") > pressure_L2, result.errors


# ---------------------------------------------------------------------------
# Orders of convergence
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("element_type", ["Quad9", "Tri6", "Quad8"])
@pytest.mark.parametrize("density", [0.0, 1.0])
def test_manufactured_flow_converges_at_the_Taylor_hood_orders(element_type, density):
    """Measured on the finest pair: 3.00, 2.00 and 2.00 on Quad9 and Quad8,
    and 3.00, 1.99 and 2.4 (the pressure still approaching 2 from above) on
    Tri6."""
    study = _flow_study(cartesian, 2, outflow={"right": (1.0, 0.0)}, density=density)
    result = study.convergence_study(
        lambda n: _distorted_square(n, element_type), [4, 8, 16], method="fem"
    )
    _check(result, "velocity_x")


@pytest.mark.parametrize("element_type", ["Quad9", "Tri6"])
def test_axisymmetric_flow_including_the_axis(element_type):
    """The radial momentum equation carries the hoop term, and the axis r = 0
    is part of the domain."""
    study = _flow_study(
        axisymmetric,
        2,
        coordinates="axisymmetric",
        outflow={"top": (0.0, 1.0)},
        density=1.0,
    )
    result = study.convergence_study(
        lambda n: dualmesh.generate_rectangle_mesh(0, 1, 0, 1, n, n, element_type=element_type),
        [4, 8, 16],
        method="fem",
    )
    _check(result, "velocity_r")


@pytest.mark.parametrize("element_type", ["Quad9", "Tri6"])
def test_enclosed_flow_with_a_pinned_pressure(element_type):
    """Every boundary is a wall, the pressure is fixed at one corner node, and
    the orders are unchanged."""
    study = _flow_study(enclosed, 2, density=1.0)
    result = study.convergence_study(
        lambda n: dualmesh.generate_rectangle_mesh(0, 1, 0, 1, n, n, element_type=element_type),
        [4, 8, 16],
        method="fem",
    )
    _check(result, "velocity_x")


def test_no_pressure_spike_at_the_pin_when_the_boundary_flux_does_not_vanish():
    """The stabilised equal-order element turns the O(h^2) net boundary flux of
    an interpolated velocity into an O(1) pressure spike at the pin (see
    ``mms.ManufacturedSolution``).  The Taylor-Hood element has no pressure
    Laplacian to do that: the pressure still converges at second order."""
    study = _flow_study(cartesian, 2)
    result = study.convergence_study(
        lambda n: dualmesh.generate_rectangle_mesh(0, 1, 0, 1, n, n, element_type="Quad9"),
        [4, 8, 16],
        method="fem",
    )
    _check(result, "velocity_x")


@pytest.mark.parametrize("element_type", ["Tet10", "Hex27"])
def test_three_dimensional_flow(element_type):
    """P2-P1 on tetrahedra and Q2-Q1 on hexahedra.  Measured on n = 2, 4:
    2.96, 1.97, 2.25 (Tet10) and 3.00, 1.99, 2.09 (Hex27).  Hex27 needs 89
    derivative slots per element (27 x 3 + 8), within the default of 96."""
    study = _flow_study(three_dimensional, 3, outflow={"right": (1.0, 0.0, 0.0)}, density=1.0)
    result = study.convergence_study(
        lambda n: dualmesh.generate_box_mesh(0, 1, 0, 1, 0, 1, n, n, n, element_type=element_type),
        [2, 4],
        method="fem",
    )
    _check(result, "velocity_x", velocity_L2=2.7)


def test_newton_converges_quadratically():
    """Navier-Stokes at a Reynolds number of about 20, started from half the
    exact solution: the steps satisfy s_{k+1} <= C s_k^2, which needs the
    exact Jacobian of the mixed-order assembly."""
    study = _flow_study(cartesian, 2, outflow={"right": (1.0, 0.0)}, density=20.0)
    problem = study.build(_distorted_square(8, "Quad9"), method="fem")
    for variable in ("velocity_x", "velocity_y", "pressure"):
        problem.set_values(variable, 0.5 * problem.values(variable))
    result = problem.solve()
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
    problem = dualmesh.Problem(mesh, method="fem")
    for velocity in ("velocity_x", "velocity_y"):
        problem.add_variable(velocity)
    problem.add_variable("pressure", order=pressure_order)
    for velocity in ("velocity_x", "velocity_y"):
        problem.add_kernel("diffusion", variable=velocity)
        problem.add_kernel(
            "pressure_gradient",
            variable=velocity,
            velocities=["velocity_x", "velocity_y"],
            pressure="pressure",
        )
        problem.add_boundary_condition(
            "Dirichlet",
            ["left", "right", "bottom", "top"],
            variable=velocity,
            value=0.0,
        )
    problem.add_kernel("reaction", variable="pressure")
    _, Jacobian = problem.linear_system()
    Jacobian = Jacobian.toarray()
    indices = numpy.arange(Jacobian.shape[0])
    variable = indices % 3

    def prescribed(i):
        """A prescribed unknown has a row that holds its diagonal alone."""
        row = Jacobian[i].copy()
        diagonal, row[i] = row[i], 0.0
        return diagonal != 0.0 and not row.any()

    velocity_degrees_of_freedom = [i for i in indices if variable[i] < 2 and not prescribed(i)]
    pressure_degrees_of_freedom = [i for i in indices if variable[i] == 2 and not prescribed(i)]
    A = Jacobian[numpy.ix_(velocity_degrees_of_freedom, velocity_degrees_of_freedom)]
    # The pressure term of the momentum equations is -(p, div w), so the
    # coupling block is -B^T.
    B = -Jacobian[numpy.ix_(velocity_degrees_of_freedom, pressure_degrees_of_freedom)].T
    M = Jacobian[numpy.ix_(pressure_degrees_of_freedom, pressure_degrees_of_freedom)]
    eigenvalues = numpy.sort(
        scipy_linalg.eigh(B @ numpy.linalg.solve(A, B.T), M, eigvals_only=True)
    )
    assert abs(eigenvalues[0]) < 1e-10  # The constant pressure
    return float(numpy.sqrt(max(eigenvalues[1], 0.0))), len(pressure_degrees_of_freedom)


@pytest.mark.parametrize("element_type", ["Quad4", "Tri3"])
def test_the_inf_sup_constant_is_bounded_below_for_Taylor_hood(element_type):
    """Measured: 0.475, 0.463, 0.455 for Q2-Q1 and 0.368, 0.366, 0.366 for
    P2-P1 at n = 4, 8, 16.  The triangles use the structured 'right' diagonal,
    which leaves two corner triangles with every vertex on the boundary; the
    constant is bounded all the same."""
    betas = []
    for n in (4, 8, 16):
        mesh = dualmesh.generate_rectangle_mesh(0, 1, 0, 1, n, n, element_type=element_type)
        beta, number_pressure = _inf_sup_constant(mesh.second_order(), "first")
        assert number_pressure == (n + 1) ** 2  # The corner nodes only
        betas.append(beta)
    assert min(betas) > 0.3, betas
    assert betas[-1] > 0.95 * betas[0], betas


@pytest.mark.parametrize("promote", [False, True])
def test_equal_order_pairs_fail_the_inf_sup_condition(promote):
    """The control: Q1-Q1 and Q2-Q2 without stabilisation have spurious
    pressure modes (the checkerboard), so the constant is zero."""
    mesh = dualmesh.generate_rectangle_mesh(0, 1, 0, 1, 8, 8)
    if promote:
        mesh = mesh.second_order()
    beta, _ = _inf_sup_constant(mesh, "mesh")
    assert beta < 1e-6


# ---------------------------------------------------------------------------
# Physics and the interface
# ---------------------------------------------------------------------------
def _cavity(n):
    mesh = dualmesh.generate_rectangle_mesh(0, 1, 0, 1, n, n, element_type="Quad9")
    problem = dualmesh.Problem(mesh, method="fem")
    problem.add_physics(
        "fluid_dynamics",
        dynamic_viscosity=1.0,
        density=100.0,
        formulation="Taylor-Hood",
    )
    problem.add_boundary_condition(
        "Dirichlet",
        "top",
        variable="velocity_x",
        value=1.0,
    )
    problem.add_boundary_condition("Dirichlet", "top", variable="velocity_y", value=0.0)
    for variable in ("velocity_x", "velocity_y"):
        problem.add_boundary_condition(
            "Dirichlet",
            ["left", "right", "bottom"],
            variable=variable,
            value=0.0,
        )
    problem.solve()
    return problem


def test_cavity_matches_ghia():
    """Re = 100 on 16 by 16 Quad9 elements (the nodes of a 32 by 32 linear
    mesh): the centreline velocities agree with Ghia, Ghia and Shin (1982)
    within 0.01."""
    spec = importlib.util.spec_from_file_location(
        "compare_cavity", _root / "verification" / "openfoam" / "compare_cavity.py"
    )
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    problem = _cavity(16)
    u = problem.sample(
        "velocity_x", numpy.column_stack([numpy.full_like(reference.ghia_y, 0.5), reference.ghia_y])
    )
    v = problem.sample(
        "velocity_y", numpy.column_stack([reference.ghia_x, numpy.full_like(reference.ghia_x, 0.5)])
    )
    assert numpy.max(numpy.abs(u - reference.ghia_u)) < 0.01
    assert numpy.max(numpy.abs(v - reference.ghia_v)) < 0.01


def test_Taylor_hood_needs_a_quadratic_mesh_and_the_finite_element_method():
    linear = dualmesh.generate_rectangle_mesh(0, 1, 0, 1, 2, 2)
    with pytest.raises(ValueError, match="second_order"):
        dualmesh.Problem(linear, method="fem").add_physics(
            "fluid_dynamics", formulation="Taylor-Hood"
        )
    quadratic = linear.second_order()
    with pytest.raises(ValueError, match="method='FEM'"):
        dualmesh.Problem(quadratic, method="dmcdm").add_physics(
            "fluid_dynamics", formulation="Taylor-Hood"
        )


def test_unstabilised_equal_order_is_refused():
    """Without stabilisation an equal-order pressure has spurious modes; the
    kernel says so instead of returning them."""
    mesh = dualmesh.generate_rectangle_mesh(0, 1, 0, 1, 2, 2, element_type="Quad9")
    problem = dualmesh.Problem(mesh, method="fem")
    for variable in ("velocity_x", "velocity_y", "pressure"):
        problem.add_variable(variable)
    problem.add_kernel(
        "mass_conservation",
        variable="pressure",
        velocities=["velocity_x", "velocity_y"],
        stabilization=False,
    )
    with pytest.raises(ValueError, match="order 'first'"):
        problem.solve()
