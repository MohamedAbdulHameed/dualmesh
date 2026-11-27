# SPDX-License-Identifier: LGPL-2.1-or-later
"""The pressure-velocity formulation of incompressible flow.

The formulation keeps the pressure as an unknown and interpolates it like the
velocity, which is stable only with the residual-based stabilisation of the
mass equation (pressure-stabilising Petrov-Galerkin for the finite element
method and its control volume counterpart in DMCDM).  The tests check

* the orders of convergence, by manufactured solutions, for both methods,
  on distorted quadrilateral, triangular, axisymmetric and three-dimensional
  meshes, for the Stokes and the Navier-Stokes equations;
* the physics, against the lid-driven cavity of Ghia et al. (1982), while
  test_examples.py checks the natural convection benchmark of de Vahl Davis
  (1983) on the example examples/natural_convection, which takes this
  formulation;
* that Newton's method converges quadratically, which it does only with an
  exact Jacobian.

With linear elements the theory of the stabilised equal-order methods
guarantees first order for the velocity in H1 and for the pressure in L2
(Hughes, Franca and Balestra 1986); the velocity is second order in L2.  The
measured pressure orders, 1.3 to 1.7, are above the guaranteed one.
"""

from __future__ import annotations

import dualmesh
import numpy
import pytest

sympy = pytest.importorskip("sympy")
from dualmesh import mms  # noqa: E402
from helpers import Ghia_differences, lid_driven_cavity  # noqa: E402

methods = ["fem", "dmcdm"]

# A divergence-free velocity (from a stream function) and a pressure with no
# symmetry, on a square that is not centred on a symmetry line of either.
cartesian = {
    "velocity_x": "sin(pi*x)*cos(pi*y)",
    "velocity_y": "-cos(pi*x)*sin(pi*y)",
    "pressure": "sin(2*x)*cos(y) + x*y",
}
# An axisymmetric field that is regular on the axis: u_r odd and u_z even in
# r, from the Stokes stream function psi = r^2 (1 + r^2) cos z.
axisymmetric = {
    "velocity_r": "x*(1 + x**2)*sin(y)",
    "velocity_z": "(2 + 4*x**2)*cos(y)",
    "pressure": "cos(x)*sin(y) + x**2*y",
}


def _square(n, element_type="Quad4"):
    mesh = dualmesh.generate_rectangle_mesh(0.1, 1.1, 0.2, 1.2, n, n, element_type=element_type)
    mesh.transform_nodes(
        lambda x, y, z: [
            x + 0.6 * (x - 0.1) * (1.1 - x) * (y - 0.2) * (1.2 - y) * numpy.sin(3 * y),
            y - 0.5 * (x - 0.1) * (1.1 - x) * (y - 0.2) * (1.2 - y),
            0.0,
        ]
    )
    return mesh


def _check(result, velocity):
    assert result.order(velocity, "l2") > 1.8
    assert result.order(velocity, "h1") > 0.9
    assert result.order("pressure", "l2") > 1.2


def _flow_study(fields, dimension, coordinates="cartesian", outflow=None, **options):
    """A manufactured incompressible flow in the stabilised pressure formulation."""
    study = mms.ManufacturedSolution(
        fields, dimension=dimension, coordinates=coordinates, flux_boundaries=outflow
    )
    study.add_physics("fluid_dynamics", formulation="equal_order", **options)
    return study


@pytest.mark.parametrize("method", methods)
@pytest.mark.parametrize("density", [0.0, 1.0])
def test_manufactured_flow_on_distorted_quadrilaterals(method, density):
    """Velocity O(h^2) in L2 and O(h) in H1, pressure better than O(h) in L2.
    The right side is an outlet with the exact traction, so the pressure is
    determined without being pinned (see ``mms.ManufacturedSolution``)."""
    study = _flow_study(cartesian, 2, outflow={"right": (1.0, 0.0)}, density=density)
    _check(study.convergence_study(_square, [8, 16, 32], method=method), "velocity_x")


@pytest.mark.parametrize("method", methods)
def test_manufactured_flow_on_triangles(method):
    """The pressure converges at 1.65 to 1.70 on triangles."""
    study = _flow_study(cartesian, 2, outflow={"right": (1.0, 0.0)}, density=1.0)
    result = study.convergence_study(lambda n: _square(n, "Tri3"), [8, 16, 32], method=method)
    _check(result, "velocity_x")


@pytest.mark.parametrize("method", methods)
def test_manufactured_axisymmetric_flow_including_the_axis(method):
    """The axis r = 0 is part of the domain.  The radial momentum equation
    carries the hoop stress sigma_tt / r = (2 mu u_r / r - p) / r, which needs the
    manufactured forcing in a form that is finite on the axis."""
    study = _flow_study(
        axisymmetric,
        2,
        coordinates="axisymmetric",
        outflow={"top": (0.0, 1.0)},
        density=1.0,
    )
    result = study.convergence_study(
        lambda n: dualmesh.generate_rectangle_mesh(0.0, 1.0, 0.0, 1.0, n, n),
        [8, 16, 32],
        method=method,
    )
    _check(result, "velocity_r")


@pytest.mark.parametrize("method", ["fem"])
def test_manufactured_flow_in_three_dimensions(method):
    """Pre-asymptotic on these meshes, so only the ratio of successive errors
    is checked: at least 2.5 for the velocity and 2 for the pressure (the
    measured ratios are 2.9 to 3.6)."""
    study = _flow_study(
        {
            "velocity_x": "sin(y+z)",
            "velocity_y": "sin(x+z)",
            "velocity_z": "sin(x+y)",
            "pressure": "cos(x)*y*z + x",
        },
        3,
        outflow={"right": (1.0, 0.0, 0.0)},
        density=1.0,
    )
    result = study.convergence_study(
        lambda n: dualmesh.generate_box_mesh(0, 1, 0, 1, 0, 1, n, n, n), [4, 8], method=method
    )
    u = result.errors[("velocity_x", "l2")]
    p = result.errors[("pressure", "l2")]
    assert u[0] / u[1] > 2.5
    assert p[0] / p[1] > 2.0


@pytest.mark.parametrize("method", methods)
def test_newton_converges_quadratically(method):
    """The steps of Newton's method on a Navier-Stokes problem started away
    from the solution satisfy s_{k+1} <= C s_k^2."""
    study = _flow_study(cartesian, 2, outflow={"right": (1.0, 0.0)}, density=20.0)
    problem = study.build(_square(16), method=method)
    for variable in ("velocity_x", "velocity_y", "pressure"):
        problem.set_values(variable, 0.5 * problem.values(variable))
    result = problem.solve()
    steps = [r.step_norm for r in result.history if r.step_norm > 1e-14]
    ratios = [steps[i + 1] / steps[i] ** 2 for i in range(len(steps) - 1) if steps[i] < 0.1]
    assert ratios, steps
    assert max(ratios) < 50.0, steps
    assert result.total_iterations <= 8


@pytest.mark.parametrize("method", methods)
def test_cavity_matches_ghia(method):
    """Re = 100 on a 32 by 32 mesh: the centreline velocities are within 0.015
    of the tabulated values of Ghia, Ghia and Shin (1982); the measured
    differences are 0.002 to 0.014."""
    problem = lid_driven_cavity(
        method, 32, dynamic_viscosity=1.0, density=100.0, formulation="equal_order"
    )
    problem.solve()
    u_difference, v_difference = Ghia_differences(problem)
    assert u_difference < 0.015
    assert v_difference < 0.015


@pytest.mark.parametrize("method", methods)
@pytest.mark.parametrize("coordinates", ["cartesian", "axisymmetric"])
def test_the_laplacian_form_of_the_viscous_term_converges(method, coordinates):
    """viscous_form="laplacian" changes the natural boundary condition to the do-nothing outflow mu dv/dn - p n = 0, which the manufactured traction on the outflow side follows, and keeps the order of every method."""
    fields, outflow = (
        (cartesian, {"right": (1.0, 0.0)})
        if coordinates == "cartesian"
        else (axisymmetric, {"top": (0.0, 1.0)})
    )
    study = _flow_study(
        fields,
        2,
        coordinates=coordinates,
        outflow=outflow,
        density=1.0,
        viscous_form="laplacian",
    )
    meshes = (
        _square
        if coordinates == "cartesian"
        else (lambda n: dualmesh.generate_rectangle_mesh(0.0, 1.0, 0.0, 1.0, n, n))
    )
    velocity = "velocity_x" if coordinates == "cartesian" else "velocity_r"
    _check(study.convergence_study(meshes, [8, 16, 32], method=method, report="none"), velocity)
