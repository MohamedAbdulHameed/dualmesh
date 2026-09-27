# SPDX-License-Identifier: LGPL-2.1-or-later
"""The pressure-velocity formulation of incompressible flow.

The formulation keeps the pressure as an unknown and interpolates it like the
velocity, which is stable only with the residual-based stabilisation of the
mass equation (pressure-stabilising Petrov-Galerkin for the finite element
method and its control volume counterparts).  The tests check

* the orders of convergence, by manufactured solutions, for all four methods,
  on distorted quadrilateral, triangular, axisymmetric and three-dimensional
  meshes, for the Stokes and the Navier-Stokes equations;
* the physics, against the lid-driven cavity of Ghia et al. (1982) and the
  natural convection benchmark of de Vahl Davis (1983);
* that Newton's method converges quadratically, which it does only with an
  exact Jacobian, including the Jacobian of the cell sources of the
  cell-centred method.

With linear elements the theory of the stabilised equal-order methods
guarantees first order for the velocity in H1 and for the pressure in L2
(Hughes, Franca and Balestra 1986); the velocity is second order in L2.  The
measured pressure orders, 1.3 to 1.7, are above the guaranteed one.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import dualmesh as dm
import numpy as np
import pytest

sympy = pytest.importorskip("sympy")
from dualmesh import mms  # noqa: E402

METHODS = ["fem", "dmcdm", "hfvm", "zfvm"]
_ROOT = Path(__file__).resolve().parents[2]

# A divergence-free velocity (from a stream function) and a pressure with no
# symmetry, on a square that is not centred on a symmetry line of either.
CARTESIAN = {
    "u": "sin(pi*x)*cos(pi*y)",
    "v": "-cos(pi*x)*sin(pi*y)",
    "pressure": "sin(2*x)*cos(y) + x*y",
}
# An axisymmetric field that is regular on the axis: u_r odd and u_z even in
# r, from the Stokes stream function psi = r^2 (1 + r^2) cos z.
AXISYMMETRIC = {
    "u": "x*(1 + x**2)*sin(y)",
    "v": "(2 + 4*x**2)*cos(y)",
    "pressure": "cos(x)*sin(y) + x**2*y",
}


def _square(n, element_type="Quad4"):
    mesh = dm.generate_rectangle_mesh(0.1, 1.1, 0.2, 1.2, n, n, element_type=element_type)
    mesh.transform_nodes(
        lambda x, y, z: [
            x + 0.6 * (x - 0.1) * (1.1 - x) * (y - 0.2) * (1.2 - y) * np.sin(3 * y),
            y - 0.5 * (x - 0.1) * (1.1 - x) * (y - 0.2) * (1.2 - y),
            0.0,
        ]
    )
    return mesh


def _check(result, velocity):
    assert result.order(velocity, "l2") > 1.8
    assert result.order(velocity, "h1") > 0.9
    assert result.order("pressure", "l2") > 1.2


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("density", [0.0, 1.0])
def test_manufactured_flow_on_distorted_quadrilaterals(method, density):
    """Velocity O(h^2) in L2 and O(h) in H1, pressure better than O(h) in L2.
    The right side is an outlet with the exact traction, so the pressure is
    determined without being pinned (see ``mms.IncompressibleFlow``)."""
    study = mms.ManufacturedSolution(
        CARTESIAN,
        [mms.IncompressibleFlow(["u", "v"], density=density, outflow={"right": (1.0, 0.0)})],
        dimension=2,
    )
    _check(study.convergence_study(_square, [8, 16, 32], method=method), "u")


@pytest.mark.parametrize("method", METHODS)
def test_manufactured_flow_on_triangles(method):
    study = mms.ManufacturedSolution(
        CARTESIAN,
        [mms.IncompressibleFlow(["u", "v"], density=1.0, outflow={"right": (1.0, 0.0)})],
        dimension=2,
    )
    result = study.convergence_study(lambda n: _square(n, "Tri3"), [8, 16, 32], method=method)
    _check(result, "u")


@pytest.mark.parametrize("method", METHODS)
def test_manufactured_axisymmetric_flow_including_the_axis(method):
    """The axis r = 0 is part of the domain.  The radial momentum equation
    carries the hoop stress sigma_tt / r = (2 mu u_r / r - p) / r, and the
    cell-centred method evaluates fluxes on the axis itself, which needs the
    manufactured forcing in a form that is finite there."""
    study = mms.ManufacturedSolution(
        AXISYMMETRIC,
        [mms.IncompressibleFlow(["u", "v"], density=1.0, outflow={"top": (0.0, 1.0)})],
        dimension=2,
        coordinates="axisymmetric",
    )
    result = study.convergence_study(
        lambda n: dm.generate_rectangle_mesh(0.0, 1.0, 0.0, 1.0, n, n), [8, 16, 32], method=method
    )
    _check(result, "u")


@pytest.mark.parametrize("method", ["fem", "zfvm"])
def test_manufactured_flow_in_three_dimensions(method):
    """Pre-asymptotic on these meshes, so only the ratio of successive errors
    is checked: at least 2.5 for the velocity and 2 for the pressure (the
    measured ratios are 2.9 to 3.6)."""
    study = mms.ManufacturedSolution(
        {"u": "sin(y+z)", "v": "sin(x+z)", "w": "sin(x+y)", "pressure": "cos(x)*y*z + x"},
        [mms.IncompressibleFlow(["u", "v", "w"], density=1.0, outflow={"right": (1.0, 0.0, 0.0)})],
        dimension=3,
    )
    result = study.convergence_study(
        lambda n: dm.generate_box_mesh(0, 1, 0, 1, 0, 1, n, n, n), [4, 8], method=method
    )
    u = result.errors[("u", "l2")]
    p = result.errors[("pressure", "l2")]
    assert u[0] / u[1] > 2.5
    assert p[0] / p[1] > 2.0


@pytest.mark.parametrize("method", METHODS)
def test_newton_converges_quadratically(method):
    """The steps of Newton's method on a Navier-Stokes problem started away
    from the solution satisfy s_{k+1} <= C s_k^2.  Before the cell-source
    gradients of the cell-centred method were differentiated, that method
    converged only linearly, at a rate of about 0.1 per iteration."""
    study = mms.ManufacturedSolution(
        CARTESIAN,
        [mms.IncompressibleFlow(["u", "v"], density=20.0, outflow={"right": (1.0, 0.0)})],
        dimension=2,
    )
    problem = study.build(_square(16), method=method)
    for variable in ("u", "v", "pressure"):
        problem.set_values(variable, 0.5 * problem.values(variable))
    result = problem.solve(relative_tolerance=1e-13, absolute_tolerance=1e-13)
    steps = [r.step_norm for r in result.history if r.step_norm > 1e-14]
    ratios = [steps[i + 1] / steps[i] ** 2 for i in range(len(steps) - 1) if steps[i] < 0.1]
    assert ratios, steps
    assert max(ratios) < 50.0, steps
    assert result.total_iterations <= 8


def _cavity(method, n, reynolds_number):
    mesh = dm.generate_rectangle_mesh(0, 1, 0, 1, n, n)
    problem = dm.Problem(mesh, method=method)
    dm.physics.add_incompressible_flow(
        problem,
        velocities=["u", "v"],
        dynamic_viscosity=1.0,
        density=reynolds_number,
        formulation="pressure",
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
    # The walls are added last, so that the two top corner nodes of the node
    # based methods belong to the walls: the lid does not leak.
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


@pytest.mark.parametrize("method", METHODS)
def test_cavity_matches_ghia(method):
    """Re = 100 on a 32 by 32 mesh: the centreline velocities are within 0.015
    of the tabulated values of Ghia, Ghia and Shin (1982); the measured
    differences are 0.002 to 0.014."""
    spec = importlib.util.spec_from_file_location(
        "compare_cavity", _ROOT / "verification" / "openfoam" / "compare_cavity.py"
    )
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    problem = _cavity(method, 32, 100.0)
    u = problem.sample(
        "u", np.column_stack([np.full_like(reference.GHIA_Y, 0.5), reference.GHIA_Y])
    )
    v = problem.sample(
        "v", np.column_stack([reference.GHIA_X, np.full_like(reference.GHIA_X, 0.5)])
    )
    assert np.max(np.abs(u - reference.GHIA_U)) < 0.015
    assert np.max(np.abs(v - reference.GHIA_V)) < 0.015


@pytest.mark.parametrize("method", ["fem", "dmcdm", "zfvm"])
def test_natural_convection_with_the_pressure_formulation(method):
    """de Vahl Davis at Ra = 10^5 on the 32 by 32 graded mesh.  The buoyancy
    force is part of the stabilisation and scales with the load factor like
    the force itself; without it the stabilisation pushes against the flow.
    Measured: Nusselt number within 2.1 %, velocity maxima within 2.1 %."""
    spec = importlib.util.spec_from_file_location(
        "natural_convection", _ROOT / "examples" / "natural_convection.py"
    )
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    rayleigh_number = 1.0e5
    problem = dm.Problem(example.cavity_mesh(32), method=method)
    problem.add_variable("temperature")
    velocities = dm.physics.add_incompressible_flow(
        problem,
        velocities=["u", "v"],
        dynamic_viscosity=example.PRANDTL,
        density=1.0,
        formulation="pressure",
        pressure_pin_point=(0.5, 0.0),
        buoyancy={
            "temperature": "temperature",
            "gravity": [0.0, -1.0],
            "thermal_expansion_coefficient": rayleigh_number * example.PRANDTL,
            "reference_temperature": 0.0,
            "scale_with_load": True,
        },
    )
    problem.add_kernel("heat_conduction", "conduction", variable="temperature")
    problem.add_kernel(
        "heat_convection", "convection", variable="temperature", velocities=velocities
    )
    for variable in velocities:
        problem.add_boundary_condition(
            "Dirichlet_boundary_condition",
            f"no_slip_{variable}",
            variable=variable,
            boundary=["left", "right", "bottom", "top"],
            value=0.0,
        )
    problem.add_boundary_condition(
        "Dirichlet_boundary_condition", "hot", variable="temperature", boundary="left", value=0.5
    )
    problem.add_boundary_condition(
        "Dirichlet_boundary_condition", "cold", variable="temperature", boundary="right", value=-0.5
    )
    problem.solve(load_factors=[0.01, 0.1, 1.0], max_iterations=40)
    computed = example.measure(problem)
    reference = example.BENCHMARK[rayleigh_number]
    for key in ("nusselt", "u_max", "v_max"):
        assert computed[key] == pytest.approx(reference[key], rel=3e-2), key


def test_cell_centred_jacobian_of_a_gradient_source_is_exact():
    """The convective inertia rho u . grad u is a cell source of the
    cell-centred method that depends on the reconstructed gradient.  Its
    Jacobian, applied to a random direction, must match a central difference
    of the residual to the accuracy of the difference."""
    mesh = dm.generate_rectangle_mesh(0, 1, 0, 1, 6, 6, element_type="Tri3")
    problem = dm.Problem(mesh, method="zfvm")
    dm.physics.add_incompressible_flow(
        problem, velocities=["u", "v"], dynamic_viscosity=1.0, density=30.0, formulation="pressure"
    )
    rng = np.random.default_rng(3)
    count = len(problem.entity_points())
    for variable in ("u", "v", "pressure"):
        problem.set_values(variable, rng.uniform(-1, 1, count))
    values = {v: problem.values(v).copy() for v in ("u", "v", "pressure")}
    residual, jacobian = problem.linear_system()
    direction = rng.uniform(-1, 1, residual.size)
    epsilon = 1e-6

    def shifted(sign):
        for index, variable in enumerate(("u", "v", "pressure")):
            problem.set_values(variable, values[variable] + sign * epsilon * direction[index::3])
        return problem.linear_system()[0]

    difference = (shifted(1.0) - shifted(-1.0)) / (2 * epsilon)
    exact = jacobian @ direction
    assert np.max(np.abs(exact - difference)) < 1e-6 * np.max(np.abs(exact))


def test_cell_centred_axis_needs_no_boundary_condition():
    """On the axis of an axisymmetric problem the coordinate factor r
    vanishes, which used to leave the cell-centred method's boundary entities
    there with an empty equation and the matrix singular.  The axis is now a
    symmetry line without a condition, and the solution converges at second
    order."""
    study = mms.ManufacturedSolution(
        {"u": "cos(x)*sin(y) + x*x"}, [mms.Diffusion("u")], dimension=2, coordinates="axisymmetric"
    )
    result = study.convergence_study(
        lambda n: dm.generate_rectangle_mesh(0, 1, 0, 1, n, n),
        [8, 16, 32],
        method="zfvm",
        boundary=["right", "top", "bottom"],
    )
    assert result.order("u", "l2") > 1.9


def test_point_dirichlet_fixes_the_nearest_entity():
    mesh = dm.generate_rectangle_mesh(0, 1, 0, 1, 4, 4)
    problem = dm.Problem(mesh, method="fem")
    problem.add_variable("u")
    problem.add_kernel("diffusion", "diffusion", variable="u")
    problem.add_boundary_condition(
        "point_Dirichlet_boundary_condition", "pin", variable="u", point=[0.49, 0.26], value=2.0
    )
    problem.add_boundary_condition(
        "Dirichlet_boundary_condition", "outer", variable="u", boundary=["left"], value=2.0
    )
    problem.solve()
    points = problem.entity_points()
    nearest = int(np.argmin(np.hypot(points[:, 0] - 0.5, points[:, 1] - 0.25)))
    assert problem.values("u")[nearest] == pytest.approx(2.0)
    assert np.allclose(problem.values("u"), 2.0)
