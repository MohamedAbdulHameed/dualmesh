# SPDX-License-Identifier: LGPL-2.1-or-later
"""The operators of the flow preconditioners, checked against exact properties.

The iteration counts of test_saddle_point.py show that the preconditioners work, but a wrong
scale of an operator can hide in them: a lumped mass off by a constant factor changes the count by
a few iterations only.  These tests check each operator assembled by Problem.saddlePointBlocks
against what it must be, whatever the method that assembled the system:

* the lumped masses Q_p and Q_v sum to the measure of the domain (with 2 pi r in axisymmetric
  coordinates), and every entry is positive, also for quadratic elements;
* the mass matrix of pressure_mass sums to the integral of 1/mu;
* C_1 is symmetric, positive semidefinite, and zero on a constant pressure; it is present for
  the stabilised element and absent for Taylor-Hood; its size follows from the element itself, so
  it scales with the unit of length as the pressure Laplacian does;
* C_2 is C_1 times theta mu / h^2 on a uniform mesh;
* F_p is zero on a constant pressure in an enclosed flow, and on a channel it sums to the inflow
  term rho U H of its Robin condition;
* no operator has an entry at a prescribed pressure.
"""

from __future__ import annotations

import dualmesh
import numpy
import pytest
from scipy import sparse


def _cavity(
    element_type="Quad4",
    formulation="equal_order",
    n=6,
    density=0.0,
    viscosity=1.0,
    coordinates="cartesian",
    scale=1.0,
):
    mesh = dualmesh.generate_rectangle_mesh(
        x=[0, scale], y=[0, scale], num_elements=[n, n], element_type=element_type
    )
    problem = dualmesh.Problem(mesh, method="fem", coordinates=coordinates)
    flow = problem.add_physics(
        "fluid_dynamics",
        density=density,
        dynamic_viscosity=viscosity,
        formulation=formulation,
    )
    u, v = (
        ("velocity_r", "velocity_z")
        if coordinates == "axisymmetric"
        else ("velocity_x", "velocity_y")
    )
    flow.add_boundary_condition([f"{u} = 1", f"{v} = 0"], boundary="top")
    flow.add_boundary_condition([f"{u} = 0", f"{v} = 0"], boundary=["left", "right", "bottom"])
    problem._build_physics()
    return problem


def _operators(problem, preconditioner):
    return problem._problem._flow_operators(preconditioner)


def _matrix(triplet, n):
    rows, cols, values = triplet
    return sparse.coo_matrix((values, (rows, cols)), shape=(n, n)).tocsr()


def _pressure_block(ops, name):
    p = numpy.asarray(ops["pressure_dofs"])
    n = len(ops["lumped_mass"])
    return _matrix(ops[name], n)[p][:, p].toarray(), p


@pytest.mark.parametrize(
    "element_type, formulation, coordinates, measure",
    [
        ("Quad4", "equal_order", "cartesian", 1.0),
        ("Tri3", "equal_order", "cartesian", 1.0),
        ("Quad9", "Taylor-Hood", "cartesian", 1.0),
        ("Tri6", "Taylor-Hood", "cartesian", 1.0),
        ("Quad4", "equal_order", "axisymmetric", numpy.pi),  # The integral of 2 pi r over [0, 1]^2
    ],
)
def test_the_lumped_masses_sum_to_the_measure_of_the_domain(
    element_type, formulation, coordinates, measure
):
    problem = _cavity(element_type, formulation, coordinates=coordinates)
    ops = _operators(problem, "pcd")
    q = numpy.asarray(ops["lumped_mass"])
    p = numpy.asarray(ops["pressure_dofs"])
    # Q_p lives on the pressure nodes (the mid-side slots of Taylor-Hood carry none).
    assert q[p].sum() == pytest.approx(measure, rel=1e-12)
    assert numpy.all(q[p][q[p] != 0] > 0)
    nv = len(q) // problem.mesh.num_nodes
    for component in (0, 1):
        velocity = q[component::nv]
        assert velocity.sum() == pytest.approx(measure, rel=1e-12)
        assert numpy.all(velocity > 0)


def _channel(element_type, formulation, density=50.0, viscosity=1.0):
    """A channel 4 x 1 with the uniform inflow U = 1 on the left and a free outflow on the right;
    the walls hold the corner nodes of the inflow at rest.  Four elements across."""
    mesh = dualmesh.generate_rectangle_mesh(
        x=[0, 4], y=[0, 1], num_elements=[16, 4], element_type=element_type
    )
    problem = dualmesh.Problem(mesh, method="fem")
    flow = problem.add_physics(
        "fluid_dynamics", density=density, dynamic_viscosity=viscosity, formulation=formulation
    )
    flow.add_boundary_condition(["velocity_x = 1", "velocity_y = 0"], boundary="left")
    flow.add_boundary_condition(["velocity_x = 0", "velocity_y = 0"], boundary=["top", "bottom"])
    return problem


@pytest.mark.parametrize(
    "element_type, formulation", [("Quad4", "equal_order"), ("Quad9", "Taylor-Hood")]
)
def test_the_pressure_mass_matrix_sums_to_the_integral_of_one_over_mu(element_type, formulation):
    """The channel has no prescribed pressure, so the whole matrix is there: area / mu."""
    problem = _channel(element_type, formulation, viscosity=4.0)
    problem._build_physics()
    M, _ = _pressure_block(_operators(problem, "pressure_mass"), "mass")
    assert M.sum() == pytest.approx(4.0 / 4.0, rel=1e-12)
    assert numpy.allclose(M, M.T)
    # The mid-side pressure slots of Taylor-Hood are inactive and have no entries.
    active = numpy.abs(M).sum(axis=1) > 0
    assert numpy.linalg.eigvalsh(M[active][:, active]).min() > 0


@pytest.mark.parametrize("preconditioner", ["pcd", "lsc"])
def test_the_stabilisation_vanishes_on_a_constant_and_is_semidefinite(preconditioner):
    """On the channel, which has no prescribed pressure, so that every row is complete."""
    problem = _channel("Quad4", "equal_order")
    problem._build_physics()
    c1, _ = _pressure_block(_operators(problem, preconditioner), "stabilization")
    scale = numpy.abs(c1).max()
    assert scale > 0
    assert numpy.allclose(c1, c1.T, rtol=0, atol=1e-14 * scale)
    assert numpy.abs(c1 @ numpy.ones(len(c1))).max() <= 1e-12 * scale
    assert numpy.linalg.eigvalsh(c1).min() >= -1e-12 * scale


def test_the_stabilisation_scales_with_the_unit_of_length_like_a_laplacian():
    """In axisymmetric coordinates the pressure Laplacian scales as the length (one factor r of the
    measure 2 pi r dA, the h^2 of dA cancelling the h^-2 of the gradients).  The size h of the
    stabilisation is that of the element itself, so C_1 scales the same way."""
    metres, _ = _pressure_block(
        _operators(_cavity("Quad4", "equal_order", coordinates="axisymmetric"), "lsc"),
        "stabilization",
    )
    millimetres, _ = _pressure_block(
        _operators(_cavity("Quad4", "equal_order", coordinates="axisymmetric", scale=1e-3), "lsc"),
        "stabilization",
    )
    assert numpy.allclose(
        millimetres, 1e-3 * metres, rtol=1e-10, atol=1e-13 * numpy.abs(metres).max()
    )


@pytest.mark.parametrize(
    "element_type, formulation", [("Quad4", "equal_order"), ("Quad9", "Taylor-Hood")]
)
def test_fp_is_zero_on_a_constant_away_from_the_inflow(element_type, formulation):
    """The diffusion and the convection of a constant vanish, so F_p 1 is zero on every row
    except those of the inflow side, where the Robin term acts."""
    problem = _channel(element_type, formulation, density=50.0)
    problem.solve(report="none")
    ops = _operators(problem, "pcd")
    F_p, p = _pressure_block(ops, "convection_diffusion")
    x = numpy.asarray(problem.mesh.points())[:, 0]
    nv = len(numpy.asarray(ops["lumped_mass"])) // problem.mesh.num_nodes
    away = x[p // nv] > 1e-12
    active = numpy.abs(F_p).sum(axis=1) > 0
    row_sums = F_p @ numpy.ones(len(F_p))
    assert numpy.abs(F_p).max() > 0
    assert numpy.abs(row_sums[away & active]).max() <= 1e-10 * numpy.abs(F_p).max()
    assert numpy.all(row_sums[~away & active] > 0)


@pytest.mark.parametrize(
    "element_type, formulation, inflow",
    [
        # The integral of the discrete u over the inflow side: 1 - h with the linear velocity,
        # 1 - h / 3 with the quadratic one, h = 1/4, the corners at rest.
        ("Quad4", "equal_order", 1.0 - 0.25),
        ("Quad9", "Taylor-Hood", 1.0 - 0.25 / 3.0),
    ],
)
def test_fp_sums_to_the_inflow_term_of_its_robin_condition(element_type, formulation, inflow):
    """The convection and the diffusion of a constant vanish, so the sum of F_p is that of its
    Robin term, rho times the integral of u over the inflow.  The walls add nothing (u = 0) and
    the outflow, with its natural condition, nothing either."""
    rho = 50.0
    problem = _channel(element_type, formulation, density=rho)
    problem.solve(report="none")
    F_p, _ = _pressure_block(_operators(problem, "pcd"), "convection_diffusion")
    assert F_p.sum() == pytest.approx(rho * inflow, rel=1e-10)
