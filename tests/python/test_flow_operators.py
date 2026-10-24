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

import dualmesh as dm
import numpy as np
import pytest
import scipy.sparse as sp


def _cavity(
    element_type="Quad4",
    formulation="pressure",
    n=6,
    density=0.0,
    viscosity=1.0,
    coordinates="cartesian",
    scale=1.0,
):
    mesh = dm.generate_rectangle_mesh(
        x=[0, scale], y=[0, scale], num_elements=[n, n], element_type=element_type
    )
    problem = dm.Problem(mesh, method="fem", coordinates=coordinates)
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
    return sp.coo_matrix((values, (rows, cols)), shape=(n, n)).tocsr()


def _pressure_block(ops, name):
    p = np.asarray(ops["pressure_dofs"])
    n = len(ops["lumped_mass"])
    return _matrix(ops[name], n)[p][:, p].toarray(), p


@pytest.mark.parametrize(
    "element_type, formulation, coordinates, measure",
    [
        ("Quad4", "pressure", "cartesian", 1.0),
        ("Tri3", "pressure", "cartesian", 1.0),
        ("Quad9", "Taylor-Hood", "cartesian", 1.0),
        ("Tri6", "Taylor-Hood", "cartesian", 1.0),
        ("Quad4", "pressure", "axisymmetric", np.pi),  # the integral of 2 pi r over [0, 1]^2
    ],
)
def test_the_lumped_masses_sum_to_the_measure_of_the_domain(
    element_type, formulation, coordinates, measure
):
    problem = _cavity(element_type, formulation, coordinates=coordinates)
    ops = _operators(problem, "pcd")
    q = np.asarray(ops["lumped_mass"])
    p = np.asarray(ops["pressure_dofs"])
    # Q_p lives on the pressure nodes (the mid-side slots of Taylor-Hood carry none).
    assert q[p].sum() == pytest.approx(measure, rel=1e-12)
    assert np.all(q[p][q[p] != 0] > 0)
    nv = len(q) // problem.mesh.num_nodes
    for component in (0, 1):
        velocity = q[component::nv]
        assert velocity.sum() == pytest.approx(measure, rel=1e-12)
        assert np.all(velocity > 0)


def _channel(element_type, formulation, density=50.0, viscosity=1.0):
    """A channel 4 x 1 with the uniform inflow U = 1 on the left and a free outflow on the right;
    the walls hold the corner nodes of the inflow at rest.  Four elements across."""
    mesh = dm.generate_rectangle_mesh(
        x=[0, 4], y=[0, 1], num_elements=[16, 4], element_type=element_type
    )
    problem = dm.Problem(mesh, method="fem")
    flow = problem.add_physics(
        "fluid_dynamics", density=density, dynamic_viscosity=viscosity, formulation=formulation
    )
    flow.add_boundary_condition(["velocity_x = 1", "velocity_y = 0"], boundary="left")
    flow.add_boundary_condition(["velocity_x = 0", "velocity_y = 0"], boundary=["top", "bottom"])
    return problem


@pytest.mark.parametrize(
    "element_type, formulation", [("Quad4", "pressure"), ("Quad9", "Taylor-Hood")]
)
def test_the_pressure_mass_matrix_sums_to_the_integral_of_one_over_mu(element_type, formulation):
    """The channel has no prescribed pressure, so the whole matrix is there: area / mu."""
    problem = _channel(element_type, formulation, viscosity=4.0)
    problem._build_physics()
    M, _ = _pressure_block(_operators(problem, "pressure_mass"), "mass")
    assert M.sum() == pytest.approx(4.0 / 4.0, rel=1e-12)
    assert np.allclose(M, M.T)
    # The mid-side pressure slots of Taylor-Hood are inactive and have no entries.
    active = np.abs(M).sum(axis=1) > 0
    assert np.linalg.eigvalsh(M[active][:, active]).min() > 0


@pytest.mark.parametrize("preconditioner", ["pcd", "lsc"])
def test_the_stabilisation_vanishes_on_a_constant_and_is_semidefinite(preconditioner):
    """On the channel, which has no prescribed pressure, so that every row is complete."""
    problem = _channel("Quad4", "pressure")
    problem._build_physics()
    C1, _ = _pressure_block(_operators(problem, preconditioner), "stabilization")
    scale = np.abs(C1).max()
    assert scale > 0
    assert np.allclose(C1, C1.T, rtol=0, atol=1e-14 * scale)
    assert np.abs(C1 @ np.ones(len(C1))).max() <= 1e-12 * scale
    assert np.linalg.eigvalsh(C1).min() >= -1e-12 * scale


def test_taylor_hood_has_no_stabilisation():
    ops = _operators(_cavity("Quad9", "Taylor-Hood"), "lsc")
    assert len(ops["stabilization"][2]) == 0
    assert len(ops["commutator_stabilization"][2]) == 0


def test_pcd_takes_the_stabilisation_a_hundred_times_larger_than_lsc():
    problem = _cavity("Quad4", "pressure")
    pcd, p = _pressure_block(_operators(problem, "pcd"), "stabilization")
    lsc, _ = _pressure_block(_operators(problem, "lsc"), "stabilization")
    assert np.allclose(pcd, 100.0 * lsc)


def test_the_stabilisation_scales_with_the_unit_of_length_like_a_laplacian():
    """In axisymmetric coordinates the pressure Laplacian scales as the length (one factor r of the
    measure 2 pi r dA, the h^2 of dA cancelling the h^-2 of the gradients).  The size h of the
    stabilisation is that of the element itself, so C_1 scales the same way."""
    metres, _ = _pressure_block(
        _operators(_cavity("Quad4", "pressure", coordinates="axisymmetric"), "lsc"), "stabilization"
    )
    millimetres, _ = _pressure_block(
        _operators(_cavity("Quad4", "pressure", coordinates="axisymmetric", scale=1e-3), "lsc"),
        "stabilization",
    )
    assert np.allclose(millimetres, 1e-3 * metres, rtol=1e-10, atol=1e-13 * np.abs(metres).max())


def test_c2_is_c1_times_mu_over_h_squared_on_a_uniform_mesh():
    n, mu = 6, 3.0
    problem = _cavity("Quad4", "pressure", n=n, viscosity=mu)
    ops = _operators(problem, "lsc")
    C1, _ = _pressure_block(ops, "stabilization")
    C2, _ = _pressure_block(ops, "commutator_stabilization")
    h = 1.0 / n
    assert np.allclose(C2, mu / h**2 * C1, rtol=1e-10, atol=0)


@pytest.mark.parametrize(
    "element_type, formulation", [("Quad4", "pressure"), ("Quad9", "Taylor-Hood")]
)
def test_fp_is_zero_on_a_constant_away_from_the_inflow(element_type, formulation):
    """The diffusion and the convection of a constant vanish, so F_p 1 is zero on every row
    except those of the inflow side, where the Robin term acts."""
    problem = _channel(element_type, formulation, density=50.0)
    problem.solve(report="none")
    ops = _operators(problem, "pcd")
    Fp, p = _pressure_block(ops, "convection_diffusion")
    x = np.asarray(problem.mesh.points())[:, 0]
    nv = len(np.asarray(ops["lumped_mass"])) // problem.mesh.num_nodes
    away = x[p // nv] > 1e-12
    active = np.abs(Fp).sum(axis=1) > 0
    row_sums = Fp @ np.ones(len(Fp))
    assert np.abs(Fp).max() > 0
    assert np.abs(row_sums[away & active]).max() <= 1e-10 * np.abs(Fp).max()
    assert np.all(row_sums[~away & active] > 0)


@pytest.mark.parametrize(
    "element_type, formulation, inflow",
    [
        # The integral of the discrete u over the inflow side: 1 - h with the linear velocity,
        # 1 - h / 3 with the quadratic one, h = 1/4, the corners at rest.
        ("Quad4", "pressure", 1.0 - 0.25),
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
    Fp, _ = _pressure_block(_operators(problem, "pcd"), "convection_diffusion")
    assert Fp.sum() == pytest.approx(rho * inflow, rel=1e-10)


def test_no_operator_has_an_entry_at_a_prescribed_pressure():
    ops = _operators(_cavity("Quad4", "pressure", density=10.0), "lsc")
    constrained = np.asarray(ops["constrained"]).astype(bool)
    for name in ("stabilization", "commutator_stabilization"):
        rows, cols, _ = ops[name]
        assert not constrained[np.asarray(rows)].any()
        assert not constrained[np.asarray(cols)].any()


def test_a_transient_step_gives_its_time_step_and_theta_to_the_operators():
    """The operators of the last linear solve take the time step and the weight theta of the
    step; a steady solve gives zero and one."""
    problem = _cavity("Quad4", "pressure", density=100.0)
    problem.solve_transient(
        end_time=0.02,
        time_step=0.01,
        implicitness=0.5,
        preconditioner="lsc",
        linear_solver="gmres",
        report="none",
    )
    assert problem._problem._linearization() == pytest.approx((0.01, 0.5))
    problem.solve(preconditioner="lsc", linear_solver="gmres", report="none")
    assert problem._problem._linearization() == pytest.approx((0.0, 1.0))
