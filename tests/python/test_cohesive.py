# SPDX-License-Identifier: LGPL-2.1-or-later
"""Cohesive zones: the bilinear traction-separation law along a path that softens, unloads,
closes and separates, the work of separation in tension and in shear, and the balance of the
forces on the two surfaces."""

from __future__ import annotations

import dualmesh
import numpy
import pytest
from dualmesh._core import InputError
from test_contact import two_blocks

E, nu = 100e9, 0.3
plane = E / (1 - nu**2)
strength, fracture_toughness = 3e6, 100.0


def bonded(
    primary="lower_top", secondary="upper_bottom", method="FEM", toughness=fracture_toughness
):
    """The two blocks of the contact tests, bonded at y = 0.5 by a cohesive zone."""
    mesh = two_blocks()
    problem = dualmesh.Problem(mesh, method=method)
    solid = problem.add_physics(
        "solid_mechanics", formulation="plane_strain", youngs_modulus=E, poissons_ratio=nu
    )
    solid.add_boundary_condition(
        "cohesive", primary=primary, secondary=secondary, strength=strength, toughness=toughness
    )
    solid.add_boundary_condition("disp_y = 0", boundary="bottom")
    return problem, solid


def bilinear(opening, reached, stiffness):
    """The traction of the law at an opening, with the largest opening reached so far."""
    if opening < 0.0:
        return stiffness * opening
    d0, df = strength / stiffness, 2 * fracture_toughness / strength
    kappa = max(reached, opening)
    damage = 0.0 if kappa <= d0 else 1.0 if kappa >= df else df * (kappa - d0) / (kappa * (df - d0))
    return stiffness * (1 - damage) * opening


@pytest.mark.parametrize(
    "method, primary, secondary",
    [
        ("FEM", "lower_top", "upper_bottom"),
        ("FEM", "upper_bottom", "lower_top"),
        ("DMCDM", "lower_top", "upper_bottom"),
    ],
)
def test_the_zone_follows_the_bilinear_law_and_does_not_heal(method, primary, secondary):
    """The top pulled into the softening branch, pushed back through zero into compression, and
    pulled until the zone separates.  The stress is uniform, so the traction is the force per
    width and the opening is uniform: at every step the traction is the bilinear law of the
    opening and of the largest opening reached (unloading follows the secant to zero, closing
    keeps the full stiffness), the bulk takes the traction (1 - nu^2)/E per height, the forces
    on the two bodies balance, the work of the top force is the toughness, and the result
    does not depend on which surface is the primary one."""
    problem, solid = bonded(primary, secondary, method)
    solid.add_boundary_condition("disp_x = 0", boundary="left")
    # The top displacement: up to 4e-5 at t = 1 (the peak is passed near 2.75e-5), down to
    # -1e-5 at t = 2, up to 9e-5 at t = 4, past the separation at 2 toughness / strength.
    solid.add_boundary_condition(
        "disp_y = 4e-5*t - 5e-5*max(t - 1, 0) + 6e-5*max(t - 2, 0)", boundary="top"
    )
    mesh = problem.mesh
    nodes = {name: mesh.boundary_nodes(name) for name in ("top", "upper_bottom", "lower_top")}
    # The stiffness of the zone, 50 E / h with h the side of the secondary surface (1/5 or 1/4).
    stiffness = 50 * E / (0.2 if secondary == "upper_bottom" else 0.25)
    steps = []

    def record(t, core):
        u = problem.values("disp_y")
        for name in nodes:
            assert u[nodes[name]] == pytest.approx(u[nodes[name]][0], rel=1e-9, abs=1e-16)
        steps.append(
            (
                u[nodes["top"]][0],
                u[nodes["upper_bottom"]][0] - u[nodes["lower_top"]][0],
                solid.reaction_force("top")["y"],
                solid.reaction_force("upper_bottom")["y"],
                solid.reaction_force("lower_top")["y"],
            )
        )

    problem.set_time_step_callback(record)
    problem.solve_transient(end_time=4.0, time_step=0.02, time_stepper="fixed", report="none")
    reached, work, before = 0.0, 0.0, (0.0, 0.0)
    for top, opening, force, through_upper, through_lower in steps:
        traction = bilinear(opening, reached, stiffness)
        assert force == pytest.approx(traction, rel=1e-8, abs=1e-6 * strength)
        assert top == pytest.approx(opening + traction / plane, rel=1e-7, abs=1e-15)
        # The zone pulls the upper body down and the lower one up.
        assert through_upper == pytest.approx(-force, rel=1e-8, abs=1e-6 * strength)
        assert through_lower == pytest.approx(force, rel=1e-8, abs=1e-6 * strength)
        reached = max(reached, opening)
        work += 0.5 * (force + before[1]) * (top - before[0])
        before = (top, force)
    assert reached > 2 * fracture_toughness / strength
    assert work == pytest.approx(fracture_toughness, rel=1e-3)


def test_a_sheared_zone_separates_with_the_work_of_the_toughness():
    """The top moved sideways until the zone fails, the bottom held.  The shear is not uniform,
    but at every step the forces on the two bodies balance, and at the end the zone carries
    nothing and the work of the top force is the toughness over the width: the work of
    separation of the law is the toughness along any path.  The toughness is ten times that of
    the other tests, so that the zone softens more slowly than the blocks unload in shear
    (sigma_c^2 / 2 G_c below G / H) and the failure is stable: an abrupt failure would lose the
    energy it releases."""
    toughness = 10 * fracture_toughness
    problem, solid = bonded(toughness=toughness)
    solid.add_boundary_condition("disp_x = 0", boundary="bottom")
    solid.add_boundary_condition("disp_x = 1e-3*t", boundary="top")
    solid.add_boundary_condition("disp_y = 0", boundary="top")
    top = problem.mesh.boundary_nodes("top")
    steps = []

    def record(t, core):
        force = solid.reaction_force("top")
        upper, lower = solid.reaction_force("upper_bottom"), solid.reaction_force("lower_top")
        for k in "xy":
            assert upper[k] == pytest.approx(-force[k], rel=1e-8, abs=1e-6 * strength)
            assert lower[k] == pytest.approx(force[k], rel=1e-8, abs=1e-6 * strength)
        steps.append((problem.values("disp_x")[top][0], force["x"]))

    problem.set_time_step_callback(record)
    problem.solve_transient(end_time=1.0, time_step=0.005, time_stepper="fixed", report="none")
    moves, forces = numpy.array([(0.0, 0.0)] + steps).T
    assert max(forces) > 0.5 * strength
    assert forces[-1] == pytest.approx(0.0, abs=1e-6 * strength)
    assert numpy.trapezoid(forces, moves) == pytest.approx(toughness, rel=1e-3)


def test_the_input_of_a_cohesive_zone_is_checked():
    mesh = two_blocks()
    problem = dualmesh.Problem(mesh)
    solid = problem.add_physics(
        "solid_mechanics", formulation="plane_strain", youngs_modulus=E, poissons_ratio=nu
    )
    with pytest.raises(InputError, match="give primary and secondary"):
        solid.add_boundary_condition(
            "cohesive", primary="lower_top", secondary="upper_bottom", strength=1e6
        )
    # Surfaces apart at the start are not a bonded interface.
    apart = two_blocks(gap=1e-3)
    problem = dualmesh.Problem(apart)
    solid = problem.add_physics(
        "solid_mechanics", formulation="plane_strain", youngs_modulus=E, poissons_ratio=nu
    )
    solid.add_boundary_condition(
        "cohesive",
        primary="lower_top",
        secondary="upper_bottom",
        strength=strength,
        toughness=fracture_toughness,
    )
    solid.add_boundary_condition("disp_y = 0", boundary="bottom")
    with pytest.raises(InputError, match="must coincide at the start"):
        problem.solve(report="none")
    # A toughness too small for the strength: the zone would fail before it softens.
    problem = dualmesh.Problem(two_blocks())
    solid = problem.add_physics(
        "solid_mechanics", formulation="plane_strain", youngs_modulus=E, poissons_ratio=nu
    )
    solid.add_boundary_condition(
        "cohesive", primary="lower_top", secondary="upper_bottom", strength=strength, toughness=1e-9
    )
    with pytest.raises(InputError, match="fail before it softens"):
        problem.solve(report="none")
