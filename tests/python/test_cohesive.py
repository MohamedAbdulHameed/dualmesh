# SPDX-License-Identifier: LGPL-2.1-or-later
"""Cohesive zones: the bilinear traction-separation law along a path that softens, unloads,
closes and separates, the work of separation in tension and in shear, the balance of the
forces on the two surfaces, and zones in axisymmetry and at finite strain."""

from __future__ import annotations

import dualmesh
import numpy
import pytest
from dualmesh._core import InputError
from scipy.optimize import brentq
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


def test_a_zone_restarted_from_its_checkpoint_keeps_its_damage():
    """The top pulled into the softening branch, the run stopped there, and the run restarted
    from its checkpoint to unload part way.  The checkpoint holds the largest opening of every
    point of the zone, so the restarted zone unloads along the secant of the opening reached
    before the stop, which the bilinear law gives in closed form, and ends where an
    uninterrupted run ends.  A zone that had lost its damage would load again from the start
    of its law and carry the traction of its law at the opening, here 21 percent more."""
    import pathlib
    import shutil

    # Under mpirun every process runs its own serial problem and writes its own checkpoint.
    directory = pathlib.Path(f"dualmesh_cohesive_checkpoint_test_{dualmesh.parallel.rank()}")
    directory.mkdir(exist_ok=True)
    # The stiffness of the zone, 50 E / h with h = 1/5 the side of the secondary surface.
    stiffness = 50 * E / 0.2
    steps = dict(time_step=0.02, time_stepper="fixed", report="none")

    def zone():
        problem, solid = bonded()
        solid.add_boundary_condition("disp_x = 0", boundary="left")
        # Up to 4e-5 at t = 1, past the peak near 2.75e-5, then down to 1.5e-5 at t = 1.5.
        solid.add_boundary_condition("disp_y = 4e-5*t - 5e-5*max(t - 1, 0)", boundary="top")
        return problem, solid

    def opening_and_force(problem, solid):
        u = problem.values("disp_y")
        mesh = problem.mesh
        opening = u[mesh.boundary_nodes("upper_bottom")][0] - u[mesh.boundary_nodes("lower_top")][0]
        return opening, solid.reaction_force("top")["y"]

    whole, whole_solid = zone()
    whole.solve_transient(end_time=1.5, **steps)
    stopped, stopped_solid = zone()
    stopped.solve_transient(
        end_time=1.0,
        output=dualmesh.Output(directory=str(directory), file_base="zone", checkpoint_interval=10),
        **steps,
    )
    reached, _ = opening_and_force(stopped, stopped_solid)
    restarted, restarted_solid = zone()
    result = restarted.solve_transient(end_time=1.5, restart=str(directory / "zone.chk"), **steps)
    assert result.time_steps == 25
    assert restarted.values("disp_y") == pytest.approx(whole.values("disp_y"), rel=1e-10, abs=1e-18)
    opening, force = opening_and_force(restarted, restarted_solid)
    assert force == pytest.approx(opening_and_force(whole, whole_solid)[1], rel=1e-10)
    # The zone was softening at the stop and unloads along the secant of the opening reached.
    assert strength / stiffness < reached < 2 * fracture_toughness / strength
    assert 0.0 < opening < reached
    assert force == pytest.approx(bilinear(opening, reached, stiffness), rel=1e-8)
    assert bilinear(opening, opening, stiffness) > 1.15 * force
    shutil.rmtree(directory, ignore_errors=True)


def test_an_axisymmetric_zone_carries_the_traction_of_its_spring_in_series_with_the_bulk():
    """Two coaxial cylinders of radius R = 1 bonded end to end, the lower one on rollers at
    z = 0, held at r = 0 on the axis, and the top pulled up by u = 1e-5, below the strength.  The
    bulk is in uniaxial stress sigma, with the strain sigma / E over the height H = 1, and the
    zone, a spring of stiffness K = 50 E / h with h = 1/5 the side of the secondary surface,
    opens by sigma / K, so u = sigma (H / E + 1 / K), springs in series.  The radial contraction
    nu sigma r / E is the same in both cylinders, so the zone does not shear.  Each body carries
    the force sigma pi R^2 through the zone, which the factor 2 pi r of the axisymmetric measure
    gives."""
    mesh = two_blocks()
    problem = dualmesh.Problem(mesh, coordinates="axisymmetric")
    solid = problem.add_physics("solid_mechanics", youngs_modulus=E, poissons_ratio=nu)
    solid.add_boundary_condition(
        "cohesive",
        primary="lower_top",
        secondary="upper_bottom",
        strength=strength,
        toughness=fracture_toughness,
    )
    solid.add_boundary_condition("disp_z = 0", boundary="bottom")
    solid.add_boundary_condition("disp_r = 0", boundary="left")
    solid.add_boundary_condition("disp_z = 1e-5", boundary="top")
    problem.solve(report="none")
    stiffness = 50 * E / 0.2
    sigma = 1e-5 / (1 / E + 1 / stiffness)
    assert sigma < strength
    assert problem.property_at_centroids("stress")[:, 1] == pytest.approx(sigma, rel=1e-8)
    u = problem.values("disp_z")
    assert u[mesh.boundary_nodes("lower_top")] == pytest.approx(0.5 * sigma / E, rel=1e-8)
    assert u[mesh.boundary_nodes("upper_bottom")] == pytest.approx(
        0.5 * sigma / E + sigma / stiffness, rel=1e-8
    )
    assert solid.reaction_force("top")["z"] == pytest.approx(sigma * numpy.pi, rel=1e-8)
    assert solid.reaction_force("upper_bottom")["z"] == pytest.approx(-sigma * numpy.pi, rel=1e-8)
    assert solid.reaction_force("lower_top")["z"] == pytest.approx(sigma * numpy.pi, rel=1e-8)


def test_a_zone_turned_with_its_bodies_resists_closing_with_its_full_stiffness():
    """The two bonded blocks, of neo-Hookean material with nu = 0, turned rigidly by 90 degrees
    about the origin in 20 steps by their bottom and top, and then the top pushed toward the
    lower block by delta = 1e-3 along the turned axis of the blocks.  A hyperelastic body is at
    finite strain, so the zone takes the normal of its deformed surface, and the push closes it
    along that normal, where it keeps the full stiffness K = 50 E / h, h = 1/5, and does not
    damage.  With nu = 0 the lateral stretch stays 1 and the stress is uniaxial along the axis,
    with the nominal stress P = mu (lambda - 1/lambda) of the stretch lambda (Bonet and Wood
    2008, Nonlinear Continuum Mechanics for Finite Element Analysis), so the closing g of the zone
    and lambda solve K g = mu (1/lambda - lambda) and delta = (1 - lambda) H + g, with H = 1 the
    height.  Measured along the undeformed normal, the same closing would be a slip of g, about
    4e-6, far beyond the opening strength / K = 1.2e-7 at which the zone starts to soften."""
    mesh = two_blocks()
    problem = dualmesh.Problem(mesh)
    solid = problem.add_physics(
        "solid_mechanics",
        formulation="plane_strain",
        youngs_modulus=E,
        poissons_ratio=0.0,
        elasticity="neo-Hookean",
    )
    solid.add_boundary_condition(
        "cohesive",
        primary="lower_top",
        secondary="upper_bottom",
        strength=strength,
        toughness=fracture_toughness,
    )
    # The angle rises to pi/2 at t = 1, and the push to 1e-3 at t = 1.1 along R (0, -1).
    angle = "(pi/2*(t - max(t - 1, 0)))"
    push = "(1e-2*max(t - 1, 0))"
    turn_x = f"(cos{angle} - 1)*x - sin{angle}*y"
    turn_y = f"sin{angle}*x + (cos{angle} - 1)*y"
    solid.add_boundary_condition([f"disp_x = {turn_x}", f"disp_y = {turn_y}"], boundary="bottom")
    solid.add_boundary_condition(
        [f"disp_x = {turn_x} + {push}*sin{angle}", f"disp_y = {turn_y} - {push}*cos{angle}"],
        boundary="top",
    )
    problem.solve_transient(end_time=1.1, time_step=0.05, time_stepper="fixed", report="none")
    stiffness, shear_modulus, delta = 50 * E / 0.2, E / 2, 1e-3

    def closing(stretch):
        return shear_modulus * (1 / stretch - stretch) / stiffness

    stretch = brentq(lambda s: 1 - s + closing(s) - delta, 0.99, 1.0, xtol=1e-16)
    force = stiffness * closing(stretch)
    # Turned by 90 degrees the lower block lies on -0.5 < x < 0 and the upper on -1 < x < -0.5:
    # the zone pushes the upper block along -x and the lower along +x.  The tolerance is that of
    # Newton's method on displacements of 1 m, against a closing of 4e-6 m.
    upper, lower = solid.reaction_force("upper_bottom"), solid.reaction_force("lower_top")
    assert upper["x"] == pytest.approx(-force, rel=1e-6)
    assert lower["x"] == pytest.approx(force, rel=1e-6)
    assert upper["y"] == pytest.approx(0.0, abs=1e-6 * force)
    # The closing g: the secondary surface lies g closer to the lower block than the primary.
    x = numpy.asarray(mesh.points())[:, 0] + problem.values("disp_x")
    assert x[mesh.boundary_nodes("upper_bottom")] - x[mesh.boundary_nodes("lower_top")][0] == (
        pytest.approx(closing(stretch), rel=1e-6)
    )


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
