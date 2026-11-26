# SPDX-License-Identifier: LGPL-2.1-or-later
"""Fracture mechanics: the J-integral and the stress intensity factors of a crack tip."""

from __future__ import annotations

import dualmesh
import numpy
import pytest

E, nu = 200e9, 0.3
mu = E / (2 * (1 + nu))
kappa = 3 - 4 * nu  # Plane strain


def cracked_square(n=16):
    """The square [-1, 1]^2 with a crack along y = 0 from x = -1 to the tip at the origin: two
    halves that share their nodes ahead of the tip only."""
    lower = dualmesh.generate_rectangle_mesh(-1, 1, -1, 0, n, n // 2, element_type="Quad9")
    upper = dualmesh.generate_rectangle_mesh(-1, 1, 0, 1, n, n // 2, element_type="Quad9")
    points = [numpy.asarray(lower.points())[:, :2], numpy.asarray(upper.points())[:, :2]]
    shared = {}
    for k, (x, y) in enumerate(points[0]):
        if abs(y) < 1e-12 and x > -1e-12:
            shared[round(x, 9)] = k
    xy = list(points[0])
    index = []
    for x, y in points[1]:
        if abs(y) < 1e-12 and round(x, 9) in shared:
            index.append(shared[round(x, 9)])
        else:
            index.append(len(xy))
            xy.append(numpy.array([x, y]))
    connectivity = [list(lower.element_nodes(e)) for e in range(lower.num_elements)]
    connectivity += [[index[n] for n in upper.element_nodes(e)] for e in range(upper.num_elements)]
    mesh = dualmesh.mesh_from_arrays(
        numpy.array(xy), [("Quad9", numpy.array(connectivity))], dimension=2
    )
    # The outer boundary in two halves: the polar angle of the lower one runs to -pi on its
    # crack face, where atan2 gives +pi.
    for name, side in (("outer_upper", 1.0), ("outer_lower", -1.0)):
        mesh.add_sideset_by_predicate(
            name,
            lambda x, y, z, side=side: (
                (abs(abs(x) - 1) < 1e-12 or abs(abs(y) - 1) < 1e-12) and side * y > 0
            ),
        )
    return mesh


def williams(K_I, K_II, lower=False):
    """The displacement texts of the Williams field of K_I and K_II (plane strain), with the
    angle of the lower half from -pi."""
    r, t = "sqrt(x^2 + y^2)", "(-atan2(-y, x))" if lower else "atan2(y, x)"
    f = f"sqrt({r}/(2*pi))/(2*{mu!r})"
    ch, sh = f"cos({t}/2)", f"sin({t}/2)"
    ux = f"{K_I}*{f}*{ch}*({kappa} - 1 + 2*{sh}^2) + {K_II}*{f}*{sh}*({kappa} + 1 + 2*{ch}^2)"
    uy = f"{K_I}*{f}*{sh}*({kappa} + 1 - 2*{ch}^2) - {K_II}*{f}*{ch}*({kappa} - 1 - 2*{sh}^2)"
    return ux, uy


@pytest.mark.parametrize("K_I, K_II", [(1e6, 0.0), (0.0, 1e6), (1e6, 5e5)])
def test_the_stress_intensity_of_an_imposed_williams_field_is_recovered(K_I, K_II):
    """The Williams displacements of K_I and K_II on the boundary of a cracked square: the
    interaction integral returns them and J = (K_I^2 + K_II^2)/E', on two rings, converging
    with the mesh: without singular elements at the tip its r^(1/2) singularity pollutes the
    solution, so the error falls at about first order (measured at 32 and 64 elements a side:
    a spurious mode of 0.96 % and 0.58 % of the applied one, and 2 % falling to 1.4 % on K_II of
    pure mode II)."""
    plane = E / (1 - nu**2)
    scale = numpy.hypot(K_I, K_II)
    errors = []
    for n in (32, 64):
        problem = dualmesh.Problem(cracked_square(n))
        solid = problem.add_physics(
            "solid_mechanics", formulation="plane_strain", youngs_modulus=E, poissons_ratio=nu
        )
        for half, lower in (("outer_upper", False), ("outer_lower", True)):
            ux, uy = williams(K_I, K_II, lower)
            solid.add_boundary_condition([f"disp_x = {ux}", f"disp_y = {uy}"], boundary=half)
        problem.solve(report="none")
        worst = 0.0
        for radii in ([0.2, 0.5], [0.3, 0.8]):
            K = solid.stress_intensity(crack_tip=[0, 0], radii=radii)
            J = solid.j_integral(crack_tip=[0, 0], radii=radii)
            worst = max(worst, abs(K["I"] - K_I) / scale, abs(K["II"] - K_II) / scale)
            worst = max(worst, abs(J * plane / scale**2 - 1) / 2)
        errors.append(worst)
    assert errors[1] < 1.5e-2 and errors[1] < 0.8 * errors[0]


def test_phase_field_damage_of_a_uniform_bar_is_the_closed_form_and_does_not_heal():
    """A bar pulled uniformly (nu = 0) before its peak stress: the damage is uniform, d = 2H /
    (Gc/l + 2H) with H = E eps^2 / 2, and sigma = (1 - d)^2 E eps, exactly; unloaded, the bar keeps
    its damage (the history does not fall) and carries (1 - d)^2 E eps on the way down."""
    e1, G_c, length = 1e9, 100.0, 0.01
    mesh = dualmesh.generate_rectangle_mesh(0, 1, 0, 1, 2, 2)
    problem = dualmesh.Problem(mesh)
    solid = problem.add_physics(
        "solid_mechanics",
        formulation="plane_strain",
        youngs_modulus=e1,
        poissons_ratio=0.0,
        phase_field_fracture={"toughness": G_c, "length": length},
    )
    solid.add_boundary_condition("disp_x = 0", boundary="left")
    solid.add_boundary_condition("disp_y = 0", boundary="bottom")
    # Up to the strain 2e-3, on the rising branch where the uniform state is stable, then back to
    # 1e-3.
    solid.add_boundary_condition("disp_x = 1e-3*(t - 2*max(t - 2, 0))", boundary="right")
    records = []
    problem.set_time_step_callback(
        lambda t, core: records.append(
            (t, problem.values("damage").copy(), problem.property_at_centroids("stress")[0][0])
        )
    )
    problem.solve_transient(end_time=2.5, time_step=0.5, time_stepper="fixed", report="none")
    peak_damage = None
    for t, damage, stress in records:
        strain = 1e-3 * (t - 2 * max(t - 2, 0))
        if t <= 2.0 + 1e-12:
            H = e1 * strain**2 / 2
            expected = 2 * H / (G_c / length + 2 * H)
            peak_damage = expected
        else:
            expected = peak_damage
        assert damage == pytest.approx(expected, rel=1e-9)
        assert stress == pytest.approx((1 - expected) ** 2 * e1 * strain, rel=1e-9)
