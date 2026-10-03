# SPDX-License-Identifier: LGPL-2.1-or-later
"""Finite strain on the undeformed mesh (total Lagrangian):
finite_strain_stress, the nine-component stress_divergence and the follower
pressure.

Each test checks one property against something known independently:

* at small strain the finite-strain stress is the small-strain stress;
* a homogeneous uniaxial stretch of a neo-Hookean bar gives the closed-form
  nominal stress, for any stretch;
* a rigid rotation produces no stress, and rotates an existing stress with
  the body (objectivity);
* a thick tube inflated by a follower pressure to large strain matches an
  independent solution of the same boundary value problem (SciPy's
  collocation solver).
"""

from __future__ import annotations

import dualmesh as dm
import numpy as np
import pytest

E, NU = 1.0e9, 0.3
MU = E / (2 * (1 + NU))
LAMBDA = E * NU / ((1 + NU) * (1 - 2 * NU))


def _stress_problem(mesh, material, displacements, coordinates="cartesian", method="fem", **extra):
    p = dm.Problem(mesh, method=method, coordinates=coordinates)
    for d in displacements:
        p.add_variable(d)
    formulation = {1: "axisymmetric_1d", 2: "plane_strain", 3: "three_dimensional"}[len(displacements)]
    if coordinates == "axisymmetric" and len(displacements) == 2:
        formulation = "axisymmetric"
    p.add_property(material, "stress", displacements=displacements, formulation=formulation, youngs_modulus=E, poissons_ratio=NU, **extra)
    stress = "first_piola_kirchhoff_stress" if material == "finite_strain_stress" else "stress"
    for i, d in enumerate(displacements):
        p.add_kernel("stress_divergence", f"equilibrium_{d}", variable=d, component=i, stress_property=stress)
    return p


# ---------------------------------------------------------------------------
# Small strain limit
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("method", ["fem", "dmcdm", "hfvm", "zfvm"])
def test_at_small_strain_the_finite_strain_stress_is_the_small_strain_stress(method):
    """A thick tube under a pressure of 1e-5 E: strains of order 1e-5, so
    the two formulations agree to about 1e-5 relative."""
    b, c, pressure = 1.0, 2.0, 1e-5 * E
    n = 16
    points = np.linspace(b, c, n + 1)[:, None]
    mesh = dm.mesh_from_arrays(points, [[i, i + 1] for i in range(n)], "Edge2", dimension=1)
    mesh.add_sideset_from_faces("inner", [[0]])
    mesh.add_sideset_from_faces("outer", [[n]])
    results = []
    for material in ("small_strain_stress", "finite_strain_stress"):
        p = _stress_problem(mesh, material, ["u"], coordinates="axisymmetric", method=method)
        p.add_boundary_condition("pressure_boundary_condition", "inside", variable="u", component=0, boundary=["inner"], pressure=pressure)
        p.solve()
        results.append(np.asarray(p.values("u")))
    small, finite = results
    assert finite == pytest.approx(small, rel=5e-5)


# ---------------------------------------------------------------------------
# Homogeneous stretch
# ---------------------------------------------------------------------------
def _uniaxial_neo_hookean(stretch):
    """The nominal stress P11 of a compressible neo-Hookean bar stretched by
    `stretch` with free lateral faces: the lateral stretch s solves
    P22 = mu (s - 1/s) + lambda ln(J) / s = 0 with J = stretch s^2."""
    from scipy.optimize import brentq

    def lateral(s):
        J = stretch * s * s
        return MU * (s - 1 / s) + LAMBDA * np.log(J) / s

    s = brentq(lateral, 0.3, 1.5)
    J = stretch * s * s
    return MU * (stretch - 1 / stretch) + LAMBDA * np.log(J) / stretch, s


@pytest.mark.parametrize("stretch", [1.05, 1.3, 0.8])
def test_a_neo_hookean_bar_stretched_uniaxially_has_the_exact_nominal_stress(stretch):
    pytest.importorskip("scipy")
    mesh = dm.generate_box_mesh(0, 1, 0, 1, 0, 1, 2, 2, 2)
    p = _stress_problem(mesh, "finite_strain_stress", ["ux", "uy", "uz"], stress_update="neo_Hookean")
    for v, side in (("ux", "left"), ("uy", "bottom"), ("uz", "back")):
        p.add_boundary_condition("Dirichlet_boundary_condition", f"hold_{v}", variable=v, boundary=[side], value=0.0)
    p.add_boundary_condition("Dirichlet_boundary_condition", "pull", variable="ux", boundary=["right"], value=stretch - 1.0)
    p.solve(load_factors=[0.25, 0.5, 0.75, 1.0])
    exact_stress, lateral = _uniaxial_neo_hookean(stretch)
    force = -p.total_reaction("ux", "left")
    assert force == pytest.approx(exact_stress * 1.0, rel=1e-8)
    corner = p.sample("uy", [[1 - 1e-9, 1 - 1e-9, 1 - 1e-9]])[0]
    assert 1.0 + corner == pytest.approx(lateral, rel=1e-8)


# ---------------------------------------------------------------------------
# Objectivity
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("update", ["rotated_small_strain", "neo_Hookean"])
def test_a_rigid_rotation_rotates_the_stress_and_adds_none(update):
    """A unit square in plane strain is stretched by 1e-3 along x (a steady
    solve), then turned by 90 degrees in 45 steps with the stretch held.  The
    Cauchy stress turns with it: sigma_yy at the end equals sigma_xx after
    the stretch, and sigma_xx equals the lateral stress.  This also checks that
    a steady solve commits the material history, so that the rotation starts
    from the stretched state."""
    mesh = dm.generate_rectangle_mesh(0, 1, 0, 1, 2, 2)
    p = _stress_problem(mesh, "finite_strain_stress", ["ux", "uy"], stress_update=update)
    stretch = 1e-3
    angle = dm.SettableFunction(0.0)
    p.add_function("angle", angle)
    # The motion x = R(angle) diag(1 + stretch, 1) X on every boundary node;
    # with a homogeneous material the interior follows.
    a, s = "(pi/2*t)", repr(stretch)
    for v, expression in (("ux", f"cos({a})*(1 + {s})*x - sin({a})*y - x"), ("uy", f"sin({a})*(1 + {s})*x + cos({a})*y - y")):
        p.add_boundary_condition("Dirichlet_boundary_condition", f"motion_{v}", variable=v, boundary=["left", "right", "top", "bottom"], value=dm.ParsedFunction(expression), scale_with_load=False)
    p.time = 0.0
    p.solve()
    before = np.asarray(p.property_at_centroids("stress"))[0]
    p.solve_transient(start_time=0.0, end_time=1.0, time_step=1.0 / 45)
    after = np.asarray(p.property_at_centroids("stress"))[0]
    # Voigt order xx, yy, zz, yz, xz, xy.  The Hughes-Winget update is exact
    # for a rigid rotation (its Cayley transform returns the angle of each
    # increment exactly), so the stresses swap to rounding.
    assert after[1] == pytest.approx(before[0], rel=1e-9)
    assert after[0] == pytest.approx(before[1], rel=1e-9)
    assert abs(after[5]) < 1e-9 * abs(before[0])


# ---------------------------------------------------------------------------
# Large inflation of a thick tube by a follower pressure
# ---------------------------------------------------------------------------
def _tube_reference(b, c, pressure):
    """Plane strain inflation of a compressible neo-Hookean tube, solved
    independently: radial equilibrium on the undeformed radius R,
    d(P_rR)/dR + (P_rR - P_tT)/R = 0, with the stretches lambda_r = r'(R),
    lambda_t = r/R, lambda_z = 1, the nominal stresses
    P_rR = mu (lambda_r - 1/lambda_r) + Lambda ln(J)/lambda_r (and P_tT likewise)
    and J = lambda_r lambda_t.  The deformed inner surface carries the pressure:
    P_rR = -p r/R there (Nanson's relation), and the outer one is free.
    Stresses are scaled by E."""
    from scipy.integrate import solve_bvp

    mu, lam, p = MU / E, LAMBDA / E, pressure / E

    def nominal(lr, lt):
        J = lr * lt
        return mu * (lr - 1 / lr) + lam * np.log(J) / lr, mu * (lt - 1 / lt) + lam * np.log(J) / lt

    def rhs(R, y):
        r, lr = y
        lt = r / R
        J = lr * lt
        Prr, Ptt = nominal(lr, lt)
        dP_dlr = mu * (1 + 1 / lr**2) + lam * (1 - np.log(J)) / lr**2
        dP_dlt = lam / J
        # d(Prr)/dR = dP_dlr r'' + dP_dlt (r' - r/R)/R = -(Prr - Ptt)/R
        d2r = (-(Prr - Ptt) / R - dP_dlt * (lr - lt) / R) / dP_dlr
        return np.vstack([lr, d2r])

    def bc(ya, yb):
        Pa, _ = nominal(ya[1], ya[0] / b)
        Pb, _ = nominal(yb[1], yb[0] / c)
        return np.array([Pa + p * ya[0] / b, Pb])

    R = np.linspace(b, c, 400)
    solution = solve_bvp(rhs, bc, R, np.vstack([R, np.ones_like(R)]), tol=1e-10, max_nodes=200000)
    assert solution.status == 0, solution.message
    return lambda radius: solution.sol(radius)[0] - radius


@pytest.mark.parametrize("method", ["fem", "dmcdm", "hfvm", "zfvm"])
def test_a_thick_tube_inflated_to_large_strain_by_a_follower_pressure(method):
    """The inner radius grows by about 25 per cent; the displacements converge
    to the reference at second order."""
    pytest.importorskip("scipy")
    b, c, pressure = 1.0, 2.0, 0.15 * E
    reference = _tube_reference(b, c, pressure)
    errors = []
    for n in (8, 16, 32):
        points = np.linspace(b, c, n + 1)[:, None]
        mesh = dm.mesh_from_arrays(points, [[i, i + 1] for i in range(n)], "Edge2", dimension=1)
        mesh.add_sideset_from_faces("inner", [[0]])
        mesh.add_sideset_from_faces("outer", [[n]])
        p = _stress_problem(mesh, "finite_strain_stress", ["u"], coordinates="axisymmetric", method=method, stress_update="neo_Hookean")
        p.add_boundary_condition("pressure_boundary_condition", "inside", variable="u", component=0, boundary=["inner"], pressure=pressure, deformation_gradient_property="deformation_gradient")
        p.solve(load_factors=list(np.linspace(0.1, 1.0, 10)))
        r = np.abs(p.entity_points()[:, 0])
        errors.append(np.abs(p.values("u") - reference(r)).max())
    assert reference(b) / b > 0.2  # it is large strain
    rates = np.log2(np.array(errors[:-1]) / np.array(errors[1:]))
    assert errors[-1] < 1e-3 * reference(b)
    assert rates.min() > 1.8


@pytest.mark.parametrize("method", ["fem", "dmcdm", "zfvm"])
def test_the_axisymmetric_formulation_inflates_the_same_tube(method):
    """The tube of the previous test meshed in r-z (the axial displacement held
    at zero on the ends, so plane strain): the radial displacement matches the
    reference, and it does not vary along the axis."""
    pytest.importorskip("scipy")
    b, c, pressure = 1.0, 2.0, 0.15 * E
    reference = _tube_reference(b, c, pressure)
    mesh = dm.generate_rectangle_mesh(b, c, 0.0, 0.25, 32, 2)
    p = _stress_problem(mesh, "finite_strain_stress", ["ur", "uz"], coordinates="axisymmetric", method=method, stress_update="neo_Hookean")
    p.add_boundary_condition("Dirichlet_boundary_condition", "ends", variable="uz", boundary=["top", "bottom"], value=0.0)
    for i, v in enumerate(("ur", "uz")):
        p.add_boundary_condition("pressure_boundary_condition", f"inside_{v}", variable=v, component=i, boundary=["left"], pressure=pressure, deformation_gradient_property="deformation_gradient")
    p.solve(load_factors=list(np.linspace(0.1, 1.0, 10)))
    points = [[r, 0.1, 0.0] for r in np.linspace(b, c, 9)]
    sampled = np.asarray(p.sample("ur", points))
    exact = reference(np.linspace(b, c, 9))
    assert np.abs(sampled - exact).max() < 2e-3 * exact[0]
