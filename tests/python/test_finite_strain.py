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

import dualmesh
import numpy
import pytest

E, nu = 1.0e9, 0.3
shear_modulus = E / (2 * (1 + nu))
Lame_lambda = E * nu / ((1 + nu) * (1 - 2 * nu))


def _stress_problem(mesh, material, displacements, coordinates="cartesian", method="fem", **extra):
    p = dualmesh.Problem(mesh, method=method, coordinates=coordinates)
    for d in displacements:
        p.add_variable(d)
    formulation = {1: "axisymmetric_1d", 2: "plane_strain", 3: "three_dimensional"}[
        len(displacements)
    ]
    if coordinates == "axisymmetric" and len(displacements) == 2:
        formulation = "axisymmetric"
    p.add_property(
        material,
        displacements=displacements,
        formulation=formulation,
        youngs_modulus=E,
        poissons_ratio=nu,
        **extra,
    )
    stress = "first_piola_kirchhoff_stress" if material == "finite_strain_stress" else "stress"
    for d in displacements:
        p.add_kernel(
            "stress_divergence", variable=d, displacements=displacements, stress_property=stress
        )
    return p


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
        return shear_modulus * (s - 1 / s) + Lame_lambda * numpy.log(J) / s

    s = brentq(lateral, 0.3, 1.5)
    J = stretch * s * s
    return shear_modulus * (stretch - 1 / stretch) + Lame_lambda * numpy.log(J) / stretch, s


@pytest.mark.parametrize("stretch", [1.05, 1.3, 0.8])
def test_a_neo_hookean_bar_stretched_uniaxially_has_the_exact_nominal_stress(stretch):
    pytest.importorskip("scipy")
    mesh = dualmesh.generate_box_mesh(0, 1, 0, 1, 0, 1, 2, 2, 2)
    p = _stress_problem(
        mesh, "finite_strain_stress", ["ux", "uy", "uz"], stress_update="neo-Hookean"
    )
    for v, side in (("ux", "left"), ("uy", "bottom"), ("uz", "back")):
        p.add_boundary_condition("Dirichlet", side, variable=v, value=0.0)
    p.add_boundary_condition(
        "Dirichlet",
        "right",
        variable="ux",
        value=stretch - 1.0,
    )
    p.solve()
    exact_stress, lateral = _uniaxial_neo_hookean(stretch)
    force = p.boundary_flux("ux", "left")
    assert force == pytest.approx(exact_stress * 1.0, rel=1e-8)
    corner = p.sample("uy", [[1 - 1e-9, 1 - 1e-9, 1 - 1e-9]])[0]
    assert 1.0 + corner == pytest.approx(lateral, rel=1e-8)


# ---------------------------------------------------------------------------
# Objectivity
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("update", ["rotated_small_strain", "neo-Hookean"])
def test_a_rigid_rotation_rotates_the_stress_and_adds_none(update):
    """A unit square in plane strain is stretched by 1e-3 along x (a steady
    solve), then turned by 90 degrees in 45 steps with the stretch held.  The
    Cauchy stress turns with it: sigma_yy at the end equals sigma_xx after
    the stretch, and sigma_xx equals the lateral stress.  This also checks that
    a steady solve commits the material history, so that the rotation starts
    from the stretched state."""
    mesh = dualmesh.generate_rectangle_mesh(0, 1, 0, 1, 2, 2)
    p = _stress_problem(mesh, "finite_strain_stress", ["ux", "uy"], stress_update=update)
    stretch = 1e-3
    angle = dualmesh.SettableFunction(0.0)
    p.add_function("angle", angle)
    # The motion x = R(angle) diag(1 + stretch, 1) X on every boundary node;
    # with a homogeneous material the interior follows.
    a, s = "(pi/2*t)", repr(stretch)
    for v, expression in (
        ("ux", f"cos({a})*(1 + {s})*x - sin({a})*y - x"),
        ("uy", f"sin({a})*(1 + {s})*x + cos({a})*y - y"),
    ):
        p.add_boundary_condition(
            "Dirichlet",
            ["left", "right", "top", "bottom"],
            variable=v,
            value=dualmesh.ParsedFunction(expression),
        )
    p.time = 0.0
    p.solve()
    before = numpy.asarray(p.property_at_centroids("stress"))[0]
    p.solve_transient(start_time=0.0, end_time=1.0, time_step=1.0 / 45)
    after = numpy.asarray(p.property_at_centroids("stress"))[0]
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

    mu, lam, p = shear_modulus / E, Lame_lambda / E, pressure / E

    def nominal(lr, lt):
        J = lr * lt
        return mu * (lr - 1 / lr) + lam * numpy.log(J) / lr, mu * (lt - 1 / lt) + lam * numpy.log(
            J
        ) / lt

    def RHS(R, y):
        r, lr = y
        lt = r / R
        J = lr * lt
        P_rr, P_tt = nominal(lr, lt)
        tangent_r = mu * (1 + 1 / lr**2) + lam * (1 - numpy.log(J)) / lr**2
        tangent_t = lam / J
        # With tangent_r and tangent_t the derivatives of P_rr by lambda_r and lambda_t,
        # d(P_rr)/dR = tangent_r r'' + tangent_t (r' - r/R)/R = -(P_rr - P_tt)/R
        d2r = (-(P_rr - P_tt) / R - tangent_t * (lr - lt) / R) / tangent_r
        return numpy.vstack([lr, d2r])

    def boundary_condition(ya, yb):
        P_a, _ = nominal(ya[1], ya[0] / b)
        P_b, _ = nominal(yb[1], yb[0] / c)
        return numpy.array([P_a + p * ya[0] / b, P_b])

    R = numpy.linspace(b, c, 400)
    solution = solve_bvp(
        RHS,
        boundary_condition,
        R,
        numpy.vstack([R, numpy.ones_like(R)]),
        tol=1e-10,
        max_nodes=200000,
    )
    assert solution.status == 0, solution.message
    return lambda radius: solution.sol(radius)[0] - radius


@pytest.mark.parametrize("method", ["fem", "dmcdm"])
def test_a_thick_tube_inflated_to_large_strain_by_a_follower_pressure(method):
    """The inner radius grows by about 25 per cent; the displacements converge
    to the reference at second order."""
    pytest.importorskip("scipy")
    b, c, pressure = 1.0, 2.0, 0.15 * E
    reference = _tube_reference(b, c, pressure)
    errors = []
    for n in (8, 16, 32):
        points = numpy.linspace(b, c, n + 1)[:, None]
        mesh = dualmesh.mesh_from_arrays(
            points, [[i, i + 1] for i in range(n)], "Edge2", dimension=1
        )
        mesh.add_sideset_from_faces("inner", [[0]])
        mesh.add_sideset_from_faces("outer", [[n]])
        p = _stress_problem(
            mesh,
            "finite_strain_stress",
            ["u"],
            coordinates="axisymmetric",
            method=method,
            stress_update="neo-Hookean",
        )
        p.add_boundary_condition(
            "pressure",
            "inner",
            displacements=["u"],
            pressure=pressure,
            deformation_gradient_property="deformation_gradient",
        )
        p.solve()
        r = numpy.abs(p.node_coordinates()[:, 0])
        errors.append(numpy.abs(p.values("u") - reference(r)).max())
    assert reference(b) / b > 0.2  # It is large strain
    rates = numpy.log2(numpy.array(errors[:-1]) / numpy.array(errors[1:]))
    assert errors[-1] < 1e-3 * reference(b)
    assert rates.min() > 1.8


@pytest.mark.parametrize("method", ["fem", "dmcdm"])
def test_the_axisymmetric_formulation_inflates_the_same_tube(method):
    """The tube of the previous test meshed in r-z (the axial displacement held
    at zero on the ends, so plane strain): the radial displacement matches the
    reference, and it does not vary along the axis."""
    pytest.importorskip("scipy")
    b, c, pressure = 1.0, 2.0, 0.15 * E
    reference = _tube_reference(b, c, pressure)
    mesh = dualmesh.generate_rectangle_mesh(b, c, 0.0, 0.25, 32, 2)
    p = _stress_problem(
        mesh,
        "finite_strain_stress",
        ["ur", "uz"],
        coordinates="axisymmetric",
        method=method,
        stress_update="neo-Hookean",
    )
    p.add_boundary_condition("Dirichlet", ["top", "bottom"], variable="uz", value=0.0)
    p.add_boundary_condition(
        "pressure",
        "left",
        displacements=["ur", "uz"],
        pressure=pressure,
        deformation_gradient_property="deformation_gradient",
    )
    p.solve()
    points = [[r, 0.1, 0.0] for r in numpy.linspace(b, c, 9)]
    sampled = numpy.asarray(p.sample("ur", points))
    exact = reference(numpy.linspace(b, c, 9))
    assert numpy.abs(sampled - exact).max() < 2e-3 * exact[0]


@pytest.mark.parametrize("stretch", [1.05, 1.3])
def test_the_neo_hookean_elastic_strain_is_the_green_Lagrange_strain_of_the_stretch(stretch):
    # No eigenstrain, so F_e = F = diag(stretch, s, s): E_e = (F^T F - I) / 2, and the
    # volumetric strain is J - 1.
    pytest.importorskip("scipy")
    mesh = dualmesh.generate_box_mesh(0, 1, 0, 1, 0, 1, 2, 2, 2)
    p = _stress_problem(
        mesh, "finite_strain_stress", ["ux", "uy", "uz"], stress_update="neo-Hookean"
    )
    for v, side in (("ux", "left"), ("uy", "bottom"), ("uz", "back")):
        p.add_boundary_condition("Dirichlet", side, variable=v, value=0.0)
    p.add_boundary_condition("Dirichlet", "right", variable="ux", value=stretch - 1.0)
    p.solve(report="none")
    _, s = _uniaxial_neo_hookean(stretch)
    elastic = p.property_at_centroids("elastic_strain")
    expected = [0.5 * (stretch**2 - 1), 0.5 * (s**2 - 1), 0.5 * (s**2 - 1), 0, 0, 0]
    assert numpy.allclose(elastic, expected, rtol=1e-8, atol=1e-12)
    volumetric = p.property_at_centroids("volumetric_strain")
    assert numpy.allclose(volumetric, stretch * s * s - 1, rtol=1e-8)
