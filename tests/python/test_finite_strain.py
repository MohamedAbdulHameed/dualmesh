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
  collocation solver);
* a strain energy given as a formula gives the stress of the closed form of
  the same law, and an exact Jacobian;
* a plane-stress strip thins by the closed-form stretch of its law.
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


# ---------------------------------------------------------------------------
# Strain energies given as formulas, and plane stress
# ---------------------------------------------------------------------------
neo_Hookean_energy = (
    "shear_modulus/2*(I_C - 3) - shear_modulus*log(J) + first_Lame_parameter/2*log(J)^2"
)
Saint_Venant_Kirchhoff_energy = (
    "first_Lame_parameter/8*(I_C - 3)^2 + shear_modulus/4*(trace_C_squared - 2*I_C + 3)"
)
# A Demiray energy with a chain-stretch term and the volumetric energy of Simo and Taylor, which
# reads exp, sqrt, log and powers. It is free of stress at F = I.
stiffening_energy = (
    "shear_modulus*(exp(0.5*(I_C - 3)) - 1) + shear_modulus*(sqrt(I_C/3) - 1)^2"
    " - shear_modulus*log(J) + first_Lame_parameter/4*(J^2 - 1 - 2*log(J))"
)


def _distort(p, names):
    """Sets a smooth displacement of up to about 10 per cent strain on the problem."""
    p.initialize()
    points = numpy.asarray(p.node_coordinates())
    x, y = points[:, 0], points[:, 1]
    z = points[:, 2] if points.shape[1] > 2 else numpy.zeros_like(x)
    shapes = (
        0.08 * numpy.sin(x + 2 * y) + 0.05 * x * y + 0.02 * z,
        0.06 * numpy.cos(2 * x - y) - 0.04 * x + 0.03 * y * z,
        0.05 * numpy.sin(x - z) + 0.04 * y,
    )
    for name, shape in zip(names, shapes):
        p.set_values(name, shape)


@pytest.mark.parametrize(
    "law, energy",
    [
        ("neo-Hookean", neo_Hookean_energy),
        ("Saint_Venant_Kirchhoff", Saint_Venant_Kirchhoff_energy),
    ],
)
def test_an_energy_given_as_a_formula_gives_the_stress_of_its_closed_form(law, energy):
    """At a distorted state, the stress of the energy differentiated by the code equals that of
    the closed form of the same law to round-off."""
    names = ["ux", "uy", "uz"]
    stresses = []
    for update in (
        {"stress_update": law},
        {"stress_update": "strain_energy", "strain_energy": energy},
    ):
        mesh = dualmesh.generate_box_mesh(0, 1, 0, 1, 0, 1, 2, 2, 2)
        p = _stress_problem(mesh, "finite_strain_stress", names, **update)
        _distort(p, names)
        stresses.append(numpy.asarray(p.property_at_centroids("first_piola_kirchhoff_stress")))
    closed, formula = stresses
    assert abs(closed).max() > 1e-2 * E
    assert abs(formula - closed).max() < 1e-10 * abs(closed).max()


def test_a_plane_stress_strip_of_Saint_Venant_Kirchhoff_thins_by_the_closed_form():
    """A strip stretched by a = 1.3 with free lateral faces: F = diag(a, s, s), with the lateral
    Green strain E_22 = -lambda E_11 / (2 (lambda + mu)) and s = sqrt(1 + 2 E_22).  The stretch of
    the thickness, the root of P_33 = 0, is s as well."""
    stretch = 1.3
    axial_strain = (stretch**2 - 1) / 2
    lateral = numpy.sqrt(1 - Lame_lambda * axial_strain / (Lame_lambda + shear_modulus))
    mesh = dualmesh.generate_rectangle_mesh(0, 2, 0, 1, 4, 2)
    problem = dualmesh.Problem(mesh)
    body = problem.add_physics(
        "solid_mechanics",
        formulation="plane_stress",
        youngs_modulus=E,
        poissons_ratio=nu,
        elasticity="Saint_Venant_Kirchhoff",
    )
    body.add_boundary_condition("disp_x = 0", boundary="left")
    body.add_boundary_condition("disp_y = 0", boundary="bottom")
    body.add_boundary_condition(f"disp_x = {2 * (stretch - 1)!r}", boundary="right")
    problem.solve(report="none")
    y = numpy.asarray(problem.node_coordinates())[:, 1]
    assert problem.values("disp_y") == pytest.approx((lateral - 1) * y, rel=1e-8, abs=1e-12)
    thickness = numpy.asarray(problem.property_at_centroids("deformation_gradient"))[:, 8]
    assert thickness == pytest.approx(lateral, rel=1e-8)


@pytest.mark.parametrize("formulation", ["plane_stress", "three_dimensional"])
def test_the_Jacobian_of_an_energy_given_as_a_formula_is_exact(formulation):
    """At a distorted state, under a follower pressure, the Jacobian of an energy that reads exp,
    sqrt, log and powers agrees with central differences of the residual.  It holds the second
    derivatives of the energy and, in plane stress, the derivative of the stretch of the
    thickness by the implicit function theorem."""
    from test_Jacobian import compare, finite_difference_Jacobian

    if formulation == "plane_stress":
        mesh = dualmesh.generate_rectangle_mesh(0, 2, 0, 1, 3, 2)
        names = ["ux", "uy"]
    else:
        mesh = dualmesh.generate_box_mesh(0, 1, 0, 1, 0, 1, 2, 2, 2)
        names = ["ux", "uy", "uz"]
    p = dualmesh.Problem(mesh)
    for name in names:
        p.add_variable(name)
    p.add_property(
        "finite_strain_stress",
        displacements=names,
        formulation=formulation,
        youngs_modulus=E,
        poissons_ratio=nu,
        stress_update="strain_energy",
        strain_energy=stiffening_energy,
    )
    for name in names:
        p.add_kernel(
            "stress_divergence",
            variable=name,
            displacements=names,
            stress_property="first_piola_kirchhoff_stress",
        )
    p.add_boundary_condition("Dirichlet", "left", variable="ux", value=0.0)
    p.add_boundary_condition("Dirichlet", "bottom", variable="uy", value=0.0)
    p.add_boundary_condition(
        "pressure",
        "right",
        displacements=names,
        pressure=0.01 * E,
        deformation_gradient_property="deformation_gradient",
    )
    _distort(p, names)
    _, Jacobian = p.linear_system()
    assert compare(Jacobian.toarray(), finite_difference_Jacobian(p)) < 1e-6
