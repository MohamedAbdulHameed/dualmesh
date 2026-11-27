# SPDX-License-Identifier: LGPL-2.1-or-later
"""Verification of the plate models against Chapters 8 and 10 of Reddy's book.

Chapter 8: axisymmetric circular plates, radius a = 10 in, thickness
h = 0.1 in, E1 = 30e6 psi, E2 = 3e6 psi, nu = 0.3, K_s = 5/6, q0 = 0.5 lb/in.

Chapter 10: rectangular plates, a = b = 1 in, h = 0.01 in, E1 = 3e7 psi,
E2 = 3e6 psi, nu = 0.3, q = 1 lb/in^2, modelled on one quarter of the plate.
"""

from __future__ import annotations

import dualmesh
import numpy
import pytest

# ---------------------------------------------------------------------------
# Chapter 8: axisymmetric circular plates
# ---------------------------------------------------------------------------
circular_radius = 10.0
circular_thickness = 0.1
circular_load = 0.5
modulus_top = 30.0e6
modulus_bottom = 3.0e6
Poisson_ratio = 0.3


def circular_plate_stiffness(power_law_index):
    return dualmesh.fgm.beam_stiffness(
        modulus_top=modulus_top,
        modulus_bottom=modulus_bottom,
        power_law_index=power_law_index,
        height=circular_thickness,
        width=1.0,
        poissons_ratio=Poisson_ratio,
        plate=True,
    )


def circular_plate(num_elements, power_law_index=0.0, support="hinged"):
    """Axisymmetric circular plate, first-order shear deformation theory."""
    s = circular_plate_stiffness(power_law_index)
    mesh = dualmesh.generate_line_mesh(start=0.0, end=circular_radius, num_elements=num_elements)
    problem = dualmesh.Problem(mesh, method="DMCDM", coordinates="axisymmetric")
    for name in ("radial_displacement", "deflection", "rotation"):
        problem.add_variable(name)
    common = dict(
        radial_displacement="radial_displacement",
        transverse_displacement="deflection",
        rotation="rotation",
        extensional_stiffness=s.extensional,
        coupling_stiffness=s.coupling,
        bending_stiffness=s.bending,
        shear_stiffness=s.shear,
        poissons_ratio=Poisson_ratio,
        transverse_load=circular_load,
    )
    for variable in ("radial_displacement", "deflection", "rotation"):
        problem.add_kernel(
            "circular_plate_first_order",
            variable=variable,
            shear_treatment="exclude",
            **common,
        )
        problem.add_kernel(
            "circular_plate_first_order",
            variable=variable,
            shear_treatment="only",
            quadrature="midpoint",
            reduced_integration=True,
            **common,
        )
    # Symmetry at r = 0
    problem.add_boundary_condition(
        "Dirichlet",
        "left",
        variable="radial_displacement",
        value=0.0,
    )
    problem.add_boundary_condition(
        "Dirichlet",
        "left",
        variable="rotation",
        value=0.0,
    )
    # Edge conditions at r = a
    problem.add_boundary_condition(
        "Dirichlet",
        "right",
        variable="radial_displacement",
        value=0.0,
    )
    problem.add_boundary_condition("Dirichlet", "right", variable="deflection", value=0.0)
    if support == "clamped":
        problem.add_boundary_condition(
            "Dirichlet",
            "right",
            variable="rotation",
            value=0.0,
        )
    problem.solve()
    return problem


def hinged_exact_deflection(power_law_index=0.0):
    """Reddy, Eq. (8.6.4b): centre deflection of a hinged FGM circular plate."""
    s = circular_plate_stiffness(power_law_index)
    reduced = s.bending * s.extensional - s.coupling**2
    a, q = circular_radius, circular_load
    bending_part = (
        s.extensional * q * a**4 / (64 * reduced) * (5 + Poisson_ratio) / (1 + Poisson_ratio)
    )
    shear_part = q * a**2 / (4 * s.shear)
    return bending_part + shear_part


def clamped_exact_deflection(power_law_index=0.0):
    """Reddy, Eq. (8.6.6b): centre deflection of a clamped FGM circular plate."""
    s = circular_plate_stiffness(power_law_index)
    reduced = s.bending * s.extensional - s.coupling**2
    a, q = circular_radius, circular_load
    return s.extensional * q * a**4 / (64 * reduced) + q * a**2 / (4 * s.shear)


def test_table_8_6_1_hinged_homogeneous_circular_plate():
    """Table 8.6.1, DM-FS(D) column: w(0) for 2, 4, 8, 16, 32 elements."""
    expected = [0.1052, 0.1134, 0.1153, 0.1158, 0.1159]
    for num_elements, reference in zip((2, 4, 8, 16, 32), expected):
        problem = circular_plate(num_elements)
        assert problem.values("deflection")[0] == pytest.approx(reference, abs=1.1e-4)
    assert hinged_exact_deflection() == pytest.approx(0.1159, abs=1.1e-4)


@pytest.mark.parametrize(
    "power_law_index, expected",
    [
        # Table 8.6.2, DM-FP(D) column (32 elements)
        (0.0, 0.0284),
        (1.0, 0.0665),
        (2.0, 0.0976),
        (3.0, 0.1152),
        (4.0, 0.1244),
        (5.0, 0.1296),
        (7.5, 0.1370),
        (10.0, 0.1425),
        (15.0, 0.1527),
        (20.0, 0.1622),
    ],
)
def test_table_8_6_2_clamped_functionally_graded_circular_plate(power_law_index, expected):
    problem = circular_plate(32, power_law_index=power_law_index, support="clamped")
    centre = problem.values("deflection")[0]
    assert centre == pytest.approx(expected, abs=1.1e-4)
    assert centre == pytest.approx(clamped_exact_deflection(power_law_index), rel=5e-3)


# ---------------------------------------------------------------------------
# Chapter 10: rectangular plates
# ---------------------------------------------------------------------------
plate_side = 1.0
plate_thickness = 0.01
plate_load = 1.0


def rectangular_plate_stiffness(power_law_index):
    return dualmesh.fgm.beam_stiffness(
        modulus_top=3.0e7,
        modulus_bottom=3.0e6,
        power_law_index=power_law_index,
        height=plate_thickness,
        width=1.0,
        poissons_ratio=Poisson_ratio,
        plate=True,
    )


def rectangular_plate(
    num_elements,
    power_law_index=0.0,
    support="clamped",
    method="dmcdm",
):
    """Quarter model of a rectangular plate, first-order shear deformation theory."""
    s = rectangular_plate_stiffness(power_law_index)
    mesh = dualmesh.generate_rectangle_mesh(
        x_min=0.0,
        x_max=plate_side / 2,
        y_min=0.0,
        y_max=plate_side / 2,
        num_x_elements=num_elements,
        num_y_elements=num_elements,
    )
    problem = dualmesh.Problem(mesh, method=method)
    variables = ["u", "v", "w", "rotation_x", "rotation_y"]
    for name in variables:
        problem.add_variable(name)
    common = dict(
        in_plane_displacements=["u", "v"],
        transverse_displacement="w",
        rotations=["rotation_x", "rotation_y"],
        extensional_stiffness=s.extensional,
        coupling_stiffness=s.coupling,
        bending_stiffness=s.bending,
        shear_stiffness=s.shear,
        poissons_ratio=Poisson_ratio,
        transverse_load=plate_load,
    )
    for variable in variables:
        problem.add_kernel(
            "plate_first_order",
            variable=variable,
            shear_treatment="exclude",
            **common,
        )
        problem.add_kernel(
            "plate_first_order",
            variable=variable,
            shear_treatment="only",
            quadrature="midpoint",
            reduced_integration=True,
            **common,
        )
    # Symmetry: u = phi_x = 0 on x = 0 and v = phi_y = 0 on y = 0
    problem.add_boundary_condition("Dirichlet", "left", variable="u", value=0.0)
    problem.add_boundary_condition("Dirichlet", "left", variable="rotation_x", value=0.0)
    problem.add_boundary_condition("Dirichlet", "bottom", variable="v", value=0.0)
    problem.add_boundary_condition(
        "Dirichlet",
        "bottom",
        variable="rotation_y",
        value=0.0,
    )
    if support == "clamped":
        for name in variables:
            problem.add_boundary_condition(
                "Dirichlet",
                ["right", "top"],
                variable=name,
                value=0.0,
            )
    else:  # Simply supported (SS-1)
        problem.add_boundary_condition("Dirichlet", "right", variable="v", value=0.0)
        problem.add_boundary_condition("Dirichlet", "right", variable="w", value=0.0)
        problem.add_boundary_condition(
            "Dirichlet",
            "right",
            variable="rotation_y",
            value=0.0,
        )
        problem.add_boundary_condition("Dirichlet", "top", variable="u", value=0.0)
        problem.add_boundary_condition("Dirichlet", "top", variable="w", value=0.0)
        problem.add_boundary_condition(
            "Dirichlet",
            "top",
            variable="rotation_x",
            value=0.0,
        )
    return problem


def normalized_centre_deflection(problem):
    """w(0, 0) E1 h^3 / (q a^4)."""
    centre = problem.values("w")[problem.node_at((0.0, 0.0))]
    return centre * 3.0e7 * plate_thickness**3 / (plate_load * plate_side**4)


table_10_5_1 = {
    # Power-law index: DMCDM values for 2x2, 4x4, 8x8, 16x16, 32x32 meshes
    0: [0.01281, 0.01359, 0.01377, 0.01383, 0.01384],
    1: [0.02997, 0.03174, 0.03222, 0.03234, 0.03237],
    2: [0.04395, 0.04659, 0.04728, 0.04746, 0.04749],
    3: [0.05187, 0.05499, 0.05580, 0.05601, 0.05607],
    4: [0.05601, 0.05937, 0.06024, 0.06048, 0.06054],
    5: [0.05838, 0.06186, 0.06279, 0.06300, 0.06306],
}


@pytest.mark.parametrize("power_law_index", sorted(table_10_5_1))
def test_table_10_5_1_clamped_functionally_graded_plate(power_law_index):
    expected = table_10_5_1[power_law_index]
    for num_elements, reference in zip((2, 4, 8, 16), expected):
        problem = rectangular_plate(num_elements, power_law_index=float(power_law_index))
        problem.solve()
        # The book prints five decimals; agreement to the last printed digit.
        assert normalized_centre_deflection(problem) == pytest.approx(reference, abs=2.1e-5)


# ---------------------------------------------------------------------------
# Chapter 8, mixed classical plate theory model DM-CP(M)
# ---------------------------------------------------------------------------
def classical_circular_plate(num_elements, power_law_index=0.0, support="hinged"):
    """Axisymmetric circular plate, mixed classical (Kirchhoff) theory."""
    s = circular_plate_stiffness(power_law_index)
    mesh = dualmesh.generate_line_mesh(start=0.0, end=circular_radius, num_elements=num_elements)
    problem = dualmesh.Problem(mesh, method="DMCDM", coordinates="axisymmetric")
    plate = problem.add_physics(
        "solid_mechanics",
        formulation="circular_plate",
        theory="classical",
        extensional_stiffness=s.extensional,
        coupling_stiffness=s.coupling,
        bending_stiffness=s.bending,
        poissons_ratio=Poisson_ratio,
        transverse_load=circular_load,
    )
    # Symmetry at r = 0: u(0) = 0.  The slope dw/dr is the natural boundary
    # quantity of the moment equation, so dw/dr(0) = 0 needs no condition.
    plate.add_boundary_condition("disp_r = 0", boundary="left")
    # A hinged edge holds the displacements, and the physics holds its moment at zero.
    if support == "hinged":
        plate.add_boundary_condition(["disp_r = 0", "disp_z = 0"], boundary="right")
    else:
        plate.add_boundary_condition("fixed", boundary="right")
    problem.solve()
    return problem


def test_table_8_6_1_mixed_classical_plate():
    """Table 8.6.1, DM-CP(M) column: w(0) for 2, 4, 8, 16, 32, and 64 elements."""
    expected = [0.1149, 0.1156, 0.1158, 0.1159, 0.1159, 0.1159]
    for num_elements, reference in zip((2, 4, 8, 16, 32, 64), expected):
        problem = classical_circular_plate(num_elements)
        centre = problem.values("disp_z")[problem.node_at((0.0,))]
        assert centre == pytest.approx(reference, abs=1.1e-4)


@pytest.mark.parametrize(
    "power_law_index, expected",
    [
        # Table 8.6.2, DM-CP(M) column (32 elements)
        (0.0, 0.0284),
        (1.0, 0.0665),
        (2.0, 0.0975),
        (3.0, 0.1151),
        (4.0, 0.1242),
        (5.0, 0.1294),
        (7.5, 0.1369),
        (10.0, 0.1424),
        (12.0, 0.1466),
        (15.0, 0.1527),
        (20.0, 0.1621),
    ],
)
def test_table_8_6_2_mixed_classical_plate(power_law_index, expected):
    problem = classical_circular_plate(32, power_law_index=power_law_index, support="clamped")
    centre = problem.values("disp_z")[problem.node_at((0.0,))]
    assert centre == pytest.approx(expected, abs=1.1e-4)


@pytest.mark.parametrize("support", ["hinged", "clamped"])
def test_mixed_classical_plate_bending_moment(support):
    """Reddy, Eqs. (8.6.4d) and (8.6.6d): the nodal moment matches the exact
    solution away from the origin.  At r = 0 the mixed models of both the FEM
    and the DMCDM are known to be inaccurate (Section 8.6.2 of the book)."""
    problem = classical_circular_plate(32, support=support)
    radius = problem.node_coordinates()[:, 0]
    moment = problem.values("bending_moment")
    xi = radius / circular_radius
    scale = circular_load * circular_radius**2 / 16.0
    if support == "hinged":
        exact = scale * (3 + Poisson_ratio) * (1 - xi**2)
    else:
        exact = scale * ((1 + Poisson_ratio) - (3 + Poisson_ratio) * xi**2)
    interior = radius > 1e-9
    assert abs(moment[interior] - exact[interior]).max() < 0.05 * scale


# ---------------------------------------------------------------------------
# The von Karman terms: cylindrical bending of a long strip with immovable edges
# ---------------------------------------------------------------------------
strip_span = 1.0
strip_thickness = 0.01
strip_modulus = 2.0e11
# The linear mid-span deflection under this load is about the thickness.
strip_load = 1.4e4
strip_extensional = strip_modulus * strip_thickness / (1 - Poisson_ratio**2)
strip_bending = strip_extensional * strip_thickness**2 / 12
strip_shear = 5 / 6 * strip_modulus * strip_thickness / (2 * (1 + Poisson_ratio))
strip_material = dict(
    youngs_modulus=strip_modulus,
    poissons_ratio=Poisson_ratio,
    thickness=strip_thickness,
    transverse_load=strip_load,
    von_karman=True,
)


def strip_mid_span_deflection(shear=None):
    r"""The mid-span deflection of a long strip of span $l$, with hinged edges that cannot move
    together, under the uniform load $q$, by the von Karman theory and by the linear theory.

    Timoshenko and Woinowsky-Krieger, Theory of Plates and Shells, 2nd ed. (1959), Sect. 2,
    give it for the Kirchhoff plate, and the derivation below adds the transverse shear
    stiffness $S$ (N/m) of the first-order theory.  The strip has no strain along its length,
    so its membrane force is $N = A \varepsilon_{xx}$, with $A = E h / (1 - \nu^2)$ the
    extensional stiffness (N/m), $E$ Young's modulus (Pa), $h$ the thickness (m) and $\nu$
    Poisson's ratio.  In-plane equilibrium makes $N$ uniform.  With
    $\varepsilon_{xx} = du/dx + (dw/dx)^2 / 2$, edges that do not move, $u(0) = u(l) = 0$, give
    the membrane equation

    $$ N = \frac{A}{2 l} \int_0^l \left( \frac{dw}{dx} \right)^2 dx . $$

    The bending moment is $M = D\, d\phi/dx$, with $D = E h^3 / (12 (1 - \nu^2))$ the bending
    stiffness (N m) and $\phi$ the rotation, and the shear force is $Q = S (\phi + dw/dx)$.
    Moment equilibrium $Q = dM/dx$ and transverse equilibrium $d(Q + N\, dw/dx)/dx = -q$ give
    for the curvature $\psi = d\phi/dx$ the equation $D (1 + N/S) \psi'' - N \psi = -q$, with
    $\psi = 0$ at the hinged edges.  With $\xi = x - l/2$, $k^2 = N / (D (1 + N/S))$, the
    parameter $u = k l / 2$ of the book and $\beta = 1 / (1 + N/S)$, its solution gives

    $$ \frac{dw}{dx} = \frac{q}{N} \left( -\xi + \beta \frac{\sinh k \xi}{k \cosh u} \right),
       \qquad
       w(l/2) = \frac{q}{N} \left( \frac{l^2}{8} - \beta \frac{1 - \mathrm{sech}\, u}{k^2}
       \right). $$

    For $S \to \infty$, $\beta = 1$, $u = (l/2) \sqrt{N/D}$, and $w(l/2) = 5 q l^4 f_0(u) /
    (384 D)$ with $f_0(u) = 24 (\mathrm{sech}\, u - 1 + u^2/2) / (5 u^4)$, the closed form of
    the book.  Bisection finds the root $N$ of the membrane equation, whose slope integral a
    64-point Gauss-Legendre rule evaluates to round-off.  The linear deflection is
    $5 q l^4 / (384 D) + q l^2 / (8 S)$."""
    q, span, extensional, bending = strip_load, strip_span, strip_extensional, strip_bending
    xi, weight = numpy.polynomial.legendre.leggauss(64)
    xi, weight = 0.5 * span * xi, 0.5 * span * weight

    def shape(force):
        ratio = 0.0 if shear is None else force / shear
        k = numpy.sqrt(force / (bending * (1 + ratio)))
        return k, 0.5 * k * span, 1 / (1 + ratio)

    def excess(force):
        # N minus the membrane force A/(2 l) * integral of (dw/dx)^2 of the deflection under N.
        k, u, beta = shape(force)
        slope = q / force * (-xi + beta * numpy.sinh(k * xi) / (k * numpy.cosh(u)))
        return force - extensional / (2 * span) * (weight * slope**2).sum()

    # The slope integral falls as N grows, so the membrane force at a small N bounds the root.
    low = 1e-6 * bending / span**2
    high = low - excess(low)
    for _ in range(100):
        middle = 0.5 * (low + high)
        low, high = (middle, high) if excess(middle) < 0 else (low, middle)
    force = 0.5 * (low + high)
    k, u, beta = shape(force)
    nonlinear = q / force * (span**2 / 8 - beta * (1 - 1 / numpy.cosh(u)) / k**2)
    linear = 5 * q * span**4 / (384 * bending)
    if shear is not None:
        linear += q * span**2 / (8 * shear)
    return nonlinear, linear


def test_von_Karman_strip_of_the_first_order_plate():
    r"""plate_first_order with von_karman: half of a strip of span $l$, hinged and held at
    $x = 0$, symmetric at $x = l/2$, and held in cylindrical bending ($v = \phi_y = 0$) on its
    long edges.  The membrane force cuts the linear deflection by about half, to the deflection
    of :func:`strip_mid_span_deflection` with the shear stiffness."""
    half = strip_span / 2
    mesh = dualmesh.generate_rectangle_mesh(
        x_min=0.0, x_max=half, y_min=0.0, y_max=half / 16, num_x_elements=32, num_y_elements=1
    )
    problem = dualmesh.Problem(mesh)
    plate = problem.add_physics("solid_mechanics", formulation="plate", **strip_material)
    plate.add_boundary_condition(["disp_x = 0", "disp_z = 0"], boundary="left")
    plate.add_boundary_condition(["disp_x = 0", "phi_x = 0"], boundary="right")
    plate.add_boundary_condition(["disp_y = 0", "phi_y = 0"], boundary=["bottom", "top"])
    problem.solve(report="none")
    expected, linear = strip_mid_span_deflection(shear=strip_shear)
    assert expected < 0.7 * linear
    assert problem.values("disp_z").max() == pytest.approx(expected, rel=2e-3)


@pytest.mark.parametrize("theory", ["first_order", "classical"])
def test_von_Karman_annulus_of_large_radius_bends_as_the_strip(theory):
    r"""circular_plate_first_order and circular_plate_classical_mixed with von_karman: an
    annulus of inner radius $R = 10^4 l$ and width $l$, hinged and held radially on both edges.
    Its axisymmetric equations are those of the strip of :func:`strip_mid_span_deflection`
    plus the terms with the factor $1/r$: the hoop force $N_{\theta\theta}/r$, the shear force
    $V_r/r$, the hoop moment $M_{\theta\theta}/r$ and the hoop strains $u/r$ and $\phi/r$.  Each
    is at most of the relative size $l/R = 10^{-4}$ against the term it accompanies, far below
    the tolerance.  The first-order theory takes the shear stiffness, and the classical theory
    is the Kirchhoff limit $S \to \infty$ of the book."""
    inner = 1.0e4 * strip_span
    mesh = dualmesh.generate_line_mesh(start=inner, end=inner + strip_span, num_elements=64)
    problem = dualmesh.Problem(mesh, coordinates="axisymmetric")
    plate = problem.add_physics(
        "solid_mechanics", formulation="circular_plate", theory=theory, **strip_material
    )
    plate.add_boundary_condition(["disp_r = 0", "disp_z = 0"], boundary=["left", "right"])
    problem.solve(report="none")
    expected, linear = strip_mid_span_deflection(
        shear=strip_shear if theory == "first_order" else None
    )
    assert expected < 0.7 * linear
    assert problem.values("disp_z").max() == pytest.approx(expected, rel=2e-3)
