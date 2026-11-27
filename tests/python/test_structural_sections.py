# SPDX-License-Identifier: LGPL-2.1-or-later
"""The structural formulations of solid_mechanics compute their section from
the material: the plate, beam and circular-plate kernels integrate the
material law through the thickness at every integration point.  The checks:

* a homogeneous section given by its material equals the same section given
  by its stiffnesses;
* a power-law functionally graded section, E(z) with the index 2, has the
  closed-form stiffnesses A, B and D, and with a graded expansion coefficient
  the closed-form thermal force and moment;
* a modulus that depends on the temperature, E(T), gives the exact Jacobian;
* an expansion coefficient of the temperature, alpha(T), gives the secant
  thermal strain alpha(T) (T - T_0) of the continuum.
"""

import numpy
import pytest

dualmesh = pytest.importorskip("dualmesh")

E, nu, H = 2.0e11, 0.3, 0.01


def test_a_modulus_that_depends_on_the_temperature_has_an_exact_Jacobian():
    from test_Jacobian import compare, finite_difference_Jacobian

    mesh = dualmesh.generate_line_mesh(start=0.0, end=1.0, num_elements=6)
    problem = dualmesh.Problem(mesh, method="fem")
    heat = problem.add_physics("heat_conduction", thermal_conductivity=50.0, heat_source=1.0e4)
    heat.add_boundary_condition("temperature = 293", boundary=["left", "right"])
    beam = problem.add_physics(
        "solid_mechanics",
        formulation="beam",
        theory="Timoshenko",
        youngs_modulus="2.0e11*(1 - 4.0e-4*(temperature - 293))",
        poissons_ratio=nu,
        cross_section={"width": 0.1, "height": 0.2},
        transverse_load=-1.0e5,
    )
    beam.add_boundary_condition("fixed", boundary="left")
    problem.solve(report="none")
    _, Jacobian = problem.linear_system()
    assert compare(Jacobian.toarray(), finite_difference_Jacobian(problem)) < 1e-6


# ---------------------------------------------------------------------------
# Thermal expansion: the thermal force and moment of the section
# ---------------------------------------------------------------------------
alpha, t0, t1 = 1.2e-5, 293.0, 393.0


def _heated(mesh, coordinates="cartesian", boundary=("left", "right")):
    problem = dualmesh.Problem(mesh, method="fem", coordinates=coordinates)
    heat = problem.add_physics("heat_conduction", thermal_conductivity=50.0)
    heat.add_boundary_condition("temperature = T1", boundary=list(boundary), T1=t1)
    return problem


def _expand(problem, solid):
    problem.add_coupling(
        "thermal_expansion",
        heat_conduction=None,
        solid_mechanics=solid,
        linear_expansion_coefficient=alpha,
        stress_free_temperature=t0,
    )
    problem.solve(report="none")


@pytest.mark.parametrize(
    "theory, form",
    [("Euler-Bernoulli", None), ("Timoshenko", None), ("Timoshenko", "displacement")],
)
def test_a_heated_cantilever_expands_freely(theory, form):
    # A homogeneous cantilever heated uniformly by 100 K: u(L) = alpha dT L, and no bending.
    problem = _heated(dualmesh.generate_line_mesh(start=0.0, end=2.0, num_elements=8))
    beam = problem.add_physics(
        "solid_mechanics",
        formulation="beam",
        theory=theory,
        beam_formulation=form,
        youngs_modulus=E,
        poissons_ratio=nu,
        cross_section={"width": 0.1, "height": 0.2},
    )
    beam.add_boundary_condition("fixed", boundary="left")
    _expand(problem, beam)
    tip = problem.sample("disp_x", [[2.0, 0.0, 0.0]])[0]
    assert tip == pytest.approx(alpha * (t1 - t0) * 2.0, rel=1e-10)
    assert numpy.abs(problem.values("disp_z")).max() < 1e-12


@pytest.mark.parametrize(
    "theory, form",
    [("Euler-Bernoulli", None), ("Timoshenko", None), ("Timoshenko", "displacement")],
)
def test_a_heated_graded_cantilever_bends_as_its_section_requires(theory, form):
    r"""A cantilever of alumina on aluminium, free of load and heated uniformly by
    $\Delta T = 100$ K, with Young's modulus $E = E_b + \Delta E\, s$ and the expansion
    coefficient $\alpha = \alpha_b + \Delta\alpha\, s$ both linear in $s = 1/2 + z/h$, where
    $z$ is the height above the mid-surface, $h$ the height of the section, $b$ its width, and
    $\Delta E = E_t - E_b$, $\Delta\alpha = \alpha_t - \alpha_b$ the differences between top
    and bottom.  A graded $\alpha$ is what bends it: with a uniform $\alpha$ the thermal strain
    is uniform and the free beam only stretches, whatever the grade of $E$.

    The free beam carries no axial force and no moment, so its strain $e_0$ and its curvature
    $k = -w''$ are uniform and solve $[A, B; B, D] [e_0; k] = \Delta T [F; G]$.  With
    $z = h (s - 1/2)$ and $\int_0^1 s^m (s - 1/2) ds = m / (2 (m + 1) (m + 2))$, the section
    integrals are

    $$ A = \int E dA = b h \left( E_b + \frac{\Delta E}{2} \right), \qquad
       B = \int E z dA = \frac{b h^2 \Delta E}{12}, \qquad
       D = \int E z^2 dA = b h^3 \left( \frac{E_b}{12} + \frac{\Delta E}{24} \right), $$

    $$ F = \int E \alpha dA = b h \left( E_b \alpha_b + \frac{E_b \Delta\alpha + \Delta E
       \alpha_b}{2} + \frac{\Delta E \Delta\alpha}{3} \right), \qquad
       G = \int E \alpha z dA = \frac{b h^2}{12} (E_b \Delta\alpha + \Delta E \alpha_b +
       \Delta E \Delta\alpha), $$

    and $e_0 = \Delta T (D F - B G) / (A D - B^2)$, $k = \Delta T (A G - B F) / (A D - B^2)$.
    From the clamped end $u(L) = e_0 L$ and $w(L) = -k L^2 / 2$."""
    top, bottom = 3.8e11, 7.0e10
    top_alpha, bottom_alpha = 7.0e-6, 2.3e-5
    width, height, length = 0.1, 0.2, 2.0
    problem = _heated(dualmesh.generate_line_mesh(start=0.0, end=length, num_elements=32))
    beam = problem.add_physics(
        "solid_mechanics",
        formulation="beam",
        theory=theory,
        beam_formulation=form,
        youngs_modulus=f"{bottom!r} + {top - bottom!r}*(0.5 + z/{height!r})",
        poissons_ratio=nu,
        cross_section={"width": width, "height": height},
    )
    beam.add_boundary_condition("fixed", boundary="left")
    problem.add_coupling(
        "thermal_expansion",
        heat_conduction=None,
        solid_mechanics=beam,
        linear_expansion_coefficient=(
            f"{bottom_alpha!r} + {top_alpha - bottom_alpha!r}*(0.5 + z/{height!r})"
        ),
        stress_free_temperature=t0,
    )
    problem.solve(report="none")
    modulus, expansion = top - bottom, top_alpha - bottom_alpha
    a = width * height * (bottom + modulus / 2)
    b = width * height**2 * modulus / 12
    d = width * height**3 * (bottom / 12 + modulus / 24)
    force = (
        width
        * height
        * (
            bottom * bottom_alpha
            + (bottom * expansion + modulus * bottom_alpha) / 2
            + modulus * expansion / 3
        )
    )
    moment = (
        width * height**2 / 12 * (bottom * expansion + modulus * bottom_alpha + modulus * expansion)
    )
    heating = t1 - t0
    strain = heating * (d * force - b * moment) / (a * d - b**2)
    curvature = heating * (a * moment - b * force) / (a * d - b**2)
    # The bending is of the order of the stretching, so the second check is not one of zero.
    assert abs(curvature) * height > 0.1 * abs(strain)
    u_tip = problem.sample("disp_x", [[length, 0.0, 0.0]])[0]
    w_tip = problem.sample("disp_z", [[length, 0.0, 0.0]])[0]
    assert u_tip == pytest.approx(strain * length, rel=1e-6)
    assert w_tip == pytest.approx(-curvature * length**2 / 2, rel=1e-3)


def test_an_unevenly_heated_beam_expands_by_its_secant_strain():
    r"""A free cantilever of length $L$, its temperature linear from $T_l$ at the clamped end to
    $T_r$ at the free end, with the secant expansion coefficient $\alpha(T) = a + b T$, an
    expression of the temperature, as in the continuum.  The free beam carries no axial force,
    so $u' = \alpha(T) (T - T_0)$, and with $\mathrm{d}x = L\,\mathrm{d}T / (T_r - T_l)$

    $$ u(L) = \frac{L}{T_r - T_l} \int_{T_l}^{T_r} (a + b T) (T - T_0)\,\mathrm{d}T
            = \frac{L}{T_r - T_l} \left[ a \left( \frac{T^2}{2} - T_0 T \right)
              + b \left( \frac{T^3}{3} - \frac{T_0 T^2}{2} \right) \right]_{T_l}^{T_r} . $$

    The strain is quadratic in $x$, which the two-point Gauss rule integrates exactly, so the
    nodal displacement is exact."""
    a, b, t_left, t_right, length = 1.0e-5, 2.0e-8, 293.0, 793.0, 2.0
    mesh = dualmesh.generate_line_mesh(start=0.0, end=length, num_elements=8)
    problem = dualmesh.Problem(mesh, method="fem")
    heat = problem.add_physics("heat_conduction", thermal_conductivity=50.0)
    heat.add_boundary_condition("temperature = T", boundary="left", T=t_left)
    heat.add_boundary_condition("temperature = T", boundary="right", T=t_right)
    beam = problem.add_physics(
        "solid_mechanics",
        formulation="beam",
        youngs_modulus=E,
        cross_section={"width": 0.1, "height": 0.2},
    )
    beam.add_boundary_condition("fixed", boundary="left")
    problem.add_coupling(
        "thermal_expansion",
        linear_expansion_coefficient=f"{a!r} + {b!r}*temperature",
        stress_free_temperature=t0,
    )
    problem.solve(report="none")

    def primitive(T):
        return a * (T**2 / 2 - t0 * T) + b * (T**3 / 3 - t0 * T**2 / 2)

    expected = length / (t_right - t_left) * (primitive(t_right) - primitive(t_left))
    tip = problem.sample("disp_x", [[length, 0.0, 0.0]])[0]
    assert tip == pytest.approx(expected, rel=1e-10)
    assert numpy.abs(problem.values("disp_z")).max() < 1e-12


def test_a_heated_plate_expands_freely():
    mesh = dualmesh.generate_rectangle_mesh(
        x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0, num_x_elements=6, num_y_elements=6
    )
    walls = ("left", "right", "bottom", "top")
    problem = _heated(mesh, boundary=walls)
    plate = problem.add_physics(
        "solid_mechanics", formulation="plate", youngs_modulus=E, poissons_ratio=nu, thickness=H
    )
    plate.add_boundary_condition("disp_x = 0", boundary="left")
    plate.add_boundary_condition("disp_y = 0", boundary="bottom")
    plate.add_boundary_condition("disp_z = 0", boundary=list(walls))
    _expand(problem, plate)
    corner = problem.sample("disp_x", [[1.0, 1.0, 0.0]])[0]
    assert corner == pytest.approx(alpha * (t1 - t0), rel=1e-10)
    assert numpy.abs(problem.values("disp_z")).max() < 1e-12


def test_a_heated_circular_plate_expands_freely():
    mesh = dualmesh.generate_line_mesh(start=0.0, end=0.5, num_elements=8)
    problem = _heated(mesh, coordinates="axisymmetric")
    plate = problem.add_physics(
        "solid_mechanics",
        formulation="circular_plate",
        youngs_modulus=E,
        poissons_ratio=nu,
        thickness=H,
    )
    plate.add_boundary_condition("disp_r = 0", boundary="left")
    plate.add_boundary_condition("disp_z = 0", boundary="right")
    _expand(problem, plate)
    edge = problem.sample("disp_r", [[0.5, 0.0, 0.0]])[0]
    assert edge == pytest.approx(alpha * (t1 - t0) * 0.5, rel=1e-10)


# ---------------------------------------------------------------------------
# The stress through the thickness
# ---------------------------------------------------------------------------
def test_a_restrained_heated_bar_carries_the_thermal_stress():
    # Both ends held axially, heated by 100 K: sigma_xx = -E alpha dT at every height.
    problem = _heated(dualmesh.generate_line_mesh(start=0.0, end=1.0, num_elements=8))
    beam = problem.add_physics(
        "solid_mechanics",
        formulation="beam",
        theory="Timoshenko",
        beam_formulation="displacement",
        youngs_modulus=E,
        poissons_ratio=nu,
        cross_section={"width": 0.1, "height": 0.2},
    )
    beam.add_boundary_condition(["disp_x = 0", "disp_z = 0"], boundary=["left", "right"])
    _expand(problem, beam)
    for z in ("top", "mid", "bottom"):
        assert beam.stress(z=z)[:, 0] == pytest.approx(-E * alpha * (t1 - t0), rel=1e-9)


def test_a_restrained_heated_plate_carries_the_thermal_stress():
    # Held in its plane on every edge, heated by 100 K: sigma_xx = sigma_yy =
    # -E alpha dT / (1 - nu), and no shear.
    mesh = dualmesh.generate_rectangle_mesh(
        x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0, num_x_elements=4, num_y_elements=4
    )
    walls = ["left", "right", "bottom", "top"]
    problem = _heated(mesh, boundary=walls)
    plate = problem.add_physics(
        "solid_mechanics", formulation="plate", youngs_modulus=E, poissons_ratio=nu, thickness=H
    )
    plate.add_boundary_condition(["disp_x = 0", "disp_y = 0", "disp_z = 0"], boundary=walls)
    _expand(problem, plate)
    stress = plate.stress(z="top")
    expected = -E * alpha * (t1 - t0) / (1 - nu)
    assert stress[:, 0] == pytest.approx(expected, rel=1e-9)
    assert stress[:, 1] == pytest.approx(expected, rel=1e-9)
    assert numpy.abs(stress[:, 2]).max() < 1e-6 * abs(expected)


def test_the_bending_stress_of_a_simply_supported_beam():
    # Under a uniform load q, |M(x)| = q x (L - x) / 2, and sigma_xx = M z / I at the height z,
    # since the section's moment is M = integral of sigma z dA.
    length, q, width, height = 2.0, 1.0e4, 0.1, 0.2
    mesh = dualmesh.generate_line_mesh(start=0.0, end=length, num_elements=64)
    problem = dualmesh.Problem(mesh, method="fem")
    beam = problem.add_physics(
        "solid_mechanics",
        formulation="beam",
        theory="Euler-Bernoulli",
        youngs_modulus=E,
        poissons_ratio=nu,
        cross_section={"width": width, "height": height},
        transverse_load=q,
    )
    beam.add_boundary_condition(
        ["disp_x = 0", "disp_z = 0", "bending_moment = 0"], boundary=["left", "right"]
    )
    problem.solve(report="none")
    x = numpy.array([mesh.element_centroid(e)[0] for e in range(mesh.num_elements)])
    moment = numpy.interp(x, problem.node_coordinates()[:, 0], problem.values("bending_moment"))
    inertia = width * height**3 / 12
    top = beam.stress(z="top")[:, 0]
    assert top == pytest.approx(moment * height / 2 / inertia, rel=1e-10)
    assert numpy.abs(beam.stress(z="mid")[:, 0]).max() < 1e-6 * numpy.abs(top).max()
    # The moment of the mixed model is that of the statics, q x (L - x) / 2, up to its sign.
    assert numpy.abs(moment).max() == pytest.approx(q * length**2 / 8, rel=1e-3)


# ---------------------------------------------------------------------------
# Free and supported ends
# ---------------------------------------------------------------------------
beam_forms = [("Euler-Bernoulli", None), ("Timoshenko", None), ("Timoshenko", "displacement")]
beam_length, beam_load, beam_width, beam_height = 2.0, 1.0e4, 0.1, 0.2


def _loaded_beam(theory, form, conditions):
    mesh = dualmesh.generate_line_mesh(start=0.0, end=beam_length, num_elements=64)
    problem = dualmesh.Problem(mesh, method="fem")
    beam = problem.add_physics(
        "solid_mechanics",
        formulation="beam",
        theory=theory,
        beam_formulation=form,
        youngs_modulus=E,
        poissons_ratio=nu,
        cross_section={"width": beam_width, "height": beam_height},
        transverse_load=beam_load,
    )
    for condition, boundary in conditions:
        beam.add_boundary_condition(condition, boundary=boundary)
    problem.solve(report="none")
    return problem


def _flexibilities(theory):
    # The bending flexibility 1/EI and, for a Timoshenko beam, the shear flexibility 1/(k G A).
    bending = 1.0 / (E * beam_width * beam_height**3 / 12)
    shear = (
        0.0
        if theory == "Euler-Bernoulli"
        else 1.0 / (5 / 6 * E / (2 * (1 + nu)) * beam_width * beam_height)
    )
    return bending, shear


@pytest.mark.parametrize("theory, form", beam_forms)
def test_a_cantilever_with_only_its_clamped_end_given_has_a_free_tip(theory, form):
    # w(L) = q L^4 / (8 EI) + q L^2 / (2 k G A): the tip carries no moment and no shear.
    problem = _loaded_beam(theory, form, [("fixed", "left")])
    bending, shear = _flexibilities(theory)
    exact = beam_load * beam_length**4 / 8 * bending + beam_load * beam_length**2 / 2 * shear
    tip = problem.sample("disp_z", [[beam_length, 0.0, 0.0]])[0]
    assert tip == pytest.approx(exact, rel=1e-3)


@pytest.mark.parametrize("theory, form", beam_forms)
def test_a_simply_supported_beam_needs_only_its_supports(theory, form):
    # w(L/2) = 5 q L^4 / (384 EI) + q L^2 / (8 k G A), with no moment given at the supports.
    conditions = [("disp_z = 0", ["left", "right"]), ("disp_x = 0", "left")]
    problem = _loaded_beam(theory, form, conditions)
    bending, shear = _flexibilities(theory)
    exact = 5 * beam_load * beam_length**4 / 384 * bending + beam_load * beam_length**2 / 8 * shear
    middle = problem.sample("disp_z", [[beam_length / 2, 0.0, 0.0]])[0]
    assert middle == pytest.approx(exact, rel=1e-3)


@pytest.mark.parametrize(
    "support, factor", [("simply_supported", (5 + nu) / (1 + nu)), ("fixed", 1.0)]
)
def test_a_classical_circular_plate_needs_only_its_edge_support(support, factor):
    # The centre deflection under a uniform load q: w(0) = factor q a^4 / (64 D), with
    # factor (5 + nu)/(1 + nu) for a simply supported edge and 1 for a clamped one.
    radius, load = 0.5, 1.0e3
    mesh = dualmesh.generate_line_mesh(start=0.0, end=radius, num_elements=64)
    problem = dualmesh.Problem(mesh, method="fem", coordinates="axisymmetric")
    plate = problem.add_physics(
        "solid_mechanics",
        formulation="circular_plate",
        theory="classical",
        youngs_modulus=E,
        poissons_ratio=nu,
        thickness=H,
        transverse_load=load,
    )
    plate.add_boundary_condition("disp_r = 0", boundary="left")
    if support == "fixed":
        plate.add_boundary_condition("fixed", boundary="right")
    else:
        plate.add_boundary_condition("disp_z = 0", boundary="right")
    problem.solve(report="none")
    rigidity = E * H**3 / (12 * (1 - nu**2))
    centre = problem.sample("disp_z", [[0.0, 0.0, 0.0]])[0]
    assert abs(centre) == pytest.approx(factor * load * radius**4 / (64 * rigidity), rel=1e-3)


@pytest.mark.parametrize("theory, form", beam_forms)
def test_half_of_a_simply_supported_beam_with_a_plane_of_symmetry(theory, form):
    # The left half, pinned at x = 0 and symmetric at the centre: w(L/2) as for the whole beam.
    mesh = dualmesh.generate_line_mesh(start=0.0, end=beam_length / 2, num_elements=32)
    problem = dualmesh.Problem(mesh, method="fem")
    beam = problem.add_physics(
        "solid_mechanics",
        formulation="beam",
        theory=theory,
        beam_formulation=form,
        youngs_modulus=E,
        poissons_ratio=nu,
        cross_section={"width": beam_width, "height": beam_height},
        transverse_load=beam_load,
    )
    beam.add_boundary_condition(["disp_x = 0", "disp_z = 0"], boundary="left")
    beam.add_boundary_condition("symmetry", boundary="right")
    problem.solve(report="none")
    bending, shear = _flexibilities(theory)
    exact = 5 * beam_load * beam_length**4 / 384 * bending + beam_load * beam_length**2 / 8 * shear
    centre = problem.sample("disp_z", [[beam_length / 2, 0.0, 0.0]])[0]
    assert centre == pytest.approx(exact, rel=1e-3)


@pytest.mark.parametrize(
    "inputs, message",
    [
        (dict(formulation="plate", youngs_modulus=E, poissons_ratio=nu), "give the thickness"),
        (dict(formulation="plate", youngs_modulus=E, thickness=0.1), "give the poissons_ratio"),
        (
            dict(
                formulation="beam",
                theory="Timoshenko",
                youngs_modulus=E,
                cross_section={"width": 0.1, "height": 0.2},
            ),
            "give the poissons_ratio",
        ),
        (
            dict(formulation="beam", bending_stiffness=1.0e5),
            "needs extensional_stiffness as well",
        ),
        (
            dict(
                formulation="plate",
                extensional_stiffness=1.0e8,
                bending_stiffness=1.0e5,
                shear_stiffness=1.0e7,
            ),
            "give the poissons_ratio",
        ),
        (
            dict(
                formulation="beam",
                theory="Timoshenko",
                extensional_stiffness=1.0e8,
                bending_stiffness=1.0e5,
            ),
            "needs shear_stiffness as well",
        ),
        (
            dict(
                formulation="beam",
                extensional_stiffness=1.0e8,
                bending_stiffness=1.0e5,
                shear_stiffness=1.0e7,
            ),
            "has no shear_stiffness",
        ),
        (
            dict(
                formulation="beam",
                theory="Timoshenko",
                youngs_modulus=E,
                poissons_ratio=nu,
                cross_section={"area": 0.02, "second_moment": 6.0e-5},
            ),
            "needs its 'shear_correction'",
        ),
        (
            dict(
                formulation="beam",
                youngs_modulus=E,
                poissons_ratio=nu,
                thickness=0.1,
                cross_section={"width": 0.1, "height": 0.2},
            ),
            "takes its cross_section",
        ),
        (
            dict(
                formulation="plate",
                youngs_modulus=E,
                poissons_ratio=nu,
                thickness=0.1,
                axial_load=1.0e3,
            ),
            "as 'in_plane_load', not 'axial_load'",
        ),
    ],
)
def test_an_incomplete_or_redundant_section_is_refused_by_name(inputs, message):
    dimension = 2 if inputs["formulation"] == "plate" else 1
    mesh = (
        dualmesh.generate_rectangle_mesh(0.0, 1.0, 0.0, 1.0, num_elements=[2, 2])
        if dimension == 2
        else dualmesh.generate_line_mesh(start=0.0, end=1.0, num_elements=4)
    )
    problem = dualmesh.Problem(mesh, method="fem")
    # A section the kernels refuse is refused when the problem builds them.
    with pytest.raises(ValueError, match=message):
        problem.add_physics("solid_mechanics", **inputs)
        problem.initialize()


def test_a_beam_on_a_quadratic_mesh_is_refused():
    mesh = dualmesh.generate_line_mesh(start=0.0, end=1.0, num_elements=4).second_order()
    problem = dualmesh.Problem(mesh, method="fem")
    with pytest.raises(ValueError, match="takes Edge2 elements"):
        problem.add_physics(
            "solid_mechanics",
            formulation="beam",
            youngs_modulus=E,
            poissons_ratio=nu,
            cross_section={"width": 0.1, "height": 0.2},
        )


def _two_block_line(length, step, num_elements):
    # Elements left of x = step are block 1 ("thick"), the others block 2 ("thin").
    x = numpy.linspace(0.0, length, num_elements + 1)
    cells = [[i, i + 1] for i in range(num_elements)]
    blocks = [1 if 0.5 * (x[i] + x[i + 1]) < step else 2 for i in range(num_elements)]
    mesh = dualmesh.mesh_from_arrays(x[:, None], cells, element_type="Edge2", blocks=blocks)
    mesh.set_block_name(1, "thick")
    mesh.set_block_name(2, "thin")
    mesh.add_bounding_box_sidesets()
    return mesh


@pytest.mark.parametrize("theory, form", beam_forms)
def test_a_stepped_cantilever_of_two_physics_has_the_exact_tip_deflection(theory, form):
    # Unit-load method: w(L) = q/(8 EI1) (L^4 - (L - a)^4) + q/(8 EI2) (L - a)^4, plus for
    # Timoshenko the shear terms q/(2 kGA1) (L^2 - (L - a)^2) + q/(2 kGA2) (L - a)^2.
    length, step, heights = 2.0, 1.0, {"thick": 0.2, "thin": 0.1}
    problem = dualmesh.Problem(_two_block_line(length, step, 64), method="fem")
    physics = {}
    for block, height in heights.items():
        physics[block] = problem.add_physics(
            "solid_mechanics",
            formulation="beam",
            theory=theory,
            beam_formulation=form,
            youngs_modulus=E,
            poissons_ratio=nu,
            cross_section={"width": beam_width, "height": height},
            transverse_load=beam_load,
            block=[block],
        )
    physics["thick"].add_boundary_condition("fixed", boundary="left")
    problem.solve(report="none")
    inertia = {b: beam_width * h**3 / 12 for b, h in heights.items()}
    shear = {b: 5 / 6 * E / (2 * (1 + nu)) * beam_width * h for b, h in heights.items()}
    rest = length - step
    exact = beam_load / (8 * E * inertia["thick"]) * (length**4 - rest**4)
    exact += beam_load / (8 * E * inertia["thin"]) * rest**4
    if theory == "Timoshenko":
        exact += (
            beam_load / (2 * shear["thick"]) * (length**2 - rest**2)
            + beam_load / (2 * shear["thin"]) * rest**2
        )
    tip = problem.sample("disp_z", [[length, 0.0, 0.0]])[0]
    assert tip == pytest.approx(exact, rel=2e-3)
    # The stress of one physics has a row per element, NaN on the other block.
    stress = physics["thin"].stress(z="top")
    assert stress.shape[0] == 64
    assert numpy.isnan(stress[0, 0]) and numpy.isfinite(stress[-1, 0])


def test_an_end_condition_after_the_first_solve_is_refused():
    problem = _loaded_beam("Euler-Bernoulli", None, [("fixed", "left")])
    beam = next(iter(problem._physics.values()))
    with pytest.raises(ValueError, match="Give every condition before the first solve"):
        beam.add_boundary_condition("fixed", boundary="right")


@pytest.mark.parametrize(
    "inputs, message",
    [
        (dict(thickness=-0.1), "thickness must be positive"),
        (dict(cross_section={"width": 0.1, "height": -0.2}), r"cross_section\['height'\] must be"),
        (dict(foundation_modulus=-1.0e8), "foundation_modulus must not be negative"),
    ],
)
def test_a_negative_size_or_modulus_is_refused(inputs, message):
    with pytest.raises(ValueError, match=message):
        dualmesh.physics.create(
            "solid_mechanics", "solid", dict(formulation="beam", youngs_modulus=E, **inputs)
        )


@pytest.mark.parametrize("method", ["fem", "dmcdm"])
def test_a_very_thin_displacement_beam_converges_at_its_round_off_level(method):
    # L/h = 1e4: the residual stops at round-off of 1e-5 of its first value, which Newton's method
    # accepts once the steps are below 1e-8; the tip is q L^4 / (8 EI) + q L^2 / (2 k G A).
    length, width = 2.0, 0.1
    height = length / 1.0e4
    mesh = dualmesh.generate_line_mesh(start=0.0, end=length, num_elements=64)
    problem = dualmesh.Problem(mesh, method=method)
    beam = problem.add_physics(
        "solid_mechanics",
        formulation="beam",
        theory="Timoshenko",
        beam_formulation="displacement",
        youngs_modulus=E,
        poissons_ratio=nu,
        cross_section={"width": width, "height": height},
        transverse_load=1.0,
    )
    beam.add_boundary_condition("fixed", boundary="left")
    problem.solve(report="none")
    inertia, area = width * height**3 / 12, width * height
    exact = length**4 / (8 * E * inertia) + length**2 / (2 * 5 / 6 * E / (2 * (1 + nu)) * area)
    assert problem.sample("disp_z", [[length, 0.0, 0.0]])[0] == pytest.approx(exact, rel=1e-4)


# ---------------------------------------------------------------------------
# A power-law graded section
# ---------------------------------------------------------------------------
# Alumina on top of aluminium: Young's modulus (Pa) and linear expansion coefficient (1/K).
top_modulus, bottom_modulus = 3.8e11, 7.0e10
top_expansion, bottom_expansion = 7.0e-6, 2.3e-5


def _power_law_modulus(height):
    """E(z) = E_b + (E_t - E_b) ((z + h/2)/h)^2, an expression of the height z above the
    mid-surface."""
    return (
        f"{bottom_modulus!r} + {top_modulus - bottom_modulus!r}*((z + {height / 2!r})/{height!r})^2"
    )


def _power_law_section(height):
    r"""The section integrals per unit width of the grade $E(z) = E_b + (E_t - E_b) s^2$, with
    $s = (z + h/2)/h$ running from 0 at the bottom to 1 at the top, $h$ the height (m) and $E_b$
    and $E_t$ the moduli (Pa) of the bottom and the top.  With $z = h (s - 1/2)$ and
    $\int_0^1 s^m (s - 1/2) ds = m / (2 (m + 1) (m + 2))$:

    $$ \int E dz = h \left( E_b + \frac{E_t - E_b}{3} \right), \qquad
       \int E z dz = \frac{h^2 (E_t - E_b)}{12}, \qquad
       \int E z^2 dz = h^3 \left( \frac{E_b}{12} + \frac{E_t - E_b}{30} \right). $$

    They are the stiffnesses of :func:`dualmesh.fgm.beam_stiffness`, Reddy's power law, at the
    index $n = 2$."""
    contrast = top_modulus - bottom_modulus
    return (
        height * (bottom_modulus + contrast / 3),
        height**2 * contrast / 12,
        height**3 * (bottom_modulus / 12 + contrast / 30),
    )


@pytest.mark.parametrize("theory, form", beam_forms)
def test_a_power_law_graded_cantilever_stretches_and_bends_as_its_section_requires(theory, form):
    r"""A cantilever of the $n = 2$ grade under the uniform load $q$ and the tip tension $P$.
    Statics gives the axial force $N = P$ and the moment $M(x) = -q (L - x)^2 / 2$.  The section
    law $N = A e_0 + B k$, $M = B e_0 + D k$, with the curvature $k = -w''$ and $A$, $B$, $D$
    the integrals of :func:`_power_law_section` times the width $b$, inverts with
    $\Delta = A D - B^2$ to $e_0 = (D N - B M) / \Delta$ and $k = (A M - B N) / \Delta$.
    Integrating from the clamped end gives

    $$ u(L) = \frac{D P L + B q L^3 / 6}{\Delta}, \qquad
       w(L) = \frac{B P L^2 / 2 + A q L^4 / 8}{\Delta} + \frac{q L^2}{2 S}, $$

    the last term only for Timoshenko, with $S = (5/6) A / (2 (1 + \nu))$.  With $P = q L^2$
    the coupling terms are comparable to the others, so a wrong $A$, $B$ or $D$ fails both
    checks."""
    q, length = beam_load, beam_length
    tension = q * length**2
    mesh = dualmesh.generate_line_mesh(start=0.0, end=length, num_elements=64)
    problem = dualmesh.Problem(mesh, method="fem")
    beam = problem.add_physics(
        "solid_mechanics",
        formulation="beam",
        theory=theory,
        beam_formulation=form,
        youngs_modulus=_power_law_modulus(beam_height),
        poissons_ratio=nu,
        cross_section={"width": beam_width, "height": beam_height},
        transverse_load=q,
    )
    beam.add_boundary_condition("fixed", boundary="left")
    problem.add_point_source(variable="disp_x", value=tension, points=[length])
    problem.solve(report="none")
    a, b, d = (beam_width * value for value in _power_law_section(beam_height))
    determinant = a * d - b**2
    u_tip = (d * tension * length + b * q * length**3 / 6) / determinant
    w_tip = (b * tension * length**2 / 2 + a * q * length**4 / 8) / determinant
    if theory == "Timoshenko":
        w_tip += q * length**2 / (2 * 5 / 6 * a / (2 * (1 + nu)))
    assert problem.sample("disp_x", [[length, 0.0, 0.0]])[0] == pytest.approx(u_tip, rel=1e-3)
    assert problem.sample("disp_z", [[length, 0.0, 0.0]])[0] == pytest.approx(w_tip, rel=1e-3)


def test_a_heated_graded_classical_circular_plate_has_its_thermal_force_and_moment():
    r"""A free classical circular plate of radius $a$, simply supported at its edge, with the
    $n = 2$ grade of $E$ and the linear grade $\alpha(z) = \alpha_b + (\alpha_t - \alpha_b) s$
    of the expansion coefficient, heated uniformly by $\Delta T = 100$ K.  The plate is free of
    load, so $N_{rr} = N_{\theta\theta} = 0$ and $M_{rr} = M_{\theta\theta} = 0$, and its
    strain $e_0$ and curvature $k$ are uniform and equal in both directions.  With the plate
    stiffnesses $A_p = \int E dz / (1 - \nu^2)$ (and so for $B_p$ and $D_p$), the thermal force
    $N_T = \Delta T \int E \alpha dz / (1 - \nu)$ and the thermal moment
    $M_T = \Delta T \int E \alpha z dz / (1 - \nu)$, the resultants vanish when
    $(1 + \nu) [A_p, B_p; B_p, D_p] [e_0; k] = [N_T; M_T]$, that is

    $$ \begin{bmatrix} \int E dz & \int E z dz \\ \int E z dz & \int E z^2 dz \end{bmatrix}
       \begin{bmatrix} e_0 \\ k \end{bmatrix}
       = \Delta T \begin{bmatrix} \int E \alpha dz \\ \int E \alpha z dz \end{bmatrix}. $$

    With $\int_0^1 s^m (s - 1/2) ds = m / (2 (m + 1) (m + 2))$, the thermal integrals are
    $\int E \alpha dz = h (E_b \alpha_b + E_b \Delta\alpha / 2 + \Delta E \alpha_b / 3 + \Delta E
    \Delta\alpha / 4)$ and $\int E \alpha z dz = h^2 (E_b \Delta\alpha / 12 + \Delta E \alpha_b /
    12 + 3 \Delta E \Delta\alpha / 40)$, with $\Delta E = E_t - E_b$ and
    $\Delta\alpha = \alpha_t - \alpha_b$.  Then $u(a) = e_0 a$ and, with $k = -w''$ and
    $w(a) = 0$, $w(0) = k a^2 / 2$.  A thermal moment left out would leave $k$ near zero."""
    radius = 0.1
    problem = _heated(
        dualmesh.generate_line_mesh(start=0.0, end=radius, num_elements=64),
        coordinates="axisymmetric",
    )
    plate = problem.add_physics(
        "solid_mechanics",
        formulation="circular_plate",
        theory="classical",
        youngs_modulus=_power_law_modulus(H),
        poissons_ratio=nu,
        thickness=H,
    )
    plate.add_boundary_condition("disp_r = 0", boundary="left")
    plate.add_boundary_condition("disp_z = 0", boundary="right")
    expansion = f"{bottom_expansion!r} + {top_expansion - bottom_expansion!r}*(z/{H!r} + 0.5)"
    problem.add_coupling(
        "thermal_expansion",
        heat_conduction=None,
        solid_mechanics=plate,
        linear_expansion_coefficient=expansion,
        stress_free_temperature=t0,
    )
    problem.solve(report="none")
    a, b, d = _power_law_section(H)
    modulus, expansion_contrast = top_modulus - bottom_modulus, top_expansion - bottom_expansion
    force = H * (
        bottom_modulus * bottom_expansion
        + bottom_modulus * expansion_contrast / 2
        + modulus * bottom_expansion / 3
        + modulus * expansion_contrast / 4
    )
    moment = H**2 * (
        bottom_modulus * expansion_contrast / 12
        + modulus * bottom_expansion / 12
        + 3 * modulus * expansion_contrast / 40
    )
    strain, curvature = numpy.linalg.solve(
        [[a, b], [b, d]], [(t1 - t0) * force, (t1 - t0) * moment]
    )
    edge = problem.sample("disp_r", [[radius, 0.0, 0.0]])[0]
    centre = problem.sample("disp_z", [[0.0, 0.0, 0.0]])[0]
    assert edge == pytest.approx(strain * radius, rel=1e-3)
    assert centre == pytest.approx(curvature * radius**2 / 2, rel=1e-3)
