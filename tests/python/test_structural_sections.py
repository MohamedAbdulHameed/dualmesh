# SPDX-License-Identifier: LGPL-2.1-or-later
"""The structural formulations of solid_mechanics compute their section from
the material: the plate, beam and circular-plate kernels integrate the
material law through the thickness at every integration point.  The checks:

* a homogeneous section given by its material equals the same section given
  by its stiffnesses;
* a power-law functionally graded section, E(z), equals the closed-form
  stiffnesses of dualmesh.fgm, for integer power-law indices exactly;
* a modulus that depends on the temperature, E(T), gives the exact Jacobian.
"""

import numpy as np
import pytest

dm = pytest.importorskip("dualmesh")

E, NU, H = 2.0e11, 0.3, 0.01


def _plate(**section):
    mesh = dm.generate_rectangle_mesh(
        x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0, num_x_elements=8, num_y_elements=8
    )
    problem = dm.Problem(mesh, method="fem")
    plate = problem.add_physics(
        "solid_mechanics", formulation="plate", transverse_load=1.0e3, **section
    )
    plate.add_boundary_condition(
        ["disp_x = 0", "disp_y = 0", "disp_z = 0"], boundary=["left", "right", "bottom", "top"]
    )
    problem.solve(report="none")
    return np.array(problem.values("disp_z"))


def test_a_plate_given_by_its_material_equals_its_stiffnesses():
    a = E * H / (1 - NU**2)
    d = E * H**3 / (12 * (1 - NU**2))
    s = 5 / 6 * E / (2 * (1 + NU)) * H
    by_stiffness = _plate(
        extensional_stiffness=a, bending_stiffness=d, shear_stiffness=s, poissons_ratio=NU
    )
    by_material = _plate(youngs_modulus=E, poissons_ratio=NU, thickness=H)
    assert np.abs(by_material - by_stiffness).max() < 1e-12 * np.abs(by_stiffness).max()


def _beam(theory, form=None, **section):
    mesh = dm.generate_line_mesh(start=0.0, end=1.0, num_elements=16)
    problem = dm.Problem(mesh, method="fem")
    beam = problem.add_physics(
        "solid_mechanics",
        formulation="beam",
        theory=theory,
        form=form,
        transverse_load=-1.0e3,
        **section,
    )
    beam.add_boundary_condition("fixed", boundary="left")
    problem.solve(report="none")
    return np.array(problem.values("disp_z"))


@pytest.mark.parametrize(
    "theory, form",
    [("Euler_Bernoulli", None), ("Timoshenko", None), ("Timoshenko", "displacement")],
)
@pytest.mark.parametrize("power_law_index", [0.0, 1.0, 2.0])
def test_a_graded_beam_equals_the_closed_form_stiffnesses(theory, form, power_law_index):
    top, bottom, width, height = 30.0e6, 3.0e6, 0.1, 0.2
    closed = dm.fgm.beam_stiffness(
        modulus_top=top,
        modulus_bottom=bottom,
        power_law_index=power_law_index,
        height=height,
        width=width,
        poissons_ratio=NU,
    )
    by_stiffness = _beam(
        theory,
        form,
        extensional_stiffness=closed.extensional,
        coupling_stiffness=closed.coupling,
        bending_stiffness=closed.bending,
        # The Euler-Bernoulli beam has no shear deformation, so it takes no shear stiffness.
        **({"shear_stiffness": closed.shear} if theory == "Timoshenko" else {}),
    )
    graded = f"({top} - {bottom})*(0.5 + z/{height})^{power_law_index} + {bottom}"
    by_material = _beam(
        theory,
        form,
        youngs_modulus=graded,
        poissons_ratio=NU,
        cross_section={"width": width, "height": height},
    )
    assert np.abs(by_material - by_stiffness).max() < 1e-10 * np.abs(by_stiffness).max()


def test_a_homogeneous_beam_by_area_and_second_moment():
    width, height = 0.1, 0.2
    rectangle = _beam(
        "Timoshenko",
        youngs_modulus=E,
        poissons_ratio=NU,
        cross_section={"width": width, "height": height},
    )
    by_moments = _beam(
        "Timoshenko",
        youngs_modulus=E,
        poissons_ratio=NU,
        cross_section={
            "area": width * height,
            "second_moment": width * height**3 / 12,
            "shear_correction": 5 / 6,
        },
    )
    assert np.abs(rectangle - by_moments).max() < 1e-12 * np.abs(rectangle).max()


def test_a_circular_plate_given_by_its_material_equals_its_stiffnesses():
    def solve(**section):
        mesh = dm.generate_line_mesh(start=0.0, end=1.0, num_elements=16)
        problem = dm.Problem(mesh, method="fem", coordinates="axisymmetric")
        plate = problem.add_physics(
            "solid_mechanics", formulation="circular_plate", transverse_load=1.0e3, **section
        )
        plate.add_boundary_condition(["disp_r = 0", "disp_z = 0"], boundary="right")
        plate.add_boundary_condition("disp_r = 0", boundary="left")
        problem.solve(report="none")
        return np.array(problem.values("disp_z"))

    by_stiffness = solve(
        extensional_stiffness=E * H / (1 - NU**2),
        bending_stiffness=E * H**3 / (12 * (1 - NU**2)),
        shear_stiffness=5 / 6 * E / (2 * (1 + NU)) * H,
        poissons_ratio=NU,
    )
    by_material = solve(youngs_modulus=E, poissons_ratio=NU, thickness=H)
    # The section summed over the points through the thickness differs from the closed form in
    # the last digits.
    assert np.abs(by_material - by_stiffness).max() < 1e-10 * np.abs(by_stiffness).max()


def test_a_section_is_given_one_way():
    problem = dm.Problem(dm.generate_line_mesh(start=0.0, end=1.0, num_elements=4))
    with pytest.raises(Exception, match="not both"):
        problem.add_physics(
            "solid_mechanics",
            formulation="beam",
            youngs_modulus=E,
            poissons_ratio=NU,
            cross_section={"area": 1.0, "second_moment": 1.0},
            bending_stiffness=1.0,
        )


def test_a_modulus_that_depends_on_the_temperature_has_an_exact_jacobian():
    from test_jacobian import compare, finite_difference_jacobian

    mesh = dm.generate_line_mesh(start=0.0, end=1.0, num_elements=6)
    problem = dm.Problem(mesh, method="fem")
    heat = problem.add_physics("heat_transfer", thermal_conductivity=50.0, heat_source=1.0e4)
    heat.add_boundary_condition("temperature = 293", boundary=["left", "right"])
    beam = problem.add_physics(
        "solid_mechanics",
        formulation="beam",
        theory="Timoshenko",
        youngs_modulus="2.0e11*(1 - 4.0e-4*(temperature - 293))",
        poissons_ratio=NU,
        cross_section={"width": 0.1, "height": 0.2},
        transverse_load=-1.0e5,
    )
    beam.add_boundary_condition("fixed", boundary="left")
    problem.solve(report="none")
    _, jacobian = problem.linear_system()
    assert compare(jacobian.toarray(), finite_difference_jacobian(problem)) < 1e-6


# ---------------------------------------------------------------------------
# Thermal expansion: the thermal force and moment of the section
# ---------------------------------------------------------------------------
ALPHA, T0, T1 = 1.2e-5, 293.0, 393.0


def _heated(mesh, coordinates="cartesian", boundary=("left", "right")):
    problem = dm.Problem(mesh, method="fem", coordinates=coordinates)
    heat = problem.add_physics("heat_transfer", thermal_conductivity=50.0)
    heat.add_boundary_condition("temperature = T1", boundary=list(boundary), T1=T1)
    return problem


def _expand(problem, solid):
    problem.add_coupling(
        "thermal_expansion",
        heat_transfer=None,
        solid_mechanics=solid,
        linear_expansion_coeff=ALPHA,
        stress_free_temperature=T0,
    )
    problem.solve(report="none")


@pytest.mark.parametrize(
    "theory, form",
    [("Euler_Bernoulli", None), ("Timoshenko", None), ("Timoshenko", "displacement")],
)
def test_a_heated_cantilever_expands_freely(theory, form):
    # A homogeneous cantilever heated uniformly by 100 K: u(L) = alpha dT L, and no bending.
    problem = _heated(dm.generate_line_mesh(start=0.0, end=2.0, num_elements=8))
    beam = problem.add_physics(
        "solid_mechanics",
        formulation="beam",
        theory=theory,
        form=form,
        youngs_modulus=E,
        poissons_ratio=NU,
        cross_section={"width": 0.1, "height": 0.2},
    )
    beam.add_boundary_condition("fixed", boundary="left")
    _expand(problem, beam)
    tip = problem.sample("disp_x", [[2.0, 0.0, 0.0]])[0]
    assert tip == pytest.approx(ALPHA * (T1 - T0) * 2.0, rel=1e-10)
    assert np.abs(problem.values("disp_z")).max() < 1e-12


@pytest.mark.parametrize(
    "theory, form",
    [("Euler_Bernoulli", None), ("Timoshenko", None), ("Timoshenko", "displacement")],
)
def test_a_heated_graded_cantilever_bends_as_its_section_requires(theory, form):
    # A graded section expands more on its stiff side: with no load, [A B; B D] [e0; k] = [N_T;
    # M_T] gives a uniform strain and curvature, so u(L) = e0 L and w(L) = -k L^2 / 2.
    top, bottom, width, height, length = 30.0e6, 3.0e6, 0.1, 0.2, 2.0
    problem = _heated(dm.generate_line_mesh(start=0.0, end=length, num_elements=32))
    graded = f"({top} - {bottom})*(0.5 + z/{height}) + {bottom}"
    beam = problem.add_physics(
        "solid_mechanics",
        formulation="beam",
        theory=theory,
        form=form,
        youngs_modulus=graded,
        poissons_ratio=NU,
        cross_section={"width": width, "height": height},
    )
    beam.add_boundary_condition("fixed", boundary="left")
    _expand(problem, beam)
    z, w = np.polynomial.legendre.leggauss(12)
    z, w = 0.5 * height * z, 0.5 * height * w * width
    modulus = (top - bottom) * (0.5 + z / height) + bottom
    a, b, d = (w * modulus).sum(), (w * modulus * z).sum(), (w * modulus * z * z).sum()
    n_t = (w * modulus).sum() * ALPHA * (T1 - T0)
    m_t = (w * modulus * z).sum() * ALPHA * (T1 - T0)
    strain, curvature = np.linalg.solve([[a, b], [b, d]], [n_t, m_t])
    u_tip = problem.sample("disp_x", [[length, 0.0, 0.0]])[0]
    w_tip = problem.sample("disp_z", [[length, 0.0, 0.0]])[0]
    assert u_tip == pytest.approx(strain * length, rel=1e-8)
    assert w_tip == pytest.approx(-curvature * length**2 / 2, rel=1e-3)


def test_a_heated_plate_expands_freely():
    mesh = dm.generate_rectangle_mesh(
        x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0, num_x_elements=6, num_y_elements=6
    )
    walls = ("left", "right", "bottom", "top")
    problem = _heated(mesh, boundary=walls)
    plate = problem.add_physics(
        "solid_mechanics", formulation="plate", youngs_modulus=E, poissons_ratio=NU, thickness=H
    )
    plate.add_boundary_condition("disp_x = 0", boundary="left")
    plate.add_boundary_condition("disp_y = 0", boundary="bottom")
    plate.add_boundary_condition("disp_z = 0", boundary=list(walls))
    _expand(problem, plate)
    corner = problem.sample("disp_x", [[1.0, 1.0, 0.0]])[0]
    assert corner == pytest.approx(ALPHA * (T1 - T0), rel=1e-10)
    assert np.abs(problem.values("disp_z")).max() < 1e-12


def test_a_heated_circular_plate_expands_freely():
    mesh = dm.generate_line_mesh(start=0.0, end=0.5, num_elements=8)
    problem = _heated(mesh, coordinates="axisymmetric")
    plate = problem.add_physics(
        "solid_mechanics",
        formulation="circular_plate",
        youngs_modulus=E,
        poissons_ratio=NU,
        thickness=H,
    )
    plate.add_boundary_condition("disp_r = 0", boundary="left")
    plate.add_boundary_condition("disp_z = 0", boundary="right")
    _expand(problem, plate)
    edge = problem.sample("disp_r", [[0.5, 0.0, 0.0]])[0]
    assert edge == pytest.approx(ALPHA * (T1 - T0) * 0.5, rel=1e-10)


# ---------------------------------------------------------------------------
# The stress through the thickness
# ---------------------------------------------------------------------------
def test_a_restrained_heated_bar_carries_the_thermal_stress():
    # Both ends held axially, heated by 100 K: sigma_xx = -E alpha dT at every height.
    problem = _heated(dm.generate_line_mesh(start=0.0, end=1.0, num_elements=8))
    beam = problem.add_physics(
        "solid_mechanics",
        formulation="beam",
        theory="Timoshenko",
        form="displacement",
        youngs_modulus=E,
        poissons_ratio=NU,
        cross_section={"width": 0.1, "height": 0.2},
    )
    beam.add_boundary_condition(["disp_x = 0", "disp_z = 0"], boundary=["left", "right"])
    _expand(problem, beam)
    for z in ("top", "mid", "bottom"):
        assert beam.stress(z=z)[:, 0] == pytest.approx(-E * ALPHA * (T1 - T0), rel=1e-9)


def test_a_restrained_heated_plate_carries_the_thermal_stress():
    # Held in its plane on every edge, heated by 100 K: sigma_xx = sigma_yy =
    # -E alpha dT / (1 - nu), and no shear.
    mesh = dm.generate_rectangle_mesh(
        x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0, num_x_elements=4, num_y_elements=4
    )
    walls = ["left", "right", "bottom", "top"]
    problem = _heated(mesh, boundary=walls)
    plate = problem.add_physics(
        "solid_mechanics", formulation="plate", youngs_modulus=E, poissons_ratio=NU, thickness=H
    )
    plate.add_boundary_condition(["disp_x = 0", "disp_y = 0", "disp_z = 0"], boundary=walls)
    _expand(problem, plate)
    stress = plate.stress(z="top")
    expected = -E * ALPHA * (T1 - T0) / (1 - NU)
    assert stress[:, 0] == pytest.approx(expected, rel=1e-9)
    assert stress[:, 1] == pytest.approx(expected, rel=1e-9)
    assert np.abs(stress[:, 2]).max() < 1e-6 * abs(expected)


def test_the_bending_stress_of_a_simply_supported_beam():
    # Under a uniform load q, |M(x)| = q x (L - x) / 2, and sigma_xx = M z / I at the height z,
    # since the section's moment is M = integral of sigma z dA.
    length, q, width, height = 2.0, 1.0e4, 0.1, 0.2
    mesh = dm.generate_line_mesh(start=0.0, end=length, num_elements=64)
    problem = dm.Problem(mesh, method="fem")
    beam = problem.add_physics(
        "solid_mechanics",
        formulation="beam",
        theory="Euler_Bernoulli",
        youngs_modulus=E,
        poissons_ratio=NU,
        cross_section={"width": width, "height": height},
        transverse_load=q,
    )
    beam.add_boundary_condition(
        ["disp_x = 0", "disp_z = 0", "bending_moment = 0"], boundary=["left", "right"]
    )
    problem.solve(report="none")
    x = np.array([mesh.element_centroid(e)[0] for e in range(mesh.num_elements)])
    moment = np.interp(x, problem.entity_points()[:, 0], problem.values("bending_moment"))
    inertia = width * height**3 / 12
    top = beam.stress(z="top")[:, 0]
    assert top == pytest.approx(moment * height / 2 / inertia, rel=1e-10)
    assert np.abs(beam.stress(z="mid")[:, 0]).max() < 1e-6 * np.abs(top).max()
    # The moment of the mixed model is that of the statics, q x (L - x) / 2, up to its sign.
    assert np.abs(moment).max() == pytest.approx(q * length**2 / 8, rel=1e-3)


def test_a_continuum_returns_its_stress_property():
    mesh = dm.generate_rectangle_mesh(
        x_min=0.0, x_max=1.0, y_min=0.0, y_max=0.2, num_x_elements=8, num_y_elements=2
    )
    problem = dm.Problem(mesh, method="fem")
    solid = problem.add_physics(
        "solid_mechanics", formulation="plane_stress", youngs_modulus=E, poissons_ratio=NU
    )
    solid.add_boundary_condition("fixed", boundary="left")
    solid.add_boundary_condition("traction = [1e6, 0]", boundary="right")
    problem.solve(report="none")
    stress = solid.stress()
    assert stress[:, 0] == pytest.approx(1.0e6, rel=1e-6)
    with pytest.raises(Exception, match="no height z"):
        solid.stress(z=0.0)


# ---------------------------------------------------------------------------
# Free and supported ends
# ---------------------------------------------------------------------------
BEAM_FORMS = [("Euler_Bernoulli", None), ("Timoshenko", None), ("Timoshenko", "displacement")]
LENGTH, LOAD, WIDTH, HEIGHT = 2.0, 1.0e4, 0.1, 0.2


def _loaded_beam(theory, form, conditions):
    mesh = dm.generate_line_mesh(start=0.0, end=LENGTH, num_elements=64)
    problem = dm.Problem(mesh, method="fem")
    beam = problem.add_physics(
        "solid_mechanics",
        formulation="beam",
        theory=theory,
        form=form,
        youngs_modulus=E,
        poissons_ratio=NU,
        cross_section={"width": WIDTH, "height": HEIGHT},
        transverse_load=LOAD,
    )
    for condition, boundary in conditions:
        beam.add_boundary_condition(condition, boundary=boundary)
    problem.solve(report="none")
    return problem


def _flexibilities(theory):
    # The bending flexibility 1/EI and, for a Timoshenko beam, the shear flexibility 1/(k G A).
    bending = 1.0 / (E * WIDTH * HEIGHT**3 / 12)
    shear = (
        0.0 if theory == "Euler_Bernoulli" else 1.0 / (5 / 6 * E / (2 * (1 + NU)) * WIDTH * HEIGHT)
    )
    return bending, shear


@pytest.mark.parametrize("theory, form", BEAM_FORMS)
def test_a_cantilever_with_only_its_clamped_end_given_has_a_free_tip(theory, form):
    # w(L) = q L^4 / (8 EI) + q L^2 / (2 k G A): the tip carries no moment and no shear.
    problem = _loaded_beam(theory, form, [("fixed", "left")])
    bending, shear = _flexibilities(theory)
    exact = LOAD * LENGTH**4 / 8 * bending + LOAD * LENGTH**2 / 2 * shear
    tip = problem.sample("disp_z", [[LENGTH, 0.0, 0.0]])[0]
    assert tip == pytest.approx(exact, rel=1e-3)


@pytest.mark.parametrize("theory, form", BEAM_FORMS)
def test_a_simply_supported_beam_needs_only_its_supports(theory, form):
    # w(L/2) = 5 q L^4 / (384 EI) + q L^2 / (8 k G A), with no moment given at the supports.
    conditions = [("disp_z = 0", ["left", "right"]), ("disp_x = 0", "left")]
    problem = _loaded_beam(theory, form, conditions)
    bending, shear = _flexibilities(theory)
    exact = 5 * LOAD * LENGTH**4 / 384 * bending + LOAD * LENGTH**2 / 8 * shear
    middle = problem.sample("disp_z", [[LENGTH / 2, 0.0, 0.0]])[0]
    assert middle == pytest.approx(exact, rel=1e-3)


@pytest.mark.parametrize(
    "support, factor", [("simply_supported", (5 + NU) / (1 + NU)), ("fixed", 1.0)]
)
def test_a_classical_circular_plate_needs_only_its_edge_support(support, factor):
    # The centre deflection under a uniform load q: w(0) = factor q a^4 / (64 D), with
    # factor (5 + nu)/(1 + nu) for a simply supported edge and 1 for a clamped one.
    radius, load = 0.5, 1.0e3
    mesh = dm.generate_line_mesh(start=0.0, end=radius, num_elements=64)
    problem = dm.Problem(mesh, method="fem", coordinates="axisymmetric")
    plate = problem.add_physics(
        "solid_mechanics",
        formulation="circular_plate",
        theory="classical",
        youngs_modulus=E,
        poissons_ratio=NU,
        thickness=H,
        transverse_load=load,
    )
    plate.add_boundary_condition("disp_r = 0", boundary="left")
    if support == "fixed":
        plate.add_boundary_condition("fixed", boundary="right")
    else:
        plate.add_boundary_condition("disp_z = 0", boundary="right")
    problem.solve(report="none")
    rigidity = E * H**3 / (12 * (1 - NU**2))
    centre = problem.sample("disp_z", [[0.0, 0.0, 0.0]])[0]
    assert abs(centre) == pytest.approx(factor * load * radius**4 / (64 * rigidity), rel=1e-3)


@pytest.mark.parametrize("theory, form", BEAM_FORMS)
def test_half_of_a_simply_supported_beam_with_a_plane_of_symmetry(theory, form):
    # The left half, pinned at x = 0 and symmetric at the centre: w(L/2) as for the whole beam.
    mesh = dm.generate_line_mesh(start=0.0, end=LENGTH / 2, num_elements=32)
    problem = dm.Problem(mesh, method="fem")
    beam = problem.add_physics(
        "solid_mechanics",
        formulation="beam",
        theory=theory,
        form=form,
        youngs_modulus=E,
        poissons_ratio=NU,
        cross_section={"width": WIDTH, "height": HEIGHT},
        transverse_load=LOAD,
    )
    beam.add_boundary_condition(["disp_x = 0", "disp_z = 0"], boundary="left")
    beam.add_boundary_condition("symmetry", boundary="right")
    problem.solve(report="none")
    bending, shear = _flexibilities(theory)
    exact = 5 * LOAD * LENGTH**4 / 384 * bending + LOAD * LENGTH**2 / 8 * shear
    centre = problem.sample("disp_z", [[LENGTH / 2, 0.0, 0.0]])[0]
    assert centre == pytest.approx(exact, rel=1e-3)


def test_a_plate_writes_its_plane_of_symmetry_as_text():
    mesh = dm.generate_rectangle_mesh(0.0, 1.0, 0.0, 1.0, num_elements=[2, 2])
    problem = dm.Problem(mesh, method="fem")
    plate = problem.add_physics(
        "solid_mechanics", formulation="plate", youngs_modulus=E, poissons_ratio=NU, thickness=H
    )
    with pytest.raises(ValueError, match="written as text"):
        plate.add_boundary_condition("symmetry", boundary="left")


@pytest.mark.parametrize(
    "inputs, message",
    [
        (dict(formulation="plate", youngs_modulus=E, poissons_ratio=NU), "give the thickness"),
        (
            dict(formulation="beam", bending_stiffness=1.0e5),
            "needs extensional_stiffness as well",
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
                poissons_ratio=NU,
                cross_section={"area": 0.02, "second_moment": 6.0e-5},
            ),
            "needs its 'shear_correction'",
        ),
        (
            dict(
                formulation="beam",
                youngs_modulus=E,
                poissons_ratio=NU,
                thickness=0.1,
                cross_section={"width": 0.1, "height": 0.2},
            ),
            "takes its cross_section",
        ),
    ],
)
def test_an_incomplete_or_redundant_section_is_refused_by_name(inputs, message):
    dimension = 2 if inputs["formulation"] == "plate" else 1
    mesh = (
        dm.generate_rectangle_mesh(0.0, 1.0, 0.0, 1.0, num_elements=[2, 2])
        if dimension == 2
        else dm.generate_line_mesh(start=0.0, end=1.0, num_elements=4)
    )
    problem = dm.Problem(mesh, method="fem")
    with pytest.raises(ValueError, match=message):
        problem.add_physics("solid_mechanics", **inputs)


def test_a_beam_on_a_quadratic_mesh_is_refused():
    mesh = dm.generate_line_mesh(start=0.0, end=1.0, num_elements=4).second_order()
    problem = dm.Problem(mesh, method="fem")
    with pytest.raises(ValueError, match="takes Edge2 elements"):
        problem.add_physics(
            "solid_mechanics",
            formulation="beam",
            youngs_modulus=E,
            poissons_ratio=NU,
            cross_section={"width": 0.1, "height": 0.2},
        )


def test_a_height_outside_the_section_is_refused():
    problem = _loaded_beam("Euler_Bernoulli", None, [("fixed", "left")])
    beam = next(iter(problem._physics.values()))
    with pytest.raises(ValueError, match="outside the section"):
        beam.stress(z=0.11)
    assert beam.stress(z=0.1).shape[0] == 64


@pytest.mark.parametrize(
    "model, inputs, message",
    [
        (
            "beam_Euler_Bernoulli_mixed",
            dict(bending_stiffness=1.0e5),
            "give 'extensional_stiffness'",
        ),
        (
            "beam_Timoshenko_mixed",
            dict(extensional_stiffness=1.0e8, bending_stiffness=1.0e5),
            "give 'shear_stiffness'",
        ),
        (
            "beam_Euler_Bernoulli_mixed",
            dict(extensional_stiffness=1.0e8, bending_stiffness=1.0e5, shear_stiffness=1.0e7),
            "takes no 'shear_stiffness'",
        ),
        (
            "beam_Euler_Bernoulli_mixed",
            dict(
                youngs_modulus=E,
                cross_section={"width": 0.1, "height": 0.2},
                temperature="T",
                linear_expansion_coeff=1.2e-5,
            ),
            "give 'stress_free_temperature'",
        ),
    ],
)
def test_a_structural_kernel_needs_every_input_its_model_uses(model, inputs, message):
    mesh = dm.generate_line_mesh(start=0.0, end=1.0, num_elements=4)
    problem = dm.Problem(mesh, method="fem")
    for name in ("u", "w", "m", "T"):
        problem.add_variable(name)
    with pytest.raises(ValueError, match=message):
        problem.add_kernel(
            model,
            variable="w",
            axial_displacement="u",
            transverse_displacement="w",
            bending_moment="m",
            **inputs,
        )
        problem.initialize()


def test_a_plate_on_triangles_is_refused():
    mesh = dm.generate_rectangle_mesh(0.0, 1.0, 0.0, 1.0, num_elements=[2, 2], element_type="Tri3")
    problem = dm.Problem(mesh, method="fem")
    with pytest.raises(ValueError, match="takes Quad4 elements"):
        problem.add_physics(
            "solid_mechanics", formulation="plate", youngs_modulus=E, poissons_ratio=NU, thickness=H
        )


def _two_block_line(length, step, num_elements):
    # Elements left of x = step are block 1 ("thick"), the others block 2 ("thin").
    x = np.linspace(0.0, length, num_elements + 1)
    cells = [[i, i + 1] for i in range(num_elements)]
    blocks = [1 if 0.5 * (x[i] + x[i + 1]) < step else 2 for i in range(num_elements)]
    mesh = dm.mesh_from_arrays(x[:, None], cells, element_type="Edge2", blocks=blocks)
    mesh.set_block_name(1, "thick")
    mesh.set_block_name(2, "thin")
    mesh.add_bounding_box_sidesets()
    return mesh


@pytest.mark.parametrize("theory, form", BEAM_FORMS)
def test_a_stepped_cantilever_of_two_physics_has_the_exact_tip_deflection(theory, form):
    # Unit-load method: w(L) = q/(8 EI1) (L^4 - (L - a)^4) + q/(8 EI2) (L - a)^4, plus for
    # Timoshenko the shear terms q/(2 kGA1) (L^2 - (L - a)^2) + q/(2 kGA2) (L - a)^2.
    length, step, heights = 2.0, 1.0, {"thick": 0.2, "thin": 0.1}
    problem = dm.Problem(_two_block_line(length, step, 64), method="fem")
    physics = {}
    for block, height in heights.items():
        physics[block] = problem.add_physics(
            "solid_mechanics",
            formulation="beam",
            theory=theory,
            form=form,
            youngs_modulus=E,
            poissons_ratio=NU,
            cross_section={"width": WIDTH, "height": height},
            transverse_load=LOAD,
            block=[block],
        )
    physics["thick"].add_boundary_condition("fixed", boundary="left")
    problem.solve(report="none")
    inertia = {b: WIDTH * h**3 / 12 for b, h in heights.items()}
    shear = {b: 5 / 6 * E / (2 * (1 + NU)) * WIDTH * h for b, h in heights.items()}
    rest = length - step
    exact = LOAD / (8 * E * inertia["thick"]) * (length**4 - rest**4)
    exact += LOAD / (8 * E * inertia["thin"]) * rest**4
    if theory == "Timoshenko":
        exact += (
            LOAD / (2 * shear["thick"]) * (length**2 - rest**2)
            + LOAD / (2 * shear["thin"]) * rest**2
        )
    tip = problem.sample("disp_z", [[length, 0.0, 0.0]])[0]
    assert tip == pytest.approx(exact, rel=2e-3)
    # The stress of one physics has a row per element, NaN on the other block.
    stress = physics["thin"].stress(z="top")
    assert stress.shape[0] == 64
    assert np.isnan(stress[0, 0]) and np.isfinite(stress[-1, 0])


def test_an_end_condition_after_the_first_solve_is_refused():
    problem = _loaded_beam("Euler_Bernoulli", None, [("fixed", "left")])
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
        dm.physics.create(
            "solid_mechanics", "solid", dict(formulation="beam", youngs_modulus=E, **inputs)
        )


@pytest.mark.parametrize("method", ["fem", "dmcdm"])
def test_a_very_thin_displacement_beam_converges_at_its_round_off_level(method):
    # L/h = 1e4: the residual stops at round-off of 1e-5 of its first value, which Newton's method
    # accepts once the steps are below 1e-8; the tip is q L^4 / (8 EI) + q L^2 / (2 k G A).
    length, width = 2.0, 0.1
    height = length / 1.0e4
    mesh = dm.generate_line_mesh(start=0.0, end=length, num_elements=64)
    problem = dm.Problem(mesh, method=method)
    beam = problem.add_physics(
        "solid_mechanics",
        formulation="beam",
        theory="Timoshenko",
        form="displacement",
        youngs_modulus=E,
        poissons_ratio=NU,
        cross_section={"width": width, "height": height},
        transverse_load=1.0,
    )
    beam.add_boundary_condition("fixed", boundary="left")
    problem.solve(report="none")
    inertia, area = width * height**3 / 12, width * height
    exact = length**4 / (8 * E * inertia) + length**2 / (2 * 5 / 6 * E / (2 * (1 + NU)) * area)
    assert problem.sample("disp_z", [[length, 0.0, 0.0]])[0] == pytest.approx(exact, rel=1e-4)
