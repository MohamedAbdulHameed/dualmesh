# SPDX-License-Identifier: LGPL-2.1-or-later
"""The solid at finite strain of the input, from its strain energy: a homogeneous stretch reaches
the stretch that its traction gives in closed form, and a small load gives linear elasticity with
the same Young modulus and Poisson ratio.

The same holds in axisymmetric coordinates, in plane stress, where the stretch of the thickness
solves P_33 = 0 at each point, with a pressure that follows the deformed boundary, with a thermal
stretch 1 + alpha (T - T_0) and with inertia. An energy given by its name and the same energy
given as a formula, which the engine differentiates, reach the same closed form. Each expected
value comes from a closed form or a root found by SciPy, never from the solid under test."""

import math

import dualmesh
import numpy
import pytest
from dualmesh.input import schema
from dualmesh.input.build import build_problem
from test_Jacobian import compare, finite_difference_Jacobian

Young_modulus, Poisson_ratio = 1.0e7, 0.3
shear_modulus = Young_modulus / (2 * (1 + Poisson_ratio))
first_Lame_parameter = (
    Young_modulus * Poisson_ratio / ((1 + Poisson_ratio) * (1 - 2 * Poisson_ratio))
)
# The thermal expansion of the thermal tests: alpha (T - T_0) = 0.1 at T = 400 K.
expansion_coefficient, stress_free_temperature = 1.0e-3, 300.0
# The two named energies written as formulas of the invariants.
neo_Hookean_formula = (
    "shear_modulus/2*(I_C - 3) - shear_modulus*log(J) + first_Lame_parameter/2*log(J)^2"
)
Saint_Venant_Kirchhoff_formula = (
    "first_Lame_parameter/8*(I_C - 3)^2 + shear_modulus/4*(trace_C_squared - 2*I_C + 3)"
)


def principal_stresses(energy, stretches, theta=1.0):
    """The principal first Piola-Kirchhoff stresses in Pa of the diagonal deformation gradient
    diag(stretches), from the closed form of each energy, with the thermal stretch theta:
    F_e = F / theta and P = theta^2 P_e(F_e)."""
    elastic = [stretch / theta for stretch in stretches]
    if energy in ("neo-Hookean", neo_Hookean_formula):
        volume_ratio = math.prod(elastic)
        stresses = [
            shear_modulus * (s - 1 / s) + first_Lame_parameter * math.log(volume_ratio) / s
            for s in elastic
        ]
    else:
        # P = F S, with S = lambda tr(E) I + 2 mu E and E = (F^T F - I)/2.
        strains = [(s * s - 1) / 2 for s in elastic]
        stresses = [
            s * (first_Lame_parameter * sum(strains) + 2 * shear_modulus * e)
            for s, e in zip(elastic, strains)
        ]
    return [theta**2 * stress for stress in stresses]


def block(kinematics="finite", **options):
    """A plane-strain block 2 m by 1 m, held at x = 0 and free to slide, with no lateral motion."""
    solid = dualmesh.problem(mesh={"rectangle": {"size": ["2 m", "1 m"], "cells": [4, 2]}})
    solid.physics("solid_mechanics", formulation="plane_strain", kinematics=kinematics, **options)
    solid.properties(Young_modulus="10 MPa", Poisson_ratio=Poisson_ratio)
    return solid


def cube(**options):
    """A cube 1 m wide at finite strain, held by its normal displacement on the three faces
    through the origin, so that it deforms homogeneously."""
    solid = dualmesh.problem(mesh={"box": {"size": ["1 m", "1 m", "1 m"], "cells": [2, 2, 2]}})
    solid.physics("solid_mechanics", kinematics="finite", **options)
    solid.properties(Young_modulus="10 MPa", Poisson_ratio=Poisson_ratio)
    solid.boundary_conditions("left", displacement_x="0 m")
    solid.boundary_conditions("bottom", displacement_y="0 m")
    solid.boundary_conditions("back", displacement_z="0 m")
    return solid


def cylinder(kinematics="finite", **options):
    """A solid cylinder of radius 1 m and height 1 m in axisymmetric coordinates, held on the axis
    (u_r = 0) and on its base (u_z = 0)."""
    solid = dualmesh.problem(
        mesh={
            "rectangle": {"size": ["1 m", "1 m"], "cells": [2, 2]},
            "coordinate_system": "axisymmetric",
        }
    )
    solid.physics("solid_mechanics", kinematics=kinematics, **options)
    solid.properties(Young_modulus="10 MPa", Poisson_ratio=Poisson_ratio)
    solid.boundary_conditions("left", displacement_r="0 m")
    solid.boundary_conditions("bottom", displacement_z="0 m")
    return solid


def tube_mesh(folder, cells):
    """The section 1 m <= R <= 2 m, 0 <= z <= 0.25 m of a thick tube, as a mesh file in
    axisymmetric coordinates whose boundaries are left (the inner surface), right, bottom and
    top."""
    pytest.importorskip("meshio")
    path = folder / "tube.vtu"
    dualmesh.write_mesh(dualmesh.generate_rectangle_mesh(1.0, 2.0, 0.0, 0.25, *cells), str(path))
    return {"file": str(path), "length_unit": "m", "coordinate_system": "axisymmetric"}


def engine_of(solid):
    """The engine's problem that the input builds, for a look at the solve and the Jacobian."""
    engine, _, _ = build_problem(schema.validate(solid.to_tree()))
    return engine


def neo_Hookean_uniaxial(s):
    """P_11 of the neo-Hookean energy at F = diag(s, 1, 1)."""
    return shear_modulus * (s - 1 / s) + first_Lame_parameter * math.log(s) / s


def Saint_Venant_Kirchhoff_uniaxial(s):
    """P_11 of the Saint Venant-Kirchhoff energy at F = diag(s, 1, 1)."""
    return s * (first_Lame_parameter + 2 * shear_modulus) * (s**2 - 1) / 2


@pytest.mark.parametrize(
    "energy, stress",
    [
        ("neo-Hookean", neo_Hookean_uniaxial),
        ("Saint_Venant_Kirchhoff", Saint_Venant_Kirchhoff_uniaxial),
        (Saint_Venant_Kirchhoff_formula, Saint_Venant_Kirchhoff_uniaxial),
    ],
)
def test_a_traction_stretches_the_block_by_its_closed_form_stretch(energy, stress):
    """With u_y = 0 everywhere, F = diag(s, 1, 1), and the traction P_11(s) on x = 2 m gives
    u_x = (s - 1) x at every node, which linear elements hold exactly.  The Cauchy stress
    sigma_xx = P_11 s / J at a point is P_11, since J = s."""
    stretch = 1.4
    solid = block(strain_energy=energy)
    solid.boundary_conditions("left", displacement_x="0 m")
    for side in ("left", "right", "bottom", "top"):
        solid.boundary_conditions(side, displacement_y="0 m")
    solid.boundary_conditions("right", traction=[f"{stress(stretch)!r} Pa", "0 Pa"])
    solid.postprocessor("axial", point_value="stress_xx", point=["1.3 m", "0.4 m"])
    solution = solid.solve(report="none")
    x = solution.coordinates("displacement")[:, 0]
    assert solution.field("displacement")[:, 0] == pytest.approx(
        (stretch - 1) * x, rel=1e-8, abs=1e-12
    )
    assert solution.postprocessor("axial")[-1] == pytest.approx(stress(stretch), rel=1e-8)
    assert solution.unit("axial") == "Pa"


def test_a_small_load_gives_linear_elasticity():
    """A traction of 1 Pa on a modulus of 10 MPa strains the block by about 1e-7, so finite and
    small strain differ by about that fraction of the displacement.  The stress of the finite
    strain is then 1e-7 of the terms of the size of the modulus that make it up, and Newton's test
    reaches it at the round-off of those terms."""
    solutions = {}
    for kinematics in ("finite", "small"):
        solid = block(kinematics)
        solid.boundary_conditions("left", displacement_x="0 m")
        solid.boundary_conditions("bottom", displacement_y="0 m")
        solid.boundary_conditions("right", traction=["1 Pa", "0.5 Pa"])
        solutions[kinematics] = solid.solve(report="none").field("displacement")
    scale = abs(solutions["small"]).max()
    assert abs(solutions["finite"] - solutions["small"]).max() < 1e-5 * scale


def test_an_energy_of_the_wrong_unit_and_an_energy_at_small_strain_are_refused():
    solid = block(strain_energy="shear_modulus*(I_C - 3)*J/[1 m]")
    solid.boundary_conditions("left", displacement_x="0 m")
    with pytest.raises(
        dualmesh.InputError, match=r"physics\.solid_mechanics\.strain_energy: .*J/m\^3"
    ):
        solid.solve(report="none")
    # The physics section refuses the energy of a small-strain solid where it is given.
    with pytest.raises(
        dualmesh.InputError,
        match=r"physics\.solid_mechanics\.strain_energy: .*takes no strain_energy",
    ):
        block("small", strain_energy="neo-Hookean")


# The follower pressure


def compressed_cube_stretches(pressure, follower=True):
    """The axial and the lateral stretch of a neo-Hookean cube under a pressure on one face, with
    free lateral faces: F = diag(a, s, s), P_22 = 0, and P_11 = -p s^2 for a pressure on the
    deformed face (the cofactor of F along the normal, per undeformed area), or -p for a dead
    load."""
    from scipy.optimize import brentq

    def lateral(axial):
        return brentq(
            lambda s: principal_stresses("neo-Hookean", [axial, s, s])[1] / Young_modulus,
            0.5,
            2.0,
            xtol=1e-15,
        )

    def axial_balance(axial):
        s = lateral(axial)
        load = pressure * s * s if follower else pressure
        return (principal_stresses("neo-Hookean", [axial, s, s])[0] + load) / Young_modulus

    axial = brentq(axial_balance, 0.5, 1.0, xtol=1e-15)
    return axial, lateral(axial)


def test_a_pressure_follows_the_deformed_face_of_a_compressed_cube():
    """A pressure of 3 MPa on the face x = 1 m compresses the cube by about 28 per cent.  On the
    deformed face the force per undeformed area is p s^2, with s the lateral stretch, which
    differs from the dead load p by about 20 per cent, so the nodal displacement tells the two
    apart.  Linear elements hold the homogeneous deformation exactly."""
    pytest.importorskip("scipy")
    pressure = 3.0e6
    axial, lateral = compressed_cube_stretches(pressure)
    dead_axial, _ = compressed_cube_stretches(pressure, follower=False)
    assert axial < 0.8
    assert abs(axial - dead_axial) > 1e-2
    solid = cube()
    solid.boundary_conditions("right", pressure=f"{pressure!r} Pa")
    solution = solid.solve(report="none")
    x, y, z = solution.coordinates("displacement").T
    displacement = solution.field("displacement")
    assert displacement[:, 0] == pytest.approx((axial - 1) * x, rel=1e-8, abs=1e-12)
    assert displacement[:, 1] == pytest.approx((lateral - 1) * y, rel=1e-8, abs=1e-12)
    assert displacement[:, 2] == pytest.approx((lateral - 1) * z, rel=1e-8, abs=1e-12)


def test_Newton_converges_in_a_few_iterations_under_a_follower_pressure():
    """At about 25 per cent strain under a follower pressure, Newton's method with the exact
    Jacobian, the load stiffness of the pressure included, converges from the undeformed state
    in at most 6 iterations, without continuation."""
    pytest.importorskip("scipy")
    pressure = 2.5e6
    axial, _ = compressed_cube_stretches(pressure)
    assert axial < 0.8
    solid = cube()
    solid.boundary_conditions("right", pressure=f"{pressure!r} Pa")
    result = engine_of(solid).solve(report="none")
    assert result.converged
    assert all(record.data_fraction == 1.0 for record in result.history)
    assert result.total_iterations <= 6


@pytest.mark.parametrize("energy", ["neo-Hookean", neo_Hookean_formula])
@pytest.mark.parametrize("coordinate_system", ["cartesian", "axisymmetric"])
def test_the_Jacobian_of_a_heated_solid_under_a_follower_pressure_is_exact(
    coordinate_system, energy, tmp_path
):
    """At a distorted state, with a non-uniform temperature: the Jacobian of the automatic
    differentiation agrees with central differences of the residual, so it holds the load
    stiffness of the pressure on the curved deformed boundary, the derivative of the stress by the
    temperature through the thermal stretch, and in axisymmetric coordinates the derivative of the
    hoop source."""
    if coordinate_system == "axisymmetric":
        solid = dualmesh.problem(mesh=tube_mesh(tmp_path, (3, 2)))
        solid.physics("solid_mechanics", kinematics="finite", strain_energy=energy)
        solid.boundary_conditions("bottom", displacement_z="0 m")
        loaded = ["left"]
        names = ("disp_r", "disp_z")
    else:
        solid = dualmesh.problem(mesh={"rectangle": {"size": ["2 m", "1 m"], "cells": [3, 2]}})
        solid.physics(
            "solid_mechanics", formulation="plane_strain", kinematics="finite", strain_energy=energy
        )
        solid.boundary_conditions("left", displacement_x="0 m")
        solid.boundary_conditions("bottom", displacement_y="0 m")
        loaded = ["right", "top"]
        names = ("disp_x", "disp_y")
    solid.physics("heat_conduction")
    solid.properties(
        thermal_conductivity="45 W/(m*K)",
        Young_modulus="10 MPa",
        Poisson_ratio=Poisson_ratio,
        linear_expansion_coefficient=f"{expansion_coefficient!r} 1/K",
        stress_free_temperature=f"{stress_free_temperature!r} K",
    )
    solid.boundary_conditions("left", temperature="300 K")
    solid.boundary_conditions("right", temperature="400 K")
    for side in loaded:
        solid.boundary_conditions(side, pressure="1 MPa")
    engine = engine_of(solid)
    engine.initialize()
    x, y = engine.node_coordinates()[:, 0], engine.node_coordinates()[:, 1]
    # The Jacobian is taken with the prescribed values in place, so the state vanishes on the
    # held bottom: there a displacement of the size of the cell height of the tube would turn
    # its first row of elements inside out.
    engine.set_values(names[0], 0.08 * numpy.sin(x + 2 * y) + 0.05 * x * y)
    engine.set_values(names[1], y * (0.06 * numpy.cos(2 * x - y) - 0.04 * x))
    engine.set_values("temperature", 300.0 + 40.0 * x * y)
    _, Jacobian = engine.linear_system()
    exact = Jacobian.toarray()
    approximate = finite_difference_Jacobian(engine)
    assert compare(exact, approximate) < 1e-6


# Axisymmetric coordinates


@pytest.mark.parametrize("energy", ["neo-Hookean", "Saint_Venant_Kirchhoff"])
def test_a_traction_stretches_a_solid_cylinder_homogeneously(energy):
    """F = diag(a, b, a) in the order r, z, theta, with the hoop stretch 1 + u_r/R = a.  The axial
    displacement (b - 1) z on the top and the traction P_rR(a, b) of the closed form on the outer
    radius give u_r = (a - 1) R and u_z = (b - 1) z at every node.  The radial balance holds only
    with the hoop source -P_33/R, which cancels the term P_rR/R of the divergence in r-z.  The
    Cauchy stress sigma = P F^T / J, with J = a^2 b, has sigma_rr = P_rR / (a b) and
    sigma_zz = P_zZ / a^2 at every point."""
    radial, axial = 1.2, 0.9
    stresses = principal_stresses(energy, [radial, axial, radial])
    traction = stresses[0]
    solid = cylinder(strain_energy=energy)
    solid.boundary_conditions("top", displacement_z=f"{axial - 1!r} m")
    solid.boundary_conditions("right", traction=[f"{traction!r} Pa", "0 Pa"])
    solid.postprocessor("radial_stress", point_value="stress_rr", point=["0.6 m", "0.3 m"])
    solid.postprocessor("axial_stress", point_value="stress_zz", point=["0.6 m", "0.3 m"])
    solution = solid.solve(report="none")
    r, z = solution.coordinates("displacement").T
    displacement = solution.field("displacement")
    assert displacement[:, 0] == pytest.approx((radial - 1) * r, rel=1e-8, abs=1e-12)
    assert displacement[:, 1] == pytest.approx((axial - 1) * z, rel=1e-8, abs=1e-12)
    assert solution.postprocessor("radial_stress")[-1] == pytest.approx(
        stresses[0] / (radial * axial), rel=1e-8
    )
    assert solution.postprocessor("axial_stress")[-1] == pytest.approx(
        stresses[1] / radial**2, rel=1e-8
    )


def test_a_small_load_on_a_cylinder_gives_axisymmetric_linear_elasticity():
    """As in plane strain, a traction of 1 Pa strains the cylinder by about 1e-7, so finite and
    small strain differ by about that fraction of the displacement."""
    solutions = {}
    for kinematics in ("finite", "small"):
        solid = cylinder(kinematics)
        solid.boundary_conditions("right", traction=["1 Pa", "0.5 Pa"])
        solutions[kinematics] = solid.solve(report="none").field("displacement")
    scale = abs(solutions["small"]).max()
    assert abs(solutions["finite"] - solutions["small"]).max() < 1e-5 * scale


# The thermal stretch


def heated(solid, temperature, boundaries):
    """Adds heat conduction with the thermal expansion of the module to a solid, at a uniform
    temperature held on the given boundaries."""
    solid.physics("heat_conduction")
    solid.properties(
        thermal_conductivity="45 W/(m*K)",
        linear_expansion_coefficient=f"{expansion_coefficient!r} 1/K",
        stress_free_temperature=f"{stress_free_temperature!r} K",
    )
    for side in boundaries:
        solid.boundary_conditions(side, temperature=f"{temperature!r} K")
    return solid


@pytest.mark.parametrize("shape", ["cube", "cylinder"])
def test_a_free_body_expands_by_its_thermal_stretch_without_stress(shape):
    """At a uniform 400 K, theta = 1 + alpha (T - T_0) = 1.1.  A body held only against rigid
    motion takes F = theta I, free of stress, so u = (theta - 1) X at every node, in three
    dimensions and in axisymmetric coordinates, where the hoop stretch is theta too."""
    theta = 1.0 + expansion_coefficient * (400.0 - stress_free_temperature)
    if shape == "cube":
        solid = heated(cube(), 400.0, ["left", "right"])
    else:
        solid = heated(cylinder(), 400.0, ["bottom", "top"])
    solution = solid.solve(report="none")
    points = solution.coordinates("displacement")
    displacement = solution.field("displacement")
    for component in range(points.shape[1]):
        assert displacement[:, component] == pytest.approx(
            (theta - 1) * points[:, component], rel=1e-8, abs=1e-12
        )


@pytest.mark.parametrize(
    "energy", ["neo-Hookean", "Saint_Venant_Kirchhoff", Saint_Venant_Kirchhoff_formula]
)
def test_a_heated_plane_strain_block_expands_in_its_plane_by_the_closed_form_stretch(energy):
    """In plane strain F = diag(a, a, 1): the in-plane stretch a solves P_11 = 0 with
    P = theta^2 P_e(F / theta), found here by SciPy from the closed form of the energy, and
    u = (a - 1) X at every node."""
    from scipy.optimize import brentq

    theta = 1.0 + expansion_coefficient * (400.0 - stress_free_temperature)
    stretch = brentq(
        lambda a: principal_stresses(energy, [a, a, 1.0], theta)[0] / Young_modulus,
        1.0,
        1.5,
        xtol=1e-15,
    )
    solid = heated(block(strain_energy=energy), 400.0, ["left", "right"])
    solid.boundary_conditions("left", displacement_x="0 m")
    solid.boundary_conditions("bottom", displacement_y="0 m")
    solution = solid.solve(report="none")
    points = solution.coordinates("displacement")
    displacement = solution.field("displacement")
    for component in range(2):
        assert displacement[:, component] == pytest.approx(
            (stretch - 1) * points[:, component], rel=1e-8, abs=1e-12
        )


def test_a_small_rise_of_temperature_gives_the_plane_strain_expansion():
    """At alpha (T - T_0) = 1e-4, the in-plane strain of a free plane-strain block is
    (1 + nu) alpha (T - T_0) of linear thermoelasticity, to a relative error of the order of the
    strain."""
    rise = 0.1
    solid = heated(block(), stress_free_temperature + rise, ["left", "right"])
    solid.boundary_conditions("left", displacement_x="0 m")
    solid.boundary_conditions("bottom", displacement_y="0 m")
    solution = solid.solve(report="none")
    x = solution.coordinates("displacement")[:, 0]
    strain = solution.field("displacement")[x > 0, 0] / x[x > 0]
    assert strain == pytest.approx((1 + Poisson_ratio) * expansion_coefficient * rise, rel=1e-3)


@pytest.mark.parametrize("energy", ["neo-Hookean", "Saint_Venant_Kirchhoff"])
def test_the_hydrostatic_stress_of_a_heated_cube_holds_it_at_its_size(energy):
    """A cube held at F = I at 400 K carries P = theta^2 P_e(I / theta), a hydrostatic stress of
    the closed form.  That stress, applied as a traction on the three free faces, keeps the cube
    at its undeformed size, as clamping would: u = 0 at every node, where a free cube expands by
    0.1 m."""
    theta = 1.0 + expansion_coefficient * (400.0 - stress_free_temperature)
    hydrostatic = principal_stresses(energy, [1.0, 1.0, 1.0], theta)[0]
    solid = heated(cube(strain_energy=energy), 400.0, ["left", "right"])
    for side, direction in (("right", 0), ("top", 1), ("front", 2)):
        traction = ["0 Pa"] * 3
        traction[direction] = f"{hydrostatic!r} Pa"
        solid.boundary_conditions(side, traction=traction)
    solution = solid.solve(report="none")
    assert abs(solution.field("displacement")).max() < 1e-9


# Inertia


def test_a_body_falls_freely_from_its_initial_displacement_without_stress():
    """A block without supports starts at the initial displacement (0.01 m, -0.02 m), at rest,
    under gravity.  It falls as a rigid body, u = u_0 + g t^2 / 2, free of stress, which the
    Newmark relations of the generalized-alpha method integrate exactly for a constant
    acceleration."""
    gravity = 9.81
    end_time = 0.05
    solid = dualmesh.problem(mesh={"rectangle": {"size": ["1 m", "1 m"], "cells": [2, 2]}})
    solid.physics("solid_mechanics", formulation="plane_strain", kinematics="finite")
    solid.properties(Young_modulus="10 MPa", Poisson_ratio=Poisson_ratio, density="1000 kg/m^3")
    solid.gravity(["0 m/s^2", f"-{gravity!r} m/s^2"])
    solid.initial_conditions(displacement=["0.01 m", "-0.02 m"])
    solid.time(end_time=f"{end_time!r} s", initial_time_step="0.01 s", step_control={"by": "fixed"})
    solution = solid.solve(report="none")
    displacement = solution.field("displacement")
    assert displacement[:, 0] == pytest.approx(0.01, rel=1e-8)
    assert displacement[:, 1] == pytest.approx(-0.02 - gravity * end_time**2 / 2, rel=1e-8)


def test_a_small_vibration_gives_the_small_strain_dynamics():
    """A bar starts at the uniform initial displacement 1 mm along its axis, held there at its
    left end, and a traction of 1 Pa on its right end sets it vibrating.  The strain is about
    1e-7, so finite and small strain differ by about that fraction of the motion."""
    solutions = {}
    for kinematics in ("finite", "small"):
        bar = dualmesh.problem(mesh={"rectangle": {"size": ["2 m", "0.5 m"], "cells": [8, 1]}})
        bar.physics("solid_mechanics", formulation="plane_strain", kinematics=kinematics)
        bar.properties(Young_modulus="10 MPa", Poisson_ratio=Poisson_ratio, density="1000 kg/m^3")
        bar.initial_conditions(displacement=["1 mm", "0 m"])
        bar.boundary_conditions("left", displacement=["1 mm", "0 m"])
        bar.boundary_conditions("right", traction=["1 Pa", "0 Pa"])
        bar.time(end_time="0.02 s", initial_time_step="0.002 s", step_control={"by": "fixed"})
        solutions[kinematics] = bar.solve(report="none").field("displacement")
    motion = solutions["small"] - numpy.array([1.0e-3, 0.0])
    scale = abs(motion).max()
    assert scale > 0.0
    assert abs(solutions["finite"] - solutions["small"]).max() < 1e-5 * scale


# Plane stress


def plate(energy="neo-Hookean", size=("1 m", "1 m"), cells=(2, 2)):
    """A plane-stress plate at finite strain, held by its normal displacement on x = 0 and y = 0,
    so that it deforms homogeneously."""
    solid = dualmesh.problem(mesh={"rectangle": {"size": list(size), "cells": list(cells)}})
    solid.physics(
        "solid_mechanics", formulation="plane_stress", kinematics="finite", strain_energy=energy
    )
    solid.properties(Young_modulus="10 MPa", Poisson_ratio=Poisson_ratio)
    solid.boundary_conditions("left", displacement_x="0 m")
    solid.boundary_conditions("bottom", displacement_y="0 m")
    return solid


def solve_plate(solid):
    """The node positions and the displacement of the solved plate, and the stretch of the thickness
    at the centre of each element."""
    engine = engine_of(solid)
    assert engine.solve(report="none").converged
    points = numpy.asarray(engine.node_coordinates())[:, :2]
    displacement = numpy.column_stack([engine.values(name) for name in ("disp_x", "disp_y")])
    thickness = numpy.asarray(engine.property_at_centroids("deformation_gradient"))[:, 8]
    return points, displacement, thickness


def assert_homogeneous(points, displacement, stretches):
    """u = (stretch - 1) X along each axis at every node, which linear elements hold exactly."""
    for component, stretch in enumerate(stretches):
        assert displacement[:, component] == pytest.approx(
            (stretch - 1) * points[:, component], rel=1e-8, abs=1e-12
        )


@pytest.mark.parametrize("stretch", [1.3, 0.8])
@pytest.mark.parametrize(
    "energy", ["neo-Hookean", "Saint_Venant_Kirchhoff", Saint_Venant_Kirchhoff_formula]
)
def test_a_plane_stress_strip_pulled_or_pushed_thins_as_a_free_bar(energy, stretch):
    """A strip with free lateral faces under the traction P_11 on x = 2 m.  By isotropy its
    lateral and out-of-plane stretches are equal, F = diag(a, s, s), the same as those of a bar in
    three dimensions, whose s solves P_22(a, s, s) = 0, found here by SciPy.  For Saint
    Venant-Kirchhoff the lateral Green strain also has the closed form
    E_22 = -lambda E_11 / (2 (lambda + mu)).  The stretch 0.8 is a compression, where the stretch
    of the thickness exceeds 1."""
    from scipy.optimize import brentq

    lateral = brentq(
        lambda s: principal_stresses(energy, [stretch, s, s])[1] / Young_modulus,
        0.5,
        1.5,
        xtol=1e-15,
    )
    if energy == "Saint_Venant_Kirchhoff":
        axial_strain = (stretch**2 - 1) / 2
        lateral_strain = (
            -first_Lame_parameter * axial_strain / (2 * (first_Lame_parameter + shear_modulus))
        )
        assert lateral == pytest.approx(math.sqrt(1 + 2 * lateral_strain), rel=1e-12)
    traction = principal_stresses(energy, [stretch, lateral, lateral])[0]
    solid = plate(energy, ("2 m", "1 m"), (4, 2))
    solid.boundary_conditions("right", traction=[f"{traction!r} Pa", "0 Pa"])
    points, displacement, thickness = solve_plate(solid)
    assert_homogeneous(points, displacement, [stretch, lateral])
    assert thickness == pytest.approx(lateral, rel=1e-8)


@pytest.mark.parametrize("stretch", [1.2, 0.85])
@pytest.mark.parametrize("energy", ["neo-Hookean", "Saint_Venant_Kirchhoff"])
def test_an_equibiaxial_traction_stretches_a_plate_by_its_closed_form_stretch(energy, stretch):
    """F = diag(a, a, t): the stretch of the thickness t solves P_33(a, a, t) = 0, found here by
    SciPy from the closed form of the energy, and the traction P_11(a, a, t) on x = 1 m and on
    y = 1 m then gives u = (a - 1) X at every node.  The stretch 0.85 is a compression, where
    Newton's method for t starts below the root."""
    from scipy.optimize import brentq

    thickness = brentq(
        lambda t: principal_stresses(energy, [stretch, stretch, t])[2] / Young_modulus,
        0.3,
        2.0,
        xtol=1e-15,
    )
    traction = principal_stresses(energy, [stretch, stretch, thickness])[0]
    solid = plate(energy)
    solid.boundary_conditions("right", traction=[f"{traction!r} Pa", "0 Pa"])
    solid.boundary_conditions("top", traction=["0 Pa", f"{traction!r} Pa"])
    points, displacement, computed = solve_plate(solid)
    assert_homogeneous(points, displacement, [stretch, stretch])
    assert computed == pytest.approx(thickness, rel=1e-8)


def test_a_plate_stretched_past_the_range_of_Saint_Venant_Kirchhoff_is_refused():
    """For Saint Venant-Kirchhoff, S_33 = 0 gives
    t^2 = (2 mu - lambda (C_11 + C_22 - 3)) / (lambda + 2 mu), so no stretch of the thickness
    exists once C_11 + C_22 >= 3 + 2 mu / lambda, about 4.33 at nu = 0.3.  An equibiaxial stretch
    of 1.6 gives 5.12.  The residual at that state is refused, and a solve that prescribes it
    reports that it did not converge, in place of an answer."""
    stretch = 1.6
    assert 2 * stretch**2 > 3 + 2 * shear_modulus / first_Lame_parameter
    solid = plate("Saint_Venant_Kirchhoff")
    engine = engine_of(solid)
    engine.initialize()
    points = numpy.asarray(engine.node_coordinates())
    engine.set_values("disp_x", (stretch - 1) * points[:, 0])
    engine.set_values("disp_y", (stretch - 1) * points[:, 1])
    with pytest.raises(RuntimeError, match="no stretch of the thickness"):
        engine.linear_system()
    solid.boundary_conditions("right", displacement_x=f"{stretch - 1!r} m")
    solid.boundary_conditions("top", displacement_y=f"{stretch - 1!r} m")
    result = engine_of(solid).solve(report="none", error_on_divergence=False)
    assert not result.converged


def test_a_small_load_on_a_plane_stress_strip_gives_the_plane_stress_modulus():
    """A strip held at u_y = 0 and pulled by 1 Pa: linear plane stress gives the axial strain
    sigma (1 - nu^2) / E, with sigma the traction.  The strain is about 1e-7, so finite and small
    strain agree with it to about that fraction."""
    traction = 1.0
    strain = traction * (1 - Poisson_ratio**2) / Young_modulus
    for kinematics in ("finite", "small"):
        solid = dualmesh.problem(mesh={"rectangle": {"size": ["2 m", "1 m"], "cells": [4, 2]}})
        solid.physics("solid_mechanics", formulation="plane_stress", kinematics=kinematics)
        solid.properties(Young_modulus="10 MPa", Poisson_ratio=Poisson_ratio)
        solid.boundary_conditions("left", displacement_x="0 m")
        for side in ("left", "right", "bottom", "top"):
            solid.boundary_conditions(side, displacement_y="0 m")
        solid.boundary_conditions("right", traction=[f"{traction!r} Pa", "0 Pa"])
        solution = solid.solve(report="none")
        x = solution.coordinates("displacement")[:, 0]
        assert solution.field("displacement")[:, 0] == pytest.approx(
            strain * x, rel=1e-5, abs=1e-5 * strain
        )


@pytest.mark.parametrize("energy", ["neo-Hookean", "Saint_Venant_Kirchhoff"])
def test_a_free_plate_expands_by_its_thermal_stretch_in_its_plane_and_its_thickness(energy):
    """At a uniform 400 K, theta = 1.1.  A plate held only against rigid motion takes
    F = theta I, free of stress, so u = (theta - 1) X at every node, and the stretch of the
    thickness that solves P_33 = 0 is theta."""
    theta = 1.0 + expansion_coefficient * (400.0 - stress_free_temperature)
    solid = heated(plate(energy), 400.0, ["left", "right"])
    points, displacement, thickness = solve_plate(solid)
    assert_homogeneous(points, displacement, [theta, theta])
    assert thickness == pytest.approx(theta, rel=1e-8)


def test_a_pressure_on_a_plane_stress_plate_follows_its_deformed_thickness():
    """A pressure of 3 MPa on x = 1 m with free lateral faces: F = diag(a, s, t), and P_22 = 0
    with P_33 = 0 give t = s by isotropy.  The deformed face carries the force p s t = p s^2 per
    undeformed area, as on the cube of three dimensions, so the stretches are those of
    compressed_cube_stretches.  A pressure on a face of fixed thickness would give p s."""
    pytest.importorskip("scipy")
    pressure = 3.0e6
    axial, lateral = compressed_cube_stretches(pressure)
    solid = plate()
    solid.boundary_conditions("right", pressure=f"{pressure!r} Pa")
    points, displacement, thickness = solve_plate(solid)
    assert_homogeneous(points, displacement, [axial, lateral])
    assert thickness == pytest.approx(lateral, rel=1e-8)


@pytest.mark.parametrize(
    "energy", ["neo-Hookean", "Saint_Venant_Kirchhoff", Saint_Venant_Kirchhoff_formula]
)
def test_the_Jacobian_of_a_heated_plane_stress_plate_under_a_follower_pressure_is_exact(energy):
    """At a distorted state, with a non-uniform temperature: the Jacobian of the automatic
    differentiation agrees with central differences of the residual.  The stress and the pressure
    read the stretch of the thickness, so the Jacobian holds its derivative by the in-plane
    gradient and the temperature, -(dP_33/dg) / (dP_33/dF_33) by the implicit function theorem."""
    solid = dualmesh.problem(mesh={"rectangle": {"size": ["2 m", "1 m"], "cells": [3, 2]}})
    solid.physics(
        "solid_mechanics", formulation="plane_stress", kinematics="finite", strain_energy=energy
    )
    solid.physics("heat_conduction")
    solid.properties(
        thermal_conductivity="45 W/(m*K)",
        Young_modulus="10 MPa",
        Poisson_ratio=Poisson_ratio,
        linear_expansion_coefficient=f"{expansion_coefficient!r} 1/K",
        stress_free_temperature=f"{stress_free_temperature!r} K",
    )
    solid.boundary_conditions("left", displacement_x="0 m", temperature="300 K")
    solid.boundary_conditions("bottom", displacement_y="0 m")
    solid.boundary_conditions("right", temperature="400 K")
    for side in ("right", "top"):
        solid.boundary_conditions(side, pressure="1 MPa")
    engine = engine_of(solid)
    engine.initialize()
    x, y = engine.node_coordinates()[:, 0], engine.node_coordinates()[:, 1]
    engine.set_values("disp_x", 0.08 * numpy.sin(x + 2 * y) + 0.05 * x * y)
    engine.set_values("disp_y", 0.06 * numpy.cos(2 * x - y) - 0.04 * x)
    engine.set_values("temperature", 300.0 + 40.0 * x * y)
    _, Jacobian = engine.linear_system()
    exact = Jacobian.toarray()
    approximate = finite_difference_Jacobian(engine)
    assert compare(exact, approximate) < 1e-6
