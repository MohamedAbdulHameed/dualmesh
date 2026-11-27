# SPDX-License-Identifier: LGPL-2.1-or-later
"""J2 plasticity: isotropic hardening, Armstrong-Frederick and Chaboche back stresses, and
plasticity with creep, against closed forms and the uniaxial law integrated by hand."""

from __future__ import annotations

import dualmesh
import numpy
import pytest
from dualmesh._core import InputError
from scipy.optimize import brentq

E, nu, size = 200e9, 0.3, 1e-3
area = size * size


def cube(stress_object="small_strain_stress", **material):
    """A unit cube in uniaxial stress along z: held on three faces by rollers."""
    mesh = dualmesh.generate_box_mesh(0, size, 0, size, 0, size, 1, 1, 1)
    problem = dualmesh.Problem(mesh)
    names = ["ux", "uy", "uz"]
    for v in names:
        problem.add_variable(v)
    if "temperature" in material:
        problem.add_variable("temperature")
        problem.add_kernel("diffusion", variable="temperature")
        problem.add_boundary_condition(
            "Dirichlet", list(mesh.sideset_names()), variable="temperature", value=700.0
        )
    problem.add_property(
        stress_object,
        displacements=names,
        formulation="three_dimensional",
        youngs_modulus=E,
        poissons_ratio=nu,
        **material,
    )
    stress = "first_piola_kirchhoff_stress" if stress_object == "finite_strain_stress" else "stress"
    for v in names:
        problem.add_kernel(
            "stress_divergence", variable=v, displacements=names, stress_property=stress
        )
    for v, side in (("ux", "left"), ("uy", "bottom"), ("uz", "back")):
        problem.add_boundary_condition("Dirichlet", side, variable=v, value=0.0)
    return problem


def axial_strain(problem):
    return problem.sample("uz", [[size / 2, size / 2, size * (1 - 1e-9)]])[0] / size


def test_a_bar_beyond_yield_has_the_strain_of_its_hardening():
    """sigma = 300 MPa on sigma_y = 250 MPa + 1 GPa p: p = 0.05 and the strain is sigma/E + p,
    in one steady solve."""
    problem = cube(yield_stress="250e6 + 1e9*equivalent_plastic_strain")
    problem.add_boundary_condition("traction", "front", variable="uz", traction=300e6)
    problem.solve(report="none")
    assert axial_strain(problem) == pytest.approx(300e6 / E + 0.05, rel=1e-8)


def test_finite_strain_plasticity_holds_the_yield_surface_and_the_force():
    """At 5 % plastic strain in finite strain: the von Mises (Cauchy) stress at the point equals
    the yield stress of its plastic strain, and the Cauchy stress on the deformed section
    carries the nominal force of the traction, both to round-off."""
    problem = cube("finite_strain_stress", yield_stress="250e6 + 1e9*equivalent_plastic_strain")
    problem.add_boundary_condition("traction", "front", variable="uz", traction=300e6)
    problem.solve(report="none", relative_tolerance=1e-14)
    stress = problem.property_at_centroids("stress")[0]
    plastic = problem.property_at_centroids("equivalent_plastic_strain")[0][0]
    assert plastic > 0.05
    von_Mises = stress[2] - stress[0]  # Uniaxial: the lateral stresses vanish
    assert abs(stress[0]) < 1e-6 * stress[2] and abs(stress[1]) < 1e-6 * stress[2]
    assert von_Mises == pytest.approx(250e6 + 1e9 * plastic, rel=1e-10)
    stretch_x = 1 + problem.sample("ux", [[size * (1 - 1e-9), size / 2, size / 2]])[0] / size
    stretch_y = 1 + problem.sample("uy", [[size / 2, size * (1 - 1e-9), size / 2]])[0] / size
    assert stress[2] * stretch_x * stretch_y == pytest.approx(300e6, rel=1e-10)


def test_below_yield_the_bar_is_elastic():
    problem = cube(yield_stress="250e6")
    problem.add_boundary_condition("traction", "front", variable="uz", traction=200e6)
    problem.solve(report="none")
    assert axial_strain(problem) == pytest.approx(200e6 / E, rel=1e-10)
    assert numpy.max(numpy.abs(problem.property_at_centroids("plastic_strain"))) == 0.0


def test_Prager_kinematic_hardening_is_linear_hardening_in_tension():
    """A back stress of modulus C hardens a monotonic uniaxial load as sigma_y + C p."""
    problem = cube(yield_stress="250e6", kinematic_hardening_moduli=[1e9])
    problem.add_boundary_condition("traction", "front", variable="uz", traction=300e6)
    problem.solve(report="none")
    assert axial_strain(problem) == pytest.approx(300e6 / E + 0.05, rel=1e-8)


def uniaxial_Armstrong_Frederick(strains, sigma0, moduli, recovery):
    """The uniaxial stress of J2 plasticity with Armstrong-Frederick back stresses under a
    prescribed axial strain, integrated by backward Euler as the return mapping does: with
    a_k = 1/(1 + gamma_k dp), xi = sigma_trial - sum a_k X_k and |xi| - (E + sum a_k C_k) dp =
    sigma0."""
    plastic, backs, stresses = 0.0, numpy.zeros(len(moduli)), []
    for strain in strains:
        trial = E * (strain - plastic)

        def f(dp, trial=trial, backs=backs):
            a = 1.0 / (1.0 + numpy.asarray(recovery) * dp)
            return abs(trial - a @ backs) - (E + a @ moduli) * dp - sigma0

        if f(0.0) <= 0.0:
            stresses.append(trial)
            continue
        dp = brentq(f, 0.0, 1.0, xtol=1e-16, rtol=1e-15)
        a = 1.0 / (1.0 + numpy.asarray(recovery) * dp)
        direction = numpy.sign(trial - a @ backs)
        plastic += direction * dp
        backs = a * (backs + numpy.asarray(moduli) * direction * dp)
        stresses.append(E * (strain - plastic))
    return numpy.array(stresses)


def cyclic_stresses(time_step, moduli, recovery):
    """The axial stress of the cube under the strain 0.01 (1 - |t - 1|), t from 0 to 3 (to 0.01,
    back to -0.01), from the force on the loaded face."""
    problem = cube(
        yield_stress="250e6",
        kinematic_hardening_moduli=moduli,
        dynamic_recovery=recovery,
    )
    problem.add_boundary_condition(
        "Dirichlet", "front", variable="uz", value=f"{size}*0.01*(1 - abs(t - 1))"
    )
    times, stresses = [], []

    def record(t, core):
        # The stress from the reaction: the force of the support on the loaded face.
        times.append(t)
        stresses.append(problem.boundary_flux("uz", "front") * -1.0 / area)

    problem.set_time_step_callback(record)
    problem.solve_transient(end_time=3.0, time_step=time_step, time_stepper="fixed", report="none")
    return numpy.array(times), numpy.array(stresses)


@pytest.mark.parametrize(
    "moduli, recovery", [([20e9], [100.0]), ([30e9, 3e9], [200.0, 10.0])], ids=["AF", "Chaboche"]
)
def test_a_strain_cycle_follows_the_uniaxial_law_of_the_back_stresses(moduli, recovery):
    """Tension to 1 %, compression to -1 %: the stress of the cube is that of the uniaxial law
    integrated by hand with the same steps, to the tolerance of Newton's method; the reversal
    shows the Bauschinger effect (yield in compression well before -sigma0 - X)."""
    times, stresses = cyclic_stresses(0.05, moduli, recovery)
    strains = 0.01 * (1 - numpy.abs(times - 1))
    expected = uniaxial_Armstrong_Frederick(strains, 250e6, numpy.array(moduli), recovery)
    assert stresses == pytest.approx(expected, rel=1e-7, abs=1e-3 * 250e6 * 1e-4)
    peak = stresses[numpy.argmax(strains)]
    first_compressive_yield = stresses[numpy.argmin(numpy.abs(strains - (0.01 - 2 * 250e6 / E)))]
    assert peak > 250e6 and first_compressive_yield < peak


def test_monotonic_Armstrong_Frederick_converges_to_its_closed_form():
    """sigma = sigma0 + (C/gamma)(1 - exp(-gamma eps_p)) under monotonic tension: the error of
    backward Euler falls at first order in the step."""
    C, gamma = 20e9, 100.0
    errors = []
    for dt in (0.05, 0.025):
        times, stresses = cyclic_stresses(dt, [C], [gamma])
        keep = times <= 1.0 + 1e-12
        strain = 0.01 * times[keep][-1]
        sigma = stresses[keep][-1]
        plastic = strain - sigma / E
        errors.append(abs(sigma - (250e6 + C / gamma * (1 - numpy.exp(-gamma * plastic)))))
    assert errors[0] / errors[1] == pytest.approx(2.0, rel=0.2)


shear_modulus = E / (2 * (1 + nu))


def sheared_cube(shear, plasticity):
    """A cube with every face displaced as the simple shear u_x = shear y, solved once."""
    mesh = dualmesh.generate_box_mesh(0, size, 0, size, 0, size, 1, 1, 1)
    problem = dualmesh.Problem(mesh)
    body = problem.add_physics(
        "solid_mechanics", youngs_modulus=E, poissons_ratio=nu, plasticity=plasticity
    )
    body.add_boundary_condition(
        [f"disp_x = {shear!r}*y", "disp_y = 0", "disp_z = 0"], boundary=list(mesh.sideset_names())
    )
    problem.solve(report="none")
    return problem


@pytest.mark.parametrize("h", [0.0, 1e9], ids=["perfect", "linear"])
def test_simple_shear_yields_at_the_shear_yield_stress_of_von_Mises(h):
    """Simple shear gamma beyond yield on sigma_y = sigma_0 + h p, sigma_0 = 250 MPa: the stress
    is the shear tau alone, whose von Mises stress is sqrt(3) tau, so the shear yield stress is
    sigma_y / sqrt(3).  The radial return (Simo and Hughes 1998, Computational Inelasticity,
    chapter 3) flows along the deviator, with the plastic engineering shear sqrt(3) p, so tau =
    mu (gamma - sqrt(3) p), with mu the shear modulus, and the yield sqrt(3) tau = sigma_0 + h p
    gives p = (sqrt(3) mu gamma - sigma_0) / (h + 3 mu).  Beyond yield the shear stress then
    rises with the slope mu h / (h + 3 mu), zero in perfect plasticity, where tau =
    sigma_0 / sqrt(3)."""
    yield_stress = 250e6 if h == 0.0 else f"250e6 + {h!r}*equivalent_plastic_strain"
    shears, taus = (5e-3, 1e-2), []
    for shear in shears:
        problem = sheared_cube(shear, {"yield_stress": yield_stress})
        plastic = (numpy.sqrt(3) * shear_modulus * shear - 250e6) / (h + 3 * shear_modulus)
        tau = shear_modulus * (shear - numpy.sqrt(3) * plastic)
        stress = problem.property_at_centroids("stress")[0]
        assert stress == pytest.approx([0, 0, 0, 0, 0, tau], abs=1e-8 * tau)
        assert problem.property_at_centroids("equivalent_plastic_strain")[0][0] == pytest.approx(
            plastic, rel=1e-8
        )
        taus.append(stress[5])
    slope = (taus[1] - taus[0]) / (shears[1] - shears[0])
    assert slope == pytest.approx(shear_modulus * h / (h + 3 * shear_modulus), abs=1e-6 * E)


@pytest.mark.parametrize("recovery", [0.0, 100.0], ids=["Prager", "Armstrong_Frederick"])
def test_the_back_stress_of_simple_shear_is_its_closed_form(recovery):
    """Simple shear gamma = 1e-2 on sigma_0 = 250 MPa with one back stress X of modulus C =
    20 GPa and dynamic recovery r: dX = (2/3) C d(eps_p) - r X dp (Armstrong and Frederick 1966,
    Prager 1956 for r = 0).  In shear the tensor plastic strain is eps_p,xy = (sqrt(3)/2) p, so
    dX_xy = (C / sqrt(3)) dp - r X_xy dp.  Backward Euler, as the return integrates it, gives in
    one step from the virgin state X_xy = C p / (sqrt(3) (1 + r p)), which is exact for Prager's
    law, linear in p.  With tau = mu (gamma - sqrt(3) p) the yield sqrt(3) (tau - X_xy) =
    sigma_0 is the equation sqrt(3) mu gamma - 3 mu p - C p / (1 + r p) = sigma_0 of p."""
    C, shear = 20e9, 1e-2
    problem = sheared_cube(
        shear,
        {"yield_stress": 250e6, "kinematic_hardening": {"C": [C], "gamma": [recovery]}},
    )

    def yield_function(p):
        return (
            numpy.sqrt(3) * shear_modulus * shear
            - 3 * shear_modulus * p
            - C * p / (1 + recovery * p)
            - 250e6
        )

    plastic = brentq(yield_function, 0.0, 1.0, xtol=1e-16, rtol=1e-15)
    tau = shear_modulus * (shear - numpy.sqrt(3) * plastic)
    back = C * plastic / (numpy.sqrt(3) * (1 + recovery * plastic))
    assert problem.property_at_centroids("equivalent_plastic_strain")[0][0] == pytest.approx(
        plastic, rel=1e-8
    )
    assert problem.property_at_centroids("stress")[0] == pytest.approx(
        [0, 0, 0, 0, 0, tau], abs=1e-8 * tau
    )
    assert problem.property_at_centroids("back_stress")[0] == pytest.approx(
        [0, 0, 0, 0, 0, back], abs=1e-8 * back
    )


def test_plasticity_and_creep_act_together():
    """Under a constant 300 MPa, sigma_y = 250 MPa + 1 GPa p yields p = 0.05 at once, and creep
    at A sigma^n adds A sigma^n t: the strain is sigma/E + 0.05 + A sigma^n t."""
    A, n = 1e-34, 3.0
    problem = cube(
        yield_stress="250e6 + 1e9*equivalent_plastic_strain",
        creep_model="parsed",
        creep_rate="A*von_mises_stress^n",
        creep_constants={"A": A, "n": n},
        temperature="temperature",
    )
    problem.add_boundary_condition("traction", "front", variable="uz", traction=300e6)
    problem.solve(report="none")
    problem.solve_transient(end_time=1e6, time_step=2.5e5, report="none")
    expected = 300e6 / E + 0.05 + A * 300e6**n * 1e6
    assert axial_strain(problem) == pytest.approx(expected, rel=1e-8)


@pytest.mark.parametrize("method", ["FEM", "DMCDM"])
def test_plane_stress_plasticity_is_uniaxial(method):
    """A plate in plane stress pulled along x beyond yield: sigma_zz = 0 holds through the
    return, so the strain is sigma/E + p and the lateral strain -nu sigma/E - p/2."""
    mesh = dualmesh.generate_rectangle_mesh(0, size, 0, size, 2, 2)
    problem = dualmesh.Problem(mesh, method=method)
    for v in ("ux", "uy"):
        problem.add_variable(v)
    problem.add_property(
        "small_strain_stress",
        displacements=["ux", "uy"],
        formulation="plane_stress",
        youngs_modulus=E,
        poissons_ratio=nu,
        yield_stress="250e6 + 1e9*equivalent_plastic_strain",
    )
    for v in ("ux", "uy"):
        problem.add_kernel("stress_divergence", variable=v, displacements=["ux", "uy"])
    problem.add_boundary_condition("Dirichlet", "left", variable="ux", value=0.0)
    problem.add_boundary_condition("Dirichlet", "bottom", variable="uy", value=0.0)
    problem.add_boundary_condition("traction", "right", variable="ux", traction=300e6)
    problem.solve(report="none")
    strain = problem.sample("ux", [[size, size / 2]])[0] / size
    lateral = problem.sample("uy", [[size / 2, size]])[0] / size
    assert strain == pytest.approx(300e6 / E + 0.05, rel=1e-8)
    assert lateral == pytest.approx(-nu * 300e6 / E - 0.025, rel=1e-8)


def test_newton_converges_quadratically_through_the_plastic_tangent():
    """A nonlinear hardening (Voce) beyond yield: the relative Newton steps fall quadratically."""
    problem = cube(yield_stress="250e6 + 100e6*(1 - exp(-20*equivalent_plastic_strain))")
    problem.add_boundary_condition("traction", "front", variable="uz", traction=320e6)
    result = problem.solve(report="none")
    steps = [r.step_norm for r in result.history if 0 < r.step_norm < 1e-2]
    assert len(steps) >= 2
    assert all(b <= 10 * a * a for a, b in zip(steps, steps[1:])), steps


def test_a_neo_hookean_bar_has_the_nominal_stress_of_its_stretch():
    """A neo-Hookean bar with nu = 0 in uniaxial nominal stress P: P = mu (lambda - 1/lambda)."""
    mesh = dualmesh.generate_box_mesh(0, size, 0, size, 0, size, 1, 1, 1)
    problem = dualmesh.Problem(mesh)
    body = problem.add_physics(
        "solid_mechanics", youngs_modulus=3e6, poissons_ratio=0.0, elasticity="neo-Hookean"
    )
    for name, side in (("disp_x", "left"), ("disp_y", "bottom"), ("disp_z", "back")):
        body.add_boundary_condition(f"{name} = 0", boundary=side)
    body.add_boundary_condition("disp_z = 0.5e-3", boundary="front")
    problem.solve(report="none")
    stretch = 1.5
    force = -problem.boundary_flux("disp_z", "front")
    # `nu` = 0: the lateral stretches are 1, and P_zz = mu (lambda - 1/lambda) + lambda_L ln(J)/lambda = mu (lambda - 1/lambda).
    assert force / area == pytest.approx(1.5e6 * (stretch - 1 / stretch), rel=1e-8)


def test_the_physics_level_refuses_what_it_does_not_have():
    mesh = dualmesh.generate_rectangle_mesh(0, 1, 0, 1, 2, 2)
    with pytest.raises(
        InputError, match="elasticity is linear, neo-Hookean or Saint_Venant_Kirchhoff"
    ):
        dualmesh.Problem(mesh).add_physics(
            "solid_mechanics", formulation="plane_strain", elasticity="rubber"
        )
    with pytest.raises(InputError, match="a hyperelastic law has neither"):
        dualmesh.Problem(mesh).add_physics(
            "solid_mechanics",
            formulation="plane_strain",
            elasticity="neo-Hookean",
            plasticity={"yield_stress": 1.0},
        )
    with pytest.raises(InputError, match="plasticity takes"):
        dualmesh.Problem(mesh).add_physics(
            "solid_mechanics", formulation="plane_strain", plasticity={"yield": 1.0}
        )
    problem = dualmesh.Problem(mesh)
    body = problem.add_physics(
        "solid_mechanics", formulation="plane_stress", youngs_modulus=1.0, large_deformation=True
    )
    body.add_boundary_condition("fixed", boundary="left")
    with pytest.raises(InputError, match="not in plane_stress"):
        problem.solve(report="none")
    problem = dualmesh.Problem(mesh)
    problem.add_physics(
        "solid_mechanics",
        formulation="plane_strain",
        youngs_modulus=1.0,
        creep={"rate": "1e-20*von_mises_stress"},
    )
    with pytest.raises(InputError, match="creep needs the temperature"):
        problem.solve(report="none")
