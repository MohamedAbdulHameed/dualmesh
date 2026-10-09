# SPDX-License-Identifier: LGPL-2.1-or-later
"""Verification of the neutron_diffusion physics and the k-eigenvalue study
against the analytical solutions of one- and two-group diffusion theory
(J. C. Lee, Nuclear Reactor Physics and Engineering, 2nd ed., Wiley 2025,
Chapters 5 and 7): k = nu Sigma_f / (Sigma_a + D B_g^2) (Eq. 5.71) for a bare core of
geometrical buckling B_g^2, (pi/H)^2 for a slab of thickness H,
(2.405/R)^2 for an infinite cylinder of radius R and (pi/R)^2 for a sphere,
and the two-group k of Eq. (7.26) with equal bucklings in both groups."""

from __future__ import annotations

import math

import dualmesh as dm
import numpy as np
import pytest
from scipy.optimize import brentq

# One group, SI units: D = 1 cm, Sigma_a = 0.02 /cm, nu Sigma_f = 0.03 /cm.
D, SIGMA_A, NU_SIGMA_F = 0.01, 2.0, 3.0
J0_ZERO = 2.404825557695773


def one_group(mesh, method="fem", coordinates="cartesian", zero_flux=None, vacuum=None):
    problem = dm.Problem(mesh, method=method, coordinates=coordinates)
    neutrons = problem.add_physics("neutron_diffusion", "neutrons", groups=1)
    problem.add_property(
        "multigroup_cross_sections",
        "core",
        diffusion_coefficient=[D],
        absorption_cross_section=[SIGMA_A],
        nu_fission_cross_section=[NU_SIGMA_F],
    )
    if zero_flux:
        neutrons.add_boundary_condition(
            "Dirichlet_boundary_condition", "zero_flux", boundary=zero_flux, value=0.0
        )
    if vacuum:
        neutrons.add_boundary_condition("vacuum_boundary_condition", "vacuum", boundary=vacuum)
    return problem


def k_bare(buckling):
    return NU_SIGMA_F / (SIGMA_A + D * buckling)


def _orders(errors):
    return [math.log2(errors[i] / errors[i + 1]) for i in range(len(errors) - 1)]


@pytest.mark.parametrize("method", ["fem", "dmcdm", "hfvm", "zfvm"])
def test_bare_slab_converges_at_second_order(method):
    """H = 1 m with zero flux at both faces: k = 3 / (2 + 0.01 pi^2) =
    1.4294590."""
    exact = k_bare(math.pi**2)
    errors = []
    for n in (20, 40, 80):
        mesh = dm.generate_line_mesh(start=0.0, end=1.0, num_elements=n)
        errors.append(
            abs(
                one_group(mesh, method, zero_flux=mesh.sideset_names())
                .solve_eigenvalue()
                .k_effective
                - exact
            )
        )
    assert errors[-1] < 2e-5
    assert min(_orders(errors)) > 1.9, errors


def test_the_power_iteration_and_the_arnoldi_method_agree():
    mesh = dm.generate_line_mesh(start=0.0, end=1.0, num_elements=40)
    problem = one_group(mesh, zero_flux=mesh.sideset_names())
    krylov = problem.solve_eigenvalue()
    power = problem.solve_eigenvalue(method="power")
    assert power.k_effective == pytest.approx(krylov.k_effective, rel=1e-8)
    assert krylov.residual_norm < 1e-8 and power.residual_norm < 1e-6
    # The fundamental mode is the cosine, positive, and normalised to one
    # fission neutron in all.
    x = problem.entity_points()[:, 0]
    phi = problem.values("neutron_flux_1")
    assert np.all(phi >= -1e-12)
    shape = np.cos(math.pi * (x - 0.5))
    assert np.max(np.abs(phi / phi.max() - shape)) < 2e-3
    assert NU_SIGMA_F * problem.integrate("neutron_flux_1") == pytest.approx(1.0, rel=1e-10)


def test_the_arnoldi_method_needs_fewer_solves_near_a_dominance_ratio_of_one():
    """A slab of 3 m: the symmetric start excites the odd modes only, and
    the dominance ratio is k_3 / k_1 = (2 + 0.01 pi^2 / 9) / (2 + 0.01 pi^2)
    = 0.958.  The power iteration then needs hundreds of solves with L, and
    the result reports the number of solves of each method."""
    mesh = dm.generate_line_mesh(start=0.0, end=3.0, num_elements=60)
    problem = one_group(mesh, zero_flux=mesh.sideset_names())
    krylov = problem.solve_eigenvalue()
    power = problem.solve_eigenvalue(method="power")
    assert power.k_effective == pytest.approx(krylov.k_effective, rel=1e-8)
    assert 0 < krylov.iterations < power.iterations / 3
    assert power.iterations == len(power.history)
    assert "solves with L" in krylov.summary()


def test_bare_infinite_cylinder_and_sphere():
    """Radius 0.5 m: B^2 = (2.405 / R)^2 in axisymmetric coordinates and
    (pi / R)^2 in spherical coordinates, with zero flux at r = R."""
    radius, n = 0.5, 80
    mesh = dm.generate_line_mesh(start=0.0, end=radius, num_elements=n)
    cylinder = (
        one_group(mesh, coordinates="axisymmetric", zero_flux=["right"])
        .solve_eigenvalue()
        .k_effective
    )
    assert cylinder == pytest.approx(k_bare((J0_ZERO / radius) ** 2), rel=2e-5)
    sphere = (
        one_group(mesh, coordinates="spherical", zero_flux=["right"]).solve_eigenvalue().k_effective
    )
    assert sphere == pytest.approx(k_bare((math.pi / radius) ** 2), rel=2e-5)


def test_finite_cylinder_in_two_dimensions():
    """A bare cylinder of radius R = 0.5 m and height H = 1 m: B^2 =
    (2.405 / R)^2 + (pi / H)^2 (Lee, Sect. 5.3.3)."""
    mesh = dm.generate_rectangle_mesh(
        x_min=0.0, x_max=0.5, y_min=0.0, y_max=1.0, num_x_elements=40, num_y_elements=80
    )
    k = (
        one_group(mesh, coordinates="axisymmetric", zero_flux=["right", "top", "bottom"])
        .solve_eigenvalue()
        .k_effective
    )
    assert k == pytest.approx(k_bare((J0_ZERO / 0.5) ** 2 + math.pi**2), rel=3e-5)


def test_vacuum_boundary_of_a_slab():
    """A slab of half thickness a = 0.5 m, with the vacuum condition
    -D dphi/dn = phi / r at both faces: the flux is cos(B x) with
    D B tan(B a) = 1 / r, the extrapolated half thickness is a + r D, and
    k = nu Sigma_f / (Sigma_a + D B^2).  Checked for r = 2 (no incoming
    current) and r = 2.1312 (the transport extrapolation distance)."""
    a = 0.5
    for ratio in (2.0, 2.1312):
        buckling = (
            brentq(
                lambda b, ratio=ratio: D * b * math.tan(b * a) - 1.0 / ratio,
                1e-6,
                math.pi / (2 * a) - 1e-9,
            )
            ** 2
        )
        mesh = dm.generate_line_mesh(start=-a, end=a, num_elements=200)
        problem = one_group(mesh)
        problem._physics["neutrons"].add_boundary_condition(
            "vacuum_boundary_condition",
            "faces",
            boundary=mesh.sideset_names(),
            extrapolation_distance_ratio=ratio,
        )
        assert problem.solve_eigenvalue().k_effective == pytest.approx(k_bare(buckling), rel=1e-6)
        # The extrapolated distance approximation of Lee, Eq. 5.4.
        assert math.sqrt(buckling) == pytest.approx(math.pi / (2 * (a + ratio * D)), rel=2e-4)


def two_group_problem(mesh, transverse_buckling=0.0, coordinates="cartesian"):
    """The fuel 1 of the IAEA PWR benchmark (ANL-7416 Suppl. 2, problem 11),
    in SI units."""
    problem = dm.Problem(mesh, method="fem", coordinates=coordinates)
    neutrons = problem.add_physics(
        "neutron_diffusion", "neutrons", groups=2, transverse_buckling=transverse_buckling
    )
    problem.add_property(
        "multigroup_cross_sections",
        "fuel",
        diffusion_coefficient=[0.015, 0.004],
        absorption_cross_section=[1.0, 8.0],
        scattering_cross_section=[[0.0, 2.0], [0.0, 0.0]],
        nu_fission_cross_section=[0.0, 13.5],
    )
    return problem, neutrons


def test_two_group_infinite_medium():
    """With reflection at every face and no buckling the flux is flat and
    k_inf = nu Sigma_f2 Sigma_12 / ((Sigma_a1 + Sigma_12) Sigma_a2) = 1.125
    (Lee, Eq. 7.27)."""
    mesh = dm.generate_rectangle_mesh(
        x_min=0.0, x_max=0.2, y_min=0.0, y_max=0.2, num_x_elements=4, num_y_elements=4
    )
    problem, _ = two_group_problem(mesh)
    result = problem.solve_eigenvalue()
    assert result.k_effective == pytest.approx(1.125, rel=1e-12)
    # Thermal to fast flux ratio Sigma_12 / Sigma_a2 = 0.25.
    assert np.allclose(
        problem.values("neutron_flux_2") / problem.values("neutron_flux_1"), 0.25, rtol=1e-10
    )


def test_two_group_bare_core_and_the_transverse_buckling():
    """A square of side 2 m with zero flux, and a transverse buckling of
    0.8e-4 /cm^2: B^2 = 2 (pi / 2)^2 + 0.8, and k = nu Sigma_f2 Sigma_12 /
    ((Sigma_a1 + Sigma_12 + D_1 B^2) (Sigma_a2 + D_2 B^2))."""
    side, transverse = 2.0, 0.8
    mesh = dm.generate_rectangle_mesh(
        x_min=0.0, x_max=side, y_min=0.0, y_max=side, num_x_elements=40, num_y_elements=40
    )
    problem, neutrons = two_group_problem(mesh, transverse_buckling=transverse)
    neutrons.add_boundary_condition(
        "Dirichlet_boundary_condition", "outer", boundary=mesh.sideset_names(), value=0.0
    )
    b2 = 2 * (math.pi / side) ** 2 + transverse
    exact = 13.5 * 2.0 / ((1.0 + 2.0 + 0.015 * b2) * (8.0 + 0.004 * b2))
    assert problem.solve_eigenvalue().k_effective == pytest.approx(exact, rel=1e-4)


def test_cross_sections_and_conditions_are_checked():
    mesh = dm.generate_line_mesh(start=0.0, end=1.0, num_elements=4)
    problem = dm.Problem(mesh)
    neutrons = problem.add_physics("neutron_diffusion", "neutrons", groups=2)
    with pytest.raises(ValueError, match="needs 2 values"):
        problem.add_property(
            "multigroup_cross_sections",
            "bad",
            diffusion_coefficient=[0.01, 0.004],
            absorption_cross_section=[1.0],
        )
        problem.initialize()
    with pytest.raises(ValueError, match="albedo must lie"):
        neutrons.add_boundary_condition("albedo_boundary_condition", "left", albedo=1.5)
    with pytest.raises(ValueError, match="no neutron_diffusion physics"):
        plain = dm.Problem(mesh)
        plain.add_physics("coefficient_form_PDE", "pde", diffusion_coefficient=1.0)
        plain.solve_eigenvalue()


def test_albedo_boundary_condition_equals_its_robin_form():
    """An albedo alpha gives -D dphi/dn = phi (1 - alpha) / (2 (1 + alpha)),
    so alpha = 0 is the vacuum condition with the ratio 2."""
    mesh = dm.generate_line_mesh(start=0.0, end=1.0, num_elements=40)
    ks = []
    for condition, options in (
        ("vacuum_boundary_condition", {}),
        ("albedo_boundary_condition", {"albedo": 0.0}),
    ):
        problem = one_group(mesh)
        problem._physics["neutrons"].add_boundary_condition(
            condition, "faces", boundary=mesh.sideset_names(), **options
        )
        ks.append(problem.solve_eigenvalue().k_effective)
    assert ks[0] == pytest.approx(ks[1], rel=1e-12)
    problem = one_group(mesh)
    problem._physics["neutrons"].add_boundary_condition(
        "albedo_boundary_condition", "faces", boundary=mesh.sideset_names(), albedo=0.5
    )
    assert problem.solve_eigenvalue().k_effective > ks[0]


def test_the_result_protocol(tmp_path):
    mesh = dm.generate_line_mesh(start=0.0, end=1.0, num_elements=10)
    result = one_group(mesh, zero_flux=mesh.sideset_names()).solve_eigenvalue(method="power")
    assert "k_eff" in result.summary()
    assert result.to_dict()["k_effective"] == result.k_effective
    result.write_json(tmp_path / "k.json")
    result.write_csv(tmp_path / "k.csv")
    rows = (tmp_path / "k.csv").read_text().splitlines()
    assert (
        rows[0]
        == "region,volume (m^3),average_flux_1 (1/(m^2 s)),fission_neutron_production (1/s),production_share (%)"
    )
    assert len(rows) == 2 and rows[1].endswith(",100.0")
    assert "Regions" in result.summary() and "k_effective" in result.summary()
