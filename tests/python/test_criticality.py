# SPDX-License-Identifier: LGPL-2.1-or-later
"""Reactor criticality with the generic PDE solver: neutron diffusion written
with coefficient_form_PDE, and k from the eigenvalue study of any problem.

The fission source carries the symbol eigenvalue, which is 1 / k, so the
smallest eigenvalue gives the largest k. The checks are the analytical
solutions of one- and two-group diffusion theory (J. C. Lee, Nuclear Reactor
Physics and Engineering, 2nd ed., Wiley 2025, Chapters 5 and 7): k = nu
Sigma_f / (Sigma_a + D B_g^2) (Eq. 5.71) for a bare core of geometrical
buckling B_g^2, which is (pi/H)^2 for a slab of thickness H, (2.405/R)^2 for
an infinite cylinder of radius R and (pi/R)^2 for a sphere, the two-group k of
Eq. (7.26), the example examples/reactor_criticality.py, and the IAEA 2D PWR
benchmark (ANL-7416 Suppl. 2, problem 11-A2, k = 1.02959).
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import dualmesh as dm
import numpy as np
import pytest
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[2]
# One group, SI units: D = 1 cm, Sigma_a = 0.02 /cm, nu Sigma_f = 0.03 /cm.
D, SIGMA_A, NU_SIGMA_F = 0.01, 2.0, 3.0
J0_ZERO = 2.404825557695773


def one_group(mesh, method="fem", coordinates="cartesian", zero_flux=None, vacuum_ratio=None):
    """One-group diffusion, with zero flux on the side sets ``zero_flux`` or
    the vacuum condition -D dphi/dn = phi / ratio on every side set."""
    problem = dm.Problem(mesh, method=method, coordinates=coordinates)
    neutrons = problem.add_physics(
        "coefficient_form_PDE",
        "neutrons",
        variables=["phi"],
        diffusion_coefficient=D,
        absorption_coefficient=SIGMA_A,
        source="eigenvalue*nu_sigma_f*phi",
        constants={"nu_sigma_f": NU_SIGMA_F},
    )
    if zero_flux:
        neutrons.add_boundary_condition(
            "Dirichlet_boundary_condition", "zero_flux", boundary=list(zero_flux), value=0.0
        )
    if vacuum_ratio:
        neutrons.add_boundary_condition(
            "Robin_boundary_condition",
            "vacuum",
            boundary=list(mesh.sideset_names()),
            transfer_coefficient=1.0 / vacuum_ratio,
        )
    return problem


def k_effective(problem) -> float:
    result = problem.solve_eigenvalue(num_modes=1, report="none")
    return 1.0 / float(result.eigenvalues[0])


def k_bare(buckling):
    return NU_SIGMA_F / (SIGMA_A + D * buckling)


def orders(errors):
    return [math.log2(errors[i] / errors[i + 1]) for i in range(len(errors) - 1)]


@pytest.mark.parametrize("method", ["fem", "dmcdm", "hfvm", "zfvm"])
def test_bare_slab_converges_at_second_order(method):
    """H = 1 m with zero flux at both faces: k = 3 / (2 + 0.01 pi^2)."""
    exact = k_bare(math.pi**2)
    errors = []
    for n in (20, 40, 80):
        mesh = dm.generate_line_mesh(start=0.0, end=1.0, num_elements=n)
        errors.append(
            abs(k_effective(one_group(mesh, method, zero_flux=mesh.sideset_names())) - exact)
        )
    assert errors[-1] < 2e-5
    assert min(orders(errors)) > 1.9, errors


def test_the_fundamental_mode_of_a_slab_is_the_cosine():
    mesh = dm.generate_line_mesh(start=0.0, end=1.0, num_elements=40)
    problem = one_group(mesh, zero_flux=mesh.sideset_names())
    k_effective(problem)
    x = problem.entity_points()[:, 0]
    phi = np.asarray(problem.values("phi"))
    assert np.all(phi >= -1e-12)
    assert np.max(np.abs(phi / phi.max() - np.cos(math.pi * (x - 0.5)))) < 2e-3


def test_bare_infinite_cylinder_and_sphere():
    """Radius 0.5 m: B^2 = (2.405 / R)^2 in axisymmetric coordinates and
    (pi / R)^2 in spherical coordinates, with zero flux at r = R."""
    radius = 0.5
    mesh = dm.generate_line_mesh(start=0.0, end=radius, num_elements=80)
    cylinder = k_effective(one_group(mesh, coordinates="axisymmetric", zero_flux=["right"]))
    assert cylinder == pytest.approx(k_bare((J0_ZERO / radius) ** 2), rel=2e-5)
    sphere = k_effective(one_group(mesh, coordinates="spherical", zero_flux=["right"]))
    assert sphere == pytest.approx(k_bare((math.pi / radius) ** 2), rel=2e-5)


def test_finite_cylinder_in_two_dimensions():
    """A bare cylinder of radius R = 0.5 m and height H = 1 m: B^2 =
    (2.405 / R)^2 + (pi / H)^2 (Lee, Sect. 5.3.3)."""
    mesh = dm.generate_rectangle_mesh(
        x_min=0.0, x_max=0.5, y_min=0.0, y_max=1.0, num_x_elements=40, num_y_elements=80
    )
    problem = one_group(mesh, coordinates="axisymmetric", zero_flux=["right", "top", "bottom"])
    assert k_effective(problem) == pytest.approx(
        k_bare((J0_ZERO / 0.5) ** 2 + math.pi**2), rel=3e-5
    )


@pytest.mark.parametrize("ratio", [2.0, 2.1312])
def test_vacuum_boundary_of_a_slab(ratio):
    """A slab of half thickness a = 0.5 m with -D dphi/dn = phi / r at both
    faces: the flux is cos(B x) with D B tan(B a) = 1 / r, and k = nu Sigma_f /
    (Sigma_a + D B^2). r = 2 is no incoming current, and r = 2.1312 the
    transport extrapolation distance."""
    a = 0.5
    buckling = (
        brentq(lambda b: D * b * math.tan(b * a) - 1.0 / ratio, 1e-6, math.pi / (2 * a) - 1e-9) ** 2
    )
    mesh = dm.generate_line_mesh(start=-a, end=a, num_elements=200)
    assert k_effective(one_group(mesh, vacuum_ratio=ratio)) == pytest.approx(
        k_bare(buckling), rel=1e-6
    )


def two_group(mesh, buckling=0.0):
    """Fuel 1 of the IAEA PWR benchmark in SI units, with a transverse buckling
    that adds D_g B^2 to the removal of each group."""
    problem = dm.Problem(mesh, method="fem")
    neutrons = problem.add_physics(
        "coefficient_form_PDE",
        "neutrons",
        variables=["phi1", "phi2"],
        diffusion_coefficient={"phi1": 0.015, "phi2": 0.004},
        absorption_coefficient={
            "phi1": 1.0 + 2.0 + 0.015 * buckling,
            "phi2": {"phi2": 8.0 + 0.004 * buckling, "phi1": -2.0},
        },
        source={"phi1": "eigenvalue*13.5*phi2"},
    )
    return problem, neutrons


def test_two_group_infinite_medium():
    """With reflection at every face the flux is flat and k_inf = nu Sigma_f2
    Sigma_12 / ((Sigma_a1 + Sigma_12) Sigma_a2) = 1.125 (Lee, Eq. 7.27), with
    the thermal to fast ratio Sigma_12 / Sigma_a2 = 0.25."""
    mesh = dm.generate_rectangle_mesh(
        x_min=0.0, x_max=0.2, y_min=0.0, y_max=0.2, num_x_elements=4, num_y_elements=4
    )
    problem, _ = two_group(mesh)
    assert k_effective(problem) == pytest.approx(1.125, rel=1e-12)
    ratio = np.asarray(problem.values("phi2")) / np.asarray(problem.values("phi1"))
    assert np.allclose(ratio, 0.25, rtol=1e-10)


def test_two_group_bare_core_with_a_transverse_buckling():
    """A square of side 2 m with zero flux and a transverse buckling of 0.8
    /m^2: B^2 = 2 (pi / 2)^2 + 0.8."""
    side, transverse = 2.0, 0.8
    mesh = dm.generate_rectangle_mesh(
        x_min=0.0, x_max=side, y_min=0.0, y_max=side, num_x_elements=40, num_y_elements=40
    )
    problem, neutrons = two_group(mesh, buckling=transverse)
    for flux in ("phi1", "phi2"):
        neutrons.add_boundary_condition(
            "Dirichlet_boundary_condition",
            f"outer_{flux}",
            variable=flux,
            boundary=list(mesh.sideset_names()),
            value=0.0,
        )
    b2 = 2 * (math.pi / side) ** 2 + transverse
    exact = 13.5 * 2.0 / ((1.0 + 2.0 + 0.015 * b2) * (8.0 + 0.004 * b2))
    assert k_effective(problem) == pytest.approx(exact, rel=1e-4)


def test_the_example_matches_two_group_theory():
    spec = importlib.util.spec_from_file_location(
        "reactor_criticality", ROOT / "examples" / "reactor_criticality.py"
    )
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    errors = [abs(example.criticality(n) - example.exact_criticality()) for n in (10, 20, 40)]
    assert errors[-1] < 2e-5 * example.exact_criticality()
    assert min(orders(errors)) > 1.8


def test_the_iaea_pwr_benchmark():
    """The IAEA 2D PWR benchmark with the generic solver: |k - 1.02959| falls
    at second order with the element size, to 5.5 pcm with 8 finite elements
    per assembly."""
    sys.path.insert(0, str(ROOT / "verification" / "benchmarks" / "neutronics"))
    sys.path.insert(0, str(ROOT / "verification" / "benchmarks"))
    import run_iaea_2d_pwr as iaea

    errors = [abs(iaea.solve("fem", divisions)["k"] - iaea.REFERENCE_K) for divisions in (4, 8)]
    assert 1e5 * errors[1] < 6.0
    assert orders(errors)[0] > 1.8
