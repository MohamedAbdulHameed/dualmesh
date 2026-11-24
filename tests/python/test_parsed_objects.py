# SPDX-License-Identifier: LGPL-2.1-or-later
"""Objects defined by expressions: parsed_property, parsed_eigenstrain and the
parsed creep of small_strain_stress.

Each is checked against a built-in object that computes the same thing, or
against an exact solution, and the automatic derivatives are checked through
Newton's method, which converges quadratically only with an exact Jacobian.
"""

from __future__ import annotations

import dualmesh
import numpy
import pytest


def test_the_expression_evaluates_arrays_and_names_its_errors():
    e = dualmesh.Expression("a*x^2 + if(x > 1, 1, 0)", ["a", "x"])
    assert e(a=2.0, x=numpy.array([0.5, 2.0])) == pytest.approx([0.5, 9.0])
    assert e(a=1.0, x=3.0) == pytest.approx(10.0)
    with pytest.raises(Exception, match="no value given for a"):
        e(x=1.0)
    with pytest.raises(Exception, match="has no variable 'b'"):
        e(a=1.0, x=1.0, b=2.0)
    with pytest.raises(Exception, match="the variables are a and x"):
        dualmesh.Expression("a + y", ["a", "x"])


def _conduction(conductivity):
    """-d/dx (k(T) dT/dx) = 1e4 on [0, 1], T = 300 at both ends, with the
    conductivity given by `conductivity(problem)`."""
    mesh = dualmesh.generate_line_mesh(0.0, 1.0, 20)
    p = dualmesh.Problem(mesh)
    p.add_variable("temperature", initial_condition=300.0)
    conductivity(p)
    p.add_kernel("body_force", variable="temperature", value=1e4)
    p.add_boundary_condition(
        "Dirichlet",
        ["left", "right"],
        variable="temperature",
        value=300.0,
    )
    result = p.solve()
    return p, result


def _free_bar(eigenstrain):
    """A 3D bar held only against rigid motion, at a uniform temperature
    of 800 K: it expands freely, with no stress."""
    mesh = dualmesh.generate_box_mesh(0, 1, 0, 1, 0, 1, 2, 2, 2)
    p = dualmesh.Problem(mesh)
    for v in ("temperature", "ux", "uy", "uz"):
        p.add_variable(v)
    p.add_kernel("diffusion", variable="temperature")
    p.add_boundary_condition(
        "Dirichlet",
        list(mesh.sideset_names()),
        variable="temperature",
        value=800.0,
    )
    eigenstrain(p)
    p.add_property(
        "small_strain_stress",
        displacements=["ux", "uy", "uz"],
        formulation="three_dimensional",
        youngs_modulus=1e11,
        poissons_ratio=0.3,
        eigenstrain_names=["expansion"],
    )
    for v in ("ux", "uy", "uz"):
        p.add_kernel("stress_divergence", variable=v, displacements=["ux", "uy", "uz"])
    for v, side in (("ux", "left"), ("uy", "bottom"), ("uz", "back")):
        p.add_boundary_condition("Dirichlet", side, variable=v, value=0.0)
    p.solve()
    corner = [1 - 1e-9, 1 - 1e-9, 1 - 1e-9]
    return numpy.array([p.sample(v, [corner])[0] for v in ("ux", "uy", "uz")])


@pytest.mark.parametrize("strain_type", ["linear", "volumetric"])
def test_parsed_eigenstrain_matches_thermal_expansion(strain_type):
    """alpha (T - T0) as a linear strain, or 3 alpha (T - T0) as a volumetric
    one, zeroed at the stress-free temperature, gives the displacements of
    thermal_expansion_eigenstrain."""
    alpha, t0 = 1e-5, 300.0

    def parsed(p):
        p.add_property(
            "parsed_eigenstrain",
            expression=("3 * " if strain_type == "volumetric" else "") + "alpha * temperature",
            strain_type=strain_type,
            eigenstrain_name="expansion",
            constants={"alpha": alpha},
            stress_free_state={"temperature": t0},
        )

    def built_in(p):
        p.add_property(
            "thermal_expansion_eigenstrain",
            temperature="temperature",
            linear_expansion_coefficient=alpha,
            stress_free_temperature=t0,
            eigenstrain_name="expansion",
        )

    u = _free_bar(parsed)
    assert u == pytest.approx(_free_bar(built_in), rel=1e-10)
    assert u == pytest.approx(numpy.full(3, alpha * (800.0 - t0)), rel=1e-8)


def test_an_axial_parsed_eigenstrain_stretches_only_along_the_axis():
    def axial(p):
        p.add_property(
            "parsed_eigenstrain",
            expression="1e-3",
            strain_type="linear",
            direction="axial",
            eigenstrain_name="expansion",
        )

    u = _free_bar(axial)
    assert u == pytest.approx([0.0, 0.0, 1e-3], abs=1e-12)


@pytest.mark.parametrize("time_scheme", ["backward_Euler", "Crank-Nicolson"])
def test_parsed_creep_reproduces_a_power_law_exactly(time_scheme):
    """A bar under a constant stress creeps at A sigma^n exp(-Q/T); backward
    Euler with the radial return gives the exact strain at any step.  The
    equilibrium has no time derivative, so a step of any theta takes it at the
    new time, where the material history is (with Crank-Nicolson it was taken
    half at the old time, without the history, and the strain came out 185 %
    too large)."""
    A, n, Q, T, stress = 1e-30, 3.0, 2000.0, 700.0, 50e6
    mesh = dualmesh.generate_box_mesh(0, 1e-3, 0, 1e-3, 0, 1e-3, 1, 1, 1)
    p = dualmesh.Problem(mesh)
    for v in ("temperature", "ux", "uy", "uz"):
        p.add_variable(v)
    p.add_kernel("diffusion", variable="temperature")
    p.add_boundary_condition(
        "Dirichlet",
        list(mesh.sideset_names()),
        variable="temperature",
        value=T,
    )
    p.add_property(
        "small_strain_stress",
        displacements=["ux", "uy", "uz"],
        formulation="three_dimensional",
        youngs_modulus=1e11,
        poissons_ratio=0.3,
        creep_model="parsed",
        creep_rate="A * von_mises_stress^n * exp(-Q / temperature)",
        creep_constants={"A": A, "n": n, "Q": Q},
        temperature="temperature",
    )
    for v in ("ux", "uy", "uz"):
        p.add_kernel("stress_divergence", variable=v, displacements=["ux", "uy", "uz"])
    for v, side in (("ux", "left"), ("uy", "bottom"), ("uz", "back")):
        p.add_boundary_condition("Dirichlet", side, variable=v, value=0.0)
    p.add_boundary_condition("traction", "front", variable="uz", traction=stress)
    p.solve()
    p.solve_transient(end_time=1e6, time_step=2.5e5, time_scheme=time_scheme)
    strain = p.sample("uz", [[0.5e-3, 0.5e-3, 1e-3 * (1 - 1e-9)]])[0] / 1e-3
    rate = A * stress**n * numpy.exp(-Q / T)
    assert strain == pytest.approx(stress / 1e11 + rate * 1e6, rel=1e-8)
    with pytest.raises(Exception, match="needs 'creep_rate'"):
        p.add_property(
            "small_strain_stress",
            displacements=["ux", "uy", "uz"],
            formulation="three_dimensional",
            creep_model="parsed",
        )


def _creep_bar(**creep):
    """A unit cube at 700 K under a uniaxial stress of 50 MPa along z."""
    mesh = dualmesh.generate_box_mesh(0, 1e-3, 0, 1e-3, 0, 1e-3, 1, 1, 1)
    p = dualmesh.Problem(mesh)
    for v in ("temperature", "ux", "uy", "uz"):
        p.add_variable(v)
    p.add_kernel("diffusion", variable="temperature")
    p.add_boundary_condition(
        "Dirichlet",
        list(mesh.sideset_names()),
        variable="temperature",
        value=700.0,
    )
    p.add_property(
        "small_strain_stress",
        displacements=["ux", "uy", "uz"],
        formulation="three_dimensional",
        youngs_modulus=1e11,
        poissons_ratio=0.3,
        temperature="temperature",
        **{"creep_model": "parsed", **creep},
    )
    for v in ("ux", "uy", "uz"):
        p.add_kernel("stress_divergence", variable=v, displacements=["ux", "uy", "uz"])
    for v, side in (("ux", "left"), ("uy", "bottom"), ("uz", "back")):
        p.add_boundary_condition("Dirichlet", side, variable=v, value=0.0)
    p.add_boundary_condition("traction", "front", variable="uz", traction=50e6)
    return p


def test_a_non_finite_Jacobian_is_reported_as_such():
    """k = 1 + sqrt(T - 300) has an infinite slope at the initial T = 300.
    The factorization fails, and the error names the cause (a property that
    is not finite) instead of blaming the boundary conditions."""

    def infinite_slope(p):
        p.add_property(
            "parsed",
            property_name="k",
            expression="1 + (temperature - 300)^(1/2)",
        )
        p.add_kernel(
            "heat_conduction",
            variable="temperature",
            thermal_conductivity="k",
        )

    with pytest.raises(RuntimeError, match="NaN or infinite entries"):
        _conduction(infinite_slope)


@pytest.mark.parametrize("method", ["FEM", "DMCDM"])
def test_parsed_creep_in_plane_stress_reproduces_the_power_law(method):
    """A thin plate under a uniaxial stress along x in plane stress creeps at A sigma^n: the
    return mapping holds sigma_zz = 0 at every point, so the axial strain is sigma/E +
    A sigma^n t, the zz strain -nu sigma/E - A sigma^n t / 2 (the creep keeps the volume), and
    Newton's method converges quadratically through the exact tangent."""
    A, n, stress, E, nu = 1e-30, 3.0, 50e6, 1e11, 0.3
    mesh = dualmesh.generate_rectangle_mesh(0, 1e-3, 0, 1e-3, 2, 2)
    p = dualmesh.Problem(mesh, method=method)
    for v in ("temperature", "ux", "uy"):
        p.add_variable(v)
    p.add_kernel("diffusion", variable="temperature")
    p.add_boundary_condition(
        "Dirichlet", list(mesh.sideset_names()), variable="temperature", value=700.0
    )
    p.add_property(
        "small_strain_stress",
        displacements=["ux", "uy"],
        formulation="plane_stress",
        youngs_modulus=E,
        poissons_ratio=nu,
        creep_model="parsed",
        creep_rate="A * von_mises_stress^n",
        creep_constants={"A": A, "n": n},
        temperature="temperature",
    )
    for v in ("ux", "uy"):
        p.add_kernel("stress_divergence", variable=v, displacements=["ux", "uy"])
    p.add_boundary_condition("Dirichlet", "left", variable="ux", value=0.0)
    p.add_boundary_condition("Dirichlet", "bottom", variable="uy", value=0.0)
    p.add_boundary_condition("traction", "right", variable="ux", traction=stress)
    p.solve(report="none")
    result = p.solve_transient(end_time=1e6, time_step=2.5e5, report="none")
    creep = A * stress**n * 1e6
    strain = p.sample("ux", [[1e-3, 0.5e-3]])[0] / 1e-3
    assert strain == pytest.approx(stress / E + creep, rel=1e-8)
    lateral = p.sample("uy", [[0.5e-3, 1e-3]])[0] / 1e-3
    assert lateral == pytest.approx(-nu * stress / E - 0.5 * creep, rel=1e-8)
    # The steps of the first time step fall quadratically (measured 4.7e-3, 7.5e-6, 1.9e-11).
    first = []
    for record in result.history:
        if record.iteration == 1 and first:
            break
        first.append(record.step_norm)
    small = [s for s in first if 0 < s < 1e-2]
    assert all(b <= 10 * a * a for a, b in zip(small, small[1:])), first


def _bar_creep_strain(rate, constants, end_time, time_step):
    """The creep strain of _creep_bar after end_time, from the axial strain."""
    p = _creep_bar(creep_rate=rate, creep_constants=constants)
    p.solve(report="none")
    p.solve_transient(end_time=end_time, time_step=time_step, report="none")
    strain = p.sample("uz", [[0.5e-3, 0.5e-3, 1e-3 * (1 - 1e-9)]])[0] / 1e-3
    return strain - 50e6 / 1e11


def test_time_hardening_creep_takes_the_rate_at_the_end_of_each_step():
    """Primary creep by time hardening, rate = A sigma^n m t^(m-1), whose strain is A sigma^n t^m:
    each backward Euler step adds dt times the rate at its end time exactly, and the strain
    converges to the closed form at the order m of the singularity t^(m-1) at t = 0 (measured
    ratio 1.36 per halving, 2^0.5 = 1.41)."""
    A, n, m = 1e-30, 3.0, 0.5
    rate = "A*von_mises_stress^n*m*t^(m-1)"
    constants = {"A": A, "n": n, "m": m}
    scale = A * 50e6**n
    steps = 4
    times = numpy.arange(1, steps + 1) * 2.5e5
    exact_sum = sum(2.5e5 * scale * m * t ** (m - 1) for t in times)
    assert _bar_creep_strain(rate, constants, 1e6, 2.5e5) == pytest.approx(exact_sum, rel=1e-8)
    closed = scale * 1e6**m
    errors = [abs(_bar_creep_strain(rate, constants, 1e6, 1e6 / k) - closed) for k in (8, 16)]
    assert errors[0] / errors[1] == pytest.approx(2.0**m, rel=0.1)


def test_strain_hardening_creep_solves_the_hardening_at_the_end_of_each_step():
    """Primary creep by strain hardening, rate = A sigma^n (p + p0)^(-k): each step solves
    dp = dt A sigma^n (p_old + dp + p0)^(-k), and the strain converges to the closed form
    (p + p0)^(1+k) - p0^(1+k) = (1 + k) A sigma^n t."""
    A, n, k, p0 = 1e-30, 3.0, 0.5, 1e-6
    rate = "A*von_mises_stress^n*(equivalent_creep_strain + p0)^(-k)"
    constants = {"A": A, "n": n, "k": k, "p0": p0}
    scale = A * 50e6**n
    dt, p = 2.5e5, 0.0
    for _ in range(4):
        old = p
        for _ in range(200):  # The step of backward Euler, by fixed-point iteration
            p = old + dt * scale * (p + p0) ** (-k)
    assert _bar_creep_strain(rate, constants, 1e6, dt) == pytest.approx(p, rel=1e-8)
    closed = ((1 + k) * scale * 1e6 + p0 ** (1 + k)) ** (1 / (1 + k)) - p0
    errors = [abs(_bar_creep_strain(rate, constants, 1e6, 1e6 / s) - closed) for s in (16, 32)]
    assert errors[1] < errors[0]
