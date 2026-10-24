# SPDX-License-Identifier: LGPL-2.1-or-later
"""Objects defined by expressions: parsed_property, parsed_eigenstrain and the
parsed creep of small_strain_stress.

Each is checked against a built-in object that computes the same thing, or
against an exact solution, and the automatic derivatives are checked through
Newton's method, which converges quadratically only with an exact Jacobian.
"""

from __future__ import annotations

import dualmesh as dm
import numpy as np
import pytest


def test_the_expression_evaluates_arrays_and_names_its_errors():
    e = dm.Expression("a*x^2 + if(x > 1, 1, 0)", ["a", "x"])
    assert e(a=2.0, x=np.array([0.5, 2.0])) == pytest.approx([0.5, 9.0])
    assert e(a=1.0, x=3.0) == pytest.approx(10.0)
    with pytest.raises(Exception, match="no value given for a"):
        e(x=1.0)
    with pytest.raises(Exception, match="has no variable 'b'"):
        e(a=1.0, x=1.0, b=2.0)
    with pytest.raises(Exception, match="the variables are a and x"):
        dm.Expression("a + y", ["a", "x"])


def _conduction(conductivity):
    """-d/dx (k(T) dT/dx) = 1e4 on [0, 1], T = 300 at both ends, with the
    conductivity given by `conductivity(problem)`."""
    mesh = dm.generate_line_mesh(0.0, 1.0, 20)
    p = dm.Problem(mesh)
    p.add_variable("temperature", initial_condition=300.0)
    conductivity(p)
    p.add_kernel("body_force", variable="temperature", value=1e4)
    p.add_boundary_condition(
        "Dirichlet",
        ["left", "right"],
        variable="temperature",
        value=300.0,
    )
    result = p.solve(relative_tolerance=1e-12, absolute_tolerance=1e-10)
    return p, result


def test_parsed_property_matches_the_built_in_polynomial_conductivity():
    """k = 2 (1 + 0.01 T) as an expression and as the polynomial option of
    heat_conduction: the same temperatures, and Newton's method converges in
    the same few iterations (the Jacobian is exact)."""

    def parsed(p):
        p.add_property(
            "parsed",
            property_name="k",
            expression="k0 * (1 + beta * temperature)",
            constants={"k0": 2.0, "beta": 0.01},
        )
        p.add_kernel(
            "heat_conduction",
            variable="temperature",
            thermal_conductivity="k",
        )

    def built_in(p):
        p.add_kernel(
            "heat_conduction",
            variable="temperature",
            thermal_conductivity=2.0,
            temperature_polynomial=[1.0, 0.01],
        )

    a, ra = _conduction(parsed)
    b, rb = _conduction(built_in)
    assert a.values("temperature") == pytest.approx(b.values("temperature"), rel=1e-12)
    assert ra.total_iterations == rb.total_iterations
    assert ra.total_iterations <= 6


def test_parsed_property_reads_functions_and_other_properties():
    mesh = dm.generate_line_mesh(0.0, 1.0, 4)
    p = dm.Problem(mesh)
    p.add_variable("u", initial_condition=2.0)
    p.add_function("ramp", dm.ParsedFunction("10*x + t"))
    p.add_property("constant", properties={"c": 3.0})
    p.add_property(
        "parsed",
        property_name="combined",
        expression="c * u + ramp",
    )
    p.add_kernel("diffusion", variable="u")
    p.apply_initial_conditions()
    centroids = np.array([0.125, 0.375, 0.625, 0.875])
    assert np.asarray(p.property_at_centroids("combined")).ravel() == pytest.approx(
        3.0 * 2.0 + 10 * centroids
    )
    q = dm.Problem(mesh)
    q.add_variable("u")
    q.add_property(
        "parsed",
        property_name="bad",
        expression="missing",
    )
    q.add_kernel("diffusion", variable="u")
    with pytest.raises(Exception, match="unknown name 'missing'"):
        q.apply_initial_conditions()
        q.property_at_centroids("bad")


def _free_bar(eigenstrain):
    """A 3D bar held only against rigid motion, at a uniform temperature
    of 800 K: it expands freely, with no stress."""
    mesh = dm.generate_box_mesh(0, 1, 0, 1, 0, 1, 2, 2, 2)
    p = dm.Problem(mesh)
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
    return np.array([p.sample(v, [corner])[0] for v in ("ux", "uy", "uz")])


@pytest.mark.parametrize("strain_type", ["linear", "volumetric"])
def test_parsed_eigenstrain_matches_thermal_expansion(strain_type):
    """alpha (T - T0) as a linear strain, or 3 alpha (T - T0) as a volumetric
    one, zeroed at the stress-free temperature, gives the displacements of
    thermal_expansion_eigenstrain."""
    alpha, T0 = 1e-5, 300.0

    def parsed(p):
        p.add_property(
            "parsed_eigenstrain",
            expression=("3 * " if strain_type == "volumetric" else "") + "alpha * temperature",
            strain_type=strain_type,
            eigenstrain_name="expansion",
            constants={"alpha": alpha},
            stress_free_state={"temperature": T0},
        )

    def built_in(p):
        p.add_property(
            "thermal_expansion_eigenstrain",
            temperature="temperature",
            linear_expansion_coeff=alpha,
            stress_free_temperature=T0,
            eigenstrain_name="expansion",
        )

    u = _free_bar(parsed)
    assert u == pytest.approx(_free_bar(built_in), rel=1e-10)
    assert u == pytest.approx(np.full(3, alpha * (800.0 - T0)), rel=1e-8)


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


@pytest.mark.parametrize("implicitness", [1.0, 0.5])
def test_parsed_creep_reproduces_a_power_law_exactly(implicitness):
    """A bar under a constant stress creeps at A sigma^n exp(-Q/T); backward
    Euler with the radial return gives the exact strain at any step.  The
    equilibrium has no time derivative, so a step of any theta takes it at the
    new time, where the material history is (with Crank-Nicolson it was taken
    half at the old time, without the history, and the strain came out 185 %
    too large)."""
    A, n, Q, T, stress = 1e-30, 3.0, 2000.0, 700.0, 50e6
    mesh = dm.generate_box_mesh(0, 1e-3, 0, 1e-3, 0, 1e-3, 1, 1, 1)
    p = dm.Problem(mesh)
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
    p.solve_transient(end_time=1e6, time_step=2.5e5, implicitness=implicitness)
    strain = p.sample("uz", [[0.5e-3, 0.5e-3, 1e-3 * (1 - 1e-9)]])[0] / 1e-3
    rate = A * stress**n * np.exp(-Q / T)
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
    mesh = dm.generate_box_mesh(0, 1e-3, 0, 1e-3, 0, 1e-3, 1, 1, 1)
    p = dm.Problem(mesh)
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


def test_parsed_creep_reads_the_grain_size():
    """A diffusional (Coble) form, rate = B sigma / (T d^3): halving the grain
    size d multiplies the creep strain by eight, exactly."""
    B, T, stress, time = 1e-29, 700.0, 50e6, 1e6
    strains = []
    for grain_size in (20e-6, 10e-6):
        p = _creep_bar(
            creep_rate="B * von_mises_stress / (temperature * grain_size^3)",
            creep_constants={"B": B},
            grain_size=grain_size,
        )
        p.solve()
        p.solve_transient(end_time=time, time_step=2.5e5)
        strain = p.sample("uz", [[0.5e-3, 0.5e-3, 1e-3 * (1 - 1e-9)]])[0] / 1e-3
        creep = strain - stress / 1e11
        assert creep == pytest.approx(B * stress / (T * grain_size**3) * time, rel=1e-8)
        strains.append(creep)
    assert strains[1] / strains[0] == pytest.approx(8.0, rel=1e-8)


def test_parsed_creep_that_uses_the_grain_size_requires_it():
    with pytest.raises(Exception, match="a creep_rate that uses grain_size needs 'grain_size'"):
        _creep_bar(creep_rate="1e-20 * von_mises_stress / grain_size^2")
    with pytest.raises(Exception, match="grain_size must be positive"):
        _creep_bar(creep_rate="1e-20 * von_mises_stress / grain_size^2", grain_size=0.0)
    with pytest.raises(Exception, match=r"unknown creep_model 'power_law' \(none, parsed\)"):
        _creep_bar(creep_model="power_law")


def test_a_non_finite_jacobian_is_reported_as_such():
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
