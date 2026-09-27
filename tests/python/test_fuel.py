# SPDX-License-Identifier: LGPL-2.1-or-later
"""The fuel performance module.

Every piece is checked against something known independently of the code:

* the material correlations against the values tabulated in, or computed
  from, the sources they come from;
* the burnup units against the defining relation 1 FIMA = E_f N_A / M_HM;
* heat transfer across the pellet-cladding gap against the exact solution of
  a heated cylinder inside a tube, with the energy balance exact, in one, two
  and three dimensions and for all four methods;
* thermal stresses and pressure loading against the classical elasticity
  solutions of a cylinder (Timoshenko and Goodier, section 150) and a thick
  tube (Lamé);
* creep, integrated by the backward Euler method with a radial return,
  against the exact strain of a bar under constant stress;
* the diffusion of fission gas out of a grain against Booth's series
  solution;
* the three rod models (axisymmetric, three-dimensional and 1.5-dimensional)
  against each other, and the rod's bookkeeping (burnup, gas, output) against
  hand calculations.
"""

from __future__ import annotations

import dualmesh as dm
import numpy as np
import pytest
from dualmesh import fuel

METHODS = ["fem", "dmcdm", "hfvm", "zfvm"]
props = fuel.properties
DAY = 86400.0


# ---------------------------------------------------------------------------
# Correlations
# ---------------------------------------------------------------------------
def test_uo2_conductivity_reproduces_fink():
    """Fink (2000): 3.47 W/m/K at 1000 K for 95 % dense UO2 (IAEA-TECDOC-1496
    lists 3.47 and, for fully dense fuel, 3.87)."""
    assert props.uo2_conductivity(1000.0, model="fink") == pytest.approx(3.467, abs=1e-3)
    assert props.uo2_conductivity(
        1000.0, model="fink", theoretical_density_fraction=1.0
    ) == pytest.approx(3.874, abs=1e-3)


def test_uo2_conductivity_burnup_models():
    """The Lucuta factors and the modified NFI model at the values computed
    from the published equations; burnups in FIMA.  The porosity factor is
    relative to 95 % density, so fresh 95 % dense fuel has Fink's
    conductivity exactly, and 96 % dense fuel the Maxwell-Eucken ratio
    ((1 - 0.04)/(1 + 0.02))/((1 - 0.05)/(1 + 0.025))."""
    k1d, k1p, k2p, k4r = props.lucuta_factors(1000.0, 0.03, 0.95)
    assert (k1d, k1p, k2p, k4r) == pytest.approx((0.8345, 1.0023, 1.0, 0.9555), abs=1e-4)
    assert props.uo2_conductivity(1000.0, 0.0, 0.95) == pytest.approx(
        props.uo2_conductivity(1000.0, 0.0, 0.95, "fink"), rel=1e-12
    )
    ratio = (0.96 / 1.02) / (0.95 / 1.025)
    assert props.uo2_conductivity(1000.0, 0.0, 0.96) == pytest.approx(
        ratio * props.uo2_conductivity(1000.0, 0.0, 0.95), rel=1e-12
    )
    for burnup, expected in ((0.0, 3.434), (30.0, 2.563), (60.0, 2.132)):
        assert props.uo2_conductivity(1000.0, burnup / 938.3, 0.95, "nfi") == pytest.approx(
            expected, abs=1e-3
        )
    # Burnup degrades the conductivity.
    assert props.uo2_conductivity(1000.0, 0.03) < props.uo2_conductivity(1000.0, 0.0)
    # Halden correlation (IAEA-TECDOC-1496, Sect. 6.1.2), 95 % dense fuel at
    # 1000 C: 1/(0.1148 + 2.475e-4 T) + 0.0132 exp(0.00188 T) = 2.8466 W/m/K
    # fresh, and with 0.0035 B and (1 - 0.00333 B) at B = 10 MWd/kgUO2,
    # 2.6568 W/m/K.  The density factor 1.0789 d/(1 + 0.5 (1 - d)) is 1 at
    # d = 0.95 to 4e-5.
    ratio = (0.2700277 - 2 * 15.999e-3) / 0.2700277  # kgHM per kgUO2
    halden = props.uo2_conductivity(1273.15, 0.0, 0.95, "halden")
    assert halden == pytest.approx(2.8466 * 0.99996, abs=1e-4)
    halden = props.uo2_conductivity(1273.15, 10.0 / ratio / 938.3, 0.95, "halden")
    assert halden == pytest.approx(2.6568 * 0.99996, abs=1e-4)
    with pytest.raises(Exception, match="fink, fink_lucuta, nfi or halden"):
        props.uo2_conductivity(1000.0, model="fink95")


def test_uo2_heat_capacity_expansion_and_swelling():
    assert props.uo2_specific_heat(1000.0) == pytest.approx(311.7, abs=0.1)
    assert props.uo2_specific_heat(3000.0) == pytest.approx(725.8, abs=0.1)
    assert props.uo2_specific_heat(3000.0, "matpro") == pytest.approx(699.7, abs=0.1)
    # Martin (1988): the density at 1000 K is 10 726 kg/m^3 from 10 963 at 273 K.
    strain = props.uo2_thermal_strain(1000.0, stress_free_temperature=273.0)
    assert 10963.0 / (1 + strain) ** 3 == pytest.approx(10726.0, abs=1.0)
    # MATPRO solid swelling, 2.5e-29 m^3 per fission: with 2.3e28 heavy-metal
    # atoms per m^3 of 95 % dense UO2, 0.58 per FIMA.
    atoms = 0.95 * 10963.0 * 6.02214076e23 / 0.2700277
    assert props.uo2_solid_swelling(0.01, 0.95) == pytest.approx(2.5e-29 * atoms * 0.01, rel=1e-9)
    # FSWELL: nothing at 2800 K and above, and linear in the burnup increment.
    assert props.uo2_gaseous_swelling_increment(2900.0, 0.01, 0.001) == 0.0
    one = props.uo2_gaseous_swelling_increment(1500.0, 0.01, 0.002)
    two = props.uo2_gaseous_swelling_increment(1500.0, 0.01, 0.001)
    assert one == pytest.approx(2 * two, rel=1e-12)


def test_uranium_nitride_correlations():
    """Hayes et al. (1990) and Ross et al. (1990), with densities as fractions
    of the theoretical density and burnups in FIMA."""
    assert props.un_conductivity(1000.0, 1.0) == pytest.approx(1.864 * 1000.0**0.361, rel=1e-12)
    assert props.un_density(298.0) == pytest.approx(14333.0, abs=1.0)
    assert props.un_youngs_modulus(298.0, 1.0) / 1e9 == pytest.approx(258.5, abs=0.1)
    assert props.un_poissons_ratio(1.0) == pytest.approx(0.281, abs=1e-3)
    assert props.un_volumetric_swelling(1500.0, 0.02, 0.95) == pytest.approx(0.0661, abs=1e-4)
    # Storms (1988): release rises with temperature.
    hot = props.un_fission_gas_release_fraction(2000.0, 0.02, 0.95)
    assert hot > props.un_fission_gas_release_fraction(1500.0, 0.02, 0.95)


def test_zircaloy_and_gas_correlations():
    # MATPRO elastic constants give nu = E / 2G - 1 = 0.386 at 600 K.
    nu = props.zircaloy_youngs_modulus(600.0) / (2 * props.zircaloy_shear_modulus(600.0)) - 1
    assert nu == pytest.approx(0.386, abs=1e-3)
    # IAEA-TECDOC-1496 Table 1 (Sect. 6.2.1.4): 15.67 W/m/K at 600 K and
    # 21.21 at 1000 K; held at the 1800 K value above 1800 K.
    assert props.zircaloy_conductivity(600.0) == pytest.approx(15.67, abs=0.005)
    assert props.zircaloy_conductivity(1000.0) == pytest.approx(21.21, abs=0.005)
    assert props.zircaloy_conductivity(2000.0) == props.zircaloy_conductivity(1800.0)
    # Kestin et al. (1984) Tables 1, 3, 4, 5 at 600 C (873.15 K), mW/m/K.
    for gas, value in (("helium", 328.43), ("argon", 39.76), ("krypton", 22.60), ("xenon", 14.11)):
        assert props.gas_conductivity({gas: 1.0}, 873.15) * 1e3 == pytest.approx(value, rel=0.01)
    # A mixture lies between its parts.
    mix = props.gas_conductivity({"helium": 0.5, "xenon": 0.5}, 600.0)
    assert props.gas_conductivity({"xenon": 1.0}, 600.0) < mix
    assert mix < props.gas_conductivity({"helium": 1.0}, 600.0)
    # The accommodation coefficient of helium is bounded below by 0.07, so the
    # jump distance stays finite and positive where the linear fit would turn
    # negative (1848 K).
    hot = props.gas_jump_distance({"helium": 1.0}, 1900.0, 5e6)
    assert 0 < hot < 1e-3
    # Kennard's jump distance scales as 1 / P.
    j1 = props.gas_jump_distance({"helium": 1.0}, 600.0, 1e6)
    j2 = props.gas_jump_distance({"helium": 1.0}, 600.0, 2e6)
    assert j1 / j2 == pytest.approx(2.0, rel=1e-12)
    growth = props.zircaloy_irradiation_growth([1e25, 2e25])
    assert 0 < growth[0] < growth[1]


# ---------------------------------------------------------------------------
# Burnup units
# ---------------------------------------------------------------------------
def test_one_fima_is_938_megawatt_days_per_kilogram():
    """E_f N_A / M_HM with 200 MeV and natural uranium is 938.3 MWd/kgHM, the
    conversion of the FRAPCON-4.0 property report; the enrichment of the fuel
    shifts it through the molar mass."""
    from dualmesh.fuel.units import mwd_per_kg_per_fima

    assert mwd_per_kg_per_fima() == pytest.approx(938.3, abs=0.1)
    converter = fuel.BurnupConverter()
    assert converter.convert(1.0, "FIMA", "GWd/tHM") == pytest.approx(938.3, abs=0.1)
    assert converter.convert(24.0, "MWh/kgHM", "MWd/kgHM") == pytest.approx(1.0)
    assert converter.convert(1000.0, "MWd/tU", "MWd/kgU") == pytest.approx(1.0)
    values = np.array([0.0, 10.0, 60.0])
    back = converter.to_fima(converter.from_fima(values, "MWd/kgHM"), "MWd/kgHM")
    assert back == pytest.approx(values)
    assert fuel.UO2Fuel(enrichment=0.05).heavy_metal_molar_mass < 0.238029
    with pytest.raises(ValueError, match="Unknown burnup unit"):
        converter.to_fima(1.0, "atom_percent")


def test_an_object_gives_the_same_answer_in_either_burnup_unit():
    """UO2_thermal given the same burnup in FIMA and in MWd/kgHM computes the
    same conductivity."""
    from dualmesh.fuel.units import mwd_per_kg_per_fima

    mesh = dm.generate_line_mesh(0.0, 1.0, 2)
    conductivities = []
    for burnup, unit in ((0.04, "FIMA"), (0.04 * mwd_per_kg_per_fima(), "MWd/kgHM")):
        p = dm.Problem(mesh)
        p.add_variable("T", initial_condition=1200.0)
        p.add_material("UO2_thermal", "uo2", temperature="T", burnup=burnup, burnup_unit=unit)
        p.add_kernel(
            "heat_conduction",
            "k",
            variable="T",
            thermal_conductivity_property="thermal_conductivity",
        )
        p.apply_initial_conditions()
        conductivities.append(np.asarray(p.property_at_centroids("thermal_conductivity")))
    assert conductivities[0] == pytest.approx(conductivities[1], rel=1e-13)
    # And the burnup lowers it: 4 % FIMA against fresh fuel.
    assert conductivities[0][0] < props.uo2_conductivity(1200.0, 0.0)


# ---------------------------------------------------------------------------
# Heat transfer across the gap
# ---------------------------------------------------------------------------
A, B, C = 4.1e-3, 4.18e-3, 4.75e-3
KF, KC, H, Q3, T0 = 3.0, 16.0, 5000.0, 3.0e8, 600.0


def _geometry(fuel_stack_height=0.02, pellet_inner_radius=0.0):
    return fuel.RodGeometry(
        pellet_outer_radius=A,
        clad_inner_radius=B,
        clad_outer_radius=C,
        fuel_stack_height=fuel_stack_height,
        pellet_inner_radius=pellet_inner_radius,
    )


def _exact_temperature(r):
    qp = Q3 * np.pi * A**2
    t_clad_inner = T0 + qp / (2 * np.pi * KC) * np.log(C / B)
    t_surface = t_clad_inner + qp / (2 * np.pi * A * H)
    return np.where(
        r <= A + 1e-9,
        t_surface + Q3 * (A**2 - r**2) / (4 * KF),
        T0 + qp / (2 * np.pi * KC) * np.log(C / np.maximum(r, B)),
    )


def _gap_problem(mesh, method, coordinates):
    p = dm.Problem(mesh, method=method, coordinates=coordinates)
    p.add_variable("T")
    p.add_kernel("heat_conduction", "kf", variable="T", thermal_conductivity=KF, block=["fuel"])
    p.add_kernel("heat_conduction", "kc", variable="T", thermal_conductivity=KC, block=["clad"])
    p.add_kernel("heat_source", "q", variable="T", heat_source=Q3, block=["fuel"])
    p.add_boundary_condition(
        "gap_heat_transfer",
        "gap",
        variable="T",
        boundary=["fuel_outer"],
        secondary_boundary=["clad_inner"],
        gap_conductance=H,
    )
    p.add_boundary_condition(
        "Dirichlet_boundary_condition", "cool", variable="T", boundary=["clad_outer"], value=T0
    )
    p.solve()
    return p


@pytest.mark.parametrize("method", METHODS)
def test_gap_heat_transfer_in_a_radial_slice(method):
    """A heated rod inside a tube, one-dimensional: second-order convergence
    to the exact temperatures, and every watt generated leaves through the
    outer surface of the tube."""
    errors = []
    for n in (8, 16, 32):
        mesh = fuel.radial_slice_mesh(
            _geometry(),
            fuel.RodMesh(
                num_fuel_radial_elements=n,
                num_clad_radial_elements=max(2, n // 4),
                fuel_surface_grading=1.0,
            ),
        )
        p = _gap_problem(mesh, method, "axisymmetric")
        r = np.abs(p.entity_points()[:, 0])
        errors.append(np.abs(p.values("T") - _exact_temperature(r)).max())
        assert -p.total_reaction("T", "clad_outer") == pytest.approx(Q3 * np.pi * A**2, rel=1e-10)
    rates = np.log2(np.array(errors[:-1]) / np.array(errors[1:]))
    assert rates.min() > 1.7


@pytest.mark.parametrize("method", METHODS)
def test_gap_heat_transfer_axisymmetric(method):
    geometry = _geometry()
    mesh = fuel.axisymmetric_rod_mesh(
        geometry,
        fuel.RodMesh(num_fuel_radial_elements=24, num_clad_radial_elements=6, num_axial_elements=3),
    )
    p = _gap_problem(mesh, method, "axisymmetric")
    r = np.abs(p.entity_points()[:, 0])
    assert np.abs(p.values("T") - _exact_temperature(r)).max() < 1.0
    power = Q3 * np.pi * A**2 * geometry.fuel_stack_height
    assert -p.total_reaction("T", "clad_outer") == pytest.approx(power, rel=1e-10)


@pytest.mark.parametrize("method", ["fem", "zfvm"])
def test_gap_heat_transfer_in_three_dimensions(method):
    """The polygonal cross-section holds less fuel than the circle, so the
    power balance is checked against the meshed volume and the temperature
    error must fall as the section is refined."""
    errors = []
    for core, rings in ((4, 6), (8, 12)):
        mesh = fuel.three_dimensional_rod_mesh(
            _geometry(),
            fuel.RodMesh(
                num_fuel_core_divisions=core,
                num_fuel_radial_elements=rings,
                num_clad_radial_elements=2,
                num_axial_elements=1,
                fuel_surface_grading=0.6,
            ),
        )
        p = _gap_problem(mesh, method, "cartesian")
        pts = p.entity_points()
        errors.append(
            np.abs(p.values("T") - _exact_temperature(np.hypot(pts[:, 0], pts[:, 1]))).max()
        )
        volume = sum(
            m
            for m, e in zip(mesh.element_measures(), range(mesh.num_elements))
            if mesh.element_block(e) == 0
        )
        assert -p.total_reaction("T", "clad_outer") == pytest.approx(Q3 * volume, rel=1e-9)
    assert errors[1] < 0.35 * errors[0]


@pytest.mark.parametrize("builder", ["slice", "axisymmetric", "three_dimensional"])
def test_annular_pellet_meshes_have_a_bore(builder):
    """An annular pellet has a fuel_inner side set instead of the axis, and
    its meshed fuel volume is the annulus."""
    geometry = _geometry(fuel_stack_height=0.01, pellet_inner_radius=0.9e-3)
    rod_mesh = fuel.RodMesh(num_axial_elements=2, num_fuel_core_divisions=6)
    if builder == "slice":
        mesh = fuel.radial_slice_mesh(geometry, rod_mesh)
    elif builder == "axisymmetric":
        mesh = fuel.axisymmetric_rod_mesh(geometry, rod_mesh)
    else:
        mesh = fuel.three_dimensional_rod_mesh(geometry, rod_mesh)
    names = set(mesh.sideset_names())
    assert "fuel_inner" in names and "axis" not in names
    if builder == "axisymmetric":
        area = sum(
            m
            for m, e in zip(mesh.element_measures(), range(mesh.num_elements))
            if mesh.element_block(e) == 0
        )
        assert area == pytest.approx((A - 0.9e-3) * 0.01, rel=1e-12)


# ---------------------------------------------------------------------------
# Mechanics
# ---------------------------------------------------------------------------
E, NU, ALPHA, DT = 2e11, 0.3, 1e-5, 500.0


@pytest.mark.parametrize("method", METHODS)
def test_thermal_stress_of_a_long_cylinder(method):
    """A solid cylinder with T = dT (1 - r^2/a^2), free to expand axially
    (generalized plane strain with the axial strain alpha dT / 2 that makes
    the axial force zero): sigma_rr, sigma_zz and sigma_tt of Timoshenko and
    Goodier.  The centroid stresses converge at order 1.5 or better."""
    a = 5e-3
    errors = []
    for n in (8, 16, 32):
        points = np.linspace(0, a, n + 1)[:, None]
        mesh = dm.mesh_from_arrays(points, [[i, i + 1] for i in range(n)], "Edge2", dimension=1)
        mesh.add_sideset_from_faces("axis", [[0]])
        mesh.add_sideset_from_faces("surface", [[n]])
        p = dm.Problem(mesh, method=method, coordinates="axisymmetric")
        p.add_variable("T")
        p.add_variable("u")
        p.add_kernel("diffusion", "conduction", variable="T")
        p.add_kernel("body_force", "heat", variable="T", value=4 * DT / a**2)
        p.add_boundary_condition(
            "Dirichlet_boundary_condition",
            "surface_T",
            variable="T",
            boundary=["surface"],
            value=0.0,
        )
        p.add_material(
            "thermal_expansion_eigenstrain",
            "thermal",
            temperature="T",
            thermal_expansion_coefficient=ALPHA,
            stress_free_temperature=0.0,
            eigenstrain_name="thermal_strain",
        )
        p.add_material(
            "small_strain_stress",
            "stress",
            displacements=["u"],
            formulation="axisymmetric_1d",
            youngs_modulus=E,
            poissons_ratio=NU,
            eigenstrain_names=["thermal_strain"],
            axial_strain=ALPHA * DT / 2,
        )
        p.add_kernel("stress_divergence", "equilibrium", variable="u", component=0)
        p.add_boundary_condition(
            "Dirichlet_boundary_condition", "axis", variable="u", boundary=["axis"], value=0.0
        )
        p.solve()
        s = np.asarray(p.property_at_centroids("stress"))
        r = p.entity_points()[:n, 0] if method == "zfvm" else 0.5 * (points[1:, 0] + points[:-1, 0])
        c = ALPHA * E * DT / (1 - NU)
        exact = np.column_stack(
            [c * (r**2 / a**2 - 1) / 4, c * (r**2 / a**2 - 0.5), c * (3 * r**2 / a**2 - 1) / 4]
        )
        errors.append(np.abs(s[:, :3] - exact).max() / c)
    assert errors[-1] < 1e-3
    assert np.log2(errors[-2] / errors[-1]) > 1.4


@pytest.mark.parametrize("method", METHODS)
def test_thick_tube_under_pressure(method):
    """Lamé: a tube with internal pressure p_i and external p_o in plane
    strain; the hoop stress sigma_tt(r) = (p_i b^2 - p_o c^2)/(c^2 - b^2) +
    (p_i - p_o) b^2 c^2 / ((c^2 - b^2) r^2)."""
    b, c, pi, po = 4.18e-3, 4.75e-3, 10e6, 15.5e6
    n = 16
    points = np.linspace(b, c, n + 1)[:, None]
    mesh = dm.mesh_from_arrays(points, [[i, i + 1] for i in range(n)], "Edge2", dimension=1)
    mesh.add_sideset_from_faces("inner", [[0]])
    mesh.add_sideset_from_faces("outer", [[n]])
    p = dm.Problem(mesh, method=method, coordinates="axisymmetric")
    p.add_variable("u")
    p.add_material(
        "small_strain_stress",
        "stress",
        displacements=["u"],
        formulation="axisymmetric_1d",
        youngs_modulus=E,
        poissons_ratio=NU,
    )
    p.add_kernel("stress_divergence", "equilibrium", variable="u", component=0)
    p.add_boundary_condition(
        "pressure_boundary_condition",
        "p_in",
        variable="u",
        component=0,
        boundary=["inner"],
        pressure=pi,
    )
    p.add_boundary_condition(
        "pressure_boundary_condition",
        "p_out",
        variable="u",
        component=0,
        boundary=["outer"],
        pressure=po,
    )
    p.solve()
    s = np.asarray(p.property_at_centroids("stress"))
    r = p.entity_points()[:n, 0] if method == "zfvm" else 0.5 * (points[1:, 0] + points[:-1, 0])
    k = (pi * b**2 - po * c**2) / (c**2 - b**2)
    hoop = k + (pi - po) * b**2 * c**2 / ((c**2 - b**2) * r**2)
    assert np.abs(s[:, 2] - hoop).max() < 2e-3 * abs(hoop).max()


def _creep_bar(model, temperature, stress, method, extra):
    mesh = dm.generate_box_mesh(0, 1e-3, 0, 1e-3, 0, 1e-3, 2, 2, 2)
    p = dm.Problem(mesh, method=method)
    for v in ("T", "ux", "uy", "uz"):
        p.add_variable(v)
    p.add_kernel("diffusion", "conduction", variable="T")
    p.add_boundary_condition(
        "Dirichlet_boundary_condition",
        "T",
        variable="T",
        boundary=list(mesh.sideset_names()),
        value=temperature,
    )
    p.add_material(
        "small_strain_stress",
        "stress",
        displacements=["ux", "uy", "uz"],
        formulation="three_dimensional",
        youngs_modulus=1e11,
        poissons_ratio=0.3,
        creep_model=model,
        temperature="T",
        **extra,
    )
    for i, v in enumerate(("ux", "uy", "uz")):
        p.add_kernel("stress_divergence", f"equilibrium_{v}", variable=v, component=i)
    for name, variable, side in (("x", "ux", "left"), ("y", "uy", "bottom"), ("z", "uz", "back")):
        p.add_boundary_condition(
            "Dirichlet_boundary_condition", name, variable=variable, boundary=[side], value=0.0
        )
    p.add_boundary_condition(
        "traction_boundary_condition", "pull", variable="uz", boundary=["front"], traction=stress
    )
    p.solve()
    p.solve_transient(end_time=1e6, dt=1e5)
    return p.sample("uz", [[0.5e-3, 0.5e-3, 1e-3 * (1 - 1e-9)]])[0] / 1e-3


@pytest.mark.parametrize("method", ["fem", "dmcdm", "zfvm"])
@pytest.mark.parametrize(
    "model, temperature, stress, extra, rate",
    [
        (
            "zircaloy",
            650.0,
            150e6,
            {"fast_neutron_flux": 1e18},
            lambda: props.zircaloy_creep_rate(150e6, 650.0, fast_neutron_flux=1e18),
        ),
        (
            "uo2",
            1400.0,
            40e6,
            {"fission_rate": 1e19},
            lambda: props.uo2_creep_rate(40e6, 1400.0, 1e19, 0.95, 5e-6),
        ),
        (
            "un",
            1400.0,
            30e6,
            {"fission_rate": 1e19},
            lambda: props.un_creep_rate(30e6, 1400.0, 1e19, 0.95, 5e-6),
        ),
    ],
)
def test_creep_under_constant_stress(method, model, temperature, stress, extra, rate):
    """Under a constant uniaxial stress the creep strain grows as rate * t,
    which the backward Euler radial return reproduces exactly, whatever the
    step, at every integration point of every method."""
    strain = _creep_bar(model, temperature, stress, method, extra)
    assert strain == pytest.approx(stress / 1e11 + rate() * 1e6, rel=1e-7)


# ---------------------------------------------------------------------------
# Fission gas
# ---------------------------------------------------------------------------
def test_booth_diffusion_is_exact():
    """The spectral solution of the diffusion in a grain reproduces Booth's
    release fraction f = 1 - 6/(pi^4 tau) sum (1 - exp(-n^2 pi^2 tau))/n^4
    over steps of any size."""
    model = fuel.BoothFissionGasRelease(1, num_modes=400, trapping="constant", micro_cracking=False)
    T, F = np.array([1600.0]), np.array([1e19])
    D = model.diffusivity(T, F)[0]
    t = 0.0
    n = np.arange(1, 20001)
    for dt in (1e5, 1e6, 1e7):
        model.advance(dt, T, F, 0.0)
        t += dt
        tau = D * t / model.grain_radius**2
        exact = 1 - 6 / (np.pi**4 * tau) * np.sum((1 - np.exp(-(n**2) * np.pi**2 * tau)) / n**4)
        arrived = 1 - model.intragranular()[0] / model.produced[0]
        assert arrived == pytest.approx(exact, rel=1e-5)


# ---------------------------------------------------------------------------
# The rod
# ---------------------------------------------------------------------------
def _rod(
    model="axisymmetric",
    method="fem",
    fuel_material=None,
    geometry=None,
    coolant=None,
    history=None,
    models=None,
    mesh=None,
    output_times=None,
    max_time_step=10 * DAY,
    **output,
):
    geometry = geometry or fuel.RodGeometry.from_diameters(8.19e-3, 8.36e-3, 9.50e-3, 0.05)
    coolant = coolant or fuel.ForcedConvection(
        inlet_temperature=565.0, pressure=15.5e6, mass_flux=3800.0, rod_pitch=12.6e-3
    )
    history = history or fuel.PowerHistory(
        linear_heat_rate=[1e3, 25e3, 25e3], time=[0, 3600, 100 * DAY]
    )
    rod = fuel.FuelRod(
        geometry,
        fuel_material or fuel.UO2Fuel(),
        fuel.ZircaloyCladding(),
        fuel.FillGas(pressure=2.0e6, plenum_volume=0.15e-6),
        coolant,
        history,
        models=models,
        numerics=fuel.RodNumerics(
            model=model,
            method=method,
            mesh=mesh or fuel.RodMesh(num_axial_elements=2, num_axial_slices=1),
            max_time_step=max_time_step,
        ),
        output=fuel.RodOutput(
            output_times=output_times, print_input=False, print_steps=False, **output
        ),
    )
    return rod, rod.run()


def test_axisymmetric_and_one_and_a_half_dimensional_rods_agree():
    """The same rod, with thermal expansion, densification, swelling,
    relocation, creep and fission gas release, in two models and two
    methods: the centerline temperatures agree within 10 K and the gaps
    within 2.5 micrometres (about 7 % of the gap after 100 days; the 1.5D
    slices have no axial shear between fuel and cladding, which the r-z model
    has, so the two drift apart slowly as the fuel creeps and swells)."""
    times = [0, 3600, 50 * DAY, 100 * DAY]
    _, rz = _rod("axisymmetric", "fem", output_times=times)
    _, slices = _rod("1.5d", "dmcdm", output_times=times)
    T_rz, T_slices = rz.fuel_centerline_temperature, slices.fuel_centerline_temperature
    assert np.abs(T_rz.max(axis=1) - T_slices.max(axis=1)).max() < 10.0
    assert np.abs(rz.gap_width.mean(axis=1) - slices.gap_width.mean(axis=1)).max() < 2.5e-6
    # The hot gap is narrower than the cold one, and the cladding creeps down
    # under the coolant pressure.
    assert rz.gap_width[1, 0] < rz.gap_width[0, 0]
    assert rz.clad_hoop_strain[-1, 0] < rz.clad_hoop_strain[1, 0]


def test_pellet_cladding_contact_closes_the_gap():
    """A rod with a 30 micrometre gap at 40 kW/m: the gap closes, the
    penetration is small (the penalty is stiff) and the four methods agree."""
    geometry = fuel.RodGeometry(
        pellet_outer_radius=4.15e-3,
        clad_inner_radius=4.18e-3,
        clad_outer_radius=4.75e-3,
        fuel_stack_height=0.05,
    )
    history = fuel.PowerHistory(linear_heat_rate=[1e3, 40e3, 40e3], time=[0, 3600, 7200])
    models = fuel.RodModels(creep=False, fission_gas_release="none")
    hoops = []
    for method in METHODS:
        _, out = _rod(
            "axisymmetric",
            method,
            geometry=geometry,
            history=history,
            models=models,
            max_time_step=None,
        )
        assert -0.2e-6 < out.gap_width[-1, 0] < 0.0
        hoops.append(out.clad_hoop_strain[-1, 0])
    assert max(hoops) - min(hoops) < 0.03 * max(hoops)


def test_uranium_nitride_rod_runs_with_all_its_models():
    """A UN rod (Hayes conductivity, Ross swelling, Hayes creep, Storms gas
    release) in the axisymmetric model."""
    history = fuel.PowerHistory(linear_heat_rate=[1e3, 40e3, 40e3], time=[0, 3600, 200 * DAY])
    rod, out = _rod(
        fuel_material=fuel.UNFuel(),
        history=history,
        output_times=[0, 3600, 100 * DAY, 200 * DAY],
        max_time_step=20 * DAY,
    )
    assert rod.active_models.relocation is False
    assert rod.active_models.fission_gas_release == "storms"
    # UN conducts heat far better than UO2: at 40 kW/m its centerline stays
    # a few hundred kelvin above the surface.
    rise = out.fuel_centerline_temperature[-1, 0] - out.fuel_surface_temperature[-1, 0]
    assert 100.0 < rise < 400.0
    assert 0.0 <= out.fission_gas_release[-1] < 1.0
    with pytest.raises(ValueError, match="relocation is not modelled for UN fuel"):
        _rod(fuel_material=fuel.UNFuel(), models=fuel.RodModels(relocation=True))


def test_three_dimensional_rod_agrees_with_the_axisymmetric_one():
    """Thermal only.  The 3D cross-section of the pellet is a polygon with
    4 num_fuel_core_divisions sides, which holds less fuel than the circle
    (98.6 % with 32 sides) and so generates less heat; with 32 sides the
    centerline rise agrees with the axisymmetric one within 3 %, and it
    approaches it as the polygon is refined."""
    history = fuel.PowerHistory(linear_heat_rate=[20e3, 20e3], time=[0, 3600])
    models = fuel.RodModels(mechanics=False, fission_gas_release="none")
    rises = {}
    for model, core in (("axisymmetric", 4), ("three_dimensional", 4), ("three_dimensional", 8)):
        _, out = _rod(
            model,
            history=history,
            models=models,
            mesh=fuel.RodMesh(
                num_axial_elements=2,
                num_fuel_core_divisions=core,
                num_fuel_radial_elements=8 if model == "three_dimensional" else 12,
            ),
            max_time_step=None,
        )
        rises[model, core] = (
            out.fuel_centerline_temperature[-1, 0] - out.clad_outer_temperature[-1, 0]
        )
    reference = rises["axisymmetric", 4]
    assert rises["three_dimensional", 8] == pytest.approx(reference, rel=0.03)
    assert abs(rises["three_dimensional", 8] - reference) < abs(
        rises["three_dimensional", 4] - reference
    )


def test_the_rod_keeps_its_books():
    """The burnup follows from the energy: q' t / (E_f n_HM A); the gas at
    zero release is the fill gas, whose pressure is n R / sum(V/T); a
    history given in burnup reaches exactly its burnups."""
    rod, out = _rod(models=fuel.RodModels(mechanics=False, fission_gas_release="none"))
    f, g = rod.fuel, rod.geometry
    energy = 1e3 * 0 + 0.5 * (1e3 + 25e3) * 3600 + 25e3 * (100 * DAY - 3600)
    fima = energy / (f.energy_per_fission * f.heavy_metal_atom_density() * g.fuel_cross_section)
    assert out.rod_average_burnup[-1] == pytest.approx(
        rod.converter.from_fima(fima, "MWd/kgHM"), rel=1e-12
    )
    assert out.gas_amount == pytest.approx(rod.fill_amount, rel=1e-12)
    assert out.helium_fraction == pytest.approx(1.0)
    # Burnup history: 20 and 40 MWd/kgU at 20 and then 15 kW/m.
    history = fuel.PowerHistory(
        linear_heat_rate=[20e3, 20e3, 15e3], burnup=[0.0, 20.0, 40.0], burnup_unit="MWd/kgU"
    )
    _, out = _rod(
        history=history,
        models=fuel.RodModels(mechanics=False, fission_gas_release="none"),
        max_time_step=None,
        burnup_unit="GWd/tHM",
    )
    assert out.rod_average_burnup == pytest.approx([0.0, 20.0, 40.0], rel=1e-12)
    assert out.burnup_in("FIMA", rod_average=True)[-1] == pytest.approx(
        40.0 / rod.converter.mwd_per_kg_per_fima, rel=1e-12
    )


def test_an_annular_rod_runs_hotter_in_burnup_and_colder_at_the_centre():
    """The bore holds no fuel, so the same linear heat rate burns the annulus
    faster (by the area ratio) and the bore sits below the centreline
    temperature of a solid pellet."""
    solid = fuel.RodGeometry.from_diameters(8.19e-3, 8.36e-3, 9.50e-3, 0.05)
    annular = fuel.RodGeometry.from_diameters(8.19e-3, 8.36e-3, 9.50e-3, 0.05, 1.8e-3)
    models = fuel.RodModels(fission_gas_release="none")
    _, a = _rod(geometry=solid, models=models)
    _, b = _rod(geometry=annular, models=models)
    ratio = solid.fuel_cross_section / annular.fuel_cross_section
    assert b.rod_average_burnup[-1] == pytest.approx(ratio * a.rod_average_burnup[-1], rel=1e-12)
    assert b.fuel_centerline_temperature[1].max() < a.fuel_centerline_temperature[1].max() - 50


def test_a_prescribed_cladding_temperature_is_held():
    """The Halden-style boundary of the IAEA benchmarks: T = 240 + 0.4162
    q'^0.75 degC with q' in kW/m."""

    def wall(linear_heat_rate, time):
        return 273.15 + 240.0 + 0.4162 * (linear_heat_rate / 1e3) ** 0.75

    coolant = fuel.PrescribedCladdingTemperature(temperature=wall, pressure=3.4e6)
    _, out = _rod(coolant=coolant, models=fuel.RodModels(fission_gas_release="none"))
    expected = wall(out.linear_heat_rate, 0.0)
    assert out.clad_outer_temperature == pytest.approx(expected, abs=1e-6)


def test_gaseous_swelling_opens_no_gap_and_closes_it_faster():
    """Gaseous swelling adds fuel volume, so the gap at the end is narrower
    with it than without it."""
    history = fuel.PowerHistory(linear_heat_rate=[1e3, 30e3, 30e3], time=[0, 3600, 300 * DAY])
    gaps = {}
    for swelling in (False, True):
        _, out = _rod(
            history=history,
            fuel_material=fuel.UO2Fuel(gaseous_swelling_model="matpro"),
            models=fuel.RodModels(gaseous_swelling=swelling, fission_gas_release="none"),
            max_time_step=30 * DAY,
        )
        gaps[swelling] = out.gap_width[-1].min()
    assert gaps[True] < gaps[False]


def test_the_output_is_announced_and_written(tmp_path, capsys):
    """Every input is printed with its unit and whether it was given; the
    CSV files read back with numpy."""
    rod, out = _rod(directory=str(tmp_path), file_base="case")
    report = out.input_report
    assert "fuel_stack_height" in report and "(default)" in report and "given" in report
    history = np.genfromtxt(tmp_path / "case_history.csv", delimiter=",", names=True, comments="#")
    assert history["gas_pressure"] == pytest.approx(out.gas_pressure)
    second = (tmp_path / "case_history.csv").read_text().splitlines()[1]
    assert second.startswith("# units: s, d, W/m, MWd/kgHM, Pa")
    axial = np.genfromtxt(tmp_path / "case_axial.csv", delimiter=",", names=True, comments="#")
    assert len(axial) == len(out.time) * len(out.axial_positions)
    assert "Summary" in out.summary()
    assert (tmp_path / "case_input.txt").exists()
    with pytest.raises(AttributeError, match="gas_pressure"):
        out.gas_presure  # noqa: B018 - the misspelling is the point


def test_specifications_refuse_nonsense_with_the_reason():
    with pytest.raises(ValueError, match="pellet_outer_radius < clad_inner_radius"):
        fuel.RodGeometry(
            pellet_outer_radius=5e-3,
            clad_inner_radius=4e-3,
            clad_outer_radius=6e-3,
            fuel_stack_height=0.1,
        )
    with pytest.raises(ValueError, match="sum to 1.2"):
        fuel.FillGas(pressure=1e6, plenum_volume=0.0, composition={"helium": 1.0, "argon": 0.2})
    with pytest.raises(ValueError, match="either 'time' or 'burnup'"):
        fuel.PowerHistory(linear_heat_rate=[1.0, 2.0])
    with pytest.raises(ValueError, match="Unknown burnup unit"):
        fuel.PowerHistory(linear_heat_rate=[1.0, 2.0], burnup=[0, 1], burnup_unit="GWd/t")
    with pytest.raises(TypeError, match="'coolant' must be a ForcedConvection"):
        fuel.FuelRod(
            fuel.RodGeometry.from_diameters(8e-3, 8.2e-3, 9e-3, 0.1),
            fuel.UO2Fuel(),
            fuel.ZircaloyCladding(),
            fuel.FillGas(pressure=1e6, plenum_volume=0.0),
            600.0,
            fuel.PowerHistory(linear_heat_rate=[1.0, 2.0], time=[0, 1]),
        )


def test_describe_marks_the_defaults():
    rows = {row[0]: row for row in fuel.describe(fuel.UO2Fuel(enrichment=0.05))}
    assert rows["enrichment"][3] == "given"
    assert rows["theoretical_density_fraction"][3] == "(default)"
    assert rows["grain_radius"][2] == "m"


# ---------------------------------------------------------------------------
# Other uses of the fuel objects
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("method", ["fem", "dmcdm", "hfvm"])
def test_conjugate_fuel_rod_and_coolant_flow(method):
    """examples/fuel_rod_conjugate.py: a UN rod, its gap, the cladding and a
    sodium coolant in the pressure-velocity formulation, solved together (with
    PETSc when available).  The heat carried out of the outlet plus the heat
    conducted back through the inlet equals the fission power within 1 %, and
    the three methods agree."""
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "examples" / "fuel_rod_conjugate.py"
    spec = importlib.util.spec_from_file_location("fuel_rod_conjugate", path)
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    problem, info = example.build(method)
    options = {"linear_solver": "petsc"} if dm.have_petsc() else {}
    problem.solve(load_factors=[0.1, 0.3, 0.6, 1.0], **options)
    heat, mixed = example.energy_balance(problem, info)
    inlet = -problem.total_reaction("temperature", "coolant_inlet")
    assert heat + inlet == pytest.approx(info["power"], rel=0.01)
    centre = problem.sample("temperature", [[0.0, 0.5 * info["top"], 0.0]])[0]
    assert centre == pytest.approx(826.8, abs=1.0)


def test_interface_conditions_are_refused_by_the_distributed_solver():
    mesh = fuel.axisymmetric_rod_mesh(
        _geometry(fuel_stack_height=0.01),
        fuel.RodMesh(num_fuel_radial_elements=4, num_clad_radial_elements=2, num_axial_elements=2),
    )
    distributed = dm.DistributedProblem(mesh, method="fem", coordinates="axisymmetric")
    p = distributed.local
    p.add_variable("T")
    p.add_kernel("heat_conduction", "k", variable="T")
    p.add_boundary_condition(
        "gap_heat_transfer",
        "gap",
        variable="T",
        boundary=["fuel_outer"],
        secondary_boundary=["clad_inner"],
        gap_conductance=1e4,
    )
    with pytest.raises(Exception, match="does not support"):
        distributed.solve()


def test_fuel_materials_start_from_a_cold_guess():
    """With the temperature left at its default initial value of zero, the
    fuel correlations (singular at absolute zero) must still give Newton's
    method a finite first Jacobian: they are evaluated at max(T, 200 K)."""
    mesh = fuel.axisymmetric_rod_mesh(
        fuel.RodGeometry.from_diameters(8.19e-3, 8.36e-3, 9.5e-3, 0.1)
    )
    p = dm.Problem(mesh, method="fem", coordinates="axisymmetric")
    p.add_variable("T")
    p.add_material(
        "UO2_thermal", "fuel", block=["fuel"], temperature="T", burnup=0.02, burnup_unit="FIMA"
    )
    p.add_material("Zircaloy_thermal", "clad", block=["clad"], temperature="T")
    p.add_kernel(
        "heat_conduction", "k", variable="T", thermal_conductivity_property="thermal_conductivity"
    )
    p.add_kernel("heat_source", "q", variable="T", heat_source=3e8, block=["fuel"])
    p.add_boundary_condition(
        "gas_gap_heat_transfer",
        "gap",
        variable="T",
        boundary=["fuel_outer"],
        secondary_boundary=["clad_inner"],
        gas_pressure=2.0e6,
    )
    p.add_boundary_condition(
        "Dirichlet_boundary_condition", "cool", variable="T", boundary=["clad_outer"], value=600.0
    )
    p.solve()
    T = np.asarray(p.solution())
    assert np.all(np.isfinite(T))
    assert T.min() >= 600.0 - 1e-9
    assert T.max() < 3000.0


# ---------------------------------------------------------------------------
# Materials defined by expressions
# ---------------------------------------------------------------------------
# The correlations of UNFuel (Hayes et al. 1990, AbdulHameed et al. 2025,
# Konovalov et al. 2016), written as expressions.
UN_AS_EXPRESSIONS = dict(
    thermal_conductivity="1.864*exp(-2.14*porosity)*temperature^0.361",
    specific_heat=(
        "(51.14*(365.7/temperature)^2*exp(365.7/temperature)/(exp(365.7/temperature) - 1)^2"
        " + 9.491e-3*temperature + 2.642e11/temperature^2*exp(-18081/temperature))/0.25204"
    ),
    youngs_modulus=(
        "1e6*0.258*(100*theoretical_density_fraction)^3.002*(1 - 2.375e-5*temperature)"
    ),
    poissons_ratio="1.26e-3*(100*theoretical_density_fraction)^1.174",
    thermal_strain="(7.096e-6 + 1.409e-9*temperature)*(temperature - 298)",
    creep_rate=(
        "2.054e-3*(von_mises_stress/1e6)^4.5*exp(-39369.5/temperature)"
        " + 582610.427*(von_mises_stress/1e6)/(temperature*(2e6*grain_radius)^3)"
        "*exp(-2.28/(8.617333262e-5*temperature))"
        " + 2.9e-22*(von_mises_stress/1e6)*(fission_rate*1e-6)*exp(20*porosity)/3600"
    ),
)


def _custom_un(**changes):
    un = fuel.UNFuel()
    data = dict(
        name="UN written out",
        theoretical_density=14330.0,
        compound_molar_mass=un.compound_molar_mass,
        heavy_metal_molar_mass=un.heavy_metal_molar_mass,
        **UN_AS_EXPRESSIONS,
    )
    data.update(changes)
    return fuel.CustomFuel(**data)


@pytest.mark.parametrize("model", ["axisymmetric", "1.5d"])
def test_a_custom_fuel_reproduces_the_built_in_fuel_it_writes_out(model):
    """UN's correlations written as expressions give the rod of UNFuel: the
    same temperatures, gaps and creep to rounding, through thermal expansion,
    contact, creep and the gas pressure.  (Swelling and gas release, which
    UNFuel takes from tabulated correlations, are switched off in both.)"""
    history = fuel.PowerHistory(linear_heat_rate=[1e3, 40e3, 40e3], time=[0, 3600, 100 * DAY])
    models = fuel.RodModels(solid_swelling=False, fission_gas_release="none")
    results = []
    for material in (fuel.UNFuel(), _custom_un()):
        _, out = _rod(
            model, fuel_material=material, history=history, models=models, max_time_step=20 * DAY
        )
        results.append(out)
    built_in, custom = results
    for name in ("fuel_centerline_temperature", "gap_width", "clad_hoop_strain", "gas_pressure"):
        assert getattr(custom, name) == pytest.approx(getattr(built_in, name), rel=1e-7, abs=1e-12)
    assert custom.burnup == pytest.approx(built_in.burnup, rel=1e-12)


def test_a_custom_fuel_has_its_own_gas_release_and_swelling():
    """A fuel with a release-fraction correlation and a linear swelling law:
    the gas released follows the correlation, and swelling narrows the gap."""
    history = fuel.PowerHistory(linear_heat_rate=[1e3, 40e3, 40e3], time=[0, 3600, 200 * DAY])
    gaps = {}
    for swelling in (None, "0.02*burnup/0.01"):
        material = _custom_un(
            volumetric_swelling=swelling,
            fission_gas_release="fraction",
            fission_gas_release_fraction="min(0.05 + 0*temperature + 0*burnup, 1)",
        )
        _, out = _rod(fuel_material=material, history=history, max_time_step=20 * DAY)
        gaps[swelling] = out.gap_width[-1].min()
        # Five per cent of the gas produced is released from the first step on.
        assert out.fission_gas_release[-1] == pytest.approx(0.05, rel=1e-9)
    assert gaps["0.02*burnup/0.01"] < gaps[None]


def test_custom_materials_refuse_mistakes_with_the_property_named():
    with pytest.raises(ValueError, match="thermal_conductivity: unknown name 'T'"):
        _custom_un(thermal_conductivity="2 + 0.001*T")
    with pytest.raises(ValueError, match="needs fission_gas_release_fraction"):
        _custom_un(fission_gas_release="fraction")
    with pytest.raises(ValueError, match="reserved"):
        _custom_un(constants={"temperature": 1.0})
    with pytest.raises(ValueError, match="relocation is not modelled for UN written out"):
        _rod(fuel_material=_custom_un(), models=fuel.RodModels(relocation=True))


def test_a_custom_cladding_reproduces_zircaloy_heat_conduction():
    """Zircaloy's conductivity and specific heat written as expressions give
    the temperatures of ZircaloyCladding (thermal only: the built-in thermal
    expansion of Zircaloy is anisotropic, which the isotropic custom strain
    does not reproduce)."""
    zircaloy_like = fuel.CustomCladding(
        name="Zircaloy written out",
        thermal_conductivity=(
            "12.767 - 5.4348e-4*min(temperature, 1800) + 8.9818e-6*min(temperature, 1800)^2"
        ),
        specific_heat="255.66 + 0.1024*temperature",
        density=6550.0,
        youngs_modulus=9.9e10,
        poissons_ratio=0.37,
        thermal_strain="6.7e-6*temperature",
        meyer_hardness=6.8e8,
        irradiation_growth="1e-28*fast_neutron_fluence",
        creep_rate="1e-18*von_mises_stress*fast_neutron_flux/1e18",
    )
    models = fuel.RodModels(mechanics=False, fission_gas_release="none")
    history = fuel.PowerHistory(linear_heat_rate=[20e3, 20e3], time=[0, 3600])
    temperatures = []
    for cladding in (fuel.ZircaloyCladding(), zircaloy_like):
        geometry = fuel.RodGeometry.from_diameters(8.19e-3, 8.36e-3, 9.50e-3, 0.05)
        rod = fuel.FuelRod(
            geometry,
            fuel.UO2Fuel(),
            cladding,
            fuel.FillGas(pressure=2.0e6, plenum_volume=0.15e-6),
            fuel.ForcedConvection(
                inlet_temperature=565.0, pressure=15.5e6, mass_flux=3800.0, rod_pitch=12.6e-3
            ),
            history,
            models=models,
            numerics=fuel.RodNumerics(mesh=fuel.RodMesh(num_axial_elements=2)),
            output=fuel.RodOutput(print_input=False, print_steps=False),
        )
        temperatures.append(rod.run().clad_inner_temperature)
    # The steady state is independent of the specific heat and the density.
    assert temperatures[1][0] == pytest.approx(temperatures[0][0], rel=1e-9)
    # The full custom cladding also runs with its mechanics.
    _, out = _rod(fuel_material=fuel.UO2Fuel(), models=fuel.RodModels(fission_gas_release="none"))
    geometry = fuel.RodGeometry.from_diameters(8.19e-3, 8.36e-3, 9.50e-3, 0.05)
    rod = fuel.FuelRod(
        geometry,
        fuel.UO2Fuel(),
        zircaloy_like,
        fuel.FillGas(pressure=2.0e6, plenum_volume=0.15e-6),
        fuel.ForcedConvection(
            inlet_temperature=565.0, pressure=15.5e6, mass_flux=3800.0, rod_pitch=12.6e-3
        ),
        fuel.PowerHistory(linear_heat_rate=[1e3, 25e3, 25e3], time=[0, 3600, 100 * DAY]),
        models=fuel.RodModels(fission_gas_release="none"),
        numerics=fuel.RodNumerics(mesh=fuel.RodMesh(num_axial_elements=2), max_time_step=20 * DAY),
        output=fuel.RodOutput(print_input=False, print_steps=False),
    )
    custom = rod.run()
    # Both claddings creep down under the coolant pressure.
    assert custom.clad_hoop_strain[-1, 0] < custom.clad_hoop_strain[1, 0]
    assert out.clad_hoop_strain[-1, 0] < out.clad_hoop_strain[1, 0]


def test_the_custom_fuel_example_runs_and_agrees():
    """examples/custom_fuel.py: the built-in UN and UN written out as
    expressions give the same rod; the subclass with its own creep law runs."""
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "examples" / "custom_fuel.py"
    import sys

    spec = importlib.util.spec_from_file_location("custom_fuel", path)
    example = importlib.util.module_from_spec(spec)
    sys.modules["custom_fuel"] = example  # the example's dataclass needs its module
    spec.loader.exec_module(example)
    a = example.run(fuel.UNFuel())
    b = example.run(example.un_written_out())
    assert b.fuel_centerline_temperature == pytest.approx(a.fuel_centerline_temperature, rel=1e-9)
    c = example.run(example.UNWithPowerLawCreep())
    assert np.all(np.isfinite(c.clad_hoop_strain))


@pytest.mark.parametrize("model", ["axisymmetric", "1.5d"])
def test_finite_strain_agrees_with_small_strain_in_normal_operation(model):
    """In normal operation the strains stay below 1 %, so the finite-strain
    rod (deformation gradient, follower pressures, rotated small-strain
    creep) gives the small-strain rod within a small fraction of each
    quantity's change."""
    results = {}
    for strain in ("small", "finite"):
        geometry = fuel.RodGeometry.from_diameters(8.19e-3, 8.36e-3, 9.50e-3, 0.05)
        rod = fuel.FuelRod(
            geometry,
            fuel.UO2Fuel(),
            fuel.ZircaloyCladding(),
            fuel.FillGas(pressure=2.0e6, plenum_volume=0.15e-6),
            fuel.ForcedConvection(
                inlet_temperature=565.0, pressure=15.5e6, mass_flux=3800.0, rod_pitch=12.6e-3
            ),
            fuel.PowerHistory(linear_heat_rate=[1e3, 25e3, 25e3], time=[0, 3600, 100 * DAY]),
            models=fuel.RodModels(fission_gas_release="none"),
            numerics=fuel.RodNumerics(
                model=model,
                strain=strain,
                mesh=fuel.RodMesh(num_axial_elements=2, num_axial_slices=1),
                max_time_step=20 * DAY,
            ),
            output=fuel.RodOutput(print_input=False, print_steps=False),
        )
        results[strain] = rod.run()
    small, finite = results["small"], results["finite"]
    assert finite.fuel_centerline_temperature == pytest.approx(
        small.fuel_centerline_temperature, rel=1e-3
    )
    # Gaps of tens of micrometres agree within 0.5 micrometre, hoop strains
    # within 2 % of their range.
    assert np.abs(finite.gap_width - small.gap_width).max() < 0.5e-6
    span = np.ptp(small.clad_hoop_strain)
    assert np.abs(finite.clad_hoop_strain - small.clad_hoop_strain).max() < 0.02 * span


# ---------------------------------------------------------------------------
# Coupled gaseous swelling, stepping on the power history, relocation cracks
# ---------------------------------------------------------------------------
def test_gaseous_swelling_of_the_gas_model_is_the_bubble_volume():
    """Pastore et al. (2013) Eqs. 9 and 10: the intragranular bubbles hold
    N (4/3) pi R^3 and the grain-face bubbles 3/(2a) N_gf V_gf per unit
    volume of fuel."""
    from dualmesh.fuel.gas import BoothFissionGasRelease

    model = BoothFissionGasRelease(3, grain_radius=10e-6)
    for k in range(20):
        burnup = np.full(3, 1e-3 * (k + 1))  # FIMA, for the micro-cracking
        model.advance(5 * DAY, np.array([1200.0, 1500.0, 1700.0]), np.full(3, 1.2e19), 5e6, burnup)
    faces = model.faces
    expected = (
        4.0 / 3.0 * np.pi * model.intragranular_bubble_radius** 3 * model.bubble_density
        + faces.density * faces.volume() * 3.0 / (2.0 * 10e-6)
    )
    assert model.gaseous_swelling() == pytest.approx(expected, rel=1e-12)
    assert np.all(model.gaseous_swelling() > 0)
    # Hotter fuel holds more gas in bigger face bubbles.
    assert model.gaseous_swelling()[1] > model.gaseous_swelling()[0]


def test_the_rod_swells_with_the_bubbles_of_its_gas_model():
    """With gaseous_swelling_model = fission_gas the swelling field of the
    fuel elements is the bubble volume of the gas model after the run."""
    history = fuel.PowerHistory(linear_heat_rate=[1e3, 35e3, 35e3], time=[0, 3600, 200 * DAY])
    rod, _ = _rod(history=history, max_time_step=20 * DAY)
    p, ids, _, _, _, model = rod._fuel_regions[0]
    field = np.asarray(p.element_field("gaseous_swelling"))[ids]
    assert field == pytest.approx(model.gaseous_swelling(), rel=1e-12)
    assert field.max() > 0


def test_simplified_points_keep_the_corners_of_a_history():
    from dualmesh.fuel.rod import simplified_points

    t = np.array([0.0, 1.0, 2.0, 3.0, 3.1, 3.2, 3.3, 4.0, 5.0, 6.0])
    q = np.array([0.0, 20.0, 20.1, 20.0, 0.0, 0.0, 20.0, 19.9, 20.0, 20.0])
    keep = simplified_points(t, q, 0.5)
    # The ramp, the shutdown and the ends are kept; the flat points are not.
    assert {0, 1, 3, 4, 5, 6, 9} <= set(keep)
    assert 2 not in keep and 8 not in keep


def test_the_steps_land_on_a_short_shutdown():
    """A shutdown shorter than the interval between output times is stepped
    to (RodNumerics.power_history_tolerance), and skipped with None."""
    hour = 3600.0
    stop = 10 * DAY
    t = [0, hour, stop, stop + 0.25 * hour, stop + 0.5 * hour, stop + hour, 30 * DAY]
    q = [1e3, 30e3, 30e3, 1.0, 1.0, 30e3, 30e3]
    history = fuel.PowerHistory(linear_heat_rate=q, time=t)
    out = np.array([0.0, 15 * DAY, 30 * DAY])
    rod, result = _rod(history=history, output_times=out)
    bounds = rod._step_boundaries(out)
    assert 10 * DAY + 0.25 * hour in bounds and 10 * DAY + 0.5 * hour in bounds
    assert len(result.time) == len(out)
    rod.numerics.power_history_tolerance = None
    assert list(rod._step_boundaries(out)) == list(out)


def test_relocation_opens_crack_volume():
    """The relocation strain of the rod's gas volume is the ESCORE form of
    the eigenstrain (checked by hand at 43 kW/m, 6 MWd/kgHM, D0 = 9.13 mm,
    G0 = 170 um), and its cracks lower the gas pressure."""
    from dualmesh import _core

    q_ft = 43e3 * 0.3048 / 1e3
    expected = 0.80 * np.cbrt(q_ft - 6.0) * (170e-6 / 9.13e-3)
    expected *= 0.005 * 6000.0**0.3 - 0.20 * 9.13e-3 / 0.0254 + 0.3
    assert _core.fuel.uo2_relocation_strain(43e3, 6.0, 9.13e-3, 170e-6) == pytest.approx(expected)
    assert _core.fuel.uo2_relocation_strain(5e3, 6.0, 9.13e-3, 170e-6) == 0.0
    history = fuel.PowerHistory(linear_heat_rate=[1e3, 35e3, 35e3], time=[0, 3600, 20 * DAY])
    models = fuel.RodModels(fission_gas_release="none")
    _, with_cracks = _rod(history=history, models=models)
    rod, _ = _rod(history=history, models=models)
    rod._relocation_strain_along_the_rod = lambda: np.zeros(len(rod.axial_positions))
    rod._update_gas_pressure(rod._state_along_the_rod())
    assert with_cracks.gas_pressure[-1] < rod._gas_pressure.get()


def test_doped_fuel_densifies_as_measured_at_halden():
    assert fuel.DopedUO2Fuel().total_densification == 0.001
    assert fuel.UO2Fuel().total_densification == 0.01
