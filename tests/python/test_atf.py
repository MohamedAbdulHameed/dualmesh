# SPDX-License-Identifier: LGPL-2.1-or-later
"""Accident tolerant fuel materials (dualmesh.fuel.atf).

* The correlations reproduce the values of the sources at points read or
  computed independently (the equations evaluated by hand, or values read
  from the sources' figures).
* Every ATF material runs in a rod, and the rods show the behaviour that
  follows from the properties: U3Si2 runs far colder than UO2 at the same
  power; FeCrAl creeps down, by irradiation creep, a tenth as much as
  Zircaloy; the larger grains of
  doped UO2 release less gas; SiC swells outward and runs hot; a thin
  chromium coating changes the temperatures by little until the cladding
  creeps, and then slows the creep-down.
"""

from __future__ import annotations

import dualmesh as dm
import numpy as np
import pytest
from dualmesh import fuel

DAY = 86400.0


def _evaluate(material, name, **values):
    """A property of an expression-defined material, evaluated in Python."""
    delegate = material._delegate()
    text = getattr(delegate, name)
    constants = delegate._constants()
    names = ["temperature", "von_mises_stress", *delegate._fields, "maximum_temperature", *constants]
    expression = dm.Expression(str(text), names)
    arguments = dict.fromkeys(names, 0.0)
    arguments.update(constants)
    arguments.update(values)
    return expression(**arguments)


def test_fecral_correlations_are_those_of_the_handbook():
    """Kanthal APMT (Field et al. 2018, Tables 1 to 3, Eq. 3.9; the Kanthal
    datasheet for the elastic constants; Terrani et al. 2016 for irradiation
    creep)."""
    apmt = fuel.FeCrAlCladding()
    # Datasheet defaults: 250 HV and emissivity 0.70.
    assert apmt.meyer_hardness == pytest.approx(2.4517e9, rel=1e-4)
    assert apmt.emissivity == 0.70
    # Conductivity: -7.223e-7 T^2 + 1.563e-2 T + 6.569.
    assert _evaluate(apmt, "thermal_conductivity", temperature=300.0) == pytest.approx(-7.223e-7 * 300**2 + 1.563e-2 * 300 + 6.569)
    # Specific heat below and above the Curie temperature (852 K).
    assert _evaluate(apmt, "specific_heat", temperature=500.0) == pytest.approx(2.540 * 500 - 4.311e-3 * 500**2 + 2.982e-6 * 500**3)
    above = 1.840 * 1000 - 1.843e-3 * 1000**2 + 0.643e-6 * 1000**3 - 5.712e3 / 1000 - 50.38 * np.log(148 / 852)
    assert _evaluate(apmt, "specific_heat", temperature=1000.0) == pytest.approx(above)
    # Young's modulus of APMT: the datasheet's 220 GPa at 20 C and 170 GPa at
    # 600 C within 2.6 GPa; the handbook's fit for the ORNL alloys.
    assert _evaluate(apmt, "youngs_modulus", temperature=293.15) == pytest.approx(220e9, abs=2.6e9)
    assert _evaluate(apmt, "youngs_modulus", temperature=873.15) == pytest.approx(170e9, abs=2.6e9)
    assert _evaluate(apmt, "poissons_ratio", temperature=600.0) == 0.30
    c35m = fuel.FeCrAlCladding(alloy="C35M", meyer_hardness=2e9)
    assert _evaluate(c35m, "youngs_modulus", temperature=673.15) == pytest.approx(1e9 * (-5.46e-5 * 400**2 - 3.85e-2 * 400 + 199))
    with pytest.raises(ValueError, match="give meyer_hardness"):
        fuel.FeCrAlCladding(alloy="C35M")
    # The thermal strain is the mean coefficient times (T - 293.15 K): at
    # 1273.15 K 1.436 %, against 1.441 % from the datasheet's mean CTE.
    alpha = 1.771e-10 * 1273.15**3 + 9.558e-7 * 1273.15**2 + 1.937e-3 * 1273.15 + 10.27
    strain = _evaluate(apmt, "thermal_strain", temperature=1273.15)
    assert strain == pytest.approx(1e-6 * alpha * 980.0, rel=1e-9)
    assert strain == pytest.approx(0.01441, rel=0.01)
    # Irradiation creep: 5e-6 /MPa/dpa x 100 MPa x 1e18 n/m^2/s x 0.9e-25 dpa
    # per n/m^2 = 4.5e-11 1/s (IAEA-TECDOC-1921 Eq. 5 gives 4.5e-31 sigma phi).
    irradiation = _evaluate(apmt, "creep_rate", temperature=600.0, von_mises_stress=100e6, fast_neutron_flux=1e18) - _evaluate(apmt, "creep_rate", temperature=600.0, von_mises_stress=100e6)
    assert irradiation == pytest.approx(4.5e-11, rel=1e-9)
    # Generalized creep: 0.83 sigma^7.1 exp(-326 kJ / RT), sigma in MPa.
    assert _evaluate(apmt, "creep_rate", temperature=1073.15, von_mises_stress=50e6) == pytest.approx(0.83 * 50**7.1 * np.exp(-326e3 / (8.314462618 * 1073.15)))
    with pytest.raises(ValueError, match="unknown alloy"):
        fuel.FeCrAlCladding(meyer_hardness=2e9, alloy="Kanthal")


def test_sic_correlations_are_those_of_the_handbook():
    """Koyanagi et al. (2018): swelling Eqs. 6 to 8, expansion Eq. 1."""
    sic = fuel.SiCCladding(dpa_per_fast_neutron_fluence=1e-25, meyer_hardness=5e9)
    T = 573.15
    saturation = 5.8366e-2 - 1.0089e-4 * T + 6.9368e-8 * T**2 - 1.8152e-11 * T**3
    critical = -0.57533 + 3.3342e-3 * T - 5.3970e-6 * T**2 + 2.9754e-9 * T**3
    dose = 0.3
    expected = saturation * (1 - np.exp(-dose / critical)) ** (2 / 3)
    assert _evaluate(sic, "volumetric_swelling", temperature=T, fast_neutron_fluence=dose * 1e25) == pytest.approx(expected)
    # About 2 % at 300 C and 1 dpa, as in the handbook's Fig. 15.
    assert _evaluate(sic, "volumetric_swelling", temperature=T, fast_neutron_fluence=1e25) == pytest.approx(0.0199, abs=5e-4)
    # The swelling is evaluated at the highest temperature reached, so it does
    # not grow when the cladding cools.
    hot = _evaluate(sic, "volumetric_swelling", temperature=T, fast_neutron_fluence=1e25)
    cooled = _evaluate(sic, "volumetric_swelling", temperature=300.0, maximum_temperature=T, fast_neutron_fluence=1e25)
    assert cooled == pytest.approx(hot, rel=1e-12)
    # Irradiation lowers the conductivity through the defect resistance.
    fresh = _evaluate(sic, "thermal_conductivity", temperature=T)
    irradiated = _evaluate(sic, "thermal_conductivity", temperature=T, fast_neutron_fluence=2e25)
    assert irradiated < 0.5 * fresh
    # The dpa conversion has no default.
    with pytest.raises(ValueError, match="missing required parameter 'dpa_per_fast_neutron_fluence'"):
        fuel.SiCCladding(meyer_hardness=5e9)


def test_chromium_correlations_follow_holzwarth_and_stamm_and_wagih_et_al():
    """Holzwarth and Stamm (2002), Eqs. (A.4) to (A.6), Wagih et al. (2018),
    Eq. (3), and creep fitted to Stephens and Klopp (1972), with the law of
    Wagih et al. (2018) as an option."""
    chromium = fuel.ChromiumCoating()
    Tc = 500.0
    assert _evaluate(chromium, "thermal_conductivity", temperature=Tc + 273.15) == pytest.approx(87.56671 - 0.04179 * Tc + 3.15147e-5 * Tc**2 - 2.06676e-8 * Tc**3)
    assert _evaluate(chromium, "specific_heat", temperature=Tc + 273.15) == pytest.approx(1000 * (0.48047 + 6.34753e-5 * Tc + 2.34120e-7 * Tc**2 - 1.27824e-10 * Tc**3))
    T = 773.15
    assert _evaluate(chromium, "youngs_modulus", temperature=T) == pytest.approx(1e9 * (264.11 - 0.01 * T - 2.5e-5 * T**2))
    # Holzwarth and Stamm (2002), Eq. (A.4), is the mean coefficient
    # Delta L/(L_0 Delta T) from 20 C (their Sect. 3.2.3).
    for Tc in (20.0, 500.0, 1000.0):
        mean = 1e-6 * (8.3159 + 1.80901e-3 * Tc + 6.45421e-7 * Tc**2 + 1.27483e-10 * Tc**3)
        assert _evaluate(chromium, "thermal_strain", temperature=Tc + 273.15) == pytest.approx(mean * (Tc - 20.0), abs=1e-15)
    wagih = fuel.ChromiumCoating(thermal_creep="wagih")
    assert _evaluate(wagih, "creep_rate", temperature=1100.0, von_mises_stress=60e6) == pytest.approx(5.1596e-3 * 60**6.2 * np.exp(-306268.8 / (8.314462618 * 1100.0)))
    # A data point of Stephens and Klopp, Table I: 1149 C, 18.6 MPa, fine
    # grains, 1.4e-5 1/s.  The fit is within the scatter (factor 2); the
    # 816 C law of Wagih et al. is 6.5 times too slow there.
    T = 1149 + 273.15
    fit = _evaluate(chromium, "creep_rate", temperature=T, von_mises_stress=18.6e6)
    assert 0.5 < fit / 1.4e-5 < 2.0
    assert _evaluate(wagih, "creep_rate", temperature=T, von_mises_stress=18.6e6) < 0.2 * 1.4e-5


def test_u3si2_and_doped_uo2_data():
    u3si2 = fuel.U3Si2Fuel(enrichment=0.0072)
    # Three uranium atoms per formula unit: 17 % more heavy metal per volume
    # than UO2, the ratio of the uranium densities 11.3 and 9.66 g/cm^3.
    uo2 = fuel.UO2Fuel(enrichment=0.0072)
    ratio = u3si2.heavy_metal_atom_density() / uo2.heavy_metal_atom_density()
    assert ratio == pytest.approx(11.3 / (10.963 * 0.8815), rel=0.01)
    # The handbook values of CASL-U-2019-1870: k = 4.996 + 0.0118 T (8.536 at
    # 300 K), E = 142.68 - 6.425 p GPa and nu = E/2G - 1 at 5 % porosity.
    assert _evaluate(u3si2, "thermal_conductivity", temperature=300.0) == pytest.approx(8.536)
    assert _evaluate(u3si2, "youngs_modulus") == pytest.approx(1.10555e11)
    assert _evaluate(u3si2, "poissons_ratio") == pytest.approx(0.1820, abs=1e-4)
    assert _evaluate(u3si2, "volumetric_swelling", burnup=0.05) == pytest.approx(0.34392 * 0.05)
    # Doped UO2: the undoped terms at T1 = 1773 K, where the doping factors are one.
    doped = fuel.DopedUO2Fuel()
    kT = 1.380649e-23 * 1773.0
    undoped = 7.6e-10 * np.exp(-4.86e-19 / kT) + 5.64e-25 * np.sqrt(1e19) * np.exp(-1.91e-19 / kT) + 8e-40 * 1e19
    assert doped.diffusion_coefficient(1773.0, 1e19) == pytest.approx(undoped, rel=1e-12)
    # Above 1773 K the factors stay at one.
    kT = 1.380649e-23 * 2000.0
    undoped = 7.6e-10 * np.exp(-4.86e-19 / kT) + 5.64e-25 * np.sqrt(1e19) * np.exp(-1.91e-19 / kT) + 8e-40 * 1e19
    assert doped.diffusion_coefficient(2000.0, 1e19) == pytest.approx(undoped, rel=1e-12)
    assert fuel.DopedUO2Fuel(diffusivity_case="upper_limit").diffusion_coefficient(1200.0, 1e19) > doped.diffusion_coefficient(1200.0, 1e19)


# ---------------------------------------------------------------------------
# Rods
# ---------------------------------------------------------------------------
def _run(fuel_material=None, cladding=None, geometry=None, history=None, models=None, step=20 * DAY):
    geometry = geometry or fuel.RodGeometry.from_diameters(8.19e-3, 8.36e-3, 9.50e-3, 0.05)
    rod = fuel.FuelRod(
        geometry,
        fuel_material or fuel.UO2Fuel(),
        cladding or fuel.ZircaloyCladding(),
        fuel.FillGas(pressure=2.0e6, plenum_volume=0.15e-6),
        fuel.ForcedConvection(inlet_temperature=565.0, pressure=15.5e6, mass_flux=3800.0, rod_pitch=12.6e-3),
        history or fuel.PowerHistory(linear_heat_rate=[1e3, 30e3, 30e3], time=[0, 3600, 200 * DAY]),
        models=models,
        numerics=fuel.RodNumerics(mesh=fuel.RodMesh(num_axial_elements=2), max_time_step=step),
        output=fuel.RodOutput(print_input=False, print_steps=False),
    )
    return rod.run()


def test_u3si2_runs_far_colder_than_uo2():
    """Its conductivity is several times that of UO2 (15 to 30 against 2 to 5
    W/(m K)), so at 30 kW/m the rise from surface to centre is several times
    smaller."""
    uo2 = _run(fuel.UO2Fuel())
    u3si2 = _run(fuel.U3Si2Fuel())
    rise = lambda out: out.fuel_centerline_temperature[1, 0] - out.fuel_surface_temperature[1, 0]  # noqa: E731
    assert rise(u3si2) < 0.25 * rise(uo2)
    assert np.all(np.isfinite(u3si2.gas_pressure))


def test_fecral_barely_creeps_down():
    """At cladding temperatures near 600 K Zircaloy creeps down under the
    coolant pressure.  FeCrAl's thermal creep is negligible there, so it
    creeps down only by irradiation creep, whose compliance (5e-6 per MPa per
    dpa at 0.9 dpa per 1e25 n/m^2) is a few times smaller than Zircaloy's
    and acts on a tube twice as stiff: about a tenth of Zircaloy's
    creep-down.  Without irradiation creep FeCrAl does not creep down."""
    zircaloy = _run(cladding=fuel.ZircaloyCladding())
    fecral = _run(cladding=fuel.FeCrAlCladding())
    thermal_only = _run(cladding=fuel.FeCrAlCladding(irradiation_creep_compliance=0.0))
    creepdown = lambda out: out.clad_hoop_strain[1, 0] - out.clad_hoop_strain[-1, 0]  # noqa: E731
    assert creepdown(zircaloy) > 0
    assert 0 < creepdown(fecral) < 0.2 * creepdown(zircaloy)
    assert creepdown(thermal_only) < 0.1 * creepdown(fecral)


def test_sic_cladding_swells_outward_and_runs_hot():
    """SiC swells under irradiation and does not creep, so its diameter grows
    while Zircaloy's shrinks.  Irradiation also cuts its conductivity to a
    few W/(m K) (against about 17 for Zircaloy), so the temperature drop
    across the wall is several times larger, and the fuel runs hotter."""
    sic = _run(cladding=fuel.SiCCladding(dpa_per_fast_neutron_fluence=1e-25, meyer_hardness=5e9))
    zircaloy = _run()
    assert sic.clad_hoop_strain[-1, 0] > sic.clad_hoop_strain[1, 0]
    assert zircaloy.clad_hoop_strain[-1, 0] < zircaloy.clad_hoop_strain[1, 0]
    drop = lambda out: out.clad_inner_temperature[-1, 0] - out.clad_outer_temperature[-1, 0]  # noqa: E731
    assert drop(sic) > 3 * drop(zircaloy)
    assert sic.fuel_centerline_temperature[-1, 0] > zircaloy.fuel_centerline_temperature[-1, 0]


def test_doped_uo2_releases_less_gas():
    """Five times larger grains outweigh the faster diffusion."""
    history = fuel.PowerHistory(linear_heat_rate=[1e3, 40e3, 40e3], time=[0, 3600, 300 * DAY])
    uo2 = _run(fuel.UO2Fuel(), history=history, step=30 * DAY)
    doped = _run(fuel.DopedUO2Fuel(), history=history, step=30 * DAY)
    assert uo2.fission_gas_release[-1] > 0
    assert doped.fission_gas_release[-1] < uo2.fission_gas_release[-1]


def test_a_chromium_coating_changes_the_temperatures_little_but_slows_creep_down():
    """A 15 um chromium layer on the outside of the cladding.

    * Before any creep, the coating changes the temperatures by less than a
      kelvin: it conducts four times better than Zircaloy and is 3 % of the
      wall.
    * Chromium barely creeps at 600 K, so as the Zircaloy creeps the coating
      takes a growing share of the hoop load, and the cladding creeps down
      less.  The gap then stays wider and the fuel warmer.  (The coating
      stress grows to several hundred MPa here.  A real coating would yield,
      and plasticity is not modelled, so this effect is an upper bound.)
    """
    coated_geometry = fuel.RodGeometry.from_diameters(8.19e-3, 8.36e-3, 9.50e-3, 0.05, clad_coating_thickness=15e-6)
    coated = _run(cladding=fuel.CoatedCladding(substrate=fuel.ZircaloyCladding(), coating=fuel.ChromiumCoating()), geometry=coated_geometry)
    bare = _run()
    early = slice(0, 2)
    difference = coated.fuel_centerline_temperature[early] - bare.fuel_centerline_temperature[early]
    assert np.abs(difference).max() < 1.0
    creepdown = lambda out: out.clad_hoop_strain[1, 0] - out.clad_hoop_strain[-1, 0]  # noqa: E731
    assert 0 < creepdown(coated) < creepdown(bare)
    assert coated.gap_width[-1, 0] > bare.gap_width[-1, 0]
    with pytest.raises(ValueError, match="coated cladding needs both"):
        _run(geometry=coated_geometry)
