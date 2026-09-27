# SPDX-License-Identifier: LGPL-2.1-or-later
"""Displacement damage (dualmesh.fuel.dpa).

* The NRT and arc-dpa displacement functions follow Nordlund et al. (2018),
  Eqs. 2-4 and Table 1.
* Robinson's damage energy partition (INDC(NDS)-0648, p. 35), applied to
  elastic scattering that is isotropic in the centre-of-mass frame,
  reproduces the mean damage energy per reaction of the NJOY-2016
  processing of ENDF/B-VIII.0 (MT444 over the total cross section) within
  1 % at 10 keV, where elastic scattering dominates and is isotropic.  This
  checks the partition formula against an independent calculation.
* The spectrum averages are exact for simple spectra, and the iron cross
  section has its known values (2570 b at 14 MeV).
"""

from __future__ import annotations

import numpy as np
import pytest
from dualmesh.fuel import dpa


def test_nrt_and_arc_displacements():
    Ed, b, c = dpa.ARC_DPA["Fe"]
    assert (Ed, b, c) == (40.0, -0.568, 0.286)
    T = np.array([30.0, 60.0, 100.0, 1e4])
    assert dpa.nrt_displacements(T, Ed) == pytest.approx([0.0, 1.0, 1.0, 100.0])
    arc = dpa.arc_displacements(T, Ed, b, c)
    # Continuous at 2 Ed / 0.8, where the efficiency is one, and tending to c
    # times NRT at high energy.
    assert arc[:3] == pytest.approx([0.0, 1.0, 1.0])
    assert dpa.arc_efficiency(100.0, Ed, b, c) == pytest.approx(1.0)
    xi = (1 - c) / 100.0**b * 1e4**b + c
    assert arc[3] == pytest.approx(100.0 * xi)
    assert dpa.arc_efficiency(1e9, Ed, b, c) == pytest.approx(c, abs=1e-3)


def test_damage_energy_reproduces_njoy_in_the_elastic_range():
    table = dpa._table()
    energy = np.sqrt(table["lower_eV"] * table["upper_eV"])
    x, w = np.polynomial.legendre.leggauss(200)
    x, w = 0.5 * (x + 1), 0.5 * w
    for element in ("C", "Si", "Cr", "Fe"):
        Z, A = dpa.ELEMENTS[element]
        k = np.searchsorted(energy, 1e4)
        njoy = table[f"{element}_damage_eV_b"][k] / table[f"{element}_total_b"][k]
        lam = 4 * A / (1 + A) ** 2
        mean = dpa.damage_energy(lam * energy[k] * x, Z, A) @ w
        assert mean == pytest.approx(njoy, rel=0.01)


def test_iron_cross_section_and_spectrum_averages():
    iron = dpa.nrt_cross_section("Fe")
    # 0.8 x 255 keV b / 80 eV at 14 MeV (the MT444 value); about 490 b at 1 MeV.
    assert iron.at(1.4e7) == pytest.approx(2570.0, rel=0.01)
    assert iron.at(1e6) == pytest.approx(493.0, rel=0.02)
    # A single-group spectrum averages to that group's value.
    one = dpa.Spectrum(np.array([1.0e6, 1.1e6]), np.array([1.0]))
    assert dpa.spectrum_averaged(iron, one) == pytest.approx(iron.on(one)[0])
    # dose per fluence above a threshold below the spectrum = the average.
    assert dpa.dpa_per_fluence(iron, one, 1e5) == pytest.approx(
        dpa.spectrum_averaged(iron, one) * 1e-28
    )
    # Splitting a group in proportion to lethargy.
    flat = dpa.Spectrum(np.array([1e5, 1e7]), np.array([2.0]))
    assert flat.fluence_above(1e6) == pytest.approx(1.0)
    # Conversions both ways.
    factor = dpa.dpa_per_fluence(
        iron, dpa.Spectrum.from_function(dpa.watt_spectrum(), np.geomspace(1e3, 2e7, 200))
    )
    assert dpa.fluence_from_dose(dpa.dose_from_fluence(1e25, factor), factor) == pytest.approx(1e25)


def test_elastic_cross_section_for_isotropic_scattering():
    """Far above threshold the NRT count is 0.8 T_d / 2E_d, so the elastic
    displacement cross section is 0.8 / (2 E_d) times sigma times the mean
    damage energy of a uniform recoil distribution."""
    Z, A = dpa.ELEMENTS["C"]
    E, sigma, Ed = 1e5, 4.0, 31.0
    lam = 4 * A / (1 + A) ** 2
    x = np.linspace(0, 1, 200001)
    mean = np.trapezoid(dpa.damage_energy(lam * E * x, Z, A), x)
    value = dpa.elastic_cross_section(E, sigma, Z, A, Ed)[0]
    # Recoils below 2 Ed / 0.8 count one displacement instead of the linear
    # law, a tiny correction at 100 keV.
    assert value == pytest.approx(sigma * 0.8 * mean / (2 * Ed), rel=1e-3)


def test_compound_and_fecral_dose_per_fluence():
    """For a U-235 fission spectrum (Watt form) the NRT dose per fluence
    above 0.1 MeV is close to the published rules of thumb, 0.9 dpa per
    1e25 n/m^2 for FeCrAl and 1 for SiC (IAEA-TECDOC-1921); per fluence
    above 1 MeV it is about 1.4 and 1.6 times that."""
    bounds = np.geomspace(1e-5, 2e7, 616)
    spectrum = dpa.Spectrum.from_function(dpa.watt_spectrum(), bounds)
    iron = dpa.nrt_cross_section("Fe")
    apmt = dpa.compound_cross_section(
        {
            "Fe": (0.70, iron),
            "Cr": (0.21, dpa.nrt_cross_section("Cr", 40.0)),
            "Al": (0.09, dpa.nrt_cross_section("Al", 27.0)),
        }
    )
    sic = dpa.compound_cross_section(
        {
            "Si": (0.5, dpa.nrt_cross_section("Si", 35.0)),
            "C": (0.5, dpa.nrt_cross_section("C", 20.0)),
        }
    )
    assert 1e25 * dpa.dpa_per_fluence(apmt, spectrum, 1e5) == pytest.approx(0.96, abs=0.1)
    assert 1e25 * dpa.dpa_per_fluence(sic, spectrum, 1e5) == pytest.approx(1.13, abs=0.1)
    assert 1e25 * dpa.dpa_per_fluence(apmt, spectrum) == pytest.approx(1.36, abs=0.05)
    with pytest.raises(ValueError, match="sum to one"):
        dpa.compound_cross_section({"Fe": (0.5, iron)})
