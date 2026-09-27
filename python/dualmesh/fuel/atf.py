# SPDX-License-Identifier: LGPL-2.1-or-later
r"""Accident tolerant fuel (ATF) materials.

Accident tolerant fuels are fuel and cladding materials meant to withstand a
loss of cooling for longer than UO2 in Zircaloy, mainly by oxidizing more
slowly in high-temperature steam and by keeping their strength.  This module
provides the concepts for which open-literature property correlations exist:

========================  ====================================================
:class:`FeCrAlCladding`   iron-chromium-aluminium alloy cladding (Kanthal APMT
                          and the ORNL alloys C06M, C35M, C36M)
:class:`SiCCladding`      silicon carbide fibre-reinforced silicon carbide
                          composite (SiC/SiC) cladding
:class:`ChromiumCoating`  a chromium coating on the outside of a cladding,
                          used with :class:`CoatedCladding`
:class:`U3Si2Fuel`        uranium silicide fuel
:class:`DopedUO2Fuel`     UO2 doped with Cr2O3 (larger grains, faster gas
                          diffusion)
========================  ====================================================

The correlations were read in the sources named with each class, and every
default states where it comes from.  Where a source gives no value for a
property a rod model needs, the property is a required input (no default),
so that no guess enters a calculation unannounced.  All the materials are
built on the expression machinery of :mod:`dualmesh.fuel.materials`: they are
compiled once and differentiated exactly, like the built-in ones.

Sources read
------------
* K. G. Field, M. A. Snead, Y. Yamamoto, K. A. Terrani, "Handbook on the
  Material Properties of FeCrAl Alloys for Nuclear Power Production
  Applications (FY18 Version: Revision 1)", ORNL/SPR-2018/905, 2018.
* T. Koyanagi, Y. Katoh, G. Jacobsen, C. Deck, "Handbook of LWR SiC/SiC
  Cladding Properties - Revision 1", ORNL/TM-2018/912, 2018.
* P. Aragon et al., "Fuel performance modelling of Cr-coated Zircaloy
  cladding under DBA/LOCA conditions", Annals of Nuclear Energy 211 (2025)
  110950 (open access).
* K. A. Gamble, J. D. Hales, G. Pastore, T. Barani, D. Pizzocri, "Behavior
  of U3Si2 Fuel and FeCrAl Cladding under Normal Operating and Accident
  Reactor Conditions", INL/EXT-16-40059, 2016.
* K. A. Gamble, G. Pastore, M. W. D. Cooper, "BISON Development and
  Validation for Priority LWR-ATF Concepts", INL/EXT-20-59969, 2020.
* K. A. Terrani, T. M. Karlsen, Y. Yamamoto, "Input Correlations for
  Irradiation Creep of FeCrAl and SiC Based on In-Pile Halden Test Results",
  ORNL/TM-2016/191, 2016.
* IAEA-TECDOC-1921, "Analysis of Options and Experimental Examination of Fuels
  for Water Cooled Reactors with Increased Accident Tolerance (ACTOF)", 2020.
* K. A. Gamble, G. Pastore, M. W. D. Cooper, D. Andersson, "ATF material model
  development and validation for priority fuel concepts",
  CASL-U-2019-1870-000, 2019.
* J. R. Stephens, W. D. Klopp, "High-Temperature Creep of Polycrystalline
  Chromium", NASA TM X-2499, 1972.
* A. R. Massih, L. O. Jernkvist, "Effects of additives on UO2 fuel behavior:
  expanded edition", SSM Report 2021:20.
* NIST-JANAF Thermochemical Tables, beta-SiC (C-101), for a check of the SiC
  specific heat.
* The Kanthal APMT tube datasheet (kanthal.com), for the density, the elastic
  constants, the hardness and the emissivity of APMT.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np

from .gas import ATHERMAL_COEFFICIENT
from .materials import (
    CladdingMaterial,
    CustomCladding,
    CustomFuel,
    FuelMaterial,
    RodContext,
    UO2Fuel,
    uranium_molar_mass,
)
from .specification import parameter

BOLTZMANN_EV = 8.617333262e-5  # eV/K
GAS_CONSTANT = 8.314462618  # J/(mol K)


class _Delegating:
    """A material defined by a CustomFuel or CustomCladding that it builds,
    once, from its own fields (see :meth:`_custom`)."""

    def _custom(self):  # pragma: no cover - abstract
        raise NotImplementedError

    def _delegate(self):
        if self.__dict__.get("_built") is None:
            self.__dict__["_built"] = self._custom()
        return self.__dict__["_built"]

    def expression(self, name: str):
        """The expression (or number) this material uses for a property,
        for checking and plotting."""
        return getattr(self._delegate(), name)


class _ExpressionCladding(_Delegating, CladdingMaterial):
    """A cladding material defined by a CustomCladding."""

    has_coating = False

    @property
    def material_name(self) -> str:
        return self._delegate().name

    def add_thermal_material(self, problem, block, context):
        self._delegate().add_thermal_material(problem, block, context)

    def add_elasticity(self, problem, block, context):
        self._delegate().add_elasticity(problem, block, context)

    def add_eigenstrains(self, problem, block, context):
        return self._delegate().add_eigenstrains(problem, block, context)

    def creep_parameters(self, context, block="clad"):
        return self._delegate().creep_parameters(context, block)


def _polynomial_integral(coefficients, variable: str, scale: float = 1.0) -> str:
    """The expression of scale * the integral of sum_k c_k x^k from 0, for a
    thermal strain from an instantaneous expansion coefficient."""
    terms = []
    for k, c in enumerate(coefficients):
        if c == 0:
            continue
        terms.append(f"({scale * c / (k + 1)!r})*{variable}^{k + 1}")
    return " + ".join(terms)


# ---------------------------------------------------------------------------
# FeCrAl
# ---------------------------------------------------------------------------
# Field et al. (2018), Tables 1 to 3.  Specific heat below the Curie
# temperature: a T + b T^2 + C T^3; above it the extra terms D/T + E ln(|T - Tc|/Tc).
_FECRAL_SPECIFIC_HEAT = {
    #        a      b           C          a_high  b_high      C_high     D        E       Tc
    "APMT": (2.540, -4.311e-3, 2.982e-6, 1.840, -1.843e-3, 0.643e-6, -5.712e3, -50.38, 852.0),
    "C06M": (2.430, -3.957e-3, 2.656e-6, 1.827, -1.807e-3, 0.6134e-6, -9.419e3, -54.54, 888.0),
    "C35M": (2.450, -4.002e-3, 2.720e-6, 1.946, -2.002e-3, 0.698e-6, -1.652e3, -53.93, 870.0),
    "C36M": (2.995, -5.953e-3, 4.516e-6, 1.456, -1.296e-3, 0.438e-6, 26.45e3, -46.89, 771.0),
}
# Thermal conductivity A1 T^2 + A2 T + A3, W/(m K).
_FECRAL_CONDUCTIVITY = {
    "APMT": (-7.223e-7, 1.563e-2, 6.569),
    "C06M": (6.762e-7, 1.032e-2, 9.956),
    "C35M": (-19.860e-7, 1.537e-2, 8.502),
    "C36M": (-9.184e-7, 1.368e-2, 8.187),
}
# Expansion coefficient A1 T^3 + A2 T^2 + A3 T + A4, in 1e-6 / K.
_FECRAL_EXPANSION = {
    "APMT": (1.771e-10, 9.558e-7, 1.937e-3, 10.27),
    "C06M": (10.74e-10, -21.36e-7, 4.694e-3, 10.03),
    "C35M": (9.095e-10, -17.46e-7, 4.530e-3, 9.810),
    "C36M": (3.079e-10, 2.719e-7, 2.535e-3, 10.56),
}


@dataclass
class FeCrAlCladding(_ExpressionCladding):
    r"""Iron-chromium-aluminium alloy cladding, with the correlations of the
    ORNL FeCrAl handbook (Field et al. 2018):

    * specific heat (Eqs. 3.1 and 3.2, Table 1), with the magnetic peak at
      the Curie temperature (the logarithm is evaluated at least 1 K from
      it, where the fit is singular);
    * thermal conductivity :math:`\kappa = A_1 T^2 + A_2 T + A_3` (Eq. 3.4,
      Table 2);
    * thermal strain :math:`\varepsilon = \alpha (T - 293.15\,\mathrm{K})` with
      :math:`\alpha = A_1 T^3 + A_2 T^2 + A_3 T + A_4` (Eq. 3.5, Table 3) read
      as the mean coefficient from room temperature.  The handbook does not
      say which coefficient it gives; read as a mean coefficient it
      reproduces the thermal strains of the Kanthal APMT datasheet within 7 %
      (0.3 % at 1273 K), while its integral, the instantaneous reading, falls
      11-15 % short above 700 K;
    * for the ORNL alloys, Young's modulus :math:`E = -5.46\times10^{-5} T^2 -
      3.85\times10^{-2} T + 199` GPa and Poisson's ratio :math:`\nu =
      4.46\times10^{-5} T + 0.27` with T in degrees Celsius (Eqs. 3.6 and
      3.7); for APMT, :math:`E = 219.85 - 7.094\times10^{-2} T - 1.928\times10^{-5}
      T^2` GPa (T in Celsius, a least-squares fit within 2.6 GPa to the
      datasheet's 220 to 130 GPa from 20 to 1000 C) and :math:`\nu = 0.30`
      (datasheet);
    * thermal creep :math:`\dot\varepsilon = A_0 \sigma^n \exp(-Q/RT)` with the
      handbook's generalized fit for all alloys, :math:`A_0 = 0.83` MPa^-n/s,
      :math:`n = 7.1`, :math:`Q = 326` kJ/mol (Eq. 3.9, Table 4), fitted over
      350 to 1200 C and 1 to 150 MPa.  It overestimates the creep of APMT,
      which the handbook describes as more creep resistant than the wrought
      alloys and for which it gives no separate fit: against the datasheet's
      secondary creep strengths of APMT the law is 30 to 280 times too fast at
      800 to 1100 C.  At LWR cladding temperatures it is negligible;
    * irradiation creep :math:`\dot\varepsilon = B \sigma \dot d` with the
      compliance :math:`B = 5\times10^{-6}` MPa^-1 dpa^-1 recommended for
      wrought FeCrAl by Terrani et al. (ORNL/TM-2016/191, Sect. 4.2; the bcc
      range 0.5 to 5e-6) and the dose rate :math:`\dot d` from the fast
      neutron flux.  With the ACTOF conversion of 0.9 dpa per 1e25 n/m^2 it
      is the benchmark law of IAEA-TECDOC-1921, Eq. 5.  At LWR conditions it
      exceeds the thermal creep by about four orders of magnitude.

    The handbook and Terrani et al. report no measurable swelling; an optional
    swelling rate per dpa allows the parametric upper bound of Terrani et al.
    (0.05 % per dpa).
    """

    meyer_hardness: float | None = parameter(
        None,
        unit="Pa",
        description="Meyer hardness of the softer of the pellet and cladding surfaces, for the "
        "solid contact conductance. Default None: 2.45 GPa for APMT (250 HV in the Kanthal "
        "datasheet, with the Meyer hardness taken equal to the Vickers hardness); required for "
        "the ORNL alloys.",
    )
    dpa_per_fast_neutron_fluence: float = parameter(
        0.9e-25,
        unit="dpa per n/m^2",
        description="Displacement damage per unit fast neutron fluence of the rod (E > 1 MeV), "
        "for irradiation creep. Default 0.9e-25, the conversion of IAEA-TECDOC-1921 (1e25 n/m^2 "
        "= 0.9 dpa, after Field et al. 2015, energy threshold not stated). dualmesh.fuel.dpa "
        "gives, for APMT in a U-235 fission spectrum, 0.96e-25 NRT dpa per n/m^2 above 0.1 MeV "
        "and 1.36e-25 above 1 MeV, so the published rule matches a 0.1 MeV basis; compute the "
        "factor for the reactor's spectrum with that module.",
    )
    irradiation_creep_compliance: float = parameter(
        5.0e-6,
        unit="1/(MPa dpa)",
        description="B of the irradiation creep. Default 5e-6, Terrani et al. (2016); 0 turns "
        "irradiation creep off.",
    )
    swelling_per_dpa: float = parameter(
        0.0,
        description="Volumetric swelling per dpa. Default 0 (none measured); 5e-4 is the "
        "parametric upper bound of Terrani et al. (2016).",
    )
    alloy: str = parameter(
        "APMT",
        description="APMT (Kanthal APMT, Fe-21Cr-5Al-3Mo), C06M, C35M or C36M (ORNL alloys). "
        "Default APMT, the commercial alloy of the lead test rods.",
    )
    density: float = parameter(
        7250.0,
        unit="kg/m^3",
        description="Default 7250, Kanthal APMT (datasheet); the handbook gives no density, so "
        "set it for the ORNL alloys.",
    )
    surface_roughness: float = parameter(
        1.0e-6, unit="m", description="Inner surface roughness. Default 1 um, a drawn tube."
    )
    emissivity: float = parameter(
        0.70,
        description="Inner surface emissivity. Default 0.70, fully oxidised APMT (Kanthal "
        "datasheet); the inner surface of a new tube is less oxidised and may be lower.",
    )

    def __post_init__(self):
        if self.alloy not in _FECRAL_CONDUCTIVITY:
            raise ValueError(
                f"FeCrAlCladding: unknown alloy '{self.alloy}'. "
                f"Use {', '.join(_FECRAL_CONDUCTIVITY)}."
            )
        if self.meyer_hardness is None:
            if self.alloy != "APMT":
                raise ValueError(
                    f"FeCrAlCladding: give meyer_hardness for {self.alloy}; the default is for "
                    "APMT only."
                )
            self.meyer_hardness = 250.0 * 9.80665e6

    def _custom(self) -> CustomCladding:
        a, b, c, a2, b2, c2, d2, e2, tc = _FECRAL_SPECIFIC_HEAT[self.alloy]
        below = f"({a}*temperature + {b}*temperature^2 + {c}*temperature^3)"
        above = (
            f"({a2}*temperature + {b2}*temperature^2 + {c2}*temperature^3 + {d2}/temperature"
            f" + {e2}*log(max(abs(temperature - {tc}), 1)/{tc}))"
        )
        k1, k2, k3 = _FECRAL_CONDUCTIVITY[self.alloy]
        e1, e2_, e3, e4 = _FECRAL_EXPANSION[self.alloy]
        tcel = "(temperature - 273.15)"
        if self.alloy == "APMT":
            youngs = f"1e9*(219.85 - 7.094e-2*{tcel} - 1.928e-5*{tcel}^2)"
            poisson = 0.30
        else:
            youngs = f"1e9*(-5.46e-5*{tcel}^2 - 3.85e-2*{tcel} + 199)"
            poisson = f"4.46e-5*{tcel} + 0.27"
        swelling = None
        if self.swelling_per_dpa:
            swelling = "swelling_per_dpa*fast_neutron_fluence*dpa_per_fluence"
        return CustomCladding(
            name=f"FeCrAl ({self.alloy})",
            thermal_conductivity=f"{k1}*temperature^2 + {k2}*temperature + {k3}",
            specific_heat=f"if(temperature <= {tc}, {below}, {above})",
            density=self.density,
            youngs_modulus=youngs,
            poissons_ratio=poisson,
            thermal_strain=(
                f"1e-6*({e1}*temperature^3 + {e2_}*temperature^2 + {e3}*temperature + {e4})"
                "*(temperature - 293.15)"
            ),
            meyer_hardness=self.meyer_hardness,
            creep_rate=(
                "0.83*(von_mises_stress/1e6)^7.1*exp(-326000/(8.314462618*temperature))"
                " + B_irr*(von_mises_stress/1e6)*fast_neutron_flux*dpa_per_fluence"
            ),
            volumetric_swelling=swelling,
            surface_roughness=self.surface_roughness,
            emissivity=self.emissivity,
            constants={
                "B_irr": self.irradiation_creep_compliance,
                "dpa_per_fluence": self.dpa_per_fast_neutron_fluence,
                "swelling_per_dpa": self.swelling_per_dpa,
            },
        )


# ---------------------------------------------------------------------------
# SiC/SiC
# ---------------------------------------------------------------------------
@dataclass
class SiCCladding(_ExpressionCladding):
    r"""SiC fibre-reinforced SiC composite cladding (CVI SiC/SiC), with the
    properties of the ORNL handbook (Koyanagi et al. 2018):

    * density 2700 kg/m^3 (Table S.1, about 2.7 g/cm^3);
    * thermal expansion from the instantaneous coefficient
      :math:`\alpha = -0.7765 + 1.4350\times10^{-2} T - 1.2209\times10^{-5} T^2 +
      3.8289\times10^{-9} T^3` (1e-6/K, 293 to 1273 K, Eq. 1), unchanged by
      irradiation;
    * specific heat of monolithic SiC (the handbook's Section 3.1.4 and
      Fig. 4), :math:`c_p = 925.65 + 0.3772 T - 7.9259\times10^{-5} T^2 -
      3.1946\times10^{7}/T^2` J/(kg K), the correlation of Snead et al. (2007,
      not read) as given in IAEA-TECDOC-1921 Eq. 17; it agrees with the
      NIST-JANAF table of beta-SiC within 1.6 % from 298 to 2000 K;
    * through-thickness thermal conductivity :math:`1/k = 1/k_0 + c_R S`: the
      unirradiated :math:`k_0` is 8.0 - 1.32e-3 (T - 293) W/(m K), a linear fit
      to the middle of the diffusivity ranges of full-composite tubes in
      Table 5 times :math:`\rho c_p`; the irradiation defect resistance grows
      linearly with the swelling S (Section 4.1.4), with
      :math:`c_R` = ``defect_thermal_resistivity_per_swelling``;
    * volumetric swelling (Eqs. 6 to 8)
      :math:`S = S_s [1 - \exp(-\gamma/\gamma_c)]^{2/3}`, with the dose
      :math:`\gamma` in dpa and :math:`S_s`, :math:`\gamma_c` cubic in the
      irradiation temperature.  The handbook prints Eq. 6 as
      :math:`[1 - \exp(1 - \gamma/\gamma_c)]^{2/3}`, which is negative below
      :math:`\gamma_c`; the form above reproduces its Fig. 15.  The
      irradiation temperature is the highest temperature the cladding has
      reached (the rod's ``maximum_temperature`` field, or the current
      temperature if higher), limited to the 473 to 1073 K of Fig. 15, so
      that the swelling does not grow when the rod cools (SiC does not swell
      on cooling; above its irradiation temperature it anneals, which is not
      modelled);
    * Young's modulus 200 GPa and Poisson's ratio 0.12: the handbook reports
      173 to 248 GPa (axial) and 158 to 213 GPa (hoop) for tubes, and 0.12
      axially (Section 3.2.1); the composite is modelled as isotropic.

    Irradiation creep is not modelled: the handbook states that it is small
    compared with the swelling and can be ignored on current data.
    """

    dpa_per_fast_neutron_fluence: float = parameter(
        unit="dpa per n/m^2",
        description="Displacement damage in SiC per unit fast neutron fluence (the fluence of "
        "the rod, E > 1 MeV). It depends on the neutron spectrum; take it from the neutronics "
        "of the reactor. No default.",
    )
    meyer_hardness: float = parameter(
        unit="Pa",
        description="Meyer hardness of the softer of the pellet and cladding surfaces, for the "
        "solid contact conductance. No default: no value was found in the sources.",
    )
    defect_thermal_resistivity_per_swelling: float = parameter(
        11.2,
        unit="m K/W",
        description="c_R in 1/k = 1/k_0 + c_R S. Default 11.2: the defect resistivity of the "
        "full SiC/SiC tube of the handbook's Fig. 17a (about 0.2 m K/W after 2.3 dpa near "
        "630 K) divided by the swelling there (1.8 %). Monolithic SiC has about 6 (Snead et "
        "al. 2007, per the public BISON documentation).",
    )
    youngs_modulus: float = parameter(
        2.0e11, unit="Pa", description="Default 200 GPa, inside the tube ranges of the handbook."
    )
    poissons_ratio: float = parameter(
        0.12, description="Default 0.12, measured axially on a tube (the handbook)."
    )
    density: float = parameter(2700.0, unit="kg/m^3", description="Default 2700 (the handbook).")
    surface_roughness: float = parameter(
        1.0e-6, unit="m", description="Inner surface roughness. Default 1 um."
    )
    emissivity: float = parameter(
        0.8, description="Inner surface emissivity. Default 0.8, as for the metals here."
    )

    def __post_init__(self):
        if not self.dpa_per_fast_neutron_fluence > 0:
            raise ValueError("SiCCladding: dpa_per_fast_neutron_fluence must be positive.")

    def swelling_expression(self) -> str:
        T = "min(max(max(temperature, maximum_temperature), 473.15), 1073.15)"
        saturation = f"(5.8366e-2 - 1.0089e-4*{T} + 6.9368e-8*{T}^2 - 1.8152e-11*{T}^3)"
        critical = f"(-0.57533 + 3.3342e-3*{T} - 5.3970e-6*{T}^2 + 2.9754e-9*{T}^3)"
        dose = "(fast_neutron_fluence*dpa_per_fluence)"
        return f"{saturation}*(1 - exp(-{dose}/{critical}))^(2/3)"

    def _custom(self) -> CustomCladding:
        swelling = self.swelling_expression()
        k0 = "(8.0 - 1.32e-3*(temperature - 293))"
        return CustomCladding(
            name="SiC/SiC",
            thermal_conductivity=f"1/(1/{k0} + c_R*{swelling})",
            specific_heat=(
                "925.65 + 0.3772*temperature - 7.9259e-5*temperature^2 - 3.1946e7/temperature^2"
            ),
            density=self.density,
            youngs_modulus=self.youngs_modulus,
            poissons_ratio=self.poissons_ratio,
            thermal_strain=_polynomial_integral(
                [-0.7765, 1.4350e-2, -1.2209e-5, 3.8289e-9], "temperature", 1e-6
            ),
            meyer_hardness=self.meyer_hardness,
            volumetric_swelling=swelling,
            surface_roughness=self.surface_roughness,
            emissivity=self.emissivity,
            constants={
                "dpa_per_fluence": self.dpa_per_fast_neutron_fluence,
                "c_R": self.defect_thermal_resistivity_per_swelling,
            },
        )


# ---------------------------------------------------------------------------
# Chromium coating
# ---------------------------------------------------------------------------
@dataclass
class ChromiumCoating(_ExpressionCladding):
    r"""A pure chromium coating, with the properties compiled by Aragon et
    al. (2025) for TRANSURANUS from the open literature (T_C is the
    temperature in degrees Celsius):

    * density 7200 kg/m^3 (Simmons and Wang 1971);
    * :math:`k = 87.56671 - 0.04179 T_C + 3.15147\times10^{-5} T_C^2 -
      2.06676\times10^{-8} T_C^3` W/(m K) (Holzwarth and Stamm 2002, 20 to
      1000 C, Eq. 1);
    * :math:`c_p = 1000 (0.48047 + 6.34753\times10^{-5} T_C +
      2.34120\times10^{-7} T_C^2 - 1.27824\times10^{-10} T_C^3)` J/(kg K)
      (Holzwarth and Stamm 2002, Eq. 2);
    * thermal expansion from the coefficient :math:`(8.3159 +
      1.80901\times10^{-3} T_C + 6.45421\times10^{-7} T_C^2 +
      1.27483\times10^{-10} T_C^3)\times10^{-6}` 1/K (Holzwarth and Stamm
      2002, Eq. 4), taken as the instantaneous coefficient;
    * :math:`E = 264.11 - 0.01 T - 2.5\times10^{-5} T^2` GPa with T in
      kelvin (Wagih et al. 2018, Eq. 5) and :math:`\nu = 0.22` (Simmons and
      Wang 1971);
    * thermal creep :math:`\dot\varepsilon = A \sigma^n \exp(-Q/RT)` with
      :math:`A = 43.19` MPa^-n/s, :math:`n = 4.769`, :math:`Q = 333.6` kJ/mol:
      a least-squares fit to all 49 minimum creep rates of Table I of
      Stephens and Klopp (NASA TM X-2499, 1972; 816 to 1316 C, 3.7 to 100
      MPa, two grain sizes; rms error of ln(rate) 0.51).  The law of Wagih et
      al. (2018) that Aragon et al. use (A = 5.1596e-3, n = 6.2, Q = 306.3
      kJ/mol) fits the 816 C data only and is 3 to 100 times too slow from
      982 to 1316 C.  ``thermal_creep="wagih"`` selects it.  Below 816 C there
      are no data; both laws are extrapolations there;
    * irradiation creep :math:`\dot\varepsilon = B \sigma \dot d` with
      ``dpa_per_fast_neutron_fluence`` given.  No value for chromium was
      found; the default compliance is the upper end of the range 0.5 to 5e-6
      MPa^-1 dpa^-1 for bcc metals quoted by Terrani et al. (2016).  Aragon et
      al. use the Zircaloy law instead, in the absence of data;
    * with ``dpa_per_fast_neutron_fluence`` given, irradiation swelling of
      0.14 % per dpa up to 5.9 dpa and 0.05 % per dpa beyond (Eq. 6).

    Plasticity and cracking of the coating are not modelled.  Its stress can
    reach several hundred MPa, above the room-temperature yield strength of
    chromium reported in the literature (a few hundred MPa), so the coating's
    share of the load, and its effect on creep-down, is an upper bound.  Its
    surface properties do not enter the gap, which the cladding's inner
    surface faces.
    """

    dpa_per_fast_neutron_fluence: float | None = parameter(
        None,
        unit="dpa per n/m^2",
        description="Displacement damage in chromium per unit fast neutron fluence, for the "
        "irradiation swelling and creep. Default None: neither is modelled.",
    )
    irradiation_creep_compliance: float = parameter(
        5.0e-6,
        unit="1/(MPa dpa)",
        description="B of the irradiation creep. Default 5e-6, the upper end of the bcc range "
        "of Terrani et al. (2016); used only with dpa_per_fast_neutron_fluence.",
    )
    thermal_creep: str = parameter(
        "stephens_klopp",
        description="stephens_klopp (the fit to all of their data) or wagih (Wagih et al. "
        "2018, the 816 C data). Default stephens_klopp.",
    )
    surface_roughness: float = parameter(1.0e-6, unit="m", description="Not used (outer surface).")
    emissivity: float = parameter(0.8, description="Not used (outer surface).")
    meyer_hardness: float = parameter(6.8e8, unit="Pa", description="Not used (outer surface).")

    def _custom(self) -> CustomCladding:
        tc = "(temperature - 273.15)"
        swelling = None
        constants = {}
        if self.thermal_creep == "stephens_klopp":
            creep = "43.19*(von_mises_stress/1e6)^4.7691*exp(-333640/(8.314462618*temperature))"
        elif self.thermal_creep == "wagih":
            creep = "5.1596e-3*(von_mises_stress/1e6)^6.2*exp(-306268.8/(8.314462618*temperature))"
        else:
            raise ValueError("ChromiumCoating: thermal_creep must be stephens_klopp or wagih.")
        if self.dpa_per_fast_neutron_fluence is not None:
            dose = "(fast_neutron_fluence*dpa_per_fluence)"
            swelling = f"0.01*if({dose} <= 5.9, 0.14*{dose}, 0.05*{dose} + 0.531)"
            constants["dpa_per_fluence"] = self.dpa_per_fast_neutron_fluence
            constants["B_irr"] = self.irradiation_creep_compliance
            creep += " + B_irr*(von_mises_stress/1e6)*fast_neutron_flux*dpa_per_fluence"
        return CustomCladding(
            name="chromium",
            thermal_conductivity=(
                f"87.56671 - 0.04179*{tc} + 3.15147e-5*{tc}^2 - 2.06676e-8*{tc}^3"
            ),
            specific_heat=(
                f"1000*(0.48047 + 6.34753e-5*{tc} + 2.34120e-7*{tc}^2 - 1.27824e-10*{tc}^3)"
            ),
            density=7200.0,
            youngs_modulus="1e9*(264.11 - 0.01*temperature - 2.5e-5*temperature^2)",
            poissons_ratio=0.22,
            thermal_strain=_polynomial_integral(
                [8.3159, 1.80901e-3, 6.45421e-7, 1.27483e-10], tc, 1e-6
            ),
            meyer_hardness=self.meyer_hardness,
            volumetric_swelling=swelling,
            creep_rate=creep,
            surface_roughness=self.surface_roughness,
            emissivity=self.emissivity,
            constants=constants,
        )


@dataclass
class CoatedCladding(CladdingMaterial):
    """A cladding with a coating bonded to its outer surface: the substrate
    fills the block ``clad`` and the coating the block ``coating`` of the rod
    mesh, whose thickness is ``RodGeometry.clad_coating_thickness``.  The
    gap sees the substrate's inner surface."""

    substrate: CladdingMaterial = parameter(description="The cladding tube, e.g. ZircaloyCladding.")
    coating: CladdingMaterial = parameter(description="The coating, e.g. ChromiumCoating.")

    has_coating = True

    @property
    def material_name(self) -> str:
        return f"{self.coating.material_name}-coated {self.substrate.material_name}"

    @property
    def surface_roughness(self):
        return self.substrate.surface_roughness

    @property
    def emissivity(self):
        return self.substrate.emissivity

    @property
    def meyer_hardness(self):
        return self.substrate.meyer_hardness

    def _part(self, block: str) -> CladdingMaterial:
        return self.coating if block == "coating" else self.substrate

    def add_thermal_material(self, problem, block, context):
        self._part(block).add_thermal_material(problem, block, context)

    def add_elasticity(self, problem, block, context):
        self._part(block).add_elasticity(problem, block, context)

    def add_eigenstrains(self, problem, block, context):
        return self._part(block).add_eigenstrains(problem, block, context)

    def creep_parameters(self, context: RodContext, block: str = "clad") -> dict:
        return self._part(block).creep_parameters(context, block)


# ---------------------------------------------------------------------------
# U3Si2
# ---------------------------------------------------------------------------
@dataclass
class U3Si2Fuel(_Delegating, FuelMaterial):
    r"""Uranium silicide fuel.  The correlations are those of the LANL U3Si2
    property handbook (J. T. White, LA-UR-18-28719, 2018, not read) as
    reproduced in CASL-U-2019-1870 (Gamble et al. 2019) and IAEA-TECDOC-1921
    (2020), and those of INL/EXT-16-40059 (Gamble et al. 2016) and
    INL/EXT-20-59969 (Gamble, Pastore and Cooper 2020):

    * :math:`k = 4.996 + 0.0118 T` W/(m K), 300 to 1773 K, 5 %: the fit of the
      corrigendum of White et al. (J. Nucl. Mater. 484, 2017, not read),
      CASL-U-2019-1870 Eq. 23 and IAEA-TECDOC-1921 Eq. 11.  The uncorrected
      2015 fit, 6.004 + 0.0151 T, is 23-27 % higher.  No irradiation
      dependence is known;
    * :math:`c_p = (0.02582 T + 140.5)/0.77026` J/(kg K) (Eq. 4.2 of the 2016
      report; within 1 % of the handbook form);
    * :math:`E = 142.68 - 6.425 p` GPa and :math:`G = 61.27 - 2.901 p` GPa with
      the porosity p in per cent, :math:`\nu = E/2G - 1` (CASL-U-2019-1870
      Eqs. 39-41, 1.5 to 10 % porosity, 29 % uncertainty): 110.6 GPa and 0.182
      at 95 % density;
    * a constant thermal expansion coefficient :math:`16.0\times10^{-6}` 1/K,
      273 to 1473 K, :math:`\pm 3\times10^{-6}` (the handbook value,
      CASL-U-2019-1870 Sect. 3.4.2);
    * solid swelling :math:`0.34392\,Bu` with Bu in FIMA (CASL-U-2019-1870
      Eq. 68, 20 %), and with ``gaseous_swelling=True`` the empirical gaseous
      part of the fit of Metzger et al. to the data of Finlay et al.,
      :math:`3.88008\,Bu^2 + 0.45419\,Bu` (Eq. 70; CASL prints 3.8808,
      Metzger and the INL reports 3.88008).  Those data come from dispersion
      fuel at 300 to 500 K, where U3Si2 becomes amorphous; power-reactor
      data disagree (about 12 % at 6 GWd/tU in AI-7-1, 0 to 1 % in the ATF-1
      rodlets to 20 GWd/tU), and no validated gaseous swelling correlation for
      LWR conditions exists.  It is therefore off by default;
    * creep as the sum of Nabarro-Herring, Coble and dislocation-climb terms
      (Eqs. 3.3 to 3.6 of the 2020 report, whose journal version is Cooper et
      al., J. Nucl. Mater. 555 (2021) 153129, not read), with the grain size
      :math:`d = 2 a`; it reproduces the compressive creep tests of Yingling
      et al. (INL/JOU-20-58799, Table 1) within a factor of 2.3;
    * fission gas by diffusion out of the grains with the xenon diffusivity
      of stoichiometric U3Si2 (Eq. 3.1 of the 2020 report) and grain-boundary
      saturation with the coverage 0.6, surface energy 1.0 J/m^2 and
      semi-dihedral angle 73 degrees given there.  Those parameters belong to
      the cluster-dynamics model of Barani et al. (2019), with trapping and
      lenticular bubbles; in this reduced model (no trapping, the UO2 bubble
      radius of 0.5 um) they are not validated.

    The theoretical density, 12190 kg/m^3, follows from the uranium density
    11.3 g/cm^3 quoted in the 2016 report and the uranium mass fraction of
    U3Si2 (0.927).
    """

    enrichment: float = parameter(0.05, description="U-235 weight fraction. Default 0.05.")
    theoretical_density_fraction: float = parameter(
        0.95, description="Fabricated density as a fraction of 12190 kg/m^3. Default 0.95."
    )
    grain_radius: float = parameter(
        26.0e-6,
        unit="m",
        description="Mean grain radius. Default 26 um, the fresh fuel of the AI-7-1 experiment "
        "(Table 4.1 of the 2016 report).",
    )
    surface_roughness: float = parameter(
        2.0e-6, unit="m", description="Pellet surface roughness. Default 2 um, as for UO2."
    )
    emissivity: float = parameter(
        0.8,
        description="Pellet surface emissivity. Default 0.8, the UO2 value (no U3Si2 value was "
        "found in the sources read).",
    )
    energy_per_fission: float = parameter(
        200.0 * 1.602176634e-13, unit="J", description="Default 200 MeV."
    )
    gaseous_swelling: bool = parameter(
        False,
        description="Add the empirical gaseous swelling of Finlay et al. (see the class). "
        "Default False.",
    )

    material_name = "U3Si2"
    optional_models = {}
    fission_gas_release_models = ("booth", "none")
    theoretical_density = 12190.0

    @property
    def heavy_metal_molar_mass(self) -> float:
        return uranium_molar_mass(self.enrichment)

    @property
    def compound_molar_mass(self) -> float:
        return 3 * self.heavy_metal_molar_mass + 2 * 28.0855e-3

    def heavy_metal_atom_density(self) -> float:
        return self._delegate().heavy_metal_atom_density()

    def _custom(self) -> CustomFuel:
        kT = "(8.617333262e-5*temperature)"
        s = "von_mises_stress"
        d = "(2*grain_radius)"
        nabarro = (
            f"{s}/{d}^2*(3.023e-15*exp(-3.246/{kT}) + 6.812e-54*fission_rate*exp(-0.5179/{kT})"
            f" + 2.59e-17*exp(-3.330/{kT}))"
        )
        coble = f"{s}/{d}^3*2.280e-24*exp(-1.381/{kT})"
        climb = f"{s}^3*(3.444e-15*exp(-4.02/{kT}) + 3.759e-58*fission_rate*exp(-0.0178/{kT}))"
        return CustomFuel(
            name="U3Si2",
            theoretical_density=self.theoretical_density,
            compound_molar_mass=self.compound_molar_mass,
            heavy_metal_atoms_per_formula_unit=3,
            heavy_metal_molar_mass=self.heavy_metal_molar_mass,
            thermal_conductivity="4.996 + 0.0118*temperature",
            specific_heat="(0.02582*temperature + 140.5)/0.77026",
            youngs_modulus="1e9*(142.68 - 6.425*100*porosity)",
            poissons_ratio="(142.68 - 6.425*100*porosity)/(2*(61.27 - 2.901*100*porosity)) - 1",
            thermal_strain="16.0e-6*temperature",
            volumetric_swelling=(
                "0.34392*burnup + 3.88008*burnup^2 + 0.45419*burnup"
                if self.gaseous_swelling
                else "0.34392*burnup"
            ),
            creep_rate=f"{nabarro} + {coble} + {climb}",
            fission_gas_release="booth",
            fission_gas_diffusion_coefficient=(f"2.85e-4*exp(-3.17/{kT}) + 3.58e-42*fission_rate"),
            booth_parameters=dict(
                saturation_coverage=0.6, surface_energy=1.0, dihedral_half_angle=73.0
            ),
            theoretical_density_fraction=self.theoretical_density_fraction,
            grain_radius=self.grain_radius,
            surface_roughness=self.surface_roughness,
            emissivity=self.emissivity,
            energy_per_fission=self.energy_per_fission,
        )

    def add_thermal_material(self, problem, block, context):
        self._delegate().add_thermal_material(problem, block, context)

    def add_elasticity(self, problem, block, context):
        self._delegate().add_elasticity(problem, block, context)

    def add_eigenstrains(self, problem, block, context):
        return self._delegate().add_eigenstrains(problem, block, context)

    def creep_parameters(self, context):
        return self._delegate().creep_parameters(context)

    def fission_gas_model(self, num_elements, context):
        return self._delegate().fission_gas_model(num_elements, context)


# ---------------------------------------------------------------------------
# Cr2O3-doped UO2
# ---------------------------------------------------------------------------
#: Table 2.1 of INL/EXT-20-59969, identical to Table 3 of Cooper et al., J.
#: Nucl. Mater. 545 (2021) 152590 (case A and case B, read first-hand):
#: (T1 = T2 in K, dH1 in eV, dH2 in eV).
_DOPED_DIFFUSIVITY = {
    "best_estimate": (1773.0, 0.3198, -0.3345),
    "upper_limit": (1773.0, 0.3282, -0.6998),
    # CASL-U-2019-1870-000 Rev. 0, Eq. 16: the upper limit (oxygen potential
    # set by the Cr2O3/Cr equilibrium) used in the BISON validation of that
    # report.
    "casl_2019": (1673.0, 0.316, -0.684),
}


@dataclass
class DopedUO2Fuel(UO2Fuel):
    r"""UO2 doped with Cr2O3, which grows larger grains and speeds up the
    diffusion of fission gas in them.  Everything is that of
    :class:`UO2Fuel` except:

    * the grain radius, default 25 um (the doped rods of the Halden test
      IFA-677.1 had 22.5 and 28 um, Table 2.2 of INL/EXT-20-59969);
    * the intragranular diffusion coefficient of fission gas (Eqs. 2.1 to
      2.4 of INL/EXT-20-59969, Table 2.1):
      :math:`D = e^{-\Delta H_1/k_B (1/T - 1/T_1)} D_1 + e^{-\Delta H_2/k_B
      (1/T - 1/T_2)} D_2 + D_3` with
      :math:`D_1 = 7.6\times10^{-10} e^{-4.86\times10^{-19}/k_B T}`,
      :math:`D_2 = 5.64\times10^{-25} \sqrt{\dot F} e^{-1.91\times10^{-19}/k_B T}`,
      :math:`D_3 = 2\times10^{-40} \dot F` (m^2/s, the athermal term of
      :class:`BoothFissionGasRelease`; Cooper et al. used BISON's
      :math:`8\times10^{-40}`, which changes the Halden release by less than
      0.01 %), and ``diffusivity_case``
      choosing the best-estimate or the upper-limit parameters.  The factors
      are evaluated at :math:`\min(T, T_1)`: the source (Cooper et al. 2021,
      via INL/EXT-20-59969 and SSM 2021:20) gives no upper limit, and above
      1773 K the fit would make doping slow the diffusion down;
    * trapping and re-solution at intragranular bubbles as for UO2, and the
      same grain-face bubbles;
    * the total densification, default 0.1 % instead of 1 % (measured on the
      Halden rods, see the parameter).

    The conductivity is that of UO2: industrial doping (up to 0.1 wt%
    Cr2O3) changes it by no measurable amount (SSM 2021:20, Sect. 3.1.3).
    Creep follows MATPRO FCREEP with the large grain, whose G^-2 factor makes
    the diffusional creep of doped fuel 25 times slower than for 5 um
    grains.  Measurements show the opposite: at 1773 K and 45 MPa the creep
    rate of UO2 with 0.1 wt% Cr2O3 is about 5 times that of undoped UO2
    (Dugay et al. 1998, as tabulated in SSM 2021:20, Table 16).  No creep
    correlation for doped fuel exists (CASL-U-2019-1870, Sect. 2.9);
    ``creep_rate_factor`` lets a user apply such a single-condition factor,
    and is 1 by default.
    """

    grain_radius: float = parameter(
        25.0e-6, unit="m", description="Mean grain radius. Default 25 um (see the class)."
    )
    total_densification: float = parameter(
        0.001,
        description="Density change in a resintering test as a fraction of the theoretical "
        "density (see UO2Fuel). Default 0.001: the doped rods of the Halden test IFA-677.1 "
        "densified about 0.1 % (0.6 % for the undoped UO2 rods of the same test), and those of "
        "IFA-716.1 negligibly (CASL-U-2019-1870, Sect. 2.7.1, citing the Halden reports).",
    )
    diffusivity_case: str = parameter(
        "best_estimate",
        description="best_estimate (case A) or upper_limit (case B) of Table 2.1 of "
        "INL/EXT-20-59969, or casl_2019 (Eq. 16 of CASL-U-2019-1870, T1 = 1673 K, "
        "0.316 and -0.684 eV, the upper limit that report used). Default best_estimate.",
    )

    creep_rate_factor: float = parameter(
        1.0,
        description="Factor on the FCREEP creep rate. Default 1 (see the class for the "
        "measured doped/undoped ratio of about 5 at 1773 K and 45 MPa).",
    )

    material_name = "Cr2O3-doped UO2"

    def creep_parameters(self, context):
        params = super().creep_parameters(context)
        params["creep_rate_factor"] = self.creep_rate_factor
        return params

    def __post_init__(self):
        super().__post_init__()
        if self.diffusivity_case not in _DOPED_DIFFUSIVITY:
            raise ValueError(
                "DopedUO2Fuel: diffusivity_case must be best_estimate, upper_limit or "
                f"casl_2019, not '{self.diffusivity_case}'."
            )

    def diffusion_coefficient(self, temperature, fission_rate):
        """The intragranular diffusion coefficient, m^2/s (Eq. 2.1)."""
        T = np.asarray(temperature, dtype=float)
        F = np.maximum(np.asarray(fission_rate, dtype=float), 0.0)
        T1, dH1, dH2 = _DOPED_DIFFUSIVITY[self.diffusivity_case]
        kT = 1.380649e-23 * T
        d1 = 7.6e-10 * np.exp(-4.86e-19 / kT)
        d2 = 5.64e-25 * np.sqrt(F) * np.exp(-1.91e-19 / kT)
        d3 = ATHERMAL_COEFFICIENT * F
        Tf = np.minimum(T, T1)
        f1 = np.exp(-dH1 / BOLTZMANN_EV * (1 / Tf - 1 / T1))
        f2 = np.exp(-dH2 / BOLTZMANN_EV * (1 / Tf - 1 / T1))
        return f1 * d1 + f2 * d2 + d3

    def fission_gas_model(self, num_elements, context):
        from .gas import BoothFissionGasRelease

        if context.models.fission_gas_release != "booth":
            return None
        return BoothFissionGasRelease(
            num_elements,
            grain_radius=self.grain_radius,
            trapping_factor=context.models.trapping_factor,
            trapping=context.models.intragranular_trapping or "speight",
            diffusion_coefficient=self.diffusion_coefficient,
        )


def describe_material(material) -> dict:
    """The fields of a material, for reports."""
    return {f.name: getattr(material, f.name) for f in dataclasses.fields(material)}
