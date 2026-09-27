# SPDX-License-Identifier: LGPL-2.1-or-later
r"""Displacement damage: from neutron fluence and spectrum to dpa, and back.

Neutrons displace atoms from their lattice sites by transferring energy to
them in collisions and reactions.  The standard measure of this damage is the
number of *displacements per atom* (dpa).  Its calculation has three steps.

1. **Damage energy.**  A recoiling atom of energy :math:`E_R` loses part of
   its energy to the electrons of the lattice (ionisation), which does not
   displace atoms.  The rest, the damage energy :math:`T_d`, goes into atomic
   collisions.  Robinson's analytic form of Lindhard's partition theory, as
   used by NJOY, gives

   .. math::

      T_d = \frac{E_R}{1 + F_L\,(3.4008\,\varepsilon^{1/6} + 0.40244\,
      \varepsilon^{3/4} + \varepsilon)},\qquad \varepsilon = \frac{E_R}{E_L},

   with :math:`E_L = 30.724\,Z_R Z_L \sqrt{Z_R^{2/3} + Z_L^{2/3}}\,(A_R +
   A_L)/A_L` eV and :math:`F_L = 0.0793\,Z_R^{2/3} Z_L^{1/2} (A_R +
   A_L)^{3/2} / [(Z_R^{2/3} + Z_L^{2/3})^{3/4} A_R^{3/2} A_L^{1/2}]`, for a
   recoil of atomic number :math:`Z_R` and mass :math:`A_R` in a lattice of
   :math:`Z_L`, :math:`A_L` (A. C. Kahler in IAEA report INDC(NDS)-0648,
   2013, p. 35, after Robinson 1970 and Lindhard et al. 1963).

2. **Displacements per recoil.**  The NRT model (Norgett, Robinson and
   Torrens 1975) counts :math:`N_d = 0` below the threshold displacement
   energy :math:`E_d`, one up to :math:`2E_d/0.8`, and :math:`0.8\,T_d /
   (2E_d)` above.  The arc-dpa model of K. Nordlund et al. (Nat. Commun. 9
   (2018) 1084, Eqs. 2-4 and Table 1) multiplies the last branch by the
   cascade efficiency

   .. math::

      \xi(T_d) = \frac{1 - c}{(2E_d/0.8)^b}\,T_d^{\,b} + c,

   which accounts for the recombination of defects in the cascade (for iron
   :math:`b = -0.568`, :math:`c = 0.286`, :math:`E_d = 40` eV).

3. **Displacement cross section and dose.**  The displacement cross section
   :math:`\sigma_d(E)` is the number of displacements per atom per unit
   fluence of neutrons of energy :math:`E`.  The dose rate is
   :math:`\dot d = \int \sigma_d(E)\,\phi(E)\,dE`, and the dose per unit
   fluence above a threshold :math:`E_t` (1 MeV or 0.1 MeV by convention) is

   .. math::

      \frac{d}{\Phi(E > E_t)} = \frac{\int_0^\infty \sigma_d \phi\,dE}
      {\int_{E_t}^\infty \phi\,dE}.

The module gives :math:`\sigma_d` in two ways:

* :func:`nrt_cross_section` from the damage energy cross sections of
  ENDF/B-VIII.0 processed with NJOY-2016 (MF3 MT444, the mean damage energy
  of every reaction times its cross section), which are bundled for C, Al,
  Si, Ti, Cr, Fe, Ni, Zr, Nb, Mo, W and U: :math:`\sigma_d = 0.8\,
  (\sigma T_d)/(2 E_d)`.  This is the standard NRT-dpa cross section (as in
  ASTM E693 for iron).  With ``model="arc"`` the arc-dpa efficiency is
  applied at the mean damage energy of the reaction, which is exact only for
  a single recoil energy and is labelled an approximation;
* :func:`elastic_cross_section` for elastic scattering that is isotropic in
  the centre-of-mass frame, from a given elastic cross section.  The recoil
  energy is then uniform between zero and :math:`\Lambda E`, :math:`\Lambda =
  4A/(1 + A)^2`, and the average of :math:`N_d(T_d(E_R))` over it is exact for
  either model.  It is the right model for light elements and for neutron
  energies below the inelastic thresholds.

The group spectra are tables of flux per group over energy group bounds
(eV); see :class:`Spectrum`.  The threshold displacement energies and the
arc-dpa constants are those of Nordlund et al. (2018) Table 1 where it gives
them; for other elements they must be given.

Two conversions used as defaults elsewhere in the package are published
rules of thumb: 0.9 dpa per 1e25 n/m^2 in FeCrAl and 1 dpa per 1e25 n/m^2
in SiC (IAEA-TECDOC-1921).  :func:`dpa_per_fluence` computes the value for a
given spectrum instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

#: Atomic numbers and standard atomic weights of the bundled elements.
ELEMENTS = {
    "C": (6, 12.011),
    "Al": (13, 26.982),
    "Si": (14, 28.085),
    "Ti": (22, 47.867),
    "Cr": (24, 51.996),
    "Fe": (26, 55.845),
    "Ni": (28, 58.693),
    "Zr": (40, 91.224),
    "Nb": (41, 92.906),
    "Mo": (42, 95.95),
    "W": (74, 183.84),
    "U": (92, 238.03),
}

#: Nordlund et al. (2018) Table 1: E_d (eV), b and c of arc-dpa.
ARC_DPA = {
    "Fe": (40.0, -0.568, 0.286),
    "Cu": (33.0, -0.68, 0.16),
    "Ni": (39.0, -1.01, 0.23),
    "Pd": (41.0, -0.88, 0.15),
    "Pt": (42.0, -1.12, 0.11),
    "W": (70.0, -0.56, 0.12),
}

_DATA = Path(__file__).resolve().parent / "data" / "damage_energy_endfb8.csv"
_TABLE = None


def _table():
    global _TABLE
    if _TABLE is None:
        with open(_DATA) as f:
            lines = [line for line in f if not line.startswith("#")]
        names = lines[0].strip().split(",")
        data = np.loadtxt(lines[1:], delimiter=",", ndmin=2)
        _TABLE = {name: data[:, k] for k, name in enumerate(names)}
    return _TABLE


# ---------------------------------------------------------------------------
# Damage energy and displacements per recoil
# ---------------------------------------------------------------------------
def damage_energy(recoil_energy, recoil_Z, recoil_A, lattice_Z=None, lattice_A=None):
    """Damage energy T_d (eV) of a recoil of energy E_R (eV): Robinson's
    partition as written in INDC(NDS)-0648, p. 35.  The lattice defaults to
    the recoil's own element."""
    ZR, AR = float(recoil_Z), float(recoil_A)
    ZL = ZR if lattice_Z is None else float(lattice_Z)
    AL = AR if lattice_A is None else float(lattice_A)
    E = np.asarray(recoil_energy, dtype=float)
    s = ZR ** (2 / 3) + ZL ** (2 / 3)
    EL = 30.724 * ZR * ZL * np.sqrt(s) * (AR + AL) / AL
    FL = 0.0793 * ZR ** (2 / 3) * ZL**0.5 * (AR + AL) ** 1.5 / (s**0.75 * AR**1.5 * AL**0.5)
    eps = E / EL
    return E / (1.0 + FL * (3.4008 * eps ** (1 / 6) + 0.40244 * eps**0.75 + eps))


def nrt_displacements(damage, threshold):
    """NRT displacements for a damage energy (eV) and a threshold
    displacement energy E_d (eV)."""
    T = np.asarray(damage, dtype=float)
    Ed = float(threshold)
    return np.where(Ed > T, 0.0, np.where(2 * Ed / 0.8 > T, 1.0, 0.8 * T / (2 * Ed)))


def arc_efficiency(damage, threshold, b, c):
    """The arc-dpa efficiency xi(T_d) (Nordlund et al. 2018, Eq. 4), one at
    2 E_d / 0.8."""
    T = np.maximum(np.asarray(damage, dtype=float), 2 * threshold / 0.8)
    return (1.0 - c) / (2 * threshold / 0.8) ** b * T**b + c


def arc_displacements(damage, threshold, b, c):
    """arc-dpa displacements (Nordlund et al. 2018, Eq. 3)."""
    T = np.asarray(damage, dtype=float)
    nrt = nrt_displacements(T, threshold)
    return np.where(2 * threshold / 0.8 > T, nrt, nrt * arc_efficiency(T, threshold, b, c))


# ---------------------------------------------------------------------------
# Spectra
# ---------------------------------------------------------------------------
@dataclass
class Spectrum:
    """A neutron spectrum as the flux in each of a set of energy groups.

    ``bounds`` (eV, ascending, one more than the groups) and ``flux`` (any
    unit of flux per group, for instance n/(m^2 s)).  Only ratios matter for
    the dose per fluence; the dose rate has the unit of the flux times
    m^2 (the cross sections are converted from barns)."""

    bounds: np.ndarray
    flux: np.ndarray
    name: str = ""

    def __post_init__(self):
        self.bounds = np.asarray(self.bounds, dtype=float)
        self.flux = np.asarray(self.flux, dtype=float)
        if self.bounds.ndim != 1 or len(self.bounds) != len(self.flux) + 1:
            raise ValueError("Spectrum: bounds must have one more entry than flux.")
        if np.any(np.diff(self.bounds) <= 0):
            raise ValueError("Spectrum: bounds must increase.")
        if np.any(self.flux < 0):
            raise ValueError("Spectrum: the flux must not be negative.")

    def fluence_above(self, threshold: float) -> float:
        """The flux above an energy (eV), with the group that contains it
        split in proportion to lethargy."""
        lo, hi = self.bounds[:-1], self.bounds[1:]
        frac = np.clip(np.log(hi / np.maximum(threshold, lo)) / np.log(hi / lo), 0.0, 1.0)
        return float(np.sum(self.flux * frac))

    @classmethod
    def from_function(cls, flux_per_energy, bounds, name=""):
        """A group spectrum from a flux per unit energy phi(E) (a function of
        E in eV), integrated over each group."""
        bounds = np.asarray(bounds, dtype=float)
        flux = []
        for lo, hi in zip(bounds[:-1], bounds[1:]):
            E = np.geomspace(lo, hi, 33)
            flux.append(np.trapezoid(flux_per_energy(E), E))
        return cls(bounds, np.array(flux), name)


def watt_spectrum(a: float = 0.988e6, b: float = 2.249e-6):
    r"""The Watt form of a fission spectrum, :math:`\chi(E) \propto e^{-E/a}
    \sinh\sqrt{bE}` (E in eV).  The default a = 0.988 MeV and b = 2.249 per
    MeV are the constants usually quoted for the thermal fission of U-235;
    they were not checked against an evaluation here, and the function is
    meant for illustration and tests."""

    def phi(E):
        E = np.asarray(E, dtype=float)
        return np.exp(-E / a) * np.sinh(np.sqrt(b * E))

    return phi


# ---------------------------------------------------------------------------
# Displacement cross sections
# ---------------------------------------------------------------------------
@dataclass
class DisplacementCrossSection:
    """A displacement cross section on energy groups: ``bounds`` (eV) and
    ``values`` (barn, displacements per atom per unit fluence times 1e28)."""

    bounds: np.ndarray
    values: np.ndarray
    description: str = ""
    notes: list = field(default_factory=list)

    def at(self, energy):
        """The cross section (barn) at energies (eV), constant in each group."""
        E = np.asarray(energy, dtype=float)
        index = np.clip(np.searchsorted(self.bounds, E, side="right") - 1, 0, len(self.values) - 1)
        return self.values[index]

    def on(self, spectrum: Spectrum) -> np.ndarray:
        """Group-averaged values on a spectrum's groups, weighted per unit
        lethargy within each spectrum group."""
        out = []
        for lo, hi in zip(spectrum.bounds[:-1], spectrum.bounds[1:]):
            E = np.geomspace(lo, hi, 65)
            u = np.log(E)
            out.append(np.trapezoid(self.at(E), u) / (u[-1] - u[0]))
        return np.array(out)


def nrt_cross_section(
    element: str,
    threshold: float | None = None,
    model: str = "nrt",
    arc_constants: tuple | None = None,
) -> DisplacementCrossSection:
    """The NRT (or approximate arc-dpa) displacement cross section of a
    bundled element from its ENDF/B-VIII.0 damage energy cross section.

    ``threshold`` is E_d (eV); it defaults to Nordlund et al.'s value where
    Table 1 gives one and must be given otherwise.  With ``model="arc"`` the
    arc-dpa efficiency is applied at the mean damage energy per reaction,
    :math:`\\bar T_d = (\\sigma T_d)/\\sigma_{total}`, an approximation."""
    if element not in ELEMENTS:
        raise ValueError(f"dpa: no bundled data for {element}. Use one of {', '.join(ELEMENTS)}.")
    if threshold is None:
        if element not in ARC_DPA:
            raise ValueError(f"dpa: give the threshold displacement energy of {element} (eV).")
        threshold = ARC_DPA[element][0]
    t = _table()
    bounds = np.concatenate([t["lower_eV"], t["upper_eV"][-1:]])
    damage = t[f"{element}_damage_eV_b"]
    values = 0.8 * damage / (2.0 * threshold)
    notes = [
        "damage energy: ENDF/B-VIII.0 MT444 (NJOY-2016, IAEA NDS), 50 groups per decade",
        f"E_d = {threshold:g} eV",
    ]
    if model == "arc":
        if arc_constants is None:
            if element not in ARC_DPA:
                raise ValueError(f"dpa: give the arc-dpa constants (b, c) of {element}.")
            arc_constants = ARC_DPA[element][1:]
        b, c = arc_constants
        total = t[f"{element}_total_b"]
        mean = np.where(total > 0, damage / np.maximum(total, 1e-300), 0.0)
        values = values * arc_efficiency(mean, threshold, b, c)
        notes.append("arc-dpa efficiency at the mean damage energy per reaction (an approximation)")
    elif model != "nrt":
        raise ValueError("dpa: model must be nrt or arc.")
    return DisplacementCrossSection(bounds, values, f"{element} {model}-dpa", notes)


def elastic_cross_section(
    energy,
    elastic,
    Z: float,
    A: float,
    threshold: float,
    model: str = "nrt",
    arc_constants: tuple | None = None,
    points: int = 400,
):
    """Displacement cross section (barn) for elastic scattering that is
    isotropic in the centre-of-mass frame, at neutron energies ``energy``
    (eV) with the elastic cross section ``elastic`` (barn) there: the recoil
    energy is uniform on [0, Lambda E] with Lambda = 4A/(1 + A)^2, and the
    displacements are averaged over it exactly (a Gauss-Legendre rule on
    ``points`` points)."""
    E = np.atleast_1d(np.asarray(energy, dtype=float))
    lam = 4.0 * A / (1.0 + A) ** 2
    x, w = np.polynomial.legendre.leggauss(points)
    x, w = 0.5 * (x + 1.0), 0.5 * w
    recoil = lam * E[:, None] * x[None, :]
    Td = damage_energy(recoil, Z, A)
    if model == "nrt":
        n = nrt_displacements(Td, threshold)
    elif model == "arc":
        b, c = arc_constants
        n = arc_displacements(Td, threshold, b, c)
    else:
        raise ValueError("dpa: model must be nrt or arc.")
    return np.asarray(elastic, dtype=float) * (n @ w)


# ---------------------------------------------------------------------------
# Dose
# ---------------------------------------------------------------------------
BARN = 1e-28


def dpa_rate(cross_section: DisplacementCrossSection, spectrum: Spectrum) -> float:
    """dpa per unit time for a spectrum given as flux per group in n/(m^2 s)."""
    return float(np.sum(cross_section.on(spectrum) * spectrum.flux) * BARN)


def spectrum_averaged(cross_section: DisplacementCrossSection, spectrum: Spectrum) -> float:
    """The flux-averaged displacement cross section, barn."""
    return float(np.sum(cross_section.on(spectrum) * spectrum.flux) / np.sum(spectrum.flux))


def dpa_per_fluence(
    cross_section: DisplacementCrossSection, spectrum: Spectrum, threshold: float = 1.0e6
) -> float:
    """dpa per unit fluence (n/m^2) above ``threshold`` (eV, default 1 MeV),
    the conversion factor between a fast fluence and a dose."""
    fast = spectrum.fluence_above(threshold)
    if fast <= 0:
        raise ValueError("dpa: the spectrum has no flux above the threshold.")
    return float(np.sum(cross_section.on(spectrum) * spectrum.flux) * BARN / fast)


def dose_from_fluence(fluence, factor: float):
    """dpa from a fast fluence (n/m^2) with a dpa-per-fluence factor."""
    return np.asarray(fluence, dtype=float) * factor


def fluence_from_dose(dose, factor: float):
    """The fast fluence (n/m^2) that gives a dose (dpa)."""
    return np.asarray(dose, dtype=float) / factor


def compound_cross_section(parts: dict) -> DisplacementCrossSection:
    """The atom-fraction-weighted cross section of a compound or alloy, from
    ``{fraction: DisplacementCrossSection}`` pairs given as a dict of
    ``name: (atom_fraction, cross_section)`` on the same groups.  Each
    element's cross section counts displacements of its own atoms by its own
    recoils only, the usual first approximation."""
    fractions = np.array([f for f, _ in parts.values()])
    if not np.isclose(fractions.sum(), 1.0):
        raise ValueError("dpa: the atom fractions must sum to one.")
    first = next(iter(parts.values()))[1]
    values = sum(f * xs.values for f, xs in parts.values())
    return DisplacementCrossSection(
        first.bounds, values, " + ".join(parts), ["atom-fraction weighted"]
    )
