# SPDX-License-Identifier: LGPL-2.1-or-later
r"""Physical constants and burnup units.

Burnup measures how much of the fuel has been used.  Two families of units are
in use, and both are accepted everywhere in :mod:`dualmesh.fuel`:

* **FIMA**, fissions per initial heavy-metal atom: a fraction of the atoms,
  natural for the physics (the fission rate integrates to it);
* **energy per mass of initial heavy metal**, natural for reactor operation:
  MWd/kgHM (megawatt-days per kilogram, equal to GWd/tHM), MWd/tHM and
  MWh/kgHM (used for CANDU fuel).  For uranium fuel "HM" and "U" are the same
  mass, so MWd/kgU is accepted as a synonym of MWd/kgHM.

They are related by the energy released per fission :math:`E_f` and the molar
mass :math:`M_{HM}` of the heavy metal:

.. math::

   1\ \mathrm{FIMA} = \frac{E_f N_A}{M_{HM}}
   = \frac{3.20435 \times 10^{-11}\ \mathrm{J} \times 6.02214 \times 10^{23}\
     \mathrm{mol^{-1}}}{0.238029\ \mathrm{kg\,mol^{-1}}}
   = 938.3\ \mathrm{MWd/kgHM}

with the default :math:`E_f = 200` MeV and natural uranium, which is the
conversion of the FRAPCON-4.0 material property report.  The conversion
follows the fuel's own enrichment when a fuel object computes it
(:attr:`dualmesh.fuel.UO2Fuel.heavy_metal_molar_mass`).
"""

from __future__ import annotations

AVOGADRO = 6.02214076e23
"""Avogadro's number, 1/mol."""
JOULES_PER_MEV = 1.602176634e-13
"""Joules in one MeV."""
JOULES_PER_MEGAWATT_DAY = 8.64e10
"""Joules in one megawatt-day."""
GAS_CONSTANT = 8.314462618
"""The molar gas constant, J/(mol K)."""
DEFAULT_ENERGY_PER_FISSION = 200.0 * JOULES_PER_MEV
"""200 MeV per fission, J: the recoverable energy commonly used for LWR fuel."""
NATURAL_URANIUM_MOLAR_MASS = 0.238029
"""kg/mol."""

#: Burnup units, as the factor that converts a value in the unit to MWd/kgHM.
#: FIMA is handled separately because its factor depends on the fuel.
ENERGY_UNITS = {
    "MWd/kgHM": 1.0,
    "MWd/kgU": 1.0,
    "GWd/tHM": 1.0,
    "GWd/tU": 1.0,
    "MWd/tHM": 1.0e-3,
    "MWd/tU": 1.0e-3,
    "MWh/kgHM": 1.0 / 24.0,
    "MWh/kgU": 1.0 / 24.0,
}
BURNUP_UNITS = ("FIMA",) + tuple(ENERGY_UNITS)


def check_burnup_unit(unit: str) -> str:
    """Return ``unit`` if it is a known burnup unit, otherwise raise an error
    that lists the known ones."""
    if unit not in BURNUP_UNITS:
        raise ValueError(f"Unknown burnup unit '{unit}'. Use one of: {', '.join(BURNUP_UNITS)}.")
    return unit


def mwd_per_kg_per_fima(
    energy_per_fission: float = DEFAULT_ENERGY_PER_FISSION,
    heavy_metal_molar_mass: float = NATURAL_URANIUM_MOLAR_MASS,
) -> float:
    """MWd/kgHM in one FIMA: :math:`E_f N_A / M_{HM}`."""
    return energy_per_fission * AVOGADRO / heavy_metal_molar_mass / JOULES_PER_MEGAWATT_DAY


class BurnupConverter:
    """Converts burnups between the units of :data:`BURNUP_UNITS` for one
    fuel, whose energy per fission and heavy-metal molar mass fix the FIMA
    conversion."""

    def __init__(
        self,
        energy_per_fission: float = DEFAULT_ENERGY_PER_FISSION,
        heavy_metal_molar_mass: float = NATURAL_URANIUM_MOLAR_MASS,
    ):
        self.energy_per_fission = float(energy_per_fission)
        self.heavy_metal_molar_mass = float(heavy_metal_molar_mass)
        self.mwd_per_kg_per_fima = mwd_per_kg_per_fima(
            self.energy_per_fission, self.heavy_metal_molar_mass
        )

    def to_fima(self, value, unit: str):
        """A burnup in ``unit``, in FIMA."""
        check_burnup_unit(unit)
        if unit == "FIMA":
            return value
        return value * ENERGY_UNITS[unit] / self.mwd_per_kg_per_fima

    def from_fima(self, fima, unit: str):
        """A burnup in FIMA, in ``unit``."""
        check_burnup_unit(unit)
        if unit == "FIMA":
            return fima
        return fima * self.mwd_per_kg_per_fima / ENERGY_UNITS[unit]

    def convert(self, value, from_unit: str, to_unit: str):
        return self.from_fima(self.to_fima(value, from_unit), to_unit)
