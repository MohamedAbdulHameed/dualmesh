# SPDX-License-Identifier: LGPL-2.1-or-later
r"""Thermal conductivity of gases at low density.

The pure gases are helium, argon, krypton, xenon, hydrogen and nitrogen.

* Helium, argon, krypton and xenon: fits to the low-density conductivities of
  Tables 1, 3, 4 and 5 of Kestin et al., J. Phys. Chem. Ref. Data 13 (1984)
  229-303, over 200 K to 2,273 K.  ``verification/correlations/kestin1984_gas_fit.py``
  reproduces the fit.
* Hydrogen and nitrogen: :math:`k = A T^B`, the MATPRO GASCON fits,
  NUREG/CR-6150 Vol. 4 (1995), Table 13-2.
* Mixtures: Eqs. (12) and (13) of R. S. Brokaw, J. Chem. Phys. 29 (1958)
  391-397.

The same functions give the gas conductance of ``gas_gap_heat_transfer``.
Units are SI: K, Pa, m and W/(m K).
"""

from __future__ import annotations

from .._core import gas as _gas

#: The names of the gases, as a composition takes them.
GASES = ("helium", "argon", "krypton", "xenon", "hydrogen", "nitrogen")


def thermal_conductivity(composition: dict[str, float], temperature):
    """Return the thermal conductivity of a gas mixture, W/(m K).

    ``composition`` maps gas names (:data:`GASES`) to mole fractions, which
    are normalized to a sum of one.  ``temperature`` (K) is a float or an
    array."""
    return _gas.thermal_conductivity(dict(composition), temperature)


def temperature_jump_distance(composition: dict[str, float], temperature: float, pressure: float, accommodation_coefficient: float | None = None) -> float:
    """Return the sum of the temperature jump distances at the two walls of a
    gap, m, by the equation of Kennard as Lanning and Hann (BNWL-1894, 1975,
    Appendix B) write it.

    Without an ``accommodation_coefficient``, the fits of Ullman et al. for
    helium and xenon are interpolated by the molar mass of the mixture."""
    return _gas.temperature_jump_distance(dict(composition), temperature, pressure, accommodation_coefficient)


__all__ = ["GASES", "temperature_jump_distance", "thermal_conductivity"]
