# SPDX-License-Identifier: LGPL-2.1-or-later
"""Properties of materials: functions of the state that return SI values.

* :mod:`dualmesh.materials.water`: compressed liquid water (IAPWS-IF97 region
  1, with the IAPWS viscosity and thermal conductivity).
* :mod:`dualmesh.materials.gas`: thermal conductivity of helium, argon,
  krypton, xenon, hydrogen and nitrogen at low density, and of their mixtures.
"""

from __future__ import annotations

from . import gas, water

__all__ = ["gas", "water"]
