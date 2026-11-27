# SPDX-License-Identifier: LGPL-2.1-or-later
"""The solid_mechanics physics.  One class, :class:`SolidMechanics`, holds the parameters
and the public methods, and delegates to the modules of this package:

- ``mechanics``: the class, its parameters and their checks.
- ``continuum``: the material law, the stress and its divergence, generalized plane strain,
  phase-field fracture, the crack-tip integrals and the manufactured solutions.
- ``structures``: the plate, beam and circular plate sections and the truss and shell members.
- ``interfaces``: contact, cohesive zones and rollers.
- ``rotations``: the rotation and the torque of a boundary about an axis.
- ``inertia``: the checks of the damping and the density, the initial state, the inertia and
  the Rayleigh damping.
- ``anisotropy``: the orthotropic stiffness, the Bond rotation and Hill's matrix.
"""

from .continuum import HYPERELASTIC
from .mechanics import FORMULATIONS, SolidMechanics
from .structures import MEMBERS, STRUCTURAL, THEORIES

__all__ = ["FORMULATIONS", "HYPERELASTIC", "MEMBERS", "STRUCTURAL", "THEORIES", "SolidMechanics"]
