# SPDX-License-Identifier: LGPL-2.1-or-later
r"""Stiffness coefficients of functionally graded beams and plates.

A functionally graded material (FGM) varies continuously through the
thickness.  Following Reddy, the modulus follows the power law

.. math::

    E(z) = (E_1 - E_2) \left( \frac{1}{2} + \frac{z}{h} \right)^{n} + E_2 ,
    \qquad -\frac{h}{2} \le z \le \frac{h}{2},

so that :math:`E(h/2) = E_1` (top) and :math:`E(-h/2) = E_2` (bottom); the
power-law index :math:`n = 0` gives a homogeneous beam with :math:`E = E_1`.

The stress resultants of a beam of width :math:`b` use

.. math::

    (A_{xx}, B_{xx}, D_{xx}) = b \int_{-h/2}^{h/2} E(z) (1, z, z^2) \, dz ,
    \qquad
    S_{xz} = \frac{K_s}{2(1+\nu)} b \int_{-h/2}^{h/2} E(z) \, dz ,

with :math:`K_s = 5/6` the shear correction factor of a rectangular section.
For plates every coefficient is divided by :math:`1 - \nu^2` (plane-stress
reduced stiffness) and :math:`b = 1` is taken per unit width.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BeamStiffness:
    """Extensional, coupling, bending, and shear stiffnesses of a beam."""

    extensional: float  # A_xx
    coupling: float  # B_xx
    bending: float  # D_xx
    shear: float  # S_xz

    @property
    def reduced_bending(self) -> float:
        r"""The reduced bending stiffness :math:`D^{*} = D_{xx} A_{xx} - B_{xx}^2`."""
        return self.bending * self.extensional - self.coupling**2

    @property
    def effective_extensional(self) -> float:
        r"""Reddy's :math:`\bar{A}_{xx} = D^{*} / D_{xx}` (mixed formulations)."""
        return self.reduced_bending / self.bending

    @property
    def effective_coupling(self) -> float:
        r"""Reddy's :math:`\bar{B}_{xx} = B_{xx} / D_{xx}` (mixed formulations)."""
        return self.coupling / self.bending


def beam_stiffness(
    modulus_top: float,
    modulus_bottom: float,
    power_law_index: float,
    height: float,
    width: float = 1.0,
    poisson_ratio: float = 0.3,
    shear_correction_factor: float = 5.0 / 6.0,
    plate: bool = False,
) -> BeamStiffness:
    r"""Closed-form stiffnesses of a power-law functionally graded section.

    ``modulus_top`` is :math:`E_1` (at :math:`z = +h/2`) and ``modulus_bottom``
    is :math:`E_2` (at :math:`z = -h/2`).  With ``plate=True`` the coefficients
    are divided by :math:`1 - \nu^2`, as required by the plate theories.
    """
    n = float(power_law_index)
    if n < 0:
        raise ValueError("power_law_index must be non-negative")
    h, b = float(height), float(width)
    e1, e2 = float(modulus_top), float(modulus_bottom)
    if e2 == 0.0:
        raise ValueError("modulus_bottom must be non-zero")
    m = e1 / e2
    extensional = e2 * b * h * (m + n) / (1 + n)
    coupling = e2 * b * h**2 * (m - 1) * n / (2 * (1 + n) * (2 + n))
    bending = (
        e2
        * b
        * h**3
        * (3 * m * (2 + n + n**2) + 8 * n + 3 * n**2 + n**3)
        / (12 * (1 + n) * (2 + n) * (3 + n))
    )
    # The shear stiffness uses the shear modulus G = E / (2 (1 + nu)) and so
    # is not divided by (1 - nu^2), even for plates.
    shear = shear_correction_factor * extensional / (2 * (1 + poisson_ratio))
    factor = 1.0 / (1.0 - poisson_ratio**2) if plate else 1.0
    extensional *= factor
    coupling *= factor
    bending *= factor
    return BeamStiffness(extensional, coupling, bending, shear)


def modulus(
    z: float, modulus_top: float, modulus_bottom: float, power_law_index: float, height: float
) -> float:
    """The through-thickness modulus :math:`E(z)` of the power-law profile."""
    return (modulus_top - modulus_bottom) * (0.5 + z / height) ** power_law_index + modulus_bottom


def stiffness_by_quadrature(
    modulus_top: float,
    modulus_bottom: float,
    power_law_index: float,
    height: float,
    width: float = 1.0,
    num_points: int = 400,
) -> tuple[float, float, float]:
    """``(A, B, D)`` obtained by numerical integration (used to check the formulas)."""
    import numpy as np

    z, w = np.polynomial.legendre.leggauss(num_points)
    z = z * height / 2
    w = w * height / 2
    e = np.array([modulus(zi, modulus_top, modulus_bottom, power_law_index, height) for zi in z])
    return (
        float(width * np.sum(w * e)),
        float(width * np.sum(w * e * z)),
        float(width * np.sum(w * e * z**2)),
    )
