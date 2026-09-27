# SPDX-License-Identifier: LGPL-2.1-or-later
r"""The irradiation fields of a rod: everything that follows from the power
history alone, not from the thermal or mechanical state of the rod.

A rod is driven by its linear heat rate :math:`q'(z, t)`, the heat generated
per unit length (W/m): the rod average of the :class:`PowerHistory` times its
axial profile.  From it:

* the volumetric heat generation in the fuel,
  :math:`q'''(r, z, t) = q'(z, t)\, f(r) / A`, where :math:`A` is the fuel
  cross-section and :math:`f(r)` the radial profile normalised to an area
  average of one;
* the fission rate density :math:`\dot F = q''' / E_f`, with :math:`E_f` the
  energy per fission;
* the local burnup in FIMA, the fraction of the initial heavy-metal atoms that
  have fissioned, :math:`\beta = \int_0^t \dot F\, dt' / n_{HM}`, where
  :math:`n_{HM}` is the number of heavy-metal atoms per unit volume of fuel;
  and its pellet average (without the radial profile);
* the fast neutron flux (E > 1 MeV), taken proportional to the linear heat
  rate, :math:`\phi = c_\phi\, q'`, and the fast fluence
  :math:`\int_0^t \phi\, dt'`.

The fields are tabulated on a grid of :math:`(r, z, t)` and handed to the
solver as :class:`dualmesh.CylinderTableFunction` objects, which every
material of the rod evaluates at any point and time.  Burnups are passed in
FIMA, and the objects that take them are told so (``burnup_unit="FIMA"``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .. import _core
from .specification import RodGeometry

#: Names of the fields, which are also the names of the functions added to the
#: problem and of the parameters of the objects that read them.
FIELD_NAMES = (
    "power_density",
    "fission_rate",
    "burnup",
    "pellet_average_burnup",
    "linear_heat_rate",
    "fast_neutron_flux",
    "fast_neutron_fluence",
)


def trapezoid(values, points) -> float:
    """The trapezoidal-rule integral of sampled values."""
    values, points = np.asarray(values, dtype=float), np.asarray(points, dtype=float)
    return float(np.sum(0.5 * (values[1:] + values[:-1]) * np.diff(points)))


class TimeHistory:
    """A power history in time: the rod-average linear heat rate at a set of
    times, linear in between.  The rod builds one from a
    :class:`PowerHistory`, converting a history given in burnup."""

    def __init__(self, time, linear_heat_rate, axial_profile=None, radial_profile=None):
        self.time = np.asarray(time, dtype=float)
        self.linear_heat_rate = np.asarray(linear_heat_rate, dtype=float)
        self.axial_profile = axial_profile
        self.radial_profile = radial_profile
        # The integral of q' up to each point (exact: q' is linear in between).
        self._energy = np.concatenate(
            [
                [0.0],
                np.cumsum(
                    0.5
                    * np.diff(self.time)
                    * (self.linear_heat_rate[1:] + self.linear_heat_rate[:-1])
                ),
            ]
        )

    @classmethod
    def from_burnup(
        cls,
        burnup,
        linear_heat_rate,
        fima_per_joule_per_metre: float,
        axial_profile=None,
        radial_profile=None,
    ) -> TimeHistory:
        """Times for a history given in rod-average burnup (FIMA).

        The history gives the burnup :math:`B_i` and the linear heat rate
        :math:`q'_i` at each point.  Taking :math:`q'` linear in time between
        two points, the burnup gained is
        :math:`B_{i+1} - B_i = \\kappa (q'_i + q'_{i+1}) (t_{i+1} - t_i) / 2`,
        where :math:`\\kappa` (``fima_per_joule_per_metre``) is the burnup per
        unit energy released per unit length of rod.  This fixes the times,
        and the burnup is exact at every point of the history.  The history
        starts at time zero."""
        burnup = np.asarray(burnup, dtype=float)
        q = np.asarray(linear_heat_rate, dtype=float)
        dt = 2.0 * np.diff(burnup) / (fima_per_joule_per_metre * (q[1:] + q[:-1]))
        return cls(np.concatenate([[0.0], np.cumsum(dt)]), q, axial_profile, radial_profile)

    def linear_heat_rate_at(self, t) -> np.ndarray:
        """The rod-average linear heat rate at times t, W/m."""
        return np.interp(t, self.time, self.linear_heat_rate)

    def energy_at(self, t) -> np.ndarray:
        """The time integral of the rod-average linear heat rate up to t, J/m,
        exact for the piecewise-linear history (constant beyond its ends)."""
        t = np.clip(np.atleast_1d(np.asarray(t, dtype=float)), self.time[0], self.time[-1])
        i = np.clip(np.searchsorted(self.time, t, side="right") - 1, 0, len(self.time) - 2)
        dt = t - self.time[i]
        q0 = self.linear_heat_rate[i]
        slope = (self.linear_heat_rate[i + 1] - q0) / (self.time[i + 1] - self.time[i])
        return self._energy[i] + q0 * dt + 0.5 * slope * dt * dt

    def axial(self, z, fuel_stack_height: float) -> np.ndarray:
        """The axial profile at positions z, as given (normalised by the rod)."""
        z = np.asarray(z, dtype=float)
        if self.axial_profile is None:
            return np.ones_like(z)
        return np.asarray(self.axial_profile(z / fuel_stack_height), dtype=float)

    def radial(self, r, geometry: RodGeometry) -> np.ndarray:
        """The radial profile at radii r, normalised to an area average of one
        over the pellet cross-section (the annulus of an annular pellet)."""
        r = np.asarray(r, dtype=float)
        if self.radial_profile is None:
            return np.ones_like(r)
        inner, outer = geometry.pellet_inner_radius, geometry.pellet_outer_radius
        grid = np.linspace(inner, outer, 801)
        values = np.asarray(self.radial_profile(grid / outer), dtype=float)
        average = trapezoid(values * 2 * grid, grid) / (outer**2 - inner**2)
        return np.asarray(self.radial_profile(r / outer), dtype=float) / average


def normalized_axial_profile(profile, fuel_stack_height: float):
    """The axial profile f(z / H) scaled to an average of one over the stack,
    or None for a flat one."""
    if profile is None:
        return None
    s = np.linspace(0.0, 1.0, 2001)
    average = trapezoid(np.asarray(profile(s), dtype=float), s)
    if average <= 0:
        raise ValueError("PowerHistory: the axial profile must have a positive average.")
    return lambda x: np.asarray(profile(x), dtype=float) / average


@dataclass
class IrradiationFields:
    """The irradiation fields of one problem of a rod, as solver functions,
    with the grids they are tabulated on."""

    power_density: object
    fission_rate: object
    burnup: object
    pellet_average_burnup: object
    linear_heat_rate: object
    fast_neutron_flux: object
    fast_neutron_fluence: object
    grids: dict = field(default_factory=dict)

    def items(self):
        return [(name, getattr(self, name)) for name in FIELD_NAMES]


def irradiation_fields(
    history: TimeHistory,
    geometry: RodGeometry,
    heavy_metal_atom_density: float,
    energy_per_fission: float,
    fast_neutron_flux_per_linear_heat_rate: float,
    formulation: str,
    slice_axial_position: float = 0.0,
    num_radial_points: int = 41,
    num_axial_points: int = 41,
    time_refinement: int = 8,
) -> IrradiationFields:
    """Tabulate the irradiation fields of ``history`` for one problem.

    ``formulation`` is ``"axisymmetric_1d"`` (a radial slice at
    ``slice_axial_position``), ``"axisymmetric"`` or ``"three_dimensional"``,
    as for :class:`dualmesh.CylinderTableFunction`.  Each interval of the
    history is split ``time_refinement`` times: the burnup is quadratic in time
    within an interval where the power ramps, and the table interpolates it
    linearly, so the split keeps the interpolation error below a per cent of
    the burnup gained in one interval (1/(8 n^2) of it, for n splits).
    """
    t = np.unique(
        np.concatenate(
            [
                np.linspace(history.time[i], history.time[i + 1], time_refinement + 1)
                for i in range(len(history.time) - 1)
            ]
        )
    )
    r = np.linspace(geometry.pellet_inner_radius, geometry.pellet_outer_radius, num_radial_points)
    if formulation == "axisymmetric_1d":
        z = np.array([slice_axial_position])
    else:
        z = np.linspace(0.0, geometry.fuel_stack_height, num_axial_points)
    radial = history.radial(r, geometry)
    axial = history.axial(z, geometry.fuel_stack_height)
    q = history.linear_heat_rate_at(t)
    energy = history.energy_at(t)
    area = geometry.fuel_cross_section
    per_fission = 1.0 / (area * energy_per_fission)
    # Arrays [t, z, r] with r fastest, the order of the table.
    shape = (len(t), len(z), len(r))
    linear = np.broadcast_to(q[:, None, None] * axial[None, :, None], shape)
    power_density = linear * radial[None, None, :] / area
    energy_zr = np.broadcast_to(energy[:, None, None] * axial[None, :, None], shape)
    average_burnup = energy_zr * per_fission / heavy_metal_atom_density
    burnup = average_burnup * radial[None, None, :]

    def table(values):
        return _core.CylinderTableFunction(
            list(r),
            list(z),
            list(t),
            list(np.ascontiguousarray(values).ravel()),
            formulation,
            slice_axial_position,
        )

    return IrradiationFields(
        power_density=table(power_density),
        fission_rate=table(power_density / energy_per_fission),
        burnup=table(burnup),
        pellet_average_burnup=table(average_burnup),
        linear_heat_rate=table(linear),
        fast_neutron_flux=table(fast_neutron_flux_per_linear_heat_rate * linear),
        fast_neutron_fluence=table(fast_neutron_flux_per_linear_heat_rate * energy_zr),
        grids={"radii": r, "axial_positions": z, "times": t},
    )
