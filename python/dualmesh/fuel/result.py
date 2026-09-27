# SPDX-License-Identifier: LGPL-2.1-or-later
"""The result of a rod calculation.

:class:`RodResult` holds one entry per output time.  Quantities of the whole
rod (the gas pressure, the fission gas release) are arrays over time;
quantities that vary along the rod are arrays ``[time, axial position]``, at
the axial positions :attr:`RodResult.axial_positions`.  Burnups are in the
unit chosen in :class:`RodOutput` (:attr:`RodResult.burnup_unit`), and
:meth:`RodResult.burnup_in` gives them in any other unit.

For post-processing, :meth:`RodResult.write_csv` writes two tidy CSV files,
each with the header followed by a comment line of units, which pandas and numpy
read directly::

    pandas.read_csv("rod_history.csv", comment="#")
    numpy.genfromtxt("rod_history.csv", delimiter=",", names=True, comments="#")
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field

import numpy as np

from .units import BurnupConverter

SECONDS_PER_DAY = 86400.0

#: Rod quantities: name, unit, description.
ROD_QUANTITIES = (
    ("time", "s", "Time."),
    ("time_days", "d", "Time in days."),
    ("rod_average_linear_heat_rate", "W/m", "Rod-average linear heat rate."),
    ("rod_average_burnup", "burnup_unit", "Rod-average burnup."),
    ("gas_pressure", "Pa", "Pressure of the rod's gas."),
    ("fission_gas_release", "-", "Fraction of the fission gas produced that has been released."),
    ("xenon_fraction", "-", "Mole fraction of xenon in the rod's gas."),
    ("helium_fraction", "-", "Mole fraction of helium in the rod's gas."),
    ("gas_amount", "mol", "Amount of gas in the rod: the fill gas plus the released gas."),
    ("plenum_temperature", "K", "Temperature of the gas in the plenum."),
    ("max_fuel_centerline_temperature", "K", "Largest fuel centerline temperature."),
    ("min_gap_width", "m", "Smallest gap width along the rod (negative: contact)."),
    ("nonlinear_iterations", "-", "Newton iterations since the previous output time."),
)

#: Quantities along the rod: name, unit, description.
AXIAL_QUANTITIES = (
    ("linear_heat_rate", "W/m", "Local linear heat rate."),
    ("burnup", "burnup_unit", "Pellet-average burnup at the axial position."),
    (
        "fuel_centerline_temperature",
        "K",
        "Fuel temperature on the axis, or at the bore of an annular pellet.",
    ),
    ("fuel_surface_temperature", "K", "Fuel temperature at the pellet surface."),
    ("clad_inner_temperature", "K", "Cladding inner surface temperature."),
    ("clad_outer_temperature", "K", "Cladding outer surface temperature."),
    ("coolant_temperature", "K", "Coolant bulk temperature (or the prescribed wall temperature)."),
    ("gap_width", "m", "Radial gap between the displaced pellet and cladding surfaces."),
    ("fuel_surface_displacement", "m", "Radial displacement of the pellet surface."),
    ("clad_outer_displacement", "m", "Radial displacement of the cladding outer surface."),
    (
        "clad_hoop_strain",
        "-",
        "Total hoop strain at the cladding outer surface, u_r / r (elastic, thermal, creep and "
        "growth together).",
    ),
)


@dataclass
class RodResult:
    """What a rod calculation reports; see the module documentation."""

    burnup_unit: str
    converter: BurnupConverter
    axial_positions: np.ndarray
    rod: dict = field(default_factory=dict)
    axial: dict = field(default_factory=dict)
    input_report: str = ""

    def __post_init__(self):
        for name, _, _ in ROD_QUANTITIES:
            self.rod.setdefault(name, [])
        for name, _, _ in AXIAL_QUANTITIES:
            self.axial.setdefault(name, [])

    # ---- access ---------------------------------------------------------------
    def __getattr__(self, name):
        # Attribute access to every quantity: result.gas_pressure, result.gap_width.
        rod, axial = self.__dict__.get("rod", {}), self.__dict__.get("axial", {})
        if name in rod:
            return np.asarray(rod[name], dtype=float)
        if name in axial:
            return np.asarray(axial[name], dtype=float)
        raise AttributeError(
            f"RodResult has no quantity '{name}'. Use one of: {', '.join(self.names())}."
        )

    def names(self) -> list[str]:
        return [n for n, _, _ in ROD_QUANTITIES] + [n for n, _, _ in AXIAL_QUANTITIES]

    def unit(self, name: str) -> str:
        for n, unit, _ in ROD_QUANTITIES + AXIAL_QUANTITIES:
            if n == name:
                return self.burnup_unit if unit == "burnup_unit" else unit
        raise KeyError(name)

    def burnup_in(self, unit: str, rod_average: bool = False) -> np.ndarray:
        """The burnups (``[time, axial position]``, or the rod average over
        time) in another unit."""
        values = self.rod_average_burnup if rod_average else self.burnup
        return self.converter.convert(values, self.burnup_unit, unit)

    def as_arrays(self) -> dict:
        """Every quantity as a numpy array, keyed by name."""
        out = {"axial_positions": np.asarray(self.axial_positions)}
        out.update({n: np.asarray(v, dtype=float) for n, v in self.rod.items()})
        out.update({n: np.asarray(v, dtype=float) for n, v in self.axial.items()})
        return out

    # ---- text -----------------------------------------------------------------
    def summary(self) -> str:
        """The peak values and the end state, as aligned text."""
        from ..console import table

        T = self.fuel_centerline_temperature
        i, j = np.unravel_index(np.argmax(T), T.shape)
        gap = self.gap_width
        rows = [
            ["end time", self.time[-1] / SECONDS_PER_DAY, "d"],
            ["rod-average burnup at the end", self.rod_average_burnup[-1], self.burnup_unit],
            ["peak fuel centerline temperature", T[i, j], "K"],
            ["  at time", self.time[i] / SECONDS_PER_DAY, "d"],
            ["  at axial position", self.axial_positions[j], "m"],
            ["peak cladding outer temperature", self.clad_outer_temperature.max(), "K"],
            ["smallest gap width", gap.min() * 1e6, "um"],
            ["gas pressure at the end", self.gas_pressure[-1] / 1e6, "MPa"],
            ["fission gas release at the end", 100 * self.fission_gas_release[-1], "%"],
            ["largest cladding hoop strain", 100 * self.clad_hoop_strain.max(), "%"],
            ["smallest cladding hoop strain", 100 * self.clad_hoop_strain.min(), "%"],
            ["Newton iterations in total", int(self.nonlinear_iterations.sum()), ""],
        ]
        return "Summary\n" + table(rows, ["quantity", "value", "unit"])

    # ---- files ----------------------------------------------------------------
    def write_csv(self, directory: str, file_base: str = "rod") -> list[str]:
        """Write ``<file_base>_history.csv`` (one row per output time) and
        ``<file_base>_axial.csv`` (one row per output time and axial
        position), and the input report as ``<file_base>_input.txt``.
        Returns the paths."""
        os.makedirs(directory, exist_ok=True)
        paths = []
        history = os.path.join(directory, f"{file_base}_history.csv")
        names = [n for n, _, _ in ROD_QUANTITIES]
        with open(history, "w", newline="") as f:
            writer = csv.writer(f, lineterminator="\n")
            writer.writerow(names)
            f.write("# units: " + ", ".join(self.unit(n) for n in names) + "\n")
            for k in range(len(self.rod["time"])):
                writer.writerow([repr(float(self.rod[n][k])) for n in names])
        paths.append(history)
        axial = os.path.join(directory, f"{file_base}_axial.csv")
        names = [n for n, _, _ in AXIAL_QUANTITIES]
        with open(axial, "w", newline="") as f:
            writer = csv.writer(f, lineterminator="\n")
            writer.writerow(["time", "time_days", "axial_position"] + names)
            f.write("# units: s, d, m, " + ", ".join(self.unit(n) for n in names) + "\n")
            for k, t in enumerate(self.rod["time"]):
                for j, z in enumerate(self.axial_positions):
                    writer.writerow(
                        [repr(float(t)), repr(float(t) / SECONDS_PER_DAY), repr(float(z))]
                        + [repr(float(self.axial[n][k][j])) for n in names]
                    )
        paths.append(axial)
        if self.input_report:
            report = os.path.join(directory, f"{file_base}_input.txt")
            with open(report, "w") as f:
                f.write(self.input_report + "\n")
            paths.append(report)
        return paths
