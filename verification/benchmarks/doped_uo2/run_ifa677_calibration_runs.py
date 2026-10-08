# SPDX-License-Identifier: LGPL-2.1-or-later
"""Runs of IFA-677.1 rods 1 and 5 for the calibration of the doped-UO2
models (the training runs of the Gaussian processes).

Each rod runs on a nested scrambled Sobol' design over its inputs: the
parameters that both rods share (the intragranular diffusivity, the
re-solution, the fuel thermal conductivity and the gas term of the gap
conductance, each as a factor on its model, and the total densification) and the power of the rod in cycles 1 to 3 and in
cycles 4 to 6.  The priors of the factors are those of Che et al. (Nucl.
Eng. Des. 337 (2018) 271), Table 2.  The total densification of doped UO2
is uniform from 0 (complete suppression) to 1 % (the undoped value), as in
CASL-U-2019-1870, Sect. 2.7.2.  The grain radius of each rod is its
measured value (an input of the case, CASL-U-2019-1870, Table 4).  The power of cycles 4 to 6 has a prior twice as wide as that of
cycles 1 to 3, because the Halden power of the second half of the
irradiation is expected to be too high by an amount that is not known
(CASL-U-2019-1870, Sect. 4.1.2).

A run returns:

* ``release``: the fission gas release (%) at the days on which the nominal
  power reaches 1, 2, ..., 29 and 29.85 MWd/kgU, the burnups of the measured
  curve (Halden's burnup comes from the same power data);
* ``thermocouple_upper``, ``thermocouple_lower``: the median difference (K)
  between the calculated centre temperature and the middle of the measured
  band of each thermocouple, in 20 windows of 25 days;
* ``pressure``: the median rod pressure (MPa) at the days of the measured
  pressure, in the same windows.

The runs are kept in ``run_ifa677_calibration_store_rod<n>.npz``, so that an
interrupted study resumes and a larger design runs only its new points.

Usage::

    python run_ifa677_calibration_runs.py ROD SAMPLES PROCESSES
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from dualmesh import fuel, uq

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_ifa677 as ifa677  # noqa: E402

HERE = Path(__file__).resolve().parent
DAY = 86400.0
#: The burnups (MWd/kgU) of the release outputs.
BURNUPS = np.append(np.arange(1.0, 30.0), 29.85)
#: The windows (days) of the thermocouple and pressure outputs.
WINDOWS = np.arange(0.0, 501.0, 25.0)

SHARED = {
    "intragranular_diffusivity": uq.LogNormal(median=1.0, factor=10.0),
    "resolution": uq.LogNormal(median=1.0, factor=10.0),
    "fuel_thermal_conductivity": uq.Normal(1.0, 0.05),
    "gap_gas_conductance": uq.Normal(1.0, 0.25, lower=0.0),
    # A factor on the undoped total densification of 1 % (the material below).
    "densification": uq.Uniform(0.0, 1.0),
}

#: The total densification of the material, to which the factor
#: ``densification`` applies: the value of undoped UO2.
UNDOPED_DENSIFICATION = 0.01


def inputs(rod: int) -> dict:
    """The inputs of a rod: the shared parameters and its two powers."""
    return {**SHARED, f"power_cycles_1_to_3_rod_{rod}": uq.Normal(1.0, 0.025), f"power_cycles_4_to_6_rod_{rod}": uq.Normal(1.0, 0.05)}


def _energy(t, q) -> np.ndarray:
    """The energy per unit length (J/m) from the start of the history."""
    return np.concatenate(([0.0], np.cumsum(0.5 * (q[1:] + q[:-1]) * np.diff(t))))


def release_at_burnups(rod: int, result: dict, power) -> np.ndarray:
    """The release at the burnups of ``BURNUPS`` on the axis of the nominal
    power: the burnup of the run times the ratio of the nominal energy to the
    energy of the run, so that a run with another power is read at the same
    days as the measurement."""
    t, q = ifa677.power_history(rod)
    scaled = np.where(t < ifa677.CYCLE_4_START * DAY, power[0], power[1]) * q
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(t > 0.0, _energy(t, q) / _energy(t, scaled), 1.0 / power[0])
    days = np.asarray(result["days"])
    nominal = np.asarray(result["burnup"]) * np.interp(days * DAY, t, ratio)
    return np.interp(np.interp(BURNUPS, nominal, days), days, result["fgr"])


def window_medians(days, values) -> np.ndarray:
    """The median of ``values`` in each window of ``WINDOWS`` (NaN for a
    window without values)."""
    out = np.full(len(WINDOWS) - 1, np.nan)
    for i in range(len(WINDOWS) - 1):
        inside = (days >= WINDOWS[i]) & (days < WINDOWS[i + 1])
        if np.any(inside):
            out[i] = float(np.median(values[inside]))
    return out


def rod_outputs(rod: int, **factors) -> dict:
    """The outputs of one run of ``rod`` at the factors."""
    spec = ifa677.ROD[rod]
    power = (factors.pop(f"power_cycles_1_to_3_rod_{rod}"), factors.pop(f"power_cycles_4_to_6_rod_{rod}"))
    material = fuel.DopedUO2Fuel(grain_radius=spec["grain_radius"], enrichment=spec["enrichment"], theoretical_density_fraction=spec["density"] / 10963.0, total_densification=UNDOPED_DENSIFICATION)
    model_factors = fuel.ModelFactors(**factors)
    solid = ifa677.run(rod, material, False, model_factors, power)
    drilled = ifa677.run(rod, material, True, model_factors, power)
    result = ifa677.combined(solid, drilled, rod)
    out = {"release": release_at_burnups(rod, result, power)}
    thermocouples, events = ifa677.read_thermocouples(), ifa677.shutdown_days()
    for position in ("upper", "lower"):
        days, middle = ifa677.steady_middle(thermocouples[(rod, position, "measured")], events)
        out[f"thermocouple_{position}"] = window_medians(days, np.interp(days, result["days"], result["centre"]) - middle)
    measured = ifa677.read_pressures()[(rod, "measured")]
    out["pressure"] = window_medians(measured[:, 0], np.interp(measured[:, 0], result["days"], result["pressure"]))
    return out


def rod_1(**factors):
    return rod_outputs(1, **factors)


def rod_5(**factors):
    return rod_outputs(5, **factors)


def main():
    rod, samples, processes = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    model = {1: rod_1, 5: rod_5}[rod]
    store = HERE / f"run_ifa677_calibration_store_rod{rod}.npz"
    uq.propagate(model, inputs(rod), samples, method="sobol", seed=rod, processes=processes, store=store, report="summary")


if __name__ == "__main__":
    main()
