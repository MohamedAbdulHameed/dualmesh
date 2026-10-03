# SPDX-License-Identifier: LGPL-2.1-or-later
"""Uncertainty and sensitivity of IFA-677.1 rod 1 with the nine parameters
and ranges of Che et al. (Nucl. Eng. Des. 337 (2018) 271), Table 2.

Each parameter multiplies a model of the rod through
:class:`dualmesh.fuel.ModelFactors`:

* linear heat rate, normal, standard deviation 2.5 % (95 % range 5 %);
* total densification, normal, 10 %;
* gas and contact terms of the gap conductance, normal, 25 % each;
* fuel thermal conductivity, normal, 5 %;
* grain radius, normal, 30 %, kept positive;
* intragranular diffusion coefficient, re-solution parameter and
  grain-boundary diffusion coefficient, log-normal, a factor of 10 each way
  as the 95 % range (a factor of 100 in all).

The rod is the combination of its solid and drilled calculations of
``run_ifa677.py`` (diffusivity case A).  The outputs are the fission gas
release at the end of the measured curve, 29.85 MWd/kgU, and the median
difference between the calculated centre temperature and the middle of the
measured band of the upper thermocouple.  The runs follow a scrambled
Sobol' design, whose first points are the design of a smaller study, and
are kept in ``run_ifa677_uq_store.npz``, so that an interrupted study resumes
and a larger study runs only the new points.  The Gaussian process of the
Sobol' indices is trained on the same runs.  The script writes
``ifa677_uq_results.csv`` beside itself.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from dualmesh import fuel, uq

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_ifa677 as ifa677  # noqa: E402

HERE = Path(__file__).resolve().parent
STORE = HERE / "run_ifa677_uq_store.npz"
END_BURNUP = 29.85  # MWd/kgU, the end of the measured release curve of rod 1
INPUTS = {
    "linear_heat_rate": uq.Normal(1.0, 0.025),
    "densification": uq.Normal(1.0, 0.10),
    "gap_gas_conductance": uq.Normal(1.0, 0.25, lower=0.0),
    "gap_contact_conductance": uq.Normal(1.0, 0.25, lower=0.0),
    "fuel_thermal_conductivity": uq.Normal(1.0, 0.05),
    "grain_radius": uq.Normal(1.0, 0.30, lower=0.05),
    "intragranular_diffusivity": uq.LogNormal(median=1.0, factor=10.0),
    "resolution": uq.LogNormal(median=1.0, factor=10.0),
    "grain_boundary_diffusivity": uq.LogNormal(median=1.0, factor=10.0),
}


def rod_1(**factors):
    """The fission gas release (%) at 29.85 MWd/kgU and the median bias (K)
    of the upper thermocouple of rod 1 for one set of factors."""
    spec = ifa677.ROD[1]
    material = fuel.DopedUO2Fuel(grain_radius=spec["grain_radius"], enrichment=spec["enrichment"], theoretical_density_fraction=spec["density"] / 10963.0)
    model_factors = fuel.ModelFactors(**factors)
    solid = ifa677.run(1, material, False, model_factors)
    drilled = ifa677.run(1, material, True, model_factors)
    rod = ifa677.combined(solid, drilled, 1)
    days, middle = ifa677.steady_middle(ifa677.read_thermocouples()[(1, "upper", "measured")], ifa677.shutdown_days())
    bias = float(np.median(np.interp(days, rod["days"], rod["centre"]) - middle))
    return {"release": float(np.interp(END_BURNUP, rod["burnup"], rod["fgr"])), "thermocouple_bias": bias}


def main():
    samples = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    processes = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    runs = uq.propagate(rod_1, INPUTS, samples, method="sobol", seed=0, processes=processes, store=STORE, progress=True)
    print(runs.summary())
    indices = uq.sobol(rod_1, INPUTS, 1024, surrogate="gaussian_process", training=runs, seed=0)
    print(indices.summary())
    lines = ["quantity,mean,standard_deviation,low_95,high_95"]
    for name in ("release", "thermocouple_bias"):
        low, high = runs.interval(name)
        lines.append(f"{name},{runs.mean(name):.3f},{runs.standard_deviation(name):.3f},{low:.3f},{high:.3f}")
    lines.append("")
    lines.append("input," + ",".join(f"{o} first order,{o} total" for o in ("release", "thermocouple_bias")))
    for n in INPUTS:
        lines.append(n + "," + ",".join(f"{indices.first_order[o][n]:.3f},{indices.total[o][n]:.3f}" for o in ("release", "thermocouple_bias")))
    (HERE / "ifa677_uq_results.csv").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
