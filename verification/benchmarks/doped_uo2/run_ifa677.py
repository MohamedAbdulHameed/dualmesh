# SPDX-License-Identifier: LGPL-2.1-or-later
"""Halden IFA-677.1 rods 1 and 5: Cr2O3-doped UO2 (CASL-U-2019-1870-000 Rev. 0).

IFA-677.1, the high initial rating test, irradiated six rods in the Halden
reactor for six cycles (December 2004 to September 2007) to a rig average
burnup of about 26.3 MWd/kgOX (29.8 MWd/kgU).  Rods 1 and 5 held UO2 doped
with Cr2O3 and Al2O3.  The fission gas release inferred from the rod
pressure reached 22 % (rod 1) and 16.7 % (rod 5); puncturing rod 5 gave
16 %.  BISON gave about 17 % and 13 % (the report, printed pages 33-35).

Rod data: Table 4 of the report.  Power histories: Fig. 8(a); fission gas
release: Figs. 10 and 13; thermocouples and rod pressures: Figs. 9, 11, 12
and 14.  The centre temperature of the drilled calculation is compared with
the middle of the measured band at both thermocouples, at power and away
from the shutdowns, and so is BISON's line on the same days.

Choices this script makes where the report is silent (each is stated in the
documentation):

* The cladding outer temperature is the Halden correlation of FUMEX-II
  (IAEA-TECDOC-1687, Table 13), as for IFA-716.1.
* More than half of each stack (218.9 mm of rod 1, 222.1 mm of rod 5) is
  drilled for the thermocouples, with a 1.8 mm hole.  Each rod is computed
  twice, with solid and with annular pellets over the whole stack, and the
  release of the rod is the average of the two releases weighted by the gas
  produced in the solid and drilled lengths.  The two calculations share
  the free volume of the rod (Table 4), less the as-fabricated gap and hole.
* The fast flux is 1.6e12 n/(m^2 s) per W/m (FUMEX-II Table 10).
* The 36 shutdowns of Fig. 8(a) are part of the history, with the shape
  ``halden_history.SHUTDOWN`` (the report gives only the day).  The time steps land on them
  (``RodNumerics.power_history_tolerance``).
* The power histories give end-of-life burnups of 30.6 (rod 1) and 28.9
  MWd/kgU (rod 5), while Figs. 10 and 13 end at 29.85 for both; the sum of
  the two agrees to 0.3 %.  The histories are used as given.

Usage::

    python run_ifa677.py [output directory for the figures] [--plot-only]

The twelve calculations are saved in ``run_ifa677_cache.npz``, and
``--plot-only`` redraws the figures (and rewrites ``ifa677_results.csv``)
from it.  Each rod has three figures, ``ifa677_rod<n>_temperature.png``,
``ifa677_rod<n>_pressure.png`` and ``ifa677_rod<n>_fgr.png``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
from dualmesh import fuel

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import plotstyle  # noqa: E402
from halden_history import insert_shutdowns  # noqa: E402
from halden_history import shutdown_days as read_shutdown_days  # noqa: E402
from plotstyle import DUALMESH, INK, MEASURED, SECOND  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CACHE = plotstyle.cache_path(__file__)
DAY = 86400.0

PELLET_DIAMETER = 9.13e-3
CLAD_OD = 10.75e-3
CLAD_ID = PELLET_DIAMETER + 170e-6  # diametral gap 170 um
HOLE = 1.8e-3
ROD = {
    1: dict(
        stack=0.3986,
        drilled=0.2189,
        free_volume=5.34e-6,
        density=10690.0,
        enrichment=0.0494,
        grain_radius=28e-6,
        bison=17.0,
        measured=22.0,
    ),
    5: dict(
        stack=0.4035,
        drilled=0.2221,
        free_volume=5.26e-6,
        density=10700.0,
        enrichment=0.0491,
        grain_radius=22.5e-6,
        bison=13.0,
        measured=16.0,
    ),
}


def power_history(rod, shutdowns=True):
    """Times (s) and linear heat rates (W/m) of the rod, with the shutdowns
    of ``ifa677_shutdowns.csv`` inserted (``SHUTDOWN``)."""
    lines = (HERE / "ifa677_rods_1_5_power.csv").read_text().splitlines()
    data = np.array(
        [ln.split(",") for ln in lines if ln and not ln.startswith(("#", "time"))], dtype=float
    )
    t = np.concatenate(([0.0], data[:, 0])) * DAY
    q = np.maximum(np.concatenate(([1.0], data[:, 1 if rod == 1 else 2])) * 1e3, 1.0)
    if shutdowns:
        t, q = insert_shutdowns(t, q, read_shutdown_days(HERE / "ifa677_shutdowns.csv"))
    return t, q


def halden_cladding_temperature(linear_heat_rate, time):
    return 273.15 + 240.0 + 0.4162 * (max(linear_heat_rate, 0.0) / 1e3) ** 0.75


def run(rod, fuel_material, bore):
    """One calculation of the rod with solid (``bore`` False) or drilled
    pellets over the whole stack.  Both keep the rod's fill gas: the plenum
    is the free volume of Table 4 less the as-fabricated gap and the bore of
    the modelled pellets."""
    spec = ROD[rod]
    t, q = power_history(rod)
    stack = spec["stack"]
    gap_volume = np.pi / 4 * (CLAD_ID**2 - PELLET_DIAMETER**2) * stack
    if bore:
        gap_volume += np.pi / 4 * HOLE**2 * stack
    result = fuel.FuelRod(
        geometry=fuel.RodGeometry(
            pellet_outer_radius=0.5 * PELLET_DIAMETER,
            clad_inner_radius=0.5 * CLAD_ID,
            clad_outer_radius=0.5 * CLAD_OD,
            fuel_stack_height=stack,
            pellet_inner_radius=0.5 * HOLE if bore else 0.0,
        ),
        fuel=fuel_material,
        cladding=fuel.ZircaloyCladding(),
        fill_gas=fuel.FillGas(pressure=1.35e6, plenum_volume=spec["free_volume"] - gap_volume),
        coolant=fuel.PrescribedCladdingTemperature(
            temperature=halden_cladding_temperature, pressure=3.4e6
        ),
        power_history=fuel.PowerHistory(
            linear_heat_rate=q, time=t, fast_neutron_flux_per_linear_heat_rate=1.6e12
        ),
        numerics=fuel.RodNumerics(max_time_step=5 * DAY),
        output=fuel.RodOutput(
            print_input=False, print_steps=False, output_times=np.arange(0.0, t[-1], 2 * DAY)
        ),
    ).run()
    return dict(
        days=np.asarray(result.time) / DAY,
        burnup=np.asarray(result.burnup_in("MWd/kgHM", rod_average=True)),
        fgr=100.0 * np.asarray(result.fission_gas_release),
        centre=np.asarray(result.max_fuel_centerline_temperature),
        pressure=np.asarray(result.gas_pressure) / 1e6,
    )


def _run(args):
    return run(*args)


def combined(solid, drilled, rod):
    """The rod from its solid and drilled calculations.  The release is
    weighted by the fuel mass of the solid and drilled lengths (the gas
    produced per unit mass is the same at the same power per unit mass,
    which the rod model keeps by scaling the power with the area).  The
    pressure is weighted by length: each calculation puts the bore either
    everywhere or nowhere, and the length weighting gives the bore of the
    drilled length at its temperature to first order."""
    spec = ROD[rod]
    area_solid = np.pi / 4 * PELLET_DIAMETER**2
    area_drilled = area_solid - np.pi / 4 * HOLE**2
    w_solid = area_solid * (spec["stack"] - spec["drilled"])
    w_drilled = area_drilled * spec["drilled"]
    fgr = (w_solid * solid["fgr"] + w_drilled * drilled["fgr"]) / (w_solid + w_drilled)
    burnup = (w_solid * solid["burnup"] + w_drilled * drilled["burnup"]) / (w_solid + w_drilled)
    f = spec["drilled"] / spec["stack"]
    pressure = f * drilled["pressure"] + (1.0 - f) * solid["pressure"]
    return dict(
        days=solid["days"],
        burnup=burnup,
        fgr=fgr,
        pressure=pressure,
        centre=drilled["centre"],
        solid=solid,
        drilled=drilled,
    )


def read_thermocouples():
    """{(rod, position, series): array (days, K)} from ifa677_thermocouples.csv,
    and the days of the shutdowns."""
    import csv

    lines = (HERE / "ifa677_thermocouples.csv").read_text().splitlines()
    out: dict = {}
    for r in csv.DictReader(ln for ln in lines if not ln.startswith("#")):
        key = (int(r["rod"]), r["position"], r["series"])
        out.setdefault(key, []).append((float(r["time_days"]), float(r["temperature_K"])))
    return {k: np.array(v) for k, v in out.items()}


def read_pressures():
    import csv

    lines = (HERE / "ifa677_pressure.csv").read_text().splitlines()
    out: dict = {}
    for r in csv.DictReader(ln for ln in lines if not ln.startswith("#")):
        out.setdefault((int(r["rod"]), r["series"]), []).append(
            (float(r["time_days"]), float(r["pressure_MPa"]))
        )
    return {k: np.array(v) for k, v in out.items()}


def shutdown_days():
    return read_shutdown_days(HERE / "ifa677_shutdowns.csv")


def steady_middle(measured, events):
    """The days and the middle of the measured thermocouple band (median
    within 1 day) at power, away from the shutdowns: from 0.5 days before
    to 1.5 days after each (the restart), and from brief dips of the trace
    below 1100 K."""
    days, value = measured[:, 0], measured[:, 1]
    away = np.all(
        (days[:, None] - events[None, :] > 1.5) | (days[:, None] < events[None, :] - 0.5), axis=1
    )
    steady = away & (value > 1100.0) & (days > 6.0)
    middle = np.array([np.median(value[steady & (np.abs(days - x) <= 1.0)]) for x in days[steady]])
    return days[steady], middle


def read_fgr():
    import csv

    lines = (HERE / "ifa677_fgr.csv").read_text().splitlines()
    out: dict = {}
    for r in csv.DictReader(ln for ln in lines if not ln.startswith("#")):
        out.setdefault((int(r["rod"]), r["series"]), []).append(
            (float(r["burnup_MWd_per_kgU"]), float(r["fgr_percent"]))
        )
    return {k: np.array(v) for k, v in out.items()}


CASES = {
    "case A (best estimate)": "best_estimate",
    "case B (upper limit)": "upper_limit",
    "CASL Eq. 16 (as BISON)": "casl_2019",
}


def compute():
    """The twelve calculations (rods 1 and 5, three diffusivities, solid and
    drilled pellets), in that order: {"0": run(...), ..., "11": run(...)}."""
    from concurrent.futures import ProcessPoolExecutor

    def material(rod, case):
        spec = ROD[rod]
        return fuel.DopedUO2Fuel(
            grain_radius=spec["grain_radius"],
            enrichment=spec["enrichment"],
            theoretical_density_fraction=spec["density"] / 10963.0,
            diffusivity_case=case,
        )

    jobs = [
        (rod, material(rod, case), bore)
        for rod in (1, 5)
        for case in CASES.values()
        for bore in (False, True)
    ]
    # The twelve calculations run in parallel, one per processor
    # (DUALMESH_WORKERS=1 runs them in this process).
    workers = int(os.environ.get("DUALMESH_WORKERS", os.cpu_count() or 1))
    if workers > 1:
        with ProcessPoolExecutor(workers) as pool:
            runs = list(pool.map(_run, jobs))
    else:
        runs = [_run(job) for job in jobs]
    return {str(i): r for i, r in enumerate(runs)}


def main():
    out, plot_only = plotstyle.parse_args(sys.argv[1:], ROOT / "docs/_static/figures/doped_uo2")
    if plot_only:
        cached = plotstyle.load_cache(CACHE)
    else:
        cached = compute()
        plotstyle.save_cache(CACHE, cached)
    runs = [cached[str(i)] for i in range(len(cached))]
    digitised = read_fgr()
    thermocouples = read_thermocouples()
    pressures = read_pressures()
    events = shutdown_days()
    cases = CASES
    results = {}
    for i, (rod, case) in enumerate((r, c) for r in (1, 5) for c in cases):
        results[(rod, case)] = combined(runs[2 * i], runs[2 * i + 1], rod)

    lines = ["rod,quantity,dualmesh,bison,measured"]
    for rod in (1, 5):
        # Thermocouples: the drilled calculation of case A against both
        # thermocouples, and BISON on the same days.
        ours = results[(rod, "case A (best estimate)")]
        for position in ("upper", "lower"):
            days, middle = steady_middle(thermocouples[(rod, position, "measured")], events)
            bison = thermocouples[(rod, position, "BISON")]
            d_ours = np.interp(days, ours["days"], ours["centre"]) - middle
            d_bison = np.interp(days, bison[:, 0], bison[:, 1]) - middle
            for label, window in (("all", days > 0), ("cycles 1-3", days < 275)):
                lines.append(
                    f"{rod},{position} thermocouple bias K (median; {label}),"
                    f"{np.median(d_ours[window]):.0f},{np.median(d_bison[window]):.0f},0"
                )
            lines.append(
                f"{rod},{position} thermocouple rms K,{np.sqrt(np.mean(d_ours**2)):.0f},"
                f"{np.sqrt(np.mean(d_bison**2)):.0f},0"
            )
        fig, ax = plotstyle.new_figure()
        m = thermocouples[(rod, "upper", "measured")]
        ax.plot(m[:, 0], m[:, 1], color=MEASURED, linewidth=0.7, label="Measured, upper")
        b = thermocouples[(rod, "upper", "BISON")]
        ax.plot(b[:, 0], b[:, 1], color=INK, linewidth=0.9, label="BISON (CASL 2019)")
        ax.plot(ours["days"], ours["centre"], color=DUALMESH, linewidth=1.6, label="dualmesh")
        plotstyle.style(
            ax, "Time (days)", "Temperature (K)", f"IFA-677.1 rod {rod}: upper thermocouple"
        )
        ax.set_ylim(1100, 1950)
        plotstyle.legend_below(ax, ncol=3)
        plotstyle.save(fig, out / f"ifa677_rod{rod}_temperature.png")
        # Pressure.
        fig, ax = plotstyle.new_figure()
        pm, pb = pressures[(rod, "measured")], pressures[(rod, "BISON")]
        on = pm[:, 1] > 2.5
        d_ours = np.interp(pm[on, 0], ours["days"], ours["pressure"]) - pm[on, 1]
        d_bison = np.interp(pm[on, 0], pb[:, 0], pb[:, 1]) - pm[on, 1]
        lines.append(
            f"{rod},rod pressure bias MPa (median),{np.median(d_ours):.2f},"
            f"{np.median(d_bison):.2f},0"
        )
        ax.plot(pm[:, 0], pm[:, 1], color=MEASURED, linewidth=0.9, label="Measured")
        ax.plot(pb[:, 0], pb[:, 1], color=INK, linewidth=0.9, label="BISON (CASL 2019)")
        ax.plot(ours["days"], ours["pressure"], color=DUALMESH, linewidth=1.6, label="dualmesh")
        plotstyle.style(
            ax, "Time (days)", "Rod pressure (MPa)", f"IFA-677.1 rod {rod}: rod pressure"
        )
        ax.set_ylim(1.5, 6.0)
        plotstyle.legend_below(ax, ncol=3)
        plotstyle.save(fig, out / f"ifa677_rod{rod}_pressure.png")
        # Release.
        fig, ax = plotstyle.new_figure()
        measured = digitised[(rod, "measured")]
        bison = digitised[(rod, "BISON")]
        end = measured[-1, 0]
        ax.plot(
            measured[:, 0],
            measured[:, 1],
            color=MEASURED,
            linewidth=1.8,
            label="Measured (rod pressure)",
        )
        ax.plot(bison[:, 0], bison[:, 1], color=INK, linewidth=0.9, label="BISON (CASL 2019)")
        if (rod, "puncture") in digitised:
            p = digitised[(rod, "puncture")][0]
            ax.plot([p[0]], [p[1]], "s", color=MEASURED, markersize=7, label="Measured (puncture)")
        on = measured[:, 1] > 0.0
        bison_rms = np.sqrt(
            np.mean((np.interp(measured[on, 0], bison[:, 0], bison[:, 1]) - measured[on, 1]) ** 2)
        )
        for name, colour in zip(cases, (DUALMESH, "#7a4fd6", SECOND)):
            r = results[(rod, name)]
            ours_end = float(np.interp(end, r["burnup"], r["fgr"]))
            rms = np.sqrt(
                np.mean((np.interp(measured[on, 0], r["burnup"], r["fgr"]) - measured[on, 1]) ** 2)
            )
            lines.append(
                f"{rod},FGR at {end:.2f} MWd/kgU % ({name}),{ours_end:.2f},"
                f"{bison[-1, 1]:.2f},{measured[-1, 1]:.2f}"
            )
            lines.append(
                f"{rod},FGR rms against the measured curve % ({name}),{rms:.2f},{bison_rms:.2f},"
            )
            lines.append(f"{rod},end burnup MWd/kgU ({name}),{r['burnup'][-1]:.2f},,{end:.2f}")
            ax.plot(r["burnup"], r["fgr"], color=colour, linewidth=2.0, label=f"dualmesh, {name}")
        plotstyle.style(
            ax,
            "Rod average burnup (MWd/kgU)",
            "Fission gas release (%)",
            f"IFA-677.1 rod {rod}: fission gas release",
        )
        ax.set_xlim(0, 31)
        plotstyle.legend_below(ax, ncol=2)
        plotstyle.save(fig, out / f"ifa677_rod{rod}_fgr.png")
    (HERE / "ifa677_results.csv").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
