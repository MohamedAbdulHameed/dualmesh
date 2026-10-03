# SPDX-License-Identifier: LGPL-2.1-or-later
"""Halden IFA-716.1 rod 1: Cr2O3-doped UO2 (CASL-U-2019-1870-000 Rev. 0, Sect. 4.1).

Rod 1 of IFA-716.1 held AREVA UO2 doped with 1580 ppm Cr2O3, with a grain
radius of 35 micrometres.  It was irradiated for 12 Halden cycles (about 700
days) to a rig average burnup of 27 MWd/kgOX.  The fission gas release
inferred from the rod pressure was about 5.6 % when the pressure transducer
failed, after about 620 days, with an uncertainty of about 1.4 % (the report,
printed page 36).  BISON gave about 2.7 % in that report.  A later BISON
calculation with the doped diffusivities of cases A and B (M. W. D. Cooper
et al., J. Nucl. Mater. 545 (2021) 152590, Figs. 8a and 10a) is also
compared.

Rod data: Table 4 of the report (printed page 32).  Power history: Fig. 8(b)
of the report; its integral gives 27.0 MWd/kgUO2, the rig average burnup of
the report.

Choices this script makes where the report is silent (each is stated in the
documentation):

* The cladding outer temperature is the Halden correlation of FUMEX-II
  (IAEA-TECDOC-1687, Table 13), 240 + 0.4162 q'^0.75 degC.  The report used
  the Jens-Lottes correlation with the Halden coolant history, which is not
  published.
* The release is computed with solid pellets over the whole stack.  The
  thermocouple sits in the 115 mm drilled section at the top, so the centre
  temperature is compared from a second calculation with drilled pellets
  over the whole stack.
* The plenum volume is the free volume of Table 4 (5.80 cm^3) less the
  as-fabricated gap volume.
* The fast flux is 1.6e12 n/(m^2 s) per W/m, the Halden value of FUMEX-II
  Table 10.
* The 47 shutdowns of Fig. 8(b) are part of the history
  (``halden_history``).  The thermocouple is compared at power, away
  from the shutdowns (0.5 days before to 1.5 days after each).

Three calculations are made, with the best-estimate and the upper-limit
doped diffusivity of INL/EXT-20-59969 (cases A and B) and with the doped
diffusivity of Eq. 16 of the report.

Usage::

    python run_ifa716.py [output directory for the figures] [--plot-only]

The calculations are saved in ``run_ifa716_cache.npz``, and ``--plot-only``
redraws the figures (and rewrites ``ifa716_results.csv``) from it.  The
power, the centre temperature and the release are drawn in
``ifa716_rod1_power.png``, ``ifa716_rod1_temperature.png`` and
``ifa716_rod1_fgr.png``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from dualmesh import fuel

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import plotstyle  # noqa: E402
from halden_history import insert_shutdowns, shutdown_days  # noqa: E402
from plotstyle import DUALMESH, INK, MEASURED, MUTED, REFERENCE_GREY, SECOND  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CACHE = plotstyle.cache_path(__file__)
DAY = 86400.0

PELLET_DIAMETER = 9.12e-3
CLAD_OD = 10.75e-3
CLAD_ID = PELLET_DIAMETER + 180e-6  # diametral gap 180 um
STACK = 0.3995
FREE_VOLUME = 5.80e-6
DENSITY = 10500.0
MEASURED_FGR = (5.6, 1.4, 620.0)  # %, uncertainty %, days


def power_history(shutdowns=True):
    """Times (s) and linear heat rates (W/m), with the shutdowns of
    ``ifa716_shutdowns.csv`` inserted (``halden_history``)."""
    lines = (HERE / "ifa716_rod1_power.csv").read_text().splitlines()
    rows = [line.split(",") for line in lines if line and not line.startswith(("#", "time"))]
    data = np.array(rows, dtype=float)
    t = np.concatenate(([0.0], data[:, 0])) * DAY
    q = np.concatenate(([1.0], data[:, 1])) * 1e3
    if shutdowns:
        t, q = insert_shutdowns(t, q, shutdown_days(HERE / "ifa716_shutdowns.csv"))
    return t, q


def halden_cladding_temperature(linear_heat_rate, time):
    return 273.15 + 240.0 + 0.4162 * (max(linear_heat_rate, 0.0) / 1e3) ** 0.75


def run(fuel_material, bore_diameter=0.0):
    t, q = power_history()
    gap_volume = np.pi / 4 * (CLAD_ID**2 - PELLET_DIAMETER**2) * STACK
    gap_volume += np.pi / 4 * bore_diameter**2 * STACK
    rod = fuel.FuelRod(
        geometry=fuel.RodGeometry(pellet_outer_radius=0.5 * PELLET_DIAMETER, clad_inner_radius=0.5 * CLAD_ID, clad_outer_radius=0.5 * CLAD_OD, fuel_stack_height=STACK, pellet_inner_radius=0.5 * bore_diameter),
        fuel=fuel_material,
        cladding=fuel.ZircaloyCladding(),
        fill_gas=fuel.FillGas(pressure=1.0e6, plenum_volume=FREE_VOLUME - gap_volume),
        coolant=fuel.PrescribedCladdingTemperature(temperature=halden_cladding_temperature, pressure=3.4e6),
        power_history=fuel.PowerHistory(linear_heat_rate=q, time=t, fast_neutron_flux_per_linear_heat_rate=1.6e12),
        numerics=fuel.RodNumerics(max_time_step=5 * DAY),
        output=fuel.RodOutput(print_input=False, print_steps=False, output_times=np.arange(0.0, t[-1], 2 * DAY)),
    )
    r = rod.run()
    return dict(days=np.asarray(r.time) / DAY, burnup=np.asarray(r.burnup_in("MWd/kgHM", rod_average=True)), fgr=100.0 * np.asarray(r.fission_gas_release), centre=np.asarray(r.max_fuel_centerline_temperature) - 273.15, pressure=np.asarray(r.gas_pressure) / 1e6)


def cooper2021():
    """BISON curves of Cooper et al. (2021): {(figure, series): array}."""
    import csv

    lines = (HERE / "cooper2021_ifa716_rod1.csv").read_text().splitlines()
    rows = csv.DictReader(ln for ln in lines if not ln.startswith("#"))
    out: dict = {}
    for r in rows:
        out.setdefault((r["figure"], r["series"]), []).append((float(r["x"]), float(r["y"])))
    return {k: np.array(v) for k, v in out.items()}


def upper_envelope(curve, x, half_width):
    """The largest value of a staircase or noisy curve near each x."""
    out = np.full(len(x), np.nan)
    for i, xi in enumerate(x):
        near = np.abs(curve[:, 0] - xi) < half_width
        if near.any():
            out[i] = curve[near, 1].max()
    return out


def median_near(curve, x, half_width):
    out = np.full(len(x), np.nan)
    for i, xi in enumerate(x):
        near = np.abs(curve[:, 0] - xi) < half_width
        if near.any():
            out[i] = np.median(curve[near, 1])
    return out


def read_csv(name):
    lines = (HERE / name).read_text().splitlines()
    rows = [line.split(",") for line in lines if line and not line[0].isalpha() and line[0] != "#"]
    return np.array(rows, dtype=float)


# Legend labels of the three calculations (every fuel is the doped UO2).
LEGEND = {"doped, case A (best estimate)": "dualmesh, case A (best estimate)", "doped, case B (upper limit)": "dualmesh, case B (upper limit)", "doped, CASL Eq. 16 (as BISON)": "dualmesh, CASL Eq. 16 (as BISON)"}


def compute():
    """The three calculations with solid pellets and the drilled calculation
    of case A: {"results": {name: run(...)}, "tc_run": run(...)}."""
    common = dict(enrichment=0.049, theoretical_density_fraction=DENSITY / 10963.0)
    cases = {"doped, case A (best estimate)": fuel.DopedUO2Fuel(grain_radius=35e-6, **common), "doped, case B (upper limit)": fuel.DopedUO2Fuel(grain_radius=35e-6, diffusivity_case="upper_limit", **common), "doped, CASL Eq. 16 (as BISON)": fuel.DopedUO2Fuel(grain_radius=35e-6, diffusivity_case="casl_2019", **common)}
    results = {name: run(material) for name, material in cases.items()}
    # The thermocouple sits in the drilled top section: an annular pellet.
    tc_run = run(fuel.DopedUO2Fuel(grain_radius=35e-6, **common), bore_diameter=1.8e-3)
    return dict(results=results, tc_run=tc_run)


def main():
    out, plot_only = plotstyle.parse_args(sys.argv[1:], ROOT / "docs/_static/figures/doped_uo2")
    if plot_only:
        cached = plotstyle.load_cache(CACHE)
    else:
        cached = compute()
        plotstyle.save_cache(CACHE, cached)
    results, tc_run = cached["results"], cached["tc_run"]

    measured_tc = read_csv("ifa716_rod1_thermocouple.csv")
    bison_tc = read_csv("ifa716_rod1_bison_tc.csv")
    measured_fgr = read_csv("ifa716_rod1_fgr.csv")
    bison19_fgr = read_csv("ifa716_rod1_bison_fgr.csv")
    t, q = power_history()
    events = shutdown_days(HERE / "ifa716_shutdowns.csv")
    away = np.all((measured_tc[:, 0][:, None] - events[None, :] > 1.5) | (measured_tc[:, 0][:, None] < events[None, :] - 0.5), axis=1)
    on = (np.interp(measured_tc[:, 0] * DAY, t, q) > 20e3) & (measured_tc[:, 0] > 5) & away
    days = measured_tc[on, 0]
    ours = np.interp(days, tc_run["days"], tc_run["centre"] + 273.15) - measured_tc[on, 1]
    bison = np.interp(days, bison_tc[:, 0], bison_tc[:, 1]) - measured_tc[on, 1]

    # BISON 2021 (Cooper et al.): its thermocouple curve against the measured
    # curve of the same figure, on the same days.
    c21 = cooper2021()
    measured_c21 = median_near(c21[("Fig. 8a", "measured")], days, 0.7)
    # The BISON lines are polylines through few vertices: interpolate along
    # them (they are drawn left to right).
    tc_a = c21[("Fig. 8a", "BISON case A")]
    bison21 = np.interp(days, tc_a[:, 0], tc_a[:, 1]) - measured_c21
    bison21 = bison21[np.isfinite(bison21)]

    lines = ["quantity,dualmesh,bison,measured"]
    lines.append(f"thermocouple bias K (median),{np.median(ours):.0f},{np.median(bison):.0f},0")
    lines.append(f"thermocouple rms K,{np.sqrt(np.mean(ours**2)):.0f},{np.sqrt(np.mean(bison**2)):.0f},0")
    lines.append(f"thermocouple bias K (median; BISON 2021 case A),,{np.median(bison21):.0f},0")
    lines.append(f"thermocouple rms K (BISON 2021 case A),,{np.sqrt(np.mean(bison21**2)):.0f},0")
    # The release against burnup, compared with the whole measured curve.
    mb = measured_fgr[:, 0]
    bison19_end = float(np.interp(mb[-1], bison19_fgr[:, 0], bison19_fgr[:, 1]))
    bison19_rms = np.sqrt(np.mean((np.interp(mb, bison19_fgr[:, 0], bison19_fgr[:, 1]) - measured_fgr[:, 1]) ** 2))
    for name, r in results.items():
        at = np.interp(MEASURED_FGR[2], r["days"], r["fgr"])
        lines.append(f"FGR at 620 days % ({name}),{at:.2f},{bison19_end:.2f},{MEASURED_FGR[0]}")
    bison21_fgr = {}
    for case in ("A", "B"):
        curve = c21[("Fig. 10a", f"BISON case {case}")]
        # The staircase outline: the upper envelope within 0.05 MWd/kgU.
        bison21_fgr[case] = upper_envelope(curve, mb, 0.05)
    for name, r in results.items():
        burnup_u = r["burnup"]  # MWd/kgHM equals MWd/kgU here
        ours_fgr = np.interp(mb, burnup_u, r["fgr"])
        rms = np.sqrt(np.mean((ours_fgr - measured_fgr[:, 1]) ** 2))
        case = "A" if "case A" in name else "B" if "case B" in name else None
        b21 = f"{bison19_rms:.2f}"
        if case:
            ok = np.isfinite(bison21_fgr[case])
            b21 = np.sqrt(np.mean((bison21_fgr[case][ok] - measured_fgr[ok, 1]) ** 2))
            b21 = f"{b21:.2f}"
        lines.append(f"FGR rms against the measured curve % ({name}),{rms:.2f},{b21},")
    for case in ("A", "B"):
        last = bison21_fgr[case][np.isfinite(bison21_fgr[case])][-1]
        lines.append(f"FGR at the last measured point % (BISON 2021 case {case}),,{last:.2f},{measured_fgr[-1, 1]:.2f}")
    (HERE / "ifa716_results.csv").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    fig, ax = plotstyle.new_figure()
    ax.plot(t / DAY, q / 1e3, color=MUTED, linewidth=1.0)
    plotstyle.style(ax, "Time (days)", "Linear heat rate (kW/m)", "IFA-716.1 rod 1: rod average power")
    ax.set_xlim(0, 700)
    ax.set_ylim(0, 36)
    plotstyle.save(fig, out / "ifa716_rod1_power.png")

    fig, ax = plotstyle.new_figure()
    ax.plot(measured_tc[:, 0], measured_tc[:, 1], color=MEASURED, linewidth=1.0, label="Measured")
    ax.plot(bison_tc[:, 0], bison_tc[:, 1], color=INK, linewidth=0.8, label="BISON (CASL report)")
    ax.plot(tc_run["days"], tc_run["centre"] + 273.15, color=DUALMESH, linewidth=1.6, label="dualmesh")
    plotstyle.style(ax, "Time (days)", "Temperature (K)", "IFA-716.1 rod 1: centre temperature at the thermocouple")
    ax.set_xlim(0, 700)
    ax.set_ylim(700, 1600)
    plotstyle.legend_below(ax, ncol=3)
    plotstyle.save(fig, out / "ifa716_rod1_temperature.png")

    fig, ax = plotstyle.new_figure()
    ax.plot(mb, measured_fgr[:, 1], color=MEASURED, linewidth=1.8, label="Measured (rod pressure)")
    ax.plot(bison19_fgr[:, 0], bison19_fgr[:, 1], "-.", color=INK, linewidth=0.9, label="BISON 2019 (CASL Eq. 16)")
    for case, ls in (("A", "-"), ("B", ":")):
        curve = c21[("Fig. 10a", f"BISON case {case}")]
        order = np.argsort(curve[:, 0], kind="stable")
        ax.plot(curve[order, 0], curve[order, 1], ls, color=INK, linewidth=0.8, label=f"BISON 2021, case {case}")
    styles = [(DUALMESH, "-"), (DUALMESH, ":"), (SECOND, "-"), (REFERENCE_GREY, "--")]
    for (name, r), (colour, ls) in zip(results.items(), styles):
        ax.plot(r["burnup"], r["fgr"], ls, color=colour, linewidth=2.0, label=LEGEND[name])
    plotstyle.style(ax, "Rod average burnup (MWd/kgU)", "Fission gas release (%)", "IFA-716.1 rod 1: fission gas release")
    ax.set_xlim(0, 31)
    plotstyle.legend_below(ax, ncol=2)
    plotstyle.save(fig, out / "ifa716_rod1_fgr.png")


if __name__ == "__main__":
    main()
