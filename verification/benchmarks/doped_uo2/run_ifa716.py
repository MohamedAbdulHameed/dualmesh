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

    python run_ifa716.py [output directory for the figures]
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from dualmesh import fuel  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from halden_history import insert_shutdowns, shutdown_days  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DAY = 86400.0
DUALMESH = "#2a78d6"
SECOND = "#1baf7a"
REFERENCE_GREY = "#a3a29c"
MEASURED = "#eb6834"
INK = "#0b0b0b"
MUTED = "#52514e"

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
        geometry=fuel.RodGeometry(
            pellet_outer_radius=0.5 * PELLET_DIAMETER,
            clad_inner_radius=0.5 * CLAD_ID,
            clad_outer_radius=0.5 * CLAD_OD,
            fuel_stack_height=STACK,
            pellet_inner_radius=0.5 * bore_diameter,
        ),
        fuel=fuel_material,
        cladding=fuel.ZircaloyCladding(),
        fill_gas=fuel.FillGas(pressure=1.0e6, plenum_volume=FREE_VOLUME - gap_volume),
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
    )
    r = rod.run()
    return dict(
        days=np.asarray(r.time) / DAY,
        burnup=np.asarray(r.burnup_in("MWd/kgHM", rod_average=True)),
        fgr=100.0 * np.asarray(r.fission_gas_release),
        centre=np.asarray(r.max_fuel_centerline_temperature) - 273.15,
        pressure=np.asarray(r.gas_pressure) / 1e6,
    )


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


def style(ax, xlabel, ylabel, title):
    ax.set_title(title, loc="left", fontsize=10, color=INK)
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(True, color="#e4e3df", linewidth=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, labelsize=8)


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "docs/_static/figures/doped_uo2"
    out.mkdir(parents=True, exist_ok=True)
    common = dict(enrichment=0.049, theoretical_density_fraction=DENSITY / 10963.0)
    cases = {
        "doped, case A (best estimate)": fuel.DopedUO2Fuel(grain_radius=35e-6, **common),
        "doped, case B (upper limit)": fuel.DopedUO2Fuel(
            grain_radius=35e-6, diffusivity_case="upper_limit", **common
        ),
        "doped, CASL Eq. 16 (as BISON)": fuel.DopedUO2Fuel(
            grain_radius=35e-6, diffusivity_case="casl_2019", **common
        ),
    }
    results = {name: run(material) for name, material in cases.items()}
    # The thermocouple sits in the drilled top section: an annular pellet.
    tc_run = run(fuel.DopedUO2Fuel(grain_radius=35e-6, **common), bore_diameter=1.8e-3)

    measured_tc = read_csv("ifa716_rod1_thermocouple.csv")
    bison_tc = read_csv("ifa716_rod1_bison_tc.csv")
    measured_fgr = read_csv("ifa716_rod1_fgr.csv")
    bison19_fgr = read_csv("ifa716_rod1_bison_fgr.csv")
    t, q = power_history()
    events = shutdown_days(HERE / "ifa716_shutdowns.csv")
    away = np.all(
        (measured_tc[:, 0][:, None] - events[None, :] > 1.5)
        | (measured_tc[:, 0][:, None] < events[None, :] - 0.5),
        axis=1,
    )
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
    lines.append(
        f"thermocouple rms K,{np.sqrt(np.mean(ours**2)):.0f},{np.sqrt(np.mean(bison**2)):.0f},0"
    )
    lines.append(f"thermocouple bias K (median; BISON 2021 case A),,{np.median(bison21):.0f},0")
    lines.append(f"thermocouple rms K (BISON 2021 case A),,{np.sqrt(np.mean(bison21**2)):.0f},0")
    # The release against burnup, compared with the whole measured curve.
    mb = measured_fgr[:, 0]
    bison19_end = float(np.interp(mb[-1], bison19_fgr[:, 0], bison19_fgr[:, 1]))
    bison19_rms = np.sqrt(
        np.mean((np.interp(mb, bison19_fgr[:, 0], bison19_fgr[:, 1]) - measured_fgr[:, 1]) ** 2)
    )
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
        lines.append(
            f"FGR at the last measured point % (BISON 2021 case {case}),,{last:.2f},"
            f"{measured_fgr[-1, 1]:.2f}"
        )
    (HERE / "ifa716_results.csv").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.8), constrained_layout=True)
    ax = axes[0]
    ax.plot(t / DAY, q / 1e3, color=MUTED, linewidth=1.0)
    style(ax, "Time (days)", "Linear heat rate (kW/m)", "Rod average power")
    ax.set_xlim(0, 700)
    ax.set_ylim(0, 36)

    ax = axes[1]
    ax.plot(measured_tc[:, 0], measured_tc[:, 1], color=MEASURED, linewidth=1.0, label="Measured")
    ax.plot(bison_tc[:, 0], bison_tc[:, 1], color=INK, linewidth=0.8, label="BISON (CASL report)")
    ax.plot(
        tc_run["days"], tc_run["centre"] + 273.15, color=DUALMESH, linewidth=1.6, label="dualmesh"
    )
    style(ax, "Time (days)", "Temperature (K)", "Centre temperature at the thermocouple")
    ax.set_xlim(0, 700)
    ax.set_ylim(700, 1600)
    ax.legend(frameon=False, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=3)

    ax = axes[2]
    ax.plot(mb, measured_fgr[:, 1], color=MEASURED, linewidth=1.6, label="Measured (rod pressure)")
    ax.plot(
        bison19_fgr[:, 0],
        bison19_fgr[:, 1],
        "-.",
        color=INK,
        linewidth=0.8,
        label="BISON 2019 (CASL Eq. 16)",
    )
    for case, ls in (("A", "-"), ("B", ":")):
        curve = c21[("Fig. 10a", f"BISON case {case}")]
        order = np.argsort(curve[:, 0], kind="stable")
        ax.plot(
            curve[order, 0],
            curve[order, 1],
            ls,
            color=INK,
            linewidth=0.7,
            label=f"BISON 2021, case {case}",
        )
    styles = [(DUALMESH, "-"), (DUALMESH, ":"), (SECOND, "-"), (REFERENCE_GREY, "--")]
    for (name, r), (colour, ls) in zip(results.items(), styles):
        ax.plot(r["burnup"], r["fgr"], ls, color=colour, linewidth=1.8, label=f"dualmesh, {name}")
    style(ax, "Rod average burnup (MWd/kgU)", "Fission gas release (%)", "Fission gas release")
    ax.set_xlim(0, 31)
    ax.legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=2)
    fig.savefig(out / "ifa716_rod1.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
