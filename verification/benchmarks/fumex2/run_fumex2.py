# SPDX-License-Identifier: LGPL-2.1-or-later
"""FUMEX-II simplified cases 27(1), 27(2a) and 27(2b) (IAEA-TECDOC-1687, 2012).

The three cases share the rod of Table 13 of the report, a Halden-type rod:
solid UO2 pellets of 10.61 mm diameter at 95 % of the theoretical density
with a grain diameter of 15 micrometres, Zircaloy-2 cladding with an inner
diameter of 10.8 mm and a wall of 0.95 mm, 5 bar of helium, one axial zone,
a large plenum and a cladding outer temperature given by

    T_co = 240 + 0.4162 q'^0.75   (degC, q' in kW/m).

* 27(2a): a constant 15 kW/m up to 100 MWd/kgU.
* 27(2b): a power falling linearly from 20 kW/m at the start to 10 kW/m at
  100 MWd/kgU.
* 27(1): the locus of the centre temperature and burnup at which the fission
  gas release reaches 1 %, found here from constant power histories.

The report gives no measurements for these cases.  They compare codes with
each other, and case 27(1) with the empirical threshold of Vitanza drawn in
Fig. 7 of the report.  The participants' curves, read from the vector
figures of the report, are in ``fumex2_participants.csv``.

Choices this script makes where the specification is silent (each is stated
in the documentation):

* the grain radius is half the stated grain diameter, 7.5 micrometres;
* the stack is 0.1 m long with a 50 cm^3 plenum ("a large plenum");
* the fast flux is small, 1e11 n/(m^2 s) per W/m ("negligible clad creep
  down");
* every other input keeps its default.

Usage::

    python run_fumex2.py [output directory for the figures] [--plot-only]

The calculations are saved in ``run_fumex2_cache.npz``, and ``--plot-only``
redraws the figures (and rewrites ``fumex2_results.csv``) from it.  Each
case is drawn in a figure of its own: ``fumex2_case27_2a_fgr.png``,
``fumex2_case27_2b_fgr.png`` and ``fumex2_case27_1.png``.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from dualmesh import fuel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import plotstyle  # noqa: E402
from plotstyle import DUALMESH  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CACHE = plotstyle.cache_path(__file__)

DAY = 86400.0
REFERENCE = plotstyle.MEASURED
PARTICIPANT = plotstyle.REFERENCE_GREY


def read_csv(path):
    """Rows of a CSV file with comment lines starting with '#', as dicts."""
    import csv

    with open(path) as fh:
        return list(csv.DictReader(line for line in fh if not line.startswith("#")))


def halden_cladding_temperature(linear_heat_rate, time):
    """FUMEX-II Table 13: T_co = 240 + 0.4162 q'^0.75 degC, q' in kW/m."""
    return 273.15 + 240.0 + 0.4162 * (max(linear_heat_rate, 0.0) / 1e3) ** 0.75


def make_rod(history: fuel.PowerHistory, output_times) -> fuel.FuelRod:
    geometry = fuel.RodGeometry(
        pellet_outer_radius=0.5 * 10.61e-3,
        clad_inner_radius=0.5 * 10.8e-3,
        clad_outer_radius=0.5 * 10.8e-3 + 0.95e-3,
        fuel_stack_height=0.1,
    )
    return fuel.FuelRod(
        geometry=geometry,
        fuel=fuel.UO2Fuel(enrichment=0.13, theoretical_density_fraction=0.95, grain_radius=7.5e-6),
        cladding=fuel.ZircaloyCladding(),
        fill_gas=fuel.FillGas(pressure=0.5e6, plenum_volume=50e-6),
        coolant=fuel.PrescribedCladdingTemperature(
            temperature=halden_cladding_temperature, pressure=3.4e6
        ),
        power_history=history,
        numerics=fuel.RodNumerics(max_time_step=20 * DAY),
        output=fuel.RodOutput(print_input=False, print_steps=False, output_times=output_times),
    )


def run(history, end_days):
    rod = make_rod(history, np.linspace(0.0, end_days * DAY, 201))
    result = rod.run()
    burnup = result.burnup_in("MWd/kgU", rod_average=True)
    return (
        burnup,
        100.0 * np.asarray(result.fission_gas_release),
        np.asarray(result.max_fuel_centerline_temperature) - 273.15,
    )


def case_2a():
    history = fuel.PowerHistory(
        linear_heat_rate=[15e3, 15e3],
        burnup=[0.0, 100.0],
        burnup_unit="MWd/kgU",
        fast_neutron_flux_per_linear_heat_rate=1e11,
    )
    return run(history, 5500.0)


def case_2b():
    history = fuel.PowerHistory(
        linear_heat_rate=[20e3, 10e3],
        burnup=[0.0, 100.0],
        burnup_unit="MWd/kgU",
        fast_neutron_flux_per_linear_heat_rate=1e11,
    )
    return run(history, 6000.0)


def case_1(powers_kw_m):
    """Burnup and centre temperature at 1 % release for each constant power."""
    points = []
    for q in powers_kw_m:
        history = fuel.PowerHistory(
            linear_heat_rate=[q * 1e3, q * 1e3],
            burnup=[0.0, 100.0],
            burnup_unit="MWd/kgU",
            fast_neutron_flux_per_linear_heat_rate=1e11,
        )
        end_days = 100.0 * 0.8119e3 / q  # 0.812 kgU/m, MWd/kgU -> days at q kW/m
        burnup, fgr, tc = run(history, end_days * 1.02)
        above = np.nonzero(fgr >= 1.0)[0]
        if len(above) == 0 or above[0] == 0:
            points.append((q, np.nan, np.nan))
            continue
        i = above[0]
        w = (1.0 - fgr[i - 1]) / (fgr[i] - fgr[i - 1])
        points.append(
            (
                q,
                burnup[i - 1] + w * (burnup[i] - burnup[i - 1]),
                tc[i - 1] + w * (tc[i] - tc[i - 1]),
            )
        )
        print(
            f"  27(1) {q:5.1f} kW/m: 1 % FGR at {points[-1][1]:6.2f} MWd/kgU, "
            f"{points[-1][2]:7.1f} degC"
        )
    return np.array(points)


def participants():
    rows = read_csv(HERE / "fumex2_participants.csv")
    data: dict[tuple, list] = {}
    for row in rows:
        # One entry per PDF path, in drawing order, so no line joins two paths.
        key = (row["case"], row["code"], row.get("piece", "0"))
        data.setdefault(key, [[], []])
        data[key][0].append(float(row["burnup"]))
        data[key][1].append(float(row["value"]))
    return {k: (np.array(v[0]), np.array(v[1])) for k, v in data.items()}


def plot_fgr(case, burnup, fgr, title, data, ymax, path):
    fig, ax = plotstyle.new_figure()
    first = True
    for (c, _code, _piece), (x, y) in sorted(data.items()):
        if c != case:
            continue
        ax.plot(x, y, color=PARTICIPANT, linewidth=1.0, label="FUMEX-II codes" if first else None)
        first = False
    ax.plot(burnup, fgr, color=DUALMESH, linewidth=2.2, label="dualmesh")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, ymax)
    plotstyle.style(ax, "Rod average burnup (MWd/kgU)", "Fission gas release (%)", title)
    plotstyle.legend(ax, loc="upper left")
    plotstyle.save(fig, path)


def compute():
    t0 = time.time()
    b2a, f2a, _ = case_2a()
    b2b, f2b, _ = case_2b()
    powers = [16, 17, 18, 19, 20, 22, 24, 27, 30, 34, 38, 43]
    locus = case_1(powers)
    print(f"wall time {time.time() - t0:.0f} s")
    return dict(b2a=b2a, f2a=f2a, b2b=b2b, f2b=f2b, locus=locus)


def main():
    out, plot_only = plotstyle.parse_args(sys.argv[1:], ROOT / "docs/_static/figures/fumex2")
    data = participants()
    if plot_only:
        z = plotstyle.load_cache(CACHE)
    else:
        z = compute()
        plotstyle.save_cache(CACHE, z)
    b2a, f2a, b2b, f2b, locus = z["b2a"], z["f2a"], z["b2b"], z["f2b"], z["locus"]

    plot_fgr(
        "27(2a)",
        b2a,
        f2a,
        "FUMEX-II case 27(2a): 15 kW/m constant",
        data,
        40,
        out / "fumex2_case27_2a_fgr.png",
    )
    plot_fgr(
        "27(2b)",
        b2b,
        f2b,
        "FUMEX-II case 27(2b): 20 to 10 kW/m",
        data,
        35,
        out / "fumex2_case27_2b_fgr.png",
    )

    fig, ax = plotstyle.new_figure()
    first = True
    for (c, code, _piece), (x, y) in sorted(data.items()):
        if c != "27(1)" or code == "Vitanza":
            continue
        ax.plot(x, y, color=PARTICIPANT, linewidth=1.0, label="FUMEX-II codes" if first else None)
        first = False
    vx, vy = data[("27(1)", "Vitanza", "0")]
    order = np.argsort(vx)
    ax.plot(
        vx[order], vy[order], color=REFERENCE, linewidth=2.2, label="Vitanza threshold (Fig. 7)"
    )
    ok = np.isfinite(locus[:, 1])
    ax.plot(
        locus[ok, 1],
        locus[ok, 2],
        "o-",
        color=DUALMESH,
        linewidth=2.2,
        markersize=6,
        markeredgecolor="white",
        label="dualmesh",
    )
    ax.set_xlim(0, 100)
    ax.set_ylim(500, 1700)
    plotstyle.style(
        ax,
        "Rod average burnup (MWd/kgU)",
        r"Centre temperature ($^\circ$C)",
        "FUMEX-II case 27(1): onset of 1 % fission gas release",
    )
    plotstyle.legend_below(ax, ncol=3)
    plotstyle.save(fig, out / "fumex2_case27_1.png")

    lines = ["case,quantity,burnup_MWd_kgU,dualmesh"]
    for b in (25, 50, 75, 100):
        lines.append(f"27(2a),FGR_percent,{b},{np.interp(b, b2a, f2a):.2f}")
        lines.append(f"27(2b),FGR_percent,{b},{np.interp(b, b2b, f2b):.2f}")
    for q, b, t in locus:
        lines.append(f"27(1),centre_temperature_degC_at_{q:g}kW/m,{b:.2f},{t:.1f}")
    (HERE / "fumex2_results.csv").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
