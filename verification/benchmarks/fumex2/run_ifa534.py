# SPDX-License-Identifier: LGPL-2.1-or-later
"""FUMEX-II priority cases 1 and 2: IFA-534.14 rods 18 and 19 (grain size effect).

Two rods from the same Goesgen base irradiation (52 MWd/kgUO2) were
refabricated and irradiated again in the Halden reactor.  Rod 18 had a grain
diameter of 22.1 micrometres and rod 19 of 8.5 micrometres.  The gas
released during the Halden irradiation, measured by puncture, was 8.89 %
(rod 18) and 4.68 % (rod 19) according to the text, but 4.7 % (rod 18) and
8.9 % (rod 19) according to Figs. 21 and 22, a ratio of 1.85 close to the grain size ratio,
as a Booth model predicts (IAEA-TECDOC-1687, 2012, section 7.1, printed
pages 51 and 52).

The rod data are Tables 8 to 10 of the report.  The power history is Fig. 20
(rod 18), read from the vector figure of the report.  The
report says the history of rod 19 was similar, so both rods use it.

Choices this script makes where the report is silent (each is stated in the
documentation):

* The history is turned from burnup into time with the heavy metal mass of
  the rod.  Points at the same burnup (start-ups and shutdowns, which
  consume no burnup) are one hour apart.  The refabrication is a shutdown of
  one hour.
* Cladding temperature.  In the base irradiation, the Goesgen coolant (inlet
  308 degC, 15.5 MPa, 3127 kg/(m^2 s)) at 15 K above the inlet and the
  Dittus-Boelter coefficient of that flow on a 14.3 mm pitch (the 15x15
  pitch of FUMEX-II case 27(2d)).  In the Halden irradiation, the Halden
  correlation of FUMEX-II Table 13, T = 240 + 0.4162 q'^0.75 degC.
* The refabrication does not reset the rod gas.  The gas released in the
  base irradiation stays in the rod, which is a small effect since that
  release is small.
* The measured values are those of Figs. 21 and 22 of the report, which
  interchange the two numbers of the sentence on printed page 51 (see
  MEASURED_FGR below).
* The grain radius is half the grain diameter of Table 8.  The total
  densification is the drop of porosity in Table 8 (3.9 to 3.5 % and 3.3
  to 2.4 %).
* The fast flux of the base irradiation (5e12 n/(cm^2 s) per kW/m) is used
  throughout.

The quantity compared is the release during the Halden irradiation as a
fraction of the gas produced over the whole life of the rod,
F_end - F_refab * Bu_refab / Bu_end, with F_refab taken at the end of the
refabrication shutdown: the gas released when the rod cooled at the end of
the base irradiation was removed when the rod was refabricated.

Usage::

    python run_ifa534.py [output directory for the figures] [--plot-only]

The calculations are saved in ``run_ifa534_cache.npz``, and ``--plot-only``
redraws the figures (and rewrites ``ifa534_results.csv``) from it.  The
release over the life of the rods is drawn in ``fumex2_ifa534_cumulative_fgr.png``
and the release during the Halden irradiation in ``fumex2_ifa534_halden_fgr.png``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from dualmesh import fuel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import plotstyle  # noqa: E402
from plotstyle import DUALMESH, INK, MEASURED  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CACHE = plotstyle.cache_path(__file__)

DAY = 86400.0
HOUR = 3600.0

REFAB_BURNUP = 52.0  # MWd/kgUO2, Fig. 20
# Measured release during the Halden irradiation (%).  The text of the report
# (printed page 51) gives 8.89 % for rod 18 and 4.68 % for rod 19, but Figs. 21
# and 22 show 4.8 % for rod 18 and 8.9 % for rod 19, which also agrees with
# the ratio rod 19 / rod 18 = 1.85 of printed page 52.  The figure values are
# used.
MEASURED_FGR = {18: 4.68, 19: 8.89}
ROD = {18: dict(porosity=0.039, final_porosity=0.035, grain_diameter=22.1e-6, enrichment=0.0384), 19: dict(porosity=0.033, final_porosity=0.024, grain_diameter=8.5e-6, enrichment=0.0379)}
PELLET_DIAMETER = 9.12e-3
CLAD_ID, CLAD_OD = 9.29e-3, 10.75e-3
STACK = 0.411
UO2_PER_U = 270.03 / 238.03  # mass of UO2 per mass of U, near enough for 3.8 % U-235


def read_csv(path):
    """Rows of a CSV file with comment lines starting with '#', as dicts."""
    import csv

    with open(path) as fh:
        return list(csv.DictReader(line for line in fh if not line.startswith("#")))


def power_history_points():
    rows = read_csv(HERE / "fumex2_participants.csv")
    pts = [(float(r["burnup"]), float(r["value"])) for r in rows if r["case"] == "IFA-534.14 rod 18 power"]
    return np.array(pts)


def time_history(points, uo2_mass_per_metre):
    """Times (s) and powers (W/m) from the burnup history of Fig. 20."""
    burnup, power = points[:, 0], points[:, 1] * 1e3
    times = [0.0]
    for i in range(1, len(burnup)):
        d_bu = burnup[i] - burnup[i - 1]
        if d_bu <= 1e-6:
            times.append(times[-1] + HOUR)
            continue
        q = 0.5 * (power[i] + power[i - 1])
        if q < 100.0:
            times.append(times[-1] + HOUR)
            continue
        # MWd/kgUO2 x kgUO2/m = MWd/m, over MW/m gives days.
        times.append(times[-1] + d_bu * uo2_mass_per_metre / (q * 1e-6) * DAY)
    power[0] = max(power[0], 1.0)
    return np.array(times), power


def refab_time(points, times):
    at = np.nonzero(points[:, 0] >= REFAB_BURNUP - 1e-3)[0][0]
    return times[at]


def run(rod_number):
    spec = ROD[rod_number]
    density = 1.0 - spec["porosity"]
    area = np.pi * (0.5 * PELLET_DIAMETER) ** 2
    uo2_mass = area * density * 10963.0
    points = power_history_points()
    times, power = time_history(points, uo2_mass)
    t_refab = refab_time(points, times)

    coolant = fuel.ForcedConvection(inlet_temperature=308.0 + 273.15, pressure=15.5e6, mass_flux=3127.0, rod_pitch=14.3e-3)
    bulk = 308.0 + 273.15 + 15.0
    h = coolant.heat_transfer_coefficient_for(0.5 * CLAD_OD, bulk)

    def wall(linear_heat_rate, t):
        q = max(linear_heat_rate, 0.0)
        if t < t_refab:
            return bulk + q / (np.pi * CLAD_OD * h)
        return 273.15 + 240.0 + 0.4162 * (q / 1e3) ** 0.75

    rod = fuel.FuelRod(
        geometry=fuel.RodGeometry(pellet_outer_radius=0.5 * PELLET_DIAMETER, clad_inner_radius=0.5 * CLAD_ID, clad_outer_radius=0.5 * CLAD_OD, fuel_stack_height=STACK),
        fuel=fuel.UO2Fuel(enrichment=spec["enrichment"], theoretical_density_fraction=density, grain_radius=0.5 * spec["grain_diameter"], total_densification=spec["porosity"] - spec["final_porosity"]),
        cladding=fuel.ZircaloyCladding(),
        fill_gas=fuel.FillGas(pressure=2.15e6, plenum_volume=5.1e-6),
        coolant=fuel.PrescribedCladdingTemperature(temperature=wall, pressure=15.5e6),
        power_history=fuel.PowerHistory(linear_heat_rate=power, time=times, fast_neutron_flux_per_linear_heat_rate=5e13),
        numerics=fuel.RodNumerics(max_time_step=10 * DAY),
        output=fuel.RodOutput(print_input=False, print_steps=False, output_times=np.unique(np.concatenate([np.linspace(0, times[-1], 300), times]))),
    )
    result = rod.run()
    t = np.asarray(result.time)
    burnup = np.asarray(result.burnup_in("MWd/kgU", rod_average=True)) * (1.0 / UO2_PER_U)
    fgr = 100.0 * np.asarray(result.fission_gas_release)
    # The base release includes the gas released in the shutdown at the end
    # of the base irradiation (burst release on cooling): the rods were
    # opened and refilled at the refabrication, so the Halden measurement
    # starts after it.  The refabrication is the two zero-power hours after
    # t_refab (see time_history).
    i = np.searchsorted(t, t_refab + 2 * HOUR - 1.0)
    halden = fgr[-1] - fgr[i] * burnup[i] / burnup[-1]
    return dict(burnup=burnup, fgr=fgr, power=np.asarray(result.rod_average_linear_heat_rate) / 1e3, centre=np.asarray(result.max_fuel_centerline_temperature) - 273.15, base_fgr=fgr[i], halden_fgr=halden)


def compute():
    """The two rods: {"18": run(18), "19": run(19)}."""
    return {str(n): run(n) for n in (18, 19)}


def main():
    out, plot_only = plotstyle.parse_args(sys.argv[1:], ROOT / "docs/_static/figures/fumex2")
    if plot_only:
        cached = plotstyle.load_cache(CACHE)
    else:
        cached = compute()
        plotstyle.save_cache(CACHE, cached)
    results = {n: cached[str(n)] for n in (18, 19)}
    for n, r in results.items():
        print(f"rod {n}: base FGR {r['base_fgr']:.2f} %, Halden FGR {r['halden_fgr']:.2f} % (measured {MEASURED_FGR[n]} %), end burnup {r['burnup'][-1]:.2f} MWd/kgUO2")
    ratio = results[18]["halden_fgr"] / max(results[19]["halden_fgr"], 1e-9)
    print(f"ratio rod 19 / rod 18: {1 / ratio:.2f} (measured {8.89 / 4.68:.2f})")

    fig, ax = plotstyle.new_figure()
    for n, colour, style_ in ((18, DUALMESH, "-"), (19, DUALMESH, "--")):
        r = results[n]
        grain = ROD[n]["grain_diameter"] * 1e6
        ax.plot(r["burnup"], r["fgr"], style_, color=colour, linewidth=2.2, label=f"dualmesh, rod {n} ({grain:g} $\\mu$m)")
    ax.set_xlim(0, 56)
    plotstyle.style(ax, "Rod average burnup (MWd/kgUO$_2$)", "Fission gas release (%)", "IFA-534.14: release over the life of the rods")
    plotstyle.legend(ax, loc="upper left")
    plotstyle.save(fig, out / "fumex2_ifa534_cumulative_fgr.png")

    fig, ax = plotstyle.new_figure()
    x = np.arange(2)
    width = 0.36
    measured = [MEASURED_FGR[18], MEASURED_FGR[19]]
    computed = [results[18]["halden_fgr"], results[19]["halden_fgr"]]
    ax.bar(x - width / 2 - 0.01, measured, width, color=MEASURED, label="Measured (puncture)")
    ax.bar(x + width / 2 + 0.01, computed, width, color=DUALMESH, label="dualmesh")
    for xi, v in zip(x - width / 2 - 0.01, measured):
        ax.text(xi, v + 0.2, f"{v:.2f}", ha="center", fontsize=plotstyle.TICK_SIZE, color=INK)
    for xi, v in zip(x + width / 2 + 0.01, computed):
        ax.text(xi, v + 0.2, f"{v:.2f}", ha="center", fontsize=plotstyle.TICK_SIZE, color=INK)
    ax.set_xticks(x, ["Rod 18 (22.1 $\\mu$m)", "Rod 19 (8.5 $\\mu$m)"])
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylim(0, 1.25 * max(measured + computed))
    plotstyle.style(ax, "", "Fission gas release (%)", "IFA-534.14: release during the Halden irradiation")
    ax.grid(False, axis="x")
    plotstyle.legend(ax, loc="upper left")
    plotstyle.save(fig, out / "fumex2_ifa534_halden_fgr.png")

    lines = ["rod,base_fgr_percent,halden_fgr_percent,measured_halden_fgr_percent"]
    for n, r in results.items():
        lines.append(f"{n},{r['base_fgr']:.2f},{r['halden_fgr']:.2f},{MEASURED_FGR[n]}")
    (HERE / "ifa534_results.csv").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
