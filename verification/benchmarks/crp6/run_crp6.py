# SPDX-License-Identifier: LGPL-2.1-or-later
"""Run the CRP-6 TRISO benchmark cases 1 to 8 and compare with the participants.

The benchmark is the coordinated research project on TRISO fuel of the
IAEA, reported in IAEA-TECDOC-1674 (2012), section 9.2.  Cases 1 to 4c have
closed form solutions, given in the report (Eqs. 9.24 to 9.29).  Cases 4d to
8 are compared with the curves of the eight participating codes, read from
the vector figures of the report into ``crp6_participants.csv``.

Usage::

    python run_crp6.py [output directory for the figures] [--plot-only]

The default output directory is ``docs/_static/figures/crp6`` of the
repository.  The script prints a table of the key values and writes
``crp6_results.csv`` beside itself.  The solutions are saved in
``run_crp6_cache.npz``, and ``--plot-only`` redraws the figures (and rewrites
the table) from it.  Each stress is drawn in a figure of its own,
``crp6_case<case>_<quantity>.png``.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from dualmesh.fuel import triso

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import plotstyle  # noqa: E402
from plotstyle import DUALMESH  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CACHE = plotstyle.cache_path(__file__)
CASES = ("1", "2", "3", "4a", "4b", "4c", "4d", "5", "6", "7", "8")

# Colours of the reference categorical palette: dualmesh in slot 1, the
# closed form solutions in the slot of the measurements, participants in a
# recessive grey.
ANALYTIC = plotstyle.MEASURED
PARTICIPANT = plotstyle.REFERENCE_GREY

# Closed form solutions of the report, section 9.2.3 (MPa).
ANALYTIC_VALUES = {
    ("1", "SiC", "start"): 125.19,
    ("2", "IPyC", "start"): 50.20,
    ("3", "IPyC", "start"): 8.8,
    ("3", "SiC", "start"): 104.4,
    ("3", "IPyC/SiC", "start"): -18.8,
    ("4a", "IPyC", "end"): 926.80,
    ("4a", "SiC", "end"): -845.71,
    ("4b", "IPyC", "end"): -25.0,
    ("4b", "SiC", "end"): 139.34,
    ("4c", "IPyC", "end"): 26.05,
    ("4c", "SiC", "end"): 86.5,
}

QUANTITY = {"IPyC_tangential": ("IPyC", "tangential"), "SiC_tangential": ("SiC", "tangential"), "interface_radial": ("IPyC/SiC", "radial")}


def load_participants():
    path = HERE / "crp6_participants.csv"
    data: dict[tuple, dict[str, list]] = {}
    with path.open() as fh:
        rows = csv.DictReader(line for line in fh if not line.startswith("#"))
        for row in rows:
            if row["participant"].startswith("unlabelled"):
                continue
            key = (row["case"], row["quantity"])
            curve = data.setdefault(key, {}).setdefault(row["participant"], [[], []])
            curve[0].append(float(row["fluence"]))
            curve[1].append(float(row["stress"]))
    return data


def _break_jumps(f, s):
    """Insert gaps where the digitised points of a curve jump back in fluence.

    Where two participants share a line style in the report their points are
    joined into one sequence.  A gap at every backward jump keeps the plot
    from drawing a line between the two curves.
    """
    jumps = np.where((np.diff(f) < -0.02) | (np.diff(f) > 0.25))[0] + 1
    return np.insert(f, jumps, np.nan), np.insert(s, jumps, np.nan)


def series(result, quantity):
    layer, kind = QUANTITY[quantity]
    if kind == "radial":
        values = result.interface_radial_stress[layer]
    else:
        values = result.inner_tangential_stress[layer]
    return result.fluence * 1e-25, values * 1e-6


FLUENCE_LABEL = r"Fast fluence ($10^{25}$ n/m$^2$, $E > 0.18$ MeV)"
TITLES = {"IPyC_tangential": "IPyC, inner surface, tangential stress", "SiC_tangential": "SiC, inner surface, tangential stress", "interface_radial": "IPyC/SiC interface, radial stress"}
SUFFIX = {"IPyC_tangential": "ipyc_tangential", "SiC_tangential": "sic_tangential", "interface_radial": "interface_radial"}


def plot_case(case, result, participants, out: Path):
    """One figure per stress: the IPyC and SiC tangential stresses, and for
    cases 4a to 4d the radial stress at the IPyC/SiC interface."""
    quantities = ["IPyC_tangential", "SiC_tangential"]
    if case.startswith("4"):
        quantities.append("interface_radial")
    for quantity in quantities:
        fig, ax = plotstyle.new_figure()
        curves = participants.get((case, quantity), {})
        for i, (_, (f, s)) in enumerate(sorted(curves.items())):
            f, s = _break_jumps(np.asarray(f), np.asarray(s))
            ax.plot(f, s, color=PARTICIPANT, linewidth=1.0, label="CRP-6 participants" if i == 0 else None)
        x, y = series(result, quantity)
        ax.plot(x, y, color=DUALMESH, linewidth=2.2, label="dualmesh")
        layer = QUANTITY[quantity][0]
        ref = ANALYTIC_VALUES.get((case, layer, "end"))
        if ref is not None and QUANTITY[quantity][1] == "tangential":
            ax.plot([3.0], [ref], "o", color=ANALYTIC, markersize=9, markeredgecolor="white", markeredgewidth=1.5, clip_on=False, zorder=5, label="Closed form (report)")
        plotstyle.style(ax, FLUENCE_LABEL, "Stress (MPa)", f"CRP-6 case {case}: {TITLES[quantity]}")
        ax.set_xlim(0.0, 3.0)
        plotstyle.legend_below(ax, ncol=3)
        plotstyle.save(fig, out / f"crp6_case{case}_{SUFFIX[quantity]}.png")


def participant_range(participants, case, quantity, statistic):
    values = []
    for f, s in participants.get((case, quantity), {}).values():
        f = np.asarray(f)
        s = np.asarray(s)
        if statistic == "start":
            return None
        if statistic == "end" and f.max() < 2.95:
            continue  # the curve stops early in the report
        if f.max() < 0.8:
            continue  # stops before the extremes, which lie at 0.5 to 0.75
        if statistic == "max":
            values.append(s.max())
        elif statistic == "min":
            values.append(s.min())
        else:
            values.append(s[np.argmax(f)])
    if not values:
        return None
    return min(values), max(values)


def compute():
    """Solve the eleven cases: {case: {end_fluence, fluence, tangential,
    radial}} with the arrays of ``triso.ParticleResult``."""
    solutions = {}
    for case in CASES:
        particle, history = triso.crp6_case(case)
        result = triso.TrisoParticleModel(particle=particle, history=history).run()
        solutions[case] = dict(end_fluence=history.end_fluence, fluence=result.fluence, tangential=result.inner_tangential_stress, radial=result.interface_radial_stress)
    return solutions


def main():
    out, plot_only = plotstyle.parse_args(sys.argv[1:], ROOT / "docs/_static/figures/crp6")
    if plot_only:
        solutions = plotstyle.load_cache(CACHE)
    else:
        solutions = compute()
        plotstyle.save_cache(CACHE, solutions)
    participants = load_participants()
    rows = []
    for case in CASES:
        cached = solutions[case]
        history = SimpleNamespace(end_fluence=cached["end_fluence"])
        result = SimpleNamespace(fluence=cached["fluence"], inner_tangential_stress=cached["tangential"], interface_radial_stress=cached["radial"])
        for layer, values in result.inner_tangential_stress.items():
            if layer == "OPyC":
                continue
            quantity = f"{layer}_tangential"
            for statistic, value in (("start", values[0]), ("max", values.max()), ("min", values.min()), ("end", values[-1])):
                if history.end_fluence == 0.0 and statistic != "start":
                    continue
                ref = ANALYTIC_VALUES.get((case, layer, statistic))
                rng = participant_range(participants, case, quantity, statistic)
                rows.append((case, layer, "tangential", statistic, value * 1e-6, ref, rng))
        for name, values in result.interface_radial_stress.items():
            if name != "IPyC/SiC":
                continue
            for statistic, value in (("start", values[0]), ("end", values[-1])):
                if history.end_fluence == 0.0 and statistic != "start":
                    continue
                ref = ANALYTIC_VALUES.get((case, name, statistic))
                rng = participant_range(participants, case, "interface_radial", statistic)
                rows.append((case, name, "radial", statistic, value * 1e-6, ref, rng))
        if history.end_fluence > 0.0:
            plot_case(case, result, participants, out)

    with (HERE / "crp6_results.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["case", "location", "component", "statistic", "dualmesh_MPa", "closed_form_MPa", "participants_low_MPa", "participants_high_MPa"])
        for case, loc, comp, stat, value, ref, rng in rows:
            writer.writerow([case, loc, comp, stat, f"{value:.2f}", "" if ref is None else ref, "" if rng is None else f"{rng[0]:.1f}", "" if rng is None else f"{rng[1]:.1f}"])
    print(f"{'case':>4} {'where':>9} {'comp':>10} {'stat':>5} {'dualmesh':>9} {'closed':>8} {'participants':>16}")
    for case, loc, comp, stat, value, ref, rng in rows:
        r = "" if ref is None else f"{ref:.2f}"
        p = "" if rng is None else f"{rng[0]:.1f} to {rng[1]:.1f}"
        print(f"{case:>4} {loc:>9} {comp:>10} {stat:>5} {value:9.2f} {r:>8} {p:>16}")


if __name__ == "__main__":
    main()
