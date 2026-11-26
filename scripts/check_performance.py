# SPDX-License-Identifier: LGPL-2.1-or-later
"""Check the results of scripts/benchmark.py against time budgets.

The gate catches a regression of an order of magnitude, the class that once
made copying the whole derivative array of an ADReal the dominant cost of
assembly, not a change of a few percent: the budgets are about ten times the
time of the calibration machine (an Apple-silicon laptop, one thread, best of
three runs), which a GitHub runner roughly doubles.  ``--slack`` multiplies
every budget, for a first measurement on a new machine.  A benchmark whose
key is not in the budget table fails the check, so that a new benchmark is
added to it consciously.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Calibration of 2026-10-17, dualmesh 0.3.0 development, quick assembly
# benchmarks (32 x 32 Quad4), best of three, one thread.
BUDGETS = {
    ("assembly", "heat conduction, 32 x 32 Quad4", "fem"): 0.05,
    ("assembly", "heat conduction, 32 x 32 Quad4", "dmcdm"): 0.05,
    ("assembly", "plane elasticity, 32 x 32 Quad4", "fem"): 0.05,
    ("assembly", "plane elasticity, 32 x 32 Quad4", "dmcdm"): 0.05,
    ("assembly", "pressure-velocity flow, 32 x 32 Quad4", "fem"): 0.15,
    ("assembly", "pressure-velocity flow, 32 x 32 Quad4", "dmcdm"): 0.15,
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("results", type=Path, help="the JSON written by scripts/benchmark.py")
    parser.add_argument(
        "--slack",
        type=float,
        default=1.0,
        help="multiplier of every budget (default 1)",
    )
    args = parser.parse_args(argv)
    record = json.loads(args.results.read_text())
    rows = {(r["group"], r["case"], r["method"]): r for r in record["results"]}

    failures = []
    print(f"{'benchmark':60s} {'budget':>8s} {'measured':>9s} {'ratio':>6s}")
    print("-" * 86)
    for key, budget in BUDGETS.items():
        if key not in rows:
            failures.append(f"{key}: in the budgets, not in the results")
            continue
        seconds = rows[key]["seconds"]
        budget *= args.slack
        ratio = seconds / budget
        verdict = "" if seconds <= budget else "   OVER"
        print(f"{key[1] + ', ' + key[2]:60s} {budget:8.3f} {seconds:9.3f} {ratio:6.2f}{verdict}")
        if seconds > budget:
            failures.append(f"{key}: {seconds:.3f} s over the budget {budget:.3f} s")
    for key in rows:
        if key not in BUDGETS:
            failures.append(f"{key}: measured but has no budget; add it to BUDGETS")
            print(f"{key[1] + ', ' + key[2]:60s} {'-':>8s} {rows[key]['seconds']:9.3f}   NEW")

    if failures:
        print()
        for failure in failures:
            print(f"performance gate: {failure}")
        return 1
    print("\nperformance gate: every benchmark within its budget")
    return 0


if __name__ == "__main__":
    sys.exit(main())
