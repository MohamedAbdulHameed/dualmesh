# SPDX-License-Identifier: LGPL-2.1-or-later
"""Group-averaged damage energy cross sections from ENDF/B-VIII.0.

The source is the NJOY-2016 processing of ENDF/B-VIII.0 for natural elements
distributed by the IAEA Nuclear Data Section with the Coordinated Research
Project "Primary Radiation Damage Cross Sections"
(https://www-nds.iaea.org/CRPdpa/, file ENDFB-8.0/Ele0/endfb8-dm.tar.bz2).
Each element's PENDF file holds, in MF3 MT444, the damage energy production
cross section (eV barn): the recoil energy of every reaction partitioned
between atoms and electrons with Robinson's form of Lindhard's theory, times
the reaction cross section.  MT1 (the total cross section) is kept for a
check of the elastic range.

This script reads MT444 and MT1 and averages them over 50 groups per decade,
equally spaced in lethargy from 1e-5 eV to 20 MeV, with a flat weight per unit
lethargy.  Run it with the directory that holds the pendf/*0p.asc files; it
writes python/dualmesh/fuel/data/damage_energy_endfb8.csv.
"""

import re
import sys
from pathlib import Path

import numpy as np

ELEMENTS = ["C", "Al", "Si", "Ti", "Cr", "Fe", "Ni", "Zr", "Nb", "Mo", "W", "U"]


def _float(field: str) -> float:
    field = field.strip()
    match = re.match(r"([+-]?[\d.]+)([+-]\d+)$", field)
    return float(match.group(1) + "e" + match.group(2)) if match else float(field)


def read_mf3(path: Path, mt: int):
    lines = [
        line
        for line in path.read_text().splitlines()
        if len(line) >= 75 and line[70:72].strip() == "3" and line[72:75].strip() == str(mt)
    ]
    control = lines[1]
    nr, npoints = int(control[44:55]), int(control[55:66])
    values = []
    for line in lines[2 + (2 * nr + 5) // 6 :]:
        values += [
            _float(line[11 * k : 11 * k + 11])
            for k in range(6)
            if line[11 * k : 11 * k + 11].strip()
        ]
    values = np.array(values[: 2 * npoints])
    return values[0::2], values[1::2]


def group_average(E, s, edges):
    """Average of a linearly interpolated table over each group, per unit
    lethargy (trapezoidal rule on a refined grid)."""
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        inside = E[(lo < E) & (hi > E)]
        grid = np.unique(np.concatenate([[lo, hi], inside, np.geomspace(lo, hi, 9)]))
        values = np.interp(grid, E, s, left=0.0, right=s[-1])
        u = np.log(grid)
        out.append(np.trapezoid(values, u) / (u[-1] - u[0]))
    return np.array(out)


def main(folder: str) -> None:
    edges = np.geomspace(1e-5, 2e7, int(round(50 * np.log10(2e7 / 1e-5))) + 1)
    columns = {"lower_eV": edges[:-1], "upper_eV": edges[1:]}
    for element in ELEMENTS:
        path = Path(folder) / "pendf" / f"{element}0p.asc"
        E, damage = read_mf3(path, 444)
        columns[f"{element}_damage_eV_b"] = group_average(E, damage, edges)
        E, total = read_mf3(path, 1)
        columns[f"{element}_total_b"] = group_average(E, total, edges)
    out = Path(__file__).resolve().parents[2] / "python/dualmesh/fuel/data/damage_energy_endfb8.csv"
    header = ",".join(columns)
    table = np.column_stack(list(columns.values()))
    note = (
        "# Group-averaged damage energy (MF3 MT444, eV b) and total (MT1, b) cross sections of\n"
        "# natural elements, ENDF/B-VIII.0 processed with NJOY-2016 by the IAEA NDS (CRP\n"
        "# Primary Radiation Damage Cross Sections), 50 groups per decade, flat in lethargy.\n"
        "# Made by verification/correlations/extract_damage_energy.py.\n"
    )
    np.savetxt(out, table, delimiter=",", header=header, comments=note, fmt="%.6e")
    print(f"wrote {out}: {table.shape[0]} groups, {len(ELEMENTS)} elements")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
