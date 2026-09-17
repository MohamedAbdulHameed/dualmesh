# SPDX-License-Identifier: LGPL-2.1-or-later
"""Compare dualmesh and OpenFOAM for steady conduction in a plate.

The problem is Example 5.4.2 of Reddy's book: a 0.2 m by 0.1 m plate with
conductivity 0.2 W/(m K), held at 500 K on the left and 300 K on the right,
insulated at the bottom, and with T = 500 (1 - 10 x^2) on top.  The OpenFOAM
side (``conduction_openfoam.sh``) marches ``laplacianFoam`` to steady state on
the same grid; the book's tabulated dual mesh values are printed alongside.

Usage::

    ./conduction_openfoam.sh 40 20
    python compare_conduction.py --nx 40 --ny 20
"""

from __future__ import annotations

import argparse
from pathlib import Path

import dualmesh as dm
import numpy as np

# Table 5.4.2 of the book (8 x 8 mesh), at y = 0 and y = 0.05.
BOOK_X = np.arange(1, 8) * 0.025
BOOK_BOTTOM = np.array([482.85, 464.48, 443.88, 420.42, 393.88, 364.48, 332.85])
BOOK_MIDDLE = np.array([485.54, 469.30, 450.01, 426.98, 400.01, 369.30, 335.54])


def solve_with_dualmesh(nx: int, ny: int, method: str = "dmcdm"):
    mesh = dm.generate_rectangle_mesh(
        x_min=0.0, x_max=0.2, y_min=0.0, y_max=0.1, num_x_elements=nx, num_y_elements=ny
    )
    problem = dm.Problem(mesh, method=method)
    problem.add_variable("temperature")
    problem.add_kernel("HeatConduction", variable="temperature", thermal_conductivity=0.2)
    problem.add_boundary_condition(
        "DirichletBC", "left", variable="temperature", boundary="left", value=500.0
    )
    problem.add_boundary_condition(
        "DirichletBC", "right", variable="temperature", boundary="right", value=300.0
    )
    problem.add_boundary_condition(
        "DirichletBC",
        "top",
        variable="temperature",
        boundary="top",
        value=lambda x, y, z, t: 500.0 * (1.0 - 10.0 * x * x),
    )
    problem.solve()
    return problem


def read_openfoam(path: Path):
    data = np.loadtxt(path)
    return data[:, 0], data[:, 1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nx", type=int, default=40)
    parser.add_argument("--ny", type=int, default=20)
    parser.add_argument("--case", type=Path, default=Path(__file__).parent / "run" / "conduction")
    arguments = parser.parse_args()

    problem = solve_with_dualmesh(arguments.nx, arguments.ny)
    sampled = sorted(arguments.case.glob("postProcessing/sampleDict/*/bottom_line_T.xy"))
    lines = {}
    if sampled:
        for name, y in (("bottom_line", 0.0), ("middle_line", 0.05)):
            path = sampled[-1].with_name(f"{name}_T.xy")
            x, temperature = read_openfoam(path)
            lines[name] = (x, temperature, y)
    else:
        print("No OpenFOAM data found; run ./conduction_openfoam.sh first.\n")

    print(f"Steady conduction, {arguments.nx} x {arguments.ny} mesh\n")
    for name, book in (("bottom_line", BOOK_BOTTOM), ("middle_line", BOOK_MIDDLE)):
        y = 0.0 if name == "bottom_line" else 0.05
        print(f"T(x, y = {y})")
        print(f"{'x':>8}{'dualmesh':>12}{'OpenFOAM':>12}{'book 8x8':>12}")
        dual = problem.sample("temperature", np.column_stack([BOOK_X, np.full_like(BOOK_X, y)]))
        foam = None
        if name in lines:
            x, temperature, _ = lines[name]
            foam = np.interp(BOOK_X, x, temperature)
        for index, x_value in enumerate(BOOK_X):
            foam_text = f"{foam[index]:12.3f}" if foam is not None else f"{'--':>12}"
            print(f"{x_value:8.3f}{dual[index]:12.3f}{foam_text}{book[index]:12.3f}")
        if foam is not None:
            print(f"maximum |dualmesh - OpenFOAM| = {np.max(np.abs(dual - foam)):.4f} K")
        print()

    # Global check: the heat entering through the prescribed-temperature faces
    # must sum to zero.  A corner node belongs to two boundaries and carries a
    # single reaction, so the nodes are collected in a dictionary first rather
    # than adding the three totals (which would count corners twice).
    by_node = {}
    for boundary in ("left", "right", "top"):
        by_node.update(dict(problem.reactions("temperature", boundary)))
    print(f"heat through the left face:  {problem.total_reaction('temperature', 'left'):12.4f} W/m")
    print(f"sum over all boundary nodes: {sum(by_node.values()):12.3e} W/m (zero by conservation)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
