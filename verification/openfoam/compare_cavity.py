# SPDX-License-Identifier: LGPL-2.1-or-later
"""Compare dualmesh, OpenFOAM and the Ghia data for the lid-driven cavity.

The OpenFOAM side is produced by ``cavity_openfoam.sh`` (icoFoam, Re = 100 on a
uniform mesh, run to steady state and sampled along the two centrelines).  This
script solves the same problem with dualmesh, on the same mesh, and prints the
three sets of profiles together with the differences.

Reference data: U. Ghia, K. N. Ghia and C. T. Shin, "High-Re solutions for
incompressible flow using the Navier-Stokes equations and a multigrid method",
Journal of Computational Physics 48(3):387-411, 1982, Tables I and II for
Re = 100.

Usage::

    ./cavity_openfoam.sh 64
    python compare_cavity.py --resolution 64
"""

from __future__ import annotations

import argparse
from pathlib import Path

import dualmesh as dm
import numpy as np

# Ghia et al. (1982), Re = 100: u on the vertical centreline (x = 1/2).
GHIA_Y = np.array(
    [
        0.0000,
        0.0547,
        0.0625,
        0.0703,
        0.1016,
        0.1719,
        0.2813,
        0.4531,
        0.5000,
        0.6172,
        0.7344,
        0.8516,
        0.9531,
        0.9609,
        0.9688,
        0.9766,
        1.0000,
    ]
)
GHIA_U = np.array(
    [
        0.00000,
        -0.03717,
        -0.04192,
        -0.04775,
        -0.06434,
        -0.10150,
        -0.15662,
        -0.21090,
        -0.20581,
        -0.13641,
        0.00332,
        0.23151,
        0.68717,
        0.73722,
        0.78871,
        0.84123,
        1.00000,
    ]
)
# v on the horizontal centreline (y = 1/2).
GHIA_X = np.array(
    [
        0.0000,
        0.0625,
        0.0703,
        0.0781,
        0.0938,
        0.1563,
        0.2266,
        0.2344,
        0.5000,
        0.8047,
        0.8594,
        0.9063,
        0.9453,
        0.9531,
        0.9609,
        0.9688,
        1.0000,
    ]
)
GHIA_V = np.array(
    [
        0.00000,
        0.09233,
        0.10091,
        0.10890,
        0.12317,
        0.16077,
        0.17507,
        0.17527,
        0.05454,
        -0.24533,
        -0.22445,
        -0.16914,
        -0.10313,
        -0.08864,
        -0.07391,
        -0.05906,
        0.00000,
    ]
)


def solve_with_dualmesh(resolution: int, reynolds_number: float = 100.0):
    """Lid-driven cavity on the unit square, penalty Navier-Stokes."""
    mesh = dm.generate_rectangle_mesh(
        x_min=0.0,
        x_max=1.0,
        y_min=0.0,
        y_max=1.0,
        num_x_elements=resolution,
        num_y_elements=resolution,
    )
    problem = dm.Problem(mesh, method="dmcdm")
    dm.physics.add_incompressible_flow(
        problem,
        velocities=["u", "v"],
        dynamic_viscosity=1.0,
        density=reynolds_number,
        penalty_parameter=1.0e8,
    )
    for variable in ("u", "v"):
        problem.add_boundary_condition(
            "DirichletBC",
            f"walls_{variable}",
            variable=variable,
            boundary=["left", "right", "bottom"],
            value=0.0,
        )
    problem.add_boundary_condition(
        "DirichletBC", "lid_u", variable="u", boundary="top", value=1.0, scale_with_load=True
    )
    problem.add_boundary_condition("DirichletBC", "lid_v", variable="v", boundary="top", value=0.0)
    problem.solve(load_factors=[0.25, 0.5, 1.0], max_iterations=50)
    return problem


def read_openfoam(path: Path, component: int, length: float = 0.1):
    """Read a sampled OpenFOAM line; returns (normalized coordinate, velocity)."""
    data = np.loadtxt(path)
    return data[:, 0] / length, data[:, 1 + component]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--reynolds-number", type=float, default=100.0)
    parser.add_argument("--case", type=Path, default=Path(__file__).parent / "run" / "cavity")
    parser.add_argument("--plot", action="store_true", help="save a comparison figure")
    arguments = parser.parse_args()

    problem = solve_with_dualmesh(arguments.resolution, arguments.reynolds_number)
    dual_u = problem.sample("u", np.column_stack([np.full_like(GHIA_Y, 0.5), GHIA_Y]))
    dual_v = problem.sample("v", np.column_stack([GHIA_X, np.full_like(GHIA_X, 0.5)]))

    sampled = sorted(arguments.case.glob("postProcessing/sampleDict/*/vertical_U.xy"))
    foam_u = foam_v = None
    if sampled:
        y, u = read_openfoam(sampled[-1], 0)
        foam_u = np.interp(GHIA_Y, y, u)
        horizontal = sampled[-1].with_name("horizontal_U.xy")
        if horizontal.exists():
            x, v = read_openfoam(horizontal, 1)
            foam_v = np.interp(GHIA_X, x, v)
    else:
        print("No OpenFOAM data found; run ./cavity_openfoam.sh first.\n")

    print(
        f"Lid-driven cavity, Re = {arguments.reynolds_number:g}, "
        f"{arguments.resolution} x {arguments.resolution} mesh\n"
    )
    print("u along the vertical centreline")
    header = f"{'y':>8}{'dualmesh':>12}{'OpenFOAM':>12}{'Ghia 1982':>12}"
    print(header)
    for index, y in enumerate(GHIA_Y):
        foam = f"{foam_u[index]:12.5f}" if foam_u is not None else f"{'--':>12}"
        print(f"{y:8.4f}{dual_u[index]:12.5f}{foam}{GHIA_U[index]:12.5f}")
    print(f"\nmaximum |dualmesh - Ghia| = {np.max(np.abs(dual_u - GHIA_U)):.4f}")
    if foam_u is not None:
        print(f"maximum |OpenFOAM - Ghia| = {np.max(np.abs(foam_u - GHIA_U)):.4f}")
        print(f"maximum |dualmesh - OpenFOAM| = {np.max(np.abs(dual_u - foam_u)):.4f}")

    print("\nv along the horizontal centreline")
    print(f"{'x':>8}{'dualmesh':>12}{'OpenFOAM':>12}{'Ghia 1982':>12}")
    for index, x in enumerate(GHIA_X):
        foam = f"{foam_v[index]:12.5f}" if foam_v is not None else f"{'--':>12}"
        print(f"{x:8.4f}{dual_v[index]:12.5f}{foam}{GHIA_V[index]:12.5f}")
    print(f"\nmaximum |dualmesh - Ghia| = {np.max(np.abs(dual_v - GHIA_V)):.4f}")
    if foam_v is not None:
        print(f"maximum |OpenFOAM - Ghia| = {np.max(np.abs(foam_v - GHIA_V)):.4f}")

    if arguments.plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figure, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot(dual_u, GHIA_Y, "-", label="dualmesh")
        if foam_u is not None:
            axes[0].plot(foam_u, GHIA_Y, "--", label="OpenFOAM (icoFoam)")
        axes[0].plot(GHIA_U, GHIA_Y, "o", markersize=4, label="Ghia et al. 1982")
        axes[0].set_xlabel("u"), axes[0].set_ylabel("y"), axes[0].legend()
        axes[1].plot(GHIA_X, dual_v, "-", label="dualmesh")
        if foam_v is not None:
            axes[1].plot(GHIA_X, foam_v, "--", label="OpenFOAM (icoFoam)")
        axes[1].plot(GHIA_X, GHIA_V, "o", markersize=4, label="Ghia et al. 1982")
        axes[1].set_xlabel("x"), axes[1].set_ylabel("v"), axes[1].legend()
        figure.suptitle(f"Lid-driven cavity, Re = {arguments.reynolds_number:g}")
        figure.tight_layout()
        figure.savefig(Path(__file__).parent / "cavity_comparison.png", dpi=150)
        print("\nwrote cavity_comparison.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
