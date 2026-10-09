# SPDX-License-Identifier: LGPL-2.1-or-later
"""The wrench of examples/wrench on a sequence of finer meshes, solved by the
sparse direct solver and by conjugate gradients preconditioned with algebraic
multigrid (the default of linear_solver="automatic" in three dimensions).

For every mesh the script records the number of unknowns, the time of the
linear solves, the number of conjugate gradient iterations, and the results
the tutorial checks: the reaction on the jaws, the moment about the nut, and
the bending stress of the handle at x = 60 mm against beam theory.  The
meshes are made with Gmsh (``pip install gmsh``).

    python run_wrench_scaling.py [output directory] [--plot-only]
"""

import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0]))
sys.path.insert(0, str(HERE.parents[2] / "examples" / "wrench"))
import plotstyle  # noqa: E402

SIZES_MM = [3.0, 2.0, 1.5, 1.2, 1.0]  # largest element edge of each mesh
DIRECT_UP_TO = 250000  # unknowns; the direct solver is skipped above this
DEFAULT_OUT = HERE.parents[2] / "docs" / "_static" / "figures" / "solvers"


def solve(mesh_file, linear_solver):
    import dualmesh as dm
    import wrench

    mesh = dm.read_mesh(str(mesh_file))
    problem = dm.Problem(mesh, method="fem")
    problem.add_physics("solid_mechanics", "solid", displacements=["u", "v", "w"], formulation="three_dimensional", youngs_modulus=wrench.YOUNGS_MODULUS, poissons_ratio=wrench.POISSONS_RATIO)
    problem.add_boundary_condition("fixed_constraint", "jaws", displacements=["u", "v", "w"])
    problem.add_boundary_condition("traction_boundary_condition", "grip", variable="v", total_force=-wrench.FORCE)
    start = time.perf_counter()
    result = problem.solve(linear_solver=linear_solver, nonlinear_solver="linear")
    elapsed = time.perf_counter() - start
    checks, _ = wrench.check(problem)
    return problem.num_active_dofs(), elapsed, result.linear_iterations, checks


def compute():
    import make_wrench_mesh

    data = {"size_mm": [], "unknowns": [], "time_amg": [], "iterations_amg": [], "time_lu": [], "reaction": [], "moment": [], "bending_stress": [], "von_mises_max": []}
    with tempfile.TemporaryDirectory() as work:
        for size in SIZES_MM:
            path = make_wrench_mesh.build(size, order=2, path=Path(work) / f"wrench_{size}.msh")
            unknowns, t_amg, iterations, checks = solve(path, "automatic")
            t_lu = np.nan
            if unknowns <= DIRECT_UP_TO:
                _, t_lu, _, _ = solve(path, "lu")
            data["size_mm"].append(size)
            data["unknowns"].append(unknowns)
            data["time_amg"].append(t_amg)
            data["iterations_amg"].append(iterations)
            data["time_lu"].append(t_lu)
            data["reaction"].append(checks["reaction on the jaws (N)"][0])
            data["moment"].append(checks["moment about the nut (N m)"][0])
            data["bending_stress"].append(checks["bending stress at x = 60 mm (MPa)"][0])
            data["von_mises_max"].append(checks["largest von Mises stress (MPa)"][0])
            print(f"h = {size} mm: {unknowns} unknowns, multigrid {t_amg:.1f} s ({iterations} iterations), direct {t_lu:.1f} s, bending stress {data['bending_stress'][-1]:.3f} MPa, peak von Mises {data['von_mises_max'][-1]:.1f} MPa", flush=True)
    return {key: np.asarray(value) for key, value in data.items()}


def plot(data, out):
    fig, ax = plotstyle.new_figure()
    n = data["unknowns"]
    ax.loglog(n, data["time_amg"], "o-", color=plotstyle.DUALMESH, lw=2, ms=7, label="conjugate gradients with algebraic multigrid")
    direct = np.isfinite(data["time_lu"])
    ax.loglog(n[direct], data["time_lu"][direct], "s-", color=plotstyle.REFERENCE_GREY, lw=2, ms=7, label="sparse direct solver (LU)")
    plotstyle.style(ax, "unknowns", "wall time of the solve (s)")
    plotstyle.legend(ax, loc="upper left")
    plotstyle.save(fig, out / "wrench_solver_time.png")

    fig, ax = plotstyle.new_figure()
    ax.semilogx(n, data["iterations_amg"], "o-", color=plotstyle.DUALMESH, lw=2, ms=7)
    ax.set_ylim(0, max(60, 1.2 * max(data["iterations_amg"])))
    plotstyle.style(ax, "unknowns", "conjugate gradient iterations")
    plotstyle.save(fig, out / "wrench_solver_iterations.png")


if __name__ == "__main__":
    out, plot_only = plotstyle.parse_args(sys.argv[1:], DEFAULT_OUT)
    cache = plotstyle.cache_path(__file__)
    if plot_only:
        data = plotstyle.load_cache(cache)
    else:
        data = compute()
        plotstyle.save_cache(cache, data)
    plot(data, out)
