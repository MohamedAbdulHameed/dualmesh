# SPDX-License-Identifier: LGPL-2.1-or-later
"""The wrench of examples/wrench on a sequence of finer meshes, solved by the
sparse direct solver and by GMRES preconditioned with smoothed aggregation
multigrid (GAMG) and the rigid body modes, the default for a solid in three
dimensions above 4000 unknowns.

For every mesh the script records the number of unknowns, the time of the
linear solves, the number of GMRES iterations, and the results
the tutorial checks: the reaction on the jaws, the moment about the nut, and
the bending stress of the handle at x = 60 mm against beam theory.  The
meshes are made with Gmsh (``pip install gmsh``).

    python run_wrench_scaling.py [output directory] [--plot-only]
"""

import sys
import tempfile
import time
from pathlib import Path

import numpy

here = Path(__file__).resolve().parent
sys.path.insert(0, str(here.parents[0]))
sys.path.insert(0, str(here.parents[2] / "examples" / "wrench"))
import plotstyle  # noqa: E402

sizes_mm = [3.0, 2.0, 1.5, 1.2, 1.0]  # Largest element edge of each mesh
direct_up_to = 250000  # Unknowns; the direct solver is skipped above this
default_out = here.parents[2] / "docs" / "_static" / "figures" / "solvers"


def solve(mesh_file, petsc_options):
    import dualmesh
    import wrench

    mesh = dualmesh.read_mesh(str(mesh_file))
    problem = dualmesh.Problem(mesh, method="fem")
    problem.add_physics(
        "solid_mechanics",
        formulation="three_dimensional",
        youngs_modulus=wrench.youngs_modulus,
        poissons_ratio=wrench.poissons_ratio,
    )
    problem.add_boundary_condition(
        "Dirichlet", "jaws", variable=["disp_x", "disp_y", "disp_z"], value=0.0
    )
    problem.add_boundary_condition("traction", "grip", variable="disp_y", total_force=-wrench.force)
    start = time.perf_counter()
    result = problem.solve(petsc_options=petsc_options, nonlinear_solver="linear")
    elapsed = time.perf_counter() - start
    checks, _ = wrench.check(problem)
    return problem.num_active_dofs(), elapsed, result.linear_iterations, checks


def compute():
    import make_wrench_mesh

    data = {
        "size_mm": [],
        "unknowns": [],
        "time_amg": [],
        "iterations_amg": [],
        "time_lu": [],
        "reaction": [],
        "moment": [],
        "bending_stress": [],
        "von_mises_max": [],
    }
    with tempfile.TemporaryDirectory() as work:
        for size in sizes_mm:
            path = make_wrench_mesh.build(size, order=2, path=Path(work) / f"wrench_{size}.msh")
            unknowns, t_AMG, iterations, checks = solve(path, None)
            t_LU = numpy.nan
            if unknowns <= direct_up_to:
                _, t_LU, _, _ = solve(path, "-ksp_type preonly -pc_type lu")
            data["size_mm"].append(size)
            data["unknowns"].append(unknowns)
            data["time_amg"].append(t_AMG)
            data["iterations_amg"].append(iterations)
            data["time_lu"].append(t_LU)
            data["reaction"].append(checks["reaction on the jaws (N)"][0])
            data["moment"].append(checks["moment about the nut (N m)"][0])
            data["bending_stress"].append(checks["bending stress at x = 60 mm (MPa)"][0])
            data["von_mises_max"].append(checks["largest von Mises stress (MPa)"][0])
            print(
                f"h = {size} mm: {unknowns} unknowns, multigrid {t_AMG:.1f} s ({iterations} iterations), direct {t_LU:.1f} s, bending stress {data['bending_stress'][-1]:.3f} MPa, peak von Mises {data['von_mises_max'][-1]:.1f} MPa",
                flush=True,
            )
    return {key: numpy.asarray(value) for key, value in data.items()}


def plot(data, out):
    fig, ax = plotstyle.new_figure()
    n = data["unknowns"]
    ax.loglog(
        n,
        data["time_amg"],
        "o-",
        color=plotstyle.dualmesh,
        lw=2,
        ms=7,
        label="GMRES with smoothed aggregation",
    )
    direct = numpy.isfinite(data["time_lu"])
    ax.loglog(
        n[direct],
        data["time_lu"][direct],
        "s-",
        color=plotstyle.reference_grey,
        lw=2,
        ms=7,
        label="sparse direct solver (LU)",
    )
    plotstyle.style(ax, "unknowns", "wall time of the solve (s)")
    plotstyle.legend(ax, location="upper left")
    plotstyle.save(fig, out / "wrench_solver_time.png")

    fig, ax = plotstyle.new_figure()
    ax.semilogx(n, data["iterations_amg"], "o-", color=plotstyle.dualmesh, lw=2, ms=7)
    ax.set_ylim(0, max(60, 1.2 * max(data["iterations_amg"])))
    plotstyle.style(ax, "unknowns", "GMRES iterations")
    plotstyle.save(fig, out / "wrench_solver_iterations.png")


if __name__ == "__main__":
    out, plot_only = plotstyle.parse_arguments(sys.argv[1:], default_out)
    cache = plotstyle.cache_path(__file__)
    if plot_only:
        data = plotstyle.load_cache(cache)
    else:
        data = compute()
        plotstyle.save_cache(cache, data)
    plot(data, out)
