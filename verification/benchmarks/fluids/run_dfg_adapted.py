# SPDX-License-Identifier: LGPL-2.1-or-later
"""Benchmark F1 on adapted meshes: the DFG steady flow around a cylinder, 2D-1, at Re = 20
(Schaefer and Turek [SchaeferTurek1996]), solved by the adaptive loop.

The flow and its reference values are those of run_dfg_steady.py.  The loop starts from a mesh of
size 0.02 on the cylinder, marks by the gradient-recovery indicators of the two velocity
components with Doerfler's fraction 0.3, and bisects the marked triangles.  After each refinement
the nodes of the cylinder move onto the circle, so that refining there also refines the geometry.
The drag and lift coefficients and the pressure difference are recorded on every mesh, and their
errors against the number of unknowns are plotted with those of the uniform meshes of
run_dfg_steady.py, when its cache is there.

    python run_dfg_adapted.py [output directory] [--plot-only] [--method dmcdm | fem]
    mpirun -n 4 python run_dfg_adapted.py [output directory]

Under mpirun every process builds the same initial mesh and holds the same hierarchy, and the
problems are distributed; the first process writes the cache and the figures.
"""

import sys
import tempfile
import time
from pathlib import Path

import numpy

here = Path(__file__).resolve().parent
sys.path.insert(0, str(here.parents[0]))
sys.path.insert(0, str(here))
import dfg_mesh  # noqa: E402
import plotstyle  # noqa: E402

reference = {"drag": 5.57953523384, "lift": 0.010618948146, "dp": 0.11752016697}
default_methods = ["dmcdm", "fem"]
initial_size = 0.02  # Element size on the cylinder of the first mesh
cycles = 9
fraction = 0.3
scale = 2.0 / (0.2**2 * 0.1)  # 2 / (U_mean^2 D)
default_out = here.parents[2] / "docs" / "_static" / "figures" / "fluids"


def on_cylinder(point):
    """The point of the cylinder of radius 0.05 around (0.2, 0.2) nearest to ``point``."""
    x, y = point[0] - dfg_mesh.center[0], point[1] - dfg_mesh.center[1]
    distance = (x * x + y * y) ** 0.5
    return (
        dfg_mesh.center[0] + dfg_mesh.radius * x / distance,
        dfg_mesh.center[1] + dfg_mesh.radius * y / distance,
        0.0,
    )


def compute(methods):
    import dualmesh

    data = {}
    with tempfile.TemporaryDirectory() as work:
        start = dualmesh.read_mesh(str(dfg_mesh.build(initial_size, Path(work) / "dfg.msh", 1)))
    for method in [m for m in methods if m in default_methods]:
        flows = {}

        def build(mesh, method=method, flows=flows):
            problem = dualmesh.Problem(mesh, method=method.upper())
            flow = problem.add_physics(
                "fluid_dynamics",
                density=1.0,
                dynamic_viscosity=1e-3,
                formulation="equal_order",
                viscous_form="laplacian",
            )
            flow.add_boundary_condition(
                ["velocity_x = 4*0.3*y*(0.41 - y)/0.41^2", "velocity_y = 0"], boundary="inlet"
            )
            flow.add_boundary_condition(
                ["velocity_x = 0", "velocity_y = 0"], boundary=["walls", "cylinder"]
            )
            flows[id(problem)] = flow
            return problem

        rows = []
        started = time.perf_counter()

        def record(_cycle, problem, _indicators, rows=rows, flows=flows, started=started):
            pressure = problem.sample("pressure", [[0.15, 0.2], [0.25, 0.2]])
            force = flows[id(problem)].force("cylinder")
            rows.append(
                {
                    "drag": scale * force["x"],
                    "lift": scale * force["y"],
                    "dp": pressure[0] - pressure[1],
                    "unknowns": problem.num_active_dofs(),
                    "time": time.perf_counter() - started,
                }
            )

        dualmesh.solve_with_adaptive_refinement(
            build,
            start,
            ["velocity_x", "velocity_y"],
            num_cycles=cycles,
            marker=lambda indicators: dualmesh.mark_by_error_fraction(indicators, fraction),
            callback=record,
            report="none",
            boundaries={"cylinder": on_cylinder},
        )
        data[f"{method}, adapted"] = {key: numpy.asarray([r[key] for r in rows]) for key in rows[0]}
        if dualmesh.parallel.is_root():
            print(
                f"{method}, adapted: C_D {data[f'{method}, adapted']['drag']}, "
                f"C_L {data[f'{method}, adapted']['lift']}, dp {data[f'{method}, adapted']['dp']}",
                flush=True,
            )
    return data


def plot(data, out):
    uniform_cache = here / "run_dfg_steady_cache.npz"
    uniform = plotstyle.load_cache(uniform_cache) if uniform_cache.exists() else {}
    colors = {"dmcdm": plotstyle.dualmesh, "fem": plotstyle.second}
    for key, label, name in (
        ("drag", "error of the drag coefficient", "dfg_2d1_adapted_drag_error.png"),
        ("lift", "error of the lift coefficient", "dfg_2d1_adapted_lift_error.png"),
        ("dp", "error of the pressure difference", "dfg_2d1_adapted_pressure_error.png"),
    ):
        fig, ax = plotstyle.new_figure()
        for method, color in colors.items():
            for curves, style, suffix in ((data, "o-", ", adapted"), (uniform, "s--", "")):
                d = curves.get(f"{method}{suffix}")
                if d is None:
                    continue
                ax.loglog(
                    d["unknowns"],
                    numpy.abs(d[key] - reference[key]) / abs(reference[key]),
                    style,
                    color=color,
                    lw=2,
                    ms=6,
                    label=f"{method.upper()}, {'adapted' if suffix else 'uniform'}",
                )
        plotstyle.style(ax, "unknowns", f"relative {label}")
        plotstyle.legend_below(ax, ncol=2)
        plotstyle.save(fig, out / name)


def table(data):
    for method, d in data.items():
        for i in range(len(d["unknowns"])):
            print(
                f"{method:14s} {int(d['unknowns'][i]):7d} unknowns: C_D {d['drag'][i]:.5f} "
                f"({100 * (d['drag'][i] / reference['drag'] - 1):+.3f} %), C_L {d['lift'][i]:.5f}, "
                f"dp {d['dp'][i]:.5f} ({100 * (d['dp'][i] / reference['dp'] - 1):+.3f} %), "
                f"{d['time'][i]:.1f} s"
            )


if __name__ == "__main__":
    import dualmesh

    arguments = list(sys.argv[1:])
    methods = list(default_methods)
    if "--method" in arguments:
        i = arguments.index("--method")
        methods = [arguments[i + 1]]
        del arguments[i : i + 2]
    out, plot_only = plotstyle.parse_arguments(arguments, default_out)
    cache = plotstyle.cache_path(__file__)
    if plot_only:
        data = plotstyle.load_cache(cache)
    else:
        data = compute(methods)
    if dualmesh.parallel.is_root():
        if not plot_only:
            if cache.exists():
                data = {**plotstyle.load_cache(cache), **data}
            plotstyle.save_cache(cache, data)
        table(data)
        plot(data, out)
