# SPDX-License-Identifier: LGPL-2.1-or-later
"""Benchmark F2: the DFG periodic vortex shedding behind a cylinder, 2D-2, at
Re = 100 (Schaefer and Turek [SchaeferTurek1996], reference values of the
FEATFLOW benchmark pages, level 6 with the step 1/200: largest drag coefficient
3.2200, largest lift coefficient 0.9859, Strouhal number 0.30188).

The geometry is that of F1 (``dfg_mesh.py``), with the inflow
u = 4 U y (0.41 - y) / 0.41^2 of U = 1.5, so that the mean velocity is 1 and
Re = 100 with the viscosity 0.001.  The flow starts from rest and is advanced
by the projection time integration (BDF2 and EXT2) at a Courant number of one
half, with the pressure zero on the outflow, the form of the do-nothing
condition that the splitting takes.  The forces on the cylinder come from the
reactions at every step; the last full cycles of the lift give the Strouhal
number St = D f / U_mean and the largest drag and lift coefficients.

    mpirun -n 4 python run_dfg_unsteady.py [output directory] [--plot-only] [--method fem]

One method per run keeps each run under ten minutes on four processes; the
results of the runs are merged in the cache.

On several processes the problem is distributed; the first process prints and
writes the results.
"""

import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0]))
sys.path.insert(0, str(HERE))
import dfg_mesh  # noqa: E402
import plotstyle  # noqa: E402

REFERENCE = {"drag": 3.2200, "lift": 0.9859, "strouhal": 0.30188}
SIZES = [0.005, 0.0025]
METHODS = ["dmcdm", "fem", "hfvm"]
# The shedding is periodic from about t = 7; the last three cycles before t = 10 are measured.
END_TIME = 10.0
SCALE = 2.0 / (1.0**2 * 0.1)  # 2 / (U_mean^2 D)
DEFAULT_OUT = HERE.parents[2] / "docs" / "_static" / "figures" / "fluids"


def run(mesh, method):
    import dualmesh as dm

    problem = dm.Problem(mesh, method=method)
    problem.add_physics(
        "incompressible_flow",
        "flow",
        velocities=["u", "v"],
        density=1.0,
        dynamic_viscosity=1e-3,
        formulation="pressure",
        time_integration="projection",
    )
    problem.add_boundary_condition(
        "Dirichlet_boundary_condition",
        "inflow_u",
        variable="u",
        boundary="inlet",
        value="4*1.5*y*(0.41 - y)/0.41^2",
    )
    problem.add_boundary_condition(
        "Dirichlet_boundary_condition", "inflow_v", variable="v", boundary="inlet", value=0.0
    )
    for variable in ("u", "v"):
        problem.add_boundary_condition(
            "Dirichlet_boundary_condition",
            f"no_slip_{variable}",
            variable=variable,
            boundary=["walls", "cylinder"],
            value=0.0,
        )
    problem.add_boundary_condition(
        "Dirichlet_boundary_condition", "outflow", variable="pressure", boundary="outlet", value=0.0
    )
    history = {"time": [], "drag": [], "lift": []}

    def record(t, _problem):
        history["time"].append(t)
        history["drag"].append(-SCALE * problem.total_reaction("u", "cylinder"))
        history["lift"].append(-SCALE * problem.total_reaction("v", "cylinder"))

    problem.set_time_step_callback(record)
    start = time.perf_counter()
    result = problem.solve_transient(
        end_time=END_TIME,
        time_step=1e-3,
        time_stepper="cfl",
        courant_number=0.5,
        report="none",
    )
    out = {key: np.asarray(value) for key, value in history.items()}
    out["time_per_step"] = np.asarray([(time.perf_counter() - start) / result.time_steps])
    out["unknowns"] = np.asarray([problem.num_global_dofs])
    return out


def cycles(history):
    """The largest drag and lift coefficients and the Strouhal number over the last three full cycles of the lift, which start at its minima."""
    t, lift, drag = history["time"], history["lift"], history["drag"]
    late = t > END_TIME / 2
    t, lift, drag = t[late], lift[late], drag[late]
    minima = [
        i for i in range(1, len(lift) - 1) if lift[i] < lift[i - 1] and lift[i] <= lift[i + 1]
    ]
    # The time of each minimum, from the parabola through it and its two neighbors.
    times = []
    for i in minima:
        a, b, c = lift[i - 1], lift[i], lift[i + 1]
        h0, h1 = t[i] - t[i - 1], t[i + 1] - t[i]
        denominator = h1 * (a - b) + h0 * (c - b)
        shift = 0.5 * (h1**2 * (a - b) - h0**2 * (c - b)) / denominator if denominator else 0.0
        times.append(t[i] + shift)
    period = (times[-1] - times[-4]) / 3
    window = (t >= times[-4]) & (t <= times[-1])
    return {
        "max_drag": float(drag[window].max()),
        "max_lift": float(lift[window].max()),
        "strouhal": 0.1 / period,
    }


def compute(methods):
    import dualmesh as dm

    data = {}
    with tempfile.TemporaryDirectory() as work:
        meshes = {
            h: dm.read_mesh(str(dfg_mesh.build(h, Path(work) / f"dfg_{h}.msh"))) for h in SIZES
        }
        for method in methods:
            for h in SIZES:
                history = run(meshes[h], method)
                values = cycles(history)
                # The figures show the last time unit; the cache keeps the last one and a half.
                late = history["time"] > END_TIME - 1.5
                for key in ("time", "drag", "lift"):
                    history[key] = history[key][late]
                data[f"{method} {h}"] = {
                    **history,
                    **{k: np.asarray([v]) for k, v in values.items()},
                }
                if dm.is_root():
                    print(
                        f"{method}, h = {h}: max C_D {values['max_drag']:.4f}, max C_L {values['max_lift']:.4f}, "
                        f"St {values['strouhal']:.5f}, {history['time_per_step'][0]:.4f} s per step, "
                        f"{int(history['unknowns'][0])} unknowns",
                        flush=True,
                    )
    return data


def plot(data, out):
    finest = min(SIZES)
    for key, label, name in (
        ("lift", "lift coefficient", "dfg_2d2_lift.png"),
        ("drag", "drag coefficient", "dfg_2d2_drag.png"),
    ):
        fig, ax = plotstyle.new_figure()
        for method, (color, _marker) in plotstyle.METHOD_STYLE.items():
            run_ = data.get(f"{method} {finest}")
            if run_ is not None:
                late = run_["time"] > END_TIME - 1.0
                ax.plot(run_["time"][late], run_[key][late], "-", color=color, lw=1.6, label=method)
        ax.axhline(
            REFERENCE[key],
            color=plotstyle.REFERENCE_GREY,
            lw=1.2,
            ls="--",
            label="largest value, FEATFLOW",
        )
        plotstyle.style(ax, "time", label)
        plotstyle.legend_below(ax, ncol=4)
        plotstyle.save(fig, out / name)


def table(data):
    for name, d in data.items():
        print(
            f"{name:16s} max C_D {d['max_drag'][0]:.4f} ({100 * (d['max_drag'][0] / REFERENCE['drag'] - 1):+.2f} %), "
            f"max C_L {d['max_lift'][0]:.4f} ({100 * (d['max_lift'][0] / REFERENCE['lift'] - 1):+.2f} %), "
            f"St {d['strouhal'][0]:.5f} ({100 * (d['strouhal'][0] / REFERENCE['strouhal'] - 1):+.2f} %), "
            f"{d['time_per_step'][0]:.4f} s per step"
        )


if __name__ == "__main__":
    import dualmesh as dm

    argv = list(sys.argv[1:])
    methods = METHODS
    if "--method" in argv:
        i = argv.index("--method")
        methods = [argv[i + 1]]
        del argv[i : i + 2]
    out, plot_only = plotstyle.parse_args(argv, DEFAULT_OUT)
    cache = plotstyle.cache_path(__file__)
    if plot_only:
        data = plotstyle.load_cache(cache)
    else:
        data = compute(methods)
        if dm.is_root():
            if cache.exists():
                data = {**plotstyle.load_cache(cache), **data}
            plotstyle.save_cache(cache, data)
    if dm.is_root():
        table(data)
        plot(data, out)
