# SPDX-License-Identifier: LGPL-2.1-or-later
"""Benchmark F1: the DFG steady flow around a cylinder, 2D-1, at Re = 20
(Schaefer and Turek [SchaeferTurek1996], reference values of the FEATFLOW
benchmark pages).

The channel [0, 2.2] x [0, 0.41] holds a cylinder of diameter D = 0.1 centered
at (0.2, 0.2).  The inflow is u = 4 U y (0.41 - y) / 0.41^2 with U = 0.3, the
outflow x = 2.2 is the do-nothing condition nu du/dn - p n = 0 (the Laplacian
form of the viscous term), and the density is 1 and the viscosity 0.001, so
that Re = U_mean D / nu = 20 with U_mean = 2 U / 3 = 0.2.  The drag and lift
coefficients C = 2 F / (U_mean^2 D) are computed from the reactions on the
cylinder (the force on it is minus the reaction), and the pressure difference
between the front and the back of the cylinder, p(0.15, 0.2) - p(0.25, 0.2), by
sampling.  The reference values are C_D = 5.57953523384, C_L = 0.010618948146
and Delta p = 0.11752016697.

    python run_dfg_steady.py [output directory] [--plot-only]
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

REFERENCE = {"drag": 5.57953523384, "lift": 0.010618948146, "dp": 0.11752016697}
SIZES = [0.01, 0.005, 0.0025, 0.00125]  # element size on the cylinder
METHODS = ["dmcdm", "fem", "hfvm", "zfvm"]
SCALE = 2.0 / (0.2**2 * 0.1)  # 2 / (U_mean^2 D)
DEFAULT_OUT = HERE.parents[2] / "docs" / "_static" / "figures" / "fluids"


def solve(mesh, method, formulation="pressure"):
    import dualmesh as dm

    problem = dm.Problem(mesh, method=method)
    problem.add_physics(
        "incompressible_flow",
        "flow",
        velocities=["u", "v"],
        density=1.0,
        dynamic_viscosity=1e-3,
        formulation=formulation,
        viscous_form="laplacian",
    )
    problem.add_boundary_condition(
        "Dirichlet_boundary_condition",
        "inflow_u",
        variable="u",
        boundary="inlet",
        value="4*0.3*y*(0.41 - y)/0.41^2",
        scale_with_load=True,
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
    start = time.perf_counter()
    # The cell-centered method diverges on the coarsest mesh when the inflow starts at full speed, and raises it in four load steps.
    problem.solve(load_factors=[0.25, 0.5, 0.75, 1.0] if method == "zfvm" else [1.0], report="none")
    elapsed = time.perf_counter() - start
    pressure = problem.sample("pressure", [[0.15, 0.2], [0.25, 0.2]])
    return {
        "drag": -SCALE * problem.total_reaction("u", "cylinder"),
        "lift": -SCALE * problem.total_reaction("v", "cylinder"),
        "dp": pressure[0] - pressure[1],
        "unknowns": problem.num_active_dofs(),
        "time": elapsed,
    }


def compute():
    import dualmesh as dm

    data = {}
    with tempfile.TemporaryDirectory() as work:
        meshes = {
            (h, order): dm.read_mesh(
                str(dfg_mesh.build(h, Path(work) / f"dfg_{h}_{order}.msh", order))
            )
            for h in SIZES
            for order in (1, 2)
        }
        for method in METHODS:
            # The Jacobian of the cell-centered method couples two rings of neighbors, and its direct factorization on the finest mesh takes several minutes; that mesh is left out for it.
            sizes = SIZES[:-1] if method == "zfvm" else SIZES
            rows = [solve(meshes[(h, 1)], method) for h in sizes]
            data[method] = {key: np.asarray([r[key] for r in rows]) for key in rows[0]}
            data[method]["h"] = np.asarray(sizes)
            print(
                f"{method}: C_D {data[method]['drag']}, C_L {data[method]['lift']}, dp {data[method]['dp']}",
                flush=True,
            )
        # The quadratic triangles of the Taylor-Hood element, twice as large, so that the two meshes have about as many nodes.
        rows = [solve(meshes[(2 * h, 2)], "fem", "Taylor_Hood") for h in SIZES[1:]]
        data["fem, Taylor-Hood"] = {key: np.asarray([r[key] for r in rows]) for key in rows[0]}
        data["fem, Taylor-Hood"]["h"] = np.asarray(SIZES[1:])
        print(
            f"Taylor-Hood: C_D {data['fem, Taylor-Hood']['drag']}, C_L {data['fem, Taylor-Hood']['lift']}, dp {data['fem, Taylor-Hood']['dp']}",
            flush=True,
        )
    return data


def plot(data, out):
    for key, label, name in (
        ("drag", "error of the drag coefficient", "dfg_2d1_drag_error.png"),
        ("lift", "error of the lift coefficient", "dfg_2d1_lift_error.png"),
        ("dp", "error of the pressure difference", "dfg_2d1_pressure_error.png"),
    ):
        fig, ax = plotstyle.new_figure()
        for method, (color, marker) in plotstyle.METHOD_STYLE.items():
            if method in data:
                d = data[method]
                ax.loglog(
                    d["unknowns"],
                    np.abs(d[key] - REFERENCE[key]) / abs(REFERENCE[key]),
                    marker + "-",
                    color=color,
                    lw=2,
                    ms=7,
                    label=method,
                )
        plotstyle.style(ax, "unknowns", f"relative {label}")
        plotstyle.legend_below(ax, ncol=3)
        plotstyle.save(fig, out / name)


def table(data):
    for method, d in data.items():
        for i in range(len(d["h"])):
            print(
                f"{method:18s} {int(d['unknowns'][i]):7d} unknowns: C_D {d['drag'][i]:.5f} "
                f"({100 * (d['drag'][i] / REFERENCE['drag'] - 1):+.3f} %), C_L {d['lift'][i]:.5f}, "
                f"dp {d['dp'][i]:.5f} ({100 * (d['dp'][i] / REFERENCE['dp'] - 1):+.3f} %), {d['time'][i]:.1f} s"
            )


if __name__ == "__main__":
    out, plot_only = plotstyle.parse_args(sys.argv[1:], DEFAULT_OUT)
    cache = plotstyle.cache_path(__file__)
    if plot_only:
        data = plotstyle.load_cache(cache)
    else:
        data = compute()
        plotstyle.save_cache(cache, data)
    table(data)
    plot(data, out)
