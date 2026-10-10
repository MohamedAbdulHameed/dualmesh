# SPDX-License-Identifier: LGPL-2.1-or-later
"""Benchmark F5: the lid-driven cavity at Re = 1000, against the centerline
velocities of Erturk, Corke and Gokcol [Erturk2005] on a 601 x 601 grid.

The unit cavity has its lid y = 1 moving at u = 1, the density Re and the
viscosity 1, so that the Reynolds number is Re.  The script solves it on
uniform meshes of 32, 64 and 128 elements per side and records the largest
difference of the velocity u along x = 0.5 and of v along y = 0.5 from the
tabulated values, for

* the stabilized pressure-velocity formulation with dmcdm, fem, hfvm and zfvm,
  by Newton's method with the lid speed raised in load steps;
* the Taylor-Hood element (fem on Quad9 elements);
* the projection time integration (dmcdm, fem, hfvm), marched from rest to
  the steady state.

    python run_cavity.py [output directory] [--plot-only]
"""

import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0]))
sys.path.insert(0, str(HERE))
import erturk2005 as reference  # noqa: E402
import plotstyle  # noqa: E402

RE = 1000.0
LEVELS = [32, 64, 128]
# The cell-centered method has no converged solution on 32 x 32 cells at this Reynolds number (Newton's method diverges, also with ten load steps), and is compared on the two finer meshes; 256 x 256 cells take more than ten minutes with the serial direct solver.
LEVELS_ZFVM = [64, 128]
METHODS = ["dmcdm", "fem", "hfvm", "zfvm"]
PROJECTION_METHODS = ["dmcdm", "fem", "hfvm"]
PROJECTION_LEVEL = 64
DEFAULT_OUT = HERE.parents[2] / "docs" / "_static" / "figures" / "fluids"


def cavity(n, method, element_type="Quad4", formulation="pressure", projection=False):
    import dualmesh as dm

    mesh = dm.generate_rectangle_mesh(0, 1, 0, 1, n, n, element_type=element_type)
    problem = dm.Problem(mesh, method=method, distributed=False)
    parameters = dict(
        velocities=["u", "v"],
        dynamic_viscosity=1.0,
        density=RE,
        formulation=formulation,
    )
    if formulation == "pressure":
        parameters["pressure_pin_point"] = (0.5, 0.0)
    if projection:
        parameters["time_integration"] = "projection"
    problem.add_physics("incompressible_flow", "flow", **parameters)
    problem.add_boundary_condition(
        "Dirichlet_boundary_condition",
        "lid",
        variable="u",
        boundary="top",
        value=1.0,
        scale_with_load=True,
    )
    problem.add_boundary_condition(
        "Dirichlet_boundary_condition", "lid_v", variable="v", boundary="top", value=0.0
    )
    # The walls are added last, so that the two top corners belong to the walls and the lid does not leak.
    for variable in ("u", "v"):
        problem.add_boundary_condition(
            "Dirichlet_boundary_condition",
            f"walls_{variable}",
            variable=variable,
            boundary=["left", "right", "bottom"],
            value=0.0,
        )
    return problem


def profiles(problem):
    y, x = np.asarray(reference.Y), np.asarray(reference.X)
    u = problem.sample("u", np.column_stack([np.full_like(y, 0.5), y]))
    v = problem.sample("v", np.column_stack([x, np.full_like(x, 0.5)]))
    return u, v


def steady(n, method, element_type="Quad4", formulation="pressure"):
    problem = cavity(n, method, element_type, formulation)
    problem.solve(load_factors=list(np.linspace(0.1, 1.0, 10)), report="none")
    return profiles(problem)


def projected(n, method):
    """From rest to the steady state: the run stops when the velocity changes by less than 1e-7 per unit time."""
    problem = cavity(n, method, projection=True)
    last = {}
    total = 0.0
    while True:
        problem.solve_transient(
            start_time=total,
            end_time=total + 10.0,
            time_step=1e-3 if total == 0.0 else 0.5 / n,
            time_stepper="cfl",
            courant_number=0.5,
            report="none",
        )
        total += 10.0
        u = problem.values("u").copy()
        if "u" in last and np.abs(u - last["u"]).max() / 10.0 < 1e-7:
            break
        last["u"] = u
        if total >= 400.0:
            break
    return (*profiles(problem), total)


def compute():
    data = {}
    U, V = np.asarray(reference.U), np.asarray(reference.V)
    for method in METHODS:
        start = time.perf_counter()
        du, dv, curves = [], [], []
        levels = LEVELS_ZFVM if method == "zfvm" else LEVELS
        for n in levels:
            u, v = steady(n, method)
            du.append(np.abs(u - U).max())
            dv.append(np.abs(v - V).max())
            curves.append((u, v))
        data[method] = {
            "n": np.asarray(levels),
            "du": np.asarray(du),
            "dv": np.asarray(dv),
            "u": curves[-1][0],
            "v": curves[-1][1],
        }
        print(
            f"{method}: max |u - u_ref| {du}, max |v - v_ref| {dv}, {time.perf_counter() - start:.0f} s",
            flush=True,
        )
    du, dv = [], []
    for n in LEVELS:
        u, v = steady(n // 2, "fem", "Quad9", "Taylor_Hood")
        du.append(np.abs(u - U).max())
        dv.append(np.abs(v - V).max())
    data["fem, Taylor-Hood"] = {
        "n": np.asarray(LEVELS),
        "du": np.asarray(du),
        "dv": np.asarray(dv),
        "u": u,
        "v": v,
    }
    print(
        f"Taylor-Hood (Quad9, n/2 elements): max |u - u_ref| {du}, max |v - v_ref| {dv}", flush=True
    )
    for method in PROJECTION_METHODS:
        start = time.perf_counter()
        u, v, end = projected(PROJECTION_LEVEL, method)
        data[f"projection {method}"] = {
            "n": np.asarray([PROJECTION_LEVEL]),
            "du": np.asarray([np.abs(u - U).max()]),
            "dv": np.asarray([np.abs(v - V).max()]),
            "u": u,
            "v": v,
            "end_time": np.asarray([end]),
        }
        print(
            f"projection {method} ({PROJECTION_LEVEL}^2, steady at t = {end:.0f}): "
            f"max |u - u_ref| {data[f'projection {method}']['du'][0]:.4f}, "
            f"max |v - v_ref| {data[f'projection {method}']['dv'][0]:.4f}, {time.perf_counter() - start:.0f} s",
            flush=True,
        )
    return data


def plot(data, out):
    for key, coordinate, values, label, name in (
        ("u", reference.Y, reference.U, "u at x = 0.5", "cavity_re1000_u.png"),
        ("v", reference.X, reference.V, "v at y = 0.5", "cavity_re1000_v.png"),
    ):
        fig, ax = plotstyle.new_figure()
        for method, (color, _marker) in plotstyle.METHOD_STYLE.items():
            if method in data:
                along = np.asarray(coordinate)
                order = np.argsort(along)
                ax.plot(
                    along[order],
                    data[method][key][order],
                    "-",
                    color=color,
                    lw=1.8,
                    label=(
                        f"{method}, {int(data[method]['n'][-1]) // 2} x {int(data[method]['n'][-1]) // 2} Quad9"
                        if method == "fem, Taylor-Hood"
                        else f"{method}, {int(data[method]['n'][-1])} x {int(data[method]['n'][-1])}"
                    ),
                )
        ax.plot(
            coordinate,
            values,
            "o",
            color=plotstyle.INK,
            ms=5,
            mfc="white",
            label="Erturk et al. (601 x 601)",
        )
        plotstyle.style(ax, "y" if key == "u" else "x", label)
        plotstyle.legend_below(ax, ncol=2)
        plotstyle.save(fig, out / name)
    fig, ax = plotstyle.new_figure()
    for method, (color, marker) in plotstyle.METHOD_STYLE.items():
        if method in data:
            d = data[method]
            ax.loglog(
                1.0 / d["n"],
                np.maximum(d["du"], d["dv"]),
                marker + "-",
                color=color,
                lw=2,
                ms=7,
                label=method,
            )
    plotstyle.style(
        ax,
        "element size h (Taylor-Hood: half its element)",
        "largest difference from Erturk et al.",
    )
    plotstyle.legend_below(ax, ncol=2)
    plotstyle.save(fig, out / "cavity_re1000_difference.png")


def table(data):
    for method, d in data.items():
        print(f"{method:24s} n {list(d['n'])}: u {np.round(d['du'], 4)}, v {np.round(d['dv'], 4)}")


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
