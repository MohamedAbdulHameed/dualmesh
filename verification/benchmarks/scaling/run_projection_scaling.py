# SPDX-License-Identifier: LGPL-2.1-or-later
"""Strong and weak scaling of the projection time integration, in the measures
of the scaling studies of Nek5000 and nekRS [Fischer2021]_: the time per step,
the iterations per step, the parallel efficiency and the number of grid points
(nodes) per process, n/P.

The problem is the decaying Taylor-Green vortex in a periodic square at a
Courant number of one half, advanced by ``dualmesh_benchmark projection`` (BDF2
and EXT2, eight projected pressures) with one thread per process:

* strong scaling: the same mesh of 256 x 256 and of 512 x 512 elements on 1 to
  4 processes;
* weak scaling: about 128 x 128 elements per process.

The time per step leaves out the first ten steps (the setup of the solvers and
the steps of lower order), and is that of the slowest process.

    python run_projection_scaling.py [output directory] [--plot-only] [--processes 4] [--build build-mpi]
"""

import os
import subprocess
import sys
from pathlib import Path

import numpy

here = Path(__file__).resolve().parent
sys.path.insert(0, str(here.parents[0]))
import plotstyle  # noqa: E402

root = here.parents[2]
methods = ["dmcdm", "fem"]
strong_sizes = [256, 512]
weak_base = 128
steps = 60
default_out = root / "docs" / "_static" / "figures" / "scaling"


def split_options(arguments):
    """The values of --processes and --build, and the other arguments."""
    options, rest = {"--processes": "4", "--build": "build-mpi"}, []
    it = iter(arguments)
    for a in it:
        if a in options:
            options[a] = next(it)
        else:
            rest.append(a)
    return int(options["--processes"]), options["--build"], rest


def run(processes, n, method, build):
    command = [
        "mpirun",
        "-n",
        str(processes),
        str(root / build / "dualmesh_benchmark"),
        "projection",
        str(n),
        str(steps),
        "2",
        "8",
        method,
    ]
    environment = dict(os.environ, OMP_NUM_THREADS="1")
    text = subprocess.run(
        command, capture_output=True, text=True, env=environment, check=True
    ).stdout
    line = next(line for line in text.splitlines() if line.startswith("RESULT"))
    values = dict(item.split("=") for item in line.split()[1:])
    return {
        key: float(values[key])
        for key in ("unknowns", "step_time", "setup", "pressure_iterations", "velocity_iterations")
    }


def compute(maximum_processes, build):
    data = {}
    counts = list(range(1, maximum_processes + 1))
    for method in methods:
        for n in strong_sizes:
            rows = [run(p, n, method, build) for p in counts]
            data[f"strong {method} {n}"] = {
                "processes": numpy.asarray(counts),
                "nodes": numpy.asarray([n * n] * len(counts)),
                **{key: numpy.asarray([r[key] for r in rows]) for key in rows[0]},
            }
            print(
                f"strong {method} {n}^2: "
                + ", ".join(f"P={p} {r['step_time']:.4f} s" for p, r in zip(counts, rows)),
                flush=True,
            )
        sizes = [round(weak_base * numpy.sqrt(p)) for p in counts]
        rows = [run(p, n, method, build) for p, n in zip(counts, sizes)]
        data[f"weak {method}"] = {
            "processes": numpy.asarray(counts),
            "nodes": numpy.asarray([n * n for n in sizes]),
            **{key: numpy.asarray([r[key] for r in rows]) for key in rows[0]},
        }
        print(
            f"weak {method}: "
            + ", ".join(
                f"P={p} ({n}^2) {r['step_time']:.4f} s" for p, n, r in zip(counts, sizes, rows)
            ),
            flush=True,
        )
    return data


def plot(data, out):
    fig, ax = plotstyle.new_figure()
    for method, (color, marker) in plotstyle.method_style.items():
        for n, style in zip(strong_sizes, ("-", "--")):
            d = data.get(f"strong {method} {n}")
            if d is None:
                continue
            speedup = d["step_time"][0] / d["step_time"]
            ax.plot(
                d["processes"],
                speedup,
                marker + style,
                color=color,
                lw=2,
                ms=7,
                mfc=color if style == "-" else "white",
                label=f"{method}, {n} x {n}",
            )
    p = data[f"strong {methods[0]} {strong_sizes[0]}"]["processes"]
    ax.plot(p, p, ":", color=plotstyle.reference_grey, lw=1.5, label="ideal")
    plotstyle.style(ax, "processes", "speedup of the time per step")
    plotstyle.legend_below(ax, ncol=3)
    plotstyle.save(fig, out / "projection_strong_speedup.png")

    fig, ax = plotstyle.new_figure()
    for method, (color, marker) in plotstyle.method_style.items():
        for n, style in zip(strong_sizes, ("-", "--")):
            d = data.get(f"strong {method} {n}")
            if d is None:
                continue
            efficiency = d["step_time"][0] / (d["step_time"] * d["processes"])
            ax.semilogx(
                d["nodes"] / d["processes"],
                100 * efficiency,
                marker + style,
                color=color,
                lw=2,
                ms=7,
                mfc=color if style == "-" else "white",
                label=f"{method}, {n} x {n}",
            )
    ax.axhline(80, color=plotstyle.reference_grey, lw=1.2, ls=":", label="80 %")
    ax.invert_xaxis()
    plotstyle.style(ax, "grid points per process, n/P", "parallel efficiency (%)")
    plotstyle.legend_below(ax, ncol=3)
    plotstyle.save(fig, out / "projection_strong_efficiency.png")

    fig, ax = plotstyle.new_figure()
    for method, (color, marker) in plotstyle.method_style.items():
        d = data.get(f"weak {method}")
        if d is None:
            continue
        ax.plot(
            d["processes"],
            d["step_time"] / d["nodes"] * d["processes"] * 1e6,
            marker + "-",
            color=color,
            lw=2,
            ms=7,
            label=method,
        )
    plotstyle.style(ax, "processes", "time per step and grid point per process (us)")
    ax.set_ylim(bottom=0)
    plotstyle.legend_below(ax, ncol=3)
    plotstyle.save(fig, out / "projection_weak.png")


def table(data):
    for name, d in data.items():
        print(name)
        for i in range(len(d["processes"])):
            p = int(d["processes"][i])
            efficiency = (
                d["step_time"][0] / (d["step_time"][i] * p)
                if name.startswith("strong")
                else d["step_time"][0] / d["step_time"][i] * (d["nodes"][i] / d["nodes"][0]) / p
            )
            print(
                f"  P={p} n={int(d['nodes'][i])} n/P={int(d['nodes'][i] / p)}: {d['step_time'][i]:.4f} s per step, "
                f"efficiency {100 * efficiency:.0f} %, setup {d['setup'][i]:.2f} s, "
                f"pressure {d['pressure_iterations'][i]:.1f}, velocity {d['velocity_iterations'][i]:.1f} iterations per step"
            )


if __name__ == "__main__":
    processes, build, rest = split_options(sys.argv[1:])
    out, plot_only = plotstyle.parse_arguments(rest, default_out)
    cache = plotstyle.cache_path(__file__)
    if plot_only:
        data = plotstyle.load_cache(cache)
    else:
        data = compute(processes, build)
        plotstyle.save_cache(cache, data)
    table(data)
    plot(data, out)
