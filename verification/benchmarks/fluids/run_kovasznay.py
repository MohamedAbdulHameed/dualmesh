# SPDX-License-Identifier: LGPL-2.1-or-later
"""Benchmark F4: the Kovasznay flow, an exact solution of the steady
Navier-Stokes equations [Kovasznay1948].

The flow behind a row of cylinders at the Reynolds number Re = 40 has the velocity
u = 1 - exp(lambda x) cos(2 pi y), v = lambda / (2 pi) exp(lambda x) sin(2 pi y)
and the pressure p = (1 - exp(2 lambda x)) / 2, with
lambda = Re / 2 - sqrt(Re^2 / 4 + 4 pi^2), for the density 1 and the viscosity 1 / Re.
The script checks symbolically that it satisfies the equations without a body
force, solves it on [-0.5, 1] x [-0.5, 1.5] with the exact velocity on three
sides and the exact traction on the outflow side x = 1, and records the errors
of every method on a sequence of meshes:

* the stabilized pressure-velocity formulation with dmcdm, fem, hfvm and zfvm,
  by Newton's method;
* the Taylor-Hood element (fem on Quad9 elements);
* the projection time integration (dmcdm, fem, hfvm), marched from the exact
  field to the steady state of its discretization, with the velocity
  prescribed on all four sides.

    python run_kovasznay.py [output directory] [--plot-only]
"""

import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0]))
import plotstyle  # noqa: E402

RE = 40.0
LAMBDA = RE / 2 - math.sqrt(RE**2 / 4 + 4 * math.pi**2)
FIELDS = {
    "u": f"1 - exp({LAMBDA!r}*x)*cos(2*pi*y)",
    "v": f"{LAMBDA / (2 * math.pi)!r}*exp({LAMBDA!r}*x)*sin(2*pi*y)",
    "pressure": f"(1 - exp({2 * LAMBDA!r}*x))/2",
}
# Elements across x, and 4/3 as many across y.  The coarsest mesh is 16 across: on 8 x 11 elements the cell-centered method has no converged solution (Newton's method and relaxed direct iteration both fail).
LEVELS = [16, 32, 64, 128]
METHODS = ["dmcdm", "fem", "hfvm", "zfvm"]
PROJECTION_METHODS = ["dmcdm", "fem", "hfvm"]
DEFAULT_OUT = HERE.parents[2] / "docs" / "_static" / "figures" / "fluids"


def mesh(n, element_type="Quad4"):
    import dualmesh as dm

    return dm.generate_rectangle_mesh(
        -0.5, 1.0, -0.5, 1.5, n, round(4 * n / 3), element_type=element_type
    )


def study(**physics):
    from dualmesh import mms

    s = mms.ManufacturedSolution(FIELDS, dimension=2, flux_boundaries={"right": (1.0, 0.0)})
    s.add_physics(
        "incompressible_flow",
        "flow",
        velocities=["u", "v"],
        density=1.0,
        dynamic_viscosity=1.0 / RE,
        **physics,
    )
    return s


def check_exact():
    """The forcing that makes the fields a solution vanishes, to rounding, at points spread over the domain."""
    import sympy

    s = study(formulation="pressure")
    x, y = sympy.symbols("x y")
    rng = np.random.default_rng(1)
    points = rng.uniform([-0.5, -0.5], [1.0, 1.5], size=(200, 2))
    largest = 0.0
    for f in s.forcing().values():
        value = sympy.lambdify((x, y), f, "numpy")
        largest = max(
            largest, float(np.max(np.abs(value(points[:, 0], points[:, 1]) + 0 * points[:, 0])))
        )
    assert largest < 1e-10, largest
    print(
        f"the Kovasznay fields satisfy the Navier-Stokes equations without a body force (largest forcing {largest:.1e})"
    )


def projection_errors(n, method):
    """The projection time integration, from the exact field to the steady state of its discretization."""
    import dualmesh as dm
    from dualmesh import _core

    grid = mesh(n)
    problem = dm.Problem(grid, method=method, distributed=False)
    problem.add_physics(
        "incompressible_flow",
        "flow",
        velocities=["u", "v"],
        density=1.0,
        dynamic_viscosity=1.0 / RE,
        formulation="pressure",
        time_integration="projection",
    )
    walls = ["left", "right", "bottom", "top"]
    for variable in ("u", "v"):
        problem.add_boundary_condition(
            "Dirichlet_boundary_condition",
            f"exact_{variable}",
            variable=variable,
            boundary=walls,
            value=FIELDS[variable],
        )
    problem.initialize()
    points = problem.entity_points()
    for variable in ("u", "v"):
        exact = _core.ParsedFunction(FIELDS[variable])
        problem.set_values(variable, np.array([exact(p[0], p[1], p[2], 0.0) for p in points]))
    # Ten time units, about five times the time the flow takes to cross the domain at its mean speed, after which the error of the initial interpolation has decayed.
    problem.solve_transient(
        end_time=10.0,
        time_step=1e-3,
        time_stepper="cfl",
        courant_number=0.5,
        report="none",
    )
    # The pressure of an enclosed flow is known up to a constant: compare it with the exact one shifted to the same mean.
    volume = problem.domain_volume()
    shift = (problem.integrate("pressure") - exact_pressure_integral()) / volume
    pressure_exact = f"{FIELDS['pressure']} + {shift!r}"
    errors = {}
    for variable, exact in (("u", FIELDS["u"]), ("pressure", pressure_exact)):
        errors[variable] = problem.error_norms(variable, exact)[0]
    return errors


def exact_pressure_integral():
    """The integral of the exact pressure over [-0.5, 1] x [-0.5, 1.5]."""
    a, b = -0.5, 1.0
    return 2.0 * (
        (b - a) / 2 - (math.exp(2 * LAMBDA * b) - math.exp(2 * LAMBDA * a)) / (4 * LAMBDA)
    )


def compute():
    check_exact()
    data = {}
    for method in METHODS:
        start = time.perf_counter()
        result = study(formulation="pressure").convergence_study(
            mesh, LEVELS, method=method, report="none"
        )
        data[method] = {
            "h": np.asarray(result.sizes),
            "u": np.asarray(result.errors[("u", "l2")]),
            "u_h1": np.asarray(result.errors[("u", "h1")]),
            "pressure": np.asarray(result.errors[("pressure", "l2")]),
        }
        print(
            f"{method}: velocity L2 {data[method]['u']}, pressure L2 {data[method]['pressure']}, "
            f"{time.perf_counter() - start:.1f} s",
            flush=True,
        )
    result = study(formulation="Taylor_Hood").convergence_study(
        lambda n: mesh(n, "Quad9"), LEVELS[:-1], method="fem", report="none"
    )
    data["fem, Taylor-Hood"] = {
        "h": np.asarray(result.sizes),
        "u": np.asarray(result.errors[("u", "l2")]),
        "u_h1": np.asarray(result.errors[("u", "h1")]),
        "pressure": np.asarray(result.errors[("pressure", "l2")]),
    }
    print(f"Taylor-Hood: velocity L2 {data['fem, Taylor-Hood']['u']}", flush=True)
    for method in PROJECTION_METHODS:
        start = time.perf_counter()
        rows = [projection_errors(n, method) for n in LEVELS]
        h = data[method]["h"]
        data[f"projection {method}"] = {
            "h": h,
            "u": np.asarray([r["u"] for r in rows]),
            "pressure": np.asarray([r["pressure"] for r in rows]),
        }
        print(
            f"projection {method}: velocity L2 {data[f'projection {method}']['u']}, "
            f"pressure L2 {data[f'projection {method}']['pressure']}, "
            f"{time.perf_counter() - start:.1f} s",
            flush=True,
        )
    return data


def rate(h, e):
    return float(np.polyfit(np.log(h[-2:]), np.log(e[-2:]), 1)[0])


def plot(data, out):
    for variable, label, name in (
        ("u", "L2 error of the velocity u", "kovasznay_velocity_error.png"),
        ("pressure", "L2 error of the pressure", "kovasznay_pressure_error.png"),
    ):
        fig, ax = plotstyle.new_figure()
        for method, (color, marker) in plotstyle.METHOD_STYLE.items():
            if method not in data:
                continue
            d = data[method]
            ax.loglog(
                d["h"],
                d[variable],
                marker + "-",
                color=color,
                lw=2,
                ms=7,
                label=f"{method} (order {rate(d['h'], d[variable]):.2f})",
            )
            projected = data.get(f"projection {method}")
            if projected is not None:
                ax.loglog(
                    projected["h"],
                    projected[variable],
                    marker + "--",
                    color=color,
                    lw=1.5,
                    ms=6,
                    mfc="white",
                    label=f"{method}, projection (order {rate(projected['h'], projected[variable]):.2f})",
                )
        plotstyle.style(ax, "element size h", label)
        plotstyle.legend_below(ax, ncol=2)
        plotstyle.save(fig, out / name)


def table(data):
    """The errors on the finest mesh and the orders, as the rows of the verification page."""
    for method, d in data.items():
        print(
            f"{method:24s} velocity {d['u'][-1]:.3e} (order {rate(d['h'], d['u']):.2f}), "
            f"pressure {d['pressure'][-1]:.3e} (order {rate(d['h'], d['pressure']):.2f})"
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
