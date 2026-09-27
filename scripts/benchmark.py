# SPDX-License-Identifier: LGPL-2.1-or-later
"""Performance benchmarks of dualmesh.

Run ``python scripts/benchmark.py`` from the repository root (with the
extension built in place or installed).  Every benchmark is timed as the best
of ``--repeat`` runs of wall-clock time on one thread, and the results are
printed as a table and written to ``benchmark_results.json`` (or the file
given with ``--output``), together with the version, the build options and
the machine, so that two runs can be compared.  ``--quick`` uses smaller
problems for a fast check.

The benchmarks cover what dominates the run time of real problems:

* **assembly**: one residual and Jacobian assembly, per method, for heat
  conduction, plane elasticity and pressure-velocity flow;
* **linear solvers**: the direct solver against the pressure mass matrix
  Schur preconditioner on a three-dimensional Taylor-Hood Stokes problem;
* **fuel rods**: a two-year irradiation of a rod segment with mechanics, in
  the 1.5D and axisymmetric models.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if (ROOT / "python" / "dualmesh").is_dir() and any((ROOT / "python" / "dualmesh").glob("_core*")):
    sys.path.insert(0, str(ROOT / "python"))

import dualmesh as dm  # noqa: E402
import numpy as np  # noqa: E402


def best_of(repeat, function):
    """The shortest of ``repeat`` wall-clock times of ``function()``, and its
    last return value."""
    best, value = float("inf"), None
    for _ in range(repeat):
        start = time.perf_counter()
        value = function()
        best = min(best, time.perf_counter() - start)
    return best, value


def heat_problem(n, method):
    mesh = dm.generate_rectangle_mesh(0, 1, 0, 1, n, n)
    problem = dm.Problem(mesh, method=method)
    problem.set_num_threads(1)
    problem.add_variable("temperature")
    problem.add_kernel(
        "heat_conduction",
        "conduction",
        variable="temperature",
        thermal_conductivity=2.0,
        temperature_polynomial=[1.0, 0.01],
    )
    problem.add_kernel("heat_source", "heating", variable="temperature", heat_source=10.0)
    problem.add_boundary_condition(
        "Dirichlet_boundary_condition", "cold", variable="temperature", boundary="left", value=0.0
    )
    return problem


def elasticity_problem(n, method):
    mesh = dm.generate_rectangle_mesh(0, 1, 0, 1, n, n)
    problem = dm.Problem(mesh, method=method)
    problem.set_num_threads(1)
    dm.physics.add_plane_elasticity(
        problem, youngs_modulus=200e9, poissons_ratio=0.3, formulation="plane_strain"
    )
    for component in ("displacement_x", "displacement_y"):
        problem.add_boundary_condition(
            "Dirichlet_boundary_condition",
            f"fixed_{component}",
            variable=component,
            boundary="left",
            value=0.0,
        )
    return problem


def flow_problem(n, method, formulation="pressure", element_type="Quad4", density=100.0):
    mesh = dm.generate_rectangle_mesh(0, 1, 0, 1, n, n, element_type=element_type)
    problem = dm.Problem(mesh, method=method)
    problem.set_num_threads(1)
    dm.physics.add_incompressible_flow(
        problem,
        velocities=["u", "v"],
        density=density,
        formulation=formulation,
        pressure_pin_point=(0.5, 0.0),
    )
    problem.add_boundary_condition(
        "Dirichlet_boundary_condition", "lid", variable="u", boundary="top", value=1.0
    )
    problem.add_boundary_condition(
        "Dirichlet_boundary_condition", "lid_v", variable="v", boundary="top", value=0.0
    )
    for variable in ("u", "v"):
        problem.add_boundary_condition(
            "Dirichlet_boundary_condition",
            f"walls_{variable}",
            variable=variable,
            boundary=["left", "right", "bottom"],
            value=0.0,
        )
    return problem


def assembly_benchmarks(quick, repeat):
    n = 32 if quick else 96
    rows = []
    for physics, build in (
        ("heat conduction", heat_problem),
        ("plane elasticity", elasticity_problem),
        ("pressure-velocity flow", flow_problem),
    ):
        for method in ("fem", "dmcdm", "hfvm", "zfvm"):
            problem = build(n, method)
            problem.initialize()
            seconds, (residual, _) = best_of(repeat, problem.linear_system)
            rows.append(
                {
                    "group": "assembly",
                    "case": f"{physics}, {n} x {n} Quad4",
                    "method": method,
                    "unknowns": int(len(residual)),
                    "seconds": seconds,
                    "microseconds_per_unknown": 1e6 * seconds / len(residual),
                }
            )
    return rows


def solver_benchmarks(quick, repeat):
    from dualmesh import mms

    n = 6 if quick else 10
    fields = {"u": "sin(y+z)", "v": "sin(x+z)", "w": "sin(x+y)", "pressure": "cos(x)*y*z + x"}
    study = mms.ManufacturedSolution(
        fields,
        [
            mms.IncompressibleFlow(
                ["u", "v", "w"], outflow={"right": (1.0, 0.0, 0.0)}, formulation="taylor_hood"
            )
        ],
        dimension=3,
    )
    variants = [
        ("direct LU", dict(linear_solver="lu")),
        (
            "GMRES + pressure mass Schur (built-in)",
            dict(linear_solver="gmres", preconditioner="pressure_mass_schur"),
        ),
    ]
    if dm.have_petsc():
        variants.append(
            (
                "PETSc Schur field split, GAMG on momentum",
                dict(
                    linear_solver="petsc",
                    preconditioner="pressure_mass_schur",
                    petsc_options="-fieldsplit_0_pc_type gamg",
                ),
            )
        )
    rows = []
    for label, options in variants:

        def run(options=options):
            mesh = dm.generate_box_mesh(0, 1, 0, 1, 0, 1, n, n, n, element_type="Tet10")
            problem = study.build(mesh, method="fem")
            problem.set_num_threads(1)
            problem.solve(linear_tolerance=1e-8, **options)
            return problem

        seconds, problem = best_of(repeat, run)
        rows.append(
            {
                "group": "linear solver",
                "case": f"Taylor-Hood Stokes, {n}^3 Tet10",
                "method": label,
                "unknowns": int(problem.num_active_dofs()),
                "seconds": seconds,
                "velocity_l2_error": float(study.errors(problem)[("u", "l2")]),
            }
        )
    return rows


def fuel_benchmarks(quick, repeat):
    from dualmesh import fuel

    year = 3.15576e7
    rows = []
    geometry = fuel.RodGeometry(stack_height=0.1)
    history = fuel.PowerHistory(times=[0.0, 3600.0, 2 * year], linear_power=[0.0, 20e3, 20e3])
    times = np.concatenate([[0.0, 3600.0], np.linspace(3600.0, 2 * year, 6 if quick else 11)[1:]])
    for model in ("1.5d", "rz"):

        def run(model=model):
            rod = fuel.FuelRod(geometry, history, model=model, mechanics=True)
            return rod.run(output_times=times)

        seconds, history_out = best_of(repeat, run)
        rows.append(
            {
                "group": "fuel rod",
                "case": f"UO2 rod, 20 kW/m for 2 years, {len(times) - 1} steps",
                "method": model,
                "seconds": seconds,
                "fission_gas_release": float(history_out.fission_gas_release[-1]),
            }
        )
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true", help="smaller problems")
    parser.add_argument("--repeat", type=int, default=3, help="runs per benchmark (best kept)")
    parser.add_argument("--output", default="benchmark_results.json")
    parser.add_argument(
        "--only", choices=["assembly", "solver", "fuel"], help="run one group of benchmarks"
    )
    args = parser.parse_args(argv)
    groups = {
        "assembly": assembly_benchmarks,
        "solver": solver_benchmarks,
        "fuel": fuel_benchmarks,
    }
    rows = []
    for name, function in groups.items():
        if args.only and name != args.only:
            continue
        rows.extend(function(args.quick, args.repeat))
    header = f"{'group':15s} {'case':42s} {'method':42s} {'seconds':>9s}"
    print(header)
    print("-" * len(header))
    for row in rows:
        print(f"{row['group']:15s} {row['case']:42s} {row['method']:42s} {row['seconds']:9.3f}")
    record = {
        "dualmesh_version": dm.__version__,
        "petsc": dm.have_petsc(),
        "mpi": dm.have_mpi(),
        "python": platform.python_version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "quick": args.quick,
        "results": rows,
    }
    Path(args.output).write_text(json.dumps(record, indent=2) + "\n")
    print(f"\nwritten to {args.output}")


if __name__ == "__main__":
    main()
