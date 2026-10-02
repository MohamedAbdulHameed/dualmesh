# SPDX-License-Identifier: LGPL-2.1-or-later
"""Command-line driver: run a problem described in an input file.

Input files are YAML and mirror the block structure of the Python API (and of
MOOSE input files)::

    mesh:
      type: rectangle
      x_min: 0.0
      x_max: 0.1
      y_min: 0.0
      y_max: 0.05
      num_x_elements: 10
      num_y_elements: 5

    problem:
      method: dmcdm            # or fem, hfvm, zfvm
      coordinates: cartesian   # or axisymmetric, spherical
      threads: 4               # optional; the default uses every core

    variables:
      temperature: {initial_condition: 0.0}

    functions:
      ambient: "40 + 10*x"     # an expression in x, y, z, t

    kernels:
      conduction:
        type: heat_conduction
        variable: temperature
        thermal_conductivity: 20.0
      heating:
        type: heat_source
        variable: temperature
        heat_source: 1.0e6

    boundary_conditions:
      left:  {type: Dirichlet_boundary_condition, variable: temperature, value: 40}
      right: {type: Dirichlet_boundary_condition, variable: temperature, value: 10}
      top:                     # a condition named after a side set acts on it
        type: convective_heat_flux_boundary_condition
        variable: temperature
        heat_transfer_coefficient: 75.0

    executioner:
      type: steady             # or transient
      nonlinear_solver: newton

    postprocessors:
      heat_out:     {type: reaction, variable: temperature, boundary: left}
      hottest:      {type: nodal_extreme_value, variable: temperature, value_type: max}

    outputs:
      vtu: bus_bar.vtu
      field_csv: bus_bar.csv
      postprocessor_csv: bus_bar_postprocessors.csv

Usage::

    dualmesh run input.yaml
    dualmesh run study.yaml                 # a file with a uq block: an uncertainty study
    mpirun -n 4 dualmesh run input.yaml     # distributed, when built with MPI
    dualmesh list --category kernel
    dualmesh describe heat_conduction

Under ``mpirun`` with more than one process the problem is solved by
:class:`~dualmesh.DistributedProblem`, configured by an optional ``parallel``
block (``partitioner``, ``linear_solver``, ``preconditioner``, ``overlap``,
``subdomain_solver``, ``linear_tolerance``, ``linear_max_iterations``); the
same input file runs unchanged on one process.
"""

from __future__ import annotations

import argparse
import difflib
import sys
from typing import Any

from . import __version__, list_objects
from . import describe as describe_object
from .expressions import parsed_function
from .meshing import (
    generate_annulus_mesh,
    generate_box_mesh,
    generate_line_mesh,
    generate_rectangle_mesh,
    read_mesh,
)
from .parallel import DistributedProblem, is_root, num_ranks
from .problem import Problem

#: The blocks an input file may contain.
BLOCKS = (
    "mesh",
    "problem",
    "parallel",
    "functions",
    "variables",
    "materials",
    "kernels",
    "boundary_conditions",
    "point_sources",
    "postprocessors",
    "executioner",
    "outputs",
)
PROBLEM_SETTINGS = ("method", "coordinates", "threads", "boundary_gradient")
OUTPUTS = ("vtu", "field_csv", "postprocessor_csv", "mesh_file", "cell_properties")


def _check_keys(where: str, given, allowed) -> None:
    """Refuse a key that is not recognised, instead of silently ignoring it:
    a misspelled block name would otherwise drop part of the problem."""
    for key in given:
        if key not in allowed:
            close = difflib.get_close_matches(str(key), allowed, n=1)
            hint = f" Did you mean '{close[0]}'?" if close else ""
            raise ValueError(f"Unknown {where} '{key}'.{hint} Allowed: {', '.join(allowed)}.")


def _coerce(value):
    """Convert input-file values to what the Python API expects.

    YAML is not strict about numbers (``1.0e6`` is a string in YAML 1.1), so a
    string that is a number becomes a number.  A string that starts with ``=``
    is an inline expression in ``x``, ``y``, ``z``, ``t``, for example
    ``value: "= 500*(1 - 10*x^2)"``.  Every other string is passed through, so
    that variable names, boundary names, and function names keep working.
    """
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("="):
            return parsed_function(text[1:])
        try:
            return float(text)
        except ValueError:
            return value
    if isinstance(value, list):
        return [_coerce(item) for item in value]
    return value


def _coerce_options(options: dict[str, Any]) -> dict[str, Any]:
    return {key: _coerce(value) for key, value in options.items()}


def build_mesh(block: dict[str, Any]):
    """Build a mesh from the ``mesh`` block of an input file."""
    block = dict(block)
    kind = str(block.pop("type", "file")).lower()
    builders = {
        "line": generate_line_mesh,
        "rectangle": generate_rectangle_mesh,
        "box": generate_box_mesh,
        "annulus": generate_annulus_mesh,
    }
    if kind in builders:
        return builders[kind](**_coerce_options(block))
    if kind == "file":
        return read_mesh(**block)
    raise ValueError(f"Unknown mesh type '{kind}'. Use line, rectangle, box, annulus, or file.")


def build_problem(document: dict[str, Any]):
    """Build the problem described by a parsed input file.

    Returns the :class:`~dualmesh.Problem`, or, when the program runs on more
    than one MPI process, the :class:`~dualmesh.DistributedProblem` whose
    ``local`` problem carries the objects.
    """
    _check_keys("block", document, BLOCKS)
    if "mesh" not in document:
        raise ValueError("The input file needs a 'mesh' block.")
    mesh = build_mesh(document["mesh"])
    settings = dict(document.get("problem") or {})
    _check_keys("problem setting", settings, PROBLEM_SETTINGS)
    method = settings.get("method", "dmcdm")
    coordinates = settings.get("coordinates", "cartesian")
    distributed = None
    if num_ranks() > 1:
        options = dict(document.get("parallel") or {})
        distributed = DistributedProblem(
            mesh, method=method, coordinates=coordinates, **_coerce_options(options)
        )
        problem = distributed.local
    else:
        problem = Problem(
            mesh,
            method=method,
            coordinates=coordinates,
            boundary_gradient=settings.get("boundary_gradient", "first_order"),
        )
    if "threads" in settings:
        problem.set_num_threads(int(settings["threads"]))
    for name, expression in (document.get("functions") or {}).items():
        if isinstance(expression, str):
            expression = parsed_function(expression.lstrip("= "))
        problem.add_function(name, expression)
    for name, options in (document.get("variables") or {}).items():
        options = dict(options or {})
        initial = _coerce(options.pop("initial_condition", None))
        problem.add_variable(
            name,
            block=options.pop("block", ()),
            initial_condition=initial,
            order=options.pop("order", "mesh"),
        )
        if options:
            raise ValueError(f"Unknown options for variable '{name}': {sorted(options)}")
    sections = (
        ("materials", problem.add_material),
        ("kernels", problem.add_kernel),
        ("boundary_conditions", problem.add_boundary_condition),
        ("point_sources", problem.add_point_source),
    )
    for section, adder in sections:
        for name, options in (document.get(section) or {}).items():
            options = dict(options)
            object_type = options.pop("type", None)
            if object_type is None:
                raise ValueError(f"'{section}.{name}' needs a 'type'.")
            adder(object_type, name, **_coerce_options(options))
    postprocessors = document.get("postprocessors") or {}
    if postprocessors and distributed is not None:
        raise ValueError(
            "Post-processors are not available in a distributed run yet; run the input file "
            "on one process to compute them."
        )
    for name, options in postprocessors.items():
        options = dict(options)
        pp_type = options.pop("type", None)
        if pp_type is None:
            raise ValueError(f"'postprocessors.{name}' needs a 'type'.")
        problem.add_postprocessor(pp_type, name, **_coerce_options(options))
    return distributed if distributed is not None else problem


def _gathered_problem(distributed, mesh, method: str, coordinates: str) -> Problem:
    """A one-process copy of a distributed solution on the whole mesh, for
    the outputs that need the whole field (sampling at points, CSV)."""
    whole = Problem(mesh, method=method, coordinates=coordinates)
    local = distributed.local
    names = [local._problem.variable_name(i) for i in range(local._problem.num_variables)]
    for name in names:
        whole.add_variable(name)
    for name in names:
        whole.set_values(name, distributed.gathered_values(name))
    return whole


def run(document: dict[str, Any], verbose: bool = False):
    """Build and solve the problem described by an input file."""
    solver = build_problem(document)
    distributed = isinstance(solver, DistributedProblem)
    problem = solver.local if distributed else solver
    root = is_root()
    executioner = _coerce_options(dict(document.get("executioner") or {}))
    kind = str(executioner.pop("type", "steady")).lower()
    executioner.setdefault("verbose", verbose)
    if kind == "steady":
        solver.solve(**executioner)
    elif kind == "transient":
        transient = {
            key: executioner.pop(key)
            for key in (
                "end_time",
                "dt",
                "start_time",
                "theta",
                "output_interval",
                "output_file_base",
            )
            if key in executioner
        }
        solver.solve_transient(**transient, **executioner)
    else:
        raise ValueError(f"Unknown executioner type '{kind}' (use steady or transient).")

    outputs = document.get("outputs") or {}
    _check_keys("output", outputs, OUTPUTS)
    if "vtu" in outputs:
        if distributed:
            base = str(outputs["vtu"])
            base = base[:-4] if base.endswith(".vtu") else base
            solver.write_vtu(base, outputs.get("cell_properties", ()))
            written = base + ".pvtu"
        else:
            problem.write_vtu(outputs["vtu"], outputs.get("cell_properties", ()))
            written = outputs["vtu"]
        if verbose and root:
            print(f"wrote {written}")
    needs_whole = any(key in outputs for key in ("field_csv", "mesh_file"))
    whole = problem
    if distributed and needs_whole:
        settings = document.get("problem") or {}
        whole = _gathered_problem(
            solver,
            build_mesh(document["mesh"]),
            settings.get("method", "dmcdm"),
            settings.get("coordinates", "cartesian"),
        )
    if root:
        if "field_csv" in outputs:
            whole.write_csv(outputs["field_csv"])
            if verbose:
                print(f"wrote {outputs['field_csv']}")
        if "mesh_file" in outputs:
            whole.write_mesh_file(outputs["mesh_file"])
    if not distributed and problem._postprocessors.postprocessors:
        if not verbose:  # a verbose solve has printed the table already
            print("Post-processors\n" + problem.postprocessor_table())
        if "postprocessor_csv" in outputs:
            problem.write_postprocessor_csv(outputs["postprocessor_csv"])
            if verbose:
                print(f"wrote {outputs['postprocessor_csv']}")
    return solver


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="dualmesh",
        description=(
            "A multiphysics framework for heat transfer, solid mechanics and fluid dynamics."
        ),
    )
    parser.add_argument("--version", action="version", version=f"dualmesh {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="solve the problem in an input file")
    run_parser.add_argument("input_file", help="YAML input file")
    run_parser.add_argument("--quiet", action="store_true", help="print less")

    list_parser = subparsers.add_parser("list", help="list the registered object types")
    list_parser.add_argument(
        "--category",
        help="kernel, boundary_condition, nodal_boundary_condition, nodal_load or material",
    )
    list_parser.add_argument("--module", help="framework, heat_transfer, solid_mechanics, ...")

    describe_parser = subparsers.add_parser("describe", help="describe one object type")
    describe_parser.add_argument("object_type")

    arguments = parser.parse_args(argv)
    if arguments.command == "run":
        try:
            import yaml
        except ImportError:  # pragma: no cover
            print("Reading input files needs PyYAML: pip install dualmesh[input]", file=sys.stderr)
            return 2
        with open(arguments.input_file) as stream:
            document = yaml.safe_load(stream)
        if isinstance(document, dict) and "uq" in document:
            # An uncertainty study of the model in the rest of the file (or
            # in the file its 'model' entry names).
            from pathlib import Path

            from .uq._input import run_study

            run_study(document, Path(arguments.input_file).parent, verbose=not arguments.quiet)
            return 0
        run(document, verbose=not arguments.quiet)
        return 0
    if arguments.command == "list":
        for name in list_objects(category=arguments.category, module=arguments.module):
            print(name)
        return 0
    print(describe_object(arguments.object_type))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
