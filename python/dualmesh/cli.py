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
      method: dmcdm            # or fem
      coordinates: cartesian   # or axisymmetric, spherical

    variables:
      temperature: {initial_condition: 0.0}

    functions:
      ambient: "40 + 10*x"     # an expression in x, y, z, t

    kernels:
      conduction:
        type: HeatConduction
        variable: temperature
        thermal_conductivity: 20.0
      heating:
        type: HeatSource
        variable: temperature
        heat_source: 1.0e6

    boundary_conditions:
      left:  {type: DirichletBC, variable: temperature, boundary: left, value: 40}
      right: {type: DirichletBC, variable: temperature, boundary: right, value: 10}
      top:
        type: ConvectiveHeatFluxBC
        variable: temperature
        boundary: top
        heat_transfer_coefficient: 75.0

    executioner:
      type: steady             # or transient
      nonlinear_solver: newton

    outputs:
      vtu: bus_bar.vtu
      csv: bus_bar.csv
      reactions: [[temperature, left]]

Usage::

    dualmesh run input.yaml
    dualmesh list --category Kernel
    dualmesh describe HeatConduction
"""

from __future__ import annotations

import argparse
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
from .problem import Problem


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


def build_problem(document: dict[str, Any]) -> Problem:
    """Build a Problem from a parsed input file."""
    if "mesh" not in document:
        raise ValueError("The input file needs a 'mesh' block.")
    mesh = build_mesh(document["mesh"])
    settings = document.get("problem", {})
    problem = Problem(
        mesh,
        method=settings.get("method", "dmcdm"),
        coordinates=settings.get("coordinates", "cartesian"),
    )
    for name, expression in (document.get("functions") or {}).items():
        if isinstance(expression, str):
            expression = parsed_function(expression.lstrip("= "))
        problem.add_function(name, expression)
    for name, options in (document.get("variables") or {}).items():
        options = dict(options or {})
        initial = _coerce(options.pop("initial_condition", None))
        problem.add_variable(name, blocks=options.pop("blocks", ()), initial_condition=initial)
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
    return problem


def run(document: dict[str, Any], verbose: bool = False) -> Problem:
    """Build and solve the problem described by an input file."""
    problem = build_problem(document)
    executioner = _coerce_options(dict(document.get("executioner") or {}))
    kind = str(executioner.pop("type", "steady")).lower()
    executioner.setdefault("verbose", verbose)
    if kind == "steady":
        result = problem.solve(**executioner)
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
        result = problem.solve_transient(**transient, **executioner)
    else:
        raise ValueError(f"Unknown executioner type '{kind}' (use steady or transient).")
    if verbose:
        print(f"converged: {result.converged} after {result.total_iterations} iterations")

    outputs = document.get("outputs") or {}
    if "vtu" in outputs:
        problem.write_vtu(outputs["vtu"], outputs.get("cell_properties", ()))
        if verbose:
            print(f"wrote {outputs['vtu']}")
    if "csv" in outputs:
        problem.write_csv(outputs["csv"])
        if verbose:
            print(f"wrote {outputs['csv']}")
    if "mesh_file" in outputs:
        problem.write_mesh_file(outputs["mesh_file"])
    for variable, boundary in outputs.get("reactions", []):
        total = problem.total_reaction(variable, boundary)
        print(f"total reaction of '{variable}' on '{boundary}': {total:.10g}")
    for entry in outputs.get("point_values", []):
        variable, point = entry[0], entry[1:]
        value = problem.sample(variable, [list(point)])[0]
        print(f"{variable} at {list(point)}: {value:.10g}")
    return problem


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="dualmesh",
        description="The dual mesh control domain method for engineering problems.",
    )
    parser.add_argument("--version", action="version", version=f"dualmesh {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="solve the problem in an input file")
    run_parser.add_argument("input_file", help="YAML input file")
    run_parser.add_argument("--quiet", action="store_true", help="print less")

    list_parser = subparsers.add_parser("list", help="list the registered object types")
    list_parser.add_argument("--category", help="Kernel, BoundaryCondition, NodalBC, Material")
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
