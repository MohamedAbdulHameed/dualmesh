# SPDX-License-Identifier: LGPL-2.1-or-later
"""Command-line tools: the object reference and a check of the side sets of a
mesh file.

A problem is described and solved in a Python script, which runs with
``python script.py``, or with ``mpirun -n 4 python script.py`` on four
processes.  The command ``dualmesh`` provides the
tools that help to write such a script::

    dualmesh list --category physics              # the registered types
    dualmesh describe heat_conduction             # the parameters of one type
    dualmesh describe solid_mechanics             # the parameters of one physics
    dualmesh mesh wrench.msh --sidesets sides.vtu # the side sets of a mesh file
"""

from __future__ import annotations

import argparse

from . import __version__, list_objects
from . import describe as describe_object
from .meshing import read_mesh, sideset_summary, write_sidesets


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="dualmesh", description=("A multiphysics framework for heat transfer, solid mechanics and fluid dynamics."))
    parser.add_argument("--version", action="version", version=f"dualmesh {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="list the physics, couplings and objects")
    list_parser.add_argument("--category", help="physics, coupling, kernel, boundary_condition, nodal_boundary_condition, nodal_load or property")
    list_parser.add_argument("--module", help="framework, heat_transfer, solid_mechanics, fluids, neutronics, ...")

    describe_parser = subparsers.add_parser("describe", help="describe one type (an object, physics or coupling)")
    describe_parser.add_argument("object_type")

    mesh_parser = subparsers.add_parser("mesh", help="list the side sets of a mesh file, without solving")
    mesh_parser.add_argument("mesh_file", help="a mesh file that meshio reads (e.g., Gmsh .msh, Exodus, VTK)")
    mesh_parser.add_argument("--sidesets", metavar="FILE.vtu", help="also write the boundary with one field per side set, for ParaView")

    arguments = parser.parse_args(argv)
    if arguments.command == "mesh":
        mesh = read_mesh(arguments.mesh_file)
        print(mesh.summary())
        print(sideset_summary(mesh))
        if arguments.sidesets:
            write_sidesets(mesh, arguments.sidesets)
            print(f"wrote {arguments.sidesets}")
        return 0
    if arguments.command == "list":
        for name in list_objects(category=arguments.category, module=arguments.module):
            print(name)
        return 0
    try:
        print(describe_object(arguments.object_type))
    except ValueError as error:
        print(error)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
