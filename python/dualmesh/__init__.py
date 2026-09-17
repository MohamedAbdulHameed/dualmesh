# SPDX-License-Identifier: LGPL-2.1-or-later
"""dualmesh: the dual mesh control domain method for computational mechanics.

The package solves boundary value problems written in the canonical
conservation form

    -div F(u, grad u, x, t) + S(u, grad u, x, t) = 0,

where ``F`` is the flux and ``S`` the source of each equation.  The same
problem definition can be discretized either with the dual mesh control
domain method (DMCDM) of J. N. Reddy, which enforces the balance law over
node-centred control domains of a dual mesh, or with the standard Galerkin
finite element method, so that the two can be compared directly.

A minimal example (Reddy, *Computational Methods in Engineering*, Example
5.3.1: a cooling fin governed by ``-u'' + 400 u = 0``)::

    import dualmesh as dm

    mesh = dm.generate_line_mesh(start=0.0, end=0.05, num_elements=5)
    problem = dm.Problem(mesh, method="dmcdm")
    problem.add_variable("temperature")
    problem.add_kernel("Diffusion", variable="temperature")
    problem.add_kernel("Reaction", variable="temperature", coefficient=400.0)
    problem.add_boundary_condition(
        "DirichletBC", variable="temperature", boundary="left", value=300.0)
    problem.add_boundary_condition(
        "RobinBC", variable="temperature", boundary="right",
        transfer_coefficient=2.0)
    problem.solve()
    print(problem.values("temperature"))
"""

from __future__ import annotations

from . import _core as _core  # the compiled extension module
from . import fgm, physics, postprocess
from ._core import (
    ADReal,
    InputParameters,
    IntegratedBC,
    Kernel,
    Material,
    Mesh,
    NodalBC,
    QpContext,
    describe_object,
    object_category,
    object_module,
    registered_types,
)
from .ad import abs, cos, exp, log, pow, sin, sqrt, tanh
from .expressions import parsed_function
from .meshing import (
    annulus_coordinates,
    generate_annulus_mesh,
    generate_box_mesh,
    generate_line_mesh,
    generate_rectangle_mesh,
    graded_coordinates,
    mesh_from_arrays,
    read_mesh,
    write_mesh,
)
from .objects import (
    PythonBoundaryCondition,
    PythonKernel,
    PythonMaterial,
    PythonNodalBoundaryCondition,
)
from .problem import Problem, SolveResult

__version__ = "0.1.0"

__all__ = [
    "ADReal",
    "InputParameters",
    "IntegratedBC",
    "Kernel",
    "Material",
    "Mesh",
    "NodalBC",
    "Problem",
    "PythonBoundaryCondition",
    "PythonKernel",
    "PythonMaterial",
    "PythonNodalBoundaryCondition",
    "QpContext",
    "SolveResult",
    "abs",
    "annulus_coordinates",
    "cos",
    "describe_object",
    "exp",
    "fgm",
    "generate_annulus_mesh",
    "generate_box_mesh",
    "generate_line_mesh",
    "generate_rectangle_mesh",
    "graded_coordinates",
    "log",
    "mesh_from_arrays",
    "object_category",
    "object_module",
    "postprocess",
    "parsed_function",
    "physics",
    "pow",
    "read_mesh",
    "registered_types",
    "sin",
    "sqrt",
    "tanh",
    "write_mesh",
]


def describe(object_type: str) -> str:
    """Return the documentation and parameter list of a registered object."""
    return (
        f"{object_type} ({object_category(object_type)}, "
        f"module {object_module(object_type)})\n{describe_object(object_type)}"
    )


def list_objects(category: str | None = None, module: str | None = None) -> list[str]:
    """List registered object types, optionally filtered by category or module."""
    out = []
    for name in registered_types():
        if category is not None and object_category(name) != category:
            continue
        if module is not None and object_module(name) != module:
            continue
        out.append(name)
    return out
