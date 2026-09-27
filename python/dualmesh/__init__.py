# SPDX-License-Identifier: LGPL-2.1-or-later
"""dualmesh: a multiphysics framework for heat transfer, solid mechanics and
fluid dynamics.

A problem is any number of fields, each governed by a conservation law in the
canonical form

    -div F(u, grad u, x, t) + S(u, grad u, x, t) = 0,

where ``F`` is the flux and ``S`` the source of each equation and ``u`` stands
for all the fields, so that the equations may be coupled.  All the fields are
solved together by Newton's method with an exact Jacobian from automatic
differentiation.  The same problem definition can be discretized with the
Galerkin finite element method (``method="fem"``), the vertex-centred or the
cell-centred finite volume method (``"hfvm"``, ``"zfvm"``), or the dual mesh
control domain method of J. N. Reddy (``"dmcdm"``), which enforces the balance
law over node-centred control domains of a dual mesh.

A minimal example (Reddy, *Computational Methods in Engineering*, Example
5.3.1: a cooling fin governed by ``-u'' + 400 u = 0``)::

    import dualmesh as dm

    mesh = dm.generate_line_mesh(start=0.0, end=0.05, num_elements=5)
    problem = dm.Problem(mesh, method="dmcdm")
    problem.add_variable("temperature")
    problem.add_kernel("diffusion", variable="temperature")
    problem.add_kernel("reaction", variable="temperature", coefficient=400.0)
    problem.add_boundary_condition(
        "Dirichlet_boundary_condition", variable="temperature", boundary="left", value=300.0)
    problem.add_boundary_condition(
        "Robin_boundary_condition", variable="temperature", boundary="right",
        transfer_coefficient=2.0)
    problem.solve()
    print(problem.values("temperature"))
"""

from __future__ import annotations

from . import _core as _core  # the compiled extension module
from . import adaptivity, fgm, fuel, mms, parallel, physics, postprocess
from ._core import (
    ADReal,
    CylinderTableFunction,
    InputParameters,
    IntegratedBC,
    Kernel,
    Material,
    Mesh,
    NodalBC,
    QpContext,
    SettableFunction,
    describe_object,
    object_category,
    object_module,
    registered_types,
)
from .ad import abs, cos, exp, log, pow, sin, sqrt, tanh
from .adaptivity import (
    mark_by_error_fraction,
    mark_by_fraction,
    mark_by_threshold,
    refine_marked,
    solve_with_adaptive_refinement,
)
from .expressions import Expression, ParsedFunction, parsed_function
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
from .parallel import (
    DistributedProblem,
    have_metis,
    have_mpi,
    have_petsc,
    is_root,
    num_ranks,
    partition_mesh,
    petsc_version,
)
from .problem import Problem, SolveResult

__version__ = "0.2.0"

__all__ = [
    "CylinderTableFunction",
    "SettableFunction",
    "fuel",
    "ADReal",
    "DistributedProblem",
    "InputParameters",
    "IntegratedBC",
    "Kernel",
    "Material",
    "Mesh",
    "NodalBC",
    "Problem",
    "Expression",
    "ParsedFunction",
    "PythonBoundaryCondition",
    "PythonKernel",
    "PythonMaterial",
    "PythonNodalBoundaryCondition",
    "QpContext",
    "SolveResult",
    "abs",
    "adaptivity",
    "annulus_coordinates",
    "cos",
    "describe_object",
    "exp",
    "fgm",
    "generate_annulus_mesh",
    "generate_box_mesh",
    "generate_line_mesh",
    "generate_rectangle_mesh",
    "have_metis",
    "have_mpi",
    "have_petsc",
    "petsc_version",
    "is_root",
    "graded_coordinates",
    "log",
    "mark_by_error_fraction",
    "mark_by_fraction",
    "mark_by_threshold",
    "mesh_from_arrays",
    "mms",
    "num_ranks",
    "object_category",
    "object_module",
    "postprocess",
    "parallel",
    "parsed_function",
    "partition_mesh",
    "physics",
    "pow",
    "read_mesh",
    "refine_marked",
    "solve_with_adaptive_refinement",
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
    """List registered object types, optionally filtered by category
    (``kernel``, ``boundary_condition``, ``nodal_boundary_condition``,
    ``nodal_load`` or ``material``) or by module."""
    categories = {object_category(name) for name in registered_types()}
    if category is not None and category not in categories:
        raise ValueError(
            f"Unknown object category '{category}'. Use one of: {', '.join(sorted(categories))}."
        )
    out = []
    for name in registered_types():
        if category is not None and object_category(name) != category:
            continue
        if module is not None and object_module(name) != module:
            continue
        out.append(name)
    return out
