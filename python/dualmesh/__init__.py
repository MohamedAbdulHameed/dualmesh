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
    problem.add_boundary_condition("Dirichlet_boundary_condition", variable="temperature", boundary="left", value=300.0)
    problem.add_boundary_condition("Robin_boundary_condition", variable="temperature", boundary="right", transfer_coefficient=2.0)
    problem.solve()
    print(problem.values("temperature"))
"""

from __future__ import annotations

from . import _core as _core  # the compiled extension module
from . import adaptivity, fgm, fuel, mms, parallel, physics, postprocess
from ._core import ADReal, CylinderTableFunction, InputParameters, IntegratedBC, Kernel, Mesh, NodalBC, QpContext, SettableFunction, describe_object, object_category, object_module, registered_types
from .ad import abs, cos, exp, log, pow, sin, sqrt, tanh
from .adaptivity import AdaptivityResult, mark_by_error_fraction, mark_by_fraction, mark_by_threshold, refine_marked, solve_with_adaptive_refinement
from .expressions import Expression, ParsedFunction, parsed_function
from .meshing import annulus_coordinates, generate_annulus_mesh, generate_box_mesh, generate_line_mesh, generate_rectangle_mesh, graded_coordinates, mesh_from_arrays, read_mesh, sideset_summary, write_mesh, write_sidesets
from .objects import PythonBoundaryCondition, PythonKernel, PythonNodalBoundaryCondition, PythonProperty
from .parallel import have_metis, have_mpi, have_petsc, is_root, num_ranks, partition_mesh, petsc_version
from .problem import Problem, SolveResult
from .tables import Table

__version__ = "0.2.0"

__all__ = [
    "CylinderTableFunction",
    "SettableFunction",
    "fuel",
    "ADReal",
    "InputParameters",
    "IntegratedBC",
    "Kernel",
    "Mesh",
    "NodalBC",
    "Problem",
    "Expression",
    "ParsedFunction",
    "PythonBoundaryCondition",
    "PythonKernel",
    "PythonProperty",
    "PythonNodalBoundaryCondition",
    "QpContext",
    "SolveResult",
    "Table",
    "AdaptivityResult",
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
    "sideset_summary",
    "write_mesh",
    "write_sidesets",
]


def _applications() -> dict:
    """Name -> (class, category, module) of the applications (complete
    models of a component, with a ``run`` method) and of their input groups
    (the dataclasses they take)."""
    import dataclasses

    from . import fuel
    from .fuel import triso

    out = {}
    for module_name, module in (("fuel", fuel), ("fuel", triso)):
        for name in module.__all__:
            value = getattr(module, name, None)
            # RodContext and IrradiationFields are computed by the rod for its
            # materials, and a user does not give them.
            if not isinstance(value, type) or name.endswith("Result") or name in out or name in ("RodContext", "IrradiationFields"):
                continue
            if callable(getattr(value, "run", None)) and not dataclasses.is_dataclass(value):
                out[name] = (value, "application", module_name)
            elif dataclasses.is_dataclass(value) and not name.endswith("Result"):
                out[name] = (value, "input_group", module_name)
    return out


def _describe_application(name: str, cls, category: str, module: str) -> str:
    import dataclasses
    import inspect

    from .parameters import describe_fields

    doc = " ".join((cls.__doc__ or "").split("\n\n")[0].split())
    if dataclasses.is_dataclass(cls):
        return f"{name} ({category.replace('_', ' ')}, module {module})\n{doc}\n\n{describe_fields(cls)}\n"
    lines = []
    for parameter_name, p in inspect.signature(cls.__init__).parameters.items():
        if parameter_name == "self":
            continue
        annotation = p.annotation if isinstance(p.annotation, str) else getattr(p.annotation, "__name__", str(p.annotation))
        annotation = annotation.replace(" | None", "")
        required = ", required" if p.default is inspect.Parameter.empty else ", default: its default group"
        lines.append(f"  {parameter_name} ({annotation}{required})")
    return f"{name} ({category}, module {module})\n{doc}\n\nInput groups (each described by 'dualmesh describe <group>'):\n" + "\n".join(lines) + "\n"


def describe(object_type: str) -> str:
    """Return the documentation and parameter list of a registered object,
    physics or coupling, or of an application or one of its input groups
    (e.g., ``FuelRod`` or ``RodNumerics``)."""
    if physics.is_registered(object_type):
        return physics.describe(object_type)
    applications = _applications()
    if object_type in applications:
        return _describe_application(object_type, *applications[object_type])
    if object_type not in registered_types():
        import difflib

        names = sorted(set(registered_types()) | set(physics.registered()) | set(applications))
        close = difflib.get_close_matches(object_type, names, n=1)
        hint = f" Did you mean '{close[0]}'?" if close else ""
        raise ValueError(f"Unknown type '{object_type}'.{hint} 'dualmesh list' prints every type.")
    return f"{object_type} ({object_category(object_type)}, module {object_module(object_type)})\n{describe_object(object_type)}"


def list_objects(category: str | None = None, module: str | None = None) -> list[str]:
    """List the registered types, optionally filtered by category
    (``application``, ``input_group``, ``physics``, ``coupling``,
    ``kernel``, ``boundary_condition``, ``nodal_boundary_condition``,
    ``nodal_load`` or ``material``) or by module."""
    categories = {object_category(name) for name in registered_types()} | {"physics", "coupling", "application", "input_group"}
    if category is not None and category not in categories:
        raise ValueError(f"Unknown category '{category}'. Use one of: {', '.join(sorted(categories))}.")
    out = []
    for name, (_cls, kind, kind_module) in _applications().items():
        if (category is None or kind == category) and (module is None or kind_module == module):
            out.append(name)
    for name in physics.registered():
        cls = physics._REGISTRY[name]
        if (category is None or cls.category == category) and (module is None or cls.module == module):
            out.append(name)
    for name in registered_types():
        if category is not None and object_category(name) != category:
            continue
        if module is not None and object_module(name) != module:
            continue
        out.append(name)
    return out
