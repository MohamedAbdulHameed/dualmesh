# SPDX-License-Identifier: LGPL-2.1-or-later
"""Setups that several test files share: the unit square mesh and the lid-driven cavity."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import dualmesh
import numpy

_root = Path(__file__).resolve().parents[2]


def unit_square(n, element_type="Quad4"):
    """The unit square with n by n elements of element_type."""
    return dualmesh.generate_rectangle_mesh(
        x_min=0.0,
        x_max=1.0,
        y_min=0.0,
        y_max=1.0,
        num_x_elements=n,
        num_y_elements=n,
        element_type=element_type,
    )


def lid_driven_cavity_conditions(problem):
    """The conditions of the lid-driven cavity on the unit square, as Dirichlet objects: the lid
    (top) moves at velocity_x = 1, and the other walls are at rest.  The walls are added last, so
    that the two top corner nodes of the node based methods belong to the walls: the lid does
    not leak."""
    problem.add_boundary_condition("Dirichlet", "top", variable="velocity_x", value=1.0)
    problem.add_boundary_condition("Dirichlet", "top", variable="velocity_y", value=0.0)
    for variable in ("velocity_x", "velocity_y"):
        problem.add_boundary_condition(
            "Dirichlet",
            ["left", "right", "bottom"],
            variable=variable,
            value=0.0,
        )


def lid_driven_cavity(method, n, element_type="Quad4", **flow):
    """The lid-driven cavity on the unit square with n by n elements of element_type, solved by
    method, with the fluid_dynamics parameters flow.  Not solved."""
    mesh = dualmesh.generate_rectangle_mesh(0, 1, 0, 1, n, n, element_type=element_type)
    problem = dualmesh.Problem(mesh, method=method)
    problem.add_physics("fluid_dynamics", **flow)
    lid_driven_cavity_conditions(problem)
    return problem


def Ghia_differences(problem):
    """The largest differences of velocity_x along the vertical centreline and of velocity_y
    along the horizontal centreline of the cavity from the values of Ghia, Ghia and Shin (1982)
    at Re = 100."""
    spec = importlib.util.spec_from_file_location(
        "compare_cavity", _root / "verification" / "openfoam" / "compare_cavity.py"
    )
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    u = problem.sample(
        "velocity_x", numpy.column_stack([numpy.full_like(reference.ghia_y, 0.5), reference.ghia_y])
    )
    v = problem.sample(
        "velocity_y", numpy.column_stack([reference.ghia_x, numpy.full_like(reference.ghia_x, 0.5)])
    )
    return numpy.max(numpy.abs(u - reference.ghia_u)), numpy.max(numpy.abs(v - reference.ghia_v))
