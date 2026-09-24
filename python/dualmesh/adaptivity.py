# SPDX-License-Identifier: LGPL-2.1-or-later
"""Adaptive mesh refinement.

An adaptive calculation repeats three steps.  It *estimates* where the error
is, by comparing the computed gradient of the solution with a smoother
gradient recovered from it; it *marks* the elements that carry most of that
error; and it *refines* them, producing a new mesh on which the problem is
solved again.  The three steps are separate functions here, so that a
calculation can replace any one of them, and
:func:`solve_with_adaptive_refinement` runs the usual loop over all three.

The refinement is conforming: no node is left hanging in the middle of a
neighbour's edge.  That is a requirement of the dual mesh method rather than a
convenience.  A hanging node owns a control domain on its own side of the edge
and nothing on the other, so the control domains stop covering the domain
exactly once, flux leaves one control domain without entering another, and the
discretisation loses the conservation property it is built on.  Finite element
codes avoid the problem by constraining the hanging degree of freedom, which
keeps the interpolation continuous, but there is no interpolation argument to
appeal to here.  Conformity is kept by bisecting the longest edge of a marked
triangle and passing the bisection on to the neighbour that shares that edge,
the algorithm of Rivara (1984); this is why refinement is available for
triangular meshes only.

A worked example::

    import dualmesh as dm

    def build(mesh):
        problem = dm.Problem(mesh)
        problem.add_variable("u")
        problem.add_kernel("Diffusion", "diffusion", variable="u")
        problem.add_kernel("BodyForce", "source", variable="u", value=1.0)
        problem.add_boundary_condition(
            "DirichletBC", "walls", variable="u",
            boundary=mesh.sideset_names(), value=0.0)
        return problem

    problem, mesh = dm.solve_with_adaptive_refinement(
        build, initial_mesh, variable="u", num_cycles=4)
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Callable

import numpy as np

from . import _core
from .problem import Problem

__all__ = [
    "mark_by_fraction",
    "mark_by_error_fraction",
    "mark_by_threshold",
    "refine_marked",
    "solve_with_adaptive_refinement",
]


def refine_marked(mesh, marked: Sequence[bool]):
    """Refine the marked elements, keeping the mesh conforming.

    ``marked`` holds one flag per element.  More elements than those marked
    are bisected, because the neighbours of a bisected edge have to be
    bisected too.  Returns ``(refined_mesh, parents)``, where ``parents[i]`` is
    the index in ``mesh`` of the element that element ``i`` of the refined mesh
    came from; a field stored per element is carried over with it.
    """
    flags = [int(bool(m)) for m in np.asarray(marked).ravel()]
    refined, parents = _core.refine_marked(mesh, flags)
    return refined, np.asarray(parents, dtype=np.int64)


def mark_by_fraction(indicators, fraction: float = 0.3) -> np.ndarray:
    """Mark the given fraction of the elements with the largest indicators.

    This is the simplest marking rule: with ``fraction=0.3`` the worst thirty
    per cent of the elements are refined at every cycle, so the mesh grows at a
    predictable rate whatever the shape of the error distribution.
    """
    values = np.asarray(indicators, dtype=float).ravel()
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must lie in (0, 1].")
    count = max(1, int(round(fraction * values.size)))
    marked = np.zeros(values.size, dtype=bool)
    marked[np.argsort(values)[-count:]] = True
    return marked


def mark_by_error_fraction(indicators, fraction: float = 0.5) -> np.ndarray:
    """Mark the smallest set of elements that carries the given share of the
    total squared error.

    This is bulk, or Dörfler, marking: with ``fraction=0.5`` just enough
    elements are refined to account for half of the total error.  It refines
    few elements when the error is concentrated in a few of them and many when
    the error is spread out, which is the behaviour that proofs of convergence
    for adaptive methods rely on.

    W. Dörfler, "A convergent adaptive algorithm for Poisson's equation",
    SIAM Journal on Numerical Analysis 33 (1996) 1106-1124.
    """
    values = np.asarray(indicators, dtype=float).ravel()
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must lie in (0, 1].")
    squared = values**2
    order = np.argsort(squared)[::-1]
    running = np.cumsum(squared[order])
    if running[-1] <= 0.0:
        return np.zeros(values.size, dtype=bool)
    count = int(np.searchsorted(running, fraction * running[-1]) + 1)
    marked = np.zeros(values.size, dtype=bool)
    marked[order[:count]] = True
    return marked


def mark_by_threshold(indicators, threshold: float) -> np.ndarray:
    """Mark every element whose indicator exceeds an absolute threshold."""
    return np.asarray(indicators, dtype=float).ravel() > float(threshold)


def solve_with_adaptive_refinement(
    build_problem: Callable[[object], Problem],
    mesh,
    variable: str,
    num_cycles: int = 3,
    marker: Callable[[np.ndarray], np.ndarray] = mark_by_error_fraction,
    max_elements: int | None = None,
    callback: Callable[[int, Problem, np.ndarray], None] | None = None,
):
    """Solve, estimate, mark and refine, ``num_cycles`` times.

    ``build_problem`` is called with a mesh and must return a solved-ready
    :class:`~dualmesh.problem.Problem` defined on it: the whole problem is
    rebuilt on every mesh rather than transferred, which keeps the boundary
    conditions and the material state exactly as the user wrote them.
    ``marker`` turns the vector of indicators into a boolean array of elements
    to refine.  ``max_elements`` stops the loop early once the mesh has grown
    past that size, and ``callback(cycle, problem, indicators)`` is called after
    every solve, which is where a calculation writes output or records a
    convergence history.

    Returns ``(problem, mesh)`` for the last, finest mesh.
    """
    if num_cycles < 1:
        raise ValueError("num_cycles must be at least one.")
    problem = None
    for cycle in range(num_cycles):
        problem = build_problem(mesh)
        problem.solve()
        indicators = np.asarray(problem.error_indicator(variable))
        if callback is not None:
            callback(cycle, problem, indicators)
        if cycle == num_cycles - 1:
            break
        if max_elements is not None and mesh.num_elements >= max_elements:
            break
        marked = marker(indicators)
        if not marked.any():
            break
        mesh, _ = refine_marked(mesh, marked)
    return problem, mesh
