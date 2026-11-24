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
discretisation loses the conservation property it is built on.  Conformity is
kept by newest-vertex bisection: a marked triangle is cut through the middle of
its refinement edge, and every triangle that shares a cut edge is cut too.  Its
triangles fall into finitely many shapes however often they are cut, and it is
the refinement for which adaptive methods are proved to converge at the optimal
rate.  :class:`~dualmesh._core.Bisection` keeps the history of the cuts, so a
mesh can also be coarsened back, and it is available for triangular meshes.

A worked example::

    import dualmesh


    def build(mesh):
        problem = dualmesh.Problem(mesh)
        problem.add_variable("u")
        problem.add_kernel("diffusion", variable="u")
        problem.add_kernel("body_force", variable="u", value=1.0)
        problem.add_boundary_condition("Dirichlet", mesh.sideset_names(), variable="u", value=0.0)
        return problem


    result = dualmesh.solve_with_adaptive_refinement(
        build, initial_mesh, variable="u", num_cycles=4
    )
    print(result.summary())
    problem = result.problem  # the problem on the finest mesh
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Callable

import numpy

from . import _core
from .console import header, inner_study, report_level
from .engine import Problem
from .tables import ResultTables

__all__ = [
    "AdaptivityResult",
    "mark_by_fraction",
    "mark_by_error_fraction",
    "mark_by_threshold",
    "refine_marked",
    "solve_with_adaptive_refinement",
]


class _Hierarchy:
    """The refinement hierarchy of a mesh, in levels that halve the size of a cell: newest-vertex
    or longest-edge bisection of triangles and tetrahedra (d bisections a level), and subdivision
    with hanging nodes of quadrilaterals and hexahedra (one split a level)."""

    def __init__(self, mesh):
        cells = mesh.cells()
        kind = cells[0][0] if cells else "Tri3"
        if kind in ("Quad4", "Hex8"):
            self._core = _core.Subdivision(mesh)
            self._per_level = 1
            self._steps = self._core.levels
        else:
            self._core = _core.Bisection(mesh)
            self._per_level = mesh.dimension
            self._steps = self._core.generations

    def mesh(self):
        return self._core.mesh()

    def refinable(self, max_level: int) -> numpy.ndarray:
        """Whether each cell is below the level max_level."""
        return numpy.asarray(self._steps()) < self._per_level * max_level

    def coarsenable(self) -> numpy.ndarray:
        """Whether each cell is finer than its cell of the initial mesh."""
        return numpy.asarray(self._steps()) > 0

    def adapt(self, refine, coarsen, max_level: int) -> bool:
        """Coarsen where coarsen marks, then refine where refine marks, in one change of the mesh."""
        return self._core.adapt(
            [int(bool(m)) for m in refine],
            [int(bool(m)) for m in coarsen],
            self._per_level * max_level,
        )

    def refine(self, marked) -> bool:
        return self._core.refine([int(bool(m)) for m in marked], 2**20)

    def project(self, name: str, projection) -> None:
        """Move the nodes of side set ``name`` onto a curved boundary: ``projection((x, y, z))``
        gives the point of each."""
        if not hasattr(self._core, "project"):
            raise ValueError(
                "A curved boundary is followed on meshes of triangles or tetrahedra in this "
                "version."
            )
        self._core.project(name, projection)

    def transfer(self, values, width: int = 1) -> numpy.ndarray:
        """Values at the nodes of the mesh before the last change, carried to the current one."""
        return numpy.asarray(self._core.transfer(numpy.asarray(values, dtype=float), width))


def refine_marked(mesh, marked: Sequence[bool]):
    """Refine the marked elements once by newest-vertex bisection, keeping the
    mesh conforming.

    ``marked`` holds one flag per element.  More elements than those marked
    are bisected, because the neighbours of a bisected edge have to be
    bisected too.  Returns ``(refined_mesh, parents)``, where ``parents[i]`` is
    the index in ``mesh`` of the element that element ``i`` of the refined mesh
    came from; a field stored per element is carried over with it.
    """
    flags = [int(bool(m)) for m in numpy.asarray(marked).ravel()]
    refined, parents = _core.refine_marked(mesh, flags)
    return refined, numpy.asarray(parents, dtype=numpy.int64)


def mark_by_fraction(indicators, fraction: float = 0.3) -> numpy.ndarray:
    """Mark the given fraction of the elements with the largest indicators.

    This is the simplest marking rule: with ``fraction=0.3`` the worst thirty
    per cent of the elements are refined at every cycle, so the mesh grows at a
    predictable rate whatever the shape of the error distribution.
    """
    values = numpy.asarray(indicators, dtype=float).ravel()
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must lie in (0, 1].")
    count = max(1, int(round(fraction * values.size)))
    marked = numpy.zeros(values.size, dtype=bool)
    marked[numpy.argsort(values)[-count:]] = True
    return marked


def mark_by_error_fraction(indicators, fraction: float = 0.5) -> numpy.ndarray:
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
    values = numpy.asarray(indicators, dtype=float).ravel()
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must lie in (0, 1].")
    squared = values**2
    order = numpy.argsort(squared)[::-1]
    running = numpy.cumsum(squared[order])
    if running[-1] <= 0.0:
        return numpy.zeros(values.size, dtype=bool)
    count = int(numpy.searchsorted(running, fraction * running[-1]) + 1)
    marked = numpy.zeros(values.size, dtype=bool)
    marked[order[:count]] = True
    return marked


def mark_by_threshold(indicators, threshold: float) -> numpy.ndarray:
    """Mark every element whose indicator exceeds an absolute threshold."""
    return numpy.asarray(indicators, dtype=float).ravel() > float(threshold)


@dataclass
class AdaptivityResult(ResultTables):
    """The outcome of :func:`solve_with_adaptive_refinement`: the problem
    solved on the finest mesh, that mesh, and one record per cycle (the
    number of elements and unknowns, and the estimated error, the square
    root of the sum of the squared indicators)."""

    problem: Problem
    mesh: object
    num_elements: list = field(default_factory=list)
    num_dofs: list = field(default_factory=list)
    estimated_error: list = field(default_factory=list)

    @property
    def tables(self) -> dict:
        """``cycles``: one row per cycle of refinement."""
        from .tables import Table

        return {
            "cycles": Table(
                ["cycle", "num_elements", "num_dofs", "estimated_error"],
                None,
                [
                    [i, e, d, eta]
                    for i, (e, d, eta) in enumerate(
                        zip(self.num_elements, self.num_dofs, self.estimated_error)
                    )
                ],
                title=f"Adaptive refinement, {len(self.num_elements)} cycles",
            )
        }

    def summary(self) -> str:
        return self.tables["cycles"].format()

    def to_dict(self) -> dict:
        return {
            "study": "adaptive_refinement",
            "num_elements": list(self.num_elements),
            "num_dofs": list(self.num_dofs),
            "estimated_error": list(self.estimated_error),
        }

    def write_json(self, path) -> None:
        """Write :meth:`to_dict` to a JSON file."""
        from .parameters import write_json_numbers

        write_json_numbers(self.to_dict(), path)


def solve_with_adaptive_refinement(
    build_problem: Callable[[object], Problem],
    mesh,
    variable: str | Sequence[str],
    num_cycles: int = 3,
    marker: Callable[[numpy.ndarray], numpy.ndarray] = mark_by_error_fraction,
    max_elements: int | None = None,
    callback: Callable[[int, Problem, numpy.ndarray], None] | None = None,
    report: str = "full",
    boundaries: dict | None = None,
):
    """Solve, estimate, mark and refine, ``num_cycles`` times.

    ``build_problem`` is called with a mesh and must return a solved-ready
    :class:`~dualmesh.Problem` defined on it: the whole problem is
    rebuilt on every mesh rather than transferred, which keeps the boundary
    conditions and the material state exactly as the user wrote them.
    ``marker`` turns the vector of indicators into a boolean array of elements
    to refine.  ``max_elements`` stops the loop early once the mesh has grown
    past that size, and ``callback(cycle, problem, indicators)`` is called after
    every solve, which is where a calculation writes output or records a
    convergence history.

    ``variable`` names the variable whose error indicator drives the marking,
    or several (the components of a velocity), whose indicators add as
    squares.  ``boundaries`` maps side set names to the projections of their
    curves, ``projection((x, y, z))`` giving the point of the curve for a
    node: after each refinement, the nodes of the side set move onto it, so
    that refining there also refines the geometry (a mesh of triangles or
    tetrahedra).  In a distributed run every process holds the whole mesh and
    marks it from the indicators of the whole mesh.

    ``report`` sets what the study prints: ``"full"`` (the default: a line
    for each cycle and the summary), ``"summary"`` (one line that names the
    study, then the summary) or ``"none"``.  The solves inside the study
    print nothing.

    Returns an :class:`AdaptivityResult` with the problem on the last,
    finest mesh, that mesh and the history of the cycles.
    """
    if num_cycles < 1:
        raise ValueError("num_cycles must be at least one.")
    level = report_level(report, "solve_with_adaptive_refinement")
    if level == "full":
        print(header("adaptive refinement"))
    names = [variable] if isinstance(variable, str) else list(variable)
    root = _core.mpi_rank() == 0
    if level != "none" and root:
        print(
            f"dualmesh adaptive refinement of {', '.join(names)}: at most {num_cycles} cycles",
            flush=True,
        )
    result = AdaptivityResult(problem=None, mesh=mesh)
    # One hierarchy of bisections for the whole loop, so that the triangles keep their refinement
    # edges from one cycle to the next, as newest-vertex bisection needs.
    hierarchy = _Hierarchy(mesh)
    for cycle in range(num_cycles):
        with inner_study():
            problem = build_problem(mesh)
            problem.solve()
        indicators = numpy.sqrt(sum(problem.gathered_error_indicator(name) ** 2 for name in names))
        result.problem, result.mesh = problem, mesh
        result.num_elements.append(int(mesh.num_elements))
        result.num_dofs.append(int(problem.num_active_dofs()))
        result.estimated_error.append(float(numpy.sqrt(numpy.sum(indicators**2))))
        if level == "full" and root:
            print(
                f"  cycle {cycle}: {mesh.num_elements} elements, {problem.num_active_dofs()} unknowns, estimated error {result.estimated_error[-1]:.6g}",
                flush=True,
            )
        if callback is not None:
            callback(cycle, problem, indicators)
        if cycle == num_cycles - 1:
            break
        if max_elements is not None and mesh.num_elements >= max_elements:
            break
        marked = marker(indicators)
        if not marked.any():
            break
        hierarchy.refine(marked)
        for name, projection in (boundaries or {}).items():
            hierarchy.project(name, projection)
        mesh = hierarchy.mesh()
    if level != "none" and root:
        print(result.summary())
    return result
