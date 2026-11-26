# SPDX-License-Identifier: LGPL-2.1-or-later
"""Adaptive mesh refinement.

An adaptive calculation repeats three steps.  It *estimates* where the error
is, by comparing the computed gradient of the solution with a smoother
gradient recovered from it; it *marks* the elements that carry most of that
error; and it *refines* them, producing a new mesh on which the problem is
solved again.  The three steps are separate functions here, so that a
calculation can replace any one of them, and
:func:`solve_with_adaptive_refinement` runs the usual loop over all three.

Triangles and tetrahedra are refined by bisection, which keeps the mesh
conforming.  Newest-vertex bisection keeps the triangles in finitely many shapes
however often they are cut, and it is the refinement for which adaptive methods
are proved to converge at the optimal rate.  Tetrahedra are cut across their
longest edge.  Quadrilaterals and hexahedra split into :math:`2^d` children, with
:math:`d` the dimension, and nodes hang on the sides of the neighbours that stay
whole, with a 2:1 balance across every side.  FEM constrains a hanging value to
the side it lies on.  DMCDM splits the large cell into control-domain pieces that
meet those of the small cells, so the balance stays conservative.  The hierarchy
keeps the history of the cuts, so a mesh can also be coarsened back.

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

import math
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
        else:
            self._core = _core.Bisection(mesh)
            self._per_level = mesh.dimension

    def mesh(self):
        return self._core.mesh()

    def _cuts(self, max_level: int | None) -> int:
        """The cuts of a cell at the level max_level, which None leaves unbounded."""
        return 2**20 if max_level is None else self._per_level * max_level

    def refinable(self, max_level: int | None) -> numpy.ndarray:
        """Whether each cell is below the level max_level."""
        return numpy.asarray(self._core.levels()) < self._cuts(max_level)

    def coarsenable(self) -> numpy.ndarray:
        """Whether each cell is finer than its cell of the initial mesh."""
        return numpy.asarray(self._core.levels()) > 0

    def adapt(self, refine, coarsen, max_level: int | None) -> bool:
        """Coarsen where coarsen marks, then refine where refine marks, in one change of the
        mesh.  It returns whether the mesh changed."""
        return self._core.adapt(
            [int(bool(m)) for m in refine],
            [int(bool(m)) for m in coarsen],
            self._cuts(max_level),
        )

    def refine(self, marked) -> bool:
        return self._core.refine([int(bool(m)) for m in marked], self._cuts(None))

    def project(self, name: str, projection) -> None:
        """Move the nodes of side set ``name`` onto a curved boundary: ``projection((x, y, z))``
        gives the point of each.  A hanging node stays on the straight side of its large
        neighbour until that side is split."""
        self._core.project(name, projection)

    def origins(self) -> numpy.ndarray:
        """The cell of the initial mesh that each cell comes from."""
        return numpy.asarray(self._core.origins())

    def transfer(self, values, width: int = 1) -> numpy.ndarray:
        """Values at the nodes of the mesh before the last change, carried to the current one."""
        return numpy.asarray(self._core.transfer(numpy.asarray(values, dtype=float), width))

    def carry(self, engine, fields: dict) -> dict:
        """Fields at the nodes of the mesh before the last change, by name, carried to the
        current one.  ``engine`` is the problem on the mesh before the change.

        Under DMCDM on linear elements each field keeps its content, the sum of its values times
        the measures of the control domains: a node that the coarsening removes gives its content
        to the nodes that interpolate it, in a measure-weighted mean, and a new node interpolates.
        Under FEM, and on quadratic elements, whose shape functions are negative at some removed
        nodes, each field is interpolated, which carries the fields exactly under refinement.
        """
        from .meshing import _QUADRATIC

        quadratic = engine.mesh.num_elements > 0 and engine.mesh.element_type(0) in _QUADRATIC
        if engine.method != "DMCDM" or quadratic or not fields:
            return {name: self.transfer(values) for name, values in fields.items()}
        measures = numpy.asarray(engine.gathered_node_measures(), dtype=float)
        return {
            name: numpy.asarray(
                self._core.restrict(numpy.asarray(values, dtype=float), 1, measures)
            )
            for name, values in fields.items()
        }


def refine_marked(mesh, marked: Sequence[bool]):
    """Refine the marked elements once.

    A triangle or a tetrahedron is bisected, and so are the neighbours that keep the mesh
    conforming.  A quadrilateral or a hexahedron splits into :math:`2^d` children, with
    :math:`d` the dimension, and so do the neighbours that keep a 2:1 balance, with nodes
    hanging on the sides of the cells that stay whole.  ``marked`` holds one flag per element.
    Returns
    ``(refined_mesh, parents)``, where ``parents[i]`` is the index in ``mesh`` of the element that
    element ``i`` of the refined mesh came from; a field stored per element is carried over with
    it.
    """
    hierarchy = _Hierarchy(mesh)
    hierarchy.refine(numpy.asarray(marked).ravel())
    return hierarchy.mesh(), numpy.asarray(hierarchy.origins(), dtype=numpy.int64)


def mark_by_fraction(indicators, fraction: float = 0.3) -> numpy.ndarray:
    """Mark the given fraction of the elements with the largest indicators.

    This is the simplest marking rule: with ``fraction=0.3`` the worst thirty
    per cent of the elements are refined at every cycle, so the mesh grows at a
    predictable rate whatever the shape of the error distribution.  It marks
    ceil(fraction N) of the N elements, and none when every indicator is zero.
    """
    values = numpy.asarray(indicators, dtype=float).ravel()
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must lie in (0, 1].")
    marked = numpy.zeros(values.size, dtype=bool)
    if values.size == 0 or not values.max() > 0.0:
        return marked
    count = math.ceil(fraction * values.size)
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


def _coarsening(indicators, fraction: float, refine, coarsenable) -> numpy.ndarray:
    """The largest set of the refined cells outside ``refine`` whose squared indicators sum to at
    most ``fraction`` of the total, the smallest first."""
    squared = numpy.asarray(indicators, dtype=float) ** 2
    marked = numpy.zeros(squared.size, dtype=bool)
    candidates = numpy.flatnonzero(~refine & coarsenable)
    if fraction <= 0.0 or candidates.size == 0:
        return marked
    order = candidates[numpy.argsort(squared[candidates])]
    count = int(numpy.searchsorted(numpy.cumsum(squared[order]), fraction * squared.sum(), "right"))
    marked[order[:count]] = True
    return marked


def _marks(hierarchy: _Hierarchy, indicators, marker, coarsen_fraction: float, max_level):
    """The cells to refine, those that ``marker`` marks below the level ``max_level``, and the
    cells to coarsen, those of the share ``coarsen_fraction`` of the error that :func:`_coarsening`
    gives."""
    refine = numpy.asarray(marker(indicators), dtype=bool) & hierarchy.refinable(max_level)
    return refine, _coarsening(indicators, coarsen_fraction, refine, hierarchy.coarsenable())


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
    coarsen_fraction: float = 0.0,
    maximum_refinement_level: int | None = None,
    **options,
):
    """Solve, estimate, mark and adapt, at most ``num_cycles`` times.

    ``build_problem`` is called with a mesh and returns a
    :class:`~dualmesh.Problem` defined on it, ready to solve.  The problem is
    built anew on every mesh, which keeps its boundary conditions and
    properties exactly as the user wrote them, and its solve starts from the
    solution of the last mesh, carried to the new one: under DMCDM each field
    keeps its content, its values times the measures of the control domains.
    ``marker`` turns the vector of indicators into a boolean array of elements
    to refine.  ``coarsen_fraction`` coarsens the refined elements that carry
    the smallest indicators, up to that share of the total squared error, and
    ``maximum_refinement_level`` bounds the levels of refinement, each of which
    halves the size of a cell.  The loop stops when no element below that level
    is marked, or once the mesh has grown past ``max_elements``.
    ``callback(cycle, problem, indicators)`` is called after every solve, which
    is where a calculation writes output or records a convergence history.

    ``variable`` names the variable whose error indicator drives the marking,
    or several (the components of a velocity), whose indicators add as
    squares.  ``boundaries`` maps side set names to the projections of their
    curves, ``projection((x, y, z))`` giving the point of the curve for a
    node: after each refinement, the nodes of the side set move onto it, so
    that refining there also refines the geometry.  A hanging node stays on
    the straight side of its large neighbour until that side is split.  In a
    distributed run every process holds the whole mesh and
    marks it from the indicators of the whole mesh.

    ``report`` sets what the study prints: ``"full"`` (the default: a line
    for each cycle and the summary), ``"summary"`` (one line that names the
    study, then the summary) or ``"none"``.  The solves inside the study
    print nothing.  The other keyword arguments are the solver options of
    :meth:`~dualmesh.Problem.solve`.

    Returns an :class:`AdaptivityResult` with the problem on the last,
    finest mesh, that mesh and the history of the cycles.
    """
    if num_cycles < 1:
        raise ValueError("num_cycles must be at least one.")
    level = report_level(report, "solve_with_adaptive_refinement")
    if level == "full":
        print(header("adaptive refinement"))
    names = [variable] if isinstance(variable, str) else list(variable)
    if level != "none":
        print(
            f"dualmesh adaptive refinement of {', '.join(names)}: at most {num_cycles} cycles",
            flush=True,
        )
    result = AdaptivityResult(problem=None, mesh=mesh)
    # One hierarchy for the whole loop, so that the triangles keep their refinement edges from
    # one cycle to the next, as newest-vertex bisection needs, and a refined cell can coarsen.
    hierarchy = _Hierarchy(mesh)
    start = {}
    for cycle in range(num_cycles):
        with inner_study():
            problem = build_problem(mesh)
            # The solve starts from the solution of the last mesh, carried to this one.
            for name, values in start.items():
                problem.set_gathered_values(name, values)
            problem.solve(**options)
        # Every process holds the same hierarchy of the whole mesh, so the indicators are those
        # of the whole mesh, and every process adapts it the same way.
        indicators = numpy.sqrt(sum(problem.gathered_error_indicator(name) ** 2 for name in names))
        result.problem, result.mesh = problem, mesh
        result.num_elements.append(int(mesh.num_elements))
        result.num_dofs.append(int(problem.num_global_dofs))
        result.estimated_error.append(float(numpy.sqrt(numpy.sum(indicators**2))))
        if level == "full":
            print(
                f"  cycle {cycle}: {mesh.num_elements} elements, {problem.num_global_dofs} unknowns, estimated error {result.estimated_error[-1]:.6g}",
                flush=True,
            )
        if callback is not None:
            callback(cycle, problem, indicators)
        if cycle == num_cycles - 1:
            break
        if max_elements is not None and mesh.num_elements >= max_elements:
            break
        refine, coarsen = _marks(
            hierarchy, indicators, marker, coarsen_fraction, maximum_refinement_level
        )
        if not refine.any():
            break
        reached = {
            name: problem.gathered_values(name) for name in problem._problem.variable_names()
        }
        if not hierarchy.adapt(refine, coarsen, maximum_refinement_level):
            break
        for name, projection in (boundaries or {}).items():
            hierarchy.project(name, projection)
        mesh = hierarchy.mesh()
        start = hierarchy.carry(problem, reached)
    if level != "none":
        print(result.summary())
    return result
