# SPDX-License-Identifier: LGPL-2.1-or-later
"""Adaptive mesh refinement.

An adaptive calculation repeats three steps.  It *estimates* where the error
is, by comparing the computed gradient of the solution with a smoother
gradient recovered from it; it *marks* the elements that carry most of that
error; and it *refines* them, producing a new mesh on which the problem is
solved again.  The three steps are separate functions here, so that a
calculation can replace any one of them.
:func:`solve_with_adaptive_refinement` runs the usual loop over all three for
a steady problem, and :func:`solve_transient_with_adaptive_refinement` adapts
the mesh of a transient run every few time steps.

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
from .engine import Problem, reached_end
from .tables import ResultTables, Table

__all__ = [
    "AdaptivityResult",
    "TransientAdaptivityResult",
    "mark_by_fraction",
    "mark_by_error_fraction",
    "mark_by_threshold",
    "refine_marked",
    "solve_transient_with_adaptive_refinement",
    "solve_with_adaptive_refinement",
]


class _Hierarchy:
    """The refinement hierarchy of a mesh, in levels that halve the size of a cell: newest-vertex
    or longest-edge bisection of triangles and tetrahedra (d bisections a level), and subdivision
    with hanging nodes of quadrilaterals and hexahedra (one split a level)."""

    def __init__(self, mesh):
        kind = mesh.element_type(0) if mesh.num_elements else "Tri3"
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
        quadratic = any(_core.element_is_quadratic(t) for t in engine.mesh.element_types())
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
    """The cells to refine, those that ``marker`` marks among the cells below the level
    ``max_level``, and the cells to coarsen, those of the share ``coarsen_fraction`` of the error
    that :func:`_coarsening` gives.  The marker sees the indicators of the cells at ``max_level``
    as zero: those cells cannot be refined, and if they carried the share of the error that the
    marker picks, nothing would be refined while the other cells still hold error."""
    refinable = hierarchy.refinable(max_level)
    values = numpy.where(refinable, numpy.asarray(indicators, dtype=float), 0.0)
    refine = numpy.asarray(marker(values), dtype=bool) & refinable
    return refine, _coarsening(indicators, coarsen_fraction, refine, hierarchy.coarsenable())


def _variable_names(variable: str | Sequence[str]) -> list:
    """The variables whose error indicators drive the marking: one name, or several."""
    return [variable] if isinstance(variable, str) else list(variable)


def _indicators(problem: Problem, names: list) -> numpy.ndarray:
    """The error indicator of each cell, the square root of the sum of the squared indicators of
    the variables ``names``.  Every process holds the same hierarchy of the whole mesh, so the
    indicators are those of the whole mesh, and every process adapts it the same way."""
    return numpy.sqrt(sum(problem.gathered_error_indicator(name) ** 2 for name in names))


def _state(problem: Problem) -> dict:
    """The values of every variable of the problem, at the nodes of the whole mesh, by name."""
    return {name: problem.gathered_values(name) for name in problem._problem.variable_names()}


def _set_state(problem: Problem, state: dict) -> None:
    """Sets the values of the variables of the problem from a state of :func:`_state`."""
    for name, values in state.items():
        problem.set_gathered_values(name, values)


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

    def _record(self, mesh, problem: Problem, indicators) -> None:
        """Appends the record of one cycle: the number of elements and unknowns, and the
        estimated error."""
        self.num_elements.append(int(mesh.num_elements))
        self.num_dofs.append(int(problem.num_global_dofs))
        self.estimated_error.append(float(numpy.sqrt(numpy.sum(indicators**2))))

    def _records(self) -> dict:
        """The records of the cycles, as lists by name."""
        return {
            "num_elements": list(self.num_elements),
            "num_dofs": list(self.num_dofs),
            "estimated_error": list(self.estimated_error),
        }

    @property
    def tables(self) -> dict:
        """``cycles``: one row per cycle of refinement."""
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
        return {"study": "adaptive_refinement", **self._records()}


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
    names = _variable_names(variable)
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
            _set_state(problem, start)
            problem.solve(**options)
        indicators = _indicators(problem, names)
        result.problem, result.mesh = problem, mesh
        result._record(mesh, problem, indicators)
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
        reached = _state(problem)
        if not hierarchy.adapt(refine, coarsen, maximum_refinement_level):
            break
        for name, projection in (boundaries or {}).items():
            hierarchy.project(name, projection)
        mesh = hierarchy.mesh()
        start = hierarchy.carry(problem, reached)
    if level != "none":
        print(result.summary())
    return result


@dataclass
class TransientAdaptivityResult(AdaptivityResult):
    """The outcome of :func:`solve_transient_with_adaptive_refinement`: the
    problem on the last mesh, that mesh, the times of the accepted steps in
    ``times``, the history of the post-processors across all the meshes in
    ``history``, and one record per interval of steps: the time it reached in
    ``interval_end_times``, and the number of elements and unknowns and the
    estimated error at that time."""

    interval_end_times: list = field(default_factory=list)
    times: list = field(default_factory=list)
    history: Table | None = None

    @property
    def tables(self) -> dict:
        """``intervals``: one row per interval of steps.  ``history``: the
        post-processors at every accepted step, across the meshes."""
        rows = [
            [i, t, e, d, eta]
            for i, (t, e, d, eta) in enumerate(
                zip(
                    self.interval_end_times,
                    self.num_elements,
                    self.num_dofs,
                    self.estimated_error,
                )
            )
        ]
        return {
            "intervals": Table(
                ["interval", "time", "num_elements", "num_dofs", "estimated_error"],
                ["", "s", "", "", ""],
                rows,
                title=f"Adaptive transient run, {len(rows)} intervals",
            ),
            "history": self.history,
        }

    def summary(self) -> str:
        return self.tables["intervals"].format()

    def to_dict(self) -> dict:
        return {
            "study": "transient_adaptive_refinement",
            "times": list(self.times),
            "interval_end_times": list(self.interval_end_times),
            **self._records(),
        }


#: The relative tolerance of the end test of the transient loop: the run has reached its end
#: time within this fraction of the end time.
def _rows_after(rows: list, history: Table) -> list:
    """The rows of the post-processor table of a mesh after the time where the rows of the meshes
    before it end."""
    return [row for row in history.rows if not rows or row[0] > rows[-1][0]]


def _adapted_problem(hierarchy, problem, build_problem, refine, coarsen, max_level):
    """Adapts the mesh and builds the problem on it, which goes on from the values and the
    velocities that ``problem`` reached, carried to the new mesh.  It returns the new problem and
    its mesh, or None when the mesh does not change."""
    state = _state(problem)
    # The velocity of a second-order equation is a state of the step too: the new problem
    # goes on from the velocity reached, not from that of the initial condition.
    velocity = problem.gathered_velocities()
    if not hierarchy.adapt(refine, coarsen, max_level):
        return None
    state = hierarchy.carry(problem, state)
    velocity = hierarchy.carry(problem, velocity)
    mesh = hierarchy.mesh()
    problem = build_problem(mesh)
    problem.set_gathered_velocities(velocity)
    _set_state(problem, state)
    return problem, mesh


def solve_transient_with_adaptive_refinement(
    build_problem: Callable[[object], Problem],
    mesh,
    variable: str | Sequence[str],
    end_time: float,
    time_step: float,
    interval: int,
    marker: Callable[[numpy.ndarray], numpy.ndarray] = mark_by_error_fraction,
    callback: Callable[[int, Problem, numpy.ndarray], None] | None = None,
    report: str = "full",
    coarsen_fraction: float = 0.0,
    maximum_refinement_level: int | None = None,
    start_time: float = 0.0,
    output=None,
    **options,
):
    """Run a transient problem from ``start_time`` to ``end_time``, adapting the mesh after every
    ``interval`` accepted time steps.

    ``build_problem`` is called with a mesh and returns a :class:`~dualmesh.Problem` defined on
    it, ready to solve, as for :func:`solve_with_adaptive_refinement`.  The loop repeats these
    steps until it reaches ``end_time``:

    #. It takes ``interval`` accepted steps with :meth:`~dualmesh.Problem.solve_transient`.
    #. It computes the indicators of ``variable`` in the state reached, and calls
       ``callback(interval_index, problem, indicators)``.
    #. It marks the cells to refine with ``marker``, among the cells below
       ``maximum_refinement_level``, and the cells to coarsen with ``coarsen_fraction``, as
       :func:`solve_with_adaptive_refinement` does.
    #. It changes the mesh, builds the problem on the new mesh, and carries to it the value of
       every variable and the velocity of every equation with a second time derivative.  Under
       DMCDM on linear elements each field keeps its content, its values times the measures of
       the control domains.
    #. It goes on from the time reached, with the step that the step controller chose last.

    The loop also ends after an interval that does not advance the time, as a solve that fails
    at its first step leaves it when the solver option ``error_on_divergence`` is off.

    Each interval is a new call of :meth:`~dualmesh.Problem.solve_transient`, so the step
    controller restarts at each adaptation: the first step of an interval is the step that the
    controller chose before the change, and the controller keeps no history of the steps before
    it.  ``time_step`` is the first step of the run.

    ``variable`` names the variable whose error indicator drives the marking, or several, whose
    indicators add as squares.  ``output`` is an :class:`~dualmesh.Output` group that every
    interval writes to, so the files of all the meshes form one series.  ``report`` sets what
    the loop prints: ``"full"`` (a line at the start, one at each change of the mesh and one at
    the end), ``"summary"`` (the lines at the start and the end) or ``"none"``.  The runs inside
    the loop print nothing.  The other keyword arguments are the options of
    :meth:`~dualmesh.Problem.solve_transient`, such as ``time_stepper``, and the solver options,
    for every interval.

    Returns a :class:`TransientAdaptivityResult` with the problem and the mesh at
    ``end_time``, the times of the accepted steps, the history of the post-processors across
    the meshes and one record per interval.
    """
    if interval < 1:
        raise ValueError("interval must be at least one.")
    level = report_level(report, "solve_transient_with_adaptive_refinement")
    if level != "none":
        print(f"dualmesh transient solve, the mesh adapted every {interval} steps", flush=True)
    names = _variable_names(variable)
    result = TransientAdaptivityResult(problem=None, mesh=mesh)
    # One hierarchy for the whole run, as in the steady loop.
    hierarchy = _Hierarchy(mesh)
    problem = build_problem(mesh)
    rows: list = []
    history: Table | None = None
    run_start = start_time
    done = False
    while not done:
        with inner_study():
            run = problem.solve_transient(
                end_time=end_time,
                time_step=time_step,
                start_time=start_time,
                max_steps=interval,
                output=output,
                **options,
            )
        result.times += [time for time, _ in run.step_history]
        history = problem.postprocessor_table()
        rows += _rows_after(rows, history)
        previous, start_time = start_time, float(problem.time)
        indicators = _indicators(problem, names)
        result.interval_end_times.append(start_time)
        result._record(mesh, problem, indicators)
        if callback is not None:
            callback(len(result.num_elements) - 1, problem, indicators)
        # An interval that does not advance the time would repeat itself, so the loop ends there
        # too.
        done = reached_end(start_time, run_start, end_time) or start_time <= previous
        if not done:
            time_step = run.next_time_step
            refine, coarsen = _marks(
                hierarchy, indicators, marker, coarsen_fraction, maximum_refinement_level
            )
            adapted = _adapted_problem(
                hierarchy, problem, build_problem, refine, coarsen, maximum_refinement_level
            )
            if adapted is not None:
                problem, mesh = adapted
                if level == "full":
                    print(f"  t = {start_time:.6g} s: {mesh.num_elements} cells", flush=True)
    if level != "none":
        print(f"  {len(result.times)} accepted steps, {mesh.num_elements} cells at the end")
    result.problem, result.mesh = problem, mesh
    result.history = Table(history.names, history.units, rows)
    return result
