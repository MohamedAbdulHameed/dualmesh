# SPDX-License-Identifier: LGPL-2.1-or-later
"""Console reports: what was run, with which inputs, and how it went.

Every report is plain text in aligned columns, so that it reads well in a
terminal and can be pasted into a log or a report.  Three are provided:

* :func:`header` -- the version, the build and the machine;
* :func:`problem_report` -- the mesh, the variables and every object with all
  of its parameters, each marked ``(default)`` when the user did not set it,
  so that no choice made on the user's behalf goes unannounced;
* :func:`solve_report` -- the outcome of a solve: convergence, Newton and
  linear iterations, and the time steps of a transient run.

The values themselves, for post-processing, are written by the problem and by
the post-processors to CSV files; these reports are for reading.
"""

from __future__ import annotations

import contextlib
import contextvars
import datetime
import platform
import socket
from collections.abc import Sequence

from . import _core
from ._core import InputError

RULE = "-" * 79

#: The levels of the parameter ``report`` of every study.  ``"full"`` (the
#: default) prints the inputs with their defaults marked, the progress and
#: the summary.  ``"summary"`` prints one line that names the study and the
#: summary at the end.  ``"none"`` prints nothing.
REPORT_LEVELS = ("full", "summary", "none")

_INNER_STUDY = contextvars.ContextVar("dualmesh_inner_study", default=False)


def report_level(report: str, where: str) -> str:
    """The report level that a study prints at.  A study inside a different
    study (for example, a solve in an uncertainty study) prints nothing."""
    if report not in REPORT_LEVELS:
        raise InputError(f"{where}: report must be one of {', '.join(REPORT_LEVELS)}, not '{report}'.")
    return "none" if _INNER_STUDY.get() else report


@contextlib.contextmanager
def inner_study():
    """The studies in this context print nothing: a study that runs other
    studies (an uncertainty study, a convergence study) runs them in it."""
    token = _INNER_STUDY.set(True)
    try:
        yield
    finally:
        _INNER_STUDY.reset(token)


def table(rows: Sequence[Sequence], headers: Sequence[str], indent: int = 2) -> str:
    """Rows as aligned columns under the given headers.  Numbers are right
    aligned, everything else left aligned."""
    text_rows = [[_text(cell) for cell in row] for row in rows]
    widths = [len(h) for h in headers]
    for row in text_rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    numeric = [bool(rows) and all(_is_number(row[i]) for row in rows if i < len(row)) for i in range(len(headers))]
    pad = " " * indent

    def line(cells):
        parts = []
        for i, cell in enumerate(cells):
            parts.append(cell.rjust(widths[i]) if numeric[i] else cell.ljust(widths[i]))
        return (pad + "  ".join(parts)).rstrip()

    out = [line(list(headers)), pad + "  ".join("-" * w for w in widths)]
    out.extend(line(row) for row in text_rows)
    return "\n".join(out)


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _text(value) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def header(title: str = "dualmesh") -> str:
    """The version, the build options and the machine."""
    from . import __version__

    build = _core.build_information()
    features = [f"OpenMP {'on' if build['openmp'] else 'off'}", f"MPI {'on' if build['mpi'] else 'off'}", f"PETSc {build['petsc_version'] if build['petsc'] else 'off'}", f"automatic differentiation budget {build['max_ad_derivatives']} slots"]
    lines = [RULE, f"{title}  (dualmesh {__version__})", f"build:   {', '.join(features)}", f"python:  {platform.python_version()} on {platform.system()} {platform.machine()}", f"host:    {socket.gethostname()}", f"started: {datetime.datetime.now().isoformat(timespec='seconds')}", RULE]
    return "\n".join(lines)


def problem_report(problem, parameters: bool = True) -> str:
    """The problem, its mesh, variables and objects.  With ``parameters`` every
    parameter of every object is listed with its value, marked ``(default)``
    when the user did not set it."""
    core = problem._problem
    mesh = problem.mesh
    lines = ["Problem", f"  method:       {problem.method}", f"  coordinates:  {problem.coordinates}", f"  unknowns:     {problem.num_active_dofs()} carrying an equation", "Mesh", "  " + mesh.summary().strip().replace("\n", "\n  "), "Variables"]
    rows = []
    for name in core.variable_names():
        rows.append([name, problem.variable_order(name)])
    lines.append(table(rows, ["name", "order"], indent=2))
    lines.append("Objects")
    for name in core.object_names():
        obj = core.object(name)
        lines.append(f"  {name}  ({obj.type})")
        if not parameters:
            continue
        rows = []
        for row in obj.parameter_rows():
            value = row["value"] if row["value"] != "" else "(none)"
            rows.append([row["name"], value, "(default)" if row["is_default"] else "given"])
        if rows:
            lines.append(table(rows, ["parameter", "value", "source"], indent=6))
    return "\n".join(lines)


def solve_report(result, wall_time: float | None = None) -> str:
    """The outcome of a solve, in a few aligned lines."""
    rows = [["converged", bool(result.converged)], ["nonlinear iterations", int(result.total_iterations)], ["linear iterations", int(result.linear_iterations)]]
    if result.time_steps:
        rows.append(["time steps accepted", int(result.time_steps)])
        rows.append(["time steps rejected", int(result.rejected_steps)])
    if result.history:
        rows.append(["final residual", float(result.history[-1].residual_norm)])
    if wall_time is not None:
        rows.append(["wall time (s)", float(wall_time)])
    return "Solve\n" + table(rows, ["quantity", "value"], indent=2)
