# SPDX-License-Identifier: LGPL-2.1-or-later
"""Small helpers for extracting and comparing results."""

from __future__ import annotations

from collections.abc import Sequence

import numpy


def sample_line(problem, variable: str, start, end, num_points: int = 21):
    """Sample a variable along a straight line; returns ``(arc_length, values)``."""
    start = numpy.asarray(list(start) + [0.0] * (3 - len(start)), dtype=float)
    end = numpy.asarray(list(end) + [0.0] * (3 - len(end)), dtype=float)
    fractions = numpy.linspace(0.0, 1.0, num_points)
    points = start + numpy.outer(fractions, end - start)
    values = problem.sample(variable, points)
    return fractions * numpy.linalg.norm(end - start), values


def values_on_line(problem, variable: str, coordinates, axis: int = 0, other=0.0):
    """Sample a variable at given coordinates along one axis."""
    points = numpy.zeros((len(coordinates), 3))
    points[:, axis] = numpy.asarray(coordinates, dtype=float)
    for d in range(3):
        if d != axis:
            points[:, d] = other if numpy.isscalar(other) else other[d]
    return problem.sample(variable, points)


def relative_error(computed, reference) -> numpy.ndarray:
    """Element-wise relative error, using the reference magnitude as the scale."""
    computed = numpy.asarray(computed, dtype=float)
    reference = numpy.asarray(reference, dtype=float)
    scale = numpy.maximum(numpy.abs(reference), numpy.finfo(float).tiny)
    return numpy.abs(computed - reference) / scale


def max_relative_error(computed, reference) -> float:
    return float(numpy.max(relative_error(computed, reference)))


def convergence_rates(mesh_sizes: Sequence[float], errors: Sequence[float]) -> numpy.ndarray:
    """Observed convergence rates between successive refinements."""
    h = numpy.asarray(mesh_sizes, dtype=float)
    e = numpy.asarray(errors, dtype=float)
    return numpy.log(e[:-1] / e[1:]) / numpy.log(h[:-1] / h[1:])


def comparison_table(
    labels: Sequence[str], computed: Sequence[float], reference: Sequence[float], title: str = ""
) -> str:
    """A fixed-width table of computed values against reference values."""
    lines = []
    if title:
        lines.append(title)
    lines.append(f"{'':>14}{'computed':>16}{'reference':>16}{'rel. error':>14}")
    for label, c, r in zip(labels, computed, reference):
        error = abs(c - r) / max(abs(r), 1e-300)
        lines.append(f"{label:>14}{c:16.6g}{r:16.6g}{error:14.3e}")
    return "\n".join(lines)
