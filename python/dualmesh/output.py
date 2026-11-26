# SPDX-License-Identifier: LGPL-2.1-or-later
"""The files that a study writes.

Every study that calculates fields takes one :class:`Output` group, which
tells the study where to write the files, at which times, which fields and
in which formats::

    output = dualmesh.Output(
        directory="results",
        file_base="plate",
        interval=10.0,
        fields=["temperature"],
        formats=["vtu", "csv"],
    )
    problem.solve_transient(end_time=100.0, time_step=1.0, output=output)

A study without an ``Output`` group writes no file.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy

from ._core import InputError
from .parameters import keyword_checked, parameter

__all__ = ["OUTPUT_FORMATS", "Output"]

#: The file formats of the fields: VTK unstructured grid (ParaView and
#: VisIt) and comma-separated values (a spreadsheet, pandas, gnuplot).
OUTPUT_FORMATS = ("vtu", "csv")


@keyword_checked
@dataclass
class Output:
    """The files that a study writes: the directory, the name, the times, the
    fields and the formats."""

    directory: str = parameter(
        ".",
        description="Directory of the files. The study makes the directory when it does not exist. Default '.', the working directory.",
    )
    file_base: str = parameter(
        "output", description="Start of the name of every file. Default output."
    )
    times: Sequence[float] | None = parameter(
        None,
        unit="s",
        description="Times at which a transient study writes the fields. The time steps land on these times. Give at most one of times, interval and step_interval. Default None: the times of interval or step_interval, or the start and the end of the study when neither is given.",
    )
    interval: float | None = parameter(
        None,
        unit="s",
        description="Time between two outputs of a transient study, from its start time. The study also writes the fields at its end time. Give at most one of times, interval and step_interval. Default None: the times of times or step_interval.",
    )
    step_interval: int | None = parameter(
        None,
        description="A transient study writes the fields at its start, every this many accepted steps and after its last step. The steps keep the sizes that the stepper chooses. Give at most one of times, interval and step_interval. Default None: the times of times or interval.",
    )
    fields: Sequence[str] | None = parameter(
        None, description="Variables that the files contain. Default None: all the variables."
    )
    checkpoint_interval: int | None = parameter(
        None,
        description="A transient study writes a checkpoint every this many accepted steps, and after its last step, to the file of the name with the extension .chk (one file, replaced each time), from which solve_transient(restart=...) continues the run on any number of processes. Default None: no checkpoint.",
    )
    formats: Sequence[str] = parameter(
        ("vtu",),
        description="File formats: vtu (a VTK unstructured grid, for ParaView and VisIt) and csv (one row for each node or cell, with the units in the header). Default vtu, which keeps the mesh with the fields.",
    )

    def __post_init__(self):
        given = [
            name
            for name in ("times", "interval", "step_interval")
            if getattr(self, name) is not None
        ]
        if len(given) > 1:
            raise InputError(
                f"Output: give at most one of times, interval and step_interval, which all set the output times, not {' and '.join(given)}."
            )
        if self.interval is not None and not self.interval > 0.0:
            raise InputError(f"Output: interval must be positive, not {self.interval}.")
        for name in ("step_interval", "checkpoint_interval"):
            steps = getattr(self, name)
            if steps is not None and (int(steps) != steps or steps < 1):
                raise InputError(
                    f"Output: {name} is a number of steps, a positive integer, not {steps}."
                )
        if self.times is not None:
            times = numpy.asarray(self.times, dtype=float).ravel()
            if times.size == 0 or numpy.any(numpy.diff(times) <= 0.0):
                raise InputError("Output: times must be a list of increasing times.")
        formats = [self.formats] if isinstance(self.formats, str) else list(self.formats)
        for name in formats:
            if name not in OUTPUT_FORMATS:
                raise InputError(
                    f"Output: unknown format '{name}'. The formats are: {', '.join(OUTPUT_FORMATS)}."
                )
        if not formats:
            raise InputError("Output: give at least one format.")
        self.formats = tuple(formats)
        if self.fields is not None:
            self.fields = [self.fields] if isinstance(self.fields, str) else list(self.fields)

    # ---- used by the studies -------------------------------------------------
    def output_times(self, start: float, end: float) -> numpy.ndarray:
        """The times at which a transient study from ``start`` to ``end``
        writes the fields, which the steps land on.  None with step_interval, whose times the
        steps choose."""
        if self.step_interval is not None:
            return None
        if self.times is not None:
            times = numpy.asarray(self.times, dtype=float).ravel()
            tolerance = 1e-10 * max(abs(end - start), 1e-300)
            if times[0] < start - tolerance or times[-1] > end + tolerance:
                raise InputError(
                    f"Output: the times must be in the interval of the study, from {start} s to {end} s."
                )
            return times
        if self.interval is not None:
            count = int(numpy.floor((end - start) / self.interval * (1.0 + 1e-12)))
            times = start + self.interval * numpy.arange(count + 1)
            if end - times[-1] > 1e-8 * abs(end - start):
                times = numpy.append(times, end)
            return times
        return numpy.array([start, end])

    def path(self, suffix: str) -> str:
        """The path of a file: the directory, the file base and ``suffix``.
        The directory is made when it does not exist."""
        Path(self.directory).mkdir(parents=True, exist_ok=True)
        return os.path.join(self.directory, self.file_base + suffix)


class Series:
    """The numbered files of the fields of a transient run, with their ParaView collection.

    A loop that runs on from one mesh to the next passes one series to the run on each mesh, so
    the files keep one numbering and one collection. Each of ``actions`` is called as
    ``action(problem, time)`` after the fields are written, for the files that the output times
    of the run also write."""

    def __init__(self, output: Output, actions=()):
        self.output = output
        self.actions = list(actions)
        #: The time of each file, by its number.
        self.times: dict[int, float] = {}
        #: The accepted steps of the runs so far.
        self.steps = 0

    def write(self, problem, time: float, index: int | None = None) -> list[str]:
        """Writes the fields at ``time`` as the file of number ``index``, by default the next."""
        index = max(self.times, default=-1) + 1 if index is None else index
        self.times[index] = float(time)
        files = problem._write_fields(self.output, index)
        for action in self.actions:
            files += action(problem, time)
        return files

    def write_collection(self, extension: str) -> str:
        """Writes the ParaView collection (``.pvd``) of the files, which ParaView opens as one
        time series.  The first process writes it."""
        from . import _core

        lines = [
            '<?xml version="1.0"?>',
            '<VTKFile type="Collection" version="0.1">',
            "<Collection>",
        ]
        for index, time in sorted(self.times.items()):
            name = f"{self.output.file_base}_{index:05d}.{extension}"
            lines.append(f'<DataSet timestep="{time:.12g}" file="{name}"/>')
        lines += ["</Collection>", "</VTKFile>", ""]
        path = self.output.path(".pvd")
        if _core.mpi_rank() == 0:
            Path(path).write_text("\n".join(lines))
        return path
