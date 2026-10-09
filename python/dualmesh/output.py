# SPDX-License-Identifier: LGPL-2.1-or-later
"""The files that a study writes.

Every study that calculates fields takes one :class:`Output` group, which
tells the study where to write the files, at which times, which fields and
in which formats::

    output = dm.Output(
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

import numpy as np

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
        description="Times at which a transient study writes the fields. The time steps land on these times. Give only one of times and interval. Default None: the times of interval, or the start and the end of the study when interval is not given.",
    )
    interval: float | None = parameter(
        None,
        unit="s",
        description="Time between two outputs of a transient study, from its start time. The study also writes the fields at its end time. Give only one of times and interval. Default None: the times of times.",
    )
    fields: Sequence[str] | None = parameter(
        None, description="Variables that the files contain. Default None: all the variables."
    )
    formats: Sequence[str] = parameter(
        ("vtu",),
        description="File formats: vtu (a VTK unstructured grid, for ParaView and VisIt) and csv (one row for each node or cell, with the units in the header). Default vtu, which keeps the mesh with the fields.",
    )

    def __post_init__(self):
        if self.times is not None and self.interval is not None:
            raise InputError(
                "Output: give only one of times and interval. The two set the output times."
            )
        if self.interval is not None and not self.interval > 0.0:
            raise InputError(f"Output: interval must be positive, not {self.interval}.")
        if self.times is not None:
            times = np.asarray(self.times, dtype=float).ravel()
            if times.size == 0 or np.any(np.diff(times) <= 0.0):
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
    def output_times(self, start: float, end: float) -> np.ndarray:
        """The times at which a transient study from ``start`` to ``end``
        writes the fields."""
        if self.times is not None:
            times = np.asarray(self.times, dtype=float).ravel()
            tolerance = 1e-10 * max(abs(end - start), 1e-300)
            if times[0] < start - tolerance or times[-1] > end + tolerance:
                raise InputError(
                    f"Output: the times must be in the interval of the study, from {start} s to {end} s."
                )
            return times
        if self.interval is not None:
            count = int(np.floor((end - start) / self.interval * (1.0 + 1e-12)))
            times = start + self.interval * np.arange(count + 1)
            if end - times[-1] > 1e-8 * abs(end - start):
                times = np.append(times, end)
            return times
        return np.array([start, end])

    def path(self, suffix: str) -> str:
        """The path of a file: the directory, the file base and ``suffix``.
        The directory is made when it does not exist."""
        Path(self.directory).mkdir(parents=True, exist_ok=True)
        return os.path.join(self.directory, self.file_base + suffix)

    def write_collection(self, times: Sequence[float], extension: str) -> str:
        """Write the ParaView collection (``.pvd``) of the files of a
        transient study, which ParaView opens as one time series."""
        lines = [
            '<?xml version="1.0"?>',
            '<VTKFile type="Collection" version="0.1">',
            "<Collection>",
        ]
        for index, time in enumerate(times):
            lines.append(
                f'<DataSet timestep="{float(time):.12g}" file="{self.file_base}_{index:05d}.{extension}"/>'
            )
        lines += ["</Collection>", "</VTKFile>", ""]
        path = self.path(".pvd")
        Path(path).write_text("\n".join(lines))
        return path
