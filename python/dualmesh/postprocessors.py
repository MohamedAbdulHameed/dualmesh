# SPDX-License-Identifier: LGPL-2.1-or-later
"""Post-processors: scalar quantities computed from a solution.

A post-processor reduces a field to one number: an integral, an average, a
point value, an extreme value, a reaction or a boundary flux.  They are added
to a problem by type name, like the other objects, and evaluated after a
steady solve and after every accepted time step of a transient one::

    problem.add_postprocessor("variable_average", "mean_temperature", variable="temperature")
    problem.add_postprocessor("total_reaction", "heat_out", variable="temperature", boundary="left")
    problem.solve()
    problem.postprocessor_table()  # the history, as aligned text
    problem.write_postprocessor_csv("out.csv")

The names follow the MOOSE post-processors they correspond to where one
exists (``point_value``, ``nodal_extreme_value``); the others say what they
compute.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass

import numpy as np


@dataclass
class Postprocessor:
    """Base class: a named scalar computed from a problem."""

    name: str

    def compute(self, problem) -> float:  # pragma: no cover - abstract
        raise NotImplementedError


@dataclass
class VariableIntegral(Postprocessor):
    """The integral of a variable over the domain, with the coordinate factor
    of the problem (so an axisymmetric integral is over the body of
    revolution)."""

    variable: str = ""

    def compute(self, problem) -> float:
        return float(problem.integrate(self.variable))


@dataclass
class VariableAverage(Postprocessor):
    """The average of a variable over the domain: its integral divided by the
    volume, both with the coordinate factor of the problem."""

    variable: str = ""

    def compute(self, problem) -> float:
        return float(problem.integrate(self.variable) / problem.domain_volume())


@dataclass
class PointValue(Postprocessor):
    """The value of a variable at a point, interpolated in the element that
    contains it (NaN outside the mesh)."""

    variable: str = ""
    point: tuple = (0.0, 0.0, 0.0)

    def compute(self, problem) -> float:
        point = list(self.point) + [0.0] * (3 - len(self.point))
        return float(problem.sample(self.variable, [point])[0])


@dataclass
class NodalExtremeValue(Postprocessor):
    """The largest (``value_type="max"``, the default) or smallest
    (``"min"``) nodal value of a variable."""

    variable: str = ""
    value_type: str = "max"

    def __post_init__(self):
        if self.value_type not in ("max", "min"):
            raise ValueError(f"Post-processor '{self.name}': value_type must be 'max' or 'min', not '{self.value_type}'.")

    def compute(self, problem) -> float:
        values = np.asarray(problem.values(self.variable))
        return float(values.max() if self.value_type == "max" else values.min())


@dataclass
class TotalReaction(Postprocessor):
    """The total reaction (secondary variable) of a variable on a boundary:
    the integrated normal flux the essential condition there has to supply,
    for instance the heat leaving through a boundary held at a fixed
    temperature."""

    variable: str = ""
    boundary: str = ""

    def compute(self, problem) -> float:
        return float(problem.total_reaction(self.variable, self.boundary))


@dataclass
class BoundaryFluxIntegral(Postprocessor):
    """The integral of the normal flux of one kernel over a side set."""

    kernel: str = ""
    boundary: str = ""

    def compute(self, problem) -> float:
        return float(problem.boundary_flux_integral(self.kernel, self.boundary))


TYPES = {"variable_integral": VariableIntegral, "variable_average": VariableAverage, "point_value": PointValue, "nodal_extreme_value": NodalExtremeValue, "total_reaction": TotalReaction, "boundary_flux_integral": BoundaryFluxIntegral}
"""Post-processor types by name.  The names are lower case with underscores,
like every object type."""


def _name_key(name: str) -> str:
    return name.replace("_", "").lower()


def create(type_name: str, name: str, **parameters) -> Postprocessor:
    """A post-processor of the registered type ``type_name``."""
    if type_name not in TYPES:
        renamed = [t for t in TYPES if _name_key(t) == _name_key(type_name)]
        if _name_key(type_name) == "reaction":
            renamed = ["total_reaction"]
        hint = f" Object types are written in lower case with underscores since dualmesh 0.2: use '{renamed[0]}'." if renamed else ""
        raise ValueError(f"Unknown post-processor type '{type_name}'.{hint} Use one of: {', '.join(sorted(TYPES))}.")
    cls = TYPES[type_name]
    allowed = {f for f in cls.__dataclass_fields__ if f != "name"}
    unknown = set(parameters) - allowed
    if unknown:
        raise ValueError(f"Post-processor '{name}' ({type_name}): unknown parameter(s) {', '.join(sorted(unknown))}. It takes: {', '.join(sorted(allowed))}.")
    if "point" in parameters:
        parameters["point"] = tuple(float(c) for c in parameters["point"])
    return cls(name=name, **parameters)


class PostprocessorHistory:
    """The values of a problem's post-processors at every evaluation: a
    column of times and one column per post-processor."""

    def __init__(self):
        self.postprocessors: list[Postprocessor] = []
        self.time: list[float] = []
        self.values: dict[str, list[float]] = {}

    def add(self, postprocessor: Postprocessor) -> None:
        if postprocessor.name in self.values:
            raise ValueError(f"A post-processor named '{postprocessor.name}' already exists.")
        self.postprocessors.append(postprocessor)
        self.values[postprocessor.name] = [float("nan")] * len(self.time)

    def evaluate(self, problem, time: float) -> dict:
        """Compute every post-processor and append the values."""
        self.time.append(float(time))
        current = {}
        for pp in self.postprocessors:
            value = pp.compute(problem)
            self.values[pp.name].append(value)
            current[pp.name] = value
        return current

    def as_arrays(self) -> dict:
        out = {"time": np.asarray(self.time)}
        out.update({name: np.asarray(v) for name, v in self.values.items()})
        return out

    def table(self) -> str:
        from .console import table

        names = [pp.name for pp in self.postprocessors]
        rows = [[t] + [self.values[n][i] for n in names] for i, t in enumerate(self.time)]
        return table(rows, ["time"] + names)

    def write_csv(self, filename: str) -> None:
        """One row per evaluation, headed by the column names and, on a second
        comment line, the post-processor types."""
        names = [pp.name for pp in self.postprocessors]
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f, lineterminator="\n")
            writer.writerow(["time"] + names)
            for i, t in enumerate(self.time):
                writer.writerow([repr(t)] + [repr(self.values[n][i]) for n in names])
