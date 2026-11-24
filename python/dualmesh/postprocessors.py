# SPDX-License-Identifier: LGPL-2.1-or-later
"""Post-processors: scalar quantities computed from a solution.

A post-processor reduces a field to one number: an integral, an average, a
point value, a maximum or a minimum, or the flow through a boundary (a heat
flow, a force, a flow rate).  They are added to a problem by type name, like
the other objects, and evaluated after a steady solve and after every
accepted time step of a transient one::

    problem.add_postprocessor("average", "mean_temperature", variable="temperature")
    problem.add_postprocessor("heat_flow", "heat_out", boundary="left")
    problem.solve()
    problem.postprocessor_table()  # the history, as aligned text
    problem.write_postprocessor_csv("out.csv")

Every name says what the post-processor computes (``point_value``,
``maximum``).  A post-processor of the flow through a boundary takes its
physics as ``physics``, which may be left out when the problem has one
physics of that type.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass

import numpy


@dataclass
class Postprocessor:
    """Base class: a named scalar computed from a problem."""

    name: str

    def compute(self, problem) -> float:  # pragma: no cover - abstract
        raise NotImplementedError


@dataclass
class Integral(Postprocessor):
    """The integral of a variable over the domain, with the coordinate factor
    of the problem (so an axisymmetric integral is over the body of
    revolution)."""

    variable: str = ""

    def compute(self, problem) -> float:
        return float(problem.integrate(self.variable))


@dataclass
class Average(Postprocessor):
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
class Maximum(Postprocessor):
    """The largest nodal value of a variable."""

    variable: str = ""

    def compute(self, problem) -> float:
        return float(problem.maximum(self.variable))


@dataclass
class Minimum(Postprocessor):
    """The smallest nodal value of a variable."""

    variable: str = ""

    def compute(self, problem) -> float:
        return float(problem.minimum(self.variable))


@dataclass
class BoundaryFlux(Postprocessor):
    """The flow of the flux of the equation of a variable out of the body
    through a boundary (:meth:`~dualmesh.Problem.boundary_flux`)."""

    variable: str = ""
    boundary: str = ""

    def compute(self, problem) -> float:
        return float(problem.boundary_flux(self.variable, self.boundary))


def _physics_of(problem, physics, type_name: str, where: str):
    """The physics named or given by ``physics``, or the only one of its type."""
    if physics is not None and not isinstance(physics, str):
        return physics
    found = [p for p in problem._physics.values() if p.type_name == type_name]
    if physics is not None:
        found = [p for p in found if p.name == physics]
    if len(found) != 1:
        names = ", ".join(p.name for p in problem._physics.values() if p.type_name == type_name)
        raise ValueError(
            f"Post-processor '{where}': give the {type_name} physics as physics (the problem has: {names or 'none'})."
        )
    return found[0]


_DIRECTIONS = {"x": 0, "y": 1, "z": 2}


@dataclass
class HeatFlow(Postprocessor):
    """The heat leaving the body through a boundary (W):
    :meth:`~dualmesh.physics.HeatTransfer.heat_flow`."""

    boundary: str = ""
    physics: object = None

    def compute(self, problem) -> float:
        heat = _physics_of(problem, self.physics, "heat_transfer", self.name)
        return float(heat.heat_flow(self.boundary))


@dataclass
class ReactionForce(Postprocessor):
    """One component (``direction`` x, y or z) of the force of the supports
    on a solid through a boundary (N):
    :meth:`~dualmesh.physics.SolidMechanics.reaction_force`."""

    boundary: str = ""
    direction: str = "x"
    physics: object = None

    def compute(self, problem) -> float:
        solid = _physics_of(problem, self.physics, "solid_mechanics", self.name)
        return float(solid.reaction_force(self.boundary)[self.direction])


@dataclass
class Force(Postprocessor):
    """One component (``direction`` x, y or z) of the force of a fluid on a
    boundary (N): :meth:`~dualmesh.physics.FluidDynamics.force`."""

    boundary: str = ""
    direction: str = "x"
    physics: object = None

    def compute(self, problem) -> float:
        flow = _physics_of(problem, self.physics, "fluid_dynamics", self.name)
        return float(flow.force(self.boundary)[self.direction])


@dataclass
class FlowRate(Postprocessor):
    """The volume flow leaving through a boundary (m^3/s):
    :meth:`~dualmesh.physics.FluidDynamics.flow_rate`."""

    boundary: str = ""
    physics: object = None

    def compute(self, problem) -> float:
        flow = _physics_of(problem, self.physics, "fluid_dynamics", self.name)
        return float(flow.flow_rate(self.boundary))


@dataclass
class KernelBoundaryFlux(Postprocessor):
    """The integral of the normal flux of one kernel over a side set: ``kernel`` is the
    object that :meth:`~dualmesh.Problem.add_kernel` returned."""

    kernel: object = None
    boundary: str = ""

    def compute(self, problem) -> float:
        return float(problem.kernel_boundary_flux(self.kernel, self.boundary))


TYPES = {
    "integral": Integral,
    "average": Average,
    "point_value": PointValue,
    "maximum": Maximum,
    "minimum": Minimum,
    "boundary_flux": BoundaryFlux,
    "heat_flow": HeatFlow,
    "reaction_force": ReactionForce,
    "force": Force,
    "flow_rate": FlowRate,
    "kernel_boundary_flux": KernelBoundaryFlux,
}
"""Post-processor types by name.  The names are lower case with underscores,
like every object type."""


def create(type_name: str, name: str, **parameters) -> Postprocessor:
    """A post-processor of the registered type ``type_name``."""
    if type_name not in TYPES:
        import difflib

        close = difflib.get_close_matches(type_name, sorted(TYPES), n=1)
        hint = f" Did you mean '{close[0]}'?" if close else ""
        raise ValueError(
            f"Unknown post-processor type '{type_name}'.{hint} Use one of: {', '.join(sorted(TYPES))}."
        )
    cls = TYPES[type_name]
    allowed = {f for f in cls.__dataclass_fields__ if f != "name"}
    unknown = set(parameters) - allowed
    if unknown:
        raise ValueError(
            f"Post-processor '{name}' ({type_name}): unknown parameter(s) {', '.join(sorted(unknown))}. It takes: {', '.join(sorted(allowed))}."
        )
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
        out = {"time": numpy.asarray(self.time)}
        out.update({name: numpy.asarray(v) for name, v in self.values.items()})
        return out

    def table(self) -> str:
        from .console import table

        names = [pp.name for pp in self.postprocessors]
        rows = [[t] + [self.values[n][i] for n in names] for i, t in enumerate(self.time)]
        return table(rows, ["time"] + names)

    def write_json(self, filename: str) -> None:
        """The history as JSON, which load_json reads back when a run restarts from a checkpoint."""
        import json

        with open(filename, "w") as f:
            json.dump({"time": self.time, "values": self.values}, f)

    def load_json(self, filename: str) -> None:
        """Continue the history written by write_json, for the post-processors of this problem."""
        import json

        with open(filename) as f:
            saved = json.load(f)
        self.time = [float(t) for t in saved["time"]]
        for name in self.values:
            values = saved["values"].get(name)
            self.values[name] = (
                [float(v) for v in values]
                if values is not None
                else [float("nan")] * len(self.time)
            )

    def write_csv(self, filename: str) -> None:
        """One row per evaluation, headed by the column names and, on a second
        comment line, the post-processor types."""
        names = [pp.name for pp in self.postprocessors]
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f, lineterminator="\n")
            writer.writerow(["time"] + names)
            for i, t in enumerate(self.time):
                writer.writerow([repr(t)] + [repr(self.values[n][i]) for n in names])
