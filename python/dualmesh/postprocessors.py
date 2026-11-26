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
    print(problem.postprocessor_table())  # the history, a table that prints aligned
    problem.write_postprocessor_csv("out.csv")

Every name says what the post-processor computes (``point_value``,
``maximum``).  A post-processor of the flow through a boundary takes its
physics as ``physics``, which may be left out when the problem has one
physics of that type.  Every post-processor takes a ``unit``, which heads its
column of the history.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy


@dataclass
class Postprocessor:
    """Base class: a named scalar computed from a problem, with the unit of its value, which
    heads its column of the history (empty when it is not given)."""

    name: str
    unit: str = ""

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


def _cell_values(problem, property_name: str):
    """The average of a scalar property over the points of each cell of this process, and the
    measure of each cell with the coordinate factor of the problem.  A property with a history,
    such as a state of a model, takes its value at the last accepted step."""
    values = numpy.asarray(problem.property_at_centroids(property_name), dtype=float)
    values = values[:, 0] if values.size else numpy.zeros(0)
    return values, numpy.asarray(problem.element_volumes())[: values.size]


@dataclass
class PropertyIntegral(Postprocessor):
    """The integral of a scalar property over the domain: the sum over the cells of the average
    of the property at the points of each cell times the measure of the cell."""

    property: str = ""

    def compute(self, problem) -> float:
        values, volumes = _cell_values(problem, self.property)
        return problem._sum(float(values @ volumes))


@dataclass
class PropertyAverage(Postprocessor):
    """The average of a scalar property over the domain: its integral (PropertyIntegral) divided
    by the measure of the domain."""

    property: str = ""

    def compute(self, problem) -> float:
        values, volumes = _cell_values(problem, self.property)
        return problem._sum(float(values @ volumes)) / problem._sum(float(volumes.sum()))


@dataclass
class PropertyMaximum(Postprocessor):
    """The largest average of a scalar property over the points of a cell."""

    property: str = ""

    def compute(self, problem) -> float:
        from . import _core

        values, _ = _cell_values(problem, self.property)
        local = float(values.max()) if values.size else -numpy.inf
        return _core.mpi_max(local) if problem._distributed is not None else local


@dataclass
class PropertyMinimum(Postprocessor):
    """The smallest average of a scalar property over the points of a cell."""

    property: str = ""

    def compute(self, problem) -> float:
        from . import _core

        values, _ = _cell_values(problem, self.property)
        local = float(values.min()) if values.size else numpy.inf
        return _core.mpi_min(local) if problem._distributed is not None else local


@dataclass
class PropertyPointValue(Postprocessor):
    """One component of a property at a point, evaluated in the element that holds the point (NaN
    outside the mesh), such as the component ``component`` of the stress.  A property with a
    history takes the average history of that element at the last accepted step."""

    property: str = ""
    component: int = 0
    point: tuple = (0.0, 0.0, 0.0)

    def compute(self, problem) -> float:
        return float(problem.property_at_points(self.property, [self.point])[0, self.component])


def von_Mises_stress(stress) -> numpy.ndarray:
    """The von Mises equivalent stress of rows of the stress in the Voigt order of the engine:
    xx, yy, zz, yz, xz and xy, or rr, zz, the hoop component and rz in its last place in
    axisymmetric coordinates."""
    s = numpy.atleast_2d(numpy.asarray(stress, dtype=float))
    normal = (s[:, 0] - s[:, 1]) ** 2 + (s[:, 1] - s[:, 2]) ** 2 + (s[:, 2] - s[:, 0]) ** 2
    return numpy.sqrt(0.5 * normal + 3.0 * (s[:, 3] ** 2 + s[:, 4] ** 2 + s[:, 5] ** 2))


@dataclass
class VonMisesStressPointValue(Postprocessor):
    """The von Mises stress of a solid at a point, from its stress in the element that holds the
    point (NaN outside the mesh)."""

    point: tuple = (0.0, 0.0, 0.0)

    def compute(self, problem) -> float:
        return float(von_Mises_stress(problem.property_at_points("stress", [self.point]))[0])


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
    :meth:`~dualmesh.physics.HeatConduction.heat_flow`."""

    boundary: str = ""
    physics: object = None

    def compute(self, problem) -> float:
        heat = _physics_of(problem, self.physics, "heat_conduction", self.name)
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
    "property_integral": PropertyIntegral,
    "property_average": PropertyAverage,
    "property_maximum": PropertyMaximum,
    "property_minimum": PropertyMinimum,
    "property_point_value": PropertyPointValue,
    "von_mises_stress_point_value": VonMisesStressPointValue,
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
        from .console import suggestion

        raise ValueError(
            f"Unknown post-processor type '{type_name}'." + suggestion(type_name, TYPES)
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

    def as_table(self):
        """The history as a :class:`~dualmesh.Table`: the time in s, then one column for each
        post-processor, with its unit."""
        from .tables import Table

        names = [pp.name for pp in self.postprocessors]
        rows = [[t] + [self.values[n][i] for n in names] for i, t in enumerate(self.time)]
        return Table(["time", *names], ["s", *(pp.unit for pp in self.postprocessors)], rows)

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
