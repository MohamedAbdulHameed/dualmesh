# SPDX-License-Identifier: LGPL-2.1-or-later
"""The :class:`Problem` class: variables, objects, solvers, and results."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

from . import _core

SolveResult = _core.SolveResult


def _solver_options(**kwargs) -> _core.SolverOptions:
    options = _core.SolverOptions()
    known = {
        "nonlinear_solver",
        "max_iterations",
        "relative_tolerance",
        "absolute_tolerance",
        "step_tolerance",
        "relaxation",
        "load_factors",
        "linear_solver",
        "linear_tolerance",
        "linear_max_iterations",
        "verbose",
        "error_on_divergence",
    }
    for key, value in kwargs.items():
        if key not in known:
            raise TypeError(
                f"Unknown solver option '{key}'. Known options: {', '.join(sorted(known))}."
            )
        setattr(options, key, value)
    return options


class Problem:
    """A boundary value problem discretized on a primal mesh and its dual mesh.

    Parameters
    ----------
    mesh:
        The primal mesh of finite elements.
    method:
        ``"dmcdm"`` for the dual mesh control domain method (the default) or
        ``"fem"`` for the Galerkin finite element method.  Everything else in
        the problem definition is identical, which makes the two directly
        comparable.
    coordinates:
        ``"cartesian"``, ``"axisymmetric"`` (the integrals carry the factor
        :math:`2 \\pi r`, with :math:`r` the first coordinate), or
        ``"spherical"`` (factor :math:`4 \\pi r^2`, one-dimensional meshes).
    """

    def __init__(self, mesh, method: str = "dmcdm", coordinates: str = "cartesian"):
        self._problem = _core.Problem(mesh, method, coordinates)
        self._mesh = mesh
        self._objects: list = []  # keeps Python-defined objects alive
        self.method = method
        self.coordinates = coordinates

    # ---- definition ------------------------------------------------------
    def add_variable(self, name: str, blocks: Sequence[str] = (), initial_condition=None) -> int:
        """Add a nodal unknown; returns its index."""
        return self._problem.add_variable(name, list(blocks), initial_condition)

    def add_function(self, name: str, function) -> None:
        """Register a named function of ``(x, y, z, t)`` (or a constant)."""
        self._problem.add_function(name, function)

    def _add(self, adder, object_or_type, name, kwargs):
        if isinstance(object_or_type, str):
            return self._problem.add_object(object_or_type, name or "", **kwargs)
        if kwargs:
            raise TypeError("Extra parameters are not accepted when adding an object instance.")
        self._objects.append(object_or_type)
        adder(object_or_type)
        return object_or_type

    def add_kernel(self, kernel, name: str | None = None, **parameters):
        """Add a kernel by registered type name, or an object built in Python."""
        return self._add(self._problem.add_kernel, kernel, name, parameters)

    def add_boundary_condition(self, condition, name: str | None = None, **parameters):
        """Add a boundary condition (essential or natural)."""
        if isinstance(condition, str):
            category = _core.object_category(condition)
            if category == "NodalBC":
                return self._problem.add_object(condition, name or "", **parameters)
            if category == "BoundaryCondition":
                return self._problem.add_object(condition, name or "", **parameters)
            raise ValueError(f"'{condition}' is a {category}, not a boundary condition.")
        if isinstance(condition, _core.NodalBC):
            return self._add(self._problem.add_nodal_bc, condition, name, parameters)
        return self._add(self._problem.add_integrated_bc, condition, name, parameters)

    def add_material(self, material, name: str | None = None, **parameters):
        """Add a material (a provider of named properties)."""
        return self._add(self._problem.add_material, material, name, parameters)

    def add_point_source(self, source="PointSource", name: str | None = None, **parameters):
        """Add a concentrated nodal source (point force or point heat source)."""
        return self._add(self._problem.add_nodal_load, source, name, parameters)

    add_nodal_load = add_point_source

    def initialize(self) -> None:
        """Resolve all objects (called automatically by :meth:`solve`)."""
        self._problem.initialize()

    # ---- solving ---------------------------------------------------------
    def solve(self, **options) -> SolveResult:
        """Solve the steady problem.

        Keyword arguments are solver options: ``nonlinear_solver`` (``"newton"``,
        ``"picard"``, or ``"linear"``), ``max_iterations``, ``relative_tolerance``,
        ``absolute_tolerance``, ``step_tolerance``, ``relaxation`` (the
        acceleration parameter of direct iteration), ``load_factors`` (load
        stepping), ``linear_solver`` (``"lu"``, ``"bicgstab"``, ``"cg"``),
        ``verbose``, and ``error_on_divergence``.
        """
        return self._problem.solve_steady(_solver_options(**options))

    def solve_transient(
        self,
        end_time: float,
        dt: float,
        start_time: float = 0.0,
        theta: float = 1.0,
        output_interval: int = 0,
        output_file_base: str = "",
        **options,
    ) -> SolveResult:
        """March in time with the theta method (1 = backward Euler, 0.5 = Crank-Nicolson)."""
        transient = _core.TransientOptions()
        transient.start_time = start_time
        transient.end_time = end_time
        transient.dt = dt
        transient.theta = theta
        transient.output_interval = output_interval
        transient.output_file_base = output_file_base
        return self._problem.solve_transient(transient, _solver_options(**options))

    def set_time_step_callback(self, callback) -> None:
        """Call ``callback(time, problem)`` after every converged time step."""
        self._problem.set_time_step_callback(callback)

    # ---- results ---------------------------------------------------------
    @property
    def mesh(self):
        return self._mesh

    @property
    def time(self) -> float:
        return self._problem.time

    @time.setter
    def time(self, value: float) -> None:
        self._problem.time = value

    def variable_index(self, name: str) -> int:
        return self._problem.variable_index(name)

    def values(self, variable: str) -> np.ndarray:
        """Nodal values of a variable, indexed by node."""
        return self._problem.values(variable)

    def set_values(self, variable: str, values) -> None:
        self._problem.set_values(variable, list(np.asarray(values, dtype=float).ravel()))

    def solution(self) -> np.ndarray:
        """The full solution vector (node-major, variable-minor)."""
        return self._problem.solution()

    def apply_initial_conditions(self) -> None:
        self._problem.apply_initial_conditions()

    def reactions(self, variable: str, boundary: str):
        """Secondary variables at the nodes of a boundary: ``[(node, value), ...]``.

        For each node the value is the integral of the normal flux over the part
        of the boundary that belongs to that node's control domain, which is the
        quantity Reddy denotes :math:`Q_I` (a reaction, a heat flow, a force).

        A node that lies on two boundaries (a corner) carries one reaction that
        covers its whole boundary portion, so it appears in both lists.  When
        summing over several boundaries, collect the nodes first
        (``dict(problem.reactions(...))``) instead of adding the totals, or
        corner nodes are counted twice.
        """
        return self._problem.reactions(variable, boundary)

    def total_reaction(self, variable: str, boundary: str) -> float:
        """Sum of the secondary variables over a boundary."""
        return self._problem.total_reaction(variable, boundary)

    def sample(self, variable: str, points) -> np.ndarray:
        """Interpolate a variable at arbitrary points (NaN outside the mesh)."""
        points = np.atleast_2d(np.asarray(points, dtype=float))
        if points.shape[1] < 3:
            points = np.hstack([points, np.zeros((points.shape[0], 3 - points.shape[1]))])
        return np.asarray(self._problem.sample(variable, [list(p) for p in points]))

    def values_at_nodes(self, variable: str, nodes: Iterable[int]) -> np.ndarray:
        values = self.values(variable)
        return np.asarray([values[int(n)] for n in nodes])

    def nodes_where(self, predicate) -> list[int]:
        """Node indices whose coordinates satisfy ``predicate(x, y, z)``."""
        points = self._mesh.points()
        return [i for i, p in enumerate(points) if predicate(p[0], p[1], p[2])]

    def node_at(self, point, tolerance: float = 1e-9) -> int:
        """Index of the node closest to ``point`` (an error if none is within ``tolerance``)."""
        point = np.asarray(list(point) + [0.0] * (3 - len(point)), dtype=float)
        points = np.asarray(self._mesh.points())
        distances = np.linalg.norm(points - point, axis=1)
        index = int(np.argmin(distances))
        if distances[index] > tolerance:
            raise ValueError(
                f"No node within {tolerance} of {point.tolist()}; nearest is at "
                f"{points[index].tolist()} (distance {distances[index]:.3e})."
            )
        return index

    def gradient_at_centroids(self, variable: str) -> np.ndarray:
        return np.asarray(self._problem.gradient_at_centroids(variable))

    def property_at_centroids(self, property_name: str) -> np.ndarray:
        return np.asarray(self._problem.property_at_centroids(property_name))

    def kernel_flux_at_centroids(self, kernel_name: str) -> np.ndarray:
        return np.asarray(self._problem.kernel_flux_at_centroids(kernel_name))

    def integrate(self, variable: str) -> float:
        return self._problem.integrate(variable)

    def boundary_flux_integral(self, kernel_name: str, boundary: str) -> float:
        return self._problem.boundary_flux_integral(kernel_name, boundary)

    # ---- output ----------------------------------------------------------
    def write_vtu(self, filename: str, cell_properties: Sequence[str] = ()) -> None:
        """Write a VTK unstructured grid (readable by ParaView and VisIt)."""
        self._problem.write_vtu(filename, list(cell_properties))

    def write_mesh_file(self, filename: str, file_format: str | None = None) -> None:
        """Write the mesh and all nodal fields through meshio (Exodus, VTU, ...)."""
        from .meshing import write_mesh

        fields = {}
        for index in range(self._problem.num_variables):
            name = self._problem.variable_name(index)
            fields[name] = self.values(name)
        write_mesh(self._mesh, filename, file_format=file_format, **fields)

    def write_csv(self, filename: str, variables: Sequence[str] = ()) -> None:
        """Write nodal coordinates and values as comma-separated values."""
        names = list(variables) or [
            self._problem.variable_name(i) for i in range(self._problem.num_variables)
        ]
        points = np.asarray(self._mesh.points())
        columns = [points[:, i] for i in range(self._mesh.dimension)]
        columns += [self.values(name) for name in names]
        header = ",".join(["x", "y", "z"][: self._mesh.dimension] + names)
        np.savetxt(filename, np.column_stack(columns), delimiter=",", header=header, comments="")

    def summary(self) -> str:
        return self._problem.summary()

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<dualmesh.Problem method={self.method} {self._mesh.num_nodes} nodes>"
