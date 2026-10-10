# SPDX-License-Identifier: LGPL-2.1-or-later
"""Verification by the method of manufactured solutions.

Background
----------

A code is *verified* when it is shown to solve its equations correctly, which is
a different question from whether the equations describe nature.  The standard
tool is the method of manufactured solutions (Roache, 2002): choose any smooth
function as the exact solution, substitute it into the governing equations,
and whatever is left over is the source that function needs.  Adding that
source to the problem and imposing the chosen function on the boundary gives a
problem whose exact solution is known, however nonlinear or coupled the
equations are.  Solving it on a sequence of refined meshes then measures the
order at which the error falls, and a discretisation that is correctly
implemented must show its theoretical order.  An error in the implementation
almost always shows up as a lower order, often as no convergence at all.

The weak point of the method is keeping the two descriptions of the equations
in step: the one the code solves, and the one the source is derived from.  A
hand-derived source for a nonlinear, axisymmetric, coupled problem is itself an
invitation to error.  This module removes that step.  The equations are the
physics of the physics level (:mod:`dualmesh.physics`), and each physics
states both which dualmesh objects it generates and, in SymPy, which flux and
source those objects contribute, so the source is derived from exactly the
equations the code assembles, by the same canonical form

.. math::

   \\mathcal{R}(u) = -\\nabla \\cdot \\mathbf{F}(u, \\nabla u, \\mathbf{x}, t)
   + S(u, \\nabla u, \\mathbf{x}, t) = 0 ,

with the divergence taken in the problem's coordinate system.

Example
-------

::

    from dualmesh import mms

    study = mms.ManufacturedSolution(fields={"u": "sin(pi*x)*cos(pi*y)"}, dimension=2)
    study.add_physics(
        "coefficient_form_PDE",
        "pde",
        diffusion_coefficient="1 + x*y",
        convection_coefficient=[1.0, 0.5],
    )
    result = study.convergence_study(
        lambda n: dm.generate_rectangle_mesh(
            x_min=0, x_max=1, y_min=0, y_max=1, num_x_elements=n, num_y_elements=n
        ),
        levels=[4, 8, 16, 32],
        method="dmcdm",
    )
    print(result.table())
    result.rates("u", "l2")  # tends to 2 for linear elements

SymPy is needed only for this module, which imports it lazily.

References: P. J. Roache, "Code verification by the method of manufactured
solutions", *Journal of Fluids Engineering* 124 (2002) 4-10; K. Salari and
P. Knupp, *Code Verification by the Method of Manufactured Solutions*, Sandia
Report SAND2000-1444, 2000.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from . import _core
from .console import header, inner_study, report_level
from .problem import Problem

__all__ = ["ConvergenceResult", "ManufacturedSolution"]


def _sympy():
    try:
        import sympy
    except ImportError as error:  # pragma: no cover - depends on the environment
        raise ImportError(
            "dualmesh.mms needs SymPy to derive the manufactured sources; install it with 'pip install sympy'."
        ) from error
    return sympy


def _symbols():
    sp = _sympy()
    return sp.symbols("x y z t", real=True)


def _expr(value):
    """A number, a string or a SymPy expression, as a SymPy expression in the
    symbols x, y, z, t."""
    sp = _sympy()
    x, y, z, t = _symbols()
    if isinstance(value, str):
        return sp.sympify(value, locals={"x": x, "y": y, "z": z, "t": t, "pi": sp.pi, "e": sp.E})
    return sp.sympify(value)


def _text(expression) -> str:
    """SymPy's printed form, which the C++ expression compiler accepts as is."""
    return str(expression)


def _gradient(u):
    x, y, z, _t = _symbols()
    return [u.diff(x), u.diff(y), u.diff(z)]


# ---------------------------------------------------------------------------
# The manufactured problem
# ---------------------------------------------------------------------------
@dataclass
class ConvergenceResult:
    """Errors on a sequence of meshes, and the observed orders.

    ``sizes`` holds the characteristic element size :math:`h` of every mesh,
    taken as :math:`(|\\Omega| / N_e)^{1/d}`, which reduces to the element
    size for a uniform mesh and is well defined for any mesh.
    """

    method: str
    sizes: list = field(default_factory=list)
    num_dofs: list = field(default_factory=list)
    errors: dict = field(default_factory=dict)  # (variable, norm) -> list

    def rates(self, variable: str, norm: str = "l2") -> list:
        """Observed order between consecutive meshes,
        :math:`\\log(e_{k}/e_{k+1}) / \\log(h_{k}/h_{k+1})`."""
        e = self.errors[(variable, norm)]
        h = self.sizes
        return [math.log(e[k] / e[k + 1]) / math.log(h[k] / h[k + 1]) for k in range(len(e) - 1)]

    def order(self, variable: str, norm: str = "l2", last: int = 2) -> float:
        """Least-squares slope of :math:`\\log e` against :math:`\\log h` over
        the ``last`` finest meshes, which is the asymptotic order to report."""
        e = np.log(self.errors[(variable, norm)][-last:])
        h = np.log(self.sizes[-last:])
        return float(np.polyfit(h, e, 1)[0])

    def summary(self) -> str:
        """The errors on every mesh and the observed orders."""
        return self.table()

    def to_dict(self) -> dict:
        """The element sizes, the numbers of unknowns, the errors and the
        orders between consecutive meshes."""
        keys = sorted(self.errors)
        return {
            "study": "convergence",
            "method": self.method,
            "sizes": list(self.sizes),
            "num_dofs": list(self.num_dofs),
            "errors": {f"{v} {n}": list(self.errors[(v, n)]) for v, n in keys},
            "orders": {f"{v} {n}": self.rates(v, n) for v, n in keys},
        }

    def write_json(self, path) -> None:
        """Write :meth:`to_dict` to a JSON file."""
        from .parameters import write_json_numbers

        write_json_numbers(self.to_dict(), path)

    @property
    def tables(self) -> dict:
        """``errors``: one row per mesh, with the element size, the unknowns
        and the errors."""
        from .tables import Table

        keys = sorted(self.errors)
        rows = [
            [float(h), int(self.num_dofs[i]), *(float(self.errors[k][i]) for k in keys)]
            for i, h in enumerate(self.sizes)
        ]
        return {
            "errors": Table(
                ["element_size", "num_dofs", *(f"{v}_{n}_error" for v, n in keys)],
                None,
                rows,
                title=f"Convergence study ({self.method})",
            )
        }

    def write_csv(self, path, table: str | None = None) -> None:
        """Write the table ``errors`` to a CSV file."""
        from .tables import ResultTables

        ResultTables.write_csv(self, path, table)

    def table(self) -> str:
        keys = sorted(self.errors)
        header = f"{'h':>10s} {'dofs':>8s}" + "".join(f" {v + ' ' + n:>18s}" for v, n in keys)
        lines = [f"method: {self.method}", header]
        for i, h in enumerate(self.sizes):
            row = f"{h:10.4e} {self.num_dofs[i]:8d}"
            for key in keys:
                row += f" {self.errors[key][i]:18.6e}"
            lines.append(row)
        lines.append(
            "orders (finest pair): "
            + ", ".join(f"{v} {n} {self.rates(v, n)[-1]:.2f}" for v, n in keys)
        )
        return "\n".join(lines)


class ManufacturedSolution:
    """A problem whose exact solution is chosen, not computed.

    ``fields`` maps each variable name to its exact solution, as text or as a
    SymPy expression in ``x``, ``y``, ``z`` and ``t``.  The equations are
    physics of the physics level, added with :meth:`add_physics` exactly as
    to a :class:`~dualmesh.Problem`.  ``coordinates`` is the problem's
    coordinate system, which decides the form of the divergence:

    * Cartesian: :math:`\\nabla \\cdot \\mathbf{F} = \\sum_d \\partial F_d / \\partial x_d`;
    * axisymmetric, with :math:`x = r` and :math:`y = z`:
      :math:`r^{-1} \\partial (r F_r) / \\partial r + \\partial F_z / \\partial z`;
    * spherical, with :math:`x = r`:
      :math:`r^{-2} \\partial (r^2 F_r) / \\partial r`.

    The exact solution is imposed on every side set, except on the
    ``flux_boundaries``, which map a side-set name to its outward unit
    normal and get the exact flux :math:`\\mathbf{F} \\cdot \\mathbf{n}`
    (for a flow, the exact traction :math:`\\boldsymbol{\\sigma} \\mathbf{n}`).
    The pressure of an incompressible flow is never prescribed on the
    boundary.  A flow without a flux boundary is enclosed, determines its
    pressure only up to a constant, and its pressure is held at its exact
    value at the ``pressure_pin_point`` of the physics (the first mesh node
    when it is not given).

    An enclosed flow with a velocity that crosses the boundary is a poor
    manufactured problem for a stabilised equal-order method: the nodal
    interpolant of the exact velocity carries a net flux
    :math:`\\oint \\mathbf{u}_h \\cdot \\mathbf{n}\\,\\mathrm{d}s` of order
    :math:`h^2`, the one mass equation that the pin replaces is the only
    place where that imbalance can go, and the pressure-Laplacian character
    of the stabilisation, with its coefficient :math:`\\tau \\sim h^2`,
    turns it into an :math:`O(1)` pressure spike at the pin.  A flux
    boundary removes both the pin and the constraint.
    """

    def __init__(
        self,
        fields: dict,
        dimension: int,
        coordinates: str = "cartesian",
        flux_boundaries: dict | None = None,
    ):
        self.fields = {name: _expr(value) for name, value in fields.items()}
        self.dimension = int(dimension)
        self.coordinates = coordinates
        self.flux_boundaries = {
            k: [float(c) for c in n] for k, n in (flux_boundaries or {}).items()
        }
        self._physics: list = []
        self._forcing = None

    def add_physics(self, physics: str, name: str | None = None, **parameters):
        """Add a physics to the equations, with the type name and the
        parameters of :meth:`dualmesh.Problem.add_physics`.  Every variable
        of the physics needs an exact field."""
        from . import physics as _physics

        name = name or physics
        if any(n == name for n, _t, _p, _i in self._physics):
            raise _physics.InputError(
                f"The manufactured solution already has a physics named '{name}'."
            )
        instance = _physics.create(physics, name, dict(parameters))
        if not isinstance(instance, _physics.Physics):
            raise _physics.InputError(
                f"'{physics}' is a coupling. A manufactured solution takes physics only."
            )
        instance._resolve(self.coordinates, self.dimension)
        missing = [v for v in instance.variable_names() if v not in self.fields]
        if missing:
            raise _physics.InputError(
                f"{physics} '{name}': give the exact field of {', '.join(missing)} in 'fields'."
            )
        self._physics.append((name, physics, dict(parameters), instance))
        self._forcing = None
        return instance

    def _terms(self):
        if not self._physics:
            raise ValueError("ManufacturedSolution: add the equations with add_physics first.")
        for _name, _type, _parameters, instance in self._physics:
            yield instance._manufactured_terms(self.fields, self.coordinates, self.dimension)

    def _pressures(self) -> set:
        return {i.pressure for _n, t, _p, i in self._physics if t == "incompressible_flow"}

    # ---- symbolic ---------------------------------------------------------
    def _divergence(self, flux):
        x, y, z, _t = _symbols()
        axes = (x, y, z)
        if self.coordinates == "axisymmetric":
            return (
                (x * flux[0]).diff(x) / x + flux[1].diff(y)
                if self.dimension > 1
                else (x * flux[0]).diff(x) / x
            )
        if self.coordinates == "spherical":
            return (x**2 * flux[0]).diff(x) / x**2
        return sum(_expr(flux[d]).diff(axes[d]) for d in range(self.dimension))

    def flux(self, variable: str) -> list:
        """The exact flux :math:`\\mathbf{F}` of ``variable``'s equation, summed
        over the physics; :math:`\\mathbf{F} \\cdot \\mathbf{n}` is the boundary
        flux a ``Neumann_boundary_condition`` prescribes."""
        total = [0, 0, 0]
        for parts in self._terms():
            if variable in parts:
                total = [a + _expr(b) for a, b in zip(total, parts[variable][0])]
        return total

    def forcing(self) -> dict:
        """The source each equation needs for the chosen fields to satisfy it:
        :math:`f = -\\nabla \\cdot \\mathbf{F}(u) + S(u)`, which a ``body_force``
        of intensity :math:`f` (whose own source is :math:`-f`) cancels."""
        if self._forcing is None:
            flux = {v: [0, 0, 0] for v in self.fields}
            source = dict.fromkeys(self.fields, 0)
            for parts in self._terms():
                for variable, (F, S) in parts.items():
                    flux[variable] = [a + _expr(b) for a, b in zip(flux[variable], F)]
                    source[variable] = source[variable] + _expr(S)
            forcing = {
                v: -self._divergence([_expr(c) for c in flux[v]]) + source[v] for v in self.fields
            }
            if self.coordinates != "cartesian":
                # The curvilinear divergence and the hoop terms divide by r,
                # and for a field that is regular on the axis the divisions
                # cancel. Brought over one denominator they do so
                # symbolically, and the forcing can then be evaluated on the
                # axis itself, as the cell-centred method does at its faces.
                sympy = _sympy()
                forcing = {
                    v: sympy.cancel(sympy.together(sympy.expand(f))) for v, f in forcing.items()
                }
            self._forcing = forcing
        return self._forcing

    def exact(self, variable: str) -> str:
        return _text(self.fields[variable])

    def exact_gradient(self, variable: str) -> list:
        return [_text(g) for g in _gradient(self.fields[variable])]

    # ---- numerical ----------------------------------------------------------
    def build(
        self,
        mesh,
        method: str = "dmcdm",
        boundary: Sequence[str] | None = None,
        start_time: float = 0.0,
        **problem_options,
    ) -> Problem:
        """A dualmesh problem for this manufactured solution on ``mesh``: the
        physics, the manufactured source of every equation, and the exact
        solution imposed on ``boundary`` (every side set when omitted) less
        the flux boundaries.  The unknowns start at the exact solution at
        ``start_time``, which is the initial condition of a transient study
        and a good first iterate for a nonlinear one."""
        problem = Problem(mesh, method=method, coordinates=self.coordinates, **problem_options)
        forcing = {v: _text(f) for v, f in self.forcing().items()}
        served, pins = set(), []
        for name, type_name, parameters, _instance in self._physics:
            parameters = dict(parameters)
            if type_name == "incompressible_flow":
                pins.append(
                    (
                        parameters.pop("pressure_pin_point", None),
                        parameters.get("pressure", "pressure"),
                    )
                )
            physics = problem.add_physics(type_name, name, **parameters)
            served |= physics._set_manufactured_source(
                {v: forcing[v] for v in physics.variable_names()}
            )
        pressures = self._pressures()
        boundaries = list(boundary) if boundary is not None else list(mesh.sideset_names())
        walls = [b for b in boundaries if b not in self.flux_boundaries]
        for variable, f in self.forcing().items():
            if variable not in served and not (variable in pressures and _expr(f) == 0):
                problem.add_kernel(
                    "body_force",
                    f"manufactured_{variable}",
                    variable=variable,
                    value=forcing[variable],
                    scale_with_load=False,
                )
            if variable in pressures:
                continue
            if walls:
                problem.add_boundary_condition(
                    "Dirichlet_boundary_condition",
                    f"exact_{variable}",
                    variable=variable,
                    boundary=walls,
                    value=self.exact(variable),
                )
            flux = self.flux(variable)
            for side, normal in self.flux_boundaries.items():
                traction = sum(flux[d] * normal[d] for d in range(len(normal)))
                problem.add_boundary_condition(
                    "Neumann_boundary_condition",
                    f"flux_{variable}_{side}",
                    variable=variable,
                    boundary=side,
                    flux=_text(traction),
                )
        if not self.flux_boundaries:
            for point, pressure in pins:
                pin = list(point) if point is not None else list(mesh.node(0))
                problem.add_boundary_condition(
                    "point_Dirichlet_boundary_condition",
                    f"exact_{pressure}_level",
                    variable=pressure,
                    point=[float(c) for c in pin],
                    value=self.exact(pressure),
                )
        points = problem.entity_points()
        for variable in self.fields:
            exact = _core.ParsedFunction(self.exact(variable))
            problem.set_values(
                variable, np.array([exact(p[0], p[1], p[2], start_time) for p in points])
            )
        return problem

    def errors(self, problem: Problem) -> dict:
        """``{(variable, "l2"): value, (variable, "h1"): value}`` for a solved
        problem, at the problem's current time."""
        out = {}
        for variable in self.fields:
            l2, h1 = problem.error_norms(
                variable, self.exact(variable), self.exact_gradient(variable)
            )
            out[(variable, "l2")] = l2
            out[(variable, "h1")] = h1
        return out

    def convergence_study(
        self,
        mesh_factory: Callable[[int], object],
        levels: Sequence[int],
        method: str = "dmcdm",
        transient: dict | None = None,
        solve_options: dict | None = None,
        report: str = "full",
        **problem_options,
    ) -> ConvergenceResult:
        """Solve on ``mesh_factory(n)`` for every ``n`` in ``levels`` and
        record the errors.

        ``transient``, when given, holds the arguments of
        :meth:`~dualmesh.Problem.solve_transient`; its ``time_step`` may be a callable
        of the element size :math:`h`, so that the time step is refined
        together with the mesh, which is what a study of the combined
        space-time order needs.

        ``report`` sets what the study prints: ``"full"`` (the default: a
        line for each mesh and the summary), ``"summary"`` (one line that
        names the study, then the summary) or ``"none"``.  The solves inside
        the study print nothing.
        """
        level = report_level(report, "ManufacturedSolution.convergence_study")
        if level == "full":
            print(header("convergence study"))
        if level != "none":
            print(f"dualmesh convergence study: method {method}, {len(levels)} meshes", flush=True)
        result = ConvergenceResult(method=method)
        solve_options = dict(solve_options or {})
        for n in levels:
            mesh = mesh_factory(n)
            start = float(transient.get("start_time", 0.0)) if transient else 0.0
            problem = self.build(mesh, method=method, start_time=start, **problem_options)
            h = _element_size(problem, mesh)
            with inner_study():
                if transient:
                    options = dict(transient)
                    if callable(options.get("time_step")):
                        options["time_step"] = float(options["time_step"](h))
                    problem.solve_transient(**options, **solve_options)
                else:
                    problem.solve(**solve_options)
            result.sizes.append(h)
            result.num_dofs.append(problem.num_active_dofs())
            for key, value in self.errors(problem).items():
                result.errors.setdefault(key, []).append(value)
            if level == "full":
                print(
                    f"  mesh {n}: element size {h:.6g}, {problem.num_active_dofs()} unknowns",
                    flush=True,
                )
        if level != "none":
            print(result.summary())
        return result


def _element_size(problem: Problem, mesh) -> float:
    """:math:`(|\\Omega| / N_e)^{1/d}`, with the volume measured in the
    problem's own coordinates (without the axisymmetric or spherical factor,
    since the size of an element is a property of the mesh)."""
    probe = Problem(mesh, method="fem")
    probe.add_variable("one")
    probe.set_values("one", np.ones(len(probe.entity_points())))
    volume = probe.integrate("one")
    return (volume / mesh.num_elements) ** (1.0 / mesh.dimension)
