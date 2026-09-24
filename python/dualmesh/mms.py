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
invitation to error.  This module removes that step.  Each :class:`Term` knows
both which dualmesh objects it adds and, in SymPy, which flux and source those
objects contribute, so the source is derived from exactly the equations the
code assembles, by the same canonical form

.. math::

   \\mathcal{R}(u) = -\\nabla \\cdot \\mathbf{F}(u, \\nabla u, \\mathbf{x}, t)
   + S(u, \\nabla u, \\mathbf{x}, t) = 0 ,

with the divergence taken in the problem's coordinate system.

Example
-------

::

    from dualmesh import mms

    study = mms.ManufacturedSolution(
        fields={"u": "sin(pi*x)*cos(pi*y)"},
        terms=[mms.Diffusion("u", diffusivity="1 + x*y", polynomial=(1, 0.5)),
               mms.Advection("u", velocity=(1, 0.5, 0))],
        dimension=2,
    )
    result = study.convergence_study(
        lambda n: dm.generate_rectangle_mesh(x_min=0, x_max=1, y_min=0, y_max=1,
                                             num_x_elements=n, num_y_elements=n),
        levels=[4, 8, 16, 32], method="dmcdm")
    print(result.table())
    result.rates("u", "l2")      # tends to 2 for linear elements

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
from .problem import Problem

__all__ = [
    "Advection",
    "ConvergenceResult",
    "Diffusion",
    "LinearElasticity",
    "ManufacturedSolution",
    "Reaction",
    "Term",
    "TimeDerivative",
]


def _sympy():
    try:
        import sympy
    except ImportError as error:  # pragma: no cover - depends on the environment
        raise ImportError(
            "dualmesh.mms needs SymPy to derive the manufactured sources; "
            "install it with 'pip install sympy'."
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


# ---------------------------------------------------------------------------
# Terms
# ---------------------------------------------------------------------------
class Term:
    """One piece of the governing equations.

    A term does two things, and the point of the class is that the two cannot
    drift apart: :meth:`add_to` adds the dualmesh objects, and
    :meth:`contributions` returns, in SymPy, the flux and the source those
    objects put into the canonical form.
    """

    def add_to(self, problem: Problem, name: str) -> None:
        raise NotImplementedError

    def contributions(self, fields: dict, coordinates: str, dimension: int) -> dict:
        """Return ``{variable: (flux, source)}`` for the exact fields, with the
        flux a list of three SymPy expressions."""
        raise NotImplementedError


def _gradient(u):
    x, y, z, _t = _symbols()
    return [u.diff(x), u.diff(y), u.diff(z)]


def _coefficient(value):
    """A coefficient for a kernel parameter: a plain number when it is one, and
    otherwise its text, which the kernel compiles as an expression."""
    expression = _expr(value)
    if expression.is_number:
        return float(expression)
    return _text(expression)


class Diffusion(Term):
    """``Diffusion`` kernel: :math:`\\mathbf{F} = k(\\mathbf{x}, t)\\, p(u)\\,
    \\nabla u` with :math:`p(u) = c_0 + c_1 u + c_2 u^2 + \\dots`.

    A non-constant ``polynomial`` makes the problem nonlinear.
    """

    def __init__(self, variable: str, diffusivity=1.0, polynomial: Sequence[float] = (1.0,)):
        self.variable = variable
        self.diffusivity = diffusivity
        self.polynomial = [float(c) for c in polynomial]

    def add_to(self, problem, name):
        problem.add_kernel(
            "Diffusion",
            name,
            variable=self.variable,
            diffusivity=_coefficient(self.diffusivity),
            solution_polynomial=self.polynomial,
        )

    def contributions(self, fields, coordinates, dimension):
        u = fields[self.variable]
        p = sum(c * u**k for k, c in enumerate(self.polynomial))
        k = _expr(self.diffusivity)
        return {self.variable: ([k * p * g for g in _gradient(u)], 0)}


class Advection(Term):
    """``Advection`` kernel with a constant velocity.  The non-conservative
    form contributes the source :math:`\\mathbf{v} \\cdot \\nabla u`, the
    conservative form the flux :math:`-\\mathbf{v} u`."""

    def __init__(self, variable: str, velocity=(1.0, 0.0, 0.0), form: str = "non_conservative"):
        self.variable = variable
        self.velocity = [float(v) for v in velocity] + [0.0] * (3 - len(velocity))
        self.form = form

    def add_to(self, problem, name):
        problem.add_kernel(
            "Advection", name, variable=self.variable, velocity=self.velocity, form=self.form
        )

    def contributions(self, fields, coordinates, dimension):
        u = fields[self.variable]
        v = self.velocity
        if self.form == "conservative":
            return {self.variable: ([-vi * u for vi in v], 0)}
        g = _gradient(u)
        return {self.variable: ([0, 0, 0], sum(vi * gi for vi, gi in zip(v, g)))}


class Reaction(Term):
    """``Reaction`` kernel: the source :math:`c\\, u^p`."""

    def __init__(self, variable: str, coefficient=1.0, exponent: float = 1.0):
        self.variable = variable
        self.coefficient = coefficient
        self.exponent = float(exponent)

    def add_to(self, problem, name):
        problem.add_kernel(
            "Reaction",
            name,
            variable=self.variable,
            coefficient=_coefficient(self.coefficient),
            exponent=self.exponent,
        )

    def contributions(self, fields, coordinates, dimension):
        u = fields[self.variable]
        exponent = int(self.exponent) if self.exponent.is_integer() else self.exponent
        return {self.variable: ([0, 0, 0], _expr(self.coefficient) * u**exponent)}


class TimeDerivative(Term):
    """``TimeDerivative`` kernel: the source :math:`c\\, \\partial u / \\partial t`
    of the continuous equation."""

    def __init__(self, variable: str, coefficient=1.0):
        self.variable = variable
        self.coefficient = coefficient

    def add_to(self, problem, name):
        problem.add_kernel(
            "TimeDerivative",
            name,
            variable=self.variable,
            coefficient=_coefficient(self.coefficient),
        )

    def contributions(self, fields, coordinates, dimension):
        _x, _y, _z, t = _symbols()
        u = fields[self.variable]
        return {self.variable: ([0, 0, 0], _expr(self.coefficient) * u.diff(t))}


class LinearElasticity(Term):
    """``LinearElasticStress`` material and one ``StressDivergence`` kernel per
    displacement component.  The flux of component :math:`i` is row :math:`i`
    of the stress; in the axisymmetric formulation the radial equation also
    carries the hoop source :math:`\\sigma_{\\theta\\theta} / r`."""

    def __init__(
        self,
        displacements: Sequence[str],
        youngs_modulus: float,
        poissons_ratio: float,
        formulation: str = "plane_strain",
    ):
        self.displacements = list(displacements)
        self.E = float(youngs_modulus)
        self.nu = float(poissons_ratio)
        self.formulation = formulation

    def add_to(self, problem, name):
        problem.add_material(
            "LinearElasticStress",
            name + "_material",
            displacements=self.displacements,
            youngs_modulus=self.E,
            poissons_ratio=self.nu,
            formulation=self.formulation,
        )
        for component, variable in enumerate(self.displacements):
            problem.add_kernel(
                "StressDivergence", f"{name}_{variable}", variable=variable, component=component
            )

    def _stress(self, fields):
        """Voigt stress (xx, yy, zz, yz, xz, xy), as the material computes it."""
        x, y, z, _t = _symbols()
        E, nu = self.E, self.nu
        u = [fields[d] for d in self.displacements] + [0] * (3 - len(self.displacements))
        ux, uy, uz = u
        if self.formulation == "axisymmetric":
            # (rr, zz, theta-theta) in slots 0, 1, 2, and the rz shear in slot 5.
            strain = [ux.diff(x), uy.diff(y), ux / x, 0, 0, ux.diff(y) + uy.diff(x)]
        else:
            strain = [
                ux.diff(x),
                uy.diff(y),
                uz.diff(z) if self.formulation == "three_dimensional" else 0,
                uy.diff(z) + (uz.diff(y) if self.formulation == "three_dimensional" else 0),
                ux.diff(z) + (uz.diff(x) if self.formulation == "three_dimensional" else 0),
                ux.diff(y) + uy.diff(x),
            ]
        if self.formulation == "plane_stress":
            f = E / (1 - nu**2)
            s0 = f * (strain[0] + nu * strain[1])
            s1 = f * (strain[1] + nu * strain[0])
            return [s0, s1, 0, 0, 0, E / (2 * (1 + nu)) * strain[5]]
        lam = E * nu / ((1 + nu) * (1 - 2 * nu))
        mu = E / (2 * (1 + nu))
        trace = strain[0] + strain[1] + strain[2]
        return [
            lam * trace + 2 * mu * strain[0],
            lam * trace + 2 * mu * strain[1],
            lam * trace + 2 * mu * strain[2],
            mu * strain[3],
            mu * strain[4],
            mu * strain[5],
        ]

    def contributions(self, fields, coordinates, dimension):
        x, _y, _z, _t = _symbols()
        s = self._stress(fields)
        out = {}
        if self.formulation == "axisymmetric":
            out[self.displacements[0]] = ([s[0], s[5], 0], s[2] / x)
            out[self.displacements[1]] = ([s[5], s[1], 0], 0)
            return out
        rows = [[s[0], s[5], s[4]], [s[5], s[1], s[3]], [s[4], s[3], s[2]]]
        for component, variable in enumerate(self.displacements):
            out[variable] = (rows[component], 0)
        return out


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
    SymPy expression in ``x``, ``y``, ``z`` and ``t``.  ``terms`` are the
    pieces of the governing equations.  ``coordinates`` is the problem's
    coordinate system, which decides the form of the divergence:

    * Cartesian: :math:`\\nabla \\cdot \\mathbf{F} = \\sum_d \\partial F_d / \\partial x_d`;
    * axisymmetric, with :math:`x = r` and :math:`y = z`:
      :math:`r^{-1} \\partial (r F_r) / \\partial r + \\partial F_z / \\partial z`;
    * spherical, with :math:`x = r`:
      :math:`r^{-2} \\partial (r^2 F_r) / \\partial r`.
    """

    def __init__(
        self,
        fields: dict,
        terms: Sequence[Term],
        dimension: int,
        coordinates: str = "cartesian",
    ):
        self.fields = {name: _expr(value) for name, value in fields.items()}
        self.terms = list(terms)
        self.dimension = int(dimension)
        self.coordinates = coordinates
        self._forcing = None

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

    def forcing(self) -> dict:
        """The source each equation needs for the chosen fields to satisfy it:
        :math:`f = -\\nabla \\cdot \\mathbf{F}(u) + S(u)`, which a ``BodyForce``
        of intensity :math:`f` (whose own source is :math:`-f`) cancels."""
        if self._forcing is None:
            flux = {v: [0, 0, 0] for v in self.fields}
            source = dict.fromkeys(self.fields, 0)
            for term in self.terms:
                for variable, (F, S) in term.contributions(
                    self.fields, self.coordinates, self.dimension
                ).items():
                    flux[variable] = [a + _expr(b) for a, b in zip(flux[variable], F)]
                    source[variable] = source[variable] + _expr(S)
            self._forcing = {
                v: -self._divergence([_expr(c) for c in flux[v]]) + source[v] for v in self.fields
            }
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
        variables, the terms, one ``BodyForce`` per equation carrying the
        manufactured source, and the exact solution imposed on ``boundary``
        (every side set when omitted).  The unknowns start at the exact
        solution at ``start_time``, which is the initial condition of a
        transient study and a good first iterate for a nonlinear one."""
        problem = Problem(mesh, method=method, coordinates=self.coordinates, **problem_options)
        for variable in self.fields:
            problem.add_variable(variable)
        for index, term in enumerate(self.terms):
            term.add_to(problem, f"term{index}")
        boundaries = list(boundary) if boundary is not None else list(mesh.sideset_names())
        for variable, f in self.forcing().items():
            problem.add_kernel(
                "BodyForce", f"manufactured_{variable}", variable=variable, value=_text(f)
            )
            problem.add_boundary_condition(
                "DirichletBC",
                f"exact_{variable}",
                variable=variable,
                boundary=boundaries,
                value=self.exact(variable),
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
        **problem_options,
    ) -> ConvergenceResult:
        """Solve on ``mesh_factory(n)`` for every ``n`` in ``levels`` and
        record the errors.

        ``transient``, when given, holds the arguments of
        :meth:`~dualmesh.Problem.solve_transient`; its ``dt`` may be a callable
        of the element size :math:`h`, so that the time step is refined
        together with the mesh, which is what a study of the combined
        space-time order needs.
        """
        result = ConvergenceResult(method=method)
        solve_options = dict(solve_options or {})
        for n in levels:
            mesh = mesh_factory(n)
            start = float(transient.get("start_time", 0.0)) if transient else 0.0
            problem = self.build(mesh, method=method, start_time=start, **problem_options)
            h = _element_size(problem, mesh)
            if transient:
                options = dict(transient)
                if callable(options.get("dt")):
                    options["dt"] = float(options["dt"](h))
                problem.solve_transient(**options, **solve_options)
            else:
                problem.solve(**solve_options)
            result.sizes.append(h)
            result.num_dofs.append(len(problem.entity_points()) * len(self.fields))
            for key, value in self.errors(problem).items():
                result.errors.setdefault(key, []).append(value)
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
