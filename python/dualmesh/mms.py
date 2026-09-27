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

    def variable_orders(self) -> dict:
        """The interpolation order of the variables this term needs other than
        the mesh order, as ``{variable: "first"}``."""
        return {}

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
    """``diffusion`` kernel: :math:`\\mathbf{F} = k(\\mathbf{x}, t)\\, p(u)\\,
    \\nabla u` with :math:`p(u) = c_0 + c_1 u + c_2 u^2 + \\dots`.

    A non-constant ``polynomial`` makes the problem nonlinear.
    """

    def __init__(self, variable: str, diffusivity=1.0, polynomial: Sequence[float] = (1.0,)):
        self.variable = variable
        self.diffusivity = diffusivity
        self.polynomial = [float(c) for c in polynomial]

    def add_to(self, problem, name):
        problem.add_kernel(
            "diffusion",
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
    """``advection`` kernel with a constant velocity.  The non-conservative
    form contributes the source :math:`\\mathbf{v} \\cdot \\nabla u`, the
    conservative form the flux :math:`-\\mathbf{v} u`."""

    def __init__(
        self, variable: str, velocity=(1.0, 0.0, 0.0), advection_form: str = "non_conservative"
    ):
        self.variable = variable
        self.velocity = [float(v) for v in velocity] + [0.0] * (3 - len(velocity))
        self.advection_form = advection_form

    def add_to(self, problem, name):
        problem.add_kernel(
            "advection",
            name,
            variable=self.variable,
            velocity=self.velocity,
            advection_form=self.advection_form,
        )

    def contributions(self, fields, coordinates, dimension):
        u = fields[self.variable]
        v = self.velocity
        if self.advection_form == "conservative":
            return {self.variable: ([-vi * u for vi in v], 0)}
        g = _gradient(u)
        return {self.variable: ([0, 0, 0], sum(vi * gi for vi, gi in zip(v, g)))}


class Reaction(Term):
    """``reaction`` kernel: the source :math:`c\\, u^p`."""

    def __init__(self, variable: str, coefficient=1.0, exponent: float = 1.0):
        self.variable = variable
        self.coefficient = coefficient
        self.exponent = float(exponent)

    def add_to(self, problem, name):
        problem.add_kernel(
            "reaction",
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
    """``time_derivative`` kernel: the source :math:`c\\, \\partial u / \\partial t`
    of the continuous equation."""

    def __init__(self, variable: str, coefficient=1.0):
        self.variable = variable
        self.coefficient = coefficient

    def add_to(self, problem, name):
        problem.add_kernel(
            "time_derivative",
            name,
            variable=self.variable,
            coefficient=_coefficient(self.coefficient),
        )

    def contributions(self, fields, coordinates, dimension):
        _x, _y, _z, t = _symbols()
        u = fields[self.variable]
        return {self.variable: ([0, 0, 0], _expr(self.coefficient) * u.diff(t))}


class LinearElasticity(Term):
    """``linear_elastic_stress`` material and one ``stress_divergence`` kernel per
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
            "linear_elastic_stress",
            name + "_material",
            displacements=self.displacements,
            youngs_modulus=self.E,
            poissons_ratio=self.nu,
            formulation=self.formulation,
        )
        for component, variable in enumerate(self.displacements):
            problem.add_kernel(
                "stress_divergence", f"{name}_{variable}", variable=variable, component=component
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


class IncompressibleFlow(Term):
    """The pressure-velocity formulation of incompressible flow, as
    :func:`dualmesh.physics.add_incompressible_flow` with
    ``formulation="pressure"`` adds it: ``viscous_stress``,
    ``pressure_gradient`` and, with inertia, ``convective_inertia`` in every
    momentum equation, and ``mass_conservation`` on the pressure.

    The flux of momentum component :math:`i` is row :math:`i` of the stress,
    :math:`\\mu (\\nabla u_i + \\partial \\mathbf{u} / \\partial x_i) - p\\,
    \\mathbf{e}_i`, its source the convection :math:`\\rho\\, \\mathbf{u}
    \\cdot \\nabla u_i` and, in the axisymmetric radial equation, the hoop
    stress :math:`\\sigma_{\\theta\\theta} / r`; the mass equation has the
    flux :math:`-\\mathbf{u}`.  The stabilisation is not part of the
    continuous equations: it is given the manufactured momentum forcing, as a
    real calculation gives it its body forces, so that it is consistent.

    The velocity is prescribed on the boundary, except on the ``outflow``
    boundaries, which map a side-set name to its outward unit normal and get
    the exact traction :math:`\\boldsymbol{\\sigma} \\mathbf{n}` instead.  The
    pressure gets a ``mass_flux_boundary_condition`` on every boundary.  Without an outflow
    boundary the flow is enclosed, determines its pressure only up to a
    constant, and the pressure is fixed to its exact value at one point
    (``pressure_pin_point``, the first mesh node by default).

    ``formulation`` is ``"pressure"`` (the default: equal-order velocity and
    pressure with the residual-based stabilisation, as above) or
    ``"taylor_hood"``: the pressure is declared first order, which on a
    quadratic mesh with ``method="fem"`` gives quadratic velocity and linear
    pressure, stable without stabilisation; ``stabilization`` and
    ``streamline_stabilization`` are then refused if set.

    An enclosed flow with a velocity that crosses the boundary is a poor
    manufactured problem for a stabilised equal-order method: the nodal
    interpolant of the exact velocity carries a net flux
    :math:`\\oint \\mathbf{u}_h \\cdot \\mathbf{n}\\, ds` of order
    :math:`h^2`, the one mass equation that the pin replaces is the only
    place where that imbalance can go, and the pressure-Laplacian character
    of the stabilisation, with its coefficient :math:`\\tau \\sim h^2`,
    turns it into an :math:`O(1)` pressure spike at the pin.  An outflow
    boundary removes both the pin and the constraint.
    """

    def __init__(
        self,
        velocities: Sequence[str],
        pressure: str = "pressure",
        density: float = 0.0,
        dynamic_viscosity: float = 1.0,
        stabilization: bool | None = None,
        streamline_stabilization: bool | None = None,
        pressure_pin_point: Sequence[float] | None = None,
        outflow: dict | None = None,
        formulation: str = "pressure",
    ):
        if formulation not in ("pressure", "taylor_hood"):
            raise ValueError(
                f"Unknown flow formulation '{formulation}'. Use 'pressure' (stabilised equal "
                "order) or 'taylor_hood' (quadratic velocity, linear pressure)."
            )
        self.formulation = formulation
        self.velocities = list(velocities)
        self.pressure = pressure
        self.density = float(density)
        self.dynamic_viscosity = float(dynamic_viscosity)
        if formulation == "taylor_hood":
            if stabilization or streamline_stabilization:
                raise ValueError(
                    "The Taylor-Hood element is stable without stabilisation; leave "
                    "'stabilization' and 'streamline_stabilization' unset."
                )
            stabilization = False
            streamline_stabilization = False
        self.stabilization = True if stabilization is None else bool(stabilization)
        if streamline_stabilization is None:
            streamline_stabilization = self.stabilization and self.density != 0.0
        self.streamline_stabilization = bool(streamline_stabilization)
        self.pressure_pin_point = pressure_pin_point
        self.outflow = {k: [float(c) for c in n] for k, n in (outflow or {}).items()}

    def variable_orders(self) -> dict:
        return {self.pressure: "first"} if self.formulation == "taylor_hood" else {}

    def add_to(self, problem, name, forcing=None):
        common = {
            "velocities": self.velocities,
            "density": self.density,
            "dynamic_viscosity": self.dynamic_viscosity,
        }
        if forcing is not None:
            common["body_force"] = [_text(forcing[v]) for v in self.velocities]
        for i, v in enumerate(self.velocities):
            problem.add_kernel(
                "viscous_stress",
                f"{name}_viscous_{v}",
                variable=v,
                component=i,
                velocities=self.velocities,
                dynamic_viscosity=self.dynamic_viscosity,
            )
            problem.add_kernel(
                "pressure_gradient",
                f"{name}_pressure_{v}",
                variable=v,
                component=i,
                pressure=self.pressure,
            )
            if self.density:
                problem.add_kernel(
                    "convective_inertia",
                    f"{name}_inertia_{v}",
                    variable=v,
                    component=i,
                    velocities=self.velocities,
                    density=self.density,
                )
            if self.streamline_stabilization:
                problem.add_kernel(
                    "momentum_stabilization",
                    f"{name}_supg_{v}",
                    variable=v,
                    component=i,
                    pressure=self.pressure,
                    **common,
                )
        problem.add_kernel(
            "mass_conservation",
            f"{name}_mass",
            variable=self.pressure,
            stabilization=self.stabilization,
            **common,
        )

    def boundary_conditions(self, problem, name, boundaries, exact_pressure, mesh):
        problem.add_boundary_condition(
            "mass_flux_boundary_condition",
            f"{name}_mass_flux",
            variable=self.pressure,
            boundary=boundaries,
            velocities=self.velocities,
        )
        if self.outflow:
            return
        pin = (
            list(self.pressure_pin_point)
            if self.pressure_pin_point is not None
            else list(mesh.node(0))
        )
        problem.add_boundary_condition(
            "point_Dirichlet_boundary_condition",
            f"{name}_pressure_level",
            variable=self.pressure,
            point=pin,
            value=exact_pressure,
        )

    def contributions(self, fields, coordinates, dimension):
        x, _y, _z, _t = _symbols()
        u = [fields[v] for v in self.velocities]
        p = fields[self.pressure]
        mu, rho = self.dynamic_viscosity, self.density
        grads = [_gradient(c) for c in u]
        out = {}
        for i, v in enumerate(self.velocities):
            flux = [mu * (grads[i][d] + grads[d][i]) for d in range(dimension)]
            flux[i] = flux[i] - p
            flux += [0] * (3 - dimension)
            source = rho * sum(u[d] * grads[i][d] for d in range(dimension))
            if coordinates == "axisymmetric" and i == 0:
                source = source + (2 * mu * u[0] / x - p) / x
            out[v] = (flux, source)
        out[self.pressure] = ([-c for c in u] + [0] * (3 - dimension), 0)
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

    def flux(self, variable: str) -> list:
        """The exact flux :math:`\\mathbf{F}` of ``variable``'s equation, summed
        over the terms; :math:`\\mathbf{F} \\cdot \\mathbf{n}` is the boundary
        flux a ``Neumann_boundary_condition`` prescribes."""
        total = [0, 0, 0]
        for term in self.terms:
            parts = term.contributions(self.fields, self.coordinates, self.dimension)
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
            for term in self.terms:
                for variable, (F, S) in term.contributions(
                    self.fields, self.coordinates, self.dimension
                ).items():
                    flux[variable] = [a + _expr(b) for a, b in zip(flux[variable], F)]
                    source[variable] = source[variable] + _expr(S)
            forcing = {
                v: -self._divergence([_expr(c) for c in flux[v]]) + source[v] for v in self.fields
            }
            if self.coordinates != "cartesian":
                # The curvilinear divergence and the hoop terms divide by r,
                # and for a field that is regular on the axis the divisions
                # cancel; brought over one denominator they do so
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
        variables, the terms, one ``body_force`` per equation carrying the
        manufactured source, and the exact solution imposed on ``boundary``
        (every side set when omitted).  The unknowns start at the exact
        solution at ``start_time``, which is the initial condition of a
        transient study and a good first iterate for a nonlinear one."""
        problem = Problem(mesh, method=method, coordinates=self.coordinates, **problem_options)
        orders = {}
        for term in self.terms:
            orders.update(term.variable_orders())
        for variable in self.fields:
            problem.add_variable(variable, order=orders.get(variable, "mesh"))
        flows = [t for t in self.terms if isinstance(t, IncompressibleFlow)]
        for index, term in enumerate(self.terms):
            if isinstance(term, IncompressibleFlow):
                term.add_to(problem, f"term{index}", forcing=self.forcing())
            else:
                term.add_to(problem, f"term{index}")
        boundaries = list(boundary) if boundary is not None else list(mesh.sideset_names())
        pressures = {flow.pressure for flow in flows}
        for variable, f in self.forcing().items():
            if not (variable in pressures and _expr(f) == 0):
                problem.add_kernel(
                    "body_force", f"manufactured_{variable}", variable=variable, value=_text(f)
                )
            if variable in pressures:
                continue
            outflow = {}
            for flow in flows:
                if variable in flow.velocities:
                    outflow.update(flow.outflow)
            walls = [b for b in boundaries if b not in outflow]
            if walls:
                problem.add_boundary_condition(
                    "Dirichlet_boundary_condition",
                    f"exact_{variable}",
                    variable=variable,
                    boundary=walls,
                    value=self.exact(variable),
                )
            flux = self.flux(variable)
            for side, normal in outflow.items():
                traction = sum(flux[d] * normal[d] for d in range(len(normal)))
                problem.add_boundary_condition(
                    "Neumann_boundary_condition",
                    f"traction_{variable}_{side}",
                    variable=variable,
                    boundary=side,
                    flux=_text(traction),
                )
        for index, flow in enumerate(flows):
            flow.boundary_conditions(
                problem, f"flow{index}", boundaries, self.exact(flow.pressure), mesh
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
            result.num_dofs.append(problem.num_active_dofs())
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
