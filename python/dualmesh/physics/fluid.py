# SPDX-License-Identifier: LGPL-2.1-or-later
"""The fluid_dynamics physics: the flow of an incompressible Newtonian fluid."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import ClassVar

from .. import _core
from ..model.sympy_form import coordinate_symbols, sympy_gradient
from ..parameters import InputError, parameter
from .base import Physics, label, list_items, register, vector_components

FLOW_FORMULATIONS = ("equal_order", "Taylor-Hood", "penalty")
FLOW_TIME_INTEGRATIONS = ("monolithic", "projection")


def _is_quadratic_mesh(mesh) -> bool:
    return all(_core.element_is_quadratic(t) for t in mesh.element_types())


@register
@dataclass
class FluidDynamics(Physics):
    r"""The steady or transient flow of an incompressible Newtonian fluid.

    The velocity :math:`\mathbf{v}` in :math:`\mathrm{m/s}` and the pressure
    :math:`p` in :math:`\mathrm{Pa}` satisfy the balances of momentum and mass

    .. math::

        \rho \left( \frac{\partial \mathbf{v}}{\partial t}
        + (\mathbf{v} \cdot \nabla) \mathbf{v} \right)
        - \nabla \cdot \boldsymbol{\sigma} = \mathbf{f},
        \qquad \nabla \cdot \mathbf{v} = 0,
        \qquad \boldsymbol{\sigma} = -p \mathbf{I}
        + \mu \left( \nabla \mathbf{v} + \nabla \mathbf{v}^{\mathsf{T}} \right),

    with the density :math:`\rho` in :math:`\mathrm{kg/m^3}`, the dynamic
    viscosity :math:`\mu` in :math:`\mathrm{Pa\,s}`, the Cauchy stress
    :math:`\boldsymbol{\sigma}` in :math:`\mathrm{Pa}`, the identity
    :math:`\mathbf{I}` and the body force per unit volume :math:`\mathbf{f}`
    in :math:`\mathrm{N/m^3}`.  The inertia terms are present when
    ``density`` is not zero, and without them the flow is a Stokes flow.  The
    variables are ``velocity_x``, ``velocity_y`` and ``velocity_z``, or
    ``velocity_r`` and ``velocity_z`` in axisymmetric coordinates, and
    ``pressure``.

    The equal_order formulation keeps the pressure as a variable of the same
    order as the velocity, with a residual-based stabilisation.  The
    Taylor-Hood formulation keeps it with a quadratic velocity and a linear
    pressure.  The penalty formulation eliminates the pressure by
    :math:`p = -\gamma \nabla \cdot \mathbf{v}`, with the penalty parameter
    :math:`\gamma` in :math:`\mathrm{Pa\,s}`."""

    type_name: ClassVar[str] = "fluid_dynamics"
    module: ClassVar[str] = "fluids"
    vector: ClassVar[bool] = True

    #: The velocity field: the variables velocity_x, velocity_y, velocity_z (velocity_r, velocity_z in axisymmetric coordinates).
    velocity: ClassVar[str] = "velocity"
    dynamic_viscosity: float = parameter(
        1.0,
        unit="Pa s",
        description="Dynamic viscosity mu. Default 1, the value of the non-dimensional equations.",
    )
    density: float = parameter(
        0.0,
        unit="kg/m^3",
        description="Density rho of the inertia terms rho dv/dt and rho (v . grad) v. Default 0: a Stokes flow, the limit of a slow or very viscous flow, whose inertia is negligible. The buoyancy force of a nonisothermal_flow coupling needs the positive density of the fluid.",
    )
    formulation: str | None = parameter(
        None,
        description="equal_order (the velocity and the pressure of the same order, with pressure stabilisation), Taylor-Hood (a quadratic velocity and a linear pressure, stable without stabilisation, which needs method='FEM' and a quadratic mesh) or penalty (the pressure eliminated by p = -gamma div v, which makes the flow slightly compressible). Default: Taylor-Hood when method='FEM' is given on a quadratic mesh, since it needs no stabilisation, and equal_order otherwise.",
    )
    penalty_parameter: float | None = parameter(
        None,
        description="Penalty parameter gamma (Pa s) of the penalty formulation, p = -gamma div v. Default 1e8: the divergence of the velocity is -p/gamma, so it stays about 1e-8 of the velocity gradient in a non-dimensional problem. Only the penalty formulation takes it.",
    )
    #: The pressure variable of the pressure and Taylor-Hood formulations.
    pressure: ClassVar[str] = "pressure"
    body_force: Sequence[float | str] | None = parameter(
        None,
        unit="N/m^3",
        description="Body force per unit volume, one value per component, each a constant or an expression of x, y, z and t. It enters the momentum equations and their streamline stabilisation. Default none: no body force.",
    )
    block: Sequence[str] = parameter(
        (),
        description="Blocks of the fluid. Default: every block, a mesh of the fluid alone.",
    )
    initial_condition: Mapping[str, object] = parameter(
        default_factory=dict,
        description="The initial values of the fields of a transient flow: field name -> value, the velocity as one value per component, for example {'velocity': ['1 - y^2', 0], 'pressure': 0}. Each value is a constant, the name of a function or an expression of x, y and z. Default: at rest, the state before the flow starts.",
    )
    viscous_form: str | None = parameter(
        None,
        description="The form of the viscous term: stress, mu (grad v + grad v^T), whose natural boundary condition is a zero traction, or laplacian, mu grad v, which gives the same equations for a constant viscosity and whose natural boundary condition is the do-nothing outflow mu dv/dn - p n = 0. Every solve of the flow, steady or transient, takes the same form. Default stress, and laplacian with the projection time integration: stress holds for a viscosity that varies, and its natural boundary quantity is the traction, the force of the fluid on the boundary. The pressure of the projection is that of the laplacian form, whose difference from the stress form, the gradient of the divergence, the splitting leaves nonzero, so the stress form costs the projection part of its pressure accuracy.",
    )
    time_integration: str = parameter(
        "monolithic",
        description="How a transient solve advances the flow: monolithic, Newton's method on the velocity and the pressure together with the time scheme of solve_transient, or projection, a splitting into a pressure Poisson equation and then the Helmholtz equations of the velocity, with the backward differentiation formula (BDF) and the extrapolation of the advection (EXT) of order time_order. The projection takes the equal_order or the Taylor-Hood formulation and a positive density. Default monolithic: it couples the velocity and the pressure in every step, with no splitting error.",
    )
    time_order: int = parameter(
        2,
        description="The order k of the backward differentiation formula BDFk and of the extrapolation EXTk of the advection in the projection time integration: 1, 2 or 3. Default 2: second-order accuracy in time from the history of two steps. The first steps take the orders below it.",
    )

    def __post_init__(self):
        if self.time_integration not in FLOW_TIME_INTEGRATIONS:
            raise InputError(
                f"fluid_dynamics: unknown time_integration '{self.time_integration}'. Use {', '.join(FLOW_TIME_INTEGRATIONS)}."
            )
        if self.viscous_form is None:
            self.viscous_form = "laplacian" if self.time_integration == "projection" else "stress"
        if self.viscous_form not in ("stress", "laplacian"):
            raise InputError(
                f"fluid_dynamics: viscous_form is stress or laplacian, not '{self.viscous_form}'."
            )
        if self.time_integration == "projection":
            if self.formulation == "penalty":
                raise InputError(
                    "fluid_dynamics: the projection time integration solves for the pressure, which the penalty formulation eliminates. Give formulation='equal_order' or 'Taylor-Hood'."
                )
            if self.time_order not in (1, 2, 3):
                raise InputError("fluid_dynamics: time_order is 1, 2 or 3.")
            if not self.density > 0:
                raise InputError(
                    "fluid_dynamics: the projection time integration needs a positive density. A flow without inertia (Stokes flow) has no time derivative to integrate."
                )
        if self.formulation is not None and self.formulation not in FLOW_FORMULATIONS:
            raise InputError(
                f"fluid_dynamics: unknown formulation '{self.formulation}'. Use {', '.join(FLOW_FORMULATIONS)}."
            )

    @property
    def _pressure_formulation(self) -> bool:
        return self.formulation in ("equal_order", "Taylor-Hood")

    @property
    def velocities(self) -> list[str]:
        """The velocity variables, one per component."""
        dimension, axisymmetric = getattr(self, "_geometry", (2, False))
        return vector_components(self.velocity, dimension, axisymmetric)

    def _resolve(self, coordinates: str, dimension: int) -> None:
        self._geometry = (dimension, coordinates == "axisymmetric")

    def variable_names(self) -> list[str]:
        return self.velocities + ([self.pressure] if self._pressure_formulation else [])

    def force(self, boundary: str) -> dict:
        """The force of the fluid on ``boundary``, in N, or in N/m in a plane
        flow, by direction: ``{"x": Fx, "y": Fy}``, the drag and the lift of a body in
        the flow.  On a wall of prescribed velocity it is the reaction there,
        conserved exactly (:meth:`~dualmesh.Problem.boundary_flux`)."""
        return {
            v[len(self.velocity) + 1 :]: self._problem.boundary_flux(v, boundary)
            for v in self.velocities
        }

    def flow_rate(self, boundary: str) -> float:
        """The volume of fluid leaving through ``boundary`` per unit time
        in m^3/s, or in m^2/s in a plane flow.  With a pressure it is the flow of the
        mass equation, which the flow conserves exactly."""
        if self._pressure_formulation:
            return self._problem.boundary_flux(self.pressure, boundary)
        return self._problem._sum(
            float(self._problem._problem.normal_integral(self.velocities, boundary))
        )

    def _initial_condition(self, variable: str):
        unknown = sorted(set(self.initial_condition) - {self.velocity, self.pressure})
        if unknown:
            raise InputError(
                f"{label(self.type_name, self.name)}: the initial_condition names {', '.join(unknown)}, which is not a field of this flow ({self.velocity}, {self.pressure})."
            )
        if variable == self.pressure:
            return self.initial_condition.get(self.pressure)
        velocity = self.initial_condition.get(self.velocity)
        if velocity is None:
            return None
        values = (
            list_items(velocity, len(self.velocities), label(self.type_name, self.name))
            if isinstance(velocity, str)
            else list(velocity)
        )
        if len(values) != len(self.velocities):
            raise InputError(
                f"{label(self.type_name, self.name)}: the initial velocity needs {len(self.velocities)} values, one per component."
            )
        return values[self.velocities.index(variable)]

    def _variable_units(self) -> dict:
        return {**dict.fromkeys(self.velocities, "m/s"), self.pressure: "Pa"}

    def _dimension(self) -> int:
        return len(self.velocities)

    def _variable_order(self) -> dict:
        return {self.pressure: "first"} if self.formulation == "Taylor-Hood" else {}

    def _attach(self, problem) -> None:
        self._resolve(problem.coordinates, problem.mesh.dimension)
        if self.formulation is None:
            # Taylor-Hood is stable without stabilisation, so it is the default where it applies:
            # FEM given on a quadratic mesh (a flow takes DMCDM when the method is left out).
            taylor_hood = (
                problem._method_given
                and problem.method == "FEM"
                and _is_quadratic_mesh(problem.mesh)
            )
            self.formulation = "Taylor-Hood" if taylor_hood else "equal_order"
        if self.penalty_parameter is not None and self.formulation != "penalty":
            raise InputError(
                f"{label(self.type_name, self.name)}: penalty_parameter is a parameter of formulation='penalty', and this flow has formulation='{self.formulation}'."
            )
        if self.formulation == "Taylor-Hood":
            if problem.method != "FEM":
                raise InputError(
                    f"{label(self.type_name, self.name)}: the Taylor-Hood element needs the finite element method, dualmesh.Problem(mesh, method='FEM'). The other methods use formulation='equal_order'."
                )
            if not _is_quadratic_mesh(problem.mesh):
                raise InputError(
                    f"{label(self.type_name, self.name)}: the Taylor-Hood element needs quadratic elements (e.g., element_type='Quad9', 'Tri6', 'Tet10' or 'Hex27', or mesh.second_order())."
                )
        super()._attach(problem)

    def _build(self, problem) -> None:
        velocities, restrict = list(self.velocities), self._restriction()
        # The penalty parameter of the penalty formulation, when it is given.
        penalty = (
            {} if self.penalty_parameter is None else {"penalty_parameter": self.penalty_parameter}
        )
        force = (
            [f if isinstance(f, str) else float(f) for f in self.body_force]
            if self.body_force is not None
            else [0.0] * len(velocities)
        )
        for component, variable in enumerate(velocities):
            self._add_momentum_kernels(problem, variable, velocities, force[component], penalty)
        if self._pressure_formulation:
            self._add_mass_equation(problem, velocities)
        else:
            problem.add_property("penalty_pressure", velocities=velocities, **penalty, **restrict)
        # A transient run starts from the projection of the initial velocity onto a field without
        # divergence that keeps the prescribed values.
        _core.attach_start_projection(
            problem._problem,
            velocities,
            self.pressure if self._pressure_formulation else "",
        )
        if self.time_integration == "projection":
            # The projection takes the terms of the flow from its kernels.
            settings = _core.ProjectionSettings()
            settings.order = self.time_order
            self._projection = _core.attach_projection(problem._problem, settings)

    def _add_momentum_kernels(self, problem, variable, velocities, force, penalty) -> None:
        """The kernels of the momentum equation of the velocity component ``variable``, with the
        body force ``force`` of that component and the ``penalty`` parameters of the penalty
        formulation."""
        restrict = self._restriction()

        def add(kernel: str, **parameters) -> None:
            problem.add_kernel(kernel, variable=variable, **parameters, **restrict)

        add(
            "viscous_stress",
            velocities=velocities,
            dynamic_viscosity=self.dynamic_viscosity,
            form=self.viscous_form,
        )
        if self._pressure_formulation:
            add("pressure_gradient", velocities=velocities, pressure=self.pressure)
            # The inertia of an equal-order flow takes the streamline stabilisation (SUPG), and
            # the mass equation finds its own pressure stabilisation. The Taylor-Hood element
            # needs neither.
            if self.formulation == "equal_order" and self.density != 0.0:
                add("momentum_stabilization", velocities=velocities)
        else:
            add("penalty_incompressibility", velocities=velocities, **penalty)
        if self.density:
            add("advection", velocity=velocities, coefficient=self.density)
        if force != 0.0:
            add("body_force", value=force)
        # The inertia rho dv/dt of a transient solve, which a steady solve leaves out. It comes
        # after the body force, so that the residual sums its terms in this order.
        if self.density:
            add("time_derivative", coefficient=self.density)

    def _add_mass_equation(self, problem, velocities) -> None:
        """The mass equation of a flow with a pressure variable, with the flow through every
        boundary."""
        # The stabilisation builds the momentum residual from the momentum kernels (and from
        # those that couplings add).
        problem.add_kernel(
            "mass_conservation",
            variable=self.pressure,
            velocities=velocities,
            **self._restriction(),
        )
        boundaries = list(problem.mesh.sideset_names())
        if boundaries:
            problem.add_boundary_condition(
                "mass_flux",
                boundaries,
                variable=self.pressure,
                velocities=velocities,
            )

    def _manufactured_terms(self, fields, coordinates, dimension):
        if not self._pressure_formulation:
            raise InputError(
                f"{label(self.type_name, self.name)}: manufactured solutions need formulation='equal_order' or 'Taylor-Hood', which keep the pressure as a variable."
            )
        x = coordinate_symbols()[0]
        u = [fields[v] for v in self.velocities]
        p = fields[self.pressure]
        mu, rho = float(self.dynamic_viscosity), float(self.density)
        grads = [sympy_gradient(c) for c in u]
        out = {}
        laplacian = self.viscous_form == "laplacian"
        for i, v in enumerate(self.velocities):
            flux = [
                mu * grads[i][d] if laplacian else mu * (grads[i][d] + grads[d][i])
                for d in range(dimension)
            ]
            flux[i] = flux[i] - p
            flux += [0] * (3 - dimension)
            source = rho * sum(u[d] * grads[i][d] for d in range(dimension))
            if coordinates == "axisymmetric" and i == 0:
                source = source + ((1 if laplacian else 2) * mu * u[0] / x - p) / x
            out[v] = (flux, source)
        out[self.pressure] = ([-c for c in u] + [0] * (3 - dimension), 0)
        return out

    def _set_manufactured_source(self, forcing):
        # The streamline stabilisation is consistent only when its residual
        # carries the same body force as the momentum equations.
        if self.body_force is not None:
            raise InputError(
                f"{label(self.type_name, self.name)}: a manufactured solution sets the body force itself, so leave body_force unset."
            )
        self.body_force = [forcing[v] for v in self.velocities]
        return set(self.velocities)
