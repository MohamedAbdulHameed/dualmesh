# SPDX-License-Identifier: LGPL-2.1-or-later
"""The couplings between two physics: thermal_expansion and nonisothermal_flow."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar

from ..parameters import InputError, parameter
from .base import Coupling, Physics, register
from .fluid import FluidDynamics
from .heat import HeatConduction
from .solid import SolidMechanics


@register
@dataclass
class ThermalExpansion(Coupling):
    r"""The thermal strain of a solid_mechanics physics, from the temperature of
    a heat_conduction physics.

    The thermal strain is

    .. math::

        \boldsymbol{\varepsilon}_{\mathrm{th}} = \alpha(T)\,(T - T_0)\,\mathbf{I},

    with the linear expansion coefficient :math:`\alpha` in
    :math:`\mathrm{1/K}`, the temperature :math:`T` in :math:`\mathrm{K}`, the
    stress-free temperature :math:`T_0` in :math:`\mathrm{K}` and the identity
    :math:`\mathbf{I}`.  The coefficient :math:`\alpha` is the secant (mean)
    coefficient between :math:`T_0` and :math:`T`, the default definition of
    the Abaqus Analysis User's Guide (section Thermal expansion), with
    :math:`T_0` as its reference temperature.  The solid takes the strain as an
    eigenstrain, a strain that produces no stress.  A beam, a plate or a shell
    integrates it through the thickness into a thermal force and moment."""

    type_name: ClassVar[str] = "thermal_expansion"
    module: ClassVar[str] = "solid_mechanics"

    linear_expansion_coefficient: float | str = parameter(
        unit="1/K",
        description="Linear expansion coefficient alpha of an isotropic solid, the secant (mean) coefficient between T_0 and T: a constant, the name of a registered function, or an expression of the temperature, the other fields and the coordinates, such as '1.0e-5 + 4.0e-9*temperature'. In a beam, a plate or a shell, z is the height above the mid-surface, so alpha(z) gives a graded section.",
    )
    stress_free_temperature: float = parameter(
        unit="K",
        description="Temperature T_0 at which the solid is free of thermal strain, which is also the reference temperature of the secant coefficient alpha.",
    )
    heat_conduction: Physics | None = parameter(
        None,
        description="The heat_conduction physics that gives the temperature, the object that add_physics returned. Default: the only heat_conduction physics of the problem, which needs no name.",
    )
    solid_mechanics: Physics | None = parameter(
        None,
        description="The solid_mechanics physics that expands, the object that add_physics returned. Default: the only solid_mechanics physics of the problem, which needs no name.",
    )

    def _apply(self, problem) -> None:
        heat = self._physics(problem, "heat_conduction", HeatConduction)
        solid = self._physics(problem, "solid_mechanics", SolidMechanics)
        solid._coupled["thermal_expansion"] = dict(
            temperature=heat.temperature,
            linear_expansion_coefficient=self.linear_expansion_coefficient,
            stress_free_temperature=self.stress_free_temperature,
        )


@register
@dataclass
class NonisothermalFlow(Coupling):
    r"""The coupling of a fluid_dynamics and a heat_conduction physics, in which
    the flow carries the heat and the temperature can drive the flow by
    buoyancy.

    The energy equation of the heat_conduction physics gains the advection term
    :math:`\rho c_p \mathbf{v} \cdot \nabla T`, with the density
    :math:`\rho` in :math:`\mathrm{kg/m^3}`, the specific heat capacity
    :math:`c_p` in :math:`\mathrm{J/(kg\,K)}`, the velocity :math:`\mathbf{v}`
    in :math:`\mathrm{m/s}` and the temperature :math:`T` in :math:`\mathrm{K}`.
    The energy equation takes :math:`\rho` and :math:`c_p` of the
    heat_conduction physics, which must give both, and a flow with inertia must
    have the same density.  When ``gravity`` and
    ``volumetric_expansion_coefficient`` are given, the momentum equations
    gain the Boussinesq buoyancy force per unit volume

    .. math::

        \mathbf{f} = -\rho \beta (T - T_0) \mathbf{g},

    with the volumetric expansion coefficient :math:`\beta` in
    :math:`\mathrm{1/K}`, the reference temperature :math:`T_0` in
    :math:`\mathrm{K}` and the gravitational acceleration :math:`\mathbf{g}`
    in :math:`\mathrm{m/s^2}`, which makes a natural convection problem.  The
    force takes the density of the fluid_dynamics physics, which must then be
    positive, and one component of :math:`\mathbf{g}` per velocity component."""

    type_name: ClassVar[str] = "nonisothermal_flow"
    module: ClassVar[str] = "fluids"

    heat_conduction: Physics | None = parameter(
        None,
        description="The heat_conduction physics, the object that add_physics returned. Default: the only heat_conduction physics of the problem, which needs no name.",
    )
    fluid_dynamics: Physics | None = parameter(
        None,
        description="The fluid_dynamics physics, the object that add_physics returned. Default: the only fluid_dynamics physics of the problem, which needs no name.",
    )
    gravity: Sequence[float] | None = parameter(
        None,
        unit="m/s^2",
        description="The gravitational acceleration g of the buoyancy force, one value per dimension. Default none: no buoyancy, a forced convection.",
    )
    volumetric_expansion_coefficient: float | None = parameter(
        None,
        unit="1/K",
        description="Volumetric thermal expansion coefficient beta of the fluid in the buoyancy force. Default none: no buoyancy, a forced convection.",
    )
    reference_temperature: float = parameter(
        0.0,
        unit="K",
        description="Temperature T_0 at which the buoyancy force is zero. Default 0, the reference of the non-dimensional equations, which measure the temperature from T_0.",
    )

    def __post_init__(self):
        if (self.gravity is None) != (self.volumetric_expansion_coefficient is None):
            raise InputError(
                "nonisothermal_flow: give both gravity and volumetric_expansion_coefficient for the buoyancy force, or neither."
            )

    def _apply(self, problem) -> None:
        # The stabilisation of the flow reads the buoyancy from the Boussinesq_buoyancy kernels.
        self._physics(problem, "heat_conduction", HeatConduction)
        self._physics(problem, "fluid_dynamics", FluidDynamics)

    def _build(self, problem) -> None:
        heat, flow = self.heat_conduction, self.fluid_dynamics
        # The energy equation takes one density and one specific heat, those of the heat_conduction
        # physics, in its heat capacity term, its convection and their stabilisation.
        if heat.specific_heat is None:
            raise InputError(
                "nonisothermal_flow: the flow carries the heat rho c_p v . grad T, so the heat_conduction physics needs its density and specific_heat. A non-dimensional problem gives the values of its equations, such as specific_heat = 1."
            )
        numbers = (int, float)
        if (
            isinstance(heat.density, numbers)
            and isinstance(flow.density, numbers)
            and flow.density != 0.0
            and heat.density != flow.density
        ):
            raise InputError(
                f"nonisothermal_flow: the heat_conduction physics has the density {heat.density} and the fluid_dynamics physics {flow.density}. The fluid has one density, so give both the same."
            )
        restrict = flow._restriction()
        # The heat that the flow carries: by default the streamline stabilisation (SUPG), which
        # removes the oscillations of the central scheme where convection dominates.
        scheme = heat.advection_scheme or "SUPG"
        problem.add_kernel(
            "advection",
            variable=heat.temperature,
            velocity=list(flow.velocities),
            coefficient=heat.heat_capacity(),
            advection_scheme="central" if scheme == "SUPG" else scheme,
            **restrict,
        )
        if scheme == "SUPG":
            # Its residual is the strong residual of the other kernels of the temperature, and it
            # takes the conductivity and the heat capacity of the diffusion and advection kernels.
            problem.add_kernel(
                "heat_streamline_stabilization",
                variable=heat.temperature,
                velocities=list(flow.velocities),
                **restrict,
            )
        if self.gravity is None:
            return
        # The buoyancy force rho beta (T - T_0) g of the momentum equations takes the density of
        # the fluid, so a flow under gravity needs it, a slow flow included.
        if not flow.density or flow.density < 0.0:
            raise InputError(
                f"nonisothermal_flow: the buoyancy force rho beta (T - T_0) g needs the density of the fluid, and the fluid_dynamics physics has the density {flow.density}. Give the fluid its density: a slow flow keeps its density, since its inertia is small by itself."
            )
        if len(self.gravity) != len(flow.velocities):
            raise InputError(
                f"nonisothermal_flow: gravity has {len(self.gravity)} components and the flow {len(flow.velocities)} velocities ({', '.join(flow.velocities)}). Give one component of gravity per velocity."
            )
        density = float(flow.density)
        for component, variable in enumerate(flow.velocities):
            if self.gravity[component] != 0.0:
                problem.add_kernel(
                    "Boussinesq_buoyancy",
                    variable=variable,
                    velocities=list(flow.velocities),
                    temperature=heat.temperature,
                    gravity=list(self.gravity),
                    density=density,
                    volumetric_expansion_coefficient=self.volumetric_expansion_coefficient,
                    reference_temperature=self.reference_temperature,
                    **restrict,
                )
