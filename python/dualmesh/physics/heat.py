# SPDX-License-Identifier: LGPL-2.1-or-later
"""The heat_conduction physics: the energy balance of a solid or of a fluid at rest."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar

from ..model.sympy_form import coordinate_symbols, sympy_expression, sympy_gradient
from ..parameters import InputError, parameter
from .base import Physics, label, register


@register
@dataclass
class HeatConduction(Physics):
    r"""Heat conduction in a solid or in a fluid at rest, with an optional heat
    source and heat capacity.

    The temperature :math:`T` in :math:`\mathrm{K}` satisfies the energy balance

    .. math::

        \rho c_p \frac{\partial T}{\partial t} + \nabla \cdot \mathbf{q} = s,
        \qquad \mathbf{q} = -k \nabla T,

    with the density :math:`\rho` in :math:`\mathrm{kg/m^3}`, the specific
    heat capacity :math:`c_p` in :math:`\mathrm{J/(kg\,K)}`, the heat flux
    :math:`\mathbf{q}` in :math:`\mathrm{W/m^2}`, the thermal conductivity
    :math:`k` in :math:`\mathrm{W/(m\,K)}` and the heat generated per unit
    volume :math:`s` in :math:`\mathrm{W/m^3}`.  The heat capacity term is
    present when ``density`` and ``specific_heat`` are both given.  The
    conductivity is a constant, the name of a function, or, when it is not
    given, the property ``thermal_conductivity`` of the property objects of
    the problem.  The boundary condition ``heat_flux`` prescribes
    :math:`\mathbf{n} \cdot \mathbf{q}`, the heat flux that leaves the body
    along the outward unit normal :math:`\mathbf{n}`."""

    type_name: ClassVar[str] = "heat_conduction"
    module: ClassVar[str] = "heat_conduction"
    #: heat_flux, the heat leaving the body, is Neumann with the flux entering, -heat_flux.
    #: heat_source is body_force with the volumetric heat generation.
    kernel_words: ClassVar[dict] = {"heat_source": ("body_force", {"heat_source": "value"})}
    flux_name: ClassVar[str] = "heat_flux"

    #: The temperature variable.
    temperature: ClassVar[str] = "temperature"
    thermal_conductivity: float | str | None = parameter(
        None,
        unit="W/(m K)",
        description="Thermal conductivity k, a constant or the name of a function of x, y, z and t. Default: the property thermal_conductivity of the property objects of the problem, such as constant or parsed, so that the conductivity can differ between blocks and depend on the temperature.",
    )
    heat_source: float | str | None = parameter(
        None,
        unit="W/m^3",
        description="Heat generated per unit volume s, a constant or the name of a function. Default none: the body generates no heat.",
    )
    density: float | str | None = parameter(
        None,
        unit="kg/m^3",
        description="Density rho of the heat capacity term rho c_p dT/dt and, with a nonisothermal_flow coupling, of the heat that the flow carries, a constant or the name of a function. Default none: the equation has a heat capacity term only when density and specific_heat are both given, and a transient solve needs it.",
    )
    specific_heat: float | str | None = parameter(
        None,
        unit="J/(kg K)",
        description="Specific heat capacity c_p of the heat capacity term and, with a nonisothermal_flow coupling, of the heat that the flow carries. Default none: no heat capacity term, as for density. A nonisothermal_flow coupling needs it.",
    )
    advection_scheme: str | None = parameter(
        None,
        description="How the heat that a flow carries is discretised, with a nonisothermal_flow coupling: SUPG (streamline upwind Petrov-Galerkin, with either method), central (no stabilisation, which oscillates where convection dominates on a coarse mesh), or a face flux of the dual mesh control domain method (DMCDM): upwind (bounded, first order), linear_upwind (second order, not bounded) or limited (the upstream face value blended towards the central one by the van Albada limiter, bounded and second order). Default SUPG: it removes the oscillations of central where convection dominates, and it adds diffusion along the streamlines only. On a mesh that resolves the boundary layers, central has no stabilisation error.",
    )
    initial_condition: float | str = parameter(
        293.15,
        unit="K",
        description="Initial temperature, a constant or the name of a function, and the starting point of Newton's method in a steady solve. Default 293.15 K, room temperature, which is close to the state of most problems.",
    )
    block: Sequence[str] = parameter(
        (),
        description="Blocks where the equation holds. Default: every block, so that a body of one material needs no block names.",
    )

    def __post_init__(self):
        if (self.density is None) != (self.specific_heat is None):
            raise InputError(
                "heat_conduction: give both density and specific_heat for the heat capacity term, or neither."
            )
        # An abbreviation keeps its capitals (SUPG); the lower-case spelling is accepted.
        if isinstance(self.advection_scheme, str) and self.advection_scheme.lower() == "supg":
            self.advection_scheme = "SUPG"

    def variable_names(self) -> list[str]:
        return [self.temperature]

    def heat_flow(self, boundary: str) -> float:
        """The heat leaving the body through ``boundary``, in W, or in W/m in a
        plane problem: the integral of the heat flux along the outward normal, the
        quantity ``heat_flux`` of the boundary conditions.  Where the boundary
        prescribes the temperature it is the reaction there, conserved
        exactly (:meth:`~dualmesh.Problem.boundary_flux`)."""
        return self._problem.boundary_flux(self.temperature, boundary)

    def _variable_units(self) -> dict:
        return {self.temperature: "K"}

    def _initial_condition(self, variable: str):
        return self.initial_condition

    def _build(self, problem) -> None:
        T, restrict = self.temperature, self._restriction()
        conductivity = self.thermal_conductivity
        problem.add_kernel(
            "diffusion",
            variable=T,
            diffusivity="thermal_conductivity" if conductivity is None else conductivity,
            **restrict,
        )
        if self.heat_source is not None:
            problem.add_kernel("body_force", variable=T, value=self.heat_source, **restrict)
        if self.density is not None:
            problem.add_kernel(
                "time_derivative", variable=T, coefficient=self.heat_capacity(), **restrict
            )

    def heat_capacity(self):
        """The volumetric heat capacity rho c_p, a number or an expression, of the heat capacity
        term and of the heat that a flow carries."""
        numbers = (int, float)
        if isinstance(self.density, numbers) and isinstance(self.specific_heat, numbers):
            return float(self.density) * float(self.specific_heat)
        return f"({self.density})*({self.specific_heat})"

    def _manufactured_terms(self, fields, coordinates, dimension):
        if self.thermal_conductivity is None:
            raise InputError(
                f"{label(self.type_name, self.name)}: a manufactured solution needs the thermal_conductivity as a constant or an expression of x, y, z and t."
            )
        T = fields[self.temperature]
        k = sympy_expression(self.thermal_conductivity)
        source = 0
        if self.heat_source is not None:
            source -= sympy_expression(self.heat_source)
        if self.density is not None:
            t = coordinate_symbols()[3]
            source += (
                sympy_expression(self.density) * sympy_expression(self.specific_heat) * T.diff(t)
            )
        return {self.temperature: ([k * g for g in sympy_gradient(T)], source)}
