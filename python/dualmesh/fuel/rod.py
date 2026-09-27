# SPDX-License-Identifier: LGPL-2.1-or-later
r"""A fuel rod under a power history: its thermal and mechanical behaviour.

:class:`FuelRod` assembles from the input groups of
:mod:`dualmesh.fuel.specification` the problem a fuel performance code solves:

* the energy equation in the fuel and the cladding,
  :math:`\rho c_p \partial T / \partial t = \nabla \cdot (k \nabla T) + q'''`,
  with temperature- and burnup-dependent properties;
* heat transfer across the pellet-cladding gap (gas conduction, radiation and
  solid contact) and to the coolant (forced convection, with the coolant
  heating up along the rod, or a prescribed wall temperature);
* the quasi-static equilibrium of the fuel and the cladding,
  :math:`\nabla \cdot \sigma = 0`, with thermal expansion, densification,
  swelling and relocation of the fuel, thermal expansion and irradiation growth
  of the cladding, creep of both, the gas pressure inside the rod and the
  coolant pressure outside it, and frictionless contact when the gap closes;
* the fission gas: its diffusion to the grain boundaries, its release, and
  the pressure and composition of the rod's gas (ideal gas law).

Three rod models share the materials (:attr:`RodNumerics.model`):

``"axisymmetric"``
    one r-z problem of the whole rod;
``"three_dimensional"``
    one three-dimensional problem;
``"1.5d"``
    a stack of radial slices, each a one-dimensional problem in generalized
    plane strain, coupled through the coolant and the gas, as in TRANSURANUS
    and FRAPCON.  The axial strains of the fuel and the cladding of each slice
    are found at every step from the axial force balance: the fuel stack
    carries the gas pressure on its top, the cladding the pressure difference
    on its end plugs.

After every time step the gas pressure, the gas composition and the gaseous
swelling are updated explicitly from the solution at the end of the step;
they are held fixed within a step.
"""

from __future__ import annotations

import time as wall_clock
import warnings

import numpy as np

from .. import _core
from ..problem import Problem
from . import report
from .fields import TimeHistory, irradiation_fields, normalized_axial_profile
from .materials import MAXIMUM_TEMPERATURE, CladdingMaterial, FuelMaterial, RodContext
from .mesh import axisymmetric_rod_mesh, radial_slice_mesh, three_dimensional_rod_mesh
from .result import SECONDS_PER_DAY, RodResult
from .specification import (
    GASES,
    FillGas,
    ForcedConvection,
    PowerHistory,
    PrescribedCladdingTemperature,
    RodGeometry,
    RodModels,
    RodNumerics,
    RodOutput,
)
from .units import AVOGADRO, GAS_CONSTANT, BurnupConverter

#: The formulation of the stress objects and of the irradiation fields for
#: each rod model.
FORMULATIONS = {
    "axisymmetric": "axisymmetric",
    "three_dimensional": "three_dimensional",
    "1.5d": "axisymmetric_1d",
}


class FuelRod:
    """A fuel rod under a power history; see the module documentation.

    Every argument is one input group (a dataclass of
    :mod:`dualmesh.fuel.specification`); ``models``, ``numerics`` and
    ``output`` have defaults, which the input report marks as such::

        rod = FuelRod(geometry, fuel, cladding, fill_gas, coolant, power_history)
        result = rod.run()
    """

    def __init__(
        self,
        geometry: RodGeometry,
        fuel: FuelMaterial,
        cladding: CladdingMaterial,
        fill_gas: FillGas,
        coolant: ForcedConvection | PrescribedCladdingTemperature,
        power_history: PowerHistory,
        models: RodModels | None = None,
        numerics: RodNumerics | None = None,
        output: RodOutput | None = None,
    ):
        _check_type("geometry", geometry, RodGeometry)
        _check_type("fuel", fuel, FuelMaterial)
        _check_type("cladding", cladding, CladdingMaterial)
        _check_type("fill_gas", fill_gas, FillGas)
        _check_type("coolant", coolant, (ForcedConvection, PrescribedCladdingTemperature))
        _check_type("power_history", power_history, PowerHistory)
        self.geometry = geometry
        self.fuel = fuel
        self.cladding = cladding
        self.fill_gas = fill_gas
        self.coolant = coolant
        self._warned_boiling = False
        self.power_history = power_history
        self.models = models if models is not None else RodModels()
        self.numerics = numerics if numerics is not None else RodNumerics()
        self.output = output if output is not None else RodOutput()
        _check_type("models", self.models, RodModels)
        _check_type("numerics", self.numerics, RodNumerics)
        _check_type("output", self.output, RodOutput)
        # The cladding blocks of the mesh: the cladding, and a coating on it.
        coated = geometry.clad_coating_thickness > 0
        self.clad_blocks = ("clad", "coating") if coated else ("clad",)
        if coated != bool(getattr(cladding, "has_coating", False)):
            raise ValueError(
                "FuelRod: a coated cladding needs both RodGeometry.clad_coating_thickness > 0 and "
                "a cladding with a coating (CoatedCladding); one of the two is missing."
            )
        # The models with the fuel-dependent defaults filled in.
        self.active_models = self.models.resolved(fuel)
        self.formulation = FORMULATIONS[self.numerics.model]
        if (
            self.numerics.model == "three_dimensional"
            and self.numerics.mesh.num_fuel_core_divisions % 2
        ):
            raise ValueError(
                "RodMesh: the three-dimensional model needs an even num_fuel_core_divisions (the "
                "rigid-body restraints sit on the x and y axes)."
            )

        # ---- burnup, and the power history in time ----
        self.converter = BurnupConverter(fuel.energy_per_fission, fuel.heavy_metal_molar_mass)
        self.heavy_metal_atom_density = fuel.heavy_metal_atom_density()
        # FIMA per unit energy per unit length of rod: 1 / (E_f n_HM A).
        self.fima_per_joule_per_metre = 1.0 / (
            fuel.energy_per_fission * self.heavy_metal_atom_density * geometry.fuel_cross_section
        )
        axial = normalized_axial_profile(power_history.axial_profile, geometry.fuel_stack_height)
        if power_history.time is not None:
            self.history = TimeHistory(
                power_history.time,
                power_history.linear_heat_rate,
                axial,
                power_history.radial_profile,
            )
        else:
            burnup = self.converter.to_fima(power_history.burnup, power_history.burnup_unit)
            if burnup[0] != 0.0:
                raise ValueError(
                    "PowerHistory: a history in burnup must start at zero burnup (fresh fuel)."
                )
            self.history = TimeHistory.from_burnup(
                burnup,
                power_history.linear_heat_rate,
                self.fima_per_joule_per_metre,
                axial,
                power_history.radial_profile,
            )

        # ---- the rod's gas ----
        self._gas_pressure = _core.SettableFunction(fill_gas.pressure)
        self._coolant_pressure = _core.SettableFunction(coolant.pressure)
        self._gas_fractions = {
            gas: _core.SettableFunction(float(fill_gas.composition.get(gas, 0.0))) for gas in GASES
        }
        self.fill_amount = (
            fill_gas.pressure * self.cold_free_volume() / (GAS_CONSTANT * fill_gas.temperature)
        )
        self.released_amount = {"xenon": 0.0, "krypton": 0.0}
        self.plenum_temperature = fill_gas.temperature
        self._axial_jacobians: dict = {}
        self.axial_balance_solves: list[int] = []
        self.wall_time = 0.0
        self.time = float(self.history.time[0])
        self._build()

    # ------------------------------------------------------------------
    # derived quantities
    # ------------------------------------------------------------------
    def cold_free_volume(self) -> float:
        """The free gas volume of the cold rod, m^3: the gap, the bore of
        annular pellets, the plenum volume and the meshed plenum height."""
        g = self.geometry
        gap = np.pi * (g.clad_inner_radius**2 - g.pellet_outer_radius**2) * g.fuel_stack_height
        bore = np.pi * g.pellet_inner_radius**2 * g.fuel_stack_height
        meshed_plenum = np.pi * g.clad_inner_radius**2 * g.plenum_height
        return gap + bore + self.fill_gas.plenum_volume + meshed_plenum

    def gas_amount(self) -> float:
        """Amount of gas in the rod, mol: the fill gas and the released gas."""
        return self.fill_amount + sum(self.released_amount.values())

    def derived_quantities(self) -> list[tuple]:
        """Quantities computed from the input, for the input report."""
        g, fuel = self.geometry, self.fuel
        heavy_metal_mass = (
            self.heavy_metal_atom_density
            * g.fuel_cross_section
            * fuel.heavy_metal_molar_mass
            / AVOGADRO
        )
        end_fima = float(self.history.energy_at(self.history.time[-1])[0])
        end_fima *= self.fima_per_joule_per_metre
        rows = [
            ("radial gap (cold)", g.radial_gap * 1e6, "um"),
            ("heavy-metal atom density", self.heavy_metal_atom_density, "1/m^3"),
            ("heavy-metal mass per length", heavy_metal_mass, "kg/m"),
            ("1 FIMA", self.converter.mwd_per_kg_per_fima, "MWd/kgHM"),
            ("cold free volume", self.cold_free_volume() * 1e6, "cm^3"),
            ("fill gas amount", self.fill_amount, "mol"),
            ("history end time", self.history.time[-1] / SECONDS_PER_DAY, "d"),
            (
                "history end burnup",
                float(self.converter.from_fima(end_fima, self.output.burnup_unit)),
                self.output.burnup_unit,
            ),
        ]
        for name in ("relocation", "densification", "gaseous_swelling", "fission_gas_release"):
            rows.append(
                (f"{name} (resolved for this fuel)", str(getattr(self.active_models, name)), "")
            )
        if isinstance(self.coolant, ForcedConvection):
            c, radius = self.coolant, g.clad_outer_radius
            rows += [
                ("coolant flow area", c.flow_area(radius) * 1e6, "mm^2"),
                ("hydraulic diameter", c.hydraulic_diameter(radius) * 1e3, "mm"),
                (
                    "heat transfer coefficient at the inlet",
                    c.heat_transfer_coefficient_for(radius),
                    "W/(m^2 K)",
                ),
                ("coolant saturation temperature", c.saturation_temperature(), "K"),
            ]
        return rows

    # ------------------------------------------------------------------
    # the coolant side
    # ------------------------------------------------------------------
    def _cumulative_axial_shape(self, z):
        """The integral of the axial profile from the bottom of the stack up
        to z, m."""
        H = self.geometry.fuel_stack_height
        grid = np.linspace(0.0, H, 401)
        shape = self.history.axial(grid, H)
        cumulative = np.concatenate(
            [[0.0], np.cumsum(0.5 * (shape[1:] + shape[:-1]) * np.diff(grid))]
        )
        return np.interp(z, grid, cumulative)

    def coolant_temperature(self, z, t) -> np.ndarray:
        """The coolant bulk temperature (forced convection), or the prescribed
        cladding outer temperature, at axial positions z and time t, K."""
        z = np.atleast_1d(np.asarray(z, dtype=float))
        q = float(self.history.linear_heat_rate_at(t))
        if isinstance(self.coolant, ForcedConvection):
            return self.coolant.bulk_temperature(
                q * self._cumulative_axial_shape(z), self.geometry.clad_outer_radius
            )
        local = q * self.history.axial(
            np.minimum(z, self.geometry.fuel_stack_height), self.geometry.fuel_stack_height
        )
        return np.array([self.coolant.value(qz, t) for qz in local])

    def _coolant_table(self, slice_axial_position: float, quantity: str = "temperature"):
        """The coolant (or wall) temperature T(z, t), or the heat transfer
        coefficient h(z, t) from the local bulk temperature, as a solver
        function."""
        h = self.history
        t = np.unique(
            np.concatenate(
                [np.linspace(h.time[i], h.time[i + 1], 9) for i in range(len(h.time) - 1)]
            )
        )
        if self.formulation == "axisymmetric_1d":
            z = np.array([slice_axial_position])
        else:
            top = self.geometry.fuel_stack_height + self.geometry.plenum_height
            z = np.linspace(0.0, top, 41)
        values = np.array([self.coolant_temperature(z, tk) for tk in t])  # [t, z]
        if quantity == "heat_transfer_coefficient":
            values = self.coolant.heat_transfer_coefficient_for(
                self.geometry.clad_outer_radius, values
            )
        data = np.repeat(values[:, :, None], 2, axis=2)
        return _core.CylinderTableFunction(
            [0.0, 1.0], list(z), list(t), list(data.ravel()), self.formulation, slice_axial_position
        )

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------
    def _build(self):
        g, mesh = self.geometry, self.numerics.mesh
        if self.numerics.model == "axisymmetric":
            self.problems = [self._make_problem(axisymmetric_rod_mesh(g, mesh), 0.0)]
            n = mesh.num_axial_elements
        elif self.numerics.model == "three_dimensional":
            self.problems = [self._make_problem(three_dimensional_rod_mesh(g, mesh), 0.0)]
            n = mesh.num_axial_elements
        else:
            n = mesh.num_axial_slices
            positions = (np.arange(n) + 0.5) * g.fuel_stack_height / n
            self.problems = [self._make_problem(radial_slice_mesh(g, mesh), z) for z in positions]
        self.axial_positions = (np.arange(n) + 0.5) * g.fuel_stack_height / n
        self._setup_fission_gas()

    def _burnup_parameters(self, burnup_field: str = "burnup") -> dict:
        fuel = self.fuel
        return dict(
            burnup=burnup_field,
            burnup_unit="FIMA",
            energy_per_fission=fuel.energy_per_fission,
            heavy_metal_molar_mass=fuel.heavy_metal_molar_mass,
        )

    def _context(self) -> RodContext:
        """What the materials are told about this rod."""
        return RodContext(
            formulation=self.formulation,
            models=self.active_models,
            geometry=self.geometry,
            stress_free_temperature=self.fill_gas.temperature,
            burnup=self._burnup_parameters(),
            mwd_per_kg_per_fima=self.converter.mwd_per_kg_per_fima,
        )

    def _make_problem(self, mesh, slice_axial_position: float) -> Problem:
        g, fuel, formulation = self.geometry, self.fuel, self.formulation
        coordinates = "cartesian" if formulation == "three_dimensional" else "axisymmetric"
        p = Problem(mesh, method=self.numerics.method, coordinates=coordinates)
        # The highest temperature each element has reached, K: a history
        # field for models that must not reverse when the rod cools
        # (densification, swelling at the irradiation temperature).
        p.set_element_field(MAXIMUM_TEMPERATURE, np.zeros(mesh.num_elements))
        fields = irradiation_fields(
            self.history,
            g,
            self.heavy_metal_atom_density,
            fuel.energy_per_fission,
            self.power_history.fast_neutron_flux_per_linear_heat_rate,
            formulation,
            slice_axial_position,
        )
        p._fields = fields
        for name, function in fields.items():
            p.add_function(name, function)
        p.add_function("coolant_temperature", self._coolant_table(slice_axial_position))
        if isinstance(self.coolant, ForcedConvection):
            p.add_function(
                "coolant_heat_transfer_coefficient",
                self._coolant_table(slice_axial_position, "heat_transfer_coefficient"),
            )
        p.add_function("gas_pressure", self._gas_pressure)
        p.add_function("coolant_pressure", self._coolant_pressure)
        for gas, function in self._gas_fractions.items():
            p.add_function(f"{gas}_fraction", function)
        start = float(self.coolant_temperature([0.5 * g.fuel_stack_height], self.time)[0])
        p.add_variable("temperature", initial_condition=start)

        # ---- heat ----
        context = self._context()
        fuel.add_thermal_material(p, "fuel", context)
        for block in self.clad_blocks:
            self.cladding.add_thermal_material(p, block, context)
        # The heat equation on the deformed body needs the deformation
        # gradient of the mechanics (see RodNumerics.heat_conduction_configuration).
        deformed = (
            self.active_models.mechanics
            and self.numerics.heat_conduction_configuration == "deformed"
        )
        on_deformed = {"deformation_gradient_property": "deformation_gradient"} if deformed else {}
        p.add_kernel(
            "heat_conduction",
            "heat_conduction",
            variable="temperature",
            thermal_conductivity_property="thermal_conductivity",
            **on_deformed,
        )
        p.add_kernel(
            "heat_conduction_time_derivative",
            "heat_capacity",
            variable="temperature",
            density_property="density",
            specific_heat_property="specific_heat",
        )
        p.add_kernel(
            "heat_source",
            "fission_heat",
            variable="temperature",
            heat_source="power_density",
            block=["fuel"],
        )
        displacements = self._add_mechanics(p) if self.active_models.mechanics else []
        p._displacements = displacements
        gap = dict(
            variable="temperature",
            boundary=["fuel_outer"],
            secondary_boundary=["clad_inner"],
            gas_pressure="gas_pressure",
            primary_emissivity=fuel.emissivity,
            secondary_emissivity=self.cladding.emissivity,
            primary_roughness=fuel.surface_roughness,
            secondary_roughness=self.cladding.surface_roughness,
            **{f"{gas}_fraction": f"{gas}_fraction" for gas in GASES},
        )
        hardness = self.cladding.meyer_hardness
        if isinstance(hardness, str):
            gap["meyer_hardness_model"] = "zircaloy"
        else:
            gap["meyer_hardness"] = float(hardness)
        if displacements:
            gap.update(displacements=displacements, contact_penalty=self.numerics.contact_penalty)
        p.add_boundary_condition("gas_gap_heat_transfer", "gap_heat_transfer", **gap, **on_deformed)
        if isinstance(self.coolant, ForcedConvection):
            p.add_boundary_condition(
                "convective_heat_flux_boundary_condition",
                "coolant",
                variable="temperature",
                boundary=["clad_outer"],
                heat_transfer_coefficient="coolant_heat_transfer_coefficient",
                ambient_temperature="coolant_temperature",
                **on_deformed,
            )
        else:
            p.add_boundary_condition(
                "Dirichlet_boundary_condition",
                "clad_outer_temperature",
                variable="temperature",
                boundary=["clad_outer"],
                value="coolant_temperature",
            )
        return p

    def _add_mechanics(self, p) -> list[str]:
        g, fuel, models = self.geometry, self.fuel, self.active_models
        formulation = self.formulation
        displacements = {
            "axisymmetric": ["disp_r", "disp_z"],
            "three_dimensional": ["disp_x", "disp_y", "disp_z"],
            "axisymmetric_1d": ["disp_r"],
        }[formulation]
        for d in displacements:
            p.add_variable(d)
        # ---- elasticity, eigenstrains and stresses, from the materials ----
        context = self._context()
        fuel.add_elasticity(p, "fuel", context)
        eigenstrains = {"fuel": fuel.add_eigenstrains(p, "fuel", context)}
        creep = {"fuel": fuel.creep_parameters(context) if models.creep else {}}
        for block in self.clad_blocks:
            self.cladding.add_elasticity(p, block, context)
            eigenstrains[block] = self.cladding.add_eigenstrains(p, block, context)
            creep[block] = self.cladding.creep_parameters(context, block) if models.creep else {}
        axial = {}
        if formulation == "axisymmetric_1d":
            # One axial strain for the fuel and one for the cladding, which a
            # coating bonded to it shares.
            p._axial_strain = {b: _core.SettableFunction(0.0) for b in ("fuel", "clad")}
            for b, function in p._axial_strain.items():
                p.add_function(f"{b}_axial_strain", function)
            axial = {
                block: {
                    "axial_strain": "fuel_axial_strain" if block == "fuel" else "clad_axial_strain"
                }
                for block in ("fuel", *self.clad_blocks)
            }
        finite = self.numerics.strain == "finite"
        for block in ("fuel", *self.clad_blocks):
            p.add_material(
                "finite_strain_stress" if finite else "small_strain_stress",
                f"{block}_stress",
                block=[block],
                displacements=displacements,
                formulation=formulation,
                eigenstrain_names=eigenstrains[block],
                **axial.get(block, {}),
                **creep[block],
            )
        # ---- equilibrium, contact and pressures ----
        gas_surfaces = ["fuel_outer", "clad_inner"]
        if g.pellet_inner_radius > 0:
            gas_surfaces.append("fuel_inner")
        if formulation != "axisymmetric_1d":
            gas_surfaces.append("fuel_top")
        follower = {"deformation_gradient_property": "deformation_gradient"} if finite else {}
        for i, d in enumerate(displacements):
            p.add_kernel(
                "stress_divergence",
                f"equilibrium_{d}",
                variable=d,
                component=i,
                stress_property="first_piola_kirchhoff_stress" if finite else "stress",
            )
            p.add_boundary_condition(
                "gap_contact",
                f"contact_{d}",
                variable=d,
                boundary=["fuel_outer"],
                secondary_boundary=["clad_inner"],
                displacements=displacements,
                component=i,
                penalty=self.numerics.contact_penalty,
                **follower,
            )
            p.add_boundary_condition(
                "pressure_boundary_condition",
                f"gas_pressure_{d}",
                variable=d,
                component=i,
                boundary=gas_surfaces,
                pressure="gas_pressure",
                **follower,
            )
            p.add_boundary_condition(
                "pressure_boundary_condition",
                f"coolant_pressure_{d}",
                variable=d,
                component=i,
                boundary=["clad_outer"],
                pressure="coolant_pressure",
                **follower,
            )
        if formulation in ("axisymmetric", "axisymmetric_1d") and g.pellet_inner_radius == 0:
            p.add_boundary_condition(
                "Dirichlet_boundary_condition",
                "axis_disp_r",
                variable="disp_r",
                boundary=["axis"],
                value=0.0,
            )
        if formulation != "axisymmetric_1d":
            p.add_boundary_condition(
                "Dirichlet_boundary_condition",
                "bottom_disp_z",
                variable="disp_z",
                boundary=["fuel_bottom", "clad_bottom"],
                value=0.0,
            )
            self._add_end_plug_load(p)
        if formulation == "three_dimensional":
            self._hold_rigid_body_motion(p)
        return displacements

    def _hold_rigid_body_motion(self, p):
        """Hold the two bodies of the three-dimensional model against rigid
        motion in the plane of the cross-section without restraining their
        expansion: each body at three points of a surface, on the +x and -x
        axes (y component) and on the +y axis (x component), where a uniform
        radial expansion has no such component."""
        g = self.geometry
        mesh = p._mesh
        tolerance = 1e-9 * g.clad_outer_radius

        def at(x0, y0):
            return lambda x, y, z: abs(x - x0) < tolerance and abs(y - y0) < tolerance

        fuel_radius = g.pellet_inner_radius if g.pellet_inner_radius > 0 else g.pellet_outer_radius
        hold_x, hold_y = [], []
        for body, radius in (("fuel", fuel_radius), ("clad", g.clad_outer_radius)):
            mesh.add_nodeset_by_predicate(f"{body}_plus_x", at(radius, 0.0))
            mesh.add_nodeset_by_predicate(f"{body}_minus_x", at(-radius, 0.0))
            mesh.add_nodeset_by_predicate(f"{body}_plus_y", at(0.0, radius))
            hold_y += [f"{body}_plus_x", f"{body}_minus_x"]
            hold_x += [f"{body}_plus_y"]
        p.add_boundary_condition(
            "Dirichlet_boundary_condition",
            "hold_disp_x",
            variable="disp_x",
            boundary=hold_x,
            value=0.0,
        )
        p.add_boundary_condition(
            "Dirichlet_boundary_condition",
            "hold_disp_y",
            variable="disp_y",
            boundary=hold_y,
            value=0.0,
        )

    def _end_plug_stress(self) -> float:
        """The axial stress that the gas and the coolant pressures put on the
        cladding through its end plugs, (p b^2 - p_c c^2) / (c^2 - b^2)."""
        g = self.geometry
        b2, c2 = g.clad_inner_radius**2, g.clad_outer_radius**2
        return (self._gas_pressure.get() * b2 - self._coolant_pressure.get() * c2) / (c2 - b2)

    def _add_end_plug_load(self, p):
        p._end_plug_stress = _core.SettableFunction(self._end_plug_stress())
        p.add_function("end_plug_stress", p._end_plug_stress)
        p.add_boundary_condition(
            "traction_boundary_condition",
            "end_plug",
            variable="disp_z",
            boundary=["clad_top"],
            traction="end_plug_stress",
        )

    # ------------------------------------------------------------------
    # fission gas and gaseous swelling
    # ------------------------------------------------------------------
    def _fuel_elements(self, p):
        """Indices, volumes (m^3), centroids and nodes of the fuel elements of
        a problem: the volume of the body of revolution for the axisymmetric
        model (Pappus: 2 pi times the centroid radius times the area, exact
        for straight-sided elements), of a slice for the 1.5D model."""
        mesh = p._mesh
        measures = np.asarray(mesh.element_measures())
        ids = np.array([e for e in range(mesh.num_elements) if mesh.element_block(e) == 0])
        centroids = np.array([mesh.element_centroid(e) for e in ids], dtype=float)
        if self.formulation == "axisymmetric":
            volumes = 2.0 * np.pi * centroids[:, 0] * measures[ids]
        elif self.formulation == "axisymmetric_1d":
            dz = self.geometry.fuel_stack_height / len(self.problems)
            ends = np.array([[mesh.node(mesh.element_nodes(e)[k])[0] for k in (0, 1)] for e in ids])
            volumes = np.pi * np.abs(ends[:, 1] ** 2 - ends[:, 0] ** 2) * dz
        else:
            volumes = measures[ids]
        nodes = None
        if not p.is_cell_centered:
            nodes = np.array([mesh.element_nodes(e) for e in ids], dtype=int)
        return ids, volumes, centroids, nodes

    def _setup_fission_gas(self):
        self._fuel_regions = []
        context = self._context()
        for p in self.problems:
            ids, volumes, centroids, nodes = self._fuel_elements(p)
            model = self.fuel.fission_gas_model(len(ids), context)
            if hasattr(model, "mwd_per_kg_per_fima"):
                model.mwd_per_kg_per_fima = self.converter.mwd_per_kg_per_fima
            self._fuel_regions.append((p, ids, volumes, centroids, nodes, model))

    @staticmethod
    def _element_temperatures(p, ids, nodes):
        temperature = np.asarray(p.values("temperature"))
        if nodes is None:
            return temperature[ids]
        return temperature[nodes].mean(axis=1)

    def _update_maximum_temperature(self):
        """Raise the element field of the highest temperature reached to the
        current element temperatures (the mean of the nodal values, or the
        cell value of a cell-centred method)."""
        for p in self.problems:
            mesh = p._mesh
            temperature = np.asarray(p.values("temperature"))
            if p.is_cell_centered:
                temperature = temperature[: mesh.num_elements]
            else:
                temperature = np.array(
                    [temperature[mesh.element_nodes(e)].mean() for e in range(mesh.num_elements)]
                )
            field = np.maximum(np.asarray(p.element_field(MAXIMUM_TEMPERATURE)), temperature)
            p.set_element_field(MAXIMUM_TEMPERATURE, field)

    @staticmethod
    def _at_centroids(function, centroids, t):
        return np.array([function(c[0], c[1], c[2], t) for c in centroids])

    def _advance_fission_gas_and_swelling(self, t_old: float, t_new: float):
        dt = t_new - t_old
        if dt <= 0:
            return
        swelling = self.active_models.gaseous_swelling and self.active_models.mechanics
        released_atoms, xenon_fraction = 0.0, 0.0
        for p, ids, volumes, centroids, nodes, model in self._fuel_regions:
            if model is None and not swelling:
                continue
            fields = p._fields
            temperature = self._element_temperatures(p, ids, nodes)
            burnup_new = self._at_centroids(fields.burnup, centroids, t_new)
            if model is not None:
                fission_rate = self._at_centroids(fields.fission_rate, centroids, t_new)
                atoms = model.advance(
                    dt, temperature, fission_rate, self._gas_pressure.get(), burnup_new
                )
                released_atoms += float(np.sum(atoms * volumes))
                xenon_fraction = model.xenon_fraction
            if swelling:
                field = p.element_field("gaseous_swelling")
                if getattr(self.fuel, "gaseous_swelling_model", "matpro") == "fission_gas":
                    # The volume of the gas model's bubbles (set, not added);
                    # without a fission gas model there are no bubbles.
                    if hasattr(model, "gaseous_swelling"):
                        field[ids] = model.gaseous_swelling()
                else:
                    burnup_old = self._at_centroids(fields.burnup, centroids, t_old)
                    field[ids] += self.fuel.gaseous_swelling_increment(
                        temperature, burnup_old, np.maximum(burnup_new - burnup_old, 0.0)
                    )
                p.set_element_field("gaseous_swelling", field)
        released = released_atoms / AVOGADRO
        self.released_amount["xenon"] += xenon_fraction * released
        self.released_amount["krypton"] += (1.0 - xenon_fraction) * released
        total = self.gas_amount()
        for gas, function in self._gas_fractions.items():
            fill = self.fill_amount * self.fill_gas.composition.get(gas, 0.0)
            function.set((fill + self.released_amount.get(gas, 0.0)) / total)

    def fission_gas_release(self) -> float:
        """The fraction of the fission gas produced so far that has been
        released."""
        produced = released = 0.0
        for _, _, volumes, _, _, model in self._fuel_regions:
            if model is not None:
                produced += float(np.sum(model.produced * volumes))
                released += float(np.sum(model.released * volumes))
        return released / produced if produced > 0 else 0.0

    # ------------------------------------------------------------------
    # sampling the solution
    # ------------------------------------------------------------------
    def _point(self, r, z):
        """The point at radius r and axial position z in the coordinates of
        the rod's problems (on the +x axis in three dimensions)."""
        if self.formulation == "axisymmetric_1d":
            return [r, 0.0, 0.0]
        if self.formulation == "axisymmetric":
            return [r, z, 0.0]
        return [r, 0.0, z]

    def _sample(self, variable: str, radius: float) -> np.ndarray:
        """A variable at one radius, at every axial sampling position."""
        if self.formulation == "axisymmetric_1d":
            point = [self._point(radius, 0.0)]
            return np.array([p.sample(variable, point)[0] for p in self.problems], dtype=float)
        points = [self._point(radius, z) for z in self.axial_positions]
        return np.asarray(self.problems[0].sample(variable, points), dtype=float)

    def _state_along_the_rod(self) -> dict:
        """Temperatures and radial displacements at the surfaces, along the
        rod.  A surface point is taken just inside the body it belongs to."""
        g = self.geometry
        inside, outside = 1.0 - 1e-9, 1.0 + 1e-9
        inner = g.pellet_inner_radius * outside if g.pellet_inner_radius > 0 else 0.0
        state = {
            "fuel_centerline_temperature": self._sample("temperature", inner),
            "fuel_surface_temperature": self._sample("temperature", g.pellet_outer_radius * inside),
            "clad_inner_temperature": self._sample("temperature", g.clad_inner_radius * outside),
            "clad_outer_temperature": self._sample("temperature", g.clad_outer_radius * inside),
        }
        zero = np.zeros(len(self.axial_positions))
        if self.active_models.mechanics:
            d = self.problems[0]._displacements[0]
            u_fuel = self._sample(d, g.pellet_outer_radius * inside)
            u_clad_inner = self._sample(d, g.clad_inner_radius * outside)
            u_clad_outer = self._sample(d, g.clad_outer_radius * inside)
            u_bore = self._sample(d, inner) if g.pellet_inner_radius > 0 else zero
        else:
            u_fuel = u_clad_inner = u_clad_outer = u_bore = zero
        state["fuel_surface_displacement"] = u_fuel
        state["clad_outer_displacement"] = u_clad_outer
        state["gap_width"] = g.clad_inner_radius + u_clad_inner - g.pellet_outer_radius - u_fuel
        state["clad_hoop_strain"] = u_clad_outer / g.clad_outer_radius
        state["bore_displacement"] = u_bore
        return state

    # ------------------------------------------------------------------
    # the rod's gas
    # ------------------------------------------------------------------
    def _update_gas_pressure(self, state: dict):
        """The ideal gas law over the free volumes at their temperatures,
        :math:`p = n R / \\sum_i V_i / T_i`: the gap of each axial segment at
        the mean of the pellet surface and cladding inner temperatures, the
        bore of annular pellets at the centerline temperature, the cracks
        opened by relocation at the mean of the centerline and surface
        temperatures, and the plenum at the coolant temperature at the top
        of the stack plus ``plenum_temperature_rise``."""
        g = self.geometry
        dz = g.fuel_stack_height / len(self.axial_positions)
        a = g.pellet_outer_radius + state["fuel_surface_displacement"]
        width = np.maximum(state["gap_width"], 0.0)
        gap_volumes = np.pi * ((a + width) ** 2 - a**2) * dz
        gap_temperatures = 0.5 * (
            state["fuel_surface_temperature"] + state["clad_inner_temperature"]
        )
        weighted = float(np.sum(gap_volumes / np.maximum(gap_temperatures, 1.0)))
        if g.pellet_inner_radius > 0:
            bore = np.pi * (g.pellet_inner_radius + state["bore_displacement"]) ** 2 * dz
            weighted += float(np.sum(bore / np.maximum(state["fuel_centerline_temperature"], 1.0)))
        if self.active_models.mechanics and self.active_models.relocation:
            # Relocation moves the fragments outwards without changing their
            # volume: the space it opens between them (cracks) holds gas.  Its
            # volume is the transverse relocation strain times the pellet
            # area, twice (r and theta); its gas is taken at the mean of the
            # centre (or bore) and surface temperatures of the pellet.
            strain = self._relocation_strain_along_the_rod()
            area = np.pi * (g.pellet_outer_radius**2 - g.pellet_inner_radius**2)
            cracks = 2.0 * strain * area * dz
            crack_temperatures = 0.5 * (
                state["fuel_centerline_temperature"] + state["fuel_surface_temperature"]
            )
            weighted += float(np.sum(cracks / np.maximum(crack_temperatures, 1.0)))
        plenum = self.fill_gas.plenum_volume + np.pi * g.clad_inner_radius**2 * g.plenum_height
        top = float(self.coolant_temperature([g.fuel_stack_height], self.time)[0])
        self.plenum_temperature = top + self.active_models.plenum_temperature_rise
        weighted += plenum / self.plenum_temperature
        if weighted > 0:
            self._gas_pressure.set(self.gas_amount() * GAS_CONSTANT / weighted)
        for p in self.problems:
            if hasattr(p, "_end_plug_stress"):
                p._end_plug_stress.set(self._end_plug_stress())

    def _relocation_strain_along_the_rod(self) -> np.ndarray:
        """The transverse relocation strain at the axial positions, from the
        same correlation as the relocation eigenstrain of the fuel."""
        g = self.geometry
        t = self.time
        shape = self.history.axial(self.axial_positions, g.fuel_stack_height)
        q = float(self.history.linear_heat_rate_at(t)) * shape
        fima = float(self.history.energy_at(t)[0]) * self.fima_per_joule_per_metre * shape
        burnup = np.asarray(self.converter.from_fima(fima, "MWd/kgHM"), dtype=float)
        return np.asarray(
            _core.fuel.uo2_relocation_strain(
                q, burnup, 2.0 * g.pellet_outer_radius, 2.0 * g.radial_gap
            ),
            dtype=float,
        )

    # ------------------------------------------------------------------
    # the axial force balance of the 1.5D model
    # ------------------------------------------------------------------
    def _axial_force_residual(self, p) -> np.ndarray:
        """Axial force carried by the fuel and by the cladding of a radial
        slice (the integral of sigma_zz over each cross-section) minus the
        force the pressures put on them: the gas on the top of the fuel
        column, the gas and the coolant on the end plugs."""
        g = self.geometry
        # The axial force through the undeformed cross-section: the Cauchy
        # sigma_zz at small strain, the nominal P_zZ at finite strain.
        if self.numerics.strain == "finite":
            axial = np.asarray(p.property_at_centroids("first_piola_kirchhoff_stress"))[:, 4]
        else:
            axial = np.asarray(p.property_at_centroids("stress"))[:, 1]
        mesh = p._mesh
        forces = np.zeros(2)
        for e in range(mesh.num_elements):
            nodes = mesh.element_nodes(e)
            r0, r1 = mesh.node(nodes[0])[0], mesh.node(nodes[1])[0]
            body = 0 if mesh.element_block(e) == 0 else 1
            forces[body] += axial[e] * np.pi * abs(r1**2 - r0**2)
        pressure, coolant = self._gas_pressure.get(), self._coolant_pressure.get()
        target = np.array(
            [
                -pressure * np.pi * (g.pellet_outer_radius**2 - g.pellet_inner_radius**2),
                pressure * np.pi * g.clad_inner_radius**2
                - coolant * np.pi * g.clad_outer_radius**2,
            ]
        )
        return forces - target

    def _solve_axial_balance(self, p, solve) -> int:
        """Find the axial strains of the fuel and the cladding of one slice
        that balance the axial forces, F(eps) = 0, by Newton's method with a
        Jacobian kept from one step to the next and updated by Broyden's
        rank-one formula (C. G. Broyden, Math. Comp. 19 (1965) 577-593).

        The axial stiffness dF/deps changes little between steps (it is
        essentially E A of each body, plus their coupling when the gap is
        closed), so a step usually costs two or three slice solves instead of
        the thirteen of a finite-difference Newton iteration.  The Jacobian is
        rebuilt by finite differences on the first step of a slice and
        whenever the iteration stalls.  Returns the Newton iterations of the
        slice solves."""
        tolerance, max_solves, perturbation = 1e-8, 12, 1e-6
        iterations = 0

        def residual_at(eps):
            nonlocal iterations
            for function, value in zip(p._axial_strain.values(), eps):
                function.set(float(value))
            iterations += solve()
            return self._axial_force_residual(p)

        def finite_difference_jacobian(eps, residual):
            J = np.zeros((2, 2))
            for k in range(2):
                trial = eps.copy()
                trial[k] += perturbation
                J[:, k] = (residual_at(trial) - residual) / perturbation
            return J

        eps = np.array([f.get() for f in p._axial_strain.values()])
        residual = residual_at(eps)
        solves = 1
        key = id(p)
        J = self._axial_jacobians.get(key)
        if J is None:
            J = finite_difference_jacobian(eps, residual)
            solves += 2
        refreshed = False
        while solves < max_solves:
            step = np.linalg.solve(J, -residual)
            # A step below the tolerance changes the stresses by less than
            # E times it (2 kPa for 1e-8), so the current solution is kept.
            if np.all(np.abs(step) < tolerance):
                break
            eps = eps + step
            new_residual = residual_at(eps)
            solves += 1
            J = J + np.outer(new_residual - residual - J @ step, step) / (step @ step)
            if np.linalg.norm(new_residual) > 0.9 * np.linalg.norm(residual) and not refreshed:
                J = finite_difference_jacobian(eps, new_residual)
                solves += 2
                refreshed = True
            residual = new_residual
        self._axial_jacobians[key] = J
        self.axial_balance_solves.append(solves)
        return iterations

    # ------------------------------------------------------------------
    # running
    # ------------------------------------------------------------------
    def _with_axial_balance(self) -> bool:
        return self.formulation == "axisymmetric_1d" and self.active_models.mechanics

    def _solve_steady(self, p) -> int:
        options = {"line_search": "backtracking", **self.numerics.solver_options}
        if not self._with_axial_balance():
            return int(p.solve(**options).total_iterations)
        # Every trial of the axial balance starts from the same history.
        p.initialize()
        state = p._problem.save_state()

        def solve():
            p._problem.restore_state(state)
            return int(p.solve(**options).total_iterations)

        return self._solve_axial_balance(p, solve)

    def _solve_step(self, p, t_old: float, t_new: float) -> int:
        options = {"line_search": "backtracking", **self.numerics.solver_options}
        dt = t_new - t_old
        if not self._with_axial_balance():
            state = p._problem.save_state()
            start = {v: np.asarray(p.values(v)).copy() for v in ["temperature"] + p._displacements}
            try:
                result = p.solve_transient(start_time=t_old, end_time=t_new, dt=dt, **options)
            except RuntimeError:
                if "line_search" in self.numerics.solver_options:
                    raise
                # The backtracking line search can stall where the pellet
                # meets the cladding during a fast power change (the contact
                # and gap conductance have a kink at zero gap, and the full
                # Newton step crosses it while a shortened one does not
                # reduce the residual).  The step is repeated with full
                # Newton steps.
                for v, values in start.items():
                    p.set_values(v, values)
                p._problem.restore_state(state)
                options["line_search"] = "none"
                result = p.solve_transient(start_time=t_old, end_time=t_new, dt=dt, **options)
            return int(result.total_iterations)
        variables = ["temperature"] + p._displacements
        start = {v: np.asarray(p.values(v)).copy() for v in variables}
        state = p._problem.save_state()

        def solve():
            for v, values in start.items():
                p.set_values(v, values)
            p._problem.restore_state(state)
            result = p.solve_transient(start_time=t_old, end_time=t_new, dt=dt, **options)
            return int(result.total_iterations)

        return self._solve_axial_balance(p, solve)

    def _step_boundaries(self, times: np.ndarray) -> np.ndarray:
        """The output times and the points of the power history that the
        steps must land on (``RodNumerics.power_history_tolerance``)."""
        tolerance = self.numerics.power_history_tolerance
        if tolerance is None:
            return times
        h = self.history
        keep = simplified_points(h.time, h.linear_heat_rate, tolerance * np.max(h.linear_heat_rate))
        inside = h.time[keep][(h.time[keep] > times[0]) & (h.time[keep] < times[-1])]
        merged = np.union1d(times, inside)
        # Drop history points within a microsecond of an output time.
        close = np.isin(merged, times) | (
            np.min(np.abs(merged[:, None] - times[None, :]), axis=1) > 1e-6
        )
        return merged[close]

    def _output_times(self) -> np.ndarray:
        if self.output.output_times is None:
            return np.asarray(self.history.time, dtype=float)
        times = np.asarray(self.output.output_times, dtype=float)
        if len(times) < 1 or np.any(np.diff(times) <= 0):
            raise ValueError("RodOutput: output_times must increase strictly.")
        return times

    def _record(self, result: RodResult, iterations: int):
        state = self._state_along_the_rod()
        self._update_gas_pressure(state)
        t = self.time
        q = float(self.history.linear_heat_rate_at(t))
        shape = self.history.axial(self.axial_positions, self.geometry.fuel_stack_height)
        average_fima = float(self.history.energy_at(t)[0]) * self.fima_per_joule_per_metre
        unit = self.output.burnup_unit
        rod = result.rod
        rod["time"].append(t)
        rod["time_days"].append(t / SECONDS_PER_DAY)
        rod["rod_average_linear_heat_rate"].append(q)
        rod["rod_average_burnup"].append(float(self.converter.from_fima(average_fima, unit)))
        rod["gas_pressure"].append(self._gas_pressure.get())
        rod["fission_gas_release"].append(self.fission_gas_release())
        rod["xenon_fraction"].append(self._gas_fractions["xenon"].get())
        rod["helium_fraction"].append(self._gas_fractions["helium"].get())
        rod["gas_amount"].append(self.gas_amount())
        rod["plenum_temperature"].append(self.plenum_temperature)
        rod["max_fuel_centerline_temperature"].append(
            float(state["fuel_centerline_temperature"].max())
        )
        rod["min_gap_width"].append(float(state["gap_width"].min()))
        rod["nonlinear_iterations"].append(iterations)
        axial = result.axial
        axial["linear_heat_rate"].append(q * shape)
        axial["burnup"].append(np.asarray(self.converter.from_fima(average_fima * shape, unit)))
        axial["coolant_temperature"].append(self.coolant_temperature(self.axial_positions, t))
        for name in (
            "fuel_centerline_temperature",
            "fuel_surface_temperature",
            "clad_inner_temperature",
            "clad_outer_temperature",
            "gap_width",
            "fuel_surface_displacement",
            "clad_outer_displacement",
            "clad_hoop_strain",
        ):
            axial[name].append(np.asarray(state[name], dtype=float))
        if isinstance(self.coolant, ForcedConvection) and not self._warned_boiling:
            saturation = self.coolant.saturation_temperature()
            wall = float(np.max(state["clad_outer_temperature"]))
            if wall > saturation:
                self._warned_boiling = True
                warnings.warn(
                    f"FuelRod: the cladding surface reaches {wall:.1f} K, above the coolant "
                    f"saturation temperature {saturation:.1f} K. Subcooled nucleate boiling, "
                    "which would hold the wall a few kelvin above saturation, is not modelled.",
                    stacklevel=3,
                )

    def run(self) -> RodResult:
        """Follow the power history from the first output time to the last
        and return the rod state at every output time (:class:`RodResult`).

        The rod starts in the steady state of the first power level.  Between
        two output times it takes steps no longer than
        ``numerics.max_time_step`` (one step when that is None); after every
        step the fission gas, the gaseous swelling and the gas pressure are
        updated."""
        out = self.output
        started = wall_clock.perf_counter()
        result = RodResult(out.burnup_unit, self.converter, self.axial_positions.copy())
        result.input_report = report.input_report(self)
        if out.print_input:
            print(result.input_report)
        times = self._output_times()
        self.time = float(times[0])
        iterations = 0
        for p in self.problems:
            p.time = self.time
            iterations += self._solve_steady(p)
        self._update_maximum_temperature()
        self._record(result, iterations)
        if out.print_steps:
            print(report.step_header(out.burnup_unit))
            print(report.step_row(result, 0))
        max_step = self.numerics.max_time_step
        bounds = self._step_boundaries(times)
        iterations, k = 0, 0
        for t_next in bounds[1:]:
            span = t_next - self.time
            steps = 1 if max_step is None else max(1, int(np.ceil(span / max_step)))
            dt = span / steps
            start = self.time
            for step in range(steps):
                t_old, t_new = start + step * dt, start + (step + 1) * dt
                for p in self.problems:
                    iterations += self._solve_step(p, t_old, t_new)
                self.time = t_new
                self._update_maximum_temperature()
                self._advance_fission_gas_and_swelling(t_old, t_new)
                if step < steps - 1:
                    self._update_gas_pressure(self._state_along_the_rod())
            if k + 1 < len(times) and t_next >= times[k + 1]:
                k += 1
                self._record(result, iterations)
                iterations = 0
                if out.print_steps:
                    print(report.step_row(result, k))
            else:
                self._update_gas_pressure(self._state_along_the_rod())
        self.wall_time = wall_clock.perf_counter() - started
        if out.print_steps:
            print(result.summary())
            print(f"  wall time: {self.wall_time:.3g} s")
        if out.directory is not None:
            for path in result.write_csv(out.directory, out.file_base):
                if out.print_steps:
                    print(f"  wrote {path}")
        return result


def simplified_points(x, y, tolerance: float) -> np.ndarray:
    """Indices of the points of the polyline (x, y) kept by the
    Douglas-Peucker algorithm with the vertical distance: a point is kept
    when it lies farther than ``tolerance`` from the straight line through
    the kept points on either side.  The first and last points are always
    kept."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    n = len(x)
    keep = np.zeros(n, dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        a, b = stack.pop()
        if b - a < 2:
            continue
        inner = np.arange(a + 1, b)
        line = y[a] + (y[b] - y[a]) * (x[inner] - x[a]) / (x[b] - x[a])
        distance = np.abs(y[inner] - line)
        worst = int(np.argmax(distance))
        if distance[worst] > tolerance:
            m = int(inner[worst])
            keep[m] = True
            stack += [(a, m), (m, b)]
    return np.nonzero(keep)[0]


def _check_type(name, value, types):
    if not isinstance(value, types):
        names = " or ".join(t.__name__ for t in (types if isinstance(types, tuple) else (types,)))
        raise TypeError(f"FuelRod: '{name}' must be a {names}, not {type(value).__name__}.")
