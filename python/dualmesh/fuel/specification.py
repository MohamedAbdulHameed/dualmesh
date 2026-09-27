# SPDX-License-Identifier: LGPL-2.1-or-later
r"""The input of a fuel rod calculation, as dataclasses.

A rod is described by independent groups, each a dataclass, in the order a
fuel performance input deck is written:

========================  =====================================================
:class:`RodGeometry`      the cold, as-fabricated dimensions
:class:`UO2Fuel`, ...     the fuel material and its fabrication data
                          (:mod:`dualmesh.fuel.materials`)
:class:`ZircaloyCladding` the cladding material (:mod:`dualmesh.fuel.materials`)
:class:`FillGas`          the gas the rod is filled and sealed with
:class:`ForcedConvection` the cladding's outer boundary: a coolant channel,
:class:`PrescribedCladdingTemperature`  or a measured wall temperature
:class:`PowerHistory`     the linear heat rate in time or in burnup
:class:`RodModels`        which physical models are switched on
:class:`RodNumerics`      the model dimension, discretisation, mesh, time step
:class:`RodOutput`        what is printed and written
========================  =====================================================

Every field carries its unit and a description, and every default carries
the reason it was chosen; :func:`describe` prints them, marking the values
the user did not set as ``(default)``.  Units are SI, except burnups, which
are given in the unit the user names (see :mod:`dualmesh.fuel.units`).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

from .units import (
    BURNUP_UNITS,
    check_burnup_unit,
)


def parameter(default=dataclasses.MISSING, unit: str = "", description: str = "", **kwargs):
    """A dataclass field with a unit and a description.  For a default, the
    description ends with the reason for it."""
    metadata = {"unit": unit, "description": description}
    if default is dataclasses.MISSING and "default_factory" not in kwargs:
        return field(metadata=metadata, **kwargs)
    if "default_factory" in kwargs:
        return field(metadata=metadata, **kwargs)
    return field(default=default, metadata=metadata, **kwargs)


def _default_of(f):
    if f.default is not dataclasses.MISSING:
        return f.default
    if f.default_factory is not dataclasses.MISSING:
        return f.default_factory()
    return dataclasses.MISSING


def _same(a, b) -> bool:
    try:
        if isinstance(a, (list, tuple, np.ndarray)) or isinstance(b, (list, tuple, np.ndarray)):
            return np.array_equal(np.asarray(a, dtype=object), np.asarray(b, dtype=object))
        return a == b
    except Exception:  # pragma: no cover - exotic values
        return False


def describe(spec) -> list[list[str]]:
    """Rows ``[name, value, unit, source]`` for a specification dataclass,
    where ``source`` is ``(default)`` when the value is the field's default
    and ``given`` otherwise."""
    rows = []
    for f in dataclasses.fields(spec):
        if f.name.startswith("_"):
            continue
        value = getattr(spec, f.name)
        default = _default_of(f)
        is_default = default is not dataclasses.MISSING and _same(value, default)
        if dataclasses.is_dataclass(value):
            text = type(value).__name__
        elif callable(value):
            text = getattr(value, "__name__", "function")
        elif isinstance(value, (list, tuple, np.ndarray)) and len(value) > 6:
            array = np.asarray(value, dtype=float)
            text = f"{len(array)} values, {array.min():.4g} to {array.max():.4g}"
        elif isinstance(value, (list, tuple, np.ndarray)):
            text = "[" + ", ".join(f"{float(v):.6g}" for v in value) + "]"
        elif isinstance(value, float):
            text = f"{value:.6g}"
        else:
            text = str(value)
        rows.append(
            [f.name, text, f.metadata.get("unit", ""), "(default)" if is_default else "given"]
        )
    return rows


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
@dataclass
class RodGeometry:
    """The cold, as-fabricated dimensions of one fuel rod.

    The fuel stack is modelled as a smeared column (no individual pellets,
    dishes or chamfers), as in BISON's layered and smeared-column models and
    in TRANSURANUS.  Construct it from radii, or from the diameters a rod
    specification usually gives with :meth:`from_diameters`.
    """

    pellet_outer_radius: float = parameter(unit="m", description="Outer radius of the pellets.")
    clad_inner_radius: float = parameter(unit="m", description="Inner radius of the cladding.")
    clad_outer_radius: float = parameter(unit="m", description="Outer radius of the cladding.")
    fuel_stack_height: float = parameter(
        unit="m", description="Height of the fuel stack (the modelled length of fuel)."
    )
    pellet_inner_radius: float = parameter(
        0.0,
        unit="m",
        description="Radius of the central hole of annular pellets (a thermocouple bore, for "
        "instance). Default 0: solid pellets, the usual LWR design.",
    )
    plenum_height: float = parameter(
        0.0,
        unit="m",
        description="Height of cladding above the stack that is meshed (the plenum region). "
        "Default 0: the plenum enters only through its gas volume (FillGas.plenum_volume).",
    )
    clad_coating_thickness: float = parameter(
        0.0,
        unit="m",
        description="Thickness of a coating on the outer surface of the cladding (for instance "
        "a chromium coating of 10 to 20 um), counted inside clad_outer_radius. It is meshed as "
        "its own block, 'coating', bonded to the cladding. Default 0: no coating.",
    )

    @classmethod
    def from_diameters(
        cls,
        pellet_outer_diameter: float,
        clad_inner_diameter: float,
        clad_outer_diameter: float,
        fuel_stack_height: float,
        pellet_inner_diameter: float = 0.0,
        plenum_height: float = 0.0,
        clad_coating_thickness: float = 0.0,
    ) -> RodGeometry:
        """The geometry from diameters (m), as rod specifications give them."""
        return cls(
            pellet_outer_radius=0.5 * pellet_outer_diameter,
            clad_inner_radius=0.5 * clad_inner_diameter,
            clad_outer_radius=0.5 * clad_outer_diameter,
            fuel_stack_height=fuel_stack_height,
            pellet_inner_radius=0.5 * pellet_inner_diameter,
            plenum_height=plenum_height,
            clad_coating_thickness=clad_coating_thickness,
        )

    def __post_init__(self):
        if not (
            0.0 <= self.pellet_inner_radius < self.pellet_outer_radius < self.clad_inner_radius
            and self.clad_inner_radius < self.clad_outer_radius
        ):
            raise ValueError(
                "RodGeometry: need 0 <= pellet_inner_radius < pellet_outer_radius < "
                "clad_inner_radius < clad_outer_radius."
            )
        if (
            not 0.0
            <= self.clad_coating_thickness
            < 0.5 * (self.clad_outer_radius - self.clad_inner_radius)
        ):
            raise ValueError(
                "RodGeometry: clad_coating_thickness must be at least 0 and less than half the "
                "cladding wall."
            )
        if self.fuel_stack_height <= 0 or self.plenum_height < 0:
            raise ValueError(
                "RodGeometry: fuel_stack_height must be positive and plenum_height not negative."
            )

    @property
    def radial_gap(self) -> float:
        """The cold radial gap between pellet and cladding, m."""
        return self.clad_inner_radius - self.pellet_outer_radius

    @property
    def clad_thickness(self) -> float:
        return self.clad_outer_radius - self.clad_inner_radius

    @property
    def fuel_cross_section(self) -> float:
        """Area of the fuel in a cross-section, m^2."""
        return np.pi * (self.pellet_outer_radius**2 - self.pellet_inner_radius**2)


@dataclass
class RodMesh:
    """How finely the rod is meshed.  The defaults give temperatures within a
    few kelvin of a converged mesh for LWR rods (checked in the test suite);
    refine them for steep power profiles or for stresses near the pellet
    surface."""

    num_fuel_radial_elements: int = parameter(
        12, description="Elements across the pellet radius. Default 12."
    )
    num_clad_radial_elements: int = parameter(
        3, description="Elements across the cladding wall. Default 3."
    )
    num_clad_coating_elements: int = parameter(
        2, description="Elements across a cladding coating, when there is one. Default 2."
    )
    num_axial_elements: int = parameter(
        10,
        description="Elements along the stack (axisymmetric and three_dimensional models). "
        "Default 10.",
    )
    num_axial_slices: int = parameter(
        5, description="Radial slices along the stack (1.5D model). Default 5."
    )
    num_fuel_core_divisions: int = parameter(
        4,
        description="Divisions of the central square of the pellet cross-section in the "
        "three_dimensional model (an even number); the pellet has 4 times as many angular "
        "divisions. Default 4.",
    )
    fuel_surface_grading: float = parameter(
        0.5,
        description="Ratio of the outermost to the innermost radial fuel element: below one "
        "refines towards the pellet surface, where temperature and power change fastest. "
        "Default 0.5.",
    )


# ---------------------------------------------------------------------------
# Gas
# ---------------------------------------------------------------------------
GASES = ("helium", "argon", "krypton", "xenon", "hydrogen", "nitrogen")


@dataclass
class FillGas:
    """The gas the rod is filled with when it is sealed."""

    pressure: float = parameter(unit="Pa", description="Fill pressure at the fill temperature.")
    plenum_volume: float = parameter(
        unit="m^3",
        description="Free gas volume outside the pellet-cladding gap (plenum, dishes, "
        "chamfers), for the modelled stack.",
    )
    composition: dict = parameter(
        default_factory=lambda: {"helium": 1.0},
        description="Mole fractions by gas (helium, argon, krypton, xenon, hydrogen, "
        "nitrogen); they must sum to one. Default pure helium, the usual LWR fill.",
    )
    temperature: float = parameter(
        293.15, unit="K", description="Fill temperature. Default 293.15 K (room temperature)."
    )

    def __post_init__(self):
        unknown = set(self.composition) - set(GASES)
        if unknown:
            raise ValueError(
                f"FillGas: unknown gas(es) {', '.join(sorted(unknown))}. Use {', '.join(GASES)}."
            )
        total = sum(self.composition.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"FillGas: the mole fractions sum to {total:g}, not 1.")
        if self.pressure <= 0 or self.plenum_volume < 0 or self.temperature <= 0:
            raise ValueError(
                "FillGas: pressure and temperature must be positive and plenum_volume not negative."
            )


# ---------------------------------------------------------------------------
# The cladding's outer boundary
# ---------------------------------------------------------------------------
@dataclass
class ForcedConvection:
    r"""Single-phase forced convection of water in a coolant channel of a
    square lattice.

    * **Energy balance.** The coolant enthalpy rises along the rod,
      :math:`h(z) = h_{in} + \int_0^z q'\,dz' / (G A)`, and the bulk
      temperature is :math:`T(p, h)`.  The water properties are those of
      IAPWS-IF97 (region 1) with the IAPWS viscosity (R12-08) and thermal
      conductivity (R15-11) at the local bulk temperature and the coolant
      pressure (:mod:`dualmesh.fuel.water`), unless ``specific_heat``,
      ``thermal_conductivity`` and ``dynamic_viscosity`` are given, in which
      case they are held constant.  The specific heat of water at 15.5 MPa
      rises by a quarter between 565 and 610 K, so a constant value misplaces
      the outlet temperature by a few kelvin.
    * **Heat transfer coefficient**, at every height and time from the local
      properties on the hydraulic diameter :math:`D_h`:
      ``correlation="dittus_boelter"`` (the default),
      :math:`Nu = 0.023 Re^{0.8} Pr^{0.4}`, or ``"weisman"``,
      :math:`Nu = C Re^{0.8} Pr^{1/3}` with :math:`C = 0.042 P/D - 0.024` for
      a square lattice of pitch-to-diameter ratio P/D (EPRI 1000215, Eqs.
      4-4 and 4-5).  EPRI 1000215 found that no single-phase correlation has
      been validated for rod bundles at PWR conditions; Dittus-Boelter gives
      the lower coefficient (about 30 % lower for a 17 x 17 lattice) and so
      the higher cladding temperature.  A fixed ``heat_transfer_coefficient``
      replaces both.
    * **Limits.** The model is single phase: a run stops if the bulk
      temperature reaches saturation, and it warns if the cladding surface
      rises above the saturation temperature, where subcooled nucleate
      boiling (not modelled) would hold the wall a few kelvin above
      saturation.
    """

    inlet_temperature: float = parameter(unit="K", description="Coolant inlet temperature.")
    pressure: float = parameter(unit="Pa", description="Coolant pressure.")
    mass_flux: float = parameter(unit="kg/(m^2 s)", description="Coolant mass flux.")
    rod_pitch: float = parameter(unit="m", description="Pitch of the square rod lattice.")
    correlation: str = parameter(
        "dittus_boelter",
        description="dittus_boelter or weisman. Default dittus_boelter (see the class).",
    )
    specific_heat: float | None = parameter(
        None,
        unit="J/(kg K)",
        description="A constant specific heat. Default None: IAPWS-IF97 at the local state.",
    )
    thermal_conductivity: float | None = parameter(
        None,
        unit="W/(m K)",
        description="A constant conductivity. Default None: IAPWS R15-11 at the local state.",
    )
    dynamic_viscosity: float | None = parameter(
        None,
        unit="Pa s",
        description="A constant viscosity. Default None: IAPWS R12-08 at the local state.",
    )
    heat_transfer_coefficient: float | None = parameter(
        None,
        unit="W/(m^2 K)",
        description="A fixed heat transfer coefficient instead of the correlation. Default None.",
    )

    def __post_init__(self):
        if self.correlation not in ("dittus_boelter", "weisman"):
            raise ValueError("ForcedConvection: correlation must be dittus_boelter or weisman.")
        given = [
            v is not None
            for v in (self.specific_heat, self.thermal_conductivity, self.dynamic_viscosity)
        ]
        if any(given) and not all(given):
            raise ValueError(
                "ForcedConvection: give all three of specific_heat, thermal_conductivity and "
                "dynamic_viscosity, or none (IAPWS properties)."
            )

    @property
    def constant_properties(self) -> bool:
        return self.specific_heat is not None

    def flow_area(self, clad_outer_radius: float) -> float:
        return self.rod_pitch**2 - np.pi * clad_outer_radius**2

    def hydraulic_diameter(self, clad_outer_radius: float) -> float:
        return 4.0 * self.flow_area(clad_outer_radius) / (2.0 * np.pi * clad_outer_radius)

    def saturation_temperature(self) -> float:
        from . import water

        return float(water.saturation_temperature(self.pressure))

    def bulk_temperature(self, heat, clad_outer_radius: float):
        """The bulk temperature (K) after the coolant has taken up ``heat``
        (W, an array) from the inlet."""
        heat = np.asarray(heat, dtype=float)
        flow = self.mass_flux * self.flow_area(clad_outer_radius)
        if self.constant_properties:
            return self.inlet_temperature + heat / (flow * self.specific_heat)
        from . import water

        h = water.enthalpy(self.pressure, self.inlet_temperature) + heat / flow
        T = water.temperature(self.pressure, h)
        saturation = self.saturation_temperature()
        if np.any(saturation <= T):
            raise ValueError(
                f"ForcedConvection: the coolant reaches its saturation temperature "
                f"({saturation:.1f} K at {self.pressure / 1e6:.2f} MPa); the single-phase "
                "model does not apply. Raise the mass flux or lower the power."
            )
        return T

    def fluid_properties(self, bulk_temperature):
        """(specific heat, conductivity, viscosity) at the bulk temperature."""
        if self.constant_properties:
            T = np.asarray(bulk_temperature, dtype=float)
            ones = np.ones_like(T)
            return (
                self.specific_heat * ones,
                self.thermal_conductivity * ones,
                self.dynamic_viscosity * ones,
            )
        from . import water

        return (
            water.isobaric_heat_capacity(self.pressure, bulk_temperature),
            water.thermal_conductivity(self.pressure, bulk_temperature),
            water.viscosity(self.pressure, bulk_temperature),
        )

    def heat_transfer_coefficient_for(self, clad_outer_radius: float, bulk_temperature=None):
        """W/(m^2 K): the given one, or the correlation at the bulk
        temperature (default: the inlet temperature)."""
        if self.heat_transfer_coefficient is not None:
            return (
                float(self.heat_transfer_coefficient)
                if bulk_temperature is None
                else np.full(np.shape(bulk_temperature), float(self.heat_transfer_coefficient))
            )
        T = self.inlet_temperature if bulk_temperature is None else bulk_temperature
        cp, k, mu = self.fluid_properties(T)
        dh = self.hydraulic_diameter(clad_outer_radius)
        reynolds = self.mass_flux * dh / mu
        prandtl = mu * cp / k
        if self.correlation == "weisman":
            constant = 0.042 * self.rod_pitch / (2.0 * clad_outer_radius) - 0.024
            nusselt = constant * reynolds**0.8 * prandtl ** (1.0 / 3.0)
        else:
            nusselt = 0.023 * reynolds**0.8 * prandtl**0.4
        value = nusselt * k / dh
        return float(value) if np.ndim(value) == 0 else value


@dataclass
class PrescribedCladdingTemperature:
    """The cladding outer surface temperature is prescribed, as a constant or
    as a function of the local linear heat rate q' (W/m) and the time t (s).
    This is how the IAEA benchmark specifications give the boundary of
    test-reactor rods, for instance the Halden correlation of FUMEX-II,
    :math:`T = 240 + 0.4162\\, q'^{0.75}` degC with q' in kW/m."""

    temperature: float | Callable = parameter(
        unit="K", description="Constant, or a function (linear_heat_rate, time) -> K."
    )
    pressure: float = parameter(
        unit="Pa", description="Coolant pressure outside the cladding (for the mechanics)."
    )

    def value(self, linear_heat_rate: float, time: float) -> float:
        if callable(self.temperature):
            return float(self.temperature(linear_heat_rate, time))
        return float(self.temperature)


# ---------------------------------------------------------------------------
# Power history
# ---------------------------------------------------------------------------
@dataclass
class PowerHistory:
    """The rod-average linear heat rate, piecewise linear in time or in
    burnup.

    Give ``time`` (s), or ``burnup`` with its ``burnup_unit``; the rod turns
    a burnup-based history into times from the heavy-metal mass of the fuel.
    The axial and radial power profiles are shapes, normalised by the rod to
    an average of one.
    """

    linear_heat_rate: Sequence[float] = parameter(
        unit="W/m", description="Rod-average linear heat rate at each point of the history."
    )
    time: Sequence[float] | None = parameter(
        None, unit="s", description="Times of the points. Default None: given in burnup."
    )
    burnup: Sequence[float] | None = parameter(
        None,
        description="Rod-average burnups of the points, in burnup_unit. Default None: given "
        "in time.",
    )
    burnup_unit: str = parameter(
        "MWd/kgHM",
        description=f"Unit of 'burnup': one of {', '.join(BURNUP_UNITS)}. Default MWd/kgHM, "
        "the unit of LWR operating data.",
    )
    axial_profile: Callable | None = parameter(
        None,
        description="Axial shape f(z / stack height), normalised to an average of one. "
        "Default None: flat.",
    )
    radial_profile: Callable | None = parameter(
        None,
        description="Radial shape f(r / pellet radius), normalised to an area average of one. "
        "Default None: flat. LWR fuel peaks at the rim at high burnup; supply a profile "
        "(for instance from TUBRNP) when that matters.",
    )
    fast_neutron_flux_per_linear_heat_rate: float = parameter(
        3.0e13,
        unit="n/(m^2 s) per W/m",
        description="Fast neutron flux (E > 1 MeV) per unit linear heat rate. Default 3e13, a "
        "typical LWR value (6e17 n/(m^2 s) at 20 kW/m); set it for the reactor modelled.",
    )

    def __post_init__(self):
        self.linear_heat_rate = np.asarray(self.linear_heat_rate, dtype=float)
        if (self.time is None) == (self.burnup is None):
            raise ValueError("PowerHistory: give either 'time' or 'burnup', not both or neither.")
        points = self.time if self.time is not None else self.burnup
        points = np.asarray(points, dtype=float)
        if points.shape != self.linear_heat_rate.shape or len(points) < 2:
            raise ValueError(
                "PowerHistory: 'linear_heat_rate' and the time or burnup points need the same "
                "length, at least two."
            )
        if np.any(np.diff(points) <= 0):
            raise ValueError("PowerHistory: the time or burnup points must increase strictly.")
        if self.time is not None:
            self.time = points
        else:
            self.burnup = points
            check_burnup_unit(self.burnup_unit)
            if np.any(self.linear_heat_rate <= 0):
                raise ValueError(
                    "PowerHistory: a history in burnup needs a positive linear heat rate at "
                    "every point (no burnup accumulates at zero power); give zero-power "
                    "periods in time."
                )


# ---------------------------------------------------------------------------
# Models, numerics and output
# ---------------------------------------------------------------------------
@dataclass
class RodModels:
    """Which physical models are switched on."""

    mechanics: bool = parameter(
        True,
        description="Solve the mechanics (thermal expansion, swelling, relocation, creep, "
        "contact, pressures). Default True: the gap, and so the temperatures, depend on it.",
    )
    relocation: bool | None = parameter(
        None,
        description="Fuel relocation (ESCORE). Default None: the fuel's own default (on for "
        "UO2; the other fuels have no relocation model).",
    )
    densification: bool | None = parameter(
        None,
        description="Densification (ESCORE form). Default None: the fuel's own default (on "
        "for UO2; the other fuels have no densification model).",
    )
    solid_swelling: bool = parameter(
        True,
        description="Solid fission-product swelling (UO2: MATPRO; UN: the Ross, El-Genk and "
        "Matthews correlation, which covers all of UN's swelling; a CustomFuel: its "
        "volumetric_swelling). Default True.",
    )
    gaseous_swelling: bool | None = parameter(
        None,
        description="Gaseous swelling of UO2 (MATPRO FSWELL), integrated between time steps. "
        "Default None: the fuel's own default (on for UO2).",
    )
    creep: bool = parameter(True, description="Fuel and cladding creep. Default True.")
    fission_gas_release: str | None = parameter(
        None,
        description="booth (UO2 and a CustomFuel), storms (UN), fraction (a CustomFuel) or "
        "none. Default None: the fuel's own default (booth for UO2, storms for UN).",
    )
    intragranular_trapping: str | None = parameter(
        None,
        description="speight (trapping in and re-solution from intragranular bubbles, with the "
        "UO2 parameters of Zullo et al. 2023) or constant (the diffusion coefficient times "
        "trapping_factor). Default None: speight for UO2 and doped UO2, constant for other "
        "fuels.",
    )
    trapping_factor: float = parameter(
        1.0,
        description="With intragranular_trapping = constant: a factor (0, 1] on the Booth "
        "diffusion coefficient. Default 1 (no trapping).",
    )

    plenum_temperature_rise: float = parameter(
        25.0,
        unit="K",
        description="Temperature of the plenum gas above the coolant at the top of the stack "
        "(or above the prescribed cladding temperature there). Default 25 K, a rough estimate "
        "of the heating of the plenum by the stack; the plenum is a small part of the gas "
        "temperature sum, so the pressure changes by a few per cent per 25 K.",
    )

    def resolved(self, fuel) -> RodModels:
        """A copy with the fuel-dependent defaults filled in, and a clear
        error for a model the fuel does not have."""
        name = fuel.material_name
        chosen = {}
        for model in ("relocation", "densification", "gaseous_swelling"):
            value = getattr(self, model)
            default = fuel.optional_models.get(model, False)
            chosen[model] = default if value is None else bool(value)
            if chosen[model] and model not in fuel.optional_models:
                raise ValueError(
                    f"RodModels: {model} is not modelled for {name} fuel; set {model}=False."
                )
        allowed = tuple(fuel.fission_gas_release_models)
        release = self.fission_gas_release if self.fission_gas_release is not None else allowed[0]
        if release != "none" and release not in allowed:
            raise ValueError(
                f"RodModels: fission_gas_release '{release}' does not apply to {name} fuel. Use "
                f"one of: {', '.join(dict.fromkeys(allowed + ('none',)))}."
            )
        return dataclasses.replace(self, fission_gas_release=release, **chosen)


@dataclass
class RodNumerics:
    """How the rod is discretised and solved."""

    model: str = parameter(
        "axisymmetric",
        description="axisymmetric (an r-z model of the whole rod), 1.5d (radial slices in "
        "generalized plane strain coupled through the coolant and the gas, as TRANSURANUS "
        "and FRAPCON) or three_dimensional. Default axisymmetric: two-dimensional accuracy at "
        "modest cost.",
    )
    method: str = parameter(
        "fem", description="fem, dmcdm, hfvm or zfvm. Default fem, the method of BISON."
    )
    strain: str = parameter(
        "small",
        description="small or finite. finite solves the mechanics on the undeformed mesh with "
        "the deformation gradient (finite_strain_stress, follower pressures), for large "
        "deformations such as cladding ballooning. Default small: in normal operation the "
        "strains stay below about 1 %, where the two agree to a fraction of a per cent of "
        "the strain and the small-strain form is cheaper.",
    )
    heat_conduction_configuration: str = parameter(
        "deformed",
        description="deformed or undeformed: the body through which the heat conducts. "
        "The mesh never moves, so with deformed the heat equation is carried back to it with "
        "the deformation gradient F: the conductivity becomes J k F^-1 F^-T and the heat "
        "fluxes through the gap and into the coolant are multiplied by the ratio of deformed "
        "to undeformed surface area. The fission heat and the heat capacity need no change, "
        "because they are given per unit undeformed volume. With undeformed the heat "
        "conducts through the as-built geometry, and only the gap width sees the "
        "displacements. Default deformed: it costs little and removes an error of the order "
        "of the strain (the pellet radius grows by about 1 % at power). It needs the "
        "mechanics; without it the geometry is the as-built one either way.",
    )
    mesh: RodMesh = parameter(default_factory=RodMesh, description="See RodMesh.")
    max_time_step: float | None = parameter(
        None,
        unit="s",
        description="Largest time step. Default None: one step per interval between output times.",
    )
    power_history_tolerance: float | None = parameter(
        0.01,
        description="The time steps land on the points of the power history needed to follow "
        "it to within this fraction of its largest linear heat rate (the points closer than "
        "that to the straight line through the kept ones are skipped, by the Douglas-Peucker "
        "algorithm), besides the output times. Default 0.01: shutdowns and power steps are "
        "resolved, while a history digitised point by point does not force a step at every "
        "point. 0 steps to every point; None only to the output times.",
    )
    contact_penalty: float = parameter(
        1.0e15,
        unit="Pa/m",
        description="Penalty stiffness of pellet-cladding contact. Default 1e15: large "
        "against E/L of the cladding (about 1e14), so the penetration is under 1 um.",
    )
    solver_options: dict = parameter(
        default_factory=dict,
        description="Options passed to every solve (see Problem.solve). Default none. The rod "
        "adds line_search='backtracking' unless it is given here: fuel creep switches from a "
        "linear to a power law at the MATPRO transition stress, and a full Newton step across "
        "that kink can diverge over a long time step.",
    )

    def __post_init__(self):
        if self.model not in ("axisymmetric", "1.5d", "three_dimensional"):
            raise ValueError(
                f"RodNumerics: unknown model '{self.model}'. Use axisymmetric, 1.5d or "
                "three_dimensional."
            )
        if self.strain not in ("small", "finite"):
            raise ValueError(f"RodNumerics: strain must be small or finite, not '{self.strain}'.")
        if self.heat_conduction_configuration not in ("deformed", "undeformed"):
            raise ValueError(
                "RodNumerics: heat_conduction_configuration must be deformed or undeformed, "
                f"not '{self.heat_conduction_configuration}'."
            )
        if self.power_history_tolerance is not None and self.power_history_tolerance < 0:
            raise ValueError("RodNumerics: power_history_tolerance must be None or >= 0.")
        if self.method not in ("fem", "dmcdm", "hfvm", "zfvm"):
            raise ValueError(
                f"RodNumerics: unknown method '{self.method}'. Use fem, dmcdm, hfvm or zfvm."
            )


@dataclass
class RodOutput:
    """What is printed and written."""

    burnup_unit: str = parameter(
        "MWd/kgHM", description="Unit of every burnup reported. Default MWd/kgHM."
    )
    output_times: Sequence[float] | None = parameter(
        None,
        unit="s",
        description="Times at which the rod state is recorded. Default None: the points of the "
        "power history (converted to times).",
    )
    print_input: bool = parameter(
        True, description="Print the input summary before the run. Default True."
    )
    print_steps: bool = parameter(
        True, description="Print one table row per output time. Default True."
    )
    directory: str | None = parameter(
        None,
        description="Directory for the CSV files (history, axial profiles, input). Default "
        "None: nothing written.",
    )
    file_base: str = parameter("rod", description="Prefix of the files written. Default rod.")

    def __post_init__(self):
        check_burnup_unit(self.burnup_unit)
