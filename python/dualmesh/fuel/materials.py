# SPDX-License-Identifier: LGPL-2.1-or-later
r"""Fuel and cladding materials of a rod.

A :class:`FuelRod` does not know any material itself.  It asks the fuel and
the cladding to add their objects to each problem it builds, through two
small interfaces:

:class:`FuelMaterial`
    thermal properties, elastic constants, eigenstrains (thermal expansion,
    densification, swelling, relocation), creep, fission gas release, and the
    heavy-metal content from which burnup follows;
:class:`CladdingMaterial`
    thermal properties, elastic constants, eigenstrains (thermal expansion,
    irradiation growth) and creep.

The built-in materials implement them with compiled C++ correlations:
:class:`UO2Fuel`, :class:`UNFuel` and :class:`ZircaloyCladding`.

A material the library does not have, or a correlation to try in place of a
built-in one, is written in one of two ways:

* :class:`CustomFuel` and :class:`CustomCladding` take every property as an
  expression, for instance ``thermal_conductivity="21.7 - 3.04e-3*temperature
  + 3.61e-6*temperature^2"``.  The expressions are compiled once and
  differentiated automatically, so a custom material runs at the speed of a
  built-in one and Newton's method keeps its exact Jacobian.
* A subclass of :class:`FuelMaterial` or :class:`CladdingMaterial` implements
  the methods directly, with any objects of the library, including materials
  written in Python (:class:`dualmesh.PythonProperty`).

Units are SI throughout; burnup is in FIMA inside expressions, and the
constant ``mwd_per_kg_per_fima`` converts it to MWd/kgHM.
"""

from __future__ import annotations

import abc
import dataclasses
from dataclasses import dataclass

import numpy as np

from .. import _core
from .specification import parameter
from .units import AVOGADRO, DEFAULT_ENERGY_PER_FISSION, NATURAL_URANIUM_MOLAR_MASS

#: The element field that holds the highest temperature each element of a
#: rod has reached, K.  The rod driver raises it after every step; models
#: that must not reverse when the rod cools (densification, the swelling of
#: SiC at its irradiation temperature) read it.
MAXIMUM_TEMPERATURE = "maximum_temperature"


# ---------------------------------------------------------------------------
# What a material is told about the rod
# ---------------------------------------------------------------------------
@dataclass
class RodContext:
    """What a material needs to know to add its objects to one problem of a
    rod.

    ``formulation`` is the formulation of the stress objects
    (``axisymmetric``, ``axisymmetric_1d`` or ``three_dimensional``).
    ``models`` are the rod's :class:`RodModels` with every choice resolved,
    and ``factors`` its :class:`ModelFactors`.
    ``burnup`` holds the parameters that every burnup-dependent built-in
    object takes (the burnup field in FIMA, its unit, the energy per fission
    and the heavy-metal molar mass)."""

    formulation: str
    models: object
    geometry: object
    stress_free_temperature: float
    burnup: dict
    mwd_per_kg_per_fima: float
    temperature: str = "temperature"
    factors: object = None


# ---------------------------------------------------------------------------
# The interfaces
# ---------------------------------------------------------------------------
def _check_grain_radius(material) -> None:
    """Refuse a fuel without its grain radius, which is fabrication data of
    the case and has no default."""
    name = type(material).__name__
    if material.grain_radius is None:
        raise ValueError(f"{name}: missing required parameter 'grain_radius', the mean grain radius of the fuel (m), from its fabrication data.")
    if not material.grain_radius > 0.0:
        raise ValueError(f"{name}: grain_radius must be positive, not {material.grain_radius}.")


class FuelMaterial(abc.ABC):
    """The interface of a fuel material.  See the module documentation.

    A subclass is usually a dataclass whose fields are the fabrication data
    and model choices.  It must have the attributes
    ``theoretical_density_fraction``, ``grain_radius`` (m),
    ``surface_roughness`` (m), ``emissivity`` and ``energy_per_fission`` (J),
    and ``heavy_metal_molar_mass`` (kg/mol), and define the class
    attributes below."""

    #: Short name used in reports and messages, for instance "UO2".
    material_name: str = ""
    #: The optional models this fuel has, with their default (on or off):
    #: any of "relocation", "densification" and "gaseous_swelling".  A model
    #: not listed is refused when switched on.
    optional_models: dict = {}
    #: The fission gas release models this fuel accepts; the first is the
    #: default.
    fission_gas_release_models: tuple = ("none",)

    #: Molar mass of the heavy metal (uranium, plutonium, thorium), kg/mol: an
    #: attribute (a field or a property) of every fuel.
    heavy_metal_molar_mass: float

    @abc.abstractmethod
    def heavy_metal_atom_density(self) -> float:
        """Heavy-metal atoms per m^3 of fuel, at the fabricated density."""

    @abc.abstractmethod
    def add_thermal_properties(self, problem, block: str, context: RodContext) -> None:
        """Declare 'thermal_conductivity', 'specific_heat' and 'density' on
        ``block``."""

    @abc.abstractmethod
    def add_elasticity(self, problem, block: str, context: RodContext) -> None:
        """Declare 'youngs_modulus' and 'poissons_ratio' on ``block``."""

    @abc.abstractmethod
    def add_eigenstrains(self, problem, block: str, context: RodContext) -> list[str]:
        """Add the eigenstrain materials and return their names."""

    def creep_parameters(self, context: RodContext) -> dict:
        """Parameters of ``small_strain_stress`` that switch creep on.  The
        default: no creep model."""
        return {}

    def fission_gas_model(self, num_elements: int, context: RodContext):
        """The fission gas model for ``num_elements`` fuel elements, or None."""
        return None

    def gaseous_swelling_increment(self, temperature, burnup, burnup_increment):
        """Increment of the volumetric gaseous swelling strain (arrays), for a
        fuel with the 'gaseous_swelling' model."""
        raise NotImplementedError(f"{self.material_name} has no gaseous swelling model.")


class CladdingMaterial(abc.ABC):
    """The interface of a cladding material.  See the module documentation.

    A subclass must have the attributes ``surface_roughness`` (m),
    ``emissivity`` and ``meyer_hardness`` (Pa)."""

    material_name: str = ""

    @abc.abstractmethod
    def add_thermal_properties(self, problem, block: str, context: RodContext) -> None:
        """Declare 'thermal_conductivity', 'specific_heat' and 'density'."""

    @abc.abstractmethod
    def add_elasticity(self, problem, block: str, context: RodContext) -> None:
        """Declare 'youngs_modulus' and 'poissons_ratio'."""

    @abc.abstractmethod
    def add_eigenstrains(self, problem, block: str, context: RodContext) -> list[str]:
        """Add the eigenstrain materials and return their names."""

    def creep_parameters(self, context: RodContext, block: str = "clad") -> dict:
        """Parameters of the stress material that switch creep on for
        ``block``.  The default: no creep model."""
        return {}


def _check_fraction(spec, name, low, high, open_low=False):
    value = getattr(spec, name)
    ok = (low < value if open_low else low <= value) and value <= high
    if not ok:
        bracket = "(" if open_low else "["
        raise ValueError(f"{type(spec).__name__}: {name} must lie in {bracket}{low}, {high}].")


def uranium_molar_mass(enrichment: float) -> float:
    """Molar mass of uranium (kg/mol) with a U-235 weight fraction, the rest
    U-238."""
    return 1.0e-3 / (enrichment / 235.0439299 + (1.0 - enrichment) / 238.0507882)


# ---------------------------------------------------------------------------
# Built-in fuels
# ---------------------------------------------------------------------------
@dataclass
class UO2Fuel(FuelMaterial):
    """Uranium dioxide fuel."""

    enrichment: float = parameter(0.045, description="U-235 weight fraction of the uranium. Default 0.045, typical of LWR fuel. It enters only the heavy-metal molar mass (the burnup conversion).")
    theoretical_density_fraction: float = parameter(0.95, description="As-fabricated density as a fraction of the theoretical density (10963 kg/m^3). Default 0.95, typical of LWR fuel.")
    grain_radius: float | None = parameter(None, unit="m", description="Grain radius, for fission gas diffusion and creep: the radius of the sphere of the grain's volume, 1.56 times half the mean linear intercept length. Required, from the fabrication data of the fuel.")
    thermal_conductivity_model: str = parameter("fink_lucuta", description="fink, fink_lucuta, nfi or halden (see UO2_thermal). Default fink_lucuta: the recommended unirradiated conductivity with a published treatment of irradiation.")
    specific_heat_model: str = parameter("fink", description="fink or matpro. Default fink, recommended by IAEA-TECDOC-1496.")
    gadolinia_weight_fraction: float = parameter(0.0, description="Gd2O3 weight fraction (nfi conductivity only). Default 0.")
    densification_model: str = parameter("escore", description="escore (the ESCORE model of FALCON MOD01, EPRI 1011307, with the pellet-average burnup) or matpro (MATPRO FUDENS). Default escore, which agrees better with the fuel benchmarks.")
    total_densification: float = parameter(0.01, description="Density change in a resintering test (1973 K, 24 h) as a fraction of the theoretical density. Default 0.01, a typical value. Set it from the fuel's own test.")
    densification_complete_burnup: float = parameter(5.0, unit="MWd/kgHM", description="Burnup at which densification is complete, for the escore model only. Default 5 MWd/kgHM (ESCORE).")
    gaseous_swelling_model: str = parameter(
        "fission_gas",
        description="fission_gas (the volume of the intragranular and grain-face bubbles of "
        "the fission gas model, Pastore et al. 2013, Eqs. 9 and 10, as BISON computes it, "
        "and none when fission_gas_release is none, because there is then no gas model) or "
        "matpro (the empirical MATPRO FSWELL correlation, "
        "independent of the fission gas model). Default fission_gas: the swelling and the "
        "release then come from the same gas, and on the Halden thermocouples of IFA-677.1 "
        "FSWELL closes the gap too fast (see the fuel benchmarks).",
    )
    surface_roughness: float = parameter(2.0e-6, unit="m", description="Pellet surface roughness. Default 2 um (ground pellets).")
    emissivity: float = parameter(0.8, description="Pellet surface emissivity. Default 0.8 (MATPRO).")
    energy_per_fission: float = parameter(DEFAULT_ENERGY_PER_FISSION, unit="J", description="Recoverable energy per fission. Default 3.2044e-11 J (200 MeV).")

    material_name = "UO2"
    optional_models = {"relocation": True, "densification": True, "gaseous_swelling": True}
    fission_gas_release_models = ("booth", "none")
    theoretical_density = 10963.0  # kg/m^3, Fink (2000)

    def __post_init__(self):
        _check_grain_radius(self)
        _check_fraction(self, "enrichment", 0.0, 1.0)
        _check_fraction(self, "theoretical_density_fraction", 0.5, 1.0, open_low=True)
        if self.thermal_conductivity_model not in ("fink", "fink_lucuta", "nfi", "halden"):
            raise ValueError(f"UO2Fuel: unknown thermal_conductivity_model '{self.thermal_conductivity_model}'. Use fink, fink_lucuta, nfi or halden.")
        if self.specific_heat_model not in ("fink", "matpro"):
            raise ValueError(f"UO2Fuel: unknown specific_heat_model '{self.specific_heat_model}'. Use fink or matpro.")
        if self.densification_model not in ("matpro", "escore"):
            raise ValueError(f"UO2Fuel: unknown densification_model '{self.densification_model}'. Use matpro or escore.")
        if self.gaseous_swelling_model not in ("fission_gas", "matpro"):
            raise ValueError(f"UO2Fuel: unknown gaseous_swelling_model '{self.gaseous_swelling_model}'. Use fission_gas or matpro.")

    @property
    def heavy_metal_molar_mass(self) -> float:
        """kg/mol, from the enrichment."""
        return uranium_molar_mass(self.enrichment)

    @property
    def compound_molar_mass(self) -> float:
        """kg/mol of UO2."""
        return self.heavy_metal_molar_mass + 2 * 15.999e-3

    def heavy_metal_atom_density(self) -> float:
        return self.theoretical_density_fraction * self.theoretical_density * AVOGADRO / self.compound_molar_mass

    def add_thermal_properties(self, problem, block, context):
        problem.add_property(
            "UO2_thermal",
            f"{block}_thermal",
            block=[block],
            temperature=context.temperature,
            thermal_conductivity_model=self.thermal_conductivity_model,
            specific_heat_model=self.specific_heat_model,
            gadolinia_weight_fraction=self.gadolinia_weight_fraction,
            theoretical_density_fraction=self.theoretical_density_fraction,
            **context.burnup,
        )

    def add_elasticity(self, problem, block, context):
        problem.add_property("UO2_elasticity", f"{block}_elasticity", block=[block], temperature=context.temperature, theoretical_density_fraction=self.theoretical_density_fraction)

    def add_eigenstrains(self, problem, block, context):
        models, g = context.models, context.geometry
        names = [f"{block}_thermal_strain", f"{block}_volumetric_strain"]
        problem.add_property("UO2_thermal_expansion_eigenstrain", f"{block}_thermal_expansion", block=[block], temperature=context.temperature, stress_free_temperature=context.stress_free_temperature, eigenstrain_name=names[0], formulation=context.formulation)
        problem.add_property(
            "UO2_volumetric_swelling_eigenstrain",
            f"{block}_volumetric_swelling",
            block=[block],
            temperature=context.temperature,
            theoretical_density_fraction=self.theoretical_density_fraction,
            total_densification=self.total_densification if models.densification else 0.0,
            densification_model=self.densification_model,
            densification_complete_burnup=self.densification_complete_burnup,
            densification_burnup="pellet_average_burnup",
            maximum_temperature_field=MAXIMUM_TEMPERATURE,
            include_solid_swelling=models.solid_swelling,
            solid_swelling_factor=(context.factors.solid_swelling if context.factors is not None else 1.0),
            gaseous_swelling_field="gaseous_swelling" if models.gaseous_swelling else "",
            eigenstrain_name=names[1],
            formulation=context.formulation,
            **context.burnup,
        )
        if models.gaseous_swelling:
            problem.set_element_field("gaseous_swelling", np.zeros(problem._mesh.num_elements))
        if models.relocation:
            names.append(f"{block}_relocation_strain")
            problem.add_property(
                "UO2_relocation_eigenstrain", f"{block}_relocation", block=[block], linear_heat_rate="linear_heat_rate", pellet_diameter=2 * g.pellet_outer_radius, diametral_gap=2 * g.radial_gap, eigenstrain_name=names[-1], formulation=context.formulation, **{**context.burnup, "burnup": "pellet_average_burnup"}
            )
        return names

    def creep_parameters(self, context):
        return dict(creep_model="uo2", temperature=context.temperature, fission_rate="fission_rate", theoretical_density_fraction=self.theoretical_density_fraction, grain_radius=self.grain_radius)

    def fission_gas_model(self, num_elements, context):
        from .gas import BoothFissionGasRelease

        if context.models.fission_gas_release == "booth":
            return BoothFissionGasRelease(num_elements, grain_radius=self.grain_radius, trapping_factor=context.models.trapping_factor, trapping=context.models.intragranular_trapping)
        return None

    def gaseous_swelling_increment(self, temperature, burnup, burnup_increment):
        return _core.fuel.uo2_gaseous_swelling_increment(temperature, burnup, burnup_increment, self.theoretical_density_fraction)


@dataclass
class UNFuel(FuelMaterial):
    """Uranium mononitride fuel."""

    enrichment: float = parameter(0.05, description="U-235 weight fraction. Default 0.05. It enters only the heavy-metal molar mass (the burnup conversion).")
    theoretical_density_fraction: float = parameter(0.95, description="Density as a fraction of the theoretical density, 14326 kg/m^3 at 298 K. Default 0.95.")
    grain_radius: float | None = parameter(None, unit="m", description="Grain radius, for the grain-boundary creep (grain size 2 x radius). Required, from the fabrication data of the fuel (fabricated UN has 5 to 30 um grains, AbdulHameed et al. 2025).")
    grain_boundary_creep: bool = parameter(True, description="Include the grain-boundary (Coble) creep of AbdulHameed et al. (2025), which dominates dislocation creep for 5-30 um grains below about 1800 K. Default True.")
    surface_roughness: float = parameter(2.0e-6, unit="m", description="Pellet surface roughness. Default 2 um (as UO2).")
    emissivity: float = parameter(0.8, description="Pellet surface emissivity. Default 0.8, the UO2 value.")
    energy_per_fission: float = parameter(DEFAULT_ENERGY_PER_FISSION, unit="J", description="Recoverable energy per fission. Default 3.2044e-11 J (200 MeV).")

    material_name = "UN"
    optional_models = {}
    fission_gas_release_models = ("storms", "none")
    theoretical_density = 14326.0  # kg/m^3 at 298 K, Hayes et al. (1990, part I), Eq. (3)

    def __post_init__(self):
        _check_grain_radius(self)
        _check_fraction(self, "enrichment", 0.0, 1.0)
        _check_fraction(self, "theoretical_density_fraction", 0.5, 1.0, open_low=True)

    @property
    def heavy_metal_molar_mass(self) -> float:
        """kg/mol, from the enrichment."""
        return uranium_molar_mass(self.enrichment)

    @property
    def compound_molar_mass(self) -> float:
        """kg/mol of UN."""
        return self.heavy_metal_molar_mass + 14.007e-3

    def heavy_metal_atom_density(self) -> float:
        return self.theoretical_density_fraction * self.theoretical_density * AVOGADRO / self.compound_molar_mass

    def add_thermal_properties(self, problem, block, context):
        problem.add_property("UN_thermal", f"{block}_thermal", block=[block], temperature=context.temperature, theoretical_density_fraction=self.theoretical_density_fraction)

    def add_elasticity(self, problem, block, context):
        problem.add_property("UN_elasticity", f"{block}_elasticity", block=[block], temperature=context.temperature, theoretical_density_fraction=self.theoretical_density_fraction)

    def add_eigenstrains(self, problem, block, context):
        names = [f"{block}_thermal_strain"]
        problem.add_property("UN_thermal_expansion_eigenstrain", f"{block}_thermal_expansion", block=[block], temperature=context.temperature, stress_free_temperature=context.stress_free_temperature, eigenstrain_name=names[0], formulation=context.formulation)
        if context.models.solid_swelling:
            names.append(f"{block}_volumetric_strain")
            problem.add_property("UN_volumetric_swelling_eigenstrain", f"{block}_volumetric_swelling", block=[block], temperature=context.temperature, theoretical_density_fraction=self.theoretical_density_fraction, eigenstrain_name=names[-1], formulation=context.formulation, **context.burnup)
        return names

    def creep_parameters(self, context):
        return dict(creep_model="un", temperature=context.temperature, fission_rate="fission_rate", theoretical_density_fraction=self.theoretical_density_fraction, grain_radius=self.grain_radius, un_grain_boundary_creep=self.grain_boundary_creep)

    def fission_gas_model(self, num_elements, context):
        from .gas import StormsFissionGasRelease

        if context.models.fission_gas_release == "storms":
            return StormsFissionGasRelease(num_elements, theoretical_density_fraction=self.theoretical_density_fraction)
        return None


# ---------------------------------------------------------------------------
# Built-in cladding
# ---------------------------------------------------------------------------
@dataclass
class ZircaloyCladding(CladdingMaterial):
    """Zircaloy cladding.  The thermal conductivity, specific heat and
    thermal expansion are the IAEA-TECDOC-1496 recommendations for Zircaloy-2
    and Zircaloy-4; the elastic constants are MATPRO's; creep and growth are
    correlations for stress-relieved Zircaloy-4 (see the correlations chapter
    of the documentation for their verification status).  Zircaloy-2 is
    modelled with the same correlations."""

    alloy: str = parameter("zircaloy_4", description="zircaloy_4 or zircaloy_2 (modelled with the Zircaloy-4 correlations). Default zircaloy_4, the PWR cladding.")
    surface_roughness: float = parameter(1.0e-6, unit="m", description="Inner surface roughness. Default 1 um.")
    emissivity: float = parameter(0.8, description="Inner surface emissivity. Default 0.8, a nominal value for oxidised Zircaloy. MATPRO makes it depend on the oxide thickness.")
    meyer_hardness: float | str = parameter(
        "matpro", unit="Pa", description="Meyer hardness, for the solid contact conductance: a value in Pa, or 'matpro' for MATPRO CMHARD at the cladding's inner surface temperature (2 GPa at room temperature, 0.7 GPa at 600 K, 0.2 GPa at 875 K, NUREG/CR-6150 Vol. 4, Eq. 4-280). Default matpro."
    )
    cold_work: float = parameter(0.0, description="Cold work (ratio of areas) for the MATPRO elastic moduli, which it lowers by 26 GPa per unit. Default 0 (recrystallised). Set it for stress-relieved tubing.")

    material_name = "Zircaloy"

    def __post_init__(self):
        if isinstance(self.meyer_hardness, str) and self.meyer_hardness != "matpro":
            raise ValueError("ZircaloyCladding: meyer_hardness must be a number (Pa) or 'matpro'.")
        if self.alloy not in ("zircaloy_4", "zircaloy_2"):
            raise ValueError(f"ZircaloyCladding: unknown alloy '{self.alloy}'. Use zircaloy_4 or zircaloy_2.")

    def add_thermal_properties(self, problem, block, context):
        problem.add_property("Zircaloy_thermal", f"{block}_thermal", block=[block], temperature=context.temperature)

    def add_elasticity(self, problem, block, context):
        problem.add_property("Zircaloy_elasticity", f"{block}_elasticity", block=[block], temperature=context.temperature, fast_neutron_fluence="fast_neutron_fluence", cold_work=self.cold_work)

    def add_eigenstrains(self, problem, block, context):
        names = [f"{block}_thermal_strain", f"{block}_growth_strain"]
        problem.add_property("Zircaloy_thermal_expansion_eigenstrain", f"{block}_thermal_expansion", block=[block], temperature=context.temperature, stress_free_temperature=context.stress_free_temperature, eigenstrain_name=names[0], formulation=context.formulation)
        problem.add_property("Zircaloy_irradiation_growth_eigenstrain", f"{block}_irradiation_growth", block=[block], fast_neutron_fluence="fast_neutron_fluence", eigenstrain_name=names[1], formulation=context.formulation)
        return names

    def creep_parameters(self, context, block="clad"):
        return dict(creep_model="zircaloy", temperature=context.temperature, fast_neutron_flux="fast_neutron_flux", fast_neutron_fluence="fast_neutron_fluence")


# ---------------------------------------------------------------------------
# Materials defined by expressions
# ---------------------------------------------------------------------------
#: The fields of the rod that fuel expressions may use, besides the
#: temperature and the constants.
FUEL_FIELDS = ("burnup", "fission_rate", "fast_neutron_flux", "fast_neutron_fluence")
#: Those that cladding expressions may use.
CLADDING_FIELDS = ("fast_neutron_flux", "fast_neutron_fluence")
#: Element fields that property and eigenstrain expressions (not creep rates)
#: may use.
ELEMENT_FIELDS = (MAXIMUM_TEMPERATURE,)


class _ExpressionMaterial:
    """What CustomFuel and CustomCladding share: checking the expressions
    and adding them to a problem as parsed objects."""

    _fields: tuple = ()

    def _constants(self) -> dict:
        """Named constants every expression of this material may use."""
        return dict(self.constants)

    def _names(self, with_stress: bool = False) -> list[str]:
        if with_stress:
            return ["von_mises_stress", "temperature", *self._fields, *self._constants()]
        return ["temperature", *self._fields, *ELEMENT_FIELDS, *self._constants()]

    def _check_expressions(self, fields_and_stress: dict) -> None:
        """Compile every expression now, so that a mistake is reported with
        the material and the property it belongs to."""
        for name, with_stress in fields_and_stress.items():
            value = getattr(self, name)
            if value is None or isinstance(value, (int, float)):
                continue
            if not isinstance(value, str):
                raise TypeError(f"{type(self).__name__} '{self.name}': {name} must be a number or an expression (a string), not {type(value).__name__}.")
            try:
                _core.Expression(value, self._names(with_stress))
            except Exception as error:  # the compiler names the position and the problem
                raise ValueError(f"{type(self).__name__} '{self.name}': {name}: {error}") from None

    def _expression_arguments(self, text: str, context: RodContext) -> dict:
        """The arguments of a parsed object for an expression: only the
        fields, the temperature and the constants it uses."""
        used = set(_core.Expression(text, self._names()).used_variables)
        constants = {k: v for k, v in self._constants().items() if k in used}
        arguments = dict(coupled_variables=[context.temperature] if "temperature" in used else [], function_names=[f for f in self._fields if f in used], element_field_names=[f for f in ELEMENT_FIELDS if f in used], constant_names=list(constants), constant_values=[float(v) for v in constants.values()])
        if context.temperature != "temperature" and "temperature" in used:
            raise ValueError(f"{type(self).__name__} '{self.name}': expressions use the name 'temperature'; the rod's temperature variable must be called temperature.")
        return arguments

    def _add_property(self, problem, block, property_name, value, context):
        if isinstance(value, (int, float)):
            problem.add_property("constant_property", f"{block}_{property_name}", block=[block], property_names=[property_name], property_values=[float(value)])
        else:
            problem.add_property("parsed_property", f"{block}_{property_name}", block=[block], property_name=property_name, expression=value, **self._expression_arguments(value, context))

    def _add_eigenstrain(self, problem, block, name, value, strain_type, context, thermal=False, direction="isotropic"):
        text = str(value)
        arguments = self._expression_arguments(text, context)
        extra = {}
        if thermal:
            if "temperature" not in arguments["coupled_variables"]:
                raise ValueError(f"{type(self).__name__} '{self.name}': the thermal strain must depend on the temperature.")
            extra = dict(stress_free_state_names=["temperature"], stress_free_state_values=[context.stress_free_temperature])
        problem.add_property("parsed_eigenstrain", f"{block}_{name}", block=[block], expression=text, strain_type=strain_type, direction=direction, formulation=context.formulation, eigenstrain_name=f"{block}_{name}", **arguments, **extra)
        return f"{block}_{name}"

    def _creep(self, context, extra_constants: dict) -> dict:
        if self.creep_rate is None:
            return {}
        constants = {**self._constants(), **extra_constants}
        # The parsed creep of small_strain_stress knows these names itself.
        for known in ("theoretical_density_fraction", "grain_radius"):
            constants.pop(known, None)
        return dict(creep_model="parsed", creep_rate=str(self.creep_rate), temperature=context.temperature, creep_constant_names=list(constants), creep_constant_values=[float(v) for v in constants.values()], **{f: f for f in self._fields if f != "burnup"})


def _expression_field(description: str, required: bool = True, unit: str = ""):
    if required:
        return parameter(unit=unit, description=description)
    return parameter(None, unit=unit, description=description + " Default None: not modelled.")


@dataclass
class CustomFuel(_ExpressionMaterial, FuelMaterial):
    """A fuel whose properties are expressions.

    Every property is a number or an expression (a string) in the names
    ``temperature`` (K), ``maximum_temperature`` (K, the highest temperature
    the element has reached, kept by the rod; not in the creep rate),
    ``burnup`` (FIMA), ``fission_rate``
    (fissions/(m^3 s)), ``fast_neutron_flux`` (n/(m^2 s)),
    ``fast_neutron_fluence`` (n/m^2), the constants
    ``theoretical_density_fraction``, ``porosity`` (one minus it),
    ``grain_radius`` (m) and ``mwd_per_kg_per_fima`` (the MWd/kgHM in one
    FIMA, so ``burnup*mwd_per_kg_per_fima`` is the burnup in MWd/kgHM), and
    the user's ``constants``.  The creep rate may also use
    ``von_mises_stress`` (Pa).  For example, uranium monocarbide with a
    conductivity and a creep law of one's choice::

        uc = CustomFuel(
            name="UC",
            theoretical_density=13630.0,
            compound_molar_mass=0.250039,
            grain_radius=10.0e-6,
            thermal_conductivity="21.7 - 3.04e-3*temperature + 3.61e-6*temperature^2",
            specific_heat="...",
            youngs_modulus="...",
            poissons_ratio=0.28,
            thermal_strain="1.007e-5*temperature",
            creep_rate="A*von_mises_stress^2*exp(-Q/temperature)",
            constants={"A": ..., "Q": ...},
        )

    The numbers above are placeholders; take them from the literature for
    the fuel modelled.
    """

    name: str = parameter(description="Short name of the fuel, for reports (for instance UC).")
    theoretical_density: float = parameter(unit="kg/m^3", description="Theoretical density.")
    compound_molar_mass: float = parameter(unit="kg/mol", description="Molar mass of one formula unit (for instance UC).")
    thermal_conductivity: str | float = _expression_field("Thermal conductivity, W/(m K).")
    specific_heat: str | float = _expression_field("Specific heat, J/(kg K).")
    youngs_modulus: str | float | None = _expression_field("Young's modulus, Pa. Needed with mechanics.", required=False, unit="Pa")
    poissons_ratio: str | float | None = _expression_field("Poisson's ratio. Needed with mechanics.", required=False)
    thermal_strain: str | float | None = _expression_field("Linear thermal strain as a function of temperature, from any reference (it is shifted to zero at the stress-free temperature). Needed with mechanics.", required=False)
    volumetric_swelling: str | float | None = _expression_field("Volumetric swelling strain dV/V as a function of burnup and temperature (solid and gaseous together).", required=False)
    creep_rate: str | None = _expression_field("Equivalent creep strain rate, 1/s.", required=False)
    fission_gas_release: str = parameter("none", description="none, booth (diffusion out of the grains with fission_gas_diffusion_coefficient, then the grain-boundary saturation of BoothFissionGasRelease) or fraction (fission_gas_release_fraction of the gas produced). Default none.")
    fission_gas_diffusion_coefficient: str | None = _expression_field("With booth: the intragranular diffusion coefficient, m^2/s, of temperature and fission_rate.", required=False)
    fission_gas_release_fraction: str | None = _expression_field("With fraction: the released fraction of the gas produced, of temperature and burnup.", required=False)
    heavy_metal_atoms_per_formula_unit: int = parameter(1, description="Heavy-metal atoms in one formula unit: 1 for UC, UN, UO2 or UB2, 3 for U3Si2. Default 1.")
    heavy_metal_molar_mass: float = parameter(NATURAL_URANIUM_MOLAR_MASS, unit="kg/mol", description="Molar mass of the heavy metal. Default 0.238029, natural uranium. Set it for enriched uranium or another heavy metal. It enters the burnup only.")
    theoretical_density_fraction: float = parameter(0.95, description="Fabricated density as a fraction of theoretical. Default 0.95.")
    grain_radius: float | None = parameter(None, unit="m", description="Mean grain radius. Required, from the fabrication data of the fuel.")
    surface_roughness: float = parameter(2.0e-6, unit="m", description="Pellet surface roughness. Default 2 um.")
    emissivity: float = parameter(0.8, description="Pellet surface emissivity. Default 0.8.")
    energy_per_fission: float = parameter(DEFAULT_ENERGY_PER_FISSION, unit="J", description="Recoverable energy per fission. Default 3.2044e-11 J (200 MeV).")
    booth_parameters: dict = parameter(
        default_factory=dict, description="Keyword arguments for BoothFissionGasRelease besides the diffusion coefficient (for instance saturation_coverage, surface_energy, dihedral_half_angle, bubble_radius, fission_gas_yield). Default none: that model's own defaults, which are those of UO2."
    )
    constants: dict = parameter(default_factory=dict, description="Named constants the expressions use. Default none.")

    optional_models = {}
    _fields = FUEL_FIELDS

    @property
    def material_name(self) -> str:  # noqa: D401 - the dataclass field says it
        return self.name

    @property
    def fission_gas_release_models(self) -> tuple:
        return (self.fission_gas_release,)

    def __post_init__(self):
        _check_grain_radius(self)
        _check_fraction(self, "theoretical_density_fraction", 0.0, 1.0, open_low=True)
        if self.fission_gas_release not in ("none", "booth", "fraction"):
            raise ValueError(f"CustomFuel '{self.name}': fission_gas_release must be none, booth or fraction.")
        needed = {"booth": "fission_gas_diffusion_coefficient", "fraction": "fission_gas_release_fraction"}
        if self.fission_gas_release in needed and getattr(self, needed[self.fission_gas_release]) is None:
            raise ValueError(f"CustomFuel '{self.name}': fission_gas_release = {self.fission_gas_release} needs {needed[self.fission_gas_release]}.")
        reserved = {"temperature", *FUEL_FIELDS, *ELEMENT_FIELDS, "von_mises_stress", "porosity", "mwd_per_kg_per_fima"}
        clash = set(self.constants) & reserved
        if clash:
            raise ValueError(f"CustomFuel '{self.name}': constant name(s) {', '.join(sorted(clash))} are reserved for the fields of the rod.")
        self._check_expressions({"thermal_conductivity": False, "specific_heat": False, "youngs_modulus": False, "poissons_ratio": False, "thermal_strain": False, "volumetric_swelling": False, "creep_rate": True, "fission_gas_diffusion_coefficient": False, "fission_gas_release_fraction": False})

    def _constants(self) -> dict:
        from .units import mwd_per_kg_per_fima

        return {"theoretical_density_fraction": self.theoretical_density_fraction, "porosity": 1.0 - self.theoretical_density_fraction, "grain_radius": self.grain_radius, "mwd_per_kg_per_fima": mwd_per_kg_per_fima(self.energy_per_fission, self.heavy_metal_molar_mass), **self.constants}

    def heavy_metal_atom_density(self) -> float:
        return self.heavy_metal_atoms_per_formula_unit * self.theoretical_density_fraction * self.theoretical_density * AVOGADRO / self.compound_molar_mass

    def add_thermal_properties(self, problem, block, context):
        self._add_property(problem, block, "thermal_conductivity", self.thermal_conductivity, context)
        self._add_property(problem, block, "specific_heat", self.specific_heat, context)
        self._add_property(problem, block, "density", self.theoretical_density_fraction * self.theoretical_density, context)

    def add_elasticity(self, problem, block, context):
        for name in ("youngs_modulus", "poissons_ratio"):
            if getattr(self, name) is None:
                raise ValueError(f"CustomFuel '{self.name}': the mechanics needs {name}.")
            self._add_property(problem, block, name, getattr(self, name), context)

    def add_eigenstrains(self, problem, block, context):
        if self.thermal_strain is None:
            raise ValueError(f"CustomFuel '{self.name}': the mechanics needs thermal_strain.")
        names = [self._add_eigenstrain(problem, block, "thermal_strain", self.thermal_strain, "linear", context, thermal=True)]
        if self.volumetric_swelling is not None and context.models.solid_swelling:
            names.append(self._add_eigenstrain(problem, block, "volumetric_swelling", self.volumetric_swelling, "volumetric", context))
        return names

    def creep_parameters(self, context):
        params = self._creep(context, {})
        if params:
            params["theoretical_density_fraction"] = self.theoretical_density_fraction
            params["grain_radius"] = self.grain_radius
        return params

    def fission_gas_model(self, num_elements, context):
        from .gas import BoothFissionGasRelease, FractionFissionGasRelease

        release = context.models.fission_gas_release
        constants = self._constants()
        if release == "booth":
            expression = _core.Expression(str(self.fission_gas_diffusion_coefficient), ["temperature", "fission_rate", *constants])

            def diffusivity(temperature, fission_rate):
                return expression(temperature=temperature, fission_rate=fission_rate, **constants)

            return BoothFissionGasRelease(num_elements, grain_radius=self.grain_radius, trapping_factor=context.models.trapping_factor, trapping=context.models.intragranular_trapping, diffusion_coefficient=diffusivity, **self.booth_parameters)
        if release == "fraction":
            expression = _core.Expression(str(self.fission_gas_release_fraction), ["temperature", "burnup", *constants])

            def fraction(temperature, burnup):
                return expression(temperature=temperature, burnup=burnup, **constants)

            return FractionFissionGasRelease(num_elements, fraction)
        return None


@dataclass
class CustomCladding(_ExpressionMaterial, CladdingMaterial):
    """A cladding whose properties are expressions of ``temperature`` (K),
    ``maximum_temperature`` (K, the highest the element has reached; not in
    the creep rate), ``fast_neutron_flux`` (n/(m^2 s)),
    ``fast_neutron_fluence`` (n/m^2) and
    the user's ``constants``; the creep rate may also use
    ``von_mises_stress`` (Pa).  See :class:`CustomFuel`."""

    name: str = parameter(description="Short name of the cladding, for reports.")
    thermal_conductivity: str | float = _expression_field("Thermal conductivity, W/(m K).")
    specific_heat: str | float = _expression_field("Specific heat, J/(kg K).")
    density: float = parameter(unit="kg/m^3", description="Density.")
    youngs_modulus: str | float = _expression_field("Young's modulus, Pa.", unit="Pa")
    poissons_ratio: str | float = _expression_field("Poisson's ratio.")
    thermal_strain: str | float = _expression_field("Linear thermal strain as a function of temperature, from any reference (shifted to zero at the stress-free temperature), the same in every direction.")
    meyer_hardness: float = parameter(unit="Pa", description="Meyer hardness, for the solid contact conductance of the gap.")
    irradiation_growth: str | float | None = _expression_field("Axial irradiation growth strain, as a function of fast_neutron_fluence. The two transverse directions contract by half of it each, conserving volume.", required=False)
    volumetric_swelling: str | float | None = _expression_field("Volumetric swelling strain dV/V under irradiation, as a function of temperature and fast_neutron_fluence.", required=False)
    creep_rate: str | None = _expression_field("Equivalent creep strain rate, 1/s.", required=False)
    surface_roughness: float = parameter(1.0e-6, unit="m", description="Inner surface roughness. Default 1 um, a drawn tube.")
    emissivity: float = parameter(0.8, description="Inner surface emissivity. Default 0.8.")
    constants: dict = parameter(default_factory=dict, description="Named constants the expressions use. Default none.")

    _fields = CLADDING_FIELDS

    @property
    def material_name(self) -> str:
        return self.name

    def __post_init__(self):
        clash = set(self.constants) & {"temperature", *CLADDING_FIELDS, *ELEMENT_FIELDS, "von_mises_stress"}
        if clash:
            raise ValueError(f"CustomCladding '{self.name}': constant name(s) {', '.join(sorted(clash))} are reserved for the fields of the rod.")
        self._check_expressions({"thermal_conductivity": False, "specific_heat": False, "youngs_modulus": False, "poissons_ratio": False, "thermal_strain": False, "irradiation_growth": False, "volumetric_swelling": False, "creep_rate": True})

    def add_thermal_properties(self, problem, block, context):
        self._add_property(problem, block, "thermal_conductivity", self.thermal_conductivity, context)
        self._add_property(problem, block, "specific_heat", self.specific_heat, context)
        self._add_property(problem, block, "density", self.density, context)

    def add_elasticity(self, problem, block, context):
        self._add_property(problem, block, "youngs_modulus", self.youngs_modulus, context)
        self._add_property(problem, block, "poissons_ratio", self.poissons_ratio, context)

    def add_eigenstrains(self, problem, block, context):
        names = [self._add_eigenstrain(problem, block, "thermal_strain", self.thermal_strain, "linear", context, thermal=True)]
        if self.irradiation_growth is not None:
            growth = str(self.irradiation_growth)
            names.append(self._add_eigenstrain(problem, block, "growth_strain", growth, "linear", context, direction="axial"))
            names.append(self._add_eigenstrain(problem, block, "growth_contraction", f"-0.5*({growth})", "linear", context, direction="transverse"))
        if self.volumetric_swelling is not None:
            names.append(self._add_eigenstrain(problem, block, "swelling", self.volumetric_swelling, "volumetric", context))
        return names

    def creep_parameters(self, context, block="clad"):
        return self._creep(context, {})


def as_dict(spec) -> dict:
    """The fields of a material, for reports."""
    return {f.name: getattr(spec, f.name) for f in dataclasses.fields(spec)}
