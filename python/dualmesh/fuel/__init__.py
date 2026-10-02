# SPDX-License-Identifier: LGPL-2.1-or-later
"""Nuclear fuel performance: the thermal and mechanical behaviour of a fuel
rod under a power history, and the correlations of the fuel and cladding
materials.

A rod is described by input groups, each a dataclass, and run by
:class:`FuelRod`::

    from dualmesh import fuel

    rod = fuel.FuelRod(
        geometry=fuel.RodGeometry.from_diameters(8.19e-3, 8.36e-3, 9.50e-3, 0.1),
        fuel=fuel.UO2Fuel(enrichment=0.045),
        cladding=fuel.ZircaloyCladding(),
        fill_gas=fuel.FillGas(pressure=2.0e6, plenum_volume=0.3e-6),
        coolant=fuel.ForcedConvection(
            inlet_temperature=565.0, pressure=15.5e6, mass_flux=3800.0, rod_pitch=12.6e-3
        ),
        power_history=fuel.PowerHistory(
            linear_heat_rate=[20e3, 20e3], burnup=[0.0, 30.0], burnup_unit="MWd/kgHM"
        ),
    )
    result = rod.run()

See the documentation chapter on fuel performance for the physics, the
source of every correlation, the verification and the benchmarks.
"""

from .._core import fuel as properties
from . import dpa, triso, water
from .atf import (
    ChromiumCoating,
    CoatedCladding,
    DopedUO2Fuel,
    FeCrAlCladding,
    SiCCladding,
    U3Si2Fuel,
)
from .fields import IrradiationFields, TimeHistory, irradiation_fields
from .gas import (
    BoothFissionGasRelease,
    FractionFissionGasRelease,
    GrainFaceBubbles,
    StormsFissionGasRelease,
)
from .materials import (
    CladdingMaterial,
    CustomCladding,
    CustomFuel,
    FuelMaterial,
    RodContext,
    UNFuel,
    UO2Fuel,
    ZircaloyCladding,
)
from .mesh import (
    axisymmetric_rod_mesh,
    graded_points,
    radial_slice_mesh,
    three_dimensional_rod_mesh,
)
from .result import RodResult
from .rod import FuelRod
from .specification import (
    FillGas,
    ForcedConvection,
    ModelFactors,
    PowerHistory,
    PrescribedCladdingTemperature,
    RodGeometry,
    RodMesh,
    RodModels,
    RodNumerics,
    RodOutput,
    describe,
)
from .units import BURNUP_UNITS, BurnupConverter

__all__ = [
    "BURNUP_UNITS",
    "ChromiumCoating",
    "CoatedCladding",
    "DopedUO2Fuel",
    "FeCrAlCladding",
    "SiCCladding",
    "U3Si2Fuel",
    "BoothFissionGasRelease",
    "GrainFaceBubbles",
    "triso",
    "water",
    "dpa",
    "BurnupConverter",
    "CladdingMaterial",
    "CustomCladding",
    "CustomFuel",
    "FractionFissionGasRelease",
    "FuelMaterial",
    "RodContext",
    "FillGas",
    "ForcedConvection",
    "ModelFactors",
    "FuelRod",
    "IrradiationFields",
    "PowerHistory",
    "PrescribedCladdingTemperature",
    "RodGeometry",
    "RodMesh",
    "RodModels",
    "RodNumerics",
    "RodOutput",
    "RodResult",
    "StormsFissionGasRelease",
    "TimeHistory",
    "UNFuel",
    "UO2Fuel",
    "ZircaloyCladding",
    "axisymmetric_rod_mesh",
    "describe",
    "graded_points",
    "irradiation_fields",
    "properties",
    "radial_slice_mesh",
    "three_dimensional_rod_mesh",
]
