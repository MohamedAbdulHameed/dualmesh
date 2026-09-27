# SPDX-License-Identifier: LGPL-2.1-or-later
"""Two ways to give a fuel rod a material the library does not have.

1. Change one correlation of a built-in fuel.  Subclass it and override the
   method that adds that correlation.  Here uranium nitride keeps every
   built-in property, and its creep law is replaced by a power law written as
   an expression.

2. Describe a whole fuel.  ``CustomFuel`` takes every property as an
   expression of the temperature, the burnup and the other fields of the rod.
   Here the fuel is uranium nitride again, written out from the correlations
   of Hayes, Thomas and Peddicord (J. Nucl. Mater. 171, 1990), the
   grain-boundary creep of AbdulHameed et al. (arXiv:2503.03231, 2025) and
   the irradiation creep of Konovalov et al. (2016), so that the result can
   be checked against the built-in ``UNFuel``: the two rods agree to
   rounding.  For a fuel the library does not have (uranium carbide, a
   uranium boride, a new silicide) the same form is filled with that fuel's
   correlations, taken from the literature.

Run it with ``python examples/custom_fuel.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

from dualmesh import fuel

DAY = 86400.0


# ---------------------------------------------------------------------------
# 1. A built-in fuel with one correlation replaced
# ---------------------------------------------------------------------------
@dataclass
class UNWithPowerLawCreep(fuel.UNFuel):
    """UN whose creep rate is A sigma^n exp(-Q/T) (sigma in Pa).  The
    constants are illustrative, not fitted to data."""

    creep_coefficient: float = 1.0e-35
    creep_exponent: float = 4.5
    creep_activation_temperature: float = 39369.5

    def creep_parameters(self, context):
        return dict(
            creep_model="parsed",
            creep_rate="A * von_mises_stress^n * exp(-Q / temperature)",
            creep_constant_names=["A", "n", "Q"],
            creep_constant_values=[
                self.creep_coefficient,
                self.creep_exponent,
                self.creep_activation_temperature,
            ],
            temperature=context.temperature,
        )


# ---------------------------------------------------------------------------
# 2. A fuel described entirely by expressions
# ---------------------------------------------------------------------------
def un_written_out() -> fuel.CustomFuel:
    un = fuel.UNFuel()
    return fuel.CustomFuel(
        name="UN (expressions)",
        theoretical_density=14330.0,
        compound_molar_mass=un.compound_molar_mass,
        heavy_metal_molar_mass=un.heavy_metal_molar_mass,
        thermal_conductivity="1.864*exp(-2.14*porosity)*temperature^0.361",
        specific_heat=(
            "(51.14*(365.7/temperature)^2*exp(365.7/temperature)"
            "/(exp(365.7/temperature) - 1)^2 + 9.491e-3*temperature"
            " + 2.642e11/temperature^2*exp(-18081/temperature))/0.25204"
        ),
        youngs_modulus=(
            "1e6*0.258*(100*theoretical_density_fraction)^3.002*(1 - 2.375e-5*temperature)"
        ),
        poissons_ratio="1.26e-3*(100*theoretical_density_fraction)^1.174",
        thermal_strain="(7.096e-6 + 1.409e-9*temperature)*(temperature - 298)",
        creep_rate=(
            "2.054e-3*(von_mises_stress/1e6)^4.5*exp(-39369.5/temperature)"
            " + 582610.427*(von_mises_stress/1e6)/(temperature*(2e6*grain_radius)^3)"
            "*exp(-2.28/(8.617333262e-5*temperature))"
            " + 2.9e-22*(von_mises_stress/1e6)*(fission_rate*1e-6)*exp(20*porosity)/3600"
        ),
    )


def run(material, verbose=False):
    rod = fuel.FuelRod(
        geometry=fuel.RodGeometry.from_diameters(
            pellet_outer_diameter=8.19e-3,
            clad_inner_diameter=8.36e-3,
            clad_outer_diameter=9.50e-3,
            fuel_stack_height=0.05,
        ),
        fuel=material,
        cladding=fuel.ZircaloyCladding(),
        fill_gas=fuel.FillGas(pressure=2.0e6, plenum_volume=0.15e-6),
        coolant=fuel.ForcedConvection(
            inlet_temperature=565.0, pressure=15.5e6, mass_flux=3800.0, rod_pitch=12.6e-3
        ),
        power_history=fuel.PowerHistory(
            linear_heat_rate=[1e3, 40e3, 40e3], time=[0.0, 3600.0, 100 * DAY]
        ),
        # The built-in UN swelling and gas release are tabulated correlations
        # the expression version leaves out, so both rods run without them.
        models=fuel.RodModels(solid_swelling=False, fission_gas_release="none"),
        numerics=fuel.RodNumerics(max_time_step=20 * DAY),
        output=fuel.RodOutput(print_input=verbose, print_steps=verbose),
    )
    return rod.run()


if __name__ == "__main__":
    built_in = run(fuel.UNFuel())
    written_out = run(un_written_out(), verbose=True)
    power_law = run(UNWithPowerLawCreep())
    print()
    print("peak centerline temperature, K")
    print(f"  UNFuel                  {built_in.fuel_centerline_temperature.max():.6f}")
    print(f"  CustomFuel (same laws)  {written_out.fuel_centerline_temperature.max():.6f}")
    print(f"  UN with power-law creep {power_law.fuel_centerline_temperature.max():.6f}")
