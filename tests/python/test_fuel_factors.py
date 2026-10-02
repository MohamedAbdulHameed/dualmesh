# SPDX-License-Identifier: LGPL-2.1-or-later
"""The model factors of a fuel rod (fuel.ModelFactors) and the generic
property scaling of materials: each factor changes its target by the
expected amount, and a factor that cannot act on a rod is refused."""

from __future__ import annotations

import dualmesh as dm
import numpy as np
import pytest
from dualmesh import fuel

DAY = 86400.0
pytest.importorskip("scipy")
from scipy.integrate import quad  # noqa: E402


def _rod(factors=None, models=None, days=0.2, q=25e3, **numerics):
    geometry = fuel.RodGeometry.from_diameters(8.19e-3, 8.36e-3, 9.50e-3, 0.05)
    coolant = fuel.ForcedConvection(
        inlet_temperature=565.0, pressure=15.5e6, mass_flux=3800.0, rod_pitch=12.6e-3
    )
    history = fuel.PowerHistory(linear_heat_rate=[1e3, q, q], time=[0, 3600, days * DAY])
    rod = fuel.FuelRod(
        geometry,
        fuel.UO2Fuel(),
        fuel.ZircaloyCladding(),
        fuel.FillGas(pressure=2.0e6, plenum_volume=0.15e-6),
        coolant,
        history,
        models=models,
        numerics=fuel.RodNumerics(
            model="1.5d",
            mesh=fuel.RodMesh(num_axial_elements=2, num_axial_slices=1),
            max_time_step=10 * DAY,
            **numerics,
        ),
        output=fuel.RodOutput(print_input=False, print_steps=False),
        factors=factors,
    )
    return rod.run()


THERMAL = fuel.RodModels(mechanics=False, fission_gas_release="none")


def test_property_scaling_material_multiplies_a_property():
    """A conduction problem with k = 2 and a property_scaling of 1.5 gives
    the solution of k = 3."""

    def solve(k, factor=None):
        mesh = dm.generate_line_mesh(0.0, 1.0, 8)
        p = dm.Problem(mesh, method="fem")
        p.add_variable("u")
        p.add_material("generic_constant_material", "k", property_names=["k"], property_values=[k])
        if factor is not None:
            p.add_material(
                "property_scaling", "scale", scaled_properties=["k"], property_factors=[factor]
            )
        p.add_kernel("diffusion", variable="u", diffusivity_property="k")
        p.add_kernel("body_force", variable="u", value=1.0)
        p.add_boundary_condition(
            "Dirichlet_boundary_condition", variable="u", boundary="left", value=0.0
        )
        p.solve()
        return np.asarray(p.values("u"))

    assert np.allclose(solve(2.0, 1.5), solve(3.0), rtol=1e-12)
    mesh = dm.generate_line_mesh(0.0, 1.0, 2)
    p = dm.Problem(mesh)
    p.add_variable("u")
    with pytest.raises(Exception, match="same length"):
        p.add_material(
            "generic_constant_material",
            "k",
            property_names=["k"],
            property_values=[1.0],
            scaled_properties=["k"],
        )


def test_conductivity_factor_scales_the_conductivity_integral():
    """For a pellet heated uniformly, int_{T_s}^{T_c} k dT = q'/(4 pi): with
    k scaled by 1.1 the integral over the new temperatures, taken with the
    unscaled k, is 1/1.1 of the nominal one."""
    k = lambda T: float(fuel.properties.uo2_conductivity(T))  # noqa: E731
    nominal = _rod(models=THERMAL)
    scaled = _rod(fuel.ModelFactors(fuel_thermal_conductivity=1.1), models=THERMAL)
    integral = [
        quad(k, r.fuel_surface_temperature[-1][0], r.fuel_centerline_temperature[-1][0])[0]
        for r in (nominal, scaled)
    ]
    assert integral[0] / integral[1] == pytest.approx(1.1, rel=5e-3)
    # The surface temperature is set by the gap and the coolant, not the fuel.
    assert scaled.fuel_surface_temperature[-1][0] == pytest.approx(
        nominal.fuel_surface_temperature[-1][0], abs=0.01
    )


def test_power_gap_and_coolant_factors():
    nominal = _rod(models=THERMAL)
    power = _rod(fuel.ModelFactors(linear_heat_rate=1.05), models=THERMAL)
    assert power.rod_average_burnup[-1] == pytest.approx(
        1.05 * nominal.rod_average_burnup[-1], rel=1e-9
    )
    assert power.rod_average_linear_heat_rate[-1] == pytest.approx(
        1.05 * nominal.rod_average_linear_heat_rate[-1], rel=1e-9
    )
    gap = _rod(fuel.ModelFactors(gap_gas_conductance=1.5), models=THERMAL)
    drop = lambda r: r.fuel_surface_temperature[-1][0] - r.clad_inner_temperature[-1][0]  # noqa: E731
    # Gas conduction carries most of an open gap's heat and radiation the
    # rest, and the cooler gap gas conducts a little less, so the drop falls
    # by a little less than the factor.
    assert 1 / 1.5 < drop(gap) / drop(nominal) < 0.75
    film = _rod(fuel.ModelFactors(coolant_heat_transfer=2.0), models=THERMAL)
    rise = lambda r: r.clad_outer_temperature[-1][0] - r.coolant_temperature[-1][0]  # noqa: E731
    assert rise(film) / rise(nominal) == pytest.approx(0.5, rel=0.01)


def test_mechanical_and_gas_factors_act():
    base = _rod(days=60)
    dens = _rod(fuel.ModelFactors(densification=0.0), days=60)
    reloc = _rod(fuel.ModelFactors(relocation=2.0), days=60)
    # Without densification the pellet stays larger, so the gap is narrower.
    assert dens.gap_width[-1].mean() < base.gap_width[-1].mean()
    assert reloc.gap_width[-1].mean() < base.gap_width[-1].mean()
    hot = fuel.RodModels(fission_gas_release="booth")
    g0 = _rod(models=hot, days=150, q=40e3)
    g1 = _rod(fuel.ModelFactors(intragranular_diffusivity=10.0), models=hot, days=150, q=40e3)
    assert g1.fission_gas_release[-1] > g0.fission_gas_release[-1]


def test_booth_factors_equal_scaled_inputs():
    """The gas model with a diffusivity factor equals the model given the
    scaled diffusion coefficient, and the temperature factor equals the
    model run at the scaled temperature."""
    from dualmesh.fuel.gas import BoothFissionGasRelease

    T, F = np.array([1600.0, 1400.0]), np.array([1.2e19, 1.0e19])
    a = BoothFissionGasRelease(2, trapping="constant")
    a.set_factors(diffusivity=3.0, temperature=1.05)
    b = BoothFissionGasRelease(2, trapping="constant")
    D = lambda t, f: 3.0 * BoothFissionGasRelease(2, trapping="constant").diffusivity(t, f)  # noqa: E731
    b.diffusion_coefficient = D
    for step in range(20):
        burnup = np.full(2, 1e-3 * (step + 1))
        ra = a.advance(5 * DAY, T, F, 5e6, burnup)
        rb = b.advance(5 * DAY, 1.05 * T, F, 5e6, burnup)
    assert np.allclose(ra, rb, rtol=1e-12) and np.allclose(a.boundary, b.boundary, rtol=1e-12)


def test_factors_without_effect_are_refused():
    with pytest.raises(ValueError, match="densification"):
        _rod(fuel.ModelFactors(densification=0.8), models=fuel.RodModels(densification=False))
    with pytest.raises(ValueError, match="grain_radius"):
        _rod(fuel.ModelFactors(grain_radius=2.0), models=THERMAL)
    with pytest.raises(ValueError, match="must be finite and > 0"):
        fuel.ModelFactors(fuel_thermal_conductivity=0.0)
    assert fuel.ModelFactors(relocation=0.0).changed() == {"relocation": 0.0}
