# SPDX-License-Identifier: LGPL-2.1-or-later
"""The criticality of a bare reactor, written as two models of your own.

dualmesh has no neutronics module. This example writes two-group neutron diffusion as two models,
one for the fast flux phi_1 and one for the thermal flux phi_2, both in 1/(m^2 s), and finds the
effective multiplication factor k with an eigenvalue study. The two balances are

    -div(D_1 grad phi_1) + (Sigma_a1 + Sigma_12) phi_1 = (1/k) nu Sigma_f2 phi_2,
    -div(D_2 grad phi_2) + Sigma_a2 phi_2 - Sigma_12 phi_1 = 0,

where the fission neutrons are born fast. The source of the fast model reads the symbol
eigenvalue as 1/k. The problem has no time derivative, so the eigenvalue is a number, and the
smallest eigenvalue gives the largest k. The core is a square of side 2 m with zero flux on its
boundary, and the cross sections are those of fuel 1 of the IAEA PWR benchmark (ANL-7416 Suppl. 2,
problem 11). For a bare core of geometrical buckling B^2 = 2 (pi / L)^2, two-group theory gives k
exactly (J. C. Lee, Nuclear Reactor Physics and Engineering, 2nd ed., Wiley 2025, Eq. 7.26):

    k = nu Sigma_f2 Sigma_12 / ((Sigma_a1 + Sigma_12 + D_1 B^2) (Sigma_a2 + D_2 B^2)).

Run it with python reactor_criticality.py.
"""

import math

import dualmesh

side = 2.0  # m
#: The cross sections, each with its value in its unit.
cross_sections = {
    "fast_diffusion_coefficient": (0.015, "m"),
    "thermal_diffusion_coefficient": (0.004, "m"),
    "fast_absorption": (1.0, "1/m"),
    "thermal_absorption": (8.0, "1/m"),
    "fast_to_thermal_scattering": (2.0, "1/m"),
    "thermal_fission_production": (13.5, "1/m"),
}
flux_unit = {"unit": "1/(m^2*s)"}


def core(cells: int = 40):
    """The bare square core on cells by cells squares."""
    reactor = dualmesh.problem(
        mesh={"rectangle": {"size": [f"{side} m", f"{side} m"], "cells": [cells, cells]}}
    )
    reactor.model(
        "fast_group",
        unknown={"fast_flux": flux_unit},
        flux="-fast_diffusion_coefficient*grad(fast_flux)",
        source="-(fast_absorption + fast_to_thermal_scattering)*fast_flux"
        " + eigenvalue*thermal_fission_production*thermal_flux",
    )
    reactor.model(
        "thermal_group",
        unknown={"thermal_flux": flux_unit},
        flux="-thermal_diffusion_coefficient*grad(thermal_flux)",
        source="-thermal_absorption*thermal_flux + fast_to_thermal_scattering*fast_flux",
    )
    reactor.properties(
        **{name: {"unit": unit, "value": value} for name, (value, unit) in cross_sections.items()}
    )
    for boundary in ("left", "right", "bottom", "top"):
        reactor.boundary_conditions(boundary, fast_flux="0 1/(m^2*s)", thermal_flux="0 1/(m^2*s)")
    reactor.eigenvalues(count=1)
    return reactor


def criticality(cells: int = 40) -> float:
    """The effective multiplication factor of the bare core, 1 over its smallest eigenvalue."""
    return 1.0 / float(core(cells).solve(report="none").eigenvalues[0])


def exact_criticality() -> float:
    c = {name: value for name, (value, _) in cross_sections.items()}
    buckling = 2.0 * (math.pi / side) ** 2
    fast = c["fast_absorption"] + c["fast_to_thermal_scattering"]
    fast += c["fast_diffusion_coefficient"] * buckling
    thermal = c["thermal_absorption"] + c["thermal_diffusion_coefficient"] * buckling
    return c["thermal_fission_production"] * c["fast_to_thermal_scattering"] / (fast * thermal)


if __name__ == "__main__":
    k, exact = criticality(), exact_criticality()
    print(f"k = {k:.6f}, exact {exact:.6f}, difference {1e5 * (k - exact):+.1f} pcm")
