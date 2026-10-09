# SPDX-License-Identifier: LGPL-2.1-or-later
"""The criticality of a bare reactor, written as a system of two diffusion equations.

dualmesh has no neutronics module.
This example writes two-group neutron diffusion with coefficient_form_PDE and finds the effective multiplication factor k with the eigenvalue study of any problem.
The equations of the fast flux phi1 and the thermal flux phi2 are

    -div(D1 grad phi1) + (Sigma_a1 + Sigma_12) phi1 = (1 / k) nu Sigma_f2 phi2,
    -div(D2 grad phi2) + Sigma_a2 phi2 - Sigma_12 phi1 = 0,

where the fission neutrons are born fast.
The symbol eigenvalue of the expressions is 1 / k, so the smallest eigenvalue gives the largest k.
The core is a square of side 2 m with zero flux on its boundary, and the cross sections are those of fuel 1 of the IAEA PWR benchmark (ANL-7416 Suppl. 2, problem 11) in SI units.
For a bare core of geometrical buckling B^2 = 2 (pi / L)^2, two-group theory gives k exactly (J. C. Lee, Nuclear Reactor Physics and Engineering, 2nd ed., Wiley 2025, Eq. 7.26):

    k = nu Sigma_f2 Sigma_12 / ((Sigma_a1 + Sigma_12 + D1 B^2) (Sigma_a2 + D2 B^2)).

Run it with python reactor_criticality.py.
"""

from __future__ import annotations

import math

import dualmesh as dm

SIDE = 2.0  # m
CROSS_SECTIONS = {
    "D1": 0.015,  # m
    "D2": 0.004,  # m
    "Sigma_a1": 1.0,  # 1/m
    "Sigma_a2": 8.0,  # 1/m
    "Sigma_12": 2.0,  # 1/m
    "nu_Sigma_f2": 13.5,  # 1/m
}


def criticality(num_elements: int = 40, method: str = "fem") -> float:
    """The effective multiplication factor of the bare core."""
    mesh = dm.generate_rectangle_mesh(
        x_min=0.0,
        x_max=SIDE,
        y_min=0.0,
        y_max=SIDE,
        num_x_elements=num_elements,
        num_y_elements=num_elements,
    )
    problem = dm.Problem(mesh, method=method)
    diffusion = {"phi1": "D1", "phi2": "D2"}
    absorption = {
        "phi1": "Sigma_a1 + Sigma_12",
        "phi2": {"phi2": "Sigma_a2", "phi1": "-Sigma_12"},
    }
    fission = {"phi1": "eigenvalue*nu_Sigma_f2*phi2"}
    neutrons = problem.add_physics(
        "coefficient_form_PDE",
        "neutrons",
        variables=["phi1", "phi2"],
        diffusion_coefficient=diffusion,
        absorption_coefficient=absorption,
        source=fission,
        constants=CROSS_SECTIONS,
        units={"phi1": "1/(m^2 s)", "phi2": "1/(m^2 s)"},
    )
    for flux in ("phi1", "phi2"):
        neutrons.add_boundary_condition(
            "Dirichlet_boundary_condition",
            f"zero_{flux}",
            variable=flux,
            boundary=list(mesh.sideset_names()),
            value=0.0,
        )
    result = problem.solve_eigenvalue(num_modes=1, report="none")
    return 1.0 / float(result.eigenvalues[0])


def exact_criticality() -> float:
    c = CROSS_SECTIONS
    buckling = 2.0 * (math.pi / SIDE) ** 2
    fast = c["Sigma_a1"] + c["Sigma_12"] + c["D1"] * buckling
    thermal = c["Sigma_a2"] + c["D2"] * buckling
    return c["nu_Sigma_f2"] * c["Sigma_12"] / (fast * thermal)


if __name__ == "__main__":
    k, exact = criticality(), exact_criticality()
    print(f"k = {k:.6f}, exact {exact:.6f}, difference {1e5 * (k - exact):+.1f} pcm")
