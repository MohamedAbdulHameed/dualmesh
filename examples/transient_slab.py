# SPDX-License-Identifier: LGPL-2.1-or-later
"""Transient conduction in a slab, compared with the analytical series.

A slab of thickness L, initially at T = 1, is held at T = 0 on both faces from
t = 0.  The exact solution is the Fourier series

    T(x, t) = sum_{n odd} (4 / (n pi)) sin(n pi x / L) exp(-(n pi / L)^2 alpha t).

The example integrates with the Crank-Nicolson rule (theta = 1/2).
"""

import dualmesh as dm
import numpy as np

LENGTH, DIFFUSIVITY, END_TIME = 1.0, 1.0, 0.05


def exact(x, t, terms=200):
    total = np.zeros_like(x)
    for n in range(1, 2 * terms, 2):
        total += (
            (4.0 / (n * np.pi))
            * np.sin(n * np.pi * x / LENGTH)
            * np.exp(-((n * np.pi / LENGTH) ** 2) * DIFFUSIVITY * t)
        )
    return total


def solve(num_elements=40, dt=0.001, theta=0.5):
    mesh = dm.generate_line_mesh(start=0.0, end=LENGTH, num_elements=num_elements)
    problem = dm.Problem(mesh)
    problem.add_variable("temperature", initial_condition=1.0)
    problem.add_kernel("HeatConduction", variable="temperature", thermal_conductivity=DIFFUSIVITY)
    problem.add_kernel(
        "HeatConductionTimeDerivative", variable="temperature", density=1.0, specific_heat=1.0
    )
    problem.add_boundary_condition(
        "DirichletBC", variable="temperature", boundary=["left", "right"], value=0.0
    )
    problem.solve_transient(end_time=END_TIME, dt=dt, theta=theta)
    return problem


if __name__ == "__main__":
    problem = solve()
    x = np.linspace(0.05, 0.95, 10)
    computed = problem.sample("temperature", x.reshape(-1, 1))
    reference = exact(x, END_TIME)
    print("    x     computed      exact     error")
    for xi, mine, ref in zip(x, computed, reference):
        print(f"{xi:6.2f}  {mine:10.6f}  {ref:9.6f}  {abs(mine - ref):8.2e}")
    print(f"maximum error: {np.max(np.abs(computed - reference)):.2e}")
