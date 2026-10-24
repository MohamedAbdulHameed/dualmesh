# SPDX-License-Identifier: LGPL-2.1-or-later
"""Flow in a lid-driven cavity by the penalty method.

The velocity of the lid is applied in load steps, because Newton's method
started from rest does not converge at Re = 1000.  The horizontal velocity
along the vertical centreline is compared with the published dual mesh values.
"""

import dualmesh as dm
import numpy as np

PUBLISHED_Y = [
    0.0625,
    0.125,
    0.1875,
    0.25,
    0.3125,
    0.375,
    0.4375,
    0.5,
    0.5625,
    0.625,
    0.6875,
    0.75,
    0.7813,
    0.8125,
    0.8438,
    0.875,
    0.9063,
    0.9375,
    0.9688,
]
PUBLISHED_U = [
    -0.1430,
    -0.2177,
    -0.2502,
    -0.2301,
    -0.1776,
    -0.1255,
    -0.0803,
    -0.0365,
    0.0091,
    0.0570,
    0.1097,
    0.1612,
    0.1861,
    0.2084,
    0.2273,
    0.2383,
    0.2496,
    0.2937,
    0.5245,
]


def solve(reynolds_number=1000.0):
    mesh = dm.generate_rectangle_mesh(
        x_coordinates=dm.meshing.coordinates_from_spacings(0.0, [0.0625] * 16),
        y_coordinates=dm.meshing.coordinates_from_spacings(0.0, [0.0625] * 12 + [0.03125] * 8),
    )
    problem = dm.Problem(mesh)
    # Re = rho V0 a / mu with V0 = a = mu = 1, so the density is the Reynolds number.
    flow = problem.add_physics(
        "fluid_dynamics",
        dynamic_viscosity=1.0,
        density=reynolds_number,
        penalty_parameter=1.0e8,
    )
    flow.add_boundary_condition(
        ["velocity_x = 0", "velocity_y = 0"], boundary=["left", "right", "bottom"]
    )
    flow.add_boundary_condition(
        ["velocity_x = 1", "velocity_y = 0"], boundary="top", scale_with_load=True
    )
    problem.solve(load_factors=[0.1, 0.25, 0.5, 0.75, 1.0], max_iterations=60)
    return problem


if __name__ == "__main__":
    problem = solve()
    points = np.column_stack([np.full(len(PUBLISHED_Y), 0.5), PUBLISHED_Y])
    computed = problem.sample("velocity_x", points)
    print("    y      velocity_x (dualmesh)   velocity_x (published)")
    for y, mine, published in zip(PUBLISHED_Y, computed, PUBLISHED_U):
        print(f"{y:7.4f}  {mine:12.4f}  {published:9.4f}")
    problem.write_vtu("cavity.vtu", cell_properties=["pressure"])
    print("wrote cavity.vtu (velocity components and recovered pressure)")
