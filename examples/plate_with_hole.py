# SPDX-License-Identifier: LGPL-2.1-or-later
"""A large plate with a circular hole under uniaxial tension.

One quarter of the plate is modelled with an annular mesh graded towards the
hole.  The classical stress concentration factor of three is recovered at the
top of the hole.  The result is written as a VTK file with the stress
components as cell data.
"""

import math

import dualmesh as dm
import numpy as np

HOLE_RADIUS, OUTER_RADIUS, APPLIED_STRESS = 1.0, 20.0, 1.0


def solve(num_radial_elements=30, num_angular_elements=30):
    mesh = dm.generate_annulus_mesh(
        inner_radius=HOLE_RADIUS,
        outer_radius=OUTER_RADIUS,
        num_radial_elements=num_radial_elements,
        num_angular_elements=num_angular_elements,
        radial_bias=1.15,
    )
    problem = dm.Problem(mesh)
    dm.physics.add_plane_elasticity(
        problem, displacements=["u", "v"], youngs_modulus=1.0, poissons_ratio=0.3
    )
    # symmetry on the two straight edges
    problem.add_boundary_condition(
        "DirichletBC", "sym_v", variable="v", boundary="start", value=0.0
    )
    problem.add_boundary_condition("DirichletBC", "sym_u", variable="u", boundary="end", value=0.0)
    # remote uniaxial tension applied on the outer arc: t = (sigma n_x, 0)
    problem.add_boundary_condition(
        "TractionBC",
        "remote_x",
        variable="u",
        boundary="outer",
        traction=lambda x, y, z, t: APPLIED_STRESS * x / math.hypot(x, y),
    )
    problem.add_boundary_condition(
        "TractionBC", "remote_y", variable="v", boundary="outer", traction=0.0
    )
    problem.solve()
    return problem


if __name__ == "__main__":
    problem = solve()
    stress = problem.property_at_centroids("stress")
    centroids = np.array(
        [problem.mesh.element_centroid(e) for e in range(problem.mesh.num_elements)]
    )
    radius = np.hypot(centroids[:, 0], centroids[:, 1])
    at_hole = radius < HOLE_RADIUS * 1.05
    print(
        f"maximum sigma_xx near the hole: {np.max(stress[at_hole, 0]):.3f}"
        f"  (exact stress concentration factor: 3)"
    )
    problem.write_vtu("plate_with_hole.vtu", cell_properties=["stress"])
    print("wrote plate_with_hole.vtu")
