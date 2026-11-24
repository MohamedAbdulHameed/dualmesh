# SPDX-License-Identifier: LGPL-2.1-or-later
"""A large plate with a circular hole under uniaxial tension.

One quarter of the plate is modelled with an annular mesh graded towards the
hole.  The classical stress concentration factor of three is recovered at the
top of the hole.  The result is written as a VTK file with the stress
components as cell data.
"""

import dualmesh
import numpy

hole_radius, outer_radius = 1.0, 20.0


def solve(num_radial_elements=30, num_angular_elements=30):
    mesh = dualmesh.generate_annulus_mesh(
        inner_radius=hole_radius,
        outer_radius=outer_radius,
        num_radial_elements=num_radial_elements,
        num_angular_elements=num_angular_elements,
        radial_bias=1.15,
    )
    problem = dualmesh.Problem(mesh)
    solid = problem.add_physics(
        "solid_mechanics",
        formulation="plane_stress",
        youngs_modulus=1.0,
        poissons_ratio=0.3,
    )
    # Symmetry on the two straight edges.
    solid.add_boundary_condition("symmetry", boundary="start")
    solid.add_boundary_condition("symmetry", boundary="end")
    # Remote uniaxial tension on the outer arc: t = (sigma n_x, 0).
    # The far-field stress sigma_xx = 1, applied on the outer circle.
    solid.add_boundary_condition("traction = [x/sqrt(x^2 + y^2), 0]", boundary="outer")
    problem.solve()
    return problem


if __name__ == "__main__":
    problem = solve()
    stress = problem.property_at_centroids("stress")
    centroids = numpy.array(
        [problem.mesh.element_centroid(e) for e in range(problem.mesh.num_elements)]
    )
    radius = numpy.hypot(centroids[:, 0], centroids[:, 1])
    at_hole = radius < hole_radius * 1.05
    print(
        f"maximum sigma_xx near the hole: {numpy.max(stress[at_hole, 0]):.3f}  (exact stress concentration factor: 3)"
    )
    problem.write_vtu("plate_with_hole.vtu", cell_properties=["stress"])
    print("wrote plate_with_hole.vtu")
