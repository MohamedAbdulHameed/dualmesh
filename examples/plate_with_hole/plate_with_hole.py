# SPDX-License-Identifier: LGPL-2.1-or-later
"""A large steel plate with a circular hole under uniaxial tension.

A plate with a hole of radius a = 10 mm is pulled far from the hole by the stress
sigma = 100 MPa along x. Two planes of symmetry, x = 0 and y = 0, leave a quarter of the
plate, an annulus of outer radius 200 mm, on the mesh of make_plate_mesh.py, which is graded
towards the hole. The outer arc carries the traction of the remote stress, (sigma n_x, 0), with
n_x = x / r. In an infinite plate the stress on the y axis is the solution of Kirsch,

    sigma_xx(0, y) = sigma (1 + a^2 / (2 y^2) + 3 a^4 / (2 y^4)),

which is 3 sigma at the top of the hole: the stress concentration factor is 3.
"""

from pathlib import Path

import dualmesh

remote_stress, hole_radius = 100.0e6, 0.01  # Pa, m


def plate():
    quarter = dualmesh.problem(
        mesh={"file": str(Path(__file__).with_name("plate_with_hole.msh")), "length_unit": "mm"}
    )
    quarter.physics("solid_mechanics", formulation="plane_stress")
    quarter.properties(Young_modulus="200 GPa", Poisson_ratio=0.3)
    quarter.boundary_conditions("left", "symmetry")
    quarter.boundary_conditions("bottom", "symmetry")
    quarter.boundary_conditions("outer", traction=["[100 MPa]*x/sqrt(x^2 + y^2)", "0 Pa"])
    quarter.postprocessor("peak", point_value="stress_xx", point=["0 m", "10 mm"])
    quarter.postprocessor(
        "ligament", line="stress_xx", from_=["0 m", "10 mm"], to=["0 m", "50 mm"], points=5
    )
    quarter.output(file="plate_with_hole", fields=["displacement"])
    return quarter


def Kirsch(y):
    ratio = (hole_radius / y) ** 2
    return remote_stress * (1.0 + ratio / 2.0 + 1.5 * ratio**2)


if __name__ == "__main__":
    solution = plate().solve()
    peak = solution.postprocessor("peak")[-1]
    print(f"stress concentration factor: {peak / remote_stress:.3f} (Kirsch: 3)")
    ligament = solution.line("ligament")
    print("   y (mm)   stress_xx (MPa)   Kirsch (MPa)")
    for y, stress in zip(ligament["y"], ligament["stress_xx"]):
        print(f"{1e3 * y:8.1f}  {stress / 1e6:14.2f}  {Kirsch(y) / 1e6:13.2f}")
    print("wrote plate_with_hole.vtu, plate_with_hole.csv and plate_with_hole_ligament.csv")
