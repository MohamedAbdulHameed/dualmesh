# SPDX-License-Identifier: LGPL-2.1-or-later
"""Reddy, Section 7.5: bending of a functionally graded beam.

A pinned-pinned beam of length L = 100 in under a uniform load is analysed
with the three dual mesh beam models, for several values of the power-law
index n of the through-thickness grading.  Half the beam is modelled.
"""

import dualmesh as dm

LENGTH, WIDTH, HEIGHT = 100.0, 1.0, 1.0
MODULUS_TOP, MODULUS_BOTTOM, POISSON = 30.0e6, 3.0e6, 0.3
LOAD = 0.5
MODELS = ("BeamEulerBernoulliMixed", "BeamTimoshenkoDisplacement", "BeamTimoshenkoMixed")


def solve(model, power_law_index, num_elements=16):
    stiffness = dm.fgm.beam_stiffness(
        modulus_top=MODULUS_TOP,
        modulus_bottom=MODULUS_BOTTOM,
        power_law_index=power_law_index,
        height=HEIGHT,
        width=WIDTH,
        poisson_ratio=POISSON,
    )
    mesh = dm.generate_line_mesh(start=0.0, end=LENGTH / 2, num_elements=num_elements)
    problem = dm.Problem(mesh)
    variables = dm.physics.add_beam(
        problem,
        model=model,
        extensional_stiffness=stiffness.extensional,
        coupling_stiffness=stiffness.coupling,
        bending_stiffness=stiffness.bending,
        shear_stiffness=stiffness.shear,
        transverse_load=LOAD,
    )
    third = variables[2]
    problem.add_boundary_condition(
        "DirichletBC", "pin_axial", variable="axial_displacement", boundary="left", value=0.0
    )
    problem.add_boundary_condition(
        "DirichletBC", "pin_deflection", variable="deflection", boundary="left", value=0.0
    )
    if third == "bending_moment":
        problem.add_boundary_condition(
            "DirichletBC", "pin_moment", variable=third, boundary="left", value=0.0
        )
    else:
        problem.add_boundary_condition(
            "DirichletBC", "symmetry_rotation", variable=third, boundary="right", value=0.0
        )
    problem.add_boundary_condition(
        "DirichletBC", "symmetry_axial", variable="axial_displacement", boundary="right", value=0.0
    )
    problem.solve()
    return problem, stiffness


if __name__ == "__main__":
    print("   n   " + "".join(f"{m.replace('Beam', ''):>26s}" for m in MODELS))
    for power_law_index in (0.0, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0):
        row = []
        for model in MODELS:
            problem, stiffness = solve(model, power_law_index)
            reduced = stiffness.bending - stiffness.coupling**2 / stiffness.extensional
            centre = problem.values("deflection")[-1]
            row.append(centre * reduced / (LOAD * LENGTH**4) * 10)
        print(f"{power_law_index:5.1f}" + "".join(f"{value:26.4f}" for value in row))
    print("(normalized centre deflection w Dhat / (q0 L^4) x 10, Table 7.5.3)")
