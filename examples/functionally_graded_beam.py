# SPDX-License-Identifier: LGPL-2.1-or-later
"""Bending of a functionally graded beam.

A pinned-pinned beam of length L = 100 in under a uniform load is analysed
with the three dual mesh beam models, for several values of the power-law
index n of the through-thickness grading.  Half the beam is modelled.
"""

import dualmesh as dm

LENGTH, WIDTH, HEIGHT = 100.0, 1.0, 1.0
MODULUS_TOP, MODULUS_BOTTOM, POISSON = 30.0e6, 3.0e6, 0.3
LOAD = 0.5
# The theory and, for Timoshenko, the form of each beam model.
MODELS = {
    "Euler-Bernoulli": ("Euler_Bernoulli", None),
    "Timoshenko, displacement": ("Timoshenko", "displacement"),
    "Timoshenko, mixed": ("Timoshenko", None),
}


def solve(model, power_law_index, num_elements=16):
    stiffness = dm.fgm.beam_stiffness(
        modulus_top=MODULUS_TOP,
        modulus_bottom=MODULUS_BOTTOM,
        power_law_index=power_law_index,
        height=HEIGHT,
        width=WIDTH,
        poissons_ratio=POISSON,
    )
    mesh = dm.generate_line_mesh(start=0.0, end=LENGTH / 2, num_elements=num_elements)
    problem = dm.Problem(mesh)
    theory, form = MODELS[model]
    beam = problem.add_physics(
        "solid_mechanics",
        formulation="beam",
        theory=theory,
        form=form,
        extensional_stiffness=stiffness.extensional,
        coupling_stiffness=stiffness.coupling,
        bending_stiffness=stiffness.bending,
        # The Euler-Bernoulli beam has no shear deformation, so it takes no shear stiffness.
        **({"shear_stiffness": stiffness.shear} if theory == "Timoshenko" else {}),
        transverse_load=LOAD,
    )
    # Pinned at the left end, where the physics holds the moment at zero, and the right end is
    # the centre of the beam, a plane of symmetry.
    beam.add_boundary_condition(["disp_x = 0", "disp_z = 0"], boundary="left")
    beam.add_boundary_condition("symmetry", boundary="right")
    problem.solve(report="none")
    return problem, stiffness


if __name__ == "__main__":
    print("   n   " + "".join(f"{m:>26s}" for m in MODELS))
    for power_law_index in (0.0, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0):
        row = []
        for model in MODELS:
            problem, stiffness = solve(model, power_law_index)
            reduced = stiffness.bending - stiffness.coupling**2 / stiffness.extensional
            centre = problem.values("disp_z")[-1]
            row.append(centre * reduced / (LOAD * LENGTH**4) * 10)
        print(f"{power_law_index:5.1f}" + "".join(f"{value:26.4f}" for value in row))
    print("(normalized centre deflection w Dhat / (q0 L^4) x 10)")
