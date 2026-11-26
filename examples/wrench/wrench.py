# SPDX-License-Identifier: LGPL-2.1-or-later
"""A steel combination wrench that tightens a nut, on an imported Gmsh mesh.

The load case follows the COMSOL Application Gallery model "Stresses and Strains in a Wrench": a
structural steel wrench, held by the nut and loaded by a hand force of 150 N. The geometry of the
wrench is built by make_wrench_mesh.py, which writes the mesh wrench.msh, in m, with two named
surfaces (Gmsh physical groups):

jaws
    The flats of the open end that bear on the nut. The nut is taken as rigid, so the three
    displacement components are zero there.
grip
    The upper and lower faces of the handle between x = 120 mm and x = 145 mm, of the area
    2 (25 mm) (12 mm) = 6e-4 m^2. A uniform shear traction of 250 kPa in the -y direction
    applies the total force of 150 N.

The script checks the solution against two results that do not depend on the finite element
model: the reaction of the jaws, which balances the hand force of 150 N, and the bending stress
of beam theory in the handle at x = 60 mm, away from the head and the grip. At the edge of the
handle the shear stress vanishes, so the stress is the bending stress alone, and the von Mises
stress there equals it.
"""

from pathlib import Path

import dualmesh
import numpy

mm = 1.0e-3
force = 150.0  # N
grip_centre_x = 132.5 * mm
width, thickness = 12.0 * mm, 5.0 * mm  # The handle section
section_x = 60.0 * mm


def beam_stress() -> float:
    """The bending stress of beam theory at the edge of the handle at x = 60 mm, sigma = M c / I,
    with the moment M = F (x_grip - x), c half the width and I the second moment of area."""
    inertia = thickness * width**3 / 12.0
    return force * (grip_centre_x - section_x) * (width / 2.0) / inertia


def wrench():
    tool = dualmesh.problem(
        mesh={"file": str(Path(__file__).with_name("wrench.msh")), "length_unit": "m"}
    )
    tool.physics("solid_mechanics")
    tool.properties(Young_modulus="200 GPa", Poisson_ratio=0.3)
    # The nut holds the jaws, and the hand pulls the grip with 150 N over its 6e-4 m^2.
    tool.boundary_conditions("jaws", displacement=["0 m", "0 m", "0 m"])
    tool.boundary_conditions("grip", traction=["0 Pa", "-250 kPa", "0 Pa"])
    tool.postprocessor("reaction", boundary_flux="traction_y", boundary="jaws")
    tool.postprocessor("edge", point_value="von_Mises_stress", point=["60 mm", "6 mm", "0 mm"])
    tool.postprocessor(
        "section",
        line="stress_xx",
        from_=["60 mm", "-6 mm", "0 mm"],
        to=["60 mm", "6 mm", "0 mm"],
        points=13,
    )
    tool.postprocessor("deflection", minimum="displacement_y")
    tool.output(file="wrench", fields=["displacement"])
    return tool


def bending_stress(solution) -> float:
    """The bending stress at the edge of the section x = 60 mm: the slope of a straight line
    fitted to sigma_xx across the width, times half the width."""
    section = solution.line("section")
    return numpy.polyfit(section["y"], section["stress_xx"], 1)[0] * width / 2.0


if __name__ == "__main__":
    solution = wrench().solve()
    rows = {
        "reaction on the jaws (N)": (solution.postprocessor("reaction")[-1], force),
        "bending stress at x = 60 mm (MPa)": (bending_stress(solution) / 1e6, beam_stress() / 1e6),
        "von Mises stress at the edge (MPa)": (
            solution.postprocessor("edge")[-1] / 1e6,
            beam_stress() / 1e6,
        ),
        "largest deflection (mm)": (-solution.postprocessor("deflection")[-1] / mm, None),
    }
    for name, (value, reference) in rows.items():
        extra = f"   (reference {reference:.4g})" if reference is not None else ""
        print(f"{name:36s} {value:10.4g}{extra}")
    print("wrote wrench.vtu, wrench.csv and wrench_section.csv")
