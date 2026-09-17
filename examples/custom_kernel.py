# SPDX-License-Identifier: LGPL-2.1-or-later
"""A kernel written in Python: the large-deformation bar of Example 6.2.4.

The axial stiffness depends on the strain,

    a(u') = EA (1 + 1.5 u' + 0.5 u'^2),

so the problem is nonlinear.  The kernel returns the flux a(u') u'; the
derivatives needed by Newton's method are carried by the AD numbers, so
nothing else has to be supplied.
"""

import dualmesh as dm
import numpy as np


class LargeDeformationBar(dm.PythonKernel):
    """Flux a(u') du/dx with a(u') = EA (1 + 1.5 u' + 0.5 u'^2)."""

    def compute_flux(self, ctx):
        strain = ctx.gradient(self.variable)[0]
        stiffness = self.axial_stiffness * (1.0 + 1.5 * strain + 0.5 * strain * strain)
        return [stiffness * strain]


def solve(load, num_elements=8, num_load_steps=5):
    mesh = dm.generate_line_mesh(start=0.0, end=1.0, num_elements=num_elements)
    problem = dm.Problem(mesh)
    problem.add_variable("displacement")
    problem.add_kernel(LargeDeformationBar(variable="displacement", axial_stiffness=1.0))
    problem.add_boundary_condition(
        "DirichletBC", variable="displacement", boundary="left", value=0.0
    )
    problem.add_boundary_condition(
        "NeumannBC", variable="displacement", boundary="right", flux=load
    )
    problem.solve(load_factors=list(np.linspace(1.0 / num_load_steps, 1.0, num_load_steps)))
    return problem


if __name__ == "__main__":
    print(" P      u(L)    book")
    for load, reference in ((0.2, 0.1597), (1.0, 0.5213), (5.0, 1.3087)):
        tip = solve(load).values("displacement")[-1]
        print(f"{load:4.1f}  {tip:8.4f}  {reference:8.4f}")
