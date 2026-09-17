# SPDX-License-Identifier: LGPL-2.1-or-later
"""Reddy, Example 5.3.1: a cooling fin, -u'' + 400 u = 0 on (0, 0.05).

u(0) = 300, and at x = L the fin loses heat to the surroundings:
u'(L) + 2 u(L) = 0.  The example prints the nodal temperatures, the heat flow
into the fin at the fixed end (the secondary variable), and the same solution
computed with the finite element method for comparison.
"""

import dualmesh as dm


def solve(num_elements, method="dmcdm"):
    mesh = dm.generate_line_mesh(start=0.0, end=0.05, num_elements=num_elements)
    problem = dm.Problem(mesh, method=method)
    problem.add_variable("temperature")
    problem.add_kernel("Diffusion", variable="temperature")
    problem.add_kernel("Reaction", variable="temperature", coefficient=400.0)
    problem.add_boundary_condition(
        "DirichletBC", variable="temperature", boundary="left", value=300.0
    )
    problem.add_boundary_condition(
        "RobinBC", variable="temperature", boundary="right", transfer_coefficient=2.0
    )
    problem.solve()
    return problem


if __name__ == "__main__":
    for method in ("dmcdm", "fem"):
        problem = solve(5, method)
        values = problem.values("temperature")
        print(f"{method:6s} u = " + " ".join(f"{v:8.3f}" for v in values))
        print(f"{method:6s} Q(0) = {problem.total_reaction('temperature', 'left'):.1f}")
    print("book (DMCDM): 300.00  257.62  225.59  202.64  187.83  180.57, Q(0) = 4817")
