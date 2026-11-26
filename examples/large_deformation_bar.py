# SPDX-License-Identifier: LGPL-2.1-or-later
"""A bar in large deformation, written as a model of your own: Example 6.2.4 of Reddy's book.

A bar of length L = 1 m is fixed at x = 0 and carries the axial load Q at x = L. Its axial
stiffness depends on the strain e = du/dx of its axial displacement u,

    a(e) = EA (1 + 1.5 e + 0.5 e^2),

so the problem is nonlinear. The axial force N = a(e) e is the flux of the model, and the
balance dN/dx = 0 holds along the bar. The input differentiates the flux for Newton's method.
With EA = 1 MN, the book gives the tip displacements u(L) = 0.1597, 0.5213 and 1.3087 m for
Q = 0.2, 1 and 5 MN on eight elements.
"""

import dualmesh

published_tip_displacements = {"0.2 MN": 0.1597, "1 MN": 0.5213, "5 MN": 1.3087}


def bar(load, cells=8):
    rod = dualmesh.problem(mesh={"segment": {"length": "1 m", "cells": cells}})
    rod.model(
        "bar",
        unknown={"axial_displacement": {"unit": "m"}},
        expressions={"axial_strain": {"unit": "1", "value": "dot(grad(axial_displacement), [1])"}},
        flux="axial_stiffness*(1 + 1.5*axial_strain + 0.5*axial_strain^2)*grad(axial_displacement)",
        flux_name="axial_force",
    )
    rod.properties(axial_stiffness={"unit": "N", "value": "1 MN"})
    rod.boundary_conditions("left", axial_displacement="0 m")
    rod.boundary_conditions("right", axial_force=load)
    return rod


if __name__ == "__main__":
    print("   Q      u(L) (m)   Reddy (m)")
    for load, published in published_tip_displacements.items():
        solution = bar(load).solve(report="none")
        tip = solution.field("axial_displacement")[:, 0].max()
        print(f"{load:>6s}  {tip:9.4f}  {published:9.4f}")
