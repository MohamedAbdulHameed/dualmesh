# SPDX-License-Identifier: LGPL-2.1-or-later
"""Flow in a lid-driven cavity at a Reynolds number of 1000, by the penalty formulation.

A square cavity of side a = 1 m holds a fluid of density 1000 kg/m^3 and viscosity 1 Pa s, and
its lid moves at V = 1 m/s, so the Reynolds number rho V a / mu is 1000. The rows of the mesh
shrink towards the lid, where the velocity changes fastest. The horizontal velocity on the
vertical centreline is compared with the dual mesh control domain method (DMCDM) solution of
Example 9.8.4 of Reddy's book, on its mesh of 16 by 20 elements.
"""

import dualmesh
import numpy

published_y = [
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
published_u = [
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


def cavity():
    flow = dualmesh.problem(
        mesh={"rectangle": {"size": ["1 m", "1 m"], "cells": [16, 20], "grading": [1, 0.964]}},
        method="DMCDM",
    )
    flow.physics("fluid_dynamics", formulation="penalty")
    flow.properties(density="1000 kg/m^3", dynamic_viscosity="1 Pa*s")
    for wall in ("left", "right", "bottom"):
        flow.boundary_conditions(wall, velocity=["0 m/s", "0 m/s"])
    flow.boundary_conditions("top", velocity=["1 m/s", "0 m/s"])
    flow.postprocessor(
        "centreline", line="velocity_x", from_=["0.5 m", "0 m"], to=["0.5 m", "1 m"], points=33
    )
    flow.output(file="cavity", fields=["velocity"])
    return flow


if __name__ == "__main__":
    solution = cavity().solve()
    line = solution.line("centreline")
    computed = numpy.interp(published_y, line["y"], line["velocity_x"])
    print("    y (m)   velocity_x (m/s)   Reddy (m/s)")
    for y, mine, published in zip(published_y, computed, published_u):
        print(f"{y:8.4f}  {mine:14.4f}  {published:12.4f}")
    print("wrote cavity.vtu and cavity_centreline.csv")
