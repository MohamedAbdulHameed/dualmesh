# SPDX-License-Identifier: LGPL-2.1-or-later
"""The quench of a steel slab, compared with the Fourier series of its temperature.

A steel slab 0.1 m thick, at 573 K, has both faces held at 293 K from the time 0. With the
thermal diffusivity alpha = k / (rho c_p), its temperature is the Fourier series

    T(x, t) = T_s + (T_0 - T_s) sum_{n odd} (4 / (n pi)) sin(n pi x / L) exp(-(n pi / L)^2 alpha t),

with T_0 = 573 K the initial temperature and T_s = 293 K that of the faces. The run ends at
40 s, when the middle has cooled by a fifth of the drop.
"""

import dualmesh
import numpy

length, end_time = 0.1, 40.0
conductivity, density, specific_heat_capacity = 45.0, 7850.0, 490.0
initial_temperature, face_temperature = 573.0, 293.0


def exact(x, t, terms=200):
    diffusivity = conductivity / (density * specific_heat_capacity)
    total = numpy.zeros_like(x)
    for n in range(1, 2 * terms, 2):
        total += (
            (4.0 / (n * numpy.pi))
            * numpy.sin(n * numpy.pi * x / length)
            * numpy.exp(-((n * numpy.pi / length) ** 2) * diffusivity * t)
        )
    return face_temperature + (initial_temperature - face_temperature) * total


def slab(cells=40, time_step="0.1 s"):
    quench = dualmesh.problem(mesh={"segment": {"length": "0.1 m", "cells": cells}})
    quench.physics("heat_conduction")
    quench.properties(
        thermal_conductivity="45 W/(m*K)",
        density="7850 kg/m^3",
        specific_heat_capacity="490 J/(kg*K)",
    )
    quench.initial_conditions(temperature="573 K")
    quench.boundary_conditions("left", temperature="293 K")
    quench.boundary_conditions("right", temperature="293 K")
    quench.time(end_time="40 s", initial_time_step=time_step, step_control={"by": "fixed"})
    return quench


if __name__ == "__main__":
    solution = slab().solve()
    x = solution.coordinates("temperature")[:, 0]
    computed = solution.field("temperature")[:, 0]
    reference = exact(x, end_time)
    print(
        f"largest temperature at {end_time} s: {computed.max():.2f} K (Fourier series {reference.max():.2f} K)"
    )
    print(f"largest error: {numpy.max(numpy.abs(computed - reference)):.3f} K")
