# SPDX-License-Identifier: LGPL-2.1-or-later
"""A cooling fin of Reddy's book, in physical units.

An aluminium fin 50 mm long, of conductivity k = 200 W/(m K), is held at 593 K at its root and
loses heat along its length to air at T_a = 293 K. Its excess temperature theta = T - T_a
solves -theta'' + m^2 theta = 0, with m^2 = 400 1/m^2 the ratio of the convection over its
perimeter to the conduction along it. The heat lost along the fin is the volumetric sink
-m^2 k (T - T_a). At the tip, the fin loses heat by convection with h = 400 W/(m^2 K), so that
theta' + (h/k) theta = 0 with h/k = 2 1/m, as in the book.

The book gives theta = 300.00, 257.62, 225.59, 202.64, 187.83 and 180.57 K at the six nodes of
five DMCDM elements.
"""

import dualmesh

ambient_temperature = 293.0
published_excess_temperatures = [300.00, 257.62, 225.59, 202.64, 187.83, 180.57]


def fin(cells=5):
    cooling_fin = dualmesh.problem(
        mesh={"segment": {"length": "50 mm", "cells": cells}}, method="DMCDM"
    )
    cooling_fin.physics("heat_conduction")
    cooling_fin.properties(
        thermal_conductivity="200 W/(m*K)",
        volumetric_heat_source="-[400 1/m^2]*[200 W/(m*K)]*(temperature - [293 K])",
    )
    cooling_fin.boundary_conditions("left", temperature="593 K")
    cooling_fin.boundary_conditions(
        "right",
        convection={"heat_transfer_coefficient": "400 W/(m^2*K)", "ambient_temperature": "293 K"},
    )
    return cooling_fin


if __name__ == "__main__":
    solution = fin().solve()
    order = solution.coordinates("temperature")[:, 0].argsort()
    excess = solution.field("temperature")[order, 0] - ambient_temperature
    print("theta (K):        " + " ".join(f"{value:8.2f}" for value in excess))
    print(
        "Reddy, theta (K): " + " ".join(f"{value:8.2f}" for value in published_excess_temperatures)
    )
