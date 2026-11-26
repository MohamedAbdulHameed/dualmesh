# SPDX-License-Identifier: LGPL-2.1-or-later
"""A bus bar with internal heat generation, Example 5.4.3 of Reddy's book.

The bar occupies 0 <= x <= 0.1 m and 0 <= y <= 0.05 m. Its conductivity is 20 W/(m K) and it
generates 1 MW/m^3. The left and right sides are held at 40 and 10 degC, the bottom is
insulated, and the top loses heat by convection, with h = 75 W/(m^2 K), to surroundings at
0 degC. The example prints the postprocessors and writes the field and the postprocessors to
files. The book gives 83.142 degC at the middle of the bottom edge.
"""

import dualmesh


def bus_bar():
    bar = dualmesh.problem(
        mesh={"rectangle": {"size": ["0.1 m", "0.05 m"], "cells": [10, 5]}}, method="DMCDM"
    )
    bar.physics("heat_conduction")
    bar.properties(thermal_conductivity="20 W/(m*K)", volumetric_heat_source="1 MW/m^3")
    bar.boundary_conditions("left", temperature="40 degC")
    bar.boundary_conditions("right", temperature="10 degC")
    bar.boundary_conditions(
        "top",
        convection={"heat_transfer_coefficient": "75 W/(m^2*K)", "ambient_temperature": "0 degC"},
    )
    bar.postprocessor("heat_through_left", boundary_flux="heat_flux", boundary="left")
    bar.postprocessor("heat_through_right", boundary_flux="heat_flux", boundary="right")
    bar.postprocessor("temperature_bottom", point_value="temperature", point=["0.05 m", "0 m"])
    bar.postprocessor("temperature_top", point_value="temperature", point=["0.05 m", "0.05 m"])
    bar.postprocessor("hottest", maximum="temperature")
    bar.output(file="bus_bar", fields=["temperature"])
    return bar


if __name__ == "__main__":
    solution = bus_bar().solve()
    for name in solution.postprocessor_names():
        print(f"{name}: {solution.postprocessor(name)[-1]:.6g} {solution.unit(name)}")
    bottom = solution.postprocessor("temperature_bottom")[-1] - 273.15
    print(f"temperature at the middle of the bottom edge: {bottom:.3f} degC (Reddy: 83.142 degC)")
