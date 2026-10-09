# SPDX-License-Identifier: LGPL-2.1-or-later
"""Uncertainty of the bus bar of Reddy's Example 5.4.3 (examples/bus_bar.py).

The conductivity, the heat source (as a factor on 1e6 W/m^3) and the
convection coefficient are uncertain.  A Latin hypercube sample of 64 runs
propagates them to the hottest temperature and to the heat through the left
side.  The statistics are printed and written to bus_bar_uq.json.
"""

import dualmesh as dm
from dualmesh import uq


def bus_bar(conductivity=20.0, heat_source=1.0, convection=75.0):
    mesh = dm.generate_rectangle_mesh(x_min=0.0, x_max=0.1, y_min=0.0, y_max=0.05, num_x_elements=10, num_y_elements=5)
    problem = dm.Problem(mesh, method="dmcdm")
    heat = problem.add_physics("heat_transfer", "heat", thermal_conductivity=conductivity, heat_source=1.0e6 * heat_source)
    heat.add_boundary_condition("Dirichlet_boundary_condition", "left", value=40.0)
    heat.add_boundary_condition("Dirichlet_boundary_condition", "right", value=10.0)
    heat.add_boundary_condition("convective_heat_flux_boundary_condition", "top", heat_transfer_coefficient=convection, ambient_temperature=0.0)
    problem.add_postprocessor("total_reaction", "heat_through_left", variable="temperature", boundary="left")
    problem.add_postprocessor("nodal_extreme_value", "hottest", variable="temperature", value_type="max")
    problem.solve()
    values = problem.postprocessor_values()
    return {"hottest": values["hottest"][-1], "heat_through_left": values["heat_through_left"][-1]}


inputs = {"conductivity": uq.Normal(20.0, 1.0), "heat_source": uq.Normal(1.0, 0.05), "convection": uq.LogNormal(median=75.0, factor=2.0)}

if __name__ == "__main__":
    runs = uq.propagate(bus_bar, inputs, samples=64, method="latin_hypercube", seed=1)
    runs.write_json("bus_bar_uq.json")
