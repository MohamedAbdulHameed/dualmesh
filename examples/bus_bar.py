# SPDX-License-Identifier: LGPL-2.1-or-later
"""Reddy, Example 5.4.3: a bus bar with internal heat generation.

The bar occupies 0 <= x <= 0.1 m and 0 <= y <= 0.05 m.  Its conductivity is
20 W/(m K) and it generates 1e6 W/m^3.  The left and right sides are held at
40 and 10 degrees, the bottom is insulated, and the top loses heat by
convection (h = 75 W/(m^2 K)) to surroundings at 0 degrees.  The example
prints the post-processors and writes the field and the post-processors to
files.  The temperature at the middle of the bottom edge is 83.142 in Table
5.4.3 of the book.
"""

import dualmesh as dm


def bus_bar(method="dmcdm"):
    mesh = dm.generate_rectangle_mesh(x_min=0.0, x_max=0.1, y_min=0.0, y_max=0.05, num_x_elements=10, num_y_elements=5)
    problem = dm.Problem(mesh, method=method)
    heat = problem.add_physics("heat_transfer", "heat", thermal_conductivity=20.0, heat_source=1.0e6)
    heat.add_boundary_condition("Dirichlet_boundary_condition", "left", value=40.0)
    heat.add_boundary_condition("Dirichlet_boundary_condition", "right", value=10.0)
    heat.add_boundary_condition("convective_heat_flux_boundary_condition", "top", heat_transfer_coefficient=75.0, ambient_temperature=0.0)
    problem.add_postprocessor("total_reaction", "heat_through_left", variable="temperature", boundary="left")
    problem.add_postprocessor("total_reaction", "heat_through_right", variable="temperature", boundary="right")
    problem.add_postprocessor("point_value", "temperature_bottom", variable="temperature", point=[0.05, 0.0])
    problem.add_postprocessor("point_value", "temperature_top", variable="temperature", point=[0.05, 0.05])
    problem.add_postprocessor("nodal_extreme_value", "hottest", variable="temperature", value_type="max")
    problem.solve()
    return problem


if __name__ == "__main__":
    problem = bus_bar()
    problem.write_vtu("bus_bar.vtu")
    problem.write_postprocessor_csv("bus_bar_postprocessors.csv")
    print("book (DMCDM): temperature_bottom = 83.142")
