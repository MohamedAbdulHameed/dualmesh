"""Uncertainty of the bus bar of Reddy's Example 5.4.3, from Python: the
same study as bus_bar_uq.yaml."""

import dualmesh as dm
from dualmesh import uq


def bus_bar(conductivity=20.0, heat_source=1.0, convection=75.0):
    mesh = dm.generate_rectangle_mesh(0.0, 0.1, 0.0, 0.05, 10, 5)
    problem = dm.Problem(mesh, method="dmcdm")
    problem.add_variable("temperature", initial_condition=0.0)
    problem.add_kernel(
        "heat_conduction", "conduction", variable="temperature", thermal_conductivity=conductivity
    )
    problem.add_kernel(
        "heat_source", "heating", variable="temperature", heat_source=1.0e6 * heat_source
    )
    for side, value in (("left", 40.0), ("right", 10.0)):
        problem.add_boundary_condition(
            "Dirichlet_boundary_condition", side, variable="temperature", boundary=side, value=value
        )
    problem.add_boundary_condition(
        "convective_heat_flux_boundary_condition",
        "top",
        variable="temperature",
        boundary="top",
        heat_transfer_coefficient=convection,
        ambient_temperature=0.0,
    )
    problem.add_postprocessor(
        "total_reaction", "heat_through_left", variable="temperature", boundary="left"
    )
    problem.add_postprocessor(
        "nodal_extreme_value", "hottest", variable="temperature", value_type="max"
    )
    problem.solve()
    values = problem.postprocessor_values()
    return {"hottest": values["hottest"][-1], "heat_through_left": values["heat_through_left"][-1]}


inputs = {
    "conductivity": uq.Normal(20.0, 1.0),
    "heat_source": uq.Normal(1.0, 0.05),
    "convection": uq.LogNormal(median=75.0, factor=2.0),
}

if __name__ == "__main__":
    runs = uq.propagate(bus_bar, inputs, samples=64, seed=1)
    print(runs.summary())
