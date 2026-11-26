"""The method by default: DMCDM for a flow of the pressure formulation, FEM otherwise."""

from __future__ import annotations

import dualmesh
import pytest
from dualmesh._core import InputError


def square(element_type="Quad4"):
    return dualmesh.generate_rectangle_mesh(0, 1, 0, 1, 4, 4, element_type=element_type)


def cavity(problem, formulation="equal_order"):
    flow = problem.add_physics(
        "fluid_dynamics", density=1.0, dynamic_viscosity=1.0, formulation=formulation
    )
    flow.add_boundary_condition(["velocity_x = 1", "velocity_y = 0"], boundary="top")
    flow.add_boundary_condition(
        ["velocity_x = 0", "velocity_y = 0"], boundary=["left", "right", "bottom"]
    )
    return flow


@pytest.mark.parametrize("name", ["fvm", "finite_element"])
def test_another_method_name_is_refused(name):
    with pytest.raises(InputError, match=f"Unknown method '{name}'. Use 'FEM'"):
        dualmesh.Problem(square(), method=name)


def test_a_flow_added_after_a_solve_on_FEM_is_refused():
    """The automatic method is chosen at the first solve; a flow added later cannot change it."""
    problem = dualmesh.Problem(square())
    heat = problem.add_physics("heat_conduction", thermal_conductivity=1.0, heat_source=1.0)
    heat.add_boundary_condition("temperature = 0", boundary="left")
    problem.solve(report="none")
    cavity(problem)
    with pytest.raises(InputError, match="method='DMCDM'"):
        problem.solve(report="none")
