"""The method by default: DMCDM for a flow of the pressure formulation, FEM otherwise."""

from __future__ import annotations

import dualmesh as dm
import pytest
from dualmesh._core import InputError


def square(element_type="Quad4"):
    return dm.generate_rectangle_mesh(0, 1, 0, 1, 4, 4, element_type=element_type)


def cavity(problem, formulation="pressure"):
    flow = problem.add_physics(
        "fluid_dynamics", density=1.0, dynamic_viscosity=1.0, formulation=formulation
    )
    flow.add_boundary_condition(["velocity_x = 1", "velocity_y = 0"], boundary="top")
    flow.add_boundary_condition(
        ["velocity_x = 0", "velocity_y = 0"], boundary=["left", "right", "bottom"]
    )
    return flow


def test_a_flow_takes_dmcdm():
    problem = dm.Problem(square())
    cavity(problem)
    problem.solve(report="none")
    assert problem.method == "DMCDM"


def test_a_flow_that_carries_heat_takes_dmcdm():
    problem = dm.Problem(square())
    flow = cavity(problem)
    heat = problem.add_physics("heat_transfer", thermal_conductivity=1.0)
    heat.add_boundary_condition("temperature = 0", boundary=["left", "right", "bottom", "top"])
    problem.add_coupling("nonisothermal_flow", heat_transfer=heat, fluid_dynamics=flow)
    problem.solve(report="none")
    assert problem.method == "DMCDM"


def test_a_solid_takes_fem():
    problem = dm.Problem(square())
    solid = problem.add_physics(
        "solid_mechanics", formulation="plane_stress", youngs_modulus=1.0, poissons_ratio=0.3
    )
    solid.add_boundary_condition("fixed", boundary="left")
    solid.add_boundary_condition("traction = [0, -1]", boundary="right")
    problem.solve(report="none")
    assert problem.method == "FEM"


def test_conduction_and_a_taylor_hood_flow_take_fem():
    problem = dm.Problem(square())
    heat = problem.add_physics("heat_transfer", thermal_conductivity=1.0, heat_source=1.0)
    heat.add_boundary_condition("temperature = 0", boundary="left")
    problem.solve(report="none")
    assert problem.method == "FEM"
    problem = dm.Problem(square("Quad9"))
    cavity(problem, formulation="Taylor-Hood")
    problem.solve(report="none")
    assert problem.method == "FEM"


def test_a_given_method_is_kept():
    problem = dm.Problem(square(), method="FEM")
    cavity(problem)
    problem.solve(report="none")
    assert problem.method == "FEM"


@pytest.mark.parametrize("name", ["HFVM", "ZFVM", "fvm", "finite_element"])
def test_another_method_name_is_refused(name):
    with pytest.raises(InputError, match=f"Unknown method '{name}'. Use 'FEM'"):
        dm.Problem(square(), method=name)


def test_a_flow_added_after_a_solve_on_fem_is_refused():
    """The automatic method is chosen at the first solve; a flow added later cannot change it."""
    problem = dm.Problem(square())
    heat = problem.add_physics("heat_transfer", thermal_conductivity=1.0, heat_source=1.0)
    heat.add_boundary_condition("temperature = 0", boundary="left")
    problem.solve(report="none")
    cavity(problem)
    with pytest.raises(InputError, match="method='DMCDM'"):
        problem.solve(report="none")
