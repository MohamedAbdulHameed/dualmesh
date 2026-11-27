"""The method by default: DMCDM for a flow of the equal_order formulation, FEM for a solid and
every other problem, and DMCDM for a solid beside such a flow, since a problem takes one method."""

from __future__ import annotations

import dualmesh
import pytest
from dualmesh._core import InputError


def square(element_type="Quad4"):
    return dualmesh.generate_rectangle_mesh(0, 1, 0, 1, 4, 4, element_type=element_type)


def cavity(problem):
    """The lid-driven cavity, with the formulation of the flow by default."""
    flow = problem.add_physics("fluid_dynamics", density=1.0, dynamic_viscosity=1.0)
    flow.add_boundary_condition(["velocity_x = 1", "velocity_y = 0"], boundary="top")
    flow.add_boundary_condition(
        ["velocity_x = 0", "velocity_y = 0"], boundary=["left", "right", "bottom"]
    )
    return flow


def test_a_solid_takes_FEM():
    """A plate in plane strain pulled by 1 MPa on the right."""
    problem = dualmesh.Problem(square())
    solid = problem.add_physics(
        "solid_mechanics", formulation="plane_strain", youngs_modulus=2.0e11, poissons_ratio=0.3
    )
    solid.add_boundary_condition("disp_x = 0", boundary="left")
    solid.add_boundary_condition("disp_y = 0", point=[0.0, 0.0])
    solid.add_boundary_condition("traction = [1.0e6, 0]", boundary="right")
    problem.solve(report="none")
    assert problem.method == "FEM"


@pytest.mark.parametrize("element_type", ["Quad4", "Quad9"])
def test_a_flow_takes_DMCDM(element_type):
    """Also on a quadratic mesh, where the flow then takes the equal_order formulation."""
    problem = dualmesh.Problem(square(element_type))
    flow = cavity(problem)
    problem.solve(report="none")
    assert problem.method == "DMCDM"
    assert flow.formulation == "equal_order"


def test_a_solid_beside_a_flow_takes_DMCDM():
    """The method follows the physics as they are added: FEM for the solid, then DMCDM for the
    whole problem once a flow joins it."""
    problem = dualmesh.Problem(square())
    problem.add_physics(
        "solid_mechanics", formulation="plane_strain", youngs_modulus=2.0e11, poissons_ratio=0.3
    )
    assert problem.method == "FEM"
    problem.add_physics("fluid_dynamics", density=1.0, dynamic_viscosity=1.0)
    assert problem.method == "DMCDM"


def test_a_given_method_is_kept():
    problem = dualmesh.Problem(square(), method="FEM")
    cavity(problem)
    problem.solve(report="none")
    assert problem.method == "FEM"


@pytest.mark.parametrize("name", ["fvm", "finite_element"])
def test_another_method_name_is_refused(name):
    with pytest.raises(InputError, match=f"Unknown method '{name}'. Use 'FEM'"):
        dualmesh.Problem(square(), method=name)


def test_a_flow_added_after_a_solve_on_FEM_is_refused():
    """The automatic method is fixed at the first solve, and a flow added later cannot change it."""
    problem = dualmesh.Problem(square())
    heat = problem.add_physics("heat_conduction", thermal_conductivity=1.0, heat_source=1.0)
    heat.add_boundary_condition("temperature = 0", boundary="left")
    problem.solve(report="none")
    with pytest.raises(InputError, match="method='DMCDM'"):
        cavity(problem)
