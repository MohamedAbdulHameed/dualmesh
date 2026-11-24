# SPDX-License-Identifier: LGPL-2.1-or-later
"""The output design: the report levels of every study, the Output group of
the fields, the time steps that land on the output times, and the units of
the variables in the files."""

from __future__ import annotations

from xml.etree import ElementTree

import dualmesh
import numpy
import pytest
from dualmesh._core import InputError


def bus_bar(method="dmcdm"):
    mesh = dualmesh.generate_rectangle_mesh(
        x_min=0.0, x_max=0.1, y_min=0.0, y_max=0.05, num_x_elements=10, num_y_elements=5
    )
    problem = dualmesh.Problem(mesh, method=method)
    heat = problem.add_physics("heat_transfer", thermal_conductivity=20.0, heat_source=1.0e6)
    heat.add_boundary_condition("temperature = 40", boundary="left")
    heat.add_boundary_condition("temperature = 10", boundary="right")
    return problem


def cooling_bar(variable_unit="K"):
    """A bar at 1 K that cools through its ends (rho c = 6, k = 0.5)."""
    problem = dualmesh.Problem(dualmesh.generate_line_mesh(start=0.0, end=1.0, num_elements=20))
    problem.add_variable("temperature", initial_condition=1.0, unit=variable_unit)
    problem.add_kernel("heat_conduction", variable="temperature", thermal_conductivity=0.5)
    problem.add_kernel(
        "heat_conduction_time_derivative", variable="temperature", density=2.0, specific_heat=3.0
    )
    problem.add_boundary_condition(
        "Dirichlet",
        ["left", "right"],
        variable="temperature",
        value=0.0,
    )
    return problem


def vtu_arrays(path):
    root = ElementTree.parse(path).getroot()
    return {array.get("Name"): array for array in root.iter("DataArray") if array.get("Name")}


def test_the_output_group_refuses_mistakes_with_the_reason():
    with pytest.raises(InputError, match="give only one of times and interval"):
        dualmesh.Output(times=[0.0, 1.0], interval=0.5)
    with pytest.raises(InputError, match="unknown format 'exodus'"):
        dualmesh.Output(formats=["exodus"])
    with pytest.raises(InputError, match="times must be a list of increasing times"):
        dualmesh.Output(times=[1.0, 0.5])
    with pytest.raises(InputError, match="unknown parameter 'interval_time'"):
        dualmesh.Output(interval_time=1.0)
    problem = bus_bar()
    with pytest.raises(InputError, match="a steady solve writes the fields once"):
        problem.solve(report="none", output=dualmesh.Output(interval=1.0))
    with pytest.raises(InputError, match="Did you mean 'temperature'"):
        problem.solve(report="none", output=dualmesh.Output(fields=["temperatur"]))
    with pytest.raises(InputError, match="the times must be in the interval of the study"):
        cooling_bar().solve_transient(
            end_time=1.0, time_step=0.1, report="none", output=dualmesh.Output(times=[0.5, 2.0])
        )


def test_the_stress_of_the_elements_is_written_with_its_values(tmp_path):
    """A plate on rollers on its left and bottom edges, pulled by 0.01 Pa on its right edge,
    carries the uniform stress sigma_xx = 0.01 and nothing else, in every element."""
    mesh = dualmesh.generate_rectangle_mesh(x=[0, 1], y=[0, 1], num_elements=[4, 4])
    problem = dualmesh.Problem(mesh)
    solid = problem.add_physics(
        "solid_mechanics", formulation="plane_stress", youngs_modulus=1.0, poissons_ratio=0.25
    )
    solid.add_boundary_condition("disp_x = 0", boundary="left")
    solid.add_boundary_condition("disp_y = 0", boundary="bottom")
    solid.add_boundary_condition("traction = [0.01, 0]", boundary="right")
    problem.solve(report="none")
    problem.write_vtu(str(tmp_path / "plate.vtu"), cell_properties=["stress"])
    stress = vtu_arrays(tmp_path / "plate.vtu")["stress"]
    assert stress.get("NumberOfComponents") == "6"
    values = numpy.array(stress.text.split(), dtype=float).reshape(-1, 6)
    assert len(values) == mesh.num_elements
    assert numpy.allclose(values[:, 0], 0.01, rtol=1e-12, atol=0)
    assert numpy.abs(values[:, 1:]).max() <= 1e-15
