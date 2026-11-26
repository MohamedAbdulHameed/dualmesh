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
    heat = problem.add_physics("heat_conduction", thermal_conductivity=20.0, heat_source=1.0e6)
    heat.add_boundary_condition("temperature = 40", boundary="left")
    heat.add_boundary_condition("temperature = 10", boundary="right")
    return problem


def cooling_bar(variable_unit="K"):
    """A bar at 1 K that cools through its ends (rho c = 6, k = 0.5)."""
    problem = dualmesh.Problem(dualmesh.generate_line_mesh(start=0.0, end=1.0, num_elements=20))
    problem.add_variable("temperature", initial_condition=1.0, unit=variable_unit)
    problem.add_kernel("diffusion", variable="temperature", diffusivity=0.5)
    problem.add_kernel("time_derivative", variable="temperature", coefficient=6.0)
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
    with pytest.raises(InputError, match="give at most one of times, interval and step_interval"):
        dualmesh.Output(times=[0.0, 1.0], interval=0.5)
    with pytest.raises(InputError, match="step_interval is a number of steps"):
        dualmesh.Output(step_interval=0)
    output = dualmesh.Output()
    with pytest.raises(InputError, match="unknown parameter 'file_bas'. Did you mean 'file_base'"):
        output.file_bas = "plate"
    with pytest.raises(InputError, match="unknown format 'exodus'"):
        dualmesh.Output(formats=["exodus"])
    with pytest.raises(InputError, match="times must be a list of increasing times"):
        dualmesh.Output(times=[1.0, 0.5])
    with pytest.raises(InputError, match="unknown parameter 'interval_time'"):
        dualmesh.Output(interval_time=1.0)
    problem = bus_bar()
    with pytest.raises(InputError, match="a steady solve writes the fields once"):
        problem.solve(report="none", output=dualmesh.Output(interval=1.0))
    with pytest.raises(InputError, match="a steady solve writes the fields once"):
        problem.solve(report="none", output=dualmesh.Output(step_interval=2))
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


def collection_times(path):
    root = ElementTree.parse(path).getroot()
    return [(float(entry.get("timestep")), entry.get("file")) for entry in root.iter("DataSet")]


def test_an_output_by_steps_writes_the_start_every_interval_and_the_end(tmp_path):
    """Five fixed steps of 0.1 s with step_interval 2 write the start, the steps 2 and 4, and the
    last step, which lands on no multiple of the interval."""
    output = dualmesh.Output(directory=str(tmp_path), file_base="bar", step_interval=2)
    cooling_bar().solve_transient(
        end_time=0.5, time_step=0.1, time_stepper="fixed", report="none", output=output
    )
    written = collection_times(tmp_path / "bar.pvd")
    assert [time for time, _ in written] == pytest.approx([0.0, 0.2, 0.4, 0.5])
    assert [name for _, name in written] == [f"bar_{index:05d}.vtu" for index in range(4)]
    assert sorted(path.name for path in tmp_path.glob("bar_*.vtu")) == [name for _, name in written]
    # The start holds the initial temperature and the boundary values of the start.
    start = vtu_arrays(tmp_path / "bar_00000.vtu")["temperature"]
    values = numpy.array(start.text.split(), dtype=float)
    assert values.max() == pytest.approx(1.0) and values.min() == pytest.approx(0.0)


def test_runs_that_go_on_from_each_other_keep_one_numbering_of_their_files(tmp_path):
    """A run that stops after three steps writes neither its last step nor the end, and the run
    that goes on from it writes on in the same numbering and collection."""
    from dualmesh.output import Series

    series = Series(dualmesh.Output(directory=str(tmp_path), file_base="bar", step_interval=2))
    problem = cooling_bar()
    problem.solve_transient(
        end_time=0.5, time_step=0.1, time_stepper="fixed", max_steps=3, report="none", output=series
    )
    assert [time for time, _ in collection_times(tmp_path / "bar.pvd")] == pytest.approx([0.0, 0.2])
    problem.solve_transient(
        start_time=problem.time,
        end_time=0.5,
        time_step=0.1,
        time_stepper="fixed",
        report="none",
        output=series,
    )
    written = collection_times(tmp_path / "bar.pvd")
    assert [time for time, _ in written] == pytest.approx([0.0, 0.2, 0.4, 0.5])
    assert [name for _, name in written] == [f"bar_{index:05d}.vtu" for index in range(4)]
