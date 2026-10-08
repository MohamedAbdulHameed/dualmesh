# SPDX-License-Identifier: LGPL-2.1-or-later
"""The output design: the report levels of every study, the Output group of
the fields, the time steps that land on the output times, and the units of
the variables in the files."""

from __future__ import annotations

import xml.etree.ElementTree as ElementTree

import dualmesh as dm
import numpy as np
import pytest
from dualmesh import fuel, uq
from dualmesh._core import InputError

DAY = 86400.0


def bus_bar(method="dmcdm"):
    mesh = dm.generate_rectangle_mesh(x_min=0.0, x_max=0.1, y_min=0.0, y_max=0.05, num_x_elements=10, num_y_elements=5)
    problem = dm.Problem(mesh, method=method)
    heat = problem.add_physics("heat_transfer", "heat", thermal_conductivity=20.0, heat_source=1.0e6)
    heat.add_boundary_condition("Dirichlet_boundary_condition", "left", value=40.0)
    heat.add_boundary_condition("Dirichlet_boundary_condition", "right", value=10.0)
    return problem


def cooling_bar(variable_unit="K"):
    """A bar at 1 K that cools through its ends (rho c = 6, k = 0.5)."""
    problem = dm.Problem(dm.generate_line_mesh(start=0.0, end=1.0, num_elements=20))
    problem.add_variable("temperature", initial_condition=1.0, unit=variable_unit)
    problem.add_kernel("heat_conduction", variable="temperature", thermal_conductivity=0.5)
    problem.add_kernel("heat_conduction_time_derivative", variable="temperature", density=2.0, specific_heat=3.0)
    problem.add_boundary_condition("Dirichlet_boundary_condition", "ends", variable="temperature", boundary=["left", "right"], value=0.0)
    return problem


def read_csv(path):
    header = path.read_text().splitlines()[0].split(",")
    return header, np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)


def vtu_arrays(path):
    root = ElementTree.parse(path).getroot()
    return {array.get("Name"): array for array in root.iter("DataArray") if array.get("Name")}


# ---- report levels ------------------------------------------------------------
def test_the_report_levels_of_a_solve(capfd):
    problem = bus_bar()
    problem.solve(report="none")
    assert capfd.readouterr().out == ""
    problem.solve(report="summary")
    out = capfd.readouterr().out
    assert out.startswith("dualmesh steady solve: 66 unknowns, method dmcdm") and "converged" in out and "(default)" not in out
    problem.solve()
    out = capfd.readouterr().out
    assert "(default)" in out and "Newton" in out and "converged" in out
    with pytest.raises(InputError, match="report must be one of full, summary, none"):
        problem.solve(report="quiet")
    with pytest.raises(TypeError, match="The parameter report sets what a solve prints"):
        problem.solve(verbose=True)


def test_a_study_inside_a_study_prints_nothing(capsys):
    """A model of an uncertainty study solves with the default report, full,
    and prints nothing because the study runs it as an inner study."""

    def peak(conductivity):
        problem = dm.Problem(dm.generate_line_mesh(start=0.0, end=1.0, num_elements=8))
        heat = problem.add_physics("heat_transfer", "heat", thermal_conductivity=conductivity, heat_source=1.0)
        heat.add_boundary_condition("Dirichlet_boundary_condition", "ends", boundary=["left", "right"], value=0.0)
        problem.solve()
        return float(np.max(problem.values("temperature")))

    runs = uq.propagate(peak, {"conductivity": uq.Uniform(1.0, 2.0)}, 4, report="none")
    assert capsys.readouterr().out == ""
    # T_max = q L^2 / (8 k) for the bar with its ends at zero.
    assert runs.outputs["output"] == pytest.approx(1.0 / (8.0 * runs.inputs["conductivity"]), rel=2e-2)
    uq.propagate(peak, {"conductivity": uq.Uniform(1.0, 2.0)}, 2)
    out = capsys.readouterr().out
    assert "Uncertain inputs" in out and "uq: run 2/2" in out and "Newton" not in out


def test_the_report_of_a_fuel_rod(capsys):
    def rod():
        return fuel.FuelRod(
            fuel.RodGeometry.from_diameters(8.19e-3, 8.36e-3, 9.50e-3, 0.05),
            fuel.UO2Fuel(grain_radius=5.0e-6),
            fuel.ZircaloyCladding(),
            fuel.FillGas(pressure=2.0e6, plenum_volume=0.15e-6),
            fuel.ForcedConvection(inlet_temperature=565.0, pressure=15.5e6, mass_flux=3800.0, rod_pitch=12.6e-3),
            fuel.PowerHistory(linear_heat_rate=[1e3, 20e3], time=[0, 3600]),
            models=fuel.RodModels(mechanics=False, fission_gas_release="none"),
            numerics=fuel.RodNumerics(mesh=fuel.RodMesh(num_axial_elements=2)),
        )

    rod().run(report="none")
    assert capsys.readouterr().out == ""
    rod().run(report="summary")
    out = capsys.readouterr().out
    assert out.startswith("dualmesh fuel rod: UO2Fuel fuel") and "Summary" in out and "(default)" not in out
    rod().run()
    assert "(default)" in capsys.readouterr().out


# ---- the Output group -----------------------------------------------------------
def test_a_steady_solve_writes_its_fields_with_their_units(tmp_path):
    problem = bus_bar()
    problem.solve(report="none", output=dm.Output(directory=str(tmp_path / "steady"), file_base="bar", formats=["vtu", "csv"]))
    header, rows = read_csv(tmp_path / "steady" / "bar.csv")
    assert header == ["x (m)", "y (m)", "temperature (K)"]
    assert rows[:, 2] == pytest.approx(problem.values("temperature"), rel=1e-11)
    array = vtu_arrays(tmp_path / "steady" / "bar.vtu")["temperature"]
    assert array.get("units") == "K"
    assert np.array(array.text.split(), dtype=float) == pytest.approx(problem.values("temperature"), rel=1e-11)


def test_the_output_group_refuses_mistakes_with_the_reason():
    with pytest.raises(InputError, match="give only one of times and interval"):
        dm.Output(times=[0.0, 1.0], interval=0.5)
    with pytest.raises(InputError, match="unknown format 'exodus'"):
        dm.Output(formats=["exodus"])
    with pytest.raises(InputError, match="times must be a list of increasing times"):
        dm.Output(times=[1.0, 0.5])
    with pytest.raises(InputError, match="unknown parameter 'interval_time'"):
        dm.Output(interval_time=1.0)
    problem = bus_bar()
    with pytest.raises(InputError, match="a steady solve writes the fields once"):
        problem.solve(report="none", output=dm.Output(interval=1.0))
    with pytest.raises(InputError, match="Did you mean 'temperature'"):
        problem.solve(report="none", output=dm.Output(fields=["temperatur"]))
    with pytest.raises(InputError, match="the times must be in the interval of the study"):
        cooling_bar().solve_transient(end_time=1.0, time_step=0.1, report="none", output=dm.Output(times=[0.5, 2.0]))


def test_the_fields_select_the_variables_of_the_files(tmp_path):
    problem = dm.Problem(dm.generate_rectangle_mesh(x_min=0.0, x_max=1.0, y_min=0.0, y_max=0.2, num_x_elements=10, num_y_elements=2))
    solid = problem.add_physics("solid_mechanics", "solid", formulation="plane_stress", youngs_modulus=1.0, poissons_ratio=0.3)
    solid.add_boundary_condition("fixed_constraint", "left")
    solid.add_boundary_condition("traction_boundary_condition", "right", traction=[0.01, 0.0])
    problem.solve(report="none", output=dm.Output(directory=str(tmp_path), file_base="plate", fields=["displacement_x"], formats=["vtu", "csv"]))
    arrays = vtu_arrays(tmp_path / "plate.vtu")
    assert "displacement_x" in arrays and "displacement_y" not in arrays and arrays["displacement_x"].get("units") == "m"
    assert read_csv(tmp_path / "plate.csv")[0] == ["x (m)", "y (m)", "displacement_x (m)"]


@pytest.mark.parametrize("time_stepper", ["fixed", "iteration"])
def test_the_time_steps_land_on_the_output_times(tmp_path, time_stepper):
    """With a step of 0.1 s and an output every 0.25 s, the steps land on
    0.25, 0.5 and 0.75 s, and the step after an output returns to the step
    that the stepper chose."""
    problem = cooling_bar()
    output = dm.Output(directory=str(tmp_path), file_base="bar", interval=0.25, formats=["vtu", "csv"])
    result = problem.solve_transient(end_time=1.0, time_step=0.1, time_stepper=time_stepper, report="none", output=output)
    times = [t for t, _ in result.step_history]
    for target in (0.25, 0.5, 0.75, 1.0):
        assert min(abs(t - target) for t in times) < 1e-12
    if time_stepper == "fixed":
        assert times == pytest.approx([0.1, 0.2, 0.25, 0.35, 0.45, 0.5, 0.6, 0.7, 0.75, 0.85, 0.95, 1.0])
    collection = ElementTree.parse(tmp_path / "bar.pvd").getroot()
    assert [float(d.get("timestep")) for d in collection.iter("DataSet")] == pytest.approx([0.0, 0.25, 0.5, 0.75, 1.0])
    for index in range(5):
        vtu = np.array(vtu_arrays(tmp_path / f"bar_{index:05d}.vtu")["temperature"].text.split(), dtype=float)
        csv = read_csv(tmp_path / f"bar_{index:05d}.csv")[1][:, 1]
        assert csv == pytest.approx(vtu, rel=1e-11, abs=1e-14)
    assert read_csv(tmp_path / "bar_00000.csv")[1][1:-1, 1] == pytest.approx(1.0)
    assert csv == pytest.approx(problem.values("temperature"), rel=1e-11, abs=1e-14)


def test_the_fields_at_an_output_time_equal_a_run_to_that_time(tmp_path):
    """The run with outputs takes the steps 0.1, 0.2 and 0.25 s to its first
    output, which are the steps of a run that ends at 0.25 s."""
    with_output = cooling_bar()
    with_output.solve_transient(end_time=1.0, time_step=0.1, report="none", output=dm.Output(directory=str(tmp_path), file_base="bar", interval=0.25, formats=["csv"]))
    short = cooling_bar()
    short.solve_transient(end_time=0.25, time_step=0.1, report="none")
    assert read_csv(tmp_path / "bar_00001.csv")[1][:, 1] == pytest.approx(short.values("temperature"), rel=1e-12, abs=1e-15)


def test_the_csv_rows_of_the_cell_centred_method_are_at_its_unknowns(tmp_path):
    """The cell-centred method has one unknown in each element and one on
    each boundary face, at their centroids: the CSV file gives these
    positions (it gave the node positions, whose count differs)."""
    problem = bus_bar("zfvm")
    problem.solve(report="none")
    problem.write_csv(str(tmp_path / "cells.csv"))
    header, rows = read_csv(tmp_path / "cells.csv")
    assert header == ["x (m)", "y (m)", "temperature (K)"]
    assert len(rows) == len(problem.values("temperature")) == 50 + 30
    assert rows[:, :2] == pytest.approx(np.asarray(problem.entity_points())[:, :2])


def test_a_variable_of_the_object_level_has_a_unit_only_when_one_is_given(tmp_path):
    problem = cooling_bar(variable_unit="")
    assert problem.variable_unit("temperature") == ""
    problem.solve_transient(end_time=0.1, time_step=0.1, report="none")
    problem.write_csv(str(tmp_path / "bar.csv"))
    assert read_csv(tmp_path / "bar.csv")[0] == ["x (m)", "temperature"]
    assert bus_bar().variable_unit("temperature") == "K"


def test_an_eigenvalue_study_writes_its_fluxes(tmp_path, capsys):
    mesh = dm.generate_line_mesh(start=0.0, end=1.0, num_elements=20)
    problem = dm.Problem(mesh)
    neutrons = problem.add_physics("neutron_diffusion", "neutrons", groups=1)
    problem.add_property("multigroup_cross_sections", "core", diffusion_coefficient=[0.01], absorption_cross_section=[2.0], nu_fission_cross_section=[3.0])
    neutrons.add_boundary_condition("Dirichlet_boundary_condition", "zero_flux", boundary=mesh.sideset_names(), value=0.0)
    result = problem.solve_eigenvalue(report="none", output=dm.Output(directory=str(tmp_path), file_base="slab", formats=["csv"]))
    assert capsys.readouterr().out == ""
    header, rows = read_csv(tmp_path / "slab.csv")
    assert header == ["x (m)", "neutron_flux_1 (1/(m^2 s))"]
    assert rows[:, 1] == pytest.approx(problem.values("neutron_flux_1"), rel=1e-11)
    assert result.k_effective == pytest.approx(3.0 / (2.0 + 0.01 * np.pi**2), rel=1e-3)


def test_the_rod_output_has_the_names_of_the_output_group(tmp_path):
    rod = fuel.FuelRod(
        fuel.RodGeometry.from_diameters(8.19e-3, 8.36e-3, 9.50e-3, 0.05),
        fuel.UO2Fuel(grain_radius=5.0e-6),
        fuel.ZircaloyCladding(),
        fuel.FillGas(pressure=2.0e6, plenum_volume=0.15e-6),
        fuel.ForcedConvection(inlet_temperature=565.0, pressure=15.5e6, mass_flux=3800.0, rod_pitch=12.6e-3),
        fuel.PowerHistory(linear_heat_rate=[1e3, 20e3, 20e3], time=[0, 3600, 4 * DAY]),
        models=fuel.RodModels(mechanics=False, fission_gas_release="none"),
        numerics=fuel.RodNumerics(mesh=fuel.RodMesh(num_axial_elements=2)),
        output=fuel.RodOutput(times=[0.0, 2 * DAY, 4 * DAY], directory=str(tmp_path), file_base="rod", formats=["csv", "json"]),
    )
    result = rod.run(report="none")
    assert list(result.time) == pytest.approx([0.0, 2 * DAY, 4 * DAY])
    assert sorted(p.name for p in tmp_path.iterdir()) == ["rod.json", "rod_axial.csv", "rod_history.csv", "rod_input.txt"]
    with pytest.raises(ValueError, match="unknown format 'vtu'"):
        fuel.RodOutput(formats=["vtu"])
