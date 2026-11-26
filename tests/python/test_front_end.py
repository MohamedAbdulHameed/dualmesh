# SPDX-License-Identifier: LGPL-2.1-or-later
"""The input of a simulation solves to the exact answer of its problem: a heated slab, a slab
cooled by convection and a bar pulled by a traction. The same problem given in other units,
or read from a YAML file, gives the same answer, and an input that no physics reads, or that
leaves out what a physics needs, is refused with its key."""

import csv
from pathlib import Path

import dualmesh
import numpy
import pytest

length, conductivity, source, wall_temperature = 0.1, 45.0, 1.0e6, 300.0


def heated_slab(size="0.1 m", cells=20):
    """A slab of conductivity k with a uniform heat source q, held at the temperature T0 at
    both faces: T(x) = T0 + q x (L - x) / (2 k)."""
    slab = dualmesh.problem(mesh={"segment": {"length": size, "cells": cells}})
    slab.physics("heat_conduction")
    slab.properties(thermal_conductivity="45 W/(m*K)", volumetric_heat_source="1 MW/m^3")
    slab.boundary_conditions("left", temperature="300 K")
    slab.boundary_conditions("right", temperature="300 K")
    return slab


def exact_slab_temperature(x):
    return wall_temperature + source * x * (length - x) / (2 * conductivity)


def test_a_heated_slab_takes_the_exact_temperature_at_the_nodes():
    """Linear elements in one dimension are exact at the nodes for a uniform source."""
    solution = heated_slab().solve(report="none")
    x = solution.coordinates("temperature")[:, 0]
    temperature = solution.field("temperature")[:, 0]
    assert temperature == pytest.approx(exact_slab_temperature(x), rel=1e-10)


def test_the_same_slab_in_millimetres_gives_the_same_temperatures():
    in_metres = heated_slab().solve(report="none")
    in_millimetres = heated_slab(size="100 mm").solve(report="none")
    assert in_millimetres.field("temperature") == pytest.approx(
        in_metres.field("temperature"), rel=1e-12
    )


def test_the_same_slab_read_from_a_YAML_file_gives_the_same_temperatures(tmp_path):
    slab = heated_slab()
    slab.to_YAML(tmp_path / "slab.yaml")
    read_back = dualmesh.problem.from_file(tmp_path / "slab.yaml")
    assert read_back.to_tree() == slab.to_tree()
    assert read_back.solve(report="none").field("temperature") == pytest.approx(
        slab.solve(report="none").field("temperature"), rel=1e-12
    )


def test_a_table_of_the_input_gives_the_solution_of_its_formula(tmp_path, monkeypatch):
    """A table of two rows is the line k = 40 + 0.01 (T - 250 K) W/(m K) over its range, given
    inline or in a CSV file, so the slab with the table takes the temperatures of the slab with
    that formula.  The source as a table in x of two equal rows is the uniform source."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "k.csv").write_text(
        "temperature [K], thermal_conductivity [W/(m*K)]\n250, 40\n1250, 50\n"
    )
    inline = dualmesh.table(
        temperature=("K", [1250, 250]), thermal_conductivity=("W/(m*K)", [50, 40])
    )
    formula = "[40 W/(m*K)] + [0.01 W/(m*K^2)]*(temperature - [250 K])"
    temperatures = []
    for conductivity_value in (formula, inline, dualmesh.table("k.csv")):
        slab = heated_slab()
        slab.properties(
            thermal_conductivity=conductivity_value,
            volumetric_heat_source=dualmesh.table(
                x=("m", [0, 0.1]), volumetric_heat_source=("MW/m^3", [1, 1])
            ),
        )
        temperatures.append(slab.solve(report="none").field("temperature")[:, 0])
    assert temperatures[0].max() > 320.0
    for temperature in temperatures[1:]:
        assert temperature == pytest.approx(temperatures[0], rel=1e-12)


def test_a_table_read_outside_its_range_stops_the_run_with_the_property_and_the_value():
    """The slab peaks near 328 K, above the 310 K that ends the table."""
    slab = heated_slab()
    slab.properties(
        thermal_conductivity=dualmesh.table(
            temperature=("K", [250, 310]), thermal_conductivity=("W/(m*K)", [45, 45])
        )
    )
    with pytest.raises(
        RuntimeError, match="the table of the property thermal_conductivity reads temperature = 3"
    ):
        slab.solve(report="none")


def test_a_slab_cooled_by_convection_has_the_heat_flow_of_its_series_resistance():
    """The heat flow through a slab of conductivity k and thickness L into a fluid at T_a with the
    coefficient h is q = (T_0 - T_a) / (L/k + 1/h), and the cooled face is at T_0 - q L/k."""
    hot, ambient, coefficient = 400.0, 300.0, 250.0
    slab = dualmesh.problem(mesh={"segment": {"length": "0.1 m", "cells": 10}})
    slab.physics("heat_conduction")
    slab.properties(thermal_conductivity="45 W/(m*K)")
    slab.boundary_conditions("left", temperature="400 K")
    slab.boundary_conditions(
        "right",
        convection={"heat_transfer_coefficient": "250 W/(m^2*K)", "ambient_temperature": "300 K"},
    )
    slab.postprocessor("heat_to_air", boundary_flux="heat_flux", boundary="right")
    solution = slab.solve(report="none")
    flow = (hot - ambient) / (length / conductivity + 1.0 / coefficient)
    x = solution.coordinates("temperature")[:, 0]
    temperature = solution.field("temperature")[:, 0]
    assert temperature[numpy.argmax(x)] == pytest.approx(
        hot - flow * length / conductivity, rel=1e-10
    )
    assert solution.postprocessor("heat_to_air")[-1] == pytest.approx(flow, rel=1e-8)
    assert solution.unit("heat_to_air") == "W/m^2"


def test_a_bar_pulled_by_a_traction_stretches_by_sigma_L_over_E():
    """In plane stress a uniform traction sigma gives u_x = sigma x / E and u_y = -nu sigma y / E."""
    bar = dualmesh.problem(mesh={"rectangle": {"size": ["2 m", "0.5 m"], "cells": [8, 2]}})
    bar.physics("solid_mechanics", formulation="plane_stress")
    bar.properties(Young_modulus="200 GPa", Poisson_ratio=0.3)
    bar.boundary_conditions("left", displacement_x="0 m")
    bar.boundary_conditions("bottom", displacement_y="0 m")
    bar.boundary_conditions("right", traction=["100 MPa", "0 Pa"])
    solution = bar.solve(report="none")
    x, y = solution.coordinates("displacement").T
    displacement = solution.field("displacement")
    assert displacement[:, 0] == pytest.approx(1.0e8 * x / 2.0e11, rel=1e-9, abs=1e-15)
    assert displacement[:, 1] == pytest.approx(-0.3 * 1.0e8 * y / 2.0e11, rel=1e-9, abs=1e-15)


def test_a_property_that_no_physics_reads_is_refused():
    slab = heated_slab()
    slab.properties(Young_modulus="200 GPa")
    with pytest.raises(
        dualmesh.input_error, match=r"properties\.Young_modulus: no physics of the problem reads"
    ):
        slab.solve(report="none")


def test_a_property_that_a_physics_needs_is_asked_for():
    slab = dualmesh.problem(mesh={"segment": {"length": "1 m", "cells": 4}})
    slab.physics("heat_conduction")
    slab.boundary_conditions("left", temperature="300 K")
    with pytest.raises(
        dualmesh.input_error, match="heat_conduction needs the property thermal_conductivity"
    ):
        slab.solve(report="none")


def test_a_transient_run_writes_its_fields_and_its_postprocessors_with_their_units(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    slab = heated_slab(cells=10)
    slab.properties(density="7850 kg/m^3", specific_heat_capacity="490 J/(kg*K)")
    slab.initial_conditions(temperature="300 K")
    slab.time(end_time="40 s", initial_time_step="10 s", step_control={"by": "fixed"})
    slab.postprocessor("peak", maximum="temperature")
    slab.output(file="slab", interval=2, fields=["temperature"])
    solution = slab.solve(report="none")
    assert solution.times == pytest.approx([10.0, 20.0, 30.0, 40.0])
    assert sorted(path.name for path in tmp_path.glob("slab_*.vtu")) == [
        "slab_00000.vtu",
        "slab_00001.vtu",
        "slab_00002.vtu",
    ]
    assert (tmp_path / "slab.pvd").exists()
    with open(tmp_path / "slab.csv") as file:
        rows = list(csv.reader(file))
    assert rows[0] == ["time [s]", "peak [K]"]
    assert [float(row[0]) for row in rows[1:]] == pytest.approx(solution.output_times)
    # The peak rises from 300 K and stays below the steady peak.
    peaks = [float(row[1]) for row in rows[1:]]
    assert peaks[0] >= 300.0 and peaks[-1] > 300.0
    assert all(peak < exact_slab_temperature(length / 2) for peak in peaks)
    assert peaks == sorted(peaks)


def test_the_bus_bar_example_gives_the_temperature_of_Reddy(tmp_path, monkeypatch):
    """Example 5.4.3 of Reddy's book, as examples/bus_bar.py writes it: 83.142 degC at the middle
    of the bottom edge, given to three decimals."""
    import importlib.util
    from pathlib import Path

    monkeypatch.chdir(tmp_path)
    path = Path(__file__).resolve().parents[2] / "examples" / "bus_bar.py"
    specification = importlib.util.spec_from_file_location("bus_bar_example", path)
    example = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(example)
    solution = example.bus_bar().solve(report="none")
    assert solution.postprocessor("temperature_bottom")[-1] - 273.15 == pytest.approx(
        83.142, abs=5e-4
    )
    # A steady run writes its fields once, without a collection.
    assert (tmp_path / "bus_bar.vtu").exists() and (tmp_path / "bus_bar.csv").exists()
    assert not (tmp_path / "bus_bar.pvd").exists()


def test_the_cooling_fin_example_gives_the_temperatures_of_Reddy():
    """The fin of examples/cooling_fin.py, five DMCDM elements, against the book to two decimals."""
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "examples" / "cooling_fin.py"
    specification = importlib.util.spec_from_file_location("cooling_fin_example", path)
    example = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(example)
    solution = example.fin().solve(report="none")
    order = solution.coordinates("temperature")[:, 0].argsort()
    excess = solution.field("temperature")[order, 0] - example.ambient_temperature
    assert excess == pytest.approx(example.published_excess_temperatures, abs=5e-3)


def test_a_pressure_on_every_face_compresses_a_plane_strain_block_uniformly():
    """A pressure p on the faces of a block in plane strain gives the strain
    -p (1 + nu)(1 - 2 nu)/E in both directions, the out-of-plane stress -2 nu p held by the
    plane strain."""
    block = dualmesh.problem(mesh={"rectangle": {"size": ["1 m", "0.5 m"], "cells": [4, 2]}})
    block.physics("solid_mechanics", formulation="plane_strain")
    block.properties(Young_modulus="200 GPa", Poisson_ratio=0.3)
    block.boundary_conditions("left", displacement_x="0 m")
    block.boundary_conditions("bottom", displacement_y="0 m")
    block.boundary_conditions("right", pressure="10 MPa")
    block.boundary_conditions("top", pressure="10 MPa")
    solution = block.solve(report="none")
    strain = -1.0e7 * (1 + 0.3) * (1 - 2 * 0.3) / 2.0e11
    x, y = solution.coordinates("displacement").T
    displacement = solution.field("displacement")
    assert displacement[:, 0] == pytest.approx(strain * x, rel=1e-9, abs=1e-16)
    assert displacement[:, 1] == pytest.approx(strain * y, rel=1e-9, abs=1e-16)


def test_the_stress_at_a_point_of_a_pulled_bar_is_the_applied_traction():
    """A uniform traction of 100 MPa gives sigma_xx = 100 MPa and sigma_yy = 0 at every point."""
    bar = dualmesh.problem(mesh={"rectangle": {"size": ["2 m", "0.5 m"], "cells": [8, 2]}})
    bar.physics("solid_mechanics", formulation="plane_stress")
    bar.properties(Young_modulus="200 GPa", Poisson_ratio=0.3)
    bar.boundary_conditions("left", displacement_x="0 m")
    bar.boundary_conditions("bottom", displacement_y="0 m")
    bar.boundary_conditions("right", traction=["100 MPa", "0 Pa"])
    bar.postprocessor("axial", point_value="stress_xx", point=["0.7 m", "0.3 m"])
    bar.postprocessor("lateral", point_value="stress_yy", point=["1.3 m", "0.1 m"])
    solution = bar.solve(report="none")
    assert solution.postprocessor("axial")[-1] == pytest.approx(1.0e8, rel=1e-9)
    assert solution.postprocessor("lateral")[-1] == pytest.approx(0.0, abs=1e-3)
    assert solution.unit("axial") == "Pa"


# ---- gravity ---------------------------------------------------------------------------------
gravity_g, steel_density, column_height = 9.81, 7850.0, 2.0


def standing_column(kinematics="small"):
    """A steel column H = 2 m high on a fixed base, on rollers on its sides, under its weight.
    It is in uniaxial strain, so sigma_yy = -rho g (H - y) and
    u_y = -rho g (H y - y^2 / 2) / M, with M = E (1 - nu) / ((1 + nu) (1 - 2 nu)) the
    constrained modulus.  Linear elements hold u_y exactly at the nodes."""
    column = dualmesh.problem(
        mesh={"rectangle": {"size": ["0.5 m", f"{column_height} m"], "cells": [2, 8]}}
    )
    column.physics("solid_mechanics", formulation="plane_strain", kinematics=kinematics)
    column.properties(Young_modulus="200 GPa", Poisson_ratio=0.3, density=f"{steel_density} kg/m^3")
    column.gravity(["0 m/s^2", f"-{gravity_g} m/s^2"])
    column.boundary_conditions("bottom", displacement_y="0 m")
    for side in ("left", "right"):
        column.boundary_conditions(side, displacement_x="0 m")
    return column


def exact_column_displacement(y):
    E, nu = 200e9, 0.3
    constrained = E * (1 - nu) / ((1 + nu) * (1 - 2 * nu))
    return -steel_density * gravity_g * (column_height * y - y**2 / 2) / constrained


def test_a_column_sinks_under_its_own_weight_by_the_uniaxial_strain_solution():
    solution = standing_column().solve(report="none")
    y = solution.coordinates("displacement")[:, 1]
    displacement = solution.field("displacement")
    assert displacement[:, 1] == pytest.approx(exact_column_displacement(y), rel=1e-8, abs=1e-16)
    assert numpy.abs(displacement[:, 0]).max() < 1e-12 * abs(exact_column_displacement(2.0))


def test_the_weight_of_a_finite_strain_column_is_that_of_the_small_strain_one():
    """The strains are about 1e-6, so finite strain differs from the small-strain solution by a
    part in a million."""
    solution = standing_column(kinematics="finite").solve(report="none")
    y = solution.coordinates("displacement")[:, 1]
    assert solution.field("displacement")[:, 1] == pytest.approx(
        exact_column_displacement(y), rel=1e-4, abs=1e-14
    )


# The velocity that the round-off eps of the balance of the pressure gradient and the weight of
# water at rest drives in a tank 1 m deep: eps rho g H^2 / mu, with mu = 1e-3 Pa*s.
rest_velocity_round_off = numpy.finfo(float).eps * 1000.0 * gravity_g * 1.0**2 / 1e-3


@pytest.mark.parametrize("method", ["DMCDM", "FEM"])
def test_water_at_rest_in_a_tank_carries_the_hydrostatic_pressure(method):
    """Water in a closed tank 1 m deep, open to p = 0 at the top, stays at rest, and its pressure
    is p = rho g (H - y): the pressure gradient balances the weight, with no spurious flow.  The
    round-off of that balance, of relative size eps (the machine epsilon), drives a velocity of
    eps rho g H^2 / mu, about 2e-9 m/s, which bounds the computed one."""
    tank = dualmesh.problem(
        mesh={"rectangle": {"size": ["1 m", "1 m"], "cells": [6, 6]}}, method=method
    )
    tank.physics("fluid_dynamics")
    tank.properties(density="1000 kg/m^3", dynamic_viscosity="1e-3 Pa*s")
    tank.gravity(["0 m/s^2", f"-{gravity_g} m/s^2"])
    for side in ("left", "right", "bottom"):
        tank.boundary_conditions(side, velocity=["0 m/s", "0 m/s"])
    tank.boundary_conditions("top", pressure="0 Pa")
    solution = tank.solve(report="none")
    y = solution.coordinates("pressure")[:, 1]
    assert solution.field("pressure")[:, 0] == pytest.approx(
        1000.0 * gravity_g * (1.0 - y), rel=1e-8, abs=1e-6
    )
    assert numpy.abs(solution.field("velocity")).max() < rest_velocity_round_off


def test_gravity_on_a_problem_without_a_solid_or_a_flow_is_refused():
    slab = heated_slab()
    slab.gravity(["-9.81 m/s^2"])
    with pytest.raises(dualmesh.input_error, match="gravity: no physics of the problem reads"):
        slab.solve(report="none")


def test_a_heated_flow_under_gravity_asks_for_its_expansion_coefficient():
    cavity = dualmesh.problem(mesh={"rectangle": {"size": ["1 m", "1 m"], "cells": [4, 4]}})
    cavity.physics("heat_conduction")
    cavity.physics("fluid_dynamics")
    cavity.properties(
        thermal_conductivity="0.6 W/(m*K)",
        specific_heat_capacity="4182 J/(kg*K)",
        density="1000 kg/m^3",
        dynamic_viscosity="1e-3 Pa*s",
    )
    cavity.gravity(["0 m/s^2", "-9.81 m/s^2"])
    with pytest.raises(
        dualmesh.input_error,
        match="fluid_dynamics needs the property volumetric_expansion_coefficient",
    ):
        cavity.solve(report="none")


def test_a_steady_flow_carries_heat_with_its_heat_capacity():
    """Water in plug flow at U = 1e-6 m/s through a channel L = 1 m long, between walls that move
    with it, heated by q = 10 W/m^3, enters at T_in = 300 K and leaves through an insulated
    outlet.  The temperature solves rho c_p U T' = k T'' + q, so with a = rho c_p U / k,
    T = T_in + q / (rho c_p U) (x - (exp(a (x - L)) - exp(-a L)) / a).  A heat capacity of 1
    in place of c_p would make the temperature rise 4182 times larger."""
    U, rho, c_p, k, q, L = 1e-6, 1000.0, 4182.0, 0.6, 10.0, 1.0
    channel = dualmesh.problem(mesh={"rectangle": {"size": [f"{L} m", "0.1 m"], "cells": [40, 2]}})
    channel.physics("heat_conduction")
    channel.physics("fluid_dynamics")
    channel.properties(
        thermal_conductivity=f"{k} W/(m*K)",
        specific_heat_capacity=f"{c_p} J/(kg*K)",
        density=f"{rho} kg/m^3",
        dynamic_viscosity="1e-3 Pa*s",
        volumetric_heat_source=f"{q} W/m^3",
    )
    for side in ("left", "bottom", "top"):
        channel.boundary_conditions(side, velocity=["1e-6 m/s", "0 m/s"])
    channel.boundary_conditions("left", temperature="300 K")
    channel.boundary_conditions("right", pressure="0 Pa")
    solution = channel.solve(report="none")
    x = solution.coordinates("temperature")[:, 0]
    a = rho * c_p * U / k
    rise = q / (rho * c_p * U) * (x - (numpy.exp(a * (x - L)) - numpy.exp(-a * L)) / a)
    computed = solution.field("temperature")[:, 0] - 300.0
    assert numpy.abs(computed - rise).max() < 5e-3 * q * L / (rho * c_p * U)


def test_a_heated_fluid_at_its_reference_temperature_stays_at_rest():
    """Water in a tank whose walls are all at the reference temperature T_0 takes no buoyancy:
    it stays at rest, at T_0, with the hydrostatic pressure rho g (H - y), and a velocity within
    the round-off of the tank without heat."""
    tank = dualmesh.problem(mesh={"rectangle": {"size": ["1 m", "1 m"], "cells": [6, 6]}})
    tank.physics("heat_conduction")
    tank.physics("fluid_dynamics")
    tank.properties(
        thermal_conductivity="0.6 W/(m*K)",
        specific_heat_capacity="4182 J/(kg*K)",
        density="1000 kg/m^3",
        dynamic_viscosity="1e-3 Pa*s",
        volumetric_expansion_coefficient="2.1e-4 1/K",
        reference_temperature="293.15 K",
    )
    tank.gravity(["0 m/s^2", f"-{gravity_g} m/s^2"])
    for side in ("left", "right", "bottom", "top"):
        tank.boundary_conditions(side, temperature="293.15 K")
    for side in ("left", "right", "bottom"):
        tank.boundary_conditions(side, velocity=["0 m/s", "0 m/s"])
    tank.boundary_conditions("top", pressure="0 Pa")
    solution = tank.solve(report="none")
    assert solution.field("temperature") == pytest.approx(293.15, abs=1e-9)
    assert numpy.abs(solution.field("velocity")).max() < rest_velocity_round_off
    y = solution.coordinates("pressure")[:, 1]
    assert solution.field("pressure")[:, 0] == pytest.approx(
        1000.0 * gravity_g * (1.0 - y), rel=1e-8, abs=1e-6
    )


def test_air_in_a_heated_cavity_convects_heat_at_the_Nusselt_number_of_de_Vahl_Davis():
    """Air (Prandtl number Pr = 0.71) in a square cavity L = 0.1 m wide, its left wall 1 K
    warmer than its right one, insulated at the top and the bottom.  The expansion coefficient
    sets the Rayleigh number Ra = g beta Delta T L^3 / (nu alpha) to 10^4, at which the average
    Nusselt number Nu = Q / (k Delta T), with Q the heat flow through the hot wall per unit
    depth, is 2.243 (de Vahl Davis 1983).  The uniform mesh of 40 by 40 cells puts four cells
    across the boundary layers, about L Ra^(-1/4) = L / 10 thick."""
    L, delta_T, rho, mu, k, Pr, Ra = 0.1, 1.0, 1.2, 1.8e-5, 0.025, 0.71, 1.0e4
    c_p = Pr * k / mu
    nu, alpha = mu / rho, k / (rho * c_p)
    beta = Ra * nu * alpha / (gravity_g * delta_T * L**3)
    cavity = dualmesh.problem(mesh={"rectangle": {"size": [f"{L} m", f"{L} m"], "cells": [40, 40]}})
    cavity.physics("heat_conduction")
    cavity.physics("fluid_dynamics")
    cavity.properties(
        thermal_conductivity=f"{k} W/(m*K)",
        specific_heat_capacity=f"{c_p!r} J/(kg*K)",
        density=f"{rho} kg/m^3",
        dynamic_viscosity=f"{mu} Pa*s",
        volumetric_expansion_coefficient=f"{beta!r} 1/K",
        reference_temperature="293.15 K",
    )
    cavity.gravity(["0 m/s^2", f"-{gravity_g} m/s^2"])
    for side in ("left", "right", "bottom", "top"):
        cavity.boundary_conditions(side, velocity=["0 m/s", "0 m/s"])
    cavity.boundary_conditions("left", temperature=f"{293.15 + delta_T / 2!r} K")
    cavity.boundary_conditions("right", temperature=f"{293.15 - delta_T / 2!r} K")
    cavity.postprocessor("hot_wall", boundary_flux="heat_flux", boundary="left")
    solution = cavity.solve(report="none")
    nusselt = -solution.postprocessor("hot_wall")[-1] / (k * delta_T)
    assert nusselt == pytest.approx(2.243, rel=0.02)


def test_gravity_takes_one_acceleration_per_direction_of_space():
    column = standing_column()
    with pytest.raises(dualmesh.input_error, match="gravity is a list of 2 accelerations"):
        column.gravity(["-9.81 m/s^2"])


def test_an_initial_displacement_of_a_steady_solid_is_refused():
    bar = dualmesh.problem(mesh={"rectangle": {"size": ["2 m", "0.5 m"], "cells": [4, 1]}})
    bar.physics("solid_mechanics", formulation="plane_stress")
    bar.properties(Young_modulus="200 GPa", Poisson_ratio=0.3)
    bar.initial_conditions(displacement=["0 m", "0 m"])
    bar.boundary_conditions("left", displacement_x="0 m", displacement_y="0 m")
    with pytest.raises(dualmesh.input_error, match="initial displacement is the start of a"):
        bar.solve(report="none")


def test_a_thermal_expansion_takes_its_two_properties_together():
    slab = dualmesh.problem(mesh={"rectangle": {"size": ["1 m", "1 m"], "cells": [2, 2]}})
    slab.physics("heat_conduction")
    slab.physics("solid_mechanics", formulation="plane_strain")
    slab.properties(
        thermal_conductivity="45 W/(m*K)",
        Young_modulus="200 GPa",
        Poisson_ratio=0.3,
        linear_expansion_coefficient="1.2e-5 1/K",
    )
    slab.boundary_conditions("left", temperature="400 K", displacement_x="0 m")
    slab.boundary_conditions("bottom", displacement_y="0 m")
    with pytest.raises(dualmesh.input_error, match="gives only linear_expansion_coefficient"):
        slab.solve(report="none")


def test_the_heat_of_the_example_case_leaves_through_its_held_and_cooled_sides():
    """examples/heated_plate.yaml: 1 MW/m^3 in 0.1 m by 0.02 m, per unit depth 2000 W/m, leaves
    through the left side, held at 300 K, and the top side, cooled by air.  The peak temperature
    lies below that of a slab insulated except at one end, 300 K + q L^2 / (2 k)."""
    pytest.importorskip("yaml")
    from pathlib import Path

    example = Path(__file__).resolve().parents[2] / "examples" / "heated_plate.yaml"
    solution = dualmesh.problem.from_file(str(example)).solve(report="none")
    left = solution.postprocessor("heat_through_left")[-1]
    top = solution.postprocessor("heat_through_top")[-1]
    assert left + top == pytest.approx(1.0e6 * 0.1 * 0.02, rel=1e-8)
    assert 300.0 < solution.postprocessor("peak_temperature")[-1] < 300.0 + 1.0e6 * 0.1**2 / 40.0


def test_a_conductivity_of_the_position_and_the_time_gives_the_exact_temperature():
    """k = k0 (1 + x/L + t/[1 s]) with k0 = 45 W/(m*K), between 400 K at x = 0 and 300 K at
    x = L = 0.1 m, with the source q = (100 K / L) (k0 / L) = 0.45 MW/m^3.  The linear
    T = 400 K - (100 K) x / L solves -(k T')' = (100 K / L) dk/dx = q at every time, and the
    elements represent it exactly, since k is linear in x."""
    slab = dualmesh.problem(mesh={"segment": {"length": "0.1 m", "cells": 10}})
    slab.physics("heat_conduction")
    slab.properties(
        thermal_conductivity="[45 W/(m*K)]*(1 + x/[0.1 m] + time/[1 s])",
        volumetric_heat_source="0.45 MW/m^3",
    )
    slab.boundary_conditions("left", temperature="400 K")
    slab.boundary_conditions("right", temperature="300 K")
    solution = slab.solve(report="none")
    x = solution.coordinates("temperature")[:, 0]
    temperature = solution.field("temperature")[:, 0]
    assert temperature == pytest.approx(400.0 - 1000.0 * x, rel=1e-10)


def test_the_input_leaves_the_flow_formulation_to_the_physics():
    """With FEM on a mesh of quadratic cells the fluid_dynamics physics takes Taylor-Hood, which
    needs no stabilisation, so the input gives the flow that the Python interface gives."""
    from dualmesh.input import schema
    from dualmesh.input.build import build_problem

    quadratic = dualmesh.generate_rectangle_mesh(0.0, 1.0, 0.0, 1.0, 2, 2, element_type="Quad9")
    tank = dualmesh.problem(
        mesh={"rectangle": {"size": ["1 m", "1 m"], "cells": [2, 2]}}, method="FEM"
    )
    tank.physics("fluid_dynamics")
    tank.properties(density="1000 kg/m^3", dynamic_viscosity="1e-3 Pa*s")
    _, physics, _ = build_problem(schema.validate(tank.to_tree()), quadratic)
    assert physics["fluid_dynamics"].formulation == "Taylor-Hood"


def test_a_distributed_run_names_its_pvtu_files_and_one_process_writes_the_collection(
    tmp_path, monkeypatch
):
    """A problem split among processes writes each output as the .pvtu index of its pieces,
    which the collection names, and only the first process writes the collection."""
    from dualmesh import _core
    from dualmesh.output import Output, Series

    class distributed_problem:
        def _write_fields(self, output, index):
            return [output.path(f"_{index:05d}.pvtu")]

    def run(directory):
        series = Series(Output(directory=str(directory), file_base="run", step_interval=1))
        files = [path for time in (1.0, 2.0) for path in series.write(distributed_problem(), time)]
        series.write_collection("pvtu")
        return [Path(path).name for path in files]

    assert run(tmp_path) == ["run_00000.pvtu", "run_00001.pvtu"]
    collection = (tmp_path / "run.pvd").read_text()
    assert 'file="run_00000.pvtu"' in collection and 'file="run_00001.pvtu"' in collection
    other = tmp_path / "other"
    monkeypatch.setattr(_core, "mpi_rank", lambda: 1)
    assert run(other) == ["run_00000.pvtu", "run_00001.pvtu"]
    assert not (other / "run.pvd").exists()
