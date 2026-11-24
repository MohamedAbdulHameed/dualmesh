# SPDX-License-Identifier: LGPL-2.1-or-later
"""The schema of the input: it fills the defaults, converts every quantity to SI, checks the
dimension of every formula and table, and stops at a wrong key or value with the key and,
for a misspelled name, the nearest valid one."""

import copy
import re

import pytest
from dualmesh.input import schema
from dualmesh.input.units import input_error


def plate() -> dict:
    """A small heated plate on a generated mesh, as a raw input tree."""
    return {
        "mesh": {"rectangle": {"size": ["0.2 m", "0.1 m"], "cells": [4, 2]}},
        "physics": {"heat_conduction": {}},
        "properties": {
            "density": "7850 kg/m^3",
            "specific_heat_capacity": "490 J/(kg*K)",
            "thermal_conductivity": "45 W/(m*K)",
        },
        "initial_conditions": {"temperature": "293 K"},
        "boundary_conditions": {
            "left": {"temperature": "573 K"},
            "right": {
                "convection": {
                    "heat_transfer_coefficient": "25 W/(m^2*K)",
                    "ambient_temperature": "293 K",
                }
            },
        },
        "time": {"end_time": "600 s", "initial_time_step": "1 s"},
    }


deleted = object()


def changed(tree: dict, path: tuple, value) -> dict:
    """A copy of tree with the value at path replaced, or removed when value is deleted."""
    result = copy.deepcopy(tree)
    node = result
    for key in path[:-1]:
        node = node.setdefault(key, {})
    if value is deleted:
        del node[path[-1]]
    else:
        node[path[-1]] = value
    return result


def raises_naming(*parts: str):
    """Expects input_error with a message that holds every part."""
    return pytest.raises(
        input_error, match="(?s)" + "".join(f"(?=.*{re.escape(part)})" for part in parts)
    )


def property_of(value, name="thermal_conductivity"):
    return schema.validate(changed(plate(), ("properties", name), value))["properties"][name]


def test_the_defaults_are_filled_and_the_quantities_converted_to_SI():
    validated = schema.validate(plate())
    assert "method" not in validated
    assert validated["mesh"] == {
        "rectangle": {"size": [0.2, 0.1], "cells": [4, 2], "cell_type": "quadrilateral"},
        "coordinate_system": "cartesian",
    }
    assert validated["time"]["step_control"] == {"by": "error", "relative_tolerance": 1.0e-3}
    assert validated["boundary_conditions"]["right"]["convection"] == {
        "heat_transfer_coefficient": 25.0,
        "ambient_temperature": 293.0,
    }
    assert property_of("4.5 kW/(m*K)")["value"] == pytest.approx(4500.0)


def test_a_mesh_file_states_the_unit_of_its_coordinates(tmp_path):
    tree = changed(plate(), ("mesh",), {"file": "meshes/rod.msh", "length_unit": "mm"})
    assert schema.validate(tree, tmp_path)["mesh"] == {
        "file": str(tmp_path / "meshes" / "rod.msh"),
        "length_unit": pytest.approx(1.0e-3),
        "coordinate_system": "cartesian",
    }
    with raises_naming("mesh", "length_unit"):
        schema.validate(changed(plate(), ("mesh",), {"file": "rod.msh"}), tmp_path)
    with raises_naming("mesh.length_unit", "a unit of length"):
        schema.validate(
            changed(plate(), ("mesh",), {"file": "rod.msh", "length_unit": "s"}), tmp_path
        )
    with raises_naming("mesh.length_unit", "takes no length_unit"):
        schema.validate(changed(plate(), ("mesh", "length_unit"), "mm"))


@pytest.mark.parametrize(
    "mesh, message",
    [
        ({}, "expects exactly one of 'segment', 'rectangle', 'box', 'file'"),
        (
            {"rectangle": {"size": ["0.2 m"], "cells": [4, 2]}},
            "mesh.rectangle.size: expects a list of 2 entries",
        ),
        ({"rectangle": {"size": ["0.2 m", "0 m"], "cells": [4, 2]}}, "expects a positive value"),
        (
            {"segment": {"length": "1 m", "cells": 4}, "coordinate_system": "axisymmetric"},
            "a segment has space dimension 1",
        ),
    ],
)
def test_a_bad_mesh_is_refused_with_its_key(mesh, message):
    with raises_naming(message):
        schema.validate(changed(plate(), ("mesh",), mesh))


@pytest.mark.parametrize(
    "path, value, message",
    [
        (("properties", "density"), 7850, "has no unit"),
        (("initial_conditions", "temperature"), 293, "has no unit"),
        (("time", "end_time"), 600, "has no unit"),
        (("properties", "density"), "7850 kg/m^2", "kg/m^3"),
        (
            ("boundary_conditions", "right", "convection", "heat_transfer_coefficient"),
            "25 W/m^2",
            "W/(m^2*K)",
        ),
    ],
)
def test_a_value_without_its_unit_or_of_the_wrong_dimension_is_refused_with_its_key(
    path, value, message
):
    with raises_naming(".".join(path), message):
        schema.validate(changed(plate(), path, value))


@pytest.mark.parametrize(
    "value, message",
    [
        ("[45 W/(m*K)]*temperature", "the formula gives the unit"),
        ("2 * densty", "unknown name 'densty'"),
        ("[45 W/(m*K)]*(1 - temperature/[2000 K]", "expected ')'"),
    ],
)
def test_a_bad_formula_is_refused_with_its_key(value, message):
    with raises_naming("properties.thermal_conductivity", message):
        property_of(value)


def test_an_inline_table_sorts_its_rows_and_converts_to_SI():
    value = {
        "table": {
            "temperature": {"unit": "K", "values": [900, 300, 600]},
            "thermal_conductivity": {"unit": "mW/(m*K)", "values": [3600, 7600, 5000]},
        }
    }
    table = property_of(value)["value"]["table"]
    assert table["inputs"] == [{"name": "temperature", "values": [300.0, 600.0, 900.0]}]
    assert table["values"] == pytest.approx([7.6, 5.0, 3.6])


def test_a_property_per_region_names_the_regions_of_the_mesh():
    assert property_of({"domain": "45 W/(m*K)"}) == {
        "unit": "W/(m*K)",
        "value": None,
        "regions": {"domain": 45.0},
    }
    with raises_naming("properties.thermal_conductivity", "unknown region 'domian'", "'domain'"):
        property_of({"domian": "45 W/(m*K)"})


def test_a_physics_takes_its_options_with_their_values():
    tree = changed(
        plate(),
        ("physics", "solid_mechanics"),
        {"formulation": "plane_stress", "kinematics": "finite"},
    )
    assert schema.validate(tree)["physics"]["solid_mechanics"] == {
        "formulation": "plane_stress",
        "kinematics": "finite",
    }
    with raises_naming("physics.solid_mechanics.kinematics", "'finite'"):
        schema.validate(changed(plate(), ("physics", "solid_mechanics"), {"kinematics": "finit"}))
    with raises_naming("physics.heat_conduction", "unknown key 'thickness'"):
        schema.validate(changed(plate(), ("physics", "heat_conduction"), {"thickness": "5 mm"}))
    with raises_naming("physics", "unknown physics 'heat_conductio'", "'heat_conduction'"):
        schema.validate(changed(plate(), ("physics",), {"heat_conductio": {}}))


@pytest.mark.parametrize(
    "step_control, expected",
    [
        ({"by": "iterations"}, {"by": "iterations", "target_iterations": 5}),
        ({"by": "Courant"}, {"by": "Courant", "Courant_number": 0.5}),
        ({"by": "fixed"}, {"by": "fixed"}),
    ],
)
def test_each_step_control_takes_its_own_setting(step_control, expected):
    validated = schema.validate(changed(plate(), ("time", "step_control"), step_control))
    assert validated["time"]["step_control"] == expected
    with raises_naming("time.step_control", "unknown key 'relative_tolerance'"):
        schema.validate(
            changed(
                plate(), ("time", "step_control"), {"by": "iterations", "relative_tolerance": 1e-3}
            )
        )


@pytest.mark.parametrize(
    "definition, message",
    [
        ({"average": "temperature", "maximum": "temperature"}, "expects exactly one of"),
        ({"average": "temperature", "region": "domian"}, "'domain'"),
        ({"point_value": "temperature", "point": ["0.2 m"]}, "expects a list of 2 entries"),
        ({"boundary_flux": "heat_flux", "boundary": "rigth"}, "'right'"),
    ],
)
def test_a_bad_postprocessor_is_refused_with_its_key(definition, message):
    with raises_naming("postprocessors.probe", message):
        schema.validate(changed(plate(), ("postprocessors", "probe"), definition))


def test_the_solver_takes_PETSc_options_without_their_dash():
    solver = {
        "relative_tolerance": "1e-8",
        "PETSc_options": {"ksp_type": "gmres", "snes_monitor": True},
    }
    assert schema.validate(changed(plate(), ("solver",), solver))["solver"] == {
        "relative_tolerance": 1.0e-8,
        "PETSc_options": {"ksp_type": "gmres", "snes_monitor": "true"},
    }
    with raises_naming("solver", "without the leading dash"):
        schema.validate(changed(plate(), ("solver",), {"PETSc_options": {"-ksp_type": "gmres"}}))


@pytest.mark.parametrize(
    "path, wrong, right",
    [
        (("boundary_conditions",), "boundary_condition", "boundary_conditions"),
        (("properties", "thermal_conductivity"), "thermal_conductivty", "thermal_conductivity"),
        (("time", "end_time"), "end_tme", "end_time"),
        (("boundary_conditions", "right"), "rigth", "right"),
    ],
)
def test_a_misspelled_key_names_the_nearest_valid_key(path, wrong, right):
    tree = copy.deepcopy(plate())
    node = tree
    for key in path[:-1]:
        node = node[key]
    node[wrong] = node.pop(path[-1])
    with raises_naming(wrong, right):
        schema.validate(tree)
