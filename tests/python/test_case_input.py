# SPDX-License-Identifier: LGPL-2.1-or-later
"""Case files: the physics level written as data (YAML) builds the same problem as the Python
input, with the same names, and every name is checked."""

from __future__ import annotations

from pathlib import Path

import dualmesh as dm
import numpy as np
import pytest
from dualmesh._core import InputError

pytest.importorskip("yaml")
EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "heated_plate.yaml"


def _python_plate():
    """The example case, written as a Python script."""
    mesh = dm.generate_rectangle_mesh(
        x_min=0.0, x_max=0.1, y_min=0.0, y_max=0.02, num_elements=[40, 8]
    )
    problem = dm.Problem(mesh, method="fem")
    heat = problem.add_physics("heat_transfer", thermal_conductivity=20.0, heat_source=1.0e6)
    heat.add_boundary_condition("temperature = 300", boundary="left")
    heat.add_boundary_condition("heat_flux = h*(temperature - 293)", boundary="top", h=25.0)
    solid = problem.add_physics(
        "solid_mechanics",
        formulation="plane_stress",
        youngs_modulus=2.0e11,
        poissons_ratio=0.3,
        thickness=0.01,
    )
    solid.add_boundary_condition("fixed", boundary="left")
    problem.add_coupling(
        "thermal_expansion",
        heat_transfer=heat,
        solid_mechanics=solid,
        linear_expansion_coeff=1.2e-5,
        stress_free_temperature=293,
    )
    problem.solve(report="none")
    return problem


def test_the_example_case_builds_the_problem_of_the_python_input(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    case = dm.run_case(EXAMPLE)
    python = _python_plate()
    for name in ("temperature", "disp_x", "disp_y"):
        assert np.array_equal(case.problem.values(name), python.values(name)), name
    # The numbers of the file are numbers, with the exponents of YAML 1.2 (1.0e6, 2.0e11).
    assert dm.load_case(EXAMPLE)["physics"]["solid"]["youngs_modulus"] == 2.0e11
    assert (tmp_path / "heated_plate.vtu").exists() and (tmp_path / "heated_plate.csv").exists()
    assert set(case.physics) == {"heat", "solid"}


def test_the_heat_of_the_example_leaves_through_its_cooled_edges(tmp_path, monkeypatch):
    # 1 MW/m^3 in 0.1 m x 0.02 m (per unit depth) is 2000 W/m: it leaves through the held left
    # edge and the cooled top edge, h * integral of (T - 293) along the top.
    monkeypatch.chdir(tmp_path)
    problem = dm.run_case(EXAMPLE).problem
    left = abs(problem.total_reaction("temperature", "left"))
    top = 25.0 * (problem.boundary_integral("temperature", "top") - 293.0 * 0.1)
    assert left + top == pytest.approx(1.0e6 * 0.1 * 0.02, rel=1e-8)
    temperature = np.asarray(problem.values("temperature"))
    assert 300.0 < temperature.max() < 300.0 + 1.0e6 * 0.1**2 / (2 * 20.0)


def test_a_transient_case_from_a_dict():
    case = {
        "mesh": {"generate": "line", "start": 0.0, "end": 1.0, "num_elements": 10},
        "physics": {
            "heat": {
                "type": "heat_transfer",
                "thermal_conductivity": 1.0,
                "density": 1.0,
                "specific_heat": 1.0,
                "initial_condition": 1.0,
                "conditions": [{"boundary": ["left", "right"], "condition": "temperature = 0"}],
            }
        },
        "solve": {"study": "transient", "end_time": 0.1, "time_step": 0.001, "report": "none"},
    }
    problem = dm.run_case(case).problem
    middle = problem.sample("temperature", [[0.5, 0.0, 0.0]])[0]
    # The first Fourier mode dominates: 4/pi exp(-pi^2 t) at t = 0.1 is 0.475.
    assert middle == pytest.approx(4 / np.pi * np.exp(-(np.pi**2) * 0.1), rel=0.03)


PLATE = {
    "mesh": {
        "generate": "rectangle",
        "x_min": 0,
        "x_max": 1,
        "y_min": 0,
        "y_max": 1,
        "num_elements": [2, 2],
    },
    "physics": {"heat": {"type": "heat_transfer", "thermal_conductivity": 1.0}},
}


@pytest.mark.parametrize(
    "change, message",
    [
        ({"sovle": {}}, "unknown key 'sovle'. Did you mean 'solve'"),
        (
            {"mesh": {"generate": "rectangel"}},
            "unknown generator 'rectangel'. Did you mean 'rectangle'",
        ),
        ({"mesh": {"generate": "rectangle", "file": "a.msh"}}, "one of 'generate'"),
        ({"solve": {"study": "stady"}}, "unknown study 'stady'. Did you mean 'steady'"),
        (
            {
                "physics": {
                    "heat": {
                        "type": "heat_transfer",
                        "thermal_conductivity": 1.0,
                        "conditions": [{"condition": "temperature = 0"}],
                    }
                }
            },
            "a condition needs 'boundary'",
        ),
        (
            {
                "couplings": [
                    {
                        "type": "thermal_expansion",
                        "heat_transfer": "warm",
                        "solid_mechanics": "heat",
                        "linear_expansion_coeff": 1e-5,
                        "stress_free_temperature": 0,
                    }
                ]
            },
            "names the physics 'warm'",
        ),
        ({"output": {"fields": "a.vtu", "plot": "a.png"}}, "unknown key 'plot'"),
    ],
)
def test_every_name_of_a_case_is_checked(change, message):
    case = {**PLATE, **change}
    with pytest.raises(InputError, match=message):
        dm.run_case(case)
