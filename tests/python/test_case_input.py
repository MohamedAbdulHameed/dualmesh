# SPDX-License-Identifier: LGPL-2.1-or-later
"""Case files: the physics level written as data (YAML) builds the same problem as the Python
input, with the same names, and every name is checked."""

from __future__ import annotations

from pathlib import Path

import dualmesh
import numpy
import pytest
from dualmesh._core import InputError

pytest.importorskip("yaml")
example = Path(__file__).resolve().parents[2] / "examples" / "heated_plate.yaml"


def test_the_heat_of_the_example_leaves_through_its_cooled_edges(tmp_path, monkeypatch):
    # 1 MW/m^3 in 0.1 m x 0.02 m (per unit depth) is 2000 W/m: it leaves through the held left
    # edge and the cooled top edge, h * integral of (T - 293) along the top.
    monkeypatch.chdir(tmp_path)
    problem = dualmesh.run_case(example).problem
    left = abs(-problem.boundary_flux("temperature", "left"))
    top = 25.0 * (problem.boundary_integral("temperature", "top") - 293.0 * 0.1)
    assert left + top == pytest.approx(1.0e6 * 0.1 * 0.02, rel=1e-8)
    temperature = numpy.asarray(problem.values("temperature"))
    assert 300.0 < temperature.max() < 300.0 + 1.0e6 * 0.1**2 / (2 * 20.0)


plate = {
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
                        "linear_expansion_coefficient": 1e-5,
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
    case = {**plate, **change}
    with pytest.raises(InputError, match=message):
        dualmesh.run_case(case)
