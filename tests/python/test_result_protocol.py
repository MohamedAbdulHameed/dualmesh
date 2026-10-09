# SPDX-License-Identifier: LGPL-2.1-or-later
"""Every result provides summary(), to_dict(), write_json(path) and
write_csv(path)."""

from __future__ import annotations

import csv
import json

import dualmesh as dm
import pytest
from dualmesh import mms, uq

PROTOCOL = ("summary", "to_dict", "write_json", "write_csv")


def _check(result, tmp_path, name):
    for method in PROTOCOL:
        assert callable(getattr(result, method, None)), f"{type(result).__name__} lacks {method}"
    assert isinstance(result.summary(), str) and result.summary()
    assert isinstance(result.tables, dict) and result.tables and all(isinstance(t, dm.Table) for t in result.tables.values())
    numbers = result.to_dict()
    result.write_json(tmp_path / f"{name}.json")
    assert json.loads((tmp_path / f"{name}.json").read_text()).keys() == numbers.keys()
    result.write_csv(tmp_path / f"{name}.csv")
    with open(tmp_path / f"{name}.csv") as stream:
        rows = list(csv.reader(stream))
    assert len(rows) >= 2
    return numbers, rows


def _square(n=4):
    return dm.generate_rectangle_mesh(x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0, num_x_elements=n, num_y_elements=n)


def test_a_solve_result_follows_the_protocol(tmp_path):
    problem = dm.Problem(_square(), method="fem")
    pde = problem.add_physics("coefficient_form_PDE", "pde", diffusion_coefficient="1 + u", source=1.0)
    pde.add_boundary_condition("Dirichlet_boundary_condition", "left", value=0.0)
    numbers, rows = _check(problem.solve(), tmp_path, "solve")
    assert numbers["converged"] is True
    assert rows[0] == ["load_step", "load_factor", "iteration", "residual_norm", "step_norm"]
    assert len(rows) - 1 == len(numbers["history"])


def test_a_convergence_study_follows_the_protocol(tmp_path):
    study = mms.ManufacturedSolution({"u": "sin(x)*y"}, dimension=2)
    study.add_physics("coefficient_form_PDE", "pde", diffusion_coefficient=1.0)
    result = study.convergence_study(_square, [4, 8], method="fem")
    numbers, rows = _check(result, tmp_path, "convergence")
    assert numbers["orders"]["u l2"][0] == pytest.approx(2.0, abs=0.3)
    assert len(rows) == 3


def test_the_uncertainty_results_follow_the_protocol(tmp_path):
    inputs = {"a": uq.Normal(1.0, 0.1), "b": uq.Uniform(0.0, 1.0)}

    def model(a, b):
        return {"y": a + 2.0 * b}

    runs = uq.propagate(model, inputs, 32, method="sobol")
    _, rows = _check(runs, tmp_path, "runs")
    assert rows[0] == ["a", "b", "y", "failed"] and len(rows) == 33
    indices = uq.sobol(model, inputs, 64)
    _, rows = _check(indices, tmp_path, "sobol")
    assert len(rows) == 3
    posterior = uq.calibrate(model, inputs, {"y": 2.0}, {"y": 0.1}, surrogate=None, samples=400, chains=2)
    _, rows = _check(posterior, tmp_path, "posterior")
    assert [r[0] for r in rows[1:]] == ["a", "b"]


def test_a_table_prints_aligned_and_writes_units_in_the_header(tmp_path):
    table = dm.Table(["region", "volume", "flux", "count"], ["", "m^3", "1/(m^2 s)", "-"], [["fuel", 1.5, 2.0e17, 3], ["reflector", 0.25, float("nan"), 4]], title="Regions")
    text = str(table)
    assert text.splitlines()[0] == "Regions"
    assert "volume (m^3)" in text and "flux (1/(m^2 s))" in text and "count" in text and "count (" not in text
    table.write_csv(tmp_path / "t.csv")
    lines = (tmp_path / "t.csv").read_text().splitlines()
    assert lines[0] == "region,volume (m^3),flux (1/(m^2 s)),count"
    assert lines[2] == "reflector,0.25,,4"
    assert table["volume"].tolist() == [1.5, 0.25]
    with pytest.raises(KeyError, match="Did you mean 'volume'"):
        table["volum"]
