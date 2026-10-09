# SPDX-License-Identifier: LGPL-2.1-or-later
"""Post-processors: the scalars computed after every solve and time step."""

from __future__ import annotations

import dualmesh as dm
import numpy as np
import pytest


def _linear_field_problem():
    """u = x on the unit square: integral 1/2, average 1/2, maximum 1."""
    mesh = dm.generate_rectangle_mesh(0, 1, 0, 1, 4, 4)
    problem = dm.Problem(mesh, method="fem")
    problem.add_variable("u")
    problem.add_kernel("diffusion", "diffusion", variable="u")
    problem.add_boundary_condition(
        "Dirichlet_boundary_condition", "left", variable="u", boundary="left", value=0.0
    )
    problem.add_boundary_condition(
        "Dirichlet_boundary_condition", "right", variable="u", boundary="right", value=1.0
    )
    return problem


def test_the_scalar_types_give_the_exact_values_of_a_linear_field():
    problem = _linear_field_problem()
    problem.add_postprocessor("variable_integral", "integral", variable="u")
    problem.add_postprocessor("variable_average", "average", variable="u")
    problem.add_postprocessor("point_value", "at_point", variable="u", point=[0.3, 0.7])
    problem.add_postprocessor("nodal_extreme_value", "largest", variable="u")
    problem.add_postprocessor("nodal_extreme_value", "smallest", variable="u", value_type="min")
    problem.add_postprocessor("total_reaction", "flux_in", variable="u", boundary="left")
    problem.add_postprocessor(
        "boundary_flux_integral", "flux_out", kernel="diffusion", boundary="right"
    )
    problem.solve()
    values = {k: v[-1] for k, v in problem.postprocessor_values().items()}
    assert values["integral"] == pytest.approx(0.5)
    assert values["average"] == pytest.approx(0.5)
    assert values["at_point"] == pytest.approx(0.3)
    assert values["largest"] == pytest.approx(1.0)
    assert values["smallest"] == pytest.approx(0.0)
    # A unit gradient carries a unit flux through the unit-height boundaries.
    assert abs(values["flux_in"]) == pytest.approx(1.0)
    assert abs(values["flux_out"]) == pytest.approx(1.0)


def test_a_transient_records_every_accepted_step(tmp_path):
    mesh = dm.generate_line_mesh(0.0, 1.0, 10)
    problem = dm.Problem(mesh, method="fem")
    problem.add_variable("u", initial_condition=1.0)
    problem.add_kernel("time_derivative", "rate", variable="u")
    problem.add_kernel("diffusion", "diffusion", variable="u")
    problem.add_postprocessor("variable_average", "mean", variable="u")
    problem.solve_transient(end_time=0.3, time_step=0.1)
    history = problem.postprocessor_values()
    assert history["time"] == pytest.approx([0.0, 0.1, 0.2, 0.3])
    # Nothing leaves through the insulated ends: the mean is conserved.
    assert history["mean"] == pytest.approx(np.ones(4))
    problem.write_postprocessor_csv(tmp_path / "history.csv")
    lines = (tmp_path / "history.csv").read_text().splitlines()
    assert lines[0] == "time,mean" and len(lines) == 5


def test_misuse_is_refused_with_the_valid_choices():
    problem = _linear_field_problem()
    with pytest.raises(ValueError, match="variable_average"):
        problem.add_postprocessor("Average", "a", variable="u")
    with pytest.raises(ValueError, match="takes: variable"):
        problem.add_postprocessor("variable_average", "a", field="u")
    with pytest.raises(ValueError, match="'max' or 'min'"):
        problem.add_postprocessor("nodal_extreme_value", "e", variable="u", value_type="largest")
    # A 0.1 name is recognised and the new one suggested.
    with pytest.raises(ValueError, match="use 'variable_average'"):
        problem.add_postprocessor("VariableAverage", "a", variable="u")
    with pytest.raises(ValueError, match="use 'total_reaction'"):
        problem.add_postprocessor("Reaction", "a", variable="u", boundary="left")
    problem.add_postprocessor("variable_average", "a", variable="u")
    with pytest.raises(ValueError, match="already exists"):
        problem.add_postprocessor("variable_average", "a", variable="u")
