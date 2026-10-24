# SPDX-License-Identifier: LGPL-2.1-or-later
"""Abbreviations in the input keep their capitals (CODE_AND_DOCS_RULES 4.1), and proper names
their capital letter; the lower-case spelling is accepted as well."""

from __future__ import annotations

import dualmesh as dm
import numpy as np
import pytest
from dualmesh._core import InputError


def heat(method):
    mesh = dm.generate_rectangle_mesh(x=[0, 1], y=[0, 1], num_elements=[4, 4])
    problem = dm.Problem(mesh, method=method)
    physics = problem.add_physics("heat_transfer", thermal_conductivity=1.0, heat_source=1.0)
    physics.add_boundary_condition("temperature = 0", boundary=["left", "right", "bottom", "top"])
    return problem


@pytest.mark.parametrize(
    "written, lower", [("FEM", "fem"), ("DMCDM", "dmcdm"), ("HFVM", "hfvm"), ("ZFVM", "zfvm")]
)
def test_a_method_is_written_in_capitals_and_accepted_in_lower_case(written, lower):
    upper, small = heat(written), heat(lower)
    assert upper.method == small.method == written
    upper.solve(report="none")
    small.solve(report="none")
    np.testing.assert_array_equal(upper.values("temperature"), small.values("temperature"))


@pytest.mark.parametrize(
    "options",
    [
        {"linear_solver": "GMRES", "preconditioner": "ILU"},
        {"linear_solver": "BiCGStab", "preconditioner": "Jacobi"},
        {"linear_solver": "LU", "nonlinear_solver": "Newton"},
        {"linear_solver": "bicgstab", "preconditioner": "amg", "nonlinear_solver": "picard"},
    ],
)
def test_solvers_and_preconditioners_take_either_spelling(options):
    reference = heat("FEM")
    reference.solve(report="none")
    problem = heat("FEM")
    problem.solve(report="none", linear_tolerance=1e-12, **options)
    np.testing.assert_allclose(
        problem.values("temperature"), reference.values("temperature"), atol=1e-10
    )


def test_a_misspelled_preconditioner_is_named_in_capitals():
    with pytest.raises(ValueError, match="Did you mean 'LSC'"):
        heat("FEM").solve(report="none", linear_solver="GMRES", preconditioner="lcs")


def test_an_unknown_method_is_refused():
    with pytest.raises((InputError, ValueError, RuntimeError)):
        heat("FVM")
