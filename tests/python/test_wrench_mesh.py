# SPDX-License-Identifier: LGPL-2.1-or-later
"""The imported wrench mesh of examples/wrench: its named surfaces, and the
geometric predicates that select the same faces on a mesh without names."""

from pathlib import Path

import pytest

dualmesh = pytest.importorskip("dualmesh")
pytest.importorskip("meshio")

wrench_mesh_file = Path(__file__).resolve().parents[2] / "examples" / "wrench" / "wrench.msh"
mm = 1.0e-3


def faces(mesh, name):
    return sorted(tuple(f) for f in mesh.sideset(name))


def test_boundary_condition_named_after_a_side_set():
    mesh = dualmesh.read_mesh(str(wrench_mesh_file))
    problem = dualmesh.Problem(mesh, method="fem")
    for v in ("u", "v", "w"):
        problem.add_variable(v)
    problem.add_boundary_condition("traction", "grip", variable="v", traction=-2.5e5)
    with pytest.raises(ValueError, match="the mesh has no boundary 'nut'"):
        problem.add_boundary_condition("Dirichlet", "nut", variable="u", value=0.0)
