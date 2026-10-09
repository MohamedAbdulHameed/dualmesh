# SPDX-License-Identifier: LGPL-2.1-or-later
"""The imported wrench mesh of examples/wrench: its named surfaces, and the
geometric predicates that select the same faces on a mesh without names."""

from pathlib import Path

import pytest

dm = pytest.importorskip("dualmesh")
pytest.importorskip("meshio")

MESH = Path(__file__).resolve().parents[2] / "examples" / "wrench" / "wrench.msh"
MM = 1.0e-3


def faces(mesh, name):
    return sorted(tuple(f) for f in mesh.sideset(name))


def test_physical_groups_become_side_sets():
    mesh = dm.read_mesh(str(MESH))
    assert mesh.dimension == 3
    assert sorted(mesh.sideset_names()) == ["grip", "jaws"]
    assert len(faces(mesh, "jaws")) == 64
    assert len(faces(mesh, "grip")) == 392


def test_predicates_select_the_same_faces():
    mesh = dm.read_mesh(str(MESH))
    mesh.add_sideset_by_predicate(
        "jaws_by_position",
        lambda x, y, z: abs(abs(y) - 6.5 * MM) < 1e-6 and -15.76 * MM < x < -8.24 * MM,
    )
    mesh.add_sideset_by_predicate(
        "grip_by_position",
        lambda x, y, z: abs(abs(z) - 2.5 * MM) < 1e-6 and 120 * MM < x < 145 * MM,
    )
    assert faces(mesh, "jaws_by_position") == faces(mesh, "jaws")
    assert faces(mesh, "grip_by_position") == faces(mesh, "grip")


def test_boundary_condition_named_after_a_side_set():
    mesh = dm.read_mesh(str(MESH))
    problem = dm.Problem(mesh, method="fem")
    for v in ("u", "v", "w"):
        problem.add_variable(v)
    problem.add_boundary_condition(
        "traction_boundary_condition", "grip", variable="v", traction=-2.5e5
    )
    with pytest.raises(ValueError, match="name the condition after a boundary"):
        problem.add_boundary_condition(
            "Dirichlet_boundary_condition", "nut", variable="u", value=0.0
        )
