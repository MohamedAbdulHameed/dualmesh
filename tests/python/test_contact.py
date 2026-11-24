# SPDX-License-Identifier: LGPL-2.1-or-later
"""Mortar contact: the contact patch test, a gap that closes, the balance of the contact forces
and Coulomb friction at its limit."""

from __future__ import annotations

import dualmesh
import numpy
import pytest
from dualmesh._core import InputError

E, nu = 100e9, 0.3
plane = E / (1 - nu**2)


def two_blocks(lower_x=4, upper_x=5, gap=0.0, rows=2):
    """Two blocks on top of each other, [0, 1] x [0, 0.5] and [0, 1] x [0.5 + gap, 1 + gap],
    meshed apart and so non-matching at their faces, with the side sets bottom, top, left,
    lower_top (primary) and upper_bottom (secondary)."""
    parts = [
        dualmesh.generate_rectangle_mesh(0, 1, 0, 0.5, lower_x, rows),
        dualmesh.generate_rectangle_mesh(0, 1, 0.5 + gap, 1.0 + gap, upper_x, rows),
    ]
    points, connectivity, blocks, offset = [], [], [], 0
    for block, part in enumerate(parts):
        points.append(numpy.asarray(part.points())[:, :2])
        for element in range(part.num_elements):
            connectivity.append([n + offset for n in part.element_nodes(element)])
            blocks.append(block)
        offset += len(points[-1])
    xy = numpy.vstack(points)
    mesh = dualmesh.mesh_from_arrays(
        xy, [("Quad4", numpy.array(connectivity))], blocks=blocks, dimension=2
    )

    def faces(first, count, y):
        row = [n for n in range(first, first + count) if abs(xy[n, 1] - y) < 1e-12]
        row.sort(key=lambda n: xy[n, 0])
        return [[a, b] for a, b in zip(row, row[1:])]

    lower_nodes = len(points[0])
    mesh.add_sideset_from_faces("lower_top", faces(0, lower_nodes, 0.5))
    mesh.add_sideset_from_faces("upper_bottom", faces(lower_nodes, len(points[1]), 0.5 + gap))
    mesh.add_sideset_by_predicate("bottom", lambda x, y, z: abs(y) < 1e-12)
    mesh.add_sideset_by_predicate("top", lambda x, y, z: abs(y - 1.0 - gap) < 1e-12)
    mesh.add_sideset_by_predicate("left", lambda x, y, z: abs(x) < 1e-12)
    return mesh


def contact_problem(mesh, friction=0.0, method="FEM"):
    problem = dualmesh.Problem(mesh, method=method)
    solid = problem.add_physics(
        "solid_mechanics", formulation="plane_strain", youngs_modulus=E, poissons_ratio=nu
    )
    solid.add_boundary_condition(
        "contact",
        primary="lower_top",
        secondary="upper_bottom",
        friction_coefficient=friction,
    )
    solid.add_boundary_condition("disp_y = 0", boundary="bottom")
    return problem, solid


@pytest.mark.parametrize("method", ["FEM", "DMCDM"])
def test_the_contact_patch_test_passes_on_non_matching_meshes(method):
    """A uniform pressure p on two stacked blocks whose faces do not match: the contact pressure
    is p at every secondary node, the stress is -p everywhere, the top moves by p (1 - nu^2)/E,
    and the contact forces on the two bodies balance (each carries p over the width)."""
    p = 5e6
    problem, solid = contact_problem(two_blocks(), method=method)
    solid.add_boundary_condition("disp_x = 0", boundary="left")
    solid.add_boundary_condition(f"traction = [0, -{p}]", boundary="top")
    problem.solve(report="none")
    pressures = numpy.array(list(solid.contact_pressure("upper_bottom").values()))
    assert pressures == pytest.approx(p, rel=1e-9)
    stress = problem.property_at_centroids("stress")
    assert stress[:, 1] == pytest.approx(-p, rel=1e-9)
    assert problem.maximum("disp_y") == pytest.approx(0.0, abs=1e-18)
    assert problem.minimum("disp_y") == pytest.approx(-p / plane, rel=1e-9)
    upper = solid.reaction_force("upper_bottom")["y"]
    lower = solid.reaction_force("lower_top")["y"]
    assert upper == pytest.approx(p, rel=1e-9)
    assert lower == pytest.approx(-p, rel=1e-9)


@pytest.mark.parametrize("closing", [0.5, 3.0])
def test_a_gap_closes_only_when_the_bodies_meet(closing):
    """An initial gap of 1e-4 and the top pushed down by closing x 1e-4: below the gap the
    bodies stay apart, with no pressure and no stress in the lower one; beyond it the extra
    displacement compresses both, sigma = (closing - 1) 1e-4 E / (1 - nu^2)."""
    gap = 1e-4
    problem, solid = contact_problem(two_blocks(gap=gap))
    solid.add_boundary_condition("disp_x = 0", boundary="left")
    solid.add_boundary_condition(f"disp_y = {-closing * gap}", boundary="top")
    problem.solve(report="none")
    pressures = numpy.array(list(solid.contact_pressure("upper_bottom").values()))
    expected = max(0.0, (closing - 1.0) * gap) * plane
    assert pressures == pytest.approx(expected, rel=1e-8, abs=1e-6 * plane * gap)
    assert solid.reaction_force("bottom")["y"] == pytest.approx(expected, rel=1e-8, abs=1e-3)


@pytest.mark.parametrize("slide", [1e-7, 1e-3])
def test_coulomb_friction_sticks_below_its_limit_and_slides_at_it(slide):
    """The upper block pressed down by 1e-4 and its top moved sideways: a small move sticks
    (the tangential traction is below mu times the pressure), a move well beyond the elastic
    shear slides with the tangential traction mu times the pressure at every node, so the
    friction force is mu times the normal force (at a move of 1e-4, of the size of the elastic
    displacements, the ends stick and the middle slides).  The forces on the two surfaces
    balance."""
    mu = 0.3
    problem, solid = contact_problem(two_blocks(lower_x=6, upper_x=5), friction=mu)
    solid.add_boundary_condition("disp_x = 0", boundary="bottom")
    solid.add_boundary_condition(f"disp_x = {slide}", boundary="top")
    solid.add_boundary_condition("disp_y = -1e-4", boundary="top")
    problem.solve(report="none")
    pressure = numpy.array(list(solid.contact_pressure("upper_bottom").values()))
    traction = problem.values("contact_traction")[list(solid.contact_pressure("upper_bottom"))]
    assert numpy.all(pressure > 0.0)
    normal = solid.reaction_force("upper_bottom")["y"]
    tangential = solid.reaction_force("upper_bottom")["x"]
    lower = solid.reaction_force("lower_top")
    assert lower["x"] == pytest.approx(-tangential, rel=1e-10)
    assert lower["y"] == pytest.approx(-normal, rel=1e-10)
    if slide < 1e-6:
        assert numpy.all(numpy.abs(traction) < mu * pressure)
        assert abs(tangential) < mu * abs(normal)
    else:
        assert numpy.abs(traction) == pytest.approx(mu * pressure, rel=1e-8)
        assert abs(tangential) == pytest.approx(mu * abs(normal), rel=1e-8)


def test_the_input_of_contact_is_checked():
    mesh = two_blocks()
    problem = dualmesh.Problem(mesh)
    solid = problem.add_physics(
        "solid_mechanics", formulation="plane_strain", youngs_modulus=E, poissons_ratio=nu
    )
    with pytest.raises(InputError, match="give primary and secondary"):
        solid.add_boundary_condition("contact", primary="lower_top")
    with pytest.raises(InputError, match="no contact has the secondary surface"):
        solid.contact_pressure("top")
