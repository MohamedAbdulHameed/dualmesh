# SPDX-License-Identifier: LGPL-2.1-or-later
"""The conditions of solids that act about an axis: a prescribed rotation and a torque."""

from __future__ import annotations

import math

import dualmesh
import numpy
import pytest
from dualmesh._core import InputError

methods = ["FEM", "DMCDM"]
E, nu = 2e11, 0.3
G = E / (2 * (1 + nu))


@pytest.mark.parametrize("method", methods)
@pytest.mark.parametrize("large", [False, True])
def test_a_body_turned_by_its_whole_boundary_is_stress_free(method, large):
    """Every face of a box turned by 0.3 rad about a skew axis through a point off the body: the
    body turns rigidly, by the linear rotation angle k x r at small strain and by the finite
    rotation of Rodrigues at large deformation, with no stress either way."""
    mesh = dualmesh.generate_box_mesh(0, 1, 0, 1, 0, 1, 2, 2, 2)
    problem = dualmesh.Problem(mesh, method=method)
    solid = problem.add_physics(
        "solid_mechanics", youngs_modulus=E, poissons_ratio=nu, large_deformation=large
    )
    axis, center, angle = numpy.array([1.0, 2.0, 2.0]) / 3.0, [0.5, -1.0, 2.0], 0.3
    solid.add_boundary_condition(
        "rotation",
        boundary=list(mesh.sideset_names()),
        axis=list(3 * axis),
        center=center,
        angle=0.3,
    )
    problem.solve(report="none")
    r = numpy.asarray(mesh.points()) - center
    if large:
        K = numpy.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        rotation = numpy.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * K @ K
        expected = r @ rotation.T - r
    else:
        expected = angle * numpy.cross(axis, r)
    u = numpy.c_[[problem.values(n) for n in ("disp_x", "disp_y", "disp_z")]].T
    assert u == pytest.approx(expected, abs=1e-12)
    stress = problem.property_at_centroids("stress")
    assert numpy.abs(stress).max() < 1e-6 * E * angle


def closed_annulus(a, b, radial, around):
    """An annulus a < r < b of quadrilaterals, closed: the nodes of the 360-degree edge are those
    of the 0-degree edge."""
    sector = dualmesh.generate_annulus_mesh(a, b, radial, around, start_angle=0.0, end_angle=360.0)
    points = numpy.asarray(sector.points())[:, :2]
    first = {}
    same = {n: first.setdefault(tuple(numpy.round(p, 9)), n) for n, p in enumerate(points)}
    kept = sorted(set(same.values()))
    index = {n: k for k, n in enumerate(kept)}
    cells = [[index[same[n]] for n in sector.element_nodes(e)] for e in range(sector.num_elements)]
    mesh = dualmesh.mesh_from_arrays(points[kept], [("Quad4", numpy.array(cells))])
    # The predicate sees the middle of a side's chord, inside the arc.
    mesh.add_sideset_by_predicate("inner", lambda x, y, z: abs(math.hypot(x, y) - a) < 1e-2 * a)
    mesh.add_sideset_by_predicate("outer", lambda x, y, z: abs(math.hypot(x, y) - b) < 1e-2 * a)
    return mesh


@pytest.mark.parametrize("method", methods)
def test_a_torque_twists_an_annulus_as_the_closed_form(method):
    """A plane-strain annulus a < r < b, its outer edge held and its inner edge loaded by a torque
    T: the support takes the moment T and no force, exactly in FEM (in DMCDM the control volumes
    lump the traction at the nodes, and the moment of the lumped loads is T to 3.4e-3 on this
    mesh, falling as h^2), and the hoop displacement is T r (1/b^2 - 1/r^2) / (4 pi G h), which the mesh
    approaches."""
    a, b, T, h = 0.5, 1.0, 1e6, 0.2
    mesh = closed_annulus(a, b, 16, 64)
    problem = dualmesh.Problem(mesh, method=method)
    solid = problem.add_physics(
        "solid_mechanics",
        formulation="plane_strain",
        youngs_modulus=E,
        poissons_ratio=nu,
        thickness=h,
    )
    solid.add_boundary_condition("fixed", boundary="outer")
    solid.add_boundary_condition(f"torque = {T}", boundary="inner", center=[0.0, 0.0])
    problem.solve(report="none")
    xy = numpy.asarray(mesh.points())[:, :2]
    fx = problem.boundary_flux("disp_x", "outer", per_node=True)
    fy = problem.boundary_flux("disp_y", "outer", per_node=True)
    # The flux leaves the body: it is the force of the body on the support.
    moment = sum(xy[n, 0] * fy[n] - xy[n, 1] * fx[n] for n in fx)
    assert moment == pytest.approx(T, rel=1e-10 if method == "FEM" else 5e-3)
    force = solid.reaction_force("outer")
    assert abs(force["x"]) < 1e-9 * T and abs(force["y"]) < 1e-9 * T
    r = numpy.hypot(xy[:, 0], xy[:, 1])
    hoop = (xy[:, 0] * problem.values("disp_y") - xy[:, 1] * problem.values("disp_x")) / r
    expected = T * r * (1 / r**2 - 1 / b**2) / (4 * math.pi * G * h)
    inner = numpy.abs(r - a) < 1e-9
    assert hoop[inner] == pytest.approx(expected[inner], rel=3e-3)


def test_the_input_of_rotation_and_torque_is_checked():
    plane = dualmesh.Problem(dualmesh.generate_rectangle_mesh(0, 1, 0, 1, 1, 1))
    solid = plane.add_physics("solid_mechanics", formulation="plane_strain", youngs_modulus=E)
    with pytest.raises(InputError, match="turns about z; give the center only"):
        solid.add_boundary_condition(
            "rotation", boundary="left", axis=[0, 0, 1], center=[0, 0], angle=0.1
        )
    with pytest.raises(InputError, match="give the boundary, the angle"):
        solid.add_boundary_condition("rotation", boundary="left", center=[0, 0])
    box = dualmesh.Problem(dualmesh.generate_box_mesh(0, 1, 0, 1, 0, 1, 1, 1, 1))
    solid = box.add_physics("solid_mechanics", youngs_modulus=E)
    with pytest.raises(InputError, match="give the axis"):
        solid.add_boundary_condition("torque = 5", boundary="left", center=[0, 0, 0])


def turned_block(dimension, rotation):
    """The unit square (or cube) on 3 elements a side, turned by 'rotation', with the side sets
    of its faces named in its own axes: left, bottom, back (x', y', z' = 0) and top (y' = 1)."""
    if dimension == 2:
        base = dualmesh.generate_rectangle_mesh(0, 1, 0, 1, 3, 3)
    else:
        base = dualmesh.generate_box_mesh(0, 1, 0, 1, 0, 1, 3, 3, 3)
    local = numpy.asarray(base.points())[:, :dimension]
    cells = numpy.array([base.element_nodes(e) for e in range(base.num_elements)])
    mesh = dualmesh.mesh_from_arrays(local @ rotation.T, [(base.element_type(0), cells)])

    def face(axis, at):
        def on(x, y, z):
            own = rotation.T @ numpy.array([x, y, z][:dimension])
            return abs(own[axis] - at) < 1e-9

        return on

    for name, axis, at in (("left", 0, 0.0), ("bottom", 1, 0.0), ("top", 1, 1.0), ("back", 2, 0.0)):
        if axis < dimension:
            mesh.add_sideset_by_predicate(name, face(axis, at))
    return mesh


@pytest.mark.parametrize("method", methods)
@pytest.mark.parametrize("dimension", [2, 3])
def test_rollers_on_inclined_faces_give_the_uniaxial_state(method, dimension):
    """A block turned in space, on rollers on its faces x' = 0, y' = 0 (and z' = 0), pressed by p
    on y' = 1: the state is the uniaxial compression of its own axes, exactly, with the normal
    displacement zero on the rollers and the faces sliding freely along them; the rollers carry
    p over the pressed area, along their normals."""
    p = 1e7
    if dimension == 2:
        c, s = math.cos(0.5), math.sin(0.5)
        rotation = numpy.array([[c, -s], [s, c]])
    else:
        rotation = numpy.array(
            [[math.cos(0.4), 0, math.sin(0.4)], [0, 1, 0], [-math.sin(0.4), 0, math.cos(0.4)]]
        ) @ numpy.array(
            [[1, 0, 0], [0, math.cos(1.1), -math.sin(1.1)], [0, math.sin(1.1), math.cos(1.1)]]
        )
    mesh = turned_block(dimension, rotation)
    problem = dualmesh.Problem(mesh, method=method)
    solid = problem.add_physics(
        "solid_mechanics",
        formulation="plane_strain" if dimension == 2 else None,
        youngs_modulus=E,
        poissons_ratio=nu,
    )
    for face in ["left", "bottom"] + (["back"] if dimension == 3 else []):
        solid.add_boundary_condition("normal_displacement = 0", boundary=face)
    solid.add_boundary_condition(f"pressure = {p}", boundary="top")
    problem.solve(report="none")
    names = ["disp_x", "disp_y", "disp_z"][:dimension]
    u = numpy.c_[[problem.values(n) for n in names]].T @ rotation  # In the block's own axes
    own = numpy.asarray(mesh.points())[:, :dimension] @ rotation
    if dimension == 2:
        strain = [p * nu * (1 + nu) / E, -p * (1 - nu**2) / E]
    else:
        strain = [p * nu / E, -p / E, p * nu / E]
    assert u == pytest.approx(own * strain, abs=1e-9 * p / E)
    support = solid.reaction_force("bottom")
    assert [support[k] for k in "xyz"[:dimension]] == pytest.approx(
        list(p * rotation[:, 1]), rel=1e-9, abs=1e-9 * p
    )
