# SPDX-License-Identifier: LGPL-2.1-or-later
"""Mortar contact: the contact patch test, a gap that closes, the balance of the contact forces,
Coulomb friction at its limit, and contact in axisymmetry and at finite strain."""

from __future__ import annotations

import dualmesh
import numpy
import pytest
from dualmesh._core import InputError
from scipy.optimize import brentq

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
    """The upper block pressed down by 1e-4 and its top moved sideways along +x.  The upper
    block carries no other load, so the friction force on it is minus the sideways force
    applied at its top (its equilibrium), and the forces on the two surfaces balance.  A small
    move sticks: the tangential traction is below mu times the pressure at every node.  A move
    well beyond the elastic shear slides: at every node the tangential traction is mu times the
    pressure, and the friction force on the upper block is mu times the normal force, along -x,
    opposite to the sliding (at a move of 1e-4, of the size of the elastic displacements, the
    ends stick and the middle slides)."""
    mu = 0.3
    problem, solid = contact_problem(two_blocks(lower_x=6, upper_x=5), friction=mu)
    solid.add_boundary_condition("disp_x = 0", boundary="bottom")
    solid.add_boundary_condition(f"disp_x = {slide}", boundary="top")
    solid.add_boundary_condition("disp_y = -1e-4", boundary="top")
    problem.solve(report="none")
    pressure = numpy.array(list(solid.contact_pressure("upper_bottom").values()))
    traction = problem.values("contact_traction")[list(solid.contact_pressure("upper_bottom"))]
    assert numpy.all(pressure > 0.0)
    # The forces on the upper block through its bottom surface: the contact pushes it up.
    normal = solid.reaction_force("upper_bottom")["y"]
    tangential = solid.reaction_force("upper_bottom")["x"]
    assert normal > 0.0
    lower = solid.reaction_force("lower_top")
    assert lower["x"] == pytest.approx(-tangential, rel=1e-10)
    assert lower["y"] == pytest.approx(-normal, rel=1e-10)
    applied = solid.reaction_force("top")["x"]
    assert applied > 0.0
    # The residual of the free nodes, below the tolerance of Newton's method, bounds the error.
    assert tangential == pytest.approx(-applied, rel=1e-8, abs=1e-8 * normal)
    if slide < 1e-6:
        assert numpy.all(numpy.abs(traction) < mu * pressure)
        assert abs(tangential) < mu * normal
    else:
        # The tangential traction of a secondary node is positive along the tangent (-n_y, n_x)
        # of its outward normal n = (0, -1), which is +x, and the force it puts on the upper
        # block is minus the traction along that tangent.
        assert traction == pytest.approx(mu * pressure, rel=1e-8)
        assert tangential == pytest.approx(-mu * normal, rel=1e-8)


def test_coaxial_cylinders_pressed_end_to_end_carry_the_pressure_in_axisymmetry():
    """Two coaxial cylinders of radius R = 1, stacked along z and meshed apart, the lower one on
    rollers at z = 0, held at r = 0 on the axis, and a uniform pressure p on the top.  The stress
    is uniaxial, sigma_zz = -p, and by Hooke's law eps_zz = -p / E and eps_rr = eps_tt =
    nu p / E: the contact pressure is p at every secondary node, the top moves down by p H / E
    with H = 1 the height, and the outer radius grows by nu p R / E.  Each body carries the
    force p pi R^2 through the contact, which the factor 2 pi r of the axisymmetric measure
    gives."""
    p = 5e6
    problem = dualmesh.Problem(two_blocks(), coordinates="axisymmetric")
    solid = problem.add_physics("solid_mechanics", youngs_modulus=E, poissons_ratio=nu)
    solid.add_boundary_condition("contact", primary="lower_top", secondary="upper_bottom")
    solid.add_boundary_condition("disp_z = 0", boundary="bottom")
    solid.add_boundary_condition("disp_r = 0", boundary="left")
    solid.add_boundary_condition(f"traction = [0, -{p}]", boundary="top")
    problem.solve(report="none")
    pressures = numpy.array(list(solid.contact_pressure("upper_bottom").values()))
    assert pressures == pytest.approx(p, rel=1e-9)
    stress = problem.property_at_centroids("stress")
    assert stress[:, 1] == pytest.approx(-p, rel=1e-9)
    assert numpy.delete(stress, 1, axis=1) == pytest.approx(0.0, abs=1e-9 * p)
    assert problem.minimum("disp_z") == pytest.approx(-p / E, rel=1e-9)
    assert problem.maximum("disp_r") == pytest.approx(nu * p / E, rel=1e-9)
    assert solid.reaction_force("upper_bottom")["z"] == pytest.approx(p * numpy.pi, rel=1e-9)
    assert solid.reaction_force("lower_top")["z"] == pytest.approx(-p * numpy.pi, rel=1e-9)


def test_contact_at_finite_strain_carries_the_Cauchy_pressure_of_the_deformed_surface():
    """The two blocks, of neo-Hookean material, compressed by 10 % between rollers.  A
    hyperelastic body is at finite strain, so the contact measures its pressure on the deformed
    surfaces.  The deformation is uniform, with the stretch lambda_y = 0.9 and the lateral
    stretch lambda_x of P_xx = 0.  For psi = mu/2 (I_C - 3) - mu ln J + Lame_lambda/2 (ln J)^2
    (Bonet and Wood 2008, Nonlinear Continuum Mechanics for Finite Element Analysis) the
    principal nominal stresses are P_i = mu (lambda_i - 1/lambda_i) + Lame_lambda ln(J) /
    lambda_i, with J = lambda_x lambda_y in plane strain.  The contact pressure is the Cauchy
    one, -sigma_yy = -P_yy / lambda_x, the force -P_yy per unit width spread over the deformed
    width lambda_x, which the undeformed measure of small strain would give as the nominal -P_yy,
    larger by the factor lambda_x, about 1.045."""
    shear_modulus = E / (2 * (1 + nu))
    Lame_lambda = E * nu / ((1 + nu) * (1 - 2 * nu))
    stretch_y = 0.9

    def nominal(stretch, J):
        return shear_modulus * (stretch - 1 / stretch) + Lame_lambda * numpy.log(J) / stretch

    stretch_x = brentq(lambda s: nominal(s, s * stretch_y), 1.0, 1.2, xtol=1e-15)
    P_yy = nominal(stretch_y, stretch_x * stretch_y)
    problem = dualmesh.Problem(two_blocks())
    solid = problem.add_physics(
        "solid_mechanics",
        formulation="plane_strain",
        youngs_modulus=E,
        poissons_ratio=nu,
        elasticity="neo-Hookean",
    )
    solid.add_boundary_condition("contact", primary="lower_top", secondary="upper_bottom")
    solid.add_boundary_condition("disp_y = 0", boundary="bottom")
    solid.add_boundary_condition("disp_x = 0", boundary="left")
    solid.add_boundary_condition(f"disp_y = {stretch_y - 1!r}", boundary="top")
    problem.solve(report="none")
    assert problem.maximum("disp_x") == pytest.approx(stretch_x - 1, rel=1e-8)
    pressures = numpy.array(list(solid.contact_pressure("upper_bottom").values()))
    assert pressures == pytest.approx(-P_yy / stretch_x, rel=1e-8)
    assert solid.reaction_force("upper_bottom")["y"] == pytest.approx(-P_yy, rel=1e-8)


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
