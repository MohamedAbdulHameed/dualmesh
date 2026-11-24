# SPDX-License-Identifier: LGPL-2.1-or-later
"""Heat transfer and contact across a gap between two bodies that share no
nodes: gap_heat_transfer and mortar_contact.

Two slabs, an inner one heated by a uniform source and an outer one cooled at
its far face, face each other across a gap. Every check has an exact answer:

* the temperatures of a gap with a given conductance (piecewise quadratic and
  linear), in one, two and three dimensions and in a cylinder;
* the contact pressure of two elastic slabs pushed together;
* the conservation of the heat that crosses the gap, to round-off, on
  matching and non-matching meshes;
* the Jacobian of the mortar contact against finite differences, with the gap
  open and closed.
"""

from __future__ import annotations

import dualmesh
import numpy
import pytest

methods = ["fem", "dmcdm"]
vertex_methods = ["fem", "dmcdm"]

# The inner slab is 0 < x < A, the gap A < x < B, the outer slab B < x < C.
A, B, C = 1.0e-2, 1.005e-2, 2.0e-2
height = 1.0e-2
k_inner, k_outer, source, conductance, t_cool = 3.0, 16.0, 1.0e7, 5000.0, 600.0


def _body(dimension, x_min, x_max, num_x, num_y):
    if dimension == 1:
        return dualmesh.generate_line_mesh(x_min, x_max, num_x)
    if dimension == 2:
        return dualmesh.generate_rectangle_mesh(
            x_min=x_min,
            x_max=x_max,
            y_min=0.0,
            y_max=height,
            num_x_elements=num_x,
            num_y_elements=num_y,
        )
    return dualmesh.generate_box_mesh(
        x_min=x_min,
        x_max=x_max,
        y_min=0.0,
        y_max=height,
        z_min=0.0,
        z_max=height,
        num_x_elements=num_x,
        num_y_elements=num_y,
        num_z_elements=num_y,
    )


def two_slabs(dimension, num_x, inner_num_y=1, outer_num_y=1, gap_end=B):
    """The two slabs as one mesh of two blocks, inner and outer, with the side
    sets left (x = 0), primary (x = A), secondary (x = gap_end), right (x = C)
    and, in two and three dimensions, bottom (y = 0)."""
    parts = [
        _body(dimension, 0.0, A, num_x, inner_num_y),
        _body(dimension, gap_end, C, num_x, outer_num_y),
    ]
    points, connectivity, blocks, offset = [], [], [], 0
    element_types = set()
    for block, part in enumerate(parts):
        points.append(numpy.asarray(part.points())[:, :dimension])
        for element in range(part.num_elements):
            connectivity.append([node + offset for node in part.element_nodes(element)])
            element_types.add(part.element_type(element))
            blocks.append(block)
        offset += part.num_nodes
    (element_type,) = element_types
    mesh = dualmesh.mesh_from_arrays(
        numpy.vstack(points),
        [(element_type, numpy.array(connectivity))],
        blocks=blocks,
        dimension=dimension,
    )
    mesh.set_block_name(0, "inner")
    mesh.set_block_name(1, "outer")
    tolerance = 1e-9 * C
    for name, position in (("left", 0.0), ("primary", A), ("secondary", gap_end), ("right", C)):
        mesh.add_sideset_by_predicate(
            name, lambda x, y, z, position=position: abs(x - position) < tolerance
        )
    if dimension > 1:
        mesh.add_sideset_by_predicate("bottom", lambda x, y, z: abs(y) < tolerance)
    return mesh


def cross_section(dimension):
    return {1: 1.0, 2: height, 3: height**2}[dimension]


def slab_temperature(x, primary_temperature, secondary_temperature, gap_end=B):
    """The exact temperature: a parabola in the heated slab and a straight
    line in the cooled one."""
    flux = source * A
    inner = primary_temperature + source * (A**2 - x**2) / (2 * k_inner)
    outer = t_cool + flux * (C - x) / k_outer
    return numpy.where(x <= 0.5 * (A + gap_end), inner, outer)


def conduction_problem(mesh, method, coordinates="cartesian", distributed=False):
    problem = dualmesh.Problem(
        mesh, method=method, coordinates=coordinates, distributed=distributed
    )
    problem.add_variable("temperature", initial_condition=t_cool)
    problem.add_kernel(
        "heat_conduction",
        variable="temperature",
        thermal_conductivity=k_inner,
        block=["inner"],
    )
    problem.add_kernel(
        "heat_conduction",
        variable="temperature",
        thermal_conductivity=k_outer,
        block=["outer"],
    )
    problem.add_kernel("body_force", variable="temperature", value=source, block=["inner"])
    problem.add_boundary_condition(
        "Dirichlet",
        "right",
        variable="temperature",
        value=t_cool,
    )
    return problem


def given_conductance_problem(
    mesh, method, primary="primary", secondary="secondary", coordinates="cartesian"
):
    problem = conduction_problem(mesh, method, coordinates)
    problem.add_boundary_condition(
        "gap_heat_transfer",
        primary,
        variable="temperature",
        secondary_boundary=[secondary],
        gap_conductance=conductance,
    )
    problem.solve()
    return problem


def given_conductance_error(problem):
    secondary_temperature = t_cool + source * A * (C - B) / k_outer
    exact = slab_temperature(
        problem.node_coordinates()[:, 0],
        secondary_temperature + source * A / conductance,
        secondary_temperature,
    )
    return numpy.abs(problem.values("temperature") - exact).max()


def convergence_rates(errors):
    errors = numpy.asarray(errors)
    return numpy.log2(errors[:-1] / errors[1:])


# ---------------------------------------------------------------------------
# A gap with a given conductance
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("dimension", [1, 2, 3])
@pytest.mark.parametrize("method", vertex_methods)
def test_a_given_conductance_is_exact_on_matching_meshes_with_the_vertex_methods(method, dimension):
    """The exact temperature is quadratic in the heated slab, and the vertex
    methods reproduce it at the nodes in one dimension and, since it does not
    vary across the slabs, in two and three as well."""
    for num_x in (2, 4) if dimension == 3 else (4, 8):
        problem = given_conductance_problem(two_slabs(dimension, num_x, 2, 2), method)
        assert given_conductance_error(problem) < 1e-9


@pytest.mark.parametrize("dimension", [2, 3])
@pytest.mark.parametrize("method", methods)
def test_the_heat_that_crosses_the_gap_is_conserved_on_non_matching_meshes(method, dimension):
    """Two elements across the inner slab face three across the outer one:
    every watt generated leaves through the cooled face."""
    problem = given_conductance_problem(two_slabs(dimension, 4, 2, 3), method)
    power = source * A * cross_section(dimension)
    assert problem.boundary_flux("temperature", "right") == pytest.approx(power, rel=1e-12)


@pytest.mark.parametrize("method", methods)
def test_non_matching_meshes_converge_at_first_order(method):
    """The closest-point pairing of the two surfaces converges at first order
    when the meshes of the surfaces do not match."""
    errors = [
        given_conductance_error(
            given_conductance_problem(two_slabs(2, num_x, num_x // 2, 3 * num_x // 4), method)
        )
        for num_x in (4, 8, 16)
    ]
    assert convergence_rates(errors).min() > 0.9


@pytest.mark.parametrize("method", vertex_methods)
def test_the_primary_and_the_secondary_side_can_be_swapped(method):
    """The condition acts on both bodies, so the solution does not depend on
    which surface is the primary one."""
    mesh = two_slabs(2, 4, 3, 3)
    forward = given_conductance_problem(mesh, method)
    backward = given_conductance_problem(mesh, method, primary="secondary", secondary="primary")
    assert backward.values("temperature") == pytest.approx(forward.values("temperature"), rel=1e-12)


@pytest.mark.parametrize("method", methods)
def test_a_given_conductance_in_a_cylinder_converges_at_second_order(method):
    """A heated rod inside a tube (a radial line in axisymmetric coordinates):
    the exact temperature is r^2 in the rod and ln r in the tube."""

    def exact(r):
        linear_power = source * numpy.pi * A**2
        tube_inner = t_cool + linear_power / (2 * numpy.pi * k_outer) * numpy.log(C / B)
        rod_surface = tube_inner + linear_power / (2 * numpy.pi * A * conductance)
        return numpy.where(
            r <= A,
            rod_surface + source * (A**2 - r**2) / (4 * k_inner),
            t_cool + linear_power / (2 * numpy.pi * k_outer) * numpy.log(C / numpy.maximum(r, B)),
        )

    errors = []
    for num_x in (8, 16, 32):
        problem = given_conductance_problem(two_slabs(1, num_x), method, coordinates="axisymmetric")
        errors.append(
            numpy.abs(
                problem.values("temperature") - exact(numpy.abs(problem.node_coordinates()[:, 0]))
            ).max()
        )
        assert problem.boundary_flux("temperature", "right") == pytest.approx(
            source * numpy.pi * A**2, rel=1e-10
        )
    assert convergence_rates(errors).min() > 1.7


@pytest.mark.parametrize("thickness", [1.0, 0.1])
def test_the_contact_pressure_of_a_plane_problem_is_per_unit_area_at_any_thickness(thickness):
    """Two plane-strain slabs pushed together across the gap, with the equations multiplied by the
    thickness h.  The contact condition carries the same factor, so the compatibility relation

        push - (B - A) = P (A + C - B) / E'

    and the displacements do not depend on h (the mortar contact is exact)."""
    youngs_modulus, poissons_ratio, push = 2.0e11, 0.3, 7.0e-5
    problem = dualmesh.Problem(two_slabs(2, 4, 2, 2), method="fem")
    displacements = ["disp_x", "disp_y"]
    for name in displacements:
        problem.add_variable(name)
    problem.add_property(
        "small_strain_stress",
        displacements=displacements,
        formulation="plane_strain",
        youngs_modulus=youngs_modulus,
        poissons_ratio=poissons_ratio,
    )
    problem.add_kernel("stress_divergence", displacements=displacements, thickness=thickness)
    problem.add_variable("contact_pressure")
    problem.add_boundary_condition(
        "mortar_contact",
        "secondary",
        variable="contact_pressure",
        primary_boundary="primary",
        displacements=displacements,
        stiffness_scale=2.0e11 / 2.5e-3,
        thickness=thickness,
    )
    problem.add_boundary_condition("Dirichlet", "left", variable="disp_x", value=0.0)
    problem.add_boundary_condition("Dirichlet", "right", variable="disp_x", value=-push)
    problem.add_boundary_condition("Dirichlet", "bottom", variable="disp_y", value=0.0)
    problem.solve(report="none")
    plane_strain_modulus = youngs_modulus / (1 - poissons_ratio**2)
    pressure = (push - (B - A)) / ((A + C - B) / plane_strain_modulus)
    points = problem.node_coordinates()
    primary_nodes = numpy.abs(points[:, 0] - A) < 1e-12
    assert problem.values("disp_x")[primary_nodes] == pytest.approx(
        -pressure * A / plane_strain_modulus, rel=1e-9
    )


@pytest.mark.parametrize("closed", [False, True])
def test_the_Jacobian_of_the_mortar_contact_agrees_with_finite_differences(closed):
    """At a state that is not a solution, with displacements and a contact pressure that vary
    along the surfaces, and with the gap open or closed."""
    from test_Jacobian import compare, finite_difference_Jacobian

    problem = dualmesh.Problem(two_slabs(2, 4, 2, 2), method="fem")
    displacements = ["disp_x", "disp_y"]
    for name in displacements:
        problem.add_variable(name)
    problem.add_property(
        "small_strain_stress",
        displacements=displacements,
        formulation="plane_strain",
        youngs_modulus=2.0e11,
        poissons_ratio=0.3,
    )
    problem.add_kernel("stress_divergence", displacements=displacements)
    problem.add_variable("contact_pressure")
    problem.add_boundary_condition(
        "mortar_contact",
        "secondary",
        variable="contact_pressure",
        primary_boundary="primary",
        displacements=displacements,
        stiffness_scale=2.0e11 / 2.5e-3,
    )
    problem.initialize()
    points = problem.node_coordinates()
    x, y = points[:, 0], points[:, 1]
    # The outer slab moves towards the inner one by more than the gap width when closed.
    shift = (-7e-5 if closed else -2e-5) * (1.0 + 20.0 * y)
    problem.set_values("disp_x", numpy.where(x > 0.5 * (A + B), shift, 1e-6 * y))
    problem.set_values("disp_y", 1e-6 * x)
    # A contact pressure that varies along the surface, well inside the active set when closed.
    problem.set_values(
        "contact_pressure", numpy.where(x > 0.5 * (A + B), 3e7 * (1 + y / height), 0.0)
    )
    _, Jacobian = problem.linear_system()
    assert compare(Jacobian.toarray(), finite_difference_Jacobian(problem, step=1e-9)) < 1e-5
