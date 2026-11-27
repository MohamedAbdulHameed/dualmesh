# SPDX-License-Identifier: LGPL-2.1-or-later
"""Tools for the boundaries of a mesh and the boundary conditions on them:
side set measures, the side set report and file, one condition on several
variables, and the total_force of a traction."""

import math

import numpy
import pytest

dualmesh = pytest.importorskip("dualmesh")


def box(n=(6, 2, 2)):
    return dualmesh.generate_box_mesh(
        x_min=0.0,
        x_max=1.0,
        y_min=0.0,
        y_max=0.2,
        z_min=0.0,
        z_max=0.1,
        num_x_elements=n[0],
        num_y_elements=n[1],
        num_z_elements=n[2],
    )


def cantilever(conditions, mesh=None):
    """A 3D cantilever clamped at x = 0 and loaded at x = 1, with the
    boundary conditions added by ``conditions(problem)``."""
    problem = dualmesh.Problem(mesh or box(), method="fem")
    problem.add_physics(
        "solid_mechanics",
        formulation="three_dimensional",
        youngs_modulus=1.0e9,
        poissons_ratio=0.3,
    )
    conditions(problem)
    problem.solve()
    return problem


# ---- measures --------------------------------------------------------------


def test_sideset_measure_of_flat_and_curved_boundaries():
    assert box().sideset_measure(["right"]) == pytest.approx(0.02, rel=1e-12)
    assert box().sideset_measure(["right", "left"]) == pytest.approx(0.04, rel=1e-12)
    # A linear mesh measures the inscribed polygon exactly, a quadratic one
    # approximates the arc to within its interpolation error.
    linear = dualmesh.generate_annulus_mesh(0.5, 1.0, 2, 8, 0.0, 90.0, element_type="Quad4")
    assert linear.sideset_measure(["outer"]) == pytest.approx(
        16 * math.sin(math.pi / 32), rel=1e-12
    )
    quadratic = dualmesh.generate_annulus_mesh(0.5, 1.0, 2, 8, 0.0, 90.0, element_type="Quad9")
    assert quadratic.sideset_measure(["outer"]) == pytest.approx(math.pi / 2, rel=1e-5)


@pytest.mark.parametrize("element_type", ["Quad4", "Tri3", "Quad9"])
def test_an_annular_sector_names_its_sides_by_the_annulus(element_type):
    """A sector 0.5 < r < 1 from 30 to 120 degrees has the side sets inner, outer, start and end
    alone, and each holds the nodes of its edge: radius 0.5 or 1, or angle 30 or 120 degrees."""
    mesh = dualmesh.generate_annulus_mesh(
        0.5, 1.0, 3, 6, start_angle=30.0, end_angle=120.0, element_type=element_type
    )
    assert sorted(mesh.sideset_names()) == ["end", "inner", "outer", "start"]
    points = numpy.asarray(mesh.points())
    radius = numpy.hypot(points[:, 0], points[:, 1])
    angle = numpy.degrees(numpy.arctan2(points[:, 1], points[:, 0]))
    on_edge = {
        "inner": numpy.isclose(radius, 0.5, rtol=1e-12),
        "outer": numpy.isclose(radius, 1.0, rtol=1e-12),
        "start": numpy.isclose(angle, 30.0, rtol=1e-12),
        "end": numpy.isclose(angle, 120.0, rtol=1e-12),
    }
    for name, edge in on_edge.items():
        assert sorted(mesh.boundary_nodes(name)) == list(numpy.flatnonzero(edge)), name
    # The straight edges are 0.5 long, and the arcs of a quadratic mesh are close to r pi / 2.
    assert mesh.sideset_measure(["start"]) == pytest.approx(0.5, rel=1e-12)
    assert mesh.sideset_measure(["end"]) == pytest.approx(0.5, rel=1e-12)
    if element_type == "Quad9":
        assert mesh.sideset_measure(["outer"]) == pytest.approx(math.pi / 2, rel=1e-4)


radial = {"inner_radius": 0.5, "outer_radius": 1.0, "num_radial_elements": 2}


@pytest.mark.parametrize(
    "arguments, message",
    [
        ({**radial, "inner_radius": 0.0}, "positive"),
        ({**radial, "outer_radius": 0.4}, "increase"),
        ({"radial_coordinates": [0.5, 0.7, 1.0], "inner_radius": 0.5}, "not both"),
        ({"radial_coordinates": [0.5, 0.7, 1.0], "radial_bias": 1.2}, "radial_bias"),
        ({"inner_radius": 0.5, "outer_radius": 1.0}, "num_radial_elements"),
        ({**radial, "end_angle": 0.0}, r"\(0, 360\]"),
        ({**radial, "start_angle": -90.0, "end_angle": 300.0}, r"\(0, 360\]"),
    ],
)
def test_an_annular_sector_refuses_wrong_arguments(arguments, message):
    with pytest.raises(ValueError, match=message):
        dualmesh.generate_annulus_mesh(num_angular_elements=4, **arguments)


def test_an_annular_sector_needs_its_angular_elements():
    with pytest.raises(ValueError, match="num_angular_elements"):
        dualmesh.generate_annulus_mesh(0.5, 1.0, 2)
    with pytest.raises(ValueError, match="at least 1"):
        dualmesh.generate_annulus_mesh(0.5, 1.0, 2, 0)


def test_boundary_measure_includes_the_coordinate_factor():
    radius, length = 0.3, 2.0
    mesh = dualmesh.generate_rectangle_mesh(
        x_min=0.1, x_max=radius, y_min=0.0, y_max=length, num_x_elements=3, num_y_elements=5
    )
    problem = dualmesh.Problem(mesh, method="fem", coordinates="axisymmetric")
    problem.add_variable("disp_x")
    assert problem.boundary_area("right") == pytest.approx(2 * math.pi * radius * length, rel=1e-12)
    # The top annulus between r = 0.1 and 0.3: pi (0.3^2 - 0.1^2).
    assert problem.boundary_area("top") == pytest.approx(math.pi * (radius**2 - 0.1**2), rel=1e-12)


# ---- report and file ---------------------------------------------------------


def test_write_sidesets_writes_mixed_faces_wound_outward(tmp_path):
    # One prism: two triangles and three quadrilaterals, each written with its
    # right-hand normal pointing out of the prism.
    meshio = pytest.importorskip("meshio")
    prism = dualmesh.mesh_from_arrays(
        [[0, 0, 0], [0, 1, 0], [1, 0, 0], [0, 0, 1], [0, 1, 1], [1, 0, 1]],
        [[0, 1, 2, 3, 4, 5]],
        element_type="Wedge6",
    )
    prism.add_bounding_box_sidesets()
    path = tmp_path / "prism.vtu"
    dualmesh.write_sidesets(prism, str(path))
    surface = meshio.read(path)
    shapes = sorted((block.type, len(block.data)) for block in surface.cells)
    assert shapes == [("quad", 3), ("triangle", 2)]
    centre = surface.points.mean(axis=0)
    for block in surface.cells:
        for face in surface.points[block.data]:
            normal = numpy.cross(face[1] - face[0], face[2] - face[0])
            assert numpy.dot(normal, face.mean(axis=0) - centre) > 0
    assert numpy.concatenate(surface.cell_data["bottom"]).sum() == len(prism.sideset("bottom"))


# ---- one condition on several variables ---------------------------------------


def clamped_by_loop(problem):
    for v in ("disp_x", "disp_y", "disp_z"):
        problem.add_boundary_condition("Dirichlet", "left", variable=v, value=0.0)
    problem.add_boundary_condition("traction", "right", variable="disp_y", traction=-50.0 / 0.02)


# ---- total force ---------------------------------------------------------------


def test_total_force_gives_the_same_solution_as_the_traction():
    reference = cantilever(clamped_by_loop)

    def by_force(problem):
        problem.add_boundary_condition(
            "Dirichlet", "left", variable=["disp_x", "disp_y", "disp_z"], value=0.0
        )
        problem.add_boundary_condition("traction", "right", variable="disp_y", total_force=-50.0)

    problem = cantilever(by_force)
    assert numpy.max(
        numpy.abs(numpy.array(problem.values("disp_y")) - numpy.array(reference.values("disp_y")))
    ) < 1e-10 * numpy.max(numpy.abs(reference.values("disp_y")))
    assert -problem.boundary_flux("disp_y", "left") == pytest.approx(50.0, rel=1e-9)


def test_total_force_on_a_plane_strip_with_a_thickness():
    # In a plane problem the total force acts on the whole thickness: the
    # reaction equals it for any thickness, and the extension is inversely
    # proportional to the thickness.
    extension = {}
    for thickness in (1.0, 0.01):
        mesh = dualmesh.generate_rectangle_mesh(
            x_min=0.0, x_max=1.0, y_min=0.0, y_max=0.1, num_x_elements=10, num_y_elements=2
        )
        problem = dualmesh.Problem(mesh, method="fem")
        problem.add_physics(
            "solid_mechanics",
            formulation="plane_stress",
            youngs_modulus=1.0e9,
            poissons_ratio=0.3,
            thickness=thickness,
        )
        problem.add_boundary_condition(
            "Dirichlet", "left", variable=["disp_x", "disp_y"], value=0.0
        )
        problem.add_boundary_condition(
            "traction",
            "right",
            variable="disp_x",
            total_force=1000.0,
        )
        problem.solve()
        assert -problem.boundary_flux("disp_x", "left") == pytest.approx(-1000.0, rel=1e-9)
        extension[thickness] = max(problem.values("disp_x"))
    assert extension[0.01] == pytest.approx(100.0 * extension[1.0], rel=1e-9)


def test_total_force_in_axisymmetric_coordinates():
    # A hollow cylinder pulled along its axis by 1 kN on its top annulus.
    mesh = dualmesh.generate_rectangle_mesh(
        x_min=0.1, x_max=0.3, y_min=0.0, y_max=1.0, num_x_elements=4, num_y_elements=8
    )
    problem = dualmesh.Problem(mesh, method="fem", coordinates="axisymmetric")
    problem.add_physics(
        "solid_mechanics",
        formulation="axisymmetric",
        youngs_modulus=1.0e9,
        poissons_ratio=0.0,
    )
    problem.add_boundary_condition("Dirichlet", "bottom", variable="disp_z", value=0.0)
    problem.add_boundary_condition("traction", "top", variable="disp_z", total_force=1000.0)
    problem.solve()
    area = math.pi * (0.3**2 - 0.1**2)
    # With nu = 0 the axial stress is uniform, F / A, and the top moves by
    # F L / (E A).
    assert max(problem.values("disp_z")) == pytest.approx(1000.0 / (1.0e9 * area), rel=1e-9)


def test_traction_and_total_force_together_are_refused():
    def both(problem):
        problem.add_boundary_condition(
            "Dirichlet", "left", variable=["disp_x", "disp_y", "disp_z"], value=0.0
        )
        problem.add_boundary_condition(
            "traction", "right", variable="disp_y", traction=-1.0, total_force=-50.0
        )

    with pytest.raises(Exception, match="give only one of 'traction' and 'total_force'"):
        cantilever(both)


# ---- singular systems ------------------------------------------------------------


def two_squares():
    """Two unit squares that share no node, the left one with a side set."""
    points = [[0, 0], [1, 0], [1, 1], [0, 1], [2, 0], [3, 0], [3, 1], [2, 1]]
    mesh = dualmesh.mesh_from_arrays(points, [[0, 1, 2, 3], [4, 5, 6, 7]], element_type="Quad4")
    mesh.add_sideset_by_predicate("left", lambda x, y, z: abs(x) < 1e-9)
    mesh.add_sideset_by_predicate("far", lambda x, y, z: abs(x - 3.0) < 1e-9)
    return mesh


def test_a_part_without_any_condition_is_reported():
    problem = dualmesh.Problem(two_squares(), method="fem")
    problem.add_variable("T")
    problem.add_kernel("diffusion", variable="T")
    problem.add_boundary_condition("Dirichlet", "left", variable="T", value=1.0)
    with pytest.raises(
        Exception,
        match=r"singular: the variable 'T' on a part of the mesh with 4 nodes \(x from 2 to 3, y from 0 to 1\)",
    ):
        problem.solve()


@pytest.mark.parametrize("anchor", ["robin", "reaction", "dirichlet"])
def test_a_part_held_otherwise_is_accepted(anchor):
    """The singularity check accepts a part held by any of these, and the answer obeys the
    maximum principle: between the prescribed 1 and 0 (the Robin and the reaction terms pull
    towards 0)."""
    problem = dualmesh.Problem(two_squares(), method="fem")
    problem.add_variable("T")
    problem.add_kernel("diffusion", variable="T")
    problem.add_boundary_condition("Dirichlet", "left", variable="T", value=1.0)
    if anchor == "robin":
        problem.add_boundary_condition("Robin", "far", variable="T", transfer_coefficient=2.0)
    elif anchor == "reaction":
        problem.add_kernel("reaction", variable="T", coefficient=1.0e-3)
    else:
        problem.add_boundary_condition("Dirichlet", "far", variable="T", value=0.0)
    assert problem.solve(report="none").converged
    T = problem.values("T")
    assert T.max() <= 1.0 + 1e-12 and T.min() >= -1e-12
    assert T.min() < 1.0 - 1e-6


def test_a_transient_part_without_condition_is_accepted():
    """The capacity term makes the system of every time step nonsingular.  The answer heats from
    0 towards the prescribed 1 and stays between them."""
    problem = dualmesh.Problem(two_squares(), method="fem")
    problem.add_variable("T", initial_condition=0.0)
    problem.add_kernel("diffusion", variable="T")
    problem.add_kernel("time_derivative", variable="T")
    problem.add_boundary_condition("Dirichlet", "left", variable="T", value=1.0)
    assert problem.solve_transient(end_time=0.1, time_step=0.05, report="none").converged
    T = problem.values("T")
    assert T.max() <= 1.0 + 1e-12 and T.min() >= -1e-12
    assert 0.0 < T.mean() < 1.0
