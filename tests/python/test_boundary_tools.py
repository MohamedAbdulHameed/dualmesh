# SPDX-License-Identifier: LGPL-2.1-or-later
"""Tools for the boundaries of a mesh and the boundary conditions on them:
side set measures, the side set report and file, one condition on several
variables, fixed_constraint, and the total_force of a traction."""

import math
from pathlib import Path

import numpy as np
import pytest

dm = pytest.importorskip("dualmesh")


def box(n=(6, 2, 2)):
    return dm.generate_box_mesh(x_min=0.0, x_max=1.0, y_min=0.0, y_max=0.2, z_min=0.0, z_max=0.1, num_x_elements=n[0], num_y_elements=n[1], num_z_elements=n[2])


def cantilever(conditions, mesh=None):
    """A 3D cantilever clamped at x = 0 and loaded at x = 1, with the
    boundary conditions added by ``conditions(problem)``."""
    problem = dm.Problem(mesh or box(), method="fem")
    problem.add_physics("solid_mechanics", "solid", displacements=["u", "v", "w"], formulation="three_dimensional", youngs_modulus=1.0e9, poissons_ratio=0.3)
    conditions(problem)
    problem.solve()
    return problem


# ---- measures --------------------------------------------------------------


def test_sideset_measure_of_flat_and_curved_boundaries():
    assert box().sideset_measure(["right"]) == pytest.approx(0.02, rel=1e-12)
    assert box().sideset_measure(["right", "left"]) == pytest.approx(0.04, rel=1e-12)
    # A linear mesh measures the inscribed polygon exactly, a quadratic one
    # approximates the arc to within its interpolation error.
    linear = dm.generate_annulus_mesh(0.5, 1.0, 2, 8, 0.0, 90.0, element_type="Quad4")
    assert linear.sideset_measure(["outer"]) == pytest.approx(16 * math.sin(math.pi / 32), rel=1e-12)
    quadratic = dm.generate_annulus_mesh(0.5, 1.0, 2, 8, 0.0, 90.0, element_type="Quad9")
    assert quadratic.sideset_measure(["outer"]) == pytest.approx(math.pi / 2, rel=1e-5)


def test_boundary_measure_includes_the_coordinate_factor():
    radius, length = 0.3, 2.0
    mesh = dm.generate_rectangle_mesh(x_min=0.1, x_max=radius, y_min=0.0, y_max=length, num_x_elements=3, num_y_elements=5)
    problem = dm.Problem(mesh, method="fem", coordinates="axisymmetric")
    problem.add_variable("u")
    assert problem.boundary_measure("right") == pytest.approx(2 * math.pi * radius * length, rel=1e-12)
    # The top annulus between r = 0.1 and 0.3: pi (0.3^2 - 0.1^2).
    assert problem.boundary_measure("top") == pytest.approx(math.pi * (radius**2 - 0.1**2), rel=1e-12)


# ---- report and file ---------------------------------------------------------


def test_sideset_summary_lists_every_side_set():
    text = dm.sideset_summary(box())
    for name in ("left", "right", "bottom", "top", "front", "back"):
        assert name in text
    row = next(line for line in text.splitlines() if line.startswith("right"))
    faces, nodes, area = row.split()[1:4]
    assert (int(faces), int(nodes), float(area)) == (4, 9, pytest.approx(0.02))


def test_write_sidesets_marks_every_face_of_every_side_set(tmp_path):
    meshio = pytest.importorskip("meshio")
    mesh = box()
    path = tmp_path / "sides.vtu"
    dm.write_sidesets(mesh, str(path))
    surface = meshio.read(path)
    exterior = len(mesh.exterior_sides())
    assert sum(len(block.data) for block in surface.cells) == exterior
    names = list(mesh.sideset_names())
    for name in names:
        assert surface.cell_data[name][0].sum() == len(mesh.sideset(name))
    # Each face of this box belongs to one side set, so side_set is never 0
    # and counts every side set once.
    side_set = surface.cell_data["side_set"][0]
    assert np.all(side_set > 0)
    assert [int(np.sum(side_set == k + 1)) for k in range(len(names))] == [len(mesh.sideset(n)) for n in names]


def test_cli_mesh_lists_the_side_sets_and_writes_them(tmp_path, capsys):
    pytest.importorskip("meshio")
    from dualmesh.cli import main

    source = Path(__file__).resolve().parents[2] / "examples" / "wrench" / "wrench.msh"
    out = tmp_path / "wrench_sides.vtu"
    assert main(["mesh", str(source), "--sidesets", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "jaws" in printed and "grip" in printed
    assert out.exists()


# ---- one condition on several variables ---------------------------------------


def clamped_by_loop(problem):
    for v in ("u", "v", "w"):
        problem.add_boundary_condition("Dirichlet_boundary_condition", f"left_{v}", variable=v, boundary="left", value=0.0)
    problem.add_boundary_condition("traction_boundary_condition", "right", variable="v", traction=-50.0 / 0.02)


def test_variables_list_and_fixed_constraint_equal_one_condition_per_variable():
    reference = np.array(cantilever(clamped_by_loop).values("v"))

    def with_variables(problem):
        problem.add_boundary_condition("Dirichlet_boundary_condition", "left", variables=["u", "v", "w"], value=0.0)
        problem.add_boundary_condition("traction_boundary_condition", "right", variable="v", traction=-50.0 / 0.02)

    def with_fixed_constraint(problem):
        problem.add_boundary_condition("fixed_constraint", "left", displacements=["u", "v", "w"])
        problem.add_boundary_condition("traction_boundary_condition", "right", variable="v", traction=-50.0 / 0.02)

    for conditions in (with_variables, with_fixed_constraint):
        values = np.array(cantilever(conditions).values("v"))
        assert np.max(np.abs(values - reference)) < 1e-12 * np.max(np.abs(reference))


def test_conditions_that_define_the_same_thing_are_refused():
    problem = dm.Problem(box((2, 1, 1)), method="fem")
    for v in ("u", "v", "w"):
        problem.add_variable(v)
    with pytest.raises(Exception, match="'variable' and 'variables' define the same thing"):
        problem.add_boundary_condition("Dirichlet_boundary_condition", "left", variable="u", variables=["u", "v"], value=0.0)
    with pytest.raises(Exception, match="'variables' and 'displacements' define the same thing"):
        problem.add_boundary_condition("fixed_constraint", "right", displacements=["u"], variables=["v"])
    with pytest.raises(Exception, match="'variable' and 'displacements' define the same thing"):
        problem.add_boundary_condition("fixed_constraint", "top", displacements=["u"], variable="v")


# ---- total force ---------------------------------------------------------------


def test_total_force_gives_the_same_solution_as_the_traction():
    reference = cantilever(clamped_by_loop)

    def by_force(problem):
        problem.add_boundary_condition("fixed_constraint", "left", displacements=["u", "v", "w"])
        problem.add_boundary_condition("traction_boundary_condition", "right", variable="v", total_force=-50.0)

    problem = cantilever(by_force)
    assert np.max(np.abs(np.array(problem.values("v")) - np.array(reference.values("v")))) < 1e-10 * np.max(np.abs(reference.values("v")))
    assert problem.total_reaction("v", "left") == pytest.approx(50.0, rel=1e-9)


def test_total_force_on_a_plane_strip_with_a_thickness():
    # In a plane problem the total force acts on the whole thickness: the
    # reaction equals it for any thickness, and the extension is inversely
    # proportional to the thickness.
    extension = {}
    for thickness in (1.0, 0.01):
        mesh = dm.generate_rectangle_mesh(x_min=0.0, x_max=1.0, y_min=0.0, y_max=0.1, num_x_elements=10, num_y_elements=2)
        problem = dm.Problem(mesh, method="fem")
        problem.add_physics("solid_mechanics", "solid", displacements=["u", "v"], formulation="plane_stress", youngs_modulus=1.0e9, poissons_ratio=0.3, thickness=thickness)
        problem.add_boundary_condition("fixed_constraint", "left", displacements=["u", "v"])
        problem.add_boundary_condition("traction_boundary_condition", "right", variable="u", total_force=1000.0, thickness=thickness)
        problem.solve()
        assert problem.total_reaction("u", "left") == pytest.approx(-1000.0, rel=1e-9)
        extension[thickness] = max(problem.values("u"))
    assert extension[0.01] == pytest.approx(100.0 * extension[1.0], rel=1e-9)


def test_total_force_in_axisymmetric_coordinates():
    # A hollow cylinder pulled along its axis by 1 kN on its top annulus.
    mesh = dm.generate_rectangle_mesh(x_min=0.1, x_max=0.3, y_min=0.0, y_max=1.0, num_x_elements=4, num_y_elements=8)
    problem = dm.Problem(mesh, method="fem", coordinates="axisymmetric")
    problem.add_physics("solid_mechanics", "solid", displacements=["u", "v"], formulation="axisymmetric", youngs_modulus=1.0e9, poissons_ratio=0.0)
    problem.add_boundary_condition("Dirichlet_boundary_condition", "bottom", variable="v", value=0.0)
    problem.add_boundary_condition("traction_boundary_condition", "top", variable="v", total_force=1000.0)
    problem.solve()
    area = math.pi * (0.3**2 - 0.1**2)
    # With nu = 0 the axial stress is uniform, F / A, and the top moves by
    # F L / (E A).
    assert max(problem.values("v")) == pytest.approx(1000.0 / (1.0e9 * area), rel=1e-9)


def test_traction_and_total_force_together_are_refused():
    def both(problem):
        problem.add_boundary_condition("fixed_constraint", "left", displacements=["u", "v", "w"])
        problem.add_boundary_condition("traction_boundary_condition", "right", variable="v", traction=-1.0, total_force=-50.0)

    with pytest.raises(Exception, match="give only one of 'traction' and 'total_force'"):
        cantilever(both)


# ---- singular systems ------------------------------------------------------------


def two_squares():
    """Two unit squares that share no node, the left one with a side set."""
    points = [[0, 0], [1, 0], [1, 1], [0, 1], [2, 0], [3, 0], [3, 1], [2, 1]]
    mesh = dm.mesh_from_arrays(points, [[0, 1, 2, 3], [4, 5, 6, 7]], element_type="Quad4")
    mesh.add_sideset_by_predicate("left", lambda x, y, z: abs(x) < 1e-9)
    mesh.add_sideset_by_predicate("far", lambda x, y, z: abs(x - 3.0) < 1e-9)
    return mesh


def test_a_part_without_any_condition_is_reported():
    problem = dm.Problem(two_squares(), method="fem")
    problem.add_variable("T")
    problem.add_kernel("diffusion", variable="T")
    problem.add_boundary_condition("Dirichlet_boundary_condition", "left", variable="T", value=1.0)
    with pytest.raises(Exception, match=r"singular: the variable 'T' on a part of the mesh with 4 nodes \(x from 2 to 3, y from 0 to 1\)"):
        problem.solve()


def test_a_floating_elastic_body_is_reported():
    mesh = two_squares()
    problem = dm.Problem(mesh, method="fem")
    problem.add_physics("solid_mechanics", "solid", displacements=["u", "v"], formulation="plane_stress", youngs_modulus=1.0e9, poissons_ratio=0.3)
    problem.add_boundary_condition("fixed_constraint", "left", displacements=["u", "v"])
    with pytest.raises(Exception, match="singular"):
        problem.solve()


@pytest.mark.parametrize("anchor", ["robin", "reaction", "dirichlet"])
def test_a_part_held_otherwise_is_accepted(anchor):
    problem = dm.Problem(two_squares(), method="fem")
    problem.add_variable("T")
    problem.add_kernel("diffusion", variable="T")
    problem.add_boundary_condition("Dirichlet_boundary_condition", "left", variable="T", value=1.0)
    if anchor == "robin":
        problem.add_boundary_condition("Robin_boundary_condition", "far", variable="T", transfer_coefficient=2.0)
    elif anchor == "reaction":
        problem.add_kernel("reaction", variable="T", coefficient=1.0e-3)
    else:
        problem.add_boundary_condition("Dirichlet_boundary_condition", "far", variable="T", value=0.0)
    assert problem.solve().converged


def test_a_transient_part_without_condition_is_accepted():
    # The capacity term makes the system of every time step nonsingular.
    problem = dm.Problem(two_squares(), method="fem")
    problem.add_variable("T", initial_condition=0.0)
    problem.add_kernel("diffusion", variable="T")
    problem.add_kernel("time_derivative", variable="T")
    problem.add_boundary_condition("Dirichlet_boundary_condition", "left", variable="T", value=1.0)
    problem.solve_transient(end_time=0.1, time_step=0.05)
