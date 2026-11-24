# SPDX-License-Identifier: LGPL-2.1-or-later
"""Tests of the error indicator, of the marking rules and of local refinement.

The benchmark at the end of the file is the classical one for adaptivity: the
Laplace equation on an L-shaped domain, whose solution

    u = r^(2/3) sin(2 theta / 3)

is harmonic everywhere but has an unbounded gradient at the re-entrant corner
at the origin.  Uniform refinement wastes most of its elements far from the
corner and converges slowly; an adaptive loop puts the elements where the
error is and recovers the rate a smooth solution would give.
"""

from __future__ import annotations

import dualmesh
import numpy
import pytest


def unit_square(num_elements=4):
    return dualmesh.generate_rectangle_mesh(
        x_min=0.0,
        x_max=1.0,
        y_min=0.0,
        y_max=1.0,
        num_x_elements=num_elements,
        num_y_elements=num_elements,
        element_type="Tri3",
    )


def signed_areas(mesh):
    points = numpy.asarray(mesh.points())
    out = []
    for _type, connectivity, _block in mesh.cells():
        x = points[list(connectivity)]
        out.append(
            0.5
            * (
                (x[1, 0] - x[0, 0]) * (x[2, 1] - x[0, 1])
                - (x[2, 0] - x[0, 0]) * (x[1, 1] - x[0, 1])
            )
        )
    return numpy.asarray(out)


def l_shaped_mesh(num_elements=2):
    """The unit square [-1, 1]^2 with the quadrant x > 0, y < 0 removed."""
    full = dualmesh.generate_rectangle_mesh(
        x_min=-1.0,
        x_max=1.0,
        y_min=-1.0,
        y_max=1.0,
        num_x_elements=2 * num_elements,
        num_y_elements=2 * num_elements,
        element_type="Tri3",
    )
    points = numpy.asarray(full.points())
    kept = []
    for _type, connectivity, _block in full.cells():
        centroid = points[list(connectivity)].mean(axis=0)
        if not (centroid[0] > 0.0 and centroid[1] < 0.0):
            kept.append(list(connectivity))
    used = sorted({node for cell in kept for node in cell})
    renumber = {old: new for new, old in enumerate(used)}
    mesh = dualmesh.mesh_from_arrays(
        points[used], [[renumber[node] for node in cell] for cell in kept], "Tri3"
    )
    mesh.add_sideset_by_predicate("boundary", lambda x, y, z: True)
    return mesh


def corner_solution(x, y):
    radius = numpy.hypot(x, y)
    angle = numpy.arctan2(y, x)
    angle = numpy.where(angle < -1e-12, angle + 2.0 * numpy.pi, angle)
    return radius ** (2.0 / 3.0) * numpy.sin(2.0 / 3.0 * angle)


def corner_problem(mesh):
    problem = dualmesh.Problem(mesh)
    problem.add_variable("u")
    problem.add_kernel("diffusion", variable="u")
    problem.add_boundary_condition(
        "Dirichlet",
        "boundary",
        variable="u",
        value=lambda x, y, z, t: float(corner_solution(numpy.array(x), numpy.array(y))),
    )
    return problem


def corner_error(problem):
    points = problem.node_coordinates()
    return float(numpy.abs(problem.values("u") - corner_solution(points[:, 0], points[:, 1])).max())


def solve_corner_problem(mesh):
    problem = corner_problem(mesh)
    problem.solve()
    return problem, corner_error(problem)


def test_repeated_refinement_keeps_every_triangle_properly_oriented():
    """Bisection through the middle of the refinement edge can never produce
    a degenerate or inverted triangle: a child keeps the orientation of its
    parent, and newest-vertex bisection gives finitely many shapes.  Both are
    checked here over many rounds of random marking, which is the way an
    adaptive run actually exercises the algorithm."""
    mesh = unit_square(2)
    generator = numpy.random.default_rng(0)
    smallest_ratio = 1.0
    for _ in range(12):
        areas = signed_areas(mesh)
        assert areas.min() > 0.0
        smallest_ratio = min(smallest_ratio, areas.min() / areas.max())
        marked = generator.random(mesh.num_elements) < 0.4
        marked[0] = True
        mesh, _ = dualmesh.refine_marked(mesh, marked)
    # Areas shrink by a factor of two per bisection, so the spread between the
    # largest and the smallest element grows; what must not happen is that it
    # collapses to zero.
    assert smallest_ratio > 1e-6
    assert mesh.num_elements > 1000


def test_refinement_leaves_no_hanging_node():
    """A hanging node would break the dual mesh, so the test is the property
    the dual mesh needs: with a conforming mesh, the control domains tile the
    domain and a linear field is reproduced exactly.  The patch test is run on
    a mesh refined several times in one corner, which is where a hanging node
    would appear if the propagation were incomplete."""
    mesh = unit_square(4)
    for _ in range(4):
        points = numpy.asarray(mesh.points())
        centroids = numpy.array([points[list(c)].mean(axis=0) for _t, c, _b in mesh.cells()])
        marked = (centroids[:, 0] < 0.25) & (centroids[:, 1] < 0.25)
        mesh, _ = dualmesh.refine_marked(mesh, marked)

    def exact(x, y, z, t):
        return 1.0 + 2.0 * x - 3.0 * y

    for method in ("dmcdm", "fem"):
        problem = dualmesh.Problem(mesh, method=method)
        problem.add_variable("u")
        problem.add_kernel("diffusion", variable="u")
        problem.add_boundary_condition(
            "Dirichlet",
            mesh.sideset_names(),
            variable="u",
            value=exact,
        )
        problem.solve()
        points = problem.node_coordinates()
        expected = 1.0 + 2.0 * points[:, 0] - 3.0 * points[:, 1]
        assert problem.values("u") == pytest.approx(expected, abs=1e-11)


def test_a_marked_quadrilateral_splits_into_four_and_leaves_nodes_hanging():
    """The cell at the origin of 2 by 2 quadrilaterals splits into four children, which come from
    it; its two neighbours across a side stay whole, with a node hanging in the middle of the
    shared side."""
    mesh = dualmesh.generate_rectangle_mesh(
        x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0, num_x_elements=2, num_y_elements=2
    )
    centres = numpy.asarray(
        [numpy.mean([mesh.points()[n] for n in cell[1]], axis=0) for cell in mesh.cells()]
    )
    corner = int(numpy.argmin(centres[:, 0] + centres[:, 1]))
    refined, parents = dualmesh.refine_marked(mesh, numpy.arange(4) == corner)
    assert refined.num_elements == 7
    assert sorted(parents.tolist()).count(corner) == 4
    # Four edge middles, two of them hanging, and the centre.
    assert refined.num_nodes == 9 + 5


def test_refinement_of_a_quadratic_quadrilateral_mesh_is_refused_with_a_reason():
    mesh = dualmesh.generate_rectangle_mesh(
        x_min=0.0,
        x_max=1.0,
        y_min=0.0,
        y_max=1.0,
        num_x_elements=2,
        num_y_elements=2,
        element_type="Quad9",
    )
    with pytest.raises(Exception, match="Quad4 or Hex8"):
        dualmesh.refine_marked(mesh, numpy.ones(mesh.num_elements, dtype=bool))


def test_a_wrong_sized_marker_is_refused():
    mesh = unit_square(2)
    with pytest.raises(Exception, match="one entry per element"):
        dualmesh.refine_marked(mesh, numpy.ones(mesh.num_elements + 1, dtype=bool))


# ---------------------------------------------------------------------------
# The error indicator
# ---------------------------------------------------------------------------
def test_the_indicator_vanishes_for_a_field_the_elements_represent_exactly():
    """The indicator compares the computed gradient with a recovered one.  For
    a linear solution the computed gradient is already constant and exact, the
    recovery reproduces it, and the indicator is zero to round-off.  An
    estimator that did not do this would refine a mesh that is already
    perfect."""
    mesh = unit_square(4)
    problem = dualmesh.Problem(mesh)
    problem.add_variable("u")
    problem.add_kernel("diffusion", variable="u")
    problem.add_boundary_condition(
        "Dirichlet",
        mesh.sideset_names(),
        variable="u",
        value=lambda x, y, z, t: 1.0 + 2.0 * x - 3.0 * y,
    )
    problem.solve()
    assert numpy.abs(problem.error_indicator("u")).max() < 1e-12


@pytest.mark.parametrize("marker", [dualmesh.mark_by_fraction, dualmesh.mark_by_error_fraction])
def test_a_marker_never_returns_an_empty_selection_for_a_positive_error(marker):
    assert marker(numpy.array([1.0, 1.0, 1.0, 1.0]), 0.1).any()


# ---------------------------------------------------------------------------
# The adaptive loop
# ---------------------------------------------------------------------------
def test_adaptive_refinement_beats_uniform_refinement_at_a_re_entrant_corner():
    """The rate at which the error falls with the number of unknowns is the
    thing to measure, because adaptive and uniform meshes have different sizes
    at every cycle.

    With linear elements on a smooth problem the nodal error falls like
    N^-1, since N grows like h^-2 and the error like h^2.  At a re-entrant
    corner of interior angle 3 pi / 2 the solution behaves like r^(2/3), and
    uniform refinement is held back to about N^(-1/3).  The adaptive loop
    should recover something close to N^-1.
    """

    def slope(history):
        sizes = numpy.array([n for n, _ in history], dtype=float)
        errors = numpy.array([e for _, e in history], dtype=float)
        return float(numpy.polyfit(numpy.log(sizes), numpy.log(errors), 1)[0])

    start = l_shaped_mesh(2)

    uniform_history = []
    mesh = start
    for cycle in range(5):
        _problem, error = solve_corner_problem(mesh)
        uniform_history.append((mesh.num_nodes, error))
        if cycle < 4:
            mesh = mesh.refined()

    # The loop keeps one hierarchy of newest-vertex bisections over its cycles.
    adaptive_history = []

    def record(_cycle, problem, _indicators):
        adaptive_history.append((problem.mesh.num_nodes, corner_error(problem)))

    dualmesh.solve_with_adaptive_refinement(
        corner_problem,
        start,
        "u",
        num_cycles=12,
        marker=lambda indicators: dualmesh.mark_by_error_fraction(indicators, 0.4),
        callback=record,
        report="none",
    )

    uniform_slope = slope(uniform_history)
    adaptive_slope = slope(adaptive_history[-7:])
    assert -0.45 < uniform_slope < -0.2  # Held back by the singularity
    assert adaptive_slope < -0.8  # Close to the optimal rate
    # At a comparable number of unknowns the adaptive mesh is the better one.
    assert adaptive_history[-1][0] < uniform_history[-1][0]
    assert adaptive_history[-1][1] < uniform_history[-1][1]


# ---------------------------------------------------------------------------
# The adaptive loop of the input
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("cell_type", ["triangle", "quadrilateral"])
def test_the_input_adapts_the_mesh_to_a_corner_where_the_boundary_temperature_jumps(cell_type):
    """A square held at 400 K on its left side and at 300 K on its bottom, and insulated
    elsewhere: the temperature jumps at the corner (0, 0), where its gradient is unbounded.  The
    loop of mesh_adaptivity refines there, by bisection of triangles or by subdivision of
    quadrilaterals with hanging nodes, and the temperature stays between the two boundary values,
    as the maximum principle of conduction requires."""
    plate = dualmesh.problem(
        mesh={"rectangle": {"size": ["1 m", "1 m"], "cells": [8, 8], "cell_type": cell_type}}
    )
    plate.physics("heat_conduction")
    plate.properties(thermal_conductivity="45 W/(m*K)")
    plate.boundary_conditions("left", temperature="400 K")
    plate.boundary_conditions("bottom", temperature="300 K")
    plate.mesh_adaptivity(
        indicator="gradient_recovery",
        field="temperature",
        refine_fraction=0.5,
        coarsen_fraction=0.05,
        interval=1,
        maximum_refinement_level=3,
    )
    solution = plate.solve(report="none")
    points = solution.coordinates("temperature")
    near_corner = int((numpy.hypot(points[:, 0], points[:, 1]) < 0.2).sum())
    # The initial mesh has 4 nodes within 0.2 m of the corner.
    assert near_corner > 12
    temperature = solution.field("temperature")
    assert temperature.min() >= 300.0 - 1e-6
    assert temperature.max() <= 400.0 + 1e-6


def test_a_transient_run_adapts_the_mesh_to_the_front_that_enters_from_a_heated_side():
    """A steel plate at 300 K whose left side is held at 400 K: heat enters from the left in a
    front about sqrt(alpha t) = 0.15 m deep after 2000 s.  The mesh adapts every two steps, the
    run goes on from each new mesh to the end time, the history of the post-processors runs on
    across the meshes, and the temperature stays between 300 K and 400 K."""
    plate = dualmesh.problem(
        mesh={"rectangle": {"size": ["1 m", "1 m"], "cells": [8, 8], "cell_type": "triangle"}}
    )
    plate.physics("heat_conduction")
    plate.properties(
        thermal_conductivity="45 W/(m*K)",
        density="7850 kg/m^3",
        specific_heat_capacity="490 J/(kg*K)",
    )
    plate.initial_conditions(temperature="300 K")
    plate.boundary_conditions("left", temperature="400 K")
    plate.time(end_time="2000 s", initial_time_step="100 s", step_control={"by": "fixed"})
    plate.postprocessor("peak", maximum="temperature")
    plate.mesh_adaptivity(
        indicator="gradient_recovery",
        field="temperature",
        refine_fraction=0.5,
        coarsen_fraction=0.05,
        interval=2,
        maximum_refinement_level=3,
    )
    solution = plate.solve(report="none")
    assert solution.times[-1] == pytest.approx(2000.0)
    times = solution.output_times
    assert all(later > earlier for earlier, later in zip(times, times[1:]))
    points = solution.coordinates("temperature")
    # The initial mesh has 18 nodes within 0.2 m of the heated side.
    assert int((points[:, 0] < 0.2).sum()) > 18
    temperature = solution.field("temperature")
    assert temperature.min() >= 300.0 - 1e-6
    assert temperature.max() <= 400.0 + 1e-6


@pytest.mark.parametrize("cell_type", ["triangle", "quadrilateral"])
def test_a_plate_pulled_by_a_traction_keeps_its_momentum_through_the_adapted_meshes(cell_type):
    """A steel plate L = 1 m long, on rollers on its bottom and free otherwise, pulled on its right
    side by sigma = 1 MPa from t = 0.  Its x-momentum grows as sigma H t whatever the waves inside,
    and the generalized-alpha method keeps that balance exactly, so the mean displacement is
    sigma t^2 / (2 rho L).  The mesh adapts to the waves every two steps by refinement alone,
    which carries the displacement and the velocity as the same functions, so the mean stays
    exact.  A run that went on from rest on each new mesh would lose the momentum."""
    sigma, rho, end_time = 1e6, 7850.0, 2e-4
    plate = dualmesh.problem(
        mesh={"rectangle": {"size": ["1 m", "0.25 m"], "cells": [8, 2], "cell_type": cell_type}}
    )
    plate.physics("solid_mechanics", formulation="plane_strain")
    plate.properties(Young_modulus="200 GPa", Poisson_ratio=0.3, density=f"{rho} kg/m^3")
    plate.boundary_conditions("bottom", displacement_y="0 m")
    plate.boundary_conditions("right", traction=[f"{sigma} Pa", "0 Pa"])
    plate.time(end_time=f"{end_time} s", initial_time_step="2e-5 s", step_control={"by": "fixed"})
    plate.postprocessor("mean", average="displacement_x")
    plate.mesh_adaptivity(
        indicator="gradient_recovery",
        field="displacement",
        refine_fraction=0.5,
        coarsen_fraction=0.0,
        interval=2,
        maximum_refinement_level=2,
    )
    solution = plate.solve(report="none")
    assert solution.times[-1] == pytest.approx(end_time)
    assert len(solution.coordinates("displacement")) > 9 * 3
    exact = sigma * end_time**2 / (2 * rho * 1.0)
    assert solution.postprocessor("mean")[-1] == pytest.approx(exact, rel=1e-6)


@pytest.mark.parametrize("element_type", ["Tri3", "Quad4"])
def test_a_body_falling_under_its_weight_keeps_its_velocity_across_a_change_of_mesh(element_type):
    """A steel square on rollers on its left side, pulled down by its weight rho g and free
    otherwise: every point falls as u_y = -g t^2 / 2 with the velocity -g t, on any mesh, and the
    generalized-alpha method integrates the constant acceleration exactly.  The run stops at
    t_1 = 0.01 s, every cell is refined, the displacement and the velocity reached are carried to the
    new mesh, and the run goes on to t_2 = 0.02 s.  A run that went on from rest would fall
    g t_1 (t_2 - t_1), half the exact fall, short."""
    from dualmesh.adaptivity import _Hierarchy

    g, rho, t_1, t_2, dt = 9.81, 7850.0, 0.01, 0.02, 0.002

    def falling(mesh):
        problem = dualmesh.Problem(mesh)
        solid = problem.add_physics(
            "solid_mechanics",
            formulation="plane_strain",
            youngs_modulus=200e9,
            poissons_ratio=0.3,
            density=rho,
            body_force=[0.0, -rho * g],
        )
        solid.add_boundary_condition("disp_x = 0", boundary="left")
        return problem

    hierarchy = _Hierarchy(
        dualmesh.generate_rectangle_mesh(0.0, 1.0, 0.0, 1.0, 4, 4, element_type=element_type)
    )
    first = falling(hierarchy.mesh())
    first.solve_transient(end_time=t_1, time_step=dt, report="none")
    reached = {name: first.gathered_values(name) for name in ("disp_x", "disp_y")}
    velocity = first.gathered_velocities()
    assert set(velocity) == {"disp_x", "disp_y"}
    assert velocity["disp_y"] == pytest.approx(-g * t_1, rel=1e-6)
    cells = hierarchy.mesh().num_elements
    assert hierarchy.adapt(numpy.ones(cells, bool), numpy.zeros(cells, bool), 1)

    second = falling(hierarchy.mesh())
    for name, values in reached.items():
        second.set_gathered_values(name, hierarchy.transfer(values))
    second.set_gathered_velocities({name: hierarchy.transfer(v) for name, v in velocity.items()})
    second.solve_transient(end_time=t_2, start_time=t_1, time_step=dt, report="none")
    assert hierarchy.mesh().num_elements > cells
    assert second.gathered_values("disp_y") == pytest.approx(-0.5 * g * t_2**2, rel=1e-6)
    assert second.gathered_velocities()["disp_y"] == pytest.approx(-g * t_2, rel=1e-6)
    assert numpy.abs(second.gathered_values("disp_x")).max() < 1e-9 * 0.5 * g * t_2**2


# ---------------------------------------------------------------------------
# Quadrilaterals with hanging nodes
# ---------------------------------------------------------------------------
def hanging_cells(dimension):
    """The unit square of 4 by 4 quadrilaterals, or the unit cube of 3 by 3 by 3 hexahedra, whose
    cell at the origin is split twice, which leaves nodes hanging on the sides of its neighbours."""
    from dualmesh import _core

    if dimension == 2:
        mesh = dualmesh.generate_rectangle_mesh(0.0, 1.0, 0.0, 1.0, 4, 4, element_type="Quad4")
    else:
        mesh = dualmesh.generate_box_mesh(0, 1, 0, 1, 0, 1, 3, 3, 3, element_type="Hex8")
    hierarchy = _core.Subdivision(mesh)
    for _ in range(2):
        points = numpy.asarray(hierarchy.mesh().points())
        marked = [
            int(numpy.allclose(points[list(cell)][0], 0.0))
            for _type, cell, _block in hierarchy.mesh().cells()
        ]
        hierarchy.refine(marked, 5)
    assert hierarchy.hanging()
    return hierarchy.mesh()


@pytest.mark.parametrize("dimension", [2, 3])
@pytest.mark.parametrize("method", ["fem", "dmcdm"])
def test_hanging_nodes_pass_the_patch_test(method, dimension):
    """FEM gives a hanging node the value of the side it lies on, so its space holds the linear
    fields.  DMCDM keeps a hanging node as an unknown and splits the large cell as a polygon whose
    control domains meet those of the small cells at the middles of the leaf edges, and the
    gradient of a linear field is exact in both.  The Laplace equation with linear boundary values
    gives the linear field exactly, at the hanging nodes too."""
    mesh = hanging_cells(dimension)
    problem = dualmesh.Problem(mesh, method=method)
    problem.add_variable("u")
    problem.add_kernel("diffusion", variable="u")
    problem.add_boundary_condition(
        "Dirichlet",
        mesh.sideset_names(),
        variable="u",
        value=lambda x, y, z, t: 1.0 + 2.0 * x - 3.0 * y + z,
    )
    problem.solve(report="none")
    points = problem.node_coordinates()
    expected = 1.0 + 2.0 * points[:, 0] - 3.0 * points[:, 1] + points[:, 2]
    assert problem.values("u") == pytest.approx(expected, abs=1e-11)


@pytest.mark.parametrize("dimension", [2, 3])
@pytest.mark.parametrize("method", ["fem", "dmcdm"])
def test_the_source_leaves_through_the_boundary_with_hanging_nodes(method, dimension):
    """-laplacian(u) = 1 on the unit square (cube) with u = 0 on its sides: the flow out is the
    source in it, 1, to round-off, with the control domains of DMCDM tiling the domain across the
    hanging nodes, and the constrained basis of FEM summing to one."""
    mesh = hanging_cells(dimension)
    problem = dualmesh.Problem(mesh, method=method)
    problem.add_variable("u")
    problem.add_kernel("diffusion", variable="u")
    problem.add_kernel("body_force", variable="u", value=1.0)
    problem.add_boundary_condition("Dirichlet", mesh.sideset_names(), variable="u", value=0.0)
    problem.solve(report="none")
    flows = {}
    for boundary in mesh.sideset_names():
        flows.update(problem.boundary_flux("u", boundary, per_node=True))
    assert sum(flows.values()) == pytest.approx(1.0, rel=1e-10)


@pytest.mark.parametrize("element_type", ["Tri3", "Quad4"])
def test_refinement_moves_the_new_nodes_of_a_curved_side_onto_its_curve(element_type):
    """A quarter annulus of triangles or quadrilaterals, refined everywhere three times: with the
    projection of the outer side onto the unit circle, every node of that side ends on the
    circle, where refinement alone would leave the new ones on the chords."""
    mesh = dualmesh.generate_annulus_mesh(0.5, 1.0, 2, 8, 0.0, 90.0, element_type=element_type)

    def build(mesh):
        problem = dualmesh.Problem(mesh)
        problem.add_variable("u")
        problem.add_kernel("diffusion", variable="u")
        problem.add_boundary_condition("Dirichlet", "outer", variable="u", value=0.0)
        problem.add_boundary_condition("Dirichlet", "inner", variable="u", value=1.0)
        return problem

    def onto_circle(point):
        radius = numpy.hypot(point[0], point[1])
        return (point[0] / radius, point[1] / radius, 0.0)

    result = dualmesh.solve_with_adaptive_refinement(
        build,
        mesh,
        "u",
        num_cycles=4,
        marker=lambda indicators: numpy.ones(len(indicators), dtype=bool),
        report="none",
        boundaries={"outer": onto_circle},
    )
    points = numpy.asarray(result.mesh.points())
    outer = list(result.mesh.boundary_nodes("outer"))
    assert len(outer) > len(list(mesh.boundary_nodes("outer")))
    radii = numpy.hypot(points[outer, 0], points[outer, 1])
    assert radii == pytest.approx(1.0, abs=1e-12)


def test_a_hanging_node_on_a_curved_side_stays_on_the_straight_side_of_its_neighbour():
    """The unit cube of 2 by 2 by 2 hexahedra with its top side bent to y = 1 + x (1 - x) / 10,
    and one cell under the top split.  The middles of the top edges that it shares with whole
    neighbours hang, and stay on the straight edges of those neighbours, where their
    constraints put them.  Every other node of the top moves onto the curve."""
    from dualmesh import _core

    def bend(point):
        return (point[0], 1.0 + 0.1 * point[0] * (1.0 - point[0]), point[2])

    hierarchy = _core.Subdivision(
        dualmesh.generate_box_mesh(0, 1, 0, 1, 0, 1, 2, 2, 2, element_type="Hex8")
    )
    hierarchy.project("top", bend)
    points = numpy.asarray(hierarchy.mesh().points())
    marked = []
    for _type, cell, _block in hierarchy.mesh().cells():
        corners = points[list(cell)]
        marked.append(
            int(
                corners[:, 0].max() < 0.6
                and corners[:, 2].max() < 0.6
                and corners[:, 1].min() > 0.4
            )
        )
    assert sum(marked) == 1
    hierarchy.refine(marked, 5)
    hierarchy.project("top", bend)
    mesh = hierarchy.mesh()
    points = numpy.asarray(mesh.points())
    hanging = dict(hierarchy.hanging())
    off_the_curve = 0
    for node in set(mesh.boundary_nodes("top")):
        curve = bend(points[node])[1]
        if node in hanging:
            straight = sum(weight * points[primary, 1] for primary, weight in hanging[node])
            assert points[node, 1] == pytest.approx(straight, abs=1e-14)
            off_the_curve += abs(straight - curve) > 1e-3
        else:
            assert points[node, 1] == pytest.approx(curve, abs=1e-14)
    assert off_the_curve >= 1
