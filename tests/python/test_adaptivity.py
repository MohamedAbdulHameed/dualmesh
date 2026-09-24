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

import dualmesh as dm
import numpy as np
import pytest


def unit_square(num_elements=4):
    return dm.generate_rectangle_mesh(
        x_min=0.0,
        x_max=1.0,
        y_min=0.0,
        y_max=1.0,
        num_x_elements=num_elements,
        num_y_elements=num_elements,
        element_type="Tri3",
    )


def signed_areas(mesh):
    points = np.asarray(mesh.points())
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
    return np.asarray(out)


def l_shaped_mesh(num_elements=2):
    """The unit square [-1, 1]^2 with the quadrant x > 0, y < 0 removed."""
    full = dm.generate_rectangle_mesh(
        x_min=-1.0,
        x_max=1.0,
        y_min=-1.0,
        y_max=1.0,
        num_x_elements=2 * num_elements,
        num_y_elements=2 * num_elements,
        element_type="Tri3",
    )
    points = np.asarray(full.points())
    kept = []
    for _type, connectivity, _block in full.cells():
        centroid = points[list(connectivity)].mean(axis=0)
        if not (centroid[0] > 0.0 and centroid[1] < 0.0):
            kept.append(list(connectivity))
    used = sorted({node for cell in kept for node in cell})
    renumber = {old: new for new, old in enumerate(used)}
    mesh = dm.mesh_from_arrays(
        points[used], [[renumber[node] for node in cell] for cell in kept], "Tri3"
    )
    mesh.add_sideset_by_predicate("boundary", lambda x, y, z: True)
    return mesh


def corner_solution(x, y):
    radius = np.hypot(x, y)
    angle = np.arctan2(y, x)
    angle = np.where(angle < -1e-12, angle + 2.0 * np.pi, angle)
    return radius ** (2.0 / 3.0) * np.sin(2.0 / 3.0 * angle)


def solve_corner_problem(mesh):
    problem = dm.Problem(mesh)
    problem.add_variable("u")
    problem.add_kernel("Diffusion", "diffusion", variable="u")
    problem.add_boundary_condition(
        "DirichletBC",
        "boundary",
        variable="u",
        boundary="boundary",
        value=lambda x, y, z, t: float(corner_solution(np.array(x), np.array(y))),
    )
    problem.solve()
    points = problem.entity_points()
    error = float(np.abs(problem.values("u") - corner_solution(points[:, 0], points[:, 1])).max())
    return problem, error


# ---------------------------------------------------------------------------
# Refinement
# ---------------------------------------------------------------------------
def test_refinement_only_grows_the_mesh_where_it_was_asked_to():
    mesh = unit_square(4)
    marked = np.zeros(mesh.num_elements, dtype=bool)
    marked[0] = True
    refined, parents = dm.refine_marked(mesh, marked)
    # Bisecting one triangle creates two, and conformity drags in a few
    # neighbours, but nothing like a uniform refinement, which would give 128.
    assert mesh.num_elements < refined.num_elements < mesh.num_elements + 8
    assert len(parents) == refined.num_elements
    assert set(parents.tolist()) <= set(range(mesh.num_elements))
    # The element that was marked is gone, replaced by children.
    assert (parents == 0).sum() >= 2


def test_repeated_refinement_keeps_every_triangle_properly_oriented():
    """Longest-edge bisection can never produce a degenerate or inverted
    triangle, and Rivara's theorem bounds the smallest angle below by half the
    smallest angle of the original mesh.  Both are checked here over many
    rounds of random marking, which is the way an adaptive run actually
    exercises the algorithm."""
    mesh = unit_square(2)
    generator = np.random.default_rng(0)
    smallest_ratio = 1.0
    for _ in range(12):
        areas = signed_areas(mesh)
        assert areas.min() > 0.0
        smallest_ratio = min(smallest_ratio, areas.min() / areas.max())
        marked = generator.random(mesh.num_elements) < 0.4
        marked[0] = True
        mesh, _ = dm.refine_marked(mesh, marked)
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
        points = np.asarray(mesh.points())
        centroids = np.array([points[list(c)].mean(axis=0) for _t, c, _b in mesh.cells()])
        marked = (centroids[:, 0] < 0.25) & (centroids[:, 1] < 0.25)
        mesh, _ = dm.refine_marked(mesh, marked)

    def exact(x, y, z, t):
        return 1.0 + 2.0 * x - 3.0 * y

    for method in ("dmcdm", "fem", "hfvm"):
        problem = dm.Problem(mesh, method=method)
        problem.add_variable("u")
        problem.add_kernel("Diffusion", "diffusion", variable="u")
        problem.add_boundary_condition(
            "DirichletBC", "walls", variable="u", boundary=mesh.sideset_names(), value=exact
        )
        problem.solve()
        points = problem.entity_points()
        expected = 1.0 + 2.0 * points[:, 0] - 3.0 * points[:, 1]
        assert problem.values("u") == pytest.approx(expected, abs=1e-11)


def test_refinement_carries_the_boundary_sets_across():
    """Marking every element bisects every hypotenuse, which leaves the
    boundary edges alone; a second round then reaches them.  Whatever the
    refinement does, the nodes of a side set must still lie on the geometric
    boundary that names it."""
    mesh = unit_square(4)
    refined = mesh
    for _ in range(2):
        refined, _ = dm.refine_marked(refined, np.ones(refined.num_elements, dtype=bool))
    assert sorted(refined.sideset_names()) == sorted(mesh.sideset_names())
    points = np.asarray(refined.points())
    for name, coordinate, value in (
        ("left", 0, 0.0),
        ("right", 0, 1.0),
        ("bottom", 1, 0.0),
        ("top", 1, 1.0),
    ):
        ids = refined.boundary_nodes(name)
        assert len(ids) > 5
        assert points[ids, coordinate] == pytest.approx(value)


def test_refinement_of_a_non_triangular_mesh_is_refused_with_a_reason():
    mesh = dm.generate_rectangle_mesh(
        x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0, num_x_elements=2, num_y_elements=2
    )
    with pytest.raises(Exception, match="triangular"):
        dm.refine_marked(mesh, np.ones(mesh.num_elements, dtype=bool))


def test_a_wrong_sized_marker_is_refused():
    mesh = unit_square(2)
    with pytest.raises(Exception, match="one entry per element"):
        dm.refine_marked(mesh, np.ones(mesh.num_elements + 1, dtype=bool))


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
    problem = dm.Problem(mesh)
    problem.add_variable("u")
    problem.add_kernel("Diffusion", "diffusion", variable="u")
    problem.add_boundary_condition(
        "DirichletBC",
        "walls",
        variable="u",
        boundary=mesh.sideset_names(),
        value=lambda x, y, z, t: 1.0 + 2.0 * x - 3.0 * y,
    )
    problem.solve()
    assert np.abs(problem.error_indicator("u")).max() < 1e-12


def test_the_indicator_is_largest_where_the_solution_bends_most():
    """For u = exp(-20 x) the curvature is concentrated near x = 0, so the
    elements there must carry the largest indicators."""
    mesh = unit_square(8)
    problem = dm.Problem(mesh)
    problem.add_variable("u")
    problem.add_kernel("Diffusion", "diffusion", variable="u")
    problem.add_kernel(
        "BodyForce", "source", variable="u", value=lambda x, y, z, t: 400.0 * np.exp(-20.0 * x)
    )
    problem.add_boundary_condition(
        "DirichletBC",
        "walls",
        variable="u",
        boundary=mesh.sideset_names(),
        value=lambda x, y, z, t: np.exp(-20.0 * x),
    )
    problem.solve()
    indicators = problem.error_indicator("u")
    points = np.asarray(mesh.points())
    centroids = np.array([points[list(c)].mean(axis=0) for _t, c, _b in mesh.cells()])
    near = centroids[:, 0] < 0.25
    assert indicators[near].mean() > 4.0 * indicators[~near].mean()
    assert indicators.argmax() in np.flatnonzero(near)


def test_the_indicator_is_not_available_for_cell_centred_finite_volume():
    mesh = unit_square(4)
    problem = dm.Problem(mesh, method="zfvm")
    problem.add_variable("u")
    problem.add_kernel("Diffusion", "diffusion", variable="u")
    problem.add_boundary_condition(
        "DirichletBC", "walls", variable="u", boundary=mesh.sideset_names(), value=0.0
    )
    problem.solve()
    with pytest.raises(Exception, match="cell-centred"):
        problem.error_indicator("u")


# ---------------------------------------------------------------------------
# Marking
# ---------------------------------------------------------------------------
def test_mark_by_fraction_takes_the_worst_elements():
    indicators = np.array([1.0, 5.0, 2.0, 4.0, 3.0])
    marked = dm.mark_by_fraction(indicators, 0.4)
    assert marked.tolist() == [False, True, False, True, False]


def test_mark_by_error_fraction_takes_as_few_elements_as_it_can():
    """Bulk marking refines the smallest set carrying the requested share of
    the squared error.  Here one element holds 100 of the total 104 units, so
    asking for half of the error marks that element alone."""
    indicators = np.array([10.0, 1.0, 1.0, 1.0, 1.0])
    marked = dm.mark_by_error_fraction(indicators, 0.5)
    assert marked.tolist() == [True, False, False, False, False]
    # Asking for nearly all of it has to reach down into the small elements.
    assert dm.mark_by_error_fraction(indicators, 0.99).sum() == 4


def test_mark_by_threshold_is_absolute():
    indicators = np.array([1.0, 5.0, 2.0])
    assert dm.mark_by_threshold(indicators, 1.5).tolist() == [False, True, True]


@pytest.mark.parametrize("marker", [dm.mark_by_fraction, dm.mark_by_error_fraction])
def test_a_marker_never_returns_an_empty_selection_for_a_positive_error(marker):
    assert marker(np.array([1.0, 1.0, 1.0, 1.0]), 0.1).any()


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
        sizes = np.array([n for n, _ in history], dtype=float)
        errors = np.array([e for _, e in history], dtype=float)
        return float(np.polyfit(np.log(sizes), np.log(errors), 1)[0])

    start = l_shaped_mesh(2)

    uniform_history = []
    mesh = start
    for cycle in range(5):
        _problem, error = solve_corner_problem(mesh)
        uniform_history.append((mesh.num_nodes, error))
        if cycle < 4:
            mesh = mesh.refined()

    adaptive_history = []
    mesh = start
    for cycle in range(12):
        problem, error = solve_corner_problem(mesh)
        adaptive_history.append((mesh.num_nodes, error))
        if cycle < 11:
            indicators = problem.error_indicator("u")
            mesh, _ = dm.refine_marked(mesh, dm.mark_by_error_fraction(indicators, 0.4))

    uniform_slope = slope(uniform_history)
    adaptive_slope = slope(adaptive_history[-7:])
    assert -0.45 < uniform_slope < -0.2  # held back by the singularity
    assert adaptive_slope < -0.8  # close to the optimal rate
    # At a comparable number of unknowns the adaptive mesh is the better one.
    assert adaptive_history[-1][0] < uniform_history[-1][0]
    assert adaptive_history[-1][1] < uniform_history[-1][1]


def test_the_adaptive_driver_runs_the_whole_loop():
    def build(mesh):
        problem = dm.Problem(mesh)
        problem.add_variable("u")
        problem.add_kernel("Diffusion", "diffusion", variable="u")
        problem.add_boundary_condition(
            "DirichletBC",
            "boundary",
            variable="u",
            boundary="boundary",
            value=lambda x, y, z, t: float(corner_solution(np.array(x), np.array(y))),
        )
        return problem

    seen = []
    problem, mesh = dm.solve_with_adaptive_refinement(
        build,
        l_shaped_mesh(2),
        variable="u",
        num_cycles=8,
        marker=lambda eta: dm.mark_by_error_fraction(eta, 0.6),
        callback=lambda cycle, pb, eta: seen.append((cycle, len(eta))),
    )
    assert [cycle for cycle, _ in seen] == list(range(8))
    assert [count for _, count in seen] == sorted(count for _, count in seen)
    assert mesh.num_elements == len(problem.error_indicator("u"))
    points = problem.entity_points()
    error = np.abs(problem.values("u") - corner_solution(points[:, 0], points[:, 1])).max()
    # Eight cycles take the 24-element starting mesh to about 180 elements and
    # the error from 3.6e-2 to a little over 1e-2, which a uniform refinement
    # needs some 3000 elements to match.
    assert error < 0.015


def test_the_adaptive_driver_stops_at_the_element_budget():
    def build(mesh):
        problem = dm.Problem(mesh)
        problem.add_variable("u")
        problem.add_kernel("Diffusion", "diffusion", variable="u")
        problem.add_boundary_condition(
            "DirichletBC", "walls", variable="u", boundary=mesh.sideset_names(), value=0.0
        )
        problem.add_kernel("BodyForce", "source", variable="u", value=1.0)
        return problem

    _problem, mesh = dm.solve_with_adaptive_refinement(
        build, unit_square(4), variable="u", num_cycles=20, max_elements=200
    )
    assert mesh.num_elements >= 200
    assert mesh.num_elements < 600
