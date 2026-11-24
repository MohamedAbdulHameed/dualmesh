# SPDX-License-Identifier: LGPL-2.1-or-later
"""Post-processors: the scalars computed after every solve and time step."""

from __future__ import annotations

import dualmesh
import numpy
import pytest


def _linear_field_problem():
    """u = x on the unit square: integral 1/2, average 1/2, maximum 1."""
    mesh = dualmesh.generate_rectangle_mesh(0, 1, 0, 1, 4, 4)
    problem = dualmesh.Problem(mesh, method="fem")
    problem.add_variable("u")
    conduction = problem.add_kernel("diffusion", variable="u")
    problem.add_boundary_condition("Dirichlet", "left", variable="u", value=0.0)
    problem.add_boundary_condition("Dirichlet", "right", variable="u", value=1.0)
    return problem, conduction


def test_the_scalar_types_give_the_exact_values_of_a_linear_field():
    problem, conduction = _linear_field_problem()
    problem.add_postprocessor("integral", "integral", variable="u")
    problem.add_postprocessor("average", "average", variable="u")
    problem.add_postprocessor("point_value", "at_point", variable="u", point=[0.3, 0.7])
    problem.add_postprocessor("maximum", "largest", variable="u")
    problem.add_postprocessor("minimum", "smallest", variable="u")
    problem.add_postprocessor("boundary_flux", "flux_out_left", variable="u", boundary="left")
    problem.add_postprocessor(
        "kernel_boundary_flux", "flux_out", kernel=conduction, boundary="right"
    )
    problem.solve()
    values = {k: v[-1] for k, v in problem.postprocessor_values().items()}
    assert values["integral"] == pytest.approx(0.5)
    assert values["average"] == pytest.approx(0.5)
    assert values["at_point"] == pytest.approx(0.3)
    assert values["largest"] == pytest.approx(1.0)
    assert values["smallest"] == pytest.approx(0.0)
    # A unit gradient carries a unit flux through the unit-height boundaries, from the right
    # (held at 1) to the left (held at 0): it leaves through the left.
    assert values["flux_out_left"] == pytest.approx(1.0)
    assert abs(values["flux_out"]) == pytest.approx(1.0)


def _distorted_mesh(dimension, element_type):
    # The unit interval, square or cube with its interior nodes moved by a
    # smooth map that keeps the boundary in place.
    if dimension == 1:
        mesh = dualmesh.generate_line_mesh(0.0, 1.0, 5)
    elif dimension == 2:
        mesh = dualmesh.generate_rectangle_mesh(
            0, 1, 0, 1, num_elements=[4, 3], element_type=element_type
        )
    else:
        mesh = dualmesh.generate_box_mesh(
            0, 1, 0, 1, 0, 1, num_elements=[3, 3, 2], element_type=element_type
        )
    if element_type in ("Edge3", "Quad9", "Hex27", "Tet10"):
        mesh = mesh.second_order()
    bump = 0.06 if dimension == 1 else 0.08

    def move(x, y, z):
        s = numpy.sin(numpy.pi * x) * (numpy.sin(numpy.pi * y) if dimension > 1 else 1.0)
        s *= numpy.sin(numpy.pi * z) if dimension > 2 else 1.0
        shift = [bump * s, 0.5 * bump * s, -0.5 * bump * s][:dimension]
        return list(numpy.add([x, y, z][:dimension], shift)) + [y, z][dimension - 1 :]

    mesh.transform_nodes(move)
    mesh.fix_orientation()
    return mesh


@pytest.mark.parametrize(
    "dimension, element_type",
    [
        (1, "Edge2"),
        (1, "Edge3"),
        (2, "Quad4"),
        (2, "Quad9"),
        (2, "Tri3"),
        (3, "Hex8"),
        (3, "Hex27"),
        (3, "Tet4"),
        (3, "Tet10"),
    ],
)
def test_a_point_value_on_a_distorted_mesh_is_exact_for_a_linear_field(dimension, element_type):
    # The solution of Laplace's equation with a linear boundary value is that
    # linear field, which every isoparametric element holds exactly, so the
    # value at a point between the nodes checks the inverse map of the
    # element that contains the point.
    mesh = _distorted_mesh(dimension, element_type)
    exact = "1 + 2*x + 3*y + 4*z"
    problem = dualmesh.Problem(mesh, method="fem")
    problem.add_variable("u")
    problem.add_kernel("diffusion", variable="u")
    for name in mesh.sideset_names():
        problem.add_boundary_condition("Dirichlet", name, variable="u", value=exact)
    point = [0.37, 0.61, 0.43][:dimension]
    problem.add_postprocessor("point_value", "at_point", variable="u", point=point)
    problem.solve()
    x, y, z = point + [0.0] * (3 - dimension)
    value = problem.postprocessor_values()["at_point"][-1]
    assert value == pytest.approx(1 + 2 * x + 3 * y + 4 * z, rel=1e-10)


def test_a_transient_records_every_accepted_step(tmp_path):
    mesh = dualmesh.generate_line_mesh(0.0, 1.0, 10)
    problem = dualmesh.Problem(mesh, method="fem")
    problem.add_variable("u", initial_condition=1.0)
    problem.add_kernel("time_derivative", variable="u")
    problem.add_kernel("diffusion", variable="u")
    problem.add_postprocessor("average", "mean", variable="u")
    problem.solve_transient(end_time=0.3, time_step=0.1)
    history = problem.postprocessor_values()
    assert history["time"] == pytest.approx([0.0, 0.1, 0.2, 0.3])
    # Nothing leaves through the insulated ends: the mean is conserved.
    assert history["mean"] == pytest.approx(numpy.ones(4))
    problem.write_postprocessor_csv(tmp_path / "history.csv")
    lines = (tmp_path / "history.csv").read_text().splitlines()
    assert lines[0] == "time,mean" and len(lines) == 5


def test_misuse_is_refused_with_the_valid_choices():
    problem, conduction = _linear_field_problem()
    with pytest.raises(ValueError, match="average"):
        problem.add_postprocessor("Average", "a", variable="u")
    with pytest.raises(ValueError, match="takes: variable"):
        problem.add_postprocessor("average", "a", field="u")
    with pytest.raises(ValueError, match="takes: variable"):
        problem.add_postprocessor("maximum", "e", variable="u", value_type="max")
    # A misspelled type is answered with the closest one.
    with pytest.raises(ValueError, match="Did you mean 'average'"):
        problem.add_postprocessor("averge", "a", variable="u")
    # A flow through a boundary needs its physics, which this problem has not.
    problem.add_postprocessor("heat_flow", "q", boundary="left")
    with pytest.raises(ValueError, match="give the heat_transfer physics"):
        problem.solve()
    problem.add_postprocessor("average", "a", variable="u")
    with pytest.raises(ValueError, match="already exists"):
        problem.add_postprocessor("average", "a", variable="u")
