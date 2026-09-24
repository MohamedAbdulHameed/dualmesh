# SPDX-License-Identifier: LGPL-2.1-or-later
"""Tests of the parallel machinery.

These run in a single process.  They check that the mesh partitioners produce
valid, balanced partitions, that a sub-mesh keeps everything a rank needs, that
the distributed code path reproduces the serial answer, and that threading the
assembly does not change the result.  The behaviour on several MPI ranks is
checked by the C++ program ``dualmesh_parallel_tests``, which is launched with
``mpirun`` (see docs/parallel.rst).
"""

from __future__ import annotations

import dualmesh as dm
import numpy as np
import pytest

PARTITIONERS = ["recursive_coordinate_bisection", "graph"] + (["metis"] if dm.have_metis() else [])


@pytest.mark.parametrize("method", PARTITIONERS)
@pytest.mark.parametrize("num_parts", [2, 4, 7])
@pytest.mark.parametrize("element_type", ["Quad4", "Tri3"])
def test_partition_is_complete_and_balanced(method, num_parts, element_type):
    mesh = dm.generate_rectangle_mesh(0.0, 2.0, 0.0, 1.0, 16, 8, element_type=element_type)
    partition = dm.partition_mesh(mesh, num_parts, method)
    parts = np.asarray(partition["element_part"])
    assert parts.min() >= 0 and parts.max() == num_parts - 1
    assert len(parts) == mesh.num_elements
    average = len(parts) / num_parts
    assert partition["largest_part"] <= 1.3 * average + 1
    assert partition["smallest_part"] >= 0.7 * average - 1
    assert partition["edge_cut"] > 0


def test_metis_cuts_less_than_the_graph_partitioner():
    """On an unstructured mesh METIS cuts fewer faces than the built-in
    graph partitioner.  It does not beat recursive coordinate bisection on a
    structured grid, where bisection is already close to optimal."""
    if not dm.have_metis():
        pytest.skip("this build has no METIS")
    mesh = dm.generate_rectangle_mesh(0.0, 1.0, 0.0, 1.0, 20, 20, element_type="Tri3")
    cuts = {m: dm.partition_mesh(mesh, 4, m)["edge_cut"] for m in PARTITIONERS}
    assert cuts["metis"] < cuts["graph"]


def test_sub_mesh_keeps_the_boundary_and_covers_every_node():
    mesh = dm.generate_rectangle_mesh(0.0, 1.0, 0.0, 1.0, 8, 8)
    partition = dm.partition_mesh(mesh, 4, "graph")
    parts = np.asarray(partition["element_part"])
    covered = np.zeros(mesh.num_nodes, dtype=bool)
    total_sides = 0
    for p in range(4):
        elements = np.flatnonzero(parts == p)
        part_mesh, local_to_global = dm.parallel.sub_mesh(mesh, elements)
        assert part_mesh.num_elements == len(elements)
        covered[list(local_to_global)] = True
        for name in ("left", "right", "bottom", "top"):
            total_sides += len(part_mesh.sideset(name))
    assert covered.all()
    expected = sum(len(mesh.sideset(name)) for name in ("left", "right", "bottom", "top"))
    assert total_sides == expected


def conduction_problem(problem):
    """Heat conduction with a source, a prescribed temperature on two sides and
    convection on the other two."""
    problem.add_variable("temperature")
    problem.add_kernel(
        "HeatConduction", "conduction", variable="temperature", thermal_conductivity=2.5
    )
    problem.add_kernel("HeatSource", "source", variable="temperature", heat_source=40.0)
    problem.add_boundary_condition(
        "DirichletBC", "cold", variable="temperature", boundary=["left", "bottom"], value=20.0
    )
    problem.add_boundary_condition(
        "ConvectiveHeatFluxBC",
        "convection",
        variable="temperature",
        boundary=["right", "top"],
        heat_transfer_coefficient=15.0,
        ambient_temperature=5.0,
    )


@pytest.mark.parametrize("method", ["dmcdm", "fem", "hfvm"])
@pytest.mark.parametrize(
    "preconditioner, overlap, subdomain_solver",
    [
        ("jacobi", 0, "ilu"),
        ("additive_schwarz", 0, "ilu"),
        ("additive_schwarz", 1, "lu"),
        ("two_level_schwarz", 1, "ilu"),
        ("two_level_schwarz", 2, "lu"),
    ],
)
def test_distributed_path_reproduces_the_serial_answer(
    method, preconditioner, overlap, subdomain_solver
):
    mesh = dm.generate_rectangle_mesh(0.0, 1.0, 0.0, 1.0, 12, 12)

    serial = dm.Problem(mesh, method=method)
    conduction_problem(serial)
    serial.solve()
    expected = serial.values("temperature")

    distributed = dm.DistributedProblem(
        mesh,
        method=method,
        preconditioner=preconditioner,
        overlap=overlap,
        subdomain_solver=subdomain_solver,
        linear_tolerance=1e-13,
    )
    conduction_problem(distributed.local)
    distributed.solve()
    got = np.asarray(distributed.gathered_values("temperature"))
    assert got == pytest.approx(expected, abs=1e-8)
    assert distributed.num_ranks == dm.num_ranks()
    assert distributed.num_global_dofs == mesh.num_nodes


def test_distributed_transient_reproduces_the_serial_answer():
    mesh = dm.generate_rectangle_mesh(0.0, 1.0, 0.0, 1.0, 10, 10)

    def define(problem):
        problem.add_variable(
            "temperature",
            initial_condition=lambda x, y, z, t: np.sin(np.pi * x) * np.sin(np.pi * y),
        )
        problem.add_kernel("Diffusion", "diffusion", variable="temperature")
        problem.add_kernel("TimeDerivative", "time", variable="temperature")
        problem.add_boundary_condition(
            "DirichletBC",
            "edges",
            variable="temperature",
            boundary=["left", "right", "bottom", "top"],
            value=0.0,
        )

    serial = dm.Problem(mesh)
    define(serial)
    serial.solve_transient(end_time=0.02, dt=0.002, theta=0.5)
    expected = serial.values("temperature")

    distributed = dm.DistributedProblem(mesh, linear_tolerance=1e-13)
    define(distributed.local)
    distributed.solve_transient(end_time=0.02, dt=0.002, theta=0.5)
    got = np.asarray(distributed.gathered_values("temperature"))
    assert got == pytest.approx(expected, abs=1e-8)


@pytest.mark.parametrize("method", ["dmcdm", "fem", "hfvm", "zfvm"])
def test_threading_does_not_change_the_answer(method):
    """The threaded assembly must give the same numbers as the serial one.

    The mesh is large enough that threading is actually switched on; the test
    uses no Python-defined objects, because those force the assembly onto one
    thread.
    """
    results = {}
    for threads in (1, 4):
        mesh = dm.generate_rectangle_mesh(0.0 if method != "zfvm" else 0.0, 1.0, 0.0, 1.0, 40, 40)
        problem = dm.Problem(mesh, method=method)
        conduction_problem(problem)
        problem.set_num_threads(threads)
        problem.solve()
        results[threads] = problem.values("temperature")
    assert results[1] == pytest.approx(results[4], rel=0, abs=1e-12)


def test_threading_is_disabled_for_python_objects():
    """A function written in Python needs the interpreter lock, so the assembly
    must fall back to one thread."""
    mesh = dm.generate_rectangle_mesh(0.0, 1.0, 0.0, 1.0, 40, 40)
    problem = dm.Problem(mesh)
    problem.add_variable("u")
    problem.add_kernel("Diffusion", "diffusion", variable="u")
    problem.add_function("source", lambda x, y, z, t: np.sin(x))
    problem.add_kernel("BodyForce", "body", variable="u", value="source")
    problem.add_boundary_condition(
        "DirichletBC", "edges", variable="u", boundary=["left", "right", "bottom", "top"], value=0.0
    )
    problem.set_num_threads(4)
    problem.solve()
    assert not problem.thread_safe
    assert problem.effective_threads() == 1


def test_number_of_threads_is_reported():
    mesh = dm.generate_rectangle_mesh(0.0, 1.0, 0.0, 1.0, 40, 40)
    problem = dm.Problem(mesh)
    conduction_problem(problem)
    problem.set_num_threads(3)
    problem.solve()
    assert problem.thread_safe
    # 1600 elements and 400 elements per thread means at most four threads.
    assert 1 <= problem.effective_threads() <= 3


def test_the_default_preconditioner_is_two_level_schwarz_with_overlap():
    """Restricted additive Schwarz on subdomains that overlap by one layer of
    elements, with a coarse level, is the default: its iteration count stays
    flat as ranks are added (the multi-rank check is in
    dualmesh_parallel_tests) and it is several times lower than Jacobi's.
    Both layers must agree on the default: the C++ options struct and the
    Python constructor that mirrors it."""
    import inspect

    options = dm._core.DistributedOptions()
    signature = inspect.signature(dm.DistributedProblem.__init__).parameters
    assert options.preconditioner == signature["preconditioner"].default == "two_level_schwarz"
    assert options.overlap == signature["overlap"].default == 1
    assert options.subdomain_solver == signature["subdomain_solver"].default == "ilu"


def test_bad_schwarz_options_are_reported():
    mesh = dm.generate_rectangle_mesh(0.0, 1.0, 0.0, 1.0, 4, 4)
    with pytest.raises(ValueError, match="overlap"):
        dm.DistributedProblem(mesh, overlap=-1)
    distributed = dm.DistributedProblem(mesh, subdomain_solver="multigrid")
    conduction_problem(distributed.local)
    with pytest.raises(ValueError, match="Unknown subdomain solver"):
        distributed.solve()


def test_the_cell_centred_finite_volume_method_is_refused_by_the_distributed_solver():
    """The distributed decomposition identifies a degree of freedom by the mesh
    node it sits on.  The cell-centred method puts its unknowns at cell and
    boundary face centroids instead, so the ownership mask, the exchange of
    shared values and the inner product do not apply to it, and it would also
    need a layer of ghost cells across each partition boundary.  Rather than
    return a wrong answer, the constructor refuses, and the message says which
    methods to use instead."""
    mesh = dm.generate_rectangle_mesh(
        x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0, num_x_elements=4, num_y_elements=4
    )
    with pytest.raises(Exception, match="cell-centred finite volume"):
        dm.DistributedProblem(mesh, method="zfvm")
    # The three node-based methods are accepted.
    for method in ("dmcdm", "fem", "hfvm"):
        dm.DistributedProblem(mesh, method=method)


def test_a_linear_solve_that_cannot_converge_says_what_to_try():
    """When the distributed iteration cannot reach the tolerance, the error
    must name the number of iterations and the residual reached and say what
    to try, rather than simply saying that it failed."""
    mesh = dm.generate_rectangle_mesh(
        x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0, num_x_elements=8, num_y_elements=8
    )
    distributed = dm.DistributedProblem(
        mesh, preconditioner="jacobi", linear_tolerance=1e-14, linear_max_iterations=2
    )
    problem = distributed.local
    problem.add_variable("u")
    problem.add_kernel("Diffusion", "diffusion", variable="u")
    problem.add_kernel("BodyForce", "source", variable="u", value=1.0)
    problem.add_boundary_condition(
        "DirichletBC", "walls", variable="u", boundary=mesh.sideset_names(), value=0.0
    )
    with pytest.raises(RuntimeError, match="did not converge in 2 iterations"):
        distributed.solve()
