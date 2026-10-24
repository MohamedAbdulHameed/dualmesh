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

PARTITIONERS = list(dm._core.available_partitioners())


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


def test_sub_mesh_keeps_the_boundary_and_covers_every_node():
    mesh = dm.generate_rectangle_mesh(0.0, 1.0, 0.0, 1.0, 8, 8)
    # Any division of the elements will do: element e goes to part e mod 4.
    parts = np.arange(mesh.num_elements) % 4
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
    problem.add_kernel("heat_conduction", variable="temperature", thermal_conductivity=2.5)
    problem.add_kernel("body_force", variable="temperature", value=40.0)
    problem.add_boundary_condition(
        "Dirichlet",
        ["left", "bottom"],
        variable="temperature",
        value=20.0,
    )
    problem.add_boundary_condition(
        "Robin",
        ["right", "top"],
        variable="temperature",
        transfer_coeff=15.0,
        ambient_value=5.0,
    )


@pytest.mark.parametrize("method", ["dmcdm", "fem", "hfvm", "zfvm"])
@pytest.mark.parametrize(
    "solver",
    [
        {},
        {"linear_solver": "lu"},
        {
            "linear_solver": "petsc",
            "petsc_options": "-ksp_type gmres -pc_type asm -sub_pc_type ilu",
        },
        {"linear_solver": "petsc", "petsc_options": "-ksp_type gmres -pc_type gamg"},
    ],
    ids=["automatic", "lu", "schwarz", "gamg"],
)
@pytest.mark.parametrize("gather_scatter", ["petsc", "gslib"])
def test_distributed_path_reproduces_the_serial_answer(method, solver, gather_scatter):
    if gather_scatter == "gslib" and not dm.have_gslib():
        pytest.skip("built without gslib")
    mesh = dm.generate_rectangle_mesh(0.0, 1.0, 0.0, 1.0, 12, 12)

    # Under mpirun a problem is distributed by default, so the reference is made serial explicitly.
    serial = dm.Problem(mesh, method=method, distributed=False)
    conduction_problem(serial)
    serial.solve()
    expected = serial.values("temperature")

    distributed = dm.Problem(mesh, method=method, distributed=True, gather_scatter=gather_scatter)
    conduction_problem(distributed)
    distributed.solve(linear_tolerance=1e-13, **solver)
    got = np.asarray(distributed.gathered_values("temperature"))
    assert got == pytest.approx(expected, abs=1e-8)
    assert distributed.num_ranks == dm.num_ranks()
    # The unknowns of zfvm sit at the cells and the boundary faces, those of the other methods at the nodes.
    assert distributed.num_global_dofs == len(expected)


def test_distributed_transient_reproduces_the_serial_answer():
    mesh = dm.generate_rectangle_mesh(0.0, 1.0, 0.0, 1.0, 10, 10)

    def define(problem):
        problem.add_variable(
            "temperature",
            initial_condition=lambda x, y, z, t: np.sin(np.pi * x) * np.sin(np.pi * y),
        )
        problem.add_kernel("diffusion", variable="temperature")
        problem.add_kernel("time_derivative", variable="temperature")
        problem.add_boundary_condition(
            "Dirichlet",
            ["left", "right", "bottom", "top"],
            variable="temperature",
            value=0.0,
        )

    serial = dm.Problem(mesh, distributed=False)
    define(serial)
    serial.solve_transient(end_time=0.02, time_step=0.002, implicitness=0.5)
    expected = serial.values("temperature")

    distributed = dm.Problem(mesh, distributed=True)
    define(distributed)
    distributed.solve_transient(
        end_time=0.02, time_step=0.002, implicitness=0.5, linear_tolerance=1e-13
    )
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
        problem = dm.Problem(mesh, method=method, distributed=False)
        conduction_problem(problem)
        problem.set_num_threads(threads)
        problem.solve()
        results[threads] = problem.values("temperature")
    assert results[1] == pytest.approx(results[4], rel=0, abs=1e-12)


def test_threading_is_disabled_for_python_objects():
    """A function written in Python needs the interpreter lock, so the assembly
    must fall back to one thread."""
    mesh = dm.generate_rectangle_mesh(0.0, 1.0, 0.0, 1.0, 40, 40)
    problem = dm.Problem(mesh, distributed=False)
    problem.add_variable("u")
    problem.add_kernel("diffusion", variable="u")
    problem.add_function("source", lambda x, y, z, t: np.sin(x))
    problem.add_kernel("body_force", variable="u", value="source")
    problem.add_boundary_condition(
        "Dirichlet",
        ["left", "right", "bottom", "top"],
        variable="u",
        value=0.0,
    )
    problem.set_num_threads(4)
    problem.solve()
    assert not problem.thread_safe
    assert problem.effective_threads() == 1


def test_number_of_threads_is_reported():
    mesh = dm.generate_rectangle_mesh(0.0, 1.0, 0.0, 1.0, 40, 40)
    problem = dm.Problem(mesh, distributed=False)
    conduction_problem(problem)
    problem.set_num_threads(3)
    problem.solve()
    assert problem.thread_safe
    # 1600 elements and 400 elements per thread means at most four threads.
    assert 1 <= problem.effective_threads() <= 3


def test_the_default_linear_solver_is_petsc_automatic():
    """A distributed problem is solved by PETSc, by default with the automatic choice (empty petsc_options).
    The C++ options struct and the Python defaults must agree."""
    from dualmesh.problem import _distributed_options

    options = dm._core.DistributedOptions()
    linear, _ = _distributed_options({})
    assert options.petsc_options == linear.petsc_options == ""
    lu, _ = _distributed_options({"linear_solver": "lu"})
    assert "mumps" in lu.petsc_options


@pytest.mark.parametrize(
    "solver",
    [{"preconditioner": "ilu"}, {"gmres_restart": 30}, {"linear_solver": "gmres"}],
)
def test_a_distributed_problem_takes_its_linear_solver_settings_from_petsc_options(solver):
    """PETSc solves a distributed problem, so the settings of the serial linear solvers are refused with a message that names petsc_options."""
    mesh = dm.generate_rectangle_mesh(0.0, 1.0, 0.0, 1.0, 4, 4)
    distributed = dm.Problem(mesh, distributed=True)
    conduction_problem(distributed)
    with pytest.raises(ValueError, match="(?i)petsc"):
        distributed.solve(**solver)


@pytest.mark.skipif(not dm.have_petsc(), reason="built without PETSc")
def test_a_linear_solve_that_cannot_converge_says_what_to_try():
    """When the distributed iteration cannot reach the tolerance, the error
    must name the number of iterations and the residual reached and say what
    to try, rather than simply saying that it failed."""
    mesh = dm.generate_rectangle_mesh(
        x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0, num_x_elements=8, num_y_elements=8
    )
    problem = dm.Problem(mesh, distributed=True)
    problem.add_variable("u")
    problem.add_kernel("diffusion", variable="u")
    problem.add_kernel("body_force", variable="u", value=1.0)
    problem.add_boundary_condition(
        "Dirichlet",
        mesh.sideset_names(),
        variable="u",
        value=0.0,
    )
    with pytest.raises(RuntimeError, match="did not converge.*Adjust petsc_options"):
        problem.solve(
            linear_solver="petsc",
            petsc_options="-ksp_type gmres -pc_type jacobi",
            linear_tolerance=1e-14,
            linear_max_iterations=2,
        )


@pytest.mark.skipif(not dm.have_petsc(), reason="built without PETSc")
@pytest.mark.parametrize("theory", ["Euler_Bernoulli", "Timoshenko"])
def test_a_distributed_mixed_beam_is_solved_or_refused_never_left_at_zero(theory):
    """The mixed beam has a zero diagonal in its deflection equation, on which algebraic multigrid
    fails without an error: PETSc then reports convergence on the preconditioned residual.  The
    true residual sends the automatic choice on to the direct solver, and the tip of the cantilever
    deflects by q L^4 / (8 EI) + q L^2 / (2 k G A)."""
    young, nu, length, load, width, height = 2.0e11, 0.3, 2.0, 1.0e4, 0.1, 0.2
    mesh = dm.generate_line_mesh(start=0.0, end=length, num_elements=32)
    problem = dm.Problem(mesh, method="fem", distributed=True)
    beam = problem.add_physics(
        "solid_mechanics",
        formulation="beam",
        theory=theory,
        youngs_modulus=young,
        poissons_ratio=nu,
        cross_section={"width": width, "height": height},
        transverse_load=load,
    )
    beam.add_boundary_condition("fixed", boundary="left")
    problem.solve(report="none")
    inertia, area = width * height**3 / 12, width * height
    exact = load * length**4 / (8 * young * inertia)
    if theory == "Timoshenko":
        exact += load * length**2 / (2 * 5 / 6 * young / (2 * (1 + nu)) * area)
    tip = problem.sample("disp_z", [[length, 0.0, 0.0]])[0]
    assert tip == pytest.approx(exact, rel=2e-3)


@pytest.mark.parametrize("method", ["dmcdm", "fem", "hfvm"])
@pytest.mark.parametrize("density", [0.0, 50.0])
def test_distributed_pressure_velocity_flow_reproduces_the_serial_answer(method, density):
    """The enclosed cavity in the pressure-velocity formulation.  Its pressure
    level is fixed at the node nearest to a point, and each process sees only
    its part of the mesh, so only the globally nearest node may be kept;
    otherwise every process would pin a node of its own."""

    def define(problem):
        problem.add_physics(
            "fluid_dynamics",
            dynamic_viscosity=1.0,
            density=density,
            formulation="pressure",
            pressure_pin_point=(0.5, 0.0),
        )
        problem.add_boundary_condition("Dirichlet", "top", variable="velocity_x", value=1.0)
        problem.add_boundary_condition("Dirichlet", "top", variable="velocity_y", value=0.0)
        for variable in ("velocity_x", "velocity_y"):
            problem.add_boundary_condition(
                "Dirichlet",
                ["left", "right", "bottom"],
                variable=variable,
                value=0.0,
            )

    mesh = dm.generate_rectangle_mesh(0.0, 1.0, 0.0, 1.0, 16, 16)
    serial = dm.Problem(mesh, method=method, distributed=False)
    define(serial)
    serial.solve()
    distributed = dm.Problem(mesh, method=method, distributed=True)
    define(distributed)
    distributed.solve(linear_tolerance=1e-12)
    for variable in ("velocity_x", "velocity_y", "pressure"):
        got = np.asarray(distributed.gathered_values(variable))
        assert got == pytest.approx(serial.values(variable), abs=1e-8)


@pytest.mark.skipif(not dm.have_petsc(), reason="built without PETSc")
@pytest.mark.parametrize("method", ["dmcdm", "fem", "hfvm"])
@pytest.mark.parametrize(
    "options",
    [None, {"ksp_type": "gmres", "pc_type": "asm", "sub_pc_type": "ilu", "ksp_gmres_restart": 400}],
)
def test_distributed_petsc_reproduces_the_serial_answer(method, options):
    """With linear_solver="petsc" the Jacobian is assembled as one distributed
    PETSc matrix, from the entries every rank computed for its own elements
    in a global numbering, and solved by PETSc."""

    def define(problem):
        problem.add_physics(
            "fluid_dynamics",
            dynamic_viscosity=1.0,
            density=50.0,
            formulation="pressure",
            pressure_pin_point=(0.5, 0.0),
        )
        problem.add_boundary_condition("Dirichlet", "top", variable="velocity_x", value=1.0)
        problem.add_boundary_condition("Dirichlet", "top", variable="velocity_y", value=0.0)
        for variable in ("velocity_x", "velocity_y"):
            problem.add_boundary_condition(
                "Dirichlet",
                ["left", "right", "bottom"],
                variable=variable,
                value=0.0,
            )

    mesh = dm.generate_rectangle_mesh(0.0, 1.0, 0.0, 1.0, 16, 16)
    serial = dm.Problem(mesh, method=method, distributed=False)
    define(serial)
    serial.solve(linear_solver="lu")
    distributed = dm.Problem(mesh, method=method, distributed=True)
    define(distributed)
    distributed.solve(linear_solver="petsc", petsc_options=options, linear_tolerance=1e-12)
    for variable in ("velocity_x", "velocity_y", "pressure"):
        got = np.asarray(distributed.gathered_values(variable))
        assert got == pytest.approx(serial.values(variable), abs=1e-8)


def _taylor_hood_cavity(problem):
    problem.add_physics(
        "fluid_dynamics",
        dynamic_viscosity=1.0,
        density=50.0,
        formulation="Taylor_Hood",
        pressure_pin_point=(0.5, 0.0),
    )
    problem.add_boundary_condition("Dirichlet", "top", variable="velocity_x", value=1.0)
    problem.add_boundary_condition("Dirichlet", "top", variable="velocity_y", value=0.0)
    for variable in ("velocity_x", "velocity_y"):
        problem.add_boundary_condition(
            "Dirichlet",
            ["left", "right", "bottom"],
            variable=variable,
            value=0.0,
        )


_TAYLOR_HOOD_SOLVERS = [{}] + (
    [
        {"linear_solver": "petsc"},
        {
            "linear_solver": "petsc",
            "petsc_options": "-ksp_type fgmres -ksp_rtol 1e-12 -pc_type fieldsplit "
            "-pc_fieldsplit_0_fields 0,1 -pc_fieldsplit_1_fields 2 -pc_fieldsplit_type schur "
            "-pc_fieldsplit_schur_fact_type full -pc_fieldsplit_schur_precondition selfp "
            "-fieldsplit_0_ksp_type preonly -fieldsplit_0_pc_type lu "
            "-fieldsplit_0_pc_factor_mat_solver_type mumps -fieldsplit_1_ksp_type gmres "
            "-fieldsplit_1_ksp_rtol 1e-10 -fieldsplit_1_pc_type jacobi",
        },
    ]
    if dm.have_petsc()
    else []
)


@pytest.mark.parametrize("options", _TAYLOR_HOOD_SOLVERS)
def test_distributed_taylor_hood_reproduces_the_serial_answer(options):
    """The mid-side pressure slots are inactive on every rank, the pin lands on
    the globally nearest corner node, and the solution agrees with the serial
    one on any number of ranks."""
    mesh = dm.generate_rectangle_mesh(0.0, 1.0, 0.0, 1.0, 8, 8, element_type="Quad9")
    serial = dm.Problem(mesh, method="fem", distributed=False)
    _taylor_hood_cavity(serial)
    serial.solve()
    distributed = dm.Problem(mesh, method="fem", distributed=True)
    _taylor_hood_cavity(distributed)
    distributed.solve(linear_tolerance=1e-12, **options)
    for variable in ("velocity_x", "velocity_y", "pressure"):
        got = np.asarray(distributed.gathered_values(variable))
        assert got == pytest.approx(serial.values(variable), abs=1e-8)


def test_an_unknown_gather_scatter_is_refused():
    mesh = dm.generate_rectangle_mesh(0.0, 1.0, 0.0, 1.0, 4, 4)
    with pytest.raises(ValueError, match="'petsc' or 'gslib'"):
        dm.Problem(mesh, distributed=True, gather_scatter="crystal")


@pytest.mark.parametrize("method", ["dmcdm", "fem", "hfvm"])
@pytest.mark.parametrize("gather_scatter", ["petsc", "gslib"])
def test_distributed_periodic_boundaries_reproduce_the_serial_answer(method, gather_scatter):
    """A domain periodic in both directions, whose periodic node pairs lie on different processes."""
    if gather_scatter == "gslib" and not dm.have_gslib():
        pytest.skip("built without gslib")

    def define(problem):
        problem.add_variable("u")
        problem.add_kernel("diffusion", variable="u")
        problem.add_kernel("reaction", variable="u")
        problem.add_kernel("body_force", variable="u", value="1 + cos(2*pi*x + 0.3)*sin(2*pi*y)")
        problem.add_boundary_condition("periodic", primary="left", secondary="right")
        problem.add_boundary_condition("periodic", primary="bottom", secondary="top")

    mesh = dm.generate_rectangle_mesh(0.0, 1.0, 0.0, 1.0, 12, 10)
    serial = dm.Problem(mesh, method=method, distributed=False)
    define(serial)
    serial.solve(report="none")
    distributed = dm.Problem(mesh, method=method, distributed=True, gather_scatter=gather_scatter)
    define(distributed)
    distributed.solve(linear_tolerance=1e-13, report="none")
    got = np.asarray(distributed.gathered_values("u"))
    assert got == pytest.approx(serial.values("u"), abs=1e-8)


@pytest.mark.parametrize("method", ["dmcdm", "fem", "hfvm"])
def test_distributed_postprocessors_equal_the_serial_ones(method):
    """Every post-processor type gives the same value on any number of processes: integrals and fluxes are summed over the processes, extremes are reduced, and a point value comes from the process whose elements contain the point."""

    def build(distributed):
        mesh = dm.generate_rectangle_mesh(0.0, 1.0, 0.0, 1.0, 12, 12)
        problem = dm.Problem(mesh, method=method, distributed=distributed)
        conduction_problem(problem)
        problem.add_postprocessor("variable_integral", "integral", variable="temperature")
        problem.add_postprocessor("variable_average", "average", variable="temperature")
        problem.add_postprocessor(
            "point_value", "inside", variable="temperature", point=[0.31, 0.67]
        )
        problem.add_postprocessor(
            "point_value", "on_a_node", variable="temperature", point=[0.5, 0.5]
        )
        problem.add_postprocessor("nodal_extreme_value", "hottest", variable="temperature")
        problem.add_postprocessor(
            "nodal_extreme_value", "coldest", variable="temperature", value_type="min"
        )
        problem.add_postprocessor(
            "total_reaction", "heat_out", variable="temperature", boundary="left"
        )
        problem.solve(linear_tolerance=1e-13, report="none")
        return {k: v[-1] for k, v in problem.postprocessor_values().items()}

    serial, distributed = build(False), build(True)
    for name, value in serial.items():
        assert distributed[name] == pytest.approx(value, rel=1e-9, abs=1e-9), name


def _creep_block(distributed):
    """A block pulled at a constant stress that creeps at A sigma^n exp(-Q/T): its creep strain lives in the history of the material."""
    A, n, Q, T, stress = 1e-30, 3.0, 2000.0, 700.0, 50e6
    mesh = dm.generate_box_mesh(0, 1e-3, 0, 1e-3, 0, 2e-3, 2, 2, 4)
    p = dm.Problem(mesh, distributed=distributed)
    for v in ("temperature", "ux", "uy", "uz"):
        p.add_variable(v)
    p.add_kernel("diffusion", variable="temperature")
    p.add_boundary_condition(
        "Dirichlet",
        list(mesh.sideset_names()),
        variable="temperature",
        value=T,
    )
    p.add_property(
        "small_strain_stress",
        displacements=["ux", "uy", "uz"],
        formulation="three_dimensional",
        youngs_modulus=1e11,
        poissons_ratio=0.3,
        creep_model="parsed",
        creep_rate="A * von_mises_stress^n * exp(-Q / temperature)",
        creep_constants={"A": A, "n": n, "Q": Q},
        temperature="temperature",
    )
    for v in ("ux", "uy", "uz"):
        p.add_kernel("stress_divergence", variable=v, displacements=["ux", "uy", "uz"])
    for v, side in (("ux", "left"), ("uy", "bottom"), ("uz", "back")):
        p.add_boundary_condition("Dirichlet", side, variable=v, value=0.0)
    p.add_boundary_condition("traction", "front", variable="uz", traction=stress)
    p.add_postprocessor("nodal_extreme_value", "stretch", variable="uz")
    p.solve(report="none")
    return p


@pytest.mark.skipif(not dm._core.have_checkpoints(), reason="checkpoints need PETSc")
@pytest.mark.parametrize("distributed", [False, True])
def test_a_restarted_creep_run_ends_where_an_uninterrupted_one_ends(distributed):
    """The checkpoint holds the solution, the history of the material (the creep strain) and the history of the post-processors, so the restarted run continues exactly."""
    import pathlib
    import shutil

    # Under mpirun every process runs its own serial problem, so each writes its own checkpoint; a distributed problem writes one for all.
    owner = "all" if distributed else str(dm.parallel.rank())
    directory = pathlib.Path(f"dualmesh_checkpoint_test_{owner}")
    directory.mkdir(exist_ok=True)

    def values(problem):
        if problem.is_distributed:
            return np.asarray(problem.gathered_values("uz"))
        return np.asarray(problem.values("uz"))

    whole = _creep_block(distributed)
    whole.solve_transient(end_time=1e6, time_step=1e5, report="none")
    stopped = _creep_block(distributed)
    stopped.solve_transient(
        end_time=5e5,
        time_step=1e5,
        report="none",
        output=dm.Output(directory=str(directory), file_base="creep", checkpoint_interval=2),
    )
    restarted = _creep_block(distributed)
    result = restarted.solve_transient(
        end_time=1e6, time_step=1e5, report="none", restart=str(directory / "creep.chk")
    )
    assert result.time_steps == 5
    assert values(restarted) == pytest.approx(values(whole), rel=1e-10, abs=1e-18)
    expected = whole.postprocessor_values()
    got = restarted.postprocessor_values()
    assert got["time"] == pytest.approx(expected["time"])
    assert got["stretch"] == pytest.approx(expected["stretch"], rel=1e-10)
    # The creep strain is in the history: without it the restarted run would end elsewhere.
    assert values(whole).max() > 1.05 * values(_creep_block(distributed)).max()
    if not distributed or restarted.rank == 0:
        shutil.rmtree(directory, ignore_errors=True)


@pytest.mark.parametrize("method", ["fem", "dmcdm", "hfvm"])
def test_a_distributed_gap_condition_reproduces_the_serial_answer(method):
    """Heat crosses a gap between two bodies whose meshes do not match, and the bodies lie on different processes: every process that holds the primary side reads the secondary elements it pairs with as ghost elements."""
    from test_gaps import CONDUCTANCE, conduction_problem, two_slabs

    def solve(distributed):
        mesh = two_slabs(2, 8, inner_num_y=3, outer_num_y=5)
        problem = conduction_problem(mesh, method, distributed=distributed)
        problem.add_boundary_condition(
            "gap_heat_transfer",
            "primary",
            variable="temperature",
            secondary_boundary=["secondary"],
            gap_conductance=CONDUCTANCE,
        )
        problem.solve(report="none", linear_tolerance=1e-13)
        if problem.is_distributed:
            return np.asarray(problem.gathered_values("temperature"))
        return np.asarray(problem.values("temperature"))

    serial, distributed = solve(False), solve(True)
    assert distributed == pytest.approx(serial, rel=1e-9)


def test_distributed_error_norms_are_those_of_the_whole_domain():
    """Each process integrates its own elements, and the squares add: the norms of a distributed
    problem equal the serial ones, so a convergence study gives the same orders on any number of
    processes (run under mpirun, this compares the parts with the whole)."""
    exact = "sin(pi*x)*sin(pi*y)"
    gradient = ["pi*cos(pi*x)*sin(pi*y)", "pi*sin(pi*x)*cos(pi*y)"]

    def norms(distributed):
        mesh = dm.generate_rectangle_mesh(0.0, 1.0, 0.0, 1.0, num_elements=[16, 16])
        problem = dm.Problem(mesh, method="fem", distributed=distributed)
        problem.add_variable("u")
        problem.add_kernel("diffusion", variable="u")
        problem.add_kernel("body_force", variable="u", value="2*pi^2*" + exact)
        problem.add_boundary_condition(
            "Dirichlet", ["left", "right", "bottom", "top"], variable="u", value=0.0
        )
        problem.solve(report="none")
        return problem.error_norms("u", exact, gradient)

    serial, distributed = norms(False), norms(True)
    assert distributed == pytest.approx(serial, rel=1e-10)


def test_a_distributed_error_controlled_transient_reproduces_the_serial_answer():
    """The error estimate of the adaptive step is formed from the global norm, so every process
    takes the same step and the run neither stalls nor drifts from the serial answer."""

    def run(distributed):
        mesh = dm.generate_rectangle_mesh(0.0, 1.0, 0.0, 1.0, num_elements=[8, 8])
        problem = dm.Problem(mesh, method="fem", distributed=distributed)
        heat = problem.add_physics(
            "heat_transfer",
            thermal_conductivity=1.0,
            density=1.0,
            specific_heat=1.0,
            heat_source=1.0,
        )
        heat.add_boundary_condition("temperature = 0", boundary=["left", "right"])
        problem.solve_transient(end_time=0.1, time_step=0.01, time_stepper="error", report="none")
        return problem.sample("temperature", [[0.5, 0.5]])[0]

    assert run(True) == pytest.approx(run(False), rel=1e-9)


@pytest.mark.parametrize(
    "option, message",
    [
        ({"nonlinear_solver": "newtn"}, "Unknown nonlinear solver"),
        ({"line_search": "bogus"}, "Unknown line_search"),
    ],
)
def test_a_distributed_solve_refuses_options_it_does_not_have(option, message):
    mesh = dm.generate_rectangle_mesh(0.0, 1.0, 0.0, 1.0, 4, 4)
    problem = dm.Problem(mesh, distributed=True)
    conduction_problem(problem)
    with pytest.raises(ValueError, match=message):
        problem.solve(report="none", **option)


def test_a_distributed_solve_with_the_line_search_reproduces_the_serial_answer():
    """One nonlinear driver serves the serial and the distributed problem, so the backtracking
    line search works on every number of processes."""

    def solve(distributed):
        mesh = dm.generate_rectangle_mesh(0.0, 1.0, 0.0, 1.0, num_elements=[8, 8])
        problem = dm.Problem(mesh, method="fem", distributed=distributed)
        problem.add_variable("u")
        problem.add_kernel("diffusion", variable="u", diffusivity="1 + 10*u^2")
        problem.add_kernel("body_force", variable="u", value=50.0)
        problem.add_boundary_condition(
            "Dirichlet", ["left", "right", "bottom", "top"], variable="u", value=0.0
        )
        problem.solve(report="none", line_search="backtracking")
        return problem.sample("u", [[0.5, 0.5]])[0]

    assert solve(True) == pytest.approx(solve(False), rel=1e-10)


def _equal_order_cavity(problem):
    problem.add_physics(
        "fluid_dynamics",
        dynamic_viscosity=1.0,
        density=50.0,
        formulation="pressure",
        pressure_pin_point=(0.5, 0.0),
    )
    problem.add_boundary_condition("Dirichlet", "top", variable="velocity_x", value=1.0)
    problem.add_boundary_condition("Dirichlet", "top", variable="velocity_y", value=0.0)
    for variable in ("velocity_x", "velocity_y"):
        problem.add_boundary_condition(
            "Dirichlet", ["left", "right", "bottom"], variable=variable, value=0.0
        )


@pytest.mark.parametrize("preconditioner", ["lsc", "pcd", "pressure_mass"])
@pytest.mark.parametrize(
    "element_type, define", [("Quad4", _equal_order_cavity), ("Quad9", _taylor_hood_cavity)]
)
def test_distributed_block_preconditioners_reproduce_the_serial_answer(
    preconditioner, element_type, define
):
    """The operators of the Schur complement approximation are assembled on every part and
    summed in the global numbering, and the sign of the Schur complement is measured from the
    coupling blocks of all parts together."""
    n = 16 if element_type == "Quad4" else 8
    mesh = dm.generate_rectangle_mesh(0.0, 1.0, 0.0, 1.0, n, n, element_type=element_type)
    serial = dm.Problem(mesh, method="fem", distributed=False)
    define(serial)
    serial.solve(linear_solver="lu")
    distributed = dm.Problem(mesh, method="fem", distributed=True)
    define(distributed)
    distributed.solve(preconditioner=preconditioner, linear_tolerance=1e-12)
    for variable in ("velocity_x", "velocity_y", "pressure"):
        got = np.asarray(distributed.gathered_values(variable))
        assert got == pytest.approx(serial.values(variable), abs=1e-8)
    if dm.have_petsc():
        # The same split with exact block solves on one process and distributed: an operator
        # summed wrongly over the parts would change the count, not the answer.  The blocks are
        # solved exactly because one V-cycle of BoomerAMG leaves round-off differences of the
        # parallel sums from run to run, which pcd amplifies (116 to 217 iterations per Newton
        # step measured on two processes).
        exact = {
            "-fieldsplit_0_pc_type": "lu",
            "-fieldsplit_0_pc_factor_mat_solver_type": "mumps",
            "-fieldsplit_1_laplacian_pc_type": "lu",
            "-fieldsplit_1_laplacian_pc_factor_mat_solver_type": "mumps",
            "-fieldsplit_1_pc_type": "lu",
            "-fieldsplit_1_pc_factor_mat_solver_type": "mumps",
        }
        split = dm.Problem(mesh, method="fem", distributed=True)
        define(split)
        result = split.solve(
            preconditioner=preconditioner, petsc_options=exact, linear_tolerance=1e-12
        )
        one = dm.Problem(mesh, method="fem", distributed=False)
        define(one)
        expected = one.solve(
            linear_solver="petsc",
            preconditioner=preconditioner,
            petsc_options=exact,
            linear_tolerance=1e-12,
        )
        per_step = result.linear_iterations / result.total_iterations
        assert per_step <= expected.linear_iterations / expected.total_iterations + 2


def test_a_distributed_lu_takes_no_preconditioner():
    mesh = dm.generate_rectangle_mesh(0.0, 1.0, 0.0, 1.0, 4, 4)
    distributed = dm.Problem(mesh, distributed=True)
    conduction_problem(distributed)
    with pytest.raises(ValueError, match="takes no preconditioner"):
        distributed.solve(linear_solver="lu", preconditioner="amg")


def test_a_distributed_problem_names_the_closest_preconditioner():
    mesh = dm.generate_rectangle_mesh(0.0, 1.0, 0.0, 1.0, 4, 4)
    distributed = dm.Problem(mesh, distributed=True)
    conduction_problem(distributed)
    with pytest.raises(ValueError, match="Did you mean 'LSC'"):
        distributed.solve(preconditioner="lcs")


def _thermoelastic_plate(problem):
    heat = problem.add_physics("heat_transfer", thermal_conductivity=20.0, heat_source=1.0e6)
    heat.add_boundary_condition("temperature = 300", boundary="left")
    solid = problem.add_physics(
        "solid_mechanics", formulation="plane_stress", youngs_modulus=2.0e11, thickness=0.01
    )
    solid.add_boundary_condition("fixed", boundary="left")
    problem.add_coupling(
        "thermal_expansion",
        heat_transfer=heat,
        solid_mechanics=solid,
        linear_expansion_coeff=1.2e-5,
        stress_free_temperature=293.0,
    )


def _cantilever(problem):
    solid = problem.add_physics(
        "solid_mechanics", formulation="plane_stress", youngs_modulus=2.0e11, thickness=0.01
    )
    solid.add_boundary_condition("fixed", boundary="left")
    solid.add_boundary_condition("traction = [0, -1e6]", boundary="right")


_SOLID = ("disp_x", "disp_y")
_THERMOELASTIC = ("temperature", "disp_x", "disp_y")


@pytest.mark.parametrize(
    "define, variables, preconditioner",
    [
        (_cantilever, _SOLID, "amg"),
        (_cantilever, _SOLID, None),
        (_thermoelastic_plate, _THERMOELASTIC, "physics_blocks"),
        (_thermoelastic_plate, _THERMOELASTIC, None),
    ],
)
def test_distributed_solid_preconditioners_reproduce_the_serial_answer(
    define, variables, preconditioner
):
    """The rigid body modes of the solid reach PETSc's smoothed aggregation on the owned rows of
    every process, and the split by physics solves the temperature, then the displacements.  The
    automatic choice takes the first for a solid and the second for a solid coupled to heat."""
    mesh = dm.generate_rectangle_mesh(x=[0.0, 0.1], y=[0.0, 0.02], num_elements=[40, 8])
    serial = dm.Problem(mesh, distributed=False)
    define(serial)
    serial.solve(linear_solver="lu", report="none")
    distributed = dm.Problem(mesh, distributed=True)
    define(distributed)
    options = {"preconditioner": preconditioner} if preconditioner else {}
    distributed.solve(linear_tolerance=1e-12, report="none", **options)
    for variable in variables:
        expected = serial.values(variable)
        got = np.asarray(distributed.gathered_values(variable))
        assert np.abs(got - expected).max() <= 1e-8 * np.abs(expected).max()


@pytest.mark.skipif(not dm.have_petsc(), reason="built without PETSc")
def test_distributed_amg_takes_the_rigid_body_modes_of_a_solid():
    """A three-dimensional cantilever: GAMG with the rigid body modes takes 31 iterations on two
    processes; without the rotations it took 169."""
    mesh = dm.generate_box_mesh(x=[0, 10], y=[0, 1], z=[0, 1], num_elements=[40, 4, 4])
    problem = dm.Problem(mesh, distributed=True)
    solid = problem.add_physics("solid_mechanics", youngs_modulus=2e11, poissons_ratio=0.3)
    solid.add_boundary_condition("fixed", boundary="left")
    solid.add_boundary_condition("traction = [0, -1e6, 0]", boundary="right")
    result = problem.solve(report="none", linear_tolerance=1e-10, preconditioner="amg")
    assert result.linear_iterations <= 60
