# SPDX-License-Identifier: LGPL-2.1-or-later
"""Tests of the parallel machinery.

These run in a single process.  They check that the mesh partitioners produce
valid, balanced partitions, that a sub-mesh keeps everything a rank needs, that
the distributed code path reproduces the serial answer, and that threading the
assembly does not change the result.  The behaviour on several MPI ranks is
checked by the C++ program ``dualmesh_parallel_tests``, which is launched with
``mpirun`` (see docs/theory/parallel.rst).
"""

from __future__ import annotations

import dualmesh
import numpy
import pytest


def conduction_problem(problem):
    """Heat conduction with a source, a prescribed temperature on two sides and
    convection on the other two."""
    problem.add_variable("temperature")
    problem.add_kernel("diffusion", variable="temperature", diffusivity=2.5)
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
        transfer_coefficient=15.0,
        ambient_value=5.0,
    )


@pytest.mark.parametrize("method", ["dmcdm", "fem"])
@pytest.mark.parametrize(
    "solver",
    [
        {},
        {"petsc_options": "-ksp_type preonly -pc_type lu -pc_factor_mat_solver_type mumps"},
        {"petsc_options": "-ksp_type gmres -pc_type asm -sub_pc_type ilu"},
        {"petsc_options": "-ksp_type gmres -pc_type gamg"},
    ],
    ids=["automatic", "lu", "schwarz", "gamg"],
)
@pytest.mark.parametrize("gather_scatter", ["petsc", "gslib"])
def test_distributed_path_reproduces_the_serial_answer(method, solver, gather_scatter):
    if gather_scatter == "gslib" and not dualmesh.have_gslib():
        pytest.skip("built without gslib")
    mesh = dualmesh.generate_rectangle_mesh(0.0, 1.0, 0.0, 1.0, 12, 12)

    # Under mpirun a problem is distributed by default, so the reference is made serial explicitly.
    serial = dualmesh.Problem(mesh, method=method, distributed=False)
    conduction_problem(serial)
    serial.solve()
    expected = serial.values("temperature")

    distributed = dualmesh.Problem(
        mesh, method=method, distributed=True, gather_scatter=gather_scatter
    )
    conduction_problem(distributed)
    distributed.solve(**solver)
    got = numpy.asarray(distributed.gathered_values("temperature"))
    assert got == pytest.approx(expected, abs=1e-8)
    assert distributed.num_ranks == dualmesh.num_ranks()
    assert distributed.num_global_dofs == len(expected)


def test_distributed_transient_reproduces_the_serial_answer():
    mesh = dualmesh.generate_rectangle_mesh(0.0, 1.0, 0.0, 1.0, 10, 10)

    def define(problem):
        problem.add_variable(
            "temperature",
            initial_condition=lambda x, y, z, t: numpy.sin(numpy.pi * x) * numpy.sin(numpy.pi * y),
        )
        problem.add_kernel("diffusion", variable="temperature")
        problem.add_kernel("time_derivative", variable="temperature")
        problem.add_boundary_condition(
            "Dirichlet",
            ["left", "right", "bottom", "top"],
            variable="temperature",
            value=0.0,
        )

    serial = dualmesh.Problem(mesh, distributed=False)
    define(serial)
    serial.solve_transient(end_time=0.02, time_step=0.002, time_scheme="Crank-Nicolson")
    expected = serial.values("temperature")

    distributed = dualmesh.Problem(mesh, distributed=True)
    define(distributed)
    distributed.solve_transient(end_time=0.02, time_step=0.002, time_scheme="Crank-Nicolson")
    got = numpy.asarray(distributed.gathered_values("temperature"))
    assert got == pytest.approx(expected, abs=1e-8)


@pytest.mark.parametrize("method", ["dmcdm", "fem"])
def test_threading_does_not_change_the_answer(method):
    """The threaded assembly must give the same numbers as the serial one.

    The mesh is large enough that threading is actually switched on; the test
    uses no Python-defined objects, because those force the assembly onto one
    thread.
    """
    results = {}
    for threads in (1, 4):
        mesh = dualmesh.generate_rectangle_mesh(0.0, 1.0, 0.0, 1.0, 40, 40)
        problem = dualmesh.Problem(mesh, method=method, distributed=False)
        conduction_problem(problem)
        problem.set_num_threads(threads)
        problem.solve()
        results[threads] = problem.values("temperature")
    assert results[1] == pytest.approx(results[4], rel=0, abs=1e-12)


def test_a_linear_solve_that_cannot_converge_says_what_to_try():
    """When the distributed iteration cannot reach the tolerance, the error
    must name the number of iterations and the residual reached and say what
    to try, rather than simply saying that it failed."""
    mesh = dualmesh.generate_rectangle_mesh(
        x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0, num_x_elements=8, num_y_elements=8
    )
    problem = dualmesh.Problem(mesh, distributed=True)
    problem.add_variable("u")
    problem.add_kernel("diffusion", variable="u")
    problem.add_kernel("body_force", variable="u", value=1.0)
    problem.add_boundary_condition(
        "Dirichlet",
        mesh.sideset_names(),
        variable="u",
        value=0.0,
    )
    with pytest.raises(RuntimeError, match="did not converge.*PETSc_options"):
        problem.solve(
            petsc_options="-ksp_type gmres -pc_type jacobi -ksp_max_it 2", linear_tolerance=1e-14
        )


@pytest.mark.parametrize("theory", ["Euler-Bernoulli", "Timoshenko"])
def test_a_distributed_mixed_beam_is_solved_or_refused_never_left_at_zero(theory):
    """The mixed beam has a zero diagonal in its deflection equation, on which algebraic multigrid
    fails without an error: PETSc then reports convergence on the preconditioned residual.  The
    true residual sends the automatic choice on to the direct solver, and the tip of the cantilever
    deflects by q L^4 / (8 EI) + q L^2 / (2 k G A)."""
    Young, nu, length, load, width, height = 2.0e11, 0.3, 2.0, 1.0e4, 0.1, 0.2
    mesh = dualmesh.generate_line_mesh(start=0.0, end=length, num_elements=32)
    problem = dualmesh.Problem(mesh, method="fem", distributed=True)
    beam = problem.add_physics(
        "solid_mechanics",
        formulation="beam",
        theory=theory,
        youngs_modulus=Young,
        poissons_ratio=nu,
        cross_section={"width": width, "height": height},
        transverse_load=load,
    )
    beam.add_boundary_condition("fixed", boundary="left")
    problem.solve(report="none")
    inertia, area = width * height**3 / 12, width * height
    exact = load * length**4 / (8 * Young * inertia)
    if theory == "Timoshenko":
        exact += load * length**2 / (2 * 5 / 6 * Young / (2 * (1 + nu)) * area)
    tip = problem.sample("disp_z", [[length, 0.0, 0.0]])[0]
    assert tip == pytest.approx(exact, rel=2e-3)


@pytest.mark.parametrize("method", ["dmcdm", "fem"])
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
            formulation="equal_order",
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

    mesh = dualmesh.generate_rectangle_mesh(0.0, 1.0, 0.0, 1.0, 16, 16)
    serial = dualmesh.Problem(mesh, method=method, distributed=False)
    define(serial)
    serial.solve()
    distributed = dualmesh.Problem(mesh, method=method, distributed=True)
    define(distributed)
    distributed.solve()
    for variable in ("velocity_x", "velocity_y", "pressure"):
        got = numpy.asarray(distributed.gathered_values(variable))
        assert got == pytest.approx(serial.values(variable), abs=1e-8)


def _Taylor_hood_cavity(problem):
    problem.add_physics(
        "fluid_dynamics",
        dynamic_viscosity=1.0,
        density=50.0,
        formulation="Taylor-Hood",
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


_Taylor_hood_solvers = [
    {},
    {"petsc_options": "-ksp_type preonly -pc_type lu -pc_factor_mat_solver_type mumps"},
    {"petsc_options": "-dualmesh_schur lsc"},
    {
        "petsc_options": "-ksp_type fgmres -ksp_rtol 1e-12 -pc_type fieldsplit "
        "-pc_fieldsplit_0_fields 0,1 -pc_fieldsplit_1_fields 2 -pc_fieldsplit_type schur "
        "-pc_fieldsplit_schur_fact_type full -pc_fieldsplit_schur_precondition selfp "
        "-fieldsplit_0_ksp_type preonly -fieldsplit_0_pc_type lu "
        "-fieldsplit_0_pc_factor_mat_solver_type mumps -fieldsplit_1_ksp_type gmres "
        "-fieldsplit_1_ksp_rtol 1e-10 -fieldsplit_1_pc_type jacobi",
    },
]


@pytest.mark.parametrize("options", _Taylor_hood_solvers)
def test_distributed_Taylor_hood_reproduces_the_serial_answer(options):
    """The mid-side pressure slots are inactive on every rank, the pin lands on
    the globally nearest corner node, and the solution agrees with the serial
    one on any number of ranks."""
    mesh = dualmesh.generate_rectangle_mesh(0.0, 1.0, 0.0, 1.0, 8, 8, element_type="Quad9")
    serial = dualmesh.Problem(mesh, method="fem", distributed=False)
    _Taylor_hood_cavity(serial)
    serial.solve()
    distributed = dualmesh.Problem(mesh, method="fem", distributed=True)
    _Taylor_hood_cavity(distributed)
    distributed.solve(**options)
    for variable in ("velocity_x", "velocity_y", "pressure"):
        got = numpy.asarray(distributed.gathered_values(variable))
        assert got == pytest.approx(serial.values(variable), abs=1e-8)


@pytest.mark.parametrize("method", ["dmcdm", "fem"])
@pytest.mark.parametrize("gather_scatter", ["petsc", "gslib"])
def test_distributed_periodic_boundaries_reproduce_the_serial_answer(method, gather_scatter):
    """A domain periodic in both directions, whose periodic node pairs lie on different processes."""
    if gather_scatter == "gslib" and not dualmesh.have_gslib():
        pytest.skip("built without gslib")

    def define(problem):
        problem.add_variable("u")
        problem.add_kernel("diffusion", variable="u")
        problem.add_kernel("reaction", variable="u")
        problem.add_kernel("body_force", variable="u", value="1 + cos(2*pi*x + 0.3)*sin(2*pi*y)")
        problem.add_boundary_condition("periodic", primary="left", secondary="right")
        problem.add_boundary_condition("periodic", primary="bottom", secondary="top")

    mesh = dualmesh.generate_rectangle_mesh(0.0, 1.0, 0.0, 1.0, 12, 10)
    serial = dualmesh.Problem(mesh, method=method, distributed=False)
    define(serial)
    serial.solve(report="none")
    distributed = dualmesh.Problem(
        mesh, method=method, distributed=True, gather_scatter=gather_scatter
    )
    define(distributed)
    distributed.solve(report="none")
    got = numpy.asarray(distributed.gathered_values("u"))
    assert got == pytest.approx(serial.values("u"), abs=1e-8)


@pytest.mark.parametrize("method", ["dmcdm", "fem"])
def test_distributed_postprocessors_equal_the_serial_ones(method):
    """Every post-processor type gives the same value on any number of processes: integrals and fluxes are summed over the processes, extremes are reduced, and a point value comes from the process whose elements contain the point."""

    def build(distributed):
        mesh = dualmesh.generate_rectangle_mesh(0.0, 1.0, 0.0, 1.0, 12, 12)
        problem = dualmesh.Problem(mesh, method=method, distributed=distributed)
        conduction_problem(problem)
        problem.add_postprocessor("integral", "integral", variable="temperature")
        problem.add_postprocessor("average", "average", variable="temperature")
        problem.add_postprocessor(
            "point_value", "inside", variable="temperature", point=[0.31, 0.67]
        )
        problem.add_postprocessor(
            "point_value", "on_a_node", variable="temperature", point=[0.5, 0.5]
        )
        problem.add_postprocessor("maximum", "hottest", variable="temperature")
        problem.add_postprocessor("minimum", "coldest", variable="temperature")
        problem.add_postprocessor(
            "boundary_flux", "heat_out", variable="temperature", boundary="left"
        )
        problem.solve(report="none")
        return {k: v[-1] for k, v in problem.postprocessor_values().items()}

    serial, distributed = build(False), build(True)
    for name, value in serial.items():
        assert distributed[name] == pytest.approx(value, rel=1e-9, abs=1e-9), name


def _creep_block(distributed):
    """A block pulled at a constant stress that creeps at A sigma^n exp(-Q/T): its creep strain lives in the history of the material."""
    A, n, Q, T, stress = 1e-30, 3.0, 2000.0, 700.0, 50e6
    mesh = dualmesh.generate_box_mesh(0, 1e-3, 0, 1e-3, 0, 2e-3, 2, 2, 4)
    p = dualmesh.Problem(mesh, distributed=distributed)
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
    p.add_postprocessor("maximum", "stretch", variable="uz")
    p.solve(report="none")
    return p


@pytest.mark.skipif(not dualmesh._core.have_checkpoints(), reason="checkpoints need PETSc")
@pytest.mark.parametrize("distributed", [False, True])
def test_a_restarted_creep_run_ends_where_an_uninterrupted_one_ends(distributed):
    """The checkpoint holds the solution, the history of the material (the creep strain) and the history of the post-processors, so the restarted run continues exactly."""
    import pathlib
    import shutil

    # Under mpirun every process runs its own serial problem, so each writes its own checkpoint; a distributed problem writes one for all.
    owner = "all" if distributed else str(dualmesh.parallel.rank())
    directory = pathlib.Path(f"dualmesh_checkpoint_test_{owner}")
    directory.mkdir(exist_ok=True)

    def values(problem):
        if problem.is_distributed:
            return numpy.asarray(problem.gathered_values("uz"))
        return numpy.asarray(problem.values("uz"))

    whole = _creep_block(distributed)
    whole.solve_transient(end_time=1e6, time_step=1e5, time_stepper="fixed", report="none")
    stopped = _creep_block(distributed)
    stopped.solve_transient(
        end_time=5e5,
        time_step=1e5,
        time_stepper="fixed",
        report="none",
        output=dualmesh.Output(directory=str(directory), file_base="creep", checkpoint_interval=2),
    )
    restarted = _creep_block(distributed)
    result = restarted.solve_transient(
        end_time=1e6,
        time_step=1e5,
        time_stepper="fixed",
        report="none",
        restart=str(directory / "creep.chk"),
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


@pytest.mark.parametrize("method", ["fem", "dmcdm"])
def test_a_distributed_gap_condition_reproduces_the_serial_answer(method):
    """Heat crosses a gap between two bodies whose meshes do not match, and the bodies lie on different processes: every process that holds the primary side reads the secondary elements it pairs with as ghost elements."""
    from test_gaps import conductance, conduction_problem, two_slabs

    def solve(distributed):
        mesh = two_slabs(2, 8, inner_num_y=3, outer_num_y=5)
        problem = conduction_problem(mesh, method, distributed=distributed)
        problem.add_boundary_condition(
            "gap_heat_transfer",
            "primary",
            variable="temperature",
            secondary_boundary=["secondary"],
            gap_conductance=conductance,
        )
        problem.solve(report="none")
        if problem.is_distributed:
            return numpy.asarray(problem.gathered_values("temperature"))
        return numpy.asarray(problem.values("temperature"))

    serial, distributed = solve(False), solve(True)
    assert distributed == pytest.approx(serial, rel=1e-9)


def test_distributed_error_norms_are_those_of_the_whole_domain():
    """Each process integrates its own elements, and the squares add: the norms of a distributed
    problem equal the serial ones, so a convergence study gives the same orders on any number of
    processes (run under mpirun, this compares the parts with the whole)."""
    exact = "sin(pi*x)*sin(pi*y)"
    gradient = ["pi*cos(pi*x)*sin(pi*y)", "pi*sin(pi*x)*cos(pi*y)"]

    def norms(distributed):
        mesh = dualmesh.generate_rectangle_mesh(0.0, 1.0, 0.0, 1.0, num_elements=[16, 16])
        problem = dualmesh.Problem(mesh, method="fem", distributed=distributed)
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
        mesh = dualmesh.generate_rectangle_mesh(0.0, 1.0, 0.0, 1.0, num_elements=[8, 8])
        problem = dualmesh.Problem(mesh, method="fem", distributed=distributed)
        heat = problem.add_physics(
            "heat_conduction",
            thermal_conductivity=1.0,
            density=1.0,
            specific_heat=1.0,
            heat_source=1.0,
        )
        heat.add_boundary_condition("temperature = 0", boundary=["left", "right"])
        problem.solve_transient(end_time=0.1, time_step=0.01, time_stepper="error", report="none")
        return problem.sample("temperature", [[0.5, 0.5]])[0]

    assert run(True) == pytest.approx(run(False), rel=1e-9)


def test_a_distributed_solve_with_the_line_search_reproduces_the_serial_answer():
    """One nonlinear driver serves the serial and the distributed problem, so the backtracking
    line search works on every number of processes."""

    def solve(distributed):
        mesh = dualmesh.generate_rectangle_mesh(0.0, 1.0, 0.0, 1.0, num_elements=[8, 8])
        problem = dualmesh.Problem(mesh, method="fem", distributed=distributed)
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
        formulation="equal_order",
    )
    problem.add_boundary_condition("Dirichlet", "top", variable="velocity_x", value=1.0)
    problem.add_boundary_condition("Dirichlet", "top", variable="velocity_y", value=0.0)
    for variable in ("velocity_x", "velocity_y"):
        problem.add_boundary_condition(
            "Dirichlet", ["left", "right", "bottom"], variable=variable, value=0.0
        )


@pytest.mark.parametrize("approximation", ["lsc", "pcd", "pressure_mass"])
@pytest.mark.parametrize(
    "element_type, define", [("Quad4", _equal_order_cavity), ("Quad9", _Taylor_hood_cavity)]
)
def test_distributed_Schur_splits_reproduce_the_serial_answer(approximation, element_type, define):
    """The operators of the Schur complement approximation are assembled on every part and
    summed in the global numbering, and the sign of the Schur complement is measured from the
    coupling blocks of all parts together."""
    n = 16 if element_type == "Quad4" else 8
    mesh = dualmesh.generate_rectangle_mesh(0.0, 1.0, 0.0, 1.0, n, n, element_type=element_type)
    serial = dualmesh.Problem(mesh, method="fem", distributed=False)
    define(serial)
    serial.solve(petsc_options="-ksp_type preonly -pc_type lu")
    distributed = dualmesh.Problem(mesh, method="fem", distributed=True)
    define(distributed)
    distributed.solve(petsc_options=f"-dualmesh_schur {approximation}")
    for variable in ("velocity_x", "velocity_y", "pressure"):
        got = numpy.asarray(distributed.gathered_values(variable))
        assert got == pytest.approx(serial.values(variable), abs=1e-8)
    # The same split with exact block solves on one process and distributed: an operator summed
    # wrongly over the parts would change the count, not the answer.  The blocks are solved
    # exactly because one V-cycle of BoomerAMG leaves round-off differences of the parallel sums
    # from run to run, which pcd amplifies (116 to 217 iterations per Newton step measured on two
    # processes).
    exact = {
        "-dualmesh_schur": approximation,
        "-fieldsplit_0_pc_type": "lu",
        "-fieldsplit_0_pc_factor_mat_solver_type": "mumps",
        "-fieldsplit_1_laplacian_pc_type": "lu",
        "-fieldsplit_1_laplacian_pc_factor_mat_solver_type": "mumps",
        "-fieldsplit_1_pc_type": "lu",
        "-fieldsplit_1_pc_factor_mat_solver_type": "mumps",
    }
    split = dualmesh.Problem(mesh, method="fem", distributed=True)
    define(split)
    result = split.solve(petsc_options=exact)
    one = dualmesh.Problem(mesh, method="fem", distributed=False)
    define(one)
    expected = one.solve(petsc_options=exact)
    per_step = result.linear_iterations / result.total_iterations
    assert per_step <= expected.linear_iterations / expected.total_iterations + 2


@pytest.mark.parametrize("distributed", [False, True])
@pytest.mark.parametrize(
    "options, message",
    [
        ("-dualmesh_schur lcs", "-dualmesh_schur takes lsc, pcd or pressure_mass.*not 'lcs'"),
        ("-dualmesh_schur", "-dualmesh_schur takes lsc, pcd or pressure_mass"),
        ("-dualmesh_split physic", "-dualmesh_split takes physics.*not 'physic'"),
        ("-dualmesh_schur lsc -dualmesh_split physics", "choose two splits"),
    ],
)
def test_a_wrong_option_of_dualmesh_names_the_right_values(distributed, options, message):
    mesh = dualmesh.generate_rectangle_mesh(0.0, 1.0, 0.0, 1.0, 4, 4)
    problem = dualmesh.Problem(mesh, distributed=distributed)
    conduction_problem(problem)
    with pytest.raises(ValueError, match=message):
        problem.solve(petsc_options=options)


def _thermoelastic_plate(problem):
    heat = problem.add_physics("heat_conduction", thermal_conductivity=20.0, heat_source=1.0e6)
    heat.add_boundary_condition("temperature = 300", boundary="left")
    solid = problem.add_physics(
        "solid_mechanics",
        formulation="plane_stress",
        youngs_modulus=2.0e11,
        poissons_ratio=1 / 3,
        thickness=0.01,
    )
    solid.add_boundary_condition("fixed", boundary="left")
    problem.add_coupling(
        "thermal_expansion",
        heat_conduction=heat,
        solid_mechanics=solid,
        linear_expansion_coefficient=1.2e-5,
        stress_free_temperature=293.0,
    )


def _cantilever(problem):
    solid = problem.add_physics(
        "solid_mechanics",
        formulation="plane_stress",
        youngs_modulus=2.0e11,
        poissons_ratio=1 / 3,
        thickness=0.01,
    )
    solid.add_boundary_condition("fixed", boundary="left")
    solid.add_boundary_condition("traction = [0, -1e6]", boundary="right")


_solid = ("disp_x", "disp_y")
_thermoelastic = ("temperature", "disp_x", "disp_y")


@pytest.mark.parametrize(
    "define, variables, petsc_options",
    [
        (_cantilever, _solid, "-ksp_type gmres -ksp_pc_side right -pc_type gamg"),
        (_cantilever, _solid, None),
        (_thermoelastic_plate, _thermoelastic, "-dualmesh_split physics"),
        (_thermoelastic_plate, _thermoelastic, None),
    ],
)
def test_distributed_solid_preconditioners_reproduce_the_serial_answer(
    define, variables, petsc_options
):
    """The rigid body modes of the solid reach PETSc's smoothed aggregation on the owned rows of
    every process, and the split by physics solves the temperature, then the displacements.  A
    larger system takes the first for a solid and the second for a solid coupled to heat, and
    this one, small enough to factorise, the direct solver."""
    mesh = dualmesh.generate_rectangle_mesh(x=[0.0, 0.1], y=[0.0, 0.02], num_elements=[40, 8])
    serial = dualmesh.Problem(mesh, distributed=False)
    define(serial)
    serial.solve(petsc_options="-ksp_type preonly -pc_type lu", report="none")
    distributed = dualmesh.Problem(mesh, distributed=True)
    define(distributed)
    distributed.solve(report="none", petsc_options=petsc_options)
    for variable in variables:
        expected = serial.values(variable)
        got = numpy.asarray(distributed.gathered_values(variable))
        assert numpy.abs(got - expected).max() <= 1e-8 * numpy.abs(expected).max()


def test_the_adaptive_loop_of_the_input_runs_on_every_process():
    """Every process holds the same hierarchy of the whole mesh and adapts it from the indicators
    of the whole mesh, so a distributed run refines at the corner where the boundary temperature
    jumps, as a serial run does, and keeps the temperature between the two boundary values."""
    plate = dualmesh.problem(
        mesh={"rectangle": {"size": ["1 m", "1 m"], "cells": [8, 8], "cell_type": "triangle"}}
    )
    plate.physics("heat_conduction")
    plate.properties(thermal_conductivity="45 W/(m*K)")
    plate.boundary_conditions("left", temperature="400 K")
    plate.boundary_conditions("bottom", temperature="300 K")
    plate.mesh_adaptivity(
        field="temperature",
        refine_fraction=0.5,
        coarsen_fraction=0.05,
        maximum_refinement_level=3,
    )
    solution = plate.solve(report="none")
    points = solution.coordinates("temperature")
    assert int((numpy.hypot(points[:, 0], points[:, 1]) < 0.2).sum()) > 12
    temperature = solution.field("temperature")
    assert temperature.min() >= 300.0 - 1e-6
    assert temperature.max() <= 400.0 + 1e-6


def refined_toward_the_corner(dimension):
    """The unit square of 4 by 4 quadrilaterals or the unit cube of 3 by 3 by 3 hexahedra,
    refined twice at the corner at the origin, which leaves nodes hanging on the sides of the
    larger cells next to the refined ones."""
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
    return hierarchy.mesh()


# The simple partitioner gives every build the same parts.  With it, on 3 processes, a process
# owns a node that a constraint folds into without holding the constraint, which the
# distributed residual has to allow for.
@pytest.mark.parametrize("partitioner", ["automatic", "simple"])
@pytest.mark.parametrize("dimension", [2, 3])
@pytest.mark.parametrize("method", ["fem", "dmcdm"])
def test_a_distributed_mesh_with_hanging_nodes_passes_the_patch_test(
    method, dimension, partitioner
):
    """Every process that holds a hanging node holds the nodes it depends on, and every process
    that holds a large cell holds the nodes hanging on its sides, so FEM folds its constraints and
    DMCDM builds its polygon cells on every process, and a linear field stays exact.  Under FEM
    the equation of a hanging node goes to its primaries before the processes add their parts:
    the owner of a primary need not hold the constraint."""
    mesh = refined_toward_the_corner(dimension)
    problem = dualmesh.Problem(mesh, method=method, distributed=True, partitioner=partitioner)
    problem.add_variable("u")
    problem.add_kernel("diffusion", variable="u")
    problem.add_boundary_condition(
        "Dirichlet",
        mesh.sideset_names(),
        variable="u",
        value=lambda x, y, z, t: 1.0 + 2.0 * x - 3.0 * y + z,
    )
    problem.solve(report="none")
    points = numpy.asarray(mesh.points())
    expected = 1.0 + 2.0 * points[:, 0] - 3.0 * points[:, 1] + points[:, 2]
    assert numpy.asarray(problem.gathered_values("u")) == pytest.approx(expected, abs=1e-10)


@pytest.mark.parametrize("partitioner", ["automatic", "simple"])
@pytest.mark.parametrize("dimension", [2, 3])
def test_the_reactions_next_to_hanging_nodes_match_the_serial_ones(dimension, partitioner):
    """The sides at the origin hold nodes hanging between prescribed nodes.  The reaction of a
    prescribed primary takes the share of the equations of the nodes that hang on it, folded on
    each process before the processes add their parts: the owner of the primary need not hold
    the constraint.  So the flow through each side is the serial one."""
    mesh = refined_toward_the_corner(dimension)

    def define(problem):
        problem.add_variable("u")
        problem.add_kernel("diffusion", variable="u")
        problem.add_boundary_condition(
            "Dirichlet",
            mesh.sideset_names(),
            variable="u",
            value=lambda x, y, z, t: 1.0 + 2.0 * x - 3.0 * y + z + x * y,
        )

    serial = dualmesh.Problem(mesh, method="fem", distributed=False)
    define(serial)
    serial.solve(report="none")
    problem = dualmesh.Problem(mesh, method="fem", distributed=True, partitioner=partitioner)
    define(problem)
    problem.solve(report="none")
    for side in mesh.sideset_names():
        assert problem.boundary_flux("u", side) == pytest.approx(
            serial.boundary_flux("u", side), abs=1e-10
        )


@pytest.mark.parametrize("mesh_kind", ["uniform", "hanging-2", "hanging-3"])
@pytest.mark.parametrize("motion", ["accelerating", "moving"])
@pytest.mark.parametrize("method", ["fem", "dmcdm"])
def test_a_distributed_dynamic_run_is_exact_for_a_field_linear_in_space_and_quadratic_in_time(
    method, motion, mesh_kind
):
    """The wave equation u_tt - lap u = f with the solution u = 1 + 2x - 3y + z + c t^2, which
    needs f = 2c and starts at rest, or u = 1 + 2x - 3y + z + t, which needs f = 0 and starts
    with the velocity 1, prescribed on the whole boundary.  It is linear in space, so the
    elements hold it exactly, hanging nodes or not, and its acceleration is constant, so
    generalized-alpha integrates it exactly: every node takes the exact value and velocity on
    any number of processes.  The start acceleration comes from the prescribed values at the
    start of the first step, from the inertia of the prescribed nodes, which every process adds
    its part of, and from the equations of the hanging nodes, which go to their primaries before
    the processes add them."""
    c = 0.75
    accelerating = motion == "accelerating"

    def exact(x, y, z, t):
        return 1.0 + 2.0 * x - 3.0 * y + z + (c * t * t if accelerating else t)

    if mesh_kind == "uniform":
        mesh = dualmesh.generate_rectangle_mesh(0.0, 1.0, 0.0, 1.0, 6, 6, element_type="Quad4")
    else:
        mesh = refined_toward_the_corner(int(mesh_kind[-1]))
    problem = dualmesh.Problem(mesh, method=method, distributed=True, partitioner="simple")
    problem.add_variable("u", initial_condition=lambda x, y, z, t: exact(x, y, z, 0.0))
    problem.add_kernel("diffusion", variable="u")
    problem.add_kernel("second_time_derivative", variable="u")
    if accelerating:
        problem.add_kernel("body_force", variable="u", value=2.0 * c)
    problem.add_boundary_condition("Dirichlet", mesh.sideset_names(), variable="u", value=exact)
    points = numpy.asarray(mesh.points())
    if not accelerating:
        problem.set_gathered_velocities({"u": numpy.ones(len(points))})
    end = 0.3
    problem.solve_transient(end_time=end, time_step=0.1, time_stepper="fixed", report="none")
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    assert numpy.asarray(problem.gathered_values("u")) == pytest.approx(
        exact(x, y, z, end), abs=1e-10
    )
    velocity = 2.0 * c * end if accelerating else 1.0
    assert problem.gathered_velocities()["u"] == pytest.approx(velocity, abs=1e-9)
