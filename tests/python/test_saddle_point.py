# SPDX-License-Identifier: LGPL-2.1-or-later
"""The block preconditioners of a flow, and the split by physics.

The Jacobian of a pressure-velocity flow is a saddle point matrix
[[F, G], [D, C]] (momentum, pressure gradient, divergence, pressure block).
The block upper-triangular preconditioner converges in a number of flexible
GMRES iterations that depends only on how well its approximation of the Schur
complement S = C - D F^{-1} G is: the pressure mass matrix scaled by 1/mu
(``preconditioner="pressure_mass"``, exact in the Stokes limit), the pressure
convection-diffusion operator (``"pcd"``) or the least squares commutator
(``"lsc"``), see SaddlePointSolver.h.  The tests check

* that the iteration count does not grow with the mesh for the Taylor-Hood
  element and for the stabilised equal-order element, for every
  approximation, and stays moderate at Re = 100;
* that pcd keeps its count on a channel with an inflow and an outflow, where
  the Robin condition of F_p acts;
* that the answer is the one of the direct solver, also in a transient solve;
* that PETSc's field split with exact block solves takes the iterations of the
  built-in solver, so the two implementations agree;
* that the split by physics solves a thermoelastic plate;
* that misuse is refused with a message that says what to do.
"""

from __future__ import annotations

import dualmesh as dm
import numpy as np
import pytest


def _cavity(formulation, element_type, n, density=0.0, method="fem"):
    mesh = dm.generate_rectangle_mesh(0, 1, 0, 1, n, n, element_type=element_type)
    problem = dm.Problem(mesh, method=method)
    problem.add_physics(
        "fluid_dynamics",
        density=density,
        formulation=formulation,
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
    return problem


def _relative_difference(problem, reference):
    return max(
        np.abs(problem.values(v) - reference.values(v)).max()
        / max(np.abs(reference.values(v)).max(), 1e-300)
        for v in ("velocity_x", "velocity_y", "pressure")
    )


# Iterations per Newton step at a relative tolerance of 1e-12 on 8 x 8 to 32 x 32 meshes (the
# Taylor-Hood meshes have half as many elements per side), with some margin.  Measured: 15-16
# (pressure_mass, Taylor-Hood), 16-19 (pressure_mass, equal order), 24-30 (pcd, Taylor-Hood),
# 20-28 (pcd, equal order), 10-13 (lsc, Taylor-Hood) and 26-30 (lsc, equal order).  pcd on the
# equal-order element grows slowly, by about a third over two refinements.
@pytest.mark.parametrize(
    "formulation, element_type, preconditioner, limit",
    [
        ("Taylor_Hood", "Quad9", "pressure_mass", 20),
        ("pressure", "Quad4", "pressure_mass", 22),
        ("Taylor_Hood", "Quad9", "pcd", 34),
        ("pressure", "Quad4", "pcd", 34),
        ("Taylor_Hood", "Quad9", "lsc", 16),
        ("pressure", "Quad4", "lsc", 34),
    ],
)
def test_iterations_do_not_grow_with_the_mesh(formulation, element_type, preconditioner, limit):
    counts = []
    for n in (8, 16, 32):
        reference = _cavity(formulation, element_type, n)
        reference.solve(linear_solver="lu")
        problem = _cavity(formulation, element_type, n)
        result = problem.solve(
            linear_solver="gmres", preconditioner=preconditioner, linear_tolerance=1e-12
        )
        counts.append(result.linear_iterations / result.total_iterations)
        assert _relative_difference(problem, reference) < 1e-8
    assert max(counts) <= limit, counts
    assert max(counts) <= 1.5 * min(counts), counts


@pytest.mark.parametrize("preconditioner, limit", [("pressure_mass", 60), ("pcd", 60), ("lsc", 25)])
def test_navier_stokes_at_a_moderate_reynolds_number(preconditioner, limit):
    """At Re = 100 the Schur complement carries convection, which the mass
    matrix ignores and pcd and lsc approximate; measured 36, 43 and 16
    iterations per Newton step."""
    reference = _cavity("Taylor_Hood", "Quad9", 16, density=100.0)
    reference.solve(linear_solver="lu")
    problem = _cavity("Taylor_Hood", "Quad9", 16, density=100.0)
    result = problem.solve(linear_solver="gmres", preconditioner=preconditioner)
    assert _relative_difference(problem, reference) < 1e-8
    assert result.linear_iterations / result.total_iterations < limit


def _channel(formulation, element_type, n, density):
    mesh = dm.generate_rectangle_mesh(
        x=[0, 4], y=[0, 1], num_elements=[4 * n, n], element_type=element_type
    )
    problem = dm.Problem(mesh, method="fem")
    flow = problem.add_physics(
        "fluid_dynamics", density=density, dynamic_viscosity=1.0, formulation=formulation
    )
    flow.add_boundary_condition(["velocity_x = 1", "velocity_y = 0"], boundary="left")
    flow.add_boundary_condition(["velocity_x = 0", "velocity_y = 0"], boundary=["top", "bottom"])
    return problem


@pytest.mark.parametrize(
    "formulation, element_type, limit", [("Taylor_Hood", "Quad9", 42), ("pressure", "Quad4", 52)]
)
def test_pcd_with_an_inflow_and_an_outflow(formulation, element_type, limit):
    """A channel at Re = 100 with a uniform inflow and a free outflow: F_p takes the Robin
    condition at the inflow, and A_p, taken from the matrix, its condition at the outflow.
    Measured 33 and 33 (Taylor-Hood) and 42 and 29 (equal order) iterations per Newton step at a
    relative tolerance of 1e-8 on the two meshes, the Stokes start included: the count does not
    grow with the mesh."""
    counts = []
    for n in (8, 16):
        n_elements = n // 2 if element_type == "Quad9" else n
        reference = _channel(formulation, element_type, n_elements, 100.0)
        reference.solve(linear_solver="lu", report="none")
        problem = _channel(formulation, element_type, n_elements, 100.0)
        result = problem.solve(
            linear_solver="gmres", preconditioner="pcd", linear_tolerance=1e-8, report="none"
        )
        counts.append(result.linear_iterations / result.total_iterations)
        assert _relative_difference(problem, reference) < 1e-6
    assert max(counts) <= limit, counts
    assert counts[1] <= counts[0] + 1, counts


def _pipe(radius):
    """Developing flow in a pipe of the given radius and four radii long, at Re = 50 whatever the
    radius: rho = 50, U = 1, mu = radius."""
    mesh = dm.generate_rectangle_mesh(x=[0, radius], y=[0, 4 * radius], num_elements=[8, 32])
    problem = dm.Problem(mesh, method="fem", coordinates="axisymmetric")
    flow = problem.add_physics(
        "fluid_dynamics", density=50.0, dynamic_viscosity=radius, formulation="pressure"
    )
    flow.add_boundary_condition(["velocity_r = 0", "velocity_z = 1"], boundary="bottom")
    flow.add_boundary_condition(["velocity_r = 0", "velocity_z = 0"], boundary="right")
    flow.add_boundary_condition("velocity_r = 0", boundary="left")
    return problem


@pytest.mark.parametrize("preconditioner", ["pcd", "lsc"])
def test_the_iterations_do_not_depend_on_the_unit_of_length(preconditioner):
    """The same axisymmetric pipe flow in metres and in millimetres.  The element size of the
    stabilisation is that of the element itself, without the factor 2 pi r of the coordinate
    system, so the counts are the same (measured 45 for pcd and 54 for lsc)."""
    counts = []
    for radius in (1.0, 1e-3):
        reference = _pipe(radius)
        reference.solve(linear_solver="lu", report="none")
        problem = _pipe(radius)
        result = problem.solve(
            linear_solver="gmres",
            preconditioner=preconditioner,
            linear_tolerance=1e-10,
            report="none",
        )
        counts.append(result.linear_iterations / result.total_iterations)
        expected = reference.values("velocity_z")
        assert np.abs(problem.values("velocity_z") - expected).max() < 1e-8
    assert abs(counts[0] - counts[1]) <= 1, counts
    assert max(counts) <= 65, counts


@pytest.mark.parametrize(
    "preconditioner, implicitness, limit",
    [("pcd", 1.0, 34), ("pcd", 0.5, 34), ("lsc", 1.0, 40), ("lsc", 0.5, 40)],
)
def test_a_transient_solve_takes_the_time_derivative_into_the_operators(
    preconditioner, implicitness, limit
):
    """F_p and the stabilisation of lsc take rho/dt and the weight theta of the steady terms of
    the step.  Measured 28 and 30 (pcd) and 33 and 33 (lsc) iterations per Newton step for
    theta = 1 and 0.5 (24 October).  The mass equation and the pressure gradient are taken at the
    new time with the weight 1 for every theta, so their blocks scale as at theta = 1 (before,
    with them weighted by theta, lsc took 23 at theta = 0.5)."""
    reference = _cavity("pressure", "Quad4", 12, density=100.0)
    reference.solve_transient(
        end_time=0.05,
        time_step=0.01,
        implicitness=implicitness,
        linear_solver="lu",
        report="none",
    )
    problem = _cavity("pressure", "Quad4", 12, density=100.0)
    result = problem.solve_transient(
        end_time=0.05,
        time_step=0.01,
        implicitness=implicitness,
        linear_solver="gmres",
        preconditioner=preconditioner,
        linear_tolerance=1e-12,
        report="none",
    )
    assert _relative_difference(problem, reference) < 1e-7
    assert result.linear_iterations / result.total_iterations <= limit


@pytest.mark.skipif(not dm.have_petsc(), reason="built without PETSc")
@pytest.mark.parametrize("options", [None, "-fieldsplit_0_pc_type gamg"])
def test_petsc_field_split_with_the_pressure_mass_matrix(options):
    reference = _cavity("Taylor_Hood", "Quad9", 12)
    reference.solve(linear_solver="lu")
    problem = _cavity("Taylor_Hood", "Quad9", 12)
    problem.solve(
        linear_solver="petsc",
        preconditioner="pressure_mass",
        petsc_options=options,
        linear_tolerance=1e-12,
    )
    assert _relative_difference(problem, reference) < 1e-7


@pytest.mark.skipif(not dm.have_petsc(), reason="built without PETSc")
@pytest.mark.parametrize("preconditioner", ["pcd", "lsc"])
@pytest.mark.parametrize(
    "formulation, element_type, n", [("Taylor_Hood", "Quad9", 8), ("pressure", "Quad4", 16)]
)
def test_petsc_with_exact_blocks_takes_the_iterations_of_the_built_in_solver(
    preconditioner, formulation, element_type, n
):
    # Both start from the same Stokes solution, so that the counts are those of the Newton steps
    # of the flow (the Stokes start of a flow at rest takes pressure_mass, whose PETSc blocks the
    # options below do not make exact).
    stokes = _cavity(formulation, element_type, n, density=0.0)
    stokes.solve(linear_solver="lu", report="none")

    def started(problem):
        problem.initialize()
        for name in ("velocity_x", "velocity_y", "pressure"):
            problem.set_values(name, stokes.values(name))
        return problem

    built_in = started(_cavity(formulation, element_type, n, density=100.0))
    expected = built_in.solve(
        linear_solver="gmres", preconditioner=preconditioner, linear_tolerance=1e-8
    )
    problem = started(_cavity(formulation, element_type, n, density=100.0))
    exact = {
        "-fieldsplit_0_pc_type": "lu",
        "-fieldsplit_0_pc_factor_mat_solver_type": "mumps",
        "-fieldsplit_1_laplacian_pc_type": "lu",
        "-fieldsplit_1_laplacian_pc_factor_mat_solver_type": "mumps",
    }
    result = problem.solve(
        linear_solver="petsc",
        preconditioner=preconditioner,
        petsc_options=exact,
        linear_tolerance=1e-8,
    )
    assert abs(result.linear_iterations - expected.linear_iterations) <= result.total_iterations
    assert _relative_difference(problem, built_in) < 1e-7


@pytest.mark.skipif(not dm.have_petsc(), reason="built without PETSc")
@pytest.mark.parametrize("preconditioner", ["pcd", "lsc"])
def test_petsc_with_algebraic_multigrid_blocks(preconditioner):
    reference = _cavity("pressure", "Quad4", 16, density=100.0)
    reference.solve(linear_solver="lu")
    problem = _cavity("pressure", "Quad4", 16, density=100.0)
    problem.solve(linear_solver="petsc", preconditioner=preconditioner, linear_tolerance=1e-12)
    assert _relative_difference(problem, reference) < 1e-7


@pytest.mark.parametrize("method", ["dmcdm", "hfvm"])
@pytest.mark.parametrize("preconditioner", ["pressure_mass", "pcd", "lsc"])
def test_it_works_with_the_control_volume_methods(method, preconditioner):
    reference = _cavity("pressure", "Quad4", 12, method=method)
    reference.solve(linear_solver="lu")
    problem = _cavity("pressure", "Quad4", 12, method=method)
    problem.solve(linear_solver="gmres", preconditioner=preconditioner)
    assert _relative_difference(problem, reference) < 1e-8


def _thermoelastic_plate():
    mesh = dm.generate_rectangle_mesh(x=[0.0, 0.1], y=[0.0, 0.02], num_elements=[40, 8])
    problem = dm.Problem(mesh)
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
    return problem


@pytest.mark.parametrize(
    "linear_solver",
    [
        "gmres",
        pytest.param("petsc", marks=pytest.mark.skipif(not dm.have_petsc(), reason="no PETSc")),
    ],
)
def test_the_split_by_physics_solves_a_thermoelastic_plate(linear_solver):
    reference = _thermoelastic_plate()
    reference.solve(linear_solver="lu", report="none")
    problem = _thermoelastic_plate()
    result = problem.solve(
        linear_solver=linear_solver,
        preconditioner="physics_blocks",
        linear_tolerance=1e-12,
        report="none",
    )
    for variable in ("temperature", "disp_x", "disp_y"):
        expected = reference.values(variable)
        assert np.abs(problem.values(variable) - expected).max() <= 1e-8 * np.abs(expected).max()
    assert result.linear_iterations < 60


def test_the_split_by_physics_refuses_a_flow():
    problem = _cavity("pressure", "Quad4", 4)
    with pytest.raises(ValueError, match="'LSC' or 'PCD'"):
        problem.solve(linear_solver="gmres", preconditioner="physics_blocks")


def test_an_unknown_preconditioner_is_named_with_the_closest_one():
    problem = _cavity("Taylor_Hood", "Quad9", 4)
    with pytest.raises(ValueError, match="Did you mean 'pressure_mass'"):
        problem.solve(preconditioner="pressure_mass_schur")


def test_it_needs_a_krylov_solver():
    problem = _cavity("Taylor_Hood", "Quad9", 4)
    with pytest.raises(ValueError, match="'GMRES' or 'automatic'"):
        problem.solve(linear_solver="bicgstab", preconditioner="pressure_mass")


def test_it_needs_a_mass_conservation_equation():
    mesh = dm.generate_rectangle_mesh(0, 1, 0, 1, 4, 4)
    problem = dm.Problem(mesh, method="fem")
    problem.add_variable("temperature")
    problem.add_kernel("diffusion", variable="temperature")
    problem.add_kernel("body_force", variable="temperature", value=1.0)
    problem.add_boundary_condition("Dirichlet", "left", variable="temperature", value=0.0)
    with pytest.raises(ValueError, match="mass_conservation"):
        problem.solve(linear_solver="gmres", preconditioner="pressure_mass")


def test_the_adjoint_refuses_pcd():
    problem = dm.Problem(dm.generate_rectangle_mesh(0, 1, 0, 1, 4, 4), method="fem")
    problem.add_parameter("k", 1.0)
    problem.add_variable("temperature")
    problem.add_kernel("heat_conduction", variable="temperature", thermal_conductivity="k")
    problem.add_kernel("body_force", variable="temperature", value=1.0)
    problem.add_boundary_condition("Dirichlet", "left", variable="temperature", value=0.0)
    problem.solve(report="none")
    with pytest.raises(ValueError, match="'LSC' or 'pressure_mass'"):
        problem.sensitivities("temperature", ["k"], preconditioner="pcd")


def test_the_automatic_solver_splits_flows_only():
    """A problem on a quadratic mesh with a first-order variable and no flow, larger than the
    direct solver takes in three dimensions (6750 unknowns): the automatic choice iterates with
    multigrid and finds the answer of the direct solver.  A flow preconditioner would refuse it,
    since it has no mass balance."""

    def define():
        mesh = dm.generate_box_mesh(
            x=[0, 1], y=[0, 1], z=[0, 1], num_elements=[7, 7, 7], element_type="Hex27"
        )
        problem = dm.Problem(mesh, method="fem")
        problem.add_variable("u")
        problem.add_variable("c", order="first")
        for variable, source in (("u", 1.0), ("c", 2.0)):
            problem.add_kernel("diffusion", variable=variable)
            problem.add_kernel("body_force", variable=variable, value=source)
            problem.add_boundary_condition(
                "Dirichlet", ["left", "right"], variable=variable, value=0.0
            )
        return problem

    reference = define()
    reference.solve(linear_solver="lu", report="none")
    problem = define()
    problem.solve(report="none")
    for variable in ("u", "c"):
        expected = reference.values(variable)
        assert np.abs(problem.values(variable) - expected).max() <= 1e-8 * np.abs(expected).max()


def test_the_split_by_physics_is_exact_for_a_one_way_coupling_with_exact_block_solves():
    """The temperature drives the displacements and not the other way, so the block lower
    triangle with the temperature first is the matrix itself.  On a mesh small enough that
    multigrid has one level, which it solves exactly, flexible GMRES converges in one
    iteration; with the blocks in the other order it takes two."""
    mesh = dm.generate_rectangle_mesh(x=[0.0, 0.1], y=[0.0, 0.02], num_elements=[10, 2])
    problem = dm.Problem(mesh)
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
    result = problem.solve(
        linear_solver="gmres",
        preconditioner="physics_blocks",
        linear_tolerance=1e-12,
        report="none",
    )
    assert result.linear_iterations == 1


def _natural_convection():
    mesh = dm.generate_rectangle_mesh(x=[0, 1], y=[0, 1], num_elements=[16, 16])
    problem = dm.Problem(mesh, method="fem")
    flow = problem.add_physics(
        "fluid_dynamics",
        dynamic_viscosity=0.71,
        density=1.0,
        formulation="pressure",
        pressure_pin_point=(0.5, 0.0),
    )
    heat = problem.add_physics("heat_transfer", thermal_conductivity=1.0)
    problem.add_coupling(
        "nonisothermal_flow",
        heat_transfer=heat,
        fluid_dynamics=flow,
        gravity=[0.0, -1.0],
        vol_expansion_coeff=1.0e4 * 0.71,
    )
    flow.add_boundary_condition(
        ["velocity_x = 0", "velocity_y = 0"], boundary=["left", "right", "bottom", "top"]
    )
    heat.add_boundary_condition("temperature = 0.5", boundary="left")
    heat.add_boundary_condition("temperature = -0.5", boundary="right")
    return problem


@pytest.mark.parametrize("preconditioner", ["lsc", "pcd"])
def test_natural_convection_takes_the_temperature_into_the_first_block(preconditioner):
    """The de Vahl Davis cavity at Ra = 10^4: the temperature joins the velocities in the first
    block of the split, and the coupling of the buoyancy and the convection of heat stays in it.
    Measured 41 (lsc) and 47 (pcd) iterations per Newton step."""
    reference = _natural_convection()
    reference.solve(linear_solver="lu", report="none")
    problem = _natural_convection()
    result = problem.solve(
        linear_solver="gmres",
        preconditioner=preconditioner,
        linear_tolerance=1e-12,
        report="none",
    )
    for variable in ("velocity_x", "velocity_y", "temperature"):
        expected = reference.values(variable)
        assert np.abs(problem.values(variable) - expected).max() <= 1e-8 * np.abs(expected).max()
    assert (
        result.linear_iterations / result.total_iterations <= {"lsc": 50, "pcd": 58}[preconditioner]
    )


def _channel_to_a_prescribed_pressure(density):
    mesh = dm.generate_rectangle_mesh(x=[0, 4], y=[0, 1], num_elements=[32, 8])
    problem = dm.Problem(mesh, method="fem")
    flow = problem.add_physics(
        "fluid_dynamics", density=density, dynamic_viscosity=1.0, formulation="pressure"
    )
    flow.add_boundary_condition(["velocity_x = 1", "velocity_y = 0"], boundary="left")
    flow.add_boundary_condition(["velocity_x = 0", "velocity_y = 0"], boundary=["top", "bottom"])
    flow.add_boundary_condition("pressure = 0", boundary="right")
    return problem


# pressure_mass ignores convection: on this channel it takes 21 iterations per Newton step at
# Re = 1 and 29 at Re = 10, and at Re = 100 it stagnates with the default restart of GMRES (176
# iterations with a restart of 400; measured 24 October, the Stokes start included).  It is tested in its range, Stokes flow; the others at
# Re = 100.
@pytest.mark.parametrize(
    "preconditioner, density", [("lsc", 100.0), ("pcd", 100.0), ("pressure_mass", 1.0)]
)
def test_a_prescribed_pressure_at_the_outlet(preconditioner, density):
    """The channel with p = 0 prescribed on the outlet: the outlet pressures are rows of the
    identity, which the Schur approximation keeps as such."""
    reference = _channel_to_a_prescribed_pressure(density)
    reference.solve(linear_solver="lu", report="none")
    problem = _channel_to_a_prescribed_pressure(density)
    problem.solve(
        linear_solver="gmres", preconditioner=preconditioner, linear_tolerance=1e-12, report="none"
    )
    assert _relative_difference(problem, reference) < 1e-7


def test_the_automatic_solver_falls_back_to_the_direct_solver():
    """A three-dimensional cavity of 8788 unknowns, more than the direct solver takes in three
    dimensions, so the automatic choice is the least squares commutator.  With one iteration
    allowed it cannot converge, and the direct solver gives the answer."""

    def define():
        mesh = dm.generate_box_mesh(x=[0, 1], y=[0, 1], z=[0, 1], num_elements=[12, 12, 12])
        problem = dm.Problem(mesh, method="fem")
        flow = problem.add_physics(
            "fluid_dynamics",
            density=10.0,
            dynamic_viscosity=1.0,
            formulation="pressure",
            pressure_pin_point=(0.5, 0.5, 0.0),
        )
        flow.add_boundary_condition(
            ["velocity_x = 1", "velocity_y = 0", "velocity_z = 0"], boundary="top"
        )
        flow.add_boundary_condition(
            ["velocity_x = 0", "velocity_y = 0", "velocity_z = 0"],
            boundary=["left", "right", "bottom", "front", "back"],
        )
        return problem

    reference = define()
    reference.solve(linear_solver="lu", report="none")
    problem = define()
    problem.solve(linear_max_iterations=1, report="none")
    assert _relative_difference(problem, reference) < 1e-8
