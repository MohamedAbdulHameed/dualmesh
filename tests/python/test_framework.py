# SPDX-License-Identifier: LGPL-2.1-or-later
"""Tests of the framework itself: meshes, parameters, solvers, examples.

These complement the verification suite (which checks published results) by
exercising the machinery: mesh generation and file input/output, parameter
validation and its error messages, blocks, point sources, transient
integration, three-dimensional elements, spherical coordinates, objects
written in Python, the examples and the command-line tools.
"""

from __future__ import annotations

import math
from pathlib import Path

import dualmesh as dm
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# meshes
# ---------------------------------------------------------------------------
def test_generators_produce_the_expected_sizes_and_boundaries():
    line = dm.generate_line_mesh(start=0.0, end=1.0, num_elements=5)
    assert (line.num_nodes, line.num_elements) == (6, 5)
    assert set(line.sideset_names()) >= {"left", "right"}

    rectangle = dm.generate_rectangle_mesh(
        x_min=0.0, x_max=2.0, y_min=0.0, y_max=1.0, num_x_elements=4, num_y_elements=2
    )
    assert (rectangle.num_nodes, rectangle.num_elements) == (15, 8)
    assert set(rectangle.sideset_names()) >= {"left", "right", "bottom", "top"}

    triangles = dm.generate_rectangle_mesh(
        x_min=0.0,
        x_max=1.0,
        y_min=0.0,
        y_max=1.0,
        num_x_elements=3,
        num_y_elements=3,
        element_type="Tri3",
    )
    assert triangles.num_elements == 18

    box = dm.generate_box_mesh(
        x_min=0.0,
        x_max=1.0,
        y_min=0.0,
        y_max=1.0,
        z_min=0.0,
        z_max=1.0,
        num_x_elements=2,
        num_y_elements=2,
        num_z_elements=2,
    )
    assert (box.num_nodes, box.num_elements) == (27, 8)
    assert set(box.sideset_names()) >= {"left", "right", "bottom", "top", "back", "front"}

    tetrahedra = dm.generate_box_mesh(
        x_min=0.0,
        x_max=1.0,
        y_min=0.0,
        y_max=1.0,
        z_min=0.0,
        z_max=1.0,
        num_x_elements=2,
        num_y_elements=2,
        num_z_elements=2,
        element_type="Tet4",
    )
    assert tetrahedra.num_elements == 8 * 6


def test_graded_coordinates_and_spacings():
    graded = dm.graded_coordinates(0.0, 1.0, 4, bias=2.0)
    sizes = np.diff(graded)
    assert graded[0] == 0.0 and graded[-1] == pytest.approx(1.0)
    assert np.allclose(sizes[1:] / sizes[:-1], 2.0)

    coordinates = dm.meshing.coordinates_from_spacings(1.0, [0.5, 0.25, 0.25])
    assert coordinates == pytest.approx([1.0, 1.5, 1.75, 2.0])


def test_annulus_mesh_geometry():
    mesh = dm.generate_annulus_mesh(
        inner_radius=1.0, outer_radius=2.0, num_radial_elements=4, num_angular_elements=8
    )
    points = np.asarray(mesh.points())
    radius = np.hypot(points[:, 0], points[:, 1])
    assert radius.min() == pytest.approx(1.0)
    assert radius.max() == pytest.approx(2.0)
    assert set(mesh.sideset_names()) >= {"inner", "outer", "start", "end"}
    # a quarter annulus: the area is pi (b^2 - a^2) / 4, approached from below
    # by the straight-sided mesh
    assert len(mesh.boundary_nodes("inner")) == 9


def test_uniform_refinement_keeps_boundaries():
    mesh = dm.generate_rectangle_mesh(
        x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0, num_x_elements=2, num_y_elements=2
    )
    refined = mesh.refined()
    assert refined.num_elements == 4 * mesh.num_elements
    assert len(refined.boundary_nodes("left")) == 5
    # and the refined mesh still solves the same problem
    for candidate in (mesh, refined):
        problem = dm.Problem(candidate, method="DMCDM")
        problem.add_variable("u")
        problem.add_kernel("diffusion", variable="u")
        problem.add_boundary_condition("Dirichlet", "left", variable="u", value=1.0)
        problem.add_boundary_condition("Dirichlet", "right", variable="u", value=0.0)
        problem.solve()
        middle = problem.sample("u", [[0.5, 0.5]])[0]
        assert middle == pytest.approx(0.5, abs=1e-12)


def test_mesh_file_round_trip(tmp_path):
    meshio = pytest.importorskip("meshio")
    mesh = dm.generate_rectangle_mesh(
        x_min=0.0, x_max=1.0, y_min=0.0, y_max=0.5, num_x_elements=4, num_y_elements=2
    )
    path = tmp_path / "plate.vtu"
    dm.write_mesh(mesh, str(path))
    assert path.exists()
    read_back = dm.read_mesh(str(path))
    assert read_back.num_nodes == mesh.num_nodes
    assert read_back.num_elements == mesh.num_elements
    assert np.allclose(np.asarray(read_back.points()), np.asarray(mesh.points()))
    del meshio


def test_mesh_from_arrays_and_predicates():
    points = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [2.0, 0.0], [2.0, 1.0]])
    cells = np.array([[0, 1, 2, 3], [1, 4, 5, 2]])
    mesh = dm.mesh_from_arrays(points, cells, element_type="Quad4", blocks=[0, 1])
    assert mesh.num_elements == 2
    assert mesh.block_ids() == [0, 1]
    mesh.add_sideset_by_predicate("far", lambda x, y, z: x > 1.999)
    assert len(mesh.boundary_nodes("far")) == 2


# ---------------------------------------------------------------------------
# parameters and error messages
# ---------------------------------------------------------------------------
def test_unknown_parameter_is_reported_with_the_accepted_names():
    mesh = dm.generate_line_mesh(start=0.0, end=1.0, num_elements=2)
    problem = dm.Problem(mesh, method="DMCDM")
    problem.add_variable("u")
    with pytest.raises(ValueError, match="Unknown parameter 'conductivity'"):
        problem.add_kernel("diffusion", variable="u", conductivity=1.0)


def test_missing_required_parameter_is_reported():
    mesh = dm.generate_line_mesh(start=0.0, end=1.0, num_elements=2)
    problem = dm.Problem(mesh, method="DMCDM")
    problem.add_variable("u")
    with pytest.raises(ValueError, match="Missing required parameter 'variable'"):
        problem.add_kernel("diffusion")


def test_unknown_variable_and_boundary_are_reported():
    mesh = dm.generate_line_mesh(start=0.0, end=1.0, num_elements=2)
    problem = dm.Problem(mesh, method="DMCDM")
    problem.add_variable("u")
    problem.add_kernel("diffusion", variable="u")
    with pytest.raises(ValueError, match="the mesh has no boundary 'norht'"):
        problem.add_boundary_condition("Dirichlet", "norht", variable="u", value=0.0)


def test_registry_is_self_documenting():
    assert "heat_conduction" in dm.registered_types()
    assert dm.object_category("heat_conduction") == "kernel"
    assert dm.object_module("heat_conduction") == "heat_transfer"
    description = dm.describe("Robin")
    assert "transfer_coeff" in description
    assert "boundary" in description
    assert set(dm.list_objects(category="property")) >= {
        "constant",
        "small_strain_stress",
    }


# ---------------------------------------------------------------------------
# physics of the framework objects
# ---------------------------------------------------------------------------
def test_blocks_allow_different_materials_in_one_mesh():
    """Two-layer wall: the flux is continuous and the interface value is known."""
    left_conductivity, right_conductivity = 1.0, 3.0
    mesh = dm.generate_line_mesh(coordinates=np.linspace(0.0, 2.0, 21))
    # block 0 for x < 1, block 1 for x > 1
    layered = dm.Mesh(1)
    for point in np.asarray(mesh.points()):
        layered.add_node([float(point[0]), 0.0, 0.0])
    for element in range(mesh.num_elements):
        nodes = mesh.element_nodes(element)
        centre = 0.5 * sum(float(np.asarray(mesh.points())[n][0]) for n in nodes)
        layered.add_element("Edge2", list(nodes), 0 if centre < 1.0 else 1)
    layered.add_bounding_box_sidesets()
    layered.add_nodeset("left", [0])
    layered.add_nodeset("right", [layered.num_nodes - 1])

    problem = dm.Problem(layered, method="DMCDM")
    problem.add_variable("temperature")
    problem.add_kernel(
        "heat_conduction",
        variable="temperature",
        thermal_conductivity=left_conductivity,
        block=["0"],
    )
    problem.add_kernel(
        "heat_conduction",
        variable="temperature",
        thermal_conductivity=right_conductivity,
        block=["1"],
    )
    problem.add_boundary_condition("Dirichlet", "left", variable="temperature", value=100.0)
    problem.add_boundary_condition("Dirichlet", "right", variable="temperature", value=0.0)
    problem.solve()

    # series resistance: T_interface = 100 * R_right / (R_left + R_right)
    resistance_left, resistance_right = 1.0 / left_conductivity, 1.0 / right_conductivity
    expected = 100.0 * resistance_right / (resistance_left + resistance_right)
    assert problem.sample("temperature", [[1.0]])[0] == pytest.approx(expected, rel=1e-12)


def test_point_source_and_reaction_balance():
    """A point heat source in a bar leaves through the two fixed ends."""
    mesh = dm.generate_line_mesh(start=0.0, end=1.0, num_elements=10)
    problem = dm.Problem(mesh, method="DMCDM")
    problem.add_variable("temperature")
    problem.add_kernel("diffusion", variable="temperature")
    problem.add_point_source("point_source", variable="temperature", value=2.0, points=[0.5])
    problem.add_boundary_condition(
        "Dirichlet",
        ["left", "right"],
        variable="temperature",
        value=0.0,
    )
    problem.solve()
    # symmetric: half the source leaves through each end
    assert problem.total_reaction("temperature", "left") == pytest.approx(-1.0, rel=1e-12)
    assert problem.total_reaction("temperature", "right") == pytest.approx(-1.0, rel=1e-12)
    # peak value: q L / 4 for a unit conductivity
    assert problem.sample("temperature", [[0.5]])[0] == pytest.approx(0.5, rel=1e-12)


def test_radiative_boundary_condition_is_nonlinear_and_converges():
    mesh = dm.generate_line_mesh(start=0.0, end=0.1, num_elements=10)
    problem = dm.Problem(mesh, method="DMCDM")
    problem.add_variable("temperature", initial_condition=1000.0)
    problem.add_kernel("heat_conduction", variable="temperature", thermal_conductivity=20.0)
    problem.add_boundary_condition("Dirichlet", "left", variable="temperature", value=1000.0)
    problem.add_boundary_condition(
        "radiative_heat_flux",
        "right",
        variable="temperature",
        emissivity=0.8,
        ambient_temperature=300.0,
    )
    result = problem.solve()
    assert result.converged
    temperature = problem.values("temperature")
    assert 300.0 < temperature[-1] < 1000.0
    # the heat conducted to the surface must equal the heat radiated away
    stefan_boltzmann = 5.670374419e-8
    radiated = 0.8 * stefan_boltzmann * (temperature[-1] ** 4 - 300.0**4)
    conducted = 20.0 * (temperature[-2] - temperature[-1]) / 0.01
    assert conducted == pytest.approx(radiated, rel=1e-3)


def test_spherical_coordinates_reproduce_the_analytical_solution():
    """Conduction in a sphere with uniform generation: T = T0 + g (R^2 - r^2) / (6k)."""
    radius, conductivity, generation, surface = 0.2, 5.0, 1.0e5, 50.0
    mesh = dm.generate_line_mesh(start=0.0, end=radius, num_elements=20)
    problem = dm.Problem(mesh, method="DMCDM", coordinates="spherical")
    problem.add_variable("temperature")
    problem.add_kernel("heat_conduction", variable="temperature", thermal_conductivity=conductivity)
    problem.add_kernel("body_force", variable="temperature", value=generation)
    problem.add_boundary_condition(
        "Dirichlet",
        "right",
        variable="temperature",
        value=surface,
    )
    problem.solve()
    r = np.linspace(0.0, radius, 21)
    exact = surface + generation * (radius**2 - r**2) / (6 * conductivity)
    assert problem.values("temperature") == pytest.approx(exact, rel=1e-3)
    # the heat leaving the surface is the heat generated in the sphere
    generated = generation * 4.0 / 3.0 * math.pi * radius**3
    assert problem.total_reaction("temperature", "right") == pytest.approx(-generated, rel=1e-12)


def test_three_dimensional_conduction_and_elasticity():
    mesh = dm.generate_box_mesh(
        x_min=0.0,
        x_max=1.0,
        y_min=0.0,
        y_max=0.5,
        z_min=0.0,
        z_max=0.5,
        num_x_elements=6,
        num_y_elements=3,
        num_z_elements=3,
    )
    conduction = dm.Problem(mesh, method="DMCDM")
    conduction.add_variable("temperature")
    conduction.add_kernel("heat_conduction", variable="temperature", thermal_conductivity=2.0)
    conduction.add_boundary_condition("Dirichlet", "left", variable="temperature", value=100.0)
    conduction.add_boundary_condition("Dirichlet", "right", variable="temperature", value=0.0)
    conduction.solve()
    # the exact solution is linear in x, which the method reproduces exactly
    points = np.asarray(mesh.points())
    assert conduction.values("temperature") == pytest.approx(100.0 * (1.0 - points[:, 0]), abs=1e-9)
    # and the heat flow is k A dT/dx = 2 * 0.25 * 100 / 1
    assert conduction.total_reaction("temperature", "left") == pytest.approx(50.0, rel=1e-9)

    bar = dm.Problem(mesh, method="DMCDM")
    bar.add_physics(
        "solid_mechanics",
        youngs_modulus=1000.0,
        poissons_ratio=0.3,
        formulation="three_dimensional",
    )
    bar.add_boundary_condition("Dirichlet", "left", variable="disp_x", value=0.0)
    bar.add_boundary_condition("Dirichlet", "bottom", variable="disp_y", value=0.0)
    bar.add_boundary_condition("Dirichlet", "back", variable="disp_z", value=0.0)
    bar.add_boundary_condition("traction", "right", variable="disp_x", traction=10.0)
    bar.solve()
    # uniaxial tension: u(L) = sigma L / E, and the lateral strain is -nu times it
    assert bar.values("disp_x").max() == pytest.approx(10.0 * 1.0 / 1000.0, rel=1e-9)
    assert bar.values("disp_y").min() == pytest.approx(-0.3 * 10.0 / 1000.0 * 0.5, rel=1e-9)


def test_transient_conduction_against_the_fourier_series():
    length, diffusivity, end_time = 1.0, 1.0, 0.05
    mesh = dm.generate_line_mesh(start=0.0, end=length, num_elements=40)
    problem = dm.Problem(mesh, method="DMCDM")
    problem.add_variable("temperature", initial_condition=1.0)
    problem.add_kernel("heat_conduction", variable="temperature", thermal_conductivity=diffusivity)
    problem.add_kernel(
        "heat_conduction_time_derivative", variable="temperature", density=1.0, specific_heat=1.0
    )
    problem.add_boundary_condition(
        "Dirichlet",
        ["left", "right"],
        variable="temperature",
        value=0.0,
    )
    problem.solve_transient(end_time=end_time, time_step=0.001, implicitness=0.5)

    x = np.linspace(0.05, 0.95, 10)
    exact = np.zeros_like(x)
    for n in range(1, 400, 2):
        exact += (
            (4.0 / (n * math.pi))
            * np.sin(n * math.pi * x / length)
            * math.exp(-((n * math.pi / length) ** 2) * diffusivity * end_time)
        )
    assert problem.sample("temperature", x.reshape(-1, 1)) == pytest.approx(exact, abs=1e-3)


def test_lumped_and_consistent_time_derivatives_agree_when_refined():
    def solve(quadrature, num_elements):
        mesh = dm.generate_line_mesh(start=0.0, end=1.0, num_elements=num_elements)
        problem = dm.Problem(mesh, method="DMCDM")
        problem.add_variable("temperature", initial_condition=1.0)
        problem.add_kernel("diffusion", variable="temperature")
        problem.add_kernel("time_derivative", variable="temperature", quadrature=quadrature)
        problem.add_boundary_condition(
            "Dirichlet",
            ["left", "right"],
            variable="temperature",
            value=0.0,
        )
        problem.solve_transient(end_time=0.05, time_step=0.001)
        return problem.sample("temperature", [[0.5]])[0]

    coarse = abs(solve("gauss2", 20) - solve("nodal", 20))
    fine = abs(solve("gauss2", 80) - solve("nodal", 80))
    assert fine < coarse


# ---------------------------------------------------------------------------
# objects written in Python
# ---------------------------------------------------------------------------
class ExponentialSource(dm.PythonKernel):
    """S = -a exp(-b u): a nonlinear source written in Python."""

    def compute_source(self, ctx):
        return -self.magnitude * dm.exp(-self.rate * ctx.value(self.variable))


class TemperatureDependentConductivity(dm.PythonProperty):
    """A property computed in Python."""

    def declare_properties(self, registry):
        self.conductivity_id = registry.declare("python_conductivity", 1)

    def setup(self, problem):
        self.temperature = problem.variable_index("temperature")

    def compute_properties(self, ctx):
        value = ctx.coefficient_value(self.temperature)
        ctx.set_property(self.conductivity_id, 20.0 + 0.2 * value)


def test_python_kernel_with_exact_derivatives():
    """The source of -u'' = exp(-2u) written in Python (S = -exp(-2u) in -div F + S = 0, with the
    diffusion flux F = grad u) gives the answer of the same equation written for the C++ parsed
    kernel (whose source is the right-hand side f), and Newton's method with its exact Jacobian
    converges in a few steps."""

    def solve(python):
        mesh = dm.generate_line_mesh(start=0.0, end=1.0, num_elements=10)
        problem = dm.Problem(mesh, method="DMCDM")
        problem.add_variable("u")
        problem.add_kernel("diffusion", variable="u")
        if python:
            problem.add_kernel(ExponentialSource(variable="u", magnitude=1.0, rate=2.0))
        else:
            problem.add_kernel("parsed_kernel", variable="u", source="exp(-2*u)")
        problem.add_boundary_condition("Dirichlet", ["left", "right"], variable="u", value=0.0)
        result = problem.solve(report="none")
        return problem.values("u"), result

    expected, _ = solve(python=False)
    values, result = solve(python=True)
    assert result.total_iterations <= 5
    assert values.max() > 0.1
    assert np.abs(values - expected).max() <= 1e-12 * np.abs(expected).max()


class PythonTimeDerivative(dm.PythonKernel):
    """(u - u_old) / dt, from the state of the last step."""

    def compute_source(self, ctx):
        u = self.variable
        return (ctx.value(u) - ctx.old_value(u)) / ctx.dt


class PythonCrankNicolsonDiffusion(dm.PythonKernel):
    """The flux (grad u + grad u_old) / 2 (the diffusion flux of the library is F = grad u):
    diffusion weighted half on the new state and half on the old one."""

    def compute_flux(self, ctx):
        new, old = ctx.gradient(self.variable), ctx.old_gradient(self.variable)
        return [0.5 * (new[d] + old[d]) for d in range(3)]


def test_a_python_kernel_reads_the_state_of_the_last_step():
    """A Crank-Nicolson step written in Python with old_value and old_gradient, solved as an
    implicit step, is the built-in theta scheme at theta = 1/2."""

    def define(python):
        mesh = dm.generate_line_mesh(start=0.0, end=1.0, num_elements=20)
        problem = dm.Problem(mesh, method="DMCDM")
        problem.add_variable("u", initial_condition=lambda x, y, z, t: np.sin(np.pi * x))
        if python:
            problem.add_kernel(PythonTimeDerivative(variable="u"))
            problem.add_kernel(PythonCrankNicolsonDiffusion(variable="u"))
        else:
            problem.add_kernel("time_derivative", variable="u")
            problem.add_kernel("diffusion", variable="u")
        problem.add_boundary_condition("Dirichlet", ["left", "right"], variable="u", value=0.0)
        return problem

    builtin = define(python=False)
    builtin.solve_transient(end_time=0.1, time_step=0.01, implicitness=0.5, report="none")
    python = define(python=True)
    python.solve_transient(end_time=0.1, time_step=0.01, implicitness=1.0, report="none")
    expected = builtin.values("u")
    assert expected.max() < 0.5  # the field has decayed: the steps were taken
    assert np.abs(python.values("u") - expected).max() <= 1e-12


class RecordingSource(dm.PythonKernel):
    """A zero source that records the linearisation and the boundary flag it is called with."""

    seen: set = set()

    def compute_source(self, ctx):
        RecordingSource.seen.add(("kernel", ctx.is_picard, ctx.on_boundary))
        return 0.0 * ctx.value(self.variable)


class RecordingFlux(dm.PythonBoundaryCondition):
    def compute_boundary_flux(self, ctx):
        RecordingSource.seen.add(("boundary", ctx.is_picard, ctx.on_boundary))
        return 0.0 * ctx.value(self.variable)


@pytest.mark.parametrize("nonlinear_solver, picard", [("newton", False), ("picard", True)])
def test_a_python_object_sees_the_linearisation_and_the_boundary(nonlinear_solver, picard):
    RecordingSource.seen = set()
    problem = dm.Problem(dm.generate_line_mesh(start=0.0, end=1.0, num_elements=4), method="DMCDM")
    problem.add_variable("u")
    problem.add_kernel("diffusion", variable="u")
    problem.add_kernel(RecordingSource(variable="u"))
    problem.add_boundary_condition("Dirichlet", "left", variable="u", value=1.0)
    problem.add_boundary_condition(RecordingFlux(variable="u", boundary=["right"]))
    problem.solve(nonlinear_solver=nonlinear_solver, report="none")
    assert RecordingSource.seen == {("kernel", picard, False), ("boundary", picard, True)}


def test_python_material_reproduces_the_builtin_nonlinearity():
    """k(T) = 20 + 0.2 T, once with a Python material and once with a kernel."""

    def solve(use_python_material):
        mesh = dm.generate_rectangle_mesh(
            x_min=0.0, x_max=0.1, y_min=0.0, y_max=0.05, num_x_elements=10, num_y_elements=5
        )
        problem = dm.Problem(mesh, method="DMCDM")
        problem.add_variable("temperature")
        if use_python_material:
            problem.add_property(TemperatureDependentConductivity())
            problem.add_kernel(
                "heat_conduction",
                variable="temperature",
                thermal_conductivity="python_conductivity",
            )
        else:
            problem.add_kernel(
                "heat_conduction",
                variable="temperature",
                thermal_conductivity=20.0,
                temperature_polynomial=[1.0, 0.01],
            )
        problem.add_kernel("body_force", variable="temperature", value=1.0e6)
        problem.add_boundary_condition(
            "Dirichlet",
            "left",
            variable="temperature",
            value=40.0,
        )
        problem.add_boundary_condition(
            "Dirichlet",
            "right",
            variable="temperature",
            value=10.0,
        )
        problem.add_boundary_condition(
            "Robin",
            "top",
            variable="temperature",
            transfer_coeff=75.0,
        )
        problem.solve(max_iterations=50)
        return problem.values("temperature")

    assert solve(True) == pytest.approx(solve(False), rel=1e-10)


class PrescribedProfile(dm.PythonNodalBoundaryCondition):
    def compute_value(self, x, t):
        return self.amplitude * math.sin(math.pi * x[0])


def test_python_nodal_boundary_condition():
    mesh = dm.generate_rectangle_mesh(
        x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0, num_x_elements=8, num_y_elements=8
    )
    problem = dm.Problem(mesh, method="DMCDM")
    problem.add_variable("u")
    problem.add_kernel("diffusion", variable="u")
    problem.add_boundary_condition(PrescribedProfile(variable="u", boundary="top", amplitude=2.0))
    problem.add_boundary_condition(
        "Dirichlet",
        ["left", "right", "bottom"],
        variable="u",
        value=0.0,
    )
    problem.solve()
    assert problem.sample("u", [[0.5, 1.0]])[0] == pytest.approx(2.0, rel=1e-12)
    assert problem.sample("u", [[0.5, 0.5]])[0] == pytest.approx(
        2.0 * math.sinh(math.pi * 0.5) / math.sinh(math.pi), rel=0.02
    )


# ---------------------------------------------------------------------------
# solvers and output
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("linear_solver", ["bicgstab", "cg"])
def test_the_symmetric_solvers_agree_with_the_direct_solver(linear_solver):
    """The conjugate gradient method needs a symmetric positive definite system: the Galerkin
    Laplacian, which the advection test below does not have."""

    def define():
        mesh = dm.generate_rectangle_mesh(x=[0, 1], y=[0, 1], num_elements=[10, 10])
        problem = dm.Problem(mesh, method="DMCDM")
        problem.add_variable("u")
        problem.add_kernel("diffusion", variable="u")
        problem.add_kernel("body_force", variable="u", value=1.0)
        problem.add_boundary_condition(
            "Dirichlet", ["left", "right", "bottom", "top"], variable="u", value=0.0
        )
        return problem

    direct = define()
    direct.solve(linear_solver="lu")
    iterative = define()
    result = iterative.solve(linear_solver=linear_solver, linear_tolerance=1e-12)
    assert result.linear_iterations > 0
    assert np.allclose(iterative.values("u"), direct.values("u"), rtol=0, atol=1e-12)


def _three_dimensional_problem(method, n=8):
    """A nonsymmetric 3D problem with a variable coefficient."""
    mesh = dm.generate_box_mesh(
        x_min=0.0,
        x_max=1.0,
        y_min=0.0,
        y_max=1.0,
        z_min=0.0,
        z_max=1.0,
        num_x_elements=n,
        num_y_elements=n,
        num_z_elements=n,
        element_type="Tet4",
    )
    problem = dm.Problem(mesh, method=method)
    problem.add_variable("u")
    problem.add_kernel("diffusion", variable="u", diffusivity="1 + x*y")
    problem.add_kernel("advection", variable="u", velocity=[1.0, 0.5, 0.0])
    problem.add_kernel("body_force", variable="u", value="sin(pi*x)*z")
    problem.add_boundary_condition(
        "Dirichlet",
        list(mesh.sideset_names()),
        variable="u",
        value="x*y",
    )
    return problem


@pytest.mark.parametrize("method", ["fem", "dmcdm"])
@pytest.mark.parametrize(
    "options",
    [
        {"linear_solver": "automatic"},
        {"linear_solver": "bicgstab", "preconditioner": "ilu"},
        {"linear_solver": "gmres", "preconditioner": "ilu"},
        {"linear_solver": "bicgstab", "preconditioner": "ilut"},
        {"linear_solver": "gmres", "preconditioner": "jacobi", "linear_max_iterations": 20000},
    ],
)
def test_the_iterative_solvers_agree_with_the_direct_solver(method, options):
    """Every combination reaches the direct solution to the linear tolerance,
    on a nonsymmetric system (advection) of each discretisation."""
    direct = _three_dimensional_problem(method)
    direct.solve(linear_solver="lu")
    iterative = _three_dimensional_problem(method)
    result = iterative.solve(linear_tolerance=1e-12, **options)
    assert np.allclose(iterative.values("u"), direct.values("u"), rtol=0, atol=1e-9)
    assert result.linear_iterations > 0 or options["linear_solver"] == "automatic"


def test_the_automatic_solver_iterates_on_a_large_three_dimensional_system():
    """17^3 = 4913 nodes is past the size at which a 3D system is factorised."""
    direct = _three_dimensional_problem("fem", n=16)
    direct.solve(linear_solver="lu")
    automatic = _three_dimensional_problem("fem", n=16)
    result = automatic.solve()
    assert result.linear_iterations > 0
    assert np.allclose(automatic.values("u"), direct.values("u"), rtol=0, atol=1e-9)


def test_the_automatic_solver_factorises_small_and_two_dimensional_systems():
    """A 2D system of this size is factorised directly: no Krylov
    iterations are counted."""
    mesh = dm.generate_rectangle_mesh(
        x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0, num_x_elements=30, num_y_elements=30
    )
    problem = dm.Problem(mesh, method="DMCDM")
    problem.add_variable("u")
    problem.add_kernel("diffusion", variable="u")
    problem.add_kernel("body_force", variable="u", value=1.0)
    problem.add_boundary_condition(
        "Dirichlet",
        ["left", "right", "bottom", "top"],
        variable="u",
        value=0.0,
    )
    assert problem.solve().linear_iterations == 0


def test_the_linear_system_is_the_one_newton_solves():
    """One Newton step from the assembled system reproduces the solution of a
    linear problem."""
    sparse = pytest.importorskip("scipy.sparse.linalg")
    problem = _three_dimensional_problem("dmcdm")
    residual, jacobian = problem.linear_system()
    assert jacobian.shape == (residual.size, residual.size)
    step = sparse.spsolve(jacobian.tocsc(), -residual)
    predicted = np.asarray(problem.solution()) + step
    problem.solve()
    assert np.allclose(predicted, problem.solution(), atol=1e-10)


def test_a_compiled_function_keeps_the_assembly_threaded():
    """A ParsedFunction object is evaluated in C++, not called back through
    Python, so the problem stays thread-safe."""
    mesh = dm.generate_rectangle_mesh(
        x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0, num_x_elements=4, num_y_elements=4
    )
    problem = dm.Problem(mesh, method="DMCDM")
    problem.add_variable("u")
    problem.add_kernel("diffusion", variable="u")
    problem.add_kernel("body_force", variable="u", value=dm.parsed_function("x*y"))
    assert problem._problem.thread_safe()
    problem.add_kernel("body_force", variable="u", value=lambda x, y, z, t: x)
    assert not problem._problem.thread_safe()


def test_divergence_is_reported():
    mesh = dm.generate_line_mesh(start=0.0, end=1.0, num_elements=10)
    problem = dm.Problem(mesh, method="DMCDM")
    problem.add_variable("u", initial_condition=1.0)
    problem.add_kernel("diffusion", variable="u")
    # an exponentially growing source with no solution nearby
    problem.add_kernel(ExponentialSource(variable="u", magnitude=1.0e8, rate=-5.0))
    problem.add_boundary_condition("Dirichlet", ["left", "right"], variable="u", value=0.0)
    with pytest.raises(RuntimeError, match="did not converge"):
        problem.solve(max_iterations=3)
    result = problem.solve(max_iterations=3, error_on_divergence=False)
    assert not result.converged


# ---------------------------------------------------------------------------
# the bus bar example
# ---------------------------------------------------------------------------
def test_bus_bar_example(tmp_path, monkeypatch):
    """examples/bus_bar.py reproduces Table 5.4.3 of Reddy's book and writes
    its field and post-processor files."""
    import runpy

    monkeypatch.chdir(tmp_path)
    path = Path(__file__).resolve().parents[2] / "examples" / "bus_bar.py"
    problem = runpy.run_path(str(path))["bus_bar"]()
    assert problem.sample("temperature", [[0.05, 0.0]])[0] == pytest.approx(83.142, abs=5e-3)
    assert problem.postprocessor_values()["temperature_bottom"][-1] == pytest.approx(
        83.142, abs=5e-3
    )
    runpy.run_path(str(path), run_name="__main__")
    assert (tmp_path / "bus_bar.vtu").exists()
    assert (tmp_path / "bus_bar_postprocessors.csv").read_text().splitlines()[0].startswith("time,")


def test_parsed_expressions():
    function = dm.parsed_function("500*(1 - 10*x^2)")
    assert function(0.0, 0.0, 0.0, 0.0) == pytest.approx(500.0)
    assert function(0.1, 0.0, 0.0, 0.0) == pytest.approx(450.0)
    time_dependent = dm.parsed_function("sin(pi*x)*exp(-t)")
    assert time_dependent(0.5, 0.0, 0.0, 1.0) == pytest.approx(math.exp(-1.0))


def test_comparison_of_the_two_methods_on_the_same_problem():
    """The dual mesh and finite element solutions differ, but only slightly."""
    mesh = dm.generate_rectangle_mesh(
        x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0, num_x_elements=8, num_y_elements=8
    )
    solutions = {}
    for method in ("dmcdm", "fem"):
        problem = dm.Problem(mesh, method=method)
        problem.add_variable("u")
        problem.add_kernel("diffusion", variable="u")
        problem.add_kernel("body_force", variable="u", value=1.0)
        problem.add_boundary_condition(
            "Dirichlet",
            ["left", "right", "bottom", "top"],
            variable="u",
            value=0.0,
        )
        problem.solve()
        solutions[method] = problem.sample("u", [[0.5, 0.5]])[0]
    exact = 0.07367  # series solution for the unit square
    for method, value in solutions.items():
        assert value == pytest.approx(exact, rel=0.02), method
    assert solutions["dmcdm"] != solutions["fem"]


def test_a_python_callable_parameter_forces_serial_assembly():
    """A Python callable can be given straight to a parameter, as in
    ``value=lambda x, y, z, t: ...``.  It is then held inside the object's
    parameters rather than registered as a named function, but it still calls
    back into the interpreter, so it still has to switch the assembly onto one
    thread.  Without this the threaded element loop deadlocks on the global
    interpreter lock as soon as the mesh is large enough to be split between
    threads.
    """
    mesh = dm.generate_rectangle_mesh(
        x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0, num_x_elements=30, num_y_elements=30
    )
    problem = dm.Problem(mesh, method="DMCDM")
    problem.add_variable("u")
    problem.add_kernel("diffusion", variable="u")
    assert problem.thread_safe  # nothing from Python yet

    problem.add_kernel("body_force", variable="u", value=lambda x, y, z, t: x + y)
    assert not problem.thread_safe
    problem.set_num_threads(2)
    assert problem.effective_threads() == 1

    problem.add_boundary_condition(
        "Dirichlet",
        mesh.sideset_names(),
        variable="u",
        value=0.0,
    )
    problem.solve()  # must finish rather than hang
    assert np.isfinite(problem.values("u")).all()
