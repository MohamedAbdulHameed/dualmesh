# SPDX-License-Identifier: LGPL-2.1-or-later
"""Heat transfer and contact across a gap between two bodies that share no
nodes: gap_heat_transfer, gas_gap_heat_transfer and gap_contact.

Two slabs, an inner one heated by a uniform source and an outer one cooled at
its far face, face each other across a gap. Every check has an exact answer:

* the temperatures of a gap with a given conductance (piecewise quadratic and
  linear), in one, two and three dimensions and in a cylinder;
* the temperatures of a gas-filled gap, whose conductance depends on the
  temperatures, the gas and the gap width, from a scalar equation solved here
  with the gas properties of :mod:`dualmesh.materials.gas`;
* the contact pressure of two elastic slabs pushed together, and the solid
  contact conductance it gives;
* the conservation of the heat that crosses the gap, to round-off, on
  matching and non-matching meshes;
* the Jacobian of the gas gap against finite differences, with the gap open
  and closed.
"""

from __future__ import annotations

import dualmesh as dm
import numpy as np
import pytest
from dualmesh.materials import gas
from scipy.optimize import brentq
from test_jacobian import compare, finite_difference_jacobian

METHODS = ["fem", "dmcdm", "hfvm", "zfvm"]
VERTEX_METHODS = ["fem", "dmcdm", "hfvm"]

# The inner slab is 0 < x < A, the gap A < x < B, the outer slab B < x < C.
A, B, C = 1.0e-2, 1.005e-2, 2.0e-2
HEIGHT = 1.0e-2
K_INNER, K_OUTER, SOURCE, CONDUCTANCE, T_COOL = 3.0, 16.0, 1.0e7, 5000.0, 600.0
STEFAN_BOLTZMANN = 5.670374419e-8


def _body(dimension, x_min, x_max, num_x, num_y):
    if dimension == 1:
        return dm.generate_line_mesh(x_min, x_max, num_x)
    if dimension == 2:
        return dm.generate_rectangle_mesh(
            x_min=x_min,
            x_max=x_max,
            y_min=0.0,
            y_max=HEIGHT,
            num_x_elements=num_x,
            num_y_elements=num_y,
        )
    return dm.generate_box_mesh(
        x_min=x_min,
        x_max=x_max,
        y_min=0.0,
        y_max=HEIGHT,
        z_min=0.0,
        z_max=HEIGHT,
        num_x_elements=num_x,
        num_y_elements=num_y,
        num_z_elements=num_y,
    )


def two_slabs(dimension, num_x, inner_num_y=1, outer_num_y=1, gap_end=B):
    """The two slabs as one mesh of two blocks, inner and outer, with the side
    sets left (x = 0), primary (x = A), secondary (x = gap_end), right (x = C)
    and, in two and three dimensions, bottom (y = 0)."""
    parts = [
        _body(dimension, 0.0, A, num_x, inner_num_y),
        _body(dimension, gap_end, C, num_x, outer_num_y),
    ]
    points, connectivity, blocks, offset = [], [], [], 0
    element_types = set()
    for block, part in enumerate(parts):
        points.append(np.asarray(part.points())[:, :dimension])
        for element in range(part.num_elements):
            connectivity.append([node + offset for node in part.element_nodes(element)])
            element_types.add(part.element_type(element))
            blocks.append(block)
        offset += part.num_nodes
    (element_type,) = element_types
    mesh = dm.mesh_from_arrays(
        np.vstack(points),
        [(element_type, np.array(connectivity))],
        blocks=blocks,
        dimension=dimension,
    )
    mesh.set_block_name(0, "inner")
    mesh.set_block_name(1, "outer")
    tolerance = 1e-9 * C
    for name, position in (("left", 0.0), ("primary", A), ("secondary", gap_end), ("right", C)):
        mesh.add_sideset_by_predicate(
            name, lambda x, y, z, position=position: abs(x - position) < tolerance
        )
    if dimension > 1:
        mesh.add_sideset_by_predicate("bottom", lambda x, y, z: abs(y) < tolerance)
    return mesh


def cross_section(dimension):
    return {1: 1.0, 2: HEIGHT, 3: HEIGHT**2}[dimension]


def slab_temperature(x, primary_temperature, secondary_temperature, gap_end=B):
    """The exact temperature: a parabola in the heated slab and a straight
    line in the cooled one."""
    flux = SOURCE * A
    inner = primary_temperature + SOURCE * (A**2 - x**2) / (2 * K_INNER)
    outer = T_COOL + flux * (C - x) / K_OUTER
    return np.where(x <= 0.5 * (A + gap_end), inner, outer)


def conduction_problem(mesh, method, coordinates="cartesian"):
    problem = dm.Problem(mesh, method=method, coordinates=coordinates)
    problem.add_variable("temperature", initial_condition=T_COOL)
    problem.add_kernel(
        "heat_conduction",
        "inner",
        variable="temperature",
        thermal_conductivity=K_INNER,
        block=["inner"],
    )
    problem.add_kernel(
        "heat_conduction",
        "outer",
        variable="temperature",
        thermal_conductivity=K_OUTER,
        block=["outer"],
    )
    problem.add_kernel(
        "heat_source", "source", variable="temperature", heat_source=SOURCE, block=["inner"]
    )
    problem.add_boundary_condition(
        "Dirichlet_boundary_condition",
        "cooled",
        variable="temperature",
        boundary=["right"],
        value=T_COOL,
    )
    return problem


def given_conductance_problem(
    mesh, method, primary="primary", secondary="secondary", coordinates="cartesian"
):
    problem = conduction_problem(mesh, method, coordinates)
    problem.add_boundary_condition(
        "gap_heat_transfer",
        "gap",
        variable="temperature",
        boundary=[primary],
        secondary_boundary=[secondary],
        gap_conductance=CONDUCTANCE,
    )
    problem.solve()
    return problem


def given_conductance_error(problem):
    secondary_temperature = T_COOL + SOURCE * A * (C - B) / K_OUTER
    exact = slab_temperature(
        problem.entity_points()[:, 0],
        secondary_temperature + SOURCE * A / CONDUCTANCE,
        secondary_temperature,
    )
    return np.abs(problem.values("temperature") - exact).max()


def convergence_rates(errors):
    errors = np.asarray(errors)
    return np.log2(errors[:-1] / errors[1:])


# ---------------------------------------------------------------------------
# A gap with a given conductance
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("dimension", [1, 2, 3])
@pytest.mark.parametrize("method", VERTEX_METHODS)
def test_a_given_conductance_is_exact_on_matching_meshes_with_the_vertex_methods(method, dimension):
    """The exact temperature is quadratic in the heated slab, and the vertex
    methods reproduce it at the nodes in one dimension and, since it does not
    vary across the slabs, in two and three as well."""
    for num_x in (2, 4) if dimension == 3 else (4, 8):
        problem = given_conductance_problem(two_slabs(dimension, num_x, 2, 2), method)
        assert given_conductance_error(problem) < 1e-9


@pytest.mark.parametrize("dimension", [1, 2])
def test_a_given_conductance_converges_at_second_order_with_the_cell_centred_method(dimension):
    errors = [
        given_conductance_error(
            given_conductance_problem(two_slabs(dimension, num_x, 2, 2), "zfvm")
        )
        for num_x in (4, 8, 16, 32)
    ]
    assert convergence_rates(errors).min() > 1.9


@pytest.mark.parametrize("dimension", [2, 3])
@pytest.mark.parametrize("method", METHODS)
def test_the_heat_that_crosses_the_gap_is_conserved_on_non_matching_meshes(method, dimension):
    """Two elements across the inner slab face three across the outer one:
    every watt generated leaves through the cooled face."""
    problem = given_conductance_problem(two_slabs(dimension, 4, 2, 3), method)
    power = SOURCE * A * cross_section(dimension)
    assert -problem.total_reaction("temperature", "right") == pytest.approx(power, rel=1e-12)


@pytest.mark.parametrize("method", METHODS)
def test_non_matching_meshes_converge_at_first_order(method):
    """The closest-point pairing of the two surfaces converges at first order
    when the meshes of the surfaces do not match."""
    errors = [
        given_conductance_error(
            given_conductance_problem(two_slabs(2, num_x, num_x // 2, 3 * num_x // 4), method)
        )
        for num_x in (4, 8, 16)
    ]
    assert convergence_rates(errors).min() > 0.9


@pytest.mark.xfail(
    strict=True, reason="the closest-point pairing converges at first order on non-matching meshes"
)
def test_non_matching_meshes_converge_at_second_order():
    errors = [
        given_conductance_error(
            given_conductance_problem(two_slabs(2, num_x, num_x // 2, 3 * num_x // 4), "fem")
        )
        for num_x in (4, 8, 16)
    ]
    assert convergence_rates(errors).min() > 1.8


@pytest.mark.parametrize("method", VERTEX_METHODS)
def test_the_primary_and_the_secondary_side_can_be_swapped(method):
    """The condition acts on both bodies, so the solution does not depend on
    which surface is the primary one."""
    mesh = two_slabs(2, 4, 3, 3)
    forward = given_conductance_problem(mesh, method)
    backward = given_conductance_problem(mesh, method, primary="secondary", secondary="primary")
    assert backward.values("temperature") == pytest.approx(forward.values("temperature"), rel=1e-12)


@pytest.mark.parametrize("method", METHODS)
def test_a_given_conductance_in_a_cylinder_converges_at_second_order(method):
    """A heated rod inside a tube (a radial line in axisymmetric coordinates):
    the exact temperature is r^2 in the rod and ln r in the tube."""

    def exact(r):
        linear_power = SOURCE * np.pi * A**2
        tube_inner = T_COOL + linear_power / (2 * np.pi * K_OUTER) * np.log(C / B)
        rod_surface = tube_inner + linear_power / (2 * np.pi * A * CONDUCTANCE)
        return np.where(
            r <= A,
            rod_surface + SOURCE * (A**2 - r**2) / (4 * K_INNER),
            T_COOL + linear_power / (2 * np.pi * K_OUTER) * np.log(C / np.maximum(r, B)),
        )

    errors = []
    for num_x in (8, 16, 32):
        problem = given_conductance_problem(two_slabs(1, num_x), method, coordinates="axisymmetric")
        errors.append(
            np.abs(
                problem.values("temperature") - exact(np.abs(problem.entity_points()[:, 0]))
            ).max()
        )
        assert -problem.total_reaction("temperature", "right") == pytest.approx(
            SOURCE * np.pi * A**2, rel=1e-10
        )
    assert convergence_rates(errors).min() > 1.7


def test_interface_conditions_are_refused_by_the_distributed_solver():
    problem = dm.Problem(two_slabs(2, 2, 2, 2), method="fem", distributed=True)
    problem.add_variable("temperature")
    problem.add_kernel(
        "heat_conduction", "conduction", variable="temperature", thermal_conductivity=1.0
    )
    problem.add_boundary_condition(
        "gap_heat_transfer",
        "gap",
        variable="temperature",
        boundary=["primary"],
        secondary_boundary=["secondary"],
        gap_conductance=1e4,
    )
    with pytest.raises(Exception, match="does not support"):
        problem.solve()


# ---------------------------------------------------------------------------
# A gas-filled gap
# ---------------------------------------------------------------------------
COMPOSITION = {"helium": 0.9, "xenon": 0.1}
GAS_PRESSURE = 2.0e6
PRIMARY_ROUGHNESS, SECONDARY_ROUGHNESS = 2.0e-6, 1.0e-6
PRIMARY_EMISSIVITY, SECONDARY_EMISSIVITY = 0.8, 0.7
ROUGHNESS_COEFFICIENT, CONTACT_COEFFICIENT = 1.5, 20.0
HARDNESS, PENALTY = 1.0e9, 1.0e13


def gas_gap_parameters(**changes):
    parameters = dict(
        helium_fraction=COMPOSITION["helium"],
        xenon_fraction=COMPOSITION["xenon"],
        gas_pressure=GAS_PRESSURE,
        primary_roughness=PRIMARY_ROUGHNESS,
        secondary_roughness=SECONDARY_ROUGHNESS,
        primary_emissivity=PRIMARY_EMISSIVITY,
        secondary_emissivity=SECONDARY_EMISSIVITY,
    )
    parameters.update(changes)
    return parameters


def gas_gap_conductance(
    primary_temperature,
    secondary_temperature,
    width,
    accommodation_coefficient=None,
    contact_pressure=0.0,
    primary_conductivity=K_INNER,
    secondary_conductivity=K_OUTER,
):
    """The conductance of Ross and Stoute's model, computed here from its
    equations and the gas properties."""
    gas_temperature = 0.5 * (primary_temperature + secondary_temperature)
    conductivity = gas.thermal_conductivity(COMPOSITION, gas_temperature)
    jump = gas.temperature_jump_distance(
        COMPOSITION, gas_temperature, GAS_PRESSURE, accommodation_coefficient
    )
    conductance = conductivity / (
        max(width, 0.0) + ROUGHNESS_COEFFICIENT * (PRIMARY_ROUGHNESS + SECONDARY_ROUGHNESS) + jump
    )
    exchange = 1.0 / (1.0 / PRIMARY_EMISSIVITY + 1.0 / SECONDARY_EMISSIVITY - 1.0)
    conductance += (
        STEFAN_BOLTZMANN
        * exchange
        * (primary_temperature**2 + secondary_temperature**2)
        * (primary_temperature + secondary_temperature)
    )
    if contact_pressure > 0.0:
        mean_conductivity = (
            2
            * primary_conductivity
            * secondary_conductivity
            / (primary_conductivity + secondary_conductivity)
        )
        roughness = np.sqrt(0.5 * (PRIMARY_ROUGHNESS**2 + SECONDARY_ROUGHNESS**2))
        conductance += (
            CONTACT_COEFFICIENT
            * mean_conductivity
            * contact_pressure
            / (np.sqrt(roughness) * HARDNESS)
        )
    return conductance


def exact_gap_temperatures(width, secondary_temperature, **conductance_options):
    """The primary temperature at which the gap carries the heat of the
    inner slab."""
    flux = SOURCE * A
    primary = brentq(
        lambda t: (
            gas_gap_conductance(t, secondary_temperature, width, **conductance_options)
            * (t - secondary_temperature)
            - flux
        ),
        secondary_temperature,
        secondary_temperature + 1e4,
        xtol=1e-12,
    )
    return primary, secondary_temperature


@pytest.mark.parametrize("accommodation_coefficient", [None, 0.3])
@pytest.mark.parametrize("method", VERTEX_METHODS)
def test_a_gas_gap_carries_the_heat_at_the_conductance_of_its_gas(
    method, accommodation_coefficient
):
    """The conductance depends on the temperatures of both surfaces, so the
    temperature drop across the gap follows from a nonlinear equation."""
    options = (
        {}
        if accommodation_coefficient is None
        else {"accommodation_coefficient": accommodation_coefficient}
    )
    problem = conduction_problem(two_slabs(2, 4, 2, 2), method)
    problem.add_boundary_condition(
        "gas_gap_heat_transfer",
        "gap",
        variable="temperature",
        boundary=["primary"],
        secondary_boundary=["secondary"],
        **gas_gap_parameters(**options),
    )
    problem.solve()
    secondary_temperature = T_COOL + SOURCE * A * (C - B) / K_OUTER
    primary_temperature, _ = exact_gap_temperatures(
        B - A, secondary_temperature, accommodation_coefficient=accommodation_coefficient
    )
    exact = slab_temperature(
        problem.entity_points()[:, 0], primary_temperature, secondary_temperature
    )
    assert problem.values("temperature") == pytest.approx(exact, abs=1e-8)


def thermomechanical_problem(method, displacement, gap_options=None):
    """The gas gap with the width taken from the displacements. ``displacement``
    is either a number, the uniform displacement of the outer slab towards
    +x, or a pressure, with which the two elastic slabs are pushed together
    (see :func:`test_the_contact_pressure_closes_the_gap_and_conducts_the_heat`)."""
    mesh = two_slabs(2, 4, 2, 2)
    problem = conduction_problem(mesh, method)
    displacements = ["displacement_x", "displacement_y"]
    for name in displacements:
        problem.add_variable(name)
    problem.add_boundary_condition(
        "gas_gap_heat_transfer",
        "gap",
        variable="temperature",
        boundary=["primary"],
        secondary_boundary=["secondary"],
        displacements=displacements,
        **(gap_options or gas_gap_parameters()),
    )
    return problem, displacements


@pytest.mark.parametrize("method", VERTEX_METHODS)
@pytest.mark.parametrize("shift", [2.0e-5, -2.0e-5, -6.0e-5])
def test_the_gap_width_follows_the_displacements(method, shift):
    """The outer slab is moved by a prescribed, uniform displacement: the gap
    opens by 20 um, narrows by 20 um, or closes by 10 um past contact
    (where the gas term uses a zero width)."""
    problem, displacements = thermomechanical_problem(method, shift)
    for name in displacements:
        problem.add_kernel("diffusion", f"smooth_{name}", variable=name)
    for side, value in (("left", 0.0), ("primary", 0.0), ("secondary", shift), ("right", shift)):
        problem.add_boundary_condition(
            "Dirichlet_boundary_condition",
            f"hold_{side}",
            variable="displacement_x",
            boundary=[side],
            value=value,
        )
    problem.add_boundary_condition(
        "Dirichlet_boundary_condition",
        "hold_y",
        variable="displacement_y",
        boundary=["left", "right"],
        value=0.0,
    )
    problem.solve()
    secondary_temperature = T_COOL + SOURCE * A * (C - B) / K_OUTER
    primary_temperature, _ = exact_gap_temperatures(B - A + shift, secondary_temperature)
    exact = slab_temperature(
        problem.entity_points()[:, 0], primary_temperature, secondary_temperature
    )
    assert problem.values("temperature") == pytest.approx(exact, abs=1e-8)


@pytest.mark.parametrize("conductivity_source", ["parameter", "property"])
@pytest.mark.parametrize("method", VERTEX_METHODS)
def test_the_contact_pressure_closes_the_gap_and_conducts_the_heat(method, conductivity_source):
    """Two elastic slabs in plane strain: the left face is held, the right
    face is pushed to the left by 70 um, so that the 50 um gap closes. The
    contact pressure P follows from compatibility,

        70 um - 50 um = P (A + C - B) / E' + P / k,

    with the plane-strain modulus E' = E / (1 - nu^2) and the penalty k, and
    it adds the solid contact conductance to the gas and the radiation."""
    youngs_modulus, poissons_ratio, push = 2.0e11, 0.3, 7.0e-5
    gap_options = gas_gap_parameters(
        contact_penalty=PENALTY, meyer_hardness=HARDNESS, secondary_conductivity=K_OUTER
    )
    if conductivity_source == "parameter":
        gap_options["primary_conductivity"] = K_INNER
    problem, displacements = thermomechanical_problem(method, None, gap_options)
    if conductivity_source == "property":
        problem.add_property(
            "constant_property",
            "inner_conductivity",
            block=["inner"],
            property_names=["thermal_conductivity"],
            property_values=[K_INNER],
        )
    problem.add_property(
        "small_strain_stress",
        "stress",
        displacements=displacements,
        formulation="plane_strain",
        youngs_modulus=youngs_modulus,
        poissons_ratio=poissons_ratio,
    )
    for component, name in enumerate(displacements):
        problem.add_kernel(
            "stress_divergence", f"equilibrium_{name}", variable=name, component=component
        )
        problem.add_boundary_condition(
            "gap_contact",
            f"contact_{name}",
            variable=name,
            boundary=["primary"],
            secondary_boundary=["secondary"],
            displacements=displacements,
            component=component,
            penalty=PENALTY,
        )
    problem.add_boundary_condition(
        "Dirichlet_boundary_condition",
        "held",
        variable="displacement_x",
        boundary=["left"],
        value=0.0,
    )
    problem.add_boundary_condition(
        "Dirichlet_boundary_condition",
        "pushed",
        variable="displacement_x",
        boundary=["right"],
        value=-push,
    )
    problem.add_boundary_condition(
        "Dirichlet_boundary_condition",
        "roller",
        variable="displacement_y",
        boundary=["bottom"],
        value=0.0,
    )
    problem.solve()
    plane_strain_modulus = youngs_modulus / (1 - poissons_ratio**2)
    pressure = (push - (B - A)) / ((A + C - B) / plane_strain_modulus + 1.0 / PENALTY)
    points = problem.entity_points()
    displacement = problem.values("displacement_x")
    primary_nodes = np.abs(points[:, 0] - A) < 1e-12
    assert displacement[primary_nodes] == pytest.approx(
        -pressure * A / plane_strain_modulus, rel=1e-9
    )
    secondary_temperature = T_COOL + SOURCE * A * (C - B) / K_OUTER
    primary_temperature, _ = exact_gap_temperatures(
        -pressure / PENALTY, secondary_temperature, contact_pressure=pressure
    )
    exact = slab_temperature(points[:, 0], primary_temperature, secondary_temperature)
    assert problem.values("temperature") == pytest.approx(exact, abs=1e-6)


@pytest.mark.parametrize("closed", [False, True])
@pytest.mark.parametrize("method", METHODS)
def test_the_jacobian_of_the_gas_gap_agrees_with_finite_differences(method, closed):
    """At a state that is not a solution, with a temperature and a gap width
    that vary along the surfaces, and with the gap open or closed."""
    gap_options = gas_gap_parameters(
        contact_penalty=PENALTY,
        meyer_hardness=HARDNESS,
        primary_conductivity=K_INNER,
        secondary_conductivity=K_OUTER,
    )
    problem, displacements = thermomechanical_problem(method, None, gap_options)
    for name in displacements:
        problem.add_kernel("diffusion", f"smooth_{name}", variable=name)
    problem.initialize()
    points = problem.entity_points()
    x, y = points[:, 0], points[:, 1]
    problem.set_values("temperature", 700.0 + 3e3 * x + 2e3 * y**2 * (1 + x / C))
    shift = (-7e-5 if closed else -2e-5) * (1.0 + 20.0 * y)
    problem.set_values("displacement_x", np.where(x > 0.5 * (A + B), shift, 1e-6 * y))
    problem.set_values("displacement_y", 1e-6 * x)
    _, jacobian = problem.linear_system()
    assert compare(jacobian.toarray(), finite_difference_jacobian(problem, step=1e-9)) < 1e-5


@pytest.mark.parametrize("method", VERTEX_METHODS)
def test_a_cold_initial_guess_still_converges(method):
    """With the temperature left at zero, the gas properties are evaluated at
    200 K or above, so that the first Jacobian of Newton's method is finite."""
    mesh = two_slabs(2, 4, 2, 2)
    problem = dm.Problem(mesh, method=method)
    problem.add_variable("temperature")
    problem.add_kernel(
        "heat_conduction",
        "inner",
        variable="temperature",
        thermal_conductivity=K_INNER,
        block=["inner"],
    )
    problem.add_kernel(
        "heat_conduction",
        "outer",
        variable="temperature",
        thermal_conductivity=K_OUTER,
        block=["outer"],
    )
    problem.add_kernel(
        "heat_source", "source", variable="temperature", heat_source=SOURCE, block=["inner"]
    )
    problem.add_boundary_condition(
        "gas_gap_heat_transfer",
        "gap",
        variable="temperature",
        boundary=["primary"],
        secondary_boundary=["secondary"],
        **gas_gap_parameters(),
    )
    problem.add_boundary_condition(
        "Dirichlet_boundary_condition",
        "cooled",
        variable="temperature",
        boundary=["right"],
        value=T_COOL,
    )
    problem.solve()
    secondary_temperature = T_COOL + SOURCE * A * (C - B) / K_OUTER
    primary_temperature, _ = exact_gap_temperatures(B - A, secondary_temperature)
    exact = slab_temperature(
        problem.entity_points()[:, 0], primary_temperature, secondary_temperature
    )
    assert problem.values("temperature") == pytest.approx(exact, abs=1e-8)


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"primary_roughness": None}, "primary_roughness"),
        ({"secondary_emissivity": None}, "secondary_emissivity"),
        ({"primary_emissivity": 1.5}, "emissivities must lie between 0 and 1"),
        ({"primary_roughness": -1e-6}, "roughnesses must not be negative"),
        ({"accommodation_coefficient": 0.0}, r"accommodation_coefficient must lie in \(0, 1\]"),
        ({"contact_penalty": 1e13}, "needs 'meyer_hardness'"),
        ({"contact_penalty": 1e13, "meyer_hardness": 1e9}, "needs 'secondary_conductivity'"),
        (
            {"contact_penalty": 1e13, "meyer_hardness": 1e9, "secondary_conductivity": 16.0},
            "needs 'primary_conductivity'",
        ),
        ({"helium_fraction": 0.0, "xenon_fraction": 0.0}, "mole fractions add up to zero"),
    ],
)
def test_an_incomplete_or_unphysical_gas_gap_is_refused(changes, message):
    problem = conduction_problem(two_slabs(1, 2), "fem")
    parameters = {
        name: value for name, value in gas_gap_parameters(**changes).items() if value is not None
    }
    with pytest.raises(Exception, match=message):
        problem.add_boundary_condition(
            "gas_gap_heat_transfer",
            "gap",
            variable="temperature",
            boundary=["primary"],
            secondary_boundary=["secondary"],
            **parameters,
        )
        problem.solve()
