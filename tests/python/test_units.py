# SPDX-License-Identifier: LGPL-2.1-or-later
"""The solvers do not depend on the units of the problem.

Every case is solved twice: in SI units and with lengths in millimetres (masses in kilograms,
times in seconds, temperatures in kelvin).  A length is then 1e3 times larger, a velocity 1e3, a
pressure and a stress 1e-3, a density 1e-9, a dynamic viscosity 1e-3, a conductivity 1e3, a
specific heat 1e6, a volumetric heat source 1e-3, a force per volume 1e-6 and an acceleration
1e3.  The physics is the same, so the two runs must take the same Newton and linear iterations and
give the same solution, converted back.  They are solved with the defaults, as a user would.
"""

from __future__ import annotations

import dualmesh
import numpy
import pytest

millimetres_per_metre = 1e3

#: The factor that turns an SI value of each variable into millimetres.
to_mm = {
    "velocity_x": millimetres_per_metre,
    "velocity_y": millimetres_per_metre,
    "pressure": 1 / millimetres_per_metre,
    "temperature": 1.0,
    "disp_x": millimetres_per_metre,
    "disp_y": millimetres_per_metre,
}


def channel(scale, solve):
    """A developing channel flow at Re = 50 that carries heat from a warm wall, at 300 K."""
    mesh = dualmesh.generate_rectangle_mesh(0, 3 * scale, 0, scale, 24, 8)
    problem = dualmesh.Problem(mesh, method="DMCDM")
    flow = problem.add_physics(
        "fluid_dynamics",
        density=1000.0 / scale**3,
        dynamic_viscosity=0.02 / scale,
        formulation="equal_order",
        viscous_form="laplacian",
    )
    heat = problem.add_physics(
        "heat_conduction",
        thermal_conductivity=0.6 * scale,
        density=1000.0 / scale**3,
        specific_heat=4000.0 * scale**2,
        initial_condition=300.0,
    )
    problem.add_coupling("nonisothermal_flow", heat_conduction=heat, fluid_dynamics=flow)
    u = 1e-3 * scale
    flow.add_boundary_condition(
        [f"velocity_x = {6 * u}*y*({scale} - y)/{scale**2}", "velocity_y = 0"], boundary="left"
    )
    flow.add_boundary_condition(["velocity_x = 0", "velocity_y = 0"], boundary=["top", "bottom"])
    heat.add_boundary_condition("temperature = 300", boundary="left")
    heat.add_boundary_condition("temperature = 310", boundary="bottom")
    return problem, solve(problem)


def cavity(scale, solve):
    """Natural convection in a square cavity of side 0.1 m, at about 300 K."""
    side = 0.1 * scale
    mesh = dualmesh.generate_rectangle_mesh(0, side, 0, side, 12, 12)
    problem = dualmesh.Problem(mesh, method="DMCDM")
    flow = problem.add_physics(
        "fluid_dynamics",
        density=1.2 / scale**3,
        dynamic_viscosity=1.8e-5 / scale,
        formulation="equal_order",
    )
    heat = problem.add_physics(
        "heat_conduction",
        thermal_conductivity=0.026 * scale,
        density=1.2 / scale**3,
        specific_heat=1005.0 * scale**2,
        initial_condition=300.0,
    )
    problem.add_coupling(
        "nonisothermal_flow",
        heat_conduction=heat,
        fluid_dynamics=flow,
        gravity=[0.0, -9.81 * scale],
        volumetric_expansion_coefficient=1 / 300,
        reference_temperature=300.0,
    )
    flow.add_boundary_condition(
        ["velocity_x = 0", "velocity_y = 0"], boundary=["left", "right", "bottom", "top"]
    )
    heat.add_boundary_condition("temperature = 300.1", boundary="left")
    heat.add_boundary_condition("temperature = 299.9", boundary="right")
    return problem, solve(problem)


def plate(scale, solve):
    """A steel plate in plane stress, heated by a source and held at both ends."""
    mesh = dualmesh.generate_rectangle_mesh(0, 0.2 * scale, 0, 0.05 * scale, 16, 4)
    problem = dualmesh.Problem(mesh, method="FEM")
    solid = problem.add_physics(
        "solid_mechanics",
        formulation="plane_stress",
        youngs_modulus=2e11 / scale,
        poissons_ratio=0.3,
        body_force=[0.0, -7.8e4 / scale**2],
    )
    heat = problem.add_physics(
        "heat_conduction",
        thermal_conductivity=50.0 * scale,
        heat_source=1e6 / scale,
    )
    problem.add_coupling(
        "thermal_expansion",
        solid_mechanics=solid,
        heat_conduction=heat,
        linear_expansion_coefficient=1.2e-5,
        stress_free_temperature=300.0,
    )
    solid.add_boundary_condition("fixed", boundary=["left", "right"])
    heat.add_boundary_condition("temperature = 300", boundary=["left", "right"])
    return problem, solve(problem)


def steady(problem, **options):
    return problem.solve(report="none", **options)


def iterative(problem, **options):
    return problem.solve(petsc_options="-dualmesh_schur lsc", report="none", **options)


def transient(problem):
    return problem.solve_transient(
        end_time=0.4, time_step=0.1, time_scheme="Crank-Nicolson", report="none"
    )


flow_variables = ["velocity_x", "velocity_y", "pressure", "temperature"]
cases = {
    "channel": (channel, steady, flow_variables),
    "channel, GMRES": (channel, iterative, flow_variables),
    "channel, Crank-Nicolson": (channel, transient, flow_variables),
    "cavity": (cavity, steady, flow_variables),
    "cavity, GMRES": (cavity, iterative, flow_variables),
    "plate": (plate, steady, ["disp_x", "disp_y", "temperature"]),
}


@pytest.mark.parametrize("case", cases)
def test_the_solve_does_not_depend_on_the_unit_of_length(case):
    build, solve, variables = cases[case]
    SI, SI_result = build(1.0, solve)
    mm, mm_result = build(millimetres_per_metre, solve)
    assert mm_result.total_iterations == SI_result.total_iterations, case
    # GMRES at the default relative tolerance of 1e-12 ends at the round-off level, where the
    # last iterations follow the rounding of each unit: measured within 3 % (193 and 198).
    linear = (SI_result.linear_iterations, mm_result.linear_iterations)
    assert abs(linear[0] - linear[1]) <= 0.05 * max(linear), (case, linear)
    for variable in variables:
        expected = SI.values(variable) * to_mm[variable]
        error = numpy.abs(mm.values(variable) - expected).max()
        assert error <= 1e-8 * numpy.abs(expected).max(), (variable, error)
