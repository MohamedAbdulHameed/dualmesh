# SPDX-License-Identifier: LGPL-2.1-or-later
"""Two-way coupled multiphysics: natural convection in a heated cavity.

The benchmark of de Vahl Davis (1983) couples the momentum equations of an
incompressible flow to the energy equation in both directions: the buoyancy
force makes the temperature drive the flow, and convection makes the flow carry
the heat.  The problem is solved monolithically, by Newton's method with the
exact Jacobian of the whole coupled system, so the tests check both that the
physics is right (the published Nusselt numbers and velocity maxima) and that
the coupling blocks of the Jacobian are exact (the number of Newton
iterations).

The tests build the benchmark with the objects of the engine, in its
non-dimensional form, with the cavity width L, the thermal diffusivity kappa
and the temperature difference Delta T as scales:

    (v . grad) v = -grad p + Pr div(grad v) + Ra Pr T e_y ,   div v = 0 ,
    v . grad T = div(grad T) ,

with the temperature +1/2 on the left wall and -1/2 on the right, adiabatic
top and bottom walls, and no slip everywhere.  Pr = 0.71 (air).  The pressure
is eliminated by the penalty method, and the Rayleigh number Ra is reached by
load stepping, which multiplies the buoyancy force.  The average Nusselt
number on the hot wall is the total heat flow through it, the sum of the
reactions of the temperature equation there.  The example
``examples/natural_convection`` solves the same cavity of air through the
input, in SI units, with the default equal-order formulation.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import dualmesh
import numpy
import pytest
from dualmesh._core import InputError

_example = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "natural_convection"
    / "natural_convection.py"
)
_spec = importlib.util.spec_from_file_location("natural_convection", _example)
_natural_convection = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_natural_convection)
#: The values of de Vahl Davis, which the example holds.
benchmark = _natural_convection.benchmark
Prandtl_number = 0.71


def cavity_mesh(num_elements=32):
    """A unit square mesh whose lines cluster at the walls (a cosine spacing), where
    the thermal and viscous boundary layers are."""
    s = numpy.linspace(0.0, 1.0, num_elements + 1)
    x = 0.5 * (1.0 - numpy.cos(numpy.pi * s))
    return dualmesh.generate_rectangle_mesh(x_coordinates=list(x), y_coordinates=list(x))


def solve(Rayleigh_number=1.0e5, num_elements=32, method="DMCDM", advection_scheme=None):
    """The non-dimensional cavity by the penalty formulation, and the result of its solve."""
    problem = dualmesh.Problem(cavity_mesh(num_elements), method=method)
    flow = problem.add_physics(
        "fluid_dynamics",
        dynamic_viscosity=Prandtl_number,
        density=1.0,
        formulation="penalty",
        penalty_parameter=1.0e7,
    )
    heat = problem.add_physics(
        "heat_conduction",
        thermal_conductivity=1.0,
        density=1.0,
        specific_heat=1.0,
        advection_scheme=advection_scheme,
    )
    problem.add_coupling(
        "nonisothermal_flow",
        heat_conduction=heat,
        fluid_dynamics=flow,
        gravity=[0.0, -1.0],
        volumetric_expansion_coefficient=Rayleigh_number * Prandtl_number,
    )
    flow.add_boundary_condition(
        ["velocity_x = 0", "velocity_y = 0"], boundary=["left", "right", "bottom", "top"]
    )
    heat.add_boundary_condition("temperature = 0.5", boundary="left")
    heat.add_boundary_condition("temperature = -0.5", boundary="right")
    result = problem.solve(report="none")
    return problem, result


def measure(problem):
    """The three benchmark quantities."""
    s = numpy.linspace(0.0, 1.0, 801)
    u = problem.sample("velocity_x", numpy.column_stack([numpy.full_like(s, 0.5), s]))
    v = problem.sample("velocity_y", numpy.column_stack([s, numpy.full_like(s, 0.5)]))
    return {
        "nusselt": -problem.boundary_flux("temperature", "left"),
        "u_max": float(numpy.nanmax(u)),
        "v_max": float(numpy.nanmax(v)),
        "u_max_height": float(s[numpy.nanargmax(u)]),
        "v_max_position": float(s[numpy.nanargmax(v)]),
    }


@pytest.mark.parametrize("method", ["dmcdm", "fem"])
@pytest.mark.parametrize("Rayleigh_number", [1.0e3, 1.0e4, 1.0e5])
def test_natural_convection_matches_de_vahl_davis(method, Rayleigh_number):
    """On a 32 by 32 mesh graded towards the walls, the average Nusselt number
    is within 0.5 % of the benchmark and the velocity maxima within 2 %.  With
    the default streamline stabilisation of the heat the measured differences
    are up to 0.35 % and 1.5 % (FEM, Ra = 10^5).  It leaves the conduction term
    out of its residual, which costs some accuracy where conduction matters as
    much as convection, as it does here.  Without it (advection_scheme
    'central', below) they are below 0.2 % and 0.7 %."""
    problem, result = solve(Rayleigh_number, 32, method)
    computed = measure(problem)
    reference = benchmark[Rayleigh_number]
    assert computed["nusselt"] == pytest.approx(reference["nusselt"], rel=5e-3)
    assert computed["u_max"] == pytest.approx(reference["u_max"], rel=2e-2)
    assert computed["v_max"] == pytest.approx(reference["v_max"], rel=2e-2)
    # The buoyancy lifts the fluid at the hot left wall, which then flows right along the top.
    # A buoyancy of the wrong sign mirrors the flow, with the same maxima and Nusselt number.
    assert computed["v_max_position"] < 0.5 and computed["u_max_height"] > 0.5
    # Energy balance: the heat entering through the hot wall leaves through
    # the cold one, because the top and bottom are adiabatic.
    assert -problem.boundary_flux("temperature", "right") == pytest.approx(
        -computed["nusselt"], rel=1e-9
    )
    # Newton's method with the exact coupled Jacobian: a handful of
    # iterations per decade of Rayleigh number.
    decades = len([r for r in (1e3, 1e4, 1e5, 1e6) if r <= Rayleigh_number])
    assert result.total_iterations <= 8 * decades


def test_the_coupling_is_exact_in_newtons_method():
    """Near the solution Newton's method converges quadratically only if the
    Jacobian is exact, including the blocks that couple the temperature and
    the velocities.  The relative size of the Newton step, s_k, must then
    satisfy s_{k+1} <= C s_k^2 once it is small.  The residual itself is not
    used, because with a penalty parameter of 10^7 it reaches its round-off
    floor (about 10^-7) one step before the solution stops changing.  The
    steps of the default solve after its Stokes start are 6.6e-2, 1.8e-3 and
    4.2e-7, so C is about 0.4 and 0.13."""
    problem, result = solve(1.0e4, 16, "dmcdm")
    # The iterations of the last solve: the history holds the Stokes start before it.
    last = max(i for i, r in enumerate(result.history) if r.iteration == 1)
    steps = [r.step_norm for r in result.history[last:] if r.step_norm > 1e-15]
    ratios = [steps[i + 1] / steps[i] ** 2 for i in range(len(steps) - 1) if steps[i] < 0.1]
    assert len(ratios) >= 2, "the solve did not reach the quadratic regime"
    assert max(ratios) < 10.0


@pytest.mark.parametrize("method", ["FEM", "DMCDM"])
def test_natural_convection_without_streamline_stabilisation_of_the_heat(method):
    """The heat carried with advection_scheme 'central' at Ra = 10^5, where conduction matters
    as much as convection: closer than the default streamline stabilisation (measured below
    0.1 % for the Nusselt number and 0.7 % for the velocity maxima)."""
    problem, _ = solve(1.0e5, 32, method, advection_scheme="central")
    computed = measure(problem)
    reference = benchmark[1.0e5]
    assert computed["nusselt"] == pytest.approx(reference["nusselt"], rel=2e-3)
    assert computed["u_max"] == pytest.approx(reference["u_max"], rel=1e-2)
    assert computed["v_max"] == pytest.approx(reference["v_max"], rel=1e-2)


@pytest.mark.parametrize(
    "flow_density, gravity, message",
    [
        (0.0, [0.0, -1.0], "needs the density of the fluid"),
        (1.0, [0.0, -1.0, 0.0], "gravity has 3 components and the flow 2 velocities"),
        (1.0, [-1.0], "gravity has 1 components and the flow 2 velocities"),
    ],
)
def test_a_buoyancy_without_the_density_or_a_gravity_per_velocity_is_refused(
    flow_density, gravity, message
):
    """The buoyancy force rho beta (T - T_0) g takes the density of the fluid and one component
    of gravity per velocity: a Stokes flow without its density, or a gravity of another length
    than the velocity, is refused."""
    mesh = dualmesh.generate_rectangle_mesh(0, 1, 0, 1, 2, 2)
    problem = dualmesh.Problem(mesh)
    flow = problem.add_physics("fluid_dynamics", density=flow_density, dynamic_viscosity=1.0)
    heat = problem.add_physics(
        "heat_conduction", thermal_conductivity=1.0, density=1.0, specific_heat=1.0
    )
    problem.add_coupling(
        "nonisothermal_flow",
        heat_conduction=heat,
        fluid_dynamics=flow,
        gravity=gravity,
        volumetric_expansion_coefficient=1.0,
    )
    with pytest.raises(InputError, match=message):
        problem.solve(report="none")
