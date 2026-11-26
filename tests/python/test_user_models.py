# SPDX-License-Identifier: LGPL-2.1-or-later
"""A model of your own solves to the exact answer of its balance law: a model of heat conduction
equals heat_conduction, a state of a held concentration follows its exact rate law in time, a
diffusing species with fast linear trapping diffuses with the effective diffusivity of Oriani,
a term of the wrong unit is refused with its key, and the amount that a saturable trap and the
solute store equals the amount that enters through the boundary."""

import dualmesh
import numpy
import pytest


def conduction(method, as_model):
    """A steel slab held at 400 K on the left and heated by 5 kW/m^2 on the right, from 300 K."""
    slab = dualmesh.problem(mesh={"segment": {"length": "0.1 m", "cells": 10}}, method=method)
    if as_model:
        slab.model(
            "conduction",
            unknown="temperature",
            storage="density*specific_heat_capacity*temperature",
            flux="-thermal_conductivity*grad(temperature)",
            flux_name="heat_flux",
        )
    else:
        slab.physics("heat_conduction")
    slab.properties(
        thermal_conductivity="45 W/(m*K)",
        density="7850 kg/m^3",
        specific_heat_capacity="490 J/(kg*K)",
    )
    slab.initial_conditions(temperature="300 K")
    slab.boundary_conditions("left", temperature="400 K")
    slab.boundary_conditions("right", heat_flux="-5 kW/m^2")
    slab.time(end_time="40 s", initial_time_step="4 s", step_control={"by": "fixed"})
    return slab


@pytest.mark.parametrize("method", ["FEM", "DMCDM"])
def test_a_model_of_heat_conduction_equals_heat_conduction(method):
    """The model holds the energy balance of heat_conduction, written as its storage and its flux,
    so the two give the same temperatures to round-off."""
    built_in = conduction(method, as_model=False).solve(report="none")
    model = conduction(method, as_model=True).solve(report="none")
    assert model.times == pytest.approx(built_in.times, rel=1e-14)
    assert model.field("temperature") == pytest.approx(
        built_in.field("temperature"), rel=1e-10, abs=1e-10
    )


def trapping_model(storage="solute + trapped", rate="trapping_rate*solute - release_rate*trapped"):
    """A solute c in mol/m^3 that diffuses and is trapped with the linear rate law
    dq/dt = k c - p q of the trapped concentration q in mol/m^3."""
    return {
        "unknown": {"solute": {"unit": "mol/m^3"}},
        "states": {"trapped": {"unit": "mol/m^3", "initial": 0, "rate": rate}},
        "storage": storage,
        "flux": "-diffusivity*grad(solute)",
        "flux_name": "solute_flux",
    }


def test_a_state_at_a_held_concentration_follows_its_exact_rate_law():
    """The solute starts at c0 and both faces hold it there, and the trap does not take from its
    storage, so the solute stays at c0 and every point traps at the same rate. Backward Euler
    gives q_n = q_e + (q_0 - q_e) (1 + p dt)^(-n), with q_e = k c0 / p, which tends to the
    exponential q(t) = q_e + (q_0 - q_e) exp(-p t) as p dt -> 0."""
    c0, k, p, dt = 1.0, 2.0, 0.5, 0.01
    cell = dualmesh.problem(mesh={"segment": {"length": "1 m", "cells": 4}})
    cell.model("trapping", **trapping_model(storage="solute"))
    cell.properties(
        diffusivity="1e-3 m^2/s",
        trapping_rate={"unit": "1/s", "value": "2 1/s"},
        release_rate={"unit": "1/s", "value": "0.5 1/s"},
    )
    cell.initial_conditions(solute="1 mol/m^3")
    cell.boundary_conditions("left", solute="1 mol/m^3")
    cell.boundary_conditions("right", solute="1 mol/m^3")
    cell.time(end_time="2 s", initial_time_step="0.01 s", step_control={"by": "fixed"})
    cell.postprocessor("trapped_average", average="trapped")
    cell.postprocessor("trapped_maximum", maximum="trapped")
    solution = cell.solve(report="none")
    # The postprocessors hold the start of the run and every accepted step.
    times = numpy.asarray(solution.output_times)
    assert times == pytest.approx(dt * numpy.arange(times.size), abs=1e-12)
    assert times[-1] == pytest.approx(2.0, rel=1e-12)
    steps = numpy.arange(times.size)
    equilibrium = k * c0 / p
    backward_Euler = equilibrium * (1.0 - (1.0 + p * dt) ** (-steps))
    trapped = solution.postprocessor("trapped_average")
    assert trapped == pytest.approx(backward_Euler, rel=1e-12, abs=1e-14)
    assert solution.postprocessor("trapped_maximum") == pytest.approx(
        backward_Euler, rel=1e-12, abs=1e-14
    )
    exact = equilibrium * (1.0 - numpy.exp(-p * times))
    assert trapped == pytest.approx(exact, abs=p * dt * equilibrium)
    assert solution.field("solute")[:, 0] == pytest.approx(c0, rel=1e-12)


def oriani_slab(method):
    """A slab of length 1 m in 20 cells, emptied through both faces from c0 = 1 mol/m^3, with a
    trap in equilibrium at the start, q0 = (k/p) c0."""
    slab = dualmesh.problem(mesh={"segment": {"length": "1 m", "cells": 20}}, method=method)
    model = trapping_model()
    model["states"]["trapped"]["initial"] = 9.0
    slab.model("trapping", **model)
    slab.properties(
        diffusivity="1 m^2/s",
        trapping_rate={"unit": "1/s", "value": "9e9 1/s"},
        release_rate={"unit": "1/s", "value": "1e9 1/s"},
    )
    slab.initial_conditions(solute="1 mol/m^3")
    slab.boundary_conditions("left", solute="0 mol/m^3")
    slab.boundary_conditions("right", solute="0 mol/m^3")
    slab.time(end_time="0.02 s", initial_time_step="1e-3 s", step_control={"by": "fixed"})
    return slab


def lumped_steps(steps, cells, dt, D, k, p, q0, length=1.0):
    """The solute at the interior nodes after each step of backward Euler with the lumped storage
    of linear elements: h (c - c_old) + h (q - q_old) + dt (D/h) A c = 0 at each interior node,
    with A the second difference, and q = (q_old + dt k c) / (1 + dt p) at each node."""
    h = length / cells
    n = cells - 1
    A = 2.0 * numpy.eye(n) - numpy.eye(n, k=1) - numpy.eye(n, k=-1)
    c, q = numpy.ones(n), numpy.full(n, q0)
    for _ in range(steps):
        capacity = h * (1.0 + dt * k / (1.0 + dt * p))
        right = h * c + h * q - h * q / (1.0 + dt * p)
        c = numpy.linalg.solve(capacity * numpy.eye(n) + dt * D / h * A, right)
        q = (q + dt * k * c) / (1.0 + dt * p)
    return c


@pytest.mark.parametrize("method", ["FEM", "DMCDM"])
def test_a_trap_in_local_equilibrium_diffuses_with_the_effective_diffusivity_of_Oriani(method):
    """Linear trapping with k = 9e9 1/s and p = 1e9 1/s exchanges far faster than a step of 1 ms,
    so the trap stays at its local equilibrium q = (k/p) c and the solute diffuses with the
    effective diffusivity D/(1 + k/p) of Oriani (1970). The solute follows the exact steps of
    the discrete balance to round-off, and the steps of a storage (1 + k/p) c with no trap to
    the relative difference of order 1/(p dt)."""
    cells, dt, steps, D, k, p = 20, 1e-3, 20, 1.0, 9e9, 1e9
    solution = oriani_slab(method).solve(report="none")
    assert len(solution.times) == steps
    x = solution.coordinates("solute")[:, 0]
    order = numpy.argsort(x)
    solute = solution.field("solute")[order, 0]
    assert solute[0] == 0.0 and solute[-1] == 0.0
    exact = lumped_steps(steps, cells, dt, D, k, p, q0=k / p)
    assert solute[1:-1] == pytest.approx(exact, rel=1e-9, abs=1e-12)
    # The limit of local equilibrium: (1 + k/p) h (c - c_old) + dt (D/h) A c = 0.
    effective = lumped_steps(steps, cells, dt, D / (1.0 + k / p), 0.0, 1.0, q0=0.0)
    assert solute[1:-1] == pytest.approx(effective, rel=1e-4)


def test_a_flux_of_the_wrong_unit_is_refused_with_its_key():
    slab = dualmesh.problem(mesh={"segment": {"length": "1 m", "cells": 4}})
    model = trapping_model()
    model["flux"] = "-grad(solute)"
    slab.model("trapping", **model)
    slab.properties(
        diffusivity="1 m^2/s",
        trapping_rate={"unit": "1/s", "value": "1 1/s"},
        release_rate={"unit": "1/s", "value": "1 1/s"},
    )
    slab.boundary_conditions("left", solute="1 mol/m^3")
    with pytest.raises(
        dualmesh.input_error, match=r"^models\.trapping: .*the terms disagree in dimension"
    ):
        slab.solve(report="none")


def test_a_rate_of_the_wrong_unit_is_refused_with_its_key():
    slab = dualmesh.problem(mesh={"segment": {"length": "1 m", "cells": 4}})
    slab.model("trapping", **trapping_model(rate="trapping_rate*solute*release_rate"))
    slab.properties(
        diffusivity="1 m^2/s",
        trapping_rate={"unit": "1/s", "value": "1 1/s"},
        release_rate={"unit": "1/s", "value": "1 1/s"},
    )
    slab.boundary_conditions("left", solute="1 mol/m^3")
    with pytest.raises(
        dualmesh.input_error, match=r"^models\.trapping: .*the rate of state 'trapped'"
    ):
        slab.solve(report="none")


def test_the_stored_amount_equals_the_amount_that_enters_through_the_boundary():
    """A solute enters a slab at J = 1e-3 mol/(m^2 s) through the left face and fills a
    saturable trap of density N, dq/dt = k c (1 - q) - p q with the occupancy q. The right face
    is closed, so the amount stored, the integral of c + N q, is J t at every accepted step,
    whatever the steps. The storage of DMCDM is lumped at the nodes, where the states live, and
    the integral of an expression takes the average of the points of each cell, so on a uniform
    segment the two sum the same values with the same weights."""
    inflow = 1e-3
    slab = dualmesh.problem(mesh={"segment": {"length": "1 m", "cells": 10}}, method="DMCDM")
    slab.model(
        "saturable_trapping",
        unknown={"solute": {"unit": "mol/m^3"}},
        states={
            "occupancy": {
                "unit": "1",
                "initial": 0,
                "rate": "trapping_rate*solute*(1 - occupancy) - release_rate*occupancy",
            }
        },
        expressions={"stored": {"unit": "mol/m^3", "value": "solute + trap_density*occupancy"}},
        storage="stored",
        flux="-diffusivity*grad(solute)",
        flux_name="solute_flux",
    )
    slab.properties(
        trap_density={"unit": "mol/m^3", "value": "10 mol/m^3"},
        trapping_rate={"unit": "m^3/(mol*s)", "value": "1 m^3/(mol*s)"},
        release_rate={"unit": "1/s", "value": "0.1 1/s"},
        diffusivity="0.01 m^2/s",
    )
    slab.initial_conditions(solute="0 mol/m^3")
    slab.boundary_conditions("left", solute_flux="-1e-3 mol/(m^2*s)")
    slab.time(end_time="100 s", initial_time_step="5 s")
    slab.postprocessor("amount", integral="stored")
    slab.postprocessor("solute_amount", integral="solute")
    solution = slab.solve(report="none")
    # The postprocessors hold the start of the run and every accepted step.
    times = numpy.asarray(solution.output_times)
    assert times[0] == 0.0 and times[-1] == pytest.approx(100.0, rel=1e-12)
    assert times.size > 3
    amount = solution.postprocessor("amount")
    assert amount == pytest.approx(inflow * times, rel=1e-8, abs=1e-15)
    assert solution.unit("amount") == solution.unit("solute_amount")
    # The trap holds part of the amount.
    assert solution.postprocessor("solute_amount")[-1] < 0.9 * amount[-1]
