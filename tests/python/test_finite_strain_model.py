# SPDX-License-Identifier: LGPL-2.1-or-later
"""The solid at finite strain, from its strain energy: a homogeneous stretch reaches the stretch
that its traction gives in closed form, a rigid rotation stays stress free, and a small load gives
linear elasticity with the same Young modulus and Poisson ratio."""

import math

import dualmesh
import pytest

Young_modulus, Poisson_ratio = 1.0e7, 0.3
shear_modulus = Young_modulus / (2 * (1 + Poisson_ratio))
first_Lame_parameter = (
    Young_modulus * Poisson_ratio / ((1 + Poisson_ratio) * (1 - 2 * Poisson_ratio))
)


def block(kinematics="finite", **options):
    """A plane-strain block 2 m by 1 m, held at x = 0 and free to slide, with no lateral motion."""
    solid = dualmesh.problem(mesh={"rectangle": {"size": ["2 m", "1 m"], "cells": [4, 2]}})
    solid.physics("solid_mechanics", formulation="plane_strain", kinematics=kinematics, **options)
    solid.properties(Young_modulus="10 MPa", Poisson_ratio=Poisson_ratio)
    return solid


@pytest.mark.parametrize(
    "energy, stress",
    [
        # P_11 of F = diag(s, 1, 1).
        (
            "neo-Hookean",
            lambda s: shear_modulus * (s - 1 / s) + first_Lame_parameter * math.log(s) / s,
        ),
        (
            "Saint_Venant_Kirchhoff",
            lambda s: s * (first_Lame_parameter + 2 * shear_modulus) * (s**2 - 1) / 2,
        ),
    ],
)
def test_a_traction_stretches_the_block_by_its_closed_form_stretch(energy, stress):
    """With u_y = 0 everywhere, F = diag(s, 1, 1), and the traction P_11(s) on x = 2 m gives
    u_x = (s - 1) x at every node, which linear elements hold exactly."""
    stretch = 1.4
    solid = block(strain_energy=energy)
    solid.boundary_conditions("left", displacement_x="0 m")
    for side in ("left", "right", "bottom", "top"):
        solid.boundary_conditions(side, displacement_y="0 m")
    solid.boundary_conditions("right", traction=[f"{stress(stretch)!r} Pa", "0 Pa"])
    solution = solid.solve(report="none")
    x = solution.coordinates("displacement")[:, 0]
    assert solution.field("displacement")[:, 0] == pytest.approx(
        (stretch - 1) * x, rel=1e-8, abs=1e-12
    )


def test_a_rigid_rotation_stays_a_rotation():
    """A rotation by 30 degrees prescribed on the boundary carries the whole block without stress,
    so the displacement inside is the rotation itself: the energy depends on C alone."""
    cosine, sine = math.cos(math.pi / 6), math.sin(math.pi / 6)
    solid = block()
    for side in ("left", "right", "bottom", "top"):
        solid.boundary_conditions(
            side,
            displacement_x=f"{cosine - 1!r}*x - {sine!r}*y",
            displacement_y=f"{sine!r}*x + {cosine - 1!r}*y",
        )
    solution = solid.solve(report="none")
    x, y = solution.coordinates("displacement").T
    displacement = solution.field("displacement")
    assert displacement[:, 0] == pytest.approx((cosine - 1) * x - sine * y, abs=1e-10)
    assert displacement[:, 1] == pytest.approx(sine * x + (cosine - 1) * y, abs=1e-10)


def test_a_small_load_gives_linear_elasticity():
    """A traction of 1 Pa on a modulus of 10 MPa strains the block by about 1e-7, so finite and
    small strain differ by about that fraction of the displacement.  The stress of the finite
    strain is then 1e-7 of the terms of the size of the modulus that make it up, and Newton's test
    reaches it at the round-off of those terms."""
    solutions = {}
    for kinematics in ("finite", "small"):
        solid = block(kinematics)
        solid.boundary_conditions("left", displacement_x="0 m")
        solid.boundary_conditions("bottom", displacement_y="0 m")
        solid.boundary_conditions("right", traction=["1 Pa", "0.5 Pa"])
        solutions[kinematics] = solid.solve(report="none").field("displacement")
    scale = abs(solutions["small"]).max()
    assert abs(solutions["finite"] - solutions["small"]).max() < 1e-5 * scale


def test_an_energy_of_the_wrong_unit_and_an_energy_at_small_strain_are_refused():
    solid = block(strain_energy="shear_modulus*(I_C - 3)*J/[1 m]")
    solid.boundary_conditions("left", displacement_x="0 m")
    with pytest.raises(
        dualmesh.input_error, match=r"physics\.solid_mechanics\.strain_energy: .*J/m\^3"
    ):
        solid.solve(report="none")
    small = block("small", strain_energy="neo-Hookean")
    small.boundary_conditions("left", displacement_x="0 m")
    with pytest.raises(dualmesh.input_error, match="takes no strain_energy"):
        small.solve(report="none")
