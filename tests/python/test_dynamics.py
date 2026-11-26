# SPDX-License-Identifier: LGPL-2.1-or-later
"""Inertia: the generalized-alpha and Newmark methods, natural frequencies, Rayleigh damping.

Each test checks a closed form, an order of accuracy, or an identity of the discrete scheme:
the energy that Newmark's average acceleration keeps, the damping of generalized-alpha at
infinite frequency (its spectral radius), the momentum that the support force changes, and
the modal damping ratios of Rayleigh damping.
"""

from __future__ import annotations

import dualmesh
import numpy
import pytest
from dualmesh._core import InputError
from dualmesh.modes import _operator

schemes = ["generalized_alpha", "Newmark"]


def string(num_elements=200):
    """The string u_tt = u_xx on [0, 1] with fixed ends, at the object level."""
    mesh = dualmesh.generate_line_mesh(start=0.0, end=1.0, num_elements=num_elements)
    problem = dualmesh.Problem(mesh, method="FEM")
    problem.add_variable("u")
    problem.add_kernel("diffusion", variable="u")
    problem.add_kernel("second_time_derivative", variable="u")
    problem.add_boundary_condition("Dirichlet", ["left", "right"], variable="u", value=0.0)
    problem.initialize()
    x = problem.node_coordinates()[:, 0]
    problem.set_values("u", numpy.sin(numpy.pi * x))
    return problem


@pytest.mark.parametrize("scheme", schemes)
def test_a_vibrating_string_is_second_order_in_time(scheme):
    """sin(pi x) released from rest is sin(pi x) cos(pi t).  Measured at t = 0.75, where the
    slope is not zero, the error falls by 4.1 and 4.5 (Newmark) per halving of the step."""
    errors = []
    for dt in (0.0375, 0.01875):
        problem = string()
        problem.solve_transient(
            end_time=0.75, time_step=dt, time_scheme=scheme, time_stepper="fixed", report="none"
        )
        errors.append(abs(problem.sample("u", [[0.5, 0, 0]])[0] - numpy.cos(0.75 * numpy.pi)))
    assert errors[0] / errors[1] == pytest.approx(4.0, rel=0.15)


def test_the_natural_frequencies_of_a_string_converge_to_k_over_2():
    problem = string(100)
    modes = problem.solve_eigenvalue(num_modes=3, report="none")
    assert numpy.sort(modes.frequencies) == pytest.approx([0.5, 1.0, 1.5], rel=5e-4)
    assert max(modes.residual_norms) < 1e-10


def test_newmark_keeps_the_energy_of_a_linear_system():
    """The average acceleration rule conserves 1/2 v.M v + 1/2 u.K u of a linear undamped
    system exactly (to round-off), at any step."""
    problem = string(50)
    stiffness, _ = _operator(problem, True, False, 0.0)
    mass, _ = _operator(problem, False, True, 0.0, order=2)
    energies = []

    def energy(time, core):
        u = numpy.asarray(core.solution())
        v = numpy.asarray(core.velocity())
        energies.append(0.5 * v @ (mass @ v) + 0.5 * u @ (stiffness @ u))

    problem.set_time_step_callback(energy)
    problem.solve_transient(
        end_time=5.0, time_step=0.13, time_scheme="Newmark", time_stepper="fixed", report="none"
    )
    assert len(energies) > 30
    assert max(energies) - min(energies) < 1e-12 * max(energies)


def single_degree_of_freedom():
    """Two linear elements with fixed ends: one free unknown with k = 4 and m = 1/3."""
    mesh = dualmesh.generate_line_mesh(start=0.0, end=1.0, num_elements=2)
    problem = dualmesh.Problem(mesh, method="FEM")
    problem.add_variable("u")
    problem.add_kernel("diffusion", variable="u")
    problem.add_kernel("second_time_derivative", variable="u")
    problem.add_boundary_condition("Dirichlet", ["left", "right"], variable="u", value=0.0)
    problem.initialize()
    problem.set_values("u", numpy.array([0.0, 1.0, 0.0]))
    return problem


@pytest.mark.parametrize("rho", [0.0, 0.5, 0.75])
def test_generalized_alpha_damps_an_unresolved_mode_at_its_spectral_radius(rho):
    """A mode far beyond the step (omega dt = 1e3) loses the factor rho_infinity per step.  At
    infinite frequency the amplification matrix has the triple root -rho_infinity (Chung and
    Hulbert 1993: the spurious root alpha_f / (alpha_f - 1) is -rho_infinity as well), so the
    amplitude goes as N^2 rho^N: the rate over the steps 20 to 40, without the factor
    (40/20)^2, is rho (measured 0.5001 and 0.7548), and rho = 0 annihilates the mode in three
    steps."""
    problem = single_degree_of_freedom()
    history = []
    problem.set_time_step_callback(lambda t, core: history.append(abs(core.solution()[1])))
    omega = numpy.sqrt(12.0)
    dt = 1e3 / omega
    problem.solve_transient(
        end_time=40 * dt, time_step=dt, spectral_radius=rho, time_stepper="fixed", report="none"
    )
    if rho == 0.0:
        # The root is -rho_infinity at infinite frequency only: (omega dt)^-2 = 1e-6 is left
        # after three steps.
        assert max(history[5:]) < 1e-10
        return
    rate = (history[-1] / history[19] / 4.0) ** (1.0 / 20.0)
    assert rate == pytest.approx(rho, rel=0.01)


def test_mass_proportional_damping_decays_a_vibration_as_the_closed_form():
    """m u'' + a m u' + k u = 0 from u = 1 at rest: u = e^(-zeta w t) (cos w_d t + zeta w / w_d
    sin w_d t), with w^2 = k/m = 12 and zeta = a / (2 w)."""
    a, k, m = 0.8, 4.0, 1.0 / 3.0
    omega = numpy.sqrt(k / m)
    zeta = a / (2 * omega)
    omega_d = omega * numpy.sqrt(1 - zeta**2)
    exact = numpy.exp(-zeta * omega * 2.0) * (
        numpy.cos(omega_d * 2.0) + zeta * omega / omega_d * numpy.sin(omega_d * 2.0)
    )
    errors = []
    for dt in (0.01, 0.005):
        problem = single_degree_of_freedom()
        problem.add_kernel("time_derivative", variable="u", coefficient=a)
        problem.solve_transient(
            end_time=2.0, time_step=dt, time_scheme="Newmark", time_stepper="fixed", report="none"
        )
        errors.append(abs(problem.values("u")[1] - exact))
    assert errors[1] < 1e-4
    assert errors[0] / errors[1] == pytest.approx(4.0, rel=0.1)


# ---- solids ----------------------------------------------------------------------------------
E, density, nu = 200e9, 7850.0, 0.3


def cantilever_beam(theory, n, **form):
    b, h, L = 0.02, 0.01, 1.0
    mesh = dualmesh.generate_line_mesh(start=0.0, end=L, num_elements=n)
    problem = dualmesh.Problem(mesh)
    beam = problem.add_physics(
        "solid_mechanics",
        formulation="beam",
        theory=theory,
        youngs_modulus=E,
        poissons_ratio=nu,
        cross_section={"width": b, "height": h},
        density=density,
        **form,
    )
    beam.add_boundary_condition("fixed", boundary="left")
    return problem


@pytest.mark.parametrize(
    "theory, form",
    [
        ("Euler-Bernoulli", {}),
        ("Timoshenko", {}),
        ("Timoshenko", {"beam_formulation": "displacement"}),
    ],
)
def test_the_frequencies_of_a_cantilever_beam_converge_to_Euler_Bernoulli(theory, form):
    """f_k = (beta_k L)^2 / (2 pi) sqrt(E I / (rho A L^4)), beta_k L = 1.8751, 4.6941, 7.8548
    (L/h = 100, so the shear of Timoshenko is small); the error of the third mode falls at
    second order in the element size (measured ratios 4.0 to 5.1: the displacement form of
    Timoshenko converges to its own frequency, a little below that of Euler-Bernoulli)."""
    b, h = 0.02, 0.01
    beta = numpy.array([1.875104, 4.694091, 7.854757])
    exact = beta**2 / (2 * numpy.pi) * numpy.sqrt(E * b * h**3 / 12 / (density * b * h))
    errors = []
    for n in (16, 32):
        modes = cantilever_beam(theory, n, **form).solve_eigenvalue(num_modes=3, report="none")
        f = numpy.sort(modes.frequencies)
        assert f[0] == pytest.approx(exact[0], rel=1e-3)
        errors.append(abs(f[2] - exact[2]))
    assert 3.5 < errors[0] / errors[1] < 5.5


def plane_cantilever(nx=40, ny=4, **parameters):
    """A plane-stress cantilever 1 m long and 0.05 m deep, fixed on the left."""
    mesh = dualmesh.generate_rectangle_mesh(0, 1.0, 0, 0.05, nx, ny, element_type="Quad9")
    problem = dualmesh.Problem(mesh)
    solid = problem.add_physics(
        "solid_mechanics",
        formulation="plane_stress",
        youngs_modulus=E,
        poissons_ratio=nu,
        density=density,
        **parameters,
    )
    solid.add_boundary_condition("fixed", boundary="left")
    return problem, solid


def test_the_first_frequency_of_a_plane_cantilever_is_that_of_the_beam():
    """L/h = 20: the plane-stress solid gives the Euler-Bernoulli frequency to within the shear
    and rotary inertia of Timoshenko, about 0.3 %."""
    h = 0.05
    exact = 1.875104**2 / (2 * numpy.pi) * numpy.sqrt(E * h**2 / 12 / density)
    problem, _ = plane_cantilever()
    f = numpy.sort(problem.solve_eigenvalue(num_modes=1, report="none").frequencies)
    assert f[0] == pytest.approx(exact, rel=6e-3)
    assert f[0] < exact


def test_Rayleigh_damping_gives_each_mode_its_damping_ratio():
    """C = a M + b K damps mode k by zeta_k = (a / w_k + b w_k) / 2: with e^(-lambda t),
    Re(lambda) / |lambda| = zeta_k exactly in the discrete system."""
    a, b = 20.0, 2e-5
    problem, _ = plane_cantilever(16, 2, damping={"mass": a, "stiffness": b})
    modes = problem.solve_eigenvalue(num_modes=4, report="none")
    for value in modes.eigenvalues:
        omega = abs(value)
        assert numpy.real(value) / omega == pytest.approx((a / omega + b * omega) / 2, rel=1e-8)


def axial_bar(nx=80, **parameters):
    """A bar along x in plane stress with nu = 0, fixed at x = 0, free at x = L = 1, held on
    its sides in y: the wave equation u_tt = (E / rho) u_xx."""
    mesh = dualmesh.generate_rectangle_mesh(0, 1.0, 0, 0.1, nx, 1, element_type="Quad9")
    problem = dualmesh.Problem(mesh)
    solid = problem.add_physics(
        "solid_mechanics",
        formulation="plane_stress",
        youngs_modulus=E,
        poissons_ratio=0.0,
        density=density,
        **parameters,
    )
    solid.add_boundary_condition("disp_x = 0", boundary="left")
    solid.add_boundary_condition("disp_y = 0", boundary=["left", "top", "bottom", "right"])
    return problem, solid


def test_an_initial_velocity_starts_the_first_axial_mode():
    """The velocity v0 sin(pi x / 2) gives u = v0 / w sin(pi x / 2) sin(w t), w = (pi / 2)
    sqrt(E / rho)."""
    omega = numpy.pi / 2 * numpy.sqrt(E / density)
    period = 2 * numpy.pi / omega
    problem, _ = axial_bar(initial_condition={"velocity": ["sin(pi*x/2)", 0]})
    problem.solve_transient(end_time=period / 4, time_step=period / 400, report="none")
    tip = problem.sample("disp_x", [[1.0, 0.05]])[0]
    assert tip == pytest.approx(1.0 / omega, rel=1e-4)


def test_the_support_force_changes_the_momentum():
    """With Newmark the equation of motion holds at the end of each step, so the momentum
    P = 1.M v changes over a step by the average of the forces of the support at its two ends
    (reaction_force), to round-off."""
    problem, solid = axial_bar(nx=20, initial_condition={"velocity": ["sin(pi*x/2)", 0]})
    problem.initialize()
    mass, _ = _operator(problem, False, True, 0.0, order=2)
    index = problem.variable_index("disp_x")
    count = len(problem._problem.variable_names())
    ones = numpy.zeros(mass.shape[0])
    ones[index::count] = 1.0
    momentum, force = [], []

    def record(time, core):
        momentum.append(ones @ (mass @ numpy.asarray(core.velocity())))
        force.append(solid.reaction_force("left")["x"])

    problem.set_time_step_callback(record)
    problem.solve_transient(
        end_time=4e-4,
        time_step=2e-5,
        time_scheme="Newmark",
        time_stepper="fixed",
        relative_tolerance=1e-14,
        report="none",
    )
    for k in range(1, len(momentum)):
        change = momentum[k] - momentum[k - 1]
        assert change == pytest.approx(0.5 * 2e-5 * (force[k] + force[k - 1]), rel=1e-9, abs=1e-12)


# ---- input -----------------------------------------------------------------------------------
def test_the_input_of_dynamics_is_checked():
    mesh = dualmesh.generate_rectangle_mesh(0, 1, 0, 1, 2, 2)
    problem = dualmesh.Problem(mesh)
    with pytest.raises(InputError, match="give the density"):
        problem.add_physics(
            "solid_mechanics", formulation="plane_strain", youngs_modulus=1.0, damping={"mass": 1.0}
        )
    with pytest.raises(InputError, match="takes 'mass'"):
        problem.add_physics(
            "solid_mechanics",
            formulation="plane_strain",
            youngs_modulus=1.0,
            density=1.0,
            damping={"viscous": 1.0},
        )
    problem, _ = axial_bar(nx=4)
    with pytest.raises(InputError, match="takes the time_scheme 'generalized_alpha'"):
        problem.solve_transient(end_time=1e-4, time_step=1e-5, time_scheme="backward_Euler")
    with pytest.raises(InputError, match="spectral_radius is the damping"):
        problem.solve_transient(
            end_time=1e-4, time_step=1e-5, time_scheme="Newmark", spectral_radius=0.5
        )
