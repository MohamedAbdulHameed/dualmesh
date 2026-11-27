# SPDX-License-Identifier: LGPL-2.1-or-later
"""The capabilities of the input that reach the engine's meshes, conditions and studies, each
against an exact answer: a graded mesh that starts away from the origin, the values along a line,
a plane of symmetry of a heated slab, of a pulled bar and of a channel, an eigenvalue study of a
cooling rod and of a vibrating bar, an eigenvalue in the source of a model, with and without a
time derivative, an initial condition given by a formula, a rigid rotation of a solid at finite
strain, a plastic bar, and the traction and the von Mises stress of a pulled bar. Each input that
the capability refuses is refused with its key."""

import csv

import dualmesh
import numpy
import pytest
from dualmesh._core import InputError
from dualmesh.input import schema


def test_a_graded_segment_that_starts_at_its_origin_has_cells_in_the_ratio_of_its_grading():
    """Three cells of 1 m from x = 2 m with the grading 2 have the lengths 1/7, 2/7 and 4/7 m."""
    rod = dualmesh.problem(
        mesh={"segment": {"length": "1 m", "cells": 3, "origin": "2 m", "grading": 2}}
    )
    rod.physics("heat_conduction")
    rod.properties(thermal_conductivity="1 W/(m*K)")
    rod.boundary_conditions("left", temperature="300 K")
    rod.boundary_conditions("right", temperature="400 K")
    solution = rod.solve(report="none")
    x = numpy.sort(solution.coordinates("temperature")[:, 0])
    assert x == pytest.approx([2.0, 2.0 + 1 / 7, 2.0 + 3 / 7, 3.0], rel=1e-14)
    # The conduction between two held faces is linear on any mesh.
    temperature = numpy.sort(solution.field("temperature")[:, 0])
    assert temperature == pytest.approx(300.0 + 100.0 * (x - 2.0), rel=1e-12)


def test_a_graded_rectangle_grades_each_axis_by_its_own_ratio():
    plate = dualmesh.problem(
        mesh={
            "rectangle": {
                "size": ["3 m", "1 m"],
                "cells": [2, 2],
                "origin": ["-1 m", "0.5 m"],
                "grading": [2, 0.5],
            }
        }
    )
    plate.physics("heat_conduction")
    plate.properties(thermal_conductivity="1 W/(m*K)")
    plate.boundary_conditions("left", temperature="300 K")
    solution = plate.solve(report="none")
    x, y = solution.coordinates("temperature").T
    assert numpy.unique(x.round(12)) == pytest.approx([-1.0, 0.0, 2.0], abs=1e-12)
    assert numpy.unique(y.round(12)) == pytest.approx([0.5, 0.5 + 2 / 3, 1.5], abs=1e-12)


def test_an_axisymmetric_mesh_that_starts_below_the_axis_is_refused():
    with pytest.raises(InputError, match=r"mesh\.rectangle\.origin: .*r = 0 m"):
        dualmesh.problem(
            mesh={
                "rectangle": {"size": ["1 m", "1 m"], "cells": [2, 2], "origin": ["-1 m", "0 m"]},
                "coordinate_system": "axisymmetric",
            }
        )


def conducting_slab(tmp_path, monkeypatch, cells=4):
    """A slab of 1 m held at 300 K and 400 K: its temperature is linear, so the values along any
    line through it are exact."""
    monkeypatch.chdir(tmp_path)
    slab = dualmesh.problem(mesh={"rectangle": {"size": ["1 m", "0.5 m"], "cells": [cells, 2]}})
    slab.physics("heat_conduction")
    slab.properties(thermal_conductivity="1 W/(m*K)")
    slab.boundary_conditions("left", temperature="300 K")
    slab.boundary_conditions("right", temperature="400 K")
    slab.postprocessor(
        "across", line="temperature", from_=["0.1 m", "0.25 m"], to=["0.9 m", "0.25 m"], points=5
    )
    slab.output(file="slab", fields=["temperature"])
    return slab


def test_a_line_writes_the_distance_the_coordinates_and_the_field_with_their_units(
    tmp_path, monkeypatch
):
    solution = conducting_slab(tmp_path, monkeypatch).solve(report="none")
    with open(tmp_path / "slab_across.csv") as file:
        rows = list(csv.reader(file))
    assert rows[0] == ["distance [m]", "x [m]", "y [m]", "temperature [K]"]
    values = numpy.array(rows[1:], dtype=float)
    assert values[:, 0] == pytest.approx([0.0, 0.2, 0.4, 0.6, 0.8], abs=1e-14)
    assert values[:, 1] == pytest.approx([0.1, 0.3, 0.5, 0.7, 0.9], rel=1e-14)
    assert values[:, 2] == pytest.approx(0.25, rel=1e-14)
    assert values[:, 3] == pytest.approx(300.0 + 100.0 * values[:, 1], rel=1e-12)
    assert solution.line("across")["temperature"] == pytest.approx(values[:, 3], rel=1e-15)


def test_a_transient_line_holds_a_row_for_each_point_at_each_output_time(tmp_path, monkeypatch):
    """The slab starts at its linear steady temperature, so it keeps it at every output time:
    the start and the two steps of 1 s."""
    slab = conducting_slab(tmp_path, monkeypatch)
    slab.properties(density="1 kg/m^3", specific_heat_capacity="1 J/(kg*K)")
    slab.initial_conditions(temperature="[300 K] + [100 K/m]*x")
    slab.time(end_time="2 s", initial_time_step="1 s", step_control={"by": "fixed"})
    table = slab.solve(report="none").line("across")
    assert table.names == ["time", "distance", "x", "y", "temperature"]
    assert table["time"] == pytest.approx(numpy.repeat([0.0, 1.0, 2.0], 5), abs=1e-14)
    assert table["temperature"] == pytest.approx(300.0 + 100.0 * table["x"], rel=1e-12)
    with open(tmp_path / "slab_across.csv") as file:
        assert len(list(csv.reader(file))) == 1 + 15


def test_a_line_without_the_output_section_is_refused(tmp_path, monkeypatch):
    slab = conducting_slab(tmp_path, monkeypatch)
    tree = slab.to_tree()
    del tree["output"]
    with pytest.raises(InputError, match=r"postprocessors\.across: .*output section"):
        schema.validate(tree)


def test_a_plane_of_symmetry_of_a_heated_slab_is_its_insulated_middle():
    """The half of a slab of width 2 L, held at T0 on both faces and heated by q, with a plane
    of symmetry at its middle: T(0) = T0 + q L^2 / (2 k), exact at the nodes."""
    half = dualmesh.problem(mesh={"segment": {"length": "0.1 m", "cells": 10}})
    half.physics("heat_conduction")
    half.properties(thermal_conductivity="45 W/(m*K)", volumetric_heat_source="1 MW/m^3")
    half.boundary_conditions("left", "symmetry")
    half.boundary_conditions("right", temperature="300 K")
    half.postprocessor("middle", point_value="temperature", point=["0 m"])
    solution = half.solve(report="none")
    assert solution.postprocessor("middle")[-1] == pytest.approx(
        300.0 + 1.0e6 * 0.1**2 / (2 * 45.0), rel=1e-12
    )


@pytest.mark.parametrize("method", ["FEM", "DMCDM"])
def test_planes_of_symmetry_of_a_pulled_bar_leave_its_exact_displacement(method):
    """A bar of 2 m by 0.5 m pulled by 100 MPa, a quarter of a bar of twice the size: the planes
    of symmetry x = 0 and y = 0 give u_x = sigma x / E and u_y = -nu sigma y / E."""
    bar = dualmesh.problem(
        mesh={"rectangle": {"size": ["2 m", "0.5 m"], "cells": [8, 2]}}, method=method
    )
    bar.physics("solid_mechanics", formulation="plane_stress")
    bar.properties(Young_modulus="200 GPa", Poisson_ratio=0.3)
    bar.boundary_conditions("left", "symmetry")
    bar.boundary_conditions("bottom", "symmetry")
    bar.boundary_conditions("right", traction=["100 MPa", "0 Pa"])
    solution = bar.solve(report="none")
    x, y = solution.coordinates("displacement").T
    displacement = solution.field("displacement")
    assert displacement[:, 0] == pytest.approx(1e8 * x / 2e11, rel=1e-9, abs=1e-18)
    assert displacement[:, 1] == pytest.approx(-0.3 * 1e8 * y / 2e11, rel=1e-9, abs=1e-18)


@pytest.mark.parametrize("method", ["FEM", "DMCDM"])
def test_planes_of_symmetry_of_a_channel_let_a_uniform_flow_slip(method):
    """A uniform inflow between two planes of symmetry: the walls hold no tangential traction, so
    the flow stays uniform, v = (U, 0), at every node."""
    channel = dualmesh.problem(
        mesh={"rectangle": {"size": ["2 m", "0.5 m"], "cells": [8, 4]}}, method=method
    )
    channel.physics("fluid_dynamics")
    channel.properties(density="1000 kg/m^3", dynamic_viscosity="1e-3 Pa*s")
    channel.boundary_conditions("left", velocity=["0.1 m/s", "0 m/s"])
    channel.boundary_conditions("bottom", "symmetry")
    channel.boundary_conditions("top", "symmetry")
    solution = channel.solve(report="none")
    velocity = solution.field("velocity")
    assert velocity[:, 0] == pytest.approx(0.1, rel=1e-8)
    assert velocity[:, 1] == pytest.approx(0.0, abs=1e-9)


def test_symmetry_takes_no_other_condition_on_its_boundary():
    slab = dualmesh.problem(mesh={"segment": {"length": "1 m", "cells": 2}})
    slab.boundary_conditions("left", "symmetry")
    with pytest.raises(InputError, match=r"boundary_conditions\.left: the word symmetry"):
        slab.boundary_conditions("left", temperature="300 K")
    with pytest.raises(InputError, match=r"boundary_conditions\.right: .*the word symmetry"):
        slab.boundary_conditions("right", "symetry")


def test_the_eigenvalues_of_a_cooling_rod_are_its_decay_rates():
    """A rod of length L held at its ends cools in the modes sin(n pi x / L), each decaying as
    exp(-alpha (n pi / L)^2 t) with the diffusivity alpha = k / (rho c_p)."""
    rod = dualmesh.problem(mesh={"segment": {"length": "0.1 m", "cells": 100}})
    rod.physics("heat_conduction")
    rod.properties(
        thermal_conductivity="45 W/(m*K)",
        density="7850 kg/m^3",
        specific_heat_capacity="490 J/(kg*K)",
    )
    rod.boundary_conditions("left", temperature="0 K")
    rod.boundary_conditions("right", temperature="0 K")
    rod.eigenvalues(count=3)
    solution = rod.solve(report="none")
    diffusivity = 45.0 / (7850.0 * 490.0)
    exact = diffusivity * (numpy.arange(1, 4) * numpy.pi / 0.1) ** 2
    # Linear elements with the consistent capacity overestimate lambda_n by (n pi h / L)^2 / 12.
    assert numpy.sort(numpy.real(solution.eigenvalues)) == pytest.approx(exact, rel=1e-3)
    with pytest.raises(InputError, match="natural frequencies belong to a vibration"):
        _ = solution.frequencies


def test_the_natural_frequencies_of_a_bar_are_those_of_its_axial_waves():
    """A bar of length L, fixed at one end and free at the other, held between two planes of
    symmetry with nu = 0, vibrates along its axis at f_n = (2 n - 1) c / (4 L), with the wave
    speed c = sqrt(E / rho). Its eigenvalues are lambda = i 2 pi f."""
    bar = dualmesh.problem(mesh={"rectangle": {"size": ["1 m", "0.05 m"], "cells": [80, 1]}})
    bar.physics("solid_mechanics", formulation="plane_stress")
    bar.properties(Young_modulus="200 GPa", Poisson_ratio=0, density="7850 kg/m^3")
    bar.boundary_conditions("left", displacement_x="0 m")
    bar.boundary_conditions("bottom", "symmetry")
    bar.boundary_conditions("top", "symmetry")
    bar.eigenvalues(count=2)
    solution = bar.solve(report="none")
    speed = numpy.sqrt(200e9 / 7850.0)
    exact = numpy.array([1.0, 3.0]) * speed / 4.0
    assert numpy.sort(solution.frequencies) == pytest.approx(exact, rel=1e-3)
    assert numpy.sort(numpy.imag(solution.eigenvalues)) == pytest.approx(
        2 * numpy.pi * numpy.sort(solution.frequencies), rel=1e-12
    )


def test_an_eigenvalue_study_takes_no_time_section_and_no_postprocessors():
    rod = dualmesh.problem(mesh={"segment": {"length": "1 m", "cells": 4}})
    rod.eigenvalues(count=2, near="1 1/s")
    with pytest.raises(InputError, match=r"^time: an eigenvalue study"):
        rod.time(end_time="1 s", initial_time_step="0.1 s")
    with pytest.raises(InputError, match=r"postprocessors\.hot: .*no postprocessors"):
        rod.postprocessor("hot", maximum="temperature")
    with pytest.raises(InputError, match=r"initial_conditions: an eigenvalue study"):
        rod.initial_conditions(temperature="300 K")


def slab_core(method):
    """A bare slab reactor of width 1 m in one-group diffusion, -D phi'' + Sigma_a phi =
    (1/k) nu Sigma_f phi, with zero flux at its faces: the source reads the eigenvalue as 1/k."""
    slab = dualmesh.problem(mesh={"segment": {"length": "1 m", "cells": 200}}, method=method)
    slab.model(
        "neutrons",
        unknown={"neutron_flux": {"unit": "1/(m^2*s)"}},
        flux="-diffusion_coefficient*grad(neutron_flux)",
        source="-absorption*neutron_flux + eigenvalue*fission_production*neutron_flux",
    )
    slab.properties(
        diffusion_coefficient={"unit": "m", "value": "1 cm"},
        absorption={"unit": "1/m", "value": 1.0},
        fission_production={"unit": "1/m", "value": 1.05},
    )
    slab.boundary_conditions("left", neutron_flux="0 1/(m^2*s)")
    slab.boundary_conditions("right", neutron_flux="0 1/(m^2*s)")
    return slab


@pytest.mark.parametrize("method", ["FEM", "DMCDM"])
def test_an_eigenvalue_without_a_time_derivative_is_the_number_of_a_criticality(method):
    """The slab has no time derivative, so the eigenvalue 1/k is a number, and near takes one.
    The mode sin(n pi x / L) has k_n = nu Sigma_f / (Sigma_a + D (n pi / L)^2), to the relative
    error (n pi h / L)^2 / 12 of a second-order method, which bounds that of k_n: 2.1e-5 and
    8.2e-5 for the first two modes on 200 cells (measured up to 1.8e-6 and 2.3e-5)."""
    slab = slab_core(method)
    slab.eigenvalues(count=2, near=0.5)
    multiplication = 1.0 / numpy.sort(slab.solve(report="none").eigenvalues)
    for mode, k in enumerate(multiplication, start=1):
        exact = 1.05 / (1.0 + 0.01 * (mode * numpy.pi) ** 2)
        assert k == pytest.approx(exact, rel=(mode * numpy.pi / 200) ** 2 / 12)


@pytest.mark.parametrize("method", ["FEM", "DMCDM"])
def test_an_eigenvalue_with_a_storage_is_a_rate(method):
    """A solute that diffuses with D and has a storage: the eigenvalue is in 1/s. With the source
    lambda c, the mode sin(pi x / L) decays as e^(-lambda t) with lambda + lambda = D (pi / L)^2,
    so the eigenvalue is half the decay rate of the diffusion alone."""
    rod = dualmesh.problem(mesh={"segment": {"length": "1 m", "cells": 200}}, method=method)
    rod.model(
        "decay",
        unknown={"solute": {"unit": "mol/m^3"}},
        storage="solute",
        flux="-diffusivity*grad(solute)",
        source="eigenvalue*solute",
    )
    rod.properties(diffusivity="1e-3 m^2/s")
    rod.boundary_conditions("left", solute="0 mol/m^3")
    rod.boundary_conditions("right", solute="0 mol/m^3")
    rod.eigenvalues(count=1, near="0 1/s")
    eigenvalue = rod.solve(report="none").eigenvalues[0]
    assert eigenvalue == pytest.approx(1e-3 * numpy.pi**2 / 2.0, rel=1e-4)


def test_an_eigenvalue_outside_the_source_or_the_study_is_refused():
    slab = slab_core("FEM")
    with pytest.raises(InputError, match=r"models\.neutrons\.source: .*eigenvalues section"):
        slab.solve(report="none")
    slab.model(
        "neutrons",
        unknown={"neutron_flux": {"unit": "1/(m^2*s)"}},
        flux="-eigenvalue*diffusion_coefficient*grad(neutron_flux)",
        source="-absorption*neutron_flux",
    )
    slab.eigenvalues(count=1)
    with pytest.raises(InputError, match=r"models\.neutrons\.flux: .*in the source of a model"):
        slab.solve(report="none")
    with pytest.raises(InputError, match=r"properties\.eigenvalue: .*other than"):
        slab.properties(eigenvalue={"unit": "1", "value": 1.0})


def test_an_initial_temperature_given_by_a_formula_starts_the_run():
    """The slab starts at the steady temperature of its faces, 300 K + (50 K/m) x, so it keeps it
    exactly. A constant start of 300 K would not."""
    slab = dualmesh.problem(mesh={"segment": {"length": "0.1 m", "cells": 10}})
    slab.physics("heat_conduction")
    slab.properties(
        thermal_conductivity="45 W/(m*K)",
        density="7850 kg/m^3",
        specific_heat_capacity="490 J/(kg*K)",
    )
    slab.initial_conditions(temperature="[300 K] + [50 K/m]*x")
    slab.boundary_conditions("left", temperature="300 K")
    slab.boundary_conditions("right", temperature="305 K")
    slab.time(end_time="10 s", initial_time_step="1 s", step_control={"by": "fixed"})
    solution = slab.solve(report="none")
    x = solution.coordinates("temperature")[:, 0]
    assert solution.field("temperature")[:, 0] == pytest.approx(300.0 + 50.0 * x, rel=1e-12)


def test_an_initial_condition_reads_the_coordinates_only():
    slab = dualmesh.problem(mesh={"segment": {"length": "1 m", "cells": 2}})
    with pytest.raises(InputError, match=r"initial_conditions\.temperature: .*reads time"):
        slab.initial_conditions(temperature="[300 K] + [1 K/s]*time")


def test_a_rigid_rotation_leaves_a_solid_at_finite_strain_free_of_stress():
    """A square starts rotated by 0.5 rad, at rest and free of load: at finite strain it stays
    rotated and free of stress, to round-off. The small strain of the same rotation is
    (cos 0.5 - 1) on the diagonal, which would give a stress of about E/8."""
    block = dualmesh.problem(mesh={"rectangle": {"size": ["1 m", "1 m"], "cells": [2, 2]}})
    block.physics("solid_mechanics", formulation="plane_strain", kinematics="finite")
    block.properties(Young_modulus="200 GPa", Poisson_ratio=0.3, density="7850 kg/m^3")
    rotation = ["(cos(0.5) - 1)*x - sin(0.5)*y", "sin(0.5)*x + (cos(0.5) - 1)*y"]
    block.initial_conditions(displacement=rotation)
    block.time(end_time="1e-4 s", initial_time_step="1e-4 s", step_control={"by": "fixed"})
    block.postprocessor("stress_xx", point_value="stress_xx", point=["0.3 m", "0.6 m"])
    block.postprocessor("stress_xy", point_value="stress_xy", point=["0.3 m", "0.6 m"])
    solution = block.solve(report="none")
    x, y = solution.coordinates("displacement").T
    expected = numpy.column_stack(
        [
            (numpy.cos(0.5) - 1) * x - numpy.sin(0.5) * y,
            numpy.sin(0.5) * x + (numpy.cos(0.5) - 1) * y,
        ]
    )
    assert solution.field("displacement") == pytest.approx(expected, abs=1e-12)
    for name in ("stress_xx", "stress_xy"):
        assert numpy.abs(solution.postprocessor(name)).max() < 1e-12 * 200e9


@pytest.mark.parametrize("method", ["FEM", "DMCDM"])
def test_a_bar_beyond_yield_has_the_strain_of_its_hardening(method):
    """A plate in plane stress pulled by 300 MPa, with the yield stress 250 MPa + (1 GPa) p of
    the equivalent plastic strain p: p = 0.05, so the axial strain is sigma / E + p and the
    lateral strain -nu sigma / E - p / 2, with the plastic flow at constant volume."""
    plate = dualmesh.problem(
        mesh={"rectangle": {"size": ["1 mm", "1 mm"], "cells": [2, 2]}}, method=method
    )
    plate.physics("solid_mechanics", formulation="plane_stress")
    plate.properties(
        Young_modulus="200 GPa",
        Poisson_ratio=0.3,
        yield_stress="[250 MPa] + [1 GPa]*equivalent_plastic_strain",
    )
    plate.boundary_conditions("left", "symmetry")
    plate.boundary_conditions("bottom", "symmetry")
    plate.boundary_conditions("right", traction=["300 MPa", "0 Pa"])
    solution = plate.solve(report="none")
    x, y = solution.coordinates("displacement").T
    displacement = solution.field("displacement")
    assert displacement[:, 0] == pytest.approx((300e6 / 200e9 + 0.05) * x, rel=1e-8, abs=1e-18)
    assert displacement[:, 1] == pytest.approx(
        (-0.3 * 300e6 / 200e9 - 0.025) * y, rel=1e-8, abs=1e-18
    )


def test_a_plastic_solid_at_finite_strain_is_refused_with_the_limit():
    block = dualmesh.problem(mesh={"rectangle": {"size": ["1 m", "1 m"], "cells": [1, 1]}})
    block.physics("solid_mechanics", formulation="plane_strain", kinematics="finite")
    block.properties(Young_modulus="200 GPa", Poisson_ratio=0.3, yield_stress="250 MPa")
    block.boundary_conditions("left", "symmetry")
    with pytest.raises(InputError, match=r"properties\.yield_stress: a plastic solid .*small"):
        block.solve(report="none")


def pulled_bar(method, formulation):
    """A bar of 2 m by 0.5 m pulled by 100 MPa on its right side, a quarter of a bar of twice the
    size between two planes of symmetry: its stress is uniaxial, sigma_xx = 100 MPa."""
    bar = dualmesh.problem(
        mesh={"rectangle": {"size": ["2 m", "0.5 m"], "cells": [8, 2]}}, method=method
    )
    bar.physics("solid_mechanics", formulation=formulation)
    bar.properties(Young_modulus="200 GPa", Poisson_ratio=0.3)
    bar.boundary_conditions("left", "symmetry")
    bar.boundary_conditions("bottom", "symmetry")
    bar.boundary_conditions("right", traction=["100 MPa", "0 Pa"])
    return bar


@pytest.mark.parametrize("method", ["FEM", "DMCDM"])
def test_the_traction_over_a_boundary_is_the_load_there_and_the_reaction_of_a_support(method):
    """The traction over the loaded side is the load, 100 MPa times 0.5 m per unit depth, and over
    the plane of symmetry it is the reaction, which balances the load. The vector gives one
    postprocessor per component."""
    bar = pulled_bar(method, "plane_stress")
    bar.postprocessor("pull", boundary_flux="traction_x", boundary="right")
    bar.postprocessor("support", boundary_flux="traction", boundary="left")
    solution = bar.solve(report="none")
    assert solution.postprocessor("pull")[-1] == pytest.approx(5e7, rel=1e-12)
    assert solution.unit("pull") == "N/m"
    assert solution.postprocessor("support_x")[-1] == pytest.approx(-5e7, rel=1e-9)
    assert solution.postprocessor("support_y")[-1] == pytest.approx(0.0, abs=1e-9 * 5e7)


def test_a_traction_needs_a_solid_and_takes_the_components_of_the_mesh():
    slab = conducting_slab_problem()
    slab.postprocessor("force", boundary_flux="traction_x", boundary="left")
    with pytest.raises(InputError, match=r"postprocessors\.force: .*needs solid_mechanics"):
        slab.solve(report="none")
    with pytest.raises(InputError, match=r"postprocessors\.force\.boundary_flux: "):
        slab.postprocessor("force", boundary_flux="traction_z", boundary="left")


def conducting_slab_problem():
    slab = dualmesh.problem(mesh={"rectangle": {"size": ["1 m", "0.5 m"], "cells": [2, 2]}})
    slab.physics("heat_conduction")
    slab.properties(thermal_conductivity="1 W/(m*K)")
    slab.boundary_conditions("left", temperature="300 K")
    return slab


@pytest.mark.parametrize("method", ["FEM", "DMCDM"])
@pytest.mark.parametrize(
    "formulation, factor", [("plane_stress", 1.0), ("plane_strain", numpy.sqrt(0.79))]
)
def test_the_von_Mises_stress_of_a_pulled_bar(method, formulation, factor):
    """In plane stress the von Mises stress of the uniaxial stress sigma is sigma. In plane strain
    the bar also carries sigma_zz = nu sigma, and the von Mises stress is
    sigma sqrt(1 - nu + nu^2) = sigma sqrt(0.79) for nu = 0.3."""
    bar = pulled_bar(method, formulation)
    bar.postprocessor("equivalent", point_value="von_Mises_stress", point=["1.3 m", "0.2 m"])
    solution = bar.solve(report="none")
    assert solution.postprocessor("equivalent")[-1] == pytest.approx(factor * 1e8, rel=1e-9)
    assert solution.unit("equivalent") == "Pa"


def test_a_von_Mises_stress_is_a_value_at_points():
    bar = pulled_bar("FEM", "plane_stress")
    bar.postprocessor("largest", maximum="von_Mises_stress")
    with pytest.raises(InputError, match=r"postprocessors\.largest: a stress is available at"):
        bar.solve(report="none")
