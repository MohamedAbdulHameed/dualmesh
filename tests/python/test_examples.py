# SPDX-License-Identifier: LGPL-2.1-or-later
"""The examples on the input give the values that they cite: the tip displacements of the bar in
large deformation of Reddy's book, the stress of Kirsch at a hole in a plate under tension, the
Nusselt numbers and the velocities of de Vahl Davis in a heated cavity, the multiplication factor
of two-group theory of a bare reactor, and the reaction and the bending stress of beam theory of
a wrench. Each test runs the problem of its example, loaded from the file, so the example and its
test cannot drift apart."""

import importlib.util
from pathlib import Path

import pytest

examples = Path(__file__).resolve().parents[2] / "examples"


def example(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_large_deformation_bar_has_the_tip_displacements_of_Reddy():
    """Example 6.2.4 of Reddy's book, on its eight elements: the book stops its iterations at a
    tolerance of 1e-3, and the tip displacements agree with it to 1e-3 m."""
    bar = example(examples / "large_deformation_bar.py")
    for load, published in bar.published_tip_displacements.items():
        solution = bar.bar(load).solve(report="none")
        assert solution.field("axial_displacement")[:, 0].max() == pytest.approx(
            published, abs=1e-3
        )


def test_the_plate_with_a_hole_has_the_stress_of_Kirsch(tmp_path, monkeypatch):
    """The quarter of the plate, with its outer radius 20 times that of the hole: the stress at
    the top of the hole is within 2 % of 3 sigma, and the stress along the y axis, from one hole
    radius above the hole, within 1 % of the solution of Kirsch (measured: 1.6 % and 0.9 %)."""
    monkeypatch.chdir(tmp_path)
    plate = example(examples / "plate_with_hole" / "plate_with_hole.py")
    solution = plate.plate().solve(report="none")
    assert solution.postprocessor("peak")[-1] == pytest.approx(3.0 * plate.remote_stress, rel=2e-2)
    ligament = solution.line("ligament")
    assert ligament["stress_xx"][1:] == pytest.approx(plate.Kirsch(ligament["y"][1:]), rel=1e-2)
    assert (tmp_path / "plate_with_hole_ligament.csv").exists()


@pytest.mark.parametrize("method", ["FEM", "DMCDM"])
@pytest.mark.parametrize("Rayleigh_number", [1.0e3, 1.0e4, 1.0e5])
def test_natural_convection_has_the_Nusselt_numbers_of_de_Vahl_Davis(
    tmp_path, monkeypatch, method, Rayleigh_number
):
    """On the mesh of 80 by 80 cells clustered at the walls, the average Nusselt number is within
    0.5 % of the benchmark and the largest velocities on the centre lines within 2 % (measured:
    up to 0.18 % and 0.19 %, at Ra = 10^5)."""
    monkeypatch.chdir(tmp_path)
    convection = example(examples / "natural_convection" / "natural_convection.py")
    solution = convection.cavity(Rayleigh_number, method).solve(report="none")
    computed = convection.measure(solution, Rayleigh_number)
    reference = convection.benchmark[Rayleigh_number]
    assert computed["nusselt"] == pytest.approx(reference["nusselt"], rel=5e-3)
    assert computed["u_max"] == pytest.approx(reference["u_max"], rel=2e-2)
    assert computed["v_max"] == pytest.approx(reference["v_max"], rel=2e-2)


def test_the_reactor_has_the_multiplication_factor_of_two_group_theory():
    """The bare core on 40 by 40 cells: k within 2e-5 of two-group theory (measured 1.4e-5)."""
    reactor = example(examples / "reactor_criticality.py")
    assert reactor.criticality(40) == pytest.approx(reactor.exact_criticality(), rel=2e-5)


def test_the_wrench_balances_its_load_and_has_the_bending_stress_of_beam_theory(
    tmp_path, monkeypatch
):
    """The reaction of the jaws balances the 150 N of the hand to the tolerance of the linear
    solve. At x = 60 mm the bending stress is within 0.1 % of beam theory, and the von Mises
    stress at the edge of the handle within 0.2 % (measured 0.002 % and 0.06 %)."""
    monkeypatch.chdir(tmp_path)
    wrench = example(examples / "wrench" / "wrench.py")
    solution = wrench.wrench().solve(report="none")
    assert solution.postprocessor("reaction")[-1] == pytest.approx(wrench.force, rel=1e-8)
    assert wrench.bending_stress(solution) == pytest.approx(wrench.beam_stress(), rel=1e-3)
    assert solution.postprocessor("edge")[-1] == pytest.approx(wrench.beam_stress(), rel=2e-3)
