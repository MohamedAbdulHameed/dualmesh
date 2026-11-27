# SPDX-License-Identifier: LGPL-2.1-or-later
"""The Schur complement split of a flow, and the split by physics.

The Jacobian of a pressure-velocity flow is a saddle point matrix
[[F, G], [D, C]] (momentum, pressure gradient, divergence, pressure block).
PETSc's Schur complement field split converges in a number of flexible GMRES
iterations that depends only on how well its approximation of the Schur
complement S = C - D F^{-1} G is: the pressure mass matrix scaled by 1/mu
(``-dualmesh_schur pressure_mass``, exact in the Stokes limit), the pressure
convection-diffusion operator (``pcd``) or the least squares commutator
(``lsc``), see SaddlePointBlocks.h.  The block solves are exact here, so that
the counts measure the approximation alone.  The tests check

* that the iteration count does not grow with the mesh for the Taylor-Hood
  element and for the stabilised equal-order element, for every
  approximation;
* that the counts do not depend on the unit of length;
* that the answer is the one of the direct solver;
* that the split by physics solves a one-way coupling exactly;
* that misuse is refused with a message that says what to do.
"""

from __future__ import annotations

import dualmesh
import numpy
import pytest
from helpers import lid_driven_cavity

direct = "-ksp_type preonly -pc_type lu"


def exact_split(approximation):
    """The Schur split with the given approximation, and exact solves of its blocks."""
    return (
        f"-dualmesh_schur {approximation} -fieldsplit_0_pc_type lu "
        "-fieldsplit_1_laplacian_pc_type lu -fieldsplit_1_pc_type lu"
    )


def _cavity(formulation, element_type, n):
    return lid_driven_cavity("fem", n, element_type, density=0.0, formulation=formulation)


def _relative_difference(problem, reference):
    return max(
        numpy.abs(problem.values(v) - reference.values(v)).max()
        / max(numpy.abs(reference.values(v)).max(), 1e-300)
        for v in ("velocity_x", "velocity_y", "pressure")
    )


# Iterations per Newton step at a relative tolerance of 1e-12 on 8 x 8 to 32 x 32 meshes (the
# Taylor-Hood meshes have half as many elements per side), with some margin, in the norm that
# measures each equation against the size of its terms.  Measured: 17-19 (pressure_mass,
# Taylor-Hood), 17-20 (pressure_mass, equal order), 24-33 (pcd, Taylor-Hood), 15-18 (pcd, equal
# order), 12-19 (lsc, Taylor-Hood) and 17-20 (lsc, equal order).  lsc on Taylor-Hood grows by
# 60 % over two refinements, the others by at most a third.
@pytest.mark.parametrize(
    "formulation, element_type, preconditioner, limit, growth",
    [
        ("Taylor-Hood", "Quad9", "pressure_mass", 22, 1.5),
        ("equal_order", "Quad4", "pressure_mass", 22, 1.5),
        ("Taylor-Hood", "Quad9", "pcd", 36, 1.5),
        ("equal_order", "Quad4", "pcd", 34, 1.5),
        ("Taylor-Hood", "Quad9", "lsc", 22, 1.65),
        ("equal_order", "Quad4", "lsc", 34, 1.5),
    ],
)
def test_iterations_do_not_grow_with_the_mesh(
    formulation, element_type, preconditioner, limit, growth
):
    counts = []
    for n in (8, 16, 32):
        reference = _cavity(formulation, element_type, n)
        reference.solve(petsc_options=direct)
        problem = _cavity(formulation, element_type, n)
        result = problem.solve(petsc_options=exact_split(preconditioner))
        counts.append(result.linear_iterations / result.total_iterations)
        assert _relative_difference(problem, reference) < 1e-8
    assert max(counts) <= limit, counts
    assert max(counts) <= growth * min(counts), counts


def _pipe(radius):
    """Developing flow in a pipe of the given radius and four radii long, at Re = 50 whatever the
    radius: rho = 50, U = 1, mu = radius."""
    mesh = dualmesh.generate_rectangle_mesh(x=[0, radius], y=[0, 4 * radius], num_elements=[8, 32])
    problem = dualmesh.Problem(mesh, method="fem", coordinates="axisymmetric")
    flow = problem.add_physics(
        "fluid_dynamics", density=50.0, dynamic_viscosity=radius, formulation="equal_order"
    )
    flow.add_boundary_condition(["velocity_r = 0", "velocity_z = 1"], boundary="bottom")
    flow.add_boundary_condition(["velocity_r = 0", "velocity_z = 0"], boundary="right")
    flow.add_boundary_condition("velocity_r = 0", boundary="left")
    return problem


@pytest.mark.parametrize("preconditioner", ["pcd", "lsc"])
def test_the_iterations_do_not_depend_on_the_unit_of_length(preconditioner):
    """The same axisymmetric pipe flow in metres and in millimetres.  The element size of the
    stabilisation is that of the element itself, without the factor 2 pi r of the coordinate
    system, so the counts are the same (measured 40.4 for pcd and 45.2 for lsc)."""
    counts = []
    for radius in (1.0, 1e-3):
        reference = _pipe(radius)
        reference.solve(petsc_options=direct, report="none")
        problem = _pipe(radius)
        result = problem.solve(
            petsc_options=exact_split(preconditioner), linear_tolerance=1e-10, report="none"
        )
        counts.append(result.linear_iterations / result.total_iterations)
        expected = reference.values("velocity_z")
        assert numpy.abs(problem.values("velocity_z") - expected).max() < 1e-8
    assert abs(counts[0] - counts[1]) <= 1, counts
    assert max(counts) <= 65, counts


def test_it_needs_a_mass_conservation_equation():
    mesh = dualmesh.generate_rectangle_mesh(0, 1, 0, 1, 4, 4)
    problem = dualmesh.Problem(mesh, method="fem")
    problem.add_variable("temperature")
    problem.add_kernel("diffusion", variable="temperature")
    problem.add_kernel("body_force", variable="temperature", value=1.0)
    problem.add_boundary_condition("Dirichlet", "left", variable="temperature", value=0.0)
    with pytest.raises(ValueError, match="mass_conservation"):
        problem.solve(petsc_options="-dualmesh_schur pressure_mass")


def test_the_split_by_physics_is_exact_for_a_one_way_coupling_with_exact_block_solves():
    """The temperature drives the displacements and not the other way, so the block lower
    triangle with the temperature first is the matrix itself.  With exact solves of the two
    blocks, GMRES converges in one iteration."""
    mesh = dualmesh.generate_rectangle_mesh(x=[0.0, 0.1], y=[0.0, 0.02], num_elements=[10, 2])
    problem = dualmesh.Problem(mesh)
    heat = problem.add_physics("heat_conduction", thermal_conductivity=20.0, heat_source=1.0e6)
    heat.add_boundary_condition("temperature = 300", boundary="left")
    solid = problem.add_physics(
        "solid_mechanics",
        formulation="plane_stress",
        youngs_modulus=2.0e11,
        poissons_ratio=1 / 3,
        thickness=0.01,
    )
    solid.add_boundary_condition("fixed", boundary="left")
    problem.add_coupling(
        "thermal_expansion",
        heat_conduction=heat,
        solid_mechanics=solid,
        linear_expansion_coefficient=1.2e-5,
        stress_free_temperature=293.0,
    )
    result = problem.solve(
        petsc_options="-dualmesh_split physics -fieldsplit_0_pc_type lu -fieldsplit_1_pc_type lu",
        report="none",
    )
    assert result.linear_iterations == 1
