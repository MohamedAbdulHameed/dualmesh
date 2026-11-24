# SPDX-License-Identifier: LGPL-2.1-or-later
"""Kovasznay flow at Re = 40, solved by FEniCSx with the Taylor-Hood element.

This is the FEniCSx counterpart of ``verification/benchmarks/fluids/run_Kovasznay.py``.
The domain is [-0.5, 1] x [-0.5, 1.5], the density is 1 and the viscosity is
1/Re. The exact velocity holds on the left, bottom and top sides, and the
exact traction of the Laplacian form holds on the outflow side x = 1. Newton's
method starts from the exact fields, as in the dualmesh script.

The meshes have (n/2) x round(2n/3) quadrilaterals with Q2 velocity and Q1
pressure, so that they have about the nodes of the n x round(4n/3) meshes of
linear quadrilaterals of the dualmesh script. The comparison is made per
unknown.

    mpirun -n 4 python Kovasznay.py
"""

import math
import time

import dolfinx
import numpy
import ufl
from basix.ufl import element, mixed_element
from common import newton_solver, save, total, unknowns
from dolfinx.fem.petsc import NonlinearProblem
from mpi4py import MPI

Reynolds_number = 40.0
viscosity = 1.0 / Reynolds_number
decay = Reynolds_number / 2 - math.sqrt(Reynolds_number**2 / 4 + 4 * math.pi**2)
levels = [16, 32, 64, 128]
x_minimum, x_maximum, y_minimum, y_maximum = -0.5, 1.0, -0.5, 1.5


def exact_velocity(x):
    return numpy.vstack(
        (
            1 - numpy.exp(decay * x[0]) * numpy.cos(2 * numpy.pi * x[1]),
            decay / (2 * numpy.pi) * numpy.exp(decay * x[0]) * numpy.sin(2 * numpy.pi * x[1]),
        )
    )


def exact_pressure(x):
    return (1 - numpy.exp(2 * decay * x[0])) / 2


def solve(n, comm):
    cells_x, cells_y = n // 2, round(2 * n / 3)
    mesh = dolfinx.mesh.create_rectangle(
        comm,
        [numpy.array([x_minimum, y_minimum]), numpy.array([x_maximum, y_maximum])],
        [cells_x, cells_y],
        dolfinx.mesh.CellType.quadrilateral,
    )
    velocity_element = element("Lagrange", mesh.basix_cell(), 2, shape=(2,))
    pressure_element = element("Lagrange", mesh.basix_cell(), 1)
    space = dolfinx.fem.functionspace(mesh, mixed_element([velocity_element, pressure_element]))
    velocity_space, _ = space.sub(0).collapse()

    # The exact fields as UFL expressions, for the traction and the errors.
    x = ufl.SpatialCoordinate(mesh)
    velocity_exact = ufl.as_vector(
        (
            1 - ufl.exp(decay * x[0]) * ufl.cos(2 * ufl.pi * x[1]),
            decay / (2 * ufl.pi) * ufl.exp(decay * x[0]) * ufl.sin(2 * ufl.pi * x[1]),
        )
    )
    pressure_exact = (1 - ufl.exp(2 * decay * x[0])) / 2

    dimension = mesh.topology.dim
    inflow_facets = dolfinx.mesh.locate_entities_boundary(
        mesh,
        dimension - 1,
        lambda x: (
            numpy.isclose(x[0], x_minimum)
            | numpy.isclose(x[1], y_minimum)
            | numpy.isclose(x[1], y_maximum)
        ),
    )
    outflow_facets = dolfinx.mesh.locate_entities_boundary(
        mesh, dimension - 1, lambda x: numpy.isclose(x[0], x_maximum)
    )
    boundary_velocity = dolfinx.fem.Function(velocity_space)
    boundary_velocity.interpolate(exact_velocity)
    inflow_degrees_of_freedom = dolfinx.fem.locate_dofs_topological(
        (space.sub(0), velocity_space), dimension - 1, inflow_facets
    )
    condition = dolfinx.fem.dirichletbc(boundary_velocity, inflow_degrees_of_freedom, space.sub(0))
    tags = dolfinx.mesh.meshtags(
        mesh, dimension - 1, outflow_facets, numpy.full_like(outflow_facets, 1)
    )
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=tags)
    dx = ufl.Measure("dx", domain=mesh, metadata={"quadrature_degree": 8})

    state = dolfinx.fem.Function(space)
    velocity, pressure = ufl.split(state)
    velocity_test, pressure_test = ufl.TestFunctions(space)
    normal = ufl.FacetNormal(mesh)
    traction = viscosity * ufl.grad(velocity_exact) * normal - pressure_exact * normal
    residual = (
        viscosity * ufl.inner(ufl.grad(velocity), ufl.grad(velocity_test))
        + ufl.inner(ufl.grad(velocity) * velocity, velocity_test)
        - pressure * ufl.div(velocity_test)
        - pressure_test * ufl.div(velocity)
    ) * ufl.dx - ufl.inner(traction, velocity_test) * ds(1)

    state.sub(0).interpolate(exact_velocity)
    state.sub(1).interpolate(exact_pressure)
    problem = NonlinearProblem(residual, state, bcs=[condition])
    solver = newton_solver(problem, comm)
    start = time.perf_counter()
    iterations, converged = solver.solve(state)
    wall_time = comm.allreduce(time.perf_counter() - start, op=MPI.MAX)
    if not converged:
        raise RuntimeError(f"Newton's method did not converge for n = {n}.")

    velocity_error = velocity - velocity_exact
    pressure_error = pressure - pressure_exact
    area = (x_maximum - x_minimum) * (y_maximum - y_minimum)
    return {
        "n": n,
        "h": math.sqrt(area / (cells_x * cells_y)),
        "unknowns": unknowns(space),
        "velocity_L2": math.sqrt(total(ufl.inner(velocity_error, velocity_error) * dx)),
        "velocity_H1_seminorm": math.sqrt(
            total(ufl.inner(ufl.grad(velocity_error), ufl.grad(velocity_error)) * dx)
        ),
        "pressure_L2": math.sqrt(total(pressure_error**2 * dx)),
        "newton_iterations": iterations,
        "wall_time": wall_time,
        "ranks": comm.size,
    }


def main():
    comm = MPI.COMM_WORLD
    results = []
    for n in levels:
        result = solve(n, comm)
        if comm.rank == 0:
            print(result)
        results.append(result)
    save("kovasznay", {"element": "Q2-Q1 Taylor-Hood", "levels": results}, comm)


if __name__ == "__main__":
    main()
