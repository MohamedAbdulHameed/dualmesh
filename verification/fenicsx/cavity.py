# SPDX-License-Identifier: LGPL-2.1-or-later
"""Lid-driven cavity at Re = 1000, solved by FEniCSx with the Taylor-Hood element.

This is the FEniCSx counterpart of ``verification/benchmarks/fluids/run_cavity.py``.
The lid y = 1 of the unit cavity moves at u = 1 and the other sides are at
rest. The top corners belong to the walls. The density is 1 and the viscosity
is 1/Re. Newton's method reaches Re = 1000 through the Reynolds numbers 100
and 400, each solve starting from the last. The pressure is zero at the
origin.

The result is the largest difference of u along x = 0.5 and of v along
y = 0.5 from the 601 x 601 solution of Erturk, Corke and Gokcol (2005), on
meshes of (n/2) x (n/2) quadrilaterals with Q2 velocity and Q1 pressure.

    mpirun -n 4 python cavity.py
"""

import sys
import time
from pathlib import Path

import dolfinx
import numpy
import ufl
from basix.ufl import element, mixed_element
from common import evaluate, newton_solver, save, unknowns
from dolfinx.fem.petsc import NonlinearProblem
from mpi4py import MPI

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks" / "fluids"))
import erturk2005  # noqa: E402

Reynolds_steps = [100.0, 400.0, 1000.0]
levels = [32, 64, 128, 256]


def lid_velocity(x):
    # The lid moves except at its two corners, which belong to the walls.
    on_side = numpy.isclose(x[0], 0.0) | numpy.isclose(x[0], 1.0)
    return numpy.vstack((numpy.where(on_side, 0.0, 1.0), numpy.zeros_like(x[0])))


def solve(n, comm):
    mesh = dolfinx.mesh.create_unit_square(
        comm, n // 2, n // 2, dolfinx.mesh.CellType.quadrilateral
    )
    velocity_element = element("Lagrange", mesh.basix_cell(), 2, shape=(2,))
    pressure_element = element("Lagrange", mesh.basix_cell(), 1)
    space = dolfinx.fem.functionspace(mesh, mixed_element([velocity_element, pressure_element]))
    velocity_space, _ = space.sub(0).collapse()
    pressure_space, _ = space.sub(1).collapse()

    dimension = mesh.topology.dim
    lid_facets = dolfinx.mesh.locate_entities_boundary(
        mesh, dimension - 1, lambda x: numpy.isclose(x[1], 1.0)
    )
    wall_facets = dolfinx.mesh.locate_entities_boundary(
        mesh,
        dimension - 1,
        lambda x: numpy.isclose(x[0], 0.0) | numpy.isclose(x[0], 1.0) | numpy.isclose(x[1], 0.0),
    )
    lid = dolfinx.fem.Function(velocity_space)
    lid.interpolate(lid_velocity)
    rest = dolfinx.fem.Function(velocity_space)
    zero_pressure = dolfinx.fem.Function(pressure_space)
    origin = dolfinx.fem.locate_dofs_geometrical(
        (space.sub(1), pressure_space),
        lambda x: numpy.isclose(x[0], 0.0) & numpy.isclose(x[1], 0.0),
    )
    conditions = [
        dolfinx.fem.dirichletbc(
            lid,
            dolfinx.fem.locate_dofs_topological(
                (space.sub(0), velocity_space), dimension - 1, lid_facets
            ),
            space.sub(0),
        ),
        dolfinx.fem.dirichletbc(
            rest,
            dolfinx.fem.locate_dofs_topological(
                (space.sub(0), velocity_space), dimension - 1, wall_facets
            ),
            space.sub(0),
        ),
        dolfinx.fem.dirichletbc(zero_pressure, origin, space.sub(1)),
    ]

    viscosity = dolfinx.fem.Constant(mesh, 1.0 / Reynolds_steps[0])
    state = dolfinx.fem.Function(space)
    velocity, pressure = ufl.split(state)
    velocity_test, pressure_test = ufl.TestFunctions(space)
    residual = (
        viscosity * ufl.inner(ufl.grad(velocity), ufl.grad(velocity_test))
        + ufl.inner(ufl.grad(velocity) * velocity, velocity_test)
        - pressure * ufl.div(velocity_test)
        - pressure_test * ufl.div(velocity)
    ) * ufl.dx
    problem = NonlinearProblem(residual, state, bcs=conditions)
    solver = newton_solver(problem, comm)

    iterations = 0
    start = time.perf_counter()
    for Reynolds in Reynolds_steps:
        viscosity.value = 1.0 / Reynolds
        steps, converged = solver.solve(state)
        if not converged:
            raise RuntimeError(f"Newton's method did not converge at Re = {Reynolds}, n = {n}.")
        iterations += steps
    wall_time = comm.allreduce(time.perf_counter() - start, op=MPI.MAX)

    velocity_solution = state.sub(0).collapse()
    vertical = numpy.column_stack((numpy.full(len(erturk2005.Y), 0.5), erturk2005.Y))
    horizontal = numpy.column_stack((erturk2005.X, numpy.full(len(erturk2005.X), 0.5)))
    u = evaluate(velocity_solution, vertical)[:, 0]
    v = evaluate(velocity_solution, horizontal)[:, 1]
    return {
        "n": n,
        "unknowns": unknowns(space),
        "largest_difference_u": float(numpy.max(numpy.abs(u - erturk2005.U))),
        "largest_difference_v": float(numpy.max(numpy.abs(v - erturk2005.V))),
        "u": u.tolist(),
        "v": v.tolist(),
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
            print({key: value for key, value in result.items() if key not in ("u", "v")})
        results.append(result)
    save("cavity", {"element": "Q2-Q1 Taylor-Hood", "levels": results}, comm)


if __name__ == "__main__":
    main()
