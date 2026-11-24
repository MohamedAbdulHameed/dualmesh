# SPDX-License-Identifier: LGPL-2.1-or-later
"""DFG 2D-1, the steady flow around a cylinder at Re = 20, solved by FEniCSx.

This is the FEniCSx counterpart of
``verification/benchmarks/fluids/run_dfg_steady.py``, on the same Gmsh meshes
(``dfg_mesh.py``) with quadratic triangles whose sides on the cylinder lie on
the circle. The velocity is P2 and the pressure P1 (Taylor-Hood).

The inflow is u = 4 U y (0.41 - y) / 0.41^2 with U = 0.3, the walls and the
cylinder hold the fluid at rest, and the outflow is the natural condition of
the Laplacian form. The density is 1 and the viscosity is 0.001.

The force on the cylinder is minus the reaction of its prescribed velocity:
the residual of the momentum equation tested with a function that is a unit
vector on the cylinder and zero elsewhere on the boundary. The drag and lift
coefficients are C = 2 F / (U_mean^2 D) with U_mean = 0.2 and D = 0.1. The
pressure difference is p(0.15, 0.2) - p(0.25, 0.2). The reference values of
the FEATFLOW benchmark pages are C_D = 5.57953523384, C_L = 0.010618948146
and Delta p = 0.11752016697.

    mpirun -n 4 python dfg_steady.py
"""

import sys
import time
from pathlib import Path

import dolfinx
import dolfinx.io.gmshio
import numpy
import ufl
from basix.ufl import element, mixed_element
from common import evaluate, newton_solver, results_directory, save, total, unknowns
from dolfinx.fem.petsc import NonlinearProblem
from mpi4py import MPI

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks" / "fluids"))
import dfg_mesh  # noqa: E402

reference = {"drag": 5.57953523384, "lift": 0.010618948146, "pressure_difference": 0.11752016697}
sizes = [0.02, 0.01, 0.005, 0.0025]  # Element size on the cylinder
viscosity = 1e-3
coefficient_scale = 2.0 / (0.2**2 * 0.1)  # 2 / (U_mean^2 D)
inlet, outlet, walls, cylinder = 2, 3, 4, 5  # Physical groups of dfg_mesh.py


def inflow_velocity(x):
    return numpy.vstack((4 * 0.3 * x[1] * (0.41 - x[1]) / 0.41**2, numpy.zeros_like(x[0])))


def solve(size, comm):
    path = results_directory / f"dfg_order_2_h_{size}.msh"
    if comm.rank == 0:
        results_directory.mkdir(exist_ok=True)
        dfg_mesh.build(size, path, order=2)
    comm.barrier()
    mesh, _, facet_tags = dolfinx.io.gmshio.read_from_msh(str(path), comm, 0, gdim=2)

    velocity_element = element("Lagrange", mesh.basix_cell(), 2, shape=(2,))
    pressure_element = element("Lagrange", mesh.basix_cell(), 1)
    space = dolfinx.fem.functionspace(mesh, mixed_element([velocity_element, pressure_element]))
    velocity_space, _ = space.sub(0).collapse()
    dimension = mesh.topology.dim

    def velocity_degrees_of_freedom(tag):
        return dolfinx.fem.locate_dofs_topological(
            (space.sub(0), velocity_space), dimension - 1, facet_tags.find(tag)
        )

    inflow = dolfinx.fem.Function(velocity_space)
    inflow.interpolate(inflow_velocity)
    rest = dolfinx.fem.Function(velocity_space)
    conditions = [
        dolfinx.fem.dirichletbc(inflow, velocity_degrees_of_freedom(inlet), space.sub(0)),
        dolfinx.fem.dirichletbc(rest, velocity_degrees_of_freedom(walls), space.sub(0)),
        dolfinx.fem.dirichletbc(rest, velocity_degrees_of_freedom(cylinder), space.sub(0)),
    ]

    state = dolfinx.fem.Function(space)
    velocity, pressure = ufl.split(state)
    velocity_test, pressure_test = ufl.TestFunctions(space)

    def momentum(test):
        return (
            viscosity * ufl.inner(ufl.grad(velocity), ufl.grad(test))
            + ufl.inner(ufl.grad(velocity) * velocity, test)
            - pressure * ufl.div(test)
        ) * ufl.dx

    residual = momentum(velocity_test) - pressure_test * ufl.div(velocity) * ufl.dx
    problem = NonlinearProblem(residual, state, bcs=conditions)
    solver = newton_solver(problem, comm)
    start = time.perf_counter()
    iterations, converged = solver.solve(state)
    wall_time = comm.allreduce(time.perf_counter() - start, op=MPI.MAX)
    if not converged:
        raise RuntimeError(f"Newton's method did not converge for h = {size}.")

    # The reaction on the cylinder, one direction at a time. The dofs are
    # blocks of two, x and y, and include the ghosts on every rank.
    cylinder_degrees_of_freedom = dolfinx.fem.locate_dofs_topological(
        velocity_space, dimension - 1, facet_tags.find(cylinder)
    )
    force = []
    for direction in range(2):
        test = dolfinx.fem.Function(velocity_space)
        test.x.array[2 * cylinder_degrees_of_freedom + direction] = 1.0
        force.append(-total(momentum(test)))

    p_front, p_back = evaluate(state.sub(1).collapse(), [[0.15, 0.2], [0.25, 0.2]])
    result = {
        "h": size,
        "unknowns": unknowns(space),
        "drag": coefficient_scale * force[0],
        "lift": coefficient_scale * force[1],
        "pressure_difference": float(p_front - p_back),
        "newton_iterations": iterations,
        "wall_time": wall_time,
        "ranks": comm.size,
    }
    for key, value in reference.items():
        result[f"relative_error_{key}"] = abs(result[key] - value) / abs(value)
    return result


def main():
    comm = MPI.COMM_WORLD
    results = []
    for size in sizes:
        result = solve(size, comm)
        if comm.rank == 0:
            print(result)
        results.append(result)
    save("dfg_steady", {"element": "P2-P1 Taylor-Hood", "levels": results}, comm)


if __name__ == "__main__":
    main()
