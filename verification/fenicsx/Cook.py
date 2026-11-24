# SPDX-License-Identifier: LGPL-2.1-or-later
"""Cook's membrane at finite strain, solved by FEniCSx.

This is the FEniCSx counterpart of ``verification/benchmarks/solids/run_Cook.py``, on the same
Gmsh meshes of second-order quadrilaterals, with a Q2 displacement. The units are mm, N and
MPa. The left edge is clamped and the right edge carries the vertical traction 20 MPa per unit
undeformed length of the edge, in plane strain. The energy is

    psi_1 = mu/2 (I_C - 3) + lambda/4 (J^2 - 1) - (lambda/2 + mu) ln J,

with lambda = 432.099 MPa and mu = 185.185 MPa. Schroeder et al. (2021) give the vertical
displacement of the corner A = (48, 60) mm as 10.59528994 mm.

    mpirun -n 4 python Cook.py
"""

import sys
import time
from pathlib import Path

import dolfinx
import dolfinx.io.gmshio
import numpy
import ufl
from basix.ufl import element
from common import evaluate, newton_solver, results_directory, save
from dolfinx.fem.petsc import NonlinearProblem
from mpi4py import MPI

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks" / "solids"))
import run_Cook  # noqa: E402

first_Lame_parameter, shear_modulus, load = 432.099, 185.185, 20.0  # MPa


def solve(n, comm):
    path = results_directory / f"Cook_{n}.msh"
    if comm.rank == 0:
        results_directory.mkdir(exist_ok=True)
        run_Cook.build_mesh(n, path)
    comm.barrier()
    mesh, _, facet_tags = dolfinx.io.gmshio.read_from_msh(str(path), comm, 0, gdim=2)
    space = dolfinx.fem.functionspace(mesh, element("Lagrange", mesh.basix_cell(), 2, shape=(2,)))
    displacement = dolfinx.fem.Function(space)
    test = ufl.TestFunction(space)

    deformation = ufl.variable(ufl.Identity(2) + ufl.grad(displacement))
    right_Cauchy_Green = deformation.T * deformation
    J = ufl.det(deformation)
    # Plane strain: the out-of-plane stretch is 1, so tr C counts it once.
    I_C = ufl.tr(right_Cauchy_Green) + 1.0
    energy = (
        shear_modulus / 2 * (I_C - 3)
        + first_Lame_parameter / 4 * (J**2 - 1)
        - (first_Lame_parameter / 2 + shear_modulus) * ufl.ln(J)
    )
    # The first Piola-Kirchhoff stress P = d psi / d F.
    stress = ufl.diff(energy, deformation)
    traction = dolfinx.fem.Constant(mesh, numpy.array([0.0, load]))
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=facet_tags)
    residual = ufl.inner(stress, ufl.grad(test)) * ufl.dx - ufl.dot(traction, test) * ds(
        run_Cook.right_tag
    )

    clamped = dolfinx.fem.locate_dofs_topological(space, 1, facet_tags.find(run_Cook.left_tag))
    zero = dolfinx.fem.Function(space)
    condition = dolfinx.fem.dirichletbc(zero, clamped)
    problem = NonlinearProblem(residual, displacement, bcs=[condition])
    solver = newton_solver(problem, comm)
    start = time.perf_counter()
    iterations, converged = solver.solve(displacement)
    elapsed = comm.allreduce(time.perf_counter() - start, op=MPI.MAX)
    if not converged:
        raise RuntimeError(f"Newton's method did not converge on the {n} by {n} mesh.")
    corner = evaluate(displacement, [[48.0, 60.0]])[0]
    count = space.dofmap.index_map.size_global * space.dofmap.index_map_bs
    return {
        "n": n,
        "displacement": float(corner[1]),
        "unknowns": count,
        "time": elapsed,
        "iterations": iterations,
    }


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rows = []
    for n in run_Cook.levels:
        row = solve(n, comm)
        rows.append(row)
        if comm.rank == 0:
            error = row["displacement"] / run_Cook.reference_displacement - 1
            print(
                f"n = {n:3d}, {row['unknowns']:6d} unknowns: u_y(A) = {row['displacement']:.8f} mm "
                f"({100 * error:+.4f} %), {row['time']:.1f} s, {row['iterations']} Newton iterations",
                flush=True,
            )
    save("Cook", rows, comm)
