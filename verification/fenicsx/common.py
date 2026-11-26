# SPDX-License-Identifier: LGPL-2.1-or-later
"""Helpers shared by the FEniCSx counterparts of the dualmesh benchmarks.

The scripts use DOLFINx 0.9 (see ``environment.yml``). Every helper works on
any number of MPI ranks.
"""

import json
from pathlib import Path

import dolfinx
import dolfinx.nls.petsc
import numpy
from mpi4py import MPI
from petsc4py import PETSc

here = Path(__file__).resolve().parent
results_directory = here / "results"


def total(expression):
    """Integral of a UFL expression over the whole mesh, summed over the ranks."""
    form = dolfinx.fem.form(expression)
    local = dolfinx.fem.assemble_scalar(form)
    return form.mesh.comm.allreduce(local, op=MPI.SUM)


def evaluate(function, points):
    """Values of ``function`` at ``points`` (shape (n, 2) or (n, 3)), on every rank.

    A point is evaluated on the first rank whose cells contain it, and the
    values are gathered to every rank.
    """
    mesh = function.function_space.mesh
    points = numpy.asarray(points, dtype=numpy.float64)
    if points.shape[1] == 2:
        points = numpy.hstack((points, numpy.zeros((len(points), 1))))
    value_size = int(numpy.prod(function.ufl_shape))
    tree = dolfinx.geometry.bb_tree(mesh, mesh.topology.dim)
    candidates = dolfinx.geometry.compute_collisions_points(tree, points)
    colliding = dolfinx.geometry.compute_colliding_cells(mesh, candidates, points)
    local = numpy.full((len(points), value_size), numpy.nan)
    found, cells = [], []
    for index in range(len(points)):
        links = colliding.links(index)
        if len(links) > 0:
            found.append(index)
            cells.append(links[0])
    if found:
        values = function.eval(points[found], numpy.asarray(cells, dtype=numpy.int32))
        local[found] = numpy.reshape(values, (len(found), value_size))
    result = numpy.full_like(local, numpy.nan)
    for part in mesh.comm.allgather(local):
        missing = numpy.isnan(result) & ~numpy.isnan(part)
        result[missing] = part[missing]
    if numpy.isnan(result).any():
        raise RuntimeError("A point lies outside the mesh.")
    return result if value_size > 1 else result[:, 0]


def unknowns(space):
    """Number of degrees of freedom of a mixed function space, over all ranks."""
    count = 0
    for index in range(space.num_sub_spaces):
        sub_space, _ = space.sub(index).collapse()
        index_map = sub_space.dofmap.index_map
        count += index_map.size_global * sub_space.dofmap.index_map_bs
    return count


def newton_solver(problem, comm, relative_tolerance=1e-10):
    """Newton's method with the parallel direct solver MUMPS for every linear step."""
    solver = dolfinx.nls.petsc.NewtonSolver(comm, problem)
    solver.convergence_criterion = "residual"
    solver.rtol = relative_tolerance
    solver.atol = 1e-13
    solver.max_it = 50
    Krylov = solver.krylov_solver
    options = PETSc.Options()
    prefix = Krylov.getOptionsPrefix()
    options[f"{prefix}ksp_type"] = "preonly"
    options[f"{prefix}pc_type"] = "lu"
    options[f"{prefix}pc_factor_mat_solver_type"] = "mumps"
    Krylov.setFromOptions()
    return solver


def save(name, data, comm):
    """Writes ``data`` to ``results/<name>.json`` from rank 0."""
    if comm.rank == 0:
        results_directory.mkdir(exist_ok=True)
        path = results_directory / f"{name}.json"
        path.write_text(json.dumps(data, indent=2))
        print(f"Wrote {path}")
