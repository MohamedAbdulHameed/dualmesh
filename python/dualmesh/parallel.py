# SPDX-License-Identifier: LGPL-2.1-or-later
r"""Parallel execution: shared-memory threads and distributed-memory processes.

dualmesh parallelises on two levels, and they compose.

**Threads (shared memory).** The element loop is threaded with OpenMP.  Every
thread keeps its own scratch space and its own list of matrix entries, and the
only shared write is an atomic update of the residual, so no locks are taken in
the inner loop.  Nothing needs to be set up: call
:meth:`dualmesh.Problem.set_num_threads` (or leave it alone, in which case the
``OMP_NUM_THREADS`` environment variable decides) and the assembly is threaded.
The number actually used is reported by
:meth:`dualmesh.Problem.effective_threads`, which returns 1 when the extension
was built without OpenMP, when the mesh is too small for threading to pay for
itself, or when any object of the problem is written in Python, because calling
back into the interpreter requires the global interpreter lock.

**Processes (distributed memory).** A :class:`dualmesh.Problem` that runs on
more than one MPI process splits the elements of the mesh among the ranks.  Each
rank holds the sub-mesh of its own elements, defines exactly the same variables,
kernels, materials and boundary conditions on it as a serial run would, and
assembles only its own element integrals.  Because every element belongs to
exactly one rank, the sum of the local residuals is the global residual.  A node
on a partition boundary collects a piece of its control domain from every rank
that touches it, and PETSc's star forests add the pieces together.  The
Jacobian is one distributed PETSc matrix, which PETSc solves, by default with
GMRES and the algebraic multigrid of hypre.

Run a distributed script with ``mpirun``::

    mpirun -n 8 python my_analysis.py

Both levels can be used at once, for example four ranks of four threads each on
a sixteen-core node.

Whether the installed extension can talk to other processes is reported by
:func:`have_mpi`.  ``dm.Problem(mesh, distributed=True)`` takes the distributed
path on one process as well, where it gives the serial answer, which is how
the test suite checks the distributed code without a parallel job.
"""

from __future__ import annotations

import atexit
from collections.abc import Sequence

from . import _core

# MPI must be shut down by every rank or the ones that do shut down wait
# forever for the ones that do not.  A shared library cannot rely on its own
# atexit handler for this, because the handler is removed when the module is
# unloaded, so the hook is registered with the interpreter instead.  Calling it
# twice, or in a build without MPI, does nothing.
atexit.register(_core.finalize_mpi)


def have_mpi() -> bool:
    """Whether the extension was built with MPI support."""
    return _core.have_mpi()


def have_petsc() -> bool:
    """Whether the extension was built with PETSc, which ``linear_solver="petsc"``
    needs (configure with ``-DDUALMESH_ENABLE_PETSC=ON``)."""
    return _core.have_petsc()


def have_gslib() -> bool:
    """Whether the extension was built with gslib, the gather-scatter of Nek5000, which ``gather_scatter="gslib"`` needs.
    A build with MPI has it unless configured with ``-DDUALMESH_ENABLE_GSLIB=OFF``."""
    return _core.have_gslib()


def petsc_version() -> str:
    """PETSc's version, ``"major.minor.subminor"``, or ``""`` without PETSc."""
    return _core.petsc_version()


def rank() -> int:
    """This process's rank, counting from zero (0 in a serial run)."""
    return _core.mpi_rank()


def num_ranks() -> int:
    """The number of processes (1 in a serial run)."""
    return _core.mpi_size()


def is_root() -> bool:
    """Whether this is rank 0, the process that should do the printing."""
    return _core.mpi_rank() == 0


def partition_mesh(mesh, num_parts: int, method: str = "automatic") -> dict:
    """Split a mesh into ``num_parts`` groups of elements with a partitioner of PETSc, the one a distributed problem uses.

    Parameters
    ----------
    mesh:
        The mesh to split.
    num_parts:
        How many parts to make.  It must not exceed the number of elements.
    method:
        ``"ptscotch"`` (PT-Scotch) and ``"parmetis"`` (ParMETIS) divide the graph of the elements that share a face so as to cut few faces.
        ``"simple"`` takes contiguous blocks of elements in their order.
        ``"automatic"`` (the default) takes PT-Scotch, else ParMETIS.
        It needs a build with PETSc (:func:`have_petsc`).

    Returns
    -------
    dict
        ``element_part`` (the part of every element), ``node_owner`` (the part
        that owns every node, which is the smallest part index among those that
        touch it), ``edge_cut`` (the number of faces whose two elements are in
        different parts, the quantity a partitioner tries to minimise), and
        ``largest_part`` and ``smallest_part`` (element counts, a measure of
        load balance).
    """
    return _core.partition_mesh(mesh, int(num_parts), method)


def sub_mesh(mesh, elements: Sequence[int]):
    """The sub-mesh made of the given elements.

    Returns the sub-mesh and, for every one of its nodes, the index of the
    corresponding node of the original mesh.  Blocks are preserved and side
    sets are restricted to the sides that survive, so a boundary condition
    written for the whole mesh still applies.
    """
    return _core.sub_mesh(mesh, [int(e) for e in elements])
