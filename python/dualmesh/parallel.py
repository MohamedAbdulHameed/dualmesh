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

**Processes (distributed memory).** :class:`DistributedProblem` splits the
elements of the mesh among the MPI ranks.  Each rank builds the sub-mesh of its
own elements, defines exactly the same variables, kernels, materials and
boundary conditions on it as a serial run would, and assembles only its own
element integrals.  Because every element belongs to exactly one rank, the sum
of the local residuals is the global residual; a node on a partition boundary
collects a piece of its control domain from every rank that touches it, and one
exchange with the neighbouring ranks adds the pieces together.  The Krylov
solver never forms a distributed matrix: it multiplies by each rank's local
matrix and adds the results across the partition boundary, which is exact.

Run a distributed script with ``mpirun``::

    mpirun -n 8 python my_analysis.py

Both levels can be used at once, for example four ranks of four threads each on
a sixteen-core node.

Whether the installed extension can actually talk to other processes is
reported by :func:`have_mpi`.  A build without MPI still exposes
:class:`DistributedProblem`, on one rank, where it gives exactly the serial
answer; that is what the test suite uses to check the distributed code path
without launching a parallel job.
"""

from __future__ import annotations

import atexit
from collections.abc import Sequence

from . import _core
from .problem import Problem, _solver_options

# MPI must be shut down by every rank or the ones that do shut down wait
# forever for the ones that do not.  A shared library cannot rely on its own
# atexit handler for this, because the handler is removed when the module is
# unloaded, so the hook is registered with the interpreter instead.  Calling it
# twice, or in a build without MPI, does nothing.
atexit.register(_core.finalize_mpi)


def have_mpi() -> bool:
    """Whether the extension was built with MPI support."""
    return _core.have_mpi()


def have_metis() -> bool:
    """Whether the extension was built against METIS.

    METIS usually cuts fewer faces than the built-in partitioners on
    unstructured meshes.  It is the partitioner libMesh, and therefore MOOSE,
    uses.
    """
    return _core.have_metis()


def rank() -> int:
    """This process's rank, counting from zero; 0 in a serial run."""
    return _core.mpi_rank()


def num_ranks() -> int:
    """The number of processes; 1 in a serial run."""
    return _core.mpi_size()


def is_root() -> bool:
    """Whether this is rank 0, the process that should do the printing."""
    return _core.mpi_rank() == 0


def partition_mesh(mesh, num_parts: int, method: str = "recursive_coordinate_bisection") -> dict:
    """Split a mesh into ``num_parts`` groups of elements.

    Parameters
    ----------
    mesh:
        The mesh to split.
    num_parts:
        How many parts to make.  It must not exceed the number of elements.
    method:
        ``"recursive_coordinate_bisection"`` repeatedly halves the set of
        element centroids along its longest axis.  It is fast and deterministic
        and needs no connectivity, but it cuts more faces than necessary on an
        unstructured mesh.  ``"graph"`` grows each part outward from a seed
        element through the face connectivity, so the parts follow the mesh
        topology.  ``"metis"`` calls METIS on the dual graph of the mesh and
        normally gives the smallest cut; it falls back to ``"graph"`` when the
        extension was built without METIS.

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


class DistributedProblem:
    r"""A problem split across MPI ranks.

    Build it, define the physics on :attr:`local` exactly as for a serial
    :class:`dualmesh.Problem`, then call :meth:`solve` or
    :meth:`solve_transient` on this object rather than on the local one.

    Parameters
    ----------
    mesh:
        The whole mesh.  Every rank reads it and then keeps only its own part,
        so the mesh itself is replicated; the degrees of freedom are not.
    method, coordinates:
        As for :class:`dualmesh.Problem`.
    partitioner:
        Which partitioner to use; see :func:`partition_mesh`.
    linear_solver:
        ``"bicgstab"`` (the default, for any matrix) or ``"cg"`` (symmetric
        positive definite matrices only, about twice as cheap per iteration).
        Both are distributed: the matrix-vector product multiplies by each
        rank's local matrix and adds the results across the partition
        boundary, and the inner products count every degree of freedom once.
        The conjugate gradient method needs a symmetric preconditioner, so
        with it the Schwarz options are applied in their classical,
        unrestricted form (every subdomain's whole correction is added) and
        the coarse level additively; this is weaker than the restricted form,
        and on the 48 by 48 Poisson problem ``"cg"`` takes 64 to 69 iterations
        on four to eight ranks with any of the three preconditioners.
    preconditioner:
        ``"two_level_schwarz"``, the default, is restricted additive Schwarz on
        overlapping subdomains with a coarse level.  Every rank's subdomain is
        its own elements extended by ``overlap`` layers of its neighbours'
        elements.  The subdomain matrix is the global matrix restricted to the
        subdomain, :math:`R_\delta A R_\delta^T`, with every row fully
        assembled (the ranks send each other the entries they hold).  Each
        rank solves its subdomain problem approximately and keeps the
        correction only on the degrees of freedom it owns (Cai and Sarkis,
        1999).  The coarse level has one unknown per subdomain and variable
        (Nicolaides, 1987) and is applied before the subdomain solves (the
        operator called A-DEF1 by Tang, Nabben, Vuik and Erlangga, 2009), so
        that information crosses the whole mesh in one application.

        ``"additive_schwarz"`` is the same without the coarse level.
        ``"jacobi"`` divides by the diagonal of the global matrix.

        On the Poisson problem of a 48 by 48 mesh, BiCGSTAB to a relative
        tolerance of :math:`10^{-10}` takes the following numbers of
        iterations (measured with this library):

        =======================  ==  ==  ==  ==  ==
        ranks                    1   2   4   8   16
        =======================  ==  ==  ==  ==  ==
        ``two_level_schwarz``    24  25  23  25  24
        ``additive_schwarz``     24  26  29  33  33
        ``jacobi``               50  49  49  50  49
        =======================  ==  ==  ==  ==  ==

        On a 128 by 128 mesh the two-level counts are 66, 60 and 64 on one,
        four and sixteen ranks, the one-level counts 66, 76 and 86, and
        Jacobi's 137, 131 and 132.  The iteration count is reported as
        ``linear_iterations`` on the result of :meth:`solve`, and it is the
        number to watch: a preconditioner whose count grows in proportion to
        the number of ranks cancels the benefit of the extra ranks.
    overlap:
        Layers of elements by which each Schwarz subdomain reaches into its
        neighbours (default 1).  Zero gives non-overlapping subdomains.  More
        overlap lowers the iteration count and raises the cost of each
        subdomain solve and of the setup exchange.
    subdomain_solver:
        ``"ilu"`` (the default: incomplete LU without fill, cheap to build and
        apply) or ``"lu"`` (an exact sparse LU of each subdomain matrix, which
        roughly halves the iteration count in two dimensions and is expensive
        for large three-dimensional subdomains).
    linear_tolerance, linear_max_iterations, verbose:
        As for the serial solver.
    """

    def __init__(
        self,
        mesh,
        method: str = "dmcdm",
        coordinates: str = "cartesian",
        partitioner: str = "graph",
        linear_solver: str = "bicgstab",
        preconditioner: str = "two_level_schwarz",
        overlap: int = 1,
        subdomain_solver: str = "ilu",
        linear_tolerance: float = 1e-10,
        linear_max_iterations: int = 5000,
        verbose: bool = False,
    ):
        options = _core.DistributedOptions()
        options.partitioner = partitioner
        options.linear_solver = linear_solver
        options.preconditioner = preconditioner
        options.overlap = overlap
        options.subdomain_solver = subdomain_solver
        options.linear_tolerance = linear_tolerance
        options.linear_max_iterations = linear_max_iterations
        options.verbose = verbose
        self._distributed = _core.DistributedProblem(mesh, method, coordinates, options)
        self._mesh = mesh
        _local_core = self._distributed.local()
        #: The rank-local problem; define the physics on it.
        self.local = Problem._wrap(_local_core, _local_core.mesh, method, coordinates)

    @property
    def rank(self) -> int:
        """This process's rank."""
        return self._distributed.rank()

    @property
    def num_ranks(self) -> int:
        """The number of processes the problem is split across."""
        return self._distributed.num_ranks()

    @property
    def num_owned_dofs(self) -> int:
        """Degrees of freedom this rank owns."""
        return self._distributed.num_owned_dofs()

    @property
    def num_global_dofs(self) -> int:
        """Degrees of freedom of the whole problem."""
        return self._distributed.num_global_dofs()

    def solve(self, **options):
        """Solve the steady problem.  Takes the same options as
        :meth:`dualmesh.Problem.solve`, except that the linear solver is
        configured in the constructor."""
        return self._distributed.solve_steady(_solver_options(**options))

    def solve_transient(
        self,
        end_time: float,
        dt: float,
        start_time: float = 0.0,
        theta: float = 1.0,
        output_interval: int = 0,
        output_file_base: str = "",
        time_stepper: str = "fixed",
        dt_min: float = 0.0,
        dt_max: float = 0.0,
        growth_factor: float = 2.0,
        cutback_factor: float = 0.5,
        error_tolerance: float = 1.0e-3,
        optimal_iterations: int = 4,
        iteration_window: int = 2,
        max_rejected_steps: int = 10,
        **options,
    ):
        """Advance the problem in time, with the same arguments and the same
        time steppers as :meth:`dualmesh.Problem.solve_transient`.

        The adaptive controllers are driven by quantities that are already
        global reductions -- the residual norm and the norm of the difference
        between the coarse and the fine step -- so every rank reaches the same
        decision and they step in lockstep without any extra communication.
        With ``output_file_base`` set, each output step writes one ``.vtu`` per
        rank and one ``.pvtu`` index.
        """
        transient = _core.TransientOptions()
        transient.start_time = start_time
        transient.end_time = end_time
        transient.dt = dt
        transient.theta = theta
        transient.output_interval = output_interval
        transient.output_file_base = output_file_base
        transient.time_stepper = time_stepper
        transient.dt_min = dt_min
        transient.dt_max = dt_max
        transient.growth_factor = growth_factor
        transient.cutback_factor = cutback_factor
        transient.error_tolerance = error_tolerance
        transient.optimal_iterations = optimal_iterations
        transient.iteration_window = iteration_window
        transient.max_rejected_steps = max_rejected_steps
        return self._distributed.solve_transient(transient, _solver_options(**options))

    def gathered_values(self, variable: str):
        """Values of a variable at every node of the whole mesh, assembled on
        every rank.

        This allocates one vector of the global size per rank, so it is meant
        for testing and for small problems; large runs should write the result
        with :meth:`write_vtu` instead.
        """
        return self._distributed.gathered_values(variable)

    def write_vtu(self, base: str, cell_properties: Sequence[str] = ()) -> None:
        """Write one ``.vtu`` file per rank plus a ``.pvtu`` index.

        ``base`` is the file name without an extension.  ParaView and VisIt
        open the ``.pvtu`` file as a single data set.
        """
        self._distributed.write_vtu(base, list(cell_properties))

    def summary(self) -> str:
        """One line describing the partition, for logs."""
        return self._distributed.summary()
