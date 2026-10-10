Parallel execution
==================

.. contents::
   :local:
   :depth: 2
   :class: this-will-duplicate-information-and-it-is-still-useful-here

Two kinds of parallelism
------------------------

``dualmesh`` parallelises on two levels, and the two are independent of each
other.

**Shared memory.**  Within one process, the assembly loops (the loops that
visit every element of the mesh and accumulate its contribution to the residual
vector and the Jacobian matrix) are threaded with OpenMP.  All threads see the
same mesh and the same solution vector, because they share an address space.
Nothing has to be set up: a threaded run needs one call to
:meth:`~dualmesh.Problem.set_num_threads`, or nothing at all if the
``OMP_NUM_THREADS`` environment variable is set.  This level is limited to the
cores of a single machine, and to problems that fit in that machine's memory.

**Distributed memory.**  Across processes, a :class:`~dualmesh.Problem` splits
the elements of the mesh among the MPI ranks.  Each rank holds only its
own part of the problem, so the memory of a whole cluster is available, and the
ranks exchange data explicitly through MPI.  A distributed script is launched
with ``mpirun``, as in ``mpirun -n 8 python my_analysis.py``, and the same
script runs serially with ``python my_analysis.py``.  The problem splits itself
when it runs on more than one process, and ``dm.Problem(mesh,
distributed=True)`` takes the distributed path on one process as well.

The two levels compose: four ranks of four threads each is a sensible
configuration for a sixteen-core node.  The appropriate level depends on the
limiting resource.  Threading is the simpler of the two and is usually
sufficient when the problem fits in one machine.  Distributed execution is
required when it does not, and it is the only way to use more than one machine.
Whether the installed extension can talk to other processes is reported by
:func:`~dualmesh.have_mpi`.  A build without MPI still runs a problem with
``distributed=True`` on one rank, where every collective is the identity and
every exchange is empty, so it reproduces the serial answer exactly.  The test suite uses this single-rank configuration to exercise the
distributed code path without launching a parallel job.

Threading the assembly
----------------------

Assembly computes the residual :math:`R` and, when a Jacobian is wanted, the
matrix :math:`J = \partial R / \partial U`.  Both are sums over elements,

.. math::
   :label: elementsum

   R = \sum_{K \in \mathcal{T}_h} R_K , \qquad
   J = \sum_{K \in \mathcal{T}_h} J_K ,

where :math:`R_K` and :math:`J_K` are supported on the degrees of freedom of the
nodes of :math:`K`.  For the dual mesh and finite volume methods
:math:`R_K` gathers two families of integration points, namely the
control-volume points interior to :math:`K` and the points on the dual faces
separating one node's control domain from another's within :math:`K`, and both
are visited
inside the single loop over :math:`K`.  Computing :math:`R_K` requires the
element's geometry, the current values of the unknowns at its nodes, the
quadrature rule, and calls into the kernels and property objects of the problem.  None
of that depends on any other element.  The element loop is therefore
*embarrassingly parallel*: the work
splits without reordering, and the only difficulty is the accumulation in
:eq:`elementsum`, where two elements sharing a node write to the same entry of
:math:`R` and of :math:`J`.  Three arrangements make that accumulation safe.

**Per-thread scratch space.**  Everything an element computation writes to,
other than the global result, lives in a ``ThreadScratch`` object held privately
by each thread: the gathered nodal values of the current, lagged and old
solutions, the global degree-of-freedom indices, the element-local residual of
automatic-differentiation numbers, the array marking which local entries were
touched, the integration points, the mapped-point geometry, and the quadrature
context handed to the kernels.  One such object is allocated per thread before
the loop begins and reused across all the elements that thread visits, so no
allocation happens inside the loop and no two threads ever write to the same
buffer.  State that looks read-only but is built lazily on first use is a
hazard, since two threads reaching an empty cache would both try to fill it, so
``Problem::initialize`` touches the mesh's list of exterior sides and its
boundary node markers and instantiates the reference element definition of every
element type present, so that every such cache is populated before the loop
starts.

**Atomic updates of the residual.**  The residual is a single shared array, and
several threads may add to the same entry.  Each addition is made under
``#pragma omp atomic``, which makes the read-modify-write of one ``double``
indivisible without taking a lock.  This is the only shared write in the inner
loop, and atomics are used only when more than one thread is running.  A serial
assembly writes directly.

**Per-thread triplet buffers.**  The Jacobian is built as a list of
:math:`(\text{row}, \text{column}, \text{value})` triplets and then converted to
compressed sparse form by Eigen [Eigen]_.  Each thread appends to a private
list, pre-reserved from an estimate of the total entry count divided by the
number of threads.  When the element loop finishes, the
per-thread lists are concatenated in thread order and handed to
``setFromTriplets``, which sums entries repeating the same row and column.
The summation of duplicates makes the split valid: a matrix entry receiving
contributions from elements assembled by different threads appears several
times in the merged list, and the contributions are added.

The boundary terms (the integrated boundary conditions and the concentrated
nodal loads) are assembled serially afterwards, in the first thread's scratch
space.  They involve only the sides on a boundary, a small subset of the mesh,
so they are a small fraction of the work.

When threading is switched off
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:meth:`~dualmesh.Problem.set_num_threads` states how many threads are
requested, and :meth:`~dualmesh.Problem.effective_threads` reports how many
will be used.  The second can be smaller than the first, for three reasons.
The first reason is that an extension compiled without OpenMP always runs the
loop serially.  The other two reasons are described below.

**An object of the problem is defined in Python.**  A kernel, a property object, a
function or an initial condition written in Python has to be called back into
the interpreter, and CPython's global interpreter lock permits only one thread
to execute Python bytecode at a time.  Several threads calling such an object
would serialise on the lock, and the lock would have to be acquired correctly
from each of them.  To avoid this, every object of the problem is asked whether
it is thread-safe when the problem is resolved.  A Python-defined object
answers no, and a single such answer sets
:meth:`~dualmesh.Problem.thread_safe` to false for the whole problem and forces
``effective_threads`` to 1.  The fallback is silent and automatic: a
calculation that adds a Python function to an otherwise C++ problem runs
correctly, but serially.  To restore threading, that object must be expressed
in C++ or as one of the built-in types.

**The mesh is too small.**  Starting a team of threads costs a few microseconds,
which is more than a small mesh takes to assemble, so the library does not
thread a problem whose assembly time is shorter than this overhead.  The
threshold is the compile-time constant
:math:`\texttt{kMinElementsPerThread} = 400`, and the number of threads actually
used is

.. math::
   :label: nthreads

   n_{\text{eff}} = \max\left\{ 1, \;
      \min\left( n_{\text{requested}},\;
      \left\lfloor \frac{N_{\text{el}}}{400} \right\rfloor \right) \right\} ,

where :math:`N_{\text{el}}` is the number of elements.  A 1600-element mesh thus
supports at most four threads however many are requested, and a mesh of fewer
than 400 elements is always assembled serially.  When no thread count is
requested, :math:`n_{\text{requested}}` is OpenMP's own default, normally
``OMP_NUM_THREADS`` or the number of cores.

Reproducibility of the threaded result
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Changing the thread count must not change the answer, and in the library's
measurements it does not.  Two properties ensure this.

First, every per-element quantity is computed identically regardless of
threading.  The scratch space is private, the loop bounds are assigned
statically, and the integrand of an element depends only on that element's
geometry, its nodal values and the problem's objects, and is independent of
which elements have already been visited.  There is no accumulator carried
across elements and no shared mutable state other than the result arrays, so
each :math:`R_K` and each :math:`J_K` in :eq:`elementsum` is bit-for-bit what
a serial run computes.

Second, the *order* in which those contributions are summed is the only thing
threading changes, and the library keeps that order as stable as it can.  The
Jacobian's triplet lists are concatenated in thread order and each thread's list
is built in static-schedule order, so the merged list is a deterministic
permutation for a given thread count.  The residual's atomic updates are the
exception: the order in which two threads add to the same entry is decided by
the hardware, and floating-point addition is not associative, so the library
guarantees only that the residuals obtained with different thread counts
differ, if at all, by rounding at the last bit.  In the measurements below
the residual norm is identical to all thirteen digits printed at every thread
count, and the test suite requires the solved nodal values of a serial run and a
four-thread run to agree to within :math:`10^{-12}` in absolute terms for all
four discretisations.  A user who requires exactly reproducible bits should set
the thread count to one.

Measured threading performance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The bundled ``dualmesh_benchmark`` program assembles the residual and Jacobian
of a three-dimensional linear elasticity problem on a structured mesh of
``Hex8`` elements.  The figures below are for a :math:`20 \times 20 \times 20`
mesh (8000 elements and 27 783 degrees of freedom) on a machine with two
physical cores, so the fourth column is measured on hardware that cannot deliver
it and is shown only to make that point.

.. table:: Assembly time per residual-and-Jacobian pass

   ===========  ========  =========  =========  ============
   Method       1 thread  2 threads  4 threads  Speedup at 2
   ===========  ========  =========  =========  ============
   ``dmcdm``    0.604 s   0.398 s    0.346 s    1.52
   ``fem``      0.261 s   0.211 s    0.187 s    1.24
   ``hfvm``     0.813 s   0.524 s    0.498 s    1.55
   ===========  ========  =========  =========  ============

The single-thread times depend on four properties of the implementation: the
geometry of every integration point is mapped once, the reference-element data
are cached, the derivative loops of the automatic differentiation vectorise,
and the derivative arrays are copied only as far as they are used.
The speedup at two threads is between 1.24 and 1.55 against an ideal of 2.  The
shortfall is caused by the part of an assembly that is not threaded (the
boundary conditions, the merge of the per-thread triplet lists and the
construction of the sparse matrix), and it is largest for the finite element
method, whose threaded element loop is the cheapest.  Going from two to four
threads on two cores adds little, as expected.  Only the assembly and the error
norms are threaded.  The linear solve is handed to Eigen [Eigen]_ and the
library makes no attempt to parallelise it, so a calculation whose cost is
dominated by the linear solve gains little from threading.

Distributed-memory execution
----------------------------

The decomposition
^^^^^^^^^^^^^^^^^

Every element of the mesh belongs to exactly one rank.  Rank :math:`r` holds
the sub-mesh made of its own elements together with *every node those elements
touch*, which includes nodes on the partition boundary that other ranks also
hold.  Nodes are numbered locally and the rank keeps the global index of each
of its local nodes.  Blocks are preserved and side sets are restricted to the
sides that the rank holds.  A side set that is empty on a rank is still
created, so a boundary condition naming it remains valid everywhere.  No rank
needs the whole mesh.  When a whole mesh is given, the first rank alone
partitions it and sends every rank its part, with the global node numbering of
the whole mesh.  A node of a side set that a rank holds without holding a side
of it (a triangle can touch the boundary with one vertex) is added to the
rank's node set of that boundary by one exchange, so that a value prescribed on
the boundary reaches every rank that holds one of its nodes.  On that
sub-mesh the rank builds an ordinary :class:`~dualmesh.Problem`, and the user
defines exactly the same variables, kernels, property objects and boundary conditions
on it as in a serial run.  Every discretisation, every physics module and the
automatic differentiation work unchanged, because the rank-local object is an
ordinary problem posed on a smaller mesh.

Because every element belongs to exactly one rank, the element sums
:eq:`elementsum` split exactly:

.. math::
   :label: ranksum

   R(U) = \sum_{r} R_r(U) , \qquad
   J(U) = \sum_{r} J_r(U) ,

where :math:`R_r` and :math:`J_r` collect the contributions of rank :math:`r`'s
elements only.  Nothing is counted twice and nothing is missed.  In the language
of the dual mesh method: a node in the interior of a partition receives its
whole control domain from one rank, while a node on a partition boundary
receives a piece of its control domain from each rank that touches it, and
:eq:`ranksum` adds the pieces together.  A local vector in which every rank
holding a shared node has the same, complete value there is called
**consistent**.  One exchange with the neighbouring ranks turns a vector of
partial contributions into a consistent one: each rank sends its partial values
for the nodes it shares with rank :math:`s` to rank :math:`s`, receives theirs,
and adds.  The two sides agree on the order of the shared nodes by sorting them
by global node index, which requires no communication.

Every such exchange goes through one gather-scatter layer.  A rank gives the
global index of each of its nodes and nothing else, and the layer finds which
ranks hold each index, which of them owns it (the smallest), and how to combine
their values.  Two libraries can do this, chosen with the ``gather_scatter``
argument of :class:`~dualmesh.Problem`, as the discretization is chosen with
``method``; the answer is the same with either.

``"petsc"``
   PETSc's star forests (PetscSF) [PETSc2023]_.  At setup, each index has a
   *home* rank in an even layout of the indices, and a minimum reduction there
   over the pairs (rank, local index) of every copy finds the owner, which every
   copy is then told.  The copies that a rank does not own become the leaves of
   a star forest whose roots are the owners' copies, and a sum is a reduction
   from the leaves to the roots followed by a broadcast back.  The same star
   forests carry PETSc's own distributed vectors and matrices.

``"gslib"``
   gslib, the gather-scatter library of Nek5000 and nekRS [gslib]_ [Fischer2021]_,
   built with dualmesh.  At setup it times three ways of exchanging the values,
   pairwise messages between the ranks that share nodes, a crystal router that
   needs :math:`\log_2 P` messages per rank, and an all-reduce, and keeps the
   fastest for the mesh and the machine at hand.  Nek5000 and nekRS do all
   their communication through it.

Which one is faster depends on the number of ranks, on the pattern of shared
nodes and on the network; the scaling studies compare them.

Node ownership
^^^^^^^^^^^^^^

Several ranks may hold the same node, but exactly one **owns** it.  The owner is
the smallest rank index among those that touch the node, which the setup of the
gather-scatter gives every rank that holds the node.  Ownership
settles three things.

It settles counting: the number of degrees of freedom of the whole problem is
the sum over ranks of the degrees of freedom each rank owns, whereas adding up
the local counts would count every shared node once per rank that holds it.

It settles inner products, which matters more.  A Krylov method is built out of
inner products, and if :math:`\langle a, b \rangle` double-counted the shared
nodes the method would be minimising the wrong norm and its scalars would be
wrong.  The library's inner product is therefore ownership-weighted,

.. math::
   :label: dotproduct

   \langle a, b \rangle \;=\; \sum_{r} \;\;
   \sum_{i \; \text{owned by} \; r} a_i \, b_i ,

that is, each rank sums over the degrees of freedom it owns and the results are
reduced across all ranks.  Every degree of freedom then contributes exactly
once, and :math:`\| a \| = \sqrt{\langle a, a \rangle}` is the true global norm.

It settles the application of nodal quantities, i.e., quantities attached to a
node.  A concentrated load belongs to a node, so only the owning rank applies
it.  If every rank holding that node applied it, the load would be counted once
per such rank.  A load given by coordinates is first resolved to the nearest
node, and since each rank sees only its own sub-mesh, each would find a nearest
node of its own.  The ranks therefore reduce the distances, and the rank whose
node is globally nearest
keeps the load, ties being broken by the ownership mask.  The test suite finds a
point-source solution agreeing with the serial one to
:math:`1.0 \times 10^{-14}` on four ranks.

The distributed linear solver
-----------------------------

One distributed matrix
^^^^^^^^^^^^^^^^^^^^^^

The global Jacobian is one distributed PETSc matrix [PETSc2023]_.  Every rank's
owned degrees of freedom are numbered contiguously, rank after rank, and a
shared degree of freedom takes its owner's number, which one owner-copy
exchange distributes.  Each rank then lists the entries of its local matrix,
the contributions of its own elements, in that global numbering, and PETSc's
coordinate-format assembly sends each entry to the rank that owns its row and
adds repeated positions.  By :eq:`ranksum` the sum over ranks of the element
contributions is exactly the global Jacobian.  The only care needed is with the
rows of prescribed degrees of freedom, which every rank holding such a node has
replaced by a row of the identity: they are entered once, by the owner, so that
the diagonal entry is 1 for any number of ranks sharing the node.  The
right-hand side contributes its owned entries, and the owned entries of the
solution are copied back to the other ranks that hold them.

The Krylov method and the preconditioner
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

PETSc solves the system, so every one of its Krylov methods and
preconditioners is available through ``petsc_options``.  The default,
``linear_solver="automatic"``, makes the choice that the serial automatic
solver makes.  It takes GMRES preconditioned by the algebraic multigrid
BoomerAMG of hypre [FalgoutYang2002]_, whose iteration count does not grow with the
mesh or with the number of ranks.  When that does not converge within 500
iterations, or when the system has a zero pressure block (the Taylor-Hood
element), on which multigrid cannot act, it takes the parallel direct solver
MUMPS [Amestoy2001]_.  ``linear_solver="lu"`` takes MUMPS directly.  Any other
choice is a PETSc option string, for example ``"-ksp_type gmres -pc_type asm
-sub_pc_type ilu"`` for restricted additive Schwarz with incomplete subdomain
solves [CaiSarkis1999]_, ``"-ksp_type cg -pc_type gamg"`` for the conjugate
gradient method with PETSc's smoothed aggregation multigrid on the symmetric
matrix of the finite element method, or a field split for a saddle point
problem.

Measured iteration counts
^^^^^^^^^^^^^^^^^^^^^^^^^

On the Poisson problem :math:`-\nabla^2 u = 1` on the unit square with
:math:`u = 0` on the boundary, :math:`48 \times 48` ``Quad4`` elements (2401
unknowns) and the dual mesh control domain method, the automatic choice
(GMRES with BoomerAMG, relative tolerance :math:`10^{-10}`) takes 15 iterations
on one, two, three and four ranks.  On the finite element discretization of the
conduction problem of the test suite (:math:`16 \times 16` ``Quad4``), the
conjugate gradient method takes 49 iterations with Jacobi, 9 with BoomerAMG and
12 with GAMG, on one to four ranks alike.  Runs on more ranks belong to the
scaling studies, which run on a cluster.

These tests are part of the suite: ``dualmesh_parallel_tests``, launched with
``mpirun``, checks that the automatic choice, additive Schwarz with incomplete
and exact subdomain solves, GAMG and MUMPS reproduce the serial solution (to
:math:`10^{-8}`, measured at about :math:`10^{-12}`), including a two-variable
elasticity problem, and that the automatic choice takes at most 20 iterations
on the :math:`48 \times 48` problem on whatever number of ranks it is run.  The
Python tests check the pressure-velocity cavity and the Taylor-Hood cavity in
the same way.

Partitioning
------------

How the elements are split among the ranks decides both the load balance (how
evenly the work is shared) and the communication volume, measured by the
*edge cut*, the number of mesh faces whose two elements lie in different parts.
The partitioners are PETSc's [PETSc2023]_, chosen with the ``partitioner``
argument of :class:`~dualmesh.Problem` and available alone through
:func:`~dualmesh.partition_mesh`.

PETSc's unstructured mesh, DMPlex, receives the corner nodes of every element as
the cone of a cell, so that every element type, the quadratic ones included, is
partitioned by its corners, and it builds the graph of the elements that share a
face.  ``"ptscotch"`` (PT-Scotch [Chevalier2008]_) and ``"parmetis"``
(ParMETIS, the parallel form of the multilevel :math:`k`-way algorithm of
Karypis and Kumar [KarypisKumar1998]_) divide that graph so as to cut few faces;
``"simple"`` takes contiguous blocks of elements in their order.
``"automatic"``, the default, takes PT-Scotch, else ParMETIS.  The root rank
holds the whole mesh, and ``DMPlexDistribute`` moves every element, with its
nodes, coordinates, block, sides and node sets, to the rank of its part, where
the rank rebuilds the mesh it holds.

On a structured :math:`20 \times 20` grid of ``Quad4`` elements, PT-Scotch cuts
40 faces into four parts, ParMETIS 46 and contiguous blocks 60; on a
triangulation of the same square into 800 ``Tri3`` elements the cuts are 40, 48
and 60.  PT-Scotch and contiguous blocks give parts of equal size, and ParMETIS
parts within 5 %.  The test suite requires no part to exceed the average by more
than a quarter, and checks that a mesh of every element type, ``Tri6``,
``Quad9``, ``Quad8``, ``Hex8``, ``Tet4``, ``Tet10``, ``Hex27`` and ``Hex20``,
arrives whole: its numbering, its coordinates, and every element and side on
exactly one rank.

What runs in parallel, and what does not
----------------------------------------

**A Python script still builds the whole mesh on every rank.**  The distributed
solver needs only each rank's part, and a whole mesh given to it is read on the
first rank alone, which partitions it and sends the parts.  But a script that
calls a mesh generator or reads a mesh file runs on every rank, so the whole
mesh is built on every rank before the solver discards it.  Until the
generators and the readers build only each rank's part, this bounds the
problem size by the memory of a single rank.  The partitioning also runs on the
first rank alone.

**The cell-centered finite volume method runs with two layers of ghost
cells.**  Its unknowns sit at the cells and at the faces of the boundary of the
domain, and it integrates over the faces.  The flux through a face between two
parts needs the reconstructed gradients of the cells on both sides, and the
gradient of a cell comes from all its neighbors, so every part also holds the
cells of other parts up to two layers away, which PETSc's ``DMPlexDistribute``
sends with the part (its overlap).  A cell is numbered by its element and a
boundary face after all cells, in the order of the elements, so that the
numbering does not depend on the partition; the gather-scatter then works on
cells and faces as it works on nodes.  Every face is integrated by one process,
the one whose cell on it has the smaller global number, and the pieces of the
residual and of the Jacobian at the cells that several processes hold are
added as at shared nodes, so the Jacobian is the serial one and Newton's method
takes the serial number of iterations.  A gap condition pairs across processes
as for the other methods.  The test suite checks every cell and boundary face
against the serial solution on one to six processes, on triangles, on
distorted quadrilaterals (where one layer of ghost cells is not enough and
gives differences of :math:`10^{-5}`), on tetrahedra, with a nonlinear
diffusivity, across a gap, and through a checkpoint and a restart on another
partition.

**Gap conditions pair across processes.**  A gap condition pairs every
integration point of its primary side with the closest point of its secondary
side, which may belong to another process.  Every process shares the elements
of its secondary sides, and a process that holds primary sides keeps the
secondary elements it lacks as *ghost elements*: it reads their nodes and
values, but does not integrate them, so that each element is still integrated
once.  The secondary side's share of the flux is added at the ghost nodes and
reaches their owners through the gather-scatter, and the Jacobian couples the
two sides through the global numbering.  The test suite checks the solution
across a gap between non-matching meshes on two bodies held by different
processes against the serial one on one to six processes.

**Adaptive refinement is not distributed.**  The refinement described in
:doc:`adaptivity` operates on a whole mesh in one process.  There is no
distributed error indicator, no parallel marking and no parallel refinement, so
an adaptive loop and an MPI run cannot be combined.

**Only the assembly is threaded.**  All four discretisations thread their
element and face loops, including the cell-centred finite volume method, which
uses the same scratch, atomic and triplet arrangement.  The remainder of a
solve runs serially: the integrated boundary conditions, the concentrated
loads, the merge of the triplet lists, the construction of the sparse matrix and
the linear solve itself, which is handed to Eigen [Eigen]_ and which the library
makes no attempt to parallelise.  The measured assembly speedup on two cores is
between 1.24 and 1.55 (see the table above), and what fraction of a whole solve that represents
depends entirely on how expensive the linear solve is for the problem at hand.

**Post-processors and reactions run on every process.**  Every element and
every boundary side belongs to one process, so an integral, an average, a
boundary flux or a reaction is the sum of what each process computes over its
own elements, sides and owned nodes, reduced over the processes; a nodal
extreme is the extreme over the processes; and a point value is taken from the
process whose elements contain the point (the average of those that do, when
the point lies on a partition boundary).  The test suite checks every
post-processor type against the serial value on one to four processes.

**Gathering the solution does not scale.**
:meth:`~dualmesh.Problem.gathered_values` builds a vector of the
global size on every rank.  It is intended for testing and for small problems.
Large runs should write results with
:meth:`~dualmesh.Problem.write_vtu`, which produces one ``.vtu`` file
per rank plus a ``.pvtu`` index that ParaView and VisIt open as a single data
set.

**The distributed answer does not depend on the decomposition.**  This property
is verified numerically.  On four ranks, for each of ``dmcdm``, ``fem`` and
``hfvm``, for every partitioner of the build (PT-Scotch, ParMETIS and contiguous blocks) and for the
automatic choice, additive Schwarz, GAMG and MUMPS, the distributed steady
conduction solution agrees with the serial one node by node to between
:math:`2 \times 10^{-14}` and :math:`3 \times 10^{-12}`.  These differences are
at the level of the linear solver's tolerance, so no effect of the
decomposition is detectable.  A transient solve over ten steps agrees to
:math:`2.0 \times 10^{-15}`, and the point-source case to
:math:`5.8 \times 10^{-15}`.
