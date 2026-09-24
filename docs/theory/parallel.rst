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

**Shared memory.**  Within one process, the assembly loops -- the loops that
visit every element of the mesh and accumulate its contribution to the residual
vector and the Jacobian matrix -- are threaded with OpenMP.  All threads see the
same mesh and the same solution vector, because they share an address space.
Nothing has to be set up: a threaded run needs one call to
:meth:`~dualmesh.Problem.set_num_threads`, or nothing at all if the
``OMP_NUM_THREADS`` environment variable is set.  This level is limited to the
cores of a single machine, and to problems that fit in that machine's memory.

**Distributed memory.**  Across processes, :class:`~dualmesh.DistributedProblem`
splits the elements of the mesh among the MPI ranks.  Each rank holds only its
own part of the problem, so the memory of a whole cluster is available, and the
ranks exchange data explicitly through MPI.  A distributed script is launched
with ``mpirun``, as in ``mpirun -n 8 python my_analysis.py``.

The two levels compose: four ranks of four threads each is a sensible
configuration for a sixteen-core node.  Which one to reach for depends on the
obstacle.  Threading is the simpler of the two and is usually enough when the
problem fits in one machine; distributed execution is what is needed when it
does not, and it is the only way to use more than one machine.  Whether the
installed extension can actually talk to other processes is reported by
:func:`~dualmesh.have_mpi`.  A build without MPI still exposes
:class:`~dualmesh.DistributedProblem`, on one rank, where every collective is
the identity and every exchange is empty, so it reproduces the serial answer
exactly; that is how the distributed code path is exercised by the test suite
without launching a parallel job.

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
:math:`R_K` gathers two families of integration points -- the control-volume
points interior to :math:`K` and the points on the dual faces separating one
node's control domain from another's within :math:`K` -- and both are visited
inside the single loop over :math:`K`.  Computing :math:`R_K` requires the
element's geometry, the current values of the unknowns at its nodes, the
quadrature rule, and calls into the kernels and materials of the problem.  None
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
element type present, leaving every such cache warm before the loop starts.

**Atomic updates of the residual.**  The residual is a single shared array, and
several threads may add to the same entry.  Each addition is made under
``#pragma omp atomic``, which makes the read-modify-write of one ``double``
indivisible without taking a lock.  This is the only shared write in the inner
loop, and atomics are used only when more than one thread is running; a serial
assembly writes directly.

**Per-thread triplet buffers.**  The Jacobian is built as a list of
:math:`(\text{row}, \text{column}, \text{value})` triplets and then converted to
compressed sparse form by Eigen [Eigen]_.  Rather than share the list, each
thread appends to its own, pre-reserved from an estimate of the total entry
count divided by the number of threads.  When the element loop finishes, the
per-thread lists are concatenated in thread order and handed to
``setFromTriplets``, which sums entries repeating the same row and column.
Summing duplicates is what makes the split legitimate: a matrix entry receiving
contributions from elements assembled by different threads simply appears
several times in the merged list.

The boundary terms -- the integrated boundary conditions and the concentrated
nodal loads -- are assembled serially afterwards, in the first thread's scratch
space.  They involve the sides of a boundary rather than all the elements of the
mesh, so they are a small fraction of the work.

When threading is switched off
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:meth:`~dualmesh.Problem.set_num_threads` states how many threads are wanted;
:meth:`~dualmesh.Problem.effective_threads` reports how many will be used.  The
second can be smaller than the first, for three reasons.  The first is trivial:
if the extension was compiled without OpenMP the loop is serial.  The other two
matter.

**An object of the problem is defined in Python.**  A kernel, a material, a
function or an initial condition written in Python has to be called back into
the interpreter, and CPython's global interpreter lock permits only one thread
to execute Python bytecode at a time.  Several threads calling such an object
would serialise on the lock, and the lock would have to be acquired correctly
from each of them.  Rather than risk that, every object of the problem is asked
whether it is thread-safe when the problem is resolved; a Python-defined object
answers no, and a single such answer sets
:meth:`~dualmesh.Problem.thread_safe` to false for the whole problem and forces
``effective_threads`` to 1.  The fallback is silent and automatic -- a
calculation that adds a Python function to an otherwise C++ problem runs
correctly, just serially.  A user who wants the threading back must express that
object in C++ or as one of the built-in types.

**The mesh is too small.**  Starting a team of threads costs a few microseconds,
which is more than a small mesh takes to assemble, so the library refuses to
thread a problem that cannot pay for it.  The threshold is the compile-time
constant :math:`\texttt{kMinElementsPerThread} = 400`, and the number of threads
actually used is

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
measurements it does not.  Two things make that true.

First, every per-element quantity is computed identically regardless of
threading.  The scratch space is private, the loop bounds are assigned
statically, and the integrand of an element depends only on that element's
geometry, its nodal values and the problem's objects -- never on which elements
have already been visited.  There is no accumulator carried across elements and
no shared mutable state other than the result arrays, so each :math:`R_K` and
each :math:`J_K` in :eq:`elementsum` is bit-for-bit what a serial run computes.

Second, the *order* in which those contributions are summed is the only thing
threading changes, and the library keeps that order as stable as it can.  The
Jacobian's triplet lists are concatenated in thread order and each thread's list
is built in static-schedule order, so the merged list is a deterministic
permutation for a given thread count.  The residual's atomic updates are the
exception: the order in which two threads add to the same entry is decided by
the hardware, and floating-point addition is not associative, so the library
does not guarantee a bit-identical residual across thread counts -- only one
that differs, if at all, by rounding at the last bit.  In the measurements below
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
mesh -- 8000 elements and 27 783 degrees of freedom -- on a machine with two
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

The single-thread times are about a third of what they were before the
efficiency work described in the changelog (1.825 s, 0.759 s and 2.165 s): the
geometry of every integration point is now mapped once, the reference-element
data are cached, the derivative loops of the automatic differentiation
vectorise, and the derivative arrays are copied only as far as they are used.
The speedup at two threads is between 1.24 and 1.55 against an ideal of 2.  The
shortfall is the part of an assembly that is not threaded -- the boundary
conditions, the merge of the per-thread triplet lists and the construction of
the sparse matrix -- and it weighs most on the finite element method, whose
threaded element loop has become the cheapest.  Going from two to four threads
on two cores adds little, as expected.  Only the assembly and the error norms
are threaded: the linear solve is handed to Eigen [Eigen]_ and the library
makes no attempt to parallelise it, so a calculation dominated by the solve
rather than the assembly will see little benefit from threads.

Distributed-memory execution
----------------------------

The decomposition
^^^^^^^^^^^^^^^^^

A distributed run begins by partitioning the elements: every element of the
global mesh is assigned to exactly one rank.  Rank :math:`r` then extracts the
sub-mesh made of its own elements together with *every node those elements
touch*, which includes nodes on the partition boundary that other ranks also
hold.  Nodes are renumbered locally and the rank remembers the global index of
each of its local nodes; blocks are preserved and side sets are restricted to
the sides that survive, with a side set that ends up empty on a rank still
created, so a boundary condition naming it remains valid everywhere.  On that
sub-mesh the rank builds an ordinary :class:`~dualmesh.Problem`, and the user
defines exactly the same variables, kernels, materials and boundary conditions
on it as in a serial run.  Every discretisation, every physics module and the
automatic differentiation work unchanged, because the rank-local object is not a
special kind of problem -- it is a problem on a smaller mesh.

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

Node ownership
^^^^^^^^^^^^^^

Several ranks may hold the same node, but exactly one **owns** it.  The owner is
the smallest rank index among those that touch the node, a rule every rank
evaluates identically from the partition without any communication.  Ownership
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

It settles the application of quantities attached to a node rather than to an
element.  A concentrated load belongs to a node, so only the owning rank applies
it; otherwise it would be applied once per rank holding that node.  A load given
by coordinates is first resolved to the nearest node, and since each rank sees
only its own sub-mesh, each would find a nearest node of its own; the ranks
therefore reduce the distances, and the rank whose node is globally nearest
keeps the load, ties being broken by the ownership mask.  The test suite finds a
point-source solution agreeing with the serial one to
:math:`1.0 \times 10^{-14}` on four ranks.

The distributed linear solver
-----------------------------

Matrix-free Krylov iteration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The global matrix :math:`J` is never assembled.  A distributed sparse matrix
would require a global numbering of its entries and a redistribution of rows,
and neither is necessary, because :eq:`ranksum` already gives the matrix-vector
product.  If :math:`x` is consistent, then

.. math::
   :label: matvec

   J x = \left( \sum_r J_r \right) x = \sum_r \left( J_r x \right) ,

so each rank multiplies by its own local matrix and the partial results are made
consistent by a single exchange with the neighbouring ranks.  The product is
exact, not an approximation: no global numbering is needed, the communication is
confined to the partition boundary, and the local matrices are the same Eigen
sparse matrices a serial run would build.

A Krylov method needs only :eq:`matvec` and the inner product :eq:`dotproduct`,
so both solvers are implemented directly on top of them.  The default is
BiCGSTAB [VanDerVorst1992]_, which converges for a general, non-symmetric matrix
and is the right default here because the dual mesh and finite volume methods do
not produce a symmetric matrix.  The alternative is the conjugate gradient
method [HestenesStiefel1952]_, which costs about half as much per iteration but
requires a symmetric positive definite matrix; the Galerkin finite element
method gives one, the other discretisations do not.  Selecting ``"cg"`` with a
non-symmetric matrix produces a failure to converge, and the error message says
so rather than leaving the user to guess.

The preconditioner
^^^^^^^^^^^^^^^^^^

Krylov iteration counts on an unpreconditioned elliptic system grow with the
mesh size, so a preconditioner :math:`M^{-1} \approx J^{-1}` is essential.
Three are offered.

**Jacobi** takes :math:`M` to be the diagonal of the global matrix.  The
diagonal is a vector, so it is formed by taking each rank's local diagonal and
making it consistent with one exchange; applying :math:`M^{-1}` is then an
elementwise division needing no communication at all.  It needs no
factorisation, it is cheap per iteration, and it is the same operator however
the mesh is divided, but it converges slowly: its iteration count grows like
the number of elements across the mesh.

**Restricted additive Schwarz** is a domain decomposition method: solve the
problem approximately on each of a set of overlapping subdomains and combine the
corrections.

*The subdomains.*  Rank :math:`r` holds the elements the partitioner gave it.
Its Schwarz subdomain :math:`\Omega_r^\delta` is those elements extended by
:math:`\delta` layers of its neighbours' elements, where :math:`\delta` is the
``overlap`` option (default 1): one layer adds every element that touches a node
of the rank's own elements, and each further layer repeats the step.  Every rank
holds the whole mesh and the whole partition, so each grows its own subdomain
without communicating.  The nodes of :math:`\Omega_r^\delta` that no element of
rank :math:`r` touches are its *ghost* nodes.

*The subdomain matrix.*  Write :math:`R_r` for the restriction of a global
vector to the degrees of freedom of :math:`\Omega_r^\delta`.  The subdomain
matrix is the global matrix restricted to the subdomain,

.. math::
   :label: subdomainmatrix

   A_r = R_r J R_r^{\mathsf{T}} ,

and every one of its rows must be the *fully assembled* row of :math:`J`.  No
rank holds such rows on its own: by :eq:`ranksum` the entry :math:`J_{ij}` is
the sum of the entries :math:`(J_s)_{ij}` of every rank :math:`s` whose elements
touch both nodes.  So at setup every rank tells each neighbour which of the
neighbour's nodes lie in its subdomain, and whenever the matrix changes (once
per Newton iteration) each rank sends every neighbour the entries of its local
matrix between those nodes, identified by global degree of freedom; the
receiver adds them to its own.  The result is exactly :eq:`subdomainmatrix`.
An earlier version of the library factorised the local matrix :math:`J_r`
instead.  At a node on the partition boundary that matrix holds only the
rank's own share of the row, a Neumann-like approximation to the true row, and
that, together with the absence of overlap, is why the earlier Schwarz
preconditioners stopped converging at eight ranks.

*The subdomain solve.*  :math:`A_r^{-1}` is replaced by an incomplete LU
factorisation without fill, ILU(0) [Saad2003]_ (``subdomain_solver = "ilu"``,
the default), or computed exactly by a sparse LU factorisation (``"lu"``).

*The restriction.*  Classical additive Schwarz would add every subdomain's
whole correction, :math:`M^{-1} = \sum_r R_r^{\mathsf{T}} A_r^{-1} R_r`, and
so add two or more corrections wherever subdomains overlap.  The **restricted**
variant of Cai and Sarkis [CaiSarkis1999]_ keeps each subdomain's correction
only on the degrees of freedom the rank owns:

.. math::
   :label: ras

   M_{\text{RAS}}^{-1} = \sum_r \tilde{R}_r^{\mathsf{T}} A_r^{-1} R_r ,

where :math:`\tilde{R}_r` restricts to the owned degrees of freedom, so the
:math:`\tilde{R}_r^{\mathsf{T}}` form a partition of unity and the corrections
tile the domain.  Applying :eq:`ras` takes one exchange to fetch the residual
at the ghost nodes from their owners, the local solve, and one exchange to make
the result consistent.

**Two-level Schwarz** (``"two_level_schwarz"``, the default) adds a coarse
correction.  A coarse level is needed because of a property of one-level
methods that no choice of subdomain solver can fix.  One application of
:eq:`ras` moves information only from a subdomain to its neighbours, so
information from one end of the domain reaches the other only after as many
iterations as there are subdomains across it.  For an elliptic problem, whose
solution at any point depends on the data everywhere, the iteration count of a
one-level method therefore grows with the number of subdomains.  A coarse level
supplies the missing global transport by solving a small problem posed on the
whole domain.  The theory is due to Dryja and Widlund [DryjaWidlund1994]_ and is
set out in full by Toselli and Widlund [ToselliWidlund2005]_: with a suitable
coarse space and an overlap proportional to the subdomain size, the condition
number of the preconditioned operator is bounded independently of the number of
subdomains.

The coarse space is Nicolaides' [Nicolaides1987]_: one basis function per
subdomain and variable, equal to one on the degrees of freedom the subdomain
owns and zero elsewhere.  Degrees of freedom whose equation has been replaced
by the identity (prescribed values, or no kernel acting on them) are left out,
because those rows would add to the coarse matrix an identity that has nothing
to do with the operator.  Writing :math:`Z` for the matrix of basis vectors,
the coarse matrix :math:`A_0 = Z^{\mathsf{T}} J Z` is dense and of size
:math:`(n_{\text{ranks}} \, n_{\text{var}})^2`; each rank accumulates the
contributions of its local entries and one reduction sums them, and its inverse
is formed once per Newton iteration.  With :math:`Q = Z A_0^{-1} Z^{\mathsf{T}}`
the coarse correction and :math:`P = I - J Q`, the two levels are combined
multiplicatively, coarse first:

.. math::
   :label: adef1

   M^{-1} = M_{\text{RAS}}^{-1} P + Q ,
   \qquad\text{that is}\qquad
   z_0 = Q r , \quad z = z_0 + M_{\text{RAS}}^{-1} (r - J z_0) .

This is the operator Tang, Nabben, Vuik and Erlangga call A-DEF1
[Tang2009]_.  Adding the two corrections instead, :math:`M_{\text{RAS}}^{-1} +
Q` (their AD), was implemented first and measured to be *worse* than the
one-level method: the restricted local corrections already carry the smooth part
of the error that the coarse correction adds again, and the sum overshoots.
:eq:`adef1` costs one more matrix-vector product per application, which is small
next to the subdomain solves.

**The conjugate gradient method** needs a symmetric positive definite
preconditioner, and the restricted form :eq:`ras` is not symmetric, nor is
:eq:`adef1`.  With ``linear_solver = "cg"`` the Schwarz options are therefore
applied in their classical form, :math:`\sum_r R_r^{\mathsf{T}} A_r^{-1} R_r`
(each ghost part of a correction is sent back to the owner and added), and the
coarse level additively, :math:`M^{-1} + Q`.  For a symmetric matrix both the
exact and the ILU(0) subdomain solves are symmetric, so the whole operator is.
The classical form is known to be the weaker of the two [CaiSarkis1999]_, and
the measurements below agree.

Measured iteration counts
^^^^^^^^^^^^^^^^^^^^^^^^^

The figures below are for the Poisson problem :math:`-\nabla^2 u = 1` on the
unit square with :math:`u = 0` on the boundary, :math:`48 \times 48` ``Quad4``
elements (2401 unknowns), the dual mesh control domain method, the graph
partitioner and BiCGSTAB to a relative tolerance of :math:`10^{-10}`.  One
BiCGSTAB iteration applies the preconditioner twice.  The problem is held fixed
while ranks are added, so this measures how well each preconditioner tolerates
a finer decomposition.

.. table:: BiCGSTAB iterations, 48 by 48 mesh, increasing rank count

   ==========================================  ======  ==  ==  ==  ==
   Preconditioner (overlap, subdomain solver)  1       2   4   8   16
   ==========================================  ======  ==  ==  ==  ==
   ``jacobi``                                  50      49  49  50  49
   ``additive_schwarz`` (1, ilu)               24      26  29  33  33
   ``additive_schwarz`` (1, lu)                1       9   13  17  17
   ``two_level_schwarz`` (0, ilu)              24      32  29  30  36
   ``two_level_schwarz`` (1, ilu), default     24      25  23  25  24
   ``two_level_schwarz`` (2, ilu)              24      24  26  25  27
   ``two_level_schwarz`` (1, lu)               1       11  13  15  16
   ==========================================  ======  ==  ==  ==  ==

On a :math:`128 \times 128` mesh (16 641 unknowns) the default takes 66, 60 and
64 iterations on one, four and sixteen ranks; without the coarse level the
counts are 66, 76 and 86, and Jacobi takes 137, 131 and 132.  In three
dimensions, on a :math:`24^3` ``Hex8`` mesh (15 625 unknowns), the default takes
15, 16 and 14 iterations on one, four and eight ranks and Jacobi 26 on each.

Three things follow.  First, the default's iteration count is flat in the
number of ranks, which is the property that makes adding ranks pay: on the
:math:`48 \times 48` problem it is between 23 and 25 from one rank to sixteen.
Second, one layer of overlap is what makes the difference: without overlap the
two-level count drifts upward (24 to 36), and a second layer buys nothing
further here.  Third, the exact subdomain solve roughly halves the iteration
count in two dimensions; it is not the default because the cost of an exact
factorisation of a three-dimensional subdomain grows much faster than the
subdomain (see :ref:`the choice of linear solver <linear-solver-choice>`).

With ``linear_solver = "cg"`` on the finite element discretisation of the same
problem, Jacobi takes 69 iterations on any number of ranks, and the classical
one- and two-level Schwarz forms take 38 on one rank and 64 to 73 on four and
eight.  For a symmetric problem on many ranks BiCGSTAB with the default is
therefore the better choice, despite its two preconditioner applications per
iteration.

These tests are part of the suite: ``dualmesh_parallel_tests``, launched with
``mpirun``, checks that every preconditioner, overlap and subdomain solver
reproduces the serial solution (to :math:`10^{-8}`, measured at
:math:`10^{-13}`), including a two-variable elasticity problem, and that the
default's iteration count on the :math:`48 \times 48` problem stays below 45
on whatever number of ranks it is run.

Partitioning
------------

How the elements are split among the ranks decides both the load balance -- how
evenly the work is shared -- and the communication volume, measured by the
*edge cut*, the number of mesh faces whose two elements lie in different parts.
Three partitioners are available through
:func:`~dualmesh.partition_mesh` and through the ``partitioner``
argument of :class:`~dualmesh.DistributedProblem`.

**Recursive coordinate bisection** is purely geometric.  It computes the
centroid of every element, finds the longest axis of the bounding box of those
centroids, splits the set in half along that axis in proportions matching the
number of parts each half must produce, and recurses.  It needs no connectivity
information, it is fast, and it is deterministic.  Its parts are rectangular
boxes, which is close to optimal on a structured grid and wasteful on an
unstructured mesh whose features do not line up with the axes.  The idea is
Berger and Bokhari's [BergerBokhari1987]_.

**Graph growing** uses the mesh topology.  It builds the element adjacency
graph, in which two elements are neighbours when they share a whole face, and
grows each part outward from a seed element through that graph until the part
reaches its quota.  The seed of each new part is chosen to be an unassigned
element with few unassigned neighbours -- a corner of what is left -- so the
parts do not start from the middle of the remaining region.  Parts produced this
way are connected and follow the mesh rather than the coordinate axes, and any
element left over from a disconnected piece joins the smallest part.  This is
the default for :class:`~dualmesh.DistributedProblem`.

**METIS** calls the multilevel :math:`k`-way algorithm of Karypis and Kumar
[KarypisKumar1998]_ on the same element adjacency graph.  It coarsens the graph
by contracting edges, partitions the small coarse graph, and refines the
partition as it uncoarsens, which finds cuts a single-pass method cannot.  It is
available only when the library was built against METIS, reported by
:func:`~dualmesh.have_metis`; asking for it in a build without METIS prints a
warning and falls back to graph growing.  METIS is the partitioner libMesh
[libMesh2006]_, and therefore MOOSE [MOOSE2025]_, uses.

The choice matters less than it might appear, and it depends on the mesh.  On a
structured :math:`20 \times 20` grid of ``Quad4`` elements, coordinate bisection
cuts 40 faces into four parts, METIS cuts 42 and graph growing cuts 76: the
geometric method wins, because a structured grid is exactly the case its
rectangular parts suit.  On a triangulation of the same square into 800 ``Tri3``
elements, coordinate bisection cuts 40, METIS 48 and graph growing 73.  All
three give parts of nearly equal size in every case measured; the test suite
requires no part to exceed the average by more than a quarter, and none does.
Use the default graph partitioner unless partition quality is measurably hurting
a run, prefer coordinate bisection on structured or nearly structured meshes,
and build with METIS for large unstructured meshes, where the multilevel
algorithm's advantage grows with the problem size.

What runs in parallel, and what does not
----------------------------------------

**The global mesh is replicated on every rank.**  Each rank reads the whole mesh
at construction and then keeps only its own sub-mesh, but the whole mesh exists
in memory on every rank while that happens.  The degrees of freedom are
distributed; the mesh, transiently, is not.  This bounds the problem size by the
memory of a single rank, and is the first thing to hit on a very large mesh.

**The cell-centred finite volume method does not run on more than one rank.**
Its unknowns live at cell centroids and at boundary faces rather than at mesh
nodes, but the distributed bookkeeping -- the ownership mask, the shared-entity
exchange and the ownership-weighted inner product -- is written in terms of mesh
nodes.  On a single rank every exchange is a no-op and the method appears to
work; on two or more ranks the exchanges address the wrong entities and the
solve fails.  Use ``dmcdm``, ``fem`` or ``hfvm`` for distributed runs.

**Adaptive refinement is not distributed.**  The refinement described in
:doc:`adaptivity` operates on a whole mesh in one process.  There is no
distributed error indicator, no parallel marking and no parallel refinement, so
an adaptive loop and an MPI run cannot currently be combined.

**Only the assembly is threaded.**  All four discretisations thread their
element and face loops, including the cell-centred finite volume method, which
uses the same scratch, atomic and triplet arrangement.  What is *not* threaded
is the rest of a solve: the integrated boundary conditions, the concentrated
loads, the merge of the triplet lists, the construction of the sparse matrix and
the linear solve itself, which is handed to Eigen [Eigen]_ and which the library
makes no attempt to parallelise.  The measured assembly speedup on two cores is
between 1.53 and 1.66, and what fraction of a whole solve that represents
depends entirely on how expensive the linear solve is for the problem at hand.

**Gathering the solution does not scale.**
:meth:`~dualmesh.DistributedProblem.gathered_values` builds a vector of the
global size on every rank.  It is convenient for testing and for small problems
and is not meant for large runs; those should write results with
:meth:`~dualmesh.DistributedProblem.write_vtu`, which produces one ``.vtu`` file
per rank plus a ``.pvtu`` index that ParaView and VisIt open as a single data
set.

**The distributed answer does not depend on the decomposition.**  This is the
one property that was verified rather than merely intended.  On four ranks, for
each of ``dmcdm``, ``fem`` and ``hfvm``, for both the coordinate-bisection and
graph partitioners and for both the Jacobi and additive Schwarz
preconditioners, the distributed steady conduction solution agrees with the
serial one node by node to between :math:`3.9 \times 10^{-13}` and
:math:`3.3 \times 10^{-12}`, which is the linear solver's tolerance rather than
any decomposition effect.  A transient solve over ten steps agrees to
:math:`2.6 \times 10^{-15}`, and the point-source case to
:math:`1.0 \times 10^{-14}`.
