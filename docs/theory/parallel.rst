Parallel execution
==================

This page states how the code runs in parallel: the threads of one process,
the distribution of a problem over several processes, the exchange of the
values at shared nodes, the partitioning of the mesh, the distributed linear
solver, and which methods every process calls.  To launch a parallel run, see
:doc:`/user_guide/solving`.

.. contents::
   :local:
   :depth: 1

Two levels of parallelism
-------------------------

The code runs in parallel on two independent levels.

Within one process, Open Multi-Processing (OpenMP) threads share the assembly
loops, the loops that visit every element and add its contribution to the
residual vector and the Jacobian matrix.  The threads share the memory of the
process, so the mesh and the solution exist once.
:meth:`~dualmesh.Problem.set_num_threads` sets the number of threads, and the
default is the value of the ``OMP_NUM_THREADS`` environment variable or, when it
is unset, the number of cores.  This level is limited to the cores and the memory
of one machine.

Across processes, a :class:`~dualmesh.Problem` splits the elements of its mesh
among the processes of the Message Passing Interface (MPI).  Each process, or
rank, holds its own part of the problem, and the ranks exchange data through
MPI, so the memory of a cluster is available.  You launch a distributed script
with ``mpirun``, as in ``mpirun -n 8 python script.py``, and the same script
runs serially with ``python script.py``.  A problem splits itself when it runs
on more than one process, and ``dualmesh.Problem(mesh, distributed=True)``
takes the distributed path on one process as well, which gives the serial
answer.

The two levels compose, so four ranks of four threads each use a node of
sixteen cores.

Threads
-------

Assembly computes the residual :math:`\mathbf{R}` and, when a Jacobian is
needed, the matrix :math:`\mathbf{J} = \partial\mathbf{R}/\partial\mathbf{U}`,
with :math:`\mathbf{U}` the vector of unknowns.  Both are sums over the elements
:math:`K` of the mesh :math:`\mathcal{T}_h`,

.. math::
   :label: parallel-elementsum

   \mathbf{R} = \sum_{K \in \mathcal{T}_h} \mathbf{R}_K , \qquad
   \mathbf{J} = \sum_{K \in \mathcal{T}_h} \mathbf{J}_K ,

where :math:`\mathbf{R}_K` and :math:`\mathbf{J}_K` hold the terms of the nodes
of :math:`K`.  In the dual mesh control domain method (DMCDM) an element
contributes the volume terms of the control-domain pieces inside it and the
fluxes through the interfaces between them, all in one loop over :math:`K`.  The
terms of one element depend on no other element, so the elements split among
the threads with no reordering.  Only the sums of :eq:`parallel-elementsum` need
care, because two elements that share a node add to the same entries.  Three
arrangements make them safe.

**Per-thread scratch space.**  Everything that an element computation writes,
other than the global result, lives in a scratch object private to each thread:
the gathered nodal values, the indices of the unknowns, the local residual, the
integration points and their geometry, and the context passed to the kernels.
Each thread allocates its scratch once and reuses it for every element it
visits.  State that is built on first use, such as the list of exterior sides
and the reference elements, is built before the loop starts, so no two threads
fill the same cache.

**Atomic updates of the residual.**  The residual is one shared array, and each
addition to it is an OpenMP atomic update, which makes the read, the addition
and the write of one entry indivisible.  It is the only shared write of the
loop, and a serial assembly writes directly.

**Per-thread lists of matrix entries.**  The Jacobian is built as a list of
(row, column, value) entries and then converted to a compressed sparse matrix by
Eigen [Eigen]_.  Each thread appends to its own list.  After the loop the lists
are joined in thread order, and the conversion sums the entries that share a
row and a column, which adds the contributions of different threads.

The boundary terms, the integrated boundary conditions and the concentrated
loads, are assembled after the loop by one thread.  Besides the assembly, the
error norms are threaded.  The linear solve runs in PETSc, the Portable,
Extensible Toolkit for Scientific Computation [PETSc2023]_, which the code does
not thread.

The number of threads used
~~~~~~~~~~~~~~~~~~~~~~~~~~

:meth:`~dualmesh.Problem.effective_threads` reports the number of threads that
the assembly uses, which can be smaller than the number requested.

- A build without OpenMP assembles on one thread.
- A kernel, property, function or initial condition written in Python calls back
  into the interpreter, and the global interpreter lock of CPython lets one
  thread run Python at a time.  One such object makes
  :attr:`~dualmesh.Problem.thread_safe` false and the assembly serial.  To thread
  the problem, write that object in C++ or use a built-in type.
- Starting a team of threads costs more than a small mesh takes to assemble.
  With :math:`N_{\mathrm{el}}` the number of elements and
  :math:`n_{\mathrm{req}}` the number of threads requested, the number used is

  .. math::

     n_{\mathrm{eff}} = \max\left\{ 1,\ \min\left( n_{\mathrm{req}},\
     \left\lfloor N_{\mathrm{el}}/400 \right\rfloor \right) \right\} ,

  so a mesh of fewer than 400 elements is assembled on one thread.

Reproducibility
~~~~~~~~~~~~~~~

The terms of each element are the same whatever the number of threads: the
scratch is private, the loop is split in a static schedule, and an element
depends only on its own geometry, its nodal values and the objects of the
problem.  Threading changes only the order of the sums.  The lists of matrix
entries are joined in a fixed order for a given number of threads.  The order of
two atomic additions to one residual entry is decided by the hardware, and
floating-point addition is not associative, so residuals of different thread
counts can differ by rounding in the last bits.  For results identical to the
last bit, run on one thread.

Distributed problems
--------------------

The decomposition
~~~~~~~~~~~~~~~~~

Every element belongs to exactly one rank.  Rank :math:`r` holds the sub-mesh of
its elements together with every node that they touch, including the nodes on
the boundary between parts, which other ranks hold too.  The rank numbers its
nodes locally and keeps the global index of each.  The blocks are kept, and each
side set is restricted to the sides that the rank holds.  Every rank creates
every side set, also where it is empty, so a boundary condition that names it is
valid everywhere.  Each side set is also a node set of the same name, so a value
prescribed on a boundary reaches every rank that holds one of its nodes, whatever
rank holds the side.

When a whole mesh is given, the first rank partitions it and sends every rank
its part, with the global numbering of the whole mesh.  A Python script that
calls a mesh generator or reads a mesh file runs on every rank, so the whole mesh
is built on every rank before the problem splits it.  The memory of one rank
therefore bounds the size of the mesh.

On its sub-mesh each rank builds an ordinary problem, with the same variables,
kernels, property objects and boundary conditions on every rank.  Every element
belongs to one rank, so the element sums :eq:`parallel-elementsum` split
exactly:

.. math::
   :label: parallel-ranksum

   \mathbf{R} = \sum_r \mathbf{R}_r , \qquad
   \mathbf{J} = \sum_r \mathbf{J}_r ,

with :math:`\mathbf{R}_r` and :math:`\mathbf{J}_r` the contributions of the
elements of rank :math:`r`.  In DMCDM, a node inside a part receives its whole
control domain from one rank, and a node on the boundary between parts receives
a piece of its control domain from each rank that holds it.

The gather-scatter
~~~~~~~~~~~~~~~~~~

A local vector in which every rank that holds a shared node has the same,
complete value there is consistent.  A sum over the ranks of the partial values
at each shared node turns the partial contributions into a consistent vector.
Every such exchange goes through one gather-scatter layer.  A rank gives the
layer the global index of each of its nodes and nothing else, and the layer
finds which ranks hold each index, which of them owns it, and how to combine
their values.

The owner of a node is the smallest rank that holds it.  The default library of
the layer, ``gather_scatter="petsc"``, finds it with PETSc's star forests.  Each
global index has a home rank in an even layout of the indices.  Every copy of
the index sends the pair (rank, local index) to the home rank, which keeps the
minimum pair and sends it back to every copy.  The copies that a rank does not
own then become the leaves of a star forest whose roots are the copies of the
owners.  A sum is a reduction from the leaves to the roots followed by a
broadcast from the roots to the leaves, and a copy from the owners is the
broadcast alone.

``gather_scatter="gslib"`` takes gslib, the gather-scatter library of Nek5000
and nekRS [gslib]_ [Fischer2021]_, in a build with gslib
(:func:`~dualmesh.have_gslib`).  At setup gslib times pairwise messages between
the ranks that share nodes, a crystal router and an all-reduce, and keeps the
fastest for the mesh and the machine.  The owner is again the smallest rank,
which gslib finds by a minimum over the ranks.  Both libraries give the same
answer.

Node ownership
~~~~~~~~~~~~~~

Several ranks may hold a node, and exactly one owns it.  Ownership settles three
things.

- Counting: the number of unknowns of the whole problem is the sum over the
  ranks of the unknowns that each owns.
- Inner products: an inner product sums over the owned unknowns of each rank
  and then over the ranks,

  .. math::

     \langle \mathbf{a}, \mathbf{b} \rangle
     = \sum_r \ \sum_{i\ \text{owned by}\ r} a_i\, b_i ,

  so every unknown counts once and :math:`\sqrt{\langle\mathbf{a},\mathbf{a}\rangle}`
  is the global norm.
- Nodal quantities: a concentrated load belongs to a node, so only its owner
  applies it.  A load given by coordinates goes to the nearest node, and each
  rank sees only its part of the mesh, so the ranks compare their distances and
  the rank of the globally nearest node keeps the load, with a tie broken by
  ownership.  A value prescribed at the node nearest to a point, such as the
  pressure level of an enclosed flow, is resolved the same way, so exactly one
  global node takes it.

Partitioning
------------

The split of the elements among the ranks sets the balance of the work and the
amount of communication.  The communication grows with the edge cut, the number
of faces whose two elements lie in different parts.  The partitioners are those
of PETSc, chosen with the ``partitioner`` argument of :class:`~dualmesh.Problem`
and available alone through :func:`~dualmesh.partition_mesh`.

PETSc's unstructured mesh, DMPlex, receives the corner nodes of every element,
so every element type, the quadratic ones included, is partitioned by its
corners.  DMPlex builds the graph of the elements that share a face.

- ``"ptscotch"``, PT-Scotch [Chevalier2008]_, and ``"parmetis"``, ParMETIS, the
  parallel form of the multilevel :math:`k`-way algorithm of Karypis and Kumar
  [KarypisKumar1998]_, divide that graph so as to cut few faces.
- ``"simple"`` takes contiguous blocks of elements in their order.
- ``"automatic"``, the default, takes PT-Scotch, else ParMETIS, else contiguous
  blocks, by what the build of PETSc has.

The first rank holds the whole mesh in DMPlex, and ``DMPlexDistribute`` moves
every element, with its nodes, coordinates, block, sides and node sets, to the
rank of its part, where the rank rebuilds the mesh it holds.

The distributed linear solver
-----------------------------

The Jacobian is one distributed PETSc matrix.  The owned unknowns of each rank
are numbered contiguously, rank after rank, and a shared unknown takes the
number of its owner.  Each rank lists the entries of its own elements in that
global numbering, and the coordinate-format assembly of PETSc sends each entry to
the rank that owns its row and adds repeated positions.  By
:eq:`parallel-ranksum` the result is the global Jacobian.  A prescribed unknown
replaces its equation on every rank that holds its node, and only the owner
enters the diagonal of that row, so the row is the same for any number of ranks
that share the node.  The right side contributes its owned entries, and the
owned entries of the solution are copied to the other ranks that hold them.

Without options, the solver chooses by the structure of the system, serial and
distributed alike.

- A system small enough to factorise is factorised: in one dimension at any
  size, in two dimensions up to 100 000 unknowns and in three dimensions up to
  4000.  The factorisation is MUMPS, the multifrontal massively parallel sparse
  direct solver [Amestoy2001]_, when PETSc has it.
- A larger flow takes flexible GMRES, the generalized minimal residual method,
  with the Schur complement split of the least squares commutator.
- A system that mixes linear and quadratic variables is factorised.
- A system without displacements takes GMRES with the algebraic multigrid
  BoomerAMG of hypre [FalgoutYang2002]_.
- A solid takes GMRES with the smoothed aggregation multigrid of PETSc (GAMG),
  given the rigid body modes of the owned rows as its near null space.
- A solid coupled to other physics takes the split by physics.

An iterative choice that has not converged in 500 iterations falls back to the
factorisation.  PETSc tests the preconditioned residual, so the code also checks
the true residual :math:`|\mathbf{b} - \mathbf{A}\mathbf{x}|` of a converged
solve, with :math:`\mathbf{A}` the matrix, :math:`\mathbf{b}` the right side and
:math:`\mathbf{x}` the solution, and marks the solve as failed when it exceeds
:math:`\sqrt{\epsilon_r}\,|\mathbf{b}|`, with :math:`\epsilon_r` the relative
tolerance.  The ``petsc_options`` of a solve replace the choice, and the options
``-dualmesh_schur`` and ``-dualmesh_split`` choose a split.
:doc:`/user_guide/solving` lists them.

Collective and local methods
----------------------------

In a distributed run every rank runs the same script.  A collective method
combines the parts of all the ranks, so every rank calls it: a call that only
some ranks make waits for the others and never returns.  Guard the printing with
:func:`~dualmesh.is_root`, and leave the call itself unguarded:

.. code-block:: python

   value = problem.integrate("temperature")  # every rank
   if dualmesh.is_root():
       print(value)

A local method answers from the part of the rank that calls it, so one rank may
call it alone.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Kind
     - Methods of :class:`~dualmesh.Problem`
   * - Collective
     - ``solve``, ``solve_transient``, ``solve_eigenvalue``, ``sample``,
       ``sample_gradient``, ``integrate``, ``error_norms``,
       ``error_indicator``, ``gathered_error_indicator``, ``boundary_flux``
       and the flows of the physics, such as ``heat_flow``,
       ``kernel_boundary_flux``, ``maximum``, ``minimum``, ``domain_volume``,
       ``gathered_values``, ``write_vtu``, ``partition_summary``
   * - Local
     - ``values``, ``node_coordinates``, ``postprocessor_values`` (the same on
       every rank), ``rank``, ``num_ranks``, ``num_owned_dofs``

A collective method returns the value of the whole domain on every rank.
``values`` returns the part of the rank, and ``gathered_values`` the whole field
in the global order.  ``gathered_values`` builds a vector of the global size on
every rank, so it suits tests and small problems.
:meth:`~dualmesh.Problem.write_vtu` writes one ``.vtu`` file per rank and a
``.pvtu`` index, which ParaView and VisIt open as one data set.

Distributed parts of a solve
----------------------------

**Post-processors and reactions.**  Every element and every boundary side
belongs to one rank, so an integral, an average, a boundary flux or a reaction is
the sum of what each rank computes over its own elements, sides and owned nodes,
reduced over the ranks.  A nodal extreme is the extreme over the ranks.  A point
value comes from the rank whose elements contain the point, and from the average
of those ranks when the point lies on the boundary between parts.

**Gap conditions.**  A gap condition pairs every integration point of its
primary side with the closest point of its secondary side, which may belong to
another rank.  Every rank shares the elements of its secondary sides, and a rank
that holds primary sides keeps the secondary elements it lacks as ghost
elements.  It reads their nodes and values and does not integrate them, so each
element is still integrated once.  The share of the flux of the secondary side
is added at the ghost nodes and reaches their owners through the gather-scatter,
and the Jacobian couples the two sides through the global numbering.

**Adaptivity.**  A distributed problem adapts its mesh as a serial one does:
every rank holds the hierarchy of the whole mesh and marks it from the
indicators of the whole mesh (see :doc:`adaptivity`).

The verification cases of parallel runs, with their sources, computed values
and errors, are on the :doc:`/verification` page.
