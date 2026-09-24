Input files
===========

Problems can also be described in a YAML input file and run from the command
line, which is convenient for parameter studies and for keeping a record of a
calculation:

.. code-block:: console

   dualmesh run bus_bar.yaml

The blocks mirror the Python API one to one.

.. literalinclude:: ../examples/bus_bar.yaml
   :language: yaml

Blocks
------

``mesh``
    ``type`` is ``line``, ``rectangle``, ``box``, ``annulus``, or ``file``; the
    remaining entries are the arguments of the corresponding generator (see
    :doc:`api`), or ``filename`` for a mesh file.

``problem``
    ``method`` (``dmcdm``, ``fem``, ``hfvm`` or ``zfvm``), ``coordinates``
    (``cartesian``, ``axisymmetric``, ``spherical``), and optionally
    ``threads`` (the number of assembly threads; every core by default) and,
    for ``zfvm``, ``boundary_gradient``.

``parallel``
    Optional.  The options of the distributed solver, used when the file is
    run under ``mpirun`` on more than one process and ignored otherwise:
    ``partitioner``, ``linear_solver``, ``preconditioner``, ``overlap``,
    ``subdomain_solver``, ``linear_tolerance`` and ``linear_max_iterations``
    (see :class:`~dualmesh.DistributedProblem`).

``variables``
    One entry per unknown; each may carry ``initial_condition`` and ``blocks``.

``functions``
    Named expressions in ``x``, ``y``, ``z``, ``t``, which can then be used
    wherever a coefficient is expected.

``kernels``, ``boundary_conditions``, ``materials``, ``point_sources``
    One entry per object.  Each needs a ``type`` — a registered object name,
    listed by ``dualmesh list`` — and the parameters of that object.

``executioner``
    ``type: steady`` or ``type: transient``, plus the solver options of
    :meth:`~dualmesh.Problem.solve` (``nonlinear_solver``, ``relaxation``,
    ``load_factors``, ``linear_solver``, ``preconditioner``,
    ``max_iterations``, ...) and, for transients, the arguments of
    :meth:`~dualmesh.Problem.solve_transient` (``end_time``, ``dt``,
    ``theta``, ``time_stepper``, ...).

``outputs``
    ``vtu``, ``csv``, ``mesh_file``, ``cell_properties``, ``reactions`` (pairs
    of variable and boundary), and ``point_values``.

A block, a problem setting or an output that is not one of these is an error,
with a suggestion when the name is close to a valid one: a misspelled
``kernel:`` would otherwise leave the kernels out of the problem without a
word.

Values
------

Because YAML is not strict about numbers, any string that is a number is read
as a number (so ``1.0e6`` works).  A string that begins with ``=`` is an
expression in ``x``, ``y``, ``z``, ``t``:

.. code-block:: yaml

   boundary_conditions:
     top:
       type: DirichletBC
       variable: temperature
       boundary: top
       value: "= 500*(1 - 10*x^2)"

Every other string is passed through unchanged, so variable names, boundary
names, and the names of functions defined in the ``functions`` block keep
working.

Expressions are compiled by the library's own parser into a short program
evaluated in C++; they are never passed to Python's ``eval``, so an input file
cannot run code.  The grammar is given in :doc:`user_guide/problem_setup`.

Running in parallel
-------------------

The same input file runs on several processes under MPI, when the library was
built with it:

.. code-block:: console

   mpirun -n 4 dualmesh run input.yaml

The mesh is partitioned, each process solves its part with the distributed
solver configured by the ``parallel`` block, a ``vtu`` output becomes one file
per process plus a ``.pvtu`` index that ParaView opens as one data set, and the
``csv`` and ``point_values`` outputs are computed from the gathered solution
and written by the first process.  ``reactions`` are not yet available in a
distributed run.  The cell-centred method (``zfvm``) runs on one process, with
threads.

Other commands
--------------

.. code-block:: console

   dualmesh list                       # every registered object type
   dualmesh list --category Kernel     # only kernels
   dualmesh list --module fluids       # only the fluids module
   dualmesh describe RobinBC           # parameters and documentation
   dualmesh --version
