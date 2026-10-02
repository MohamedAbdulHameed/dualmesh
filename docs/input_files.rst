Input files
===========

Every problem can be described by a Python script or by a YAML input file.
An input file is run from the command line, and it keeps a complete record of
a calculation:

.. code-block:: console

   dualmesh run bus_bar.yaml

The blocks of an input file correspond one to one to the calls of the Python
API.  The bus bar of Example 5.4.3 of Reddy's book reads:

.. literalinclude:: ../examples/bus_bar.yaml
   :language: yaml

Blocks
------

``mesh``
    ``type`` is ``line``, ``rectangle``, ``box``, ``annulus`` or ``file``.  The
    remaining entries are the arguments of the corresponding generator (see
    :doc:`api`), or ``filename`` for a mesh file.

``problem``
    ``method`` (``dmcdm``, ``fem``, ``hfvm`` or ``zfvm``), ``coordinates``
    (``cartesian``, ``axisymmetric``, ``spherical``), and optionally
    ``threads`` (the number of assembly threads, by default every core) and,
    for ``zfvm``, ``boundary_gradient``.

``parallel``
    Optional.  The options of the distributed solver, used when the file is
    run under ``mpirun`` on more than one process and ignored otherwise:
    ``partitioner``, ``linear_solver``, ``preconditioner``, ``overlap``,
    ``subdomain_solver``, ``linear_tolerance`` and ``linear_max_iterations``
    (see :class:`~dualmesh.DistributedProblem`).

``variables``
    One entry per unknown.  Each entry may carry ``initial_condition`` and
    ``blocks``.

``functions``
    Named expressions in ``x``, ``y``, ``z``, ``t``, which can then be used
    wherever a coefficient is expected.

``kernels``, ``boundary_conditions``, ``materials``, ``point_sources``
    One entry per object.  Each entry needs a ``type`` (a registered object
    name, listed by ``dualmesh list``) and the parameters of that object.  A
    boundary condition without a ``boundary`` entry acts on the side set or
    node set whose name equals the name of the entry, e.g., the entry ``left``
    of the example above acts on the side set ``left``.  An explicit
    ``boundary`` is needed when the name of the entry differs from the name
    of the boundary, or when one condition acts on several boundaries.

``executioner``
    ``type: steady`` or ``type: transient``, plus the solver options of
    :meth:`~dualmesh.Problem.solve` (``nonlinear_solver``, ``relaxation``,
    ``load_factors``, ``linear_solver``, ``preconditioner``,
    ``max_iterations``, ...) and, for transients, the arguments of
    :meth:`~dualmesh.Problem.solve_transient` (``end_time``, ``dt``,
    ``theta``, ``time_stepper``, ...).

``postprocessors``
    One entry per post-processor, with its ``type`` (e.g. ``point_value``,
    ``total_reaction`` or ``nodal_extreme_value``) and its parameters.  The
    values are evaluated after every solve and every time step.

``outputs``
    ``vtu`` (the solution for ParaView), ``field_csv`` (the nodal values),
    ``postprocessor_csv`` (the history of the post-processors), ``mesh_file``
    and ``cell_properties``.

A block, a problem setting or an output that is not one of these is reported
as an error, with a suggestion when the name is close to a valid one.  A
misspelt ``kernel:`` therefore stops the run, and the kernels cannot be left
out of the problem unnoticed.

``uq``
    An uncertainty or sensitivity study of the problem (see
    `Uncertainty studies`_).

Values
------

YAML reads some numbers, e.g. ``1.0e6``, as strings.  Every string that
represents a number is therefore converted to a number.  A string that begins with ``=`` is an
expression in ``x``, ``y``, ``z``, ``t``:

.. code-block:: yaml

   boundary_conditions:
     top:
       type: Dirichlet_boundary_condition
       variable: temperature
       value: "= 500*(1 - 10*x^2)"

Every other string is passed unchanged, e.g. the names of variables, of
boundaries and of the functions defined in the ``functions`` block.

Expressions are compiled by the parser of the library into a short program
that is evaluated in C++.  They are never passed to the ``eval`` function of
Python, so that an input file cannot execute code.  The grammar is given in :doc:`user_guide/problem_setup`.

Uncertainty studies
-------------------

A ``uq`` block turns an input file into an uncertainty or sensitivity study of
a model.  The model is another input file, named by ``model``, or the blocks
of the problem in the same file.  The study runs the model many times with the
uncertain inputs drawn from their distributions, and follows post-processors
of the model as outputs.  The following study propagates the uncertainties of
the conductivity, the heat source and the convection coefficient of the bus
bar to the hottest temperature and to the heat through the left side:

.. literalinclude:: ../examples/bus_bar_uq.yaml
   :language: yaml

The same study in Python is ``examples/bus_bar_uq.py``, and both give the same
runs.  The entries of the block are:

``study``
    ``propagate`` (sampling of the outputs and their statistics), ``sobol``
    (Sobol' sensitivity indices) or ``calibrate`` (Bayesian calibration of
    the inputs against measurements).  These are the functions
    :func:`dualmesh.uq.propagate`, :func:`dualmesh.uq.sobol` and
    :func:`dualmesh.uq.calibrate`, and every other entry of the block is an
    argument of the function.

``model``
    The input file of the model, relative to the file of the study.

``inputs``
    One entry per uncertain input, with ``distribution`` (``normal``,
    ``log_normal``, ``uniform`` or ``log_uniform``) and its parameters
    (``mean`` and ``std``, ``median`` and ``factor`` or ``sigma``, ``lower``
    and ``upper``), the ``parameter`` of the model that it sets, written as a
    path through the blocks of the model input (e.g.
    ``kernels/conduction/thermal_conductivity``), and ``apply``.  With
    ``apply: value`` (the default) the sample replaces the value of the
    parameter, and with ``apply: factor`` it multiplies the value.

``outputs``
    A list of post-processors of the model, whose final values are the
    outputs, or a mapping from post-processor names to ``final`` or
    ``history`` (the whole time history).

``samples``, ``method``, ``seed``
    The number of runs, the design (``latin_hypercube``, ``sobol`` or
    ``monte_carlo``) and the seed of the random numbers.

``processes``, ``store``, ``results``
    The number of parallel processes, a file that stores every run so that an
    interrupted study resumes, and a JSON file that receives the results.

The ``sobol`` study also takes ``surrogate``, ``second_order``,
``training_samples``, ``realizations`` and ``level``, and the ``calibrate``
study takes ``observed``, ``noise``, ``locations``, ``discrepancy``,
``sampler``, ``walkers`` and ``chains``, with the meanings of the arguments of
the Python functions.

Running in parallel
-------------------

The same input file runs on several processes under MPI, when the library was
built with it:

.. code-block:: console

   mpirun -n 4 dualmesh run input.yaml

The mesh is partitioned, each process solves its part with the distributed
solver configured by the ``parallel`` block, a ``vtu`` output becomes one file
per process and a ``.pvtu`` index that ParaView opens as one data set, and the
``field_csv`` and ``mesh_file`` outputs are computed from the gathered solution
and written by the first process.  Post-processors are not yet available in a
distributed run.  The cell-centred method (``zfvm``) runs on one process, with
threads.

Other commands
--------------

.. code-block:: console

   dualmesh list                       # every registered object type
   dualmesh list --category kernel     # only kernels
   dualmesh list --module fluids       # only the fluids module
   dualmesh describe Robin_boundary_condition  # parameters and documentation
   dualmesh --version
