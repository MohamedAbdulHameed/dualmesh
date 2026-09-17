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
    ``method`` (``dmcdm`` or ``fem``) and ``coordinates`` (``cartesian``,
    ``axisymmetric``, ``spherical``).

``variables``
    One entry per unknown; each may carry ``initial_condition`` and ``blocks``.

``functions``
    Named expressions in ``x``, ``y``, ``z``, ``t``, which can then be used
    wherever a coefficient is expected.

``kernels``, ``boundary_conditions``, ``materials``, ``point_sources``
    One entry per object.  Each needs a ``type`` — a registered object name,
    listed by ``dualmesh list`` — and the parameters of that object.

``executioner``
    ``type: steady`` or ``type: transient``, plus the solver options
    (``nonlinear_solver``, ``relaxation``, ``load_factors``,
    ``linear_solver``, ``max_iterations``, and for transients ``end_time``,
    ``dt``, ``theta``).

``outputs``
    ``vtu``, ``csv``, ``mesh_file``, ``cell_properties``, ``reactions`` (pairs
    of variable and boundary), and ``point_values``.

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

Input files are trusted input: expressions are evaluated by Python, so do not
run an input file you have not read.

Other commands
--------------

.. code-block:: console

   dualmesh list                       # every registered object type
   dualmesh list --category Kernel     # only kernels
   dualmesh list --module fluids       # only the fluids module
   dualmesh describe RobinBC           # parameters and documentation
   dualmesh --version
