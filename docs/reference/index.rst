API reference
=============

This reference describes the public classes, functions and methods of the
``dualmesh`` Python package: the parameters of each, with their types and
defaults, and what each returns.  :doc:`/user_guide/index` explains how they
work together.  The physics, couplings, kernels, boundary conditions and
property objects that you add by type name, as in
``problem.add_physics("heat_conduction", ...)``, list their parameters in
:doc:`/objects`.

.. code-block:: python

   import dualmesh

The examples write a name of the top level as ``dualmesh.<name>`` and a name
of a submodule as ``dualmesh.<module>.<name>``, for example
``dualmesh.mms.ManufacturedSolution``.

.. toctree::
   :maxdepth: 1

   input
   problem
   meshes
   physics
   python_objects
   results
   expressions
   adaptivity
   eigenvalues
   manufactured_solutions
   sections
   parallel
   automatic_differentiation
   registry
   command_line
