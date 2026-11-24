API reference
=============

This reference describes every public class, function and method of the
``dualmesh`` Python package: the parameters each one takes, with their types
and defaults, what it returns, and the inputs it refuses.  The user guide
explains how the pieces fit together.  The physics, couplings, kernels,
boundary conditions and property objects that are added by their type name
(``problem.add_physics("heat_transfer", ...)``) are listed, with every
parameter, in the :doc:`/objects`.

.. code-block:: python

   import dualmesh

The functions and classes of the top level are written ``dualmesh.<name>`` in the
examples, and those of a submodule ``dualmesh.<module>.<name>``, for example
``dualmesh.mms.ManufacturedSolution``.

.. toctree::
   :maxdepth: 1

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
   postprocessing
   parallel
   automatic_differentiation
   registry
   command_line
