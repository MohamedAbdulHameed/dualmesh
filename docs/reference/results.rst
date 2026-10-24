Results, tables and output files
================================

.. currentmodule:: dualmesh

Every solve and every study returns a result object.  Its ``summary()``
prints it, its ``tables`` hold its numbers as :class:`Table` objects, and it
writes itself to JSON and CSV files.  An :class:`Output` group writes the
fields during a solve.

.. autosummary::
   :toctree: generated/
   :template: class.rst

   SolveResult
   Table
   Output

Post-processors
---------------

A post-processor computes one number after every solve and every time step,
which the problem keeps as a history (:meth:`Problem.add_postprocessor`).
The types are:

.. currentmodule:: dualmesh.postprocessors

.. autosummary::
   :toctree: generated/
   :template: class_alone.rst

   PointValue
   NodalExtremeValue
   VariableIntegral
   VariableAverage
   TotalReaction
   BoundaryFluxIntegral
