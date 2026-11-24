Results, tables and output files
================================

.. currentmodule:: dualmesh

Every solve and every study of a :class:`Problem` returns a result object.
Its ``summary()`` method returns a report as text, its ``tables`` attribute
holds its numbers as :class:`Table` objects, and its ``write_json`` and
``write_csv`` methods write them to JSON (JavaScript Object Notation) and CSV
(comma-separated values) files.  :meth:`Problem.sensitivities` returns a
:class:`SensitivityResult`.  An :class:`Output` group sets the files of the
fields that a study writes.  The result of the input, :class:`results`, is on
:doc:`input`.

.. autosummary::
   :toctree: generated/
   :template: class.rst

   SolveResult
   SensitivityResult
   Table
   Output

Post-processors
---------------

A post-processor computes one number after every solve and every time step,
and the problem keeps the numbers as a history
(:meth:`Problem.add_postprocessor`).  The types are:

.. currentmodule:: dualmesh.postprocessors

.. autosummary::
   :toctree: generated/
   :template: class_alone.rst

   PointValue
   Maximum
   Minimum
   Integral
   Average
   HeatFlow
   ReactionForce
   Force
   FlowRate
   BoundaryFlux
   KernelBoundaryFlux
