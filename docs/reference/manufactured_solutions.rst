Manufactured solutions
======================

.. currentmodule:: dualmesh.mms

The method of manufactured solutions verifies the order of accuracy of a
method.  You choose an exact solution, the code derives the source that makes
it exact, and a solve on a sequence of meshes gives the order of convergence.

.. autosummary::
   :toctree: generated/
   :template: class.rst

   ManufacturedSolution
   ConvergenceResult

:func:`convergence_rates` gives the observed orders of any sequence of
refinements, such as a study of the time step or of a benchmark that you solve
yourself.

.. autosummary::
   :toctree: generated/

   convergence_rates
