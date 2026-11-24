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
