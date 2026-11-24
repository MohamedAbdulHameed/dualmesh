Adaptive mesh refinement
========================

.. currentmodule:: dualmesh

Solving, estimating the error of every element, marking the elements with
the largest errors and refining them, in cycles.

.. autosummary::
   :toctree: generated/

   solve_with_adaptive_refinement
   mark_by_fraction
   mark_by_error_fraction
   mark_by_threshold
   refine_marked

.. autosummary::
   :toctree: generated/
   :template: class.rst

   AdaptivityResult
