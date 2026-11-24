Adaptive mesh refinement
========================

.. currentmodule:: dualmesh

An adaptive solve repeats a cycle: it solves the problem, estimates the error
of every element, marks the elements with the largest errors, and refines
them.  :func:`solve_with_adaptive_refinement` runs the cycles, and the other
functions carry out its steps one at a time.

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
