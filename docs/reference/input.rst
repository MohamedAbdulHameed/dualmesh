The input
=========

.. currentmodule:: dualmesh

A :class:`problem` holds the input of one simulation as a tree of sections:
the mesh, the physics, the properties, the conditions, the time span and the
settings of the solve.  Each method of :class:`problem` sets one section, and
:meth:`problem.from_file` reads the tree from a case file in YAML (YAML Ain't
Markup Language) or TOML (Tom's Obvious Minimal Language).
:meth:`problem.solve` solves the problem and returns a :class:`results`
object.  :doc:`/user_guide/input` describes every section and key.

.. autosummary::
   :toctree: generated/
   :template: class.rst

   problem
   results

.. autosummary::
   :toctree: generated/

   table
