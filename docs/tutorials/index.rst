Tutorials
=========

Each tutorial builds one problem from a script in the ``examples`` directory
of the repository, step by step, and shows how to check the result.  The
tutorials run from the simplest problem to the most involved.  Read
:doc:`/quickstart` and :doc:`/concepts` first.

Every tutorial uses the input, :class:`dualmesh.problem`.  The tutorial on
criticality also writes the equations of a core with regions with the
object-level API, :class:`dualmesh.Problem`.

.. toctree::
   :maxdepth: 1

   heat_conduction
   user_model
   elasticity
   flow
   transient
   wrench
   criticality
