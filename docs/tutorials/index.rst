Tutorials
=========

Each tutorial builds one problem from a script in the ``examples`` directory
of the repository, step by step, and shows how to check the result.  The
tutorials run from the simplest problem to the most involved.  Read
:doc:`/quickstart` and :doc:`/concepts` first.

The tutorials on heat conduction, elasticity and transient problems use the
input, :class:`dualmesh.problem`.  The others use the object-level API,
:class:`dualmesh.Problem`, for what the input does not describe: a kernel of
your own, an annular mesh, the penalty formulation of a flow, beams, the side
sets of an imported mesh, and equations written as text.

.. toctree::
   :maxdepth: 1

   heat_conduction
   custom_kernel
   elasticity
   flow
   structures
   transient
   wrench
   criticality
