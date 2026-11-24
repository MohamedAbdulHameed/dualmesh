Parallel execution
==================

.. currentmodule:: dualmesh

These functions report what the build supports and which Message Passing
Interface (MPI) processes a run has, and they partition a mesh among the
processes.  A :class:`Problem` distributes itself among the processes.

.. autosummary::
   :toctree: generated/

   have_gslib
   petsc_version
   num_ranks
   is_root
   parallel.rank
   partition_mesh
   parallel.sub_mesh
