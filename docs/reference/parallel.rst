Parallel execution
==================

.. currentmodule:: dualmesh

The capabilities of the build and the processes of a run.  A problem is
distributed among the MPI processes by :class:`Problem` itself.

.. autosummary::
   :toctree: generated/

   have_mpi
   have_petsc
   have_gslib
   petsc_version
   num_ranks
   is_root
   parallel.rank
   partition_mesh
   parallel.sub_mesh
