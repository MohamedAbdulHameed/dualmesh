Python API
==========

.. currentmodule:: dualmesh

The problem
-----------

.. autoclass:: dualmesh.Problem
   :members:
   :undoc-members:

Meshes
------

.. autoclass:: dualmesh.Mesh
   :members:

.. autofunction:: dualmesh.generate_line_mesh
.. autofunction:: dualmesh.generate_rectangle_mesh
.. autofunction:: dualmesh.generate_box_mesh
.. autofunction:: dualmesh.generate_annulus_mesh
.. autofunction:: dualmesh.read_mesh
.. autofunction:: dualmesh.write_mesh
.. autofunction:: dualmesh.mesh_from_arrays
.. autofunction:: dualmesh.graded_coordinates
.. autofunction:: dualmesh.annulus_coordinates
.. autofunction:: dualmesh.meshing.coordinates_from_spacings
.. autofunction:: dualmesh.meshing.mesh_summary

Results and expressions
-----------------------

.. autoclass:: dualmesh.SolveResult
   :members:

.. autoclass:: dualmesh.ParsedFunction
   :members:

Parallel execution
------------------

.. autoclass:: dualmesh.DistributedProblem
   :members:

.. autofunction:: dualmesh.partition_mesh
.. autofunction:: dualmesh.have_mpi
.. autofunction:: dualmesh.have_metis
.. autofunction:: dualmesh.is_root
.. autofunction:: dualmesh.num_ranks

Adaptive refinement
-------------------

.. autofunction:: dualmesh.solve_with_adaptive_refinement
.. autofunction:: dualmesh.refine_marked
.. autofunction:: dualmesh.mark_by_fraction
.. autofunction:: dualmesh.mark_by_error_fraction
.. autofunction:: dualmesh.mark_by_threshold

Manufactured solutions
----------------------

.. automodule:: dualmesh.mms
   :members:

Physics helpers
---------------

.. automodule:: dualmesh.physics
   :members:

Objects written in Python
-------------------------

.. automodule:: dualmesh.objects
   :members:
   :show-inheritance:

Functionally graded sections
----------------------------

.. automodule:: dualmesh.fgm
   :members:

Expressions
-----------

.. automodule:: dualmesh.expressions
   :members: parsed_function

Post-processing
---------------

.. automodule:: dualmesh.postprocess
   :members:

Automatic differentiation
-------------------------

.. automodule:: dualmesh.ad
   :members:

Command line
------------

.. automodule:: dualmesh.cli
   :members: main, run, build_problem, build_mesh
