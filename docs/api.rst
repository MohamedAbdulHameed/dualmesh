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
.. autofunction:: dualmesh.sideset_summary
.. autofunction:: dualmesh.write_sidesets
.. autofunction:: dualmesh.mesh_from_arrays
.. autofunction:: dualmesh.graded_coordinates
.. autofunction:: dualmesh.annulus_coordinates
.. autofunction:: dualmesh.meshing.coordinates_from_spacings
.. autofunction:: dualmesh.meshing.mesh_summary

Results and expressions
-----------------------

.. autoclass:: dualmesh.SolveResult
   :members:

.. autoclass:: dualmesh.Table
   :members:

.. autoclass:: dualmesh.Output
   :members: output_times

.. autoclass:: dualmesh.ParsedFunction
   :members:

Parallel execution
------------------

.. autofunction:: dualmesh.partition_mesh
.. autofunction:: dualmesh.have_mpi
.. autofunction:: dualmesh.have_metis
.. autofunction:: dualmesh.is_root
.. autofunction:: dualmesh.num_ranks

Adaptive refinement
-------------------

.. autofunction:: dualmesh.solve_with_adaptive_refinement
.. autoclass:: dualmesh.AdaptivityResult
   :members:
.. autofunction:: dualmesh.refine_marked
.. autofunction:: dualmesh.mark_by_fraction
.. autofunction:: dualmesh.mark_by_error_fraction
.. autofunction:: dualmesh.mark_by_threshold

Manufactured solutions
----------------------

.. automodule:: dualmesh.mms
   :members:

Uncertainty quantification
--------------------------

The theory, the sources and the verification are in
:doc:`theory/uncertainty`.

.. automodule:: dualmesh.uq
   :members: Normal, LogNormal, Uniform, LogUniform, Distribution, propagate, Runs, wilks_samples, sobol, SobolIndices, GaussianProcess, calibrate, Posterior

Material properties
-------------------

.. automodule:: dualmesh.materials.water
   :members: specific_volume, density, enthalpy, isobaric_heat_capacity, isochoric_heat_capacity, temperature, saturation_temperature, viscosity, viscosity_from_density, thermal_conductivity, properties

.. automodule:: dualmesh.materials.gas
   :members: thermal_conductivity, temperature_jump_distance

Physics and couplings
---------------------

The parameters of every physics and coupling are listed in the syntax
reference (:doc:`syntax/index`) and by ``dualmesh describe <type>``.

.. automodule:: dualmesh.physics
   :members: Physics, Coupling, register, registered, describe

Parameter declarations
----------------------

.. automodule:: dualmesh.parameters
   :members: parameter, keyword_checked, check_keywords, describe, describe_fields

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
   :members: main
