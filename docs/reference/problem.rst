The problem
===========

.. currentmodule:: dualmesh

A :class:`Problem` holds a mesh, the method that discretizes it, the
variables, and the equations and conditions that act on them.  It solves
them, and it gives access to the solution and the quantities derived from
it.

.. autosummary::
   :toctree: generated/
   :template: class_alone.rst

   Problem

Defining the equations
----------------------

.. autosummary::
   :toctree: generated/

   Problem.add_physics
   Problem.add_coupling
   Problem.add_variable
   Problem.add_kernel
   Problem.add_boundary_condition
   Problem.add_property
   Problem.add_point_source
   Problem.add_function
   Problem.add_postprocessor
   Problem.set_element_field
   Problem.initialize

Solving
-------

.. autosummary::
   :toctree: generated/

   Problem.solve
   Problem.solve_transient
   Problem.solve_eigenvalue
   Problem.set_time_step_callback
   Problem.apply_initial_conditions
   Problem.linear_system
   Problem.set_num_threads
   Problem.effective_threads

The solution
------------

.. autosummary::
   :toctree: generated/

   Problem.values
   Problem.set_values
   Problem.solution
   Problem.sample
   Problem.values_at_nodes
   Problem.gradient_at_centroids
   Problem.property_at_centroids
   Problem.kernel_flux_at_centroids
   Problem.element_field
   Problem.nodal_extreme
   Problem.time

Reactions and integrals
-----------------------

.. autosummary::
   :toctree: generated/

   Problem.reactions
   Problem.total_reaction
   Problem.boundary_flux_integral
   Problem.integrate
   Problem.element_integrals
   Problem.element_volumes
   Problem.domain_volume
   Problem.boundary_measure

Accuracy
--------

.. autosummary::
   :toctree: generated/

   Problem.error_norms
   Problem.error_indicator

Post-processors
---------------

.. autosummary::
   :toctree: generated/

   Problem.postprocessor_values
   Problem.postprocessor_table
   Problem.write_postprocessor_csv

Writing files
-------------

.. autosummary::
   :toctree: generated/

   Problem.write_vtu
   Problem.write_csv
   Problem.write_mesh_file
   Problem.summary

Mesh entities and variables
---------------------------

.. autosummary::
   :toctree: generated/

   Problem.mesh
   Problem.entity_points
   Problem.nodes_where
   Problem.node_at
   Problem.variable_index
   Problem.variable_unit
   Problem.variable_order
   Problem.num_active_dofs
   Problem.thread_safe

Distributed problems
--------------------

.. autosummary::
   :toctree: generated/

   Problem.is_distributed
   Problem.rank
   Problem.num_ranks
   Problem.num_owned_dofs
   Problem.num_global_dofs
   Problem.gathered_values
   Problem.partition_summary
