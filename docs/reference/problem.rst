The problem
===========

.. currentmodule:: dualmesh

A :class:`Problem` holds a mesh, the method that discretizes it, the
variables, and the equations and conditions that act on them.  It solves the
equations and gives you the solution and the quantities derived from it.
:class:`Problem` is the object-level application programming interface (API),
and :class:`problem` of
:doc:`input` builds a :class:`Problem` from the input sections.

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
   Problem.solve_buckling
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
   Problem.sample_gradient
   Problem.gradient_at_centroids
   Problem.property_at_centroids
   Problem.property_at_points
   Problem.kernel_flux_at_centroids
   Problem.element_field
   Problem.maximum
   Problem.minimum
   Problem.time

Flows through boundaries and integrals
--------------------------------------

Each physics also gives the flows through a boundary by their physical
names: ``heat_flow`` of :class:`~dualmesh.physics.HeatConduction`,
``reaction_force`` of :class:`~dualmesh.physics.SolidMechanics`, ``force``
and ``flow_rate`` of :class:`~dualmesh.physics.FluidDynamics`, and
``boundary_flux`` of :class:`~dualmesh.physics.GeneralFormPDE`.

.. autosummary::
   :toctree: generated/

   Problem.boundary_flux
   Problem.kernel_boundary_flux
   Problem.boundary_integral
   Problem.integrate
   Problem.element_integrals
   Problem.element_volumes
   Problem.node_measures
   Problem.block_volume
   Problem.domain_volume
   Problem.boundary_area

Parameters and sensitivities
----------------------------

A parameter is a named constant that coefficients, sources and boundary
values read.  :meth:`Problem.sensitivities` gives the derivative of an
integral of the solution with respect to the parameters, by one adjoint
solve.

.. autosummary::
   :toctree: generated/

   Problem.add_parameter
   Problem.set_parameter
   Problem.sensitivities

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
   Problem.summary

Mesh entities and variables
---------------------------

.. autosummary::
   :toctree: generated/

   Problem.mesh
   Problem.method
   Problem.node_coordinates
   Problem.nodes_where
   Problem.node_at
   Problem.variable_index
   Problem.variable_unit
   Problem.variable_order
   Problem.num_active_dofs
   Problem.thread_safe

Distributed problems
--------------------

A distributed problem runs on several Message Passing Interface (MPI)
processes, and each process owns a part of the mesh.  The ``gathered``
methods work on the whole mesh, on every process.

.. autosummary::
   :toctree: generated/

   Problem.is_distributed
   Problem.rank
   Problem.num_ranks
   Problem.num_owned_dofs
   Problem.num_global_dofs
   Problem.partition_summary
   Problem.gathered_values
   Problem.set_gathered_values
   Problem.gathered_node_coordinates
   Problem.gathered_node_measures
   Problem.gathered_error_indicator
   Problem.gathered_velocities
   Problem.set_gathered_velocities
