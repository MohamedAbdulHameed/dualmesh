Viscous incompressible flow
===========================

In this tutorial, you solve the flow in a lid-driven square cavity at a
Reynolds number of 1000 with the penalty formulation, and you compare the
velocity on the vertical centreline with the dual mesh control domain method
(DMCDM) solution of Example 9.8.4 of [Reddy2024]_.  The input,
:class:`dualmesh.problem`, offers the ``equal_order`` and ``Taylor-Hood``
formulations on uniform meshes, so this tutorial uses the object-level API,
:class:`dualmesh.Problem`.

The equations
-------------

A steady incompressible flow of a Newtonian fluid balances momentum and
conserves volume:

.. math::

   \rho\,(\mathbf{v}\cdot\nabla)\mathbf{v}
   = -\nabla p + \nabla\cdot\left[\mu\left(\nabla\mathbf{v} + \nabla\mathbf{v}^{T}\right)\right],
   \qquad \nabla\cdot\mathbf{v} = 0,

where :math:`\mathbf{v}` is the velocity in :math:`\mathrm{m/s}`,
:math:`p` the pressure in :math:`\mathrm{Pa}`, :math:`\rho` the density in
:math:`\mathrm{kg/m^3}` and :math:`\mu` the dynamic viscosity in
:math:`\mathrm{Pa\,s}`.  The penalty formulation replaces the pressure by

.. math::

   p = -\gamma\, \nabla\cdot\mathbf{v},

where :math:`\gamma` is the penalty parameter in :math:`\mathrm{Pa\,s}`.  A
large :math:`\gamma` keeps the divergence of the velocity small, and the
velocity is then the only unknown.

The cavity is a square of side :math:`a`, whose lid moves along :math:`x`
with the speed :math:`V`, and whose other walls are at rest.  The Reynolds
number is :math:`Re = \rho V a / \mu`.  With :math:`V = a = \mu = 1` in a
consistent set of units, the density equals :math:`Re`.

Build and solve the cavity
--------------------------

The script is ``examples/lid_driven_cavity.py``.

1. Generate the mesh of :math:`16 \times 20` elements of the book, with
   elements of half the height in the eight rows below the lid:

   .. literalinclude:: ../../examples/lid_driven_cavity.py
      :language: python
      :start-at: mesh = dualmesh.generate_rectangle_mesh(
      :end-at: problem = dualmesh.Problem(mesh)
      :dedent: 4

   A problem that names no method takes the DMCDM for a flow.

2. Add the flow in the penalty formulation, with :math:`\gamma = 10^{8}`, the
   value of the book:

   .. code-block:: python

      flow = problem.add_physics(
          "fluid_dynamics",
          dynamic_viscosity=1.0,
          density=Reynolds_number,
          formulation="penalty",
          penalty_parameter=1.0e8,
      )

   The physics creates the variables ``velocity_x`` and ``velocity_y``.
   With ``density=0`` the convective term drops out, and the equations are
   those of Stokes flow.

3. Hold the velocity on the walls, one condition per component:

   .. literalinclude:: ../../examples/lid_driven_cavity.py
      :language: python
      :start-at: flow.add_boundary_condition(
      :end-at: problem.solve(max_iterations=60)
      :dedent: 4

   When Newton's method does not converge from the fluid at rest, the solve
   ramps the lid velocity up from zero in steps, and needs no setting for
   it.  The solve also takes direct iteration with a relaxation, for example
   ``problem.solve(nonlinear_solver="Picard", relaxation=0.5)``.

4. Sample the horizontal velocity on the vertical centreline at the points of
   the published table, and write the velocity and the pressure to a file of
   the Visualization Toolkit (VTK):

   .. literalinclude:: ../../examples/lid_driven_cavity.py
      :language: python
      :start-at: points = numpy.column_stack(
      :end-at: problem.write_vtu("cavity.vtu", cell_properties=["pressure"])
      :dedent: 4

   The penalty formulation has no pressure variable.  The property
   ``pressure`` recovers :math:`p = -\gamma\,\nabla\cdot\mathbf{v}` in each
   element, and ``cell_properties`` writes it as cell data.

Check the result
----------------

The script prints the computed ``velocity_x`` next to the DMCDM column of
Table 9.8.3 of [Reddy2024]_, from :math:`-0.1430` at :math:`y = 0.0625` to
:math:`0.5245` at :math:`y = 0.9688`.  The test
``test_example_9_8_4_cavity_Reynolds_1000`` in ``tests/python/test_fluids.py``
builds the same cavity from its kernels, solves it with the DMCDM, and checks
every value of the table to :math:`2 \times 10^{-3}`.  The test
``test_example_9_8_4_cavity_Reynolds_1000_by_direct_iteration`` checks the
same values with direct iteration.

:doc:`/theory/fluids` gives the formulations of the flow.
