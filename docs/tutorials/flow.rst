Viscous incompressible flow
===========================

In this tutorial, you solve the flow in a lid-driven square cavity at a
Reynolds number of 1000 with the penalty formulation, and you compare the
velocity on the vertical centreline with the dual mesh control domain method
(DMCDM) solution of Example 9.8.4 of [Reddy2024]_, with the input
:class:`dualmesh.problem`.

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

The cavity is a square of side :math:`a = 1\ \mathrm{m}`, whose lid moves
along :math:`x` with the speed :math:`V = 1\ \mathrm{m/s}`, and whose other
walls are at rest.  The Reynolds number is :math:`Re = \rho V a / \mu`.  The
fluid has the viscosity :math:`\mu = 1\ \mathrm{Pa\,s}` and the density
:math:`\rho = 1000\ \mathrm{kg/m^3}`, so :math:`Re = 1000`.

Build and solve the cavity
--------------------------

The script is ``examples/lid_driven_cavity.py``.

1. Create the problem on a mesh of :math:`16 \times 20` cells, solved by the
   DMCDM:

   .. literalinclude:: ../../examples/lid_driven_cavity.py
      :language: python
      :start-at: flow = dualmesh.problem(
      :end-at: method="DMCDM",
      :dedent: 4

   The ``grading`` of each axis is the ratio of the length of each cell to
   the one before it.  Along :math:`y` the rows shrink by the ratio 0.964,
   so the row under the lid is half as tall as the bottom row, as in the
   mesh of the book.  The book refines the eight rows under the lid by half,
   so its mesh and this one differ.

2. Add the flow in the penalty formulation, with its two properties:

   .. literalinclude:: ../../examples/lid_driven_cavity.py
      :language: python
      :start-at: flow.physics(
      :end-at: flow.properties(
      :dedent: 4

   The penalty parameter keeps its default,
   :math:`\gamma = 10^{8}\ \mathrm{Pa\,s}`, the value of the book.  The
   penalty formulation has no pressure variable, so
   the flow solves for the velocity alone.

3. Hold the velocity on the walls, one value per component:

   .. literalinclude:: ../../examples/lid_driven_cavity.py
      :language: python
      :start-at: for wall in
      :end-at: flow.boundary_conditions("top"
      :dedent: 4

   When Newton's method does not converge from the fluid at rest, the solve
   ramps the lid velocity up from zero in steps, and needs no setting for
   it.

4. Sample the horizontal velocity at 33 points of the vertical centreline,
   one every :math:`1/32\ \mathrm{m}`, and write the velocity to a file of
   the Visualization Toolkit (VTK):

   .. literalinclude:: ../../examples/lid_driven_cavity.py
      :language: python
      :start-at: flow.postprocessor(
      :end-at: flow.output(
      :dedent: 4

   The line writes ``cavity_centreline.csv``, with the distance along the
   line, the coordinates and the velocity, each with its unit.  The Python
   call writes the key ``from`` as ``from_``, since ``from`` is a word of
   Python.

5. Solve, and read the line from the results at the points of the published
   table:

   .. literalinclude:: ../../examples/lid_driven_cavity.py
      :language: python
      :start-at: solution = cavity().solve()
      :end-at: computed = numpy.interp(
      :dedent: 4

Check the result
----------------

The script prints the computed ``velocity_x`` next to the DMCDM column of
Table 9.8.3 of [Reddy2024]_, from :math:`-0.1430\ \mathrm{m/s}` at
:math:`y = 0.0625\ \mathrm{m}` to :math:`0.5245\ \mathrm{m/s}` at
:math:`y = 0.9688\ \mathrm{m}`.  On the graded mesh the two differ by up to
:math:`0.017\ \mathrm{m/s}`, the difference of the two meshes.  The test
``test_example_9_8_4_cavity_Reynolds_1000`` in ``tests/python/test_fluids.py``
builds the cavity on the mesh of the book from its kernels, solves it with
the DMCDM, and checks every value of the table to :math:`2 \times 10^{-3}`.

:doc:`/theory/fluids` gives the formulations of the flow.
