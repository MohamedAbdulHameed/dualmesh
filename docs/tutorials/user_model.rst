A model of your own
===================

In this tutorial, you write the balance law of a bar in large deformation as a
model of your own in the input, :class:`dualmesh.problem`, and you compare
its tip displacement with Example 6.2.4 of [Reddy2024]_.  A model holds the
storage, the flux and the source of one scalar unknown, as formulas with
units, and the input differentiates them for Newton's method.

The problem
-----------

A bar of length :math:`L = 1\ \mathrm{m}`, with the axial position :math:`x`
in :math:`\mathrm{m}`, is fixed at :math:`x = 0` and carries the axial load
:math:`Q` in :math:`\mathrm{N}` at :math:`x = L`.  Its axial displacement
:math:`u` in :math:`\mathrm{m}` satisfies

.. math::

   \frac{\mathrm{d}N}{\mathrm{d}x} = 0 \quad (0 < x < L) ,
   \qquad u(0) = 0 , \qquad N(L) = Q ,

where :math:`N = a(e)\, e` is the axial force in :math:`\mathrm{N}`, with
the dimensionless strain :math:`e = \mathrm{d}u/\mathrm{d}x` and the axial
stiffness

.. math::

   a(e) = E A \left( 1 + 1.5\, e + 0.5\, e^2 \right) .

Here :math:`E` is Young's modulus in :math:`\mathrm{Pa}` and :math:`A` the
cross-sectional area in :math:`\mathrm{m^2}`.  The script takes
:math:`EA = 1\ \mathrm{MN}`.  The stiffness grows with the strain, so the
problem is nonlinear.

A model of your own is the balance law
:math:`\partial m / \partial t + \nabla \cdot \mathbf{F} = s` of its unknown,
with the storage :math:`m`, the flux :math:`\mathbf{F}` and the source
:math:`s`.  The bar has no storage and no source, and its flux is the axial
force, :math:`\mathbf{F} = a(e)\, \nabla u`.  A condition on the flux sets
its outward normal part :math:`\mathbf{F} \cdot \mathbf{n}`, which at
:math:`x = L` is the load :math:`Q`.

Build and solve the bar
-----------------------

The script is ``examples/large_deformation_bar.py``.  Its function ``bar``
takes the load and the number of cells, eight by default.

1. Create the problem on a segment of eight cells:

   .. literalinclude:: ../../examples/large_deformation_bar.py
      :language: python
      :start-at: rod = dualmesh.problem(
      :end-at: rod = dualmesh.problem(
      :dedent: 4

2. Add the model.  The unknown ``axial_displacement`` declares its unit, the
   expression ``axial_strain`` names the strain :math:`e`, the component of
   the gradient along :math:`x`, and ``flux_name`` names the flux so that a
   boundary condition can set it:

   .. literalinclude:: ../../examples/large_deformation_bar.py
      :language: python
      :start-at: rod.model(
      :end-before: rod.properties(
      :dedent: 4

3. Declare the axial stiffness :math:`EA`, a name outside the vocabulary of
   the input, with its unit:

   .. literalinclude:: ../../examples/large_deformation_bar.py
      :language: python
      :start-at: rod.properties(
      :end-at: rod.properties(
      :dedent: 4

4. Fix the left end, and load the right end with the axial force:

   .. literalinclude:: ../../examples/large_deformation_bar.py
      :language: python
      :start-at: rod.boundary_conditions("left"
      :end-at: rod.boundary_conditions("right"
      :dedent: 4

5. Solve for each load, and read the tip displacement, the largest value of
   the field:

   .. literalinclude:: ../../examples/large_deformation_bar.py
      :language: python
      :start-at: for load, published in
      :end-at: tip =
      :dedent: 4

Check the result
----------------

For the loads :math:`Q = 0.2`, :math:`1` and :math:`5\ \mathrm{MN}`, the
script prints the tip displacements 0.1597, 0.5214 and :math:`1.3089\
\mathrm{m}`, beside the values 0.1597, 0.5213 and :math:`1.3087\
\mathrm{m}` that [Reddy2024]_ reports for eight elements.  The book stops its
iterations at a tolerance of :math:`10^{-3}`.  The strain is uniform, so the
exact tip displacement is :math:`u(L) = L\, e`, with :math:`e` the root of

.. math::

   E A \left( e + 1.5\, e^2 + 0.5\, e^3 \right) = Q .

The test ``test_the_large_deformation_bar_has_the_tip_displacements_of_Reddy``
in ``tests/python/test_examples.py`` runs the script and checks each tip
displacement against the book to :math:`10^{-3}\ \mathrm{m}`.

:doc:`/user_guide/input` describes every key of a model.
