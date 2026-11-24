Plane elasticity
================

In this tutorial, you solve two problems of linear elasticity in plane
stress: a bar pulled by a traction, with the input
:class:`dualmesh.problem`, and a plate with a circular hole, with the
object-level API :class:`dualmesh.Problem`.  You check the first against its
exact displacement and the second against the stress concentration factor of
a hole in an infinite plate.

The equations
-------------

A solid at rest, without body forces, balances its stresses:

.. math::

   \nabla \cdot \boldsymbol{\sigma} = \mathbf{0} \quad \text{in } \Omega,
   \qquad
   \boldsymbol{\varepsilon} = \tfrac{1}{2}\left(\nabla\mathbf{u} + \nabla\mathbf{u}^{T}\right),

where :math:`\boldsymbol{\sigma}` is the Cauchy stress in :math:`\mathrm{Pa}`,
:math:`\boldsymbol{\varepsilon}` the small strain, :math:`\mathbf{u}` the
displacement in :math:`\mathrm{m}` and :math:`\Omega` the body.  Hooke's law
relates the stress to the strain through Young's modulus :math:`E` in
:math:`\mathrm{Pa}` and Poisson's ratio :math:`\nu`.  In plane stress, the
stresses out of the plane are zero, which suits a thin plate loaded in its
plane.  Each part of the boundary takes a displacement or a traction
:math:`\boldsymbol{\sigma}\mathbf{n}` in :math:`\mathrm{Pa}`, where
:math:`\mathbf{n}` is the outward unit normal.  A boundary without a
condition is free of traction.

A bar pulled by a traction
--------------------------

A bar of :math:`2 \times 0.5\ \mathrm{m}` of steel, with :math:`E = 200\
\mathrm{GPa}` and :math:`\nu = 0.3`, is held on rollers on its left and
bottom sides and pulled on its right side by a traction of :math:`\sigma_0 =
100\ \mathrm{MPa}`.  Its stress is uniform, :math:`\sigma_{xx} = \sigma_0`,
and its displacement is

.. math::

   u_x = \frac{\sigma_0\, x}{E}, \qquad u_y = -\frac{\nu\, \sigma_0\, y}{E},

where :math:`x` and :math:`y` are the coordinates in :math:`\mathrm{m}`.
The code is that of the tests ``test_a_bar_pulled_by_a_traction_stretches_by_sigma_L_over_E``
and ``test_the_stress_at_a_point_of_a_pulled_bar_is_the_applied_traction`` in
``tests/python/test_front_end.py``.

1. Create the problem on a rectangle of :math:`8 \times 2` cells, with the
   solid in plane stress and its two properties:

   .. code-block:: python

      import dualmesh

      bar = dualmesh.problem(mesh={"rectangle": {"size": ["2 m", "0.5 m"], "cells": [8, 2]}})
      bar.physics("solid_mechanics", formulation="plane_stress")
      bar.properties(Young_modulus="200 GPa", Poisson_ratio=0.3)

   A solid in a plane takes ``plane_stress`` or ``plane_strain``.  A solid
   takes the finite element method (FEM) when the problem names no method.

2. Hold each roller in its normal direction, and give the traction as one
   value per component:

   .. code-block:: python

      bar.boundary_conditions("left", displacement_x="0 m")
      bar.boundary_conditions("bottom", displacement_y="0 m")
      bar.boundary_conditions("right", traction=["100 MPa", "0 Pa"])

   The top takes no condition, so it is free of traction.

3. Ask for the stress at two points, and solve:

   .. code-block:: python

      bar.postprocessor("axial", point_value="stress_xx", point=["0.7 m", "0.3 m"])
      bar.postprocessor("lateral", point_value="stress_yy", point=["1.3 m", "0.1 m"])
      solution = bar.solve(report="none")

4. Read the displacement at the nodes:

   .. code-block:: python

      x, y = solution.coordinates("displacement").T
      displacement = solution.field("displacement")

The tests check that ``displacement[:, 0]`` and ``displacement[:, 1]`` equal
the exact displacement to a relative :math:`10^{-9}`, and that the
post-processor ``axial`` is :math:`10^{8}\ \mathrm{Pa}` with the unit
``Pa``.

A plate with a hole
-------------------

A large plate with a hole of radius :math:`a` is pulled far from the hole by
a uniform stress :math:`\sigma_\infty` along :math:`x`.  At the top of the
hole, :math:`\sigma_{xx} = 3\sigma_\infty` in an infinite plate
[TimoshenkoGoodier1970]_: the stress concentration factor is 3.  The model is
a quarter of the plate, an annulus of inner radius :math:`a = 1` and outer
radius 20, with :math:`E = 1` and :math:`\sigma_\infty = 1` in a consistent
set of units.  The input generates segments, rectangles and boxes only, so
this problem uses the object-level API.

The script is ``examples/plate_with_hole.py``.

1. Generate a mesh of the quarter annulus, graded towards the hole:

   .. literalinclude:: ../../examples/plate_with_hole.py
      :language: python
      :start-at: mesh = dualmesh.generate_annulus_mesh(
      :end-at: problem = dualmesh.Problem(mesh)
      :dedent: 4

   The annulus spans :math:`0^\circ` to :math:`90^\circ`, and its side sets
   are ``inner`` (the hole), ``outer``, ``start`` (the edge along
   :math:`x`) and ``end`` (the edge along :math:`y`).

2. Add the solid in plane stress:

   .. literalinclude:: ../../examples/plate_with_hole.py
      :language: python
      :start-at: solid = problem.add_physics(
      :end-before: # Symmetry on the two straight edges.
      :dedent: 4

3. Add the conditions.  The two straight edges are planes of symmetry, where
   the word ``symmetry`` holds the normal displacement at zero and leaves the
   tangential one free.  On the outer arc the traction of the remote stress
   is :math:`\boldsymbol{\sigma}\mathbf{n} = (\sigma_\infty n_x, 0)`, with
   :math:`n_x = x/\sqrt{x^2 + y^2}`:

   .. literalinclude:: ../../examples/plate_with_hole.py
      :language: python
      :start-at: solid.add_boundary_condition("symmetry", boundary="start")
      :end-at: problem.solve()
      :dedent: 4

   A traction written as text takes one expression of :math:`x`,
   :math:`y`, :math:`z` and :math:`t` per component.  The hole takes no
   condition, so it is free of traction.

4. Read the stress at the element centroids, and write it to a file of the
   Visualization Toolkit (VTK):

   .. literalinclude:: ../../examples/plate_with_hole.py
      :language: python
      :start-at: stress = problem.property_at_centroids("stress")
      :end-at: problem.write_vtu("plate_with_hole.vtu", cell_properties=["stress"])
      :dedent: 4

   :meth:`~dualmesh.Problem.property_at_centroids` returns six stress
   components per element, in the order :math:`\sigma_{xx}`,
   :math:`\sigma_{yy}`, :math:`\sigma_{zz}`, :math:`\sigma_{yz}`,
   :math:`\sigma_{xz}`, :math:`\sigma_{xy}`.  ``cell_properties`` writes them
   as cell data, which ParaView shows one value per element.

The script prints the largest :math:`\sigma_{xx}` at the centroids of the
elements next to the hole, 2.868.  The centroid of an element of the first
ring lies off the edge of the hole, where :math:`\sigma_{xx}` is below its
peak value of 3.

An axisymmetric body, such as a thick cylinder under internal pressure, is
modelled on a mesh of its cross-section with
``dualmesh.Problem(mesh, coordinates="axisymmetric")``.  Its solid then takes
the axisymmetric formulation, and ``add_physics`` takes no ``formulation``.
