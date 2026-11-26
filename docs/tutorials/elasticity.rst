Plane elasticity
================

In this tutorial, you solve two problems of linear elasticity in plane
stress with the input :class:`dualmesh.problem`: a bar pulled by a traction,
and a plate with a circular hole.  You check the first against its exact
displacement and the second against the stress of a hole in an infinite
plate.

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
a uniform stress :math:`\sigma_\infty` along :math:`x`.  In an infinite
plate, the stress on the :math:`y` axis is the solution of Kirsch
[TimoshenkoGoodier1970]_,

.. math::

   \sigma_{xx}(0, y) = \sigma_\infty \left( 1 + \frac{a^2}{2 y^2}
   + \frac{3 a^4}{2 y^4} \right),

which is :math:`3\sigma_\infty` at the top of the hole, :math:`y = a`: the
stress concentration factor is 3.  The steel plate has :math:`E = 200\
\mathrm{GPa}`, :math:`\nu = 0.3`, :math:`a = 10\ \mathrm{mm}` and
:math:`\sigma_\infty = 100\ \mathrm{MPa}`.  The planes :math:`x = 0` and
:math:`y = 0` are planes of symmetry, so the model is a quarter of the plate,
an annulus of outer radius :math:`200\ \mathrm{mm}`.

The scripts are in ``examples/plate_with_hole``.
``make_plate_mesh.py`` builds the mesh with Gmsh and writes
``plate_with_hole.msh``, which is part of the repository.  It has
:math:`30 \times 30` quadrilaterals, whose radial size grows by the ratio
1.15 from the hole outwards, and four named curves: ``hole``, ``outer``,
``bottom`` (the edge on the :math:`x` axis) and ``left`` (the edge on the
:math:`y` axis).  ``plate_with_hole.py`` solves the problem.

1. Read the mesh file, whose coordinates are in :math:`\mathrm{mm}`, and add
   the solid in plane stress with its two properties:

   .. literalinclude:: ../../examples/plate_with_hole/plate_with_hole.py
      :language: python
      :start-at: quarter = dualmesh.problem(
      :end-at: quarter.properties(
      :dedent: 4

   A mesh file does not state the unit of its coordinates, so the input
   takes it as ``length_unit``.

2. Make the two straight edges planes of symmetry, and load the outer arc
   with the traction of the remote stress,
   :math:`\boldsymbol{\sigma}\mathbf{n} = (\sigma_\infty n_x, 0)` with
   :math:`n_x = x/\sqrt{x^2 + y^2}`:

   .. literalinclude:: ../../examples/plate_with_hole/plate_with_hole.py
      :language: python
      :start-at: quarter.boundary_conditions("left"
      :end-at: quarter.boundary_conditions("outer"
      :dedent: 4

   The word ``symmetry`` holds the displacement normal to the edge at zero,
   and leaves the tangential one free.  Each component of the traction is a
   constant or a formula of the coordinates.  The hole takes no condition,
   so it is free of traction.

3. Ask for the stress at the top of the hole, and for its values along the
   :math:`y` axis, from the hole to :math:`50\ \mathrm{mm}`:

   .. literalinclude:: ../../examples/plate_with_hole/plate_with_hole.py
      :language: python
      :start-at: quarter.postprocessor("peak"
      :end-at: quarter.output(
      :dedent: 4

   The stress at a point is that of the element that holds the point.  The
   line writes ``plate_with_hole_ligament.csv``, with the distance along the
   line, the coordinates and :math:`\sigma_{xx}`, each with its unit.  The
   Python call writes the key ``from`` as ``from_``, since ``from`` is a
   word of Python.

4. Solve, and read the line from the results:

   .. literalinclude:: ../../examples/plate_with_hole/plate_with_hole.py
      :language: python
      :start-at: solution = plate().solve()
      :end-at: ligament = solution.line("ligament")
      :dedent: 4

The script prints the stress concentration factor 3.047, and the stress
along the :math:`y` axis next to the solution of Kirsch.  The test
``test_the_plate_with_a_hole_has_the_stress_of_Kirsch`` in
``tests/python/test_examples.py`` runs the script and checks that the peak is
within 2 % of :math:`3\sigma_\infty`, and that the stress from :math:`y =
2a` outwards is within 1 % of the solution of Kirsch.  The outer radius
:math:`r` is 20 times that of the hole, where the terms of the hole in the
solution of Kirsch scale with :math:`(a/r)^2 = 1/400`, so the finite plate is
close to the infinite one.

An axisymmetric body, such as a thick cylinder under internal pressure, is
modelled on a mesh of its cross-section with ``coordinate_system:
axisymmetric`` in the mesh section.  Its solid then takes the axisymmetric
formulation, and the physics takes no ``formulation``.
