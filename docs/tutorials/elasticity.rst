Plane elasticity
================

The solid mechanics module provides a ``LinearElasticStress`` material and one
``StressDivergence`` kernel per displacement component, in plane stress, plane
strain, axisymmetric, or three-dimensional form.  The helper
:func:`dualmesh.physics.add_plane_elasticity` adds all of them at once.

.. literalinclude:: ../../examples/plate_with_hole.py
   :language: python

Points to note:

* Tractions are prescribed with ``TractionBC`` (a component of the traction
  vector) or ``PressureBC`` (a normal pressure), both of which are natural
  conditions on the equations of the corresponding components.
* Stresses are material properties, evaluated wherever they are asked for:
  ``problem.property_at_centroids("stress")`` returns them at element centres,
  in the Voigt order :math:`(\sigma_{xx}, \sigma_{yy}, \sigma_{zz},
  \sigma_{yz}, \sigma_{xz}, \sigma_{xy})`, and ``write_vtu`` writes them as
  cell data.
* For a thick cylinder under internal pressure the same set-up reproduces
  Table 9.9.1 of the book; for an axisymmetric model use
  ``coordinates="axisymmetric"`` and ``formulation="axisymmetric"``, which adds
  the hoop stress to the radial equation automatically.
