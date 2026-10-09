Plane elasticity
================

The ``solid_mechanics`` physics solves the equilibrium equations of a linear
elastic solid in plane stress, plane strain, axisymmetric or three-dimensional
form.  It creates the displacement variables, a ``linear_elastic_stress``
property object and one ``stress_divergence`` kernel per displacement component.  Its
boundary conditions take one value per component, e.g., a traction vector.

.. literalinclude:: ../../examples/plate_with_hole.py
   :language: python

Points to note:

* Tractions are prescribed with ``traction_boundary_condition`` (the traction
  vector, one entry per component, each a constant or a function) or
  ``pressure_boundary_condition`` (a normal pressure).  Both are natural
  conditions of the equilibrium equations.
* ``symmetry_boundary_condition`` holds the displacement normal to a plane of
  symmetry at zero and leaves the tangential displacements free.
* Stresses are properties, evaluated wherever they are asked for:
  ``problem.property_at_centroids("stress")`` returns them at element centres,
  in the Voigt order :math:`(\sigma_{xx}, \sigma_{yy}, \sigma_{zz},
  \sigma_{yz}, \sigma_{xz}, \sigma_{xy})`, and ``write_vtu`` writes them as
  cell data.
* For a thick cylinder under internal pressure the same set-up reproduces
  Table 9.9.1 of the book.  For an axisymmetric model use
  ``dm.Problem(mesh, coordinates="axisymmetric")``.  The physics then takes the
  axisymmetric formulation, which adds the hoop stress to the radial
  equation.
