Plane elasticity
================

The ``solid_mechanics`` physics solves the equilibrium equations of a linear
elastic solid in plane stress, plane strain, axisymmetric or three-dimensional
form.  It creates the displacement variables, a ``small_strain_stress``
property object and one ``stress_divergence`` kernel per displacement component.  Its
boundary conditions take one value per component, e.g., a traction vector.

.. literalinclude:: ../../examples/plate_with_hole.py
   :language: python

Points to note:

* Tractions are prescribed with ``traction`` (the traction
  vector, one entry per component, each a constant or a function) or
  ``pressure`` (a normal pressure).  Both are natural
  conditions of the equilibrium equations.
* ``symmetry`` holds the displacement normal to a plane of
  symmetry at zero and leaves the tangential displacements free.
* Stresses are properties, evaluated wherever they are asked for:
  ``problem.property_at_centroids("stress")`` returns them at element centres,
  in the Voigt order :math:`(\sigma_{xx}, \sigma_{yy}, \sigma_{zz},
  \sigma_{yz}, \sigma_{xz}, \sigma_{xy})`, and ``write_vtu`` writes them as
  cell data.
* A thick cylinder under internal pressure takes the same set-up.  For an
  axisymmetric model use
  ``dm.Problem(mesh, coordinates="axisymmetric")``.  The physics then takes the
  axisymmetric formulation, which adds the hoop stress to the radial
  equation.
