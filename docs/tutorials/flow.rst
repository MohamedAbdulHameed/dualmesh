Viscous incompressible flow
===========================

The ``incompressible_flow`` physics solves the equations of an incompressible
Newtonian fluid.  This tutorial uses its default, the penalty formulation: the
pressure is replaced by :math:`P = -\gamma\,\nabla\cdot\mathbf{v}`, and the
penalty term is integrated with a reduced rule, which prevents locking of the
velocity field.  The ``pressure`` and ``Taylor_Hood`` formulations keep the
pressure as an unknown (see :doc:`../theory/heat_and_fluids`).

.. literalinclude:: ../../examples/lid_driven_cavity.py
   :language: python

Points to note:

* The physics adds the viscous, penalty and (with a density) convective
  kernels, and a property object that recovers the pressure.  With ``density=0`` the
  equations are the Stokes equations.
* The velocity conditions take one value per component, e.g.,
  ``value=[1.0, 0.0]`` on the lid.
* At :math:`Re = 1000` Newton's method started from rest does not converge.
  Ramping the lid velocity with ``scale_with_load=True`` and a few load steps
  does converge, and so does direct iteration with ``relaxation=0.5``.
* The recovered pressure is most accurate at the element centres, which is
  where ``property_at_centroids("pressure")`` evaluates it.
* :doc:`../openfoam` compares this solver with OpenFOAM on the same problem.
