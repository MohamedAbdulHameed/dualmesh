Viscous incompressible flow
===========================

The fluids module uses the penalty formulation: the pressure is replaced by
:math:`P = -\gamma\,\nabla\cdot\mathbf{v}`, and the penalty term is integrated
with a reduced rule, which is what keeps the velocity field from locking.

.. literalinclude:: ../../examples/lid_driven_cavity.py
   :language: python

Points to note:

* :func:`dualmesh.physics.add_incompressible_flow` adds the viscous, penalty,
  and (optionally) convective kernels, and a material that recovers the
  pressure.  With ``density=0`` the equations are the Stokes equations.
* At :math:`Re = 1000` Newton's method started from rest does not converge.
  Ramping the lid velocity with ``scale_with_load=True`` and a few load steps
  does converge, and so does direct iteration with ``relaxation=0.5``.
* The recovered pressure is most accurate at the element centres, which is
  where ``property_at_centroids("pressure")`` evaluates it.
* :doc:`../openfoam` compares this solver with OpenFOAM on the same problem.
