Beams and plates
================

The structural members of the solid mechanics module are three beam models
(the mixed Euler-Bernoulli model, and the displacement and mixed Timoshenko
models), the first-order shear deformation model of an axisymmetric circular
plate, and the first-order shear deformation model of a rectangular plate with
five variables.  All of them accept functionally graded sections and the von
Kármán nonlinearity.

.. literalinclude:: ../../examples/functionally_graded_beam.py
   :language: python

Points to note:

* The stiffnesses :math:`A_{xx}`, :math:`B_{xx}`, :math:`D_{xx}`, and
  :math:`S_{xz}` of a power-law graded section are computed by
  :func:`dualmesh.fgm.beam_stiffness`; ``plate=True`` divides them by
  :math:`1-\nu^2` as the plate theories require.
* A mixed model carries the bending moment as an unknown, so a simple support
  is ``M = 0`` (a Dirichlet condition on the moment) and a clamped end is
  ``dw/dx = 0``, which is a *natural* condition there and needs nothing at all.
* In the displacement Timoshenko model the shear term must be evaluated at the
  element centre, or the beam locks.  :func:`dualmesh.physics.add_beam` does
  that for you; ``tests/python/test_beams.py`` shows how large the difference
  is.
* For the plate models the same is done by adding the kernel twice per
  variable, once for bending and once for shear with reduced integration,
  which is what :func:`dualmesh.physics.add_plate` and
  :func:`dualmesh.physics.add_circular_plate` do.
* Geometric nonlinearity is switched on with ``von_karman=True``; use load
  steps, and optionally ``nonlinear_solver="picard"`` with ``relaxation=0.35``
  as in Section 7.6 of the book.
