Beams and plates
================

The structural members of the solid mechanics module are three beam models
(the mixed Euler-Bernoulli model, and the displacement and mixed Timoshenko
models), two models of an axisymmetric circular plate (the first-order shear
deformation model and the mixed classical model), and the first-order shear
deformation model of a plate with five variables.  The beams and circular
plates take ``Edge2`` elements and the plate ``Quad4`` elements: the one-point
integration of the shear unlocks these, while a linear triangle still locks.  All of them accept functionally graded sections and the von
Kármán nonlinearity.

.. literalinclude:: ../../examples/functionally_graded_beam.py
   :language: python

Points to note:

* The stiffnesses :math:`A_{xx}`, :math:`B_{xx}`, :math:`D_{xx}`, and
  :math:`S_{xz}` of a power-law graded section are computed by
  :func:`dualmesh.fgm.beam_stiffness`, and ``plate=True`` divides them by
  :math:`1-\nu^2` as the plate theories require.
* A mixed model carries the bending moment as an unknown, and the natural
  condition of its moment equation is a zero slope.  So ``solid_mechanics``
  holds the moment at zero on every end that is not clamped (``"fixed"``), is
  not a plane of symmetry (``"symmetry"``) and has no moment of its own.  A
  free end and a simple support need no condition on the moment, and a
  clamped end keeps its zero slope.
* The centre of a half beam is a plane of symmetry:
  ``beam.add_boundary_condition("symmetry", boundary="right")`` holds the axial
  displacement and the slope (``phi`` in the displacement model), and leaves
  the moment free.
* A section given by its stiffnesses takes the stiffnesses that its theory
  uses: ``extensional_stiffness`` and ``bending_stiffness``, and
  ``shear_stiffness`` for the Timoshenko beam and the first-order plates.
  ``coupling_stiffness`` may be left out, because it is zero for a section
  symmetric about its mid-surface.  A plate given by its material needs its
  ``thickness``.
* In the displacement Timoshenko model the shear term must be evaluated at the
  element centre, or the beam locks.  ``solid_mechanics`` with
  ``formulation="beam"`` applies this
  evaluation, and ``tests/python/test_beams.py`` shows the size of the
  difference.
* The plate and circular plate formulations do the same by adding the
  kernel twice per model, once for bending and once for shear with reduced
  integration.
* Geometric nonlinearity is switched on with ``von_karman=True``.  Use load
  steps, and optionally ``nonlinear_solver="picard"`` with ``relaxation=0.35``.
