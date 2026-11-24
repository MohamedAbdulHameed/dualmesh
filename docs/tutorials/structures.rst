Beams and plates
================

In this tutorial, you bend a functionally graded beam with the three beam
models of the ``solid_mechanics`` physics, and you compare the deflection
with Table 7.5.3 of [Reddy2024]_.  The input, :class:`dualmesh.problem`, has
no structural formulations, so this tutorial uses the object-level API,
:class:`dualmesh.Problem`.

The beam
--------

A beam of length :math:`L = 100\ \mathrm{in}` with a square section of width
:math:`b = 1\ \mathrm{in}` and height :math:`h_b = 1\ \mathrm{in}` is pinned
at both ends and carries the uniform load :math:`q_0 = 0.5\
\mathrm{lb/in}`.  Its material is graded through the height: Young's modulus
follows the power law

.. math::

   E(z) = (E_1 - E_2)\left(\frac{1}{2} + \frac{z}{h_b}\right)^{n} + E_2,
   \qquad -\frac{h_b}{2} \le z \le \frac{h_b}{2},

where :math:`z` is the position through the height, :math:`E_1 = 30 \times
10^6\ \mathrm{psi}` the modulus at the top, :math:`E_2 = 3 \times 10^6\
\mathrm{psi}` the modulus at the bottom and :math:`n` the power-law index.
The index :math:`n = 0` gives a homogeneous beam with :math:`E = E_1`.
Poisson's ratio is :math:`\nu = 0.3`.

A beam model reads the section through its stiffnesses

.. math::

   (A_{xx}, B_{xx}, D_{xx}) = b \int_{-h_b/2}^{h_b/2} E(z)\,(1, z, z^2)\,\mathrm{d}z,
   \qquad
   S_{xz} = \frac{K_s}{2(1 + \nu)}\, A_{xx},

where :math:`A_{xx}` is the extensional stiffness, :math:`B_{xx}` the
coupling of extension and bending, :math:`D_{xx}` the bending stiffness,
:math:`S_{xz}` the shear stiffness and :math:`K_s = 5/6` the shear
correction factor of a rectangle.  The grading makes :math:`B_{xx}` nonzero,
so the load stretches the beam as it bends it.

The models are the Euler-Bernoulli beam and the Timoshenko beam, which adds
the shear deformation.  The Euler-Bernoulli beam and the mixed form of the
Timoshenko beam carry the bending moment as a variable.  The displacement
form of the Timoshenko beam carries the rotation :math:`\phi` of the section,
and evaluates its shear term at the element centre, which keeps a thin beam
from locking.  :doc:`/theory/solid_mechanics` gives the equations.

Build and solve the beam
------------------------

The script is ``examples/functionally_graded_beam.py``.  It models the left
half of the beam, :math:`0 \le x \le L/2`, in inches and pounds.

1. Name the three models by their theory and, for the Timoshenko beam, their
   form.  ``None`` takes the mixed form:

   .. literalinclude:: ../../examples/functionally_graded_beam.py
      :language: python
      :start-at: models = {
      :end-at: }

2. Compute the stiffnesses of the graded section:

   .. literalinclude:: ../../examples/functionally_graded_beam.py
      :language: python
      :start-at: stiffness = dualmesh.fgm.beam_stiffness(
      :end-before: mesh = dualmesh.generate_line_mesh(
      :dedent: 4

   :func:`dualmesh.fgm.beam_stiffness` returns :math:`A_{xx}`,
   :math:`B_{xx}`, :math:`D_{xx}` and :math:`S_{xz}` as ``extensional``,
   ``coupling``, ``bending`` and ``shear``.  With ``plate=True`` it divides
   :math:`A_{xx}`, :math:`B_{xx}` and :math:`D_{xx}` by :math:`1 - \nu^2`, as
   the plate theories need.

3. Generate a line mesh of the half beam, with 16 elements by default:

   .. literalinclude:: ../../examples/functionally_graded_beam.py
      :language: python
      :start-at: mesh = dualmesh.generate_line_mesh(
      :end-at: problem = dualmesh.Problem(mesh)
      :dedent: 4

4. Add the beam, given by its section stiffnesses and its load:

   .. literalinclude:: ../../examples/functionally_graded_beam.py
      :language: python
      :start-at: theory, form = models[model]
      :end-before: # Pinned at the left end
      :dedent: 4

   The physics creates the axial displacement ``disp_x``, the deflection
   ``disp_z`` and the bending moment ``bending_moment``, or the rotation
   ``phi`` in the displacement form.

5. Add the supports, and solve:

   .. literalinclude:: ../../examples/functionally_graded_beam.py
      :language: python
      :start-at: # Pinned at the left end
      :end-at: problem.solve(report="none")
      :dedent: 4

   The left end holds both displacements.  A mixed model holds the bending
   moment at zero on every end that is not clamped (``fixed``), is not a
   plane of symmetry (``symmetry``) and has no moment condition of its own,
   so the pinned end needs no condition on the moment.  The right end is the
   centre of the beam: ``symmetry`` holds the axial displacement and the
   slope, and leaves the moment free.

6. Normalize the deflection at the centre, the last node of the mesh, by the
   reduced bending stiffness :math:`\hat{D} = D_{xx} - B_{xx}^2/A_{xx}`:

   .. literalinclude:: ../../examples/functionally_graded_beam.py
      :language: python
      :start-at: reduced = stiffness.bending
      :end-at: row.append(
      :dedent: 12

Check the result
----------------

The script prints :math:`\bar{w} = 10\, w(L/2)\, \hat{D} / (q_0 L^4)`, where
:math:`w(L/2)` is the deflection at the centre, for each model and each
index.  For :math:`n = 0`, the section is homogeneous, :math:`B_{xx} = 0`,
and Euler-Bernoulli theory gives :math:`w(L/2) = 5 q_0 L^4 / (384 D_{xx})`,
so :math:`\bar{w} = 50/384 = 0.1302`.  Table 7.5.3 of [Reddy2024]_ gives the
values of :math:`\bar{w}` by the dual mesh control domain method (DMCDM) with
16 elements in the half beam:

.. list-table::
   :header-rows: 1

   * - :math:`n`
     - Euler-Bernoulli, mixed
     - Timoshenko, displacement
     - Timoshenko, mixed
   * - 0
     - 0.1302
     - 0.1300
     - 0.1302
   * - 1
     - 0.1069
     - 0.1068
     - 0.1069
   * - 2
     - 0.0919
     - 0.0918
     - 0.0919
   * - 3
     - 0.0879
     - 0.0878
     - 0.0879
   * - 5
     - 0.0900
     - 0.0899
     - 0.0900
   * - 10
     - 0.1012
     - 0.1011
     - 0.1012
   * - 20
     - 0.1141
     - 0.1140
     - 0.1142

The script solves with the finite element method (FEM), which a solid takes
when the problem names no method.  To solve with the DMCDM, as the book does, create the problem with
``dualmesh.Problem(mesh, method="DMCDM")``.  The test
``test_table_7_5_3_functionally_graded_pinned_beam`` in
``tests/python/test_beams.py`` builds the same half beam from its kernels,
solves it with the DMCDM, and checks every value of the table to
:math:`1.1 \times 10^{-4}`.

Other structural members
------------------------

The ``formulation`` of ``solid_mechanics`` also names the plate (the
first-order theory on a mesh of the mid-plane, with the variables
``disp_x``, ``disp_y``, ``disp_z``, ``phi_x`` and ``phi_y``) and the
circular plate (the first-order or the classical theory on a radial mesh in
axisymmetric coordinates).  A plate given by its material needs its
``thickness``, and a beam given by its material needs its ``cross_section``.
``von_karman=True`` adds the nonlinearity of moderately large deflections to
any structural formulation.
