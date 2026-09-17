Verification
============

Every number this library produces for a published problem is checked against
the published value.  The suite in ``tests/python`` reproduces the dual mesh
control domain results of Reddy's book chapter by chapter; ``tests/cpp`` holds
the structural tests that do not depend on a published table; and
:doc:`openfoam` compares the solvers with an independent code.

.. code-block:: console

   pytest                                       # the verification suite
   ctest --test-dir build --output-on-failure   # the C++ unit tests

What is covered
---------------

.. list-table::
   :header-rows: 1
   :widths: 22 48 30

   * - Source
     - Problem
     - What is checked
   * - Example 3.3.1
     - :math:`-u'' = 10\cos x`, Dirichlet and mixed cases, four elements,
       trapezoidal source rule
     - nodal values (both cases)
   * - Example 5.3.1, Table 5.3.1
     - cooling fin :math:`-u'' + 400u = 0` with a convective end
     - nodal values on 5, 10 and 20 elements; the heat flow :math:`Q(0)`
   * - Example 5.3.2
     - axisymmetric conduction in a cylinder with heat generation
     - exact nodal values; :math:`Q(R_0) = \pi R_0^2 g_0`
   * - Example 5.4.1, Table 5.4.1
     - conduction in a 3a × 2a rectangle with insulated edges
     - the 3 × 2 nodal values; the 6 × 4 and 12 × 8 tables; the identity of the
       dual mesh and finite element solutions on triangles
   * - Example 5.4.2, Table 5.4.2
     - conduction with a parabolic edge temperature
     - both tabulated rows (8 × 8 mesh)
   * - Example 5.4.3, Table 5.4.3
     - bus bar with heat generation and surface convection
     - both tabulated rows, to every printed digit
   * - Example 5.4.4, Table 5.4.4
     - advection–diffusion at :math:`Pe = 75` on 50 × 50 and 100 × 100 meshes
     - the diagonal profile, to :math:`2\times10^{-5}`
   * - Example 6.2.2
     - :math:`-u'' + 2u^3 = 0` with a nonlinear flux condition
     - nodal values on 4 and 8 elements
   * - Example 6.2.3, Table 6.2.2
     - conduction with :math:`k(T) = k_0(1 + k_1 T)`
     - nodal values on 4 and 8 elements
   * - Example 6.2.4
     - large-deformation bar, :math:`a(u') = EA(1 + 1.5u' + 0.5u'^2)`
     - tip displacement at three load levels (a kernel written in Python)
   * - Example 6.3.1, Table 6.3.1
     - two-dimensional nonlinear conduction
     - both tabulated rows, with Newton's method and with direct iteration
   * - Example 6.3.2
     - nonlinear bus bar, :math:`k = 20 + 0.2T`
     - the linear and nonlinear nodal values
   * - Tables 7.5.1–7.5.4
     - pinned and clamped beams, three dual mesh models, four meshes
     - centre deflections; the exact centre moment of the mixed models
   * - Table 7.5.3
     - functionally graded beams, seven power-law indices
     - normalized centre deflections for all three models
   * - Table 7.6.1
     - von Kármán bending of a pinned beam, six load levels
     - centre deflections with Newton's method and with direct iteration
       (:math:`\gamma = 0.35`)
   * - Table 8.6.1
     - hinged circular plate, five meshes
     - centre deflection; agreement with the analytical solution
   * - Table 8.6.2
     - clamped functionally graded circular plate, ten power-law indices
     - centre deflections (32 elements)
   * - Table 9.8.1
     - plate under a uniform edge load, four meshes
     - displacements and element-centre stresses
   * - Example 9.8.1
     - uniform edge stress
     - :math:`u = t_0 a / E` exactly; exactly uniform stress
   * - Table 9.8.2
     - creeping flow squeezed between plates (20 × 16 graded mesh)
     - the horizontal velocity profiles at two stations; the recovered pressure
   * - Table 9.8.3
     - lid-driven cavity at :math:`Re = 0` and :math:`Re = 1000`
     - the centreline profile for both, by load stepping and by direct
       iteration
   * - Table 9.9.1
     - pressurized thick cylinder, quadrilateral and triangular meshes
     - radial displacements; monotone convergence to the Lamé solution
   * - Table 10.5.1
     - clamped functionally graded plate, six power-law indices, four meshes
     - normalized centre deflections

Additional checks that do not come from a table: the patch test on distorted
meshes of all five element types, the geometric closure of the dual mesh (the
control volumes tile each element and their interface areas balance), exactness
of the automatic differentiation, exactness of the Gauss rules, global
conservation of the reactions, the stress concentration factor of a plate with
a hole, the transient solution against a Fourier series, the equivalence of the
dual mesh and finite element methods on simplices, and the effect of reduced
integration on shear locking in beams and plates.

Agreement
---------

Where the mesh and the data are fully specified, the code reproduces the
published values to the last printed digit.  Two classes of small differences
remain, both documented in the tests:

* **Rounding.** The book prints four or five decimals; the tests therefore
  accept a difference of one unit in the last printed digit (for example
  :math:`1.1\times10^{-4}` for Table 7.5.1).
* **Meshes that the text does not pin down.** For the pressurized cylinder
  (Table 9.9.1) the book gives the number of subdivisions across the wall and
  the total element count but not the exact circumferential distribution, and
  for triangles it does not say how the quadrilaterals are split.  The tests
  therefore compare the quadrilateral results within half a percent and, in
  addition, check monotone convergence to the analytical solution for both
  element types.

Two discrepancies in the book
-----------------------------

Two published values could not be reproduced, and in both cases the evidence
points to the table rather than to the code.

1. **Table 5.4.2, last entry.**  The table gives the dual mesh temperature at
   :math:`(x, y) = (0.175, 0.05)` as 335.34 K.  Every other entry of that table
   is reproduced here to the last printed digit, the finite element value at
   the same point is 335.49 K, and the dual mesh solution is above the finite
   element solution at every other point of the table.  This code gives
   335.545 K, which suggests that 335.34 is a typographical error for 335.54.

2. **Table 5.4.3, caption.**  The table is captioned "20 × 10 mesh", but
   Fig. 5.4.15(b) shows a 10 × 5 primal mesh, and Example 6.3.2 quotes the same
   numbers for its 10 × 5 mesh.  On a 10 × 5 mesh this code reproduces all
   eighteen tabulated values to every printed digit; on a 20 × 10 mesh the
   values differ in the second decimal.  The data therefore come from the
   10 × 5 mesh of the figure.

Reproducing a single case
-------------------------

Every test is an ordinary function, so a case can be run on its own:

.. code-block:: console

   pytest tests/python/test_heat_transfer.py -k example_5_4_3 -v
   pytest tests/python/test_beams.py -k table_7_5_1 -v

and the ``examples`` directory holds stand-alone scripts for several of them.
