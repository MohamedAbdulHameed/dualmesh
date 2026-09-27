Verification
============

Every number this library produces for a published problem is checked against
the published value.  The suite in ``tests/python`` reproduces the dual mesh
control domain results of Reddy's book chapter by chapter; ``tests/cpp`` holds
the structural tests that do not depend on a published table; a
manufactured-solution study measures the order of convergence of every method
on every element type; and :doc:`openfoam` compares the solvers with an
independent code.

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
meshes of every element type, the geometric closure of the dual mesh (the
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

Order of accuracy: manufactured solutions
-----------------------------------------

A published table checks a discretisation at one mesh.  Whether it converges,
and at the rate its theory predicts, is checked by the **method of
manufactured solutions** [Roache2002]_ [SalariKnupp2000]_.  A smooth function
:math:`u^\star(\mathbf{x}, t)` is chosen, which need not satisfy any physical
problem; the governing operator is applied to it symbolically, and the result,
:math:`f = -\nabla \cdot \mathbf{F}(u^\star) + S(u^\star)`, is added to the
problem as a source, so that :math:`u^\star` is its exact solution.  The
boundary values are taken from :math:`u^\star` as well.  Solving on a sequence
of meshes of size :math:`h` then gives the error :math:`e_h = u_h - u^\star`
exactly, and its observed order between two meshes is

.. math::

   p = \frac{\ln \left( \lVert e_{h_1} \rVert / \lVert e_{h_2} \rVert \right)}
            {\ln (h_1 / h_2)} .

:mod:`dualmesh.mms` does the symbolic work with SymPy: a
:class:`~dualmesh.mms.ManufacturedSolution` holds the fields and the terms of
the equations (diffusion with a variable or solution-dependent coefficient,
advection in either form, reaction, the time derivative, and linear
elasticity), derives the forcing, including the metric factors of the
axisymmetric and spherical divergences, and runs the convergence study.  The
errors are measured by :meth:`~dualmesh.Problem.error_norms` in the two norms
in which the convergence theory of every method is stated,

.. math::

   \lVert e \rVert_{L^2} = \Bigl( \int_\Omega e^2 \, dV \Bigr)^{1/2} ,
   \qquad
   \lvert e \rvert_{H^1} = \Bigl( \int_\Omega \lvert \nabla e \rvert^2 \, dV \Bigr)^{1/2} ,

where :math:`u_h` is the field each method represents: the element
interpolation of the nodal values for the node-based methods, and the linear
reconstruction :math:`U_c + \mathbf{G}_c \cdot (\mathbf{x} - \mathbf{x}_c)` in
every cell for the cell-centred method.

**Expected orders.**  For an element of polynomial order :math:`p`:

====================  =================  ====================
Method                :math:`L^2` error  :math:`H^1` seminorm
====================  =================  ====================
``fem``               :math:`p + 1`      :math:`p`
``dmcdm``, ``hfvm``   2                  :math:`p`
``zfvm``              2                  1
====================  =================  ====================

The first row is classical finite element theory, the extra order in
:math:`L^2` coming from the duality argument of Aubin and Nitsche
[Aubin1967]_ [Nitsche1968]_.  The control volume methods are optimal in the
energy seminorm but do not gain that extra order on quadratic elements,
because a control volume balance is not a Galerkin projection
(:doc:`theory/elements` explains where the interfaces sit and why that
matters).  The cell-centred method carries one value per cell whatever the
element and reconstructs a linear field from it, so the element order improves
only its geometry.

**Coverage.**  ``tests/python/test_mms.py`` runs 161 convergence studies: the
four methods on every element type each accepts, in one, two and three
dimensions; Cartesian, axisymmetric and spherical coordinates; variable,
solution-dependent (nonlinear) and constant coefficients; advection in the
conservative and the non-conservative form with reaction; the transient
problem with the time step refined with the mesh (Crank-Nicolson,
:math:`\Delta t = h/4`); and plane-strain, plane-stress, axisymmetric and
three-dimensional elasticity.  Every mesh sequence is a structured grid bent by
a smooth map, so that no element is a parallelogram and the rates are those of
a general mesh.  A rate is accepted within 0.2 below and 0.4 above the
expected one; the upper margin allows for the pre-asymptotic behaviour of a
method converging to a lower order.

**Results.**  The observed orders on the finest pair of meshes for the diffusion
problem with a variable coefficient, as :math:`L^2` / :math:`H^1`, are:

============  =============  =============  =============  =============
Element       ``fem``        ``dmcdm``      ``hfvm``       ``zfvm``
============  =============  =============  =============  =============
``Tri3``      1.99 / 1.00    1.99 / 1.00    1.99 / 1.00    2.00 / 1.00
``Quad4``     2.00 / 1.00    2.00 / 1.00    1.99 / 1.00    2.00 / 1.00
``Tri6``      3.00 / 1.99    2.01 / 1.99    2.04 / 1.99    2.00 / 1.00
``Quad8``     3.00 / 2.00    --             --             2.00 / 1.00
``Quad9``     3.00 / 2.00    2.05 / 2.00    2.23 / 2.00    2.00 / 1.00
``Tet4``      1.94 / 0.98    1.96 / 0.98    1.96 / 0.98    1.97 / 0.98
``Hex8``      1.99 / 1.01    2.00 / 1.01    1.98 / 1.01    1.97 / 0.99
``Wedge6``    1.95 / 0.98    1.96 / 0.98    1.96 / 0.98    1.98 / 0.99
``Pyramid5``  1.99 / 1.00    --             --             1.98 / 0.99
``Tet10``     3.00 / 1.96    2.66 / 1.96    2.86 / 1.89    1.95 / 0.97
``Hex20``     3.06 / 2.08    --             --             1.94 / 0.98
``Hex27``     2.99 / 2.00    2.72 / 2.00    2.96 / 2.00    1.94 / 0.98
============  =============  =============  =============  =============

The two-dimensional sequences use :math:`4, 8, 16, 32` elements per side; the
three-dimensional ones :math:`3, 6, 12` for the linear elements and
:math:`2, 4, 8` for the quadratic ones, which is why the quadratic control
volume rates in three dimensions are still on their way down from the
finite element method's 3 to their asymptote of 2 (the two-dimensional
sequences, which reach finer meshes, show it at 2.0).  Every entry agrees with
the table of expected orders.

The study found two defects, both now fixed.  The old-time part of the
:math:`\theta` method was evaluated at the new time, which cost the
Crank-Nicolson scheme its second order in time when the coefficients or the
sources depend on time (the observed order was 1.27); and the cell-centred
method's Jacobian omitted the dependence of the non-orthogonal correction on
the neighbours' values, which made Newton's method converge linearly on skewed
meshes.  The study also measures cost: it runs in about 80 seconds.

A study of your own takes a few lines:

.. code-block:: python

   import dualmesh as dm
   from dualmesh import mms

   study = mms.ManufacturedSolution(
       {"u": "sin(pi*x)*cos(pi*y) + x*y"},
       [mms.Diffusion("u", diffusivity="1 + 0.5*x*y")],
       dimension=2,
   )
   result = study.convergence_study(
       lambda n: dm.generate_rectangle_mesh(0, 1, 0, 1, n, n, element_type="Tri6"),
       [4, 8, 16, 32],
       method="fem",
   )
   print(result.table())          # errors and observed orders

Coupled problems
----------------

The natural convection benchmark of de Vahl Davis [DeVahlDavis1983]_ checks a
two-way coupled problem (``tests/python/test_multiphysics.py``): the average
Nusselt number and the velocity maxima of the square cavity at Rayleigh
numbers from :math:`10^3` to :math:`10^5`, with the dual mesh, finite element
and vertex-centred finite volume methods, and the quadratic convergence of
Newton's method on the coupled system.  The numbers are in
:doc:`theory/heat_and_fluids`.

Reproducing a single case
-------------------------

Every test is an ordinary function, so a case can be run on its own:

.. code-block:: console

   pytest tests/python/test_heat_transfer.py -k example_5_4_3 -v
   pytest tests/python/test_beams.py -k table_7_5_1 -v

and the ``examples`` directory holds stand-alone scripts for several of them.
