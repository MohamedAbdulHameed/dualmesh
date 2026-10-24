Verification
============

Every problem of the verification suite has a reference: an exact solution, a
published value, or an independent code.  The suite in ``tests/python`` solves
the problems below and compares every computed value with its reference.  The
directory ``tests/cpp`` holds the structural tests.  A manufactured-solution
study measures the order of convergence of every method on every element type,
:doc:`benchmarks/index` gives the benchmarks of flow and of the solvers, and
:doc:`openfoam` compares the solvers with an independent code.

.. code-block:: console

   pytest                                       # the verification suite
   ctest --test-dir build --output-on-failure   # the C++ unit tests

What is covered
---------------

Where a problem has no exact solution, its reference is the published dual
mesh solution on the same mesh [Reddy2024]_, which checks that the code
implements the method as published.

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Problem
     - What is checked
   * - :math:`-u'' = 10\cos x`, Dirichlet and mixed cases, four elements,
       trapezoidal source rule
     - nodal values (both cases)
   * - cooling fin :math:`-u'' + 400u = 0` with a convective end
     - nodal values on 5, 10 and 20 elements, and the heat flow :math:`Q(0)`
   * - axisymmetric conduction in a cylinder with heat generation
     - exact nodal values, and :math:`Q(R_0) = \pi R_0^2 g_0`
   * - conduction in a :math:`3a \times 2a` rectangle with insulated edges
     - nodal values on :math:`3 \times 2`, :math:`6 \times 4` and
       :math:`12 \times 8` meshes, and the identity of the dual mesh and finite
       element solutions on triangles
   * - conduction with a parabolic edge temperature
     - nodal values on an :math:`8 \times 8` mesh
   * - bus bar with heat generation and surface convection
     - nodal values on a :math:`10 \times 5` mesh, to every printed digit
   * - advection-diffusion at :math:`Pe = 75` on :math:`50 \times 50` and
       :math:`100 \times 100` meshes
     - the diagonal profile, to :math:`2\times10^{-5}`
   * - :math:`-u'' + 2u^3 = 0` with a nonlinear flux condition
     - nodal values on 4 and 8 elements
   * - conduction with :math:`k(T) = k_0(1 + k_1 T)`
     - nodal values on 4 and 8 elements
   * - large-deformation bar, :math:`a(u') = EA(1 + 1.5u' + 0.5u'^2)`
     - tip displacement at three load levels (a kernel written in Python)
   * - two-dimensional nonlinear conduction
     - nodal values, with Newton's method and with direct iteration
   * - nonlinear bus bar, :math:`k = 20 + 0.2T`
     - the linear and nonlinear nodal values
   * - pinned and clamped beams, three dual mesh models, four meshes
     - centre deflections, and the exact centre moment of the mixed models
   * - functionally graded beams, seven power-law indices
     - normalized centre deflections for all three models
   * - von Kármán bending of a pinned beam, six load levels
     - centre deflections with Newton's method and with direct iteration
       (:math:`\gamma = 0.35`)
   * - hinged circular plate, five meshes
     - centre deflection, and agreement with the analytical solution
   * - clamped functionally graded circular plate, ten power-law indices
     - centre deflections (32 elements)
   * - plate under a uniform edge load, four meshes
     - displacements and element-centre stresses
   * - uniform edge stress
     - :math:`u = t_0 a / E` exactly, and an exactly uniform stress
   * - creeping flow squeezed between plates (:math:`20 \times 16` graded mesh)
     - the horizontal velocity profiles at two stations, and the recovered
       pressure
   * - lid-driven cavity at :math:`Re = 0` and :math:`Re = 1000`
     - the centreline profile for both, by load stepping and by direct
       iteration
   * - pressurized thick cylinder, quadrilateral and triangular meshes
     - radial displacements, and monotone convergence to the Lamé solution
   * - clamped functionally graded plate, six power-law indices, four meshes
     - normalized centre deflections

Additional checks without a published reference are the patch test on
distorted meshes of every element type, the geometric closure of the dual mesh
(the control volumes partition each element and their interface areas balance),
exactness
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

* **Rounding.** The published values have four or five decimals, so the tests
  accept a difference of one unit in the last printed digit.
* **Meshes that are not fully specified.** For the pressurized cylinder the
  published mesh is given by the number of subdivisions across the wall and
  the total element count, without the exact circumferential distribution or
  the split of the quadrilaterals into triangles.  The tests therefore compare
  the quadrilateral results within half a percent and, in addition, check
  monotone convergence to the analytical solution for both element types.

Order of accuracy: manufactured solutions
-----------------------------------------

A published table checks a discretisation at one mesh.  Whether it converges,
and at the rate its theory predicts, is checked by the **method of
manufactured solutions** [Roache2002]_ [SalariKnupp2000]_.  A smooth function
:math:`u^\star(\mathbf{x}, t)` is chosen, which need not satisfy any physical
problem.  The governing operator is applied to it symbolically, and the result,
:math:`f = -\nabla \cdot \mathbf{F}(u^\star) + S(u^\star)`, is added to the
problem as a source, so that :math:`u^\star` is its exact solution.  The
boundary values are taken from :math:`u^\star` as well.  Solving on a sequence
of meshes of size :math:`h` then gives the error :math:`e_h = u_h - u^\star`
exactly, and its observed order between two meshes is

.. math::

   p = \frac{\ln \left( \lVert e_{h_1} \rVert / \lVert e_{h_2} \rVert \right)}
            {\ln (h_1 / h_2)} .

:mod:`dualmesh.mms` does the symbolic work with SymPy: a
:class:`~dualmesh.mms.ManufacturedSolution` holds the exact fields and the
physics of the equations, added with ``add_physics`` as to a problem
(``general_form_PDE`` for any equation written as text or by its
coefficients, ``heat_transfer``, ``solid_mechanics`` and ``fluid_dynamics``).  Each
physics states, in SymPy, the flux and the source of the objects it
generates, so that the forcing is derived from the equations the code
assembles.  The study derives the forcing, including the metric factors of the
axisymmetric and spherical divergences, and runs the convergence study.  The
errors are measured by :meth:`~dualmesh.Problem.error_norms` in the two norms
in which the convergence theory of every method is stated,

.. math::

   \lVert e \rVert_{L^2} = \Bigl( \int_\Omega e^2 \,\mathrm{d}V \Bigr)^{1/2} ,
   \qquad
   \lvert e \rvert_{H^1} = \Bigl( \int_\Omega \lvert \nabla e \rvert^2 \,\mathrm{d}V \Bigr)^{1/2} ,

where :math:`u_h` is the element interpolation of the nodal values.

**Expected orders.**  For an element of polynomial order :math:`p`:

====================  =================  ====================
Method                :math:`L^2` error  :math:`H^1` seminorm
====================  =================  ====================
``FEM``               :math:`p + 1`      :math:`p`
``DMCDM``             2                  :math:`p`
====================  =================  ====================

The first row is classical finite element theory, the extra order in
:math:`L^2` coming from the duality argument of Aubin and Nitsche
[Aubin1967]_ [Nitsche1968]_.  DMCDM is optimal in the
energy seminorm, but their :math:`L^2` order remains 2 on quadratic elements,
because a control volume balance is not a Galerkin projection
(:doc:`theory/elements` explains where the interfaces lie and why their
position matters).

**Coverage.**  ``tests/python/test_mms.py`` runs 161 convergence studies.  They
cover both methods on every element type each accepts, in one, two and
three dimensions, and in Cartesian, axisymmetric and spherical coordinates.
They include variable, solution-dependent (nonlinear) and constant
coefficients, advection in the conservative and the non-conservative form with
reaction, the transient problem with the time step refined with the mesh
(Crank-Nicolson, :math:`\Delta t = h/4`), and plane-strain, plane-stress,
axisymmetric and three-dimensional elasticity.  Every mesh sequence is a
structured grid mapped by a smooth nonlinear transformation, so that no element
is a parallelogram and the rates are those of a general mesh.  A rate is
accepted within 0.2 below and 0.4 above the expected one.  The upper margin
allows for the pre-asymptotic behaviour of a method converging to a lower
order.

**Results.**  The observed orders on the finest pair of meshes for the diffusion
problem with a variable coefficient, as :math:`L^2` / :math:`H^1`, are:

============  =============  =============
Element       ``FEM``        ``DMCDM``
============  =============  =============
``Tri3``      1.99 / 1.00    1.99 / 1.00
``Quad4``     2.00 / 1.00    2.00 / 1.00
``Tri6``      3.00 / 1.99    2.01 / 1.99
``Quad8``     3.00 / 2.00    n/a
``Quad9``     3.00 / 2.00    2.05 / 2.00
``Tet4``      1.94 / 0.98    1.96 / 0.98
``Hex8``      1.99 / 1.01    2.00 / 1.01
``Wedge6``    1.95 / 0.98    1.96 / 0.98
``Pyramid5``  1.99 / 1.00    n/a
``Tet10``     3.00 / 1.96    2.66 / 1.96
``Hex20``     3.06 / 2.08    n/a
``Hex27``     2.99 / 2.00    2.72 / 2.00
============  =============  =============

The two-dimensional sequences use :math:`4, 8, 16, 32` elements per side.  The
three-dimensional sequences use :math:`3, 6, 12` for the linear elements and
:math:`2, 4, 8` for the quadratic ones.  On these coarser three-dimensional
meshes the quadratic control volume rates are pre-asymptotic: they lie between
the finite element method's 3 and their asymptotic value of 2, which the
two-dimensional sequences, reaching finer meshes, show at 2.0.  Every entry
agrees with the table of expected orders.  The complete study runs in about
80 seconds.

A user-defined study requires only a few lines:

.. code-block:: python

   import dualmesh as dm
   from dualmesh import mms

   study = mms.ManufacturedSolution({"u": "sin(pi*x)*cos(pi*y) + x*y"}, dimension=2)
   study.add_physics("general_form_PDE", variables="u", equations="-div((1 + 0.5*x*y)*grad(u)) = 0")  # the forcing is derived
   result = study.convergence_study(
       lambda n: dm.generate_rectangle_mesh(0, 1, 0, 1, n, n, element_type="Tri6"),
       [4, 8, 16, 32],
       method="FEM",
   )
   print(result.table())  # errors and observed orders

Reactor criticality: the 2D IAEA PWR benchmark
----------------------------------------------

The two-dimensional IAEA PWR benchmark is a quarter of a PWR core
([ANL7416]_, problem 11-A2).  The core has 177 fuel assemblies of
:math:`20\ \mathrm{cm}`, two fuel compositions and nine fully rodded
assemblies.  A water reflector of :math:`20\ \mathrm{cm}` surrounds the core.
The model is two-group diffusion theory with an axial buckling of
:math:`0.8 \times 10^{-4}\ \mathrm{cm}^{-2}` in all regions and groups.
The outer boundary has no incoming current, which the benchmark gives as
:math:`\partial \phi_g / \partial n = -0.4692\, \phi_g / D_g`.  The script
``verification/benchmarks/neutronics/run_iaea_2d_pwr.py`` writes the
two-group equations with ``general_form_PDE`` (see
:ref:`tutorial-criticality`), and this condition as
``"-diffusion_1*dphi1/dn = phi1/2.1312"`` and its thermal counterpart.  The reference is the extrapolated finite-difference
solution of problem 11-A2-1: :math:`k_\mathrm{eff} = 1.02959` (Table 1) and
the zone average thermal fluxes (Table 3).  The assembly powers follow from
these fluxes and are normalised to a core average of one.

:numref:`iaea-2d-results` gives the error of :math:`k_\mathrm{eff}` and
of the assembly powers.  Each 20 cm assembly has :math:`m \times m`
square elements.

.. _iaea-2d-results:

.. table:: The 2D IAEA PWR benchmark: error of :math:`k_\mathrm{eff}` (pcm)
   and largest error of the assembly powers (%) for :math:`m \times m`
   elements per assembly.

   ======  ===============  ===============
   m       ``FEM``          ``DMCDM``
   ======  ===============  ===============
   4       25.9 / 8.38      46.8 / 12.4
   8       5.5 / 2.18       9.9 / 3.21
   16      1.2 / 0.61       2.2 / 0.87
   32      0.2 / 0.22       0.4 / 0.29
   ======  ===============  ===============

Both methods converge to the reference at second order in the element size.
At :math:`m = 32` their error of :math:`k_\mathrm{eff}` is less than 0.5 pcm,
and their largest error of the assembly powers is less than 0.3 %.

.. figure:: _static/figures/neutronics/iaea_2d_pwr_k_convergence.png
   :width: 85%

   The 2D IAEA PWR benchmark: error of :math:`k_\mathrm{eff}` against the
   element size.

.. figure:: _static/figures/neutronics/iaea_2d_pwr_assembly_power_error.png
   :width: 85%

   The 2D IAEA PWR benchmark: error of the assembly powers with ``FEM`` and
   32 elements per assembly.

Coupled problems
----------------

The natural convection benchmark of de Vahl Davis [DeVahlDavis1983]_ checks a
two-way coupled problem (``tests/python/test_multiphysics.py``): the average
Nusselt number and the velocity maxima of the square cavity at Rayleigh
numbers from :math:`10^3` to :math:`10^5`, with both methods, and the quadratic convergence of
Newton's method on the coupled system.  The numbers are in
:doc:`theory/heat_and_fluids`.

Reproducing a single case
-------------------------

Every test is an ordinary function, so a file or a single test can be run on
its own:

.. code-block:: console

   pytest tests/python/test_heat_transfer.py -v
   pytest tests/python/test_beams.py -v

and the ``examples`` directory holds stand-alone scripts for several of them.
