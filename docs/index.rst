dualmesh
========

**dualmesh** is a multiphysics framework for **heat transfer**, **solid
mechanics** and **fluid dynamics**.  A problem is any number of fields --
temperatures, displacements, velocities, or quantities of your own -- each
governed by a conservation law written as a sum of named terms, and every term
may depend on every field.  All the fields are solved together, as one
monolithic system, by Newton's method with an exact Jacobian computed by
automatic differentiation, so the coupling between the physics is as exact as
the physics itself.

The same problem description can be discretised by four methods, chosen by one
keyword: the Galerkin **finite element method**, the vertex-centred and the
cell-centred **finite volume methods**, and the **dual mesh control domain
method** (DMCDM) of J. N. Reddy, which combines the finite element
interpolation with the finite volume balance.  The finite element and finite
volume methods are held to the same verification standard as the DMCDM, and an
element or a physics that a method cannot handle is refused with an
explanation rather than solved wrongly.

The library is written in C++17 and driven from Python.  Its structure follows
MOOSE [MOOSE2025]_: physics is added as *kernels*, *boundary conditions* and
*materials*, which are registered objects with validated, self-documenting
parameters; meshes are generated or read from standard files; and the solvers
(Newton with exact automatic differentiation, direct iteration, load stepping,
adaptive time integration, threaded and distributed linear algebra) are shared
by every physics.  :doc:`scope` sets out how far that goes and how it compares
with MOOSE and COMSOL Multiphysics.

.. code-block:: python

   import dualmesh as dm

   mesh = dm.generate_rectangle_mesh(
       x_min=0.0, x_max=0.1, y_min=0.0, y_max=0.05,
       num_x_elements=10, num_y_elements=5)

   problem = dm.Problem(mesh, method="dmcdm")
   problem.add_variable("temperature")
   problem.add_kernel("HeatConduction", variable="temperature",
                      thermal_conductivity=20.0)
   problem.add_kernel("HeatSource", variable="temperature", heat_source=1.0e6)
   problem.add_boundary_condition("DirichletBC", variable="temperature",
                                  boundary="left", value=40.0)
   problem.add_boundary_condition("DirichletBC", variable="temperature",
                                  boundary="right", value=10.0)
   problem.add_boundary_condition("ConvectiveHeatFluxBC", variable="temperature",
                                  boundary="top", heat_transfer_coefficient=75.0)
   problem.solve()

   print(problem.sample("temperature", [[0.05, 0.0]]))   # 83.142 (Reddy, Table 5.4.3)

What the method is
------------------

The dual mesh control domain method keeps two meshes.  The **primal mesh** is a
mesh of finite elements and provides the interpolation of the unknowns.  The
**dual mesh** is the set of node-centred *control domains* obtained by joining
edge midpoints, face centroids, and element centroids.  The governing equation
is then satisfied in the *integral* sense over each control domain, without a
weight function:

.. math::

   \int_{CD_I} \left[ -\nabla \cdot \mathbf{F} + S \right] \, dV = 0
   \quad \Longrightarrow \quad
   -\oint_{\partial CD_I} \mathbf{F} \cdot \mathbf{n} \, dS
   + \int_{CD_I} S \, dV = 0 .

The surface integral makes the secondary variables (fluxes, forces, moments)
appear naturally on the control domain interfaces, which is the physical
appeal of the finite volume method, while the primal interpolation removes the
ad-hoc gradient reconstructions that the finite volume method needs.  Chapter 5
of Reddy's book develops the method; :doc:`theory/index` develops it in
the form the code implements.

What it solves
--------------

The physics comes in three modules, in order of coverage, plus the framework
they are built on.

**Heat transfer**: steady and transient conduction with constant, spatially
varying and temperature-dependent conductivity, volumetric sources, convection
of heat by a computed flow, and the full set of boundary conditions --
prescribed temperature, prescribed flux, convection and radiation -- in
Cartesian, axisymmetric and spherical coordinates.

**Solid mechanics**: the continuum -- linear elasticity in plane stress, plane
strain, axisymmetric and three-dimensional form, isotropic or orthotropic, with
thermal strain -- and the reduced theories of structural members:
Euler-Bernoulli and Timoshenko beams, classical and first-order shear
deformation plates, and axisymmetric circular plates, with the von Kármán
nonlinearity and sections functionally graded through the thickness
[Reddy2000]_.  Beams and plates are solid mechanics with the through-thickness
behaviour integrated out, so they are one module, not two.

**Fluid dynamics**: steady and transient viscous incompressible flow through
the penalty formulation of the Navier-Stokes equations [HughesLiuBrooks1979]_,
with buoyancy in the Boussinesq approximation.

**Coupled problems** are built from the same pieces.  Natural convection in a
heated cavity couples the flow and the energy equation in both directions
(buoyancy and convection) and reproduces the benchmark of de Vahl Davis
[DeVahlDavis1983]_ to within half a per cent in the Nusselt number with every
node-based method; thermal stress couples the temperature to the displacement;
and a kernel of your own can couple anything to anything, with its Jacobian
blocks differentiated for you.  Any advection-diffusion-reaction equation that
can be written as a flux and a source is available through the framework
kernels, which is also how a new physics is added.

What it offers
--------------

**Four discretisations of the same problem description.**  The dual mesh
control domain method, the Galerkin finite element method, and the
vertex-centred and cell-centred finite volume methods of Chapter 3 of the book.
Changing one keyword changes the method and nothing else, which is what makes a
method-to-method comparison meaningful.

**Arbitrary meshes.**  Fourteen element types -- ``Edge2``, ``Tri3``,
``Quad4``, ``Tet4``, ``Hex8``, ``Wedge6`` (prism), ``Pyramid5``, and the
quadratic ``Edge3``, ``Tri6``, ``Quad8``, ``Quad9``, ``Tet10``, ``Hex20`` and
``Hex27`` -- generated by the library or read from any format meshio supports,
and mixed freely in one mesh.  Each method accepts every element it can treat
correctly; the dual mesh methods need a dual mesh, which the serendipity
elements and the pyramid do not have, and they refuse those with the reason.

**Exact Jacobians.**  Every kernel is written in forward-mode automatic
differentiation, so Newton's method converges quadratically and a new nonlinear
term needs no hand differentiation.

**Steady, transient and nonlinear throughout.**  The :math:`\theta` family of
time integrators with fixed, error-controlled and iteration-controlled adaptive
stepping; Newton and direct iteration with relaxation; load stepping.

**Adaptive mesh refinement.**  A gradient-recovery error indicator, three
marking rules, and conforming longest-edge bisection.

**Efficient linear algebra.**  A direct solver where a factorisation is cheap
and a preconditioned Krylov solver where it is not, chosen automatically, and a
distributed solver with a two-level overlapping Schwarz preconditioner whose
iteration count does not grow with the number of processes.

**Parallel execution.**  Threaded assembly within a process and a distributed
solver across processes.

**Verification, not assertion.**  Every claim above is checked by a test.  The
tests reproduce published tables from the book, analytical solutions and
independent runs of OpenFOAM, and a systematic method-of-manufactured-solutions
study measures the order of convergence of every method on every element type
it accepts, in one, two and three dimensions.  Where a method has a limitation, the
documentation says so: see for instance the honest account in
:doc:`theory/elements` of what quadratic elements do and do not buy the dual
mesh method.

Contents
--------

.. toctree::
   :maxdepth: 2

   installation
   getting_started
   scope
   user_guide/index
   theory/index
   tutorials/index
   input_files
   objects
   api
   verification
   openfoam
   developing
   references

Citing
------

If this software contributes to your work, please cite both the method and the
software; see :doc:`references` and the ``CITATION.cff`` file in the
repository.
