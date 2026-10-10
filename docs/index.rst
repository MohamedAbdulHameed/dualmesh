dualmesh
========

**dualmesh** is a multiphysics framework for **heat transfer**, **solid
mechanics**, **fluid dynamics** and **general partial differential
equations**.  A problem
consists of any number of fields (e.g., temperatures, displacements,
velocities, or quantities defined by the user), each governed by a
conservation law written as a sum of named terms.  Every term may depend on
every field.  All the fields are solved together as one monolithic system by
Newton's method, with an exact Jacobian computed by automatic
differentiation.  The coupling between the physics is therefore represented
exactly in the Jacobian.

The same problem description can be discretised by four methods, which are
selected by one keyword: the Galerkin **finite element method**, the
vertex-centred and the cell-centred **finite volume methods**, and the **dual
mesh control domain method** (DMCDM) of J. N. Reddy [Reddy2024]_, which combines the
finite element interpolation with the finite volume balance.  All four
methods are held to the same verification standard.  An element type or a
physics that a method cannot treat correctly is refused with an explanation.

The library is written in C++17 and is driven from Python.  A model is
described in physical terms: *physics* such as ``heat_transfer``,
``solid_mechanics`` and ``incompressible_flow``, their boundary conditions,
and *couplings* such as ``thermal_expansion``.  Each physics generates the
*kernels*, *property objects* and *boundary conditions* of its equations, which
are registered objects with validated and self-documenting parameters.  An equation without a physics is
written with these objects directly.  Meshes are generated
by the library or read from standard file formats.  The solvers (Newton's
method with exact derivatives, direct iteration, load stepping, adaptive time
integration, and threaded and distributed linear algebra) are shared by every
physics.  :doc:`scope` describes the coverage of the framework and compares
it with the codes whose capabilities it measures itself against: MOOSE, COMSOL
Multiphysics, OpenFOAM, and Nek5000 and nekRS.

A two-dimensional bus bar with internal heat generation, fixed temperatures on
two sides and convection on the top (Example 5.4.3 of Reddy's book) reads:

.. code-block:: python

   import dualmesh as dm

   mesh = dm.generate_rectangle_mesh(
       x_min=0.0, x_max=0.1, y_min=0.0, y_max=0.05, num_x_elements=10, num_y_elements=5
   )

   problem = dm.Problem(mesh, method="dmcdm")
   heat = problem.add_physics("heat_transfer", "heat", thermal_conductivity=20.0, heat_source=1.0e6)
   heat.add_boundary_condition("Dirichlet_boundary_condition", "left", value=40.0)
   heat.add_boundary_condition("Dirichlet_boundary_condition", "right", value=10.0)
   heat.add_boundary_condition(
       "convective_heat_flux_boundary_condition", "top", heat_transfer_coefficient=75.0
   )
   problem.solve()

   print(problem.sample("temperature", [[0.05, 0.0]]))  # 83.142 (Reddy, Table 5.4.3)

A boundary condition named after a side set of the mesh acts on that side
set.  The complete script, with post-processors and output files, is
``examples/bus_bar.py``.

The dual mesh control domain method
-----------------------------------

The dual mesh control domain method uses two meshes.  The **primal mesh** is a
mesh of finite elements, which provides the interpolation of the unknowns.
The **dual mesh** is the set of node-centred *control domains* obtained by
joining the edge midpoints, the face centroids and the element centroids.  The
governing equation is satisfied in the integral sense over each control
domain, without a weight function:

.. math::

   \int_{\Omega_I} \left[ -\nabla \cdot \mathbf{F} + S \right] \,\mathrm{d}V = 0
   \quad \Longrightarrow \quad
   -\oint_{\partial \Omega_I} \mathbf{F} \cdot \mathbf{n} \,\mathrm{d}S
   + \int_{\Omega_I} S \,\mathrm{d}V = 0,

where :math:`\Omega_I` is the control domain of node :math:`I`,
:math:`\mathbf{F}` is the flux, :math:`S` is the source and :math:`\mathbf{n}`
is the outward unit normal.  The secondary variables (fluxes, forces and
moments) appear on the interfaces of the control domains, as in the finite
volume method.  The gradients on these interfaces are evaluated from the
finite element interpolation, so that no gradient reconstruction is needed.
Chapter 5 of Reddy's book [Reddy2024]_ develops the method, and :doc:`theory/index`
presents it in the form that the code implements.

Physics modules
---------------

**Heat transfer.**  Steady and transient conduction with constant, spatially
varying and temperature-dependent conductivity, volumetric sources, the
convection of heat by a computed flow, and prescribed temperature, prescribed
flux, convection and radiation boundary conditions, in Cartesian,
axisymmetric and spherical coordinates.

**Solid mechanics.**  Linear elasticity in plane stress, plane strain,
axisymmetric and three-dimensional form, isotropic or orthotropic, with
thermal strain, small or finite strain, and creep.  The reduced theories of
structural members are included: Euler-Bernoulli and Timoshenko beams,
classical and first-order shear deformation plates, and axisymmetric circular
plates, with the von Kármán nonlinearity and sections functionally graded
through the thickness [Reddy2000]_.

**Fluid dynamics.**  Steady and transient viscous incompressible flow, in the
penalty formulation of the Navier-Stokes equations [HughesLiuBrooks1979]_ or
in the pressure-velocity formulation with Taylor-Hood elements or stabilised
equal-order interpolation, with buoyancy in the Boussinesq approximation.

**Coupled problems** are built from the same components.  Natural convection
in a heated cavity couples the flow and the energy equation in both
directions and reproduces the benchmark of de Vahl Davis [DeVahlDavis1983]_
to within 0.2 % in the Nusselt number with every node-based method.  Thermal
stress couples the temperature to the displacement.  A kernel written by the
user can couple any field to any other, and its Jacobian is obtained by
automatic differentiation.

Capabilities
------------

**Four discretisations of one problem description.**  The dual mesh control
domain method, the Galerkin finite element method, and the vertex-centred and
cell-centred finite volume methods.  Changing one
keyword changes the method and leaves the rest of the problem unchanged,
which makes a comparison between methods meaningful.

**Arbitrary meshes.**  Fourteen element types (``Edge2``, ``Tri3``,
``Quad4``, ``Tet4``, ``Hex8``, ``Wedge6``, ``Pyramid5``, and the quadratic
``Edge3``, ``Tri6``, ``Quad8``, ``Quad9``, ``Tet10``, ``Hex20`` and
``Hex27``), generated by the library or read from any format supported by
meshio, and mixed freely in one mesh.  The dual mesh methods require a dual
mesh, which the serendipity elements and the pyramid do not have, and these
elements are refused by those methods.  Boundaries of an imported mesh are
taken from its physical groups or defined geometrically (see
:doc:`user_guide/meshes`).

**Exact Jacobians.**  Every kernel is evaluated in forward-mode automatic
differentiation.  Newton's method therefore converges quadratically, and a
new nonlinear term requires no hand differentiation.

**Steady, transient and nonlinear problems.**  The :math:`\theta` family of
time integrators with fixed, error-controlled and iteration-controlled
adaptive time steps, Newton's method and direct iteration with relaxation,
and load stepping.

**Adaptive mesh refinement.**  A gradient-recovery error indicator, three
marking rules, and conforming longest-edge bisection.

**Linear algebra.**  A direct solver where a factorisation is inexpensive and a
preconditioned Krylov solver otherwise, chosen automatically.  Distributed
problems exchange the values at shared nodes through PETSc's star forests or
through gslib, the gather-scatter of Nek5000, and are solved by PETSc, by
default with the algebraic multigrid of hypre, whose iteration count does not
grow with the number of processes.

**Parallel execution.**  Threaded assembly within a process and a
distributed solver across processes.

**Verification.**  Every capability listed above is checked by a test.  The
tests reproduce published reference solutions, analytical solutions and
independent calculations with OpenFOAM.  A study by the method of
manufactured solutions measures the order of convergence of every method on
every element type it accepts, in one, two and three dimensions.  The
limitations of each method are stated in the documentation, e.g., the effect
of quadratic elements on the dual mesh method in :doc:`theory/elements`.

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: Getting started

   installation
   getting_started
   scope

.. toctree::
   :maxdepth: 2
   :caption: Using dualmesh

   user_guide/index
   tutorials/index
   objects
   api

.. toctree::
   :maxdepth: 2
   :caption: Theory and verification

   theory/index
   verification
   benchmarks/index
   openfoam

.. toctree::
   :maxdepth: 1
   :caption: Project

   developing
   work_in_progress
   references

Citing
------

If this software contributes to your work, please cite the software (see the
``CITATION.cff`` file in the repository).
