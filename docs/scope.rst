Scope: a multiphysics framework
===============================

.. contents::
   :local:
   :depth: 2
   :class: this-will-duplicate-information-and-it-is-still-useful-here

What "multiphysics" means here
------------------------------

A *multiphysics* problem is one in which several physical processes act on
the same body at the same time and each changes the others.  Examples are a
flow that carries heat while the heat drives the flow by buoyancy, or a solid
whose temperature makes it expand while its deformation changes the heat
generated in it.  Mathematically it is a system of partial differential
equations for several fields, in which the equation of one field contains the
others.

A *multiphysics framework* is software in which such a system can be written
down term by term, with no term knowing in advance which others it will be
combined with, and solved.  dualmesh is one in the following precise sense:

1. **Any number of fields.**  A problem declares its variables by name, and
   each is governed by a conservation law in the canonical form
   :math:`-\nabla \cdot \mathbf{F} + S = 0`.

2. **Physics as composable terms.**  The flux :math:`\mathbf{F}` and the
   source :math:`S` of each equation are the sum of the contributions of
   *kernels*, and each kernel may read the value and the gradient of any
   variable and any *property*.  A property may in turn
   depend on any variable.  Two physics are therefore coupled by adding a
   kernel or a property object that reads the other field, such as
   ``heat_convection`` (the flow carries the heat), ``Boussinesq_buoyancy``
   (the temperature drives the flow), the thermal strain of
   ``linear_elastic_stress`` (the temperature loads the solid), or
   ``coupled_force`` for any linear coupling.

3. **Monolithic, fully coupled solution.**  All fields are solved together as
   one nonlinear system :math:`R(U) = 0`, where :math:`U` holds every degree of
   freedom of every field.  Newton's method uses the exact Jacobian
   :math:`\partial R / \partial U`, including the off-diagonal blocks that
   couple one field to another, because every kernel is evaluated in
   forward-mode automatic differentiation.  A coupled problem therefore
   converges quadratically, like a single one, which the natural convection
   test (:file:`tests/python/test_multiphysics.py`) checks.

4. **One description, several discretisations.**  The same problem is
   discretised by the finite element method, the vertex-centred or the
   cell-centred finite volume method, or the dual mesh control domain method,
   chosen by one keyword.  An element type or a physics that a method cannot
   treat correctly is rejected with an error message that states the reason.

In this design the three physics modules are three sets of kernels that share
one assembly and solution code, and a coupled problem needs no code beyond that
already needed for the single-physics problems.

What is in place
----------------

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Area
     - Capability
   * - Heat transfer
     - Conduction (linear, spatially varying, temperature dependent), storage,
       generation and convection by a computed flow.  Prescribed temperature,
       flux, convection and radiation boundaries.
   * - Solid mechanics
     - Linear elasticity in plane stress, plane strain, axisymmetric and
       three-dimensional form, for isotropic and orthotropic materials and
       with thermal strain.  Finite strain with a neo-Hookean material, creep
       laws given as expressions, eigenstrains, and frictionless contact
       between bodies by a penalty.  Euler-Bernoulli and Timoshenko beams,
       classical and first-order shear deformation plates, axisymmetric
       plates, von Kármán nonlinearity and functionally graded sections.
   * - Fluid dynamics
     - Steady and transient incompressible Navier-Stokes flow by the penalty
       method, by a stabilized equal-order pressure-velocity formulation
       (PSPG and SUPG) or with Taylor-Hood elements, and Boussinesq buoyancy.
       Transient flow monolithically or by the projection splitting of
       Nek5000 and nekRS (finite element, dual mesh control domain and
       vertex-centered finite volume methods).
   * - General equations
     - Systems of any number of fields in general form (flux and source as
       expressions of the fields and their gradients) and in coefficient
       form, compiled and differentiated automatically.
   * - Eigenvalues
     - The eigenvalues and modes of any problem, for a linear or quadratic
       dependence on the eigenvalue.
   * - Coupling
     - Monolithic, exact-Jacobian coupling of any fields on one mesh.
       Verified on natural convection [DeVahlDavis1983]_.
   * - Discretisation
     - Galerkin finite elements, vertex- and cell-centred finite volumes and
       the dual mesh control domain method.  Fourteen element types, including
       prisms, pyramids and serendipity elements.  Cartesian, axisymmetric and
       spherical coordinates.
   * - Time
     - The :math:`\theta` family (forward and backward Euler, Crank-Nicolson)
       with fixed, error-controlled and iteration-controlled step sizes.
   * - Nonlinear solvers
     - Newton with exact AD Jacobians, direct (Picard) iteration with
       relaxation, load stepping.
   * - Linear solvers
     - Sparse LU, and BiCGSTAB, GMRES and CG with ILU(0), ILUT, Jacobi or
       smoothed aggregation algebraic multigrid preconditioning, chosen
       automatically.  A saddle point solver for flow.  PETSc for distributed
       problems (required by a build with MPI) and as an option in serial.
   * - Gather-scatter
     - PETSc's star forests or gslib (Nek5000), chosen per problem.
   * - Parallelism
     - OpenMP threads in assembly, and MPI across processes.
   * - Meshes
     - Built-in generators, every format that meshio reads, local refinement
       of triangular meshes and uniform refinement of all linear types.
   * - Verification
     - Reddy's book examples, analytical solutions, OpenFOAM cross-checks, and
       a method-of-manufactured-solutions study of every method and element.
   * - Interfaces
     - Python API, command-line tools for the object reference and the side
       sets of a mesh file.

How far it is from MOOSE, COMSOL, OpenFOAM and Nek5000
------------------------------------------------------

The comparison below is between dualmesh and the codes whose capabilities it
measures itself against.  MOOSE [MOOSE2025]_ is an open-source framework built
on libMesh [libMesh2006]_ and PETSc, and COMSOL Multiphysics a commercial
package.  OpenFOAM [OpenFOAM1998]_ is the open-source finite volume code of
computational fluid dynamics, and Nek5000 and its successor nekRS
[Fischer2021]_ are the spectral element codes of incompressible flow at scale.  Its purpose is to
state the differences plainly, so that a reader can judge whether dualmesh
suits a given problem.

**Where dualmesh is of the same kind.**  Like MOOSE, dualmesh has registered
objects with validated parameters, a canonical residual form, monolithic
coupling, and Jacobians by automatic differentiation, and a coupled problem is
set up by listing variables, kernels, property objects and boundary conditions.

**Where dualmesh offers something the others do not.**  dualmesh provides the
dual mesh control domain method, and it solves one problem description by four
discretisations that can be compared directly, with the same assembly, the same
solvers and the same verification.  This is the purpose for which dualmesh was
written.

**Where dualmesh is substantially more limited**, in decreasing order of how
much each limitation restricts the problems that can be solved:

1. **Breadth of physics.**  dualmesh has 52 objects, eight physics and two
   couplings.  MOOSE's physics modules and COMSOL's add-on modules cover,
   among much else, plasticity, contact with friction, turbulent flow, porous
   media, phase field, electromagnetics and chemical reactions.  In dualmesh
   such a model is written in general form, without the dedicated objects,
   material models and verified defaults of those codes.

2. **Scale.**  dualmesh distributes the mesh with PETSc's DMPlex (PT-Scotch or
   ParMETIS partitions it) and solves through PETSc with algebraic multigrid
   (hypre BoomerAMG), field split and the parallel direct solver MUMPS; the
   values at shared nodes are exchanged through PETSc's star forests or gslib.
   Its multigrid iteration count stays flat from one to four processes, the
   largest count run so far, and the projection time integration of a flow
   keeps 65 to 78 % parallel efficiency at 16 000 nodes per process on four
   processes of a laptop.  Scaling studies on large machines are still to be
   done, where MOOSE, OpenFOAM and Nek5000 have been run for years (Nek5000 to
   millions of processes, at 2 000 to 4 000 points per process).  The whole mesh is still read or
   generated by the first process, and a Python script builds it on every
   process.

3. **Coupling across meshes and time scales.**  Every field of a dualmesh
   problem is defined on one mesh and advances with one time step.  MOOSE's
   MultiApps couple separate applications on different meshes and time scales
   with transfers between them, and COMSOL couples physics on different domains
   and dimensions.

4. **Computational fluid dynamics.**  dualmesh solves laminar incompressible
   flow, steady or transient, monolithically or by the projection splitting of
   Nek5000, and natural convection.  OpenFOAM offers RANS and LES turbulence
   models with wall functions, upwind-biased and limited convection schemes,
   compressible and multiphase solvers and moving meshes, and Nek5000 and nekRS
   offer high-order spectral elements, LES and runs on GPUs.  dualmesh has no
   turbulence model yet, and it stabilizes convection by streamline upwinding
   only.

5. **Time integration and analysis types.**  dualmesh has the :math:`\theta`
   family, and the backward differences of order 1 to 3 in the projection
   integration of a flow only, with no scheme for second time derivatives, and
   no frequency-domain or optimisation solvers.

6. **Contact.**  The contact of dualmesh is frictionless and pairs the two
   surfaces at their closest points, which converges at first order when
   their meshes do not match.  MOOSE and COMSOL offer friction and mortar
   methods.

7. **Geometry and user interface.**  COMSOL provides CAD, meshing and a
   graphical interface, and MOOSE provides input-file syntax checking and a
   graphical front end.  dualmesh generates simple meshes itself and reads
   everything else (for instance from Gmsh) through meshio, and is driven from
   Python.

In summary, dualmesh is a multiphysics framework in its architecture and in the
way problems are coupled and solved, and every capability is verified against
analytical solutions, manufactured solutions or published reference results.
It covers a far narrower range of physics, and it does not yet run very
large parallel problems.  Items 1, 2 and 4 would most extend its capabilities:
dedicated models for plasticity, turbulence and phase field, scaling studies
on large machines, and the convection schemes and turbulence models of
computational fluid dynamics.
