Scope: a multiphysics framework
===============================

.. contents::
   :local:
   :depth: 2

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
combined with, and solved.  dualmesh is one in the following precise sense,
which is the sense of MOOSE [MOOSE2025]_:

1. **Any number of fields.**  A problem declares its variables by name, and
   each is governed by a conservation law in the canonical form
   :math:`-\nabla \cdot \mathbf{F} + S = 0`.

2. **Physics as composable terms.**  The flux :math:`\mathbf{F}` and the
   source :math:`S` of each equation are the sum of the contributions of
   *kernels*, and each kernel may read the value and the gradient of any
   variable and any *material property*.  A material property may in turn
   depend on any variable.  Coupling two physics is therefore a matter of
   adding a kernel or a material that reads the other field, such as
   ``HeatConvection`` (the flow carries the heat), ``BoussinesqBuoyancy``
   (the temperature drives the flow), the thermal strain of
   ``LinearElasticStress`` (the temperature loads the solid), or
   ``CoupledForce`` for any linear coupling.

3. **Monolithic, fully coupled solution.**  All fields are solved together as
   one nonlinear system :math:`R(U) = 0`, where :math:`U` holds every degree of
   freedom of every field.  Newton's method uses the exact Jacobian
   :math:`\partial R / \partial U`, including the off-diagonal blocks that
   couple one field to another, because every kernel is evaluated in
   forward-mode automatic differentiation.  A coupled problem therefore
   converges quadratically, like a single one; the natural convection test
   (:file:`tests/python/test_multiphysics.py`) checks this.

4. **One description, several discretisations.**  The same problem is
   discretised by the finite element method, the vertex-centred or the
   cell-centred finite volume method, or the dual mesh control domain method,
   chosen by one keyword.  An element type or a physics that a method cannot
   treat correctly is refused with the reason.

Built this way, the three physics modules are three sets of kernels on one
engine, not three programs, and a coupled problem needs no code that is not
already needed for the single-physics ones.

What is in place
----------------

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Area
     - Capability
   * - Heat transfer
     - Conduction (linear, spatially varying, temperature dependent), storage,
       generation, convection by a computed flow; prescribed temperature,
       flux, convection and radiation boundaries.
   * - Solid mechanics
     - Small-strain linear elasticity (plane stress, plane strain,
       axisymmetric, three-dimensional; isotropic and orthotropic; thermal
       strain); Euler-Bernoulli and Timoshenko beams, classical and first-order
       shear deformation plates, axisymmetric plates, von Kármán nonlinearity,
       functionally graded sections.
   * - Fluid dynamics
     - Steady and transient incompressible Navier-Stokes flow by the penalty
       method; Boussinesq buoyancy.
   * - Coupling
     - Monolithic, exact-Jacobian coupling of any fields on one mesh.
       Verified on natural convection [DeVahlDavis1983]_.
   * - Discretisation
     - Galerkin finite elements, vertex- and cell-centred finite volumes, dual
       mesh control domain method; fourteen element types including prisms,
       pyramids and serendipity elements; Cartesian, axisymmetric and spherical
       coordinates.
   * - Time
     - The :math:`\theta` family (forward and backward Euler, Crank-Nicolson)
       with fixed, error-controlled and iteration-controlled step sizes.
   * - Nonlinear solvers
     - Newton with exact AD Jacobians, direct (Picard) iteration with
       relaxation, load stepping.
   * - Linear solvers
     - Sparse LU; BiCGSTAB, GMRES and CG with ILU(0), ILUT or Jacobi
       preconditioning, chosen automatically; a distributed solver with a
       two-level overlapping Schwarz preconditioner.
   * - Parallelism
     - OpenMP threads in assembly; MPI across processes.
   * - Meshes
     - Built-in generators; every format meshio reads; local refinement of
       triangular meshes; uniform refinement of all linear types.
   * - Verification
     - Reddy's book examples, analytical solutions, OpenFOAM cross-checks, and
       a method-of-manufactured-solutions study of every method and element.
   * - Interfaces
     - Python API, YAML input files, command-line driver.

How far it is from MOOSE and COMSOL
-----------------------------------

The comparison below is between dualmesh and MOOSE [MOOSE2025]_, an open-source
framework built on libMesh [libMesh2006]_ and PETSc, and COMSOL Multiphysics, a
commercial package.  Its purpose is to say plainly what the gap is, so that a
reader can judge whether dualmesh fits a problem.

**Where dualmesh is of the same kind.**  The architecture is the one MOOSE
uses: registered objects with validated parameters, a canonical residual form,
monolithic coupling, and Jacobians by automatic differentiation.  A coupled
problem is set up in dualmesh as it would be in MOOSE, by listing variables,
kernels, materials and boundary conditions.

**Where dualmesh offers something the others do not.**  The dual mesh control
domain method, and the ability to solve one problem description by four
discretisations and compare them directly, with the same assembly, the same
solvers and the same verification.  This is the purpose dualmesh was written
for.

**Where the gap is large**, in decreasing order of how much it limits the
problems that can be solved:

1. **Breadth of physics.**  dualmesh has thirty-five objects.  MOOSE's physics
   modules and COMSOL's add-on modules cover, among much else, finite-strain
   solid mechanics with plasticity and contact, turbulent flow, porous media,
   phase field, electromagnetics and chemical reactions.  In dualmesh the solid
   mechanics is small-strain and linear (apart from the von Kármán terms of
   beams and plates), there is no contact, and the flow is laminar and
   incompressible.

2. **Incompressible flow formulation.**  The penalty method is simple and
   robust on the node-based methods, but it requires reduced integration, it
   makes the linear systems ill-conditioned, and it cannot be used with the
   cell-centred finite volume method.  A mixed or stabilised finite element
   formulation, and a pressure-velocity coupling for the finite volume
   methods, are what the other codes offer.

3. **Scale.**  MOOSE distributes the mesh as well as the unknowns, and solves
   through PETSc with algebraic multigrid, field-split and scalable direct
   solvers; it runs on very large parallel machines.  dualmesh replicates the
   mesh on every process and has its own Krylov solvers and Schwarz
   preconditioner.  Its iteration counts do not grow with the number of
   processes, but its memory and its setup do, and the cell-centred method is
   not yet distributed.  It is a code for workstations and small clusters.

4. **Coupling across meshes and time scales.**  Every field of a dualmesh
   problem lives on one mesh and advances with one time step.  MOOSE's
   MultiApps couple separate applications on different meshes and time scales
   with transfers between them; COMSOL couples physics on different domains
   and dimensions.

5. **Time integration and analysis types.**  dualmesh has the :math:`\theta`
   family only, and no eigenvalue, frequency-domain or optimisation solvers.

6. **Geometry and user interface.**  COMSOL provides CAD, meshing and a
   graphical interface; MOOSE provides input-file syntax checking and a
   graphical front end.  dualmesh generates simple meshes itself and reads
   everything else (for instance from Gmsh) through meshio, and is driven from
   Python or a YAML file.

In short, dualmesh is a multiphysics framework in architecture and in the way
problems are coupled and solved, and it is verified to the same standard as its
larger relatives; it is far narrower in the physics it ships and it is not
built for very large parallel runs.  Items 2 and 3 are the ones that would most
change what it can do: a pressure-velocity formulation of flow, and a PETSc
backend for the linear algebra and the distributed mesh.
