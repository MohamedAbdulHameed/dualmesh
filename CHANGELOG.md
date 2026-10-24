# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project uses
[semantic versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Advection schemes.** The heat that a `nonisothermal_flow` coupling carries
  is stabilised by default: the streamline upwind Petrov-Galerkin term (SUPG,
  the new `heat_streamline_stabilization`) with FEM, DMCDM and HFVM, and the
  limited face flux with ZFVM. `advection_scheme` of `heat_transfer` chooses
  `"central"`, `"upwind"`, `"linear_upwind"`, `"limited"` or `"SUPG"` where the
  case calls for it. The inertia of a flow takes its scheme from the method,
  with no option: SUPG with FEM, DMCDM and HFVM, limited with ZFVM, which
  measured the closest on the cavity at Re = 100 and 1000, the Kovasznay flow
  and natural convection. The object-level kernels `advection`,
  `heat_convection` and `convective_inertia` take `advection_scheme`. The face
  schemes are `"upwind"`, `"linear_upwind"` (the upstream value corrected by
  the upstream gradient) and `"limited"` (blended towards the central value by
  the van Albada limiter), on every control volume method: DMCDM and HFVM
  reconstruct the gradient at a node by least squares, with the exact Jacobian
  (not yet on a distributed mesh). The heat and the momentum carry the
  velocity whose flux the mass equation conserves (v - tau r_m through the
  faces, as OpenFOAM carries phi), so they are conserved exactly, the
  temperature does not depend on its scale, and upwind is bounded. On a
  boundary layer at a cell Péclet number of 5, upwind and limited stay in
  [0, 1] where central reaches -1.5 with ZFVM and -0.43 with the node methods,
  limited converges at order 1.7 to 2.05, and upwind at order 0.9. On the
  skew jump of Brooks and Hughes with a held outflow, the temperature of
  Galerkin reaches 409 and that of SUPG 1.6. The SUPG residual is the sum of
  the sources of every other kernel of the temperature.

- **Block preconditioners of a flow.** `preconditioner="lsc"` (the least
  squares commutator of Elman et al. 2006) and `"pcd"` (the pressure
  convection-diffusion preconditioner of Kay, Loghin and Wathen 2002, in the
  form of Elman and Tuminaro 2009, with the Robin condition at an inflow),
  beside `"pressure_mass"`. They approximate the Schur complement of the
  pressure in the block upper-triangular preconditioner of the saddle point
  system: `lsc` from the blocks of the matrix, `pcd` from the operator of the
  momentum equation on the pressure nodes, which dualmesh assembles at the
  current velocity and time step. For the stabilised equal-order element the
  pressure Laplacian of both takes the element stabilisation of Elman et al.
  2007, a hundred times larger in `pcd`. They run in the built-in flexible GMRES with exact block solves, and
  in PETSc's Schur complement field split with one V-cycle of BoomerAMG per
  block, on one process or many. The PETSc version with exact block solves
  takes the iterations of the built-in one. On the lid-driven cavity `lsc`
  takes 10 to 30 iterations per Newton step for Taylor-Hood and 17 to 70 for
  the equal-order element of `fem` and `dmcdm` up to Re = 400, and `pcd` 17
  to 79 (the Stokes start included). The counts grow by at most 73 per cent over two refinements of the
  mesh. On a
  three-dimensional Taylor-Hood cavity on two processes `lsc` is about twice
  as fast as MUMPS from 37 000 unknowns on (23 s against 44 s at 144 000).
  `automatic` takes `lsc` for a flow too large to factorise, in serial and in
  parallel, with the direct solver when it does not converge in 500
  iterations. Before, the distributed `automatic` took BoomerAMG on the whole
  equal-order system (29.9 s against 8.1 s for `lsc` on the 62 500-unknown
  three-dimensional cavity at Re = 100), and the serial one tried multigrid
  and ILU(0) on a large equal-order flow before the direct solver.
- **Split by physics.** `preconditioner="physics_blocks"` solves the blocks
  of the physics of a coupled problem one after the other (the displacements
  of a solid together, every other variable alone, in the order in which the
  physics were added), each with one V-cycle of algebraic multigrid, inside
  flexible GMRES: 14 to 18 iterations on a thermoelastic plate from 2000 to
  32 000 elements, in the built-in solver and in PETSc. The distributed
  `automatic` takes it for a solid coupled to other physics.
- **Rigid body modes in PETSc.** Every PETSc solve gives the near null space
  of the problem (the rigid body modes of a solid, the constant of every other
  variable) to PETSc's smoothed aggregation multigrid, which a distributed
  problem selects with `preconditioner="amg"` and takes by default for a
  solid: 51 to 54 iterations on a three-dimensional cantilever on two
  processes against 78 to 82 for BoomerAMG.
- **Case files.** The physics level of a problem as data, in YAML, run by
  `dualmesh run case.yaml` or `dualmesh.run_case`: the mesh (a generator with
  its keywords, or a mesh file), the method and coordinates, the physics keyed
  by the name a script gives their variable with their conditions as text,
  the couplings naming physics by key, the study and the output. It has the
  same names as the Python input and nothing else, every name is checked with
  the closest valid one before any work is done, and numbers follow YAML 1.2.
  `examples/heated_plate.yaml` builds the same problem as its Python script,
  field for field. PyYAML is the optional dependency `dualmesh[yaml]`.
- **Adjoint sensitivities.** `Problem.sensitivities(objective, parameters)`
  gives d(objective)/d(parameter) of a converged solution for every named
  parameter at the cost of one adjoint solve plus one residual assembly per
  parameter. The objective is the integral of a variable over the domain (the
  variable's name) or over one side set
  (`{"variable": ..., "boundary": ...}`). The parameters are settable
  constants registered with `Problem.add_parameter(name, value)` and read by
  name wherever a coefficient, a source or a boundary value takes a function
  (`Problem.set_parameter` assigns one between solves, and
  `Problem.boundary_integral` gives the value of a boundary objective). The
  adjoint solves with the exact Jacobian of the converged solution and the
  same linear solvers as the primal, and dR/dp is a central difference of the
  residual at the converged solution, so no parameter is solved for.
  Verified against the central difference of whole re-solves, on linear and
  nonlinear conduction, for every method, for the domain and the boundary
  objective, and for the value of a prescribed temperature. The result
  (`SensitivityResult`) follows the result protocol. Periodic boundary
  conditions, stateful material histories, distributed problems and a
  solution that is not the converged steady solution of the model as it
  stands (no solve yet, a change of the model or of a parameter since, a
  transient or eigenvalue study) are not differentiated and are refused.
- **Gradient-enhanced Gaussian processes.** `uq.GaussianProcess().fit(x, y,
  gradients=g)` conditions the surrogate of a scalar output on its values and
  its gradients together, for example the gradients of
  `Problem.sensitivities`. The covariances of the gradients are the
  derivatives of the kernel in closed form, for all three kernels (the Matern
  3/2 process is once mean-square differentiable), checked against central
  differences. The length scales start from the fit to the values and then
  maximise the likelihood of the values and gradients together. On a smooth
  function of two inputs with 16 runs, the error of the surrogate falls from
  0.135 to 0.008 (Matern 5/2) and from 0.059 to 0.0018 (squared exponential).
- The word `symmetry` on a beam end: it holds the axial displacement and the
  slope (`phi` in the displacement form), and the moment stays free. It is the
  centre of a half beam.
- `small_strain_stress` takes `plane_stress`, and `solid_mechanics` in plane
  stress reads its moduli from the property objects as in the other
  formulations.
- **Surface meshes: two-dimensional elements in three-dimensional space**,
  the meshes of shells. A mesh has an element dimension (`dimension`) and a
  space dimension (`space_dimension`). A surface element is mapped by its
  3 x 2 Jacobian, measured by sqrt(det(J^T J)), has tangential gradients and
  edge conormals t x n, and is oriented consistently with its neighbours
  (outward on a closed surface, and a surface that cannot be oriented is
  refused). Curves in the plane or in space work the same way. The finite
  element and dual mesh control domain methods run on surfaces, in serial
  and distributed: a tilted rectangle reproduces the flat solution to
  round-off, and the Laplace-Beltrami problem on the sphere converges at
  second order. The cell-centred finite volume method, flow and the
  continuum formulations of `solid_mechanics` refuse surfaces, and shells come
  next. `mesh_from_arrays` and `read_mesh` place a mesh in the space its
  coordinates use.
- **Surface generators.** `generate_cylinder_surface_mesh` (radius, length,
  `arc_angle` in degrees; edges `near_curved_edge`, `far_curved_edge`,
  `left_straight_edge`, `right_straight_edge`),
  `generate_spherical_shell_mesh` (`polar_angle` and `azimuth` in degrees;
  caps closed at a pole are butterflies of quadrilaterals, with no collapsed
  element; edges `upper_circular_edge`, `lower_circular_edge`,
  `start_meridian_edge`, `end_meridian_edge`) and `generate_surface_mesh`,
  which takes a surface written as equations with its definitions as
  keywords: a height surface `"z = f(x, y)"`, a parametric surface in `u`
  and `v` (seams merged), `f(x, y, z) = c` seen from a `center` (the sphere's
  quadrilaterals moved along their rays onto the surface), and any
  `f(x, y, z) = c` cut out of a `box` (quadrilaterals by surface nets, closed
  along the cut by a net of the box face, then projected and smoothed along
  the surface). Every node lies on the exact surface, quadratic elements
  included.
- `num_elements=[...]` in `generate_rectangle_mesh` and `generate_box_mesh`,
  beside the per-axis counts.
- **A performance regression gate** (`.github/workflows/performance.yml`):
  `scripts/benchmark.py --quick --only assembly` times one residual and
  Jacobian assembly of every method for three physics, weekly, and
  `scripts/check_performance.py` compares the times with budgets of about ten
  times the calibration machine. A regression of an order of magnitude (the
  class that once made copying the whole derivative array of an ADReal the
  dominant cost of assembly) fails the run, and the noise of a shared runner does
  not. The benchmark itself follows the physics level as it is now (the
  variables of `solid_mechanics` and `fluid_dynamics` had been renamed under
  it).
- **The software bill of materials of a release** (the `sbom` job of the
  Wheels workflow): the SBOM action of Anchore, pinned by its commit as every
  action is, scans the source distribution, and its SPDX document is kept
  with the wheels and the sdist as an artifact of the tag.
- The weekly mutation test also mutates `uq/distributions.py`, against its
  tests of the moments, the quantiles and the truncation.
- **An inverted or degenerate element is refused.** `initialize()` maps the
  centroid of every element and the points half way from it to the nodes
  (inside the element, where the integration points lie, and away from the
  apex of a pyramid, whose map is degenerate there by construction). It names
  the first element whose Jacobian determinant is negative, or zero, by its
  index and block, instead of integrating it with a positive measure and
  giving a wrong answer silently. The generators and `read_mesh` fix the orientation
  themselves, so this is reached by a mesh built node by node. Surfaces and
  curves in space are exempt (their measure is a square root, and the side of
  their normal is a convention that `fix_orientation` sets).
- `Mesh.fix_orientation(preferred=...)`: one normal per element, shape
  `(num_elements, 3)`. Each connected part of a surface takes the side that
  agrees best with these normals, weighted by the element areas.
- `read_mesh` names the element blocks after the physical names of the file
  (`mesh.block_names()`).

- **Equations written as they are printed.** `general_form_PDE` takes
  `variables` and `equations`, e.g.
  `equations="rho*cp*dT/dt = div(k*grad(T)) + q"`, with `du/dt`, `du/dx`
  (or `grad_x(u)`), `grad`, `div`, `laplacian`, `curl`, `dot`, `cross`,
  `norm`, vectors `[a, b]`, tensors (`I`, `transpose`, `trace`, `det`,
  `inverse`, `sym`, `skew`, `dev`, `ddot`, `outer`) and vector fields
  (`variables={"U": "vector"}`, components `U_x`, `U_y`). The equations are
  reduced symbolically in the C++ core (`reduceEquations`) to the flux, the
  source and the time coefficient of each scalar unknown, with the chain
  rule for the derivatives of expressions, and matched to the fields they
  govern. Keywords that are not parameters are definitions: a number is a
  named constant, a text a named expression. An equation of a scalar field
  may instead be given by its coefficients `d`, `c`, `alpha`, `gamma`,
  `beta`, `a` and `f`.
- **Boundary conditions written as text**, for every physics: a variable on
  the left prescribes its value (`"temperature = 300"`, `"u = 0"`), and the
  name of the flux of the physics prescribes the flux through the boundary
  along the outward normal: `heat_flux` (the heat leaving the body),
  `traction`, `total_force` and `pressure` of a solid, `normal_flux` of
  `general_form_PDE`, which also takes the flux written out
  (`"-k*du/dn = h*(u - 1)"`, checked against the flux of the equation). The
  right side may depend on the fields, so convection and radiation are
  written as they are printed.
- **The user guide page "From paper to code"**: the language of the
  equations and of their boundary conditions, and twelve worked examples,
  from Poisson's equation to the Cahn-Hilliard equation, two-group
  criticality and linear elasticity of a vector field, each run by the test
  suite.

- **The projection time integration of incompressible flow**
  (`incompressible_flow(time_integration="projection", time_order=k)`), the
  splitting of Nek5000 and nekRS: BDFk and EXTk with coefficients for
  variable steps, a pressure Poisson equation with the curl-curl boundary
  term, solved by the conjugate gradient method with BoomerAMG from the
  projection of the earlier pressures, and Helmholtz equations for the
  velocity, solved by the conjugate gradient method with a Jacobi
  preconditioner. For equal-order elements the weight of the divergence in
  the pressure equation is capped by the stabilization parameter of the
  pressure-velocity formulation, which keeps the accuracy at small steps.
  The finite element, dual mesh control domain and vertex-centered finite
  volume methods, serial and distributed, with periodic boundaries.
  Checked at second order in space on the Taylor-Green vortex, at the order
  of the backward difference in time on an exactly represented flow, and
  against the serial run on one to four processes.
- **A Courant number step controller**: `solve_transient(time_stepper="cfl",
  courant_number=0.5)`.
- **Reactions of the projection time integration**: the forces on boundaries
  with a prescribed velocity, from its momentum equations at every step, so
  that `total_reaction` gives drag and lift.
- **The Laplacian form of the viscous term**: `incompressible_flow(viscous_form=
  "laplacian")`, whose natural boundary condition is the do-nothing outflow of
  the DFG benchmarks and of Nek5000.
- **Benchmarks** (docs/benchmarks, one page each, with their scripts in
  verification/benchmarks): F1, the DFG flow around a cylinder at Re = 20
  (drag within 0.009 % with equal-order elements, every printed digit with
  Taylor-Hood); F2, the DFG vortex shedding at Re = 100 with the projection
  time integration on four processes (drag, lift and Strouhal number within
  0.5 %); F4, the Kovasznay flow (every method at its order); F5, the
  lid-driven cavity at Re = 1000 against Erturk et al. (2005).
- **A scaling study of the projection time integration**
  (`verification/benchmarks/scaling/run_projection_scaling.py`): strong and
  weak scaling of every node method on one to four processes, in the measures
  of Nek5000 and nekRS.
- A nonlinear solve whose residual or Jacobian is no longer finite stops with a
  message that names divergence or a non-finite property, rather than failing
  in the linear solver.
- **Transient monolithic flow**: `incompressible_flow` adds the inertia
  rho dv/dt, which a transient solve includes and a steady solve leaves out.
- `dualmesh_benchmark projection` times the steps of the projection time
  integration on any number of processes.
- **The cell-centered finite volume method in distributed problems**: every
  part holds two layers of ghost cells, sent by PETSc's DMPlexDistribute, and
  every face is integrated by one process. Cells and boundary faces are
  numbered in the order of the elements, so checkpoints do not depend on the
  partition. Checked against the serial solution at every cell and boundary
  face on one to six processes, with the serial number of Newton iterations,
  across a gap, and through a restart on another partition.
- **Gap conditions in distributed problems**: an interface condition pairs
  its primary and secondary sides across processes; a process that holds a
  primary side reads the secondary elements it lacks as ghost elements, which
  it does not integrate. Checked for `gap_heat_transfer` between non-matching
  meshes on one to six processes.
- **Checkpoints and restarts**: `dm.Output(checkpoint_interval=...)` writes
  the time, the step, the solution, the history of the material and the
  history of the post-processors to one file in PETSc's binary format, in
  the order of the global numbers, and `solve_transient(restart=...)`
  continues the run on any number of processes. A run restarted on another
  partition ends where an uninterrupted run ends, to round-off.
- In distributed transient runs, the history of the material is committed
  at every step and the post-processors and the step callback are evaluated
  at every step, as in serial runs.
- **Post-processors and reactions in distributed problems**: every
  post-processor type and `total_reaction` give the serial value on any
  number of processes.
- **Periodic boundary conditions** (`periodic_boundary_condition`, a new
  category of objects, the constraints): the field on a secondary boundary is
  the field on a primary boundary carried over by a translation, for every
  node-based method, in one, two or three directions, serially and on any
  number of processes. Every method converges at second order on manufactured
  solutions periodic in one direction and in two.
- **A distributed core built on PETSc, in which no process needs the whole
  mesh.** A build with MPI now requires PETSc. The values at nodes that
  several processes hold are exchanged through PETSc's star forests (PetscSF),
  which find the owner of every node from the global node numbers alone. The
  distributed solver takes the part of the mesh of each process; a whole mesh
  is read on the first process only, and PETSc's DMPlex partitions it
  (PT-Scotch or ParMETIS) and moves every element, of any type, to its
  process. `partition_mesh` uses the same partitioners. The exchanges of
  shared values go through PETSc's star forests or through gslib, the
  gather-scatter library of Nek5000, chosen with `gather_scatter`. The boundary node sets are completed across processes,
  and the areas of the side sets are summed over them. The linear systems are
  one distributed PETSc matrix. By default a flow is factorised by MUMPS
  where that is inexpensive and split by the least squares commutator
  otherwise, a solid is solved with GAMG and its rigid body modes, a solid
  coupled to other physics with the split by physics, and any other system
  with GMRES and hypre BoomerAMG (15 iterations on one to four processes for
  a 48 x 48 Poisson problem), with MUMPS when an iteration fails.
- **`dualmesh.materials`**: functions of the state that return material
  properties in SI units. `materials.water` gives compressed liquid water
  (IAPWS-IF97 region 1, with the IAPWS viscosity and thermal conductivity),
  and `materials.gas` gives the thermal conductivity of helium, argon,
  krypton, xenon, hydrogen and nitrogen at low density and of their mixtures,
  and the temperature jump distance at a wall.
- **General partial differential equations.** `general_form_PDE` solves a
  system $`d \, \partial u / \partial t + \nabla \cdot \boldsymbol{\Gamma} = f`$ whose flux $`\boldsymbol{\Gamma}`$ and source $f$ are
  expressions of all the fields, the components of their gradients
  (`grad_x(u)`, `grad_y(u)`, `grad_z(u)`), x, y, z, t and named constants.
  The expressions are compiled and differentiated automatically, so the
  Jacobian is exact and the assembly runs on every thread and every process
  with every method. `parsed_kernel` is the object of one such term.
- **Eigenvalue study of any problem.** `Problem.solve_eigenvalue` returns
  the eigenvalues closest to `near` and their modes for every problem.
  Every field varies in time as
  u_hat exp(-lambda t), so the time derivatives define the eigenvalue, and
  the expressions may use the symbol `eigenvalue`, at most quadratically.
  The operators come from exact assemblies, and shift-invert Arnoldi
  iteration solves the linear or the companion quadratic problem. Complex
  eigenvalues are reported with both parts. `examples/reactor_criticality.py`
  and the tutorial on reactor criticality write multigroup neutron diffusion
  with `coefficient_form_PDE` and solve the IAEA 2D PWR benchmark this way.
- `coefficient_form_PDE` solves systems: every input is a dict keyed by the
  field of the equation, and a coefficient couples to other fields with a
  dict field name -> value. Every coefficient and the source may be an
  expression of the fields, their gradients and the position together.
- The `flux` of `Neumann_boundary_condition` and the `flux`,
  `transfer_coefficient` and `ambient_value` of `Robin_boundary_condition`
  may be expressions of the fields and their gradients.
- `gas_gap_heat_transfer` takes an `accommodation_coefficient`.
- The parsed creep rate of `small_strain_stress` reads `grain_size`, which is
  required only when the expression uses it.

- `uq.sobol` and `uq.calibrate` take `output_transform={"name": "log"}`:
  the Gaussian process models the logarithm of a positive output, which
  suits an output that changes by factors (a creep strain). The Sobol'
  indices are those of the output itself, and the calibration of that output
  is on the logarithmic scale (its noise is the standard deviation of the
  logarithm, about the relative error).
- `uq.calibrate` calibrates several experiments together: `model` is a dict
  of experiment name -> function, the experiments share the inputs their
  functions have in common, and the log-likelihoods add up. Verified against
  the normal posterior of two linear experiments.
- `uq.sobol` on a Gaussian process also gives the indices of the global
  process (Marrel et al. 2009, Eq. 12), with their intervals:
  `first_order_global_process`, `total_global_process` and the columns of the
  same names in the table `indices`.
- **Output design.** Every study takes `report="full"` (the default: the
  inputs with the defaults marked, the progress and the result),
  `report="summary"` or `report="none"`: `Problem.solve`,
  `Problem.solve_transient`, `Problem.solve_eigenvalue`, `uq.propagate`,
  `uq.sobol`, `uq.calibrate`, `ManufacturedSolution.convergence_study` and
  `solve_with_adaptive_refinement`. A study inside another study prints
  nothing. The `dm.Output` group (directory, file base, output times or
  interval, fields, formats `vtu` and `csv`) writes the fields of a solve, a
  transient solve or an eigenvalue study. The time steps land on the output
  times, and a transient study writes a ParaView collection (`.pvd`). Each
  physics gives the SI units of its variables, which the CSV headers and the
  VTU data arrays carry.

- **Physics level**: `Problem.add_physics(type, name, ...)` adds a set of
  equations named in physical terms: `heat_transfer`, `solid_mechanics`,
  `incompressible_flow`, `beam`, `plate` and `circular_plate`. A physics
  creates its variables and generates its kernels and materials. Its boundary
  conditions and extra terms (`physics.add_boundary_condition`,
  `physics.add_kernel`) take the type names and parameters of the objects,
  with the variables, the components and the thickness filled in, and a
  vector value (a traction, a total force, a prescribed displacement or
  velocity) given as a list with one entry per component. A property that is
  not given is read from the materials.
  `Problem.add_coupling(type, name, ...)` couples two physics:
  `thermal_expansion` and `nonisothermal_flow` (heat carried by a flow, and
  the Boussinesq buoyancy). `dualmesh list --category physics` and
  `dualmesh describe <physics>` document them, and the syntax reference has a
  page for each.
- `Problem.element_integrals(variable)` and `Problem.element_volumes()`, and
  `Mesh.block_names()`.
- **Tables.** `dm.Table` is the one form of every tabular result: it prints
  in aligned columns, writes a CSV file with one header line whose column
  names carry their units (for example `temperature (K)`), and gives its
  columns as arrays and as a pandas DataFrame. Every result has `tables` and
  `write_csv(path, table=None)`.
- A finite-difference check of the Jacobian (`tests/python/test_jacobian.py`)
  and, in the continuous integration, builds with AddressSanitizer and
  UndefinedBehaviorSanitizer, coverage reports, clang-tidy, CodeQL, a weekly
  mutation test, actions pinned to commit hashes, read-only job
  permissions and a software bill of materials of each release. The release
  script refuses a workflow with an action that is not pinned to a commit.
- **`coefficient_form_PDE`**: a physics for a scalar equation written by its
  coefficients, $`d \, \partial u / \partial t + \nabla \cdot (-c \nabla u - \boldsymbol{\alpha} u) + \boldsymbol{\beta} \cdot \nabla u + a u = f`$,
  with a diffusion coefficient that is a constant, an expression of position
  and time or of the variable, a constant tensor, or the property
  `diffusion_coefficient`, and an absorption coefficient that may depend on
  the variable.
- `dualmesh list` and `dualmesh describe` answer an unknown name with the
  closest one.
- `scripts/audit_names.py` checks every public name and parameter
  declaration against the naming rules, and the tests run it.
- Every result provides `summary()`, `to_dict()`, `write_json(path)` and
  `write_csv(path)`: the `SolveResult` of a solve, the new `AdaptivityResult`
  of `solve_with_adaptive_refinement` (which returned the tuple
  `(problem, mesh)`), the `ConvergenceResult` of a manufactured solution, and
  the `Runs`, `SobolIndices` and `Posterior` of the uncertainty studies.
- The `reaction` kernel takes its coefficient from a property
  (`coefficient_property`).
- `uq.sobol` takes the runs of `uq.propagate` to train its Gaussian process
  surrogate (`training`), as `uq.calibrate` does, so that one set of runs
  serves the propagation, the indices and the calibration. With the nested
  scrambled Sobol' design of `propagate(method="sobol")`, a larger study
  reuses the runs of a smaller one from its store.
- **`symmetry_boundary_condition`**: a plane of symmetry, on which the
  displacement (or velocity) normal to the plane is zero.
- **Distributed problems in `Problem`**: a `Problem` run under `mpirun` on
  several processes splits itself among them (`distributed` and
  `partitioner` of the constructor), and its `solve` and `solve_transient`
  solve it with PETSc. `rank`, `num_ranks`,
  `num_owned_dofs`, `num_global_dofs` and `gathered_values` describe a
  distributed problem.
- **Keyword checks**: a misspelled parameter of a physics, a coupling or an
  input group is answered with the closest name and the list of accepted
  parameters.

- **Algebraic multigrid preconditioner** (`preconditioner="amg"`): smoothed
  aggregation with the rigid body modes of the elastic materials and the
  constants of the other variables as near null space, Chebyshev smoothing
  and a factorised coarsest level, for the conjugate gradient method and
  BiCGSTAB. `linear_solver="automatic"` uses it for the systems it does not
  factorise (three-dimensional systems above a few thousand unknowns), with
  BiCGSTAB and ILU(0) and then the direct solver as fallbacks. The solve of
  the wrench tutorial takes 1.8 s on 56 787 unknowns, and a solve of
  340 000 unknowns about 30 s, in two threads. An iteration that reaches the
  round-off level of a stiff system is accepted. The option
  `amg_strength_threshold` sets the aggregation.
- **Reuse of factorisations**: the direct solver and the multigrid
  hierarchy are kept while the matrix does not change, so the second Newton
  iteration of a linear problem costs one back substitution.
- **One boundary condition on several variables**: every boundary condition
  accepts a `variables` list and is created once per variable.
  `fixed_constraint` clamps the displacements it lists.
- **`total_force`** of `traction_boundary_condition`: a load given as the
  total force on a side set, spread uniformly over its area, in Cartesian,
  plane and axisymmetric problems.
- **Side set tools**: `sideset_summary` (faces, nodes, area and extent of
  every side set), `write_sidesets` (the boundary with one field per side
  set, for ParaView), `Mesh.sideset_measure`, `Problem.boundary_measure`, the
  command `dualmesh mesh <mesh file> --sidesets <file.vtu>`.
- **Uncertainty quantification and sensitivity analysis** (`dualmesh.uq`).
  Input distributions `Normal`, `LogNormal`,
  `Uniform` and `LogUniform`, with optional truncation. `propagate` samples a
  model by Latin hypercube, scrambled Sobol' or Monte Carlo designs and
  returns the statistics, bootstrap intervals, Wilks tolerance limits and the
  Pearson, Spearman and partial rank correlation coefficients. `sobol`
  computes the first-order and total Sobol' indices (Saltelli 2010, Jansen
  1999) with bootstrap intervals, either directly or on a Gaussian process
  surrogate. `GaussianProcess` is a kriging surrogate with a Matern 5/2, Matern 3/2 or
  squared exponential kernel, maximum likelihood hyperparameters, a fitted
  nugget, principal components for vector outputs and closed-form
  leave-one-out validation. `calibrate` performs Bayesian calibration on a
  surrogate, with the surrogate variance and an optional model discrepancy
  in the likelihood, and an affine-invariant ensemble sampler or adaptive
  Metropolis, and reports the split R-hat, the effective sample size and the
  contraction of every parameter. Runs are executed in parallel processes
  and stored, so that an interrupted study resumes. The results write their
  statistics to JSON files (`write_json`).
- **Wrench example** (`examples/wrench`, tutorial "Boundary conditions on an
  imported mesh"): a three-dimensional steel wrench on a Gmsh mesh, with the
  boundaries named by physical groups or by geometric predicates, verified against the statics of the wrench and beam theory.
- **`property_scaling` property object** and the `scaled_properties` and
  `property_factors` parameters of every property object, which multiply
  named properties by constant factors.

### Changed

- **A steady flow at rest starts from the Stokes solution.** Newton's method
  starts from the same problem without the inertia, solved at the first load
  factor, when the velocity is zero wherever no condition prescribes it. On a
  channel at Re = 100 with a varying inflow, the cell-centred method diverged
  from rest ("structurally singular" after about 40 iterations) and now
  converges in 3 to 4 iterations. A start set by the user is kept. The counts
  of a solve include the Stokes iterations.
- **A diverging Newton iteration stops** when the residual reaches a million
  times its first value, with an error that says so and names the remedies.
- **The stabilisation parameter tau has the factor 9 on its viscous (and
  conductive) term**, as MOOSE's navier_stokes module takes it, for the
  stabilised flow of every method, the SUPG of the heat, and the cap of the
  projection solver. Its viscous limit h^2 / (12 nu) is that of the optimal
  parameter of linear elements; Tezduyar's form gave h^2 / (4 nu), three times
  too much. Measured against a fine Taylor-Hood cavity at Re = 100 the error of
  the centreline velocities halves for every method, the Kovasznay errors fall
  by about a quarter, the Nusselt number at Ra = 1e5 is within 0.5 % (1.1 %
  before), and the SUPG boundary layer has the error 0.0024 (0.143 before).
  The pressure of zfvm on triangles converges at order 1.06 (1.33 before).
- **Abbreviations in the input are written in capitals and proper names with a
  capital**, as the naming rule of the input requires: `advection_scheme="SUPG"`,
  the methods `"FEM"`, `"DMCDM"`, `"HFVM"`, `"ZFVM"` (`Problem.method` holds this
  spelling), the linear solvers `"LU"`, `"CG"`, `"GMRES"`, `"BiCGStab"`,
  `"PETSc"`, the preconditioners `"ILU"`, `"ILUT"`, `"AMG"`, `"Jacobi"`, `"LSC"`,
  `"PCD"`, and the nonlinear solvers `"Newton"`, `"Picard"`. The lower-case
  spelling is accepted everywhere.
- **`line_search` takes `"automatic"`, the new default**: backtracking when a
  kernel uses the limited advection scheme, whose limiter has kinks on which
  Newton's method cycled (on cells of alternating size it never converged),
  and no line search otherwise, as before.

- A distributed problem accepts `preconditioner`: the block preconditioners
  of a flow, `physics_blocks` and `amg`. The preconditioners of the serial
  Krylov methods (`ilu`, `ilut`, `jacobi`, `none`) are refused there with the
  PETSc option that replaces them. An unknown preconditioner is refused
  with the closest valid name. With the default linear solver it was ignored
  silently.
- The pressure mass matrix preconditioner of a flow is named
  `preconditioner="pressure_mass"`, like `lsc` and `pcd`.
- **One nonlinear driver and one transient driver** serve the serial and the
  distributed problem. A `Parallel` interface of the problem (the global sum,
  norm and any, the residual entries shared by processes, the prescribed
  values, the linear solve, the checkpoint numbering and the field files) has
  the serial identity by default, and a `DistributedProblem` implements it
  for its local problem, whose own drivers then solve the whole problem. The
  two copies of each driver had drifted apart: the distributed solve now has
  the backtracking line search, and a step of a stateful material starts from
  its committed history by an explicit flag of the step instead of a
  comparison of the solution vectors.
- The bounds of an axis of `generate_rectangle_mesh` and `generate_box_mesh`
  are also given as a pair, `x=[0, 0.1]`, in place of `x_min` and `x_max`
  (both together are refused), and so in a case file. Coordinates given with
  bounds are refused, where the bounds were ignored.
- A free end of a mixed beam or of a classical circular plate holds no
  moment by itself: `solid_mechanics` holds the moment at zero on every end
  that is not clamped (`fixed`), not a plane of symmetry and has no moment of
  its own. The natural condition of the moment equation is a zero slope, so an
  end left without a condition was a guided end: a cantilever deflected 0.34
  (Timoshenko) or 0.33 (Euler-Bernoulli) of its exact tip deflection.
- Poisson's ratio is 1/3 by default everywhere it is not given (the property
  `poissons_ratio` when a property object declares it), and the report marks
  it as a default. Young's modulus has no default, and its absence is named.
- A plate or circular plate given by its material needs its `thickness`. The
  thickness is refused in three-dimensional, axisymmetric and beam problems,
  and with section stiffnesses.
- A section given by its stiffnesses takes those of its theory: A and D, and S
  for the Timoshenko beam and the first-order plates (B may be left out). A
  stiffness the theory does not use is refused, and so is a Timoshenko beam
  section given by area and second moment without its `shear_correction`.
- The beam and plate formulations refuse quadratic elements: their reduced
  integration of the shear is right for linear elements only.
- `stress(z)` refuses a height outside the section, and `body_force` needs one
  value per displacement component.
- The structural kernels require `stress_free_temperature` with a temperature
  and their stiffnesses with no material, and default Poisson's ratio to 1/3.
- The units of `transverse_load` and `foundation_modulus` are stated per
  member (plate or beam).
- `elastic_strain` of the neo-Hookean law is the Green-Lagrange strain of the
  elastic part of the deformation, and `volumetric_strain` at finite strain is
  J - 1.
- A property object may skip an output that no object reads. The registry of
  properties knows the outputs that kernels, conditions, expressions and
  post-processors read, and `small_strain_stress` forms the deformation
  gradient and the von Mises stress only for them. Assembly of linear
  elasticity takes the time it took with `linear_elastic_stress`.
- `refined()` carries the side sets over in a time proportional to the mesh:
  1.26 million `Hex8` elements take 7.8 s, against 131 s before, and 1.28
  million `Quad4` elements 2.6 s, against 9.6 s.
- The outward normal of an element side comes from the stored normal of the
  reference side, mapped by the element (the relation of Nanson), in place of
  a test against the element centroid. `write_sidesets` writes through meshio.
- **`general_form_PDE`** takes the equations as text or by their
  coefficients; `flux`, `source`, `time_derivative_coeff`, `constants` and
  `expressions` are removed, and `coefficient_form_PDE` is merged into it.
- **Input without repetition** (sweep of the input for information given
  twice): the object types are named without the category
  (`Dirichlet`, `Neumann`, `Robin`, `point_Dirichlet`, `periodic`,
  `traction`, `pressure`, `symmetry`, `mass_flux`, `radiative_heat_flux`,
  `constant`, `function`, `parsed`; the old names are removed, and a
  misspelled name is answered with the closest registered one); `variable` takes a list, which adds one object per variable;
  the component of a vector equation follows from its variable; every
  coefficient is `*_coeff`; a parameter and its `*_property` twin are one
  parameter that takes a number, a function, an expression or a property;
  lists of names and values are dicts (`properties`, `constants`,
  `creep_constants`, `property_factors`, `stress_free_state`); nodal
  conditions take no quadrature parameters.
- The expansion coefficients name their kind: `vol_expansion_coeff` for
  the Boussinesq buoyancy of a flow, `linear_expansion_coeff` for a solid.
- `time_derivative` takes its coefficient as an expression, which may depend
  on the fields.
- **Three modules**: `heat_transfer`, `solid_mechanics` and `fluid_dynamics`
  (was `incompressible_flow`), with their couplings, and `general_form_PDE`.
  The `beam`, `plate` and `circular_plate` physics are formulations of
  `solid_mechanics` (`formulation="plate"`, `"beam"` with
  `theory="Euler_Bernoulli"` or `"Timoshenko"` and, for Timoshenko,
  `form="mixed"` or `"displacement"`, `"circular_plate"` with
  `theory="first_order"` or `"classical"`), whose section is given by its
  material (`youngs_modulus`, `poissons_ratio` and the `thickness` or the
  `cross_section`) or by its section stiffnesses. The results equal those of
  the former physics. The variables of a plate are `disp_x`, `disp_y`,
  `disp_z`, `phi_x`, `phi_y`; of a beam `disp_x`, `disp_z` and
  `bending_moment` or `phi`; of a circular plate `disp_r`, `disp_z` and
  `phi` or `bending_moment`. The former names are removed.
- **The structural sections are computed from the material**, as MOOSE's
  shells and beams do: given `youngs_modulus` (and `poissons_ratio` with the
  `thickness` or the `cross_section`), the plate, beam and circular-plate
  kernels integrate the material law through the thickness at every
  integration point, at `through_thickness_points` Gauss points. A modulus
  that depends on z (a graded section) or on the fields (E(T)) enters
  exactly, with its derivatives. `thermal_expansion` acts on the structural
  formulations: the thermal strain alpha (T - T_0) gives the thermal force
  and moment of the section. Checked against the closed-form stiffnesses of
  graded sections and against free thermal expansion and the thermal bending
  of a graded cantilever.
- **`solid_mechanics.stress(z=...)`**: the stress at the element centres; for
  a structural formulation at a height above the mid-surface (a number, or
  "top", "mid", "bottom"), from the material law of the section with its
  thermal strain.
- Objects added without a name by one call for several variables no longer
  take a name that such a call already used.
- **Each physics defines its variables**, and names are given only in
  `general_form_PDE`: `temperature` (heat_transfer); `disp_x`, `disp_y`,
  `disp_z` (`disp_r`, `disp_z` axisymmetric; solid_mechanics);
  `velocity_x`, `velocity_y`, `velocity_z` and `pressure`
  (fluid_dynamics). The parameters `temperature`,
  `displacements`, `velocities`, `pressure`, `in_plane_displacements`,
  `rotations`, `axial_displacement` and the like are removed.
- `general_form_PDE` takes its fields as a list whose items are names
  (scalar) or one-entry dicts, `variables=[{"U": "vector"}, "p"]`.
- A boundary condition written as text takes definitions as keywords
  (`"total_force = [0, -F, 0]", F=150.0`); `normal_flux` finds its field by
  the kind of its value; a traction and a pressure may depend on the
  displacements.
- `nonisothermal_flow` reads the density from the flow and the specific heat
  from the heat transfer, and no longer takes them itself;
  `fluid_dynamics` takes `initial_condition`; the in-plane load of a
  plate is one list, `in_plane_load=[f_x, f_y]`.
- **A boundary condition of a physics is written as text**; the type with
  keywords (`heat.add_boundary_condition("Dirichlet", "left", value=40.0)`)
  is refused with a message that shows the text form. The words without
  data stay: `fixed` and `symmetry` of a solid, and `periodic`. The object
  level (`problem.add_boundary_condition("Dirichlet", ...)`) is unchanged.

- **A physics has no name.** `add_physics` takes the type and the parameters
  only; the variable it is assigned to names it. A coupling takes its physics
  as the objects that `add_physics` returned (`heat_transfer=heat`), or none
  when the problem has only one physics of each type it couples.
- The documentation states its problems by themselves rather than by the
  numbers of the textbook examples they came from, and the verification page
  lists each problem with what it is checked against.
- The method `variables()` of a physics is renamed `variable_names()`.
- `gas_gap_heat_transfer` takes the data of the surfaces from the case, with
  no default: the roughnesses, the emissivities and, with a
  `contact_penalty`, the Meyer hardness and the conductivities of the two
  bodies. Every mole fraction of the gas has the default 0.
- The author of the software is Mohamed AbdulHameed, with his ORCID iD, in
  `CITATION.cff`, `pyproject.toml` and the documentation.
- The citation is the software alone (`CITATION.cff`, the README and the
  documentation). Its abstract covers heat transfer, solid mechanics, fluid
  dynamics, general partial differential equations and uncertainty
  quantification.
- Names written out in full: `solve_transient` takes `time_step`,
  `implicitness`, `min_time_step` and `max_time_step` (were `dt`, `theta`,
  `dt_min` and `dt_max`), `solve_particle` takes `implicitness`, the
  uncertainty quantification uses `standard_deviation` (`uq.Normal`, the
  `standard_deviation` methods of the results, `return_standard_deviation`
  of `GaussianProcess.predict`), and `fgm.beam_stiffness` takes
  `poissons_ratio`.
- The Taylor-Hood formulation of `incompressible_flow` is named
  `Taylor_Hood`.
- `linear_solver="automatic"` iterates with algebraic multigrid where it
  used BiCGSTAB with ILU(0).
- The Python code is formatted by `ruff format` with a line length of 100
  characters, which wraps a long call with one argument on each line.

- **Property objects.** The objects that compute named properties at the
  integration points (conductivities, stresses, eigenstrains, cross
  sections) are property objects: they are added with
  `problem.add_property(type, name, ...)` (was `add_material`), their
  category is `property` (`dualmesh list --category property`), and the
  word material is kept for physical substances. Renamed with them:
  `constant_property` (was `generic_constant_material`), `function_property`
  (was `generic_function_material`), `parsed_property` (was
  `parsed_material`, whose `material_property_names` is now
  `coupled_properties`), `PythonProperty` (was `PythonMaterial`) and the
  C++ base class `Property` (was `Material`).

### Removed

- `linear_elastic_stress`, merged into `small_strain_stress`: one isotropic
  stress object for every formulation. The thermal strain is an eigenstrain of
  `thermal_expansion_eigenstrain`. The orthotropic option (`stiffness_c11` to
  `stiffness_c66`), which was untested and ignored in axisymmetric problems,
  is removed.
- `coefficient_form_PDE` (merged into `general_form_PDE`), and the objects
  that repeated another one: `heat_flux_boundary_condition` and
  `convective_heat_flux_boundary_condition` (now `Neumann` and `Robin`, or
  the words `heat_flux` and `convective_heat_flux` of `heat_transfer`),
  `fixed_constraint` (`Dirichlet` on the displacements, or the word `fixed`
  of `solid_mechanics`) and the `heat_source` kernel (`body_force`).

- **Publication on PyPI**: the publish job of the Wheels workflow, the PyPI
  gates of `scripts/release.py` and the installation from PyPI. dualmesh is
  installed from source; the wheels of a release are the artifacts of its
  tag build.
- The solver option `verbose` and the `progress` parameter of the
  uncertainty studies (both replaced by `report`), and the `output_interval`
  and `output_file_base` parameters of `Problem.solve_transient` (replaced by
  `output=dm.Output(...)`).
- The classes `mms.Term`, `mms.Diffusion`, `mms.Advection`, `mms.Reaction`,
  `mms.TimeDerivative`, `mms.LinearElasticity` and `mms.IncompressibleFlow`.
  `mms.ManufacturedSolution` takes the physics of the physics level
  (`study.add_physics(...)`, as for a problem), and the boundaries that get
  the exact flux in place of the exact solution are its `flux_boundaries`.
- The helpers `dualmesh.physics.add_plane_elasticity`,
  `add_incompressible_flow`, `add_boussinesq_buoyancy`, `add_beam`,
  `add_plate` and `add_circular_plate`, replaced by the physics level.
- The class `DistributedProblem`, replaced by `Problem`, which splits itself
  under `mpirun` (or with `distributed=True`).
- YAML input files and the command `dualmesh run`. A problem and an
  uncertainty study are described in a Python script, which runs with
  `python script.py`, or with `mpirun -n 4 python script.py` for a
  distributed problem. The example `examples/bus_bar.yaml` is replaced by
  `examples/bus_bar.py`, the optional dependency `input` (PyYAML) is removed,
  and `dualmesh mesh` reads a mesh file.

### Fixed

- **The theta method with an equation that has no time derivative.** The
  mass equation of an incompressible flow and the equilibrium of a
  quasi-static solid are now taken at the new time with the weight 1 for
  every implicitness, and so are the pressure gradient and the stabilisation
  of a flow, whose residual is that of the discrete momentum equation.
  Before, Crank-Nicolson made the pressure an average of two steps, whose
  error changed sign at every step and grew (5.2e-2 after four steps on a
  flow that backward Euler gives to round-off, now exact to round-off), and
  it evaluated creep at the old time without the material history (a strain
  185 % too large, now exact). An explicit step solves such an equation at
  the new time.

- **The time derivative inside a streamline stabilisation with
  Crank-Nicolson.** `momentum_stabilization` and the new
  `heat_streamline_stabilization` put the time derivative inside a steady
  term, which the theta method weights by theta and evaluates again at the old
  time without it. The derivative now enters with the weight 1. Before, SUPG
  of the energy equation converged at order 0.7 to 1.1 with Crank-Nicolson.
  Backward Euler is unchanged. The explicit step (implicitness 0), which
  cannot weight that derivative, refuses these kernels and the stabilised
  `mass_conservation` of a flow with inertia, and so does the eigenvalue
  study, whose mass operator does not hold that derivative. A stabilised
  Stokes flow has none and keeps both.
- **An error raised inside the threaded assembly** reached Python as a
  RuntimeError, not as the InputError it was.
- **An error of an object named it twice** ("advection 'advection_0':
  advection 'advection_0': ..."). The type and the name now come once.

- A run on several processes could hang when Python's cycle collector freed
  a distributed problem: its MPI, PETSc and gslib objects must be destroyed
  by every process together, and the collector runs whenever each process
  happens to run it (a sequence of the parallel tests hung on two processes).
  A collected distributed problem is now kept until the next collective point
  (a new problem, a solve, or the end of the run), where every process
  collects its cycles and destroys the kept problems in the order of their
  creation.
- `error_norms` of a distributed problem returned the norm of the part of each
  process (too small by sqrt(2) on two processes). The squares now add over
  the processes, so a convergence study gives the same numbers on any number
  of them.
- The error-controlled time step (`time_stepper="error"`) estimated its error
  with the norm of each process, so the processes could disagree on a step
  and wait for each other forever. The estimate uses the global norm.
- A distributed nonlinear solve that stopped on the size of its step kept the
  residual of the iterate before, so its reactions were stale. It assembles
  the residual at the solution, as the serial solve does.
- A distributed solve accepted any nonlinear solver or line search name and
  ignored the backtracking line search. Unknown names are refused, and the
  line search, which only the serial solve has, is refused by name.
- A checkpoint check that failed on one process left the others waiting in
  the next collective call. Every process now stops with the message.
- In a run on several processes every process printed the report of a study
  and wrote the `.pvd` index. The first process does both.
- Numbers outside the range a parameter states are refused when the object is
  made, naming the parameter and the value: a non-positive diffusivity,
  thermal conductivity, viscosity, specific heat, thickness or penalty, and a
  negative density, emissivity, transfer coefficient or foundation modulus.
  A negative diffusivity solved the anti-diffusion problem without a word.
- A mesh without elements, a mesh that reaches x < 0 in axisymmetric or
  spherical coordinates, and a second `Dirichlet` value for the same variable
  on the same boundary are refused (the second value silently replaced the
  first, and the negative radius failed later in the nonlinear solver).
- The method aliases `fvm`, `finite_volume`, `vertex_finite_volume`,
  `finite_element`, `dual_mesh` and `cell_finite_volume` are removed: each
  method has one name, and "finite volume" alone names two methods.
- `mesh_from_arrays` takes the `(element_type, nodes, block)` triples of
  `Mesh.cells()`, so a mesh is rebuilt from its own cells.
- Newton's method accepts the round-off level of a badly scaled system: from
  the third iteration, a residual that no longer falls, after a step of at
  most 1e-8 of the solution, below 64 eps |J| |U|. A thin displacement
  Timoshenko beam (L/h of 1000 and 10000) stalled there with fem and was
  reported as not converged, with its exact answer in hand. The serial and
  the distributed solvers decide it by one rule.
- Two physics on two blocks that share a field (a stepped beam) gave the
  second block no unknowns, and their junction was taken for a free end. A
  shared field now lives on the blocks of every physics that uses it, and the
  ends are those of the field, with the end conditions of all of them.
- The plate formulation refuses triangles, which lock with its one-point
  shear rule (0.04 of the deflection of a thin plate on an 8 x 8 mesh).
- A traction, pressure or gap contact condition took the thickness of the
  first stress_divergence kernel of its displacement, whatever its block. It
  now refuses kernels of different thicknesses, and stress_divergence refuses
  a thickness in a three-dimensional or axisymmetric problem.
- A negative thickness, section dimension or foundation modulus is refused,
  and so is an end condition of a beam added after the first solve, whose
  free ends were derived already.
- `symmetry` on a boundary that does not determine its plane (a single node)
  held disp_x. It is refused.
- `stress(z)` of a beam on some blocks failed on the empty rows of the other
  blocks, which are now NaN rows.
- The object-level `beam_Timoshenko_displacement` described itself as
  integrating the shear at the centre, which only the one-point rule of its
  rotation equation does (the physics adds it so). The description says so.
- `fgm.beam_stiffness` took Poisson's ratio 0.3 by default, against the 1/3
  of the library.
- The documentation named condition types that no longer exist
  (`traction_boundary_condition` and the others) and stated that a wrong
  thickness of an axisymmetric problem is accepted.
- The distributed linear solve accepted a solve that PETSc reported as
  converged on its preconditioned residual: algebraic multigrid on a matrix
  with zero diagonal entries (the mixed beams) gave a zero step after 0
  iterations, and the beam stayed at zero deflection on two processes. Every
  PETSc solve is now checked by its true residual, and the automatic choice
  falls back to the direct solver.
- `gap_contact` of a plane problem ignored the thickness of the equations, so
  its penalty acted as k / h.
- The plane-stress `volumetric_strain` left out the out-of-plane strain, which
  the `strain` property now holds.
- The `stress()` docstring gave the Voigt order wrongly (it is xx, yy, zz, yz,
  xz, xy).
- Four of the five sides of the `Wedge6` element were stored in an order
  whose right-hand normal pointed into the prism, so `write_sidesets` wrote
  those faces wound inward. Every side of every element type is now wound
  outward, which a test checks against the divergence theorem.
- The cell-centred finite volume method (`zfvm`) joined only two of the
  elements at a face shared by three or more (overlapping or duplicated
  elements) and solved on the wrong connections. Such a mesh is now refused
  with the numbers of the elements.
- A mesh whose nodes had a constant coordinate that was not zero (a plate at
  z = 0.5) became a surface in space, and so did a plane mesh with round-off
  in z. A coordinate now moves a mesh into a higher space only when its range
  is larger than 1e-12 times the size of the mesh.
- `read_mesh` found a physical name by its tag alone. Gmsh numbers its
  physical groups separately in each dimension, so a boundary could take the
  name of a volume group with the same tag. Names are now found by tag and
  dimension.
- Promotion to second order, uniform refinement and adaptive refinement each
  grew the node sets by their own rule, and the midpoint of an interior chord
  between two boundary nodes joined a boundary node set. One rule now serves
  all three: a node set gains an added node whose parents are all members,
  and a set on the boundary gains only added nodes on the boundary.
- `refined()` oriented a surface again by the default rule, which could
  reverse the orientation that a generator gave. It now keeps the
  orientation of the coarse surface.
- `add_bounding_box_sidesets` used the axes of the elements, so a surface in
  space had no `back` and `front` side sets. It now uses the axes of the space
  and skips an axis along which the mesh is flat.
- `add_sideset_from_faces` kept the faces it had found when another face was
  not on the boundary. It now refuses the whole set and adds nothing.
- A Gauss rule name with other characters after its number (`gauss3x`) was
  read as `gauss3`, and `gaussx` failed with an internal error. Such names are
  now refused with a message, and `gauss11` to `gauss20` are accepted.
- An `Expression` of no variables raised an `IndexError` when it was
  evaluated. It now returns its value.
- `mesh_from_arrays(..., dimension=2)` for a plane truss of `Edge2`
  elements raised an error (the element dimension had to equal the mesh
  dimension); `dimension` is now the space the nodes live in, as documented.
- The C++ coverage job of CI stopped at the hit counts above 2^32 of the
  automatic differentiation, which gcovr takes for a gcov bug; it now warns
  about them.
- The interval of a Sobol' index on a Gaussian process came from the random
  functions of the process alone, so it could exclude the estimate (a total
  index of 0.031 with the interval 0.049 to 0.142). The interval now holds
  the bootstrap error of the estimate and the spread of the random functions
  around their mean, and it is centred on the estimate.
- `Problem.write_csv` wrote the node coordinates beside the values, which for
  the cell-centred method (`zfvm`) are at the element and boundary face
  centroids. It now writes the positions of the unknowns.
- A store of the uncertainty studies matched a run only when its input values
  were equal to the last bit, so a store written with other versions of
  NumPy and SciPy ran its runs again. It now matches the input values to a
  relative tolerance of 1e-12.
- `describe` of an input group failed on a list of groups and on a nested
  list of numbers.
- The cell-centred finite volume method (`zfvm`) took the coefficient of
  only one side at a face between two blocks, which made it first order
  wherever a coefficient (a conductivity, a diffusion coefficient) jumps
  between materials. It now uses the harmonic mean of the two sides, and
  converges at second order (`tests/python/test_finite_volume.py`).

- A singular system (a part of the mesh that no boundary condition holds,
  e.g., a floating body or a region without any temperature condition) is
  reported with the variables and the extent of that part. It was solved
  without an error and returned zero there.
- The thermal strain of a chromium coating uses the coefficient of
  Holzwarth and Stamm as the mean coefficient from 20 degrees Celsius,
  which is how they define it.
- The temperature jump distance uses the constant of Lanning and Hann,
  0.013748 in SI units.
- `uq.calibrate` with `discrepancy="gaussian_process"` integrates the
  discrepancy out of the likelihood as Kennedy and O'Hagan do, with its
  hyperparameters at the joint posterior mode. The discrepancy was fitted
  at the prior mean and subtracted from the data, which held the posterior
  at the prior mean.
- The singularity check runs again when the pattern of the Jacobian changes
  (a constraint that appears folds rows into others, an interface condition
  starts to couple two bodies), which the number of rows and entries is a
  cheap signature of; it ran once per solve however the system changed.
- The symmetry of the linear system is computed when a solver asks for it
  (the conjugate gradient method and the automatic choice with the
  multigrid), not every time a nonlinear problem stores a new matrix with a
  Newton iteration.
- The convergence diagnostic of `uq.calibrate` is the rank-normalized and
  folded split R-hat of Vehtari et al. (2021), which detects chains of
  different scales.

## [0.2.0] - 2026-09-27

### Added

- **Taylor-Hood element** for the finite element method
  (`add_incompressible_flow(..., formulation="taylor_hood")`): quadratic
  velocity and linear pressure, stable without stabilisation. Variables take
  an interpolation order, `add_variable(..., order="first")` (also
  `order: first` in input files). A first-order variable on a quadratic mesh
  is defined on the corner nodes, is interpolated with the corner (linear)
  shape functions through the element's own Jacobian, and reports the linear
  field at the other nodes, so that all post-processing applies to it without
  modification. Local degrees of freedom are numbered compactly, so the
  automatic differentiation seeds only the slots in use. The element is
  verified by manufactured solutions on Quad9, Tri6, Quad8, axisymmetric
  meshes including the axis, Tet10 and Hex27 (velocity third order in L2 and
  second order in H1, pressure second order in L2, as the theory predicts), by
  the numerical inf-sup test (the constant stays at 0.455 for Q2-Q1 and 0.366
  for P2-P1 under refinement, and is zero for the unstable equal-order pairs),
  by Newton's quadratic convergence and by the cavity of Ghia et al. (within
  0.01 at Re = 100 on 16 x 16 Quad9). New functions
  `Problem.num_active_dofs()` and `Problem.variable_order()`.
- **Pressure-velocity formulation of incompressible flow** for all four
  methods (`add_incompressible_flow(..., formulation="pressure")`): the
  pressure is an unknown, the mass equation carries the residual-based
  pressure stabilisation (PSPG for the finite element method, the equivalent
  stabilised face flux for the control volume methods), and flows with inertia
  receive streamline-upwind stabilisation. New objects `pressure_gradient`,
  `mass_conservation`, `momentum_stabilization`,
  `mass_flux_boundary_condition` and `point_Dirichlet_boundary_condition`.
  Axisymmetric flows include the hoop stress. Body forces and Boussinesq
  buoyancy enter the stabilisation with the same load scaling as the force.
  The formulation is verified by manufactured solutions (velocity second order
  in L2, pressure 1.3 to 1.7 in L2 on distorted, axisymmetric and 3D meshes),
  the cavity of Ghia et al. (within 0.0074 at Re = 100 on 64 x 64) and de Vahl
  Davis's natural convection (within 2 % up to Ra = 10^6).
  `mms.IncompressibleFlow` manufactures flows, with optional outlets carrying
  the exact traction.
- **PETSc backend** (`linear_solver="petsc"`, `petsc_options`), optional at
  build time (`-DDUALMESH_ENABLE_PETSC=ON`, found with pkg-config). In serial
  the Jacobian is passed to PETSc as a sequential matrix. In a distributed run
  it is assembled as one distributed PETSc matrix from every rank's element
  contributions in a global numbering (coordinate-format assembly), so that
  PETSc's algebraic multigrid (GAMG, hypre BoomerAMG), parallel direct solver
  (MUMPS) and field-split preconditioners are available. The number of
  variables is the matrix block size, which the field split of velocity and
  pressure uses. The options of each solve are kept in a private options
  database. New functions `have_petsc()` and `petsc_version()`. A CI job
  builds with MPI and PETSc and runs the tests on 1 to 4 processes.
- **Framework support for multi-body problems**. `InterfaceBC` is a
  boundary condition between two surfaces paired by closest-point
  projection. It is conservative by construction (the secondary side
  receives exactly what the primary side loses) and works in all four
  methods: `gap_heat_transfer`, `gas_gap_heat_transfer` (the conductance of
  Ross and Stoute with the gas mixture of Brokaw, the jump distance of
  Kennard, radiation and solid contact) and `gap_contact` (penalty,
  frictionless). A **material state store** (`stateSize`, committed on
  accepted steps, saved and restored with `save_state`/`restore_state`)
  holds the state of history-dependent materials such as creep. **Element
  fields** (`set_element_field`) provide values that materials read. New
  function `SettableFunction`.

### Changed

- The default automatic differentiation budget
  (`DUALMESH_MAX_AD_DERIVATIVES`) is raised from 48 to 96 slots, which admits
  three fields on Hex27 and the Taylor-Hood element on Hex27 (89 slots). The
  increase has no measurable cost in time or memory, because an automatic
  differentiation number copies only the slots in use.
- `linear_solver="automatic"` factorises a mixed-order (Taylor-Hood) system
  directly at any size, because its zero pressure block causes the incomplete
  LU preconditioner to fail.
- `mass_conservation` with `stabilization = false` is refused unless the
  pressure is first order on a quadratic mesh: an equal-order pressure
  without stabilisation has spurious modes.
- `point_Dirichlet_boundary_condition` on a first-order variable fixes the
  nearest corner node.

### Fixed

- The cell-centred finite volume method evaluated volume sources that depend
  on the solution at the cell value. Near the axis of an axisymmetric problem
  this made the thermal stress error of order one. The sources are now
  evaluated at the linear reconstruction at each quadrature point, and the
  error near the axis is of the same size as elsewhere in the mesh.
- In a distributed run a `point_Dirichlet_boundary_condition` was applied by
  every rank at the node of its own part nearest to the point. Only the
  globally nearest node is now constrained.
- The cell-centred method lagged the gradient in cell sources (such as the
  convective inertia), so Newton's method converged only linearly on a
  Navier-Stokes problem. The gradient is now differentiated through the
  least-squares stencil.
- The cell-centred method gave a singular matrix for an axisymmetric or
  spherical problem whose axis carried no boundary condition, because the
  boundary unknowns on the axis had an equation weighted by r = 0.
- `mms.ManufacturedSolution` simplifies the forcing of a curvilinear problem
  symbolically, so that it can be evaluated on the axis.

## [0.1.0] - 2026-09-25

The first formal release, tagged `v0.1.0` and published on PyPI as
`dualmesh-multiphysics` (the package is imported as `dualmesh` and the command
is `dualmesh`), with the documentation on <https://dualmesh.readthedocs.io>.

This release defines and organises dualmesh as a multiphysics framework for
heat transfer, solid mechanics and fluid dynamics, in which the finite element
method, two finite volume methods and the dual mesh control domain method are
four discretisations of one problem description.

### Added

- **Four new element types**: the serendipity `Quad8` and `Hex20`, the
  triangular prism `Wedge6` and the pyramid `Pyramid5` (with Bedrosian's
  rational basis). Meshes may mix types freely. Every type works with the
  finite element and cell-centred finite volume methods. The dual mesh and
  vertex-centred methods, which need node-centred control domains, accept the
  prism (whose dual is the triangle's median dual times a half-segment). They
  refuse the serendipity elements and the pyramid with a message that gives
  the reason and names the methods that accept them. New
  `Mesh.second_order(serendipity=True)`, generators for all four types,
  uniform refinement of prisms and pyramids, and meshio reading and writing.
- **Systematic verification by manufactured solutions**: `dualmesh.mms`
  derives the forcing of a chosen solution symbolically for diffusion,
  advection, reaction, time derivative and linear elasticity terms in
  Cartesian, axisymmetric and spherical coordinates, and runs convergence
  studies. `Problem.error_norms` measures the L2 error and the H1 seminorm.
  The suite runs 161 studies and every observed order matches the theory of
  its method.
- **Coupled heat transfer and flow**: `heat_convection` (the flow carries the
  heat) and `Boussinesq_buoyancy` (the temperature drives the flow), with
  `physics.add_boussinesq_buoyancy`. The natural convection benchmark of de
  Vahl Davis is reproduced to within 0.2 % in the Nusselt number
  (`examples/natural_convection.py`).
- **Automatic linear solver** (`linear_solver="automatic"`, the new default):
  a direct factorisation where it is cheap and BiCGSTAB with a new ILU(0)
  preconditioner where it is expensive, with a switch to the direct solver if
  the iteration fails. The `linear_solver` option also accepts `"gmres"`, and
  the new `preconditioner` option accepts `ilu`, `ilut`, `jacobi` and `none`.
  On a three-dimensional elasticity system of 36 000 unknowns the solve time
  fell from 18 s to 0.3 s.
- `Problem.linear_system()` returns the residual and the Jacobian as NumPy and
  SciPy objects, for use with other solvers.
- Expressions are compiled by a C++ parser (`ParsedFunction`), and a string
  that is not the name of a registered function is compiled as an expression.
- The command-line driver runs an input file under `mpirun` with the
  distributed solver (configured by a `parallel` block), accepts `threads`, and
  refuses an unknown block or setting with a suggestion.
- "Did you mean" suggestions for misspelled object types, parameters,
  variables, boundaries, blocks and solver options.
- A continuous-integration job that runs the distributed tests on 1 to 4
  processes, and a workflow that builds binary wheels.
- Documentation: a scope chapter that defines the package as a multiphysics
  framework and compares it with MOOSE and COMSOL, a manufactured-solutions
  chapter, documentation of the new elements, solvers and preconditioners,
  and the current MOOSE citation (MOOSE 4.0, 2025).

### Changed

- **The structural module is merged into solid mechanics.** Beams and plates
  are the reduced theories of solid mechanics and are now registered in the
  `solid_mechanics` module, next to continuum elasticity.
- **The cell-centred finite volume method has an exact Jacobian**: the
  non-orthogonal correction, which was lagged, is differentiated through the
  least-squares stencils, so that Newton's method converges quadratically on
  skewed meshes.
- **Efficiency**: element geometry is mapped once per integration point and
  reference-element data are cached. The derivative arrays of the automatic
  differentiation are copied only as far as they are used, and their loops
  vectorise. Symmetric quadrature rules on triangles, tetrahedra and prisms
  replace the collapsed tensor rules for the finite element method (8 points
  on a tetrahedron, compared with 27 for the collapsed rule). Gauss rules are
  tabulated. On the bundled benchmark (three-dimensional elasticity on 8000
  `Hex8` elements) the assembly of every method is about three times faster
  on one thread, and the Python test suite runs in about 60 % of its former
  time.
- All element types use the VTK node numbering, so files need no permutation.
- The package metadata links to the GitHub repository.

### Fixed

- **The theta-method evaluated the old part of the residual at the new
  time**, so the Crank-Nicolson scheme lost its second-order accuracy whenever
  a coefficient or a source depends on time (serial and distributed).
- **The penalty formulation of incompressible flow locked on the
  vertex-centred finite volume method**, because the two-point gradient
  correction was applied to the reduced-integration term. The correction is
  removed from that term, and the cell-centred method, with which the
  formulation cannot work, now refuses it.
- **A compiled expression passed as a parameter was evaluated through
  Python**, because pybind11's function test is Python's `callable()`, which
  also serialised the assembly onto one thread.
- **Thermal stress in plane strain was wrong by a factor of `1 + nu`.** The
  stiffness matrix left the out-of-plane column empty, so the thermal strain
  subtracted from the out-of-plane normal strain never reached the in-plane
  rows. For a fully restrained body the code gave
  `E alpha dT / [(1 + nu)(1 - 2 nu)]` where the correct value is
  `E alpha dT / (1 - 2 nu)`. Plane stress was unaffected.
- **A Python callable passed directly to a parameter deadlocked the threaded
  assembly.**
- **The distributed solver returned a wrong answer for the cell-centred
  finite volume method** and now refuses that method.
- When the extension had not been built in place, the Python test suite
  imported the source tree. It now imports the installed package.

### Added earlier in development

- **Quadratic elements** `Edge3`, `Tri6`, `Quad9`, `Tet10` and `Hex27`, with
  the dual mesh built for each, **adaptive mesh refinement** by gradient
  recovery, marking and longest-edge bisection, the automatic quadrature
  order, and `DUALMESH_MAX_AD_DERIVATIVES`. The manual is reorganised into a
  user guide and a theory manual with every method cited to its primary
  source.

### First development snapshot (2026-09-18)


The first development snapshot contains the framework, five physics modules
and a verification suite that reproduces the published results of Reddy's
book.

#### Added

- **Framework**: the canonical conservation form $`-\nabla \cdot \mathbf{F} + S = 0`$, and the
  median dual mesh for `Edge2`, `Tri3`, `Quad4`, `Tet4` and `Hex8`. Assembly
  for the dual mesh control domain method and for the Galerkin finite element
  method uses the same kernels. Forward-mode automatic differentiation
  provides exact Jacobians. Quadrature is chosen per kernel (Gauss, midpoint,
  trapezoid, Simpson, nodal, interface, control-domain trapezoid), with
  selective reduced integration. Cartesian, axisymmetric and spherical
  coordinates, blocks (subdomains), side sets and node sets are supported.
- **Solvers**: Newton's method, direct (Picard) iteration with relaxation,
  load stepping, steady and transient (theta-method) executioners, sparse LU,
  BiCGSTAB and conjugate gradient linear solvers.
- **Heat transfer module**: conduction with temperature-dependent
  conductivity, volumetric heating, heat capacity, convective, radiative and
  prescribed-flux boundary conditions.
- **Solid mechanics module**: linear elasticity (plane stress, plane strain,
  axisymmetric, three-dimensional), traction and pressure boundary conditions,
  stress and strain as material properties.
- **Structural module**: mixed Euler-Bernoulli beam, displacement and mixed
  Timoshenko beams, first-order axisymmetric circular plate, first-order
  rectangular plate, functionally graded sections, von Kármán nonlinearity.
- **Fluids module**: penalty formulation of the Stokes and Navier-Stokes
  equations with recovered pressure.
- **Meshing**: line, rectangle, box and annulus generators with graded
  spacing, uniform refinement, geometric side sets and node sets, and readers
  and writers for Gmsh, Exodus, VTK, Abaqus and the other formats meshio
  supports.
- **Python interface**: `Problem`, mesh generators, physics helpers,
  post-processing, functionally graded stiffness formulas, and base classes
  for kernels, boundary conditions and materials written in Python.
- **Command line**: `dualmesh run input.yaml`, `dualmesh list`,
  `dualmesh describe <type>`.
- **Verification**: the published dual mesh results of Chapters 3, 5, 6, 7, 8,
  9 and 10 of Reddy's book, plus patch tests, conservation tests and
  cross-verification against OpenFOAM.
