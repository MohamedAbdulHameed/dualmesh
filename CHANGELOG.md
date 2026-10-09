# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project uses
[semantic versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **`dualmesh.materials`**: functions of the state that return material
  properties in SI units. `materials.water` gives compressed liquid water
  (IAPWS-IF97 region 1, with the IAPWS viscosity and thermal conductivity),
  and `materials.gas` gives the thermal conductivity of helium, argon,
  krypton, xenon, hydrogen and nitrogen at low density and of their mixtures,
  and the temperature jump distance at a wall.
- **General partial differential equations.** `general_form_PDE` solves a
  system d du/dt + div Gamma = f whose flux Gamma and source f are
  expressions of all the fields, the components of their gradients
  (`grad_x(u)`, `grad_y(u)`, `grad_z(u)`), x, y, z, t and named constants.
  The expressions are compiled and differentiated automatically, so the
  Jacobian is exact and the assembly runs on every thread and every process
  with every method. `parsed_kernel` is the object of one such term.
- **Eigenvalue study of any problem.** `Problem.solve_eigenvalue` returns
  the eigenvalues closest to `near` and their modes for every problem
  without `neutron_diffusion`. Every field varies in time as
  u_hat exp(-lambda t), so the time derivatives define the eigenvalue, and
  the expressions may use the symbol `eigenvalue`, at most quadratically.
  The operators come from exact assemblies, and shift-invert Arnoldi
  iteration solves the linear or the companion quadratic problem. Complex
  eigenvalues are reported with both parts.
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
- **Neutronics.** The physics `neutron_diffusion` solves the multigroup
  neutron diffusion equations with any number of groups, scattering between
  all groups, a transverse buckling, and the boundary conditions
  `vacuum_boundary_condition` (no incoming current, with the extrapolation
  distance as a ratio of the diffusion coefficient) and
  `albedo_boundary_condition`. The property object `multigroup_cross_sections`
  gives the cross sections of a region. `Problem.solve_eigenvalue()` computes
  the effective multiplication factor and the fundamental mode by the Arnoldi
  method or the power iteration, and reports the volume, the average fluxes
  and the fission neutron production of each region. Verified against the
  analytical one- and two-group solutions of bare slabs, cylinders and
  spheres, and against the 2D IAEA PWR benchmark (ANL-7416 Suppl. 2, problem
  11-A2): with 32 x 32 elements per assembly, k_eff is within 1.3 pcm and
  the assembly powers are within 0.34 % of the reference, with all four
  methods.
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
  mutation test, Dependabot, actions pinned to commit hashes, read-only job
  permissions and a software bill of materials of each release. The release
  script refuses a workflow with an action that is not pinned to a commit.
- **`coefficient_form_PDE`**: a physics for a scalar equation written by its
  coefficients, d_t du/dt + div(-c grad u - alpha u) + beta . grad u + a u = f,
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
  several processes splits itself among them (`distributed`, `partitioner`
  and `overlap` of the constructor), and its `solve` and `solve_transient`
  take the distributed linear solver options. `rank`, `num_ranks`,
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

- `coefficient_form_PDE` takes `variables` (a list) in place of `variable`,
  and `units` in place of `unit`. Without a diffusion coefficient, the
  equation of a field u of a system reads the property
  `diffusion_coefficient_u`.
- The method `variables()` of a physics is renamed `variable_names()`.
- `gas_gap_heat_transfer` takes the data of the surfaces from the case, with
  no default: the roughnesses, the emissivities and, with a
  `contact_penalty`, the Meyer hardness and the conductivities of the two
  bodies. Every mole fraction of the gas has the default 0.
- The author of the software is Mohamed AbdulHameed, with his ORCID iD, in
  `CITATION.cff`, `pyproject.toml` and the documentation.
- The citation is the software alone (`CITATION.cff`, the README and the
  documentation). Its abstract covers neutron diffusion, general partial
  differential equations and uncertainty quantification.
- A boundary condition given no `boundary` acts on the side set or node set
  whose name equals the name of the condition, so that
  `add_boundary_condition("Dirichlet_boundary_condition", "left", ...)` needs
  no separate `boundary="left"`.
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

- The neutronics module: the `neutron_diffusion` physics, the
  `multigroup_cross_sections` property object, the
  `vacuum_boundary_condition` and `albedo_boundary_condition`, and the power
  iteration of the k-eigenvalue study. Neutron diffusion is written with
  `coefficient_form_PDE` and solved by the eigenvalue study of any problem:
  `examples/reactor_criticality.py` and the tutorial on reactor criticality
  show how. The IAEA 2D PWR benchmark written in this form gives the same
  k_eff and fluxes as the removed module to round-off.
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
- **Overlapping Schwarz**: the distributed preconditioners now work on
  subdomains that overlap by `overlap` layers of elements (default 1), with
  fully assembled subdomain matrices exchanged between ranks, an exact or
  incomplete subdomain solve (`subdomain_solver`), and the two levels combined
  multiplicatively. The iteration count of the default preconditioner is
  independent of the number of processes: 23 to 25 iterations from 1 to 16
  ranks on a 48 by 48 Poisson problem. With `linear_solver="cg"` the
  symmetric (classical) forms are used.
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
- **The default distributed preconditioner is `two_level_schwarz` with an
  overlap of one element layer** (it was `jacobi`).
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
- **The Schwarz preconditioners factorised each rank's partial local
  matrix**, which caused them to stop converging at eight ranks. They now
  factorise the fully assembled subdomain matrix.
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
- **The distributed BiCGSTAB detects near-breakdown and restarts.**
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

- **Framework**: the canonical conservation form `-div F + S = 0`, and the
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
