# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project uses
[semantic versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `RodOutput.output_interval` records the rod state at a fixed interval from
  the start of the power history and at its end. The points of the power
  history remain the default output times.

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
  columns as arrays and as a pandas DataFrame. Every result has `tables`
  (for example `rod.tables["axial"]`) and `write_csv(path, table=None)`.
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
- `dualmesh list` and `dualmesh describe` cover the applications (`FuelRod`,
  `TrisoParticleModel`) and their input groups (`dualmesh describe
  RodNumerics`), and an unknown name is answered with the closest one.
- `scripts/audit_names.py` checks every public name and parameter
  declaration against the naming rules, and the tests run it. The input
  groups of the TRISO model declare their parameters with units and
  descriptions, and the surface properties of `ChromiumCoating`, which never
  enter a calculation, are no longer inputs.
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
- **Model factors for fuel rods** (`fuel.ModelFactors`): multipliers on the
  linear heat rate, the thermal conductivities, the gap conductances, the
  coolant heat transfer, the thermal expansion, relocation, densification,
  swelling, creep and the fission gas parameters, for sensitivity and
  uncertainty studies.
- **Wrench example** (`examples/wrench`, tutorial "Boundary conditions on an
  imported mesh"): a three-dimensional steel wrench on a Gmsh mesh, with the
  boundaries named by physical groups or by geometric predicates, verified against the statics of the wrench and beam theory.
- **`property_scaling` property object** and the `scaled_properties` and
  `property_factors` parameters of every property object, which multiply
  named properties by constant factors.

### Changed

- The author of the software is Mohamed AbdulHameed, with his ORCID iD, in
  `CITATION.cff`, `pyproject.toml` and the documentation.
- The citation is the software alone (`CITATION.cff`, the README and the
  documentation). Its abstract covers neutron diffusion, nuclear fuel
  performance and uncertainty quantification.
- A boundary condition given no `boundary` acts on the side set or node set
  whose name equals the name of the condition, so that
  `add_boundary_condition("Dirichlet_boundary_condition", "left", ...)` needs
  no separate `boundary="left"`.
- The time steps of a fuel rod are limited by the increase of the rod
  average burnup (`RodNumerics.max_burnup_step`, default 0.2 MWd/kgHM) and
  the change of its linear heat rate (`max_power_step`, default 2 kW/m), in
  addition to the step length (`max_time_step`, default 30 days). The fission
  gas release of a rod at constant power then lies within 2 % of its limit
  at zero step size. The benchmarks IFA-677.1, IFA-716.1, IFA-534.14 and
  FUMEX-II case 27 were recomputed with these steps.
- Names written out in full: `solve_transient` takes `time_step`,
  `implicitness`, `min_time_step` and `max_time_step` (were `dt`, `theta`,
  `dt_min` and `dt_max`), `solve_particle` takes `implicitness`, the
  uncertainty quantification uses `standard_deviation` (`uq.Normal`, the
  `standard_deviation` methods of the results, `return_standard_deviation`
  of `GaussianProcess.predict`), and `fgm.beam_stiffness` takes
  `poissons_ratio`.
- The Taylor-Hood formulation of `incompressible_flow` is named
  `Taylor_Hood`.
- The fuel performance chapters form a part of their own in the
  documentation.
- `linear_solver="automatic"` iterates with algebraic multigrid where it
  used BiCGSTAB with ILU(0).
- The Python code keeps every statement on one line (ruff line length 320,
  no magic trailing comma).
- Three defaults of the fuel module were chosen by running the fuel
  benchmarks (IFA-534.14, IFA-716.1, FUMEX-II case 27 and IFA-677.1) with
  every option of every model, one option at a time, and keeping the option
  that agrees best with the measurements: the ESCORE densification
  (`UO2Fuel.densification_model="escore"`, was `matpro`), the solid contact
  conductance of Ross and Stoute with their roughness
  ((R_1^2 + R_2^2)/2)^(1/2) and C_s = 20 m^(-1/2) (was 0.8 (R_1 + R_2) and
  10), and the coolant correlation of
  Weisman (`ForcedConvection(correlation="weisman")`, was `dittus_boelter`).

- `RodResult.write_csv(path, table=...)` writes one table, and its CSV
  files carry the units in the header line (was a second line `# units:`,
  which spreadsheets and pandas read as data). The files of
  `RodOutput.directory` are unchanged in name.
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
- YAML input files and the command `dualmesh run`. A problem, a fuel rod, a
  TRISO particle and an uncertainty study are described in a Python script,
  which runs with `python script.py`, or with `mpirun -n 4 python script.py`
  for a distributed problem. The example `examples/bus_bar.yaml` is replaced
  by `examples/bus_bar.py`, the optional dependency `input` (PyYAML) is
  removed, and `dualmesh mesh` reads a mesh file.

### Fixed

- The cell-centred finite volume method (`zfvm`) took the coefficient of
  only one side at a face between two blocks, which made it first order
  wherever a coefficient (a conductivity, a diffusion coefficient) jumps
  between materials. It now uses the harmonic mean of the two sides, and
  converges at second order (`tests/python/test_finite_volume.py`).

- A singular system (a part of the mesh that no boundary condition holds,
  e.g., a floating body or a region without any temperature condition) is
  reported with the variables and the extent of that part. It was solved
  without an error and returned zero there.
- The Halden UO2 conductivity limits the temperature of its phonon term to
  1650 degrees Celsius, as Wiesenack defines it. Results change above
  1923 K only.
- The theoretical density of UN follows Eq. (3) of Hayes et al. (part I),
  whose linear coefficient is 2.997e-4. The value 2.779e-4 of the abstract
  is a misprint. The density at 298 K is 14326 kg/m^3.
- The thermal strain of a chromium coating uses the coefficient of
  Holzwarth and Stamm as the mean coefficient from 20 degrees Celsius,
  which is how they define it.
- The ESCORE densification uses the pellet-average burnup, as FALCON MOD01
  defines it.
- The radius of a gas atom in the intragranular fission gas model is 0.2 nm,
  the value of Pizzocri et al. (2018), Table 1.
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
- **Nuclear fuel performance module** (`dualmesh.fuel`, C++ module
  `fuel_performance`), with the architecture of
  BISON (one fully coupled Newton system with exact Jacobians), the 1.5D rod
  of TRANSURANUS and the finite volume option of OFFBEAT, and with all four methods available.
  `FuelRod` runs a rod through a power history in 1.5D (generalized plane
  strain, with a secant iteration on the axial force balance), 2D-RZ or 3D,
  for UO2 or UN fuel in Zircaloy-4. Materials: `UO2_thermal` (Fink, Fink with
  Lucuta factors, NFI), `UN_thermal` (Hayes), `Zircaloy_thermal`,
  `FuelElasticProperties`, `FuelThermalExpansionEigenstrain`,
  `UO2VolumetricEigenstrain` (densification and swelling),
  `UNSwellingEigenstrain` (Ross), `FuelRelocationEigenstrain`,
  `ZircaloyGrowthEigenstrain` (Franklin), and creep inside
  `EigenstrainElasticStress` (MATPRO UO2, Hayes UN, Limbäck-Andersson
  Zircaloy) by backward Euler radial return with an exact derivative. The
  module also provides `FuelGapHeatTransfer` (Ross-Stoute conductance with the
  Brokaw gas mixture, Kennard jump distance, radiation and solid contact),
  `gap_contact` (penalty, frictionless), the ideal-gas rod pressure, and
  fission gas release by `BoothGasModel` (exact spectral intragranular
  diffusion with the Turnbull coefficient and grain-boundary saturation) and
  by `StormsGasModel` for UN. `PowerHistory` and `rod_fields` tabulate power,
  fission rate, burnup and fast fluence. `Coolant` models a coolant channel
  with the Dittus-Boelter correlation, and the coolant can alternatively be
  computed by the flow module (`examples/fuel_rod_conjugate.py`). The
  correlations are also exposed one by one in `dualmesh._core.fuel`. The
  module has 41 tests, including exact solutions (Lamé cylinder, thermal
  stress, gap conduction, creep, Booth diffusion) and the agreement of 1.5D,
  RZ and 3D rods and of all four methods. A documentation chapter lists each
  correlation with its source.
- **Fission gas behaviour of UO2 and Cr2O3-doped UO2**: intragranular
  diffusion with the Turnbull coefficient, trapping in and re-solution from
  intragranular bubbles, grain-face bubbles that grow by vacancy absorption,
  coalesce and vent at saturation (White 2004, Pastore et al. 2013), and
  burst release by micro-cracking of the grain faces (Barani et al. 2017).
  `DopedUO2Fuel` with the doped diffusivity of Cooper et al. (2021) and the
  densification measured on the Halden rods.
- **Gaseous swelling from the fission gas model**
  (`UO2Fuel.gaseous_swelling_model = "fission_gas"`, the default): the
  volume of the intragranular and grain-face bubbles (Pastore et al. 2013,
  Eqs. 9 and 10). MATPRO FSWELL remains available as `"matpro"`.
- **Halden conductivity of irradiated UO2** (`thermal_conductivity_model =
  "halden"`, IAEA-TECDOC-1496).
- **Rod gas volume of the relocation cracks** in the ideal gas law.
- **Time steps on the power history** (`RodNumerics.power_history_tolerance`):
  the steps fall on the points of the history that are needed to follow it to
  within 1 % of its largest power, so that shutdowns and power steps are
  resolved.
- **TRISO particles** (`dualmesh.fuel.triso`): the spherical multilayer
  model of the CRP-6 benchmark, verified against its closed forms and the
  eight participating codes.
- **Fuel benchmarks** (`verification/benchmarks`): FUMEX-II cases 27(1),
  27(2a), 27(2b) and IFA-534.14, and the Cr2O3-doped Halden rods IFA-716.1
  and IFA-677.1 against measured temperatures, pressures and fission gas
  release and against BISON.
- **Framework support for fuel and other multi-body problems**.
  `InterfaceBC` is a boundary condition between two surfaces paired by
  closest-point projection. It is conservative by construction (the secondary
  side receives exactly what the primary side loses) and works in all four
  methods. A **material state store** (`stateSize`, committed on accepted
  steps, saved and restored with `save_state`/`restore_state`) holds the
  state of history-dependent materials such as creep. **Element fields**
  (`set_element_field`) provide values that materials read. New functions
  `CylinderTableFunction` and `SettableFunction`.

### Changed

- A rod time step whose backtracking line search stalls (as happens when the
  pellet comes into contact with the cladding during a fast power change) is
  repeated with full Newton steps.
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
