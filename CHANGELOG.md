# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project uses
[semantic versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Uncertainty quantification and sensitivity analysis** (`dualmesh.uq`, and
  the `uq` block of input files). Input distributions `Normal`, `LogNormal`,
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
  and stored, so that an interrupted study resumes. The same studies run
  from YAML input files with `dualmesh run`.
- **Model factors for fuel rods** (`fuel.ModelFactors`): multipliers on the
  linear heat rate, the thermal conductivities, the gap conductances, the
  coolant heat transfer, the thermal expansion, relocation, densification,
  swelling, creep and the fission gas parameters, for sensitivity and
  uncertainty studies.
- **Wrench example** (`examples/wrench`, tutorial "Boundary conditions on an
  imported mesh"): a three-dimensional steel wrench on a Gmsh mesh, with the
  boundaries named by physical groups or by geometric predicates, in Python
  and YAML, verified against the statics of the wrench and beam theory.
- **`property_scaling` material** and the `scaled_properties` and
  `property_factors` parameters of every material, which multiply named
  material properties by constant factors.

### Changed

- A boundary condition given no `boundary` acts on the side set or node set
  whose name equals the name of the condition, so that
  `add_boundary_condition("Dirichlet_boundary_condition", "left", ...)` and
  a YAML entry named `left` need no separate `boundary: [left]`.
- The fuel performance chapters form a part of their own in the
  documentation.

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
  `fuel_performance`), written from the published literature without the
  source code of any other fuel performance code, with the architecture of
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
