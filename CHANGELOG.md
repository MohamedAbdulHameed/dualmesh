# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project uses
[semantic versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **The input.** A simulation is described as data: the mesh, the method, the
  physics, the properties, models of your own, the initial and boundary
  conditions, gravity, the time span or an eigenvalue study, the mesh
  adaptivity, the postprocessors, the output and the solver settings. You write
  it in Python with `dualmesh.problem`, or as a case file in YAML or TOML that
  `dualmesh run case.yaml` solves and `dualmesh.problem.from_file` reads. All
  three build the same tree and pass the same checks. A quantity is a text
  with its unit, such as `"45 W/(m*K)"`, which the input converts to SI units
  and checks against the dimension of its key. Fields, properties and boundary
  quantities take standard names with fixed units
  (`python/dualmesh/standard_names.toml`). A property is a constant, a formula,
  a table (`dualmesh.table`) or a mapping of regions. The `models` section
  holds balance laws of your own. Every key is checked before the solve, and a
  misspelled key is answered with the nearest valid name. `solve` returns a
  `dualmesh.results` object, and `to_YAML`, `to_TOML` and `to_tree` write the
  input back. `examples/heated_plate.yaml` couples heat conduction and solid
  mechanics. PyYAML is the optional dependency `dualmesh[yaml]`.
- **Physics level.** `Problem.add_physics(type, **parameters)` adds a set of
  equations named in physical terms: `heat_conduction`, `solid_mechanics`,
  `fluid_dynamics` and `general_form_PDE`. A physics defines its variables
  (`temperature`, the displacements `disp_x`, `disp_y` and `disp_z`, or
  `velocity_x`, `velocity_y`, `velocity_z` and `pressure`) and generates its
  kernels and property objects. A property that is not given is read from the
  property objects of the problem. `Problem.add_coupling(type, **parameters)`
  couples two physics: `thermal_expansion`, and `nonisothermal_flow` (the heat
  that a flow carries and the Boussinesq buoyancy). A coupling takes its
  physics as the objects that `add_physics` returned (`heat_conduction=heat`),
  or none when the problem has one physics of each type it couples.
  `dualmesh list --category physics` and `dualmesh describe <physics>`
  document them. A misspelled parameter is answered with the closest name and
  the list of accepted parameters.
- **Boundary conditions written as text**, for every physics. A variable on
  the left prescribes its value (`"temperature = 300"`), and the name of the
  flux of the physics prescribes the flux along the outward normal:
  `heat_flux` (the heat leaving the body), `traction`, `total_force` and
  `pressure` of a solid, and `normal_flux` of `general_form_PDE`. The right
  side may depend on the fields, so convection and radiation are written as
  they are printed. Any other keyword is a definition that the text uses
  (`"total_force = [0, -F, 0]", F=150.0`). A value may hold at a point:
  `flow.add_boundary_condition("pressure = 0", point=[0.5, 0.0])`. The words
  `fixed`, `symmetry` and `periodic` need no data. A type with keywords, such
  as `heat.add_boundary_condition("Dirichlet", "left", value=40.0)`, is
  refused with a message that shows the text form.
- **Equations written as they are printed.** `general_form_PDE` takes
  `variables` and `equations`, for example
  `equations="rho*cp*dT/dt = div(k*grad(T)) + q"`. The language has `du/dt`,
  `d2u/dt2`, `du/dx` (or `grad_x(u)`), `du/dn`, `grad`, `div`, `laplacian`,
  `curl`, `dot`, `cross`, `norm`, vectors `[a, b]`, the tensor functions `I`,
  `transpose`, `trace`, `determinant`, `inverse`, `symmetric`, `skew`,
  `deviatoric`, `inner` and `outer`, and `conditional`. Each function has one
  name. The fields are names or one-entry dicts,
  `variables=[{"U": "vector"}, "p"]`, with the components `U_x` and `U_y`.
  The equations are reduced symbolically to the flux, the source and the
  storage of each scalar unknown. Keywords that are not parameters are
  definitions: a number is a named constant and a text a named expression. An
  equation may instead be given by its coefficients `m`, `d`, `c`, `alpha`,
  `gamma`, `beta`, `a` and `f`. A flux condition may be written out
  (`"-k*du/dn = h*(u - 1)"`), and is checked against the flux of the
  equation. The expressions are compiled and differentiated automatically, so
  the Jacobian is exact and the assembly runs on every thread and every
  process. `parsed_kernel` is the object of one such term. The user guide page
  "Equations written as text" defines the language and works ten examples,
  from Poisson's equation to the Cahn-Hilliard equation, two-group neutron
  diffusion and incompressible flow.
- **Automatic continuation of a steady solve.** When Newton's method does not
  converge on the problem itself, the solve takes it again from the starting
  state with its driving data ramped: the sources, body forces and buoyancy,
  the prescribed fluxes, tractions and pressures, the point loads and the
  distributed loads of plates and beams from zero, and the prescribed values
  from those of the starting state. A step that converges doubles the next
  one, and a step that fails is halved, down to $1/1024$ of the data. A
  lid-driven cavity at the Reynolds number $Re = 3200$ converges this way with
  no setting. `result.history` records each step (`continuation_step`,
  `data_fraction`).
- **Flows through a boundary named by their physics.** `heat.heat_flow(b)`
  (the heat leaving, in W), `solid.reaction_force(b)` (the force of the
  support on the body, by direction), `flow.force(b)` (the force of the fluid
  on the wall: drag and lift), `flow.flow_rate(b)` (the volume flow leaving),
  `pde.boundary_flux(b)` of a `general_form_PDE`, and
  `Problem.boundary_flux(variable, b, per_node=False)`. Each takes the sign of
  the flux in the boundary conditions, out of the body. On a boundary of
  prescribed values it is the reaction there, conserved exactly. On any other
  boundary it is the integral of the flux its conditions prescribe, so a
  boundary without a condition carries nothing. A corner node belongs to the
  boundary whose condition holds it. The post-processors `heat_flow`,
  `reaction_force`, `force`, `flow_rate` and `boundary_flux` give the same
  values.
- **Tests of signs, conservation and symmetry.**
  `tests/python/test_signs.py` checks the direction of every flux, load,
  source and result against hand results, with the zero cases and the
  corners, in both methods. `tests/python/test_conservation.py` checks the
  balance of every physics and coupling to round-off, and states the measured
  gap of the penalty and Taylor-Hood formulations.
  `tests/python/test_symmetry.py` checks that symmetric problems of every
  physics and coupling have symmetric solutions in both methods, that FEM
  gives symmetric matrices for self-adjoint operators, and reciprocity
  (Maxwell-Betti).
- **Linear buckling.** `problem.solve_buckling(num_modes)` after a solve
  solves $(\mathbf{K} + \lambda \mathbf{K}_\sigma)\mathbf{x} = 0$, with
  $\mathbf{K}$ the stiffness of the undeformed body, $\mathbf{K}_\sigma$ the
  geometric stiffness of the solved stresses, taken from the geometrically
  nonlinear equations (`large_deformation`, or `von_karman` for beams and
  plates), and $\lambda$ the load factor. `load_factors` are the critical
  multiples of the load. Verified on pinned Euler and Timoshenko columns at
  second order in the element size, and on a plane-strain cantilever column
  within $8 \times 10^{-5}$ of the Euler load.
- **Generalized plane strain.** `formulation="generalized_plane_strain"` with
  `axial_force`: the uniform axial strain `strain_zz` is one unknown of the
  section (the constraint `uniform_field`), whose equation states that the
  axial stress integrated over the section equals the axial force
  (`out_of_plane_stress`). Verified against the closed forms, a free thermal
  expansion and Lame's open-ended tube.
- **Viscoelasticity.** `viscoelasticity={"shear": [[G_i, tau_i], ...], "bulk":
  [[K_i, tau_i], ...]}` of `solid_mechanics` adds generalized Maxwell branches
  (a Prony series) of moduli $G_i$ and $K_i$ and relaxation times $\tau_i$ to
  the long-term moduli. Each branch is integrated exactly over a step by the
  recursive update of Herrmann and Taylor. A steady solve takes the long-term
  moduli. Verified: a held shear and a held dilatation relax as the Prony
  series at any step.
- **Fracture.** `solid.j_integral(crack_tip, radii)` gives the domain
  J-integral, and `solid.stress_intensity(crack_tip, radii)` the stress
  intensity factors $K_I$ and $K_{II}$ by the interaction integral with the
  Williams fields, in plane strain and plane stress.
  `phase_field_fracture={"toughness": ..., "length": ...}` adds the AT2 phase
  field with Miehe's split of the strain energy and a history field that does
  not heal (object `phase_field_damage`, field `damage`). Verified: J and K of
  the Williams fields on rings of several radii, mixed mode, and the closed
  form of a uniform bar.
- **Trusses and shells.** `formulation="truss"` takes the bars of a line mesh
  in one, two or three dimensions, with axial force only
  (`cross_section={"area": A}`, object `truss`). `formulation="shell"` takes
  the flat facets of a quadrilateral surface mesh in three dimensions, with
  membrane, Mindlin bending, transverse shear at the centre of the element and
  the rotation about the normal tied to the rotation of the membrane (Hughes
  and Brezzi), and the variables `disp_x`, `disp_y`, `disp_z`, `rotation_x`,
  `rotation_y` and `rotation_z` (object `shell`). Both are linear elastic.
  Verified: a flat shell equals the plate formulation and stays so when turned
  in space, and with FEM the Scordelis-Lo roof reaches 0.990 of its reference
  on 16 elements a side and the pinched cylinder 0.997 on 32, from below.
  `reaction_force` takes a node set, a support at a point.
- **Cohesive zones** (two dimensions, linear sides):
  `solid.add_boundary_condition("cohesive", primary=..., secondary=...,
  strength=..., toughness=...)`, object `cohesive_zone`. A bilinear
  traction-separation law, normal and tangential, on surfaces that coincide at
  the start and need not match. The damage grows with the largest effective
  opening and does not heal, and a closing zone keeps its full stiffness.
  Verified: a path that softens, unloads, closes and separates follows the law
  at every step, in both methods and with the surfaces swapped, the work of
  separation equals the toughness within a relative $10^{-3}$, and the forces
  on the two bodies balance.
- **Mortar contact with Coulomb friction** (two dimensions, linear sides):
  `solid.add_boundary_condition("contact", primary=..., secondary=...,
  friction_coefficient=...)`, object `mortar_contact`. The contact pressure and
  the tangential traction are Lagrange multipliers on the secondary surface,
  the complementarity conditions are solved by Newton's method with an active
  set, and the surfaces need not match and may slide. `solid.contact_pressure(
  secondary)` gives the pressure, and `reaction_force` the contact force on
  each surface. Verified: the contact patch test on non-matching meshes, a gap
  that closes, stick and full slip, the balance of the forces on the two
  surfaces, and the Jacobian against finite differences.
- **J2 plasticity.** `plasticity={"yield_stress": ..., "kinematic_hardening":
  {"C": [...], "gamma": [...]}, "constants": {...}}` of `solid_mechanics`:
  isotropic hardening as any expression of the equivalent plastic strain and
  the temperature, and Armstrong-Frederick or Chaboche back stresses. Backward
  Euler radial return with the exact tangent, steady and in time, at small and
  finite strain, in plane stress, and with creep in one return. The results
  are `plastic_strain`, `equivalent_plastic_strain` and `back_stress`.
  Verified against closed forms, a strain cycle against the uniaxial law
  integrated by hand, and quadratic convergence of Newton's method.
- **Material laws of `solid_mechanics`.** `large_deformation` (finite strain
  and rotation, with a pressure that follows the deformed surface),
  `elasticity` (`"linear"`, `"neo-Hookean"` or `"Saint_Venant_Kirchhoff"`),
  `strain_energy` (a hyperelastic law written as its strain energy),
  `creep={"rate": ..., "constants": ..., "grain_size": ...}`, `plasticity` and
  `temperature` (a field or a number). The creep rate may read the time `t`
  (time hardening) and `equivalent_creep_strain` (strain hardening, solved at
  the end of the step in the scalar return map), and `grain_size` is needed
  only when the rate uses it. In plane stress the return map solves the
  out-of-plane strain at each point so that the out-of-plane stress is zero,
  with the exact tangent.
- **Anisotropic elasticity.** `youngs_modulus`, `poissons_ratio` and
  `shear_modulus` take three values each for an orthotropic continuum,
  `stiffness` a $6 \times 6$ matrix, and `orientation` the material axes (an
  angle in two dimensions). Hill's equivalent stress
  (`"hill": [F, G, H, L, M, N]`) serves creep and plasticity, by a
  closest-point return with the exact tangent. Verified against the compliance
  along each axis, the rotation of the fourth-order tensor, and the closed
  forms of Hill.
- **Structural dynamics.** `solid_mechanics` takes a `density`, Rayleigh
  `damping={"mass": a, "stiffness": b}` and an `initial_condition` of
  displacement and velocity. Beams take the inertia of their section and
  plates that of their thickness. `solve_transient` integrates a problem with
  inertia by the generalized-$\alpha$ method of Chung and Hulbert (the
  default, with `spectral_radius` 0.75) or by `time_scheme="Newmark"`
  (average acceleration, which conserves the energy), from the acceleration
  of the start. The new objects are `second_time_derivative` and
  `stiffness_damping`, and `general_form_PDE` takes the coefficient `m` of a
  second time derivative. `solve_eigenvalue` takes the inertia for the natural
  frequencies (`result.frequencies`, in Hz). Verified: a string at second
  order in time, the energy of Newmark, a damped oscillator, cantilever beam
  frequencies at second order in the element size, the modal damping ratios
  of Rayleigh damping, and the momentum that the support force changes.
- **Eigenvalue study of any problem.** `Problem.solve_eigenvalue` returns the
  eigenvalues closest to `near` and their modes. Every field varies in time as
  $\hat{u} e^{-\lambda t}$, with $\lambda$ the eigenvalue, so the time
  derivatives define it, and the expressions may use the symbol `eigenvalue`,
  at most quadratically. The operators come from exact assemblies, each
  measured in its own units, and shift-invert Arnoldi iteration solves the
  linear or the scaled companion quadratic problem. Complex eigenvalues are
  reported with both parts. For a vibration, where no term uses `eigenvalue`,
  `near` is the angular frequency in 1/s that the modes are closest to. The
  study folds periodic copies and hanging nodes into the unknowns they depend
  on. `examples/reactor_criticality.py` and the tutorial on reactor
  criticality write two-group neutron diffusion as balance laws and find the
  multiplication factor this way.
- **Adjoint sensitivities.** `Problem.sensitivities(objective, parameters)`
  gives the derivative of an objective of a converged solution with respect to
  every named parameter, at the cost of one adjoint solve plus one residual
  assembly per parameter. The objective is the integral of a variable over the
  domain (the variable's name) or over one side set
  (`{"variable": ..., "boundary": ...}`). The parameters are constants
  registered with `Problem.add_parameter(name, value)` and read by name
  wherever a coefficient, a source or a boundary value takes a function.
  `Problem.set_parameter` assigns one between solves, and
  `Problem.boundary_integral` gives the value of a boundary objective. The
  derivative of the residual with respect to a parameter is a central
  difference at the converged solution. Verified against the central
  difference of whole solves, on linear and nonlinear conduction, for both
  methods and both objectives. The result is a `SensitivityResult`. Periodic
  conditions, material histories, distributed problems and a solution that is
  not the converged steady solution of the model as it stands are refused.
- **Advection schemes.** The `advection` kernel carries the heat of a flow
  and the momentum of its inertia, and takes `advection_scheme`: `"central"`
  with either method, or the face fluxes of DMCDM `"upwind"`,
  `"linear_upwind"` (the upstream value corrected by the upstream gradient,
  reconstructed by least squares) and `"limited"` (blended towards the
  central value by the van Albada limiter), each with the exact Jacobian. The
  heat that a `nonisothermal_flow` coupling carries takes the streamline
  upwind Petrov-Galerkin stabilisation (SUPG) by default, with the new kernel
  `heat_streamline_stabilization`, whose residual is the strong residual of
  every other kernel of the temperature. `advection_scheme` of
  `heat_conduction` chooses another scheme. The inertia of an equal-order
  flow takes SUPG through `momentum_stabilization`, with no option. At a cell
  Peclet number of 5, the upwind and limited schemes stay bounded where the
  central scheme oscillates.
- **The projection time integration of incompressible flow**
  (`fluid_dynamics(time_integration="projection", time_order=k)`), the
  splitting of Nek5000 and nekRS: the backward difference formula and the
  extrapolation of order $k$ (1, 2 or 3), with coefficients for variable
  steps, a pressure Poisson equation with the rotational boundary term of the
  viscous term, solved from the projection of the earlier pressures with
  BoomerAMG, and Helmholtz equations for the velocity with a Jacobi
  preconditioner, by the conjugate gradient method with FEM and GMRES with
  DMCDM. It runs serial and distributed, with periodic boundaries and hanging
  nodes, and gives the forces on boundaries with a prescribed velocity at
  every step, so that `flow.force` gives drag and lift. Checked at second
  order in space on the Taylor-Green vortex, at the order of the backward
  difference in time on an exactly represented flow, and against the serial
  run on one to four processes.
- **A Courant number step controller**: `solve_transient(time_stepper="cfl",
  courant_number=0.5)`.
- **The Laplacian form of the viscous term**: `fluid_dynamics(viscous_form=
  "laplacian")`, whose natural boundary condition is the do-nothing outflow of
  the DFG benchmarks and of Nek5000. It is the default of the projection.
- **Transient monolithic flow**: `fluid_dynamics` adds the inertia
  $\rho\, \partial \mathbf{v} / \partial t$, with $\rho$ the density and
  $\mathbf{v}$ the velocity, which a transient solve includes and a steady
  solve leaves out. A transient run first projects the initial velocity onto a
  field without divergence that keeps the prescribed values
  (`result.start_correction`).
- **Benchmarks** (`docs/benchmarks`, one page each, with their scripts in
  `verification/benchmarks/fluids`): the steady flow around a cylinder at
  $Re = 20$ (DFG 2D-1) and the vortex shedding at $Re = 100$ (DFG 2D-2)
  against Schaefer and Turek, the Kovasznay flow against its exact solution,
  and the lid-driven cavity at $Re = 1000$ against Erturk et al. (2005).
- **A scaling study of the projection time integration**
  (`verification/benchmarks/scaling/run_projection_scaling.py`): strong and
  weak scaling of both methods on one to four processes, in the measures of
  Nek5000 and nekRS. `dualmesh_benchmark projection` times its steps on any
  number of processes.
- **Splits of a flow and of coupled physics.** Two PETSc options of dualmesh
  choose a split that PETSc cannot name. `-dualmesh_schur lsc`, `pcd` or
  `pressure_mass` takes the upper block-triangular preconditioner of a
  pressure-velocity flow with that approximation of the Schur complement of
  the pressure: the least squares commutator of Elman et al. (2006), the
  pressure convection-diffusion preconditioner of Kay, Loghin and Wathen
  (2002) in the form of Elman and Tuminaro (2009), with the Robin condition at
  an inflow, or the pressure mass matrix. The equal-order element takes the
  stabilisation of Elman et al. (2007) in `lsc` and `pcd`. The defaults are
  flexible GMRES and one V-cycle of BoomerAMG per block, on one process or
  many. `-dualmesh_split physics` solves the blocks of the physics of a
  coupled problem one after the other, the displacements of a solid together
  and every other variable alone, each with one cycle of GAMG.
- **Rigid body modes.** Every PETSc solve of a problem with the displacements
  of a solid gives its near null space (the rigid body modes of the solid and
  the constant of every other variable) to PETSc's smoothed aggregation
  multigrid (GAMG).
- **Distributed problems in `Problem`.** A `Problem` run under `mpirun` on
  several processes splits itself among them (`distributed`, `partitioner`
  and `gather_scatter` of the constructor). No process needs the whole mesh:
  a mesh is read on the first process only, and PETSc's DMPlex partitions it
  (PT-Scotch or ParMETIS) and moves every element, of any type, to its
  process. The values at nodes that several processes hold are exchanged
  through PETSc's star forests or through gslib, the gather-scatter library
  of Nek5000. `partition_mesh` uses the same partitioners. `rank`,
  `num_ranks`, `num_owned_dofs`, `num_global_dofs` and `gathered_values`
  describe a distributed problem. Every post-processor and every flow through
  a boundary gives the serial value on any number of processes. The history
  of the material is committed, and the post-processors and the step callback
  are evaluated, at every step, as in serial runs.
- **Gap conditions in distributed problems**: an interface condition pairs
  its primary and secondary sides across processes. A process that holds a
  primary side reads the secondary elements it lacks as ghost elements, which
  it does not integrate. Checked for `gap_heat_transfer` between non-matching
  meshes on one to six processes.
- **Checkpoints and restarts**: `dualmesh.Output(checkpoint_interval=...)`
  writes the time, the step, the solution, the history of the material, the
  damage of the cohesive zones and the history of the post-processors to one
  file in PETSc's binary format, in the order of the global numbers.
  `solve_transient(restart=...)` continues the run on any number of
  processes, and a run restarted on another partition ends where an
  uninterrupted run ends, to round-off. A checkpoint records a fingerprint of
  its mesh, so a restart on a mesh with the same counts but moved or
  renumbered nodes or elements is refused, and sizes beyond the PETSc integer
  range are refused with the name of `--with-64-bit-indices`.
- **Periodic boundary conditions** (`periodic`, in the new category of objects
  `constraint`): the field on a secondary boundary is the field on a primary
  boundary carried over by a translation, in one, two or three directions,
  serially and on any number of processes. Both methods converge at second
  order on manufactured solutions periodic in one direction and in two.
- **Surface meshes: two-dimensional elements in three-dimensional space.** A
  mesh has an element dimension (`dimension`) and a space dimension
  (`space_dimension`). A surface element is mapped by its $3 \times 2$
  Jacobian $\mathbf{J}$, measured by $\sqrt{\det(\mathbf{J}^T \mathbf{J})}$,
  has tangential gradients and edge conormals, and is oriented consistently
  with its neighbours (outward on a closed surface, and a surface that cannot
  be oriented is refused). Curves in the plane or in space work the same way.
  Both methods run on surfaces, serial and distributed: a tilted rectangle
  reproduces the flat solution to round-off, and the Laplace-Beltrami problem
  on the sphere converges at second order. Flow and the continuum
  formulations of `solid_mechanics` refuse surfaces. `mesh_from_arrays` and
  `read_mesh` place a mesh in the space its coordinates use.
  `Mesh.fix_orientation(preferred=...)` takes one normal per element and turns
  each connected part of a surface to the side that agrees best with them.
- **Surface generators.** `generate_cylinder_surface_mesh` (radius, length,
  `arc_angle` in degrees, edges `near_curved_edge`, `far_curved_edge`,
  `left_straight_edge` and `right_straight_edge`),
  `generate_spherical_shell_mesh` (`polar_angle` and `azimuth` in degrees,
  caps closed at a pole by quadrilaterals with no collapsed element, edges
  `upper_circular_edge`, `lower_circular_edge`, `start_meridian_edge` and
  `end_meridian_edge`) and `generate_surface_mesh`, which takes a surface
  written as equations with its definitions as keywords: a height surface
  `"z = f(x, y)"`, a parametric surface in `u` and `v`, or `f(x, y, z) = c`
  seen from a `center` or cut out of a `box` (quadrilaterals by surface nets).
  Every node lies on the exact surface, quadratic elements included.
- **Mesh tools.** `num_elements=[...]` in `generate_rectangle_mesh` and
  `generate_box_mesh`, beside the per-axis counts. `read_mesh` names the
  element blocks after the physical names of the file (`Mesh.block_names()`).
  The element facts come from the C++ core: `element_dimension`,
  `element_is_quadratic`, `element_is_serendipity`, `element_corner_type` and
  `Mesh.element_types()`.
- **An inverted or degenerate element is refused.** `initialize()` maps the
  centroid of every element and the points half way from it to the nodes, and
  names the first element whose Jacobian determinant is negative or zero by
  its index and block. The generators and `read_mesh` fix the orientation
  themselves, so this is reached by a mesh built node by node. Surfaces and
  curves in space are exempt.
- **Side set tools**: `sideset_summary` (faces, nodes, area and extent of
  every side set), `write_sidesets` (the boundary with one field per side
  set, for ParaView), `Mesh.sideset_measure`, `Problem.boundary_area` and the
  command `dualmesh mesh <mesh file> --sidesets <file.vtu>`.
- **Wrench example** (`examples/wrench`, tutorial "Boundary conditions on an
  imported mesh"): a three-dimensional steel wrench on a Gmsh mesh, with the
  boundaries named by physical groups or by geometric predicates, verified
  against the statics of the wrench and beam theory.
- **`dualmesh.solve_transient_with_adaptive_refinement`** adapts the mesh of a
  transient run after every `interval` accepted steps, and the
  `mesh_adaptivity` section of a transient input runs through it.
- **Output design.** Every study takes `report="full"`, `report="summary"` or
  `report="none"`: `Problem.solve`, `Problem.solve_transient`,
  `Problem.solve_eigenvalue`, `Problem.solve_buckling`,
  `ManufacturedSolution.convergence_study` and the adaptive refinements. A
  study inside another study prints nothing, and in a distributed run the
  first process prints. The `dualmesh.Output` group (directory, file base,
  output times or interval, fields, formats `vtu` and `csv`) writes the fields
  of a solve, a transient solve or an eigenvalue study. The time steps land on
  the output times, and a transient study writes a ParaView collection
  (`.pvd`). Each physics gives the SI units of its variables, which the CSV
  headers and the VTU data arrays carry.
- **Tables and the result protocol.** `dualmesh.Table` is the one form of
  every tabular result: it prints in aligned columns, writes a CSV file whose
  header names each column with its unit (for example `temperature [K]`), and
  gives each column as an array by its name. Every result has `summary()`,
  `to_dict()`, `tables`, `write_json(path)` and `write_csv(path, table=None)`:
  the `SolveResult` of a solve, the new `AdaptivityResult` of
  `solve_with_adaptive_refinement` (which returned the tuple
  `(problem, mesh)`) and the `ConvergenceResult` of a manufactured solution.
- `Problem.element_integrals(variable)` and `Problem.element_volumes()`.
- **`dualmesh.mms.convergence_rates(sizes, errors)`**, the observed orders
  between consecutive refinements of any study.
- **`property_scaling`** and the parameter `property_factors` of every
  property object, which multiply named properties by constant factors.
- `small_strain_stress` takes `plane_stress`.
- Expressions in the `body_force` of a solid, as in every other load, and
  point sources with one coordinate per dimension of the mesh (`points=[0.5]`
  on a line).
- The `flux` of `Neumann` and the `transfer_coefficient` and `ambient_value`
  of `Robin` may be expressions of the fields and their gradients.
- A nonlinear solve whose residual or Jacobian is no longer finite stops with a
  message that names divergence or a property that is not finite.
- **Continuous integration.** A finite-difference check of the Jacobian, builds
  with AddressSanitizer and UndefinedBehaviorSanitizer, coverage reports,
  clang-tidy, CodeQL, a weekly mutation test, actions pinned to commit hashes
  and read-only job permissions. The release script refuses a workflow with an
  action that is not pinned to a commit. `scripts/audit_names.py` checks every
  public name and parameter declaration against the naming rules.
- **A performance regression gate** (`.github/workflows/performance.yml`):
  `scripts/benchmark.py --quick --only assembly` times one residual and
  Jacobian assembly of every method for three physics, weekly, and
  `scripts/check_performance.py` compares the times with budgets of about ten
  times the calibration machine, so a regression of an order of magnitude
  fails the run and the noise of a shared runner does not.
- **The software bill of materials of a release** (the `sbom` job of the
  Wheels workflow): the SBOM action of Anchore scans the source distribution,
  and its SPDX document is kept with the wheels and the source distribution
  as an artifact of the tag.
- `dualmesh list` and `dualmesh describe` answer an unknown name with the
  closest one.

### Changed

- **PETSc and MPI are required.** SNES, the nonlinear solver of PETSc, runs
  the iteration, and PETSc solves every linear system, in the same way on one
  process and on several. One nonlinear driver and one transient driver serve
  the serial and the distributed problem. Without PETSc options of the linear
  solver, the solver factorises with MUMPS where that is cheap (always in one
  dimension, up to $10^5$ unknowns in two and up to 4000 in three). A larger
  flow takes flexible GMRES with `lsc`, a solid GMRES with GAMG and its rigid
  body modes, a solid coupled to other physics the split by physics, and any
  other system GMRES with BoomerAMG, each at the tolerance of Eisenstat and
  Walker. An iteration that does not converge in 500 iterations falls back to
  the direct factorisation. `petsc_options` takes PETSc's options as a string
  or a dictionary. Every PETSc solve is checked by its true residual, and a
  direct solve on several processes without MUMPS is refused.
- **The solvers do not depend on units.** Newton's method converges when the
  residual of every equation is at most `relative_tolerance` of the size of
  its own terms (the reference residual, the sum of the magnitudes of every
  contribution to each unknown), or at the round-off level of its rows. The
  line search, the divergence test and the iterative linear solvers use the
  same scaled norm, the step test compares each variable with its range, and
  a prescribed value takes the size of the diagonal of its equation.
  `tests/python/test_units.py` solves each case in metres and in millimetres,
  with the same iterations.
- **`line_search` takes `"automatic"`, the new default**: backtracking when a
  kernel uses the limited advection scheme, whose limiter has kinks on which
  Newton's method can cycle, and no line search otherwise.
- **A diverging Newton iteration stops** when the residual reaches $10^6$
  times its first value, with an error that says so and names the remedies.
- **`solve` and `solve_transient` take their defaults from the C++
  `SolverOptions` and `TransientOptions`**, which also fill the defaults in
  their docstrings.
- **The method follows the physics by default.** A problem with an
  equal-order flow takes DMCDM, together with any solid or heat conduction
  beside the flow, and every other problem takes FEM. `method="FEM"` or
  `"DMCDM"` chooses one, and `Problem.method` holds this spelling.
- **Abbreviations in the input are written in capitals and proper names with a
  capital**: `advection_scheme="SUPG"`, the methods `"FEM"` and `"DMCDM"`,
  and the nonlinear solvers `"Newton"` and `"Picard"`. The lower-case
  spelling is accepted.
- **The default formulation of `fluid_dynamics`** keeps the pressure:
  Taylor-Hood (`"Taylor-Hood"`) when `method="FEM"` is given on a quadratic
  mesh, and the equal-order formulation (`"equal_order"`, formerly
  `"pressure"`) otherwise. `penalty_parameter` is refused with any formulation
  but `"penalty"`.
- **The advection of a flow conserves heat and momentum exactly.** In an
  equal-order flow the `advection` kernel carries the velocity
  $\mathbf{w} = \mathbf{v} - \tau \mathbf{r}_m$ that the stabilised mass
  equation conserves (Hughes and Wells 2005), with $\tau$ the stabilisation
  parameter and $\mathbf{r}_m$ the momentum residual: a source for FEM, and
  for DMCDM a face flux with every scheme, on the boundary faces too. The
  heat and the momentum are conserved exactly against the discrete mass flux,
  and the temperature does not depend on its scale. At the old time of a
  Crank-Nicolson step $\mathbf{w}$ is the velocity that the previous step
  conserved. The penalty and Taylor-Hood formulations carry $\mathbf{v}$.
- **The stabilised residual is consistent on quadratic elements.** The
  momentum residual of the stabilisation takes the viscous term, and the SUPG
  of the heat the conduction, from the second derivatives of the shape
  functions, with the exact Jacobian. Poiseuille flow is exact to round-off on
  `Quad9` and `Tri6` with both methods. Linear elements, inside which the
  Laplacian vanishes, keep the standard residual.
- **The stabilisation parameter $\tau$ has the factor 9 on its viscous and
  conductive term**, as MOOSE's navier_stokes module takes it, for the
  stabilised flow of every method, the SUPG of the heat and the projection.
  Its viscous limit $h^2 / (12 \nu)$, with $h$ the element size and $\nu$ the
  kinematic viscosity, is that of the optimal parameter of linear elements.
- **The pressure level of an enclosed flow is fixed automatically.** When no
  boundary determines the pressure, Newton's method holds the pressure
  unknown nearest the centre of the fluid and shifts the converged pressure to
  a mean of zero over the fluid, with the reactions taken after the shift.
  The result is the same on any number of processes. A boundary velocity whose
  net flux is not zero is reported with its size, and a level that changes the
  flow (a property or source that depends on the pressure) is refused. A
  `point_Dirichlet` condition on the pressure still prescribes its value at a
  point.
- **A steady flow at rest starts from the Stokes solution**, the same problem
  without the inertia, when the velocity is zero wherever no condition
  prescribes it. A start set by the user is kept.
- **The initial temperature of `heat_conduction` is 293.15 K** (COMSOL's
  default).
- **Structural formulations of `solid_mechanics`.** The beams and plates are
  formulations of `solid_mechanics`: `formulation="plate"`, `"beam"` with
  `theory="Euler-Bernoulli"` or `"Timoshenko"` and, for Timoshenko,
  `beam_formulation="mixed"` or `"displacement"`, and `"circular_plate"` with
  `theory="first_order"` or `"classical"`. The variables of a plate are
  `disp_x`, `disp_y`, `disp_z`, `phi_x` and `phi_y`, of a beam `disp_x`,
  `disp_z` and `bending_moment` or `phi`, and of a circular plate `disp_r`,
  `disp_z` and `phi` or `bending_moment`. A section is given by its material
  (`youngs_modulus`, `poissons_ratio` and the `thickness` or the
  `cross_section`), integrated through the thickness at every integration
  point, so a modulus that depends on the height $z$ above the mid-surface or
  on the fields enters exactly, or by the section stiffnesses of its theory.
  The shell takes the section of the plate. `thermal_expansion` acts on the
  structures. An end without a condition is free, and the word `symmetry` on
  a beam end holds the axial displacement and the slope. A plate takes
  `Quad4` elements and a beam or circular plate `Edge2`, the elements that the
  one-point integration of the shear unlocks. `solid.stress(z=...)` gives the
  stress at the element centres, for a structure at a height above the
  mid-surface (a number, `"top"`, `"mid"` or `"bottom"`). Checked against the
  closed-form stiffnesses of graded sections, free thermal expansion and the
  thermal bending of a graded cantilever.
- **Poisson's ratio has no default value.** A continuum reads the property
  `poissons_ratio` of its property objects, and a plate, a circular plate and
  a Timoshenko beam given by its material, or a plate given by its section
  stiffnesses, need it given. `fgm.beam_stiffness` takes `poissons_ratio` as
  a required keyword.
- **One thermal strain**: `thermal_expansion_eigenstrain` takes
  `linear_expansion_coefficient` as a constant, a function or an expression of
  the temperature, also in a case file, with the secant definition
  $\varepsilon = \alpha(T)(T - T_0)$, where $\varepsilon$ is the thermal
  strain, $\alpha$ the mean coefficient between the stress-free temperature
  $T_0$ and the temperature $T$. The coupling passes it to the continuum and
  the structures alike. `parsed_eigenstrain` is for an eigenstrain given as an
  expression of the state, and refuses a `stress_free_state` whose expression
  reads a property.
- **One isotropic stress object.** `small_strain_stress` serves every
  formulation, and reads its moduli from the property objects in plane stress
  as in the other formulations. The thermal strain is an eigenstrain of
  `thermal_expansion_eigenstrain`. A property object may skip an output that
  no object reads, so `small_strain_stress` forms the deformation gradient and
  the von Mises stress only when they are read.
- **Plain names in the input.** The post-processors are `integral`,
  `average`, `maximum`, `minimum`, `point_value` and the flows through a
  boundary. `solve_transient` takes `time_step`, `time_scheme`
  (`"backward_Euler"`, the default, `"Crank-Nicolson"` or `"forward_Euler"`),
  `min_time_step` and `max_time_step` (formerly `dt`, `theta` or
  `implicitness`, `dt_min` and `dt_max`). The expansion coefficients name
  their kind: `volumetric_expansion_coefficient` for the Boussinesq buoyancy
  of a flow and `linear_expansion_coefficient` for a solid, and every `coeff`
  is `coefficient`. `add_point_source` takes no type name. The objects that
  compute named properties are property objects: `problem.add_property(type,
  name, ...)` (formerly `add_material`), category `property`, `PythonProperty`
  (formerly `PythonMaterial`) and the C++ base class `Property` (formerly
  `Material`). The word material is kept for physical substances.
- **Input without repetition.** The object types are named without their
  category: `Dirichlet`, `Neumann`, `Robin`, `point_Dirichlet`, `periodic`,
  `traction`, `pressure`, `symmetry`, `mass_flux`, `constant` and `parsed`
  (formerly `generic_constant_material` and `parsed_material`), and a
  misspelled name is answered with the closest registered one. `variable`
  takes a list, which adds one object per variable, and the component of a
  vector equation follows from its variable. A parameter and its `*_property`
  twin are one parameter that takes a number, a function, an expression or a
  property. Lists of names and values are dicts (`properties`, `constants`,
  `creep_constants`, `property_factors`, `stress_free_state`). Nodal
  conditions take no quadrature parameters.
- `time_derivative` takes its coefficient as an expression, which may depend
  on the fields.
- **Quadrature names are canonical and case-sensitive**: `automatic`, `gauss1`
  to `gauss20`, `midpoint`, `trapezoid`, `simpson`, `nodal`, `interface` and
  `control_domain_trapezoid`. A Gauss rule name with other characters after
  its number is refused.
- **`**` takes a float base with an `ADReal` exponent.**
- **The bounds of an axis** of `generate_rectangle_mesh` and
  `generate_box_mesh` are also given as a pair, `x=[0, 0.1]`, in place of
  `x_min` and `x_max` (both together are refused). Coordinates given with
  bounds are refused, where the bounds were ignored.
- **The default diagonal of a triangle mesh** is `"alternate"`, which keeps
  the mesh symmetric under reflection.
- `mesh_from_arrays` takes the `(element_type, nodes, block)` triples of
  `Mesh.cells()`, so a mesh is rebuilt from its own cells.
- `refined()` carries the side sets over in a time proportional to the mesh.
- The outward normal of an element side comes from the stored normal of the
  reference side, mapped by the element (the relation of Nanson), in place of
  a test against the element centroid. `write_sidesets` writes through meshio.
- **One input-error class**: `dualmesh.InputError`, raised by the C++ core and
  the Python input alike.
- **`dualmesh.SolveResult` is a Python class** that wraps the library's
  result, and the extension type is no longer patched at import.
- **The Python modules import in one direction**, from `_core` and `units` up
  through `model`, `physics` and `engine` to `input` and `case`, with no
  import cycles. `dualmesh.physics` is a package with one module per physics,
  and every public name keeps its import path.
- **The C++ libraries include in one direction**: `core`, `mesh`, `fe`,
  `linalg` and `base`, with `modules` and `parallel` on top and independent of
  each other. `ReferenceElement` moves to `mesh/` and `Communicator` to
  `linalg/`. The modules register their objects through `registerAllObjects()`
  (`dualmesh/modules/Registry.h`), which a C++ program calls once before it
  makes a `Problem`, and the Python module calls on import.
- **The flow and solid work leaves `Problem`**: the flow preconditioner
  blocks, the pressure level of an enclosed flow, the at-rest test and the
  projection of the initial velocity are in the fluid module behind the
  `FlowSystem` hook, and the near null space, the displacement groups, the
  crack tip integrals and the section stresses in the solid module behind
  `SolidSystem`. One C++ operator builder, `Problem::operatorOf`, serves the
  projection and the eigenvalue study. The solid mechanics source is split by
  responsibility within the one module, with shared declarations in
  `SolidMechanics.h`.
- **`Kernel` keeps the hooks every kernel needs.** The hooks of one family
  move to small base classes: `StorageKernel`, `MassEquation`,
  `AdvectiveKernel`, and `SectionKernel` in the solid module. The assembly
  finds them once per kernel group. The methods of `Problem` that initialise
  on first use are non-const. `dualmesh/core/Coordinates.h` holds the measure
  factor ($1$, $2\pi r$ or $4\pi r^2$, with $r$ the radius) and the hoop terms
  of a divergence and of a strain. The functions of the expression language
  have one table in C++.
- **C++20.** The core builds with the C++20 standard, so it needs a C++20
  compiler: GCC 10 or Clang 10, or newer.
- The Python code is formatted by `ruff format` with a line length of 100
  characters.
- The author of the software is Mohamed AbdulHameed, with his ORCID iD, in
  `CITATION.cff`, `pyproject.toml` and the documentation. The citation is the
  software alone.

### Removed

- **The vertex-centred (`"HFVM"`) and cell-centred (`"ZFVM"`) finite volume
  methods**, with `boundary_gradient`, `Problem.is_cell_centered`,
  `Problem.boundary_entities` and the method aliases `fvm`, `finite_volume`,
  `vertex_finite_volume`, `finite_element`, `dual_mesh` and
  `cell_finite_volume`. Two methods remain: FEM, the more accurate for solids
  (every element type, the structures and Taylor-Hood), and DMCDM for fluids,
  locally conservative.
- **The built-in linear solvers**, with the options `linear_solver` and
  `preconditioner`, the function `have_petsc` and the CMake option
  `DUALMESH_ENABLE_PETSC`. PETSc solves every system, and `petsc_options`
  chooses its solver.
- `absolute_tolerance` of `solve` and `solve_transient`: it had a unit, and the
  convergence of each equation against its own terms replaces it.
- `load_factors` of `solve` and `scale_with_load` of every object: the solve
  continues a problem by itself, and whether an object is data is a property
  of its type.
- The solver option `verbose` (replaced by `report`), and the
  `output_interval` and `output_file_base` parameters of
  `Problem.solve_transient` (replaced by `output=dualmesh.Output(...)`).
- **The kernels `heat_convection` and `convective_inertia`**, replaced by
  `advection`, and the parameter `stabilization` of `mass_conservation`: the
  pressure stabilisation is always on for the equal-order element and off for
  Taylor-Hood.
- `linear_elastic_stress`, merged into `small_strain_stress`, with its
  untested orthotropic option (`stiffness_c11` to `stiffness_c66`).
- `heat_flux_boundary_condition`, `convective_heat_flux_boundary_condition`
  and the `heat_source` kernel: use `Neumann`, `Robin` and `body_force`, or a
  condition written as text.
- `gap_contact` (penalty contact), replaced by mortar contact, and
  `gas_gap_heat_transfer`.
- The classes `mms.Term`, `mms.Diffusion`, `mms.Advection`, `mms.Reaction`,
  `mms.TimeDerivative`, `mms.LinearElasticity` and `mms.IncompressibleFlow`.
  `mms.ManufacturedSolution` takes the physics of the physics level
  (`study.add_physics(...)`), and the boundaries that get the exact flux in
  place of the exact solution are its `flux_boundaries`.
- The helpers `dualmesh.physics.add_plane_elasticity`,
  `add_incompressible_flow`, `add_boussinesq_buoyancy`, `add_beam`,
  `add_plate` and `add_circular_plate`, replaced by the physics level.
- The Python class `DistributedProblem`, replaced by `Problem`, which splits
  itself under `mpirun` (or with `distributed=True`), and the C++
  `DistributedOptions.verbose`.
- `dualmesh.postprocess` (sample a profile with `Problem.sample` or a `line`
  postprocessor of the input, and compute orders with
  `dualmesh.mms.convergence_rates`),
  and `Mesh.vtk_node_order`, which was the identity for every element.
- `dualmesh.input_error` (use `dualmesh.InputError`) and `dualmesh.pow` (write
  `**`).
- `Problem.values_at_nodes`: `values(v)[nodes]` gives it.
- `save_state` and `restore_state`.
- **Second spellings of names.** The coordinate systems `xyz`, `rz` and
  `cylindrical` (use `cartesian` and `axisymmetric`), the element names of
  Exodus and meshio (`QUAD4`, `triangle`, `Prism6` and the rest: use `Quad4`,
  `Tri3` and `Wedge6`, and `read_mesh` converts the names of a file), the
  quadrature aliases `auto`, `trapezoidal`, `lumped`, `centroid`,
  `cd_trapezoid` and a bare `gauss`, and the spellings `sym`, `dev`, `det`,
  `ddot` and `if` of the expression language.
- **Publication on PyPI**: the publish job of the Wheels workflow, the PyPI
  gates of `scripts/release.py` and the installation from PyPI. dualmesh is
  installed from source, and the wheels of a release are the artifacts of its
  tag build.

### Fixed

- **Spherical flows carry their hoop terms.** The divergence of the velocity
  includes $2 v_r / r$, with $v_r$ the radial velocity, and the radial
  momentum equation the hoop stresses of both hoop directions, where the flow
  kernels treated spherical coordinates as Cartesian.
- **The penalty pressure of axisymmetric and spherical flows under DMCDM** is
  the pressure of the discrete equations, to second order, where it was wrong
  by a multiple of $\gamma h^2$, with $\gamma$ the penalty parameter and $h$
  the element size: the hoop term of the divergence takes the radius of the
  point where the reduced rule evaluates the velocity.
- **The stabilisation of a flow of several blocks** takes the density and
  viscosity of each block, where it took the first block's everywhere.
- **Lumped weights in axisymmetric and spherical coordinates** integrate the
  coordinate factor: $\int_{CD_a} 2\pi r \, dA$ for DMCDM and
  $\int \psi_a \, 2\pi r \, dA$ for FEM ($4\pi r^2$ in spherical
  coordinates), with $CD_a$ the control domain of node $a$ and $\psi_a$ its
  shape function. They used the Cartesian measure times the factor at the
  node, which gave zero storage and source at nodes on the axis or at the
  centre. The node measures and the reported boundary flux follow.
- **Buoyancy needs the density of the fluid.** A flow without a positive
  density no longer gets $\rho = 1$ in its Boussinesq force, and a gravity
  of the wrong length is refused.
- **`gap_heat_transfer` refuses an emissivity outside $(0, 1]$** and one
  emissivity without the other.
- **The theta method with an equation that has no time derivative.** The mass
  equation of an incompressible flow and the equilibrium of a quasi-static
  solid are taken at the new time with the weight 1 for every time scheme, and
  so are the pressure gradient and the stabilisation of a flow. Before,
  Crank-Nicolson made the pressure an average of two steps, whose error
  changed sign at every step and grew ($5.2 \times 10^{-2}$ on a flow that
  backward Euler gives to round-off), and it evaluated creep at the old time
  without the material history (a strain 185 % too large). An explicit step
  solves such an equation at the new time.
- **The time derivative inside a streamline stabilisation with
  Crank-Nicolson.** `momentum_stabilization` and
  `heat_streamline_stabilization` put the time derivative inside a steady
  term, which the theta method weighted by $\theta$ and evaluated again at the
  old time without it. The derivative now enters with the weight 1. The
  explicit step (`forward_Euler`) and the eigenvalue study refuse these
  kernels and the stabilised `mass_conservation` of a flow with inertia. A
  stabilised Stokes flow has no such derivative and keeps both.
- **No sliver step before an output or end time.** A target less than two
  steps away is reached in two equal steps, and the steps after a step
  shortened to land on a target grow back by at most `growth_factor` each,
  within what the error estimate allows. Before, a step of 0.01 to the end
  time 0.1003 ended with a step of $3 \times 10^{-4}$, and the step after
  such a sliver was up to 33 times larger, past the ratio $1 + \sqrt{2}$ at
  which the backward difference of order 2 is stable.
- **`pow` of a zero or negative base.** In an equation or a property,
  `pow(u, 2)` gave NaN where $u \le 0$, because its derivatives went through
  $\exp(2 \ln u)$. It now differentiates as $p u^{p-1}$ for a constant
  exponent $p$, as `u^2` does.
- **Adaptive marking at the maximum refinement level.** The marker picked the
  cells of the largest error, which near a singularity are the cells at the
  maximum level, and then dropped them, so the loop stopped while other cells
  still held error. The marker now sees only the cells that can be refined.
- **Unconverged return maps are rejected.** The Newton iterations of the
  creep and plasticity return maps, and of the zero out-of-plane stress of
  plane stress, ended at their iteration limit without an error. A state they
  do not solve is now rejected, and the nonlinear solve takes a smaller step.
  They judge convergence on the full Newton step and on the stress scale.
- **Distributed FEM with hanging nodes.** Each process folds the equation of a
  hanging node or a periodic copy into its primaries before the processes add
  their residuals, as it already did with the Jacobian. Before, the solver
  read a row from the process that owns it, which need not hold the
  constraint that folds into the row, so the patch test on a locally refined
  mesh missed the linear field on several processes. The reactions read the
  residual folded the same way.
- **`Mesh.refined()` and `Mesh.second_order()` refuse a mesh with hanging
  nodes**, which they would have left without constraints.
- **FEM refuses the nodal and interface rules on `Tri6`, `Tet10`, `Quad8` and
  `Hex20`**, where a corner shape function integrates to zero or less.
- **`generate_annulus_mesh` names the sides of a sector** `inner`, `outer`,
  `start` and `end`, and checks its arguments.
- **Input checks that let bad values through:** a boolean parameter takes only
  true or false, an integer parameter refuses a boolean and any real that is
  not whole, `add_function` refuses a name in use, and `plate_first_order`
  refuses `axial_load` (use `in_plane_load`).
- Numbers outside the range a parameter states are refused when the object is
  made, naming the parameter and the value: a non-positive diffusivity,
  thermal conductivity, viscosity, specific heat, thickness or penalty, and a
  negative density, emissivity, transfer coefficient or foundation modulus.
  A negative diffusivity solved the anti-diffusion problem without a word.
- A mesh without elements, a mesh that reaches $x < 0$ in axisymmetric or
  spherical coordinates, and a second `Dirichlet` value for the same variable
  on the same boundary are refused. The second value silently replaced the
  first, and the negative radius failed later in the nonlinear solver.
- A singular system (a part of the mesh that no boundary condition holds,
  such as a floating body or a region without any temperature condition) is
  reported with the variables and the extent of that part. It was solved
  without an error and returned zero there. The check runs again when the
  pattern of the Jacobian changes.
- A traction or pressure condition took the thickness of the first
  `stress_divergence` kernel of its displacement, whatever its block. It now
  refuses kernels of different thicknesses, and `stress_divergence` refuses a
  thickness in a three-dimensional or axisymmetric problem.
- The plane-stress `volumetric_strain` left out the out-of-plane strain, which
  the `strain` property now holds.
- Four of the five sides of the `Wedge6` element were stored in an order whose
  right-hand normal pointed into the prism, so `write_sidesets` wrote those
  faces wound inward. Every side of every element type is now wound outward,
  which a test checks against the divergence theorem.
- `read_mesh` found a physical name by its tag alone. Gmsh numbers its
  physical groups separately in each dimension, so a boundary could take the
  name of a volume group with the same tag. Names are now found by tag and
  dimension.
- Promotion to second order, uniform refinement and adaptive refinement each
  grew the node sets by their own rule, and the midpoint of an interior chord
  between two boundary nodes joined a boundary node set. One rule now serves
  all three: a node set gains an added node whose parents are all members,
  and a set on the boundary gains only added nodes on the boundary.
- `add_sideset_from_faces` kept the faces it had found when another face was
  not on the boundary. It now refuses the whole set and adds nothing.
- An `Expression` of no variables raised an `IndexError` when it was
  evaluated. It now returns its value.
- `error_norms` of a distributed problem returned the norm of the part of each
  process. The squares now add over the processes, so a convergence study
  gives the same numbers on any number of them.
- The error-controlled time step (`time_stepper="error"`) estimated its error
  with the norm of each process, so the processes could disagree on a step
  and wait for each other forever. The estimate uses the global norm.
- A run on several processes could hang when Python's cycle collector freed a
  distributed problem, whose MPI, PETSc and gslib objects every process must
  destroy together. A collected distributed problem is now kept until the next
  collective point (a new problem, a solve, or the end of the run), where
  every process destroys the kept problems in the order of their creation.
- An error raised inside the threaded assembly reached Python as a
  `RuntimeError` instead of the `InputError` it was.
- The message of an error of an object names its type and name once.
- The object-level `beam_Timoshenko_displacement` described itself as
  integrating the shear at the centre, which only the one-point rule of its
  rotation equation does. The description says so.

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
