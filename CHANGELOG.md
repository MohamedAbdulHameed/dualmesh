# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project uses
[semantic versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

dualmesh is now described, and organised, as what it is: a multiphysics
framework for heat transfer, solid mechanics and fluid dynamics, in which the
finite element method, two finite volume methods and the dual mesh control
domain method are four discretisations of one problem description.

### Added

- **Four new element types**: the serendipity `Quad8` and `Hex20`, the
  triangular prism `Wedge6` and the pyramid `Pyramid5` (with Bedrosian's
  rational basis). Meshes may mix types freely. Every type works with the
  finite element and cell-centred finite volume methods; the dual mesh and
  vertex-centred methods, which need node-centred control domains, accept the
  prism (whose dual is the triangle's median dual times a half-segment) and
  refuse the serendipity elements and the pyramid with the reason and the
  methods that do accept them. `Mesh.second_order(serendipity=True)`,
  generators for all four, uniform refinement of prisms and pyramids, and
  meshio reading and writing.
- **Systematic verification by manufactured solutions**: `dualmesh.mms`
  derives the forcing of a chosen solution symbolically (diffusion,
  advection, reaction, time derivative, linear elasticity; Cartesian,
  axisymmetric and spherical) and runs convergence studies;
  `Problem.error_norms` measures the L2 error and the H1 seminorm. The suite
  runs 161 studies and every observed order matches the theory of its method.
- **Coupled heat transfer and flow**: `HeatConvection` (the flow carries the
  heat) and `BoussinesqBuoyancy` (the temperature drives the flow), with
  `physics.add_boussinesq_buoyancy`. The natural convection benchmark of de
  Vahl Davis is reproduced to within 0.2 % in the Nusselt number
  (`examples/natural_convection.py`).
- **Overlapping Schwarz**: the distributed preconditioners now work on
  subdomains that overlap by `overlap` layers of elements (default 1), with
  fully assembled subdomain matrices exchanged between ranks, an exact or
  incomplete subdomain solve (`subdomain_solver`), and the two levels combined
  multiplicatively. The iteration count of the default no longer grows with
  the number of processes: 23 to 25 iterations from 1 to 16 ranks on a 48 by 48
  Poisson problem. With `linear_solver="cg"` the symmetric (classical) forms
  are used.
- **Automatic linear solver** (`linear_solver="automatic"`, the new default):
  a direct factorisation where it is cheap and BiCGSTAB with a new ILU(0)
  preconditioner where it is not, falling back to the direct solver if the
  iteration fails. Also `"gmres"` and the `preconditioner` option (`ilu`,
  `ilut`, `jacobi`, `none`). On a three-dimensional elasticity system of 36 000
  unknowns the solve fell from 18 s to 0.3 s.
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
  framework and compares it honestly with MOOSE and COMSOL; a
  manufactured-solutions chapter; the new elements, solvers and
  preconditioners; the current MOOSE citation (MOOSE 4.0, 2025).

### Changed

- **The structural module is merged into solid mechanics.** Beams and plates
  are the reduced theories of solid mechanics and are now registered in the
  `solid_mechanics` module, next to continuum elasticity.
- **The default distributed preconditioner is `two_level_schwarz` with an
  overlap of one element layer** (it was `jacobi`).
- **The cell-centred finite volume method has an exact Jacobian**: the
  non-orthogonal correction is differentiated through the least-squares
  stencils instead of being lagged, so Newton's method converges
  quadratically on skewed meshes.
- **Efficiency**: element geometry is mapped once per integration point and
  reference-element data are cached; the derivative arrays of the automatic
  differentiation are copied only as far as they are used and their loops
  vectorise; symmetric quadrature rules on triangles, tetrahedra and prisms
  replace the collapsed tensor rules for the finite element method (8 points
  instead of 27 on a tetrahedron); Gauss rules are tabulated. On the bundled
  benchmark (three-dimensional elasticity on 8000 `Hex8` elements) the
  assembly of every method is about three times faster on one thread; the
  Python test suite runs in about 60 % of its former time.
- All element types use the VTK node numbering, so files need no permutation.
- Package metadata points to the GitHub repository.

### Fixed

- **The θ-method evaluated the old part of the residual at the new time**,
  which cost Crank-Nicolson its second order whenever a coefficient or a
  source depends on time (serial and distributed).
- **The Schwarz preconditioners factorised each rank's partial local matrix**
  instead of the fully assembled subdomain matrix, which is why they stopped
  converging at eight ranks.
- **The penalty formulation of incompressible flow locked on the
  vertex-centred finite volume method**: the two-point gradient correction was
  applied to the reduced-integration term. It is no longer, and the
  cell-centred method, on which the formulation cannot work, now refuses it.
- **A compiled expression passed as a parameter was evaluated through
  Python**, because pybind11's function test is Python's `callable()`, which
  also serialised the assembly onto one thread.
- **Thermal stress in plane strain was wrong by a factor of `1 + nu`.** The
  stiffness matrix left the out-of-plane column empty, so the thermal strain
  subtracted from the out-of-plane normal strain never reached the in-plane
  rows. For a fully restrained body the code gave
  `E alpha dT / [(1 + nu)(1 - 2 nu)]` where the correct value is
  `E alpha dT / (1 - 2 nu)`. Plane stress was unaffected.
- **A Python callable passed directly to a parameter no longer deadlocks the
  threaded assembly.**
- **The cell-centred finite volume method is refused by the distributed
  solver** instead of returning a wrong answer.
- **The distributed BiCGSTAB detects near-breakdown and restarts.**
- The Python test suite no longer imports the source tree instead of the
  installed package when the extension has not been built in place.

### Earlier unreleased work

- **Quadratic elements** `Edge3`, `Tri6`, `Quad9`, `Tet10` and `Hex27`, with
  the dual mesh built for each; **adaptive mesh refinement** by gradient
  recovery, marking and longest-edge bisection; the automatic quadrature
  order; `DUALMESH_MAX_AD_DERIVATIVES`; the manual reorganised into a user
  guide and a theory manual with every method cited to its primary source.

## [0.1.0] - 2026-09-18

The first release: the framework, five physics modules, and a verification
suite that reproduces the published results of Reddy's book.

### Added

- **Framework**: the canonical conservation form `-div F + S = 0`; the median
  dual mesh for `Edge2`, `Tri3`, `Quad4`, `Tet4` and `Hex8`; assembly for the
  dual mesh control domain method and for the Galerkin finite element method
  from the same kernels; forward-mode automatic differentiation for exact
  Jacobians; per-kernel quadrature (Gauss, midpoint, trapezoid, Simpson,
  nodal, interface, control-domain trapezoid) and selective reduced
  integration; Cartesian, axisymmetric and spherical coordinates; blocks
  (subdomains), side sets and node sets.
- **Solvers**: Newton's method, direct (Picard) iteration with relaxation,
  load stepping, steady and transient (θ-method) executioners, sparse LU,
  BiCGSTAB and conjugate gradient linear solvers.
- **Heat transfer module**: conduction with temperature-dependent
  conductivity, volumetric heating, heat capacity, convective, radiative and
  prescribed-flux boundary conditions.
- **Solid mechanics module**: linear elasticity (plane stress, plane strain,
  axisymmetric, three-dimensional), traction and pressure boundary conditions,
  stress and strain as material properties.
- **Structural module**: mixed Euler–Bernoulli beam, displacement and mixed
  Timoshenko beams, first-order axisymmetric circular plate, first-order
  rectangular plate, functionally graded sections, von Kármán nonlinearity.
- **Fluids module**: penalty formulation of the Stokes and Navier–Stokes
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
