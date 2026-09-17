# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project uses
[semantic versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
