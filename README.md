# dualmesh

**The dual mesh control domain method for computational fluid dynamics and solid mechanics.**

`dualmesh` solves boundary value problems of heat transfer, solid mechanics,
structural mechanics and viscous incompressible flow with the *dual mesh control
domain method* (DMCDM) of J. N. Reddy, and — from the same problem definition —
with the standard Galerkin finite element method, so that the two can be
compared directly. The core is C++17; everything is driven from Python.

The structure follows the [MOOSE framework](https://mooseframework.inl.gov):
physics is added as **kernels**, **boundary conditions** and **materials**,
which are registered objects with validated, self-documenting parameters;
meshes are generated or read from standard files; and the solvers (Newton with
exact automatic differentiation, direct iteration, load stepping, time
integration) are shared by every physics module.

```python
import dualmesh as dm

mesh = dm.generate_rectangle_mesh(
    x_min=0.0, x_max=0.1, y_min=0.0, y_max=0.05,
    num_x_elements=10, num_y_elements=5)

problem = dm.Problem(mesh, method="dmcdm")      # or method="fem"
problem.add_variable("temperature")
problem.add_kernel("HeatConduction", variable="temperature", thermal_conductivity=20.0)
problem.add_kernel("HeatSource", variable="temperature", heat_source=1.0e6)
problem.add_boundary_condition("DirichletBC", variable="temperature", boundary="left", value=40.0)
problem.add_boundary_condition("DirichletBC", variable="temperature", boundary="right", value=10.0)
problem.add_boundary_condition("ConvectiveHeatFluxBC", variable="temperature",
                               boundary="top", heat_transfer_coefficient=75.0)
problem.solve()

problem.sample("temperature", [[0.05, 0.0]])    # 83.142 (Reddy, Table 5.4.3)
problem.total_reaction("temperature", "left")   # heat flow through the left face
problem.write_vtu("bus_bar.vtu")
```

## What the method is

The **primal mesh** is a mesh of finite elements and supplies the interpolation.
The **dual mesh** is the set of node-centred *control domains* built from edge
midpoints, face centroids and element centroids. The governing equation is
satisfied in the integral sense over each control domain, with no weight
function:

```
  -∮(∂CD) F·n dS + ∫(CD) S dV = 0
```

so the secondary variables (fluxes, forces, moments) appear naturally on the
control-domain interfaces — the physical appeal of the finite volume method —
while the primal interpolation removes the ad-hoc gradient reconstructions that
the finite volume method needs. Every equation in the library is written in the
canonical form `-div F + S = 0`, which is why the same kernels can be
discretized either way.

## Features

- **Methods**: dual mesh control domain method and Galerkin finite elements,
  selected by one argument.
- **Elements**: `Edge2`, `Tri3`, `Quad4`, `Tet4`, `Hex8`, on straight or
  distorted meshes, in 1D, 2D and 3D.
- **Coordinate systems**: Cartesian, axisymmetric (`2πr`), spherical (`4πr²`).
- **Physics modules**
  - *heat transfer*: conduction with temperature-dependent conductivity,
    volumetric heating, capacity, convective, radiative and flux boundaries;
  - *solid mechanics*: linear elasticity in plane stress, plane strain,
    axisymmetric and three-dimensional form, tractions and pressures;
  - *structural*: mixed Euler–Bernoulli beams, displacement and mixed
    Timoshenko beams, axisymmetric circular plates and rectangular plates
    (first-order shear deformation), functionally graded sections, von Kármán
    nonlinearity;
  - *fluids*: Stokes and Navier–Stokes flow by the penalty formulation with
    recovered pressure;
  - *framework*: diffusion, anisotropic diffusion, reaction, advection, body
    force, time derivative, coupled force, Dirichlet/Neumann/Robin conditions,
    point sources, generic materials.
- **Solvers**: Newton's method with exact Jacobians from forward-mode automatic
  differentiation, direct (Picard) iteration with relaxation, load stepping,
  steady and transient (θ-method) executioners, sparse LU / BiCGSTAB / CG.
- **Quadrature per kernel**: Gauss rules, midpoint, trapezoid, Simpson, nodal
  lumping, interface and control-domain-trapezoid rules, and selective reduced
  integration for locking and penalty terms.
- **Meshing**: generators (line, rectangle, box, annulus, graded spacing) and
  readers for Gmsh, Exodus, VTK, Abaqus and more through
  [meshio](https://github.com/nschloe/meshio).
- **Extensible from Python**: kernels, boundary conditions and materials can be
  written in Python and still get exact derivatives.
- **Input files**: `dualmesh run input.yaml`, plus `dualmesh list` and
  `dualmesh describe <type>` for the object reference.

## Verification

Every worked example of Reddy's book with published dual mesh results is
reproduced by the test suite — one-dimensional and two-dimensional conduction,
axisymmetric conduction, advection–diffusion at high Péclet number, nonlinear
conduction, plane elasticity, pressurized cylinders, squeezed flow, the
lid-driven cavity at Re = 0 and Re = 1000, functionally graded beams (linear and
von Kármán), circular plates and rectangular plates. See
[`docs/verification.rst`](docs/verification.rst) and `tests/python`. The suite
also cross-checks against OpenFOAM (see `verification/openfoam`).

## Installation

```console
pip install .[all]           # from a clone
pytest                       # run the verification suite
```

Requirements: a C++17 compiler, CMake ≥ 3.18, Python ≥ 3.9. Eigen and pybind11
are found if installed and downloaded otherwise.

Documentation: <https://dualmesh.readthedocs.io>

## Citing

If this software contributes to your work, please cite the method

> J. N. Reddy, *Computational Methods in Engineering: Finite Difference, Finite
> Volume, Finite Element, and Dual Mesh Control Domain Methods*, CRC Press,
> 2024,

and the software itself; see `CITATION.cff`.

## License

GNU Lesser General Public License, version 2.1 or later (the licence of the
MOOSE framework, whose object model inspired this one). See `LICENSE`.
