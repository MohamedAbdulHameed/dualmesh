# dualmesh

**A multiphysics framework for heat transfer, solid mechanics and fluid dynamics, with the dual mesh control domain method as an option.**

`dualmesh` solves coupled boundary value problems: any number of fields
(temperatures, displacements, velocities, or quantities of your own), each
governed by a conservation law built from named terms that may depend on every
field, solved together as one monolithic system by Newton's method with an
exact Jacobian from automatic differentiation. The same problem description is
discretised, by changing one keyword, with the Galerkin **finite element
method**, the vertex-centred or cell-centred **finite volume method**, or the
**dual mesh control domain method** (DMCDM) of J. N. Reddy. The core is C++17;
everything is driven from Python.

The structure follows the [MOOSE framework](https://mooseframework.inl.gov):
physics is added as **kernels**, **boundary conditions** and **materials**,
which are registered objects with validated, self-documenting parameters;
meshes are generated or read from standard files; and the solvers are shared
by every physics. See [`docs/scope.rst`](docs/scope.rst) for what that covers
and an honest comparison with MOOSE and COMSOL Multiphysics.

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

- **Methods**: Galerkin finite elements, vertex-centred and cell-centred
  finite volumes, and the dual mesh control domain method, selected by one
  argument. A method refuses, with the reason, an element or a physics it
  cannot treat correctly.
- **Elements**: `Edge2`, `Edge3`, `Tri3`, `Tri6`, `Quad4`, `Quad8`, `Quad9`,
  `Tet4`, `Tet10`, `Hex8`, `Hex20`, `Hex27`, `Wedge6`, `Pyramid5`, mixed freely
  in one mesh, on straight or distorted meshes, in 1D, 2D and 3D.
- **Coordinate systems**: Cartesian, axisymmetric (`2πr`), spherical (`4πr²`).
- **Physics modules**
  - *heat transfer*: conduction with temperature-dependent conductivity,
    volumetric heating, capacity, convection by a computed flow, convective,
    radiative and flux boundaries;
  - *solid mechanics*: linear elasticity in plane stress, plane strain,
    axisymmetric and three-dimensional form with thermal strain, tractions and
    pressures; and the structural members: mixed Euler–Bernoulli beams,
    displacement and mixed Timoshenko beams, axisymmetric circular plates and
    rectangular plates, functionally graded sections, von Kármán nonlinearity;
  - *fluid dynamics*: Stokes and Navier–Stokes flow by the penalty formulation
    with recovered pressure, and Boussinesq buoyancy;
  - *framework*: diffusion, anisotropic diffusion, reaction, advection, body
    force, time derivative, coupled force, Dirichlet/Neumann/Robin conditions,
    point sources, generic materials, and expressions such as
    `"sin(pi*x)*exp(-t)"` compiled to C++.
- **Coupled problems**: monolithic and fully coupled, verified on the natural
  convection benchmark of de Vahl Davis (Nusselt numbers within 0.2 %).
- **Solvers**: Newton's method with exact Jacobians from forward-mode automatic
  differentiation, direct (Picard) iteration with relaxation, load stepping,
  steady and transient (θ-method, adaptive step size) executioners. Linear
  systems are solved directly where that is cheap and by BiCGSTAB or GMRES with
  an ILU(0) preconditioner where it is not, chosen automatically.
- **Parallel**: threaded assembly, and an MPI solver with a two-level
  overlapping Schwarz preconditioner whose iteration count does not grow with
  the number of processes.
- **Quadrature per kernel**: Gauss rules, midpoint, trapezoid, Simpson, nodal
  lumping, interface and control-domain-trapezoid rules, and selective reduced
  integration for locking and penalty terms.
- **Meshing**: generators (line, rectangle, box, annulus, graded spacing),
  local and uniform refinement, and readers for Gmsh, Exodus, VTK, Abaqus and
  more through [meshio](https://github.com/nschloe/meshio).
- **Verification**: Reddy's book examples, analytical solutions, OpenFOAM
  cross-checks, and a method-of-manufactured-solutions study of the order of
  convergence of every method on every element type.
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
von Kármán), circular plates and rectangular plates. Natural convection is
checked against de Vahl Davis (1983), and a manufactured-solution study checks
the convergence order of every method and element type. See
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
