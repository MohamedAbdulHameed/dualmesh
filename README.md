# dualmesh

dualmesh solves coupled problems of heat transfer, solid mechanics and incompressible flow, and partial differential equations that you write yourself. It discretizes a problem by the finite element method (FEM), and a problem with a flow by the dual mesh control domain method (DMCDM), which conserves the transported quantity in every control domain. The `method` argument chooses the other method where it applies. The core is C++17 with PETSc and MPI, and you drive it from Python or from a YAML case file.

## A first case

A slab $`L = 0.1\ \mathrm{m}`$ thick, with thermal conductivity $`k = 45\ \mathrm{W/(m\,K)}`$, generates heat at $`q = 1\ \mathrm{MW/m^3}`$ and is held at $`T_0 = 300\ \mathrm{K}`$ on both faces. Its temperature is $`T(x) = T_0 + q x (L - x)/(2k)`$.

Write the case as a YAML file, `slab.yaml`:

```yaml
mesh:
  segment: {length: 0.1 m, cells: 20}
physics:
  heat_conduction: {}
properties:
  thermal_conductivity: 45 W/(m*K)
  volumetric_heat_source: 1 MW/m^3
boundary_conditions:
  left: {temperature: 300 K}
  right: {temperature: 300 K}
postprocessors:
  middle: {point_value: temperature, point: [0.05 m]}
```

Solve it with the command line tool:

```bash
dualmesh run slab.yaml
```

Or build the same case in Python:

```python
import dualmesh

slab = dualmesh.problem(mesh={"segment": {"length": "0.1 m", "cells": 20}})
slab.physics("heat_conduction")
slab.properties(thermal_conductivity="45 W/(m*K)", volumetric_heat_source="1 MW/m^3")
slab.boundary_conditions("left", temperature="300 K")
slab.boundary_conditions("right", temperature="300 K")
slab.postprocessor("middle", point_value="temperature", point=["0.05 m"])
solution = slab.solve()
print(solution.postprocessor("middle")[-1])  # 327.78 K, the exact T0 + q L^2 / (8 k)
```

Each dimensional value carries its unit, which the input converts to SI units.

## What it solves

- **Heat transfer**: conduction with a conductivity that may depend on the temperature, heat sources, storage, prescribed temperatures and fluxes, convection, radiation, and heat transfer across gaps.
- **Incompressible flow**: the equal-order formulation with pressure stabilization, Taylor-Hood elements and the penalty formulation, steady or transient. Transient flow runs monolithically or by a projection (splitting) scheme. A heated flow under gravity takes the Boussinesq buoyancy force.
- **Solid mechanics**:
  - small and finite strain, in plane strain, plane stress, generalized plane strain, axisymmetric and three-dimensional bodies;
  - anisotropic elasticity, thermal expansion, creep, plasticity, viscoelasticity and phase-field fracture;
  - cohesive zones, contact and inertia.
- **Structures**: Euler-Bernoulli and Timoshenko beams, trusses, plates and shells.
- **Your own equations**: systems of partial differential equations written as text, which the code reduces to fluxes and sources and differentiates for the Jacobian of Newton's method.
- **Eigenvalue problems**: natural frequencies, buckling loads, and the criticality of a reactor.

Every problem solves its fields together by Newton's method, with the exact Jacobian from automatic differentiation.

## Features

- Elements in one, two and three dimensions, linear and quadratic, mixed in one mesh, with Cartesian, axisymmetric and spherical coordinates.
- Meshes from the built-in generators or from the file formats that [meshio](https://github.com/nschloe/meshio) reads (Gmsh, Exodus, VTK, Abaqus and others).
- Adaptive mesh refinement and coarsening:
  - by bisection of triangles and tetrahedra, and by subdivision of quadrilaterals and hexahedra with hanging nodes;
  - in steady and transient runs, on curved boundaries and in parallel.
- Linear solvers from PETSc, chosen for each system: a direct factorization where it is cheap, otherwise a Krylov method with a preconditioner that suits the physics. Any PETSc solver can be chosen by its options.
- Parallel runs: threads in the assembly, and MPI across processes, with the shared values exchanged through PETSc or gslib.
- Results as VTU files for ParaView and CSV tables, with units in the column names.

## The dual mesh control domain method

Each physics states a balance law

```math
\frac{\partial m}{\partial t} + \nabla \cdot \mathbf{F} = s \quad \text{in } \Omega ,
```

where $`m`$ is the stored quantity, $`\mathbf{F}`$ its flux and $`s`$ its source. The primal mesh of finite elements interpolates the unknowns. The dual mesh joins the midpoints of the edges and the centroids of the faces and elements into a control domain $`V_i`$ around each node $`i`$. DMCDM states the balance over each control domain,

```math
\int_{V_i} \frac{\partial m}{\partial t} \, \mathrm{d}V + \oint_{\partial V_i} \mathbf{F} \cdot \mathbf{n} \, \mathrm{d}S = \int_{V_i} s \, \mathrm{d}V ,
```

with $`\mathbf{n}`$ the outward unit normal. The flux leaving one control domain enters its neighbour, so the method conserves the quantity locally. The fluxes on the interfaces come from the finite element interpolation, so no gradient reconstruction is needed.

## Installation

The build needs a C++17 compiler on Linux or macOS, CMake 3.18 or later, Python 3.11 or later, MPI and PETSc. With conda:

```bash
conda install -c conda-forge openmpi petsc pkg-config
```

Then install from a clone of the repository:

```bash
pip install ".[all]"
```

The package imports as `dualmesh` and installs the command `dualmesh`.

## Documentation

The documentation is in [`docs/`](docs/): installation, a quickstart, the user guide with the reference of every input key, tutorials, the theory and the verification against published results.

## Citation

If this software contributes to your work, cite it as [`CITATION.cff`](CITATION.cff) describes.

## License

GNU Lesser General Public License, version 2.1 or later. See [`LICENSE`](LICENSE).
