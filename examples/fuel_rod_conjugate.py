# SPDX-License-Identifier: LGPL-2.1-or-later
"""A fuel rod cooled by a flowing coolant: conjugate heat transfer.

The fuel, the gap, the cladding and the coolant are solved together, in one
Newton iteration: heat conduction in the fuel and cladding with their
temperature-dependent properties, heat transfer across the pellet-cladding
gap (gas_gap_heat_transfer), and the incompressible Navier-Stokes equations in
the pressure-velocity formulation with heat convection in the coolant, which
shares its nodes with the cladding surface.  The linear systems are solved
with PETSc when the extension has it.

The rod is a uranium nitride rod cooled by liquid sodium, the combination
of a liquid-metal fast reactor, with approximate sodium properties at about
700 K (k = 70 W/m/K, rho = 850 kg/m^3, c_p = 1270 J/kg/K, mu = 2.5e-4 Pa s).
The cladding is Zircaloy-4, because it is the one cladding the library
models; a fast reactor would use a steel.  The coolant channel is the
annulus of a square lattice cell with the same flow area.  The flow is
laminar (Reynolds number about 800) because the library has no turbulence
model, and a real sodium coolant flows faster and is turbulent.  The large
thermal diffusivity of sodium keeps the cell Peclet number below one, so the
energy equation needs no upwinding on this mesh.  The example therefore
demonstrates, and the test suite checks, the coupling and its conservation:
the heat generated in the fuel leaves with the coolant.
"""

from __future__ import annotations

import dualmesh as dm
import numpy as np
from dualmesh import fuel


def build(method: str = "fem", linear_heat_rate: float = 5000.0, inlet_velocity: float = 0.02):
    geometry = fuel.RodGeometry.from_diameters(pellet_outer_diameter=8.19e-3, clad_inner_diameter=8.36e-3, clad_outer_diameter=9.50e-3, fuel_stack_height=0.05)
    pitch = 12.6e-3
    # The annulus with the flow area of a square lattice cell of this pitch.
    r_eq = pitch / np.sqrt(np.pi)
    mesh = fuel.axisymmetric_rod_mesh(geometry, fuel.RodMesh(num_fuel_radial_elements=10, num_clad_radial_elements=3, num_axial_elements=20), num_coolant_radial_elements=8, coolant_outer_radius=r_eq)
    p = dm.Problem(mesh, method=method, coordinates="axisymmetric")
    t_inlet, rho, cp, k_sodium, mu = 670.0, 850.0, 1270.0, 70.0, 2.5e-4
    q3 = linear_heat_rate / (np.pi * geometry.pellet_outer_radius**2)
    # The thermal conductivity of every block comes from its material.
    p.add_property("UN_thermal", "fuel_thermal", block=["fuel"], temperature="temperature")
    p.add_property("Zircaloy_thermal", "clad_thermal", block=["clad"], temperature="temperature")
    p.add_property("constant_property", "sodium", block=["coolant"], property_names=["thermal_conductivity"], property_values=[k_sodium])
    heat = p.add_physics("heat_transfer", "heat", initial_condition=t_inlet)
    heat.add_kernel("heat_source", "fission", heat_source=q3, block=["fuel"], scale_with_load=True)
    heat.add_boundary_condition("gas_gap_heat_transfer", "gap", boundary=["fuel_outer"], secondary_boundary=["clad_inner"], gas_pressure=2.0e6)
    # The coolant.
    flow = p.add_physics("incompressible_flow", "flow", velocities=["u", "v"], dynamic_viscosity=mu, density=rho, formulation="pressure", block=["coolant"], mass_flux_boundaries=["coolant_inlet", "coolant_outlet", "coolant_outer"])
    p.add_coupling("nonisothermal_flow", "coupling", heat_transfer="heat", incompressible_flow="flow", specific_heat=cp)
    flow.add_boundary_condition("Dirichlet_boundary_condition", "coolant_inlet", value=[0.0, inlet_velocity], scale_with_load=True)
    flow.add_boundary_condition("symmetry_boundary_condition", "coolant_outer")
    # No slip on the cladding (added last, so that the corner nodes shared
    # with the inlet belong to the wall).
    flow.add_boundary_condition("Dirichlet_boundary_condition", "clad_outer", value=[0.0, 0.0])
    heat.add_boundary_condition("Dirichlet_boundary_condition", "coolant_inlet", value=t_inlet)
    info = dict(power=linear_heat_rate * geometry.fuel_stack_height, rho=rho, cp=cp, t_inlet=t_inlet, r_in=geometry.clad_outer_radius, r_out=r_eq, top=geometry.fuel_stack_height)
    return p, info


def energy_balance(p, info, samples: int = 400):
    """Heat carried out through the outlet by the coolant, rho c_p v (T - T_in)
    integrated over the outlet annulus (W), and the mixed-mean outlet
    temperature."""
    r = np.linspace(info["r_in"], info["r_out"], samples)
    points = [[ri, info["top"] * (1 - 1e-9), 0.0] for ri in r]
    v = p.sample("v", points)
    T = p.sample("temperature", points)
    integrand = info["rho"] * info["cp"] * v * (T - info["t_inlet"]) * 2 * np.pi * r
    heat = float(np.sum(0.5 * (integrand[1:] + integrand[:-1]) * np.diff(r)))
    flow = v * 2 * np.pi * r
    mass = float(np.sum(0.5 * (flow[1:] + flow[:-1]) * np.diff(r)))
    return heat, info["t_inlet"] + heat / (info["rho"] * info["cp"] * mass)


if __name__ == "__main__":
    options = {"linear_solver": "petsc"} if dm.have_petsc() else {}
    # The cell-centred method is not included: the fluid-solid interface is
    # interior to the mesh, and that method has no unknowns on interior faces
    # to hold the no-slip condition.
    for method in ("fem", "dmcdm", "hfvm"):
        p, info = build(method)
        # The power and the flow are ramped up together in load steps: from
        # the uniform inlet temperature and a fluid at rest the full load
        # would overshoot in the first Newton steps.
        result = p.solve(load_factors=[0.1, 0.3, 0.6, 1.0], report="none", **options)
        heat, mixed = energy_balance(p, info)
        centre = p.sample("temperature", [[0.0, 0.5 * info["top"], 0.0]])[0]
        inlet_heat = -p.total_reaction("temperature", "coolant_inlet")
        print(f"{method:6s} Newton iterations {result.total_iterations}, centreline {centre:.1f} K, heat to coolant {heat:.2f} W of {info['power']:.2f} W ({100 * heat / info['power']:.2f} %) plus {inlet_heat:.2f} W conducted back through the inlet, mixed outlet {mixed:.2f} K")
