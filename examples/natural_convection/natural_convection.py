# SPDX-License-Identifier: LGPL-2.1-or-later
"""Natural convection of air in a square cavity heated from one side: the benchmark of de Vahl
Davis (1983).

A square cavity of side L = 100 mm holds air at 300 K. Its left wall is warmer than its right
wall by Delta T, its top and bottom walls are insulated, and the air sticks to every wall. The
temperature drives the flow through the Boussinesq buoyancy force, and the flow carries the
heat, so the problem is coupled both ways. The Rayleigh number Ra = g beta Delta T L^3 / (nu
kappa), with the gravitational acceleration g, the expansion coefficient beta = 1/(300 K) of an
ideal gas, the kinematic viscosity nu and the thermal diffusivity kappa, sets the strength of the
convection, and the Prandtl number nu / kappa of air is 0.71. Each Rayleigh number takes its own
Delta T, from 0.01 K for Ra = 10^3 to 10.6 K for Ra = 10^6.

The mesh of make_cavity_mesh.py clusters its cells at the walls, where the thermal and viscous
boundary layers are. The flow takes the default equal-order formulation. The benchmark gives
three numbers without a unit: the average Nusselt number Nu = Q / (k Delta T), with the heat
flow Q through the hot wall per unit depth and the conductivity k, and the largest velocities
u L / kappa along the two centre lines.
"""

from pathlib import Path

import dualmesh
import numpy

side = 0.1  # m
# Air at 300 K: the conductivity in W/(m K), the density in kg/m^3, the specific heat
# capacity in J/(kg K) and the dynamic viscosity in Pa s, whose Prandtl number is 0.71.
conductivity, density, heat_capacity, viscosity = 0.0263, 1.177, 1007.0, 1.854e-5
expansion, gravity = 1.0 / 300.0, 9.81  # 1/K, m/s^2
diffusivity = conductivity / (density * heat_capacity)  # m^2/s

#: de Vahl Davis (1983): the average Nusselt number, the largest horizontal velocity on the
#: vertical centre line and the largest vertical velocity on the horizontal centre line.
benchmark = {
    1.0e3: {"nusselt": 1.118, "u_max": 3.649, "v_max": 3.697},
    1.0e4: {"nusselt": 2.243, "u_max": 16.178, "v_max": 19.617},
    1.0e5: {"nusselt": 4.519, "u_max": 34.73, "v_max": 68.59},
    1.0e6: {"nusselt": 8.800, "u_max": 64.63, "v_max": 219.36},
}


def temperature_difference(Rayleigh_number: float) -> float:
    """The difference Delta T in K between the walls that gives the Rayleigh number."""
    kinematic_viscosity = viscosity / density
    return Rayleigh_number * kinematic_viscosity * diffusivity / (gravity * expansion * side**3)


def cavity(Rayleigh_number: float, method: str | None = None):
    half = temperature_difference(Rayleigh_number) / 2.0
    air = dualmesh.problem(
        mesh={"file": str(Path(__file__).with_name("cavity.msh")), "length_unit": "mm"},
        method=method,
    )
    air.physics("fluid_dynamics")
    air.physics("heat_conduction")
    air.properties(
        thermal_conductivity=f"{conductivity} W/(m*K)",
        density=f"{density} kg/m^3",
        specific_heat_capacity=f"{heat_capacity} J/(kg*K)",
        dynamic_viscosity=f"{viscosity} Pa*s",
        volumetric_expansion_coefficient=f"{expansion!r} 1/K",
        reference_temperature="300 K",
    )
    air.gravity(["0 m/s^2", f"{-gravity} m/s^2"])
    for wall in ("left", "right", "bottom", "top"):
        air.boundary_conditions(wall, velocity=["0 m/s", "0 m/s"])
    air.boundary_conditions("left", temperature=f"{300.0 + half!r} K")
    air.boundary_conditions("right", temperature=f"{300.0 - half!r} K")
    air.postprocessor("heat_in", boundary_flux="heat_flux", boundary="left")
    air.postprocessor("heat_out", boundary_flux="heat_flux", boundary="right")
    centre, length = ["50 mm", "0 mm"], ["50 mm", "100 mm"]
    air.postprocessor("vertical", line="velocity_x", from_=centre, to=length, points=801)
    air.postprocessor(
        "horizontal", line="velocity_y", from_=centre[::-1], to=length[::-1], points=801
    )
    air.output(file="natural_convection", fields=["velocity", "pressure", "temperature"])
    return air


def measure(solution, Rayleigh_number: float) -> dict:
    """The three numbers of the benchmark: the heat that enters through the hot wall, as a
    Nusselt number, and the largest velocities on the centre lines in units of kappa / L."""
    heat = -solution.postprocessor("heat_in")[-1]
    scale = diffusivity / side
    return {
        "nusselt": heat / (conductivity * temperature_difference(Rayleigh_number)),
        "u_max": float(numpy.nanmax(solution.line("vertical")["velocity_x"])) / scale,
        "v_max": float(numpy.nanmax(solution.line("horizontal")["velocity_y"])) / scale,
    }


if __name__ == "__main__":
    print("   Ra      quantity   dualmesh   de Vahl Davis   difference")
    for Rayleigh_number, reference in benchmark.items():
        computed = measure(cavity(Rayleigh_number).solve(report="none"), Rayleigh_number)
        for key, value in computed.items():
            difference = 100.0 * (value - reference[key]) / reference[key]
            print(
                f"{Rayleigh_number:7.0e}  {key:9s}  {value:9.3f}  {reference[key]:13.3f}"
                f"   {difference:+6.2f} %"
            )
    print("wrote natural_convection.vtu, natural_convection.csv and the centre lines")
