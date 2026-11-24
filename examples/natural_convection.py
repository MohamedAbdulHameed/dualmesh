# SPDX-License-Identifier: LGPL-2.1-or-later
"""Natural convection in a differentially heated square cavity.

This is the benchmark of de Vahl Davis (1983), and it is a two-way coupled
multiphysics problem: the temperature drives the flow through the buoyancy
force, and the flow carries the heat.  The three equations are solved together,
by Newton's method with the exact Jacobian of the coupled system, including the
off-diagonal blocks that couple the velocity to the temperature and back.

In the non-dimensional form of the benchmark, with the cavity width L, the
thermal diffusivity kappa and the temperature difference Delta T as scales,

    (v . grad) v = -grad p + Pr div(grad v) + Ra Pr T e_y ,   div v = 0 ,
    v . grad T = div(grad T) ,

with the temperature +1/2 on the left wall and -1/2 on the right, adiabatic
top and bottom walls, and no slip everywhere.  Pr = 0.71 (air), and the
Rayleigh number Ra measures the strength of the buoyancy.  The pressure is
eliminated by the penalty method.  The Rayleigh number is reached by load
stepping, which multiplies the buoyancy force.

The average Nusselt number on the hot wall is the total heat flow through it,
which is the sum of the reactions of the temperature equation there.
"""

import dualmesh
import numpy

Prandtl_number = 0.71

#: de Vahl Davis (1983), Table: average Nusselt number, the largest horizontal
#: velocity on the vertical centreline and the largest vertical velocity on the
#: horizontal centreline.
benchmark = {
    1.0e3: {"nusselt": 1.118, "u_max": 3.649, "v_max": 3.697},
    1.0e4: {"nusselt": 2.243, "u_max": 16.178, "v_max": 19.617},
    1.0e5: {"nusselt": 4.519, "u_max": 34.73, "v_max": 68.59},
    1.0e6: {"nusselt": 8.800, "u_max": 64.63, "v_max": 219.36},
}


def cavity_mesh(num_elements=32):
    """A square mesh whose lines cluster at the walls (a cosine spacing), where
    the thermal and viscous boundary layers are."""
    s = numpy.linspace(0.0, 1.0, num_elements + 1)
    x = 0.5 * (1.0 - numpy.cos(numpy.pi * s))
    return dualmesh.generate_rectangle_mesh(x_coordinates=list(x), y_coordinates=list(x))


def solve(Rayleigh_number=1.0e5, num_elements=32, method="DMCDM", advection_scheme=None):
    mesh = cavity_mesh(num_elements)
    problem = dualmesh.Problem(mesh, method=method)
    flow = problem.add_physics(
        "fluid_dynamics",
        dynamic_viscosity=Prandtl_number,
        density=1.0,
        formulation="penalty",
        penalty_parameter=1.0e7,
    )
    heat = problem.add_physics(
        "heat_transfer", thermal_conductivity=1.0, advection_scheme=advection_scheme
    )
    problem.add_coupling(
        "nonisothermal_flow",
        heat_transfer=heat,
        fluid_dynamics=flow,
        gravity=[0.0, -1.0],
        volumetric_expansion_coefficient=Rayleigh_number * Prandtl_number,
    )
    flow.add_boundary_condition(
        ["velocity_x = 0", "velocity_y = 0"], boundary=["left", "right", "bottom", "top"]
    )
    heat.add_boundary_condition("temperature = 0.5", boundary="left")
    heat.add_boundary_condition("temperature = -0.5", boundary="right")
    result = problem.solve(report="none")
    return problem, result


def measure(problem):
    """The three benchmark quantities."""
    s = numpy.linspace(0.0, 1.0, 801)
    u = problem.sample("velocity_x", numpy.column_stack([numpy.full_like(s, 0.5), s]))
    v = problem.sample("velocity_y", numpy.column_stack([s, numpy.full_like(s, 0.5)]))
    return {
        "nusselt": -problem.boundary_flux("temperature", "left"),
        "u_max": float(numpy.nanmax(u)),
        "v_max": float(numpy.nanmax(v)),
    }


if __name__ == "__main__":
    print("   Ra      quantity   dualmesh   de Vahl Davis   difference")
    for Rayleigh_number, reference in benchmark.items():
        problem, result = solve(Rayleigh_number)
        computed = measure(problem)
        for key, value in computed.items():
            difference = 100.0 * (value - reference[key]) / reference[key]
            print(
                f"{Rayleigh_number:7.0e}  {key:9s}  {value:9.3f}  {reference[key]:13.3f}   {difference:+6.2f} %"
            )
