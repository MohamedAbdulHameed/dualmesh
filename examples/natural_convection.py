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

import dualmesh as dm
import numpy as np

PRANDTL = 0.71

#: de Vahl Davis (1983), Table: average Nusselt number, the largest horizontal
#: velocity on the vertical centreline and the largest vertical velocity on the
#: horizontal centreline.
BENCHMARK = {
    1.0e3: {"nusselt": 1.118, "u_max": 3.649, "v_max": 3.697},
    1.0e4: {"nusselt": 2.243, "u_max": 16.178, "v_max": 19.617},
    1.0e5: {"nusselt": 4.519, "u_max": 34.73, "v_max": 68.59},
    1.0e6: {"nusselt": 8.800, "u_max": 64.63, "v_max": 219.36},
}


def cavity_mesh(num_elements=32):
    """A square mesh whose lines cluster at the walls (a cosine spacing), where
    the thermal and viscous boundary layers are."""
    s = np.linspace(0.0, 1.0, num_elements + 1)
    x = 0.5 * (1.0 - np.cos(np.pi * s))
    return dm.generate_rectangle_mesh(x_coordinates=list(x), y_coordinates=list(x))


def solve(rayleigh_number=1.0e5, num_elements=32, method="dmcdm"):
    mesh = cavity_mesh(num_elements)
    problem = dm.Problem(mesh, method=method)
    velocities = dm.physics.add_incompressible_flow(
        problem,
        velocities=["u", "v"],
        dynamic_viscosity=PRANDTL,
        density=1.0,
        penalty_parameter=1.0e7,
    )
    problem.add_variable("temperature")
    problem.add_kernel("HeatConduction", "conduction", variable="temperature")
    problem.add_kernel(
        "HeatConvection", "convection", variable="temperature", velocities=velocities
    )
    problem.add_kernel(
        "BoussinesqBuoyancy",
        "buoyancy",
        variable="v",
        component=1,
        temperature="temperature",
        gravity=[0.0, -1.0],
        thermal_expansion=rayleigh_number * PRANDTL,
        scale_with_load=True,
    )
    for variable in velocities:
        problem.add_boundary_condition(
            "DirichletBC",
            f"no_slip_{variable}",
            variable=variable,
            boundary=["left", "right", "bottom", "top"],
            value=0.0,
        )
    problem.add_boundary_condition(
        "DirichletBC", "hot", variable="temperature", boundary="left", value=0.5
    )
    problem.add_boundary_condition(
        "DirichletBC", "cold", variable="temperature", boundary="right", value=-0.5
    )
    # Continuation in the Rayleigh number, one decade at a time.
    steps = [ra / rayleigh_number for ra in (1e3, 1e4, 1e5, 1e6) if ra <= rayleigh_number]
    result = problem.solve(load_factors=steps, max_iterations=40)
    return problem, result


def measure(problem):
    """The three benchmark quantities."""
    s = np.linspace(0.0, 1.0, 801)
    u = problem.sample("u", np.column_stack([np.full_like(s, 0.5), s]))
    v = problem.sample("v", np.column_stack([s, np.full_like(s, 0.5)]))
    return {
        "nusselt": problem.total_reaction("temperature", "left"),
        "u_max": float(np.nanmax(u)),
        "v_max": float(np.nanmax(v)),
    }


if __name__ == "__main__":
    print("   Ra      quantity   dualmesh   de Vahl Davis   difference")
    for rayleigh_number, reference in BENCHMARK.items():
        problem, result = solve(rayleigh_number)
        computed = measure(problem)
        for key, value in computed.items():
            difference = 100.0 * (value - reference[key]) / reference[key]
            print(
                f"{rayleigh_number:7.0e}  {key:9s}  {value:9.3f}  {reference[key]:13.3f}"
                f"   {difference:+6.2f} %"
            )
