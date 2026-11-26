"""The physics of the input, with their options and the properties they read.

The schema reads the options of a physics from the parameters of its function after
`properties`: a parameter without a default is a key the input must give, and the schema fills
every other default, except None, which leaves the choice to the engine. Each function returns
the standard names of the properties that the physics reads, split into those it needs and those
it reads when they are given, so that the front end can say which property is missing and refuse
a property that no physics reads. The build adds the properties that a physics needs from the
same lists.
"""

from dualmesh.physics import FLOW_FORMULATIONS

kinematics_choices = ("small", "finite")
solid_formulations = ("plane_strain", "plane_stress", "axisymmetric", "three_dimensional")


def heat_conduction(properties):
    """The energy balance, with the flux of Fourier's law.

    A problem with time terms, a transient run or an eigenvalue study, reads the density and the
    specific heat capacity, and so does a flow that carries the heat.  A steady body at rest
    reads neither.
    """
    needed = ["thermal_conductivity"]
    if properties.transient or properties.flow:
        needed += ["density", "specific_heat_capacity"]
    return {"needed": needed, "optional": ["volumetric_heat_source"]}


def solid_mechanics(properties, formulation=None, kinematics="small", strain_energy="neo-Hookean"):
    """The balance of momentum of an elastic solid.

    `formulation` is plane_strain or plane_stress on a plane mesh. It follows from the mesh
    in three dimensions and in axisymmetric coordinates, where the input leaves it out.
    `kinematics` is small or finite strain. At finite strain, `strain_energy` is neo-Hookean
    (the default), Saint_Venant_Kirchhoff, or a formula of I_C, trace_C_squared, J,
    shear_modulus and first_Lame_parameter in J/m^3. A problem with time terms reads the density,
    for the inertia, and so does a problem with gravity, for the weight. A yield_stress, a formula
    of equivalent_plastic_strain and temperature, makes a small-strain solid J2-plastic.
    """
    needed = ["Young_modulus", "Poisson_ratio"]
    if properties.transient or properties.gravity:
        needed += ["density"]
    optional = [
        "body_force_density",
        "linear_expansion_coefficient",
        "stress_free_temperature",
        "yield_stress",
    ]
    return {"needed": needed, "optional": optional}


def fluid_dynamics(properties, formulation=None):
    """The balance of mass and momentum of an incompressible Newtonian fluid.

    `formulation` is equal_order, with a pressure stabilization, Taylor-Hood, a quadratic
    velocity and a linear pressure on a quadratic mesh with FEM, or penalty, the pressure
    eliminated by a penalty on the divergence. Left out, the physics chooses it: Taylor-Hood
    with FEM on a mesh of quadratic cells, and equal_order otherwise.  A heated fluid under gravity
    reads its volumetric expansion coefficient and the reference temperature at which its
    density holds, for the Boussinesq buoyancy.
    """
    needed = ["density", "dynamic_viscosity"]
    if properties.heated and properties.gravity:
        needed += ["volumetric_expansion_coefficient", "reference_temperature"]
    return {"needed": needed, "optional": ["body_force_density"]}


physics_functions = {
    "heat_conduction": heat_conduction,
    "solid_mechanics": solid_mechanics,
    "fluid_dynamics": fluid_dynamics,
}

# The values that each option of a physics takes.
option_choices = {
    "solid_mechanics": {"formulation": solid_formulations, "kinematics": kinematics_choices},
    "fluid_dynamics": {"formulation": FLOW_FORMULATIONS},
}
# The options that a physics reads only when another option has one value, with the reason:
# option -> (other option, value, reason).
read_only_with = {
    "solid_mechanics": {
        "strain_energy": (
            "kinematics",
            "finite",
            "at small strain every energy is linear elasticity with the Young modulus and the "
            "Poisson ratio",
        )
    }
}
