# SPDX-License-Identifier: LGPL-2.1-or-later
"""Ready-made physics: one call adds all the kernels of a model.

The models of the structural, solid mechanics, and fluids modules need one
kernel per equation (and, for shear-deformable plates, a second kernel for the
transverse shear terms so that they can be integrated with a reduced rule).
These helpers add them consistently, in the spirit of MOOSE's Physics syntax::

    import dualmesh as dm

    problem = dm.Problem(mesh)
    dm.physics.add_plane_elasticity(
        problem, youngs_modulus=30e6, poissons_ratio=0.25, thickness=0.036)
"""

from __future__ import annotations

from collections.abc import Sequence


def _ensure_variable(problem, name: str) -> None:
    """Add a variable unless the problem already has it."""
    try:
        problem.variable_index(name)
    except ValueError:
        problem.add_variable(name)


def add_plane_elasticity(
    problem,
    displacements: Sequence[str] = ("displacement_x", "displacement_y"),
    youngs_modulus: float = 1.0,
    poissons_ratio: float = 0.0,
    formulation: str = "plane_stress",
    thickness: float = 1.0,
    body_force: Sequence[float] | None = None,
    name: str = "elasticity",
    **material_parameters,
):
    """Add the two (or three) equilibrium equations of linear elasticity.

    Creates the displacement variables if they do not exist yet, a
    ``LinearElasticStress`` material, and one ``StressDivergence`` kernel per
    component.  ``formulation`` is ``"plane_stress"``, ``"plane_strain"``,
    ``"axisymmetric"``, or ``"three_dimensional"``.
    """
    displacements = list(displacements)
    for variable in displacements:
        _ensure_variable(problem, variable)
    problem.add_material(
        "LinearElasticStress",
        f"{name}_material",
        displacements=displacements,
        formulation=formulation,
        youngs_modulus=youngs_modulus,
        poissons_ratio=poissons_ratio,
        **material_parameters,
    )
    for component, variable in enumerate(displacements):
        problem.add_kernel(
            "StressDivergence",
            f"{name}_equilibrium_{variable}",
            variable=variable,
            component=component,
            thickness=thickness,
        )
        if body_force is not None and body_force[component] != 0.0:
            problem.add_kernel(
                "BodyForce",
                f"{name}_body_force_{variable}",
                variable=variable,
                value=thickness * body_force[component],
            )
    return displacements


def add_incompressible_flow(
    problem,
    velocities: Sequence[str] = ("velocity_x", "velocity_y"),
    dynamic_viscosity: float = 1.0,
    density: float = 0.0,
    penalty_parameter: float = 1.0e8,
    body_force: Sequence[float] | None = None,
    name: str = "flow",
):
    """Add the penalty momentum equations of an incompressible flow.

    With ``density = 0`` the equations are the Stokes equations; a positive
    density adds the convective term of the Navier-Stokes equations.  A
    ``PenaltyPressure`` material provides the recovered pressure as the
    material property ``"pressure"``.
    """
    velocities = list(velocities)
    for variable in velocities:
        _ensure_variable(problem, variable)
    for component, variable in enumerate(velocities):
        problem.add_kernel(
            "ViscousStress",
            f"{name}_viscous_{variable}",
            variable=variable,
            component=component,
            velocities=velocities,
            dynamic_viscosity=dynamic_viscosity,
        )
        problem.add_kernel(
            "PenaltyIncompressibility",
            f"{name}_penalty_{variable}",
            variable=variable,
            component=component,
            velocities=velocities,
            penalty_parameter=penalty_parameter,
        )
        if density:
            problem.add_kernel(
                "ConvectiveInertia",
                f"{name}_inertia_{variable}",
                variable=variable,
                component=component,
                velocities=velocities,
                density=density,
            )
        if body_force is not None and body_force[component] != 0.0:
            problem.add_kernel(
                "BodyForce",
                f"{name}_body_force_{variable}",
                variable=variable,
                value=body_force[component],
            )
    problem.add_material(
        "PenaltyPressure",
        f"{name}_pressure",
        velocities=velocities,
        penalty_parameter=penalty_parameter,
    )
    return velocities


def add_beam(
    problem,
    model: str = "BeamEulerBernoulliMixed",
    axial_displacement: str = "axial_displacement",
    transverse_displacement: str = "deflection",
    third_variable: str | None = None,
    name: str = "beam",
    **parameters,
):
    """Add a beam model (three kernels, one per variable).

    ``model`` is ``"BeamEulerBernoulliMixed"``, ``"BeamTimoshenkoMixed"``, or
    ``"BeamTimoshenkoDisplacement"``.  For the displacement Timoshenko model
    the shear term of the rotation equation is integrated with a reduced rule,
    which is what prevents shear locking.
    """
    displacement_model = model == "BeamTimoshenkoDisplacement"
    if third_variable is None:
        third_variable = "rotation" if displacement_model else "bending_moment"
    variables = [axial_displacement, transverse_displacement, third_variable]
    for variable in variables:
        _ensure_variable(problem, variable)
    mapping = dict(
        axial_displacement=axial_displacement,
        transverse_displacement=transverse_displacement,
    )
    mapping["rotation" if displacement_model else "bending_moment"] = third_variable
    for variable in variables:
        options = dict(parameters, **mapping, variable=variable)
        if displacement_model and variable == third_variable:
            options.update(quadrature="midpoint", reduced_integration=True)
        problem.add_kernel(model, f"{name}_{variable}", **options)
    return variables


def add_circular_plate(
    problem,
    radial_displacement: str = "radial_displacement",
    transverse_displacement: str = "deflection",
    rotation: str = "rotation",
    name: str = "plate",
    **parameters,
):
    """Add the first-order (Mindlin) axisymmetric circular plate model.

    Two kernels are added per variable: one for the bending and membrane terms
    and one for the transverse shear force, the latter with reduced integration.
    The problem must use ``coordinates="axisymmetric"`` on a radial mesh.
    """
    variables = [radial_displacement, transverse_displacement, rotation]
    for variable in variables:
        _ensure_variable(problem, variable)
    mapping = dict(
        radial_displacement=radial_displacement,
        transverse_displacement=transverse_displacement,
        rotation=rotation,
    )
    for variable in variables:
        problem.add_kernel(
            "CircularPlateFirstOrder",
            f"{name}_bending_{variable}",
            variable=variable,
            shear_treatment="exclude",
            **dict(parameters, **mapping),
        )
        problem.add_kernel(
            "CircularPlateFirstOrder",
            f"{name}_shear_{variable}",
            variable=variable,
            shear_treatment="only",
            quadrature="midpoint",
            reduced_integration=True,
            **dict(parameters, **mapping),
        )
    return variables


def add_plate(
    problem,
    in_plane_displacements: Sequence[str] = ("displacement_x", "displacement_y"),
    transverse_displacement: str = "deflection",
    rotations: Sequence[str] = ("rotation_x", "rotation_y"),
    name: str = "plate",
    **parameters,
):
    """Add the first-order (Mindlin) rectangular plate model (five variables).

    Two kernels are added per variable: one for the bending and membrane terms
    and one for the transverse shear forces, the latter with reduced
    integration, which removes shear locking in thin plates.
    """
    in_plane_displacements = list(in_plane_displacements)
    rotations = list(rotations)
    variables = in_plane_displacements + [transverse_displacement] + rotations
    for variable in variables:
        _ensure_variable(problem, variable)
    mapping = dict(
        in_plane_displacements=in_plane_displacements,
        transverse_displacement=transverse_displacement,
        rotations=rotations,
    )
    for variable in variables:
        problem.add_kernel(
            "PlateFirstOrder",
            f"{name}_bending_{variable}",
            variable=variable,
            shear_treatment="exclude",
            **dict(parameters, **mapping),
        )
        problem.add_kernel(
            "PlateFirstOrder",
            f"{name}_shear_{variable}",
            variable=variable,
            shear_treatment="only",
            quadrature="midpoint",
            reduced_integration=True,
            **dict(parameters, **mapping),
        )
    return variables
