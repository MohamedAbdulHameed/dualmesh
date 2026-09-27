# SPDX-License-Identifier: LGPL-2.1-or-later
"""Ready-made physics: one call adds all the kernels of a model.

The models of the solid mechanics and fluids modules need one kernel per
equation (and, for shear-deformable plates, a second kernel for the
transverse shear terms so that they can be integrated with a reduced rule).
These helpers add them consistently, in the spirit of MOOSE's Physics syntax::

    import dualmesh as dm

    problem = dm.Problem(mesh)
    dm.physics.add_plane_elasticity(
        problem, youngs_modulus=30e6, poissons_ratio=0.25, thickness=0.036)
"""

from __future__ import annotations

from collections.abc import Sequence


def _ensure_variable(problem, name: str, block: Sequence[str] = (), order: str = "mesh") -> None:
    """Add a variable unless the problem already has it; an existing variable
    must have been declared with the same order."""
    try:
        problem.variable_index(name)
    except ValueError:
        problem.add_variable(name, block=list(block), order=order)
        return
    existing = problem.variable_order(name)
    if existing != order:
        raise ValueError(
            f"The variable '{name}' already exists with order '{existing}', but this equation "
            f"needs order '{order}'. Declare it with order='{order}', or let the helper add it."
        )


def _is_quadratic_mesh(mesh) -> bool:
    """Whether every element of the mesh is quadratic."""
    linear = {"Edge2", "Tri3", "Quad4", "Tet4", "Hex8", "Wedge6", "Pyramid5"}
    return all(mesh.element_type(e) not in linear for e in range(mesh.num_elements))


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
    ``linear_elastic_stress`` material, and one ``stress_divergence`` kernel per
    component.  ``formulation`` is ``"plane_stress"``, ``"plane_strain"``,
    ``"axisymmetric"``, or ``"three_dimensional"``.
    """
    displacements = list(displacements)
    for variable in displacements:
        _ensure_variable(problem, variable)
    problem.add_material(
        "linear_elastic_stress",
        f"{name}_material",
        displacements=displacements,
        formulation=formulation,
        youngs_modulus=youngs_modulus,
        poissons_ratio=poissons_ratio,
        **material_parameters,
    )
    for component, variable in enumerate(displacements):
        problem.add_kernel(
            "stress_divergence",
            f"{name}_equilibrium_{variable}",
            variable=variable,
            component=component,
            thickness=thickness,
        )
        if body_force is not None and body_force[component] != 0.0:
            problem.add_kernel(
                "body_force",
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
    formulation: str = "penalty",
    pressure: str = "pressure",
    stabilization: bool | None = None,
    streamline_stabilization: bool | None = None,
    buoyancy: dict | None = None,
    pressure_pin_point: Sequence[float] | None = None,
    mass_flux_boundaries: Sequence[str] | None = None,
    block: Sequence[str] | None = None,
):
    r"""Add the equations of an incompressible flow of a Newtonian fluid.

    ``formulation="penalty"`` (the default, the formulation of Chapter 9 of
    Reddy's book) eliminates the pressure with the penalty relation
    :math:`p = -\gamma \nabla \cdot v`: the momentum equations are the only
    equations, the penalty term is integrated with a reduced rule, and a
    ``penalty_pressure`` material recovers the pressure as the material
    property ``"pressure"``.  It works with the dual mesh, finite element and
    vertex-centred finite volume methods.

    ``formulation="pressure"`` keeps the pressure as an unknown, the variable
    named ``pressure``, and adds the mass conservation equation.  Velocity and
    pressure are interpolated with the same shape functions, and the mass
    equation carries the residual-based pressure stabilisation
    (``stabilization=True``, the default) that makes this equal-order
    interpolation stable.  It works with all four methods.
    ``streamline_stabilization`` adds SUPG to the momentum equations; by
    default it is on when the flow has inertia (``density > 0``).

    ``formulation="taylor_hood"`` is the pressure formulation with the
    Taylor-Hood element: quadratic velocity and linear pressure, the pressure
    variable declared with ``order="first"``.  The pair satisfies the discrete
    inf-sup condition, so it needs no stabilisation, and on smooth solutions
    it converges at third order in the velocity and second order in the
    pressure (in the :math:`L^2` norm).  It needs the finite element method
    (``method="fem"``) and a quadratic mesh (``mesh.second_order()``);
    ``stabilization`` and ``streamline_stabilization`` do not apply and are
    refused if set to True.  The linear systems are saddle point systems with
    a zero pressure block, which the default linear solver factorises
    directly.

    In both pressure formulations a ``mass_flux_boundary_condition`` is applied on
    ``mass_flux_boundaries`` (all side sets by default), which is where the
    mass equation needs the actual flow through the boundary.

    With ``density = 0`` the equations are the Stokes equations; a positive
    density adds the convective term of the Navier-Stokes equations.
    ``body_force`` is a constant force per unit volume.  ``buoyancy`` adds a
    Boussinesq force: a dictionary with the keys ``temperature``, ``gravity``,
    ``thermal_expansion_coefficient``, ``reference_temperature`` and optionally
    ``scale_with_load``; in the pressure formulation it is included in the
    stabilisation as well.  ``pressure_pin_point`` fixes the pressure to zero at the
    entity nearest to a point, which an enclosed flow (no outlet, every
    boundary a wall) needs because its pressure is otherwise determined only
    up to a constant.  ``block`` restricts the flow to the named element
    blocks (the fluid of a conjugate heat transfer problem); the velocity and
    pressure variables are then defined on those blocks only.  Returns the
    list of velocity names.
    """
    blocks = list(block) if block is not None else []
    restrict = {"block": blocks} if blocks else {}
    velocities = list(velocities)
    formulations = ("penalty", "pressure", "taylor_hood")
    if formulation not in formulations:
        raise ValueError(
            f"Unknown flow formulation '{formulation}'. Use one of: {', '.join(formulations)}."
        )
    taylor_hood = formulation == "taylor_hood"
    pressure_formulation = formulation in ("pressure", "taylor_hood")
    if taylor_hood:
        if problem.method not in ("fem", "finite_element"):
            raise ValueError(
                "The Taylor-Hood element is a finite element construction; create the problem "
                f"with method='fem' (it has method='{problem.method}'). The other methods use "
                "formulation='pressure', the stabilised equal-order element."
            )
        if not _is_quadratic_mesh(problem.mesh):
            raise ValueError(
                "The Taylor-Hood element needs quadratic elements (quadratic velocity, linear "
                "pressure), and this mesh has linear ones. Generate a quadratic mesh (for "
                "example element_type='Quad9', 'Tri6', 'Tet10' or 'Hex27'), or promote this one "
                "with mesh.second_order() before creating the problem."
            )
        if stabilization:
            raise ValueError(
                "The Taylor-Hood element is stable without pressure stabilisation; leave "
                "'stabilization' unset (or False) with formulation='taylor_hood'."
            )
        if streamline_stabilization:
            raise ValueError(
                "Streamline stabilisation (SUPG) is not available with the Taylor-Hood element: "
                "its momentum residual omits the viscous term, which is not zero for a quadratic "
                "velocity, so SUPG would no longer be consistent and would cost accuracy. Use a "
                "finer mesh for a convection-dominated flow, or formulation='pressure'."
            )
        stabilization = False
        streamline_stabilization = False
    elif not pressure_formulation and (stabilization or streamline_stabilization):
        raise ValueError(
            "'stabilization' and 'streamline_stabilization' apply to the pressure formulations "
            "only; the penalty formulation has no pressure unknown to stabilise."
        )
    if stabilization is None:
        stabilization = True
    for variable in velocities:
        _ensure_variable(problem, variable, blocks)
    if pressure_formulation:
        _ensure_variable(problem, pressure, blocks, order="first" if taylor_hood else "mesh")
    if streamline_stabilization is None:
        streamline_stabilization = pressure_formulation and stabilization and density != 0.0
    force = [0.0] * len(velocities)
    if body_force is not None:
        force = [float(f) for f in body_force][: len(velocities)]
        force += [0.0] * (len(velocities) - len(force))
    residual_parameters = {
        "velocities": velocities,
        "density": density,
        "dynamic_viscosity": dynamic_viscosity,
    }
    if any(f != 0.0 for f in force):
        residual_parameters["body_force"] = [repr(f) for f in force]
    if buoyancy is not None:
        residual_parameters.update(
            temperature=buoyancy["temperature"],
            gravity=list(buoyancy["gravity"]),
            thermal_expansion_coefficient=buoyancy["thermal_expansion_coefficient"],
            reference_temperature=buoyancy["reference_temperature"],
            buoyancy_scale_with_load=bool(buoyancy.get("scale_with_load", False)),
            buoyancy_density=float(density) if density else 1.0,
        )
    for component, variable in enumerate(velocities):
        problem.add_kernel(
            "viscous_stress",
            f"{name}_viscous_{variable}",
            variable=variable,
            component=component,
            velocities=velocities,
            dynamic_viscosity=dynamic_viscosity,
            **restrict,
        )
        if pressure_formulation:
            problem.add_kernel(
                "pressure_gradient",
                f"{name}_pressure_{variable}",
                variable=variable,
                component=component,
                pressure=pressure,
                **restrict,
            )
            if streamline_stabilization:
                problem.add_kernel(
                    "momentum_stabilization",
                    f"{name}_supg_{variable}",
                    variable=variable,
                    component=component,
                    pressure=pressure,
                    **residual_parameters,
                    **restrict,
                )
        else:
            problem.add_kernel(
                "penalty_incompressibility",
                f"{name}_penalty_{variable}",
                variable=variable,
                component=component,
                velocities=velocities,
                penalty_parameter=penalty_parameter,
                **restrict,
            )
        if density:
            problem.add_kernel(
                "convective_inertia",
                f"{name}_inertia_{variable}",
                variable=variable,
                component=component,
                velocities=velocities,
                density=density,
                **restrict,
            )
        if force[component] != 0.0:
            problem.add_kernel(
                "body_force",
                f"{name}_body_force_{variable}",
                variable=variable,
                value=force[component],
                **restrict,
            )
    if buoyancy is not None:
        add_boussinesq_buoyancy(
            problem,
            buoyancy["temperature"],
            velocities,
            gravity=buoyancy["gravity"],
            density=density if density else 1.0,
            thermal_expansion_coefficient=buoyancy["thermal_expansion_coefficient"],
            reference_temperature=buoyancy["reference_temperature"],
            scale_with_load=buoyancy.get("scale_with_load", False),
            name=f"{name}_buoyancy",
        )
    if pressure_formulation:
        problem.add_kernel(
            "mass_conservation",
            f"{name}_mass",
            variable=pressure,
            stabilization=stabilization,
            **residual_parameters,
            **restrict,
        )
        boundaries = (
            list(mass_flux_boundaries)
            if mass_flux_boundaries is not None
            else list(problem._mesh.sideset_names())
        )
        if boundaries:
            problem.add_boundary_condition(
                "mass_flux_boundary_condition",
                f"{name}_mass_flux",
                variable=pressure,
                boundary=boundaries,
                velocities=velocities,
            )
        if pressure_pin_point is not None:
            problem.add_boundary_condition(
                "point_Dirichlet_boundary_condition",
                f"{name}_pressure_level",
                variable=pressure,
                point=[float(c) for c in pressure_pin_point],
                value=0.0,
            )
    else:
        problem.add_material(
            "penalty_pressure",
            f"{name}_pressure",
            velocities=velocities,
            penalty_parameter=penalty_parameter,
            **restrict,
        )
    return velocities


def add_boussinesq_buoyancy(
    problem,
    temperature: str,
    velocities: Sequence[str],
    gravity: Sequence[float],
    thermal_expansion_coefficient: float,
    reference_temperature: float,
    density: float = 1.0,
    scale_with_load: bool = False,
    name: str = "buoyancy",
):
    r"""Add the Boussinesq buoyancy force to the momentum equations.

    One ``Boussinesq_buoyancy`` kernel is added for every velocity component
    along which gravity acts, giving the body force
    :math:`\mathbf{f} = -\rho_0 \beta (T - T_0) \mathbf{g}`.  With
    ``scale_with_load=True`` load stepping ramps the buoyancy, which is how a
    high Rayleigh number is reached from rest.  Together with a
    ``heat_convection`` kernel on the temperature this makes a natural
    convection problem; see ``examples/natural_convection.py``.
    """
    velocities = list(velocities)
    added = []
    for component, variable in enumerate(velocities):
        if component < len(gravity) and gravity[component] != 0.0:
            problem.add_kernel(
                "Boussinesq_buoyancy",
                f"{name}_{variable}",
                variable=variable,
                component=component,
                temperature=temperature,
                gravity=list(gravity),
                density=density,
                thermal_expansion_coefficient=thermal_expansion_coefficient,
                reference_temperature=reference_temperature,
                scale_with_load=scale_with_load,
            )
            added.append(variable)
    return added


def add_beam(
    problem,
    model: str = "beam_Euler_Bernoulli_mixed",
    axial_displacement: str = "axial_displacement",
    transverse_displacement: str = "deflection",
    rotation: str = "rotation",
    bending_moment: str = "bending_moment",
    name: str = "beam",
    **parameters,
):
    """Add a beam model (three kernels, one per variable).

    ``model`` is ``"beam_Euler_Bernoulli_mixed"``, ``"beam_Timoshenko_mixed"``, or
    ``"beam_Timoshenko_displacement"``.  The mixed models solve for the
    ``bending_moment`` as their third variable, the displacement model for the
    ``rotation``; the name of the other one is not used.  For the displacement
    Timoshenko model the shear term of the rotation equation is integrated with
    a reduced rule, which is what prevents shear locking.  Returns the names of
    the three variables.
    """
    models = ("beam_Euler_Bernoulli_mixed", "beam_Timoshenko_mixed", "beam_Timoshenko_displacement")
    if model not in models:
        raise ValueError(f"Unknown beam model '{model}'. Use one of: {', '.join(models)}.")
    displacement_model = model == "beam_Timoshenko_displacement"
    third_variable = rotation if displacement_model else bending_moment
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
    bending_moment: str = "bending_moment",
    theory: str = "first_order",
    name: str = "plate",
    **parameters,
):
    """Add an axisymmetric circular plate model on a radial mesh.

    The problem must use ``coordinates="axisymmetric"``, so that every integral
    carries the factor :math:`2 \\pi r`.

    With ``theory="first_order"`` (the default) this adds the first-order shear
    deformation, or Mindlin, model in terms of the radial displacement
    :math:`u`, the deflection :math:`w`, and the rotation :math:`\\phi_r`.  Two
    kernels are added per variable: one for the bending and membrane terms, and
    one for the transverse shear force, which is integrated at the centre of
    the element so that thin plates do not lock.

    With ``theory="classical"`` this adds the mixed classical, or Kirchhoff,
    model in terms of :math:`u`, :math:`w`, and the radial bending moment
    :math:`M_{rr}`.  The classical theory gives a fourth-order equation in
    :math:`w`, which the dual mesh control domain method cannot discretize, so
    the bending moment is carried as a third unknown and the system becomes
    three second-order equations.  There is no shear term to under-integrate,
    so one kernel per variable is enough.  The natural boundary quantity of the
    moment equation is the slope :math:`dw/dr`, which means a clamped edge needs
    no condition at all on that equation, while a simply supported edge
    prescribes :math:`M_{rr} = 0`.
    """
    if theory in ("classical", "cpt", "kirchhoff"):
        variables = [radial_displacement, transverse_displacement, bending_moment]
        for variable in variables:
            _ensure_variable(problem, variable)
        mapping = dict(
            radial_displacement=radial_displacement,
            transverse_displacement=transverse_displacement,
            bending_moment=bending_moment,
        )
        for variable in variables:
            problem.add_kernel(
                "circular_plate_classical_mixed",
                f"{name}_{variable}",
                variable=variable,
                **dict(parameters, **mapping),
            )
        return variables
    if theory not in ("first_order", "fsdt", "mindlin"):
        raise ValueError(
            "theory must be 'first_order' (shear deformable) or 'classical' (Kirchhoff), "
            f"not {theory!r}"
        )
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
            "circular_plate_first_order",
            f"{name}_bending_{variable}",
            variable=variable,
            shear_treatment="exclude",
            **dict(parameters, **mapping),
        )
        problem.add_kernel(
            "circular_plate_first_order",
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
            "plate_first_order",
            f"{name}_bending_{variable}",
            variable=variable,
            shear_treatment="exclude",
            **dict(parameters, **mapping),
        )
        problem.add_kernel(
            "plate_first_order",
            f"{name}_shear_{variable}",
            variable=variable,
            shear_treatment="only",
            quadrature="midpoint",
            reduced_integration=True,
            **dict(parameters, **mapping),
        )
    return variables
