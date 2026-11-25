"""Builds the engine's problem from a validated input tree.

The schema has checked the keys, the units and the dimensions. This module lowers what is
left onto the objects of the engine: the mesh, one property object per property and
region, the physics with the names of the properties they read, the boundary conditions as
the text that the physics read, and the postprocessors. An input that the engine cannot run
stops here with its key, so that no part of the input is dropped without a message.
"""

import numpy

from dualmesh.input import units
from dualmesh.input.lower import engine_formula
from dualmesh.input.physics_options import physics_functions
from dualmesh.input.schema import generated_region, join
from dualmesh.input.solid_model import add_finite_strain_solid, finite_strain_solid
from dualmesh.input.units import input_error

# The engine's names of the properties of the input.
engine_property_names = {
    "thermal_conductivity": "thermal_conductivity",
    "density": "density",
    "specific_heat_capacity": "specific_heat",
    "volumetric_heat_source": "volumetric_heat_source",
    "Young_modulus": "youngs_modulus",
    "Poisson_ratio": "poissons_ratio",
    "dynamic_viscosity": "dynamic_viscosity",
    "linear_expansion_coefficient": "linear_expansion_coefficient",
    "stress_free_temperature": "stress_free_temperature",
    "volumetric_expansion_coefficient": "volumetric_expansion_coefficient",
    "reference_temperature": "reference_temperature",
}
# The properties that the kernels read by name, from property objects on their regions. The
# volumetric heat source is a source term on each region, which may read the fields, and the others are numbers of a physics or a coupling: the density of a flow or of a moving solid, the
# viscosity, and the thermal expansion.
property_objects = (
    "thermal_conductivity",
    "specific_heat_capacity",
    "Young_modulus",
    "Poisson_ratio",
)
# The fields that each physics solves for, by standard name.
physics_fields = {
    "heat_conduction": ("temperature",),
    "solid_mechanics": ("displacement",),
    "fluid_dynamics": ("velocity", "pressure"),
}
engine_element_types = {
    "quadrilateral": "Quad4",
    "triangle": "Tri3",
    "hexahedron": "Hex8",
    "tetrahedron": "Tet4",
}
Stefan_Boltzmann_constant = 5.670374419e-8


class region_view:
    """What the physics functions of `physics_options` ask of a problem: whether it is transient,
    has gravity, has a flow and is heated."""

    def __init__(self, transient: bool, gravity: bool = False, physics=()):
        self.transient = transient
        self.gravity = gravity
        self.flow = "fluid_dynamics" in physics
        self.heated = "heat_conduction" in physics


def fail(path: str, message: str):
    raise input_error(f"{path}: {message}")


# The mesh


def build_mesh(mesh: dict):
    """The engine's mesh of the mesh section."""
    import dualmesh

    if "segment" in mesh:
        segment = mesh["segment"]
        return dualmesh.generate_line_mesh(0.0, segment["length"], segment["cells"])
    if "rectangle" in mesh:
        rectangle = mesh["rectangle"]
        (width, height), (columns, rows) = rectangle["size"], rectangle["cells"]
        return dualmesh.generate_rectangle_mesh(
            0.0,
            width,
            0.0,
            height,
            columns,
            rows,
            element_type=engine_element_types[rectangle["cell_type"]],
        )
    if "box" in mesh:
        box = mesh["box"]
        (length, width, height), (columns, rows, layers) = box["size"], box["cells"]
        return dualmesh.generate_box_mesh(
            0.0,
            length,
            0.0,
            width,
            0.0,
            height,
            columns,
            rows,
            layers,
            element_type=engine_element_types[box["cell_type"]],
        )
    return dualmesh.read_mesh(mesh["file"], length_unit=mesh["length_unit"])


def region_blocks(mesh: dict, engine_mesh, region: str, path: str) -> dict:
    """The keyword that restricts an object to a region: none for the one region of a generated mesh."""
    if "file" not in mesh:
        return {}
    names = set(engine_mesh.block_names().values())
    if region not in names:
        fail(
            path,
            f"the mesh file has no region '{region}'. Its regions are {', '.join(sorted(names))}.",
        )
    return {"block": [region]}


# Values


def engine_value(value, unit: str, path: str, coordinate_system: str) -> str:
    """A constant or a formula of the validated tree as an expression of the engine."""
    if isinstance(value, dict) and "table" in value:
        fail(path, "a table is not available. Give a constant or a formula.")
    if isinstance(value, (int, float)):
        return repr(float(value))
    return engine_formula(value, unit, path, coordinate_system)


def constant_everywhere(entry: dict, name: str, path: str) -> float:
    """The value of a property that the engine takes as one number."""
    value = entry["value"]
    if entry["regions"] or not isinstance(value, (int, float)):
        fail(path, f"{name} is one constant in this version, the same in every region.")
    return float(value)


# The problem


def build_problem(validated: dict, engine_mesh=None):
    """The engine's problem of a validated tree, with its physics by name and the fields they solve for.

    It is built on ``engine_mesh`` when given (an adapted mesh), and otherwise on the mesh section.
    It returns (engine problem, physics by input name, engine variable names by field name).
    """
    import dualmesh

    mesh = validated["mesh"]
    coordinate_system = mesh["coordinate_system"]
    transient = "time" in validated
    if engine_mesh is None:
        engine_mesh = build_mesh(mesh)
    physics = validated.get("physics", {})
    if not physics:
        fail("physics", "the problem needs a physics, such as heat_conduction.")
    if "models" in validated:
        fail("models", "models of your own are not available.")
    methods = {entry["method"] for entry in physics.values() if "method" in entry}
    if "method" in validated:
        methods.add(validated["method"])
    if len(methods) > 1:
        fail(
            "method",
            "one problem takes one method in this version, and the input gives "
            + " and ".join(sorted(methods))
            + ".",
        )
    options = {"coordinates": coordinate_system} if coordinate_system != "cartesian" else {}
    if methods:
        options["method"] = methods.pop()
    problem = dualmesh.Problem(engine_mesh, **options)

    properties = validated.get("properties", {})
    gravity = _gravity(validated, physics, engine_mesh, coordinate_system)
    _check_properties(physics, properties, transient, gravity is not None)
    finite = physics.get("solid_mechanics", {}).get("kinematics") == "finite"
    for name in properties:
        if finite and name in ("Young_modulus", "Poisson_ratio"):
            continue
        if name in property_objects or (
            name == "density"
            and "heat_conduction" in physics
            and (transient or "fluid_dynamics" in physics)
        ):
            _add_property(problem, mesh, engine_mesh, name, properties[name], coordinate_system)

    engine_physics = {}
    initial = validated.get("initial_conditions", {})
    solved = {field for name in physics for field in physics_fields[name]}
    for key in initial:
        if key not in solved:
            fail(join("initial_conditions", key), f"no physics of the problem solves for {key}.")
    for name, entry in physics.items():
        engine_physics[name] = _add_physics(
            problem,
            name,
            entry,
            properties,
            initial,
            transient,
            engine_mesh,
            coordinate_system,
            gravity,
            physics,
        )
    if "volumetric_heat_source" in properties:
        _add_heat_source(
            problem, mesh, engine_mesh, properties["volumetric_heat_source"], coordinate_system
        )
    expansion = [
        key
        for key in ("linear_expansion_coefficient", "stress_free_temperature")
        if key in properties
    ]
    if len(expansion) == 1:
        fail(
            join("properties", expansion[0]),
            "a thermal expansion takes linear_expansion_coefficient and stress_free_temperature "
            f"together, and the input gives only {expansion[0]}.",
        )
    if expansion and "heat_conduction" not in engine_physics:
        fail(
            "properties.linear_expansion_coefficient",
            "a thermal expansion follows the temperature of heat_conduction, which the problem "
            "does not have.",
        )
    if "heat_conduction" in engine_physics and "linear_expansion_coefficient" in properties:
        if isinstance(engine_physics.get("solid_mechanics"), finite_strain_solid):
            fail(
                "properties.linear_expansion_coefficient",
                "the thermal expansion at finite strain is not available.",
            )
        if "solid_mechanics" not in engine_physics:
            fail(
                "properties.linear_expansion_coefficient",
                "the thermal expansion needs solid_mechanics.",
            )
        problem.add_coupling(
            "thermal_expansion",
            heat_transfer=engine_physics["heat_conduction"],
            solid_mechanics=engine_physics["solid_mechanics"],
            linear_expansion_coefficient=constant_everywhere(
                properties["linear_expansion_coefficient"],
                "linear_expansion_coefficient",
                "properties.linear_expansion_coefficient",
            ),
            stress_free_temperature=constant_everywhere(
                properties["stress_free_temperature"],
                "stress_free_temperature",
                "properties.stress_free_temperature",
            ),
        )
    if "heat_conduction" in engine_physics and "fluid_dynamics" in engine_physics:
        # Under gravity the fluid takes the Boussinesq force -rho beta (T - T_0) g, on top of
        # its weight rho g, which the flow took with its density at T_0.
        buoyancy = {}
        if gravity is not None:
            buoyancy = {
                "gravity": gravity,
                **{
                    name: constant_everywhere(properties[name], name, join("properties", name))
                    for name in ("volumetric_expansion_coefficient", "reference_temperature")
                },
            }
        problem.add_coupling(
            "nonisothermal_flow",
            heat_transfer=engine_physics["heat_conduction"],
            fluid_dynamics=engine_physics["fluid_dynamics"],
            **buoyancy,
        )
    variables = _variables(engine_physics)
    _add_boundary_conditions(
        validated.get("boundary_conditions", {}), engine_physics, variables, coordinate_system
    )
    return problem, engine_physics, variables


def _gravity(validated: dict, physics: dict, engine_mesh, coordinate_system: str):
    """The gravitational acceleration of the input in m/s^2, one value per direction of space, or
    None.  The solid and the flow take its force rho g.  A body force along the radius would
    break the symmetry of an axisymmetric problem, so there it acts along the axis alone."""
    if "gravity" not in validated:
        return None
    g = list(validated["gravity"])
    if len(g) != engine_mesh.dimension:
        fail(
            "gravity",
            f"gravity takes one acceleration per direction of the mesh, which has "
            f"{engine_mesh.dimension}, and the input gives {len(g)}.",
        )
    if not {"solid_mechanics", "fluid_dynamics"} & set(physics):
        fail("gravity", "no physics of the problem reads gravity: it acts on solids and flows.")
    if coordinate_system == "axisymmetric" and g[0] != 0.0:
        fail(
            "gravity",
            "in axisymmetric coordinates gravity acts along the axis z, so its radial component "
            "is 0.",
        )
    return g


def _check_properties(physics: dict, properties: dict, transient: bool, gravity: bool) -> None:
    """A property that no physics reads, or one that a physics needs and the input leaves out,
    stops the run."""
    needed, readable = {}, set()
    for name in physics:
        reads = physics_functions[name](region_view(transient, gravity, physics))
        for property_name in reads["needed"]:
            needed.setdefault(property_name, name)
        readable.update(reads["needed"], reads["optional"])
    for property_name, physics_name in needed.items():
        if property_name not in properties:
            fail("properties", f"{physics_name} needs the property {property_name}.")
    for property_name in properties:
        if property_name not in readable:
            fail(
                join("properties", property_name), "no physics of the problem reads this property."
            )
        if property_name not in engine_property_names:
            fail(
                join("properties", property_name),
                "this property is not available.",
            )


def _add_property(
    problem, mesh, engine_mesh, name: str, entry: dict, coordinate_system: str
) -> None:
    engine_name = engine_property_names[name]
    path = join("properties", name)
    values = {generated_region: entry["value"]} if not entry["regions"] else entry["regions"]
    for region, value in values.items():
        region_path = path if not entry["regions"] else join(path, region)
        restriction = (
            region_blocks(mesh, engine_mesh, region, region_path) if entry["regions"] else {}
        )
        if isinstance(value, (int, float)):
            problem.add_property("constant", properties={engine_name: float(value)}, **restriction)
        else:
            expression = engine_value(value, entry["unit"], region_path, coordinate_system)
            problem.add_property(
                "parsed", property_name=engine_name, expression=expression, **restriction
            )


def _add_heat_source(problem, mesh, engine_mesh, entry: dict, coordinate_system: str) -> None:
    """The volumetric heat source: a source term of the temperature on each region, which may read
    the fields."""
    path = "properties.volumetric_heat_source"
    values = {generated_region: entry["value"]} if not entry["regions"] else entry["regions"]
    for region, value in values.items():
        region_path = path if not entry["regions"] else join(path, region)
        restriction = (
            region_blocks(mesh, engine_mesh, region, region_path) if entry["regions"] else {}
        )
        source = engine_value(value, entry["unit"], region_path, coordinate_system)
        problem.add_kernel("parsed_kernel", variable="temperature", source=source, **restriction)


def _add_physics(
    problem,
    name,
    entry,
    properties,
    initial,
    transient,
    engine_mesh,
    coordinate_system,
    gravity,
    physics,
):
    path = join("physics", name)
    if name == "heat_conduction":
        options = {"thermal_conductivity": "thermal_conductivity"}
        # The heat capacity stores heat in time and carries it with a flow.
        if transient or "fluid_dynamics" in physics:
            options.update(density="density", specific_heat="specific_heat")
        if "temperature" in initial:
            options["initial_condition"] = float(initial["temperature"])
        elif transient:
            fail("initial_conditions", "a transient heat_conduction needs the initial temperature.")
        return problem.add_physics("heat_transfer", **options)
    # The weight rho g per volume of the solid or the fluid.
    weight = None
    if gravity is not None:
        density = constant_everywhere(properties["density"], "density", "properties.density")
        weight = [density * g for g in gravity]
    if name == "solid_mechanics":
        formulation = _solid_formulation(entry, engine_mesh, coordinate_system, path)
        if entry.get("kinematics", "small") == "finite":
            if formulation not in ("plane_strain", "three_dimensional"):
                fail(
                    join(path, "formulation"),
                    "finite strain is available in plane strain and in three "
                    "dimensions in this version.",
                )
            if transient:
                fail(path, "a moving solid at finite strain is not available.")
            if "displacement" in initial:
                fail(
                    "initial_conditions.displacement",
                    "an initial displacement is the start of a transient run, which a solid at "
                    "finite strain does not take.",
                )
            return add_finite_strain_solid(
                problem,
                entry.get("strain_energy", "neo-Hookean"),
                engine_mesh.dimension,
                constant_everywhere(
                    properties["Young_modulus"], "Young_modulus", "properties.Young_modulus"
                ),
                constant_everywhere(
                    properties["Poisson_ratio"], "Poisson_ratio", "properties.Poisson_ratio"
                ),
                join(path, "strain_energy"),
                weight,
            )
        if "strain_energy" in entry:
            fail(
                join(path, "strain_energy"),
                "at small strain every energy is linear elasticity with "
                "the Young modulus and the Poisson ratio, so a small-strain solid takes no strain_energy.",
            )
        options = {"formulation": formulation}
        if transient:
            options["density"] = constant_everywhere(
                properties["density"], "density", "properties.density"
            )
        if weight is not None:
            options["body_force"] = weight
        if "displacement" in initial:
            if not transient:
                fail(
                    "initial_conditions.displacement",
                    "an initial displacement is the start of a transient run with inertia, so it "
                    "needs the time section.",
                )
            options["initial_condition"] = {"displacement": list(initial["displacement"])}
        return problem.add_physics("solid_mechanics", **options)
    options = {
        "density": constant_everywhere(properties["density"], "density", "properties.density"),
        "dynamic_viscosity": constant_everywhere(
            properties["dynamic_viscosity"], "dynamic_viscosity", "properties.dynamic_viscosity"
        ),
        "formulation": entry.get("formulation", "equal_order"),
    }
    if weight is not None:
        options["body_force"] = weight
    flow_initial = {key: initial[key] for key in ("velocity", "pressure") if key in initial}
    if flow_initial:
        options["initial_condition"] = flow_initial
    return problem.add_physics("fluid_dynamics", **options)


def _solid_formulation(entry: dict, engine_mesh, coordinate_system: str, path: str) -> str:
    """The formulation of the solid: it follows from the mesh, except plane strain or plane stress."""
    given = entry.get("formulation")
    implied = None
    if coordinate_system == "axisymmetric":
        implied = "axisymmetric"
    elif engine_mesh.dimension == 3:
        implied = "three_dimensional"
    if implied is not None:
        if given not in (None, implied):
            fail(
                join(path, "formulation"),
                f"the mesh makes the solid {implied}, so it takes no formulation {given}.",
            )
        return implied
    if engine_mesh.dimension != 2:
        fail(path, "solid_mechanics needs a plane, axisymmetric or three-dimensional mesh.")
    if given not in ("plane_strain", "plane_stress"):
        fail(path, "a plane solid needs formulation: plane_strain or plane_stress.")
    return given


def _variables(engine_physics: dict) -> dict:
    """The engine's variable names of each field of the input."""
    variables = {}
    for name, physics in engine_physics.items():
        names = list(physics.variable_names())
        if name == "heat_conduction":
            variables["temperature"] = names[:1]
        elif name == "solid_mechanics":
            variables["displacement"] = names
        else:
            variables["velocity"] = list(physics.velocities)
            variables["pressure"] = [physics.pressure]
    return variables


# Boundary conditions


def _add_boundary_conditions(
    conditions: dict, engine_physics: dict, variables: dict, coordinate_system: str
):
    owner = {
        field: name
        for name, fields in physics_fields.items()
        for field in fields
        if name in engine_physics
    }
    solid = engine_physics.get("solid_mechanics")
    for boundary, entries in conditions.items():
        for key, value in entries.items():
            path = join(join("boundary_conditions", boundary), key)
            if (
                key == "pressure"
                and isinstance(solid, finite_strain_solid)
                and "pressure" not in owner
            ):
                fail(
                    path,
                    "a pressure on a solid at finite strain follows the deformed boundary, "
                    "which is not available. Give a traction.",
                )
            if key == "traction" and isinstance(solid, finite_strain_solid):
                components = [
                    engine_value(component, "Pa", path, coordinate_system)
                    for component in _listed(value, path)
                ]
                solid.set_traction(components, boundary)
                continue
            texts, physics_name = _condition_texts(
                key, value, variables, owner, coordinate_system, path
            )
            for text in texts:
                if isinstance(engine_physics[physics_name], finite_strain_solid):
                    variable, expression = text.split(" = ", 1)
                    engine_physics[physics_name].set_value(variable, expression, boundary)
                else:
                    engine_physics[physics_name].add_boundary_condition(text, boundary=boundary)


def _condition_texts(key, value, variables, owner, coordinate_system, path):
    """The texts of the engine's boundary conditions of one key of the input, and the physics they belong to."""
    if key in ("heat_flux", "convection", "radiation"):
        if "heat_conduction" not in owner.values():
            fail(path, "a heat flux needs heat_conduction.")
        return [f"heat_flux = {_heat_flux(key, value, coordinate_system, path)}"], "heat_conduction"
    if key == "pressure" and "pressure" not in owner and "solid_mechanics" in owner.values():
        # Without a flow, the pressure on a boundary is the load -p n on the solid.
        return [
            f"pressure = {engine_value(value, 'Pa', path, coordinate_system)}"
        ], "solid_mechanics"
    if key == "traction":
        if "solid_mechanics" not in owner.values():
            fail(path, "a traction needs solid_mechanics.")
        components = [
            engine_value(component, "Pa", path, coordinate_system)
            for component in _listed(value, path)
        ]
        return [f"traction = [{', '.join(components)}]"], "solid_mechanics"
    base_name, component = units.split_component(key, coordinate_system) or (key, ())
    if base_name not in owner:
        fail(path, f"no physics of the problem solves for {base_name}.")
    names = variables[base_name]
    unit = units.standard_names()[base_name].unit
    if component:
        return [
            f"{names[component[0]]} = {engine_value(value, unit, path, coordinate_system)}"
        ], owner[base_name]
    if len(names) == 1:
        return [f"{names[0]} = {engine_value(value, unit, path, coordinate_system)}"], owner[
            base_name
        ]
    listed = _listed(value, path)
    if len(listed) != len(names):
        fail(path, f"{base_name} takes {len(names)} components, got {len(listed)}.")
    return [
        f"{name} = {engine_value(entry, unit, path, coordinate_system)}"
        for name, entry in zip(names, listed)
    ], owner[base_name]


def _listed(value, path):
    if not isinstance(value, (list, tuple)):
        fail(path, "expects one value per component, as a list.")
    return list(value)


def _heat_flux(key, value, coordinate_system, path) -> str:
    """The outward heat flux of a heat_flux, convection or radiation condition, in W/m^2."""
    if key == "heat_flux":
        return engine_value(value, "W/m^2", path, coordinate_system)
    if not isinstance(value, dict):
        fail(path, f"{key} takes a mapping of its constants.")
    expected = {
        "convection": ("heat_transfer_coefficient", "ambient_temperature"),
        "radiation": ("emissivity", "ambient_temperature"),
    }[key]
    for constant in value:
        if constant not in expected:
            fail(join(path, constant), f"{key} takes {' and '.join(expected)}.")
    for constant in expected:
        if constant not in value:
            fail(path, f"{key} needs {constant}.")
    ambient = engine_value(
        value["ambient_temperature"], "K", join(path, "ambient_temperature"), coordinate_system
    )
    if key == "convection":
        coefficient = engine_value(
            value["heat_transfer_coefficient"],
            "W/(m^2*K)",
            join(path, "heat_transfer_coefficient"),
            coordinate_system,
        )
        return f"{coefficient}*(temperature - {ambient})"
    emissivity = engine_value(value["emissivity"], "1", join(path, "emissivity"), coordinate_system)
    return f"{Stefan_Boltzmann_constant!r}*{emissivity}*(temperature^4 - ({ambient})^4)"


# Time and postprocessors


def transient_options(time: dict) -> dict:
    """The keywords of the engine's transient solve of the time section."""
    control = time["step_control"]
    options = {"end_time": time["end_time"], "time_step": time["initial_time_step"]}
    by = control["by"]
    if by == "error":
        options.update(time_stepper="error", error_tolerance=control["relative_tolerance"])
    elif by == "iterations":
        options.update(time_stepper="iteration", optimal_iterations=control["target_iterations"])
    elif by == "Courant":
        options.update(time_stepper="cfl", courant_number=control["Courant_number"])
    else:
        options.update(time_stepper="fixed")
    return options


def add_postprocessors(
    problem, postprocessors: dict, variables: dict, dimension: int, coordinate_system: str
) -> tuple[dict, dict]:
    """Adds the postprocessors that the engine computes, and returns the SI unit of each, and the
    stress components at points, which `stresses_at_points` computes after the solve."""
    unit_of, stress_requests = {}, {}
    # An axisymmetric integral spans the revolved body (the factor 2 pi r), so it has the
    # dimension of a three-dimensional one.
    measured = 3 if coordinate_system == "axisymmetric" else dimension
    for name, definition in postprocessors.items():
        path = join("postprocessors", name)
        if "region" in definition:
            fail(
                join(path, "region"),
                "a postprocessor over one region is not available.",
            )
        if "boundary_flux" in definition:
            flux = definition["boundary_flux"]
            if flux != "heat_flux" or "temperature" not in variables:
                fail(path, "the boundary flux of heat_flux is the one available in this version.")
            problem.add_postprocessor(
                "boundary_flux",
                name,
                variable=variables["temperature"][0],
                boundary=definition["boundary"],
            )
            unit_of[name] = "W" if measured == 3 else ("W/m" if measured == 2 else "W/m^2")
            continue
        reduction = next(
            key
            for key in ("integral", "average", "maximum", "minimum", "point_value")
            if key in definition
        )
        field = definition[reduction]
        base_name, component = units.split_component(field, coordinate_system) or (field, ())
        if base_name == "stress":
            if reduction != "point_value" or len(component) != 2:
                fail(
                    path,
                    "a stress is available as the point_value of one component, such as stress_yy.",
                )
            stress_requests[name] = (component, list(definition["point"]))
            continue
        if base_name not in variables or (len(variables[base_name]) > 1 and not component):
            fail(
                path,
                f"a postprocessor reads one scalar field or component, and {field} is not one of the problem.",
            )
        variable = variables[base_name][component[0] if component else 0]
        parameters = {"variable": variable}
        if reduction == "point_value":
            parameters["point"] = list(definition["point"]) + [0.0] * (3 - len(definition["point"]))
        problem.add_postprocessor(reduction, name, **parameters)
        field_unit = units.standard_names()[base_name].unit
        unit_of[name] = f"{field_unit}*m^{measured}" if reduction == "integral" else field_unit
    return unit_of, stress_requests


def stresses_at_points(engine, requests: dict, variables: dict, validated: dict) -> dict:
    """The Cauchy stress components of the requests, in Pa, from the displacement gradient at each
    point: sigma = lambda tr(e) I + 2 mu e, with e = epsilon - alpha (T - T_0) I the mechanical
    strain, epsilon the symmetric part of the gradient and alpha (T - T_0) I the thermal strain of
    a thermal expansion, in plane strain, plane stress or three dimensions."""
    if not requests:
        return {}
    path = "postprocessors"
    solid = validated.get("physics", {}).get("solid_mechanics")
    if solid is None:
        fail(path, "a stress needs solid_mechanics.")
    if "time" in validated or solid.get("kinematics", "small") != "small":
        fail(
            path,
            "a stress at a point is available for a steady solid at small strain in this version.",
        )
    if validated["mesh"]["coordinate_system"] != "cartesian":
        fail(path, "a stress at a point is available in Cartesian coordinates in this version.")
    properties = validated["properties"]
    Young_modulus = constant_everywhere(
        properties["Young_modulus"], "Young_modulus", "properties.Young_modulus"
    )
    Poisson_ratio = constant_everywhere(
        properties["Poisson_ratio"], "Poisson_ratio", "properties.Poisson_ratio"
    )
    shear_modulus = Young_modulus / (2 * (1 + Poisson_ratio))
    first_Lame_parameter = (
        Young_modulus * Poisson_ratio / ((1 + Poisson_ratio) * (1 - 2 * Poisson_ratio))
    )
    names = variables["displacement"]
    dimension = len(names)
    plane_stress = dimension == 2 and solid.get("formulation") == "plane_stress"
    if plane_stress:
        # The out-of-plane stress is zero, which sets the out-of-plane strain.
        first_Lame_parameter = (
            2 * first_Lame_parameter * shear_modulus / (first_Lame_parameter + 2 * shear_modulus)
        )
    points = [request[1] for request in requests.values()]
    gradients = [engine.sample_gradient(name, points) for name in names]
    # The thermal strain of a thermal expansion, which the solve applied (build_problem).
    thermal = numpy.zeros(len(points))
    if "linear_expansion_coefficient" in properties and "temperature" in variables:
        expansion = constant_everywhere(
            properties["linear_expansion_coefficient"],
            "linear_expansion_coefficient",
            "properties.linear_expansion_coefficient",
        )
        stress_free = constant_everywhere(
            properties["stress_free_temperature"],
            "stress_free_temperature",
            "properties.stress_free_temperature",
        )
        temperatures = numpy.asarray(engine.sample(variables["temperature"][0], points))
        thermal = expansion * (temperatures - stress_free)
    # In plane stress the thermal strain enters the in-plane strains, and the condensed first Lame
    # parameter takes the out-of-plane one into account.
    thermal_axes = numpy.diag([1.0, 1.0, 0.0]) if plane_stress else numpy.eye(3)
    result = {}
    for index, (name, (component, _point)) in enumerate(requests.items()):
        if any(numpy.isnan(gradient[index]).any() for gradient in gradients):
            fail(join(path, name), "the point lies outside the mesh.")
        strain = numpy.zeros((3, 3))
        for row in range(dimension):
            for column in range(dimension):
                strain[row, column] = 0.5 * (
                    gradients[row][index][column] + gradients[column][index][row]
                )
        strain = strain - thermal[index] * thermal_axes
        trace = numpy.trace(strain)
        stress = first_Lame_parameter * trace * numpy.eye(3) + 2 * shear_modulus * strain
        if plane_stress:
            stress[2, 2] = 0.0
        result[name] = float(stress[component])
    return result
