"""Builds the engine's problem from a validated input tree.

The schema has checked the keys, the units and the dimensions. This module lowers what is
left onto the objects of the engine: the mesh, one property object per property and
region, the physics with the names of the properties they read, the boundary conditions as
the text that the physics read, and the postprocessors. The schema refuses what the engine
cannot run, and the checks that need the mesh stop here with their key, so that no part of
the input is dropped without a message.
"""

from dataclasses import dataclass

import numpy

from dualmesh import units
from dualmesh._core import InputError
from dualmesh.engine import Problem
from dualmesh.input import user_model
from dualmesh.input.physics_options import physics_functions
from dualmesh.input.schema import context, generated_region, join, symmetry
from dualmesh.meshing import (
    generate_box_mesh,
    generate_line_mesh,
    generate_rectangle_mesh,
    read_mesh,
)
from dualmesh.model import graph, parse
from dualmesh.model.engine_form import add_boundary_condition, engine_formula, write_expression
from dualmesh.model.graph import field_component
from dualmesh.physics import HYPERELASTIC
from dualmesh.postprocessors import von_Mises_stress
from dualmesh.tables import Table
from dualmesh.units import fail

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
    "yield_stress": "yield_stress",
}
# The properties that the kernels read by name, from property objects on their regions. The
# density and the heat source of heat_conduction are property objects too. The others are
# numbers of a physics or a coupling: the density of a flow or of a moving solid, the viscosity
# and the thermal expansion.
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
# The postprocessors that read one field: a reduction over the domain or the value at a point.
field_reductions = ("integral", "average", "maximum", "minimum", "point_value")
# The unit of the heat through a boundary, by the dimension of the integral: per unit area at
# the end of a line, per unit thickness of a plane body, and the whole heat of a solid body.
heat_flow_units = {1: "W/m^2", 2: "W/m", 3: "W"}


class region_view:
    """What the physics functions of `physics_options` ask of a problem: whether it is transient,
    has gravity, has a flow and is heated."""

    def __init__(self, transient: bool, gravity: bool = False, physics=()):
        self.transient = transient
        self.gravity = gravity
        self.flow = "fluid_dynamics" in physics
        self.heated = "heat_conduction" in physics


# The mesh


def build_mesh(mesh: dict):
    """The engine's mesh of the mesh section. The grading of a generated mesh is the bias of the
    engine's generators: the ratio of the length of each cell to the one before it."""
    if "segment" in mesh:
        segment = mesh["segment"]
        start = segment["origin"]
        return generate_line_mesh(
            start, start + segment["length"], segment["cells"], bias=segment["grading"]
        )
    for form, generate in (
        ("rectangle", generate_rectangle_mesh),
        ("box", generate_box_mesh),
    ):
        if form not in mesh:
            continue
        generated = mesh[form]
        axes = "xyz"[: len(generated["size"])]
        options = {}
        for axis, start, size, cells, grading in zip(
            axes, generated["origin"], generated["size"], generated["cells"], generated["grading"]
        ):
            options |= {axis: [start, start + size], f"num_{axis}_elements": cells}
            options[f"{axis}_bias"] = grading
        return generate(element_type=engine_element_types[generated["cell_type"]], **options)
    return read_mesh(mesh["file"], length_unit=mesh["length_unit"])


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


def engine_value(
    value, unit: str, path: str, coordinate_system: str, variables: dict | None = None, symbols=None
) -> str:
    """A constant or a formula of the validated tree as an expression of the engine.

    `variables` maps each field to the engine's names of its variables, and `symbols` the names
    outside the vocabulary that a formula reads, such as the unknown of a model."""
    if isinstance(value, (int, float)):
        return repr(float(value))
    return engine_formula(value, unit, path, coordinate_system, variables, symbols)


def constant_everywhere(properties: dict, name: str) -> float:
    """The value of a property that the engine takes as one number of its physics."""
    return float(_value_everywhere(properties, name, formula=False))


def _value_everywhere(properties: dict, name: str, formula: bool):
    """The value of a property that has one value in every region: one constant, or with
    `formula` one constant or one formula of the solid. Any other value stops the run."""
    entry = properties[name]
    value = entry["value"]
    allowed = not isinstance(value, dict) if formula else isinstance(value, (int, float))
    if entry["regions"] or not allowed:
        if formula:
            message = (
                f"the {name.replace('_', ' ')} is one constant or one formula of the solid, the "
                "same in every region."
            )
        else:
            message = (
                f"the engine takes {name} as one number of its physics, so it is one constant, "
                "the same in every region."
            )
        fail(join("properties", name), message)
    return value


# The problem


def build_problem(validated: dict, engine_mesh=None):
    """The engine's problem of a validated tree, with its physics by name and the fields they solve for.

    It is built on ``engine_mesh`` when given (an adapted mesh), and otherwise on the mesh section.
    It returns (engine problem, physics by input name, engine variable names by field name).
    """
    mesh = validated["mesh"]
    coordinate_system = mesh["coordinate_system"]
    transient = "time" in validated
    # A transient run and an eigenvalue study both take the time terms: the storage, the inertia
    # and the properties that they read.
    dynamic = transient or "eigenvalues" in validated
    if engine_mesh is None:
        engine_mesh = build_mesh(mesh)
    physics = validated.get("physics", {})
    models = user_model.prepare(validated, engine_mesh)
    if not physics and not models:
        fail("physics", "the problem needs a physics, such as heat_conduction.")
    problem = Problem(engine_mesh, **_problem_options(validated, physics, coordinate_system))

    properties = validated.get("properties", {})
    gravity = _gravity(validated, physics, engine_mesh, coordinate_system)
    model_reads = set().union(*(model.reads for model in models))
    view = region_view(dynamic, gravity is not None, physics)
    reads = {name: physics_functions[name](view) for name in physics}
    physics_reads = _check_properties(reads, properties, model_reads)

    initial = _initial_values(validated)
    _check_fields(validated, physics, models, transient, initial)
    setting = _physics_setting(
        properties,
        initial,
        transient,
        dynamic,
        engine_mesh,
        coordinate_system,
        gravity,
        "heat_conduction" in physics,
    )
    engine_physics = {
        name: _add_physics(problem, setting, name, entry, reads[name]["needed"])
        for name, entry in physics.items()
    }
    # The engine's names of the variables of each field, which the formulas take.
    variables = _variables(engine_physics)
    user_model.add_variables(problem, models, initial, variables)
    heat_needed = reads.get("heat_conduction", {}).get("needed", ())
    _add_properties(problem, setting, mesh, physics_reads, heat_needed, variables)
    _add_thermal_expansion(problem, engine_physics, properties, coordinate_system, variables)
    _add_nonisothermal_flow(problem, engine_physics, properties, gravity)
    user_model.add(
        problem, models, variables, dynamic, engine_mesh.dimension, _postprocessed_fields(validated)
    )
    _add_boundary_conditions(
        validated.get("boundary_conditions", {}),
        engine_physics,
        variables,
        coordinate_system,
        problem,
        models,
        validated,
    )
    return problem, engine_physics, variables


@dataclass
class _physics_setting:
    """What the physics of the input read when they are added: the properties, the initial
    values, whether the run is transient, whether it takes the time terms (dynamic, which an
    eigenvalue study takes too), the engine's mesh, the coordinate system, the gravity or None,
    and whether the problem has heat_conduction (heated)."""

    properties: dict
    initial: dict
    transient: bool
    dynamic: bool
    engine_mesh: object
    coordinate_system: str
    gravity: list | None
    heated: bool


def _problem_options(validated: dict, physics: dict, coordinate_system: str) -> dict:
    """The keywords of the engine's Problem: the coordinate system, and the method, which the
    schema keeps one for the whole problem, given at the top or in its entries."""
    methods = {
        entry["method"]
        for entry in [validated, *physics.values(), *validated.get("models", {}).values()]
        if "method" in entry
    }
    options = {"coordinates": coordinate_system} if coordinate_system != "cartesian" else {}
    if methods:
        options["method"] = methods.pop()
    return options


def _check_fields(validated: dict, physics: dict, models: list, transient: bool, initial: dict):
    """Stops the run at a model that does not fit the fields of the physics, at the states of a
    model in a transient run with mesh_adaptivity, and at an initial condition of a field that no
    physics or model solves for."""
    solved = {field for name in physics for field in physics_fields[name]}
    user_model.check(models, solved, transient, initial)
    if transient and "mesh_adaptivity" in validated and any(model.law.states() for model in models):
        fail(
            "mesh_adaptivity",
            "the states of a model live at the integration points, which a new mesh does not "
            "keep, so a transient problem with states takes no mesh_adaptivity.",
        )
    solved |= {model.unknown for model in models}
    for key in initial:
        if key not in solved:
            fail(join("initial_conditions", key), f"no physics of the problem solves for {key}.")


def _add_properties(
    problem, setting: _physics_setting, mesh: dict, physics_reads: set, heat_needed, variables
) -> None:
    """Adds the property objects that the kernels read: the properties of property_objects, and
    those that the kernels of heat_conduction read (heat_needed) with the source of heat, a
    property that the source term reads."""
    heat_objects = [*heat_needed, "volumetric_heat_source"]
    for name, entry in setting.properties.items():
        if name in physics_reads and (name in property_objects or name in heat_objects):
            _add_property(
                problem,
                mesh,
                setting.engine_mesh,
                name,
                entry,
                setting.coordinate_system,
                variables,
            )


def _add_thermal_expansion(
    problem, engine_physics: dict, properties: dict, coordinate_system: str, variables: dict
) -> None:
    """Adds the thermal expansion of the solid by the temperature of heat_conduction, when the
    input gives the linear expansion coefficient and the stress-free temperature."""
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
    if "heat_conduction" not in engine_physics or "linear_expansion_coefficient" not in properties:
        return
    if "solid_mechanics" not in engine_physics:
        fail(
            "properties.linear_expansion_coefficient",
            "the thermal expansion needs solid_mechanics.",
        )
    problem.add_coupling(
        "thermal_expansion",
        heat_conduction=engine_physics["heat_conduction"],
        solid_mechanics=engine_physics["solid_mechanics"],
        linear_expansion_coefficient=_expansion_coefficient(
            properties, coordinate_system, variables
        ),
        stress_free_temperature=constant_everywhere(properties, "stress_free_temperature"),
    )


def _add_nonisothermal_flow(problem, engine_physics: dict, properties: dict, gravity) -> None:
    """Couples heat_conduction and fluid_dynamics, when the problem has both. Under gravity the
    fluid takes the Boussinesq force -rho beta (T - T_0) g, on top of its weight rho g, which
    the flow took with its density at T_0."""
    if "heat_conduction" not in engine_physics or "fluid_dynamics" not in engine_physics:
        return
    buoyancy = {}
    if gravity is not None:
        buoyancy = {
            "gravity": gravity,
            **{
                name: constant_everywhere(properties, name)
                for name in ("volumetric_expansion_coefficient", "reference_temperature")
            },
        }
    problem.add_coupling(
        "nonisothermal_flow",
        heat_conduction=engine_physics["heat_conduction"],
        fluid_dynamics=engine_physics["fluid_dynamics"],
        **buoyancy,
    )


def _postprocessed_fields(validated: dict) -> set:
    """The fields and the quantities of the models that the postprocessors read."""
    return {
        definition[reduction]
        for definition in validated.get("postprocessors", {}).values()
        for reduction in field_reductions
        if reduction in definition
    }


def _initial_values(validated: dict) -> dict:
    """The initial conditions as the engine takes them: a number, or the text of a formula of the
    coordinates, for each field or component."""
    coordinate_system = validated["mesh"]["coordinate_system"]
    tree_context = context(validated, complete=True)
    result = {}
    for name, value in validated.get("initial_conditions", {}).items():
        path = join("initial_conditions", name)
        unit = tree_context.quantity_of(name, path)[1]
        if isinstance(value, list):
            result[name] = [
                _initial_value(entry, unit, f"{path}[{index}]", coordinate_system)
                for index, entry in enumerate(value)
            ]
        else:
            result[name] = _initial_value(value, unit, path, coordinate_system)
    return result


def _initial_value(value, unit: str, path: str, coordinate_system: str):
    if isinstance(value, (int, float)):
        return float(value)
    return engine_formula(value, unit, path, coordinate_system)


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


def _check_properties(reads: dict, properties: dict, model_reads: set) -> set:
    """A property that no physics or model reads, or one that a physics needs and the input leaves
    out, stops the run. `reads` holds what each physics function returns. It returns the
    properties that the physics read. A model takes the properties it reads into its own terms."""
    needed, readable = {}, set()
    for name, read in reads.items():
        for property_name in read["needed"]:
            needed.setdefault(property_name, name)
        readable.update(read["needed"], read["optional"])
    for property_name, physics_name in needed.items():
        if property_name not in properties:
            fail("properties", f"{physics_name} needs the property {property_name}.")
    for property_name in properties:
        if property_name not in readable and property_name not in model_reads:
            reader = "physics or model" if model_reads else "physics"
            fail(
                join("properties", property_name),
                f"no {reader} of the problem reads this property.",
            )
        if property_name in readable and property_name not in engine_property_names:
            fail(
                join("properties", property_name),
                "this property is not available.",
            )
    return readable


def _add_property(
    problem, mesh, engine_mesh, name: str, entry: dict, coordinate_system: str, variables: dict
) -> None:
    """One property object per region: a constant, a formula or a table of one input."""
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
        elif isinstance(value, dict):
            table = value["table"]
            argument = table["inputs"][0]
            node = parse(argument["name"], None, coordinate_system)
            problem.add_property(
                "table",
                property_name=engine_name,
                argument=write_expression(node, region_path, variables),
                points=argument["values"],
                values=table["values"],
                **restriction,
            )
        else:
            expression = engine_value(
                value, entry["unit"], region_path, coordinate_system, variables
            )
            problem.add_property(
                "parsed", property_name=engine_name, expression=expression, **restriction
            )
        if name == "volumetric_heat_source":
            problem.add_kernel(
                "body_force", variable="temperature", value=engine_name, **restriction
            )


def _add_physics(problem, setting: _physics_setting, name: str, entry: dict, needed):
    """Adds one physics of the input, with the names of the properties it reads from property
    objects and the numbers of the others. `needed` lists the properties that its physics
    function needs."""
    if name == "heat_conduction":
        return _add_heat_conduction(problem, setting, needed)
    # The density of the solid or the fluid, and its weight rho g per volume.
    density = constant_everywhere(setting.properties, "density") if "density" in needed else None
    weight = None if setting.gravity is None else [density * g for g in setting.gravity]
    if name == "solid_mechanics":
        return _add_solid_mechanics(problem, setting, entry, density, weight)
    return _add_fluid_dynamics(problem, setting, entry, density, weight)


def _add_heat_conduction(problem, setting: _physics_setting, needed):
    """Adds heat_conduction, which reads its properties from property objects."""
    options = {"thermal_conductivity": "thermal_conductivity"}
    # The heat capacity stores heat in time and carries it with a flow.
    if "density" in needed:
        options.update(density="density", specific_heat="specific_heat")
    if "temperature" in setting.initial:
        options["initial_condition"] = setting.initial["temperature"]
    elif setting.transient:
        fail("initial_conditions", "a transient heat_conduction needs the initial temperature.")
    return problem.add_physics("heat_conduction", **options)


def _add_solid_mechanics(problem, setting: _physics_setting, entry: dict, density, weight):
    """Adds solid_mechanics, with its density in a run with the time terms and its weight under
    gravity."""
    path = join("physics", "solid_mechanics")
    initial = setting.initial
    if "displacement" in initial and not setting.transient:
        fail(
            "initial_conditions.displacement",
            "an initial displacement is the start of a transient run with inertia, so it "
            "needs the time section.",
        )
    formulation = _solid_formulation(entry, setting.engine_mesh, setting.coordinate_system, path)
    options = {"formulation": formulation}
    if entry["kinematics"] == "finite":
        options["large_deformation"] = True
        options.update(_strain_energy(entry["strain_energy"], join(path, "strain_energy")))
    if "yield_stress" in setting.properties:
        options.update(
            _plasticity(setting.properties, entry, setting.coordinate_system, setting.heated)
        )
    # The density and the weight are per volume of the undeformed body at finite strain.
    if setting.dynamic:
        options["density"] = density
    if weight is not None:
        options["body_force"] = weight
    if "displacement" in initial:
        options["initial_condition"] = {"displacement": list(initial["displacement"])}
    return problem.add_physics("solid_mechanics", **options)


def _add_fluid_dynamics(problem, setting: _physics_setting, entry: dict, density, weight):
    """Adds fluid_dynamics, with its density, its viscosity and its weight under gravity."""
    options = {
        "density": density,
        "dynamic_viscosity": constant_everywhere(setting.properties, "dynamic_viscosity"),
        "formulation": entry.get("formulation"),
    }
    if weight is not None:
        options["body_force"] = weight
    initial = setting.initial
    flow_initial = {key: initial[key] for key in ("velocity", "pressure") if key in initial}
    if flow_initial:
        options["initial_condition"] = flow_initial
    return problem.add_physics("fluid_dynamics", **options)


def _expansion_coefficient(properties: dict, coordinate_system: str, variables: dict):
    """The linear expansion coefficient of the thermal expansion as the engine takes it: a number,
    or the text of a formula of the temperature, the other fields and the coordinates. It is the
    secant coefficient between the stress-free temperature and the temperature, one value of the
    solid, so it is the same in every region and takes no table."""
    path = join("properties", "linear_expansion_coefficient")
    entry = properties["linear_expansion_coefficient"]
    value = _value_everywhere(properties, "linear_expansion_coefficient", formula=True)
    if isinstance(value, (int, float)):
        return float(value)
    return engine_formula(value, entry["unit"], path, coordinate_system, variables)


def _plasticity(properties: dict, entry: dict, coordinate_system: str, heated: bool) -> dict:
    """The options of solid_mechanics for J2 plasticity: the yield stress, a formula of
    equivalent_plastic_strain and temperature that holds the isotropic hardening, and the
    temperature field that it reads."""
    path = join("properties", "yield_stress")
    if entry["kinematics"] == "finite":
        fail(
            path,
            "a plastic solid of the input has small strain, because at finite strain the solid "
            "module updates a plastic stress by a hypoelastic rate, which has no multiplicative "
            "split of the deformation gradient.",
        )
    value = _value_everywhere(properties, "yield_stress", formula=True)
    options = {"plasticity": {"yield_stress": engine_value(value, "Pa", path, coordinate_system)}}
    node = getattr(value, "node", None)
    if node is not None and any(
        leaf.kind == "field" and leaf.name == "temperature" for leaf in graph.leaves(node)
    ):
        if not heated:
            fail(
                path,
                "the yield stress reads the temperature, which heat_conduction solves for, and "
                "the problem has no heat_conduction.",
            )
        options["temperature"] = "temperature"
    return options


def _strain_energy(text: str, path: str) -> dict:
    """The options of solid_mechanics for the strain energy of the input: the elasticity of a
    named law, or the formula of the energy per unit volume in the invariants of the elastic part
    of the deformation, which the engine differentiates."""
    if text in HYPERELASTIC:
        return {"elasticity": text}
    pressure = units.parse_unit("Pa")[1]
    leaves = {
        "I_C": units.dimensionless,
        "trace_C_squared": units.dimensionless,
        "J": units.dimensionless,
        "shear_modulus": pressure,
        "first_Lame_parameter": pressure,
    }
    symbols = {name: field_component(name, (), dimension) for name, dimension in leaves.items()}
    try:
        energy = parse(text, symbols)
    except InputError as error:
        raise InputError(f"{path}: {error}") from None
    if energy.rank != 0 or energy.dimension != pressure:
        fail(
            path,
            f"the strain energy '{text}' must be a scalar in J/m^3, "
            f"and it is in {units.SI_unit_text(energy.dimension)}.",
        )
    return {"strain_energy": write_expression(energy, path)}


def _solid_formulation(entry: dict, engine_mesh, coordinate_system: str, path: str):
    """The formulation of the solid: plane strain or plane stress on a plane mesh, and none
    elsewhere, where the engine takes it from the mesh."""
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
        return None
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
            # The penalty formulation eliminates the pressure, so it has no pressure variable.
            if physics.pressure in names:
                variables["pressure"] = [physics.pressure]
    return variables


# Boundary conditions


def _add_boundary_conditions(
    conditions: dict,
    engine_physics: dict,
    variables: dict,
    coordinate_system: str,
    problem,
    models: list,
    validated: dict,
):
    # The unknown of a model takes its value, and its flux_name the outward normal flux. Their
    # formulas may read the names that the models declare.
    unknowns = {model.unknown: model for model in models}
    fluxes = {model.flux_name: model for model in models if model.flux_name is not None}
    symbols = user_model.formula_symbols(validated) if models else None
    owner = {
        field: name
        for name, fields in physics_fields.items()
        for field in fields
        if name in engine_physics
    }
    for boundary, entries in conditions.items():
        if entries == symmetry:
            _add_symmetry(boundary, engine_physics)
            continue
        for key, value in entries.items():
            path = join(join("boundary_conditions", boundary), key)
            if key in unknowns or key in fluxes:
                model = unknowns.get(key) or fluxes[key]
                unit = model.unit if key in unknowns else model.flux_unit
                text = engine_value(value, unit, path, coordinate_system, variables, symbols)
                add_boundary_condition(
                    problem, boundary, model.unknown, text, flux=key not in unknowns
                )
                continue
            texts, physics_name = _condition_texts(
                key, value, variables, owner, coordinate_system, path
            )
            for text in texts:
                engine_physics[physics_name].add_boundary_condition(text, boundary=boundary)


def _add_symmetry(boundary: str, engine_physics: dict) -> None:
    """A plane of symmetry, where each physics applies its own condition. The solid holds the
    displacement normal to the boundary at zero, and the flow the normal velocity, which leaves
    the tangential traction zero. The heat flux and the flux of a model, whose unknown is a
    scalar, are zero there, the natural condition of their equations, so they take no object."""
    for name in ("solid_mechanics", "fluid_dynamics"):
        if name in engine_physics:
            engine_physics[name].add_boundary_condition(symmetry, boundary=boundary)


def _condition_texts(key, value, variables, owner, coordinate_system, path):
    """The texts of the engine's boundary conditions of one key of the input, and the physics they belong to."""
    if key in ("heat_flux", "convection", "radiation"):
        if "heat_conduction" not in owner.values():
            fail(path, "a heat flux needs heat_conduction.")
        return [f"heat_flux = {_heat_flux(key, value, coordinate_system, path)}"], "heat_conduction"
    if key == "pressure" and "pressure" not in owner and "solid_mechanics" in owner.values():
        # Without a flow, the pressure on a boundary is the load -p n on the solid.
        return [
            f"pressure = {engine_value(value, 'Pa', path, coordinate_system, variables)}"
        ], "solid_mechanics"
    if key == "traction":
        if "solid_mechanics" not in owner.values():
            fail(path, "a traction needs solid_mechanics.")
        components = [
            engine_value(component, "Pa", path, coordinate_system, variables)
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
            f"{names[component[0]]} = {engine_value(value, unit, path, coordinate_system, variables)}"
        ], owner[base_name]
    if len(names) == 1:
        return [
            f"{names[0]} = {engine_value(value, unit, path, coordinate_system, variables)}"
        ], owner[base_name]
    listed = _listed(value, path)
    if len(listed) != len(names):
        fail(path, f"{base_name} takes {len(names)} components, got {len(listed)}.")
    return [
        f"{name} = {engine_value(entry, unit, path, coordinate_system, variables)}"
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
    sigma = units.standard_names()["Stefan_Boltzmann_constant"].value
    return f"{sigma!r}*{emissivity}*(temperature^4 - ({ambient})^4)"


# Time and postprocessors


def transient_options(time: dict) -> dict:
    """The keywords of the engine's transient solve of the time section. A step control or a
    setting of one that the input leaves out takes the engine's default."""
    options = {"end_time": time["end_time"], "time_step": time["initial_time_step"]}
    control = time.get("step_control")
    if control is not None:
        stepper, setting = {
            "error": ("error", "error_tolerance"),
            "iterations": ("iteration", "optimal_iterations"),
            "fixed": ("fixed", None),
        }[control["by"]]
        options["time_stepper"] = stepper
        given = [value for key, value in control.items() if key != "by"]
        if given:
            options[setting] = given[0]
    return options


def add_postprocessors(problem, validated: dict, variables: dict) -> None:
    """Adds the postprocessors of the input that the engine computes, each with its SI unit.

    The models of the tree name the states, the expressions and the fluxes that a postprocessor
    may read. A state or an expression lives at the integration points, so its integral,
    average, maximum and minimum read the average of the points of each cell. A stress component
    or the von Mises stress at a point is that of the stress of the solid in the cell that holds
    the point, and the traction over a boundary is the reaction force of the solid there."""
    quantities, model_fluxes = {}, {}
    if validated.get("models"):
        quantities = user_model.quantity_units(validated)
        model_fluxes = user_model.flux_unknowns(validated)
    coordinate_system = validated["mesh"]["coordinate_system"]
    # An axisymmetric integral spans the revolved body (the factor 2 pi r), so it has the
    # dimension of a three-dimensional one.
    measured = 3 if coordinate_system == "axisymmetric" else problem.mesh.dimension
    names = set(validated.get("postprocessors", {}))
    for name, definition in validated.get("postprocessors", {}).items():
        if "boundary_flux" in definition:
            flux = definition["boundary_flux"]
            if (units.split_component(flux, coordinate_system) or (flux,))[0] == "traction":
                _add_traction(
                    problem, name, definition, variables, coordinate_system, measured, names
                )
            else:
                _add_boundary_flux(
                    problem, name, definition, variables, measured, quantities, model_fluxes
                )
            continue
        if "line" in definition:
            continue
        reduction = next(key for key in field_reductions if key in definition)
        field = definition[reduction]
        if field in quantities and field not in variables:
            _add_model_quantity(problem, name, reduction, quantities[field], field, measured)
        elif reduction == "point_value":
            _add_point_value(problem, name, definition, variables, coordinate_system)
        else:
            path = join("postprocessors", name)
            variable, unit = _scalar_variable(field, variables, coordinate_system, problem, path)
            problem.add_postprocessor(
                reduction, name, variable=variable, unit=_reduction_unit(unit, reduction, measured)
            )


def _reduction_unit(unit: str, reduction: str, measured: int) -> str:
    """The unit of a reduction of a field in `unit`: an integral over `measured` dimensions
    takes the unit times m^measured, and the average, maximum, minimum and point value the unit
    itself."""
    return f"{unit}*m^{measured}" if reduction == "integral" else unit


def _add_boundary_flux(
    problem, name: str, definition: dict, variables: dict, measured, quantities, model_fluxes
):
    """The integral of the outward normal flux of a model, or of the heat flux, over a
    boundary of measured - 1 dimensions."""
    flux = definition["boundary_flux"]
    if flux in model_fluxes:
        flow = units.parse_unit(quantities[flux])[1] * units.base("length") ** (measured - 1)
        variable, unit = model_fluxes[flux], units.SI_unit_text(flow)
    else:
        if "temperature" not in variables:
            fail(join("postprocessors", name), "the boundary flux heat_flux needs heat_conduction.")
        variable = variables["temperature"][0]
        unit = heat_flow_units[measured]
    problem.add_postprocessor(
        "boundary_flux", name, variable=variable, boundary=definition["boundary"], unit=unit
    )


def _add_model_quantity(problem, name: str, reduction: str, unit: str, field: str, measured):
    """A reduction of a state or an expression of a model, which lives at the integration
    points, so it takes no value at a point."""
    if reduction == "point_value":
        fail(
            join("postprocessors", name),
            f"{field} lives at the integration points, so it takes an integral, an "
            "average, a maximum or a minimum.",
        )
    problem.add_postprocessor(
        f"property_{reduction}",
        name,
        property=field,
        unit=_reduction_unit(unit, reduction, measured),
    )


def _add_point_value(problem, name: str, definition: dict, variables: dict, coordinate_system):
    """The value of a field, of a stress component or of the von Mises stress at a point."""
    path = join("postprocessors", name)
    field = definition["point_value"]
    kind, source, unit = point_quantity(field, variables, coordinate_system, problem, path)
    point = list(definition["point"])
    if kind == "stress":
        problem.add_postprocessor(
            "property_point_value",
            name,
            property="stress",
            component=source,
            point=point,
            unit=unit,
        )
    elif kind == "von_Mises_stress":
        problem.add_postprocessor("von_mises_stress_point_value", name, point=point, unit=unit)
    else:
        point += [0.0] * (3 - len(point))
        problem.add_postprocessor("point_value", name, variable=source, point=point, unit=unit)


def _add_traction(
    problem, name: str, definition: dict, variables: dict, coordinate_system, measured, names
):
    """The integral of the traction of the solid over a boundary: the force on the body through
    it, which is the reaction where the boundary prescribes the displacement and the load where
    it prescribes the traction. It is the reaction_force of the solid module, the residual of
    the equations at the nodes of the boundary, so it balances the loads to the tolerance of the
    solve. The vector traction gives one postprocessor per component, named <name>_x, <name>_y
    and <name>_z, or <name>_r and <name>_z in axisymmetric coordinates."""
    path = join("postprocessors", name)
    if "displacement" not in variables:
        fail(path, "the boundary flux traction needs solid_mechanics.")
    letters = units.position_names[coordinate_system][: problem.mesh.dimension]
    split = units.split_component(definition["boundary_flux"], coordinate_system)
    if split is None:
        components = {f"{name}_{letter}": letter for letter in letters}
    elif split[1][0] < len(letters):
        components = {name: letters[split[1][0]]}
    else:
        fail(path, f"the solid has the traction components {', '.join(letters)}.")
    # An axisymmetric integral spans the revolved surface, and a plane one is per unit thickness.
    unit = "N" if measured == 3 else "N/m"
    for component_name, letter in components.items():
        if component_name != name and component_name in names:
            fail(path, f"the component {component_name} takes the name of another postprocessor.")
        problem.add_postprocessor(
            "reaction_force",
            component_name,
            boundary=definition["boundary"],
            direction=letter,
            unit=unit,
        )


def _scalar_variable(field: str, variables: dict, coordinate_system: str, problem, path: str):
    """The engine's variable of a scalar field or of one component of a field, with its unit."""
    base_name, component = units.split_component(field, coordinate_system) or (field, ())
    if base_name in ("stress", "von_Mises_stress"):
        fail(
            path,
            "a stress is available at points, as the point_value or the line of one component, "
            "such as stress_yy, or of von_Mises_stress.",
        )
    if base_name not in variables or (len(variables[base_name]) > 1 and not component):
        fail(
            path,
            f"a postprocessor reads one scalar field or component, and {field} is not one of the "
            "problem.",
        )
    variable = variables[base_name][component[0] if component else 0]
    standard = units.standard_names().get(base_name)
    return variable, standard.unit if standard is not None else problem.variable_unit(variable)


def point_quantity(field: str, variables: dict, coordinate_system: str, problem, path: str):
    """What a value at a point reads, as (kind, source, unit): ("variable", the engine's variable,
    its unit) for a scalar field or one component of a field, ("stress", the Voigt index of the
    component, "Pa") for one component of the Cauchy stress of the solid, which is the stress in
    the cell that holds the point, and ("von_Mises_stress", None, "Pa") for the von Mises stress
    of that stress."""
    if field == "von_Mises_stress":
        if "displacement" not in variables:
            fail(path, "a stress needs solid_mechanics.")
        return "von_Mises_stress", None, "Pa"
    base_name, component = units.split_component(field, coordinate_system) or (field, ())
    if base_name != "stress":
        return ("variable", *_scalar_variable(field, variables, coordinate_system, problem, path))
    if len(component) != 2:
        fail(path, "a stress at a point is one component, such as stress_yy.")
    if "displacement" not in variables:
        fail(path, "a stress needs solid_mechanics.")
    # The Voigt order of the engine: xx, yy, zz, yz, xz and xy, or rr, zz, the hoop component
    # and rz in axisymmetric coordinates.
    voigt = ((0, 5, 4), (5, 1, 3), (4, 3, 2))
    return "stress", voigt[component[0]][component[1]], "Pa"


class line_sampler:
    """The values of one field at `count` evenly spaced points from `start` to `end`, both ends
    included, as the rows of a table: the time in a transient run, the distance along the line,
    the coordinates and the field, each with its unit."""

    def __init__(self, name: str, definition: dict, variables: dict, validated: dict, problem):
        coordinate_system = validated["mesh"]["coordinate_system"]
        path = join("postprocessors", name)
        self.name = name
        self.field = definition["line"]
        self.kind, self.source, unit = point_quantity(
            self.field, variables, coordinate_system, problem, path
        )
        self.start = numpy.asarray(definition["from"], dtype=float)
        self.end = numpy.asarray(definition["to"], dtype=float)
        self.fractions = numpy.linspace(0.0, 1.0, definition["points"])
        self.transient = "time" in validated
        coordinates = list(units.position_names[coordinate_system][: len(self.start)])
        self.columns = [*(["time"] if self.transient else []), "distance", *coordinates]
        self.columns.append(self.field)
        self.units = [*(["s"] if self.transient else []), "m", *(["m"] * len(coordinates)), unit]
        self.rows = []

    def sample(self, engine, time: float) -> None:
        """Adds the rows of the line at `time`. Every process of a distributed run samples."""
        points = self.start + numpy.outer(self.fractions, self.end - self.start)
        if self.kind == "stress":
            values = engine.property_at_points("stress", points)[:, self.source]
        elif self.kind == "von_Mises_stress":
            values = von_Mises_stress(engine.property_at_points("stress", points))
        else:
            values = engine.sample(self.source, points)
        distances = self.fractions * float(numpy.linalg.norm(self.end - self.start))
        for distance, point, value in zip(distances, points, values):
            row = [distance, *point, value]
            self.rows.append([time, *row] if self.transient else row)

    def table(self):
        return Table(self.columns, self.units, self.rows, title=self.name)


def line_samplers(problem, validated: dict, variables: dict) -> list:
    """The line postprocessors of the input."""
    return [
        line_sampler(name, definition, variables, validated, problem)
        for name, definition in validated.get("postprocessors", {}).items()
        if "line" in definition
    ]
