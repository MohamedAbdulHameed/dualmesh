"""Lowers the models of your own of the input onto the objects of the engine.

A model is the balance law dm/dt + div F = s of one scalar unknown u, with internal states q
that follow the rate laws dq/dt = g at each point (`dualmesh.model.conservation_law`). The
properties of each region are substituted into its terms, which are then expanded into scalar
components, differentiated where the engine needs a derivative (`dualmesh.model.differentiate`)
and written in the syntax of the engine (`lower.write_expression`). `add_law` lowers the law on
each region where its properties differ, or on the whole mesh where they do not, as it lowers the
equations of general_form_PDE, and `add_boundary_condition` lowers a condition on the unknown or
its flux_name. A parsed_tensor property of one component holds each expression of the model that
a postprocessor reads.
"""

import collections
import importlib
from collections.abc import Mapping

from dualmesh.input import units
from dualmesh.input.lower import formula_node, write_expression
from dualmesh.input.schema import context, eigenvalue_symbol, generated_region, join
from dualmesh.input.units import fail, input_error
from dualmesh.model import conservation_law_from_input, differentiate, parse
from dualmesh.model.dimensions import is_zero

# import_module returns the module itself, which dualmesh.model hides behind the class of the
# same name.
graph = importlib.import_module("dualmesh.model.expression")

# The leaves that a model of your own does not take: the step and the cell size are numbers of
# the discretisation, and the normal exists on a boundary only.
discretisation_leaves = ("time_step", "cell_size", "normal", "normal_component")


class property_symbols(Mapping):
    """The names that the formulas of one model read on one region.

    A property becomes the expression of its value on the region: a number with the unit of the
    property, or its formula, parsed with these same names, so a property that reads another
    one reads its value. A state of the model is the state node of its law. Every other name
    falls back to `base`, the declared names of the tree. `reads` collects the properties read.
    """

    def __init__(self, properties: dict, region, base: Mapping, own: dict, coordinate_system):
        self.properties = properties
        self.region = region
        self.base = base
        self.own = own
        self.coordinate_system = coordinate_system
        self.cache = {}
        self.building = []
        self.reads = set()

    def __contains__(self, name):
        return name in self.own or name in self.properties or name in self.base

    def __getitem__(self, name):
        if name in self.own:
            return self.own[name]
        if name in self.properties:
            return self.property_expression(name)
        return self.base[name]

    def __iter__(self):
        return iter({*self.own, *self.properties, *self.base})

    def __len__(self):
        return len({*self.own, *self.properties, *self.base})

    def property_expression(self, name: str):
        if name in self.cache:
            return self.cache[name]
        path = join("properties", name)
        if name in self.building:
            fail(path, "the formula reads itself through " + ", ".join(self.building) + ".")
        entry = self.properties[name]
        value = entry["value"]
        if entry["regions"]:
            if self.region not in entry["regions"]:
                where = (
                    f"the region '{self.region}'"
                    if self.region is not None
                    else "a boundary condition, which takes one value everywhere"
                )
                fail(path, f"a model reads this property, which has no value for {where}.")
            value = entry["regions"][self.region]
            path = join(path, self.region)
        self.reads.add(name)
        if isinstance(value, dict):
            fail(path, "a model of your own reads constants and formulas, so it takes no table.")
        self.building.append(name)
        try:
            if isinstance(value, (int, float)):
                node = graph.number(float(value), units.parse_unit(entry["unit"])[1])
            else:
                try:
                    node = formula_node(value, self, self.coordinate_system)
                except input_error as error:
                    raise type(error)(f"{path}: {error}") from None
        finally:
            self.building.pop()
        self.cache[name] = node
        return node


def own_states(entry: dict) -> dict:
    """The state nodes of a model entry, built as conservation_law_from_input builds them."""
    states = {}
    for state_name, state_entry in (entry.get("states") or {}).items():
        if isinstance(state_entry, dict):
            states[state_name] = graph.state(
                state_name,
                unit=state_entry.get("unit"),
                initial=state_entry.get("initial", 0.0),
                shape=state_entry.get("shape", "scalar"),
            )
    return states


def model_expressions(entry: dict, law, symbols: Mapping, coordinate_system: str) -> dict:
    """The expressions of a model entry, each as the expression it names, in the order given."""
    local = {law.unknown.name: law.unknown, **{node.name: node for node in law.states()}}
    names = collections.ChainMap(local, symbols)
    result = {}
    for name, value in (entry.get("expressions") or {}).items():
        if isinstance(value, dict):
            value = value.get("value")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            node = graph.number(float(value))
        else:
            node = parse(value, names, coordinate_system)
        local[name] = node
        result[name] = node
    return result


class lowered_model:
    """One model of the input, with its law on each region, ready for the engine."""

    def __init__(self, name: str, entry: dict):
        self.name = name
        self.entry = entry
        self.path = join("models", name)
        self.laws = {}  # Region, or None for the whole mesh, to its conservation law
        self.expressions = {}  # Region, or None, to the expressions of the model
        self.reads = set()

    @property
    def law(self):
        return next(iter(self.laws.values()))

    @property
    def unknown(self) -> str:
        return self.law.unknown.name

    @property
    def unit(self) -> str:
        return units.SI_unit_text(self.law.unknown.dimension)

    @property
    def flux_name(self):
        return self.entry.get("flux_name")

    @property
    def flux_unit(self) -> str:
        return units.SI_unit_text(self.law.flux.dimension)

    def initial_states(self) -> list:
        return [node.initial for node in self.law.states()]


def prepare(validated: dict, engine_mesh) -> list:
    """The models of the tree with their laws on each region and the properties they read.

    A model takes the value of each property on each region of the mesh. Where no property it
    reads differs between regions, one law holds on the whole mesh.
    """
    entries = validated.get("models", {})
    if not entries:
        return []
    coordinate_system = validated["mesh"]["coordinate_system"]
    properties = validated.get("properties", {})
    base = context(validated, complete=True).model_symbols
    if "file" in validated["mesh"]:
        regions = sorted(set(engine_mesh.block_names().values())) or [None]
    else:
        regions = [generated_region]
    prepared = []
    for name, entry in entries.items():
        model = lowered_model(name, entry)
        for region in regions:
            symbols = property_symbols(
                properties, region, base, own_states(entry), coordinate_system
            )
            law = conservation_law_from_input(name, entry, symbols, coordinate_system)
            model.laws[region] = law
            model.expressions[region] = model_expressions(entry, law, symbols, coordinate_system)
            model.reads |= symbols.reads
        laws = list(model.laws.values())
        if all(_terms(law) == _terms(laws[0]) for law in laws) and all(
            expressions == model.expressions[regions[0]]
            for expressions in model.expressions.values()
        ):
            model.laws = {None: laws[0]}
            model.expressions = {None: model.expressions[regions[0]]}
        prepared.append(model)
    return prepared


def _terms(law) -> tuple:
    return (
        law.storage,
        law.inertia,
        law.flux,
        law.source,
        tuple((node, rate) for node, rate in sorted(law.state_rates.items(), key=_by_name)),
    )


def _by_name(item):
    return item[0].name


def check(models: list, physics_fields: set, transient: bool, initial: dict) -> None:
    """Refuses what the engine cannot run, with the key of the input that asks for it."""
    unknowns = {}
    for model in models:
        law = model.law
        if law.unknown.rank != 0:
            fail(
                join(model.path, "unknown"),
                "the unknown of a model of your own is a scalar.",
            )
        name = law.unknown.name
        if name in physics_fields or name in unknowns:
            other = "a physics" if name in physics_fields else f"the model {unknowns[name]}"
            fail(join(model.path, "unknown"), f"{name} is the unknown of {other} as well.")
        unknowns[name] = model.name
        for node in law.states():
            if node.rank != 0:
                fail(
                    f"{model.path}.states.{node.name}",
                    "a state of a model of your own is a scalar.",
                )
        if not is_zero(law.inertia):
            fail(
                join(model.path, "inertia"),
                "an inertia is not available in a model of your own, which holds a storage, a "
                "flux, a source and states.",
            )
        if law.boundary_fluxes:
            fail(
                join(model.path, "boundary_fluxes"),
                "a flux through a boundary is given in boundary_conditions, under the flux_name "
                "of the model, as a constant or a formula of the fields.",
            )
        if transient and not is_zero(law.storage) and name not in initial:
            fail(
                "initial_conditions", f"the transient model {model.name} needs the initial {name}."
            )
        if not transient and is_zero(law.flux) and is_zero(law.source):
            fail(model.path, "a steady model needs a flux or a source.")
    for name in initial:
        state_owner = next(
            (model for model in models if any(node.name == name for node in model.law.states())),
            None,
        )
        if state_owner is not None:
            fail(
                join("initial_conditions", name),
                f"a state takes its initial value in {join(state_owner.path, 'states')}.{name}.initial.",
            )


def add_variables(problem, models: list, initial: dict, variables: dict) -> None:
    """Adds the unknown of each model as a variable of the engine, in its SI unit, with its initial
    value: a number or the engine's text of a formula of the coordinates."""
    for model in models:
        name = model.unknown
        start = initial.get(name)
        problem.add_variable(name, initial_condition=start, unit=model.unit)
        variables[name] = [name]


def add(problem, models: list, variables: dict, transient: bool, dimension: int, read: set) -> None:
    """Adds the objects of each model. `read` holds the names that the postprocessors read."""
    others = {node.name: model.name for model in models for node in model.law.states()}
    for model in models:
        for region, law in model.laws.items():
            restriction = {} if region is None else {"block": [region]}
            where = model.path if region is None else f"{model.path} on the region {region}"
            _check_leaves(model, law.cell_expressions(), law, variables, others, where)
            if not transient:
                _check_steady_states(model, law, dimension)
            add_law(problem, law, variables, dimension, transient, restriction, where)
            for name, node in model.expressions[region].items():
                if name not in read:
                    continue
                path = f"{model.path}.expressions.{name}"
                if node.rank != 0:
                    fail(path, "a postprocessor reads a scalar expression.")
                _check_leaves(model, [node], law, variables, others, path)
                problem.add_property(
                    "parsed_tensor",
                    property_name=name,
                    components=[write_expression(graph.expand(node, dimension), path, variables)],
                    **restriction,
                )


def _check_leaves(model, nodes, law, variables, others, path) -> None:
    own = {node.name for node in law.states()}
    for node in graph_leaves(nodes):
        if node.kind in discretisation_leaves:
            fail(path, f"the model reads {node.kind}, which a model of your own does not take.")
        if node.kind == "constant" and node.value is None and node.name != eigenvalue_symbol:
            fail(path, f"the constant {node.name} has no value.")
        if node.kind in ("field", "field_component", "gradient_component"):
            if node.name in others and node.name not in own:
                fail(
                    path,
                    f"the model reads {node.name}, a state of the model {others[node.name]}. "
                    "A model reads its own states.",
                )
            if node.name not in variables:
                fail(
                    path,
                    f"the model reads {node.name}, which is neither a property of the input nor "
                    "a field that a physics or a model of the problem solves for.",
                )


def _check_steady_states(model, law, dimension: int) -> None:
    """Refuses a state of a steady problem whose rate reads no state, which has no root."""
    states = law.states()
    components = [graph.state_component(node.name, (), node.dimension) for node in states]
    for node in states:
        rate = graph.expand(law.state_rates[node], dimension)
        if all(is_zero(differentiate(rate, component)) for component in components):
            fail(
                f"{model.path}.states.{node.name}.rate",
                "a steady problem holds each state at the root of its rate, and this rate "
                "reads no state, so it has no root. Give the time section, or a rate that "
                "reads its state.",
            )


def graph_leaves(nodes) -> set:
    found = set()
    for node in nodes:
        found |= graph.leaves(node)
    return found


def add_law(problem, law, variables: dict, dimension: int, transient: bool, restriction, where):
    """Adds the objects of the engine of one conservation law, on the blocks of restriction.

    This is the one lowering of a conservation law, for the models of the input and the
    equations of general_form_PDE alike. `variables` maps each field to the engine's names of
    its variables, and `where` names the law in messages. The law becomes:

    - a parsed_state property object, which integrates the states at each integration point by
      the backward Euler method and holds each as the property of its name;
    - for each scalar component of the unknown, with the terms of `conservation_law.components`:
      a second_time_derivative kernel with the coefficient of the inertia, a time_derivative
      kernel with the capacity c of a storage written as an antiderivative (the rate form
      c du/dt) or, in a transient problem, with any other storage m (the difference
      (m - m_old)/dt), and a parsed_kernel with the flux and the source.
    """

    def write(node) -> str:
        return write_expression(node, where, variables)

    states = law.states()
    state_object = None
    if states:
        components = [graph.state_component(node.name, (), node.dimension) for node in states]
        rates = [graph.expand(law.state_rates[node], dimension) for node in states]
        derivatives = [differentiate(rate, component) for rate in rates for component in components]
        state_object = problem.add_property(
            "parsed_state",
            states=[node.name for node in states],
            rates=[write(rate) for rate in rates],
            rate_derivatives=[write(entry) for entry in derivatives],
            initial_values=[float(node.initial) for node in states],
            **restriction,
        )
    names = variables[law.unknown.name]
    for variable, terms in zip(names, law.components(dimension), strict=True):
        if terms.mass is not None:
            problem.add_kernel(
                "second_time_derivative",
                variable=variable,
                coefficient=write(terms.mass),
                **restriction,
            )
        if terms.capacity is not None:
            problem.add_kernel(
                "time_derivative",
                variable=variable,
                coefficient=write(terms.capacity),
                **restriction,
            )
        if transient and terms.storage is not None:
            storage = {"variable": variable, "storage": write(terms.storage), **restriction}
            if state_object is not None:
                storage["state_object"] = state_object.name
            problem.add_kernel("time_derivative", **storage)
        kernel = {}
        if terms.flux is not None:
            kernel["flux"] = [write(component) for component in terms.flux]
        if terms.source is not None:
            kernel["source"] = write(terms.source)
        if kernel:
            advective = _advective(terms.flux or [], terms.source, terms.unknown)
            problem.add_kernel(
                "parsed_kernel", variable=variable, **kernel, advective=advective, **restriction
            )


def add_boundary_condition(
    problem, boundary, variable: str, value: str, flux: bool, **parameters
) -> None:
    """Adds the condition of one variable on a boundary: its value, or with flux the outward
    normal flux F . n of its law. Both the models of the input and general_form_PDE take it."""
    if flux:
        # Neumann takes the flux entering the body, -F . n.
        problem.add_boundary_condition(
            "Neumann", boundary, variable=variable, flux=f"-({value})", **parameters
        )
    else:
        problem.add_boundary_condition(
            "Dirichlet", boundary, variable=variable, value=value, **parameters
        )


def _advective(flux: list, source, unknown) -> bool:
    """Whether the term carries the unknown: a flux left where the gradient of the unknown is zero,
    or a source that reads that gradient."""
    if source is not None and any(
        leaf.kind == "gradient_component" and leaf.name == unknown.name
        for leaf in graph.leaves(source)
    ):
        return True
    for component in flux:
        gradients = {
            leaf: graph.number(0.0, leaf.dimension)
            for leaf in graph.leaves(component)
            if leaf.kind == "gradient_component" and leaf.name == unknown.name
        }
        if not is_zero(graph.substitute(component, gradients)):
            return True
    return False


def formula_symbols(validated: dict) -> Mapping:
    """The names that a boundary condition of a model reads: the properties with one value, and
    the declared names of the tree."""
    return property_symbols(
        validated.get("properties", {}),
        None,
        context(validated, complete=True).symbols,
        {},
        validated["mesh"]["coordinate_system"],
    )


def quantity_units(validated: dict) -> dict:
    """The SI unit of each state, expression and flux_name of the models, by name."""
    tree_context = context(validated, complete=True)
    result = {}
    for name, entry in validated.get("models", {}).items():
        for section in ("states", "expressions"):
            for quantity in entry.get(section) or {}:
                if quantity in tree_context.declared or quantity in units.standard_names():
                    result[quantity] = tree_context.quantity_of(
                        quantity, f"models.{name}.{section}.{quantity}"
                    )[1]
        flux_name = entry.get("flux_name")
        if isinstance(flux_name, str) and (
            flux_name in tree_context.declared or flux_name in units.standard_names()
        ):
            result[flux_name] = tree_context.quantity_of(flux_name, f"models.{name}")[1]
    return result


def flux_unknowns(validated: dict) -> dict:
    """The unknown of each model, by the flux_name of the model."""
    result = {}
    for entry in validated.get("models", {}).values():
        flux_name, unknown = entry.get("flux_name"), entry.get("unknown")
        if isinstance(unknown, dict) and len(unknown) == 1:
            unknown = next(iter(unknown))
        if isinstance(flux_name, str) and isinstance(unknown, str):
            result[flux_name] = unknown
    return result
