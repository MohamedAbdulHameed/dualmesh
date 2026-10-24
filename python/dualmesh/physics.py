# SPDX-License-Identifier: LGPL-2.1-or-later
r"""The physics level: sets of equations named in physical terms.

A physics is added to a problem by its type name, creates its variables, and
generates the kernels and the property objects of its equations when the problem is
initialised::

    import dualmesh as dm

    problem = dm.Problem(dm.read_mesh("wrench.msh"), method="fem")
    solid = problem.add_physics("solid_mechanics", youngs_modulus=200.0e9, poissons_ratio=0.3)
    solid.add_boundary_condition("fixed", boundary="jaws")
    solid.add_boundary_condition("total_force = [0, -150, 0]", boundary="grip")
    problem.solve()

Each physics defines its variables: temperature (heat_transfer); disp_x,
disp_y, disp_z (solid_mechanics); velocity_x, velocity_y, velocity_z and
pressure (fluid_dynamics); the fields of general_form_PDE are named in
its ``variables``.  The boundary conditions of a physics are written as text: a variable on the
left prescribes its value, and the name of the flux of the physics on the
left prescribes the flux; a vector (``traction``, ``total_force``) is split
into one object per component.  The extra terms of a physics take the type
names and the parameters of the objects (:doc:`/objects`), with the
variables of the physics filled in.  A coupling between two physics
(``thermal_expansion`` or ``nonisothermal_flow``) is added with
:meth:`dualmesh.Problem.add_coupling`.
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import ClassVar

from .parameters import InputError, check_keywords, describe_fields, parameter

__all__ = [
    "Coupling",
    "NonisothermalFlow",
    "HeatTransfer",
    "FluidDynamics",
    "Physics",
    "SolidMechanics",
    "ThermalExpansion",
    "create",
    "describe",
    "registered",
]

#: Type name -> class, for the physics and the couplings.
_REGISTRY: dict[str, type] = {}


def register(cls):
    """Register a physics or a coupling under its ``type_name``."""
    _REGISTRY[cls.type_name] = cls
    return cls


def registered(category: str | None = None) -> list[str]:
    """The registered type names, of one category (``physics`` or
    ``coupling``) or of both."""
    return sorted(
        name for name, cls in _REGISTRY.items() if category is None or cls.category == category
    )


def is_registered(type_name: str) -> bool:
    return type_name in _REGISTRY


def describe(type_name: str) -> str:
    """The description and the parameters of a physics or a coupling."""
    cls = _REGISTRY[type_name]
    doc = " ".join((cls.__doc__ or "").split("\n\n")[0].split())
    return f"{type_name} ({cls.category}, module {cls.module})\n{doc}\n\n{describe_fields(cls)}\n"


def create(type_name: str, name: str, parameters: dict):
    """A physics or a coupling of a registered type, with its parameters
    checked."""
    import difflib

    if type_name not in _REGISTRY:
        close = difflib.get_close_matches(type_name, list(_REGISTRY), n=1)
        hint = f" Did you mean '{close[0]}'?" if close else ""
        raise InputError(
            f"Unknown physics or coupling type '{type_name}'.{hint} Registered types: {' '.join(sorted(_REGISTRY))}"
        )
    cls = _REGISTRY[type_name]
    where = label(type_name, name)
    if getattr(cls, "takes_definitions", False):
        # Keywords that are not parameters are definitions, kept in their order; one close to a
        # parameter name is taken for a misspelling.
        import difflib

        names = [f.name for f in dataclasses.fields(cls) if not f.name.startswith("_")]
        definitions = {k: v for k, v in parameters.items() if k not in names}
        for key in definitions:
            close = difflib.get_close_matches(key, names, n=1, cutoff=0.8)
            if close and len(key) > 2:
                raise InputError(
                    f"{where}: '{key}' is not a parameter, and it is too close to the parameter '{close[0]}' to be taken for a definition. Did you mean '{close[0]}'?"
                )
        parameters = {k: v for k, v in parameters.items() if k in names}
        parameters["_definitions"] = definitions
    else:
        check_keywords(where, cls, parameters)
    for f in dataclasses.fields(cls):
        if (
            f.default is dataclasses.MISSING
            and f.default_factory is dataclasses.MISSING
            and f.name not in parameters
        ):
            raise InputError(
                f"{where}: missing required parameter '{f.name}' ({f.metadata.get('description', '')})"
            )
    instance = cls(**parameters)
    instance.name = name
    return instance


def label(type_name: str, name: str) -> str:
    """How a message names a physics or a coupling: its type, and its name when
    that differs from the type."""
    return type_name if name == type_name else f"{type_name} '{name}'"


def unique_name(type_name: str, taken) -> str:
    """The name of a new physics of the type ``type_name``: the type name, or
    the type name with a number when the problem has a physics of that type."""
    name, count = type_name, 1
    while name in taken:
        count += 1
        name = f"{type_name}_{count}"
    return name


#: Keywords of a boundary condition written as text that are passed on to its objects; any
#: other keyword is a definition that the text uses by its name.
_CONDITION_PARAMETERS = ("scale_with_load",)


def _expand_conditions(condition, parameters: dict, where: str) -> list[str]:
    """The texts of ``condition`` (one text or a list) with the definitions
    among ``parameters`` written into them, and those definitions removed
    from ``parameters``.  A number is written as its value and a text, in
    parentheses, as itself; a definition may use the ones before it."""
    import difflib

    texts = [condition] if isinstance(condition, str) else list(condition)
    names = [k for k in parameters if k not in _CONDITION_PARAMETERS]
    definitions = {k: parameters.pop(k) for k in names}
    if not definitions:
        return texts

    def substitute(text: str, known: dict) -> str:
        return re.sub(
            r"(?<![A-Za-z0-9_.])[A-Za-z_][A-Za-z0-9_]*",
            lambda m: known.get(m.group(0), m.group(0)),
            text,
        )

    def hint(name: str) -> str:
        close = difflib.get_close_matches(name, [*_CONDITION_PARAMETERS, "boundary"], n=1)
        return f" Did you mean '{close[0]}'?" if close else ""

    known: dict = {}
    for name, value in definitions.items():
        if isinstance(value, bool) or not isinstance(value, (int, float, str)):
            raise InputError(
                f"{where}: the definition {name} = {value!r} is not a number or an expression (a text).{hint(name)}"
            )
        known[name] = (
            repr(float(value)) if not isinstance(value, str) else f"({substitute(value, known)})"
        )
    used = set(
        re.findall(
            r"[A-Za-z_][A-Za-z0-9_]*",
            " ".join(texts + [v for v in definitions.values() if isinstance(v, str)]),
        )
    )
    for name in definitions:
        if name not in used:
            raise InputError(
                f"{where}: '{name}' is defined and not used by the condition.{hint(name)}"
            )
    return [substitute(text, known) for text in texts]


def _components(name: str, dimension: int, axisymmetric: bool = False) -> list[str]:
    """The variables of the vector field ``name``: name_x, name_y, name_z,
    or name_r, name_z in axisymmetric coordinates."""
    axes = ("r", "z") if axisymmetric else ("x", "y", "z")[:dimension]
    return [f"{name}_{axis}" for axis in axes]


def _field_kinds(variables, where: str) -> dict:
    """The fields of ``variables`` and their kinds: a name, or a list whose
    items are names (scalar fields) or one-entry dicts name -> kind, as
    ``[{"U": "vector"}, "p"]``."""
    items = [variables] if isinstance(variables, (str, Mapping)) else list(variables)
    kinds: dict = {}
    for item in items:
        if isinstance(item, Mapping):
            if len(item) != 1:
                raise InputError(
                    f"{where}: give each field as its own item of the list, a name or {{name: kind}}, e.g. variables=[{{'U': 'vector'}}, 'p'], not {dict(item)!r}."
                )
            ((name, kind),) = item.items()
        else:
            name, kind = item, "scalar"
        if name in kinds:
            raise InputError(f"{where}: every name in 'variables' must be different.")
        kinds[name] = kind
    return kinds


def _is_condition_text(condition) -> bool:
    """Whether ``condition`` is a boundary condition written as text, with a
    single '=' between its two sides."""
    return isinstance(condition, str) and re.search(r"(?<![<>=!])=(?!=)", condition) is not None


def _sides(condition: str) -> tuple[str, str]:
    match = re.search(r"(?<![<>=!])=(?!=)", condition)
    return condition[: match.start()].strip(), condition[match.end() :].strip()


def _list_items(text: str, size: int, where: str) -> list[str]:
    """The items of a vector written as text, [a, b, c], split at the commas
    outside parentheses."""
    text = text.strip()
    if not (text.startswith("[") and text.endswith("]")):
        raise InputError(f"{where}: give the vector as [a, b] with one value per component.")
    items, depth, current = [], 0, ""
    for char in text[1:-1]:
        if char == "," and depth == 0:
            items.append(current.strip())
            current = ""
            continue
        depth += char in "(["
        depth -= char in ")]"
        current += char
    items.append(current.strip())
    if len(items) != size:
        raise InputError(
            f"{where}: the vector has {len(items)} values, and it needs {size}, one per component."
        )
    return items


class Physics:
    """Base of the physics.  A subclass is a dataclass whose fields are its
    parameters, and it provides :meth:`variables` and :meth:`_build`."""

    category: ClassVar[str] = "physics"
    module: ClassVar[str] = "framework"
    type_name: ClassVar[str] = ""
    #: The physics has one variable per component of a vector.
    vector: ClassVar[bool] = False

    name: str = ""

    def variable_names(self) -> list[str]:
        raise NotImplementedError

    # ---- set up by Problem.add_physics ---------------------------------
    def _attach(self, problem) -> None:
        self._problem = problem
        self._built = False
        self._coupled: dict = {}
        # The boundaries of the word "fixed", and of each variable whose value a condition sets.
        self._clamped: list = []
        self._symmetric: list = []
        self._prescribed: dict = {}
        order = self._variable_order()
        for variable in self.variable_names():
            try:
                problem.variable_index(variable)
            except ValueError:
                problem.add_variable(
                    variable,
                    block=list(getattr(self, "block", ())),
                    initial_condition=self._initial_condition(variable),
                    order=order.get(variable, "mesh"),
                    unit=self._variable_units().get(variable, ""),
                )
                continue
            existing = problem.variable_order(variable)
            if existing != order.get(variable, "mesh"):
                raise InputError(
                    f"{label(self.type_name, self.name)}: the variable '{variable}' exists with order '{existing}', and this physics needs order '{order.get(variable, 'mesh')}'. Declare it with order='{order.get(variable, 'mesh')}', or let the physics create it."
                )
            # A field shared with another physics lives on the blocks of both.
            problem._problem.extend_variable_blocks(variable, list(getattr(self, "block", ())))

    def _variable_order(self) -> dict:
        return {}

    def _variable_units(self) -> dict:
        """The SI unit of each variable of the physics."""
        return {}

    def _initial_condition(self, variable: str):
        return None

    def _object_name(self, name: str) -> str:
        return f"{self.name}_{name}"

    # ---- conditions and extra terms ------------------------------------
    #: The name of the normal flux in a boundary condition written as text, e.g. heat_flux.
    flux_name: ClassVar[str] = ""
    #: Words of the physics for kernels of the object reference: word -> (type, {parameter of
    #: the physics: parameter of the object}), e.g., heat_source -> body_force.
    kernel_words: ClassVar[dict] = {}

    def add_boundary_condition(self, condition, boundary=None, **parameters):
        """Add a boundary condition, written as text, on the side sets or
        node sets named by ``boundary``.

        A variable of the physics on the left prescribes its value on the
        boundary, and the name of the normal flux of the physics on the left
        prescribes the flux through it, along the outward normal: ``heat_flux``
        (the heat leaving the body) for ``heat_transfer``, ``traction``,
        ``total_force`` and ``pressure`` for ``solid_mechanics``,
        ``normal_flux`` for ``general_form_PDE``.  The right side is an
        expression of x, y, z, t, the functions, the properties and the
        fields.  A list of texts adds each.  Three words need no data:
        ``"fixed"`` (every displacement zero), ``"symmetry"`` (the normal
        displacement zero) and ``"periodic"`` (with ``primary`` and
        ``secondary``).  Any other keyword is a definition that the text uses
        by its name, a number or an expression, as in ``add_physics``::

            heat.add_boundary_condition("temperature = 300", boundary="left")
            heat.add_boundary_condition("heat_flux = 75*(temperature - 293)", boundary="top")
            solid.add_boundary_condition("fixed", boundary="jaws")
            solid.add_boundary_condition("total_force = [0, -F, 0]", boundary="grip", F=150.0)
        """
        if isinstance(condition, (list, tuple)) or _is_condition_text(condition):
            where = f"{label(self.type_name, self.name)}, boundary condition {condition!r}"
            for text in _expand_conditions(condition, parameters, where):
                if not _is_condition_text(text):
                    raise InputError(f"{where}: '{text}' is not a condition written as text.")
                self._text_condition(text, boundary, dict(parameters))
            return None
        problem = self._problem
        where = f"{label(self.type_name, self.name)}, boundary condition '{condition}'"
        variables = self.variable_names()
        words = ["periodic"] + (["fixed", "symmetry"] if self.vector else [])
        if condition not in words:
            example = f"'{variables[0]} = 0'" + (
                f" or '{self.flux_name} = ...'" if self.flux_name else ""
            )
            raise InputError(
                f"{where}: write the condition as text, with a variable of this physics or the name of its flux on the left, for example {example}. The words without data are {', '.join(words)}."
            )
        if condition == "periodic":
            return problem.add_boundary_condition("periodic", variable=variables, **parameters)
        if boundary is None:
            mesh = problem.mesh
            names = sorted(set(mesh.sideset_names()) | set(mesh.nodeset_names()))
            raise InputError(f"{where}: give boundary, one or more of: {', '.join(names)}.")
        if parameters:
            raise InputError(f"{where}: the word '{condition}' takes no data, only the boundary.")
        if condition == "fixed":
            self._clamped += _names(boundary)
            return problem.add_boundary_condition(
                "Dirichlet", boundary, variable=self._fixed_variables(), value=0.0
            )
        return self._symmetry(boundary, where)

    def _symmetry(self, boundary, where: str):
        """The word ``symmetry``: the displacement normal to the boundary is zero."""
        components = self.variable_names()[: self._dimension()]
        return self._problem.add_boundary_condition("symmetry", boundary, variable=components)

    def _fixed_variables(self) -> list[str]:
        """The variables that the word ``fixed`` holds at zero."""
        return self.variable_names()[: self._dimension()]

    def _text_condition(self, condition: str, boundary, parameters: dict):
        """A boundary condition written as text: a variable of the physics
        on the left prescribes its value, and the name of the normal flux of
        the physics (``flux_name``) prescribes the flux."""
        problem = self._problem
        where = f"{label(self.type_name, self.name)}, boundary condition '{condition}'"
        if boundary is None:
            mesh = problem.mesh
            names = sorted(set(mesh.sideset_names()) | set(mesh.nodeset_names()))
            raise InputError(f"{where}: give boundary, one or more of: {', '.join(names)}.")
        left, right = _sides(condition)
        variables = self.variable_names()
        if getattr(self, "_structural", False) and left in ("traction", "total_force", "pressure"):
            raise InputError(
                f"{where}: the {self.formulation} formulation takes its loads as transverse_load, axial_load and in_plane_load of add_physics; a boundary condition prescribes its variables ({', '.join(variables)})."
            )
        if left in variables:
            self._prescribed.setdefault(left, []).extend(_names(boundary))
            return problem.add_boundary_condition(
                "Dirichlet", boundary, variable=left, value=right, **parameters
            )
        if self.vector and left == "pressure" and self.flux_name == "traction":
            # A pressure p on the surface: the traction -p n.
            return problem.add_boundary_condition(
                "pressure",
                boundary,
                displacements=variables[: self._dimension()],
                pressure=right,
                **parameters,
            )
        if self.vector and left == "total_force" and self.flux_name == "traction":
            components = variables[: self._dimension()]
            values = _list_items(right, len(components), where)
            for variable, value in zip(components, values):
                if value not in ("0", "0.0"):
                    problem.add_boundary_condition(
                        "traction", boundary, variable=variable, total_force=value, **parameters
                    )
            return None
        if self.flux_name and left == self.flux_name:
            if not self.vector:
                # Neumann takes the flux entering the body, -n . q.
                return problem.add_boundary_condition(
                    "Neumann", boundary, variable=variables[0], flux=f"-({right})", **parameters
                )
            components = variables[: self._dimension()]
            values = _list_items(right, len(components), where)
            for variable, value in zip(components, values):
                if value not in ("0", "0.0"):
                    problem.add_boundary_condition(
                        "traction", boundary, variable=variable, traction=value, **parameters
                    )
            return None
        names = [self.flux_name] if self.flux_name else []
        if self.flux_name == "traction":
            names += ["total_force", "pressure"]
        accepted = ", ".join(variables) + (f" or {', '.join(names)}" if names else "")
        raise InputError(
            f"{where}: the left side is a variable of this physics (its value on the boundary) or the name of its normal flux: {accepted}."
        )

    def add_kernel(self, kernel: str, **parameters):
        """Add a term of the object reference (or a word of this physics,
        such as ``"heat_source"`` of ``heat_transfer``) to the equations of
        this physics.  The variable is filled in for a physics with one
        variable, and must be given for one with several::

            heat.add_kernel("heat_source", heat_source=2.0e8, block=["wall"])
        """
        if kernel in self.kernel_words:
            kernel, renames = self.kernel_words[kernel]
            parameters = {renames.get(k, k): v for k, v in parameters.items()}
        variables = self.variable_names()
        if "variable" not in parameters:
            if len(variables) != 1:
                raise InputError(
                    f"{label(self.type_name, self.name)}: give 'variable' (one of {', '.join(variables)}) for the kernel '{kernel}'."
                )
            parameters["variable"] = variables[0]
        return self._problem.add_kernel(kernel, **parameters)

    def _dimension(self) -> int:
        return len(self.variable_names())

    def _restriction(self) -> dict:
        block = list(getattr(self, "block", ()))
        return {"block": block} if block else {}

    def _build(self, problem) -> None:
        raise NotImplementedError

    # ---- manufactured solutions (dualmesh.mms) -------------------------
    def _resolve(self, coordinates: str, dimension: int) -> None:
        """Settle the defaults that depend on the problem, without a
        problem, for the symbolic form of the equations."""

    def _manufactured_terms(self, fields: dict, coordinates: str, dimension: int) -> dict:
        """``{variable: (flux, source)}`` of the canonical form
        :math:`-\\nabla \\cdot \\mathbf{F} + S = 0` of the equations this physics
        generates, in SymPy, for the exact ``fields``.  The flux is a list of
        three expressions."""
        raise InputError(
            f"{label(self.type_name, self.name)}: manufactured solutions are not available for this physics."
        )

    def _set_manufactured_source(self, forcing: dict) -> set:
        """Take the manufactured sources ``{variable: text}`` into the
        physics' own source terms where the physics needs them there (e.g.,
        for a stabilisation that uses the body force), and return the
        variables so served.  The sources of the other variables are added
        as body_force kernels."""
        return set()


def _names(boundary) -> list[str]:
    """One boundary name or a list of them, as a list."""
    return [boundary] if isinstance(boundary, str) else list(boundary)


def _sympy_expression(value):
    """A number or a text in x, y, z, t (and variable names), as SymPy."""
    from .mms import _expr

    return _expr(value)


def _gradient(u):
    from .mms import _gradient as gradient

    return gradient(u)


class Coupling:
    """Base of the couplings between two physics."""

    category: ClassVar[str] = "coupling"
    module: ClassVar[str] = "framework"
    type_name: ClassVar[str] = ""
    name: str = ""

    def _physics(self, problem, key: str, cls):
        """The physics of the parameter ``key``: the physics object given, or,
        when none is given, the only physics of the type ``cls`` of the problem."""
        physics = getattr(self, key)
        where = f"{label(self.type_name, self.name)}"
        if physics is None:
            candidates = [p for p in problem._physics.values() if isinstance(p, cls)]
            if len(candidates) != 1:
                raise InputError(
                    f"{where}: the problem has {len(candidates)} {cls.type_name} physics. Give the one to couple as {key}=, the object that add_physics returned."
                )
            physics = candidates[0]
        elif not isinstance(physics, Physics):
            raise InputError(
                f"{where}: give {key} as the object that add_physics returned, not {physics!r}."
            )
        elif not isinstance(physics, cls):
            raise InputError(
                f"{where}: {key} is a {physics.type_name} physics, and needs a {cls.type_name} physics."
            )
        elif all(p is not physics for p in problem._physics.values()):
            raise InputError(f"{where}: the {key} physics belongs to another problem.")
        setattr(self, key, physics)
        return physics

    def _apply(self, problem) -> None:
        """Change the physics before they are built."""

    def _build(self, problem) -> None:
        """Add the objects of the coupling after the physics are built."""


# ---------------------------------------------------------------------------
# Heat transfer
# ---------------------------------------------------------------------------
@register
@dataclass
class HeatTransfer(Physics):
    """Heat conduction in a solid or a fluid at rest, with an optional heat
    source and heat capacity: ρ c_p ∂T/∂t − ∇·(k ∇T) = q. The
    conductivity is a constant, the name of a function, or, when it is not
    given, the property thermal_conductivity of the property objects of
    the problem."""

    type_name: ClassVar[str] = "heat_transfer"
    module: ClassVar[str] = "heat_transfer"
    #: heat_flux is Neumann with the heat flux entering the body, and convective_heat_flux is
    #: Robin with Newton's law of cooling.
    #: heat_source is body_force with the volumetric heat generation.
    kernel_words: ClassVar[dict] = {"heat_source": ("body_force", {"heat_source": "value"})}
    flux_name: ClassVar[str] = "heat_flux"

    #: The temperature variable.
    temperature: ClassVar[str] = "temperature"
    thermal_conductivity: float | str | None = parameter(
        None,
        unit="W/(m K)",
        description="Thermal conductivity k, a constant or the name of a function of (x, y, z, t). Default: the property thermal_conductivity, which a property object such as constant_property or parsed_property declares.",
    )
    heat_source: float | str | None = parameter(
        None,
        unit="W/m^3",
        description="Volumetric heat generation q, a constant or the name of a function. Default none.",
    )
    density: float | str | None = parameter(
        None,
        unit="kg/m^3",
        description="Mass density rho of the heat capacity term. Default none: the heat capacity term is included only when both density and specific_heat are given, which a transient solve needs.",
    )
    specific_heat: float | str | None = parameter(
        None,
        unit="J/(kg K)",
        description="Specific heat c_p of the heat capacity term and, with a nonisothermal_flow coupling, of the heat that the flow carries. Default none (see density), and 1 with the coupling, the value of the non-dimensional equations.",
    )
    advection_scheme: str | None = parameter(
        None,
        description="How the heat that a flow carries (with a nonisothermal_flow coupling) is discretised: 'SUPG' (streamline upwind Petrov-Galerkin), the face fluxes of DMCDM 'limited' (the upstream face value blended towards the central one by the van Albada limiter, bounded and second order), 'linear_upwind' (second order, not bounded) and 'upwind' (bounded, first order), or 'central' (no stabilisation, which oscillates where convection dominates). Default SUPG, which measured the closest on boundary layers and jumps. Where conduction matters as much as convection (resolved natural convection), central can be closer.",
    )
    initial_condition: float | str | None = parameter(
        None,
        unit="K",
        description="Initial temperature, a constant or the name of a function. Default 0.",
    )
    block: Sequence[str] = parameter(
        (), description="Blocks where the equation holds. Default: every block."
    )

    def __post_init__(self):
        if (self.density is None) != (self.specific_heat is None):
            raise InputError(
                "heat_transfer: give both density and specific_heat for the heat capacity term, or neither."
            )
        # An abbreviation keeps its capitals (SUPG); the lower-case spelling is accepted.
        if isinstance(self.advection_scheme, str) and self.advection_scheme.lower() == "supg":
            self.advection_scheme = "SUPG"

    def variable_names(self) -> list[str]:
        return [self.temperature]

    def _variable_units(self) -> dict:
        return {self.temperature: "K"}

    def _initial_condition(self, variable: str):
        return self.initial_condition

    def _build(self, problem) -> None:
        T, restrict = self.temperature, self._restriction()
        if self.thermal_conductivity is None:
            problem.add_kernel(
                "heat_conduction",
                variable=T,
                thermal_conductivity="thermal_conductivity",
                **restrict,
            )
        else:
            problem.add_kernel(
                "heat_conduction",
                variable=T,
                thermal_conductivity=self.thermal_conductivity,
                **restrict,
            )
        if self.heat_source is not None:
            problem.add_kernel(
                "body_force",
                variable=T,
                value=self.heat_source,
                **restrict,
            )
        if self.density is not None:
            problem.add_kernel(
                "heat_conduction_time_derivative",
                variable=T,
                density=self.density,
                specific_heat=self.specific_heat,
                **restrict,
            )

    def _manufactured_terms(self, fields, coordinates, dimension):
        from .mms import _symbols

        if self.thermal_conductivity is None:
            raise InputError(
                f"{label(self.type_name, self.name)}: a manufactured solution needs the thermal_conductivity as a constant or an expression of x, y, z and t."
            )
        T = fields[self.temperature]
        k = _sympy_expression(self.thermal_conductivity)
        source = 0
        if self.heat_source is not None:
            source -= _sympy_expression(self.heat_source)
        if self.density is not None:
            t = _symbols()[3]
            source += (
                _sympy_expression(self.density) * _sympy_expression(self.specific_heat) * T.diff(t)
            )
        return {self.temperature: ([k * g for g in _gradient(T)], source)}


# ---------------------------------------------------------------------------
# General form PDE
# ---------------------------------------------------------------------------
_GRADIENT = r"grad_([xyz])\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)"


def _general_form_expression(text, fields: dict, constants: dict):
    """A flux or source expression of general_form_PDE as SymPy, with every
    field replaced by its exact solution and every grad_x(u), grad_y(u) and
    grad_z(u) by the derivative of that solution."""

    x, y, z, _t = _symbols_of_mms()
    axes = {"x": x, "y": y, "z": z}
    placeholders = {}

    def replace(match):
        name = f"_gradient_{match.group(1)}_{match.group(2)}"
        placeholders[name] = (match.group(1), match.group(2))
        return name

    expression = _sympy_expression(re.sub(_GRADIENT, replace, str(text)))
    substitutions = {}
    for name, (axis, field) in placeholders.items():
        if field not in fields:
            raise InputError(
                f"general_form_PDE: grad_{axis}({field}) names '{field}', which is not a field."
            )
        substitutions[_sympy_expression(name)] = fields[field].diff(axes[axis])
    for field, exact in fields.items():
        substitutions[_sympy_expression(field)] = exact
    for constant, value in constants.items():
        substitutions[_sympy_expression(constant)] = value
    return expression.subs(substitutions)


def _symbols_of_mms():
    from .mms import _symbols

    return _symbols()


@register
@dataclass
class GeneralFormPDE(Physics):
    """Partial differential equations written as they are printed, or by
    their coefficients.

    ``equations`` holds the equations as text, for example
    ``"rho*cp*dT/dt = div(k*grad(T)) + q"``; the page *From paper to code*
    of the user guide defines the language.  The coefficients ``m``, ``d``,
    ``c``, ``alpha``, ``gamma``, ``beta``, ``a`` and ``f`` give instead, for
    a scalar field u, the equation

        m d2u/dt2 + d du/dt + div(-c grad u - alpha u + gamma) + beta . grad u + a u = f.

    Any other keyword is a definition that the equations use by its name: a
    number is a named constant and a text a named expression, which may use
    the definitions before it.  The equations are reduced symbolically to
    their fluxes and sources, compiled once and differentiated
    automatically, so Newton's method has the exact Jacobian."""

    type_name: ClassVar[str] = "general_form_PDE"
    module: ClassVar[str] = "framework"
    #: Keywords that are not parameters are definitions.
    takes_definitions: ClassVar[bool] = True

    variables: str | Sequence[str | Mapping[str, str]] = parameter(
        description="The fields, one equation each: a name, or a list whose items are names or {name: kind}, with the kind 'scalar' (the default) or 'vector', e.g. [{'U': 'vector'}, 'p']. A vector field U has the components U_x, U_y, U_z (U_r, U_z in axisymmetric coordinates).",
    )
    equations: str | Sequence[str] | Mapping[str, str] = parameter(
        (),
        description="The equations as text, one per field: one text, a list, which is matched to the fields automatically (by the time derivative, then by the second derivative, then the field left over), or a dict field -> equation. Default: the equations of the coefficients.",
    )
    m: object = parameter(None, description="Mass coefficient m of d2u/dt2. Default none.")
    d: object = parameter(
        None, description="Damping (or mass) coefficient d of du/dt. Default none."
    )
    c: object = parameter(
        None,
        description="Diffusion coefficient c in div(-c grad u): an expression, a list of one per dimension (orthotropic) or of the dimension squared, row by row (a tensor), or with several fields a dict field -> value, whose value may itself be a dict field -> value that couples the equation to the gradients of other fields. Default none.",
    )
    alpha: object = parameter(
        None,
        description="Conservative flux convection coefficient alpha in div(-alpha u), one value per dimension. Default none.",
    )
    gamma: object = parameter(
        None,
        description="Conservative flux source gamma in div(gamma), one value per dimension. Default none.",
    )
    beta: object = parameter(
        None,
        description="Convection coefficient beta of beta . grad u, one value per dimension. Default none.",
    )
    a: object = parameter(None, description="Absorption coefficient a of a u. Default none.")
    f: object = parameter(None, description="Source term f. Default none.")
    units: str | Mapping[str, str] = parameter(
        default_factory=dict,
        description="The SI unit of each field, which the output files write with its values: one unit, or a dict field -> unit. Default: dimensionless fields.",
    )
    initial_condition: object = parameter(
        default_factory=dict,
        description="The initial value of each field: a constant, the name of a function or an expression of x, y and z, or a dict field -> such a value. Default 0.",
    )
    block: Sequence[str] = parameter(
        (), description="Blocks where the equations hold. Default: every block."
    )
    _definitions: Mapping[str, object] = parameter(default_factory=dict)

    _COEFFICIENTS: ClassVar[tuple] = ("m", "d", "c", "alpha", "gamma", "beta", "a", "f")

    def __post_init__(self):
        where = "general_form_PDE"
        variables = _field_kinds(self.variables, where)
        for field, kind in variables.items():
            if kind not in ("scalar", "vector"):
                raise InputError(
                    f"{where}: the field '{field}' is '{kind}'; a field is 'scalar' or 'vector'."
                )
            if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", field):
                raise InputError(
                    f"{where}: '{field}' is not a name for a field: use letters, digits and underscores, starting with a letter."
                )
        if not variables:
            raise InputError(f"{where}: give the fields in 'variables'.")
        self._kinds = dict(variables)
        fields = list(variables)
        equations = self.equations
        if isinstance(equations, str):
            equations = [equations]
        if isinstance(equations, Mapping):
            self._keys, self._texts = list(equations), [str(e) for e in equations.values()]
        else:
            self._texts = [str(e) for e in equations]
            self._keys = [""] * len(self._texts)
        for key in self._keys:
            if key and key not in fields:
                raise InputError(
                    f"{where}: the equation of '{key}' names a field that is not in 'variables' ({', '.join(fields)})."
                )
        # The equations of the coefficients, as text.
        given = {k: getattr(self, k) for k in self._COEFFICIENTS if getattr(self, k) is not None}
        if given:
            scalar_fields = [v for v in fields if self._kinds[v] == "scalar"]
            for key, value in given.items():
                if not isinstance(value, Mapping) or (
                    len(scalar_fields) == 1
                    and key in ("c", "alpha", "beta", "a")
                    and scalar_fields[0] not in value
                ):
                    if len(scalar_fields) != 1:
                        raise InputError(
                            f"{where}: with several fields, give the coefficient {key} as a dict field -> value."
                        )
                    given[key] = {scalar_fields[0]: value}
                unknown = sorted(set(given[key]) - set(scalar_fields))
                if unknown:
                    raise InputError(
                        f"{where}: the coefficient {key} names {', '.join(unknown)}, which is not a scalar field of 'variables'; the coefficients are for scalar fields."
                    )
            for u in scalar_fields:
                text = _coefficient_equation(u, {k: v.get(u) for k, v in given.items()})
                if text is None:
                    continue
                if u in self._keys:
                    raise InputError(
                        f"{where}: the field '{u}' has an equation and coefficients; give one of them."
                    )
                self._texts.append(text)
                self._keys.append(u)
        if len(self._texts) != len(fields):
            raise InputError(
                f"{where}: there are {len(fields)} field(s) ({', '.join(fields)}) and {len(self._texts)} equation(s); give one equation for each field, in 'equations' or by its coefficients."
            )
        # The definitions: numbers are constants, texts are expressions.
        self._constants, self._expressions = {}, []
        # The coordinates are checked when the problem is known (r is a coordinate only in
        # axisymmetric coordinates).
        reserved = set(fields) | {"t", "n", "I", "pi", "e", "eigenvalue"}
        for name, value in self._definitions.items():
            if name in reserved:
                raise InputError(
                    f"{where}: '{name}' is a name of the equation language or a field, so it cannot be defined. Rename the definition."
                )
            if isinstance(value, bool) or not isinstance(value, (int, float, str)):
                raise InputError(
                    f"{where}: the definition {name} = {value!r} is not a number or an expression (a text)."
                )
            if isinstance(value, str):
                self._expressions.append((name, value))
            else:
                self._constants[name] = float(value)
        units = self.units
        self.units = {fields[0]: units} if isinstance(units, str) else dict(units)
        initial = self.initial_condition
        if not isinstance(initial, Mapping):
            if len(fields) != 1:
                raise InputError(
                    f"{where}: with several fields, give initial_condition as a dict field -> value."
                )
            initial = {fields[0]: initial}
        self.initial_condition = dict(initial)
        for key in ("units", "initial_condition"):
            unknown = sorted(set(getattr(self, key)) - set(self._components(3)) - set(fields))
            if unknown:
                raise InputError(
                    f"{where}: the {key} names {', '.join(unknown)}, which is not a field of 'variables' ({', '.join(fields)})."
                )

    def _components(self, dimension: int) -> list[str]:
        names = []
        for field, kind in self._kinds.items():
            if kind == "scalar":
                names.append(field)
            else:
                axes = ("r", "z") if self._axisymmetric() else ("x", "y", "z")
                names += [f"{field}_{a}" for a in axes[:dimension]]
        return names

    def _axisymmetric(self) -> bool:
        return getattr(self, "_geometry", (3, False))[1]

    def _resolve(self, coordinates: str, dimension: int) -> None:
        self._geometry = (dimension, coordinates == "axisymmetric")
        axes = ["r", "z"] if self._geometry[1] else ["x", "y", "z"][:dimension]
        for name in self._definitions:
            if name in axes:
                raise InputError(
                    f"general_form_PDE: '{name}' is a coordinate of this problem, so it cannot be defined. Rename the definition."
                )
        self._reduced = self._reduce(*self._geometry)

    def _attach(self, problem) -> None:
        self._problem = problem
        self._resolve(problem.coordinates, problem.mesh.space_dimension)
        super()._attach(problem)

    def variable_names(self) -> list[str]:
        return self._components(self._geometry[0])

    def _scope(self, dimension: int, axisymmetric: bool) -> tuple:
        fields = [(f, 0 if k == "scalar" else 1) for f, k in self._kinds.items()]
        return (dimension, axisymmetric, fields, list(self._constants), self._expressions)

    def _reduce(self, dimension: int, axisymmetric: bool) -> list[dict]:
        from . import _core

        try:
            return _core.reduce_equations(
                *self._scope(dimension, axisymmetric), self._texts, self._keys
            )
        except _core.InputError as error:
            raise InputError(f"general_form_PDE: {error}") from None

    def _variable_units(self) -> dict:
        units = {}
        for field, unit in self.units.items():
            for name in self.variable_names():
                if name == field or name.startswith(field + "_"):
                    units[name] = unit
        return units

    def _initial_condition(self, variable: str):
        if variable in self.initial_condition:
            return self.initial_condition[variable]
        field = variable.rsplit("_", 1)[0]
        return self.initial_condition.get(field) if self._kinds.get(field) == "vector" else None

    def _inline_constants(self, text: str) -> str:
        """``text`` with the named constants replaced by their values, for
        the objects that take no constants."""
        return re.sub(
            r"(?<![A-Za-z0-9_.])[A-Za-z_][A-Za-z0-9_]*",
            lambda m: (
                f"({self._constants[m.group(0)]!r})"
                if m.group(0) in self._constants
                else m.group(0)
            ),
            text,
        )

    def _used_names(self) -> set:
        texts = list(self._texts) + [t for _, t in self._expressions]
        texts += getattr(self, "_condition_texts", [])
        return set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", " ".join(texts)))

    def add_boundary_condition(self, condition, boundary=None, **parameters):
        """Add a boundary condition written as text, on the side sets or
        node sets named by ``boundary``: ``"u = 0"`` prescribes the value of
        a field, and a condition with the normal flux on the left,
        ``"-k*du/dn = h*(u - 1)"`` or ``"normal_flux = 5"``, prescribes
        n . Gamma, the flux of the equation through the boundary along the
        outward normal n.  A list of texts adds several conditions.

            pde.add_boundary_condition("u = 0", boundary=["left", "right"])
            pde.add_boundary_condition("-k*du/dn = h*(u - 1)", boundary="top")
        """
        if isinstance(condition, (list, tuple)) or _is_condition_text(condition):
            where = f"general_form_PDE, boundary condition {condition!r}"
            texts = _expand_conditions(condition, parameters, where)
            if len(texts) > 1 or isinstance(condition, (list, tuple)):
                for text in texts:
                    self.add_boundary_condition(text, boundary, **parameters)
                return None
            condition = texts[0]
        if not _is_condition_text(condition):
            return super().add_boundary_condition(condition, boundary, **parameters)
        from . import _core

        if boundary is None:
            mesh = self._problem.mesh
            names = sorted(set(mesh.sideset_names()) | set(mesh.nodeset_names()))
            listed = f", one or more of: {', '.join(names)}" if names else ""
            raise InputError(
                f"general_form_PDE, boundary condition '{condition}': give boundary{listed}."
            )
        problem = self._problem
        try:
            reduced = _core.reduce_condition(
                *self._scope(problem.mesh.space_dimension, problem.coordinates == "axisymmetric"),
                self._texts,
                self._keys,
                condition,
                ["normal_flux"],
            )
        except _core.InputError as error:
            raise InputError(f"general_form_PDE: {error}") from None
        self._condition_texts = getattr(self, "_condition_texts", []) + [condition]
        for item in reduced:
            value = self._inline_constants(item["value"])
            if item["dirichlet"]:
                problem.add_boundary_condition(
                    "Dirichlet", boundary, variable=item["field"], value=value, **parameters
                )
            else:
                # Neumann takes the flux entering the body, -n . Gamma.
                problem.add_boundary_condition(
                    "Neumann", boundary, variable=item["field"], flux=f"-({value})", **parameters
                )
        return None

    def _build(self, problem) -> None:
        unused = [name for name in self._definitions if name not in self._used_names()]
        if unused:
            import difflib

            close = difflib.get_close_matches(
                unused[0],
                [f.name for f in dataclasses.fields(self) if not f.name.startswith("_")],
                n=1,
            )
            hint = f" Did you mean the parameter '{close[0]}'?" if close else ""
            raise InputError(
                f"general_form_PDE: '{unused[0]}' is defined and not used by an equation, a definition or a boundary condition.{hint}"
            )
        restrict = self._restriction()
        constants = {"constants": dict(self._constants)} if self._constants else {}
        for q in self._reduced:
            if q["second_time_coeff"]:
                raise InputError(
                    f"general_form_PDE: the equation '{q['equation']}' has a second time derivative, which needs the generalized-alpha time integrator; it is not available yet."
                )
            if q["flux"] or q["source"]:
                problem.add_kernel(
                    "parsed_kernel",
                    variable=q["field"],
                    flux=list(q["flux"]),
                    source=q["source"],
                    **constants,
                    **restrict,
                )
            if q["time_coeff"]:
                problem.add_kernel(
                    "time_derivative",
                    variable=q["field"],
                    coeff=self._inline_constants(q["time_coeff"]),
                    **restrict,
                )

    def _manufactured_terms(self, fields, coordinates, dimension):
        _x, _y, _z, t = _symbols_of_mms()
        terms = {}
        for q in self._reduce(dimension, coordinates == "axisymmetric"):
            flux = [-_general_form_expression(c, fields, self._constants) for c in q["flux"]]
            flux += [0] * (3 - len(flux))
            source = 0
            if q["source"]:
                source -= _general_form_expression(q["source"], fields, self._constants)
            if q["time_coeff"]:
                d = _general_form_expression(q["time_coeff"], fields, self._constants)
                source += d * fields[q["field"]].diff(t)
            terms[q["field"]] = (flux, source)
        return terms


def _coefficient_equation(u: str, coefficients: dict) -> str | None:
    """The equation of the scalar field ``u`` of its coefficients, as text:
    m d2u/dt2 + d du/dt + div(-c grad u - alpha u + gamma) + beta . grad u + a u = f."""

    def text(value):
        if isinstance(value, (list, tuple)):
            return "[" + ", ".join(text(v) for v in value) + "]"
        return f"({value})"

    def coupled(value):
        # A dict couples the equation to other fields; a value acts on u.
        return list(value.items()) if isinstance(value, Mapping) else [(u, value)]

    left, flux = [], []
    if coefficients.get("m") is not None:
        left.append(f"{text(coefficients['m'])}*d2{u}/dt2")
    if coefficients.get("d") is not None:
        left.append(f"{text(coefficients['d'])}*d{u}/dt")
    for w, c in coupled(coefficients["c"]) if coefficients.get("c") is not None else []:
        if isinstance(c, (list, tuple)):
            size = round(len(c) ** 0.5)
            if size * size == len(c) and len(c) > 3:
                rows = [c[i * size : (i + 1) * size] for i in range(size)]
                flux.append(f"- dot({text(rows)}, grad({w}))")
                continue
            tensor = [[c[i] if i == j else 0 for j in range(len(c))] for i in range(len(c))]
            flux.append(f"- dot({text(tensor)}, grad({w}))")
        else:
            flux.append(f"- {text(c)}*grad({w})")
    for w, alpha in coupled(coefficients["alpha"]) if coefficients.get("alpha") is not None else []:
        flux.append(f"- {text(alpha)}*{w}")
    if coefficients.get("gamma") is not None:
        flux.append(f"+ {text(coefficients['gamma'])}")
    if flux:
        left.append("div(" + " ".join(flux).removeprefix("+ ") + ")")
    for w, beta in coupled(coefficients["beta"]) if coefficients.get("beta") is not None else []:
        left.append(f"dot({text(beta)}, grad({w}))")
    for w, a in coupled(coefficients["a"]) if coefficients.get("a") is not None else []:
        left.append(f"{text(a)}*{w}")
    right = text(coefficients["f"]) if coefficients.get("f") is not None else "0"
    if not left:
        if right == "0":
            return None
        left = ["0"]
    return " + ".join(left) + " = " + right


# ---------------------------------------------------------------------------
# Solid mechanics
# ---------------------------------------------------------------------------
FORMULATIONS = (
    "plane_stress",
    "plane_strain",
    "axisymmetric",
    "three_dimensional",
    "plate",
    "beam",
    "circular_plate",
)
#: The formulations of the structural theories, which reduce the body to a surface or a line.
STRUCTURAL = ("plate", "beam", "circular_plate")
#: The theories of each structural formulation; the first is the default.
THEORIES = {
    "plate": ("first_order",),
    "beam": ("Euler-Bernoulli", "Timoshenko"),
    "circular_plate": ("first_order", "classical"),
}


@register
@dataclass
class SolidMechanics(Physics):
    """Linear elasticity of an isotropic solid at small strain: ∇·σ + f
    = 0. The formulation is three_dimensional on a three-dimensional mesh and
    axisymmetric in axisymmetric coordinates, and is given on a
    two-dimensional Cartesian mesh (plane_stress or plane_strain).

    The structural formulations reduce a thin body to its mid-surface or its
    axis: plate (the first-order, Mindlin, theory on a two-dimensional mesh of
    the mid-plane; variables disp_x, disp_y, disp_z and phi_x, phi_y), beam
    (Euler-Bernoulli or Timoshenko on a one-dimensional mesh along x;
    disp_x, disp_z and bending_moment, or phi in the displacement form) and
    circular_plate (first_order or classical, on a radial mesh in
    axisymmetric coordinates; disp_r, disp_z and phi or bending_moment).  A
    section is given by its material, youngs_modulus, poissons_ratio and the
    thickness (plates) or the cross_section (beams), or by its section
    stiffnesses, as for a laminated or functionally graded section."""

    type_name: ClassVar[str] = "solid_mechanics"
    flux_name: ClassVar[str] = "traction"
    module: ClassVar[str] = "solid_mechanics"
    vector: ClassVar[bool] = True

    youngs_modulus: float | str | None = parameter(
        None,
        unit="Pa",
        description="Young's modulus E: a constant, a function or an expression of the fields and the coordinates (in a structural formulation z is the position through the thickness, so E(z) gives a graded section). Default: the property youngs_modulus, which a property object such as constant or parsed declares.",
    )
    poissons_ratio: float | None = parameter(
        None,
        description="Poisson's ratio nu, with -1 < nu < 0.5. Default: the property poissons_ratio when a property object declares it, and otherwise 1/3, the default of the library.",
    )
    #: The displacement field: the variables disp_x, disp_y, disp_z (disp_r, disp_z in axisymmetric coordinates).
    displacement: ClassVar[str] = "disp"
    formulation: str | None = parameter(
        None,
        description="plane_stress, plane_strain, axisymmetric or three_dimensional. Default: three_dimensional on a three-dimensional mesh and axisymmetric in axisymmetric coordinates. A two-dimensional Cartesian problem must give plane_stress or plane_strain, because neither is right for every body.",
    )
    thickness: float | None = parameter(
        None,
        unit="m",
        description="The thickness of a plate or a circular plate given by its material, where it is required, or the out-of-plane thickness of a plane problem, which multiplies its equations and its traction and pressure conditions. A three-dimensional, axisymmetric or beam problem refuses it. Default: none for a plate, and 1 m (the equations per unit thickness) for a plane problem.",
    )
    body_force: Sequence[float] | None = parameter(
        None,
        unit="N/m^3",
        description="Constant body force per unit volume, one value per component (e.g., the weight [0, 0, -rho g]). Default none.",
    )
    theory: str | None = parameter(
        None,
        description="The theory of a structural formulation: Euler-Bernoulli (default) or Timoshenko for a beam, first_order (default) or classical for a circular plate, first_order for a plate.",
    )
    form: str | None = parameter(
        None,
        description="The form of the Timoshenko beam: mixed (the default, with the bending moment as a variable, free of shear locking) or displacement (the rotation phi as a variable, with the shear term integrated at the element centre). The Euler-Bernoulli beam and the classical circular plate are mixed, the first-order plates displacement.",
    )
    cross_section: Mapping[str, float] | None = parameter(
        None,
        description="The cross-section of a beam given by its material: {'width': b, 'height': h}, a rectangle, through whose height the material law is integrated, or {'area': A, 'second_moment': I}, a homogeneous section, and 'shear_correction' (default 5/6, a rectangle). Default none.",
    )
    extensional_stiffness: float | None = parameter(
        None,
        unit="N",
        description="Section stiffness A of a structural formulation (per unit width for plates, with the factor 1/(1 - nu^2)), for a section that the material and geometry do not describe, e.g. laminated or functionally graded. Default: from the material.",
    )
    coupling_stiffness: float | None = parameter(
        None,
        unit="N m",
        description="Section stiffness B, the coupling of extension and bending, zero for a section symmetric about its mid-surface. Default: 0, or from the material.",
    )
    bending_stiffness: float | None = parameter(
        None, unit="N m^2", description="Section stiffness D. Default: from the material."
    )
    shear_stiffness: float | None = parameter(
        None,
        unit="N",
        description="Section stiffness S of the transverse shear (K G A). Default: from the material.",
    )
    transverse_load: float | str = parameter(
        0.0,
        unit="N/m^2 (plate) or N/m (beam)",
        description="Distributed load along the deflection disp_z of a structural formulation: per unit area of a plate (N/m^2) and per unit length of a beam (N/m), a constant or an expression. Default 0.",
    )
    axial_load: float | str = parameter(
        0.0,
        unit="N/m",
        description="Distributed axial (beam) or radial (circular plate) load. Default 0.",
    )
    in_plane_load: Sequence[float | str] | None = parameter(
        None,
        unit="N/m^2",
        description="Distributed in-plane load [f_x, f_y] of a plate. Default none.",
    )
    foundation_modulus: float = parameter(
        0.0,
        unit="N/m^3 (plate) or N/m^2 (beam)",
        description="Modulus of an elastic (Winkler) foundation under a structural formulation: the pressure per unit deflection under a plate (N/m^3) and the force per unit length and unit deflection under a beam (N/m^2). Default 0.",
    )
    von_karman: bool = parameter(
        False,
        description="The von Karman nonlinearity of moderately large deflections, for a structural formulation. Default false.",
    )
    block: Sequence[str] = parameter(
        (), description="Blocks where the equations hold. Default: every block."
    )

    @property
    def _structural(self) -> bool:
        return self.formulation in STRUCTURAL

    def _section(self) -> dict:
        """The parameters of the section for the structural kernels: the
        material and the geometry, whose law the kernels integrate through
        the thickness, or the section stiffnesses."""
        where = label(self.type_name, self.name)
        given = {
            k: getattr(self, k)
            for k in (
                "extensional_stiffness",
                "coupling_stiffness",
                "bending_stiffness",
                "shear_stiffness",
            )
            if getattr(self, k) is not None
        }
        if given:
            if self.cross_section is not None or self.youngs_modulus is not None:
                raise InputError(
                    f"{where}: give the section by its material (youngs_modulus, poissons_ratio and the thickness or cross_section) or by its section stiffnesses, not both."
                )
            if self.thickness is not None:
                raise InputError(
                    f"{where}: the section stiffnesses describe the section, so the thickness is not given with them."
                )
            needed = ["extensional_stiffness", "bending_stiffness"]
            if self.theory in ("Timoshenko", "first_order"):
                needed.append("shear_stiffness")
            missing = [k for k in needed if k not in given]
            if missing:
                raise InputError(
                    f"{where}: the {self.formulation} of the {self.theory} theory given by its section stiffnesses needs {' and '.join(missing)} as well (coupling_stiffness may be left out: it is 0 for a section symmetric about its mid-surface)."
                )
            unused = [k for k in given if k not in needed + ["coupling_stiffness"]]
            if unused:
                raise InputError(
                    f"{where}: the {self.theory} theory has no {' and '.join(unused)}."
                )
            stiffnesses = {k: float(v) for k, v in given.items()}
            if self.poissons_ratio is not None:
                stiffnesses["poissons_ratio"] = self.poissons_ratio
            return stiffnesses
        if self.youngs_modulus is None:
            raise InputError(
                f"{where}: give youngs_modulus (with the thickness or the cross_section), or the section stiffnesses."
            )
        out = dict(youngs_modulus=self.youngs_modulus)
        if self.poissons_ratio is not None:
            out["poissons_ratio"] = self.poissons_ratio
        if self.formulation == "beam":
            section = dict(self.cross_section or {})
            shapes = ({"width", "height"}, {"area", "second_moment"})
            keys = set(section) - {"shear_correction"}
            if keys not in shapes:
                raise InputError(
                    f"{where}: give cross_section={{'width': b, 'height': h}} (a rectangle, through whose height the material is integrated) or {{'area': A, 'second_moment': I}} (a homogeneous section), and 'shear_correction' (default 5/6)."
                )
            if (
                self.theory == "Timoshenko"
                and "area" in section
                and "shear_correction" not in section
            ):
                raise InputError(
                    f"{where}: a Timoshenko beam with a section given by its area and second_moment needs its 'shear_correction' (5/6 for a rectangle, 0.9 for a solid circle), because it depends on the shape of the section."
                )
            if "shear_correction" in section:
                out["shear_correction"] = float(section.pop("shear_correction"))
            out["cross_section"] = {k: float(v) for k, v in section.items()}
            return out
        if self.cross_section is not None:
            raise InputError(f"{where}: a plate takes its thickness, not a cross_section.")
        if self.thickness is None:
            raise InputError(
                f"{where}: give the thickness (m) of the {self.formulation}, which its material needs to make its section."
            )
        out["thickness"] = float(self.thickness)
        return out

    def __post_init__(self):
        if self.thickness is not None and not self.thickness > 0.0:
            raise InputError(
                f"solid_mechanics: the thickness must be positive, not {self.thickness}."
            )
        for key, value in (self.cross_section or {}).items():
            if not (isinstance(value, (int, float)) and value > 0.0):
                raise InputError(
                    f"solid_mechanics: cross_section['{key}'] must be a positive number, not {value!r}."
                )
        if not self.foundation_modulus >= 0.0:
            raise InputError("solid_mechanics: foundation_modulus must not be negative.")
        if self.poissons_ratio is not None and not -1.0 < self.poissons_ratio < 0.5:
            raise InputError(
                f"solid_mechanics: Poisson's ratio must lie between -1 and 0.5, and {self.poissons_ratio} does not."
            )
        if isinstance(self.youngs_modulus, (int, float)) and self.youngs_modulus <= 0.0:
            raise InputError("solid_mechanics: Young's modulus must be positive.")
        if self.formulation is not None and self.formulation not in FORMULATIONS:
            raise InputError(
                f"solid_mechanics: unknown formulation '{self.formulation}'. Use {', '.join(FORMULATIONS)}."
            )
        if self.formulation in STRUCTURAL:
            theories = THEORIES[self.formulation]
            self.theory = self.theory or theories[0]
            if self.theory not in theories:
                raise InputError(
                    f"solid_mechanics: the {self.formulation} takes the theory {' or '.join(theories)}, not '{self.theory}'."
                )
            if self.form is not None and not (
                self.formulation == "beam" and self.theory == "Timoshenko"
            ):
                raise InputError(
                    "solid_mechanics: only the Timoshenko beam has a choice of form (mixed or displacement)."
                )
            if self.form not in (None, "mixed", "displacement"):
                raise InputError(
                    f"solid_mechanics: the form is mixed or displacement, not '{self.form}'."
                )
            if self.body_force is not None:
                raise InputError(
                    f"solid_mechanics: the {self.formulation} takes its loads as transverse_load, axial_load or in_plane_load, not body_force."
                )
            if self.in_plane_load is not None and self.formulation != "plate":
                raise InputError("solid_mechanics: in_plane_load is the load of a plate.")
        else:
            structural = [
                k
                for k in (
                    "theory",
                    "form",
                    "cross_section",
                    "extensional_stiffness",
                    "coupling_stiffness",
                    "bending_stiffness",
                    "shear_stiffness",
                    "in_plane_load",
                )
                if getattr(self, k) is not None
            ] + [
                k
                for k in ("transverse_load", "axial_load", "foundation_modulus")
                if getattr(self, k) not in (0, 0.0)
            ]
            if self.von_karman:
                structural.append("von_karman")
            if structural:
                raise InputError(
                    f"solid_mechanics: {', '.join(structural)} belong to the structural formulations (plate, beam, circular_plate); give formulation= one of them."
                )

    def _attach(self, problem) -> None:
        if problem.mesh.space_dimension > problem.mesh.dimension:
            raise InputError(
                f"{label(self.type_name, self.name)}: the mesh is a surface (or a curve) in space, which makes a shell; shells are not available yet."
            )
        if self.formulation is None:
            if problem.coordinates == "axisymmetric":
                self.formulation = "axisymmetric"
            elif problem.mesh.dimension == 3:
                self.formulation = "three_dimensional"
            else:
                raise InputError(
                    f"{label(self.type_name, self.name)}: give formulation='plane_stress' (a thin plate loaded in its plane) or formulation='plane_strain' (a long body restrained along its axis) for a two-dimensional problem."
                )
        if (
            self.formulation in ("axisymmetric", "circular_plate")
            and problem.coordinates != "axisymmetric"
        ):
            raise InputError(
                f"{label(self.type_name, self.name)}: the {self.formulation} formulation needs dm.Problem(mesh, coordinates='axisymmetric')."
            )
        where = label(self.type_name, self.name)
        needed = {"plate": 2, "beam": 1, "circular_plate": 1}.get(self.formulation)
        if needed is not None and problem.mesh.dimension != needed:
            raise InputError(
                f"{where}: the {self.formulation} formulation needs a {needed}-dimensional mesh ({'of the mid-plane' if needed == 2 else 'along the axis or the radius'}), and this mesh has {problem.mesh.dimension} dimensions."
            )
        if self.thickness is not None and self.formulation in (
            "three_dimensional",
            "axisymmetric",
            "beam",
        ):
            what = "its cross_section" if self.formulation == "beam" else "no thickness"
            raise InputError(
                f"{where}: the {self.formulation} formulation takes {what}, so the thickness is not given."
            )
        if self.body_force is not None:
            components = 3 if self.formulation == "three_dimensional" else 2
            if len(self.body_force) != components:
                raise InputError(
                    f"{where}: body_force has {len(self.body_force)} values, and the {self.formulation} formulation needs {components}, one per component of the displacement."
                )
        if self.formulation in STRUCTURAL:
            # The one-point reduced integration of the shear unlocks the linear line and
            # quadrilateral elements only; a linear triangle still locks (0.04 of the
            # deflection of a thin plate on an 8 x 8 mesh), and triangles come with the shells.
            allowed = {"plate": {"Quad4"}}.get(self.formulation, {"Edge2"})
            mesh = problem.mesh
            other = sorted({mesh.element_type(e) for e in range(mesh.num_elements)} - allowed)
            if other:
                raise InputError(
                    f"{where}: the {self.formulation} formulation takes {', '.join(sorted(allowed))} elements, and this mesh has {', '.join(other)}. Its reduced integration of the shear is one point, which unlocks these elements only."
                )
        if self._structural:
            self._section()
        super()._attach(problem)

    def _plane_thickness(self) -> float:
        """The out-of-plane thickness of a plane problem: 1 m (per unit thickness) when not given."""
        return 1.0 if self.thickness is None else float(self.thickness)

    def _third(self) -> str:
        """The third variable of a beam or a circular plate."""
        mixed = self.theory in ("Euler-Bernoulli", "classical") or (
            self.theory == "Timoshenko" and self.form != "displacement"
        )
        return "bending_moment" if mixed else "phi"

    def variable_names(self) -> list[str]:
        if self.formulation == "plate":
            return ["disp_x", "disp_y", "disp_z", "phi_x", "phi_y"]
        if self.formulation == "beam":
            return ["disp_x", "disp_z", self._third()]
        if self.formulation == "circular_plate":
            return ["disp_r", "disp_z", self._third()]
        if self.formulation == "axisymmetric":
            return _components(self.displacement, 2, axisymmetric=True)
        return _components(self.displacement, 3 if self.formulation == "three_dimensional" else 2)

    def _dimension(self) -> int:
        return len([v for v in self.variable_names() if v.startswith("disp_")])

    def add_boundary_condition(self, condition, boundary=None, **parameters):
        # The free ends of a mixed beam or classical plate are derived from the end conditions when
        # the problem is first built; a condition added later would leave that derivation stale.
        if getattr(self, "_holds_free_ends", False):
            raise InputError(
                f"{label(self.type_name, self.name)}: the end conditions of a {self.formulation} are read when the problem is first solved, and this one comes after. Give every condition before the first solve."
            )
        return super().add_boundary_condition(condition, boundary=boundary, **parameters)

    def _symmetry(self, boundary, where: str):
        """A plane of symmetry.  A beam end of symmetry holds the axial displacement and the slope
        (phi in the displacement form, the natural condition of the moment equation in the mixed
        forms), and its moment stays free."""
        if not self._structural:
            return super()._symmetry(boundary, where)
        if self.formulation != "beam":
            raise InputError(
                f"{where}: a plane of symmetry of a {self.formulation} is written as text, e.g. 'disp_x = 0' and 'phi_x = 0'."
            )
        self._symmetric += _names(boundary)
        held = ["disp_x"] + (["phi"] if self._third() == "phi" else [])
        return self._problem.add_boundary_condition("Dirichlet", boundary, variable=held, value=0.0)

    def _fixed_variables(self) -> list[str]:
        # Clamped: every displacement and rotation; in the mixed forms the slope vanishes by
        # itself at a clamped end.
        return [v for v in self.variable_names() if v != "bending_moment"]

    @property
    def displacements(self) -> list[str]:
        """The displacement variables, one per component."""
        return self.variable_names()

    def _variable_units(self) -> dict:
        units = {"phi": "rad", "phi_x": "rad", "phi_y": "rad", "bending_moment": "N m"}
        return {v: units.get(v, "m") for v in self.variable_names()}

    def _build_structural(self, problem) -> None:
        restrict = self._restriction()
        # A thermal strain, uniform through the thickness, enters the section as its thermal force
        # and moment.
        expansion = self._coupled.get("thermal_expansion") or {}
        sections = dict(
            self._section(),
            **expansion,
            transverse_load=self.transverse_load,
            axial_load=self.axial_load,
            foundation_modulus=self.foundation_modulus,
            von_karman=self.von_karman,
        )
        variables = self.variable_names()
        if self.formulation == "plate":
            options = dict(
                sections,
                in_plane_displacements=variables[:2],
                transverse_displacement=variables[2],
                rotations=variables[3:],
                **({"in_plane_load": list(self.in_plane_load)} if self.in_plane_load else {}),
                **restrict,
            )
            self._section_kernel = problem.add_kernel(
                "plate_first_order", shear_treatment="exclude", **options
            )
            problem.add_kernel(
                "plate_first_order",
                shear_treatment="only",
                quadrature="midpoint",
                reduced_integration=True,
                **options,
            )
            return
        third = variables[2]
        if self.formulation == "beam":
            model = {
                ("Euler-Bernoulli", "bending_moment"): "beam_Euler-Bernoulli_mixed",
                ("Timoshenko", "bending_moment"): "beam_Timoshenko_mixed",
                ("Timoshenko", "phi"): "beam_Timoshenko_displacement",
            }[(self.theory, third)]
            mapping = dict(axial_displacement=variables[0], transverse_displacement=variables[1])
            mapping["rotation" if third == "phi" else "bending_moment"] = third
            for variable in variables:
                options = dict(sections, **mapping, variable=variable, **restrict)
                if third == "phi" and variable == third:
                    options.update(quadrature="midpoint", reduced_integration=True)
                self._section_kernel = problem.add_kernel(model, **options)
            if third == "bending_moment":
                self._hold_free_ends(problem, third)
            return
        mapping = dict(radial_displacement=variables[0], transverse_displacement=variables[1])
        options = dict(sections, **restrict)
        if self.theory == "classical":
            self._section_kernel = problem.add_kernel(
                "circular_plate_classical_mixed", bending_moment=third, **mapping, **options
            )
            self._hold_free_ends(problem, third)
            return
        self._section_kernel = problem.add_kernel(
            "circular_plate_first_order",
            rotation=third,
            shear_treatment="exclude",
            **mapping,
            **options,
        )
        problem.add_kernel(
            "circular_plate_first_order",
            rotation=third,
            shear_treatment="only",
            quadrature="midpoint",
            reduced_integration=True,
            **mapping,
            **options,
        )

    def _hold_free_ends(self, problem, moment: str) -> None:
        """Hold the bending moment at zero on every end of a mixed beam or
        classical circular plate that is not clamped, not a plane of symmetry
        and has no moment of its own.  The natural condition of the moment equation is a zero slope,
        so an end left without a condition would act as a guided end.  The
        axis of a circular plate keeps its natural condition, the zero slope
        of symmetry."""
        import numpy as np

        mesh = problem._global_mesh if problem._distributed else problem.mesh
        points = np.asarray(mesh.points())
        # An end is an end of the field: the blocks of every physics that shares the moment (a
        # stepped beam of two physics joins at a node that is no end).
        sharing = [p for p in problem._physics.values() if moment in p.variable_names()]
        self._holds_free_ends = True
        # The ends of the field are derived once, by the first physics that shares it, from the
        # end conditions of all of them.
        if sharing and sharing[0] is not self:
            return
        blocks = set()
        for physics in sharing:
            if not getattr(physics, "block", ()):
                blocks = set()
                break
            blocks |= {str(b) for b in physics.block}
        names = mesh.block_names()
        uses = np.zeros(mesh.num_nodes, dtype=int)
        for _type, nodes, block in mesh.cells():
            if not blocks or str(block) in blocks or names.get(block) in blocks:
                uses[list(nodes[:2])] += 1
        ends = set(np.flatnonzero(uses == 1).tolist())
        for physics in sharing or [self]:
            held = physics._clamped + physics._symmetric + physics._prescribed.get(moment, [])
            for boundary in held:
                ends -= set(mesh.boundary_nodes(boundary))
        size = float(np.ptp(points, axis=0).max())
        if self.formulation == "circular_plate":
            ends = {n for n in ends if abs(points[n, 0]) > 1e-12 * size}
        if not ends:
            return
        targets = points[sorted(ends)]
        name = self._object_name("free_ends")
        problem.mesh.add_nodeset_by_predicate(
            name,
            lambda x, y, z: bool(
                np.min(np.linalg.norm(targets - [x, y, z], axis=1)) <= 1e-9 * size
            ),
        )
        problem.add_boundary_condition("Dirichlet", name, variable=moment, value=0.0)

    def stress(self, z=None):
        """The stress at the element centres, one row per element.

        Parameters
        ----------
        z : float or {"top", "mid", "bottom"}, optional
            For a structural formulation, the height above the mid-surface
            (m) at which the stress is evaluated; the words are the faces and
            the mid-surface of a plate or a rectangular beam section.  A
            continuum takes no height.

        Returns
        -------
        numpy.ndarray
            A continuum: the six components of the stress property of its
            stress model (xx, yy, zz, yz, xz, xy).  A plate: sigma_xx,
            sigma_yy, sigma_xy.  A beam: sigma_xx.  A circular plate:
            sigma_rr, sigma_tt.  The structural stresses come from the
            material law of the section, with its thermal strain.
        """
        import numpy as np

        where = label(self.type_name, self.name)
        problem = self._problem
        if not self._structural:
            if z is not None:
                raise InputError(f"{where}: a continuum has no height z; give stress() without it.")
            return problem.property_at_centroids("stress")
        if z is None:
            raise InputError(
                f"{where}: give the height z of the {self.formulation} section: a number in m, or 'top', 'mid' or 'bottom'."
            )
        # The height of the section, when its material and shape give it.
        if self.formulation == "beam":
            height = (self.cross_section or {}).get("height")
        else:
            height = self.thickness
        if isinstance(z, str):
            if height is None:
                raise InputError(
                    f"{where}: the faces of this section are not known (it is given by its area or its stiffnesses); give z as a number in m."
                )
            heights = {"top": 0.5 * height, "mid": 0.0, "bottom": -0.5 * height}
            if z not in heights:
                raise InputError(f"{where}: z is a number, 'top', 'mid' or 'bottom', not '{z}'.")
            z = heights[z]
        elif height is not None and abs(float(z)) > 0.5 * float(height) * (1 + 1e-12):
            raise InputError(
                f"{where}: z = {z} m lies outside the section, whose faces are at -{0.5 * float(height)} m and {0.5 * float(height)} m."
            )
        problem.initialize()
        from .problem import _object_name

        rows = problem._problem.section_stress_at_centroids(
            _object_name(self._section_kernel), float(z)
        )
        # An element of a block the physics does not cover has no row: NaN there.
        width = max((len(r) for r in rows), default=0)
        return np.array([list(r) if len(r) == width else [np.nan] * width for r in rows])

    def _build(self, problem) -> None:
        if self._structural:
            self._build_structural(problem)
            return
        names, restrict = self.variable_names(), self._restriction()
        expansion = self._coupled.get("thermal_expansion")
        eigenstrains = []
        if expansion is not None:
            problem.add_property(
                "thermal_expansion_eigenstrain",
                eigenstrain_name=self._object_name("thermal_strain"),
                temperature=expansion["temperature"],
                linear_expansion_coeff=expansion["linear_expansion_coeff"],
                stress_free_temperature=expansion["stress_free_temperature"],
                **restrict,
            )
            eigenstrains.append(self._object_name("thermal_strain"))
        # A modulus that is not given is read from the property of the same name.
        moduli = {
            k: getattr(self, k)
            for k in ("youngs_modulus", "poissons_ratio")
            if getattr(self, k) is not None
        }
        problem.add_property(
            "small_strain_stress",
            displacements=names,
            formulation=self.formulation,
            eigenstrain_names=eigenstrains,
            **moduli,
            **restrict,
        )
        force = list(self.body_force) if self.body_force is not None else [0.0] * len(names)
        for component, variable in enumerate(names):
            problem.add_kernel(
                "stress_divergence",
                variable=variable,
                displacements=names,
                thickness=self._plane_thickness(),
                **restrict,
            )
            if force[component] != 0.0:
                problem.add_kernel(
                    "body_force",
                    variable=variable,
                    value=self._plane_thickness() * force[component],
                    **restrict,
                )

    def _resolve(self, coordinates, dimension):
        if self.formulation is None:
            if coordinates == "axisymmetric":
                self.formulation = "axisymmetric"
            elif dimension == 3:
                self.formulation = "three_dimensional"
            else:
                raise InputError(
                    f"{label(self.type_name, self.name)}: give formulation='plane_stress' or formulation='plane_strain' for a two-dimensional problem."
                )

    def _manufactured_terms(self, fields, coordinates, dimension):
        from .mms import _symbols

        if self._structural:
            raise InputError(
                f"{label(self.type_name, self.name)}: a manufactured solution of the {self.formulation} formulation is not available."
            )
        if self.youngs_modulus is None:
            raise InputError(
                f"{label(self.type_name, self.name)}: a manufactured solution needs youngs_modulus as a constant."
            )
        x = _symbols()[0]
        nu = 1.0 / 3.0 if self.poissons_ratio is None else float(self.poissons_ratio)
        E, f = float(self.youngs_modulus), self.formulation
        names = self.variable_names()
        u = [fields[d] for d in names] + [0] * (3 - len(names))
        ux, uy, uz = u
        if f == "axisymmetric":
            # (rr, zz, theta-theta) in slots 0, 1, 2, and the rz shear in slot 5.
            strain = [
                ux.diff(x),
                uy.diff(_symbols()[1]),
                ux / x,
                0,
                0,
                ux.diff(_symbols()[1]) + uy.diff(x),
            ]
        else:
            gx, gy, gz = (
                _gradient(ux),
                _gradient(uy),
                _gradient(uz) if f == "three_dimensional" else [0, 0, 0],
            )
            strain = [
                gx[0],
                gy[1],
                gz[2] if f == "three_dimensional" else 0,
                gy[2] + gz[1],
                gx[2] + gz[0],
                gx[1] + gy[0],
            ]
        if f == "plane_stress":
            c = E / (1 - nu**2)
            stress = [
                c * (strain[0] + nu * strain[1]),
                c * (strain[1] + nu * strain[0]),
                0,
                0,
                0,
                E / (2 * (1 + nu)) * strain[5],
            ]
        else:
            lam, mu = E * nu / ((1 + nu) * (1 - 2 * nu)), E / (2 * (1 + nu))
            trace = strain[0] + strain[1] + strain[2]
            stress = [
                lam * trace + 2 * mu * strain[0],
                lam * trace + 2 * mu * strain[1],
                lam * trace + 2 * mu * strain[2],
                mu * strain[3],
                mu * strain[4],
                mu * strain[5],
            ]
        h = self._plane_thickness()
        force = (
            [float(b) for b in self.body_force]
            if self.body_force is not None
            else [0.0] * len(names)
        )
        if f == "axisymmetric":
            return {
                names[0]: ([h * stress[0], h * stress[5], 0], h * stress[2] / x - h * force[0]),
                names[1]: ([h * stress[5], h * stress[1], 0], -h * force[1]),
            }
        rows = [
            [stress[0], stress[5], stress[4]],
            [stress[5], stress[1], stress[3]],
            [stress[4], stress[3], stress[2]],
        ]
        return {
            variable: ([h * r for r in rows[component]], -h * force[component])
            for component, variable in enumerate(names)
        }


# ---------------------------------------------------------------------------
# Incompressible flow
# ---------------------------------------------------------------------------
FLOW_FORMULATIONS = ("penalty", "pressure", "Taylor-Hood")
FLOW_TIME_INTEGRATIONS = ("monolithic", "projection")


def _is_quadratic_mesh(mesh) -> bool:
    linear = {"Edge2", "Tri3", "Quad4", "Tet4", "Hex8", "Wedge6", "Pyramid5"}
    return all(mesh.element_type(e) not in linear for e in range(mesh.num_elements))


@register
@dataclass
class FluidDynamics(Physics):
    r"""The steady or transient flow of an incompressible Newtonian fluid:
    ρ (v·∇)v = −∇p + μ Δv + f, ∇·v = 0. The penalty
    formulation eliminates the pressure, the pressure formulation keeps it
    with equal-order interpolation and stabilisation, and the Taylor-Hood
    formulation keeps it with a quadratic velocity and a linear pressure."""

    type_name: ClassVar[str] = "fluid_dynamics"
    module: ClassVar[str] = "fluids"
    vector: ClassVar[bool] = True

    #: The velocity field: the variables velocity_x, velocity_y, velocity_z (velocity_r, velocity_z in axisymmetric coordinates).
    velocity: ClassVar[str] = "velocity"
    dynamic_viscosity: float = parameter(
        1.0,
        unit="Pa s",
        description="Dynamic viscosity mu. Default 1, the value of the non-dimensional equations.",
    )
    density: float = parameter(
        0.0,
        unit="kg/m^3",
        description="Density rho of the convective term. Default 0, i.e., Stokes flow.",
    )
    formulation: str = parameter(
        "penalty",
        description="penalty, pressure or Taylor-Hood. Default penalty. Taylor-Hood needs method='FEM' and a quadratic mesh.",
    )
    penalty_parameter: float = parameter(
        1.0e8,
        description="Penalty parameter gamma of the penalty formulation, p = −γ ∇·v. Default 1e8, large enough for the velocity to be divergence-free to about 1e-8 of its gradient in non-dimensional problems.",
    )
    #: The pressure variable of the pressure and Taylor-Hood formulations.
    pressure: ClassVar[str] = "pressure"
    body_force: Sequence[float | str] | None = parameter(
        None,
        unit="N/m^3",
        description="Body force per unit volume, one value per component, each a constant or an expression of (x, y, z, t). It enters the momentum equations and their streamline stabilisation. Default none.",
    )
    mass_flux_boundaries: Sequence[str] | None = parameter(
        None,
        description="Side sets where the mass equation of the pressure formulations takes the flow through the boundary. Default: every side set.",
    )
    block: Sequence[str] = parameter((), description="Blocks of the fluid. Default: every block.")
    initial_condition: Mapping[str, object] = parameter(
        default_factory=dict,
        description="The initial values of the fields of a transient flow: field name -> value, the velocity as one value per component, e.g. {'velocity': ['1 - y^2', 0], 'pressure': 0}. Each value is a constant, the name of a function or an expression of x, y, z. Default: at rest.",
    )
    viscous_form: str = parameter(
        "stress",
        description="The form of the viscous term: stress (the default: mu (grad v + grad v^T), whose natural boundary condition is a zero traction) or laplacian (mu grad v, the same equations for a constant viscosity, whose natural boundary condition is the do-nothing outflow mu dv/dn - p n = 0 of the DFG benchmarks and of Nek5000). The projection time integration always takes the laplacian form.",
    )
    time_integration: str = parameter(
        "monolithic",
        description="How a transient solve advances the flow: monolithic (the default: Newton's method on the velocity and the pressure together, with the theta method of solve_transient) or projection (the splitting of Nek5000 and nekRS: a pressure Poisson equation, then one Helmholtz equation per velocity component, with BDFk and EXTk of order time_order). The projection needs formulation='pressure' and method 'FEM' or 'DMCDM'.",
    )
    time_order: int = parameter(
        2,
        description="The order k of the backward difference (BDFk) and of the extrapolation of the advection (EXTk) of the projection time integration: 1, 2 or 3. Default 2. The first steps take the orders below it.",
    )

    def __post_init__(self):
        if self.viscous_form not in ("stress", "laplacian"):
            raise InputError(
                f"fluid_dynamics: viscous_form is stress or laplacian, not '{self.viscous_form}'."
            )
        if self.time_integration not in FLOW_TIME_INTEGRATIONS:
            raise InputError(
                f"fluid_dynamics: unknown time_integration '{self.time_integration}'. Use {', '.join(FLOW_TIME_INTEGRATIONS)}."
            )
        if self.time_integration == "projection":
            if self.formulation != "pressure":
                raise InputError(
                    "fluid_dynamics: the projection time integration needs formulation='pressure'."
                )
            if self.time_order not in (1, 2, 3):
                raise InputError("fluid_dynamics: time_order is 1, 2 or 3.")
            if not self.density > 0:
                raise InputError(
                    "fluid_dynamics: the projection time integration needs a positive density; a flow without inertia (Stokes flow) has no time derivative to integrate."
                )
        if self.formulation not in FLOW_FORMULATIONS:
            raise InputError(
                f"fluid_dynamics: unknown formulation '{self.formulation}'. Use {', '.join(FLOW_FORMULATIONS)}."
            )

    @property
    def _pressure_formulation(self) -> bool:
        return self.formulation in ("pressure", "Taylor-Hood")

    @property
    def velocities(self) -> list[str]:
        """The velocity variables, one per component."""
        dimension, axisymmetric = getattr(self, "_geometry", (2, False))
        return _components(self.velocity, dimension, axisymmetric)

    def _resolve(self, coordinates: str, dimension: int) -> None:
        self._geometry = (dimension, coordinates == "axisymmetric")

    def variable_names(self) -> list[str]:
        return self.velocities + ([self.pressure] if self._pressure_formulation else [])

    def _initial_condition(self, variable: str):
        unknown = sorted(set(self.initial_condition) - {self.velocity, self.pressure})
        if unknown:
            raise InputError(
                f"{label(self.type_name, self.name)}: the initial_condition names {', '.join(unknown)}, which is not a field of this flow ({self.velocity}, {self.pressure})."
            )
        if variable == self.pressure:
            return self.initial_condition.get(self.pressure)
        velocity = self.initial_condition.get(self.velocity)
        if velocity is None:
            return None
        values = (
            _list_items(velocity, len(self.velocities), label(self.type_name, self.name))
            if isinstance(velocity, str)
            else list(velocity)
        )
        if len(values) != len(self.velocities):
            raise InputError(
                f"{label(self.type_name, self.name)}: the initial velocity needs {len(self.velocities)} values, one per component."
            )
        return values[self.velocities.index(variable)]

    def _variable_units(self) -> dict:
        return {**dict.fromkeys(self.velocities, "m/s"), self.pressure: "Pa"}

    def _dimension(self) -> int:
        return len(self.velocities)

    def _variable_order(self) -> dict:
        return {self.pressure: "first"} if self.formulation == "Taylor-Hood" else {}

    def _attach(self, problem) -> None:
        self._resolve(problem.coordinates, problem.mesh.dimension)
        if self.formulation == "Taylor-Hood":
            if problem.method != "FEM":
                raise InputError(
                    f"{label(self.type_name, self.name)}: the Taylor-Hood element needs the finite element method, dm.Problem(mesh, method='FEM'). The other methods use formulation='pressure'."
                )
            if not _is_quadratic_mesh(problem.mesh):
                raise InputError(
                    f"{label(self.type_name, self.name)}: the Taylor-Hood element needs quadratic elements (e.g., element_type='Quad9', 'Tri6', 'Tet10' or 'Hex27', or mesh.second_order())."
                )
        super()._attach(problem)

    def _build(self, problem) -> None:
        velocities, restrict = list(self.velocities), self._restriction()
        existing = set(problem._problem.object_names())
        # The equal-order pressure needs the pressure stabilisation of the mass equation, and the
        # inertia takes the streamline stabilisation (SUPG). The Taylor-Hood element needs neither.
        stabilization = self.formulation == "pressure"
        supg = stabilization and self.density != 0.0
        force = (
            [f if isinstance(f, str) else float(f) for f in self.body_force]
            if self.body_force is not None
            else [0.0] * len(velocities)
        )
        for component, variable in enumerate(velocities):
            problem.add_kernel(
                "viscous_stress",
                variable=variable,
                velocities=velocities,
                dynamic_viscosity=self.dynamic_viscosity,
                form=self.viscous_form,
                **restrict,
            )
            if self._pressure_formulation:
                problem.add_kernel(
                    "pressure_gradient",
                    variable=variable,
                    velocities=velocities,
                    pressure=self.pressure,
                    **restrict,
                )
                if supg:
                    problem.add_kernel(
                        "momentum_stabilization",
                        variable=variable,
                        velocities=velocities,
                        pressure=self.pressure,
                        **restrict,
                    )
            else:
                problem.add_kernel(
                    "penalty_incompressibility",
                    variable=variable,
                    velocities=velocities,
                    penalty_parameter=self.penalty_parameter,
                    **restrict,
                )
            if self.density:
                problem.add_kernel(
                    "convective_inertia",
                    variable=variable,
                    velocities=velocities,
                    density=self.density,
                    **restrict,
                )
            if force[component] != 0.0:
                problem.add_kernel(
                    "body_force",
                    variable=variable,
                    value=force[component],
                    **restrict,
                )
            # The inertia rho dv/dt of a transient solve; a steady solve leaves it out.
            if self.density:
                problem.add_kernel(
                    "time_derivative",
                    variable=variable,
                    coeff=self.density,
                    **restrict,
                )
        if self._pressure_formulation:
            # The stabilisation reads the density, the viscosity, the body forces and the
            # buoyancy from the momentum kernels above (and from those that couplings add).
            problem.add_kernel(
                "mass_conservation",
                variable=self.pressure,
                velocities=velocities,
                stabilization=stabilization,
                **restrict,
            )
            boundaries = (
                list(self.mass_flux_boundaries)
                if self.mass_flux_boundaries is not None
                else list(problem.mesh.sideset_names())
            )
            if boundaries:
                problem.add_boundary_condition(
                    "mass_flux",
                    boundaries,
                    variable=self.pressure,
                    velocities=velocities,
                )
        else:
            problem.add_property("penalty_pressure", velocities=velocities, **restrict)
        if self.time_integration == "projection":
            from . import _core
            from .problem import _as_function

            settings = _core.ProjectionSettings()
            settings.velocities = velocities
            settings.pressure = self.pressure
            settings.density = self.density
            settings.dynamic_viscosity = self.dynamic_viscosity
            settings.order = self.time_order
            settings.set_body_force([None if f == 0.0 else _as_function(f) for f in force])
            settings.own_objects = [n for n in problem._problem.object_names() if n not in existing]
            self._projection = _core.attach_projection(
                problem._problem, problem._distributed, settings
            )

    def _manufactured_terms(self, fields, coordinates, dimension):
        from .mms import _symbols

        if not self._pressure_formulation:
            raise InputError(
                f"{label(self.type_name, self.name)}: manufactured solutions need formulation='pressure' or 'Taylor-Hood', which keep the pressure as a variable."
            )
        x = _symbols()[0]
        u = [fields[v] for v in self.velocities]
        p = fields[self.pressure]
        mu, rho = float(self.dynamic_viscosity), float(self.density)
        grads = [_gradient(c) for c in u]
        out = {}
        laplacian = self.viscous_form == "laplacian"
        for i, v in enumerate(self.velocities):
            flux = [
                mu * grads[i][d] if laplacian else mu * (grads[i][d] + grads[d][i])
                for d in range(dimension)
            ]
            flux[i] = flux[i] - p
            flux += [0] * (3 - dimension)
            source = rho * sum(u[d] * grads[i][d] for d in range(dimension))
            if coordinates == "axisymmetric" and i == 0:
                source = source + ((1 if laplacian else 2) * mu * u[0] / x - p) / x
            out[v] = (flux, source)
        out[self.pressure] = ([-c for c in u] + [0] * (3 - dimension), 0)
        return out

    def _set_manufactured_source(self, forcing):
        # The streamline stabilisation is consistent only when its residual
        # carries the same body force as the momentum equations.
        if self.body_force is not None:
            raise InputError(
                f"{label(self.type_name, self.name)}: a manufactured solution sets the body force itself, so leave body_force unset."
            )
        self.body_force = [forcing[v] for v in self.velocities]
        return set(self.velocities)


# ---------------------------------------------------------------------------
# Couplings
# ---------------------------------------------------------------------------
@register
@dataclass
class ThermalExpansion(Coupling):
    """The thermal strain alpha (T - T_0) of a solid_mechanics physics, from the
    temperature of a heat_transfer physics."""

    type_name: ClassVar[str] = "thermal_expansion"
    module: ClassVar[str] = "solid_mechanics"

    linear_expansion_coeff: float = parameter(
        unit="1/K", description="Isotropic linear thermal expansion coefficient alpha."
    )
    stress_free_temperature: float = parameter(
        unit="K", description="Temperature T_0 at which the solid is free of thermal strain."
    )
    heat_transfer: Physics | None = parameter(
        None,
        description="The heat_transfer physics that gives the temperature, the object that add_physics returned. Default: the only heat_transfer physics of the problem.",
    )
    solid_mechanics: Physics | None = parameter(
        None,
        description="The solid_mechanics physics that expands, the object that add_physics returned. Default: the only solid_mechanics physics of the problem.",
    )

    def _apply(self, problem) -> None:
        heat = self._physics(problem, "heat_transfer", HeatTransfer)
        solid = self._physics(problem, "solid_mechanics", SolidMechanics)
        solid._coupled["thermal_expansion"] = dict(
            temperature=heat.temperature,
            linear_expansion_coeff=self.linear_expansion_coeff,
            stress_free_temperature=self.stress_free_temperature,
        )


@register
@dataclass
class NonisothermalFlow(Coupling):
    """The coupling of an fluid_dynamics and a heat_transfer physics: the
    flow carries the heat, rho c_p v . grad T in the energy equation, and,
    when gravity and vol_expansion_coeff are given, the temperature
    drives the flow by the Boussinesq buoyancy force
    f = -rho beta (T - T_0) g, which makes a natural convection problem."""

    type_name: ClassVar[str] = "nonisothermal_flow"
    module: ClassVar[str] = "fluids"

    heat_transfer: Physics | None = parameter(
        None,
        description="The heat_transfer physics, the object that add_physics returned. Default: the only heat_transfer physics of the problem.",
    )
    fluid_dynamics: Physics | None = parameter(
        None,
        description="The fluid_dynamics physics, the object that add_physics returned. Default: the only fluid_dynamics physics of the problem.",
    )
    gravity: Sequence[float] | None = parameter(
        None,
        unit="m/s^2",
        description="The gravitational acceleration vector g of the buoyancy force. Default none: no buoyancy.",
    )
    vol_expansion_coeff: float | None = parameter(
        None,
        unit="1/K",
        description="Volumetric thermal expansion coefficient beta of the fluid in the buoyancy force. Default none: no buoyancy.",
    )
    reference_temperature: float = parameter(
        0.0,
        unit="K",
        description="Temperature T_0 at which the buoyancy force is zero. Default 0, the reference of non-dimensional problems.",
    )
    scale_with_load: bool = parameter(
        False,
        description="Multiply the buoyancy force by the load factor of load stepping, which reaches a high Rayleigh number from rest. Default false.",
    )

    def __post_init__(self):
        if (self.gravity is None) != (self.vol_expansion_coeff is None):
            raise InputError(
                "nonisothermal_flow: give both gravity and vol_expansion_coeff for the buoyancy force, or neither."
            )

    @staticmethod
    def _density(flow) -> float:
        # The density of the flow; a Stokes flow has none, and 1 is the value of the
        # non-dimensional equations.
        return float(flow.density) if flow.density else 1.0

    @staticmethod
    def _specific_heat(heat):
        # The specific heat of the heat transfer; 1 is the value of the non-dimensional equations.
        return heat.specific_heat if heat.specific_heat is not None else 1.0

    def _apply(self, problem) -> None:
        # The stabilisation of the flow reads the buoyancy from the Boussinesq_buoyancy kernels.
        self._physics(problem, "heat_transfer", HeatTransfer)
        self._physics(problem, "fluid_dynamics", FluidDynamics)

    def _build(self, problem) -> None:
        heat, flow = self.heat_transfer, self.fluid_dynamics
        restrict = flow._restriction()
        # The heat that the flow carries: by default the streamline stabilisation (SUPG), which
        # measured the closest on the boundary layer, the skew jump and natural convection
        # (24 October 2026).
        scheme = heat.advection_scheme or "SUPG"
        problem.add_kernel(
            "heat_convection",
            variable=heat.temperature,
            velocities=list(flow.velocities),
            density=self._density(flow),
            specific_heat=self._specific_heat(heat),
            advection_scheme="central" if scheme == "SUPG" else scheme,
            **restrict,
        )
        if scheme == "SUPG":
            # Its residual is the sum of the sources of the other kernels of the temperature.
            problem.add_kernel(
                "heat_streamline_stabilization",
                variable=heat.temperature,
                velocities=list(flow.velocities),
                density=self._density(flow),
                specific_heat=self._specific_heat(heat),
                thermal_conductivity=str(
                    heat.thermal_conductivity
                    if heat.thermal_conductivity is not None
                    else "thermal_conductivity"
                ),
                **restrict,
            )
        if self.gravity is None:
            return
        for component, variable in enumerate(flow.velocities):
            if component < len(self.gravity) and self.gravity[component] != 0.0:
                problem.add_kernel(
                    "Boussinesq_buoyancy",
                    variable=variable,
                    velocities=list(flow.velocities),
                    temperature=heat.temperature,
                    gravity=list(self.gravity),
                    density=self._density(flow),
                    vol_expansion_coeff=self.vol_expansion_coeff,
                    reference_temperature=self.reference_temperature,
                    scale_with_load=self.scale_with_load,
                    **restrict,
                )
