# SPDX-License-Identifier: LGPL-2.1-or-later
"""The base of the physics level: the :class:`Physics` and :class:`Coupling` base classes,
the registry of the physics and coupling types, and the helpers that read a boundary condition
written as text.

The other modules of the package share the helpers :func:`vector_components`,
:func:`is_condition_text`, :func:`condition_sides`, :func:`list_items` and :func:`boundary_names`,
and the tuple :data:`CONDITION_PARAMETERS`.  The package does not export them."""

from __future__ import annotations

import dataclasses
import difflib
import re
from typing import ClassVar

from ..parameters import InputError, check_keywords, describe_fields, suggestion

#: Type name -> class, for the physics and the couplings.
_REGISTRY: dict[str, type] = {}


def register(cls):
    """Register a physics or a coupling under its ``type_name``, which must be new."""
    if cls.type_name in _REGISTRY:
        raise TypeError(f"The physics type '{cls.type_name}' is registered twice.")
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
    if type_name not in _REGISTRY:
        raise InputError(
            f"Unknown physics or coupling type '{type_name}'." + suggestion(type_name, _REGISTRY)
        )
    cls = _REGISTRY[type_name]
    where = label(type_name, name)
    if getattr(cls, "takes_definitions", False):
        # Keywords that are not parameters are definitions, kept in their order; one close to a
        # parameter name is taken for a misspelling.
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


def problem_method(physics) -> str:
    """The method of a problem that leaves it out, from its physics: DMCDM, the dual mesh control
    domain method, when one of them is a fluid_dynamics of the equal_order formulation, and FEM,
    the finite element method, otherwise.  A problem takes one method, so a solid or a heat
    conduction beside such a flow takes DMCDM, which solves it too."""
    flow = any(p.type_name == "fluid_dynamics" and p.formulation == "equal_order" for p in physics)
    return "DMCDM" if flow else "FEM"


#: Keywords of a boundary condition written as text that are passed on to its objects; any
#: other keyword is a definition that the text uses by its name.
CONDITION_PARAMETERS = ("point",)


def _expand_conditions(condition, parameters: dict, where: str) -> list[str]:
    """The texts of ``condition`` (one text or a list) with the definitions
    among ``parameters`` written into them, and those definitions removed
    from ``parameters``.  A number is written as its value and a text, in
    parentheses, as itself.  A definition may use the ones before it."""
    texts = [condition] if isinstance(condition, str) else list(condition)
    names = [k for k in parameters if k not in CONDITION_PARAMETERS]
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
        return suggestion(name, [*CONDITION_PARAMETERS, "boundary"], listing=False)

    known: dict = {}
    for name, value in definitions.items():
        if isinstance(value, bool) or not isinstance(value, (int, float, str)):
            raise InputError(
                f"{where}: the definition {name} = {value!r} is not a number or an expression (a text).{hint(name)}"
            )
        # In parentheses, so that a negative number keeps its sign under a power (2*F^2).
        known[name] = (
            f"({float(value)!r})" if not isinstance(value, str) else f"({substitute(value, known)})"
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


# ---- helpers shared by the modules of the physics package -----------------

#: The '=' between the two sides of a condition written as text, apart from <=, >=, == and !=.
_CONDITION_EQUALS = re.compile(r"(?<![<>=!])=(?!=)")


def vector_components(name: str, dimension: int, axisymmetric: bool = False) -> list[str]:
    """The variables of the vector field ``name``: name_x, name_y, name_z,
    or name_r, name_z in axisymmetric coordinates."""
    axes = ("r", "z") if axisymmetric else ("x", "y", "z")[:dimension]
    return [f"{name}_{axis}" for axis in axes]


def is_condition_text(condition) -> bool:
    """Whether ``condition`` is a boundary condition written as text, with a
    single '=' between its two sides."""
    return isinstance(condition, str) and _CONDITION_EQUALS.search(condition) is not None


def condition_sides(condition: str) -> tuple[str, str]:
    """The left and the right side of a boundary condition written as text."""
    match = _CONDITION_EQUALS.search(condition)
    return condition[: match.start()].strip(), condition[match.end() :].strip()


def list_items(text: str, size: int, where: str) -> list[str]:
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


def boundary_names(boundary) -> list[str]:
    """One boundary name or a list of them, as a list."""
    return [boundary] if isinstance(boundary, str) else list(boundary)


class Physics:
    """Base of the physics.  A subclass is a dataclass whose fields are its
    parameters, and it provides :meth:`variable_names` and :meth:`_build`."""

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

    def _initial_velocity(self, variable: str):
        """The initial velocity of a variable of a second-order equation (None: at rest)."""
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
        (the heat leaving the body) for ``heat_conduction``, ``traction``,
        ``total_force`` and ``pressure`` for ``solid_mechanics``,
        ``normal_flux`` for ``general_form_PDE``.  The right side is an
        expression of x, y, z, t, the functions, the properties and the
        fields.  A list of texts adds each.  Three words need no data:
        ``"fixed"`` (every displacement zero), ``"symmetry"`` (the normal
        displacement zero) and ``"periodic"`` (with ``primary`` and
        ``secondary``).  A value prescribed at one ``point`` in place of a
        boundary holds the node closest to it, for example the level of the
        pressure of an enclosed flow.  Any other keyword is a definition that
        the text uses by its name, a number or an expression, as in
        ``add_physics``::

            heat.add_boundary_condition("temperature = 300", boundary="left")
            flow.add_boundary_condition("pressure = 0", point=[0.5, 0.0])
            heat.add_boundary_condition("heat_flux = 75*(temperature - 293)", boundary="top")
            solid.add_boundary_condition("fixed", boundary="jaws")
            solid.add_boundary_condition("total_force = [0, -F, 0]", boundary="grip", F=150.0)
        """
        if isinstance(condition, (list, tuple)) or is_condition_text(condition):
            where = f"{label(self.type_name, self.name)}, boundary condition {condition!r}"
            for text in _expand_conditions(condition, parameters, where):
                if not is_condition_text(text):
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
            self._clamped += boundary_names(boundary)
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
        point = parameters.pop("point", None)
        if point is not None:
            left, right = condition_sides(condition)
            if boundary is not None:
                raise InputError(f"{where}: give a boundary or a point, not both.")
            if left not in self.variable_names():
                raise InputError(
                    f"{where}: a condition at a point prescribes a variable of this physics ({', '.join(self.variable_names())})."
                )
            return problem.add_boundary_condition(
                "point_Dirichlet",
                variable=left,
                point=[float(c) for c in point],
                value=right,
                **parameters,
            )
        if boundary is None:
            mesh = problem.mesh
            names = sorted(set(mesh.sideset_names()) | set(mesh.nodeset_names()))
            raise InputError(f"{where}: give boundary, one or more of: {', '.join(names)}.")
        left, right = condition_sides(condition)
        variables = self.variable_names()
        if getattr(self, "_structural", False) and left in ("traction", "total_force", "pressure"):
            raise InputError(
                f"{where}: the {self.formulation} formulation takes its loads as transverse_load, axial_load and in_plane_load of add_physics; a boundary condition prescribes its variables ({', '.join(variables)})."
            )
        if left in variables:
            self._prescribed.setdefault(left, []).extend(boundary_names(boundary))
            return problem.add_boundary_condition(
                "Dirichlet", boundary, variable=left, value=right, **parameters
            )
        if self.vector and left == "pressure" and self.flux_name == "traction":
            # A pressure p on the surface: the traction -p n.
            # At finite strain the pressure follows the deformed surface.
            if getattr(self, "_finite", False):
                parameters = {"deformation_gradient_property": "deformation_gradient", **parameters}
            return problem.add_boundary_condition(
                "pressure",
                boundary,
                displacements=variables[: self._dimension()],
                pressure=right,
                **parameters,
            )
        if self.vector and left == "total_force" and self.flux_name == "traction":
            components = variables[: self._dimension()]
            values = list_items(right, len(components), where)
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
            values = list_items(right, len(components), where)
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
        such as ``"heat_source"`` of ``heat_conduction``) to the equations of
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
