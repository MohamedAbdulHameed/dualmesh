# SPDX-License-Identifier: LGPL-2.1-or-later
"""The general_form_PDE physics: partial differential equations written as text or by their
coefficients, read by the model language."""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import ClassVar

from ..model import engine_form
from ..model.dimensions import comparisons
from ..model.sympy_form import coordinate_symbols, sympy_module, to_sympy
from ..model.syntax import equation_scope, read_condition, read_equations
from ..parameters import InputError, parameter, suggestion
from .base import CONDITION_PARAMETERS, Physics, is_condition_text, label, register


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


def _holds_condition(node) -> bool:
    """Whether an expression of the model language holds a condition: conditional, maximum,
    minimum or a comparison."""
    pending, seen = [node], set()
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        if current.kind in ("conditional", "maximum", "minimum") or current.kind in comparisons:
            return True
        pending.extend(current.operands)
    return False


@register
@dataclass
class GeneralFormPDE(Physics):
    r"""Partial differential equations written as they are printed, or by
    their coefficients.

    ``equations`` holds the equations as text, for example
    ``"rho*cp*dT/dt = div(k*grad(T)) + q"``, in the language that
    :doc:`/user_guide/equations` defines.  The coefficients ``m``, ``d``,
    ``c``, ``alpha``, ``gamma``, ``beta``, ``a`` and ``f`` give instead the
    equation of a scalar field :math:`u`

    .. math::

        m \frac{\partial^2 u}{\partial t^2} + d \frac{\partial u}{\partial t}
        + \nabla \cdot \mathbf{F} + \boldsymbol{\beta} \cdot \nabla u + a u = f,
        \qquad \mathbf{F} = -c \nabla u - \boldsymbol{\alpha} u + \boldsymbol{\gamma},

    with the flux :math:`\mathbf{F}`, which the boundary condition
    ``normal_flux`` prescribes as :math:`\mathbf{n} \cdot \mathbf{F}` along
    the outward unit normal :math:`\mathbf{n}`.  Any other keyword is a
    definition that the equations use by its name.  A number is a named
    constant, and a text is a named expression, which may use the
    definitions before it.  The model language (:mod:`dualmesh.model`) reads
    the equations into conservation laws, which lower onto the objects of the
    engine as the models of the input do.  The engine compiles their fluxes
    and sources once and differentiates them automatically, so Newton's
    method has the exact Jacobian."""

    type_name: ClassVar[str] = "general_form_PDE"
    module: ClassVar[str] = "framework"
    #: Keywords that are not parameters are definitions.
    takes_definitions: ClassVar[bool] = True

    variables: str | Sequence[str | Mapping[str, str]] = parameter(
        description="The fields, one equation each: a name, or a list whose items are names or {name: kind}, with the kind 'scalar' (the default, one component) or 'vector', for example [{'U': 'vector'}, 'p']. A vector field U has the components U_x, U_y, U_z, or U_r, U_z in axisymmetric coordinates.",
    )
    equations: str | Sequence[str] | Mapping[str, str] = parameter(
        (),
        description="The equations as text, one per field: one text, a list that is matched to the fields by the time derivative, then by the second derivative, then by the field left over, or a dict field -> equation. Default none: the equations come from the coefficients m, d, c, alpha, gamma, beta, a and f.",
    )
    m: object = parameter(
        None,
        description="Mass coefficient m of d2u/dt2. Default none: no second time derivative.",
    )
    d: object = parameter(
        None,
        description="Damping or mass coefficient d of du/dt. Default none: no first time derivative.",
    )
    c: object = parameter(
        None,
        description="Diffusion coefficient c in div(-c grad u): an expression, a list of one value per dimension (orthotropic), or a list of dimension squared values, row by row (a tensor). With several fields it is a dict field -> value, and a value that is itself a dict field -> value couples the equation to the gradients of other fields. Default none: no diffusion.",
    )
    alpha: object = parameter(
        None,
        description="Conservative flux convection coefficient alpha in div(-alpha u), one value per dimension. Default none: no such term.",
    )
    gamma: object = parameter(
        None,
        description="Conservative flux source gamma in div(gamma), one value per dimension. Default none: no such term.",
    )
    beta: object = parameter(
        None,
        description="Convection coefficient beta of beta . grad u, one value per dimension. Default none: no convection.",
    )
    a: object = parameter(
        None, description="Absorption coefficient a of the term a u. Default none: no absorption."
    )
    f: object = parameter(None, description="Source term f. Default none: no source.")
    units: str | Mapping[str, str] = parameter(
        default_factory=dict,
        description="The SI unit of each field, which the output files write with its values: one unit, or a dict field -> unit. Default: no unit, since the field of a general equation may be dimensionless.",
    )
    initial_condition: object = parameter(
        default_factory=dict,
        description="The initial value of each field: a constant, the name of a function or an expression of x, y and z, or a dict field -> such a value. Default 0, a start that needs no knowledge of the solution.",
    )
    block: Sequence[str] = parameter(
        (),
        description="Blocks where the equations hold. Default: every block, so that a single domain needs no block names.",
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
        # The definitions, numbers and texts, are read with the equations. The coordinates are
        # checked when the problem is known (r is a coordinate only in axisymmetric coordinates).
        reserved = set(fields) | {"t", "n", "I", "pi", "e", "eigenvalue"}
        for name in self._definitions:
            if name in reserved:
                raise InputError(
                    f"{where}: '{name}' is a name of the equation language or a field, so it cannot be defined. Rename the definition."
                )
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
        self._scope, self._laws = self._read(coordinates, dimension)

    def _read(self, coordinates: str, dimension: int) -> tuple:
        """The scope of the equations, with their definitions, and the conservation law of each
        field, read by the model language."""
        ranks = {field: 0 if kind == "scalar" else 1 for field, kind in self._kinds.items()}
        try:
            scope = equation_scope(ranks, dimension, coordinates, self._definitions)
            return scope, read_equations(self._texts, self._keys, scope)
        except InputError as error:
            raise InputError(f"general_form_PDE: {error}") from None

    def _attach(self, problem) -> None:
        self._problem = problem
        self._resolve(problem.coordinates, problem.mesh.space_dimension)
        super()._attach(problem)

    def variable_names(self) -> list[str]:
        return self._components(self._geometry[0])

    def boundary_flux(self, boundary: str, variable: str | None = None) -> float:
        """The integral of n . Gamma, the flux of the equation of ``variable``
        along the outward normal, over ``boundary``: the quantity
        ``normal_flux`` of the boundary conditions.  Where the boundary
        prescribes the value it is the reaction there, conserved exactly.
        ``variable`` may be left out when the equation has one."""
        variables = self.variable_names()
        if variable is None:
            if len(variables) != 1:
                raise InputError(
                    f"{label(self.type_name, self.name)}: give the variable of boundary_flux, one of {', '.join(variables)}."
                )
            variable = variables[0]
        elif variable not in variables:
            raise InputError(
                f"{label(self.type_name, self.name)}: '{variable}' is not a variable of this equation; use one of {', '.join(variables)}."
            )
        return self._problem.boundary_flux(variable, boundary)

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

    def add_boundary_condition(self, condition, boundary=None, **parameters):
        """Add a boundary condition written as text, on the side sets or
        node sets named by ``boundary``: ``"u = 0"`` prescribes the value of
        a field, and a condition with the normal flux on the left,
        ``"-k*du/dn = h*(u - 1)"`` or ``"normal_flux = 5"``, prescribes
        n . Gamma, the flux of the equation through the boundary along the
        outward normal n.  A list of texts adds several conditions.  Any
        other keyword is a definition that the texts use by its name, a
        number or an expression, which may use the definitions before it.

            pde.add_boundary_condition("u = 0", boundary=["left", "right"])
            pde.add_boundary_condition("-k*du/dn = h*(u - 1)", boundary="top")
        """
        if not isinstance(condition, (list, tuple)) and not is_condition_text(condition):
            return super().add_boundary_condition(condition, boundary, **parameters)
        where = f"general_form_PDE, boundary condition {condition!r}"
        if boundary is None:
            mesh = self._problem.mesh
            names = sorted(set(mesh.sideset_names()) | set(mesh.nodeset_names()))
            listed = f", one or more of: {', '.join(names)}" if names else ""
            raise InputError(f"{where}: give boundary{listed}.")
        passed = {key: parameters.pop(key) for key in CONDITION_PARAMETERS if key in parameters}
        texts = [condition] if isinstance(condition, str) else list(condition)
        conditions = []
        try:
            scope = self._scope.with_definitions(parameters)
            for text in texts:
                if not is_condition_text(text):
                    raise InputError(f"'{text}' is not a condition written as text")
                conditions += read_condition(text, scope, self._laws)
            values = [
                (self._scope.variable(field, component), flux, value)
                for field, component, flux, value in conditions
            ]
            values = [
                (variable, flux, engine_form.write_expression(value, where, self._scope.variables))
                for variable, flux, value in values
            ]
        except InputError as error:
            raise InputError(f"{where}: {error}") from None
        for name in parameters:
            if name not in scope.reads:
                hint = suggestion(name, [*CONDITION_PARAMETERS, "boundary"], listing=False)
                raise InputError(
                    f"{where}: '{name}' is defined and not used by the condition.{hint}"
                )
        self._scope.reads |= scope.reads
        for variable, flux, value in values:
            engine_form.add_boundary_condition(
                self._problem, boundary, variable, value, flux, **passed
            )
        return None

    def _build(self, problem) -> None:
        unused = [name for name in self._definitions if name not in self._scope.reads]
        if unused:
            hint = suggestion(
                unused[0],
                [f.name for f in dataclasses.fields(self) if not f.name.startswith("_")],
                listing=False,
            )
            raise InputError(
                f"general_form_PDE: '{unused[0]}' is defined and not used by an equation, a definition or a boundary condition.{hint}"
            )
        try:
            for law in self._laws.values():
                engine_form.add_law(
                    problem,
                    law,
                    self._scope.variables,
                    self._geometry[0],
                    True,
                    self._restriction(),
                    "general_form_PDE",
                )
        except InputError as error:
            raise InputError(f"general_form_PDE: {error}") from None

    def _manufactured_terms(self, fields, coordinates, dimension):
        sympy = sympy_module()
        x, y, z, t = coordinate_symbols()
        axes = (x, y, z)
        scope, laws = self._read(coordinates, dimension)

        def leaf(node):
            if node.kind == "position_component":
                return axes[node.direction]
            if node.kind == "time":
                return t
            if node.kind in ("field_component", "gradient_component"):
                exact = fields[scope.variable(node.name, node.component)]
                if node.kind == "field_component":
                    return exact
                return exact.diff(axes[node.direction])
            # A name that the engine resolves, such as eigenvalue.
            return sympy.Symbol(node.name)

        terms = {}
        for law in laws.values():
            names = scope.variables[law.unknown.name]
            for variable, scalar in zip(names, law.components(dimension), strict=True):
                parts = [*(scalar.flux or []), scalar.source, scalar.capacity, scalar.mass]
                for part in parts:
                    if part is not None and _holds_condition(part):
                        raise InputError(
                            f"general_form_PDE: a manufactured solution takes the derivatives of each term with SymPy, which has no smooth derivative of a condition (if, min, max, and, or or a comparison), so it refuses the term '{part}', which holds a condition."
                        )
                flux = [-to_sympy(component, leaf) for component in scalar.flux or []]
                flux += [0] * (3 - len(flux))
                exact = fields[variable]
                source = 0
                if scalar.source is not None:
                    source -= to_sympy(scalar.source, leaf)
                if scalar.capacity is not None:
                    source += to_sympy(scalar.capacity, leaf) * exact.diff(t)
                if scalar.mass is not None:
                    source += to_sympy(scalar.mass, leaf) * exact.diff(t, 2)
                terms[variable] = (flux, source)
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
