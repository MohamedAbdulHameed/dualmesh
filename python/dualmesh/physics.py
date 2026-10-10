# SPDX-License-Identifier: LGPL-2.1-or-later
r"""The physics level: sets of equations named in physical terms.

A physics is added to a problem by its type name, creates its variables, and
generates the kernels and the property objects of its equations when the problem is
initialised::

    import dualmesh as dm

    problem = dm.Problem(dm.read_mesh("wrench.msh"), method="fem")
    solid = problem.add_physics(
        "solid_mechanics",
        "solid",
        displacements=["u", "v", "w"],
        youngs_modulus=200.0e9,
        poissons_ratio=0.3,
    )
    solid.add_boundary_condition("fixed_constraint", "jaws")
    solid.add_boundary_condition(
        "traction_boundary_condition", "grip", total_force=[0.0, -150.0, 0.0]
    )
    problem.solve()

The boundary conditions and the extra terms of a physics take the type names
and the parameters of the objects (:doc:`/objects`), with the variables of
the physics filled in.  For a physics with several components, a vector value
(``traction``, ``total_force``, or ``value`` with ``None`` for a free
component) is split into one object per component, and the ``component`` and
the ``thickness`` of each object are filled in.  A coupling between two
physics (``thermal_expansion`` or ``nonisothermal_flow``) is added with
:meth:`dualmesh.Problem.add_coupling`.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import ClassVar

from .parameters import InputError, check_keywords, describe_fields, parameter

__all__ = [
    "Beam",
    "CircularPlate",
    "CoefficientFormPDE",
    "Coupling",
    "NonisothermalFlow",
    "HeatTransfer",
    "IncompressibleFlow",
    "Physics",
    "Plate",
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
    check_keywords(f"{type_name} '{name}'", cls, parameters)
    for f in dataclasses.fields(cls):
        if (
            f.default is dataclasses.MISSING
            and f.default_factory is dataclasses.MISSING
            and f.name not in parameters
        ):
            raise InputError(
                f"{type_name} '{name}': missing required parameter '{f.name}' ({f.metadata.get('description', '')})"
            )
    instance = cls(**parameters)
    instance.name = name
    return instance


def _vector(value, size: int, where: str):
    if isinstance(value, (int, float, str)) or value is None:
        raise InputError(
            f"{where}: give one value per component, a list of {size} values (None for a free component)."
        )
    value = list(value)
    if len(value) != size:
        raise InputError(f"{where}: give {size} values, one per component, not {len(value)}.")
    return value


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
                    f"{self.type_name} '{self.name}': the variable '{variable}' exists with order '{existing}', and this physics needs order '{order.get(variable, 'mesh')}'. Declare it with order='{order.get(variable, 'mesh')}', or let the physics create it."
                )

    def _variable_order(self) -> dict:
        return {}

    def _variable_units(self) -> dict:
        """The SI unit of each variable of the physics."""
        return {}

    def _initial_condition(self, variable: str):
        return None

    def _object_name(self, name: str) -> str:
        return f"{self.name}_{name}"

    def _thickness(self) -> float:
        return float(getattr(self, "thickness", 1.0))

    # ---- conditions and extra terms ------------------------------------
    def add_boundary_condition(self, condition: str, name: str, **parameters):
        """Add a boundary condition of the object reference to the equations
        of this physics.

        ``name`` is the name of the side set it acts on, unless ``boundary``
        is given.  The variable is filled in.  For a physics with several
        components, ``traction`` and ``total_force`` of
        ``traction_boundary_condition`` and ``value`` of
        ``Dirichlet_boundary_condition`` are lists with one entry per
        component (``None`` for a component that is left free), and
        ``fixed_constraint`` and ``symmetry_boundary_condition`` act on all
        the components."""
        problem = self._problem
        where = f"{self.type_name} '{self.name}', boundary condition '{name}'"
        if "boundary" not in parameters:
            parameters["boundary"] = [name]
        variables = self.variable_names()
        object_name = self._object_name(name)
        if condition in ("fixed_constraint", "symmetry_boundary_condition") and self.vector:
            parameters.setdefault("displacements", variables[: self._dimension()])
            return problem.add_boundary_condition(condition, object_name, **parameters)
        if self.vector and condition == "traction_boundary_condition":
            keys = [k for k in ("traction", "total_force") if k in parameters]
            if len(keys) != 1:
                raise InputError(
                    f"{where}: give one of 'traction' and 'total_force', as a list with one value per component. The two define the same load."
                )
            values = _vector(parameters.pop(keys[0]), self._dimension(), where)
            parameters.setdefault("thickness", self._thickness())
            for variable, value in zip(variables, values):
                if value is not None and value != 0.0:
                    problem.add_boundary_condition(
                        condition,
                        f"{object_name}_{variable}",
                        variable=variable,
                        **{keys[0]: value},
                        **parameters,
                    )
            return None
        if self.vector and condition == "pressure_boundary_condition":
            parameters.setdefault("thickness", self._thickness())
            for component, variable in enumerate(variables[: self._dimension()]):
                problem.add_boundary_condition(
                    condition,
                    f"{object_name}_{variable}",
                    variable=variable,
                    component=component,
                    **parameters,
                )
            return None
        if (
            self.vector
            and condition == "Dirichlet_boundary_condition"
            and "value" in parameters
            and not isinstance(parameters["value"], (int, float, str))
        ):
            values = _vector(parameters.pop("value"), self._dimension(), where)
            for variable, value in zip(variables, values):
                if value is not None:
                    problem.add_boundary_condition(
                        condition,
                        f"{object_name}_{variable}",
                        variable=variable,
                        value=value,
                        **parameters,
                    )
            return None
        if "variable" not in parameters and "variables" not in parameters:
            if len(variables) == 1:
                parameters["variable"] = variables[0]
            elif self.vector:
                parameters["variables"] = variables[: self._dimension()]
        return problem.add_boundary_condition(condition, object_name, **parameters)

    def add_kernel(self, kernel: str, name: str | None = None, **parameters):
        """Add a term of the object reference to the equations of this
        physics.  The variable is filled in for a physics with one variable,
        and must be given for one with several."""
        variables = self.variable_names()
        if "variable" not in parameters:
            if len(variables) != 1:
                raise InputError(
                    f"{self.type_name} '{self.name}': give 'variable' (one of {', '.join(variables)}) for the kernel '{kernel}'."
                )
            parameters["variable"] = variables[0]
        return self._problem.add_kernel(kernel, self._object_name(name or kernel), **parameters)

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
            f"{self.type_name} '{self.name}': manufactured solutions are not available for this physics."
        )

    def _set_manufactured_source(self, forcing: dict) -> set:
        """Take the manufactured sources ``{variable: text}`` into the
        physics' own source terms where the physics needs them there (e.g.,
        for a stabilisation that uses the body force), and return the
        variables so served.  The sources of the other variables are added
        as body_force kernels."""
        return set()


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
        physics_name = getattr(self, key)
        physics = problem._physics.get(physics_name)
        if physics is None:
            raise InputError(
                f"{self.type_name} '{self.name}': the problem has no physics '{physics_name}' ({key}). Physics: {', '.join(problem._physics) or 'none'}."
            )
        if not isinstance(physics, cls):
            raise InputError(
                f"{self.type_name} '{self.name}': '{physics_name}' is a {physics.type_name} physics, and '{key}' needs a {cls.type_name} physics."
            )
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

    temperature: str = parameter(
        "temperature", description="Name of the temperature variable. Default temperature."
    )
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
        description="Specific heat c_p of the heat capacity term. Default none (see density).",
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
                self._object_name("conduction"),
                variable=T,
                thermal_conductivity_property="thermal_conductivity",
                **restrict,
            )
        else:
            problem.add_kernel(
                "heat_conduction",
                self._object_name("conduction"),
                variable=T,
                thermal_conductivity=self.thermal_conductivity,
                **restrict,
            )
        if self.heat_source is not None:
            problem.add_kernel(
                "heat_source",
                self._object_name("heat_source"),
                variable=T,
                heat_source=self.heat_source,
                **restrict,
            )
        if self.density is not None:
            problem.add_kernel(
                "heat_conduction_time_derivative",
                self._object_name("heat_capacity"),
                variable=T,
                density=self.density,
                specific_heat=self.specific_heat,
                **restrict,
            )

    def _manufactured_terms(self, fields, coordinates, dimension):
        from .mms import _symbols

        if self.thermal_conductivity is None:
            raise InputError(
                f"heat_transfer '{self.name}': a manufactured solution needs the thermal_conductivity as a constant or an expression of x, y, z and t."
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
# Coefficient form PDE
# ---------------------------------------------------------------------------
def _uses(expression, name: str) -> tuple[bool, bool]:
    """Whether a coefficient uses the variable ``name``, and whether it uses
    x, y, z or t."""
    if not isinstance(expression, str):
        return False, False
    symbols = {str(s) for s in _sympy_expression(expression).free_symbols}
    return name in symbols, bool(symbols & {"x", "y", "z", "t"})


_FIELD_NAME = r"[A-Za-z_][A-Za-z0-9_]*"


def _number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _text_uses_fields(text, fields) -> bool:
    """Whether an expression uses one of the names ``fields`` (the fields, the
    properties and the symbol eigenvalue), or a gradient, so that it needs a
    parsed_kernel."""
    import re

    if _number(text):
        return False
    names = set(re.findall(_FIELD_NAME, str(text)))
    special = ("grad_x", "grad_y", "grad_z", "eigenvalue")
    return bool(names & set(fields)) or any(n in names for n in special)


def _add_term(flux: list, component: int, text: str) -> None:
    """Add a term to one component of a flux written as text."""
    flux[component] = f"{flux[component]} + ({text})" if flux[component] != "0" else f"({text})"


@register
@dataclass
class CoefficientFormPDE(Physics):
    """One or several partial differential equations written by their coefficients. The equation of each field u is d ∂u/∂t + ∇·(−c ∇u − α u) + β·∇u + a u = f. A coefficient of an equation is one value, which acts on the field of that equation, or a dict field name -> value, which couples the equation to other fields. A value is a constant, or an expression of the fields, their gradients (grad_x(u), grad_y(u), grad_z(u)), x, y, z, t and named constants, which may make the equations nonlinear. With one field, every input may be given as a plain value; with several, every input is a dict keyed by the field of the equation."""

    type_name: ClassVar[str] = "coefficient_form_PDE"
    module: ClassVar[str] = "framework"

    variables: Sequence[str] = parameter(
        ("u",), description="Names of the dependent variables, one equation each. Default ['u']."
    )
    diffusion_coefficient: object = parameter(
        None,
        description="Diffusion coefficient c of each equation: a constant, an expression, or a constant tensor given as one value per dimension (its diagonal) or as dimension squared values row by row. Default: the property diffusion_coefficient (with one field) or diffusion_coefficient_<field> (with several) of the property objects of the problem. Give 0 for an equation without diffusion.",
    )
    absorption_coefficient: object = parameter(
        None, description="Absorption coefficient a of the term a u of each equation. Default none."
    )
    source: object = parameter(
        None, description="Source f of each equation, a constant or an expression. Default none."
    )
    time_derivative_coefficient: object = parameter(
        None,
        description="Coefficient d of ∂u/∂t of each equation: a constant or an expression of (x, y, z, t). Default none: the equation is steady.",
    )
    convection_coefficient: object = parameter(
        None,
        description="Convection vector beta of the term beta . grad u, one value per dimension, of each equation. Default none.",
    )
    conservative_flux_convection_coefficient: object = parameter(
        None,
        description="Vector alpha of the conservative flux -alpha u, one value per dimension, of each equation. Default none.",
    )
    constants: Mapping[str, float] = parameter(
        default_factory=dict,
        description="Named constants that the expressions use: name -> value. Default none.",
    )
    properties: Sequence[str] = parameter(
        (),
        description="Scalar properties of the property objects of the problem that the expressions use, each by its name. A property object on each block gives coefficients that differ between regions. Default none.",
    )
    units: object = parameter(
        None,
        description="The SI unit of each field, which the output files write with its values. Default: dimensionless fields.",
    )
    initial_condition: object = parameter(
        None,
        description="Initial value of each field, a constant or the name of a function. Default 0.",
    )
    block: Sequence[str] = parameter(
        (), description="Blocks where the equations hold. Default: every block."
    )

    _COEFFICIENTS: ClassVar[tuple] = (
        "diffusion_coefficient",
        "absorption_coefficient",
        "convection_coefficient",
        "conservative_flux_convection_coefficient",
    )
    _PER_EQUATION: ClassVar[tuple] = (
        "source",
        "time_derivative_coefficient",
        "units",
        "initial_condition",
    )

    def __post_init__(self):
        fields = list(self.variables)
        if not fields:
            raise InputError("coefficient_form_PDE: give the names of the fields in 'variables'.")
        if len(set(fields)) != len(fields):
            raise InputError("coefficient_form_PDE: every name in 'variables' must be different.")
        clash = sorted(set(self.constants) & set(fields))
        if clash:
            raise InputError(
                f"coefficient_form_PDE: the constant(s) {', '.join(clash)} have the name of a field. Rename them."
            )
        for key in self._COEFFICIENTS + self._PER_EQUATION:
            value = getattr(self, key)
            if value is None:
                continue
            if not isinstance(value, Mapping):
                if len(fields) > 1:
                    raise InputError(
                        f"coefficient_form_PDE: with several fields, give the {key} as a dict keyed by the field of each equation, for example {{'{fields[0]}': ...}}."
                    )
                continue
            unknown = sorted(set(value) - set(fields))
            if unknown:
                raise InputError(
                    f"coefficient_form_PDE: the {key} names {', '.join(unknown)}, which is not in 'variables' ({', '.join(fields)})."
                )
            if key in self._COEFFICIENTS:
                for equation, entry in value.items():
                    if isinstance(entry, Mapping):
                        unknown = sorted(set(entry) - set(fields))
                        if unknown:
                            raise InputError(
                                f"coefficient_form_PDE: the {key} of the equation of {equation} couples to {', '.join(unknown)}, which is not in 'variables'."
                            )
            elif any(isinstance(entry, Mapping) for entry in value.values()):
                raise InputError(
                    f"coefficient_form_PDE: the {key} takes one value for each equation, not a dict of fields."
                )

    def variable_names(self) -> list[str]:
        return list(self.variables)

    # ---- the inputs, per equation -----------------------------------------
    def _per_equation(self, key: str, field: str):
        value = getattr(self, key)
        if value is None:
            return None
        if isinstance(value, Mapping):
            value = value.get(field)
        if key in ("units", "initial_condition"):
            return value
        return self._with_constants(value)

    def _with_constants(self, value):
        """``value`` with every named constant replaced by its number, in a
        text, a list or a dict of them."""
        import re

        if isinstance(value, str) and self.constants:
            for name, number in self.constants.items():
                value = re.sub(rf"\b{re.escape(name)}\b", f"({float(number)!r})", value)
            return value
        if isinstance(value, (list, tuple)):
            return [self._with_constants(v) for v in value]
        if isinstance(value, Mapping):
            return {k: self._with_constants(v) for k, v in value.items()}
        return value

    def _couplings(self, key: str, field: str) -> dict:
        """The terms of coefficient ``key`` in the equation of ``field``:
        coupled field -> value."""
        entry = self._per_equation(key, field)
        if entry is None:
            return {}
        if isinstance(entry, Mapping):
            return dict(entry)
        return {field: entry}

    def _property_parameters(self) -> dict:
        return {"coupled_properties": list(self.properties)} if self.properties else {}

    def _variable_units(self) -> dict:
        return {f: self._per_equation("units", f) or "" for f in self.variable_names()}

    def _initial_condition(self, variable: str):
        return self._per_equation("initial_condition", variable)

    def _diffusion_property(self, field: str) -> str:
        return (
            "diffusion_coefficient"
            if len(self.variable_names()) == 1
            else f"diffusion_coefficient_{field}"
        )

    def _tensor(self, value, dimension: int, field: str):
        """A diffusion tensor as dimension x dimension entries."""
        n = len(value)
        if n == dimension:
            return [
                [value[i] if i == j else 0.0 for j in range(dimension)] for i in range(dimension)
            ]
        if n == dimension * dimension:
            return [[value[i * dimension + j] for j in range(dimension)] for i in range(dimension)]
        raise InputError(
            f"coefficient_form_PDE '{self.name}': the diffusion tensor of the equation of {field} has {n} values; give {dimension} (the diagonal) or {dimension * dimension} (row by row)."
        )

    def _vector(self, value, dimension: int, key: str, field: str):
        value = list(value)
        if len(value) != dimension:
            raise InputError(
                f"coefficient_form_PDE '{self.name}': the {key} of the equation of {field} needs {dimension} values, one per dimension."
            )
        return value

    # ---- the equations ----------------------------------------------------
    def _build(self, problem) -> None:
        dimension = problem.mesh.dimension
        restrict = self._restriction()
        fields = self.variable_names()
        # Names that need a parsed_kernel: the fields and the properties.
        names = fields + list(self.properties)
        axes = "xyz"[:dimension]
        for u in fields:
            flux = ["0"] * dimension
            source = []
            dedicated = 0
            # Diffusion: Gamma = -c grad u.
            diffusion = self._couplings("diffusion_coefficient", u)
            if not diffusion:
                problem.add_kernel(
                    "diffusion",
                    self._object_name(f"{u}_diffusion"),
                    variable=u,
                    diffusivity_property=self._diffusion_property(u),
                    **restrict,
                )
                dedicated += 1
            for v, c in diffusion.items():
                if isinstance(c, (list, tuple)):
                    tensor = self._tensor(c, dimension, u)
                    if v == u and all(_number(e) for row in tensor for e in row):
                        problem.add_kernel(
                            "anisotropic_diffusion",
                            self._object_name(f"{u}_diffusion"),
                            variable=u,
                            diffusivity_tensor=[float(e) for row in tensor for e in row],
                            **restrict,
                        )
                        dedicated += 1
                        continue
                    for i in range(dimension):
                        _add_term(
                            flux,
                            i,
                            " + ".join(
                                f"-({tensor[i][j]})*grad_{axes[j]}({v})" for j in range(dimension)
                            ),
                        )
                elif v == u and not _text_uses_fields(c, names):
                    if c != 0.0:
                        problem.add_kernel(
                            "diffusion",
                            self._object_name(f"{u}_diffusion"),
                            variable=u,
                            diffusivity=c,
                            **restrict,
                        )
                        dedicated += 1
                else:
                    for i in range(dimension):
                        _add_term(flux, i, f"-({c})*grad_{axes[i]}({v})")
            # Conservative convection: Gamma = -alpha u.
            for v, alpha in self._couplings("conservative_flux_convection_coefficient", u).items():
                alpha = self._vector(
                    alpha, dimension, "conservative_flux_convection_coefficient", u
                )
                if v == u and all(_number(a) for a in alpha):
                    problem.add_kernel(
                        "advection",
                        self._object_name(f"{u}_conservative_convection"),
                        variable=u,
                        velocity=[-float(a) for a in alpha] + [0.0] * (3 - dimension),
                        advection_form="conservative",
                        **restrict,
                    )
                    dedicated += 1
                else:
                    for i in range(dimension):
                        _add_term(flux, i, f"-({alpha[i]})*{v}")
            # Convection: f gets -beta . grad u.
            for v, beta in self._couplings("convection_coefficient", u).items():
                beta = self._vector(beta, dimension, "convection_coefficient", u)
                if v == u and all(_number(b) for b in beta):
                    problem.add_kernel(
                        "advection",
                        self._object_name(f"{u}_convection"),
                        variable=u,
                        velocity=[float(b) for b in beta] + [0.0] * (3 - dimension),
                        advection_form="non_conservative",
                        **restrict,
                    )
                    dedicated += 1
                else:
                    source.append(
                        " - ".join(
                            [""] + [f"({beta[i]})*grad_{axes[i]}({v})" for i in range(dimension)]
                        ).strip()
                    )
            # Absorption: f gets -a u.
            for v, a in self._couplings("absorption_coefficient", u).items():
                if v == u and not _text_uses_fields(a, names):
                    if a != 0.0:
                        problem.add_kernel(
                            "reaction",
                            self._object_name(f"{u}_absorption"),
                            variable=u,
                            coefficient=a,
                            **restrict,
                        )
                        dedicated += 1
                else:
                    source.append(f"-({a})*{v}")
            # Source f.
            f = self._per_equation("source", u)
            if f is not None and f != 0.0:
                if _text_uses_fields(f, names):
                    source.append(f"({f})")
                else:
                    problem.add_kernel(
                        "body_force",
                        self._object_name(f"{u}_source"),
                        variable=u,
                        value=f,
                        scale_with_load=False,
                        **restrict,
                    )
                    dedicated += 1
            parsed_flux = flux if any(c != "0" for c in flux) else []
            if parsed_flux or source:
                problem.add_kernel(
                    "parsed_kernel",
                    self._object_name(u),
                    variable=u,
                    flux=parsed_flux,
                    source=" + ".join(source) if source else "",
                    **self._property_parameters(),
                    **restrict,
                )
            d = self._per_equation("time_derivative_coefficient", u)
            if d is not None and d != 0.0:
                if _text_uses_fields(d, names):
                    raise InputError(
                        f"coefficient_form_PDE '{self.name}': the time_derivative_coefficient of the equation of {u} must not depend on the fields or the properties. Give a constant or an expression of x, y, z and t."
                    )
                problem.add_kernel(
                    "time_derivative",
                    self._object_name(f"{u}_time_derivative"),
                    variable=u,
                    coefficient=d,
                    **restrict,
                )

    def _manufactured_terms(self, fields, coordinates, dimension):
        x, y, z, t = _symbols_of_mms()
        axes = (x, y, z)
        names = self.variable_names()
        terms = {}

        def value(expression):
            return _general_form_expression(expression, fields, self.constants)

        for u in names:
            flux = [0, 0, 0]
            source = 0
            diffusion = self._couplings("diffusion_coefficient", u)
            if not diffusion:
                raise InputError(
                    f"coefficient_form_PDE '{self.name}': a manufactured solution needs the diffusion_coefficient of the equation of {u}."
                )
            for v, c in diffusion.items():
                grad = [fields[v].diff(a) for a in axes]
                if isinstance(c, (list, tuple)):
                    tensor = self._tensor(c, dimension, u)
                    for i in range(dimension):
                        flux[i] += sum(value(tensor[i][j]) * grad[j] for j in range(dimension))
                else:
                    for i in range(dimension):
                        flux[i] += value(c) * grad[i]
            for v, alpha in self._couplings("conservative_flux_convection_coefficient", u).items():
                for i in range(dimension):
                    flux[i] += value(alpha[i]) * fields[v]
            for v, beta in self._couplings("convection_coefficient", u).items():
                source += sum(value(beta[i]) * fields[v].diff(axes[i]) for i in range(dimension))
            for v, a in self._couplings("absorption_coefficient", u).items():
                source += value(a) * fields[v]
            f = self._per_equation("source", u)
            if f is not None:
                source -= value(f)
            d = self._per_equation("time_derivative_coefficient", u)
            if d is not None:
                source += _sympy_expression(d) * fields[u].diff(t)
            terms[u] = (flux, source)
        return terms


_GRADIENT = r"grad_([xyz])\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)"


def _general_form_expression(text, fields: dict, constants: dict):
    """A flux or source expression of general_form_PDE as SymPy, with every
    field replaced by its exact solution and every grad_x(u), grad_y(u) and
    grad_z(u) by the derivative of that solution."""
    import re

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
    """A system of partial differential equations in general form: for every field u, d ∂u/∂t + ∇·Γ = f, where the flux Γ and the source f are expressions of all the fields, the components of their gradients (grad_x(u), grad_y(u), grad_z(u)), the coordinates x, y, z, the time t and named constants. The expressions are compiled and differentiated automatically, so Newton's method has the exact Jacobian, and the assembly runs on every thread and every process with every method."""

    type_name: ClassVar[str] = "general_form_PDE"
    module: ClassVar[str] = "framework"

    variables: Sequence[str] = parameter(
        description="Names of the dependent variables, one equation each. Required."
    )
    flux: Mapping[str, Sequence[str | float]] = parameter(
        default_factory=dict,
        description="The flux Γ of each equation: field name -> one expression for each space dimension. For diffusion with a coefficient c, Γ = −c ∇u, for example ['-c*grad_x(u)', '-c*grad_y(u)']. Default: no flux.",
    )
    source: Mapping[str, str | float] = parameter(
        default_factory=dict,
        description="The source f of each equation: field name -> expression. Default: no source.",
    )
    time_derivative_coefficient: Mapping[str, str | float] = parameter(
        default_factory=dict,
        description="The coefficient d of ∂u/∂t in each equation: field name -> a constant or the name of a function of (x, y, z, t). Default: no time derivative, so the equation is steady.",
    )
    constants: Mapping[str, float] = parameter(
        default_factory=dict,
        description="Named constants that the expressions use: name -> value. Default none.",
    )
    properties: Sequence[str] = parameter(
        (),
        description="Scalar properties of the property objects of the problem that the expressions use, each by its name. A property object on each block gives coefficients that differ between regions. Default none.",
    )
    units: Mapping[str, str] = parameter(
        default_factory=dict,
        description="The SI unit of each field, which the output files write with its values: field name -> unit. Default: dimensionless fields.",
    )
    initial_condition: Mapping[str, str | float] = parameter(
        default_factory=dict,
        description="The initial value of each field: field name -> a constant or the name of a function. Default 0.",
    )
    block: Sequence[str] = parameter(
        (), description="Blocks where the equations hold. Default: every block."
    )

    def __post_init__(self):
        fields = list(self.variables)
        if not fields:
            raise InputError("general_form_PDE: give the names of the fields in 'variables'.")
        if len(set(fields)) != len(fields):
            raise InputError("general_form_PDE: every name in 'variables' must be different.")
        for key in ("flux", "source", "time_derivative_coefficient", "units", "initial_condition"):
            unknown = sorted(set(getattr(self, key)) - set(fields))
            if unknown:
                raise InputError(
                    f"general_form_PDE: the {key} names {', '.join(unknown)}, which is not in 'variables' ({', '.join(fields)})."
                )
        silent = [
            f
            for f in fields
            if f not in self.flux
            and f not in self.source
            and f not in self.time_derivative_coefficient
        ]
        if silent:
            raise InputError(
                f"general_form_PDE: the field(s) {', '.join(silent)} have no flux, source or time derivative, so their equations are empty."
            )

    def variable_names(self) -> list[str]:
        return list(self.variables)

    def _variable_units(self) -> dict:
        return dict(self.units)

    def _initial_condition(self, variable: str):
        return self.initial_condition.get(variable)

    def _constant_parameters(self) -> dict:
        if not self.constants:
            return {}
        return {
            "constant_names": list(self.constants),
            "constant_values": [float(v) for v in self.constants.values()],
        }

    def _build(self, problem) -> None:
        restrict = self._restriction()
        for field in self.variables:
            flux = [str(component) for component in self.flux.get(field, ())]
            source = self.source.get(field)
            if flux or source is not None:
                problem.add_kernel(
                    "parsed_kernel",
                    self._object_name(field),
                    variable=field,
                    flux=flux,
                    source="" if source is None else str(source),
                    **self._constant_parameters(),
                    **({"coupled_properties": list(self.properties)} if self.properties else {}),
                    **restrict,
                )
            coefficient = self.time_derivative_coefficient.get(field)
            if coefficient is not None and coefficient != 0.0:
                problem.add_kernel(
                    "time_derivative",
                    self._object_name(f"{field}_time_derivative"),
                    variable=field,
                    coefficient=coefficient,
                    **restrict,
                )

    def _manufactured_terms(self, fields, coordinates, dimension):
        _x, _y, _z, t = _symbols_of_mms()
        terms = {}
        for field in self.variables:
            flux = [
                -_general_form_expression(c, fields, self.constants)
                for c in self.flux.get(field, ())
            ]
            flux += [0] * (3 - len(flux))
            source = 0
            if field in self.source:
                source -= _general_form_expression(self.source[field], fields, self.constants)
            coefficient = self.time_derivative_coefficient.get(field)
            if coefficient is not None:
                source += _sympy_expression(coefficient) * fields[field].diff(t)
            terms[field] = (flux, source)
        return terms


# ---------------------------------------------------------------------------
# Solid mechanics
# ---------------------------------------------------------------------------
FORMULATIONS = ("plane_stress", "plane_strain", "axisymmetric", "three_dimensional")


@register
@dataclass
class SolidMechanics(Physics):
    """Linear elasticity of an isotropic solid at small strain: ∇·σ + f
    = 0. The formulation is three_dimensional on a three-dimensional mesh and
    axisymmetric in axisymmetric coordinates, and is given on a
    two-dimensional Cartesian mesh (plane_stress or plane_strain)."""

    type_name: ClassVar[str] = "solid_mechanics"
    module: ClassVar[str] = "solid_mechanics"
    vector: ClassVar[bool] = True

    youngs_modulus: float | None = parameter(
        None,
        unit="Pa",
        description="Young's modulus E. Default: the property youngs_modulus, which a property object such as constant_property or parsed_property declares.",
    )
    poissons_ratio: float | None = parameter(
        None,
        description="Poisson's ratio nu, with -1 < nu < 0.5. Default: the property poissons_ratio.",
    )
    displacements: Sequence[str] | None = parameter(
        None,
        description="Names of the displacement variables, one per coordinate. Default displacement_x, displacement_y and displacement_z (displacement_r and displacement_z in axisymmetric coordinates).",
    )
    formulation: str | None = parameter(
        None,
        description="plane_stress, plane_strain, axisymmetric or three_dimensional. Default: three_dimensional on a three-dimensional mesh and axisymmetric in axisymmetric coordinates. A two-dimensional Cartesian problem must give plane_stress or plane_strain, because neither is right for every body.",
    )
    thickness: float = parameter(
        1.0,
        unit="m",
        description="Out-of-plane thickness of a plane problem, applied to the equations and to the traction and pressure conditions of this physics. Default 1 m, i.e., the equations per unit thickness.",
    )
    body_force: Sequence[float] | None = parameter(
        None,
        unit="N/m^3",
        description="Constant body force per unit volume, one value per component (e.g., the weight [0, 0, -rho g]). Default none.",
    )
    block: Sequence[str] = parameter(
        (), description="Blocks where the equations hold. Default: every block."
    )

    def __post_init__(self):
        if self.poissons_ratio is not None and not -1.0 < self.poissons_ratio < 0.5:
            raise InputError(
                f"solid_mechanics: Poisson's ratio must lie between -1 and 0.5, and {self.poissons_ratio} does not."
            )
        if self.youngs_modulus is not None and self.youngs_modulus <= 0.0:
            raise InputError("solid_mechanics: Young's modulus must be positive.")
        if (self.youngs_modulus is None) != (self.poissons_ratio is None):
            raise InputError(
                "solid_mechanics: give both youngs_modulus and poissons_ratio, or neither (both are then read from the property objects)."
            )
        if self.formulation is not None and self.formulation not in FORMULATIONS:
            raise InputError(
                f"solid_mechanics: unknown formulation '{self.formulation}'. Use {', '.join(FORMULATIONS)}."
            )

    def _attach(self, problem) -> None:
        if self.formulation is None:
            if problem.coordinates == "axisymmetric":
                self.formulation = "axisymmetric"
            elif problem.mesh.dimension == 3:
                self.formulation = "three_dimensional"
            else:
                raise InputError(
                    f"solid_mechanics '{self.name}': give formulation='plane_stress' (a thin plate loaded in its plane) or formulation='plane_strain' (a long body restrained along its axis) for a two-dimensional problem."
                )
        if self.formulation == "axisymmetric" and problem.coordinates != "axisymmetric":
            raise InputError(
                f"solid_mechanics '{self.name}': the axisymmetric formulation needs dm.Problem(mesh, coordinates='axisymmetric')."
            )
        super()._attach(problem)

    def variable_names(self) -> list[str]:
        if self.displacements is not None:
            return list(self.displacements)
        if self.formulation == "axisymmetric":
            return ["displacement_r", "displacement_z"]
        return ["displacement_x", "displacement_y", "displacement_z"][
            : 3 if self.formulation == "three_dimensional" else 2
        ]

    def _variable_units(self) -> dict:
        return dict.fromkeys(self.variable_names(), "m")

    def _build(self, problem) -> None:
        names, restrict = self.variable_names(), self._restriction()
        expansion = self._coupled.get("thermal_expansion")
        if self.youngs_modulus is not None:
            material = dict(
                displacements=names,
                formulation=self.formulation,
                youngs_modulus=self.youngs_modulus,
                poissons_ratio=self.poissons_ratio,
            )
            material.update(expansion or {})
            problem.add_property(
                "linear_elastic_stress", self._object_name("stress"), **material, **restrict
            )
        else:
            if self.formulation == "plane_stress":
                raise InputError(
                    f"solid_mechanics '{self.name}': elastic properties read from the property objects are available in plane_strain, axisymmetric and three_dimensional problems. Give youngs_modulus and poissons_ratio for plane stress."
                )
            eigenstrains = []
            if expansion is not None:
                problem.add_property(
                    "thermal_expansion_eigenstrain",
                    self._object_name("thermal_strain"),
                    eigenstrain_name=self._object_name("thermal_strain"),
                    temperature=expansion["temperature"],
                    thermal_expansion_coefficient=expansion["thermal_expansion_coefficient"],
                    stress_free_temperature=expansion["stress_free_temperature"],
                    **restrict,
                )
                eigenstrains.append(self._object_name("thermal_strain"))
            problem.add_property(
                "small_strain_stress",
                self._object_name("stress"),
                displacements=names,
                formulation=self.formulation,
                eigenstrain_names=eigenstrains,
                **restrict,
            )
        force = list(self.body_force) if self.body_force is not None else [0.0] * len(names)
        for component, variable in enumerate(names):
            problem.add_kernel(
                "stress_divergence",
                self._object_name(f"equilibrium_{variable}"),
                variable=variable,
                component=component,
                thickness=self.thickness,
                **restrict,
            )
            if force[component] != 0.0:
                problem.add_kernel(
                    "body_force",
                    self._object_name(f"body_force_{variable}"),
                    variable=variable,
                    value=self.thickness * force[component],
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
                    f"solid_mechanics '{self.name}': give formulation='plane_stress' or formulation='plane_strain' for a two-dimensional problem."
                )

    def _manufactured_terms(self, fields, coordinates, dimension):
        from .mms import _symbols

        if self.youngs_modulus is None:
            raise InputError(
                f"solid_mechanics '{self.name}': a manufactured solution needs youngs_modulus and poissons_ratio as constants."
            )
        x = _symbols()[0]
        E, nu, f = float(self.youngs_modulus), float(self.poissons_ratio), self.formulation
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
        h = float(self.thickness)
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
FLOW_FORMULATIONS = ("penalty", "pressure", "Taylor_Hood")
FLOW_TIME_INTEGRATIONS = ("monolithic", "projection")


def _is_quadratic_mesh(mesh) -> bool:
    linear = {"Edge2", "Tri3", "Quad4", "Tet4", "Hex8", "Wedge6", "Pyramid5"}
    return all(mesh.element_type(e) not in linear for e in range(mesh.num_elements))


@register
@dataclass
class IncompressibleFlow(Physics):
    r"""The steady or transient flow of an incompressible Newtonian fluid:
    ρ (v·∇)v = −∇p + μ Δv + f, ∇·v = 0. The penalty
    formulation eliminates the pressure, the pressure formulation keeps it
    with equal-order interpolation and stabilisation, and the Taylor_Hood
    formulation keeps it with a quadratic velocity and a linear pressure."""

    type_name: ClassVar[str] = "incompressible_flow"
    module: ClassVar[str] = "fluids"
    vector: ClassVar[bool] = True

    velocities: Sequence[str] = parameter(
        ("velocity_x", "velocity_y"),
        description="Names of the velocity variables, one per coordinate. Default velocity_x and velocity_y.",
    )
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
        description="penalty, pressure or Taylor_Hood. Default penalty, the formulation of Chapter 9 of Reddy's book, which every method except the cell-centred finite volume method supports. Taylor_Hood needs method='fem' and a quadratic mesh.",
    )
    penalty_parameter: float = parameter(
        1.0e8,
        description="Penalty parameter gamma of the penalty formulation, p = −γ ∇·v. Default 1e8, large enough for the velocity to be divergence-free to about 1e-8 of its gradient in non-dimensional problems.",
    )
    pressure: str = parameter(
        "pressure",
        description="Name of the pressure variable of the pressure and Taylor_Hood formulations. Default pressure.",
    )
    stabilization: bool | None = parameter(
        None,
        description="Pressure stabilisation of the mass equation of the pressure formulation. Default: on, which the equal-order interpolation needs.",
    )
    streamline_stabilization: bool | None = parameter(
        None,
        description="Streamline stabilisation (SUPG) of the momentum equations of the pressure formulation. Default: on when density > 0.",
    )
    body_force: Sequence[float | str] | None = parameter(
        None,
        unit="N/m^3",
        description="Body force per unit volume, one value per component, each a constant or an expression of (x, y, z, t). It enters the momentum equations and their streamline stabilisation. Default none.",
    )
    pressure_pin_point: Sequence[float] | None = parameter(
        None,
        unit="m",
        description="A point where the pressure is held at zero, which an enclosed flow needs because its pressure is otherwise determined up to a constant. Default none.",
    )
    mass_flux_boundaries: Sequence[str] | None = parameter(
        None,
        description="Side sets where the mass equation of the pressure formulations takes the flow through the boundary. Default: every side set.",
    )
    block: Sequence[str] = parameter((), description="Blocks of the fluid. Default: every block.")
    viscous_form: str = parameter(
        "stress",
        description="The form of the viscous term: stress (the default: mu (grad v + grad v^T), whose natural boundary condition is a zero traction) or laplacian (mu grad v, the same equations for a constant viscosity, whose natural boundary condition is the do-nothing outflow mu dv/dn - p n = 0 of the DFG benchmarks and of Nek5000). The projection time integration always takes the laplacian form.",
    )
    time_integration: str = parameter(
        "monolithic",
        description="How a transient solve advances the flow: monolithic (the default: Newton's method on the velocity and the pressure together, with the theta method of solve_transient) or projection (the splitting of Nek5000 and nekRS: a pressure Poisson equation, then one Helmholtz equation per velocity component, with BDFk and EXTk of order time_order). The projection needs formulation='pressure' and method 'fem', 'dmcdm' or 'hfvm'.",
    )
    time_order: int = parameter(
        2,
        description="The order k of the backward difference (BDFk) and of the extrapolation of the advection (EXTk) of the projection time integration: 1, 2 or 3. Default 2. The first steps take the orders below it.",
    )

    def __post_init__(self):
        if self.viscous_form not in ("stress", "laplacian"):
            raise InputError(
                f"incompressible_flow: viscous_form is stress or laplacian, not '{self.viscous_form}'."
            )
        if self.time_integration not in FLOW_TIME_INTEGRATIONS:
            raise InputError(
                f"incompressible_flow: unknown time_integration '{self.time_integration}'. Use {', '.join(FLOW_TIME_INTEGRATIONS)}."
            )
        if self.time_integration == "projection":
            if self.formulation != "pressure":
                raise InputError(
                    "incompressible_flow: the projection time integration needs formulation='pressure'."
                )
            if self.time_order not in (1, 2, 3):
                raise InputError("incompressible_flow: time_order is 1, 2 or 3.")
            if not self.density > 0:
                raise InputError(
                    "incompressible_flow: the projection time integration needs a positive density; a flow without inertia (Stokes flow) has no time derivative to integrate."
                )
        if self.formulation not in FLOW_FORMULATIONS:
            raise InputError(
                f"incompressible_flow: unknown formulation '{self.formulation}'. Use {', '.join(FLOW_FORMULATIONS)}."
            )
        if self.formulation == "Taylor_Hood":
            if self.stabilization:
                raise InputError(
                    "incompressible_flow: the Taylor-Hood element is stable without pressure stabilisation, so leave 'stabilization' unset."
                )
            if self.streamline_stabilization:
                raise InputError(
                    "incompressible_flow: streamline stabilisation is not available with the Taylor-Hood element, whose momentum residual omits the viscous term of the quadratic velocity. Use a finer mesh, or formulation='pressure'."
                )
        elif self.formulation == "penalty" and (
            self.stabilization or self.streamline_stabilization
        ):
            raise InputError(
                "incompressible_flow: 'stabilization' and 'streamline_stabilization' apply to the pressure formulations only."
            )

    @property
    def _pressure_formulation(self) -> bool:
        return self.formulation in ("pressure", "Taylor_Hood")

    def variable_names(self) -> list[str]:
        return list(self.velocities) + ([self.pressure] if self._pressure_formulation else [])

    def _variable_units(self) -> dict:
        return {**dict.fromkeys(self.velocities, "m/s"), self.pressure: "Pa"}

    def _dimension(self) -> int:
        return len(self.velocities)

    def _variable_order(self) -> dict:
        return {self.pressure: "first"} if self.formulation == "Taylor_Hood" else {}

    def _attach(self, problem) -> None:
        if self.formulation == "Taylor_Hood":
            if problem.method not in ("fem", "finite_element"):
                raise InputError(
                    f"incompressible_flow '{self.name}': the Taylor-Hood element needs the finite element method, dm.Problem(mesh, method='fem'). The other methods use formulation='pressure'."
                )
            if not _is_quadratic_mesh(problem.mesh):
                raise InputError(
                    f"incompressible_flow '{self.name}': the Taylor-Hood element needs quadratic elements (e.g., element_type='Quad9', 'Tri6', 'Tet10' or 'Hex27', or mesh.second_order())."
                )
        super()._attach(problem)

    def _build(self, problem) -> None:
        velocities, restrict = list(self.velocities), self._restriction()
        existing = set(problem._problem.object_names())
        taylor_hood = self.formulation == "Taylor_Hood"
        stabilization = (
            False if taylor_hood else (True if self.stabilization is None else self.stabilization)
        )
        supg = self.streamline_stabilization
        if supg is None:
            supg = self._pressure_formulation and stabilization and self.density != 0.0
        force = (
            [f if isinstance(f, str) else float(f) for f in self.body_force]
            if self.body_force is not None
            else [0.0] * len(velocities)
        )
        residual = {
            "velocities": velocities,
            "density": self.density,
            "dynamic_viscosity": self.dynamic_viscosity,
        }
        if any(f != 0.0 for f in force):
            residual["body_force"] = [f if isinstance(f, str) else repr(f) for f in force]
        buoyancy = self._coupled.get("Boussinesq_buoyancy")
        if buoyancy is not None:
            residual.update(
                temperature=buoyancy["temperature"],
                gravity=list(buoyancy["gravity"]),
                thermal_expansion_coefficient=buoyancy["thermal_expansion_coefficient"],
                reference_temperature=buoyancy["reference_temperature"],
                buoyancy_scale_with_load=bool(buoyancy["scale_with_load"]),
                buoyancy_density=buoyancy["density"],
            )
        for component, variable in enumerate(velocities):
            problem.add_kernel(
                "viscous_stress",
                self._object_name(f"viscous_{variable}"),
                variable=variable,
                component=component,
                velocities=velocities,
                dynamic_viscosity=self.dynamic_viscosity,
                form=self.viscous_form,
                **restrict,
            )
            if self._pressure_formulation:
                problem.add_kernel(
                    "pressure_gradient",
                    self._object_name(f"pressure_{variable}"),
                    variable=variable,
                    component=component,
                    pressure=self.pressure,
                    **restrict,
                )
                if supg:
                    problem.add_kernel(
                        "momentum_stabilization",
                        self._object_name(f"supg_{variable}"),
                        variable=variable,
                        component=component,
                        pressure=self.pressure,
                        **residual,
                        **restrict,
                    )
            else:
                problem.add_kernel(
                    "penalty_incompressibility",
                    self._object_name(f"penalty_{variable}"),
                    variable=variable,
                    component=component,
                    velocities=velocities,
                    penalty_parameter=self.penalty_parameter,
                    **restrict,
                )
            if self.density:
                problem.add_kernel(
                    "convective_inertia",
                    self._object_name(f"inertia_{variable}"),
                    variable=variable,
                    component=component,
                    velocities=velocities,
                    density=self.density,
                    **restrict,
                )
            if force[component] != 0.0:
                problem.add_kernel(
                    "body_force",
                    self._object_name(f"body_force_{variable}"),
                    variable=variable,
                    value=force[component],
                    **restrict,
                )
            # The inertia rho dv/dt of a transient solve; a steady solve leaves it out.
            if self.density:
                problem.add_kernel(
                    "time_derivative",
                    self._object_name(f"time_derivative_{variable}"),
                    variable=variable,
                    coefficient=self.density,
                    **restrict,
                )
        if self._pressure_formulation:
            problem.add_kernel(
                "mass_conservation",
                self._object_name("mass"),
                variable=self.pressure,
                stabilization=stabilization,
                **residual,
                **restrict,
            )
            boundaries = (
                list(self.mass_flux_boundaries)
                if self.mass_flux_boundaries is not None
                else list(problem.mesh.sideset_names())
            )
            if boundaries:
                problem.add_boundary_condition(
                    "mass_flux_boundary_condition",
                    self._object_name("mass_flux"),
                    variable=self.pressure,
                    boundary=boundaries,
                    velocities=velocities,
                )
            if self.pressure_pin_point is not None:
                problem.add_boundary_condition(
                    "point_Dirichlet_boundary_condition",
                    self._object_name("pressure_level"),
                    variable=self.pressure,
                    point=[float(c) for c in self.pressure_pin_point],
                    value=0.0,
                )
        else:
            problem.add_property(
                "penalty_pressure",
                self._object_name("pressure"),
                velocities=velocities,
                penalty_parameter=self.penalty_parameter,
                **restrict,
            )
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
                f"incompressible_flow '{self.name}': manufactured solutions need formulation='pressure' or 'Taylor_Hood', which keep the pressure as a variable."
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
                f"incompressible_flow '{self.name}': a manufactured solution sets the body force itself, so leave body_force unset."
            )
        self.body_force = [forcing[v] for v in self.velocities]
        return set(self.velocities)


# ---------------------------------------------------------------------------
# Beams and plates
# ---------------------------------------------------------------------------
_STRUCTURAL_FIELDS = {
    "extensional_stiffness": (
        0.0,
        "N",
        "Extensional stiffness A (E A for a homogeneous beam). Default 0.",
    ),
    "coupling_stiffness": (
        0.0,
        "N m",
        "Extension-bending coupling stiffness B, zero for a section symmetric about its mid-plane. Default 0.",
    ),
    "bending_stiffness": (
        0.0,
        "N m^2",
        "Bending stiffness D (E I for a homogeneous beam). Default 0.",
    ),
    "shear_stiffness": (
        0.0,
        "N",
        "Transverse shear stiffness S (k G A), used by the shear-deformable models. Default 0.",
    ),
    "transverse_load": (
        0.0,
        "N/m",
        "Distributed transverse load q, a constant or the name of a function. Default 0.",
    ),
    "axial_load": (
        0.0,
        "N/m",
        "Distributed axial load f, a constant or the name of a function. Default 0.",
    ),
    "foundation_modulus": (
        0.0,
        "N/m^2",
        "Modulus k of an elastic (Winkler) foundation. Default 0.",
    ),
    "von_karman": (
        False,
        "",
        "Include the von Karman nonlinearity of moderately large deflections. Default false.",
    ),
}


def _structural(key: str):
    default, unit, description = _STRUCTURAL_FIELDS[key]
    return parameter(default, unit=unit, description=description)


def _structural_parameters(physics, *, poissons_ratio: bool = False) -> dict:
    keys = list(_STRUCTURAL_FIELDS) + (["poissons_ratio"] if poissons_ratio else [])
    return {k: getattr(physics, k) for k in keys}


BEAM_MODELS = (
    "beam_Euler_Bernoulli_mixed",
    "beam_Timoshenko_mixed",
    "beam_Timoshenko_displacement",
)


@register
@dataclass
class Beam(Physics):
    """A straight beam on a one-dimensional mesh, in the mixed Euler-Bernoulli,
    the mixed Timoshenko or the displacement Timoshenko model. The mixed models
    solve for the bending moment as their third variable, the displacement
    model for the rotation, whose shear term is integrated at the element
    centre so that thin beams do not lock."""

    type_name: ClassVar[str] = "beam"
    module: ClassVar[str] = "solid_mechanics"

    model: str = parameter(
        "beam_Euler_Bernoulli_mixed",
        description="beam_Euler_Bernoulli_mixed, beam_Timoshenko_mixed or beam_Timoshenko_displacement. Default beam_Euler_Bernoulli_mixed.",
    )
    axial_displacement: str = parameter(
        "axial_displacement",
        description="Name of the axial displacement variable. Default axial_displacement.",
    )
    transverse_displacement: str = parameter(
        "deflection",
        description="Name of the transverse displacement variable. Default deflection.",
    )
    rotation: str = parameter(
        "rotation",
        description="Name of the rotation variable of the displacement model. Default rotation.",
    )
    bending_moment: str = parameter(
        "bending_moment",
        description="Name of the bending moment variable of the mixed models. Default bending_moment.",
    )
    extensional_stiffness: float = _structural("extensional_stiffness")
    coupling_stiffness: float = _structural("coupling_stiffness")
    bending_stiffness: float = _structural("bending_stiffness")
    shear_stiffness: float = _structural("shear_stiffness")
    transverse_load: float | str = _structural("transverse_load")
    axial_load: float | str = _structural("axial_load")
    foundation_modulus: float = _structural("foundation_modulus")
    von_karman: bool = _structural("von_karman")
    block: Sequence[str] = parameter((), description="Blocks of the beam. Default: every block.")

    def __post_init__(self):
        if self.model not in BEAM_MODELS:
            raise InputError(f"beam: unknown model '{self.model}'. Use {', '.join(BEAM_MODELS)}.")

    def variable_names(self) -> list[str]:
        third = (
            self.rotation if self.model == "beam_Timoshenko_displacement" else self.bending_moment
        )
        return [self.axial_displacement, self.transverse_displacement, third]

    def _variable_units(self) -> dict:
        return {
            self.axial_displacement: "m",
            self.transverse_displacement: "m",
            self.rotation: "rad",
            self.bending_moment: "N m",
        }

    def _build(self, problem) -> None:
        displacement_model = self.model == "beam_Timoshenko_displacement"
        variables = self.variable_names()
        mapping = dict(
            axial_displacement=self.axial_displacement,
            transverse_displacement=self.transverse_displacement,
        )
        mapping["rotation" if displacement_model else "bending_moment"] = variables[2]
        for variable in variables:
            options = dict(
                _structural_parameters(self), **mapping, variable=variable, **self._restriction()
            )
            if displacement_model and variable == variables[2]:
                options.update(quadrature="midpoint", reduced_integration=True)
            problem.add_kernel(self.model, self._object_name(variable), **options)


@register
@dataclass
class Plate(Physics):
    """A rectangular plate in the first-order shear deformation (Mindlin)
    theory, with five variables: two in-plane displacements, the deflection
    and two rotations. The transverse shear terms are integrated at the
    element centre, which prevents shear locking of thin plates."""

    type_name: ClassVar[str] = "plate"
    module: ClassVar[str] = "solid_mechanics"

    in_plane_displacements: Sequence[str] = parameter(
        ("displacement_x", "displacement_y"),
        description="Names of the two in-plane displacement variables. Default displacement_x and displacement_y.",
    )
    transverse_displacement: str = parameter(
        "deflection", description="Name of the deflection variable. Default deflection."
    )
    rotations: Sequence[str] = parameter(
        ("rotation_x", "rotation_y"),
        description="Names of the two rotation variables. Default rotation_x and rotation_y.",
    )
    poissons_ratio: float = parameter(
        0.3, description="Poisson's ratio nu of the plate. Default 0.3, typical of metals."
    )
    extensional_stiffness: float = _structural("extensional_stiffness")
    coupling_stiffness: float = _structural("coupling_stiffness")
    bending_stiffness: float = _structural("bending_stiffness")
    shear_stiffness: float = _structural("shear_stiffness")
    transverse_load: float | str = _structural("transverse_load")
    axial_load: float | str = _structural("axial_load")
    foundation_modulus: float = _structural("foundation_modulus")
    von_karman: bool = _structural("von_karman")
    in_plane_load_x: float | str = parameter(
        0.0, unit="N/m^2", description="Distributed in-plane load along x. Default 0."
    )
    in_plane_load_y: float | str = parameter(
        0.0, unit="N/m^2", description="Distributed in-plane load along y. Default 0."
    )
    block: Sequence[str] = parameter((), description="Blocks of the plate. Default: every block.")

    def variable_names(self) -> list[str]:
        return (
            list(self.in_plane_displacements)
            + [self.transverse_displacement]
            + list(self.rotations)
        )

    def _variable_units(self) -> dict:
        return {
            **dict.fromkeys(self.in_plane_displacements, "m"),
            self.transverse_displacement: "m",
            **dict.fromkeys(self.rotations, "rad"),
        }

    def _build(self, problem) -> None:
        options = dict(
            _structural_parameters(self, poissons_ratio=True),
            in_plane_load_x=self.in_plane_load_x,
            in_plane_load_y=self.in_plane_load_y,
            in_plane_displacements=list(self.in_plane_displacements),
            transverse_displacement=self.transverse_displacement,
            rotations=list(self.rotations),
            **self._restriction(),
        )
        for variable in self.variable_names():
            problem.add_kernel(
                "plate_first_order",
                self._object_name(f"bending_{variable}"),
                variable=variable,
                shear_treatment="exclude",
                **options,
            )
            problem.add_kernel(
                "plate_first_order",
                self._object_name(f"shear_{variable}"),
                variable=variable,
                shear_treatment="only",
                quadrature="midpoint",
                reduced_integration=True,
                **options,
            )


CIRCULAR_PLATE_THEORIES = ("first_order", "classical")


@register
@dataclass
class CircularPlate(Physics):
    """An axisymmetric circular plate on a radial mesh, in axisymmetric
    coordinates. The first_order (Mindlin) theory solves for the radial
    displacement, the deflection and the rotation, with the shear terms
    integrated at the element centre. The classical (Kirchhoff) theory is
    written in mixed form, with the radial bending moment as the third
    variable, because its fourth-order equation is outside the reach of the
    dual mesh control domain method."""

    type_name: ClassVar[str] = "circular_plate"
    module: ClassVar[str] = "solid_mechanics"

    theory: str = parameter(
        "first_order", description="first_order or classical. Default first_order."
    )
    radial_displacement: str = parameter(
        "radial_displacement",
        description="Name of the radial displacement variable. Default radial_displacement.",
    )
    transverse_displacement: str = parameter(
        "deflection", description="Name of the deflection variable. Default deflection."
    )
    rotation: str = parameter(
        "rotation",
        description="Name of the rotation variable of the first-order theory. Default rotation.",
    )
    bending_moment: str = parameter(
        "bending_moment",
        description="Name of the bending moment variable of the classical theory. Default bending_moment.",
    )
    poissons_ratio: float = parameter(
        0.3, description="Poisson's ratio nu of the plate. Default 0.3, typical of metals."
    )
    extensional_stiffness: float = _structural("extensional_stiffness")
    coupling_stiffness: float = _structural("coupling_stiffness")
    bending_stiffness: float = _structural("bending_stiffness")
    shear_stiffness: float = _structural("shear_stiffness")
    transverse_load: float | str = _structural("transverse_load")
    axial_load: float | str = _structural("axial_load")
    foundation_modulus: float = _structural("foundation_modulus")
    von_karman: bool = _structural("von_karman")
    block: Sequence[str] = parameter((), description="Blocks of the plate. Default: every block.")

    def __post_init__(self):
        if self.theory not in CIRCULAR_PLATE_THEORIES:
            raise InputError(
                f"circular_plate: unknown theory '{self.theory}'. Use first_order (shear deformable) or classical (Kirchhoff)."
            )

    def _attach(self, problem) -> None:
        if problem.coordinates != "axisymmetric":
            raise InputError(
                f"circular_plate '{self.name}': give dm.Problem(mesh, coordinates='axisymmetric'), so that every integral carries the factor 2 pi r."
            )
        super()._attach(problem)

    def variable_names(self) -> list[str]:
        third = self.bending_moment if self.theory == "classical" else self.rotation
        return [self.radial_displacement, self.transverse_displacement, third]

    def _variable_units(self) -> dict:
        return {
            self.radial_displacement: "m",
            self.transverse_displacement: "m",
            self.rotation: "rad",
            self.bending_moment: "N m/m",
        }

    def _build(self, problem) -> None:
        variables = self.variable_names()
        if self.theory == "classical":
            mapping = dict(
                radial_displacement=variables[0],
                transverse_displacement=variables[1],
                bending_moment=variables[2],
            )
            for variable in variables:
                problem.add_kernel(
                    "circular_plate_classical_mixed",
                    self._object_name(variable),
                    variable=variable,
                    **_structural_parameters(self, poissons_ratio=True),
                    **mapping,
                    **self._restriction(),
                )
            return
        mapping = dict(
            radial_displacement=variables[0],
            transverse_displacement=variables[1],
            rotation=variables[2],
        )
        for variable in variables:
            options = dict(
                _structural_parameters(self, poissons_ratio=True), **mapping, **self._restriction()
            )
            problem.add_kernel(
                "circular_plate_first_order",
                self._object_name(f"bending_{variable}"),
                variable=variable,
                shear_treatment="exclude",
                **options,
            )
            problem.add_kernel(
                "circular_plate_first_order",
                self._object_name(f"shear_{variable}"),
                variable=variable,
                shear_treatment="only",
                quadrature="midpoint",
                reduced_integration=True,
                **options,
            )


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

    heat_transfer: str = parameter(
        description="Name of the heat_transfer physics that gives the temperature."
    )
    solid_mechanics: str = parameter(
        description="Name of the solid_mechanics physics that expands."
    )
    thermal_expansion_coefficient: float = parameter(
        unit="1/K", description="Isotropic linear thermal expansion coefficient alpha."
    )
    stress_free_temperature: float = parameter(
        unit="K", description="Temperature T_0 at which the solid is free of thermal strain."
    )

    def _apply(self, problem) -> None:
        heat = self._physics(problem, "heat_transfer", HeatTransfer)
        solid = self._physics(problem, "solid_mechanics", SolidMechanics)
        solid._coupled["thermal_expansion"] = dict(
            temperature=heat.temperature,
            thermal_expansion_coefficient=self.thermal_expansion_coefficient,
            stress_free_temperature=self.stress_free_temperature,
        )


@register
@dataclass
class NonisothermalFlow(Coupling):
    """The coupling of an incompressible_flow and a heat_transfer physics: the
    flow carries the heat, rho c_p v . grad T in the energy equation, and,
    when gravity and thermal_expansion_coefficient are given, the temperature
    drives the flow by the Boussinesq buoyancy force
    f = -rho beta (T - T_0) g, which makes a natural convection problem."""

    type_name: ClassVar[str] = "nonisothermal_flow"
    module: ClassVar[str] = "fluids"

    heat_transfer: str = parameter(description="Name of the heat_transfer physics.")
    incompressible_flow: str = parameter(description="Name of the incompressible_flow physics.")
    density: float | None = parameter(
        None,
        unit="kg/m^3",
        description="Density rho of the fluid in the convective term and in the buoyancy force. Default: the density of the flow, or 1 when the flow has none (Stokes flow), the value of non-dimensional problems.",
    )
    specific_heat: float | str = parameter(
        1.0,
        unit="J/(kg K)",
        description="Specific heat c_p of the fluid. Default 1, the value of non-dimensional problems.",
    )
    gravity: Sequence[float] | None = parameter(
        None,
        unit="m/s^2",
        description="The gravitational acceleration vector g of the buoyancy force. Default none: no buoyancy.",
    )
    thermal_expansion_coefficient: float | None = parameter(
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
        if (self.gravity is None) != (self.thermal_expansion_coefficient is None):
            raise InputError(
                "nonisothermal_flow: give both gravity and thermal_expansion_coefficient for the buoyancy force, or neither."
            )

    def _density(self, flow) -> float:
        return (
            float(self.density)
            if self.density is not None
            else (float(flow.density) if flow.density else 1.0)
        )

    def _apply(self, problem) -> None:
        self._physics(problem, "heat_transfer", HeatTransfer)
        flow = self._physics(problem, "incompressible_flow", IncompressibleFlow)
        if self.gravity is not None:
            heat = problem._physics[self.heat_transfer]
            flow._coupled["Boussinesq_buoyancy"] = dict(
                temperature=heat.temperature,
                gravity=list(self.gravity),
                thermal_expansion_coefficient=self.thermal_expansion_coefficient,
                reference_temperature=self.reference_temperature,
                density=self._density(flow),
                scale_with_load=self.scale_with_load,
            )

    def _build(self, problem) -> None:
        heat = problem._physics[self.heat_transfer]
        flow = problem._physics[self.incompressible_flow]
        restrict = flow._restriction()
        problem.add_kernel(
            "heat_convection",
            f"{self.name}_convection",
            variable=heat.temperature,
            velocities=list(flow.velocities),
            density=self._density(flow),
            specific_heat=self.specific_heat,
            **restrict,
        )
        if self.gravity is None:
            return
        for component, variable in enumerate(flow.velocities):
            if component < len(self.gravity) and self.gravity[component] != 0.0:
                problem.add_kernel(
                    "Boussinesq_buoyancy",
                    f"{self.name}_buoyancy_{variable}",
                    variable=variable,
                    component=component,
                    temperature=heat.temperature,
                    gravity=list(self.gravity),
                    density=self._density(flow),
                    thermal_expansion_coefficient=self.thermal_expansion_coefficient,
                    reference_temperature=self.reference_temperature,
                    scale_with_load=self.scale_with_load,
                    **restrict,
                )
