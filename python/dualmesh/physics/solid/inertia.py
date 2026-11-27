# SPDX-License-Identifier: LGPL-2.1-or-later
"""The inertia of a solid: the initial displacement and velocity of a transient run, the
inertia of each variable and the Rayleigh damping."""

from __future__ import annotations

from ...parameters import InputError
from ..base import label, list_items
from . import anisotropy


def check_inertia(solid) -> None:
    """The checks of the damping and the density."""
    if solid.damping is not None:
        unknown = sorted(set(solid.damping) - {"mass", "stiffness"})
        if unknown:
            raise InputError(
                f"solid_mechanics: damping takes 'mass' (a, 1/s) and 'stiffness' (b, s), not {', '.join(unknown)}."
            )
        if any(not (isinstance(v, (int, float)) and v >= 0.0) for v in solid.damping.values()):
            raise InputError("solid_mechanics: the damping coefficients are numbers, not negative.")
        if solid.density is None:
            raise InputError(
                "solid_mechanics: damping acts on a solid with inertia; give the density."
            )
    if isinstance(solid.density, (int, float)) and not solid.density > 0.0:
        raise InputError(f"solid_mechanics: the density must be positive, not {solid.density}.")


def initial_state(solid, field: str, variable: str):
    """The initial displacement or velocity of one displacement variable."""
    where = label(solid.type_name, solid.name)
    unknown = sorted(set(solid.initial_condition) - {"displacement", "velocity"})
    if unknown:
        raise InputError(
            f"{where}: the initial_condition names {', '.join(unknown)}; it takes displacement and velocity."
        )
    if solid.initial_condition and solid.density is None:
        raise InputError(
            f"{where}: an initial_condition is the start of a run with inertia; give the density."
        )
    values = solid.initial_condition.get(field)
    displacements = [v for v in solid.variable_names() if v.startswith("disp_")]
    if values is None or variable not in displacements:
        return None
    values = (
        list_items(values, len(displacements), where) if isinstance(values, str) else list(values)
    )
    if len(values) != len(displacements):
        raise InputError(
            f"{where}: the initial {field} needs {len(displacements)} values, one per displacement ({', '.join(displacements)})."
        )
    return values[displacements.index(variable)]


def inertia_factors(solid) -> dict:
    """The factor of the density in the inertia of each variable: the thickness of a plane
    problem, the area and second moment of a beam section, h and h^3/12 of a plate."""
    names = solid.variable_names()
    if not solid._structural:
        return dict.fromkeys([v for v in names if v.startswith("disp_")], solid._plane_thickness())
    if solid.formulation == "beam":
        section = solid.cross_section or {}
        if "width" in section:
            area = section["width"] * section["height"]
            second = section["width"] * section["height"] ** 3 / 12.0
        else:
            area, second = section.get("area"), section.get("second_moment")
        factors = {"disp_x": area, "disp_z": area}
        if "phi" in names:
            factors["phi"] = second
        return factors
    h = solid.thickness
    return {v: (h**3 / 12.0 if v.startswith("phi") else h) for v in names if v != "bending_moment"}


def build_inertia(solid, problem) -> None:
    """The inertia rho d2u/dt2 of every variable, and the Rayleigh damping a M + b K."""
    if solid.density is None:
        return
    restrict = solid._restriction()
    damping = solid.damping or {}

    def scaled(factor):
        if isinstance(solid.density, str):
            return f"{factor}*({solid.density})"
        return factor * float(solid.density)

    for variable, factor in inertia_factors(solid).items():
        problem.add_kernel(
            "second_time_derivative", variable=variable, coefficient=scaled(factor), **restrict
        )
        if damping.get("mass"):
            problem.add_kernel(
                "time_derivative",
                variable=variable,
                coefficient=scaled(factor * float(damping["mass"])),
                **restrict,
            )
    if damping.get("stiffness"):
        if solid.formulation == "generalized_plane_strain":
            raise InputError(
                f"{label(solid.type_name, solid.name)}: stiffness-proportional damping in generalized_plane_strain is not available."
            )
        names = [v for v in solid.variable_names() if v.startswith("disp_")]
        moduli = {
            k: getattr(solid, k)
            for k in ("youngs_modulus", "poissons_ratio")
            if getattr(solid, k) is not None
        }
        if solid._anisotropic:
            moduli = {"stiffness": [float(v) for v in anisotropy.global_stiffness(solid).ravel()]}
        for variable in names:
            problem.add_kernel(
                "stiffness_damping",
                variable=variable,
                displacements=names,
                formulation=solid.formulation,
                coefficient=float(damping["stiffness"]),
                thickness=solid._plane_thickness(),
                **moduli,
                **restrict,
            )
