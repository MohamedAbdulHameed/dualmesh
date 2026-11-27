# SPDX-License-Identifier: LGPL-2.1-or-later
"""The continuum formulations of a solid: the material law with creep, plasticity and
viscoelasticity, the stress and its divergence, the axial strain of generalized plane strain,
phase-field fracture, the crack-tip integrals and the terms of a manufactured solution."""

from __future__ import annotations

import re

from ...model.sympy_form import coordinate_symbols, sympy_expression, sympy_gradient
from ...parameters import InputError
from ..base import label
from . import anisotropy
from .structures import STRUCTURAL

#: The named hyperelastic laws of elasticity, each at finite strain.
HYPERELASTIC = ("neo-Hookean", "Saint_Venant_Kirchhoff")


def check_material(solid) -> None:
    """The checks of the material law, creep, plasticity and the isotropic elastic constants."""
    if solid.elasticity not in ("linear", *HYPERELASTIC):
        raise InputError(
            f"solid_mechanics: elasticity is linear, neo-Hookean or Saint_Venant_Kirchhoff, not '{solid.elasticity}'."
        )
    if solid.strain_energy is not None and solid.elasticity != "linear":
        raise InputError(
            "solid_mechanics: strain_energy gives the elastic law, so it takes no elasticity."
        )
    allowed = {
        "creep": ({"rate"}, {"rate", "constants", "grain_size", "hill"}),
        "plasticity": (
            {"yield_stress"},
            {"yield_stress", "kinematic_hardening", "constants", "hill"},
        ),
    }
    for name, (needed, known) in allowed.items():
        spec = getattr(solid, name)
        if spec is None:
            continue
        unknown = sorted(set(spec) - known)
        missing = sorted(needed - set(spec))
        if unknown or missing:
            raise InputError(
                f"solid_mechanics: {name} takes {', '.join(sorted(known))}, with {', '.join(sorted(needed))} required."
            )
    kinematic = (solid.plasticity or {}).get("kinematic_hardening")
    if kinematic is not None and (set(kinematic) - {"C", "gamma"} or "C" not in kinematic):
        raise InputError(
            "solid_mechanics: kinematic_hardening takes 'C' (the moduli, Pa) and 'gamma' (the dynamic recovery, default 0)."
        )
    if solid._hyperelastic and (solid.creep or solid.plasticity):
        raise InputError(
            "solid_mechanics: creep and plasticity act with linear elasticity, and a hyperelastic law has neither."
        )
    if solid.formulation in STRUCTURAL and (
        solid.large_deformation
        or solid._hyperelastic
        or solid.creep
        or solid.plasticity
        or solid.viscoelasticity
        or solid._anisotropic
        or solid.orientation is not None
    ):
        raise InputError(
            f"solid_mechanics: the {solid.formulation} is linear elastic at small strain (with von_karman for moderate rotations); large_deformation, elasticity, creep, plasticity and anisotropy act on the continua."
        )
    if isinstance(solid.poissons_ratio, (int, float)) and not -1.0 < solid.poissons_ratio < 0.5:
        raise InputError(
            f"solid_mechanics: Poisson's ratio must lie between -1 and 0.5, and {solid.poissons_ratio} does not."
        )
    if isinstance(solid.youngs_modulus, (int, float)) and solid.youngs_modulus <= 0.0:
        raise InputError("solid_mechanics: Young's modulus must be positive.")


def material_parameters(solid) -> dict:
    """The parameters of the stress object for the material law, creep and plasticity."""
    where = label(solid.type_name, solid.name)
    parameters: dict = {}
    if solid.strain_energy is not None:
        parameters.update(stress_update="strain_energy", strain_energy=solid.strain_energy)
    elif solid.elasticity != "linear":
        parameters["stress_update"] = solid.elasticity
    rate = (solid.creep or {}).get("rate")
    yield_stress = (solid.plasticity or {}).get("yield_stress")
    if yield_stress is not None and not isinstance(yield_stress, str):
        yield_stress = repr(float(yield_stress))
    uses_temperature = any(
        text is not None and re.search(r"(?<![A-Za-z0-9_])temperature(?![A-Za-z0-9_])", text)
        for text in (rate, yield_stress)
    )
    temperature = solid.temperature
    if temperature is None and "thermal_expansion" in solid._coupled:
        temperature = solid._coupled["thermal_expansion"]["temperature"]
    if solid.creep is not None and temperature is None:
        raise InputError(
            f"{where}: creep needs the temperature: give temperature (a field or a number), or couple the solid to a heat_conduction by thermal_expansion."
        )
    if uses_temperature and temperature is None:
        raise InputError(f"{where}: the yield stress reads the temperature; give temperature.")
    if isinstance(temperature, (int, float)):
        # A fixed temperature is written into the expressions.
        value = f"({float(temperature)!r})"
        pattern = r"(?<![A-Za-z0-9_])temperature(?![A-Za-z0-9_])"
        rate = re.sub(pattern, value, rate) if rate else rate
        yield_stress = re.sub(pattern, value, yield_stress) if yield_stress else yield_stress
        temperature = None
    if solid.creep is not None:
        parameters.update(
            creep_model="parsed",
            creep_rate=rate,
            creep_constants=dict(solid.creep.get("constants", {})),
        )
        if "grain_size" in solid.creep:
            parameters["grain_size"] = float(solid.creep["grain_size"])
    if solid.plasticity is not None:
        parameters.update(
            yield_stress=yield_stress,
            plasticity_constants=dict(solid.plasticity.get("constants", {})),
        )
        kinematic = solid.plasticity.get("kinematic_hardening")
        if kinematic is not None:

            def values(x):
                return [float(v) for v in (x if isinstance(x, (list, tuple)) else [x])]

            moduli = values(kinematic["C"])
            recovery = values(kinematic.get("gamma", [0.0] * len(moduli)))
            parameters.update(kinematic_hardening_moduli=moduli, dynamic_recovery=recovery)
    if solid.viscoelasticity is not None:
        unknown = sorted(set(solid.viscoelasticity) - {"shear", "bulk"})
        if unknown:
            raise InputError(
                f"{where}: viscoelasticity takes shear and bulk branches, not {', '.join(unknown)}."
            )
        for kind in ("shear", "bulk"):
            branches = solid.viscoelasticity.get(kind, [])
            if any(len(branch) != 2 for branch in branches):
                raise InputError(
                    f"{where}: each viscoelastic branch is [modulus (Pa), relaxation time (s)]."
                )
            parameters[f"viscous_{kind}_moduli"] = [float(m) for m, _ in branches]
            parameters[f"viscous_{kind}_times"] = [float(t) for _, t in branches]
    hills = [spec["hill"] for spec in (solid.creep, solid.plasticity) if spec and "hill" in spec]
    if len(hills) == 2 and [float(v) for v in hills[0]] != [float(v) for v in hills[1]]:
        raise InputError(
            f"{where}: creep and plasticity take one Hill equivalent stress; give the same hill."
        )
    if hills:
        parameters["hill_coefficients"] = [
            float(v) for v in anisotropy.hill_matrix(solid, hills[0]).ravel()
        ]
    if solid._anisotropic:
        parameters["stiffness"] = [float(v) for v in anisotropy.global_stiffness(solid).ravel()]
    if temperature is not None:
        parameters["temperature"] = temperature
    return parameters


def build_axial_strain(solid, problem) -> None:
    """Generalized plane strain: strain_zz is one unknown of the section, whose equation is
    int sigma_zz dA = F, the axial force."""
    where = label(solid.type_name, solid.name)
    if solid.thickness is not None:
        raise InputError(
            f"{where}: generalized_plane_strain describes a long body by its section, with no thickness."
        )
    restrict = solid._restriction()
    problem.add_boundary_condition("uniform_field", variable="strain_zz")
    problem.add_kernel("out_of_plane_stress", variable="strain_zz", **restrict)
    force = solid.axial_force
    if force != 0.0:
        # The force acts on the section of this solid, which its blocks bound.
        area = problem.block_volume(restrict["block"]) if restrict else problem.domain_volume()
        value = f"({force})/{area!r}" if isinstance(force, str) else float(force) / area
        problem.add_kernel("body_force", variable="strain_zz", value=value, **restrict)


def build_continuum(solid, problem) -> None:
    """The stress object of the material law, the stress divergence and the body force of
    each displacement, the axial strain of generalized plane strain and the damage of
    phase-field fracture."""
    names = [v for v in solid.variable_names() if v.startswith("disp_")]
    restrict = solid._restriction()
    expansion = solid._coupled.get("thermal_expansion")
    eigenstrains = []
    if expansion is not None:
        problem.add_property(
            "thermal_expansion_eigenstrain",
            eigenstrain_name=solid._object_name("thermal_strain"),
            temperature=expansion["temperature"],
            linear_expansion_coefficient=expansion["linear_expansion_coefficient"],
            stress_free_temperature=expansion["stress_free_temperature"],
            **restrict,
        )
        eigenstrains.append(solid._object_name("thermal_strain"))
    # A modulus that is not given is read from the property of the same name.
    moduli = {
        k: getattr(solid, k)
        for k in ("youngs_modulus", "poissons_ratio")
        if getattr(solid, k) is not None and not solid._anisotropic
    }
    if solid._anisotropic and solid._finite:
        raise InputError(
            f"{label(solid.type_name, solid.name)}: an anisotropic material is available at small strain only."
        )
    if solid._finite and not solid._hyperelastic and solid.formulation == "plane_stress":
        raise InputError(
            f"{label(solid.type_name, solid.name)}: large deformation with linear elasticity is available in plane_strain, axisymmetric and three_dimensional problems, not in plane_stress. A hyperelastic law (elasticity neo-Hookean or Saint_Venant_Kirchhoff, or strain_energy) takes plane_stress."
        )
    problem.add_property(
        "finite_strain_stress" if solid._finite else "small_strain_stress",
        displacements=names,
        formulation=solid.formulation,
        eigenstrain_names=eigenstrains,
        **moduli,
        **material_parameters(solid),
        **(
            {"out_of_plane_strain": "strain_zz"}
            if solid.formulation == "generalized_plane_strain"
            else {}
        ),
        **({"damage": "damage"} if solid.phase_field_fracture is not None else {}),
        **restrict,
    )
    if solid.formulation == "generalized_plane_strain":
        build_axial_strain(solid, problem)
    if solid.phase_field_fracture is not None:
        spec = solid.phase_field_fracture
        if set(spec) != {"toughness", "length"}:
            raise InputError(
                f"{label(solid.type_name, solid.name)}: phase_field_fracture takes toughness (J/m^2) and length (m)."
            )
        problem.add_kernel(
            "phase_field_damage",
            variable="damage",
            toughness=float(spec["toughness"]),
            length=float(spec["length"]),
            **restrict,
        )
    force = (
        [f if isinstance(f, str) else float(f) for f in solid.body_force]
        if solid.body_force is not None
        else [0.0] * len(names)
    )
    thickness = solid._plane_thickness()
    for component, variable in enumerate(names):
        problem.add_kernel(
            "stress_divergence",
            variable=variable,
            displacements=names,
            thickness=solid._plane_thickness(),
            **({"stress_property": "first_piola_kirchhoff_stress"} if solid._finite else {}),
            **restrict,
        )
        value = force[component]
        if value != 0.0:
            problem.add_kernel(
                "body_force",
                variable=variable,
                value=f"{thickness}*({value})" if isinstance(value, str) else thickness * value,
                **restrict,
            )


def crack_tip_integrals(solid, crack_tip, radii):
    """The J-integral and the stress intensity factors K_I and K_II of a crack along x whose
    tip is at ``crack_tip``, over the ring between ``radii``."""
    where = label(solid.type_name, solid.name)
    if solid.formulation not in ("plane_strain", "plane_stress"):
        raise InputError(
            f"{where}: the crack tip integrals are available in plane_strain and plane_stress."
        )
    if not (
        isinstance(solid.youngs_modulus, (int, float))
        and isinstance(solid.poissons_ratio, (int, float))
    ):
        raise InputError(
            f"{where}: the crack tip integrals take an isotropic material of constant moduli."
        )
    inner, outer = (float(r) for r in radii)
    names = [v for v in solid.variable_names() if v.startswith("disp_")]
    return solid._problem._problem.crack_tip_integrals(
        names,
        [float(crack_tip[0]), float(crack_tip[1])],
        0.0,
        inner,
        outer,
        float(solid.youngs_modulus),
        float(solid.poissons_ratio),
        solid.formulation == "plane_strain",
    )


def manufactured_terms(solid, fields, coordinates, dimension):
    """``{variable: (flux, source)}`` of the equations of a linear elastic isotropic
    continuum, in SymPy, for the exact ``fields``."""
    if solid._structural:
        raise InputError(
            f"{label(solid.type_name, solid.name)}: a manufactured solution of the {solid.formulation} formulation is not available."
        )
    if solid.youngs_modulus is None or solid.poissons_ratio is None:
        raise InputError(
            f"{label(solid.type_name, solid.name)}: a manufactured solution needs youngs_modulus and poissons_ratio as constants."
        )
    x = coordinate_symbols()[0]
    nu = float(solid.poissons_ratio)
    E, f = float(solid.youngs_modulus), solid.formulation
    names = solid.variable_names()
    u = [fields[d] for d in names] + [0] * (3 - len(names))
    ux, uy, uz = u
    if f == "axisymmetric":
        # (rr, zz, theta-theta) in slots 0, 1, 2, and the rz shear in slot 5.
        strain = [
            ux.diff(x),
            uy.diff(coordinate_symbols()[1]),
            ux / x,
            0,
            0,
            ux.diff(coordinate_symbols()[1]) + uy.diff(x),
        ]
    else:
        gx, gy, gz = (
            sympy_gradient(ux),
            sympy_gradient(uy),
            sympy_gradient(uz) if f == "three_dimensional" else [0, 0, 0],
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
    h = solid._plane_thickness()
    force = (
        [sympy_expression(b) for b in solid.body_force]
        if solid.body_force is not None
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
