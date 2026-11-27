# SPDX-License-Identifier: LGPL-2.1-or-later
"""The conditions of a solid that join two surfaces or hold a boundary along its normal:
mortar contact, cohesive zones and rollers (normal_displacement)."""

from __future__ import annotations

import numpy

from ...parameters import InputError
from ..base import boundary_names, label
from . import anisotropy
from .structures import MEMBERS


def add_contact(solid, boundary, parameters: dict):
    """Mortar contact of two surfaces: the contact pressure (and with friction the tangential
    traction) becomes a variable on the secondary surface."""
    where = f"{label(solid.type_name, solid.name)}, condition 'contact'"
    unknown = sorted(set(parameters) - {"primary", "secondary", "friction_coefficient"})
    if boundary is not None or unknown or not {"primary", "secondary"} <= set(parameters):
        raise InputError(
            f"{where}: give primary and secondary (the two surfaces) and, for friction, friction_coefficient; no boundary."
        )
    if solid._structural or solid.formulation in MEMBERS:
        raise InputError(f"{where}: contact acts between continua.")
    problem = solid._problem
    primary, secondary = parameters["primary"], parameters["secondary"]
    friction = float(parameters.get("friction_coefficient", 0.0))
    solid._contacts = getattr(solid, "_contacts", [])
    suffix = "" if not solid._contacts else f"_{len(solid._contacts)}"
    pressure = f"contact_pressure{suffix}"
    problem.add_variable(pressure, unit="Pa")
    traction = None
    if friction > 0.0:
        traction = f"contact_traction{suffix}"
        problem.add_variable(traction, unit="Pa")
    solid._contacts.append(
        {
            "primary": primary,
            "secondary": secondary,
            "pressure": pressure,
            "traction": traction,
            "friction": friction,
        }
    )
    return None


def add_cohesive_zone(solid, boundary, parameters: dict):
    """A cohesive zone between two surfaces that coincide at the start."""
    where = f"{label(solid.type_name, solid.name)}, condition 'cohesive'"
    needed = {"primary", "secondary", "strength", "toughness"}
    if boundary is not None or set(parameters) != needed:
        raise InputError(
            f"{where}: give primary and secondary (the two surfaces), strength (Pa) and toughness (J/m^2); no boundary."
        )
    if solid._structural or solid.formulation in MEMBERS:
        raise InputError(f"{where}: a cohesive zone acts between continua.")
    solid._cohesive_zones = getattr(solid, "_cohesive_zones", []) + [dict(parameters)]
    return None


def add_roller(solid, value: str, boundary, parameters: dict):
    """A prescribed displacement along the outward normal, the tangential ones free (a roller
    on an inclined or curved face): exact, with the normal force at each node of the boundary
    a variable."""
    where = f"{label(solid.type_name, solid.name)}, condition 'normal_displacement'"
    if boundary is None or parameters:
        raise InputError(
            f"{where}: give the boundary only; the value is an expression of x, y, z and t."
        )
    if solid._structural or solid.formulation in MEMBERS:
        raise InputError(f"{where}: available for continua.")
    solid._rollers = getattr(solid, "_rollers", [])
    force = f"normal_force{'' if not solid._rollers else f'_{len(solid._rollers)}'}"
    solid._problem.add_variable(force, unit="N")
    solid._rollers.append({"boundary": boundary_names(boundary), "value": value, "force": force})
    return None


def scale_modulus(solid, physical: bool = False) -> float:
    """The modulus E (Pa) of the stiffness of a condition: the Young's modulus, or the largest
    diagonal stiffness of an anisotropic material.  Where E only scales the equations of a
    condition, a modulus that varies (an expression or a property) takes 1 Pa, which leaves the
    solution unchanged.  Where E sets a physical stiffness (``physical``), the modulus must be a
    number."""
    if solid._anisotropic:
        return float(max(anisotropy.global_stiffness(solid).diagonal()[:3]))
    if isinstance(solid.youngs_modulus, (int, float)):
        return float(solid.youngs_modulus)
    if physical:
        raise InputError(
            f"{label(solid.type_name, solid.name)}: a cohesive zone is 50 E / h stiff before "
            "it opens, with E the Young's modulus, so it needs youngs_modulus as a number."
        )
    return 1.0


def mean_side_length(mesh, boundaries) -> float:
    """The mean length h (m) of the sides of the side sets ``boundaries``, the distance between
    the first two corners of each side."""
    sides = [side for boundary in boundaries for side in mesh.sideset(boundary)]
    corners = numpy.array([nodes[:2] for nodes in mesh.side_nodes(sides)])
    points = numpy.asarray(mesh.points())
    return float(numpy.linalg.norm(points[corners[:, 0]] - points[corners[:, 1]], axis=1).mean())


def surface_stiffness(solid, mesh, boundary: str, physical: bool = False) -> float:
    """E / h of a surface, with E from :func:`scale_modulus` and h the mean length of the sides
    of the surface."""
    return scale_modulus(solid, physical) / mean_side_length(mesh, [boundary])


def build_rollers(solid, problem) -> None:
    """The normal_displacement objects, with the scale of their equations E h^(d - 2) times
    the thickness, the stiffness of a node (it changes the conditioning only)."""
    names = [v for v in solid.variable_names() if v.startswith("disp_")]
    for roller in getattr(solid, "_rollers", []):
        h = mean_side_length(problem.mesh, roller["boundary"])
        scale = scale_modulus(solid) * h ** (len(names) - 2) * solid._plane_thickness()
        problem.add_boundary_condition(
            "normal_displacement",
            roller["boundary"],
            variable=roller["force"],
            displacements=names,
            value=roller["value"],
            stiffness_scale=scale,
        )


def build_interfaces(solid, problem) -> None:
    """The mortar_contact objects, with the scale of their complementarity functions E / h
    (it changes the convergence only), and the cohesive_zone objects, 50 E / h stiff."""
    names = [v for v in solid.variable_names() if v.startswith("disp_")]
    mesh = problem.mesh
    for zone in getattr(solid, "_cohesive_zones", []):
        stiffness = surface_stiffness(solid, mesh, zone["secondary"], physical=True)
        problem.add_boundary_condition(
            "cohesive_zone",
            zone["secondary"],
            variable=names[0],
            primary_boundary=zone["primary"],
            displacements=names,
            strength=float(zone["strength"]),
            toughness=float(zone["toughness"]),
            stiffness=50.0 * stiffness,
            thickness=solid._plane_thickness(),
            large_deformation=solid._finite,
        )
    for contact in getattr(solid, "_contacts", []):
        scale = surface_stiffness(solid, mesh, contact["secondary"])
        problem.add_boundary_condition(
            "mortar_contact",
            contact["secondary"],
            variable=contact["pressure"],
            primary_boundary=contact["primary"],
            displacements=names,
            friction_coefficient=contact["friction"],
            **({"tangential_traction": contact["traction"]} if contact["traction"] else {}),
            stiffness_scale=scale,
            thickness=solid._plane_thickness(),
            large_deformation=solid._finite,
        )


def contact_pressure(solid, secondary: str) -> dict:
    """The contact pressure at the nodes of the secondary surface ``secondary`` of a contact
    (:meth:`SolidMechanics.contact_pressure`)."""
    for contact in getattr(solid, "_contacts", []):
        if contact["secondary"] == secondary:
            values = solid._problem.values(contact["pressure"])
            return {n: float(values[n]) for n in solid._problem.mesh.boundary_nodes(secondary)}
    raise InputError(
        f"{label(solid.type_name, solid.name)}: no contact has the secondary surface '{secondary}'."
    )
