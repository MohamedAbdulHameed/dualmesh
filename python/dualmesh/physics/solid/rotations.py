# SPDX-License-Identifier: LGPL-2.1-or-later
"""The rotation and the torque of a solid boundary about an axis, written as the
displacement and the traction conditions that they prescribe."""

from __future__ import annotations

import numpy

from ...parameters import InputError
from ..base import boundary_names, label
from .structures import MEMBERS


def axis(solid, where: str, parameters: dict) -> tuple[list[float], list[float]]:
    """The unit axis and the center of a rotation or a torque: about z in a plane problem,
    where the axis is not given, and given in three dimensions."""
    dimension = solid._dimension()
    if (
        solid._structural
        or solid.formulation in MEMBERS
        or solid.formulation in ("axisymmetric", "generalized_plane_strain")
    ):
        raise InputError(
            f"{where}: available in plane_strain, plane_stress and three_dimensional problems."
        )
    if dimension == 2:
        if "axis" in parameters:
            raise InputError(f"{where}: a plane problem turns about z; give the center only.")
        axis = numpy.array([0.0, 0.0, 1.0])
    else:
        if "axis" not in parameters:
            raise InputError(f"{where}: give the axis, a vector, and the center, a point on it.")
        axis = numpy.array([float(c) for c in parameters["axis"]])
        if axis.shape != (3,) or not numpy.linalg.norm(axis) > 0.0:
            raise InputError(f"{where}: the axis is a vector of three numbers, not zero.")
        axis = axis / numpy.linalg.norm(axis)
    center = numpy.array(
        [float(c) for c in parameters.get("center", [])]
        + [0.0] * (3 - len(parameters.get("center", [])))
    )
    if "center" not in parameters or len(parameters["center"]) != dimension:
        raise InputError(
            f"{where}: give the center, a point of {dimension} coordinates on the axis."
        )
    return [float(v) for v in axis], [float(v) for v in center]


def add_rotation(solid, boundary, parameters: dict):
    """A prescribed rotation of a boundary by 'angle' (rad, an expression of t) about an axis:
    u = R (x - c) - (x - c), with R the rotation of Rodrigues at finite strain, and its
    linear part u = angle k x (x - c) at small strain, where a rotation is infinitesimal."""
    where = f"{label(solid.type_name, solid.name)}, condition 'rotation'"
    if (
        boundary is None
        or "angle" not in parameters
        or set(parameters) - {"axis", "center", "angle"}
    ):
        raise InputError(
            f"{where}: give the boundary, the angle (rad), the center and, in three dimensions, the axis."
        )
    k, c = axis(solid, where, parameters)
    angle = f"({parameters['angle']})"
    r = [f"(x - {c[0]!r})", f"(y - {c[1]!r})", f"(z - {c[2]!r})"]
    cross = [
        f"({k[1]!r}*{r[2]} - {k[2]!r}*{r[1]})",
        f"({k[2]!r}*{r[0]} - {k[0]!r}*{r[2]})",
        f"({k[0]!r}*{r[1]} - {k[1]!r}*{r[0]})",
    ]
    along = f"({k[0]!r}*{r[0]} + {k[1]!r}*{r[1]} + {k[2]!r}*{r[2]})"
    names = solid.variable_names()[: solid._dimension()]
    for i, name in enumerate(names):
        if solid._finite:
            # R r - r = (cos a - 1) r + sin a (k x r) + (1 - cos a)(k . r) k.
            value = f"(cos({angle}) - 1)*{r[i]} + sin({angle})*{cross[i]} + (1 - cos({angle}))*{along}*{k[i]!r}"
        else:
            value = f"{angle}*{cross[i]}"
        solid._text_condition(f"{name} = {value}", boundary, {})
    return None


def add_torque(solid, torque: str, boundary, parameters: dict):
    """A torque T (N m, an expression of t) on a boundary about an axis, as the traction
    T (k x r) / J of the torsion of a circular shaft, J = int |k x r|^2 dA the polar moment
    of the boundary (times the thickness of a plane problem): its moment about the axis is
    T, and its resultant is zero when the axis passes through the centroid of the boundary."""
    where = f"{label(solid.type_name, solid.name)}, condition 'torque'"
    if boundary is None or set(parameters) - {"axis", "center"}:
        raise InputError(
            f"{where}: give the boundary, the center and, in three dimensions, the axis."
        )
    k, c = axis(solid, where, parameters)
    names = boundary_names(boundary)
    J = solid._problem.mesh.sideset_polar_moment(names, k, c) * solid._plane_thickness()
    if not J > 0.0:
        raise InputError(f"{where}: the boundary lies on the axis, so it cannot carry a torque.")
    r = [f"(x - {c[0]!r})", f"(y - {c[1]!r})", f"(z - {c[2]!r})"]
    cross = [
        f"({k[1]!r}*{r[2]} - {k[2]!r}*{r[1]})",
        f"({k[2]!r}*{r[0]} - {k[0]!r}*{r[2]})",
        f"({k[0]!r}*{r[1]} - {k[1]!r}*{r[0]})",
    ]
    components = [f"({torque})/{J!r}*{cross[i]}" for i in range(solid._dimension())]
    solid._text_condition(f"traction = [{', '.join(components)}]", boundary, {})
    return None
