# SPDX-License-Identifier: LGPL-2.1-or-later
"""Surface meshes: two-dimensional elements in three-dimensional space.

A surface mesh is what makes a shell: ``solid_mechanics`` on a mesh of 2-D
elements whose nodes lie in 3-D space solves a shell, flat plates included.
The generators here build the common surfaces::

    import dualmesh as dm

    # the Scordelis-Lo roof: 80 degrees of a cylinder of radius 25, 50 long
    roof = dm.generate_cylinder_surface_mesh(
        radius=25.0, length=50.0, arc_angle=80.0, num_elements=[16, 16]
    )

    # a quadrant of a hemisphere with an 18-degree hole at the top
    shell = dm.generate_spherical_shell_mesh(
        radius=10.0, polar_angle=[18, 90], azimuth=[0, 90], num_elements=[8, 8]
    )

    # a surface written as an equation, with its definitions as keywords
    ellipsoid = dm.generate_surface_mesh(
        surface="x^2/a^2 + y^2/b^2 + z^2/c^2 = 1",
        a=2.0,
        b=1.5,
        c=1.0,
        center=[0, 0, 0],
        num_elements=[16, 16],
    )

Every generator names the edges of the surface (side sets of the 2-D
elements, which are lines); an edge that closes up, at a pole or along a
360-degree seam, has no name.  Angles are in degrees; inside an equation the
trigonometric functions take radians, as everywhere in dualmesh.

The normals of the elements are oriented: away from the axis on a cylinder,
away from the centre on a sphere and on a surface seen from a centre, in the
direction of the height coordinate on a height surface z = f(x, y), and along
r_u x r_v on a parametric surface x(u, v), y(u, v), z(u, v).
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Sequence

import numpy as np

from . import _core
from ._core import Mesh
from .parameters import InputError

__all__ = [
    "generate_cylinder_surface_mesh",
    "generate_spherical_shell_mesh",
    "generate_surface_mesh",
]

_ELEMENT_TYPES = ("Quad4", "Quad8", "Quad9", "Tri3", "Tri6")
_QUADRATIC = {"Quad8": "Quad4", "Quad9": "Quad4", "Tri6": "Tri3"}
_RESERVED = {"x", "y", "z", "u", "v", "t", "pi", "e"}


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def _counts(num_elements, size: int, name: str) -> list[int]:
    if num_elements is None:
        raise InputError(f"{name}: give num_elements=[...] ({size} numbers).")
    counts = [int(n) for n in np.atleast_1d(num_elements)]
    if len(counts) != size or any(n < 1 for n in counts):
        raise InputError(
            f"{name}: num_elements must be {size} positive integers, got {num_elements}."
        )
    return counts


def _check_type(element_type: str, name: str) -> None:
    if element_type not in _ELEMENT_TYPES:
        raise InputError(
            f"{name}: element_type must be one of {', '.join(_ELEMENT_TYPES)}, not '{element_type}'."
        )


def _pair(value, default, name: str) -> tuple[float, float]:
    if value is None:
        return default
    a, b = (float(v) for v in value)
    if not b > a:
        raise InputError(f"{name} must be [from, to] with to > from, got {list(value)}.")
    return a, b


def _scale(points: np.ndarray) -> float:
    return max(float(np.ptp(points, axis=0).max()), 1e-300)


def _merge(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Merge coincident points (seams that close): returns the unique points
    and, for every input point, its index among them."""
    tolerance = 1e-9 * _scale(points)
    keys = np.round(points / tolerance).astype(np.int64)
    _, first, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    inverse = inverse.ravel()
    order = np.argsort(first)  # keep the original order of first appearance
    rank = np.empty_like(order)
    rank[order] = np.arange(len(order))
    return points[first[order]], rank[inverse]


def _promote(mesh: Mesh, element_type: str) -> Mesh:
    if element_type not in _QUADRATIC:
        return mesh
    return mesh.second_order(serendipity=element_type == "Quad8")


def _area_vectors(points: np.ndarray, cells) -> np.ndarray:
    """(t1 x t2) of every element from its corners."""
    out = []
    for element_type, nodes, _block in cells:
        p = points[np.asarray(nodes)]
        if element_type.startswith("Tri"):
            out.append(np.cross(p[1] - p[0], p[2] - p[0]))
        else:
            out.append(np.cross(p[2] - p[0], p[3] - p[1]))
    return np.array(out)


def _finish(
    points: np.ndarray,
    cells,
    edges: dict[str, list[list[int]]],
    preferred: Callable[[np.ndarray], np.ndarray] | None = None,
) -> Mesh:
    """The surface mesh from mapped points and cells (type, nodes, block):
    coincident points are merged (seams that close), the elements are
    oriented consistently by the core, which refuses a surface with one
    side, on the side of `preferred` (a direction at each element centroid;
    by default the side the elements were built with), and each named edge
    becomes a side set and a node set with the faces that remain on the
    boundary."""
    merged, index = _merge(points)
    cells = [(t, [int(index[n]) for n in nodes], b) for t, nodes, b in cells]
    mesh = Mesh(2, space_dimension=3)
    for p in merged:
        mesh.add_node([float(c) for c in p])
    for element_type, nodes, block in cells:
        mesh.add_element(element_type, nodes, block)
    if preferred is None:
        normals = _area_vectors(merged, cells)
    else:
        centroids = np.array([merged[np.asarray(n)].mean(axis=0) for _t, n, _b in cells])
        normals = preferred(centroids)
    mesh.fix_orientation(np.ascontiguousarray(normals, dtype=float))
    exterior = {frozenset(f) for f in mesh.side_nodes(mesh.exterior_sides())}
    for name, faces in edges.items():
        kept = [[int(index[n]) for n in face] for face in faces]
        kept = [f for f in kept if frozenset(f) in exterior]
        if kept:
            mesh.add_sideset_from_faces(name, kept)
            mesh.add_nodeset(name, sorted({n for f in kept for n in f}))
    return mesh


def _parameter_edges(param: Mesh, names: dict[str, str]) -> dict[str, list[list[int]]]:
    """The faces of a rectangle parameter mesh's side sets, renamed."""
    return {new: param.side_nodes(param.sideset(old)) for old, new in names.items()}


# ---------------------------------------------------------------------------
# equations and their definitions
# ---------------------------------------------------------------------------
def _split_equation(text: str, name: str) -> tuple[str, str]:
    parts = re.split(r"(?<![<>=!])=(?!=)", text)
    if len(parts) != 2:
        raise InputError(f"{name}: '{text}' must be one equation, 'left side = right side'.")
    return parts[0].strip(), parts[1].strip()


class _Scope:
    """The definitions given as keywords: numbers, or text in terms of the
    definitions before them.  Each text is compiled once; every definition
    must be used."""

    def __init__(self, definitions: dict, name: str):
        self.name = name
        self.values: dict[str, float] = {}
        self.used: set[str] = set()
        for key, value in definitions.items():
            if key in _RESERVED:
                raise InputError(
                    f"{name}: '{key}' is a coordinate, a parameter or a constant of the surface, so it cannot be defined."
                )
            if isinstance(value, str):
                self.values[key] = self.number(value, key)
            elif isinstance(value, (int, float, np.integer, np.floating)):
                self.values[key] = float(value)
            else:
                raise InputError(
                    f"{name}: the definition {key}={value!r} must be a number or text; is it a misspelt keyword?"
                )

    def function(self, text: str, coordinates: Sequence[str]) -> Callable[..., np.ndarray]:
        """`text` compiled once as a function of `coordinates`, with the
        definitions bound."""
        try:
            expression = _core.Expression(text, list(coordinates) + list(self.values))
        except Exception as error:  # the parser's message names the problem
            raise InputError(f"{self.name}: cannot read '{text}': {error}") from None
        self.used.update(n for n in expression.used_variables if n in self.values)
        constants = dict(self.values)

        def evaluate(*arrays):
            shape = np.broadcast(*arrays).shape if arrays else ()
            value = expression(**dict(zip(coordinates, arrays)), **constants)
            return np.broadcast_to(np.asarray(value, dtype=float), shape).copy()

        return evaluate

    def number(self, value, label: str) -> float:
        if value is None:
            raise InputError(f"{self.name}: give {label}.")
        if isinstance(value, str):
            return float(self.function(value, [])())
        return float(value)

    def check_all_used(self) -> None:
        unused = sorted(set(self.values) - self.used)
        if unused:
            raise InputError(f"{self.name}: the definition(s) {', '.join(unused)} are not used.")


# ---------------------------------------------------------------------------
# the cylinder
# ---------------------------------------------------------------------------
def generate_cylinder_surface_mesh(
    radius: float,
    length: float,
    arc_angle: float = 360.0,
    num_elements: Sequence[int] | None = None,
    element_type: str = "Quad4",
) -> Mesh:
    """A part of a cylinder's surface: radius ``radius``, axis along y from
    y = 0 to y = ``length``, and an arc of ``arc_angle`` degrees centred on
    the +z direction (the crown).  ``num_elements=[around the arc, along the
    axis]``.

    Edges: ``near_curved_edge`` (y = 0) and ``far_curved_edge`` (y = length),
    and, for an arc short of 360 degrees, ``left_straight_edge`` (the
    straight edge on the -x side) and ``right_straight_edge`` (on the +x
    side).  The normals point away from the axis.
    """
    name = "generate_cylinder_surface_mesh"
    n_arc, n_axis = _counts(num_elements, 2, name)
    _check_type(element_type, name)
    if not 0 < arc_angle <= 360:
        raise InputError(f"{name}: arc_angle must be in (0, 360] degrees, got {arc_angle}.")
    half = math.radians(arc_angle) / 2
    param = _core.generate_rectangle_mesh(
        list(np.linspace(-half, half, n_arc + 1)),
        list(np.linspace(0.0, float(length), n_axis + 1)),
        _QUADRATIC.get(element_type, element_type),
        "right",
    )
    param = _promote(param, element_type)
    uv = np.asarray(param.points())
    points = np.column_stack([radius * np.sin(uv[:, 0]), uv[:, 1], radius * np.cos(uv[:, 0])])
    edges = _parameter_edges(
        param,
        {
            "left": "left_straight_edge",
            "right": "right_straight_edge",
            "bottom": "near_curved_edge",
            "top": "far_curved_edge",
        },
    )
    outward = lambda x: np.column_stack([x[:, 0], np.zeros(len(x)), x[:, 2]])  # noqa: E731
    return _finish(points, param.cells(), edges, outward)


# ---------------------------------------------------------------------------
# spheres and surfaces seen from a centre
# ---------------------------------------------------------------------------
def _cap_chart(rho1: float, phi0: float, phi1: float, n_meridian: int, n_circle: int, name: str):
    """A closed cap in its polar chart: (X, Y) = rho (cos phi, sin phi), rho
    from 0 at the pole to rho1, phi from phi0 to phi1.  Each sub-sector of at
    most 90 degrees is a rhombus at the pole and two ring patches (a
    butterfly), so no element collapses at the pole."""
    span = phi1 - phi0
    k = max(1, math.ceil(span / (math.pi / 2) - 1e-9))
    if n_circle % (2 * k):
        raise InputError(
            f"{name}: a cap closed at the pole over {math.degrees(span):g} degrees is built from {k} sector(s) of two halves each, so the number of elements around the circle must be a multiple of {2 * k}, got {n_circle}."
        )
    s = n_circle // (2 * k)
    n_r = n_meridian - s
    if n_r < 1:
        raise InputError(
            f"{name}: the cap needs more elements along the meridian than {s}, the half-sector count around the circle; got {n_meridian}."
        )
    rho_s = rho1 * s / n_meridian
    points, quads = [], []

    def grid(P):  # P: (n+1, m+1, 2) points; quads in index order
        base = sum(len(p) for p in points)
        n, m = P.shape[0] - 1, P.shape[1] - 1
        points.append(P.reshape(-1, 2))
        idx = base + np.arange((n + 1) * (m + 1)).reshape(n + 1, m + 1)
        quads.append(
            np.stack([idx[:-1, :-1], idx[1:, :-1], idx[1:, 1:], idx[:-1, 1:]], -1).reshape(-1, 4)
        )

    def e(phi):
        return np.array([math.cos(phi), math.sin(phi)])

    for j in range(k):
        a = phi0 + span * j / k
        b = phi0 + span * (j + 1) / k
        m = (a + b) / 2
        A, B = rho_s * e(a), rho_s * e(b)
        C = A + B
        if np.linalg.norm(C) >= 0.95 * rho1:
            raise InputError(
                f"{name}: with {n_meridian} elements along the meridian the cap's central patch reaches its edge; give more elements along the meridian."
            )
        t = np.linspace(0, 1, s + 1)
        corner = t[:, None, None] * A + t[None, :, None] * B  # P0 = 0
        grid(corner)
        for start, end, p, q in ((A, C, a, m), (C, B, m, b)):
            inner = start + t[:, None] * (end - start)
            outer = rho1 * np.stack([np.cos(p + t * (q - p)), np.sin(p + t * (q - p))], -1)
            r = np.linspace(0, 1, n_r + 1)
            grid(inner[:, None, :] + r[None, :, None] * (outer - inner)[:, None, :])
    return np.concatenate(points), np.concatenate(quads)


def _band_chart(theta0, theta1, phi0, phi1, n_meridian, n_circle):
    theta = np.linspace(theta0, theta1, n_meridian + 1)
    phi = np.linspace(phi0, phi1, n_circle + 1)
    T, P = np.meshgrid(theta, phi, indexing="ij")
    idx = np.arange(T.size).reshape(T.shape)
    quads = np.stack([idx[:-1, :-1], idx[1:, :-1], idx[1:, 1:], idx[:-1, 1:]], -1).reshape(-1, 4)
    return np.column_stack([T.ravel(), P.ravel()]), quads


def _directions(polar_angle, azimuth, num_elements, name):
    """Unit directions of a linear mesh of a sphere's zone, with its
    elements, and the polar and azimuth limits in radians."""
    theta0, theta1 = (math.radians(a) for a in _pair(polar_angle, (0.0, 180.0), "polar_angle"))
    phi0, phi1 = (math.radians(a) for a in _pair(azimuth, (0.0, 360.0), "azimuth"))
    if theta0 < 0 or theta1 > math.pi + 1e-12:
        raise InputError(f"{name}: polar_angle must lie in [0, 180] degrees.")
    if phi1 - phi0 > 2 * math.pi + 1e-12:
        raise InputError(f"{name}: azimuth spans at most 360 degrees.")
    n_meridian, n_circle = _counts(num_elements, 2, name)
    north, south = theta0 <= 1e-12, theta1 >= math.pi - 1e-12
    pieces = []  # (points3d, quads)

    def cap(rho1, n_m, sign):
        P, Q = _cap_chart(rho1, phi0, phi1, n_m, n_circle, name)
        rho = np.hypot(P[:, 0], P[:, 1])
        phi = np.arctan2(P[:, 1], P[:, 0])
        d = np.column_stack(
            [np.sin(rho) * np.cos(phi), np.sin(rho) * np.sin(phi), sign * np.cos(rho)]
        )
        pieces.append((d, Q))

    if north and south:
        if n_meridian % 2:
            raise InputError(
                f"{name}: a zone closed at both poles is two caps meeting at the equator, so the number of elements along the meridian must be even, got {n_meridian}."
            )
        cap(math.pi / 2, n_meridian // 2, 1.0)
        cap(math.pi / 2, n_meridian // 2, -1.0)
    elif north:
        cap(theta1, n_meridian, 1.0)
    elif south:
        cap(math.pi - theta0, n_meridian, -1.0)
    else:
        TP, Q = _band_chart(theta0, theta1, phi0, phi1, n_meridian, n_circle)
        T, P = TP[:, 0], TP[:, 1]
        pieces.append(
            (np.column_stack([np.sin(T) * np.cos(P), np.sin(T) * np.sin(P), np.cos(T)]), Q)
        )
    offset, points, quads = 0, [], []
    for d, Q in pieces:
        points.append(d)
        quads.append(Q + offset)
        offset += len(d)
    return np.concatenate(points), np.concatenate(quads), (theta0, theta1, phi0, phi1)


def _zone_edges(directions, faces, limits):
    """Name the boundary faces of a sphere's zone from their unit directions."""
    theta0, theta1, phi0, phi1 = limits
    theta = np.arccos(np.clip(directions[:, 2], -1, 1))
    phi = np.arctan2(directions[:, 1], directions[:, 0])
    pole = np.hypot(directions[:, 0], directions[:, 1]) < 1e-9
    tol = 1e-7
    full_circle = phi1 - phi0 >= 2 * math.pi - 1e-12

    def on_azimuth(nodes, value):
        diff = np.angle(np.exp(1j * (phi[nodes] - value)))
        return bool(np.all(pole[nodes] | (np.abs(diff) < tol)))

    edges: dict[str, list[list[int]]] = {
        "upper_circular_edge": [],
        "lower_circular_edge": [],
        "start_meridian_edge": [],
        "end_meridian_edge": [],
    }
    for face in faces:
        nodes = np.asarray(face)
        if theta0 > 1e-12 and np.all(np.abs(theta[nodes] - theta0) < tol):
            edges["upper_circular_edge"].append(face)
        elif theta1 < math.pi - 1e-12 and np.all(np.abs(theta[nodes] - theta1) < tol):
            edges["lower_circular_edge"].append(face)
        elif not full_circle and on_azimuth(nodes, phi0):
            edges["start_meridian_edge"].append(face)
        elif not full_circle and on_azimuth(nodes, phi1):
            edges["end_meridian_edge"].append(face)
    return edges


def _centred_surface(directions, quads, limits, element_type, name, place):
    """A surface seen from a centre: the linear mesh of unit directions is
    merged, given the element type, and every node is moved to `place(d)`."""
    points, index = _merge(directions)
    quads = index[quads]
    mesh = Mesh(2, space_dimension=3)
    for p in points:
        mesh.add_node([float(c) for c in p])
    for t, nodes, block in _split_quads(points, quads, element_type):
        mesh.add_element(t, nodes, block)
    mesh = _promote(mesh, element_type)
    d = np.asarray(mesh.points())
    d = d / np.linalg.norm(d, axis=1)[:, None]
    faces = mesh.side_nodes(mesh.exterior_sides())
    edges = _zone_edges(d, faces, limits)
    x = place(d)
    return _finish(x, mesh.cells(), edges, lambda c: c - place.center)


def generate_spherical_shell_mesh(
    radius: float,
    polar_angle: Sequence[float] | None = None,
    azimuth: Sequence[float] | None = None,
    num_elements: Sequence[int] | None = None,
    center: Sequence[float] = (0.0, 0.0, 0.0),
    element_type: str = "Quad4",
) -> Mesh:
    """A zone of a sphere's surface: ``polar_angle=[from, to]`` in degrees
    from the +z axis (the pole; 90 is the equator) and ``azimuth=[from, to]``
    in degrees from +x towards +y; the whole sphere by default.
    ``num_elements=[along a meridian, around the circle]``.

    Where the zone closes at a pole, the cap is a butterfly of quadrilaterals
    (no element collapses at the pole); the number of elements around the
    circle must then be a multiple of twice the number of 90-degree sectors.

    Edges: ``upper_circular_edge`` and ``lower_circular_edge`` (constant
    polar angle; upper is nearer the pole) and ``start_meridian_edge`` and
    ``end_meridian_edge`` (constant azimuth).  An edge that closes up has no
    name.  The normals point away from the centre.
    """
    name = "generate_spherical_shell_mesh"
    _check_type(element_type, name)
    centre = np.asarray(center, dtype=float)
    directions, quads, limits = _directions(polar_angle, azimuth, num_elements, name)

    def place(d):
        return centre + radius * d

    place.center = centre
    return _centred_surface(directions, quads, limits, element_type, name, place)


def _crossing_radius(g, centre, d):
    """The radius r > 0 at which the ray centre + r d meets g = 0, by doubling
    from r = 1 outwards (or halving inwards) until g changes sign, so that no
    length scale is assumed; nan where no crossing is found."""
    g0 = np.sign(g(*centre))
    r = np.ones(len(d))
    found = np.zeros(len(d), dtype=bool)
    inside_at_one = np.sign(g(*(centre + d).T)) == g0
    # Outwards from 1 where g has not changed sign at r = 1, inwards otherwise.
    factor = np.where(inside_at_one, 2.0, 0.5)
    lo, hi = np.full(len(d), np.nan), np.full(len(d), np.nan)
    for _ in range(200):  # 2^200: far beyond any geometry
        nxt = r * factor
        crossed = (np.sign(g(*(centre + nxt[:, None] * d).T)) != g0) == inside_at_one
        new = crossed & ~found
        lo[new] = np.minimum(r, nxt)[new]
        hi[new] = np.maximum(r, nxt)[new]
        found |= crossed
        if found.all():
            break
        r = np.where(found, r, nxt)
    return lo, hi


def _ray_place(g, centre, name):
    """Move every unit direction d to the point where the ray centre + r d
    meets g = 0, checking, at the scale of the surface itself, that the ray
    meets it once."""

    def place(d):
        if g(*centre) == 0.0:
            raise InputError(f"{name}: the center lies on the surface; give a point inside it.")
        lo, hi = _crossing_radius(g, centre, d)
        if np.any(np.isnan(lo)):
            raise InputError(
                f"{name}: a ray from the center does not meet the surface; give a center inside a closed surface, or use box=[...] for a general surface."
            )
        # One crossing: g keeps its sign at the center's side up to the
        # crossing and the other sign beyond it, up to twice the crossing.
        samples = np.linspace(0.0, 2.0, 401)[1:]
        outer = hi[:, None] * samples[None, :]
        values = g(*(centre[None, None, :] + outer[..., None] * d[:, None, :]).transpose(2, 0, 1))
        changes = np.count_nonzero(np.diff(values > 0, axis=1), axis=1)
        if np.any(changes > 1):
            raise InputError(
                f"{name}: a ray from the center meets the surface more than once, so the surface cannot be built from this center; move the center, or use box=[...] for a general surface."
            )
        glo = g(*(centre + lo[:, None] * d).T)
        for _ in range(100):  # bisection to round-off
            mid = 0.5 * (lo + hi)
            gm = g(*(centre + mid[:, None] * d).T)
            same = (gm > 0) == (glo > 0)
            lo, glo, hi = np.where(same, mid, lo), np.where(same, gm, glo), np.where(same, hi, mid)
            if np.all(hi - lo <= 4 * np.finfo(float).eps * hi):
                break
        return centre + (0.5 * (lo + hi))[:, None] * d

    place.center = centre
    return place


def _gradient(g, points, scale, name):
    """The gradient of g by central differences at the scale of the surface;
    refused where it vanishes (an apex or a crossing)."""
    h = 1e-6 * scale
    grad = np.column_stack(
        [(g(*(points + h * e).T) - g(*(points - h * e).T)) / (2 * h) for e in np.eye(3)]
    )
    size = np.linalg.norm(grad, axis=1)
    if np.any(size <= 1e-10 * max(size.max(), 1e-300)):
        bad = points[np.argmin(size)]
        raise InputError(
            f"{name}: the gradient of the surface's equation vanishes near ({bad[0]:.4g}, {bad[1]:.4g}, {bad[2]:.4g}), where f(x, y, z) = c is not a smooth surface (an apex or a crossing)."
        )
    return grad


def _project(g, x, scale, name, lo=None, hi=None):
    """Newton steps along grad g onto g = 0, until every node is within
    round-off of the surface at its scale.  A node on a face of the box
    (lo, hi) moves within that face."""
    keep = np.ones_like(x)
    if lo is not None:
        tolerance = 1e-9 * scale
        keep[(np.abs(x - lo) < tolerance) | (np.abs(x - hi) < tolerance)] = 0.0
    for _ in range(50):
        grad = _gradient(g, x, scale, name) * keep
        r = g(*x.T)
        size = np.einsum("ij,ij->i", grad, grad)
        step = r / size
        x = x - step[:, None] * grad
        if np.max(np.abs(step) * np.sqrt(size)) <= 1e-12 * scale:
            return x
    raise InputError(
        f"{name}: the nodes could not be moved onto the surface (Newton's method did not converge); check that the equation is smooth near the surface."
    )


def _surface_nets(g, lo, hi, h):
    """The surface g = 0 inside the box [lo, hi] as quadrilaterals, by surface
    nets: a regular grid of spacing about h; one vertex in each cell the
    surface crosses (the mean of the crossings on the cell's edges); one
    quadrilateral for each grid edge the surface crosses, joining the four
    cells around it.  Returns the vertices, the quadrilaterals (not yet
    oriented) and which vertices lie on a face of the box."""
    n = np.maximum(1, np.ceil((hi - lo) / h - 1e-9).astype(int))
    axes = [np.linspace(lo[k], hi[k], n[k] + 1) for k in range(3)]
    X = np.stack(np.meshgrid(*axes, indexing="ij"), -1)
    G = g(X[..., 0], X[..., 1], X[..., 2])
    others = {0: (1, 2), 1: (2, 0), 2: (0, 1)}  # cyclic, so e_o1 x e_o2 = e_axis
    total = np.zeros(tuple(n) + (3,))
    count = np.zeros(tuple(n))
    edges = []
    for axis in range(3):
        a = [slice(None)] * 3
        b = [slice(None)] * 3
        a[axis], b[axis] = slice(0, -1), slice(1, None)
        ga, gb = G[tuple(a)], G[tuple(b)]
        cross = (ga < 0) != (gb < 0)
        t = np.where(cross, ga / np.where(cross, ga - gb, 1.0), 0.0)
        p = X[tuple(a)] + t[..., None] * (X[tuple(b)] - X[tuple(a)])
        o1, o2 = others[axis]
        # A cell c touches the edges whose node indices are c_d and c_d + 1
        # along the other two axes.
        for s1 in (0, 1):
            for s2 in (0, 1):
                sl = [slice(None)] * 3
                sl[o1], sl[o2] = slice(s1, s1 + n[o1]), slice(s2, s2 + n[o2])
                c = cross[tuple(sl)]
                total += np.where(c[..., None], p[tuple(sl)], 0.0)
                count += c
        edges.append(cross)
    crossed = count > 0
    vertex = -np.ones(tuple(n), dtype=np.int64)
    vertex[crossed] = np.arange(int(crossed.sum()))
    V = total[crossed] / count[crossed][:, None]
    quads = []
    for axis, cross in enumerate(edges):
        o1, o2 = others[axis]
        e = np.argwhere(cross)
        e = e[(e[:, o1] >= 1) & (e[:, o1] < n[o1]) & (e[:, o2] >= 1) & (e[:, o2] < n[o2])]

        def around(s1, s2, e=e, o1=o1, o2=o2):
            c = e.copy()
            c[:, o1] += s1 - 1
            c[:, o2] += s2 - 1
            return vertex[c[:, 0], c[:, 1], c[:, 2]]

        quads.append(np.stack([around(0, 0), around(1, 0), around(1, 1), around(0, 1)], 1))

    # Where the surface leaves the box, a net of each face (one vertex in each
    # face square the cut curve crosses, in the face plane) and a strip of
    # quadrilaterals from the first layer of cells to it close the surface
    # along the cut; where the cut curve crosses a line along which two faces
    # meet, a vertex at the crossing closes the corner.
    points = [V]
    count_vertices = len(V)

    def new_vertices(p):
        nonlocal count_vertices
        ids = count_vertices + np.arange(len(p))
        count_vertices += len(p)
        points.append(p)
        return ids

    faces = {}  # (d, layer) -> (fid over the squares (index along A, index along B), A, B)
    face_edges = []  # (d, layer, cell_layer, E, cross, ax, p)
    for d in range(3):
        A, B = [k for k in range(3) if k != d]
        for layer, cell_layer in ((0, 0), (n[d], n[d] - 1)):
            sl = [slice(None)] * 3
            sl[d] = layer
            Gf, Xf = G[tuple(sl)], X[tuple(sl)]  # shape (n_A + 1, n_B + 1)
            ftotal = np.zeros((n[A], n[B], 3))
            fcount = np.zeros((n[A], n[B]))
            for E in (A, B):
                ax = 0 if E == A else 1
                a = [slice(None)] * 2
                b = [slice(None)] * 2
                a[ax], b[ax] = slice(0, -1), slice(1, None)
                ga, gb = Gf[tuple(a)], Gf[tuple(b)]
                cross = (ga < 0) != (gb < 0)
                t = np.where(cross, ga / np.where(cross, ga - gb, 1.0), 0.0)
                p = Xf[tuple(a)] + t[..., None] * (Xf[tuple(b)] - Xf[tuple(a)])
                nF = n[B] if E == A else n[A]
                for s_ in (0, 1):
                    sq = [slice(None)] * 2
                    sq[1 - ax] = slice(s_, s_ + nF)
                    c = cross[tuple(sq)]
                    ftotal += np.where(c[..., None], p[tuple(sq)], 0.0)
                    fcount += c
                face_edges.append((d, layer, cell_layer, E, cross, ax, p))
            fid = -np.ones((n[A], n[B]), dtype=np.int64)
            fcrossed = fcount > 0
            fid[fcrossed] = new_vertices(ftotal[fcrossed] / fcount[fcrossed][:, None])
            faces[(d, layer)] = (fid, A, B)

    def square(d, layer, index3):
        """The face vertex of face (d, layer) over the cells index3 (n x 3)."""
        fid, A, B = faces[(d, layer)]
        return fid[index3[:, A], index3[:, B]]

    for d, layer, cell_layer, E, cross, ax, p in face_edges:
        A, B = faces[(d, layer)][1:]
        F = B if E == A else A
        nF = n[F]
        e = np.argwhere(cross)  # (index along A, index along B) of the edge
        if len(e) == 0:
            continue
        node_F = e[:, 1 - ax]
        # Edges inside the face: two cells of the first layer, two squares.
        inner = (node_F >= 1) & (node_F < nF)
        ei = e[inner]
        cells2 = []
        for s_ in (0, 1):
            cell = np.zeros((len(ei), 3), dtype=np.int64)
            cell[:, d] = cell_layer
            cell[:, A], cell[:, B] = ei[:, 0], ei[:, 1]
            cell[:, F] += s_ - 1
            cells2.append(cell)
        c0, c1 = (vertex[c[:, 0], c[:, 1], c[:, 2]] for c in cells2)
        f0, f1 = (square(d, layer, c) for c in cells2)
        quads.append(np.stack([c0, c1, f1, f0], 1))
        # Edges on a line where this face meets face (F, F-layer): once, from
        # the face of smaller d.
        for edge_layer, cell_F in ((0, 0), (nF, nF - 1)):
            if d > F:
                continue
            on_line = node_F == edge_layer
            el = e[on_line]
            if len(el) == 0:
                continue
            crossing = p[el[:, 0], el[:, 1]]
            ids = new_vertices(crossing)
            cell = np.zeros((len(el), 3), dtype=np.int64)
            cell[:, d] = cell_layer
            cell[:, A], cell[:, B] = el[:, 0], el[:, 1]
            cell[:, F] = cell_F
            c = vertex[cell[:, 0], cell[:, 1], cell[:, 2]]
            here = square(d, layer, cell)
            there = square(F, edge_layer, cell)
            quads.append(np.stack([c, here, ids, there], 1))
    V = np.concatenate(points)
    quads = np.concatenate(quads)
    face_vertex = np.zeros(len(V), dtype=bool)
    face_vertex[len(points[0]) :] = True
    return V, quads, face_vertex


def _edges_of(quads: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The distinct edges of quadrilaterals and how many elements share each."""
    e = np.sort(np.stack([quads, np.roll(quads, -1, 1)], -1).reshape(-1, 2), 1)
    return np.unique(e, axis=0, return_counts=True)


def _smooth_on_surface(g, V, quads, fixed, scale, name, lo, hi, iterations=20):
    """Move every free vertex halfway towards the mean of its neighbours,
    along the surface, and back onto it, to even out the elements."""
    edges, _count = _edges_of(quads)
    degree = np.bincount(edges.ravel(), minlength=len(V)).astype(float)
    free = ~fixed
    for _ in range(iterations):
        total = np.zeros_like(V)
        np.add.at(total, edges[:, 0], V[edges[:, 1]])
        np.add.at(total, edges[:, 1], V[edges[:, 0]])
        normal = _gradient(g, V, scale, name)
        normal /= np.linalg.norm(normal, axis=1)[:, None]
        step = total / degree[:, None] - V
        step -= np.einsum("ij,ij->i", step, normal)[:, None] * normal
        V = V + 0.5 * step * free[:, None]
        V = _project(g, V, scale, name, lo, hi)
    return V


def _corner_angles(points, quads):
    angles = []
    for k in range(4):
        u = points[quads[:, k - 1]] - points[quads[:, k]]
        v = points[quads[:, (k + 1) % 4]] - points[quads[:, k]]
        cosine = np.einsum("ij,ij->i", u, v) / (
            np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1)
        )
        angles.append(np.degrees(np.arccos(np.clip(cosine, -1, 1))))
    return np.array(angles)


def _split_quads(V, quads, element_type):
    """Cells (type, nodes, block) of the linear elements: the quadrilaterals,
    or each split along its shorter diagonal into two triangles."""
    if _QUADRATIC.get(element_type, element_type) != "Tri3":
        return [("Quad4", [int(i) for i in q], 0) for q in quads]
    cells = []
    for q in (list(map(int, q)) for q in quads):
        if np.linalg.norm(V[q[0]] - V[q[2]]) <= np.linalg.norm(V[q[1]] - V[q[3]]):
            cells += [("Tri3", [q[0], q[1], q[2]], 0), ("Tri3", [q[0], q[2], q[3]], 0)]
        else:
            cells += [("Tri3", [q[0], q[1], q[3]], 0), ("Tri3", [q[1], q[2], q[3]], 0)]
    return cells


def _box_surface(g, box, element_size, element_type, name):
    """Any surface g = 0 inside a box, by surface nets, smoothed along the
    surface; the cut by the box is the edge box_edge."""
    lo, hi = (np.asarray(c, dtype=float) for c in box)
    if lo.shape != (3,) or hi.shape != (3,) or np.any(hi <= lo):
        raise InputError(f"{name}: box must be [[x_min, y_min, z_min], [x_max, y_max, z_max]].")
    h = float(element_size)
    if not h > 0:
        raise InputError(f"{name}: element_size must be positive.")
    scale = float(np.max(hi - lo))
    V, quads, face_vertex = _surface_nets(g, lo, hi, h)
    if len(quads) == 0:
        raise InputError(f"{name}: the surface does not pass through the box at this element_size.")
    used = np.unique(quads)
    renumber = -np.ones(len(V), dtype=np.int64)
    renumber[used] = np.arange(len(used))
    V, face_vertex, quads = V[used], face_vertex[used], renumber[quads]
    edges, count = _edges_of(quads)
    if np.any(count > 2):
        raise InputError(
            f"{name}: the surface touches itself at this element_size ({int(np.sum(count > 2))} edges shared by more than two elements); give a smaller element_size."
        )
    fixed = face_vertex.copy()
    fixed[edges[count == 1].ravel()] = True
    V = _project(g, V, scale, name, lo, hi)
    V = _smooth_on_surface(g, V, quads, fixed, scale, name, lo, hi)
    if _corner_angles(V, quads).min() < 5.0:
        raise InputError(
            f"{name}: some elements of this surface are degenerate at this element_size; give a smaller element_size."
        )
    linear = Mesh(2, space_dimension=3)
    for p in V:
        linear.add_node([float(c) for c in p])
    for t, nodes, block in _split_quads(V, quads, element_type):
        linear.add_element(t, nodes, block)
    promoted = _promote(linear, element_type)
    x = _project(g, np.asarray(promoted.points()), scale, name, lo, hi)
    mesh = _finish(x, promoted.cells(), {}, lambda c: _gradient(g, c, scale, name))
    sides = mesh.exterior_sides()
    if sides:
        faces = mesh.side_nodes(sides)
        mesh.add_sideset_from_faces("box_edge", faces)
        mesh.add_nodeset("box_edge", sorted({i for f in faces for i in f}))
    return mesh


def generate_surface_mesh(
    surface: str | Sequence[str],
    num_elements: Sequence[int] | None = None,
    element_type: str = "Quad4",
    center: Sequence[float] | None = None,
    polar_angle: Sequence[float] | None = None,
    azimuth: Sequence[float] | None = None,
    box: Sequence[Sequence[float]] | None = None,
    element_size: float | None = None,
    x_min=None,
    x_max=None,
    y_min=None,
    y_max=None,
    z_min=None,
    z_max=None,
    u_min=None,
    u_max=None,
    v_min=None,
    v_max=None,
    **definitions,
) -> Mesh:
    """A surface written as equations, with the definitions they use given as
    keywords (numbers, or text in terms of earlier definitions).

    * A height surface, one coordinate as a function of the other two:
      ``surface="z = c*(x^2 - y^2)"`` with the bounds of the other two
      (``x_min``, ``x_max``, ``y_min``, ``y_max``) and
      ``num_elements=[n_x, n_y]``.  Edges ``x_min_edge``, ``x_max_edge``,
      ``y_min_edge``, ``y_max_edge``.  The normals point towards increasing
      height.
    * A parametric surface: ``surface=["x = ...", "y = ...", "z = ..."]`` in
      the parameters ``u`` and ``v``, with ``u_min``, ``u_max``, ``v_min``,
      ``v_max`` and ``num_elements=[n_u, n_v]``.  Edges ``u_min_edge``,
      ``u_max_edge``, ``v_min_edge``, ``v_max_edge``; seams that close are
      merged.  The normals point along r_u x r_v.
    * A surface f(x, y, z) = c seen from a centre: ``center=[...]`` with
      ``num_elements=[along a meridian, around the circle]`` gives
      structured quadrilaterals, the mesh of a sphere moved along its rays
      from the centre onto the surface.  Every ray must meet the surface
      once.  ``polar_angle`` and ``azimuth`` take a part of it, with the edge
      names of :func:`generate_spherical_shell_mesh`.  The normals point
      away from the centre.
    * Any surface f(x, y, z) = c: ``box=[[x_min, y_min, z_min], [x_max,
      y_max, z_max]]`` with ``element_size`` gives quadrilaterals of about
      that size by surface nets on a regular grid, smoothed along the
      surface (``element_type="Tri3"`` splits them).  The part inside the box
      is meshed; the edge where the surface meets the box is ``box_edge``.
      The normals point towards f > c.

    The bounds may be numbers or text (``u_max="2*pi"``).  On f(x, y, z) = c
    the gradient of f must not vanish: an apex or a crossing is refused.
    """
    name = "generate_surface_mesh"
    scope = _Scope(definitions, name)
    given = {
        key
        for key, value in dict(
            num_elements=num_elements,
            center=center,
            polar_angle=polar_angle,
            azimuth=azimuth,
            box=box,
            element_size=element_size,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            z_min=z_min,
            z_max=z_max,
            u_min=u_min,
            u_max=u_max,
            v_min=v_min,
            v_max=v_max,
        ).items()
        if value is not None
    }
    bounds = dict(
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        z_min=z_min,
        z_max=z_max,
        u_min=u_min,
        u_max=u_max,
        v_min=v_min,
        v_max=v_max,
    )
    _check_type(element_type, name)

    def only(mode: str, allowed: set[str]) -> None:
        extra = sorted(given - allowed)
        if extra:
            raise InputError(f"{name}: {', '.join(extra)} do not apply to a {mode}; remove them.")

    if not isinstance(surface, str):
        only("parametric surface", {"num_elements", "u_min", "u_max", "v_min", "v_max"})
        mesh = _parametric_surface(surface, scope, num_elements, element_type, bounds, name)
    else:
        lhs, rhs = _split_equation(surface, name)
        if center is None and box is None and lhs in ("x", "y", "z"):
            plan = [c for c in "xyz" if c != lhs]
            only(
                "height surface",
                {"num_elements"} | {f"{c}_{e}" for c in plan for e in ("min", "max")},
            )
            mesh = _height_surface(lhs, rhs, plan, scope, num_elements, element_type, bounds, name)
        else:
            left = scope.function(lhs, ["x", "y", "z"])
            right = scope.function(rhs, ["x", "y", "z"])

            def g(x, y, z):
                return left(x, y, z) - right(x, y, z)

            if (center is None) == (box is None):
                raise InputError(
                    f"{name}: for the surface '{surface}' give either center=[...] (structured quadrilaterals, for a surface every ray from the center meets once) or box=[[x_min, y_min, z_min], [x_max, y_max, z_max]] with element_size (quadrilaterals for any surface)."
                )
            if center is not None:
                only(
                    "surface seen from a center",
                    {"center", "num_elements", "polar_angle", "azimuth"},
                )
                centre = np.asarray(center, dtype=float)
                directions, quads, limits = _directions(polar_angle, azimuth, num_elements, name)
                mesh = _centred_surface(
                    directions, quads, limits, element_type, name, _ray_place(g, centre, name)
                )
                points = np.asarray(mesh.points())
                _gradient(g, points, _scale(points), name)
            else:
                only("surface cut out of a box", {"box", "element_size"})
                if element_size is None:
                    raise InputError(f"{name}: with box=[...] give element_size.")
                mesh = _box_surface(g, box, element_size, element_type, name)
    scope.check_all_used()
    return mesh


def _parametric_surface(surface, scope, num_elements, element_type, bounds, name):
    sides = dict(_split_equation(s, name) for s in surface)
    if sorted(sides) != ["x", "y", "z"]:
        raise InputError(
            f"{name}: a parametric surface is three equations 'x = ...', 'y = ...', 'z = ...'."
        )
    n_u, n_v = _counts(num_elements, 2, name)
    u0, u1, v0, v1 = (scope.number(bounds[k], k) for k in ("u_min", "u_max", "v_min", "v_max"))
    functions = {c: scope.function(sides[c], ["u", "v"]) for c in "xyz"}
    param = _core.generate_rectangle_mesh(
        list(np.linspace(u0, u1, n_u + 1)),
        list(np.linspace(v0, v1, n_v + 1)),
        _QUADRATIC.get(element_type, element_type),
        "right",
    )
    param = _promote(param, element_type)
    uv = np.asarray(param.points())
    points = np.column_stack([functions[c](uv[:, 0], uv[:, 1]) for c in "xyz"])
    edges = _parameter_edges(
        param,
        {"left": "u_min_edge", "right": "u_max_edge", "bottom": "v_min_edge", "top": "v_max_edge"},
    )
    return _finish(points, param.cells(), edges)


def _height_surface(lhs, rhs, plan, scope, num_elements, element_type, bounds, name):
    labels = [f"{c}_{e}" for c in plan for e in ("min", "max")]
    if any(bounds[k] is None for k in labels):
        raise InputError(
            f"{name}: for the height surface '{lhs} = {rhs}' give {', '.join(labels)}; for a surface seen from a centre give center=[...], or box=[...] for any surface."
        )
    n_a, n_b = _counts(num_elements, 2, name)
    a0, a1, b0, b1 = (scope.number(bounds[k], k) for k in labels)
    f = scope.function(rhs, plan)
    param = _core.generate_rectangle_mesh(
        list(np.linspace(a0, a1, n_a + 1)),
        list(np.linspace(b0, b1, n_b + 1)),
        _QUADRATIC.get(element_type, element_type),
        "right",
    )
    param = _promote(param, element_type)
    ab = np.asarray(param.points())
    points = np.zeros((len(ab), 3))
    k = "xyz".index(lhs)
    points[:, "xyz".index(plan[0])] = ab[:, 0]
    points[:, "xyz".index(plan[1])] = ab[:, 1]
    points[:, k] = f(ab[:, 0], ab[:, 1])
    edges = _parameter_edges(
        param,
        {
            "left": f"{plan[0]}_min_edge",
            "right": f"{plan[0]}_max_edge",
            "bottom": f"{plan[1]}_min_edge",
            "top": f"{plan[1]}_max_edge",
        },
    )
    up = np.eye(3)[k]
    return _finish(points, param.cells(), edges, lambda x: np.tile(up, (len(x), 1)))
