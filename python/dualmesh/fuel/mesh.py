# SPDX-License-Identifier: LGPL-2.1-or-later
"""Meshes of a fuel rod.

A light water reactor fuel rod is a stack of cylindrical pellets inside a
cladding tube, with a gas-filled gap of a few tens of micrometres between
them.  The pellets and the cladding are two separate bodies: they touch only
when the gap closes.  Each mesh below therefore has two blocks, ``fuel`` and
``clad``, that share no nodes.

Three meshes are provided, one per rod model (:attr:`RodNumerics.model`):

* :func:`radial_slice_mesh` -- a radial line through the fuel and one through
  the cladding, for one axial slice of the ``"1.5d"`` model;
* :func:`axisymmetric_rod_mesh` -- an r-z mesh (x = r, y = z) of the fuel
  stack, smeared into one column, and of the cladding;
* :func:`three_dimensional_rod_mesh` -- a hexahedral mesh with the rod axis on
  the z axis.

Side sets, in every mesh:

``fuel_outer``, ``clad_inner``, ``clad_outer``
    the pellet surface and the two cladding surfaces;
``axis`` or ``fuel_inner``
    the rod axis of a solid pellet (radial slice and r-z meshes), or the bore
    of an annular pellet (every mesh);
``fuel_bottom``, ``fuel_top``, ``clad_bottom``, ``clad_top``
    the ends (r-z and three-dimensional meshes).

Lengths are in metres; the dimensions come from a :class:`RodGeometry` and
the numbers of elements from a :class:`RodMesh`.
"""

from __future__ import annotations

import numpy as np

from ..meshing import mesh_from_arrays
from .specification import RodGeometry, RodMesh


def graded_points(start: float, end: float, num_intervals: int, grading: float) -> np.ndarray:
    """``num_intervals + 1`` points from ``start`` to ``end`` whose last
    interval is ``grading`` times the first (geometric grading; below one it
    clusters the points towards ``end``)."""
    if num_intervals < 1:
        raise ValueError("At least one element is needed in every direction.")
    if abs(grading - 1.0) < 1e-12 or num_intervals == 1:
        return np.linspace(start, end, num_intervals + 1)
    q = grading ** (1.0 / (num_intervals - 1))
    x = np.concatenate([[0.0], np.cumsum(q ** np.arange(num_intervals))])
    return start + (end - start) * x / x[-1]


#: Block ids of the rod meshes.
FUEL_BLOCK, CLAD_BLOCK, COATING_BLOCK = 0, 1, 2


def _name_blocks(mesh, coated: bool = False) -> None:
    mesh.set_block_name(FUEL_BLOCK, "fuel")
    mesh.set_block_name(CLAD_BLOCK, "clad")
    if coated:
        mesh.set_block_name(COATING_BLOCK, "coating")


def _clad_radii(geometry: RodGeometry, mesh: RodMesh) -> tuple[np.ndarray, list[int]]:
    """Radial points through the cladding wall, with the coating (if any) as
    its own layer of elements, and the block of each interval."""
    t = geometry.clad_coating_thickness
    nc = mesh.num_clad_radial_elements
    substrate = np.linspace(geometry.clad_inner_radius, geometry.clad_outer_radius - t, nc + 1)
    if t <= 0:
        return substrate, [CLAD_BLOCK] * nc
    ncoat = mesh.num_clad_coating_elements
    coating = np.linspace(geometry.clad_outer_radius - t, geometry.clad_outer_radius, ncoat + 1)
    return np.concatenate([substrate, coating[1:]]), [CLAD_BLOCK] * nc + [COATING_BLOCK] * ncoat


def _axial_points(geometry: RodGeometry, num_axial: int) -> tuple[np.ndarray, np.ndarray]:
    """Axial points of the fuel stack and of the cladding, which continues
    ``plenum_height`` above the stack with the same spacing."""
    zf = np.linspace(0.0, geometry.fuel_stack_height, num_axial + 1)
    if geometry.plenum_height <= 0:
        return zf, zf
    dz = zf[1] - zf[0]
    extra = max(1, int(round(geometry.plenum_height / dz)))
    top = geometry.fuel_stack_height + geometry.plenum_height
    return zf, np.concatenate([zf, np.linspace(zf[-1], top, extra + 1)[1:]])


def radial_slice_mesh(
    geometry: RodGeometry, mesh: RodMesh | None = None, element_type: str = "Edge2"
):
    """One axial slice: a line of ``Edge2`` (or ``Edge3``) elements through
    the fuel, from the axis (or the bore of an annular pellet) to the pellet
    surface, and a separate line through the cladding wall."""
    mesh = mesh or RodMesh()
    rf = graded_points(
        geometry.pellet_inner_radius,
        geometry.pellet_outer_radius,
        mesh.num_fuel_radial_elements,
        mesh.fuel_surface_grading,
    )
    rc, clad_blocks = _clad_radii(geometry, mesh)
    nf, nc = mesh.num_fuel_radial_elements, len(clad_blocks)
    cells = [[i, i + 1] for i in range(nf)] + [[nf + 1 + i, nf + 2 + i] for i in range(nc)]
    out = mesh_from_arrays(
        np.concatenate([rf, rc])[:, None],
        cells,
        "Edge2",
        blocks=[FUEL_BLOCK] * nf + clad_blocks,
        dimension=1,
    )
    _name_blocks(out, geometry.clad_coating_thickness > 0)
    out.add_sideset_from_faces("axis" if geometry.pellet_inner_radius == 0 else "fuel_inner", [[0]])
    out.add_sideset_from_faces("fuel_outer", [[nf]])
    out.add_sideset_from_faces("clad_inner", [[nf + 1]])
    out.add_sideset_from_faces("clad_outer", [[nf + nc + 1]])
    if element_type == "Edge2":
        return out
    if element_type == "Edge3":
        return out.second_order()
    raise ValueError(f"radial_slice_mesh: element_type must be Edge2 or Edge3, not {element_type}.")


def _structured_grid(xs, ys, offset):
    """Nodes, Quad4 cells and the node index array [y, x] of a structured
    grid, numbered from ``offset``."""
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    points = np.column_stack([X.ravel(), Y.ravel()])
    idx = np.arange(len(xs) * len(ys)).reshape(len(ys), len(xs)) + offset
    cells = np.column_stack(
        [idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel(), idx[1:, 1:].ravel(), idx[1:, :-1].ravel()]
    )
    return points, cells, idx


def _grid_edges(idx, side):
    column = {"left": idx[:, 0], "right": idx[:, -1], "bottom": idx[0, :], "top": idx[-1, :]}[side]
    return [[int(a), int(b)] for a, b in zip(column[:-1], column[1:])]


def axisymmetric_rod_mesh(
    geometry: RodGeometry,
    mesh: RodMesh | None = None,
    element_type: str = "Quad4",
    num_coolant_radial_elements: int = 0,
    coolant_outer_radius: float | None = None,
):
    """An r-z mesh (x = r, y = z) of the fuel stack as one smeared column and
    of the cladding, which extends ``plenum_height`` above the stack with the
    same axial spacing.

    With ``num_coolant_radial_elements`` > 0 the mesh also covers the coolant
    channel, as a third block ``coolant`` from the cladding outer surface to
    ``coolant_outer_radius`` sharing its nodes with the cladding (for a square
    lattice of pitch p, the annulus of the same flow area has the outer radius
    :math:`\\sqrt{p^2/\\pi}`).  ``clad_outer`` is then the fluid-solid
    interface, given as a node set, and the coolant has the side sets
    ``coolant_inlet`` (bottom), ``coolant_outlet`` (top) and
    ``coolant_outer``.
    """
    mesh = mesh or RodMesh()
    g = geometry
    rf = graded_points(
        g.pellet_inner_radius,
        g.pellet_outer_radius,
        mesh.num_fuel_radial_elements,
        mesh.fuel_surface_grading,
    )
    rc, clad_blocks = _clad_radii(g, mesh)
    nc = len(clad_blocks)  # elements through the wall, the coating included
    coated = g.clad_coating_thickness > 0
    if num_coolant_radial_elements > 0 and coated:
        raise ValueError(
            "axisymmetric_rod_mesh: a coolant mesh with a coated cladding is not supported."
        )
    if num_coolant_radial_elements > 0:
        if coolant_outer_radius is None or coolant_outer_radius <= g.clad_outer_radius:
            raise ValueError(
                "axisymmetric_rod_mesh: coolant_outer_radius must exceed clad_outer_radius."
            )
        if g.plenum_height > 0:
            raise ValueError("axisymmetric_rod_mesh: a coolant mesh needs plenum_height = 0.")
        water = np.linspace(
            g.clad_outer_radius, coolant_outer_radius, num_coolant_radial_elements + 1
        )
        rc = np.concatenate([rc, water[1:]])
    zf, zc = _axial_points(g, mesh.num_axial_elements)
    pf, cf, idf = _structured_grid(rf, zf, 0)
    pc, cc, idc = _structured_grid(rc, zc, len(pf))
    column = np.tile(np.arange(len(rc) - 1), len(zc) - 1)
    blocks = [FUEL_BLOCK] * len(cf) + [clad_blocks[c] if c < nc else 2 for c in column]
    out = mesh_from_arrays(
        np.vstack([pf, pc]), np.vstack([cf, cc]), "Quad4", blocks=blocks, dimension=2
    )
    _name_blocks(out, coated)
    out.add_sideset_from_faces(
        "axis" if g.pellet_inner_radius == 0 else "fuel_inner", _grid_edges(idf, "left")
    )
    out.add_sideset_from_faces("fuel_outer", _grid_edges(idf, "right"))
    out.add_sideset_from_faces("fuel_bottom", _grid_edges(idf, "bottom"))
    out.add_sideset_from_faces("fuel_top", _grid_edges(idf, "top"))
    out.add_sideset_from_faces("clad_inner", _grid_edges(idc, "left"))
    if num_coolant_radial_elements > 0:
        out.set_block_name(2, "coolant")
        clad, coolant = idc[:, : nc + 1], idc[:, nc:]
        out.add_sideset_from_faces("clad_bottom", _grid_edges(clad, "bottom"))
        out.add_sideset_from_faces("clad_top", _grid_edges(clad, "top"))
        out.add_sideset_from_faces("coolant_inlet", _grid_edges(coolant, "bottom"))
        out.add_sideset_from_faces("coolant_outlet", _grid_edges(coolant, "top"))
        out.add_sideset_from_faces("coolant_outer", _grid_edges(coolant, "right"))
        # The fluid-solid interface is interior to the mesh, so it is named by
        # its nodes, for boundary conditions on the flow.
        out.add_nodeset("clad_outer", [int(n) for n in idc[:, nc]])
    else:
        out.add_sideset_from_faces("clad_outer", _grid_edges(idc, "right"))
        out.add_sideset_from_faces("clad_bottom", _grid_edges(idc, "bottom"))
        out.add_sideset_from_faces("clad_top", _grid_edges(idc, "top"))
    if element_type == "Quad4":
        return out
    if element_type in ("Quad8", "Quad9"):
        return out.second_order(serendipity=element_type == "Quad8")
    raise ValueError(
        f"axisymmetric_rod_mesh: element_type must be Quad4, Quad8 or Quad9, not {element_type}."
    )


def _disk_section(radius: float, core: int, rings: int, grading: float):
    """A disk as an O-grid: a central square of ``core`` x ``core`` cells and
    ``rings`` layers of cells joining it to the circle, which avoids the
    degenerate elements of a polar grid at the centre.  Returns the points,
    the quadrilaterals and the loop of nodes on the circle."""
    half_width = 0.5 * radius / np.sqrt(2.0) * 1.2
    s = np.linspace(-half_width, half_width, core + 1)
    X, Y = np.meshgrid(s, s, indexing="xy")
    points = [np.column_stack([X.ravel(), Y.ravel()])]
    grid = np.arange((core + 1) ** 2).reshape(core + 1, core + 1)
    quads = [
        np.column_stack(
            [
                grid[:-1, :-1].ravel(),
                grid[:-1, 1:].ravel(),
                grid[1:, 1:].ravel(),
                grid[1:, :-1].ravel(),
            ]
        )
    ]
    # The boundary of the square, counter-clockwise from (a, -a).
    loop = np.array(
        list(grid[0, :]) + list(grid[1:, -1]) + list(grid[-1, -2::-1]) + list(grid[-2:0:-1, 0])
    )
    square = points[0][loop]
    angles = np.arctan2(square[:, 1], square[:, 0])
    circle = radius * np.column_stack([np.cos(angles), np.sin(angles)])
    t = graded_points(0.0, 1.0, rings, grading)
    count, previous = len(points[0]), loop
    for j in range(1, rings + 1):
        ids = np.arange(count, count + len(loop))
        count += len(loop)
        points.append((1 - t[j]) * square + t[j] * circle)
        quads.append(np.column_stack([previous, np.roll(previous, -1), np.roll(ids, -1), ids]))
        previous = ids
    return np.vstack(points), np.vstack(quads), previous


def _annulus_section(radii: np.ndarray, angles: np.ndarray, offset: int):
    """A ring of quadrilaterals on the given radii and angles.  Returns the
    points, the quadrilaterals and the inner and outer loops of nodes."""
    n = len(angles)
    points = np.vstack([np.column_stack([r * np.cos(angles), r * np.sin(angles)]) for r in radii])
    ids = np.arange(len(points)).reshape(len(radii), n) + offset
    quads = np.column_stack(
        [
            ids[:-1, :].ravel(),
            np.roll(ids[:-1, :], -1, axis=1).ravel(),
            np.roll(ids[1:, :], -1, axis=1).ravel(),
            ids[1:, :].ravel(),
        ]
    )
    return points, quads, ids[0, :], ids[-1, :]


def three_dimensional_rod_mesh(geometry: RodGeometry, mesh: RodMesh | None = None):
    """A Hex8 mesh of the fuel column and the cladding tube, with the rod axis
    on the z axis.

    A solid pellet's cross-section is an O-grid of a
    ``num_fuel_core_divisions`` squared central square and
    ``num_fuel_radial_elements`` layers out to the pellet surface; an annular
    pellet's is a ring of ``num_fuel_radial_elements`` layers.  Both have
    ``4 num_fuel_core_divisions`` divisions around, which the cladding ring
    shares."""
    mesh = mesh or RodMesh()
    g = geometry
    core = mesh.num_fuel_core_divisions
    if g.pellet_inner_radius > 0:
        angles = np.linspace(0.0, 2 * np.pi, 4 * core, endpoint=False)
        rf = graded_points(
            g.pellet_inner_radius,
            g.pellet_outer_radius,
            mesh.num_fuel_radial_elements,
            mesh.fuel_surface_grading,
        )
        fp, fq, fuel_inner_loop, fuel_loop = _annulus_section(rf, angles, 0)
    else:
        fp, fq, fuel_loop = _disk_section(
            g.pellet_outer_radius, core, mesh.num_fuel_radial_elements, mesh.fuel_surface_grading
        )
        fuel_inner_loop = None
        angles = np.arctan2(fp[fuel_loop, 1], fp[fuel_loop, 0])
    rc, clad_blocks = _clad_radii(g, mesh)
    cp, cq, clad_in, clad_out = _annulus_section(rc, angles, len(fp))
    section, quads = np.vstack([fp, cp]), np.vstack([fq, cq])
    # The quadrilaterals of the ring come layer by layer, len(angles) per layer.
    ring_blocks = np.repeat(clad_blocks, len(angles))
    block2d = np.concatenate([np.full(len(fq), FUEL_BLOCK), ring_blocks])
    num_axial = mesh.num_axial_elements
    _, z = _axial_points(g, num_axial)
    nsec = len(section)
    points = np.vstack([np.column_stack([section, np.full(nsec, zk)]) for zk in z])
    hexes, blocks = [], []
    for k in range(len(z) - 1):
        for q, b in zip(quads, block2d):
            if b == 0 and k >= num_axial:
                continue  # the plenum holds no fuel
            hexes.append(list(q + k * nsec) + list(q + (k + 1) * nsec))
            blocks.append(b)
    out = mesh_from_arrays(points, np.array(hexes), "Hex8", blocks=blocks, dimension=3)
    _name_blocks(out, g.clad_coating_thickness > 0)

    def lateral(loop, layers, inward=False):
        faces = []
        for k in layers:
            for i in range(len(loop)):
                a, b = loop[i], loop[(i + 1) % len(loop)]
                if inward:
                    a, b = b, a
                faces.append(
                    [
                        int(a + k * nsec),
                        int(b + k * nsec),
                        int(b + (k + 1) * nsec),
                        int(a + (k + 1) * nsec),
                    ]
                )
        return faces

    def cap(block, k):
        # The cladding caps include the coating, which is bonded to it.
        wanted = (CLAD_BLOCK, COATING_BLOCK) if block == CLAD_BLOCK else (block,)
        return [[int(n + k * nsec) for n in q] for q, b in zip(quads, block2d) if b in wanted]

    out.add_sideset_from_faces("fuel_outer", lateral(fuel_loop, range(num_axial)))
    if fuel_inner_loop is not None:
        out.add_sideset_from_faces("fuel_inner", lateral(fuel_inner_loop, range(num_axial), True))
    out.add_sideset_from_faces("clad_inner", lateral(clad_in, range(len(z) - 1)))
    out.add_sideset_from_faces("clad_outer", lateral(clad_out, range(len(z) - 1)))
    out.add_sideset_from_faces("fuel_bottom", cap(0, 0))
    out.add_sideset_from_faces("fuel_top", cap(0, num_axial))
    out.add_sideset_from_faces("clad_bottom", cap(1, 0))
    out.add_sideset_from_faces("clad_top", cap(1, len(z) - 1))
    return out
