# SPDX-License-Identifier: LGPL-2.1-or-later
"""Mesh generation and mesh file input/output.

Two ways of getting a mesh:

* **generate** one with the built-in generators (line, rectangle, box,
  annulus), including graded (non-uniform) spacing;
* **read** one written by another tool.  Reading and writing go through
  `meshio <https://github.com/nschloe/meshio>`_, so Gmsh (``.msh``),
  Exodus II (``.e``, ``.exo``), VTK/VTU, Abaqus (``.inp``), MED, and the
  other formats meshio supports are all available.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from . import _core

Mesh = _core.Mesh

#: meshio cell names accepted by dualmesh, mapped to dualmesh element types.
#: dualmesh numbers every element as VTK does, which is also meshio's
#: convention, so no reordering is needed in either direction.
_MESHIO_TO_DUALMESH = {
    "line": "Edge2",
    "triangle": "Tri3",
    "quad": "Quad4",
    "tetra": "Tet4",
    "hexahedron": "Hex8",
    "wedge": "Wedge6",
    "pyramid": "Pyramid5",
    "line3": "Edge3",
    "triangle6": "Tri6",
    "quad8": "Quad8",
    "quad9": "Quad9",
    "tetra10": "Tet10",
    "hexahedron20": "Hex20",
    "hexahedron27": "Hex27",
}
_DUALMESH_TO_MESHIO = {v: k for k, v in _MESHIO_TO_DUALMESH.items()}
_DIMENSION = {
    "Edge2": 1,
    "Edge3": 1,
    "Tri3": 2,
    "Quad4": 2,
    "Tri6": 2,
    "Quad8": 2,
    "Quad9": 2,
    "Tet4": 3,
    "Hex8": 3,
    "Wedge6": 3,
    "Pyramid5": 3,
    "Tet10": 3,
    "Hex20": 3,
    "Hex27": 3,
}
#: Element types whose nodes carry the quadratic interpolation.
_QUADRATIC = {"Edge3", "Tri6", "Quad8", "Quad9", "Tet10", "Hex20", "Hex27"}
#: The quadratic types reached by the serendipity promotion.
_SERENDIPITY = {"Quad8", "Hex20"}
#: The linear element with the same corners, for the generators, which build a
#: linear mesh first and promote it afterwards.
_CORNER_TYPE = {
    "Edge3": "Edge2",
    "Tri6": "Tri3",
    "Quad8": "Quad4",
    "Quad9": "Quad4",
    "Tet10": "Tet4",
    "Hex20": "Hex8",
    "Hex27": "Hex8",
}


def _corner_type(element_type: str) -> str:
    """The linear element a generator should build for a requested type."""
    return _CORNER_TYPE.get(element_type, element_type)


def _promote(mesh: Mesh, element_type: str) -> Mesh:
    """Promote to second order when a quadratic element type was asked for."""
    if element_type not in _QUADRATIC:
        return mesh
    return mesh.second_order(serendipity=element_type in _SERENDIPITY)


def _meshio_permutation(element_type: str):
    """The permutation between meshio's node order, which is VTK's, and
    dualmesh's.  dualmesh follows VTK for every element, so this is the
    identity; it is kept so that the reader and the writer have one place to
    change should that ever stop being true."""
    return Mesh.vtk_node_order(element_type)


def graded_coordinates(
    start: float, end: float, num_elements: int, bias: float = 1.0
) -> list[float]:
    """Node coordinates from ``start`` to ``end`` with a geometric grading.

    ``bias`` is the ratio between the lengths of successive elements: 1.0 gives
    a uniform mesh, values larger than one make the elements grow towards
    ``end``, values smaller than one refine towards ``end``.
    """
    if num_elements < 1:
        raise ValueError("num_elements must be at least 1")
    if bias <= 0:
        raise ValueError("bias must be positive")
    if bias == 1.0:
        return list(np.linspace(start, end, num_elements + 1))
    sizes = np.array([bias**i for i in range(num_elements)], dtype=float)
    sizes *= (end - start) / sizes.sum()
    return [start] + list(start + np.cumsum(sizes))


def coordinates_from_spacings(start: float, spacings: Sequence[float]) -> list[float]:
    """Node coordinates built by accumulating the given element sizes."""
    return [start] + list(start + np.cumsum(np.asarray(spacings, dtype=float)))


def _axis_coordinates(
    coordinates: Sequence[float] | None,
    minimum: float | None,
    maximum: float | None,
    num_elements: int | None,
    bias: float,
    name: str,
) -> list[float]:
    if coordinates is not None:
        if minimum is not None or maximum is not None:
            raise ValueError(
                f"Give {name}_coordinates or the bounds of {name}, not both: the coordinates hold the bounds."
            )
        return [float(c) for c in coordinates]
    if minimum is None or maximum is None or num_elements is None:
        raise ValueError(
            f"Give either {name}_coordinates or ({name}_min, {name}_max, num_{name}_elements)."
        )
    return graded_coordinates(minimum, maximum, num_elements, bias)


def _bounds(pair, minimum, maximum, name: str, where: str) -> tuple:
    """The bounds of an axis from ``x=[x_min, x_max]`` or from ``x_min`` and ``x_max``; giving
    both is refused."""
    if pair is None:
        return minimum, maximum
    if minimum is not None or maximum is not None:
        raise ValueError(
            f"{where}: give {name}=[{name}_min, {name}_max] or {name}_min and {name}_max, not both."
        )
    values = [float(v) for v in pair]
    if len(values) != 2 or not values[1] > values[0]:
        raise ValueError(
            f"{where}: {name} is the pair [{name}_min, {name}_max] with {name}_max > {name}_min, not {list(pair)}."
        )
    return values[0], values[1]


def _element_counts(num_elements, per_axis: tuple, name: str) -> tuple:
    """The element counts from ``num_elements=[n_x, n_y(, n_z)]`` or from the
    per-axis counts; giving both is refused."""
    if num_elements is None:
        return per_axis
    if any(n is not None for n in per_axis):
        raise ValueError(f"{name}: give num_elements or the per-axis counts, not both.")
    counts = [int(n) for n in num_elements]
    if len(counts) != len(per_axis):
        raise ValueError(f"{name}: num_elements needs {len(per_axis)} counts, got {len(counts)}.")
    return tuple(counts)


def generate_line_mesh(
    start: float | None = None,
    end: float | None = None,
    num_elements: int | None = None,
    bias: float = 1.0,
    coordinates: Sequence[float] | None = None,
    element_type: str = "Edge2",
) -> Mesh:
    """One-dimensional mesh of ``Edge2`` or ``Edge3`` elements.

    Side sets and node sets ``"left"`` and ``"right"`` are created
    automatically.  ``num_elements`` always counts elements, so an ``Edge3``
    mesh with ``num_elements=4`` has four elements and nine nodes.
    """
    x = _axis_coordinates(coordinates, start, end, num_elements, bias, "x")
    return _promote(_core.generate_line_mesh(x), element_type)


def generate_rectangle_mesh(
    x_min: float | None = None,
    x_max: float | None = None,
    y_min: float | None = None,
    y_max: float | None = None,
    num_x_elements: int | None = None,
    num_y_elements: int | None = None,
    element_type: str = "Quad4",
    diagonal: str = "right",
    x_bias: float = 1.0,
    y_bias: float = 1.0,
    x_coordinates: Sequence[float] | None = None,
    y_coordinates: Sequence[float] | None = None,
    num_elements: Sequence[int] | None = None,
    x: Sequence[float] | None = None,
    y: Sequence[float] | None = None,
) -> Mesh:
    """Structured two-dimensional mesh of ``Quad4``, ``Tri3``, ``Quad8``,
    ``Quad9`` or ``Tri6`` elements.  ``Quad8`` works with ``method="FEM"`` only.

    Side sets ``"left"``, ``"right"``, ``"bottom"`` and ``"top"`` are created
    automatically.  Pass ``x_coordinates``/``y_coordinates`` for a fully
    non-uniform grid, or the bounds (``x=[x_min, x_max]`` and ``y=[y_min,
    y_max]``, or ``x_min``, ``x_max``, ``y_min`` and ``y_max``) together with
    the element counts
    (``num_elements=[n_x, n_y]``, or ``num_x_elements`` and
    ``num_y_elements``) and an optional geometric ``bias``.  The element counts always count elements, so
    asking for a quadratic type gives the same number of elements and more
    nodes, not fewer elements.
    """
    where = "generate_rectangle_mesh"
    num_x_elements, num_y_elements = _element_counts(
        num_elements, (num_x_elements, num_y_elements), where
    )
    x_min, x_max = _bounds(x, x_min, x_max, "x", where)
    y_min, y_max = _bounds(y, y_min, y_max, "y", where)
    x = _axis_coordinates(x_coordinates, x_min, x_max, num_x_elements, x_bias, "x")
    y = _axis_coordinates(y_coordinates, y_min, y_max, num_y_elements, y_bias, "y")
    linear = _core.generate_rectangle_mesh(x, y, _corner_type(element_type), diagonal)
    return _promote(linear, element_type)


def generate_box_mesh(
    x_min: float | None = None,
    x_max: float | None = None,
    y_min: float | None = None,
    y_max: float | None = None,
    z_min: float | None = None,
    z_max: float | None = None,
    num_x_elements: int | None = None,
    num_y_elements: int | None = None,
    num_z_elements: int | None = None,
    element_type: str = "Hex8",
    x_bias: float = 1.0,
    y_bias: float = 1.0,
    z_bias: float = 1.0,
    x_coordinates: Sequence[float] | None = None,
    y_coordinates: Sequence[float] | None = None,
    z_coordinates: Sequence[float] | None = None,
    num_elements: Sequence[int] | None = None,
    x: Sequence[float] | None = None,
    y: Sequence[float] | None = None,
    z: Sequence[float] | None = None,
) -> Mesh:
    """Structured three-dimensional mesh of ``Hex8``, ``Tet4``, ``Wedge6``,
    ``Pyramid5``, ``Hex20``, ``Hex27`` or ``Tet10`` elements.  A ``Tet4`` mesh
    has six tetrahedra per cell, a ``Wedge6`` mesh two prisms split on the base
    diagonal, and a ``Pyramid5`` mesh six pyramids meeting at an added centre
    node.  ``Pyramid5`` and ``Hex20`` work with ``method="FEM"``
    only.

    Side sets ``"left"``, ``"right"``, ``"bottom"``, ``"top"``, ``"back"`` and
    ``"front"`` are created automatically.  The bounds are ``x=[x_min,
    x_max]``, ``y`` and ``z``, or ``x_min``, ``x_max`` and so on, and the
    element counts are ``num_elements=[n_x, n_y, n_z]``, or
    ``num_x_elements``, ``num_y_elements`` and ``num_z_elements``.
    """
    where = "generate_box_mesh"
    num_x_elements, num_y_elements, num_z_elements = _element_counts(
        num_elements, (num_x_elements, num_y_elements, num_z_elements), where
    )
    x_min, x_max = _bounds(x, x_min, x_max, "x", where)
    y_min, y_max = _bounds(y, y_min, y_max, "y", where)
    z_min, z_max = _bounds(z, z_min, z_max, "z", where)
    x = _axis_coordinates(x_coordinates, x_min, x_max, num_x_elements, x_bias, "x")
    y = _axis_coordinates(y_coordinates, y_min, y_max, num_y_elements, y_bias, "y")
    z = _axis_coordinates(z_coordinates, z_min, z_max, num_z_elements, z_bias, "z")
    linear = _core.generate_box_mesh(x, y, z, _corner_type(element_type))
    return _promote(linear, element_type)


def annulus_coordinates(
    inner_radius: float, outer_radius: float, num_elements: int, bias: float = 1.0
) -> list[float]:
    """Radial node coordinates for an annulus (a convenience alias)."""
    return graded_coordinates(inner_radius, outer_radius, num_elements, bias)


def generate_annulus_mesh(
    inner_radius: float,
    outer_radius: float,
    num_radial_elements: int,
    num_angular_elements: int,
    start_angle: float = 0.0,
    end_angle: float = 90.0,
    element_type: str = "Quad4",
    radial_bias: float = 1.0,
    radial_coordinates: Sequence[float] | None = None,
) -> Mesh:
    """Mesh of an annular sector (angles in degrees, measured from the x axis).

    Side sets: ``"inner"`` (r = inner_radius), ``"outer"`` (r = outer_radius),
    ``"start"`` (the start_angle edge) and ``"end"`` (the end_angle edge).
    This is the mesh used for thick pressurized cylinders and for the classical
    plate-with-a-hole problem.
    """
    radii = (
        [float(r) for r in radial_coordinates]
        if radial_coordinates is not None
        else graded_coordinates(inner_radius, outer_radius, num_radial_elements, radial_bias)
    )
    angle_start, angle_end = np.radians(start_angle), np.radians(end_angle)
    angles = np.linspace(angle_start, angle_end, num_angular_elements + 1)
    mesh = _core.generate_rectangle_mesh(radii, list(angles), _corner_type(element_type), "right")
    # The promotion happens while the mesh is still a rectangle in the (r,
    # theta) plane, so that the added mid-side nodes are mapped onto the true
    # circle by the transformation below rather than onto the straight chord
    # between their corners.  This is what makes a quadratic annulus mesh
    # isoparametric: the element edges follow the boundary.
    mesh = _promote(mesh, element_type)
    mesh.transform_nodes(lambda r, theta, z: [r * np.cos(theta), r * np.sin(theta), 0.0])
    mesh.fix_orientation()

    # The rectangle generator produced the side sets in (r, theta) space;
    # give them their geometric names.
    for source, name in (
        ("left", "inner"),
        ("right", "outer"),
        ("bottom", "start"),
        ("top", "end"),
    ):
        mesh.alias_sideset(source, name)
    return mesh


def _space_dimension(points, dim: int) -> int:
    """The dimension of the space the nodes occupy: the element dimension,
    raised when the nodes spread along a further axis by more than round-off
    at the size of the mesh (2-D elements whose z varies form a surface in
    3-D space, a shell).  A coordinate that is constant along a further axis,
    zero or not, keeps the mesh in its plane; ``dimension=3`` places a flat
    plate in space."""
    points = np.atleast_2d(np.asarray(points, dtype=float))
    size = float(np.ptp(points, axis=0).max()) if len(points) else 0.0
    space = dim
    for axis in range(dim, min(3, points.shape[1])):
        if np.ptp(points[:, axis]) > 1e-12 * size:
            space = axis + 1
    return space


def mesh_from_arrays(
    points, cells, element_type: str | None = None, blocks=None, dimension: int | None = None
) -> Mesh:
    """Build a mesh from a node array and a connectivity array.

    ``points`` has shape ``(num_nodes, 1..3)``.  ``cells`` is either an array of
    shape ``(num_elements, nodes_per_element)`` with a single ``element_type``,
    a list of ``(element_type, connectivity)`` pairs, or the
    ``(element_type, nodes, block)`` triples of :meth:`Mesh.cells`, one per
    element, which carry the blocks themselves.  ``blocks`` optionally gives
    one subdomain id per element.

    The mesh lies in the space its coordinates use: 2-D elements whose z
    varies form a surface in 3-D space (a shell), 1-D elements whose y varies
    a curve in the plane.  A constant coordinate, zero or not, keeps the mesh
    in its plane.  ``dimension`` raises the space dimension
    explicitly, for instance ``dimension=3`` for a flat plate meant as a
    shell.
    """
    points = np.atleast_2d(np.asarray(points, dtype=float))
    if (
        isinstance(cells, (list, tuple))
        and cells
        and isinstance(cells[0], (tuple, list))
        and len(cells[0]) == 3
        and isinstance(cells[0][0], str)
    ):
        # The triples of Mesh.cells(): one element each, with its block.
        if blocks is not None:
            raise ValueError(
                "mesh_from_arrays: the (element_type, nodes, block) triples carry the blocks, so blocks is not given with them."
            )
        blocks = [int(block) for _, _, block in cells]
        cells = [(element_type_name, [list(nodes)]) for element_type_name, nodes, _ in cells]
    if isinstance(cells, np.ndarray) or (cells and not isinstance(cells[0], (tuple, list))):
        if element_type is None:
            raise ValueError("Give element_type when cells is a plain connectivity array.")
        cell_groups = [(element_type, np.asarray(cells, dtype=np.int64))]
    elif cells and isinstance(cells[0], (tuple, list)) and isinstance(cells[0][0], str):
        cell_groups = [(t, np.asarray(c, dtype=np.int64)) for t, c in cells]
    else:
        if element_type is None:
            raise ValueError("Give element_type when cells is a plain connectivity array.")
        cell_groups = [(element_type, np.asarray(cells, dtype=np.int64))]

    dim = max(_DIMENSION[t] for t, _ in cell_groups)
    space = _space_dimension(points, dim)
    if dimension is not None:
        if not dim <= dimension <= 3:
            raise ValueError(
                f"mesh_from_arrays: dimension (the space the nodes live in) must be between the element dimension {dim} and 3, got {dimension}."
            )
        space = max(space, int(dimension))
    mesh = Mesh(dim, space_dimension=space)
    for p in points:
        coordinates = list(p) + [0.0] * (3 - len(p))
        mesh.add_node([float(coordinates[0]), float(coordinates[1]), float(coordinates[2])])
    index = 0
    for element_type_name, connectivity in cell_groups:
        for row in connectivity:
            block = 0 if blocks is None else int(np.asarray(blocks).ravel()[index])
            mesh.add_element(element_type_name, [int(i) for i in row], block)
            index += 1
    mesh.fix_orientation()
    return mesh


def read_mesh(
    filename: str,
    file_format: str | None = None,
    boundary_names: dict | None = None,
    add_bounding_box_sidesets: bool = False,
) -> Mesh:
    """Read a mesh file through meshio and convert it to a dualmesh mesh.

    Physical groups (Gmsh) or element blocks (Exodus) become subdomain ids, and
    lower-dimensional cell groups become side sets named after their physical
    name (or ``"boundary_<id>"`` when they are unnamed).  ``boundary_names``
    can rename them, for example ``{"boundary_1": "inlet"}``.
    """
    import meshio  # imported lazily so the core package has no hard dependency

    m = meshio.read(filename, file_format=file_format)
    volume_types = {}
    for block in m.cells:
        if block.type in _MESHIO_TO_DUALMESH:
            volume_types[block.type] = _DIMENSION[_MESHIO_TO_DUALMESH[block.type]]
    if not volume_types:
        raise ValueError(
            f"{filename}: no supported cells found. dualmesh supports "
            + ", ".join(sorted(_MESHIO_TO_DUALMESH))
        )
    dim = max(volume_types.values())

    mesh = Mesh(dim, space_dimension=_space_dimension(m.points, dim))
    for p in m.points:
        coordinates = list(p) + [0.0] * (3 - len(p))
        mesh.add_node([float(coordinates[0]), float(coordinates[1]), float(coordinates[2])])

    # physical/material tags, if present
    tag_arrays = None
    for key in ("gmsh:physical", "medit:ref", "cell_tags", "material"):
        if key in m.cell_data:
            tag_arrays = m.cell_data[key]
            break

    faces_by_tag: dict[int, list[list[int]]] = {}
    for block_index, block in enumerate(m.cells):
        if block.type not in _MESHIO_TO_DUALMESH:
            continue
        element_type = _MESHIO_TO_DUALMESH[block.type]
        tags = None
        if tag_arrays is not None and block_index < len(tag_arrays):
            tags = np.asarray(tag_arrays[block_index]).ravel()
        permutation = _meshio_permutation(element_type)
        if _DIMENSION[element_type] == dim:
            for row_index, row in enumerate(block.data):
                block_id = int(tags[row_index]) if tags is not None else 0
                mesh.add_element(element_type, [int(row[k]) for k in permutation], block_id)
        elif _DIMENSION[element_type] == dim - 1:
            for row_index, row in enumerate(block.data):
                tag = int(tags[row_index]) if tags is not None else 0
                # A side is identified by its node set, so its order is
                # immaterial here.
                faces_by_tag.setdefault(tag, []).append([int(i) for i in row])
    mesh.fix_orientation()

    # Gmsh numbers its physical groups per dimension, so a name is found by
    # its tag and its dimension: the element blocks are named from the groups
    # of the mesh's dimension and the side sets from those one below.
    physical_names = {}
    for name, data in getattr(m, "field_data", {}).items():
        data = np.atleast_1d(data)
        group_dimension = int(data[1]) if len(data) > 1 else None
        physical_names[(int(data[0]), group_dimension)] = name

    def physical_name(tag: int, dimension: int):
        return physical_names.get((tag, dimension), physical_names.get((tag, None)))

    for block_id in mesh.block_ids():
        block_name = physical_name(block_id, dim)
        if block_name is not None:
            mesh.set_block_name(block_id, block_name)
    for tag, faces in faces_by_tag.items():
        name = physical_name(tag, dim - 1) or f"boundary_{tag}"
        if boundary_names:
            name = boundary_names.get(name, name)
        mesh.add_sideset_from_faces(name, faces)
        mesh.add_nodeset(name, sorted({n for face in faces for n in face}))
    if add_bounding_box_sidesets or not faces_by_tag:
        mesh.add_bounding_box_sidesets()
    return mesh


def write_mesh(mesh: Mesh, filename: str, file_format: str | None = None, **point_data) -> None:
    """Write a mesh (and optional nodal fields) through meshio."""
    import meshio

    cells: dict[str, list[list[int]]] = {}
    for element_type, connectivity, _ in mesh.cells():
        permutation = _meshio_permutation(element_type)
        cells.setdefault(_DUALMESH_TO_MESHIO[element_type], []).append(
            [int(connectivity[k]) for k in permutation]
        )
    blocks: list[int] = [block for _, _, block in mesh.cells()]
    meshio_mesh = meshio.Mesh(
        points=np.asarray(mesh.points()),
        cells=[(t, np.asarray(c, dtype=np.int64)) for t, c in cells.items()],
        point_data={k: np.asarray(v) for k, v in point_data.items()},
        cell_data={"block": [np.asarray(blocks, dtype=np.int32)]},
    )
    meshio.write(filename, meshio_mesh, file_format=file_format)


def sideset_summary(mesh: Mesh) -> str:
    """A table of the side sets of a mesh: for each one the number of faces, the
    number of nodes, the measure (length in two dimensions, area in three, in the
    units of the coordinates) and the range of the coordinates of its nodes.
    Printing it is the first check of the boundaries of an imported mesh."""
    points = np.asarray(mesh.points())
    unit = {1: "", 2: "length", 3: "area"}[mesh.dimension]
    rows = []
    for name in mesh.sideset_names():
        sides = mesh.sideset(name)
        nodes = sorted({n for face in mesh.side_nodes(sides) for n in face})
        lo, hi = points[nodes].min(axis=0), points[nodes].max(axis=0)
        extent = "  ".join(
            f"{'xyz'[k]} {lo[k]:.4g} to {hi[k]:.4g}" for k in range(max(mesh.space_dimension, 1))
        )
        measure = mesh.sideset_measure([name]) if mesh.dimension > 1 else float("nan")
        rows.append((name, len(sides), len(nodes), measure, extent))
    if not rows:
        return "The mesh has no side sets."
    width = max(len("side set"), *(len(r[0]) for r in rows))
    header = (
        f"{'side set':<{width}}  {'faces':>7}  {'nodes':>7}  {unit:>11}  coordinates of the nodes"
    )
    lines = [header, "-" * len(header)]
    for name, faces, nodes, measure, extent in rows:
        lines.append(f"{name:<{width}}  {faces:>7d}  {nodes:>7d}  {measure:>11.5g}  {extent}")
    return "\n".join(lines)


#: VTK cell types of the linear boundary faces, by number of corners.
_MESHIO_FACE_TYPES = {1: "vertex", 2: "line", 3: "triangle", 4: "quad"}


def _face_corners(dimension: int, num_nodes: int) -> int:
    """The number of corner nodes of a side of an element of this dimension."""
    if dimension < 3:
        return dimension
    return 3 if num_nodes in (3, 6) else 4


def write_sidesets(mesh: Mesh, filename: str) -> None:
    """Write the exterior boundary of a mesh as a VTK surface file (``.vtu``)
    for viewing the side sets, e.g., in ParaView.

    Every exterior face is written with its corner nodes.  The cell field
    ``side_set`` holds, for each face, the position of its side set in
    :meth:`~dualmesh.Mesh.sideset_names` plus one (0 for a face that belongs to
    no side set, and the last match for a face in several).  Each side set
    also has a field of its own name, equal to 1 on its faces and 0 elsewhere,
    so that overlapping side sets can be shown one at a time."""
    import meshio

    names = list(mesh.sideset_names())
    exterior = [tuple(side) for side in mesh.exterior_sides()]
    index = {side: k for k, side in enumerate(exterior)}
    membership = np.zeros((len(names), len(exterior)), dtype=np.int8)
    for k, name in enumerate(names):
        for side in mesh.sideset(name):
            row = index.get(tuple(side))
            if row is not None:
                membership[k, row] = 1
    # The last side set that holds a face, or 0 for none.
    numbered = membership * np.arange(1, len(names) + 1)[:, None]
    side_set = numbered.max(axis=0, initial=0).astype(np.int32)

    # The corner nodes of each face, in one cell block per face shape.
    faces = [
        nodes[: _face_corners(mesh.dimension, len(nodes))] for nodes in mesh.side_nodes(exterior)
    ]
    used = sorted({n for face in faces for n in face})
    renumber = {n: i for i, n in enumerate(used)}
    points = np.asarray(mesh.points())[used]
    cells, cell_data = [], {"side_set": []}
    cell_data.update({name: [] for name in names})
    for corners in sorted({len(face) for face in faces}):
        rows = [r for r, face in enumerate(faces) if len(face) == corners]
        cells.append(
            (_MESHIO_FACE_TYPES[corners], np.array([[renumber[n] for n in faces[r]] for r in rows]))
        )
        cell_data["side_set"].append(side_set[rows])
        for k, name in enumerate(names):
            cell_data[name].append(membership[k, rows])
    meshio.write(filename, meshio.Mesh(points, cells, cell_data=cell_data), file_format="vtu")


def mesh_summary(mesh: Mesh) -> str:
    """Human-readable description of a mesh."""
    return mesh.summary()
