# SPDX-License-Identifier: LGPL-2.1-or-later
"""Mesh generation and mesh file input/output.

Two ways of getting a mesh, as in MOOSE:

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
    start: float,
    end: float,
    num_elements: int,
    bias: float = 1.0,
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
        return [float(c) for c in coordinates]
    if minimum is None or maximum is None or num_elements is None:
        raise ValueError(
            f"Give either {name}_coordinates or ({name}_min, {name}_max, num_{name}_elements)."
        )
    return graded_coordinates(minimum, maximum, num_elements, bias)


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
) -> Mesh:
    """Structured two-dimensional mesh of ``Quad4``, ``Tri3``, ``Quad8``,
    ``Quad9`` or ``Tri6`` elements.  ``Quad8`` works with ``method="fem"`` and
    ``method="zfvm"`` only.

    Side sets ``"left"``, ``"right"``, ``"bottom"`` and ``"top"`` are created
    automatically.  Pass ``x_coordinates``/``y_coordinates`` for a fully
    non-uniform grid, or the bounds together with the element counts and an
    optional geometric ``bias``.  The element counts always count elements, so
    asking for a quadratic type gives the same number of elements and more
    nodes, not fewer elements.
    """
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
) -> Mesh:
    """Structured three-dimensional mesh of ``Hex8``, ``Tet4``, ``Wedge6``,
    ``Pyramid5``, ``Hex20``, ``Hex27`` or ``Tet10`` elements.  A ``Tet4`` mesh
    has six tetrahedra per cell, a ``Wedge6`` mesh two prisms split on the base
    diagonal, and a ``Pyramid5`` mesh six pyramids meeting at an added centre
    node.  ``Pyramid5`` and ``Hex20`` work with ``method="fem"`` and
    ``method="zfvm"`` only.

    Side sets ``"left"``, ``"right"``, ``"bottom"``, ``"top"``, ``"back"`` and
    ``"front"`` are created automatically.
    """
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


def mesh_from_arrays(
    points,
    cells,
    element_type: str | None = None,
    blocks=None,
    dimension: int | None = None,
) -> Mesh:
    """Build a mesh from a node array and a connectivity array.

    ``points`` has shape ``(num_nodes, 1..3)``.  ``cells`` is either an array of
    shape ``(num_elements, nodes_per_element)`` with a single ``element_type``,
    or a list of ``(element_type, connectivity)`` pairs.  ``blocks`` optionally
    gives one subdomain id per element.
    """
    points = np.atleast_2d(np.asarray(points, dtype=float))
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

    dim = dimension or max(_DIMENSION[t] for t, _ in cell_groups)
    mesh = Mesh(dim)
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

    mesh = Mesh(dim)
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

    physical_names = {}
    for name, (tag, _dimension) in getattr(m, "field_data", {}).items():
        physical_names[int(tag)] = name
    for tag, faces in faces_by_tag.items():
        name = physical_names.get(tag, f"boundary_{tag}")
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


def mesh_summary(mesh: Mesh) -> str:
    """Human-readable description of a mesh."""
    return mesh.summary()
