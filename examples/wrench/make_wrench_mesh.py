# SPDX-License-Identifier: LGPL-2.1-or-later
"""Build the mesh of a combination wrench with Gmsh and write wrench.msh.

The wrench lies in the x-y plane with its thickness along z.  The handle runs
from x = 0 to x = 150 mm, the open end at x < 0 holds a 13 mm nut whose centre
is at (-12, 0) mm, and the ring end at x > 150 mm carries a round hole.  The
dimensions are in metres in the file.

Two named surfaces (Gmsh physical groups) carry the boundary conditions:

``jaws``
    The two flat faces of the open end that bear on the flats of the nut.
``grip``
    The upper and lower faces of the handle where the hand applies the load,
    120 mm < x < 145 mm.

The script needs the ``gmsh`` Python package (``pip install gmsh``).  The
mesh file is part of the repository, so the example runs without Gmsh.

    python make_wrench_mesh.py [element size in mm, default 2.0] [--linear]
"""

import math
import sys
from pathlib import Path

import gmsh

MM = 1.0e-3
THICKNESS = 5.0  # mm
HANDLE_LENGTH, HANDLE_WIDTH = 150.0, 12.0
HEAD_CENTRE, HEAD_RADIUS = (-8.0, 0.0), 17.0
NUT_CENTRE, NUT_ACROSS_FLATS = (-12.0, 0.0), 13.0
RING_CENTRE, RING_RADIUS, HOLE_RADIUS = (HANDLE_LENGTH + 15.0, 0.0), 13.0, 7.0
GRIP = (120.0, 145.0)


def build(size_mm=2.0, order=2, path=None):
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("wrench")
    occ = gmsh.model.occ

    # The handle runs into the ring, so that the two fuse into one body.
    handle = occ.addRectangle(0.0, -HANDLE_WIDTH / 2, 0.0, RING_CENTRE[0] - 5.0, HANDLE_WIDTH)
    head = occ.addDisk(*HEAD_CENTRE, 0.0, HEAD_RADIUS, HEAD_RADIUS)
    ring = occ.addDisk(*RING_CENTRE, 0.0, RING_RADIUS, RING_RADIUS)
    hole = occ.addDisk(*RING_CENTRE, 0.0, HOLE_RADIUS, HOLE_RADIUS)

    # The slot of the open end: a hexagon (flats at y = +-6.5 mm) joined to a
    # straight channel that opens towards -x.
    half = NUT_ACROSS_FLATS / 2.0
    corner = half / math.cos(math.pi / 6.0)
    cx, cy = NUT_CENTRE
    points = [
        occ.addPoint(cx + corner * math.cos(a), cy + corner * math.sin(a), 0.0)
        for a in [k * math.pi / 3.0 for k in range(6)]
    ]
    lines = [occ.addLine(points[k], points[(k + 1) % 6]) for k in range(6)]
    hexagon = occ.addPlaneSurface([occ.addCurveLoop(lines)])
    channel = occ.addRectangle(cx - 40.0, cy - half, 0.0, 40.0, 2.0 * half)
    slot, _ = occ.fuse([(2, hexagon)], [(2, channel)])

    body, _ = occ.fuse([(2, handle)], [(2, head), (2, ring)])
    body, _ = occ.cut(body, slot + [(2, hole)])

    # Split the jaw flats where the flats of the nut end, and the handle where
    # the grip begins and ends, so that both regions are surfaces of their own.
    flat_end = cx - corner / 2.0  # the nut flats span cx -+ corner/2
    cutters = [occ.addPoint(flat_end, half, 0.0), occ.addPoint(flat_end, -half, 0.0)]
    grip = occ.addRectangle(GRIP[0], -HANDLE_WIDTH / 2, 0.0, GRIP[1] - GRIP[0], HANDLE_WIDTH)
    body, _ = occ.fragment(body, [(2, grip)] + [(0, p) for p in cutters])
    faces = [e for e in body if e[0] == 2]
    occ.extrude(faces, 0.0, 0.0, THICKNESS)
    occ.translate(occ.getEntities(3), 0.0, 0.0, -THICKNESS / 2.0)
    occ.dilate(occ.getEntities(3), 0.0, 0.0, 0.0, MM, MM, MM)
    occ.removeAllDuplicates()
    occ.synchronize()

    def surfaces_in(xmin, ymin, zmin, xmax, ymax, zmax):
        eps = 1e-3  # mm, larger than the tolerance of the OpenCASCADE boxes
        return [
            tag
            for _, tag in gmsh.model.getEntitiesInBoundingBox(
                (xmin - eps) * MM,
                (ymin - eps) * MM,
                (zmin - eps) * MM,
                (xmax + eps) * MM,
                (ymax + eps) * MM,
                (zmax + eps) * MM,
                dim=2,
            )
        ]

    t2 = THICKNESS / 2.0
    jaws = surfaces_in(flat_end, half, -t2, cx + corner / 2.0, half, t2) + surfaces_in(
        flat_end, -half, -t2, cx + corner / 2.0, -half, t2
    )
    grip_faces = surfaces_in(GRIP[0], -HANDLE_WIDTH / 2, t2, GRIP[1], HANDLE_WIDTH / 2, t2) + (
        surfaces_in(GRIP[0], -HANDLE_WIDTH / 2, -t2, GRIP[1], HANDLE_WIDTH / 2, -t2)
    )
    if len(jaws) != 2 or len(grip_faces) != 2:
        raise RuntimeError(f"expected 2 jaw and 2 grip faces, found {jaws} and {grip_faces}")
    gmsh.model.addPhysicalGroup(3, [t for _, t in gmsh.model.getEntities(3)], 1, "steel")
    gmsh.model.addPhysicalGroup(2, jaws, 2, "jaws")
    gmsh.model.addPhysicalGroup(2, grip_faces, 3, "grip")

    gmsh.option.setNumber("Mesh.MeshSizeMax", size_mm * MM)
    gmsh.option.setNumber("Mesh.MeshSizeMin", 0.25 * size_mm * MM)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 12)
    gmsh.option.setNumber("Mesh.ElementOrder", order)
    gmsh.option.setNumber("Mesh.SecondOrderLinear", 0)
    gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
    gmsh.option.setNumber("Mesh.Binary", 1)
    gmsh.model.mesh.generate(3)
    path = Path(path or Path(__file__).with_name("wrench.msh"))
    gmsh.write(str(path))
    gmsh.finalize()
    return path


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    size = float(args[0]) if args else 2.0
    print(build(size, order=1 if "--linear" in sys.argv else 2))
