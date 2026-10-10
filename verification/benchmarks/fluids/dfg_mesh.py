# SPDX-License-Identifier: LGPL-2.1-or-later
"""The mesh of the DFG flow around a cylinder benchmarks (Schaefer and Turek 1996,
the FEATFLOW benchmark pages): the channel [0, 2.2] x [0, 0.41] without the
disk of radius 0.05 centered at (0.2, 0.2), meshed with Gmsh (``pip install gmsh``).

The triangles are graded from the size ``h`` on the cylinder to 4 h at a
distance of 0.3 from it.  The side sets are ``inlet`` (x = 0), ``outlet``
(x = 2.2), ``walls`` (y = 0 and y = 0.41) and ``cylinder``.  With ``order=2``
the elements are quadratic and their sides on the cylinder lie on the circle.
"""

from pathlib import Path

LENGTH, HEIGHT = 2.2, 0.41
CENTER, RADIUS = (0.2, 0.2), 0.05


def build(h, path, order=1):
    import gmsh

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("dfg")
    occ = gmsh.model.occ
    channel = occ.addRectangle(0.0, 0.0, 0.0, LENGTH, HEIGHT)
    disk = occ.addDisk(CENTER[0], CENTER[1], 0.0, RADIUS, RADIUS)
    occ.cut([(2, channel)], [(2, disk)])
    occ.synchronize()

    def curves_in(xmin, ymin, xmax, ymax):
        eps = 1e-6
        return [
            tag
            for _, tag in gmsh.model.getEntitiesInBoundingBox(
                xmin - eps, ymin - eps, -eps, xmax + eps, ymax + eps, eps, dim=1
            )
        ]

    inlet = curves_in(0.0, 0.0, 0.0, HEIGHT)
    outlet = curves_in(LENGTH, 0.0, LENGTH, HEIGHT)
    walls = curves_in(0.0, 0.0, LENGTH, 0.0) + curves_in(0.0, HEIGHT, LENGTH, HEIGHT)
    cylinder = curves_in(
        CENTER[0] - RADIUS, CENTER[1] - RADIUS, CENTER[0] + RADIUS, CENTER[1] + RADIUS
    )
    if not (inlet and outlet and len(walls) == 2 and cylinder):
        raise RuntimeError(f"boundary curves not found: {inlet} {outlet} {walls} {cylinder}")
    gmsh.model.addPhysicalGroup(2, [t for _, t in gmsh.model.getEntities(2)], 1, "fluid")
    gmsh.model.addPhysicalGroup(1, inlet, 2, "inlet")
    gmsh.model.addPhysicalGroup(1, outlet, 3, "outlet")
    gmsh.model.addPhysicalGroup(1, walls, 4, "walls")
    gmsh.model.addPhysicalGroup(1, cylinder, 5, "cylinder")

    # The element size grows from h on the cylinder to 4 h at the distance 0.3.
    field = gmsh.model.mesh.field
    distance = field.add("Distance")
    field.setNumbers(distance, "CurvesList", cylinder)
    field.setNumber(distance, "Sampling", 200)
    threshold = field.add("Threshold")
    field.setNumber(threshold, "InField", distance)
    field.setNumber(threshold, "SizeMin", h)
    field.setNumber(threshold, "SizeMax", 4 * h)
    field.setNumber(threshold, "DistMin", 0.0)
    field.setNumber(threshold, "DistMax", 0.3)
    field.setAsBackgroundMesh(threshold)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.Algorithm", 6)
    gmsh.option.setNumber("Mesh.ElementOrder", order)
    gmsh.option.setNumber("Mesh.SecondOrderLinear", 0)
    gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
    gmsh.model.mesh.generate(2)
    path = Path(path)
    gmsh.write(str(path))
    gmsh.finalize()
    return path
