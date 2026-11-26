# SPDX-License-Identifier: LGPL-2.1-or-later
"""Build the mesh of a quarter of a plate with a circular hole with Gmsh, and write
plate_with_hole.msh.

The quarter is an annulus of inner radius 10 mm, the hole, and outer radius 200 mm, between the
x and y axes. It has 30 by 30 quadrilaterals: the radial cells grow by the ratio 1.15 from the
hole outwards, and the angular cells are uniform. The coordinates of the file are in mm. Four
named curves (Gmsh physical groups) carry the boundary conditions:

``hole``
    The edge of the hole, r = 10 mm.
``outer``
    The outer arc, r = 200 mm.
``bottom``
    The straight edge on the x axis, y = 0.
``left``
    The straight edge on the y axis, x = 0.

The script needs the ``gmsh`` Python package (``pip install gmsh``). The mesh file is part of
the repository, so the example runs without Gmsh.
"""

from pathlib import Path

import gmsh

hole_radius, outer_radius = 10.0, 200.0  # mm
radial_cells, angular_cells, radial_grading = 30, 30, 1.15


def build(path=None):
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("plate_with_hole")
    geo = gmsh.model.geo
    centre = geo.addPoint(0.0, 0.0, 0.0)
    hole_on_x = geo.addPoint(hole_radius, 0.0, 0.0)
    outer_on_x = geo.addPoint(outer_radius, 0.0, 0.0)
    outer_on_y = geo.addPoint(0.0, outer_radius, 0.0)
    hole_on_y = geo.addPoint(0.0, hole_radius, 0.0)
    bottom = geo.addLine(hole_on_x, outer_on_x)
    outer = geo.addCircleArc(outer_on_x, centre, outer_on_y)
    # The left edge runs outwards, as the bottom one does, so both take the same grading.
    left = geo.addLine(hole_on_y, outer_on_y)
    hole = geo.addCircleArc(hole_on_y, centre, hole_on_x)
    loop = geo.addCurveLoop([bottom, outer, -left, hole])
    plate = geo.addPlaneSurface([loop])
    for curve in (bottom, left):
        geo.mesh.setTransfiniteCurve(curve, radial_cells + 1, "Progression", radial_grading)
    for curve in (outer, hole):
        geo.mesh.setTransfiniteCurve(curve, angular_cells + 1)
    geo.mesh.setTransfiniteSurface(plate, cornerTags=[hole_on_x, outer_on_x, outer_on_y, hole_on_y])
    geo.mesh.setRecombine(2, plate)
    geo.synchronize()
    for curve, name in ((hole, "hole"), (outer, "outer"), (bottom, "bottom"), (left, "left")):
        gmsh.model.addPhysicalGroup(1, [curve], name=name)
    gmsh.model.addPhysicalGroup(2, [plate], name="plate")
    gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
    gmsh.model.mesh.generate(2)
    path = Path(path or Path(__file__).with_name("plate_with_hole.msh"))
    gmsh.write(str(path))
    gmsh.finalize()
    return path


if __name__ == "__main__":
    print(build())
