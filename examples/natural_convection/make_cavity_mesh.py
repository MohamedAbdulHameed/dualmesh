# SPDX-License-Identifier: LGPL-2.1-or-later
"""Build the mesh of a square cavity whose cells cluster at its four walls with Gmsh, and write
cavity.msh.

The cavity has the side 100 mm and 80 by 80 quadrilaterals. The thermal and viscous boundary
layers of natural convection lie along the walls, so the lines of the mesh follow the cosine
spacing x_i = (L/2) (1 - cos(pi i / n)), i = 0, ..., n, with the side L and n = 80 cells along
each axis. The first cell at a wall is 0.039 mm wide, and the middle cell 1.96 mm. The script
places the nodes itself and gives them to Gmsh as a discrete mesh. The coordinates of the file
are in mm. Four named curves (Gmsh physical groups) carry the boundary conditions: ``left``
(x = 0), ``right`` (x = 100 mm), ``bottom`` (y = 0) and ``top`` (y = 100 mm).

The script needs the ``gmsh`` Python package (``pip install gmsh``). The mesh file is part of
the repository, so the example runs without Gmsh.
"""

from pathlib import Path

import gmsh
import numpy

side, cells = 100.0, 80  # mm, cells along each axis


def build(path=None):
    lines = 0.5 * side * (1.0 - numpy.cos(numpy.pi * numpy.arange(cells + 1) / cells))
    # The node of the lines i and j along x and y has the tag 1 + i + (cells + 1) j.
    x, y = numpy.meshgrid(lines, lines)
    coordinates = numpy.column_stack([x.ravel(), y.ravel(), numpy.zeros(x.size)])

    def tag(i, j):
        return 1 + i + (cells + 1) * j

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("cavity")
    cavity = gmsh.model.addDiscreteEntity(2)
    gmsh.model.mesh.addNodes(2, cavity, list(range(1, x.size + 1)), coordinates.ravel().tolist())
    # Quadrilaterals (Gmsh type 3), each with its corners counterclockwise.
    quadrilaterals = [
        [tag(i, j), tag(i + 1, j), tag(i + 1, j + 1), tag(i, j + 1)]
        for j in range(cells)
        for i in range(cells)
    ]
    gmsh.model.mesh.addElementsByType(cavity, 3, [], numpy.ravel(quadrilaterals).tolist())
    walls = {
        "left": [tag(0, j) for j in range(cells + 1)],
        "right": [tag(cells, j) for j in range(cells + 1)],
        "bottom": [tag(i, 0) for i in range(cells + 1)],
        "top": [tag(i, cells) for i in range(cells + 1)],
    }
    for name, nodes in walls.items():
        wall = gmsh.model.addDiscreteEntity(1)
        # Two-node lines (Gmsh type 1) along the wall.
        segments = numpy.column_stack([nodes[:-1], nodes[1:]]).ravel().tolist()
        gmsh.model.mesh.addElementsByType(wall, 1, [], segments)
        gmsh.model.addPhysicalGroup(1, [wall], name=name)
    gmsh.model.addPhysicalGroup(2, [cavity], name="air")
    gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
    path = Path(path or Path(__file__).with_name("cavity.msh"))
    gmsh.write(str(path))
    gmsh.finalize()
    return path


if __name__ == "__main__":
    print(build())
