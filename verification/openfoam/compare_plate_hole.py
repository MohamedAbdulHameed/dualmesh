# SPDX-License-Identifier: LGPL-2.1-or-later
"""Compare dualmesh and OpenFOAM on the classical plate-with-a-hole problem.

``plate_hole_openfoam.sh`` runs OpenFOAM's ``plateHole`` tutorial
(``solidDisplacementFoam``): a quarter of a square plate, 2 m by 2 m, with a
hole of radius 0.5 m, loaded by a uniform traction of 10 kPa in the x
direction, with symmetry planes on x = 0 and y = 0 (E = 200 GPa, nu = 0.3,
plane stress).  It also exports the mesh.

This script reads **that same mesh**, solves the same problem with dualmesh,
and compares sigma_xx along the line x = 0, where the exact infinite-plate
solution is

    sigma_xx(0, y) = sigma [ 1 + (a/y)^2 / 2 + 3 (a/y)^4 / 2 ] / 1 ,

whose value at the hole is three times the applied stress.

Usage::

    ./plate_hole_openfoam.sh
    python compare_plate_hole.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import dualmesh as dm
import numpy as np

HOLE_RADIUS = 0.5
APPLIED_STRESS = 1.0e4
YOUNGS_MODULUS = 2.0e11
POISSONS_RATIO = 0.3


def read_openfoam_mesh(vtu_path: Path) -> dm.Mesh:
    """Convert the one-cell-thick OpenFOAM mesh into a two-dimensional mesh."""
    import meshio

    foam = meshio.read(vtu_path)
    points = foam.points
    z_min = points[:, 2].min()
    on_plane = np.isclose(points[:, 2], z_min)
    index_map = -np.ones(len(points), dtype=int)
    index_map[on_plane] = np.arange(on_plane.sum())

    # The extruded direction is not always the local z of the hexahedron, so
    # the face lying in the plane is found among the six faces of each cell.
    hexahedron_faces = [
        (0, 3, 2, 1),
        (4, 5, 6, 7),
        (0, 1, 5, 4),
        (1, 2, 6, 5),
        (2, 3, 7, 6),
        (3, 0, 4, 7),
    ]
    quads = []
    for block in foam.cells:
        if block.type != "hexahedron":
            continue
        for cell in block.data:
            for face in hexahedron_faces:
                nodes = [cell[i] for i in face]
                if all(on_plane[node] for node in nodes):
                    quads.append([index_map[node] for node in nodes])
                    break
            else:  # pragma: no cover - the mesh is not one cell thick
                raise ValueError("this mesh is not a single layer of cells")
    mesh = dm.mesh_from_arrays(points[on_plane][:, :2], np.array(quads), element_type="Quad4")

    tolerance = 1e-8
    mesh.add_sideset_by_predicate("left", lambda x, y, z: abs(x) < tolerance)
    mesh.add_sideset_by_predicate("down", lambda x, y, z: abs(y) < tolerance)
    mesh.add_sideset_by_predicate("right", lambda x, y, z: abs(x - 2.0) < tolerance)
    mesh.add_sideset_by_predicate("up", lambda x, y, z: abs(y - 2.0) < tolerance)
    mesh.add_sideset_by_predicate("hole", lambda x, y, z: abs(np.hypot(x, y) - HOLE_RADIUS) < 1e-3)
    for name in ("left", "down", "right", "up", "hole"):
        mesh.add_nodeset(name, mesh.boundary_nodes(name))
    return mesh


def solve_with_dualmesh(mesh, method: str = "dmcdm"):
    problem = dm.Problem(mesh, method=method)
    dm.physics.add_plane_elasticity(
        problem,
        displacements=["u", "v"],
        youngs_modulus=YOUNGS_MODULUS,
        poissons_ratio=POISSONS_RATIO,
        formulation="plane_stress",
    )
    problem.add_boundary_condition(
        "DirichletBC", "symmetry_x", variable="u", boundary="left", value=0.0
    )
    problem.add_boundary_condition(
        "DirichletBC", "symmetry_y", variable="v", boundary="down", value=0.0
    )
    problem.add_boundary_condition(
        "TractionBC", "load", variable="u", boundary="right", traction=APPLIED_STRESS
    )
    problem.solve()
    return problem


def nodal_stress(problem, component: int = 0) -> np.ndarray:
    """Average the element-centre stresses onto the nodes."""
    centroid_stress = problem.property_at_centroids("stress")[:, component]
    total = np.zeros(problem.mesh.num_nodes)
    count = np.zeros(problem.mesh.num_nodes)
    for element in range(problem.mesh.num_elements):
        for node in problem.mesh.element_nodes(element):
            total[node] += centroid_stress[element]
            count[node] += 1
    return total / np.maximum(count, 1)


def exact_stress(y: np.ndarray) -> np.ndarray:
    """Kirsch's solution for an infinite plate: sigma_xx on the line x = 0."""
    ratio = (HOLE_RADIUS / y) ** 2
    return APPLIED_STRESS * (1.0 + 0.5 * ratio + 1.5 * ratio**2)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, default=Path(__file__).parent / "run" / "plateHole")
    arguments = parser.parse_args()

    vtus = sorted(arguments.case.glob("VTK/*/internal.vtu"))
    if not vtus:
        print("No OpenFOAM results found; run ./plate_hole_openfoam.sh first.")
        return 1
    mesh = read_openfoam_mesh(vtus[-1])
    print(
        f"mesh imported from OpenFOAM: {mesh.num_nodes} nodes, {mesh.num_elements} quadrilaterals"
    )

    problem = solve_with_dualmesh(mesh)
    stress = nodal_stress(problem, component=0)

    graphs = sorted(arguments.case.glob("postProcessing/singleGraph/*/line_sigmaxx.xy"))
    foam = np.loadtxt(graphs[-1]) if graphs else None

    # The line x = 0 runs from the hole (y = 0.5) to the top edge (y = 2).
    line_nodes = list(mesh.boundary_nodes("left"))
    coordinates = np.asarray(mesh.points())[line_nodes, 1]
    order = np.argsort(coordinates)
    y = coordinates[order]
    dual = stress[np.asarray(line_nodes)[order]]

    print("\nsigma_xx along x = 0 (Pa)")
    print(f"{'y':>8}{'dualmesh':>14}{'OpenFOAM':>14}{'Kirsch':>14}")
    for y_value in (0.5, 0.6, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0):
        mine = np.interp(y_value, y, dual)
        reference = np.interp(y_value, foam[:, 0], foam[:, 1]) if foam is not None else np.nan
        print(f"{y_value:8.3f}{mine:14.1f}{reference:14.1f}{exact_stress(y_value):14.1f}")

    concentration_dual = dual[0] / APPLIED_STRESS
    print(f"\nstress concentration factor at the hole: dualmesh {concentration_dual:.3f}", end="")
    if foam is not None:
        print(f", OpenFOAM {foam[0, 1] / APPLIED_STRESS:.3f}", end="")
    print(", infinite plate 3.000")
    print("(the finite width of this plate raises the exact factor slightly above 3)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
