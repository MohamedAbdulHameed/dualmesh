# SPDX-License-Identifier: LGPL-2.1-or-later
"""A steel combination wrench that tightens a nut, on an imported Gmsh mesh.

The load case follows the COMSOL Application Gallery model "Stresses and
Strains in a Wrench": a structural steel wrench, held by the nut and loaded
by a hand force of 150 N.  The geometry of the wrench is built by
``make_wrench_mesh.py``, which writes the mesh ``wrench.msh`` with two named
surfaces (Gmsh physical groups):

``jaws``
    The flats of the open end that bear on the nut.  The nut is taken as
    rigid, and the three displacement components are zero there
    (``fixed_constraint``).
``grip``
    The upper and lower faces of the handle between x = 120 mm and
    x = 145 mm, where a uniform shear traction in the -y direction applies a
    total force of 150 N (``traction_boundary_condition`` with
    ``total_force``).

The script checks the solution against three independent results: the total reaction on the jaws
(150 N), the moment of the reactions about the nut (the force times its arm),
and the bending stress of the handle from beam theory.
"""

import sys
import time

import dualmesh as dm
import numpy as np

MM = 1.0e-3
FORCE = 150.0  # N
YOUNGS_MODULUS, POISSONS_RATIO = 200.0e9, 0.3  # structural steel
GRIP_AREA = 2 * (25.0 * MM) * (12.0 * MM)  # two faces of 25 mm x 12 mm
GRIP_CENTRE_X = 132.5 * MM
NUT_CENTRE = np.array([-12.0 * MM, 0.0])
WIDTH, THICKNESS = 12.0 * MM, 5.0 * MM  # the handle section


def solve(mesh_file="wrench.msh", method="fem"):
    mesh = dm.read_mesh(mesh_file)
    print(dm.sideset_summary(mesh))
    problem = dm.Problem(mesh, method=method)
    solid = problem.add_physics(
        "solid_mechanics",
        "solid",
        displacements=["u", "v", "w"],
        youngs_modulus=YOUNGS_MODULUS,
        poissons_ratio=POISSONS_RATIO,
    )
    # The nut holds the jaws: the three displacement components are zero there.
    solid.add_boundary_condition("fixed_constraint", "jaws")
    # The hand pulls the grip with 150 N in -y, spread uniformly over its area.
    solid.add_boundary_condition(
        "traction_boundary_condition", "grip", total_force=[0.0, -FORCE, 0.0]
    )
    problem.solve()
    return problem


def check(problem):
    """Compare the solution with the statics of the wrench and beam theory."""
    mesh = problem.mesh
    nodes = np.array([mesh.node(n) for n in range(mesh.num_nodes)])
    fx = dict(problem.reactions("u", "jaws"))
    fy = dict(problem.reactions("v", "jaws"))
    reaction_y = sum(fy.values())
    moment = sum(
        (nodes[n, 0] - NUT_CENTRE[0]) * fy[n] - (nodes[n, 1] - NUT_CENTRE[1]) * fx.get(n, 0.0)
        for n in fy
    )
    expected_moment = FORCE * (GRIP_CENTRE_X - NUT_CENTRE[0])

    # Bending stress of the handle at x = 60 mm, away from the head and the
    # grip: sigma_xx = M y / I, with M = F (x_grip - x).
    stress = problem.property_at_centroids("stress")
    centroids = np.array([mesh.element_centroid(e) for e in range(mesh.num_elements)])
    section = np.abs(centroids[:, 0] - 60.0 * MM) < 3.0 * MM
    slope = np.polyfit(centroids[section, 1], stress[section, 0], 1)[0]
    computed = slope * WIDTH / 2.0
    inertia = THICKNESS * WIDTH**3 / 12.0
    beam = FORCE * (GRIP_CENTRE_X - 60.0 * MM) * (WIDTH / 2.0) / inertia

    s = stress
    von_mises = np.sqrt(
        0.5 * ((s[:, 0] - s[:, 1]) ** 2 + (s[:, 1] - s[:, 2]) ** 2 + (s[:, 2] - s[:, 0]) ** 2)
        + 3.0 * (s[:, 3] ** 2 + s[:, 4] ** 2 + s[:, 5] ** 2)
    )
    v = np.array(problem.values("v"))
    return {
        "reaction on the jaws (N)": (reaction_y, FORCE),
        "moment about the nut (N m)": (moment, expected_moment),
        "bending stress at x = 60 mm (MPa)": (computed / 1e6, beam / 1e6),
        "largest von Mises stress (MPa)": (von_mises.max() / 1e6, None),
        "largest deflection (mm)": (np.abs(v).max() / MM, None),
    }, von_mises


def plot_top_face(problem, von_mises, path):
    """The von Mises stress on the upper face of the wrench (z = +2.5 mm),
    one colour per element, written as a PNG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    mesh = problem.mesh
    nodes = np.array([mesh.node(n) for n in range(mesh.num_nodes)])
    top = THICKNESS / 2.0 - 1e-9
    triangles, colours = [], []
    for e in range(mesh.num_elements):
        corners = [n for n in mesh.element_nodes(e)[:4] if nodes[n, 2] > top]
        if len(corners) == 3:
            triangles.append(nodes[corners, :2] / MM)
            colours.append(von_mises[e] / 1e6)
    fig, ax = plt.subplots(figsize=(7.0, 2.6))
    faces = PolyCollection(triangles, array=np.array(colours), cmap="viridis", linewidths=0)
    ax.add_collection(faces)
    ax.set_xlim(-30.0, 185.0)
    ax.set_ylim(-20.0, 20.0)
    ax.set_aspect("equal")
    ax.set_xlabel("x (mm)", fontsize=11)
    ax.set_ylabel("y (mm)", fontsize=11)
    ax.tick_params(labelsize=10)
    bar = fig.colorbar(faces, ax=ax, shrink=0.9, pad=0.02)
    bar.set_label("von Mises stress (MPa)", fontsize=11)
    bar.ax.tick_params(labelsize=10)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {path}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    method = args[0] if args else "fem"
    start = time.perf_counter()
    problem = solve(method=method)
    print(f"{method}: solved in {time.perf_counter() - start:.1f} s")
    results, von_mises = check(problem)
    for name, (value, reference) in results.items():
        extra = f"   (reference {reference:.4g})" if reference is not None else ""
        print(f"{name:36s} {value:10.4g}{extra}")
    problem.write_vtu(f"wrench_{method}.vtu", cell_properties=["stress"])
    print(f"wrote wrench_{method}.vtu")
    if "--figure" in sys.argv:
        plot_top_face(problem, von_mises, f"wrench_{method}_von_mises.png")
