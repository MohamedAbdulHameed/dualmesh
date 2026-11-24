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
    (the word ``fixed``).
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

import dualmesh
import numpy

mm = 1.0e-3
force = 150.0  # N
youngs_modulus, poissons_ratio = 200.0e9, 0.3  # Structural steel
grip_area = 2 * (25.0 * mm) * (12.0 * mm)  # Two faces of 25 mm x 12 mm
grip_centre_x = 132.5 * mm
nut_centre = numpy.array([-12.0 * mm, 0.0])
width, thickness = 12.0 * mm, 5.0 * mm  # The handle section


def solve(mesh_file="wrench.msh", method="FEM"):
    mesh = dualmesh.read_mesh(mesh_file)
    print(dualmesh.sideset_summary(mesh))
    problem = dualmesh.Problem(mesh, method=method)
    solid = problem.add_physics(
        "solid_mechanics",
        youngs_modulus=youngs_modulus,
        poissons_ratio=poissons_ratio,
    )
    # The nut holds the jaws: the three displacement components are zero there.
    solid.add_boundary_condition("fixed", boundary="jaws")
    # The hand pulls the grip with 150 N in -y, spread uniformly over its area.
    solid.add_boundary_condition("total_force = [0, -F, 0]", boundary="grip", F=force)
    problem.solve()
    return problem


def check(problem):
    """Compare the solution with the statics of the wrench and beam theory."""
    mesh = problem.mesh
    nodes = numpy.array([mesh.node(n) for n in range(mesh.num_nodes)])
    # The force of the jaws on the wrench at each node: the flux of each equation out of the body,
    # reversed.
    fx = {n: -q for n, q in problem.boundary_flux("disp_x", "jaws", per_node=True).items()}
    fy = {n: -q for n, q in problem.boundary_flux("disp_y", "jaws", per_node=True).items()}
    reaction_y = sum(fy.values())
    moment = sum(
        (nodes[n, 0] - nut_centre[0]) * fy[n] - (nodes[n, 1] - nut_centre[1]) * fx.get(n, 0.0)
        for n in fy
    )
    expected_moment = force * (grip_centre_x - nut_centre[0])

    # Bending stress of the handle at x = 60 mm, away from the head and the
    # grip: sigma_xx = M y / I, with M = F (x_grip - x).
    stress = problem.property_at_centroids("stress")
    centroids = numpy.array([mesh.element_centroid(e) for e in range(mesh.num_elements)])
    section = numpy.abs(centroids[:, 0] - 60.0 * mm) < 3.0 * mm
    slope = numpy.polyfit(centroids[section, 1], stress[section, 0], 1)[0]
    computed = slope * width / 2.0
    inertia = thickness * width**3 / 12.0
    beam = force * (grip_centre_x - 60.0 * mm) * (width / 2.0) / inertia

    s = stress
    von_Mises = numpy.sqrt(
        0.5 * ((s[:, 0] - s[:, 1]) ** 2 + (s[:, 1] - s[:, 2]) ** 2 + (s[:, 2] - s[:, 0]) ** 2)
        + 3.0 * (s[:, 3] ** 2 + s[:, 4] ** 2 + s[:, 5] ** 2)
    )
    v = numpy.array(problem.values("disp_y"))
    return {
        "reaction on the jaws (N)": (reaction_y, force),
        "moment about the nut (N m)": (moment, expected_moment),
        "bending stress at x = 60 mm (MPa)": (computed / 1e6, beam / 1e6),
        "largest von Mises stress (MPa)": (von_Mises.max() / 1e6, None),
        "largest deflection (mm)": (numpy.abs(v).max() / mm, None),
    }, von_Mises


def plot_top_face(problem, von_Mises, path):
    """The von Mises stress on the upper face of the wrench (z = +2.5 mm),
    one colour per element, written as a PNG."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot
    from matplotlib.collections import PolyCollection

    mesh = problem.mesh
    nodes = numpy.array([mesh.node(n) for n in range(mesh.num_nodes)])
    top = thickness / 2.0 - 1e-9
    triangles, colours = [], []
    for e in range(mesh.num_elements):
        corners = [n for n in mesh.element_nodes(e)[:4] if nodes[n, 2] > top]
        if len(corners) == 3:
            triangles.append(nodes[corners, :2] / mm)
            colours.append(von_Mises[e] / 1e6)
    fig, ax = pyplot.subplots(figsize=(7.0, 2.6))
    faces = PolyCollection(triangles, array=numpy.array(colours), cmap="viridis", linewidths=0)
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
    pyplot.close(fig)
    print(f"wrote {path}")


if __name__ == "__main__":
    arguments = [a for a in sys.argv[1:] if not a.startswith("--")]
    method = arguments[0] if arguments else "FEM"
    start = time.perf_counter()
    problem = solve(method=method)
    print(f"{method}: solved in {time.perf_counter() - start:.1f} s")
    results, von_Mises = check(problem)
    for name, (value, reference) in results.items():
        extra = f"   (reference {reference:.4g})" if reference is not None else ""
        print(f"{name:36s} {value:10.4g}{extra}")
    problem.write_vtu(f"wrench_{method}.vtu", cell_properties=["stress"])
    print(f"wrote wrench_{method}.vtu")
    if "--figure" in sys.argv:
        plot_top_face(problem, von_Mises, f"wrench_{method}_von_mises.png")
