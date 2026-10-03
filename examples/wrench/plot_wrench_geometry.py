# SPDX-License-Identifier: LGPL-2.1-or-later
"""Draw the figures of the boundary conditions of the wrench tutorial.

The script reads ``wrench.msh`` and writes three PNG files:

``wrench_geometry.png``
    The wrench in top view with its main dimensions, the nut, the fixed jaws
    and the loaded grip.
``wrench_sidesets_3d.png``
    The surface mesh in a three-dimensional view, with the faces of the side
    sets ``jaws`` and ``grip`` coloured.
``wrench_jaws_3d.png``
    A closer view of the open end, which shows the two jaw faces.

The two views are renderings of the file ``wrench_sidesets.vtu`` that
:func:`dualmesh.write_sidesets` writes, which is the file to open in
ParaView.  They need the ``pyvista`` package (``pip install pyvista``).

    python plot_wrench_geometry.py [output directory, default: this directory]
"""

import math
import sys
import tempfile
from pathlib import Path

import dualmesh as dm
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402
from matplotlib.patches import Polygon, Rectangle  # noqa: E402

MM = 1.0e-3
JAWS, GRIP = "#2a78d6", "#eb6834"  # categorical slots 1 and 3 of the benchmark style
BODY, EDGE, INK, MUTED = "#d9d8d2", "#8f8e88", "#0b0b0b", "#52514e"
NUT_CENTRE, NUT_ACROSS_FLATS = (-12.0, 0.0), 13.0
HANDLE_WIDTH, THICKNESS, GRIP_RANGE = 12.0, 5.0, (120.0, 145.0)
FORCE = 150.0


def exterior_faces(mesh):
    """Corner coordinates (mm) of every exterior face and the name of its side set."""
    points = np.asarray(mesh.points()) / MM
    sides = [tuple(s) for s in mesh.exterior_sides()]
    owner = {}
    for name in mesh.sideset_names():
        for side in mesh.sideset(name):
            owner[tuple(side)] = name
    faces = [points[nodes[:3]] for nodes in mesh.side_nodes(sides)]
    return faces, [owner.get(side, "") for side in sides]


def dimension(ax, start, end, text, offset=(0.0, 0.0), text_offset=(0.0, 0.0), **kwargs):
    """A dimension line with arrows at both ends and its value as text."""
    ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="<|-|>", color=MUTED, lw=0.9, shrinkA=0, shrinkB=0, mutation_scale=9))
    middle = ((start[0] + end[0]) / 2 + text_offset[0], (start[1] + end[1]) / 2 + text_offset[1])
    ax.text(*middle, text, ha="center", va="center", fontsize=11, color=INK, **kwargs)


def plot_top_view(mesh, path):
    faces, owners = exterior_faces(mesh)
    top = [f[:, :2] for f in faces if np.all(f[:, 2] > THICKNESS / 2 - 1e-6)]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.add_collection(PolyCollection(top, facecolors=BODY, edgecolors=BODY, linewidths=0.3))

    # The nut: a rigid hexagon whose flats bear on the jaws.
    half = NUT_ACROSS_FLATS / 2
    corner = half / math.cos(math.pi / 6)
    hexagon = [(NUT_CENTRE[0] + corner * math.cos(k * math.pi / 3), NUT_CENTRE[1] + corner * math.sin(k * math.pi / 3)) for k in range(6)]
    ax.add_patch(Polygon(hexagon, closed=True, facecolor="white", edgecolor=MUTED, hatch="////", linewidth=0.8))
    ax.plot(*NUT_CENTRE, marker="+", color=INK, markersize=7)

    # The jaws: the two faces at |y| = 6.5 mm between the ends of the nut
    # flats, labelled through the open end of the slot.
    for y in (half, -half):
        ax.plot([NUT_CENTRE[0] - corner / 2, NUT_CENTRE[0] + corner / 2], [y, y], color=JAWS, linewidth=3.2, solid_capstyle="butt")
        ax.annotate("", xy=(NUT_CENTRE[0] - corner / 2 + 0.6, y), xytext=(-33.0, 20.5), arrowprops=dict(arrowstyle="-", color=JAWS, lw=1.0, shrinkA=0, shrinkB=0))
    ax.text(-58.0, 22.0, "jaws: $u = v = w = 0$", ha="left", va="bottom", fontsize=11, color=INK)
    ax.annotate("nut (rigid)", xy=(NUT_CENTRE[0] + 2.5, -3.0), xytext=(5.0, -17.0), fontsize=10.5, color=MUTED, ha="left", arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))

    # The grip: a uniform traction in -y on the upper and lower faces.
    x0, x1 = GRIP_RANGE
    ax.add_patch(Rectangle((x0, -HANDLE_WIDTH / 2), x1 - x0, HANDLE_WIDTH, facecolor=GRIP, alpha=0.45, edgecolor=GRIP, linewidth=1.0))
    for x in np.linspace(x0 + 2.5, x1 - 2.5, 5):
        ax.annotate("", xy=(x, HANDLE_WIDTH / 2 - 1.0), xytext=(x, HANDLE_WIDTH / 2 + 9.0), arrowprops=dict(arrowstyle="-|>", color=GRIP, lw=1.3, mutation_scale=10))
    ax.text((x0 + x1) / 2, HANDLE_WIDTH / 2 + 11.0, f"grip: $F = {FORCE:.0f}$ N in $-y$", ha="center", va="bottom", fontsize=11, color=INK)

    # Dimensions.
    dimension(ax, (100.0, -HANDLE_WIDTH / 2), (100.0, HANDLE_WIDTH / 2), "12", text_offset=(6.5, 0.0))
    dimension(ax, (x0, -12.0), (x1, -12.0), "25", text_offset=(19.0, 0.0))
    dimension(ax, (NUT_CENTRE[0], -30.0), ((x0 + x1) / 2, -30.0), "144.5 (moment arm)", text_offset=(0.0, 5.0))
    for x, y in ((NUT_CENTRE[0], -half - 1.0), ((x0 + x1) / 2, -2.0)):
        ax.plot([x, x], [-30.5, y], color=MUTED, linewidth=0.6, linestyle=(0, (2, 2)))
    for x in (x0, x1):
        ax.plot([x, x], [-13.5, -HANDLE_WIDTH / 2], color=MUTED, linewidth=0.6, linestyle=(0, (2, 2)))

    ax.set_xlim(-60.0, 185.0)
    ax.set_ylim(-33.0, 32.0)
    ax.set_aspect("equal")
    ax.set_xlabel("$x$ (mm)", fontsize=12, color=MUTED)
    ax.set_ylabel("$y$ (mm)", fontsize=12, color=MUTED)
    ax.tick_params(labelsize=11, colors=MUTED)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {path}")


def render_surface(sideset_file, view, zoom, anchors, offsets, body_opacity=1.0, focus=None, window=(2000, 1200)):
    """An image of the side-set file written by dualmesh.write_sidesets, as
    ParaView shows it, rendered off screen with PyVista, and the pixel
    positions of the points ``anchors`` (in metres) in that image."""
    import pyvista as pv
    from vtkmodules.vtkRenderingCore import vtkCoordinate

    surface = pv.read(sideset_file)
    plotter = pv.Plotter(off_screen=True, window_size=window)
    plotter.set_background("white")
    names = {0: (BODY, body_opacity), 1: (GRIP, 1.0), 2: (JAWS, 1.0)}  # side_set: 0 none, 1 grip, 2 jaws
    for value, (colour, opacity) in names.items():
        part = surface.extract_cells(np.flatnonzero(surface.cell_data["side_set"] == value))
        if part.n_cells:
            plotter.add_mesh(part, color=colour, opacity=opacity, show_edges=True, edge_color="#6f6e69", line_width=0.4, ambient=0.0 if value == 0 else 0.45)
    plotter.view_vector(view, viewup=(0.0, 0.0, 1.0))
    plotter.reset_camera()
    if focus is not None:
        # Look at the point ``focus`` from the same direction and distance.
        distance = plotter.camera.distance
        direction = np.asarray(view, dtype=float) / np.linalg.norm(view)
        plotter.camera.focal_point = focus
        plotter.camera.position = tuple(np.asarray(focus) + distance * direction)
    plotter.camera.zoom(zoom)
    image = plotter.screenshot(return_img=True)
    pixels = []
    coordinate = vtkCoordinate()
    coordinate.SetCoordinateSystemToWorld()
    for point in anchors:
        coordinate.SetValue(*point)
        x, y = coordinate.GetComputedDoubleDisplayValue(plotter.renderer)
        pixels.append((x, image.shape[0] - y))
    plotter.close()
    # Crop the white margin around the wrench, keeping room for the labels.
    ink = np.argwhere(np.any(image[:, :, :3] < 250, axis=2))
    (top, left), (bottom, right) = ink.min(axis=0) - 20, ink.max(axis=0) + 20
    for (x, y), (dx, dy) in zip(pixels, offsets):
        top, bottom = min(top, y + dy - 50), max(bottom, y + dy + 50)
        left, right = min(left, x + dx - 220), max(right, x + dx + 220)
    top, left = int(max(top, 0)), int(max(left, 0))
    bottom, right = int(min(bottom, image.shape[0])), int(min(right, image.shape[1]))
    return image[top:bottom, left:right], [(x - left, y - top) for x, y in pixels]


def plot_surface(sideset_file, path, view, zoom, labels, body_opacity=1.0, focus=None, width=6.4):
    """``labels``: (text, anchor point in metres, text offset in pixels, colour)."""
    image, pixels = render_surface(sideset_file, view, zoom, [label[1] for label in labels], [label[2] for label in labels], body_opacity, focus)
    fig, ax = plt.subplots(figsize=(width, width * image.shape[0] / image.shape[1]))
    ax.imshow(image)
    ax.set_axis_off()
    for (text, _, offset, colour), (x, y) in zip(labels, pixels):
        ax.annotate(text, xy=(x, y), xytext=(x + offset[0], y + offset[1]), fontsize=11.5, color=INK, ha="center", va="center", arrowprops=dict(arrowstyle="-", color=colour, lw=1.2, shrinkA=4, shrinkB=0))
    fig.subplots_adjust(0, 0, 1, 1)
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)
    print(f"wrote {path}")


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else here
    mesh = dm.read_mesh(str(here / "wrench.msh"))
    plot_top_view(mesh, out / "wrench_geometry.png")
    work = tempfile.TemporaryDirectory()
    sideset_file = str(Path(work.name) / "wrench_sidesets.vtu")
    dm.write_sidesets(mesh, sideset_file)
    labels = [("grip", (0.1325, 0.0, 0.0025), (0, -170), GRIP), ("jaws", (-0.0105, 0.0065, 0.0), (120, 250), JAWS)]
    plot_surface(sideset_file, out / "wrench_sidesets_3d.png", (0.25, -0.75, 0.62), 1.05, labels)
    labels = [("jaws ($y = +6.5$ mm)", (-0.0105, 0.0065, 0.001), (-60, -330), JAWS), ("jaws ($y = -6.5$ mm)", (-0.0105, -0.0065, -0.001), (330, 170), JAWS)]
    plot_surface(sideset_file, out / "wrench_jaws_3d.png", (-0.55, -0.45, 0.70), 4.0, labels, body_opacity=0.45, focus=(-0.010, 0.0, 0.0))
    work.cleanup()
