# SPDX-License-Identifier: LGPL-2.1-or-later
"""Cook's membrane at finite strain, the hyperelastic benchmark of Schroeder et al. (2021).

The tapered plate has the corners (0, 0), (48, 44), (48, 60) and (0, 44) mm and the thickness
1 mm, in plane strain. Its left edge is clamped, and its right edge carries the vertical
traction p0 = 20 MPa per unit undeformed area. The energy is

    psi_1 = mu/2 (I_C - 3) + lambda/4 (J^2 - 1) - (lambda/2 + mu) ln J,

with lambda = 432.099 MPa and mu = 185.185 MPa, that is E = 500 MPa and nu = 0.35. The paper
gives the vertical displacement of the corner A = (48, 60) mm as 10.59528994 mm, converged with
p-FEM (Schroeder et al., Arch. Comput. Methods Eng. 28 (2021) 713-751,
doi:10.1007/s11831-020-09477-3).

The meshes are structured quadrilaterals of second order, n by n, from Gmsh.

    python run_Cook.py [output directory] [--plot-only]
"""

import sys
import tempfile
import time
from pathlib import Path

import numpy

here = Path(__file__).resolve().parent
sys.path.insert(0, str(here.parents[0]))
import plotstyle  # noqa: E402

reference_displacement = 10.59528994  # mm
left_tag, right_tag, membrane_tag = 1, 2, 3  # Physical groups of the mesh
levels = [4, 8, 16, 32, 64, 128]
default_out = here.parents[2] / "docs" / "_static" / "figures" / "solids"
energy = (
    "shear_modulus/2*(I_C - 3) + first_Lame_parameter/4*(J^2 - 1)"
    " - (first_Lame_parameter/2 + shear_modulus)*log(J)"
)


def build_mesh(n, path):
    """The membrane as n by n quadrilaterals of second order, in mm, with the edges left and right."""
    import gmsh

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("Cook")
    geometry = gmsh.model.geo
    corners = [geometry.addPoint(x, y, 0.0) for x, y in ((0, 0), (48, 44), (48, 60), (0, 44))]
    edges = [geometry.addLine(corners[k], corners[(k + 1) % 4]) for k in range(4)]
    surface = geometry.addPlaneSurface([geometry.addCurveLoop(edges)])
    for edge in edges:
        geometry.mesh.setTransfiniteCurve(edge, n + 1)
    geometry.mesh.setTransfiniteSurface(surface)
    geometry.mesh.setRecombine(2, surface)
    geometry.synchronize()
    gmsh.model.addPhysicalGroup(1, [edges[3]], tag=left_tag, name="left")
    gmsh.model.addPhysicalGroup(1, [edges[1]], tag=right_tag, name="right")
    gmsh.model.addPhysicalGroup(2, [surface], tag=membrane_tag, name="membrane")
    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.setOrder(2)
    gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
    gmsh.write(str(path))
    gmsh.finalize()
    return path


def solve(mesh_file):
    import dualmesh

    membrane = dualmesh.problem(mesh={"file": str(mesh_file), "length_unit": "mm"}, method="FEM")
    membrane.physics(
        "solid_mechanics", formulation="plane_strain", kinematics="finite", strain_energy=energy
    )
    membrane.properties(Young_modulus="500 MPa", Poisson_ratio=0.35)
    membrane.boundary_conditions("left", displacement=["0 m", "0 m"])
    membrane.boundary_conditions("right", traction=["0 Pa", "20 MPa"])
    membrane.postprocessor("corner_A", point_value="displacement_y", point=["48 mm", "60 mm"])
    start = time.perf_counter()
    solution = membrane.solve(report="none")
    elapsed = time.perf_counter() - start
    return (
        solution.postprocessor("corner_A")[-1] * 1e3,
        solution.field("displacement").size,
        elapsed,
    )


def compute():
    data = {"n": [], "displacement": [], "unknowns": [], "time": []}
    with tempfile.TemporaryDirectory() as work:
        for n in levels:
            displacement, unknowns, elapsed = solve(build_mesh(n, Path(work) / f"Cook_{n}.msh"))
            for key, value in (
                ("n", n),
                ("displacement", displacement),
                ("unknowns", unknowns),
                ("time", elapsed),
            ):
                data[key].append(value)
            print(
                f"n = {n:3d}, {unknowns:6d} unknowns: u_y(A) = {displacement:.8f} mm "
                f"({100 * (displacement / reference_displacement - 1):+.4f} %), {elapsed:.1f} s",
                flush=True,
            )
    return {"FEM": {key: numpy.asarray(values) for key, values in data.items()}}


def plot(data, out):
    fig, ax = plotstyle.new_figure()
    for method, (color, marker) in plotstyle.method_style.items():
        if method.upper() in data:
            run = data[method.upper()]
            error = numpy.abs(run["displacement"] - reference_displacement) / reference_displacement
            ax.loglog(run["unknowns"], error, marker + "-", color=color, lw=2, ms=7, label=method)
    plotstyle.style(ax, "unknowns", "relative error of the displacement of A")
    plotstyle.legend_below(ax, ncol=2)
    plotstyle.save(fig, out / "Cook_displacement_error.png")


if __name__ == "__main__":
    out, plot_only = plotstyle.parse_arguments(sys.argv[1:], default_out)
    cache = plotstyle.cache_path(__file__)
    data = plotstyle.load_cache(cache) if plot_only else compute()
    if not plot_only:
        plotstyle.save_cache(cache, data)
    plot(data, out)
