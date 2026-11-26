# SPDX-License-Identifier: LGPL-2.1-or-later
"""NAFEMS LE1, the elliptic membrane, in plane stress.

A quarter of a membrane lies between the ellipses (x/2)^2 + y^2 = 1 and
(x/3.25)^2 + (y/2.75)^2 = 1, in m, with symmetry on x = 0 and on y = 0. A uniform outward
pressure of 10 MPa acts on the outer edge. The material is linear elastic, with E = 210 GPa and
nu = 0.3. The target is the tangential stress sigma_yy = 92.7 MPa at D = (2, 0), on the inner
edge (NAFEMS, The Standard NAFEMS Benchmarks, TNSB Rev. 3, 1990, problem LE1). The meshes are
structured quadrilaterals of second order, n elements around and n/2 across, from Gmsh.

    python run_NAFEMS_LE1.py [output directory] [--plot-only]
"""

import sys
import tempfile
import time
from pathlib import Path

import numpy

here = Path(__file__).resolve().parent
sys.path.insert(0, str(here.parents[0]))
import plotstyle  # noqa: E402

reference_stress = 92.7  # MPa
levels = [4, 8, 16, 32, 64]
default_out = here.parents[2] / "docs" / "_static" / "figures" / "solids"


def build_mesh(n, path):
    """The quarter membrane, n quadrilaterals around and n/2 across, with the edges x0, y0 and outer."""
    import gmsh

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("LE1")
    geometry = gmsh.model.geo
    centre = geometry.addPoint(0, 0, 0)
    A, B = geometry.addPoint(0, 1, 0), geometry.addPoint(0, 2.75, 0)
    C, D = geometry.addPoint(3.25, 0, 0), geometry.addPoint(2, 0, 0)
    inner = geometry.addEllipseArc(D, centre, D, A)
    outer = geometry.addEllipseArc(C, centre, C, B)
    on_x0 = geometry.addLine(A, B)
    on_y0 = geometry.addLine(D, C)
    surface = geometry.addPlaneSurface([geometry.addCurveLoop([on_y0, outer, -on_x0, -inner])])
    for curve in (inner, outer):
        geometry.mesh.setTransfiniteCurve(curve, n + 1)
    for curve in (on_x0, on_y0):
        geometry.mesh.setTransfiniteCurve(curve, n // 2 + 1)
    geometry.mesh.setTransfiniteSurface(surface)
    geometry.mesh.setRecombine(2, surface)
    geometry.synchronize()
    gmsh.model.addPhysicalGroup(1, [on_x0], tag=1, name="x0")
    gmsh.model.addPhysicalGroup(1, [on_y0], tag=2, name="y0")
    gmsh.model.addPhysicalGroup(1, [outer], tag=3, name="outer")
    gmsh.model.addPhysicalGroup(2, [surface], tag=4, name="membrane")
    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.setOrder(2)
    gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
    gmsh.write(str(path))
    gmsh.finalize()
    return path


def solve(mesh_file):
    import dualmesh

    membrane = dualmesh.problem(mesh={"file": str(mesh_file), "length_unit": "m"}, method="FEM")
    membrane.physics("solid_mechanics", formulation="plane_stress")
    membrane.properties(Young_modulus="210 GPa", Poisson_ratio=0.3)
    membrane.boundary_conditions("x0", displacement_x="0 m")
    membrane.boundary_conditions("y0", displacement_y="0 m")
    membrane.boundary_conditions("outer", pressure="-10 MPa")
    membrane.postprocessor("stress_at_D", point_value="stress_yy", point=["2 m", "0 m"])
    start = time.perf_counter()
    solution = membrane.solve(report="none")
    elapsed = time.perf_counter() - start
    return (
        solution.postprocessor("stress_at_D")[-1] / 1e6,
        solution.field("displacement").size,
        elapsed,
    )


def compute():
    data = {"n": [], "stress": [], "unknowns": [], "time": []}
    with tempfile.TemporaryDirectory() as work:
        for n in levels:
            stress, unknowns, elapsed = solve(build_mesh(n, Path(work) / f"LE1_{n}.msh"))
            for key, value in (
                ("n", n),
                ("stress", stress),
                ("unknowns", unknowns),
                ("time", elapsed),
            ):
                data[key].append(value)
            print(
                f"n = {n:3d}, {unknowns:6d} unknowns: sigma_yy(D) = {stress:.4f} MPa "
                f"({100 * (stress / reference_stress - 1):+.3f} %), {elapsed:.1f} s",
                flush=True,
            )
    return {"FEM": {key: numpy.asarray(values) for key, values in data.items()}}


def plot(data, out):
    fig, ax = plotstyle.new_figure()
    for method, (color, marker) in plotstyle.method_style.items():
        if method.upper() in data:
            run = data[method.upper()]
            error = numpy.abs(run["stress"] - reference_stress) / reference_stress
            ax.loglog(run["unknowns"], error, marker + "-", color=color, lw=2, ms=7, label=method)
    plotstyle.style(ax, "unknowns", "relative error of the stress at D")
    plotstyle.legend_below(ax, ncol=2)
    plotstyle.save(fig, out / "NAFEMS_LE1_stress_error.png")


if __name__ == "__main__":
    out, plot_only = plotstyle.parse_arguments(sys.argv[1:], default_out)
    cache = plotstyle.cache_path(__file__)
    data = plotstyle.load_cache(cache) if plot_only else compute()
    if not plot_only:
        plotstyle.save_cache(cache, data)
    plot(data, out)
