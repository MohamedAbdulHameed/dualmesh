# SPDX-License-Identifier: LGPL-2.1-or-later
"""The two-dimensional IAEA PWR benchmark (Argonne Code Center, Benchmark
Problem Book, ANL-7416 Suppl. 2, 1977, problem 11-A2).

A quarter of a PWR core of 177 fuel assemblies of 20 cm, with two fuel
compositions, nine fully rodded assemblies (four of them in the quarter, two
of them halved by the planes of symmetry) and a water reflector of 20 cm, in
two-group diffusion theory with an axial buckling of 0.8e-4 /cm^2 in every
region and group (Source Situation 11, Fig. 1, and problem 11-A2).  The
diffusion equations are written with the generic general_form_PDE and
solved by the eigenvalue study of any problem.  The
external boundary has no incoming current, which the benchmark states for
diffusion codes as d phi_g / d n = -0.4692 phi_g / D_g, i.e., the
extrapolation distance 2.1312 D_g.  The reference is the extrapolated
finite-difference solution of 11-A2-1 (Vondy and Fowler, VENTURE, mesh
spacing extrapolated from 136 x 136 and 272 x 272 meshes): k_eff = 1.02959
(Table 1) and the zone average thermal fluxes of Table 3, from which the
assembly powers follow, normalised to a core average of one.

The script computes the benchmark with every method on meshes of 2 to 32
elements per assembly, writes ``iaea_2d_pwr_results.csv``, caches the
results in ``run_iaea_2d_pwr_cache.npz`` and draws the figures of the
documentation (``--plot-only`` redraws them from the cache).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import dualmesh
import numpy

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import plotstyle  # noqa: E402

here = Path(__file__).resolve().parent
cache = plotstyle.cache_path(__file__)
cm = 0.01  # m

# Two-group constants of 11-A2 (cm and 1/cm): D_1, D_2, Sigma_1->2,
# Sigma_a1, Sigma_a2, nu Sigma_f2.
constants = {
    "fuel_1": (1.5, 0.4, 0.02, 0.01, 0.08, 0.135),
    "fuel_2": (1.5, 0.4, 0.02, 0.01, 0.085, 0.135),
    "fuel_2_rodded": (1.5, 0.4, 0.02, 0.01, 0.13, 0.135),
    "reflector": (2.0, 0.3, 0.04, 0.0, 0.01, 0.0),
}
block_names = list(constants)
axial_buckling = 0.8e-4  # 1/cm^2
reference_k = 1.02959

# The zones of the lower octant (Source Situation 11, Fig. 1), numbered as
# in the benchmark: (zone, x interval, y interval) in cm.
edges = [0.0, 10.0, 30.0, 50.0, 70.0, 90.0, 110.0, 130.0, 150.0, 170.0]
zone_cells = [
    (1, 0, 0),
    (2, 1, 0),
    (3, 2, 0),
    (4, 3, 0),
    (5, 4, 0),
    (6, 5, 0),
    (7, 6, 0),
    (8, 7, 0),
    (9, 8, 0),
    (10, 1, 1),
    (11, 2, 1),
    (12, 3, 1),
    (13, 4, 1),
    (14, 5, 1),
    (15, 6, 1),
    (16, 7, 1),
    (17, 8, 1),
    (18, 2, 2),
    (19, 3, 2),
    (20, 4, 2),
    (21, 5, 2),
    (22, 6, 2),
    (23, 7, 2),
    (24, 8, 2),
    (25, 3, 3),
    (26, 4, 3),
    (27, 5, 3),
    (28, 6, 3),
    (29, 7, 3),
    (30, 8, 3),
    (31, 4, 4),
    (32, 5, 4),
    (33, 6, 4),
    (34, 7, 4),
    (35, 5, 5),
    (36, 6, 5),
    (37, 7, 5),
    (38, 6, 6),
]
# Extrapolated zone average thermal flux of 11-A2-1, Table 3 (any
# normalisation: the powers are renormalised to a core average of one).
reference_thermal_flux = {
    1: 3.120,
    2: 5.482,
    3: 6.085,
    4: 5.068,
    5: 2.552,
    6: 3.914,
    7: 3.911,
    8: 3.160,
    10: 6.006,
    11: 6.193,
    12: 5.503,
    13: 4.477,
    14: 4.336,
    15: 3.977,
    16: 3.079,
    18: 6.149,
    19: 5.630,
    20: 4.935,
    21: 4.480,
    22: 4.080,
    23: 2.896,
    25: 4.992,
    26: 4.047,
    27: 3.794,
    28: 3.540,
    31: 1.969,
    32: 2.869,
    33: 2.498,
    35: 2.448,
}
methods = ("fem", "dmcdm")
all_divisions = (2, 4, 8, 16, 32)


def region(x: float, y: float) -> str | None:
    """The region at (x, y) in cm, in the quarter core (None outside)."""
    big, small = max(x, y), min(x, y)
    if not (
        (small < 70 and big < 170) or (small < 110 and big < 150) or (small < 130 and big < 130)
    ):
        return None
    if (big < 10) or (70 <= big < 90 and small < 10) or (70 <= big < 90 and 70 <= small < 90):
        return "fuel_2_rodded"
    if (small < 30 and big < 130) or (small < 70 and big < 110) or (small < 90 and big < 90):
        return "fuel_2"
    if (small < 50 and big < 150) or (small < 90 and big < 130) or (small < 110 and big < 110):
        return "fuel_1"
    return "reflector"


def core_mesh(divisions: int):
    """Quadrilaterals of 20 / divisions cm on the quarter core, one block per
    region, and the side set 'outer' on the external boundary.  The planes
    of symmetry x = 0 and y = 0 carry no condition, which reflects the
    neutrons."""
    h = 20.0 / divisions
    lines = numpy.arange(0.0, 170.0 + 0.5 * h, h)
    n = len(lines)
    points = numpy.array([[x * cm, y * cm] for y in lines for x in lines])
    cells, blocks = [], []
    for j in range(n - 1):
        for i in range(n - 1):
            name = region(0.5 * (lines[i] + lines[i + 1]), 0.5 * (lines[j] + lines[j + 1]))
            if name is None:
                continue
            a = j * n + i
            cells.append([a, a + 1, a + n + 1, a + n])
            blocks.append(block_names.index(name))
    used = numpy.unique(numpy.array(cells).ravel())
    renumber = -numpy.ones(len(points), dtype=int)
    renumber[used] = numpy.arange(len(used))
    mesh = dualmesh.mesh_from_arrays(
        points[used], renumber[numpy.array(cells)], element_type="Quad4", blocks=blocks
    )
    for index, name in enumerate(block_names):
        mesh.set_block_name(index, name)
    tolerance = 1e-9
    mesh.add_sideset_by_predicate("outer", lambda x, y, z: x > tolerance and y > tolerance)
    return mesh


def solve(method: str, divisions: int) -> dict:
    """Two-group diffusion in general form: the fast flux phi1 and the thermal
    flux phi2, with the cross sections of every region as properties on its
    block and the transverse buckling B^2 added to the removal of each group
    as D_g B^2.  The fission source carries the symbol eigenvalue, which is
    1 / k.  The vacuum condition is the Robin condition
    n . D grad phi = -phi / 2.1312."""
    mesh = core_mesh(divisions)
    problem = dualmesh.Problem(mesh, method=method)
    buckling = axial_buckling / cm**2
    for name, (d1, d2, s12, a1, a2, nf2) in constants.items():
        d1, d2, s12, a1, a2, nf2 = d1 * cm, d2 * cm, s12 / cm, a1 / cm, a2 / cm, nf2 / cm
        problem.add_property(
            "constant",
            block=[name],
            properties={
                "diffusion_1": d1,
                "diffusion_2": d2,
                "removal_1": a1 + s12 + d1 * buckling,
                "removal_2": a2 + d2 * buckling,
                "scattering_12": s12,
                "nu_fission_2": nf2,
            },
        )
    # The cross sections are the properties of the blocks, used by their names.
    neutrons = problem.add_physics(
        "general_form_PDE",
        variables=["phi1", "phi2"],
        equations=[
            "-div(diffusion_1*grad(phi1)) + removal_1*phi1 = eigenvalue*nu_fission_2*phi2",
            "-div(diffusion_2*grad(phi2)) + removal_2*phi2 = scattering_12*phi1",
        ],
    )
    # No incoming current: the extrapolation distance 2.1312 D_g.
    neutrons.add_boundary_condition("-diffusion_1*dphi1/dn = phi1/2.1312", boundary="outer")
    neutrons.add_boundary_condition("-diffusion_2*dphi2/dn = phi2/2.1312", boundary="outer")
    start = time.perf_counter()
    result = problem.solve_eigenvalue(num_modes=1, report="none")
    seconds = time.perf_counter() - start
    # Zone (assembly) averages of the thermal flux, from the element
    # integrals of the elements whose centres lie in the zone.
    integrals = problem.element_integrals("phi2")
    volumes = numpy.asarray(mesh.element_measures())
    centres = numpy.array([mesh.element_centroid(e) for e in range(mesh.num_elements)]) / cm
    flux = {}
    for zone, i, j in zone_cells:
        inside = (
            (centres[:, 0] >= edges[i])
            & (centres[:, 0] < edges[i + 1])
            & (centres[:, 1] >= edges[j])
            & (centres[:, 1] < edges[j + 1])
        )
        if inside.any():
            flux[zone] = float(integrals[inside].sum() / volumes[inside].sum())
    k = 1.0 / float(result.eigenvalues[0])
    return dict(k=k, seconds=seconds, unknowns=result.num_dofs, flux=flux)


def assembly_powers(flux: dict) -> dict:
    """nu Sigma_f2 phi_2 of every fuel zone, normalised to a core average
    of one over the volume of the fuel in the quarter core."""
    power = {zone: flux[zone] for zone in reference_thermal_flux}
    weights = {}
    for zone, i, j in zone_cells:
        if zone not in power:
            continue
        # Volume of the zone in the quarter core: a zone off the diagonal
        # stands for itself and its mirror image.
        area = (edges[i + 1] - edges[i]) * (edges[j + 1] - edges[j])
        weights[zone] = area * (1.0 if i == j else 2.0)
    average = sum(weights[z] * power[z] for z in power) / sum(weights.values())
    return {z: power[z] / average for z in power}


def compute() -> dict:
    out = {}
    for method in methods:
        for divisions in all_divisions:
            run = solve(method, divisions)
            out[f"{method}/{divisions}"] = dict(
                k=run["k"],
                seconds=run["seconds"],
                unknowns=run["unknowns"],
                zones=numpy.array(sorted(run["flux"])),
                flux=numpy.array([run["flux"][z] for z in sorted(run["flux"])]),
            )
            print(
                f"{method:6s} {divisions:3d} per assembly: k = {run['k']:.6f} ({1e5 * (run['k'] - reference_k):+.1f} pcm), {run['unknowns']} unknowns, {run['seconds']:.2f} s"
            )
    return out


def main():
    out_directory, plot_only = plotstyle.parse_arguments(sys.argv[1:], here)
    data = plotstyle.load_cache(cache) if plot_only else compute()
    if not plot_only:
        plotstyle.save_cache(cache, data)
    reference = assembly_powers(reference_thermal_flux)
    lines = [
        "method,divisions_per_assembly,unknowns,k_effective,k_error_pcm,max_power_error_percent,rms_power_error_percent,seconds"
    ]
    for method in methods:
        for divisions in all_divisions:
            run = data[f"{method}/{divisions}"]
            powers = assembly_powers(dict(zip([int(z) for z in run["zones"]], run["flux"])))
            errors = numpy.array([100.0 * (powers[z] / reference[z] - 1.0) for z in reference])
            lines.append(
                f"{method},{divisions},{int(run['unknowns'])},{float(run['k']):.6f},{1e5 * (float(run['k']) - reference_k):.1f},{numpy.max(numpy.abs(errors)):.2f},{numpy.sqrt(numpy.mean(errors**2)):.2f},{float(run['seconds']):.2f}"
            )
    (here / "iaea_2d_pwr_results.csv").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    # Convergence of k with the element size.
    fig, ax = plotstyle.new_figure()
    colours = [plotstyle.dualmesh, plotstyle.second, plotstyle.measured, plotstyle.muted]
    for method, colour in zip(methods, colours):
        h = [20.0 / d for d in all_divisions]
        error = [
            abs(1e5 * (float(data[f"{method}/{d}"]["k"]) - reference_k)) for d in all_divisions
        ]
        ax.loglog(h, error, "o-", color=colour, label=method)
    ax.loglog(
        [0.625, 10.0],
        [3.0, 3.0 * 256],
        color=plotstyle.reference_grey,
        linestyle="--",
        label="second order",
    )
    plotstyle.style(ax, "element size (cm)", "|k - 1.02959| (pcm)", "2D IAEA PWR: convergence of k")
    plotstyle.legend(ax)
    plotstyle.save(fig, out_directory / "iaea_2d_pwr_k_convergence.png")
    # Assembly powers of the finest finite element solution against the reference.
    run = data[f"fem/{all_divisions[-1]}"]
    powers = assembly_powers(dict(zip([int(z) for z in run["zones"]], run["flux"])))
    zones = sorted(reference)
    fig, ax = plotstyle.new_figure()
    ax.bar(
        numpy.arange(len(zones)),
        [100.0 * (powers[z] / reference[z] - 1.0) for z in zones],
        color=plotstyle.dualmesh,
    )
    ax.set_xticks(numpy.arange(len(zones)))
    ax.set_xticklabels([str(z) for z in zones], fontsize=7)
    plotstyle.style(
        ax,
        "assembly",
        "power error (%)",
        f"2D IAEA PWR: assembly powers, fem, {all_divisions[-1]} elements per assembly",
    )
    plotstyle.save(fig, out_directory / "iaea_2d_pwr_assembly_power_error.png")


if __name__ == "__main__":
    main()
