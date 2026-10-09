# SPDX-License-Identifier: LGPL-2.1-or-later
"""The two-dimensional IAEA PWR benchmark (Argonne Code Center, Benchmark
Problem Book, ANL-7416 Suppl. 2, 1977, problem 11-A2).

A quarter of a PWR core of 177 fuel assemblies of 20 cm, with two fuel
compositions, nine fully rodded assemblies (four of them in the quarter, two
of them halved by the planes of symmetry) and a water reflector of 20 cm, in
two-group diffusion theory with an axial buckling of 0.8e-4 /cm^2 in every
region and group (Source Situation 11, Fig. 1, and problem 11-A2).  The
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

import dualmesh as dm
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import plotstyle  # noqa: E402

HERE = Path(__file__).resolve().parent
CACHE = plotstyle.cache_path(__file__)
CM = 0.01  # m

# Two-group constants of 11-A2 (cm and 1/cm): D_1, D_2, Sigma_1->2,
# Sigma_a1, Sigma_a2, nu Sigma_f2.
CONSTANTS = {
    "fuel_1": (1.5, 0.4, 0.02, 0.01, 0.08, 0.135),
    "fuel_2": (1.5, 0.4, 0.02, 0.01, 0.085, 0.135),
    "fuel_2_rodded": (1.5, 0.4, 0.02, 0.01, 0.13, 0.135),
    "reflector": (2.0, 0.3, 0.04, 0.0, 0.01, 0.0),
}
BLOCKS = list(CONSTANTS)
AXIAL_BUCKLING = 0.8e-4  # 1/cm^2
REFERENCE_K = 1.02959

# The zones of the lower octant (Source Situation 11, Fig. 1), numbered as
# in the benchmark: (zone, x interval, y interval) in cm.
EDGES = [0.0, 10.0, 30.0, 50.0, 70.0, 90.0, 110.0, 130.0, 150.0, 170.0]
ZONE_CELLS = [
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
REFERENCE_THERMAL_FLUX = {
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
METHODS = ("fem", "dmcdm", "hfvm", "zfvm")
DIVISIONS = (2, 4, 8, 16, 32)


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
    lines = np.arange(0.0, 170.0 + 0.5 * h, h)
    n = len(lines)
    points = np.array([[x * CM, y * CM] for y in lines for x in lines])
    cells, blocks = [], []
    for j in range(n - 1):
        for i in range(n - 1):
            name = region(0.5 * (lines[i] + lines[i + 1]), 0.5 * (lines[j] + lines[j + 1]))
            if name is None:
                continue
            a = j * n + i
            cells.append([a, a + 1, a + n + 1, a + n])
            blocks.append(BLOCKS.index(name))
    used = np.unique(np.array(cells).ravel())
    renumber = -np.ones(len(points), dtype=int)
    renumber[used] = np.arange(len(used))
    mesh = dm.mesh_from_arrays(
        points[used], renumber[np.array(cells)], element_type="Quad4", blocks=blocks
    )
    for index, name in enumerate(BLOCKS):
        mesh.set_block_name(index, name)
    tolerance = 1e-9
    mesh.add_sideset_by_predicate("outer", lambda x, y, z: x > tolerance and y > tolerance)
    return mesh


def solve(method: str, divisions: int) -> dict:
    mesh = core_mesh(divisions)
    problem = dm.Problem(mesh, method=method)
    neutrons = problem.add_physics(
        "neutron_diffusion", "neutrons", groups=2, transverse_buckling=AXIAL_BUCKLING / CM**2
    )
    for name, (d1, d2, s12, a1, a2, nf2) in CONSTANTS.items():
        problem.add_property(
            "multigroup_cross_sections",
            name,
            block=[name],
            diffusion_coefficient=[d1 * CM, d2 * CM],
            absorption_cross_section=[a1 / CM, a2 / CM],
            scattering_cross_section=[[0.0, s12 / CM], [0.0, 0.0]],
            nu_fission_cross_section=[0.0, nf2 / CM],
        )
    neutrons.add_boundary_condition(
        "vacuum_boundary_condition", "outer", extrapolation_distance_ratio=2.1312
    )
    start = time.perf_counter()
    result = problem.solve_eigenvalue()
    seconds = time.perf_counter() - start
    # Zone (assembly) averages of the thermal flux, from the element
    # integrals of the elements whose centres lie in the zone.
    integrals = problem.element_integrals("neutron_flux_2")
    volumes = np.asarray(mesh.element_measures())
    centres = np.array([mesh.element_centroid(e) for e in range(mesh.num_elements)]) / CM
    flux = {}
    for zone, i, j in ZONE_CELLS:
        inside = (
            (centres[:, 0] >= EDGES[i])
            & (centres[:, 0] < EDGES[i + 1])
            & (centres[:, 1] >= EDGES[j])
            & (centres[:, 1] < EDGES[j + 1])
        )
        if inside.any():
            flux[zone] = float(integrals[inside].sum() / volumes[inside].sum())
    return dict(k=result.k_effective, seconds=seconds, unknowns=result.num_dofs, flux=flux)


def assembly_powers(flux: dict) -> dict:
    """nu Sigma_f2 phi_2 of every fuel zone, normalised to a core average
    of one over the volume of the fuel in the quarter core."""
    power = {zone: flux[zone] for zone in REFERENCE_THERMAL_FLUX}
    weights = {}
    for zone, i, j in ZONE_CELLS:
        if zone not in power:
            continue
        # Volume of the zone in the quarter core: a zone off the diagonal
        # stands for itself and its mirror image.
        area = (EDGES[i + 1] - EDGES[i]) * (EDGES[j + 1] - EDGES[j])
        weights[zone] = area * (1.0 if i == j else 2.0)
    average = sum(weights[z] * power[z] for z in power) / sum(weights.values())
    return {z: power[z] / average for z in power}


def compute() -> dict:
    out = {}
    for method in METHODS:
        for divisions in DIVISIONS:
            run = solve(method, divisions)
            out[f"{method}/{divisions}"] = dict(
                k=run["k"],
                seconds=run["seconds"],
                unknowns=run["unknowns"],
                zones=np.array(sorted(run["flux"])),
                flux=np.array([run["flux"][z] for z in sorted(run["flux"])]),
            )
            print(
                f"{method:6s} {divisions:3d} per assembly: k = {run['k']:.6f} ({1e5 * (run['k'] - REFERENCE_K):+.1f} pcm), {run['unknowns']} unknowns, {run['seconds']:.2f} s"
            )
    return out


def main():
    out_dir, plot_only = plotstyle.parse_args(sys.argv[1:], HERE)
    data = plotstyle.load_cache(CACHE) if plot_only else compute()
    if not plot_only:
        plotstyle.save_cache(CACHE, data)
    reference = assembly_powers(REFERENCE_THERMAL_FLUX)
    lines = [
        "method,divisions_per_assembly,unknowns,k_effective,k_error_pcm,max_power_error_percent,rms_power_error_percent,seconds"
    ]
    for method in METHODS:
        for divisions in DIVISIONS:
            run = data[f"{method}/{divisions}"]
            powers = assembly_powers(dict(zip([int(z) for z in run["zones"]], run["flux"])))
            errors = np.array([100.0 * (powers[z] / reference[z] - 1.0) for z in reference])
            lines.append(
                f"{method},{divisions},{int(run['unknowns'])},{float(run['k']):.6f},{1e5 * (float(run['k']) - REFERENCE_K):.1f},{np.max(np.abs(errors)):.2f},{np.sqrt(np.mean(errors**2)):.2f},{float(run['seconds']):.2f}"
            )
    (HERE / "iaea_2d_pwr_results.csv").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    # Convergence of k with the element size.
    fig, ax = plotstyle.new_figure()
    colours = [plotstyle.DUALMESH, plotstyle.SECOND, plotstyle.MEASURED, plotstyle.MUTED]
    for method, colour in zip(METHODS, colours):
        h = [20.0 / d for d in DIVISIONS]
        error = [abs(1e5 * (float(data[f"{method}/{d}"]["k"]) - REFERENCE_K)) for d in DIVISIONS]
        ax.loglog(h, error, "o-", color=colour, label=method)
    ax.loglog(
        [0.625, 10.0],
        [3.0, 3.0 * 256],
        color=plotstyle.REFERENCE_GREY,
        linestyle="--",
        label="second order",
    )
    plotstyle.style(ax, "element size (cm)", "|k - 1.02959| (pcm)", "2D IAEA PWR: convergence of k")
    plotstyle.legend(ax)
    plotstyle.save(fig, out_dir / "iaea_2d_pwr_k_convergence.png")
    # Assembly powers of the finest finite element solution against the reference.
    run = data[f"fem/{DIVISIONS[-1]}"]
    powers = assembly_powers(dict(zip([int(z) for z in run["zones"]], run["flux"])))
    zones = sorted(reference)
    fig, ax = plotstyle.new_figure()
    ax.bar(
        np.arange(len(zones)),
        [100.0 * (powers[z] / reference[z] - 1.0) for z in zones],
        color=plotstyle.DUALMESH,
    )
    ax.set_xticks(np.arange(len(zones)))
    ax.set_xticklabels([str(z) for z in zones], fontsize=7)
    plotstyle.style(
        ax,
        "assembly",
        "power error (%)",
        f"2D IAEA PWR: assembly powers, fem, {DIVISIONS[-1]} elements per assembly",
    )
    plotstyle.save(fig, out_dir / "iaea_2d_pwr_assembly_power_error.png")


if __name__ == "__main__":
    main()
