# SPDX-License-Identifier: LGPL-2.1-or-later
r"""The k-eigenvalue study of the neutron diffusion equations.

The steady multigroup diffusion equations of a reactor,

.. math::

   \mathbf{L} \boldsymbol{\phi} = \frac{1}{k} \mathbf{F} \boldsymbol{\phi} ,

where :math:`\mathbf{L}` is the operator of the leakage, the removal and the
scattering between groups, :math:`\mathbf{F}` the operator of the fission
source and :math:`\boldsymbol{\phi}` the fluxes of all groups (J. C. Lee,
*Nuclear Reactor Physics and Engineering*, 2nd ed., Wiley 2025, Eq. 7.20),
have nontrivial solutions only for discrete values of :math:`k`.  The
largest is the effective multiplication factor :math:`k_\mathrm{eff}`, and
its eigenvector, which is the only one that is positive everywhere, is the
fundamental mode of the flux.

:meth:`dualmesh.Problem.solve_eigenvalue` assembles the two discrete
operators from the objects of the problem: the Jacobian with the fission
source switched off is :math:`\mathbf{L}`, and the Jacobian with it switched
on is :math:`\mathbf{L} - \mathbf{F}`.  It factorises :math:`\mathbf{L}` once
and computes the dominant eigenvalue of :math:`\mathbf{L}^{-1}\mathbf{F}`,
either by the Arnoldi method (``method="krylov"``, ARPACK through SciPy) or by
the power iteration of Lee, Eqs. 6.40 and 6.44 (``method="power"``), in which
:math:`k` is updated by the ratio of the total fission source of two
successive iterates.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .tables import ResultTables

__all__ = ["EigenvalueResult", "solve_eigenvalue"]


@dataclass
class EigenvalueResult(ResultTables):
    """The outcome of :meth:`dualmesh.Problem.solve_eigenvalue`: the
    effective multiplication factor, the convergence of the study, and the
    volume, the average flux of every group and the share of the fission
    neutron production of every region (element block).

    ``summary()`` is the report, ``tables`` holds the table of the regions
    (and the iterations of the power method), ``write_csv(path)`` writes the
    table of the regions (``table="iterations"`` the other), with the units
    in the column names, and ``write_json(path)`` writes everything."""

    k_effective: float
    method: str
    iterations: int  # the number of solves with the loss operator L
    num_dofs: int
    residual_norm: float
    normalization: float = 1.0
    history: list = field(default_factory=list)
    regions: list = field(default_factory=list)

    @property
    def reactivity(self) -> float:
        """The reactivity :math:`\\rho = (k - 1)/k`."""
        return (self.k_effective - 1.0) / self.k_effective

    @property
    def tables(self) -> dict:
        """``regions``: the volume, the average flux of every group and the
        fission neutron production of every region, and, for the power
        iteration, ``iterations``: the estimate of k at every iteration."""
        from .tables import Table

        out = {}
        if self.regions:
            groups = len(self.regions[0]["average_flux"])
            columns = (
                ["region", "volume"]
                + [f"average_flux_{g}" for g in range(1, groups + 1)]
                + ["fission_neutron_production", "production_share"]
            )
            units = ["", "m^3"] + ["1/(m^2 s)"] * groups + ["1/s", "%"]
            rows = [
                [
                    r["region"],
                    r["volume"],
                    *r["average_flux"],
                    r["production"],
                    100.0 * r["production"] / self.normalization,
                ]
                for r in self.regions
            ]
            out["regions"] = Table(
                columns,
                units,
                rows,
                title=f"Regions (fluxes for a total production of {self.normalization:g} fission neutron{'' if self.normalization == 1.0 else 's'} per second)",
            )
        if self.history:
            out["iterations"] = Table(
                ["iteration", "k_effective"],
                ["", ""],
                [[i + 1, k] for i, k in enumerate(self.history)],
                title="Power iteration",
            )
        return out

    def summary(self) -> str:
        from .tables import Table

        head = Table(
            ["quantity", "value"],
            None,
            [
                ["k_effective", f"{self.k_effective:.7f}"],
                ["reactivity (pcm)", f"{1e5 * self.reactivity:.1f}"],
                ["method", self.method],
                ["solves with L", self.iterations],
                ["unknowns", self.num_dofs],
                ["relative residual", f"{self.residual_norm:.2e}"],
            ],
            title="Eigenvalue study",
        )
        parts = [head.format()]
        if "regions" in self.tables:
            parts.append(self.tables["regions"].format())
        return "\n\n".join(parts)

    def __str__(self) -> str:
        return self.summary()

    def to_dict(self) -> dict:
        return {
            "study": "eigenvalue",
            "k_effective": self.k_effective,
            "reactivity": self.reactivity,
            "method": self.method,
            "iterations": self.iterations,
            "num_dofs": self.num_dofs,
            "residual_norm": self.residual_norm,
            "normalization": self.normalization,
            "regions": list(self.regions),
            "history": list(self.history),
        }

    def write_json(self, path) -> None:
        """Write :meth:`to_dict` to a JSON file."""
        from .parameters import write_json_numbers

        write_json_numbers(self.to_dict(), path)


def region_table(problem, physics) -> list[dict]:
    """Volume, average flux of every group and fission neutron production of
    every element block."""
    mesh = problem.mesh
    blocks = np.array([mesh.element_block(e) for e in range(mesh.num_elements)])
    names = dict(mesh.block_names())
    volumes = problem.element_volumes()
    integrals, production = {}, np.zeros(mesh.num_elements)
    for p in physics:
        for g, name in enumerate(p.variables(), start=1):
            integrals[name] = problem.element_integrals(name)
            production += (
                np.asarray(
                    problem.property_at_centroids(f"nu_fission_cross_section_{g}"), dtype=float
                ).ravel()
                * integrals[name]
            )
    out = []
    for block in sorted(set(blocks.tolist())):
        inside = blocks == block
        volume = float(volumes[inside].sum())
        out.append(
            dict(
                region=names.get(block, str(block)),
                volume=volume,
                average_flux=[float(integrals[n][inside].sum() / volume) for n in integrals],
                production=float(production[inside].sum()),
            )
        )
    return out


def fission_production(problem) -> float:
    r"""The total production rate of fission neutrons of the neutron
    diffusion physics of the problem, :math:`\int \sum_g \nu\Sigma_{f,g}
    \phi_g \, dV`, from the element integrals of the fluxes and the
    cross sections of every element."""
    total = 0.0
    for p in _neutron_physics(problem):
        for g, name in enumerate(p.variables(), start=1):
            cross_section = np.asarray(
                problem.property_at_centroids(f"nu_fission_cross_section_{g}"), dtype=float
            ).ravel()
            total += float(np.dot(cross_section, problem.element_integrals(name)))
    return total


def _neutron_physics(problem):
    from .physics import NeutronDiffusion

    found = [p for p in problem._physics.values() if isinstance(p, NeutronDiffusion)]
    if not found:
        raise ValueError(
            "solve_eigenvalue: the problem has no neutron_diffusion physics, whose fission source defines the eigenvalue."
        )
    return found


def solve_eigenvalue(
    problem,
    method: str = "krylov",
    tolerance: float = 1.0e-10,
    max_iterations: int = 2000,
    normalization: float = 1.0,
) -> EigenvalueResult:
    """See :meth:`dualmesh.Problem.solve_eigenvalue`."""
    import scipy.sparse.linalg as sparse_linalg

    if method not in ("krylov", "power"):
        raise ValueError(
            f"solve_eigenvalue: unknown method '{method}'. Use krylov (the Arnoldi method) or power (the power iteration)."
        )
    if problem.is_distributed:
        raise ValueError("solve_eigenvalue: the eigenvalue study runs on one process.")
    physics = _neutron_physics(problem)
    problem.initialize()
    n = len(problem.solution())
    for p in physics:
        p._fission_scale.set(0.0)
    _, loss = problem.linear_system()
    for p in physics:
        p._fission_scale.set(1.0)
    _, with_fission = problem.linear_system()
    fission = (loss - with_fission).tocsc()
    fission.eliminate_zeros()
    if fission.nnz == 0:
        raise ValueError(
            "solve_eigenvalue: the fission source is zero. Give a nu_fission_cross_section to a fissile region."
        )
    factor = sparse_linalg.splu(loss.tocsc())
    free = (
        np.asarray(abs(fission).sum(axis=0)).ravel() + np.asarray(abs(fission).sum(axis=1)).ravel()
        > 0.0
    )
    start = np.where(free, 1.0, 0.0)
    history = []
    if method == "krylov":
        solves = [0]

        def apply(v):
            solves[0] += 1
            return factor.solve(fission @ v)

        operator = sparse_linalg.LinearOperator((n, n), matvec=apply, dtype=float)
        values, vectors = sparse_linalg.eigs(
            operator, k=1, which="LM", v0=start, tol=tolerance, maxiter=max_iterations
        )
        k = float(values[0].real)
        phi = vectors[:, 0].real
        # The cost of the Arnoldi method is its number of solves with L.
        iterations = solves[0]
    else:
        # Lee, Eqs. 6.40 and 6.44: L phi_{i+1} = F phi_i / k_i, and k_{i+1}
        # = k_i (sum F phi_{i+1}) / (sum F phi_i).
        phi, k = start, 1.0
        source = fission @ phi
        iterations = 0
        for _ in range(max_iterations):
            iterations += 1
            phi_new = factor.solve(source / k)
            source_new = fission @ phi_new
            k_new = k * source_new.sum() / source.sum()
            history.append(float(k_new))
            change = abs(k_new - k) / abs(k_new)
            point = np.max(np.abs(source_new / source_new.sum() - source / source.sum())) / max(
                np.max(np.abs(source_new / source_new.sum())), 1e-300
            )
            phi, source, k = phi_new, source_new, k_new
            if change < tolerance and point < np.sqrt(tolerance):
                break
        else:
            raise RuntimeError(
                f"solve_eigenvalue: the power iteration did not converge in {max_iterations} iterations (last k = {k:.8f})."
            )
    if phi.sum() < 0.0:
        phi = -phi
    residual = np.linalg.norm(loss @ phi - fission @ phi / k) / max(
        np.linalg.norm(fission @ phi / k), 1e-300
    )
    names = list(problem._problem.variable_names())
    count = len(names)
    for v, name in enumerate(names):
        if any(name in p.variables() for p in physics):
            problem.set_values(name, phi[v::count])
    total = fission_production(problem)
    for p in physics:
        for name in p.variables():
            problem.set_values(name, problem.values(name) * (normalization / total))
    return EigenvalueResult(
        k_effective=k,
        method=method,
        iterations=iterations,
        num_dofs=int(n),
        residual_norm=float(residual),
        normalization=float(normalization),
        history=history,
        regions=region_table(problem, physics),
    )
