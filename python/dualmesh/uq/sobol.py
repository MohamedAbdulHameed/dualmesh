# SPDX-License-Identifier: LGPL-2.1-or-later
r"""Variance-based (Sobol') sensitivity indices.

The first-order index :math:`S_i = V[E(Y|X_i)]/V(Y)` is the fraction of the
output variance that input :math:`i` explains alone, and the total index
:math:`S_{Ti} = E[V(Y|X_{\sim i})]/V(Y)` the fraction it explains with all
its interactions (Sobol' 2001, Definition 3).  Both come from two
independent sample matrices :math:`A` and :math:`B` of :math:`n` rows and the
:math:`k` matrices :math:`A_B^{(i)}`, equal to :math:`A` with column
:math:`i` taken from :math:`B` (Saltelli et al. 2010, Table 2, estimators (b)
and (f)):

.. math::

    S_i \approx \frac{1}{V}\,\frac1n\sum_j f(B)_j \left[f(A_B^{(i)})_j -
    f(A)_j\right], \qquad
    S_{Ti} \approx \frac{1}{2V}\,\frac1n\sum_j \left[f(A)_j -
    f(A_B^{(i)})_j\right]^2,

the second being the estimator of Jansen (1999), Sect. 2.3.  This costs
:math:`n(k+2)` runs.  The second-order indices need the matrices
:math:`B_A^{(i)}` as well, :math:`n(2k+2)` runs in all: the closed index
:math:`V^c_{ij}` is the mean of :math:`f(B_A^{(i)}) f(A_B^{(j)})` less
:math:`E^2(y)`, and :math:`S_{ij} = (V^c_{ij} - V_i - V_j)/V` (Saltelli 2002,
Theorem 2 and Sect. 5).  The intervals come from
bootstrap resampling of the :math:`n` rows.  With a surrogate, the indices
are those of its mean, and the intervals also include the uncertainty of
the surrogate itself, by computing the indices of random realisations of the
Gaussian process (Marrel et al. 2009).
"""

from __future__ import annotations

import itertools

import numpy as np

from ._engine import check_inputs, evaluate, to_inputs, unit_design, write_json

__all__ = ["SobolIndices", "sobol"]


def _indices(fa, fb, fab, fba=None):
    """First, total and (optionally) second-order indices from the model
    values: fa, fb (n, m), fab, fba (k, n, m).  Returns arrays (k, m) and
    (k, k, m)."""
    y = np.concatenate([fa, fb])
    v = y.var(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        first = np.mean(fb[None] * (fab - fa[None]), axis=1) / v
        total = 0.5 * np.mean((fa[None] - fab) ** 2, axis=1) / v
        second = None
        if fba is not None:
            k = fab.shape[0]
            second = np.full((k, k) + fa.shape[1:], np.nan)
            base = np.mean(fa * fb, axis=0)
            for i, j in itertools.combinations(range(k), 2):
                vij = (np.mean(fba[i] * fab[j], axis=0) - base) / v
                second[i, j] = second[j, i] = vij - first[i] - first[j]
    return first, total, second


class SobolIndices:
    """Sobol' indices of every output.

    Attributes
    ----------
    first_order, total : dict
        Output name -> {input name -> index} (an array for a vector output).
    first_order_interval, total_interval : dict
        Output name -> {input name -> (low, high)}.
    second_order : dict or None
        Output name -> {(input a, input b) -> index}, when requested.
    runs : int
        Number of model runs.
    """

    def __init__(self, names, outputs, first, total, first_ci, total_ci, second, runs, level):
        self.names = names
        self.runs = runs
        self.level = level
        self.first_order, self.total = {}, {}
        self.first_order_interval, self.total_interval = {}, {}
        self.second_order = {} if second is not None else None
        for o, shape in outputs.items():

            def val(a, shape=shape):
                return float(a[0]) if shape == () else a.reshape(shape)

            self.first_order[o] = {n: val(first[o][j]) for j, n in enumerate(names)}
            self.total[o] = {n: val(total[o][j]) for j, n in enumerate(names)}
            self.first_order_interval[o] = {n: (val(first_ci[o][0][j]), val(first_ci[o][1][j])) for j, n in enumerate(names)}
            self.total_interval[o] = {n: (val(total_ci[o][0][j]), val(total_ci[o][1][j])) for j, n in enumerate(names)}
            if second is not None:
                self.second_order[o] = {(a, b): val(second[o][i, j]) for (i, a), (j, b) in itertools.combinations(enumerate(names), 2)}

    def summary(self) -> str:
        """A table of the indices of every scalar output."""
        lines = [f"Sobol' indices from {self.runs} model runs ({self.level:.0%} intervals)"]
        for o in self.first_order:
            vals = self.first_order[o]
            if not isinstance(next(iter(vals.values())), float):
                lines.append(f"\n{o}: vector output (see first_order['{o}'])")
                continue
            lines.append(f"\n{o}")
            lines.append(f"  {'input':<28}{'first order':>26}{'total':>26}")
            order = sorted(self.names, key=lambda n: -np.nan_to_num(self.total[o][n]))
            for n in order:
                f, (fl, fh) = self.first_order[o][n], self.first_order_interval[o][n]
                t, (tl, th) = self.total[o][n], self.total_interval[o][n]
                lines.append(f"  {n:<28}{f:7.3f} [{fl:6.3f}, {fh:6.3f}]   {t:7.3f} [{tl:6.3f}, {th:6.3f}]")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """The first-order and total indices of every output with their
        intervals, and the second-order indices when they were computed."""
        numbers = {"study": "sobol", "runs": self.runs, "level": self.level, "first_order": self.first_order, "first_order_interval": self.first_order_interval, "total": self.total, "total_interval": self.total_interval}
        if self.second_order is not None:
            numbers["second_order"] = {o: {f"{a},{b}": v for (a, b), v in pairs.items()} for o, pairs in self.second_order.items()}
        return numbers

    def write_json(self, path):
        """Write :meth:`to_dict` to a JSON file (a non-finite value is written
        as null)."""
        write_json(self.to_dict(), path)

    @property
    def tables(self) -> dict:
        """``indices``: one row per scalar output and input, with the
        first-order and total indices and their intervals."""
        from ..tables import Table

        rows = []
        for o in self.first_order:
            for n in self.names:
                if not isinstance(self.first_order[o][n], float):
                    continue
                (fl, fh), (tl, th) = self.first_order_interval[o][n], self.total_interval[o][n]
                rows.append([o, n, self.first_order[o][n], float(fl), float(fh), self.total[o][n], float(tl), float(th)])
        return {"indices": Table(["output", "input", "first_order", "first_order_low", "first_order_high", "total", "total_low", "total_high"], None, rows, title=f"Sobol' indices from {self.runs} model runs")}

    def write_csv(self, path, table: str | None = None):
        """Write the table ``indices`` to a CSV file."""
        from ..tables import ResultTables

        ResultTables.write_csv(self, path, table)

    def __repr__(self):
        return f"SobolIndices({self.runs} runs, outputs: {', '.join(self.first_order)})"


def _bootstrap(fa, fb, fab, resamples, rng, level):
    n = fa.shape[0]
    reps_f, reps_t = [], []
    for _ in range(resamples):
        i = rng.integers(0, n, n)
        f, t, _ = _indices(fa[i], fb[i], fab[:, i])
        reps_f.append(f)
        reps_t.append(t)
    a = 50.0 * (1.0 - level)
    rf, rt = np.array(reps_f), np.array(reps_t)
    return ((np.nanpercentile(rf, a, axis=0), np.nanpercentile(rf, 100 - a, axis=0)), (np.nanpercentile(rt, a, axis=0), np.nanpercentile(rt, 100 - a, axis=0)), rf, rt)


def sobol(model, inputs, samples, seed=0, processes=1, store=None, surrogate=None, second_order=False, training_samples=None, training=None, realizations=100, level=0.95, resamples=500, progress=False) -> SobolIndices:
    r"""First-order and total Sobol' indices of every output of ``model``.

    ``samples``
        :math:`n`, the rows of each sample matrix.  The model runs
        :math:`n(k+2)` times, or :math:`n(2k+2)` with ``second_order``.  A
        power of 2 suits the Sobol' sequence the matrices are drawn from.
    ``surrogate``
        None (run the model) or ``"gaussian_process"``: run the model at
        ``training_samples`` Latin hypercube points (default :math:`10k`,
        at least 50), or take the given ``training`` runs of
        :func:`propagate`, fit a :class:`GaussianProcess`, and compute the
        indices on it.  ``realizations`` random functions of the process give
        its contribution to the intervals.  Only one of ``training_samples``
        and ``training`` is given.
    ``processes``, ``store``, ``progress``
        As for :func:`propagate`.
    """
    dists = check_inputs(inputs)
    names = list(dists)
    k = len(names)
    n = int(samples)
    if n < 2:
        raise ValueError("sobol: samples must be at least 2.")
    if surrogate is None and (training is not None or training_samples is not None):
        raise ValueError("sobol: training and training_samples are used only with surrogate='gaussian_process'.")
    u = unit_design(n, 2 * k, "sobol", seed)
    ua, ub = u[:, :k], u[:, k:]
    uab = np.repeat(ua[None], k, axis=0)
    for i in range(k):
        uab[i, :, i] = ub[:, i]
    uba = None
    if second_order:
        uba = np.repeat(ub[None], k, axis=0)
        for i in range(k):
            uba[i, :, i] = ua[:, i]
    blocks = [ua, ub] + list(uab) + (list(uba) if second_order else [])
    uall = np.concatenate(blocks)

    gp = None
    if surrogate is None:
        xall = to_inputs(uall, dists)
        outputs, _ = evaluate(model, names, xall, processes, store, progress)
        runs = len(xall)
    elif surrogate == "gaussian_process":
        from .gaussian_process import GaussianProcess
        from .propagate import propagate

        if training is not None and training_samples is not None:
            raise ValueError("sobol: give training_samples or training, not both.")
        if training is None:
            m = int(training_samples) if training_samples else max(10 * k, 50)
            training = propagate(model, dists, m, "latin_hypercube", seed, processes, store, progress)
        if list(training.distributions) != names:
            raise ValueError("sobol: the training runs must have the same inputs, in order.")
        m = training.successful
        gp = GaussianProcess(seed=seed).fit(training)
        pred = gp.predict(to_inputs(uall, dists))
        outputs = pred if isinstance(pred, dict) else {gp.output_names[0]: pred}
        runs = m
    else:
        raise ValueError("sobol: surrogate must be None or 'gaussian_process'.")

    rng = np.random.default_rng(seed)
    first, total, second, fci, tci, shapes = {}, {}, {}, {}, {}, {}
    nb = 2 + k
    for o, y in outputs.items():
        shapes[o] = y.shape[1:]
        f = y.reshape(len(y), -1).reshape(len(blocks), n, -1)
        ok = np.all(np.isfinite(f), axis=(0, 2))
        f = f[:, ok]
        fa, fb, fab = f[0], f[1], f[2:nb]
        fba = f[nb:] if second_order else None
        first[o], total[o], second[o] = _indices(fa, fb, fab, fba)
        (fl, fh), (tl, th), rf, rt = _bootstrap(fa, fb, fab, resamples, rng, level)
        if gp is not None:
            # Realisations of the surrogate on a sub-design, each with a
            # bootstrap resample, give the combined interval.
            ns = min(len(fa), 256)
            sub = np.concatenate([b[:ns] for b in blocks[:nb]])
            real = gp.sample(to_inputs(sub, dists), realizations, seed)
            real = real[o] if isinstance(real, dict) else real
            real = real.reshape(realizations, nb, ns, -1)
            reps_f, reps_t = [], []
            for r in range(realizations):
                i = rng.integers(0, ns, ns)
                fr, tr, _ = _indices(real[r, 0, i], real[r, 1, i], real[r, 2:, i].swapaxes(0, 1))
                reps_f.append(fr)
                reps_t.append(tr)
            a = 50.0 * (1.0 - level)
            rf, rt = np.array(reps_f), np.array(reps_t)
            fl, fh = np.nanpercentile(rf, a, axis=0), np.nanpercentile(rf, 100 - a, axis=0)
            tl, th = np.nanpercentile(rt, a, axis=0), np.nanpercentile(rt, 100 - a, axis=0)
        fci[o], tci[o] = (fl, fh), (tl, th)
    return SobolIndices(names, shapes, first, total, fci, tci, second if second_order else None, runs, level)
