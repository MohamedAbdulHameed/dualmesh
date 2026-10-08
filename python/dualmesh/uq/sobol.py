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

from ._engine import check_inputs, check_output_transform, evaluate, study_header, surrogate_mean, surrogate_sample, to_inputs, transformed_runs, unit_design, write_json

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
    first_order_global_process, total_global_process : dict or None
        With a surrogate: the indices of the global process (Marrel et al.
        2009, Eq. 12), which add the error of the surrogate to the variance
        of every effect, in the form of ``first_order``.  None without a
        surrogate.
    first_order_global_process_interval, total_global_process_interval : dict or None
        Their intervals (the distribution of Eq. 11, with a bootstrap of the
        rows).
    runs : int
        Number of model runs.
    """

    def __init__(self, names, outputs, first, total, first_ci, total_ci, second, runs, level, process=None):
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
        self.first_order_global_process = self.total_global_process = None
        self.first_order_global_process_interval = self.total_global_process_interval = None
        if process is not None:
            self.first_order_global_process, self.total_global_process = {}, {}
            self.first_order_global_process_interval, self.total_global_process_interval = {}, {}
            for o, shape in outputs.items():

                def value(a, shape=shape):
                    return float(a[0]) if shape == () else a.reshape(shape)

                self.first_order_global_process[o] = {n: value(process["first"][o][j]) for j, n in enumerate(names)}
                self.total_global_process[o] = {n: value(process["total"][o][j]) for j, n in enumerate(names)}
                self.first_order_global_process_interval[o] = {n: (value(process["first_interval"][o][0][j]), value(process["first_interval"][o][1][j])) for j, n in enumerate(names)}
                self.total_global_process_interval[o] = {n: (value(process["total_interval"][o][0][j]), value(process["total_interval"][o][1][j])) for j, n in enumerate(names)}

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
            if self.first_order_global_process is not None:
                lines.append("  global process (Marrel et al. 2009, Eq. 12):")
                for n in order:
                    f, (fl, fh) = self.first_order_global_process[o][n], self.first_order_global_process_interval[o][n]
                    t, (tl, th) = self.total_global_process[o][n], self.total_global_process_interval[o][n]
                    lines.append(f"  {n:<28}{f:7.3f} [{fl:6.3f}, {fh:6.3f}]   {t:7.3f} [{tl:6.3f}, {th:6.3f}]")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """The first-order and total indices of every output with their
        intervals, and the second-order indices when they were computed."""
        numbers = {"study": "sobol", "runs": self.runs, "level": self.level, "first_order": self.first_order, "first_order_interval": self.first_order_interval, "total": self.total, "total_interval": self.total_interval}
        if self.second_order is not None:
            numbers["second_order"] = {o: {f"{a},{b}": v for (a, b), v in pairs.items()} for o, pairs in self.second_order.items()}
        if self.first_order_global_process is not None:
            numbers.update(first_order_global_process=self.first_order_global_process, first_order_global_process_interval=self.first_order_global_process_interval, total_global_process=self.total_global_process, total_global_process_interval=self.total_global_process_interval)
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
                row = [o, n, self.first_order[o][n], float(fl), float(fh), self.total[o][n], float(tl), float(th)]
                if self.first_order_global_process is not None:
                    (gfl, gfh), (gtl, gth) = self.first_order_global_process_interval[o][n], self.total_global_process_interval[o][n]
                    row += [self.first_order_global_process[o][n], float(gfl), float(gfh), self.total_global_process[o][n], float(gtl), float(gth)]
                rows.append(row)
        columns = ["output", "input", "first_order", "first_order_low", "first_order_high", "total", "total_low", "total_high"]
        if self.first_order_global_process is not None:
            columns += ["first_order_global_process", "first_order_global_process_low", "first_order_global_process_high", "total_global_process", "total_global_process_low", "total_global_process_high"]
        return {"indices": Table(columns, None, rows, title=f"Sobol' indices from {self.runs} model runs")}

    def write_csv(self, path, table: str | None = None):
        """Write the table ``indices`` to a CSV file."""
        from ..tables import ResultTables

        ResultTables.write_csv(self, path, table)

    def __repr__(self):
        return f"SobolIndices({self.runs} runs, outputs: {', '.join(self.first_order)})"


def _numerators(fa, fb, fab):
    """The numerators of the first-order (Saltelli 2010) and total (Jansen)
    estimators and the variance, from the model values fa, fb (n, m) and
    fab (k, n, m).  All three are quadratic forms of the values."""
    y = np.concatenate([fa, fb])
    first = np.mean(fb[None] * (fab - fa[None]), axis=1)
    total = 0.5 * np.mean((fa[None] - fab) ** 2, axis=1)
    return first, total, y.var(axis=0)


def _surrogate_intervals(mean, realizations, rng, level):
    """The intervals of the first-order and total indices of a Gaussian
    process surrogate, which hold the two errors of the estimate:

    * the Monte Carlo error of the estimate on the mean of the process, by a
      bootstrap of the n rows of the sample matrices;
    * the error of the surrogate, by the spread of the indices of the random
      functions of the process around their own mean (Marrel et al. 2009,
      Sect. 3.3), on the first ns rows.

    Each random function adds its deviation to one bootstrap estimate, so
    the interval is centred on the estimate of the mean of the process.
    ``mean`` (k+2, n, m) is the mean of the process on the sample matrices
    A, B and A_B^(i), and ``realizations`` (R, k+2, ns, m) are random
    functions on their first ns rows."""
    n = mean.shape[1]
    with np.errstate(invalid="ignore", divide="ignore"):
        functions = []
        for real in realizations:
            r_first, r_total, r_var = _numerators(real[0], real[1], real[2:])
            functions.append((r_first / r_var, r_total / r_var))
        f_first = np.array([f for f, _ in functions])
        f_total = np.array([t for _, t in functions])
        d_first, d_total = f_first - np.nanmean(f_first, axis=0), f_total - np.nanmean(f_total, axis=0)
        reps_f, reps_t = [], []
        for r in range(len(realizations)):
            i = rng.integers(0, n, n)
            b_first, b_total, b_var = _numerators(mean[0, i], mean[1, i], mean[2:, i])
            reps_f.append(b_first / b_var + d_first[r])
            reps_t.append(b_total / b_var + d_total[r])
    a = 50.0 * (1.0 - level)
    rf, rt = np.array(reps_f), np.array(reps_t)
    # A total index is not negative, so neither is the low end of its
    # interval (the deviations of the random functions can reach below zero
    # for an index near zero).
    return (np.nanpercentile(rf, a, axis=0), np.nanpercentile(rf, 100 - a, axis=0)), (np.maximum(np.nanpercentile(rt, a, axis=0), 0.0), np.nanpercentile(rt, 100 - a, axis=0))


def _global_process_indices(mean, realizations, rng, level):
    """The first-order and total indices of the global process (Marrel et
    al. 2009, Eqs. 11 and 12): the expectation over the random functions of
    the process of the variance of each effect, divided by the expectation
    of the variance of the output, and the interval of the distribution of
    Eq. (11) with a bootstrap of the rows.

    Every numerator is a quadratic form of the values, so its expectation is
    its value for the mean of the process plus its expectation for the
    deviation from the mean.  The mean part takes the n rows of ``mean``
    (k+2, n, m), and the deviation part the ns rows of ``realizations``
    (R, k+2, ns, m)."""
    ns = realizations.shape[2]
    m_first, m_total, m_var = _numerators(mean[0], mean[1], mean[2:])
    sub = mean[:, :ns]
    s_first, s_total, s_var = _numerators(sub[0], sub[1], sub[2:])
    deviations = []
    for real in realizations:
        r_first, r_total, r_var = _numerators(real[0], real[1], real[2:])
        deviations.append((r_first - s_first, r_total - s_total, r_var - s_var))
    d_first = np.mean([d[0] for d in deviations], axis=0)
    d_total = np.mean([d[1] for d in deviations], axis=0)
    variance = m_var + np.mean([d[2] for d in deviations], axis=0)
    n = mean.shape[1]
    with np.errstate(invalid="ignore", divide="ignore"):
        first = (m_first + d_first) / variance
        total = (m_total + d_total) / variance
        reps_f, reps_t = [], []
        for real in realizations:
            i, j = rng.integers(0, n, n), rng.integers(0, ns, ns)
            b_first, b_total, _ = _numerators(mean[0, i], mean[1, i], mean[2:, i])
            r_first, r_total, _ = _numerators(real[0, j], real[1, j], real[2:, j])
            p_first, p_total, _ = _numerators(sub[0, j], sub[1, j], sub[2:, j])
            reps_f.append((b_first + r_first - p_first) / variance)
            reps_t.append((b_total + r_total - p_total) / variance)
    a = 50.0 * (1.0 - level)
    rf, rt = np.array(reps_f), np.array(reps_t)
    return first, total, (np.nanpercentile(rf, a, axis=0), np.nanpercentile(rf, 100 - a, axis=0)), (np.nanpercentile(rt, a, axis=0), np.nanpercentile(rt, 100 - a, axis=0))


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


def sobol(model, inputs, samples, seed=0, processes=1, store=None, surrogate=None, second_order=False, training_samples=None, training=None, realizations=100, level=0.95, resamples=500, output_transform=None, report="full") -> SobolIndices:
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
        indices on the mean of the process.  The intervals hold the Monte
        Carlo error of the estimate (a bootstrap of the rows) and the error of
        the surrogate (the spread of the indices of ``realizations`` random
        functions of the process around their mean, on the first 256 rows of
        the sample matrices), and they are centred on the estimate.  Only one
        of ``training_samples`` and ``training`` is given.
    ``output_transform``
        Only with a surrogate: output name -> ``"log"``.  The process models
        the logarithm of the output, which suits a positive output that
        changes by factors (a fission gas release), and the indices are
        those of the output itself.  Default None: the process models every
        output as it is.
    ``processes``, ``store``, ``report``
        As for :func:`propagate`.
    """
    from ..console import report_level

    report = report_level(report, "uq.sobol")
    progress = report == "full"
    dists = check_inputs(inputs)
    study_header(report, "Sobol' sensitivity study", dists)
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
        if output_transform:
            raise ValueError("sobol: output_transform applies to a surrogate. Give surrogate='gaussian_process', or remove output_transform.")
        xall = to_inputs(uall, dists)
        outputs, _ = evaluate(model, names, xall, processes, store, progress)
        runs = len(xall)
    elif surrogate == "gaussian_process":
        from .gaussian_process import GaussianProcess
        from .propagate import _propagate

        if training is not None and training_samples is not None:
            raise ValueError("sobol: give training_samples or training, not both.")
        if training is None:
            m = int(training_samples) if training_samples else max(10 * k, 50)
            training = _propagate(model, dists, m, "latin_hypercube", seed, processes, store, progress)
        if list(training.distributions) != names:
            raise ValueError("sobol: the training runs must have the same inputs, in order.")
        m = training.successful
        transform = check_output_transform(output_transform, training, "sobol")
        gp = GaussianProcess(seed=seed).fit(transformed_runs(training, transform))
        outputs = surrogate_mean(gp, to_inputs(uall, dists), transform)
        ns = min(n, 256)
        sub_design = np.concatenate([b[:ns] for b in blocks[: 2 + k]])
        sampled = surrogate_sample(gp, to_inputs(sub_design, dists), realizations, seed, transform)
        runs = m
    else:
        raise ValueError("sobol: surrogate must be None or 'gaussian_process'.")

    rng = np.random.default_rng(seed)
    first, total, second, fci, tci, shapes = {}, {}, {}, {}, {}, {}
    process = {"first": {}, "total": {}, "first_interval": {}, "total_interval": {}} if gp is not None else None
    nb = 2 + k
    for o, y in outputs.items():
        shapes[o] = y.shape[1:]
        f = y.reshape(len(y), -1).reshape(len(blocks), n, -1)
        ok = np.all(np.isfinite(f), axis=(0, 2))
        f = f[:, ok]
        fa, fb, fab = f[0], f[1], f[2:nb]
        fba = f[nb:] if second_order else None
        first[o], total[o], second[o] = _indices(fa, fb, fab, fba)
        if gp is None:
            (fl, fh), (tl, th), _, _ = _bootstrap(fa, fb, fab, resamples, rng, level)
        else:
            real = sampled[o].reshape(realizations, nb, ns, -1)
            (fl, fh), (tl, th) = _surrogate_intervals(f[:nb], real, rng, level)
            g_first, g_total, g_fci, g_tci = _global_process_indices(f[:nb], real, rng, level)
            process["first"][o], process["total"][o], process["first_interval"][o], process["total_interval"][o] = g_first, g_total, g_fci, g_tci
        fci[o], tci[o] = (fl, fh), (tl, th)
    indices = SobolIndices(names, shapes, first, total, fci, tci, second if second_order else None, runs, level, process)
    if report != "none":
        print(indices.summary())
    return indices
