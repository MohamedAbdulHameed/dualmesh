# SPDX-License-Identifier: LGPL-2.1-or-later
r"""Forward propagation of input uncertainty through a model, the statistics
of the outputs, Wilks tolerance limits and correlation-based sensitivity."""

from __future__ import annotations

import numpy as np
from scipy import stats

from ._engine import check_inputs, evaluate, to_inputs, unit_design

__all__ = ["Runs", "propagate", "wilks_samples"]


def _rank(a, axis=0):
    return stats.rankdata(a, axis=axis)


def _corr_columns(x, y):
    """Pearson correlation of every column of ``x`` (n, k) with every
    column of ``y`` (n, m): an array (k, m)."""
    xc = x - x.mean(axis=0)
    yc = y - y.mean(axis=0)
    sx = np.sqrt((xc * xc).sum(axis=0))
    sy = np.sqrt((yc * yc).sum(axis=0))
    with np.errstate(invalid="ignore", divide="ignore"):
        return (xc.T @ yc) / np.outer(sx, sy)


def _prcc(x, y):
    """Partial rank correlation coefficients (Marino et al. 2008): for each
    input, the correlation of the ranks of the input and of the output once
    the linear effect of the ranks of the other inputs is removed from both.
    ``x`` (n, k), ``y`` (n, m); returns (k, m)."""
    rx, ry = _rank(x), _rank(y)
    n, k = rx.shape
    out = np.empty((k, ry.shape[1]))
    for i in range(k):
        others = np.column_stack([np.ones(n), np.delete(rx, i, axis=1)])
        coef, *_ = np.linalg.lstsq(others, np.column_stack([rx[:, i], ry]), rcond=None)
        res = np.column_stack([rx[:, i], ry]) - others @ coef
        out[i] = _corr_columns(res[:, :1], res[:, 1:])[0]
    return out


def _binom_cdf(k, n, p):
    return float(stats.binom.cdf(k, n, p))


def wilks_samples(coverage=0.95, confidence=0.95, order=1, side="upper") -> int:
    r"""The smallest number of runs for a Wilks (1941) tolerance limit.

    With :math:`n` independent runs, the :math:`r`-th largest output is an
    upper limit that covers at least the fraction :math:`\gamma`
    (``coverage``) of the output distribution with probability

    .. math:: P = \sum_{j=0}^{n-r} \binom{n}{j} \gamma^j (1-\gamma)^{n-j},

    whatever the distribution.  For a two-sided interval between the
    :math:`r`-th smallest and the :math:`r`-th largest, the sum ends at
    :math:`n - 2r`.  The result is the smallest :math:`n` with :math:`P \ge
    \beta` (``confidence``): 59 runs for one-sided and 93 for two-sided
    95/95 limits of the first order.
    """
    _check_side(side)
    if not (0 < coverage < 1 and 0 < confidence < 1) or order < 1:
        raise ValueError("wilks_samples: need 0 < coverage, confidence < 1 and order >= 1.")
    m = 2 * order if side == "two" else order
    n = m
    while _binom_cdf(n - m, n, coverage) < confidence:
        n += 1
    return n


def _check_side(side):
    if side not in ("upper", "lower", "two"):
        raise ValueError("side must be upper, lower or two.")


class Runs:
    """The runs of a propagation study: the inputs of every run, the
    outputs, and their statistics.

    Attributes
    ----------
    inputs : dict
        Input name -> array of the values of every run.
    outputs : dict
        Output name -> array with one row per run (NaN for failed runs).
    failed : dict
        Run index -> error message of the runs that failed.
    distributions : dict
        Input name -> distribution.
    method : str
        The design: latin_hypercube, sobol or monte_carlo.
    """

    def __init__(self, distributions, x, u, outputs, failed, method):
        self.distributions = dict(distributions)
        self.names = list(self.distributions)
        self.x = np.asarray(x, dtype=float)
        self.u = np.asarray(u, dtype=float)
        self.inputs = {name: self.x[:, j] for j, name in enumerate(self.names)}
        self.outputs = outputs
        self.failed = failed
        self.method = method

    # ---- helpers ----------------------------------------------------------
    def __len__(self):
        return len(self.x)

    def _values(self, name):
        if name not in self.outputs:
            raise KeyError(f"Runs: no output '{name}'. Outputs: {', '.join(self.outputs)}.")
        y = self.outputs[name]
        ok = np.all(np.isfinite(y.reshape(len(y), -1)), axis=1)
        return y[ok], ok

    @property
    def successful(self) -> int:
        """Number of runs that succeeded."""
        return len(self.x) - len(self.failed)

    # ---- statistics ---------------------------------------------------------
    def mean(self, name):
        """Sample mean of an output (per component for a vector output)."""
        return self._values(name)[0].mean(axis=0)

    def std(self, name):
        """Sample standard deviation (with :math:`n-1`; NaN for one run)."""
        y = self._values(name)[0]
        if len(y) < 2:
            return np.full(y.shape[1:], np.nan) if y.ndim > 1 else np.nan
        return y.std(axis=0, ddof=1)

    def percentile(self, name, q):
        """Percentile ``q`` (0 to 100) of an output."""
        return np.percentile(self._values(name)[0], q, axis=0)

    def interval(self, name, level=0.95):
        """The central interval of the output distribution holding the
        fraction ``level`` of the runs: ``(low, high)``."""
        a = 50.0 * (1.0 - level)
        y = self._values(name)[0]
        return np.percentile(y, a, axis=0), np.percentile(y, 100.0 - a, axis=0)

    def bootstrap(self, name, statistic="mean", level=0.95, resamples=2000, seed=0):
        """A statistic of an output with its bootstrap confidence interval
        (percentile method, Efron and Tibshirani 1993): ``(value, low,
        high)``.  ``statistic`` is mean, std, median or a function of an
        array of runs (first axis) that returns one value per component."""
        funcs = {
            "mean": lambda a: a.mean(axis=0),
            "std": lambda a: a.std(axis=0, ddof=1),
            "median": lambda a: np.median(a, axis=0),
        }
        f = funcs.get(statistic, statistic) if isinstance(statistic, str) else statistic
        if isinstance(f, str) or not callable(f):
            raise ValueError("bootstrap: statistic must be mean, std, median or a function.")
        y = self._values(name)[0]
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, len(y), size=(resamples, len(y)))
        reps = np.array([f(y[i]) for i in idx])
        a = 50.0 * (1.0 - level)
        return f(y), np.percentile(reps, a, axis=0), np.percentile(reps, 100.0 - a, axis=0)

    def tolerance_limit(self, name, coverage=0.95, confidence=0.95, side="upper"):
        """Distribution-free Wilks tolerance limit of an output: with
        probability ``confidence``, at least the fraction ``coverage`` of the
        output distribution lies below (``side="upper"``), above
        (``"lower"``) or between (``"two"``, a pair) the returned values.
        The highest order that meets the confidence is used, which gives
        the tightest limit.  The runs must be simple random samples
        (``method="monte_carlo"``)."""
        _check_side(side)
        if self.method != "monte_carlo":
            raise ValueError(
                "tolerance_limit: Wilks limits need independent random runs; propagate with "
                "method='monte_carlo'."
            )
        y = np.sort(self._values(name)[0], axis=0)
        n = len(y)
        per = 2 if side == "two" else 1
        if _binom_cdf(n - per, n, coverage) < confidence:
            need = wilks_samples(coverage, confidence, 1, side)
            raise ValueError(
                f"tolerance_limit: {n} successful runs are too few for a {coverage:.0%}/"
                f"{confidence:.0%} {side} limit; at least {need} are needed."
            )
        r = 1
        while _binom_cdf(n - per * (r + 1), n, coverage) >= confidence:
            r += 1
        if side == "upper":
            return y[n - r]
        if side == "lower":
            return y[r - 1]
        return y[r - 1], y[n - r]

    def sensitivity(self, name, method="spearman"):
        """Correlation of each input with an output: ``spearman`` (rank
        correlation, for monotonic effects), ``pearson`` (linear) or
        ``prcc`` (partial rank correlation, the rank correlation once the
        other inputs are accounted for).  Returns a dict input name ->
        coefficient (an array for a vector output)."""
        y, ok = self._values(name)
        x = self.x[ok]
        shape = y.shape[1:]
        y2 = y.reshape(len(y), -1)
        if method == "pearson":
            c = _corr_columns(x, y2)
        elif method == "spearman":
            c = _corr_columns(_rank(x), _rank(y2))
        elif method == "prcc":
            c = _prcc(x, y2)
        else:
            raise ValueError("sensitivity: method must be spearman, pearson or prcc.")
        return {
            n: (float(c[j, 0]) if shape == () else c[j].reshape(shape))
            for j, n in enumerate(self.names)
        }

    # ---- reporting --------------------------------------------------------
    def summary(self, level=0.95) -> str:
        """A table of every output: mean and standard deviation with their
        bootstrap intervals, the central interval, and the two inputs of
        largest rank correlation."""
        lines = [
            f"{len(self)} runs ({self.method}), {self.successful} successful, "
            f"{len(self.failed)} failed",
            "",
        ]
        for name, y in self.outputs.items():
            if y.ndim > 1:
                m = self.mean(name)
                s = self.std(name)
                lines.append(
                    f"{name}: shape {y.shape[1:]}, mean {np.nanmin(m):.4g} to {np.nanmax(m):.4g}, "
                    f"largest std {np.nanmax(s):.4g}"
                )
                continue
            m, ml, mh = self.bootstrap(name, "mean", level)
            s, sl, sh = self.bootstrap(name, "std", level)
            lo, hi = self.interval(name, level)
            rho = self.sensitivity(name)
            top = sorted(rho.items(), key=lambda kv: -abs(kv[1]) if np.isfinite(kv[1]) else 0)
            top_s = ", ".join(f"{k} {v:+.2f}" for k, v in top[:2])
            lines.append(
                f"{name}: mean {m:.4g} [{ml:.4g}, {mh:.4g}], std {s:.4g} [{sl:.4g}, {sh:.4g}], "
                f"{level:.0%} of runs in [{lo:.4g}, {hi:.4g}]; Spearman: {top_s}"
            )
        if self.failed:
            first = next(iter(self.failed.values())).strip().splitlines()[-1]
            lines += ["", f"first failure: {first}"]
        return "\n".join(lines)

    def __repr__(self):
        return f"Runs({len(self)} runs, outputs: {', '.join(self.outputs)})"

    def save(self, path):
        """Write the inputs, the design and the outputs to a ``.npz`` file."""
        np.savez(
            path,
            input_names=np.array(self.names),
            x=self.x,
            u=self.u,
            method=np.array(self.method),
            failed=np.array(sorted(self.failed), dtype=int),
            output_names=np.array(list(self.outputs)),
            **{"output:" + k: v for k, v in self.outputs.items()},
        )


def propagate(
    model,
    inputs,
    samples,
    method="latin_hypercube",
    seed=0,
    processes=1,
    store=None,
    progress=False,
) -> Runs:
    r"""Run ``model`` at ``samples`` points drawn from the ``inputs`` and
    return the :class:`Runs`.

    ``model`` is any function of keyword arguments, one per input, that
    returns a number, an array, or a dict of name -> number or array.
    ``inputs`` is a dict of input name -> distribution.

    ``method``
        ``latin_hypercube`` (default: one run in each of ``samples``
        equal-probability intervals of every input), ``sobol`` (a scrambled
        Sobol' sequence, whose means converge faster for smooth models;
        best with a power of 2) or ``monte_carlo`` (independent random
        runs, which Wilks tolerance limits need).
    ``processes``
        Number of processes that run the model at the same time.  Default
        1.  With more, the model must be importable by name (defined at
        the top level of a module or script, and the script guarded by
        ``if __name__ == "__main__":``).
    ``store``
        A ``.npz`` file that keeps every run as it finishes.  A study that
        is interrupted resumes where it stopped, and runs with the same
        input values are reused by later studies.
    ``progress``
        Print a line per finished run.
    """
    dists = check_inputs(inputs)
    u = unit_design(int(samples), len(dists), method, seed)
    x = to_inputs(u, dists)
    outputs, failed = evaluate(model, list(dists), x, processes, store, progress)
    return Runs(dists, x, u, outputs, failed, method)
