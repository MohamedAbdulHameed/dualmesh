# SPDX-License-Identifier: LGPL-2.1-or-later
r"""Bayesian calibration (inverse uncertainty quantification) of model
parameters against measurements.

The measurements :math:`y_E` and the model :math:`y_M(\theta)` are related
by the model updating equation of Kennedy and O'Hagan (2001) and Wu et al. (2018)

.. math:: y_E = y_M(\theta) + \delta + \varepsilon,

with the measurement error :math:`\varepsilon \sim N(0, \Sigma_\text{noise})`
and the model discrepancy :math:`\delta`.  The posterior density of the
parameters is

.. math::

    p(\theta | y_E) \propto p(\theta)\,
    |\Sigma|^{-1/2} \exp\!\left[-\tfrac12 r^T \Sigma^{-1} r\right], \quad
    r = y_E - y_M(\theta) - \hat\delta, \quad
    \Sigma = \Sigma_\text{noise} + \Sigma_\text{surrogate}(\theta) +
    \Sigma_\delta.

All the outputs and all the measurements enter one likelihood, in the space
of the measurements.  With a Gaussian process surrogate,
:math:`\Sigma_\text{surrogate}` is its predictive covariance, of low rank in
the principal components plus the variance of the discarded components.  The
likelihood is then evaluated by the Woodbury identity at the cost of the
number of components.  With ``discrepancy="gaussian_process"``,
:math:`\delta` is a zero-mean Gaussian process in the location of the
measurements (time, position) with a Matern 5/2 correlation, integrated out of
the likelihood as Kennedy and O'Hagan (2001) do (Sect. 4.4): :math:`r = y_E -
y_M(\theta)` and :math:`\Sigma_\delta` is its prior covariance.  Its variance
and length scale are fixed at the joint posterior mode of the parameters and
the hyperparameters (their Sect. 4.5).  A discrepancy fitted to the
residuals at fixed parameter values and subtracted from the data would hold
the posterior at those values whatever the data (Wu et al. 2018, Sect. 5).

The posterior is sampled by independent ensembles of the affine-invariant
stretch move (Goodman and Weare 2010), or by adaptive Metropolis chains
(Andrieu and Thoms 2008, Algorithm 4).  The chains give the rank
normalized and folded split :math:`\hat R` of Vehtari et al. (2021), and the
integrated autocorrelation time (Goodman and Weare 2010, Eq. 17) the
effective sample size.
"""

from __future__ import annotations

import inspect
import math

import numpy as np
from scipy import linalg, optimize

from ._engine import (
    check_inputs,
    check_output_transform,
    evaluate,
    study_header,
    to_inputs,
    transformed_runs,
    unit_design,
    write_json,
)
from .gaussian_process import _kernel

__all__ = ["Posterior", "calibrate"]


# ---------------------------------------------------------------------------
# data layout
# ---------------------------------------------------------------------------
def _layout(observed, noise):
    if not isinstance(observed, dict) or not observed:
        raise ValueError("calibrate: observed must be a dict of output name -> measured values.")
    if not isinstance(noise, dict) or set(noise) != set(observed):
        raise ValueError("calibrate: noise must be a dict with the same names as observed.")
    names, shapes, ys, covs = [], [], [], []
    for name, value in observed.items():
        y = np.asarray(value, dtype=float)
        m = y.size
        e = np.asarray(noise[name], dtype=float)
        if e.ndim == 2 and e.shape == (m, m) and m > 1:
            cov = 0.5 * (e + e.T)
        else:
            sd = np.broadcast_to(e, y.shape).reshape(-1)
            if np.any(sd <= 0):
                raise ValueError(f"calibrate: the noise of '{name}' must be positive.")
            cov = np.diag(sd * sd)
        if not np.all(np.isfinite(y)):
            raise ValueError(f"calibrate: the observed '{name}' holds non-finite values.")
        names.append(name)
        shapes.append(y.shape)
        ys.append(y.reshape(-1))
        covs.append(cov)
    return names, shapes, np.concatenate(ys), linalg.block_diag(*covs)


def _flatten(outputs, names, shapes, rows):
    parts = []
    for name, shape in zip(names, shapes):
        if name not in outputs:
            raise KeyError(f"calibrate: the model does not return the observed output '{name}'.")
        v = np.asarray(outputs[name], dtype=float).reshape(rows, -1)
        if v.shape[1] != int(np.prod(shape)):
            raise ValueError(
                f"calibrate: the model output '{name}' has {v.shape[1]} values per run, the measurement {int(np.prod(shape))}."
            )
        parts.append(v)
    return np.concatenate(parts, axis=1)


# ---------------------------------------------------------------------------
# discrepancy
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# likelihood
# ---------------------------------------------------------------------------
class _Likelihood:
    """Gaussian log-likelihood with covariance D0 + A diag(v) A^T,
    evaluated for many parameter sets at once (Woodbury identity)."""

    def __init__(self, y, D0, A=None):
        self.y = y
        self.c = linalg.cho_factor(D0, lower=True)
        self.logdet0 = 2.0 * float(np.sum(np.log(np.diag(self.c[0]))))
        self.m = len(y)
        self.A = A
        if A is not None:
            self.DiA = linalg.cho_solve(self.c, A)
            self.B = A.T @ self.DiA

    def __call__(self, mean, v=None):
        r = self.y[None, :] - mean
        Dir = linalg.cho_solve(self.c, r.T).T
        q = np.sum(r * Dir, axis=1)
        logdet = np.full(len(r), self.logdet0)
        if self.A is not None and v is not None:
            s = np.sqrt(np.maximum(v, 0.0))
            w = (Dir @ self.A) * s
            nc = self.B.shape[0]
            M = np.eye(nc)[None] + s[:, :, None] * self.B[None] * s[:, None, :]
            L = np.linalg.cholesky(M)
            z = np.linalg.solve(L, w[..., None])[..., 0]
            q -= np.sum(z * z, axis=1)
            logdet += 2.0 * np.sum(np.log(np.diagonal(L, axis1=1, axis2=2)), axis=1)
        return -0.5 * (q + logdet + self.m * math.log(2.0 * math.pi))


# ---------------------------------------------------------------------------
# samplers
# ---------------------------------------------------------------------------
def _ensemble(logpost, x0, steps, rng, a=2.0):
    """Stretch-move ensembles (Goodman and Weare 2010, Eqs. (7) and (9), with
    the acceptance probability min(1, Z^(k-1) p(Y)/p(X))).  x0 is (C, W, k).
    The two halves of each ensemble move in turn, each walker with a partner
    drawn from the other half, which leaves the target invariant because
    each half is updated given the other.  Returns the chain (steps, C, W,
    k), its log posterior and the acceptance fraction."""
    C, W, k = x0.shape
    x = x0.copy()
    lp = logpost(x.reshape(-1, k)).reshape(C, W)
    chain = np.empty((steps, C, W, k))
    lps = np.empty((steps, C, W))
    accepted = 0
    half = W // 2
    groups = (np.arange(half), np.arange(half, W))
    for t in range(steps):
        for g in range(2):
            act, other = groups[g], groups[1 - g]
            n = len(act)
            z = ((a - 1.0) * rng.random((C, n)) + 1.0) ** 2 / a
            partner = x[:, other][np.arange(C)[:, None], rng.integers(0, len(other), (C, n))]
            prop = partner + z[..., None] * (x[:, act] - partner)
            lpp = logpost(prop.reshape(-1, k)).reshape(C, n)
            logr = (k - 1) * np.log(z) + lpp - lp[:, act]
            acc = np.log(rng.random((C, n))) < logr
            xa, la = x[:, act], lp[:, act]
            xa[acc], la[acc] = prop[acc], lpp[acc]
            x[:, act], lp[:, act] = xa, la
            accepted += int(acc.sum())
        chain[t], lps[t] = x, lp
    return chain, lps, accepted / (steps * C * W)


def _metropolis(logpost, x0, cov0, steps, rng, target=0.234):
    """Adaptive Metropolis with global adaptive scaling (Andrieu and Thoms
    2008, Algorithm 4), one chain per row of x0 (C, k).  Returns the chain
    (steps, C, 1, k), its log posterior and the acceptance fraction."""
    C, k = x0.shape
    x = x0.copy()
    lp = logpost(x)
    mu = x.copy()
    sig = np.repeat(cov0[None], C, axis=0)
    loglam = np.full(C, math.log(2.38**2 / k))
    chain = np.empty((steps, C, 1, k))
    lps = np.empty((steps, C, 1))
    accepted = 0
    for t in range(steps):
        g = 1.0 / (t + 2.0) ** 0.6
        prop = np.empty_like(x)
        for c in range(C):
            L = np.linalg.cholesky(np.exp(loglam[c]) * sig[c] + 1e-12 * np.eye(k))
            prop[c] = x[c] + L @ rng.standard_normal(k)
        lpp = logpost(prop)
        alpha = np.exp(np.minimum(0.0, lpp - lp))
        acc = rng.random(C) < alpha
        x[acc], lp[acc] = prop[acc], lpp[acc]
        accepted += int(acc.sum())
        d = x - mu
        mu += g * d
        sig += g * (np.einsum("ci,cj->cij", d, d) - sig)
        loglam += g * (alpha - target)
        chain[t, :, 0], lps[t, :, 0] = x, lp
    return chain, lps, accepted / (steps * C)


# ---------------------------------------------------------------------------
# diagnostics
# ---------------------------------------------------------------------------
def _classic_split_rhat(s):
    """Split R-hat of Vehtari et al. (2021), Eqs. (1) to (4), of the columns
    of ``s`` (one split chain per column)."""
    N = s.shape[0]
    B = N * s.mean(axis=0).var(ddof=1)
    Wv = s.var(axis=0, ddof=1).mean()
    if Wv <= 0:
        return math.nan
    return math.sqrt(((N - 1) / N * Wv + B / N) / Wv)


def _normal_scores(s):
    """Rank normalization of Vehtari et al. (2021), Eq. (14): the ranks r
    of the pooled draws (average ranks for ties) mapped to
    Phi^-1((r - 3/8)/(S + 1/4))."""
    from scipy import special, stats

    r = stats.rankdata(s.reshape(-1)).reshape(s.shape)
    return special.ndtri((r - 0.375) / (s.size + 0.25))


def _split_rhat(chain):
    """R-hat of one parameter as Vehtari et al. (Bayesian Analysis 2021)
    recommend (Sect. 4.2): the larger of the split R-hat of the rank
    normalized draws and that of the rank normalized folded draws
    |theta - median(theta)|.  ``chain`` is (steps, C, W): C independent
    ensembles (or chains, W = 1) of W walkers.  Each ensemble is one Markov
    chain whose samples are the positions of all its walkers.  The first and
    second halves of each are compared, 2C chains in all."""
    steps, C, W = chain.shape
    n = steps // 2
    if n < 2:
        return math.nan
    halves = [chain[:n, c].reshape(-1) for c in range(C)]
    halves += [chain[n : 2 * n, c].reshape(-1) for c in range(C)]
    s = np.column_stack(halves)
    folded = np.abs(s - np.median(s))
    return max(_classic_split_rhat(_normal_scores(s)), _classic_split_rhat(_normal_scores(folded)))


def _autocorr_time(seq, c=5.0):
    r"""Integrated autocorrelation time of sequences (n, s),
    :math:`\tau = \sum_t C(t)/C(0)` (Goodman and Weare 2010, Eq. 17), with the
    autocorrelation averaged over the sequences and the sum truncated at the
    first lag m with m >= c tau(m), the self-consistent window that Goodman
    and Weare use (Sect. 4)."""
    n = seq.shape[0]
    x = seq - seq.mean(axis=0)
    f = np.fft.rfft(x, n=2 * n, axis=0)
    acf = np.fft.irfft(f * np.conj(f), axis=0)[:n]
    var = acf[0]
    ok = var > 0
    if not np.any(ok):
        return math.nan
    rho = (acf[:, ok] / var[ok]).mean(axis=1)
    taus = 2.0 * np.cumsum(rho) - 1.0
    m = np.arange(n)
    win = np.nonzero(m >= c * taus)[0]
    w = win[0] if len(win) else n - 1
    return float(max(taus[w], 1.0))


def _hpd(x, level):
    x = np.sort(x)
    n = len(x)
    m = max(1, int(math.floor(level * n)))
    widths = x[m:] - x[: n - m]
    i = int(np.argmin(widths))
    return float(x[i]), float(x[i + m])


# ---------------------------------------------------------------------------
# result
# ---------------------------------------------------------------------------
class Posterior:
    """The result of :func:`calibrate`.

    Attributes
    ----------
    samples : dict
        Parameter name -> posterior samples.
    contraction : dict
        :math:`1 - \\sigma^2_\\text{post}/\\sigma^2_\\text{prior}`: near 1,
        the data determine the parameter.  Near 0, the data carry no
        information on it (not identified).
    at_bound : dict
        ``"lower"`` or ``"upper"`` when more than a quarter of the posterior
        lies in the outer 2.5 % of the prior on that side (the parameter is
        pushed against its prior, often a sign of model discrepancy), else
        None.
    r_hat, effective_sample_size : dict
        Convergence diagnostics: split :math:`\\hat R` (below 1.01 is
        converged) and the number of independent samples the chains are
        worth.
    acceptance : float
        Fraction of accepted proposals.
    surrogate : GaussianProcess or None
        The surrogate used, with its leave-one-out ``q2``.
    discrepancy : dict or None
        Output name -> ``variance`` and ``length_scale`` (relative to the
        span of the locations) of the discrepancy process.
    mode : dict or None
        With a discrepancy, the parameters at the joint posterior mode at
        which the hyperparameters of the discrepancy were estimated.
    """

    def __init__(self, dists, x, lp, chain, acceptance, level, context):
        self.distributions = dists
        self.names = list(dists)
        self.x = x
        self.log_posterior = lp
        self.samples = {n: x[:, j] for j, n in enumerate(self.names)}
        self.acceptance = acceptance
        self.level = level
        self._ctx = context
        if "experiments" in context:
            experiments = context["experiments"]
            self.surrogate = {name: ctx.get("gp") for name, ctx in experiments.items()}
            self.discrepancy = (
                {name: ctx.get("discrepancy") for name, ctx in experiments.items()}
                if any(ctx.get("discrepancy") for ctx in experiments.values())
                else None
            )
            self.mode = (
                {name: ctx.get("mode") for name, ctx in experiments.items()}
                if any(ctx.get("mode") for ctx in experiments.values())
                else None
            )
        else:
            self.surrogate = context.get("gp")
            self.discrepancy = context.get("discrepancy")
            self.mode = context.get("mode")
        steps, C, W, k = chain.shape
        seqs = chain.reshape(steps, C * W, k)
        self.r_hat = {n: _split_rhat(chain[..., j]) for j, n in enumerate(self.names)}
        self.effective_sample_size = {}
        for j, n in enumerate(self.names):
            tau = _autocorr_time(seqs[:, :, j])
            self.effective_sample_size[n] = (steps * C * W / tau) if tau == tau else math.nan
        self.contraction, self.at_bound = {}, {}
        for j, n in enumerate(self.names):
            d = dists[n]
            v0 = d.standard_deviation**2
            self.contraction[n] = 1.0 - float(np.var(x[:, j])) / v0 if v0 > 0 else math.nan
            lo, hi = d.ppf(0.025), d.ppf(0.975)
            plo, phi = float(np.mean(x[:, j] < lo)), float(np.mean(x[:, j] > hi))
            self.at_bound[n] = "lower" if plo > 0.25 else ("upper" if phi > 0.25 else None)

    def mean(self, name):
        return float(np.mean(self.samples[name]))

    def standard_deviation(self, name):
        return float(np.std(self.samples[name], ddof=1))

    def interval(self, name, level=None):
        """Highest posterior density interval (the shortest interval holding
        the fraction ``level`` of the samples)."""
        return _hpd(self.samples[name], level or self.level)

    @property
    def map(self) -> dict:
        """The sample of highest posterior density."""
        i = int(np.argmax(self.log_posterior))
        return {n: float(self.x[i, j]) for j, n in enumerate(self.names)}

    def correlation(self):
        """Posterior correlation matrix of the parameters."""
        return np.corrcoef(self.x.T)

    def summary(self) -> str:
        """Prior against posterior for every parameter, with the
        diagnostics."""
        lv = f"{self.level:.0%}"
        head = f"{'parameter':<28}{'prior mean':>11}{'prior std':>10}{'post. mean':>11}{'post. std':>10}{'MAP':>10}  {lv + ' HPD':<22}{'contr.':>7}{'R-hat':>7}{'ESS':>7}  note"
        lines = [head, "-" * len(head)]
        mp = self.map
        for n in self.names:
            d = self.distributions[n]
            lo, hi = self.interval(n)
            note = []
            if self.contraction[n] < 0.1:
                note.append("not identified by the data")
            if self.at_bound[n]:
                note.append(f"at the {self.at_bound[n]} end of the prior")
            if self.r_hat[n] > 1.01:
                note.append("not converged")
            lines.append(
                f"{n:<28}{d.mean:11.4g}{d.standard_deviation:10.3g}{self.mean(n):11.4g}{self.standard_deviation(n):10.3g}{mp[n]:10.4g}  [{lo:9.4g}, {hi:9.4g}] {self.contraction[n]:7.2f}{self.r_hat[n]:7.3f}{self.effective_sample_size[n]:7.0f}  {'; '.join(note)}"
            )
        lines.append("")
        lines.append(f"{len(self.x)} samples, acceptance {self.acceptance:.2f}")
        surrogates = self.surrogate if isinstance(self.surrogate, dict) else {None: self.surrogate}
        for experiment, gp in surrogates.items():
            if gp is None:
                continue
            q2 = np.atleast_1d(gp.q2)
            label = "surrogate" if experiment is None else f"surrogate of {experiment}"
            lines.append(
                f"{label}: {len(gp._u_train)} runs, {gp.components} component(s), leave-one-out Q2 {np.nanmin(q2):.3f} (lowest) to {np.nanmax(q2):.3f}"
            )
        return "\n".join(lines)

    def __repr__(self):
        return f"Posterior({len(self.x)} samples of {', '.join(self.names)})"

    def predict(self, samples=200, level=0.95, seed=0):
        """Posterior predictive distribution of the observed outputs.

        Returns a dict output name -> dict with ``mean``, ``low``, ``high``
        (the central interval of the predictive distribution, which includes
        the measurement noise), ``model_low``, ``model_high`` (the interval
        of the model alone) and ``coverage`` (the fraction of the
        measurements inside ``[low, high]``).  With several experiments,
        it returns a dict experiment name -> that dict."""
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(self.x), size=min(samples, len(self.x)), replace=False)
        theta = self.x[idx]
        if "experiments" in self._ctx:
            return {
                name: _predict(ctx, theta[:, ctx["input_index"]], rng, level)
                for name, ctx in self._ctx["experiments"].items()
            }
        return _predict(self._ctx, theta, rng, level)

    def to_dict(self) -> dict:
        """The posterior mean, standard deviation, highest density interval
        and maximum of every parameter, with its contraction, its position
        against the prior bounds and the convergence diagnostics."""
        numbers = {
            name: dict(
                mean=self.mean(name),
                standard_deviation=self.standard_deviation(name),
                interval=self.interval(name),
                map=self.map[name],
                contraction=self.contraction[name],
                at_bound=self.at_bound[name],
                r_hat=self.r_hat[name],
                effective_sample_size=self.effective_sample_size[name],
            )
            for name in self.names
        }
        return {
            "study": "calibrate",
            "level": self.level,
            "acceptance": self.acceptance,
            "parameters": numbers,
        }

    def write_json(self, path):
        """Write :meth:`to_dict` to a JSON file (a non-finite value is written
        as null)."""
        write_json(self.to_dict(), path)

    @property
    def tables(self) -> dict:
        """``parameters``: one row per parameter, with the prior and
        posterior statistics, the highest density interval and the
        diagnostics."""
        from ..tables import Table

        rows = []
        for n in self.names:
            d = self.distributions[n]
            lo, hi = self.interval(n)
            rows.append(
                [
                    n,
                    float(d.mean),
                    float(d.standard_deviation),
                    float(self.mean(n)),
                    float(self.standard_deviation(n)),
                    float(self.map[n]),
                    float(lo),
                    float(hi),
                    float(self.contraction[n]),
                    float(self.r_hat[n]),
                    float(self.effective_sample_size[n]),
                ]
            )
        return {
            "parameters": Table(
                [
                    "parameter",
                    "prior_mean",
                    "prior_standard_deviation",
                    "posterior_mean",
                    "posterior_standard_deviation",
                    "map",
                    "interval_low",
                    "interval_high",
                    "contraction",
                    "r_hat",
                    "effective_sample_size",
                ],
                None,
                rows,
                title="Posterior of the parameters",
            )
        }

    def write_csv(self, path, table: str | None = None):
        """Write the table ``parameters`` to a CSV file."""
        from ..tables import ResultTables

        ResultTables.write_csv(self, path, table)

    def save(self, path):
        """Write the samples and the diagnostics to a ``.npz`` file."""
        np.savez(
            path,
            names=np.array(self.names),
            samples=self.x,
            log_posterior=self.log_posterior,
            r_hat=np.array([self.r_hat[n] for n in self.names]),
            effective_sample_size=np.array([self.effective_sample_size[n] for n in self.names]),
            contraction=np.array([self.contraction[n] for n in self.names]),
        )


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def _predict(ctx, theta, rng, level):
    """The posterior predictive of one experiment at the parameters ``theta``."""
    mean, v = ctx["predict"](theta)
    if ctx.get("Sdelta") is not None:
        mean = mean + (ctx["Sdelta"] @ np.linalg.solve(ctx["D0"], (ctx["y"][None, :] - mean).T)).T
    else:
        mean = mean + ctx["delta"]
    D0 = ctx["D0"]
    L0 = np.linalg.cholesky(D0)
    draws = mean + rng.standard_normal(mean.shape) @ L0.T
    A = ctx.get("A")
    if A is not None and v is not None:
        draws += (rng.standard_normal(v.shape) * np.sqrt(np.maximum(v, 0))) @ A.T
    a = 50.0 * (1.0 - level)
    out, j = {}, 0
    y = ctx["y"]
    log_rows = ctx.get("log_rows")
    if log_rows is not None and np.any(log_rows):
        # Quantiles keep their order under exp, so the intervals and the
        # coverage are those of the measurements.
        draws = np.where(log_rows, np.exp(draws), draws)
        mean = np.where(log_rows, np.exp(mean), mean)
        y = ctx["measured"]
    for name, shape in zip(ctx["names"], ctx["shapes"]):
        m = int(np.prod(shape)) if shape else 1
        sl = slice(j, j + m)
        lo = np.percentile(draws[:, sl], a, axis=0)
        hi = np.percentile(draws[:, sl], 100 - a, axis=0)
        inside = (y[sl] >= lo) & (y[sl] <= hi)
        out[name] = dict(
            mean=mean[:, sl].mean(axis=0).reshape(shape),
            low=lo.reshape(shape),
            high=hi.reshape(shape),
            model_low=np.percentile(mean[:, sl], a, axis=0).reshape(shape),
            model_high=np.percentile(mean[:, sl], 100 - a, axis=0).reshape(shape),
            coverage=float(np.mean(inside)),
        )
        j += m
    return out


def _experiment(
    model,
    dists,
    observed,
    noise,
    surrogate,
    training_samples,
    training,
    discrepancy,
    locations,
    output_transform,
    seed,
    processes,
    store,
    progress,
) -> dict:
    """The likelihood of one experiment: its data, its surrogate (or its
    model) and its discrepancy, as a context with ``predict`` and ``like``."""
    names = list(dists)
    k = len(names)
    onames, shapes, y, Sn = _layout(observed, noise)
    transform = dict(output_transform or {})
    log_rows = np.zeros(len(y), dtype=bool)
    j = 0
    for name, shape in zip(onames, shapes):
        m = int(np.prod(shape)) if shape else 1
        if name in transform:
            if transform[name] != "log":
                raise ValueError(
                    f"calibrate: unknown output transform '{transform[name]}' of '{name}'. The transforms are: log."
                )
            if np.any(y[j : j + m] <= 0.0):
                raise ValueError(
                    f"calibrate: the log transform needs positive measurements, and '{name}' has the value {float(np.min(y[j : j + m])):.6g}."
                )
            log_rows[j : j + m] = True
        j += m
    unknown = [name for name in transform if name not in onames]
    if unknown:
        raise ValueError(
            f"calibrate: output_transform names the outputs {unknown}, which are not observed. Observed: {', '.join(onames)}."
        )
    measured = y.copy()
    y = np.where(log_rows, np.log(np.where(log_rows, y, 1.0)), y)
    context = dict(names=onames, shapes=shapes, y=y, measured=measured, log_rows=log_rows)

    gp = None
    if surrogate == "gaussian_process":
        from .gaussian_process import GaussianProcess
        from .propagate import _propagate

        if training is not None and training_samples is not None:
            raise ValueError("calibrate: give training_samples or training, not both.")
        if training is None:
            m = int(training_samples) if training_samples else max(10 * k, 30)
            training = _propagate(
                model, dists, m, "latin_hypercube", seed, processes, store, progress
            )
        if list(training.distributions) != names:
            raise ValueError("calibrate: the training runs must have the same inputs, in order.")
        missing = [n for n in onames if n not in training.outputs]
        if missing:
            raise KeyError(f"calibrate: the training runs lack the outputs {missing}.")
        # Check the shapes of the training outputs against the measurements.
        _flatten({n: training.outputs[n][:1] for n in onames}, onames, shapes, 1)
        check_output_transform(transform, training, "calibrate")
        gp = GaussianProcess(seed=seed).fit(transformed_runs(training, transform), output=onames)
        A = gp._V * gp._sd[:, None]
        Sd = np.diag(gp._discarded * gp._sd**2)

        def predict(theta):
            u = gp._to_unit(theta)
            mean, pv = gp._predict_flat(u)
            return mean, pv

        context.update(gp=gp, A=A)
    elif surrogate is None:
        A, Sd = None, np.zeros_like(Sn)

        def predict(theta):
            outs, _ = evaluate(model, names, theta, processes, store, False)
            mean = _flatten(outs, onames, shapes, len(theta))
            if np.any(log_rows):
                with np.errstate(invalid="ignore", divide="ignore"):
                    mean = np.where(log_rows, np.log(np.where(mean > 0.0, mean, np.nan)), mean)
            return mean, None
    else:
        raise ValueError("calibrate: surrogate must be 'gaussian_process' or None.")
    context["predict"] = predict

    # Discrepancy: a zero-mean Gaussian process in the location of the
    # measurements, integrated out of the likelihood (Kennedy and O'Hagan
    # 2001, Sect. 4.4), with its hyperparameters fixed at the joint posterior
    # mode of the parameters and the hyperparameters (their Sect. 4.5).
    delta = np.zeros_like(y)
    Sdelta = np.zeros_like(Sn)
    if discrepancy == "gaussian_process":
        blocks = []
        j = 0
        for name, shape in zip(onames, shapes):
            m = int(np.prod(shape)) if shape else 1
            loc = (
                np.asarray(locations[name], dtype=float).reshape(m, -1)
                if locations and name in locations
                else np.arange(m, dtype=float).reshape(m, 1)
            )
            span = np.ptp(loc, axis=0)
            x = (loc - loc.min(axis=0)) / np.where(span > 0, span, 1.0)
            blocks.append(
                (
                    name,
                    shape,
                    slice(j, j + m),
                    ((x[:, None, :] - x[None, :, :]) ** 2).sum(axis=-1)[..., None],
                )
            )
            j += m
        theta0 = np.array([dists[n].mean for n in names])
        scale = np.array([dists[n].standard_deviation for n in names])
        r0 = y - predict(theta0[None])[0][0]
        v0 = [
            max(float(np.var(r0[b[2]])), float(np.mean(np.diag(Sn)[b[2]])), 1e-30) for b in blocks
        ]

        def covariance(hyper):
            P = np.zeros_like(Sn)
            for b, (ls2, lell) in zip(blocks, hyper.reshape(-1, 2)):
                P[b[2], b[2]] = math.exp(ls2) * _kernel("matern52", b[3] / math.exp(2 * lell))[0]
            return P

        def negative_log_posterior(z):
            theta = theta0 + scale * z[:k]
            prior = sum(dists[n].logpdf(np.array([theta[i]]))[0] for i, n in enumerate(names))
            if not np.isfinite(prior):
                return 1e25
            mean, v = predict(theta[None])
            C = Sn + Sd + covariance(z[k:])
            if A is not None and v is not None:
                C = C + (A * v[0]) @ A.T
            try:
                c = linalg.cho_factor(C, lower=True)
            except linalg.LinAlgError:
                return 1e25
            r = y - mean[0]
            return (
                0.5 * float(r @ linalg.cho_solve(c, r))
                + float(np.sum(np.log(np.diag(c[0]))))
                - prior
            )

        best = None
        for le in (math.log(0.2), math.log(1.0)):
            z0 = np.concatenate([np.zeros(k)] + [[math.log(v), le] for v in v0])
            bounds = [(-6.0, 6.0)] * k + [
                b
                for v in v0
                for b in ((math.log(v) - 20, math.log(v) + 8), (math.log(1e-2), math.log(10.0)))
            ]
            res = optimize.minimize(negative_log_posterior, z0, method="L-BFGS-B", bounds=bounds)
            if best is None or res.fun < best.fun:
                best = res
        Sdelta = covariance(best.x[k:])
        theta_mode = theta0 + scale * best.x[:k]
        info = {}
        for (name, *_), (ls2, lell) in zip(blocks, best.x[k:].reshape(-1, 2)):
            info[name] = dict(variance=math.exp(ls2), length_scale=math.exp(lell))
        context["discrepancy"] = info
        context["mode"] = dict(zip(names, theta_mode))
    elif discrepancy is not None:
        raise ValueError("calibrate: discrepancy must be None or 'gaussian_process'.")
    context["delta"] = delta

    D0 = Sn + Sd + Sdelta
    D0 = 0.5 * (D0 + D0.T) + 1e-12 * np.mean(np.diag(D0)) * np.eye(len(y))
    context["D0"] = D0
    context["Sdelta"] = Sdelta if discrepancy == "gaussian_process" else None
    like = _Likelihood(y - delta, D0, A)
    context["like"] = like
    return context


def _experiments(
    model,
    dists,
    observed,
    noise,
    surrogate,
    training_samples,
    training,
    discrepancy,
    locations,
    output_transform,
    seed,
    processes,
    store,
    progress,
) -> dict:
    """The contexts of the experiments: one for a single model, or one per
    experiment for a dict of models, each with the indices of its inputs."""
    names = list(dists)
    if not isinstance(model, dict):
        context = _experiment(
            model,
            dists,
            observed,
            noise,
            surrogate,
            training_samples,
            training,
            discrepancy,
            locations,
            output_transform,
            seed,
            processes,
            store,
            progress,
        )
        context["input_index"] = list(range(len(names)))
        return {None: context}

    def per(value, label, name):
        if value is None:
            return None
        if not isinstance(value, dict) or name not in value:
            raise ValueError(
                f"calibrate: with several experiments, {label} is a dict of experiment name -> the {label} of that experiment, and it has no entry '{name}'."
            )
        return value[name]

    contexts, used = {}, set()
    for name, function in model.items():
        runs = per(training, "training", name)
        if runs is not None:
            parameters = list(runs.distributions)
        else:
            signature = inspect.signature(function).parameters.values()
            if any(p.kind in (p.VAR_KEYWORD, p.VAR_POSITIONAL) for p in signature):
                raise ValueError(
                    f"calibrate: the model of experiment '{name}' takes **keywords, so its inputs are not known. Give its training runs, or name its inputs as parameters."
                )
            parameters = [p.name for p in signature]
        missing = [n for n in parameters if n not in dists]
        if missing:
            raise ValueError(
                f"calibrate: the experiment '{name}' has the inputs {missing}, which have no prior. Inputs: {', '.join(names)}."
            )
        context = _experiment(
            function,
            {n: dists[n] for n in parameters},
            per(observed, "observed", name),
            per(noise, "noise", name),
            surrogate,
            training_samples,
            runs,
            discrepancy,
            per(locations, "locations", name) if locations else None,
            (output_transform or {}).get(name),
            seed,
            processes,
            store,
            progress,
        )
        context["input_index"] = [names.index(n) for n in parameters]
        contexts[name] = context
        used.update(parameters)
    unused = [n for n in names if n not in used]
    if unused:
        raise ValueError(
            f"calibrate: the inputs {unused} enter no experiment. Remove them, or give them to a model."
        )
    return contexts


def calibrate(
    model,
    inputs,
    observed,
    noise,
    surrogate="gaussian_process",
    training_samples=None,
    training=None,
    discrepancy=None,
    locations=None,
    sampler="ensemble",
    samples=20000,
    walkers=None,
    chains=4,
    level=0.95,
    seed=0,
    processes=1,
    store=None,
    output_transform=None,
    report="full",
) -> Posterior:
    r"""Sample the posterior distribution of the ``inputs`` of ``model``
    given measurements.  See the module documentation.

    ``model``
        A function of the inputs, or a dict of experiment name -> function
        for several experiments (for example two test specimens).  The
        experiments share the inputs that their functions have in common,
        and each takes its other inputs alone.  The inputs of an experiment
        are those of its ``training`` runs, or else the parameter names of
        its function.  With several experiments, ``observed``, ``noise``,
        ``training``, ``locations`` and ``output_transform`` are dicts of
        experiment name -> the value for that experiment, and the
        log-likelihoods of the experiments add up.
    ``inputs``
        Parameter name -> prior distribution.
    ``observed``
        Output name -> measured values (a number or an array), for outputs
        that ``model`` returns with the same shapes.  Several outputs (for
        example the release and the temperature of two rods) are
        calibrated jointly.
    ``noise``
        Output name -> standard deviation of the measurement error (a number
        or an array of the shape of the measurement) or its covariance
        matrix.
    ``surrogate``
        ``"gaussian_process"`` (default): fit a :class:`GaussianProcess` to
        ``training_samples`` Latin hypercube runs over the priors (default
        :math:`10k`, at least 30), or the given ``training`` runs of
        :func:`propagate`, and sample on it.  None: run the model at every
        proposal (only for fast models).
    ``discrepancy``
        None or ``"gaussian_process"`` (a discrepancy function of
        ``locations``: output name -> coordinates of the measurements,
        default their index).
    ``sampler``
        ``"ensemble"`` (the default, which needs no tuning) or
        ``"metropolis"``.
    ``samples``, ``walkers``, ``chains``
        Samples kept after the burn-in (an equal number of steps is
        discarded, and on a surrogate the ensembles are extended until the
        split :math:`\hat R` of every parameter is below 1.01), walkers per
        ensemble (default :math:`\max(2k+2, 16)`)
        and independent ensembles or chains (default 4).
    ``output_transform``
        Output name -> ``"log"``.  The calibration of that output is on the
        logarithmic scale: the surrogate models the logarithm of the output,
        and the likelihood compares the logarithms of the model and of the
        measurements.  Its ``noise`` is then the standard deviation of the
        logarithm of the measurement, which is about its relative error (0.1
        for 10%).  This suits a positive output that changes by factors (a
        creep strain).  The predictions of :meth:`Posterior.predict`
        are on the scale of the measurements.  Default None: every output is
        calibrated as it is.
    ``processes``, ``store``, ``report``
        As for :func:`propagate`.
    """
    from ..console import report_level

    report = report_level(report, "uq.calibrate")
    progress = report == "full"
    dists = check_inputs(inputs)
    study_header(report, "Bayesian calibration", dists)
    names = list(dists)
    k = len(names)
    contexts = _experiments(
        model,
        dists,
        observed,
        noise,
        surrogate,
        training_samples,
        training,
        discrepancy,
        locations,
        output_transform,
        seed,
        processes,
        store,
        progress,
    )
    rng = np.random.default_rng(seed)

    def logpost(theta):
        lp = np.zeros(len(theta))
        for j, n in enumerate(names):
            lp += dists[n].logpdf(theta[:, j])
        ok = np.isfinite(lp)
        if np.any(ok):
            for ctx in contexts.values():
                mean, v = ctx["predict"](theta[ok][:, ctx["input_index"]])
                lp[ok] += ctx["like"](mean, v)
        lp[~np.isfinite(lp)] = -np.inf
        return lp

    C = max(1, int(chains))
    if sampler == "ensemble":
        W = int(walkers) if walkers else max(2 * k + 2, 16)
        W += W % 2
        keep = max(int(math.ceil(samples / (C * W))), 50)
        pool = 10 * W if surrogate is not None else W
        x0 = np.empty((C, W, k))
        for c in range(C):
            cand = to_inputs(unit_design(pool, k, "monte_carlo", rng), dists)
            lpc = logpost(cand)
            x0[c] = cand[np.argsort(-lpc)[:W]]
        chain, lps, acc = _ensemble(logpost, x0, 2 * keep, rng)
        # On a surrogate, extend the ensembles until the split R-hat of every
        # parameter is below 1.01 (at most 16 times the first length).
        while surrogate is not None and len(chain) < 32 * keep:
            half = chain[len(chain) // 2 :]
            if max(_split_rhat(half[..., j]) for j in range(k)) <= 1.01:
                break
            more, lmore, acc2 = _ensemble(logpost, chain[-1], len(chain), rng)
            acc = 0.5 * (acc + acc2)
            chain, lps = np.concatenate([chain, more]), np.concatenate([lps, lmore])
        keep = len(chain) // 2
    elif sampler == "metropolis":
        keep = max(int(math.ceil(samples / C)), 100)
        pool = 50 if surrogate is not None else 1
        x0 = np.empty((C, k))
        for c in range(C):
            cand = to_inputs(unit_design(pool, k, "monte_carlo", rng), dists)
            x0[c] = cand[int(np.argmax(logpost(cand)))]
        cov0 = np.diag([dists[n].standard_deviation ** 2 for n in names])
        chain, lps, acc = _metropolis(logpost, x0, cov0, 2 * keep, rng)
    else:
        raise ValueError("calibrate: sampler must be ensemble or metropolis.")
    chain, lps = chain[keep:], lps[keep:]
    flat = chain.reshape(-1, k)
    context = contexts.get(None, {"experiments": contexts})
    posterior = Posterior(dists, flat, lps.reshape(-1), chain, acc, level, context)
    if report != "none":
        print(posterior.summary())
    return posterior
