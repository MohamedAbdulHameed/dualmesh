# SPDX-License-Identifier: LGPL-2.1-or-later
r"""Bayesian calibration (inverse uncertainty quantification) of model
parameters against measurements.

The measurements :math:`y_E` and the model :math:`y_M(\theta)` are related
by the model updating equation (Kennedy and O'Hagan 2001; Wu et al. 2018)

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
the principal components plus the variance of the discarded components; the
likelihood is then evaluated by the Woodbury identity at the cost of the
number of components.  With ``discrepancy="gaussian_process"``,
:math:`\delta` is a Gaussian process in the location of the measurements
(time, burnup), fitted to the residuals at the prior mean of the parameters
and then held fixed: its posterior mean is :math:`\hat\delta` and its
posterior covariance :math:`\Sigma_\delta` (the improved modular Bayesian
approach of Wu, Kozlowski, Meidani and Shirvan 2018).

The posterior is sampled by independent ensembles of the affine-invariant
stretch move (Goodman and Weare 2010), or by adaptive Metropolis chains
(Andrieu and Thoms 2008, Algorithm 4).  The chains give the split
:math:`\hat R` of Gelman et al. (2013), and the autocorrelation time the
effective sample size.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import linalg, optimize

from ._engine import check_inputs, evaluate, to_inputs, unit_design
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
                f"calibrate: the model output '{name}' has {v.shape[1]} values per run, the "
                f"measurement {int(np.prod(shape))}."
            )
        parts.append(v)
    return np.concatenate(parts, axis=1)


# ---------------------------------------------------------------------------
# discrepancy
# ---------------------------------------------------------------------------
def _fit_discrepancy(d, loc, noise_cov):
    """Zero-mean Matern 5/2 process in the location, fitted by maximum
    likelihood to the residuals ``d`` with the known noise covariance.
    Returns the posterior mean and covariance of the discrepancy at the
    measurement locations."""
    loc = np.asarray(loc, dtype=float).reshape(len(d), -1)
    span = np.ptp(loc, axis=0)
    span = np.where(span > 0, span, 1.0)
    x = (loc - loc.min(axis=0)) / span
    d2 = ((x[:, None, :] - x[None, :, :]) ** 2).sum(axis=-1)[..., None]

    def corr(log_ell):
        return _kernel("matern52", d2 / math.exp(2 * log_ell))[0]

    def nll(theta):
        s2, K = math.exp(theta[0]), corr(theta[1])
        C = s2 * K + noise_cov
        try:
            c = linalg.cho_factor(C, lower=True)
        except linalg.LinAlgError:
            return 1e25
        a = linalg.cho_solve(c, d)
        return 0.5 * float(d @ a) + float(np.sum(np.log(np.diag(c[0]))))

    v0 = max(float(np.var(d)), 1e-30)
    best = None
    for le in (math.log(0.05), math.log(0.2), math.log(1.0)):
        res = optimize.minimize(
            nll,
            [math.log(v0), le],
            method="L-BFGS-B",
            bounds=[(math.log(v0) - 20, math.log(v0) + 5), (math.log(1e-3), math.log(10.0))],
        )
        if best is None or res.fun < best.fun:
            best = res
    s2, K = math.exp(best.x[0]), corr(best.x[1])
    P = s2 * K
    c = linalg.cho_factor(P + noise_cov, lower=True)
    mean = P @ linalg.cho_solve(c, d)
    cov = P - P @ linalg.cho_solve(c, P)
    return mean, 0.5 * (cov + cov.T), dict(variance=s2, length_scale=math.exp(best.x[1]))


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
    """Stretch-move ensembles; x0 (C, W, k).  Returns the chain (steps, C,
    W, k), its log posterior and the acceptance fraction."""
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
def _split_rhat(chain):
    """Split R-hat (Gelman et al. 2013, Sect. 11.4) of one parameter.
    ``chain`` is (steps, C, W): C independent ensembles (or chains, W = 1)
    of W walkers.  Each ensemble is one Markov chain whose samples are the
    positions of all its walkers; the first and second halves of each are
    compared, 2C chains in all."""
    steps, C, W = chain.shape
    n = steps // 2
    if n < 2:
        return math.nan
    halves = [chain[:n, c].reshape(-1) for c in range(C)]
    halves += [chain[n : 2 * n, c].reshape(-1) for c in range(C)]
    s = np.column_stack(halves)
    N = s.shape[0]
    B = N * s.mean(axis=0).var(ddof=1)
    Wv = s.var(axis=0, ddof=1).mean()
    if Wv <= 0:
        return math.nan
    # With W walkers the within-chain samples are not independent in N;
    # the classical formula is kept, with N the number of samples per chain.
    return math.sqrt(((N - 1) / N * Wv + B / N) / Wv)


def _autocorr_time(seq, c=5.0):
    """Integrated autocorrelation time of sequences (n, s), the
    autocorrelation averaged over the sequences, with the automatic window
    of Sokal (1997)."""
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
        the data determine the parameter; near 0, the data carry no
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
        Output name -> (mean, covariance, hyperparameters) of the fitted
        discrepancy.
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
        self.surrogate = context.get("gp")
        self.discrepancy = context.get("discrepancy")
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
            v0 = d.std**2
            self.contraction[n] = 1.0 - float(np.var(x[:, j])) / v0 if v0 > 0 else math.nan
            lo, hi = d.ppf(0.025), d.ppf(0.975)
            plo, phi = float(np.mean(x[:, j] < lo)), float(np.mean(x[:, j] > hi))
            self.at_bound[n] = "lower" if plo > 0.25 else ("upper" if phi > 0.25 else None)

    def mean(self, name):
        return float(np.mean(self.samples[name]))

    def std(self, name):
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
        head = (
            f"{'parameter':<28}{'prior mean':>11}{'prior std':>10}{'post. mean':>11}"
            f"{'post. std':>10}{'MAP':>10}  {lv + ' HPD':<22}{'contr.':>7}{'R-hat':>7}"
            f"{'ESS':>7}  note"
        )
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
                f"{n:<28}{d.mean:11.4g}{d.std:10.3g}{self.mean(n):11.4g}{self.std(n):10.3g}"
                f"{mp[n]:10.4g}  [{lo:9.4g}, {hi:9.4g}] {self.contraction[n]:7.2f}"
                f"{self.r_hat[n]:7.3f}{self.effective_sample_size[n]:7.0f}  {'; '.join(note)}"
            )
        lines.append("")
        lines.append(f"{len(self.x)} samples, acceptance {self.acceptance:.2f}")
        if self.surrogate is not None:
            q2 = np.atleast_1d(self.surrogate.q2)
            lines.append(
                f"surrogate: {len(self.surrogate._u_train)} runs, {self.surrogate.components} "
                f"component(s), leave-one-out Q2 {np.nanmin(q2):.3f} (lowest) to "
                f"{np.nanmax(q2):.3f}"
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
        measurements inside ``[low, high]``)."""
        ctx = self._ctx
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(self.x), size=min(samples, len(self.x)), replace=False)
        theta = self.x[idx]
        mean, v = ctx["predict"](theta)
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
    progress=False,
) -> Posterior:
    r"""Sample the posterior distribution of the ``inputs`` of ``model``
    given measurements; see the module documentation.

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
    """
    dists = check_inputs(inputs)
    names = list(dists)
    k = len(names)
    onames, shapes, y, Sn = _layout(observed, noise)
    rng = np.random.default_rng(seed)
    context = dict(names=onames, shapes=shapes, y=y)

    gp = None
    if surrogate == "gaussian_process":
        from .gaussian_process import GaussianProcess
        from .propagate import propagate

        if training is None:
            m = int(training_samples) if training_samples else max(10 * k, 30)
            training = propagate(
                model, dists, m, "latin_hypercube", seed, processes, store, progress
            )
        if list(training.distributions) != names:
            raise ValueError("calibrate: the training runs must have the same inputs, in order.")
        missing = [n for n in onames if n not in training.outputs]
        if missing:
            raise KeyError(f"calibrate: the training runs lack the outputs {missing}.")
        # Check the shapes of the training outputs against the measurements.
        _flatten({n: training.outputs[n][:1] for n in onames}, onames, shapes, 1)
        gp = GaussianProcess(seed=seed).fit(training, output=onames)
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
            return mean, None
    else:
        raise ValueError("calibrate: surrogate must be 'gaussian_process' or None.")
    context["predict"] = predict

    # Discrepancy, fitted at the prior mean and held fixed.
    delta = np.zeros_like(y)
    Sdelta = np.zeros_like(Sn)
    if discrepancy == "gaussian_process":
        theta0 = np.array([[dists[n].mean for n in names]])
        y0, v0 = predict(theta0)
        resid = y - y0[0]
        info = {}
        j = 0
        for name, shape in zip(onames, shapes):
            m = int(np.prod(shape)) if shape else 1
            sl = slice(j, j + m)
            loc = (
                np.asarray(locations[name], dtype=float).reshape(m, -1)
                if locations and name in locations
                else np.arange(m, dtype=float)
            )
            ncov = Sn[sl, sl] + Sd[sl, sl]
            if A is not None and v0 is not None:
                ncov = ncov + (A[sl] * v0[0]) @ A[sl].T
            dm, dc, hyp = _fit_discrepancy(resid[sl], loc, ncov)
            delta[sl], Sdelta[sl, sl] = dm, dc
            info[name] = (dm.reshape(shape), dc, hyp)
            j += m
        context["discrepancy"] = info
    elif discrepancy is not None:
        raise ValueError("calibrate: discrepancy must be None or 'gaussian_process'.")
    context["delta"] = delta

    D0 = Sn + Sd + Sdelta
    D0 = 0.5 * (D0 + D0.T) + 1e-12 * np.mean(np.diag(D0)) * np.eye(len(y))
    context["D0"] = D0
    like = _Likelihood(y - delta, D0, A)

    def logpost(theta):
        lp = np.zeros(len(theta))
        for j, n in enumerate(names):
            lp += dists[n].logpdf(theta[:, j])
        ok = np.isfinite(lp)
        if np.any(ok):
            mean, v = predict(theta[ok])
            lp[ok] += like(mean, v)
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
        cov0 = np.diag([dists[n].std ** 2 for n in names])
        chain, lps, acc = _metropolis(logpost, x0, cov0, 2 * keep, rng)
    else:
        raise ValueError("calibrate: sampler must be ensemble or metropolis.")
    chain, lps = chain[keep:], lps[keep:]
    flat = chain.reshape(-1, k)
    return Posterior(dists, flat, lps.reshape(-1), chain, acc, level, context)
