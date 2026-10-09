# SPDX-License-Identifier: LGPL-2.1-or-later
r"""Gaussian process (kriging) surrogate of a model.

The output is modelled as :math:`y(x) = f(x)^T \beta + z(x)`, with a trend
:math:`f` (a constant or a linear function) and a stationary Gaussian process
:math:`z` of variance :math:`\sigma^2` and correlation :math:`R(x, x')`
(Sacks et al. 1989, Eqs. (7) and (8)).  With the correlation
matrix :math:`\mathbf R` of the training points (plus a nugget
:math:`\tau \mathbf I` for numerical noise), the generalised least-squares
trend :math:`\hat\beta = (F^T R^{-1} F)^{-1} F^T R^{-1} y` and the
maximum-likelihood variance :math:`\hat\sigma^2 = (y - F\hat\beta)^T R^{-1}
(y - F\hat\beta)/n`, the predictor and its variance are

.. math::

    \hat y(x) = f(x)^T\hat\beta + r(x)^T R^{-1}(y - F\hat\beta), \\
    s^2(x) = \hat\sigma^2 \left[1 - r^T R^{-1} r + u^T (F^T R^{-1} F)^{-1} u
    \right], \quad u = F^T R^{-1} r - f(x).

The length scales (one per input) and the nugget maximise the concentrated
likelihood :math:`-\tfrac{n}{2}\ln\hat\sigma^2 - \tfrac12 \ln|R|`, by L-BFGS-B
with its exact gradient from several starting points.

A vector output (a time series) is standardised column by column and reduced
to its principal components, which keep the fraction ``variance_kept`` of the
variance of the training outputs.  Each component score has its own process.
The variance of the discarded components is added to the predictive variance,
so that no part of the output is predicted with false confidence.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import linalg, optimize

__all__ = ["GaussianProcess"]

KERNELS = ("matern52", "matern32", "squared_exponential")
SQRT3, SQRT5 = math.sqrt(3.0), math.sqrt(5.0)


def _sq_dists(a, b, ell):
    """Scaled squared differences per dimension: (na, nb, k)."""
    d = (a[:, None, :] - b[None, :, :]) / ell
    return d * d


def _kernel(kind, d2):
    """Correlation from the per-dimension scaled squared differences, and
    the factor g with dk/dlog(ell_p) = g * d2_p."""
    r2 = d2.sum(axis=-1)
    if kind == "squared_exponential":
        k = np.exp(-0.5 * r2)
        return k, k
    r = np.sqrt(r2)
    if kind == "matern52":
        e = np.exp(-SQRT5 * r)
        return (1.0 + SQRT5 * r + 5.0 / 3.0 * r2) * e, 5.0 / 3.0 * (1.0 + SQRT5 * r) * e
    e = np.exp(-SQRT3 * r)
    return (1.0 + SQRT3 * r) * e, 3.0 * e


class _Process:
    """A Gaussian process for one scalar output on the unit cube."""

    def __init__(self, kernel, trend, nugget, restarts, rng):
        self.kernel, self.trend, self.nugget = kernel, trend, nugget
        self.restarts, self.rng = restarts, rng

    def _basis(self, x):
        if self.trend == "constant":
            return np.ones((len(x), 1))
        return np.column_stack([np.ones(len(x)), x])

    def _factor(self, log_ell, log_tau):
        ell = np.exp(log_ell)
        d2 = _sq_dists(self.x, self.x, ell)
        k, g = _kernel(self.kernel, d2)
        tau = math.exp(log_tau)
        R = k + (tau + 1e-10) * np.eye(len(self.x))
        return R, d2, g, tau

    def _nll(self, theta, grad=True):
        k = self.x.shape[1]
        log_ell = theta[:k]
        log_tau = theta[k] if self.fit_nugget else self.log_tau
        R, d2, g, tau = self._factor(log_ell, log_tau)
        try:
            c = linalg.cho_factor(R, lower=True, check_finite=False)
        except linalg.LinAlgError:
            return (1e25, np.zeros_like(theta)) if grad else 1e25
        n = len(self.y)
        Ri_F = linalg.cho_solve(c, self.F, check_finite=False)
        Ri_y = linalg.cho_solve(c, self.y, check_finite=False)
        G = self.F.T @ Ri_F
        beta = np.linalg.solve(G, self.F.T @ Ri_y)
        alpha = Ri_y - Ri_F @ beta
        e = self.y - self.F @ beta
        s2 = max(float(e @ alpha) / n, 1e-300)
        logdet = 2.0 * np.sum(np.log(np.diag(c[0])))
        nll = 0.5 * n * math.log(s2) + 0.5 * logdet
        if not grad:
            return nll
        Ri = linalg.cho_solve(c, np.eye(n), check_finite=False)
        W = Ri - np.outer(alpha, alpha) / s2
        gk = W * g
        dl = 0.5 * np.einsum("ij,ijp->p", gk, d2)
        out = [dl]
        if self.fit_nugget:
            out.append([0.5 * tau * np.trace(W)])
        return nll, np.concatenate(out)

    def fit(self, x, y):
        self.x = np.asarray(x, dtype=float)
        self.y = np.asarray(y, dtype=float)
        n, k = self.x.shape
        self.F = self._basis(self.x)
        if n <= self.F.shape[1]:
            raise ValueError("GaussianProcess: too few training points for the trend.")
        self.fit_nugget = self.nugget == "fit"
        if not self.fit_nugget:
            self.log_tau = math.log(max(float(self.nugget or 0.0), 1e-300))
        bounds = [(math.log(1e-2), math.log(1e2))] * k
        if self.fit_nugget:
            bounds.append((math.log(1e-10), math.log(1.0)))
        s0 = np.full(k, math.log(0.5))
        starts = [np.append(s0, math.log(1e-6)) if self.fit_nugget else s0]
        for _ in range(self.restarts - 1):
            s = self.rng.uniform(math.log(0.05), math.log(3.0), k)
            if self.fit_nugget:
                s = np.append(s, self.rng.uniform(math.log(1e-8), math.log(1e-2)))
            starts.append(s)
        best = None
        for s in starts:
            res = optimize.minimize(
                self._nll, s, jac=True, method="L-BFGS-B", bounds=bounds, options={"maxiter": 500}
            )
            if best is None or res.fun < best.fun:
                best = res
        theta = best.x
        self.log_ell = theta[:k]
        if self.fit_nugget:
            self.log_tau = theta[k]
        self._condition()
        return self

    def _condition(self):
        R, _, _, tau = self._factor(self.log_ell, self.log_tau)
        self.tau = tau
        self.c = linalg.cho_factor(R, lower=True, check_finite=False)
        Ri_F = linalg.cho_solve(self.c, self.F, check_finite=False)
        Ri_y = linalg.cho_solve(self.c, self.y, check_finite=False)
        self.G = self.F.T @ Ri_F
        self.Gc = linalg.cho_factor(self.G, lower=True, check_finite=False)
        self.beta = linalg.cho_solve(self.Gc, self.F.T @ Ri_y, check_finite=False)
        self.alpha = Ri_y - Ri_F @ self.beta
        self.s2 = float((self.y - self.F @ self.beta) @ self.alpha) / len(self.y)
        self.ell = np.exp(self.log_ell)

    def _cross(self, xs):
        return _kernel(self.kernel, _sq_dists(xs, self.x, self.ell))[0]

    def predict(self, xs, var=True):
        r = self._cross(xs)
        mean = self._basis(xs) @ self.beta + r @ self.alpha
        if not var:
            return mean, None
        Ri_r = linalg.cho_solve(self.c, r.T, check_finite=False)
        u = self.F.T @ Ri_r - self._basis(xs).T
        v = 1.0 - np.sum(r.T * Ri_r, axis=0)
        v += np.sum(u * linalg.cho_solve(self.Gc, u, check_finite=False), axis=0)
        return mean, self.s2 * np.maximum(v, 0.0)

    def covariance(self, xs):
        r = self._cross(xs)
        Ri_r = linalg.cho_solve(self.c, r.T, check_finite=False)
        u = self.F.T @ Ri_r - self._basis(xs).T
        kxx = _kernel(self.kernel, _sq_dists(xs, xs, self.ell))[0]
        C = kxx - r @ Ri_r + u.T @ linalg.cho_solve(self.Gc, u, check_finite=False)
        return self.s2 * 0.5 * (C + C.T)

    def leave_one_out(self):
        """Closed-form leave-one-out residuals, with the hyperparameters
        fixed and the trend re-estimated without each point (Dubrule 1983):
        :math:`e_i = (Q y)_i / Q_{ii}`, :math:`\\mathrm{var}_i =
        \\hat\\sigma^2 / Q_{ii}`, :math:`Q = R^{-1} - R^{-1}F G^{-1} F^T
        R^{-1}`."""
        n = len(self.y)
        Ri = linalg.cho_solve(self.c, np.eye(n), check_finite=False)
        RiF = Ri @ self.F
        Q = Ri - RiF @ linalg.cho_solve(self.Gc, RiF.T, check_finite=False)
        q = np.diag(Q)
        resid = (Q @ self.y) / q
        return self.y - resid, self.s2 / q


class GaussianProcess:
    """A Gaussian process (kriging) surrogate.  See the module documentation.

    ``kernel``
        ``matern52`` (default, twice differentiable, one of the Matern kernels
        that Wu et al. 2018, Table 1, list as widely used),
        ``matern32`` or ``squared_exponential``.
    ``trend``
        ``constant`` (ordinary kriging, default) or ``linear``.
    ``nugget``
        ``"fit"`` (default: a small white-noise variance estimated with the
        length scales, for models with numerical noise) or a fixed ratio of
        the noise to the process variance (0 interpolates exactly).
    ``variance_kept``
        For vector outputs, the fraction of the variance kept by the
        principal components.  Default 0.999.
    ``restarts``
        Starting points of the likelihood maximisation.  Default 5.
    """

    def __init__(
        self,
        kernel="matern52",
        trend="constant",
        nugget="fit",
        variance_kept=0.999,
        restarts=5,
        seed=0,
    ):
        if kernel not in KERNELS:
            raise ValueError(f"GaussianProcess: kernel must be one of {', '.join(KERNELS)}.")
        if trend not in ("constant", "linear"):
            raise ValueError("GaussianProcess: trend must be constant or linear.")
        if nugget != "fit" and not (isinstance(nugget, (int, float)) and nugget >= 0):
            raise ValueError("GaussianProcess: nugget must be 'fit' or a number >= 0.")
        if not 0.0 < variance_kept <= 1.0:
            raise ValueError("GaussianProcess: variance_kept must lie in (0, 1].")
        self.kernel, self.trend, self.nugget = kernel, trend, nugget
        self.variance_kept, self.restarts, self.seed = variance_kept, int(restarts), seed
        self._fitted = False
        self.output_names = None

    # ---- input and output mappings --------------------------------------------
    def _transform(self, x):
        """Inputs of log-normal and log-uniform distributions enter by
        their logarithm, the others as they are."""
        x = np.atleast_2d(np.asarray(x, dtype=float))
        if self._log is None or not np.any(self._log):
            return x
        z = x.copy()
        with np.errstate(divide="ignore", invalid="ignore"):
            z[:, self._log] = np.log(x[:, self._log])
        return z

    def _to_unit(self, x):
        return (self._transform(x) - self._lo) / self._span

    def _pack(self, y):
        """Flatten an output (array or dict of arrays, first axis = runs)."""
        if isinstance(y, dict):
            self._layout = [(k, np.asarray(v).shape[1:]) for k, v in y.items()]
            return np.column_stack([np.asarray(v, float).reshape(len(v), -1) for v in y.values()])
        y = np.asarray(y, dtype=float)
        self._layout = None
        self._yshape = y.shape[1:]
        return y.reshape(len(y), -1)

    def _unpack(self, flat):
        """Flat (n, m) -> the output layout of the training data."""
        if self._layout is None:
            n = flat.shape[0]
            return flat.reshape((n,) + self._yshape)
        out, j = {}, 0
        for k, shape in self._layout:
            m = int(np.prod(shape)) if shape else 1
            out[k] = flat[:, j : j + m].reshape((flat.shape[0],) + shape)
            j += m
        return out

    # ---- fitting -----------------------------------------------------------------
    def fit(self, x, y=None, output=None):
        """Fit to training data.

        ``fit(runs)`` uses the runs of :func:`propagate` (all the outputs, or
        those named by ``output``, a name or a list of names).  Inputs with a
        log-normal or log-uniform distribution enter by their logarithm, in
        which such models are usually smooth.  In both forms the inputs are
        scaled to the unit cube by the range of the training points.  ``fit(x, y)`` takes an
        array of inputs (n, k), scaled to the unit cube by their range, and
        an array (n,) or (n, m) or a dict of such arrays.
        """
        from .propagate import Runs

        if isinstance(x, Runs):
            runs = x
            if output is None:
                outs = runs.outputs
            else:
                wanted = [output] if isinstance(output, str) else list(output)
                outs = {n: runs.outputs[n] for n in wanted}
            ok = np.ones(len(runs), dtype=bool)
            for v in outs.values():
                ok &= np.all(np.isfinite(v.reshape(len(v), -1)), axis=1)
            from .distributions import LogNormal, LogUniform

            self.input_names = runs.names
            self._log = np.array(
                [isinstance(d, (LogNormal, LogUniform)) for d in runs.distributions.values()]
            )
            z = self._transform(runs.x[ok])
            self._lo = z.min(axis=0)
            self._span = np.where(z.max(axis=0) > self._lo, z.max(axis=0) - self._lo, 1.0)
            u = (z - self._lo) / self._span
            # One output predicts as an array, several as a dict.
            flat = (
                self._pack(next(iter(outs.values()))[ok])
                if len(outs) == 1
                else self._pack({k: v[ok] for k, v in outs.items()})
            )
            self.output_names = list(outs)
        else:
            if y is None:
                raise ValueError("GaussianProcess.fit: give fit(runs) or fit(x, y).")
            x = np.atleast_2d(np.asarray(x, dtype=float))
            if x.shape[0] == 1 and x.shape[1] > 1 and np.ndim(y) == 1 and len(y) == x.shape[1]:
                x = x.T
            self._log = None
            self.input_names = None
            self._lo = x.min(axis=0)
            self._span = np.where(x.max(axis=0) > self._lo, x.max(axis=0) - self._lo, 1.0)
            u = (x - self._lo) / self._span
            flat = self._pack(y)
            ok = np.all(np.isfinite(flat), axis=1) & np.all(np.isfinite(u), axis=1)
            u, flat = u[ok], flat[ok]
        n, m = flat.shape
        if n < 3:
            raise ValueError("GaussianProcess.fit: need at least 3 training points.")
        self._mu = flat.mean(axis=0)
        sd = flat.std(axis=0)
        self._sd = np.where(sd > 1e-300 * np.maximum(1.0, np.abs(self._mu)), sd, 1.0)
        z = (flat - self._mu) / self._sd
        if m == 1:
            self._V = np.ones((1, 1))
            scores = z
            self._discarded = np.zeros(1)
        else:
            _, s, vt = np.linalg.svd(z, full_matrices=False)
            frac = np.cumsum(s * s) / max(np.sum(s * s), 1e-300)
            r = int(np.searchsorted(frac, self.variance_kept - 1e-12) + 1)
            r = max(1, min(r, len(s)))
            self._V = vt[:r].T
            scores = z @ self._V
            resid = z - scores @ self._V.T
            self._discarded = np.mean(resid * resid, axis=0)
        rng = np.random.default_rng(self.seed)
        self._processes = [
            _Process(self.kernel, self.trend, self.nugget, self.restarts, rng).fit(u, scores[:, c])
            for c in range(scores.shape[1])
        ]
        self._u_train = u
        self._flat_train = flat
        self._fitted = True
        self._loo = None
        return self

    @property
    def components(self) -> int:
        """Number of principal components (1 for a scalar output)."""
        return len(self._processes)

    @property
    def length_scales(self):
        """Length scales of each process, in the unit cube: (components, k)."""
        return np.array([p.ell for p in self._processes])

    # ---- prediction ----------------------------------------------------------------
    def _predict_flat(self, u, var=True):
        """Mean (nx, m) in output units, and the variances of the component
        scores (nx, r) in standardised units."""
        means, variances = [], []
        for p in self._processes:
            mu, v = p.predict(u, var)
            means.append(mu)
            variances.append(v)
        t = np.column_stack(means)
        mean = self._mu + (t @ self._V.T) * self._sd
        return mean, (np.column_stack(variances) if var else None)

    def predict(self, x, return_standard_deviation=False, return_cov=False):
        """Predict at the inputs ``x`` (n, k), the input values themselves
        (also for a fit to runs).  Returns
        the mean, and with ``return_standard_deviation`` the standard deviation, or with
        ``return_cov`` the covariance between the output components at each
        point (n, m, m).  Both include the variance of the discarded
        principal components."""
        self._check()
        u = self._to_unit(x)
        mean, pv = self._predict_flat(u, return_standard_deviation or return_cov)
        if not (return_standard_deviation or return_cov):
            return self._unpack(mean)
        sd2 = self._sd * self._sd
        if return_cov:
            cov = np.einsum("jc,nc,kc->njk", self._V, pv, self._V)
            cov += np.diag(self._discarded)[None]
            cov *= np.sqrt(sd2)[None, :, None] * np.sqrt(sd2)[None, None, :]
            return self._unpack(mean), cov
        var = (pv @ (self._V * self._V).T + self._discarded) * sd2
        return self._unpack(mean), self._unpack(np.sqrt(var))

    def sample(self, x, size, seed=None):
        """Joint random functions of the process at the points ``x``:
        ``size`` realisations, (size, n) or (size, n, m) or a dict."""
        self._check()
        rng = np.random.default_rng(seed)
        u = self._to_unit(x)
        nx = len(u)
        t = np.empty((size, nx, len(self._processes)))
        for c, p in enumerate(self._processes):
            mu, _ = p.predict(u, var=False)
            C = p.covariance(u)
            w, v = np.linalg.eigh(C)
            L = v * np.sqrt(np.maximum(w, 0.0))
            t[:, :, c] = mu + rng.standard_normal((size, nx)) @ L.T
        z = t @ self._V.T
        z += rng.standard_normal(z.shape) * np.sqrt(self._discarded)
        flat = self._mu + z * self._sd
        if self._layout is None and self._yshape == ():
            return flat[..., 0]
        return (
            np.stack([self._unpack(f) for f in flat])
            if self._layout is None
            else {k: np.stack([self._unpack(f)[k] for f in flat]) for k, _ in self._layout}
        )

    # ---- diagnostics ---------------------------------------------------------------
    def leave_one_out(self) -> dict:
        """Leave-one-out cross-validation in closed form (no refits).

        Returns a dict with ``prediction`` and ``standard_deviation`` at each training point
        (flattened outputs, (n, m)), ``q2`` (the predictivity coefficient
        :math:`Q^2 = 1 - \\sum (y_i - \\hat y_{-i})^2 / \\sum (y_i - \\bar
        y)^2` per output component, the predictivity coefficient of Marrel et
        al. (2009) on the left-out points.  They regard a surrogate with
        :math:`Q^2` below 0.7 as a poor approximation) and
        ``standardized_residuals`` (should lie within :math:`\\pm 3`)."""
        self._check()
        if self._loo is None:
            t, v = [], []
            for p in self._processes:
                tp, vp = p.leave_one_out()
                t.append(tp)
                v.append(vp)
            t, v = np.column_stack(t), np.column_stack(v)
            pred = self._mu + (t @ self._V.T) * self._sd
            var = (v @ (self._V * self._V).T + self._discarded) * self._sd**2
            y = self._flat_train
            sst = np.sum((y - y.mean(axis=0)) ** 2, axis=0)
            sse = np.sum((y - pred) ** 2, axis=0)
            with np.errstate(invalid="ignore", divide="ignore"):
                q2 = np.where(sst > 0, 1.0 - sse / sst, np.nan)
                zres = (y - pred) / np.sqrt(var)
            self._loo = dict(
                prediction=pred, standard_deviation=np.sqrt(var), q2=q2, standardized_residuals=zres
            )
        return self._loo

    @property
    def q2(self):
        """Leave-one-out predictivity coefficient (a number for a scalar
        output, an array per component otherwise)."""
        q = self.leave_one_out()["q2"]
        return float(q[0]) if q.size == 1 else q

    def _check(self):
        if not self._fitted:
            raise RuntimeError("GaussianProcess: call fit first.")

    def __repr__(self):
        if not self._fitted:
            return f"GaussianProcess(kernel={self.kernel!r}, not fitted)"
        return f"GaussianProcess(kernel={self.kernel!r}, {len(self._u_train)} points, {self.components} component(s))"
