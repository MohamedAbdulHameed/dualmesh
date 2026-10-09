# SPDX-License-Identifier: LGPL-2.1-or-later
r"""Probability distributions of uncertain inputs.

Each distribution maps a number :math:`u` in :math:`(0, 1)` to a value by its
inverse cumulative distribution function (:meth:`Distribution.ppf`), so that
any design of the unit cube (Monte Carlo, Latin hypercube, Sobol' sequence)
becomes a sample of the inputs.  The same objects give the prior densities of
a calibration (:meth:`Distribution.logpdf`).

Every distribution may be truncated to ``[lower, upper]``.  A truncated
distribution is the untruncated one conditioned on the interval, so its
density is the untruncated density divided by the probability of the
interval, and its moments are those of the conditioned variable.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import special

__all__ = ["Distribution", "Normal", "LogNormal", "Uniform", "LogUniform"]

# Two-sided 95 % point of the standard normal distribution.
Z95 = float(special.ndtri(0.975))


def _phi(z):
    """Standard normal density, zero at infinite arguments."""
    z = np.asarray(z, dtype=float)
    with np.errstate(over="ignore"):
        return np.where(np.isfinite(z), np.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi), 0.0)


class Distribution:
    """Base class: a one-dimensional distribution.

    Subclasses define :meth:`ppf`, :meth:`logpdf`, :meth:`cdf`, the support
    (``lower``, ``upper``) and the moments (:attr:`mean`, :attr:`standard_deviation`).
    """

    lower: float
    upper: float

    def ppf(self, u):
        """Inverse cumulative distribution function at ``u`` in (0, 1)."""
        raise NotImplementedError

    def cdf(self, x):
        """Cumulative distribution function."""
        raise NotImplementedError

    def logpdf(self, x):
        """Natural logarithm of the density, ``-inf`` outside the support."""
        raise NotImplementedError

    def pdf(self, x):
        """Probability density."""
        return np.exp(self.logpdf(x))

    @property
    def mean(self) -> float:
        raise NotImplementedError

    @property
    def standard_deviation(self) -> float:
        raise NotImplementedError

    def sample(self, size, seed=None):
        """``size`` independent random values."""
        rng = np.random.default_rng(seed)
        return self.ppf(rng.random(size))

    def interval(self, level: float = 0.95):
        """The central interval holding the fraction ``level`` of the
        probability."""
        a = 0.5 * (1.0 - level)
        return float(self.ppf(a)), float(self.ppf(1.0 - a))


class _GaussianBase(Distribution):
    """A normal variable :math:`Y \\sim N(\\mu, s^2)` truncated to
    ``[a, b]`` in :math:`Y`.  The subclasses map :math:`Y` to the value."""

    def _setup(self, mu, s, a, b, name):
        if not s > 0.0 or not math.isfinite(s):
            raise ValueError(f"{name}: the standard deviation must be positive and finite.")
        if a >= b:
            raise ValueError(f"{name}: lower must be below upper.")
        self._mu, self._s = float(mu), float(s)
        self._alpha = (a - mu) / s
        self._beta = (b - mu) / s
        self._pa = float(special.ndtr(self._alpha))
        self._pb = float(special.ndtr(self._beta))
        # In the upper tail the difference of the survival functions keeps
        # its digits; the difference of the distribution functions would not.
        self._upper_tail = self._alpha > 0.0
        if self._upper_tail:
            self._z = float(special.ndtr(-self._alpha) - special.ndtr(-self._beta))
        else:
            self._z = self._pb - self._pa
        if not self._z > 1e-300:
            raise ValueError(f"{name}: the interval [lower, upper] holds no probability.")
        # Log of the normalising probability, accurate in the tails.
        if self._upper_tail:
            # Both bounds in the upper tail: use the survival functions.
            self._logz = float(
                special.log_ndtr(-self._alpha)
                + np.log1p(-np.exp(special.log_ndtr(-self._beta) - special.log_ndtr(-self._alpha)))
            )
        else:
            self._logz = math.log(self._z)

    def _ppf_y(self, u):
        u = np.asarray(u, dtype=float)
        if self._upper_tail:
            # Upper tail: invert the survival function for accuracy.
            qa, qb = special.ndtr(-self._alpha), special.ndtr(-self._beta)
            return self._mu - self._s * special.ndtri(qa - u * (qa - qb))
        return self._mu + self._s * special.ndtri(self._pa + u * self._z)

    def _cdf_y(self, y):
        z = (np.asarray(y, dtype=float) - self._mu) / self._s
        if self._upper_tail:
            c = (special.ndtr(-self._alpha) - special.ndtr(-z)) / self._z
        else:
            c = (special.ndtr(z) - self._pa) / self._z
        return np.clip(c, 0.0, 1.0)

    def _logpdf_y(self, y):
        y = np.asarray(y, dtype=float)
        z = (y - self._mu) / self._s
        out = -0.5 * z * z - math.log(self._s * math.sqrt(2.0 * math.pi)) - self._logz
        inside = (z >= self._alpha) & (z <= self._beta)
        return np.where(inside, out, -np.inf)


class Normal(_GaussianBase):
    r"""Normal distribution of mean ``mean`` and standard deviation ``standard_deviation``,
    truncated to ``[lower, upper]`` when a bound is given.

    For a truncated normal, ``mean`` and ``standard_deviation`` are the parameters of the
    untruncated distribution, and :attr:`mean` and :attr:`standard_deviation` return the
    moments of the truncated one:

    .. math::

        E[X] = \mu + \sigma \frac{\phi(\alpha) - \phi(\beta)}{Z}, \qquad
        \mathrm{Var}[X] = \sigma^2 \left[ 1 + \frac{\alpha\phi(\alpha) -
        \beta\phi(\beta)}{Z} - \left(\frac{\phi(\alpha) -
        \phi(\beta)}{Z}\right)^2 \right],

    with :math:`\alpha = (a-\mu)/\sigma`, :math:`\beta = (b-\mu)/\sigma` and
    :math:`Z = \Phi(\beta) - \Phi(\alpha)`.
    """

    def __init__(self, mean: float, standard_deviation: float, lower=None, upper=None):
        self.lower = -math.inf if lower is None else float(lower)
        self.upper = math.inf if upper is None else float(upper)
        self._setup(mean, standard_deviation, self.lower, self.upper, "Normal")

    def __repr__(self):
        bounds = ""
        if math.isfinite(self.lower) or math.isfinite(self.upper):
            bounds = f", lower={self.lower:g}, upper={self.upper:g}"
        return f"Normal(mean={self._mu:g}, standard_deviation={self._s:g}{bounds})"

    def ppf(self, u):
        return np.clip(self._ppf_y(u), self.lower, self.upper)

    def cdf(self, x):
        return self._cdf_y(x)

    def logpdf(self, x):
        return self._logpdf_y(x)

    @property
    def mean(self) -> float:
        a, b = self._alpha, self._beta
        return self._mu + self._s * float(_phi(a) - _phi(b)) / self._z

    @property
    def standard_deviation(self) -> float:
        a, b = self._alpha, self._beta
        pa, pb = float(_phi(a)), float(_phi(b))
        ta = a * pa if math.isfinite(a) else 0.0
        tb = b * pb if math.isfinite(b) else 0.0
        d = (pa - pb) / self._z
        return self._s * math.sqrt(max(1.0 + (ta - tb) / self._z - d * d, 0.0))


class LogNormal(_GaussianBase):
    r"""Log-normal distribution: :math:`\ln X \sim N(\ln m, \sigma^2)`,
    with median :math:`m` (``median``).

    Give the spread either as ``sigma`` (the standard deviation of
    :math:`\ln X`) or as ``factor`` :math:`f`: 95 % of the values lie
    between :math:`m/f` and :math:`m f`, so that :math:`\sigma = \ln f /
    1.95996`.  A "factor of 100 range" from 0.1 to 10 at 95 % is
    ``factor=10``.  The distribution is truncated to ``[lower, upper]`` when
    a bound is given (bounds on :math:`X`, both positive).

    Moments, with :math:`\alpha, \beta` the bounds of :math:`(\ln X - \ln
    m)/\sigma` and :math:`Z = \Phi(\beta) - \Phi(\alpha)`:

    .. math::

        E[X^n] = m^n e^{n^2\sigma^2/2}\,
        \frac{\Phi(\beta - n\sigma) - \Phi(\alpha - n\sigma)}{Z}.
    """

    def __init__(self, median: float, factor=None, sigma=None, lower=None, upper=None):
        if not median > 0.0:
            raise ValueError("LogNormal: median must be positive.")
        if (factor is None) == (sigma is None):
            raise ValueError("LogNormal: give either factor or sigma.")
        if factor is not None:
            if not factor > 1.0:
                raise ValueError("LogNormal: factor must exceed 1.")
            sigma = math.log(float(factor)) / Z95
        self.median = float(median)
        self.sigma = float(sigma)
        self.lower = 0.0 if lower is None else float(lower)
        self.upper = math.inf if upper is None else float(upper)
        if self.lower < 0.0:
            raise ValueError("LogNormal: lower must not be negative.")
        a = -math.inf if self.lower == 0.0 else math.log(self.lower)
        b = math.log(self.upper) if math.isfinite(self.upper) else math.inf
        self._setup(math.log(self.median), self.sigma, a, b, "LogNormal")

    def __repr__(self):
        return f"LogNormal(median={self.median:g}, sigma={self.sigma:g})"

    def ppf(self, u):
        return np.clip(np.exp(self._ppf_y(u)), self.lower, self.upper)

    def cdf(self, x):
        x = np.asarray(x, dtype=float)
        with np.errstate(divide="ignore"):
            return np.where(x > 0.0, self._cdf_y(np.log(np.where(x > 0.0, x, 1.0))), 0.0)

    def logpdf(self, x):
        x = np.asarray(x, dtype=float)
        pos = x > 0.0
        lx = np.log(np.where(pos, x, 1.0))
        return np.where(pos, self._logpdf_y(lx) - lx, -np.inf)

    def _moment(self, n):
        s, a, b = self._s, self._alpha, self._beta
        num = special.ndtr(b - n * s) - special.ndtr(a - n * s)
        return self.median**n * math.exp(0.5 * n * n * s * s) * float(num) / self._z

    @property
    def mean(self) -> float:
        return self._moment(1)

    @property
    def standard_deviation(self) -> float:
        m1 = self._moment(1)
        return math.sqrt(max(self._moment(2) - m1 * m1, 0.0))


class Uniform(Distribution):
    """Uniform distribution on ``[lower, upper]``."""

    def __init__(self, lower: float, upper: float):
        if not float(lower) < float(upper):
            raise ValueError("Uniform: lower must be below upper.")
        self.lower, self.upper = float(lower), float(upper)

    def __repr__(self):
        return f"Uniform(lower={self.lower:g}, upper={self.upper:g})"

    def ppf(self, u):
        return self.lower + np.asarray(u, dtype=float) * (self.upper - self.lower)

    def cdf(self, x):
        return np.clip((np.asarray(x, dtype=float) - self.lower) / (self.upper - self.lower), 0, 1)

    def logpdf(self, x):
        x = np.asarray(x, dtype=float)
        inside = (x >= self.lower) & (x <= self.upper)
        return np.where(inside, -math.log(self.upper - self.lower), -np.inf)

    @property
    def mean(self) -> float:
        return 0.5 * (self.lower + self.upper)

    @property
    def standard_deviation(self) -> float:
        return (self.upper - self.lower) / math.sqrt(12.0)


class LogUniform(Distribution):
    r"""Log-uniform distribution: :math:`\ln X` uniform on
    :math:`[\ln a, \ln b]`, with :math:`0 < a < b`.  Its moments are
    :math:`E[X^n] = (b^n - a^n) / (n \ln(b/a))`."""

    def __init__(self, lower: float, upper: float):
        if not 0.0 < float(lower) < float(upper):
            raise ValueError("LogUniform: need 0 < lower < upper.")
        self.lower, self.upper = float(lower), float(upper)
        self._w = math.log(self.upper / self.lower)

    def __repr__(self):
        return f"LogUniform(lower={self.lower:g}, upper={self.upper:g})"

    def ppf(self, u):
        return self.lower * np.exp(np.asarray(u, dtype=float) * self._w)

    def cdf(self, x):
        x = np.asarray(x, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            c = np.log(np.maximum(x, 1e-300) / self.lower) / self._w
        return np.clip(c, 0.0, 1.0)

    def logpdf(self, x):
        x = np.asarray(x, dtype=float)
        inside = (x >= self.lower) & (x <= self.upper)
        with np.errstate(divide="ignore", invalid="ignore"):
            out = -np.log(np.where(inside, x, 1.0)) - math.log(self._w)
        return np.where(inside, out, -np.inf)

    def _moment(self, n):
        return (self.upper**n - self.lower**n) / (n * self._w)

    @property
    def mean(self) -> float:
        return self._moment(1)

    @property
    def standard_deviation(self) -> float:
        m1 = self._moment(1)
        return math.sqrt(max(self._moment(2) - m1 * m1, 0.0))
