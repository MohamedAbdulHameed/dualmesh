# SPDX-License-Identifier: LGPL-2.1-or-later
"""Verification of the uncertainty quantification module against closed
forms: moments of the distributions, propagation of a linear model, Wilks
limits and their coverage, the Sobol' indices of the Ishigami function, the
Gaussian process (interpolation, closed-form leave-one-out), and Bayesian
calibration of a linear-Gaussian model whose posterior is known exactly."""

from __future__ import annotations

import math

import numpy as np
import pytest

pytest.importorskip("scipy")
from dualmesh import uq  # noqa: E402
from dualmesh.uq.gaussian_process import _Process  # noqa: E402
from dualmesh.uq.propagate import Runs  # noqa: E402
from scipy import integrate, stats  # noqa: E402


# ---------------------------------------------------------------------------
# distributions
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("dist", [uq.Normal(1.0, 0.3), uq.Normal(1.0, 0.3, lower=0.2), uq.Normal(1.0, 0.3, lower=0.9, upper=1.6), uq.LogNormal(median=1.0, factor=10), uq.LogNormal(median=2.0, sigma=0.4, lower=1.0, upper=5.0), uq.Uniform(-1.0, 3.0), uq.LogUniform(0.1, 10.0)])
def test_distribution_moments_and_quantiles(dist):
    """Mean and standard deviation against numerical integration of the
    density, the density integrates to one, and ppf inverts cdf."""
    lo = dist.lower if math.isfinite(dist.lower) else dist.ppf(1e-12)
    hi = dist.upper if math.isfinite(dist.upper) else dist.ppf(1 - 1e-12)
    pts = [float(dist.ppf(q)) for q in (0.01, 0.5, 0.99)]
    total = integrate.quad(lambda x: float(dist.pdf(x)), lo, hi, points=pts, limit=200)[0]
    m1 = integrate.quad(lambda x: x * float(dist.pdf(x)), lo, hi, points=pts, limit=200)[0]
    m2 = integrate.quad(lambda x: x * x * float(dist.pdf(x)), lo, hi, points=pts, limit=200)[0]
    assert total == pytest.approx(1.0, rel=1e-6)
    assert dist.mean == pytest.approx(m1, rel=1e-6, abs=1e-9)
    assert dist.standard_deviation == pytest.approx(math.sqrt(m2 - m1 * m1), rel=1e-5)
    u = np.linspace(0.01, 0.99, 25)
    assert np.allclose(dist.cdf(dist.ppf(u)), u, atol=1e-9)
    assert np.all(np.isinf(dist.logpdf(np.array([lo - 1.0]))) | (lo == -np.inf) | (lo <= 0))


def test_lognormal_factor_is_the_95_percent_range():
    d = uq.LogNormal(median=1.0, factor=10)
    lo, hi = d.interval(0.95)
    assert lo == pytest.approx(0.1, rel=1e-12) and hi == pytest.approx(10.0, rel=1e-12)
    # The sigma of Che et al. (2018), Table 2: 0.5 in log10 is +/-2 sigma.
    che = uq.LogNormal(median=1.0, sigma=0.5 * math.log(10.0))
    assert che.interval(0.9545)[1] == pytest.approx(10.0, rel=1e-3)
    with pytest.raises(ValueError):
        uq.LogNormal(median=1.0)
    with pytest.raises(ValueError):
        uq.LogNormal(median=1.0, factor=2, sigma=0.3)


def test_normal_upper_tail_truncation_is_accurate():
    """A normal truncated far in its upper tail: the ppf stays inside the
    bounds and the mean matches the closed form."""
    d = uq.Normal(0.0, 1.0, lower=6.0, upper=7.0)
    x = d.ppf(np.linspace(0.0, 1.0, 11))
    assert x.min() >= 6.0 and x.max() <= 7.0
    ref = stats.truncnorm(6.0, 7.0)
    assert d.mean == pytest.approx(ref.mean(), rel=1e-10)
    assert d.standard_deviation == pytest.approx(ref.std(), rel=1e-7)


# ---------------------------------------------------------------------------
# propagation
# ---------------------------------------------------------------------------
def _linear(a, b):
    return {"y": 2.0 * a + 3.0 * b, "v": np.array([a, b, a * b])}


def test_propagate_linear_model_moments():
    """y = 2a + 3b with a ~ N(1, 0.1), b ~ U(0, 1): mean 3.5 and variance
    4(0.01) + 9/12 = 0.79."""
    inputs = {"a": uq.Normal(1.0, 0.1), "b": uq.Uniform(0.0, 1.0)}
    runs = uq.propagate(_linear, inputs, 4000, seed=3)
    assert runs.mean("y") == pytest.approx(3.5, abs=5e-3)
    assert runs.standard_deviation("y") ** 2 == pytest.approx(0.79, rel=0.02)
    assert runs.outputs["v"].shape == (4000, 3)
    rho = runs.sensitivity("y")
    assert rho["b"] > 0.9 and abs(rho["a"]) < 0.3
    prcc = runs.sensitivity("y", "prcc")
    assert prcc["a"] > 0.9 and prcc["b"] > 0.99
    value, low, high = runs.bootstrap("y", "mean")
    assert low < 3.5 < high
    text = runs.summary()
    assert "4000 runs" in text and "y: mean" in text


def test_designs_converge_in_the_expected_order():
    """RMS error of the mean of exp(x1 + x2), x uniform on (0, 1)^2, over 20
    designs of 256 runs: Latin hypercube beats Monte Carlo, and the Sobol'
    sequence beats both."""
    exact = (math.e - 1.0) ** 2

    def f(x1, x2):
        return math.exp(x1 + x2)

    inputs = {"x1": uq.Uniform(0, 1), "x2": uq.Uniform(0, 1)}
    err = {}
    for method in ("monte_carlo", "latin_hypercube", "sobol"):
        e = [uq.propagate(f, inputs, 256, method, seed=s).mean("output") - exact for s in range(20)]
        err[method] = math.sqrt(np.mean(np.square(e)))
    assert err["latin_hypercube"] < 0.5 * err["monte_carlo"]
    assert err["sobol"] < 0.5 * err["latin_hypercube"]


def test_wilks_sample_numbers_and_coverage():
    assert uq.wilks_samples() == 59
    assert uq.wilks_samples(side="two") == 93
    assert uq.wilks_samples(order=2) == 93
    assert uq.wilks_samples(0.95, 0.99) == 90
    # Coverage: in 4000 studies of 59 standard normal runs, the largest run
    # exceeds the 95 % quantile in at least 95 % of studies (exactly
    # 1 - 0.95^59 = 0.9515).
    rng = np.random.default_rng(0)
    q95 = stats.norm.ppf(0.95)
    hits = 0
    dists = {"x": uq.Normal(0, 1)}
    for _ in range(4000):
        x = rng.standard_normal((59, 1))
        runs = Runs(dists, x, x, {"y": x[:, 0]}, {}, "monte_carlo")
        hits += runs.tolerance_limit("y") >= q95
    assert hits / 4000 > 0.94
    x = rng.standard_normal((30, 1))
    few = Runs(dists, x, x, {"y": x[:, 0]}, {}, "monte_carlo")
    with pytest.raises(ValueError, match="at least 59"):
        few.tolerance_limit("y")
    lhs = Runs(dists, x, x, {"y": x[:, 0]}, {}, "latin_hypercube")
    with pytest.raises(ValueError, match="monte_carlo"):
        lhs.tolerance_limit("y")


def test_tolerance_limit_uses_the_tightest_order():
    """With 200 runs a 95/95 upper limit may use the 5th largest run."""
    x = np.arange(200.0)[:, None]
    runs = Runs({"x": uq.Uniform(0, 1)}, x, x, {"y": x[:, 0]}, {}, "monte_carlo")
    r = 1
    while stats.binom.cdf(200 - (r + 1), 200, 0.95) >= 0.95:
        r += 1
    assert runs.tolerance_limit("y") == 200 - r
    low, high = runs.tolerance_limit("y", side="two")
    assert low < 10 and high > 190


_CALLS = {"n": 0}


def _counted(a):
    _CALLS["n"] += 1
    if a > 0.95:
        raise RuntimeError("deliberate failure")
    return a * a


def test_store_resumes_and_records_failures(tmp_path):
    store = tmp_path / "runs.npz"
    inputs = {"a": uq.Uniform(0.0, 1.0)}
    _CALLS["n"] = 0
    first = uq.propagate(_counted, inputs, 40, seed=1, store=store)
    assert _CALLS["n"] == 40 and store.exists()
    assert len(first.failed) == int(np.sum(first.inputs["a"] > 0.95))
    second = uq.propagate(_counted, inputs, 40, seed=1, store=store)
    assert _CALLS["n"] == 40  # every run came from the store
    assert np.array_equal(np.isnan(second.outputs["output"]), np.isnan(first.outputs["output"]))
    assert "deliberate failure" in next(iter(second.failed.values()))


def test_store_finds_runs_whose_inputs_differ_in_the_last_digit(tmp_path):
    # Other versions of NumPy and SciPy calculate the inverse distribution
    # functions with a different last digit (6.7e-16 relative between two
    # studies). The store must find these runs.
    store = tmp_path / "runs.npz"
    inputs = {"a": uq.Normal(1.0, 0.1), "b": uq.LogNormal(median=1.0, factor=10.0)}
    _CALLS["n"] = 0
    uq.propagate(_counted_sum, inputs, 16, method="sobol", seed=0, store=store)
    assert _CALLS["n"] == 16
    with np.load(store, allow_pickle=False) as f:
        arrays = {k: f[k] for k in f.files}
    arrays["x"] = arrays["x"] * (1.0 + 4.0 * np.finfo(float).eps)
    np.savez(store, **arrays)
    uq.propagate(_counted_sum, inputs, 16, method="sobol", seed=0, store=store)
    assert _CALLS["n"] == 16
    arrays["x"] = arrays["x"] * (1.0 + 1e-9)
    np.savez(store, **arrays)
    uq.propagate(_counted_sum, inputs, 16, method="sobol", seed=0, store=store)
    assert _CALLS["n"] == 32


def _counted_sum(a, b):
    _CALLS["n"] += 1
    return a + b


def _slow_square(a):
    return a * a


def test_parallel_processes_give_the_serial_result():
    inputs = {"a": uq.Uniform(0.0, 1.0)}
    serial = uq.propagate(_slow_square, inputs, 12, seed=2)
    parallel = uq.propagate(_slow_square, inputs, 12, seed=2, processes=2)
    assert np.array_equal(serial.outputs["output"], parallel.outputs["output"])


# ---------------------------------------------------------------------------
# Sobol' indices
# ---------------------------------------------------------------------------
A_ISH, B_ISH = 7.0, 0.1


def _ishigami(x1, x2, x3):
    return math.sin(x1) + A_ISH * math.sin(x2) ** 2 + B_ISH * x3**4 * math.sin(x1)


def _ishigami_exact():
    a, b, p = A_ISH, B_ISH, math.pi
    V = a * a / 8 + b * p**4 / 5 + b * b * p**8 / 18 + 0.5
    V1 = b * p**4 / 5 + b * b * p**8 / 50 + 0.5
    V2 = a * a / 8
    V13 = 8 * b * b * p**8 / 225
    return (V1 / V, V2 / V, 0.0), ((V1 + V13) / V, V2 / V, V13 / V), V13 / V


def test_sobol_indices_of_the_ishigami_function():
    """First, total and second-order indices against the closed form
    (derived from the variance decomposition), with the run count
    n(2k+2) and intervals that hold the exact values."""
    inputs = {n: uq.Uniform(-math.pi, math.pi) for n in ("x1", "x2", "x3")}
    first, total, s13 = _ishigami_exact()
    s = uq.sobol(_ishigami, inputs, 2048, second_order=True, seed=1)
    assert s.runs == 2048 * (2 * 3 + 2)
    for j, n in enumerate(("x1", "x2", "x3")):
        assert s.first_order["output"][n] == pytest.approx(first[j], abs=0.03)
        assert s.total["output"][n] == pytest.approx(total[j], abs=0.03)
        lo, hi = s.first_order_interval["output"][n]
        assert lo - 0.01 < first[j] < hi + 0.01
        lo, hi = s.total_interval["output"][n]
        assert lo - 0.01 < total[j] < hi + 0.01
    assert s.second_order["output"][("x1", "x3")] == pytest.approx(s13, abs=0.05)
    assert abs(s.second_order["output"][("x1", "x2")]) < 0.05
    assert "x1" in s.summary()


def test_sobol_on_a_gaussian_process_surrogate():
    inputs = {n: uq.Uniform(-math.pi, math.pi) for n in ("x1", "x2", "x3")}
    first, total, _ = _ishigami_exact()
    s = uq.sobol(_ishigami, inputs, 1024, surrogate="gaussian_process", training_samples=200)
    assert s.runs == 200
    for j, n in enumerate(("x1", "x2", "x3")):
        assert s.first_order["output"][n] == pytest.approx(first[j], abs=0.06)
        assert s.total["output"][n] == pytest.approx(total[j], abs=0.06)


def test_sobol_on_given_training_runs():
    # Runs of propagate on a nested Sobol' design train the same surrogate as
    # training_samples, and a larger design reuses the runs of a smaller one.
    inputs = {n: uq.Uniform(-math.pi, math.pi) for n in ("x1", "x2", "x3")}
    first, total, _ = _ishigami_exact()
    runs = uq.propagate(_ishigami, inputs, 256, method="sobol")
    s = uq.sobol(_ishigami, inputs, 1024, surrogate="gaussian_process", training=runs)
    assert s.runs == 256
    for j, n in enumerate(("x1", "x2", "x3")):
        assert s.first_order["output"][n] == pytest.approx(first[j], abs=0.06)
        assert s.total["output"][n] == pytest.approx(total[j], abs=0.06)
    small = uq.propagate(_ishigami, inputs, 64, method="sobol")
    assert np.array_equal(small.x, runs.x[:64])
    lhs = uq.propagate(_ishigami, inputs, 200)
    a = uq.sobol(_ishigami, inputs, 256, surrogate="gaussian_process", training=lhs)
    b = uq.sobol(_ishigami, inputs, 256, surrogate="gaussian_process", training_samples=200)
    assert a.total["output"] == pytest.approx(b.total["output"], abs=1e-12)
    with pytest.raises(ValueError, match="not both"):
        uq.sobol(_ishigami, inputs, 64, surrogate="gaussian_process", training=runs, training_samples=50)
    with pytest.raises(ValueError, match="only with surrogate"):
        uq.sobol(_ishigami, inputs, 64, training=runs)


def test_the_interval_of_a_surrogate_index_holds_its_estimate():
    """The interval of an index on a Gaussian process is centred on the
    estimate on the mean of the process. On this design of 60 runs the
    interval of the realizations alone (Marrel et al. 2009, Eq. 11) gave the
    total index of x3 as 0.031 with the interval [0.049, 0.142]."""
    inputs = {n: uq.Uniform(-math.pi, math.pi) for n in ("x1", "x2", "x3")}
    runs = uq.propagate(_ishigami, inputs, 60, seed=101, report="none")
    s = uq.sobol(_ishigami, inputs, 1024, surrogate="gaussian_process", training=runs, seed=1, report="none")
    for n in ("x1", "x2", "x3"):
        for estimate, interval in ((s.first_order, s.first_order_interval), (s.total, s.total_interval)):
            low, high = interval["output"][n]
            assert low <= estimate["output"][n] <= high


def test_the_indices_of_the_global_process_are_reported_beside():
    """With a surrogate, the result also gives the indices of the global
    process (Marrel et al. 2009, Eq. 12), which add the error of the
    surrogate to every effect: close to the exact Ishigami indices for a
    good surrogate, and never below the estimate on the mean of the process
    for the total index of an input without effect."""
    inputs = {n: uq.Uniform(-math.pi, math.pi) for n in ("x1", "x2", "x3")}
    first, total, _ = _ishigami_exact()
    runs = uq.propagate(_ishigami, inputs, 200, report="none")
    s = uq.sobol(_ishigami, inputs, 1024, surrogate="gaussian_process", training=runs, report="none")
    for j, n in enumerate(("x1", "x2", "x3")):
        assert s.first_order_global_process["output"][n] == pytest.approx(first[j], abs=0.08)
        assert s.total_global_process["output"][n] == pytest.approx(total[j], abs=0.08)
        low, high = s.total_global_process_interval["output"][n]
        assert low <= s.total_global_process["output"][n] <= high
    table = s.tables["indices"]
    assert "total_global_process" in table.names and len(table) == 3
    assert "global process" in s.summary() and "total_global_process" in s.to_dict()
    plain = uq.sobol(_ishigami, inputs, 256, report="none")
    assert plain.first_order_global_process is None and "total_global_process" not in plain.tables["indices"].names


C_LOG = (0.8, 0.5, 0.3)


def _exponential_of_a_sum(x1, x2, x3):
    return math.exp(C_LOG[0] * x1 + C_LOG[1] * x2 + C_LOG[2] * x3)


def test_a_surrogate_of_the_logarithm_of_a_positive_output():
    """Y = exp(c . X) with standard normal X: S_i = (e^(c_i^2) - 1) /
    (e^(|c|^2) - 1) and T_i = 1 - (e^(|c|^2 - c_i^2) - 1) / (e^(|c|^2) - 1).
    The logarithm of Y is linear, which a process of the logarithm fits
    exactly from 40 runs."""
    names = ("x1", "x2", "x3")
    inputs = {n: uq.Normal(0.0, 1.0) for n in names}
    c2 = np.array(C_LOG) ** 2
    first = (np.exp(c2) - 1.0) / (np.exp(c2.sum()) - 1.0)
    total = 1.0 - (np.exp(c2.sum() - c2) - 1.0) / (np.exp(c2.sum()) - 1.0)
    runs = uq.propagate(_exponential_of_a_sum, inputs, 40, method="sobol", report="none")
    s = uq.sobol(_exponential_of_a_sum, inputs, 8192, surrogate="gaussian_process", training=runs, output_transform={"output": "log"}, report="none")
    # The output is log-normal, so the Monte Carlo error of the estimators is
    # large: 8192 rows hold it below 0.03. On the same rows the surrogate
    # gives the indices of the model itself.
    direct = uq.sobol(_exponential_of_a_sum, inputs, 8192, report="none")
    for j, n in enumerate(names):
        assert s.first_order["output"][n] == pytest.approx(first[j], abs=0.03)
        assert s.total["output"][n] == pytest.approx(total[j], abs=0.03)
        assert s.first_order["output"][n] == pytest.approx(direct.first_order["output"][n], abs=2e-3)
    with pytest.raises(ValueError, match="applies to a surrogate"):
        uq.sobol(_exponential_of_a_sum, inputs, 64, output_transform={"output": "log"}, report="none")
    with pytest.raises(ValueError, match="which the runs do not have"):
        uq.sobol(_exponential_of_a_sum, inputs, 64, surrogate="gaussian_process", training=runs, output_transform={"release": "log"}, report="none")
    with pytest.raises(ValueError, match="unknown output transform 'sqrt'"):
        uq.sobol(_exponential_of_a_sum, inputs, 64, surrogate="gaussian_process", training=runs, output_transform={"output": "sqrt"}, report="none")
    negative = uq.propagate(_ishigami, {n: uq.Uniform(-math.pi, math.pi) for n in names}, 16, report="none")
    with pytest.raises(ValueError, match="needs a positive output"):
        uq.sobol(_ishigami, {n: uq.Uniform(-math.pi, math.pi) for n in names}, 64, surrogate="gaussian_process", training=negative, output_transform={"output": "log"}, report="none")


def _scaled_exponential(theta):
    return 3.0 * math.exp(theta)


@pytest.mark.parametrize("surrogate", [None, "gaussian_process"])
def test_a_calibration_on_the_logarithmic_scale(surrogate):
    """y = 3 exp(theta) measured with a relative error of 5%: on the log
    scale log y = theta + log 3 is linear with Gaussian noise of standard
    deviation 0.05, so the posterior of theta under the prior N(1, 0.5) is
    normal with the conjugate mean and standard deviation."""
    observed, sigma = 3.0 * math.exp(1.3), 0.05
    prior_mean, prior_sd = 1.0, 0.5
    precision = 1.0 / prior_sd**2 + 1.0 / sigma**2
    mean = (prior_mean / prior_sd**2 + (math.log(observed) - math.log(3.0)) / sigma**2) / precision
    post = uq.calibrate(_scaled_exponential, {"theta": uq.Normal(prior_mean, prior_sd)}, {"output": observed}, {"output": sigma}, surrogate=surrogate, samples=8000, output_transform={"output": "log"}, report="none")
    assert post.mean("theta") == pytest.approx(mean, abs=0.01)
    assert post.standard_deviation("theta") == pytest.approx(precision**-0.5, rel=0.1)
    prediction = post.predict()["output"]
    assert prediction["low"] < observed < prediction["high"]
    with pytest.raises(ValueError, match="needs positive measurements"):
        uq.calibrate(_scaled_exponential, {"theta": uq.Normal(1.0, 0.5)}, {"output": -1.0}, {"output": 0.05}, surrogate=None, output_transform={"output": "log"}, report="none")


def _first_experiment(a, b1):
    return a + b1


def _second_experiment(a, b2):
    return 2.0 * a - b2


@pytest.mark.parametrize("surrogate", [None, "gaussian_process"])
def test_a_calibration_of_two_experiments_with_a_shared_parameter(surrogate):
    """Two experiments share the parameter a, and each has its own input:
    y1 = a + b1 and y2 = 2a - b2, with normal priors and noise. The model is
    linear, so the posterior is normal with the covariance (P0^-1 + H^T R^-1
    H)^-1 and the mean cov H^T R^-1 y."""
    inputs = {"a": uq.Normal(0.0, 1.0), "b1": uq.Normal(0.0, 0.5), "b2": uq.Normal(0.0, 0.5)}
    observed, noise = {"first": {"output": 1.0}, "second": {"output": 1.5}}, {"first": {"output": 0.1}, "second": {"output": 0.1}}
    H = np.array([[1.0, 1.0, 0.0], [2.0, 0.0, -1.0]])
    covariance = np.linalg.inv(np.diag([1.0, 4.0, 4.0]) + H.T @ H / 0.01)
    mean = covariance @ H.T @ np.array([1.0, 1.5]) / 0.01
    models = {"first": _first_experiment, "second": _second_experiment}
    training = None
    if surrogate is not None:
        training = {"first": uq.propagate(_first_experiment, {n: inputs[n] for n in ("a", "b1")}, 30, report="none"), "second": uq.propagate(_second_experiment, {n: inputs[n] for n in ("a", "b2")}, 30, report="none")}
    post = uq.calibrate(models, inputs, observed, noise, surrogate=surrogate, training=training, samples=12000, seed=2, report="none")
    for j, name in enumerate(("a", "b1", "b2")):
        assert post.mean(name) == pytest.approx(mean[j], abs=0.1 * math.sqrt(covariance[j, j]))
        assert post.standard_deviation(name) == pytest.approx(math.sqrt(covariance[j, j]), rel=0.1)
    prediction = post.predict()
    assert set(prediction) == {"first", "second"} and prediction["second"]["output"]["low"] < 1.5 < prediction["second"]["output"]["high"]
    with pytest.raises(ValueError, match="enter no experiment"):
        uq.calibrate(models, {**inputs, "c": uq.Normal(0.0, 1.0)}, observed, noise, surrogate=None, report="none")
    with pytest.raises(ValueError, match="no entry 'second'"):
        uq.calibrate(models, inputs, {"first": {"output": 1.0}}, noise, surrogate=None, report="none")
    with pytest.raises(ValueError, match="which have no prior"):
        uq.calibrate(models, {n: inputs[n] for n in ("a", "b1")}, observed, noise, surrogate=None, report="none")


# ---------------------------------------------------------------------------
# Gaussian process
# ---------------------------------------------------------------------------
def _branin(x):
    x1, x2 = 15 * x[:, 0] - 5, 15 * x[:, 1]
    return (x2 - 5.1 / (4 * np.pi**2) * x1**2 + 5 / np.pi * x1 - 6) ** 2 + 10 * (1 - 1 / (8 * np.pi)) * np.cos(x1) + 10


def test_gaussian_process_interpolates_and_predicts():
    rng = np.random.default_rng(0)
    x = rng.random((60, 2))
    y = _branin(x)
    gp = uq.GaussianProcess(nugget=0.0).fit(x, y)
    assert np.allclose(gp.predict(x), y, atol=1e-4 * np.ptp(y))
    xt = rng.random((400, 2))
    pred, sd = gp.predict(xt, return_standard_deviation=True)
    q2 = 1 - np.sum((pred - _branin(xt)) ** 2) / np.sum((_branin(xt) - _branin(xt).mean()) ** 2)
    assert q2 > 0.99
    z = (pred - _branin(xt)) / sd
    assert np.mean(np.abs(z) < 3) > 0.95
    assert gp.q2 > 0.97


def test_leave_one_out_closed_form_equals_refits():
    """The closed-form residuals (Dubrule 1983) against explicit refits with
    the same hyperparameters, for both trends."""
    rng = np.random.default_rng(1)
    x = rng.random((25, 2))
    y = np.sin(3 * x[:, 0]) + x[:, 1] ** 2
    for trend in ("constant", "linear"):
        p = _Process("matern52", trend, "fit", 3, np.random.default_rng(0)).fit(x, y)
        pred, var = p.leave_one_out()
        for i in (0, 7, 24):
            keep = np.arange(25) != i
            q = _Process("matern52", trend, "fit", 1, np.random.default_rng(0))
            q.x, q.y, q.F = x[keep], y[keep], q._basis(x[keep])
            q.fit_nugget = True
            q.log_ell, q.log_tau = p.log_ell, p.log_tau
            q._condition()
            m, _ = q.predict(x[i : i + 1])
            assert m[0] == pytest.approx(pred[i], rel=1e-6, abs=1e-9)


def test_gaussian_process_vector_output_keeps_the_discarded_variance():
    rng = np.random.default_rng(2)
    theta = rng.random((50, 2))
    t = np.linspace(0, 1, 40)
    y = theta[:, :1] * np.exp(-(1 + 3 * theta[:, 1:]) * t)
    gp = uq.GaussianProcess(variance_kept=0.99).fit(theta, y)
    assert 1 <= gp.components < 6
    pred, sd = gp.predict(theta[:5], return_standard_deviation=True)
    assert pred.shape == (5, 40) and np.all(sd > 0)
    _, cov = gp.predict(theta[:2], return_cov=True)
    assert cov.shape == (2, 40, 40)
    assert np.allclose(np.sqrt(np.diagonal(cov, axis1=1, axis2=2)), sd[:2])
    assert np.min(gp.q2) > 0.95


def test_gaussian_process_fit_to_runs_uses_log_space():
    inputs = {"d": uq.LogNormal(median=1.0, factor=10), "k": uq.Normal(1.0, 0.1)}

    def model(d, k):
        return math.log(d) + k

    runs = uq.propagate(model, inputs, 40)
    gp = uq.GaussianProcess().fit(runs)
    x = np.array([[0.3, 1.05], [4.0, 0.9]])
    assert np.allclose(gp.predict(x), np.log(x[:, 0]) + x[:, 1], atol=1e-3)


# ---------------------------------------------------------------------------
# calibration
# ---------------------------------------------------------------------------
T_OBS = np.linspace(0, 1, 8)
G_OBS = np.column_stack([np.ones_like(T_OBS), T_OBS, T_OBS**2])


def _quadratic(a, b, c):
    return {"y": G_OBS @ np.array([a, b, c])}


def _conjugate_case():
    rng = np.random.default_rng(1)
    mu0, s0, sig = np.array([1.0, 0.5, -0.2]), np.array([0.5, 0.4, 0.3]), 0.05
    y = G_OBS @ np.array([1.2, 0.2, 0.1]) + sig * rng.standard_normal(len(T_OBS))
    Sp = np.linalg.inv(np.diag(1 / s0**2) + G_OBS.T @ G_OBS / sig**2)
    mp = Sp @ (mu0 / s0**2 + G_OBS.T @ y / sig**2)
    inputs = {n: uq.Normal(m, s) for n, m, s in zip("abc", mu0, s0)}
    return inputs, y, sig, mp, np.sqrt(np.diag(Sp))


@pytest.mark.parametrize("surrogate,sampler", [(None, "ensemble"), (None, "metropolis"), ("gaussian_process", "ensemble")])
def test_calibration_recovers_the_conjugate_posterior(surrogate, sampler):
    """Linear model, normal prior and noise: the posterior is normal with
    the closed-form mean and covariance."""
    inputs, y, sig, mp, sp = _conjugate_case()
    post = uq.calibrate(_quadratic, inputs, {"y": y}, {"y": sig}, surrogate=surrogate, sampler=sampler, samples=12000, seed=4)
    for j, n in enumerate("abc"):
        assert post.mean(n) == pytest.approx(mp[j], abs=0.1 * sp[j] + 0.002)
        assert post.standard_deviation(n) == pytest.approx(sp[j], rel=0.1)
        assert post.r_hat[n] < 1.02
        assert post.effective_sample_size[n] > 300
    assert post.contraction["a"] > 0.9
    pred = post.predict()
    assert pred["y"]["coverage"] >= 0.75
    assert "contr." in post.summary()


def _only_a(a, b):
    return {"y": a * T_OBS}


def test_calibration_flags_parameters_the_data_do_not_inform():
    inputs = {"a": uq.Normal(1.0, 0.5), "b": uq.Normal(0.0, 1.0)}
    y = 1.3 * T_OBS
    post = uq.calibrate(_only_a, inputs, {"y": y}, {"y": 0.02}, surrogate=None, samples=8000)
    assert post.contraction["a"] > 0.95
    assert abs(post.contraction["b"]) < 0.1
    assert "not identified by the data" in post.summary()
    # Data far above the prior push the parameter to the upper end.
    far = uq.calibrate(_only_a, {"a": uq.Normal(1.0, 0.1, upper=1.3), "b": inputs["b"]}, {"y": 3.0 * T_OBS}, {"y": 0.02}, surrogate=None, samples=4000)
    assert far.at_bound["a"] == "upper"


def _slope(theta):
    return {"y": theta * T_OBS}


def test_discrepancy_widens_the_posterior_and_does_not_hold_it_at_the_prior():
    """Measurements y = t + 0.3 t^2 of a model y = theta t. Without a
    discrepancy theta = 1.24 with an interval as narrow as the noise allows.
    With the discrepancy integrated out (Kennedy and O'Hagan 2001) the
    interval is many times wider, the posterior does not depend on the prior
    mean, and the predictive interval covers every measurement. A
    discrepancy fitted at the prior mean and subtracted from the data holds
    the posterior at the prior mean whatever the data (Wu et al. 2018,
    Sect. 5): with prior means 0.7, 1.0 and 1.4 it gave 0.70, 1.00 and 1.40,
    each with a standard deviation of 0.009."""
    y = T_OBS + 0.3 * T_OBS**2
    plain = uq.calibrate(_slope, {"theta": uq.Normal(1.0, 0.5)}, {"y": y}, {"y": 0.01}, surrogate=None, samples=6000)
    assert plain.mean("theta") == pytest.approx(1.24, abs=0.01)
    means = []
    for prior_mean in (0.7, 1.0, 1.4):
        post = uq.calibrate(_slope, {"theta": uq.Normal(prior_mean, 0.5)}, {"y": y}, {"y": 0.01}, surrogate=None, samples=6000, discrepancy="gaussian_process", locations={"y": T_OBS})
        means.append(post.mean("theta"))
        assert post.standard_deviation("theta") > 5 * plain.standard_deviation("theta")
        assert post.predict()["y"]["coverage"] == 1.0
    assert max(means) - min(means) < 0.03
    assert set(post.discrepancy["y"]) == {"variance", "length_scale"}


def test_calibration_input_errors():
    inputs = {"a": uq.Normal(1.0, 0.5), "b": uq.Normal(0.0, 1.0)}
    with pytest.raises(ValueError, match="same names"):
        uq.calibrate(_only_a, inputs, {"y": T_OBS}, {"z": 0.1}, surrogate=None)
    with pytest.raises(ValueError, match="values per run"):
        uq.calibrate(_only_a, inputs, {"y": T_OBS[:3]}, {"y": 0.1}, surrogate=None, samples=100)
    with pytest.raises(TypeError):
        uq.propagate(_only_a, {"a": 1.0}, 10)


# ---------------------------------------------------------------------------
# the bus bar example and the JSON summaries
# ---------------------------------------------------------------------------
EXAMPLES = __import__("pathlib").Path(__file__).resolve().parents[2] / "examples"


def _bus_bar_example():
    import importlib.util

    spec = importlib.util.spec_from_file_location("bus_bar_uq", EXAMPLES / "bus_bar_uq.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bus_bar_propagation_is_reproducible_and_written_to_json(tmp_path):
    """The study of examples/bus_bar_uq.py gives the same runs from the same
    seed, and its summary is written to JSON."""
    import json

    module = _bus_bar_example()
    inputs = {k: module.inputs[k] for k in ("conductivity", "heat_source")}
    runs = uq.propagate(module.bus_bar, inputs, 16, seed=1)
    again = uq.propagate(module.bus_bar, inputs, 16, seed=1)
    assert np.allclose(runs.outputs["hottest"], again.outputs["hottest"], rtol=1e-12)
    runs.write_json(tmp_path / "out.json")
    numbers = json.loads((tmp_path / "out.json").read_text())
    assert numbers["study"] == "propagate" and numbers["runs"] == 16 and numbers["failed_runs"] == 0
    assert numbers["outputs"]["hottest"]["mean"] == pytest.approx(runs.mean("hottest"))
    assert set(numbers["outputs"]["hottest"]["spearman"]) == {"conductivity", "heat_source"}


def test_bus_bar_sobol_and_calibration_summaries(tmp_path):
    import json

    module = _bus_bar_example()
    inputs = {k: module.inputs[k] for k in ("conductivity", "heat_source")}
    s = uq.sobol(module.bus_bar, inputs, 64)
    assert s.runs == 64 * 4
    assert s.total["hottest"]["conductivity"] > 0.2
    s.write_json(tmp_path / "sobol.json")
    assert json.loads((tmp_path / "sobol.json").read_text())["total"]["hottest"]["conductivity"] == pytest.approx(s.total["hottest"]["conductivity"])
    # Calibrate the conductivity on a measurement made with k = 21.
    measured = module.bus_bar(conductivity=21.0)["hottest"]
    post = uq.calibrate(module.bus_bar, {"conductivity": inputs["conductivity"]}, {"hottest": measured}, {"hottest": 0.2}, training_samples=12, samples=4000)
    assert post.mean("conductivity") == pytest.approx(21.0, abs=0.15)
    post.write_json(tmp_path / "posterior.json")
    numbers = json.loads((tmp_path / "posterior.json").read_text())
    assert numbers["parameters"]["conductivity"]["mean"] == pytest.approx(post.mean("conductivity"))


def test_jsonable_writes_non_finite_values_as_null():
    from dualmesh.parameters import jsonable

    assert jsonable({"a": np.array([1.0, np.nan]), "b": (np.int64(2), math.inf)}) == {"a": [1.0, None], "b": [2, None]}


def test_split_rhat_is_rank_normalized_and_folded():
    """Vehtari et al. (Bayesian Analysis 2021), Sects. 4.1 and 4.2: the
    split R-hat of rank-normalized draws, and the maximum with that of the
    folded draws |theta - median|, which detects chains with the same
    location but different scales; the classic split R-hat does not."""
    from dualmesh.uq.calibrate import _classic_split_rhat, _normal_scores, _split_rhat

    rng = np.random.default_rng(1)
    mixed = rng.standard_normal((2000, 4, 1))
    assert _split_rhat(mixed) < 1.01
    scales = mixed * np.array([1.0, 1.0, 1.0, 3.0])[None, :, None]
    assert _split_rhat(scales) > 1.05
    # The classic split R-hat of the same draws misses the different scales.
    halves = np.column_stack([scales[:1000, c, 0] for c in range(4)] + [scales[1000:, c, 0] for c in range(4)])
    assert _classic_split_rhat(halves) < 1.01
    # The rank-normalized (bulk) part is invariant to a monotone transformation.
    assert _classic_split_rhat(_normal_scores(np.exp(halves))) == pytest.approx(_classic_split_rhat(_normal_scores(halves)), rel=1e-12)
    # Defined for heavy tails (Cauchy draws in stationary chains).
    assert _split_rhat(rng.standard_cauchy((2000, 4, 1))) < 1.01
