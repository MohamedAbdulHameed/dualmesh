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
@pytest.mark.parametrize(
    "dist",
    [
        uq.Normal(1.0, 0.3),
        uq.Normal(1.0, 0.3, lower=0.2),
        uq.Normal(1.0, 0.3, lower=0.9, upper=1.6),
        uq.LogNormal(median=1.0, factor=10),
        uq.LogNormal(median=2.0, sigma=0.4, lower=1.0, upper=5.0),
        uq.Uniform(-1.0, 3.0),
        uq.LogUniform(0.1, 10.0),
    ],
)
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
    assert dist.std == pytest.approx(math.sqrt(m2 - m1 * m1), rel=1e-5)
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
    assert d.std == pytest.approx(ref.std(), rel=1e-7)


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
    assert runs.std("y") ** 2 == pytest.approx(0.79, rel=0.02)
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
    (Ishigami and Homma 1990; Saltelli et al. 2008), with the run count
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


# ---------------------------------------------------------------------------
# Gaussian process
# ---------------------------------------------------------------------------
def _branin(x):
    x1, x2 = 15 * x[:, 0] - 5, 15 * x[:, 1]
    return (
        (x2 - 5.1 / (4 * np.pi**2) * x1**2 + 5 / np.pi * x1 - 6) ** 2
        + 10 * (1 - 1 / (8 * np.pi)) * np.cos(x1)
        + 10
    )


def test_gaussian_process_interpolates_and_predicts():
    rng = np.random.default_rng(0)
    x = rng.random((60, 2))
    y = _branin(x)
    gp = uq.GaussianProcess(nugget=0.0).fit(x, y)
    assert np.allclose(gp.predict(x), y, atol=1e-4 * np.ptp(y))
    xt = rng.random((400, 2))
    pred, sd = gp.predict(xt, return_std=True)
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
    pred, sd = gp.predict(theta[:5], return_std=True)
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


@pytest.mark.parametrize(
    "surrogate,sampler",
    [(None, "ensemble"), (None, "metropolis"), ("gaussian_process", "ensemble")],
)
def test_calibration_recovers_the_conjugate_posterior(surrogate, sampler):
    """Linear model, normal prior and noise: the posterior is normal with
    the closed-form mean and covariance."""
    inputs, y, sig, mp, sp = _conjugate_case()
    post = uq.calibrate(
        _quadratic,
        inputs,
        {"y": y},
        {"y": sig},
        surrogate=surrogate,
        sampler=sampler,
        samples=12000,
        seed=4,
    )
    for j, n in enumerate("abc"):
        assert post.mean(n) == pytest.approx(mp[j], abs=0.1 * sp[j] + 0.002)
        assert post.std(n) == pytest.approx(sp[j], rel=0.1)
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
    far = uq.calibrate(
        _only_a,
        {"a": uq.Normal(1.0, 0.1, upper=1.3), "b": inputs["b"]},
        {"y": 3.0 * T_OBS},
        {"y": 0.02},
        surrogate=None,
        samples=4000,
    )
    assert far.at_bound["a"] == "upper"


def _slope(theta):
    return {"y": theta * T_OBS}


def test_discrepancy_removes_the_bias_of_a_missing_term():
    """Measurements y = t + 0.3 t^2 of a model y = theta t whose nominal
    theta = 1 is right: without a discrepancy term the missing t^2 biases
    theta by about 0.23; the modular discrepancy term removes the bias."""
    y = T_OBS + 0.3 * T_OBS**2
    inputs = {"theta": uq.Normal(1.0, 0.5)}
    plain = uq.calibrate(_slope, inputs, {"y": y}, {"y": 0.01}, surrogate=None, samples=6000)
    fixed = uq.calibrate(
        _slope,
        inputs,
        {"y": y},
        {"y": 0.01},
        surrogate=None,
        samples=6000,
        discrepancy="gaussian_process",
        locations={"y": T_OBS},
    )
    assert plain.mean("theta") - 1.0 > 0.15
    assert abs(fixed.mean("theta") - 1.0) < 0.05
    assert fixed.discrepancy["y"][0].shape == (8,)


def test_calibration_input_errors():
    inputs = {"a": uq.Normal(1.0, 0.5), "b": uq.Normal(0.0, 1.0)}
    with pytest.raises(ValueError, match="same names"):
        uq.calibrate(_only_a, inputs, {"y": T_OBS}, {"z": 0.1}, surrogate=None)
    with pytest.raises(ValueError, match="values per run"):
        uq.calibrate(_only_a, inputs, {"y": T_OBS[:3]}, {"y": 0.1}, surrogate=None, samples=100)
    with pytest.raises(TypeError):
        uq.propagate(_only_a, {"a": 1.0}, 10)


# ---------------------------------------------------------------------------
# YAML input
# ---------------------------------------------------------------------------
EXAMPLES = __import__("pathlib").Path(__file__).resolve().parents[2] / "examples"


def _yaml_study(tmp_path, block, inline=False):
    yaml = pytest.importorskip("yaml")
    from dualmesh.uq._input import run_study

    if inline:
        with open(EXAMPLES / "bus_bar.yaml") as stream:
            document = yaml.safe_load(stream)
        document["uq"] = block
    else:
        (tmp_path / "bus_bar.yaml").write_text((EXAMPLES / "bus_bar.yaml").read_text())
        document = {"uq": {"model": "bus_bar.yaml", **block}}
    return run_study(document, tmp_path, verbose=False)


INPUTS_YAML = {
    "conductivity": {
        "distribution": "normal",
        "mean": 20.0,
        "std": 1.0,
        "parameter": "kernels/conduction/thermal_conductivity",
    },
    "heat_source": {
        "distribution": "normal",
        "mean": 1.0,
        "std": 0.05,
        "parameter": "kernels/heating/heat_source",
        "apply": "factor",
    },
}


def test_yaml_propagation_equals_python(tmp_path):
    """The YAML study of examples/bus_bar_uq.yaml gives the same runs as
    the Python model of examples/bus_bar_uq.py."""
    import importlib.util

    block = dict(
        study="propagate",
        inputs=INPUTS_YAML,
        outputs=["hottest"],
        samples=16,
        seed=1,
        results="out.json",
    )
    runs = _yaml_study(tmp_path, block)
    spec = importlib.util.spec_from_file_location("bus_bar_uq", EXAMPLES / "bus_bar_uq.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    inputs = {k: module.inputs[k] for k in ("conductivity", "heat_source")}
    python = uq.propagate(module.bus_bar, inputs, 16, seed=1)
    assert np.allclose(runs.outputs["hottest"], python.outputs["hottest"], rtol=1e-12)
    assert (tmp_path / "out.json").exists()
    inline = _yaml_study(tmp_path, dict(block, results=None), inline=True)
    assert np.allclose(inline.outputs["hottest"], runs.outputs["hottest"], rtol=1e-12)


def test_yaml_sobol_and_calibration(tmp_path):
    s = _yaml_study(
        tmp_path, dict(study="sobol", inputs=INPUTS_YAML, outputs=["hottest"], samples=64)
    )
    assert s.runs == 64 * 4
    assert s.total["hottest"]["conductivity"] > 0.2
    # Calibrate the conductivity on a measurement made with k = 21.
    nominal = _yaml_study(
        tmp_path,
        dict(
            study="propagate",
            outputs=["hottest"],
            samples=1,
            inputs={
                "conductivity": {
                    "distribution": "uniform",
                    "lower": 20.999999,
                    "upper": 21.0,
                    "parameter": "kernels/conduction/thermal_conductivity",
                }
            },
        ),
    )
    measured = float(nominal.outputs["hottest"][0])
    post = _yaml_study(
        tmp_path,
        dict(
            study="calibrate",
            inputs={"conductivity": INPUTS_YAML["conductivity"]},
            outputs=["hottest"],
            observed={"hottest": measured},
            noise={"hottest": 0.2},
            training_samples=12,
            samples=4000,
        ),
    )
    assert post.mean("conductivity") == pytest.approx(21.0, abs=0.15)


def test_yaml_errors(tmp_path):
    with pytest.raises(ValueError, match="Did you mean 'samples'"):
        _yaml_study(
            tmp_path, dict(study="propagate", inputs=INPUTS_YAML, outputs=["hottest"], sample=4)
        )
    bad = {"k": dict(INPUTS_YAML["conductivity"], parameter="kernels/conductor/k")}
    with pytest.raises(ValueError, match="no 'kernels/conductor'"):
        _yaml_study(tmp_path, dict(study="propagate", inputs=bad, outputs=["hottest"], samples=2))
    with pytest.raises(RuntimeError, match="no post-processor 'coldest'"):
        _yaml_study(
            tmp_path, dict(study="propagate", inputs=INPUTS_YAML, outputs=["coldest"], samples=2)
        )
