# SPDX-License-Identifier: LGPL-2.1-or-later
"""Designs of the unit cube, and the evaluation of a model at many inputs,
in parallel and with a store file that keeps every run."""

from __future__ import annotations

import concurrent.futures as cf
import multiprocessing as mp
import os
import tempfile
import time
import traceback
import warnings
from pathlib import Path

import numpy as np

from ..parameters import write_json_numbers
from .distributions import Distribution

#: Write the dict of a result to a JSON file.
write_json = write_json_numbers

METHODS = ("latin_hypercube", "sobol", "monte_carlo")


def check_inputs(inputs) -> dict:
    """The inputs as an ordered dict of name -> Distribution."""
    if not isinstance(inputs, dict) or not inputs:
        raise ValueError("uq: inputs must be a non-empty dict of name -> distribution.")
    for name, dist in inputs.items():
        if not isinstance(name, str) or not name.isidentifier():
            raise ValueError(f"uq: input name {name!r} must be a valid keyword argument name.")
        if not isinstance(dist, Distribution):
            raise TypeError(f"uq: input '{name}' must be a distribution (uq.Normal, uq.LogNormal, uq.Uniform or uq.LogUniform), not {type(dist).__name__}.")
    return dict(inputs)


def unit_design(n: int, d: int, method: str, seed) -> np.ndarray:
    """``n`` points of the unit cube in ``d`` dimensions, strictly inside it."""
    if method not in METHODS:
        raise ValueError(f"uq: unknown method '{method}'. Use one of {', '.join(METHODS)}.")
    if n < 1:
        raise ValueError("uq: the number of samples must be at least 1.")
    rng = np.random.default_rng(seed)
    if method == "monte_carlo":
        u = rng.random((n, d))
    elif method == "latin_hypercube":
        # McKay, Beckman and Conover (1979): one point in each of the n
        # equal-probability strata of every input, strata paired at random.
        u = np.empty((n, d))
        for j in range(d):
            u[:, j] = (rng.permutation(n) + rng.random(n)) / n
    else:
        from scipy.stats import qmc

        with warnings.catch_warnings():
            # The balance of the sequence is best for n a power of 2; other n
            # are allowed and only converge a little slower.
            warnings.simplefilter("ignore", UserWarning)
            u = qmc.Sobol(d, scramble=True, seed=rng).random(n)
    # Keep away from 0 and 1, where an unbounded inverse CDF is infinite.
    return np.clip(u, 1e-12, 1.0 - 1e-12)


def to_inputs(u: np.ndarray, inputs: dict) -> np.ndarray:
    """Map a design of the unit cube to input values, column by column."""
    x = np.empty_like(u)
    for j, dist in enumerate(inputs.values()):
        x[:, j] = dist.ppf(u[:, j])
    return x


def to_unit(x: np.ndarray, inputs: dict) -> np.ndarray:
    """Map input values back to the unit cube (the CDF of each input)."""
    u = np.empty_like(np.asarray(x, dtype=float))
    for j, dist in enumerate(inputs.values()):
        u[:, j] = dist.cdf(x[:, j])
    return u


def normalise_output(value) -> dict:
    """A model's return value as a dict of name -> float array."""
    if isinstance(value, dict):
        if not value:
            raise ValueError("uq: the model returned an empty dict.")
        out = {}
        for k, v in value.items():
            if not isinstance(k, str):
                raise TypeError("uq: the output names of the model must be strings.")
            out[k] = np.asarray(v, dtype=float)
        return out
    return {"output": np.asarray(value, dtype=float)}


def _call(model, kwargs):
    from ..console import inner_study

    t0 = time.perf_counter()
    try:
        # A study of the model (a solve, a transient) inside the run prints
        # nothing.
        with inner_study():
            out = normalise_output(model(**kwargs))
        return out, None, time.perf_counter() - t0
    except Exception:  # noqa: BLE001 - a failed run is recorded, not fatal
        return None, traceback.format_exc(limit=3), time.perf_counter() - t0


def _worker_init(threads):
    # Divide the cores among the processes; the compiled library reads this
    # when it starts its OpenMP threads.
    os.environ["OMP_NUM_THREADS"] = str(threads)


def _key(row) -> tuple:
    return tuple(float(v).hex() for v in row)


#: Relative tolerance with which a store matches the input values of a run.
#: Other versions of NumPy and SciPy can calculate the inverse distribution
#: functions with a different last digit, and the store must find the runs
#: of these versions.
STORE_TOLERANCE = 1e-12


class Store:
    """The runs of a model kept in a ``.npz`` file: the input names, the
    input values, every output and the failures.  Rows are matched by their
    input values to the relative tolerance ``STORE_TOLERANCE``, so a study
    that is interrupted resumes, and runs are shared between studies with
    the same inputs."""

    def __init__(self, path, names):
        self.path = None if path is None else Path(path)
        self.names = list(names)
        self.rows: dict[tuple, int] = {}
        self.x: list[np.ndarray] = []
        self.outputs: list[dict | None] = []
        self.errors: list[str | None] = []
        self.seconds: list[float] = []
        if self.path is not None and self.path.exists():
            self._load()

    def _load(self):
        with np.load(self.path, allow_pickle=False) as f:
            names = [str(s) for s in f["input_names"]]
            if names != self.names:
                raise ValueError(f"uq: the store {self.path} holds the inputs {names}, not {self.names}. Use another file.")
            x = f["x"]
            ok = f["ok"]
            errors = [str(s) for s in f["errors"]]
            seconds = f["seconds"]
            out_names = [str(s) for s in f["output_names"]]
            arrays = {k: f["output:" + k] for k in out_names}
        for i in range(len(x)):
            out = {k: arrays[k][i] for k in out_names} if ok[i] else None
            self._append(x[i], out, errors[i] or None, float(seconds[i]))

    def _append(self, row, out, error, seconds):
        self.rows[_key(row)] = len(self.x)
        self.x.append(np.asarray(row, dtype=float))
        self.outputs.append(out)
        self.errors.append(error)
        self.seconds.append(seconds)

    def find(self, row):
        index = self.rows.get(_key(row))
        if index is not None or not self.x:
            return index
        row = np.asarray(row, dtype=float)
        stored = np.asarray(self.x)
        if stored.shape[1] != row.size:
            return None
        close = np.all(np.abs(stored - row) <= STORE_TOLERANCE * np.maximum(np.abs(stored), np.abs(row)), axis=1)
        matches = np.flatnonzero(close)
        return int(matches[0]) if matches.size else None

    def output_shapes(self) -> dict:
        for out in self.outputs:
            if out is not None:
                return {k: v.shape for k, v in out.items()}
        return {}

    def save(self):
        if self.path is None:
            return
        shapes = self.output_shapes()
        arrays = {}
        for k, shape in shapes.items():
            a = np.full((len(self.x),) + shape, np.nan)
            for i, out in enumerate(self.outputs):
                if out is not None:
                    a[i] = out[k]
            arrays["output:" + k] = a
        payload = dict(input_names=np.array(self.names), x=np.array(self.x).reshape(len(self.x), len(self.names)), ok=np.array([o is not None for o in self.outputs], dtype=bool), errors=np.array([e or "" for e in self.errors]), seconds=np.array(self.seconds), output_names=np.array(list(shapes)), **arrays)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, suffix=".npz")
        os.close(fd)
        np.savez(tmp, **payload)
        os.replace(tmp, self.path)


#: The transforms of an output that a surrogate models.
OUTPUT_TRANSFORMS = ("log",)


def check_output_transform(transform, runs, where: str) -> dict:
    """The output transform of a surrogate study, checked against the
    training runs: output name -> "log"."""
    if not transform:
        return {}
    if not isinstance(transform, dict):
        raise ValueError(f"{where}: output_transform is a dict of output name -> 'log'.")
    for name, kind in transform.items():
        if name not in runs.outputs:
            raise ValueError(f"{where}: output_transform names the output '{name}', which the runs do not have. Outputs: {', '.join(runs.outputs)}.")
        if kind not in OUTPUT_TRANSFORMS:
            raise ValueError(f"{where}: unknown output transform '{kind}' of '{name}'. The transforms are: {', '.join(OUTPUT_TRANSFORMS)}.")
        values = np.asarray(runs.outputs[name], dtype=float)
        values = values[np.isfinite(values)]
        if values.size and values.min() <= 0.0:
            raise ValueError(f"{where}: the log transform needs a positive output, and '{name}' has the value {values.min():.6g}.")
    return dict(transform)


def transformed_runs(runs, transform: dict):
    """The runs with the transformed outputs, for the fit of a surrogate."""
    if not transform:
        return runs
    from .propagate import Runs

    outputs = {name: (np.log(values) if transform.get(name) == "log" else values) for name, values in runs.outputs.items()}
    return Runs(runs.distributions, runs.x, runs.u, outputs, runs.failed, runs.method)


def _as_dict(gp, value) -> dict:
    return value if isinstance(value, dict) else {gp.output_names[0]: value}


def surrogate_mean(gp, x, transform: dict) -> dict:
    """The mean of the surrogate process at the inputs ``x``, for each output
    on its own scale.  For a log transform the mean of the output is the
    mean of the log-normal variable, exp(mu + sigma^2 / 2)."""
    if not transform:
        return _as_dict(gp, gp.predict(x))
    mean, sd = gp.predict(x, return_standard_deviation=True)
    mean, sd = _as_dict(gp, mean), _as_dict(gp, sd)
    return {name: (np.exp(mean[name] + 0.5 * sd[name] ** 2) if transform.get(name) == "log" else mean[name]) for name in mean}


def surrogate_sample(gp, x, size: int, seed, transform: dict) -> dict:
    """Random functions of the surrogate process at the inputs ``x``, for
    each output on its own scale: (size, n) or (size, n, m) arrays."""
    sample = _as_dict(gp, gp.sample(x, size, seed))
    return {name: (np.exp(values) if transform.get(name) == "log" else values) for name, values in sample.items()}


def study_header(level: str, study: str, inputs: dict, runs: int | None = None) -> None:
    """Print the start of the report of a study of many runs."""
    from ..console import header, table

    if level == "full":
        print(header(study))
        print("Uncertain inputs")
        print(table([[name, repr(dist)] for name, dist in inputs.items()], ["input", "distribution"]))
    if level in ("full", "summary"):
        count = "" if runs is None else f", {runs} runs of the model"
        print(f"dualmesh {study}: {len(inputs)} uncertain inputs{count}", flush=True)


def evaluate(model, names, x, processes=1, store=None, progress=False):
    """Evaluate ``model`` at every row of ``x`` (columns in the order of
    ``names``).  Returns ``(outputs, failed)``: a dict of name -> array with
    one row per input row (NaN for failed runs), and a dict of row index ->
    error message."""
    x = np.atleast_2d(np.asarray(x, dtype=float))
    st = store if isinstance(store, Store) else Store(store, names)
    todo = [i for i in range(len(x)) if st.find(x[i]) is None]
    # Rows repeated within this request are evaluated once.
    unique, seen = [], set()
    for i in todo:
        k = _key(x[i])
        if k not in seen:
            seen.add(k)
            unique.append(i)
    kwargs = [dict(zip(names, map(float, x[i]))) for i in unique]
    n_total, n_done = len(unique), 0
    last_save = time.perf_counter()

    def record(i, result):
        nonlocal n_done, last_save
        out, err, sec = result
        st._append(x[i], out, err, sec)
        n_done += 1
        if progress:
            status = "failed" if out is None else f"{sec:.1f} s"
            print(f"uq: run {n_done}/{n_total} ({status})", flush=True)
        if time.perf_counter() - last_save > 30.0:
            st.save()
            last_save = time.perf_counter()

    try:
        if processes is None or processes <= 1 or n_total <= 1:
            for i, kw in zip(unique, kwargs):
                record(i, _call(model, kw))
        else:
            processes = int(processes)
            threads = max(1, (os.cpu_count() or 1) // processes)
            ctx = mp.get_context("spawn")
            with cf.ProcessPoolExecutor(processes, mp_context=ctx, initializer=_worker_init, initargs=(threads,)) as pool:
                futures = {pool.submit(_call, model, kw): i for i, kw in zip(unique, kwargs)}
                for fut in cf.as_completed(futures):
                    record(futures[fut], fut.result())
    finally:
        if n_done:
            st.save()

    shapes = st.output_shapes()
    if not shapes:
        first_error = next((e for e in st.errors if e), "")
        raise RuntimeError(f"uq: every run of the model failed. The first error:\n{first_error}")
    outputs = {k: np.full((len(x),) + s, np.nan) for k, s in shapes.items()}
    failed = {}
    for i in range(len(x)):
        j = st.find(x[i])
        out = st.outputs[j]
        if out is None:
            failed[i] = st.errors[j]
            continue
        for k in shapes:
            if k not in out or out[k].shape != shapes[k]:
                raise ValueError(f"uq: the model returned output '{k}' with different shapes in different runs; every run must return the same names and shapes.")
            outputs[k][i] = out[k]
    return outputs, failed
