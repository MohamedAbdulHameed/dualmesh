# SPDX-License-Identifier: LGPL-2.1-or-later
"""The YAML form of an uncertainty study: a ``uq`` block that names the
uncertain inputs of a dualmesh input file, the outputs to follow, and the
study (propagate, sobol or calibrate).  It mirrors the Python functions one
to one; see the documentation page on input files."""

from __future__ import annotations

import contextlib
import copy
import io
import json
import math
from pathlib import Path

import numpy as np

from .distributions import LogNormal, LogUniform, Normal, Uniform

STUDIES = ("propagate", "sobol", "calibrate")
COMMON = (
    "study",
    "model",
    "inputs",
    "outputs",
    "seed",
    "processes",
    "store",
    "results",
    "progress",
)
STUDY_KEYS = {
    "propagate": COMMON + ("samples", "method"),
    "sobol": COMMON
    + ("samples", "surrogate", "second_order", "training_samples", "realizations", "level"),
    "calibrate": COMMON
    + (
        "observed",
        "noise",
        "surrogate",
        "training_samples",
        "discrepancy",
        "locations",
        "sampler",
        "samples",
        "walkers",
        "chains",
        "level",
    ),
}
DISTRIBUTIONS = {
    "normal": (Normal, ("mean", "std", "lower", "upper")),
    "log_normal": (LogNormal, ("median", "factor", "sigma", "lower", "upper")),
    "uniform": (Uniform, ("lower", "upper")),
    "log_uniform": (LogUniform, ("lower", "upper")),
}
INPUT_KEYS = ("distribution", "parameter", "apply")


def _check_keys(where, given, allowed):
    from ..cli import _check_keys as check

    check(where, given, allowed)


def _number(value, where):
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError(f"uq: {where} must be a number, not {value!r}.") from None


def _distribution(name, spec):
    if not isinstance(spec, dict) or "distribution" not in spec or "parameter" not in spec:
        raise ValueError(
            f"uq: input '{name}' needs 'distribution' (normal, log_normal, uniform or "
            "log_uniform) and 'parameter' (the path of the value in the model input)."
        )
    kind = str(spec["distribution"])
    if kind not in DISTRIBUTIONS:
        raise ValueError(
            f"uq: input '{name}': unknown distribution '{kind}'. Use {', '.join(DISTRIBUTIONS)}."
        )
    cls, keys = DISTRIBUTIONS[kind]
    _check_keys(f"key of input '{name}'", spec, INPUT_KEYS + keys)
    args = {k: _number(spec[k], f"'{k}' of input '{name}'") for k in keys if k in spec}
    return cls(**args)


def _split(path):
    parts = [p for p in str(path).split("/") if p]
    if not parts:
        raise ValueError("uq: an empty parameter path.")
    return [int(p) if p.lstrip("-").isdigit() else p for p in parts]


def _locate(document, path):
    """The container and the key of a parameter path such as
    ``kernels/conduction/thermal_conductivity``."""
    node = document
    parts = _split(path)
    for i, p in enumerate(parts[:-1]):
        try:
            node = node[p]
        except (KeyError, IndexError, TypeError):
            where = "/".join(str(q) for q in parts[: i + 1])
            raise ValueError(f"uq: the model input has no '{where}' (in '{path}').") from None
    return node, parts[-1]


class DocumentModel:
    """A dualmesh input file as a model: each input sets (or multiplies) one
    value of the document, and the outputs are post-processors."""

    def __init__(self, document, parameters, apply, outputs):
        self.document = document
        self.parameters = parameters
        self.apply = apply
        self.outputs = outputs
        for name, path in parameters.items():
            node, key = _locate(document, path)
            if apply[name] == "factor":
                try:
                    float(node[key])
                except (KeyError, IndexError, TypeError, ValueError):
                    raise ValueError(
                        f"uq: input '{name}' multiplies '{path}', which must hold a number."
                    ) from None

    def __call__(self, **values):
        from ..cli import run

        doc = copy.deepcopy(self.document)
        doc.pop("outputs", None)  # no files from the individual runs
        for name, value in values.items():
            node, key = _locate(doc, self.parameters[name])
            node[key] = float(node[key]) * value if self.apply[name] == "factor" else value
        with contextlib.redirect_stdout(io.StringIO()):
            solver = run(doc, verbose=False)
        problem = getattr(solver, "local", solver)
        history = problem.postprocessor_values()
        out = {}
        for name, kind in self.outputs.items():
            if name not in history:
                raise ValueError(
                    f"uq: the model has no post-processor '{name}'. Post-processors: "
                    f"{', '.join(k for k in history if k != 'time')}."
                )
            series = np.asarray(history[name], dtype=float)
            out[name] = series if kind == "history" else float(series[-1])
        return out


def _outputs(spec):
    if isinstance(spec, (list, tuple)):
        return {str(n): "final" for n in spec}
    if isinstance(spec, dict):
        out = {}
        for n, kind in spec.items():
            if kind not in ("final", "history"):
                raise ValueError(f"uq: output '{n}' must be final or history, not {kind!r}.")
            out[str(n)] = kind
        return out
    raise ValueError("uq: outputs must be a list of post-processor names or a dict.")


def _array_dict(spec, where):
    if not isinstance(spec, dict):
        raise ValueError(f"uq: {where} must map output names to values.")
    return {str(k): np.asarray(v, dtype=float) for k, v in spec.items()}


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.floating, float)):
        v = float(value)
        return v if math.isfinite(v) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    return value


def run_study(document, base_dir=".", verbose=True):
    """Run the study of a document with a ``uq`` block; return the result
    (:class:`Runs`, :class:`SobolIndices` or :class:`Posterior`)."""
    from . import calibrate, propagate, sobol

    spec = dict(document["uq"])
    study = str(spec.get("study", "propagate"))
    if study not in STUDIES:
        raise ValueError(f"uq: unknown study '{study}'. Use {', '.join(STUDIES)}.")
    _check_keys(f"key of the uq block ({study})", spec, STUDY_KEYS[study])
    base = Path(base_dir)
    if "model" in spec:
        import yaml

        others = [k for k in document if k != "uq"]
        if others:
            raise ValueError(
                "uq: give the model either as 'model: file.yaml' or as the other blocks of this "
                f"file, not both (found {', '.join(others)})."
            )
        with open(base / str(spec["model"])) as stream:
            model_doc = yaml.safe_load(stream)
    else:
        model_doc = {k: v for k, v in document.items() if k != "uq"}
    if not isinstance(spec.get("inputs"), dict) or not spec["inputs"]:
        raise ValueError("uq: the uq block needs 'inputs'.")
    dists, paths, apply = {}, {}, {}
    for name, s in spec["inputs"].items():
        dists[name] = _distribution(name, s)
        paths[name] = s["parameter"]
        apply[name] = str(s.get("apply", "value"))
        if apply[name] not in ("value", "factor"):
            raise ValueError(f"uq: input '{name}': apply must be value or factor.")
    if "outputs" not in spec:
        raise ValueError("uq: the uq block needs 'outputs' (post-processor names).")
    model = DocumentModel(model_doc, paths, apply, _outputs(spec["outputs"]))
    common = dict(
        seed=int(spec.get("seed", 0)),
        processes=int(spec.get("processes", 1)),
        store=(base / spec["store"]) if spec.get("store") else None,
        progress=bool(spec.get("progress", verbose)),
    )
    if study == "propagate":
        if "samples" not in spec:
            raise ValueError("uq: propagate needs 'samples'.")
        result = propagate(
            model,
            dists,
            int(spec["samples"]),
            method=str(spec.get("method", "latin_hypercube")),
            **common,
        )
        numbers = {
            name: dict(
                mean=result.mean(name),
                std=result.std(name),
                interval_95=result.interval(name, 0.95),
                spearman=result.sensitivity(name),
            )
            for name in result.outputs
        }
        numbers["failed_runs"] = len(result.failed)
    elif study == "sobol":
        if "samples" not in spec:
            raise ValueError("uq: sobol needs 'samples'.")
        extra = {
            k: spec[k]
            for k in ("surrogate", "training_samples", "realizations", "level")
            if k in spec
        }
        result = sobol(
            model,
            dists,
            int(spec["samples"]),
            second_order=bool(spec.get("second_order", False)),
            **extra,
            **common,
        )
        numbers = dict(
            runs=result.runs,
            first_order=result.first_order,
            first_order_interval=result.first_order_interval,
            total=result.total,
            total_interval=result.total_interval,
        )
    else:
        for key in ("observed", "noise"):
            if key not in spec:
                raise ValueError(f"uq: calibrate needs '{key}'.")
        extra = {
            k: spec[k]
            for k in (
                "surrogate",
                "training_samples",
                "discrepancy",
                "sampler",
                "samples",
                "walkers",
                "chains",
                "level",
            )
            if k in spec
        }
        if "locations" in spec:
            extra["locations"] = _array_dict(spec["locations"], "locations")
        result = calibrate(
            model,
            dists,
            _array_dict(spec["observed"], "observed"),
            _array_dict(spec["noise"], "noise"),
            **extra,
            **common,
        )
        numbers = {
            name: dict(
                mean=result.mean(name),
                std=result.std(name),
                interval=result.interval(name),
                map=result.map[name],
                contraction=result.contraction[name],
                at_bound=result.at_bound[name],
                r_hat=result.r_hat[name],
                effective_sample_size=result.effective_sample_size[name],
            )
            for name in result.names
        }
    if verbose:
        print(result.summary())
    if spec.get("results"):
        target = base / str(spec["results"])
        with open(target, "w") as stream:
            json.dump(_jsonable({"study": study, **numbers}), stream, indent=2)
        if verbose:
            print(f"wrote {target}")
    return result
