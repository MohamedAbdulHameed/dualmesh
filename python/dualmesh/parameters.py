# SPDX-License-Identifier: LGPL-2.1-or-later
"""The declaration of the parameters of the Python capabilities.

The physics, the couplings and the input groups of the applications declare
their parameters as dataclass fields made by :func:`parameter`, with a unit
and a description, which ends with the reason for the default.  The same
declaration validates the keywords (:func:`check_keywords`), prints the
reference of a type (:func:`describe_fields`) and the input report of an
application (:func:`describe`), in the format of the objects of the C++
library.
"""

from __future__ import annotations

import dataclasses
import difflib
import functools
import re
from dataclasses import field

import numpy as np

from ._core import InputError

__all__ = ["InputError", "check_keywords", "describe", "describe_fields", "keyword_checked", "parameter", "write_csv_rows", "write_json_numbers"]


def parameter(default=dataclasses.MISSING, unit: str = "", description: str = "", **kwargs):
    """A dataclass field with a unit and a description.  For a default, the
    description ends with the reason for it."""
    metadata = {"unit": unit, "description": description}
    if default is dataclasses.MISSING or "default_factory" in kwargs:
        return field(metadata=metadata, **kwargs)
    return field(default=default, metadata=metadata, **kwargs)


def _names(cls) -> list[str]:
    return [f.name for f in dataclasses.fields(cls) if not f.name.startswith("_")]


def check_keywords(where: str, cls, keywords) -> None:
    """Refuse a keyword that is not a parameter of the dataclass ``cls``, with
    the closest parameter name and the list of the accepted ones."""
    names = _names(cls)
    for key in keywords:
        if key not in names:
            close = difflib.get_close_matches(key, names, n=1)
            hint = f" Did you mean '{close[0]}'?" if close else ""
            raise InputError(f"{where}: unknown parameter '{key}'.{hint} Accepted parameters are: {' '.join(sorted(names))}")


def keyword_checked(cls):
    """Class decorator for a dataclass: a misspelled keyword is answered with
    the closest parameter name, and a missing required parameter is named
    with its description."""
    init = cls.__init__

    @functools.wraps(init)
    def checked_init(self, *args, **kwargs):
        check_keywords(cls.__name__, cls, kwargs)
        given = set(kwargs) | set(_names(cls)[: len(args)])
        for f in dataclasses.fields(cls):
            if f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING and f.init and f.name not in given:
                raise InputError(f"{cls.__name__}: missing required parameter '{f.name}' ({f.metadata.get('description', '')})")
        init(self, *args, **kwargs)

    cls.__init__ = checked_init
    return cls


def _default_of(f):
    if f.default is not dataclasses.MISSING:
        return f.default
    if f.default_factory is not dataclasses.MISSING:
        return f.default_factory()
    return dataclasses.MISSING


def _same(a, b) -> bool:
    try:
        if isinstance(a, (list, tuple, np.ndarray)) or isinstance(b, (list, tuple, np.ndarray)):
            return np.array_equal(np.asarray(a, dtype=object), np.asarray(b, dtype=object))
        return a == b
    except Exception:  # pragma: no cover - exotic values
        return False


def _sequence_text(value) -> str:
    """A list or an array as text: the numbers when there are six or fewer,
    otherwise their count and range."""
    if any(v is None or isinstance(v, str) for v in value):
        return "[" + ", ".join(str(v) for v in value) + "]"
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return "[" + ", ".join(str(v) for v in value) + "]"
    if array.size > 6:
        return f"{array.size} values, {array.min():.4g} to {array.max():.4g}"
    if array.ndim > 1:
        return "[" + ", ".join(_sequence_text(row) for row in array) + "]"
    return "[" + ", ".join(f"{float(v):.6g}" for v in array) + "]"


def describe(spec) -> list[list[str]]:
    """Rows ``[name, value, unit, source]`` for a dataclass instance, where
    ``source`` is ``(default)`` when the value is the field's default and
    ``given`` otherwise."""
    rows = []
    for f in dataclasses.fields(spec):
        if f.name.startswith("_"):
            continue
        value = getattr(spec, f.name)
        default = _default_of(f)
        is_default = default is not dataclasses.MISSING and _same(value, default)
        if dataclasses.is_dataclass(value):
            text = type(value).__name__
        elif callable(value):
            text = getattr(value, "__name__", "function")
        elif isinstance(value, (list, tuple)) and value and all(dataclasses.is_dataclass(v) for v in value):
            text = "[" + ", ".join(str(getattr(v, "name", type(v).__name__)) for v in value) + "]"
        elif isinstance(value, (list, tuple, np.ndarray)):
            text = _sequence_text(value)
        elif isinstance(value, float):
            text = f"{value:.6g}"
        else:
            text = str(value)
        rows.append([f.name, text, f.metadata.get("unit", ""), "(default)" if is_default else "given"])
    return rows


def describe_fields(cls) -> str:
    """The parameters of a dataclass type, one line each, in the format of
    :func:`dualmesh.describe` for the objects of the C++ library."""
    lines = []
    for f in sorted(dataclasses.fields(cls), key=lambda f: f.name):
        if f.name.startswith("_"):
            continue
        required = f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING
        unit = f.metadata.get("unit", "")
        kind = ", required" if required else ""
        unit_text = f", {unit}" if unit else ""
        lines.append(f"  {f.name} ({_type_name(f.type)}{unit_text}{kind}): {f.metadata.get('description', '')}")
    return "\n".join(lines)


def _type_name(annotation) -> str:
    text = annotation if isinstance(annotation, str) else getattr(annotation, "__name__", str(annotation))
    text = text.replace(" | None", "").replace("Sequence[str]", "list of strings").replace("Sequence[float]", "list of reals").replace("Sequence[float | None]", "list of reals")
    for word, name in (("float", "real"), ("str", "string"), ("bool", "boolean"), ("int", "integer"), ("dict", "dictionary")):
        text = re.sub(rf"\b{word}\b", name, text)
    return text.replace(" | ", " or ")


def jsonable(value):
    """Convert a nested structure of dicts, sequences, arrays and numbers to
    one that :mod:`json` writes, with a non-finite number written as null."""
    import math

    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, (np.floating, float)):
        v = float(value)
        return v if math.isfinite(v) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def write_json_numbers(numbers: dict, path) -> None:
    """Write the dict of a result to a JSON file."""
    import json

    with open(path, "w") as stream:
        json.dump(jsonable(numbers), stream, indent=2)


def write_csv_rows(path, header, rows) -> None:
    """Write a header and rows of numbers or texts to a CSV file (a
    non-finite number is written as an empty field)."""
    import csv
    import math

    def cell(value):
        if isinstance(value, (bool, np.bool_)):
            return int(value)
        if isinstance(value, (float, np.floating)):
            return repr(float(value)) if math.isfinite(value) else ""
        return value

    with open(path, "w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        for row in rows:
            writer.writerow([cell(v) for v in row])
