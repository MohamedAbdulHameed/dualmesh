# SPDX-License-Identifier: LGPL-2.1-or-later
"""The declaration of the parameters of the Python capabilities.

The physics, the couplings and the input groups of the applications declare
their parameters as dataclass fields made by :func:`parameter`, with a unit
and a description, which ends with the reason for the default.  The same
declaration validates the keywords (:func:`check_keywords`), prints the
reference of a type (:func:`describe_fields`) and the input report of an
application (:func:`describe`), in the format of the objects of the C++
library.

A parameter has one of three kinds:

* required: :func:`parameter` without a default;
* a default: :func:`parameter` with a default, which is a fixed property of
  a material or a well-established value, with its source;
* a substitute: :func:`substitute`, for the data of the case (for example
  the grain size of a metal).  When the value is not given, the application
  uses a typical value of the class of material and warns once per run with
  a :class:`SubstitutedValueWarning` that names the value, its source and its
  effect.

A parameter that only some models read declares the condition with
``read_when`` (a :class:`When`).  An application calls
:func:`resolve_inputs` with its resolved models: a parameter that no active
model reads is not required and is not substituted, and a value given for it
is accepted and listed as not used by the active models.
"""

from __future__ import annotations

import dataclasses
import difflib
import functools
import re
import warnings
from dataclasses import field

import numpy as np

from ._core import InputError

__all__ = ["InputError", "InputResolution", "SubstitutedValueWarning", "When", "check_keywords", "describe", "describe_fields", "is_read", "keyword_checked", "parameter", "resolve_inputs", "substitute", "write_csv_rows", "write_json_numbers"]


class SubstitutedValueWarning(UserWarning):
    """A value of the data of the case was not given, and the application
    used a typical value of the class of material in its place."""


class When:
    """The condition under which an active model reads a parameter.

    ``text`` completes the phrase "read ..." in the reference, for example
    ``"with creep"``.  ``test(group, models)`` returns True when a model
    that reads the parameter is active, where ``group`` is the input group
    and ``models`` are the resolved models of the application."""

    def __init__(self, text: str, test):
        self.text = text
        self.test = test

    def __call__(self, group, models) -> bool:
        if models is None:
            return True
        return bool(self.test(group, models))


def parameter(default=dataclasses.MISSING, unit: str = "", description: str = "", *, read_when: When | None = None, required_when_read: bool = False, reference: str = "", **kwargs):
    """A dataclass field with a unit and a description.  For a default, the
    description ends with the reason for it.

    ``read_when`` is the condition under which a model reads the parameter
    (default: always).  ``required_when_read`` makes a parameter with the
    default None required when a model reads it.  ``reference`` is the
    short source of the default (for example ``"IAPWS R7-97"``), which the input
    report prints beside the value."""
    if required_when_read and default is not None:
        raise ValueError("parameter: required_when_read needs the default None.")
    metadata = {"unit": unit, "description": description, "read_when": read_when, "required_when_read": required_when_read, "reference": reference}
    if default is dataclasses.MISSING or "default_factory" in kwargs:
        return field(metadata=metadata, **kwargs)
    return field(default=default, metadata=metadata, **kwargs)


def substitute(value, unit: str = "", description: str = "", *, source: str, effect: str, read_when: When | None = None):
    """A dataclass field for the data of a case that has a typical value of
    its class of material.

    The field defaults to None, which means "not given".  When a model reads
    the parameter and it is not given, :func:`resolve_inputs` puts ``value``
    in its place (a number, or a function of the input group) and records
    it, so that the application warns once per run.  ``source`` is the
    source of the typical value, and ``effect`` says what the value changes
    in the result."""
    if not source or not effect:
        raise ValueError("substitute: give the source of the typical value and its effect.")
    metadata = {"unit": unit, "description": description, "read_when": read_when, "required_when_read": False, "reference": source, "substitute": value, "effect": effect}
    return field(default=None, metadata=metadata)


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

    def checked_setattr(self, name, value):
        # A parameter can be set as an attribute (group.grain_size = 8e-6).
        # A misspelled name is refused, so that it does not create a new
        # attribute that no model reads.
        if not name.startswith("_") and name not in _names(cls) and not hasattr(type(self), name):
            check_keywords(cls.__name__, cls, [name])
        object.__setattr__(self, name, value)

    cls.__setattr__ = checked_setattr
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


def _is_given(f, value) -> bool:
    """Whether the value of field ``f`` was given (it differs from the
    default, or the field has no default)."""
    default = _default_of(f)
    return default is dataclasses.MISSING or not _same(value, default)


def is_read(group, name_or_field, models) -> bool:
    """Whether an active model reads the parameter of ``group`` (a field or
    its name) for the resolved ``models`` (always when ``models`` is None)."""
    f = name_or_field if isinstance(name_or_field, dataclasses.Field) else next(f for f in dataclasses.fields(group) if f.name == name_or_field)
    condition = f.metadata.get("read_when")
    return condition is None or condition(group, models)


def _value_text(value) -> str:
    if dataclasses.is_dataclass(value):
        return type(value).__name__
    if callable(value):
        return getattr(value, "__name__", "function")
    if isinstance(value, (list, tuple)) and value and all(dataclasses.is_dataclass(v) for v in value):
        return "[" + ", ".join(str(getattr(v, "name", type(v).__name__)) for v in value) + "]"
    if isinstance(value, (list, tuple, np.ndarray)):
        return _sequence_text(value)
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _fields(spec):
    return [f for f in dataclasses.fields(spec) if not f.name.startswith("_")]


def describe(spec, models=None, substituted=()) -> list[list[str]]:
    """Rows ``[name, value, unit, source, reference]`` for a dataclass
    instance, for the parameters that the active ``models`` read (all of
    them when ``models`` is None).  ``source`` is ``(default)`` when the
    value is the field's default, ``(substitute)`` for a name in
    ``substituted`` and ``given`` otherwise.  ``reference`` is the declared
    source of a default or a substitute."""
    rows = []
    for f in _fields(spec):
        if not is_read(spec, f, models):
            continue
        value = getattr(spec, f.name)
        if f.name in substituted:
            source = "(substitute)"
        else:
            source = "given" if _is_given(f, value) else "(default)"
        reference = f.metadata.get("reference", "") if source != "given" else ""
        rows.append([f.name, _value_text(value), f.metadata.get("unit", ""), source, reference])
    return rows


DESCRIBE_HEADERS = ["parameter", "value", "unit", "source", "reference"]
"""The column headers of the rows of :func:`describe`."""


@dataclasses.dataclass
class Substitution:
    """A typical value that an application used for a value of the case
    that was not given."""

    path: str
    value: object
    unit: str
    source: str
    effect: str


@dataclasses.dataclass
class UnusedValue:
    """A given value that no active model reads."""

    path: str
    value: str
    unit: str
    condition: str


@dataclasses.dataclass
class InputResolution:
    """What :func:`resolve_inputs` did: the substituted values and the
    given values that no active model reads."""

    substituted: list = dataclasses.field(default_factory=list)
    unused: list = dataclasses.field(default_factory=list)

    def substituted_names(self, path: str) -> set:
        """The names of the substituted parameters of the group at ``path``."""
        prefix = path + "."
        return {s.path[len(prefix) :] for s in self.substituted if s.path.startswith(prefix) and "." not in s.path[len(prefix) :]}

    def substituted_rows(self) -> list[list[str]]:
        return [[s.path, _value_text(s.value), s.unit, s.source, s.effect] for s in self.substituted]

    def unused_rows(self) -> list[list[str]]:
        return [[u.path, u.value, u.unit, f"read {u.condition}"] for u in self.unused]

    def warn(self, where: str, stacklevel: int = 3) -> None:
        """One :class:`SubstitutedValueWarning` for all the substituted
        values, when there are any."""
        if not self.substituted:
            return
        items = "; ".join(f"{s.path} = {_value_text(s.value)}{' ' + s.unit if s.unit else ''} ({s.source}; {s.effect})" for s in self.substituted)
        warnings.warn(f"{where}: these values of the case were not given, and typical values are used: {items}. Give the values of the case.", SubstitutedValueWarning, stacklevel=stacklevel)


def _resolve_group(path: str, group, models, resolution: InputResolution):
    changes = {}
    for f in _fields(group):
        if not f.init:
            continue
        value = getattr(group, f.name)
        if not is_read(group, f, models):
            if _is_given(f, value):
                resolution.unused.append(UnusedValue(f"{path}.{f.name}", _value_text(value), f.metadata.get("unit", ""), f.metadata["read_when"].text))
            continue
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            resolved = _resolve_group(f"{path}.{f.name}", value, models, resolution)
            if resolved is not value:
                changes[f.name] = resolved
            continue
        if value is not None:
            continue
        if "substitute" in f.metadata:
            typical = f.metadata["substitute"]
            typical = typical(group) if callable(typical) else typical
            changes[f.name] = typical
            resolution.substituted.append(Substitution(f"{path}.{f.name}", typical, f.metadata.get("unit", ""), f.metadata["reference"], f.metadata["effect"]))
        elif f.metadata.get("required_when_read"):
            condition = f.metadata.get("read_when")
            when = f", read {condition.text}" if condition is not None else ""
            raise InputError(f"{type(group).__name__}: missing required parameter '{f.name}' ({f.metadata.get('description', '')}){when}.")
    # The copy is made through the constructor, so that the values set as
    # attributes after construction are validated too.
    return dataclasses.replace(group, **changes)


def resolve_inputs(groups, models=None) -> tuple[dict, InputResolution]:
    """Resolve the input groups ``{path: group}`` of an application for its
    active ``models``.

    Returns copies of the groups, made through their constructors so that
    every value is validated (also a value set as an attribute), with the
    typical values of the substitutes (the groups of the user are not
    changed), and the :class:`InputResolution`.  A missing parameter that
    is ``required_when_read`` and read by an active model raises an
    :class:`InputError`."""
    resolution = InputResolution()
    resolved = {path: _resolve_group(path, group, models, resolution) for path, group in groups.items()}
    return resolved, resolution


def describe_fields(cls) -> str:
    """The parameters of a dataclass type, one line each, in the format of
    :func:`dualmesh.describe` for the objects of the C++ library."""
    lines = []
    for f in sorted(dataclasses.fields(cls), key=lambda f: f.name):
        if f.name.startswith("_"):
            continue
        required = f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING
        unit = f.metadata.get("unit", "")
        condition = f.metadata.get("read_when")
        when = f" {condition.text}" if condition is not None else ""
        if required:
            kind = ", required"
        elif f.metadata.get("required_when_read"):
            kind = f", required{when}" if when else ", required"
        elif "substitute" in f.metadata:
            typical = f.metadata["substitute"]
            typical_text = "calculated" if callable(typical) else _value_text(typical)
            kind = f", substitute {typical_text} ({f.metadata['reference']}) when not given"
            kind += f", read{when}" if when else ""
        else:
            kind = f", read{when}" if when else ""
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
