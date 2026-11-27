# SPDX-License-Identifier: LGPL-2.1-or-later
"""The declaration of the parameters of the Python capabilities.

The physics, the couplings and the input groups such as :class:`~dualmesh.Output` declare their
parameters as dataclass fields made by :func:`parameter`, with a unit and a description, which
ends with the reason for the default.  The same declaration validates the keywords
(:func:`check_keywords`, :func:`keyword_checked`) and prints the reference of a type
(:func:`describe_fields`), in the format of the objects of the C++ library.  A message that
refuses a misspelled name ends with the nearest valid name, from :func:`suggestion`.
"""

from __future__ import annotations

import dataclasses
import difflib
import functools
import json
import math
import re
from dataclasses import field

import numpy

from ._core import InputError

__all__ = [
    "InputError",
    "check_keywords",
    "describe_fields",
    "keyword_checked",
    "parameter",
    "suggestion",
    "write_json_numbers",
]


def suggestion(word, choices, listing: bool = True) -> str:
    """The end of a message that refuses a misspelled ``word``: the nearest of ``choices``, as
    " Did you mean 'x'?", or the list of the choices when none is near and ``listing`` holds."""
    choices = sorted(str(choice) for choice in choices)
    nearest = difflib.get_close_matches(str(word), choices, n=1)
    if nearest:
        return f" Did you mean '{nearest[0]}'?"
    if not choices or not listing:
        return ""
    return " The valid names are " + ", ".join(f"'{choice}'" for choice in choices) + "."


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
    the closest parameter name, or the list of the accepted ones."""
    names = _names(cls)
    for key in keywords:
        if key not in names:
            raise InputError(f"{where}: unknown parameter '{key}'." + suggestion(key, names))


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
            if (
                f.default is dataclasses.MISSING
                and f.default_factory is dataclasses.MISSING
                and f.init
                and f.name not in given
            ):
                raise InputError(
                    f"{cls.__name__}: missing required parameter '{f.name}' ({f.metadata.get('description', '')})"
                )
        init(self, *args, **kwargs)

    cls.__init__ = checked_init

    def checked_setattr(self, name, value):
        # A parameter can be set as an attribute (output.file_base = "plate").
        # A misspelled name is refused, so that it does not create a new
        # attribute that nothing reads.
        if not name.startswith("_") and name not in _names(cls) and not hasattr(type(self), name):
            check_keywords(cls.__name__, cls, [name])
        object.__setattr__(self, name, value)

    cls.__setattr__ = checked_setattr
    return cls


def describe_fields(cls) -> str:
    """The parameters of a dataclass type, one line each, in the format of
    :func:`dualmesh.describe` for the objects of the C++ library."""
    lines = []
    for f in sorted(dataclasses.fields(cls), key=lambda f: f.name):
        if f.name.startswith("_"):
            continue
        required = f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING
        unit = f.metadata.get("unit", "")
        unit_text = f", {unit}" if unit else ""
        kind = ", required" if required else ""
        lines.append(
            f"  {f.name} ({_type_name(f.type)}{unit_text}{kind}): {f.metadata.get('description', '')}"
        )
    return "\n".join(lines)


def _type_name(annotation) -> str:
    text = (
        annotation
        if isinstance(annotation, str)
        else getattr(annotation, "__name__", str(annotation))
    )
    text = (
        text.replace(" | None", "")
        .replace("Sequence[str]", "list of strings")
        .replace("Sequence[float]", "list of reals")
        .replace("Sequence[float | None]", "list of reals")
    )
    for word, name in (
        ("float", "real"),
        ("str", "string"),
        ("bool", "boolean"),
        ("int", "integer"),
        ("dict", "dictionary"),
    ):
        text = re.sub(rf"\b{word}\b", name, text)
    return text.replace(" | ", " or ")


def jsonable(value):
    """Convert a nested structure of dicts, sequences, arrays and numbers to
    one that :mod:`json` writes, with a non-finite number written as null."""
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, numpy.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, (numpy.floating, float)):
        v = float(value)
        return v if math.isfinite(v) else None
    if isinstance(value, numpy.integer):
        return int(value)
    return value


def write_json_numbers(numbers: dict, path) -> None:
    """Write the dict of a result to a JSON file."""
    with open(path, "w") as stream:
        json.dump(jsonable(numbers), stream, indent=2)
