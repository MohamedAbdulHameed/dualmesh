# SPDX-License-Identifier: LGPL-2.1-or-later
"""Audit of the public names and parameter declarations.

Every public name (the object types of the C++ library and their parameters,
and the physics and couplings and their parameters) is checked against the
naming rules of the project:

* names are written in lower case and joined by underscores, except the
  proper names and abbreviations of :data:`CAPITALISED`, which keep their
  capitals, and the Python classes, which follow PEP 8;
* names are written in full words, except the common abbreviations of
  :data:`ALLOWED_ABBREVIATIONS`, so that the words of :data:`FORBIDDEN_WORDS`
  do not appear;

and every declaration is checked for completeness:

* every parameter has a description;
* every parameter of a physics or a coupling that has a default states that default and the reason for it in its description;
* no description contains a semicolon, a dash used as punctuation or a
  Unicode symbol.

Run it with ``python scripts/audit_names.py``.  It prints one line per
finding and exits with the number of findings, and the test suite runs it.
"""

from __future__ import annotations

import dataclasses
import re
import sys

from dualmesh import _core
from dualmesh import physics as physics_module

#: Proper names and abbreviations that keep their capitals (rule 4.1).
CAPITALISED = {"Dirichlet", "Neumann", "Robin", "Boussinesq", "Euler", "Bernoulli", "Timoshenko", "Taylor", "Hood", "NAFEMS", "PDE", "Newmark"}

#: Abbreviations allowed by rule 4.2.
ALLOWED_ABBREVIATIONS = {"num", "max", "min", "x", "y", "z", "r", "id"}

#: Abbreviations that rule 4.2 replaces by the full word.
FORBIDDEN_WORDS = {
    "dt": "time_step",
    "theta": "implicitness",
    "std": "standard_deviation",
    "k": "thermal_conductivity or the full name",
    "temp": "temperature",
    "coeff": "coefficient",
    "coef": "coefficient",
    "tol": "tolerance",
    "eps": "the full name",
    "iter": "iteration",
    "iters": "iterations",
    "disp": "displacement",
    "vel": "velocity",
    "nu": "poissons_ratio",
    "E": "youngs_modulus",
    "rho": "density",
    "mu": "dynamic_viscosity",
    "alpha": "the full name",
    "beta": "the full name",
    "dim": "dimension",
    "elem": "element",
    "elems": "elements",
    "pt": "point",
    "pts": "points",
    "bc": "boundary_condition",
    "bcs": "boundary_conditions",
    "nl": "nonlinear",
    "lin": "linear",
    "prop": "property",
    "props": "properties",
    "var": "variable",
    "vars": "variables",
    "func": "function",
    "fn": "function",
    "init": "initial",
    "cfg": "configuration",
    "params": "parameters",
    "poisson_ratio": "poissons_ratio",
    "material": "property (an object that computes properties is a property object)",
    "materials": "properties",
}

UNICODE = re.compile(r"[°²³µ×‐-―←-⇿∀-⋿Ͱ-Ͽ]")
DASH = re.compile(r"\s--\s|—|–")


def _words(name: str) -> list[str]:
    return [w for w in name.split("_") if w]


def check_name(name: str, where: str) -> list[str]:
    """Findings for one parameter or type name."""
    findings = []
    words = _words(name)
    for i, word in enumerate(words):
        # nu_fission is the established name of the product of the number of
        # neutrons per fission and the fission cross section.
        if word == "nu" and i + 1 < len(words) and words[i + 1] == "fission":
            continue
        if word in FORBIDDEN_WORDS:
            findings.append(f"{where}: '{name}' abbreviates '{word}'. Write {FORBIDDEN_WORDS[word]}.")
        if word != word.lower() and word not in CAPITALISED and not re.fullmatch(r"[A-Z][a-z0-9]*[0-9]+[A-Za-z0-9]*|[A-Z]+[0-9]*", word):
            findings.append(f"{where}: '{name}' has the capitalised word '{word}', which is not a proper name or an abbreviation of the list.")
    if "__" in name or name.startswith("_") or name.endswith("_"):
        findings.append(f"{where}: '{name}' has a stray underscore.")
    return findings


def check_description(text: str, where: str) -> list[str]:
    findings = []
    if not text.strip():
        findings.append(f"{where}: no description.")
    if ";" in text:
        findings.append(f"{where}: the description has a semicolon.")
    if DASH.search(text):
        findings.append(f"{where}: the description has a dash used as punctuation.")
    if UNICODE.search(text):
        findings.append(f"{where}: the description has a Unicode symbol ({UNICODE.search(text).group()}). Write it in words or in plain characters.")
    return findings


def audit_objects() -> list[str]:
    findings = []
    for type_name in _core.registered_types():
        findings += check_name(type_name, f"object type '{type_name}'")
        for p in _core.object_parameters(type_name):
            where = f"{type_name}.{p['name']}"
            findings += check_name(p["name"], where)
            findings += check_description(p["description"], where)
    return findings


def _fields(cls):
    return [f for f in dataclasses.fields(cls) if not f.name.startswith("_") and f.init]


def _has_default(f) -> bool:
    return f.default is not dataclasses.MISSING or f.default_factory is not dataclasses.MISSING


def audit_dataclass(cls, label: str) -> list[str]:
    findings = []
    for f in _fields(cls):
        where = f"{label}.{f.name}"
        findings += check_name(f.name, where)
        description = f.metadata.get("description", "")
        findings += check_description(description, where)
        # A parameter whose default is None and whose description says
        # "Required" has no default: the input of the case gives it.
        required = f.default is None and re.search(r"\bRequired\b", description)
        if _has_default(f) and description and not required and not re.search(r"(?i)\bdefault", description):
            findings.append(f"{where}: the description does not state the default and its reason.")
    return findings


def audit_physics() -> list[str]:
    findings = []
    for name in physics_module.registered():
        findings += check_name(name, f"physics type '{name}'")
        findings += audit_dataclass(physics_module._REGISTRY[name], name)
    return findings


def audit() -> list[str]:
    return audit_objects() + audit_physics()


def main() -> int:
    findings = audit()
    for line in findings:
        print(line)
    print(f"{len(findings)} findings", file=sys.stderr)
    return min(len(findings), 255)


if __name__ == "__main__":
    raise SystemExit(main())
