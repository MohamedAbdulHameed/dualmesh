# SPDX-License-Identifier: LGPL-2.1-or-later
"""Every public name and parameter declaration follows the naming rules and
is complete (scripts/audit_names.py)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "audit_names.py"


def _audit():
    spec = importlib.util.spec_from_file_location("audit_names", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_every_public_name_and_declaration_passes_the_audit():
    findings = _audit().audit()
    assert findings == [], "\n".join(findings)


def test_the_audit_finds_what_it_checks():
    audit = _audit()
    assert audit.check_name("dt", "x")
    assert audit.check_name("thermal_k", "x")
    assert audit.check_name("Young_modulus", "x")
    assert not audit.check_name("Dirichlet_boundary_condition", "x")
    assert not audit.check_name("num_x_elements", "x")
    assert audit.check_description("a; b", "x")
    assert audit.check_description("10 °C", "x")
    assert audit.check_description("", "x")
    assert not audit.check_description("Thermal conductivity. Default 1.", "x")
