# SPDX-License-Identifier: LGPL-2.1-or-later
"""Tests of the release script's static checks, and of the repository against
them, so that the fixes needed for v0.1.0 cannot be undone unnoticed."""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "release.py"

pytestmark = pytest.mark.skipif(
    not SCRIPT.exists(), reason="scripts/release.py is not in this tree"
)


@pytest.fixture(scope="module")
def release():
    spec = importlib.util.spec_from_file_location("dualmesh_release_script", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_rst_title_underline_is_not_a_conflict_marker(release):
    assert release.conflict_markers("Solving\n=======\n\nText.\n") == []


def test_genuine_conflict_is_found(release):
    text = "a\n<<<<<<< HEAD\nours\n=======\ntheirs\n>>>>>>> branch\nb\n"
    assert release.conflict_markers(text) == [2, 4, 6]


def test_diff3_conflict_is_found(release):
    text = "<<<<<<< HEAD\nx\n||||||| base\ny\n=======\nz\n>>>>>>> other\n"
    assert release.conflict_markers(text) == [1, 3, 5, 7]


def test_contents_needs_the_furo_class(release):
    bare = "Title\n=====\n\n.. contents::\n   :local:\n   :depth: 2\n\nText\n"
    good = f".. contents::\n   :local:\n   :depth: 2\n   :class: {release.FURO_CONTENTS_CLASS}\n\nText\n"
    toctree = ".. toctree::\n   :maxdepth: 2\n\n   page\n"
    assert release.contents_without_furo_class(bare) == [4]
    assert release.contents_without_furo_class(good) == []
    assert release.contents_without_furo_class(toctree) == []


def test_workflow_guard_rejects_the_fixed_mistakes(release):
    good = (ROOT / ".github" / "workflows" / "wheels.yml").read_text(encoding="utf-8")
    assert release.workflow_problems(good, {"wheels.yml": good}) == []
    retired = good.replace("macos-15-intel", "macos-13")
    assert any("macos-13" in p for p in release.workflow_problems(retired, {}))
    assert "# v4.2.1" in good
    old = good.replace("# v4.2.1", "# v2.21")
    assert any("older than" in p for p in release.workflow_problems(old, {}))
    tag_only = re.sub(r"pypa/cibuildwheel@[0-9a-f]{40} # v4.2.1", "pypa/cibuildwheel@v4.2.1", good)
    assert tag_only != good
    assert any(
        "pinned to a commit" in p
        for p in release.workflow_problems(tag_only, {"wheels.yml": tag_only})
    )
    no_dry_run = good.replace("workflow_dispatch:", "")
    assert any("workflow_dispatch" in p for p in release.workflow_problems(no_dry_run, {}))
    token = good + "\n        with:\n          password: ${{ secrets.PYPI_API_TOKEN }}\n"
    assert any(
        "Trusted Publishing" in p for p in release.workflow_problems(token, {"wheels.yml": token})
    )


def test_repository_matches_the_released_configuration(release, monkeypatch):
    monkeypatch.setattr(release, "ROOT", ROOT)
    assert release.metadata_problems(release.project_metadata()) == []
    meta = release.project_metadata()
    assert meta["name"] == "dualmesh-multiphysics"
    assert release.module_version() == meta["version"]
    assert release.citation_version() == meta["version"]
    workflows = {
        p.name: p.read_text(encoding="utf-8")
        for p in (ROOT / ".github" / "workflows").glob("*.yml")
    }
    assert release.workflow_problems(workflows["wheels.yml"], workflows) == []
    missing = [
        f"{path.relative_to(ROOT)}:{line}"
        for path in sorted((ROOT / "docs").rglob("*.rst"))
        if "_build" not in path.parts
        for line in release.contents_without_furo_class(path.read_text(encoding="utf-8"))
    ]
    assert missing == []
