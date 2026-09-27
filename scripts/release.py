#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Release dualmesh to GitHub, Read the Docs and PyPI, one checked gate at a time.

The distribution on PyPI is ``dualmesh-multiphysics``; the package is imported
as ``dualmesh`` and the command-line program is ``dualmesh``.  The first
release, v0.1.0 (commit bc96db4512378f525729f6dc48e83652fc982db6), needed a
series of manual fixes (see docs/developing.rst, "Making a release"); every one
of them is a gate here, so that the next release does not repeat them.

Usage (from the root of a clone, on branch main)::

    python scripts/release.py check              # gates 1-15, all local
    python scripts/release.py push               # gate 16: push main (fast-forward only)
    python scripts/release.py ci --wait          # gate 17: CI green on the pushed commit
    python scripts/release.py dry-run --wait     # gates 18-19: Wheels dry run from main
    python scripts/release.py tag                # gate 20: create and push vX.Y.Z
    python scripts/release.py monitor --wait     # gates 21-22: tag build and PyPI upload
    python scripts/release.py verify-pypi        # gates 23-24: fresh install from PyPI
    python scripts/release.py release            # all of the above, in order
    python scripts/release.py status             # read-only summary

Every command stops at the first failed gate with a message that says what
failed and what to do.  Nothing is ever force-pushed, no tag is created from
a dirty tree or before the dry run of the Wheels workflow has passed on the
same commit, and publication uses PyPI Trusted Publishing from the Wheels
workflow (no API token is used or stored).

GitHub is queried through its public REST API.  A token is optional (it only
raises the rate limit); it is taken from GH_TOKEN or GITHUB_TOKEN, or from
``gh auth token`` when the GitHub CLI is installed.  Starting the Wheels dry
run needs the GitHub CLI; without it the script prints the two clicks to make
in the browser and waits for the run to appear.

Only the Python standard library is used.  The local checks create their own
virtual environment (under build/release-check/, which git ignores) and
install the build, test, lint and documentation tools into it.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
import venv
from pathlib import Path

# -----------------------------------------------------------------------------
# The known-good release configuration.  Changing any of these is a deliberate
# decision, not something a release should do on the way.
# -----------------------------------------------------------------------------
OWNER = "MohamedAbdulHameed"
REPO = "dualmesh"
SLUG = f"{OWNER}/{REPO}"
EXPECTED_REMOTE = f"git@github-personal:{SLUG}.git"
BRANCH = "main"
DISTRIBUTION = "dualmesh-multiphysics"
IMPORT_NAME = "dualmesh"
CLI_NAME = "dualmesh"
CLI_TARGET = "dualmesh.cli:main"
DOCS_URL = "https://dualmesh.readthedocs.io"
CLANG_FORMAT_MAJOR = 18  # the version on the ubuntu-latest runner of the Lint job
WHEEL_OSES = ["ubuntu-latest", "macos-15-intel", "macos-14", "windows-latest"]
RETIRED_RUNNERS = ["macos-13", "macos-12", "macos-11"]
MIN_CIBUILDWHEEL = (4, 2, 1)
FURO_CONTENTS_CLASS = "this-will-duplicate-information-and-it-is-still-useful-here"
PYTHON_VERSIONS = ["cp39", "cp310", "cp311", "cp312", "cp313"]
WHEEL_PLATFORMS = {  # a wheel of each Python version is expected for each of these
    "Linux x86_64": re.compile(r"manylinux.*x86_64"),
    "macOS x86_64": re.compile(r"macosx_.*x86_64"),
    "macOS arm64": re.compile(r"macosx_.*arm64"),
    "Windows x86_64": re.compile(r"win_amd64"),
}

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "build" / "release-check"
STAMP = WORK / "passed.json"


class GateError(RuntimeError):
    """A release gate failed; the message says what to do."""


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------
def gate(number: str, title: str) -> None:
    """Announce a gate; the numbers are those of docs/developing.rst."""
    print(f"\n==> [gate {number}] {title}", flush=True)


def ok(message: str) -> None:
    print(f"    ok: {message}", flush=True)


def note(message: str) -> None:
    print(f"    {message}", flush=True)


def run(cmd, cwd=None, env=None, capture=False, check=True, quiet=False) -> str:
    """Run a command; on failure raise GateError with its output."""
    if not quiet:
        print("    $ " + " ".join(str(c) for c in cmd), flush=True)
    proc = subprocess.run(
        [str(c) for c in cmd],
        cwd=cwd or ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
    )
    if check and proc.returncode != 0:
        tail = "\n".join((proc.stdout or "").splitlines()[-40:]) if capture else ""
        raise GateError(f"command failed ({proc.returncode}): {' '.join(map(str, cmd))}\n{tail}")
    return proc.stdout or ""


def git(*args, capture=True, check=True) -> str:
    return run(["git", *args], capture=capture, check=check, quiet=True).strip()


def which(name: str) -> str | None:
    return shutil.which(name)


# -----------------------------------------------------------------------------
# Reading the project metadata without third-party packages
# -----------------------------------------------------------------------------
def _toml_tables(text: str) -> dict[str, dict[str, str]]:
    """A deliberately small reader for the flat string keys of pyproject.toml
    (enough for [project], [project.scripts] and [project.urls])."""
    try:
        import tomllib  # Python 3.11+

        data = tomllib.loads(text)
        flat: dict[str, dict[str, str]] = {}

        def walk(prefix, table):
            flat[prefix] = {k: v for k, v in table.items() if not isinstance(v, dict)}
            for k, v in table.items():
                if isinstance(v, dict):
                    walk(f"{prefix}.{k}" if prefix else k, v)

        walk("", data)
        return flat
    except ImportError:
        tables: dict[str, dict[str, str]] = {"": {}}
        current = ""
        for line in text.splitlines():
            m = re.match(r"^\[([^\]]+)\]\s*$", line)
            if m:
                current = m.group(1).strip()
                tables.setdefault(current, {})
                continue
            m = re.match(r'^([A-Za-z0-9_.\-"]+)\s*=\s*"([^"]*)"\s*$', line)
            if m:
                tables.setdefault(current, {})[m.group(1).strip('"')] = m.group(2)
        return tables


def project_metadata() -> dict:
    tables = _toml_tables((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = tables.get("project", {})
    return {
        "name": project.get("name"),
        "version": project.get("version"),
        "scripts": tables.get("project.scripts", {}),
        "urls": tables.get("project.urls", {}),
    }


def module_version() -> str | None:
    text = (ROOT / "python" / IMPORT_NAME / "__init__.py").read_text(encoding="utf-8")
    m = re.search(r'^__version__\s*=\s*"([^"]+)"', text, re.M)
    return m.group(1) if m else None


def citation_version() -> str | None:
    path = ROOT / "CITATION.cff"
    if not path.exists():
        return None
    m = re.search(r'^version:\s*"?([^"\s]+)"?', path.read_text(encoding="utf-8"), re.M)
    return m.group(1) if m else None


def changelog_has_release(version: str) -> bool:
    text = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    return (
        re.search(rf"^## \[{re.escape(version)}\] - \d{{4}}-\d{{2}}-\d{{2}}\s*$", text, re.M)
        is not None
    )


# -----------------------------------------------------------------------------
# Static checks of the repository (pure functions, tested in
# tests/python/test_release_script.py)
# -----------------------------------------------------------------------------
_CONFLICT_START = re.compile(r"^<{7}(?: |$)")
_CONFLICT_BASE = re.compile(r"^\|{7}(?: |$)")
_CONFLICT_END = re.compile(r"^>{7}(?: |$)")
_CONFLICT_MID = re.compile(r"^={7}$")


def conflict_markers(text: str) -> list[int]:
    """Line numbers (1-based) of genuine merge-conflict markers.

    ``git diff --check`` reports any line of seven '=' as a leftover conflict
    marker, which is wrong for a reStructuredText document: ``Solving`` over
    ``=======`` is a valid title.  Here a line of '=' counts only between a
    ``<<<<<<<`` and a ``>>>>>>>`` marker, which is where git puts it."""
    found: list[int] = []
    inside = False
    for number, line in enumerate(text.splitlines(), start=1):
        if _CONFLICT_START.match(line):
            inside = True
            found.append(number)
        elif _CONFLICT_BASE.match(line):
            found.append(number)
        elif _CONFLICT_END.match(line):
            inside = False
            found.append(number)
        elif inside and _CONFLICT_MID.match(line):
            found.append(number)
    return found


def contents_without_furo_class(text: str) -> list[int]:
    """Line numbers of ``.. contents::`` directives that lack the class with
    which Furo accepts an inline table of contents.  Toctrees are not
    concerned."""
    lines = text.splitlines()
    missing = []
    for i, line in enumerate(lines):
        if re.match(r"^\s*\.\.\s+contents::", line):
            indent = len(line) - len(line.lstrip())
            options = []
            for following in lines[i + 1 :]:
                if following.strip() == "":
                    break
                if len(following) - len(following.lstrip()) <= indent:
                    break
                options.append(following.strip())
            if not any(o.startswith(":class:") and FURO_CONTENTS_CLASS in o for o in options):
                missing.append(i + 1)
    return missing


def workflow_problems(wheels_yml: str, all_workflows: dict[str, str]) -> list[str]:
    """What is wrong with the Wheels workflow, measured against the
    configuration that produced v0.1.0."""
    problems = []
    for runner in RETIRED_RUNNERS:
        if re.search(rf"\b{re.escape(runner)}\b", wheels_yml):
            problems.append(
                f"the retired runner '{runner}' is back in wheels.yml (use macos-15-intel)"
            )
    m = re.search(r"os:\s*\[([^\]]*)\]", wheels_yml)
    oses = [o.strip() for o in m.group(1).split(",")] if m else []
    if oses != WHEEL_OSES:
        problems.append(f"the wheel matrix is {oses}, expected {WHEEL_OSES}")
    m = re.search(r"pypa/cibuildwheel@v(\d+)(?:\.(\d+))?(?:\.(\d+))?", wheels_yml)
    if not m:
        problems.append("pypa/cibuildwheel is not pinned to a version")
    else:
        version = tuple(int(x or 0) for x in m.groups())
        if version < MIN_CIBUILDWHEEL:
            problems.append(
                f"cibuildwheel v{'.'.join(map(str, version))} is older than the "
                f"v{'.'.join(map(str, MIN_CIBUILDWHEEL))} that fixed the Linux wheel tests"
            )
    if not re.search(r"^\s*workflow_dispatch:", wheels_yml, re.M):
        problems.append("wheels.yml has lost its workflow_dispatch trigger (the dry run)")
    if not re.search(r"tags:\s*\[\s*[\"']v\*[\"']\s*\]", wheels_yml):
        problems.append("wheels.yml no longer runs on tags 'v*'")
    if "if: startsWith(github.ref, 'refs/tags/v')" not in wheels_yml:
        problems.append(
            "the Publish to PyPI job is not guarded by startsWith(github.ref, 'refs/tags/v')"
        )
    if not re.search(r"^\s*environment:\s*pypi\s*$", wheels_yml, re.M):
        problems.append("the publish job does not use the GitHub environment 'pypi'")
    if not re.search(r"id-token:\s*write", wheels_yml):
        problems.append("the publish job lacks 'id-token: write' (Trusted Publishing)")
    if "pypa/gh-action-pypi-publish@release/v1" not in wheels_yml:
        problems.append("the publish job does not use pypa/gh-action-pypi-publish@release/v1")
    for name, text in all_workflows.items():
        if re.search(r"PYPI_API_TOKEN|password:\s*\$\{\{|twine upload", text):
            problems.append(f"{name} uploads with a stored token or twine; use Trusted Publishing")
    return problems


def metadata_problems(meta: dict) -> list[str]:
    problems = []
    if meta["name"] != DISTRIBUTION:
        problems.append(f"pyproject.toml name is {meta['name']!r}, expected {DISTRIBUTION!r}")
    if meta["scripts"].get(CLI_NAME) != CLI_TARGET:
        problems.append(f"[project.scripts] must map {CLI_NAME!r} to {CLI_TARGET!r}")
    if meta["urls"].get("Documentation") != DOCS_URL:
        problems.append(f"[project.urls] Documentation must be {DOCS_URL}")
    for key in ("Homepage", "Repository"):
        if meta["urls"].get(key) != f"https://github.com/{SLUG}":
            problems.append(f"[project.urls] {key} must be https://github.com/{SLUG}")
    return problems


# -----------------------------------------------------------------------------
# Gates 1-7: the state of the repository
# -----------------------------------------------------------------------------
def repository_gates(version: str | None, release: bool) -> str:
    """Gates 1-7.  Returns the version being released."""
    gate("1", "Remote 'origin' is the personal SSH alias of the dualmesh repository")
    url = git("remote", "get-url", "origin", check=False)
    if url != EXPECTED_REMOTE:
        raise GateError(
            f"origin is {url!r}, expected {EXPECTED_REMOTE!r} (the github-personal SSH alias, "
            "identity ~/.ssh/id_ed25519_mohamed).  Do not switch to another account."
        )
    ok(url)

    gate("2", f"On branch {BRANCH}")
    branch = git("rev-parse", "--abbrev-ref", "HEAD")
    if branch != BRANCH:
        raise GateError(f"on branch {branch!r}; releases are made from {BRANCH!r} only")
    ok(branch)

    gate("3", f"Synchronised with origin/{BRANCH} (fast-forward only)")
    git("fetch", "origin", "--tags", "--prune")
    pull = subprocess.run(
        ["git", "pull", "--ff-only", "origin", BRANCH], cwd=ROOT, text=True, capture_output=True
    )
    if pull.returncode != 0:
        raise GateError(
            (pull.stdout + pull.stderr).strip() + "\n"
            "local main and origin/main have diverged (for example after an edit made in the "
            "GitHub web interface).  Recover with\n"
            "      git fetch origin\n      git rebase origin/main\n"
            "resolve any conflict file by file (never with a global --ours or --theirs), rerun "
            "the checks, and push normally.  Never force-push main."
        )
    ahead = int(git("rev-list", "--count", f"origin/{BRANCH}..HEAD"))
    behind = int(git("rev-list", "--count", f"HEAD..origin/{BRANCH}"))
    if behind:
        raise GateError(f"still {behind} commit(s) behind origin/{BRANCH}")
    ok(f"HEAD {git('rev-parse', '--short', 'HEAD')}, {ahead} commit(s) not yet pushed")

    gate("4", "Working tree clean, no genuine conflict markers")
    dirty = git("status", "--porcelain")
    if dirty:
        raise GateError("the working tree is not clean:\n" + dirty)
    offenders = []
    for path in git("ls-files").splitlines():
        p = ROOT / path
        if (
            p.suffix.lower() in {".png", ".jpg", ".pdf", ".ico", ".gz", ".zip", ".bundle"}
            or not p.is_file()
        ):
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        lines = conflict_markers(text)
        if lines:
            offenders.append(f"{path}: lines {lines}")
    if offenders:
        raise GateError("unresolved merge conflicts:\n" + "\n".join(offenders))
    stray = sorted(
        str(p.relative_to(ROOT))
        for pattern in ("_core*.so", "_core*.pyd")
        for p in (ROOT / "python" / IMPORT_NAME).glob(pattern)
    )
    if stray:
        note(
            "note: an in-place extension exists in python/dualmesh ("
            + ", ".join(stray)
            + "); the checks build from 'git archive' and ignore it"
        )
    ok("clean")

    gate("5", "Package name, import name and command")
    meta = project_metadata()
    problems = metadata_problems(meta)
    if problems:
        raise GateError("\n".join(problems))
    ok(f"pip install {DISTRIBUTION}  ->  import {IMPORT_NAME}  ->  {CLI_NAME} command")

    gate("6", "Version agrees everywhere")
    wanted = version or meta["version"]
    found = {
        "pyproject.toml": meta["version"],
        f"python/{IMPORT_NAME}/__init__.py": module_version(),
        "CITATION.cff": citation_version(),
    }
    wrong = {k: v for k, v in found.items() if v is not None and v != wanted}
    if wrong or found["pyproject.toml"] is None:
        raise GateError(f"expected version {wanted}, found {found}")
    if not re.fullmatch(r"\d+\.\d+\.\d+(?:(?:a|b|rc)\d+)?", wanted):
        raise GateError(f"{wanted!r} is not a release version (X.Y.Z)")
    if release and not changelog_has_release(wanted):
        raise GateError(f"CHANGELOG.md has no '## [{wanted}] - YYYY-MM-DD' section")
    ok(wanted)

    gate("7", f"Tag v{wanted} does not exist yet (locally or on origin)")
    tag = f"v{wanted}"
    local = git("tag", "--list", tag)
    remote = git("ls-remote", "--tags", "origin", f"refs/tags/{tag}")
    if local or remote:
        where = " and ".join(w for w, t in (("locally", local), ("on origin", remote)) if t)
        if release:
            raise GateError(
                f"{tag} already exists {where}.  Bump the version in pyproject.toml, "
                f"python/{IMPORT_NAME}/__init__.py and CITATION.cff, and add a CHANGELOG section."
            )
        note(f"note: {tag} exists {where}; continuing because --dev was given")
    else:
        ok(f"{tag} is free")
    return wanted


def static_gates() -> None:
    gate("7+", "Wheels workflow and documentation keep the fixes of v0.1.0")
    workflows = {
        p.name: p.read_text(encoding="utf-8")
        for p in (ROOT / ".github" / "workflows").glob("*.yml")
    }
    problems = workflow_problems(workflows.get("wheels.yml", ""), workflows)
    for path in sorted((ROOT / "docs").rglob("*.rst")):
        if "_build" in path.parts:
            continue
        for line in contents_without_furo_class(path.read_text(encoding="utf-8")):
            problems.append(
                f"{path.relative_to(ROOT)}:{line}: '.. contents::' needs "
                f"':class: {FURO_CONTENTS_CLASS}'"
            )
    if problems:
        raise GateError("\n".join(problems))
    ok("runner matrix, cibuildwheel, Trusted Publishing and Furo contents classes as released")


# -----------------------------------------------------------------------------
# Gates 8-15: build, test, lint, document and package, locally
# -----------------------------------------------------------------------------
def _venv_python(path: Path) -> Path:
    return path / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def _venv_bin(path: Path, name: str) -> Path:
    return (
        path
        / ("Scripts" if os.name == "nt" else "bin")
        / (name + (".exe" if os.name == "nt" else ""))
    )


def _make_venv(path: Path) -> Path:
    if path.exists():
        shutil.rmtree(path)
    venv.EnvBuilder(with_pip=True, clear=True).create(path)
    python = _venv_python(path)
    run([python, "-m", "pip", "install", "--quiet", "--upgrade", "pip"])
    return python


def _find_clang_format(explicit: str | None) -> str:
    candidates = [explicit] if explicit else [f"clang-format-{CLANG_FORMAT_MAJOR}", "clang-format"]
    if not explicit and sys.platform == "darwin":
        for prefix in ("/opt/homebrew", "/usr/local"):
            candidates.append(f"{prefix}/opt/llvm@{CLANG_FORMAT_MAJOR}/bin/clang-format")
    for candidate in candidates:
        path = candidate if candidate and Path(candidate).is_file() else which(candidate or "")
        if not path:
            continue
        out = run([path, "--version"], capture=True, quiet=True)
        m = re.search(r"version (\d+)\.", out)
        if m and int(m.group(1)) == CLANG_FORMAT_MAJOR:
            return path
        note(f"skipping {path}: {out.strip()}")
    raise GateError(
        f"clang-format {CLANG_FORMAT_MAJOR} not found (the Lint job uses "
        f"{CLANG_FORMAT_MAJOR}; other major versions format differently).  Install it "
        f"(macOS: brew install llvm@{CLANG_FORMAT_MAJOR}; "
        f"Ubuntu: apt install clang-format-{CLANG_FORMAT_MAJOR}) or pass --clang-format PATH."
    )


def local_gates(version: str, args) -> None:
    sha = git("rev-parse", "HEAD")
    WORK.mkdir(parents=True, exist_ok=True)
    source = WORK / "source"

    gate(
        "11",
        "Export the committed tree (git archive), so untracked and generated files cannot leak in",
    )
    if source.exists():
        shutil.rmtree(source)
    source.mkdir(parents=True)
    archive = WORK / "source.tar"
    git("archive", "--format=tar", f"--output={archive}", "HEAD")
    with tarfile.open(archive) as tar:
        if hasattr(tarfile, "data_filter"):
            tar.extractall(source, filter="data")
        else:  # Python < 3.12 without the backported filters; our own archive
            tar.extractall(source)
    archive.unlink()
    ok(f"{sha[:12]} exported to {source.relative_to(ROOT)}")

    gate("8", "clang-format --dry-run --Werror (the exact command of the Lint job)")
    clang_format = _find_clang_format(args.clang_format)
    files = [
        str(p.relative_to(source))
        for d in ("include", "src", "python/bindings", "tests/cpp")
        for p in sorted((source / d).rglob("*"))
        if p.suffix in (".h", ".cpp")
    ]
    out = run(
        [clang_format, "--dry-run", "--Werror", *files],
        cwd=source,
        capture=True,
        check=False,
        quiet=True,
    )
    if out.strip():
        raise GateError(
            "clang-format would change files (fix with 'clang-format -i <file>' using version "
            f"{CLANG_FORMAT_MAJOR}):\n" + "\n".join(out.splitlines()[:30])
        )
    ok(f"{len(files)} C++ files formatted")

    if not args.skip_cxx:
        gate("9", "Native C++ configure, build and CTest")
        cxx = WORK / "cxx"
        run(
            [
                "cmake",
                "-S",
                source,
                "-B",
                cxx,
                "-DCMAKE_BUILD_TYPE=Release",
                "-DDUALMESH_BUILD_PYTHON=OFF",
            ],
            capture=True,
        )
        run(
            ["cmake", "--build", cxx, "--config", "Release", "-j", str(os.cpu_count() or 2)],
            capture=True,
        )
        run(
            ["ctest", "--test-dir", cxx, "--build-config", "Release", "--output-on-failure"],
            capture=True,
        )
        ok("C++ library and unit tests")

    gate(
        "-",
        "Tools: a fresh virtual environment with build, twine, ruff and the documentation tools",
    )
    tools = _make_venv(WORK / "venv")
    run([tools, "-m", "pip", "install", "--quiet", "build", "twine", "ruff"])
    ok(str(WORK / "venv"))

    gate("10", "Ruff")
    run(
        [_venv_bin(WORK / "venv", "ruff"), "check", "python", "tests", "examples", "scripts"],
        cwd=source,
    )
    run(
        [
            _venv_bin(WORK / "venv", "ruff"),
            "format",
            "--check",
            "python",
            "tests",
            "examples",
            "scripts",
        ],
        cwd=source,
    )
    ok("lint and formatting")

    gate("11", "Isolated build of the sdist, and of the wheel from that sdist")
    dist = WORK / "dist"
    if dist.exists():
        shutil.rmtree(dist)
    run(
        [tools, "-m", "build", "--outdir", dist, source]
    )  # sdist first, then the wheel built from it
    stem = DISTRIBUTION.replace("-", "_")
    sdists = sorted(dist.glob(f"{stem}-{version}.tar.gz"))
    wheels = sorted(dist.glob(f"{stem}-{version}-*.whl"))
    if len(sdists) != 1 or len(wheels) != 1:
        raise GateError(
            f"expected one sdist and one wheel for {version}, "
            f"found {[p.name for p in dist.iterdir()]}"
        )
    ok(f"{sdists[0].name}, {wheels[0].name}")

    gate("12", "twine check")
    run([tools, "-m", "twine", "check", "--strict", *sdists, *wheels])
    ok("metadata and long description render")

    gate("12", "Artifact audit: what the wheel and the sdist contain")
    import zipfile

    with zipfile.ZipFile(wheels[0]) as whl:
        names = whl.namelist()
    tops = {n.split("/")[0] for n in names}
    expected_tops = {IMPORT_NAME, f"{stem}-{version}.dist-info"}
    if tops != expected_tops:
        raise GateError(
            f"the wheel's top level is {sorted(tops)}, expected {sorted(expected_tops)}"
        )
    if not any(re.match(rf"{IMPORT_NAME}/_core\.[^/]*(so|pyd)$", n) for n in names):
        raise GateError("the wheel has no compiled extension dualmesh/_core")
    with tarfile.open(sdists[0]) as sd:
        members = sd.getnames()
    leaked = [
        m for m in members if re.search(r"/(build|_build|dist|\.git)/|_core[^/]*\.(so|pyd)$", m)
    ]
    if leaked:
        raise GateError("the sdist contains build products:\n" + "\n".join(leaked[:20]))
    ok(
        f"wheel: {len(names)} files under {sorted(tops)}; "
        f"sdist: {len(members)} files, no build products"
    )

    gate("13", "Install the built wheel with its test and documentation extras; full Python suite")
    test_env = WORK / "test-venv"
    tpy = _make_venv(test_env)
    run([tpy, "-m", "pip", "install", "--quiet", f"{wheels[0]}[all,test,docs]"])
    # The exported tree has no in-place extension, so tests/python/conftest.py
    # imports the installed wheel, which is what users will get.
    run([tpy, "-m", "pytest", "-q", "-x", "tests/python"], cwd=source)
    ok("suite passed against the installed wheel")

    gate("14", "Strict documentation build (sphinx -W --keep-going)")
    shutil.rmtree(source / "docs" / "_build", ignore_errors=True)
    run(
        [tpy, "-m", "sphinx", "-W", "--keep-going", "-b", "html", "docs", "docs/_build/html"],
        cwd=source,
    )
    ok("no warnings")

    gate("15", "Clean install of the wheel alone: import, metadata, command")
    clean_env = WORK / "clean-venv"
    cpy = _make_venv(clean_env)
    run([cpy, "-m", "pip", "install", "--quiet", wheels[0]])
    _check_installed(cpy, _venv_bin(clean_env, CLI_NAME), version, clean_env)
    ok("import dualmesh, importlib.metadata and 'dualmesh --version' all report " + version)

    STAMP.write_text(
        json.dumps({"sha": sha, "version": version, "time": time.time()}), encoding="utf-8"
    )
    if not args.keep:
        for d in (test_env, clean_env, source, WORK / "cxx"):
            shutil.rmtree(d, ignore_errors=True)
    note(f"local gates passed for {sha[:12]} (recorded in {STAMP.relative_to(ROOT)})")


def _check_installed(python: Path, cli: Path, version: str, env_root: Path) -> None:
    probe = (
        "import importlib.metadata as m, dualmesh, json;"
        f"print(json.dumps([dualmesh.__file__, dualmesh.__version__, m.version('{DISTRIBUTION}')]))"
    )
    with tempfile.TemporaryDirectory() as neutral:  # away from any source tree
        out = run([python, "-c", probe], cwd=neutral, capture=True, quiet=True)
        location, module_ver, dist_ver = json.loads(out.strip().splitlines()[-1])
        if Path(env_root).resolve() not in Path(location).resolve().parents:
            raise GateError(f"dualmesh was imported from {location}, not from the test environment")
        if module_ver != version or dist_ver != version:
            raise GateError(
                f"expected {version}; dualmesh.__version__={module_ver}, metadata={dist_ver}"
            )
        cli_out = run([cli, "--version"], cwd=neutral, capture=True, quiet=True).strip()
        if version not in cli_out:
            raise GateError(f"'{CLI_NAME} --version' printed {cli_out!r}")
        note(f"{location}\n    {cli_out}")


def require_stamp(version: str) -> None:
    sha = git("rev-parse", "HEAD")
    try:
        stamp = json.loads(STAMP.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        stamp = {}
    if stamp.get("sha") != sha or stamp.get("version") != version:
        raise GateError(
            f"the local checks have not passed on {sha[:12]} for {version}; run "
            "'python scripts/release.py check' first"
        )
    ok(f"local checks passed on {sha[:12]}")


# -----------------------------------------------------------------------------
# GitHub Actions, through the public REST API
# -----------------------------------------------------------------------------
def _token() -> str | None:
    for key in ("GH_TOKEN", "GITHUB_TOKEN"):
        if os.environ.get(key):
            return os.environ[key]
    if which("gh"):
        proc = subprocess.run(["gh", "auth", "token"], text=True, capture_output=True)
        if proc.returncode == 0 and proc.stdout.strip():
            return proc.stdout.strip()
    return None


def github(path: str) -> dict:
    request = urllib.request.Request(
        f"https://api.github.com/repos/{SLUG}/{path}",
        headers={"Accept": "application/vnd.github+json", "User-Agent": "dualmesh-release"},
    )
    token = _token()
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        raise GateError(f"GitHub API {path}: HTTP {error.code} {error.reason}") from error
    except urllib.error.URLError as error:
        raise GateError(f"GitHub API {path}: {error.reason}") from error


def latest_run(workflow: str, sha: str, event: str, branch: str | None = None) -> dict | None:
    query = f"actions/workflows/{workflow}/runs?head_sha={sha}&event={event}&per_page=20"
    if branch:
        query += f"&branch={branch}"
    runs = github(query).get("workflow_runs", [])
    runs.sort(key=lambda r: r.get("created_at", ""), reverse=True)
    return runs[0] if runs else None


def run_jobs(run_id: int) -> list[dict]:
    return github(f"actions/runs/{run_id}/jobs?per_page=100").get("jobs", [])


def wait_for_run(workflow, sha, event, branch=None, wait=False, timeout=4 * 3600) -> dict:
    start = time.time()
    announced = False
    while True:
        found = latest_run(workflow, sha, event, branch)
        if found and found["status"] == "completed":
            return found
        if not wait:
            state = found["status"] if found else "not started"
            raise GateError(f"{workflow} ({event}) on {sha[:12]} is {state}; rerun with --wait")
        if time.time() - start > timeout:
            raise GateError(f"timed out waiting for {workflow} on {sha[:12]}")
        if found and not announced:
            note(f"following {found['html_url']}")
            announced = True
        time.sleep(30)


def describe_jobs(jobs: list[dict]) -> str:
    return "\n".join(f"      {j['name']:<40} {j.get('conclusion') or j['status']}" for j in jobs)


def ci_gate(sha: str, wait: bool) -> None:
    gate("17", f"GitHub CI passed on {sha[:12]}")
    result = wait_for_run("ci.yml", sha, "push", BRANCH, wait)
    jobs = run_jobs(result["id"])
    print(describe_jobs(jobs))
    if result["conclusion"] != "success" or any(j.get("conclusion") != "success" for j in jobs):
        raise GateError(f"CI did not pass: {result['html_url']}")
    ok(result["html_url"])


def wheels_dry_run_gate(sha: str, wait: bool, dispatch: bool) -> None:
    gate(
        "18-19",
        f"Wheels dry run (workflow_dispatch from {BRANCH}) on {sha[:12]}: "
        "all platforms, publish skipped",
    )
    if dispatch and latest_run("wheels.yml", sha, "workflow_dispatch") is None:
        if which("gh"):
            run(["gh", "workflow", "run", "wheels.yml", "--repo", SLUG, "--ref", BRANCH])
            time.sleep(10)
        else:
            note(
                "the GitHub CLI is not installed; start the dry run in the browser:\n"
                f"      https://github.com/{SLUG}/actions/workflows/wheels.yml"
                f" -> Run workflow -> {BRANCH}"
            )
            wait = True
    result = wait_for_run("wheels.yml", sha, "workflow_dispatch", None, wait)
    jobs = run_jobs(result["id"])
    print(describe_jobs(jobs))
    by_name = {j["name"]: j for j in jobs}
    failures = []
    for os_name in WHEEL_OSES:
        job = by_name.get(f"Wheels ({os_name})")
        if not job or job.get("conclusion") != "success":
            failures.append(f"Wheels ({os_name}): {job.get('conclusion') if job else 'missing'}")
    sdist = by_name.get("Source distribution")
    if not sdist or sdist.get("conclusion") != "success":
        failures.append("Source distribution did not succeed")
    publish = by_name.get("Publish to PyPI")
    if publish and publish.get("conclusion") != "skipped":
        failures.append(
            f"Publish to PyPI was {publish.get('conclusion')} on a branch run; it must be skipped"
        )
    if failures:
        raise GateError(
            "the dry run is not green:\n      "
            + "\n      ".join(failures)
            + f"\n      {result['html_url']}"
        )
    ok(f"{result['html_url']} (publication skipped, as it must be on a branch)")


# -----------------------------------------------------------------------------
# Tag, publication and the check from PyPI
# -----------------------------------------------------------------------------
def tag_gate(version: str, yes: bool) -> None:
    tag = f"v{version}"
    sha = git("rev-parse", "HEAD")
    gate("20", f"Create and push the annotated tag {tag} on {sha[:12]}")
    if git("rev-parse", f"origin/{BRANCH}") != sha:
        raise GateError(f"HEAD is not origin/{BRANCH}; push main first ('release.py push')")
    if not yes:
        answer = input(f"    Type {tag} to tag {sha[:12]} and publish it to PyPI: ").strip()
        if answer != tag:
            raise GateError("not confirmed; nothing was tagged")
    git("tag", "-a", tag, "-m", f"dualmesh {tag}")
    tagged = git("rev-list", "-n", "1", tag)
    if tagged != sha:
        raise GateError(f"{tag} points to {tagged}, not {sha}")
    print(git("show", "--stat", "--format=%H %s", tag))
    run(["git", "push", "origin", tag])  # the tag only; never --force
    ok(f"{tag} -> {sha}")


def monitor_gate(version: str, wait: bool) -> None:
    tag = f"v{version}"
    gate("21-22", f"Tag-triggered Wheels workflow for {tag}, including Publish to PyPI")
    sha = git("rev-list", "-n", "1", tag)
    result = wait_for_run("wheels.yml", sha, "push", tag, wait)
    jobs = run_jobs(result["id"])
    print(describe_jobs(jobs))
    names = [f"Wheels ({o})" for o in WHEEL_OSES] + ["Source distribution", "Publish to PyPI"]
    by_name = {j["name"]: j for j in jobs}
    bad = [n for n in names if by_name.get(n, {}).get("conclusion") != "success"]
    if bad or result["conclusion"] != "success":
        raise GateError(
            f"the release workflow failed ({', '.join(bad) or result['conclusion']}): "
            f"{result['html_url']}\n"
            "      Do not delete and re-push the tag of a version that reached PyPI; fix forward "
            "with a new patch version."
        )
    ok(result["html_url"])


def pypi_files(version: str) -> list[str]:
    url = f"https://pypi.org/pypi/{DISTRIBUTION}/{version}/json"
    with urllib.request.urlopen(url, timeout=30) as response:
        return [f["filename"] for f in json.load(response)["urls"]]


def verify_pypi_gate(version: str, retries: int = 20) -> None:
    gate("23", f"PyPI lists {DISTRIBUTION} {version} with the sdist and every wheel")
    files: list[str] = []
    for attempt in range(retries):
        try:
            files = pypi_files(version)
            break
        except urllib.error.URLError:
            if attempt == retries - 1:
                raise GateError(
                    f"https://pypi.org/project/{DISTRIBUTION}/{version}/ is not available"
                ) from None
            time.sleep(30)
    stem = DISTRIBUTION.replace("-", "_")
    missing = [] if f"{stem}-{version}.tar.gz" in files else ["sdist"]
    for label, pattern in WHEEL_PLATFORMS.items():
        for py in PYTHON_VERSIONS:
            if not any(
                f.startswith(f"{stem}-{version}-{py}-") and pattern.search(f) for f in files
            ):
                missing.append(f"{label} {py}")
    if missing:
        raise GateError("missing from PyPI: " + ", ".join(missing))
    ok(f"{len(files)} files at https://pypi.org/project/{DISTRIBUTION}/{version}/")

    gate(
        "23-24",
        "Fresh environment: pip install from PyPI, import, metadata, command; then clean up",
    )
    root = Path(tempfile.mkdtemp(prefix="dualmesh-pypi-test-"))
    try:
        python = _make_venv(root / "venv")
        for attempt in range(retries):
            proc = subprocess.run(
                [
                    str(python),
                    "-m",
                    "pip",
                    "install",
                    "--no-cache-dir",
                    f"{DISTRIBUTION}=={version}",
                ],
                text=True,
                capture_output=True,
                cwd=root,
            )
            if proc.returncode == 0:
                break
            if attempt == retries - 1:
                raise GateError(proc.stdout[-3000:] + proc.stderr[-3000:])
            note("the index has not caught up yet; retrying in 30 s")
            time.sleep(30)
        _check_installed(python, _venv_bin(root / "venv", CLI_NAME), version, root / "venv")
        ok(f"pip install {DISTRIBUTION}=={version} works")
    finally:
        shutil.rmtree(root, ignore_errors=True)
        if root.exists():
            raise GateError(f"could not delete the test environment {root}")
        note(f"test environment {root} deleted")


# -----------------------------------------------------------------------------
# Commands
# -----------------------------------------------------------------------------
def cmd_check(args) -> None:
    version = repository_gates(args.version, release=not args.dev)
    static_gates()
    local_gates(version, args)


def cmd_push(args) -> None:
    version = repository_gates(args.version, release=True)
    require_stamp(version)
    gate("16", f"Push {BRANCH} (fast-forward only, never --force)")
    if git("rev-parse", "HEAD") == git("rev-parse", f"origin/{BRANCH}"):
        ok("nothing to push")
        return
    run(["git", "push", "origin", BRANCH])
    ok("pushed")


def cmd_ci(args) -> None:
    ci_gate(git("rev-parse", "HEAD"), args.wait)


def cmd_dry_run(args) -> None:
    sha = git("rev-parse", "HEAD")
    if git("rev-parse", f"origin/{BRANCH}") != sha:
        raise GateError(f"HEAD is not origin/{BRANCH}; the dry run must build the pushed commit")
    wheels_dry_run_gate(sha, args.wait, dispatch=True)


def cmd_tag(args) -> None:
    version = repository_gates(args.version, release=True)
    static_gates()
    require_stamp(version)
    sha = git("rev-parse", "HEAD")
    ci_gate(sha, wait=False)
    wheels_dry_run_gate(sha, wait=False, dispatch=False)
    tag_gate(version, args.yes)


def cmd_monitor(args) -> None:
    monitor_gate(args.version or project_metadata()["version"], args.wait)


def cmd_verify_pypi(args) -> None:
    verify_pypi_gate(args.version or project_metadata()["version"])


def cmd_release(args) -> None:
    args.wait = True
    cmd_check(args)
    cmd_push(args)
    cmd_ci(args)
    cmd_dry_run(args)
    cmd_tag(args)
    cmd_monitor(args)
    cmd_verify_pypi(args)
    print(f"\nReleased {DISTRIBUTION} {args.version or project_metadata()['version']}.")


def cmd_status(args) -> None:
    meta = project_metadata()
    sha = git("rev-parse", "HEAD")
    print(f"distribution {meta['name']}  version {meta['version']}  HEAD {sha[:12]}")
    print(
        f"origin {git('remote', 'get-url', 'origin', check=False)}  "
        f"branch {git('rev-parse', '--abbrev-ref', 'HEAD')}"
    )
    print("latest tags: " + " ".join(git("tag", "--sort=-creatordate").splitlines()[:5]))
    for workflow, event in (("ci.yml", "push"), ("wheels.yml", "workflow_dispatch")):
        try:
            found = latest_run(workflow, sha, event)
        except GateError as error:
            print(f"{workflow}: {error}")
            continue
        state = (
            f"{found['status']} {found.get('conclusion') or ''} {found['html_url']}"
            if found
            else "no run"
        )
        print(f"{workflow} ({event}) on HEAD: {state}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    def add(name, func, help_text):
        p = sub.add_parser(name, help=help_text)
        p.set_defaults(func=func)
        p.add_argument("--version", help="the version being released (default: pyproject.toml)")
        return p

    p = add("check", cmd_check, "gates 1-15: repository state and every local build and test")
    p.add_argument(
        "--dev", action="store_true", help="allow an existing tag and no CHANGELOG section"
    )
    p.add_argument("--skip-cxx", action="store_true", help="skip the native C++ build and CTest")
    p.add_argument("--clang-format", help=f"path to clang-format {CLANG_FORMAT_MAJOR}")
    p.add_argument(
        "--keep", action="store_true", help="keep the build directories under build/release-check"
    )
    add("push", cmd_push, "gate 16: push main, fast-forward only")
    p = add("ci", cmd_ci, "gate 17: the CI workflow passed on HEAD")
    p.add_argument("--wait", action="store_true")
    p = add(
        "dry-run",
        cmd_dry_run,
        "gates 18-19: Wheels workflow_dispatch from main, publication skipped",
    )
    p.add_argument("--wait", action="store_true")
    p = add("tag", cmd_tag, "gate 20: recheck everything, then create and push the tag")
    p.add_argument("--yes", action="store_true", help="do not ask for the tag name to confirm")
    p = add("monitor", cmd_monitor, "gates 21-22: the tag build and the PyPI publication")
    p.add_argument("--wait", action="store_true")
    add("verify-pypi", cmd_verify_pypi, "gates 23-24: install from PyPI in a fresh environment")
    p = add("release", cmd_release, "every gate, in order, stopping at the first failure")
    p.add_argument("--skip-cxx", action="store_true")
    p.add_argument("--clang-format")
    p.add_argument("--keep", action="store_true")
    p.add_argument("--yes", action="store_true")
    p.set_defaults(dev=False)
    add("status", cmd_status, "read-only summary of the release state")

    args = parser.parse_args(argv)
    try:
        args.func(args)
    except GateError as failure:
        print(f"\nRELEASE GATE FAILED: {failure}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
