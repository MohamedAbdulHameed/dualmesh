# SPDX-License-Identifier: LGPL-2.1-or-later
"""Sphinx configuration for the dualmesh documentation."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

project = "dualmesh"
copyright = "2026, the dualmesh developers"
author = "the dualmesh developers"
release = "0.1.0"
version = "0.1"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}

html_theme = "furo"
html_static_path = ["_static"]
html_title = "dualmesh"

autodoc_member_order = "bysource"
autodoc_typehints = "description"
napoleon_google_docstring = True
napoleon_numpy_docstring = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}

myst_enable_extensions = ["dollarmath", "amsmath", "colon_fence"]

# The reference list is a bibliography; the entries need not be cited inline.
suppress_warnings = ["ref.citation"]

# The compiled extension module may not be available when the documentation is
# built (for example on a machine without a compiler).  Autodoc then falls back
# to mocking it, so that the prose still builds.
try:  # pragma: no cover - documentation build helper
    import dualmesh  # noqa: F401

    _HAVE_DUALMESH = True
except Exception:  # pragma: no cover
    _HAVE_DUALMESH = False
    autodoc_mock_imports = ["dualmesh._core", "numpy", "meshio", "yaml"]


def _write_object_reference(app):  # pragma: no cover - documentation build helper
    """Generate the object (syntax) reference from the registry."""
    target = Path(app.srcdir) / "objects.rst"
    if not _HAVE_DUALMESH:
        if not target.exists():
            target.write_text(
                "Object reference\n================\n\n"
                "The object reference is generated from the compiled library, which was "
                "not available when this documentation was built. Run ``dualmesh list`` "
                "and ``dualmesh describe <type>`` locally instead.\n"
            )
        return
    import dualmesh as dm

    lines = [
        "Object reference",
        "================",
        "",
        "Every object below is created by name, either from Python::",
        "",
        "    problem.add_kernel(\"HeatConduction\", variable=\"temperature\",",
        "                       thermal_conductivity=20.0)",
        "",
        "or from an input file (see :doc:`input_files`). The same list is available at",
        "the command line with ``dualmesh list`` and ``dualmesh describe <type>``.",
        "",
    ]
    modules: dict[str, list[str]] = {}
    for name in dm.registered_types():
        modules.setdefault(dm.object_module(name), []).append(name)
    titles = {
        "framework": "Framework",
        "heat_transfer": "Heat transfer",
        "solid_mechanics": "Solid mechanics",
        "structural": "Beams and plates",
        "fluids": "Viscous incompressible flows",
    }
    for module in sorted(modules, key=lambda m: list(titles).index(m) if m in titles else 99):
        title = titles.get(module, module)
        lines += [title, "-" * len(title), ""]
        for name in sorted(modules[module]):
            lines += [name, "^" * len(name), "", "*Category:* " + dm.object_category(name), ""]
            body = dm.describe_object(name).strip("\n")
            lines += ["::", ""]
            lines += ["    " + row for row in body.splitlines()]
            lines += [""]
    target.write_text("\n".join(lines) + "\n")


def setup(app):  # pragma: no cover - documentation build helper
    app.connect("builder-inited", _write_object_reference)
    return {"parallel_read_safe": True}
