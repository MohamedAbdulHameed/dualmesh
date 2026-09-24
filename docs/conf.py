# SPDX-License-Identifier: LGPL-2.1-or-later
"""Sphinx configuration for the dualmesh documentation."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

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
# "Edit this page" and "view source" links to the repository.
html_theme_options = {
    "source_repository": "https://github.com/MohamedAbdulHameed/dualmesh/",
    "source_branch": "main",
    "source_directory": "docs/",
}

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


def _write_generated_pages(app):  # pragma: no cover - documentation build helper
    """Generate the syntax reference from the object registry."""
    from _generate_syntax import write_syntax_reference

    write_syntax_reference(Path(app.srcdir), _HAVE_DUALMESH)


def setup(app):  # pragma: no cover - documentation build helper
    app.connect("builder-inited", _write_generated_pages)
    return {"parallel_read_safe": True}
