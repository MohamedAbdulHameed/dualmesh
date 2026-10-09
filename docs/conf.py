# SPDX-License-Identifier: LGPL-2.1-or-later
"""Sphinx configuration for the dualmesh documentation."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# The source tree is a complete package only after an in-place build, which
# writes the compiled extension into python/dualmesh.  Otherwise (Read the
# Docs, or a pip-installed checkout) it would shadow the installed package and
# the import of the extension would fail, so it goes on the path only when the
# extension is there (the same rule as tests/python/conftest.py).
_SOURCE_PACKAGE = ROOT / "python" / "dualmesh"
if any(_SOURCE_PACKAGE.glob("_core*.so")) or any(_SOURCE_PACKAGE.glob("_core*.pyd")):
    sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

project = "dualmesh"
copyright = "2026, Mohamed AbdulHameed"
author = "Mohamed AbdulHameed"
# The version has one source, pyproject.toml.
release = re.search(r'^version\s*=\s*"([^"]+)"', (ROOT / "pyproject.toml").read_text(encoding="utf-8"), re.M).group(1)
version = ".".join(release.split(".")[:2])

extensions = ["sphinx.ext.autodoc", "sphinx.ext.napoleon", "sphinx.ext.viewcode", "sphinx.ext.mathjax", "sphinx.ext.intersphinx", "myst_parser"]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}

html_theme = "furo"
html_static_path = ["_static"]
html_title = "dualmesh"
# "Edit this page" and "view source" links to the repository.
html_theme_options = {"source_repository": "https://github.com/MohamedAbdulHameed/dualmesh/", "source_branch": "main", "source_directory": "docs/"}

# Figures are numbered, so that the text can refer to them as "Figure n".
numfig = True
numfig_format = {"figure": "Figure %s", "table": "Table %s", "code-block": "Listing %s", "section": "Section %s"}

autodoc_member_order = "bysource"
autodoc_typehints = "description"
napoleon_google_docstring = True
napoleon_numpy_docstring = True

intersphinx_mapping = {"python": ("https://docs.python.org/3", None), "numpy": ("https://numpy.org/doc/stable", None)}

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
    autodoc_mock_imports = ["dualmesh._core", "numpy", "meshio"]


def _write_generated_pages(app):  # pragma: no cover - documentation build helper
    """Generate the syntax reference from the object registry."""
    from _generate_syntax import write_syntax_reference

    write_syntax_reference(Path(app.srcdir), _HAVE_DUALMESH)


def setup(app):  # pragma: no cover - documentation build helper
    app.connect("builder-inited", _write_generated_pages)
    return {"parallel_read_safe": True}
