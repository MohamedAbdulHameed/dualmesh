# SPDX-License-Identifier: LGPL-2.1-or-later
"""Decide which copy of the package the tests import.

There are two ways to run the suite.  A developer builds in place, and CMake
writes the compiled extension into ``python/dualmesh``, so the source tree is a
complete package.  Continuous integration and ``pip install .`` build a wheel
instead, the extension goes into site-packages, and ``python/dualmesh`` is only
the pure-Python half of the package.

Putting ``python/`` unconditionally on the path, as an earlier version of the
test configuration did, works in the first case and breaks the second: the
source tree shadows the installed package and the import of ``_core`` fails.
So the source tree is used only when it actually contains a built extension.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SOURCE = Path(__file__).resolve().parents[2] / "python"
_PACKAGE = _SOURCE / "dualmesh"


def _has_built_extension(package: Path) -> bool:
    return any(package.glob("_core*.so")) or any(package.glob("_core*.pyd"))


if _has_built_extension(_PACKAGE) and str(_SOURCE) not in sys.path:
    sys.path.insert(0, str(_SOURCE))
