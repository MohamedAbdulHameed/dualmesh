# SPDX-License-Identifier: LGPL-2.1-or-later
"""Shared figure style and result cache of the benchmark scripts.

Every benchmark figure of the documentation has one axes, drawn at
6.4 x 4.2 inches and saved at 200 dpi, so that the documentation theme
shows it at about its natural size with legible text.  The scripts save the
curves they compute in a compressed ``<script>_cache.npz`` beside
themselves, and ``--plot-only`` redraws the figures from it.

The scripts import this module with::

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import plotstyle
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Colours of the reference categorical palette: dualmesh in slot 1, a second
# dualmesh series in slot 2, measurements and closed form values in slot 3,
# other codes in a recessive grey.
DUALMESH = "#2a78d6"
SECOND = "#1baf7a"
MEASURED = "#eb6834"
INK = "#0b0b0b"
MUTED = "#52514e"
REFERENCE_GREY = "#a3a29c"
GRID = "#e4e3df"
SPINE = "#c3c2b7"

FIGSIZE = (6.4, 4.2)
DPI = 200
TITLE_SIZE = 13
LABEL_SIZE = 12
TICK_SIZE = 11
LEGEND_SIZE = 10.5


def new_figure(figsize=FIGSIZE):
    """One figure with one axes."""
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


def style(ax, xlabel="", ylabel="", title=None):
    """Axis labels, title, light grid, and no top or right spine."""
    if title:
        ax.set_title(title, loc="left", fontsize=TITLE_SIZE, color=INK)
    ax.set_xlabel(xlabel, fontsize=LABEL_SIZE, color=MUTED)
    ax.set_ylabel(ylabel, fontsize=LABEL_SIZE, color=MUTED)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(SPINE)
    ax.tick_params(colors=MUTED, labelsize=TICK_SIZE)


def legend(ax, loc="best", ncol=1, **kwargs):
    """A frameless legend inside the axes."""
    return ax.legend(frameon=False, fontsize=LEGEND_SIZE, labelcolor=INK, loc=loc, ncol=ncol, **kwargs)


def legend_below(ax, ncol=2, offset=0.17, **kwargs):
    """A frameless legend centred below the axes, under the x-axis label."""
    return ax.legend(frameon=False, fontsize=LEGEND_SIZE, labelcolor=INK, loc="upper center", bbox_to_anchor=(0.5, -offset), ncol=ncol, **kwargs)


def save(fig, path):
    """Save at 200 dpi with a tight bounding box and close the figure."""
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {path}")


def cache_path(script):
    """``<script name>_cache.npz`` beside the script."""
    script = Path(script).resolve()
    return script.with_name(script.stem + "_cache.npz")


def parse_args(argv, default_out):
    """The output directory (first positional argument or ``default_out``)
    and whether ``--plot-only`` was given."""
    positional = [a for a in argv if not a.startswith("--")]
    unknown = [a for a in argv if a.startswith("--") and a != "--plot-only"]
    if unknown:
        raise SystemExit(f"unknown option(s): {' '.join(unknown)}")
    out = Path(positional[0]) if positional else Path(default_out)
    out.mkdir(parents=True, exist_ok=True)
    return out, "--plot-only" in argv


def _escape(key):
    return str(key).replace("%", "%25").replace("/", "%2F")


def _unescape(key):
    return key.replace("%2F", "/").replace("%25", "%")


def _flatten(data, prefix=""):
    flat = {}
    for key, value in data.items():
        name = f"{prefix}{_escape(key)}"
        if isinstance(value, dict) and not value:
            flat[name + "/"] = np.zeros(0)  # an empty dict
        elif isinstance(value, dict):
            flat.update(_flatten(value, name + "/"))
        else:
            flat[name] = np.asarray(value)
    return flat


def save_cache(path, data):
    """Store a (nested) dict of arrays and numbers in a compressed .npz.
    Nested keys are joined with '/' (a '/' inside a key is escaped) and the
    order of the keys is kept."""
    np.savez_compressed(path, **_flatten(data))
    print(f"wrote {path}")


def load_cache(path):
    """The nested dict written by ``save_cache``, in the same key order.
    Zero-dimensional arrays come back as NumPy scalars."""
    path = Path(path)
    if not path.exists():
        sys.exit(f"{path} does not exist: run the script without --plot-only first")
    out: dict = {}
    with np.load(path, allow_pickle=False) as z:
        for name in z.files:
            *parents, leaf = (_unescape(k) for k in name.split("/"))
            node = out
            for p in parents:
                node = node.setdefault(p, {})
            if name.endswith("/"):
                continue  # an empty dict
            value = z[name]
            node[leaf] = value[()] if value.ndim == 0 else value
    return out
