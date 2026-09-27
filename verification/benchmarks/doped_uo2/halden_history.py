# SPDX-License-Identifier: LGPL-2.1-or-later
"""Shutdowns of the Halden power histories (IFA-677.1, IFA-716.1).

The power charts of CASL-U-2019-1870-000 Rev. 0 draw each shutdown as a
near-vertical line down and up within about a day, and the thermocouple
charts show the centre temperature falling to the coolant temperature on
the same days.  The power history files hold the operating power, the
shutdown files the days of the shutdowns, and ``insert_shutdowns`` puts the
shutdowns into the history with the shape ``SHUTDOWN``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

DAY = 86400.0

#: The shape of a shutdown (hours from the day of the shutdown line): the
#: power falls to zero in 15 minutes, stays there for 15 minutes and returns
#: in 30 minutes.  The report gives only the day.  The fuel cools to the
#: coolant temperature within minutes, and the release by micro-cracking
#: depends on the temperature drop, not on the duration; the short shape
#: removes about 0.2 % of the energy of the history.
SHUTDOWN = (-0.25, 0.0, 0.25, 0.75)


def shutdown_days(path: Path) -> np.ndarray:
    lines = Path(path).read_text().splitlines()
    return np.array([float(v) for v in lines if v[:1].isdigit()])


def insert_shutdowns(t, q, days, shape=SHUTDOWN, floor=1.0):
    """The history (t in s, q in W/m) with a shutdown to ``floor`` W/m at each
    day of ``days``: the points within the shutdown are replaced by the four
    points of ``shape``.  Shutdowns at the ends of the history are skipped."""
    t, q = np.asarray(t, dtype=float), np.asarray(q, dtype=float)
    for day in days:
        a, s, e, b = (day * DAY + h * 3600.0 for h in shape)
        if a <= t[0] or b >= t[-1]:
            continue
        qa, qb = np.interp([a, b], t, q)
        keep = (t < a) | (t > b)
        t = np.concatenate((t[keep], [a, s, e, b]))
        q = np.concatenate((q[keep], [qa, floor, floor, qb]))
        order = np.argsort(t)
        t, q = t[order], q[order]
    return t, q
