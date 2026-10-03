# SPDX-License-Identifier: LGPL-2.1-or-later
"""Console reports of a rod calculation.

Before a run, :func:`input_report` lists every input group with every value,
its unit, and whether the user gave it or it is a default (``(default)``), so
that no choice made on the user's behalf goes unannounced.  During the run,
:func:`step_header` and :func:`step_row` print one aligned line per output
time.  The reports are for reading; :meth:`RodResult.write_csv` writes the
same numbers for post-processing.
"""

from __future__ import annotations

import dataclasses

from ..console import RULE, header, table
from ..parameters import describe


def _group(title: str, spec, indent: int = 0) -> list[str]:
    pad = " " * indent
    lines = [f"{pad}{title}  ({type(spec).__name__})"]
    lines.append(table(describe(spec), ["parameter", "value", "unit", "source"], indent=indent + 4))
    for f in dataclasses.fields(spec):
        value = getattr(spec, f.name)
        if dataclasses.is_dataclass(value):
            lines.extend(_group(f"{title}.{f.name}", value, indent + 4))
    return lines


def input_report(rod) -> str:
    """Every input group of a :class:`FuelRod`, with derived quantities."""
    lines = [header("dualmesh fuel rod")]
    groups = [("geometry", rod.geometry), ("fuel", rod.fuel), ("cladding", rod.cladding), ("fill_gas", rod.fill_gas), ("coolant", rod.coolant), ("power_history", rod.power_history), ("models", rod.models), ("numerics", rod.numerics), ("output", rod.output)]
    for title, spec in groups:
        lines.extend(_group(title, spec))
    derived = rod.derived_quantities()
    lines.append("derived")
    lines.append(table([[n, v, u] for n, v, u in derived], ["quantity", "value", "unit"], indent=4))
    lines.append(RULE)
    return "\n".join(lines)


STEP_COLUMNS = (("time", "d"), ("q'", "kW/m"), ("burnup", None), ("T_centerline max", "K"), ("T_clad_outer max", "K"), ("gap min", "um"), ("pressure", "MPa"), ("FGR", "%"), ("Newton", ""))


def _step_headers(burnup_unit: str) -> list[str]:
    return [f"{name} ({burnup_unit if unit is None else unit})" if unit != "" else name for name, unit in STEP_COLUMNS]


def _widths(burnup_unit: str) -> list[int]:
    return [max(len(h), 10) for h in _step_headers(burnup_unit)]


def step_header(burnup_unit: str) -> str:
    """The header of the step table, printed once before the first row."""
    widths = _widths(burnup_unit)
    names = "  ".join(h.rjust(w) for h, w in zip(_step_headers(burnup_unit), widths))
    rule = "  ".join("-" * w for w in widths)
    return f"  {names}\n  {rule}"


def step_row(result, k: int) -> str:
    """Row ``k`` of the step table: the state at output time ``k``."""
    rod, axial = result.rod, result.axial
    values = [rod["time_days"][k], rod["rod_average_linear_heat_rate"][k] / 1e3, rod["rod_average_burnup"][k], rod["max_fuel_centerline_temperature"][k], float(max(axial["clad_outer_temperature"][k])), rod["min_gap_width"][k] * 1e6, rod["gas_pressure"][k] / 1e6, 100.0 * rod["fission_gas_release"][k]]
    cells = [f"{v:.6g}" for v in values] + [str(int(rod["nonlinear_iterations"][k]))]
    return "  " + "  ".join(c.rjust(w) for c, w in zip(cells, _widths(result.burnup_unit)))
