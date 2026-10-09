# SPDX-License-Identifier: LGPL-2.1-or-later
"""Tables: the one form of every tabular result.

Every table of every result (the history of a transient, the regions of a
reactor core, the Newton iterations of a solve, the runs of an uncertainty
study) is a :class:`Table`: named columns, each with its unit, and rows of
numbers or texts.  A table prints itself in aligned columns, writes a CSV
file with one header line in which every column is named with its unit, for
instance ``temperature (K)``, so that a spreadsheet, pandas or gnuplot reads
it without parsing, and gives its columns as arrays::

    table = result.tables["history"]
    print(table)
    table.write_csv("history.csv")
    temperature = table["temperature"]  # an array, by column name
    frame = table.to_pandas()  # when pandas is installed
"""

from __future__ import annotations

import csv
import math
from collections.abc import Sequence

import numpy as np

__all__ = ["Table", "column_header"]


def column_header(name: str, unit: str) -> str:
    """The header of a column: the name and, when there is one, the unit in
    parentheses, for example ``temperature (K)``."""
    return f"{name} ({unit})" if unit and unit not in ("-", "1") else name


def _is_number(value) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, (bool, np.bool_))


class Table:
    """A table of named columns with units.

    ``columns`` are the column names, ``units`` their units (an empty text
    for a quantity without a unit, e.g., a count, a name or a ratio), and
    ``rows`` the rows.  ``title`` heads the printed table."""

    def __init__(self, columns: Sequence[str], units: Sequence[str] | None = None, rows: Sequence[Sequence] = (), title: str = "", digits: int = 6):
        self.names = [str(c) for c in columns]
        # A dimensionless quantity has no unit in the header ("-" is dropped).
        self.units = [("" if str(u) in ("-", "1") else str(u)) for u in units] if units is not None else [""] * len(self.names)
        if len(self.units) != len(self.names):
            raise ValueError(f"Table '{title}': {len(self.names)} columns and {len(self.units)} units.")
        self.rows = [list(r) for r in rows]
        for row in self.rows:
            if len(row) != len(self.names):
                raise ValueError(f"Table '{title}': a row has {len(row)} values for {len(self.names)} columns.")
        self.title = title
        self.digits = int(digits)

    # ---- access -------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, name: str) -> np.ndarray:
        """The values of one column, by name, as an array."""
        if name not in self.names:
            import difflib

            close = difflib.get_close_matches(name, self.names, n=1)
            hint = f" Did you mean '{close[0]}'?" if close else ""
            raise KeyError(f"Table '{self.title}' has no column '{name}'.{hint} Columns: {', '.join(self.names)}")
        j = self.names.index(name)
        values = [row[j] for row in self.rows]
        return np.asarray(values, dtype=float) if all(_is_number(v) for v in values) else np.asarray(values, dtype=object)

    @property
    def columns(self) -> dict:
        """Column name -> array of its values."""
        return {name: self[name] for name in self.names}

    def headers(self) -> list[str]:
        """The column names with their units, as in the CSV header."""
        return [column_header(n, u) for n, u in zip(self.names, self.units)]

    def unit(self, name: str) -> str:
        return self.units[self.names.index(name)]

    def to_pandas(self):
        """A pandas DataFrame whose column names carry the units."""
        import pandas

        return pandas.DataFrame(self.rows, columns=self.headers())

    def to_dict(self) -> dict:
        return {"title": self.title, "columns": self.names, "units": self.units, "rows": self.rows}

    # ---- output -------------------------------------------------------------
    def _cell(self, value) -> str:
        if isinstance(value, (bool, np.bool_)):
            return "yes" if value else "no"
        if _is_number(value):
            value = float(value) if not isinstance(value, (int, np.integer)) else int(value)
            if isinstance(value, int):
                return str(value)
            if not math.isfinite(value):
                return "-"
            return f"{value:.{self.digits}g}"
        return str(value)

    def format(self, indent: int = 2) -> str:
        """The table in aligned columns: numbers right aligned, texts left
        aligned, a rule under the header."""
        headers = self.headers()
        text = [[self._cell(v) for v in row] for row in self.rows]
        widths = [max([len(h)] + [len(r[j]) for r in text]) for j, h in enumerate(headers)]
        numeric = [bool(self.rows) and all(_is_number(row[j]) or row[j] is None for row in self.rows) for j in range(len(headers))]
        pad = " " * indent

        def line(cells):
            return (pad + "  ".join(c.rjust(w) if numeric[j] else c.ljust(w) for j, (c, w) in enumerate(zip(cells, widths)))).rstrip()

        out = [self.title] if self.title else []
        out += [line(headers), pad + "  ".join("-" * w for w in widths)]
        out += [line(r) for r in text]
        return "\n".join(out)

    def __str__(self) -> str:
        return self.format()

    def __repr__(self) -> str:
        return f"Table('{self.title}', {len(self.rows)} rows, columns: {', '.join(self.headers())})"

    def write_csv(self, path) -> None:
        """Write the table: one header line of names with units, then one line
        per row, numbers at full precision (a non-finite number as an empty
        field)."""

        def cell(value):
            if isinstance(value, (bool, np.bool_)):
                return int(value)
            if _is_number(value):
                if isinstance(value, (int, np.integer)):
                    return int(value)
                return repr(float(value)) if math.isfinite(value) else ""
            return value

        with open(path, "w", newline="") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(self.headers())
            for row in self.rows:
                writer.writerow([cell(v) for v in row])


class ResultTables:
    """The part of the result protocol that every result shares: its tables
    in ``tables`` (name -> :class:`Table`, the main table first), and
    ``write_csv(path, table=None)``, which writes the main table or the one
    named."""

    tables: dict

    def write_csv(self, path, table: str | None = None) -> None:
        """Write one table of the result to a CSV file: the main table by
        default, or the table named (``result.tables`` lists them)."""
        tables = self.tables
        if not tables:
            raise ValueError(f"{type(self).__name__}: the result has no table to write.")
        name = table if table is not None else next(iter(tables))
        if name not in tables:
            raise KeyError(f"{type(self).__name__}: no table '{name}'. Tables: {', '.join(tables)}")
        tables[name].write_csv(path)
