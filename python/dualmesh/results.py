"""The fields, the postprocessor table, the lines and the eigenvalues of a solved problem."""

import numpy

from dualmesh.console import suggestion
from dualmesh.input.units import input_error


class results:
    """The accepted times, the final fields and the postprocessor table of one solve.

    Each field array holds every node of the whole mesh, gathered on every rank of a
    distributed run. The postprocessor table is the same on every rank. It has one row for each
    evaluation of the postprocessors: after a steady solve, at time 0 s, and at the start and
    after every accepted time step of a transient run. A line postprocessor has a table of its
    own, with the rows of every output time. An eigenvalue study holds its eigenvalues, and its
    fields are those of the first mode.
    """

    def __init__(self, times, fields: dict, coordinates: dict, table, lines=None, study=None):
        self.times: list[float] = list(times)
        self._fields = dict(fields)
        self._coordinates = dict(coordinates)
        self._table = table
        self._lines = dict(lines or {})
        self._study = study

    @property
    def eigenvalues(self) -> numpy.ndarray:
        """The eigenvalues of an eigenvalue study, closest to its near first: in 1/s when the
        problem has a time derivative, and numbers when the source of a model defines them.
        Every field varies in time as u e^(-lambda t), so a vibration has the imaginary
        eigenvalues lambda = i omega, with omega the angular frequency in rad/s."""
        if self._study is None:
            raise input_error("the results of a steady or transient run hold no eigenvalues.")
        return numpy.asarray(self._study.eigenvalues)

    @property
    def frequencies(self) -> numpy.ndarray:
        """The natural frequencies f = |Im lambda| / (2 pi) in Hz of a vibration, a problem whose
        only time term is a second time derivative."""
        if self._study is None or not self._study.vibration:
            raise input_error(
                "natural frequencies belong to a vibration, an eigenvalue study whose only time "
                "term is a second time derivative, and these results are not one."
            )
        return self._study.frequencies

    def line(self, name: str):
        """The table of a line postprocessor, whose columns are those of its CSV file: the time
        in a transient run, the distance along the line, the coordinates and the field."""
        return self._lines[self._known(name, self._lines, "line")]

    @property
    def output_times(self) -> list[float]:
        """The times of the rows of the postprocessor table, in s."""
        return list(self._table["time"])

    def field(self, name: str):
        """The values of a field, an array of shape (nodes, components)."""
        return self._fields[self._known(name, self._fields, "field")]

    def coordinates(self, name: str):
        """The node coordinates of a field in m, an array of shape (nodes, dimension)."""
        return self._coordinates[self._known(name, self._fields, "field")]

    def field_names(self) -> list[str]:
        """The names of the fields of the result, such as temperature."""
        return list(self._fields)

    def postprocessor(self, name: str):
        """The values of a postprocessor in its SI unit, an array with one entry per row of the table."""
        return self._table[self._known(name, self.postprocessor_names(), "postprocessor")]

    def unit(self, name: str) -> str:
        """The SI unit of a postprocessor, for example "K"."""
        return self._table.unit(self._known(name, self.postprocessor_names(), "postprocessor"))

    def postprocessor_names(self) -> list[str]:
        """The names of the post-processors of the result."""
        return self._table.names[1:]

    def table(self):
        """The postprocessor table, a :class:`~dualmesh.Table` whose headers are those of the CSV
        file: "time [s]", then "<name> [<unit>]" for each postprocessor."""
        return self._table

    @staticmethod
    def _known(name: str, names, what: str) -> str:
        if name not in names:
            raise input_error(f"the results hold no {what} '{name}'." + suggestion(name, names))
        return name
