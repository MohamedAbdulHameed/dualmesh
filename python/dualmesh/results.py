"""The fields and the postprocessor table of a solved problem."""

import numpy

from dualmesh.input.schema import suggestion
from dualmesh.input.units import input_error


class results:
    """The accepted times, the final fields and the postprocessor table of one solve.

    Each field array holds the owned nodes of this rank, so under MPI every rank holds its
    part. The postprocessor table is the same on every rank. It has one row per output time:
    every interval-th accepted time step and the end of the run, or every accepted step when
    the input has no output section. A steady run has one row, at time 0 s.
    """

    def __init__(
        self,
        times,
        fields: dict,
        coordinates: dict,
        output_times,
        postprocessors: dict,
        units: dict,
    ):
        self.times: list[float] = list(times)
        self.output_times: list[float] = list(output_times)
        self._fields = dict(fields)
        self._coordinates = dict(coordinates)
        self._postprocessors = {
            name: numpy.asarray(values) for name, values in postprocessors.items()
        }
        self._units = dict(units)

    def field(self, name: str):
        """The values of a field, an array of shape (owned nodes, components)."""
        return self._fields[self._known(name, self._fields, "field")]

    def coordinates(self, name: str):
        """The node coordinates of a field in m, an array of shape (owned nodes, dimension)."""
        return self._coordinates[self._known(name, self._fields, "field")]

    def field_names(self) -> list[str]:
        return list(self._fields)

    def postprocessor(self, name: str):
        """The values of a postprocessor in its SI unit, an array with one entry per output time."""
        return self._postprocessors[self._known(name, self._postprocessors, "postprocessor")]

    def unit(self, name: str) -> str:
        """The SI unit of a postprocessor, for example "K"."""
        return self._units[self._known(name, self._postprocessors, "postprocessor")]

    def postprocessor_names(self) -> list[str]:
        return list(self._postprocessors)

    def table(self) -> dict:
        """The postprocessor table as columns, keyed by the headers of the CSV file.

        The first column is "time [s]". Each postprocessor follows as "<name> [<unit>]".
        """
        columns = {"time [s]": numpy.asarray(self.output_times)}
        for name, values in self._postprocessors.items():
            columns[f"{name} [{self._units[name]}]"] = values
        return columns

    @staticmethod
    def _known(name: str, names, what: str) -> str:
        if name not in names:
            raise input_error(f"the results hold no {what} '{name}'." + suggestion(name, names))
        return name
