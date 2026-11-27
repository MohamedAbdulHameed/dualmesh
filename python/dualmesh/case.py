"""The input of a simulation, built in Python as a YAML or TOML file builds it, and its solve.

The Python calls, a YAML file and a TOML file all build one input tree. A problem holds that
tree as the user wrote it, with units in text. `problem.solve` validates the tree, lowers it
onto the engine, solves it and returns the results.
"""

import copy
import os
from pathlib import Path
from typing import NamedTuple

import numpy

from dualmesh import _core
from dualmesh.adaptivity import (
    mark_by_error_fraction,
    solve_transient_with_adaptive_refinement,
    solve_with_adaptive_refinement,
)
from dualmesh.input import schema
from dualmesh.input.build import (
    add_postprocessors,
    build_mesh,
    build_problem,
    line_samplers,
    transient_options,
)
from dualmesh.input.read import read as read_file
from dualmesh.input.read import write_TOML, write_YAML
from dualmesh.output import Output, Series
from dualmesh.solution import results


def table(file: str | None = None, **columns) -> dict:
    """A table value of a property, linear between its rows.

    table("zircaloy_conductivity.csv") reads a CSV file whose header names each column with
    its unit. table(temperature=("K", [300, 600]), Young_modulus=("GPa", [96, 83])) gives the
    columns inline, each as (unit, values). The column named after the property holds its
    values, and the other column is its input: a field, the time or a coordinate.
    """
    if (file is None) == (not columns):
        raise _core.InputError("table: give either a file or the columns, exactly one of the two.")
    if file is not None:
        return {"table": {"file": str(file)}}
    entries = {}
    for name, column in columns.items():
        if not isinstance(column, (tuple, list)) or len(column) != 2:
            raise _core.InputError(
                f"table: the column {name} takes (unit, values), got {column!r}."
            )
        unit, values = column
        entries[name] = {"unit": unit, "values": list(values)}
    return {"table": entries}


class problem:
    """One simulation: a mesh, physics, properties, conditions and an analysis."""

    def __init__(self, mesh: dict, method: str | None = None):
        self._start({}, os.getcwd())
        sections = {"mesh": copy.deepcopy(mesh)}
        if method is not None:  # The tree holds only what the user chose
            sections["method"] = method
        self._set(**sections)

    # Building the tree

    def physics(self, name: str, **options) -> None:
        """Adds a physics, such as heat_conduction, with its own options and an optional method."""
        self._merge("physics", {name: options})

    def model(self, name: str, **entry) -> None:
        """Adds a model of your own: unknown, states, storage, flux, source, expressions, flux_name
        and method, the keys of the models section. Give the model before the properties whose
        formulas read its unknown or its states. In an eigenvalue study the source may read the
        symbol eigenvalue, such as the 1/k of a fission source."""
        self._merge("models", {name: entry})

    def properties(self, **values) -> None:
        """Sets properties by standard name: a constant, a formula, a table, or a mapping of regions."""
        self._merge("properties", values)

    def initial_conditions(self, **values) -> None:
        """The value of each field at the start, by standard name, for example temperature="293 K"."""
        self._merge("initial_conditions", values)

    def boundary_conditions(self, boundary: str, condition: str | None = None, **values) -> None:
        """The conditions on one named boundary, keyed by the quantity each one sets, or the word
        "symmetry" alone, which makes the boundary a plane of symmetry, for example
        boundary_conditions("left", "symmetry")."""
        given = self._tree.get("boundary_conditions", {}).get(boundary, {})
        if condition is not None or isinstance(given, str):
            if values or (condition is None and given == schema.symmetry):
                raise _core.InputError(
                    f"boundary_conditions.{boundary}: the word {schema.symmetry} takes the place "
                    "of the conditions of a boundary, so it takes no other condition."
                )
            self._merge("boundary_conditions", {boundary: condition})
            return
        self._merge("boundary_conditions", {boundary: {**given, **values}})

    def gravity(self, components: list) -> None:
        """The gravitational acceleration, one quantity per direction of space, for example
        ["0 m/s^2", "-9.81 m/s^2"]. Solids and flows take its force rho g."""
        self._set(gravity=list(components))

    def time(self, **values) -> None:
        """Makes the analysis transient: end_time, initial_time_step and step_control. Without
        step_control the step follows the estimated error of each step, with the relative
        tolerance 1e-3."""
        self._set(time=dict(values))

    def eigenvalues(self, **values) -> None:
        """Makes the problem an eigenvalue study: count, the number of eigenvalues, default 1,
        and near, the value that they are closest to, default 0. Every field varies in time as
        u e^(-lambda t), so the eigenvalue lambda is in 1/s when the problem has a time
        derivative, such as near="0 1/s". A problem without one takes lambda from the symbol
        eigenvalue in the source of a model, and lambda is a number, such as 1/k of a reactor."""
        self._set(eigenvalues=dict(values))

    def mesh_adaptivity(self, **values) -> None:
        """Adapts the mesh to the error indicator of a field: field, refine_fraction,
        coarsen_fraction, maximum_refinement_level, and in a transient run interval, the number
        of accepted steps between two adaptations."""
        self._set(mesh_adaptivity=dict(values))

    def postprocessor(self, name: str, **definition) -> None:
        """Adds a named output: a scalar, for example maximum="temperature",
        point_value="von_Mises_stress" with point=["60 mm", "6 mm", "0 mm"], or the force through
        a boundary, boundary_flux="traction_y" with boundary="jaws", or the values along a line,
        for example line="velocity_x", from_=["0.5 m", "0 m"], to=["0.5 m", "1 m"], points=101.
        The key from is a word of Python, so the call writes it from_."""
        if "from_" in definition:
            definition["from"] = definition.pop("from_")
        self._merge("postprocessors", {name: definition})

    def output(self, file: str, fields: list[str], interval: int | None = None) -> None:
        """Writes the fields: once in a steady run, and at the start, every interval-th accepted
        time step and the end of a transient run, with interval 1 when it is left out. A steady
        run takes no interval.

        A steady run writes <file>.vtu, and a transient run <file>_<index>.vtu with a <file>.pvd
        collection, which ParaView opens as one time series. Both write <file>.csv with the
        postprocessors, a unit in each header, and <file>_<name>.csv for each line
        postprocessor. An eigenvalue study writes each mode as <file>_mode_<number>.vtu.
        """
        values = {"file": file, "fields": list(fields)}
        if interval is not None:
            values["interval"] = interval
        self._set(output=values)

    def solver(self, **values) -> None:
        """The relative tolerance of the nonlinear solve and PETSc options without the leading dash."""
        self._merge("solver", values)

    def _start(self, tree: dict, directory: str) -> None:
        """Holds a tree and the folder that its relative paths resolve against."""
        self._tree = tree
        self._validated: dict | None = None  # The validated tree, kept until a section changes
        self._directory = directory  # Relative mesh and table paths resolve against it

    def _merge(self, section: str, values: dict) -> None:
        self._set(**{section: {**self._tree.get(section, {}), **copy.deepcopy(values)}})

    def _set(self, **sections) -> None:
        """Sets sections after each validates, so a wrong call fails at its own line and changes nothing."""
        candidate = {**self._tree, **sections}
        for key in sections:
            schema.validate_section(candidate, key, self._directory)
        self._tree = candidate
        self._validated = None

    def _valid(self) -> dict:
        """The validated tree, which the schema makes once for each state of the tree."""
        if self._validated is None:
            self._validated = schema.validate(self._tree, self._directory)
        return self._validated

    # The tree and its files

    def to_tree(self) -> dict:
        """The input tree, with values as the user wrote them, in the order of the sections."""
        return {
            key: copy.deepcopy(self._tree[key]) for key in schema.section_order if key in self._tree
        }

    def to_YAML(self, path: str) -> None:
        """Writes the input to a YAML file, with relative paths that resolve against its folder."""
        write_YAML(self._tree_for_file(path), path)

    def to_TOML(self, path: str) -> None:
        """Writes the input to a TOML file, with relative paths that resolve against its folder."""
        write_TOML(self._tree_for_file(path), path)

    def _tree_for_file(self, path: str) -> dict:
        """The validated input tree, with each relative path rewritten relative to the folder of path."""
        self._valid()  # Refuses an invalid tree before it is written
        tree = self.to_tree()
        folder = Path(path).resolve().parent

        def relative(file: str) -> str:
            if os.path.isabs(file):
                return file
            return os.path.relpath(os.path.join(self._directory, file), folder)

        return with_paths(tree, relative)

    @classmethod
    def from_file(cls, path: str) -> "problem":
        """Reads a problem from a YAML or TOML file. Relative paths in it resolve against its folder."""
        case = cls.__new__(cls)
        case._start(read_file(path), str(Path(path).resolve().parent))
        case._valid()  # Refuses an invalid file at once
        return case

    # Solving

    def solve(self, report: str = "summary") -> results:
        """Solves the problem: steady without a time section, transient with one, and an
        eigenvalue study with an eigenvalues section."""
        validated = self._valid()
        mesh = build_mesh(validated["mesh"])
        engine, _, variables = build_problem(validated, mesh)
        add_postprocessors(engine, validated, variables)
        _check_adaptivity_field(validated, variables)
        output = _engine_output(validated, variables, self._directory)
        lines = line_samplers(engine, validated, variables)
        write_lines = _line_writer(lines, output)
        if "eigenvalues" in validated:
            run = _solve_eigenvalue(validated, engine, variables, output, report)
        elif "time" not in validated:
            run = _solve_steady(validated, mesh, engine, variables, output, write_lines, report)
        else:
            run = _solve_transient(validated, mesh, engine, variables, output, write_lines, report)
        engine, variables, history = run.engine, run.variables, run.history
        # One table for the whole run: the first process writes it.
        if output is not None and len(history.names) > 1 and _core.mpi_rank() == 0:
            history.write_csv(output.path(".csv"))
        # The fields of the whole mesh, on every process of a distributed run.
        fields = {
            name: numpy.column_stack([engine.gathered_values(variable) for variable in names])
            for name, names in variables.items()
        }
        coordinates = engine.gathered_node_coordinates()[:, : engine.mesh.dimension]
        return results(
            run.times,
            fields,
            dict.fromkeys(fields, coordinates),
            history,
            lines={line.name: line.table() for line in lines},
            study=run.study,
        )


class _run(NamedTuple):
    """What a solve of the input leaves: the problem on the last mesh, its fields by name, the
    postprocessor table, the times of the solution and the eigenvalue study, or None."""

    engine: object
    variables: dict
    history: object
    times: list
    study: object = None


def _check_adaptivity_field(validated: dict, variables: dict) -> None:
    """Stops the run when mesh_adaptivity names a field that no physics solves for."""
    adaptivity = validated.get("mesh_adaptivity")
    if adaptivity is not None and adaptivity["field"] not in variables:
        raise _core.InputError(
            f"mesh_adaptivity.field: '{adaptivity['field']}' is not a field of the physics, "
            f"which solve for {', '.join(sorted(variables))}."
        )


def _solve_eigenvalue(validated: dict, engine, variables: dict, output, report: str) -> _run:
    """The eigenvalue study of the input, which writes each mode when the input has an output."""
    study = engine.solve_eigenvalue(
        num_modes=validated["eigenvalues"]["count"],
        near=validated["eigenvalues"]["near"],
        report=report,
    )
    if output is not None:
        _write_modes(engine, study, output)
    return _run(engine, variables, engine.postprocessor_table(), [], study)


def _solve_steady(validated: dict, mesh, engine, variables: dict, output, write_lines, report):
    """The steady solve of the input, on an adapted mesh when the input has mesh_adaptivity."""
    options = validated.get("solver", {})
    if validated.get("mesh_adaptivity") is not None:
        engine, variables = _adapt(validated, mesh, engine, variables, report, options)
        if output is not None:
            engine._write_fields(output)
    else:
        engine.solve(report=report, output=output, **options)
    write_lines(engine, 0.0)
    return _run(engine, variables, engine.postprocessor_table(), [0.0])


def _solve_transient(validated: dict, mesh, engine, variables: dict, output, write_lines, report):
    """The transient run of the input, on a mesh adapted in time when the input has
    mesh_adaptivity."""
    options = validated.get("solver", {})
    series = None if output is None else Series(output, [write_lines])
    if validated.get("mesh_adaptivity") is not None:
        return _adapt_in_time(validated, mesh, engine, variables, series, report, options)
    result = engine.solve_transient(
        **transient_options(validated["time"]), output=series, report=report, **options
    )
    times = [time for time, _ in result.step_history]
    return _run(engine, variables, engine.postprocessor_table(), times)


def _line_writer(lines: list, output):
    """The writer of the line postprocessors at an output time: it samples each line on the
    problem of that time, and the first process writes <output file>_<name>.csv with every row
    so far. It returns the paths of the files."""

    def write(engine, time: float) -> list[str]:
        paths = []
        for line in lines:
            line.sample(engine, time)
            path = output.path(f"_{line.name}.csv")
            if _core.mpi_rank() == 0:
                line.table().write_csv(path)
            paths.append(path)
        return paths

    return write


def _write_modes(engine, study, output: Output) -> None:
    """Writes each mode of an eigenvalue study, the real part of its fields, as
    <output file>_mode_<number>.vtu, and leaves the first mode in the solution."""
    for number, mode in reversed(list(enumerate(study.modes, start=1))):
        for variable, values in mode.items():
            engine.set_gathered_values(variable, numpy.real(values))
        engine.write_vtu(output.path(f"_mode_{number}.vtu"), fields=list(output.fields or ()))


def _engine_output(validated: dict, variables: dict, directory: str) -> Output | None:
    """The engine's Output group of the output section: the fields every interval-th accepted
    step of a transient run, or once in a steady run."""
    output = validated.get("output")
    if output is None:
        return None
    for index, name in enumerate(output["fields"]):
        if name not in variables:
            raise _core.InputError(
                f"output.fields[{index}]: the files hold the fields that the physics and the "
                f"models solve for, here {', '.join(variables)}."
            )
    path = os.path.join(directory, output["file"])
    return Output(
        directory=os.path.dirname(path),
        file_base=os.path.basename(path),
        step_interval=output["interval"] if "time" in validated else None,
        fields=[variable for name in output["fields"] for variable in variables[name]],
    )


def _marker(settings: dict):
    """The bulk marking of mesh_adaptivity, of its refine_fraction of the squared error."""
    return lambda indicators: mark_by_error_fraction(indicators, settings["refine_fraction"])


class _rebuilder:
    """Builds the problem of the input on each mesh of an adaptive loop, and holds the last
    problem built and its fields. The problem of the first mesh is built already."""

    def __init__(self, validated: dict, mesh, engine, variables: dict):
        self._validated = validated
        self._mesh = mesh
        self.engine = engine
        self.variables = variables

    def __call__(self, adapted):
        if adapted is not self._mesh:
            self.engine, _, self.variables = build_problem(self._validated, adapted)
            add_postprocessors(self.engine, self._validated, self.variables)
        return self.engine


def _adaptive_options(settings: dict) -> dict:
    """The keywords that both adaptive loops take from the mesh_adaptivity section."""
    return {
        "marker": _marker(settings),
        "coarsen_fraction": settings["coarsen_fraction"],
        "maximum_refinement_level": settings["maximum_refinement_level"],
    }


def _adapt(validated: dict, mesh, engine, variables: dict, report: str, options: dict):
    """The steady adaptive loop, solve_with_adaptive_refinement on the problems of the input, until
    no cell below the maximum refinement level is marked to refine. It returns the problem on
    the last mesh and its fields."""
    settings = validated["mesh_adaptivity"]
    build = _rebuilder(validated, mesh, engine, variables)
    solve_with_adaptive_refinement(
        build,
        mesh,
        variables[settings["field"]],
        # Refinement and coarsening may undo each other, so the cycles are bounded.
        num_cycles=4 * mesh.dimension * settings["maximum_refinement_level"] + 1,
        report=report,
        **_adaptive_options(settings),
        **options,
    )
    return build.engine, build.variables


def _adapt_in_time(validated, mesh, engine, variables, series, report, options) -> _run:
    """The transient adaptive loop, solve_transient_with_adaptive_refinement on the problems of
    the input, from the time 0 to the end time. It returns the problem on the last mesh, its
    fields, the postprocessor table of all the meshes and the times of the accepted steps."""
    settings = validated["mesh_adaptivity"]
    build = _rebuilder(validated, mesh, engine, variables)
    result = solve_transient_with_adaptive_refinement(
        build,
        mesh,
        variables[settings["field"]],
        interval=settings["interval"],
        report=report,
        output=series,
        **_adaptive_options(settings),
        **transient_options(validated["time"]),
        **options,
    )
    return _run(build.engine, build.variables, result.history, result.times)


def with_paths(tree: dict, change) -> dict:
    """A copy of tree with change applied to every file path: the mesh file and the CSV tables."""
    result = copy.deepcopy(tree)

    def visit(value) -> None:
        if isinstance(value, dict):
            for key, entry in value.items():
                if key == "file" and isinstance(entry, str) and value is not result.get("output"):
                    value[key] = change(entry)
                else:
                    visit(entry)
        elif isinstance(value, list):
            for entry in value:
                visit(entry)

    visit(result)
    return result
