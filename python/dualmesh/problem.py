"""The input of a simulation, built in Python as a YAML or TOML file builds it, and its solve.

The Python calls, a YAML file and a TOML file all build one input tree. A problem holds that
tree as the user wrote it, with units in text. `problem.solve` validates the tree, lowers it
onto the engine, solves it and returns the results.
"""

import copy
import os
from pathlib import Path

import numpy

from dualmesh import _core
from dualmesh.adaptivity import (
    _Hierarchy,
    _marks,
    mark_by_error_fraction,
    solve_with_adaptive_refinement,
)
from dualmesh.console import inner_study, report_level
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
from dualmesh.input.units import input_error
from dualmesh.output import Output, Series
from dualmesh.results import results
from dualmesh.tables import Table


def table(file: str | None = None, **columns) -> dict:
    """A table value of a property, linear between its rows.

    table("zircaloy_conductivity.csv") reads a CSV file whose header names each column with
    its unit. table(temperature=("K", [300, 600]), Young_modulus=("GPa", [96, 83])) gives the
    columns inline, each as (unit, values). The column named after the property holds its
    values, and the other column is its input: a field, the time or a coordinate.
    """
    if (file is None) == (not columns):
        raise input_error("table: give either a file or the columns, exactly one of the two.")
    if file is not None:
        return {"table": {"file": str(file)}}
    entries = {}
    for name, column in columns.items():
        if not isinstance(column, (tuple, list)) or len(column) != 2:
            raise input_error(f"table: the column {name} takes (unit, values), got {column!r}.")
        unit, values = column
        entries[name] = {"unit": unit, "values": list(values)}
    return {"table": entries}


class problem:
    """One simulation: a mesh, physics, properties, conditions and an analysis."""

    def __init__(self, mesh: dict, method: str | None = None):
        self._tree: dict = {}
        self._validated: dict | None = None  # The validated tree, kept until a section changes
        self._directory = os.getcwd()  # Relative mesh and table paths resolve against it
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
                raise input_error(
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
        self._valid()
        tree = self.to_tree()
        folder = Path(path).resolve().parent

        def relative(file: str) -> str:
            if os.path.isabs(file):
                return file
            return os.path.relpath(os.path.join(self._directory, file), folder)

        return with_paths(tree, relative)

    @staticmethod
    def from_file(path: str) -> "problem":
        """Reads a problem from a YAML or TOML file. Relative paths in it resolve against its folder."""
        case = problem.__new__(problem)
        case._tree = read_file(path)
        case._directory = str(Path(path).resolve().parent)
        case._validated = None
        case._valid()
        return case

    # Solving

    def solve(self, report: str = "summary") -> results:
        """Solves the problem: steady without a time section, transient with one, and an
        eigenvalue study with an eigenvalues section."""
        validated = self._valid()
        options = validated.get("solver", {})
        mesh = build_mesh(validated["mesh"])
        engine, _, variables = build_problem(validated, mesh)
        add_postprocessors(engine, validated, variables)
        adaptivity = validated.get("mesh_adaptivity")
        if adaptivity is not None and adaptivity["field"] not in variables:
            raise input_error(
                f"mesh_adaptivity.field: '{adaptivity['field']}' is not a field of the physics, "
                f"which solve for {', '.join(sorted(variables))}."
            )
        output = _engine_output(validated, variables, self._directory)
        lines = line_samplers(engine, validated, variables)
        write_lines = _line_writer(lines, output)
        study = None
        if "eigenvalues" in validated:
            times = []
            study = engine.solve_eigenvalue(
                num_modes=validated["eigenvalues"]["count"],
                near=validated["eigenvalues"]["near"],
                report=report,
            )
            if output is not None:
                _write_modes(engine, study, output)
            history = engine.postprocessor_table()
        elif "time" not in validated:
            times = [0.0]
            if adaptivity is not None:
                engine, variables = _adapt(validated, mesh, engine, variables, report, options)
                if output is not None:
                    engine._write_fields(output)
            else:
                engine.solve(report=report, output=output, **options)
            write_lines(engine, 0.0)
            history = engine.postprocessor_table()
        else:
            series = None if output is None else Series(output, [write_lines])
            if adaptivity is not None:
                engine, variables, history, times = _adapt_in_time(
                    validated, mesh, engine, variables, series, report, options
                )
            else:
                result = engine.solve_transient(
                    **transient_options(validated["time"]), output=series, report=report, **options
                )
                times = [time for time, _ in result.step_history]
                history = engine.postprocessor_table()
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
            times,
            fields,
            dict.fromkeys(fields, coordinates),
            history,
            lines={line.name: line.table() for line in lines},
            study=study,
        )


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
            raise input_error(
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


def _adapt(validated: dict, mesh, engine, variables: dict, report: str, options: dict):
    """The steady adaptive loop, solve_with_adaptive_refinement on the problems of the input, until
    no cell below the maximum refinement level is marked to refine.  It returns the problem on
    the last mesh and its fields."""
    settings = validated["mesh_adaptivity"]
    built = {"engine": engine, "variables": variables}

    def build(adapted):
        # The problem of the first mesh is built already.
        if adapted is not mesh:
            built["engine"], _, built["variables"] = build_problem(validated, adapted)
            add_postprocessors(built["engine"], validated, built["variables"])
        return built["engine"]

    max_level = settings["maximum_refinement_level"]
    solve_with_adaptive_refinement(
        build,
        mesh,
        variables[settings["field"]],
        # Refinement and coarsening may undo each other, so the cycles are bounded.
        num_cycles=4 * mesh.dimension * max_level + 1,
        marker=_marker(settings),
        report=report,
        coarsen_fraction=settings["coarsen_fraction"],
        maximum_refinement_level=max_level,
        **options,
    )
    return built["engine"], built["variables"]


def _adapt_in_time(validated, mesh, engine, variables, series, report, options):
    """The transient adaptive loop: run interval accepted steps, adapt the mesh to the indicators
    of the solution reached, and go on from that time on the new mesh, with the solution carried
    over and the step that the stepper chose.  It returns the problem on the last mesh, its
    fields, the postprocessor table of all the meshes and the times of the accepted steps."""
    settings = validated["mesh_adaptivity"]
    time_options = transient_options(validated["time"])
    end_time = time_options["end_time"]
    hierarchy = _Hierarchy(mesh)
    max_level = settings["maximum_refinement_level"]
    level = report_level(report, "problem.solve")
    if level != "none":
        print(
            f"dualmesh transient solve, the mesh adapted every {settings['interval']} steps",
            flush=True,
        )
    start_time, time_step = 0.0, time_options["time_step"]
    rows, times = [], []
    while True:
        with inner_study():
            result = engine.solve_transient(
                **{**time_options, "start_time": start_time, "time_step": time_step},
                max_steps=settings["interval"],
                output=series,
                **options,
            )
        times += [time for time, _ in result.step_history]
        # The rows of this mesh after the time where those of the last one ended.
        history = engine.postprocessor_table()
        rows += [row for row in history.rows if not rows or row[0] > rows[-1][0]]
        start_time = float(engine.time)
        if start_time >= end_time - 1e-12 * abs(end_time):
            break
        time_step = result.next_time_step
        indicators = numpy.sqrt(
            sum(engine.gathered_error_indicator(name) ** 2 for name in variables[settings["field"]])
        )
        refine, coarsen = _marks(
            hierarchy, indicators, _marker(settings), settings["coarsen_fraction"], max_level
        )
        state = {
            name: engine.gathered_values(name) for names in variables.values() for name in names
        }
        # The velocity of a second-order equation is a state of the step too: the new problem
        # goes on from the velocity reached, not from that of the initial condition.
        velocity = engine.gathered_velocities()
        if hierarchy.adapt(refine, coarsen, max_level):
            state = hierarchy.carry(engine, state)
            velocity = hierarchy.carry(engine, velocity)
            engine, _, variables = build_problem(validated, hierarchy.mesh())
            add_postprocessors(engine, validated, variables)
            engine.set_gathered_velocities(velocity)
            for name, values in state.items():
                engine.set_gathered_values(name, values)
            if level == "full":
                print(
                    f"  t = {start_time:.6g} s: {hierarchy.mesh().num_elements} cells", flush=True
                )
    if level != "none":
        print(f"  {len(times)} accepted steps, {hierarchy.mesh().num_elements} cells at the end")
    return engine, variables, Table(history.names, history.units, rows), times


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
